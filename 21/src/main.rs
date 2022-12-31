use std::{collections::HashMap, error::Error, fs, str::FromStr};

use clap::Parser;
use nom::{
    branch::alt,
    bytes::complete::take_while,
    character::complete::{alpha1, anychar, char, multispace0},
    combinator::{all_consuming, map, map_res, opt, recognize},
    error::{Error as NomError, ParseError},
    multi::fold_many1,
    sequence::{delimited, separated_pair, tuple},
    IResult,
};
use num::Signed;

const fn is_int_digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn signed<N: Signed + FromStr>(input: &str) -> IResult<&str, N> {
    map_res(
        recognize(tuple((
            opt(alt((char('+'), char('-')))),
            take_while(is_int_digit),
        ))),
        str::parse,
    )(input)
}

fn ws<'a, F: 'a, O, E: ParseError<&'a str>>(
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(multispace0, inner, multispace0)
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

impl Op {
    fn eval(self, lhs: f64, rhs: f64) -> f64 {
        use Op::*;

        match self {
            Add => lhs + rhs,
            Sub => lhs - rhs,
            Mul => lhs * rhs,
            Div => lhs / rhs,
        }
    }
}

impl TryFrom<char> for Op {
    type Error = String;

    fn try_from(value: char) -> Result<Self, Self::Error> {
        use Op::*;

        match value {
            '+' => Ok(Add),
            '-' => Ok(Sub),
            '*' => Ok(Mul),
            '/' => Ok(Div),
            _ => Err(format!("'{value}' is not a valid operation")),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Number {
    Known(f64),
    Unknown,
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
enum Item<'a> {
    Num(Number),
    Pending { op: Op, lhs: &'a str, rhs: &'a str },
}

impl Item<'_> {
    const fn is_pending(self) -> bool {
        matches!(self, Self::Pending { .. })
    }
}

type Monkey<'a> = (&'a str, Item<'a>);

fn item(input: &str) -> IResult<&str, Item<'_>> {
    alt((
        map(ws(signed), |n| Item::Num(Number::Known(n))),
        map_res(
            tuple((alpha1, ws(anychar), alpha1)),
            |(lhs, opc, rhs)| -> Result<Item, String> {
                Ok(Item::Pending {
                    op: opc.try_into()?,
                    lhs,
                    rhs,
                })
            },
        ),
    ))(input)
}

fn monkey(input: &str) -> IResult<&str, Monkey> {
    separated_pair(alpha1, ws(char(':')), item)(input)
}

fn monkeys(input: &str) -> IResult<&str, HashMap<&str, Item>> {
    all_consuming(fold_many1(
        delimited(multispace0, monkey, multispace0),
        HashMap::new,
        |mut h, (n, bp)| {
            h.insert(n, bp);
            h
        },
    ))(input)
}

/// AoC problem for Dec 21 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

fn eval<'a>(key: &'a str, entries: &mut HashMap<&'a str, Item<'a>>) -> Number {
    use Item::*;
    use Number::*;

    let Some(&wanted) = entries.get(key) else {
        return Unknown;
    };

    let num = if let Num(num) = wanted {
        num
    } else {
        let mut stack = vec![key];

        while let Some(pending) = stack.pop() {
            if let &Pending { op, lhs, rhs } = &entries[pending] {
                let (&l, &r) = (&entries[lhs], &entries[rhs]);

                match (l, r) {
                    (Num(Known(lhs)), Num(Known(rhs))) => {
                        entries.insert(pending, Num(Known(op.eval(lhs, rhs))));
                    }
                    (Num(Unknown), _) | (_, Num(Unknown)) => {
                        return Unknown; // abort trees that contain unknowns
                    }
                    _ => {
                        stack.push(pending);

                        if l.is_pending() {
                            stack.push(lhs);
                        }

                        if r.is_pending() {
                            stack.push(rhs);
                        }
                    }
                }
            }
        }

        let Num(val) = entries[key] else {
            unreachable!();
        };

        val
    };

    num
}

const WANTED: &str = "root";
const UKN: &str = "humn";

fn solve_tree<'a>(mut tree: HashMap<&'a str, Item<'a>>) -> f64 {
    use Item::*;
    use Number::*;
    use Op::*;

    // special case: the root is the == operator
    let &Pending { lhs, rhs, .. } = &tree[WANTED] else {
        panic!("malformed tree");
    };

    // attempt to solve the tree for both
    let sol_left = eval(lhs, &mut tree);
    let sol_right = eval(rhs, &mut tree);

    let (mut bad, mut good) = match (sol_left, sol_right) {
        (Known(ls), Unknown) => (rhs, ls),
        (Unknown, Known(rs)) => (lhs, rs),
        _ => panic!("unsupported"),
    };

    loop {
        match tree[bad] {
            Pending { op, lhs, rhs } => {
                // attempt to solve the tree for both
                let sol_left = eval(lhs, &mut tree);
                let sol_right = eval(rhs, &mut tree);

                (good, bad) = match (op, sol_left, sol_right) {
                    // k + $bad = $good -> $bad = $good - k
                    (Add, Known(k), Unknown) => (good - k, rhs),
                    (Add, Unknown, Known(k)) => (good - k, lhs),

                    // k - $bad = $good => $bad = k - $good
                    (Sub, Known(k), Unknown) => (k - good, rhs),

                    // $bad - k = $good => $bad = $good + k
                    (Sub, Unknown, Known(k)) => (k + good, lhs),

                    // k * $bad = $good => $bad = $good / k
                    (Mul, Known(k), Unknown) => (good / k, rhs),
                    (Mul, Unknown, Known(k)) => (good / k, lhs),

                    // k / $bad = $good => $bad = k / $good
                    (Div, Known(k), Unknown) => (k / good, rhs),

                    // $bad / k = $good => $bad = k * $good
                    (Div, Unknown, Known(k)) => (k * good, lhs),

                    other => panic!("got unsupported combination {other:?}"),
                }
            }
            Num(Unknown) => break good,
            Num(Known(_)) => unreachable!(),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let input = fs::read_to_string(file)?;

    let mut mks = monkeys(&input)
        .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
        .1;

    mks.insert(UKN, Item::Num(Number::Unknown));

    println!("{UKN} == {}", solve_tree(mks));

    Ok(())
}
