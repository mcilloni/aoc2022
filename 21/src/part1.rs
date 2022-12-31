use std::{collections::HashMap, error::Error, fs, str::FromStr};

use clap::Parser;
use nom::{
    branch::alt,
    bytes::complete::{take_while},
    character::complete::{alpha1, anychar, char, multispace0},
    combinator::{all_consuming, map, map_res, opt, recognize},
    error::{Error as NomError, ParseError},
    multi::{fold_many1},
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
    fn eval(self, lhs: isize, rhs: isize) -> isize {
        use Op::*;

        match self {
            Add => lhs.checked_add(rhs),
            Sub => lhs.checked_sub(rhs),
            Mul => lhs.checked_mul(rhs),
            Div => lhs.checked_div(rhs),
        }.expect("no integer overflows")
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
#[repr(u8)]
enum Item<'a> {
    Num(isize),
    Pending { op: Op, lhs: &'a str, rhs: &'a str },
}

impl Item<'_> {
    const fn is_num(self) -> bool {
        matches!(self, Self::Num(_))
    }

    const fn is_pending(self) -> bool {
        matches!(self, Self::Pending {..})
    }

    const fn num(self) -> Option<isize> {
        match self {
            Self::Num(n) => Some(n),
            _ => None,
        }
    }
}

type Monkey<'a> = (&'a str, Item<'a>);

fn item(input: &str) -> IResult<&str, Item<'_>> {
    alt((
        map(ws(signed), Item::Num),
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

fn eval<'a>(key: &'a str, entries: &'a mut HashMap<&'a str, Item>) -> Option<isize> {
    use Item::*;

    let Some(&wanted) = entries.get(key) else {
        return None;
    };

    let num = if let Num(num) = wanted {
        num
    } else {
        let mut stack = vec![key];
        
        while let Some(pending) = stack.pop() {
            if let &Pending { op, lhs, rhs } = &entries[pending] {
                let (&l, &r) = (&entries[lhs], &entries[rhs]);
                if let (Num(lhs), Num(rhs)) = (l, r) {
                    entries.insert(pending, Num(op.eval(lhs, rhs)));
                } else {
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
        
        let Num(val) = entries[key] else {
            unreachable!();
        };
        
        val
    };

    Some(num)
}

const WANTED : &str = "root";

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let input = fs::read_to_string(&file)?;

    let mut mks = monkeys(&input)
        .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
        .1; 

    println!("{WANTED} == {}", eval(WANTED, &mut mks).ok_or_else(|| format!("no '{WANTED}' in '{file}'"))?);

    Ok(())
}
