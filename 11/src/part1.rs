use std::{
    cmp::{max, min},
    collections::BTreeMap,
    error::Error,
    fs::{self},
    mem,
    str::FromStr,
};

use clap::Parser;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while},
    character::complete::{char, multispace0, space0},
    combinator::{complete, map, map_res, value},
    error::Error as NomError,
    multi::{separated_list0, separated_list1},
    sequence::{delimited, preceded, separated_pair, tuple},
    IResult,
};
use num::Unsigned;

const fn is_int_digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn unsigned<N: Unsigned + FromStr>(input: &str) -> IResult<&str, N> {
    map_res(take_while(is_int_digit), str::parse)(input)
}

fn items(input: &str) -> IResult<&str, Vec<usize>> {
    preceded(
        tuple((
            tag("Starting"),
            space0,
            tag("items"),
            space0,
            char(':'),
            space0,
        )),
        separated_list1(tuple((char(','), space0)), unsigned),
    )(input)
}

#[derive(Clone, Copy, Debug)]
enum OpVal {
    Old,
    Val(usize),
}

impl OpVal {
    fn value_or(self, old: usize) -> usize {
        use OpVal::*;

        match self {
            Old => old,
            Val(val) => val,
        }
    }
}

fn opval(input: &str) -> IResult<&str, OpVal> {
    use OpVal::*;

    alt((value(Old, tag("old")), map(unsigned, OpVal::Val)))(input)
}

#[derive(Clone, Copy, Debug)]
enum Operation {
    Add(OpVal),
    Mul(OpVal),
}

impl Operation {
    fn apply(self, worry: usize) -> usize {
        use Operation::*;

        match self {
            Add(n) => worry.checked_add(n.value_or(worry)),
            Mul(n) => worry.checked_mul(n.value_or(worry)),
        }
        .unwrap()
    }
}

fn operation(input: &str) -> IResult<&str, Operation> {
    preceded(
        tuple((
            tag("Operation"),
            space0,
            char(':'),
            space0,
            tag("new"),
            space0,
            char('='),
            space0,
            tag("old"),
            space0,
        )),
        alt((
            map(separated_pair(char('+'), space0, opval), |(_, n)| {
                Operation::Add(n)
            }),
            map(separated_pair(char('*'), space0, opval), |(_, n)| {
                Operation::Mul(n)
            }),
        )),
    )(input)
}

#[derive(Debug)]
struct Test {
    rem: usize,

    zero: usize,
    nzero: usize,
}

impl Test {
    const fn dispatch(&self, worry: usize) -> usize {
        if worry % self.rem == 0 {
            self.zero
        } else {
            self.nzero
        }
    }
}

fn test(input: &str) -> IResult<&str, Test> {
    fn rem(input: &str) -> IResult<&str, usize> {
        preceded(
            tuple((
                tag("Test"),
                space0,
                char(':'),
                space0,
                tag("divisible"),
                space0,
                tag("by"),
                space0,
            )),
            unsigned,
        )(input)
    }

    fn throw_to_monkey<'a>(
        exp_cond: &'static str,
    ) -> impl FnMut(&'a str) -> IResult<&'a str, usize> + 'a {
        // can almost certainly be optimized, don't care
        preceded(
            tuple((
                tag("If"),
                space0,
                tag(exp_cond),
                space0,
                char(':'),
                space0,
                tag("throw"),
                space0,
                tag("to"),
                space0,
                tag("monkey"),
                space0,
            )),
            unsigned,
        )
    }

    map(
        tuple((
            rem,
            multispace0,
            throw_to_monkey("true"),
            multispace0,
            throw_to_monkey("false"),
        )),
        |(rem, _, zero, _, nzero)| Test { rem, zero, nzero },
    )(input)
}

struct MonkeyRes {
    item: usize,
    to: usize,
}

#[derive(Debug)]
struct Monkey {
    items: Vec<usize>,
    op: Operation,
    test: Test,

    counter: usize,
}

const RELAX_FACTOR: usize = 3;

impl Monkey {
    fn counter(&self) -> usize {
        self.counter
    }

    fn do_keepaway(&mut self) -> impl Iterator<Item = MonkeyRes> + '_ {
        self.counter = self.counter.checked_add(self.items.len()).unwrap();

        mem::take(&mut self.items).into_iter().map(|mut worry| {
            worry = self.op.apply(worry);
            worry /= RELAX_FACTOR;

            MonkeyRes {
                item: worry,
                to: self.test.dispatch(worry),
            }
        })
    }

    fn push_item(&mut self, item: usize) {
        self.items.push(item);
    }
}

fn monkey(input: &str) -> IResult<&str, (usize, Monkey)> {
    fn monkey_n(input: &str) -> IResult<&str, usize> {
        delimited(
            tuple((tag("Monkey"), space0)),
            unsigned,
            tuple((space0, char(':'))),
        )(input)
    }

    map(
        tuple((
            monkey_n,
            multispace0,
            items,
            multispace0,
            operation,
            multispace0,
            test,
        )),
        |(n, _, items, _, op, _, test)| {
            (
                n,
                Monkey {
                    items,
                    op,
                    test,
                    counter: 0,
                },
            )
        },
    )(input)
}

// nobody said monkeys have to be contiguous
fn monkeys(input: &str) -> IResult<&str, BTreeMap<usize, Monkey>> {
    map(
        complete(delimited(
            multispace0,
            separated_list0(multispace0, monkey),
            multispace0,
        )),
        |v| v.into_iter().collect(),
    )(input)
}

/// AoC problem for Dec 11 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let mut monkeys = {
        // even my ESP32s have enough RAM for this
        monkeys(&fs::read_to_string(file)?)
            .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
            .1
    };

    // pointless if we assume the input cannot have holes and is sorted, and thus we can use Vec<_>
    // necessary otherwise - we must mutate the map, and we can't hold a mutable iterator to `monkeys`
    // while editing random keys at the same time
    let monkey_ids: Vec<_> = monkeys.keys().cloned().collect();

    for round in 1..=20 {
        for mn in &monkey_ids {
            let thrown_items: Vec<_> = monkeys.get_mut(&mn).unwrap().do_keepaway().collect();

            for MonkeyRes { item, to } in thrown_items {
                monkeys
                    .get_mut(&to)
                    .ok_or(format!("invalid monkey #{to}"))?
                    .push_item(item);
            }
        }

        println!("After round {round}, the monkeys are holding items with these worry levels:");
        for (n, monkey) in &monkeys {
            println!("Monkey {n}: {:?}", &monkey.items as &[_]);
        }

        println!()
    }

    for (n, monkey) in &monkeys {
        println!("Monkey {n} inspected items {} times", monkey.counter());
    }

    let (first, second) = monkeys.values().fold((0, 0), |(first, second), monkey| {
        let cur = monkey.counter();

        let new_first = max(first, cur);
        let new_second = max(min(first, cur), second);

        (new_first, new_second)
    });

    println!("monkey business = {}", first.checked_mul(second).unwrap());

    Ok(())
}
