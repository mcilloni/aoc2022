use std::{
    cmp::Ordering,
    error::Error,
    fs::{self},
    str::FromStr,
};

use clap::Parser;
use itertools::{EitherOrBoth, Itertools};
use nom::{
    branch::alt,
    bytes::complete::take_while,
    character::complete::{char, multispace0, space0},
    combinator::{all_consuming, map, map_res},
    error::{Error as NomError, ParseError},
    multi::separated_list0,
    sequence::{delimited, separated_pair},
    IResult,
};
use num::Unsigned;

const fn is_int_digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn unsigned<N: Unsigned + FromStr>(input: &str) -> IResult<&str, N> {
    map_res(take_while(is_int_digit), str::parse)(input)
}

fn ws<'a, F: 'a, O, E: ParseError<&'a str>>(
    inner: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, O, E>
where
    F: Fn(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(space0, inner, space0)
}

#[derive(Debug)]
enum ListItem {
    Num(usize),
    Seq(Box<List>),
}

fn list_item(input: &str) -> IResult<&str, ListItem> {
    alt((
        map(unsigned, ListItem::Num),
        map(list, |l| ListItem::Seq(Box::new(l))),
    ))(input)
}

#[derive(Debug, Default)]
struct List(Vec<ListItem>);

impl List {
    fn single(num: usize) -> Self {
        Self(vec![ListItem::Num(num)])
    }

    fn iter(&self) -> impl Iterator<Item = &ListItem> + '_ {
        self.0.iter()
    }
}

fn list(input: &str) -> IResult<&str, List> {
    map(
        delimited(
            ws(char('[')),
            separated_list0(ws(char(',')), list_item),
            ws(char(']')),
        ),
        List,
    )(input)
}

fn list_pair(input: &str) -> IResult<&str, (List, List)> {
    separated_pair(ws(list), multispace0, ws(list))(input)
}

fn valid(left: &List, right: &List) -> Option<bool> {
    for eob in left.iter().zip_longest(right.iter()) {
        use EitherOrBoth::*;
        use ListItem::*;
        use Ordering::*;

        match eob {
            Both(Num(n1), Num(n2)) => match n1.cmp(n2) {
                Less => return Some(true),
                Greater => return Some(false),
                Equal => {} // continue
            },
            Both(Seq(l1), Seq(l2)) => {
                if let v @ Some(_) = valid(l1, l2) {
                    return v;
                }
            }
            Both(Seq(l), Num(n)) => {
                if let v @ Some(_) = valid(l, &List::single(*n)) {
                    return v;
                }
            }
            Both(Num(n), Seq(l)) => {
                if let v @ Some(_) = valid(&List::single(*n), l) {
                    return v;
                }
            }
            Left(_) => return Some(false),
            Right(_) => return Some(true),
        }
    }

    None
}

fn list_pairs(input: &str) -> IResult<&str, Vec<(List, List)>> {
    all_consuming(delimited(
        multispace0,
        separated_list0(multispace0, list_pair),
        multispace0,
    ))(input)
}

/// AoC problem for Dec 13 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let sum: usize = list_pairs(&fs::read_to_string(file)?)
        .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
        .1
        .into_iter()
        .map(|(ref l1, ref l2)| valid(l1, l2).unwrap_or_default())
        .enumerate()
        .filter_map(|(i, v)| v.then_some(i + 1))
        .sum();

    println!("sum = {sum}");

    Ok(())
}
