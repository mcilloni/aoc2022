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
    sequence::delimited,
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

#[derive(Clone, Debug, Eq, PartialEq, PartialOrd, Ord)]
enum ListItem {
    Num(usize),
    Seq(Box<List>),
}

impl From<usize> for ListItem {
    fn from(n: usize) -> Self {
        Self::Num(n)
    }
}

impl<T: Into<ListItem>, const N: usize> From<[T; N]> for ListItem {
    fn from(arr: [T; N]) -> Self {
        Self::Seq(Box::new(List(arr.into_iter().map(Into::into).collect())))
    }
}

fn list_item(input: &str) -> IResult<&str, ListItem> {
    alt((
        map(unsigned, ListItem::Num),
        map(list, |l| ListItem::Seq(Box::new(l))),
    ))(input)
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct List(Vec<ListItem>);

impl List {
    fn single<T: Into<ListItem>>(v: T) -> Self {
        Self(vec![v.into()])
    }

    fn iter(&self) -> impl Iterator<Item = &ListItem> + '_ {
        self.0.iter()
    }
}

impl<I: IntoIterator<Item = T>, T: Into<ListItem>> From<I> for List {
    fn from(el: I) -> Self {
        el.into_iter().map(Into::into).collect()
    }
}

impl<I: Into<ListItem>> FromIterator<I> for List {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        Self(iter.into_iter().map(Into::into).collect())
    }
}

impl Ord for List {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other)
            .expect("all lists should be sortable")
    }
}

impl PartialOrd for List {
    fn partial_cmp(&self, right: &List) -> Option<Ordering> {
        for eob in self.iter().zip_longest(right.iter()) {
            use EitherOrBoth::*;
            use ListItem::*;
            use Ordering::*;

            match eob {
                Both(Num(n1), Num(n2)) => match n1.cmp(n2) {
                    Equal => {} // continue
                    v => return Some(v),
                },
                Both(Seq(l1), Seq(l2)) => {
                    if let v @ Some(_) = l1.partial_cmp(l2) {
                        return v;
                    }
                }
                Both(Seq(l), Num(n)) => {
                    if let v @ Some(_) = (l as &List).partial_cmp(&List::single(*n)) {
                        return v;
                    }
                }
                Both(Num(n), Seq(l)) => {
                    if let v @ Some(_) = List::single(*n).partial_cmp(l) {
                        return v;
                    }
                }
                Left(_) => return Some(Greater),
                Right(_) => return Some(Less),
            }
        }

        None
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

fn lists(input: &str) -> IResult<&str, Vec<List>> {
    all_consuming(delimited(
        multispace0,
        separated_list0(multispace0, list),
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
    let d1: List = [[2usize]].into();
    let d2: List = [[6usize]].into();

    let Args { file } = Args::parse();

    let mut ls = lists(&fs::read_to_string(file)?)
        .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
        .1;

    ls.extend([d1.clone(), d2.clone()]);

    ls.sort();

    let p: usize = ls
        .into_iter()
        .enumerate()
        .filter_map(|(i, l)| (l == d1 || l == d2).then_some(i + 1))
        .product();

    println!("dk = {p}");

    Ok(())
}
