use std::{
    cmp::{max, min},
    error::Error,
    fs::{self},
    str::FromStr,
};

use clap::Parser;

use nom::{
    branch::alt,
    bytes::complete::{tag, take_while},
    character::complete::{char, multispace0, space0},
    combinator::{all_consuming, map, map_res, opt, recognize},
    error::{Error as NomError, ParseError},
    multi::separated_list0,
    sequence::{delimited, preceded, separated_pair, tuple},
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
    F: Fn(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(space0, inner, space0)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Point2(isize, isize);

impl Point2 {
    fn manhattan(self, other: Self) -> usize {
        self.0.abs_diff(other.0) + self.1.abs_diff(other.1)
    }
}

impl From<(isize, isize)> for Point2 {
    fn from((x, y): (isize, isize)) -> Self {
        Self(x, y)
    }
}

impl From<Point2> for (isize, isize) {
    fn from(Point2(x, y): Point2) -> Self {
        (x, y)
    }
}

fn coord<'a>(c: char) -> impl FnMut(&'a str) -> IResult<&'a str, isize> {
    map(separated_pair(char(c), ws(char('=')), signed), |(_, v)| v)
}

fn point2(input: &str) -> IResult<&str, Point2> {
    map(
        separated_pair(coord('x'), ws(char(',')), coord('y')),
        Into::into,
    )(input)
}

fn beacon_at(input: &str) -> IResult<&str, Point2> {
    preceded(ws(tag("closest beacon is at")), point2)(input)
}

fn sensor_at(input: &str) -> IResult<&str, Point2> {
    preceded(ws(tag("Sensor at")), point2)(input)
}

#[derive(Debug)]
struct Sensor {
    loc: Point2,
    closest: Point2,
    range: usize, // memoized manhattan distance
}

impl Sensor {
    fn covers(&self, point: Point2) -> bool {
        point.manhattan(self.loc) <= self.range
    }
}

impl From<(Point2, Point2)> for Sensor {
    fn from((loc, closest): (Point2, Point2)) -> Self {
        Self {
            loc,
            closest,
            range: loc.manhattan(closest),
        }
    }
}

fn sensor(input: &str) -> IResult<&str, Sensor> {
    map(
        separated_pair(sensor_at, ws(char(':')), beacon_at),
        Sensor::from,
    )(input)
}

fn sensors(input: &str) -> IResult<&str, Vec<Sensor>> {
    all_consuming(delimited(
        multispace0,
        separated_list0(multispace0, sensor),
        multispace0,
    ))(input)
}

/// AoC problem for Dec 15 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,

    /// y line to test
    y: isize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file, y } = Args::parse();

    let slist = sensors(&fs::read_to_string(file)?)
        .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
        .1;

    let (min_x, max_x) = slist
        .iter()
        .fold(
            None,
            |acc,
             Sensor {
                 loc: Point2(x_s, _),
                 range,
                 ..
             }| {
                let edge_low = x_s.checked_sub_unsigned(*range).unwrap();
                let edge_high = x_s.checked_add_unsigned(*range).unwrap();

                Some(match acc {
                    Some((best_low, best_high)) => {
                        (min(best_low, edge_low), max(best_high, edge_high))
                    }
                    None => (edge_low, edge_high),
                })
            },
        )
        .unwrap();

    println!("x_boundaries = ({min_x}, {max_x})");

    let mut cnt = 0usize;
    'test: for x in min_x..=max_x {
        let tp = Point2(x, y);

        for sensor in &slist {
            if sensor.covers(tp) && tp != sensor.closest {
                cnt = cnt.checked_add(1).unwrap();

                continue 'test;
            }
        }
    }

    println!("y={y} has {cnt} points covered by a sensor");

    Ok(())
}
