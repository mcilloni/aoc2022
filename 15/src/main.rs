use std::{
    error::Error,
    fs::{self},
    ops::{Add, Sub, AddAssign},
    str::FromStr, mem::replace, collections::HashSet,
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

use rayon::prelude::*;

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

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Point2(i64, i64);

impl Point2 {
    fn manhattan(self, other: Self) -> u64 {
        self.0.abs_diff(other.0) + self.1.abs_diff(other.1)
    }
}

impl Add for Point2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(
            self.0.checked_add(rhs.0).unwrap(),
            self.1.checked_add(rhs.1).unwrap(),
        )
    }
}

impl AddAssign<Direction> for Point2 {
    fn add_assign(&mut self, rhs: Direction) {
        *self = *self + rhs.versor()
    }
}

impl Add<Direction> for Point2 {
    type Output = Self;

    fn add(self, rhs: Direction) -> Self::Output {
        self + rhs.versor()
    }
}

impl Add<(i64, i64)> for Point2 {
    type Output = Self;

    fn add(self, rhs: (i64, i64)) -> Self::Output {
        self + Self::from(rhs)
    }
}

impl Sub for Point2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(
            self.0.checked_sub(rhs.0).unwrap(),
            self.1.checked_sub(rhs.1).unwrap(),
        )
    }
}

impl Sub<(i64, i64)> for Point2 {
    type Output = Self;

    fn sub(self, rhs: (i64, i64)) -> Self::Output {
        Self(
            self.0.checked_sub(rhs.0).unwrap(),
            self.1.checked_sub(rhs.1).unwrap(),
        )
    }
}

impl From<(i64, i64)> for Point2 {
    fn from((x, y): (i64, i64)) -> Self {
        Self(x, y)
    }
}

impl From<Point2> for (i64, i64) {
    fn from(Point2(x, y): Point2) -> Self {
        (x, y)
    }
}

fn coord<'a>(c: char) -> impl FnMut(&'a str) -> IResult<&'a str, i64> {
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
    range: u64, // memoized manhattan distance
}

impl Sensor {
    fn covers(&self, point: Point2) -> bool {
        point.manhattan(self.loc) <= self.range
    }

    fn edge(&self) -> Diamond {
        let range = self.range as i64;

        Diamond { 
            north: self.loc + (0, range),
            east: self.loc + (range, 0),
            south: self.loc - (0, range),
            west: self.loc - (range, 0),
        }
    }

    fn border(&self) -> Diamond {
        self.edge().expand(1)
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

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Direction {
    SouthEast,
    SouthWest,
    NorthWest,
    NorthEast,
}

impl Direction {
    const fn turn(self) -> Self {
        // turns 90 degrees 
        use Direction::*;

        match self {
            SouthEast => SouthWest,
            SouthWest => NorthWest,
            NorthWest => NorthEast,
            NorthEast => SouthEast,
        }
    }
    const fn versor(self) -> Point2 {
        use Direction::*;

        match self {
            SouthEast => Point2(1, -1),
            SouthWest => Point2(-1, -1),
            NorthWest => Point2(-1, 1),
            NorthEast => Point2(1, 1),
        }
    }
}

#[derive(Debug)]
struct Diamond {
    north: Point2,
    east: Point2,
    south: Point2,
    west: Point2,
}

impl Diamond {
    fn expand(&self, n: i64) -> Self {
        let &Self { north, east, south, west } = self;

        Self {
            north: north + (0, n),
            east: east + (n, 0),
            south: south - (0, n),
            west: west - (n, 0),
        }
    }

    fn perimeter(&self) -> PerimIter<'_> {
        PerimIter {
            diamond: self,
            cur: self.north,
            dir: Direction::SouthEast,
        }
    }
}

struct PerimIter<'a> {
    diamond: &'a Diamond,
    cur: Point2,
    dir: Direction,
}

impl <'a> PerimIter<'a> {
    fn ended(&self) -> bool {
        self.cur == self.diamond.north && self.dir != Direction::SouthEast
    }
}

impl <'a> Iterator for PerimIter<'a> {
    type Item = Point2;

    fn next(&mut self) -> Option<Self::Item> {
        use Direction::*;

        if self.ended() {
            None
        } else {
            let target = match self.dir {
                SouthEast => self.diamond.east,
                SouthWest => self.diamond.south,
                NorthWest => self.diamond.west,
                NorthEast => self.diamond.north,
            };

            if self.cur == target {
                self.dir = self.dir.turn();
            }
            
            let next = self.cur + self.dir;

            Some(replace(&mut self.cur, next))
        }

    }
}

const LINE_SLOPE: i64 = 4000000;

/// AoC problem for Dec 15 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,

    /// max boundary for search space cutting
    boundary: i64,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file, boundary } = Args::parse();

    let slist = sensors(&fs::read_to_string(file)?)
        .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
        .1;

    let vr = 0..=boundary;
    let is_valid = move |&Point2(x, y) : &Point2| vr.contains(&x) && vr.contains(&y);
        
    // reasoning (thanks Reddit):
    // the spot we're looking for is *the only one* not covered by any diamond-shaped view area
    // this means the point we're searching must reside on one of the edges of the sensors.
    // While this is still O(a lot), they can't be much worse than searching for the whole search space, right?
    
    let mut set = HashSet::new();
    for s in &slist {
        set.extend(s.border().perimeter().filter(&is_valid));
    }

    // do some rayon magic here...
    set.par_iter().find_any(|&&tp| {
        for sensor in &slist {
            if sensor.covers(tp) || tp == sensor.closest {
                return false;
            }
        }

        let Point2(x, y) = tp;
        println!("found at {tp:?}, f = {}", x * LINE_SLOPE + y);

        return true;
    });

    Ok(())
}
