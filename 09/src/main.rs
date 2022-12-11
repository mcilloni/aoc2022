use std::{
    collections::HashSet,
    error::Error,
    fs::File,
    io::{self, BufRead, BufReader},
    iter::repeat,
    ops::{Add, AddAssign, Sub},
    str::FromStr,
};

use clap::Parser;
use nom::{
    branch::alt,
    bytes::complete::take_while,
    character::complete::{char, space1},
    combinator::{map, map_res, value},
    error::Error as NomError,
    sequence::separated_pair,
    IResult,
};
use num::Unsigned;

fn is_int_digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn unsigned<N: Unsigned + FromStr>(input: &str) -> IResult<&str, N> {
    map_res(take_while(is_int_digit), str::parse)(input)
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    fn from_x(x: i16) -> Option<Self> {
        match x.signum() {
            -1 => Some(Self::Left),
            1 => Some(Self::Right),
            0 => None,
            _ => unreachable!(),
        }
    }
    fn from_y(y: i16) -> Option<Self> {
        match y.signum() {
            -1 => Some(Self::Down),
            1 => Some(Self::Up),
            0 => None,
            _ => unreachable!(),
        }
    }

    fn versor(self) -> Point2 {
        use Direction::*;

        match self {
            Up => Point2(0, 1),
            Down => Point2(0, -1),
            Left => Point2(-1, 0),
            Right => Point2(1, 0),
        }
    }
}

fn direction(input: &str) -> IResult<&str, Direction> {
    use Direction::*;

    alt((
        value(Up, char('U')),
        value(Down, char('D')),
        value(Left, char('L')),
        value(Right, char('R')),
    ))(input)
}

#[derive(Clone, Copy, Debug)]
struct Command {
    direction: Direction,
    steps: u8,
}

impl Command {
    fn directions(self) -> impl Iterator<Item = Direction> {
        repeat(self.direction).take(self.steps as usize)
    }
}

impl From<(Direction, u8)> for Command {
    fn from((direction, steps): (Direction, u8)) -> Self {
        Self { direction, steps }
    }
}

fn command(input: &str) -> IResult<&str, Command> {
    map(separated_pair(direction, space1, unsigned), Command::from)(input)
}

struct Commands<I> {
    source: I,
}

impl<I> Iterator for Commands<I>
where
    I: Iterator<Item = Result<String, io::Error>>,
{
    type Item = Result<Command, Box<dyn Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.next().map(|line_res| {
            line_res.map_err(Into::into).and_then(|line| {
                command(line.trim())
                    .map(|(_, cmd)| cmd)
                    .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)).into())
            })
        })
    }
}

fn commands<I>(source: I) -> Commands<I>
where
    I: Iterator<Item = Result<String, io::Error>>,
{
    Commands { source }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct Point2(i16, i16);

impl Add for Point2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(
            self.0.checked_add(rhs.0).unwrap(),
            self.1.checked_add(rhs.1).unwrap(),
        )
    }
}

impl AddAssign for Point2 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl From<Direction> for Point2 {
    fn from(dir: Direction) -> Self {
        dir.versor()
    }
}

impl From<(i16, i16)> for Point2 {
    fn from((x, y): (i16, i16)) -> Self {
        Self(x, y)
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

fn fixup_pair(lead: Point2, follow: Point2) -> Point2 {
    fn compute_step(mut point: Point2, x: i16, y: i16) -> Point2 {
        if let Some(dir) = Direction::from_x(x) {
            point += dir.versor();
        }

        if let Some(dir) = Direction::from_y(y) {
            point += dir.versor();
        }

        point
    }

    let Point2(diff_x, diff_y) = lead - follow;

    assert!(diff_x.abs() < 3 && diff_y.abs() < 3); // if this is not true, this code is very broken

    match (diff_x, diff_y) {
        (x, y) if x.abs() < 2 && y.abs() < 2 => follow, // do nothing, tail is fine
        (x, y) => compute_step(follow, x, y),           // fixup
    }
}

#[derive(Debug, Default)]
struct Rope {
    knots: [Point2; 10],
}

impl Rope {
    fn fixup(&mut self) {
        let mut prev = None;

        for point in self.knots.iter_mut() {
            if let Some(prev) = prev {
                *point = fixup_pair(prev, *point);
            }

            prev = Some(*point);
        }
    }

    fn move_head(&mut self, dir: Direction) -> Point2 {
        let [head, ..] = &mut self.knots;

        *head = *head + dir.versor();

        self.fixup();

        let [.., tail] = &self.knots;

        *tail
    }
}

/// AoC problem for Dec 09 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let commands = commands(BufReader::new(File::open(file)?).lines());

    let mut tail_pos = HashSet::new();
    let mut rope = Rope::default();

    for maybe_cmd in commands {
        for dir in maybe_cmd?.directions() {
            let tail = rope.move_head(dir);

            tail_pos.insert(tail);
        }
    }

    println!("tail visited {} positions", tail_pos.len());

    Ok(())
}
