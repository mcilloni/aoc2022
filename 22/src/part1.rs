use std::{
    cmp::max,
    collections::HashMap,
    error::Error,
    fmt::Display,
    fs,
    num::TryFromIntError,
    ops::{Add, AddAssign, Index, IndexMut, Sub},
    str::FromStr,
};

use clap::Parser;
use nom::{
    branch::alt,
    bytes::complete::take_while,
    character::complete::{char, line_ending, multispace0},
    combinator::{all_consuming, map, map_res, value},
    error::Error as NomError,
    multi::{fold_many0, many0, many1},
    sequence::{separated_pair, terminated},
    IResult,
};
use num::{FromPrimitive, Unsigned};
use num_derive::FromPrimitive;
use strum::EnumCount;

const fn is_int_digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn unsigned<N: Unsigned + FromStr>(input: &str) -> IResult<&str, N> {
    map_res(take_while(is_int_digit), str::parse)(input)
}

#[derive(Clone, Copy, Debug)]
#[repr(i8)]
enum Rotate {
    Left = -1,
    Right = 1,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, EnumCount, FromPrimitive)]
enum Direction {
    East = 0,
    South = 1,
    West = 2,
    North = 3,
}

impl Direction {
    const fn dim(self) -> Dimension {
        use Dimension::*;
        use Direction::*;

        match self {
            North | South => X,
            East | West => Y,
        }
    }

    const fn opposite(self) -> Self {
        use Direction::*;

        match self {
            North => South,
            East => West,
            South => North,
            West => East,
        }
    }

    fn turn(self, rot: Rotate) -> Self {
        Self::from_isize((self as isize + rot as isize).rem_euclid(Self::COUNT as isize)).unwrap()
    }

    const fn versor(self) -> Point2 {
        use Direction::*;

        match self {
            North => Point2(0, -1),
            East => Point2(1, 0),
            South => Point2(0, 1),
            West => Point2(-1, 0),
        }
    }
}

impl Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Direction::*;

        match self {
            East => '>',
            South => 'v',
            West => '<',
            North => '^',
        }
        .fmt(f)
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
enum RowDirection {
    Start,
    End,
}

impl From<Direction> for RowDirection {
    fn from(value: Direction) -> Self {
        use Direction::*;
        use RowDirection::*;

        match value {
            North | West => Start,
            South | East => End,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
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

impl Add<(i16, i16)> for Point2 {
    type Output = Self;

    fn add(self, rhs: (i16, i16)) -> Self::Output {
        self + Self::from(rhs)
    }
}

impl Index<Dimension> for Point2 {
    type Output = i16;

    fn index(&self, index: Dimension) -> &Self::Output {
        use Dimension::*;

        match index {
            X => &self.0,
            Y => &self.1,
        }
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

impl Sub<Direction> for Point2 {
    type Output = Self;

    fn sub(self, rhs: Direction) -> Self::Output {
        self - rhs.versor()
    }
}

impl Sub<(i16, i16)> for Point2 {
    type Output = Self;

    fn sub(self, rhs: (i16, i16)) -> Self::Output {
        Self(
            self.0.checked_sub(rhs.0).unwrap(),
            self.1.checked_sub(rhs.1).unwrap(),
        )
    }
}

impl From<(i16, i16)> for Point2 {
    fn from((x, y): (i16, i16)) -> Self {
        Self(x, y)
    }
}

impl From<Point2> for (i16, i16) {
    fn from(Point2(x, y): Point2) -> Self {
        (x, y)
    }
}

impl TryFrom<(usize, usize)> for Point2 {
    type Error = TryFromIntError;

    fn try_from((x, y): (usize, usize)) -> Result<Self, Self::Error> {
        Ok(Self(x.try_into()?, y.try_into()?))
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Material {
    #[default]
    Outside,
    Open,
    Wall,
}

impl Display for Material {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Material::*;

        match self {
            Outside => ' ',
            Open => '.',
            Wall => '#',
        }
        .fmt(f)
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Dimension {
    X,
    Y,
}

#[derive(Clone, Copy, Debug)]
struct Row<'a> {
    grid: &'a Grid,
    locked_on: Dimension,
    ix: usize,
}

impl Row<'_> {
    fn end(&self) -> Option<Point2> {
        use Dimension::*;

        self.iter()
            .enumerate()
            .rfind(|&(_, m)| m != Material::Outside)
            .map(|(n, _)| {
                match self.locked_on {
                    X => (self.ix, n),
                    Y => (n, self.ix),
                }
                .try_into()
                .expect("this should never overflow an i16")
            })
    }

    fn first_at(&self, dir: RowDirection) -> Option<Point2> {
        use RowDirection::*;

        match dir {
            Start => self.start(),
            End => self.end(),
        }
    }

    fn iter(
        &self,
    ) -> impl ExactSizeIterator<Item = Material> + DoubleEndedIterator<Item = Material> + '_ {
        let (dim_w, dim_h) = self.grid.dims();

        use Dimension::*;

        match self.locked_on {
            X => 0..dim_h,
            Y => 0..dim_w,
        }
        .map(move |ix| self[ix])
    }

    fn start(&self) -> Option<Point2> {
        use Dimension::*;

        self.iter()
            .enumerate()
            .find(|&(_, m)| m != Material::Outside)
            .map(|(n, _)| {
                match self.locked_on {
                    X => (self.ix, n),
                    Y => (n, self.ix),
                }
                .try_into()
                .expect("this should never overflow an i16")
            })
    }
}

impl Index<usize> for Row<'_> {
    type Output = Material;

    fn index(&self, a: usize) -> &Self::Output {
        use Dimension::*;

        let coord = match self.locked_on {
            X => (self.ix, a),
            Y => (a, self.ix),
        };

        &self.grid[coord]
    }
}

#[derive(Debug)]
struct Grid {
    grid: Vec<Vec<Material>>,
    dim: (usize, usize),
}

impl Grid {
    fn new(mut grid: Vec<Vec<Material>>, dim_x: usize) -> Self {
        for row in &mut grid {
            row.resize(dim_x, Material::Outside);
        }

        let dim_y = grid.len();

        assert!(i16::try_from(dim_x).and(i16::try_from(dim_y)).is_ok());

        Self {
            grid,
            dim: (dim_x, dim_y),
        }
    }

    const fn dims(&self) -> (usize, usize) {
        self.dim
    }

    fn is_inside(&self, p @ Point2(x, y): Point2) -> bool {
        let (dim_x, dim_y) = (self.dims().0 as i16, self.dims().1 as i16);

        (0..dim_x).contains(&x) && (0..dim_y).contains(&y) && self[p] != Material::Outside
    }

    fn is_outside(&self, p: Point2) -> bool {
        !self.is_inside(p)
    }

    fn row_on(&self, dim: Dimension, ix: usize) -> Row<'_> {
        Row {
            grid: self,
            locked_on: dim,
            ix,
        }
    }

    fn rows(&self) -> impl Iterator<Item = Row<'_>> {
        self.rows_on(Dimension::Y)
    }

    fn rows_on(&self, dim: Dimension) -> impl Iterator<Item = Row<'_>> {
        let (dim_x, dim_y) = self.dims();

        use Dimension::*;

        match dim {
            X => 0..dim_x,
            Y => 0..dim_y,
        }
        .map(move |ix| self.row_on(dim, ix))
    }
}

impl Index<Point2> for Grid {
    type Output = Material;

    fn index(&self, Point2(i, j): Point2) -> &Self::Output {
        &self[(i as usize, j as usize)]
    }
}

impl IndexMut<Point2> for Grid {
    fn index_mut(&mut self, Point2(i, j): Point2) -> &mut Self::Output {
        &mut self[(i as usize, j as usize)]
    }
}

impl Index<(usize, usize)> for Grid {
    type Output = Material;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.grid[j][i]
    }
}

impl IndexMut<(usize, usize)> for Grid {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.grid[j][i]
    }
}

fn dump_map(g: &Grid, traversed: &HashMap<Point2, Direction>) {
    for (j, row) in g.rows().enumerate() {
        for (i, m) in row.iter().enumerate() {
            let p = (i, j).try_into().unwrap();

            if let Some(dir) = traversed.get(&p) {
                print!("{dir}");
            } else {
                print!("{m}");
            }
        }

        println!();
    }

    println!();
}

fn advance_in(g: &Grid, Pos { mut pos, dir }: Pos, count: usize) -> Pos {
    assert!(g.is_inside(pos));

    for _n in 0..count {
        let mut next = pos + dir;

        if g.is_outside(next) {
            // we are outside the map, roll over

            let dim = dir.dim();

            next = g
                .row_on(dim, pos[dim] as usize)
                .first_at(dir.opposite().into())
                .expect("a value on the other side");
        }

        if g[next] == Material::Wall {
            // go back and stop

            break;
        }

        pos = next;
    }

    Pos { pos, dir }
}

fn spawn_in(g: &Grid) -> Option<Pos> {
    for (j, row) in g.rows().enumerate() {
        if let Some((i, _)) = row.iter().enumerate().find(|(_, m)| *m == Material::Open) {
            return Some(Pos {
                pos: (i, j).try_into().expect("map is too big"),
                dir: Direction::East,
            });
        }
    }

    None
}

fn row(input: &str) -> IResult<&str, Vec<Material>> {
    use Material::*;

    terminated(
        many1(alt((
            value(Outside, char(' ')),
            value(Open, char('.')),
            value(Wall, char('#')),
        ))),
        line_ending,
    )(input)
}

fn grid(input: &str) -> IResult<&str, Grid> {
    map(
        fold_many0(
            row,
            || (0, Vec::new()),
            |(mut max_len, mut vec), cr| {
                max_len = max(max_len, cr.len());
                vec.push(cr);

                (max_len, vec)
            },
        ),
        |(dim_x, vec)| Grid::new(vec, dim_x),
    )(input)
}

#[derive(Clone, Copy, Debug)]
enum Command {
    Forward(usize),
    Turn(Rotate),
}

fn commands(input: &str) -> IResult<&str, Vec<Command>> {
    use Command::*;
    use Rotate::*;

    terminated(
        many0(alt((
            map(unsigned, Forward),
            value(Turn(Left), char('L')),
            value(Turn(Right), char('R')),
        ))),
        multispace0,
    )(input)
}

fn map_input(input: &str) -> IResult<&str, (Grid, Vec<Command>)> {
    all_consuming(terminated(
        separated_pair(grid, line_ending, commands),
        multispace0,
    ))(input)
}

#[derive(Clone, Copy, Debug)]
struct Pos {
    pos: Point2,
    dir: Direction,
}

impl Pos {
    const fn eval(self) -> isize {
        let Self {
            pos: Point2(x, y),
            dir,
        } = self;

        let (x, y, dir) = (x as isize + 1, y as isize + 1, dir as isize);

        1000 * y + 4 * x + dir
    }

    fn rotate(self, rot: Rotate) -> Self {
        Self {
            pos: self.pos,
            dir: self.dir.turn(rot),
        }
    }
}

/// AoC problem for Dec 22 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let (g, cmds) = map_input(&fs::read_to_string(file)?)
        .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
        .1;

    let mut cur = spawn_in(&g).ok_or("grid is degenerate: no open spots")?;

    let mut traversed = HashMap::from_iter([(cur.pos, cur.dir)]);

    for cmd in cmds {
        cur = match cmd {
            Command::Forward(no) => advance_in(&g, cur, no),
            Command::Turn(rot) => cur.rotate(rot),
        };

        traversed.insert(cur.pos, cur.dir);
    }

    dump_map(&g, &traversed);

    println!("{cur:?}, eval = {}", cur.eval());

    Ok(())
}
