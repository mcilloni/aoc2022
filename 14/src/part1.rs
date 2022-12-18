use std::{
    cmp::{max, min},
    error::Error,
    fmt::Display,
    fs::{self},
    iter::from_fn,
    str::FromStr,
    vec,
};

use clap::Parser;
use ndarray::Array2;
use nom::{
    bytes::complete::{tag, take_while},
    character::complete::{char, multispace0, space0},
    combinator::{all_consuming, map, map_res},
    error::{Error as NomError, ParseError},
    multi::{separated_list0, separated_list1},
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Point2(usize, usize);

impl From<(usize, usize)> for Point2 {
    fn from((x, y): (usize, usize)) -> Self {
        Self(x, y)
    }
}

impl From<Point2> for (usize, usize) {
    fn from(Point2(x, y): Point2) -> Self {
        (x, y)
    }
}

fn point2(input: &str) -> IResult<&str, Point2> {
    map(
        separated_pair(ws(unsigned), char(','), ws(unsigned)),
        Into::into,
    )(input)
}

#[derive(Clone, Debug, Default)]
struct Path(Vec<Point2>);

impl Path {
    fn pairs(&self) -> impl Iterator<Item = (Point2, Point2)> + '_ {
        self.0.windows(2).map(|w| {
            let [e1, e2] = w else {
                panic!("windows is broken");
            };

            (*e1, *e2)
        })
    }
}

impl<I: Into<Point2>> FromIterator<I> for Path {
    fn from_iter<T: IntoIterator<Item = I>>(iter: T) -> Self {
        Self(iter.into_iter().map(Into::into).collect())
    }
}

impl IntoIterator for Path {
    type Item = Point2;

    type IntoIter = vec::IntoIter<<Self as IntoIterator>::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

fn path(input: &str) -> IResult<&str, Path> {
    map(separated_list1(tag("->"), ws(point2)), Path)(input)
}

fn paths(input: &str) -> IResult<&str, Vec<Path>> {
    all_consuming(delimited(
        multispace0,
        separated_list0(multispace0, path),
        multispace0,
    ))(input)
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Material {
    Rock,
    Sand,
    Void,
}

impl Material {
    fn is_void(self) -> bool {
        self == Self::Void
    }
}

impl Display for Material {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Material::*;

        match self {
            Rock => '#',
            Sand => 'o',
            Void => '.',
        }
        .fmt(f)
    }
}

const fn normalize(zero: usize) -> impl Fn((usize, usize)) -> (usize, usize) + Sync {
    move |(x, y)| (x.checked_sub(zero).unwrap(), y)
}

const ZERO: Point2 = Point2(500, 0);

fn mark_straight(grid: &mut Array2<Material>, coords: impl Iterator<Item = (usize, usize)>) {
    for coord in coords {
        *grid
            .get_mut(coord)
            .unwrap_or_else(|| panic!("must have {coord:?}")) = Material::Rock;
    }
}

fn mark_line(
    grid: &mut Array2<Material>,
    base_x: usize,
    (start, end): (Point2, Point2),
    _val: Material,
) {
    let norm = normalize(base_x);

    let (s_x, s_y) = norm(start.into());
    let (e_x, e_y) = norm(end.into());

    if s_x == e_x {
        let (s_y, e_y) = (min(s_y, e_y), max(s_y, e_y));

        mark_straight(grid, (s_y..=e_y).map(move |y| (s_x, y)));
    } else if s_y == e_y {
        let (s_x, e_x) = (min(s_x, e_x), max(s_x, e_x));
        mark_straight(grid, (s_x..=e_x).map(move |x| (x, s_y)));
    } else {
        panic!("diagonals are unsupported ATM");
    }
}

struct TerrainMap {
    grid: Array2<Material>,
    base: usize,
}

impl TerrainMap {
    fn from_iter<I, T>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        <I as IntoIterator>::IntoIter: Clone,
        T: Into<Path> + IntoIterator<Item = Point2>,
    {
        let iter = iter.into_iter();

        let (min_x, max_x, max_d) =
            iter.clone()
                .flatten()
                .fold((ZERO.0, ZERO.0, ZERO.1), |best, cur| {
                    let (min_x, max_x, max_d) = best;
                    let Point2(cur_x, cur_d) = cur;

                    (min(min_x, cur_x), max(max_x, cur_x), max(max_d, cur_d))
                });

        let mut lines = Array2::from_elem((max_x - min_x + 1, max_d + 1), Material::Void);

        for path in iter {
            for line in path.into().pairs() {
                mark_line(&mut lines, min_x, line, Material::Rock);
            }
        }

        Self {
            base: min_x,
            grid: lines,
        }
    }

    fn below(&self, point: Point2) -> Option<(Point2, &Material)> {
        let point = self.down(point);
        self.get(point.into()).map(|m| (point, m))
    }

    fn below_left(&self, point: Point2) -> Option<(Point2, &Material)> {
        self.diag_left(point)
            .and_then(|point| self.get(point.into()).map(|m| (point, m)))
    }

    fn below_right(&self, point: Point2) -> Option<(Point2, &Material)> {
        self.diag_right(point)
            .and_then(|point| self.get(point.into()).map(|m| (point, m)))
    }

    fn dump(&self) {
        for column in self.grid.columns() {
            for c in column {
                print!("{}", c);
            }

            println!();
        }
    }

    fn diag_left(&self, Point2(i, j): Point2) -> Option<Point2> {
        i.checked_sub(1)
            .map(|ni| Point2(ni, j.checked_add(1).unwrap()))
    }

    fn diag_right(&self, Point2(i, j): Point2) -> Option<Point2> {
        let ni = i.checked_add(1).unwrap();

        if ni < self.grid.dim().0 {
            Some(Point2(ni, j.checked_add(1).unwrap()))
        } else {
            None
        }
    }

    fn down(&self, Point2(i, j): Point2) -> Point2 {
        Point2(i, j.checked_add(1).unwrap())
    }

    fn get(&self, coord: (usize, usize)) -> Option<&Material> {
        self.grid.get(coord)
    }

    fn get_mut(&mut self, coord: (usize, usize)) -> Option<&mut Material> {
        self.grid.get_mut(coord)
    }

    fn normalized(&self, point: Point2) -> Point2 {
        normalize(self.base)(point.into()).into()
    }
}

fn spawn_at(tmap: &mut TerrainMap, mat: Material, coord: Point2) -> Option<()> {
    let blocked = |coord| *tmap.get(coord).unwrap() != Material::Void;

    let mut coord = tmap.normalized(coord);

    if blocked(coord.into()) {
        panic!("source blocked - not handled atm");
    }

    loop {
        // assume we are on a Void cell and attempt to move below
        let (below, m) = tmap.below(coord)?;

        if m.is_void() {
            coord = below;
            continue;
        }

        let (bl, m) = tmap.below_left(coord)?;

        if m.is_void() {
            coord = bl;
            continue;
        }

        let (br, m) = tmap.below_right(coord)?;

        if m.is_void() {
            coord = br;
            continue;
        }

        // commit the material to this spot
        let val_cur = tmap.get_mut(coord.into())?;

        *val_cur = mat;

        break Some(());
    }
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

    let mut terrain = TerrainMap::from_iter(
        paths(&fs::read_to_string(file)?)
            .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
            .1
            .into_iter(),
    );

    let cnt = from_fn(|| spawn_at(&mut terrain, Material::Sand, ZERO)).count();

    terrain.dump();

    println!("cnt = {cnt}");

    Ok(())
}
