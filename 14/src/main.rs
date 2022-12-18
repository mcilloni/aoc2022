use std::{
    cmp::{max, min},
    collections::VecDeque,
    error::Error,
    fmt::Display,
    fs::{self},
    iter::from_fn,
    str::FromStr,
    vec,
};

use clap::Parser;
use nom::{
    bytes::complete::{tag, take_while},
    character::complete::{char, multispace0, space0},
    combinator::{all_consuming, map, map_res},
    error::{Error as NomError, ParseError},
    multi::{separated_list0, separated_list1},
    sequence::{delimited, separated_pair},
    IResult,
};
use num::{Integer, Unsigned};

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

fn new_row(floor: usize) -> Vec<Material> {
    let mut ret = vec![Material::Void; floor + 1];

    ret[floor] = Material::Rock;

    ret
}

const ZERO: Point2 = Point2(500, 0);

const fn skew_for(x: usize) -> usize {
    x.abs_diff(ZERO.0)
}

const fn fix_xindex(x: usize, grid_len: usize) -> usize {
    let skew = skew_for(x);

    if x > ZERO.0 {
        grid_len / 2 + skew
    } else {
        grid_len / 2 - skew
    }
}

fn remap_xindex(grid: &mut VecDeque<Vec<Material>>, floor: usize, x: usize) -> usize {
    assert!(grid.len().is_odd());

    let skew = x.abs_diff(ZERO.0);

    let cur_skew = grid.len() / 2;

    if cur_skew < skew {
        // push lines on both directions
        for _ in 0..skew - cur_skew {
            grid.push_front(new_row(floor));
            grid.push_back(new_row(floor));
        }
    }

    let row_id = fix_xindex(x, grid.len());

    assert!(row_id < grid.len());

    row_id
}

fn mark_point(
    grid: &mut VecDeque<Vec<Material>>,
    floor: usize,
    mat: Material,
    (x, y): (usize, usize),
) {
    assert!(floor >= y);

    let x = remap_xindex(grid, floor, x);

    grid[x][y] = mat;
}

fn mark_straight(
    grid: &mut VecDeque<Vec<Material>>,
    floor: usize,
    coords: impl Iterator<Item = (usize, usize)>,
) {
    for coord in coords {
        mark_point(grid, floor, Material::Rock, coord);
    }
}

fn mark_line(
    grid: &mut VecDeque<Vec<Material>>,
    floor: usize,
    (start, end): (Point2, Point2),
    _val: Material,
) {
    let Point2(s_x, s_y) = start;
    let Point2(e_x, e_y) = end;

    if s_x == e_x {
        let (s_y, e_y) = (min(s_y, e_y), max(s_y, e_y));

        mark_straight(grid, floor, (s_y..=e_y).map(move |y| (s_x, y)));
    } else if s_y == e_y {
        let (s_x, e_x) = (min(s_x, e_x), max(s_x, e_x));
        mark_straight(grid, floor, (s_x..=e_x).map(move |x| (x, s_y)));
    } else {
        panic!("diagonals are unsupported ATM");
    }
}

fn mark_path(
    grid: &mut VecDeque<Vec<Material>>,
    floor: usize,
    path: impl Into<Path>,
    mat: Material,
) {
    for line in path.into().pairs() {
        mark_line(grid, floor, line, mat);
    }
}

const FLOOR_DIFF: usize = 2;

struct TerrainMap {
    grid: VecDeque<Vec<Material>>,
    floor: usize,
}

impl TerrainMap {
    fn from_iter<I, T>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>,
        <I as IntoIterator>::IntoIter: Clone,
        T: Into<Path> + IntoIterator<Item = Point2>,
    {
        let iter = iter.into_iter();

        let floor = iter.clone().flatten().fold(ZERO.1, |max_d, cur| {
            let Point2(_, cur_d) = cur;

            max(max_d, cur_d)
        }) + FLOOR_DIFF;

        let mut lines = VecDeque::from([new_row(floor)]);

        for path in iter {
            mark_path(&mut lines, floor, path, Material::Rock);
        }

        Self { grid: lines, floor }
    }

    fn below(&mut self, point: Point2) -> Option<(Point2, Material)> {
        let point = self.down(point);
        self.get_mut(point.into()).map(|m| (point, *m))
    }

    fn below_left(&mut self, point: Point2) -> Option<(Point2, Material)> {
        self.diag_left(point)
            .and_then(|point| self.get_mut(point.into()).map(|m| (point, *m)))
    }

    fn below_right(&mut self, point: Point2) -> Option<(Point2, Material)> {
        self.diag_right(point)
            .and_then(|point| self.get_mut(point.into()).map(|m| (point, *m)))
    }

    fn dump(&self) {
        let mut rits: Vec<_> = self.grid.iter().map(|v| v.iter()).collect();

        for _ in 0..=self.floor {
            for rit in &mut rits {
                print!(
                    "{}",
                    rit.next().expect("vectors must all have size == floor + 1")
                );
            }

            println!();
        }
    }

    fn diag_left(&mut self, Point2(i, j): Point2) -> Option<Point2> {
        i.checked_sub(1).and_then(|ni| {
            // ensure mapping of row
            remap_xindex(&mut self.grid, self.floor, ni);

            match j.checked_add(1) {
                Some(nj) if nj <= self.floor => Some(Point2(ni, nj)),
                _ => None,
            }
        })
    }

    fn diag_right(&mut self, Point2(i, j): Point2) -> Option<Point2> {
        let ni = i.checked_add(1).unwrap();

        // ensure mapping of row
        remap_xindex(&mut self.grid, self.floor, ni);

        let nj = j.checked_add(1).unwrap();
        if nj <= self.floor {
            Some(Point2(ni, nj))
        } else {
            None
        }
    }

    fn down(&self, Point2(i, j): Point2) -> Point2 {
        Point2(i, j.checked_add(1).unwrap())
    }

    fn get(&self, (i, j): (usize, usize)) -> Option<&Material> {
        let i = fix_xindex(i, self.grid.len());

        self.grid.get(i).and_then(|v| v.get(j))
    }

    fn get_mut(&mut self, (i, j): (usize, usize)) -> Option<&mut Material> {
        let i = fix_xindex(i, self.grid.len());

        self.grid.get_mut(i).and_then(|v| v.get_mut(j))
    }
}

fn spawn_at(tmap: &mut TerrainMap, mat: Material, mut coord: Point2) -> Option<()> {
    let blocked = |coord| *tmap.get(coord).unwrap() != Material::Void;

    if blocked(coord.into()) {
        return None;
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
