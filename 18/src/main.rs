use std::{
    cmp::max,
    collections::VecDeque,
    error::Error,
    fs,
    num::TryFromIntError,
    ops::{Add, Index, IndexMut},
    str::FromStr,
    vec,
};

use clap::Parser;
use nom::{
    bytes::complete::take_while,
    character::complete::{char, multispace0, space0},
    combinator::{all_consuming, map, map_res},
    error::{Error as NomError, ParseError},
    multi::separated_list0,
    sequence::{delimited, tuple},
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
struct Point3(usize, usize, usize);

impl From<(usize, usize, usize)> for Point3 {
    fn from((x, y, z): (usize, usize, usize)) -> Self {
        Self(x, y, z)
    }
}

impl From<Point3> for (usize, usize) {
    fn from(Point3(x, y, _z): Point3) -> Self {
        (x, y)
    }
}

impl TryFrom<(isize, isize, isize)> for Point3 {
    type Error = TryFromIntError;

    fn try_from((x, y, z): (isize, isize, isize)) -> Result<Self, Self::Error> {
        Ok(Self(x.try_into()?, y.try_into()?, z.try_into()?))
    }
}

impl Add for Point3 {
    type Output = Point3;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
    }
}

impl Add<(usize, usize, usize)> for Point3 {
    type Output = Point3;

    fn add(self, rhs: (usize, usize, usize)) -> Self::Output {
        self + Self::from(rhs)
    }
}

impl Add<Point3> for (isize, isize, isize) {
    type Output = (isize, isize, isize);

    fn add(self, rhs: Point3) -> Self::Output {
        (
            self.0 + rhs.0 as isize,
            self.1 + rhs.1 as isize,
            self.2 + rhs.2 as isize,
        )
    }
}

fn point3(input: &str) -> IResult<&str, Point3> {
    map(
        tuple((
            ws(unsigned),
            char(','),
            ws(unsigned),
            char(','),
            ws(unsigned),
        )),
        |(x, _, y, _, z)| Point3(x, y, z),
    )(input)
}

fn edge_detect(line: impl Iterator<Item = Material>) -> usize {
    let mut tot = 0;

    let mut inside = false;

    for point in line {
        use Material::*;

        match (point, inside) {
            (Rock, false) => {
                tot += 1;
                inside = true;
            }
            (Water, true) => {
                tot += 1;
                inside = false;
            }
            _ => {}
        }
    }

    // add 1 if on the edge
    tot + inside as usize
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Dimension2 {
    W,
    H,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Dimension3 {
    X,
    Y,
    Z,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Material {
    #[default]
    Unknown,
    Water,
    Rock,
}

#[derive(Clone, Copy, Debug)]
struct Row<'a> {
    plane: Plane<'a>,
    locked_on: Dimension2,
    ix: usize,
}

impl Row<'_> {
    fn iter(&self) -> impl Iterator<Item = Material> + '_ {
        let (dim_w, dim_h) = self.plane.dims();

        use Dimension2::*;

        match self.locked_on {
            W => 0..dim_h,
            H => 0..dim_w,
        }
        .map(move |ix| self[ix])
    }
}

impl Index<usize> for Row<'_> {
    type Output = Material;

    fn index(&self, a: usize) -> &Self::Output {
        use Dimension2::*;

        let coord = match self.locked_on {
            W => (self.ix, a),
            H => (a, self.ix),
        };

        &self.plane[coord]
    }
}

#[derive(Clone, Copy, Debug)]
struct Plane<'a> {
    cbd: &'a Cuboid,
    locked_on: Dimension3,
    ix: usize,
}

impl Plane<'_> {
    fn dims(&self) -> (usize, usize) {
        let (dim_x, dim_y, dim_z) = self.cbd.dim;

        use Dimension3::*;

        match self.locked_on {
            X => (dim_y, dim_z),
            Y => (dim_x, dim_z),
            Z => (dim_x, dim_y),
        }
    }

    fn row_on(&self, dim: Dimension2, ix: usize) -> Row<'_> {
        Row {
            plane: *self,
            locked_on: dim,
            ix,
        }
    }

    fn rows_on(&self, dim: Dimension2) -> impl Iterator<Item = Row<'_>> {
        let (dim_w, dim_h) = self.dims();

        use Dimension2::*;

        match dim {
            W => 0..dim_w,
            H => 0..dim_h,
        }
        .map(move |ix| self.row_on(dim, ix))
    }
}

impl Index<(usize, usize)> for Plane<'_> {
    type Output = Material;

    fn index(&self, (a, b): (usize, usize)) -> &Self::Output {
        use Dimension3::*;

        let coord = match self.locked_on {
            X => (self.ix, a, b),
            Y => (a, self.ix, b),
            Z => (a, b, self.ix),
        }
        .into();

        &self.cbd[coord]
    }
}

#[derive(Debug)]
struct Cuboid {
    grid: Vec<Vec<Vec<Material>>>,
    dim: (usize, usize, usize),
}

impl Cuboid {
    fn new(dim_x: usize, dim_y: usize, dim_z: usize) -> Self {
        Self {
            grid: vec![vec![vec![Material::Unknown; dim_z]; dim_y]; dim_x],
            dim: (dim_x, dim_y, dim_z),
        }
    }

    fn from_points<I, T>(iter: T) -> Self
    where
        T: IntoIterator<Item = I>,
        T::IntoIter: Iterator<Item = I> + Clone,
        I: Into<Point3>,
    {
        let iter = iter.into_iter();

        let (dim_x, dim_y, dim_z) = iter.clone().map(Into::into).fold(
            (0usize, 0usize, 0usize),
            |(best_x, best_y, best_z), Point3(x, y, z)| {
                (max(best_x, x), max(best_y, y), max(best_z, z))
            },
        );

        let mut ret = Self::new(dim_x + 1, dim_y + 1, dim_z + 1);

        for Point3(x, y, z) in iter.map(Into::into) {
            ret.grid[x][y][z] = Material::Rock;
        }

        ret.flood_fill();

        ret
    }

    fn flood_fill(&mut self) {
        use Material::*;

        let (dim_x, dim_y, dim_z) = self.dim;

        let Some(first_void) = ('search: {
            for i in 0..dim_x {
                for j in 0..dim_y {
                    for k in 0..dim_z {
                        if self[Point3(i,j,k)] == Unknown {
                            break 'search Some(Point3(i, j, k));
                        }
                    }
                }
            }

            None
        }) else { return; };

        let mut queue = VecDeque::from([first_void]);

        const NEIGH: [(isize, isize, isize); 6] = [
            (0, 0, 1),
            (0, 1, 0),
            (1, 0, 0),
            (0, 0, -1),
            (0, -1, 0),
            (-1, 0, 0),
        ];

        while let Some(point) = queue.pop_front() {
            if self[point] == Unknown {
                self[point] = Water;

                queue.extend(
                    NEIGH
                        .into_iter()
                        .map(|neigh| (neigh + point).try_into())
                        .filter_map(|maybe_p: Result<Point3, _>| maybe_p.ok())
                        .filter(|&Point3(x, y, z)| x < dim_x && y < dim_y && z < dim_z),
                )
            }
        }
    }

    fn plane_on(&self, dim: Dimension3, ix: usize) -> Plane<'_> {
        Plane {
            cbd: self,
            locked_on: dim,
            ix,
        }
    }

    fn planes_on(&self, dim: Dimension3) -> impl Iterator<Item = Plane<'_>> {
        let (dim_x, dim_y, dim_z) = self.dim;

        use Dimension3::*;

        match dim {
            X => 0..dim_x,
            Y => 0..dim_y,
            Z => 0..dim_z,
        }
        .map(move |ix| self.plane_on(dim, ix))
    }

    fn surface_area(&self) -> usize {
        use Dimension2::*;
        use Dimension3::*;

        let mut area = 0;

        for (axis, dir) in [(X, W), (Z, W), (Y, H)] {
            for plane in self.planes_on(axis) {
                for row in plane.rows_on(dir) {
                    area += edge_detect(row.iter());
                }
            }
        }

        area
    }
}

impl Index<Point3> for Cuboid {
    type Output = Material;

    fn index(&self, Point3(i, j, k): Point3) -> &Self::Output {
        &self.grid[i][j][k]
    }
}

impl IndexMut<Point3> for Cuboid {
    fn index_mut(&mut self, Point3(i, j, k): Point3) -> &mut Self::Output {
        &mut self.grid[i][j][k]
    }
}

fn grid(input: &str) -> IResult<&str, Cuboid> {
    map(
        all_consuming(delimited(
            multispace0,
            separated_list0(multispace0, point3),
            multispace0,
        )),
        Cuboid::from_points,
    )(input)
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

    let grd = grid(&fs::read_to_string(file)?)
        .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
        .1;

    println!("grd = {grd:?}");
    println!("surface_area = {}", grd.surface_area());

    Ok(())
}
