use std::{
    cmp::{max, min},
    collections::{HashMap, HashSet},
    error::Error,
    fs::File,
    io::Read,
    iter::zip,
    ops::{Add, AddAssign, Sub},
    path::Path,
};

use clap::Parser;
use num::FromPrimitive;
use num_derive::FromPrimitive;
use strum::{EnumCount, EnumIter, IntoEnumIterator};

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Point2(i64, i64);

impl Point2 {
    fn neighbourhood(self) -> [Self; Direction::COUNT] {
        let mut ret = [self; Direction::COUNT];

        for (neigh, dir) in zip(&mut ret, Direction::iter()) {
            *neigh += dir;
        }

        ret
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

#[repr(u8)]
#[derive(Clone, Copy, Debug, EnumCount, EnumIter, Eq, FromPrimitive, PartialEq)]
enum Direction {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}

impl Direction {
    fn side(self) -> [Self; 3] {
        let ix = self as isize;

        let mut ret = [self; 3];

        (ret[0], ret[2]) = Option::zip(
            Self::from_isize((ix - 1).rem_euclid(Self::COUNT as isize)),
            Self::from_isize((ix + 1).rem_euclid(Self::COUNT as isize)),
        )
        .expect("bug - enum Direction is broken");

        ret
    }

    const fn versor(self) -> Point2 {
        use Direction::*;

        match self {
            North => Point2(0, -1),
            NorthEast => Point2(1, -1),
            East => Point2(1, 0),
            SouthEast => Point2(1, 1),
            South => Point2(0, 1),
            SouthWest => Point2(-1, 1),
            West => Point2(-1, 0),
            NorthWest => Point2(-1, -1),
        }
    }
}

fn dump_pointset(points: &HashSet<Point2>) {
    let (Point2(minx, miny), Point2(maxx, maxy)) = pointset_minmax(points);

    print!("{:pad$}", "", pad = 3);

    for i in minx..=maxx {
        print!("{i:02} ");
    }

    println!();

    for j in miny..=maxy {
        print!("{j:02} ");

        for i in minx..=maxx {
            print!(
                " {} ",
                if points.contains(&Point2(i, j)) {
                    '#'
                } else {
                    '.'
                }
            );
        }

        println!("\n");
    }
}

fn pointset_minmax(points: &HashSet<Point2>) -> (Point2, Point2) {
    // find max and min in pointset - O(n)
    points
        .iter()
        .fold(None, |some_best, &p @ Point2(curx, cury)| match some_best {
            Some((Point2(minx, miny), Point2(maxx, maxy))) => Some((
                Point2(min(minx, curx), min(miny, cury)),
                Point2(max(maxx, curx), max(maxy, cury)),
            )),
            None => Some((p, p)),
        })
        .unwrap_or_default()
}

fn read_points<P: AsRef<Path>>(p: P) -> Result<HashSet<Point2>, Box<dyn Error>> {
    let mut cur = Point2::default();
    let mut ret = HashSet::new();

    for maybe_byte in File::open(p)?.bytes() {
        match maybe_byte? {
            b'\r' | b' ' | b'\t' => continue, // slurp random whitespace
            b'.' => {}
            b'\n' => {
                cur = Point2(-1, cur.1 + 1); // -1 because we're going to increase X immediately
            }
            b'#' => {
                ret.insert(cur);
            }
            b => return Err(format!("invalid byte {}", b as char).into()),
        }

        cur.0 += 1;
    }

    Ok(ret)
}

#[derive(Debug, Default)]
struct Neighbourhood {
    points: [(Point2, bool); Direction::COUNT],
    alone: bool, // to avoid traversing the array above multiple times
}

fn probe_neighbours(point: Point2, points: &HashSet<Point2>) -> Neighbourhood {
    let mut ret = [Default::default(); Direction::COUNT];
    let mut has_neigh = false;

    for (neigh, np) in zip(&mut ret, point.neighbourhood()) {
        *neigh = (np, points.contains(&np));
        has_neigh |= neigh.1;
    }

    Neighbourhood {
        points: ret,
        alone: !has_neigh,
    }
}

#[derive(Clone, Copy)]
#[repr(u8)]
enum Slot {
    Taken(Point2),
    Spoilt,
}

impl Slot {
    const fn good(self) -> Option<Point2> {
        use Slot::*;

        match self {
            Taken(p) => Some(p),
            Spoilt => None,
        }
    }
}

#[derive(Debug)]
enum State {
    Complete,
    Incomplete,
}

fn do_round(mut points: HashSet<Point2>, sides: &[Direction]) -> (State, HashSet<Point2>) {
    use Slot::*;
    use State::*;

    let mut movements = HashMap::new();

    'point_scan: for &point in &points {
        let Neighbourhood {
            points: neigh,
            alone,
        } = probe_neighbours(point, &points);

        if !alone {
            // probe all sides depending on the input directions slice

            for (d, side) in sides.iter().map(|d| (*d, d.side())) {
                if !side
                    .iter()
                    .map(|nd| neigh[*nd as usize])
                    .any(|(_, taken)| taken)
                {
                    use std::collections::hash_map::Entry::*;

                    match movements.entry(neigh[d as usize].0) {
                        Occupied(mut oe) => {
                            // mark slot as spoilt
                            *oe.get_mut() = Spoilt;
                        }
                        Vacant(ve) => {
                            ve.insert(Taken(point));
                        }
                    }

                    continue 'point_scan;
                }
            }
        }
    }

    // step 2: only apply non-spoilt translations

    let movements: Vec<_> = movements
        .into_iter()
        .filter_map(|(dest, slot)| slot.good().map(|src| (src, dest)))
        .collect();

    let state = if !movements.is_empty() {
        for (src, dest) in movements {
            points.remove(&src);
            points.insert(dest);
        }

        Incomplete
    } else {
        Complete
    };

    (state, points)
}

fn do_rounds(mut points: HashSet<Point2>) -> (usize, HashSet<Point2>) {
    use Direction::*;

    let mut directions = [North, South, West, East];

    let mut r = 1;

    loop {
        use State::*;

        match do_round(points, &directions) {
            (Complete, np) => break (r, np),
            (Incomplete, np) => points = np,
        }

        directions.rotate_left(1);

        r = r.checked_add(1).expect("no integer overflow");
    }
}

/// AoC problem for Dec 23 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let initial = read_points(file)?;

    println!("Initial:");
    dump_pointset(&initial);

    let (r, result) = do_rounds(initial);

    println!("Rounds = {r}:");
    dump_pointset(&result);

    Ok(())
}
