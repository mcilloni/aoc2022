use std::{
    cmp::max,
    error::Error,
    fmt::Display,
    fs::File,
    io::{stdin, Read},
    mem::replace,
    ops::{Add, AddAssign, Sub, SubAssign},
};

use clap::Parser;

/// AoC problem for Dec 17 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,

    /// enables debug mode
    #[arg(long, default_value_t = false)]
    debug: bool,

    // skips to a certain block
    #[arg(long, default_value_t = 0)]
    skip_to: usize,
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
enum Direction {
    Left,
    Right,
}

impl Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::Left => '<',
            Direction::Right => '>',
        }
        .fmt(f)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Point2(isize, isize);

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

impl Add for Point2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(
            self.0.checked_add(rhs.0).unwrap(),
            self.1.checked_add(rhs.1).unwrap(),
        )
    }
}

impl Add<(isize, isize)> for Point2 {
    type Output = Self;

    fn add(self, rhs: (isize, isize)) -> Self::Output {
        Self(
            self.0.checked_add(rhs.0).unwrap(),
            self.1.checked_add(rhs.1).unwrap(),
        )
    }
}

impl AddAssign<(isize, isize)> for Point2 {
    fn add_assign(&mut self, rhs: (isize, isize)) {
        *self = *self + rhs;
    }
}

impl PartialEq<(isize, isize)> for Point2 {
    fn eq(&self, other: &(isize, isize)) -> bool {
        *self == Self::from(*other)
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

impl Sub<(isize, isize)> for Point2 {
    type Output = Self;

    fn sub(self, rhs: (isize, isize)) -> Self::Output {
        Self(
            self.0.checked_sub(rhs.0).unwrap(),
            self.1.checked_sub(rhs.1).unwrap(),
        )
    }
}

impl SubAssign<(isize, isize)> for Point2 {
    fn sub_assign(&mut self, rhs: (isize, isize)) {
        *self = *self - rhs;
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug)]
enum Shape {
    Line,
    Plus,
    L,
    I,
    Square,
}

impl Shape {
    const fn index(self) -> usize {
        self as usize
    }

    fn bottom_for(self, left: Point2) -> impl Iterator<Item = Point2> + Clone + 'static {
        BOTTOM[self.index()].iter().cloned().map(move |p| p + left)
    }

    fn right_for(self, left: Point2) -> Point2 {
        RVER[self.index()] + left
    }

    fn outline_for(self, left: Point2) -> impl Iterator<Item = Point2> + Clone + 'static {
        SHAPES[self.index()].iter().cloned().map(move |p| p + left)
    }

    fn top_for(self, left: Point2) -> Point2 {
        TVER[self.index()] + left
    }
}

const SHAPE_KINDS: usize = 5;
const SHAPE_PATTERNS: [Shape; SHAPE_KINDS] =
    [Shape::Line, Shape::Plus, Shape::L, Shape::I, Shape::Square];

const SHAPES: [&[Point2]; SHAPE_KINDS] = [
    &[Point2(0, 0), Point2(1, 0), Point2(2, 0), Point2(3, 0)],
    &[
        Point2(0, 0),
        Point2(1, 0),
        Point2(1, -1),
        Point2(2, 0),
        Point2(1, 1),
    ],
    &[
        Point2(0, 0),
        Point2(1, 0),
        Point2(2, 0),
        Point2(2, 1),
        Point2(2, 2),
    ],
    &[Point2(0, 0), Point2(0, 1), Point2(0, 2), Point2(0, 3)],
    &[Point2(0, 0), Point2(1, 0), Point2(1, 1), Point2(0, 1)],
];

const BOTTOM: [&[Point2]; SHAPE_KINDS] = [
    &[Point2(0, 0), Point2(1, 0), Point2(2, 0), Point2(3, 0)],
    &[Point2(0, 0), Point2(1, -1), Point2(2, 0)],
    &[Point2(0, 0), Point2(1, 0), Point2(2, 0)],
    &[Point2(0, 0)],
    &[Point2(0, 0), Point2(1, 0)],
];

const BVER: [Point2; SHAPE_KINDS] = [
    Point2(0, 0),
    Point2(1, -1),
    Point2(0, 0),
    Point2(0, 0),
    Point2(0, 0),
];

const RVER: [Point2; SHAPE_KINDS] = [
    Point2(3, 0),
    Point2(2, 0),
    Point2(2, 0),
    Point2(0, 0),
    Point2(1, 0),
];

const TVER: [Point2; SHAPE_KINDS] = [
    Point2(0, 0),
    Point2(1, 1),
    Point2(2, 2),
    Point2(0, 3),
    Point2(1, 1),
];

const WIDTH: isize = 7;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum Material {
    #[default] Air,
    Rock,
    Wall,
}

#[derive(Debug, Default)]
struct Board {
    grid: Vec<[Material; WIDTH as usize]>,
    top: isize,
}

impl Board {
    fn new() -> Self {
        let mut ret = Self {
            grid: vec![Default::default(); 40],
            top: 0,
        };

        ret.grid[0] = [Material::Wall; WIDTH as usize];

        ret
    }

    fn shape_intersects(&self, shape: Shape, t: Point2) -> bool {
        for shape_point in shape.outline_for(t) {
            if self.intersects(shape_point) {
                return true;
            }
        }

        false
    }

    fn commit(&mut self, points: impl Iterator<Item = Point2> + Clone) {
        for point in points {
            let cell = self.get_at_mut(point).unwrap();

            *cell = Material::Rock;

            self.top = max(self.top, point.1);
        }
    }

    fn extend(&mut self) {
        self.grid.resize(self.grid.len() + 40, Default::default());
    }

    fn get_at(&self, t: Point2) -> Option<Material> {
        assert!(t.0 >= 0 && t.0 < WIDTH && t.1 >= 0 && t.1 <= self.grid.len() as isize);

        self.grid
            .get(t.1 as usize)
            .and_then(|r| r.get(t.0 as usize))
            .cloned()
    }

    fn get_at_mut(&mut self, t: Point2) -> Option<&mut Material> {
        assert!(t.0 >= 0 && t.0 < WIDTH && t.1 >= 0 && t.1 <= self.grid.len() as isize);

        self.grid
            .get_mut(t.1 as usize)
            .and_then(|r| r.get_mut(t.0 as usize))
    }

    fn intersects(&self, t: Point2) -> bool {
        self.get_at(t)
            .map(|mat| mat != Material::Air)
            .unwrap_or_default()
    }

    fn is_close(&self, left: Point2) -> bool {
        left.1 <= self.top + 2
    }

    fn spawn_point(&self, shape: Shape) -> Point2 {
        const X_SKEW: isize = 2;
        const Y_SKEW: isize = 4; // floor is 0, thus 3 + 1

        // (X_SKEW, Ly) + (Bx, By) = (X_SKEW + Bx, Ly + By) = (X_SKEW + Bx, top + Y_SKEW) =>
        // Ly + By = top + Y_SKEW => Ly = top + Y_SKEW - By

        let bvx = BVER[shape.index()];

        Point2(X_SKEW, Y_SKEW + self.top - bvx.1)
    }

    fn try_insert_shape_at(&mut self, shape: Shape, left: Point2) -> bool {
        // assume left is the (0, 0) point of a shape of kind = shape

        // all shapes have (0,0) in their bottom, and only '+' doesn't have it as its minimum vertex
        // skip any checks if this is true and avoid pointless checks when floating amidst air
        if self.is_close(left) {
            if left.1 as usize > self.grid.len() - 6 {
                self.extend();
            }

            // check if the shape's nottom intersects when translated at left - 1. If this is true, commit shape at 'left'
            for bottom_point in shape.bottom_for(left - (0, 1)) {
                if self.intersects(bottom_point) {
                    self.commit(shape.outline_for(left));

                    return true;
                }
            }
        }

        false
    }

    fn try_move_at(&mut self, shape: Shape, left: Point2, dir: Direction) -> Option<Point2> {
        use Direction::*;

        if left.1 as usize > self.grid.len() - 6 {
            self.extend();
        }

        match dir {
            Left => {
                let next = left - (1, 0);

                (next.0 >= 0 && !self.shape_intersects(shape, next)).then_some(next)
            }
            Right => {
                let next = left + (1, 0);

                (shape.right_for(next).0 < WIDTH && !self.shape_intersects(shape, next))
                    .then_some(next)
            }
        }
    }
}

struct DebugGrid {
    grid: [[Material; WIDTH as usize]; 40],
    slide: usize,
    delta: isize,
}

impl DebugGrid {
    fn new() -> Self {
        let mut ret = Self {
            grid: [[Material::Air; WIDTH as usize]; 40],
            slide: 0,
            delta: 0,
        };

        ret.grid[0] = [Material::Wall; WIDTH as usize];

        ret
    }

    fn draw(&self) {
        for (rn, row) in self.grid.iter().enumerate().rev() {
            let rn = rn + self.slide;

            let div = if rn == 0 { '+' } else { '|' };

            print!("{rn:02} {div}");

            use Material::*;

            for ch in row.iter().map(|p| match *p {
                Air => '.',
                Rock => '#',
                Wall => '-',
            }) {
                print!("{ch}");
            }

            println!("{div}");
        }
    }

    const fn blanks(&self) -> isize {
        self.rows() as isize - self.delta
    }

    fn paint(&mut self, shape: Shape, mut pos: Point2, with: Material) {
        pos -= (0, self.slide as isize);

        for Point2(x, y) in shape.outline_for(pos) {
            self.delta = max(self.delta, y);

            let (x, y) = (x as usize, y as usize);

            let old = replace(&mut self.grid[y][x], with);

            assert!(old != with);
        }
    }

    const fn rows(&self) -> usize {
        self.grid.len()
    }

    fn slide(&mut self) {
        let k = self.rows() - 20;

        self.grid.rotate_left(k);

        self.slide += k;

        let overwrite_ix = self.rows() - k;

        self.grid[overwrite_ix..].fill([Material::Air; WIDTH as usize]);
        self.delta = self.delta.saturating_sub(k as isize);
    }

    fn transl(&mut self, shape: Shape, from: Point2, to: Point2) {
        use Material::*;

        self.paint(shape, from, Air);
        self.paint(shape, to, Rock);
    }
}

const ROCK_NO: usize = 2022;

fn main() -> Result<(), Box<dyn Error>> {
    let Args {
        file,
        debug,
        skip_to,
    } = Args::parse();

    let dirs = File::open(file)?
        .bytes()
        .filter(|some_b| !matches!(some_b, Ok(b'\n' | b'\r' | b' ')))
        .map(|some_b| {
            some_b.map_err(Into::into).and_then(|b| match b {
                b'<' => Ok(Direction::Left),
                b'>' => Ok(Direction::Right),
                b => Err(format!("invalid character: {}", b as char).into()),
            })
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    let mut dirloop = dirs.into_iter().enumerate().cycle();

    let mut board = Board::new();

    let mut dbgrid = DebugGrid::new();

    for (n, &shape) in SHAPE_PATTERNS.iter().cycle().take(ROCK_NO).enumerate() {
        // start one position above the required one, so that we can subtract immediately
        // (0, 1) and avoid issues with the first iteration
        let mut cur = board.spawn_point(shape);

        if debug {
            if dbgrid.blanks() < 6 {
                dbgrid.slide();
            }

            dbgrid.paint(shape, cur, Material::Rock);

            println!("\nBLOCK #{}\n", n + 1);
            dbgrid.draw();

            if skip_to < n + 1 {
                stdin().read(&mut [0u8]).unwrap();
            }
        }

        for (dn, dir) in &mut dirloop {
            if let Some(new_cur) = board.try_move_at(shape, cur, dir) {
                if debug {
                    dbgrid.transl(shape, cur, new_cur);
                }

                cur = new_cur;
            }

            if debug {
                println!("to: {dir} [{dn}]");
                dbgrid.draw();
                println!();
            }

            if board.try_insert_shape_at(shape, cur) {
                break;
            }

            if debug {
                dbgrid.transl(shape, cur, cur - (0, 1));
            }

            cur -= (0, 1);
        }
    }

    println!("height: {}", board.top);

    Ok(())
}
