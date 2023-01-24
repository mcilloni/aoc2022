use std::{
    cell::{Ref, RefCell},
    cmp::Ordering,
    collections::{BinaryHeap, HashMap, HashSet},
    error::Error,
    fmt::Display,
    fs,
    iter::zip,
    num::TryFromIntError,
    ops::{Add, AddAssign, Index, IndexMut, Sub},
};

use clap::Parser;
use nom::{
    branch::alt,
    character::complete::{anychar, char, multispace1},
    combinator::{map, map_res, value},
    error::{Error as NomError, ParseError},
    multi::many1,
    sequence::{delimited, preceded, terminated, tuple},
    IResult,
};
use nom_locate::{position, LocatedSpan};
use strum::{EnumCount, EnumIter, IntoEnumIterator};

type Span<'a> = LocatedSpan<&'a str>;

fn get_point(pos: Span) -> Point2 {
    Point2(
        i16::try_from(pos.get_column()).unwrap() - 1,
        i16::try_from(pos.location_line()).unwrap() - 1,
    )
}

fn with_point<'a, F: 'a, O, E: ParseError<Span<'a>>>(
    inner: F,
) -> impl FnMut(Span<'a>) -> IResult<Span<'a>, (Point2, O), E>
where
    F: Fn(Span<'a>) -> IResult<Span<'a>, O, E>,
{
    move |s: Span<'a>| {
        let (s, pos) = position(s)?;

        let point = get_point(pos);

        let (s, res) = inner(s)?;

        Ok((s, (point, res)))
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, EnumCount, EnumIter, Eq, Hash, PartialEq)]
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
            North | South => Y,
            East | West => X,
        }
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

impl TryFrom<char> for Direction {
    type Error = String;

    fn try_from(c: char) -> Result<Self, Self::Error> {
        use Direction::*;

        match c {
            '>' => Ok(East),
            'v' => Ok(South),
            '<' => Ok(West),
            '^' => Ok(North),
            _ => Err(format!("invalid direction `{c}`")),
        }
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

impl Point2 {
    fn manhattan(self, other: Self) -> usize {
        (self.0.abs_diff(other.0) + self.1.abs_diff(other.1)).into()
    }

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

impl IndexMut<Dimension> for Point2 {
    fn index_mut(&mut self, index: Dimension) -> &mut Self::Output {
        use Dimension::*;

        match index {
            X => &mut self.0,
            Y => &mut self.1,
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

fn wall(s: Span) -> IResult<Span, ()> {
    char('#')(s).map(|(s, _)| (s, ()))
}

#[derive(Clone, Copy, Debug)]
struct Blizzard {
    loc: Point2,
    dir: Direction,
}

fn blizzard(s: Span) -> IResult<Span, Blizzard> {
    map_res(with_point(anychar), |(loc, ch)| {
        Direction::try_from(ch).map(|dir| Blizzard { loc, dir })
    })(s)
}

type MapState = HashMap<Point2, Vec<Direction>>;

#[derive(Debug)]
struct MapStates {
    state_at: RefCell<Vec<MapState>>,
    dims: MapSize,
}

impl MapStates {
    fn new(initial: MapState, dims: MapSize) -> Self {
        Self {
            state_at: vec![initial].into(),
            dims,
        }
    }

    fn generate_until(&self, n: usize) {
        let mut states = self.state_at.borrow_mut();

        for i in states.len()..=n {
            let next = advance_state(&states[i - 1], self.dims);

            states.push(next);
        }
    }

    fn get(&self, n: usize) -> Ref<MapState> {
        self.generate_until(n);

        Ref::map(self.state_at.borrow(), |v| &v[n])
    }
}

#[derive(Clone, Copy, Debug)]
struct MapSize(i16, i16);

impl MapSize {
    fn from_edge(Point2(x, y): Point2) -> Self {
        Self(x + 1, y + 1)
    }

    fn is_inside(self, Point2(x, y): Point2) -> bool {
        // walls are on every border of the square
        let Self(cols, rows) = self;

        (1..(cols - 1)).contains(&x) && (1..(rows - 1)).contains(&y)
    }
}

impl Index<Dimension> for MapSize {
    type Output = i16;

    fn index(&self, index: Dimension) -> &Self::Output {
        use Dimension::*;

        match index {
            X => &self.0,
            Y => &self.1,
        }
    }
}

fn rehome(p: Point2, map_sz: MapSize, direction: Direction) -> Point2 {
    let dim = direction.dim();

    let mut next = p + direction;

    // Columns `0` and `cols - 1` are walls
    // Rows `0` and `rows - 1` are walls too
    next[dim] = match next[dim] {
        0 => map_sz[dim] - 2,
        n if n == map_sz[dim] - 1 => 1,
        n => n,
    };

    next
}

#[derive(Clone, Copy, Debug)]
struct MapInfo {
    dims: MapSize,

    start: Point2,
    end: Point2,
}

struct Wall {
    hole: Point2,
    end: Point2,
}

fn pierced_wall(s: Span) -> IResult<Span, Wall> {
    let (r, hole) = map(
        delimited(many1(wall), with_point(char('.')), many1(wall)),
        |(p, _)| p,
    )(s)?;

    let end = get_point(r) - (1, 0);

    Ok((r, Wall { hole, end }))
}

fn single_line(s: Span) -> IResult<Span, Vec<Blizzard>> {
    #[derive(Clone)]
    enum LinePoint {
        Blizzard(Blizzard),
        Open,
    }

    map(
        delimited(
            wall,
            many1(alt((
                map(blizzard, LinePoint::Blizzard),
                value(LinePoint::Open, char('.')),
            ))),
            wall,
        ),
        |lps| {
            lps.into_iter()
                .filter_map(|lp| match lp {
                    LinePoint::Blizzard(bz) => Some(bz),
                    LinePoint::Open => None,
                })
                .collect()
        },
    )(s)
}

struct StartingMap(MapState, MapInfo);

fn starting_map(s: Span) -> IResult<Span, StartingMap> {
    map(
        tuple((
            pierced_wall,
            preceded(multispace1, many1(terminated(single_line, multispace1))),
            pierced_wall,
        )),
        |(st_w, lines, nd_w)| {
            StartingMap(
                lines
                    .into_iter()
                    .flatten()
                    .map(|Blizzard { loc, dir }| (loc, vec![dir]))
                    .collect(),
                MapInfo {
                    dims: MapSize::from_edge(nd_w.end),
                    start: st_w.hole,
                    end: nd_w.hole,
                },
            )
        },
    )(s)
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Dimension {
    X,
    Y,
}

#[allow(unused)]
fn dump_mapstate(ms: Ref<MapState>, info: &MapInfo) {
    let MapSize(cols, rows) = info.dims;
    let Point2(start_col, _) = info.start;
    let Point2(end_col, _) = info.end;

    println!(
        "{:#>pre$}S{:#>post$}",
        "",
        "",
        pre = start_col as usize,
        post = (cols - start_col - 1) as usize
    );

    for j in 1..(rows - 1) {
        print!("#");

        for i in 1..(cols - 1) {
            let p = (i, j).try_into().unwrap();

            if let Some(dirs) = ms.get(&p) {
                if let [dir] = dirs[..] {
                    print!("{dir}");
                } else {
                    print!("{}", dirs.len());
                }
            } else {
                print!(".");
            }
        }

        println!("#");
    }

    println!(
        "{:#>pre$}F{:#>post$}",
        "",
        "",
        pre = end_col as usize,
        post = (cols - end_col - 1) as usize
    );

    println!();
}

fn advance_state(ms: &MapState, bounds: MapSize) -> MapState {
    let mut next = MapState::new();

    for (&loc, dirs) in ms {
        use std::collections::hash_map::Entry::*;

        for &dir in dirs {
            let rehomed = rehome(loc, bounds, dir);

            match next.entry(rehomed) {
                Occupied(mut oe) => {
                    oe.get_mut().push(dir);
                }
                Vacant(ve) => {
                    ve.insert(vec![dir]);
                }
            }
        }
    }

    next
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PlayerState {
    pos: Point2,
    distance: usize,
    elapsed: usize,
}

impl PlayerState {
    fn start(start: Point2, distance: usize, elapsed: usize) -> Self {
        Self {
            pos: start,
            distance,
            elapsed,
        }
    }
}

impl Ord for PlayerState {
    fn cmp(&self, other: &Self) -> Ordering {
        // reverse in order to get the lowest nodes as greatest.
        // this turns the binary heap into a min-priority queue (pop() pops the lowest value)
        // this heuristic uses both distance and time as weights
        (self.distance + self.elapsed)
            .cmp(&(other.distance + other.elapsed))
            .reverse()
    }
}

impl PartialOrd for PlayerState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn neigh_of(ps: PlayerState, goal: Point2, ms: &MapStates) -> Vec<PlayerState> {
    // get the map for the next step
    let next_tick = ps.elapsed + 1;
    let next_state = ms.get(next_tick);

    let mut neighs: Vec<_> = ps
        .pos
        .neighbourhood()
        .into_iter()
        .filter(|&p| ms.dims.is_inside(p) && !next_state.contains_key(&p) || p == goal)
        .collect();

    if !next_state.contains_key(&ps.pos) {
        neighs.push(ps.pos);
    }

    neighs
        .into_iter()
        .map(|pos| PlayerState {
            pos,
            distance: pos.manhattan(goal),
            elapsed: next_tick,
        })
        .collect()
}

// After I don't know how many years, I finally actually implemented the
// Dijkstra's algorithm it seems
fn find_shortest_path(ms: &MapStates, [start, end]: [Point2; 2], elapsed: usize) -> Option<usize> {
    let start_state = PlayerState::start(start, start.manhattan(end), elapsed);

    let mut visited = HashSet::from([start_state]);
    let mut queue = BinaryHeap::from([start_state]);

    while let Some(node) = queue.pop() {
        if node.pos == end {
            return Some(node.elapsed);
        }

        for n in neigh_of(node, end, ms) {
            if visited.insert(n) {
                queue.push(n);
            }
        }
    }

    None
}

/// AoC problem for Dec 24 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let contents = fs::read_to_string(file)?;

    let StartingMap(lines, bounds @ MapInfo { start, end, dims }) =
        starting_map(Span::new(&contents))
            .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
            .1;

    let states = MapStates::new(lines, dims);

    println!("Map:");
    dump_mapstate(states.get(0), &bounds);

    let Some(first) = find_shortest_path(&states, [start, end], 0) else {
        return Err("no path found (first leg)".into());
    };

    let Some(second) = find_shortest_path(&states, [end, start], first) else {
        return Err("no path found (second leg)".into());
    };

    let Some(third) = find_shortest_path(&states, [start, end], second) else {
        return Err("no path found (third leg)".into());
    };

    println!(
        "first: {first}, second: {}, third: {}, sum : {}",
        second - first,
        third - second,
        third
    );

    Ok(())
}
