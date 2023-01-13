use std::{
    cmp::max,
    collections::HashMap,
    error::Error,
    fmt::Display,
    fs,
    iter::zip,
    num::TryFromIntError,
    ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Range, Sub},
    str::FromStr,
};

use clap::Parser;
use itertools::Itertools;
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
use strum::{EnumCount, EnumIter, IntoEnumIterator};

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
#[derive(Clone, Copy, Debug, EnumCount, EnumIter, FromPrimitive)]
enum Direction {
    East = 0,
    South = 1,
    West = 2,
    North = 3,
}

impl Direction {
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

impl TryFrom<Point2> for Direction {
    type Error = String;

    fn try_from(p @ Point2(x, y): Point2) -> Result<Self, Self::Error> {
        let trunc = Point2(x.signum(), y.signum());

        Direction::iter()
            .find(|d| d.versor() == trunc)
            .ok_or(format!("no valid Direction conversion for {p}"))
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
        *self = *self + rhs
    }
}

impl AddAssign<Direction> for Point2 {
    fn add_assign(&mut self, rhs: Direction) {
        *self += rhs.versor()
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

impl Display for Point2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self(x, y) = self;

        write!(f, "({x}, {y})")
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

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct Point3(i16, i16, i16);

impl Display for Point3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self(x, y, z) = self;

        write!(f, "({x}, {y}, {z})")
    }
}

impl From<(i16, i16, i16)> for Point3 {
    fn from((x, y, z): (i16, i16, i16)) -> Self {
        Self(x, y, z)
    }
}

impl From<[i16; 3]> for Point3 {
    fn from([x, y, z]: [i16; 3]) -> Self {
        Self(x, y, z)
    }
}

impl From<Point3> for (i16, i16) {
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

impl Add<(i16, i16, i16)> for Point3 {
    type Output = Point3;

    fn add(self, rhs: (i16, i16, i16)) -> Self::Output {
        self + Self::from(rhs)
    }
}

impl Add<[i16; 3]> for Point3 {
    type Output = Point3;

    fn add(self, rhs: [i16; 3]) -> Self::Output {
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

impl Sub for Point3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0, self.1 - rhs.1, self.2 - rhs.2)
    }
}

impl Mul<i16> for Point3 {
    type Output = Point3;

    fn mul(self, rhs: i16) -> Self::Output {
        let Self(x, y, z) = self;

        x.checked_mul(rhs)
            .zip(y.checked_mul(rhs))
            .zip(z.checked_mul(rhs))
            .map(|((x, y), z)| Self(x, y, z))
            .expect("integer overflow")
    }
}

impl MulAssign<i16> for Point3 {
    fn mul_assign(&mut self, rhs: i16) {
        *self = *self * rhs;
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct UVBound(Point2, Point3);

impl Display for UVBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self(p2, p3) = *self;

        write!(f, "{p2} -> {p3}")
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

#[derive(Clone, Copy, Debug)]
struct Row<'a> {
    grid: &'a Grid,
    ix: usize,
}

impl Row<'_> {
    fn iter(
        &self,
    ) -> impl ExactSizeIterator<Item = Material> + DoubleEndedIterator<Item = Material> + '_ {
        let (dim_w, _) = self.grid.dims();

        (0..dim_w).map(move |ix| self[ix])
    }

    fn start(&self) -> Option<Point2> {
        self.iter()
            .enumerate()
            .find(|&(_, m)| m != Material::Outside)
            .map(|(n, _)| {
                (n, self.ix)
                    .try_into()
                    .expect("this should never overflow an i16")
            })
    }
}

impl Index<usize> for Row<'_> {
    type Output = Material;

    fn index(&self, a: usize) -> &Self::Output {
        &self.grid[(a, self.ix)]
    }
}

#[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
struct FaceBounds {
    cols: Range<i16>,
    rows: Range<i16>,
}

impl FaceBounds {
    fn contains(&self, Point2(ref x, ref y): Point2) -> bool {
        self.cols.contains(x) && self.rows.contains(y)
    }

    fn corners(&self) -> [Point2; 4] {
        let FaceBounds { cols, rows } = self;

        [
            Point2(cols.end - 1, rows.start),   // East
            Point2(cols.end - 1, rows.end - 1), // South
            Point2(cols.start, rows.end - 1),   // West
            Point2(cols.start, rows.start),     // North
        ]
    }

    fn find_edge_direction(&self, [p1, p2]: [Point2; 2]) -> Direction {
        // given two points, assert they are two edge points and find the direction towards the inside of the square
        let corners = self.corners();

        for i in 0..corners.len() {
            let nxt = (i + 1) % corners.len();

            let (c, n) = (corners[i], corners[nxt]);

            let first = if (p1, p2) == (c, n) || (p2, p1) == (c, n) {
                i
            } else {
                continue;
            };

            return Direction::from_usize(first)
                .expect("square has too many corners")
                .opposite();
        }

        panic!("no corner found in set [{p1}, {p2}]");
    }

    fn rehome_edgepoint_with_sidemap(
        &self,
        cur: Point2,
        [p1, p2]: [(Point2, Point2); 2],
    ) -> Point2 {
        let (mut start, mut end) = (p1, p2);

        // sort the side correctly
        for corner in self.corners() {
            if start.0 == corner {
                break;
            } else if start.1 == corner {
                (start, end) = (p2, p1);
                break;
            }
        }

        let ((s1, s2), (_, e2)) = (start, end);

        let delta = if cur.0 == s1.0 {
            cur.1 - s1.1
        } else if cur.1 == s1.1 {
            cur.0 - s1.0
        } else {
            panic!("not on an edge")
        }
        .abs();

        match e2 - s2 {
            Point2(d, 0) => s2 + if d > 0 { (delta, 0) } else { (-delta, 0) },
            Point2(0, d) => s2 + if d > 0 { (0, delta) } else { (0, -delta) },
            _ => panic!("broken vertices - not on same axis"),
        }
    }
}

struct FaceEdge {
    vertices: [Point3; 4],
    neighs: [FacePos; 4],
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, EnumCount, Eq, Hash, PartialEq)]
/*
 *     .+------+
 *   .' | T  .'|   <- K (back)
 *  +---+--+'  |
 *  | L |  |   |   <- R
 *  |  ,+--+---+
 *  |.' F  | .' <- B (bottom)
 *  +------+'
 */
enum FacePos {
    #[default]
    Front,
    Bottom,
    Left,
    Right,
    Top,
    Back,
}

static VERTICES: [[[i16; 3]; 4]; 6] = [
    [[1, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0]], // Front
    [[1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]], // Bottom
    [[0, 1, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1]], // Left
    [[1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]], // Right
    [[1, 1, 1], [1, 1, 0], [0, 1, 0], [0, 1, 1]], // Top
    [[1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 1]], // Back
];

impl FacePos {
    fn edge(self) -> FaceEdge {
        FaceEdge {
            vertices: self.vertices(),
            neighs: self.neigh(),
        }
    }

    fn edge_with(self, face: Self, at: Direction) -> FaceEdge {
        let mut neighs = self.neigh();
        let mut vertices = self.vertices();

        let ix = at as usize;
        let mut skew = 0usize;

        while neighs[ix] != face {
            neighs.rotate_right(1);
            skew += 1;
        }

        vertices.rotate_right(skew);

        FaceEdge { neighs, vertices }
    }

    fn neigh(self) -> [Self; 4] {
        use FacePos::*;

        // these are the ORIENTED neighbouring faces for every face of the cube
        // in the [E, S, W, N] orientation
        match self {
            Front => [Right, Bottom, Left, Top],
            Bottom => [Right, Back, Left, Front],
            Left => [Front, Bottom, Back, Top],
            Right => [Back, Bottom, Front, Top],
            Top => [Right, Front, Left, Back],
            Back => [Right, Top, Left, Bottom],
        }
    }

    fn vertices(self) -> [Point3; 4] {
        let mut ret = [Point3::default(); 4];

        for (v, dp) in zip(VERTICES[self as usize], &mut ret) {
            *dp = Point3::from(v);
        }

        ret
    }
}

impl Display for FacePos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use FacePos::*;

        match self {
            Front => 'F',
            Bottom => 'B',
            Left => 'L',
            Right => 'R',
            Top => 'T',
            Back => 'K',
        }
        .fmt(f)
    }
}

fn find_faces(g: &Grid) -> [FaceBounds; 6] {
    // find the size of the faces

    let face_len = g
        .rows()
        .map(|r| r.iter().filter(|&m| m != Material::Outside).count())
        .min()
        .expect("can't be zero") as i16;

    const EMPTY_RANGES: FaceBounds = FaceBounds {
        cols: 0..0,
        rows: 0..0,
    };

    let mut ret = [EMPTY_RANGES; 6];

    // find all faces
    let mut cur = g.row(0).start().unwrap();

    for bound in &mut ret {
        while g.is_outside(cur) {
            let ncord = (cur.1 + face_len) as usize;

            cur = g.row(ncord).start().expect("unexpected out of bounds");
        }

        assert!(g.is_inside(cur));

        *bound = FaceBounds {
            cols: cur.0..cur.0 + face_len,
            rows: cur.1..cur.1 + face_len,
        };

        cur += Point2(face_len, 0);
    }

    ret
}

#[derive(Debug)]
struct FaceTree {
    id: usize,
    pos: FacePos,

    to: [Option<Box<FaceTree>>; 4],
    vertices: [UVBound; 4],
    grid_neighs: [FacePos; 4],
}

impl FaceTree {
    fn children(&self) -> impl Iterator<Item = &FaceTree> {
        self.to.iter().filter_map(|some_ft| some_ft.as_deref())
    }
}

fn build_ftree(faces: &[FaceBounds]) -> FaceTree {
    use FacePos::*;

    let (cur, rest) = faces.split_first().expect("a non-empty slice");

    let side = cur.cols.len();

    let rest = HashMap::from_iter(rest.iter().enumerate().map(|(i, f)| (i + 1, f)));

    struct State<'a> {
        cur: &'a FaceBounds,
        id: usize,
        pos: FacePos,
        from: Option<(FacePos, Direction)>,
        rest: HashMap<usize, &'a FaceBounds>,
    }

    // TL;DR: this is just DFS with extra steps
    // this neat little function:
    // - recursively probes the corners of a face in order to find the neibouring ones (the ones containing the extended
    //   vertex) by extending them in the 4 directions
    //
    //     *
    //     .
    //     +------+.*
    //     |      |
    //     |      |
    //     |      |
    //   *.+------+
    //            ˙
    //            *
    //
    //   and then checking for intersections.
    // - tags the faces, guessing their orientation using the direction of the "source face" as a hint. Every face has a "canonical"
    //   orientation, with neighbours in the [E, S, W, N] directions (assuming the cube is rotated with the face in front of the observer)
    //   The "(face_from, from_dir)" tuple is used to rotate the cube in order to align its neighbour vector with its current orientation
    //   in the UV space
    fn map_recurse(
        State {
            cur,
            id,
            pos,
            from,
            mut rest,
        }: State,
        side: i16,
    ) -> (FaceTree, HashMap<usize, &FaceBounds>) {
        const NONE: Option<Box<FaceTree>> = None;

        let mut to = [NONE; 4]; // sigh, this is a very dumb limitation

        let FaceEdge {
            neighs,
            mut vertices,
            ..
        } = match from {
            // get neighbours using the node we came from as a reference to determine the current orientation
            Some((old_fp, old_dir)) => pos.edge_with(old_fp, old_dir.opposite()),
            None => pos.edge(),
        };

        // fix up vertices by multiplying every dimension with a constant
        for v in vertices.iter_mut() {
            *v *= side - 1; // cells, not actual points
        }

        // extend every corner of the square of a single point in every direction
        for (i, &v) in cur.corners().iter().enumerate() {
            let dir = Direction::from_usize(i).unwrap();

            let probe = v + dir;

            if let Some((&n, &cf)) = rest.iter().find(|(_, f)| f.contains(probe)) {
                rest.remove(&n);

                // get the position of the next face using the oriented neighbours vector [E, S, W, N]
                let npos = neighs[dir as usize];

                let (ft, rem) = map_recurse(
                    State {
                        cur: cf,
                        id: n,
                        pos: npos,
                        from: Some((pos, dir)),
                        rest,
                    },
                    side,
                );

                to[i] = Some(Box::new(ft));
                rest = rem;
            }
        }

        let vertices = {
            let mut uvlist = [UVBound::default(); 4];

            for ((p2, p3), db) in zip(zip(cur.corners(), vertices), &mut uvlist) {
                *db = UVBound(p2, p3);
            }

            uvlist
        };

        (
            FaceTree {
                id,
                pos,
                to,
                vertices,
                grid_neighs: neighs,
            },
            rest,
        )
    }

    map_recurse(
        State {
            cur,
            id: 0,
            pos: Front,
            from: None,
            rest,
        },
        side as i16,
    )
    .0
}

fn dump_ftree(ft: &FaceTree) {
    fn dump_ftree_pad(
        FaceTree {
            id,
            pos,
            to,
            vertices,
            grid_neighs,
        }: &FaceTree,
        pad: usize,
    ) {
        println!(
            "#{id} <{pos:?}, {{{grid_neighs:?}}}> {}: ",
            vertices.iter().format(", ")
        );

        for (n, some_ft) in to.iter().enumerate() {
            let dir = Direction::from_usize(n).unwrap();

            if let Some(ft) = some_ft {
                print!("{:pad$}@{dir:?}: ", "", pad = pad);

                dump_ftree_pad(ft, pad + 4);
            }
        }
    }

    dump_ftree_pad(ft, 4)
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

    fn row(&self, ix: usize) -> Row<'_> {
        Row { grid: self, ix }
    }

    fn rows(&self) -> impl Iterator<Item = Row<'_>> {
        let (_, dim_y) = self.dims();

        (0..dim_y).map(move |ix| self.row(ix))
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

// execute commands, one at a time, and return the final position and a map of traversed positions and directions
fn execute(cbd: &CubeProjection, cmds: &[Command]) -> (Pos, HashMap<Point2, Direction>) {
    let mut cur = spawn_in(cbd.grid()).expect("grid is degenerate: no open spots");

    let mut traversed = HashMap::from_iter([(cur.pos, cur.dir)]);
    let mut cur_face = FacePos::Front; // start from the front

    // compute the bounds.
    let bounds = cbd.mapped_bounds();
    let mut cur_neighs = bounds.grid_adjacencies(cur_face);

    for cmd in cmds {
        cur = match *cmd {
            Command::Forward(count) => {
                let Pos { mut pos, mut dir } = cur;

                assert!(cbd.grid().is_inside(pos));

                // attempt to advance `count` steps in a certain direction.
                // due to the fact this is a cube, it may cause the point to warp on another face and change direction
                for _ in 0..count {
                    traversed.insert(pos, dir);

                    let mut next = pos + dir;

                    // assume we don't have to warp on another face
                    let mut next_dir = dir;
                    let mut next_face = cur_face;

                    let cur_face_bounds = bounds.sides(cur_face);

                    if !cur_face_bounds.contains(next) {
                        // we are outside the face, roll over using the cube faces as a guide

                        // find the next face
                        next_face = cur_neighs[dir as usize];

                        // match the vertices of the edge we just got out from with the new face using their 3D coords
                        // this code is quite tricky, because it attempts to find the two 3D vertices in common between
                        // the next and the current face (we computed UV mappings during the tagging phase)
                        let common_vertices: Vec<_> = bounds
                            .vertices(cur_face)
                            .iter()
                            .chain(bounds.vertices(next_face).iter())
                            .combinations(2) // generate all possible combinations
                            .filter(|v| v[0].1 == v[1].1) // keep pairs that match in the 3D space
                            .map(|v| (v[0].0, v[1].0)) // keep only the 2D points
                            .collect();

                        // assert: we must only find 2 pairs of matching vertices, otherwise either UV mapping is broken
                        // or current == next (which should never happen)
                        let &[p1, p2] = &common_vertices[..] else {
                            panic!("wrong array length: {}", common_vertices.len())
                        };

                        // compute the new position in the face we just accessed
                        next = cur_face_bounds.rehome_edgepoint_with_sidemap(pos, [p1, p2]);

                        // compute the direction towards the centre of the new face (which is the direction we want to
                        // use from next loop)
                        next_dir = bounds.sides(next_face).find_edge_direction([p1.1, p2.1]);
                    }

                    if cbd.grid()[next] == Material::Wall {
                        // if we hit a wall, go back and undo this last step

                        break;
                    }

                    // update the neighbours array after changing faces
                    if next_face != cur_face {
                        cur_neighs = bounds.grid_adjacencies(next_face);
                    }

                    // commit the current status
                    pos = next;
                    dir = next_dir;
                    cur_face = next_face;
                }

                Pos { pos, dir }
            }
            Command::Turn(rot) => cur.rotate(rot), // rotations are refreshingly easy
        };
    }

    (cur, traversed)
}

fn spawn_in(g: &Grid) -> Option<Pos> {
    // finds the first NW non-outside point on the grid
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

#[derive(Debug, Default)]
struct MappedBounds {
    // an array of (boundaries, [uv-mapped vertices]) for every face of the cube
    bounds: [(FaceBounds, [UVBound; 4]); 6],

    // face adjacencies for every face of the cube, in an ESWN orientation
    grid_adj: [[FacePos; 4]; 6],
}

impl MappedBounds {
    const fn grid_adjacencies(&self, index: FacePos) -> &[FacePos; 4] {
        &self.grid_adj[index as usize]
    }

    const fn sides(&self, index: FacePos) -> &FaceBounds {
        &self.bounds[index as usize].0
    }

    const fn vertices(&self, index: FacePos) -> &[UVBound; 4] {
        &self.bounds[index as usize].1
    }
}

impl Index<FacePos> for MappedBounds {
    type Output = (FaceBounds, [UVBound; 4]);

    fn index(&self, index: FacePos) -> &Self::Output {
        let ix = index as usize;

        &self.bounds[ix]
    }
}

struct CubeProjection {
    uv: Grid,
    faces: [FaceBounds; 6],
    ftree: FaceTree,
}

impl CubeProjection {
    fn new(uv: Grid) -> Self {
        let faces = find_faces(&uv);
        let ftree = build_ftree(&faces);

        Self { uv, faces, ftree }
    }

    const fn grid(&self) -> &Grid {
        &self.uv
    }

    fn mapped_bounds(&self) -> MappedBounds {
        // traverse the tree and collect the info about the boundaries of every face
        // in more manageable structures (arrays)
        let mut ret = MappedBounds::default();

        let mut fnodes = vec![&self.ftree];

        while let Some(fnode) = fnodes.pop() {
            fnodes.extend(fnode.children());

            let id = fnode.pos as usize;

            ret.bounds[id] = (self.faces[fnode.id].clone(), fnode.vertices);
            ret.grid_adj[id] = fnode.grid_neighs;
        }

        ret
    }
}

fn dump_faces(cbd: &CubeProjection) {
    dump_ftree(&cbd.ftree);

    let (dim_x, dim_y) = cbd.grid().dims();

    let fpos: Vec<_> = {
        let mut faces = vec![];

        let mut stack = vec![&cbd.ftree];

        while let Some(cur) = stack.pop() {
            faces.push((cur.id, (cur.pos, &cbd.faces[cur.id])));

            stack.extend(cur.children());
        }

        faces
            .into_iter()
            .sorted_by_key(|(n, _)| *n)
            .map(|(_, e)| e)
            .collect()
    };

    for j in 0..dim_y {
        for i in 0..dim_x {
            let cur = Point2(i as i16, j as i16);

            if let Some((_, (pos, _))) = fpos.iter().enumerate().find(|(_, (_, f))| f.contains(cur))
            {
                print!("{pos}");
            } else {
                print!(" ");
            }
        }

        println!();
    }

    println!();
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

    // First step:
    // Map the entire input into a grid of Materials, where 'Outside' is the space outside the map.
    // The map will be as wide as the easternmost concrete point is.
    // Commands will also be parsed as-is into a vector.
    let (g, cmds) = map_input(&fs::read_to_string(file)?)
        .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
        .1;

    // Second step:
    // Interpolate the grid as an exploded cube.
    // 1. The grid is probed in order to find the side length of the cube, which is assumed to be equal to the narrowest
    //    stretch of non-empty materials. This always finds a solution in every possible "straight" geometry, in every possible exploded
    //    cube;
    // 2. After probing the side of the cube as being of size N, the face probing algorithm extracts areas of size NxN
    //    from the grid in an iterative fashion, starting from the source point and then moving east and southwards.
    // 3. After deducing the face boundaries as 2D coordinates, the first face is tagged as "Front" and a tree of
    //    faces is generated by detecting adjacencies and deducing how the face is rotated in the UV space compared to the
    //    3D faces of the cube. The vertices of every face are then mapped to their 3D coordinates.
    let cbd = CubeProjection::new(g);

    // dump faces for easier understanding of what we've done so far
    dump_faces(&cbd);

    // Third step:
    // Actually execute the command list we obtained before, one step at a time, taking care to correctly warp the cursor
    // along the faces of the cube when needed.
    // 1. The cursor is spawned at the northernmost point coming from west. The cursor is facing East;
    // 2. Commands are then executed sequentially. Rotations are trivial for obvious reasons;
    // 3. When the cursor exits face boundaries:
    //    a. The rehoming algorithm deduces the next face using the previously found adjacencies;
    //    b. The corners of the two faces are then compared to find which 2D corners match up in the 3D space;
    //    c. The point is transposed from one edge to the other in 2D space. This is not done using the 3D coordinates
    //       due to the fact I've better stuff to do than devise how to do bidirectional UV mapping for AoC;
    //    d. The direction is fixed to correctly point towards the inside of the next face.
    let (cur, traversed) = execute(&cbd, &cmds);

    // dump steps in a 2D material map. In the full input paths will inevitably cross, there's nothing that can be done
    // about that.
    dump_map(cbd.grid(), &traversed);

    println!("{cur:?}, eval = {}", cur.eval());

    Ok(())
}
