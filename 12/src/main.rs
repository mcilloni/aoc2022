use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader, Read},
    iter::{once, repeat},
};

use ansi_term::{Colour, Style};
use clap::Parser;
use pathfinding::prelude::{astar, Matrix};

fn manhattan(p1: (usize, usize), p2: (usize, usize)) -> usize {
    p1.0.abs_diff(p2.0) + p1.1.abs_diff(p2.1)
}

#[derive(Debug)]
struct Trail {
    start: (usize, usize),
    path: Vec<(usize, usize)>,
    steps: usize,
}

#[derive(Debug)]
struct HeightMap {
    points: Matrix<u8>,
}

impl HeightMap {
    fn find_trail(&self, start: (usize, usize), end: (usize, usize)) -> Option<Trail> {
        astar(
            &start,
            |p| {
                let neigh = self.neighbours(*p);

                //println!("neigh({p:?}) = {:?}", &neigh as &[_]);

                neigh.into_iter().map(|p| (p, 1))
            },
            |p| manhattan(*p, end),
            |p| *p == end,
        )
        .map(|(path, steps)| Trail { start, path, steps })
    }

    fn neighbours(&self, p: (usize, usize)) -> Vec<(usize, usize)> {
        self.points
            .get(p)
            .map(|&ch| {
                self.points
                    .neighbours(p, false)
                    .filter(move |&neigh| self.points.get(neigh).cloned().unwrap() <= ch + 1)
            })
            .map(Vec::from_iter)
            .unwrap_or_default()
    }
}

fn dump_hmap(hmap: &HeightMap, (start, end): ((usize, usize), (usize, usize))) {
    for (i, row) in hmap.points.iter().enumerate() {
        for (j, &height) in row.iter().enumerate() {
            fn compute_shade(lvl: u8) -> u8 {
                const SLOPE: f64 = u8::MAX as f64 / 26.0f64;

                (SLOPE * lvl as f64) as u8
            }

            let is_start = (i, j) == start;
            let is_end = (i, j) == end;

            let (c, fg, bg) = if is_start {
                ('S', Colour::White, Colour::Green)
            } else if is_end {
                ('E', Colour::Black, Colour::Red)
            } else {
                let lvl = compute_shade(height);

                let fg = if lvl < 128 {
                    Colour::White
                } else {
                    Colour::Black
                };

                let bg = Colour::RGB(lvl, lvl, lvl);

                (char::from(height + b'a'), fg, bg)
            };

            print!("{}", Style::new().on(bg).fg(fg).paint(c.to_string()));
        }

        println!();
    }

    println!();
}

fn dump_path(hmap: &HeightMap, path: &[(usize, usize)]) {
    let mut p: Vec<char> = repeat('.').take(hmap.points.len()).collect();

    for w in path.windows(2) {
        let (this, next) = (w[0], w[1]);

        let ch = match (this, next) {
            ((a, b), (x, y)) if a == x && b < y => '>',
            ((a, b), (x, y)) if a == x && b > y => '<',
            ((a, b), (x, y)) if a < x && b == y => 'v',
            ((a, b), (x, y)) if a > x && b == y => '^',
            _ => unreachable!(),
        };

        p[this.0 * hmap.points.rows + this.1] = ch;
    }

    for (i, c) in p.into_iter().enumerate() {
        if i > 0 && i % hmap.points.rows == 0 {
            println!();
        }

        print!("{}", c);
    }

    println!();
}

fn load_heightmap<R: Read>(
    r: R,
) -> Result<(HeightMap, ((usize, usize), (usize, usize))), Box<dyn Error>> {
    let mut some_start = None;
    let mut some_end = None;

    let lines = BufReader::new(r).lines();
    let mut some_matrix: Option<Matrix<u8>> = None;

    for (i, line_res) in lines.enumerate() {
        let new_row = line_res?
            .trim()
            .chars()
            .enumerate()
            .map(|(j, c)| {
                match c {
                    'a'..='z' => Ok(c),
                    'E' => {
                        some_end = Some((i, j));

                        Ok('z')
                    }
                    'S' => {
                        some_start = Some((i, j)); // always starts at 'a'

                        Ok('a')
                    }
                    c => Err(format!("invalid charater '{c}'").into()),
                }
                .map(|c| c as u8 - b'a')
            })
            .collect::<Result<Vec<u8>, Box<dyn Error>>>()?;

        if let Some(matrix) = &mut some_matrix {
            matrix.extend(&new_row)?;
        } else {
            some_matrix = Some(once(new_row.into_iter()).collect());
        }
    }

    let hm = HeightMap {
        points: some_matrix.ok_or("empty file")?,
    };

    match (some_start, some_end) {
        (Some(start), Some(end)) => Ok((hm, (start, end))),
        _ => Err("missing start or end".into()),
    }
}

/// AoC problem for Dec 12 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let (hmap, (start, end)) = File::open(file)
        .map_err(Into::into)
        .and_then(load_heightmap)?;

    dump_hmap(&hmap, (start, end));

    let Some(trail) = hmap.find_trail(
        start, end,
    ) else {
        return Err("failed to find a path".into());
    };

    println!(
        "found path of len = {}, cost = {}",
        trail.path.len(),
        trail.steps
    );

    dump_path(&hmap, &trail.path);

    let (best_start, best_steps) = hmap
        .points
        .items()
        .filter_map(|(p, val)| (*val == 0 && p != start).then_some(p))
        .fold(
            (start, trail.steps),
            |(best_start, best_steps), start| match hmap.find_trail(start, end) {
                Some(trail) if trail.steps < best_steps => (trail.start, trail.steps),
                _ => (best_start, best_steps),
            },
        );

    println!("best path starts from {best_start:?}, steps = {best_steps}");

    Ok(())
}
