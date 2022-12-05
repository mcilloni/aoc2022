use std::{
    cmp::{max, min},
    collections::VecDeque,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    ops::ControlFlow,
    str::FromStr,
};

use clap::Parser;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while},
    character::complete::{anychar, char, space0},
    combinator::{eof, map, map_res, opt},
    error::{Error as NomError, ErrorKind as NomErrKind},
    multi::fold_many1,
    sequence::{delimited, preceded, tuple},
    IResult,
};
use num::Unsigned;

fn is_int_digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn unsigned<N: Unsigned + FromStr>(input: &str) -> IResult<&str, N> {
    map_res(take_while(is_int_digit), str::parse)(input)
}

fn uppercase(input: &str) -> IResult<&str, char> {
    match anychar(input)? {
        (rem, c @ 'A'..='Z') => Ok((rem, c)),
        _ => Err(nom::Err::Failure(NomError::new(input, NomErrKind::Tag))),
    }
}

struct Crate(char);

fn crate_entry(input: &str) -> IResult<&str, Crate> {
    let (input, c) = delimited(tag("["), uppercase, tag("]"))(input)?;

    Ok((input, Crate(c)))
}

struct Hole;

fn crate_hole(input: &str) -> IResult<&str, Hole> {
    map(tag("   "), |_| Hole)(input)
}

fn crate_or_hole(input: &str) -> IResult<&str, Option<Crate>> {
    alt((map(crate_entry, Some), map(crate_hole, |_| None)))(input)
}

fn crates_line(line: &str) -> IResult<&str, Vec<(usize, Crate)>> {
    let (rem, (pairs, _)) = fold_many1(
        preceded(opt(char(' ')), crate_or_hole),
        || (Vec::new(), 0usize),
        |(mut v, ix), el| {
            if let Some(cr) = el {
                v.push((ix, cr));
            }

            // after every hole or crate, add 1 to the index
            (v, ix + 1)
        },
    )(line)?;

    eof(rem)?;

    Ok(("", pairs))
}

struct Mov {
    count: u64,
    from: usize,
    to: usize,
}

fn ws<'a, O>(f: impl Fn(&'a str) -> IResult<&'a str, O>) -> impl FnMut(&'a str) -> IResult<&'a str, O> {
    delimited(space0, f, space0)
}

fn tag_ws<'a>(ts: &'a str) -> impl FnMut(&'a str) -> IResult<&'a str, &str> {
    ws(tag(ts))
}

fn mov_line(input: &str) -> IResult<&str, Mov> {
    let (rem, mov) = map(
        tuple((
            preceded(tag_ws("move"), unsigned),
            preceded(tag_ws("from"), unsigned),
            preceded(tag_ws("to"), unsigned),
        )),
        |(count, from, to)| Mov { count, from, to },
    )(input)?;

    eof(rem)?;

    Ok(("", mov))
}

fn get_two_mut<T>(s: &mut [T], i: usize, j: usize) -> (&mut T, &mut T) {
    assert!(i != j);

    let h = max(i, j);
    let l = min(i, j);

    // hack: Rust does not allow multiple mutable borrowings from the same slice, despite them being independent.
    // this function is a hack that uses split_at_mut to bypass this limitation
    let (lh, hh) = s.split_at_mut(h);

    let (lr, hr) = (&mut lh[l], &mut hh[0]);

    // return the elements in the order that was requested
    if i < j {
        (lr, hr)
    } else {
        (hr, lr)
    }
}

#[derive(Clone, Copy, Debug)]
enum MovPolicy {
    Single,
    Bulk,
}

#[derive(Default)]
struct Crates {
    stacks: Vec<VecDeque<Crate>>,
}

impl Crates {
    fn len(&self) -> usize {
        self.stacks.len()
    }

    fn mov(&mut self, Mov { count, from, to }: Mov, pol: MovPolicy) -> Result<u64, String> {
        if self.len() < from || from == 0 {
            return Err(format!("from '{from}' out of bounds"));
        }

        if self.len() < to || to == 0 {
            return Err(format!("to '{to}' out of bounds"));
        }

        if from == to {
            return Err(format!(
                "refusing to move {count} element from stack #{from} onto itself"
            ));
        }

        // hack: Rust does not allow multiple mutable borrowings from the same slice, despite them being independent.
        let (f, t) = get_two_mut(&mut self.stacks, from - 1, to - 1);

        use MovPolicy::*;

        match pol {
            Bulk => {
                let len = f.len();
                let count = count as usize;
                
                if len < count {
                    return Err(format!("stack #'{from}' has less than {count} elements"));
                }

                t.extend(f.drain(f.len() - count..))
            }
            Single => for _ in 0..count {
                let Some(cr) = f.pop_back() else {
                    return Err(format!("stack #'{from}' has less than {count} elements"));
                };

                t.push_back(cr);
            }
    }

        Ok(count)
    }

    fn push_at(&mut self, ix: usize, cr: Crate) {
        let wanted_len = ix + 1;

        if self.stacks.len() < wanted_len {
            self.stacks.resize_with(ix + 1, Default::default);
        }

        // Rust's deque is just a vector and it's implemented as a ring buffer instead of a vec of vecs - this is
        // better than C++'s deques, which are often vectors of vectors and thus often useless on modern CPUs
        self.stacks[ix].push_front(cr);
    }

    fn skim(&self) -> Option<String> {
        self.stacks.iter().try_fold(String::new(), |mut acc, el| {
            if let Some(&Crate(letter)) = el.back() {
                acc.push(letter);
                Some(acc)
            } else {
                None
            }
        })
    }
}

impl Extend<(usize, Crate)> for Crates {
    fn extend<T: IntoIterator<Item = (usize, Crate)>>(&mut self, iter: T) {
        for (ix, cr) in iter {
            self.push_at(ix, cr)
        }
    }
}

/// AoC problem for Dec 05 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
   /// Use multiple algoritm, as specified by part 2
   #[arg(short, long, default_value_t = false)]
   multiple: bool,

   /// File to parse
   file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { multiple, file } = Args::parse();

    let pol = if multiple { MovPolicy::Bulk } else { MovPolicy::Single };

    let mut lines = BufReader::new(File::open(file)?).lines().enumerate();

    let mut crates = Crates::default();

    loop {
        let line = match lines.next() {
            Some((_, Ok(line))) => line,
            Some((_, Err(err))) => return Err(err.into()),
            None => return Err("unexpected EOF".into()),
        };

        let (_, cf) = alt((
            map(crates_line, ControlFlow::Continue),
            map(
                fold_many1(ws(unsigned), u64::default, max),
                ControlFlow::Break,
            ),
        ))(&line)
        .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?;

        match cf {
            ControlFlow::Break(n) => {
                if n != crates.len() as u64 {
                    return Err(format!("wrong number of stacks: {n} != {}", crates.len()).into());
                } else {
                    break;
                }
            }
            ControlFlow::Continue(pairs) => crates.extend(pairs),
        }
    }

    // expect empty line

    match lines.next().map(|(_, line_res) | line_res) {
        Some(Ok(line)) if line.trim().len() == 0 => {},
        Some(Ok(line)) => return Err(format!("malformed line '{line}'").into()),
        Some(Err(e)) => return Err(e.into()),
        None => return Err("unexpected EOF".into()),
    }

    // parse commands
    for (lineno, line_res) in lines {
        let line = line_res?;

        let (_, mov) =
            mov_line(&line).map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?;

        if let Err(e) = crates.mov(mov, pol) {
            eprintln!("error: failed at line {}", lineno + 1);

            return Err(e.into());
        }
    }

    if let Some(sk) = crates.skim() {
        println!("res = {sk}");
    } else {
        return Err("some stack has holes".into());
    }

    Ok(())
}
