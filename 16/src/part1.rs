use std::{
    cmp::{max, min},
    collections::{btree_map::Entry, BTreeMap, VecDeque},
    error::Error,
    fs::{self},
    str::FromStr,
};

use clap::Parser;
use itertools::Itertools;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
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
    F: FnMut(&'a str) -> IResult<&'a str, O, E>,
{
    delimited(space0, inner, space0)
}

fn uppercase1(input: &str) -> IResult<&str, &str> {
    take_while1(|c| ('A'..='Z').contains(&c))(input)
}

#[derive(Debug)]
struct Valve {
    fr: usize,
    to: Vec<String>,
}

impl Valve {
    fn new(fr: usize, to: Vec<String>) -> Self {
        Self { fr, to }
    }
}

fn valve(input: &str) -> IResult<&str, (String, Valve)> {
    map(
        tuple((
            tag("Valve"),
            ws(map(uppercase1, String::from)),
            tag("has flow rate"),
            ws(char('=')),
            unsigned,
            ws(char(';')),
            ws(alt((tag("tunnel leads"), tag("tunnels lead")))),
            tag("to"),
            ws(alt((tag("valves"), tag("valve")))),
            ws(separated_list0(
                ws(char(',')),
                map(uppercase1, String::from),
            )),
        )),
        |(_, name, _, _, fr, _, _, _, _, to)| (name, Valve::new(fr, to)),
    )(input)
}

fn valves(input: &str) -> IResult<&str, Valves> {
    all_consuming(delimited(
        multispace0,
        map(separated_list0(multispace0, valve), Valves::from),
        multispace0,
    ))(input)
}

const MAX_TIME: usize = 30;

struct Valves {
    valves: BTreeMap<String, usize>,
    elems: Vec<Valve>,
    distances: Vec<Vec<usize>>,
}

impl Valves {
    fn dump(&self) {
        let name_of = |j| self.valves.iter().find(|(_, n)| **n == j).unwrap().0;

        for (name, &i) in &self.valves {
            let valve = &self.elems[i];
            let dv = &self.distances[i];

            println!(
                r#""{}"({}): [{}] => {{{}}}"#,
                name,
                valve.fr,
                valve.to.join(", "),
                dv.iter()
                    .enumerate()
                    .map(|(a, b)| format!("{}: {b}", name_of(a)))
                    .join(", ")
            );
        }
    }

    fn best_flow(&self) -> usize {
        // poor people's BFS, thanks to people that know graph theory better than me for directing me towards a solid
        // solution
        let mut results = BTreeMap::new();

        let &a_id = self.valves.get("AA").expect("no AA in list");

        let mut queue = VecDeque::from([Choice::start(a_id)]);

        loop {
            let Some(cur) = queue.pop_front() else { break; };

            for (i, &d) in self.distances[cur.el].iter().enumerate() {
                let flow = self.elems[i].fr;

                // previous expense + distance traveled + time spent opening this valve
                let paid = (cur.paid + 1).saturating_add(d);
                let left = MAX_TIME.saturating_sub(paid);

                if cur.open & (1 << i) == 0 && flow > 0 && left > 0 {
                    let open = cur.open | (1 << i);
                    let gained = cur.gained + left * flow;

                    match results.entry(open) {
                        Entry::Vacant(ve) => {
                            ve.insert(gained);
                        }
                        Entry::Occupied(mut oe) => {
                            let old = *oe.get();

                            *oe.get_mut() = max(old, gained);
                        }
                    }

                    // explore the next node
                    queue.push_back(Choice {
                        el: i,
                        gained,
                        paid,
                        open,
                    });
                }
            }
        }

        let (open, best) = results
            .into_iter()
            .fold(None, |acc: Option<(usize, usize)>, (open, gained)| {
                if let Some(acc) = acc {
                    if acc.1 < gained {
                        Some((open, gained))
                    } else {
                        Some(acc)
                    }
                } else {
                    Some((open, gained))
                }
            })
            .expect("no results found");

        for i in 0..self.elems.len() {
            if open & (1 << i) != 0 {
                print!(
                    "{} ",
                    self.valves.iter().find(|(_n, k)| **k == i).unwrap().0
                );
            }
        }

        println!();

        best
    }
}

impl From<Vec<(String, Valve)>> for Valves {
    fn from(source: Vec<(String, Valve)>) -> Self {
        let mut valves = BTreeMap::new();
        let mut elems = Vec::new();

        for (i, (name, valve)) in source.into_iter().enumerate() {
            valves.insert(name, i);
            elems.push(valve);
        }

        let mut distances = vec![vec![usize::MAX; valves.len()]; valves.len()];

        for (n, Valve { to, .. }) in elems.iter().enumerate() {
            let dv = &mut distances[n];

            dv[n] = 0; // distance to self is 0
            for neigh in to {
                let j = valves[neigh];

                dv[j] = 1; //distance to other nodes is 1
            }
        }

        // Floyd–Warshall to compute distances

        for k in 0..valves.len() {
            for i in 0..valves.len() {
                for j in 0..valves.len() {
                    distances[i][j] = min(
                        distances[i][j],
                        distances[i][k].saturating_add(distances[k][j]),
                    );
                }
            }
        }

        Self {
            valves,
            elems,
            distances,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct Choice {
    el: usize,
    paid: usize,
    gained: usize,
    open: usize,
}

impl Choice {
    fn start(el: usize) -> Self {
        Self {
            el,
            ..Default::default()
        }
    }
}

/// AoC problem for Dec 16 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let vmap = valves(&fs::read_to_string(file)?)
        .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
        .1;

    vmap.dump();

    println!("best = {}", vmap.best_flow());

    Ok(())
}
