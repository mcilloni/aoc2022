use std::{
    collections::{BTreeMap, HashSet},
    error::Error,
    fs::{self},
    iter::{once, zip},
    ops::Index,
    str::FromStr,
};

use clap::Parser;
use itertools::chain;
use nom::{
    bytes::complete::{tag, take_while},
    character::complete::{alpha1, char, multispace0, multispace1, space1},
    combinator::{all_consuming, map, map_res},
    error::{Error as NomError, ParseError},
    multi::{fold_many1, separated_list0, separated_list1},
    sequence::{delimited, preceded, separated_pair, terminated},
    IResult,
};
use num::Unsigned;
use strum::{Display, EnumCount, EnumIter, EnumString, IntoEnumIterator};

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
    delimited(multispace0, inner, multispace0)
}

#[derive(Clone, Copy, Debug, Display, EnumCount, EnumIter, EnumString)]
#[strum(ascii_case_insensitive)]
#[repr(u8)]
enum Material {
    Ore,
    Clay,
    Obsidian,
    Geode,
}

fn material(input: &str) -> IResult<&str, Material> {
    map_res(alpha1, str::parse)(input)
}

#[derive(Clone, Copy, Debug, Default)]
struct Costs {
    costs: [isize; Material::COUNT],
}

impl Index<Material> for Costs {
    type Output = isize;

    fn index(&self, index: Material) -> &Self::Output {
        &self.costs[index as usize]
    }
}

fn costs(input: &str) -> IResult<&str, Costs> {
    map_res(
        separated_list0(
            ws(tag("and")),
            separated_pair(map(unsigned, |u: usize| u as isize), space1, material),
        ),
        |vec| {
            let mut costs = Costs::default();

            let mut hits = vec![false; Material::COUNT];

            for (cost, mat) in vec {
                let k = mat as usize;

                if hits[k] {
                    return Err(format!(
                        "definition for a '{mat}' cost requirement found multiple times in a blueprint"
                    ));
                } else {
                    costs.costs[k] = cost;
                    hits[k] = true;
                }
            }

            Ok(costs)
        },
    )(input)
}

#[derive(Clone, Copy, Debug, Default)]
struct Blueprint {
    bots: [Costs; Material::COUNT],
}

impl Index<Material> for Blueprint {
    type Output = Costs;

    fn index(&self, mat: Material) -> &Self::Output {
        &self.bots[mat as usize]
    }
}

fn bot(input: &str) -> IResult<&str, (Material, Costs)> {
    terminated(
        separated_pair(
            delimited(tag("Each"), ws(material), tag("robot")),
            ws(tag("costs")),
            costs,
        ),
        char('.'),
    )(input)
}

fn bots(input: &str) -> IResult<&str, Blueprint> {
    map_res(separated_list1(multispace1, bot), |vec| {
        let mut bp = Blueprint::default();

        let mut hits = vec![false; Material::COUNT];

        for (mat, cost) in vec {
            let k = mat as usize;

            if hits[k] {
                return Err(format!(
                    "definition for a '{mat}' robot found multiple times in a blueprint"
                ));
            } else {
                bp.bots[k] = cost;
                hits[k] = true;
            }
        }

        Ok(bp)
    })(input)
}

fn blueprint(input: &str) -> IResult<&str, (usize, Blueprint)> {
    separated_pair(
        preceded(ws(tag("Blueprint")), unsigned),
        ws(char(':')),
        bots,
    )(input)
}

fn blueprints(input: &str) -> IResult<&str, BTreeMap<usize, Blueprint>> {
    all_consuming(fold_many1(
        delimited(multispace0, blueprint, multispace0),
        BTreeMap::new,
        |mut h, (n, bp)| {
            h.insert(n, bp);
            h
        },
    ))(input)
}

const MINUTES_1: usize = 24;
const MINUTES_2: usize = 32;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct State {
    left: usize,
    materials: [isize; Material::COUNT],
    bots: [isize; Material::COUNT],
}

impl State {
    fn start(left: usize) -> Self {
        let mut ret = Self {
            left,
            ..Default::default()
        };

        ret.bots[Material::Ore as usize] = 1; // start with a single ore-collecting bot

        ret
    }

    fn add_produce(&mut self, prod: &[isize; Material::COUNT]) {
        for (mat, add) in zip(&mut self.materials, prod) {
            *mat += add;
        }
    }

    const fn bots(&self, mat: Material) -> isize {
        self.bots[mat as usize]
    }

    fn build_bots<'a>(&'a self, bp: &'a Blueprint) -> impl Iterator<Item = Self> + 'a {
        Material::iter()
            .filter_map(|bmat| {
                let costs = bp[bmat];

                let new_mats = Material::iter().fold(self.materials.clone(), |mut mats, mat| {
                    mats[mat as usize] -= costs[mat];

                    mats
                });

                if new_mats.iter().any(|v| *v < 0) {
                    None
                } else {
                    let mut ret = Self {
                        materials: new_mats,
                        ..self.clone()
                    };

                    ret.bots[bmat as usize] += 1;

                    Some(ret)
                }
            })
    }

    const fn amount(&self, mat: Material) -> isize {
        self.materials[mat as usize]
    }

    fn estimate(&self) -> isize {
        use Material::*;

        let left = self.left as isize;

        // magical bnb heuristic that everybody is using, but no one knows why
        self.amount(Geode) + (self.bots(Geode) + left / 2) * left 
    }

    fn produce(&self) -> [isize; Material::COUNT] {
        let mut ret = [0isize; Material::COUNT];

        for (n, &nbots) in zip(&mut ret, &self.bots) {
            *n = nbots;
        }

        ret
    }
}

fn find_geodes_for(bp: &Blueprint, avail: usize) -> isize {
    let mut result = 0isize;
    let mut visited = HashSet::new();

    let mut stack = vec![State::start(avail)];

    while let Some(state) = stack.pop() {
        if !visited.contains(&state) {
            visited.insert(state.clone());

            //println!("visiting {state:?}, best == {result})");

            // estimate production BEFORE making any bot
            let produced = state.produce();

            // try all combinations of the bots we can afford and add them as nodes of the current one
            // also, add the plain, no bots added node (may be stupid)
            for mut mat in chain(once(state), state.build_bots(bp)) {
                mat.left -= 1;
                
                // now add production
                mat.add_produce(&produced);

                if mat.left == 0 {
                    let amount = mat.amount(Material::Geode);

                    if amount > result {
                        //println!("!!> pushing {mat:?}, with {amount} > {result})");
                        result = amount;
                    }
                } else {
                    let est = mat.estimate();

                    if est >= result {
                      //  println!("--> pushing {mat:?}, which is projected at {est} (better than cur = {result})");

                        stack.push(mat);
                    } else {
                       // println!("xx> rejecting {mat:?}, which is projected at {est} (worse than cur = {result})");
                    }
                }
            }

            // println!();
        }
    }

    result
}

/// AoC problem for Dec 19 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,

    /// run part 2 instead of part 1
    #[clap(long, default_value_t = false)]
    part2: bool,
}

const BP_NO_2 : usize = 3;

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file, part2 } = Args::parse();

    let vmap = blueprints(&fs::read_to_string(file)?)
        .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
        .1;

    if part2 {
        let pm : isize = vmap.iter().take(BP_NO_2).map(|(_, bp)| find_geodes_for(bp, MINUTES_2)).product();

        println!("pm = {pm}");
    } else {
        let ql = vmap
            .iter()
            .map(|(n, bp)| (*n, find_geodes_for(bp, MINUTES_1)))
            .fold(0, |ci, (n, max)| ci + n as isize * max);

        println!("ql = {ql}");
    }

    Ok(())
}
