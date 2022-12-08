use std::{
    collections::{hash_map::Entry, HashMap},
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    str::FromStr, mem,
};

use clap::Parser;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while},
    character::complete::{char, space0, space1},
    combinator::{all_consuming, map, map_res, rest},
    error::{Error as NomError},
    sequence::{delimited, separated_pair, tuple},
    IResult,
};
use num::Unsigned;

fn is_int_digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn unsigned<N: Unsigned + FromStr>(input: &str) -> IResult<&str, N> {
    map_res(take_while(is_int_digit), str::parse)(input)
}

#[derive(Debug)]
enum Cmd {
    Chdir(String),
    Ls,
}

fn command(input: &str) -> IResult<&str, Cmd> {
    fn chdir(input: &str) -> IResult<&str, Cmd> {
        map(tuple((tag("cd"), space1, rest)), |(_, _, arg)| {
            Cmd::Chdir(str::to_owned(arg))
        })(input)
    }

    fn ls(input: &str) -> IResult<&str, Cmd> {
        map(tag("ls"), |_| Cmd::Ls)(input)
    }

    fn acommand(input: &str) -> IResult<&str, Cmd> {
        map(
            separated_pair(char('$'), space1, alt((chdir, ls))),
            |(_, cmd)| cmd,
        )(input)
    }

    all_consuming(delimited(space0, acommand, space0))(input)
}

#[derive(Debug, Default)]
struct Dir {
    entries: HashMap<String, u64>,
}

impl Dir {
    fn new() -> Self {
        Dir::default()
    }

    fn push_entry(&mut self, name: String, nn: u64) -> Result<(), String> {
        use Entry::*;

        match self.entries.entry(name) {
            Occupied(oe) => Err(format!("duplicated directory entry '{}'", oe.key())),
            Vacant(ve) => {
                ve.insert(nn);

                Ok(())
            }
        }
    }
}

fn dirname(input: &str) -> IResult<&str, &str> {
    map(separated_pair(tag("dir"), space1, rest), |(_, name)| name)(input)
}

#[derive(Debug)]
enum Dirent {
    Dir(Dir),
    File(FileInfo),
}

fn dirent(input: &str) -> IResult<&str, (String, Dirent)> {
    all_consuming(delimited(
        space0,
        alt((
            map(dirname, |dn| (dn.into(), Dirent::Dir(Dir::new()))),
            map(fileinfo, |(n, fi)| (n, Dirent::File(fi))),
        )),
        space0,
    ))(input)
}

#[derive(Debug)]
struct FileInfo {
    size: u64,
}

fn fileinfo(input: &str) -> IResult<&str, (String, FileInfo)> {
    map(separated_pair(unsigned, space1, rest), |(sz, name)| {
        (name.to_owned(), FileInfo { size: sz })
    })(input)
}

#[derive(Debug)]
struct Tree {
    free: u64,
    ents: HashMap<u64, Dirent>,
}

impl Tree {
    fn new() -> Self {
        Default::default()
    }

    fn get(&self, n: u64) -> Option<&Dirent> {
        self.ents.get(&n)
    }

    fn get_mut(&mut self, n: u64) -> Option<&mut Dirent> {
        self.ents.get_mut(&n)
    }

    fn newent(&mut self, ent: Dirent) -> u64 {
        let nn = self.free;

        self.ents.insert(nn, ent);

        self.free += 1;

        nn
    }
}

impl Default for Tree {
    fn default() -> Self {
        Self {
            free: 1,
            ents: {
                let mut map = HashMap::default();

                map.insert(0, Dirent::Dir(Dir::default())); // root at 0

                map
            },
        }
    }
}

const TOT_CAP : u64 = 70000000;
const REQ_FREE : u64 = 30000000;

#[derive(Clone, Copy)]
struct DirStat<'a> {
    name: &'a str,
    size: u64,
}

#[derive(Default)]
struct Stats<'a> {
    sizes: Vec<DirStat<'a>>,
    tot: u64,
}

fn probe_ent<'a>(dest: &mut Vec<DirStat<'a>>, tree: &'a Tree, name: &'a str, n: u64) -> u64 {
    let ent = tree.get(n).expect("must exist");
    
    match ent {
        Dirent::Dir(dir) => {
                let (mut udest, size) = dir.entries.iter().fold(
                    (mem::take(dest), 0u64), 
                    |(mut dest, acc), (fname, el)| {
                        let size = probe_ent(&mut dest, tree, fname, *el);

                        
                        (dest, acc + size)
                    }
                );
                
                udest.push(DirStat { name, size });
                *dest = udest;

                size
            }
            Dirent::File(FileInfo { size }) => *size,
        }
    }
    
fn probe_sizes(tree: &Tree) -> Stats<'_> {
    let mut sizes = vec![];

    let tot = probe_ent(&mut sizes, tree, "/", 0);

    Stats {
        sizes, tot
    }
}

/// AoC problem for Dec 05 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let mut lines = BufReader::new(File::open(file)?).lines().peekable();

    let mut tree = Tree::new();
    let mut stack = vec![0u64]; // root

    loop {
        let cmdline = {
            match lines.by_ref().next() {
                Some(Ok(line)) => line,
                Some(Err(err)) => return Err(format!("{err}").into()),
                None => break,
            }
        };

        let (_, cmd) =
            command(&cmdline).map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?;

        match cmd {
            Cmd::Chdir(d) => match &d as &str {
                "/" => {
                    stack.drain(1..);
                }
                ".." => {
                    if stack.len() > 1 {
                        stack.pop();
                    }
                }
                d => {
                    let cur_node = *stack.last().unwrap();

                    let Some(Dirent::Dir(cdir)) = tree.get(cur_node) else {
                            panic!("{cur_node} doesn't refer to a dir");
                        };

                    let sentr = *cdir.entries.get(d).ok_or(format!("'{d}': not found"))?;

                    if let Dirent::Dir(_) = tree.get(sentr).expect("must exist") {
                    } else {
                        return Err(format!("can't cd to '{d}': not a directory").into());
                    }

                    stack.push(sentr);
                }
            },
            Cmd::Ls => {
                let mut dl = Dir::new();
                let cur_node = *stack.last().unwrap();

                loop {
                    match lines.by_ref().peek() {
                        Some(Ok(dline)) if dline.trim_start().starts_with('$') => break,
                        Some(Ok(dline)) => {
                            let (_, (name, ent)) = dirent(dline).map_err(|e| {
                                e.map(|e| NomError::new(e.input.to_string(), e.code))
                            })?;

                            let nn = tree.newent(ent);

                            dl.push_entry(name.to_owned(), nn)?;
                        }
                        Some(Err(e)) => return Err(format!("{e}").into()),
                        None => break,
                    }

                    lines.next(); // skip
                }

                let Some(Dirent::Dir(cdir)) = tree.get_mut(cur_node) else {
                    panic!("{cur_node} doesn't refer to a dir");
                };

                *cdir = dl;
            }
        }
    }
    
    let Stats { sizes: dirs, tot: in_use } = probe_sizes(&tree);

    let cur_free = TOT_CAP - in_use;
    let to_free = REQ_FREE - cur_free;

    let res = dirs.iter().fold(DirStat { size: in_use, name: "/" }, |best, current| {
        if current.size < best.size && current.size > to_free {
            *current
        } else {
            best
        }
    });

    println!("tot_cap = {TOT_CAP}, cur_free = {cur_free}, in_use = {in_use}, to_free = {to_free}");
    println!("solution: rm '{}', size = {}", res.name, res.size);

    Ok(())
}
