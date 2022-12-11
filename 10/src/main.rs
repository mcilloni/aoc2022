use std::{
    any::Any,
    error::Error,
    fs::File,
    io::{self, BufRead, BufReader},
    mem,
    str::FromStr,
};

use clap::Parser;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while},
    character::complete::{char, space1},
    combinator::{map, map_res, opt, recognize, value},
    error::Error as NomError,
    sequence::{separated_pair, tuple},
    IResult,
};
use num::Signed;

const fn is_int_digit(c: char) -> bool {
    c.is_ascii_digit()
}

fn signed<N: Signed + FromStr>(input: &str) -> IResult<&str, N> {
    map_res(
        recognize(tuple((
            opt(alt((char('+'), char('-')))),
            take_while(is_int_digit),
        ))),
        str::parse,
    )(input)
}

#[derive(Clone, Copy, Debug)]
enum Opcode {
    Addx(isize),
    Noop,
}

impl Opcode {
    fn to_uopcodes(self) -> Vec<UOpcode> {
        use UOpcode::*;

        match self {
            Self::Addx(n) => vec![Noop, Add(n)],
            Self::Noop => vec![Noop],
        }
    }
}

fn opcode(input: &str) -> IResult<&str, Opcode> {
    alt((
        map(separated_pair(tag("addx"), space1, signed), |(_, x)| {
            Opcode::Addx(x)
        }),
        value(Opcode::Noop, tag("noop")),
    ))(input)
}

struct Opcodes<I> {
    source: I,
}

impl<I> Iterator for Opcodes<I>
where
    I: Iterator<Item = Result<String, io::Error>>,
{
    type Item = Result<Opcode, Box<dyn Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.source.next().map(|line_res| {
            line_res.map_err(Into::into).and_then(|line| {
                opcode(line.trim())
                    .map(|(_, cmd)| cmd)
                    .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)).into())
            })
        })
    }
}

fn opcodes<I>(source: I) -> Opcodes<I>
where
    I: Iterator<Item = Result<String, io::Error>>,
{
    Opcodes { source }
}

#[derive(Clone, Copy, Debug)]
enum UOpcode {
    Noop,
    Add(isize),
}

trait Inspector {
    fn inspect(&mut self, x: isize, cycle_cnt: isize);
    fn to_any(&mut self) -> Box<dyn Any>;
}

#[derive(Debug, Default)]
struct CycleInspector {
    ss: isize,
}

impl Inspector for CycleInspector {
    fn inspect(&mut self, x: isize, c: isize) {
        const START_CYCLE: isize = 20;
        const PERIOD_CYCLE: isize = 40;
        const END_CYCLE: isize = 220;

        if (START_CYCLE..=END_CYCLE).contains(&c)
            && (c == START_CYCLE || (c - START_CYCLE) % PERIOD_CYCLE == 0)
        {
            self.ss = self.ss.checked_add(signal_strength(x, c)).unwrap();
        }
    }

    fn to_any(&mut self) -> Box<dyn Any> {
        Box::new(mem::take(self))
    }
}

struct Cpu {
    x: isize,

    cycle_cnt: isize,

    inspector: Option<Box<dyn Inspector + 'static>>,
}

impl Cpu {
    fn new() -> Self {
        Cpu {
            x: 1,
            cycle_cnt: 0,
            inspector: None,
        }
    }

    fn draw(&self) {
        let (x, c) = self.state();

        if (x - 1..=x + 1).contains(&((c - 1) % 40)) {
            print!("#");
        } else {
            print!(".");
        }

        if c % 40 == 0 {
            println!();
        }
    }

    fn exec(&mut self, uop: UOpcode) {
        use UOpcode::*;

        self.cycle_cnt = self.cycle_cnt.checked_add(1).unwrap();

        if let Some(inspector) = &mut self.inspector {
            inspector.inspect(self.x, self.cycle_cnt);
        }

        self.draw();

        match uop {
            Add(n) => {
                self.x = self.x.checked_add(n).unwrap();
            }
            Noop => {}
        }
    }

    fn exec_batch(&mut self, it: impl IntoIterator<Item = UOpcode>) {
        it.into_iter().for_each(|uopc| self.exec(uopc))
    }

    fn register_inspect(&mut self, f: impl Inspector + 'static) -> Option<Box<dyn Inspector>> {
        mem::replace(&mut self.inspector, Some(Box::new(f)))
    }

    fn state(&self) -> (isize, isize) {
        (self.x, self.cycle_cnt)
    }

    fn unregister_inspect(&mut self) -> Option<Box<dyn Inspector>> {
        mem::take(&mut self.inspector)
    }
}

fn signal_strength(x: isize, cycle_cnt: isize) -> isize {
    x.checked_mul(cycle_cnt).unwrap()
}

/// AoC problem for Dec 10 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let opcs = opcodes(BufReader::new(File::open(file)?).lines());

    let mut cpu = Cpu::new();

    cpu.register_inspect(CycleInspector::default());

    for maybe_opc in opcs {
        cpu.exec_batch(maybe_opc?.to_uopcodes());
    }

    let ci = cpu
        .unregister_inspect()
        .map(|mut b| b.to_any())
        .and_then(|b| b.downcast::<CycleInspector>().ok())
        .unwrap();

    println!("signal strenght at given cycles = {}", ci.ss);

    Ok(())
}
