use std::{error::Error, fs};

use clap::Parser;
use nom::{
    branch::alt,
    character::complete::{char, multispace1},
    combinator::value,
    error::Error as NomError,
    multi::{many0, many1},
    sequence::terminated,
    IResult,
};

fn wonky_cipher(input: &str) -> IResult<&str, isize> {
    alt((
        value(0, char('0')),
        value(1, char('1')),
        value(2, char('2')),
        value(-1, char('-')),
        value(-2, char('=')),
    ))(input)
}

fn wonky_num(input: &str) -> IResult<&str, isize> {
    let (rem, cs) = many1(wonky_cipher)(input)?;

    let (val, _) = cs
        .into_iter()
        .rev()
        .fold((0, 1), |(acc, pos), cipher| (acc + cipher * pos, pos * 5));

    Ok((rem, val))
}

fn wonky_nums(input: &str) -> IResult<&str, Vec<isize>> {
    many0(terminated(wonky_num, multispace1))(input)
}

fn as_snafu(mut n: isize) -> String {
    if n == 0 {
        return '0'.into();
    }

    let mut ret = String::new();

    let mut carry = 0;

    while n != 0 {
        let cipher = n.rem_euclid(5) as u8 + carry;
        n /= 5;

        let (ch, ncarry) = match cipher {
            0..=2 => ((cipher + b'0') as char, 0),
            3 => ('=', 1),
            4 => ('-', 1),
            _ => unreachable!(),
        };

        carry = ncarry;
        ret.push(ch);
    }

    ret.chars().rev().collect()
}

/// AoC problem for Dec 25 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let input = fs::read_to_string(file)?;

    let nums: isize = wonky_nums(&input)
        .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?
        .1
        .into_iter()
        .sum();

    println!("{nums} -> {}", as_snafu(nums));

    Ok(())
}
