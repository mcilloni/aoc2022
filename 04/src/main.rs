use std::{
    env::args,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    ops::RangeInclusive,
};

use nom::{
    bytes::complete::{tag, take_while},
    combinator::map_res,
    error::Error as NomError,
    sequence::separated_pair,
    IResult,
};

#[derive(Debug)]
struct RangePair {
    first: RangeInclusive<u64>,
    second: RangeInclusive<u64>,
}

impl RangePair {
    fn is_overlapping(&self) -> bool {
        self.first.contains(&self.second.start()) || self.first.contains(&self.second.end())
            || self.second.contains(&self.first.start()) || self.second.contains(&self.first.end())
    }

    fn is_fully_overlapping(&self) -> bool {
        self.first.contains(&self.second.start()) && self.first.contains(&self.second.end())
            || self.second.contains(&self.first.start()) && self.second.contains(&self.first.end())
    }
}

fn irange_pair<'a>(input: &'a str) -> IResult<&'a str, RangePair> {
    fn is_int_digit(c: char) -> bool {
        c.is_digit(10)
    }

    fn u64_num(input: &str) -> IResult<&str, u64> {
        map_res(take_while(is_int_digit), str::parse)(input)
    }

    fn irange_u64(input: &str) -> IResult<&str, RangeInclusive<u64>> {
        let (input, (start, end)) = separated_pair(u64_num, tag("-"), u64_num)(input)?;

        Ok((input, start..=end))
    }

    let (input, (first, second)) = separated_pair(irange_u64, tag(","), irange_u64)(input)?;

    Ok((input, RangePair { first, second }))
}

fn main() -> Result<(), Box<dyn Error>> {
    let file = args().nth(1).ok_or("error: no file name passed")?;

    let lines = BufReader::new(File::open(file)?).lines();

    let mut acc = 0u64;
    let mut acc_tot = 0u64;

    for line_res in lines {
        let line = line_res?;
        let (rem, rp) = irange_pair(line.trim())
            .map_err(|e| e.map(|e| NomError::new(e.input.to_string(), e.code)))?;

        if rem.len() != 0 {
            return Err(format!("stray characters in line '{line}'").into());
        }

        if rp.is_overlapping() {
            acc = acc.checked_add(1).ok_or("integer overflow")?;

            if rp.is_fully_overlapping() {
                acc_tot = acc_tot.checked_add(1).ok_or("integer overflow")?;
            }
        }
    }

    println!("overlaps = {acc}, full_overlap = {acc_tot}");

    Ok(())
}
