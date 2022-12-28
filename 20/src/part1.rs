use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
};

use clap::Parser;

fn shuffle(mut values: Vec<(i16, i16)>) -> Vec<i16> {
    let len = values.len() as i16;

    // rotate by GAPS, not items, this is the trick
    // the issue here is that by moving forward, we must remember we leave a gap behind, by moving backwards we
    // are also jumping over the last element when we reach the end
    let gaps = len - 1;

    for n in 0..len {
        let (nn, &(_, cur)) = values
            .iter()
            .enumerate()
            .find(|(_, (i, _))| *i == n)
            .unwrap();

        if cur != 0 {
            let nn = nn as i16;

            let skew = nn.checked_add(cur).unwrap();

            let new_ix = skew.rem_euclid(gaps);

            let el = values.remove(nn as usize);

            values.insert(new_ix as usize, el);
        }
    }

    values.into_iter().map(|(_, v)| v).collect()
}

/// AoC problem for Dec 20 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

type BoxErr = Box<dyn Error>;

const GAP: usize = 1000;

const TAKE_AT: [usize; 3] = [GAP, GAP * 2, GAP * 3];

fn main() -> Result<(), BoxErr> {
    let Args { file } = Args::parse();

    let nums: Vec<_> = BufReader::new(File::open(file)?)
        .lines()
        .enumerate()
        .map(|(n, some_line)| {
            some_line.map_err(BoxErr::from).and_then(|line| {
                line.trim()
                    .parse()
                    .map_err(BoxErr::from)
                    .map(|v| (n.try_into().unwrap(), v))
            })
        })
        .collect::<Result<_, _>>()?;

    if !nums.iter().any(|(_, n)| *n == 0) {
        return Err("no zero found in sequence".into());
    }

    let adj = shuffle(nums);

    let zpos = adj.iter().enumerate().find(|&(_, n)| *n == 0).unwrap().0;

    let res: i16 = TAKE_AT
        .into_iter()
        .map(|p| adj[(p + zpos) % adj.len()])
        .sum();

    println!("res = {res}");

    Ok(())
}
