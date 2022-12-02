use std::{
    env::args,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
};

#[derive(Clone, Copy, Eq, PartialEq)]
enum Outcome {
    Win = 6,
    Draw = 3,
    Loss = 0,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Pick {
    Rock = 1,
    Paper = 2,
    Scissors = 3,
}

impl Pick {
    fn against(self, other: Self) -> Outcome {
        use Outcome::*;
        use Pick::*;

        match (self, other) {
            (Rock, Scissors) | (Scissors, Paper) | (Paper, Rock) => Win,
            _ if self == other => Draw,
            _ => Loss,
        }
    }
}

impl <'a> TryFrom<&'a str> for Pick {
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        use Pick::*;

        match value.trim() {
            "A" | "X" => Ok(Rock),
            "B" | "Y" => Ok(Paper),
            "C" | "Z" => Ok(Scissors),
            str => Err(format!("invalid pattern '{str}'"))
        }
    }
}

struct Picks {
    us: Pick,
    them: Pick,
}

impl Picks {
    fn evaluate(&self) -> usize {
        let Picks { us, them } = self;

        let outcome = us.against(*them);

        usize::checked_add(*us as usize, outcome as usize).unwrap()
    }
}

impl TryFrom<&str> for Picks {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let (a, b) = value.trim().split_once(char::is_whitespace).ok_or(format!("invalid string '{value}'"))?;

        Ok(Picks { us: b.try_into()?, them: a.try_into()? })
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let file = args().nth(1).ok_or("error: no file name passed")?;

    let lines = BufReader::new(File::open(file)?).lines();

    let mut score = 0usize;

    for line_res in lines {
        score = score.checked_add(Picks::try_from(&line_res? as &str)?.evaluate()).ok_or("integer overflow")?;
    }

    println!("final score = {score}");

    Ok(())
}
