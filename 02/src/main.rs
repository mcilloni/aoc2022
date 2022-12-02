use std::{
    env::args,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
};

const PICKS: [Pick; 3] = [Pick::Rock, Pick::Paper, Pick::Scissors];

struct Game {
    them: Pick,
    outcome: Outcome,
}

impl Game {
    fn picks(&self) -> Picks {
        let Game { them, outcome } = *self;

        /*
         * A results LUT can be computed using the following table:
         *
         *
         *        +----+----+----+
         *        | L0 | D3 | W6 |
         *   +----+----+----+----+
         *   | R1 | S3 | R1 | P2 |
         *   +----+----+----+----+
         *   | P2 | R1 | P2 | S3 |
         *   +----+----+----+----+
         *   | S3 | P2 | S3 | R1 |
         *   +----+----+----+----+
         *
         *   from this we can deduce our pick as
         *
         *   us = PICKS[((them - 1) + sign(outcome - 3)) modulo 3]
         *
         *   where modulo is the Euclidean reminder operator
         */
        let us = PICKS[((them as isize - 1) + (outcome as isize - 3).signum())
            .rem_euclid(PICKS.len() as isize) as usize];

        Picks { us, them }
    }
}

impl TryFrom<&str> for Game {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        let (a, b) = value
            .trim()
            .split_once(char::is_whitespace)
            .ok_or(format!("invalid string '{value}'"))?;

        Ok(Self {
            them: a.try_into()?,
            outcome: b.try_into()?,
        })
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum Outcome {
    Win = 6,
    Draw = 3,
    Loss = 0,
}

impl<'a> TryFrom<&'a str> for Outcome {
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        use Outcome::*;

        match value.trim() {
            "X" => Ok(Loss),
            "Y" => Ok(Draw),
            "Z" => Ok(Win),
            str => Err(format!("invalid pattern '{str}'")),
        }
    }
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

impl<'a> TryFrom<&'a str> for Pick {
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        use Pick::*;

        match value.trim() {
            "A" => Ok(Rock),
            "B" => Ok(Paper),
            "C" => Ok(Scissors),
            str => Err(format!("invalid pattern '{str}'")),
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

fn main() -> Result<(), Box<dyn Error>> {
    let file = args().nth(1).ok_or("error: no file name passed")?;

    let lines = BufReader::new(File::open(file)?).lines();

    let mut score = 0usize;

    for line_res in lines {
        score = score
            .checked_add(Game::try_from(&line_res? as &str)?.picks().evaluate())
            .ok_or("integer overflow")?;
    }

    println!("final score = {score}");

    Ok(())
}
