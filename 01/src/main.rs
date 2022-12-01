use std::{
    env::args,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    mem::{swap, take},
};

#[derive(Default)]
struct Stats {
    vals: [usize; 3],
}

impl Stats {
    fn evaluate(&mut self, mut cur: usize) {
        for el in &mut self.vals {
            if cur > *el {
                swap(el, &mut cur)
            }
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let Some(file) = args().nth(1) else {
        return Err("error: no file name passed".into());
    };

    let lines = BufReader::new(File::open(file)?).lines();

    let mut stats = Stats::default();
    let mut acc = 0usize;

    for line_res in lines {
        let line = line_res?;

        if line.chars().all(char::is_whitespace) && acc != 0 {
            stats.evaluate(take(&mut acc));
        } else {
            acc = acc
                .checked_add(line.parse()?)
                .ok_or("error: integer overflow")?;
        }
    }

    let [first, second, third] = stats.vals;

    println!("first = {first}, second = {second}, third = {third}");
    println!("sum = {}", first + second + third);

    Ok(())
}
