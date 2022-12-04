use std::{
    env::args,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    iter::Peekable,
};

fn skip_below<T: PartialOrd>(it: &mut Peekable<impl Iterator<Item = T>>, val: &T) {
    while let Some(e) = it.peek() {
        if *e >= *val {
            break;
        } else {
            it.next();
        }
    }
}

fn find_common(mut bytes: Vec<u8>) -> Result<u8, &'static str> {
    let cutoff = bytes.len() / 2;

    let (first, second) = bytes.split_at_mut(cutoff);

    first.sort();
    second.sort();

    let mut it1 = first.iter().cloned().peekable();
    let mut it2 = second.iter().cloned().peekable();

    loop {
        let Some((&b1, &b2)) = Option::zip(it1.peek(), it2.peek()) else {
            return Err("no common value found");
        };

        use std::cmp::Ordering::*;

        match b1.cmp(&b2) {
            Less => skip_below(&mut it1, &b2),
            Equal => return Ok(b1),
            Greater => skip_below(&mut it2, &b1),
        }
    }
}

fn priority_of(b: u8) -> usize {
    (match b {
        b'a'..=b'z' => b - b'a' + 1u8,
        b'A'..=b'Z' => b - b'A' + 27u8,
        _ => 0u8,
    }) as usize
}

fn main() -> Result<(), Box<dyn Error>> {
    let file = args().nth(1).ok_or("error: no file name passed")?;

    let lines = BufReader::new(File::open(file)?).lines();

    let mut acc = 0usize;

    for line_res in lines {
        let line = line_res?;

        if !line.is_ascii() {
            return Err(format!(r#"string "{line}" contains non-ASCII characters"#).into());
        }

        acc = acc
            .checked_add(priority_of(find_common(line.into_bytes())?))
            .ok_or("integer overflow")?;
    }

    println!("pri = {acc}");

    Ok(())
}
