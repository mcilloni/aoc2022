use std::{
    env::args,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    collections::BTreeSet,
};

fn priority_of(b: u8) -> usize {
    (match b {
        b'a'..=b'z' => b - b'a' + 1u8,
        b'A'..=b'Z' => b - b'A' + 27u8,
        _ => 0u8,
    }) as usize
}

fn main() -> Result<(), Box<dyn Error>> {
    let file = args().nth(1).ok_or("error: no file name passed")?;

    let mut lines = BufReader::new(File::open(file)?).lines();

    let mut acc = 0usize;
    let mut lineno = 1usize;

    loop {
        let Some(one_res) = lines.next() else {
            break;
        };

        let one = one_res?.into_bytes();
        let two = lines.next().ok_or("early EOF 2")??.into_bytes();
        let three = lines.next().ok_or("early EOF 3")??.into_bytes();

        lineno += 3usize;

        if !(one.is_ascii() && two.is_ascii() && three.is_ascii()) {
            return Err(format!("file contains non-ASCII characters").into());
        }

        let one = BTreeSet::from_iter(one);
        let two = BTreeSet::from_iter(two);
        let three = BTreeSet::from_iter(three);
        
        let first_two : BTreeSet<_> = one.intersection(&two).cloned().collect();
        let mut intersect = first_two.intersection(&three).cloned();
        
        let Some(common) = intersect.next() else {
            return Err(format!("not enough common values at {lineno}").into());
        };
        
        if intersect.next().is_some() {
            return Err(format!("too many common values at {lineno}").into());
        };

        acc = acc
            .checked_add(priority_of(common))
            .ok_or("integer overflow")?;
    }

    println!("pri = {acc}");

    Ok(())
}
