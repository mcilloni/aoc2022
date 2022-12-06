use std::{
    env::args,
    error::Error,
    fs::File,
    io::{BufReader, Read},
    mem::{replace, transmute, transmute_copy, ManuallyDrop, MaybeUninit},
};

#[derive(Debug)]
struct Buffer<T, const N: usize> {
    items: [T; N],
    cur: usize,
}

impl<T: Default, const N: usize> Buffer<T, N> {
    fn new() -> Self {
        Default::default()
    }

    fn get(&self) -> &[T; N] {
        &self.items
    }

    fn push_back(&mut self, el: T) -> T {
        let ret = replace(&mut self.items[self.cur], el);

        self.cur = self.cur.saturating_add(1) % N;

        ret
    }
}

impl<T: Default, const N: usize> Default for Buffer<T, N> {
    fn default() -> Self {
        let items = {
            // Create an uninitialized array of `MaybeUninit`. The `assume_init` is
            // safe because the type we are claiming to have initialized here is a
            // bunch of `MaybeUninit`s, which do not require initialization.
            let mut data: [MaybeUninit<T>; N] = unsafe { MaybeUninit::uninit().assume_init() };

            // Dropping a `MaybeUninit` does nothing, so if there is a panic during this loop,
            // we have a memory leak, but there is no memory safety issue.
            for elem in &mut data[..] {
                elem.write(Default::default());
            }

            // Everything is initialized. Transmute the array to the
            // initialized type.

            data
        };

        // bypasses a Rust limitation about transmute not supporting const generics
        unsafe {
            Self {
                items: transmute_copy(&ManuallyDrop::new(items)),
                cur: Default::default(),
            }
        }
    }
}

const WINDOW_SIZE: usize = 14;

fn main() -> Result<(), Box<dyn Error>> {
    let file = args().nth(1).ok_or("error: no file name passed")?;

    let file = File::open(file)?;

    let mut cbuf = Buffer::<_, WINDOW_SIZE>::new();
    let mut cnt = 0u64;

    'probe: for byte_res in file.bytes() {
        cbuf.push_back(byte_res?);

        cnt = cnt.checked_add(1).ok_or("integer overflow")?;

        let raw = cbuf.get();

        // optimization: we know the input is just ASCII alpha chars.
        // At start cbuf contains all zeroes - do not do anything until we've slurped at least WINDOW_SIZE bytes
        if raw[WINDOW_SIZE - 1] != 0 {
            let mut stk = raw.clone();

            stk.sort();

            for w in stk.windows(2) {
                if let [e1, e2] = w {
                    if e1 == e2 {
                        continue 'probe;
                    }
                }
            }

            // found a probe sequence
            println!("found a sequence at byte {}", cnt);

            return Ok(());
        }
    }

    Err("no packet start found in input".into())
}
