use std::{
    error::Error,
    fs::File,
    io::Read,
    iter::{repeat, zip},
    mem,
    ops::ControlFlow,
};

use ansi_term::{Colour, Style};
use clap::Parser;

#[derive(Clone, Copy, Debug)]
struct Tree {
    height: u8,
    visible: bool,
}

impl Tree {
    fn new(height: u8) -> Self {
        Self {
            height,
            visible: false,
        }
    }
}

struct ForestBuilder {
    trees: Vec<Tree>,
    dim: Option<usize>,
    cur_line: usize,
}

impl ForestBuilder {
    fn new() -> Self {
        ForestBuilder {
            trees: Default::default(),
            dim: None,
            cur_line: 0,
        }
    }

    fn done(self) -> Result<Forest, String> {
        if self.is_full() {
            Ok(Forest {
                trees: self.trees,
                dim: self.dim.unwrap(),
            })
        } else {
            Err(format!("forest not full yet"))
        }
    }

    fn at_eol(&self) -> bool {
        self.dim.map(|dim| self.cur_line >= dim).unwrap_or(false)
    }

    fn is_full(&self) -> bool {
        self.dim
            .map(|dim| self.trees.len() >= dim * dim)
            .unwrap_or(false)
    }

    fn knows_dim(&self) -> bool {
        self.dim.is_some()
    }

    fn newline(&mut self) -> Result<(), String> {
        // ignore newlines on full matrices
        if !self.is_full() {
            match (self.at_eol(), self.knows_dim()) {
                (true, true) => {
                    self.cur_line = 0;
                }
                (false, false) => self.dim = Some(mem::take(&mut self.cur_line)),
                (false, true) => {
                    return Err(format!(
                        "invalid newline neither at line end nor at start of file"
                    ))
                }
                (true, false) => unreachable!(),
            }
        }

        Ok(())
    }

    fn push(&mut self, tree: Tree) -> Result<(), String> {
        if self.at_eol() || self.is_full() {
            let dim = self.dim.unwrap();

            Err(format!(
                "pushing one too many trees to a {dim}x{dim} matrix"
            ))
        } else {
            self.trees.push(tree);
            self.cur_line += 1;

            Ok(())
        }
    }
}

struct ColumnIter<T, I>
where
    I: DoubleEndedIterator<Item = T>,
{
    source: I,

    dim: usize,
    col: usize,
}

impl<T, I> ColumnIter<T, I>
where
    I: DoubleEndedIterator<Item = T>,
{
    // replace with advance_by when stable
    fn skip_elements(&mut self, n: usize) -> Option<()> {
        for _ in 0..n {
            if self.source.next().is_none() {
                return None;
            }
        }

        Some(())
    }

    // replace with advance_back_by when stable
    fn skip_elements_back(&mut self, n: usize) -> Option<()> {
        for _ in 0..n {
            if self.source.next_back().is_none() {
                return None;
            }
        }

        Some(())
    }
}

impl<T, I> Iterator for ColumnIter<T, I>
where
    I: DoubleEndedIterator<Item = T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // skip the next `col` items
        let ret = self
            .skip_elements(self.col)
            .and_then(|()| self.source.next());

        self.skip_elements(self.dim - self.col - 1); // ignore result

        ret
    }
}

impl<T, I> DoubleEndedIterator for ColumnIter<T, I>
where
    I: DoubleEndedIterator<Item = T>,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        let ret = self
            .skip_elements_back(self.dim - self.col - 1)
            .and_then(|()| self.source.next_back()); // ignore result

        self.skip_elements_back(self.col); // ignore result

        ret
    }
}

struct Forest {
    trees: Vec<Tree>, // will be used as a 2D array
    dim: usize,
}

impl Forest {
    fn column(&self, j: usize) -> impl DoubleEndedIterator<Item = Tree> + '_ {
        ColumnIter {
            source: self.trees.iter().cloned(),
            dim: self.dim,
            col: j,
        }
    }

    fn column_mut(&mut self, j: usize) -> impl DoubleEndedIterator<Item = &mut Tree> {
        ColumnIter {
            source: self.trees.iter_mut(),
            dim: self.dim,
            col: j,
        }
    }

    const fn index_of(&self, i: usize, j: usize) -> usize {
        i * self.dim + j
    }

    // fn get(&self, i: usize, j: usize) -> Option<Tree> {
    //     let ix = self.index_of(i, j);

    //     self.trees.get(ix).cloned()
    // }

    // fn get_mut(&mut self, i: usize, j: usize) -> Option<&mut Tree> {
    //     let ix = self.index_of(i,j);

    //     self.trees.get_mut(ix)
    // }

    fn row(&self, i: usize) -> impl DoubleEndedIterator<Item = Tree> + '_ {
        let offs = self.index_of(i, 0);

        self.trees[offs..(offs + self.dim)].into_iter().cloned()
    }

    fn row_mut(&mut self, i: usize) -> impl DoubleEndedIterator<Item = &mut Tree> {
        let offs = self.index_of(i, 0);

        self.trees[offs..(offs + self.dim)].as_mut().into_iter()
    }

    fn score_tree(&self, i: usize, j: usize) -> Option<(usize, u64)> {
        let ix = self.index_of(i, j);

        const fn should_cut(this: Tree, neigh: Tree) -> bool {
            neigh.height >= this.height
        }

        if let Some(this) = self.trees.get(ix).cloned() {
            let mut sum = 0u64;

            for neigh in self.row(i).rev().skip(self.dim - j) {
                sum += 1; // gets a one for simply having a neighbour

                if should_cut(this, neigh) {
                    break; // stop, we will not see anything past this tree
                }
            }

             let mut score = mem::take(&mut sum);

            for neigh in self.row(i).skip(j + 1) {
                sum += 1; // gets a one for simply having a neighbour

                if should_cut(this, neigh) {
                    break; // stop, we will not see anything past this tree
                }
            }

            score *= mem::take(&mut sum);

            for neigh in self.column(j).rev().skip(self.dim - i) {
                sum += 1; // gets a one for simply having a neighbour

                if should_cut(this, neigh) {
                    break; // stop, we will not see anything past this tree
                }
            }

            score *= mem::take(&mut sum);

            for neigh in self.column(j).skip(i + 1) {
                sum += 1; // gets a one for simply having a neighbour

                if should_cut(this, neigh) {
                    break; // stop, we will not see anything past this tree
                }
            }

            score *= mem::take(&mut sum);

            Some((ix, score))
        } else {
            None
        }
    }
}

fn dump_forest(fmap: &Forest, best: usize) {
    for (ix, tree) in fmap.trees.iter().enumerate() {
        if ix > 0 && ix % fmap.dim == 0 {
            println!();
        }

        print!(
            "{}",
            Style::new()
                .on(match (tree.visible, best == ix) {
                    (_, true) => Colour::Blue,
                    (true, _) => Colour::Green,
                    (false, _) => Colour::Red,
                })
                .paint(tree.height.to_string())
        );
    }
}

fn load_forest<R: Read>(r: R) -> Result<Forest, Box<dyn Error>> {
    let mut fb = ForestBuilder::new();

    let source = r.bytes();

    for maybe_byte in source {
        match maybe_byte? {
            b if b.is_ascii_digit() => fb.push(Tree::new(b - b'0')),
            b'\r' | b' ' | b'\t' => continue,
            b'\n' => fb.newline(),
            b => Err(format!("invalid charater '{}'", b as char)),
        }?
    }

    fb.done().map_err(Into::into)
}

const TREE_MAX_HEIGHT: u8 = 9;

fn probe_edges(fmap: &mut Forest) -> usize {
    // probe all trees in all columns and rows of the forest

    fn probe_seq(prec_height: Option<u8>, tree: &mut Tree) -> ControlFlow<(), Option<u8>> {
        use ControlFlow::*;

        match prec_height {
            Some(TREE_MAX_HEIGHT) => Break(()), // cut if the highest tree possible as been reached
            Some(ph) if tree.height <= ph => Continue(Some(ph)), // ignore this tree
            Some(_) | None => {
                // we're at the edge or the tree is taller than the preceding one
                tree.visible = true;
                Continue(Some(tree.height))
            }
        }
    }

    // probe rows first

    for i in 0..fmap.dim {
        fmap.row_mut(i).try_fold(None, probe_seq);
        fmap.row_mut(i).rev().try_fold(None, probe_seq);
    }

    // probe columns
    for j in 0..fmap.dim {
        fmap.column_mut(j).try_fold(None, probe_seq);
        fmap.column_mut(j).rev().try_fold(None, probe_seq);
    }

    // probably useless traversal, could be optimized in the steps above
    fmap.trees
        .iter()
        .fold(0, |acc, tree| if tree.visible { acc + 1 } else { acc })
}

fn find_best_tree(fmap: &Forest) -> Option<(usize, u64)> {
    (0..fmap.dim)
        .map(|i| zip(repeat(i).take(fmap.dim), 0..fmap.dim))
        .flatten()
        .map(|(i, j)| fmap.score_tree(i, j).expect("out of bound access"))
        .fold(None, |some_best, cur| {
            if let Some(best) = some_best {
                Some(if cur.1 > best.1 { cur } else { best })
            } else {
                Some(cur)
            }
        })
}

/// AoC problem for Dec 05 2022
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File to parse
    file: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    let Args { file } = Args::parse();

    let mut fmap = File::open(file).map_err(Into::into).and_then(load_forest)?;

    let (best, score) = find_best_tree(&fmap).ok_or("failed to find the best tree")?;

    let visible = probe_edges(&mut fmap);

    dump_forest(&fmap, best);

    println!("\nThe forest has {visible} visible trees. The best one has score = {score}");

    Ok(())
}
