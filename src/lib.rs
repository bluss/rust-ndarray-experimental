#![feature(specialization, test)]
extern crate ndarray;
extern crate rayon;
extern crate test;

pub mod prelude {
    pub use ndarray::{
        ArrayView,
        ArrayViewMut,
        OwnedArray,
        RcArray,
        ArrayBase,
        Dimension,
        Ix, Ixs,
        Axis,
        NdFloat,
        AsArray,
    };
}
use ndarray::DataMut;

use prelude::*;

const SPLIT_SIZE: usize = 10000;

/// Apply a function to each element in the array
pub fn apply<A, S, D, F>(array: &mut ArrayBase<S, D>, f: F)
    where S: DataMut<Elem=A>,
          D: Dimension,
          F: Fn(&mut A)
{
    Apply::apply(array, f)
}

trait Apply<F> {
    fn apply(&mut self, f: F);
}

impl<A, S, D, F> Apply<F> for ArrayBase<S, D>
    where S: DataMut<Elem=A>,
          D: Dimension,
          F: Fn(&mut A)
{

    default fn apply(&mut self, f: F) {
        for elt in self {
            f(elt);
        }
    }
}

impl<A, S, D, F> Apply<F> for ArrayBase<S, D>
    where S: DataMut<Elem=A>,
          D: Dimension,
          F: Fn(&mut A) + Sync,
          A: Send,
{

    fn apply(&mut self, f: F) {
        apply_helper(self.view_mut(), &f)
    }
}

fn apply_helper<A, D, F>(array: ArrayViewMut<A, D>, f: &F)
    where D: Dimension,
          F: Fn(&mut A) + Sync,
          A: Send,
{
    let len = array.len();
    // find axis to split by
    if len > SPLIT_SIZE {
        let mut split_axis = 0;
        for (axis, &axis_len) in array.shape().iter().enumerate() {
            if axis_len > 1 {
                split_axis = axis;
                break;
            }
        }
        let len = array.shape()[split_axis];
        let (a, b) = array.split_at(Axis(split_axis), len / 2);
        rayon::join(move || apply_helper(a, f), move || apply_helper(b, f));
    } else {
        // sequential case
        for elt in array {
            f(elt)
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
    }
}

use test::Bencher;
#[bench]
fn map_parallel(b: &mut Bencher) {
    let m = 1000;
    let n = 1000;
    let mut data = OwnedArray::linspace(0., 1., m * n).into_shape((m, n)).unwrap();
    b.iter(|| {
        apply(&mut data, |x| *x = f32::exp(*x));
    })
}

#[bench]
fn map_not_parallel(b: &mut Bencher) {
    use std::cell::Cell;
    let m = 1000;
    let n = 1000;
    let mut data = OwnedArray::linspace(0., 1., m * n).into_shape((m, n)).unwrap();
    let c = Cell::new(0);
    b.iter(|| {
        apply(&mut data, |x| {
            *x = f32::exp(*x);
            let _notsync = &c;
        });
    })
}
