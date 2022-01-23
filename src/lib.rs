#![deny(missing_docs)]

//! The **set-partition** crate provides ways to represent, enumerate and count
//! all the possible partitions of a set of a given finite size into subsets.
//!
//! You can use the `set_partitions` function to get the number of partitions of a set
//! with `n` elements, and you can use the `SetPartition` struct and its `increment`()
//! function to enumerate them (as well as its more generic variants).
//!
//! Set partitions are represented as sequences of values, with one value for each
//! element of the partitioned set, such that if two elements are in the same subset,
//! the sequence will have the same values at the corresponding indices.
//!
//! Among all the possible sequences, the one which is lexicographically minimal,
//! and uses values starting from Default::default() and incrementing with increment(),
//! is chosen as the representative.
//!
//! See <http://www-cs-faculty.stanford.edu/~uno/fasc3b.ps.gz> page 27 for more information
//! and a description of the algorithm used here.
//!
//! # How to use
//!
//! For fixed size set partitions, use `SetPartition#`; use `GenSetPartition#` if you want to use something other than `u8` as the data type.
//! For variable size set partitions up to 16 elements (more than 16 result in more than 2^32 partitions), use `SmallSetPartition` or `GenSetPartition`, 
//! For medium sized variable size set partitions, use `ArrayVecSetPartition`.
//! For arbitrary sized set partitions, use `VecSetPartition`.
//!
//! If you don't care about getting the subset contents, pass `()` or omit the type parameter.
//! Otherwise, use `SmallSubsets` or `GenSmallSubsets` to represent subsets using ArrayVecs without memory allocation, for set partitions of size 16 or less.
//! For subsets in other data structures, you can use `ArrayVecSubsets`, `VecSubsets`,`HashSubsets` or `BTreeSubsets`.
//!
//! To enumerate set partitions, call `increment()` until it returns `false`, and use `get()`, `num_subsets()` or `subsets().subsets()` to examine the set partition.

use arrayvec::ArrayVec;
use std::fmt;
use std::fmt::Debug;
use std::cmp::{PartialOrd, Ord, PartialEq, Eq, Ordering};
use std::hash::{Hash, Hasher};
use std::collections::hash_set::HashSet;
use std::collections::btree_set::BTreeSet;
use std::ops;
use std::ops::{Deref, Index};
use std::slice;
use std::borrow::Borrow;
use std::collections::{btree_map, hash_map};

/// Module for the Incrementable trait
pub mod traits
{
    use num_traits::One;
    use std::ops::{Add, AddAssign};

    /// Trait for things that can be incremented, like numbers
    pub trait Incrementable
    {
        /// Increment self by mutable reference
        fn increment(&mut self);

        /// Increment self and return it
        fn incremented(mut self) -> Self 
            where Self: Sized
        {
            self.increment();
            self
        }
    }

    impl<T> Incrementable for T
        where T: One + Add<T, Output = T> + AddAssign<T>
    {
        fn increment(&mut self) {
            *self += One::one();
        }

        fn incremented(self) -> T {
            self + <T as One>::one()
        }
    }

    /*
    // this needs specialization to work...
    impl<T> Incrementable for T
        where T: Iterator
    {
        fn increment(&mut self) {
            self.next();
        }
    }
    */
}
use crate::traits::Incrementable;

/// Trait used by `SetPartition` structs to notify the subsets structure of changes.
///
/// A no-op implementation is available for `()`.
///
/// The `VecSubsets`, `HashSubsets`, `BTreeSubsets` structs implement this
/// trait to provide an updated view of the subsets in the partition, represented
/// with `Vec`, `HashSet` or `BTreeSet` types respectively.
pub trait Subsets<T> : Default
{
    /// Set the number of non-empty sets to `m` converted to usize
    fn set_limit(&mut self, m: &T);

    /// Increment the number of non-empty sets
    fn inc_limit(&mut self);

    /// Clear all sets, reset the number of non-empty sets to 0
    fn clear(&mut self);

    /// Truncate the number of sets to at most `n`
    fn done(&mut self, n: usize);

    /// Reset to the trivial partition of size `n`
    fn reset(&mut self, n: usize);

    /// Notify addition of `idx` to the set identified by `v`
    ///
    /// Items must be added to a set in their correct order.
    fn add(&mut self, idx: usize, v: &T);

    /// Notify removal of `idx` from the set identified by `v`
    ///
    /// This must be the last index added to that set.
    fn remove(&mut self, idx: usize, v: &T);
}

impl<T> Subsets<T> for ()
{
    fn set_limit(&mut self, _: &T) {}
    fn inc_limit(&mut self) {}
    fn clear(&mut self) {}
    fn done(&mut self, _: usize) {}
    fn reset(&mut self, _: usize) {}
    fn add(&mut self, _: usize, _: &T) {}
    fn remove(&mut self, _: usize, _: &T) {}
}

macro_rules! impl_subsets_for_subsets {
    ($T:ty) => {
        fn set_limit(&mut self, m: &$T)
        {
            self.n = TryInto::try_into((*m).clone()).unwrap();
        }

        fn inc_limit(&mut self)
        {
            self.n += 1;
        }

        fn clear(&mut self)
        {
            for s in self.subsets.iter_mut() {
                s.clear();
            }
            self.n = 0;
        }

        fn done(&mut self, n: usize)
        {
            self.truncate(n);
        }
    }
}

macro_rules! impl_traits_for_subsets {
    ($S:ty, {$($impl:tt)*}, {$($where:tt)*}) => {
        $($impl)* Default for $S
            $($where)*
        {
            fn default() -> Self {
                Self {
                    subsets: Default::default(),
                    n: 0
                }
            }
        }
    }
}

macro_rules! impl_set_subsets {
    ($S:ty, $T:ty, {$($impl:tt)*}, {$($where:tt)*}) => {
        $($impl)* Subsets<T> for $S
            $($where)*
        {
            impl_subsets_for_subsets!($T);

            fn reset(&mut self, n: usize)
            {
                self.done(n);

                for s in self.subsets.iter_mut() {
                    s.clear();
                }

                if self.subsets.is_empty() {
                    self.subsets.push(Default::default());
                }

                if let Some(s0) = self.subsets.first_mut() {
                    for i in 0..n {
                        s0.insert(TryInto::try_into(i).unwrap());
                    }
                }
                self.n = if n > 0 {1} else {0}
            }

            fn add(&mut self, idx: usize, v: &$T)
            {
                let vi: usize = TryInto::try_into((*v).clone()).unwrap();
                self.enlarge(vi + 1);
                let it: $T = TryInto::try_into(idx).unwrap();
                self.subsets[vi].insert(it);
            }

            fn remove(&mut self, idx: usize, v: &$T)
            {
                let vi: usize = TryInto::try_into((*v).clone()).unwrap();
                let it: $T = TryInto::try_into(idx).unwrap();
                self.subsets[vi].remove(&it);
            }
        }

        impl_traits_for_subsets!($S, {$($impl)*}, {$($where)*});
    }
}

macro_rules! impl_vec_subsets {
    ($S:ty, $T:ty, {$($impl:tt)*}, {$($where:tt)*}) => {
        $($impl)* Subsets<$T> for $S
            $($where)*
        {
            impl_subsets_for_subsets!($T);

            fn reset(&mut self, n: usize)
            {
                self.done(n);

                for s in self.subsets.iter_mut() {
                    s.clear();
                }

                if self.subsets.is_empty() {
                    self.subsets.push(Default::default());
                }

                if let Some(s0) = self.subsets.first_mut() {
                    for i in 0..n {
                        s0.push(TryInto::try_into(i).unwrap());
                    }
                }
                self.n = if n > 0 {1} else {0}
            }

            fn add(&mut self, idx: usize, v: &$T)
            {
                let vi: usize = TryInto::try_into((*v).clone()).unwrap();
                self.enlarge(vi + 1);
                let it: $T = TryInto::try_into(idx).unwrap();
                self.subsets[vi].push(it);
            }

            fn remove(&mut self, _: usize, v: &$T)
            {
                let vi: usize = TryInto::try_into((*v).clone()).unwrap();
                self.subsets[vi].pop();
                //debug_assert_eq!(r, Some(idx.into()));
            }
        }

        impl_traits_for_subsets!($S, {$($impl)*}, {$($where)*});
    }
}

macro_rules! impl_subsets_helpers {
    () => {
        fn truncate(&mut self, n: usize)
        {
            self.subsets.truncate(n);
        }

        fn enlarge(&mut self, new_n: usize)
        {
            let n = self.subsets.len();
            if n < new_n {
                self.subsets.reserve(new_n - n);
                loop {
                    self.subsets.push(Default::default());
                    if self.subsets.len() == new_n {break;}
                }
            }
        }
    }
}

/// Maintains subsets of a set partition using `HashSet`s
///
/// Pass this as the `Subsets` implementation to the `SetPartition` structs.
#[derive(Clone, Debug)]
pub struct HashSubsets<T>
    where <T as TryInto<usize>>::Error : Debug, <T as TryFrom<usize>>::Error : Debug, T: Clone + TryInto<usize> + TryFrom<usize> + Hash + Eq
{
    subsets: Vec<HashSet<T>>,
    n: usize
}

impl<T> HashSubsets<T>
    where <T as TryInto<usize>>::Error : Debug, <T as TryFrom<usize>>::Error : Debug, T: Clone + TryInto<usize> + TryFrom<usize> + Hash + Eq
{
    /// Return the non-empty subsets part of the set partition
    pub fn subsets(&self) -> &[HashSet<T>] {
        &self.subsets[..self.n]
    }

    impl_subsets_helpers!();
}

impl_set_subsets!(HashSubsets<T>, T, {impl<T>}, {where <T as TryInto<usize>>::Error : Debug, <T as TryFrom<usize>>::Error : Debug, T: Clone + TryInto<usize> + TryFrom<usize> + Hash + Eq});

/// Maintains subsets of a set partition using `BTreeSet`s
///
/// Pass this as the `Subsets` implementation to the `SetPartition` structs.
#[derive(Clone, Debug)]
pub struct BTreeSubsets<T>
    where <T as TryInto<usize>>::Error : Debug, <T as TryFrom<usize>>::Error : Debug, T: Clone + TryInto<usize> + TryFrom<usize> + Ord
{
    subsets: Vec<BTreeSet<T>>,
    n: usize
}

impl<T> BTreeSubsets<T>
    where <T as TryInto<usize>>::Error : Debug, <T as TryFrom<usize>>::Error : Debug, T: Clone + TryInto<usize> + TryFrom<usize> + Ord
{
    /// Return the non-empty subsets part of the set partition
    pub fn subsets(&self) -> &[BTreeSet<T>] {
        &self.subsets[..self.n]
    }

    impl_subsets_helpers!();
}

impl_set_subsets!(BTreeSubsets<T>, T, {impl<T>}, {where <T as TryInto<usize>>::Error : Debug, <T as TryFrom<usize>>::Error : Debug, T: Clone + TryInto<usize> + TryFrom<usize> + Ord});

/// Maintains subsets of a set partition using `Vec`s
///
/// Pass this as the `Subsets` implementation to the `SetPartition` structs.
#[derive(Clone, Debug)]
pub struct VecSubsets<T>
    where <T as TryInto<usize>>::Error : Debug, <T as TryFrom<usize>>::Error : Debug, T: Clone + TryInto<usize> + TryFrom<usize>
{
    subsets: Vec<Vec<T>>,
    n: usize
}

impl<T> VecSubsets<T>
    where <T as TryInto<usize>>::Error : Debug, <T as TryFrom<usize>>::Error : Debug, T: Clone + TryInto<usize> + TryFrom<usize>
{
    /// Return the non-empty subsets part of the set partition
    pub fn subsets(&self) -> &[Vec<T>] {
        &self.subsets[..self.n]
    }

    impl_subsets_helpers!();
}

impl_vec_subsets!(VecSubsets<T>, T, {impl<T>}, {where <T as TryInto<usize>>::Error : Debug, <T as TryFrom<usize>>::Error : Debug, T: Clone + TryInto<usize> + TryFrom<usize>});


/// Maintains subsets of a set partition using `ArrayVec`s
///
/// Pass this as the `Subsets` implementation to the `SetPartition` structs.
#[derive(Debug, Clone)]
pub struct ArrayVecSubsets<T, const N: usize>
{
    subsets: ArrayVec<ArrayVec<T, N>, N>,
    n: usize
}

impl<T, const N: usize> ArrayVecSubsets<T, N>
{
    /// Return the non-empty subsets part of the set partition
    pub fn subsets(&self) -> &[ArrayVec<T, N>] {
        &self.subsets[..self.n]
    }

    fn truncate(&mut self, n: usize)
    {
        while self.subsets.len() > n {
            self.subsets.pop();
        }
    }

    fn enlarge(&mut self, new_n: usize)
    {
        let n = self.subsets.len();
        if n < new_n {
            loop {
                self.subsets.push(Default::default());
                if self.subsets.len() == new_n {break;}
            }
        }
    }
}

impl_vec_subsets!(ArrayVecSubsets<T, N>, T, {impl<T, const N: usize>}, {where <T as TryInto<usize>>::Error : Debug, <T as TryFrom<usize>>::Error : Debug, T: Clone + TryInto<usize> + TryFrom<usize>});

/// Maintains subsets of a set partition using 16-entry `ArrayVec`s
///
/// Pass this as the `Subsets` implementation to the `SetPartition` structs.
pub type GenSmallSubsets<T> = ArrayVecSubsets<T, 16>;

/// Maintains subsets of a set partition using 16-entry `ArrayVec`s
///
/// Pass this as the `Subsets` implementation to the `SetPartition` structs.
pub type SmallSubsets = GenSmallSubsets<u8>;

trait Push
{
    type Item;

    fn try_push(&mut self, x: Self::Item) -> Result<(), ()>;
    fn done(&mut self) -> Result<(), ()>;
}

struct SlicePusher<'a, T: 'a>
{
    iter: std::slice::IterMut<'a, T>
}

impl<'a, T: 'a> SlicePusher<'a, T>
{
    pub fn new(s: &'a mut [T]) -> Self
    {
        SlicePusher {iter: s.iter_mut()}
    }
}

impl<'a, T> Push for SlicePusher<'a, T>
{
    type Item = T;

    fn try_push(&mut self, x: Self::Item) -> Result<(), ()>
    {
        if let Some(a) = self.iter.next() {
            *a = x;
            Ok(())
        } else {
            Err(())
        }
    }

    fn done(&mut self) -> Result<(), ()>
    {
        if self.iter.next().is_none() {
            Ok(())
        } else {
            Err(())
        }
    }
}

impl<T> Push for Vec<T>
{
    type Item = T;

    fn try_push(&mut self, x: Self::Item) -> Result<(), ()>
    {
        self.push(x);
        Ok(())
    }

    fn done(&mut self) -> Result<(), ()>
    {
        Ok(())
    }
}

impl<T, const N: usize> Push for ArrayVec<T, N>
{
    type Item = T;

    fn try_push(&mut self, x: Self::Item) -> Result<(), ()>
    {
        self.try_push(x).map_err(|_| ())
    }

    fn done(&mut self) -> Result<(), ()>
    {
        Ok(())
    }
}

macro_rules! do_try_set {
    ($T:ty, $self:expr, $s:expr, $map:expr, $map_mod:tt) => {
        {
            let this = $self;
            let mut s = $s;
            let mut map = $map;
            let mut old_m = <$T>::default();
            let mut m = <$T>::default().incremented();

            {
                let mut i = 0;
                #[allow(unused_mut)]
                let (mut a, mut b, h) = this.pushers();
                {
                    if let Some(ai) = s.next() {
                        map.insert(ai, <$T>::default());
                        let v = <$T>::default();
                        h.add(i, &v);
                        i += 1;
                        a.try_push(v).map_err(|_| ())?;
                    }
                }

                for si in s {
                    if old_m == <$T>::default() {
                        old_m.increment();
                    }
                    b.try_push(old_m).map_err(|_| ())?;
                    old_m = m.clone();

                    let ai = match map.entry(si) {
                        $map_mod::Entry::Occupied(e) => e.get().clone(),
                        $map_mod::Entry::Vacant(e) => {
                            let ai = m.clone();
                            e.insert(ai.clone());
                            m.increment();
                            ai
                        }
                    };
                    h.add(i, &ai);
                    i += 1;
                    a.try_push(ai).map_err(|_| ())?;
                }
                
                a.done()?;
                b.done()?;
                h.done(i);
                if i != 0 {
                    h.set_limit(&m);
                }
            }
            this.m = old_m;

            Ok(map)
        }
    }
}

macro_rules! impl_set_partition {
    ($SP:ty, $T:ty, {$($impl:tt)*}, {$($impl_a:tt)*}, {$($where:tt)*}) => {
        $($impl)* $SP
            $($where)* {
            /// Returns a reference to the sequence representing the partition
            pub fn get(&self) -> &[$T] {
                &self.a
            }

            /// Returns the size of the set being partitioned
            pub fn len(&self) -> usize {
                self.a.len()
            }

            /// Reset to the trivial set partition of size `self`.`len`()
            pub fn reset(&mut self) {
                self.do_reset();
                self.h.reset(self.a.len());
            }

            fn do_reset(&mut self) {
                let n = self.len() as usize;
                for i in self.a.iter_mut() {
                    *i = <$T>::default();
                }
                for i in self.b.iter_mut() {
                    *i = <$T>::default().incremented();
                }
                if n > 1 {
                    self.m = <$T>::default().incremented();
                } else {
                    self.m = <$T>::default();
                }
            }

            /// Move to the next set partition in lexicographic order of sequences,
            /// returning `true`, or to the first trivial partition, returning `false`.
            #[inline]
            pub fn increment(&mut self) -> bool {
                // algorithm from TAoCP 3b page 27
                let n = self.a.len();
                if let Some(al) = self.a.last_mut() {
                    if *al != self.m {
                        self.h.remove(n - 1, al);
                        al.increment();
                        self.h.add(n - 1, al);
                        if *al == self.m {
                            self.h.inc_limit();
                        }
                        return true;
                    }
                } else {
                    return false;
                }

                self.increment_slowpath()
            }

            fn increment_slowpath(&mut self) -> bool {
                let n = self.len();
                if n <= 1 {
                    return false;
                }

                let mut j = n - 2;
                while self.a[j] == self.b[j] {
                    j = j - 1;
                }
                if j == 0 {
                    self.reset();
                    return false;
                }

                for k in ((j + 1)..n).rev() {
                    let ak = &mut self.a[k];
                    self.h.remove(k, ak);
                }

                let m = {
                    let aj = &mut self.a[j];
                    self.h.remove(j, aj);
                    aj.increment();
                    self.h.add(j, aj);
                    let bj = self.b[j].clone();
                    if *aj == bj {bj.incremented()} else {bj}
                };
                j += 1;

                for k in j..n {
                    let ak = &mut self.a[k];
                    *ak = <$T>::default();
                    self.h.add(k, ak);
                }
                for bi in &mut self.b[j..] {
                    *bi = m.clone();
                }
                self.m = m;
                self.h.set_limit(&self.m);
                true
            }
        
            /// Create a set partition from an iterator of `Hash`-implementing elements.
            ///
            /// Returns the set partition and a map from iterator elements to representative values.
            ///
            /// Use `try_from_ord` instead if `Hash` is not implemented on the iterator elements.
            pub fn try_from<S>(s: S) -> Result<($SP, hash_map::HashMap<<S as Iterator>::Item, $T>), ()>
                where S: Iterator, <S as Iterator>::Item : Hash + Eq
            {
                let mut sp = <$SP>::new();
                sp.try_set(s).map(|map| (sp, map))
            }

            /// Sets a set partition from an iterator of `Hash`-implementing elements.
            ///
            /// Returns a map from iterator elements to representative values.
            ///
            /// Use `try_set_ord` instead if `Hash` is not implemented on the iterator elements.
            pub fn try_set<S>(&mut self, s: S) -> Result<hash_map::HashMap<<S as Iterator>::Item, $T>, ()>
                where S: Iterator, <S as Iterator>::Item : Hash + Eq
            {
                self.do_try_set(s).map_err(|e| {
                    self.reset_default();
                    e
                })
            }

            fn do_try_set<S>(&mut self, s: S) -> Result<hash_map::HashMap<<S as Iterator>::Item, $T>, ()>
                where S: Iterator, <S as Iterator>::Item : Hash + Eq
            {
                do_try_set!($T, self, s, hash_map::HashMap::new(), hash_map)
            }

            /// Create a set partition from an iterator of `Ord`-implementing elements.
            ///
            /// Returns the set partition and a map from iterator elements to representative values.
            ///
            /// Use `from` instead if `Hash` is implemented on the iterator elements.
            pub fn try_from_ord<S>(s: S) -> Result<($SP, btree_map::BTreeMap<<S as Iterator>::Item, $T>), ()>
                where S: Iterator, <S as Iterator>::Item : Ord
            {
                let mut sp = <$SP>::new();
                sp.try_set_ord(s).map(|map| (sp, map))
            }

            fn do_try_set_ord<S>(&mut self, s: S) -> Result<btree_map::BTreeMap<<S as Iterator>::Item, $T>, ()>
                where S: Iterator, <S as Iterator>::Item : Ord
            {
                do_try_set!($T, self, s, btree_map::BTreeMap::new(), btree_map)
            }

            /// Sets a set partition from an iterator of `Ord`-implementing elements.
            ///
            /// Returns a map from iterator elements to representative values.
            ///
            /// Use `set` instead if `Hash` is implemented on the iterator elements.
            pub fn try_set_ord<S>(&mut self, s: S) -> Result<btree_map::BTreeMap<<S as Iterator>::Item, $T>, ()>
                where S: Iterator, <S as Iterator>::Item : Ord
            {
                self.do_try_set_ord(s).map_err(|e| {
                    self.reset_default();
                    e
                })
            }

            /// Returns the number of non-empty subsets in the set partition.
            pub fn num_subsets(&self) -> $T {
                if let Some(al) = self.a.last() {
                    if *al == self.m {
                        return al.clone().incremented()
                    }
                }

                self.m.clone()
            }

            /// Returns the subsets data structure, usually representing the subsets if present.
            pub fn subsets(&self) -> &H {
                &self.h
            }
        }

        $($impl)* $SP
            $($where)* + PartialOrd<$T> {
            /// Create a set partition from a canonical representative sequence in an iterator.
            ///
            /// Returns Err(()) if the sequence is not a canonical representative sequence.
            pub fn try_from_repr<I>(iter: I) -> Result<$SP, ()>
                where I: Iterator<Item = $T>
            {
                let mut sp = <$SP>::new();
                sp.do_try_set_repr(iter).map(|_| sp)
            }

            /// Checks if a sequence is a canonical representative sequence.
            pub fn is_repr<I>(iter: I) -> bool
                where I: Iterator<Item = $T>
            {
                let mut m: $T = <$T>::default();
                let mut n = 0;
                let max_len = Self::max_len();
                for ai in iter {
                    n += 1;
                    if n > max_len {
                        return false;
                    }
                    if !(ai <= m) {
                        return false;
                    }
                    let ai1 = ai.incremented();
                    if ai1 > m {
                        m = ai1;
                    }
                }
                n >= Self::min_len()
            }

            /// Set the set partition to a canonical representative sequence in an iterator.
            ///
            /// Returns Err(()) if the sequence is not a canonical representative sequence,
            /// and also sets it to the default value
            pub fn try_set_repr<I>(&mut self, iter: I) -> Result<(), ()>
                where I: Iterator<Item = $T>
            {
                self.do_try_set_repr(iter).map_err(|e| {
                    self.reset_default();
                    e
                })
            }

            /// Set the set partition to a canonical representative sequence in an iterator.
            ///
            /// Returns Err(()) if the sequence is not a canonical representative sequence,
            /// and also sets it to the default value
            fn do_try_set_repr<I>(&mut self, mut iter: I) -> Result<(), ()>
                where I: Iterator<Item = $T>
            {
                let mut old_m = <$T>::default();
                let mut m = <$T>::default().incremented();

                {
                    let mut i = 0;
                    #[allow(unused_mut)]
                    let (mut a, mut b, h) = self.pushers();
                    {
                        if let Some(ai) = iter.next() {
                            if ai != <$T>::default() {
                                return Err(());
                            }
                            h.add(i, &ai);
                            i += 1;
                            a.try_push(ai).map_err(|_| ())?;
                        }
                    }
                    for ai in iter {
                        if old_m == <$T>::default() {
                            old_m.increment();
                        }
                        b.try_push(old_m).map_err(|_| ())?;

                        if !(ai <= m) {
                            return Err(());
                        }
                        h.add(i, &ai);
                        i += 1;
                        a.try_push(ai.clone()).map_err(|_| ())?;

                        old_m = m.clone();
                        let ai1 = ai.incremented();
                        if ai1 > m {
                            m = ai1;
                        }
                    }
                    a.done()?;
                    b.done()?;
                    h.done(i);
                    if i != 0 {
                        h.set_limit(&m);
                    }
                }
                self.m = old_m;

                Ok(())
            }
        }

        $($impl)* Incrementable for $SP
            $($where)* {
            fn increment(&mut self) {
                Self::increment(self);
            }
        }

        $($impl)* Default for $SP
            $($where)* {
            fn default() -> Self {
                Self::new()
            }
        }

        $($impl)* PartialOrd<$SP> for $SP
            $($where)* + PartialOrd {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.a.partial_cmp(&other.a)
            }

            fn lt(&self, other: &Self) -> bool {
                self.a.lt(&other.a)
            }

            fn gt(&self, other: &Self) -> bool {
                self.a.gt(&other.a)
            }

            fn le(&self, other: &Self) -> bool {
                self.a.le(&other.a)
            }

            fn ge(&self, other: &Self) -> bool {
                self.a.ge(&other.a)
            }
        }

        $($impl)* Ord for $SP
            $($where)* + Ord {
            fn cmp(&self, other: &Self) -> Ordering {
                self.a.cmp(&other.a)
            }
        }

        $($impl)* PartialEq<$SP> for $SP
            $($where)* {
            fn eq(&self, other: &Self) -> bool {
                self.a.eq(&other.a)
            }

            fn ne(&self, other: &Self) -> bool {
                self.a.ne(&other.a)
            }
        }

        $($impl)* Eq for $SP
            $($where)* + Eq {
        }
        
        $($impl)* Hash for $SP
            $($where)* + Hash {
            fn hash<HH: Hasher>(&self, state: &mut HH) {
                self.a.hash(state);
            }
        }

        $($impl)* Deref for $SP
            $($where)* {
            type Target = [$T];

            fn deref(&self) -> &Self::Target {
                self.get()
            }
        }

        $($impl)* AsRef<[$T]> for $SP
            $($where)* {
            fn as_ref(&self) -> &[$T] {
                self.get()
            }
        }

        $($impl)* Borrow<[$T]> for $SP
            $($where)* {
            fn borrow(&self) -> &[$T] {
                self.get()
            }
        }

        $($impl)* Index<usize> for $SP
            $($where)* {
            type Output = $T;

            fn index(&self, index: usize) -> &$T {
                &self.get()[index]
            }
        }

        $($impl)* Index<ops::Range<usize>> for $SP
            $($where)* {
            type Output = [$T];

            fn index(&self, index: ops::Range<usize>) -> &[$T] {
                Index::index(&**self, index)
            }
        }

        $($impl)* Index<ops::RangeTo<usize>> for $SP
            $($where)* {
            type Output = [$T];

            fn index(&self, index: ops::RangeTo<usize>) -> &[$T] {
                Index::index(&**self, index)
            }
        }

        $($impl)* Index<ops::RangeFrom<usize>> for $SP
            $($where)* {
            type Output = [$T];

            fn index(&self, index: ops::RangeFrom<usize>) -> &[$T] {
                Index::index(&**self, index)
            }
        }

        $($impl)* Index<ops::RangeFull> for $SP
            $($where)* {
            type Output = [$T];

            fn index(&self, index: ops::RangeFull) -> &[$T] {
                Index::index(&**self, index)
            }
        }

        $($impl_a)* IntoIterator for &'a $SP
            $($where)* {
            type Item = &'a $T;
            type IntoIter = slice::Iter<'a, $T>;

            fn into_iter(self) -> slice::Iter<'a, $T> {
                (&self.a).into_iter()
            }
        }

        $($impl_a)* IntoIterator for &'a mut $SP
            $($where)* {
            type Item = &'a mut $T;
            type IntoIter = slice::IterMut<'a, $T>;

            fn into_iter(self) -> slice::IterMut<'a, $T> {
                (&mut self.a).into_iter()
            }
        }
    }
}

/// Represents a set partition stored in a `Vec`
///
/// For small sizes, use `ArrayVecSetPartition` instead.
///
/// If the size is constant, use the appropriate `SetPartition#` struct instead for better performance.
#[derive(Debug, Clone)]
pub struct VecSetPartition<T = usize, H: Subsets<T> = ()>
    where T: Default + Incrementable + PartialEq<T> + Clone
{
    a: Vec<T>,
    b: Vec<T>,
    m: T,
    h: H
}

impl<T, H: Subsets<T>> VecSetPartition<T, H>
    where T: Default + Incrementable + PartialEq<T> + Clone
{
    /// Create the trivial set partition of size 0
    pub fn new() -> Self {
        VecSetPartition {a: Vec::new(), b: Vec::new(), m: T::default(), h: Default::default()}
    }

    /// Returns the minimum sequence length supported by this type
    pub fn min_len() -> usize {0}

    /// Returns the maximum sequence length supported by this type
    pub fn max_len() -> usize {usize::max_value()}

    /// Reset to the trivial set partition of the supported size
    pub fn reset_default(&mut self) {
        self.a.clear();
        self.b.clear();
        self.m = T::default();
    }

    /// Create the trivial set partition of size `n`
    pub fn with_size(n: usize) -> Self {
        let mut r: Self = VecSetPartition {a: vec![T::default(); n], b: vec![T::default().incremented(); if n > 0 {n - 1} else {0}], m: if n > 1 {T::default().incremented()} else {T::default()}, h: Default::default()};
        r.h.reset(n);
        r
    }

    /// Create the trivial set partition of size `n`
    pub fn try_with_size(n: usize) -> Result<VecSetPartition<T, H>, ()> {
        Ok(Self::with_size(n))
    }

    /// Reset to the trivial set partition of size `n`
    pub fn resize(&mut self, n: usize) {
        self.do_reset();
        self.a.resize(n, T::default());
        self.b.resize(if n > 0 {n - 1} else {0}, T::default().incremented());
        self.m = if n > 1 {T::default().incremented()} else {T::default()};
        self.h.reset(n);
    }

    /// Reset to the trivial set partition of size `n`
    pub fn try_resize(&mut self, n: usize) -> Result<(), ()> {
        self.resize(n);
        Ok(())
    }

    /// Returns the vector storing the representative sequence
    pub fn to_vec(self) -> Vec<T> {
        self.a
    }

    fn pushers(&mut self) -> (&mut Vec<T>, &mut Vec<T>, &mut H) {
        self.a.clear();
        self.b.clear();
        self.h.clear();
        (&mut self.a, &mut self.b, &mut self.h)
    }
}

impl<T, H: Subsets<T>> IntoIterator for VecSetPartition<T, H>
    where T: Default + Incrementable + PartialEq<T> + Clone {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> std::vec::IntoIter<T> {
        self.to_vec().into_iter()
    }
}

impl_set_partition!(VecSetPartition<T, H>, T, {impl<T, H: Subsets<T>>}, {impl<'a, T, H: Subsets<T>>}, {where T: Default + Incrementable + PartialEq<T> + Clone});

/// Represents a set partition stored in an `ArrayVec`.
///
/// For big sizes, use `VecSetPartition` instead, which uses a `Vec`.
///
/// If the size is constant, use the appropriate `SetPartition#` struct instead for better performance.
#[derive(Clone)]
pub struct ArrayVecSetPartition<T, H: Subsets<T>, const N: usize>
    where T: Default + Incrementable + PartialEq<T> + Clone
{
    a: ArrayVec<T, N>,
    b: ArrayVec<T, N>,
    m: T,
    h: H
}

// derive doesn't generate the correct where clause
impl<T, H: Subsets<T>, const N: usize> Debug for ArrayVecSetPartition<T, H, N>
    where H: Debug, T: Default + Incrementable + PartialEq<T> + Clone + Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ArrayVecSetPartition")
            .field("a", &self.a)
            .field("b", &self.b)
            .field("m", &self.m)
            .field("h", &self.h)
            .finish()
    }
}

impl<T, H: Subsets<T>, const N: usize> ArrayVecSetPartition<T, H, N>
    where T: Default + Incrementable + PartialEq<T> + Clone
{
    /// Create the trivial set partition of size 0
    pub fn new() -> Self {
        ArrayVecSetPartition {
            a: ArrayVec::new(),
            b: ArrayVec::new(),
            m: T::default(),
            h: Default::default()
        }
    }

    /// Returns the minimum sequence length supported by this type
    pub fn min_len() -> usize {0}

    /// Returns the maximum sequence length supported by this type
    pub fn max_len() -> usize {N}

    /// Reset to the trivial set partition of the supported size
    pub fn reset_default(&mut self) {
        self.a.clear();
        self.b.clear();
        self.m = T::default();
    }

    /// Create the trivial set partition of size `n`
    pub fn try_with_size(n: usize) -> Result<Self, ()> {
        let mut r = Self::new();
        r.try_resize(n).map(|_| r)
    }

    /// Reset to the trivial set partition of size `n`
    pub fn try_resize(&mut self, n: usize) -> Result<(), ()> {
        if n > self.a.capacity() {
            return Err(());
        }

        self.do_reset();
        if self.len() == 0 && n != 0 {
            self.a.push(T::default());
        }
        while self.len() > n {
            self.a.pop();
            self.b.pop();
        }
        while self.len() < n{
            self.a.push(T::default());
            self.b.push(T::default().incremented());
        }
        if n > 1 {
            self.m = T::default().incremented();
        } else {
            self.m = T::default();
        }
        self.h.reset(self.a.len());
        Ok(())
    }

    /// Returns the vector storing the representative sequence
    pub fn to_vec(self) -> ArrayVec<T, N> {
        self.a
    }

    fn pushers(&mut self) -> (&mut ArrayVec<T, N>, &mut ArrayVec<T, N>, &mut H) {
        self.a.clear();
        self.b.clear();
        self.h.clear();
        (&mut self.a, &mut self.b, &mut self.h)
    }
}

impl<T, H: Subsets<T>, const N: usize> IntoIterator for ArrayVecSetPartition<T, H, N>
    where T: Default + Incrementable + PartialEq<T> + Clone {
    type Item = T;
    type IntoIter = arrayvec::IntoIter<T, N>;

    fn into_iter(self) -> arrayvec::IntoIter<T, N> {
        self.to_vec().into_iter()
    }
}

impl_set_partition!(ArrayVecSetPartition<T, H, N>, T, {impl<T, H: Subsets<T>, const N: usize>}, {impl<'a, T, H: Subsets<T>, const N: usize>}, {where T: Default + Incrementable + PartialEq<T> + Clone});

/// Represents a set partition stored in a 16-entry `ArrayVec`.
///
/// For big sizes, use `VecSetPartition` instead, which uses a `Vec`.
///
/// If the size is constant, use the appropriate `SetPartition#` struct instead for better performance.
pub type GenSmallSetPartition<T>
    //where T: Default + Incrementable + PartialEq<T> + Clone
    = ArrayVecSetPartition<T, (), 16>;

/// Represents a set partition stored in a 16-entry `ArrayVec`.
///
/// For big sizes, use `VecSetPartition` instead, which uses a `Vec`.
///
/// If the size is constant, use the appropriate `SetPartition#` struct instead for better performance.
pub type SmallSetPartition = GenSmallSetPartition<u8>;

macro_rules! fixed_set_partition {
    ($t:tt, $tu8:tt, $an:expr, $bn:expr, $a:tt, $b:tt, $ac:expr, $bc:expr) => {
        /// Represents a fixed size set partition stored in a fixed-size array
        #[derive(Debug)]
        pub struct $t<T, H: Subsets<T> = ()>
            where T: Default + Incrementable + PartialEq<T> + Clone
        {
            a: [T; $an],
            b: [T; $bn],
            m: T,
            h: H
        }

        impl<T, H: Subsets<T>> $t<T, H>
            where T: Default + Incrementable + PartialEq<T> + Clone
        {
            /// Create the trivial set partition of the supported size
            pub fn new() -> Self {
                let mut r = $t {
                    a: $a,
                    b: $b,
                    m: if $an > 1 {T::default().incremented()} else {T::default()},
                    h: H::default()
                };
                r.h.reset($an);
                r
            }

            /// Returns the minimum sequence length supported by this type
            pub fn min_len() -> usize {$an}

            /// Returns the maximum sequence length supported by this type
            pub fn max_len() -> usize {$an}

            /// Reset to the trivial set partition of the supported size
            pub fn reset_default(&mut self) {
                *self = Self::new();
            }

            /// Returns the array storing the representative sequence
            pub fn to_array(self) -> [T; $an] {
                self.a
            }

            fn pushers(&mut self) -> (SlicePusher<T>, SlicePusher<T>, &mut H) {
                self.h.clear();
                (SlicePusher::new(&mut self.a), SlicePusher::new(&mut self.b), &mut self.h)
            }
        }

        impl<T, H: Subsets<T>> Clone for $t<T, H>
            where T: Default + Incrementable + PartialEq<T> + Clone
        {
            fn clone(&self) -> Self
            {
                let fa = $ac;
                let fb = $bc;
                $t {
                    a: fa(self),
                    b: fb(self),
                    m: self.m.clone(),
                    h: Default::default()
                }
            }
        }

        impl<T, H: Subsets<T>> IntoIterator for $t<T, H>
            where T: Default + Incrementable + PartialEq<T> + Clone {
            type Item = T;
            type IntoIter = arrayvec::IntoIter<T, $an>;

            fn into_iter(self) -> arrayvec::IntoIter<T, $an> {
                ArrayVec::from(self.to_array()).into_iter()
            }
        }

        impl_set_partition!($t<T, H>, T, {impl<T, H: Subsets<T>>}, {impl<'a, T, H: Subsets<T>>}, {where T: Default + Incrementable + PartialEq<T> + Clone});

        /// Represents a fixed size set partition stored in a fixed-size `u8` array
        #[derive(Debug, Clone)]
        pub struct $tu8<H: Subsets<u8> = ()>
        {
            a: [u8; $an],
            b: [u8; $bn],
            m: u8,
            h: H
        }

        impl<H: Subsets<u8>> $tu8<H>
        {
            /// Create the trivial set partition of the supported size
            pub fn new() -> Self {
                let mut r = $tu8 {
                    a: [0u8; $an],
                    b: [1u8; $bn],
                    m: if $an > 1 {1u8} else {0u8},
                    h: H::default()
                };
                r.h.reset($an);
                r
            }

            /// Returns the minimum sequence length supported by this type
            pub fn min_len() -> usize {$an}

            /// Returns the maximum sequence length supported by this type
            pub fn max_len() -> usize {$an}

            /// Reset to the trivial set partition of the supported size
            pub fn reset_default(&mut self) {
                *self = Self::new();
            }

            /// Returns the array storing the representative sequence
            pub fn to_array(self) -> [u8; $an] {
                self.a
            }

            fn pushers(&mut self) -> (SlicePusher<u8>, SlicePusher<u8>, &mut H) {
                self.h.clear();
                (SlicePusher::new(&mut self.a), SlicePusher::new(&mut self.b), &mut self.h)
            }
        }

        impl<H: Subsets<u8>> IntoIterator for $tu8<H>
        {
            type Item = u8;
            type IntoIter = arrayvec::IntoIter<u8, $an>;

            fn into_iter(self) -> arrayvec::IntoIter<u8, $an> {
                ArrayVec::from(self.to_array()).into_iter()
            }
        }

        impl_set_partition!($tu8<H>, u8, {impl<H: Subsets<u8>>}, {impl<'a, H: Subsets<u8>>}, {where u8: Default + Incrementable + PartialEq<u8> + Clone});
    }
}

// unfortunately we need all this boilerplate to avoid an unnecessary Copy bound on T...
fixed_set_partition!(GenSetPartition0, SetPartition0, 0, 0, [], [], |_: &GenSetPartition0<T, H>| [], |_: &GenSetPartition0<T, H>| []);
fixed_set_partition!(GenSetPartition1, SetPartition1, 1, 0, [T::default()], [], |s: &GenSetPartition1<T, H>| [s.a[0].clone()], |_: &GenSetPartition1<T, H>| []);
// generated with for i in `seq 2 32`; do echo "fixed_set_partition!(SetPartition$i, $i, $(($i - 1)), [$(for j in `seq 2 $i`; do echo -n "T::default(), "; done)T::default()], [$(for j in `seq 3 $i`; do echo -n "T::default().incremented(), "; done)T::default().incremented()], |s: &GenSetPartition$i<T, H>| [s.a[0].clone()$(for j in `seq 1 $(($i-1))`; do echo -n ", s.a[$j].clone()"; done)], |s: &GenSetPartition$i<T, H>| [s.b[0].clone()$(for j in `seq 1 $(($i-2))`; do echo -n ", s.b[$j].clone()"; done)]);"; done
fixed_set_partition!(GenSetPartition2, SetPartition2, 2, 1, [T::default(), T::default()], [T::default().incremented()], |s: &GenSetPartition2<T, H>| [s.a[0].clone(), s.a[1].clone()], |s: &GenSetPartition2<T, H>| [s.b[0].clone()]);
fixed_set_partition!(GenSetPartition3, SetPartition3, 3, 2, [T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented()], |s: &GenSetPartition3<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone()], |s: &GenSetPartition3<T, H>| [s.b[0].clone(), s.b[1].clone()]);
fixed_set_partition!(GenSetPartition4, SetPartition4, 4, 3, [T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition4<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone()], |s: &GenSetPartition4<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone()]);
fixed_set_partition!(GenSetPartition5, SetPartition5, 5, 4, [T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition5<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone()], |s: &GenSetPartition5<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone()]);
fixed_set_partition!(GenSetPartition6, SetPartition6, 6, 5, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition6<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone()], |s: &GenSetPartition6<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone()]);
fixed_set_partition!(GenSetPartition7, SetPartition7, 7, 6, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition7<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone()], |s: &GenSetPartition7<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone()]);
fixed_set_partition!(GenSetPartition8, SetPartition8, 8, 7, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition8<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone()], |s: &GenSetPartition8<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone()]);
fixed_set_partition!(GenSetPartition9, SetPartition9, 9, 8, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition9<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone()], |s: &GenSetPartition9<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone()]);
fixed_set_partition!(GenSetPartition10, SetPartition10, 10, 9, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition10<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone()], |s: &GenSetPartition10<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone()]);
fixed_set_partition!(GenSetPartition11, SetPartition11, 11, 10, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition11<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone()], |s: &GenSetPartition11<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone()]);
fixed_set_partition!(GenSetPartition12, SetPartition12, 12, 11, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition12<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone()], |s: &GenSetPartition12<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone()]);
fixed_set_partition!(GenSetPartition13, SetPartition13, 13, 12, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition13<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone()], |s: &GenSetPartition13<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone()]);
fixed_set_partition!(GenSetPartition14, SetPartition14, 14, 13, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition14<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone()], |s: &GenSetPartition14<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone()]);
fixed_set_partition!(GenSetPartition15, SetPartition15, 15, 14, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition15<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone()], |s: &GenSetPartition15<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone()]);
fixed_set_partition!(GenSetPartition16, SetPartition16, 16, 15, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition16<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone()], |s: &GenSetPartition16<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone()]);
fixed_set_partition!(GenSetPartition17, SetPartition17, 17, 16, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition17<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone()], |s: &GenSetPartition17<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone()]);
fixed_set_partition!(GenSetPartition18, SetPartition18, 18, 17, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition18<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone()], |s: &GenSetPartition18<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone()]);
fixed_set_partition!(GenSetPartition19, SetPartition19, 19, 18, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition19<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone()], |s: &GenSetPartition19<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone()]);
fixed_set_partition!(GenSetPartition20, SetPartition20, 20, 19, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition20<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone(), s.a[19].clone()], |s: &GenSetPartition20<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone(), s.b[18].clone()]);
fixed_set_partition!(GenSetPartition21, SetPartition21, 21, 20, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition21<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone(), s.a[19].clone(), s.a[20].clone()], |s: &GenSetPartition21<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone(), s.b[18].clone(), s.b[19].clone()]);
fixed_set_partition!(GenSetPartition22, SetPartition22, 22, 21, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition22<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone(), s.a[19].clone(), s.a[20].clone(), s.a[21].clone()], |s: &GenSetPartition22<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone(), s.b[18].clone(), s.b[19].clone(), s.b[20].clone()]);
fixed_set_partition!(GenSetPartition23, SetPartition23, 23, 22, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition23<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone(), s.a[19].clone(), s.a[20].clone(), s.a[21].clone(), s.a[22].clone()], |s: &GenSetPartition23<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone(), s.b[18].clone(), s.b[19].clone(), s.b[20].clone(), s.b[21].clone()]);
fixed_set_partition!(GenSetPartition24, SetPartition24, 24, 23, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition24<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone(), s.a[19].clone(), s.a[20].clone(), s.a[21].clone(), s.a[22].clone(), s.a[23].clone()], |s: &GenSetPartition24<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone(), s.b[18].clone(), s.b[19].clone(), s.b[20].clone(), s.b[21].clone(), s.b[22].clone()]);
fixed_set_partition!(GenSetPartition25, SetPartition25, 25, 24, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition25<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone(), s.a[19].clone(), s.a[20].clone(), s.a[21].clone(), s.a[22].clone(), s.a[23].clone(), s.a[24].clone()], |s: &GenSetPartition25<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone(), s.b[18].clone(), s.b[19].clone(), s.b[20].clone(), s.b[21].clone(), s.b[22].clone(), s.b[23].clone()]);
fixed_set_partition!(GenSetPartition26, SetPartition26, 26, 25, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition26<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone(), s.a[19].clone(), s.a[20].clone(), s.a[21].clone(), s.a[22].clone(), s.a[23].clone(), s.a[24].clone(), s.a[25].clone()], |s: &GenSetPartition26<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone(), s.b[18].clone(), s.b[19].clone(), s.b[20].clone(), s.b[21].clone(), s.b[22].clone(), s.b[23].clone(), s.b[24].clone()]);
fixed_set_partition!(GenSetPartition27, SetPartition27, 27, 26, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition27<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone(), s.a[19].clone(), s.a[20].clone(), s.a[21].clone(), s.a[22].clone(), s.a[23].clone(), s.a[24].clone(), s.a[25].clone(), s.a[26].clone()], |s: &GenSetPartition27<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone(), s.b[18].clone(), s.b[19].clone(), s.b[20].clone(), s.b[21].clone(), s.b[22].clone(), s.b[23].clone(), s.b[24].clone(), s.b[25].clone()]);
fixed_set_partition!(GenSetPartition28, SetPartition28, 28, 27, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition28<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone(), s.a[19].clone(), s.a[20].clone(), s.a[21].clone(), s.a[22].clone(), s.a[23].clone(), s.a[24].clone(), s.a[25].clone(), s.a[26].clone(), s.a[27].clone()], |s: &GenSetPartition28<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone(), s.b[18].clone(), s.b[19].clone(), s.b[20].clone(), s.b[21].clone(), s.b[22].clone(), s.b[23].clone(), s.b[24].clone(), s.b[25].clone(), s.b[26].clone()]);
fixed_set_partition!(GenSetPartition29, SetPartition29, 29, 28, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition29<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone(), s.a[19].clone(), s.a[20].clone(), s.a[21].clone(), s.a[22].clone(), s.a[23].clone(), s.a[24].clone(), s.a[25].clone(), s.a[26].clone(), s.a[27].clone(), s.a[28].clone()], |s: &GenSetPartition29<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone(), s.b[18].clone(), s.b[19].clone(), s.b[20].clone(), s.b[21].clone(), s.b[22].clone(), s.b[23].clone(), s.b[24].clone(), s.b[25].clone(), s.b[26].clone(), s.b[27].clone()]);
fixed_set_partition!(GenSetPartition30, SetPartition30, 30, 29, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition30<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone(), s.a[19].clone(), s.a[20].clone(), s.a[21].clone(), s.a[22].clone(), s.a[23].clone(), s.a[24].clone(), s.a[25].clone(), s.a[26].clone(), s.a[27].clone(), s.a[28].clone(), s.a[29].clone()], |s: &GenSetPartition30<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone(), s.b[18].clone(), s.b[19].clone(), s.b[20].clone(), s.b[21].clone(), s.b[22].clone(), s.b[23].clone(), s.b[24].clone(), s.b[25].clone(), s.b[26].clone(), s.b[27].clone(), s.b[28].clone()]);
fixed_set_partition!(GenSetPartition31, SetPartition31, 31, 30, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition31<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone(), s.a[19].clone(), s.a[20].clone(), s.a[21].clone(), s.a[22].clone(), s.a[23].clone(), s.a[24].clone(), s.a[25].clone(), s.a[26].clone(), s.a[27].clone(), s.a[28].clone(), s.a[29].clone(), s.a[30].clone()], |s: &GenSetPartition31<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone(), s.b[18].clone(), s.b[19].clone(), s.b[20].clone(), s.b[21].clone(), s.b[22].clone(), s.b[23].clone(), s.b[24].clone(), s.b[25].clone(), s.b[26].clone(), s.b[27].clone(), s.b[28].clone(), s.b[29].clone()]);
fixed_set_partition!(GenSetPartition32, SetPartition32, 32, 31, [T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default(), T::default()], [T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented(), T::default().incremented()], |s: &GenSetPartition32<T, H>| [s.a[0].clone(), s.a[1].clone(), s.a[2].clone(), s.a[3].clone(), s.a[4].clone(), s.a[5].clone(), s.a[6].clone(), s.a[7].clone(), s.a[8].clone(), s.a[9].clone(), s.a[10].clone(), s.a[11].clone(), s.a[12].clone(), s.a[13].clone(), s.a[14].clone(), s.a[15].clone(), s.a[16].clone(), s.a[17].clone(), s.a[18].clone(), s.a[19].clone(), s.a[20].clone(), s.a[21].clone(), s.a[22].clone(), s.a[23].clone(), s.a[24].clone(), s.a[25].clone(), s.a[26].clone(), s.a[27].clone(), s.a[28].clone(), s.a[29].clone(), s.a[30].clone(), s.a[31].clone()], |s: &GenSetPartition32<T, H>| [s.b[0].clone(), s.b[1].clone(), s.b[2].clone(), s.b[3].clone(), s.b[4].clone(), s.b[5].clone(), s.b[6].clone(), s.b[7].clone(), s.b[8].clone(), s.b[9].clone(), s.b[10].clone(), s.b[11].clone(), s.b[12].clone(), s.b[13].clone(), s.b[14].clone(), s.b[15].clone(), s.b[16].clone(), s.b[17].clone(), s.b[18].clone(), s.b[19].clone(), s.b[20].clone(), s.b[21].clone(), s.b[22].clone(), s.b[23].clone(), s.b[24].clone(), s.b[25].clone(), s.b[26].clone(), s.b[27].clone(), s.b[28].clone(), s.b[29].clone(), s.b[30].clone()]);

// bigger ones don't fit into u64
static BELL_NUMBERS: [u64; 26] = [
    1,
    1,
    2,
    5,
    15,
    52,
    203,
    877,
    4140,
    21147,
    115975,
    678570,
    4213597,
    27644437,
    190899322,
    1382958545,
    10480142147,
    82864869804,
    682076806159,
    5832742205057,
    51724158235372,
    474869816156751,
    4506715738447323,
    44152005855084346,
    445958869294805289,
    4638590332229999353
];

/// Number of partitions of a set of `n` elements.
///
/// Simply returns the `n`-th Bell number, or `None`. if it's too large to fit into `u64`.
pub fn set_partitions(n: usize) -> Option<u64>
{
    BELL_NUMBERS.get(n).map(|x| *x)
}

// for enumeration it's required and sufficient to test that we are enumerating all restricted growth sequences of length n in lexicographic order
// the number of test iterations varies due to the varying execution speeds of the test subsets loops
#[cfg(test)]
mod tests {
    macro_rules! test {
        ($T:ty, $new:expr) => {
            fn check_subsets(s: &$T) {
                super::check_subsets(s.get(), s.subsets());
            }

            fn range(n: usize) -> ::std::ops::Range<usize> {
                let a = <$T>::min_len();
                let mut b = <$T>::max_len();
                if n <= b {
                    b = n - 1;
                }
                a..(b+1)
            }

            #[test]
            fn subsets() {
                for n in range(9) {
                    let mut s = $new(n);

                    loop {
                        check_subsets(&s);
                        if !s.increment() {break;}
                    }
                    check_subsets(&s);
                }
            }

            #[test]
            fn zeroes() {
                for n in range(12) {
                    let mut s = $new(n);

                    assert_eq!(s.get().len(), n);
                    assert!(s.get().iter().all(|x| *x == Default::default()));

                    loop {
                        if !s.increment() {break;}
                    }

                    assert_eq!(s.get().len(), n);
                    assert!(s.get().iter().all(|x| *x == Default::default()));
                }
            }

            #[test]
            fn lexicographic() {
                for n in range(10) {
                    let mut s = $new(n);
                    let mut last = Vec::new();
                    loop {
                        if s.len() > 0 {
                            assert!(&*last < s.get());
                        }
                        last.clear();
                        last.extend_from_slice(s.get());

                        if !s.increment() {break;}
                    }
                }
            }

            #[test]
            fn restricted_growth_of_len_n() {
                use std::cmp::max;
                for n in range(11) {
                    let mut s = $new(n);

                    loop {
                        let mut m = 0;
                        assert_eq!(s.get().len(), n);

                        for ai in s.get() {
                            assert!(*ai <= m);
                            m = max(m, (*ai).clone() + 1);
                        }

                        if !s.increment() {break;}
                    }
                }
            }

            #[test]
            fn all() {
                for n in range(12) {
                    let mut s = $new(n);
                    let mut i = 0;
                    loop {
                        i += 1;
                        if !s.increment() {break;}
                    }

                    assert_eq!(Some(i), crate::set_partitions(s.len()));
                }
            }

            #[test]
            fn try_set_repr() {
                for n in range(9) {
                    let mut s = $new(n);
                    let mut sets = $new(n);
                    loop {
                        let seq = s.get().iter().map(|x| *x).collect::<Vec<_>>();
                        assert!(sets.try_set_repr(seq.into_iter()).is_ok());
                        assert_eq!((&s.a, &s.b, &s.m), (&sets.a, &sets.b, &sets.m));
                        check_subsets(&sets);
                        if !s.increment() {break;}
                    }
                }
            }

            #[test]
            fn try_set() {
                for n in range(9) {
                    let mut s = $new(n);
                    let mut sets = $new(n);
                    loop {
                        assert!(sets.try_set(s.get().iter()).is_ok());
                        assert_eq!((&s.a, &s.b, &s.m), (&sets.a, &sets.b, &sets.m));
                        println!("{} {:?}", n, &sets);
                        check_subsets(&sets);
                        if !s.increment() {break;}
                    }
                }
            }

            #[test]
            fn try_set_ord() {
                for n in range(9) {
                    let mut s = $new(n);
                    let mut sets = $new(n);
                    loop {
                        assert!(sets.try_set_ord(s.get().iter()).is_ok());
                        assert_eq!((&s.a, &s.b, &s.m), (&sets.a, &sets.b, &sets.m));
                        check_subsets(&sets);
                        if !s.increment() {break;}
                    }
                }
            }

            #[test]
            fn reset() {
                for n in range(9) {
                    let mut s = $new(n);
                    let new = $new(n);
                    loop {
                        let mut r = s.clone();
                        r.reset();
                        assert_eq!((&new.a, &new.b, &new.m), (&r.a, &r.b, &r.m));
                        check_subsets(&r);
                        if !s.increment() {break;}
                    }
                }
            }
        }
    }

    macro_rules! test_var_size {
        ($T:ty, $new:expr) => {
            test!($T, $new);

            #[test]
            fn len() {
                for n in range(12) {
                    let s = $new(n);
                    assert_eq!(s.len(), n);
                }
            }

            #[test]
            fn resize() {
                for n in range(8) {
                    let mut s = $new(n);
                    loop {
                        for m in 0..16 {
                            let mut r = s.clone();
                            assert!(r.try_resize(m).is_ok());
                            check_subsets(&r);
                            let sm = $new(m);
                            assert_eq!((&sm.a, &sm.b, &sm.m), (&r.a, &r.b, &r.m));
                        }
                        if !s.increment() {break;}
                    }
                }
            }
        }
    }

    macro_rules! test_subsets {
        ($S:ty, $E:ty) => {
            fn new_small(n: usize) -> crate::ArrayVecSetPartition<[$E; 16], $S> {
                crate::ArrayVecSetPartition::try_with_size(n).unwrap()
            }

            fn new<T: Default>(_: usize) -> T {
                T::default()
            }

            mod avecsp {test_var_size!(crate::ArrayVecSetPartition<[$E; 16], $S>, super::new_small);}
            mod vecsp {test_var_size!(crate::VecSetPartition<$E, $S>, crate::VecSetPartition::<$E, $S>::with_size);}

            // generated with for i in `seq 0 9`; do echo "mod sp$i {test!(::SetPartition$i<\$S>, super::new::<::SetPartition$i<\$S>>);} mod gsp$i {test!(::GenSetPartition$i<\$E, \$S>, super::new::<::GenSetPartition$i<\$E, \$S>>);}"; done
            mod sp0 {test!(crate::SetPartition0<$S>, super::new::<crate::SetPartition0<$S>>);} mod gsp0 {test!(crate::GenSetPartition0<$E, $S>, super::new::<crate::GenSetPartition0<$E, $S>>);}
            mod sp1 {test!(crate::SetPartition1<$S>, super::new::<crate::SetPartition1<$S>>);} mod gsp1 {test!(crate::GenSetPartition1<$E, $S>, super::new::<crate::GenSetPartition1<$E, $S>>);}
            mod sp2 {test!(crate::SetPartition2<$S>, super::new::<crate::SetPartition2<$S>>);} mod gsp2 {test!(crate::GenSetPartition2<$E, $S>, super::new::<crate::GenSetPartition2<$E, $S>>);}
            mod sp3 {test!(crate::SetPartition3<$S>, super::new::<crate::SetPartition3<$S>>);} mod gsp3 {test!(crate::GenSetPartition3<$E, $S>, super::new::<crate::GenSetPartition3<$E, $S>>);}
            mod sp4 {test!(crate::SetPartition4<$S>, super::new::<crate::SetPartition4<$S>>);} mod gsp4 {test!(crate::GenSetPartition4<$E, $S>, super::new::<crate::GenSetPartition4<$E, $S>>);}
            mod sp5 {test!(crate::SetPartition5<$S>, super::new::<crate::SetPartition5<$S>>);} mod gsp5 {test!(crate::GenSetPartition5<$E, $S>, super::new::<crate::GenSetPartition5<$E, $S>>);}
            mod sp6 {test!(crate::SetPartition6<$S>, super::new::<crate::SetPartition6<$S>>);} mod gsp6 {test!(crate::GenSetPartition6<$E, $S>, super::new::<crate::GenSetPartition6<$E, $S>>);}
            mod sp7 {test!(crate::SetPartition7<$S>, super::new::<crate::SetPartition7<$S>>);} mod gsp7 {test!(crate::GenSetPartition7<$E, $S>, super::new::<crate::GenSetPartition7<$E, $S>>);}
            mod sp8 {test!(crate::SetPartition8<$S>, super::new::<crate::SetPartition8<$S>>);} mod gsp8 {test!(crate::GenSetPartition8<$E, $S>, super::new::<crate::GenSetPartition8<$E, $S>>);}
            mod sp9 {test!(crate::SetPartition9<$S>, super::new::<crate::SetPartition9<$S>>);} mod gsp9 {test!(crate::GenSetPartition9<$E, $S>, super::new::<crate::GenSetPartition9<$E, $S>>);}
        }
    }

    mod noss {
        fn check_subsets<T>(_: &[T], _: &()) {}

        test_subsets!((), u16);
    }

    fn compute_subsets(s: &[u8]) -> Vec<Vec<u8>> {
        let mut r = Vec::new();
        r.resize(s.len(), Vec::new());
        let mut idx = 0;
        for i in s {
            r[*i as usize].push(idx);
            idx += 1;
        }

        loop {
            if let Some(last) = r.last_mut() {
                if !last.is_empty() {break;}
            } else {
                break;
            }
            r.pop();
        }
        r
    }

    mod vecss {
        fn check_subsets(s: &[u8], ss: &crate::VecSubsets<u8>) {
            let r = super::compute_subsets(s);

            assert_eq!(ss.subsets(), &*r);
        }

        test_subsets!(crate::VecSubsets<u8>, u8);
    }

    mod avecss {
        use ::arrayvec::ArrayVec;
        fn check_subsets(s: &[u8], ss: &crate::ArrayVecSubsets<[ArrayVec<[u8; 16]>; 16], [u8; 16]>) {
            let r: ArrayVec<[ArrayVec<[u8; 16]>; 16]> = super::compute_subsets(s).into_iter().map(|ss| ss.into_iter().collect()).collect();

            assert_eq!(ss.subsets(), &*r);
        }

        test_subsets!(crate::ArrayVecSubsets<[::arrayvec::ArrayVec<[u8; 16]>; 16], [u8; 16]>, u8);
    }

    mod hashss {
        use ::std::collections::HashSet;
        fn check_subsets(s: &[u8], ss: &crate::HashSubsets<u8>) {
            let r: Vec<HashSet<u8>> = super::compute_subsets(s).into_iter().map(|ss| ss.into_iter().collect()).collect();

            assert_eq!(ss.subsets(), &*r);
        }

        test_subsets!(crate::HashSubsets<u8>, u8);
    }

    mod btreess {
        use ::std::collections::BTreeSet;
        fn check_subsets(s: &[u8], ss: &crate::BTreeSubsets<u8>) {
            let r: Vec<BTreeSet<u8>> = super::compute_subsets(s).into_iter().map(|ss| ss.into_iter().collect()).collect();

            assert_eq!(ss.subsets(), &*r);
        }

        test_subsets!(crate::BTreeSubsets<u8>, u8);
    }

    #[test]
    fn set_partitions()
    {
        let mut bell_numbers = vec![1u64];
        let mut n = 0;

        assert_eq!(crate::set_partitions(0), Some(1));
        loop {
            if let Some(sp) = crate::set_partitions(n + 1) {
                let mut b = 0u64;
                for k in 0..(n+1) {
                    let mut c = 1u64;
                    for i in 1..k+1 {
                        c *= (n - k + i) as u64;
                        c /= i as u64;
                    }
                    b += bell_numbers[k] * c;
                }
                assert_eq!(b, sp);
                bell_numbers.push(b);
            } else {
                break;
            }      
            n += 1;
        }
    }
}
