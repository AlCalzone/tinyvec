use core::slice;
use std::{
    borrow::{Borrow, BorrowMut},
    cmp,
    fmt::Debug,
    mem::{self, MaybeUninit},
    ops::{self, Deref, DerefMut},
    ptr,
    slice::SliceIndex,
};

/// A fixed-size array-backed Vec, which stores at most 255 elements.
/// Attempting to push() when the vec is at capacity will panic.
pub struct TinyVec<T, const N: usize> {
    data: [MaybeUninit<T>; N],
    len: u8,
}

impl<T, const N: usize> Default for TinyVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> TinyVec<T, N> {
    pub const fn new() -> Self {
        assert!(N >= 1, "N must be at least 1!");
        assert!(N < 256, "N must be less than 256!");
        Self {
            data: [const { MaybeUninit::uninit() }; N],
            len: 0,
        }
    }

    pub fn len(&self) -> u8 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn push(&mut self, value: T) {
        assert!(
            (self.len as usize) < N,
            "Cannot push to a TinyVec that is full!"
        );
        // SAFETY: We keep track of the length of the vec, and we did a bounds check before
        let entry = unsafe { self.data.get_unchecked_mut(self.len as usize) };
        entry.write(value);
        self.len += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        self.len -= 1;

        // SAFETY: We keep track of the length of the vec, and we did a bounds check before
        let entry = unsafe { self.data.get_unchecked_mut(self.len as usize) };

        // Swap the existing entry with uninitalized memory
        let mut ret = MaybeUninit::uninit();
        mem::swap(entry, &mut ret);
        // SAFETY: We just swapped an initialized value into ret
        let ret = unsafe { ret.assume_init() };

        // and return the value that previously was in the array
        Some(ret)
    }
}

impl<T, const N: usize> From<Vec<T>> for TinyVec<T, N> {
    fn from(mut vec: Vec<T>) -> Self {
        assert!(vec.len() < N, "vec.len() must not exceed {}", N);
        let mut ret = Self::new();
        ret.len = vec.len() as u8;
        // SAFETY: We copy data of the same type, keep track of the length,
        // did a bounds check on the target, and prevent the vec from dropping its elements
        unsafe {
            // Prevent the vec from dropping its elements
            vec.set_len(0);
            // Copy the contents into our data array
            ptr::copy_nonoverlapping(
                vec.as_ptr(),
                ret.data.as_mut_ptr() as *mut T,
                ret.len as usize,
            );
        }
        ret
    }
}

impl<T, const N: usize, const M: usize> From<[T; M]> for TinyVec<T, N> {
    fn from(arr: [T; M]) -> Self {
        assert!(
            M <= N,
            "Cannot create TinyVec with size {} from array with size {}",
            N,
            M
        );
        let mut ret = Self::new();
        // SAFETY: We copy data of the same type, keep track of the length,
        // did a bounds check on the target, and prevent the array from dropping its elements
        unsafe {
            // Copy the contents into our data array
            ptr::copy_nonoverlapping(arr.as_ptr(), ret.data.as_mut_ptr() as *mut T, M);
            mem::forget(arr);
        }
        ret.len = M as u8;
        ret
    }
}

impl<T, const N: usize> Drop for TinyVec<T, N> {
    fn drop(&mut self) {
        // SAFETY: By creating a mutable slice, we only drop the elements that are initialized
        unsafe {
            ptr::drop_in_place(&mut self[..]);
        }
    }
}

impl<T, const N: usize> From<&[T]> for TinyVec<T, N>
where
    T: Clone,
{
    fn from(slice: &[T]) -> Self {
        assert!(slice.len() < N, "slice.len() must not exceed {}", N);
        Self {
            data: core::array::from_fn(|i| {
                slice
                    .get(i)
                    .map_or(MaybeUninit::uninit(), |v| MaybeUninit::new(v.clone()))
            }),
            len: slice.len() as u8,
        }
    }
}

impl<T, const N: usize, const M: usize> From<&[T; M]> for TinyVec<T, N>
where
    T: Clone,
{
    fn from(slice: &[T; M]) -> Self {
        Self::from(slice.clone())
    }
}

impl<T, const N: usize> Clone for TinyVec<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self::from(&self[..])
    }
}

impl<T, const N: usize> Deref for TinyVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        // SAFETY: We use a pointer to an array of the same type, and we know the number of initialized elements
        unsafe { slice::from_raw_parts(self.data.as_ptr() as *const T, self.len as usize) }
    }
}

impl<T, const N: usize> DerefMut for TinyVec<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: We use a pointer to an array of the same type, and we know the number of initialized elements
        unsafe { slice::from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, self.len as usize) }
    }
}

impl<T, const N: usize> AsRef<[T]> for TinyVec<T, N> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T, const N: usize> AsMut<[T]> for TinyVec<T, N> {
    fn as_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T, const N: usize> Borrow<[T]> for TinyVec<T, N> {
    fn borrow(&self) -> &[T] {
        self
    }
}

impl<T, const N: usize> BorrowMut<[T]> for TinyVec<T, N> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self
    }
}

impl<T, const N: usize, I: SliceIndex<[T]>> ops::Index<I> for TinyVec<T, N> {
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        &(**self)[index]
    }
}

impl<T, const N: usize, I: SliceIndex<[T]>> ops::IndexMut<I> for TinyVec<T, N> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        &mut (&mut **self)[index]
    }
}

impl<A, B, const N: usize, const M: usize> PartialEq<TinyVec<B, M>> for TinyVec<A, N>
where
    A: PartialEq<B>,
{
    fn eq(&self, other: &TinyVec<B, M>) -> bool {
        self[..] == other[..]
    }
}

impl<T, const N: usize> Eq for TinyVec<T, N> where T: Eq {}

impl<T, const N: usize> PartialOrd for TinyVec<T, N>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &TinyVec<T, N>) -> Option<cmp::Ordering> {
        PartialOrd::partial_cmp(self.as_ref(), other.as_ref())
    }
}

impl<T, const N: usize> Ord for TinyVec<T, N>
where
    T: Ord,
{
    fn cmp(&self, other: &TinyVec<T, N>) -> cmp::Ordering {
        Ord::cmp(self.as_ref(), other.as_ref())
    }
}

impl<T, const N: usize> Debug for TinyVec<T, N>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self[..].fmt(f)
    }
}

impl<T, const N: usize> Extend<T> for TinyVec<T, N> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.push(item);
        }
    }
}

unsafe impl<T, const N: usize> Send for TinyVec<T, N> where T: Send {}

pub struct IntoIter<T, const N: usize> {
    vec: TinyVec<T, N>,
    // Tracking the length separately allows using its data without
    // having to worry about the vec dropping its elements
    len: u8,
    pos: u8,
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == self.len {
            return None;
        }

        // SAFETY: We did a bounds check and we keep track of the length
        // to ensure we're reading initialized memory, and we only read each element once
        let ret = unsafe {
            self.vec
                .data
                .get_unchecked(self.pos as usize)
                .assume_init_read()
        };
        self.pos += 1;
        Some(ret)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.len as usize - self.pos as usize;
        (size, Some(size))
    }
}

impl<T, const N: usize> Drop for IntoIter<T, N> {
    fn drop(&mut self) {
        // Easiest is to just continue iterating until we reach the end
        for _ in self {}
    }
}

impl<T, const N: usize> IntoIterator for TinyVec<T, N> {
    type IntoIter = IntoIter<T, N>;
    type Item = T;

    fn into_iter(mut self) -> Self::IntoIter {
        let len = self.len;
        // Prevent the vec from dropping its elements that were already dropped while iterating
        self.len = 0;
        IntoIter {
            vec: self,
            len,
            pos: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use crate::TinyVec;

    #[test]
    fn test_push_pop() {
        let mut vec: TinyVec<u8, 3> = TinyVec::new();
        vec.push(8);
        assert_eq!(vec.pop(), Some(8));
        assert_eq!(vec.pop(), None);
        assert_eq!(vec.pop(), None);
        assert_eq!(vec.pop(), None);

        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.pop(), Some(3));
        assert_eq!(vec.pop(), Some(2));
        assert_eq!(vec.pop(), Some(1));
        assert_eq!(vec.pop(), None)
    }

    #[test]
    fn test_from_slice_n_m() {
        let vec = TinyVec::<u8, 3>::from(&[1u8, 2, 3]);
        assert_eq!(&*vec, &[1u8, 2, 3]);
    }

    #[test]
    #[should_panic]
    fn test_from_slice_too_large() {
        let _ = TinyVec::<u8, 3>::from(&[1u8, 2, 3, 4]);
    }

    #[test]
    fn test_from_arr_n_m() {
        let vec = TinyVec::<u8, 3>::from([1u8, 2]);
        assert_eq!(&vec[..], &[1u8, 2]);
    }

    #[test]
    #[should_panic]
    fn test_from_arr_too_large() {
        let _ = TinyVec::<u8, 3>::from([1u8, 2, 3, 4]);
    }

    #[test]
    fn test_from_vec() {
        let tvec = TinyVec::<u8, 3>::from(vec![1u8]);
        assert_eq!(&tvec[..], &[1u8]);
    }

    #[test]
    #[should_panic]
    fn test_from_vec_too_large() {
        let _ = TinyVec::<u8, 3>::from(vec![1u8, 2, 3, 4]);
    }

    #[test]
    #[should_panic]
    fn test_lower_bounds_check() {
        TinyVec::<u8, 0>::new();
    }

    #[test]
    #[should_panic]
    fn test_upper_bounds_check() {
        TinyVec::<u8, 256>::new();
    }

    #[test]
    fn test_deref() {
        let vec: TinyVec<u8, 3> = TinyVec::new();
        assert_eq!(&vec[..], &[]);
    }

    #[test]
    #[should_panic]
    fn test_write_oob() {
        let mut vec = TinyVec::<u8, 3>::new();
        vec[0] = 1;
    }

    #[test]
    fn test_write_at_index() {
        let mut vec = TinyVec::<u8, 3>::new();
        vec.push(2);
        vec[0] = 1;
        assert_eq!(&vec[..], &[1])
    }

    #[test]
    fn test_drop() {
        let mut counter: u8 = 0;
        struct Foo<'a> {
            counter: &'a mut u8,
        }
        impl Drop for Foo<'_> {
            fn drop(&mut self) {
                *self.counter += 1;
            }
        }

        let mut vec = TinyVec::<Foo, 3>::new();
        vec.push(Foo {
            counter: &mut counter,
        });
        drop(vec);
        assert_eq!(counter, 1)
    }

    #[test]
    fn test_debug() {
        let mut vec = TinyVec::<u8, 6>::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!("[1, 2, 3]", format!("{:?}", &vec));
    }

    #[test]
    fn test_clone() {
        let mut vec = TinyVec::<u8, 6>::new();
        vec.push(1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.clone().as_ref(), &[1, 2, 3]);
    }

    #[test]
    fn test_extend() {
        let mut vec = TinyVec::<u8, 6>::new();
        vec.extend([1u8, 2, 3]);
        assert_eq!(vec.as_ref(), &[1, 2, 3]);
    }

    #[test]
    fn test_into_iter() {
        let dropped_1: Cell<bool> = false.into();
        let dropped_2: Cell<bool> = false.into();
        let dropped_3: Cell<bool> = false.into();

        #[derive(PartialEq, Debug)]
        struct Item<'a>(u8, &'a Cell<bool>);

        impl Drop for Item<'_> {
            fn drop(&mut self) {
                if self.1.get() {
                    panic!("Item was dropped twice!");
                }
                self.1.set(true);
            }
        }

        let vec = TinyVec::<Item, 6>::from([
            Item(1, &dropped_1),
            Item(2, &dropped_2),
            Item(3, &dropped_3),
        ]);
        let mut iter = vec.into_iter();

        assert!(matches!(iter.next(), Some(Item(1, _))));
        assert!(dropped_1.get());
        assert!(!dropped_2.get());
        assert!(!dropped_3.get());

        assert!(matches!(iter.next(), Some(Item(2, _))));
        assert!(dropped_2.get());
        assert!(!dropped_3.get());

        drop(iter);
        assert!(dropped_3.get());
    }
}
