/// Helper functions to write CPU kernels.
use super::CpuExecutor;
use crate::nditer::NdIter;
use crate::{Error, Layout, Result, WithDType};

type C = super::CpuStorage;
pub trait Map1 {
    fn f<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Result<Vec<T>>;

    fn map(&self, vs: &C, layout: &Layout) -> Result<C> {
        match vs {
            C::U8(vs) => Ok(C::U8(self.f(vs, layout)?)),
            C::U32(vs) => Ok(C::U32(self.f(vs, layout)?)),
            C::I16(vs) => Ok(C::I16(self.f(vs, layout)?)),
            C::I32(vs) => Ok(C::I32(self.f(vs, layout)?)),
            C::I64(vs) => Ok(C::I64(self.f(vs, layout)?)),
            C::BF16(vs) => Ok(C::BF16(self.f(vs, layout)?)),
            C::F16(vs) => Ok(C::F16(self.f(vs, layout)?)),
            C::F32(vs) => Ok(C::F32(self.f(vs, layout)?)),
            C::F64(vs) => Ok(C::F64(self.f(vs, layout)?)),
            C::F8E4M3(vs) => Ok(C::F8E4M3(self.f(vs, layout)?)),
            // Two F4 elements share a byte: no host element type holds one.
            C::F4(_) => Err(Error::UnsupportedDTypeForOp(vs.dtype(), "map1").bt()),
        }
    }
}

pub trait Map1Any {
    fn f<T: WithDType, W: Fn(Vec<T>) -> C>(&self, vs: &[T], layout: &Layout, wrap: W) -> Result<C>;

    fn map(&self, vs: &C, layout: &Layout) -> Result<C> {
        match vs {
            C::U8(vs) => Ok(self.f(vs, layout, C::U8)?),
            C::U32(vs) => Ok(self.f(vs, layout, C::U32)?),
            C::I16(vs) => Ok(self.f(vs, layout, C::I16)?),
            C::I32(vs) => Ok(self.f(vs, layout, C::I32)?),
            C::I64(vs) => Ok(self.f(vs, layout, C::I64)?),
            C::BF16(vs) => Ok(self.f(vs, layout, C::BF16)?),
            C::F16(vs) => Ok(self.f(vs, layout, C::F16)?),
            C::F32(vs) => Ok(self.f(vs, layout, C::F32)?),
            C::F64(vs) => Ok(self.f(vs, layout, C::F64)?),
            C::F8E4M3(vs) => Ok(self.f(vs, layout, C::F8E4M3)?),
            // Two F4 elements share a byte: no host element type holds one.
            C::F4(_) => Err(Error::UnsupportedDTypeForOp(vs.dtype(), "map1any").bt()),
        }
    }
}

pub trait Map2 {
    const OP: &'static str;
    fn f<T: WithDType>(&self, v1: &[T], l1: &Layout, v2: &[T], l2: &Layout) -> Result<Vec<T>>;

    fn map(&self, v1: &C, l1: &Layout, v2: &C, l2: &Layout) -> Result<C> {
        match (v1, v2) {
            (C::U8(v1), C::U8(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::U32(v1), C::U32(v2)) => Ok(C::U32(self.f(v1, l1, v2, l2)?)),
            (C::I16(v1), C::I16(v2)) => Ok(C::I16(self.f(v1, l1, v2, l2)?)),
            (C::I32(v1), C::I32(v2)) => Ok(C::I32(self.f(v1, l1, v2, l2)?)),
            (C::I64(v1), C::I64(v2)) => Ok(C::I64(self.f(v1, l1, v2, l2)?)),
            (C::BF16(v1), C::BF16(v2)) => Ok(C::BF16(self.f(v1, l1, v2, l2)?)),
            (C::F16(v1), C::F16(v2)) => Ok(C::F16(self.f(v1, l1, v2, l2)?)),
            (C::F32(v1), C::F32(v2)) => Ok(C::F32(self.f(v1, l1, v2, l2)?)),
            (C::F64(v1), C::F64(v2)) => Ok(C::F64(self.f(v1, l1, v2, l2)?)),
            (C::F8E4M3(v1), C::F8E4M3(v2)) => Ok(C::F8E4M3(self.f(v1, l1, v2, l2)?)),
            _ => Err(Error::DTypeMismatchBinaryOp {
                lhs: v1.dtype(),
                rhs: v2.dtype(),
                op: Self::OP,
            }
            .bt()),
        }
    }
}

pub trait Map2InPlace {
    const OP: &'static str;
    fn f<T: WithDType>(&self, v1: &mut [T], l1: &Layout, v2: &[T], l2: &Layout) -> Result<()>;

    fn map(&self, v1: &mut C, l1: &Layout, v2: &C, l2: &Layout) -> Result<()> {
        match (v1, v2) {
            (C::U8(v1), C::U8(v2)) => self.f(v1, l1, v2, l2)?,
            (C::U32(v1), C::U32(v2)) => self.f(v1, l1, v2, l2)?,
            (C::I16(v1), C::I16(v2)) => self.f(v1, l1, v2, l2)?,
            (C::I32(v1), C::I32(v2)) => self.f(v1, l1, v2, l2)?,
            (C::I64(v1), C::I64(v2)) => self.f(v1, l1, v2, l2)?,
            (C::BF16(v1), C::BF16(v2)) => self.f(v1, l1, v2, l2)?,
            (C::F16(v1), C::F16(v2)) => self.f(v1, l1, v2, l2)?,
            (C::F32(v1), C::F32(v2)) => self.f(v1, l1, v2, l2)?,
            (C::F64(v1), C::F64(v2)) => self.f(v1, l1, v2, l2)?,
            (C::F8E4M3(v1), C::F8E4M3(v2)) => self.f(v1, l1, v2, l2)?,
            (v1, v2) => Err(Error::DTypeMismatchBinaryOp {
                lhs: v1.dtype(),
                rhs: v2.dtype(),
                op: Self::OP,
            }
            .bt())?,
        };
        Ok(())
    }
}

pub trait Map2U8 {
    const OP: &'static str;
    fn f<T: WithDType>(&self, v1: &[T], l1: &Layout, v2: &[T], l2: &Layout) -> Result<Vec<u8>>;

    fn map(&self, v1: &C, l1: &Layout, v2: &C, l2: &Layout) -> Result<C> {
        match (v1, v2) {
            (C::U8(v1), C::U8(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::U32(v1), C::U32(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::I16(v1), C::I16(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::I32(v1), C::I32(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::I64(v1), C::I64(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::BF16(v1), C::BF16(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::F16(v1), C::F16(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::F32(v1), C::F32(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::F64(v1), C::F64(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            (C::F8E4M3(v1), C::F8E4M3(v2)) => Ok(C::U8(self.f(v1, l1, v2, l2)?)),
            _ => Err(Error::DTypeMismatchBinaryOp {
                lhs: v1.dtype(),
                rhs: v2.dtype(),
                op: Self::OP,
            }
            .bt()),
        }
    }
}

pub fn binary_map<T: Copy, U: Copy, F: FnMut(T, T) -> U>(
    lhs_l: &Layout,
    rhs_l: &Layout,
    lhs: &[T],
    rhs: &[T],
    mut f: F,
) -> Vec<U> {
    let el_count = lhs_l.shape().elem_count();
    let mut result: Vec<U> = Vec::with_capacity(el_count);

    let nd_iter = NdIter::new([lhs_l, rhs_l]);
    let inner_size = nd_iter.inner_size;
    let [inner_ls, inner_rs] = nd_iter.inner_strides;

    for [lhs_off, rhs_off] in nd_iter {
        for i in 0..inner_size {
            result.push(f(lhs[lhs_off + i * inner_ls], rhs[rhs_off + i * inner_rs]));
        }
    }

    result
}

// Similar to binary_map but with vectorized variants.
pub fn binary_map_vec<
    T: Copy,
    F: FnMut(T, T) -> T,
    FV: FnMut(&[T], &[T], &mut [T]),
    FSV: FnMut(T, &[T], &mut [T]),
>(
    lhs_l: &Layout,
    rhs_l: &Layout,
    lhs: &[T],
    rhs: &[T],
    mut f: F,
    mut f_vec: FV,
    mut f_scalar_vec: FSV,
) -> Vec<T> {
    let el_count = lhs_l.shape().elem_count();
    let mut ys: Vec<T> = Vec::with_capacity(el_count);
    // SAFETY: the spare capacity is `el_count` elements of `T`, a plain numeric type; the loop
    // below writes each of them before `set_len` exposes them.
    let ys_to_set = unsafe {
        let s = ys.spare_capacity_mut();
        std::mem::transmute::<&mut [std::mem::MaybeUninit<T>], &mut [T]>(s)
    };

    let nd_iter = NdIter::new([lhs_l, rhs_l]);
    let inner_size = nd_iter.inner_size;
    let [inner_ls, inner_rs] = nd_iter.inner_strides;

    let mut dst_off = 0usize;

    for [lhs_off, rhs_off] in nd_iter {
        match (inner_ls, inner_rs) {
            (1, 1) => f_vec(
                &lhs[lhs_off..lhs_off + inner_size],
                &rhs[rhs_off..rhs_off + inner_size],
                &mut ys_to_set[dst_off..dst_off + inner_size],
            ),
            (1, 0) => {
                let r = rhs[rhs_off];
                f_scalar_vec(
                    r,
                    &lhs[lhs_off..lhs_off + inner_size],
                    &mut ys_to_set[dst_off..dst_off + inner_size],
                );
            }
            (0, 1) => {
                let l = lhs[lhs_off];
                for i in 0..inner_size {
                    ys_to_set[dst_off + i] = f(l, rhs[rhs_off + i]);
                }
            }
            _ => {
                for i in 0..inner_size {
                    ys_to_set[dst_off + i] =
                        f(lhs[lhs_off + i * inner_ls], rhs[rhs_off + i * inner_rs]);
                }
            }
        }
        dst_off += inner_size;
    }

    // SAFETY: all el_count elements have been written in the dispatch loop above.
    unsafe { ys.set_len(el_count) };
    ys
}

pub fn unary_map<T: Copy, U: Copy, F: FnMut(T) -> U>(
    vs: &[T],
    layout: &Layout,
    mut f: F,
) -> Vec<U> {
    match layout.strided_blocks() {
        crate::StridedBlocks::SingleBlock { start_offset, len } => vs
            [start_offset..start_offset + len]
            .iter()
            .map(|&v| f(v))
            .collect(),
        crate::StridedBlocks::UniformBlocks {
            start_offset,
            block_len,
            count,
            src_stride,
        } => {
            let mut result = Vec::with_capacity(count * block_len);
            if block_len == 1 {
                for i in 0..count {
                    // SAFETY: the strided blocks of `layout` address elements of `vs`, the storage
                    // it describes.
                    let v = unsafe { vs.get_unchecked(start_offset + i * src_stride) };
                    result.push(f(*v))
                }
            } else {
                for i in 0..count {
                    let src_start = start_offset + i * src_stride;
                    for offset in 0..block_len {
                        // SAFETY: the strided blocks of `layout` address elements of `vs`, the
                        // storage it describes.
                        let v = unsafe { vs.get_unchecked(src_start + offset) };
                        result.push(f(*v))
                    }
                }
            }
            result
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            let mut result = Vec::with_capacity(layout.shape().elem_count());
            // Specialize the case where block_len is one to avoid the second loop.
            if block_len == 1 {
                for index in block_start_index {
                    // SAFETY: the strided blocks of `layout` address elements of `vs`, the storage
                    // it describes.
                    let v = unsafe { vs.get_unchecked(index) };
                    result.push(f(*v))
                }
            } else {
                for index in block_start_index {
                    for offset in 0..block_len {
                        // SAFETY: the strided blocks of `layout` address elements of `vs`, the
                        // storage it describes.
                        let v = unsafe { vs.get_unchecked(index + offset) };
                        result.push(f(*v))
                    }
                }
            }
            result
        }
    }
}

pub fn unary_map_vec<T: Copy, U: Copy, F: FnMut(T) -> U, FV: FnMut(&[T], &mut [U])>(
    vs: &[T],
    layout: &Layout,
    mut f: F,
    mut f_vec: FV,
) -> Vec<U> {
    match layout.strided_blocks() {
        crate::StridedBlocks::SingleBlock { start_offset, len } => {
            let mut ys: Vec<U> = Vec::with_capacity(len);
            let ys_to_set = ys.spare_capacity_mut();
            // SAFETY: the spare capacity is `len` elements of `U`, a plain value type; `f_vec`
            // writes each of them before `set_len` exposes them.
            let ys_to_set = unsafe {
                std::mem::transmute::<&mut [std::mem::MaybeUninit<U>], &mut [U]>(ys_to_set)
            };
            f_vec(&vs[start_offset..start_offset + len], ys_to_set);
            // SAFETY: values are all set by f_vec.
            unsafe { ys.set_len(len) };
            ys
        }
        crate::StridedBlocks::UniformBlocks {
            start_offset,
            block_len,
            count,
            src_stride,
        } => {
            let el_count = count * block_len;
            if block_len == 1 {
                let mut result = Vec::with_capacity(count);
                for i in 0..count {
                    // SAFETY: the strided blocks of `layout` address elements of `vs`, the storage
                    // it describes.
                    let v = unsafe { vs.get_unchecked(start_offset + i * src_stride) };
                    result.push(f(*v))
                }
                result
            } else {
                let mut ys: Vec<U> = Vec::with_capacity(el_count);
                let ys_to_set = ys.spare_capacity_mut();
                // SAFETY: the spare capacity is `el_count` elements of `U`, a plain value type; the
                // calls of `f_vec` below write each of them before `set_len` exposes them.
                let ys_to_set = unsafe {
                    std::mem::transmute::<&mut [std::mem::MaybeUninit<U>], &mut [U]>(ys_to_set)
                };
                let mut dst_index = 0;
                for i in 0..count {
                    let src_start = start_offset + i * src_stride;
                    f_vec(
                        &vs[src_start..src_start + block_len],
                        &mut ys_to_set[dst_index..dst_index + block_len],
                    );
                    dst_index += block_len;
                }
                // SAFETY: values are all set by f_vec.
                unsafe { ys.set_len(el_count) };
                ys
            }
        }
        crate::StridedBlocks::MultipleBlocks {
            block_start_index,
            block_len,
        } => {
            let el_count = layout.shape().elem_count();
            // Specialize the case where block_len is one to avoid the second loop.
            if block_len == 1 {
                let mut result = Vec::with_capacity(el_count);
                for index in block_start_index {
                    // SAFETY: the strided blocks of `layout` address elements of `vs`, the storage
                    // it describes.
                    let v = unsafe { vs.get_unchecked(index) };
                    result.push(f(*v))
                }
                result
            } else {
                let mut ys: Vec<U> = Vec::with_capacity(el_count);
                let ys_to_set = ys.spare_capacity_mut();
                // SAFETY: the spare capacity is `el_count` elements of `U`, a plain value type; the
                // calls of `f_vec` below write each of them before `set_len` exposes them.
                let ys_to_set = unsafe {
                    std::mem::transmute::<&mut [std::mem::MaybeUninit<U>], &mut [U]>(ys_to_set)
                };
                let mut dst_index = 0;
                for src_index in block_start_index {
                    let vs = &vs[src_index..src_index + block_len];
                    let ys = &mut ys_to_set[dst_index..dst_index + block_len];
                    f_vec(vs, ys);
                    dst_index += block_len;
                }
                // SAFETY: values are all set by f_vec.
                unsafe { ys.set_len(el_count) };
                ys
            }
        }
    }
}

/// [`unary_map_vec`] split across the threads of `executor` when the layout is one block of
/// enough elements ([`ExecutorConfig::parallel_elements`]); any other layout is mapped on the
/// calling thread.
///
/// [`ExecutorConfig::parallel_elements`]: super::ExecutorConfig::parallel_elements
pub fn unary_map_vec_par<
    T: Copy + Send + Sync,
    U: Copy + Send + Sync,
    F: Fn(T) -> U + Sync,
    FV: Fn(&[T], &mut [U]) + Sync,
>(
    executor: &CpuExecutor,
    vs: &[T],
    layout: &Layout,
    f: F,
    f_vec: FV,
) -> Vec<U> {
    if let crate::StridedBlocks::SingleBlock { start_offset, len } = layout.strided_blocks()
        && executor.splits(len)
    {
        let src = &vs[start_offset..start_offset + len];
        let mut ys: Vec<U> = Vec::with_capacity(len);
        let ys_ptr = ys.as_mut_ptr() as usize;
        executor.for_each_chunk(len, |range| {
            let ys_ptr = ys_ptr as *mut U;
            // SAFETY: the chunk lies within the capacity `len`, and each chunk owns its disjoint
            // range of the output, which it writes before `set_len`.
            let dst =
                unsafe { std::slice::from_raw_parts_mut(ys_ptr.add(range.start), range.len()) };
            f_vec(&src[range], dst);
        });
        // SAFETY: every element is written by exactly one chunk.
        unsafe { ys.set_len(len) };
        return ys;
    }
    unary_map_vec(vs, layout, f, f_vec)
}

/// [`binary_map_vec`] split across the threads of `executor` when the output has enough elements
/// ([`ExecutorConfig::parallel_elements`]); a smaller output is mapped on the calling thread.
///
/// [`ExecutorConfig::parallel_elements`]: super::ExecutorConfig::parallel_elements
#[allow(clippy::too_many_arguments)]
#[expect(
    clippy::expect_used,
    reason = "a chunk's offsets are below `el_count`, which the `NdIter` of the same layouts walks \
              in blocks of `inner_size`"
)]
pub fn binary_map_vec_par<
    T: Copy + Send + Sync,
    F: Fn(T, T) -> T + Sync,
    FV: Fn(&[T], &[T], &mut [T]) + Sync,
    FSV: Fn(T, &[T], &mut [T]) + Sync,
>(
    executor: &CpuExecutor,
    lhs_l: &Layout,
    rhs_l: &Layout,
    lhs: &[T],
    rhs: &[T],
    f: F,
    f_vec: FV,
    f_scalar_vec: FSV,
) -> Vec<T> {
    let el_count = lhs_l.shape().elem_count();
    if !executor.splits(el_count) {
        return binary_map_vec(lhs_l, rhs_l, lhs, rhs, f, f_vec, f_scalar_vec);
    }

    let nd_iter = NdIter::new([lhs_l, rhs_l]);
    let inner_size = nd_iter.inner_size;
    let [lhs_stride, rhs_stride] = nd_iter.inner_strides;
    debug_assert!(inner_size > 0);

    let mut ys: Vec<T> = Vec::with_capacity(el_count);
    let ys_ptr = ys.as_mut_ptr() as usize;
    executor.for_each_chunk(el_count, |range| {
        let ys_ptr = ys_ptr as *mut T;
        let (mut dst_offset, hi) = (range.start, range.end);
        let mut inner_offset = dst_offset % inner_size;
        let mut nd_iter = NdIter::new([lhs_l, rhs_l]);
        let mut offsets = nd_iter
            .nth(dst_offset / inner_size)
            .expect("logical output offset must map to an NdIter block");

        while dst_offset < hi {
            let len = (inner_size - inner_offset).min(hi - dst_offset);
            let lhs_offset = offsets[0] + inner_offset * lhs_stride;
            let rhs_offset = offsets[1] + inner_offset * rhs_stride;
            // SAFETY: chunks own disjoint output ranges, all written before set_len.
            let dst = unsafe { std::slice::from_raw_parts_mut(ys_ptr.add(dst_offset), len) };

            match (lhs_stride, rhs_stride) {
                (1, 1) => f_vec(
                    &lhs[lhs_offset..lhs_offset + len],
                    &rhs[rhs_offset..rhs_offset + len],
                    dst,
                ),
                (1, 0) => f_scalar_vec(rhs[rhs_offset], &lhs[lhs_offset..lhs_offset + len], dst),
                _ => {
                    for (i, dst) in dst.iter_mut().enumerate() {
                        *dst = f(
                            lhs[lhs_offset + i * lhs_stride],
                            rhs[rhs_offset + i * rhs_stride],
                        );
                    }
                }
            }

            dst_offset += len;
            inner_offset = 0;
            if dst_offset < hi {
                offsets = nd_iter
                    .next()
                    .expect("logical output range must stay within NdIter blocks");
            }
        }
    });
    // SAFETY: every element is written by exactly one chunk.
    unsafe { ys.set_len(el_count) };
    ys
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Shape;
    use crate::cpu_backend::ExecutorConfig;

    /// The chunk of [`splitting_executor`]: a few dozen elements cross chunk boundaries inside
    /// and between rows.
    const CHUNK: usize = 5;

    /// An executor of its own that splits every map into chunks of [`CHUNK`] elements.
    fn splitting_executor() -> Result<CpuExecutor> {
        let config = ExecutorConfig {
            threads: 3,
            parallel_elements: 1,
            chunk_elements: CHUNK,
            ..ExecutorConfig::for_process()?
        };
        CpuExecutor::start(config)
    }

    fn difference_vec(lhs: &[i64], rhs: &[i64], ys: &mut [i64]) {
        for ((&lhs, &rhs), y) in lhs.iter().zip(rhs).zip(ys) {
            *y = lhs - rhs;
        }
    }

    fn difference_scalar_vec(scalar: i64, xs: &[i64], ys: &mut [i64]) {
        for (&x, y) in xs.iter().zip(ys) {
            *y = x - scalar;
        }
    }

    #[test]
    fn binary_map_vec_par_splits_non_contiguous_broadcasts() -> Result<()> {
        const ROWS: usize = 3;
        const COLS: usize = 2 * CHUNK + 2;
        const LHS_ROW_STRIDE: usize = COLS + 1;

        let mut lhs = vec![i64::MAX; ROWS * LHS_ROW_STRIDE];
        for row in 0..ROWS {
            for col in 0..COLS {
                lhs[row * LHS_ROW_STRIDE + col] = (row * COLS + col) as i64;
            }
        }
        let rhs = [11i64, 29, 47];
        let shape = Shape::from((ROWS, COLS));
        let lhs_l = Layout::new(shape.clone(), vec![LHS_ROW_STRIDE, 1], 0);
        let rhs_l = Layout::new(shape, vec![1, 0], 0);

        let result = binary_map_vec_par(
            &splitting_executor()?,
            &lhs_l,
            &rhs_l,
            &lhs,
            &rhs,
            |lhs, rhs| lhs - rhs,
            difference_vec,
            difference_scalar_vec,
        );

        let expected: Vec<_> = (0..ROWS)
            .flat_map(|row| (0..COLS).map(move |col| (row * COLS + col) as i64 - rhs[row]))
            .collect();
        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn binary_map_vec_par_splits_two_strided_inputs() -> Result<()> {
        const ROWS: usize = 2;
        const COLS: usize = 2 * CHUNK + 3;
        const LHS_ROW_STRIDE: usize = COLS + 1;
        const RHS_ROW_STRIDE: usize = COLS + 2;

        let mut lhs = vec![0i64; ROWS * LHS_ROW_STRIDE];
        let mut rhs = vec![0i64; ROWS * RHS_ROW_STRIDE];
        for row in 0..ROWS {
            for col in 0..COLS {
                lhs[row * LHS_ROW_STRIDE + col] = (row * COLS + col) as i64;
                rhs[row * RHS_ROW_STRIDE + col] = (2 * row * COLS + col) as i64;
            }
        }
        let shape = Shape::from((COLS, ROWS));
        let lhs_l = Layout::new(shape.clone(), vec![1, LHS_ROW_STRIDE], 0);
        let rhs_l = Layout::new(shape, vec![1, RHS_ROW_STRIDE], 0);

        let result = binary_map_vec_par(
            &splitting_executor()?,
            &lhs_l,
            &rhs_l,
            &lhs,
            &rhs,
            |lhs, rhs| lhs - rhs,
            difference_vec,
            difference_scalar_vec,
        );

        let lhs = &lhs;
        let rhs = &rhs;
        let expected: Vec<_> = (0..COLS)
            .flat_map(|col| {
                (0..ROWS).map(move |row| {
                    lhs[row * LHS_ROW_STRIDE + col] - rhs[row * RHS_ROW_STRIDE + col]
                })
            })
            .collect();
        assert_eq!(result, expected);
        Ok(())
    }

    #[test]
    fn unary_map_vec_par_splits_one_block() -> Result<()> {
        let vs: Vec<u32> = (0..4 * CHUNK as u32 + 3).collect();
        let layout = Layout::contiguous_with_offset(vs.len() - 2, 2);

        let result = unary_map_vec_par(
            &splitting_executor()?,
            &vs,
            &layout,
            |v| v * 3,
            |xs, ys| {
                for (&x, y) in xs.iter().zip(ys) {
                    *y = x * 3;
                }
            },
        );

        let expected: Vec<_> = vs[2..].iter().map(|v| v * 3).collect();
        assert_eq!(result, expected);
        Ok(())
    }
}
