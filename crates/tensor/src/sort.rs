use crate::cpu_backend::CpuExecutor;
use crate::tensor::from_storage;
use crate::{Result, Storage, Tensor};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy)]
struct ArgSort {
    asc: bool,
    last_dim: usize,
}

impl ArgSort {
    fn asort<T: crate::WithDType>(&self, vs: &[T], layout: &crate::Layout) -> Result<Vec<u32>> {
        let vs = match layout.contiguous_offsets() {
            None => crate::bail!("input has to be contiguous"),
            Some((o1, o2)) => &vs[o1..o2],
        };
        #[allow(clippy::uninit_vec)]
        // Safety: indexes are set later in the parallelized section.
        let mut sort_indexes = unsafe {
            let el_count = layout.shape().elem_count();
            let mut v = Vec::with_capacity(el_count);
            v.set_len(el_count);
            v
        };
        if self.asc {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&i, &j| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        } else {
            sort_indexes
                .par_chunks_exact_mut(self.last_dim)
                .zip(vs.par_chunks_exact(self.last_dim))
                .for_each(|(indexes, vs)| {
                    indexes
                        .iter_mut()
                        .enumerate()
                        .for_each(|(i, v)| *v = i as u32);
                    indexes.sort_by(|&j, &i| {
                        vs[i as usize]
                            .partial_cmp(&vs[j as usize])
                            .unwrap_or(std::cmp::Ordering::Greater)
                    })
                });
        }
        Ok(sort_indexes)
    }
}

impl ArgSort {
    /// The sorting indices of `storage` viewed through `layout`, along its last dimension, as a
    /// U32 storage of the layout's shape.
    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<crate::CpuStorage> {
        let sort_indexes = match storage {
            crate::CpuStorage::U8(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::U32(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::I16(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::I32(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::I64(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::BF16(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::F16(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::F32(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::F64(vs) => self.asort(vs, layout)?,
            crate::CpuStorage::F8E4M3(vs) => self.asort(vs, layout)?,
            // Two F4 elements share a byte: no host element type holds one to compare.
            crate::CpuStorage::F4(_) => {
                return Err(crate::Error::UnsupportedDTypeForOp(crate::DType::F4, "argsort").bt());
            }
        };
        Ok(crate::CpuStorage::U32(sort_indexes))
    }
}

impl Tensor {
    /// Returns the indices that sort the tensor along the last dimension.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    ///
    /// The sort is computed by the CPU backend only: on a tensor of a backend device this is an
    /// error.
    pub fn arg_sort_last_dim(&self, asc: bool) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "arg_sort_last_dim",
            });
        }
        let last_dim = match self.dims().last() {
            None => crate::bail!("empty last-dim in arg-sort"),
            Some(last_dim) => *last_dim,
        };
        let sort = ArgSort { asc, last_dim };
        let storage = match &*self.storage() {
            // The rows are sorted in parallel on the CPU executor's pool.
            Storage::Cpu(storage) => {
                Storage::Cpu(CpuExecutor::current()?.run(|| sort.cpu_fwd(storage, self.layout()))?)
            }
            Storage::Backend(_) => crate::bail!("no backend implementation for argsort"),
        };
        Ok(from_storage(storage, self.shape().clone()))
    }

    /// Sorts the tensor along the last dimension, returns the sorted tensor together with the
    /// sorted indexes.
    ///
    /// If `asc` is `true`, sorting is in ascending order. Otherwise sorting is performed in
    /// descending order. The sort is unstable so there is no guarantees on the final order when it
    /// comes to ties.
    pub fn sort_last_dim(&self, asc: bool) -> Result<(Tensor, Tensor)> {
        if !self.is_contiguous() {
            return Err(crate::Error::RequiresContiguous {
                op: "sort_last_dim",
            });
        }
        let asort = self.arg_sort_last_dim(asc)?;
        let sorted = self.gather(&asort, crate::D::Minus1)?;
        Ok((sorted, asort))
    }
}
