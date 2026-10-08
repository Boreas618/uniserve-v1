//! Typed windows of byte arenas bound to a binding once.
//!
//! A binding receives each device's product arena (the storage of every
//! cross-call tensor product on that device) once, as a `uint8` tensor.
//! Calls then name products by [`Region`]: the byte offset of the product's
//! storage in the arena, its dtype and its shape, crossing as the TVM-FFI
//! array `[offset, dtype, shape]` (`uniserve_models.interface.regions`).
//! Naming a product costs no DLPack import.

use tvm_ffi::Any;

use super::{DType, list};
use crate::storage::memory::StorageView;
use crate::{Error, Result};

/// A contiguous typed window of a bound arena.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Region {
    offset: u64,
    dtype: DType,
    shape: Vec<i64>,
}

impl Region {
    /// A window of `shape` elements of `dtype` at byte `offset`; the offset
    /// is aligned to the element size.
    pub fn new(offset: u64, dtype: DType, shape: Vec<i64>) -> Result<Self> {
        if !offset.is_multiple_of(dtype.bytes() as u64) || shape.iter().any(|&extent| extent < 0) {
            return Err(Error::Invalid(format!(
                "arena region at byte {offset} misaligns {} or has a negative extent",
                dtype.name()
            )));
        }
        Ok(Self {
            offset,
            dtype,
            shape,
        })
    }

    /// The region `view` occupies in `arena`, with the view's dtype and
    /// shape. The view must be contiguous and lie within the arena's bytes.
    pub fn within(arena: &StorageView, view: &StorageView) -> Result<Self> {
        let start = arena.address();
        let end = start + arena.nbytes();
        let address = view.address();
        if !view.is_contiguous()
            || view.device() != arena.device()
            || address < start
            || address + view.nbytes() > end
        {
            return Err(Error::Invariant(
                "a product view lies outside its device's bound arena".into(),
            ));
        }
        Self::new(address - start, view.dtype(), view.shape().to_vec())
    }

    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Bytes the region spans.
    pub fn bytes(&self) -> u64 {
        self.shape.iter().product::<i64>() as u64 * self.dtype.bytes() as u64
    }

    /// The region as the argument `[offset, dtype, shape]`.
    pub fn to_any(&self) -> Any {
        Any::from(list([
            Any::from(self.offset as i64),
            Any::from(tvm_ffi::String::from(self.dtype.name())),
            Any::from(tvm_ffi::Shape::from(self.shape.as_slice())),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::storage::memory::Memory;

    #[test]
    fn regions_locate_contiguous_views_inside_their_arena() -> Result<()> {
        let memory = Memory::host(256);
        let arena = StorageView::contiguous(Arc::clone(&memory) as _, 0, DType::U8, &[256])?;
        let product = StorageView::contiguous(memory as _, 64, DType::BF16, &[4, 8])?;

        let region = Region::within(&arena, &product)?;
        assert_eq!(region, Region::new(64, DType::BF16, vec![4, 8])?);
        assert_eq!(region.bytes(), 64);

        // A strided view cannot be named by offset and shape.
        assert!(Region::within(&arena, &product.permute(&[1, 0])?).is_err());
        assert!(Region::within(&product, &arena).is_err());
        assert!(Region::new(3, DType::F32, vec![1]).is_err());
        Ok(())
    }
}
