//! The tensor ABI `nn` shares with `ops`, and the storage a layer's weight view is carved from.
//!
//! The view types are not defined here. [`TensorView`], [`Shape`], [`Strides`], [`DType`] and
//! [`DevicePtr`] are [`ops`] types, because they are the ABI of an operator's arguments and `ops`
//! depends on `platform` alone (specs/worker.md 4.2, 4.3); `nn` re-exports them so a model writes
//! `nn::TensorView`, not two names for one thing.
//!
//! What `nn` adds is [`Backing`]: the retained region a weight view borrows. The executor publishes
//! a converted weight to retained storage and binds the storage read-only; a layer receives the
//! view and the region together, so nothing in a layer has to assume a lifetime.

use crate::{Error, Result};

pub use ops::{DType, DevicePtr, MAX_RANK, Shape, Strides, TensorView};

/// The device storage a borrowed view is carved from.
///
/// The executor owns the storage (a read-only mapping of retained weights, or a region of the graph
/// pool) and keeps it alive for as long as the model is loaded; [`Backing`] is only the address and
/// size that prove a view's bytes are there. It owns nothing and copies nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Backing {
    address: DevicePtr,
    bytes: u64,
}

impl Backing {
    /// The region of `bytes` bytes at `address`.
    pub const fn new(address: DevicePtr, bytes: u64) -> Self {
        Self { address, bytes }
    }

    /// The region's first address.
    pub const fn address(self) -> DevicePtr {
        self.address
    }

    /// The region's size in bytes.
    pub const fn bytes(self) -> u64 {
        self.bytes
    }

    /// A row-major view of `shape` elements of `dtype` starting at the region's first address.
    ///
    /// Fails when the elements do not fit the region: a view that reaches past its backing is a
    /// loading defect, not a launch-time argument error.
    pub fn view(self, dtype: DType, shape: Shape) -> Result<TensorView> {
        let bytes = dtype.bytes().ok_or_else(|| {
            Error::Unsupported(format!(
                "{dtype} elements are sub-byte and named by their storage layout, not a view"
            ))
        })?;
        let needed = shape.numel().checked_mul(bytes).ok_or_else(|| {
            Error::shape(
                "Backing::view",
                format!("{shape:?} elements overflow a byte count"),
            )
        })?;
        if needed > self.bytes {
            return Err(Error::shape(
                "Backing::view",
                format!(
                    "{needed} bytes of {} {shape:?} do not fit a {}-byte region",
                    dtype, self.bytes
                ),
            ));
        }
        Ok(TensorView::new(self.address, dtype, shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_view_must_fit_its_backing() {
        let backing = Backing::new(DevicePtr::new(0x1000), 16);
        let view = backing.view(DType::F32, Shape::new(&[2, 2])).unwrap();
        assert_eq!(view.numel(), 4);
        assert_eq!(view.ptr.address(), 0x1000);
        assert!(backing.view(DType::F32, Shape::new(&[5])).is_err());
    }
}
