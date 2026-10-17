//! Opaque Python objects held by Rust.

use tvm_ffi::{Any, AnyView};

use super::{interpreter, is_none};
use crate::{Error, Result};

/// An opaque value produced by Python (a loaded model, a process group, a
/// captured graph, the torch tensor wrapping a view of Rust's storage) that
/// Rust only stores and passes back.
///
/// Dropping the last reference releases the Python object, which runs
/// Python. A handle dropped on another thread therefore submits its
/// reference to the Python thread, which drops it: objects are released in
/// the order their owners drop them, and the thread that drops one does not
/// wait for the release.
#[derive(Clone)]
pub struct Handle(Any);

impl Drop for Handle {
    fn drop(&mut self) {
        if !is_none(&self.0) {
            interpreter::release(std::mem::take(&mut self.0));
        }
    }
}

// SAFETY: a handle is an immutable reference-counted TVM-FFI object with
// atomic counts. Rust never dereferences it; Python accesses it under the GIL.
unsafe impl Send for Handle {}
// SAFETY: as above.
unsafe impl Sync for Handle {}

impl Handle {
    /// Hold `value` as an opaque handle.
    ///
    /// A handle may move to and be shared with other threads, so a TVM-FFI
    /// tensor is refused with [`Error::TensorHandle`]: it stays on the thread
    /// that received it, and tensor data crosses threads only as Rust-owned
    /// storage. A torch tensor returned through the import path arrives as
    /// an opaque Python object and is held like any other.
    pub fn new(value: Any) -> Result<Self> {
        if value.type_index() == tvm_ffi::TypeIndex::kTVMFFITensor as i32 {
            return Err(Error::TensorHandle);
        }
        Ok(Self(value))
    }

    /// The handle of no object, as Python's `None` crosses the interface.
    pub fn none() -> Self {
        Self(Any::default())
    }

    pub(crate) fn view(&self) -> AnyView<'_> {
        AnyView::from(&self.0)
    }

    /// The handle as an argument value.
    pub(crate) fn to_any(&self) -> Any {
        self.0.clone()
    }

    /// Whether Python returned `None` instead of an object.
    pub(crate) fn is_none(&self) -> bool {
        is_none(&self.0)
    }
}

impl std::fmt::Debug for Handle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Handle")
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::ffi::{Backing, DType};
    use crate::storage::memory::{Memory, StorageView};

    #[test]
    fn a_tvm_ffi_tensor_is_refused_as_an_opaque_handle() -> Result<()> {
        let memory: Arc<dyn Backing> = Memory::host(16);
        let tensor = StorageView::contiguous(memory, 0, DType::F32, &[4])?.to_tensor()?;

        assert!(matches!(
            Handle::new(Any::from(tensor)),
            Err(Error::TensorHandle)
        ));
        assert!(Handle::new(Any::from(7_i64)).is_ok());
        assert!(Handle::none().is_none());
        Ok(())
    }
}
