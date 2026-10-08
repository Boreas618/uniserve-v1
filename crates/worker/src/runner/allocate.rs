//! Workspace allocations numerical contexts request on demand.
//!
//! Some call sites learn their workspace only at their first numerical
//! call (a sparse attention layer prepares its operators for the block
//! pattern it first sees). Their contexts request it through a callback
//! (`uniserve_models.interface.workspace.RuntimeWorkspace`) that allocates
//! from the binding stream's [`Workspaces`]: a persistent request gets its
//! own allocation; a shared request (`role` non-empty) borrows the field's
//! backing on that stream, grown when a larger request arrives. The returned
//! DLPack tensor retains its allocation for as long as Python references
//! it.

use std::sync::{Arc, Mutex, PoisonError};

use tvm_ffi::{Any, AnyView};

use crate::ffi::integers;
use crate::storage::memory::parse_dtype;
use crate::storage::{WorkspaceRequest, Workspaces};

/// The allocation callback of one stream's workspaces:
/// `allocate(name, role, dtype, shape, host) -> Tensor`.
pub(crate) fn allocator(workspaces: Arc<Mutex<Workspaces>>) -> tvm_ffi::Function {
    tvm_ffi::Function::from_packed(move |args: &[AnyView]| {
        let [name, role, dtype, shape, host] = args else {
            return Err(tvm_ffi::Error::new(
                tvm_ffi::TYPE_ERROR,
                "allocate takes name, role, dtype, shape and host",
                "",
            ));
        };
        let name = tvm_ffi::String::try_from(*name)?.as_str().to_owned();
        let role = tvm_ffi::String::try_from(*role)?.as_str().to_owned();
        let dtype = tvm_ffi::String::try_from(*dtype)?;
        let shape = integers(Any::from(*shape))?;
        let host = bool::try_from(*host)?;
        let request = WorkspaceRequest {
            name: name.clone(),
            dtype: parse_dtype(dtype.as_str()).map_err(|error| {
                tvm_ffi::Error::new(tvm_ffi::VALUE_ERROR, &error.to_string(), "")
            })?,
            shape,
            host,
            role: (!role.is_empty()).then_some(role),
        };
        let mut workspaces = workspaces.lock().unwrap_or_else(PoisonError::into_inner);
        let bound = workspaces
            .bind(&[request])
            .map_err(|error| tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, &error.to_string(), ""))?;
        let view = bound
            .into_iter()
            .next()
            .map(|(_, view)| view)
            .ok_or_else(|| {
                tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, "workspace was not bound", "")
            })?;
        let tensor = view
            .to_tensor()
            .map_err(|error| tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, &error.to_string(), ""))?;
        Ok(Any::from(tensor))
    })
}
