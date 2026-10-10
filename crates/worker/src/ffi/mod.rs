//! TVM-FFI values exchanged with the Python numerical library.
//!
//! The worker receives exactly one value from Python at startup: the
//! interface table, a map from `<area>.<verb>` names to functions. Loading and
//! binding functions return opaque handles and per-component maps of further
//! functions. Everything Rust sends back is a TVM-FFI value: integers, floats,
//! strings, shapes, arrays, maps, tensors and the handles it was given.
//!
//! Python callbacks acquire the GIL for the duration of a call, so the
//! wrappers here are callable from any thread. Callers must not hold a lock
//! while calling into Python: a callback may block on work that needs it.

mod dlpack;
mod record;
mod region;

use tvm_ffi::{Any, AnyView};

use crate::{Error, Result};

pub use dlpack::{Backing, DType, Device, contiguous_strides, export};
pub use record::{Fields, Record, integer_list, integers, is_none, json, list, strings, to_json};
pub use region::Region;
pub use tvm_ffi::{Array, Map, Shape, Tensor};

/// A function from the interface table or a binding map.
///
/// TVM-FFI functions are reference-counted objects with atomic counts whose
/// call ABI is thread-safe; Python-backed functions acquire the GIL inside
/// the call. The SDK does not mark the handle `Send`/`Sync` because it wraps
/// raw pointers, so this newtype asserts those properties for function
/// objects only. Tensor and other value handles are not covered.
#[derive(Clone)]
pub struct Function {
    name: std::sync::Arc<str>,
    inner: tvm_ffi::Function,
}

// SAFETY: see the type documentation; function objects are immutable after
// construction and their reference counts are atomic.
unsafe impl Send for Function {}
// SAFETY: as above.
unsafe impl Sync for Function {}

impl Function {
    pub fn new(name: impl Into<std::sync::Arc<str>>, inner: tvm_ffi::Function) -> Self {
        Self {
            name: name.into(),
            inner,
        }
    }

    /// The table or binding key this function was resolved from.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Call with packed arguments. A Python exception becomes
    /// [`Error::Numerical`] naming this function.
    pub fn call(&self, args: &[AnyView<'_>]) -> Result<Any> {
        self.inner
            .call_packed(args)
            .map_err(|error| Error::numerical(&self.name, &error))
    }

    /// Call and convert the returned value.
    pub fn call_as<T>(&self, args: &[AnyView<'_>]) -> Result<T>
    where
        T: TryFrom<Any, Error = tvm_ffi::Error>,
    {
        let value = self.call(args)?;
        T::try_from(value).map_err(|error| Error::numerical(&self.name, &error))
    }

    /// Call a function that returns nothing.
    pub fn invoke(&self, args: &[AnyView<'_>]) -> Result<()> {
        self.call(args).map(drop)
    }

    /// Call with owned argument values.
    pub fn call_values(&self, args: &[Any]) -> Result<Any> {
        let views: Vec<AnyView<'_>> = args.iter().map(AnyView::from).collect();
        self.call(&views)
    }

    /// The function as a value, to pass to another function.
    pub fn to_any(&self) -> Any {
        Any::from(self.inner.clone())
    }
}

impl std::fmt::Debug for Function {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_tuple("Function").field(&self.name).finish()
    }
}

/// An opaque value produced by Python (a loaded model, a process group, a
/// captured graph) that Rust only stores and passes back.
///
/// Dropping the last reference releases the Python object; TVM-FFI acquires
/// the GIL for that release.
#[derive(Clone)]
pub struct Handle(Any);

// SAFETY: a handle is an immutable reference-counted TVM-FFI object with
// atomic counts. Rust never dereferences it; Python accesses it under the GIL.
unsafe impl Send for Handle {}
// SAFETY: as above.
unsafe impl Sync for Handle {}

impl Handle {
    /// Hold `value` as an opaque handle.
    ///
    /// A handle may move to and be shared with other threads, so a tensor is
    /// refused with [`Error::TensorHandle`]: a TVM-FFI tensor stays on the
    /// thread that received it, and tensor data crosses threads only as
    /// Rust-owned storage.
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

    pub fn view(&self) -> AnyView<'_> {
        AnyView::from(&self.0)
    }

    /// The handle as an argument value.
    pub fn to_any(&self) -> Any {
        self.0.clone()
    }

    /// Whether Python returned `None` instead of an object.
    pub fn is_none(&self) -> bool {
        is_none(&self.0)
    }
}

impl std::fmt::Debug for Handle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Handle")
    }
}

/// A function and its owned arguments, made later, possibly on another
/// thread (a microbatch's numerical call).
pub struct DeferredCall {
    function: Function,
    args: Vec<Any>,
}

// SAFETY: the arguments are integers, shapes, strings and handles: immutable
// reference-counted TVM-FFI objects with atomic counts, which Rust never
// dereferences; Python accesses handles under the GIL. Tensor views, which
// must not cross threads, are never deferred.
unsafe impl Send for DeferredCall {}

impl DeferredCall {
    pub fn new(function: Function, args: Vec<Any>) -> Self {
        Self { function, args }
    }

    /// Make the call and drop the result.
    pub fn invoke(self) -> Result<()> {
        self.function.call_values(&self.args).map(drop)
    }
}

/// A map of functions returned by a Python loading or binding function.
pub struct Functions {
    owner: String,
    functions: Map<tvm_ffi::String, tvm_ffi::Function>,
}

impl Functions {
    /// `owner` names the producing function for error messages, for example
    /// `text.bind`.
    pub fn new(owner: impl Into<String>, value: Any) -> Result<Self> {
        let owner = owner.into();
        let functions = Map::try_from(value).map_err(|error| Error::numerical(&owner, &error))?;
        Ok(Self { owner, functions })
    }

    /// Wrap a map received as a typed argument.
    pub fn from_map(
        owner: impl Into<String>,
        functions: Map<tvm_ffi::String, tvm_ffi::Function>,
    ) -> Self {
        Self {
            owner: owner.into(),
            functions,
        }
    }

    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// A required function. Startup fails naming the missing key.
    pub fn get(&self, key: &str) -> Result<Function> {
        self.optional(key)?.ok_or_else(|| {
            Error::Interface(format!("{} did not provide function `{key}`", self.owner))
        })
    }

    pub fn optional(&self, key: &str) -> Result<Option<Function>> {
        let name = format!("{}.{key}", self.owner);
        self.functions
            .get(&tvm_ffi::String::from(key))
            .map(|function| function.map(|function| Function::new(name.as_str(), function)))
            .map_err(|error| Error::numerical(&self.owner, &error))
    }

    pub fn keys(&self) -> impl Iterator<Item = String> + '_ {
        self.functions.keys().map(|key| key.as_str().to_owned())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::storage::memory::{Memory, StorageView};

    #[test]
    fn a_tensor_is_refused_as_an_opaque_handle() -> Result<()> {
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
