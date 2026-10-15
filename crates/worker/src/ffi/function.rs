//! Python callables and calls made through the interpreter.

use tvm_ffi::{Any, AnyView, Array};

use super::interpreter;
use crate::{Error, Result};

/// A library function the worker imported by path, or one a library call
/// returned.
///
/// A function may be held and called by any thread. The call itself runs on
/// the rank's Python thread (see [`super::Interpreter`]) while the caller
/// waits for its result.
///
/// TVM-FFI functions are reference-counted objects with atomic counts. The
/// SDK does not mark the handle `Send`/`Sync` because it wraps raw pointers,
/// so this newtype asserts those properties for function objects only.
/// Tensor and other value handles are not covered.
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
        interpreter::call(&self.inner, args).map_err(|error| Error::numerical(&self.name, &error))
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

// Flat values of a call's arguments and results: scalars' absence and
// lists of integers, strings and opaque values.

/// Whether a value is Python's `None`.
pub fn is_none(value: &Any) -> bool {
    value.type_index() == tvm_ffi::TypeIndex::kTVMFFINone as i32
}

/// Convert a `Shape` or a list of integers.
pub fn integers(value: Any) -> std::result::Result<Vec<i64>, tvm_ffi::Error> {
    if value.type_index() == tvm_ffi::TypeIndex::kTVMFFIShape as i32 {
        return Ok(tvm_ffi::Shape::try_from(value)?.as_slice().to_vec());
    }
    let values: Array<Any> = Array::try_from(value)?;
    values.iter().map(i64::try_from).collect()
}

/// Convert a list of strings.
pub fn texts(value: Any) -> std::result::Result<Vec<String>, tvm_ffi::Error> {
    let values: Array<tvm_ffi::String> = Array::try_from(value)?;
    Ok(values
        .iter()
        .map(|value| value.as_str().to_owned())
        .collect())
}

/// A list of strings as a TVM-FFI array.
pub fn strings<S: AsRef<str>>(values: impl IntoIterator<Item = S>) -> Array<tvm_ffi::String> {
    values
        .into_iter()
        .map(|value| tvm_ffi::String::from(value.as_ref()))
        .collect()
}

/// A list of integers as a TVM-FFI array.
pub fn integer_list(values: impl IntoIterator<Item = i64>) -> Array<Any> {
    values.into_iter().map(Any::from).collect()
}

/// A list of arbitrary values as a TVM-FFI array.
pub fn list(values: impl IntoIterator<Item = Any>) -> Array<Any> {
    values.into_iter().collect()
}
