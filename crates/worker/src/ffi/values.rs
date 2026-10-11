//! Flat values exchanged with Python: scalars' absence and lists of
//! integers, strings and opaque values, in both directions.

use tvm_ffi::{Any, Array};

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
