//! The configuration map a harness function receives from its Python test,
//! read by key with errors that name the function and the key. Each harness
//! library declares this module public, so the readers it does not call
//! remain part of its interface.

use tvm_ffi::{Any, Array, Map};
use worker::ffi::is_none;
use worker::{Error, Result};

/// The keyword configuration of one harness call.
pub struct Config {
    /// The harness function, named in errors.
    owner: String,
    fields: Map<tvm_ffi::String, Any>,
}

impl Config {
    pub fn new(owner: impl Into<String>, value: Any) -> Result<Self> {
        let owner = owner.into();
        let fields = Map::try_from(value).map_err(|error| Error::numerical(&owner, &error))?;
        Ok(Self { owner, fields })
    }

    /// The raw value of a required field.
    pub fn value(&self, key: &str) -> Result<Any> {
        self.optional_value(key)?
            .ok_or_else(|| self.mismatch(key, "is missing"))
    }

    /// The raw value of a field, `None` when absent or `None` in Python.
    pub fn optional_value(&self, key: &str) -> Result<Option<Any>> {
        let value = self
            .fields
            .get(&tvm_ffi::String::from(key))
            .map_err(|error| Error::numerical(&self.owner, &error))?;
        Ok(value.filter(|value| !is_none(value)))
    }

    pub fn get<T>(&self, key: &str) -> Result<T>
    where
        T: TryFrom<Any, Error = tvm_ffi::Error>,
    {
        T::try_from(self.value(key)?).map_err(|error| self.mismatch(key, error.message()))
    }

    pub fn optional<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: TryFrom<Any, Error = tvm_ffi::Error>,
    {
        self.optional_value(key)?
            .map(|value| T::try_from(value).map_err(|error| self.mismatch(key, error.message())))
            .transpose()
    }

    /// A non-negative integer field.
    pub fn count(&self, key: &str) -> Result<usize> {
        let value: i64 = self.get(key)?;
        usize::try_from(value).map_err(|_| self.mismatch(key, "must be non-negative"))
    }

    pub fn string(&self, key: &str) -> Result<String> {
        Ok(self.get::<tvm_ffi::String>(key)?.as_str().to_owned())
    }

    /// A list of values (a Python list or tuple).
    pub fn list(&self, key: &str) -> Result<Vec<Any>> {
        let values: Array<Any> = self.get(key)?;
        Ok(values.iter().collect())
    }

    /// A list of strings.
    pub fn strings(&self, key: &str) -> Result<Vec<String>> {
        self.list(key)?
            .into_iter()
            .map(|value| {
                tvm_ffi::String::try_from(value)
                    .map(|value| value.as_str().to_owned())
                    .map_err(|error| self.mismatch(key, error.message()))
            })
            .collect()
    }

    fn mismatch(&self, key: &str, reason: &str) -> Error {
        Error::Interface(format!("{} field `{key}` {reason}", self.owner))
    }
}
