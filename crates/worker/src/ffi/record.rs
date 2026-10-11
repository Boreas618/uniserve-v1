//! Typed access to metadata maps exchanged with Python.

use tvm_ffi::{Any, Array, Map};

use super::values::{integers, is_none};
use crate::{Error, Result};

/// A `Map<String, Any>` describing loaded state, for example the cache plan or
/// a component's buffer requirements. Accessors name the producing function
/// and the key in their errors so a contract mismatch is diagnosable.
pub struct Record {
    owner: String,
    fields: Map<tvm_ffi::String, Any>,
}

impl Record {
    pub fn new(owner: impl Into<String>, value: Any) -> Result<Self> {
        let owner = owner.into();
        let fields = Map::try_from(value).map_err(|error| Error::numerical(&owner, &error))?;
        Ok(Self { owner, fields })
    }

    pub fn owner(&self) -> &str {
        &self.owner
    }

    pub fn contains(&self, key: &str) -> bool {
        self.fields.contains_key(&tvm_ffi::String::from(key))
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

    /// An optional non-negative integer field.
    pub fn optional_count(&self, key: &str) -> Result<Option<usize>> {
        self.optional::<i64>(key)?
            .map(|value| {
                usize::try_from(value).map_err(|_| self.mismatch(key, "must be non-negative"))
            })
            .transpose()
    }

    pub fn flag(&self, key: &str) -> Result<bool> {
        self.get(key)
    }

    pub fn string(&self, key: &str) -> Result<String> {
        Ok(self.get::<tvm_ffi::String>(key)?.as_str().to_owned())
    }

    pub fn optional_string(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .optional::<tvm_ffi::String>(key)?
            .map(|value| value.as_str().to_owned()))
    }

    /// A nested record.
    pub fn record(&self, key: &str) -> Result<Record> {
        Record::new(format!("{}[{key}]", self.owner), self.value(key)?)
    }

    /// An optional nested record.
    pub fn optional_record(&self, key: &str) -> Result<Option<Record>> {
        self.optional_value(key)?
            .map(|value| Record::new(format!("{}[{key}]", self.owner), value))
            .transpose()
    }

    /// A list of records.
    pub fn records(&self, key: &str) -> Result<Vec<Record>> {
        self.list(key)?
            .into_iter()
            .enumerate()
            .map(|(index, value)| Record::new(format!("{}[{key}][{index}]", self.owner), value))
            .collect()
    }

    /// A list of values (a Python list or tuple).
    pub fn list(&self, key: &str) -> Result<Vec<Any>> {
        let values: Array<Any> = self.get(key)?;
        Ok(values.iter().collect())
    }

    /// An integer list, carried as a `Shape` or a list of integers.
    pub fn shape(&self, key: &str) -> Result<Vec<i64>> {
        let value = self.value(key)?;
        integers(value).map_err(|error| self.mismatch(key, error.message()))
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

    /// The record's keys.
    pub fn keys(&self) -> Vec<String> {
        self.fields
            .keys()
            .map(|key| key.as_str().to_owned())
            .collect()
    }

    fn mismatch(&self, key: &str, reason: &str) -> Error {
        Error::Interface(format!("{} field `{key}` {reason}", self.owner))
    }
}
