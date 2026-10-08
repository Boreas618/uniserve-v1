//! Typed access to metadata maps returned by Python.

use tvm_ffi::{Any, Map};

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
        Ok(value.filter(|value| value.type_index() != tvm_ffi::TypeIndex::kTVMFFINone as i32))
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

    /// A nested record.
    pub fn record(&self, key: &str) -> Result<Record> {
        Record::new(format!("{}[{key}]", self.owner), self.value(key)?)
    }

    /// Integer list fields travel as `Shape`.
    pub fn shape(&self, key: &str) -> Result<Vec<i64>> {
        Ok(self.get::<tvm_ffi::Shape>(key)?.as_slice().to_vec())
    }

    fn mismatch(&self, key: &str, reason: &str) -> Error {
        Error::Interface(format!("{} field `{key}` {reason}", self.owner))
    }
}
