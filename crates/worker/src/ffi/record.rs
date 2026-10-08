//! Typed access to metadata maps exchanged with Python.

use tvm_ffi::{Any, Array, Map};

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

/// A map argument for an interface function, built field by field.
#[derive(Default)]
pub struct Fields {
    entries: Vec<(tvm_ffi::String, Any)>,
}

impl Fields {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a field. Integers, floats, booleans, strings, maps, arrays,
    /// tensors, functions and handles convert through [`Any`].
    pub fn with(mut self, key: &str, value: impl Into<Any>) -> Self {
        self.entries.push((key.into(), value.into()));
        self
    }

    /// Add Python `None`.
    pub fn none(self, key: &str) -> Self {
        self.with(key, Any::default())
    }

    /// Add an optional value; `None` becomes Python `None`.
    pub fn optional(self, key: &str, value: Option<impl Into<Any>>) -> Self {
        match value {
            Some(value) => self.with(key, value),
            None => self.none(key),
        }
    }

    pub fn text(self, key: &str, value: &str) -> Self {
        self.with(key, tvm_ffi::String::from(value))
    }

    pub fn optional_text(self, key: &str, value: Option<&str>) -> Self {
        match value {
            Some(value) => self.text(key, value),
            None => self.none(key),
        }
    }

    pub fn map(self) -> Map<tvm_ffi::String, Any> {
        self.entries.into_iter().collect()
    }

    pub fn into_any(self) -> Any {
        Any::from(self.map())
    }
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

/// Convert a JSON value: objects become maps, arrays become arrays, integers
/// stay integers and `null` becomes Python `None`.
pub fn json(value: &serde_json::Value) -> crate::Result<Any> {
    use serde_json::Value;
    Ok(match value {
        Value::Null => Any::default(),
        Value::Bool(value) => Any::from(*value),
        Value::Number(number) => match (number.as_i64(), number.as_f64()) {
            (Some(integer), _) => Any::from(integer),
            (None, Some(float)) => Any::from(float),
            (None, None) => {
                return Err(crate::Error::Invalid(format!(
                    "the number {number} has no FFI representation"
                )));
            }
        },
        Value::String(text) => Any::from(tvm_ffi::String::from(text.as_str())),
        Value::Array(values) => Any::from(list(
            values.iter().map(json).collect::<crate::Result<Vec<_>>>()?,
        )),
        Value::Object(fields) => {
            let mut map = Fields::new();
            for (key, value) in fields {
                map = map.with(key, json(value)?);
            }
            map.into_any()
        }
    })
}

/// Convert a value Python returned into JSON: maps, arrays, strings,
/// integers, floats, booleans and `None`. Other values (tensors, functions,
/// handles) have no JSON form and are refused.
pub fn to_json(value: &Any) -> crate::Result<serde_json::Value> {
    use serde_json::Value;
    use tvm_ffi::TypeIndex;

    let index = value.type_index();
    let unsupported = || {
        crate::Error::Interface(format!(
            "a value of FFI type index {index} has no JSON representation"
        ))
    };
    let convert = |error: tvm_ffi::Error| crate::Error::Interface(error.message().to_owned());
    Ok(if index == TypeIndex::kTVMFFINone as i32 {
        Value::Null
    } else if index == TypeIndex::kTVMFFIBool as i32 {
        Value::Bool(bool::try_from(value.clone()).map_err(convert)?)
    } else if index == TypeIndex::kTVMFFIInt as i32 {
        Value::from(i64::try_from(value.clone()).map_err(convert)?)
    } else if index == TypeIndex::kTVMFFIFloat as i32 {
        Value::from(f64::try_from(value.clone()).map_err(convert)?)
    } else if index == TypeIndex::kTVMFFIStr as i32 || index == TypeIndex::kTVMFFISmallStr as i32 {
        Value::String(
            tvm_ffi::String::try_from(value.clone())
                .map_err(convert)?
                .as_str()
                .to_owned(),
        )
    } else if index == TypeIndex::kTVMFFIArray as i32 {
        let values: Array<Any> = Array::try_from(value.clone()).map_err(convert)?;
        Value::Array(
            values
                .iter()
                .map(|value| to_json(&value))
                .collect::<crate::Result<_>>()?,
        )
    } else if index == TypeIndex::kTVMFFIMap as i32 {
        let fields: Map<tvm_ffi::String, Any> = Map::try_from(value.clone()).map_err(convert)?;
        Value::Object(
            fields
                .iter()
                .map(|(key, value)| Ok((key.as_str().to_owned(), to_json(&value)?)))
                .collect::<crate::Result<_>>()?,
        )
    } else {
        return Err(unsupported());
    })
}

impl Record {
    /// The record as a JSON object.
    pub fn to_json(&self) -> crate::Result<serde_json::Map<String, serde_json::Value>> {
        self.fields
            .iter()
            .map(|(key, value)| Ok((key.as_str().to_owned(), to_json(&value)?)))
            .collect()
    }
}
