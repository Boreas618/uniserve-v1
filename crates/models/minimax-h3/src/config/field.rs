//! Reading one field of a checkpoint sidecar into a typed value.
//!
//! Every accessor names the sidecar and the field it read, so a checkpoint that states a value the
//! architecture cannot use is refused with the message that says which field and why. A missing
//! field is an error, never a default: a checkpoint states every dimension the layers read.

use serde_json::{Map, Value};

use crate::{Error, Result};

/// The JSON object `value` must hold.
pub(super) fn object<'a>(value: &'a Value, owner: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| Error::config(format!("{owner} must be a JSON object")))
}

/// The JSON object a sidecar's field must hold.
pub(super) fn nested<'a>(
    values: &'a Map<String, Value>,
    key: &str,
    owner: &str,
) -> Result<&'a Map<String, Value>> {
    object(member(values, key, owner)?, &format!("{owner}.{key}"))
}

/// A field that must be present.
pub(super) fn member<'a>(
    values: &'a Map<String, Value>,
    key: &str,
    owner: &str,
) -> Result<&'a Value> {
    values
        .get(key)
        .ok_or_else(|| Error::config(format!("{owner} is missing the field `{key}`")))
}

/// A required positive integer that fits a `u32`: a count, a width or a stride.
pub(super) fn count(values: &Map<String, Value>, key: &str, owner: &str) -> Result<u32> {
    let value = integer(values, key, owner)?;
    if value == 0 {
        return Err(Error::config(format!(
            "{owner}.{key} must be a positive integer, got 0"
        )));
    }
    Ok(value)
}

/// A required non-negative integer that fits a `u32`: an index, an offset or a token id.
pub(super) fn index(values: &Map<String, Value>, key: &str, owner: &str) -> Result<u32> {
    integer(values, key, owner)
}

/// An optional non-negative integer that fits a `u32`.
pub(super) fn optional_index(
    values: &Map<String, Value>,
    key: &str,
    owner: &str,
) -> Result<Option<u32>> {
    match values.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(_) => integer(values, key, owner).map(Some),
    }
}

/// A required finite number.
pub(super) fn number(values: &Map<String, Value>, key: &str, owner: &str) -> Result<f64> {
    match member(values, key, owner)? {
        Value::Number(number) => number
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| Error::config(format!("{owner}.{key} must be a finite number"))),
        _ => Err(Error::config(format!(
            "{owner}.{key} must be a finite number"
        ))),
    }
}

/// A required finite positive number.
pub(super) fn positive(values: &Map<String, Value>, key: &str, owner: &str) -> Result<f64> {
    let value = number(values, key, owner)?;
    if value <= 0.0 {
        return Err(Error::config(format!(
            "{owner}.{key} must be finite and positive, got {value}"
        )));
    }
    Ok(value)
}

/// A required string.
pub(super) fn text<'a>(values: &'a Map<String, Value>, key: &str, owner: &str) -> Result<&'a str> {
    member(values, key, owner)?
        .as_str()
        .ok_or_else(|| Error::config(format!("{owner}.{key} must be a string")))
}

/// A required boolean.
pub(super) fn flag(values: &Map<String, Value>, key: &str, owner: &str) -> Result<bool> {
    member(values, key, owner)?
        .as_bool()
        .ok_or_else(|| Error::config(format!("{owner}.{key} must be a boolean")))
}

/// A required non-empty list of positive integers that fit a `u32`.
pub(super) fn counts(values: &Map<String, Value>, key: &str, owner: &str) -> Result<Vec<u32>> {
    let list = list(values, key, owner)?;
    if list.is_empty() {
        return Err(Error::config(format!(
            "{owner}.{key} must be a non-empty list"
        )));
    }
    list.iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| Error::config(format!("{owner}.{key} must hold positive integers")))
        })
        .collect()
}

/// A required list of non-negative integers that fit a `u32`.
pub(super) fn indices(values: &Map<String, Value>, key: &str, owner: &str) -> Result<Vec<u32>> {
    let list = list(values, key, owner)?;
    if list.is_empty() {
        return Err(Error::config(format!(
            "{owner}.{key} must be a non-empty list"
        )));
    }
    list.iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| Error::config(format!("{owner}.{key} must hold integers")))
        })
        .collect()
}

/// A required list of finite numbers.
pub(super) fn numbers(values: &Map<String, Value>, key: &str, owner: &str) -> Result<Vec<f64>> {
    let list = list(values, key, owner)?;
    if list.is_empty() {
        return Err(Error::config(format!(
            "{owner}.{key} must be a non-empty list"
        )));
    }
    list.iter()
        .map(|value| {
            value
                .as_f64()
                .filter(|value| value.is_finite())
                .ok_or_else(|| Error::config(format!("{owner}.{key} must hold finite numbers")))
        })
        .collect()
}

/// A required list of finite numbers of exactly `length` entries.
pub(super) fn fixed_numbers(
    values: &Map<String, Value>,
    key: &str,
    owner: &str,
    length: usize,
) -> Result<Vec<f64>> {
    let numbers = numbers(values, key, owner)?;
    if numbers.len() != length {
        return Err(Error::config(format!(
            "{owner}.{key} must hold {length} numbers, got {}",
            numbers.len()
        )));
    }
    Ok(numbers)
}

/// A required list of lists of positive integers that fit a `u32`.
pub(super) fn nested_counts(
    values: &Map<String, Value>,
    key: &str,
    owner: &str,
) -> Result<Vec<Vec<u32>>> {
    let list = list(values, key, owner)?;
    list.iter()
        .enumerate()
        .map(|(index, rows)| {
            let rows = rows
                .as_array()
                .ok_or_else(|| Error::config(format!("{owner}.{key}[{index}] must be a list")))?;
            if rows.is_empty() {
                return Err(Error::config(format!(
                    "{owner}.{key}[{index}] must be a non-empty list"
                )));
            }
            rows.iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| u32::try_from(value).ok())
                        .filter(|value| *value > 0)
                        .ok_or_else(|| {
                            Error::config(format!(
                                "{owner}.{key}[{index}] must hold positive integers"
                            ))
                        })
                })
                .collect()
        })
        .collect()
}

/// The list a sidecar's field must hold.
fn list<'a>(values: &'a Map<String, Value>, key: &str, owner: &str) -> Result<&'a Vec<Value>> {
    member(values, key, owner)?
        .as_array()
        .ok_or_else(|| Error::config(format!("{owner}.{key} must be a list")))
}

/// The integer a sidecar's field must hold, as a `u32`.
fn integer(values: &Map<String, Value>, key: &str, owner: &str) -> Result<u32> {
    member(values, key, owner)?
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| Error::config(format!("{owner}.{key} must be a non-negative integer")))
}
