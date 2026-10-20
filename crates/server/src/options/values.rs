//! Serde forms of serving-option values.
//!
//! Each form reads what a configuration file or a `--set` value states and
//! writes what `--print-config` prints, so a printed configuration reads back
//! unchanged:
//!
//! - a duration is whole seconds (`30`) or a value with units (`500ms`,
//!   `30s`, `5m`, `1h`), printed with units;
//! - a byte size is whole bytes or an integer with a decimal (`20MB`) or
//!   binary (`512MiB`) unit, printed in the largest binary unit that divides
//!   it, or as whole bytes;
//! - a list is an array, a comma-separated string (`"1024,10240"`) or one
//!   element, printed as an array.

use std::fmt;
use std::marker::PhantomData;
use std::str::FromStr;
use std::time::Duration;

use serde::de::{self, DeserializeOwned, IntoDeserializer, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::serving::systemone::CanvasMode;

/// Parses a duration: whole seconds or a value with units.
pub(super) fn parse_duration(text: &str) -> Result<Duration, String> {
    let text = text.trim();
    if let Ok(seconds) = text.parse::<u64>() {
        return Ok(Duration::from_secs(seconds));
    }
    humantime::parse_duration(text).map_err(|error| {
        format!("{error}; a duration is whole seconds or a value with units, such as 30s or 5m")
    })
}

/// Parses a positive byte size: whole bytes or an integer with a decimal
/// (`kB`, `MB`, `GB`, `TB`) or binary (`KiB`, `MiB`, `GiB`, `TiB`) unit,
/// case-insensitively.
pub(super) fn parse_bytes(text: &str) -> Result<u64, String> {
    let text = text.trim();
    let split = text
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(text.len());
    let (number, unit) = text.split_at(split);
    let unit = unit.trim().to_ascii_lowercase();
    let scale = if unit.is_empty() {
        Some(1)
    } else {
        BYTE_UNITS
            .iter()
            .find(|(name, _)| name.eq_ignore_ascii_case(&unit))
            .map(|(_, scale)| *scale)
    };

    let bytes = number
        .parse::<u64>()
        .ok()
        .zip(scale)
        .and_then(|(number, scale)| number.checked_mul(scale))
        .ok_or_else(|| {
            format!(
                "invalid byte size {text:?}; a size is whole bytes or an integer with a unit, \
                 such as 512MiB or 20MB"
            )
        })?;
    if bytes == 0 {
        return Err("a byte size must be positive".into());
    }
    Ok(bytes)
}

/// Byte units, the binary ones largest first so printing picks the largest
/// that divides a size.
const BYTE_UNITS: [(&str, u64); 9] = [
    ("TiB", 1 << 40),
    ("GiB", 1 << 30),
    ("MiB", 1 << 20),
    ("KiB", 1 << 10),
    ("TB", 1_000_000_000_000),
    ("GB", 1_000_000_000),
    ("MB", 1_000_000),
    ("kB", 1_000),
    ("B", 1),
];

/// Prints a byte size in the largest binary unit that divides it, or as
/// whole bytes.
fn serialize_bytes<S: Serializer>(bytes: u64, serializer: S) -> Result<S::Ok, S::Error> {
    match BYTE_UNITS[..4]
        .iter()
        .find(|(_, scale)| bytes.is_multiple_of(*scale))
    {
        Some((unit, scale)) => serializer.serialize_str(&format!("{}{unit}", bytes / scale)),
        None => serializer.serialize_u64(bytes),
    }
}

/// A value stated as a number or as text, before its form parses it.
#[derive(Deserialize)]
#[serde(untagged)]
enum NumberOrText {
    Number(u64),
    Text(String),
}

/// The serde form of a [`Duration`].
pub(super) mod duration {
    use super::*;

    pub(in crate::options) fn serialize<S: Serializer>(
        value: &Duration,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        humantime::format_duration(*value)
            .to_string()
            .serialize(serializer)
    }

    pub(in crate::options) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Duration, D::Error> {
        match NumberOrText::deserialize(deserializer)? {
            NumberOrText::Number(seconds) => Ok(Duration::from_secs(seconds)),
            NumberOrText::Text(text) => parse_duration(&text).map_err(de::Error::custom),
        }
    }
}

/// The serde form of an optional [`Duration`].
pub(super) mod optional_duration {
    use super::*;

    pub(in crate::options) fn serialize<S: Serializer>(
        value: &Option<Duration>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        value
            .map(|value| humantime::format_duration(value).to_string())
            .serialize(serializer)
    }

    pub(in crate::options) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Duration>, D::Error> {
        duration::deserialize(deserializer).map(Some)
    }
}

/// The serde form of a positive byte size.
pub(super) mod bytes {
    use super::*;

    pub(in crate::options) fn serialize<S: Serializer>(
        value: &u64,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serialize_bytes(*value, serializer)
    }

    pub(in crate::options) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<u64, D::Error> {
        match NumberOrText::deserialize(deserializer)? {
            NumberOrText::Number(0) => Err(de::Error::custom("a byte size must be positive")),
            NumberOrText::Number(bytes) => Ok(bytes),
            NumberOrText::Text(text) => parse_bytes(&text).map_err(de::Error::custom),
        }
    }
}

/// The serde form of a positive byte size held as a `usize`.
pub(super) mod byte_count {
    use super::*;

    pub(in crate::options) fn serialize<S: Serializer>(
        value: &usize,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serialize_bytes(*value as u64, serializer)
    }

    pub(in crate::options) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<usize, D::Error> {
        let bytes = bytes::deserialize(deserializer)?;
        usize::try_from(bytes).map_err(|_| {
            de::Error::custom(format!(
                "{bytes} bytes exceed this platform's address space"
            ))
        })
    }
}

/// Reads a list from an array, a comma-separated string or one element.
struct ListVisitor<T>(PhantomData<T>);

impl<'de, T> Visitor<'de> for ListVisitor<T>
where
    T: DeserializeOwned + FromStr,
    T::Err: fmt::Display,
{
    type Value = Vec<T>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("an array, a comma-separated string or one element")
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut sequence: A) -> Result<Self::Value, A::Error> {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element()? {
            values.push(value);
        }
        Ok(values)
    }

    fn visit_str<E: de::Error>(self, text: &str) -> Result<Self::Value, E> {
        text.split(',')
            .map(|element| {
                element
                    .trim()
                    .parse::<T>()
                    .map_err(|error| E::custom(format!("{:?}: {error}", element.trim())))
            })
            .collect()
    }

    fn visit_u64<E: de::Error>(self, value: u64) -> Result<Self::Value, E> {
        T::deserialize(value.into_deserializer()).map(|value| vec![value])
    }

    fn visit_i64<E: de::Error>(self, value: i64) -> Result<Self::Value, E> {
        T::deserialize(value.into_deserializer()).map(|value| vec![value])
    }
}

/// The serde form of a list.
pub(super) mod list {
    use super::*;

    pub(in crate::options) fn deserialize<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
    where
        D: Deserializer<'de>,
        T: DeserializeOwned + FromStr,
        T::Err: fmt::Display,
    {
        deserializer.deserialize_any(ListVisitor(PhantomData))
    }
}

/// The serde form of an optional list.
pub(super) mod optional_list {
    use super::*;

    pub(in crate::options) fn deserialize<'de, D, T>(
        deserializer: D,
    ) -> Result<Option<Vec<T>>, D::Error>
    where
        D: Deserializer<'de>,
        T: DeserializeOwned + FromStr,
        T::Err: fmt::Display,
    {
        list::deserialize(deserializer).map(Some)
    }
}

/// A `HEIGHTxWIDTH` image size in pixels, both positive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ImageSize(pub [u32; 2]);

impl FromStr for ImageSize {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let invalid =
            || format!("invalid image size {text:?}; expected HEIGHTxWIDTH, such as 1152x2048");
        let lowered = text.trim().to_ascii_lowercase();
        let (height, width) = lowered.split_once('x').ok_or_else(invalid)?;
        match (height.trim().parse::<u32>(), width.trim().parse::<u32>()) {
            (Ok(height), Ok(width)) if height > 0 && width > 0 => Ok(Self([height, width])),
            _ => Err(invalid()),
        }
    }
}

impl fmt::Display for ImageSize {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}x{}", self.0[0], self.0[1])
    }
}

impl Serialize for ImageSize {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_string().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ImageSize {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(de::Error::custom)
    }
}

/// The serde form of a readout canvas length: `"full"`, `"compact"` or a
/// positive multiple of 16.
pub(super) mod canvas {
    use super::*;

    pub(in crate::options) fn serialize<S: Serializer>(
        value: &CanvasMode,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match value {
            CanvasMode::Full => serializer.serialize_str("full"),
            CanvasMode::Compact => serializer.serialize_str("compact"),
            CanvasMode::Fixed(length) => serializer.serialize_u32(*length),
        }
    }

    pub(in crate::options) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<CanvasMode, D::Error> {
        let text = match NumberOrText::deserialize(deserializer)? {
            NumberOrText::Number(length) => length.to_string(),
            NumberOrText::Text(text) => text,
        };
        text.parse().map_err(de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

    #[test]
    fn durations_take_seconds_or_units() {
        assert_eq!(parse_duration("30"), Ok(Duration::from_secs(30)));
        assert_eq!(parse_duration("500ms"), Ok(Duration::from_millis(500)));
        assert_eq!(parse_duration("1h 30m"), Ok(Duration::from_secs(5400)));
        assert!(parse_duration("soon").is_err());
    }

    #[test]
    fn byte_sizes_take_bytes_or_units_and_print_in_binary_units() {
        assert_eq!(parse_bytes("1048576"), Ok(1 << 20));
        assert_eq!(parse_bytes("512MiB"), Ok(512 << 20));
        assert_eq!(parse_bytes("1 GiB"), Ok(1 << 30));
        assert_eq!(parse_bytes("20mb"), Ok(20_000_000));
        for refused in [
            "0",
            "0MiB",
            "1.5GiB",
            "12 parsecs",
            "MiB",
            "-1",
            "99999999999TiB",
        ] {
            assert!(parse_bytes(refused).is_err(), "{refused:?} is not a size");
        }

        #[derive(Serialize)]
        struct Size(#[serde(with = "bytes")] u64);
        let printed = |size| toml::Value::try_from(Size(size)).unwrap();
        assert_eq!(printed(512 << 20), toml::Value::String("512MiB".into()));
        assert_eq!(printed(20_000_000), toml::Value::Integer(20_000_000));
    }

    #[derive(Debug, PartialEq, Deserialize)]
    struct Lists {
        #[serde(deserialize_with = "list::deserialize")]
        sizes: Vec<NonZeroU32>,
        #[serde(deserialize_with = "list::deserialize")]
        shapes: Vec<ImageSize>,
    }

    /// A list reads the same from an array, a comma-separated string or one
    /// element, and each element keeps its own checks.
    #[test]
    fn lists_take_arrays_comma_separated_strings_or_one_element() {
        let lists = |text: &str| toml::from_str::<Lists>(text);
        let expected = Lists {
            sizes: [1024, 10240]
                .map(|size| NonZeroU32::new(size).unwrap())
                .to_vec(),
            shapes: vec![ImageSize([1152, 2048])],
        };
        assert_eq!(
            lists("sizes = [1024, 10240]\nshapes = [\"1152x2048\"]").unwrap(),
            expected
        );
        assert_eq!(
            lists("sizes = \"1024, 10240\"\nshapes = \"1152x2048\"").unwrap(),
            expected
        );
        assert_eq!(
            lists("sizes = 1024\nshapes = []").unwrap().sizes,
            vec![NonZeroU32::new(1024).unwrap()]
        );
        assert!(lists("sizes = \"1024,0\"\nshapes = []").is_err());
        assert!(lists("sizes = []\nshapes = \"1152\"").is_err());
    }
}
