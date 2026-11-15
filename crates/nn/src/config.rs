//! Strict reading of a checkpoint's JSON configuration into typed values.
//!
//! A model family normalizes its configuration once, at the loading boundary, and every module below
//! it consumes typed fields. [`Fields`] is the one reader every family uses for that: it wraps one
//! JSON object with the name of the document it came from, and each accessor returns a typed value
//! or an [`Error::Config`] naming the document, the field and what the reader requires of it.
//!
//! The reading is strict: a required field that is absent, `null` or of another JSON type is an
//! error, never a default. A field a family may omit is read with an `optional_*` accessor, which
//! returns `None` for an absent or `null` value and still refuses a present value of the wrong
//! type; the family states its own default beside the call, where the rule it follows is visible.

use ops::activation::Activation;
use serde_json::{Map, Value};

use crate::capability::PixelNorm;
use crate::error::{Error, Result};

/// One JSON object of a checkpoint configuration, and the document it was read from.
#[derive(Clone, Debug)]
pub struct Fields<'a> {
    /// The document or object this is, as a refusal names it (`text_encoder/config.json`, or
    /// `text_encoder/config.json.vision_config` for a nested object).
    owner: String,
    values: &'a Map<String, Value>,
}

impl<'a> Fields<'a> {
    /// The fields of `value`, which must be a JSON object, read as `owner`.
    ///
    /// # Errors
    ///
    /// When `value` is not an object.
    pub fn of(owner: impl Into<String>, value: &'a Value) -> Result<Self> {
        let owner = owner.into();
        match value.as_object() {
            Some(values) => Ok(Self { owner, values }),
            None => Err(Error::config(owner, "", "must be a JSON object")),
        }
    }

    /// The document this object belongs to, as refusals name it.
    pub fn owner(&self) -> &str {
        &self.owner
    }

    /// Whether the object states `key` with a value other than `null`.
    pub fn contains(&self, key: &str) -> bool {
        !matches!(self.values.get(key), None | Some(Value::Null))
    }

    /// The keys the object states, in the document's order.
    pub fn keys(&self) -> impl Iterator<Item = &'a str> + use<'a> {
        self.values.keys().map(String::as_str)
    }

    /// The raw value of `key`, for a field whose shape a family interprets itself.
    pub fn get(&self, key: &str) -> Option<&'a Value> {
        self.values.get(key).filter(|value| !value.is_null())
    }

    /// The refusal of `key` for `problem`.
    pub fn refuse(&self, key: &str, problem: impl Into<String>) -> Error {
        Error::config(self.owner.clone(), key, problem)
    }

    /// The value of a required field.
    fn member(&self, key: &str) -> Result<&'a Value> {
        self.get(key).ok_or_else(|| self.refuse(key, "is missing"))
    }

    /// The nested object a required field holds.
    ///
    /// # Errors
    ///
    /// When the field is absent or not an object.
    pub fn object(&self, key: &str) -> Result<Fields<'a>> {
        Fields::of(format!("{}.{key}", self.owner), self.member(key)?)
    }

    /// The nested object an optional field holds, `None` when it is absent or `null`.
    ///
    /// # Errors
    ///
    /// When the field is present and not an object.
    pub fn optional_object(&self, key: &str) -> Result<Option<Fields<'a>>> {
        match self.get(key) {
            None => Ok(None),
            Some(value) => Fields::of(format!("{}.{key}", self.owner), value).map(Some),
        }
    }

    /// A required positive integer that fits a `u32`: a count, a width or a stride.
    ///
    /// # Errors
    ///
    /// When the field is absent, not a non-negative integer that fits, or zero.
    pub fn count(&self, key: &str) -> Result<u32> {
        let value = self.index(key)?;
        if value == 0 {
            return Err(self.refuse(key, "must be a positive integer, got 0"));
        }
        Ok(value)
    }

    /// An optional positive integer that fits a `u32`.
    ///
    /// # Errors
    ///
    /// When the field is present and not a positive integer that fits.
    pub fn optional_count(&self, key: &str) -> Result<Option<u32>> {
        self.optional(key, Self::count)
    }

    /// A required non-negative integer that fits a `u32`: an index, an offset or a token id.
    ///
    /// # Errors
    ///
    /// When the field is absent or not a non-negative integer that fits.
    pub fn index(&self, key: &str) -> Result<u32> {
        self.member(key)?
            .as_u64()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| self.refuse(key, "must be a non-negative integer"))
    }

    /// An optional non-negative integer that fits a `u32`.
    ///
    /// # Errors
    ///
    /// When the field is present and not a non-negative integer that fits.
    pub fn optional_index(&self, key: &str) -> Result<Option<u32>> {
        self.optional(key, Self::index)
    }

    /// A required finite number.
    ///
    /// # Errors
    ///
    /// When the field is absent or not a finite number.
    pub fn number(&self, key: &str) -> Result<f64> {
        self.member(key)?
            .as_f64()
            .filter(|value| value.is_finite())
            .ok_or_else(|| self.refuse(key, "must be a finite number"))
    }

    /// An optional finite number.
    ///
    /// # Errors
    ///
    /// When the field is present and not a finite number.
    pub fn optional_number(&self, key: &str) -> Result<Option<f64>> {
        self.optional(key, Self::number)
    }

    /// A required finite positive number.
    ///
    /// # Errors
    ///
    /// When the field is absent, not a finite number, or not above zero.
    pub fn positive(&self, key: &str) -> Result<f64> {
        let value = self.number(key)?;
        if value <= 0.0 {
            return Err(self.refuse(key, format!("must be finite and positive, got {value}")));
        }
        Ok(value)
    }

    /// An optional finite positive number.
    ///
    /// # Errors
    ///
    /// When the field is present and not a finite positive number.
    pub fn optional_positive(&self, key: &str) -> Result<Option<f64>> {
        self.optional(key, Self::positive)
    }

    /// A required string.
    ///
    /// # Errors
    ///
    /// When the field is absent or not a string.
    pub fn text(&self, key: &str) -> Result<&'a str> {
        self.member(key)?
            .as_str()
            .ok_or_else(|| self.refuse(key, "must be a string"))
    }

    /// An optional string.
    ///
    /// # Errors
    ///
    /// When the field is present and not a string.
    pub fn optional_text(&self, key: &str) -> Result<Option<&'a str>> {
        self.optional(key, Self::text)
    }

    /// A required boolean.
    ///
    /// # Errors
    ///
    /// When the field is absent or not a boolean.
    pub fn flag(&self, key: &str) -> Result<bool> {
        self.member(key)?
            .as_bool()
            .ok_or_else(|| self.refuse(key, "must be a boolean"))
    }

    /// An optional boolean.
    ///
    /// # Errors
    ///
    /// When the field is present and not a boolean.
    pub fn optional_flag(&self, key: &str) -> Result<Option<bool>> {
        self.optional(key, Self::flag)
    }

    /// A required non-empty list of positive integers that fit a `u32`.
    ///
    /// # Errors
    ///
    /// When the field is absent, not a list, empty, or holds anything but positive integers.
    pub fn counts(&self, key: &str) -> Result<Vec<u32>> {
        self.non_empty(key)?
            .iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .filter(|value| *value > 0)
                    .ok_or_else(|| self.refuse(key, "must hold positive integers"))
            })
            .collect()
    }

    /// A required non-empty list of non-negative integers that fit a `u32`.
    ///
    /// # Errors
    ///
    /// When the field is absent, not a list, empty, or holds anything but such integers.
    pub fn indices(&self, key: &str) -> Result<Vec<u32>> {
        let list = self.non_empty(key)?;
        self.integers(key, list)
    }

    /// An optional list of non-negative integers that fit a `u32`, which may be empty.
    ///
    /// # Errors
    ///
    /// When the field is present and not a list of such integers.
    pub fn optional_indices(&self, key: &str) -> Result<Option<Vec<u32>>> {
        self.optional(key, |fields, key| {
            let list = fields.list(key)?;
            fields.integers(key, list)
        })
    }

    /// A required non-empty list of finite numbers.
    ///
    /// # Errors
    ///
    /// When the field is absent, not a list, empty, or holds anything but finite numbers.
    pub fn numbers(&self, key: &str) -> Result<Vec<f64>> {
        self.non_empty(key)?
            .iter()
            .map(|value| {
                value
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .ok_or_else(|| self.refuse(key, "must hold finite numbers"))
            })
            .collect()
    }

    /// A required list of exactly `length` finite numbers.
    ///
    /// # Errors
    ///
    /// As [`Self::numbers`], and when the list holds another count.
    pub fn fixed_numbers(&self, key: &str, length: usize) -> Result<Vec<f64>> {
        let numbers = self.numbers(key)?;
        if numbers.len() != length {
            return Err(self.refuse(
                key,
                format!("must hold {length} numbers, got {}", numbers.len()),
            ));
        }
        Ok(numbers)
    }

    /// A required list of non-empty lists of positive integers that fit a `u32`.
    ///
    /// # Errors
    ///
    /// When the field is absent, not a list of lists, or an inner list is empty or holds anything
    /// but positive integers.
    pub fn nested_counts(&self, key: &str) -> Result<Vec<Vec<u32>>> {
        self.list(key)?
            .iter()
            .enumerate()
            .map(|(index, rows)| {
                let at = format!("{key}[{index}]");
                let rows = rows
                    .as_array()
                    .ok_or_else(|| self.refuse(&at, "must be a list"))?;
                if rows.is_empty() {
                    return Err(self.refuse(&at, "must be a non-empty list"));
                }
                rows.iter()
                    .map(|value| {
                        value
                            .as_u64()
                            .and_then(|value| u32::try_from(value).ok())
                            .filter(|value| *value > 0)
                            .ok_or_else(|| self.refuse(&at, "must hold positive integers"))
                    })
                    .collect()
            })
            .collect()
    }

    /// The model class a transformers checkpoint declares, its one `architectures` entry, which
    /// must be one of `classes`.
    ///
    /// # Errors
    ///
    /// When the field is absent, does not hold exactly one class, or holds another class.
    pub fn class(&self, classes: &[&'static str]) -> Result<&'static str> {
        const KEY: &str = "architectures";

        let declared = self
            .optional_texts(KEY)?
            .ok_or_else(|| self.refuse(KEY, "is required: the checkpoint declares no class"))?;
        let [class] = declared.as_slice() else {
            return Err(self.refuse(KEY, format!("must hold one class, not {declared:?}")));
        };

        classes
            .iter()
            .copied()
            .find(|known| known == class)
            .ok_or_else(|| self.refuse(KEY, format!("states {class}, not one of {classes:?}")))
    }

    /// An optional list of strings, which may be empty.
    ///
    /// # Errors
    ///
    /// When the field is present and not a list of strings.
    pub fn optional_texts(&self, key: &str) -> Result<Option<Vec<&'a str>>> {
        self.optional(key, |fields, key| {
            fields
                .list(key)?
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .ok_or_else(|| fields.refuse(key, "must hold strings"))
                })
                .collect()
        })
    }

    /// A required activation, by its name in transformers' activation table (`ACT2CLS`).
    ///
    /// The names served are the table's entries for the functions the operators implement:
    /// `silu`, `gelu` (the exact GELU) and `gelu_pytorch_tanh` (its tanh approximation). Any
    /// other name is refused: an activation read as a different one is a model that runs and
    /// computes something else.
    ///
    /// # Errors
    ///
    /// When the field is absent, not a string, or names an activation no operator implements.
    pub fn activation(&self, key: &str) -> Result<Activation> {
        activation_named(self.text(key)?).ok_or_else(|| {
            self.refuse(
                key,
                format!(
                    "names `{}`, and the activations served are silu, gelu and gelu_pytorch_tanh",
                    self.text(key).unwrap_or_default()
                ),
            )
        })
    }

    /// An optional activation, as [`Self::activation`] reads it.
    ///
    /// # Errors
    ///
    /// When the field is present and does not name a served activation.
    pub fn optional_activation(&self, key: &str) -> Result<Option<Activation>> {
        self.optional(key, Self::activation)
    }

    /// The pixel normalization an image processor's configuration states by `image_mean` and
    /// `image_std`: each RGB channel maps `x` in `[0, 255]` to `(x / 255 - mean) / std`.
    ///
    /// # Errors
    ///
    /// Fields of the wrong type or length, or a deviation that is not positive.
    pub fn pixel_norm(&self) -> Result<PixelNorm> {
        let channels = |key: &str| -> Result<[f32; 3]> {
            let values = self.fixed_numbers(key, 3)?;
            Ok([values[0] as f32, values[1] as f32, values[2] as f32])
        };
        let (mean, std) = (channels("image_mean")?, channels("image_std")?);

        if std.iter().any(|value| *value <= 0.0) {
            return Err(self.refuse("image_std", format!("must be positive, got {std:?}")));
        }
        Ok(PixelNorm { mean, std })
    }

    /// Which layers of a hybrid stack of `layers` layers attend linearly, as `layer_types`
    /// declares each layer: `linear_attention` or `full_attention`.
    ///
    /// transformers derives `layer_types` from `full_attention_interval` when it constructs a
    /// configuration and serializes both; the interval, when stated, is the same statement in
    /// another form (every `interval`-th layer attends fully) and must agree with the list.
    ///
    /// # Errors
    ///
    /// A missing list, a list of another length, a kind other than the two, or an interval
    /// that disagrees with the list.
    pub fn linear_attention(&self, layers: u32) -> Result<Vec<bool>> {
        let kinds = self.optional_texts("layer_types")?.ok_or_else(|| {
            self.refuse(
                "layer_types",
                "is required: a hybrid stack declares the kind of every layer",
            )
        })?;
        let linear = kinds
            .iter()
            .map(|kind| match *kind {
                "full_attention" => Ok(false),
                "linear_attention" => Ok(true),
                other => Err(self.refuse(
                    "layer_types",
                    format!("names {other}, neither full_attention nor linear_attention"),
                )),
            })
            .collect::<Result<Vec<_>>>()?;

        if linear.len() != layers as usize {
            return Err(self.refuse(
                "layer_types",
                format!("states {} layers of a {layers}-layer stack", linear.len()),
            ));
        }
        if let Some(interval) = self.optional_count("full_attention_interval")?
            && (0..layers)
                .any(|layer| linear[layer as usize] == (layer + 1).is_multiple_of(interval))
        {
            return Err(self.refuse(
                "full_attention_interval",
                format!("of {interval} disagrees with layer_types"),
            ));
        }
        Ok(linear)
    }

    /// `read(key)` for a present field, `None` for an absent or `null` one.
    fn optional<T>(
        &self,
        key: &str,
        read: impl FnOnce(&Self, &str) -> Result<T>,
    ) -> Result<Option<T>> {
        if self.contains(key) {
            read(self, key).map(Some)
        } else {
            Ok(None)
        }
    }

    /// The list a required field holds.
    fn list(&self, key: &str) -> Result<&'a Vec<Value>> {
        self.member(key)?
            .as_array()
            .ok_or_else(|| self.refuse(key, "must be a list"))
    }

    /// The non-empty list a required field holds.
    fn non_empty(&self, key: &str) -> Result<&'a Vec<Value>> {
        let list = self.list(key)?;
        if list.is_empty() {
            return Err(self.refuse(key, "must be a non-empty list"));
        }
        Ok(list)
    }

    /// `list` as non-negative integers that fit a `u32`.
    fn integers(&self, key: &str, list: &[Value]) -> Result<Vec<u32>> {
        list.iter()
            .map(|value| {
                value
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| self.refuse(key, "must hold integers"))
            })
            .collect()
    }
}

/// The activation transformers' activation table (`ACT2CLS`) names, or `None` for one no
/// operator implements: one entry per function served.
fn activation_named(name: &str) -> Option<Activation> {
    Some(match name {
        "silu" => Activation::Silu,
        "gelu" => Activation::Gelu,
        "gelu_pytorch_tanh" => Activation::GeluTanh,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document(text: &str) -> Value {
        serde_json::from_str(text).expect("a fixture is JSON")
    }

    #[test]
    fn a_required_field_is_refused_by_its_document_and_key() {
        let value = document(r#"{"depth": 0, "width": "wide", "nested": {"heads": 4}}"#);
        let fields = Fields::of("vae/config.json", &value).unwrap();
        assert_eq!(
            fields.count("depth").unwrap_err().to_string(),
            "vae/config.json.depth must be a positive integer, got 0"
        );
        assert_eq!(
            fields.count("width").unwrap_err().to_string(),
            "vae/config.json.width must be a non-negative integer"
        );
        assert_eq!(
            fields.count("missing").unwrap_err().to_string(),
            "vae/config.json.missing is missing"
        );
        let nested = fields.object("nested").unwrap();
        assert_eq!(nested.count("heads").unwrap(), 4);
        assert_eq!(
            nested.flag("heads").unwrap_err().to_string(),
            "vae/config.json.nested.heads must be a boolean"
        );
    }

    #[test]
    fn an_optional_field_is_absent_or_null_and_otherwise_still_typed() {
        let value = document(r#"{"eps": null, "bias": true, "steps": "one"}"#);
        let fields = Fields::of("config.json", &value).unwrap();
        assert_eq!(fields.optional_positive("eps").unwrap(), None);
        assert_eq!(fields.optional_flag("missing").unwrap(), None);
        assert_eq!(fields.optional_flag("bias").unwrap(), Some(true));
        assert!(fields.optional_count("steps").is_err());
    }

    /// An activation is read by its entry in transformers' table, one name per function; a name
    /// the table defines for a function no operator implements, or a second name, is refused.
    #[test]
    fn an_activation_is_read_by_its_table_name() {
        let value = document(
            r#"{"a": "silu", "c": "gelu", "d": "gelu_pytorch_tanh", "e": "relu", "f": "swish"}"#,
        );
        let fields = Fields::of("config.json", &value).unwrap();
        assert_eq!(fields.activation("a").unwrap(), Activation::Silu);
        assert_eq!(fields.activation("c").unwrap(), Activation::Gelu);
        assert_eq!(fields.activation("d").unwrap(), Activation::GeluTanh);
        for refused in ["e", "f"] {
            let error = fields.activation(refused).unwrap_err().to_string();
            assert!(error.contains(fields.text(refused).unwrap()), "{error}");
        }
    }

    /// An image processor's normalization is three means and three positive deviations.
    #[test]
    fn a_pixel_normalization_is_three_channels_of_positive_deviation() {
        let value = document(r#"{"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.25, 1]}"#);
        let norm = Fields::of("preprocessor_config.json", &value)
            .unwrap()
            .pixel_norm()
            .unwrap();
        assert_eq!(
            norm,
            PixelNorm {
                mean: [0.5; 3],
                std: [0.5, 0.25, 1.0]
            }
        );

        let value = document(r#"{"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0, 1]}"#);
        let fields = Fields::of("preprocessor_config.json", &value).unwrap();
        assert!(fields.pixel_norm().is_err());
    }

    /// A hybrid stack declares each layer's kind; the interval, when stated, must say the same.
    #[test]
    fn a_hybrid_stack_declares_each_layer_s_kind() {
        let kinds = r#""layer_types": ["linear_attention", "linear_attention", "full_attention",
                         "linear_attention", "linear_attention", "full_attention"]"#;
        let read = |text: String| {
            let value = document(&text);
            Fields::of("config.json", &value)
                .unwrap()
                .linear_attention(6)
        };

        let expected = vec![true, true, false, true, true, false];
        assert_eq!(read(format!("{{{kinds}}}")).unwrap(), expected);
        assert_eq!(
            read(format!(r#"{{{kinds}, "full_attention_interval": 3}}"#)).unwrap(),
            expected
        );
        assert!(read(format!(r#"{{{kinds}, "full_attention_interval": 2}}"#)).is_err());
        assert!(read(r#"{"full_attention_interval": 3}"#.to_owned()).is_err());
        assert!(
            read(r#"{"layer_types": ["full_attention", "sliding_attention"]}"#.to_owned()).is_err()
        );
    }

    #[test]
    fn lists_are_typed_element_by_element() {
        let value = document(
            r#"{"counts": [1, 2], "zero": [0], "empty": [], "nested": [[1], []], "names": ["a"]}"#,
        );
        let fields = Fields::of("config.json", &value).unwrap();
        assert_eq!(fields.counts("counts").unwrap(), vec![1, 2]);
        assert!(fields.counts("zero").is_err());
        assert_eq!(fields.indices("zero").unwrap(), vec![0]);
        assert!(fields.indices("empty").is_err());
        assert_eq!(fields.optional_indices("empty").unwrap(), Some(Vec::new()));
        assert_eq!(
            fields.nested_counts("nested").unwrap_err().to_string(),
            "config.json.nested[1] must be a non-empty list"
        );
        assert_eq!(fields.optional_texts("names").unwrap(), Some(vec!["a"]));
        assert_eq!(fields.fixed_numbers("counts", 2).unwrap(), vec![1.0, 2.0]);
        assert!(fields.fixed_numbers("counts", 3).is_err());
    }
}
