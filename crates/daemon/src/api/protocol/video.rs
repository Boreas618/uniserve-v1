//! The video generation request, as `/v1/videos` and `/v1/videos/sync` take it.
//!
//! The request is the same for every video model: a prompt, an output size and length, and
//! reference media, each anchoring a generated frame or conditioning the generation as a whole.
//! What a model makes of it is the model's (`processing::VideoProcessor`). The fields are
//! listed once ([`REQUEST_FIELDS`]), with the kind of value each takes, for the multipart
//! form of the same request.

use serde::Deserialize;
use serde_json::{Map, Value};

/// One reference medium of a request, in request order.
///
/// The order is semantic: a model presents references in it.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VideoReference {
    /// `image`, `video` or `audio`.
    #[serde(rename = "type")]
    pub kind: String,
    /// `data:`, `file://` or `http(s)://` location of the medium.
    pub url: String,
    /// The generated frame the reference anchors, a negative index counted from the last; absent
    /// for a reference that conditions the generation as a whole.
    #[serde(default)]
    pub frame_index: Option<i64>,
    /// Where a video or audio reference starts, in seconds.
    #[serde(default)]
    pub start_time_seconds: Option<f64>,
}

/// A duration, stated as a number or as a numeric string.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(untagged)]
pub enum Seconds {
    /// A number of seconds.
    Number(f64),
    /// A numeric string of seconds.
    Text(#[serde(deserialize_with = "numeric_text")] f64),
}

/// A numeric string's value.
fn numeric_text<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<f64, D::Error> {
    let text = String::deserialize(deserializer)?;
    text.trim()
        .parse()
        .map_err(|_| serde::de::Error::custom(format!("{text:?} is not a number of seconds")))
}

/// A video generation request, as `/v1/videos` and `/v1/videos/sync` take it.
///
/// `deny_unknown_fields` refuses every field not listed here, at every level; a model refuses
/// what it does not take, and states its own options in `extra_params`.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VideoGenerationRequest {
    /// Served model name.
    pub model: String,
    /// The prompt.
    pub prompt: String,
    /// What the generation steers away from.
    #[serde(default)]
    pub negative_prompt: Option<String>,
    /// The output raster, `WxH` in pixels.
    #[serde(default)]
    pub size: Option<String>,
    /// The requested duration; the model aligns it to a frame count it generates.
    #[serde(default)]
    pub seconds: Option<Seconds>,
    /// The exact frame count; one the model generates.
    #[serde(default)]
    pub num_frames: Option<u32>,
    /// The output's frame rate.
    #[serde(default)]
    pub fps: Option<u32>,
    /// The noise seed.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Denoiser evaluations of the schedule.
    #[serde(default)]
    pub num_inference_steps: Option<u32>,
    /// Classifier-free guidance scale.
    #[serde(default)]
    pub guidance_scale: Option<f64>,
    /// Reference media, in request order.
    #[serde(default)]
    pub input_references: Vec<VideoReference>,
    /// Options the model defines.
    #[serde(default)]
    pub extra_params: Map<String, Value>,
}

/// The kind of value a field of the video request takes, which a multipart text part of it is
/// read as.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    /// Text.
    Text,
    /// A JSON array or object.
    Json,
    /// An unsigned integer.
    Unsigned,
    /// A number.
    Number,
}

/// The fields of the video request body ([`VideoGenerationRequest`]), in schema order, with the
/// kind of value each takes: the one list of them, which a multipart body is read by, a refusal
/// names its parameter from and `GET /v1/capabilities` reports.
pub const REQUEST_FIELDS: [(&str, FieldKind); 12] = [
    ("model", FieldKind::Text),
    ("prompt", FieldKind::Text),
    ("negative_prompt", FieldKind::Text),
    ("size", FieldKind::Text),
    ("seconds", FieldKind::Number),
    ("num_frames", FieldKind::Unsigned),
    ("fps", FieldKind::Unsigned),
    ("seed", FieldKind::Unsigned),
    ("num_inference_steps", FieldKind::Unsigned),
    ("guidance_scale", FieldKind::Number),
    ("input_references", FieldKind::Json),
    ("extra_params", FieldKind::Json),
];

/// The field of the video request body named `name`, with the kind of value it takes.
pub fn request_field(name: &str) -> Option<(&'static str, FieldKind)> {
    REQUEST_FIELDS
        .iter()
        .copied()
        .find(|(field, _)| *field == name)
}

/// The names of the video request body's fields, in schema order.
pub fn request_field_names() -> Vec<&'static str> {
    REQUEST_FIELDS.iter().map(|(name, _)| *name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fields a type's derived `Deserialize` takes, as it states them to its deserializer.
    fn deserialized_fields<T: serde::de::DeserializeOwned>() -> &'static [&'static str] {
        use serde::de::value::Error;

        /// A deserializer that records the fields a struct asks for and reads nothing.
        struct Fields<'a>(&'a mut &'static [&'static str]);

        impl<'de> serde::Deserializer<'de> for Fields<'_> {
            type Error = Error;

            fn deserialize_any<V: serde::de::Visitor<'de>>(self, _: V) -> Result<V::Value, Error> {
                Err(serde::de::Error::custom("not a struct"))
            }

            fn deserialize_struct<V: serde::de::Visitor<'de>>(
                self,
                _: &'static str,
                fields: &'static [&'static str],
                _: V,
            ) -> Result<V::Value, Error> {
                *self.0 = fields;
                Err(serde::de::Error::custom("fields recorded"))
            }

            serde::forward_to_deserialize_any! {
                bool i8 i16 i32 i64 i128 u8 u16 u32 u64 u128 f32 f64 char str string bytes
                byte_buf option unit unit_struct newtype_struct seq tuple tuple_struct map enum
                identifier ignored_any
            }
        }

        let mut fields: &'static [&'static str] = &[];
        let _ = T::deserialize(Fields(&mut fields));
        fields
    }

    /// The field list a multipart body is read by and the capabilities report is exactly the
    /// fields the request schema deserializes, in its order.
    #[test]
    fn the_field_list_is_the_schemas() {
        assert_eq!(
            request_field_names(),
            deserialized_fields::<VideoGenerationRequest>()
        );
    }
}
