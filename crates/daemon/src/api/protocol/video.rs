//! The video generation request, as `/v1/videos` and `/v1/videos/sync` take it.
//!
//! The request is the same for every video model: a prompt, an output size and length, and
//! reference media, each anchoring a generated frame or conditioning the generation as a whole.
//! What a model makes of it is the model's (`processing::VideoProcessor`). A JSON body states it
//! as [`VideoGenerationRequest`]; a multipart form states each field as a text part, which the
//! schema's own field types read ([`from_form`]).

use serde::de::{self, IntoDeserializer, Visitor};
use serde::{Deserialize, Deserializer};
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
    /// The requested duration in seconds, a number; the model aligns it to a frame count it
    /// generates.
    #[serde(default)]
    pub seconds: Option<f64>,
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
    /// Scheduler priority assigned to the request, as a chat request states it.
    #[serde(default)]
    pub priority: Option<i32>,
    /// Reference media, in request order.
    #[serde(default)]
    pub input_references: Vec<VideoReference>,
    /// Options the model defines.
    #[serde(default)]
    pub extra_params: Map<String, Value>,
}

/// The names of the video request's fields, in schema order, as its derived `Deserialize`
/// states them.
pub fn request_field_names() -> &'static [&'static str] {
    serde_aux::serde_introspection::serde_introspect::<VideoGenerationRequest>()
}

/// The request a multipart form's text parts state, each `(field, text)` read as the value its
/// field's type takes: text for a string, the number the text states for a number, and the JSON
/// the text states for the reference list and the model's options.
///
/// # Errors
///
/// Fails, naming the field's path, when a part names no field, states a field twice, or holds
/// text its field's type does not read.
pub fn from_form(
    parts: Vec<(String, String)>,
) -> Result<VideoGenerationRequest, serde_path_to_error::Error<de::value::Error>> {
    let form = de::value::MapDeserializer::new(
        parts
            .into_iter()
            .map(|(field, text)| (field, TextPart(text))),
    );
    serde_path_to_error::deserialize(form)
}

/// The text of one multipart part as the value of a field, read as the field's type asks.
struct TextPart(String);

impl<'de> IntoDeserializer<'de, de::value::Error> for TextPart {
    type Deserializer = Self;

    fn into_deserializer(self) -> Self {
        self
    }
}

impl TextPart {
    /// The JSON value the text states, for a field whose type is structured.
    fn json(&self) -> Result<Value, de::value::Error> {
        serde_json::from_str(&self.0)
            .map_err(|error| de::Error::custom(format!("not JSON: {error}")))
    }

    /// The number the text states.
    fn number<T: std::str::FromStr>(&self) -> Result<T, de::value::Error> {
        self.0
            .parse()
            .map_err(|_| de::Error::custom(format!("{:?} is not a number of its type", self.0)))
    }
}

/// Reads a number of each primitive type from the part's text.
macro_rules! parsed {
    ($($method:ident => $visit:ident: $type:ty),* $(,)?) => {
        $(
            fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
                visitor.$visit(self.number::<$type>()?)
            }
        )*
    };
}

/// Reads a structured value from the JSON the part's text states.
macro_rules! structured {
    ($($method:ident),* $(,)?) => {
        $(
            fn $method<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
                self.json()?.$method(visitor).map_err(de::Error::custom)
            }
        )*
    };
}

impl<'de> Deserializer<'de> for TextPart {
    type Error = de::value::Error;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_string(self.0)
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        visitor.visit_some(self)
    }

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, Self::Error> {
        match self.0.as_str() {
            "true" => visitor.visit_bool(true),
            "false" => visitor.visit_bool(false),
            text => Err(de::Error::custom(format!("{text:?} is not true or false"))),
        }
    }

    parsed! {
        deserialize_i8 => visit_i8: i8,
        deserialize_i16 => visit_i16: i16,
        deserialize_i32 => visit_i32: i32,
        deserialize_i64 => visit_i64: i64,
        deserialize_u8 => visit_u8: u8,
        deserialize_u16 => visit_u16: u16,
        deserialize_u32 => visit_u32: u32,
        deserialize_u64 => visit_u64: u64,
        deserialize_f32 => visit_f32: f32,
        deserialize_f64 => visit_f64: f64,
    }

    structured! {
        deserialize_seq,
        deserialize_map,
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        name: &'static str,
        fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Self::Error> {
        self.json()?
            .deserialize_struct(name, fields, visitor)
            .map_err(de::Error::custom)
    }

    serde::forward_to_deserialize_any! {
        i128 u128 char str string bytes byte_buf unit unit_struct newtype_struct tuple
        tuple_struct enum identifier ignored_any
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The parts of a form stating each field of `body` as its text.
    fn form(body: &[(&str, &str)]) -> Vec<(String, String)> {
        body.iter()
            .map(|(field, text)| ((*field).to_owned(), (*text).to_owned()))
            .collect()
    }

    /// A form's text parts read as the JSON body of the same request reads.
    #[test]
    fn a_form_states_the_request_a_json_body_states() {
        let references =
            r#"[{"type": "image", "url": "data:image/png;base64,AA", "frame_index": 0}]"#;
        let parsed = from_form(form(&[
            ("model", "model"),
            ("prompt", "42"),
            ("seconds", "5.5"),
            ("num_frames", "121"),
            ("guidance_scale", "3"),
            ("input_references", references),
            ("extra_params", r#"{"steps": 4}"#),
        ]))
        .unwrap();

        let json: VideoGenerationRequest = serde_json::from_value(json!({
            "model": "model",
            "prompt": "42",
            "seconds": 5.5,
            "num_frames": 121,
            "guidance_scale": 3.0,
            "input_references": serde_json::from_str::<Value>(references).unwrap(),
            "extra_params": {"steps": 4},
        }))
        .unwrap();
        assert_eq!(parsed, json);
    }

    /// A part its field's type cannot read, a field stated twice, and a part naming no field are
    /// refused, each naming the field.
    #[test]
    fn a_form_part_the_schema_refuses_names_its_field() {
        let error = from_form(form(&[
            ("model", "m"),
            ("prompt", "p"),
            ("num_frames", "many"),
        ]))
        .unwrap_err();
        assert_eq!(error.path().to_string(), "num_frames");

        let error =
            from_form(form(&[("model", "m"), ("prompt", "p"), ("prompt", "q")])).unwrap_err();
        assert!(
            error.to_string().contains("duplicate field `prompt`"),
            "{error}"
        );

        let error =
            from_form(form(&[("model", "m"), ("prompt", "p"), ("width", "4")])).unwrap_err();
        assert!(
            error.to_string().contains("unknown field `width`"),
            "{error}"
        );
    }

    /// The field names the capability report lists are the schema's.
    #[test]
    fn the_field_names_are_the_schemas() {
        assert_eq!(
            request_field_names(),
            [
                "model",
                "prompt",
                "negative_prompt",
                "size",
                "seconds",
                "num_frames",
                "fps",
                "seed",
                "num_inference_steps",
                "guidance_scale",
                "priority",
                "input_references",
                "extra_params",
            ]
        );
    }
}
