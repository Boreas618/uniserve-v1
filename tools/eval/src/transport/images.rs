//! The images API, and the embedded image parts chat responses carry.

use anyhow::Result;
use base64::Engine;
use serde_json::{Map, Value};

use crate::record::{Clock, Record};

/// An image-generation request. A status of 400 or more, or a body that is not
/// JSON, is `transport_status_<code>` whatever it holds (error bodies follow no
/// images schema); otherwise every entry is decoded, each taking the whole
/// response's latency.
pub async fn generations(
    client: &reqwest::Client,
    clock: &Clock,
    url: &str,
    payload: &Map<String, Value>,
    record: &mut Record,
) -> Result<()> {
    let body = super::post_json(client, clock, url, payload, record).await?;
    let data = match serde_json::from_slice::<Value>(&body.data) {
        Ok(data) if body.status < 400 => data,
        _ => {
            record.mark_failure(
                &format!("transport_status_{}", body.status),
                Some(body.text(500)),
            );
            return Ok(());
        }
    };
    let entries = match data.get("data").and_then(Value::as_array) {
        Some(entries) if !entries.is_empty() => entries,
        _ => {
            record.mark_failure("empty_image_data", None);
            return Ok(());
        }
    };
    let first_payload = entries[0]
        .as_object()
        .and_then(|first| first.get("b64_json"))
        .and_then(Value::as_str)
        .is_some_and(|encoded| !encoded.is_empty());
    if !first_payload {
        record.mark_failure("missing_image_payload", None);
        return Ok(());
    }
    let Some(parts) = entries
        .iter()
        .map(|entry| entry.as_object().cloned())
        .collect::<Option<Vec<_>>>()
    else {
        record.mark_failure("invalid_image_part", Some("invalid_image_part".into()));
        return Ok(());
    };
    if !attach(record, &parts, true) {
        record.mark_success();
    }
    Ok(())
}

/// Decodes image parts into the record; returns whether one failed, in which
/// case the record carries the failure's classifier.
pub fn attach(record: &mut Record, parts: &[Map<String, Value>], json_latency: bool) -> bool {
    match parts
        .iter()
        .map(decode_part)
        .collect::<Result<Vec<_>, &'static str>>()
    {
        Ok(decoded) => {
            record.attach_images(decoded, json_latency);
            false
        }
        Err(classifier) => {
            record.mark_failure(classifier, Some(classifier.to_owned()));
            true
        }
    }
}

/// One embedded image part: a non-empty `b64_json`, else an `image_url` (a
/// string or `{"url": ...}`) that must be a base64 `data:` URL whose declared
/// type matches the decoded format.
fn decode_part(part: &Map<String, Value>) -> Result<crate::record::DecodedImage, &'static str> {
    let (declared, encoded) = match part
        .get("b64_json")
        .and_then(Value::as_str)
        .filter(|encoded| !encoded.is_empty())
    {
        Some(encoded) => (None, encoded.to_owned()),
        None => {
            let url = match part.get("image_url") {
                Some(Value::Object(inner)) => inner.get("url").and_then(Value::as_str),
                Some(Value::String(url)) => Some(url.as_str()),
                _ => None,
            }
            .filter(|url| !url.is_empty())
            .ok_or("response_missing_image_payload")?;
            let (mime, encoded) = data_url(url)?;
            (Some(mime), encoded)
        }
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(encoded.as_bytes())
        .map_err(|_| "response_invalid_image_base64")?;
    if bytes.is_empty() {
        return Err("response_empty_image_payload");
    }
    crate::media::inspect_image(bytes, declared.as_deref())
}

/// A base64 image data URL's type and payload.
fn data_url(value: &str) -> Result<(String, String), &'static str> {
    let rest = value
        .strip_prefix("data:")
        .ok_or("response_image_payload_not_embedded")?;
    let (header, encoded) = rest
        .split_once(',')
        .ok_or("response_image_payload_not_embedded")?;
    let mut segments = header.split(';');
    let mime = segments.next().unwrap_or_default().to_lowercase();
    if !mime.starts_with("image/") || !segments.any(|segment| segment == "base64") {
        return Err("response_invalid_image_data_url");
    }
    if encoded.is_empty() {
        return Err("response_empty_image_payload");
    }
    Ok((mime, encoded.to_owned()))
}
