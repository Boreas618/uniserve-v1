//! HTTP transport: one request through its endpoint's API, recorded as it
//! observably went.
//!
//! Every request of a run goes through one `reqwest` client with an unbounded
//! connection pool, the client shape of the SGLang and vLLM serving benchmarks.
//! A JSON response is read whole before its record closes; a streamed one is
//! read as it arrives and each event is stamped as its dispatching line is read.

pub mod images;
pub mod sse;
pub mod video;

use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use serde_json::{Map, Value};

use crate::record::{Clock, Record, TaskRequest};
use crate::spec::IMAGES_GENERATIONS;

/// What a request is sent with beside its body.
pub struct Send<'a> {
    pub client: &'a reqwest::Client,
    pub clock: &'a Clock,
    pub base_url: &'a str,
    pub request_id: String,
    pub task: &'a str,
    /// Token counts that stand in for those the server does not report: the
    /// prompt's, and the output limit asked for (the record's
    /// `requested_output_len` on every endpoint).
    pub prompt_len: u64,
    pub output_len: u64,
    pub scheduled_time: Option<f64>,
    pub timeout: Duration,
}

/// Sends one request through its transport: a video request (one that carries
/// its point's video settings) through the video API, an image-generation one
/// through the images API, the others as a streamed or a whole chat completion.
///
/// The returned record is closed. HTTP and content failures carry their own
/// classifiers; the request's one deadline is `request_deadline`, any other
/// error `transport_failure`.
pub async fn send(send: Send<'_>, request: &TaskRequest) -> Record {
    // A JSON null field is left out of the body.
    let payload: Map<String, Value> = request
        .payload
        .iter()
        .filter(|(_, value)| !value.is_null())
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let url = format!(
        "{}{}",
        send.base_url.trim_end_matches('/'),
        request.endpoint
    );
    let mut record = Record::new(send.request_id.clone(), send.task);
    record.begin(
        send.clock,
        &request.endpoint,
        send.scheduled_time,
        send.output_len,
    );
    record.base_url = send.base_url.to_owned();
    let outcome = tokio::time::timeout(send.timeout, async {
        if request.video.is_some() {
            record.prompt_len = send.prompt_len;
            record.video_shape = request.video_shape;
            video::receive(send.client, send.clock, send.base_url, request, &mut record).await
        } else if request.endpoint == IMAGES_GENERATIONS {
            images::generations(send.client, send.clock, &url, &payload, &mut record).await
        } else if request.stream {
            chat_stream(&send, &url, &payload, &mut record).await
        } else {
            chat(&send, &url, &payload, &mut record).await
        }
    })
    .await;
    match outcome {
        Ok(Ok(())) => {}
        Ok(Err(error)) => record.mark_transport_failure(send.clock, &error),
        Err(_) => {
            let deadline = record.start_time + send.timeout.as_secs_f64();
            record.close_at(deadline);
            record.mark_failure(
                "request_deadline",
                Some(format!(
                    "logical request exceeded {}s",
                    send.timeout.as_secs_f64()
                )),
            );
        }
    }
    record
}

/// A complete response: its status, body and content type.
pub struct Body {
    pub status: u16,
    pub data: Vec<u8>,
    pub content_type: String,
}

impl Body {
    /// The body as UTF-8, invalid bytes replaced, at most `limit` characters.
    pub fn text(&self, limit: usize) -> String {
        String::from_utf8_lossy(&self.data)
            .chars()
            .take(limit)
            .collect()
    }
}

/// POSTs a JSON body and reads the whole response into `record`: the status
/// when the headers arrive, the close once the body is read, before the caller
/// parses it.
pub async fn post_json(
    client: &reqwest::Client,
    clock: &Clock,
    url: &str,
    payload: &Map<String, Value>,
    record: &mut Record,
) -> Result<Body> {
    let response = client.post(url).json(payload).send().await?;
    let status = response.status().as_u16();
    record.note_http(clock, status);
    let content_type = header(&response, "content-type");
    let data = response.bytes().await?.to_vec();
    record.close_now(clock);
    Ok(Body {
        status,
        data,
        content_type,
    })
}

/// A response header's value, empty when absent.
pub fn header(response: &reqwest::Response, name: &str) -> String {
    response
        .headers()
        .get(name)
        .and_then(|value| value.to_str().ok())
        .unwrap_or_default()
        .to_owned()
}

/// A whole chat completion: its text, images and usage. A single body has no
/// per-token arrival times, so TTFT and ITL are unavailable and every image
/// takes the whole response's latency.
async fn chat(
    send: &Send<'_>,
    url: &str,
    payload: &Map<String, Value>,
    record: &mut Record,
) -> Result<()> {
    let body = post_json(send.client, send.clock, url, payload, record).await?;
    let Ok(data) = serde_json::from_slice::<Value>(&body.data) else {
        record.mark_failure(
            &format!("transport_status_{}", body.status),
            Some(body.text(500)),
        );
        return Ok(());
    };
    let choice = data
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .and_then(Value::as_object);
    let message = choice
        .and_then(|choice| choice.get("message"))
        .and_then(Value::as_object);
    if let Some(choice) = choice {
        record.apply_choice(choice);
    }
    let content = sse::message_text(message);
    let images = sse::message_images(message);
    record.generated_text = content.clone();
    record.token_timing_available = false;
    let image_error = images::attach(record, &images, true);
    if let Some(usage) = data.get("usage").and_then(Value::as_object) {
        record.apply_usage(usage);
    }
    record.apply_token_fallbacks(send.prompt_len, send.output_len);
    // Usage is recorded even when the images already failed the request.
    if image_error {
        return Ok(());
    }
    if body.status < 400 && (!content.is_empty() || !record.decoded_images.is_empty()) {
        record.mark_success();
    } else if body.status >= 400 {
        record.mark_failure(&format!("transport_status_{}", body.status), None);
    } else {
        record.mark_failure("empty_completion", None);
    }
    Ok(())
}

/// A streamed chat completion, each event folded into the record as it
/// arrives and none kept. Latency, TTFT, ITL and image latencies come from the
/// events' stamps; a text gap that spans an image event is left out of ITL.
async fn chat_stream(
    send: &Send<'_>,
    url: &str,
    payload: &Map<String, Value>,
    record: &mut Record,
) -> Result<()> {
    let clock = send.clock;
    let response = send.client.post(url).json(payload).send().await?;
    let status = response.status().as_u16();
    record.note_http(clock, status);
    // Any status but 200 fails, other 2xx codes included.
    if status != 200 {
        let body = response.bytes().await.unwrap_or_default();
        record.close_now(clock);
        let detail: String = String::from_utf8_lossy(&body).chars().take(500).collect();
        record.mark_failure(&format!("transport_status_{status}"), Some(detail));
        return Ok(());
    }
    // A JSON body in reply to a stream request carries no event timing.
    if is_json(&header(&response, "content-type")) {
        let _ = response.bytes().await;
        record.close_now(clock);
        record.mark_failure(
            "response_expected_sse",
            Some("stream request received a JSON response".into()),
        );
        return Ok(());
    }
    // The fallback counts are written first; the stream's usage replaces them.
    record.prompt_len = send.prompt_len;
    record.output_len = send.output_len;
    record.token_timing_available = false;
    let mut outcome = sse::Outcome::default();
    let mut parser = sse::Parser::default();
    let mut lines = sse::Lines::default();
    let mut state = Fold::default();
    let mut body = response.bytes_stream();
    // A stream whose transport fails keeps what it folded before the failure.
    let mut failure = None;
    while let Some(chunk) = body.next().await {
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(error) => {
                failure = Some(error);
                break;
            }
        };
        for line in lines.push(&chunk) {
            if let Some(event) = parser.feed(&line) {
                state.fold(&event, clock.now(), record, &mut outcome);
            }
        }
    }
    if let Some(line) = lines.finish()
        && let Some(event) = parser.feed(&line)
    {
        state.fold(&event, clock.now(), record, &mut outcome);
    }
    if let Some(event) = parser.finish() {
        state.fold(&event, clock.now(), record, &mut outcome);
    }
    record.close_at(state.last_event.unwrap_or_else(|| clock.now()));
    if let Some(error) = failure {
        record.mark_failure("transport_failure", Some(error.to_string()));
        return Ok(());
    }
    match outcome.classify() {
        (true, _) => record.mark_success(),
        // An error event is the server's own statement of the failure, which
        // the record keeps beside its classifier.
        (false, classifier) => record.mark_failure(classifier, outcome.error.clone()),
    }
    // Images decode once timing is folded; a decode failure replaces the
    // stream's classification.
    images::attach(record, &state.images, false);
    Ok(())
}

/// What a stream's folding carries from one event to the next.
#[derive(Default)]
struct Fold {
    last_event: Option<f64>,
    last_text: Option<f64>,
    image_since_text: bool,
    images: Vec<Map<String, Value>>,
}

impl Fold {
    fn fold(
        &mut self,
        event: &sse::Event,
        time: f64,
        record: &mut Record,
        outcome: &mut sse::Outcome,
    ) {
        outcome.observe(event);
        self.last_event = Some(self.last_event.map_or(time, |last| last.max(time)));
        let payload = match event {
            sse::Event::Done => return,
            sse::Event::ParseError(_) => {
                record.add_chunk(time);
                return;
            }
            sse::Event::Json(Value::Object(payload)) => payload,
            sse::Event::Json(_) => return,
        };
        record.add_chunk(time);
        for choice in payload
            .get("choices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(choice) = choice.as_object() {
                record.apply_choice(choice);
                record.add_tokens(choice, Some(time));
            }
        }
        if let Some(usage) = payload.get("usage").and_then(Value::as_object) {
            record.apply_usage(usage);
        }
        let content = sse::delta_text(payload);
        let images = sse::delta_images(payload);
        if !content.is_empty() {
            record.add_text(&content, Some(time), self.last_text, !self.image_since_text);
            self.last_text = Some(time);
            self.image_since_text = false;
        }
        if !images.is_empty() {
            record.add_image_arrival(images.len(), Some(time));
            self.images.extend(images);
            self.image_since_text = true;
        }
    }
}

/// Whether a media type is JSON.
fn is_json(content_type: &str) -> bool {
    let media = content_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    media == "application/json" || media.ends_with("+json")
}
