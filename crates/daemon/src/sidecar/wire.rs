//! What the sidecar writes and reads on Dynamo's planes, as Dynamo's own types state it
//! (`lib/runtime` and `lib/llm` of `ai-dynamo/dynamo`): the records of its discovery, the
//! request a frontend sends and the stream a worker answers with, the KV events and loads of its
//! event plane, and the key names of the planner's virtual connector.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use xxhash_rust::xxh3::xxh3_64_with_seed;

/// The seed of Dynamo's block hash over a page's tokens (`CHAIN_XXH3_SEED`), for a page of no
/// adapter and no cache salt.
pub(super) const BLOCK_SEED: u64 = 1337;

/// The event plane's topic of the KV events, and of the workers' loads.
pub(super) const KV_EVENTS: &str = "kv-events";
pub(super) const KV_METRICS: &str = "kv_metrics";

/// The endpoint every worker serves, as Dynamo's sidecars name it.
pub(super) const ENDPOINT: &str = "generate";

/// The component of the decode and aggregated workers, and of the prefill workers.
pub(super) const DECODE: &str = "backend";
pub(super) const PREFILL: &str = "prefill";

/// What a worker does in a deployment Dynamo pairs: its component and its card's worker type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Kind {
    /// It computes prompts only.
    Prefill,
    /// It generates tokens only, after a prefill worker computed the prompt.
    Decode,
    /// It does both.
    Aggregated,
}

impl Kind {
    /// The component it registers under.
    pub(super) fn component(self) -> &'static str {
        match self {
            Self::Prefill => PREFILL,
            Self::Decode | Self::Aggregated => DECODE,
        }
    }

    fn worker_type(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Aggregated => "aggregated",
        }
    }
}

/// A field of a TCP address as Dynamo escapes it: `%` then `/`.
fn escape(field: &str) -> String {
    field.replace('%', "%25").replace('/', "%2F")
}

/// The path a frontend names a worker's endpoint by in a request frame, the part of its
/// instance's `tcp` transport after the address.
pub(super) fn endpoint_path(instance: u64, namespace: &str, component: &str) -> String {
    format!(
        "{instance:x}/{}/{}/{}",
        escape(namespace),
        escape(component),
        escape(ENDPOINT)
    )
}

/// The discovery key of a worker's instance (`v1/instances/...`) and of its card (`v1/mdc/...`).
pub(super) fn instance_key(namespace: &str, component: &str, instance: u64) -> String {
    format!("v1/instances/{namespace}/{component}/{ENDPOINT}/{instance:x}")
}

pub(super) fn card_key(namespace: &str, component: &str, instance: u64) -> String {
    format!("v1/mdc/{namespace}/{component}/{ENDPOINT}/{instance:x}")
}

/// A segment of an event key, percent-encoded but for `[A-Za-z0-9_-]`.
fn segment(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-' {
            out.push(char::from(byte));
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}

/// The discovery key of an event publisher's channel (`v1/event_channels/...`) or source
/// (`v1/event_sources/...`), of the endpoint of `component` and `topic`.
pub(super) fn event_key(
    bucket: &str,
    namespace: &str,
    component: &str,
    topic: &str,
    publisher: u64,
) -> String {
    format!(
        "v1/{bucket}/namespace/{}/component/{}/endpoint/{}/topic/{}/{publisher:x}",
        segment(namespace),
        segment(component),
        segment(ENDPOINT),
        segment(topic)
    )
}

/// A worker's instance record: its endpoint and the TCP address and path a frontend sends its
/// requests to, in JSON.
pub(super) fn instance(namespace: &str, component: &str, instance: u64, address: &str) -> Value {
    json!({
        "type": "Endpoint",
        "component": component,
        "endpoint": ENDPOINT,
        "namespace": namespace,
        "instance_id": instance,
        "transport": {"tcp": format!("{address}/{}", endpoint_path(instance, namespace, component))},
        "device_type": "cuda",
        "request_plane_codec": "json",
    })
}

/// What a worker's model card states of the model it serves and of the worker.
pub(super) struct Card<'a> {
    pub(super) namespace: &'a str,
    pub(super) instance: u64,
    pub(super) kind: Kind,
    /// The served model's name.
    pub(super) model: &'a str,
    /// The checkpoint the cards of one model name in common.
    pub(super) source: &'a str,
    pub(super) context_length: u32,
    /// Tokens per KV page.
    pub(super) block_size: u32,
    pub(super) total_blocks: Option<u64>,
    pub(super) max_num_seqs: u64,
    pub(super) max_num_batched_tokens: Option<u64>,
    /// The data-parallel ranks the worker hosts: its schedulers.
    pub(super) ranks: u32,
    /// Where a prefill worker states its bootstrap, which makes a frontend send its decode
    /// request beside its prefill request.
    pub(super) bootstrap: Option<(&'a str, u16)>,
    /// The checkpoint's files a frontend processes requests with, by the card's field
    /// (`model_info`, `tokenizer`, `prompt_formatter`, `gen_config`).
    pub(super) files: serde_json::Map<String, Value>,
}

impl Card<'_> {
    /// The card's discovery record (`v1/mdc/...`), in JSON.
    pub(super) fn record(&self) -> Value {
        let (model_type, needs) = match self.kind {
            Kind::Prefill => ("Prefill", json!([["decode"]])),
            Kind::Decode => ("Chat | Completions", json!([["prefill"]])),
            Kind::Aggregated => ("Chat | Completions", json!([])),
        };
        let mut runtime = json!({
            "context_length": self.context_length,
            "total_kv_blocks": self.total_blocks,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "tool_call_parser": null,
            "reasoning_parser": null,
            "data_parallel_start_rank": 0,
            "data_parallel_size": self.ranks,
            "enable_local_indexer": true,
            "kv_event_publishing_enabled": true,
            "enable_eagle": false,
            "exclude_tools_when_tool_choice_none": true,
        });
        if let Some((host, port)) = self.bootstrap {
            runtime["disaggregated_endpoint"] =
                json!({"bootstrap_host": host, "bootstrap_port": port});
        }
        let slug: String = self
            .model
            .to_lowercase()
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let mut card = json!({
                "display_name": self.model,
                "slug": slug,
                "source_path": self.source,
                "model_info": null,
                "tokenizer": null,
                "context_length": self.context_length,
                "kv_cache_block_size": self.block_size,
                "migration_limit": 0,
                "model_type": model_type,
                "model_input": "Tokens",
                "worker_type": self.kind.worker_type(),
                "needs": needs,
                "runtime_config": runtime,
                "media_decoder": null,
                "media_fetcher": null,
        });
        for (field, file) in &self.files {
            card[field] = file.clone();
        }
        json!({
            "type": "Model",
            "namespace": self.namespace,
            "component": self.kind.component(),
            "endpoint": ENDPOINT,
            "instance_id": self.instance,
            "card_json": card,
        })
    }
}

/// An event publisher's channel record (`v1/event_channels/...`): where its ZMQ PUB socket is.
pub(super) fn channel(
    namespace: &str,
    component: &str,
    topic: &str,
    publisher: u64,
    endpoint: &str,
) -> Value {
    json!({
        "type": "EventChannel",
        "scope": {"kind": "endpoint", "endpoint": {"namespace": namespace, "component": component, "name": ENDPOINT}},
        "topic": topic,
        "instance_id": publisher,
        "transport": {"kind": "Zmq", "config": {"endpoint": endpoint}},
    })
}

/// A KV event source's record (`v1/event_sources/...`): the worker and rank whose KV events the
/// publisher publishes, live only.
pub(super) fn source(
    namespace: &str,
    component: &str,
    publisher: u64,
    worker: u64,
    rank: u32,
) -> Value {
    let endpoint = json!({"namespace": namespace, "component": component, "name": ENDPOINT});
    json!({
        "type": "EventSource",
        "scope": {"kind": "endpoint", "endpoint": endpoint},
        "topic": KV_EVENTS,
        "publisher_id": publisher,
        "metadata": {
            "kv_state_endpoint": endpoint,
            "worker": {"worker_id": worker, "dp_rank": rank},
            "publisher_id": publisher,
        },
    })
}

/// The control message a request frame's header part carries.
#[derive(Clone, Debug, Deserialize)]
pub(super) struct Control {
    /// The request's context, which its response connection names again.
    pub(super) id: String,
    #[serde(default)]
    pub(super) payload_codec: Option<String>,
    pub(super) connection_info: Connection,
}

/// Where the worker connects back to stream the response.
#[derive(Clone, Debug, Deserialize)]
pub(super) struct Connection {
    pub(super) transport: String,
    /// The JSON of a [`Home`].
    pub(super) info: String,
}

/// The frontend's response server, and the stream the worker states in its handshake.
#[derive(Clone, Debug, Deserialize)]
pub(super) struct Home {
    pub(super) address: String,
    pub(super) subject: String,
    pub(super) context: String,
    pub(super) stream_type: String,
}

/// A token request as a frontend sends it (`PreprocessedRequest`), of what the sidecar serves.
#[derive(Clone, Debug, Default, Deserialize)]
pub(super) struct Request {
    pub(super) token_ids: Vec<u32>,
    #[serde(default)]
    pub(super) stop_conditions: Stops,
    #[serde(default)]
    pub(super) sampling_options: Sampling,
    #[serde(default)]
    pub(super) eos_token_ids: Vec<u32>,
    #[serde(default)]
    pub(super) routing: Option<Routing>,
    #[serde(default)]
    pub(super) bootstrap_info: Option<Bootstrap>,
    #[serde(default)]
    pub(super) prefill_result: Option<PrefillResult>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(super) struct Stops {
    pub(super) max_tokens: Option<u32>,
    pub(super) stop_token_ids: Option<Vec<u32>>,
    pub(super) min_tokens: Option<u32>,
    pub(super) ignore_eos: Option<bool>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(super) struct Sampling {
    pub(super) temperature: Option<f32>,
    pub(super) top_p: Option<f32>,
    pub(super) top_k: Option<i32>,
    pub(super) min_p: Option<f32>,
    pub(super) seed: Option<i64>,
    pub(super) presence_penalty: Option<f32>,
    pub(super) frequency_penalty: Option<f32>,
    pub(super) repetition_penalty: Option<f32>,
}

/// The workers a frontend chose for a request, of what the sidecar reads: the prefill worker of
/// a disaggregated request's decode half. The worker that receives a request is the one the
/// frontend chose for it.
#[derive(Clone, Debug, Default, Deserialize)]
pub(super) struct Routing {
    pub(super) prefill_worker_id: Option<u64>,
}

/// What pairs a prefill request with its decode request.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub(super) struct Bootstrap {
    pub(super) bootstrap_host: String,
    pub(super) bootstrap_port: u16,
    pub(super) bootstrap_room: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct PrefillResult {
    pub(super) disaggregated_params: Value,
}

impl Request {
    /// The room that pairs it with the other half of its disaggregated request: its bootstrap's,
    /// or the one its prefill's result states.
    pub(super) fn room(&self) -> Option<u64> {
        self.bootstrap_info
            .as_ref()
            .map(|bootstrap| bootstrap.bootstrap_room)
            .or_else(|| {
                self.prefill_result
                    .as_ref()?
                    .disaggregated_params
                    .get("bootstrap_room")?
                    .as_u64()
            })
    }
}

/// One frame of a response stream's data: an output of the engine (`Annotated<LLMEngineOutput>`
/// in a `NetworkStreamWrapper`), or the stream's end.
pub(super) fn output(output: Value) -> Value {
    json!({"data": {"data": output}, "complete_final": false})
}

/// The data frame that ends a response stream.
pub(super) fn complete() -> Value {
    json!({"complete_final": true})
}

/// The data frame of an error, which ends the stream's outputs.
pub(super) fn error(message: &str) -> Value {
    json!({
        "data": {"event": "error", "error": {"error_type": "Unknown", "message": message}},
        "complete_final": false,
    })
}

/// Dynamo's block hash of a page's tokens (`compute_block_hash_for_seq`): XXH3 over the tokens as
/// little-endian `u32`s, seeded with [`BLOCK_SEED`].
pub(super) fn tokens_hash(tokens: &[u32]) -> u64 {
    let bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect();
    xxh3_64_with_seed(&bytes, BLOCK_SEED)
}

/// A KV event of one worker (`RouterEvent`), as the event plane carries it.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct RouterEvent {
    pub(super) worker_id: u64,
    pub(super) storage_tier: &'static str,
    pub(super) residency_domain: &'static str,
    pub(super) event: KvEvent,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct KvEvent {
    pub(super) event_id: u64,
    pub(super) data: KvData,
    pub(super) dp_rank: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum KvData {
    Stored(Stored),
    Removed(Removed),
    Cleared,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct Stored {
    pub(super) parent_hash: Option<u64>,
    pub(super) start_position: Option<u32>,
    pub(super) blocks: Vec<Block>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct Block {
    pub(super) block_hash: u64,
    pub(super) tokens_hash: u64,
    pub(super) mm_extra_info: Option<()>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct Removed {
    pub(super) block_hashes: Vec<u64>,
}

/// The storage tier Dynamo names a KV event's medium by.
pub(super) fn tier(medium: storage::events::Medium) -> &'static str {
    use storage::events::Medium;
    match medium {
        Medium::Gpu => "device",
        Medium::CpuPinned => "host_pinned",
        Medium::Disk => "disk",
        Medium::External => "external",
    }
}

/// A worker's load on one rank (`ActiveLoad`), which the router weighs it by.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct Load {
    pub(super) worker_id: u64,
    pub(super) dp_rank: u32,
    pub(super) active_decode_blocks: Option<u64>,
    pub(super) active_prefill_tokens: Option<u64>,
    pub(super) kv_used_blocks: Option<u64>,
}

/// The envelope of one event-plane message (`EventEnvelope`).
#[derive(Clone, Debug, Serialize)]
pub(super) struct Envelope<'a> {
    pub(super) publisher_id: u64,
    pub(super) sequence: u64,
    /// Milliseconds since the Unix epoch.
    pub(super) published_at: u64,
    pub(super) topic: &'a str,
    #[serde(with = "serde_bytes")]
    pub(super) payload: Vec<u8>,
}

/// The four frames a ZMQ event publisher sends an envelope as: the topic, the publisher and the
/// sequence number (eight big-endian bytes each), and the envelope in MessagePack after its
/// frame header (version 1 and the envelope's length as four big-endian bytes).
pub(super) fn frames(envelope: &Envelope<'_>) -> Result<[Vec<u8>; 4], rmp_serde::encode::Error> {
    let encoded = rmp_serde::to_vec_named(envelope)?;
    let mut frame = Vec::with_capacity(5 + encoded.len());
    frame.push(1);
    frame.extend_from_slice(
        &u32::try_from(encoded.len())
            .unwrap_or(u32::MAX)
            .to_be_bytes(),
    );
    frame.extend_from_slice(&encoded);
    Ok([
        envelope.topic.as_bytes().to_vec(),
        envelope.publisher_id.to_be_bytes().to_vec(),
        envelope.sequence.to_be_bytes().to_vec(),
        frame,
    ])
}

/// The planner's virtual connector: the prefix of its keys in `namespace`, and their names.
pub(super) fn planner_prefix(namespace: &str) -> String {
    format!("v1/{namespace}/planner/")
}

pub(super) const PREFILL_WORKERS: &str = "num_prefill_workers";
pub(super) const DECODE_WORKERS: &str = "num_decode_workers";
pub(super) const DECISION: &str = "decision_id";
pub(super) const SCALED: &str = "scaled_decision_id";

#[cfg(test)]
mod tests {
    use super::*;

    /// Dynamo's block hash of a page's tokens equals its own test vector's construction: XXH3 of
    /// the tokens' little-endian bytes under seed 1337.
    #[test]
    fn tokens_hash_is_xxh3_of_little_endian_tokens_under_dynamos_seed() {
        let bytes: Vec<u8> = [1u32, 2, 3, 4]
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        assert_eq!(tokens_hash(&[1, 2, 3, 4]), xxh3_64_with_seed(&bytes, 1337));
        // `compute_block_hash(b"test data")` of `lib/kv-events/src/event.rs`.
        assert_eq!(xxh3_64_with_seed(b"test data", 1337), 13226331709069118873);
    }

    /// A KV event's MessagePack names its variant as Dynamo's `rmp_serde` encoding does: a map
    /// of one entry for a store or a removal, a string for a clearing.
    #[test]
    fn router_events_encode_as_dynamo_decodes_them() {
        let event = |data| RouterEvent {
            worker_id: 7,
            storage_tier: "device",
            residency_domain: "worker",
            event: KvEvent {
                event_id: 3,
                data,
                dp_rank: 1,
            },
        };
        let decoded = |data| -> serde_json::Value {
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&vec![event(data)]).unwrap()).unwrap()
        };
        assert_eq!(
            decoded(KvData::Stored(Stored {
                parent_hash: None,
                start_position: None,
                blocks: vec![Block {
                    block_hash: 9,
                    tokens_hash: 11,
                    mm_extra_info: None,
                }],
            })),
            json!([{
                "worker_id": 7,
                "storage_tier": "device",
                "residency_domain": "worker",
                "event": {
                    "event_id": 3,
                    "data": {"stored": {"parent_hash": null, "start_position": null,
                        "blocks": [{"block_hash": 9, "tokens_hash": 11, "mm_extra_info": null}]}},
                    "dp_rank": 1,
                },
            }])
        );
        assert_eq!(
            decoded(KvData::Cleared)[0]["event"]["data"],
            json!("cleared")
        );
    }
}
