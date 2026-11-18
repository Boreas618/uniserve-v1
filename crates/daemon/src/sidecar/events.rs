//! The event plane of the sidecar's workers: each worker's KV events, relayed from its
//! schedulers' publishers in SGLang's format (`kv_events_config`) to Dynamo's ZMQ event plane as
//! Dynamo's router reads them, and its loads.
//!
//! A worker publishes each topic from a PUB socket of its own, bound on every interface at a
//! port the system picks and registered in discovery as the topic's channel
//! (`v1/event_channels/...`, [`wire::channel`]); the KV events' publisher is also registered as
//! the worker's KV event source (`v1/event_sources/...`). Each message is four frames
//! ([`wire::frames`]): the topic, the publisher, the sequence number from 0, and the envelope
//! whose payload is the MessagePack of the message's events (`Vec<RouterEvent>`) or load.

use std::collections::HashMap;

use anyhow::{Context as _, Result};
use storage::events::{Batch, Event, Medium};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};
use zeromq::{Socket as _, SocketRecv as _, SocketSend as _, ZmqMessage};

use super::wire;

/// A ZMQ PUB socket of one topic on the event plane, and its sequence.
pub(super) struct Publisher {
    socket: zeromq::PubSocket,
    /// The publisher's id: what its channel and its messages name it by.
    pub(super) id: u64,
    pub(super) topic: &'static str,
    /// The endpoint a subscriber connects to.
    pub(super) endpoint: String,
    sequence: u64,
}

/// A publisher id as Dynamo draws one: random, within the integers a JSON number holds exactly.
pub(super) fn publisher_id() -> u64 {
    rand_u64() & ((1 << 53) - 1)
}

fn rand_u64() -> u64 {
    let id = uuid::Uuid::new_v4();
    let bytes = id.as_bytes();
    u64::from_le_bytes(bytes[..8].try_into().unwrap_or_default())
}

impl Publisher {
    /// A publisher of `topic` bound on every interface, which subscribers reach at `host`.
    pub(super) async fn bind(topic: &'static str, host: &str) -> Result<Self> {
        let mut socket = zeromq::PubSocket::new();
        let bound = socket
            .bind("tcp://0.0.0.0:0")
            .await
            .context("binding an event-plane publisher")?;
        let port = match bound {
            zeromq::Endpoint::Tcp(_, port) => port,
            other => anyhow::bail!("an event-plane publisher bound {other}"),
        };
        Ok(Self {
            socket,
            id: publisher_id(),
            topic,
            endpoint: format!("tcp://{host}:{port}"),
            sequence: 0,
        })
    }

    /// Publishes `payload`, the MessagePack of the topic's message.
    pub(super) async fn send(&mut self, payload: Vec<u8>) -> Result<()> {
        let published_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| {
                u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
            });
        let envelope = wire::Envelope {
            publisher_id: self.id,
            sequence: self.sequence,
            published_at,
            topic: self.topic,
            payload,
        };
        let [first, rest @ ..] = wire::frames(&envelope)?;
        let mut message = ZmqMessage::from(first);
        for frame in rest {
            message.push_back(frame.into());
        }
        self.socket.send(message).await?;
        self.sequence += 1;
        Ok(())
    }
}

/// The Dynamo KV events of one batch of a scheduler's SGLang KV events: `worker`'s, on rank
/// `rank`, numbered from `next` on, of pages of `block_size` tokens. A store of pages of another
/// size is left out, as Dynamo's own relay leaves it out.
pub(super) fn router_events(
    batch: &Batch,
    worker: u64,
    rank: u32,
    block_size: u32,
    next: &mut u64,
) -> Vec<wire::RouterEvent> {
    let mut events = Vec::new();
    for event in &batch.events {
        let (tier, data) = match event {
            Event::Stored {
                hashes,
                parent,
                tokens,
                block_size: size,
                medium,
            } => {
                if *size != block_size || block_size == 0 {
                    continue;
                }
                let pages = tokens.chunks_exact(block_size as usize);
                let blocks = hashes
                    .iter()
                    .zip(pages)
                    .map(|(hash, page)| wire::Block {
                        // Dynamo takes the engine's signed hash bit for bit.
                        block_hash: *hash as u64,
                        tokens_hash: wire::tokens_hash(page),
                        mm_extra_info: None,
                    })
                    .collect();
                (
                    wire::tier(*medium),
                    wire::KvData::Stored(wire::Stored {
                        parent_hash: parent.map(|parent| parent as u64),
                        start_position: None,
                        blocks,
                    }),
                )
            }
            Event::Removed { hashes, medium } => (
                wire::tier(*medium),
                wire::KvData::Removed(wire::Removed {
                    block_hashes: hashes.iter().map(|hash| *hash as u64).collect(),
                }),
            ),
            Event::Cleared => (wire::tier(Medium::Gpu), wire::KvData::Cleared),
        };
        events.push(wire::RouterEvent {
            worker_id: worker,
            storage_tier: tier,
            residency_domain: "worker",
            event: wire::KvEvent {
                event_id: *next,
                data,
                dp_rank: rank,
            },
        });
        *next += 1;
    }
    events
}

/// What one rank of a worker relays: the endpoint of its scheduler's SGLang publisher and topic.
#[derive(Clone, Debug)]
pub(super) struct Source {
    pub(super) endpoint: String,
    pub(super) topic: String,
    pub(super) rank: u32,
}

/// Relays the KV events of `sources` to `publisher` as `worker`'s, until `stop`: each source's
/// batches as they arrive, every batch's events in one message.
pub(super) async fn relay(
    mut publisher: Publisher,
    sources: Vec<Source>,
    worker: u64,
    block_size: u32,
    stop: CancellationToken,
) {
    let (batches, mut received) = mpsc::unbounded_channel::<(u32, Batch)>();
    for source in sources {
        let batches = batches.clone();
        let stop = stop.clone();
        tokio::spawn(async move {
            tokio::select! {
                () = stop.cancelled() => {}
                result = subscribe(&source, &batches) => {
                    if let Err(error) = result {
                        warn!(endpoint = source.endpoint, "the KV events of rank {} are not relayed: {error:#}", source.rank);
                    }
                }
            }
        });
    }
    drop(batches);

    let mut next = 0u64;
    loop {
        let (rank, batch) = tokio::select! {
            () = stop.cancelled() => return,
            batch = received.recv() => match batch {
                Some(batch) => batch,
                None => return,
            },
        };
        let events = router_events(&batch, worker, rank, block_size, &mut next);
        if events.is_empty() {
            continue;
        }
        let payload = match rmp_serde::to_vec_named(&events) {
            Ok(payload) => payload,
            Err(error) => {
                warn!("a KV event batch does not encode: {error}");
                continue;
            }
        };
        if let Err(error) = publisher.send(payload).await {
            debug!("a KV event message was not sent: {error:#}");
        }
    }
}

/// Reads one scheduler's SGLang batches into `batches`, tagged with its rank.
async fn subscribe(source: &Source, batches: &mpsc::UnboundedSender<(u32, Batch)>) -> Result<()> {
    let mut socket = zeromq::SubSocket::new();
    socket
        .connect(&source.endpoint)
        .await
        .with_context(|| format!("connecting to {}", source.endpoint))?;
    socket.subscribe(&source.topic).await?;
    loop {
        let message = socket.recv().await?;
        let frames: Vec<&[u8]> = message.iter().map(|frame| frame.as_ref()).collect();
        let [_, _, payload] = frames[..] else {
            continue;
        };
        match Batch::decode(payload) {
            Ok(batch) => {
                if batches.send((source.rank, batch)).is_err() {
                    return Ok(());
                }
            }
            Err(error) => debug!("a KV event batch does not decode: {error}"),
        }
    }
}

/// Publishes `loads` of the worker's ranks, a message per rank, each time they change.
pub(super) async fn publish_loads(publisher: &mut Publisher, loads: &[wire::Load]) {
    for load in loads {
        match rmp_serde::to_vec_named(load) {
            Ok(payload) => {
                if let Err(error) = publisher.send(payload).await {
                    debug!("a load message was not sent: {error:#}");
                }
            }
            Err(error) => warn!("a load does not encode: {error}"),
        }
    }
}

/// The KV pages each scheduler holds now, by its number across the deployment: its last
/// reported cache usage over the pages its cache holds.
pub(super) fn used_blocks(
    rows: &serde_json::Value,
    totals: &HashMap<u32, u64>,
) -> HashMap<u32, u64> {
    let mut used = HashMap::new();
    let Some(rows) = rows.as_array() else {
        return used;
    };
    for row in rows {
        let number = |field: &str| {
            row.get(field)
                .and_then(serde_json::Value::as_str)
                .and_then(|text| text.parse::<u32>().ok())
        };
        let (Some(replica), Some(scheduler)) = (number("replica"), number("scheduler")) else {
            continue;
        };
        let Some(ratio) = row.get("value").and_then(serde_json::Value::as_f64) else {
            continue;
        };
        let at = replica + scheduler;
        if let Some(total) = totals.get(&at) {
            used.insert(at, (ratio.clamp(0.0, 1.0) * *total as f64).round() as u64);
        }
    }
    used
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A batch of SGLang events becomes Dynamo's KV events of the worker and rank: a store with
    /// each page's engine hash and Dynamo's hash of its tokens, a removal, and a clearing; a store
    /// of pages of another size is left out.
    #[test]
    fn sglang_batches_become_dynamos_router_events() {
        let batch = Batch {
            time: 1.0,
            events: vec![
                Event::Cleared,
                Event::Stored {
                    hashes: vec![-5, 6],
                    parent: Some(-1),
                    tokens: vec![1, 2, 3, 4],
                    block_size: 2,
                    medium: Medium::Gpu,
                },
                Event::Stored {
                    hashes: vec![9],
                    parent: None,
                    tokens: vec![1, 2, 3],
                    block_size: 3,
                    medium: Medium::Gpu,
                },
                Event::Removed {
                    hashes: vec![6],
                    medium: Medium::CpuPinned,
                },
            ],
            rank: Some(0),
        };
        let mut next = 10;
        let events = router_events(&batch, 42, 1, 2, &mut next);
        assert_eq!(next, 13);
        let data: Vec<_> = events
            .iter()
            .map(|event| {
                (
                    event.event.event_id,
                    event.storage_tier,
                    event.event.data.clone(),
                )
            })
            .collect();
        assert_eq!(
            data,
            [
                (10, "device", wire::KvData::Cleared),
                (
                    11,
                    "device",
                    wire::KvData::Stored(wire::Stored {
                        parent_hash: Some(u64::MAX),
                        start_position: None,
                        blocks: vec![
                            wire::Block {
                                block_hash: (-5i64) as u64,
                                tokens_hash: wire::tokens_hash(&[1, 2]),
                                mm_extra_info: None,
                            },
                            wire::Block {
                                block_hash: 6,
                                tokens_hash: wire::tokens_hash(&[3, 4]),
                                mm_extra_info: None,
                            },
                        ],
                    })
                ),
                (
                    12,
                    "host_pinned",
                    wire::KvData::Removed(wire::Removed {
                        block_hashes: vec![6]
                    })
                ),
            ]
        );
        assert!(
            events
                .iter()
                .all(|event| event.worker_id == 42 && event.event.dp_rank == 1)
        );
    }
}
