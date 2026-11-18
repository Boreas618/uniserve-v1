//! The KV events of the pages a cache holds, in SGLang's ZMQ format.
//!
//! A holder of KV pages (each scheduler of an engine, for the pages its queue pair commits and
//! learns evicted) publishes which pages it holds on which medium as a stream of event batches,
//! the format SGLang's `ZmqEventPublisher` publishes (`python/sglang/srt/disaggregation/
//! kv_events.py`) and its consumers read: Dynamo's router, `sgl-router` and Mooncake's
//! conductor. This module is the format: the events ([`Event`]), their batch ([`Batch`]) in
//! msgspec's MessagePack layout, the multipart message a publisher sends ([`frames`]) and the
//! block hashes the events name pages by ([`Chain`]).
//!
//! # The wire
//!
//! A batch is a MessagePack array `[ts, events, attn_dp_rank]`: the publisher's wall clock in
//! seconds (a float64), the events in the order they happened, and the publisher's
//! data-parallel rank. Each event is a map whose `type` entry names its class and whose other
//! entries are its fields, an optional field left unset omitted:
//!
//! | `type` | Fields |
//! | --- | --- |
//! | `BlockStored` | `block_hashes` (int64 each), `parent_block_hash` (int64 or nil), `token_ids` (the pages' tokens, concatenated), `block_size` (tokens per page), `lora_id` (nil), `medium` |
//! | `BlockRemoved` | `block_hashes`, `medium` |
//! | `AllBlocksCleared` | none |
//!
//! `medium` is `GPU`, `CPU_PINNED`, `DISK` or `EXTERNAL`. A publisher sends each batch as three
//! frames: its topic, the batch's sequence number (eight bytes, big-endian, from 0) and the
//! encoded batch; a replay socket answers a subscriber's starting sequence number with the
//! batches it still buffers from there, the same three frames after the subscriber's identity
//! and an empty frame, and ends with an empty topic, sequence number -1 and an empty payload.
//!
//! # Block hashes
//!
//! A page's block hash is SGLang's: the SHA-256 digest of the previous page's digest (none for
//! a request's first page) followed by the page's tokens as little-endian `u32`s, of which the
//! hash is the first eight bytes as a big-endian signed integer. A consumer that hashes tokens
//! itself (Dynamo's router, with the page size the events state) needs no mapping of UniServe's
//! page keys, and one that indexes SGLang's hashes (Mooncake's conductor) finds them as SGLang
//! would publish them. A page computed under an isolation key (a request's cache salt or
//! adapter, `crate::key::isolation`) has its hash and its parent's namespaced by the key
//! ([`Chain::namespaced`]), as SGLang namespaces the hashes of an adapter's pages: such a hash
//! is opaque, and no unisolated page's equals it.

use sha2::{Digest as _, Sha256};

/// Where a page's copy lies, as an event's `medium` names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Medium {
    /// A device's memory (HBM).
    Gpu,
    /// The node's host memory, which its Storage Server holds.
    CpuPinned,
    /// A local disk.
    Disk,
    /// A store outside the node: Mooncake Store.
    External,
}

impl Medium {
    /// The medium as an event states it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Gpu => "GPU",
            Self::CpuPinned => "CPU_PINNED",
            Self::Disk => "DISK",
            Self::External => "EXTERNAL",
        }
    }

    /// The medium an event's `medium` names.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "GPU" => Some(Self::Gpu),
            "CPU_PINNED" => Some(Self::CpuPinned),
            "DISK" => Some(Self::Disk),
            "EXTERNAL" => Some(Self::External),
            _ => None,
        }
    }
}

/// One KV event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// Pages entered `medium`: consecutive pages of one chain, the first a child of `parent`
    /// (`None` for a request's first page), `tokens` holding each page's `block_size` tokens in
    /// order.
    Stored {
        hashes: Vec<i64>,
        parent: Option<i64>,
        tokens: Vec<u32>,
        block_size: u32,
        medium: Medium,
    },
    /// Pages left `medium`.
    Removed { hashes: Vec<i64>, medium: Medium },
    /// The publisher holds no page any more: a subscriber forgets what it published before.
    Cleared,
}

impl Event {
    /// Appends `event` to the events of a step, merging it into the last one when SGLang's
    /// recorder would (`mem_cache/events.py`, `enqueue`): pages stored on one medium that
    /// continue the last event's chain join it, and pages removed from one medium join the last
    /// event's removal from it.
    pub fn push(events: &mut Vec<Event>, event: Event) {
        match (events.last_mut(), event) {
            (
                Some(Event::Stored {
                    hashes,
                    tokens,
                    block_size,
                    medium,
                    ..
                }),
                Event::Stored {
                    hashes: more,
                    parent: Some(parent),
                    tokens: more_tokens,
                    block_size: size,
                    medium: on,
                },
            ) if *medium == on && *block_size == size && hashes.last() == Some(&parent) => {
                hashes.extend(more);
                tokens.extend(more_tokens);
            }
            (
                Some(Event::Removed { hashes, medium }),
                Event::Removed {
                    hashes: more,
                    medium: on,
                },
            ) if *medium == on => hashes.extend(more),
            (_, event) => events.push(event),
        }
    }
}

/// A batch of events: what a publisher sends at once.
#[derive(Clone, Debug, PartialEq)]
pub struct Batch {
    /// The publisher's wall clock when it sent the batch, in seconds since the Unix epoch.
    pub time: f64,
    pub events: Vec<Event>,
    /// The publisher's data-parallel rank (`attn_dp_rank`).
    pub rank: Option<u32>,
}

impl Batch {
    /// The batch in msgspec's MessagePack layout of SGLang's `KVEventBatch` (module
    /// documentation): an array of three, each event a map whose `type` comes first and whose
    /// fields follow in their declaration order.
    pub fn encode(&self) -> Vec<u8> {
        use rmp::encode::{
            write_array_len, write_f64, write_map_len, write_nil, write_sint, write_str, write_uint,
        };

        // Writing into a vector cannot fail.
        let mut out = Vec::new();
        let _ = write_array_len(&mut out, 3);
        let _ = write_f64(&mut out, self.time);
        let _ = write_array_len(&mut out, len(self.events.len()));
        for event in &self.events {
            match event {
                Event::Stored {
                    hashes,
                    parent,
                    tokens,
                    block_size,
                    medium,
                } => {
                    let _ = write_map_len(&mut out, 7);
                    let _ = write_str(&mut out, "type");
                    let _ = write_str(&mut out, "BlockStored");
                    let _ = write_str(&mut out, "block_hashes");
                    let _ = write_array_len(&mut out, len(hashes.len()));
                    for hash in hashes {
                        let _ = write_sint(&mut out, *hash);
                    }
                    let _ = write_str(&mut out, "parent_block_hash");
                    match parent {
                        Some(parent) => {
                            let _ = write_sint(&mut out, *parent);
                        }
                        None => {
                            let _ = write_nil(&mut out);
                        }
                    }
                    let _ = write_str(&mut out, "token_ids");
                    let _ = write_array_len(&mut out, len(tokens.len()));
                    for token in tokens {
                        let _ = write_uint(&mut out, u64::from(*token));
                    }
                    let _ = write_str(&mut out, "block_size");
                    let _ = write_uint(&mut out, u64::from(*block_size));
                    // Declared without a default, so msgspec writes it even when unset.
                    let _ = write_str(&mut out, "lora_id");
                    let _ = write_nil(&mut out);
                    let _ = write_str(&mut out, "medium");
                    let _ = write_str(&mut out, medium.name());
                }
                Event::Removed { hashes, medium } => {
                    let _ = write_map_len(&mut out, 3);
                    let _ = write_str(&mut out, "type");
                    let _ = write_str(&mut out, "BlockRemoved");
                    let _ = write_str(&mut out, "block_hashes");
                    let _ = write_array_len(&mut out, len(hashes.len()));
                    for hash in hashes {
                        let _ = write_sint(&mut out, *hash);
                    }
                    let _ = write_str(&mut out, "medium");
                    let _ = write_str(&mut out, medium.name());
                }
                Event::Cleared => {
                    let _ = write_map_len(&mut out, 1);
                    let _ = write_str(&mut out, "type");
                    let _ = write_str(&mut out, "AllBlocksCleared");
                }
            }
        }
        match self.rank {
            Some(rank) => {
                let _ = write_uint(&mut out, u64::from(rank));
            }
            None => {
                let _ = write_nil(&mut out);
            }
        }
        out
    }

    /// The batch `payload` encodes, read as SGLang's consumers read it: the array of three, each
    /// event's map by its `type`, its unknown entries skipped.
    ///
    /// # Errors
    ///
    /// Fails for a payload that is no such batch: another shape, an event of an unknown `type`
    /// or medium, or a field of the wrong type.
    pub fn decode(payload: &[u8]) -> Result<Self, String> {
        let (time, events, rank): (f64, Vec<Wire>, Option<u32>) =
            rmp_serde::from_slice(payload).map_err(|error| error.to_string())?;
        let medium = |name: Option<String>| {
            let name = name.ok_or("an event states no medium")?;
            Medium::parse(&name).ok_or_else(|| format!("unknown medium `{name}`"))
        };
        let events = events
            .into_iter()
            .map(|event| {
                Ok(match event {
                    Wire::BlockStored {
                        block_hashes,
                        parent_block_hash,
                        token_ids,
                        block_size,
                        medium: on,
                    } => Event::Stored {
                        hashes: block_hashes,
                        parent: parent_block_hash,
                        tokens: token_ids,
                        block_size,
                        medium: medium(on)?,
                    },
                    Wire::BlockRemoved {
                        block_hashes,
                        medium: on,
                    } => Event::Removed {
                        hashes: block_hashes,
                        medium: medium(on)?,
                    },
                    Wire::AllBlocksCleared {} => Event::Cleared,
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(Self { time, events, rank })
    }
}

/// An event as the wire's map states it.
#[derive(serde::Deserialize)]
#[serde(tag = "type")]
enum Wire {
    BlockStored {
        block_hashes: Vec<i64>,
        parent_block_hash: Option<i64>,
        token_ids: Vec<u32>,
        block_size: u32,
        #[serde(default)]
        medium: Option<String>,
    },
    BlockRemoved {
        block_hashes: Vec<i64>,
        #[serde(default)]
        medium: Option<String>,
    },
    AllBlocksCleared {},
}

/// A MessagePack length: the events and pages of a batch number far below `u32::MAX`.
fn len(count: usize) -> u32 {
    u32::try_from(count).unwrap_or(u32::MAX)
}

/// The sequence number a replay socket ends its answer with.
pub const END: i64 = -1;

/// The three frames a publisher sends batch `sequence` as: its topic, the sequence number as
/// eight big-endian bytes, and the encoded batch.
pub fn frames(topic: &str, sequence: u64, payload: Vec<u8>) -> [Vec<u8>; 3] {
    [
        topic.as_bytes().to_vec(),
        sequence.to_be_bytes().to_vec(),
        payload,
    ]
}

/// A page's place in its chain of SGLang digests, from which its block hash follows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chain {
    /// The SHA-256 digest of the page's chain: the previous page's digest followed by its own
    /// tokens.
    pub digest: [u8; 32],
}

impl Chain {
    /// The chain of the page holding `tokens` after the page of `previous`, or of a request's
    /// first page when `previous` is `None`.
    pub fn next(previous: Option<&Chain>, tokens: &[u32]) -> Self {
        let mut digest = Sha256::new();
        if let Some(previous) = previous {
            digest.update(previous.digest);
        }
        for token in tokens {
            digest.update(token.to_le_bytes());
        }
        Self {
            digest: digest.finalize().into(),
        }
    }

    /// The chains of the complete pages of `block_size` tokens of `tokens`, in order.
    pub fn of(tokens: &[u32], block_size: usize) -> Vec<Self> {
        let mut chains: Vec<Self> = Vec::new();
        for page in tokens.chunks_exact(block_size.max(1)) {
            let next = Self::next(chains.last(), page);
            chains.push(next);
        }
        chains
    }

    /// The page's block hash: the digest's first eight bytes, as a big-endian signed integer.
    pub fn hash(&self) -> i64 {
        let mut leading = [0; 8];
        leading.copy_from_slice(&self.digest[..8]);
        i64::from_be_bytes(leading)
    }

    /// The page's block hash under the isolation key `isolation`: the first eight bytes of the
    /// SHA-256 digest of the key's namespace seed and the unisolated hash as eight big-endian
    /// bytes, as a big-endian signed integer (SGLang's `namespaced_block_hash`).
    pub fn namespaced(hash: i64, isolation: Option<u64>) -> i64 {
        let Some(isolation) = isolation else {
            return hash;
        };
        let mut seed = Sha256::new();
        seed.update(b"uniserve-kv-event-isolation");
        seed.update(isolation.to_le_bytes());
        let seed: [u8; 32] = seed.finalize().into();

        let mut digest = Sha256::new();
        digest.update(seed);
        digest.update(hash.to_be_bytes());
        let digest: [u8; 32] = digest.finalize().into();
        let mut leading = [0; 8];
        leading.copy_from_slice(&digest[..8]);
        i64::from_be_bytes(leading)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block hashes of SGLang's parity fixture (`sgl-router/tests/fixtures/
    /// kv_events_hash_parity.json`, which SGLang's own recorder validates), without a salt.
    #[test]
    fn block_hashes_are_sglangs() {
        let cases: [(&[u32], usize, &[i64]); 4] = [
            (&[1, 2, 3, 4], 4, &[-3488128144981237669]),
            (
                &[10, 20, 30, 40, 50, 60, 70, 80],
                2,
                &[
                    978178666101069530,
                    -895308556211281782,
                    -8033692805846017938,
                    835415944263129316,
                ],
            ),
            (
                &[7, 8, 9],
                1,
                &[
                    -1702009526849766914,
                    903318264012984157,
                    -8265893088400908305,
                ],
            ),
            (
                &[100, 200, 300, 400, 500, 600, 700],
                3,
                &[-7293070039731858224, -5869816562584529365],
            ),
        ];
        for (tokens, block_size, expected) in cases {
            let hashes: Vec<i64> = Chain::of(tokens, block_size)
                .iter()
                .map(Chain::hash)
                .collect();
            assert_eq!(hashes, expected, "{tokens:?} in pages of {block_size}");
        }
    }

    /// A batch decodes as it was encoded, and a step's stores of one chain and its removals
    /// from one medium merge as SGLang's recorder merges them.
    #[test]
    fn batches_round_trip_and_events_merge() {
        let chains = Chain::of(&[1, 2, 3, 4, 5, 6], 2);
        let mut events = Vec::new();
        Event::push(&mut events, Event::Cleared);
        for (index, chain) in chains.iter().enumerate() {
            let page = 2 * index as u32;
            Event::push(
                &mut events,
                Event::Stored {
                    hashes: vec![chain.hash()],
                    parent: index.checked_sub(1).map(|previous| chains[previous].hash()),
                    tokens: vec![page + 1, page + 2],
                    block_size: 2,
                    medium: Medium::Gpu,
                },
            );
        }
        Event::push(
            &mut events,
            Event::Removed {
                hashes: vec![chains[2].hash()],
                medium: Medium::Gpu,
            },
        );
        Event::push(
            &mut events,
            Event::Removed {
                hashes: vec![chains[1].hash()],
                medium: Medium::Gpu,
            },
        );
        assert_eq!(
            events,
            [
                Event::Cleared,
                Event::Stored {
                    hashes: chains.iter().map(Chain::hash).collect(),
                    parent: None,
                    tokens: vec![1, 2, 3, 4, 5, 6],
                    block_size: 2,
                    medium: Medium::Gpu,
                },
                Event::Removed {
                    hashes: vec![chains[2].hash(), chains[1].hash()],
                    medium: Medium::Gpu,
                },
            ]
        );

        let batch = Batch {
            time: 1.5,
            events,
            rank: Some(3),
        };
        assert_eq!(Batch::decode(&batch.encode()).unwrap(), batch);
    }
}
