//! Mooncake Store's object events, as its master publishes them, and the keys they state the
//! store holds.
//!
//! A master started with `--enable_kv_events --kv_events_bind_endpoint=tcp://HOST:PORT`
//! publishes on a ZMQ PUB socket every object it stores and removes
//! (`mooncake-store/src/kv_event/kv_event_publisher.cpp`), batched, each batch three frames: an
//! empty topic, the batch's sequence number (eight bytes, big-endian, from 1) and a MessagePack
//! array `[timestamp_ms, events, dp_rank]`. Each event is a map; the entries read here are
//! `event_type` (`stored` or `removed`; `type`, `BlockStored` or `BlockRemoved`, beside it in
//! the legacy layout), `object_key` and `medium` (the replica's medium, such as `cpu`). Every
//! other entry is skipped.
//!
//! The Mooncake connector names a page's object `uniserve_{hex(page key)}_{hex(pool
//! layout)}_{kv head offset}-{kv head count}` (`crate::mooncake`), one object per page, rank part
//! and representation. The store holds a page while it holds any object of its key: the
//! directory counts each object and medium apart, so an object's replica removed from one medium
//! leaves its key held while another stands. An object whose key is not one of UniServe's is
//! skipped.
//!
//! The subscription runs on a thread of its own ([`Subscription`]), which connects to the
//! master, connects again when the connection fails (the master restarts, or starts after the
//! server), and hands each batch to the server's loop through a channel whose wake the loop
//! waits on. A batch whose sequence number does not follow the last one is a gap: events were
//! lost (the master drops its oldest when its queue is full), or the master restarted and lost
//! its objects with it. What the directory knows of the store may then be wrong until the
//! events that follow correct it, which costs only reuse; a sequence number below the last one
//! is a master that restarted, and the directory forgets every key it learned from the one
//! before.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash as _, Hasher as _};
use std::io;
use std::os::fd::{AsFd as _, BorrowedFd, OwnedFd};
use std::sync::mpsc;
use std::time::Duration;

use ::storage::protocol::Key;
use serde::Deserialize;

/// How long the subscription waits before it connects again after its connection to the master
/// failed or could not be made.
const RECONNECT: Duration = Duration::from_secs(1);

/// The prefix of the object keys the Mooncake connector writes pages under.
const PREFIX: &str = "uniserve_";

/// One object event of the master's, of a page of UniServe's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ObjectEvent {
    /// Whether the object was stored; removed otherwise.
    pub(crate) stored: bool,
    /// The page's key.
    pub(crate) key: Key,
    /// The object and the medium of its replica, which tell apart the objects of one key.
    pub(crate) object: u64,
}

/// Which keys Mooncake Store holds, as the master's events stated them.
#[derive(Default)]
pub(crate) struct Store {
    /// Per key, the objects (and media) of it the store holds.
    objects: HashMap<Key, HashSet<u64>>,
    /// The sequence number of the last batch applied.
    last: Option<u64>,
}

impl Store {
    /// Applies batch `sequence`'s `events`. A sequence number below the last one is a master
    /// that restarted, whose store holds nothing of what its predecessor's did.
    pub(crate) fn apply(&mut self, sequence: u64, events: &[ObjectEvent]) {
        if let Some(last) = self.last {
            if sequence <= last {
                tracing::info!(
                    sequence,
                    last,
                    "Mooncake's master restarted: the store's keys are learned anew"
                );
                self.objects.clear();
            } else if sequence != last + 1 {
                tracing::debug!(
                    lost = sequence - last - 1,
                    "Mooncake's master skipped event batches"
                );
            }
        }
        self.last = Some(sequence);

        for event in events {
            if event.stored {
                self.objects
                    .entry(event.key)
                    .or_default()
                    .insert(event.object);
                continue;
            }
            if let Some(objects) = self.objects.get_mut(&event.key) {
                objects.remove(&event.object);
                if objects.is_empty() {
                    self.objects.remove(&event.key);
                }
            }
        }
    }

    /// The leading keys of `chain` the store holds.
    pub(crate) fn leading(&self, chain: &[Key]) -> u32 {
        let held = chain
            .iter()
            .take_while(|key| self.objects.contains_key(*key))
            .count();
        u32::try_from(held).unwrap_or(u32::MAX)
    }

    /// Every key the store holds, in no order.
    pub(crate) fn keys(&self) -> impl Iterator<Item = &Key> {
        self.objects.keys()
    }
}

/// An event as the master's map states it: the entries read here, every other skipped.
#[derive(Deserialize)]
struct Wire {
    #[serde(default)]
    event_type: Option<String>,
    #[serde(default, rename = "type")]
    legacy: Option<String>,
    #[serde(default)]
    object_key: Option<String>,
    #[serde(default)]
    medium: Option<String>,
}

/// The events of UniServe's pages the batch `payload` states: `[timestamp, events, dp_rank]`,
/// each event a map. An event of another kind, or of an object that is not a page of UniServe's,
/// is skipped.
///
/// # Errors
///
/// Fails for a payload that is no such batch.
pub(crate) fn decode(payload: &[u8]) -> Result<Vec<ObjectEvent>, String> {
    let (_, events, _): (serde::de::IgnoredAny, Vec<Wire>, serde::de::IgnoredAny) =
        rmp_serde::from_slice(payload).map_err(|error| error.to_string())?;
    let decoded = events
        .into_iter()
        .filter_map(|event| {
            let stored = match (event.event_type.as_deref(), event.legacy.as_deref()) {
                (Some("stored"), _) | (None, Some("BlockStored")) => true,
                (Some("removed"), _) | (None, Some("BlockRemoved")) => false,
                _ => return None,
            };
            let object_key = event.object_key?;
            let key = page_key(&object_key)?;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            object_key.hash(&mut hasher);
            event.medium.hash(&mut hasher);
            Some(ObjectEvent {
                stored,
                key,
                object: hasher.finish(),
            })
        })
        .collect();
    Ok(decoded)
}

/// The page key an object key of the Mooncake connector's names: the 64 hexadecimal digits after
/// [`PREFIX`]; `None` for any other key.
fn page_key(object: &str) -> Option<Key> {
    let digits = object.strip_prefix(PREFIX)?.get(..64)?;
    let mut key = [0_u8; 32];
    for (index, byte) in key.iter_mut().enumerate() {
        *byte = u8::from_str_radix(digits.get(index * 2..index * 2 + 2)?, 16).ok()?;
    }
    Some(Key(key))
}

/// A subscription to the master's events on its own thread.
pub(crate) struct Subscription {
    batches: mpsc::Receiver<(u64, Vec<ObjectEvent>)>,
    /// The read end of the pipe the thread writes a byte into after each batch.
    wake: OwnedFd,
}

impl Subscription {
    /// Subscribes to the publisher at `endpoint` (`host:port`) from a thread that runs for the
    /// server's life.
    ///
    /// # Errors
    ///
    /// Fails when the pipe or the thread cannot be made.
    pub(crate) fn start(endpoint: String) -> io::Result<Self> {
        let (reader, writer) = rustix::pipe::pipe_with(
            rustix::pipe::PipeFlags::CLOEXEC | rustix::pipe::PipeFlags::NONBLOCK,
        )?;
        let (sender, batches) = mpsc::channel();
        std::thread::Builder::new()
            .name("uniserve-mooncake-events".into())
            .spawn(move || subscribe(&endpoint, &sender, &writer))?;
        Ok(Self {
            batches,
            wake: reader,
        })
    }

    /// The descriptor that is readable once a batch arrived.
    pub(crate) fn wake(&self) -> BorrowedFd<'_> {
        self.wake.as_fd()
    }

    /// Every batch that arrived, in order, with its sequence number.
    pub(crate) fn drain(&self) -> Vec<(u64, Vec<ObjectEvent>)> {
        let mut drained = [0_u8; 64];
        while matches!(rustix::io::read(&self.wake, &mut drained), Ok(read) if read > 0) {}
        self.batches.try_iter().collect()
    }
}

/// The subscription's thread: connects to `endpoint` and sends each batch until the server's
/// end drops the receiver.
fn subscribe(endpoint: &str, sender: &mpsc::Sender<(u64, Vec<ObjectEvent>)>, wake: &OwnedFd) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            tracing::warn!(%error, "the subscription to Mooncake's events has no runtime");
            return;
        }
    };
    runtime.block_on(async {
        use zeromq::{Socket as _, SocketRecv as _};

        let address = format!("tcp://{endpoint}");
        loop {
            let mut socket = zeromq::SubSocket::new();
            let connected = async {
                socket.connect(&address).await?;
                socket.subscribe("").await
            };
            if let Err(error) = connected.await {
                tracing::debug!(%error, endpoint, "Mooncake's master publishes no events yet");
                tokio::time::sleep(RECONNECT).await;
                continue;
            }
            tracing::info!(endpoint, "subscribed to Mooncake's object events");

            loop {
                let message = match socket.recv().await {
                    Ok(message) => message,
                    Err(error) => {
                        tracing::warn!(%error, endpoint, "the subscription to Mooncake's events failed");
                        break;
                    }
                };
                let frames: Vec<&[u8]> = message.iter().map(|frame| frame.as_ref()).collect();
                let [_, sequence, payload] = frames[..] else {
                    tracing::debug!(frames = frames.len(), "an event message of another shape");
                    continue;
                };
                let Ok(sequence) = <[u8; 8]>::try_from(sequence).map(u64::from_be_bytes) else {
                    tracing::debug!("an event batch without its sequence number");
                    continue;
                };
                let events = match decode(payload) {
                    Ok(events) => events,
                    Err(error) => {
                        tracing::debug!(%error, "an event batch that does not decode");
                        continue;
                    }
                };
                if sender.send((sequence, events)).is_err() {
                    return;
                }
                let _ = rustix::io::write(wake, &[1]);
            }
            tokio::time::sleep(RECONNECT).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(key: u8, layout: u8, heads: &str) -> String {
        format!(
            "{PREFIX}{}_{}_{heads}",
            Key([key; 32]).hex(),
            Key([layout; 32]).hex()
        )
    }

    /// A batch as the master packs it (`kv_event_publisher.cpp`, legacy fields and the object
    /// key emitted): a stored and a removed event of a page's object, and an object of another
    /// client's, which is skipped.
    fn batch(events: &[(&str, &str, &str)]) -> Vec<u8> {
        use rmp::encode::{write_array_len, write_map_len, write_nil, write_str, write_uint};

        let mut out = Vec::new();
        write_array_len(&mut out, 3).unwrap();
        write_uint(&mut out, 1_795_012_729_927).unwrap();
        write_array_len(&mut out, events.len() as u32).unwrap();
        for (kind, key, medium) in events {
            let legacy = if *kind == "stored" {
                "BlockStored"
            } else {
                "BlockRemoved"
            };
            write_map_len(&mut out, 6).unwrap();
            write_str(&mut out, "event_id").unwrap();
            write_uint(&mut out, 1).unwrap();
            write_str(&mut out, "event_type").unwrap();
            write_str(&mut out, kind).unwrap();
            write_str(&mut out, "type").unwrap();
            write_str(&mut out, legacy).unwrap();
            write_str(&mut out, "medium").unwrap();
            write_str(&mut out, medium).unwrap();
            write_str(&mut out, "object_key").unwrap();
            write_str(&mut out, key).unwrap();
            write_str(&mut out, "parent_hash").unwrap();
            write_nil(&mut out).unwrap();
        }
        write_uint(&mut out, 0).unwrap();
        out
    }

    /// The store holds a page while any object of its key stands, on any medium; another
    /// client's object counts for nothing.
    #[test]
    fn the_store_holds_a_page_while_any_object_of_its_key_stands() {
        let (part0, part1) = (object(1, 9, "0-2"), object(1, 9, "2-2"));
        let second = object(2, 9, "0-2");
        let mut store = Store::default();
        let stored = decode(&batch(&[
            ("stored", &part0, "cpu"),
            ("stored", &part1, "cpu"),
            ("stored", &second, "cpu"),
            ("stored", "0x1a2b", "cpu"),
        ]))
        .unwrap();
        assert_eq!(stored.len(), 3);
        store.apply(1, &stored);
        let chain = [Key([1; 32]), Key([2; 32]), Key([3; 32])];
        assert_eq!(store.leading(&chain), 2);

        store.apply(2, &decode(&batch(&[("removed", &part0, "cpu")])).unwrap());
        assert_eq!(store.leading(&chain), 2);
        store.apply(3, &decode(&batch(&[("removed", &part1, "cpu")])).unwrap());
        assert_eq!(store.leading(&chain), 0);
        assert_eq!(store.keys().count(), 1);
    }

    /// A batch numbered below the last is a master that restarted: what the store held is
    /// forgotten.
    #[test]
    fn a_restarted_master_starts_the_store_anew() {
        let mut store = Store::default();
        let first = object(1, 9, "0-4");
        store.apply(5, &decode(&batch(&[("stored", &first, "cpu")])).unwrap());
        store.apply(1, &[]);
        assert_eq!(store.leading(&[Key([1; 32])]), 0);
    }
}
