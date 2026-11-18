//! The KV events a scheduler publishes (`kv_events_config`): which pages its cache holds on
//! which medium, in SGLang's ZMQ format (`storage::events`), for KV-aware routers outside the
//! deployment.
//!
//! The scheduler states what it knows of the cache, which is what the events state: the pages
//! of the full-attention class its queue pair commits, as they enter and leave the device, the
//! host tier and the remote store (`Cached::tiers`). Each page is offered with its tokens when
//! its commit is handed to the executor ([`Events::offer`]), which is the one moment the
//! scheduler holds both its key and its tokens; it is stored on a medium once the scheduler
//! learns a copy of it there ([`Events::changed`]), and forgotten once its commit was answered
//! without a copy ([`Events::settled`]) or its last copy left. A step's events leave as one
//! batch at the step's end ([`Events::flush`]), as SGLang's scheduler publishes once per
//! iteration.
//!
//! A page's events name it by its SGLang block hash ([`Chain`]), computed from its tokens and
//! its predecessor's, so a consumer that hashes the tokens itself finds the same pages. Two keys
//! of one hash (the same tokens under two page seeds) are one block to a consumer: the hash is
//! stored once the first of them is and removed once the last of them is.
//!
//! The publisher runs on a thread of its own with a PUB socket at the scheduler's endpoint and,
//! when configured, a ROUTER socket answering replays from the batches it buffers. A
//! publisher that cannot bind is logged, and the scheduler serves on without one until its
//! settings change.

use std::collections::{HashMap, VecDeque};
use std::sync::mpsc as sync_mpsc;
use std::thread::JoinHandle;

use proto::config::KvEventsConfig;
use storage::Key;
use storage::events::{Batch, Chain, END, Event, Medium, frames};
use tokio::sync::mpsc;
use tracing::warn;
use zeromq::{Socket as _, SocketRecv as _, SocketSend as _, ZmqMessage};

use super::cache::{DEVICE, HOST, REMOTE};

/// The media of the tier bits of a cached block (`Cached::tiers`), each with its bit.
const MEDIA: [(u8, Medium); 3] = [
    (DEVICE, Medium::Gpu),
    (HOST, Medium::CpuPinned),
    (REMOTE, Medium::External),
];

/// A page whose events the scheduler publishes.
struct Page {
    /// Its place in its chain of SGLang digests.
    chain: Chain,
    /// Its block hash and its predecessor's, namespaced by the request's isolation key.
    hash: i64,
    parent: Option<i64>,
    tokens: Box<[u32]>,
    /// The tier bits of the copies its events have stored.
    tiers: u8,
}

/// A scheduler's KV events: the pages it publishes, the events of its step, and its publisher.
pub(super) struct Events {
    /// The scheduler's ordinal across the deployment: the offset of its endpoints' ports and its
    /// batches' rank.
    ordinal: u32,
    /// The settings the publisher was last configured with; `None` while unconfigured.
    settings: Option<KvEventsConfig>,
    publisher: Option<Publisher>,
    pages: HashMap<Key, Page>,
    /// How many published pages hold each block hash on each medium.
    holders: HashMap<(i64, Medium), u32>,
    /// The step's events, in order.
    events: Vec<Event>,
}

impl Default for Events {
    /// The events of scheduler 0, which publish nothing.
    fn default() -> Self {
        Self::new(0)
    }
}

impl Events {
    /// The events of the deployment's scheduler `ordinal`, publishing nothing until configured.
    pub(super) fn new(ordinal: u32) -> Self {
        Self {
            ordinal,
            settings: None,
            publisher: None,
            pages: HashMap::new(),
            holders: HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Numbers the scheduler `ordinal` across the deployment (`EngineArgs::replica` plus its
    /// index in its engine); a publisher already bound keeps its endpoint until the next change
    /// of the settings.
    pub(super) fn number(&mut self, ordinal: u32) {
        self.ordinal = ordinal;
    }

    /// Whether the scheduler publishes: only then are pages offered and tracked.
    pub(super) fn publishing(&self) -> bool {
        self.publisher.is_some()
    }

    /// Takes the step's settings: a publisher starts, moves or stops as they changed. A new
    /// publisher first states that no page is held, then every page the scheduler tracks on
    /// every medium it holds it on.
    pub(super) fn configure(&mut self, settings: &KvEventsConfig) {
        if self.settings.as_ref() == Some(settings) {
            return;
        }
        self.settings = Some(settings.clone());
        self.publisher = None;
        self.events.clear();

        let Some(endpoint) = &settings.endpoint else {
            self.pages.clear();
            self.holders.clear();
            return;
        };
        match Publisher::start(settings, endpoint, self.ordinal) {
            Ok(publisher) => self.publisher = Some(publisher),
            Err(error) => {
                warn!(
                    scheduler = self.ordinal,
                    endpoint, "the KV event publisher does not publish: {error}"
                );
                self.pages.clear();
                self.holders.clear();
                return;
            }
        }

        self.events.push(Event::Cleared);
        let mut pages: Vec<&Page> = self.pages.values().collect();
        // A page's predecessor first, so a subscriber links every chain.
        pages.sort_by_key(|page| page.parent.is_some());
        for page in pages {
            for &(bit, medium) in &MEDIA {
                if page.tiers & bit != 0 {
                    Event::push(&mut self.events, stored(page, medium));
                }
            }
        }
    }

    /// Offers the page of `key`, the block at `position` of a request's `tokens` in pages of
    /// `block` tokens, whose commit is handed to the executor: `previous` is the key of the page
    /// before it, `isolation` the request's isolation key. It is tracked from now on, and stored
    /// once a copy of it is learned.
    pub(super) fn offer(
        &mut self,
        key: Key,
        previous: Option<&Key>,
        position: usize,
        block: usize,
        tokens: &[u32],
        isolation: Option<u64>,
    ) {
        if !self.publishing() || self.pages.contains_key(&key) {
            return;
        }
        let start = position * block;
        let Some(own) = tokens.get(start..start + block) else {
            return;
        };

        // The predecessor's chain, which a page committed before it states; one this scheduler
        // never committed is chained again from the request's tokens.
        let before = match previous.and_then(|key| self.pages.get(key)) {
            Some(page) => Some(page.chain),
            None if position == 0 => None,
            None => Chain::of(&tokens[..start], block).last().copied(),
        };
        let chain = Chain::next(before.as_ref(), own);
        self.pages.insert(
            key,
            Page {
                chain,
                hash: Chain::namespaced(chain.hash(), isolation),
                parent: before.map(|before| Chain::namespaced(before.hash(), isolation)),
                tokens: own.into(),
                tiers: 0,
            },
        );
    }

    /// The cached block of `key` is on the tiers `after` now (`Cached::tiers`, none once it left
    /// the cache): a tracked page is stored on the media it entered and removed from those it
    /// left, and forgotten once it is on none.
    pub(super) fn changed(&mut self, key: &Key, after: u8) {
        let Some(page) = self.pages.get_mut(key) else {
            return;
        };
        let (entered, left) = (after & !page.tiers, page.tiers & !after);
        page.tiers = after;
        for &(bit, medium) in &MEDIA {
            if entered & bit != 0 {
                let holders = self.holders.entry((page.hash, medium)).or_default();
                *holders += 1;
                if *holders == 1 {
                    Event::push(&mut self.events, stored(page, medium));
                }
            }
            if left & bit != 0
                && let Some(holders) = self.holders.get_mut(&(page.hash, medium))
            {
                *holders -= 1;
                if *holders == 0 {
                    self.holders.remove(&(page.hash, medium));
                    Event::push(
                        &mut self.events,
                        Event::Removed {
                            hashes: vec![page.hash],
                            medium,
                        },
                    );
                }
            }
        }
        if after == 0 {
            self.pages.remove(key);
        }
    }

    /// The commit of `key` was answered: a page offered for it that no copy holds is forgotten.
    pub(super) fn settled(&mut self, key: &Key) {
        if self.pages.get(key).is_some_and(|page| page.tiers == 0) {
            self.pages.remove(key);
        }
    }

    /// Publishes the step's events as one batch, if it had any.
    pub(super) fn flush(&mut self) {
        if self.events.is_empty() {
            return;
        }
        let events = std::mem::take(&mut self.events);
        let Some(publisher) = &self.publisher else {
            return;
        };
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0.0, |since| since.as_secs_f64());
        let batch = Batch {
            time,
            events,
            rank: Some(self.ordinal),
        };
        publisher.send(batch.encode());
    }
}

/// The event that stores `page` on `medium`.
fn stored(page: &Page, medium: Medium) -> Event {
    Event::Stored {
        hashes: vec![page.hash],
        parent: page.parent,
        tokens: page.tokens.to_vec(),
        block_size: u32::try_from(page.tokens.len()).unwrap_or(u32::MAX),
        medium,
    }
}

/// A publisher's thread, which sends every batch it is given, numbered in order from 0.
struct Publisher {
    batches: Option<mpsc::UnboundedSender<Vec<u8>>>,
    thread: Option<JoinHandle<()>>,
}

impl Publisher {
    /// Starts the publisher of scheduler `ordinal` at `endpoint` and, when `settings` states
    /// one, its replay socket, both offset by `ordinal`; returns once both are bound.
    fn start(settings: &KvEventsConfig, endpoint: &str, ordinal: u32) -> Result<Self, String> {
        let bound = |endpoint: &str| {
            KvEventsConfig::offset(endpoint, ordinal).ok_or_else(|| {
                format!("`{endpoint}` offset by {ordinal} is no `tcp://<host>:<port>` endpoint")
            })
        };
        let endpoint = bound(endpoint)?;
        let replay = settings.replay_endpoint.as_deref().map(bound).transpose()?;
        let topic = settings.topic.clone();
        let kept = settings.buffer_steps.get();

        let (batches, received) = mpsc::unbounded_channel();
        let (ready, readiness) = sync_mpsc::channel();
        let thread = std::thread::Builder::new()
            .name(format!("kv-events-{ordinal}"))
            .spawn(move || {
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        let _ = ready.send(Err(error.to_string()));
                        return;
                    }
                };
                runtime.block_on(publish(endpoint, replay, topic, kept, received, ready));
            })
            .map_err(|error| error.to_string())?;
        match readiness.recv() {
            Ok(Ok(())) => Ok(Self {
                batches: Some(batches),
                thread: Some(thread),
            }),
            Ok(Err(error)) => {
                let _ = thread.join();
                Err(error)
            }
            Err(_) => {
                let _ = thread.join();
                Err("the publisher's thread ended before it bound".into())
            }
        }
    }

    fn send(&self, payload: Vec<u8>) {
        if let Some(batches) = &self.batches {
            let _ = batches.send(payload);
        }
    }
}

impl Drop for Publisher {
    /// Ends the thread once it sent every batch it was given.
    fn drop(&mut self) {
        self.batches = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// The publisher's loop: binds its sockets, reports whether they bound, then sends each batch
/// as SGLang's publisher does ([`frames`]) and answers replays, until its sender is dropped.
async fn publish(
    endpoint: String,
    replay: Option<String>,
    topic: String,
    kept: usize,
    mut batches: mpsc::UnboundedReceiver<Vec<u8>>,
    ready: sync_mpsc::Sender<Result<(), String>>,
) {
    let mut socket = zeromq::PubSocket::new();
    if let Err(error) = socket.bind(&endpoint).await {
        let _ = ready.send(Err(format!("binding {endpoint}: {error}")));
        return;
    }
    let mut router = match replay {
        Some(replay) => {
            let mut router = zeromq::RouterSocket::new();
            if let Err(error) = router.bind(&replay).await {
                let _ = ready.send(Err(format!("binding {replay}: {error}")));
                return;
            }
            Some(router)
        }
        None => None,
    };
    let _ = ready.send(Ok(()));

    // The batches a replay answers from, by sequence number.
    let mut buffer: VecDeque<(u64, Vec<u8>)> = VecDeque::new();
    let mut sequence = 0u64;
    loop {
        let asked = async {
            match router.as_mut() {
                Some(router) => router.recv().await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            batch = batches.recv() => {
                let Some(payload) = batch else {
                    break;
                };
                if let Err(error) = socket.send(message(frames(&topic, sequence, payload.clone()))).await {
                    warn!(endpoint, "a KV event batch was not sent: {error}");
                }
                buffer.push_back((sequence, payload));
                if buffer.len() > kept {
                    buffer.pop_front();
                }
                sequence += 1;
            }
            request = asked => {
                let (Ok(request), Some(router)) = (request, router.as_mut()) else {
                    continue;
                };
                answer(router, &request, &buffer, &topic).await;
            }
        }
    }
}

/// Answers a replay request, `[identity, "", start]` with `start` eight big-endian bytes, with
/// every buffered batch from `start` on and the end marker, each after the identity and an
/// empty frame; a request of another shape is skipped.
async fn answer(
    router: &mut zeromq::RouterSocket,
    request: &ZmqMessage,
    buffer: &VecDeque<(u64, Vec<u8>)>,
    topic: &str,
) {
    let frames_of: Vec<&[u8]> = request.iter().map(|frame| frame.as_ref()).collect();
    let [identity, _, start] = frames_of[..] else {
        return;
    };
    let Ok(start) = <[u8; 8]>::try_from(start).map(u64::from_be_bytes) else {
        return;
    };
    let reply = |rest: [Vec<u8>; 3]| {
        let mut reply = message(rest);
        reply.push_front(Vec::new().into());
        reply.push_front(identity.to_vec().into());
        reply
    };
    for (sequence, payload) in buffer.iter().filter(|(sequence, _)| *sequence >= start) {
        let _ = router
            .send(reply(frames(topic, *sequence, payload.clone())))
            .await;
    }
    let _ = router
        .send(reply([Vec::new(), END.to_be_bytes().to_vec(), Vec::new()]))
        .await;
}

/// A multipart message of `parts`, in order.
fn message(parts: [Vec<u8>; 3]) -> ZmqMessage {
    let [first, rest @ ..] = parts;
    let mut message = ZmqMessage::from(first);
    for part in rest {
        message.push_back(part.into());
    }
    message
}
