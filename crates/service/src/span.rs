//! Spans: the stages a request passes through in every process, recorded as fixed-size records
//! in rings that each process keeps of its own most recent work.
//!
//! **Records.** A stage writes two records of 16 bytes, one as it begins and one as it ends:
//! the host's `CLOCK_REALTIME` in nanoseconds, and one word holding the owner's id (a request's
//! trace id or a batch id, 48 bits), the point (14 bits: a [`Stage`] or, from [`KERNEL`] on, a
//! timed kernel) and the record's mark (2 bits: begin, end, a request's membership of a batch,
//! or a byte count). Writing one is a clock read and three stores into the writer's own ring:
//! no lock is taken and nothing is allocated.
//!
//! **Rings.** Each thread writes the ring it claimed at its first record ([`begin`], [`end`],
//! [`member`], [`bytes`]), which it releases when it exits for the next thread to claim, so a
//! process holds as many rings as it ran threads at once. A device's reader writes a ring of its
//! own ([`Ring::new`]). A ring keeps its last [`Ring::capacity`] records and overwrites the
//! oldest; a reader copies every ring without stopping its writer ([`records`]) and drops the
//! records its writer overwrote meanwhile. Each ring carries the place its records come from
//! ([`Label`]: a worker's group and rank, a scheduler's index).
//!
//! **Spans.** A process turns its records into [`Span`]s itself ([`spans`]), naming each stage
//! and kernel, so a reader of another build reads its spans by name. A begin pairs with the
//! next end of the same owner and point in time order, whichever thread wrote either, so a
//! stage may begin and end on different threads of an asynchronous runtime; a begin whose end
//! is not written yet, and an end whose begin was overwritten, make no span. A membership record
//! names a request and a batch and becomes a span of the request over the scheduler's
//! formation of the batch ([`Stage::Batch`]); a byte count belongs to the next end its thread
//! writes of the same owner and point.
//!
//! **Pages.** A reader asks for the spans that ended since a time and receives them in pages
//! ([`page`]): the spans of the earliest ends first, each page's encoding within the bytes a
//! message of its interface takes, and the end the next page starts from. A process reads its
//! rings again for each page, so a page holds what its rings keep then.

use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::lock;

/// The records a thread's ring holds: 1 MiB of records, the last tens of thousands of stages a
/// thread passed through.
pub const THREAD_RECORDS: usize = 1 << 16;

/// The first point of a timed kernel: points below it are stages ([`Stage`]), and point
/// `KERNEL + site` is the kernel at a site the process names ([`spans`]).
pub const KERNEL: u16 = 1024;

/// The points a record can name: 14 bits.
pub const POINTS: u16 = 1 << 14;

/// The ids a record can name: 48 bits. A trace id or a batch id at or past it is recorded modulo
/// it.
pub const IDS: u64 = 1 << 48;

/// A record's mark: what the record states of its owner at its point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Mark {
    /// The stage began at the record's time.
    Begin = 0,
    /// The stage ended at the record's time.
    End = 1,
    /// The request the record names is a member of the batch its time word holds.
    Member = 2,
    /// The stage's next end moved the bytes its time word holds.
    Bytes = 3,
}

/// The process a stage runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Process {
    /// A node's daemon: the leader's request input and output.
    Daemon,
    /// A main process's scheduler, and its control thread's hand-over of a request to it.
    Scheduler,
    /// A main process's worker of one device, and that device's timed kernels.
    Worker,
    /// A node's Storage Server.
    Storage,
}

impl Process {
    /// The process's name in a span's `process` label.
    pub fn name(self) -> &'static str {
        match self {
            Self::Daemon => "daemon",
            Self::Scheduler => "scheduler",
            Self::Worker => "worker",
            Self::Storage => "storage",
        }
    }
}

/// What a stage's id names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Owner {
    /// A request, by its trace id.
    Request,
    /// A batch, by its id.
    Batch,
}

/// Declares the stages: each with its point, its name, its process and what its id names.
macro_rules! stages {
    ($($(#[$doc:meta])* $stage:ident = $point:literal, $name:literal, $process:ident, $owner:ident;)*) => {
        /// A stage a request or a batch passes through, as the process that runs it records it.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[repr(u16)]
        pub enum Stage {
            $($(#[$doc])* $stage = $point,)*
        }

        impl Stage {
            /// Every stage, in point order.
            pub const ALL: &'static [Self] = &[$(Self::$stage),*];

            /// The stage's name in a span's `stage` field.
            pub fn name(self) -> &'static str {
                match self {
                    $(Self::$stage => $name,)*
                }
            }

            /// The process the stage runs in.
            pub fn process(self) -> Process {
                match self {
                    $(Self::$stage => Process::$process,)*
                }
            }

            /// What the stage's id names.
            pub fn owner(self) -> Owner {
                match self {
                    $(Self::$stage => Owner::$owner,)*
                }
            }

            /// The stage at `point`, if one is.
            pub fn of(point: u16) -> Option<Self> {
                match point {
                    $($point => Some(Self::$stage),)*
                    _ => None,
                }
            }
        }
    };
}

stages! {
    /// The daemon read the request's head and body.
    Receive = 1, "receive", Daemon, Request;
    /// The daemon parsed and validated the request's body.
    Parse = 2, "parse", Daemon, Request;
    /// The daemon rendered the chat template.
    Template = 3, "template", Daemon, Request;
    /// The daemon tokenized the prompt.
    Tokenize = 4, "tokenize", Daemon, Request;
    /// The daemon downloaded and decoded the request's media.
    Fetch = 5, "fetch", Daemon, Request;
    /// The daemon preprocessed the request's media for the model's encoders.
    Preprocess = 6, "preprocess", Daemon, Request;
    /// The daemon chose the replica and scheduler of the request.
    Route = 7, "route", Daemon, Request;
    /// The daemon handed the request to its engine, until the engine accepted it.
    Submit = 8, "submit", Daemon, Request;
    /// The daemon detokenized one event's tokens.
    Detokenize = 9, "detokenize", Daemon, Request;
    /// The daemon assembled one piece of the response: a streamed chunk or the document.
    Assemble = 10, "assemble", Daemon, Request;
    /// The daemon sent one streamed chunk, or the response document, to the client.
    Send = 11, "send", Daemon, Request;
    /// The main process received the request, until its scheduler took it.
    Enqueue = 32, "enqueue", Scheduler, Request;
    /// The scheduler admitted the request: from the waiting queue to a slot and its cache.
    Admit = 33, "admit", Scheduler, Request;
    /// One call of the scheduling program for the step that forms the batch.
    Schedule = 34, "schedule", Scheduler, Batch;
    /// The scheduler formed the batch; each member's [`Mark::Member`] record joins it.
    Batch = 35, "batch", Scheduler, Batch;
    /// The worker copied the batch's inputs from the host to the device.
    CopyIn = 64, "copy_in", Worker, Batch;
    /// The worker launched the batch's graph or its eager forward.
    Launch = 65, "launch", Worker, Batch;
    /// The worker copied the batch's outputs from the device to the host.
    CopyOut = 66, "copy_out", Worker, Batch;
    /// The worker's host work on the batch's outputs: sampled tokens, stop conditions, logprobs.
    Postprocess = 67, "postprocess", Worker, Batch;
    /// The batch's step on the device, between its step markers.
    Step = 68, "step", Worker, Batch;
    /// A prefill-to-decode handoff: the buffer pool copied the request's blocks into its decode
    /// group's memory, with the bytes it copied.
    Handoff = 96, "handoff", Storage, Request;
}

impl Stage {
    /// The stage's point in a record.
    pub fn point(self) -> u16 {
        self as u16
    }
}

/// The place a ring's records come from, beyond its process.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Label {
    /// None beyond the process.
    #[default]
    None,
    /// A worker: its worker group, by the engine's number of it, and its rank in the group.
    Worker { group: u8, rank: u32 },
    /// A scheduler's index in its main process.
    Scheduler(u32),
}

impl Label {
    fn word(self) -> u64 {
        match self {
            Self::None => 0,
            Self::Worker { group, rank } => (1 << 40) | (u64::from(group) << 32) | u64::from(rank),
            Self::Scheduler(scheduler) => (2 << 40) | u64::from(scheduler),
        }
    }

    fn of(word: u64) -> Self {
        let value = word as u32;
        match word >> 40 {
            1 => Self::Worker {
                group: (word >> 32) as u8,
                rank: value,
            },
            2 => Self::Scheduler(value),
            _ => Self::None,
        }
    }
}

/// One record as a reader copied it: its time word and its tagged word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record {
    /// Nanoseconds since the Unix epoch; for a [`Mark::Member`] the batch id and for a
    /// [`Mark::Bytes`] the byte count.
    pub time: u64,
    pub id: u64,
    pub point: u16,
    pub mark: Mark,
}

impl Record {
    /// The record's tagged word: `id << 16 | point << 2 | mark`.
    pub fn word(&self) -> u64 {
        ((self.id % IDS) << 16) | (u64::from(self.point % POINTS) << 2) | self.mark as u64
    }

    /// The record of `time` and a tagged `word`.
    pub fn of(time: u64, word: u64) -> Self {
        let mark = match word & 3 {
            0 => Mark::Begin,
            1 => Mark::End,
            2 => Mark::Member,
            _ => Mark::Bytes,
        };
        Self {
            time,
            id: word >> 16,
            point: ((word >> 2) & u64::from(POINTS - 1)) as u16,
            mark,
        }
    }
}

/// A ring of records with one writer at a time and any number of readers.
///
/// Slot `i mod capacity` holds record `i` as two words; the head counts the records written and
/// is published after the slot's words, so a reader that acquires head `h` reads every record
/// below it, and one that reads the head again after its copy knows which of the copied slots
/// the writer may have overwritten meanwhile.
pub struct Ring {
    /// `2 * capacity` words: each record's time, then its tagged word.
    slots: Box<[AtomicU64]>,
    head: AtomicU64,
    label: AtomicU64,
    /// Whether a thread writes the ring; a thread ring that is free waits for the next thread.
    claimed: AtomicBool,
}

/// Every ring of the process, which [`records`] reads.
static RINGS: Mutex<Vec<&'static Ring>> = Mutex::new(Vec::new());

impl Ring {
    /// A ring of `capacity` records (at least one) of the place `label`, registered with the
    /// process's rings for as long as the process lives. Its storage is zeroed by the allocator,
    /// so the pages of a ring that has not wrapped yet are not taken.
    pub fn new(capacity: usize, label: Label) -> &'static Self {
        let ring: &'static Self = Box::leak(Box::new(Self {
            slots: zeroed(2 * capacity.max(1)),
            head: AtomicU64::new(0),
            label: AtomicU64::new(label.word()),
            claimed: AtomicBool::new(true),
        }));
        lock(&RINGS).push(ring);
        ring
    }

    /// The records the ring keeps.
    pub fn capacity(&self) -> usize {
        self.slots.len() / 2
    }

    /// Names the place the ring's records come from.
    pub fn label(&self, label: Label) {
        self.label.store(label.word(), Ordering::Relaxed);
    }

    /// Writes `record`. Only the ring's one writer calls this.
    pub fn write(&self, record: &Record) {
        let head = self.head.load(Ordering::Relaxed);
        let slot = 2 * (head % self.capacity() as u64) as usize;

        self.slots[slot].store(record.time, Ordering::Relaxed);
        self.slots[slot + 1].store(record.word(), Ordering::Relaxed);
        // Publishes the slot's words with the record.
        self.head.store(head + 1, Ordering::Release);
    }

    /// The records the ring holds, oldest first, without stopping its writer: those its writer
    /// may have overwritten while they were copied are left out, and so is the oldest slot, which
    /// a write in progress may be overwriting.
    pub fn read(&self) -> Vec<Record> {
        let capacity = self.capacity() as u64;
        let written = self.head.load(Ordering::Acquire);
        let first = written.saturating_sub(capacity);

        let mut copied = Vec::with_capacity((written - first) as usize);
        for index in first..written {
            let slot = 2 * (index % capacity) as usize;
            let time = self.slots[slot].load(Ordering::Relaxed);
            let word = self.slots[slot + 1].load(Ordering::Relaxed);
            copied.push((index, time, word));
        }

        // The writer at head `now` may be writing record `now`, which overwrites record
        // `now - capacity`; every record below that one was overwritten before it.
        std::sync::atomic::fence(Ordering::Acquire);
        let now = self.head.load(Ordering::Relaxed);
        let kept = (now + 1).saturating_sub(capacity);
        copied
            .into_iter()
            .filter(|&(index, _, _)| index >= kept)
            .map(|(_, time, word)| Record::of(time, word))
            .collect()
    }
}

/// `words` atomic words, zeroed.
fn zeroed(words: usize) -> Box<[AtomicU64]> {
    let zeroed = Box::<[AtomicU64]>::new_zeroed_slice(words);
    // SAFETY: an `AtomicU64` of zero bits is a valid zero.
    unsafe { zeroed.assume_init() }
}

/// The ring a thread writes, claimed at its first record and freed as the thread exits.
struct Claim(Cell<Option<&'static Ring>>);

impl Claim {
    fn ring(&self) -> &'static Ring {
        if let Some(ring) = self.0.get() {
            return ring;
        }
        let free = lock(&RINGS)
            .iter()
            .copied()
            .find(|ring| ring.capacity() == THREAD_RECORDS && claim(ring));
        let ring = match free {
            Some(ring) => {
                ring.label(Label::None);
                ring
            }
            None => Ring::new(THREAD_RECORDS, Label::None),
        };
        self.0.set(Some(ring));
        ring
    }
}

/// Claims `ring` if no thread writes it.
fn claim(ring: &Ring) -> bool {
    ring.claimed
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_ok()
}

impl Drop for Claim {
    fn drop(&mut self) {
        if let Some(ring) = self.0.get() {
            ring.claimed.store(false, Ordering::Release);
        }
    }
}

thread_local! {
    static CLAIM: Claim = const { Claim(Cell::new(None)) };
}

/// The host's `CLOCK_REALTIME`, in nanoseconds since the Unix epoch.
pub fn now() -> u64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `time` is a valid `timespec` the call writes.
    unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut time) };
    (time.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(time.tv_nsec as u64)
}

/// Writes `record` into the calling thread's ring. A thread that is exiting, whose ring was
/// released, writes nothing.
fn write(record: Record) {
    let _ = CLAIM.try_with(|claim| claim.ring().write(&record));
}

/// `stage` of `id` begins now.
pub fn begin(stage: Stage, id: u64) {
    write(Record {
        time: now(),
        id,
        point: stage.point(),
        mark: Mark::Begin,
    });
}

/// `stage` of `id` ends now.
pub fn end(stage: Stage, id: u64) {
    write(Record {
        time: now(),
        id,
        point: stage.point(),
        mark: Mark::End,
    });
}

/// `stage` of `id` ran from `start` to `end`, nanoseconds of [`now`]: a stage whose owner was
/// known only once it ended, such as the scheduling of a step whose batches it formed.
pub fn interval(stage: Stage, id: u64, start: u64, end: u64) {
    for (time, mark) in [(start, Mark::Begin), (end, Mark::End)] {
        write(Record {
            time,
            id,
            point: stage.point(),
            mark,
        });
    }
}

/// The request of trace id `request` is a member of batch `batch`: written by the scheduler as
/// it forms the batch, between the batch's [`Stage::Batch`] begin and end.
pub fn member(request: u64, batch: u64) {
    write(Record {
        time: batch,
        id: request,
        point: Stage::Batch.point(),
        mark: Mark::Member,
    });
}

/// The next end of `stage` of `id` this thread writes moved `bytes`.
pub fn bytes(stage: Stage, id: u64, bytes: u64) {
    write(Record {
        time: bytes,
        id,
        point: stage.point(),
        mark: Mark::Bytes,
    });
}

/// Names the place of the calling thread's records.
pub fn label(label: Label) {
    let _ = CLAIM.try_with(|claim| claim.ring().label(label));
}

/// A stage that ends when it is dropped.
#[must_use = "the stage ends when the guard is dropped"]
pub struct Guard {
    stage: Stage,
    id: u64,
}

/// `stage` of `id` begins now and ends when the guard is dropped.
pub fn enter(stage: Stage, id: u64) -> Guard {
    begin(stage, id);
    Guard { stage, id }
}

impl Drop for Guard {
    fn drop(&mut self) {
        end(self.stage, self.id);
    }
}

/// The records of every ring of the process, each ring's with its place, oldest first.
pub fn records() -> Vec<(Label, Vec<Record>)> {
    let rings: Vec<&'static Ring> = lock(&RINGS).clone();
    rings
        .into_iter()
        .map(|ring| (Label::of(ring.label.load(Ordering::Relaxed)), ring.read()))
        .collect()
}

/// A kernel a process timed, as it names the site of a [`KERNEL`] point.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Kernel {
    /// The operator the launch table names.
    pub op: String,
    /// The operator's descriptor key.
    pub descriptor: String,
    /// The module path the launch is inside; empty outside every module.
    pub module: String,
    /// The layers of the kind whose representative instance holds the launch: what the
    /// kernel's time stands for in a step, once for a kernel outside every layer.
    pub layers: u32,
}

/// What a span names: a stage, or a timed kernel.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Name {
    Stage(String),
    Kernel(Kernel),
}

/// One span of a process, as the process names it: what a reader writes as one `span` event.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    /// Nanoseconds since the Unix epoch on the host clock.
    pub start: u64,
    pub end: u64,
    pub process: Process,
    /// The place of the ring that recorded it.
    pub label: Label,
    pub name: Name,
    /// The request's trace id, for a stage of a request and a membership record.
    pub request: Option<u64>,
    /// The batch id, for a stage of a batch, a kernel and a membership record.
    pub batch: Option<u64>,
    /// The bytes the stage moved, when it states them.
    pub bytes: Option<u64>,
}

/// One begin or end of a ring, in the order of [`spans`]'s pairing.
struct Edge {
    time: u64,
    id: u64,
    point: u16,
    begins: bool,
    label: Label,
    bytes: Option<u64>,
}

/// The spans of `records`, the rings of a process ([`records`]): each begin paired with the next
/// end of the same id and point in time order, the members of each batch over its
/// [`Stage::Batch`] span, and every kernel point named by `kernels` (a point it names no kernel
/// of makes no span). In the order the spans began.
pub fn spans(
    records: &[(Label, Vec<Record>)],
    kernels: impl Fn(u16) -> Option<Kernel>,
) -> Vec<Span> {
    let mut edges = Vec::new();
    let mut members = Vec::new();
    for (label, ring) in records {
        // A byte count belongs to the next end its writer writes of the same id and point.
        let mut moved: Vec<(u64, u16, u64)> = Vec::new();
        for record in ring {
            match record.mark {
                Mark::Member => members.push((record.id, record.time)),
                Mark::Bytes => moved.push((record.id, record.point, record.time)),
                Mark::Begin | Mark::End => {
                    let begins = record.mark == Mark::Begin;
                    let bytes = (!begins)
                        .then(|| {
                            let at = moved.iter().position(|&(id, point, _)| {
                                id == record.id && point == record.point
                            })?;
                            Some(moved.swap_remove(at).2)
                        })
                        .flatten();
                    edges.push(Edge {
                        time: record.time,
                        id: record.id,
                        point: record.point,
                        begins,
                        label: *label,
                        bytes,
                    });
                }
            }
        }
    }
    // A begin and its end of one instant pair: the begin first.
    edges.sort_by_key(|edge| (edge.time, !edge.begins));

    let mut open: std::collections::HashMap<(u64, u16), std::collections::VecDeque<&Edge>> =
        std::collections::HashMap::new();
    let mut spans = Vec::new();
    for edge in &edges {
        let key = (edge.id, edge.point);
        if edge.begins {
            open.entry(key).or_default().push_back(edge);
            continue;
        }
        let Some(begin) = open
            .get_mut(&key)
            .and_then(std::collections::VecDeque::pop_front)
        else {
            continue;
        };
        let named = match Stage::of(edge.point) {
            Some(stage) => Some((
                stage.process(),
                Name::Stage(stage.name().to_owned()),
                stage.owner(),
            )),
            None if edge.point >= KERNEL => kernels(edge.point - KERNEL)
                .map(|kernel| (Process::Worker, Name::Kernel(kernel), Owner::Batch)),
            None => None,
        };
        let Some((process, name, owner)) = named else {
            continue;
        };
        // A stage handed between threads, such as a request's from a main process's control
        // thread to its scheduler, is placed where it ended when it began nowhere in particular.
        let label = match begin.label {
            Label::None => edge.label,
            label => label,
        };
        spans.push(Span {
            start: begin.time,
            end: edge.time,
            process,
            label,
            name,
            request: (owner == Owner::Request).then_some(edge.id),
            batch: (owner == Owner::Batch).then_some(edge.id),
            bytes: edge.bytes,
        });
    }

    // Each member spans its batch's formation, in the scheduler that formed it.
    let formed: std::collections::HashMap<u64, (u64, u64, Label)> = spans
        .iter()
        .filter(|span| span.name == Name::Stage(Stage::Batch.name().to_owned()))
        .filter_map(|span| Some((span.batch?, (span.start, span.end, span.label))))
        .collect();
    for (request, batch) in members {
        if let Some(&(start, end, label)) = formed.get(&(batch % IDS)) {
            spans.push(Span {
                start,
                end,
                process: Process::Scheduler,
                label,
                name: Name::Stage(Stage::Batch.name().to_owned()),
                request: Some(request),
                batch: Some(batch % IDS),
                bytes: None,
            });
        }
    }

    spans.sort_by_key(|span| (span.start, std::cmp::Reverse(span.end)));
    spans
}

/// The page of `spans` that starts at `since` (nanoseconds since the Unix epoch): of the spans
/// that ended at or after it, those of the earliest ends, in the order of their ends, as many as
/// their MessagePack encoding (a `Span` list, [`crate::msgpack`]) holds within `bytes`. Returns
/// the encoding and, when spans are left, the end the next page starts from. The spans of one end
/// are never divided between two pages, so a page that starts at the next end holds every span
/// this one left; a first end whose spans alone take more than `bytes` makes a page of its own.
///
/// # Errors
///
/// Fails with `InvalidInput` when a span has no MessagePack form.
pub fn page(
    mut spans: Vec<Span>,
    since: u64,
    bytes: usize,
) -> std::io::Result<(Vec<u8>, Option<u64>)> {
    spans.retain(|span| span.end >= since);
    // A stable sort: the spans of one end keep the order they began in.
    spans.sort_by_key(|span| span.end);

    // A list's encoding is its length's (at most 5 bytes) and then each span's.
    let mut taken = 5;
    let mut cut = spans.len();
    for (at, span) in spans.iter().enumerate() {
        taken += crate::msgpack::encode(span)?.len();
        if taken <= bytes {
            continue;
        }
        let first = spans[..at]
            .iter()
            .rposition(|earlier| earlier.end != span.end)
            .map_or(0, |before| before + 1);
        cut = if first > 0 {
            first
        } else {
            spans
                .iter()
                .position(|later| later.end != span.end)
                .unwrap_or(spans.len())
        };
        break;
    }

    let next = spans.get(cut).map(|span| span.end);
    spans.truncate(cut);
    Ok((crate::msgpack::encode(&spans)?, next))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(time: u64, id: u64, stage: Stage, mark: Mark) -> Record {
        Record {
            time,
            id,
            point: stage.point(),
            mark,
        }
    }

    fn stage(name: &str) -> Name {
        Name::Stage(name.to_owned())
    }

    /// A record keeps its time, its owner, its point and its mark through its 16 bytes, and an
    /// id past 48 bits keeps its low 48.
    #[test]
    fn a_record_round_trips_through_its_two_words() {
        for mark in [Mark::Begin, Mark::End, Mark::Member, Mark::Bytes] {
            let record = Record {
                time: u64::MAX - 7,
                id: IDS - 1,
                point: POINTS - 1,
                mark,
            };
            assert_eq!(Record::of(record.time, record.word()), record);
        }

        let wide = Record {
            time: 1,
            id: IDS + 5,
            point: Stage::Launch.point(),
            mark: Mark::End,
        };
        assert_eq!(Record::of(1, wide.word()).id, 5);
    }

    /// Every stage has its own point below the kernels' points, and is found by it.
    #[test]
    fn each_stage_has_its_own_point_below_the_kernels() {
        let mut points: Vec<u16> = Stage::ALL.iter().map(|stage| stage.point()).collect();
        points.dedup();
        assert_eq!(points.len(), Stage::ALL.len());
        for &stage in Stage::ALL {
            assert!(stage.point() < KERNEL);
            assert_eq!(Stage::of(stage.point()), Some(stage));
        }
    }

    /// A ring keeps its most recent records, oldest first, once its writer wrapped it: all but
    /// the oldest slot, which a write in progress may be overwriting.
    #[test]
    fn a_ring_keeps_its_most_recent_records() {
        let ring = Ring::new(4, Label::Worker { group: 1, rank: 3 });
        for time in 0..10 {
            ring.write(&at(time, 1, Stage::Launch, Mark::Begin));
        }

        let times: Vec<u64> = ring.read().iter().map(|record| record.time).collect();
        assert_eq!(times, [7, 8, 9]);
    }

    /// A reader copying while its writer writes takes only records the writer has not
    /// overwritten: each one it takes is whole, and they are consecutive and in order.
    #[test]
    fn a_reader_beside_its_writer_takes_only_whole_records() {
        let ring = Ring::new(64, Label::None);
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let writer = {
            let stop = std::sync::Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut time = 0;
                while !stop.load(Ordering::Relaxed) {
                    // The id repeats the time, so a torn record shows.
                    ring.write(&at(time, time % IDS, Stage::Launch, Mark::End));
                    time += 1;
                }
            })
        };

        for _ in 0..2000 {
            let read = ring.read();
            assert!(read.len() <= 64);
            for record in &read {
                assert_eq!(record.id, record.time % IDS);
            }
            for pair in read.windows(2) {
                assert_eq!(pair[1].time, pair[0].time + 1);
            }
        }
        stop.store(true, Ordering::Relaxed);
        writer.join().unwrap();
    }

    /// A thread that exits frees its ring, and the next thread to record claims it.
    #[test]
    fn an_exited_threads_ring_is_claimed_again() {
        let ring_of = || {
            std::thread::spawn(|| {
                begin(Stage::Receive, 1);
                CLAIM.with(|claim| claim.ring() as *const Ring as usize)
            })
            .join()
            .unwrap()
        };
        let first = ring_of();
        let second = ring_of();
        let rings = lock(&RINGS).len();
        let third = ring_of();

        // Other tests' threads may hold rings meanwhile; a ring is reused when one is free.
        assert!(first == second || second == third || lock(&RINGS).len() == rings);
    }

    /// A begin pairs with the next end of the same id and point in time order, whichever ring
    /// wrote either; an unended begin and an end without its begin make no span, a span begun on
    /// a ring of no place takes the place of the ring that ended it, and the byte count goes to
    /// the end its writer wrote next.
    #[test]
    fn begins_pair_with_ends_across_rings() {
        let daemon = vec![
            at(10, 7, Stage::Submit, Mark::Begin),
            at(30, 7, Stage::Detokenize, Mark::Begin),
            // Its begin was overwritten.
            at(5, 8, Stage::Route, Mark::End),
        ];
        let other = vec![
            at(20, 7, Stage::Submit, Mark::End),
            Record {
                time: 4096,
                id: 7,
                point: Stage::Handoff.point(),
                mark: Mark::Bytes,
            },
            at(40, 7, Stage::Handoff, Mark::Begin),
            at(50, 7, Stage::Handoff, Mark::End),
        ];
        let records = vec![
            (Label::None, daemon),
            (Label::Worker { group: 0, rank: 1 }, other),
        ];

        let spans = spans(&records, |_| None);
        assert_eq!(spans.len(), 2);
        assert_eq!(
            (
                spans[0].start,
                spans[0].end,
                &spans[0].name,
                spans[0].request
            ),
            (10, 20, &stage("submit"), Some(7))
        );
        assert_eq!(spans[0].process, Process::Daemon);
        // Begun on a ring of no place, it takes the place of the ring that ended it.
        assert_eq!(spans[0].label, Label::Worker { group: 0, rank: 1 });
        assert_eq!(
            (spans[1].process, spans[1].bytes, spans[1].label),
            (
                Process::Storage,
                Some(4096),
                Label::Worker { group: 0, rank: 1 }
            )
        );
    }

    /// A stage that repeats for one owner pairs each begin with its own end, in order.
    #[test]
    fn a_repeated_stage_pairs_in_order() {
        let records = vec![(
            Label::None,
            vec![
                at(1, 3, Stage::Send, Mark::Begin),
                at(2, 3, Stage::Send, Mark::End),
                at(5, 3, Stage::Send, Mark::Begin),
                at(9, 3, Stage::Send, Mark::End),
            ],
        )];

        let spans: Vec<(u64, u64)> = spans(&records, |_| None)
            .iter()
            .map(|span| (span.start, span.end))
            .collect();
        assert_eq!(spans, [(1, 2), (5, 9)]);
    }

    /// A member record becomes a span of its request and batch over the batch's formation; a
    /// member of a batch whose formation is not held makes none.
    #[test]
    fn a_member_spans_its_batchs_formation() {
        let scheduler = vec![
            at(100, 41, Stage::Batch, Mark::Begin),
            Record {
                time: 41,
                id: 9,
                point: Stage::Batch.point(),
                mark: Mark::Member,
            },
            Record {
                time: 42,
                id: 9,
                point: Stage::Batch.point(),
                mark: Mark::Member,
            },
            at(120, 41, Stage::Batch, Mark::End),
        ];
        let records = vec![(Label::Scheduler(2), scheduler)];

        let spans = spans(&records, |_| None);
        let member: Vec<&Span> = spans.iter().filter(|span| span.request.is_some()).collect();
        assert_eq!(member.len(), 1);
        assert_eq!(
            (
                member[0].start,
                member[0].end,
                member[0].request,
                member[0].batch
            ),
            (100, 120, Some(9), Some(41))
        );
        assert_eq!(member[0].label, Label::Scheduler(2));
        assert_eq!(member[0].process, Process::Scheduler);
    }

    /// A kernel point is named by the process's sites, and one it names nothing of is dropped.
    #[test]
    fn a_kernel_point_takes_its_sites_name() {
        let point = |site: u16| KERNEL + site;
        let kernel = |site, mark, time| Record {
            time,
            id: 5,
            point: point(site),
            mark,
        };
        let records = vec![(
            Label::Worker { group: 0, rank: 0 },
            vec![
                kernel(0, Mark::Begin, 1),
                kernel(0, Mark::End, 2),
                kernel(1, Mark::Begin, 3),
                kernel(1, Mark::End, 4),
            ],
        )];

        let spans = spans(&records, |site| {
            (site == 0).then(|| Kernel {
                op: "gemm".to_owned(),
                descriptor: "bf16".to_owned(),
                module: "model.layers.0.mlp".to_owned(),
                layers: 48,
            })
        });
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].batch, Some(5));
        assert_eq!(spans[0].process, Process::Worker);
        assert!(matches!(&spans[0].name, Name::Kernel(kernel) if kernel.layers == 48));
    }

    /// A read of many spans pages them: every page's encoding fits the bytes it is given, the
    /// pages hold every span that ended since the first page's start exactly once, in the order
    /// of their ends, and no end's spans are divided between two pages.
    #[test]
    fn a_read_of_many_spans_pages_them_within_the_bytes_a_page_takes() {
        // Three spans end at every even time; the ones ended before the read's start are left.
        let spans: Vec<Span> = (0..30_000_u64)
            .map(|at| Span {
                start: at,
                end: 1_000 + (at / 3) * 2,
                process: Process::Daemon,
                label: Label::None,
                name: stage("receive"),
                request: Some(at),
                batch: None,
                bytes: None,
            })
            .collect();
        let since = 1_100;
        const BYTES: usize = 64 << 10;

        let mut read: Vec<Span> = Vec::new();
        let mut start = Some(since);
        let mut pages = 0;
        while let Some(from) = start {
            let (encoded, next) = page(spans.clone(), from, BYTES).unwrap();
            assert!(encoded.len() <= BYTES, "a page of {} bytes", encoded.len());
            let part: Vec<Span> = crate::msgpack::decode(&encoded).unwrap();
            if let (Some(last), Some(next)) = (part.last(), next) {
                assert!(
                    next > last.end,
                    "the next page starts past this page's last end"
                );
            }
            read.extend(part);
            start = next;
            pages += 1;
        }

        let mut expected: Vec<Span> = spans.into_iter().filter(|span| span.end >= since).collect();
        expected.sort_by_key(|span| span.end);
        assert!(pages > 1, "the spans take more than one page");
        assert_eq!(read, expected);
    }

    /// An end whose spans alone take more than a page's bytes makes a page of its own, so a read
    /// always advances.
    #[test]
    fn an_end_whose_spans_exceed_a_page_makes_a_page_of_its_own() {
        let span = |request: u64, end: u64| Span {
            start: 0,
            end,
            process: Process::Daemon,
            label: Label::None,
            name: stage("receive"),
            request: Some(request),
            batch: None,
            bytes: None,
        };
        let spans = vec![span(1, 10), span(2, 10), span(3, 10), span(4, 20)];

        let (encoded, next) = page(spans.clone(), 0, 16).unwrap();
        let part: Vec<Span> = crate::msgpack::decode(&encoded).unwrap();
        assert_eq!(part, spans[..3]);
        assert_eq!(next, Some(20));
    }

    /// The cost of one record on this host: a clock read and the three stores, measured over a
    /// million records of one thread. Run with `--ignored --nocapture` to read it.
    #[test]
    #[ignore = "a measurement, not a check"]
    fn the_cost_of_one_record() {
        const RECORDS: u64 = 1_000_000;
        // The ring is claimed before the measurement.
        begin(Stage::Launch, 0);

        let mut costs = Vec::new();
        for _ in 0..7 {
            let started = std::time::Instant::now();
            for id in 0..RECORDS {
                begin(Stage::Launch, id);
            }
            costs.push(started.elapsed().as_nanos() as f64 / RECORDS as f64);
        }
        costs.sort_by(f64::total_cmp);
        let clock = {
            let started = std::time::Instant::now();
            let mut sum = 0u64;
            for _ in 0..RECORDS {
                sum = sum.wrapping_add(now());
            }
            std::hint::black_box(sum);
            started.elapsed().as_nanos() as f64 / RECORDS as f64
        };
        println!(
            "one record: median {:.1} ns (min {:.1}, max {:.1}); the clock read alone {clock:.1} ns",
            costs[costs.len() / 2],
            costs[0],
            costs[costs.len() - 1]
        );
    }
}
