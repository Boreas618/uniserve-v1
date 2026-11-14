//! The buffer pool's trace and its replay (`specs/buffer-pool.md` §18).
//!
//! The buffer pool is deterministic in its inputs, so the trace of its inputs
//! is the trace of everything it did. The buffer pool's thread appends every
//! input it acts on, in the order it acts, to a buffered file of its own per
//! deployment ([`Trace`]): the pool it starts from, as the pool's image (an
//! empty pool, or the pool a restart in place handed over); each session
//! opened with its `Open`, and each session closed; each request taken, with
//! its queue pair and its record as it arrived; each value of a `completed`
//! word it read and acted on; the clock, when a tick passed a deadline and
//! before any input that follows a change of the clock, since the clock sets
//! the deadlines of the tasks that start; each task handed to whoever runs
//! it, each key a task reports done and each task's end; and what else the
//! server tells the pool: the limits of tasks, an update committed or rolled
//! back, a peer's holds, a segment granted or withdrawn. A second file beside
//! it keeps every completion the pool produced, in the order it produced
//! them.
//!
//! A record is a kind byte, the length of its body as a little-endian `u32`,
//! and the body: little-endian fields, a request or a completion as the
//! record of [`storage::runtime`] it arrived or left as, in the version of
//! its session, and an `Open` and a pool's image in MessagePack.
//!
//! [`replay`] drives a new buffer pool with a trace's inputs, checks every
//! completion it produces against the run's, and writes after every input
//! the pool's state in the terms of the design model ([`stated`]), which
//! `verification/buffer-pool/steps.py` checks against the model's steps.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::rc::Rc;

use storage::protocol::{Key, Segment};
use storage::runtime::{
    Completion, GroupId, Open, OwnerId, Request, RequestId, Stamps, Tier, VERSION,
};

use super::{Block, BufferPool, ClientId, Copy, Deferred, Held, QueueId, SLOT, Target};

/// A task: the queue pair and the number of the request that started it.
type Task = (QueueId, RequestId);

// The kinds of the records of a trace's inputs.
const IMAGE: u8 = 0;
const OPEN: u8 = 1;
const CLOSE: u8 = 2;
const REQUEST: u8 = 3;
const COMPLETED: u8 = 4;
const TICK: u8 = 5;
const START: u8 = 6;
const DONE: u8 = 7;
const ENDED: u8 = 8;
const LIMITS: u8 = 9;
const PUBLISH: u8 = 10;
const ROLL_BACK: u8 = 11;
const HOLD: u8 = 12;
const RELEASE_PEER: u8 = 13;
const GRANT: u8 = 14;
const WITHDRAW: u8 = 15;
/// The kind of the records of the completions beside the inputs.
const COMPLETION: u8 = 16;

/// The bytes a trace buffers before it writes them out.
const BUFFER: usize = 1 << 20;

/// An input the buffer pool's thread acts on, as a trace records it.
pub(crate) enum Input {
    /// The pool the trace goes on from, as its image.
    Image(Vec<u8>),
    Open(ClientId, Open),
    Close(ClientId),
    /// A request's record as it arrived on its queue pair, readable or not.
    Request(QueueId, Vec<u8>),
    /// A value of a group's `completed` word.
    Completed(ClientId, GroupId, u64),
    /// The host's monotonic clock, in nanoseconds.
    Tick(u64),
    /// The pool handed a task to whoever runs it. The pool does not act on
    /// it; a replay checks that its pool started the same task.
    Start(Task),
    /// A task's next key is done, with the tokens its block holds.
    Done(Task, u16),
    Ended(Task, bool),
    /// How long a transfer and a put to the remote store may take.
    Limits((u64, u64)),
    /// The update of an epoch on trial is committed, or rolled back.
    Publish(u64),
    RollBack(u64),
    /// A peer holds host copies, or lets go of them.
    Hold(u64, Vec<Key>),
    ReleasePeer(u64),
    /// A segment of a region, by the region's name, granted with its first
    /// unit and its units, or withdrawn.
    Grant(String, Segment, u32, u32),
    Withdraw(String, Segment),
}

impl Input {
    /// The kind and the body of the input's record.
    fn encode(&self) -> io::Result<(u8, Vec<u8>)> {
        let mut body = Vec::new();
        let kind = match self {
            Self::Image(image) => {
                body.extend_from_slice(image);
                IMAGE
            }
            Self::Open(client, open) => {
                body.extend_from_slice(&client.to_le_bytes());
                body.extend(rmp_serde::to_vec(open).map_err(io::Error::other)?);
                OPEN
            }
            Self::Close(client) => {
                body.extend_from_slice(&client.to_le_bytes());
                CLOSE
            }
            Self::Request(queue, record) => {
                queued(&mut body, *queue);
                body.extend_from_slice(record);
                REQUEST
            }
            Self::Completed(client, group, value) => {
                body.extend_from_slice(&client.to_le_bytes());
                body.push(group.0);
                body.extend_from_slice(&value.to_le_bytes());
                COMPLETED
            }
            Self::Tick(now) => {
                body.extend_from_slice(&now.to_le_bytes());
                TICK
            }
            Self::Start(task) => {
                tasked(&mut body, *task);
                START
            }
            Self::Done(task, tokens) => {
                tasked(&mut body, *task);
                body.extend_from_slice(&tokens.to_le_bytes());
                DONE
            }
            Self::Ended(task, ok) => {
                tasked(&mut body, *task);
                body.push(u8::from(*ok));
                ENDED
            }
            Self::Limits((transfer, put)) => {
                body.extend_from_slice(&transfer.to_le_bytes());
                body.extend_from_slice(&put.to_le_bytes());
                LIMITS
            }
            Self::Publish(epoch) | Self::RollBack(epoch) => {
                body.extend_from_slice(&epoch.to_le_bytes());
                if matches!(self, Self::Publish(_)) {
                    PUBLISH
                } else {
                    ROLL_BACK
                }
            }
            Self::Hold(peer, keys) => {
                body.extend_from_slice(&peer.to_le_bytes());
                keys.iter().for_each(|key| body.extend_from_slice(&key.0));
                HOLD
            }
            Self::ReleasePeer(peer) => {
                body.extend_from_slice(&peer.to_le_bytes());
                RELEASE_PEER
            }
            Self::Grant(region, segment, first, units) => {
                segmented(&mut body, *segment);
                body.extend_from_slice(&first.to_le_bytes());
                body.extend_from_slice(&units.to_le_bytes());
                body.extend_from_slice(region.as_bytes());
                GRANT
            }
            Self::Withdraw(region, segment) => {
                segmented(&mut body, *segment);
                body.extend_from_slice(region.as_bytes());
                WITHDRAW
            }
        };
        Ok((kind, body))
    }

    /// The input of a record of the kind `kind` with the body `body`.
    fn decode(kind: u8, mut body: &[u8]) -> Result<Self, String> {
        let body = &mut body;
        let input = match kind {
            IMAGE => Self::Image(rest(body)),
            OPEN => {
                let client = u32::from_le_bytes(field(body)?);
                let open = rmp_serde::from_slice(&rest(body)).map_err(|error| error.to_string())?;
                Self::Open(client, open)
            }
            CLOSE => Self::Close(u32::from_le_bytes(field(body)?)),
            REQUEST => Self::Request(queue(body)?, rest(body)),
            COMPLETED => Self::Completed(
                u32::from_le_bytes(field(body)?),
                GroupId(u8::from_le_bytes(field(body)?)),
                u64::from_le_bytes(field(body)?),
            ),
            TICK => Self::Tick(u64::from_le_bytes(field(body)?)),
            START => Self::Start(task(body)?),
            DONE => Self::Done(task(body)?, u16::from_le_bytes(field(body)?)),
            ENDED => Self::Ended(task(body)?, field::<1>(body)? != [0]),
            LIMITS => Self::Limits((
                u64::from_le_bytes(field(body)?),
                u64::from_le_bytes(field(body)?),
            )),
            PUBLISH => Self::Publish(u64::from_le_bytes(field(body)?)),
            ROLL_BACK => Self::RollBack(u64::from_le_bytes(field(body)?)),
            HOLD => {
                let peer = u64::from_le_bytes(field(body)?);
                let mut keys = Vec::new();
                while !body.is_empty() {
                    keys.push(Key(field(body)?));
                }
                Self::Hold(peer, keys)
            }
            RELEASE_PEER => Self::ReleasePeer(u64::from_le_bytes(field(body)?)),
            GRANT => {
                let segment = segment(body)?;
                let (first, units) = (
                    u32::from_le_bytes(field(body)?),
                    u32::from_le_bytes(field(body)?),
                );
                Self::Grant(named(body)?, segment, first, units)
            }
            WITHDRAW => {
                let segment = segment(body)?;
                Self::Withdraw(named(body)?, segment)
            }
            other => return Err(format!("no input is of the kind {other}")),
        };
        if body.is_empty() {
            Ok(input)
        } else {
            Err("a record has bytes after its last field".into())
        }
    }
}

fn queued(body: &mut Vec<u8>, queue: QueueId) {
    body.extend_from_slice(&queue.0.to_le_bytes());
    body.extend_from_slice(&queue.1.to_le_bytes());
}

fn tasked(body: &mut Vec<u8>, task: Task) {
    queued(body, task.0);
    body.extend_from_slice(&task.1.0.to_le_bytes());
}

fn segmented(body: &mut Vec<u8>, segment: Segment) {
    body.extend_from_slice(&segment.epoch.to_le_bytes());
    body.extend_from_slice(&segment.ordinal.to_le_bytes());
}

/// The next `N` bytes of a body.
fn field<const N: usize>(body: &mut &[u8]) -> Result<[u8; N], String> {
    let (head, tail) = body
        .split_first_chunk::<N>()
        .ok_or("a record ends inside a field")?;
    *body = tail;
    Ok(*head)
}

/// The bytes left of a body.
fn rest(body: &mut &[u8]) -> Vec<u8> {
    std::mem::take(body).to_vec()
}

fn queue(body: &mut &[u8]) -> Result<QueueId, String> {
    Ok((
        u32::from_le_bytes(field(body)?),
        u16::from_le_bytes(field(body)?),
    ))
}

fn task(body: &mut &[u8]) -> Result<Task, String> {
    Ok((queue(body)?, RequestId(u64::from_le_bytes(field(body)?))))
}

fn segment(body: &mut &[u8]) -> Result<Segment, String> {
    Ok(Segment {
        epoch: u64::from_le_bytes(field(body)?),
        ordinal: u32::from_le_bytes(field(body)?),
    })
}

fn named(body: &mut &[u8]) -> Result<String, String> {
    String::from_utf8(rest(body)).map_err(|error| error.to_string())
}

/// Appends a record; returns its bytes.
fn frame(file: &mut impl Write, kind: u8, body: &[u8]) -> io::Result<u64> {
    let length = u32::try_from(body.len()).map_err(io::Error::other)?;
    file.write_all(&[kind])?;
    file.write_all(&length.to_le_bytes())?;
    file.write_all(body)?;
    Ok(5 + u64::from(length))
}

/// A record of a file: where it begins, in bytes, its kind and its body.
type Record<'a> = (u64, u8, &'a [u8]);

/// The records of a file.
fn frames(bytes: &[u8]) -> Result<Vec<Record<'_>>, String> {
    let mut found = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let length = bytes.get(at + 1..at + 5).map(|length| {
            u32::from_le_bytes([length[0], length[1], length[2], length[3]]) as usize
        });
        let body = length.and_then(|length| bytes.get(at + 5..at + 5 + length));
        let Some(body) = body else {
            return Err(format!("the record at byte {at} ends early"));
        };
        found.push((at as u64, bytes[at], body));
        at += 5 + body.len();
    }
    Ok(found)
}

/// The files of the trace of `deployment` in `directory`: its inputs, and
/// the completions beside them. A character of the deployment's name that a
/// file name may not hold safely is `_` there.
pub fn files(directory: &Path, deployment: &str) -> (PathBuf, PathBuf) {
    let name: String = deployment
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || "-_.".contains(character) {
                character
            } else {
                '_'
            }
        })
        .collect();
    (
        directory.join(format!("{name}.inputs")),
        directory.join(format!("{name}.completions")),
    )
}

/// The trace of one deployment's buffer pool, being written: its inputs and,
/// beside them, the completions its pool produced. It writes from the buffer
/// pool's thread, through buffers of its own, never on a step's path.
pub(crate) struct Trace {
    inputs: BufWriter<File>,
    completions: BufWriter<File>,
    /// Where the next record of the inputs begins, in bytes of their file.
    position: u64,
    /// The clock as the pool was last told it, and as the trace last
    /// recorded it.
    clock: (u64, u64),
}

impl Trace {
    /// Starts the trace of `pool`, the pool of `deployment`, in `directory`:
    /// its files are appended to, and it begins with the pool's image.
    ///
    /// # Errors
    ///
    /// Fails when a file cannot be opened or written.
    pub(crate) fn start(directory: &Path, deployment: &str, pool: &BufferPool) -> io::Result<Self> {
        std::fs::create_dir_all(directory)?;
        let (inputs, completions) = files(directory, deployment);
        let open = |path: &Path| OpenOptions::new().create(true).append(true).open(path);
        let inputs = open(&inputs)?;
        let position = inputs.metadata()?.len();
        let mut trace = Self {
            inputs: BufWriter::with_capacity(BUFFER, inputs),
            completions: BufWriter::with_capacity(BUFFER, open(&completions)?),
            position,
            clock: (pool.now, pool.now),
        };
        let image = rmp_serde::to_vec(pool).map_err(io::Error::other)?;
        trace.record(&Input::Image(image))?;
        Ok(trace)
    }

    /// Appends an input, after the clock when the clock changed since the
    /// trace last recorded it. Returns where the input's record begins.
    ///
    /// # Errors
    ///
    /// Fails when the record cannot be written.
    pub(crate) fn record(&mut self, input: &Input) -> io::Result<u64> {
        if let Input::Tick(now) = input {
            self.clock = (*now, *now);
        } else if self.clock.0 != self.clock.1 {
            self.clock.1 = self.clock.0;
            self.append(&Input::Tick(self.clock.0))?;
        }
        self.append(input)
    }

    fn append(&mut self, input: &Input) -> io::Result<u64> {
        let (kind, body) = input.encode()?;
        let at = self.position;
        self.position += frame(&mut self.inputs, kind, &body)?;
        Ok(at)
    }

    /// The pool was told the time and no deadline passed: the trace states
    /// the time before the next input.
    pub(crate) fn clock(&mut self, now: u64) {
        self.clock.0 = now;
    }

    /// Appends the completions `pool` produced, in order, each in the
    /// version its queue pair is written in.
    ///
    /// # Errors
    ///
    /// Fails when a record cannot be written.
    pub(crate) fn completed(
        &mut self,
        pool: &BufferPool,
        produced: &[(QueueId, RequestId, Completion)],
    ) -> io::Result<()> {
        for (queue, request, completion) in produced {
            let mut body = Vec::new();
            queued(&mut body, *queue);
            completion.encode_as(version(pool, *queue), *request, &mut body);
            frame(&mut self.completions, COMPLETION, &body)?;
        }
        Ok(())
    }

    /// Writes out what the trace buffers.
    ///
    /// # Errors
    ///
    /// Fails when a file cannot be written.
    pub(crate) fn flush(&mut self) -> io::Result<()> {
        self.inputs.flush()?;
        self.completions.flush()
    }
}

/// The version of the runtime interface a queue pair's records are in: its
/// session's, or this build's once the session is gone, as the buffer pool's
/// thread writes them.
fn version(pool: &BufferPool, queue: QueueId) -> u32 {
    pool.version(queue.0).unwrap_or(VERSION)
}

/// Acts on an input as the buffer pool's thread does: the replay's pool and
/// the enumeration's are driven by the same inputs the thread records. An
/// unreadable request is answered by the thread and changes nothing.
///
/// # Errors
///
/// Fails on an image that is not one.
pub(crate) fn apply(pool: &mut BufferPool, input: &Input) -> Result<(), String> {
    match input {
        Input::Image(image) => {
            *pool = rmp_serde::from_slice(image).map_err(|error| error.to_string())?;
        }
        Input::Open(client, open) => {
            // A layout the pool refused changed nothing, in the run too.
            let _ = pool.admit(*client, open);
        }
        Input::Close(client) => pool.end(*client),
        Input::Request(queue, record) => {
            if let Ok((id, request)) = Request::decode_as(record, version(pool, *queue)) {
                pool.take(*queue, id, request);
            }
        }
        Input::Completed(client, group, value) => pool.advance(*client, *group, *value),
        Input::Tick(now) => {
            pool.tick(*now);
        }
        Input::Start(_) => {}
        Input::Done(task, tokens) => pool.done(*task, *tokens),
        Input::Ended(task, ok) => pool.ended(*task, *ok),
        Input::Limits(limits) => pool.set_limits(*limits),
        Input::Publish(epoch) => pool.commit_update(*epoch),
        Input::RollBack(epoch) => pool.roll_back(*epoch),
        Input::Hold(peer, keys) => {
            pool.hold(*peer, keys);
        }
        Input::ReleasePeer(peer) => {
            pool.release_peer(*peer);
        }
        Input::Grant(region, segment, first, units) => {
            if let Some(number) = pool.region_number(region) {
                let _ = pool.grant_segment(number, *segment, *first, *units);
            }
        }
        Input::Withdraw(region, segment) => {
            if let Some(number) = pool.region_number(region) {
                let _ = pool.withdraw_segment(number, *segment);
            }
        }
    }
    Ok(())
}

/// The copies a completion asks a worker to run: the model's `ncopy`
/// counts them.
pub(crate) fn copies_asked(completion: &Completion) -> u32 {
    match completion {
        Completion::Copy { to, .. } => to.len() as u32,
        _ => 0,
    }
}

/// A completion's record of `version` as a replay states it.
fn shown(record: &[u8], version: u32) -> String {
    match Completion::decode_as(record, version) {
        Ok((request, completion)) => format!("{completion:?} for request {}", request.0),
        Err(error) => format!("an unreadable completion ({error})"),
    }
}

/// Replays the trace whose inputs are in the file `inputs` (the completions
/// are beside it, [`files`]) on a new buffer pool. Each completion the pool
/// produces is checked against the run's, in order; the pool's own check
/// holds after every input; and each task the run started is the next the
/// pool started. With `states`, the pool's state in the model's terms is
/// written there after every input ([`stated`], with [`classes`]), each
/// after the step that led to it (`step N N+1 <input>`, [`stated_input`])
/// and after the layout of the session ([`layout`]) whenever an input may
/// change it. The owners stated are those an `Attach` names. A run's last
/// completions may be missing from its file when the run stopped before it
/// wrote them; any other difference is an error.
///
/// Returns the inputs replayed.
///
/// # Errors
///
/// The first difference: the input after which it arose, with the run's
/// record and the replay's.
pub fn replay(inputs: &Path, mut states: Option<&mut dyn Write>) -> Result<u64, String> {
    let read =
        |path: &Path| std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()));
    let bytes = read(inputs)?;
    let mut records = Vec::new();
    for (at, kind, body) in frames(&bytes)? {
        let input = Input::decode(kind, body)
            .map_err(|error| format!("the input at byte {at}: {error}"))?;
        records.push((at, input));
    }
    let completions = read(&inputs.with_extension("completions"))?;
    let mut recorded = VecDeque::new();
    for (at, kind, mut body) in frames(&completions)? {
        if kind != COMPLETION {
            return Err(format!("the completion at byte {at} is of the kind {kind}"));
        }
        recorded.push_back((at, queue(&mut body)?, body.to_vec()));
    }

    // An attach is written alike in every version, so this build's reads
    // each one, whatever its session's version.
    let mut owners = Vec::new();
    for (_, input) in &records {
        if let Input::Request(queue, record) = input
            && let Ok((_, Request::Attach { owner, .. })) = Request::decode(record)
            && !owners.contains(&(*queue, owner))
        {
            owners.push((*queue, owner));
        }
    }

    let mut pool = BufferPool::new((0, 0));
    let mut started = VecDeque::new();
    let (mut copies, mut replayed) = (0, 0);
    let mut write = |text: String| match states.as_mut() {
        Some(states) => states
            .write_all(text.as_bytes())
            .map_err(|error| error.to_string()),
        None => Ok(()),
    };
    write(format!(
        "{}state 0\n{}\n\n",
        layout(&pool),
        stated(&pool, &owners, 0)
    ))?;
    for (at, input) in &records {
        let named =
            |pool: &BufferPool| format!("the input at byte {at} ({})", stated_input(pool, input));
        if let Input::Start(task) = input {
            match started.pop_front() {
                Some(found) if found == *task => continue,
                found => {
                    return Err(format!(
                        "{}: the run started the task of request {}, the replay {:?}",
                        named(&pool),
                        task.1.0,
                        found.map(|(_, request): Task| request.0)
                    ));
                }
            }
        }
        apply(&mut pool, input).map_err(|error| format!("{}: {error}", named(&pool)))?;
        if let Input::Image(_) = input {
            // The tasks a pool started and the run did not record are gone with it.
            started.clear();
        }
        replayed += 1;

        for (queue, request, completion) in pool.completions() {
            copies += copies_asked(&completion);
            let mut record = Vec::new();
            completion.encode_as(version(&pool, queue), request, &mut record);
            match recorded.pop_front() {
                Some((_, ran, ran_record)) if ran == queue && ran_record == record => {}
                Some((_, ran, ran_record)) => {
                    return Err(format!(
                        "after {}: the run completed {} on queue pair {ran:?}, the replay {completion:?} for request {} on {queue:?}",
                        named(&pool),
                        shown(&ran_record, version(&pool, ran)),
                        request.0
                    ));
                }
                None => {}
            }
        }
        started.extend(pool.started());
        pool.stopped();
        pool.check()
            .map_err(|inconsistency| format!("after {}: {inconsistency}", named(&pool)))?;

        let changes_layout = matches!(input, Input::Image(_) | Input::Open(..) | Input::Close(_));
        write(format!(
            "step {} {replayed} {}\n{}state {replayed}\n{}\n{}\n\n",
            replayed - 1,
            stated_input(&pool, input),
            if changes_layout {
                layout(&pool)
            } else {
                String::new()
            },
            stated(&pool, &owners, copies),
            classes(&pool)
        ))?;
    }
    if let Some((at, queue, record)) = recorded.pop_front() {
        return Err(format!(
            "the run completed {} on queue pair {queue:?} (byte {at} of its completions), and the replay did not",
            shown(&record, version(&pool, queue))
        ));
    }
    Ok(replayed)
}

/// An input to `pool` as `verification/buffer-pool/steps.py` reads it: its
/// kind, and what it names in the terms of [`stated`] (owners and positions
/// by number, groups as `g<n>`, keys by their 32 bytes in hex, stamps as
/// `g<n>:<sequence>`); a request is read in its session's version.
pub(crate) fn stated_input(pool: &BufferPool, input: &Input) -> String {
    match input {
        Input::Image(_) => "image".into(),
        Input::Open(..) => "open".into(),
        Input::Close(_) => "close".into(),
        Input::Request(queue, record) => match Request::decode_as(record, version(pool, *queue)) {
            Ok((_, request)) => stated_request(&request),
            Err(_) => "unreadable".into(),
        },
        Input::Completed(_, group, value) => format!("completed g=g{} value={value}", group.0),
        Input::Tick(_) => "tick".into(),
        Input::Start(_) => "start".into(),
        Input::Done(..) => "done".into(),
        Input::Ended(_, ok) => format!("ended ok={}", u8::from(*ok)),
        Input::Limits(_) => "limits".into(),
        Input::Publish(_) => "publish".into(),
        Input::RollBack(_) => "roll-back".into(),
        Input::Hold(..) => "hold".into(),
        Input::ReleasePeer(_) => "release-peer".into(),
        Input::Grant(..) => "grant".into(),
        Input::Withdraw(..) => "withdraw".into(),
    }
}

/// A key as the stated inputs and states name it: its 32 bytes in hex, so
/// that the keys of a serving deployment, which share leading bytes, are
/// told apart.
fn key_name(key: &Key) -> String {
    key.0
        .iter()
        .fold(String::with_capacity(64), |mut name, byte| {
            let _ = write!(name, "{byte:02x}");
            name
        })
}

fn stated_request(request: &Request) -> String {
    let keys = |keys: &[Key]| {
        let named: Vec<String> = keys.iter().map(key_name).collect();
        named.join(",")
    };
    let after = |stamps: &Stamps| {
        let named: Vec<String> = stamps
            .iter()
            .map(|(group, at)| format!("g{}:{at}", group.0))
            .collect();
        named.join(",")
    };
    let span = |at: u32, count: usize| {
        let named: Vec<String> = (0..count as u32)
            .map(|offset| (at + offset).to_string())
            .collect();
        named.join(",")
    };
    match request {
        Request::Attach { owner, group } => format!("attach o={} g=g{}", owner.0, group.0),
        Request::Detach {
            owner,
            after: stamps,
        } => format!("detach o={} after={}", owner.0, after(stamps)),
        Request::Allocate {
            owner,
            class,
            at,
            count,
            ..
        } => {
            format!(
                "allocate o={} class={class} ps={}",
                owner.0,
                span(*at, *count as usize)
            )
        }
        Request::Pin {
            owner,
            class,
            at,
            keys: named,
        } => format!(
            "pin o={} class={class} ps={} ks={}",
            owner.0,
            span(*at, named.len()),
            keys(named)
        ),
        Request::Release {
            owner,
            class,
            at,
            count,
            after: stamps,
        } => format!(
            "release o={} class={class} ps={} after={}",
            owner.0,
            span(*at, *count as usize),
            after(stamps)
        ),
        Request::Commit {
            owner,
            class,
            at,
            key,
            after: stamps,
            ..
        } => format!(
            "commit o={} class={class} ps={at} ks={} after={}",
            owner.0,
            keys(&[*key]),
            after(stamps)
        ),
        Request::Clone {
            owner,
            class,
            at,
            key,
        } => {
            format!(
                "clone o={} class={class} ps={at} ks={}",
                owner.0,
                keys(&[*key])
            )
        }
        Request::Transfer {
            owner,
            group,
            after: stamps,
        } => {
            format!(
                "transfer o={} to=g{} after={}",
                owner.0,
                group.0,
                after(stamps)
            )
        }
        Request::Evict { keys: named } => format!("evict ks={}", keys(named)),
        Request::Victims { .. } => "victims".into(),
        Request::List => "list".into(),
        Request::Resize { .. } => "resize".into(),
        Request::Load {
            owner,
            class,
            at,
            keys: named,
        } => format!(
            "load o={} class={class} ps={} ks={}",
            owner.0,
            span(*at, named.len()),
            keys(named)
        ),
        Request::Store {
            group,
            keys: named,
            tier,
        } => {
            let tier = match tier {
                Tier::Device => "device",
                Tier::Host => "host",
                Tier::Remote => "remote",
            };
            format!("store g=g{} ks={} tier={tier}", group.0, keys(named))
        }
        Request::Save { owner, key, .. } => format!("save o={} ks={}", owner.0, keys(&[*key])),
        Request::Restore { owner, key } => format!("restore o={} ks={}", owner.0, keys(&[*key])),
        Request::Finish {
            ok, after: stamps, ..
        } => format!("finish ok={} after={}", u8::from(*ok), after(stamps)),
        Request::Prefetch {
            group,
            class,
            keys: named,
            source,
            ..
        } => {
            let source = match source {
                storage::runtime::Source::Remote => "remote",
                storage::runtime::Source::Peers => "peers",
            };
            format!(
                "prefetch g=g{} class={class} ks={} source={source}",
                group.0,
                keys(named)
            )
        }
        Request::Replicate {
            owner,
            class,
            at,
            group,
            after: stamps,
            ..
        } => format!(
            "replicate o={} class={class} ps={at} to=g{} after={}",
            owner.0,
            group.0,
            after(stamps)
        ),
    }
}

/// What a state in the model's terms leaves out and the model has no terms
/// for: the class of each cached key, in lines that start with `#`, which
/// `verification/buffer-pool/steps.py` reads and `agreement.py` never sees.
/// The model has one class, so a key cached in another is outside its terms
/// (a key names what its block holds, A6).
pub(crate) fn classes(pool: &BufferPool) -> String {
    let classes: BTreeMap<String, u8> = pool
        .cache
        .iter()
        .map(|(key, entry)| (key_name(key), entry.class))
        .collect();
    let lines: Vec<String> = classes
        .into_iter()
        .map(|(key, class)| format!("# class {key} {class}"))
        .collect();
    lines.join("\n")
}

/// The layout of the pool's session in the terms of [`stated`], or
/// `layout none` unless exactly one session is open: per group, the bit of
/// its memory, the region of its slot rows, its regions of the class 0 and
/// the bits of the host parts it copies that class with; per class, its
/// host region.
pub(crate) fn layout(pool: &BufferPool) -> String {
    if pool.sessions.len() != 1 {
        return "layout none\n".into();
    }
    let mut text = String::new();
    let groups: BTreeMap<u8, _> = pool
        .groups
        .iter()
        .map(|((_, group), found)| (group.0, found))
        .collect();
    for (group, found) in groups {
        let kv: Vec<String> = found
            .regions
            .first()
            .into_iter()
            .flatten()
            .map(u8::to_string)
            .collect();
        let parts = found.parts.first().copied().unwrap_or(0);
        let parts: Vec<String> = (0..64)
            .filter(|bit| parts & (1 << bit) != 0)
            .map(|bit| format!("m{bit}"))
            .collect();
        let _ = writeln!(
            text,
            "layout group g{group} memory m{} rows {} kv {} parts {}",
            found.memory,
            found.slots,
            kv.join(","),
            parts.join(",")
        );
    }
    let hosts: BTreeSet<u8> = pool
        .classes
        .values()
        .flat_map(|(classes, _)| {
            classes
                .iter()
                .filter_map(|class| class.host.map(|(region, _)| region))
        })
        .collect();
    for region in hosts {
        let _ = writeln!(text, "layout host {region}");
    }
    text
}

/// The pool's state in the terms of the design model, one line per fact, in
/// order: the copies the workers were asked for (`copies`), each group's
/// `completed` word, the free units, each owner of `owners` by the queue pair
/// that attaches it (live with its group, slot and its table of the class 0
/// by position, or gone, or new), each cached key with its copies, the
/// memories that hold each block an owner or the cache holds on the device
/// and the host parts that hold each host copy, the copies in flight with
/// their keys and the host parts a store or a save writes, what waits for a
/// stamp with the stamps it still waits for (`after=g<n>:<sequence>,...`,
/// one `release` line per group and stamp a release waits for), and the
/// tasks with their keys and the host parts a prefetch writes, a task of
/// several keys as one per key. A block a waiting commit holds states its
/// memories too. A unit is `region:unit`, a key its
/// 32 bytes in hex. `verification/buffer-pool/agreement.py` writes the
/// model's states in the same lines.
pub(crate) fn stated(pool: &BufferPool, owners: &[(QueueId, OwnerId)], copies: u32) -> String {
    let unit = |block: &Rc<Block>| {
        let runs: Vec<String> = block
            .stated()
            .into_iter()
            .map(|run| format!("{}:{}", run.0, run.1))
            .collect();
        runs.join("+")
    };
    let units = |blocks: BTreeSet<String>| blocks.into_iter().collect::<Vec<_>>().join(" ");
    // Memories as the bits of a block's validity they are: `m2,m3`.
    let bits = |mask: u64| {
        let named: Vec<String> = (0..64)
            .filter(|bit| mask & (1 << bit) != 0)
            .map(|bit| format!("m{bit}"))
            .collect();
        named.join(",")
    };
    let mut lines = vec![format!("ncopy {copies}")];

    let completed: BTreeMap<u8, u64> = pool
        .groups
        .iter()
        .map(|((_, group), found)| (group.0, found.completed))
        .collect();
    lines.push(
        completed
            .iter()
            .fold("completed".to_owned(), |line, (group, at)| {
                format!("{line} g{group}={at}")
            }),
    );

    let mut free = BTreeSet::new();
    for (id, region) in pool.regions.iter().enumerate() {
        for (first, count) in region.free_runs() {
            free.extend((first..first + count).map(|unit| format!("{id}:{unit}")));
        }
    }
    lines.push(format!("free {}", units(free)));

    // An owner's number is taken once on its queue pair: a number at or
    // below the highest taken there that names no owner is of one that is
    // gone.
    for (queue, number) in owners {
        let taken = pool.numbers.get(queue).map_or(0, |numbers| numbers.0);
        let line = match pool.owners.get(&(queue.0, *number)) {
            Some(owner) => {
                let mut line = format!(
                    "owner {} live g{} slot={}",
                    number.0,
                    owner.group.0,
                    unit(&owner.slot)
                );
                for (position, held) in owner.tables.first().into_iter().flatten() {
                    let kind = if matches!(held, Held::Private(_)) {
                        "private"
                    } else {
                        "shared"
                    };
                    let _ = write!(line, " p{position}={kind}:{}", unit(held.block()));
                }
                line
            }
            None if number.0 <= taken => format!("owner {} gone", number.0),
            None => format!("owner {} new", number.0),
        };
        lines.push(line);
    }

    let mut valid = BTreeMap::new();
    let mut note = |block: &Rc<Block>| {
        let memories: Vec<String> = (0..64)
            .filter(|bit| block.valid.get() & (1 << bit) != 0)
            .map(|bit| format!("m{bit}"))
            .collect();
        valid.insert(unit(block), memories.join(" "));
    };
    for owner in pool.owners.values() {
        owner
            .tables
            .first()
            .into_iter()
            .flatten()
            .for_each(|(_, held)| note(held.block()));
    }
    // A commit that waits holds its block, whose memories the commit's
    // effect keeps when the owner is gone.
    for (_, work) in pool.groups.values().flat_map(|group| &group.deferred) {
        if let Deferred::Commit { block, .. } = work {
            note(block);
        }
    }
    for (key, entry) in &pool.cache {
        entry.device.iter().for_each(|(block, _)| note(block));
        entry.host.iter().for_each(|(block, ..)| note(block));
        let device = entry
            .device
            .first()
            .map_or("-".to_owned(), |(block, _)| unit(block));
        let host = entry
            .host
            .as_ref()
            .map_or("-".to_owned(), |(block, ..)| unit(block));
        lines.push(format!(
            "key {} d={device} h={host} remote={}",
            key_name(key),
            u8::from(entry.remote)
        ));
    }
    lines.extend(
        valid
            .into_iter()
            .map(|(unit, memories)| format!("valid {unit} {memories}").trim_end().to_owned()),
    );

    let copy_line = |copy: &Copy| {
        // The key a copy caches its block under: a load's, a store's and a
        // save's; a clone and a restore cache nothing.
        let key = match &copy.to {
            Target::Owner { key: Some(key), .. } | Target::Entry { key, .. } => key_name(key),
            _ => "-".to_owned(),
        };
        let (kind, owner, position, memory) = match copy.to {
            Target::Owner {
                owner,
                at,
                key: None,
                ..
            } => ("clone", owner.0.to_string(), at.to_string(), copy.memory),
            Target::Owner {
                owner,
                at,
                key: Some(_),
                ..
            } => ("load", owner.0.to_string(), at.to_string(), copy.memory),
            Target::Slot { owner } => ("restore", owner.0.to_string(), "-".to_owned(), 0),
            Target::Entry { .. } if copy.reads.first().is_some_and(|block| block.class == SLOT) => {
                // A save reads its owner's slot, while that owner is there.
                let owner = pool
                    .owners
                    .iter()
                    .find(|(_, owner)| Rc::ptr_eq(&owner.slot, &copy.reads[0]));
                let owner = owner.map_or("-".to_owned(), |((_, owner), _)| owner.0.to_string());
                ("save", owner, "-".to_owned(), 0)
            }
            _ => ("store", "-".to_owned(), "-".to_owned(), 0),
        };
        let memory = if memory == 0 {
            "-".to_owned()
        } else {
            format!("m{}", memory.trailing_zeros())
        };
        // The host parts a store or a save writes: a store's fill those its
        // host block lacks, otherwise those of its group.
        let parts = if matches!(kind, "store" | "save") {
            bits(copy.memory)
        } else {
            String::new()
        };
        format!(
            "kind={kind} o={owner} p={position} read={} write={} fill={} mem={memory} k={key} parts={parts}",
            unit(&copy.reads[0]),
            unit(&copy.write),
            u8::from(copy.fill)
        )
    };

    // The stamps work still waits for: the one it waits at and those of its
    // `rest` that its group's `completed` word has not reached, by group.
    let pending = |client: ClientId, at: (GroupId, u64), rest: &Stamps| {
        let completed = |group: GroupId| {
            pool.groups
                .get(&(client, group))
                .map_or(0, |found| found.completed)
        };
        let waits: BTreeMap<u8, u64> = std::iter::once(at)
            .chain(rest.iter().copied())
            .filter(|(group, stamp)| *stamp > completed(*group))
            .map(|(group, stamp)| (group.0, stamp))
            .collect();
        let named: Vec<String> = waits
            .into_iter()
            .map(|(group, stamp)| format!("g{group}:{stamp}"))
            .collect();
        named.join(",")
    };

    for ((client, group), found) in &pool.groups {
        for (stamp, work) in &found.deferred {
            let at = (*group, *stamp);
            match work {
                Deferred::Release { blocks } => {
                    let held: BTreeSet<String> = blocks.iter().map(&unit).collect();
                    lines.push(format!(
                        "release after={} {}",
                        pending(*client, at, &Vec::new()),
                        units(held)
                    ));
                }
                Deferred::Commit {
                    owner,
                    at: position,
                    block,
                    key,
                    rest,
                    ..
                } => lines.push(format!(
                    "commit o={} p={position} u={} k={} after={}",
                    owner.0,
                    unit(block),
                    key_name(key),
                    pending(*client, at, rest)
                )),
                Deferred::Finish {
                    queue,
                    copy,
                    ok,
                    rest,
                    ..
                } => {
                    let copies = pool
                        .copies
                        .get(&(*queue, *copy))
                        .map_or(&[][..], Vec::as_slice);
                    let after = pending(*client, at, rest);
                    lines.extend(copies.iter().map(|copy| {
                        format!(
                            "finish ok={} after={after} {}",
                            u8::from(*ok),
                            copy_line(copy)
                        )
                    }));
                }
                Deferred::Transfer {
                    owner, to, rest, ..
                } => lines.push(format!(
                    "transfer o={} to=g{} after={}",
                    owner.0,
                    to.0,
                    pending(*client, at, rest)
                )),
                Deferred::Replicate {
                    owner,
                    at: position,
                    block,
                    to,
                    rest,
                    ..
                } => lines.push(format!(
                    "replicate o={} p={position} to=g{} u={} after={}",
                    owner.0,
                    to.0,
                    unit(block),
                    pending(*client, at, rest)
                )),
            }
        }
    }

    for (task, copies) in &pool.copies {
        let expired = u8::from(!pool.answers.contains_key(task));
        match copies.first().map(|copy| &copy.to) {
            Some(Target::Group { owner, to }) => {
                let held = copies
                    .iter()
                    .flat_map(|copy| &copy.reads)
                    .map(&unit)
                    .collect();
                let writes = copies
                    .iter()
                    .filter(|copy| copy.fill)
                    .map(|copy| unit(&copy.write))
                    .collect();
                let slot = copies
                    .iter()
                    .find(|copy| !copy.fill)
                    .map_or("-".to_owned(), |copy| unit(&copy.write));
                let line = format!(
                    "task transfer o={} to=g{} slot={slot} units={} writes={}",
                    owner.0,
                    to.0,
                    units(held),
                    units(writes)
                );
                lines.push(line.trim_end().to_owned());
            }
            Some(Target::Memory {
                owner, at, group, ..
            }) => lines.push(format!(
                "task replicate o={} p={at} to=g{} u={}",
                owner.0,
                group.0,
                unit(&copies[0].write)
            )),
            Some(Target::Entry { remote, .. }) if copies[0].deadline.is_some() => {
                // A task of several keys is a task per key to the model,
                // which makes one per key. A prefetch writes the host parts
                // of the group that asked for it; a put writes none.
                let kind = if *remote { "put" } else { "prefetch" };
                for copy in copies {
                    if let Target::Entry { key, .. } = &copy.to {
                        lines.push(format!(
                            "task {kind} h={} expired={expired} k={} parts={}",
                            unit(&copy.write),
                            key_name(key),
                            bits(copy.memory)
                        ));
                    }
                }
            }
            _ => lines.extend(
                copies
                    .iter()
                    .map(|copy| format!("copy {}", copy_line(copy))),
            ),
        }
    }
    // A fact is stated once: two releases of the same blocks at the same
    // stamp are one line, as they are one release of the model.
    lines.sort_unstable();
    lines.dedup();
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::{BufWriter, Write};
    use std::path::PathBuf;

    use storage::protocol::{Key, Segment};
    use storage::runtime::{GroupId, Open, RequestId, VERSION};

    use super::{Input, replay};

    /// Every kind of input reads back from its record as it was written,
    /// so a replay acts on what the buffer pool's thread acted on.
    #[test]
    fn every_kind_of_input_reads_back_as_it_was_recorded() {
        let task = ((1, 2), RequestId(3));
        let segment = Segment {
            epoch: 4,
            ordinal: 5,
        };
        let open = Open {
            version: VERSION,
            epoch: 6,
            trial: true,
            regions: Vec::new(),
            classes: Vec::new(),
            trajectory: 0,
            groups: Vec::new(),
            schedulers: vec![vec![GroupId(0)]],
            peers: vec!["peer:1".into()],
        };
        let inputs = [
            Input::Image(vec![1, 2, 3]),
            Input::Open(1, open),
            Input::Close(1),
            Input::Request((1, 0), vec![9; 12]),
            Input::Completed(1, GroupId(2), 7),
            Input::Tick(8),
            Input::Start(task),
            Input::Done(task, 64),
            Input::Ended(task, true),
            Input::Limits((10, 11)),
            Input::Publish(12),
            Input::RollBack(13),
            Input::Hold(14, vec![Key([1; 32]), Key([2; 32])]),
            Input::ReleasePeer(15),
            Input::Grant("units".into(), segment, 16, 17),
            Input::Withdraw("units".into(), segment),
        ];
        let mut kinds = Vec::new();
        for input in inputs {
            let (kind, body) = input.encode().unwrap();
            let read = Input::decode(kind, &body).unwrap();
            assert_eq!(read.encode().unwrap(), (kind, body));
            kinds.push(kind);
        }
        kinds.dedup();
        assert_eq!(kinds.len(), 16, "each input has a kind of its own");
    }

    /// The replay of a run's traces: `replay.sh` runs it with the run's
    /// trace directory in `UNISERVE_POOL_TRACE` and, to have the states
    /// written for the check against the model, a directory in
    /// `UNISERVE_POOL_STATES`.
    #[test]
    #[ignore = "replays the traces of a run; run by verification/buffer-pool/replay.sh"]
    fn the_traces_of_a_run_replay_with_every_completion_equal() {
        let traces = PathBuf::from(std::env::var("UNISERVE_POOL_TRACE").unwrap());
        let states = std::env::var("UNISERVE_POOL_STATES")
            .ok()
            .map(PathBuf::from);
        let mut found: Vec<PathBuf> = std::fs::read_dir(&traces)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "inputs")
            })
            .collect();
        found.sort();
        assert!(!found.is_empty(), "{} holds no trace", traces.display());

        for inputs in found {
            let name = inputs.file_stem().unwrap().to_string_lossy().into_owned();
            let path = |states: &PathBuf| states.join(format!("{name}.steps"));
            let mut written = states
                .as_ref()
                .map(|states| BufWriter::new(File::create(path(states)).unwrap()));
            let replayed = replay(&inputs, written.as_mut().map(|file| file as &mut dyn Write));
            match replayed {
                Ok(count) => println!(
                    "replayed {name}: {count} inputs, every completion equal and every check clean"
                ),
                Err(difference) => panic!("{name}: {difference}"),
            }
        }
    }
}
