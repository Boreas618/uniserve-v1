//! The buffer pool's trace (`specs/buffer-pool.md` §18).
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
//! server tells the pool: an update committed or rolled
//! back, a peer's holds, a segment granted or withdrawn. A second file beside
//! it keeps every completion the pool produced, in the order it produced
//! them.
//!
//! A record is a kind byte, the length of its body as a little-endian `u32`,
//! and the body: little-endian fields, a request or a completion as the
//! record of [`storage::runtime`] it arrived or left as, and an `Open` and a
//! pool's image in MessagePack. The replay reads it back (`super::replay`, with
//! the crate's `verification` feature).

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use storage::protocol::{Key, Segment};
use storage::runtime::{Completion, GroupId, Open, OwnerId, RequestId};

use super::{BufferPool, ClientId, Encoding, QueueId};

/// A task: the queue pair and the number of the request that started it.
pub(super) type Task = (QueueId, RequestId);

// The kinds of the records of a trace's inputs.
pub(super) const IMAGE: u8 = 0;
pub(super) const OPEN: u8 = 1;
pub(super) const CLOSE: u8 = 2;
pub(super) const REQUEST: u8 = 3;
pub(super) const COMPLETED: u8 = 4;
pub(super) const TICK: u8 = 5;
pub(super) const START: u8 = 6;
pub(super) const DONE: u8 = 7;
pub(super) const ENDED: u8 = 8;
pub(super) const PUBLISH: u8 = 10;
pub(super) const ROLL_BACK: u8 = 11;
pub(super) const HOLD: u8 = 12;
pub(super) const RELEASE_PEER: u8 = 13;
pub(super) const GRANT: u8 = 14;
pub(super) const WITHDRAW: u8 = 15;
/// The kind of the records of the completions beside the inputs.
pub(super) const COMPLETION: u8 = 16;
pub(super) const PEER_COPIES: u8 = 17;
pub(super) const BORROW: u8 = 18;
pub(super) const GIVE_BACK: u8 = 19;
pub(super) const RELEASE_BORROWS: u8 = 20;

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
    /// What the peers hold of a prefetch's keys: per copy of the task, in
    /// order, the encoding of the peer's host copy it reads, if any.
    PeerCopies(Task, Vec<Option<Encoding>>),
    /// Another host's server, by its connection, borrows the lend of a session's owner.
    Borrow(u64, ClientId, OwnerId),
    /// The lend of a session's owner ends.
    GiveBack(ClientId, OwnerId),
    /// The borrows of a connection end: of one lend, or every one.
    ReleaseBorrows(u64, Option<(ClientId, OwnerId)>),
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
            Self::PeerCopies(task, held) => {
                tasked(&mut body, *task);
                // Per copy a byte, 1 when its encoding's two digests follow.
                for encoding in held {
                    body.push(u8::from(encoding.is_some()));
                    if let Some((tensors, layout)) = encoding {
                        body.extend_from_slice(tensors);
                        body.extend_from_slice(layout);
                    }
                }
                PEER_COPIES
            }
            Self::Borrow(peer, client, owner) => {
                body.extend_from_slice(&peer.to_le_bytes());
                body.extend_from_slice(&client.to_le_bytes());
                body.extend_from_slice(&owner.0.to_le_bytes());
                BORROW
            }
            Self::GiveBack(client, owner) => {
                body.extend_from_slice(&client.to_le_bytes());
                body.extend_from_slice(&owner.0.to_le_bytes());
                GIVE_BACK
            }
            Self::ReleaseBorrows(peer, lend) => {
                body.extend_from_slice(&peer.to_le_bytes());
                if let Some((client, owner)) = lend {
                    body.extend_from_slice(&client.to_le_bytes());
                    body.extend_from_slice(&owner.0.to_le_bytes());
                }
                RELEASE_BORROWS
            }
        };
        Ok((kind, body))
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

/// Appends a record; returns its bytes.
fn frame(file: &mut impl Write, kind: u8, body: &[u8]) -> io::Result<u64> {
    let length = u32::try_from(body.len()).map_err(io::Error::other)?;
    file.write_all(&[kind])?;
    file.write_all(&length.to_le_bytes())?;
    file.write_all(body)?;
    Ok(5 + u64::from(length))
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
            inputs: BufWriter::new(inputs),
            completions: BufWriter::new(open(&completions)?),
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

    /// Appends the completions the pool produced, in order.
    ///
    /// # Errors
    ///
    /// Fails when a record cannot be written.
    pub(crate) fn completed(
        &mut self,
        produced: &[(QueueId, RequestId, Completion)],
    ) -> io::Result<()> {
        for (queue, request, completion) in produced {
            let mut body = Vec::new();
            queued(&mut body, *queue);
            completion.encode(*request, &mut body);
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

#[cfg(test)]
mod tests {
    use storage::protocol::{Key, Segment};
    use storage::runtime::{GroupId, Open, OwnerId, RequestId, VERSION};

    use super::Input;

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
            Input::Publish(12),
            Input::RollBack(13),
            Input::Hold(14, vec![Key([1; 32]), Key([2; 32])]),
            Input::ReleasePeer(15),
            Input::Grant("units".into(), segment, 16, 17),
            Input::Withdraw("units".into(), segment),
            Input::PeerCopies(task, vec![Some(([18; 32], [19; 32])), None]),
            Input::Borrow(20, 2, OwnerId(21)),
            Input::GiveBack(2, OwnerId(21)),
            Input::ReleaseBorrows(20, Some((2, OwnerId(21)))),
        ];
        let mut kinds = Vec::new();
        for input in inputs {
            let (kind, body) = input.encode().unwrap();
            let read = Input::decode(kind, &body).unwrap();
            assert_eq!(read.encode().unwrap(), (kind, body));
            kinds.push(kind);
        }
        // Each input has a kind of its own: no two kinds read back as one.
        let mut distinct = kinds.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), kinds.len(), "{kinds:?}");
    }
}
