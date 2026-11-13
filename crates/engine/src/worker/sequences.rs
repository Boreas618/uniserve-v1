//! `Sequences`: the ended sequences of one worker group's ranks, and the group's `completed`
//! word in the buffer pool's registers.
//!
//! The engine numbers a group's jobs that read or write its device block tables (a batch, a
//! shadow batch) when it hands them to the group's ranks, and every rank takes them in that
//! order. A rank has *ended* number `n` once every numbered job it was handed up to `n` has ended
//! on its device: its stream event awaited. The group's `completed` is the smallest ended
//! sequence over its ranks, this node's workers and the processes on other hosts that hold the
//! group's other ranks, each of which reports the smallest over its own. It is a level the
//! buffer pool reads in shared memory: a request stamped with a sequence at or below it no longer
//! names anything a batch of the group can still address.
//!
//! The registers are the engine's: per queue pair whose registers hold the group's word, an
//! engine-end mapping of the queue pair ([`storage::QueuePair::open`] of the queue pair's
//! descriptor) and the word's index, which the answer to `Open` states. The mapping is used for
//! that word alone and is owned here, so it lives as long as anything may write the word. On a
//! host that holds no register (an engine process whose ranks belong to a group scheduled on
//! another host), the value is reported to the scheduler node instead.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use storage::QueuePair;
use storage::runtime::GroupId;

use service::lock;

/// The ended sequences of a worker group's ranks and its `completed` word (module
/// documentation).
pub struct Sequences {
    group: GroupId,
    /// Per rank that reports here, in the order the engine numbered them: this node's workers
    /// of the group, then each peer link that holds ranks of it. Each only rises.
    ended: Vec<AtomicU64>,
    /// The value last written into the registers, with the registers: a write is taken under the
    /// lock, so the word only rises whichever rank's thread writes it.
    registers: Mutex<(u64, Vec<(QueuePair, usize)>)>,
}

impl Sequences {
    /// The sequences of `group`, whose `ranks` ranks report here, writing the group's word into
    /// `registers`: per queue pair whose registers hold it, the engine's mapping of the queue
    /// pair and the word's index, which must be one of its registers. Every rank starts at 0:
    /// nothing has ended, and a request stamped 0 waits for nothing.
    pub fn new(group: GroupId, ranks: usize, registers: Vec<(QueuePair, usize)>) -> Arc<Self> {
        Arc::new(Self {
            group,
            ended: (0..ranks).map(|_| AtomicU64::new(0)).collect(),
            registers: Mutex::new((0, registers)),
        })
    }

    /// The worker group.
    pub(crate) fn group(&self) -> GroupId {
        self.group
    }

    /// How many ranks report here.
    pub(crate) fn ranks(&self) -> usize {
        self.ended.len()
    }

    /// The group's `completed`: the smallest ended sequence over its ranks.
    pub fn completed(&self) -> u64 {
        self.ended
            .iter()
            .map(|ended| ended.load(Ordering::Acquire))
            .min()
            .unwrap_or_default()
    }

    /// States that `rank` has ended every numbered job up to `sequence`, and raises the group's
    /// word in every register when the smallest over the ranks rose. A sequence below the rank's
    /// last changes nothing, and so does a rank that does not report here.
    ///
    /// The caller has awaited the device's completion of every job it states ended; the word is
    /// stored with release ordering, after that wait.
    pub(crate) fn ended(&self, rank: usize, sequence: u64) {
        let Some(ended) = self.ended.get(rank) else {
            return;
        };
        ended.fetch_max(sequence, Ordering::AcqRel);

        let mut registers = lock(&self.registers);
        let completed = self.completed();
        if completed <= registers.0 {
            return;
        }

        registers.0 = completed;
        for (pair, index) in &registers.1 {
            pair.register(*index).store(completed, Ordering::Release);
        }
    }
}

impl std::fmt::Debug for Sequences {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Sequences")
            .field("group", &self.group)
            .field("completed", &self.completed())
            .finish_non_exhaustive()
    }
}
