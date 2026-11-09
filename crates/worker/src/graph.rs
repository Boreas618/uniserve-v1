//! The captured graphs one runner holds, by bucket and generation.
//!
//! A runner serves every batch from a graph captured for a bucket (specs/worker.md 4.5). During a
//! program-layer update it holds the graphs of two generations at once: the ones it is serving
//! from and the ones precopy is preparing for the next generation (7.3). Both are keyed by
//! `(bucket, generation)`, which is what lets purge find the graphs that still reference the old
//! generation's kernels.
//!
//! Only three operations touch the table on a serving path:
//!
//! - [`Graphs::select`], once per batch, picks the smallest captured bucket of the serving
//!   generation that holds the batch's shape;
//! - [`Graphs::captured`] finds one bucket's graph for a recapture or a rebind;
//! - [`Graphs::purge`] drops the graphs of a generation after its last batch.
//!
//! `using` answers the question precopy asks before deciding how to change a generation: which
//! buckets of the old generation launch this operator, and therefore must be rebound or
//! recaptured.

use std::collections::HashMap;

use crate::captured::Captured;
use crate::plan::{Bucket, Shape};
use crate::{Error, Result};

/// Every captured graph of one runner.
#[derive(Default, Debug)]
pub struct Graphs {
    /// `(bucket, generation)` to the graph captured for it.
    captured: HashMap<(Bucket, u64), Captured>,
}

/// The table's selection rule, over the `(bucket, generation)` keys it holds: among the entries whose
/// bucket `accepts`, the **newest** generation at or below `generation`, and within it the
/// **smallest** bucket.
///
/// Newest first is what lets a bucket an update did not prepare keep serving: its programs did not
/// change, so the newest generation that named it is the same program. Smallest within a generation
/// is the planning rule (4.5): a batch is served by the smallest bucket that holds it.
fn served_key(
    keys: impl Iterator<Item = (Bucket, u64)>,
    generation: u64,
    accepts: impl Fn(&Bucket) -> bool,
) -> Option<(Bucket, u64)> {
    let mut best: Option<(Bucket, u64)> = None;
    for (bucket, captured_generation) in keys {
        if captured_generation > generation || !accepts(&bucket) {
            continue;
        }
        let better = match best {
            None => true,
            Some((smallest, newest)) => {
                captured_generation > newest || (captured_generation == newest && bucket < smallest)
            }
        };
        if better {
            best = Some((bucket, captured_generation));
        }
    }
    best
}

/// The entries a purge of `generation` releases, over the table's keys and the generation the
/// runner serves.
///
/// Which captures an ending generation leaves behind depends on which side of the serving generation
/// it ended on, and the two cases are opposite:
///
/// 1. **A commit retired it** (`generation < serving`): release a capture of `generation` only when
///    a **newer** capture of the same bucket exists, because that newer one is what serves from then
///    on. A capture with no newer sibling is the program its bucket still serves (`served_key` takes
///    the newest at or below the serving generation), so releasing it would leave the bucket with
///    nothing to answer.
/// 2. **A revert failed it** (`generation > serving`): release **every** capture of `generation`.
///    Nothing serves from it - a bucket it shares with the serving generation keeps that
///    generation's capture, and a bucket only it captured is one nobody serves any more - and the
///    whole capture set is what the failed update leaves: executables, kernels and their arena
///    regions. Keeping it leaks a capture set per reverted update.
///
/// `generation == serving` releases nothing: a purge never names what serves.
fn released(keys: &[(Bucket, u64)], generation: u64, serving: u64) -> Vec<(Bucket, u64)> {
    if generation == serving {
        return Vec::new();
    }
    if generation > serving {
        return keys
            .iter()
            .filter(|(_, captured_generation)| *captured_generation == generation)
            .copied()
            .collect();
    }
    let above: Vec<Bucket> = keys
        .iter()
        .filter(|(_, captured_generation)| *captured_generation > generation)
        .map(|(bucket, _)| *bucket)
        .collect();
    keys.iter()
        .filter(|(bucket, captured_generation)| {
            *captured_generation == generation && above.contains(bucket)
        })
        .copied()
        .collect()
}

impl Graphs {
    /// No captured graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records `captured` as the graph of `bucket` for `generation`, replacing one already there.
    ///
    /// A replacement drops the old executable first, which releases the kernels it held; the
    /// runner only replaces a graph whose last batch completed.
    pub fn insert(&mut self, bucket: Bucket, generation: u64, captured: Captured) {
        self.captured.insert((bucket, generation), captured);
    }

    /// The graph of `bucket` in `generation`.
    pub fn captured(&self, bucket: Bucket, generation: u64) -> Option<&Captured> {
        self.captured.get(&(bucket, generation))
    }

    /// `bucket`'s graph for `generation`, or, when that generation prepared none, the newest
    /// generation older than it that did.
    ///
    /// A bucket is prepared for a new generation only when an operator it launches changed, so a
    /// bucket that launches none of them has no graph of the new generation and, by construction,
    /// runs exactly the programs it ran in the last generation that named it - the same
    /// `Arc<Kernel>`s, whose keys moved nowhere (8). It is also what keeps an update that changes
    /// only the scheduling policy, and so prepares no bucket at all, from stopping serving at the
    /// switchover.
    pub fn captured_for(&self, bucket: Bucket, generation: u64) -> Option<(u64, &Captured)> {
        let served = served_key(self.captured.keys().copied(), generation, |held| {
            *held == bucket
        })?;
        self.captured
            .get(&served)
            .map(|captured| (served.1, captured))
    }

    /// The smallest captured bucket that holds `shape` and belongs to `generation`, or, when that
    /// generation prepared none, to the newest generation older than it that did.
    ///
    /// The fallback is the program-layer update's own rule rather than a convenience: a bucket is
    /// prepared for a new generation only when an operator it launches changed, so a bucket that
    /// launches none of them has no graph of the new generation and, by construction, runs exactly
    /// the programs it ran in the last generation that named it - the same `Arc<Kernel>`s, whose
    /// keys moved nowhere (8). It is also what keeps an update that changes only the scheduling
    /// policy, and so prepares no bucket at all, from stopping serving at the switchover.
    ///
    /// A shape no generation's bucket holds is an error: the design forbids serving it from another
    /// bucket and forbids planning a new one during serving (4.5). The error names the shape so the
    /// plan and the batch can be compared.
    pub fn select(&self, shape: &Shape, generation: u64) -> Result<(Bucket, &Captured)> {
        let served = served_key(self.captured.keys().copied(), generation, |bucket| {
            bucket.holds(shape)
        })
        .ok_or_else(|| {
            Error::Invariant(format!(
                "no captured bucket of generation {generation} or of a generation it replaces \
                 holds {shape}"
            ))
        })?;
        self.captured
            .get(&served)
            .map(|captured| (served.0, captured))
            .ok_or_else(|| Error::Invariant("the selected graph was lost from the table".into()))
    }

    /// The buckets of `generation` whose graph launches `operator`, in bucket order.
    ///
    /// A program-layer update rebinds or recaptures exactly these buckets for a changed operator
    /// (4.5, 7.3); a bucket whose graph never launches the operator keeps its executable.
    pub fn using(&self, operator: &'static str, generation: u64) -> Vec<Bucket> {
        let mut buckets: Vec<Bucket> = self
            .captured
            .iter()
            .filter(|&(&(_, captured_generation), captured)| {
                captured_generation == generation && captured.uses(operator)
            })
            .map(|(&(bucket, _), _)| bucket)
            .collect();
        buckets.sort_unstable();
        buckets
    }

    /// Whether any generation still has a graph of `bucket`.
    ///
    /// A bucket's arena plan is shared by every generation's graph of that bucket (the pool records
    /// one plan per bucket), so a purge may release the plan only while this answers false.
    pub fn holds(&self, bucket: Bucket) -> bool {
        self.captured.keys().any(|(held, _)| *held == bucket)
    }

    /// Every bucket captured for `generation`, in bucket order.
    pub fn buckets(&self, generation: u64) -> Vec<Bucket> {
        let mut buckets: Vec<Bucket> = self
            .captured
            .keys()
            .filter(|(_, captured_generation)| *captured_generation == generation)
            .map(|(bucket, _)| *bucket)
            .collect();
        buckets.sort_unstable();
        buckets
    }

    /// Every generation with at least one captured graph, in numeric order.
    #[cfg(test)]
    pub fn generations(&self) -> Vec<u64> {
        let mut generations: Vec<u64> = self
            .captured
            .keys()
            .map(|(_, generation)| *generation)
            .collect();
        generations.sort_unstable();
        generations.dedup();
        generations
    }

    /// Drops what an ending `generation` leaves behind, in bucket order, and answers the buckets it
    /// dropped. `serving` is the generation the runner serves at the moment of the purge: it decides
    /// which of the two lifetimes applies (commit: what a newer capture replaced; revert: the whole
    /// abandoned set), and a purge naming it releases nothing.
    ///
    /// This is purge's last step (7.3): a dropped graph releases its executable and its nodes,
    /// and with them the last references that kept that generation's kernels loaded. It runs only
    /// after the last batch that used the generation completed.
    ///
    /// **Exactly that generation, not every older one.** A generation is released by name - a
    /// switchover's replaced one, or one a revert abandoned - and a reverted generation's number is
    /// *newer* than the serving one, so a rule of "everything older than this" would have released
    /// the graphs of the generation that serves: the device would keep answering from nothing.
    ///
    /// A bucket whose only capture is this generation's - one the update did not prepare - is
    /// **kept**, because it is still the program that bucket serves: releasing it would leave the
    /// bucket with no graph to answer from and every request in it would fail after the commit. A
    /// purge releases what a newer generation has replaced, which is what a commit retires.
    ///
    /// The buckets come back so the caller can give each one's arena region to its graph pool:
    /// `Captured` guarantees that the executable goes first, so the regions are released only
    /// after the graphs that addressed them are gone.
    pub fn purge(&mut self, generation: u64, serving: u64) -> Vec<Bucket> {
        // Which captures go depends on which side of `serving` this generation ended on: a commit
        // releases what a newer capture replaced and keeps what still serves, a revert releases the
        // whole abandoned set. `released` states both, and releases nothing for the serving one.
        let keys: Vec<(Bucket, u64)> = self.captured.keys().copied().collect();
        let stale = released(&keys, generation, serving);
        let mut buckets: Vec<Bucket> = stale.iter().map(|(bucket, _)| *bucket).collect();
        buckets.sort_unstable();
        buckets.dedup();
        for key in stale {
            // Dropping the entry drops the executable and then its nodes.
            self.captured.remove(&key);
        }
        buckets
    }

    /// Drops exactly `generation`'s graphs, in bucket order, and answers the buckets it dropped.
    ///
    /// This is what a precopy that never switched over needs (7.3 step 2): the generation was
    /// prepared - rebound or recaptured, its arena regions reserved - and then abandoned, so its
    /// graphs and kernels have no owner left but the runner. It is not `purge`: purge drops what is
    /// *older* than the serving generation, and an abandoned generation is newer than it, so
    /// `purge` cannot reach it. Nothing a serving generation addresses is touched: the entries are
    /// keyed by `(bucket, generation)` and only this generation's keys go.
    ///
    /// The buckets come back so the caller can release each one's arena region when no other
    /// generation still holds it, in the same order `purge` uses: the executable goes first, so
    /// the region is released only after the graphs that addressed it are gone.
    pub fn discard(&mut self, generation: u64) -> Vec<Bucket> {
        let stale: Vec<(Bucket, u64)> = self
            .captured
            .keys()
            .filter(|(_, captured_generation)| *captured_generation == generation)
            .copied()
            .collect();
        let mut buckets: Vec<Bucket> = stale.iter().map(|(bucket, _)| *bucket).collect();
        buckets.sort_unstable();
        buckets.dedup();
        for key in stale {
            // Dropping the entry drops the executable and then its nodes.
            self.captured.remove(&key);
        }
        buckets
    }

    /// Whether the runner holds no graph.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.captured.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Variant;

    /// The selection rule, over keys only: the newest generation at or below the one asked for, and
    /// the smallest bucket within it. This is what lets a bucket an update did not prepare keep
    /// serving - an update that swaps only the scheduling policy prepares no bucket at all - so it
    /// is tested without a device, where `Captured` is not needed.
    #[test]
    fn the_newest_generation_at_or_below_the_asked_one_serves_a_bucket() {
        let bucket = Bucket::decode(4, Variant::DecodeGreedy);
        let larger = Bucket::decode(8, Variant::DecodeGreedy);
        let keys = [(bucket, 1), (larger, 1), (bucket, 3), (larger, 5)];
        let only = |held: &Bucket| *held == bucket;

        // The generation asked for serves the bucket it prepared.
        assert_eq!(served_key(keys.into_iter(), 3, only), Some((bucket, 3)));
        // One that prepared nothing for it serves from the newest older generation that did.
        assert_eq!(served_key(keys.into_iter(), 4, only), Some((bucket, 3)));
        assert_eq!(served_key(keys.into_iter(), 2, only), Some((bucket, 1)));
        // A generation prepared later never serves an earlier request: a revert must not be served
        // by the generation that replaced it.
        assert_eq!(served_key(keys.into_iter(), 0, only), None);
        assert_eq!(served_key(keys.into_iter(), 5, |_| true), Some((larger, 5)));
        // Within one generation the smallest bucket that accepts the batch wins.
        assert_eq!(served_key(keys.into_iter(), 1, |_| true), Some((bucket, 1)));
    }

    /// `Graphs` is exercised through its selection and purge rules; a `Captured` needs a device,
    /// so the table's own rules are what these tests pin, with the entries built by the runner
    /// tests on a device (the caller runs those).
    #[test]
    fn selection_is_by_generation_and_smallest_bucket() {
        let mut graphs = Graphs::new();
        assert!(graphs.is_empty());

        // Without a captured graph no bucket serves a shape, and the error names the shape.
        let shape = Shape::decode(4, Variant::DecodeGreedy);
        let error = graphs.select(&shape, 1).expect_err("no graph, no bucket");
        assert!(error.to_string().contains("generation 1"), "{error}");

        // A table of no graph has no buckets of any generation.
        assert!(graphs.buckets(1).is_empty());
        assert!(graphs.using("rms_norm", 1).is_empty());
        assert!(graphs.generations().is_empty());
        assert!(graphs.purge(1, 1).is_empty());
    }

    /// What a commit releases, over keys only: the captures a newer generation has replaced, and
    /// never the capture a bucket the update did not prepare still serves from. A purge that took
    /// that one would leave the bucket with no graph, and every request in it would fail after the
    /// commit - the failure this rule exists to prevent.
    #[test]
    fn a_purge_keeps_the_capture_an_unprepared_bucket_serves_from() {
        let prepared = Bucket::decode(4, Variant::DecodeGreedy);
        let untouched = Bucket::decode(8, Variant::DecodeGreedy);
        let keys = [(prepared, 1), (prepared, 2), (untouched, 1)];

        // Generation 2 replaced the prepared bucket's generation-1 capture: that one is released.
        assert_eq!(released(&keys, 1, 2), vec![(prepared, 1)]);
        // The untouched bucket's only capture is generation 1's, and it still serves: kept.
        assert!(!released(&keys, 1, 2).contains(&(untouched, 1)));
        // Purging the generation that serves releases nothing.
        assert!(released(&keys, 2, 2).is_empty());
        // A bucket prepared for the first time in generation 2 has no older capture to release.
        assert!(released(&[(prepared, 2)], 1, 2).is_empty());
    }

    /// A reverted update is the other lifetime: the failed generation sits **above** the serving
    /// one, so nothing "newer" supersedes it and a rule written for a commit would never release it.
    /// The whole capture set goes - every bucket, whether or not the serving generation also holds
    /// that bucket - because nothing serves from it and keeping it leaks a capture set per revert.
    #[test]
    fn a_purge_releases_the_whole_set_of_a_failed_generation() {
        let shared = Bucket::decode(4, Variant::DecodeGreedy);
        let only_the_failed = Bucket::decode(8, Variant::DecodeGreedy);
        let keys = [(shared, 1), (shared, 2), (only_the_failed, 2)];

        // Served generation 1, reverted from 2: both captures of generation 2 are released, and the
        // shared bucket keeps the one generation 1 captured.
        assert_eq!(
            released(&keys, 2, 1),
            vec![(shared, 2), (only_the_failed, 2)]
        );
        // And nothing of the serving generation is ever named.
        assert!(released(&keys, 1, 1).is_empty());
    }
}
