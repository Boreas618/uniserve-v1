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

    /// The smallest captured bucket of `generation` that holds `shape`, with the number of the
    /// generation it belongs to.
    ///
    /// A shape no captured bucket of the generation holds is an error: the design forbids serving
    /// it from any other graph and forbids an eager fallback (4.5). The caller plans buckets so
    /// that it does not happen; the error names the shape so the plan and the batch can be
    /// compared.
    pub fn select(&self, shape: &Shape, generation: u64) -> Result<(Bucket, &Captured)> {
        let mut best: Option<(Bucket, &Captured)> = None;
        for (&(bucket, captured_generation), captured) in &self.captured {
            if captured_generation != generation || !bucket.holds(shape) {
                continue;
            }
            if best.is_none_or(|(smallest, _)| bucket < smallest) {
                best = Some((bucket, captured));
            }
        }
        best.ok_or_else(|| {
            Error::Invariant(format!(
                "no captured bucket of generation {generation} holds {shape}"
            ))
        })
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

    /// Drops every graph of a generation older than `generation`, in bucket order, and answers the
    /// buckets it dropped.
    ///
    /// This is purge's last step (7.3): a dropped graph releases its executable and its nodes,
    /// and with them the last references that kept the old generation's kernels loaded. It runs
    /// only after the last batch that used the old generation completed.
    ///
    /// The buckets come back so the caller can give each one's arena region to its graph pool:
    /// `Captured` guarantees that the executable goes first, so the regions are released only
    /// after the graphs that addressed them are gone.
    pub fn purge(&mut self, generation: u64) -> Vec<Bucket> {
        let stale: Vec<(Bucket, u64)> = self
            .captured
            .keys()
            .filter(|(_, captured_generation)| *captured_generation < generation)
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

    /// How many graphs the runner holds, of every generation.
    pub fn len(&self) -> usize {
        self.captured.len()
    }

    /// Whether the runner holds no graph.
    pub fn is_empty(&self) -> bool {
        self.captured.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::Variant;

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
        assert!(graphs.purge(1).is_empty());
    }
}
