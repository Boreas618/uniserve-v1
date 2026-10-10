//! Canvas warmup: graph capture, or eager sampler shapes, on scratch KV.
//!
//! Startup lends the leading units of the KV pool as scratch: every
//! synthetic canvas row reads a one-token prefix from its own page of every
//! cache group. Each runner then captures its graphs (or, without graph
//! pools, runs its sampler chunk shapes eagerly; see
//! [`crate::runner::canvas::CanvasRunner::warm`]), and the scratch units are
//! reset once the lanes drained.

use super::CanvasRuntime;
use crate::Result;
use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::storage::KVCache;
use crate::warmup::WarmupHooks;

impl CanvasRuntime {
    /// Capture every runner's graphs over scratch prefixes. With experts,
    /// every graph is captured at every expert capacity holding its tokens,
    /// largest first, each warmup an announced expert step.
    pub(crate) fn warm(
        &mut self,
        kv: &mut KVCache,
        experts: Option<&super::super::experts::ExpertRuntime>,
    ) -> Result<()> {
        let groups = kv.plan().group_shapes()?;
        let row_units: usize = groups
            .iter()
            .map(|group| group.units_per_page as usize)
            .sum();
        let rows = self
            .runners
            .iter()
            .map(|runner| runner.shapes().capacity())
            .max()
            .unwrap_or(0);
        if rows == 0 {
            return Ok(());
        }
        let stream = self
            .runners
            .first()
            .and_then(|runner| runner.stream())
            .cloned();
        let scratch = kv.scratch_units((rows * row_units) as u32, stream.as_ref())?;
        // Runners on other streams read the zeroed units.
        if let Some(stream) = &stream {
            stream.synchronize()?;
        }
        let first = *scratch.units().start() as usize;
        // Row `r` reads units `first + r * row_units ..`, one page of every
        // group in table order.
        let units = |row: usize| -> Vec<u32> {
            (0..row_units)
                .map(|unit| (first + row * row_units + unit) as u32)
                .collect()
        };
        let variants = |tokens: usize| -> Vec<Option<usize>> {
            match experts {
                None => vec![None],
                Some(experts) => experts
                    .parallel()
                    .capacities()
                    .iter()
                    .rev()
                    .copied()
                    .filter(|&capacity| capacity >= tokens)
                    .map(Some)
                    .collect(),
            }
        };
        let mut warmed = Ok(());
        for (index, runner) in self.runners.iter_mut().enumerate() {
            let mut coordinate = |extent: usize, forced: Option<usize>| {
                super::coordinator(experts, index, forced)(extent)
            };
            warmed = runner.warm(&units, &variants, &mut coordinate);
            if warmed.is_err() {
                break;
            }
        }
        let finished = scratch.finish(kv, stream.as_ref());
        warmed.and(finished)?;
        for runner in &self.runners {
            let [readouts, tails, steps] = runner.captured();
            tracing::info!(
                target: "uniserve_worker.warmup",
                "{}: captured {} canvas graphs: {readouts} readout, {tails} readout tail and {steps} step buckets",
                runner.owner(),
                readouts + tails + steps,
            );
        }
        Ok(())
    }
}

/// The canvas runtime's warmup work.
pub(crate) struct CanvasWarmup;

impl WarmupHooks for CanvasWarmup {
    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        let parts = backend.parts();
        let (Some(runtime), Some(storage)) = (parts.canvas, parts.storage) else {
            return Ok(());
        };
        let kv = storage
            .kv
            .as_mut()
            .ok_or(crate::Error::State("canvas warmup requires a KV cache"))?;
        runtime.warm(kv, parts.experts).map_err(|error| {
            WorkerError::classify(error, Some("canvas graph capture")).with_phase("warmup")
        })
    }
}
