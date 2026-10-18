//! Text warmup: graph capture on scratch prefix cache and a representative request.
//!
//! Capture runs every runner's prefill buckets (largest footprint first, so
//! smaller captures reuse pool blocks), then its decode buckets (largest
//! first). Each bucket's rows get startup scratch units installed as block
//! tables of reserved slots `1..=rows`; nothing else is resident during
//! startup. A decode bucket first prefills one real token per row and
//! commits its cache length, so the captured decode attends to
//! model-produced K/V; that commit captures the bucket's commit graph. A
//! decode bucket has two graphs: one selecting greedily and one sampling
//! each row from its top-k candidates, captured over neutral parameter
//! rows (the sampler's kernels do not depend on the values). Every capture
//! runs its step eagerly first, which also compiles the step's kernels.
//!
//! Each step and each commit is one job of the runner's lane: the eager
//! call, then the capture, whose graph the job's completion carries. The
//! runner stores it before the next step is prepared, so later steps launch
//! the graphs of earlier ones. A bucket's scratch tables are two more jobs
//! of the lane: one installs them before the bucket's steps, one clears
//! them and resets the scratch units after.
//!
//! The serving path of these graphs is warmed by the engine, with a real
//! request through its scheduler once the rank reported.

use uniserve_ipc::{CallKind, ForwardMode};

use std::sync::Arc;

use super::context::{block_row, next_position, text_row};
use super::{CausalLMRunners, Embeddings};
use crate::cuda_graph::capture_order;
use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;
use crate::ffi::Backing;
use crate::lane::{BLOCK_ON_TIMEOUT, Job, JobQueues};
use crate::parallel::ExpertUnion;
use crate::runner::TokenSelection;
use crate::runner::causal_lm::{COMMIT_CACHE, PromptChunks, RowSource, StepRequest, TokenRow};
use crate::runner::denoiser::image_model::ImageModel;
use crate::sampling::RowSampling;
use crate::storage::PrefixCache;
use crate::storage::memory::{Memory, StorageView};
use crate::stream::Issue;
use crate::warmup::WarmupHooks;
use crate::{Error, Result};

/// The causal LM runners' warmup work.
pub(crate) struct CausalLMWarmup {
    /// Tokens one capture row may hold.
    pub(crate) row_tokens: usize,
}

impl WarmupHooks for CausalLMWarmup {
    fn capture(&mut self, backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        let lanes = backend.job_queues().clone();
        let parts = backend.parts();
        let image = parts.image.as_ref().map(|image| *image.facts());
        let (Some(runtime), Some(storage)) = (parts.text, parts.storage) else {
            return Ok(());
        };
        let prefix_cache = storage
            .prefix_cache
            .as_mut()
            .ok_or(Error::State("text warmup requires a prefix cache"))?;
        runtime
            .warmup(prefix_cache, self.row_tokens, parts.experts, &lanes)
            .map_err(|error| {
                WorkerError::classify(error, Some("text graph capture")).with_phase("warmup")
            })?;
        // Dedicated expert ranks follow these eager steps as announced
        // warmup steps, as they follow the captures' eager warmups.
        let Some(image) = image else {
            return Ok(());
        };
        runtime
            .warm_context(prefix_cache, &image, parts.experts, &lanes)
            .map_err(|error| {
                WorkerError::classify(error, Some("text context warmup")).with_phase("warmup")
            })
    }
}

/// Rows of `lengths` host tokens for consecutive slots from `first`.
fn rows(
    first: usize,
    lengths: &[usize],
    prefix: usize,
    selection: TokenSelection,
) -> Vec<TokenRow> {
    lengths
        .iter()
        .enumerate()
        .map(|(row, &length)| TokenRow {
            slot: (first + row) as u32,
            source: RowSource::Host {
                tokens: vec![0; length],
                position: prefix as u64,
            },
            prefix: prefix as u32,
            query: length as u32,
            selection,
            gated: false,
            force_finish: false,
            causal: true,
        })
        .collect()
}

impl CausalLMRunners {
    /// Capture every runner's graphs, each step a job of its lane.
    ///
    /// With experts, each microbatch runner captures every bucket at every
    /// step capacity that holds it, largest first: each capture's eager
    /// warmup is a collective expert step at that capacity (announced to
    /// dedicated expert ranks), in which the other microbatches join.
    fn warmup(
        &mut self,
        prefix_cache: &mut PrefixCache,
        row_tokens: usize,
        experts: Option<&Arc<ExpertUnion>>,
        lanes: &JobQueues,
    ) -> Result<()> {
        let shapes = prefix_cache.plan().group_shapes()?;
        let pages: Vec<(usize, usize)> = shapes
            .iter()
            .map(|shape| (shape.page_tokens as usize, shape.units_per_page as usize))
            .collect();
        let units = prefix_cache.planes().num_units() as usize - 1;
        // Step capacities a graph of `tokens` tokens is captured at.
        let variants = |tokens: usize| -> Vec<Option<usize>> {
            match experts {
                None => vec![None],
                Some(experts) => {
                    let mut capacities: Vec<Option<usize>> = experts
                        .parallel()
                        .capacities()
                        .iter()
                        .copied()
                        .filter(|&capacity| capacity >= tokens)
                        .map(Some)
                        .collect();
                    capacities.reverse();
                    capacities
                }
            }
        };
        for index in 0..self.runners.len() {
            let (group, peer) = self.locate(index)?;
            let prefill = capture_order(self.runners[index].shapes().prefill().to_vec(), |shape| {
                (shape.token_bucket * shape.row_bucket, shape.token_bucket)
            });
            let decode = capture_order(self.runners[index].shapes().decode().to_vec(), |&rows| {
                (rows, rows)
            });
            let prime = self.runners[index]
                .shapes()
                .prefill()
                .iter()
                .map(|shape| shape.row_bucket - 1)
                .max()
                .unwrap_or(usize::MAX)
                .max(1);
            let stream = self.runners[index].stream().cloned();
            let stream = stream.as_ref();
            let kind = self.kind(group)?;
            let run = |runtime: &mut CausalLMRunners,
                       prefix_cache: &mut PrefixCache,
                       rows: &[TokenRow],
                       mode: ForwardMode,
                       row_sampling: Option<&RowSampling>,
                       capacity: Option<usize>,
                       job: Job| {
                runtime.warmup_step(
                    experts,
                    lanes,
                    (group, peer),
                    prefix_cache,
                    (rows, &[], mode, row_sampling),
                    capacity,
                    job,
                )
            };
            for shape in prefill {
                let lengths = shape.scratch_lengths(row_tokens, &pages, units);
                let selection = if shape.outputs {
                    TokenSelection::LastLogits
                } else {
                    TokenSelection::Cache
                };
                for capacity in variants(shape.token_bucket) {
                    let rows = rows(1, &lengths, 0, selection);
                    self.warmup_bucket(
                        lanes,
                        kind,
                        prefix_cache,
                        &lengths,
                        stream,
                        |runtime, prefix_cache, _| {
                            run(
                                runtime,
                                prefix_cache,
                                &rows,
                                ForwardMode::Prefill,
                                None,
                                capacity,
                                Job::new(),
                            )
                        },
                    )?;
                }
            }
            for size in decode {
                let lengths = vec![1; size];
                let tables = vec![2; size];
                self.warmup_bucket(
                    lanes,
                    kind,
                    prefix_cache,
                    &tables,
                    stream,
                    |runtime, prefix_cache, after| {
                        // One real token per row, prefilled in batches no larger than
                        // the scheduler admits (the largest prefill bucket's live
                        // rows) and committed as each slot's cache; then the decode
                        // the graph captures with every row's continuation flag set.
                        for first in (0..size).step_by(prime) {
                            let count = prime.min(size - first);
                            run(
                                runtime,
                                prefix_cache,
                                &rows(first + 1, &lengths[..count], 0, TokenSelection::LastLogits),
                                ForwardMode::Prefill,
                                None,
                                None,
                                Job::new(),
                            )?;
                        }
                        let slots: Vec<usize> = (1..=size).collect();
                        let prefilled: Vec<[i64; 8]> = slots
                            .iter()
                            .map(|&slot| [slot as i64, COMMIT_CACHE, -1, 1, 0, 0, 0, 0])
                            .collect();
                        runtime.warmup_commit(lanes, index, group, &prefilled)?;

                        // The flags are set by the job of the first decode step
                        // and cleared by the job that clears the tables.
                        let mut job = Job::new();
                        runtime.state.set_predicates(&slots, true, job.on(stream))?;
                        runtime
                            .state
                            .set_predicates(&slots, false, after.on(stream))?;
                        let neutral = RowSampling::neutral(size);
                        for capacity in variants(size) {
                            for row_sampling in [None, Some(&neutral)] {
                                run(
                                    runtime,
                                    prefix_cache,
                                    &rows(1, &lengths, 1, TokenSelection::LastLogits),
                                    ForwardMode::Decode,
                                    row_sampling,
                                    capacity,
                                    std::mem::take(&mut job),
                                )?;
                            }
                        }
                        Ok(())
                    },
                )?;
            }
            let runner = &self.runners[index];
            tracing::info!(
                "captured {} text graphs: {} decode and {} prefill buckets",
                runner.captured(),
                runner.shapes().decode().len(),
                runner.shapes().prefill().len(),
            );
        }
        Ok(())
    }

    /// Run a capture bucket's `steps` over scratch tables of `lengths`
    /// tokens for slots `1..`. One job of `kind`'s lane installs the tables
    /// before the steps; one clears them and resets the scratch units after
    /// the steps, whether or not they succeeded. `steps` records into that
    /// second job what the device undoes before the tables are cleared.
    fn warmup_bucket(
        &mut self,
        lanes: &JobQueues,
        kind: CallKind,
        prefix_cache: &mut PrefixCache,
        lengths: &[usize],
        stream: Option<&uniserve_cuda::Stream>,
        steps: impl FnOnce(&mut Self, &mut PrefixCache, &Job) -> Result<()>,
    ) -> Result<()> {
        let perform = |job: Job| -> Result<()> {
            let completion = lanes.block_on(kind, job, BLOCK_ON_TIMEOUT)?;
            completion
                .check()
                .map(drop)
                .map_err(|failure| failure.error)
        };

        let before = Job::new();
        let scratch = prefix_cache.install_scratch(lengths, before.on(stream))?;
        perform(before)?;

        let after = Job::new();
        let stepped = steps(self, prefix_cache, &after);
        let released = prefix_cache.release_scratch(scratch, lengths.len(), after.on(stream));
        let performed = perform(after);
        stepped.and(released).and(performed)
    }

    /// Run the eager context prefills of every runner whose rows place
    /// vision features, over scratch tables of slot 1, on a model whose
    /// image facts are `image`.
    ///
    /// A context prefill writes a vision block (non-causal rows that attend
    /// to all of themselves, at explicit positions, their features replacing
    /// the placeholder embeddings) whole, either in a chunk with prompt text
    /// before it or alone. It always runs eagerly, so no capture prepares
    /// its attention inputs; each of the two compositions runs here once,
    /// so that the kernels serving them are selected and compiled before
    /// the first image request. A block of four features in a 2 x 2 grid stands for
    /// every image: kernel choice depends on the attention input class,
    /// not on lengths.
    fn warm_context(
        &mut self,
        prefix_cache: &mut PrefixCache,
        image: &ImageModel,
        experts: Option<&Arc<ExpertUnion>>,
        lanes: &JobQueues,
    ) -> Result<()> {
        const FEATURES: usize = 4;
        let Some(injection) = image.processing.and_then(|processing| processing.injection) else {
            return Ok(());
        };
        let spatial = image.spatial_positions();
        // A prompt token at position 0, then the block from position 1; the
        // second block starts where the first one's positions end.
        let first = injection.block(FEATURES, 1, false, Some([2, 2]))?;
        let width = first.len();
        let text = text_row(1, &[0], 0, 0, TokenSelection::Cache, spatial)?;
        let mixed = block_row(first.clone(), 1, 1, spatial, TokenSelection::LastLogits)?;
        let next = next_position(&mixed)?;
        let second = injection.block(FEATURES, next, false, Some([2, 2]))?;
        let alone = block_row(
            second.clone(),
            1,
            1 + width as u64,
            spatial,
            TokenSelection::LastLogits,
        )?;
        let lengths = [1 + width + second.len()];

        for index in 0..self.runners.len() {
            let dimensions = self.runners[index].dimensions();
            if !dimensions.embeddings {
                continue;
            }
            let (group, peer) = self.locate(index)?;
            let kind = self.kind(group)?;
            let stream = self.runners[index].stream().cloned();
            let stream = stream.as_ref();

            // Zero features, in the embedding column's dtype, stand for an
            // encoder's: their values do not choose kernels.
            let bytes = FEATURES * dimensions.hidden * dimensions.dtype.bytes();
            let memory: Arc<dyn Backing> = match self.runners[index].device() {
                Some(device) => Memory::device(device, bytes)?,
                None => Memory::host(bytes),
            };
            let features = StorageView::contiguous(
                memory,
                0,
                dimensions.dtype,
                &[FEATURES as i64, dimensions.hidden as i64],
            )?;
            // The zeroing job waits for its stream: a lane posts a job's
            // completion once it issued the job's work, so without the wait
            // the fill could still be pending when `block_on` returns.
            let zeroed = Job::new();
            // SAFETY: a new allocation, which nothing else addresses yet;
            // the `block_on` below waits for the zeroing job, whose wait for
            // the stream follows the fill, so the fill ran before the warm
            // forwards read `features` and before it is dropped.
            unsafe { features.zero(zeroed.on(stream)) }?;
            zeroed.on(stream).synchronize()?;
            lanes
                .block_on(kind, zeroed, BLOCK_ON_TIMEOUT)?
                .check()
                .map_err(|failure| failure.error)?;

            let embeddings = |token: usize| {
                [Embeddings {
                    token,
                    values: features.clone(),
                }]
            };
            let (text, mixed, alone) = (text.clone(), mixed.clone(), alone.clone());
            // The warm forwards read `features` on `stream` after their jobs
            // posted their completions. `features` is dropped only after
            // `warmup_bucket` returned, whose last job synchronizes `stream`
            // behind them (`PrefixCache::release_scratch`, through `ScratchUnits::finish`):
            // `cuStreamSynchronize` returns once the device completed every
            // task issued to the stream, so no read is pending when the drop
            // queues the allocation's `cuMemFree`, whose own implicit
            // synchronization the driver does not promise.
            self.warmup_bucket(
                lanes,
                kind,
                prefix_cache,
                &lengths,
                stream,
                |runtime, prefix_cache, _| {
                    runtime.warmup_step(
                        experts,
                        lanes,
                        (group, peer),
                        prefix_cache,
                        (
                            &[text, mixed],
                            &embeddings(1 + first.features.start),
                            ForwardMode::Prefill,
                            None,
                        ),
                        None,
                        Job::new(),
                    )?;
                    runtime.warmup_step(
                        experts,
                        lanes,
                        (group, peer),
                        prefix_cache,
                        (
                            &[alone],
                            &embeddings(second.features.start),
                            ForwardMode::Prefill,
                            None,
                        ),
                        None,
                        Job::new(),
                    )
                },
            )?;
        }
        tracing::info!("warmed text context prefills");
        Ok(())
    }

    /// One warmup step of `rows` in `mode` on microbatch `peer` of `group`,
    /// as a job of the group's lane; the group's other microbatches join.
    /// `embeddings` replace the embeddings of the rows' feature tokens. A
    /// decode with `row_sampling` samples its rows with those draws and
    /// parameters; without, it selects greedily. `capacity` forces the
    /// expert step capacity. The step continues `job`, which holds what the
    /// device does before it.
    #[expect(
        clippy::too_many_arguments,
        reason = "one warmup step's shape and owners"
    )]
    fn warmup_step(
        &mut self,
        experts: Option<&Arc<ExpertUnion>>,
        lanes: &JobQueues,
        (group, peer): (usize, usize),
        prefix_cache: &mut PrefixCache,
        (rows, embeddings, mode, row_sampling): (
            &[TokenRow],
            &[Embeddings],
            ForwardMode,
            Option<&RowSampling>,
        ),
        capacity: Option<usize>,
        job: Job,
    ) -> Result<()> {
        let prompt = PromptChunks::default();
        let request = StepRequest {
            mode,
            rows,
            finish_ids: &[],
            row_sampling,
            sampling: None,
            prompt: &prompt,
            embeddings,
        };
        let mut requests: Vec<Option<StepRequest<'_>>> =
            (0..self.groups[group].len()).map(|_| None).collect();
        requests[peer] = Some(request);

        let step = self.forward(experts, group, &requests, prefix_cache, capacity, &job)?;
        let completion = lanes.block_on(self.kind(group)?, job, BLOCK_ON_TIMEOUT)?;
        let mut completion = completion.check().map_err(|failure| failure.error)?;
        self.complete(step, &mut completion).map(drop)
    }

    /// Commit `rows` on runner `index` of `group` as a job of its lane,
    /// storing the commit graph a bucket's first commit captures.
    fn warmup_commit(
        &mut self,
        lanes: &JobQueues,
        index: usize,
        group: usize,
        rows: &[[i64; 8]],
    ) -> Result<()> {
        let job = Job::new();
        let captured = self.runners[index].commit(rows, &job)?;
        let completion = lanes.block_on(self.kind(group)?, job, BLOCK_ON_TIMEOUT)?;
        let mut completion = completion.check().map_err(|failure| failure.error)?;
        match captured {
            Some(key) => self.runners[index].complete_commit(key, &mut completion),
            None => Ok(()),
        }
    }
}
