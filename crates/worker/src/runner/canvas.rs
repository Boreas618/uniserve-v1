//! `CanvasRunner`: token-denoising readouts and canvas steps of one lane.
//!
//! A readout evaluates canvas rows over their requests' cached prefixes and
//! reads candidate log-probabilities at answer slots; a canvas step denoises
//! each row's resident canvas by one block-diffusion step (see
//! `uniserve_models/interface/canvas.py`). The runner owns the lane's fixed
//! buffers, its graph table and its binding; it writes host metadata into
//! the buffers, gathers resident bank rows natively, pads graph buckets and
//! decides which graph an invocation replays.
//!
//! With graph pools, startup captures (largest first):
//!
//! - a readout graph per row and length bucket: the pass up to the final
//!   layer's attention over the bucket's tokens in up to twice as many
//!   sequences ([`shapes::readout_padding`]);
//! - on the final pipeline stage of a local tail, a readout tail graph per
//!   slot bucket: the rest of the final layer, the output norm, the head and
//!   the log-softmax over the slot rows alone; a call with more slots
//!   replays them in chunks;
//! - on generating ranks, a step graph per row bucket and first-step flag:
//!   the whole step, from the pass through the sampler and the commit of
//!   the stepped state to the banks.
//!
//! Without pools (CPU lanes, graph policy off) every invocation runs
//! eagerly. A tail that exchanges tokens with expert peers always runs
//! eagerly, once over every slot of the invocation.
//!
//! With expert-parallel layers, a readout (its pass and an exchanging
//! tail) and a step are each one expert step: the caller's coordinator
//! ([`Coordinate`]) turns the invocation's token extent into the step's
//! agreed capacity, and graphs are captured and replayed per capacity.

pub(crate) mod inputs;
pub(crate) mod shapes;

use std::ops::Range;
use std::sync::Arc;

use uniserve_core::CanvasSampling;
use uniserve_cuda::Stream;

use super::graphs::{GraphMode, GraphTable, Lookup};
use crate::ffi::Handle;
use crate::model::canvas::{Attention, CanvasFacts, CanvasModel, Capture};
use crate::storage::canvas::{CanvasBanks, CanvasBuffers, ColumnWrites};
use crate::storage::{GroupShape, GroupTable, TablePages, table_pages};
use crate::{Error, Result};

pub(crate) use inputs::{CanvasRow, ReadoutPass, StepRow};
pub(crate) use shapes::{CanvasGraphKey, CanvasShapes, SLOT_BUCKETS, step_rows};

/// Turns an invocation's expert token extent into its expert step's
/// capacity; `None` without expert-parallel layers.
pub(crate) type Coordinate<'a> = &'a mut dyn FnMut(usize) -> Result<Option<usize>>;

/// Candidates selected and read back per readout pass of the bound
/// candidate buffers.
pub(crate) const CANDIDATES: usize = 1 << 16;

/// What one lane's canvas runner is built from.
pub(crate) struct CanvasLane {
    /// Names the runner in errors and graph tables.
    pub(crate) owner: String,
    pub(crate) model: CanvasModel,
    pub(crate) facts: Arc<CanvasFacts>,
    pub(crate) buffers: CanvasBuffers,
    /// The resident banks of a generating rank.
    pub(crate) banks: Option<Arc<CanvasBanks>>,
    /// The block-diffusion sampling the deployment serves.
    pub(crate) sampling: Option<CanvasSampling>,
    pub(crate) shapes: CanvasShapes,
    /// Page columns every table's rows hold in a graph.
    pub(crate) graph_widths: Vec<usize>,
    /// The cache groups of the numerical tables, in table order.
    pub(crate) groups: Vec<GroupShape>,
    /// The allocator pool graphs are captured into; `None` runs eagerly.
    pub(crate) pool: Option<Handle>,
    /// The lane stream; `None` for a CPU lane.
    pub(crate) stream: Option<Stream>,
}

/// The runner of one canvas lane.
pub(crate) struct CanvasRunner {
    lane: CanvasLane,
    graphs: GraphTable<CanvasGraphKey>,
}

/// How one invocation ran and the candidates it produced.
pub(crate) struct ReadoutRun {
    pub(crate) mode: GraphMode,
    pub(crate) tokens: usize,
    pub(crate) padded_tokens: usize,
}

/// How one step invocation ran.
pub(crate) struct StepRun {
    pub(crate) mode: GraphMode,
    pub(crate) rows: usize,
    pub(crate) padded_rows: usize,
}

impl CanvasRunner {
    pub(crate) fn new(lane: CanvasLane) -> Self {
        let graphs = GraphTable::new(lane.owner.clone(), lane.model.release());
        Self { lane, graphs }
    }

    /// The runner's name in graph tables, kernel records and errors.
    pub(crate) fn owner(&self) -> &str {
        &self.lane.owner
    }

    pub(crate) fn facts(&self) -> &CanvasFacts {
        &self.lane.facts
    }

    pub(crate) fn shapes(&self) -> &CanvasShapes {
        &self.lane.shapes
    }

    pub(crate) fn buffers(&self) -> &CanvasBuffers {
        &self.lane.buffers
    }

    pub(crate) fn stream(&self) -> Option<&Stream> {
        self.lane.stream.as_ref()
    }

    pub(crate) fn model(&self) -> &CanvasModel {
        &self.lane.model
    }

    /// The served sampling of a generating lane.
    pub(crate) fn sampling(&self) -> Option<&CanvasSampling> {
        self.lane
            .sampling
            .as_ref()
            .filter(|_| self.lane.banks.is_some())
    }

    /// Captured graphs by kind: readout passes, readout tails and steps.
    pub(crate) fn captured(&self) -> [usize; 3] {
        let mut counts = [0; 3];
        for (key, _) in self.graphs.keys() {
            counts[match key {
                CanvasGraphKey::Readout { .. } => 0,
                CanvasGraphKey::Tail { .. } => 1,
                CanvasGraphKey::Step { .. } => 2,
            }] += 1;
        }
        counts
    }

    /// Refuse further captures; serving may only replay.
    pub(crate) fn seal(&mut self) {
        self.graphs.seal();
    }

    /// Release every graph; the lane stream has drained.
    pub(crate) fn close(&mut self, aborted: bool) -> Result<()> {
        self.graphs.close(aborted)
    }

    fn device(&self) -> Option<i32> {
        self.lane.buffers.device()
    }

    fn submit(&self, writes: ColumnWrites) -> Result<()> {
        writes.submit(self.device(), self.lane.stream.as_ref())
    }

    /// Run `capture` as the graph of `key` at expert `capacity`, or replay
    /// the graph resident for them; without pools run `eager`.
    fn run(
        &mut self,
        key: CanvasGraphKey,
        capacity: Option<usize>,
        capture: Capture<'_>,
        attention: Option<&Attention<'_>>,
        eager: impl FnOnce(&CanvasModel) -> Result<()>,
    ) -> Result<GraphMode> {
        let Some(pool) = &self.lane.pool else {
            eager(&self.lane.model)?;
            self.graphs.record(GraphMode::Eager);
            return Ok(GraphMode::Eager);
        };
        let mode = match self.graphs.lookup(&key, capacity)? {
            Lookup::Replay(graph) => {
                self.lane.model.replay(graph, attention)?;
                GraphMode::Replay
            }
            Lookup::Capture => {
                let graph = self
                    .lane
                    .model
                    .capture(capture, pool, capacity.unwrap_or(0))?;
                self.graphs.insert(key, capacity, graph)?;
                GraphMode::Capture
            }
        };
        self.graphs.record(mode);
        Ok(mode)
    }

    fn tail(&mut self, slots: usize) -> Result<Handle> {
        let key = CanvasGraphKey::Tail { slots };
        let Some(pool) = &self.lane.pool else {
            return Err(Error::Invariant(
                "readout tails replay only with graph pools".into(),
            ));
        };
        if let Lookup::Capture = self.graphs.lookup(&key, None)? {
            // A local tail exchanges no tokens.
            let graph = self.lane.model.capture(Capture::Tail { slots }, pool, 0)?;
            self.graphs.insert(key, None, graph)?;
            self.graphs.record(GraphMode::Capture);
        } else {
            self.lane.model.replay(self.graph(&key)?, None)?;
            self.graphs.record(GraphMode::Replay);
        }
        Ok(self.graph(&key)?.clone())
    }

    fn graph(&self, key: &CanvasGraphKey) -> Result<&Handle> {
        match self.graphs.lookup(key, None)? {
            Lookup::Replay(graph) => Ok(graph),
            Lookup::Capture => Err(Error::Invariant(format!("{key:?} was not captured"))),
        }
    }

    /// Evaluate a readout pass and read every candidate's log-probability.
    ///
    /// Candidates are read back in passes of at most [`CANDIDATES`]: after
    /// each pass, `sink(range, values)` receives the FP32 values of the
    /// candidates `range` (in row, slot and candidate order) in the bound
    /// candidate buffer, which the next pass overwrites in stream order.
    /// Every pipeline stage receives the same values.
    pub(crate) fn readout(
        &mut self,
        pass: &ReadoutPass<'_>,
        coordinate: Coordinate<'_>,
        mut sink: impl FnMut(Range<usize>, &crate::storage::memory::StorageView) -> Result<()>,
    ) -> Result<ReadoutRun> {
        let facts = Arc::clone(&self.lane.facts);
        let layout = self.lane.buffers.layout().clone();
        let tokens = pass.tokens.len();
        if pass.rows.is_empty() || pass.rows.iter().map(|row| row.query).sum::<usize>() != tokens {
            return Err(Error::Invalid(
                "readout rows must partition their canvas tokens".into(),
            ));
        }
        if pass.slots.is_empty()
            || pass.slots.len() > layout.tokens
            || pass.slot_candidates.len() != pass.slots.len() + 1
        {
            return Err(Error::Invalid(
                "a readout reads between one slot and its token capacity".into(),
            ));
        }
        // Every stage refuses the same calls, before any collective.
        let vocab = facts.vocab_size as i64;
        if pass.candidates.iter().any(|&id| id < 0 || id >= vocab) {
            return Err(Error::Invalid(format!(
                "a readout candidate is outside the {vocab}-token vocabulary"
            )));
        }

        // The canvas pass: the smallest bucket holding the rows, or eagerly.
        let graph = self.lane.pool.is_some();
        let (bucket, padding, widths) = if graph {
            let (rows, length) = self.lane.shapes.readout(pass.rows.len(), pass.longest())?;
            (
                CanvasGraphKey::Readout { rows, length },
                shapes::readout_padding(pass.rows.len(), tokens, rows, length)?,
                self.lane.graph_widths.clone(),
            )
        } else {
            (
                CanvasGraphKey::Readout {
                    rows: pass.rows.len(),
                    length: pass.longest(),
                },
                Vec::new(),
                pass.pages.iter().map(TablePages::width).collect(),
            )
        };
        // The tokens the pass evaluates: the bucket's, or the live ones.
        let padded_tokens = if graph {
            bucket.tokens(facts.canvas_length)
        } else {
            tokens
        };
        // Every sequence's query tokens fit the bucket length, which sizes
        // the captured attention launch.
        let CanvasGraphKey::Readout {
            length: visible, ..
        } = bucket
        else {
            return Err(Error::Invariant("a readout runs a readout graph".into()));
        };
        if padded_tokens > layout.tokens {
            return Err(Error::Invalid(format!(
                "a readout of {padded_tokens} tokens exceeds its {} token capacity",
                layout.tokens
            )));
        }
        let mut writes = ColumnWrites::new();
        let ids = self.lane.buffers.view("input_ids")?;
        writes.write(&ids, 0, &pass.tokens)?;
        writes.zero(&ids, tokens, padded_tokens - tokens)?;
        inputs::write_positions(&self.lane.buffers, &mut writes, &pass.rows, padded_tokens)?;
        let metadata = inputs::write_attention(
            &self.lane.buffers,
            &mut writes,
            &pass.rows,
            &padding,
            pass.pages,
            &widths,
        )?;
        writes.write(&self.lane.buffers.view("slot_tokens")?, 0, &pass.slots)?;
        self.submit(writes)?;

        let attention = Attention {
            rows: metadata.rows(),
            visible,
            queries: &metadata.queries,
            prefixes: &metadata.prefixes,
            widths: &metadata.widths,
            starts: &metadata.starts,
        };
        // The expert step carries the pass's tokens and, through an
        // exchanging tail, the slot rows.
        let extent = if facts.local_tail {
            padded_tokens
        } else {
            padded_tokens.max(pass.slots.len())
        };
        let capacity = coordinate(extent)?;
        let expert_capacity = capacity.unwrap_or(0);
        let mode = self.run(
            bucket,
            capacity,
            Capture::Attend {
                tokens: padded_tokens,
                attention,
            },
            Some(&attention),
            |model| model.readout_attend(padded_tokens, &attention, expert_capacity),
        )?;

        // The tail: replayed slot chunks on a local final stage with graphs,
        // otherwise one eager evaluation over every slot.
        let chunked = graph && facts.local_tail && facts.last_stage();
        let eager = if facts.last_stage() && !chunked {
            Some(
                self.lane
                    .model
                    .readout_finish(pass.slots.len(), expert_capacity)?,
            )
        } else {
            None
        };
        let values = self.lane.buffers.view("candidate_logprobs")?;
        let index = self.lane.buffers.view("candidate_index")?;
        for (start, live, bucket) in shapes::tail_chunks(pass.slots.len()) {
            let tail = if chunked {
                let mut writes = ColumnWrites::new();
                let rows = self.lane.buffers.view("tail_rows")?;
                writes.write(&rows, 0, &pass.slots[start..start + live])?;
                writes.zero(&rows, live, bucket - live)?;
                self.submit(writes)?;
                Some(self.tail(bucket)?)
            } else {
                eager.clone()
            };
            // The flat index of candidate `id` at slot `slot` in the tail's
            // `[slots, vocab]` log-probabilities.
            let base = if chunked { start } else { 0 };
            let first = pass.slot_candidates[start];
            let last = pass.slot_candidates[start + live];
            let mut slot = start;
            for chunk in (first..last).step_by(CANDIDATES) {
                let end = (chunk + CANDIDATES).min(last);
                if let Some(tail) = &tail {
                    let mut flat = Vec::with_capacity(end - chunk);
                    for candidate in chunk..end {
                        while pass.slot_candidates[slot + 1] <= candidate {
                            slot += 1;
                        }
                        flat.push((slot - base) as i64 * vocab + pass.candidates[candidate]);
                    }
                    let mut writes = ColumnWrites::new();
                    writes.write(&index, 0, &flat)?;
                    self.submit(writes)?;
                    self.lane.model.readout_candidates(tail, 0, end - chunk)?;
                }
                if facts.pipeline_size > 1 {
                    self.lane.model.readout_broadcast(end - chunk)?;
                }
                sink(chunk..end, &values.prefix((end - chunk) as i64)?)?;
            }
        }
        Ok(ReadoutRun {
            mode,
            tokens,
            padded_tokens,
        })
    }

    /// Step the resident canvases of `rows` once.
    ///
    /// Returns how the step ran; the rows' int64 `[1 + canvas]` results are
    /// the leading rows of [`Self::step_results`].
    pub(crate) fn step(
        &mut self,
        rows: &[StepRow],
        pages: &[TablePages],
        coordinate: Coordinate<'_>,
    ) -> Result<StepRun> {
        let banks = self
            .lane
            .banks
            .clone()
            .ok_or_else(|| Error::Invalid("this worker keeps no generating canvases".into()))?;
        let step = self
            .lane
            .buffers
            .step()
            .ok_or_else(|| Error::Invariant("a generating lane has step buffers".into()))?;
        let length = self.lane.facts.canvas_length;
        let live = rows.len();
        let graph = self.lane.pool.is_some();
        let padded = if graph {
            self.lane.shapes.step(live)?
        } else {
            live
        };
        if live == 0 || padded > step.rows {
            return Err(Error::Invalid(format!(
                "a canvas step of {live} rows exceeds its {} step rows",
                step.rows
            )));
        }

        let mut writes = ColumnWrites::new();
        let columns = self.lane.buffers.view("step.columns")?;
        let fields: [Vec<i64>; 4] = [
            rows.iter().map(|row| i64::from(row.slot)).collect(),
            // Seeds are u64 carried bit for bit in int64 columns.
            rows.iter().map(|row| row.seed as i64).collect(),
            rows.iter().map(|row| i64::from(row.block)).collect(),
            rows.iter().map(|row| i64::from(row.step)).collect(),
        ];
        for (field, values) in fields.iter().enumerate() {
            // A padding row starts a canvas of seed and block zero in the
            // sentinel slot zero, which its commit skips.
            writes.write(&columns, field * step.rows, values)?;
            writes.zero(&columns, field * step.rows + live, padded - live)?;
        }
        let canvases: Vec<CanvasRow> = rows
            .iter()
            .map(|row| CanvasRow {
                query: length,
                prefix: row.prefix,
                position: row.position,
            })
            .collect();
        inputs::write_positions(&self.lane.buffers, &mut writes, &canvases, padded * length)?;
        let widths: Vec<usize> = if graph {
            self.lane.graph_widths.clone()
        } else {
            pages.iter().map(TablePages::width).collect()
        };
        let metadata = inputs::write_attention(
            &self.lane.buffers,
            &mut writes,
            &canvases,
            &vec![length; padded - live],
            pages,
            &widths,
        )?;
        self.submit(writes)?;
        let slots: Vec<u32> = rows.iter().map(|row| row.slot).collect();
        banks.gather(&slots, padded, step, self.lane.stream.as_ref())?;

        let first = rows.iter().all(|row| row.step == 0);
        let attention = Attention {
            rows: padded,
            visible: length,
            queries: &metadata.queries,
            prefixes: &metadata.prefixes,
            widths: &metadata.widths,
            starts: &metadata.starts,
        };
        let capacity = coordinate(padded * length)?;
        let mode = self.run(
            CanvasGraphKey::Step {
                rows: padded,
                first,
            },
            capacity,
            Capture::Step { first, attention },
            Some(&attention),
            |model| model.step(first, &attention, capacity.unwrap_or(0)),
        )?;
        Ok(StepRun {
            mode,
            rows: live,
            padded_rows: padded,
        })
    }

    /// The int64 `[rows, 1 + canvas]` results of the last step's rows.
    pub(crate) fn step_results(&self, rows: usize) -> Result<crate::storage::memory::StorageView> {
        self.lane.buffers.view("step.results")?.prefix(rows as i64)
    }

    /// Prepare every invocation shape serving can run, over scratch
    /// prefixes: row `r` of a synthetic invocation reads a one-token prefix
    /// from `scratch(r)`, one unit per numerical table. Steps run in slots
    /// `1..`; their synthetic state is harmless because a request's first
    /// step starts its canvas before reading any of it.
    ///
    /// With graph pools, every configured graph is captured, largest first;
    /// each capture runs its function eagerly first. Without pools, a
    /// generating CUDA lane runs one eager step of every sampler chunk
    /// shape, whose first self-conditioning product chooses its algorithm
    /// and synchronizes the device, so serving never does.
    ///
    /// With expert-parallel layers, `variants(extent)` lists the expert
    /// capacities an invocation of that token extent is captured at
    /// (largest first) and `coordinate(extent, capacity)` runs its warmup
    /// step at one of them; every rank warms the same sequence.
    pub(crate) fn warm(
        &mut self,
        scratch: &dyn Fn(usize) -> Vec<u32>,
        variants: &dyn Fn(usize) -> Vec<Option<usize>>,
        coordinate: &mut dyn FnMut(usize, Option<usize>) -> Result<Option<usize>>,
    ) -> Result<()> {
        if self.lane.pool.is_none() {
            let chunk = match (&self.lane.stream, self.lane.buffers.layout().step) {
                (Some(_), Some(step)) if self.sampling().is_some() => step.chunk_rows,
                _ => return Ok(()),
            };
            for count in 1..=chunk {
                let pages = self.scratch_pages(count, self.lane.facts.canvas_length, scratch)?;
                self.step(&synthetic_steps(count, true), &pages, &mut |extent| {
                    coordinate(extent, None)
                })?;
            }
            return self.synchronize();
        }
        let facts = Arc::clone(&self.lane.facts);
        let rows: Vec<usize> = self.lane.shapes.rows().iter().rev().copied().collect();
        let lengths: Vec<usize> = self.lane.shapes.lengths().iter().rev().copied().collect();
        let mut tails = facts.local_tail && facts.last_stage();
        for &count in &rows {
            for &length in &lengths {
                let pages = self.scratch_pages(count, length, scratch)?;
                let pass = ReadoutPass {
                    rows: vec![
                        CanvasRow {
                            query: length,
                            prefix: 1,
                            position: 1,
                        };
                        count
                    ],
                    tokens: vec![0; count * length],
                    slots: (0..count).map(|row| (row * length) as i64).collect(),
                    slot_candidates: (0..=count).collect(),
                    candidates: vec![0; count],
                    pages: &pages,
                };
                if tails {
                    // The first readout capture also captures every tail.
                    for slots in SLOT_BUCKETS {
                        self.capture_tail(slots)?;
                    }
                    tails = false;
                }
                let (rows, length) = self.lane.shapes.readout(count, length)?;
                let tokens = CanvasGraphKey::Readout { rows, length }.tokens(facts.canvas_length);
                let extent = if facts.local_tail {
                    tokens
                } else {
                    tokens.max(count)
                };
                for capacity in variants(extent) {
                    self.readout(&pass, &mut |extent| coordinate(extent, capacity), |_, _| {
                        Ok(())
                    })?;
                }
            }
        }
        if self.sampling().is_some() {
            for &count in &rows {
                let pages = self.scratch_pages(count, facts.canvas_length, scratch)?;
                let padded = self.lane.shapes.step(count)?;
                for first in [true, false] {
                    for capacity in variants(padded * facts.canvas_length) {
                        self.step(&synthetic_steps(count, first), &pages, &mut |extent| {
                            coordinate(extent, capacity)
                        })?;
                    }
                }
            }
        }
        self.synchronize()
    }

    fn synchronize(&self) -> Result<()> {
        if let Some(stream) = &self.lane.stream {
            stream.synchronize()?;
        }
        Ok(())
    }

    fn capture_tail(&mut self, slots: usize) -> Result<()> {
        let mut writes = ColumnWrites::new();
        let rows = self.lane.buffers.view("tail_rows")?;
        writes.write(&rows, 0, &(0..slots as i64).collect::<Vec<_>>())?;
        self.submit(writes)?;
        self.tail(slots).map(drop)
    }

    /// Pages of `count` rows of `length` canvas tokens reading a one-token
    /// prefix on scratch units.
    fn scratch_pages(
        &self,
        count: usize,
        length: usize,
        scratch: &dyn Fn(usize) -> Vec<u32>,
    ) -> Result<Vec<TablePages>> {
        let tables = (0..count)
            .map(|row| {
                let units = scratch(row);
                let mut next = units.into_iter();
                self.lane
                    .groups
                    .iter()
                    .map(|&shape| {
                        let units = (&mut next)
                            .take(shape.units_per_page as usize)
                            .map(uniserve_core::UnitId)
                            .collect();
                        Arc::new(GroupTable {
                            shape,
                            start_page: 0,
                            units,
                            allocated_tokens: shape.page_tokens,
                        })
                    })
                    .collect()
            })
            .collect::<Vec<Vec<_>>>();
        table_pages(&tables, &vec![1; count], &vec![length as u32; count])
    }
}

/// Canvas steps of slots `1..=count` with a one-token prefix, at step zero
/// (`first`) or one.
fn synthetic_steps(count: usize, first: bool) -> Vec<StepRow> {
    (0..count)
        .map(|row| StepRow {
            slot: row as u32 + 1,
            seed: row as u64 + 1,
            block: 0,
            step: u32::from(!first),
            prefix: 1,
            position: 1,
        })
        .collect()
}
