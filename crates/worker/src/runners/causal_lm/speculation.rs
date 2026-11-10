//! Speculative decoding with the model's own drafter: the eager step a speculating deployment
//! serves every token batch with.
//!
//! A deployment that speculates (`speculative_config.num_draft_tokens` above zero) keeps the draft
//! cache in its pool and serves each batch in one eager step of four stages, separated by the two
//! host readbacks the step needs:
//!
//! ```text
//! target    the model over the batch: a prompt batch as it is; a verification's tokens each as a
//!           one-token attention row (a decode step's shape at its position) while one state row
//!           per call walks them in order and keeps the recurrent state after every token (the
//!           trail)
//! sample    every sampled position through the sampling module: a verification samples each of
//!           its tokens, with the drafts before it counted in the row's penalties
//!                                         ── readback: the sampled tokens ──
//! accept    per verification, the longest prefix of drafts its own samples agree with, then the
//!           sample after it; settle each slot's recurrent state on the trail's entry of its last
//!           accepted token; commit the accepted tokens into the slots' state (`slot_commit`)
//! draft     the drafter's entries over the positions whose next token is now known, then the
//!           drafts each call proposes, chained step by step on the device
//!                                         ── readback: the proposals, the slots' positions ──
//! ```
//!
//! ## Why the acceptance keeps the target distribution
//!
//! The drafter proposes deterministically (the argmax of its logits), so the draft distribution
//! `q` is a point mass at the draft `d`. The rejection rule (Leviathan et al. 2023; vLLM's and
//! SGLang's rejection samplers) accepts `d` with probability `min(1, p(d)/q(d)) = p(d)` and on
//! rejection draws from `norm(max(0, p - q))`, which is `p` with `d` removed. Drawing `x ~ p` and
//! accepting exactly when `x = d` has the same law: `P(x = d) = p(d)`, and given `x ≠ d`, `x` is
//! distributed as `p` without `d`. Each position draws with the uniform its request's seed and the
//! position address (`ipc::philox`), the draw a decode at that position makes, so a speculated run
//! of a seeded request reproduces itself, and a greedy request's verification takes the argmax a
//! decode would: its accepted tokens are the tokens it generates without speculation, up to the
//! decode kernels rounding a verification's rows in another order than one decode row.
//!
//! ## What is the model's and what is the runner's
//!
//! The model computes its forward and its drafter's ([`nn::family::Drafter`]); which positions an
//! entry covers, the draft cache's pages, the acceptance, the commit and the state's settling are
//! this runner's, as `AGENTS.md`'s Model Library Boundary assigns them.

use std::ops::Range;
use std::time::Instant;

use ipc::execution::{BatchOutput, CallOutput, RowKind, RowState};
use ipc::philox::DrawAddress;
use ipc::{DraftWork, TimingCounters};
use nn::family::TextModel;
use nn::{Forward, KvView, TokenBatch};
use ops::{DType, Shape, TensorView};
use platform::DeviceId;
use proto::program::PhaseKind;
use proto::program::commit::{COMMIT_FIELDS, column, mode};
use proto::program::sampler::{PARAMETER_COLUMNS, TEMPERATURE};

use super::{ATTENTION_WORKSPACE_BYTES, CausalLmRunner, LOGITS_DTYPE};
use crate::runner::Runner;
use crate::runners::columns::{Column, encode_f32, encode_i32};
use crate::runners::recording::{Footprint, Walk};
use crate::runners::sampling::{SamplerState, Samples, commit_block, commit_descriptor};
use crate::runners::state::StateRows;
use crate::{Error, Result};

/// Sampling parameter columns of one row.
const COLUMNS: usize = PARAMETER_COLUMNS as usize;

/// The draft cache's geometry in the pool, read from the cache plan when the runner binds.
#[derive(Clone, Copy, Debug)]
struct DraftPool {
    /// The draft group's index among the cache groups.
    group: u32,
    /// K/V pages of `page` tokens one draft page holds (`CachePlan::draft_span`).
    span: u32,
    /// Tokens of one K/V page, which the drafter's attention reads a draft page in.
    page: u32,
    /// Units of the pool: a draft layer's page `c · units + u` is run `c` of the draft page in unit
    /// `u`.
    units: u32,
}

impl DraftPool {
    /// The attention page of position `position` in a draft table of `first` page and `units`, and
    /// whether the table holds a unit there.
    fn page(&self, first: u32, units: &[u32], position: u32) -> (i32, bool) {
        let tokens = self.page * self.span;
        let draft_page = position / tokens;
        let run = (position % tokens) / self.page;
        let unit = draft_page
            .checked_sub(first)
            .and_then(|held| units.get(held as usize).copied())
            .unwrap_or(0);
        ((run * self.units + unit) as i32, unit != 0)
    }

    /// The cache slot the drafter writes position `position` of a table at, `-1` where the table
    /// holds no unit: the entry is then not written, and the draft cache does not count it.
    fn slot(&self, first: u32, units: &[u32], position: u32) -> i32 {
        match self.page(first, units, position) {
            (page, true) => page * self.page as i32 + (position % self.page) as i32,
            _ => -1,
        }
    }

    /// One row of the drafter's block table: the attention page of every page of `width`.
    fn row(&self, first: u32, units: &[u32], width: u32, out: &mut Vec<i32>) {
        for page in 0..width {
            out.push(self.page(first, units, page * self.page).0);
        }
    }
}

/// One call of a step, as the step evaluates it.
#[derive(Clone, Debug, Default)]
struct CallPlan {
    /// The call's request slot.
    slot: i32,
    /// What the call computes.
    kind: Option<RowKind>,
    /// The call's rows among the staged batch's.
    rows: Range<usize>,
    /// The call's tokens among the batch's.
    tokens: Range<usize>,
    /// The position of its first token.
    start: u32,
    /// The drafts a verification verifies: its tokens after its first.
    drafts: Vec<i32>,
    /// Its rows of the sampling step.
    samples: Range<usize>,
    /// What it does for its request's draft cache.
    draft: DraftWork,
}

/// One row of the sampling step.
#[derive(Clone, Debug)]
struct SampleRow {
    /// The token of the batch whose hidden state it projects.
    hidden: i32,
    slot: i32,
    parameters: [f32; COLUMNS],
    draw: DrawAddress,
    /// The tokens before it its penalties count beyond the slot's history: a verification's drafts
    /// ahead of the position, `-1` elsewhere; `drafts` of them.
    extras: Vec<i32>,
}

/// One step's plan, read from the staged batch.
#[derive(Debug, Default)]
struct Plan {
    /// Whether the batch verifies: its tokens attend as one-token rows of the decode phase.
    verify: bool,
    calls: Vec<CallPlan>,
    samples: Vec<SampleRow>,
    /// The batch's tokens.
    tokens: usize,
}

/// What a call accepted: its committed tokens and their log-probabilities.
#[derive(Clone, Debug, Default)]
struct Accepted {
    tokens: Vec<u32>,
    logprobs: Vec<f32>,
}

/// The longest prefix of `drafts` the samples `sampled` agree with, then the sample after it: the
/// tokens a verification of `drafts` commits, given that `sampled[j]` is the token drawn at the
/// draft `j`'s position (`sampled` is one longer than `drafts`).
fn accepted_prefix(drafts: &[i32], sampled: &[u32]) -> usize {
    let agreed = drafts
        .iter()
        .zip(sampled)
        .take_while(|(draft, token)| u32::try_from(**draft).ok() == Some(**token))
        .count();
    (agreed + 1).min(sampled.len())
}

/// The host and device columns of one eager step, sized for the largest batch the runner serves.
///
/// They are pinned host memory the device reads in place: every stage writes its columns before it
/// launches and the step synchronizes before the next stage writes, so no column is rewritten while
/// a launch may read it.
pub(crate) struct Speculation {
    /// The most drafts one verification carries.
    drafts: u32,
    pool: Option<DraftPool>,
    /// The most tokens a verification holds: every slot's `drafts + 1`, each attending as a row.
    verify_tokens: u64,
    /// The most tokens and rows a step's columns hold.
    tokens: u64,
    rows: u64,
    /// The pages one row of the drafter's block table holds.
    width: u32,
    vocabulary: u32,
    // The target forward's attention rows and state rows.
    positions: Column,
    ids: Column,
    slot_mapping: Column,
    seq_lens: Column,
    query_offsets: Column,
    logits_rows: Column,
    state_offsets: Column,
    state_lens: Column,
    state_slots: Column,
    /// `[3]` the token counts of the target, the drafter's entries and its chained steps.
    live: Column,
    // The sampling step.
    indices: Column,
    parameters: Column,
    sample_slots: Column,
    draws: Column,
    extras: Column,
    greedy: Column,
    greedy_logprobs: Column,
    drawn: Column,
    drawn_logprobs: Column,
    // The commit.
    table: Column,
    commit_tokens: Column,
    points: Column,
    flags: Column,
    continuation: Column,
    capacity: u64,
    // The drafter.
    draft_positions: Column,
    draft_ids: Column,
    draft_slots: Column,
    draft_lens: Column,
    draft_offsets: Column,
    draft_hidden: Column,
    draft_tables: Column,
    /// `[drafts, rows]` each chained step's positions, cache slots and lengths.
    chained_positions: Column,
    chained_slots: Column,
    chained_lens: Column,
    chained_offsets: Column,
    chained_tables: Column,
    /// The rows whose drafter output each call's first proposal reads.
    last_entries: Column,
    /// `[drafts, rows]` the proposals, one step per run.
    proposals: Column,
    encoded: Vec<u8>,
}

impl Speculation {
    /// The columns of a runner on `device` whose batches hold at most `tokens` tokens in `rows`
    /// rows, over `slots` request slots, verifying at most `drafts` drafts per call, with a
    /// `vocabulary`-wide head and sequences of at most `max_kv_len` keys in pages of `page` tokens.
    #[allow(
        clippy::too_many_arguments,
        reason = "the device, the batch bounds, the slots, the draft limit and the model's widths"
    )]
    pub(crate) fn new(
        device: DeviceId,
        tokens: u64,
        rows: u64,
        slots: u32,
        drafts: u32,
        vocabulary: u32,
        max_kv_len: u32,
        page: u32,
    ) -> Result<Self> {
        if drafts == 0 {
            return Err(Error::Invalid(
                "a deployment that speculates verifies at least one draft".into(),
            ));
        }
        // A verification of every slot's `drafts + 1` tokens attends as one row per token.
        let verify_tokens = u64::from(slots) * u64::from(drafts + 1);
        let tokens = tokens.max(verify_tokens);
        let rows = rows.max(u64::from(slots)).max(verify_tokens);
        let width = max_kv_len.div_ceil(page.max(1));
        let i32s = |count: u64| Column::new(device, DType::I32, count.max(1));
        let f32s = |count: u64| Column::new(device, DType::F32, count.max(1));
        let capacity = rows + tokens;
        let chained = u64::from(drafts) * pitch(rows);
        Ok(Self {
            drafts,
            pool: None,
            verify_tokens,
            tokens,
            rows,
            width,
            vocabulary,
            positions: i32s(tokens)?,
            ids: i32s(tokens)?,
            slot_mapping: i32s(tokens)?,
            seq_lens: i32s(rows)?,
            query_offsets: i32s(rows + 1)?,
            logits_rows: i32s(rows)?,
            state_offsets: i32s(rows + 1)?,
            state_lens: i32s(rows)?,
            state_slots: i32s(rows)?,
            live: i32s(3 * pitch(1))?,
            indices: i32s(tokens)?,
            parameters: f32s(tokens * COLUMNS as u64)?,
            sample_slots: i32s(tokens)?,
            draws: f32s(tokens)?,
            extras: i32s(tokens * u64::from(drafts))?,
            greedy: i32s(tokens)?,
            greedy_logprobs: f32s(tokens)?,
            drawn: i32s(tokens)?,
            drawn_logprobs: f32s(tokens)?,
            table: i32s(COMMIT_FIELDS as u64 * capacity)?,
            commit_tokens: i32s(capacity)?,
            points: i32s(capacity)?,
            flags: i32s(capacity)?,
            continuation: Column::new(device, DType::U8, capacity)?,
            capacity,
            draft_positions: i32s(tokens)?,
            draft_ids: i32s(tokens)?,
            draft_slots: i32s(tokens)?,
            draft_lens: i32s(rows)?,
            draft_offsets: i32s(rows + 1)?,
            draft_hidden: i32s(tokens)?,
            draft_tables: i32s(rows * u64::from(width))?,
            chained_positions: i32s(chained)?,
            chained_slots: i32s(chained)?,
            chained_lens: i32s(chained)?,
            chained_offsets: i32s(rows + 1)?,
            chained_tables: i32s(rows * u64::from(width))?,
            last_entries: i32s(rows)?,
            proposals: i32s(chained)?,
            encoded: Vec::new(),
        })
    }

    /// The most tokens a verification batch holds: the rows its recurrent state's trail keeps.
    pub(crate) fn trail_tokens(&self) -> u32 {
        u32::try_from(self.verify_tokens).unwrap_or(u32::MAX)
    }

    /// Reads the draft cache's geometry from the cache plan.
    pub(crate) fn bind(&mut self, cache: &crate::memory::prefix_cache::PrefixCache) -> Result<()> {
        let plan = cache.plan();
        let group = plan.draft_group().ok_or_else(|| {
            Error::Invalid(
                "the deployment speculates and its cache keeps no draft group: the model's drafter \
                 states no cache"
                    .into(),
            )
        })?;
        let span = plan
            .draft_span()
            .ok_or_else(|| Error::Invariant("a draft group lies over no column".into()))?;
        let page = plan.groups[group].block_size / span;
        self.pool = Some(DraftPool {
            group: group as u32,
            span,
            page,
            units: cache.num_units(),
        });
        Ok(())
    }

    fn write_i32(column: &mut Column, encoded: &mut Vec<u8>, values: &[i32]) -> Result<()> {
        encode_i32(encoded, values);
        column.write(encoded)
    }

    fn write_f32(column: &mut Column, encoded: &mut Vec<u8>, values: &[f32]) -> Result<()> {
        encode_f32(encoded, values);
        column.write(encoded)
    }

    /// The values the device wrote into the first `count` elements of an i32 column.
    fn read_i32(column: &Column, count: usize) -> Vec<i32> {
        // SAFETY: the column's host side holds at least `count` i32, the stream has completed the
        // launches that wrote them, and nothing writes the column while it is read.
        unsafe { std::slice::from_raw_parts(column.host.as_ptr().cast::<i32>(), count).to_vec() }
    }

    /// The values the device wrote into the first `count` elements of an f32 column.
    fn read_f32(column: &Column, count: usize) -> Vec<f32> {
        // SAFETY: as `read_i32`'s.
        unsafe { std::slice::from_raw_parts(column.host.as_ptr().cast::<f32>(), count).to_vec() }
    }
}

impl Plan {
    /// The plan of the staged batch: its calls, its tokens and its sampled positions.
    fn of(staging: &super::Staging, drafts: u32) -> Result<Self> {
        let rows = staging.batch_rows();
        let calls = staging.calls().len();
        let tokens = staging.live_tokens() as usize;
        let row_count = rows
            .calls
            .get(..calls)
            .and_then(|calls| calls.last())
            .map_or(0, |call| call.rows.end as usize);
        let slots = staging.slots(row_count);
        let parameters = staging.parameters(row_count);
        let draws = staging.draws(row_count);
        let ids = staging.input_ids(tokens);
        let work = staging.drafts();
        let mut plan = Plan {
            tokens,
            ..Plan::default()
        };
        for (index, call) in rows.calls[..calls].iter().enumerate() {
            let range = call.rows.start as usize..call.rows.end as usize;
            let first = rows.query_offsets[range.start] as usize;
            let last = rows.query_offsets[range.end] as usize;
            let start = rows.positions.get(first).copied().unwrap_or(0).max(0) as u32;
            let verify = matches!(call.kind, RowKind::Verify | RowKind::Decode);
            plan.verify |= verify;
            let samples = plan.samples.len();
            let mut drafted = Vec::new();
            if verify {
                if range.len() != 1 {
                    return Err(Error::Invariant(format!(
                        "a verification of {} rows: one row carries its tokens",
                        range.len()
                    )));
                }
                drafted = ids[first + 1..last].to_vec();
                if drafted.len() > drafts as usize {
                    return Err(Error::Invalid(format!(
                        "a verification carries {} drafts, the deployment verifies {drafts}",
                        drafted.len()
                    )));
                }
                // Every token samples the position after it, its penalties counting the drafts
                // before it.
                if rows.logits_rows[range.start] >= 0 {
                    for token in first..last {
                        let ahead = token - first;
                        let mut extras = vec![-1; drafts as usize];
                        extras[..ahead].copy_from_slice(&drafted[..ahead]);
                        let row = range.start;
                        plan.samples.push(SampleRow {
                            hidden: token as i32,
                            slot: slots[row],
                            parameters: parameters[row * COLUMNS..(row + 1) * COLUMNS]
                                .try_into()
                                .map_err(|_| Error::Invariant("a parameter row".into()))?,
                            draw: DrawAddress {
                                key: draws[row].key,
                                position: u64::from(start) + ahead as u64 + 1,
                            },
                            extras,
                        });
                    }
                }
            } else {
                for row in range.clone() {
                    if rows.logits_rows[row] < 0 {
                        continue;
                    }
                    plan.samples.push(SampleRow {
                        hidden: rows.query_offsets[row + 1] - 1,
                        slot: slots[row],
                        parameters: parameters[row * COLUMNS..(row + 1) * COLUMNS]
                            .try_into()
                            .map_err(|_| Error::Invariant("a parameter row".into()))?,
                        draw: draws[row],
                        extras: vec![-1; drafts as usize],
                    });
                }
            }
            plan.calls.push(CallPlan {
                slot: slots[range.start],
                kind: Some(call.kind),
                rows: range,
                tokens: first..last,
                start,
                drafts: drafted,
                samples: samples..plan.samples.len(),
                draft: work.get(index).copied().unwrap_or_default(),
            });
        }
        if plan.verify
            && plan
                .calls
                .iter()
                .any(|call| call.kind == Some(RowKind::Prefill))
        {
            return Err(Error::Invariant(
                "a batch mixes prompt rows with verifications".into(),
            ));
        }
        Ok(plan)
    }

    /// A plan of the worst-case step of `bucket`'s shape, for a recorded walk: a prompt bucket's
    /// rows each sampling and proposing, a decode bucket's rows as verifications of every draft.
    fn recorded(bucket: crate::plan::Bucket, drafts: u32) -> Self {
        let parameters = [0.0; COLUMNS];
        let sample = |hidden: usize, slot: i32| SampleRow {
            hidden: hidden as i32,
            slot,
            parameters,
            draw: DrawAddress::default(),
            extras: vec![-1; drafts as usize],
        };
        let work = DraftWork {
            proposals: drafts,
            lookahead: None,
            refill: None,
        };
        let mut plan = Plan::default();
        if bucket.variant.is_decode() {
            let width = drafts as usize + 1;
            let calls = (bucket.rows as usize / width).max(1);
            plan.verify = true;
            plan.tokens = calls * width;
            for call in 0..calls {
                let first = call * width;
                plan.calls.push(CallPlan {
                    slot: call as i32 + 1,
                    kind: Some(RowKind::Verify),
                    rows: call..call + 1,
                    tokens: first..first + width,
                    start: 0,
                    drafts: vec![0; drafts as usize],
                    samples: first..first + width,
                    draft: work,
                });
                for token in first..first + width {
                    plan.samples.push(sample(token, call as i32 + 1));
                }
            }
        } else {
            let rows = bucket.rows as usize;
            let tokens = bucket.tokens as usize;
            plan.tokens = tokens;
            let mut first = 0;
            for row in 0..rows {
                let end = if row + 1 == rows {
                    tokens
                } else {
                    (first + 1).min(tokens)
                };
                plan.calls.push(CallPlan {
                    slot: row as i32 + 1,
                    kind: Some(RowKind::Prefill),
                    rows: row..row + 1,
                    tokens: first..end,
                    start: 0,
                    drafts: Vec::new(),
                    samples: row..row + 1,
                    draft: work,
                });
                plan.samples
                    .push(sample(end.saturating_sub(1), row as i32 + 1));
                first = end;
            }
        }
        plan
    }
}

/// One drafter entry: the position it writes, the token after it and the target token whose hidden
/// state it reads.
#[derive(Clone, Copy, Debug)]
struct Entry {
    position: u32,
    next: i32,
    hidden: i32,
}

/// The drafter's work of one call: its entries, in position order, and the drafts it proposes.
#[derive(Clone, Debug, Default)]
struct CallEntries {
    entries: Vec<Entry>,
    proposals: u32,
}

impl<M: TextModel> CausalLmRunner<M> {
    /// Serves the staged batch in one eager speculative step (the module's stages) and answers its
    /// completion.
    pub(super) fn speculate(
        &mut self,
        generation: u64,
        batch: ipc::execution::BatchId,
    ) -> Result<BatchOutput> {
        let launched = Instant::now();
        let kernels = self.lifecycle.generation_named(generation)?;
        self.install_pending(batch)?;
        let drafts = self.spec()?.drafts;
        let plan = Plan::of(&self.staging, drafts)?;
        // A row that resumes from a cached prefix continues from the checkpoint its prefix ends at.
        self.copy_state(generation, ipc::execution::StateDirection::Restore)?;

        let stream = self.stream.clone();
        let base = self.pool()?.base(self.stream_id())?;
        let reserved = self.pool()?.reserved_on(self.stream_id());
        let arena = nn::Arena::new(ops::DevicePtr::new(base.address()), reserved);
        let tensor = self.tensor.clone();
        let mut forward = Forward::new(&stream, &kernels, &arena);
        if let Some(group) = tensor.as_deref() {
            forward = forward.with_group(group);
        }

        let workspace = arena.tensor(DType::U8, ops::Shape::new(&[ATTENTION_WORKSPACE_BYTES]))?;
        self.write_target(&plan)?;
        let hidden = self.target(&forward, &plan, workspace)?;
        // The rows' checkpoints move into the state pages behind the forward that wrote them.
        self.copy_state(generation, ipc::execution::StateDirection::Keep)?;
        self.write_samples(&plan)?;
        self.sample_rows(&forward, &plan, hidden)?;
        stream.synchronize()?;

        let accepted = self.accept(&plan)?;
        if let Some(state) = self.state.as_ref() {
            for (call, taken) in plan.calls.iter().zip(&accepted) {
                // A verification that rejected a draft leaves its slot's state after the last token
                // it accepted, which the trail holds; one that accepted every draft already ends
                // there.
                if plan.verify && taken.tokens.len() < call.tokens.len() && !taken.tokens.is_empty()
                {
                    let token = call.tokens.start + taken.tokens.len() - 1;
                    state.settle(&forward, call.slot as u32, token as u32, &stream)?;
                }
            }
        }
        self.commit_accepted(&forward, &plan, &accepted)?;
        let entries = self.entries(&plan, &accepted);
        let proposals = self.draft(&forward, &plan, &entries, hidden, workspace)?;
        let slots = self.commit.slots();
        // SAFETY: the readback buffer holds `2 · slots` i32 and the two per-slot columns `slots`
        // each; the stream completes the copies before they are read below.
        unsafe {
            platform::memcpy_async(
                self.commit.readback_address(0),
                self.commit.logical_ptr().address(),
                slots * 4,
                &stream,
            )?;
            platform::memcpy_async(
                self.commit.readback_address(0) + (slots * 4) as u64,
                self.commit.sampling_ptr().address(),
                slots * 4,
                &stream,
            )?;
        }
        if let Some(fence) = self.offload_fence.take() {
            stream.wait_event(&fence)?;
        }
        stream.synchronize()?;
        let device_us = launched.elapsed().as_micros() as u64;

        let host_start = Instant::now();
        // SAFETY: the copies above filled both halves of the readback and the stream completed.
        let (logical, sampling) = unsafe {
            let base = self.commit.readback_ptr(0).cast::<i32>();
            (
                std::slice::from_raw_parts(base, slots),
                std::slice::from_raw_parts(base.add(slots), slots),
            )
        };
        let calls = self.staging.calls().to_vec();
        let mut outputs = Vec::with_capacity(calls.len());
        for (index, ((call, taken), proposed)) in
            plan.calls.iter().zip(accepted).zip(proposals).enumerate()
        {
            let slot = usize::try_from(call.slot).unwrap_or(usize::MAX);
            let state = match (logical.get(slot), sampling.get(slot)) {
                (Some(&logical), Some(&sampling)) => RowState {
                    step_index: sampling.max(0) as u32,
                    cache_computed_len: logical.max(0) as u32,
                    cache_visible_len: sampling.max(0) as u32,
                },
                _ => RowState::default(),
            };
            let mut output = CallOutput::completed(calls[index], taken.tokens, state);
            output.token_logprobs = taken.logprobs;
            output.draft_token_ids = proposed;
            outputs.push(output);
        }
        Ok(BatchOutput {
            calls: outputs,
            timing: TimingCounters {
                queued_us: 0,
                device_us,
                copy_us: 0,
                host_us: host_start.elapsed().as_micros() as u64,
            },
            forward_stats: None,
        })
    }

    /// The speculation's columns, which a speculating runner holds.
    fn spec(&mut self) -> Result<&mut Speculation> {
        self.speculation
            .as_mut()
            .ok_or(Error::State("the runner does not speculate"))
    }

    /// Writes the target forward's columns: the batch's attention rows (a verification's tokens one
    /// row each) and its state rows (one per row of the staged batch).
    fn write_target(&mut self, plan: &Plan) -> Result<()> {
        let rows = self.staging.batch_rows().clone();
        let ids = self.staging.input_ids(plan.tokens).to_vec();
        let staged_rows = plan.calls.last().map_or(0, |call| call.rows.end);
        let checkpoints = self.staging.checkpoints(staged_rows).to_vec();
        let mut seq_lens = Vec::new();
        let mut offsets = vec![0];
        let mut state_offsets = vec![0];
        let mut state_lens = Vec::new();
        let mut state_slots = Vec::new();
        for call in &plan.calls {
            for row in call.rows.clone() {
                let first = rows.query_offsets[row];
                let last = rows.query_offsets[row + 1];
                let end = rows.seq_lens[row];
                if plan.verify {
                    // One attention row per token, each the decode step at its position.
                    for token in first..last {
                        seq_lens.push(end - (last - token) + 1);
                        offsets.push(token + 1);
                    }
                } else {
                    seq_lens.push(end);
                    offsets.push(last);
                }
                state_offsets.push(last);
                state_lens.push(end);
                state_slots.push(call.slot);
            }
        }
        let logits = vec![-1; seq_lens.len()];
        if let Some(state) = self.state.as_mut() {
            state.stage(&checkpoints)?;
            state.checkpoints.push(&self.stream)?;
        }
        let tokens = plan.tokens;
        let spec = self
            .speculation
            .as_mut()
            .ok_or(Error::State("speculation"))?;
        if tokens as u64 > spec.tokens || seq_lens.len() as u64 > spec.rows {
            return Err(Error::Invalid(format!(
                "a speculative step of {tokens} tokens in {} rows, the runner holds {} and {}",
                seq_lens.len(),
                spec.tokens,
                spec.rows
            )));
        }
        let encoded = &mut spec.encoded;
        Speculation::write_i32(&mut spec.positions, encoded, &rows.positions[..tokens])?;
        Speculation::write_i32(&mut spec.ids, encoded, &ids)?;
        Speculation::write_i32(
            &mut spec.slot_mapping,
            encoded,
            &rows.slot_mapping[..tokens],
        )?;
        Speculation::write_i32(&mut spec.seq_lens, encoded, &seq_lens)?;
        Speculation::write_i32(&mut spec.query_offsets, encoded, &offsets)?;
        Speculation::write_i32(&mut spec.logits_rows, encoded, &logits)?;
        Speculation::write_i32(&mut spec.state_offsets, encoded, &state_offsets)?;
        Speculation::write_i32(&mut spec.state_lens, encoded, &state_lens)?;
        Speculation::write_i32(&mut spec.state_slots, encoded, &state_slots)?;
        Ok(())
    }

    /// The target forward over the written columns: the embedding and the model, into a hidden
    /// state of the batch's tokens, which the step's later stages read.
    fn target(
        &mut self,
        forward: &Forward<'_>,
        plan: &Plan,
        workspace: TensorView,
    ) -> Result<TensorView> {
        let tokens = plan.tokens as u64;
        let attention_rows: usize = if plan.verify {
            plan.tokens
        } else {
            plan.calls.iter().map(|call| call.rows.len()).sum()
        };
        let state_rows: u64 = plan.calls.iter().map(|call| call.rows.len() as u64).sum();
        let mut slots = Vec::with_capacity(attention_rows);
        for call in &plan.calls {
            let count = if plan.verify {
                call.tokens.len()
            } else {
                call.rows.len()
            };
            slots.extend(std::iter::repeat_n(call.slot, count));
        }
        let spec = self
            .speculation
            .as_mut()
            .ok_or(Error::State("speculation"))?;
        let live = [tokens as i32];
        Speculation::write_i32(&mut spec.live, &mut spec.encoded, &live)?;
        let spec = self
            .speculation
            .as_ref()
            .ok_or(Error::State("speculation"))?;
        let rows = attention_rows as u64;
        let batch = TokenBatch {
            positions: spec.positions.extent(0..tokens),
            slot_mapping: spec.slot_mapping.extent(0..tokens),
            seq_lens: spec.seq_lens.extent(0..rows),
            query_offsets: spec.query_offsets.extent(0..rows + 1),
            logits_rows: spec.logits_rows.extent(0..rows),
            phase: if plan.verify {
                PhaseKind::Decode
            } else {
                PhaseKind::Prefill
            },
            page_size: self.kv.page_size(),
            num_tokens: plan.tokens as u32,
            max_kv_len: self.max_kv_len,
            draft_lens: None,
        };
        let state_view = StateRows {
            query_offsets: spec.state_offsets.extent(0..state_rows + 1),
            seq_lens: spec.state_lens.extent(0..state_rows),
            slots: spec.state_slots.extent(0..state_rows),
            count: state_rows,
        };
        let ids = spec.ids.extent(0..tokens);
        let live = spec.live.extent(0..1);
        let forward = forward_with_live(forward, live);
        let hidden = forward.arena().tensor(
            self.model.dtype(),
            Shape::new(&[tokens, u64::from(self.model.hidden())]),
        )?;
        let quiet = forward.is_recording();
        self.with_kv_over(
            workspace,
            &slots,
            None,
            quiet,
            Some((state_view, plan.verify)),
            |cache| {
                self.model
                    .embed_input_ids(&forward, ids, hidden)
                    .map_err(Error::from)?;
                self.model
                    .forward(&forward, &batch, cache, hidden)
                    .map_err(Error::from)
            },
        )?;
        Ok(hidden)
    }

    /// Writes the sampling step's columns.
    fn write_samples(&mut self, plan: &Plan) -> Result<()> {
        let spec = self
            .speculation
            .as_mut()
            .ok_or(Error::State("speculation"))?;
        let indices: Vec<i32> = plan.samples.iter().map(|row| row.hidden).collect();
        let slots: Vec<i32> = plan.samples.iter().map(|row| row.slot).collect();
        let parameters: Vec<f32> = plan.samples.iter().flat_map(|row| row.parameters).collect();
        let draws: Vec<f32> = plan.samples.iter().map(|row| row.draw.uniform()).collect();
        let extras: Vec<i32> = plan
            .samples
            .iter()
            .flat_map(|row| row.extras.iter().copied())
            .collect();
        let encoded = &mut spec.encoded;
        Speculation::write_i32(&mut spec.indices, encoded, &indices)?;
        Speculation::write_i32(&mut spec.sample_slots, encoded, &slots)?;
        Speculation::write_f32(&mut spec.parameters, encoded, &parameters)?;
        Speculation::write_f32(&mut spec.draws, encoded, &draws)?;
        Speculation::write_i32(&mut spec.extras, encoded, &extras)?;
        Ok(())
    }

    /// Samples every row of the plan's sampling step through the sampling module: the greedy rows
    /// by the argmax, the others by their draws, each path over every row and each row read from
    /// its own path.
    fn sample_rows(&self, forward: &Forward<'_>, plan: &Plan, hidden: TensorView) -> Result<()> {
        let rows = plan.samples.len() as u64;
        if rows == 0 {
            return Ok(());
        }
        let spec = self
            .speculation
            .as_ref()
            .ok_or(Error::State("speculation"))?;
        let sampling = self
            .sampling
            .as_ref()
            .ok_or(Error::State("a speculating runner samples"))?;
        let arena = forward.arena();
        let gathered = arena.tensor(
            self.model.dtype(),
            Shape::new(&[rows, u64::from(self.model.hidden())]),
        )?;
        nn::IndexRows
            .gather(forward, hidden, spec.indices.extent(0..rows), gathered)
            .map_err(Error::from)?;
        let logits = arena.tensor(
            LOGITS_DTYPE,
            Shape::new(&[rows, u64::from(self.model.vocabulary())]),
        )?;
        self.model
            .compute_logits(forward, gathered, logits)
            .map_err(Error::from)?;
        let state = SamplerState {
            counts: sampling.counts,
            parameters: spec.parameters.shaped(&[rows, COLUMNS as u64]),
            slots: spec.sample_slots.extent(0..rows),
            extras: Some(spec.extras.shaped(&[rows, u64::from(spec.drafts)])),
        };
        let greedy = plan
            .samples
            .iter()
            .any(|row| row.parameters[TEMPERATURE as usize] == 0.0);
        let drawn = plan
            .samples
            .iter()
            .any(|row| row.parameters[TEMPERATURE as usize] != 0.0);
        let sampler = self
            .verifier
            .as_ref()
            .ok_or(Error::State("a speculating runner verifies"))?;
        if greedy || forward.is_recording() {
            sampler.greedy(
                forward,
                logits,
                &state,
                &Samples {
                    tokens: spec.greedy.extent(0..rows),
                    logprobabilities: Some(spec.greedy_logprobs.extent(0..rows)),
                },
            )?;
        }
        if drawn || forward.is_recording() {
            sampler.forward(
                forward,
                logits,
                &state,
                spec.draws.extent(0..rows),
                &Samples {
                    tokens: spec.drawn.extent(0..rows),
                    logprobabilities: Some(spec.drawn_logprobs.extent(0..rows)),
                },
            )?;
        }
        Ok(())
    }

    /// Each call's accepted tokens from the sampled ones: a verification's agreed drafts and the
    /// sample after them, every other call's samples.
    fn accept(&self, plan: &Plan) -> Result<Vec<Accepted>> {
        let spec = self
            .speculation
            .as_ref()
            .ok_or(Error::State("speculation"))?;
        let rows = plan.samples.len();
        let (greedy, greedy_logprobs, drawn, drawn_logprobs) = (
            Speculation::read_i32(&spec.greedy, rows),
            Speculation::read_f32(&spec.greedy_logprobs, rows),
            Speculation::read_i32(&spec.drawn, rows),
            Speculation::read_f32(&spec.drawn_logprobs, rows),
        );
        let mut tokens = Vec::with_capacity(rows);
        let mut logprobs = Vec::with_capacity(rows);
        for (index, row) in plan.samples.iter().enumerate() {
            let (token, logprob) = if row.parameters[TEMPERATURE as usize] == 0.0 {
                (greedy[index], greedy_logprobs[index])
            } else {
                (drawn[index], drawn_logprobs[index])
            };
            let token = u32::try_from(token)
                .ok()
                .filter(|token| *token < spec.vocabulary)
                .ok_or_else(|| {
                    Error::Invariant(format!(
                        "sampling row {index} drew token {token} outside the vocabulary of {}",
                        spec.vocabulary
                    ))
                })?;
            tokens.push(token);
            logprobs.push(logprob);
        }
        Ok(plan
            .calls
            .iter()
            .map(|call| {
                let sampled = &tokens[call.samples.clone()];
                let scores = &logprobs[call.samples.clone()];
                let count = if plan.verify && !sampled.is_empty() {
                    accepted_prefix(&call.drafts, sampled)
                } else {
                    sampled.len()
                };
                Accepted {
                    tokens: sampled[..count].to_vec(),
                    logprobs: scores[..count].to_vec(),
                }
            })
            .collect())
    }

    /// Commits each call's accepted tokens into its slot's device state: the call's own commit row
    /// with the token it installs and the span it advances by, and a count row per accepted draft,
    /// which joins the slot's penalty history before that token.
    fn commit_accepted(
        &mut self,
        forward: &Forward<'_>,
        plan: &Plan,
        accepted: &[Accepted],
    ) -> Result<()> {
        let staged = self.staging.commit().to_vec();
        let calls = plan.calls.len();
        if staged.is_empty() || calls == 0 {
            return Ok(());
        }
        let stride = staged.len() / COMMIT_FIELDS;
        let mut rows: Vec<[i64; COMMIT_FIELDS]> = Vec::new();
        let mut tokens = Vec::new();
        let mut points = Vec::new();
        for (index, taken) in accepted.iter().enumerate() {
            let mut cells = [0_i64; COMMIT_FIELDS];
            for (field, cell) in cells.iter_mut().enumerate() {
                *cell = staged[field * stride + index];
            }
            if cells[column::TASK] >= 0 {
                let Some(&last) = taken.tokens.last() else {
                    return Err(Error::Invariant(format!(
                        "call {index} samples and accepted no token"
                    )));
                };
                cells[column::TASK] = tokens.len() as i64;
                tokens.push(last as i32);
                points.push(taken.tokens.len() as i32);
                let counts = cells[column::PENALTIES];
                let slot = cells[column::SLOT];
                rows.push(cells);
                // The accepted drafts precede the installed token in the slot's history.
                for &draft in &taken.tokens[..taken.tokens.len() - 1] {
                    let mut count = [0_i64; COMMIT_FIELDS];
                    count[column::SLOT] = slot;
                    count[column::MODE] = mode::COUNT;
                    count[column::TASK] = tokens.len() as i64;
                    count[column::PENALTIES] = counts;
                    tokens.push(draft as i32);
                    points.push(0);
                    rows.push(count);
                }
            } else {
                rows.push(cells);
            }
        }
        let spec = self
            .speculation
            .as_mut()
            .ok_or(Error::State("speculation"))?;
        let capacity = spec.capacity as usize;
        if rows.len() > capacity {
            return Err(Error::Invariant(format!(
                "a step commits {} rows, the table holds {capacity}",
                rows.len()
            )));
        }
        let mut table = vec![0_i32; COMMIT_FIELDS * capacity];
        for row in rows.len()..capacity {
            table[column::SLOT * capacity + row] = -1;
            table[column::TASK * capacity + row] = -1;
        }
        for (row, cells) in rows.iter().enumerate() {
            for (field, value) in cells.iter().enumerate() {
                table[field * capacity + row] = i32::try_from(*value).map_err(|_| {
                    Error::Invalid(format!("a commit column value {value} does not fit i32"))
                })?;
            }
        }
        let encoded = &mut spec.encoded;
        Speculation::write_i32(&mut spec.table, encoded, &table)?;
        Speculation::write_i32(&mut spec.commit_tokens, encoded, &tokens)?;
        Speculation::write_i32(&mut spec.points, encoded, &points)?;
        Speculation::write_i32(&mut spec.flags, encoded, &vec![1; capacity])?;
        spec.continuation.write(&vec![0_u8; capacity])?;
        self.launch_commit(forward, rows.len() as u64)
    }

    /// Launches `slot_commit` over the first `count` rows of the written table.
    fn launch_commit(&self, forward: &Forward<'_>, count: u64) -> Result<()> {
        let spec = self
            .speculation
            .as_ref()
            .ok_or(Error::State("speculation"))?;
        let capacity = spec.capacity;
        let descriptor = commit_descriptor(
            capacity,
            commit_block(capacity as u32),
            self.model.vocabulary(),
        );
        let slots = u64::from(self.commit.slots);
        let args = ops::commit::CommitArgs {
            columns: TensorView::with_strides(
                spec.table.device.ptr,
                DType::I32,
                Shape::new(&[COMMIT_FIELDS as u64, capacity]),
                ops::Strides::new(&[capacity, 1]),
            ),
            tokens: spec.commit_tokens.extent(0..capacity),
            valid: spec.flags.extent(0..capacity),
            active: spec.flags.extent(0..capacity),
            continuation: spec.continuation.extent(0..capacity),
            points: spec.points.extent(0..capacity),
            future: TensorView::with_strides(
                self.commit.future.device.ptr,
                DType::I32,
                Shape::new(&[slots, 1]),
                ops::Strides::new(&[1, 1]),
            ),
            predicates: self.commit.predicates.extent(0..slots),
            logical: self.commit.logical.extent(0..slots),
            sampling: self.commit.sampling.extent(0..slots),
            verified: self.commit.verified.extent(0..slots),
            penalties: self
                .sampling
                .as_ref()
                .map(|sampling| sampling.counts)
                .ok_or(Error::State("a speculating runner samples"))?,
            count,
        };
        forward
            .operate::<ops::commit::SlotCommit>(&descriptor, args)
            .map_err(Error::from)
    }

    /// The drafter's entries of every call: the positions whose next token the step now knows.
    ///
    /// A verification's are its accepted positions, each reading the token committed after it; a
    /// prompt row's are its positions, each reading the prompt's next token, the last reading the
    /// row's sampled token or the call's lookahead and writing nothing when it knows neither.
    fn entries(&self, plan: &Plan, accepted: &[Accepted]) -> Vec<CallEntries> {
        let rows = self.staging.batch_rows();
        let ids = self.staging.input_ids(plan.tokens);
        plan.calls
            .iter()
            .zip(accepted)
            .map(|(call, taken)| {
                let mut entries = Vec::new();
                if plan.verify {
                    for (offset, token) in taken.tokens.iter().enumerate() {
                        entries.push(Entry {
                            position: call.start + offset as u32,
                            next: *token as i32,
                            hidden: (call.tokens.start + offset) as i32,
                        });
                    }
                } else {
                    for row in call.rows.clone() {
                        let first = rows.query_offsets[row] as usize;
                        let last = rows.query_offsets[row + 1] as usize;
                        for token in first..last {
                            let next = if token + 1 < last {
                                Some(ids[token + 1])
                            } else if rows.logits_rows[row] >= 0 {
                                taken.tokens.last().map(|token| *token as i32)
                            } else {
                                call.draft.lookahead.map(|token| token as i32)
                            };
                            if let Some(next) = next {
                                entries.push(Entry {
                                    position: rows.positions[token].max(0) as u32,
                                    next,
                                    hidden: token as i32,
                                });
                            }
                        }
                    }
                }
                CallEntries {
                    entries,
                    proposals: call
                        .draft
                        .proposals
                        .min(self.speculation.as_ref().map_or(0, |spec| spec.drafts)),
                }
            })
            .collect()
    }

    /// The draft cache table of `slot`: its first page and units, empty when it holds none.
    fn draft_table(&self, slot: i32) -> (u32, Vec<u32>) {
        let (Some(spec), Some(cache)) = (self.speculation.as_ref(), self.kv.cache()) else {
            return (0, Vec::new());
        };
        let Some(pool) = spec.pool else {
            return (0, Vec::new());
        };
        let Ok(slot) = u32::try_from(slot) else {
            return (0, Vec::new());
        };
        cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .installed_pages(slot, pool.group)
            .unwrap_or_default()
    }

    /// Runs the drafter: its entries over the target's hidden states, then the drafts each call
    /// proposes, step by step on the device, each step embedding the draft the step before chose
    /// and reading the hidden state it left. Answers each call's proposals.
    fn draft(
        &mut self,
        forward: &Forward<'_>,
        plan: &Plan,
        entries: &[CallEntries],
        hidden: TensorView,
        workspace: TensorView,
    ) -> Result<Vec<Vec<u32>>> {
        let recording = forward.is_recording();
        let pool = self
            .speculation
            .as_ref()
            .and_then(|spec| spec.pool)
            .ok_or(Error::State("a speculating runner's draft cache is bound"))?;
        let tables: Vec<(u32, Vec<u32>)> = plan
            .calls
            .iter()
            .map(|call| {
                if recording {
                    (0, Vec::new())
                } else {
                    self.draft_table(call.slot)
                }
            })
            .collect();
        let width = self.speculation.as_ref().map_or(0, |spec| spec.width);

        // The entry rows: one per call with entries, its positions contiguous.
        let mut positions = Vec::new();
        let mut ids = Vec::new();
        let mut slots = Vec::new();
        let mut hidden_rows = Vec::new();
        let mut lens = Vec::new();
        let mut offsets = vec![0];
        let mut table_rows = Vec::new();
        // Per proposing call: its last entry's row in the entries, its last position and its table.
        let mut proposing: Vec<(usize, i32, u32, u32)> = Vec::new();
        for (index, (call, (first, units))) in entries.iter().zip(&tables).enumerate() {
            let Some(last) = call.entries.last() else {
                continue;
            };
            for entry in &call.entries {
                positions.push(entry.position as i32);
                ids.push(entry.next);
                slots.push(pool.slot(*first, units, entry.position));
                hidden_rows.push(entry.hidden);
            }
            lens.push(last.position as i32 + 1);
            offsets.push(positions.len() as i32);
            pool.row(*first, units, width, &mut table_rows);
            if call.proposals > 0 {
                proposing.push((
                    index,
                    positions.len() as i32 - 1,
                    last.position,
                    call.proposals,
                ));
            }
        }
        let mut proposals = vec![Vec::new(); plan.calls.len()];
        if positions.is_empty() {
            return Ok(proposals);
        }
        let drafter_rows = lens.len() as u64;
        let tokens = positions.len() as u64;
        let calls = proposing.len() as u64;
        let steps = proposing
            .iter()
            .map(|(_, _, _, count)| *count)
            .max()
            .unwrap_or(0);

        // Each chained step's rows: the proposing calls, one token each at the next position.
        let mut chained_positions = Vec::new();
        let mut chained_slots = Vec::new();
        let mut chained_lens = Vec::new();
        let mut chained_tables = Vec::new();
        for step in 1..steps {
            for (index, _, last, count) in &proposing {
                let (first, units) = &tables[*index];
                let position = last + step;
                chained_positions.push(position as i32);
                // A call that proposes fewer drafts writes nothing past its own.
                chained_slots.push(if step < *count {
                    pool.slot(*first, units, position)
                } else {
                    -1
                });
                chained_lens.push(position as i32 + 1);
            }
            // Each step's run starts at a 16-byte boundary, as the attention program reads it.
            let end = (step as usize) * pitch(calls) as usize;
            chained_positions.resize(end, 0);
            chained_slots.resize(end, -1);
            chained_lens.resize(end, 0);
        }
        for (index, _, _, _) in &proposing {
            let (first, units) = &tables[*index];
            pool.row(*first, units, width, &mut chained_tables);
        }
        let last_entries: Vec<i32> = proposing.iter().map(|(_, row, _, _)| *row).collect();
        let chained_offsets: Vec<i32> = (0..=calls as i32).collect();
        {
            let spec = self.spec()?;
            if tokens > spec.tokens || drafter_rows > spec.rows || calls > spec.rows {
                return Err(Error::Invalid(format!(
                    "the drafter's {tokens} entries in {drafter_rows} rows exceed the runner's \
                     columns"
                )));
            }
            let encoded = &mut spec.encoded;
            Speculation::write_i32(&mut spec.draft_positions, encoded, &positions)?;
            Speculation::write_i32(&mut spec.draft_ids, encoded, &ids)?;
            Speculation::write_i32(&mut spec.draft_slots, encoded, &slots)?;
            Speculation::write_i32(&mut spec.draft_hidden, encoded, &hidden_rows)?;
            Speculation::write_i32(&mut spec.draft_lens, encoded, &lens)?;
            Speculation::write_i32(&mut spec.draft_offsets, encoded, &offsets)?;
            Speculation::write_i32(&mut spec.draft_tables, encoded, &table_rows)?;
            Speculation::write_i32(&mut spec.chained_positions, encoded, &chained_positions)?;
            Speculation::write_i32(&mut spec.chained_slots, encoded, &chained_slots)?;
            Speculation::write_i32(&mut spec.chained_lens, encoded, &chained_lens)?;
            Speculation::write_i32(&mut spec.chained_offsets, encoded, &chained_offsets)?;
            Speculation::write_i32(&mut spec.chained_tables, encoded, &chained_tables)?;
            Speculation::write_i32(&mut spec.last_entries, encoded, &last_entries)?;
            let mut live = vec![0; 3 * pitch(1) as usize];
            live[pitch(1) as usize] = tokens as i32;
            live[2 * pitch(1) as usize] = calls as i32;
            Speculation::write_i32(&mut spec.live, encoded, &live)?;
        }

        let drafter = self
            .model
            .drafter()
            .ok_or(Error::State("a speculating model carries its drafter"))?;
        let spec = self
            .speculation
            .as_ref()
            .ok_or(Error::State("speculation"))?;
        let planes = self.kv.draft_planes()?;
        let dtype = self.model.dtype();
        let width_hidden = u64::from(self.model.hidden());
        let arena = forward.arena();
        let page = pool.page;
        let view = |tables: TensorView| -> [nn::KvGroup<'_>; 1] {
            [nn::KvGroup {
                block_tables: tables,
                page_size: page,
                dtype: self.kv.dtype(),
                layers: &planes,
            }]
        };

        // The entries.
        let embedded = arena.tensor(dtype, Shape::new(&[tokens, width_hidden]))?;
        let states = arena.tensor(dtype, Shape::new(&[tokens, width_hidden]))?;
        let entries_forward = forward_with_live(forward, spec.live.extent(0..1).shifted(pitch(1)));
        self.model
            .embed_input_ids(&entries_forward, spec.draft_ids.extent(0..tokens), embedded)
            .map_err(Error::from)?;
        nn::IndexRows
            .gather(
                &entries_forward,
                hidden,
                spec.draft_hidden.extent(0..tokens),
                states,
            )
            .map_err(Error::from)?;
        let batch = TokenBatch {
            positions: spec.draft_positions.extent(0..tokens),
            slot_mapping: spec.draft_slots.extent(0..tokens),
            seq_lens: spec.draft_lens.extent(0..drafter_rows),
            query_offsets: spec.draft_offsets.extent(0..drafter_rows + 1),
            logits_rows: spec.draft_lens.extent(0..drafter_rows),
            phase: PhaseKind::Prefill,
            page_size: page,
            num_tokens: tokens as u32,
            max_kv_len: self.max_kv_len,
            draft_lens: None,
        };
        let groups = view(
            spec.draft_tables
                .shaped(&[drafter_rows, u64::from(spec.width)]),
        );
        drafter
            .forward(
                &entries_forward,
                &batch,
                &KvView {
                    groups: &groups,
                    workspace,
                    state: None,
                },
                embedded,
                states,
            )
            .map_err(Error::from)?;
        if calls == 0 {
            return Ok(proposals);
        }

        // The first proposal of each call, from its last entry, then the chained steps.
        let chained = forward_with_live(forward, spec.live.extent(0..1).shifted(2 * pitch(1)));
        let current = arena.tensor(dtype, Shape::new(&[calls, width_hidden]))?;
        nn::IndexRows
            .gather(
                &chained,
                states,
                spec.last_entries.extent(0..calls),
                current,
            )
            .map_err(Error::from)?;
        let logits = arena.tensor(
            LOGITS_DTYPE,
            Shape::new(&[calls, u64::from(self.model.vocabulary())]),
        )?;
        let argmax = ops::sampler::ArgMaxDescriptor {
            dtype: LOGITS_DTYPE,
            width: self.model.vocabulary(),
            ids: DType::I32,
            pdl: nn::layers::launch::PDL,
        };
        let proposal = |step: u32| -> TensorView {
            let at = u64::from(step) * pitch(calls);
            TensorView::new(
                spec.proposals.device.offset_elements(at),
                DType::I32,
                Shape::new(&[calls]),
            )
        };
        let propose = |step: u32| -> Result<()> {
            self.model
                .compute_logits(&chained, current, logits)
                .map_err(Error::from)?;
            chained
                .operate::<ops::sampler::ArgMax>(
                    &argmax,
                    ops::sampler::ArgMaxArgs {
                        values: logits,
                        out: proposal(step),
                    },
                )
                .map_err(Error::from)
        };
        propose(0)?;
        let chained_groups = view(spec.chained_tables.shaped(&[calls, u64::from(spec.width)]));
        let step_embedded = arena.tensor(dtype, Shape::new(&[calls, width_hidden]))?;
        for step in 1..steps {
            let at = u64::from(step - 1) * pitch(calls);
            let column = |column: &Column| -> TensorView {
                TensorView::new(
                    column.device.offset_elements(at),
                    DType::I32,
                    Shape::new(&[calls]),
                )
            };
            self.model
                .embed_input_ids(&chained, proposal(step - 1), step_embedded)
                .map_err(Error::from)?;
            let batch = TokenBatch {
                positions: column(&spec.chained_positions),
                slot_mapping: column(&spec.chained_slots),
                seq_lens: column(&spec.chained_lens),
                query_offsets: spec.chained_offsets.extent(0..calls + 1),
                logits_rows: column(&spec.chained_lens),
                phase: PhaseKind::Decode,
                page_size: page,
                num_tokens: calls as u32,
                max_kv_len: self.max_kv_len,
                draft_lens: None,
            };
            drafter
                .forward(
                    &chained,
                    &batch,
                    &KvView {
                        groups: &chained_groups,
                        workspace,
                        state: None,
                    },
                    step_embedded,
                    current,
                )
                .map_err(Error::from)?;
            propose(step)?;
        }
        if recording {
            return Ok(proposals);
        }
        forward.stream().map_err(Error::from)?.synchronize()?;
        let drawn =
            Speculation::read_i32(&spec.proposals, (u64::from(steps) * pitch(calls)) as usize);
        for (rank, (index, _, _, count)) in proposing.iter().enumerate() {
            for step in 0..*count {
                let token = drawn[step as usize * pitch(calls) as usize + rank];
                // A drafter row with no numeric logit answers past the vocabulary; its proposals
                // stop there, and the verification verifies those before it.
                match u32::try_from(token) {
                    Ok(token) if token < spec.vocabulary => proposals[*index].push(token),
                    _ => break,
                }
            }
        }
        Ok(proposals)
    }

    /// Walks the speculative step of every bucket on recording forwards: the programs its target
    /// forward, sampling, commit and drafter launch, and the extent they reach in the stream's
    /// region, at each bucket's worst case.
    pub(super) fn record_speculation(&mut self, footprint: &mut Footprint) -> Result<()> {
        let Some(drafts) = self.speculation.as_ref().map(|spec| spec.drafts) else {
            return Ok(());
        };
        for bucket in self.capture_plan() {
            if !bucket.variant.samples() || bucket.variant == ipc::execution::Variant::ExpertJoin {
                continue;
            }
            let plan = Plan::recorded(bucket, drafts);
            let bounds = self
                .speculation
                .as_ref()
                .map_or((0, 0), |spec| (spec.tokens, spec.verify_tokens));
            if plan.tokens as u64 > bounds.0 || (plan.verify && plan.tokens as u64 > bounds.1) {
                continue;
            }
            footprint.walk(Some(bucket), |walk| self.walk_speculation(walk, &plan))?;
        }
        Ok(())
    }

    /// One recorded speculative step of `plan`.
    fn walk_speculation(&mut self, walk: &Walk, plan: &Plan) -> Result<()> {
        let forward = walk.forward(self.tensor.as_deref())?;
        let workspace = forward
            .arena()
            .tensor(DType::U8, ops::Shape::new(&[ATTENTION_WORKSPACE_BYTES]))?;
        let hidden = self.target(&forward, plan, workspace)?;
        self.sample_rows(&forward, plan, hidden)?;
        self.launch_commit(&forward, plan.calls.len() as u64)?;
        let entries: Vec<CallEntries> = plan
            .calls
            .iter()
            .map(|call| CallEntries {
                entries: call
                    .tokens
                    .clone()
                    .map(|token| Entry {
                        position: token as u32,
                        next: 0,
                        hidden: token as i32,
                    })
                    .collect(),
                proposals: call.draft.proposals,
            })
            .collect();
        self.draft(&forward, plan, &entries, hidden, workspace)?;
        Ok(())
    }
}

/// The i32 elements a run of `count` takes in a column of several runs: whole 16-byte units, as
/// the programs that read a run's first element in vector loads need it aligned.
fn pitch(count: u64) -> u64 {
    count.max(1).next_multiple_of(4)
}

/// `forward` bounded by the token count at `live`: what a step's MoE sort reads.
fn forward_with_live<'a>(forward: &Forward<'a>, live: TensorView) -> Forward<'a> {
    forward.clone().with_live_tokens(live)
}

/// A view's start moved past its first `elements`: the column's entry the view names.
trait Shifted {
    fn shifted(self, elements: u64) -> TensorView;
}

impl Shifted for TensorView {
    fn shifted(self, elements: u64) -> TensorView {
        TensorView::new(self.offset_elements(elements), self.dtype, self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::{DraftPool, accepted_prefix};

    #[test]
    fn a_verification_commits_the_agreed_drafts_and_the_sample_after_them() {
        // The samples agree with the first two drafts, so the third position's sample stands.
        assert_eq!(accepted_prefix(&[7, 8, 9], &[7, 8, 4, 5]), 3);
        // Every draft agreed: the bonus sample after the last stands as well.
        assert_eq!(accepted_prefix(&[7, 8], &[7, 8, 3]), 3);
        // The first draft disagreed: the first sample alone stands, as a decode's token would.
        assert_eq!(accepted_prefix(&[7, 8], &[1, 8, 3]), 1);
        // No drafts: a verification of one token decodes it.
        assert_eq!(accepted_prefix(&[], &[5]), 1);
    }

    #[test]
    fn a_draft_page_is_read_as_runs_of_the_pools_pages() {
        // Pages of 512 tokens over eight runs of 64, in a pool of 100 units.
        let pool = DraftPool {
            group: 1,
            span: 8,
            page: 64,
            units: 100,
        };
        // A table from draft page 2 holding units 7 and 0 (no unit for page 3).
        let units = [7, 0];
        // Position 1100 is draft page 2, run 1, offset 12: run 1 of unit 7 is page 107.
        assert_eq!(pool.page(2, &units, 1100), (107, true));
        assert_eq!(pool.slot(2, &units, 1100), 107 * 64 + 12);
        // Page 3 holds no unit: nothing is written there.
        assert_eq!(pool.slot(2, &units, 1600), -1);
        // A page before the table's first holds none either.
        assert_eq!(pool.slot(2, &units, 10), -1);
        let mut row = Vec::new();
        pool.row(2, &units, 18, &mut row);
        assert_eq!(&row[16..18], &[7, 100 + 7]);
    }
}
