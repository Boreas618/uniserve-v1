//! A token batch's invocation: input rows, sampling tasks and prompt scoring.
//!
//! Every active call becomes one row. Sampled calls become tasks: prefill and
//! decode sample their last projected row, a verification every candidate
//! row. A prefill chunk that scores its prompt projects all of its rows; they
//! score its tokens, its last row also samples. A decode invocation whose
//! calls all qualify uses the decode body's greedy selections and only
//! applies terminal policy; otherwise the tasks form a [`SamplingPlan`].

use uniserve_core::SamplingParams;
use uniserve_ipc::{CallKind, ForwardMode};

use super::StepParts;
use super::batch::TokenBatch;
use super::calls;
use crate::runner::graphs::GraphMode;
use crate::runner::text::{
    Executed, RowSource, SamplingColumns, StepRequest, TokenRow, logit_layout,
};
use crate::sampling::SamplingMetadata;
use crate::sampling::plan::{
    SampleTask, SamplingPlan, greedy_finish_words, logprob_request, logprob_words,
};
use crate::storage::LogprobLayout;
use crate::{Error, Result};

/// A prefill chunk's prompt scoring.
pub(super) struct PromptScore {
    /// The chunk's candidate rows: first and count.
    span: (usize, usize),
    /// Offset of the chunk's first token in the invocation.
    token: usize,
    /// The slot's retained logits predict the chunk's first token.
    continued: bool,
    slot: usize,
    /// Row of the binding's tail column receiving the chunk's last logits.
    pub(super) tail: usize,
    /// Offset of the chunk's words in the binding's logprob column.
    pub(super) words: usize,
    pub(super) size: usize,
    /// The words' layout; absent when no position is scored.
    pub(super) layout: Option<LogprobLayout>,
}

/// One microbatch's invocation, as it ran.
pub(super) struct Invoked {
    /// The runner that ran it.
    pub(super) runner: usize,
    /// The tasks' plan; absent for greedy decode and unsampled calls.
    pub(super) sampling: Option<SamplingPlan>,
    pub(super) tasks: usize,
}

impl Invoked {
    /// A task's column in this invocation's selection outputs.
    pub(super) fn position(&self, task: usize) -> usize {
        self.sampling
            .as_ref()
            .map_or(task, |plan| plan.position(task))
    }
}

/// What a step produced, for capture and commit.
pub(super) struct Stepped {
    /// Every populated microbatch's invocation, in microbatch order.
    pub(super) invocations: Vec<Invoked>,
    /// How the step ran: the populated invocations' tokens summed, and
    /// their mode (any capture, else any replay, else eager).
    pub(super) executed: Executed,
    pub(super) rows: usize,
    pub(super) drafts: u64,
}

impl Stepped {
    /// The invocation of microbatch runner `runner`.
    pub(super) fn invocation(&self, runner: usize) -> Result<&Invoked> {
        self.invocations
            .iter()
            .find(|invoked| invoked.runner == runner)
            .ok_or_else(|| Error::Invariant("a sampled call's microbatch did not run".into()))
    }
}

/// One microbatch's invocation before it runs.
struct Planned {
    runner: usize,
    rows: Vec<TokenRow>,
    sampling: Option<SamplingPlan>,
    greedy: Vec<i64>,
    prompt: Vec<i64>,
    tasks: usize,
}

impl Planned {
    fn request(&self, decode: bool) -> StepRequest<'_> {
        let columns = if !self.greedy.is_empty() {
            Some(SamplingColumns {
                words: &self.greedy,
                draws: &[],
                params: &[],
                slots: &[],
                gates: &[],
                greedy: true,
            })
        } else {
            self.sampling.as_ref().map(|plan| SamplingColumns {
                words: &plan.words,
                draws: &plan.draws,
                params: &plan.params,
                slots: &plan.slots,
                gates: &plan.gates,
                greedy: false,
            })
        };
        StepRequest {
            decode,
            rows: &self.rows,
            sampling: columns,
            prompt: &self.prompt,
        }
    }
}

impl PromptScore {
    /// The chunk's record in the step's prompt descriptor: `[row, count,
    /// token, continued, slot, words, tail, top, requested, ids...]`.
    fn record(&self, words: &mut Vec<i64>, top: usize, ids: &[u32]) {
        words.extend([
            self.span.0 as i64,
            self.span.1 as i64,
            self.token as i64,
            i64::from(self.continued),
            self.slot as i64,
            self.words as i64,
            self.tail as i64,
            top as i64,
            ids.len() as i64,
        ]);
        words.extend(ids.iter().map(|&id| i64::from(id)));
    }
}

impl TokenBatch<'_> {
    /// Build the active calls' invocations, one per microbatch, and run
    /// them as one step.
    ///
    /// Active calls split into contiguous microbatches of
    /// `[i * R / P, (i + 1) * R / P)` rows for `R` rows and `P`
    /// microbatches; each microbatch runner samples and scores its own
    /// rows.
    pub(super) fn step(&mut self) -> Result<Stepped> {
        let plan = self.plan;
        let peers = self.runtime.groups[self.group].clone();
        let indexed = self.runtime.runners[peers[0]].indexed();
        let kv = self
            .storage
            .kv
            .as_ref()
            .ok_or_else(|| Error::Unsupported("text calls require a KV cache".into()))?;
        let tables = kv.tables();

        let active: Vec<usize> = (0..plan.calls.len())
            .filter(|&index| self.calls[index].active)
            .collect();
        let mut params: Vec<Option<(SamplingParams, Vec<u32>)>> = vec![None; plan.calls.len()];
        let mut rows: Vec<TokenRow> = Vec::with_capacity(active.len());
        for &index in &active {
            let call = &plan.calls[index];
            let request = self.requests.get(call.request_key.request_id.0)?;
            let admitted = calls::parameters(request.admission())?.clone();
            let finish = request
                .admission()
                .ar
                .as_ref()
                .map(|ar| ar.finish_token_ids.clone())
                .unwrap_or_default();
            let state = &mut self.calls[index];
            let prepared = calls::prepare(
                call,
                &state.output,
                &admitted,
                state.slot as u32,
                tables,
                indexed,
            )?;
            rows.push(prepared.row.clone());
            state.prepared = Some(prepared);
            params[index] = Some((admitted, finish));
        }

        let count = peers.len();
        let total = active.len();
        let mut planned = Vec::with_capacity(count);
        let mut metadata = Vec::new();
        for (index, &runner) in peers.iter().enumerate() {
            let range = index * total / count..(index + 1) * total / count;
            if range.is_empty() {
                planned.push(None);
                continue;
            }
            planned.push(Some(self.plan_invocation(
                runner,
                &active[range.clone()],
                rows[range].to_vec(),
                &params,
                &mut metadata,
            )?));
        }

        let decode = self.mode == ForwardMode::Decode;
        let requests: Vec<Option<StepRequest<'_>>> = planned
            .iter()
            .map(|planned| planned.as_ref().map(|planned| planned.request(decode)))
            .collect();
        let kv = self
            .storage
            .kv
            .as_mut()
            .ok_or_else(|| Error::Unsupported("text calls require a KV cache".into()))?;
        let (tables, cache) = kv.split();
        let executed = self.runtime.invoke(
            self.experts,
            self.group,
            &requests,
            &mut StepParts { tables, cache },
            None,
        )?;
        drop(requests);
        for (index, controls) in metadata {
            self.calls[index].metadata = Some(controls);
        }

        let mut invocations = Vec::new();
        let mut summary = Executed {
            mode: GraphMode::Eager,
            live_tokens: 0,
            padded_tokens: 0,
        };
        for (planned, executed) in planned.into_iter().zip(executed) {
            let (Some(planned), Some(executed)) = (planned, executed) else {
                continue;
            };
            summary.live_tokens += executed.live_tokens;
            summary.padded_tokens += executed.padded_tokens;
            summary.mode = match (summary.mode, executed.mode) {
                (GraphMode::Capture, _) | (_, GraphMode::Capture) => GraphMode::Capture,
                (GraphMode::Replay, _) | (_, GraphMode::Replay) => GraphMode::Replay,
                _ => GraphMode::Eager,
            };
            invocations.push(Invoked {
                runner: planned.runner,
                sampling: planned.sampling,
                tasks: planned.tasks,
            });
        }
        Ok(Stepped {
            invocations,
            executed: summary,
            rows: rows.len(),
            drafts: rows
                .iter()
                .map(|row| u64::from(row.query.saturating_sub(1)))
                .sum(),
        })
    }

    /// Plan one microbatch's invocation of `active` calls with input
    /// `rows`: its sampling tasks and prompt scoring. Each call's
    /// microbatch, task and prompt state is recorded; its sampling
    /// metadata is appended to `metadata` for after the step.
    fn plan_invocation(
        &mut self,
        runner: usize,
        active: &[usize],
        mut rows: Vec<TokenRow>,
        params: &[Option<(SamplingParams, Vec<u32>)>],
        metadata: &mut Vec<(usize, SamplingMetadata)>,
    ) -> Result<Planned> {
        let plan = self.plan;
        // An all-indexed decode gathers on the device; mixed calls read the
        // indexed rows' tokens and positions from the slot state.
        if !rows.iter().all(|row| row.source == RowSource::Indexed) {
            for row in &mut rows {
                if row.source == RowSource::Indexed {
                    row.source = RowSource::Resident;
                }
            }
        }

        let layout = logit_layout(&rows);
        let vocab = self.runtime.vocab;
        let dimensions = self.runtime.runners[runner].dimensions();
        let mut sampled = Vec::new();
        let mut greedy = self.mode == ForwardMode::Decode;
        let mut prompts: Vec<(usize, (usize, usize), usize, bool)> = Vec::new();
        for (position, &index) in active.iter().enumerate() {
            let call = &plan.calls[index];
            let state = &mut self.calls[index];
            state.microbatch = runner;
            let (admitted, finish) = params[index]
                .as_ref()
                .ok_or_else(|| Error::Invariant("active call lost its parameters".into()))?;
            let prepared = state
                .prepared
                .as_ref()
                .ok_or_else(|| Error::Invariant("active call has no row".into()))?;
            let start = state.output.progress.logical_position;
            let Some(positions) = calls::sample_positions(call, prepared, start)? else {
                greedy = false;
                continue;
            };
            let (controls, draws) = calls::metadata(
                call,
                admitted,
                finish,
                &positions,
                prepared.drafts.clone(),
                state.transition_write.is_some(),
            )?;
            greedy &= calls::greedy_eligible(
                call,
                admitted,
                prepared,
                state.token_write.is_some(),
                state.transition_write.is_some(),
            );
            let (first, count) = layout[position];
            // A verification samples every candidate row; the others their
            // last projected row.
            let row = if self.mode == ForwardMode::Verify {
                first
            } else {
                first + count.saturating_sub(1)
            };
            if call.code == CallKind::Forward(ForwardMode::Prefill) && prepared.scores {
                let continued = start > 0;
                if continued && !state.output.progress.prompt_logits_ready {
                    return Err(Error::Invalid(
                        "continued prompt scoring has no preceding logits".into(),
                    ));
                }
                let token = rows[..position].iter().map(|row| row.query as usize).sum();
                prompts.push((index, (first, count), token, continued));
            }
            state.task = Some(sampled.len());
            sampled.push((index, controls, draws, row, prepared.tagged));
        }
        let tasks: Vec<SampleTask<'_>> = sampled
            .iter()
            .map(|(index, controls, draws, row, gated)| {
                Ok(SampleTask {
                    params: &params[*index]
                        .as_ref()
                        .ok_or_else(|| Error::Invariant("sampled call lost its parameters".into()))?
                        .0,
                    metadata: controls.clone(),
                    draws: draws.clone(),
                    slot: self.calls[*index].slot as u32,
                    row: *row,
                    gated: *gated,
                })
            })
            .collect::<Result<_>>()?;
        // The decode body's greedy selections are final only when every call
        // qualifies; otherwise its logits are sampled like any other.
        greedy &= tasks.len() == active.len();
        let sampling = if tasks.is_empty() || greedy {
            None
        } else {
            Some(SamplingPlan::build(
                &tasks,
                vocab,
                self.device.is_some(),
                0,
                dimensions.logprob_words(),
            )?)
        };

        // Prompt words follow the sampled tasks' logprobs.
        let mut cursor = sampling
            .as_ref()
            .and_then(SamplingPlan::logprob_end)
            .unwrap_or(0);
        let mut prompt_words = Vec::new();
        for (tail, &(index, span, token, continued)) in prompts.iter().enumerate() {
            let (admitted, _) = params[index]
                .as_ref()
                .ok_or_else(|| Error::Invariant("scored call lost its parameters".into()))?;
            let (top, ids) = logprob_request(
                &SamplingParams {
                    n_logprobs: admitted.n_prompt_logprobs,
                    ..admitted.clone()
                },
                vocab,
            );
            // A first chunk has no prediction for its first token.
            let scored = span.1 - usize::from(!continued);
            let score = PromptScore {
                span,
                token,
                continued,
                slot: self.calls[index].slot,
                tail,
                words: cursor,
                size: logprob_words(scored, top, ids.len()),
                layout: (scored > 0).then(|| LogprobLayout {
                    rows: (0..scored).collect(),
                    counts: vec![top; scored],
                    requested_ids: vec![ids.clone(); scored],
                    max_count: top,
                    max_requested: ids.len(),
                }),
            };
            score.record(&mut prompt_words, top, &ids);
            cursor += score.size;
            self.calls[index].prompt = Some(score);
        }
        if cursor > dimensions.logprob_words() {
            return Err(Error::Resource(
                "logprob words exceed the binding's capacity",
            ));
        }
        let greedy_words = if greedy {
            greedy_finish_words(&tasks, vocab)
        } else {
            Vec::new()
        };
        let task_count = tasks.len();
        drop(tasks);
        metadata.extend(
            sampled
                .into_iter()
                .map(|(index, controls, ..)| (index, controls)),
        );
        Ok(Planned {
            runner,
            rows,
            sampling,
            greedy: greedy_words,
            prompt: prompt_words,
            tasks: task_count,
        })
    }
}
