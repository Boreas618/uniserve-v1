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

use super::batch::TokenBatch;
use super::calls::{self, Prepared};
use super::context::{self, Sources};
use crate::runner::diffusion::image::ConditionRow;
use crate::runner::graphs::GraphMode;
use crate::runner::text::{
    Embeddings, Executed, RowSource, SamplingColumns, StepContext, StepRequest, TokenRow,
    logit_layout,
};
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

/// What a step produced, for capture and commit.
pub(super) struct Stepped {
    pub(super) executed: Executed,
    /// The tasks' plan; absent for greedy decode and unsampled calls.
    pub(super) sampling: Option<SamplingPlan>,
    pub(super) tasks: usize,
    pub(super) rows: usize,
    pub(super) drafts: u64,
}

impl Stepped {
    /// A task's column in the selection outputs.
    pub(super) fn position(&self, task: usize) -> usize {
        self.sampling
            .as_ref()
            .map_or(task, |plan| plan.position(task))
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
    /// Build the invocation of the active calls and run it.
    pub(super) fn step(&mut self) -> Result<Stepped> {
        let plan = self.plan;
        let indexed = self.runtime.runners[self.runner].indexed();
        let kv = self
            .storage
            .kv
            .as_ref()
            .ok_or_else(|| Error::Unsupported("text calls require a KV cache".into()))?;
        let tables = kv.tables();

        let active: Vec<usize> = (0..plan.calls.len())
            .filter(|&index| self.calls[index].active)
            .collect();
        let mut forward_rows: Vec<Vec<usize>> = vec![Vec::new(); plan.calls.len()];
        for (row, &call) in plan.forward.call_indices.iter().enumerate() {
            if let Some(rows) = forward_rows.get_mut(call as usize) {
                rows.push(row);
            }
        }
        let mut params: Vec<Option<(SamplingParams, Vec<u32>)>> = vec![None; plan.calls.len()];
        let mut rows: Vec<TokenRow> = Vec::with_capacity(active.len());
        // Each call's last row, from which it samples.
        let mut last_rows: Vec<Option<usize>> = vec![None; plan.calls.len()];
        let mut embeddings: Vec<Embeddings> = Vec::new();
        let mut conditions: Vec<ConditionRow> = Vec::new();
        let mut reads = Vec::new();
        for &index in &active {
            let call = &plan.calls[index];
            let request = self.requests.get(call.request_key.request_id.0)?;
            let admitted = calls::parameters(request.admission())?.clone();
            let input_images = request.admission().input_images;
            let finish = request
                .admission()
                .ar
                .as_ref()
                .map(|ar| ar.finish_token_ids.clone())
                .unwrap_or_default();
            let state = &mut self.calls[index];
            let prepared = if call.writes_context() || call.writes_visual_state() {
                if admitted.prompt_logprobs_requested() {
                    return Err(Error::Unsupported(
                        "prompt scoring of image prompts is not served".into(),
                    ));
                }
                let sources = Sources {
                    tensors: &self.storage.tensors,
                    image: self.image.as_deref(),
                    tables,
                    device: self.device,
                    stream: self.stream.as_ref(),
                    input_images,
                };
                let slot = state.slot as u32;
                let (row, media) = if call.writes_context() {
                    context::context(
                        &sources,
                        call,
                        &state.output,
                        slot,
                        &forward_rows[index],
                        &plan.forward,
                    )?
                } else {
                    context::visual(&sources, call, &state.output, slot)?
                };
                Prepared {
                    row,
                    media: Some(media),
                    scores: false,
                    drafts: Vec::new(),
                    tagged: false,
                }
            } else {
                calls::prepare(
                    call,
                    &state.output,
                    &admitted,
                    state.slot as u32,
                    tables,
                    indexed,
                )?
            };
            match &prepared.media {
                Some(media) => {
                    reads.extend(media.reads.iter().cloned());
                    if let Some(condition) = &media.latent {
                        // The image denoiser writes this call's K/V.
                        conditions.push(condition.clone());
                    } else {
                        let base: usize = rows.iter().map(|row| row.query as usize).sum();
                        for (token, values) in &media.embeddings {
                            embeddings.push(Embeddings {
                                token: base + token,
                                values: values.clone(),
                            });
                        }
                        rows.extend(media.rows.iter().cloned());
                        last_rows[index] = Some(rows.len());
                        rows.push(prepared.row.clone());
                    }
                }
                None => {
                    last_rows[index] = Some(rows.len());
                    rows.push(prepared.row.clone());
                }
            }
            state.prepared = Some(prepared);
            params[index] = Some((admitted, finish));
        }
        self.reads.extend(reads);
        // An all-indexed decode gathers on the device; mixed calls read the
        // indexed rows' tokens and positions from the slot state.
        if !rows.iter().all(|row| row.source == RowSource::Indexed) {
            for row in &mut rows {
                if row.source == RowSource::Indexed {
                    row.source = RowSource::Resident;
                }
            }
        }
        if !conditions.is_empty() {
            self.condition_latents(&conditions)?;
        }

        let layout = logit_layout(&rows);
        let vocab = self.runtime.vocab;
        let dimensions = self.runtime.runners[self.runner].dimensions();
        let mut metadata = Vec::new();
        let mut greedy = self.mode == ForwardMode::Decode;
        let mut prompts: Vec<(usize, (usize, usize), usize, bool)> = Vec::new();
        for &index in &active {
            let call = &plan.calls[index];
            let state = &mut self.calls[index];
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
            let position = last_rows[index]
                .ok_or_else(|| Error::Invalid("a call sampling a token has no text row".into()))?;
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
            state.task = Some(metadata.len());
            metadata.push((index, controls, draws, row, prepared.tagged));
        }
        let tasks: Vec<SampleTask<'_>> = metadata
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
        let columns = if greedy {
            Some(SamplingColumns {
                words: &greedy_words,
                draws: &[],
                params: &[],
                slots: &[],
                gates: &[],
                greedy: true,
            })
        } else {
            sampling.as_ref().map(|plan| SamplingColumns {
                words: &plan.words,
                draws: &plan.draws,
                params: &plan.params,
                slots: &plan.slots,
                gates: &plan.gates,
                greedy: false,
            })
        };
        let kv = self
            .storage
            .kv
            .as_mut()
            .ok_or_else(|| Error::Unsupported("text calls require a KV cache".into()))?;
        let (tables, cache) = kv.split();
        let executed = if rows.is_empty() {
            // Only latent conditionings, which the image denoiser wrote.
            Executed {
                mode: GraphMode::Eager,
                live_tokens: 0,
                padded_tokens: 0,
            }
        } else {
            self.runtime.runners[self.runner].execute(
                &StepRequest {
                    decode: self.mode == ForwardMode::Decode,
                    rows: &rows,
                    sampling: columns,
                    prompt: &prompt_words,
                    embeddings: &embeddings,
                },
                &mut StepContext {
                    tables,
                    cache,
                    state: &self.runtime.state,
                },
            )?
        };
        let task_count = tasks.len();
        drop(tasks);
        for (index, controls, ..) in metadata {
            self.calls[index].metadata = Some(controls);
        }
        Ok(Stepped {
            executed,
            sampling,
            tasks: task_count,
            rows: rows.len(),
            drafts: rows
                .iter()
                .map(|row| u64::from(row.query.saturating_sub(1)))
                .sum(),
        })
    }
}

impl TokenBatch<'_> {
    /// Write the K/V of latent images inserted as conditioning, through the
    /// image denoiser on its stream, ordered between this batch's text
    /// work.
    fn condition_latents(&mut self, rows: &[ConditionRow]) -> Result<()> {
        let image = self.image.as_deref_mut().ok_or_else(|| {
            Error::Unsupported("latent conditioning requires the image runtime".into())
        })?;
        let diffusion = image.diffusion_mut().ok_or_else(|| {
            Error::Unsupported("latent conditioning requires an image denoiser".into())
        })?;
        let stream = diffusion
            .stream()
            // SAFETY: the lanes own the stream for the runtime's lifetime.
            .map(|stream| unsafe { uniserve_cuda::Stream::borrowed(stream.device(), stream.raw()) })
            .transpose()?;
        let kv = self
            .storage
            .kv
            .as_mut()
            .ok_or_else(|| Error::Unsupported("text calls require a KV cache".into()))?;
        let (tables, cache) = kv.split();
        self.runtime.order(stream.as_ref())?;
        diffusion.condition(rows, tables, |table, prefix, query| {
            cache.require_writable(&table.spans(u64::from(prefix), u64::from(query))?)
        })?;
        self.runtime.ordered(stream.as_ref())?;
        self.runtime.order(self.stream.as_ref())
    }
}
