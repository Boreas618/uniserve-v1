//! A token batch's invocation: input rows, sampling tasks and prompt scoring.
//!
//! Every active call contributes its input rows ([`TokenBatch::rows`]): one row,
//! or for a prefill placing image features one row per prompt run and
//! vision block. The calls split into one invocation per expert microbatch
//! of the batch's runner group (one invocation without microbatches), and
//! [`TokenBatch::step`] records their forwards in the batch's job. Sampled calls become
//! tasks: prefill and decode sample their last projected row, a
//! verification every candidate row. A prefill chunk that scores its prompt
//! projects all of its rows; they score its tokens, its last row also
//! samples. A decode invocation whose calls all qualify uses the decode
//! forward's greedy selections, made with each call's finish tokens;
//! otherwise the tasks form the forward's [`SamplingMetadata`].

use uniserve_core::SamplingParams;
use uniserve_ipc::{CallKind, ForwardMode};

use super::CausalLMRunners;
use super::batch::TokenBatch;
use super::calls::{self, Prepared};
use super::context::{self, Sources};
use super::runners::Step;
use crate::executor::image::ImageParts;
use crate::lane::Job;
use crate::runner::causal_lm::{
    Embeddings, FINISH_IDS, PromptChunks, SamplingColumns, StepRequest, TokenRow, logit_layout,
};
use crate::runner::denoiser::image::ConditionRow;
use crate::sampling::{
    SampleTask, SamplingControls, SamplingMetadata, logprob_request, logprob_words,
};
use crate::storage::LogprobLayout;
use crate::storage::memory::StorageView;
use crate::worker::RankStorage;
use crate::{Error, Result};

/// A prefill chunk's prompt scoring.
pub(super) struct PromptScore {
    /// The chunk's candidate rows: first and count.
    span: (usize, usize),
    /// Offset of the chunk's first token in the invocation.
    token: usize,
    /// What predicts the chunk's first token: nothing (0), the slot's
    /// retained logits (1), or candidate row `continued - 2` (the vision
    /// block before it). A chunk of no candidate rows only retains the
    /// logits of its row.
    continued: i64,
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
    pub(super) sampling: Option<SamplingMetadata>,
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

/// A step as the batch's job holds it, for capture and commit.
pub(super) struct Stepped {
    /// Every populated microbatch's invocation, in microbatch order.
    pub(super) invocations: Vec<Invoked>,
    /// The invocations' forwards, which the job's completion says how they
    /// ran; none when no call has a text row.
    pub(super) step: Option<Step>,
    pub(super) rows: usize,
    pub(super) drafts: u64,
}

impl Stepped {
    /// The step of a batch whose calls were all skipped.
    pub(super) fn skipped() -> Self {
        Self {
            invocations: Vec::new(),
            step: None,
            rows: 0,
            drafts: 0,
        }
    }

    /// The invocation of microbatch runner `runner`.
    pub(super) fn invocation(&self, runner: usize) -> Result<&Invoked> {
        self.invocations
            .iter()
            .find(|invoked| invoked.runner == runner)
            .ok_or_else(|| Error::Invariant("a sampled call's microbatch did not run".into()))
    }
}

/// The active calls' input rows, before the calls split into microbatches.
pub(crate) struct Rows {
    /// The calls that have text rows.
    inputs: Vec<CallRows>,
    /// Each active call's sampling parameters and finish tokens, by call.
    params: Vec<Option<(SamplingParams, Vec<u32>)>>,
    /// The latent images the image denoiser writes the K/V of before the
    /// invocations run.
    pub(super) conditions: Vec<ConditionRow>,
}

/// One active call's input rows.
struct CallRows {
    /// The call's index in the batch.
    call: usize,
    /// Its rows in sequence order; the call samples from the last.
    rows: Vec<TokenRow>,
    /// Vision features by their first token, counted from the call's first
    /// row.
    embeddings: Vec<(usize, StorageView)>,
}

/// One microbatch's invocation before it runs.
struct Planned {
    runner: usize,
    rows: Vec<TokenRow>,
    embeddings: Vec<Embeddings>,
    /// Each row's finish tokens when the decode forward's greedy
    /// selections are final; empty otherwise.
    finish_ids: Vec<[i64; FINISH_IDS]>,
    sampling: Option<SamplingMetadata>,
    prompt: PromptChunks,
    tasks: usize,
}

impl Planned {
    fn request(&self, mode: ForwardMode) -> StepRequest<'_> {
        StepRequest {
            mode,
            rows: &self.rows,
            finish_ids: &self.finish_ids,
            sampling: self.sampling.as_ref().map(|metadata| SamplingColumns {
                groups: &metadata.groups,
                draws: &metadata.draws,
                params: &metadata.params,
                slots: &metadata.slots,
                gates: &metadata.gates,
            }),
            prompt: &self.prompt,
            embeddings: &self.embeddings,
        }
    }
}

impl PromptScore {
    /// Add the chunk, which reports its `top` tokens and `ids`, to the
    /// forward's prompt chunks.
    fn record(&self, chunks: &mut PromptChunks, top: usize, ids: &[u32]) {
        chunks.rows.push(self.span.0 as i64);
        chunks.row_counts.push(self.span.1 as i64);
        chunks.tokens.push(self.token as i64);
        chunks.continued.push(self.continued);
        chunks.slots.push(self.slot as i64);
        chunks.offsets.push(self.words as i64);
        chunks.tails.push(self.tail as i64);
        chunks.tops.push(top as i64);
        chunks.ids.push(ids.to_vec());
    }
}

impl TokenBatch<'_> {
    /// Build the active calls' input rows. A latent conditioning has no
    /// text row: the image denoiser writes its K/V before the invocations
    /// run.
    pub(super) fn rows(&mut self) -> Result<Rows> {
        let plan = self.plan;
        let prefix_cache = self
            .storage
            .prefix_cache
            .as_ref()
            .ok_or_else(|| Error::Unsupported("text calls require a prefix cache".into()))?;
        let tables = prefix_cache.tables();

        let active: Vec<usize> = (0..plan.calls.len())
            .filter(|&index| self.batch.calls[index].active)
            .collect();
        let mut forward_rows: Vec<Vec<usize>> = vec![Vec::new(); plan.calls.len()];
        for (row, &call) in plan.forward.call_indices.iter().enumerate() {
            if let Some(rows) = forward_rows.get_mut(call as usize) {
                rows.push(row);
            }
        }
        let mut params: Vec<Option<(SamplingParams, Vec<u32>)>> = vec![None; plan.calls.len()];
        let mut inputs: Vec<CallRows> = Vec::with_capacity(active.len());
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
            let state = &mut self.batch.calls[index];
            let prepared = if call.writes_context() || call.writes_visual_state() {
                // A context prefill scores its prompt text; a visual-state
                // row has no prompt tokens.
                let scores = call.writes_context() && admitted.prompt_logprobs_requested();
                let sources = Sources {
                    tensors: &self.storage.tensors,
                    image: self.image.as_ref(),
                    tables,
                    device: self.batch.device,
                    stream: self.job.on(self.batch.stream.as_ref()),
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
                        scores,
                    )?
                } else {
                    context::visual(&sources, call, &state.output, slot)?
                };
                Prepared {
                    row,
                    media: Some(media),
                    scores,
                    drafts: Vec::new(),
                    tagged: false,
                }
            } else {
                calls::prepare(call, &state.output, &admitted, state.slot as u32, tables)?
            };
            match &prepared.media {
                Some(media) => {
                    reads.extend(media.reads.iter().cloned());
                    if let Some(condition) = &media.latent {
                        // The image denoiser writes this call's K/V.
                        conditions.push(condition.clone());
                        let start = state.output.progress.logical_position;
                        if calls::sample_positions(call, &prepared, start)?.is_some() {
                            return Err(Error::Invalid(
                                "a call sampling a token has no text row".into(),
                            ));
                        }
                    } else {
                        let mut rows = media.rows.clone();
                        rows.push(prepared.row.clone());
                        inputs.push(CallRows {
                            call: index,
                            rows,
                            embeddings: media.embeddings.clone(),
                        });
                    }
                }
                None => inputs.push(CallRows {
                    call: index,
                    rows: vec![prepared.row.clone()],
                    embeddings: Vec::new(),
                }),
            }
            state.forward.prepared = Some(prepared);
            params[index] = Some((admitted, finish));
        }
        self.batch.reads.extend(reads);
        Ok(Rows {
            inputs,
            params,
            conditions,
        })
    }

    /// Build the invocations of `rows`, one per microbatch, and record
    /// them in the batch's job as one step.
    ///
    /// The calls that have input rows split into contiguous microbatches
    /// of `[i * R / P, (i + 1) * R / P)` calls for `R` such calls and `P`
    /// microbatches; each microbatch runner samples and scores its own
    /// calls.
    pub(super) fn step(&mut self, rows: Rows) -> Result<Stepped> {
        let Rows { inputs, params, .. } = rows;
        let peers = self.runtime.groups[self.group].clone();

        let count = peers.len();
        let total = inputs.len();
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
                &inputs[range],
                &params,
                &mut metadata,
            )?));
        }

        let step = if total == 0 {
            // Only latent conditionings, which the image denoiser wrote.
            None
        } else {
            let requests: Vec<Option<StepRequest<'_>>> = planned
                .iter()
                .map(|planned| planned.as_ref().map(|planned| planned.request(self.mode)))
                .collect();
            let prefix_cache =
                self.storage.prefix_cache.as_mut().ok_or_else(|| {
                    Error::Unsupported("text calls require a prefix cache".into())
                })?;
            Some(self.runtime.forward(
                self.experts.as_ref(),
                self.group,
                &requests,
                prefix_cache,
                None,
                &self.job,
            )?)
        };
        for (index, controls) in metadata {
            self.batch.calls[index].forward.controls = Some(controls);
        }

        let invocations = planned
            .into_iter()
            .flatten()
            .map(|planned| Invoked {
                runner: planned.runner,
                sampling: planned.sampling,
                tasks: planned.tasks,
            })
            .collect();
        let rows = || inputs.iter().flat_map(|input| &input.rows);
        Ok(Stepped {
            invocations,
            step,
            rows: rows().count(),
            drafts: rows()
                .map(|row| u64::from(row.query.saturating_sub(1)))
                .sum(),
        })
    }

    /// Plan one microbatch's invocation of the calls of `inputs`: its
    /// rows, sampling tasks and prompt scoring. Each call's microbatch,
    /// task and prompt state is recorded; its sampling metadata is
    /// appended to `metadata` for after the step.
    fn plan_invocation(
        &mut self,
        runner: usize,
        inputs: &[CallRows],
        params: &[Option<(SamplingParams, Vec<u32>)>],
        metadata: &mut Vec<(usize, SamplingControls)>,
    ) -> Result<Planned> {
        let plan = self.plan;
        // Each call's rows in the invocation (first and count) and its
        // features, placed from the invocation's first token.
        let mut rows: Vec<TokenRow> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(inputs.len());
        let mut embeddings: Vec<Embeddings> = Vec::new();
        for input in inputs {
            let base: usize = rows.iter().map(|row| row.query as usize).sum();
            for (token, values) in &input.embeddings {
                embeddings.push(Embeddings {
                    token: base + token,
                    values: values.clone(),
                });
            }
            spans.push((rows.len(), input.rows.len()));
            rows.extend(input.rows.iter().cloned());
        }
        let layout = logit_layout(&rows);
        let vocab = self.runtime.vocab;
        let dimensions = self.runtime.runners[runner].dimensions();
        let mut sampled = Vec::new();
        let mut greedy = self.mode == ForwardMode::Decode;
        let mut finish_ids = vec![[-1_i64; FINISH_IDS]; rows.len()];
        // Prompt chunks: call, candidate rows, first token and what
        // predicts the first token (see `PromptScore::continued`).
        let mut prompts: Vec<(usize, (usize, usize), usize, i64)> = Vec::new();
        for (input, &(first_row, row_count)) in inputs.iter().zip(&spans) {
            let index = input.call;
            let call = &plan.calls[index];
            let state = &mut self.batch.calls[index];
            state.forward.microbatch = runner;
            let (admitted, finish) = params[index]
                .as_ref()
                .ok_or_else(|| Error::Invariant("active call lost its parameters".into()))?;
            let prepared = state
                .forward
                .prepared
                .as_ref()
                .ok_or_else(|| Error::Invariant("active call has no row".into()))?;
            let start = state.output.progress.logical_position;
            let Some(positions) = calls::sample_positions(call, prepared, start)? else {
                greedy = false;
                continue;
            };
            let (controls, draws) = calls::controls(
                call,
                admitted,
                finish,
                &positions,
                prepared.drafts.clone(),
                state.forward.transition_write.is_some(),
            )?;
            greedy &= calls::greedy_eligible(
                call,
                admitted,
                prepared,
                &controls.finish_token_ids,
                state.forward.token_write.is_some(),
                state.forward.transition_write.is_some(),
            );
            // The call samples from its last row.
            let position = first_row + row_count - 1;
            for (id, &token) in finish_ids[position]
                .iter_mut()
                .zip(&controls.finish_token_ids)
            {
                *id = i64::from(token);
            }
            let (first, count) = layout[position];
            // A verification samples its candidate rows, one per token
            // position; the others the last-position logits of their row.
            let row = if self.mode == ForwardMode::Verify {
                first
            } else {
                position
            };
            if call.code == CallKind::Forward(ForwardMode::Prefill)
                && prepared.scores
                && prepared.media.is_none()
            {
                let continued = start > 0;
                if continued && !state.output.progress.prompt_logits_ready {
                    return Err(Error::Invalid(
                        "continued prompt scoring has no preceding logits".into(),
                    ));
                }
                let token = rows[..position].iter().map(|row| row.query as usize).sum();
                prompts.push((index, (first, count), token, i64::from(continued)));
            }
            state.forward.task = Some(sampled.len());
            sampled.push((index, controls, draws, row, prepared.tagged));
        }
        // An image prompt scores each prompt run; a run after a vision block
        // continues from the block's last logits, and a prompt ending with a
        // block retains its logits for the next chunk.
        for (input, &(first_row, row_count)) in inputs.iter().zip(&spans) {
            let index = input.call;
            let state = &self.batch.calls[index];
            let Some(prepared) = &state.forward.prepared else {
                continue;
            };
            if !prepared.scores || prepared.media.is_none() {
                continue;
            }
            let start = state.output.progress.logical_position;
            if start > 0 && !state.output.progress.prompt_logits_ready {
                return Err(Error::Invalid(
                    "continued prompt scoring has no preceding logits".into(),
                ));
            }
            let mut previous = i64::from(start > 0);
            for row in first_row..first_row + row_count {
                let (first, candidates) = layout[row];
                if rows[row].causal {
                    let token = rows[..row].iter().map(|row| row.query as usize).sum();
                    prompts.push((index, (first, candidates), token, previous));
                } else if row + 1 == first_row + row_count {
                    prompts.push((index, (first + candidates - 1, 0), 0, 0));
                }
                previous = 2 + (first + candidates - 1) as i64;
            }
        }
        let tasks: Vec<SampleTask<'_>> = sampled
            .iter()
            .map(|(index, controls, draws, row, gated)| {
                Ok(SampleTask {
                    params: &params[*index]
                        .as_ref()
                        .ok_or_else(|| Error::Invariant("sampled call lost its parameters".into()))?
                        .0,
                    controls: controls.clone(),
                    draws: draws.clone(),
                    slot: self.batch.calls[*index].slot as u32,
                    row: *row,
                    gated: *gated,
                })
            })
            .collect::<Result<_>>()?;
        // The decode forward's greedy selections are final only when every
        // call qualifies; otherwise its logits are sampled like any other.
        greedy &= tasks.len() == inputs.len();
        if !greedy {
            finish_ids.clear();
        }
        let sampling = if tasks.is_empty() || greedy {
            None
        } else {
            Some(SamplingMetadata::build(
                &tasks,
                vocab,
                self.batch.device.is_some(),
                0,
                dimensions.logprob_words(),
            )?)
        };

        // Prompt words follow the sampled tasks' logprobs.
        let mut cursor = sampling
            .as_ref()
            .and_then(SamplingMetadata::logprob_end)
            .unwrap_or(0);
        let mut prompt_chunks = PromptChunks::default();
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
            let scored = span.1.saturating_sub(usize::from(continued == 0));
            let score = PromptScore {
                span,
                token,
                continued,
                slot: self.batch.calls[index].slot,
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
            score.record(&mut prompt_chunks, top, &ids);
            cursor += score.size;
            self.batch.calls[index].forward.prompt.push(score);
        }
        if cursor > dimensions.logprob_words() {
            return Err(Error::Resource(
                "logprob words exceed the binding's capacity",
            ));
        }
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
            embeddings,
            finish_ids,
            sampling,
            prompt: prompt_chunks,
            tasks: task_count,
        })
    }
}

/// Record into `job` the write of the K/V of latent images inserted as
/// conditioning, through the image denoiser on its stream, ordered between
/// the text work the job holds for `stream` before and after it.
pub(super) fn condition_latents(
    runtime: &mut CausalLMRunners,
    image: &mut Option<ImageParts<'_>>,
    storage: &mut RankStorage,
    rows: &[ConditionRow],
    stream: Option<&uniserve_cuda::Stream>,
    job: &Job,
) -> Result<()> {
    let image = image.as_mut().ok_or_else(|| {
        Error::Unsupported("latent conditioning requires the image runtime".into())
    })?;
    let denoiser = image.denoiser_mut().ok_or_else(|| {
        Error::Unsupported("latent conditioning requires an image denoiser".into())
    })?;
    let denoiser_stream = denoiser.stream().cloned();
    let prefix_cache = storage
        .prefix_cache
        .as_mut()
        .ok_or_else(|| Error::Unsupported("text calls require a prefix cache".into()))?;
    runtime.order().wait(job.on(denoiser_stream.as_ref()))?;
    denoiser.condition_latents(rows, prefix_cache, job)?;
    runtime
        .order_mut()
        .record(job.on(denoiser_stream.as_ref()))?;
    runtime.order().wait(job.on(stream))
}
