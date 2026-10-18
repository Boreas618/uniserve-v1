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
//! samples. A decode invocation whose calls all qualify selects in the
//! decode forward itself, with each call's finish tokens: greedily when
//! every row is greedy, otherwise each row from its top-k candidates with
//! its parameters and draw ([`RowSampling`]). Otherwise the tasks form the
//! forward's [`SamplingMetadata`], whose sampling methods follow the
//! forward.

use uniserve_ipc::{CallKind, ForwardMode};

use super::CausalLMRunners;
use super::batch::{ForwardRows, TokenBatch};
use super::calls::{self, OwnSelection, Prepared};
use super::context::{self, Sources};
use super::runners::Step;
use crate::executor::image::ImageParts;
use crate::lane::Job;
use crate::runner::causal_lm::{
    Embeddings, FINISH_IDS, PromptChunks, SamplingColumns, StepRequest, TokenRow, logit_layout,
};
use crate::runner::denoiser::image::ConditionRow;
use crate::sampling::{
    RowSampling, SampleTask, SamplingControls, SamplingMetadata, logprob_count, logprob_words,
};
use crate::storage::LogprobLayout;
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
    /// The tasks' plan; absent where the decode forward selects itself and
    /// for unsampled calls.
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
    /// The calls that have text rows, by their index in the batch. A call's
    /// rows are those of its [`Prepared`]: a media call's rows, then its
    /// last row, from which it samples.
    inputs: Vec<usize>,
    /// The latent images the image denoiser writes the K/V of before the
    /// invocations run.
    pub(super) conditions: Vec<ConditionRow>,
}

/// One microbatch's invocation before it runs.
struct Planned {
    runner: usize,
    rows: Vec<TokenRow>,
    embeddings: Vec<Embeddings>,
    /// Each row's finish tokens when the decode forward's own selections
    /// are final; empty otherwise.
    finish_ids: Vec<[i64; FINISH_IDS]>,
    /// The rows' draws and parameters when the decode forward samples its
    /// rows itself.
    row_sampling: Option<RowSampling>,
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
            row_sampling: self.row_sampling.as_ref(),
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
    /// Add the chunk, which reports its `top` tokens, to the forward's
    /// prompt chunks.
    fn record(&self, chunks: &mut PromptChunks, top: usize) {
        chunks.rows.push(self.span.0 as i64);
        chunks.row_counts.push(self.span.1 as i64);
        chunks.tokens.push(self.token as i64);
        chunks.continued.push(self.continued);
        chunks.slots.push(self.slot as i64);
        chunks.offsets.push(self.words as i64);
        chunks.tails.push(self.tail as i64);
        chunks.tops.push(top as i64);
    }
}

impl TokenBatch<'_> {
    /// Build the active calls' input rows from the batch's forward rows,
    /// grouped by call. A latent conditioning has no text row: the image
    /// denoiser writes its K/V before the invocations run.
    pub(super) fn rows(&mut self, forward_rows: &ForwardRows) -> Result<Rows> {
        let plan = self.plan;
        let prefix_cache = self
            .storage
            .prefix_cache
            .as_ref()
            .ok_or_else(|| Error::Unsupported("text calls require a prefix cache".into()))?;
        let tables = prefix_cache.tables();

        let mut inputs = Vec::with_capacity(plan.calls.len());
        let mut conditions: Vec<ConditionRow> = Vec::new();
        let mut reads = Vec::new();
        for (index, call) in plan.calls.iter().enumerate() {
            if !self.batch.calls[index].active {
                continue;
            }
            let admission = self
                .requests
                .get(call.request_key.request_id.0)?
                .admission();
            let admitted = calls::parameters(admission)?;
            let input_images = admission.input_images;
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
                        forward_rows.of(index),
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
                calls::prepare(call, &state.output, admitted, state.slot as u32, tables)?
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
                        inputs.push(index);
                    }
                }
                None => inputs.push(index),
            }
            state.forward.prepared = Some(prepared);
        }
        self.batch.reads.extend(reads);
        Ok(Rows { inputs, conditions })
    }

    /// Build the invocations of `rows`, one per microbatch, and record
    /// them in the batch's job as one step.
    ///
    /// The calls that have input rows split into contiguous microbatches
    /// of `[i * R / P, (i + 1) * R / P)` calls for `R` such calls and `P`
    /// microbatches; each microbatch runner samples and scores its own
    /// calls.
    pub(super) fn step(&mut self, rows: Rows) -> Result<Stepped> {
        let Rows { inputs, .. } = rows;

        let count = self.runtime.groups[self.group].len();
        let total = inputs.len();
        let mut planned = Vec::with_capacity(count);
        let mut metadata = Vec::new();
        for index in 0..count {
            let runner = self.runtime.groups[self.group][index];
            let range = index * total / count..(index + 1) * total / count;
            if range.is_empty() {
                planned.push(None);
                continue;
            }
            planned.push(Some(self.plan_invocation(
                runner,
                &inputs[range],
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

        // Every input row is in exactly one microbatch's invocation.
        let rows = planned.iter().flatten().flat_map(|planned| &planned.rows);
        let drafts = rows
            .clone()
            .map(|row| u64::from(row.query.saturating_sub(1)))
            .sum();
        let rows = rows.count();
        let invocations = planned
            .into_iter()
            .flatten()
            .map(|planned| Invoked {
                runner: planned.runner,
                sampling: planned.sampling,
                tasks: planned.tasks,
            })
            .collect();
        Ok(Stepped {
            invocations,
            step,
            rows,
            drafts,
        })
    }

    /// Plan one microbatch's invocation of the calls of `inputs`: its
    /// rows, sampling tasks and prompt scoring. Each call's microbatch,
    /// task and prompt state is recorded; its sampling controls are
    /// appended to `metadata` for after the step.
    fn plan_invocation(
        &mut self,
        runner: usize,
        inputs: &[usize],
        metadata: &mut Vec<(usize, SamplingControls)>,
    ) -> Result<Planned> {
        let plan = self.plan;
        // Each call's rows in the invocation (first and count) and its
        // first token, with its features placed from the invocation's
        // first token.
        let mut rows: Vec<TokenRow> = Vec::with_capacity(inputs.len());
        let mut spans: Vec<(usize, usize, usize)> = Vec::with_capacity(inputs.len());
        let mut embeddings: Vec<Embeddings> = Vec::new();
        let mut tokens = 0;
        for &index in inputs {
            let prepared = self.batch.calls[index]
                .forward
                .prepared
                .as_ref()
                .ok_or_else(|| Error::Invariant("active call has no row".into()))?;
            let first = rows.len();
            if let Some(media) = &prepared.media {
                for (token, values) in &media.embeddings {
                    embeddings.push(Embeddings {
                        token: tokens + token,
                        values: values.clone(),
                    });
                }
                rows.extend(media.rows.iter().cloned());
            }
            rows.push(prepared.row.clone());
            spans.push((first, rows.len() - first, tokens));
            tokens += rows[first..]
                .iter()
                .map(|row| row.query as usize)
                .sum::<usize>();
        }
        let layout = logit_layout(&rows);
        let vocab = self.runtime.vocab;
        let dimensions = self.runtime.runners[runner].dimensions();
        // The sampled calls' tasks, and each task's call.
        let mut tasks: Vec<SampleTask<'_>> = Vec::with_capacity(inputs.len());
        let mut sampled: Vec<usize> = Vec::with_capacity(inputs.len());
        // The decode forward selects every row itself when each call allows
        // it, and samples when any row is not greedy.
        let mut own = (self.mode == ForwardMode::Decode).then_some(OwnSelection::Greedy);
        let mut finish_ids = vec![[-1_i64; FINISH_IDS]; rows.len()];
        // Prompt chunks: call, candidate rows, first token and what
        // predicts the first token (see `PromptScore::continued`).
        let mut prompts: Vec<(usize, (usize, usize), usize, i64)> = Vec::new();
        for (&index, &(first_row, row_count, first_token)) in inputs.iter().zip(&spans) {
            let call = &plan.calls[index];
            // The request's admitted parameters, borrowed for the tasks.
            let admission = self
                .requests
                .get(call.request_key.request_id.0)?
                .admission();
            let admitted = calls::parameters(admission)?;
            let finish = admission
                .generation
                .as_ref()
                .map_or(&[][..], |generation| generation.finish_token_ids.as_slice());
            let state = &mut self.batch.calls[index];
            state.forward.microbatch = runner;
            let prepared = state
                .forward
                .prepared
                .as_ref()
                .ok_or_else(|| Error::Invariant("active call has no row".into()))?;
            let start = state.output.progress.logical_position;
            let Some(positions) = calls::sample_positions(call, prepared, start)? else {
                own = None;
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
            let path = controls.path(admitted, vocab);
            own = own
                .zip(calls::own_selection(
                    call,
                    prepared,
                    &controls,
                    path,
                    state.forward.token_write.is_some(),
                    state.forward.transition_write.is_some(),
                ))
                .map(|(selection, call)| selection.max(call));
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
            if call.kind == CallKind::Forward(ForwardMode::Prefill)
                && prepared.scores
                && prepared.media.is_none()
            {
                let continued = start > 0;
                if continued && !state.output.progress.prompt_logits_ready {
                    return Err(Error::Invalid(
                        "continued prompt scoring has no preceding logits".into(),
                    ));
                }
                // A call without media has one row, at its first token.
                prompts.push((index, (first, count), first_token, i64::from(continued)));
            }
            let gated = prepared.tagged;
            state.forward.task = Some(tasks.len());
            tasks.push(SampleTask {
                params: admitted,
                controls,
                draws,
                slot: state.slot as u32,
                row,
                gated,
            });
            sampled.push(index);
        }
        // An image prompt scores each prompt run; a run after a vision block
        // continues from the block's last logits, and a prompt ending with a
        // block retains its logits for the next chunk.
        for (&index, &(first_row, row_count, first_token)) in inputs.iter().zip(&spans) {
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
            let mut token = first_token;
            for row in first_row..first_row + row_count {
                let (first, candidates) = layout[row];
                if rows[row].causal {
                    prompts.push((index, (first, candidates), token, previous));
                } else if row + 1 == first_row + row_count {
                    prompts.push((index, (first + candidates - 1, 0), 0, 0));
                }
                previous = 2 + (first + candidates - 1) as i64;
                token += rows[row].query as usize;
            }
        }
        // The decode forward's own selections are final only when every
        // call qualifies; otherwise its logits are sampled like any other.
        if tasks.len() != inputs.len() {
            own = None;
        }
        if own.is_none() {
            finish_ids.clear();
        }
        let row_sampling = match own {
            Some(OwnSelection::TopK) => Some(RowSampling::build(&tasks, vocab)?),
            Some(OwnSelection::Greedy) | None => None,
        };
        let sampling = if tasks.is_empty() || own.is_some() {
            None
        } else {
            Some(SamplingMetadata::build(
                &tasks,
                vocab,
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
            let admission = self
                .requests
                .get(plan.calls[index].request_key.request_id.0)?
                .admission();
            let admitted = calls::parameters(admission)?;
            let top = logprob_count(admitted.n_prompt_logprobs, vocab);
            // A first chunk has no prediction for its first token.
            let scored = span.1.saturating_sub(usize::from(continued == 0));
            let score = PromptScore {
                span,
                token,
                continued,
                slot: self.batch.calls[index].slot,
                tail,
                words: cursor,
                size: logprob_words(scored, top),
                layout: (scored > 0).then(|| LogprobLayout {
                    rows: (0..scored).collect(),
                    counts: vec![top; scored],
                    max_count: top,
                }),
            };
            score.record(&mut prompt_chunks, top);
            cursor += score.size;
            self.batch.calls[index].forward.prompt.push(score);
        }
        if cursor > dimensions.logprob_words() {
            return Err(Error::Resource(
                "logprob words exceed the binding's capacity",
            ));
        }
        let task_count = tasks.len();
        // Each sampled call keeps its controls for the commit.
        metadata.extend(
            sampled
                .into_iter()
                .zip(tasks)
                .map(|(index, task)| (index, task.controls)),
        );
        Ok(Planned {
            runner,
            rows,
            embeddings,
            finish_ids,
            row_sampling,
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
