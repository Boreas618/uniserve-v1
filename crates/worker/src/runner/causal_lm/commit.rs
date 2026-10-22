//! A token batch's output captures, relays and commit.
//!
//! Behind the step in the batch's last job ([`TokenBatch::finish`]), each call
//! advances its progress and contributes one slot-state commit row. The job
//! then copies the batch's completion section, its logprob words and its
//! prompt scores into the output lease, relays sampled tokens and
//! transition decisions into their products, and commits: every owner
//! validates its proposed update first, and the operations past the commit
//! point advance the slot state on the device.
//!
//! Once that job completed ([`TokenBatch::complete`]), the host follows: products
//! become visible, captured graphs are stored and the calls become pending.
//! A job that failed before its commit point retires what the batch
//! reserved and reports every call as failed; a failure at or past it is
//! fatal, because slot state on the device may have advanced
//! ([`crate::executor::batch::Failure`]).

use cuda::{CopyBatch, SourceOrder};

use super::batch::{TokenBatch, TokenCall};
use super::calls;
use super::step::{Invoked, Stepped};
use crate::cuda_graph::{GraphMode, ShapeKey};
use crate::executor::batch::{Batch, CallState, Failure};
use crate::lane::Completion;
use crate::lock;
use crate::memory::{LogprobLayout, ProductBuffer};
use crate::runner::causal_lm::Executed;
use crate::stream::Issue;
use crate::{Error, Result};

/// The step a token batch's failure is reported in.
pub(super) const PHASE: &str = "token batch";

/// What a batch's last job leaves for its completion.
pub(crate) struct Commit {
    /// The step's forwards (none when no call has a text row) and what its
    /// statistics count: its rows and draft tokens. `None` when every call
    /// was skipped.
    step: Option<(Option<super::runners::Step>, usize, u64)>,
    /// The job's first operation past the commit point.
    point: usize,
    /// The commit graphs the job captures: each runner's bucket.
    captures: Vec<(usize, ShapeKey)>,
}

/// How a committed batch's step ran, for its statistics.
pub(super) struct Committed {
    /// The populated invocations' tokens summed, and their mode (any
    /// capture, else any replay, else eager).
    pub(super) executed: Executed,
    pub(super) rows: usize,
    pub(super) drafts: u64,
}

impl TokenBatch<'_> {
    /// Advance progress, then record the capture of outputs, the relay of
    /// products and the commit behind `stepped` in the batch's job. A batch
    /// whose calls were all skipped has no step: only their false flags
    /// were published.
    pub(super) fn finish(
        &mut self,
        stepped: Option<Stepped>,
    ) -> std::result::Result<Commit, Failure> {
        let skipped = Stepped::skipped();
        let invoked = stepped.as_ref().unwrap_or(&skipped);
        let failed = Failure::new(PHASE);
        let commits = self.advance(invoked).map_err(&failed)?;
        self.capture(invoked).map_err(&failed)?;
        self.relay(invoked).map_err(&failed)?;
        let mut commit = self.commit(&commits)?;
        commit.step = stepped.map(|stepped| (stepped.step, stepped.rows, stepped.drafts));
        Ok(commit)
    }

    /// Apply the completion of the batch's last job: store the graphs it
    /// captured, make the products visible and the calls pending. Every
    /// failure here is past the commit point.
    pub(super) fn complete(
        &mut self,
        commit: Commit,
        completion: &mut Completion,
    ) -> std::result::Result<Option<Committed>, Failure> {
        let committed = match commit.step {
            Some((step, rows, drafts)) => {
                let mut summary = Executed {
                    mode: GraphMode::Eager,
                    live_tokens: 0,
                    padded_tokens: 0,
                };
                let executed = match step {
                    Some(step) => self
                        .runtime
                        .complete(step, completion)
                        .map_err(Failure::fatal(PHASE))?,
                    // Only latent conditionings, which the image denoiser
                    // wrote.
                    None => Vec::new(),
                };
                for executed in executed.into_iter().flatten() {
                    summary.live_tokens += executed.live_tokens;
                    summary.padded_tokens += executed.padded_tokens;
                    summary.mode = match (summary.mode, executed.mode) {
                        (GraphMode::Capture, _) | (_, GraphMode::Capture) => GraphMode::Capture,
                        (GraphMode::Replay, _) | (_, GraphMode::Replay) => GraphMode::Replay,
                        _ => GraphMode::Eager,
                    };
                }
                Some(Committed {
                    executed: summary,
                    rows,
                    drafts,
                })
            }
            None => None,
        };
        for (runner, key) in commit.captures {
            self.runtime.runners[runner]
                .complete_commit(key, completion)
                .map_err(Failure::fatal(PHASE))?;
        }

        self.batch
            .commit(self.plan, &self.storage.tensors, self.requests)?;
        Ok(committed)
    }

    /// Whether `operation` of the batch's last job lies past its commit
    /// point.
    pub(super) fn committed(commit: &Commit, operation: usize) -> bool {
        operation >= commit.point
    }

    /// Each active call's progress and slot-state commit row, by the
    /// runner of its microbatch.
    fn advance(&mut self, stepped: &Stepped) -> Result<Vec<(usize, Vec<[i64; 8]>)>> {
        let plan = self.plan;
        let tables = self
            .storage
            .prefix_cache
            .as_ref()
            .ok_or_else(|| Error::Unsupported("text calls require a prefix cache".into()))?
            .tables();
        let mut commits: Vec<(usize, Vec<[i64; 8]>)> = self.runtime.groups[self.group]
            .iter()
            .map(|&runner| (runner, Vec::new()))
            .collect();
        for (call, state) in plan.calls.iter().zip(&mut self.batch.calls) {
            if !state.active {
                continue;
            }
            let prepared = state
                .forward
                .prepared
                .as_ref()
                .ok_or_else(|| Error::Invariant("active call has no row".into()))?;
            let penalties = self
                .requests
                .get(call.request_key.request_id.0)?
                .admission()
                .generation
                .as_ref()
                .is_some_and(|generation| generation.sampling.uses_penalties());
            let position = match state.forward.task {
                Some(task) => Some(stepped.invocation(state.forward.microbatch)?.position(task)),
                None => None,
            };
            let rows = commits
                .iter_mut()
                .find(|(runner, _)| *runner == state.forward.microbatch)
                .ok_or_else(|| {
                    Error::Invariant("a call's microbatch is outside its group".into())
                })?;
            rows.1.push(calls::finish(
                call,
                &mut state.output,
                prepared,
                tables,
                position,
                state.forward.controls.as_ref(),
                penalties,
            )?);
            if !state.forward.prompt.is_empty() {
                state.output.progress.prompt_logits_ready = true;
            }
        }
        Ok(commits)
    }

    /// Copy completion fields, logprob words and prompt scores into the
    /// output lease and record each call's spans.
    fn capture(&mut self, stepped: &Stepped) -> Result<()> {
        for invoked in &stepped.invocations {
            self.capture_invocation(invoked)?;
        }
        Ok(())
    }

    /// Capture one microbatch invocation's outputs from its runner's
    /// columns. The lane stream already waits on the microbatch's stream.
    fn capture_invocation(&mut self, invoked: &Invoked) -> Result<()> {
        let stream = self.job.on(self.batch.stream.as_ref());
        let runner = &self.runtime.runners[invoked.runner];
        let events = &self.storage.events;
        let mut output = lock(&self.batch.buffer);
        let mine = |state: &CallState<TokenCall>| state.forward.microbatch == invoked.runner;
        if invoked.tasks > 0 {
            // Fields of the sampled tasks: `[valid | active | token |
            // accepted]`, one section per field.
            let completion = runner
                .column("completion")?
                .narrow(1, 0, invoked.tasks as i64)?;
            let (offset, extent) = output.capture(&completion, stream, events)?;
            for state in self.batch.calls.iter_mut().filter(|state| mine(state)) {
                if let Some(task) = state.forward.task {
                    state.output.sampling_range = Some((offset, extent, invoked.position(task)));
                }
            }
        }
        if let Some(sampling) = &invoked.sampling
            && let Some(layout) = &sampling.logprobs
        {
            // The reporting tasks' compact words start the column.
            let words = runner
                .column("logprobs")?
                .narrow(0, 0, sampling.logprob_end() as i64)?;
            let (offset, count) = output.capture(&words, stream, events)?;
            output.register_logprobs(
                offset,
                count,
                LogprobLayout {
                    rows: layout.rows.clone(),
                    counts: layout.counts.clone(),
                },
            )?;
            for state in self.batch.calls.iter_mut().filter(|state| mine(state)) {
                if let Some(task) = state.forward.task
                    && let Some(row) = sampling.logprob_row(task)
                {
                    state.output.logprob_range = Some((offset, count, row));
                }
            }
        }
        for state in self.batch.calls.iter_mut().filter(|state| mine(state)) {
            // Chunks of one prompt report their scores in prompt order.
            let mut ranges = Vec::new();
            for prompt in &mut state.forward.prompt {
                let Some(layout) = prompt.layout.take() else {
                    continue;
                };
                let words = runner.column("logprobs")?.narrow(
                    0,
                    prompt.words as i64,
                    prompt.size as i64,
                )?;
                let (offset, count) = output.capture(&words, stream, events)?;
                let scored = layout.rows.len();
                output.register_logprobs(offset, count, layout)?;
                ranges.extend((0..scored).map(|row| (offset, count, row)));
            }
            if !state.forward.prompt.is_empty() {
                state.output.prompt_logprob_ranges = ranges;
            }
        }
        Ok(())
    }

    /// Publish sampled tokens and transition decisions into the products
    /// that relay them, in runs of consecutive task columns of each
    /// microbatch.
    fn relay(&mut self, stepped: &Stepped) -> Result<()> {
        for invoked in &stepped.invocations {
            self.relay_invocation(invoked)?;
        }
        Ok(())
    }

    fn relay_invocation(&mut self, invoked: &Invoked) -> Result<()> {
        let runner = &self.runtime.runners[invoked.runner];
        for (column, transition) in [("tagged", false), ("transition", true)] {
            let mut writes: Vec<(usize, ProductBuffer)> = self
                .batch
                .calls
                .iter()
                .filter(|state| state.forward.microbatch == invoked.runner)
                .filter_map(|state| {
                    let write = if transition {
                        &state.forward.transition_write
                    } else {
                        &state.forward.token_write
                    };
                    Some((invoked.position(state.forward.task?), write.clone()?))
                })
                .collect();
            writes.sort_by_key(|(position, _)| *position);
            let mut start = 0;
            while start < writes.len() {
                let mut end = start + 1;
                while end < writes.len() && writes[end].0 == writes[end - 1].0 + 1 {
                    end += 1;
                }
                let values = runner.column(column)?.narrow(
                    0,
                    writes[start].0 as i64,
                    (end - start) as i64,
                )?;
                let run: Vec<ProductBuffer> = writes[start..end]
                    .iter()
                    .map(|(_, write)| write.clone())
                    .collect();
                self.storage.tensors.write_scalars(
                    &run,
                    &values,
                    self.job.on(self.batch.stream.as_ref()),
                )?;
                start = end;
            }
        }
        Ok(())
    }

    /// Validate every owner's update, then record their commit on the
    /// device. The host commits once the job completed.
    fn commit(
        &mut self,
        commits: &[(usize, Vec<[i64; 8]>)],
    ) -> std::result::Result<Commit, Failure> {
        let failed = Failure::new(PHASE);
        let plan = self.plan;
        let tensors = &self.storage.tensors;

        // Readers end on the consuming stream, after this batch's relay
        // writes when every read has one; completion flags no numerical
        // step wrote become true.
        let after: Vec<ProductBuffer> = self
            .batch
            .calls
            .iter()
            .filter_map(|state| state.forward.token_write.clone())
            .collect();
        self.batch
            .end_reads(tensors, &self.job, &after)
            .map_err(&failed)?;
        self.batch.set_flags(tensors, &self.job).map_err(&failed)?;
        self.batch.validate(plan, tensors).map_err(&failed)?;
        self.batch
            .seal(&self.storage.events, &self.job)
            .map_err(&failed)?;
        self.requests
            .validate_pending(&Batch::<TokenCall>::pending(plan))
            .map_err(&failed)?;

        // Every owner accepted its update. The job's operations from here
        // on change slot state, and any failure from here is fatal.
        let point = self.job.len();
        // Each microbatch runner commits its calls' slot state on its own
        // stream; the lane stream then orders after all of them.
        let mut captures = Vec::new();
        for (runner, rows) in commits {
            let captured = self.runtime.runners[*runner]
                .commit(rows, &self.job)
                .map_err(Failure::fatal(PHASE))?;
            captures.extend(captured.map(|key| (*runner, key)));
        }
        self.install_prompt_tails().map_err(Failure::fatal(PHASE))?;
        if let (Some(experts), Some(lane)) = (&self.experts, &self.batch.stream) {
            experts
                .join_streams(&self.job, lane)
                .map_err(Failure::fatal(PHASE))?;
        }
        self.runtime
            .order
            .record(self.job.on(self.batch.stream.as_ref()))
            .map_err(Failure::fatal(PHASE))?;
        Ok(Commit {
            step: None,
            point,
            captures,
        })
    }

    /// Record the copies of scored chunks' last logits rows into their
    /// slots' retained prompt logits.
    fn install_prompt_tails(&self) -> Result<()> {
        let tails: Vec<(usize, usize, usize)> = self
            .batch
            .calls
            .iter()
            .filter_map(|state| {
                Some((
                    state.forward.microbatch,
                    state.forward.prompt.last()?.tail,
                    state.slot,
                ))
            })
            .collect();
        if tails.is_empty() {
            return Ok(());
        }
        let mut copies = CopyBatch::new(SourceOrder::Stream);
        for &(runner, row, slot) in &tails {
            let tail = self.runtime.runners[runner].column("prompt_tail")?;
            let row_bytes = tail.nbytes() / tail.shape()[0] as u64;
            copies.push(
                self.runtime.state.prompt_row(slot)?,
                tail.address() + row as u64 * row_bytes,
                row_bytes as usize,
            );
        }
        // SAFETY: rows of the binding's tail column, written by this
        // batch's step, into distinct slots' retained logits.
        unsafe { self.job.on(self.batch.stream.as_ref()).copy(&copies) }
    }
}
