//! A token batch's output captures, relays and commit.
//!
//! Behind the step in the batch's last job ([`Batch::finish`]), each call
//! advances its progress and contributes one slot-state commit row. The job
//! then copies the batch's completion section, its logprob words and its
//! prompt scores into the output lease, relays sampled tokens and
//! transition decisions into their products, and commits: every owner
//! validates its proposed update first, and the operations past the commit
//! point advance the slot state on the device.
//!
//! Once that job completed ([`Batch::complete`]), the host follows: products
//! become visible, captured graphs are stored and the calls become pending.
//! A job that failed before its commit point retires what the batch
//! reserved and reports every call as failed; a failure at or past it is
//! fatal, because slot state on the device may have advanced.

use uniserve_core::CallId;
use uniserve_cuda::{CopyBatch, SourceOrder};
use uniserve_ipc::RequestKey;

use super::batch::{Batch, CallState};
use super::calls;
use super::step::{Invoked, Stepped};
use crate::cuda_graph::{GraphMode, ShapeKey};
use crate::executor::forward::launch::lock;
use crate::lane::Completion;
use crate::runner::causal_lm::Executed;
use crate::storage::{LogprobLayout, ProductBuffer, Scalar};
use crate::stream::Issue;
use crate::{Error, Result};

/// A failure, and whether it happened past the commit point.
pub(crate) struct Failure {
    pub(super) error: Error,
    pub(super) committed: bool,
}

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
    /// The calls that become pending.
    pending: Vec<(RequestKey, CallId, bool)>,
}

/// How a committed batch's step ran, for its statistics.
pub(super) struct Committed {
    /// The populated invocations' tokens summed, and their mode (any
    /// capture, else any replay, else eager).
    pub(super) executed: Executed,
    pub(super) rows: usize,
    pub(super) drafts: u64,
}

impl From<Error> for Failure {
    fn from(error: Error) -> Self {
        Self {
            error,
            committed: false,
        }
    }
}

fn fatal(error: Error) -> Failure {
    Failure {
        error,
        committed: true,
    }
}

impl Batch<'_> {
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
        let commits = self.advance(invoked)?;
        self.capture(invoked)?;
        self.relay(invoked)?;
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
                    Some(step) => self.runtime.complete(step, completion).map_err(fatal)?,
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
                .map_err(fatal)?;
        }

        let writes = std::mem::take(&mut self.writes);
        self.storage.tensors.commit_writes(&writes).map_err(fatal)?;
        self.requests.add_pending(&commit.pending).map_err(fatal)?;
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
        for (call, state) in plan.calls.iter().zip(&mut self.calls) {
            if !state.active {
                continue;
            }
            let prepared = state
                .prepared
                .as_ref()
                .ok_or_else(|| Error::Invariant("active call has no row".into()))?;
            let penalties = self
                .requests
                .get(call.request_key.request_id.0)?
                .admission()
                .ar
                .as_ref()
                .is_some_and(|ar| ar.sampling.uses_penalties());
            let position = match state.task {
                Some(task) => Some(stepped.invocation(state.microbatch)?.position(task)),
                None => None,
            };
            let rows = commits
                .iter_mut()
                .find(|(runner, _)| *runner == state.microbatch)
                .ok_or_else(|| {
                    Error::Invariant("a call's microbatch is outside its group".into())
                })?;
            rows.1.push(calls::finish(
                call,
                &mut state.output,
                prepared,
                tables,
                position,
                state.controls.as_ref(),
                penalties,
            )?);
            if !state.prompt.is_empty() {
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
        let stream = self.job.on(self.stream.as_ref());
        let runner = &self.runtime.runners[invoked.runner];
        let events = &self.storage.events;
        let mut output = lock(&self.buffer);
        let mine = |state: &CallState| state.microbatch == invoked.runner;
        if invoked.tasks > 0 {
            // Fields of the sampled tasks: `[valid | active | token |
            // accepted]`, one section per field.
            let completion = runner
                .column("completion")?
                .narrow(1, 0, invoked.tasks as i64)?;
            let (offset, extent) = output.capture(&completion, stream, events)?;
            for state in self.calls.iter_mut().filter(|state| mine(state)) {
                if let Some(task) = state.task {
                    state.output.sampling_range = Some((offset, extent, invoked.position(task)));
                }
            }
        }
        if let Some(sampling) = &invoked.sampling {
            for group in sampling.logprobs.iter().flatten() {
                let words = runner.column("logprobs")?.narrow(
                    0,
                    group.offset as i64,
                    group.words as i64,
                )?;
                let (offset, count) = output.capture(&words, stream, events)?;
                output.register_logprobs(
                    offset,
                    count,
                    LogprobLayout {
                        rows: group.layout.rows.clone(),
                        counts: group.layout.counts.clone(),
                        requested_ids: group.layout.requested_ids.clone(),
                        max_count: group.layout.max_count,
                        max_requested: group.layout.max_requested,
                    },
                )?;
                for state in self.calls.iter_mut().filter(|state| mine(state)) {
                    if let Some(task) = state.task
                        && let Some((first, _, row)) = sampling.logprob_span(task)
                        && first == group.offset
                    {
                        state.output.logprob_range = Some((offset, count, row));
                    }
                }
            }
        }
        for state in self.calls.iter_mut().filter(|state| mine(state)) {
            // Chunks of one prompt report their scores in prompt order.
            let mut ranges = Vec::new();
            for prompt in &mut state.prompt {
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
            if !state.prompt.is_empty() {
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
                .calls
                .iter()
                .filter(|state| state.microbatch == invoked.runner)
                .filter_map(|state| {
                    let write = if transition {
                        &state.transition_write
                    } else {
                        &state.token_write
                    };
                    Some((invoked.position(state.task?), write.clone()?))
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
                    self.job.on(self.stream.as_ref()),
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
        let stream = self.job.on(self.stream.as_ref());
        let plan = self.plan;

        // Readers end on the consuming stream, after this batch's writes when
        // every read has a corresponding relay write.
        let after: Vec<ProductBuffer> = self
            .calls
            .iter()
            .filter_map(|state| state.token_write.clone())
            .collect();
        let reads = std::mem::take(&mut self.reads);
        let after = if after.len() == reads.len() {
            after
        } else {
            Vec::new()
        };
        self.storage
            .tensors
            .complete_reads(&reads, stream, &after)?;

        // Completion flags no numerical step wrote become true.
        for state in self.calls.iter().filter(|state| state.active) {
            if let Some(write) = &state.completion_write {
                self.storage
                    .tensors
                    .write_scalar(write, Scalar::Int(1), stream)?;
            }
        }
        {
            let output = lock(&self.buffer);
            for (state, call) in self.calls.iter().zip(&plan.calls) {
                state.output.validate_output(&output, call)?;
            }
        }
        self.storage.tensors.validate_writes(&self.writes)?;
        self.seal()?;
        for state in &mut self.calls {
            state.output.reports_output = self.reports;
        }
        let pending: Vec<_> = plan
            .calls
            .iter()
            .map(|call| (call.request_key, call.call_id, call.advances_state()))
            .collect();
        self.requests.validate_pending(&pending)?;

        // Every owner accepted its update. The job's operations from here
        // on change slot state, and any failure from here is fatal.
        let point = self.job.len();
        // Each microbatch runner commits its calls' slot state on its own
        // stream; the lane stream then orders after all of them.
        let mut captures = Vec::new();
        for (runner, rows) in commits {
            let captured = self.runtime.runners[*runner]
                .commit(rows, &self.job)
                .map_err(fatal)?;
            captures.extend(captured.map(|key| (*runner, key)));
        }
        self.install_prompt_tails().map_err(fatal)?;
        if let (Some(experts), Some(lane)) = (&self.experts, &self.stream) {
            experts.join_streams(&self.job, lane).map_err(fatal)?;
        }
        self.runtime.order.record(stream).map_err(fatal)?;
        Ok(Commit {
            step: None,
            point,
            captures,
            pending,
        })
    }

    /// Record the copies of scored chunks' last logits rows into their
    /// slots' retained prompt logits.
    fn install_prompt_tails(&self) -> Result<()> {
        let tails: Vec<(usize, usize, usize)> = self
            .calls
            .iter()
            .filter_map(|state| Some((state.microbatch, state.prompt.last()?.tail, state.slot)))
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
        unsafe { self.job.on(self.stream.as_ref()).copy(&copies) }
    }
}
