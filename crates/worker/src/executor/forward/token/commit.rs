//! A token batch's output captures, relays and commit.
//!
//! After the step, each call advances its progress and contributes one
//! slot-state commit row. The batch then copies its completion section, its
//! logprob words and its prompt scores into the output lease, relays sampled
//! tokens and transition decisions into their products, and commits:
//! every owner validates its proposed update first; past the commit point,
//! products become visible, the slot state advances and the calls become
//! pending, and any failure is fatal.

use uniserve_cuda::{CopyBatch, SourceOrder};

use super::batch::{TokenBatch, lock};
use super::calls;
use super::step::Stepped;
use crate::storage::{LogprobLayout, ProductBuffer, Scalar};
use crate::{Error, Result};

/// A failure, and whether it happened past the commit point.
pub(super) struct Failure {
    pub(super) error: Error,
    pub(super) committed: bool,
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

impl TokenBatch<'_> {
    /// Advance progress, capture outputs, relay products and commit.
    pub(super) fn finish(&mut self, stepped: &Stepped) -> std::result::Result<(), Failure> {
        let commits = self.advance(stepped)?;
        self.capture(stepped)?;
        self.relay(stepped)?;
        self.commit(&commits)
    }

    /// Each active call's progress and slot-state commit row.
    fn advance(&mut self, stepped: &Stepped) -> Result<Vec<[i64; 8]>> {
        let plan = self.plan;
        let tables = self
            .storage
            .kv
            .as_ref()
            .ok_or_else(|| Error::Unsupported("text calls require a KV cache".into()))?
            .tables();
        let mut commits = Vec::new();
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
            commits.push(calls::finish(
                call,
                &mut state.output,
                prepared,
                tables,
                state.task.map(|task| stepped.position(task)),
                state.metadata.as_ref(),
                penalties,
            )?);
            if state.prompt.is_some() {
                state.output.progress.prompt_logits_ready = true;
            }
        }
        Ok(commits)
    }

    /// Copy completion fields, logprob words and prompt scores into the
    /// output lease and record each call's spans.
    fn capture(&mut self, stepped: &Stepped) -> Result<()> {
        let stream = self.stream.as_ref();
        let runner = &self.runtime.runners[self.runner];
        let events = &self.storage.events;
        let mut output = lock(&self.buffer);
        if stepped.tasks > 0 {
            // Fields of the sampled tasks: `[valid | active | token |
            // accepted]`, one section per field.
            let completion = runner
                .column("completion")?
                .narrow(1, 0, stepped.tasks as i64)?;
            let (offset, extent) = output.capture(&completion, stream, events)?;
            for state in &mut self.calls {
                if let Some(task) = state.task {
                    state.output.sampling_range = Some((offset, extent, stepped.position(task)));
                }
            }
        }
        if let Some(sampling) = &stepped.sampling {
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
                for state in &mut self.calls {
                    if let Some(task) = state.task
                        && let Some((first, _, row)) = sampling.logprob_span(task)
                        && first == group.offset
                    {
                        state.output.logprob_range = Some((offset, count, row));
                    }
                }
            }
        }
        for state in &mut self.calls {
            let Some(layout) = state
                .prompt
                .as_mut()
                .and_then(|prompt| prompt.layout.take())
            else {
                continue;
            };
            let prompt = state
                .prompt
                .as_ref()
                .ok_or_else(|| Error::Invariant("prompt score lost its span".into()))?;
            let words =
                runner
                    .column("logprobs")?
                    .narrow(0, prompt.words as i64, prompt.size as i64)?;
            let (offset, count) = output.capture(&words, stream, events)?;
            let scored = layout.rows.len();
            output.register_logprobs(offset, count, layout)?;
            state.output.prompt_logprob_ranges =
                (0..scored).map(|row| (offset, count, row)).collect();
        }
        Ok(())
    }

    /// Publish sampled tokens and transition decisions into the products
    /// that relay them, in runs of consecutive task columns.
    fn relay(&mut self, stepped: &Stepped) -> Result<()> {
        let runner = &self.runtime.runners[self.runner];
        for (column, transition) in [("tagged", false), ("transition", true)] {
            let mut writes: Vec<(usize, ProductBuffer)> = self
                .calls
                .iter()
                .filter_map(|state| {
                    let write = if transition {
                        &state.transition_write
                    } else {
                        &state.token_write
                    };
                    Some((stepped.position(state.task?), write.clone()?))
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
                self.storage
                    .tensors
                    .write_scalars(&run, &values, self.stream.as_ref())?;
                start = end;
            }
        }
        Ok(())
    }

    /// Validate every owner's update, then commit them together.
    fn commit(&mut self, commits: &[[i64; 8]]) -> std::result::Result<(), Failure> {
        let stream = self.stream.as_ref();
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

        // Every owner accepted its update. Any failure from here is fatal.
        let writes = std::mem::take(&mut self.writes);
        self.storage.tensors.commit_writes(&writes).map_err(fatal)?;
        let runner = &self.runtime.runners[self.runner];
        runner.commit(commits).map_err(fatal)?;
        self.install_prompt_tails().map_err(fatal)?;
        self.runtime.ordered(stream).map_err(fatal)?;
        self.requests.add_pending(&pending).map_err(fatal)
    }

    /// Copy scored chunks' last logits rows into their slots' retained
    /// prompt logits.
    fn install_prompt_tails(&self) -> Result<()> {
        let tails: Vec<(usize, usize)> = self
            .calls
            .iter()
            .filter_map(|state| Some((state.prompt.as_ref()?.tail, state.slot)))
            .collect();
        if tails.is_empty() {
            return Ok(());
        }
        let tail = self.runtime.runners[self.runner].column("prompt_tail")?;
        let row_bytes = tail.nbytes() / tail.shape()[0] as u64;
        let mut copies = CopyBatch::new(SourceOrder::Stream);
        for &(row, slot) in &tails {
            copies.push(
                self.runtime.state.prompt_row(slot)?,
                tail.address() + row as u64 * row_bytes,
                row_bytes as usize,
            );
        }
        match self.stream.as_ref() {
            // SAFETY: rows of the binding's tail column, written by this
            // batch's step, into distinct slots' retained logits.
            Some(stream) => unsafe { copies.submit(stream) }.map_err(Error::from),
            None => {
                for (destination, source, bytes) in copies.iter() {
                    // SAFETY: host columns and state of this rank.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            source as *const u8,
                            destination as *mut u8,
                            bytes,
                        );
                    }
                }
                Ok(())
            }
        }
    }
}
