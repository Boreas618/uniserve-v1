//! What one call's completion carries, and what one batch's does, as the
//! mechanism reads them.
//!
//! Design §4.4's completion path is the mechanism's own: it is fed by whichever
//! executor ran the batch, and it must not be shaped by the wire protocol one of
//! them happens to speak. This module is that shape. `notes/sched/completion.md`
//! states, field by field, whether the device produces a value or the mechanism
//! already owns it; the short version is that the executor states *what
//! execution did* — the tokens it accepted, the cache extent it initialized and
//! what a successor may read, the log-probabilities a client asked for, the
//! artifact and the cache export — and the mechanism keeps *its own rules*: the
//! stop conditions are evaluated here from the sampled token
//! (`OutputSender::und_token_outcome`), and the call's budget, cache window and
//! sampling rows are the plan's.
//!
//! Two fillers produce these types, one per executor, both at the edge of the
//! mechanism and neither inventing a value:
//!
//! - [`CallCompletion::from_wire`] / [`BatchReport::from_wire`] for the
//!   worker-process executor, which speaks `ipc`. They are temporary: they are
//!   deleted with that executor (§2.4).
//! - [`BatchReport::from_output`] / [`CallCompletion::from_output`] for the
//!   in-process executor, which reports `executor::batch::BatchOutput`: the
//!   device's own account of each call (2.4). This is the filler the
//!   worker-process one is deleted with.

use std::sync::Arc;

use ipc::{
    CallId, CallStatus, ErrorCode, ForwardStats, RankLatentLayout, RequestKey, SharedMedia,
    SharedMediaError, TimingCounters, TokenLogprob,
};

use crate::executor::{BatchResult, CallResult};

/// One call's completion, as the mechanism reads it.
#[derive(Clone, Debug)]
pub(crate) struct CallCompletion {
    /// The request the call advanced.
    pub request_key: RequestKey,
    /// The request-local call identifier.
    pub call_id: CallId,
    /// Whether execution performed the call.
    pub status: CallStatus,
    /// Why it failed, when it did.
    pub error_code: Option<ErrorCode>,
    /// What the rank said about a refused input, when it refused one.
    pub error_message: Option<String>,
    /// The call's own measurements: folded into its domain's statistics.
    pub timing_counters: TimingCounters,
    /// The accepted cache prefix a successor may read.
    pub cache_visible_len: u32,
    /// The cache extent execution initialized, rejected speculative positions
    /// included.
    pub cache_computed_len: u32,
    /// A denoising trajectory's next step.
    pub step_index: u32,
    /// The tokens the sampler accepted.
    pub committed_tokens: Vec<u32>,
    /// Each committed token's log-probabilities, in `committed_tokens` order.
    pub token_logprobs: Vec<Vec<TokenLogprob>>,
    /// Ranked candidates for each scored prompt position, in input order.
    pub prompt_logprobs: Vec<Vec<TokenLogprob>>,
    /// A token-denoising readout's candidate probabilities.
    pub candidate_logprobs: Vec<f32>,
    /// The drafter's proposals after the last committed token.
    pub draft_token_ids: Vec<u32>,
    /// A video latent preparation's per-rank page layout.
    pub latent_layouts: Vec<RankLatentLayout>,
    /// The immutable storage the device published, claimed by the mechanism —
    /// or its claim's error, which belongs to the request that asked for it.
    pub media: Result<Option<Arc<SharedMedia>>, SharedMediaError>,
}

impl CallCompletion {
    /// The worker-process edge: what a rank reported, in the mechanism's type.
    ///
    /// A worker's `finish_flags` are dropped rather than carried: the stop
    /// conditions are the request's own rules and the mechanism evaluates them
    /// from the sampled token, so a second, device-side copy of that rule would
    /// be a weaker one. Temporary — this filler goes with the worker-process
    /// executor.
    pub(crate) fn from_wire(result: CallResult) -> Self {
        let output = result.output;
        Self {
            request_key: output.request_key,
            call_id: output.call_id,
            status: output.status,
            error_code: output.error_code,
            error_message: output.error_message,
            timing_counters: output.timing_counters,
            cache_visible_len: output.cache_visible_len,
            cache_computed_len: output.cache_computed_len,
            step_index: output.step_index,
            committed_tokens: output.committed_tokens,
            token_logprobs: output.token_logprobs,
            prompt_logprobs: output.prompt_logprobs,
            candidate_logprobs: output.candidate_logprobs,
            draft_token_ids: output.draft_token_ids,
            latent_layouts: output.latent_layouts,
            media: result.media,
        }
    }
}

impl CallCompletion {
    /// The in-process edge: what a device's runner reported, in the mechanism's
    /// type (2.4).
    ///
    /// The mechanism names the call: the executor numbers a batch's calls by
    /// their index, so the pair comes from the batch that was submitted. A host
    /// call publishes its artifact into a POSIX shared-memory object and names
    /// it, and that name is claimed here — the mapping is what the mechanism
    /// delivers, and claiming it is what unlinks the name.
    ///
    /// Two things the device reports differently from a worker, both stated
    /// rather than filled in: a call has no timing of its own (the batch
    /// measures the step, and `BatchReport::timing` carries it), and a token's
    /// log-probability arrives as one number per accepted token, not as ranked
    /// candidates. The mechanism's type wants candidates whose first entry is
    /// the accepted token — which is what `generation` checks — so the accepted
    /// token is stated as the one candidate the device named, never a
    /// neighbour invented to fill the list.
    pub(crate) fn from_output(
        output: executor::batch::CallOutput,
        request_key: RequestKey,
        call_id: CallId,
    ) -> Self {
        let token_logprobs = output
            .tokens
            .iter()
            .zip(&output.token_logprobs)
            .map(|(token, logprob)| {
                vec![TokenLogprob {
                    token_id: *token,
                    logprob: *logprob,
                    rank: 1,
                }]
            })
            .collect();
        Self {
            request_key,
            call_id,
            status: output.status,
            error_code: output.error_code,
            error_message: output.error_message,
            timing_counters: TimingCounters::default(),
            cache_visible_len: output.cache_visible_len,
            cache_computed_len: output.cache_computed_len,
            step_index: output.step_index,
            committed_tokens: output.tokens,
            token_logprobs,
            prompt_logprobs: output.prompt_logprobs,
            candidate_logprobs: output.candidate_logprobs,
            draft_token_ids: output.draft_token_ids,
            latent_layouts: output.latent_layouts,
            media: match output.media_output.as_ref() {
                Some(published) => claim(published),
                None => Ok(None),
            },
        }
    }
}

/// Claims the artifact an in-process host call published.
///
/// The executor's media stage copies the mux's bytes into a shared-memory object and relinquishes
/// it (`ipc::MediaSource::into_locator`), which states the object's name and its exact length; the
/// claim is the mechanism's, and it is the call that unlinks the name and maps the bytes.
///
/// # Safety
///
/// The publisher writes nothing after it relinquishes the object and never resizes it, which is the
/// contract `SharedMedia::open` requires. The mechanism is the only claimer of a name the executor
/// published for one completion.
fn claim(published: &ipc::MediaOutput) -> Result<Option<Arc<SharedMedia>>, SharedMediaError> {
    let ipc::ArtifactHandle::PosixShm { name } = &published.handle;
    // SAFETY: see this function's contract - the bytes are written before the publisher
    // relinquishes the object and no process modifies it while the mapping lives.
    let media = unsafe { SharedMedia::open(name, published.bytes)? };
    Ok(Some(Arc::new(media)))
}

/// One batch's completion, as the mechanism reads it.
#[derive(Clone, Debug)]
pub(crate) struct BatchReport {
    /// The batch's number, which the mechanism assigned at assembly.
    pub batch_id: u64,
    /// One completion per call that finished, in the executor's order.
    pub calls: Vec<CallCompletion>,
    /// The receipts of the batch's lifecycle commands.
    pub command_results: Vec<crate::executor::CommandResult>,
    /// Whether this report ends the batch.
    pub done: bool,
    /// The batch's own measurements: the execution time the report is measured
    /// by.
    pub timing: TimingCounters,
    /// Per-worker model-forward statistics.
    pub forward_stats: Vec<ForwardStats>,
}

impl BatchReport {
    /// The worker-process edge: a logical result, in the mechanism's type.
    /// Temporary, with [`CallCompletion::from_wire`].
    pub(crate) fn from_wire(result: BatchResult) -> Self {
        Self {
            batch_id: result.batch_id,
            calls: result
                .results
                .into_iter()
                .map(CallCompletion::from_wire)
                .collect(),
            command_results: result.command_results,
            done: result.done,
            // The wire states a duration per worker; the mechanism measures a
            // step by the device work it waited for, which is their sum.
            timing: TimingCounters {
                device_us: result
                    .worker_exec_us
                    .into_iter()
                    .fold(0u64, u64::saturating_add),
                ..TimingCounters::default()
            },
            forward_stats: result.forward_stats,
        }
    }

    /// The in-process edge: what this process's devices reported, in the
    /// mechanism's type.
    ///
    /// `calls` names the batch's calls as the mechanism named them: the
    /// executor numbers a batch's calls by their index in `Batch.calls`, so a
    /// report's call is resolved against the batch that was submitted, and a
    /// report naming a call the batch does not have is refused rather than
    /// filed against a neighbour.
    ///
    /// A completed batch acknowledges its lifecycle commands by reporting none
    /// of them failed: the mechanism knots a failed receipt to a command it
    /// keeps for a later attempt, so silence is the receipt of a command that
    /// was applied. A serving batch whose report carries no call is refused —
    /// a runner that serves rows and reports nothing has not been wired to
    /// report them, and advancing a call's state on tokens nobody accounted for
    /// is worse than stopping.
    pub(crate) fn from_output(
        batch: u64,
        outcome: executor::completion::Outcome<executor::batch::BatchOutput>,
        calls: &[(RequestKey, CallId)],
    ) -> Result<Self, crate::executor::ExecutorError> {
        let output = match outcome {
            executor::completion::Outcome::Success(output) => output,
            executor::completion::Outcome::Failed(error) => {
                return Err(anyhow::anyhow!("batch {batch} failed: {error}"));
            }
            executor::completion::Outcome::Cancelled => {
                return Err(anyhow::anyhow!(
                    "batch {batch} was cancelled by the executor"
                ));
            }
        };
        let mut reported: Vec<Option<executor::batch::CallOutput>> = vec![None; calls.len()];
        for completion in output.calls {
            let index = completion.call.0 as usize;
            anyhow::ensure!(
                index < calls.len(),
                "batch {batch} reported call {} of {} the batch submitted",
                completion.call.0,
                calls.len()
            );
            anyhow::ensure!(
                reported[index].is_none(),
                "batch {batch} reported call {} twice",
                completion.call.0
            );
            reported[index] = Some(completion);
        }
        let mut finished = Vec::with_capacity(calls.len());
        for (index, completion) in reported.into_iter().enumerate() {
            let Some(completion) = completion else {
                continue;
            };
            let (request_key, call_id) = calls[index];
            finished.push(CallCompletion::from_output(
                completion,
                request_key,
                call_id,
            ));
        }
        anyhow::ensure!(
            !finished.is_empty() || calls.is_empty(),
            "batch {batch} completed with no call reported for its {} calls",
            calls.len()
        );
        Ok(Self {
            batch_id: batch,
            calls: finished,
            // No lifecycle command of an in-process batch failed: the batch is
            // its commands' own delivery, and a command that failed failed the
            // batch.
            command_results: Vec::new(),
            // A submission completes the batch: the device executor reports a
            // batch's calls together, never a part of one.
            done: true,
            timing: output.timing,
            // Device-side forward statistics are a worker's readout, not a
            // rank's own (`crate::executors`); a process reports none here.
            forward_stats: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use executor::batch::{BatchOutput, CallId as DeviceCallId, CallOutput as DeviceCall};
    use executor::completion::Outcome;
    use ipc::CallStatus;

    /// What one device call reports: `committed` tokens, the batches it stands
    /// at, and the log-probability of each when the deployment planned them.
    fn device_call(index: u64, committed: &[u32], logprobs: &[f32]) -> DeviceCall {
        let mut call = DeviceCall::completed(
            DeviceCallId(index),
            committed.to_vec(),
            executor::batch::RowState {
                step_index: 1,
                cache_computed_len: 3,
                cache_visible_len: 3,
            },
        );
        call.token_logprobs = logprobs.to_vec();
        call
    }

    /// The names the mechanism gave a batch's calls: one per call, in the
    /// batch's order.
    fn submitted(count: usize) -> Vec<(RequestKey, CallId)> {
        (0..count)
            .map(|index| {
                (
                    RequestKey::new(1, ipc::RequestId(index as u64 + 1), 0),
                    CallId::new(9, index as u32),
                )
            })
            .collect()
    }

    fn output(calls: Vec<DeviceCall>) -> Outcome<BatchOutput> {
        Outcome::Success(BatchOutput {
            calls,
            timing: TimingCounters {
                device_us: 11,
                ..TimingCounters::default()
            },
        })
    }

    /// A device reports a batch's calls in the order they finished; the
    /// mechanism files each under the name it gave that call, so a report's
    /// order never decides which request a completion advances.
    #[test]
    fn a_reports_calls_are_named_through_the_batch_that_was_submitted() {
        let calls = submitted(2);
        let report = BatchReport::from_output(
            4,
            output(vec![
                device_call(1, &[7], &[-0.5]),
                device_call(0, &[5], &[-0.25]),
            ]),
            &calls,
        )
        .expect("a device's report is read");
        assert_eq!(report.batch_id, 4);
        assert!(report.done);
        assert_eq!(report.calls.len(), 2);
        assert_eq!(report.calls[0].request_key, calls[0].0);
        assert_eq!(report.calls[0].call_id, calls[0].1);
        assert_eq!(report.calls[0].committed_tokens, vec![5]);
        assert_eq!(report.calls[1].request_key, calls[1].0);
        assert_eq!(report.calls[1].call_id, calls[1].1);
        assert_eq!(report.calls[1].committed_tokens, vec![7]);
        // A device names one log-probability per accepted token; the mechanism
        // records the accepted token as the one candidate, which is the shape
        // `generation` checks against the tokens it emits.
        assert_eq!(report.calls[0].token_logprobs.len(), 1);
        assert_eq!(report.calls[0].token_logprobs[0].len(), 1);
        assert_eq!(report.calls[0].token_logprobs[0][0].token_id, 5);
        assert_eq!(report.calls[0].token_logprobs[0][0].logprob, -0.25);
        assert_eq!(report.calls[0].token_logprobs[0][0].rank, 1);
        assert_eq!(report.timing.device_us, 11);
        assert!(report.command_results.is_empty());
    }

    /// A call the batch does not have is refused, not filed against a
    /// neighbour: the executor numbers calls by their index into `Batch.calls`.
    #[test]
    fn a_call_the_batch_does_not_have_is_refused() {
        let error =
            BatchReport::from_output(4, output(vec![device_call(2, &[7], &[])]), &submitted(2))
                .expect_err("a call index outside the batch is a broken report");
        assert!(
            format!("{error}").contains("of 2 the batch submitted"),
            "{error}"
        );
    }

    #[test]
    fn a_call_reported_twice_is_refused() {
        let error = BatchReport::from_output(
            4,
            output(vec![device_call(0, &[1], &[]), device_call(0, &[2], &[])]),
            &submitted(1),
        )
        .expect_err("one call reports once");
        assert!(format!("{error}").contains("twice"), "{error}");
    }

    /// A batch with calls whose report carries none is a runner that serves
    /// rows and reports nothing: the mechanism refuses it rather than advance a
    /// call's state on tokens nobody accounted for.
    #[test]
    fn a_serving_batch_that_reports_no_call_is_refused() {
        let error = BatchReport::from_output(4, output(Vec::new()), &submitted(2))
            .expect_err("silence is not a completion");
        assert!(format!("{error}").contains("no call reported"), "{error}");
    }

    /// A lifecycle batch carries no call and completes by reporting nothing:
    /// that is the one report silence is right for.
    #[test]
    fn a_batch_of_commands_completes_with_no_call() {
        let report = BatchReport::from_output(4, output(Vec::new()), &[])
            .expect("a lifecycle batch reports no call");
        assert!(report.calls.is_empty());
        assert!(report.command_results.is_empty());
        assert!(report.done);
    }

    /// A device that fails a batch fails it for the mechanism: the batch is
    /// not reported as complete with nothing in it.
    #[test]
    fn a_failed_batch_is_an_error() {
        let error = BatchReport::from_output(
            4,
            Outcome::Failed(std::sync::Arc::new(executor::Error::State("device gone"))),
            &submitted(1),
        )
        .expect_err("a failed batch is not a completion");
        assert!(format!("{error}").contains("batch 4 failed"), "{error}");
        let cancelled = BatchReport::from_output(4, Outcome::Cancelled, &submitted(1))
            .expect_err("a cancelled batch is not a completion");
        assert!(format!("{cancelled}").contains("cancelled"), "{cancelled}");
    }
}
