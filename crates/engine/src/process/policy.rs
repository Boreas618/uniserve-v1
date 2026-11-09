//! The scheduling policy's half of the program-layer update: every scheduler prepares, switches,
//! commits or reverts the update's policy program on its own thread, and the control thread checks
//! that a first switchover installed one. A policy whose token limit exceeds this rank's captured
//! prefill buckets is refused with where each figure came from ([`limit_refusal`]).

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use ipc::UpdateFailure;
use program_server::Program;
use scheduler::{PolicyProgram, PolicySource, ProgramSource};
use worker::watchdog::Running;

use super::{MainProcess, answer};
use crate::rank::buckets::PrefillBound;

/// Logs one scheduler's answer to a policy step when it comes, off the control thread.
///
/// A scheduler answers at its next batch boundary, and it reaches that boundary only while the
/// control thread drains what it sends: a control thread that waited here held the scheduler it
/// waited for, and a switchover under traffic stood still for the whole stall bound.
fn report_policy_answer(
    receiver: crossbeam_channel::Receiver<std::result::Result<(), String>>,
    bound: Duration,
    what: &'static str,
) {
    std::thread::spawn(move || {
        if let Err(error) = answer(receiver, bound, what) {
            tracing::error!(%error, "a scheduling policy step failed");
        }
    });
}

/// How often a device that is busy delays the question asked of the policy slots: a probe that finds
/// a device running proves nothing about whether a policy serves, so it is asked again after this.
const POLICY_PROBE_INTERVAL: Duration = Duration::from_millis(250);

/// The argument a policy's step is bounded by: the figure the policy's hard constraints check
/// against this rank's catalogue of prefill buckets.
pub(super) const TOKEN_LIMIT: &str = "max_num_batched_tokens";

/// The sentence a policy's token-limit refusal needs beside it.
///
/// The mechanism's own words are `max_num_batched_tokens is 8192, above the build config's 512`,
/// and every figure in them is right and none of them is actionable: `512` is this rank's
/// catalogue of captured prefill buckets rather than a name, and `8192` is a figure no one in the
/// deployment chose when the policy's program names none. The refusal itself is the mechanism's
/// and is not changed here - what is added is where each figure came from, and the two changes
/// that would satisfy it.
///
/// A message that is not this refusal is answered unchanged, so every other policy failure reads
/// exactly as the mechanism wrote it.
pub(super) fn limit_refusal(error: &str, stated: Option<&str>, catalogue: PrefillBound) -> String {
    if !error.contains(TOKEN_LIMIT) || !error.contains("above the build config's") {
        return error.to_owned();
    }

    // The policy's own figure: the mechanism renders `{name} is {limit}, above …`, and that figure
    // is what a deployment would have to raise the catalogue to for the policy to fit.
    let declared = error
        .split(&format!("{TOKEN_LIMIT} is "))
        .nth(1)
        .map(|rest| {
            rest.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
        })
        .filter(|digits| !digits.is_empty());
    let origin = if catalogue.stated {
        "the deployment's cuda_graph.prefill_token_sizes"
    } else {
        "the build's default catalogue, which this deployment's launch file does not state"
    };

    // Whose figure the policy's is: a program that names no such argument states none, and the
    // policy then applies its own default - the number nobody in this deployment chose.
    let source_of_policy = match stated {
        Some(value) => format!("the deployment states {TOKEN_LIMIT} = {value}"),
        None => {
            format!("the deployment states no {TOKEN_LIMIT}, so this is the policy's own default")
        }
    };
    let bound = catalogue.bound;
    let ways = match &declared {
        Some(figure) => format!(
            "state {TOKEN_LIMIT} at or below {bound}, or capture prefill buckets up to {figure}"
        ),
        None => {
            format!("state {TOKEN_LIMIT} at or below {bound}, or capture wider prefill buckets")
        }
    };
    format!(
        "{error}. This rank captures prefill buckets up to {bound} token(s), from {origin}; \
         {source_of_policy}. Either {ways}."
    )
}

impl MainProcess {
    /// The switchover's policy check: a first switchover that installed no policy, said by name
    /// instead of waited for.
    ///
    /// A first switchover with no prepared candidate adopts no policy and answers `Ok(None)`; a
    /// scheduler with no policy forms no batch, so the epoch sits with nothing running while the
    /// daemon believes the cut-over succeeded - the same silence the missing startup report had.
    /// The policy slot is asked whether one serves, and the answer is read rather than waited for:
    /// a probe is due at once after the switchover and again one stall bound later, and the failure
    /// is only spoken when no device has a job in flight, so a busy scheduler cannot be mistaken
    /// for one that never adopted a policy.
    pub(super) fn advance_policy_check(
        &mut self,
        running: &[Running],
    ) -> Result<Option<UpdateFailure>> {
        let Some(due) = self.policy_probe_at else {
            return Ok(None);
        };
        let now = Instant::now();
        if now < due {
            return Ok(None);
        }
        self.policy_probe_at = None;

        // The scheduler's own signal, not a reading of its policy slot: `note_policy_serves` fires
        // where a policy actually adopts, so a probe cannot disagree with the fact it reports. A
        // probe that finds every scheduler serving clears the missing state and says nothing - a log
        // that states both that a policy serves and that none was installed is worse than either.
        let serving = self
            .schedulers
            .iter()
            .filter(|scheduler| *scheduler.serving().borrow())
            .count();
        if serving == self.schedulers.len() {
            self.policy_missing_since = None;
            return Ok(None);
        }
        if !running.is_empty() {
            // A device is busy, so the slots' silence is not evidence: look again next pass.
            self.policy_probe_at = Some(now + POLICY_PROBE_INTERVAL);
            return Ok(None);
        }
        match self.policy_missing_since {
            None => {
                self.policy_missing_since = Some(now);
                self.policy_probe_at = Some(now + self.watchdog.stall_bound());
                tracing::warn!(
                    serving,
                    schedulers = self.schedulers.len(),
                    "a switchover installed no policy on every scheduler; looking again in the watchdog's stall bound"
                );
                Ok(None)
            }
            Some(since) => {
                self.policy_missing_since = None;
                let failure = UpdateFailure::Policy {
                    error: format!(
                        "the epoch's first switchover installed no policy: {} of this node's {} \
                         schedulers serve one after {}s, and a scheduler with no policy forms no \
                         batch, so an admitted request cannot be planned",
                        serving,
                        self.schedulers.len(),
                        since.elapsed().as_secs()
                    ),
                };
                self.failure(failure)
            }
        }
    }

    /// The precopy's policy half: every scheduler prepares the update's policy program on its own
    /// state, from the module the Program Server resolves under the store's wasm key, and answers on
    /// a channel the serve pass polls.
    ///
    /// The build runs on the scheduler's own blocking thread, never here: a policy's module is
    /// compiled the first time it is asked for, and the control thread never waits on a job.
    pub(super) fn request_policy(
        &self,
        policy: &Program,
    ) -> Result<Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>>> {
        let program = PolicyProgram::from_arguments(
            policy.backend.clone(),
            policy.revision,
            &policy
                .arguments
                .iter()
                .map(|(name, value)| (name.clone(), value.to_string()))
                .collect(),
        )
        .map_err(|error| anyhow::anyhow!("the update's policy program: {error}"))?;
        let source = Arc::new(ProgramSource::connect(&self.program_server));
        let source: Arc<dyn PolicySource + Send + Sync> = source;
        // The start marker of the precopy's policy half: the resolve that follows runs on a
        // scheduler's blocking thread and prints its own target and spans, and without this line a
        // log cannot tell a precopy that began from one whose command was never sent.
        tracing::info!(
            backend = %program.backend,
            revision = program.revision,
            schedulers = self.schedulers.len(),
            socket = %self.program_server.display(),
            "this precopy's policy preparation has begun"
        );
        Ok(self
            .schedulers
            .iter()
            .map(|scheduler| scheduler.precopy_policy(program.clone(), Arc::clone(&source)))
            .collect())
    }

    /// The switchover's policy half: every scheduler's next batch boundary answers from the prepared
    /// policy. `step` is the boundary the epoch's schedulers agreed on, which only the scheduler
    /// node can name; the control thread does not choose one.
    /// Records on every scheduler the generation whose kernels the next step's batches serve.
    ///
    /// A step reads this cell once and stamps every batch it forms with it, and a runner serves a
    /// batch from the graphs captured for that number. A control thread that cut the devices over
    /// without recording it left the first batch of the deployment stamped `0`, and the runner
    /// answered `no graph of bucket ... in generation 0 or in a generation it replaces`: the graphs
    /// were captured under the generation the daemon named, and no batch ever asked for it.
    pub(super) fn record_generation(&self, generation: u64) {
        for scheduler in &self.schedulers {
            scheduler.set_generation(generation);
        }
        // One line, because the figure a batch carries is invisible until a runner refuses it: a run
        // whose first batch names a generation this line never stated is a run where something cut a
        // generation over without recording it.
        tracing::info!(
            generation,
            schedulers = self.schedulers.len(),
            "the generation this process's batches carry is recorded"
        );
    }

    pub(super) fn switchover_policy(&self, generation: u64, _step: Option<u64>) {
        // This states the candidate, not the serving policy: a scheduler adopts at its next batch
        // boundary, which is after this returns, so a "serving" figure here would be a second
        // statement of a fact `serving()` reports - and the two disagreed in a run. What this can
        // say is whether anything was prepared to adopt.
        let prepared = self
            .schedulers
            .iter()
            .filter(|scheduler| scheduler.policy_mut().prepared().is_some())
            .count();
        if prepared == self.schedulers.len() {
            tracing::info!(
                generation,
                prepared,
                schedulers = self.schedulers.len(),
                "every scheduler holds a prepared policy candidate"
            );
        } else {
            tracing::warn!(
                generation,
                prepared,
                schedulers = self.schedulers.len(),
                "a scheduler holds no prepared policy candidate: a switchover adopts nothing there"
            );
        }
        for scheduler in &self.schedulers {
            report_policy_answer(
                scheduler.switchover_policy(),
                self.watchdog.stall_bound(),
                "the policy switchover",
            );
        }
    }

    /// The commit's policy half: probation ended in a commit, so the prepared policy is the serving
    /// one and the previous module is gone.
    pub(super) fn commit_policy(&self, _generation: u64) {
        for scheduler in &self.schedulers {
            scheduler.commit_policy();
        }
    }

    /// The revert's policy half: the previous policy module answers again from the next boundary,
    /// and the candidate is dropped at it.
    pub(super) fn revert_policy(&self, _generation: u64) {
        for scheduler in &self.schedulers {
            report_policy_answer(
                scheduler.revert_policy(),
                self.watchdog.stall_bound(),
                "the policy revert",
            );
        }
    }

    /// A precopy that failed discards its policy candidate, so the next update's program is the only
    /// one that can serve.
    pub(super) fn discard_policy(&self) {
        // The candidate is dropped by the revert every scheduler applies; a scheduler that never
        // switched has nothing to drop, and the stage it is asked for is the one that says so.
        for scheduler in &self.schedulers {
            let _ = scheduler.revert_policy();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal a deployment reader acts on names both figures, where each came from, and the
    /// two changes that would satisfy it - and keeps the mechanism's own words in front of them.
    #[test]
    fn a_token_limit_refusal_names_the_catalogue_and_where_the_policys_figure_came_from() {
        let catalogue = PrefillBound {
            bound: 512,
            stated: true,
        };
        let mechanism = "max_num_batched_tokens is 8192, above the build config's 512";

        let refusal = limit_refusal(mechanism, None, catalogue);

        assert!(
            refusal.starts_with(mechanism),
            "the mechanism's own refusal was not kept: {refusal}"
        );
        assert!(refusal.contains("up to 512 token(s)"), "{refusal}");
        assert!(
            refusal.contains("the deployment's cuda_graph.prefill_token_sizes"),
            "the catalogue's origin is not named: {refusal}"
        );
        assert!(
            refusal.contains("the deployment states no max_num_batched_tokens"),
            "the policy's figure is not attributed to the policy: {refusal}"
        );
        assert!(
            refusal.contains("state max_num_batched_tokens at or below 512"),
            "the first way out is missing: {refusal}"
        );
        assert!(
            refusal.contains("capture prefill buckets up to 8192"),
            "the second way out is missing: {refusal}"
        );
    }

    /// A deployment that states the token bound is named as the one that chose it, and a catalogue
    /// the launch file does not state says so rather than reading as the deployment's own figure.
    #[test]
    fn a_token_limit_the_deployment_stated_is_attributed_to_the_deployment() {
        let catalogue = PrefillBound {
            bound: 512,
            stated: false,
        };

        let refusal = limit_refusal(
            "max_num_batched_tokens is 4096, above the build config's 512",
            Some("4096"),
            catalogue,
        );

        assert!(
            refusal.contains("the deployment states max_num_batched_tokens = 4096"),
            "{refusal}"
        );
        assert!(
            refusal.contains("the build's default catalogue"),
            "an unstated catalogue reads as stated: {refusal}"
        );
    }

    /// Every other policy failure is the mechanism's sentence and nothing else: a rendering that
    /// guessed at a message it does not know would put a wrong cause in a log.
    #[test]
    fn a_policy_failure_that_is_not_a_token_limit_is_answered_unchanged() {
        let catalogue = PrefillBound {
            bound: 512,
            stated: true,
        };
        for mechanism in [
            "max_batch is 512, above the build config's 16",
            "the policy's trial refused every decision",
            "timeout is zero; a resident request would be starved from the first step",
        ] {
            assert_eq!(
                limit_refusal(mechanism, None, catalogue),
                mechanism,
                "a message that is not the token refusal was rewritten"
            );
        }
    }
}
