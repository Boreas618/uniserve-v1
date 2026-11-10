//! The scheduling policy's half of the program-layer update: every scheduler prepares, switches,
//! commits or reverts the update's policy program on its own thread, and the control thread reads
//! each scheduler's answer to a switchover, so a switchover that installs no policy fails the
//! update. A policy whose token limit exceeds this rank's captured
//! prefill buckets is refused with where each figure came from ([`limit_refusal`]).

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use ipc::UpdateFailure;
use proto::program::Program;
use scheduler::{PolicyProgram, PolicySource, ProgramSource};

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
    /// The precopy's policy half: every scheduler prepares the update's policy program on its own
    /// state, from the module the Program Server resolves under the store's wasm key, and answers on
    /// a channel the serve pass polls.
    ///
    /// The build runs on the scheduler's own blocking thread, never here: a policy's module is
    /// compiled the first time it is asked for, and the control thread never waits on a job.
    pub(super) fn request_policy(
        &mut self,
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
        let mut answers = Vec::with_capacity(self.schedulers.len() + 1);
        // A node with several schedulers routes by the policy, so the control thread's route
        // instance is prepared beside theirs, on a thread of its own: the module compiles there.
        self.route_prepared = None;
        if self.schedulers.len() > 1 {
            let (answer, answered) = crossbeam_channel::bounded(1);
            let (prepared, candidate) = crossbeam_channel::bounded(1);
            let ranks = self.ranks();
            let ceilings = self.router.ceilings();
            let program = program.clone();
            let source = Arc::clone(&source);
            std::thread::Builder::new()
                .name("route-precopy".to_owned())
                .spawn(move || {
                    let built =
                        scheduler::Router::prepare(source.as_ref(), program, &ranks, &ceilings);
                    let _ = match built {
                        Ok(built) => {
                            let _ = prepared.send(built);
                            answer.send(Ok(()))
                        }
                        Err(error) => answer.send(Err(format!("the route policy: {error}"))),
                    };
                })?;
            self.route_prepared = Some(candidate);
            answers.push(answered);
        }
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
        answers.extend(
            self.schedulers
                .iter()
                .map(|scheduler| scheduler.precopy_policy(program.clone(), Arc::clone(&source))),
        );
        Ok(answers)
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

    ///
    /// A scheduler answers at its next batch boundary, after this returns, so the answers are kept
    /// and read by later serve passes ([`Self::read_policy_answers`]): a scheduler that refuses
    /// the switchover - it holds no policy a precopy prepared - fails the update rather than
    /// leaving a scheduler with no policy, which forms no batch.
    pub(super) fn switchover_policy(&mut self, generation: u64, _step: Option<u64>) {
        tracing::info!(
            generation,
            schedulers = self.schedulers.len(),
            "every scheduler is asked to switch over its policy"
        );
        let deadline = Instant::now() + self.watchdog.stall_bound();
        let answers: Vec<_> = self
            .schedulers
            .iter()
            .map(|scheduler| (scheduler.switchover_policy(), deadline))
            .collect();
        self.policy_answers.extend(answers);
        // The control thread's route instance switches here and now, and its answer joins the
        // schedulers': a node that routes by the policy and holds no prepared route instance
        // fails the switchover as a scheduler without a prepared policy does.
        if self.schedulers.len() > 1 {
            if let Some(candidate) = self
                .route_prepared
                .take()
                .and_then(|prepared| prepared.try_recv().ok())
            {
                self.router.adopt(candidate);
            }
            let ranks = self.ranks();
            let (answer, answered) = crossbeam_channel::bounded(1);
            let _ = answer.send(
                self.router
                    .switchover(&ranks)
                    .map_err(|error| format!("the route policy: {error}")),
            );
            self.policy_answers.push((answered, deadline));
        }
    }

    /// Reads the schedulers' answers to the last policy switchover that have arrived: one that
    /// failed, or that did not arrive within the stall bound, fails the update by name.
    pub(super) fn read_policy_answers(&mut self) -> Result<Option<UpdateFailure>> {
        let now = Instant::now();
        let mut failed = None;
        self.policy_answers
            .retain(|(answer, deadline)| match answer.try_recv() {
                Ok(Ok(())) => false,
                Ok(Err(error)) => {
                    failed.get_or_insert(error);
                    false
                }
                // A scheduler that stopped is reported by `watch_schedulers`.
                Err(crossbeam_channel::TryRecvError::Disconnected) => false,
                Err(crossbeam_channel::TryRecvError::Empty) if now >= *deadline => {
                    failed.get_or_insert_with(|| {
                        "a scheduler did not answer the policy switchover within the stall bound"
                            .to_owned()
                    });
                    false
                }
                Err(crossbeam_channel::TryRecvError::Empty) => true,
            });
        let Some(error) = failed else {
            return Ok(None);
        };
        // The answers of the other schedulers belong to the same switchover, which this failure
        // ends.
        self.policy_answers.clear();
        tracing::error!(%error, "a scheduler's policy switchover failed");
        self.failure(UpdateFailure::Policy {
            error: format!("the policy switchover failed: {error}"),
        })
    }

    /// The commit's policy half: probation ended in a commit, so the prepared policy is the serving
    /// one and the previous module is gone.
    pub(super) fn commit_policy(&mut self, _generation: u64) {
        for scheduler in &self.schedulers {
            scheduler.commit_policy();
        }
        self.router.commit();
    }

    /// The revert's policy half: the previous policy module answers again from the next boundary,
    /// and the candidate is dropped at it.
    pub(super) fn revert_policy(&mut self, _generation: u64) {
        let ranks = self.ranks();
        if let Err(error) = self.router.revert(&ranks) {
            tracing::error!(%error, "the route policy revert failed");
        }
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
    pub(super) fn discard_policy(&mut self) {
        self.router.discard();
        self.route_prepared = None;
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
