//! The scheduling policy's half of a generation: every scheduler prepares, switches to, commits
//! or rolls back the generation's policy package on its own thread, and responds to the control
//! thread, so a switch that installs no policy fails the update. A policy whose token limit
//! exceeds this rank's captured prefill buckets is refused with where each figure came from
//! ([`limit_refusal`]).

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use proto::module::PackageRevision;
use scheduler::{ModuleSource, PolicyPackage, PolicySource};

use super::{MainProcess, response};
use crate::rank::buckets::PrefillBound;

/// Logs one scheduler's response to a policy step when it comes, off the control thread.
///
/// A scheduler responds at its next batch boundary, and it reaches that boundary only while the
/// control thread drains what it sends: a control thread that waited here held the scheduler it
/// waited for, and a rollback under traffic stood still for the whole stall bound.
fn log_policy_response(
    receiver: crossbeam_channel::Receiver<std::result::Result<(), String>>,
    bound: Duration,
    what: &'static str,
) {
    std::thread::spawn(move || {
        if let Err(error) = response(receiver, bound, what) {
            tracing::error!(%error, "a scheduling policy step failed");
        }
    });
}

/// The argument a policy's step is bounded by: the figure the policy's hard constraints check
/// against the prefill buckets this rank captures.
pub(super) const TOKEN_LIMIT: &str = "max_num_batched_tokens";

/// The sentence a policy's token-limit refusal needs beside it.
///
/// The mechanism's own words are `max_num_batched_tokens is 8192, above the build config's 512`,
/// and every figure in them is right and none of them is actionable: `512` is the widest
/// prefill bucket this rank captures rather than a name, and `8192` is a figure no one in the
/// deployment chose when the policy's package names none. The refusal itself is the mechanism's
/// and is not changed here - what is added is where each figure came from, and the two changes
/// that would satisfy it.
///
/// A message that is not this refusal is returned unchanged, so every other policy failure reads
/// exactly as the mechanism wrote it.
pub(super) fn limit_refusal(error: &str, stated: Option<&str>, buckets: PrefillBound) -> String {
    if !error.contains(TOKEN_LIMIT) || !error.contains("above the build config's") {
        return error.to_owned();
    }

    // The policy's own figure: the mechanism renders `{name} is {limit}, above …`, and that figure
    // is what a deployment would have to capture buckets up to for the policy to fit.
    let declared = error
        .split(&format!("{TOKEN_LIMIT} is "))
        .nth(1)
        .map(|rest| {
            rest.chars()
                .take_while(char::is_ascii_digit)
                .collect::<String>()
        })
        .filter(|digits| !digits.is_empty());
    let origin = if buckets.stated {
        "the deployment's cuda_graph.prefill_token_sizes"
    } else {
        "the build's default buckets, which this deployment's launch file does not state"
    };

    // Whose figure the policy's is: a package that names no such argument states none, and the
    // policy then applies its own default - the number nobody in this deployment chose.
    let source_of_policy = match stated {
        Some(value) => format!("the deployment states {TOKEN_LIMIT} = {value}"),
        None => {
            format!("the deployment states no {TOKEN_LIMIT}, so this is the policy's own default")
        }
    };
    let bound = buckets.bound;
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
    /// The preparation's policy half: every scheduler prepares the generation's policy package
    /// on its own state, from the module the Build Server resolves under the store's wasm key,
    /// and responds on a channel; its response wakes the control thread.
    ///
    /// The build runs on the scheduler's own blocking thread, never here: a policy's module is
    /// compiled the first time it is asked for, and the control thread never waits on a job.
    pub(super) fn request_policy(
        &mut self,
        policy: &PackageRevision,
    ) -> Result<Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>>> {
        let package = PolicyPackage::from_arguments(
            policy.backend.clone(),
            policy.revision,
            &policy
                .arguments
                .iter()
                .map(|(name, value)| (name.clone(), value.to_string()))
                .collect(),
        )
        .map_err(|error| anyhow::anyhow!("the generation's policy package: {error}"))?;
        let source = Arc::new(ModuleSource::connect(&self.build_server));
        let source: Arc<dyn PolicySource + Send + Sync> = source;
        let mut responses = Vec::with_capacity(self.schedulers.len() + 1);
        // A node with several schedulers routes by the policy, so the control thread's route
        // instance is prepared beside theirs, on a thread of its own: the module compiles there.
        self.route_prepared = None;
        if self.schedulers.len() > 1 {
            let (respond, responded) = crossbeam_channel::bounded(1);
            let (prepared, candidate) = crossbeam_channel::bounded(1);
            let ranks = self.ranks();
            let ceilings = self.router.ceilings();
            let package = package.clone();
            let source = Arc::clone(&source);
            let wake = self.wake.clone();
            std::thread::Builder::new()
                .name("route-prepare".to_owned())
                .spawn(move || {
                    let built =
                        scheduler::Router::prepare(source.as_ref(), package, &ranks, &ceilings);
                    let _ = match built {
                        Ok(built) => {
                            let _ = prepared.send(built);
                            respond.send(Ok(()))
                        }
                        Err(error) => respond.send(Err(format!("the route policy: {error}"))),
                    };
                    wake.wake();
                })?;
            self.route_prepared = Some(candidate);
            responses.push(responded);
        }
        // The start marker of the preparation's policy half: the resolve that follows runs on a
        // scheduler's blocking thread and prints its own target and spans, and without this line
        // a log cannot tell a preparation that began from one that never did.
        tracing::info!(
            backend = %package.backend,
            revision = package.revision,
            schedulers = self.schedulers.len(),
            socket = %self.build_server.display(),
            "the generation's policy preparation has begun"
        );
        responses.extend(
            self.schedulers
                .iter()
                .map(|scheduler| scheduler.prepare_policy(package.clone(), Arc::clone(&source))),
        );
        Ok(responses)
    }

    /// Records on every scheduler the generation whose kernels the next step's batches execute.
    ///
    /// A step reads this cell once and stamps every batch it forms with it, and a runner serves
    /// a batch from the graphs captured for that number: a control thread that switched the
    /// devices without recording it would leave batches stamped with a generation no graph was
    /// captured for.
    pub(super) fn record_generation(&self, generation: u64) {
        for scheduler in &self.schedulers {
            scheduler.set_generation(generation);
        }
        // One line, because the figure a batch carries is invisible until a runner refuses it.
        tracing::info!(
            generation,
            schedulers = self.schedulers.len(),
            "the generation this process's batches carry is recorded"
        );
    }

    /// The switch's policy half: every scheduler's next batch boundary decides by the prepared
    /// policy. Returns the schedulers' responses, which arrive after this returns - a scheduler
    /// responds at its next batch boundary - and wake the control thread: a scheduler that
    /// refuses the switch holds no policy a preparation made ready, and fails the update rather
    /// than being left with no policy, which forms no batch.
    pub(super) fn switch_policy(
        &mut self,
        generation: u64,
    ) -> Vec<crossbeam_channel::Receiver<std::result::Result<(), String>>> {
        tracing::info!(
            generation,
            schedulers = self.schedulers.len(),
            "every scheduler is asked to switch its policy"
        );
        let mut responses: Vec<_> = self
            .schedulers
            .iter()
            .map(scheduler::SchedulerHandle::switch_policy)
            .collect();
        // The control thread's route instance switches here and now, and its response joins the
        // schedulers': a node that routes by the policy and holds no prepared route instance
        // fails the switch as a scheduler without a prepared policy does.
        if self.schedulers.len() > 1 {
            if let Some(candidate) = self
                .route_prepared
                .take()
                .and_then(|prepared| prepared.try_recv().ok())
            {
                self.router.adopt(candidate);
            }
            let ranks = self.ranks();
            let (respond, responded) = crossbeam_channel::bounded(1);
            let _ = respond.send(
                self.router
                    .switch(&ranks)
                    .map_err(|error| format!("the route policy: {error}")),
            );
            responses.push(responded);
        }
        responses
    }

    /// The commit's policy half: the trial ended in a commit, so the prepared policy is the
    /// serving one and the previous module is gone.
    pub(super) fn commit_policy(&mut self) {
        for scheduler in &self.schedulers {
            scheduler.commit_policy();
        }
        self.router.commit();
    }

    /// The rollback's policy half: the previous policy module decides again from the next
    /// boundary, and the candidate is dropped at it.
    pub(super) fn rollback_policy(&mut self) {
        let ranks = self.ranks();
        if let Err(error) = self.router.rollback(&ranks) {
            tracing::error!(%error, "the route policy's rollback failed");
        }
        for scheduler in &self.schedulers {
            log_policy_response(
                scheduler.rollback_policy(),
                self.watchdog.stall_bound(),
                "the policy rollback",
            );
        }
    }

    /// A preparation that failed or was abandoned discards its policy candidate, so the next
    /// update's package is the only one that can serve.
    pub(super) fn discard_policy(&mut self) {
        self.router.discard();
        self.route_prepared = None;
        // The candidate is dropped by the rollback every scheduler applies; a scheduler that
        // never switched has nothing to drop, and the stage it is asked for is the one that
        // says so.
        for scheduler in &self.schedulers {
            let _ = scheduler.rollback_policy();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The refusal a deployment reader acts on names both figures, where each came from, and the
    /// two changes that would satisfy it - and keeps the mechanism's own words in front of them.
    #[test]
    fn a_token_limit_refusal_names_the_buckets_and_where_the_policys_figure_came_from() {
        let buckets = PrefillBound {
            bound: 512,
            stated: true,
        };
        let mechanism = "max_num_batched_tokens is 8192, above the build config's 512";

        let refusal = limit_refusal(mechanism, None, buckets);

        assert!(
            refusal.starts_with(mechanism),
            "the mechanism's own refusal was not kept: {refusal}"
        );
        assert!(refusal.contains("up to 512 token(s)"), "{refusal}");
        assert!(
            refusal.contains("the deployment's cuda_graph.prefill_token_sizes"),
            "the buckets' origin is not named: {refusal}"
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

    /// A deployment that states the token bound is named as the one that chose it, and buckets
    /// the launch file does not state says so rather than reading as the deployment's own figure.
    #[test]
    fn a_token_limit_the_deployment_stated_is_attributed_to_the_deployment() {
        let buckets = PrefillBound {
            bound: 512,
            stated: false,
        };

        let refusal = limit_refusal(
            "max_num_batched_tokens is 4096, above the build config's 512",
            Some("4096"),
            buckets,
        );

        assert!(
            refusal.contains("the deployment states max_num_batched_tokens = 4096"),
            "{refusal}"
        );
        assert!(
            refusal.contains("the build's default buckets"),
            "unstated buckets read as stated: {refusal}"
        );
    }

    /// Every other policy failure is the mechanism's sentence and nothing else: a rendering that
    /// guessed at a message it does not know would put a wrong cause in a log.
    #[test]
    fn a_policy_failure_that_is_not_a_token_limit_is_returned_unchanged() {
        let buckets = PrefillBound {
            bound: 512,
            stated: true,
        };
        for mechanism in [
            "max_batch is 512, above the build config's 16",
            "the policy's trial refused every decision",
            "timeout is zero; a resident request would be starved from the first step",
        ] {
            assert_eq!(
                limit_refusal(mechanism, None, buckets),
                mechanism,
                "a message that is not the token refusal was rewritten"
            );
        }
    }
}
