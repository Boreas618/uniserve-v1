//! What a scheduler reports (`specs/scheduler-design.md` §7.5): counting into
//! [`proto::report::SchedulerStats`] as it works, and `publish_stats`, which sends the report as
//! [`Output::Stats`] once a report period (`worker_config.report_period`) has passed.
//!
//! The report message is the accumulator. Counters add into it where the scheduler acts: a batch
//! submitted, a batch's output applied, a verification's drafts, an admission's queue wait and
//! hit, a call left out of a step because its blocks had not arrived, a transfer's answer.
//! Publishing takes the report
//! and starts the next interval from an empty one that keeps the lifetime values (the batches
//! submitted, the longest queue wait, each domain's credits in flight and their peak); the gauges
//! (requests running and waiting, the cache's usage) and the lifetime counts other parts of the
//! scheduler keep (its steps, its policy's calls and fuel) are read as the report leaves.
//!
//! A call is counted in the domain of its batch's phase: `prefill` for context ingestion
//! (prefills, refills, encodes), `decode` for decodes and verifications, `flow` for media work and
//! latent preparation. A batch's times are its workers' measurements (`BatchOutput::timing`),
//! once per batch in its one domain: the wait in the worker's queue, the device time, and the
//! copy and host time.

use std::sync::LazyLock;
use std::time::{Duration, Instant};

use crate::batch::{Batch, BatchOutput, CallOutput, CallStatus, ErrorCode, Phase};
use proto::report::{DomainSchedulerStats, ForwardStats, Pages, SchedulerStats};
use proto::request::RequestId;
use scheduling::PhaseClass;
use storage::runtime::{Completion, Tier};

use super::{Output, Scheduler};

/// The domain a call is counted in: the class of its batch's phase, with media work and the work
/// of no phase of its own (a latent preparation) counted together as a media request's flow.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum Domain {
    /// Context ingestion: prefills, refills and encodes.
    Prefill,
    /// Token progress: decodes and verifications.
    Decode,
    /// A media request's work and its latent preparation.
    Flow,
}

/// How often a scheduler reports: one report per second.
pub(crate) const STATS_PERIOD: Duration = Duration::from_secs(1);

impl Domain {
    /// The domain of the calls of a phase of `class`.
    const fn of(class: PhaseClass) -> Self {
        match class {
            PhaseClass::Context => Self::Prefill,
            PhaseClass::Token => Self::Decode,
            PhaseClass::Media | PhaseClass::Background => Self::Flow,
        }
    }

    /// The domain's name in the report (`DomainSchedulerStats::domain`).
    const fn name(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Flow => "flow",
        }
    }
}

/// The domains a report states, in its order: the domain of every phase, each once, in the order
/// of [`Domain`]. Derived from the phases, so a phase's calls always have their domain's entry.
static DOMAINS: LazyLock<Vec<Domain>> = LazyLock::new(|| {
    let mut domains: Vec<Domain> = Phase::ALL
        .iter()
        .map(|phase| Domain::of(phase.class()))
        .collect();
    domains.sort_unstable();
    domains.dedup();
    domains
});

/// The report a scheduler counts into from its start: every counter zero, with every domain in
/// place.
pub(super) fn empty() -> SchedulerStats {
    SchedulerStats {
        domain_stats: DOMAINS
            .iter()
            .map(|domain| DomainSchedulerStats {
                domain: domain.name().to_owned(),
                ..DomainSchedulerStats::default()
            })
            .collect(),
        ..SchedulerStats::default()
    }
}

impl Scheduler {
    /// Counts a batch handed to its executor: one more batch submitted, and its calls launched in
    /// its phase's domain, where each holds a credit until its output is applied. `formed` is the
    /// start of the step that formed the batch: each call is charged the scheduler's time from
    /// there to its submission.
    pub(super) fn count_submitted(&mut self, batch: &Batch, formed: Instant) {
        let calls = batch.requests.len() as u64;
        let waited = micros(formed.elapsed());
        add(&mut self.stats.step_counter, 1);

        let domain = &mut self.stats.domain_stats[domain_of(batch.phase)];
        add(&mut domain.launched_calls, calls);
        add(&mut domain.active_credits, calls);
        domain.peak_credits = domain.peak_credits.max(domain.active_credits);
        add(&mut domain.queue_us, calls.saturating_mul(waited));
    }

    /// Counts the output of a batch, once, as `update` applies it: its calls return their
    /// credits, failed calls and non-finite logits are counted, and the batch's time, its query
    /// tokens and its workers' forward counters join the interval.
    ///
    /// The batch's roundtrip is the time from its submission to its output as its workers
    /// measured it: the wait in its queue, its device time, and the copy and host time after it.
    pub(super) fn count_batch(&mut self, batch: &Batch, output: &BatchOutput) {
        let timing = output.timing;
        let calls = output.calls.len() as u64;
        let failed = output
            .calls
            .iter()
            .filter(|call| call.status == CallStatus::Error)
            .count() as u64;
        let faulted = output
            .calls
            .iter()
            .filter(|call| call.error_code == Some(ErrorCode::NumericalFault))
            .count() as u64;

        let domain = &mut self.stats.domain_stats[domain_of(batch.phase)];
        add(&mut domain.completed_calls, calls);
        add(&mut domain.error_calls, failed);
        add(&mut domain.reclaimed_credits, calls);
        domain.active_credits = domain.active_credits.saturating_sub(calls);
        add(&mut domain.completed_batches, 1);
        add(&mut domain.launch_us, timing.queued_us);
        add(&mut domain.device_us, timing.device_us);
        add(
            &mut domain.completion_us,
            timing.copy_us.saturating_add(timing.host_us),
        );

        // The batch's query tokens are those of its token calls; a media call computes none.
        let tokens = match batch.phase {
            Phase::Prefill | Phase::Refill | Phase::Decode | Phase::Verify => {
                batch.columns.tokens() as u64
            }
            _ => 0,
        };
        let roundtrip = [timing.device_us, timing.copy_us, timing.host_us]
            .into_iter()
            .fold(timing.queued_us, u64::saturating_add);

        let stats = &mut self.stats;
        add(&mut stats.numerical_faults, faulted);
        add(&mut stats.batch_count, 1);
        add(&mut stats.batch_tokens, tokens);
        add(&mut stats.worker_exec_us, timing.device_us);
        add(&mut stats.batch_roundtrip_us, roundtrip);
        if let Some(forward) = &output.forward_stats {
            stats
                .worker_forward_stats
                .get_or_insert_with(ForwardStats::default)
                .merge(forward);
        }
    }

    /// Counts the verification of row `row` of `batch`, which `call` answered: its drafts are the
    /// call's input tokens after the first, the accepted ones the leading committed tokens equal
    /// to them, the rejected ones the rest, and the committed ones every token `call` committed.
    /// A call with no drafts is a decode and counts nothing here.
    pub(super) fn count_speculation(&mut self, batch: &Batch, row: usize, call: &CallOutput) {
        // The call's input tokens span the query offsets of its rows: the committed token, then
        // the drafts.
        let rows = &batch.columns.rows;
        let offset = |row: u32| usize::try_from(*rows.query_offsets.get(row as usize)?).ok();
        let span = rows.calls.get(row).and_then(|call| {
            Some(offset(call.rows.start)?.saturating_add(1)..offset(call.rows.end)?)
        });
        let drafts = span
            .and_then(|span| batch.columns.input_ids.get(span))
            .unwrap_or_default();
        if drafts.is_empty() {
            return;
        }

        let accepted = call
            .tokens
            .iter()
            .zip(drafts)
            .take_while(|&(&token, &draft)| i64::from(token) == i64::from(draft))
            .count() as u64;
        let drafts = drafts.len() as u64;

        let forward = self
            .stats
            .worker_forward_stats
            .get_or_insert_with(ForwardStats::default);
        add(&mut forward.spec_verify_rows, 1);
        add(&mut forward.spec_verify_draft_tokens, drafts);
        add(&mut forward.spec_verify_accepted_tokens, accepted);
        add(&mut forward.spec_verify_rejected_tokens, drafts - accepted);
        add(
            &mut forward.spec_verify_committed_tokens,
            call.tokens.len() as u64,
        );
    }

    /// Counts the admission of request `id` at `at`: its wait from arrival, and for a token
    /// request the prefix query and its hit by tier (SD §4.4). The query is `prompt_len - 1`
    /// tokens; the hits are the tokens of its hit, and the host hits those of them loaded from
    /// the host.
    pub(super) fn count_admitted(&mut self, id: RequestId, at: Instant) {
        let Some(request) = self.requests.get(&id) else {
            return;
        };
        let waited = micros(at.saturating_duration_since(request.arrival));

        let stats = &mut self.stats;
        add(&mut stats.num_admitted_reqs, 1);
        add(&mut stats.queue_wait_us_total, waited);
        stats.max_queue_wait_us = stats.max_queue_wait_us.max(waited);

        if !matches!(request.input, proto::request::Request::Generation(_)) {
            return;
        }
        let hits = u64::from(request.num_cached_tokens);
        let host = u64::from(request.num_host_tokens).min(hits);

        let prefix = &mut stats.prefix_cache_stats;
        add(
            &mut prefix.base.queries,
            u64::from(request.num_prompt_tokens.saturating_sub(1)),
        );
        add(&mut prefix.base.hits, hits);
        add(&mut prefix.host_hits, host);
    }

    /// Counts a call left out of a step because blocks it writes had not arrived from the buffer
    /// pool (SD §7.6).
    pub(super) fn count_without_blocks(&mut self) {
        add(&mut self.stats.calls_without_blocks, 1);
    }

    /// Counts a running request preempted for room (SD §3.6): it ended as migratable for the
    /// daemon to resubmit.
    pub(super) fn count_preempted(&mut self) {
        add(&mut self.stats.preempted_requests, 1);
    }

    /// Counts `blocks` cached blocks evicted from a region of this scheduler's.
    pub(super) fn count_evicted(&mut self, blocks: usize) {
        add(&mut self.stats.evicted_blocks, blocks as u64);
    }

    /// Counts `blocks` draft-cache blocks a running request gave up for room (SD §3.6 "Draft
    /// release").
    pub(super) fn count_given_up_drafts(&mut self, blocks: u32) {
        add(&mut self.stats.given_up_draft_blocks, u64::from(blocks));
    }

    /// Counts the answer to a handoff's `Import` of pages (`super::handoff`): `Imported` is one
    /// more import and the blocks it filled, a run each; `Failed` is one more failed import.
    pub(super) fn count_import(&mut self, completion: &Completion) {
        match completion {
            Completion::Imported { runs, .. } => {
                add(&mut self.stats.imports, 1);
                add(&mut self.stats.imported_blocks, runs.len() as u64);
            }
            Completion::Failed => add(&mut self.stats.failed_imports, 1),
            _ => {}
        }
    }

    /// When the next report is due: the loop need not wake for statistics before it.
    pub(super) fn stats_due(&self) -> Instant {
        self.published + self.config.report_period
    }

    /// Sends the interval's report as [`Output::Stats`] once a report period has passed since the
    /// last, and starts the next interval. Before that, it does nothing.
    pub(super) fn publish_stats(&mut self, now: Instant) {
        if now < self.stats_due() {
            return;
        }

        // The interval's counts leave in the report; the lifetime values carry into the next.
        let mut report = std::mem::replace(&mut self.stats, empty());
        self.published = now;
        self.stats.step_counter = report.step_counter;
        self.stats.max_queue_wait_us = report.max_queue_wait_us;
        for (next, ended) in self.stats.domain_stats.iter_mut().zip(&report.domain_stats) {
            next.active_credits = ended.active_credits;
            next.peak_credits = ended.peak_credits;
        }

        // The gauges as they are now, and the lifetime counts the scheduler keeps elsewhere.
        report.num_running_reqs = self.running.len() as u64;
        report.num_waiting_reqs = self.waiting.len() as u64;
        report.prefix_cache_usage = self.cache_usage();
        report.queued_tokens = self.queued_tokens();
        report.device_pages = self.pages(&self.kv_regions());
        report.host_pages = self.pages(&self.host_regions());
        report.avg_queue_wait_us = report
            .queue_wait_us_total
            .checked_div(report.num_admitted_reqs)
            .unwrap_or(0);
        report.steps = self.step;
        // The interval ran under the step values the scheduler last read.
        report.revision = self.budget.revision;
        report.policy_calls = self.policy.calls;
        report.policy_fuel = self.policy.fuel;

        // The worker counters are absent when none changed in the interval.
        if report
            .worker_forward_stats
            .as_ref()
            .is_some_and(ForwardStats::is_empty)
        {
            report.worker_forward_stats = None;
        }

        // An engine that stopped listening takes no report; the scheduler goes on regardless.
        self.emit(Output::Stats(report));
    }

    /// The device regions of the KV cache, each once, as indices into the session's regions: the
    /// regions the groups take the blocks of the cache groups' classes from, but the state
    /// region of the recurrent-state class, whose units are checkpoints (`specs/buffer-pool.md`
    /// §14.1). Those are the classes of a fixed block size; encoder outputs, products and
    /// trajectories are sized per request (SD §4.4).
    pub(super) fn kv_regions(&self) -> Vec<usize> {
        let mut regions: Vec<usize> = self
            .kv
            .classes()
            .iter()
            .zip(0u8..)
            .filter(|(class, _)| class.units > 0 && !class.state_in_slot)
            .flat_map(|(_, class)| {
                self.groups
                    .iter()
                    .flat_map(move |group| group.regions(class).iter().map(|&region| region.into()))
            })
            .collect();
        regions.sort_unstable();
        regions.dedup();
        regions
    }

    /// The fraction of the KV cache's units that an allocation cannot take now, over its device
    /// regions: 1 less the regions' `available` over their capacity; 0 without a KV cache.
    /// The prompt tokens no call has computed yet, over the waiting and the running requests.
    fn queued_tokens(&self) -> u64 {
        let owed = |id: &RequestId| {
            self.requests.get(id).map_or(0, |request| {
                u64::from(
                    request
                        .num_prompt_tokens
                        .saturating_sub(request.num_computed_tokens),
                )
            })
        };
        self.waiting.iter().chain(&self.running).map(owed).sum()
    }

    /// The host regions of the session, as indices into its regions: the host tier's pages.
    fn host_regions(&self) -> Vec<usize> {
        let host =
            |(_, (_, region)): &(usize, &(u8, storage::runtime::Region))| region.tier == Tier::Host;
        self.kv
            .regions()
            .iter()
            .enumerate()
            .filter(host)
            .map(|(index, _)| index)
            .collect()
    }

    /// The pages of `regions` in use and their capacity: a region's units less those an
    /// allocation can take now.
    fn pages(&self, regions: &[usize]) -> Pages {
        let mut pages = Pages::default();
        for &region in regions {
            let capacity = self.kv.capacity(region);
            let available = self.kv.available(region);
            pages.capacity += u64::from(capacity);
            pages.used += u64::from(capacity.saturating_sub(available));
        }
        pages
    }

    fn cache_usage(&self) -> f64 {
        let (mut units, mut free) = (0_u64, 0_u64);
        for region in self.kv_regions() {
            let capacity = self.kv.capacity(region);
            let available = self.kv.available(region);
            units += u64::from(capacity);
            free += u64::from(available.min(capacity));
        }
        if units == 0 {
            return 0.0;
        }
        1.0 - free as f64 / units as f64
    }
}

/// The index in [`DOMAINS`], and in a report's `domain_stats`, of the domain a phase's calls are
/// counted in.
#[expect(
    clippy::expect_used,
    reason = "the domains are derived from every phase, so each phase's domain is among them"
)]
fn domain_of(phase: Phase) -> usize {
    let domain = Domain::of(phase.class());
    DOMAINS
        .iter()
        .position(|stated| *stated == domain)
        .expect("every phase's domain is a domain of the report")
}

/// Adds `value` to a counter, which saturates at the wire's limit as `ForwardStats::merge` does.
fn add(counter: &mut u64, value: u64) {
    *counter = counter.saturating_add(value);
}

/// A duration in whole microseconds, saturating.
fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::batch::{
        Batch, BatchId, BatchOutput, CallOutput, CallStatus, Columns, ErrorCode, Extent, Phase,
        Row, Shape, TimingCounters, Variant,
    };
    use crossbeam_channel::Receiver;
    use proto::report::{DomainSchedulerStats, ForwardStats, Pages, SchedulerStats};
    use proto::request::RequestId;
    use storage::runtime::GroupId;

    use super::super::Output;
    use super::super::tests::{Bench, bench};

    /// A scheduler of the shared bench whose outputs the test reads.
    fn scheduler() -> (Bench, Receiver<Output>) {
        let mut bench = bench(1024);
        let (output, outputs) = crossbeam_channel::unbounded();
        bench.scheduler.output = output;
        (bench, outputs)
    }

    /// The reports published so far.
    fn reports(outputs: &Receiver<Output>) -> Vec<SchedulerStats> {
        outputs
            .try_iter()
            .filter_map(|output| match output {
                Output::Stats(stats) => Some(stats),
                _ => None,
            })
            .collect()
    }

    /// A token batch of `phase` with one call per entry of `calls`, each the input ids of one row.
    fn token_batch(phase: Phase, calls: &[&[i32]]) -> Batch {
        let mut columns = Columns::default();
        columns.rows.query_offsets.push(0);
        for (index, inputs) in calls.iter().enumerate() {
            let row = index as u32;
            columns.rows.calls.push(Row {
                slot: row + 1,
                kind: phase,
                replay: 0..0,
                rows: row..row + 1,
            });
            columns.input_ids.extend_from_slice(inputs);
            columns
                .rows
                .positions
                .extend((0..inputs.len()).map(|position| position as i32));
            columns
                .rows
                .query_offsets
                .push(columns.input_ids.len() as i32);
        }
        Batch {
            id: BatchId(1),
            group: GroupId(0),
            phase,
            kernels: 0,
            policy: 0,
            revision: 0,
            copy_budget: proto::config::DEFAULT_COPY_BUDGET,
            shape: Shape {
                rows: calls.len() as u32,
                tokens: columns.input_ids.len() as u32,
                variant: Variant::DecodeGreedy,
                extent: Extent::None,
            },
            requests: (0..calls.len() as u64).map(RequestId).collect(),
            columns,
            media: Vec::new(),
            new_slots: Vec::new(),
        }
    }

    fn output(status: CallStatus, error_code: Option<ErrorCode>, tokens: &[u32]) -> CallOutput {
        CallOutput {
            status,
            error_code,
            error_message: None,
            tokens: tokens.to_vec(),
            step_index: 0,
            cache_computed_len: 0,
            cache_visible_len: 0,
            token_logprobs: Vec::new(),
            prompt_logprobs: Vec::new(),
            media_output: None,
            draft_token_ids: Vec::new(),
        }
    }

    fn domain<'a>(stats: &'a SchedulerStats, name: &str) -> &'a DomainSchedulerStats {
        stats
            .domain_stats
            .iter()
            .find(|domain| domain.domain == name)
            .unwrap()
    }

    #[test]
    fn a_batch_counts_its_calls_and_the_time_its_executor_measured_in_its_phase_domain() {
        let (mut bench, outputs) = scheduler();
        let scheduler = &mut bench.scheduler;
        let start = scheduler.published;

        let decode = token_batch(Phase::Decode, &[&[7], &[9]]);
        let prefill = token_batch(Phase::Prefill, &[&[1, 2, 3]]);
        scheduler.count_submitted(&decode, start);
        scheduler.count_submitted(&prefill, start);

        // One decode returns, with a numerical fault on its second call; the prefill is still in
        // flight when the interval ends.
        let timing = TimingCounters {
            queued_us: 10,
            device_us: 100,
            copy_us: 1,
            host_us: 5,
        };
        let returned = BatchOutput {
            calls: vec![
                output(CallStatus::Ok, None, &[11]),
                output(CallStatus::Error, Some(ErrorCode::NumericalFault), &[]),
            ],
            timing,
            forward_stats: None,
        };
        scheduler.count_batch(&decode, &returned);
        scheduler.publish_stats(start + Duration::from_secs(1));

        let [report] = reports(&outputs).try_into().unwrap();
        let names: Vec<&str> = report
            .domain_stats
            .iter()
            .map(|d| d.domain.as_str())
            .collect();
        assert_eq!(names, ["prefill", "decode", "flow"]);
        assert_eq!(
            domain(&report, "decode"),
            &DomainSchedulerStats {
                domain: "decode".to_owned(),
                active_credits: 0,
                peak_credits: 2,
                launched_calls: 2,
                completed_calls: 2,
                error_calls: 1,
                reclaimed_credits: 2,
                completed_batches: 1,
                launch_us: 10,
                device_us: 100,
                completion_us: 6,
                queue_us: domain(&report, "decode").queue_us,
                ..DomainSchedulerStats::default()
            }
        );
        let prefill = domain(&report, "prefill");
        assert_eq!(
            (
                prefill.launched_calls,
                prefill.active_credits,
                prefill.completed_batches
            ),
            (1, 1, 0)
        );

        assert_eq!(report.step_counter, 2);
        assert_eq!(report.batch_count, 1);
        assert_eq!(report.batch_tokens, 2);
        assert_eq!(report.worker_exec_us, 100);
        assert_eq!(report.batch_roundtrip_us, 116);
        assert_eq!(report.numerical_faults, 1);
        assert_eq!(report.worker_forward_stats, None);
    }

    #[test]
    fn a_report_leaves_once_a_second_and_the_next_interval_keeps_only_lifetime_values() {
        let (mut bench, outputs) = scheduler();
        bench.attach(&[]);
        let scheduler = &mut bench.scheduler;
        let start = scheduler.published;
        let id = RequestId(1);
        let arrival = scheduler.requests[&id].arrival;

        let batch = token_batch(Phase::Decode, &[&[7]]);
        scheduler.count_submitted(&batch, start);
        scheduler.count_admitted(id, arrival + Duration::from_millis(3));
        scheduler.count_without_blocks();
        let forward = ForwardStats {
            cuda_graph_replays: 4,
            ..ForwardStats::default()
        };
        let returned = BatchOutput {
            calls: vec![output(CallStatus::Ok, None, &[8])],
            timing: TimingCounters::default(),
            forward_stats: Some(forward.clone()),
        };
        scheduler.count_batch(&batch, &returned);

        // Not yet a second: nothing leaves.
        scheduler.publish_stats(start + Duration::from_millis(999));
        assert!(reports(&outputs).is_empty());

        let first = start + Duration::from_secs(1);
        scheduler.publish_stats(first);
        let [report] = reports(&outputs).try_into().unwrap();
        assert_eq!(report.num_admitted_reqs, 1);
        assert_eq!(report.queue_wait_us_total, 3_000);
        assert_eq!(report.avg_queue_wait_us, 3_000);
        assert_eq!(report.max_queue_wait_us, 3_000);
        assert_eq!(report.calls_without_blocks, 1);
        assert_eq!(report.worker_forward_stats, Some(forward));
        assert_eq!(domain(&report, "decode").peak_credits, 1);

        // The next report is due a second after the first, and carries the lifetime values alone.
        assert_eq!(scheduler.stats_due(), first + Duration::from_secs(1));
        scheduler.count_submitted(&batch, first);
        scheduler.publish_stats(first + Duration::from_secs(1));
        let [next] = reports(&outputs).try_into().unwrap();
        assert_eq!(next.step_counter, 2);
        assert_eq!(next.max_queue_wait_us, 3_000);
        assert_eq!(
            (
                next.num_admitted_reqs,
                next.queue_wait_us_total,
                next.avg_queue_wait_us
            ),
            (0, 0, 0)
        );
        assert_eq!(next.calls_without_blocks, 0);
        assert_eq!(next.batch_count, 0);
        assert_eq!(next.prefix_cache_stats.base.queries, 0);
        assert_eq!(next.worker_forward_stats, None);
        let decode = domain(&next, "decode");
        assert_eq!(
            (
                decode.launched_calls,
                decode.completed_calls,
                decode.active_credits
            ),
            (1, 0, 1)
        );
        assert_eq!(decode.peak_credits, 1);
    }

    #[test]
    fn a_verification_counts_its_drafts_and_those_its_committed_tokens_accepted() {
        let (mut bench, outputs) = scheduler();
        let scheduler = &mut bench.scheduler;
        let start = scheduler.published;

        // Row 0 verifies drafts 21, 22, 23 after token 20 and commits 21 and a correction; row 1
        // ends on a finish token among its drafts, 32, which it commits with 31; row 2 is a
        // decode, which carries no drafts.
        let batch = token_batch(
            Phase::Verify,
            &[&[20, 21, 22, 23], &[30, 31, 32, 33], &[40]],
        );
        scheduler.count_speculation(&batch, 0, &output(CallStatus::Ok, None, &[21, 99]));
        scheduler.count_speculation(&batch, 1, &output(CallStatus::Ok, None, &[31, 32]));
        scheduler.count_speculation(&batch, 2, &output(CallStatus::Ok, None, &[41]));
        scheduler.publish_stats(start + Duration::from_secs(1));

        let [report] = reports(&outputs).try_into().unwrap();
        let forward = report.worker_forward_stats.unwrap();
        assert_eq!(forward.spec_verify_rows, 2);
        assert_eq!(forward.spec_verify_draft_tokens, 6);
        assert_eq!(forward.spec_verify_accepted_tokens, 3);
        assert_eq!(forward.spec_verify_rejected_tokens, 3);
        assert_eq!(forward.spec_verify_committed_tokens, 4);
    }

    #[test]
    fn an_admission_counts_its_prefix_query_and_its_hit_by_tier() {
        let (mut bench, outputs) = scheduler();
        bench.attach(&[]);
        let scheduler = &mut bench.scheduler;
        let start = scheduler.published;
        let id = RequestId(1);

        // A prompt of 300 tokens with a hit of 256: 64 pinned on the device, 192 loaded from the
        // host.
        let request = scheduler.requests.get_mut(&id).unwrap();
        request.num_prompt_tokens = 300;
        request.num_cached_tokens = 256;
        request.num_host_tokens = 192;
        let arrival = request.arrival;
        scheduler.count_admitted(id, arrival);
        scheduler.publish_stats(start + Duration::from_secs(1));

        let [report] = reports(&outputs).try_into().unwrap();
        let prefix = report.prefix_cache_stats;
        assert_eq!(prefix.base.queries, 299);
        assert_eq!(prefix.base.hits, 256);
        assert_eq!(prefix.host_hits, 192);
    }

    #[test]
    fn a_report_states_the_queues_the_kv_cache_in_use_and_the_steps_and_policy_counts() {
        let (mut bench, outputs) = scheduler();
        bench.attach(&[]);
        let scheduler = &mut bench.scheduler;
        let start = scheduler.published;

        // The bench's KV regions are 0 and 1, 64 units each; region 2 holds sized products and
        // region 3 the slot rows, neither of which is the cache.
        *scheduler.kv.available_mut() = vec![32, 64, 0, 0];
        scheduler.running = vec![RequestId(1), RequestId(2)];
        scheduler.waiting.push_back(RequestId(3));
        // Request 1 runs with 200 of its 300 prompt tokens to compute; the bench holds no
        // request 2 or 3.
        let request = scheduler.requests.get_mut(&RequestId(1)).unwrap();
        request.num_prompt_tokens = 300;
        request.num_computed_tokens = 100;
        scheduler.step = 17;
        scheduler.policy.calls = 34;
        scheduler.policy.fuel = 5_000;
        scheduler.publish_stats(start + Duration::from_secs(1));

        let [report] = reports(&outputs).try_into().unwrap();
        assert_eq!(scheduler.kv_regions(), [0, 1]);
        assert_eq!(report.prefix_cache_usage, 0.25);
        assert_eq!((report.num_running_reqs, report.num_waiting_reqs), (2, 1));
        assert_eq!(report.queued_tokens, 200);
        assert_eq!(
            report.device_pages,
            Pages {
                used: 32,
                capacity: 128
            }
        );
        assert_eq!(report.host_pages, Pages::default());
        assert_eq!(report.steps, 17);
        assert_eq!((report.policy_calls, report.policy_fuel), (34, 5_000));
    }
}
