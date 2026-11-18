//! Statistics and warm-up (SD §8.1): the report a scheduler sends once a second (SD §2.5
//! `Output::Stats`, §7.5 "The statistics message", "Statistics rules") and the warm-up requests
//! the engine adds when admission opens (SD §3.2, §7.2 "Warm-up requests").
//!
//! A report's counters count the second it covers (admissions, their queue waits and prefix
//! hits, the batches whose outputs were applied and their tokens), its gauges state the scheduler
//! as it is when the report leaves (requests running and waiting), and its lifetime values carry
//! from report to report (the batches submitted, the longest queue wait, the steps). A warm-up
//! request is the engine's, under an id the scheduler gives it: it is served and counted like any
//! other request, and its events are dropped.

use engine::batch::{Batch, Phase};
use proto::report::SchedulerStats;
use proto::request::{CachePolicy, FinishReason, GenerationRequest, RequestId, SamplingParams};

use crate::harness::{Deployment, sharing, token};

/// The calls a report counts in `domain` as launched and as completed.
fn domain(report: &SchedulerStats, domain: &str) -> (u64, u64) {
    let stats = report.domain_stats.iter().find(|of| of.domain == domain);
    stats.map_or((0, 0), |stats| {
        (stats.launched_calls, stats.completed_calls)
    })
}

/// Each report counts the second it covers and states the gauges as they are when it leaves:
/// the first second admits a request and a reader of its prompt (queries of `prompt_len - 1`
/// tokens each, the reader's hit of three blocks) and leaves the reader running and a third
/// request waiting; the next second admits nothing, and keeps the longest queue wait; the last
/// admits the third. Over every report, the batches and their tokens are each counted once, and
/// the batches submitted are a lifetime count (SD §7.5).
#[test]
fn each_report_counts_its_second_and_states_the_gauges_as_they_are() {
    let mut harness = Deployment::tokens().slots(1).build();
    // The second before any request.
    let idle = harness.next_report();
    assert_eq!(
        (
            idle.num_admitted_reqs,
            idle.num_running_reqs,
            idle.batch_count
        ),
        (0, 0, 0)
    );
    let domains: Vec<&str> = idle
        .domain_stats
        .iter()
        .map(|stats| stats.domain.as_str())
        .collect();
    assert_eq!(domains, ["prefill", "decode", "flow"]);

    let first = harness.submit(token(1, 13, 4));
    harness.settle(50);
    let reader = harness.submit(sharing(2, 1, 13, 200));
    let waiting = harness.submit(token(3, 9, 2));
    harness.run_until(20, |harness| !harness.tokens(reader).is_empty());
    assert_eq!(harness.tokens(first).len(), 4);

    let busy = harness.next_report();
    assert_eq!(busy.num_admitted_reqs, 2);
    assert!(busy.queue_wait_us_total > 0);
    assert_eq!(busy.avg_queue_wait_us, busy.queue_wait_us_total / 2);
    assert!(busy.max_queue_wait_us >= busy.avg_queue_wait_us);
    assert!(busy.max_queue_wait_us <= busy.queue_wait_us_total);
    let prefix = &busy.prefix_cache_stats;
    assert_eq!((prefix.base.queries, prefix.base.hits), (12 + 12, 12));
    assert_eq!(prefix.host_hits, 0);
    assert_eq!((busy.num_running_reqs, busy.num_waiting_reqs), (1, 1));
    assert_eq!(domain(&busy, "prefill"), (2, 2), "the two prefills");
    assert!(domain(&busy, "decode").0 > 0);
    assert_eq!(domain(&busy, "flow"), (0, 0));
    assert!(busy.steps > idle.steps);

    // A second that admits nothing counts no admission and no query; the lifetime values carry.
    let quiet = harness.next_report();
    assert_eq!(quiet.num_admitted_reqs, 0);
    assert_eq!(
        (
            quiet.prefix_cache_stats.base.queries,
            quiet.queue_wait_us_total
        ),
        (0, 0)
    );
    assert_eq!(quiet.max_queue_wait_us, busy.max_queue_wait_us);
    assert_eq!((quiet.num_running_reqs, quiet.num_waiting_reqs), (1, 1));
    assert_eq!(domain(&quiet, "prefill").0, 0);
    assert!(quiet.batch_count > 0, "the reader's decodes");
    assert!(quiet.step_counter >= busy.step_counter + quiet.batch_count);

    harness.apply(engine::scheduler::Message::Cancel(
        reader,
        FinishReason::Cancelled,
    ));
    harness.settle(100);
    assert_eq!(harness.tokens(waiting).len(), 2);
    let last = harness.next_report();
    assert_eq!(last.num_admitted_reqs, 1);
    assert_eq!(last.prefix_cache_stats.base.queries, 8);
    assert_eq!((last.num_running_reqs, last.num_waiting_reqs), (0, 0));

    // Every batch was answered before the last report: each is counted in one second.
    let batches = harness.batches();
    let reports = harness.reports();
    let counted: u64 = reports.iter().map(|report| report.batch_count).sum();
    let tokens: u64 = reports.iter().map(|report| report.batch_tokens).sum();
    let token_phase = |phase: Phase| {
        matches!(
            phase,
            Phase::Prefill | Phase::Decode | Phase::Verify | Phase::Refill
        )
    };
    let computed: u64 = batches
        .iter()
        .filter(|batch| token_phase(batch.phase))
        .map(|batch| batch.columns.tokens() as u64)
        .sum();
    assert_eq!(counted, batches.len() as u64);
    assert_eq!(tokens, computed);
    assert_eq!(last.step_counter, batches.len() as u64);
}

/// The warm-up request the engine hands a scheduler (SD §3.2): the prompt `[bos]`, greedy,
/// ignoring the end of text, seed 0, cache read and write off, two tokens.
fn warm_up_request() -> GenerationRequest {
    GenerationRequest {
        prompt_token_ids: vec![1],
        sampling: SamplingParams {
            temperature: 0.0,
            ignore_eos: true,
            seed: Some(0),
            ..SamplingParams::default()
        },
        cache: CachePolicy {
            read: false,
            write: false,
            ..CachePolicy::default()
        },
        ..token(0, 1, 2)
    }
}

/// The id the scheduler gave the warm-up request whose prefill is the first batch it formed.
fn warm_up_id(batches: &[Batch]) -> RequestId {
    batches[0].requests[0]
}

/// A warm-up request is served like any request, and gives back what it held, but no event of it
/// reaches the daemon; it is counted as admitted like any request (SD §3.2, §7.2 "Warm-up
/// requests", §9.2 question 4). A request after it is served as usual.
#[test]
fn a_warm_up_request_is_served_with_its_events_dropped() {
    let mut harness = Deployment::tokens().build();
    harness.warm_up(warm_up_request());
    harness.run_until(30, |harness| !harness.batches().is_empty());
    let id = warm_up_id(&harness.batches());
    harness.run_until(30, |harness| {
        let phases: Vec<Phase> = harness
            .batches_of(id)
            .iter()
            .map(|batch| batch.phase)
            .collect();
        phases.contains(&Phase::Prefill) && phases.contains(&Phase::Decode)
    });
    harness.run_until(30, |harness| harness.held() == 0);

    let events = harness.sent().filter(|(of, _)| *of == id).count();
    assert_eq!(events, 0, "no event of a warm-up request is sent");
    let batches = harness.batches_of(id).len();
    assert_eq!(
        batches, 2,
        "its prefill samples one token and one decode the other"
    );
    let report = harness.next_report();
    let admitted: u64 = harness
        .reports()
        .iter()
        .map(|report| report.num_admitted_reqs)
        .sum();
    assert_eq!(admitted, 1);
    assert_eq!(report.num_running_reqs, 0);
    assert_eq!(harness.batches_of(id).len(), batches, "it ended");

    // A warm-up request has no terminal event, so the steps end with the request after it.
    let after = harness.submit(token(id.0 + 1, 10, 3));
    harness.run_until(100, |harness| {
        harness.terminal(after).is_some() && harness.held() == 0
    });
    assert_eq!(harness.tokens(after).len(), 3);
}

/// A request of the daemon that takes the id of the warm-up request the scheduler holds is a
/// request of its own: it waits until the warm-up request ended, then is served whole, with every
/// event of its own and none of the warm-up's (SD §3.2).
#[test]
fn a_request_of_a_warm_up_request_s_id_waits_for_its_end() {
    let mut harness = Deployment::tokens().build();
    harness.warm_up(warm_up_request());
    harness.run_until(30, |harness| !harness.batches().is_empty());
    let id = warm_up_id(&harness.batches());

    let client = harness.submit(token(id.0, 10, 3));
    harness.run_until(100, |harness| {
        harness.terminal(client).is_some() && harness.held() == 0
    });

    // The warm-up's prefill and decode run before the request's first call, its prefill.
    let phases: Vec<Phase> = harness
        .batches_of(id)
        .iter()
        .map(|batch| batch.phase)
        .collect();
    assert_eq!(phases[..3], [Phase::Prefill, Phase::Decode, Phase::Prefill]);
    assert_eq!(harness.tokens(client).len(), 3, "the request's own tokens");
    assert!(harness.finished_with(client, FinishReason::MaxTokens));
}
