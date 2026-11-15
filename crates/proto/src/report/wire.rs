//! The request interface's wire form of a scheduler's statistics (`request.proto`'s
//! `SchedulerStats` and the messages it holds), and the one conversion between it and the
//! snapshot types of this module.

use crate::request::wire;

use super::{
    BaseCacheStats, DomainSchedulerStats, FenceHistogram, ForwardStats, PrefixCacheStats,
    SchedulerStats,
};

impl From<&SchedulerStats> for wire::SchedulerStats {
    fn from(stats: &SchedulerStats) -> Self {
        Self {
            num_running_reqs: stats.num_running_reqs,
            num_waiting_reqs: stats.num_waiting_reqs,
            step_counter: stats.step_counter,
            current_wave: stats.current_wave,
            prefix_cache_usage: stats.prefix_cache_usage,
            num_admitted_reqs: stats.num_admitted_reqs,
            avg_queue_wait_us: stats.avg_queue_wait_us,
            queue_wait_us_total: stats.queue_wait_us_total,
            max_queue_wait_us: stats.max_queue_wait_us,
            prefix_cache_stats: Some((&stats.prefix_cache_stats).into()),
            worker_forward_stats: stats.worker_forward_stats.as_ref().map(Into::into),
            worker_exec_us: stats.worker_exec_us,
            batch_roundtrip_us: stats.batch_roundtrip_us,
            batch_count: stats.batch_count,
            batch_tokens: stats.batch_tokens,
            numerical_faults: stats.numerical_faults,
            domain_stats: stats.domain_stats.iter().map(Into::into).collect(),
            calls_without_blocks: stats.calls_without_blocks,
            transfers: stats.transfers,
            failed_transfers: stats.failed_transfers,
            transferred_blocks: stats.transferred_blocks,
            preempted_requests: stats.preempted_requests,
            evicted_blocks: stats.evicted_blocks,
            given_up_draft_blocks: stats.given_up_draft_blocks,
            steps: stats.steps,
            policy_calls: stats.policy_calls,
            policy_fuel: stats.policy_fuel,
        }
    }
}

impl From<wire::SchedulerStats> for SchedulerStats {
    fn from(stats: wire::SchedulerStats) -> Self {
        Self {
            num_running_reqs: stats.num_running_reqs,
            num_waiting_reqs: stats.num_waiting_reqs,
            step_counter: stats.step_counter,
            current_wave: stats.current_wave,
            prefix_cache_usage: stats.prefix_cache_usage,
            num_admitted_reqs: stats.num_admitted_reqs,
            avg_queue_wait_us: stats.avg_queue_wait_us,
            queue_wait_us_total: stats.queue_wait_us_total,
            max_queue_wait_us: stats.max_queue_wait_us,
            // An absent message is the empty snapshot of its counters, as Protobuf states every
            // absent scalar as zero.
            prefix_cache_stats: stats.prefix_cache_stats.unwrap_or_default().into(),
            worker_forward_stats: stats.worker_forward_stats.map(Into::into),
            worker_exec_us: stats.worker_exec_us,
            batch_roundtrip_us: stats.batch_roundtrip_us,
            batch_count: stats.batch_count,
            batch_tokens: stats.batch_tokens,
            numerical_faults: stats.numerical_faults,
            domain_stats: stats.domain_stats.into_iter().map(Into::into).collect(),
            calls_without_blocks: stats.calls_without_blocks,
            transfers: stats.transfers,
            failed_transfers: stats.failed_transfers,
            transferred_blocks: stats.transferred_blocks,
            preempted_requests: stats.preempted_requests,
            evicted_blocks: stats.evicted_blocks,
            given_up_draft_blocks: stats.given_up_draft_blocks,
            steps: stats.steps,
            policy_calls: stats.policy_calls,
            policy_fuel: stats.policy_fuel,
        }
    }
}

impl From<&PrefixCacheStats> for wire::PrefixCacheStats {
    fn from(stats: &PrefixCacheStats) -> Self {
        Self {
            requests: stats.base.requests,
            queries: stats.base.queries,
            hits: stats.base.hits,
            host_hits: stats.host_hits,
            peer_hits: stats.peer_hits,
            preempted_requests: stats.preempted_requests,
            preempted_queries: stats.preempted_queries,
            preempted_hits: stats.preempted_hits,
        }
    }
}

impl From<wire::PrefixCacheStats> for PrefixCacheStats {
    fn from(stats: wire::PrefixCacheStats) -> Self {
        Self {
            base: BaseCacheStats {
                requests: stats.requests,
                queries: stats.queries,
                hits: stats.hits,
            },
            host_hits: stats.host_hits,
            peer_hits: stats.peer_hits,
            preempted_requests: stats.preempted_requests,
            preempted_queries: stats.preempted_queries,
            preempted_hits: stats.preempted_hits,
        }
    }
}

impl From<&DomainSchedulerStats> for wire::DomainSchedulerStats {
    fn from(stats: &DomainSchedulerStats) -> Self {
        Self {
            domain: stats.domain.clone(),
            active_credits: stats.active_credits,
            peak_credits: stats.peak_credits,
            launched_calls: stats.launched_calls,
            completed_calls: stats.completed_calls,
            predicated_calls: stats.predicated_calls,
            error_calls: stats.error_calls,
            backpressure_events: stats.backpressure_events,
            reclaimed_credits: stats.reclaimed_credits,
            completed_batches: stats.completed_batches,
            queue_us: stats.queue_us,
            launch_us: stats.launch_us,
            device_us: stats.device_us,
            completion_us: stats.completion_us,
        }
    }
}

impl From<wire::DomainSchedulerStats> for DomainSchedulerStats {
    fn from(stats: wire::DomainSchedulerStats) -> Self {
        Self {
            domain: stats.domain,
            active_credits: stats.active_credits,
            peak_credits: stats.peak_credits,
            launched_calls: stats.launched_calls,
            completed_calls: stats.completed_calls,
            predicated_calls: stats.predicated_calls,
            error_calls: stats.error_calls,
            backpressure_events: stats.backpressure_events,
            reclaimed_credits: stats.reclaimed_credits,
            completed_batches: stats.completed_batches,
            queue_us: stats.queue_us,
            launch_us: stats.launch_us,
            device_us: stats.device_us,
            completion_us: stats.completion_us,
        }
    }
}

impl From<&ForwardStats> for wire::ForwardStats {
    fn from(stats: &ForwardStats) -> Self {
        let counts = |map: &std::collections::BTreeMap<String, u64>| {
            map.iter()
                .map(|(name, count)| (name.clone(), *count))
                .collect()
        };
        Self {
            mode_counts: counts(&stats.mode_counts),
            mode_tokens: counts(&stats.mode_tokens),
            mode_us: counts(&stats.mode_us),
            stream_us: counts(&stats.stream_us),
            blocking_calls: counts(&stats.blocking_calls),
            fence_us: stats
                .fence_us
                .iter()
                .map(|(name, histogram)| (name.clone(), histogram.into()))
                .collect(),
            fence_records_lost: stats.fence_records_lost,
            profile: counts(&stats.profile),
            attention_launches: stats.attention_launches,
            attention_us: stats.attention_us,
            attention_backend_counts: counts(&stats.attention_backend_counts),
            cuda_graph_captures: stats.cuda_graph_captures,
            cuda_graph_replays: stats.cuda_graph_replays,
            cuda_graph_misses: stats.cuda_graph_misses,
            cuda_graph_fallbacks: stats.cuda_graph_fallbacks,
            cuda_graph_unpadded_tokens: stats.cuda_graph_unpadded_tokens,
            cuda_graph_padded_tokens: stats.cuda_graph_padded_tokens,
            cuda_graph_runtime_mode_counts: counts(&stats.cuda_graph_runtime_mode_counts),
            text_decode_token_relay_hits: stats.text_decode_token_relay_hits,
            text_decode_token_relay_misses: stats.text_decode_token_relay_misses,
            text_decode_position_relay_hits: stats.text_decode_position_relay_hits,
            text_decode_position_relay_misses: stats.text_decode_position_relay_misses,
            spec_verify_rows: stats.spec_verify_rows,
            spec_verify_draft_tokens: stats.spec_verify_draft_tokens,
            spec_verify_accepted_tokens: stats.spec_verify_accepted_tokens,
            spec_verify_rejected_tokens: stats.spec_verify_rejected_tokens,
            spec_verify_committed_tokens: stats.spec_verify_committed_tokens,
            spec_verify_path_counts: counts(&stats.spec_verify_path_counts),
        }
    }
}

impl From<wire::ForwardStats> for ForwardStats {
    fn from(stats: wire::ForwardStats) -> Self {
        Self {
            mode_counts: stats.mode_counts.into_iter().collect(),
            mode_tokens: stats.mode_tokens.into_iter().collect(),
            mode_us: stats.mode_us.into_iter().collect(),
            stream_us: stats.stream_us.into_iter().collect(),
            blocking_calls: stats.blocking_calls.into_iter().collect(),
            fence_us: stats
                .fence_us
                .into_iter()
                .map(|(name, histogram)| (name, histogram.into()))
                .collect(),
            fence_records_lost: stats.fence_records_lost,
            profile: stats.profile.into_iter().collect(),
            attention_launches: stats.attention_launches,
            attention_us: stats.attention_us,
            attention_backend_counts: stats.attention_backend_counts.into_iter().collect(),
            cuda_graph_captures: stats.cuda_graph_captures,
            cuda_graph_replays: stats.cuda_graph_replays,
            cuda_graph_misses: stats.cuda_graph_misses,
            cuda_graph_fallbacks: stats.cuda_graph_fallbacks,
            cuda_graph_unpadded_tokens: stats.cuda_graph_unpadded_tokens,
            cuda_graph_padded_tokens: stats.cuda_graph_padded_tokens,
            cuda_graph_runtime_mode_counts: stats
                .cuda_graph_runtime_mode_counts
                .into_iter()
                .collect(),
            text_decode_token_relay_hits: stats.text_decode_token_relay_hits,
            text_decode_token_relay_misses: stats.text_decode_token_relay_misses,
            text_decode_position_relay_hits: stats.text_decode_position_relay_hits,
            text_decode_position_relay_misses: stats.text_decode_position_relay_misses,
            spec_verify_rows: stats.spec_verify_rows,
            spec_verify_draft_tokens: stats.spec_verify_draft_tokens,
            spec_verify_accepted_tokens: stats.spec_verify_accepted_tokens,
            spec_verify_rejected_tokens: stats.spec_verify_rejected_tokens,
            spec_verify_committed_tokens: stats.spec_verify_committed_tokens,
            spec_verify_path_counts: stats.spec_verify_path_counts.into_iter().collect(),
        }
    }
}

impl From<&FenceHistogram> for wire::FenceHistogram {
    fn from(histogram: &FenceHistogram) -> Self {
        Self {
            buckets: histogram.buckets.clone(),
            sum_us: histogram.sum_us,
        }
    }
}

impl From<wire::FenceHistogram> for FenceHistogram {
    fn from(histogram: wire::FenceHistogram) -> Self {
        Self {
            buckets: histogram.buckets,
            sum_us: histogram.sum_us,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A snapshot crosses the request interface and reads back as itself: every counter, the
    /// domains in order, the worker counters with their fence histograms, and their absence.
    #[test]
    fn a_snapshot_reads_back_as_itself() {
        let mut fence = FenceHistogram::default();
        fence.observe(40);
        fence.observe(9_000);
        let forward = ForwardStats {
            mode_counts: [("graph".to_owned(), 7)].into(),
            fence_us: [("decode/step".to_owned(), fence)].into(),
            spec_verify_accepted_tokens: 3,
            ..ForwardStats::default()
        };
        let stats = SchedulerStats {
            num_running_reqs: 4,
            prefix_cache_usage: 0.25,
            prefix_cache_stats: PrefixCacheStats {
                base: BaseCacheStats {
                    requests: 2,
                    queries: 100,
                    hits: 60,
                },
                host_hits: 20,
                peer_hits: 5,
                ..PrefixCacheStats::default()
            },
            worker_forward_stats: Some(forward),
            domain_stats: vec![
                DomainSchedulerStats {
                    domain: "prefill".to_owned(),
                    launched_calls: 3,
                    ..DomainSchedulerStats::default()
                },
                DomainSchedulerStats {
                    domain: "decode".to_owned(),
                    device_us: 11,
                    ..DomainSchedulerStats::default()
                },
            ],
            policy_fuel: 9,
            ..SchedulerStats::default()
        };

        for stats in [stats.clone(), SchedulerStats::default()] {
            let encoded = prost::Message::encode_to_vec(&wire::SchedulerStats::from(&stats));
            let decoded: wire::SchedulerStats = prost::Message::decode(encoded.as_slice()).unwrap();
            assert_eq!(SchedulerStats::from(decoded), stats);
        }
    }
}
