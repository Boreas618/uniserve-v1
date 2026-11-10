//! The pools a fake worker states, and what a handle over one is built with.
//!
//! The mechanism builds its whole view of a worker from [`Inventory`]: its ranks, their call kinds
//! and request rows, and the layout the deployment launched the pool with. A case states its pool
//! here and the handle derives the rest exactly as it does for a launch.

use ipc::execution::{Batches, Inventory, PoolLayout};
use ipc::{MainProcessEndpoint, MainProcessInfo};
use proto::config::{BlockConfig, EngineArgs};
use scheduler::SchedulerConfig;

/// One rank of a text pool on `host`: prefill and decode, eight request rows, the batch bounds a
/// launch states, and the paged KV pool every token-serving rank advertises.
pub fn rank(host: &str) -> MainProcessInfo {
    MainProcessInfo {
        endpoint: MainProcessEndpoint {
            host: host.to_owned(),
            address_space: format!("{host}-space"),
            incarnation: format!("{host}-rank"),
        },
        supported_calls: vec![
            ipc::CallKind::Forward(ipc::ForwardMode::Prefill),
            ipc::CallKind::Forward(ipc::ForwardMode::Decode),
        ],
        request_slots: 8,
        max_batch_calls: 16,
        max_batch_tokens: 512,
        // The prefix cache is `MainProcessInfo::default`'s: a rank without one is refused every
        // token request at admission, which is the rule for a media-only rank and not this one.
        buffer_pool_bytes: 1 << 20,
        ..MainProcessInfo::default()
    }
}

/// A text pool named `group` of one [`rank`], one batch in flight per device, serving the default
/// block on that rank.
pub fn pool(group: &str) -> Inventory {
    Inventory {
        group: group.to_owned(),
        ranks: vec![rank("gpu-a")],
        layout: PoolLayout {
            queue_depth: 1,
            ranks: 1,
            components: std::collections::BTreeMap::from([(
                proto::config::DEFAULT_BLOCK.to_owned(),
                BlockConfig {
                    ranks: vec![0],
                    tensor_parallel_size: 1,
                    pipeline_parallel_size: 1,
                    sequence_parallel: Default::default(),
                    units_per_rank: 1,
                    weights: Default::default(),
                    distribution: None,
                },
            )]),
        },
        max_model_len: 4096,
    }
}

/// The launch arguments of a deployment named `name` on the CPU: one pool of one rank on this
/// host, with the simulation's generation limits and end of text.
pub fn args(name: &str) -> EngineArgs {
    scheduler::sim_args(name)
}

/// The mechanism's limits for `batches`: as many running requests as its ranks hold request rows.
///
/// `SchedulerHandle::over_batches` refuses a configuration that asks for more running requests
/// than the ranks hold rows, so the count is read from the pool rather than restated per case.
pub fn limits(batches: &dyn Batches) -> SchedulerConfig {
    let rows: usize = batches
        .inventory()
        .iter()
        .flat_map(|inventory| &inventory.ranks)
        .map(|rank| rank.request_slots as usize)
        .sum();
    SchedulerConfig {
        max_running_requests: rows.max(1),
        ..SchedulerConfig::default()
    }
}
