//! Scheduler construction shared by the engine's integration tests.
//!
//! Serving states a deployment's model limits from the model's description;
//! these tests have no model, so a simulated worker's limits follow from the
//! call kinds it serves. Each test binary uses a subset of these functions.

#![allow(dead_code)]

pub(crate) mod storage;
pub(crate) mod video;

use engine::{Executor, Scheduler, SchedulerConfig, SimEngine, SpecialTokenIds};
use engine::{GenerationFeatures, GenerationLimits, ModelDtype};
use ipc::{CallKind, ForwardMode, MediaCall};
use proto::request::SchedulingPolicy;

/// Builds a scheduler whose model limits follow from `executor`'s call kinds.
///
/// A token-denoising worker without text decode serves canvas readouts over
/// prompts that may carry encoded images; one that prepares latents without
/// decoding text serves video diffusion, which needs no generation feature.
/// Any other denoising, vision-encoding or latent-encoding worker serves
/// interleaved text and images without bound, and the rest serve text.
pub(crate) fn scheduler_with(
    executor: Box<dyn Executor>,
    special_tokens: SpecialTokenIds,
    config: SchedulerConfig,
) -> anyhow::Result<Scheduler> {
    let info = executor.info().runtime_info()?;
    let calls = &info.supported_calls;
    let decodes = calls.contains(&CallKind::Forward(ForwardMode::Decode));
    let limits = if calls.contains(&CallKind::Forward(ForwardMode::TokenDenoising)) && !decodes {
        GenerationLimits {
            features: GenerationFeatures::TOKEN_DENOISING | GenerationFeatures::VISION_ENCODE,
            latent_downsample: 1,
            max_vit_grid_tokens: u32::MAX,
            max_vision_feature_bytes: 256 << 20,
            max_cfg_branches: 1,
            encoder_cache_entries: 256,
            ..Default::default()
        }
    } else if calls.contains(&CallKind::Media(MediaCall::LatentPreparation)) && !decodes {
        GenerationLimits {
            latent_downsample: 1,
            max_cfg_branches: 1,
            ..Default::default()
        }
    } else if calls.contains(&CallKind::Media(MediaCall::Denoising))
        || calls.contains(&CallKind::Media(MediaCall::VisionEncoding))
        || calls.contains(&CallKind::Media(MediaCall::LatentEncoding))
    {
        SimEngine::generation_limits()
    } else {
        GenerationLimits {
            features: GenerationFeatures::UNDERSTANDING,
            latent_downsample: 1,
            max_cfg_branches: 1,
            ..Default::default()
        }
    };
    Scheduler::new(
        executor,
        special_tokens,
        config,
        ModelDtype::BFloat16,
        limits,
    )
}

/// Builds a scheduler that batches at most `max_batch` calls per pass.
pub(crate) fn scheduler(
    executor: Box<dyn Executor>,
    special_tokens: SpecialTokenIds,
    max_batch: usize,
) -> anyhow::Result<Scheduler> {
    scheduler_with(
        executor,
        special_tokens,
        SchedulerConfig {
            max_batch,
            ..Default::default()
        },
    )
}

/// Builds a scheduler that batches at most `max_batch` calls per pass and
/// orders waiting requests by `policy`.
pub(crate) fn scheduler_with_policy(
    executor: Box<dyn Executor>,
    special_tokens: SpecialTokenIds,
    max_batch: usize,
    policy: SchedulingPolicy,
) -> anyhow::Result<Scheduler> {
    scheduler_with(
        executor,
        special_tokens,
        SchedulerConfig {
            max_batch,
            policy,
            ..Default::default()
        },
    )
}

/// The leader daemon this test process runs for the worker groups its tests
/// launch, as a deployment of one node whose host is `localhost`; returns
/// its process socket. The workers it starts are children of its service
/// thread and end with the test process.
pub(crate) fn daemon() -> String {
    static DAEMON: std::sync::OnceLock<(daemon::Daemon, String)> = std::sync::OnceLock::new();
    DAEMON
        .get_or_init(|| {
            let directory = tempfile::tempdir().unwrap().keep();
            let daemon = daemon::Daemon::start(daemon::DaemonOptions {
                nnodes: 1,
                node_rank: 0,
                host: "localhost".into(),
                dist_init_addr: None,
                socket: directory.join("daemon.sock"),
                executable: uniserve_executable(),
                join_timeout: daemon::JOIN_TIMEOUT,
            })
            .unwrap();
            let address = daemon.address().unwrap().to_owned();
            (daemon, address)
        })
        .1
        .clone()
}

/// The `uniserve` executable the workers run as, and that starts the node's
/// Storage Server: `UNISERVE_WORKER_EXECUTABLE`, or else the repository
/// environment's.
pub(crate) fn uniserve_executable() -> std::path::PathBuf {
    std::env::var_os("UNISERVE_WORKER_EXECUTABLE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .join(".venv/bin/uniserve")
        })
}

/// Every process the threads of this test process started, with its
/// arguments: the daemon's workers among them.
pub(crate) fn child_processes() -> Vec<(u32, Vec<String>)> {
    let mut children = Vec::new();
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return children;
    };
    for task in tasks.flatten() {
        let Ok(pids) = std::fs::read_to_string(task.path().join("children")) else {
            continue;
        };
        for pid in pids.split_whitespace().filter_map(|pid| pid.parse().ok()) {
            let arguments = std::fs::read(format!("/proc/{pid}/cmdline"))
                .unwrap_or_default()
                .split(|byte| *byte == 0)
                .map(|argument| String::from_utf8_lossy(argument).into_owned())
                .collect();
            children.push((pid, arguments));
        }
    }
    children
}

/// The worker processes of group `worker_id` this test process's daemon
/// started: each process id with the value of its `--rank` and
/// `--generation` arguments.
pub(crate) fn group_processes(worker_id: &str) -> Vec<(u32, u32, u64)> {
    child_processes()
        .into_iter()
        .filter_map(|(pid, arguments)| {
            let value = |name: &str| {
                arguments
                    .iter()
                    .position(|argument| argument == name)
                    .and_then(|index| arguments.get(index + 1))
            };
            if value("--worker-id").map(String::as_str) != Some(worker_id) {
                return None;
            }
            let rank = value("--rank")?.parse().ok()?;
            let generation = value("--generation")?.parse().ok()?;
            Some((pid, rank, generation))
        })
        .collect()
}
