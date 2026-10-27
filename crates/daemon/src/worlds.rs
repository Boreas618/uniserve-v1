//! The deployment's worlds and how it moves between them
//! (`PUT /admin/config`, `specs/online-operations.md`).
//!
//! A world is the processes that execute one configuration of the
//! deployment: an engine and its workers, their communicators, graphs and
//! the memory they created. A deployment has two slots for worlds, and at
//! most one world executes. An update to a configuration that changes what
//! a world executes takes four steps:
//!
//! 1. **Preparation.** The serving world gives back the tail of its KV
//!    cache the successor needs (its requests run on; its capacity drops),
//!    and the successor starts in the other slot: its workers map what they
//!    share with the serving world read-only (the weights of the same keys,
//!    and the KV cache segments of the serving world's workers of the same
//!    scope), create the rest of their own, warm up and capture their graphs.
//!    The serving world executes throughout.
//! 2. **Promotion.** At a step boundary new requests wait, the serving world
//!    parks (it ends every token request that resumes from its generated
//!    tokens as migratable and executes nothing more, keeping its memory and
//!    its prefix cache index), and the successor imports that index into
//!    the units it maps and serves: every migrated request is submitted
//!    again with the tokens it generated and reuses the pages of every
//!    complete page of them, computing at most its last page.
//! 3. **Trial.** The successor executes; the previous world is kept as
//!    `update_config.previous_world` states: parked, its weights alone, or
//!    its weights in host memory. The trial ends only with the operator's
//!    commit, and rolls back by itself on a fault of the successor (a fatal
//!    worker error, its engine process's exit, a stalled job, non-finite
//!    logits) or past a threshold of `update_config.rollback`.
//! 4. **Commit** or **rollback.** A commit releases the previous world: its
//!    processes exit, the successor takes the segments it mapped read-only
//!    and retires the previous world's memory, and grows its cache into the
//!    memory freed. A rollback parks and closes the successor, resumes the
//!    previous world (or starts it again on its kept weights), retires the
//!    successor's memory, and marks the configuration unavailable: applying
//!    it again needs `force`.

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Context as _;
use proto::report::SchedulerStats;
use proto::request::wire;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::engine::client::{EngineClient, EngineState};
use crate::leader::{Leader, UpdateError, locked, stats_sink};
use crate::options::{PreviousWorld, RollbackOptions, ServeConfig};

/// How the requests in flight at a promotion continue when the new world
/// computes other numerics than the one they ran on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InFlight {
    /// Each continues with the state it reached, which the new world reads
    /// as its own; the pages it writes afterwards reach no shared tier.
    #[default]
    Keep,
    /// Each computes its state again from its tokens, exactly.
    Recompute,
}

/// A change of the deployment's configuration (`PUT /admin/config`).
#[derive(Debug, Clone, Default)]
pub struct ApplyRequest {
    /// The whole target configuration as TOML, as `--config` reads it;
    /// `None` takes the current one.
    pub config: Option<String>,
    /// `PATH=VALUE` assignments over that configuration, as `--set` states
    /// them.
    pub set: Vec<String>,
    pub in_flight: InFlight,
    /// Report the plan and change nothing.
    pub dry_run: bool,
    /// Apply a configuration a rollback marked unavailable.
    pub force: bool,
}

/// What an apply changed.
#[derive(Debug, Clone, PartialEq)]
pub enum Applied {
    /// The target is the current configuration.
    Unchanged,
    /// The schedulers took the new limits at their next step, or the daemon
    /// reads the options that changed from its next request on.
    InPlace(serde_json::Value),
    /// The plan of a dry run, which changed nothing.
    Planned(serde_json::Value),
    /// The daemon's own options or executable change: a hot restart of the
    /// daemon with the target configuration applies them.
    Daemon {
        config: Box<ServeConfig>,
        executable: Option<String>,
    },
    /// A new world executes on trial.
    World(serde_json::Value),
}

/// The previous world a trial keeps.
pub(crate) enum Previous {
    /// Its engine, parked: a rollback resumes it.
    Parked(Arc<EngineClient>),
    /// Its weights alone: a rollback starts its engine again, which binds
    /// them.
    Weights,
}

/// A world on trial and what it rolls back to.
pub(crate) struct Trial {
    /// The world on trial.
    pub(crate) world: u64,
    /// Its configuration's digest, which a rollback marks unavailable.
    pub(crate) digest: String,
    /// The previous world's identity, spec and configuration.
    pub(crate) previous_world: u64,
    pub(crate) previous_spec: wire::EngineSpec,
    pub(crate) previous_config: ServeConfig,
    pub(crate) previous: Previous,
    /// The previous world's retention mode, as the status reports it.
    pub(crate) mode: PreviousWorld,
    pub(crate) promoted: Instant,
    pub(crate) promoted_at: f64,
    /// The previous world's step time and throughput before the promotion.
    pub(crate) baseline: Option<Rates>,
    pub(crate) thresholds: RollbackOptions,
    /// The request outcomes counted when the trial began.
    pub(crate) outcomes: (u64, u64),
    /// How long requests waited at the promotion.
    pub(crate) pause: Duration,
}

/// A configuration a rollback marked unavailable, and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Unavailable {
    pub digest: String,
    /// The world that executed it and was rolled back.
    pub world: u64,
    pub reason: String,
    /// Seconds since the Unix epoch.
    pub at: f64,
}

/// A world change in progress, as the status reports it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Progress {
    pub phase: String,
    pub started: f64,
    /// The world being prepared, once it starts.
    pub world: Option<u64>,
}

/// The deployment's worlds as the leader tracks them.
#[derive(Default)]
pub(crate) struct WorldState {
    /// The world the executing engine runs.
    pub(crate) executing: u64,
    pub(crate) trial: Option<Trial>,
    pub(crate) progress: Option<Progress>,
    pub(crate) unavailable: BTreeMap<String, Unavailable>,
    /// What the last world change, commit or rollback reported.
    pub(crate) last: Option<serde_json::Value>,
}

/// Average step time and throughput of one world over a span.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rates {
    /// Microseconds per batch.
    pub step_us: f64,
    /// Tokens per second the forwards computed.
    pub tokens_per_second: f64,
}

/// One statistics report of one scheduler of one world.
struct Sample {
    at: Instant,
    world: u64,
    batch_us: u64,
    batches: u64,
    tokens: u64,
    numerical_faults: u64,
}

/// The schedulers' statistics of every world, kept over the longest span a
/// trial measures, from which a trial reads the previous world's baseline
/// and its own rates and faults.
#[derive(Default)]
pub(crate) struct Observer {
    samples: std::sync::Mutex<VecDeque<Sample>>,
}

/// The longest span statistics are kept for.
const OBSERVED: Duration = Duration::from_secs(600);

impl Observer {
    /// Records one scheduler's report of `world`.
    pub(crate) fn record(&self, world: u64, stats: &SchedulerStats) {
        let tokens = stats
            .worker_forward_stats
            .as_ref()
            .map_or(0, |forward| forward.mode_tokens.values().sum());
        let now = Instant::now();
        let mut samples = locked(&self.samples);
        samples.push_back(Sample {
            at: now,
            world,
            batch_us: stats.batch_roundtrip_us,
            batches: stats.batch_count,
            tokens,
            numerical_faults: stats.numerical_faults,
        });
        while samples
            .front()
            .is_some_and(|sample| now.duration_since(sample.at) > OBSERVED)
        {
            samples.pop_front();
        }
    }

    /// `world`'s rates over the `window` before now, from `since` on; `None`
    /// without a batch in it.
    pub(crate) fn rates(&self, world: u64, since: Instant, window: Duration) -> Option<Rates> {
        let now = Instant::now();
        let from = since.max(now.checked_sub(window).unwrap_or(since));
        let samples = locked(&self.samples);
        let (mut batch_us, mut batches, mut tokens) = (0_u64, 0_u64, 0_u64);
        let mut first: Option<Instant> = None;
        for sample in samples
            .iter()
            .filter(|sample| sample.world == world && sample.at >= from)
        {
            batch_us += sample.batch_us;
            batches += sample.batches;
            tokens += sample.tokens;
            first.get_or_insert(sample.at);
        }
        if batches == 0 {
            return None;
        }
        let span = now
            .duration_since(first.unwrap_or(from))
            .max(Duration::from_secs(1));
        Some(Rates {
            step_us: batch_us as f64 / batches as f64,
            tokens_per_second: tokens as f64 / span.as_secs_f64(),
        })
    }

    /// The calls of `world` that produced non-finite logits since `since`.
    pub(crate) fn numerical_faults(&self, world: u64, since: Instant) -> u64 {
        locked(&self.samples)
            .iter()
            .filter(|sample| sample.world == world && sample.at >= since)
            .map(|sample| sample.numerical_faults)
            .sum()
    }
}

/// The digest of a configuration, which names it in the status and the
/// unavailable marks: of its printed TOML.
pub(crate) fn digest(config: &ServeConfig) -> anyhow::Result<String> {
    use sha2::Digest as _;
    let printed = crate::options::printed(config)?;
    Ok(sha2::Sha256::digest(printed.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

/// How often a trial checks its thresholds.
const TRIAL_POLL: Duration = Duration::from_millis(500);

/// How long the closed engine of a world takes to exit.
const CLOSED_EXIT: Duration = Duration::from_secs(60);

/// The parts of `spec` a configuration states: without the world's identity
/// and the layout of its cache, which the world's history sets.
fn comparable(spec: &wire::EngineSpec) -> wire::EngineSpec {
    wire::EngineSpec {
        world: 0,
        predecessor: None,
        segments: Vec::new(),
        ..spec.clone()
    }
}

/// What a world change keeps of the serving world's derived state, by the
/// parts of the engine spec that decide each one's keys: the weights and
/// the KV cache the successor maps. Names the parts that differ.
fn differing(served: &wire::EngineSpec, target: &wire::EngineSpec) -> Vec<&'static str> {
    let mut parts = Vec::new();
    let mut differs = |name: &'static str, same: bool| {
        if !same {
            parts.push(name);
        }
    };
    differs("the deployment", served.deployment == target.deployment);
    differs("the model", served.model == target.model);
    differs("the host", served.host == target.host);
    differs("the placement", served.placement == target.placement);
    differs("the product transport", served.transfer == target.transfer);
    differs("the KV cache", served.kv_cache == target.kv_cache);
    differs("the experts", served.experts == target.experts);
    differs("the video options", served.video == target.video);
    if let (Some(old), Some(new)) = (&served.workers, &target.workers) {
        differs("the worker mesh", old.mesh == new.mesh);
        differs("the dtype", old.dtype == new.dtype);
        differs(
            "the quantization",
            old.quantization_config == new.quantization_config,
        );
        differs("the load format", old.load_format == new.load_format);
        differs("the weight source", old.weights == new.weights);
    }
    parts
}

impl Leader {
    /// Changes the deployment to the configuration `request` states, as
    /// the module describes, choosing the least the change needs: nothing
    /// for the current configuration; the schedulers' next step for their
    /// limits alone; a hot restart of the daemon for its own options; and a
    /// new world for anything a world executes.
    ///
    /// # Errors
    ///
    /// Refuses another change in progress or a world on trial (`409`), a
    /// configuration that does not parse or validate, one a rollback marked
    /// unavailable unless `force`, and a change the successor cannot map;
    /// fails when the successor does not start (the serving world serves
    /// on) or its promotion fails (the serving world resumes).
    pub async fn apply(&self, request: ApplyRequest) -> Result<Applied, UpdateError> {
        let _replacing = self
            .engines
            .replacing
            .try_lock()
            .map_err(|_| UpdateError::InProgress)?;
        if let Some(trial) = &locked(&self.engines.worlds).trial {
            return Err(UpdateError::OnTrial(trial.world));
        }
        if self.engines.asleep.load(Ordering::Acquire) {
            return Err(UpdateError::Refused(
                "the engine sleeps: wake it (/admin/wake_up) before updating it".to_owned(),
            ));
        }
        let current = locked(&self.engines.config).clone();
        let (mut target, stated) = match &request.config {
            Some(text) => crate::options::resolve_text(text, &request.set),
            None => crate::options::overridden(&current, &request.set)
                .map(|(config, paths)| (config, crate::options::Stated::from_paths(paths))),
        }
        .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
        target
            .validate(&stated)
            .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
        // The update options an update's assignments state govern that
        // update alone; the configuration keeps its own.
        let update = target.update_config.clone();
        if request.config.is_none() {
            target.update_config = current.update_config.clone();
        }
        let target_digest =
            digest(&target).map_err(|error| UpdateError::Failed(error.context("digest")))?;
        if !request.force
            && let Some(unavailable) = locked(&self.engines.worlds).unavailable.get(&target_digest)
        {
            return Err(UpdateError::Refused(format!(
                "the configuration {target_digest} was rolled back from world {} ({}); \
                 apply it again with force",
                unavailable.world, unavailable.reason
            )));
        }
        if target == current {
            return Ok(Applied::Unchanged);
        }
        // A configuration of another served name or model processes requests
        // otherwise: it is another deployment.
        if target.served_model_name() != current.served_model_name() {
            return Err(UpdateError::Refused(
                "the served model name names the deployment: deploy another one instead".to_owned(),
            ));
        }

        let served = locked(&self.engines.served)
            .clone()
            .ok_or(UpdateError::NoEngine)?;
        // The target's engine spec is resolved as a deployment's start
        // resolves it: from the options, completed by what the model's files
        // determine.
        let prepared = Leader::prepare(&target, &stated)
            .await
            .map_err(|error| UpdateError::Refused(format!("{error:#}")))?;
        let target_spec = wire::EngineSpec {
            world: 0,
            predecessor: None,
            segments: Vec::new(),
            ..prepared.spec
        };
        let world_changes = comparable(&served) != comparable(&target_spec);
        let daemon_changes = crate::options::daemon_options_differ(&current, &target);

        // A world change: what each derived state of the serving world
        // becomes in its successor.
        let unmapped = differing(&served, &target_spec);
        let running = self.migrator.hold().client().ok_or(UpdateError::NoEngine)?;
        let plan = serde_json::json!({
            "from": locked(&self.engines.worlds).executing,
            "digest": target_digest,
            "world": world_changes,
            "daemon": daemon_changes,
            "previous_world": update.previous_world,
            "in_flight": request.in_flight,
            "weights": if unmapped.is_empty() { "mapped" } else { "unmapped" },
            "kv_cache": if unmapped.is_empty() { "mapped" } else { "unmapped" },
            "differs": unmapped,
            "in_flight_requests": running.in_flight(),
        });
        if request.dry_run {
            return Ok(Applied::Planned(plan));
        }
        if !world_changes {
            if daemon_changes {
                return Ok(Applied::Daemon {
                    config: Box::new(target),
                    executable: None,
                });
            }
            *locked(&self.engines.config) = target;
            return Ok(Applied::InPlace(plan));
        }
        // Scheduling alone, within the batch the serving world's workers
        // were started for, changes in place at the schedulers' next step.
        if crate::leader::configurable(
            &comparable(&served),
            &comparable(&target_spec),
            running.info().capacity.as_ref(),
        ) {
            let scheduling = target_spec.scheduling.clone().unwrap_or_default();
            running
                .configure(scheduling)
                .await
                .map_err(|error| anyhow::anyhow!("the engine did not take the limits: {error}"))?;
            info!("the engine took the new scheduling limits in place");
            let mut spec = served;
            spec.scheduling = target_spec.scheduling;
            *locked(&self.engines.served) = Some(spec);
            *locked(&self.engines.config) = target;
            return Ok(Applied::InPlace(plan));
        }
        if daemon_changes {
            return Err(UpdateError::Refused(
                "the update changes both the daemon's own options and what a world executes: \
                 apply the daemon's options first, which restarts the daemon, then the rest"
                    .to_owned(),
            ));
        }
        if !unmapped.is_empty() {
            return Err(UpdateError::Refused(format!(
                "the update changes {}, which keys the weights and the KV cache the successor \
                 world maps: a world change maps them only where their keys stay",
                unmapped.join(", ")
            )));
        }
        if update.previous_world == PreviousWorld::Host {
            return Err(UpdateError::Refused(
                "`update_config.previous_world = \"host\"` moves the previous world's weights \
                 through the host's Transfer Server, which this deployment's Storage Server \
                 does not run"
                    .to_owned(),
            ));
        }
        let report = self
            .change_world(
                &running,
                served,
                current,
                target,
                target_spec,
                target_digest,
                &update,
            )
            .await?;
        Ok(Applied::World(report))
    }

    /// Moves the deployment from the world `running` executes to a
    /// successor executing `target`, up to its trial (steps 1 to 3 of the
    /// module).
    #[expect(
        clippy::too_many_arguments,
        reason = "one world change: both worlds' specs and configurations and its options"
    )]
    async fn change_world(
        &self,
        running: &Arc<EngineClient>,
        served: wire::EngineSpec,
        current: ServeConfig,
        target: ServeConfig,
        mut spec: wire::EngineSpec,
        target_digest: String,
        update: &crate::options::UpdateOptions,
    ) -> Result<serde_json::Value, UpdateError> {
        let started = Instant::now();
        let previous_world = running.world();
        self.progress("shrink", None);
        // The serving world's cache gives back the tail the successor needs;
        // its requests run on.
        running
            .resize_cache_within(wire::CacheChange::Shrink, update.preempt_after)
            .await
            .map_err(|error| {
                self.progress_done();
                anyhow::anyhow!(
                    "the KV cache did not make room for a successor world; the serving world \
                     serves on: {error}"
                )
            })?;
        let segments = running.describe_cache().await.map_err(|error| {
            self.progress_done();
            anyhow::anyhow!("the serving world did not describe its KV cache: {error}")
        })?;
        spec.world = 0;
        spec.predecessor = Some(wire::Predecessor {
            world: previous_world,
            segments,
        });

        self.progress("prepare", None);
        let control = Arc::clone(self.engines.control.get().ok_or(UpdateError::NoEngine)?);
        let sink = stats_sink(
            self.engines.model_name.clone(),
            Arc::clone(&self.engines.observer),
        );
        info!(previous_world, "preparing a successor world");
        let start_spec = spec.clone();
        let prepared = tokio::task::spawn_blocking(move || control.start(start_spec, sink))
            .await
            .context("the successor's start did not complete")
            .and_then(|started| started.context("the successor world did not start"));
        let successor = match prepared {
            Ok(successor) => Arc::new(successor),
            Err(error) => {
                self.progress_done();
                let restored = running.resize_cache(wire::CacheChange::Restore).await;
                return Err(match restored {
                    Ok(_) => error.context("the serving world serves on"),
                    Err(restoring) => error.context(format!(
                        "the serving world serves on without the tail of its KV cache, which it \
                         did not create again: {restoring}"
                    )),
                }
                .into());
            }
        };
        let world = successor.world();
        let prepare = started.elapsed();
        self.progress("promote", Some(world));

        // The promotion: the serving world's step boundary, its index, the
        // successor's import.
        let baseline = update.rollback.window.and_then(|window| {
            self.engines.observer.rates(
                previous_world,
                started.checked_sub(window).unwrap_or(started),
                window,
            )
        });
        let hold = Arc::clone(self.migrator.hold());
        let paused = Instant::now();
        hold.hold();
        let index = match running.park().await {
            Ok(index) => index,
            Err(error) => {
                // The serving world serves on: it resumes, and the successor
                // closes with its memory.
                let _ = running.resume(vec![world]).await;
                hold.serve(Arc::clone(running));
                self.close(&successor).await;
                self.progress_done();
                return Err(anyhow::anyhow!("the serving world did not park: {error}").into());
            }
        };
        let parked = paused.elapsed();
        if let Err(error) = successor.promote(index).await {
            let _ = running.resume(vec![world]).await;
            hold.serve(Arc::clone(running));
            self.close(&successor).await;
            self.progress_done();
            return Err(anyhow::anyhow!(
                "the successor world was not promoted; the previous world resumed: {error}"
            )
            .into());
        }
        hold.serve(Arc::clone(&successor));
        let pause = paused.elapsed();
        info!(
            world,
            previous_world,
            pause_ms = pause.as_millis() as u64,
            park_ms = parked.as_millis() as u64,
            "the successor world serves on trial"
        );
        crate::leader::reload_adapters(&self.engines, &successor).await;

        // The previous world as the trial keeps it.
        let previous = match update.previous_world {
            PreviousWorld::Parked => Previous::Parked(Arc::clone(running)),
            PreviousWorld::Weights | PreviousWorld::Host => {
                let closed = Arc::clone(running);
                let exit = tokio::task::spawn_blocking(move || {
                    closed.shutdown();
                    closed.await_exit(CLOSED_EXIT)
                })
                .await
                .context("the previous world's shutdown did not complete")?;
                if let Err(error) = exit {
                    warn!("the previous world's engine process: {error:#}");
                }
                Previous::Weights
            }
        };
        // The successor's cache grows into the memory the previous world
        // freed.
        if let Err(error) = successor.resize_cache(wire::CacheChange::Grow).await {
            warn!("the successor's KV cache did not grow: {error}");
        }

        let mut served_spec = spec;
        served_spec.world = world;
        *locked(&self.engines.served) = Some(served_spec);
        *locked(&self.engines.config) = target;
        let report = serde_json::json!({
            "world": world,
            "previous_world": previous_world,
            "prepare_seconds": prepare.as_secs_f64(),
            "park_seconds": parked.as_secs_f64(),
            "pause_seconds": pause.as_secs_f64(),
            "previous_world_kept": update.previous_world,
        });
        {
            let mut worlds = locked(&self.engines.worlds);
            worlds.executing = world;
            worlds.progress = None;
            worlds.trial = Some(Trial {
                world,
                digest: target_digest,
                previous_world,
                previous_spec: served,
                previous_config: current,
                previous,
                mode: update.previous_world,
                promoted: paused,
                promoted_at: now(),
                baseline,
                thresholds: update.rollback.clone(),
                outcomes: crate::leader::outcomes(),
                pause,
            });
            worlds.last = Some(report.clone());
        }
        tokio::spawn(watch_trial(Arc::clone(&self.engines), world));
        Ok(report)
    }

    /// Commits the world on trial: the previous world's processes exit and
    /// its memory is retired, the successor takes the segments it mapped
    /// read-only and grows its cache into the memory freed, and relaunches
    /// its worker groups on a loss from then on.
    ///
    /// # Errors
    ///
    /// Refuses when no world is on trial; fails when the successor does not
    /// commit, which leaves it serving on trial.
    pub async fn commit(&self) -> Result<serde_json::Value, UpdateError> {
        let _replacing = self.engines.replacing.lock().await;
        let trial = locked(&self.engines.worlds)
            .trial
            .take()
            .ok_or_else(|| UpdateError::Refused("no world is on trial".to_owned()))?;
        let client = self.migrator.hold().client().ok_or(UpdateError::NoEngine)?;
        if let Previous::Parked(previous) = &trial.previous {
            let previous = Arc::clone(previous);
            let exit = tokio::task::spawn_blocking(move || {
                previous.shutdown();
                previous.await_exit(CLOSED_EXIT)
            })
            .await
            .context("the previous world's shutdown did not complete")?;
            if let Err(error) = exit {
                warn!("the previous world's engine process: {error:#}");
            }
        }
        let results = match client.commit(vec![trial.previous_world]).await {
            Ok(results) => results,
            Err(error) => {
                let world = trial.world;
                locked(&self.engines.worlds).trial = Some(Trial {
                    previous: Previous::Weights,
                    ..trial
                });
                return Err(anyhow::anyhow!(
                    "world {world} did not commit and stays on trial, its previous world's \
                     processes closed: {error}"
                )
                .into());
            }
        };
        if let Err(error) = client.resize_cache(wire::CacheChange::Grow).await {
            warn!("the committed world's KV cache did not grow: {error}");
        }
        // A relaunch of the committed world lays its planes out as they are:
        // the previous world's segments are its own.
        let segments = client.describe_cache().await.unwrap_or_default();
        if let Some(spec) = locked(&self.engines.served).as_mut() {
            spec.predecessor = None;
            spec.segments = segments;
        }
        let report = serde_json::json!({
            "world": trial.world,
            "previous_world": trial.previous_world,
            "trial_seconds": trial.promoted.elapsed().as_secs_f64(),
            "workers": results.len(),
        });
        info!(
            world = trial.world,
            previous_world = trial.previous_world,
            "the world is committed"
        );
        locked(&self.engines.worlds).last = Some(report.clone());
        Ok(report)
    }

    /// Rolls the world on trial back to the previous world, for `reason`.
    ///
    /// # Errors
    ///
    /// Refuses when no world is on trial; fails when the previous world
    /// does not serve again, which leaves no world serving.
    pub async fn rollback(&self, reason: String) -> Result<serde_json::Value, UpdateError> {
        let _replacing = self.engines.replacing.lock().await;
        rollback(&self.engines, reason).await
    }

    /// The deployment's configuration and worlds, as `GET /admin/config`
    /// reports them.
    pub fn config_status(&self) -> serde_json::Value {
        let config = locked(&self.engines.config).clone();
        let printed = crate::options::printed(&config).unwrap_or_default();
        let digest = digest(&config).unwrap_or_default();
        let worlds = locked(&self.engines.worlds);
        let hold = self.migrator.hold();
        let executing = hold.client().map(|client| {
            serde_json::json!({
                "world": client.world(),
                "state": match client.state() {
                    EngineState::Standby => "preparing",
                    EngineState::Ready => "executing",
                    EngineState::Parked => "parked",
                    EngineState::Failed(_) => "failed",
                    EngineState::Closed => "closed",
                },
            })
        });
        let other = worlds.trial.as_ref().map(|trial| {
            serde_json::json!({
                "world": trial.previous_world,
                "state": match trial.mode {
                    PreviousWorld::Parked => "parked",
                    PreviousWorld::Weights => "weights",
                    PreviousWorld::Host => "host",
                },
            })
        });
        let other = other.or_else(|| {
            worlds.progress.as_ref().and_then(|progress| {
                progress
                    .world
                    .map(|world| serde_json::json!({ "world": world, "state": "preparing" }))
            })
        });
        let trial = worlds.trial.as_ref().map(|trial| {
            let rates = trial.thresholds.window.and_then(|window| {
                self.engines
                    .observer
                    .rates(trial.world, trial.promoted, window)
            });
            serde_json::json!({
                "world": trial.world,
                "previous_world": trial.previous_world,
                "previous_world_kept": trial.mode,
                "seconds": trial.promoted.elapsed().as_secs_f64(),
                "promoted_at": trial.promoted_at,
                "pause_seconds": trial.pause.as_secs_f64(),
                "baseline": trial.baseline,
                "rates": rates,
                "step_time_change": match (trial.baseline, rates) {
                    (Some(baseline), Some(rates)) if baseline.step_us > 0.0 =>
                        Some(rates.step_us / baseline.step_us - 1.0),
                    _ => None,
                },
            })
        });
        serde_json::json!({
            "config": printed,
            "digest": digest,
            "slots": [executing, other],
            "update": worlds.progress,
            "trial": trial,
            "unavailable": worlds.unavailable.values().collect::<Vec<_>>(),
            "last": worlds.last,
        })
    }

    fn progress(&self, phase: &str, world: Option<u64>) {
        let mut worlds = locked(&self.engines.worlds);
        let started = worlds
            .progress
            .as_ref()
            .map_or_else(now, |progress| progress.started);
        worlds.progress = Some(Progress {
            phase: phase.to_owned(),
            started,
            world: world.or(worlds.progress.as_ref().and_then(|progress| progress.world)),
        });
    }

    fn progress_done(&self) {
        locked(&self.engines.worlds).progress = None;
    }

    /// Shuts the engine `client` serves down and waits for its process.
    async fn close(&self, client: &Arc<EngineClient>) {
        close(client).await;
    }
}

/// Shuts the engine `client` serves down and waits for its process.
async fn close(client: &Arc<EngineClient>) {
    let closed = Arc::clone(client);
    let exit = tokio::task::spawn_blocking(move || {
        closed.shutdown();
        closed.await_exit(CLOSED_EXIT)
    })
    .await;
    match exit {
        Ok(Ok(_)) => {}
        Ok(Err(error)) => warn!("an engine process: {error:#}"),
        Err(error) => warn!("an engine's shutdown: {error}"),
    }
}

/// Rolls the world on trial back to the previous world (see the module);
/// the caller holds the replacement lock.
pub(crate) async fn rollback(
    engines: &Arc<crate::leader::Engines>,
    reason: String,
) -> Result<serde_json::Value, UpdateError> {
    let trial = locked(&engines.worlds)
        .trial
        .take()
        .ok_or_else(|| UpdateError::Refused("no world is on trial".to_owned()))?;
    warn!(world = trial.world, previous_world = trial.previous_world, %reason, "rolling back");
    let hold = Arc::clone(&engines.hold);
    let paused = Instant::now();
    // The hold names no engine while it holds: the world on trial is taken
    // first.
    let failing = hold.client();
    hold.hold();
    // The world on trial preempts its requests, which wait for the previous
    // world: its live schedulers park, and a scheduler that failed handed
    // its requests back as it stopped. An engine whose process ended handed
    // them back as it ended (`EngineClient`).
    if let Some(failing) = failing {
        if failing.state() != EngineState::Closed {
            match tokio::time::timeout(CLOSED_EXIT, failing.park()).await {
                Ok(Ok(_)) => {}
                Ok(Err(error)) => info!(%error, "the world on trial parked in part"),
                Err(_) => warn!(
                    "the world on trial did not park within {CLOSED_EXIT:?}; closing it with \
                     the requests it holds"
                ),
            }
        }
        close(&failing).await;
    }
    let previous = match trial.previous {
        Previous::Parked(previous) => {
            previous.resume(vec![trial.world]).await.map_err(|error| {
                hold.close(format!("the previous world did not resume: {error}"));
                anyhow::anyhow!("the previous world did not resume: {error}")
            })?;
            previous
        }
        Previous::Weights => {
            let control = Arc::clone(engines.control.get().ok_or(UpdateError::NoEngine)?);
            let sink = stats_sink(engines.model_name.clone(), Arc::clone(&engines.observer));
            let spec = trial.previous_spec.clone();
            let started = tokio::task::spawn_blocking(move || control.start(spec, sink))
                .await
                .context("the previous world's start did not complete")
                .and_then(|started| started.context("the previous world did not start"));
            let started = match started {
                Ok(started) => Arc::new(started),
                Err(error) => {
                    hold.close(format!("the previous world did not start again: {error:#}"));
                    return Err(error.into());
                }
            };
            started
                .resume(vec![trial.world])
                .await
                .map_err(|error| anyhow::anyhow!("retiring world {}: {error}", trial.world))?;
            started
        }
    };
    hold.serve(Arc::clone(&previous));
    crate::leader::reload_adapters(engines, &previous).await;
    if let Err(error) = previous.resize_cache(wire::CacheChange::Restore).await {
        warn!("the previous world's KV cache was not restored: {error}");
    }
    let pause = paused.elapsed();
    *locked(&engines.served) = Some(trial.previous_spec);
    *locked(&engines.config) = trial.previous_config;
    let report = serde_json::json!({
        "rolled_back": trial.world,
        "world": trial.previous_world,
        "reason": reason,
        "pause_seconds": pause.as_secs_f64(),
        "trial_seconds": trial.promoted.elapsed().as_secs_f64(),
    });
    {
        let mut worlds = locked(&engines.worlds);
        worlds.executing = trial.previous_world;
        worlds.unavailable.insert(
            trial.digest.clone(),
            Unavailable {
                digest: trial.digest,
                world: trial.world,
                reason,
                at: now(),
            },
        );
        worlds.last = Some(report.clone());
    }
    watch(Arc::clone(engines), Arc::downgrade(&previous));
    info!(
        world = trial.previous_world,
        "the previous world serves again"
    );
    Ok(report)
}

/// Watches the engine `client` serves for its loss (`watch_engine`), on a
/// task of its own whose future the rollback it may run does not contain.
fn watch(engines: Arc<crate::leader::Engines>, client: std::sync::Weak<EngineClient>) {
    let watching: std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> =
        Box::pin(crate::leader::watch_engine(engines, client));
    tokio::spawn(watching);
}

/// Watches world `world`'s trial and rolls it back on a fault or past a
/// threshold; ends with the trial.
async fn watch_trial(engines: Arc<crate::leader::Engines>, world: u64) {
    loop {
        tokio::time::sleep(TRIAL_POLL).await;
        let reason = {
            let worlds = locked(&engines.worlds);
            let Some(trial) = worlds.trial.as_ref().filter(|trial| trial.world == world) else {
                return;
            };
            trial_fault(&engines, trial)
        };
        let Some(reason) = reason else {
            continue;
        };
        let _replacing = engines.replacing.lock().await;
        if !locked(&engines.worlds)
            .trial
            .as_ref()
            .is_some_and(|trial| trial.world == world)
        {
            return;
        }
        if let Err(error) = rollback(&engines, reason).await {
            tracing::error!("the rollback failed: {error:#}");
        }
        return;
    }
}

/// Why `trial` rolls back now, if it does.
fn trial_fault(engines: &crate::leader::Engines, trial: &Trial) -> Option<String> {
    let client = engines.hold.client();
    match client.as_ref().map(|client| client.state()) {
        Some(EngineState::Failed(reason)) => {
            return Some(format!("world {} failed: {reason}", trial.world));
        }
        Some(EngineState::Closed) | None => {
            return Some(format!("world {}'s engine process ended", trial.world));
        }
        _ => {}
    }
    let faults = engines
        .observer
        .numerical_faults(trial.world, trial.promoted);
    if faults > 0 {
        return Some(format!(
            "world {} produced non-finite logits in {faults} calls",
            trial.world
        ));
    }
    let thresholds = &trial.thresholds;
    if let Some(rate) = thresholds.failure_rate {
        let (ended, failed) = crate::leader::outcomes();
        let ended = ended.saturating_sub(trial.outcomes.0);
        let failed = failed.saturating_sub(trial.outcomes.1);
        if ended > 0 && failed as f64 / ended as f64 > rate {
            return Some(format!(
                "{failed} of the {ended} requests that ended on world {} failed, past the \
                 failure rate {rate}",
                trial.world
            ));
        }
    }
    if let (Some(window), Some(baseline)) = (thresholds.window, trial.baseline)
        && trial.promoted.elapsed() >= window
        && let Some(rates) = engines.observer.rates(trial.world, trial.promoted, window)
    {
        if let Some(regression) = thresholds.step_time_regression
            && baseline.step_us > 0.0
            && rates.step_us > baseline.step_us * (1.0 + regression)
        {
            return Some(format!(
                "world {}'s steps take {:.0} us against {:.0} us before, past the regression \
                 {regression}",
                trial.world, rates.step_us, baseline.step_us
            ));
        }
        if let Some(regression) = thresholds.throughput_regression
            && baseline.tokens_per_second > 0.0
            && rates.tokens_per_second < baseline.tokens_per_second * (1.0 - regression)
        {
            return Some(format!(
                "world {} computes {:.0} tokens/s against {:.0} before, past the regression \
                 {regression}",
                trial.world, rates.tokens_per_second, baseline.tokens_per_second
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(batch_us: u64, batches: u64, tokens: u64, faults: u64) -> SchedulerStats {
        SchedulerStats {
            batch_roundtrip_us: batch_us,
            batch_count: batches,
            numerical_faults: faults,
            worker_forward_stats: Some(proto::report::ForwardStats {
                mode_tokens: BTreeMap::from([("decode".to_owned(), tokens)]),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// A world's rates average its own batches over the window, apart from
    /// another world's.
    #[test]
    fn rates_average_one_worlds_batches() {
        let observer = Observer::default();
        let since = Instant::now();
        observer.record(1, &stats(1_000, 10, 500, 0));
        observer.record(1, &stats(3_000, 10, 500, 0));
        observer.record(2, &stats(90_000, 1, 1, 2));
        let rates = observer
            .rates(1, since, Duration::from_secs(60))
            .expect("world 1 ran batches");
        assert!((rates.step_us - 200.0).abs() < 1e-9, "{rates:?}");
        assert!(rates.tokens_per_second > 0.0);
        assert_eq!(observer.numerical_faults(1, since), 0);
        assert_eq!(observer.numerical_faults(2, since), 2);
        assert!(observer.rates(3, since, Duration::from_secs(60)).is_none());
    }

    /// A change of the scheduling alone, within the batch the workers were
    /// started for, is in place for a world that a change started, whose
    /// spec carries its identity and its cache's layout.
    #[test]
    fn a_scheduling_change_of_a_successor_world_is_in_place() {
        let scheduling = |running: u32| wire::Scheduling {
            max_running_requests: running,
            max_batch: 8,
            ..Default::default()
        };
        let served = wire::EngineSpec {
            model: "a".into(),
            world: 3,
            segments: vec![wire::CacheSegments {
                worker_group: "model".into(),
                ..Default::default()
            }],
            scheduling: Some(scheduling(8)),
            ..Default::default()
        };
        let target = wire::EngineSpec {
            world: 0,
            segments: Vec::new(),
            scheduling: Some(scheduling(4)),
            ..served.clone()
        };
        let capacity = wire::BatchCapacity {
            max_batch: 8,
            max_num_batched_tokens: 8192,
            max_running_requests: 8,
        };
        assert!(crate::leader::configurable(
            &comparable(&served),
            &comparable(&target),
            Some(&capacity)
        ));
    }

    /// Only the parts that key the weights and the KV cache a successor
    /// maps differ; the build and the scheduling do not.
    #[test]
    fn differing_names_the_parts_that_key_shared_state() {
        let served = wire::EngineSpec {
            model: "a".into(),
            workers: Some(wire::Workers {
                executable: "old".into(),
                ..Default::default()
            }),
            ..Default::default()
        };
        let mut target = served.clone();
        if let Some(workers) = target.workers.as_mut() {
            workers.executable = "new".into();
        }
        assert!(differing(&served, &target).is_empty());
        target.model = "b".into();
        assert_eq!(differing(&served, &target), vec!["the model"]);
    }
}
