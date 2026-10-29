//! Transactions that change the number of independent replicas.
//!
//! Every predecessor keeps serving until all successor engines are ready.
//! Promotion, publication, retirement and rollback operate on the entire
//! group. A predecessor's continuations have one successor owner, selected
//! by its ordinal; admission uses that same assignment.

use super::*;
use crate::leader::Replicas;
use crate::options::{Stated, UpdateOptions};
use storage::manifest::Gpu;
use storage::manifest::weights::RankBudget;

pub(crate) struct Trial {
    previous: Vec<Arc<Replica>>,
    current: Vec<Arc<Replica>>,
    retained: Vec<Previous>,
    cache_backup: Vec<storage::manifest::cache::Snapshot>,
    digest: String,
    mode: PreviousWorld,
    host_refused: Option<String>,
    promoted: Instant,
    promoted_at: f64,
    baseline: Option<Rates>,
    thresholds: RollbackOptions,
    outcomes: (u64, u64),
    eager: bool,
    pause: Duration,
}

impl Trial {
    pub(crate) fn world(&self) -> u64 {
        locked(&self.current[0].worlds).executing
    }

    pub(crate) fn contains(&self, world: u64) -> bool {
        self.current
            .iter()
            .any(|replica| locked(&replica.worlds).executing == world)
    }

    pub(crate) fn status(&self) -> serde_json::Value {
        serde_json::json!({
            "worlds": worlds(&self.current),
            "previous_worlds": worlds(&self.previous),
            "previous_world_kept": self.mode,
            "host_refused": self.host_refused,
            "seconds": self.promoted.elapsed().as_secs_f64(),
            "promoted_at": self.promoted_at,
            "pause_seconds": self.pause.as_secs_f64(),
            "baseline": self.baseline,
        })
    }
}

fn worlds(replicas: &[Arc<Replica>]) -> Vec<u64> {
    replicas
        .iter()
        .map(|replica| locked(&replica.worlds).executing)
        .collect()
}

fn clients(replicas: &[Arc<Replica>]) -> Result<Vec<Arc<EngineClient>>, UpdateError> {
    replicas
        .iter()
        .map(|replica| replica.hold.client().ok_or(UpdateError::NoEngine))
        .collect()
}

async fn lock_replicas(replicas: &[Arc<Replica>]) -> Vec<tokio::sync::OwnedMutexGuard<()>> {
    let mut guards = Vec::with_capacity(replicas.len());
    for replica in replicas {
        guards.push(Arc::clone(&replica.replacing).lock_owned().await);
    }
    guards
}

fn rates(replicas: &[Arc<Replica>], since: Instant, window: Duration) -> Option<Rates> {
    let now = Instant::now();
    let from = since.max(now.checked_sub(window).unwrap_or(since));
    let mut first = now;
    let (mut batches, mut micros, mut tokens) = (0_u64, 0_u64, 0_u64);
    for replica in replicas {
        let world = locked(&replica.worlds).executing;
        replica.observer.samples(world, from, now, |sample| {
            first = first.min(sample.at);
            batches += sample.batches;
            micros += sample.batch_us;
            tokens += sample.tokens;
        });
    }
    (batches != 0).then(|| Rates {
        step_us: micros as f64 / batches as f64,
        tokens_per_second: tokens as f64
            / now
                .duration_since(first)
                .max(Duration::from_secs(1))
                .as_secs_f64(),
    })
}

/// Adds target owners that share a GPU. Their quotas overlap, while their
/// allocations and assigned request continuations are distinct.
fn add_budget(total: &mut RankBudget, rank: RankBudget) {
    total.quota = total.quota.min(rank.quota);
    total.process += rank.process;
    total.graphs += rank.graphs;
    total.fixed += rank.fixed;
    total.cache += rank.cache;
    total.cache_capacity += rank.cache_capacity;
    // Each independent target owner rounds its continuation reservation
    // before aggregation; a sum of segment sizes would overcharge it.
    total.cache_segment = 0;
    total.weights += rank.weights;
    let aggregate = total.in_flight.get_or_insert_default();
    let reservation = rank.in_flight.unwrap_or_default();
    aggregate.bytes += reservation.bytes;
    aggregate.mapped += reservation.mapped;
    total.pages.extend(rank.pages);
}

/// Restores predecessors after a preparation or promotion failed. The
/// predecessors that parked resume; the rest continue serving throughout.
async fn restore(
    previous: &[Arc<Replica>],
    serving: &[Arc<EngineClient>],
    candidates: &[Arc<EngineClient>],
    parked: &[u64],
    eager: bool,
    backup: &[storage::manifest::cache::Snapshot],
) {
    let retiring: Vec<_> = candidates.iter().map(|client| client.world()).collect();
    for candidate in candidates {
        close(candidate).await;
    }
    for (replica, client) in previous.iter().zip(serving) {
        replica.hold.keep(None);
        if let Err(error) = warming::resume_prefill(std::slice::from_ref(client)).await {
            warn!(%error, "the predecessor did not restore its prefill budget");
        }
        let restored = if parked.contains(&client.world()) {
            client.resume(retiring.clone()).await
        } else {
            client.commit(retiring.clone()).await.map(drop)
        };
        match restored {
            Ok(()) => match warming::restore(client, backup).await {
                Ok(()) => replica.hold.serve(Arc::clone(client)),
                Err(error) => replica
                    .hold
                    .close(format!("the predecessor cache did not restore: {error}")),
            },
            Err(error) => replica
                .hold
                .close(format!("the predecessor did not resume: {error}")),
        }
        replica.restore_tail(client).await;
        replay_graphs(client, eager).await;
        replica.progress_done();
    }
}

/// Applies a change spanning replica boundaries. The caller holds the
/// deployment replacement lock; this function owns every predecessor's
/// replacement lock through preparation and promotion.
pub(crate) async fn apply(
    leader: &Leader,
    request: &ApplyRequest,
    target: ServeConfig,
    stated: Stated,
    mut update: UpdateOptions,
) -> Result<Applied, UpdateError> {
    let previous = leader.replicas();
    let _owners = lock_replicas(&previous).await;
    if previous
        .iter()
        .any(|replica| replica.asleep.load(Ordering::Acquire))
    {
        return Err(UpdateError::Refused(
            "the deployment sleeps: wake it (/admin/wake_up) before updating it".into(),
        ));
    }
    let serving = clients(&previous)?;
    let source_worlds = worlds(&previous);
    let target_digest = previous[0].available(&target, request.force)?;
    for replica in &previous[1..] {
        replica.available(&target, request.force)?;
    }
    let current = locked(&previous[0].config).clone();
    if crate::options::daemon_options_differ(&current, &target) {
        return Ok(Applied::Daemon {
            config: Box::new(crate::options::with_daemon_options(&current, &target)),
            executable: None,
            then: Some(Box::new(request.clone())),
        });
    }
    let prepared = Leader::prepare(&target, &stated).await?;
    let mut specs = engine::EngineConfig::replica_specs(&prepared.spec)?;
    let control = Arc::clone(previous[0].control.get().ok_or(UpdateError::NoEngine)?);
    let first_world = control.next_engine();
    let mut prefixes = vec![Vec::new(); specs.len()];
    for (index, client) in serving.iter().enumerate() {
        let owner = index % specs.len();
        prefixes[owner].extend(
            client
                .describe_requests()
                .await
                .map_err(|error| anyhow::anyhow!("reading accepted request prefixes: {error}"))?,
        );
    }
    let mut budgets = BTreeMap::<Gpu, RankBudget>::new();
    let mut hosts = BTreeMap::new();
    let mut plans = Vec::new();
    let mut refusals = Vec::new();
    for (index, spec) in specs.iter_mut().enumerate() {
        for replica in &previous {
            let served = locked(&replica.served)
                .clone()
                .ok_or(UpdateError::NoEngine)?;
            if served.model != spec.model {
                crate::leader::same_processing(&served.model, &spec.model)
                    .await
                    .map_err(UpdateError::Refused)?;
                if served.model_facts != spec.model_facts {
                    return Err(UpdateError::Refused(
                        "the target model changes generation limits, control tokens or context length: deploy it instead".into(),
                    ));
                }
            }
            let incompatible = differing(&served, spec);
            if !incompatible.is_empty() {
                return Err(UpdateError::Refused(format!(
                    "the group update changes unsupported resource contracts: {}",
                    incompatible.join(", ")
                )));
            }
        }
        spec.world = first_world + index as u64;
        let plan = redistribution::plan(
            &previous[0].manifests,
            control.process_address(),
            spec.world,
            &source_worlds,
            spec,
            false,
            &prefixes[index],
        )
        .await?
        .context("the target ranks did not provide numerical storage budgets")?;
        if let Some(refusal) = plan.refusal {
            refusals.push(refusal);
        }
        spec.received_weights = plan.received;
        // Group reshapes lay out their own writable caches. The predecessor
        // marks standby startup; the transfer manifests name every source.
        spec.predecessor = Some(wire::Predecessor {
            world: source_worlds[0],
            segments: Vec::new(),
        });
        hosts.extend(plan.hosts);
        for (gpu, budget) in plan.budgets {
            match budgets.entry(gpu) {
                std::collections::btree_map::Entry::Occupied(mut entry) => {
                    add_budget(entry.get_mut(), budget);
                }
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(budget);
                }
            }
        }
        plans.push(plan.report);
    }
    let mut adds = Successor {
        loads_weights: false,
        unmapped_cache: true,
        host_cache: false,
        eager_source: update.eager_prepare,
        captures_graphs: specs.iter().any(|spec| {
            spec.cuda_graphs
                .as_ref()
                .is_none_or(|graphs| graphs.policy != "off")
        }),
    };
    let (admission, refusal) = storage_plan(
        &serving,
        &mut adds,
        &budgets,
        &hosts,
        &control,
        update.previous_world,
    )
    .await;
    if let Some(refusal) = refusal {
        refusals.push(refusal);
    }
    for spec in &mut specs {
        spec.stage_cache = adds.host_cache;
    }
    update.eager_prepare = adds.eager_source;
    let plan = serde_json::json!({
        "previous_worlds": source_worlds,
        "replicas": specs.len(),
        "redistribution": plans,
        "storage": admission,
        "feasible": refusals.is_empty(),
        "refusals": refusals,
    });
    if request.dry_run {
        return Ok(Applied::Planned(plan));
    }
    if !refusals.is_empty() {
        return Err(UpdateError::Refused(refusals.join("; ")));
    }
    let started = Instant::now();
    let baseline = update.rollback.window.and_then(|window| {
        rates(
            &previous,
            started.checked_sub(window).unwrap_or(started),
            window,
        )
    });
    let mut candidates = Vec::new();
    let mut observers = Vec::new();
    let mut parked = Vec::new();
    let mut interference = None;
    let mut warmed = Vec::new();
    let result: anyhow::Result<()> = async {
        for (replica, client) in previous.iter().zip(&serving) {
            replica.progress("prepare", None);
            if update.eager_prepare {
                client
                    .hook("uniserve.eager", r#"{"eager":true}"#)
                    .await
                    .map_err(anyhow::Error::msg)?;
            }
            client
                .make_room(
                    wire::CacheChange::Shrink,
                    adds,
                    update.preempt_after,
                    &budgets,
                )
                .await
                .map_err(anyhow::Error::msg)?;
        }
        let eager_since = update.eager_prepare.then(Instant::now);
        if let Some(since) = eager_since {
            for (replica, client) in previous.iter().zip(&serving) {
                let idle = crate::interference::eager_baseline(
                    &replica.observer,
                    client.world(),
                    client.info().schedulers,
                    since,
                )
                .await;
                if !idle.is_empty() {
                    replica
                        .progress_note(format!("idle schedulers {idle:?} have no eager baseline"));
                }
            }
        }
        let preparing = Instant::now();
        let observed = previous
            .iter()
            .zip(&serving)
            .map(|(replica, client)| {
                let throttling = crate::interference::Throttling::new(
                    &replica.observer,
                    client.world(),
                    update.transfer_slowdown,
                    preparing,
                    eager_since,
                );
                (Arc::clone(&replica.observer), throttling)
            })
            .collect();
        let done = tokio_util::sync::CancellationToken::new();
        let task = tokio::spawn(crate::interference::control(
            Arc::clone(&serving[0]),
            observed,
            crate::interference::INTERVAL,
            done.clone(),
        ));
        interference = Some((done, task));
        let mut starting = Vec::new();
        for (ordinal, spec) in specs.iter().cloned().enumerate() {
            let control = Arc::clone(&control);
            let observer = Arc::new(Observer::default());
            let sink = stats_sink(target.model().to_owned(), Arc::clone(&observer), ordinal);
            starting.push(tokio::task::spawn_blocking(move || {
                control.start(spec, sink)
            }));
            observers.push(observer);
        }
        let mut failure = None;
        for started in futures::future::join_all(starting).await {
            match started
                .context("the successor startup task did not complete")
                .and_then(|started| started)
            {
                Ok(candidate) => candidates.push(Arc::new(candidate)),
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        // Admission and migration use this same predecessor-to-successor
        // assignment. Each source is leased by only one target owner.
        let mut warming = Vec::new();
        for (ordinal, candidate) in candidates.iter().enumerate() {
            let sources: Vec<_> = serving
                .iter()
                .enumerate()
                .filter(|(index, _)| index % candidates.len() == ordinal)
                .map(|(_, client)| Arc::clone(client))
                .collect();
            let changed = sources
                .iter()
                .any(|source| source.info().prefix_seeds != candidate.info().prefix_seeds);
            warming.push(async move {
                let recompute = changed && request.in_flight == InFlight::Recompute;
                if recompute && !adds.host_cache {
                    Ok(warming::Report {
                        recompute: Some("the update recomputes in-flight state".into()),
                        ..warming::Report::default()
                    })
                } else {
                    warming::prepare(
                        &sources,
                        candidate,
                        if recompute {
                            warming::Reuse::Recompute
                        } else if changed {
                            warming::Reuse::Keep
                        } else {
                            warming::Reuse::Exact
                        },
                        update.cache_tail_tokens.map(std::num::NonZeroU32::get),
                        update.cache_prefill_tokens.map(std::num::NonZeroU32::get),
                        adds.host_cache,
                    )
                    .await
                }
            });
        }
        for result in futures::future::join_all(warming).await {
            warmed.push(result.map_err(anyhow::Error::msg)?);
        }
        Ok(())
    }
    .await;
    let throttled = if let Some((done, task)) = interference {
        done.cancel();
        task.await.unwrap_or_default()
    } else {
        crate::interference::Throttled::default()
    };
    if let Err(error) = result {
        restore(
            &previous,
            &serving,
            &candidates,
            &parked,
            update.eager_prepare,
            &warmed
                .iter()
                .flat_map(|report| report.backup.clone())
                .collect::<Vec<_>>(),
        )
        .await;
        return Err(error
            .context("the group successor did not prepare; predecessors serve on")
            .into());
    }
    let promoted = Instant::now();
    let mut park_time = Duration::ZERO;
    let paused = previous.iter().all(|replica| replica.hold.is_paused());
    for (index, replica) in previous.iter().enumerate() {
        replica.progress("promote", None);
        let before = serving[index].info();
        let after = candidates[index % candidates.len()].info();
        if request.in_flight == InFlight::Keep {
            replica.hold.keep(Some(Kept {
                world: serving[index].world(),
                prefix_seeds: (before.prefix_seeds != after.prefix_seeds)
                    .then(|| before.prefix_seeds.clone()),
                latent_seed: (before.latent_seed != after.latent_seed)
                    .then_some(before.latent_seed),
            }));
        }
        replica.hold.hold();
    }
    let result: anyhow::Result<()> = async {
        for (replica, client) in previous.iter().zip(&serving) {
            crate::profiling::end_for_world(replica, Some(client), "group change").await;
            client.park().await.map_err(anyhow::Error::msg)?;
            parked.push(client.world());
        }
        park_time = promoted.elapsed();
        warming::resume_prefill(&serving)
            .await
            .map_err(anyhow::Error::msg)?;
        if adds.host_cache {
            for source in &serving {
                source
                    .resize_cache(wire::CacheChange::Evict)
                    .await
                    .map_err(anyhow::Error::msg)?;
            }
            for candidate in &candidates {
                warming::materialize(candidate)
                    .await
                    .map_err(anyhow::Error::msg)?;
            }
        }
        for (candidate, warmed) in candidates.iter().zip(&warmed) {
            candidate
                .promote(
                    Vec::new(),
                    warmed.recompute.is_some(),
                    update.previous_world != PreviousWorld::Parked,
                )
                .await?;
        }
        Ok(())
    }
    .await;
    if let Err(error) = result {
        restore(
            &previous,
            &serving,
            &candidates,
            &parked,
            update.eager_prepare,
            &warmed
                .iter()
                .flat_map(|report| report.backup.clone())
                .collect::<Vec<_>>(),
        )
        .await;
        return Err(error
            .context("the group promotion failed; predecessors resumed")
            .into());
    }
    let mut current = Vec::new();
    for (ordinal, (client, mut spec)) in candidates.iter().zip(specs).enumerate() {
        spec.world = client.world();
        spec.received_weights.clear();
        spec.stage_cache = false;
        spec.cache_keys.clear();
        let replica = Replica::new(
            &target,
            Arc::clone(client),
            Arc::clone(&observers[ordinal]),
            ordinal,
            Arc::downgrade(&leader.replicas),
            Arc::clone(&leader.replicas.adapters),
        );
        let _ = replica.control.set(Arc::clone(&control));
        *locked(&replica.served) = Some(spec);
        crate::leader::reload_adapters(&replica, client).await;
        if paused {
            replica.hold.pause();
        }
        current.push(replica);
    }
    let pause = promoted.elapsed();
    *locked(&leader.replicas.group) = Some(Trial {
        previous: previous.clone(),
        current: current.clone(),
        retained: serving
            .iter()
            .map(|client| Previous::Parked(Arc::clone(client)))
            .collect(),
        cache_backup: warmed
            .iter()
            .flat_map(|report| report.backup.clone())
            .collect(),
        digest: target_digest,
        mode: update.previous_world,
        host_refused: None,
        promoted,
        promoted_at: now(),
        baseline,
        thresholds: update.rollback.clone(),
        outcomes: crate::leader::outcomes(),
        eager: update.eager_prepare,
        pause,
    });
    for replica in &current {
        replica.monitor();
    }
    for (index, replica) in previous.iter().enumerate() {
        replica
            .hold
            .redirect(Arc::clone(&current[index % current.len()].hold));
        replica.progress_done();
    }
    leader.replicas.serving.send_replace(current.clone());
    leader.replicas.schedulers.store(
        candidates
            .iter()
            .map(|client| client.info().schedulers as usize)
            .sum(),
        Ordering::Relaxed,
    );
    match update.previous_world {
        PreviousWorld::Parked => {}
        PreviousWorld::Weights | PreviousWorld::Host => {
            for client in &serving {
                close(client).await;
            }
            if let Some(trial) = locked(&leader.replicas.group).as_mut() {
                trial.retained = serving.iter().map(|_| Previous::Weights).collect();
            }
            for client in &candidates {
                let refusal = match client
                    .grow_keeping_weights(
                        source_worlds.clone(),
                        update.previous_world == PreviousWorld::Host,
                        warmed
                            .iter()
                            .flat_map(|report| warming::keys(&report.backup))
                            .collect(),
                    )
                    .await
                {
                    Ok(nodes) => {
                        let refusals: Vec<_> = nodes
                            .iter()
                            .filter_map(|node| node["refused"].as_str())
                            .collect();
                        (!refusals.is_empty()).then(|| refusals.join("; "))
                    }
                    Err(error) => Some(error),
                };
                if let Some(refusal) = refusal {
                    warn!(%refusal, "the previous group keeps its weights on the GPUs");
                    if let Some(trial) = locked(&leader.replicas.group).as_mut() {
                        trial.mode = PreviousWorld::Weights;
                        trial.host_refused = Some(refusal);
                    }
                }
            }
        }
    }
    for client in &candidates {
        if let Err(error) = client.resize_cache(wire::CacheChange::Grow).await {
            warn!(%error, "the successor did not grow its cache");
        }
    }
    #[expect(
        clippy::unwrap_used,
        reason = "the deployment replacement lock retains the group trial through promotion"
    )]
    let mode = locked(&leader.replicas.group).as_ref().unwrap().mode;
    let report = serde_json::json!({
        "world": (current.len() == 1).then(|| worlds(&current)[0]),
        "worlds": worlds(&current),
        "previous_worlds": source_worlds,
        "prepare_seconds": promoted.duration_since(started).as_secs_f64(),
        "pause_seconds": pause.as_secs_f64(),
        "park_seconds": park_time.as_secs_f64(),
        "parallel": "changed",
        "redistribution": plans,
        "cache_warming": warmed,
        "kv_cache": if warmed.iter().any(|report| report.recompute.is_some()) { "recomputed" } else { "warmed" },
        "previous_world_kept": mode,
        "transfer_halvings": throttled.halvings,
        "transfer_restorations": throttled.restorations,
        "plan": plan,
    });
    for replica in &current {
        locked(&replica.worlds).last = Some(report.clone());
    }
    tokio::spawn(watch_group(Arc::clone(&leader.replicas), worlds(&current)));
    Ok(Applied::World(report))
}

/// Records one commit decision for every successor, then closes retained
/// predecessors. The caller holds the deployment replacement lock.
pub(crate) async fn decide_commit(registry: &Arc<Replicas>) -> Result<(), UpdateError> {
    let current = locked(&registry.group)
        .as_ref()
        .context("no group is on trial")?
        .current
        .clone();
    let _owners = lock_replicas(&current).await;
    let trial = locked(&registry.group)
        .take()
        .context("no group is on trial")?;
    begin_commit(registry, &current, worlds(&trial.previous));
    for retained in trial.retained {
        if let Previous::Parked(client) = retained {
            close(&client).await;
        }
    }
    Ok(())
}

/// Restores all predecessor owners. The caller holds the deployment lock.
pub(crate) async fn rollback(
    registry: &Arc<Replicas>,
    reason: String,
) -> Result<serde_json::Value, UpdateError> {
    let current = locked(&registry.group)
        .as_ref()
        .context("no group is on trial")?
        .current
        .clone();
    let _owners = lock_replicas(&current).await;
    let trial = locked(&registry.group)
        .take()
        .context("no group is on trial")?;
    let retiring = worlds(&current);
    let paused = current.iter().all(|replica| replica.hold.is_paused());
    for replica in &current {
        if let Some(client) = replica.hold.client() {
            crate::profiling::end_for_world(replica, Some(&client), "group rollback").await;
            replica.hold.hold();
            if client.state() != EngineState::Closed {
                let _ = tokio::time::timeout(CLOSED_EXIT, client.park()).await;
            }
            close(&client).await;
        }
    }
    let mut failures = Vec::new();
    for (replica, retained) in trial.previous.iter().zip(trial.retained) {
        let restored: Result<Arc<EngineClient>, UpdateError> = async {
            Ok(match retained {
                Previous::Parked(client) => {
                    client
                        .resume(retiring.clone())
                        .await
                        .map_err(anyhow::Error::msg)?;
                    replay_graphs(&client, trial.eager).await;
                    client
                }
                Previous::Weights => {
                    let control = Arc::clone(replica.control.get().ok_or(UpdateError::NoEngine)?);
                    let mut spec = locked(&replica.served)
                        .clone()
                        .ok_or(UpdateError::NoEngine)?;
                    spec.retire = retiring.clone();
                    spec.cache_keys = warming::keys(&trial.cache_backup);
                    let sink = stats_sink(
                        replica.model_name.clone(),
                        Arc::clone(&replica.observer),
                        replica.ordinal,
                    );
                    Arc::new(
                        tokio::task::spawn_blocking(move || control.start(spec, sink))
                            .await
                            .context("the predecessor restart task did not complete")??,
                    )
                }
            })
        }
        .await;
        locked(&replica.worlds).unavailable.insert(
            trial.digest.clone(),
            Unavailable {
                digest: trial.digest.clone(),
                world: retiring[0],
                reason: reason.clone(),
                at: now(),
            },
        );
        let client = match restored {
            Ok(client) => client,
            Err(error) => {
                let reason = format!("replica {} did not resume: {error}", replica.ordinal);
                replica.hold.close(reason.clone());
                failures.push(reason);
                continue;
            }
        };
        replica.hold.keep(None);
        if let Err(error) = warming::restore(&client, &trial.cache_backup).await {
            replica
                .hold
                .close(format!("the predecessor cache did not restore: {error}"));
            failures.push(error);
            continue;
        }
        if paused {
            replica.hold.pause();
        } else {
            replica.hold.resume();
        }
        crate::leader::reload_adapters(replica, &client).await;
        replica.hold.serve(Arc::clone(&client));
        replica.restore_tail(&client).await;
        watch(Arc::clone(replica), Arc::downgrade(&client));
    }
    for (index, replica) in current.iter().enumerate() {
        replica.hold.redirect(Arc::clone(
            &trial.previous[index % trial.previous.len()].hold,
        ));
    }
    registry.serving.send_replace(trial.previous.clone());
    registry.schedulers.store(
        locked(&trial.previous[0].config).data_parallel_size.get(),
        Ordering::Relaxed,
    );
    let report = serde_json::json!({
        "worlds": worlds(&trial.previous),
        "rolled_back": retiring,
        "reason": reason,
        "failures": failures,
    });
    for replica in &trial.previous {
        locked(&replica.worlds).last = Some(report.clone());
    }
    if !failures.is_empty() {
        return Err(anyhow::anyhow!("group rollback: {}", failures.join("; ")).into());
    }
    Ok(report)
}

async fn watch_group(registry: Arc<Replicas>, watched: Vec<u64>) {
    loop {
        tokio::time::sleep(TRIAL_POLL).await;
        let reason = {
            let state = locked(&registry.group);
            let Some(trial) = state.as_ref() else { return };
            if worlds(&trial.current) != watched {
                return;
            }
            fault(trial)
        };
        if let Some(reason) = reason {
            let _deployment = registry.replacing.lock().await;
            if locked(&registry.group)
                .as_ref()
                .is_some_and(|trial| worlds(&trial.current) == watched)
                && let Err(error) = rollback(&registry, reason).await
            {
                tracing::error!(%error, "the group rollback failed");
            }
            return;
        }
    }
}

fn fault(trial: &Trial) -> Option<String> {
    for replica in &trial.current {
        let world = locked(&replica.worlds).executing;
        match replica.hold.client().map(|client| client.state()) {
            Some(EngineState::Failed(reason)) => {
                return Some(format!("world {world} failed: {reason}"));
            }
            Some(EngineState::Closed) | None => return Some(format!("world {world} exited")),
            _ => {}
        }
        if replica.observer.numerical_faults(world, trial.promoted) > 0 {
            return Some(format!("world {world} produced non-finite logits"));
        }
    }
    if let Some(limit) = trial.thresholds.failure_rate {
        let (ended, failed) = crate::leader::outcomes();
        let ended = ended.saturating_sub(trial.outcomes.0);
        let failed = failed.saturating_sub(trial.outcomes.1);
        if ended > 0 && failed as f64 / ended as f64 > limit {
            return Some(format!(
                "{failed} of {ended} requests failed during the group trial"
            ));
        }
    }
    if let (Some(window), Some(baseline)) = (trial.thresholds.window, trial.baseline)
        && trial.promoted.elapsed() >= window
        && let Some(current) = rates(&trial.current, trial.promoted, window)
    {
        if let Some(limit) = trial.thresholds.step_time_regression
            && baseline.step_us > 0.0
            && current.step_us > baseline.step_us * (1.0 + limit)
        {
            return Some("group step time exceeded the configured regression threshold".into());
        }
        if let Some(limit) = trial.thresholds.throughput_regression
            && baseline.tokens_per_second > 0.0
            && current.tokens_per_second < baseline.tokens_per_second * (1.0 - limit)
        {
            return Some("group throughput exceeded the configured regression threshold".into());
        }
    }
    None
}
