//! A standby rank group beside a serving one (`worker_config.standby`).
//!
//! The standby's ranks start while the group serves. Each maps the weights
//! the serving ranks retained at the host's Storage Server read-only and
//! their exclusive storages (cache planes, buffer pool, latent banks) in
//! standby mode, adopting the serving ranks' cache units; it warms up and
//! captures its graphs on the warmup scratch units, which the scheduler
//! never allocates (`PrefixCacheInfo::scratch_units`), then parks with the
//! storages read-only. When the serving group fails, the group promotes the
//! standby instead of relaunching: each rank takes write access once the
//! Storage Server has the failed rank out, makes its mappings writable at
//! unchanged addresses, and serves. The group then starts a new standby.
//!
//! A standby serves in place a group whose ranks all run on the engine's
//! node and that is its own failure domain. Any other group that keeps a
//! standby (a group of a shared expert world, or on several nodes) offers
//! its launch to its failure domain's standby world (`Relaunch`), which a
//! round of the domain promotes as a whole.

use super::WorkerGroup;
use super::{holding, validate_replacement_info};
use crate::worker::WorkerProcess;

/// The standby rank group of a serving group.
pub(super) struct Standby {
    workers: Vec<WorkerProcess>,
    /// Whether every rank reported its startup and parked.
    ready: bool,
}

impl WorkerGroup {
    /// Whether the group keeps a standby in place: `worker_config.standby`,
    /// for a group that is its own failure domain on the engine's host.
    fn keeps_standby(&self) -> bool {
        self.process_args.keep_standby
            && self.relaunch.0.groups() == 1
            && self
                .process_args
                .ranks
                .iter()
                .all(|rank| rank.host == self.process_args.host)
    }

    /// Start a standby when the group keeps one and has none. Its ranks
    /// register now and warm up while the group serves; `poll_standby`
    /// follows them. A standby that cannot start is reported and left out:
    /// the group is then replaced by a relaunch when it fails.
    pub(super) fn start_standby(&mut self) {
        // A world on trial keeps no standby: a fault rolls it back.
        if self.closed
            || self.standby.is_some()
            || !self.process_args.keep_standby
            || self.process_args.predecessor.is_some()
        {
            return;
        }
        let mut args = self.process_args.clone();
        args.standby = true;
        args.segments = self.cache_table();
        if !self.keeps_standby() {
            // The group's failure domain keeps the standby world.
            let (domain, index) = &self.relaunch;
            domain.offer_standby(*index, args);
            return;
        }
        let started = (|| -> anyhow::Result<Vec<WorkerProcess>> {
            let launched =
                args.spawn_workers(None, &self.registry, &self.processes, &self.claims, None)?;
            let mut workers = args.adopt_workers(launched)?;
            for worker in &mut workers {
                worker.request_startup()?;
            }
            Ok(workers)
        })();
        match started {
            Ok(workers) => {
                tracing::info!(
                    worker = %self.process_args.worker_id,
                    "a standby rank group is starting"
                );
                self.standby = Some(Standby {
                    workers,
                    ready: false,
                });
            }
            Err(error) => tracing::warn!(
                worker = %self.process_args.worker_id,
                "the standby rank group could not start: {error:#}"
            ),
        }
    }

    /// Follow the standby's startup, and notice a parked standby's loss.
    /// A standby that fails, or reports capabilities its serving ranks do
    /// not have, is terminated and left out.
    pub(super) fn poll_standby(&mut self) {
        let Some(standby) = self.standby.as_mut() else {
            return;
        };
        let mut ready = true;
        let mut polled = Ok(());
        for worker in &mut standby.workers {
            match worker.poll_startup() {
                Ok(started) => ready &= started,
                Err(error) => {
                    polled = Err(error);
                    break;
                }
            }
        }
        if polled.is_ok() && ready && !standby.ready {
            // The standby maps the planes as the serving ranks hold them now.
            let planes = self.info.prefix_cache.as_ref();
            polled = standby
                .workers
                .iter()
                .zip(&self.workers)
                .enumerate()
                .try_for_each(|(rank, (standby, serving))| {
                    validate_replacement_info(
                        &holding(serving.info(), planes),
                        standby.info(),
                        rank,
                    )
                });
            if polled.is_ok() {
                standby.ready = true;
                tracing::info!(
                    worker = %self.process_args.worker_id,
                    "the standby rank group is parked"
                );
            }
        }
        if let Err(error) = polled {
            tracing::warn!(
                worker = %self.process_args.worker_id,
                "the standby rank group failed: {error:#}"
            );
            self.drop_standby();
        }
    }

    /// Whether a parked standby stands by to replace the group, once its
    /// startup is followed to the present.
    pub fn standby_ready(&mut self) -> bool {
        self.poll_standby();
        self.standby_parked()
    }

    /// Whether a standby has parked, as last followed.
    pub(super) fn standby_parked(&self) -> bool {
        self.standby.as_ref().is_some_and(|standby| standby.ready)
    }

    /// Terminate the standby's ranks.
    pub(super) fn drop_standby(&mut self) {
        if let Some(mut standby) = self.standby.take() {
            for worker in &mut standby.workers {
                worker.terminate();
            }
        }
    }

    /// The segments the group's planes are made of, as its ranks reported
    /// them, and the leading ones they hold now (`record_held`); `None` for
    /// planes not split.
    pub(super) fn cache_table(&self) -> Option<ipc::CacheTable> {
        self.info
            .prefix_cache
            .as_ref()
            .filter(|cache| !cache.segments.is_empty())
            .map(|cache| ipc::CacheTable {
                segments: cache.segments.clone(),
                held: cache.held_segments,
            })
    }

    /// The deployment's storage claims.
    pub(crate) fn claims(&self) -> &crate::worker::claims::StorageClaims {
        &self.claims
    }

    /// Commits the group's world, which succeeded another
    /// (`WorkerProcessArgs::predecessor`): each rank takes the
    /// predecessor's segments it mapped read-only, once the Storage Server
    /// has their writer out (the predecessor's processes are gone), and the
    /// group relaunches on a loss and keeps its standby from then on.
    /// Returns each rank's report; a group of a world that succeeded none
    /// reports nothing.
    pub(crate) fn commit(&mut self) -> anyhow::Result<Vec<String>> {
        if self.process_args.predecessor.is_none() {
            return Ok(Vec::new());
        }
        let results = self.hook("uniserve.take_cache", "")?;
        self.process_args.predecessor = None;
        // The predecessor's segments are the world's own: the planes inherit
        // none, as a relaunch lays them out.
        if let Some(cache) = self.info.prefix_cache.as_mut() {
            cache.inherited_start = 0;
            cache.inherited_units = 0;
        }
        tracing::info!(
            worker = %self.process_args.worker_id,
            "the group's world is committed: it relaunches on a loss from now on"
        );
        self.start_standby();
        Ok(results)
    }

    /// Runs the worker hook `name` on the group's serving ranks; each rank's
    /// result, in rank order.
    ///
    /// The hooks that give cache segments back or create them again answer
    /// with the segments the rank's planes hold after them, which the group
    /// records (`record_held`). A sleep's answer states none: the ranks that
    /// replace a sleeping group's start awake, and warm up on the planes as
    /// they held them before it.
    pub(crate) fn hook(&mut self, name: &str, payload: &str) -> anyhow::Result<Vec<String>> {
        let results = crate::worker::process::hook_all(
            &mut self.workers,
            name,
            payload,
            self.process_args.watchdog_timeout,
        )?;
        if matches!(
            name,
            "uniserve.release_cache" | "uniserve.restore_cache" | "uniserve.wake"
        ) {
            self.record_held(&results)?;
        }
        Ok(results)
    }

    /// Records the leading segments the group's planes hold, as every rank
    /// answered a hook that gives segments back or creates them again
    /// (`{"held": n}`): the ranks that replace lost ones hold those and
    /// leave the others absent (`cache_table`, `replacement_infos`). Planes
    /// not split record nothing.
    ///
    /// Fails when a rank states no count or another one than its peers: the
    /// ranks of a group give back and create the same segments.
    fn record_held(&mut self, results: &[String]) -> anyhow::Result<()> {
        if self.cache_table().is_none() {
            return Ok(());
        }
        let mut held = None;
        for (rank, result) in results.iter().enumerate() {
            let stated = serde_json::from_str::<serde_json::Value>(result)
                .ok()
                .and_then(|answer| answer["held"].as_u64())
                .and_then(|count| u32::try_from(count).ok())
                .ok_or_else(|| {
                    anyhow::anyhow!("rank {rank} did not state the cache segments it holds")
                })?;
            anyhow::ensure!(
                *held.get_or_insert(stated) == stated,
                "rank {rank} holds {stated} cache segments, another rank {}",
                held.unwrap_or_default()
            );
        }
        if let (Some(held), Some(cache)) = (held, self.info.prefix_cache.as_mut()) {
            cache.held_segments = held;
        }
        Ok(())
    }

    /// Each rank's description as a rank that replaces it must report it
    /// (`validate_replacement_info`): its startup report, with what the
    /// group's planes hold now ([`holding`]).
    pub(super) fn replacement_infos(&self) -> Vec<ipc::WorkerInfo> {
        let planes = self.info.prefix_cache.as_ref();
        self.workers
            .iter()
            .map(|worker| holding(worker.info(), planes))
            .collect()
    }

    /// Record recovery's distribution before publishing it in the live ranks.
    /// Standbys capture the new order through their ordinary startup path.
    pub(crate) fn commit_experts(
        &mut self,
        order: &std::collections::BTreeMap<String, Vec<u32>>,
    ) -> anyhow::Result<Vec<String>> {
        self.drop_standby();
        self.relaunch.0.withdraw_standby(self.relaunch.1);
        self.process_args.expert_order = order.clone();
        let result = self.hook("uniserve.commit_experts", "")?;
        self.start_standby();
        Ok(result)
    }

    /// Promote the parked standby in place of the failed ranks, which the
    /// caller terminated, and start a new standby. Returns `None` without a
    /// parked standby; otherwise whether the promotion succeeded, after
    /// which the standby's ranks serve as the group's.
    pub(super) fn promote_standby(&mut self) -> Option<anyhow::Result<()>> {
        if !self.standby_parked() {
            return None;
        }
        let mut standby = self.standby.take()?;
        let promoted = crate::worker::process::promote_all(
            &mut standby.workers,
            self.process_args.watchdog_timeout,
        )
        .and_then(|()| self.map_replacement_host_cache(&mut standby.workers));
        Some(match promoted {
            Ok(()) => {
                for worker in &mut standby.workers {
                    worker.set_startup_cancel(None);
                }
                self.install_replacement(standby.workers);
                self.start_standby();
                Ok(())
            }
            Err(error) => {
                for worker in &mut standby.workers {
                    worker.terminate();
                }
                Err(error)
            }
        })
    }
}
