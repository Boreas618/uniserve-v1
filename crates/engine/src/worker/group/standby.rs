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
//! A standby serves a group whose ranks all run on the head's host and
//! that is its own failure domain: a group of a shared expert world is
//! replaced with its domain, and a launcher stops every rank of a group on
//! its host when the group fails.
//!
//! A control-plane handover (`crate::handover`) stands the successor's
//! workers by for the same groups: the predecessor's groups stop their
//! standbys for it (`prepare_handover`), and each successor group promotes
//! its own workers, which started standing by (`promote_workers`).

use super::WorkerGroup;
use super::validate_replacement_info;
use crate::worker::WorkerProcess;

/// The standby rank group of a serving group.
pub(super) struct Standby {
    workers: Vec<WorkerProcess>,
    /// Whether every rank reported its startup and parked.
    ready: bool,
}

impl WorkerGroup {
    /// Whether the group keeps a standby: `worker_config.standby`, for a
    /// group that is its own failure domain on the head's host.
    fn keeps_standby(&self) -> bool {
        self.process_args.keep_standby && !self.handing_over && self.stands_by()
    }

    /// Whether ranks can stand by for the group: it is its own failure
    /// domain on the head's host.
    fn stands_by(&self) -> bool {
        self.failure_domain.0.groups() == 1
            && self
                .process_args
                .ranks
                .iter()
                .all(|rank| rank.host == self.process_args.host)
    }

    /// Predecessor of a handover: stop the standby and keep none, and
    /// return the group's prefix cache units, which the successor's workers
    /// adopt, and the claim they stand by under. Fails for a group no rank
    /// can stand by for.
    pub(crate) fn prepare_handover(&mut self) -> anyhow::Result<(Option<u32>, [u8; 16])> {
        anyhow::ensure!(
            self.stands_by(),
            "worker group {} cannot be handed over: its ranks run on another host or share \
             an expert world",
            self.process_args.worker_id
        );
        // The successor's workers adopt the group's storages and claim; a
        // relaunch in progress would start ranks of its own under them.
        anyhow::ensure!(
            self.awaiting.is_none() && !self.failure_domain.0.failing(),
            "worker group {} is being relaunched and cannot be handed over",
            self.process_args.worker_id
        );
        let claim = self.claims.launch(&self.process_args.host)?.claim;
        self.handing_over = true;
        self.drop_standby();
        Ok((
            self.info.prefix_cache.as_ref().map(|cache| cache.num_units),
            claim,
        ))
    }

    /// Predecessor of a handover that did not happen: keep a standby again.
    pub(crate) fn abandon_handover(&mut self) {
        self.handing_over = false;
        self.start_standby();
    }

    /// Successor of a handover, once the predecessor's workers closed: take
    /// the deployment's claim over and promote the group's workers, which
    /// started standing by. The group then serves, and relaunches and
    /// standbys start as at any start.
    pub(crate) fn promote_workers(&mut self) -> anyhow::Result<()> {
        self.claims.take_over()?;
        let timeout = self.process_args.watchdog_timeout;
        for worker in &mut self.workers {
            worker.promote(timeout)?;
        }
        // A relaunch serves at once, over the planes of the adopted units.
        self.process_args.standby = false;
        self.last_progress = std::time::Instant::now();
        self.start_standby();
        Ok(())
    }

    /// Start a standby when the group keeps one and has none. Its ranks
    /// register now and warm up while the group serves; `poll_standby`
    /// follows them. A standby that cannot start is reported and left out:
    /// the group is then replaced by a relaunch when it fails.
    pub(super) fn start_standby(&mut self) {
        if self.closed || self.standby.is_some() || !self.keeps_standby() {
            return;
        }
        let mut args = self.process_args.clone();
        args.standby = true;
        args.cache_units = self.info.prefix_cache.as_ref().map(|cache| cache.num_units);
        let started = (|| -> anyhow::Result<Vec<WorkerProcess>> {
            let launched = args.spawn_workers(None, &self.registry, None, &self.claims, None)?;
            let (mut workers, _) = args.adopt_workers(launched)?;
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
            polled = standby
                .workers
                .iter()
                .zip(&self.workers)
                .enumerate()
                .try_for_each(|(rank, (standby, serving))| {
                    validate_replacement_info(serving.info(), standby.info(), rank)
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

    /// Promote the parked standby in place of the failed ranks, which the
    /// caller terminated, and start a new standby. Returns `None` without a
    /// parked standby; otherwise whether the promotion succeeded, after
    /// which the standby's ranks serve as the group's.
    pub(super) fn promote_standby(&mut self) -> Option<anyhow::Result<()>> {
        if !self.standby_parked() {
            return None;
        }
        let mut standby = self.standby.take()?;
        let timeout = self.process_args.watchdog_timeout;
        let promoted = standby
            .workers
            .iter_mut()
            .try_for_each(|worker| worker.promote(timeout));
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
