//! The scheduler's part in its world's changes.
//!
//! A deployment updates by starting a world that succeeds the serving one on
//! the same devices; the daemon decides each step, and each scheduler acts:
//!
//! - **Park** (the serving world, at a promotion): at its next step boundary
//!   the scheduler ends every request that resumes from its generated tokens
//!   as migratable, lets the others finish, and once nothing runs, offloads
//!   the pages it reserved for the host KV cache and executes nothing more.
//!   It answers with its prefix cache index: every published page with its
//!   units, which the successor reads from the planes it maps read-only.
//!   Its memory and index stay as they were, so a rollback resumes it.
//! - **Promote** (the successor): its workers copy the predecessor's cache
//!   flags and scales of the units they inherit (`uniserve.adopt_cache`),
//!   and its pool imports the predecessor's pages into those units, from
//!   which migrated requests reuse their prefixes.
//! - **Resume** (the parked world, at a rollback): it admits and executes
//!   again, with the index it parked with.
//! - **Commit** (the successor, once the predecessor is gone): its workers
//!   take the segments they mapped read-only (`uniserve.take_cache`), its
//!   pool allocates the inherited units, and its worker groups relaunch on
//!   a loss and keep their standbys from then on.
//!
//! Until its commit a successor is on trial: a worker fault or non-finite
//! logits fail it as a whole rather than being repaired, and its requests
//! that resume from their generated tokens leave as migratable
//! (`Scheduler::hand_back_requests`), for the daemon to continue on the
//! previous world.

use super::*;
use crate::kv_cache::IndexedPage;

/// One worker group's KV cache as a world that succeeds this one maps it:
/// its segments in unit order, the leading ones it holds, the units of its
/// planes and the warmup scratch it lent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupCache {
    pub worker_group: String,
    pub segments: Vec<ipc::CacheSegment>,
    pub held: u32,
    pub units: u32,
    pub scratch: u32,
}

/// What a worker reports of its KV cache (`uniserve.cache_memory`).
#[derive(serde::Deserialize)]
struct CacheReport {
    #[serde(default)]
    segments: Vec<ipc::CacheSegment>,
    #[serde(default)]
    held: u32,
    #[serde(default)]
    scratch_units: u32,
}

/// A park waiting for the scheduler's requests to end.
pub(super) struct Parking {
    answer: crossbeam_channel::Sender<Result<Vec<IndexedPage>, String>>,
}

impl Scheduler {
    /// Parks the scheduler (see the module documentation), answering with
    /// its prefix cache index once it is parked.
    pub(super) fn park(
        &mut self,
        answer: crossbeam_channel::Sender<Result<Vec<IndexedPage>, String>>,
    ) {
        if self.parked {
            let _ = answer.send(Ok(self.index()));
            return;
        }
        if self.parking.is_some() {
            let _ = answer.send(Err("the scheduler parks already".to_owned()));
            return;
        }
        tracing::info!("parking: the requests that resume elsewhere end as migratable");
        self.migrate_requests(MigrationCause::Drain);
        self.parking = Some(Parking { answer });
        self.poll_park();
    }

    /// Completes a park once no request is queued, runs or retires and no
    /// batch is in flight.
    pub(super) fn poll_park(&mut self) {
        if self.parking.is_none()
            || !(self.running.is_empty()
                && self.running_media.is_empty()
                && self.retiring_requests.is_empty()
                && self.waiting.is_empty()
                && self.waiting_media.is_empty()
                && self.inflight.pending_batches.is_empty()
                && self.inflight.pending_submissions.is_empty())
        {
            return;
        }
        let Some(parking) = self.parking.take() else {
            return;
        };
        // The pages reserved for the host KV cache reach it, so the
        // successor and a later eviction of the parked memory find them
        // there.
        self.flush_offloads();
        self.parked = true;
        let index = self.index();
        tracing::info!(pages = index.len(), "the scheduler parked");
        let _ = parking.answer.send(Ok(index));
    }

    /// Whether admission and scheduling wait for a park: the scheduler
    /// parked, or parks once its requests end.
    pub(super) fn parks(&self) -> bool {
        self.parked
    }

    /// Resumes a parked scheduler: it admits and executes again.
    pub(super) fn resume(&mut self) -> Result<(), String> {
        if self.parking.is_some() {
            return Err("the scheduler is still parking".to_owned());
        }
        if self.parked {
            self.parked = false;
            self.closing = false;
            tracing::info!("the parked scheduler resumes");
        }
        Ok(())
    }

    /// Promotes a successor scheduler: its workers adopt the predecessor's
    /// cache flags of the units they inherit, and its pool imports
    /// `index`, the predecessor's published pages, each unit moved to where
    /// the workers map it.
    pub(super) fn promote(&mut self, index: &[IndexedPage]) -> Result<(), String> {
        let Some(cache) = self.storage.cache() else {
            return Ok(());
        };
        let inherited = cache.block_pool.inherited();
        if inherited.is_empty() {
            return Ok(());
        }
        self.run_hook("uniserve.adopt_cache", "")?;
        let source = self
            .executor
            .info()
            .workers
            .iter()
            .find(|(_, info)| {
                info.prefix_cache
                    .as_ref()
                    .is_some_and(|cache| cache.inherited() == inherited)
            })
            .map(|(_, info)| Arc::new(info.endpoint.clone()))
            .ok_or_else(|| "no worker maps the inherited cache units".to_owned())?;
        let Some(cache) = self.storage.cache() else {
            return Ok(());
        };
        let imported = cache.block_pool.import(index, inherited.start, &source);
        tracing::info!(
            pages = imported,
            offered = index.len(),
            units = inherited.end - inherited.start,
            "imported the predecessor's prefix cache index"
        );
        Ok(())
    }

    /// Commits a successor scheduler: its workers take the predecessor's
    /// segments they mapped read-only, its pool allocates their units, and
    /// its worker groups relaunch on a loss from then on. Returns what each
    /// worker reported.
    pub(super) fn commit(&mut self) -> Result<Vec<String>, String> {
        let closing = std::mem::replace(&mut self.closing, true);
        while !(self.inflight.pending_batches.is_empty()
            && self.inflight.pending_submissions.is_empty())
        {
            if self.fatal {
                self.closing = closing;
                return Err("the engine failed before its commit".to_owned());
            }
            if !self.step_nonblocking() {
                self.park_for_progress();
            }
        }
        self.closing = closing;
        let results = self
            .executor
            .commit()
            .map_err(|error| format!("{error:#}"))?;
        if let Some(cache) = self.storage.cache() {
            cache.block_pool.take_inherited();
        }
        self.trial = false;
        tracing::info!(
            units = self.storage.usable_units(),
            "the scheduler's world is committed"
        );
        Ok(results)
    }

    /// The KV cache every worker group holds now, as a successor maps it;
    /// the ranks of a group agree on it.
    pub(super) fn describe_cache(&mut self) -> Result<Vec<GroupCache>, String> {
        let units: BTreeMap<String, u32> = self
            .executor
            .info()
            .workers
            .iter()
            .filter_map(|(group, info)| {
                info.prefix_cache
                    .as_ref()
                    .map(|cache| (group.to_string(), cache.num_units))
            })
            .collect();
        let closing = std::mem::replace(&mut self.closing, true);
        while !(self.inflight.pending_batches.is_empty()
            && self.inflight.pending_submissions.is_empty())
        {
            if self.fatal {
                self.closing = closing;
                return Err("the engine failed before it described its cache".to_owned());
            }
            if !self.step_nonblocking() {
                self.park_for_progress();
            }
        }
        self.closing = closing;
        let reports = self
            .executor
            .hook_groups("uniserve.cache_memory", "")
            .map_err(|error| format!("{error:#}"))?;
        let mut groups = Vec::new();
        for (group, results) in reports {
            let Some(units) = units.get(&group.to_string()).copied() else {
                continue;
            };
            let mut described: Option<CacheReport> = None;
            for result in &results {
                let report: CacheReport = serde_json::from_str(result)
                    .map_err(|error| format!("a worker's cache report: {error}"))?;
                match &described {
                    Some(first)
                        if (&first.segments, first.held) != (&report.segments, report.held) =>
                    {
                        return Err(format!(
                            "the ranks of worker group {group} hold different cache segments"
                        ));
                    }
                    Some(_) => {}
                    None => described = Some(report),
                }
            }
            if let Some(report) = described {
                groups.push(GroupCache {
                    worker_group: group.to_string(),
                    segments: report.segments,
                    held: report.held,
                    units,
                    scratch: report.scratch_units,
                });
            }
        }
        Ok(groups)
    }

    /// The scheduler's prefix cache index: every page its pool publishes.
    fn index(&self) -> Vec<IndexedPage> {
        self.storage
            .cache()
            .map(|cache| cache.block_pool.export())
            .unwrap_or_default()
    }
}
