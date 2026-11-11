//! Giving back and restoring the KV cache's device storage.
//!
//! A worker splits its cache planes into segments of whole units
//! (`EpochOperation::CacheStorage` reports them), and gives back the tail from a
//! segment on. Making room for device storage beside the cache (a profiling
//! session's) withdraws the tail's units from allocation, revoking the cached
//! pages they hold (whose host KV cache copies stay), and lets the requests
//! that hold them run to their end, admitting new ones on the units left: the
//! serving epoch's capacity drops, and nothing pauses. Once no request holds a
//! withdrawn unit, the workers give the tail back. A request holding the tail
//! that cannot get a unit to continue, because the requests running hold every
//! unit left to allocation, is recomputed at once: it could not run on to its
//! end. Whenever every running request waits for units the others hold and
//! nothing in flight will free any, none could end: the one of the lowest
//! priority admitted last is recomputed, as vLLM's recompute preemption does,
//! and its units go to the others. A growth creates as many segments of the
//! tail as the storage now holds. A sleep gives back every segment and the CUDA
//! graph storage once no request runs, and admits nothing until a wake
//! restores both; it preempts video requests at their next denoising step
//! boundary, which resume from the trajectories stored there once it wakes.
//! An in-place weight update recomputes the requests as a sleep does and
//! rewrites the weights while the workers sleep, so that the cache's storage
//! holds the new checkpoint as it loads.

use super::quota::{growable, room, room_refusal};
use super::*;
use crate::handle::CacheChange;
use crate::kv_cache::PageRef;
use crate::worker::WorkerGroupId;
use ipc::memory::CacheStorageReport;

/// A change of the cache that waits for the requests holding the withdrawn
/// units to let them go (`Scheduler::resize_cache`,
/// `Scheduler::update_weights`).
pub(super) struct PendingResize {
    change: Pending,
    response: crossbeam_channel::Sender<Result<Vec<String>, String>>,
    /// Requests holding withdrawn units that could not get the units their
    /// next call needs (`Scheduler::note_stuck_holder`): recomputed at the
    /// next poll.
    stuck: Vec<RequestId>,
}

/// What a pending change does once its units are let go.
enum Pending {
    /// The workers give back the segments from this one on; the change
    /// responds with the output of each device it made room on.
    Room { from: u32, output: Vec<String> },
    /// The workers sleep, once no request runs.
    Sleep,
    /// The workers rewrite the weights from the checkpoint at this path,
    /// once no request runs.
    Weights(String),
}

/// Every worker's `EpochOperation::CacheStorage` report: of the workers that hold
/// the cache, and of the others (`Scheduler::storage_reports`).
struct StorageReports {
    cache: Vec<CacheStorageReport>,
    beside: Vec<CacheStorageReport>,
}

impl Pending {
    /// Whether the change waits until no request runs, admitting none.
    fn quiesces(&self) -> bool {
        matches!(self, Self::Sleep | Self::Weights(_))
    }
}

impl Scheduler {
    /// Applies `change` to the KV cache's device storage, and responds with
    /// what each worker reported once it is done. A change whose withdrawn
    /// units requests still hold responds once they let them go
    /// (`Scheduler::poll_resize`).
    pub(super) fn resize_cache(
        &mut self,
        change: CacheChange,
        response: crossbeam_channel::Sender<Result<Vec<String>, String>>,
    ) {
        if self.resizing.is_some() {
            let _ = response.send(Err("another KV cache change is in progress".to_owned()));
            return;
        }
        let started = match change {
            CacheChange::MakeRoom { bytes, devices } => {
                self.start_room(bytes, &devices).map(|(from, output)| {
                    from.map(|from| Pending::Room {
                        from,
                        output: output.clone(),
                    })
                    .ok_or(output)
                })
            }
            CacheChange::Sleep => Ok(self.start_sleep().ok_or_else(Vec::new)),
            CacheChange::SleepAborting => {
                self.abort_held_requests();
                Ok(self.start_sleep().ok_or_else(Vec::new))
            }
            CacheChange::Hold { held } => {
                let _ = response.send(self.hold_batches(held));
                return;
            }
            CacheChange::Grow => {
                let _ = response.send(self.grow_cache());
                return;
            }
            CacheChange::Wake => {
                let _ = response.send(self.wake());
                return;
            }
        };
        match started {
            Ok(Ok(pending)) => {
                self.resizing = Some(PendingResize {
                    change: pending,
                    response,
                    stuck: Vec::new(),
                });
                self.poll_resize();
            }
            // Nothing to give back: the change is done, with its output.
            Ok(Err(output)) => {
                let _ = response.send(Ok(output));
            }
            Err(error) => {
                let _ = response.send(Err(error));
            }
        }
    }

    /// Rewrites the weights in place from the checkpoint at `model_path`,
    /// and responds with what each worker reported. The requests in flight
    /// continue as `in_flight` says:
    ///
    /// - `Keep`: once the batches in flight complete, the workers rewrite
    ///   the weights beside the KV cache, and every request in flight
    ///   continues on the new weights with the cache it reached. The state
    ///   it computes afterwards follows neither the old weights nor the new
    ///   ones, so it publishes no page from then on.
    /// - `Recompute`: the requests running are recomputed and the others
    ///   wait; the workers sleep while they rewrite, as for a sleep.
    /// - `Abort`: every request the scheduler holds ends, aborted, before
    ///   the workers sleep and rewrite.
    pub(super) fn update_weights(
        &mut self,
        model_path: String,
        in_flight: crate::WeightsInFlight,
        response: crossbeam_channel::Sender<Result<Vec<String>, String>>,
    ) {
        if self.resizing.is_some() {
            let _ = response.send(Err("a KV cache change is in progress".to_owned()));
            return;
        }
        if self.asleep {
            let _ = response.send(self.apply_weights(&model_path));
            return;
        }
        match in_flight {
            crate::WeightsInFlight::Keep => {
                let _ = response.send(self.apply_weights_keeping(&model_path));
                return;
            }
            crate::WeightsInFlight::Abort => self.abort_held_requests(),
            crate::WeightsInFlight::Recompute => {}
        }
        self.withdraw_from(0, true);
        // Video requests are preempted at their next denoising step boundary
        // and resume on the new weights, which compute other trajectories:
        // they start from noise.
        self.preempt_media(MigrationCause::Drain, |_| true);
        self.resizing = Some(PendingResize {
            change: Pending::Weights(model_path),
            response,
            stuck: Vec::new(),
        });
        self.poll_resize();
    }

    /// Stops forming batches and waits for those in flight, or starts again.
    /// Every request keeps its state, so a held scheduler serves on from the
    /// step it stopped at.
    fn hold_batches(&mut self, held: bool) -> Result<Vec<String>, String> {
        self.held = held;
        if held {
            self.drain_batches()?;
        }
        Ok(Vec::new())
    }

    /// Whether admission waits for a sleep or a weight update: the
    /// scheduler sleeps, or waits for its requests to finish.
    pub(super) fn admission_sleeps(&self) -> bool {
        self.asleep
            || self.held
            || self.parks()
            || self
                .resizing
                .as_ref()
                .is_some_and(|resize| resize.change.quiesces())
    }

    /// Completes a pending shrink or sleep once no request holds a
    /// withdrawn unit, and a sleep once no request runs either: the workers
    /// give the storage back between batches.
    pub(super) fn poll_resize(&mut self) {
        // A request holding the tail that could not get a unit to continue
        // cannot run on to its end, which the change waits for: it is
        // recomputed now.
        if let Some(resize) = self.resizing.as_mut()
            && !resize.stuck.is_empty()
        {
            let stuck = std::mem::take(&mut resize.stuck);
            tracing::info!(
                requests = stuck.len(),
                "requests holding the withdrawn tail cannot continue: they are recomputed"
            );
            self.migrate_running(MigrationCause::Drain, |id| stuck.contains(&id));
        }
        let Some(resize) = self.resizing.as_ref() else {
            return;
        };
        let held = self
            .storage
            .cache()
            .map_or(0, |cache| cache.pool().withdrawn_held());
        if held > 0
            || (resize.change.quiesces()
                && !(self.running.is_empty() && self.retiring_requests.is_empty()))
        {
            return;
        }
        let Some(resize) = self.resizing.take() else {
            return;
        };
        // A change that fails part of the way leaves the units withdrawn and
        // a sleeping scheduler asleep: a growth or a wake creates whatever the
        // workers gave back.
        let result = match resize.change {
            Pending::Sleep => {
                self.asleep = true;
                self.run_operation(ipc::memory::Sleep)
                    .and_then(|replies| responses(&replies))
                    .inspect(|_| {
                        tracing::info!(
                            "the scheduler sleeps: its KV cache and graph storage are given back"
                        );
                    })
            }
            Pending::Room { from, output } => self
                .run_operation(ipc::memory::ReleaseCache { from })
                .map(|_| {
                    tracing::info!(
                        from,
                        units = self.storage.usable_units(),
                        "gave back the tail of the KV cache"
                    );
                    output
                }),
            Pending::Weights(path) => self.apply_weights(&path),
        };
        let _ = resize.response.send(result);
    }

    /// Rewrites the weights from the checkpoint at `path` while no request
    /// runs. The workers sleep for it unless they sleep already, so that
    /// the cache's storage holds the new checkpoint as it loads, and wake
    /// after, whether or not the update succeeded. Every cached prefix page
    /// was computed with the old weights: the prefix cache is flushed, and
    /// the host KV cache keys pages by the new weights' seeds.
    fn apply_weights(&mut self, path: &str) -> Result<Vec<String>, String> {
        let slept = !self.asleep;
        if slept {
            self.asleep = true;
            self.run_operation(ipc::memory::Sleep)?;
        }
        let updated = self.run_operation(ipc::memory::UpdateWeights);
        if slept && let Err(waking) = self.wake() {
            return Err(match updated {
                Ok(_) => format!("the workers did not wake after the weight update: {waking}"),
                Err(error) => format!("{error}; the workers did not wake after: {waking}"),
            });
        }
        let results = updated?;
        let pages = self.storage.flush_prefix_cache();
        self.reseed_pages(&results)?;
        tracing::info!(path, pages, "updated the weights in place");
        responses(&results)
    }

    /// Rewrites the weights from the checkpoint at `path` while the requests
    /// in flight hold their KV cache (`WeightsInFlight::Keep`): the workers
    /// load the checkpoint beside the cache once the batches in flight
    /// complete. Every request in flight then publishes no page, and every
    /// cached prefix page is revoked, as the new weights key pages anew.
    fn apply_weights_keeping(&mut self, path: &str) -> Result<Vec<String>, String> {
        let results = self.run_operation(ipc::memory::UpdateWeights)?;
        let state_seeds = self
            .storage
            .cache()
            .map(|cache| cache.seeds().iter().map(|seed| seed.0).collect::<Vec<_>>())
            .unwrap_or_default();
        let mut kept = 0;
        for (_, request) in self.requests.admitted_mut() {
            let Some(state) = request.tokens_mut() else {
                continue;
            };
            // Repeated updates retain the original private lineage. New
            // requests alone begin under the newly published stage keys.
            if state.req.cache.state_seeds.is_empty() {
                state.req.cache.state_seeds = state_seeds.clone();
            }
            if state.req.cache.write {
                state.req.cache.write = false;
                kept += 1;
            }
        }
        let pages = self.storage.flush_prefix_cache();
        self.reseed_pages(&results)?;
        tracing::info!(
            path,
            pages,
            kept,
            "updated the weights in place; the requests in flight keep their KV cache"
        );
        responses(&results)
    }

    /// Ends every request the scheduler holds, aborted: those queued, and
    /// those running once their batches in flight drain.
    fn abort_held_requests(&mut self) {
        let mut held: Vec<RequestId> = self.running.clone();
        held.extend(self.waiting.iter().chain(&self.held_for_index).copied());
        tracing::info!(
            requests = held.len(),
            "aborting the requests in flight for a weight update"
        );
        for id in held {
            self.mark_cancelled(id, true);
        }
    }

    /// Roots the page chains of every cache level at the page identities
    /// the weight update `results` report (one per worker).
    fn reseed_pages(&mut self, results: &[ipc::memory::WeightsUpdated]) -> Result<(), String> {
        if let Some(cache) = self.storage.cache.as_mut() {
            // Pipeline stages contribute different layers to the same
            // scheduler group. Union their stages just as startup does.
            let mut groups = vec![std::collections::BTreeSet::new(); cache.groups().len()];
            for result in results
                .iter()
                .filter(|result| !result.identities.is_empty())
            {
                if result.identities.len() != groups.len() {
                    return Err("updated cache identities do not cover every group".into());
                }
                for (group, identity) in groups.iter_mut().zip(&result.identities) {
                    group.extend(identity.split('|').map(str::to_owned));
                }
            }
            if groups.iter().all(|group| !group.is_empty()) {
                cache.set_seeds(
                    groups
                        .into_iter()
                        .map(|group| {
                            crate::kv_cache::prefix_seed(
                                &group.into_iter().collect::<Vec<_>>().join("|"),
                            )
                        })
                        .collect(),
                );
            }
        }
        // The trajectories the new weights compute are keyed apart from the
        // old ones.
        if let Some(store) = self.trajectories.as_mut() {
            let identities: Vec<&str> = results
                .iter()
                .filter(|result| {
                    result
                        .group
                        .as_ref()
                        .is_some_and(|group| store.serves(&crate::WorkerGroupId(group.clone())))
                })
                .filter_map(|result| result.identity.as_deref())
                .collect();
            if !identities.is_empty() {
                store.reseed(storage::key::latent_seed(identities));
            }
        }
        Ok(())
    }

    /// The segments each worker that holds the cache reports, which agree
    /// on their segments and the ones they hold, with its device's usage;
    /// `None` when the cache is not split into segments. A worker that
    /// holds no cache, such as an expert worker, reports none and is left
    /// out.
    pub(super) fn cache_storage(&mut self) -> Result<Option<Vec<CacheStorageReport>>, String> {
        Ok(self.storage_reports()?.map(|reports| reports.cache))
    }

    /// Every worker's storage report: those of the workers that hold the
    /// cache, which agree on their segments and the ones they hold, and
    /// those of the others (an expert worker's, with its GPU's usage when
    /// it keeps storage at the Storage Server). `None` when the cache is not
    /// split into segments.
    fn storage_reports(&mut self) -> Result<Option<StorageReports>, String> {
        if self.storage.cache().is_none() {
            return Ok(None);
        }
        let (cache, beside): (Vec<CacheStorageReport>, Vec<CacheStorageReport>) = self
            .run_operation(ipc::memory::CacheStorage)?
            .into_iter()
            .partition(|report| !report.segments.is_empty());
        let Some(first) = cache.first() else {
            return Ok(None);
        };
        if cache
            .iter()
            .any(|report| (&report.segments, report.held) != (&first.segments, first.held))
        {
            return Err("the workers split the KV cache into different segments".to_owned());
        }
        Ok(Some(StorageReports { cache, beside }))
    }

    /// The segments of a cache split as `first` reports that it cannot give
    /// back: the first, which holds the padding and warmup units, and every
    /// segment it inherited and does not own yet.
    fn kept_floor(&mut self, first: &CacheStorageReport) -> u32 {
        let inherited = self
            .storage
            .cache()
            .map_or(0..0, |cache| cache.pool().inherited());
        (1..first.held)
            .rev()
            .find(|&index| first.first_unit(index) < inherited.end)
            .map_or(1, |index| index + 1)
    }

    /// Withdraws the tail past what leaves every device's usage room for
    /// the requested device storage beside everything it
    /// reserves ([`room`]); the requests holding it run on. Returns the first
    /// segment to give back, `None` when every device has the room already
    /// (or the cache is not split into segments, which gives back nothing),
    /// and each device's output.
    ///
    /// Fails, withdrawing nothing, when even the segments the cache cannot
    /// give back leave a device too little room: the error names every
    /// device's bytes and usage entries.
    fn start_room(
        &mut self,
        bytes: u64,
        devices: &std::collections::BTreeMap<storage::manifest::Gpu, u64>,
    ) -> Result<(Option<u32>, Vec<String>), String> {
        let Some(StorageReports {
            cache: reports,
            beside,
        }) = self.storage_reports()?
        else {
            return Ok((None, Vec::new()));
        };
        let held = reports.first().map_or(0, |first| first.held);
        let floor = reports.first().map_or(0, |first| self.kept_floor(first));
        let requested = |report: &CacheStorageReport| {
            report
                .usage
                .as_ref()
                .and_then(|usage| devices.get(&usage.gpu))
                .copied()
                .unwrap_or(bytes)
        };
        let rooms = reports
            .iter()
            .map(|report| room(report, requested(report), floor))
            .collect::<Result<Vec<_>, _>>()?;
        let keep = rooms
            .iter()
            .map(|room| room.segments_kept)
            .min()
            .unwrap_or(held);
        // Dedicated expert ranks have no KV tail to release, but their
        // allocation requirements still participate in admission.
        let mut rooms = rooms;
        for report in &beside {
            if requested(report) != 0 {
                rooms.push(room(report, requested(report), 0)?);
            }
        }
        if rooms.iter().any(|room| !room.feasible) {
            return Err(room_refusal(&rooms));
        }
        let schedules = rooms
            .iter()
            .map(|room| serde_json::to_string(room).unwrap_or_default())
            .collect();
        if keep >= held {
            return Ok((None, schedules));
        }
        let unit = reports[0].first_unit(keep);
        self.offload_idle_from(unit);
        self.withdraw_from(unit, false);
        Ok((Some(keep), schedules))
    }

    /// Holds every idle cached page a withdrawal from `unit` evicts and has
    /// the host KV cache verify or make its copy under the key it was
    /// offloaded with: a page the pool holds already is released at once,
    /// another once its copy commits. Pages of a sliding-window group, and
    /// pages never offloaded (no host KV cache serves their worker), are
    /// evicted without a copy: a later prompt computes them again. Returns
    /// how many pages wait for their copies.
    fn offload_idle_from(&mut self, unit: u32) -> usize {
        let Some(cache) = self.storage.cache() else {
            return 0;
        };
        let idle = cache.pool().acquire_idle_from(unit);
        self.offload_pages(idle)
    }

    /// Offloads exactly the pages a policy named, by their device units.
    ///
    /// A dispatch output's `offload` list names units, not a range: asking the pool for
    /// everything idle from the first unit on would move pages the policy
    /// judged worth keeping.
    pub(super) fn offload_units(&mut self, units: &[u32]) -> usize {
        let Some(cache) = self.storage.cache() else {
            return 0;
        };
        let idle = cache.pool().acquire_idle_units(units);
        self.offload_pages(idle)
    }

    /// Hands idle pages to the host KV cache, grouped by the worker group and
    /// group index they belong to. Returns how many pages went.
    ///
    /// A page whose worker group does not serve it, or whose group is not
    /// full-attention, is dropped uncopied: the withdrawal that owns it evicts
    /// it then.
    fn offload_pages(&mut self, idle: Vec<crate::kv_cache::IdlePage>) -> usize {
        if idle.is_empty() {
            return 0;
        }
        let Some(kv_offload) = self.kv_offload.as_mut() else {
            return 0;
        };
        let workers: Vec<(WorkerGroupId, ipc::MainProcessEndpoint)> = self
            .worker
            .info()
            .workers
            .iter()
            .map(|(id, worker)| (id.clone(), worker.endpoint.clone()))
            .collect();
        let mut offloads: BTreeMap<(WorkerGroupId, usize), (Vec<storage::Key>, Vec<PageRef>)> =
            BTreeMap::new();
        for page in idle {
            let worker = workers
                .iter()
                .find(|(_, endpoint)| *endpoint == *page.source)
                .map(|(id, _)| id.clone());
            let group = page.group as usize;
            let full = worker.as_ref().is_some_and(|worker| {
                kv_offload.serves(worker)
                    && matches!(
                        kv_offload.managers(worker).get(group),
                        Some(crate::kv_cache::manager::AttentionManager::Full(_))
                    )
            });
            // Dropping the page lets the withdrawal evict it uncopied.
            if let (Some(worker), Some(key), true) = (worker, page.host_key, full) {
                let entry = offloads.entry((worker, group)).or_default();
                entry.0.push(key);
                entry.1.push(page.page);
            }
        }
        let mut waiting = 0;
        for ((worker, group), (keys, pages)) in offloads {
            waiting += pages.len();
            kv_offload.offload(
                &worker,
                group,
                keys,
                pages,
                crate::kv_cache::OffloadKind::Evicted,
            );
        }
        if waiting > 0 {
            tracing::info!(
                pages = waiting,
                "the idle cached pages of the withdrawn units go to the host KV cache first"
            );
        }
        waiting
    }

    /// Withdraws every unit and lets the requests that run finish or
    /// recompute; admission waits until a wake. `None` for a scheduler
    /// asleep already.
    fn start_sleep(&mut self) -> Option<Pending> {
        if self.asleep {
            return None;
        }
        self.withdraw_from(0, true);
        // Video requests are preempted at their next denoising step boundary
        // and resume once the workers wake.
        self.preempt_media(MigrationCause::Drain, |_| true);
        Some(Pending::Sleep)
    }

    /// Withdraws the units from `first` on, revoking the cached pages they
    /// hold, and, when `recompute`, ends as migratable every running request
    /// that resumes from its generated tokens and holds one, which the
    /// daemon submits again: it recomputes on the units left. Otherwise the
    /// requests holding them run to their end, or until they cannot get a
    /// unit to continue (`Self::note_stuck_holder`).
    fn withdraw_from(&mut self, first: u32, recompute: bool) {
        let Some(cache) = self.storage.cache() else {
            return;
        };
        let held = cache.pool().withdraw(first);
        let withdrawn = cache.pool().withdrawn();
        tracing::info!(
            first = withdrawn,
            held,
            recompute,
            "withdrew the tail of the KV cache from allocation"
        );
        if recompute {
            self.migrate_holders(withdrawn);
        }
    }

    /// While a change waits for the requests holding withdrawn units to let
    /// them go, a running request that holds one and could not get the
    /// units its next call needs cannot run on: only the units left to
    /// allocation can grow it. `Self::poll_resize` recomputes it.
    pub(super) fn note_stuck_holder(&mut self, id: RequestId) {
        let withdrawing = self
            .resizing
            .as_ref()
            .is_some_and(|resize| matches!(resize.change, Pending::Room { .. }));
        if !withdrawing {
            return;
        }
        let Some(first) = self.storage.cache().map(|cache| cache.pool().withdrawn()) else {
            return;
        };
        let holds = self
            .requests
            .tokens(&id)
            .is_some_and(|state| holds_from(state, first));
        if let Some(resize) = self.resizing.as_mut()
            && holds
            && !resize.stuck.contains(&id)
        {
            resize.stuck.push(id);
        }
    }

    /// Ends as migratable every running request that resumes from its
    /// generated tokens and holds a unit from `first` on.
    fn migrate_holders(&mut self, first: u32) {
        let holders: Vec<RequestId> = self
            .requests
            .running_tokens()
            .filter(|(_, _, state)| holds_from(state, first))
            .map(|(id, _, _)| id)
            .collect();
        self.migrate_running(MigrationCause::Drain, |id| holders.contains(&id));
    }

    /// Creates the first `segments` of the given-back segments again, and
    /// returns their units to allocation.
    fn restore_cache(&mut self, segments: u32) -> Result<Vec<String>, String> {
        if self.asleep {
            return Err("the scheduler sleeps: wake it instead".to_owned());
        }
        let Some(cache) = self.storage.cache() else {
            return Ok(Vec::new());
        };
        if !cache.pool().has_withdrawn() {
            return Ok(Vec::new());
        }
        let results = self.run_operation(ipc::memory::RestoreCache {
            segments: Some(segments),
        })?;
        let end = self
            .cache_storage()?
            .map(|reports| reports[0].first_unit(reports[0].held));
        if let (Some(cache), Some(end)) = (self.storage.cache(), end) {
            cache.pool().restore_to(end);
        }
        tracing::info!(units = self.storage.usable_units(), "restored the KV cache");
        responses(&results)
    }

    /// Creates as many of the given-back segments as every device's usage
    /// has room for within its quota, beside everything the deployment's
    /// epochs and the Transfer Server hold there (and its free storage
    /// holds, beside processes outside the deployment), and returns their
    /// units to allocation: what a previous epoch released is the
    /// successor's on the usage at once.
    fn grow_cache(&mut self) -> Result<Vec<String>, String> {
        let Some(reports) = self.cache_storage()? else {
            return Ok(Vec::new());
        };
        let fit = reports
            .iter()
            .map(|report| growable(report, false))
            .min()
            .unwrap_or(0);
        if fit == 0 {
            return Ok(Vec::new());
        }
        tracing::info!(
            segments = fit,
            "growing the KV cache into the storage freed"
        );
        self.restore_cache(fit)
    }

    /// Restores the graph storage and the cache a sleep gave back, and admits
    /// again.
    fn wake(&mut self) -> Result<Vec<String>, String> {
        if !self.asleep {
            return Ok(Vec::new());
        }
        let results = self.run_operation(ipc::memory::Wake)?;
        if let Some(cache) = self.storage.cache() {
            cache.pool().restore();
        }
        self.asleep = false;
        tracing::info!("the scheduler woke");
        responses(&results)
    }
}

/// The JSON texts of the workers' `replies`, which a cache change responses.
fn responses<R: serde::Serialize>(replies: &[R]) -> Result<Vec<String>, String> {
    ipc::memory::responses(replies).map_err(|error| error.to_string())
}

/// Returns whether the request holds a unit from `first` on.
fn holds_from(state: &TokenState, first: u32) -> bool {
    state.allocations.as_ref().is_some_and(|allocations| {
        allocations
            .block_tables()
            .iter()
            .any(|table| table.unit_ids().iter().any(|unit| unit.0 >= first))
    })
}
