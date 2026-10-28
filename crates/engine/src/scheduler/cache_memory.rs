//! Giving back and restoring the KV cache's device memory.
//!
//! A worker splits its cache planes into segments of whole units
//! (`uniserve.cache_memory` reports them), and gives back the tail from a
//! segment on. A shrink makes room for a world that succeeds this one on the
//! same devices: the scheduler withdraws the tail's units from allocation,
//! revoking the cached pages they hold (whose host KV cache copies stay), and
//! lets the requests that hold them run to their end, admitting new ones on
//! the units left: the serving world's capacity drops, and nothing pauses.
//! Once no request holds a withdrawn unit, the workers give the tail back.
//! A shrink may name a bound after which the requests still holding the
//! tail are recomputed instead (ended as migratable, and submitted again by
//! the daemon with their generated tokens, as vLLM's recompute preemption
//! does). A restore creates the tail again, all of it or as many segments
//! as the memory now holds. A sleep gives back every segment and the CUDA
//! graph memory once no request runs, and admits nothing until a wake
//! restores both; it preempts video requests at their next denoising step
//! boundary, which resume from the trajectories stored there once it wakes.
//! An in-place weight update recomputes the requests as a sleep does and
//! rewrites the weights while the workers sleep, so that the cache's memory
//! holds the new checkpoint as it loads.

use super::ledger::{CacheMemory, Preparing, growable, preparing, room, room_refusal};
use super::*;
use crate::executor::WorkerGroupId;
use crate::handle::CacheChange;
use crate::kv_cache::PageRef;

/// A shrink, sleep or weight update that waits for the requests holding
/// the withdrawn units to let them go (`Scheduler::resize_cache`,
/// `Scheduler::update_weights`).
pub(super) struct PendingResize {
    change: Pending,
    answer: crossbeam_channel::Sender<Result<Vec<String>, String>>,
    /// When a shrink recomputes the requests still holding withdrawn units;
    /// `None` while it waits for them to end.
    preempt_at: Option<std::time::Instant>,
}

/// What a pending change does once its units are let go.
enum Pending {
    /// The workers give back the segments from this one on; the shrink
    /// answers with the plan of each device it made room on.
    Shrink { from: u32, plan: Vec<String> },
    /// The workers sleep, once no request runs.
    Sleep,
    /// The workers rewrite the weights from the checkpoint at this path,
    /// once no request runs.
    Weights(String),
}

/// Every worker's `uniserve.cache_memory` report: of the workers that hold
/// the cache, and of the others (`Scheduler::memory_reports`).
struct MemoryReports {
    cache: Vec<CacheMemory>,
    beside: Vec<CacheMemory>,
}

/// The serving world's preparation for a world that succeeds it
/// (`Scheduler::plan_preparation`).
pub(super) struct Preparation {
    /// The first segment every device gives back and its first unit;
    /// `None` when each keeps all it holds.
    tail: Option<(u32, u32)>,
    /// Each device's plan, as JSON (`super::ledger::Preparing`).
    plans: Vec<String>,
}

impl Pending {
    /// Whether the change waits until no request runs, admitting none.
    fn quiesces(&self) -> bool {
        matches!(self, Self::Sleep | Self::Weights(_))
    }
}

impl Scheduler {
    /// Applies `change` to the KV cache's device memory, and answers with
    /// what each worker reported once it is done. A shrink or sleep whose
    /// withdrawn units requests still hold answers once they let them go
    /// (`Scheduler::poll_resize`).
    pub(super) fn resize_cache(
        &mut self,
        change: CacheChange,
        answer: crossbeam_channel::Sender<Result<Vec<String>, String>>,
    ) {
        if self.resizing.is_some() {
            let _ = answer.send(Err("another KV cache change is in progress".to_owned()));
            return;
        }
        let started = match change.clone() {
            CacheChange::Shrink { successor, .. } => {
                self.start_shrink(&successor).map(|(from, plan)| {
                    from.map(|from| Pending::Shrink {
                        from,
                        plan: plan.clone(),
                    })
                    .ok_or(plan)
                })
            }
            CacheChange::MakeRoom { bytes, devices } => {
                self.start_room(bytes, &devices).map(|(from, plan)| {
                    from.map(|from| Pending::Shrink {
                        from,
                        plan: plan.clone(),
                    })
                    .ok_or(plan)
                })
            }
            CacheChange::Plan(successor) => {
                let _ = answer.send(
                    self.plan_preparation(&successor)
                        .map(|preparation| preparation.plans),
                );
                return;
            }
            CacheChange::Sleep => Ok(self.start_sleep().ok_or_else(Vec::new)),
            CacheChange::Evict => self.start_evict(),
            CacheChange::Restore => {
                let _ = answer.send(self.restore_cache(None));
                return;
            }
            CacheChange::Grow => {
                let _ = answer.send(self.grow_cache());
                return;
            }
            CacheChange::Wake => {
                let _ = answer.send(self.wake());
                return;
            }
        };
        match started {
            Ok(Ok(pending)) => {
                let preempt_at = match change {
                    CacheChange::Shrink {
                        preempt_after: Some(after),
                        ..
                    } => Some(std::time::Instant::now() + after),
                    _ => None,
                };
                self.resizing = Some(PendingResize {
                    change: pending,
                    answer,
                    preempt_at,
                });
                self.poll_resize();
            }
            // Nothing to give back: the change is done, with its plan.
            Ok(Err(plan)) => {
                let _ = answer.send(Ok(plan));
            }
            Err(error) => {
                let _ = answer.send(Err(error));
            }
        }
    }

    /// Rewrites the weights in place from the checkpoint at `model_path`,
    /// and answers with what each worker reported. The requests in flight
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
        answer: crossbeam_channel::Sender<Result<Vec<String>, String>>,
    ) {
        if self.resizing.is_some() {
            let _ = answer.send(Err("a KV cache change is in progress".to_owned()));
            return;
        }
        if self.asleep {
            let _ = answer.send(self.apply_weights(&model_path));
            return;
        }
        match in_flight {
            crate::WeightsInFlight::Keep => {
                let _ = answer.send(self.apply_weights_keeping(&model_path));
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
            answer,
            preempt_at: None,
        });
        self.poll_resize();
    }

    /// Whether admission waits for a sleep or a weight update: the
    /// scheduler sleeps, or waits for its requests to finish.
    pub(super) fn admission_sleeps(&self) -> bool {
        self.asleep
            || self.parks()
            || self
                .resizing
                .as_ref()
                .is_some_and(|resize| resize.change.quiesces())
    }

    /// Completes a pending shrink or sleep once no request holds a
    /// withdrawn unit, and a sleep once no request runs either: the workers
    /// give the memory back between batches.
    pub(super) fn poll_resize(&mut self) {
        // A shrink past its bound recomputes the requests still holding the
        // tail, once.
        if let Some(resize) = self.resizing.as_mut()
            && resize
                .preempt_at
                .is_some_and(|at| std::time::Instant::now() >= at)
        {
            resize.preempt_at = None;
            let withdrawn = self
                .storage
                .cache()
                .map_or(u32::MAX, |cache| cache.block_pool.withdrawn());
            tracing::info!(
                first = withdrawn,
                "the shrink's bound passed: the requests holding the tail are recomputed"
            );
            self.migrate_holders(withdrawn);
        }
        let Some(resize) = self.resizing.as_ref() else {
            return;
        };
        let held = self
            .storage
            .cache()
            .map_or(0, |cache| cache.block_pool.withdrawn_held());
        if held > 0
            || (resize.change.quiesces()
                && !(self.running.is_empty()
                    && self.running_media.is_empty()
                    && self.retiring_requests.is_empty()))
        {
            return;
        }
        let Some(resize) = self.resizing.take() else {
            return;
        };
        // A change that fails part of the way leaves the units withdrawn and
        // a sleeping scheduler asleep: a restore or a wake creates whatever
        // the workers gave back.
        let result = match resize.change {
            Pending::Sleep => {
                self.asleep = true;
                self.run_hook("uniserve.sleep", "").inspect(|_| {
                    tracing::info!(
                        "the scheduler sleeps: its KV cache and graph memory are given back"
                    );
                })
            }
            Pending::Shrink { from, plan } => {
                let payload = serde_json::json!({ "from": from }).to_string();
                self.run_hook("uniserve.release_cache", &payload).map(|_| {
                    tracing::info!(
                        from,
                        units = self.storage.usable_units(),
                        "gave back the tail of the KV cache"
                    );
                    plan
                })
            }
            Pending::Weights(path) => self.apply_weights(&path),
        };
        let _ = resize.answer.send(result);
    }

    /// Rewrites the weights from the checkpoint at `path` while no request
    /// runs. The workers sleep for it unless they sleep already, so that
    /// the cache's memory holds the new checkpoint as it loads, and wake
    /// after, whether or not the update succeeded. Every cached prefix page
    /// was computed with the old weights: the prefix cache is flushed, and
    /// the host KV cache keys pages by the new weights' seeds.
    fn apply_weights(&mut self, path: &str) -> Result<Vec<String>, String> {
        let slept = !self.asleep;
        if slept {
            self.asleep = true;
            self.run_hook("uniserve.sleep", "")?;
        }
        let payload = serde_json::json!({ "model_path": path }).to_string();
        let updated = self.run_hook("uniserve.update_weights", &payload);
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
        Ok(results)
    }

    /// Rewrites the weights from the checkpoint at `path` while the requests
    /// in flight hold their KV cache (`WeightsInFlight::Keep`): the workers
    /// load the checkpoint beside the cache once the batches in flight
    /// complete. Every request in flight then publishes no page, and every
    /// cached prefix page is revoked, as the new weights key pages anew.
    fn apply_weights_keeping(&mut self, path: &str) -> Result<Vec<String>, String> {
        let payload = serde_json::json!({ "model_path": path, "keep": true }).to_string();
        let results = self.run_hook("uniserve.update_weights", &payload)?;
        let mut kept = 0;
        for state in self.running.values_mut() {
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
        Ok(results)
    }

    /// Ends every request the scheduler holds, aborted: those queued, and
    /// those running once their batches in flight drain.
    fn abort_held_requests(&mut self) {
        let mut held: Vec<RequestId> = self.running.keys().copied().collect();
        held.extend(self.waiting.iter().map(|state| state.req.request_id));
        held.extend(self.media_ids());
        held.extend(
            self.waiting_media
                .iter()
                .map(|submission| submission.request.request_id),
        );
        tracing::info!(
            requests = held.len(),
            "aborting the requests in flight for a weight update"
        );
        for id in held {
            self.mark_cancelled(id, true);
        }
    }

    /// Roots the prefix hash chains at the identity the weight update
    /// `results` report (one per worker), and gives the host KV cache each
    /// worker group's page key seeds they report, which the ranks of a group
    /// agree on.
    fn reseed_pages(&mut self, results: &[String]) -> Result<(), String> {
        let identity = results.iter().find_map(|result| {
            serde_json::from_str::<serde_json::Value>(result)
                .ok()?
                .get("identity")?
                .as_str()
                .map(str::to_owned)
        });
        if let (Some(identity), Some(cache)) = (identity, self.storage.cache.as_mut()) {
            cache
                .coordinator
                .set_seed(crate::kv_cache::prefix_seed(&identity));
        }
        // The trajectories the new weights compute are keyed apart from the
        // old ones.
        if let Some(store) = self.trajectories.as_mut() {
            let identities: Vec<String> = results
                .iter()
                .filter_map(|result| {
                    let value: serde_json::Value = serde_json::from_str(result).ok()?;
                    let group = value.get("group")?.as_str()?;
                    store
                        .serves(&crate::WorkerGroupId(group.to_owned()))
                        .then(|| value.get("identity")?.as_str().map(str::to_owned))
                        .flatten()
                })
                .collect();
            if !identities.is_empty() {
                store.reseed(crate::trajectories::latent_seed(
                    identities.iter().map(String::as_str),
                ));
            }
        }
        let Some(kv_offload) = self.kv_offload.as_mut() else {
            return Ok(());
        };
        let mut seeds: BTreeMap<String, Vec<[u8; 32]>> = BTreeMap::new();
        for result in results {
            let value: serde_json::Value = serde_json::from_str(result)
                .map_err(|error| format!("a worker's weight update report: {error}"))?;
            let (Some(group), Some(contents)) =
                (value["group"].as_str(), value["page_contents"].as_array())
            else {
                continue;
            };
            if contents.is_empty() {
                continue;
            }
            let contents = contents
                .iter()
                .map(|content| {
                    content
                        .as_str()
                        .and_then(seed)
                        .ok_or_else(|| "a page key seed is not 32 hexadecimal bytes".to_owned())
                })
                .collect::<Result<Vec<_>, _>>()?;
            match seeds.get(group) {
                Some(agreed) if *agreed != contents => {
                    return Err(format!(
                        "the ranks of worker group {group} report other page key seeds"
                    ));
                }
                Some(_) => {}
                None => {
                    seeds.insert(group.to_owned(), contents);
                }
            }
        }
        for (group, contents) in seeds {
            kv_offload.reseed(&WorkerGroupId(group), &contents)?;
        }
        Ok(())
    }

    /// The segments each worker that holds the cache reports, which agree
    /// on their segments and the ones they hold, with its device's ledger;
    /// `None` when the cache is not split into segments. A worker that
    /// holds no cache, such as an expert worker, reports none and is left
    /// out.
    pub(super) fn cache_memory(&mut self) -> Result<Option<Vec<CacheMemory>>, String> {
        Ok(self.memory_reports()?.map(|reports| reports.cache))
    }

    /// Every worker's memory report: those of the workers that hold the
    /// cache, which agree on their segments and the ones they hold, and
    /// those of the others (an expert worker's, with its GPU's ledger when
    /// it keeps memory at the Storage Server). `None` when the cache is not
    /// split into segments.
    fn memory_reports(&mut self) -> Result<Option<MemoryReports>, String> {
        if self.storage.cache().is_none() {
            return Ok(None);
        }
        let (cache, beside): (Vec<CacheMemory>, Vec<CacheMemory>) = self
            .run_hook("uniserve.cache_memory", "")?
            .iter()
            .map(|report| serde_json::from_str::<CacheMemory>(report))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("a worker's cache memory report: {error}"))?
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
        Ok(Some(MemoryReports { cache, beside }))
    }

    /// The serving world's preparation for a world that succeeds it on the
    /// same devices (`super::ledger::preparing`): for each worker that holds
    /// the cache, its device's KV cache limit from the GPU's ledger, beside
    /// everything else the deployment and the Transfer Server hold there
    /// and what the successor adds, and the segments the cache keeps within
    /// it. A cache of a configured token capacity also keeps no more than
    /// leaves the successor's head room within that capacity beside the
    /// segments it inherits, so that a world change never grows the cache
    /// past it. Returns the first segment every device gives back with its
    /// first unit (`None` when each keeps all it holds), and each device's
    /// plan as JSON. The first segment, which holds the padding and warmup
    /// units, stays, as do the segments the world inherited and does not
    /// own yet.
    ///
    /// The devices of the workers that hold no cache (an expert worker's)
    /// are planned too: their successor's graph pool and fixed part must
    /// fit beside everything the ledger books there.
    ///
    /// Fails, changing nothing, when a device's limit cannot hold the
    /// segments the cache keeps in any case: the error names every item of
    /// every device and host.
    pub(super) fn plan_preparation(
        &mut self,
        successor: &crate::Successor,
    ) -> Result<Preparation, String> {
        let (in_flight_units, prefixes) = self.migrating_cache();
        let in_flight_tokens: Vec<u32> = prefixes.iter().map(|prefix| prefix.tokens).collect();
        let Some(MemoryReports {
            cache: reports,
            beside,
        }) = self.memory_reports()?
        else {
            return Ok(Preparation {
                tail: None,
                plans: Vec::new(),
            });
        };
        let first = &reports[0];
        let held = first.held;
        let floor = self.kept_floor(first);
        let mut keep = held;
        let mut plans: Vec<Preparing> = Vec::new();
        for report in &reports {
            let mut plan = preparing(report, floor, successor, in_flight_units, &in_flight_tokens)?;
            let mut kept = plan.segments_kept;
            // A configured capacity holds the successor's cache: the
            // segments it inherits from this one and its head. A successor
            // that maps none of this cache inherits none, and its own
            // capacity sizes the cache it lays out.
            if let Some(capacity) = report
                .capacity_units
                .filter(|_| !successor.recomputes_cache)
            {
                let head_units = report.head_bytes.div_ceil(report.unit_bytes.max(1));
                while kept > floor && u64::from(report.first_unit(kept)) + head_units > capacity {
                    kept -= 1;
                }
                if u64::from(report.first_unit(kept)) + head_units > capacity {
                    return Err(format!(
                        "the KV cache's configured capacity of {capacity} units cannot hold a \
                         successor world's {head_units}-unit head segment beside the cache's \
                         first segment"
                    ));
                }
            }
            plan.segments_kept = kept;
            plan.kv_kept = (0..kept as usize)
                .map(|index| report.segment_bytes(index))
                .sum();
            plan.idle_cached_tail = self.storage.cache().map_or(0, |cache| {
                u64::from(cache.block_pool.cached_from(report.first_unit(kept))) * report.unit_bytes
            });
            keep = keep.min(kept);
            plans.push(plan);
        }
        // A worker without a cache that keeps its memory at the Storage
        // Server: its successor adds beside it what it holds, and the limit
        // left (of no cache) must not be negative.
        for report in beside.iter().filter(|report| report.ledger.is_some()) {
            plans.push(preparing(report, 0, successor, 0, &in_flight_tokens)?);
        }
        if plans.iter().any(|plan| !plan.feasible) {
            return Err(super::ledger::refusal(&plans));
        }
        let plans = plans
            .iter()
            .map(|plan| serde_json::to_string(plan).unwrap_or_default())
            .collect();
        Ok(Preparation {
            tail: (keep < held).then(|| (keep, first.first_unit(keep))),
            plans,
        })
    }

    /// The segments of a cache split as `first` reports that it cannot give
    /// back: the first, which holds the padding and warmup units, and every
    /// segment it inherited and does not own yet.
    fn kept_floor(&mut self, first: &CacheMemory) -> u32 {
        let inherited = self
            .storage
            .cache()
            .map_or(0..0, |cache| cache.block_pool.inherited());
        (1..first.held)
            .rev()
            .find(|&index| first.first_unit(index) < inherited.end)
            .map_or(1, |index| index + 1)
    }

    /// Withdraws the tail past what leaves every device's ledger room for
    /// the requested device storage beside everything it
    /// books ([`room`]); the requests holding it run on. Returns the first
    /// segment to give back, `None` when every device has the room already
    /// (or the cache is not split into segments, which gives back nothing),
    /// and each device's plan.
    ///
    /// Fails, withdrawing nothing, when even the segments the cache cannot
    /// give back leave a device too little room: the error names every
    /// device's bytes and ledger entries.
    fn start_room(
        &mut self,
        bytes: u64,
        devices: &std::collections::BTreeMap<storage::manifest::Gpu, u64>,
    ) -> Result<(Option<u32>, Vec<String>), String> {
        let Some(MemoryReports {
            cache: reports,
            beside,
        }) = self.memory_reports()?
        else {
            return Ok((None, Vec::new()));
        };
        let held = reports.first().map_or(0, |first| first.held);
        let floor = reports.first().map_or(0, |first| self.kept_floor(first));
        let requested = |report: &CacheMemory| {
            report
                .ledger
                .as_ref()
                .and_then(|ledger| devices.get(&ledger.gpu))
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
        let plans = rooms
            .iter()
            .map(|room| serde_json::to_string(room).unwrap_or_default())
            .collect();
        if keep >= held {
            return Ok((None, plans));
        }
        let unit = reports[0].first_unit(keep);
        self.offload_idle_from(unit);
        self.withdraw_from(unit, false);
        Ok((Some(keep), plans))
    }

    /// Withdraws the tail past what each device's limit keeps
    /// ([`Self::plan_preparation`]); the requests holding it run on.
    /// Returns the first segment to give back, `None` when every device
    /// keeps all its cache, and each device's plan.
    fn start_shrink(
        &mut self,
        successor: &crate::Successor,
    ) -> Result<(Option<u32>, Vec<String>), String> {
        let Preparation { tail, plans } = self.plan_preparation(successor)?;
        let Some((segment, unit)) = tail else {
            return Ok((None, plans));
        };
        // The idle cached pages of the tail reach the host KV cache before
        // the tail is given back (step 1 of `specs/online-operations.md`
        // §6.5): a later prompt reads them back from there.
        self.offload_idle_from(unit);
        self.withdraw_from(unit, false);
        Ok((Some(segment), plans))
    }

    /// Withdraws every segment of a parked world's cache past its first
    /// (and past those it inherited and does not own yet), once the idle
    /// cached pages they hold reached the host KV cache (step 5 of
    /// `specs/online-operations.md` §6.5): its successor maps none of them
    /// and grows into their memory, and a rollback that resumes this world
    /// creates them again and reads the pages back from the host KV cache.
    /// No request runs on a parked world. Answers, once the workers gave the
    /// segments back, with what it evicted.
    fn start_evict(&mut self) -> Result<Result<Pending, Vec<String>>, String> {
        if !self.parks() {
            return Err("only a parked world evicts its KV cache".to_owned());
        }
        let Some(reports) = self.cache_memory()? else {
            return Ok(Err(Vec::new()));
        };
        let first = &reports[0];
        let inherited = self
            .storage
            .cache()
            .map_or(0..0, |cache| cache.block_pool.inherited());
        let floor = (1..first.held)
            .rev()
            .find(|&index| first.first_unit(index) < inherited.end)
            .map_or(1, |index| index + 1);
        if floor >= first.held {
            return Ok(Err(Vec::new()));
        }
        let unit = first.first_unit(floor);
        let bytes: u64 = (floor as usize..first.held as usize)
            .map(|index| first.segment_bytes(index))
            .sum();
        let pages = self.offload_idle_from(unit);
        self.withdraw_from(unit, false);
        let evicted = serde_json::json!({
            "evicted_segments": first.held - floor,
            "evicted_bytes": bytes,
            "offloaded_pages": pages,
        });
        tracing::info!(%evicted, "a parked world evicts its KV cache past its first segment");
        Ok(Ok(Pending::Shrink {
            from: floor,
            plan: vec![evicted.to_string()],
        }))
    }

    /// Holds every idle cached page a withdrawal from `unit` evicts and has
    /// the host KV cache verify or make its copy under the key it was
    /// offloaded with: a page the pool holds already is released at once,
    /// another once its copy commits. Pages of a sliding-window group, and
    /// pages never offloaded (no host KV cache serves their worker), are
    /// evicted without a copy: a later prompt computes them again. Returns
    /// how many pages wait for their copies.
    fn offload_idle_from(&mut self, unit: u32) -> usize {
        let (Some(cache), Some(kv_offload)) = (self.storage.cache(), self.kv_offload.as_mut())
        else {
            return 0;
        };
        let idle = cache.block_pool.acquire_idle_from(unit);
        let workers: Vec<(WorkerGroupId, ipc::WorkerEndpoint)> = self
            .executor
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
    /// requests holding them run to their end.
    fn withdraw_from(&mut self, first: u32, recompute: bool) {
        let Some(cache) = self.storage.cache() else {
            return;
        };
        let held = cache.block_pool.withdraw(first);
        let withdrawn = cache.block_pool.withdrawn();
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

    /// Ends as migratable every running request that resumes from its
    /// generated tokens and holds a unit from `first` on.
    fn migrate_holders(&mut self, first: u32) {
        let holders: Vec<RequestId> = self
            .running
            .iter()
            .filter(|(_, state)| {
                state.allocations.as_ref().is_some_and(|allocations| {
                    allocations
                        .block_tables()
                        .iter()
                        .any(|table| table.unit_ids().iter().any(|unit| unit.0 >= first))
                })
            })
            .map(|(id, _)| *id)
            .collect();
        self.migrate_running(MigrationCause::Drain, |id| holders.contains(&id));
    }

    /// Creates the given-back segments again, the first `limit` of them or
    /// all, and returns their units to allocation.
    fn restore_cache(&mut self, limit: Option<u32>) -> Result<Vec<String>, String> {
        if self.asleep {
            return Err("the scheduler sleeps: wake it instead".to_owned());
        }
        let Some(cache) = self.storage.cache() else {
            return Ok(Vec::new());
        };
        if !cache.block_pool.has_withdrawn() {
            return Ok(Vec::new());
        }
        let payload = match limit {
            Some(segments) => serde_json::json!({ "segments": segments }).to_string(),
            None => String::new(),
        };
        let results = self.run_hook("uniserve.restore_cache", &payload)?;
        let end = match limit {
            Some(_) => self
                .cache_memory()?
                .map(|reports| reports[0].first_unit(reports[0].held)),
            None => None,
        };
        if let Some(cache) = self.storage.cache() {
            match end {
                Some(end) => cache.block_pool.restore_to(end),
                None => cache.block_pool.restore(),
            }
        }
        tracing::info!(units = self.storage.usable_units(), "restored the KV cache");
        Ok(results)
    }

    /// Creates as many of the given-back segments as every device's ledger
    /// has room for within its quota, beside everything the deployment's
    /// worlds and the Transfer Server hold there (and its free memory
    /// holds, beside processes outside the deployment), and returns their
    /// units to allocation: what a previous world released is the
    /// successor's on the ledger at once.
    fn grow_cache(&mut self) -> Result<Vec<String>, String> {
        let Some(reports) = self.cache_memory()? else {
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
        tracing::info!(segments = fit, "growing the KV cache into the memory freed");
        self.restore_cache(Some(fit))
    }

    /// Restores the graph memory and the cache a sleep gave back, and admits
    /// again.
    fn wake(&mut self) -> Result<Vec<String>, String> {
        if !self.asleep {
            return Ok(Vec::new());
        }
        let results = self.run_hook("uniserve.wake", "")?;
        if let Some(cache) = self.storage.cache() {
            cache.block_pool.restore();
        }
        self.asleep = false;
        tracing::info!("the scheduler woke");
        Ok(results)
    }
}

/// A page key seed from its hexadecimal text.
fn seed(text: &str) -> Option<[u8; 32]> {
    if text.len() != 64 {
        return None;
    }
    let mut seed = [0; 32];
    for (index, byte) in seed.iter_mut().enumerate() {
        *byte = u8::from_str_radix(text.get(2 * index..2 * index + 2)?, 16).ok()?;
    }
    Some(seed)
}
