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
//! restores both. An in-place weight update recomputes the requests as a
//! sleep does and rewrites the weights while the workers sleep, so that the
//! cache's memory holds the new checkpoint as it loads.

use super::*;
use crate::executor::WorkerGroupId;
use crate::handle::CacheChange;

/// What `uniserve.cache_memory` reports of one worker.
#[derive(serde::Deserialize)]
struct CacheMemory {
    /// The segments of each cache plane, in unit order; empty when the
    /// planes are not split.
    #[serde(default)]
    segments: Vec<ipc::CacheSegment>,
    /// Segments the worker holds: the first given back.
    #[serde(default)]
    held: u32,
    /// Device bytes of one unit across the worker's planes.
    #[serde(default)]
    unit_bytes: u64,
    /// Device bytes of the head segment a successor creates beside this
    /// cache's segments.
    #[serde(default)]
    head_bytes: u64,
    /// The units a configured token capacity holds; a successor's head and
    /// the segments it inherits fit in them together. `None` for a cache
    /// sized by memory.
    #[serde(default)]
    capacity_units: Option<u64>,
    /// Free bytes of the worker's device.
    free_bytes: u64,
    /// Device bytes the worker process holds beside the shared weights and
    /// cache, which a successor world needs as much of.
    process_bytes: u64,
    /// Device bytes the share of the device a worker's capacity plan keeps
    /// for graphs and activations, which a successor captures in.
    #[serde(default)]
    reserve_bytes: u64,
    /// Device bytes of the exclusive storages the worker's world keeps
    /// beside its value planes, which a successor creates as many of.
    #[serde(default)]
    exclusive_bytes: u64,
}

impl CacheMemory {
    /// The first unit of segment `index`.
    fn first_unit(&self, index: u32) -> u32 {
        self.segments
            .iter()
            .take(index as usize)
            .map(|segment| segment.units)
            .sum()
    }
}

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
    /// The workers give back the segments from this one on.
    Shrink(u32),
    /// The workers sleep, once no request runs.
    Sleep,
    /// The workers rewrite the weights from the checkpoint at this path,
    /// once no request runs.
    Weights(String),
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
        let started = match change {
            CacheChange::Shrink { .. } => self.start_shrink(),
            CacheChange::Sleep => self.start_sleep(),
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
            Ok(Some(from)) => {
                let preempt_at = match change {
                    CacheChange::Shrink {
                        preempt_after: Some(after),
                    } => Some(std::time::Instant::now() + after),
                    _ => None,
                };
                self.resizing = Some(PendingResize {
                    change: if change == CacheChange::Sleep {
                        Pending::Sleep
                    } else {
                        Pending::Shrink(from)
                    },
                    answer,
                    preempt_at,
                });
                self.poll_resize();
            }
            Ok(None) => {
                let _ = answer.send(Ok(Vec::new()));
            }
            Err(error) => {
                let _ = answer.send(Err(error));
            }
        }
    }

    /// Rewrites the weights in place from the checkpoint at `model_path`
    /// once no request runs, and answers with what each worker reported:
    /// the requests running are recomputed and the others wait, as for a
    /// sleep.
    pub(super) fn update_weights(
        &mut self,
        model_path: String,
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
        self.withdraw_from(0, true);
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
            Pending::Shrink(from) => {
                let payload = serde_json::json!({ "from": from }).to_string();
                self.run_hook("uniserve.release_cache", &payload)
                    .inspect(|_| {
                        tracing::info!(
                            from,
                            units = self.storage.usable_units(),
                            "gave back the tail of the KV cache"
                        );
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
    /// on their segments and the ones they hold; `None` when the cache is
    /// not split into segments. A worker that holds no cache, such as an
    /// expert worker, reports none and is left out.
    fn cache_memory(&mut self) -> Result<Option<Vec<CacheMemory>>, String> {
        if self.storage.cache().is_none() {
            return Ok(None);
        }
        let reports = self
            .run_hook("uniserve.cache_memory", "")?
            .iter()
            .map(|report| serde_json::from_str::<CacheMemory>(report))
            .filter(|report| {
                report
                    .as_ref()
                    .map_or(true, |report| !report.segments.is_empty())
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("a worker's cache memory report: {error}"))?;
        let Some(first) = reports.first() else {
            return Ok(None);
        };
        if reports
            .iter()
            .any(|report| (&report.segments, report.held) != (&first.segments, first.held))
        {
            return Err("the workers split the KV cache into different segments".to_owned());
        }
        Ok(Some(reports))
    }

    /// Withdraws the tail that leaves each device the memory a world
    /// succeeding this one needs beside what they share: as much as the
    /// worker holds beside the weights and the cache, the graph share, the
    /// exclusive storages beside the cache planes and a head segment. A
    /// cache of a configured token capacity also withdraws what leaves the
    /// successor's head room within that capacity beside the segments it
    /// inherits, so that a world change never grows the cache past it. The
    /// requests holding the tail run on. Returns the first segment to give
    /// back, or `None` when every device has the room already. The first
    /// segment, which holds the padding and warmup units, stays, as do the
    /// segments the world inherited and does not own yet.
    fn start_shrink(&mut self) -> Result<Option<u32>, String> {
        let Some(reports) = self.cache_memory()? else {
            return Ok(None);
        };
        let first = &reports[0];
        let held = first.held;
        // The last segment that stays: the first, and every inherited one.
        let inherited = self
            .storage
            .cache()
            .map_or(0..0, |cache| cache.block_pool.inherited());
        let floor = (1..held)
            .rev()
            .find(|&index| first.first_unit(index) < inherited.end)
            .map_or(1, |index| index + 1);
        let mut keep = held;
        for report in &reports {
            let needed = report.process_bytes
                + report.reserve_bytes
                + report.exclusive_bytes
                + report.head_bytes;
            let mut short = needed.saturating_sub(report.free_bytes);
            let head_units = report.head_bytes.div_ceil(report.unit_bytes.max(1));
            let mut over = report.capacity_units.map_or(0, |capacity| {
                (u64::from(report.first_unit(held)) + head_units).saturating_sub(capacity)
            });
            let mut released = held;
            while (short > 0 || over > 0) && released > floor {
                released -= 1;
                let units = u64::from(report.segments[released as usize].units);
                short = short.saturating_sub(units * report.unit_bytes);
                over = over.saturating_sub(units);
            }
            if over > 0 {
                return Err(format!(
                    "the KV cache's configured capacity of {} units cannot hold a successor \
                     world's {head_units}-unit head segment beside the cache's first segment",
                    report.capacity_units.unwrap_or_default(),
                ));
            }
            if short > 0 {
                return Err(format!(
                    "the KV cache cannot make room for a successor world: it needs {needed} \
                     bytes beside what the worlds share ({} the worker holds beside the weights \
                     and the cache, {} of graph share, {} of exclusive storages and a {}-byte \
                     head segment), {} bytes are free, and the cache's segments past its first \
                     hold {} bytes",
                    report.process_bytes,
                    report.reserve_bytes,
                    report.exclusive_bytes,
                    report.head_bytes,
                    report.free_bytes,
                    u64::from(report.first_unit(held) - report.first_unit(floor))
                        * report.unit_bytes,
                ));
            }
            keep = keep.min(released);
        }
        if keep == held {
            return Ok(None);
        }
        self.withdraw_from(first.first_unit(keep), false);
        Ok(Some(keep))
    }

    /// Withdraws every unit and lets the requests that run finish or
    /// recompute; admission waits until a wake.
    fn start_sleep(&mut self) -> Result<Option<u32>, String> {
        if self.asleep {
            return Ok(None);
        }
        self.withdraw_from(0, true);
        Ok(Some(0))
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

    /// Creates as many of the given-back segments as every device's free
    /// memory holds, beside the graph share each keeps, and returns their
    /// units to allocation.
    fn grow_cache(&mut self) -> Result<Vec<String>, String> {
        let Some(reports) = self.cache_memory()? else {
            return Ok(Vec::new());
        };
        let held = reports[0].held;
        let count = reports[0].segments.len() as u32;
        let mut fit = count - held;
        for report in &reports {
            let mut free = report.free_bytes;
            let mut segments = 0;
            for segment in &report.segments[held as usize..] {
                let bytes = u64::from(segment.units) * report.unit_bytes;
                if bytes > free {
                    break;
                }
                free -= bytes;
                segments += 1;
            }
            fit = fit.min(segments);
        }
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
