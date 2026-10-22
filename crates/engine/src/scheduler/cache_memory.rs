//! Giving back and restoring the KV cache's device memory.
//!
//! A worker splits its cache planes into segments of whole units
//! (`uniserve.cache_memory` reports them), and gives back the tail from a
//! segment on. A shrink makes room for a standby engine on the same devices:
//! the scheduler withdraws the tail's units from allocation, recomputes the
//! requests that hold them (it ends them as migratable, and the daemon
//! submits them again with their generated tokens, as vLLM's recompute
//! preemption does), and once no request holds one, the workers give the
//! tail back. A restore creates it again. A sleep gives back every segment
//! and the CUDA graph memory once no request runs, and admits nothing until
//! a wake restores both. An in-place weight update recomputes the requests
//! as a sleep does and rewrites the weights while the workers sleep, so
//! that the cache's memory holds the new checkpoint as it loads.

use super::*;
use crate::executor::WorkerGroupId;
use crate::handle::CacheChange;

/// What `uniserve.cache_memory` reports of one worker.
#[derive(serde::Deserialize)]
struct CacheMemory {
    /// Segments of each cache plane; zero when the planes are not split.
    segments: u32,
    /// Units of each segment.
    segment_units: u32,
    /// Device bytes of one segment across the worker's planes.
    segment_bytes: u64,
    /// Segments the worker holds: the first given back.
    released_from: u32,
    /// Free bytes of the worker's device.
    free_bytes: u64,
    /// Device bytes the worker process holds beside the shared weights and
    /// cache, which a standby engine needs as much of.
    process_bytes: u64,
    /// Device bytes a parked standby leaves free beside the workers it
    /// stands by for: the share a serving worker's capacity plan keeps for
    /// graphs and activations, which the standby's promotion maps.
    #[serde(default)]
    reserve_bytes: u64,
}

/// A shrink, sleep or weight update that waits for the requests holding
/// the withdrawn units to let them go (`Scheduler::resize_cache`,
/// `Scheduler::update_weights`).
pub(super) struct PendingResize {
    change: Pending,
    answer: crossbeam_channel::Sender<Result<Vec<String>, String>>,
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
            CacheChange::Shrink => self.start_shrink(),
            CacheChange::Sleep => self.start_sleep(),
            CacheChange::Restore => {
                let _ = answer.send(self.restore_cache());
                return;
            }
            CacheChange::Wake => {
                let _ = answer.send(self.wake());
                return;
            }
        };
        match started {
            Ok(Some(from)) => {
                self.resizing = Some(PendingResize {
                    change: if change == CacheChange::Sleep {
                        Pending::Sleep
                    } else {
                        Pending::Shrink(from)
                    },
                    answer,
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
        self.withdraw_from(0);
        self.resizing = Some(PendingResize {
            change: Pending::Weights(model_path),
            answer,
        });
        self.poll_resize();
    }

    /// Whether admission waits for a sleep or a weight update: the
    /// scheduler sleeps, or waits for its requests to finish.
    pub(super) fn admission_sleeps(&self) -> bool {
        self.asleep
            || self
                .resizing
                .as_ref()
                .is_some_and(|resize| resize.change.quiesces())
    }

    /// Completes a pending shrink or sleep once no request holds a
    /// withdrawn unit, and a sleep once no request runs either: the workers
    /// give the memory back between batches.
    pub(super) fn poll_resize(&mut self) {
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

    /// Gives the host KV cache each worker group's page key seeds that the
    /// weight update `results` report (one per worker), which the ranks of a
    /// group agree on.
    fn reseed_pages(&mut self, results: &[String]) -> Result<(), String> {
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

    /// The segments each worker reports, which agree on their count and
    /// units; `None` when the cache is not split into segments.
    fn cache_memory(&mut self) -> Result<Option<Vec<CacheMemory>>, String> {
        if self.storage.cache().is_none() {
            return Ok(None);
        }
        let reports = self
            .run_hook("uniserve.cache_memory", "")?
            .iter()
            .map(|report| serde_json::from_str::<CacheMemory>(report))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("a worker's cache memory report: {error}"))?;
        let Some(first) = reports.first() else {
            return Ok(None);
        };
        if reports.iter().any(|report| report.segments == 0) {
            return Ok(None);
        }
        if reports.iter().any(|report| {
            (report.segments, report.segment_units) != (first.segments, first.segment_units)
        }) {
            return Err("the workers split the KV cache into different segments".to_owned());
        }
        Ok(Some(reports))
    }

    /// Withdraws the tail that leaves each device as much free memory as its
    /// worker holds beside the weights and the cache, and recomputes the
    /// requests that hold it. Returns the first segment to give back, or
    /// `None` when every device has the room already. The first segment,
    /// which holds the padding and warmup units, stays.
    fn start_shrink(&mut self) -> Result<Option<u32>, String> {
        let Some(reports) = self.cache_memory()? else {
            return Ok(None);
        };
        let units = reports[0].segment_units;
        let held = reports
            .iter()
            .map(|report| report.released_from)
            .min()
            .unwrap_or(0);
        let mut keep = held;
        for report in &reports {
            let short =
                (report.process_bytes + report.reserve_bytes).saturating_sub(report.free_bytes);
            let release = short.div_ceil(report.segment_bytes.max(1));
            let room = u64::from(report.released_from.saturating_sub(1));
            if release > room {
                return Err(format!(
                    "the KV cache cannot make room for a standby engine: a worker holds {} bytes \
                     beside the weights and the cache, a standby keeps {} bytes free beside it, \
                     {} bytes are free, and the cache beyond its first segment holds {} bytes",
                    report.process_bytes,
                    report.reserve_bytes,
                    report.free_bytes,
                    room * report.segment_bytes,
                ));
            }
            keep = keep.min(report.released_from - release as u32);
        }
        if keep == held {
            return Ok(None);
        }
        self.withdraw_from(keep * units);
        Ok(Some(keep))
    }

    /// Withdraws every unit and lets the requests that run finish or
    /// recompute; admission waits until a wake.
    fn start_sleep(&mut self) -> Result<Option<u32>, String> {
        if self.asleep {
            return Ok(None);
        }
        self.withdraw_from(0);
        Ok(Some(0))
    }

    /// Withdraws the units from `first` on, and ends as migratable every
    /// running request that resumes from its generated tokens and holds one,
    /// which the daemon submits again: it recomputes on the units left.
    fn withdraw_from(&mut self, first: u32) {
        let Some(cache) = self.storage.cache() else {
            return;
        };
        let held = cache.block_pool.withdraw(first);
        let withdrawn = cache.block_pool.withdrawn();
        let holders: Vec<RequestId> = self
            .running
            .iter()
            .filter(|(_, state)| {
                state.allocations.as_ref().is_some_and(|allocations| {
                    allocations
                        .block_tables()
                        .iter()
                        .any(|table| table.unit_ids().iter().any(|unit| unit.0 >= withdrawn))
                })
            })
            .map(|(id, _)| *id)
            .collect();
        tracing::info!(
            first = withdrawn,
            held,
            requests = holders.len(),
            "withdrew the tail of the KV cache from allocation"
        );
        self.migrate_running(MigrationCause::Drain, |id| holders.contains(&id));
    }

    /// Creates the given-back segments again and returns their units to
    /// allocation.
    fn restore_cache(&mut self) -> Result<Vec<String>, String> {
        if self.asleep {
            return Err("the scheduler sleeps: wake it instead".to_owned());
        }
        let Some(cache) = self.storage.cache() else {
            return Ok(Vec::new());
        };
        if !cache.block_pool.has_withdrawn() {
            return Ok(Vec::new());
        }
        let results = self.run_hook("uniserve.restore_cache", "")?;
        if let Some(cache) = self.storage.cache() {
            cache.block_pool.restore();
        }
        tracing::info!(units = self.storage.usable_units(), "restored the KV cache");
        Ok(results)
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
