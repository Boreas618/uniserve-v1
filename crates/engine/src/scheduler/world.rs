//! The scheduler's part in its world's changes.
//!
//! A deployment updates by starting a world that succeeds the serving one on
//! the same devices; the daemon decides each step, and each scheduler acts:
//!
//! - **Park** (the serving world, at a promotion): at its next step boundary
//!   the scheduler ends every request that resumes from its generated tokens
//!   as migratable, ends every video request as migratable at its next
//!   denoising step boundary once its trajectory is stored (`preemption`),
//!   lets its readouts finish, and once nothing runs, offloads
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

/// What a parked scheduler hands the world that succeeds it: its prefix
/// cache index, and the units and count of the requests it ended as
/// migratable, which the successor's cache must hold to continue them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParkedCache {
    pub pages: Vec<IndexedPage>,
    /// The cache units the migrating requests held: those a successor
    /// recomputing them computes again.
    pub held_units: u32,
    pub requests: u32,
    /// Of `held_units`, the units of the migrating requests that reserved
    /// their whole cache at admission (`RequestState::reserve_worstcase`,
    /// a request with images), and how many they were: a successor
    /// reserves their whole cache again, since it computes their context
    /// anew, whether or not it recomputes the others.
    pub reserved_units: u32,
    pub reserving: u32,
}

/// A park waiting for the scheduler's requests to end.
pub(super) struct Parking {
    answer: crossbeam_channel::Sender<Result<ParkedCache, String>>,
    held_units: u32,
    requests: u32,
    reserved_units: u32,
    reserving: u32,
}

impl Scheduler {
    /// Parks the scheduler (see the module documentation), answering with
    /// its prefix cache index once it is parked.
    pub(super) fn park(&mut self, answer: crossbeam_channel::Sender<Result<ParkedCache, String>>) {
        if self.parked {
            let _ = answer.send(Ok(ParkedCache {
                pages: self.index(),
                ..ParkedCache::default()
            }));
            return;
        }
        if self.parking.is_some() {
            let _ = answer.send(Err("the scheduler parks already".to_owned()));
            return;
        }
        // What the requests that leave as migratable hold, before they let
        // it go, and whether they reserved their whole cache at admission.
        let migrating: Vec<(u32, bool)> = self
            .running
            .values()
            .filter(|state| state.req.is_migratable() && !state.terminal_intent.is_terminal())
            .map(|state| {
                let units = state.allocations.as_ref().map_or(0, |allocations| {
                    allocations
                        .block_tables()
                        .iter()
                        .map(|table| table.unit_ids().len() as u32)
                        .sum()
                });
                (units, state.reserve_worstcase)
            })
            .collect();
        let reserved: Vec<u32> = migrating
            .iter()
            .filter(|(_, reserves)| *reserves)
            .map(|(units, _)| *units)
            .collect();
        tracing::info!("parking: the requests that resume elsewhere end as migratable");
        self.migrate_requests(MigrationCause::Drain);
        // Video requests leave at their next denoising step boundary, their
        // trajectories stored for the world that serves next.
        self.preempt_media(MigrationCause::Drain, |_| true);
        self.parking = Some(Parking {
            answer,
            held_units: migrating.iter().map(|(units, _)| units).sum(),
            requests: migrating.len() as u32,
            reserved_units: reserved.iter().sum(),
            reserving: reserved.len() as u32,
        });
        self.poll_park();
    }

    /// Source units and accepted prefix lengths of requests that can move.
    /// The target sizes recomputation from these lengths at its own layout.
    pub(super) fn migrating_cache(&self) -> (u32, Vec<u32>) {
        let mut units = 0;
        let mut tokens = Vec::new();
        for state in self
            .running
            .values()
            .filter(|state| state.req.is_migratable() && !state.terminal_intent.is_terminal())
        {
            units += state.allocations.as_ref().map_or(0, |allocations| {
                allocations
                    .block_tables()
                    .iter()
                    .map(|table| table.unit_ids().len() as u32)
                    .sum::<u32>()
            });
            tokens.push(state.cache_visible_len);
        }
        (units, tokens)
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
        tracing::info!(
            pages = index.len(),
            held_units = parking.held_units,
            requests = parking.requests,
            "the scheduler parked"
        );
        let _ = parking.answer.send(Ok(ParkedCache {
            pages: index,
            held_units: parking.held_units,
            requests: parking.requests,
            reserved_units: parking.reserved_units,
            reserving: parking.reserving,
        }));
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
    /// cache flags of the units they inherit, and its pool imports the
    /// predecessor's published pages, each unit moved to where the workers
    /// map it.
    ///
    /// Fails, changing nothing, when the world's KV cache limit in the
    /// trial cannot hold what the requests the predecessor hands over
    /// compute: the whole cache a request reserved at admission (one with
    /// images), which it reserves again, and of every other request all the
    /// units it held when they `recompute` from their tokens, beside a page
    /// each, the last they continue. The limit is the
    /// units its own segments hold free, and those of the segments it has
    /// yet to create that every device's memory ledger has room for: the
    /// quota less everything booked, the predecessor's kept memory
    /// included, and, when the predecessor's processes exit at the
    /// promotion (it `grows`), with what they booked transferred to this
    /// world. The world has no more until its commit gives it the
    /// predecessor's segments, and the requests would wait for it.
    pub(super) fn promote(
        &mut self,
        parked: &ParkedCache,
        recompute: bool,
        grows: bool,
    ) -> Result<(), String> {
        let Some(cache) = self.storage.cache() else {
            return Ok(());
        };
        let page_units: u32 = cache
            .coordinator
            .groups()
            .iter()
            .map(|group| group.units_per_page as u32)
            .sum();
        // A request that reserved its whole cache at admission reserves it
        // again, and its context, rebuilt anew, reads no page the
        // predecessor published.
        let others = parked.requests.saturating_sub(parked.reserving);
        let needed = others * page_units
            + parked.reserved_units
            + if recompute {
                parked.held_units.saturating_sub(parked.reserved_units)
            } else {
                0
            };
        let own_free = cache.block_pool.free_units() as u32;
        let growable = match self.cache_memory()? {
            Some(reports) => {
                let fit = reports
                    .iter()
                    .map(|report| super::ledger::growable(report, grows))
                    .min()
                    .unwrap_or(0);
                let first = &reports[0];
                first.segments[first.held as usize..]
                    .iter()
                    .take(fit as usize)
                    .map(|segment| segment.units)
                    .sum()
            }
            None => 0,
        };
        let free = own_free + growable;
        if needed > free {
            return Err(format!(
                "the successor world's KV cache holds {free} free units within its memory \
                 ledgers until its commit ({own_free} in its own segments, {growable} in the \
                 segments the ledgers have room for), and the {} requests in flight need \
                 {needed}{}{}",
                parked.requests,
                if parked.reserving > 0 {
                    format!(
                        ", {} of them reserving their whole cache of {} units again",
                        parked.reserving, parked.reserved_units
                    )
                } else {
                    String::new()
                },
                if recompute {
                    ", recomputing the units they held"
                } else {
                    ""
                },
            ));
        }
        let Some(cache) = self.storage.cache() else {
            return Ok(());
        };
        let index = &parked.pages;
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
