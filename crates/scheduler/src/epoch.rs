//! The scheduler's part in its epoch's changes.
//!
//! A deployment updates by starting an epoch that succeeds the serving one on
//! the same devices; the daemon decides each step, and each scheduler acts:
//!
//! - **Park** (the serving epoch, at a promotion): at its next step boundary
//!   the scheduler ends every request that resumes from its generated tokens
//!   as migratable, ends every video request as migratable at its next
//!   denoising step boundary once its trajectory is stored (`preemption`),
//!   and once nothing runs, offloads the pages it reserved for the host KV
//!   cache and executes nothing more.
//!   It answers with its cache index: every published page's public and
//!   private continuation aliases with their units, which the successor
//!   reads from the planes it maps read-only without duplicating the bytes.
//!   When the successor maps its cache, it lends it the segments no
//!   request in flight continues from, whose pages it forgets, since the
//!   successor writes them on trial. Its storage and the rest of its index
//!   stay as they were, so a rollback resumes it.
//! - **Promote** (the successor): its workers copy the predecessor's cache
//!   flags and scales of the units they inherit and map the lent segments
//!   writable (`EpochOperation::AdoptCache`), its pool imports the predecessor's
//!   pages into those units, from which migrated requests reuse their
//!   prefixes, and allocates the lent ones as its own.
//! - **Resume** (the parked epoch, at a rollback): it admits and executes
//!   again, with the index it parked with, and allocates the units it lent
//!   once the epoch on trial writes none of them any more.
//! - **Inherit** (a successor whose planes map the predecessor's segments
//!   whole, as every epoch on the same devices does): once the predecessor
//!   drained, its index arrives through the daemon (`Drained`, `Inherit`).
//!   The workers copy the predecessor's cache flags and scales of the units
//!   its pages hold (`EpochOperation::AdoptCache`), and the pool adopts the
//!   pages at their own units, except those this epoch wrote meanwhile;
//!   the requests the predecessor handed over wait for it, and continue
//!   from those pages.
//! - **Commit** (the successor, once the predecessor is gone): its workers
//!   take the segments they mapped read-only (`uniserve.take_cache`), and its
//!   pool allocates the inherited units.
//!
//! Until its commit a successor is on trial: a worker fault or non-finite
//! logits fail it as a whole rather than being repaired, and its requests
//! that resume from their generated tokens leave as migratable
//! (`Scheduler::hand_back_requests`), for the daemon to continue on the
//! previous epoch.

use super::*;
use crate::kv_cache::IndexedPage;

/// One worker group's KV cache as an epoch that succeeds this one maps it:
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

/// What a parked scheduler hands the epoch that succeeds it: its public and
/// private cache index, and the units and count of the requests it ended as
/// migratable, which the successor's cache must hold to continue them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParkedCache {
    pub pages: Vec<IndexedPage>,
    /// The units of the segments the scheduler lent the successor at its
    /// park, in its own planes: the successor maps them writable and
    /// allocates them on trial (`specs/online-operations.md` §6.4).
    pub lent: Vec<std::ops::Range<u32>>,
    /// The cache units the migrating requests held: those a successor
    /// recomputing them computes again.
    pub held_units: u32,
    pub requests: u32,
    /// Of `held_units`, the units of the migrating requests that reserved
    /// their whole cache at admission (`TokenState::reserve_worstcase`,
    /// a request with images), and how many they were: a successor
    /// reserves their whole cache again, since it computes their context
    /// anew, whether or not it recomputes the others.
    pub reserved_units: u32,
    pub reserving: u32,
}

impl ParkedCache {
    /// The index as the process interface carries it: one entry per worker
    /// group of `workers` (each group with the endpoint of its worker), the
    /// pages whose planes that group's worker holds. The first entry carries
    /// what the migrating requests held and the lent units, which are the
    /// scheduler's; a page of no named worker is left out.
    pub fn into_indexes(
        self,
        workers: &[(String, ipc::MainProcessEndpoint)],
    ) -> Vec<proto::request::GroupIndex> {
        use proto::request::wire;
        let mut indexes: Vec<wire::GroupIndex> = workers
            .iter()
            .map(|(group, _)| wire::GroupIndex {
                worker_group: group.clone(),
                ..wire::GroupIndex::default()
            })
            .collect();
        for page in self.pages {
            let Some(at) = workers.iter().position(|(_, endpoint)| {
                page.source
                    .as_deref()
                    .is_some_and(|source| source == endpoint)
            }) else {
                continue;
            };
            indexes[at].pages.push(wire::IndexedPage {
                group: page.group,
                hash: page.hash.0.to_vec(),
                tokens: page.tokens,
                units: page.units,
                resume: page.resume.map(|resume| wire::CacheResume {
                    epoch: resume.epoch,
                    request_id: resume.request_id,
                }),
            });
        }
        if let Some(first) = indexes.first_mut() {
            first.held_units = self.held_units;
            first.requests = self.requests;
            first.reserved_units = self.reserved_units;
            first.reserving = self.reserving;
            first.lent = self
                .lent
                .into_iter()
                .map(|units| wire::UnitRange {
                    start: units.start,
                    end: units.end,
                })
                .collect();
        }
        indexes
    }

    /// The parked cache an index carried. A page whose key is not 32 bytes
    /// is no page of this build's chains: it is left out, and a prompt
    /// computes it.
    pub fn from_index(index: &proto::request::GroupIndex) -> Self {
        Self {
            pages: index
                .pages
                .iter()
                .filter_map(|page| {
                    Some(IndexedPage {
                        group: page.group,
                        hash: storage::Key(page.hash.as_slice().try_into().ok()?),
                        tokens: page.tokens.clone(),
                        units: page.units.clone(),
                        resume: page
                            .resume
                            .as_ref()
                            .map(|resume| proto::request::CacheResume {
                                epoch: resume.epoch,
                                request_id: resume.request_id,
                            }),
                        source: None,
                    })
                })
                .collect(),
            held_units: index.held_units,
            requests: index.requests,
            reserved_units: index.reserved_units,
            reserving: index.reserving,
            lent: index
                .lent
                .iter()
                .map(|units| units.start..units.end)
                .collect(),
        }
    }
}

/// A park waiting for the scheduler's requests to end.
pub(super) struct Parking {
    answer: crossbeam_channel::Sender<Result<ParkedCache, String>>,
    /// Whether the epoch that succeeds this one maps its cache, and borrows
    /// the segments it lends.
    lend: bool,
    held_units: u32,
    requests: u32,
    reserved_units: u32,
    reserving: u32,
}

impl Scheduler {
    /// Parks the scheduler (see the module documentation), answering with
    /// its prefix cache index once it is parked; when it `lend`s, the epoch
    /// that succeeds it maps its cache and borrows the segments it lends
    /// ([`Self::lend_cache`]).
    pub(super) fn park(
        &mut self,
        answer: crossbeam_channel::Sender<Result<ParkedCache, String>>,
        lend: bool,
    ) {
        if matches!(self.lifecycle, crate::Lifecycle::Parked) {
            let _ = answer.send(Ok(ParkedCache {
                pages: self.index(),
                lent: self
                    .storage
                    .cache()
                    .map(|cache| cache.pool().lent())
                    .unwrap_or_default(),
                ..ParkedCache::default()
            }));
            return;
        }
        if matches!(self.lifecycle, crate::Lifecycle::Parking(_)) {
            let _ = answer.send(Err("the scheduler parks already".to_owned()));
            return;
        }
        // What the requests that leave as migratable hold, before they let
        // it go, and whether they reserved their whole cache at admission.
        let migrating: Vec<(u32, bool)> = self
            .requests
            .running_tokens()
            .filter(|(_, status, _)| !status.is_finished())
            .map(|(_, _, state)| {
                let units = state.allocations.as_ref().map_or(0, |allocations| {
                    allocations
                        .block_tables()
                        .iter()
                        .map(|table| table.held_units() as u32)
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
        // The requests held for a predecessor's index leave with the others.
        self.release_held_for_index();
        self.migrate_requests(MigrationCause::Drain);
        // Video requests leave at their next denoising step boundary, their
        // trajectories stored for the epoch that serves next.
        self.preempt_media(MigrationCause::Drain, |_| true);
        self.lifecycle = crate::Lifecycle::Parking(Parking {
            answer,
            lend,
            held_units: migrating.iter().map(|(units, _)| units).sum(),
            requests: migrating.len() as u32,
            reserved_units: reserved.iter().sum(),
            reserving: reserved.len() as u32,
        });
        self.poll_park();
    }

    /// Source units and accepted prefix lengths of requests that can move.
    /// The target sizes recomputation from these lengths at its own layout.
    pub(super) fn migrating_cache(&self) -> (u32, Vec<proto::request::wire::RequestPrefix>) {
        let mut units = 0;
        let mut tokens = Vec::new();
        for (_, _, state) in self
            .requests
            .running_tokens()
            .filter(|(_, status, _)| !status.is_finished())
        {
            units += state.allocations.as_ref().map_or(0, |allocations| {
                allocations
                    .block_tables()
                    .iter()
                    .map(|table| table.held_units() as u32)
                    .sum::<u32>()
            });
            tokens.push(proto::request::wire::RequestPrefix {
                request_id: state.req.request_id.0,
                tokens: state.cache_visible_len,
                resume: state
                    .req
                    .cache
                    .resume
                    .map(|resume| proto::request::wire::CacheResume {
                        epoch: resume.epoch,
                        request_id: resume.request_id,
                    }),
            });
        }
        (units, tokens)
    }

    /// Completes a park once no request is queued, runs or retires and no
    /// batch is in flight.
    pub(super) fn poll_park(&mut self) {
        if !matches!(self.lifecycle, crate::Lifecycle::Parking(_))
            || !(self.running.is_empty()
                && self.retiring_requests.is_empty()
                && self.waiting.is_empty()
                && self.inflight.pending_batches.is_empty()
                && self.inflight.pending_submissions.is_empty())
        {
            return;
        }
        let crate::Lifecycle::Parking(parking) =
            std::mem::replace(&mut self.lifecycle, crate::Lifecycle::Parked)
        else {
            return;
        };
        // The pages reserved for the host KV cache reach it, so the
        // successor and a later eviction of the parked storage find them
        // there.
        self.flush_offloads();
        let index = self.index();
        let lent = if parking.lend {
            self.lend_cache(&index)
        } else {
            Vec::new()
        };
        tracing::info!(
            pages = index.len(),
            held_units = parking.held_units,
            requests = parking.requests,
            lent_units = lent.iter().map(|units| units.len()).sum::<usize>(),
            "the scheduler parked"
        );
        let _ = parking.answer.send(Ok(ParkedCache {
            pages: index,
            lent,
            held_units: parking.held_units,
            requests: parking.requests,
            reserved_units: parking.reserved_units,
            reserving: parking.reserving,
        }));
    }

    /// Whether admission and scheduling wait for a park: the scheduler
    /// parked, or parks once its requests end.
    pub(super) fn parks(&self) -> bool {
        matches!(self.lifecycle, crate::Lifecycle::Parked)
    }

    /// Resumes a parked scheduler: it admits and executes again. When it
    /// `reclaim`s, the epoch it lent cache units to at its park writes none
    /// of them any more, and the pool allocates them again.
    pub(super) fn resume(&mut self, reclaim: bool) -> Result<(), String> {
        if matches!(self.lifecycle, crate::Lifecycle::Parking(_)) {
            return Err("the scheduler is still parking".to_owned());
        }
        if matches!(self.lifecycle, crate::Lifecycle::Parked) {
            self.lifecycle = crate::Lifecycle::Serving;
            tracing::info!("the parked scheduler resumes");
        }
        if reclaim && let Some(cache) = self.storage.cache() {
            let reclaimed = cache.pool().reclaim();
            if reclaimed > 0 {
                tracing::info!(
                    units = reclaimed,
                    "took back the cache units lent to the successor epoch"
                );
            }
        }
        Ok(())
    }

    /// Lends the epoch that succeeds this one, which maps this epoch's
    /// cache, the segments no request in flight continues from
    /// (`specs/online-operations.md` §6.4): the segments the planes hold
    /// past the first, which holds the padding and warmup units, and past
    /// those the epoch inherited and does not own yet, none of whose units
    /// holds a page of `index` under a private continuation alias. The
    /// successor maps them writable and allocates their units on trial, so
    /// the pool forgets their pages and allocates none of their units until
    /// it reclaims them ([`Self::resume`]). Returns their units.
    fn lend_cache(&self, index: &[IndexedPage]) -> Vec<std::ops::Range<u32>> {
        let Some(cache) = self.storage.cache() else {
            return Vec::new();
        };
        // Every worker splits the planes alike (`Self::describe_cache`).
        let Some(segments) = self.worker.info().workers.iter().find_map(|(_, info)| {
            info.prefix_cache
                .as_ref()
                .map(|cache| cache.segments.clone())
                .filter(|segments| !segments.is_empty())
        }) else {
            return Vec::new();
        };
        let pool = cache.pool();
        let inherited = pool.inherited();
        let held = pool.withdrawn();
        let continued: std::collections::BTreeSet<u32> = index
            .iter()
            .filter(|page| page.resume.is_some())
            .flat_map(|page| page.units.iter().copied())
            .collect();
        let mut lent = Vec::new();
        let mut start = 0;
        for (position, segment) in segments.iter().enumerate() {
            let units = start..start + segment.units;
            start = units.end;
            let foreign = units.start < inherited.end && inherited.start < units.end;
            if position == 0
                || units.end > held
                || foreign
                || continued.range(units.clone()).next().is_some()
            {
                continue;
            }
            lent.push(units);
        }
        let forgotten = pool.lend(&lent);
        tracing::info!(
            segments = lent.len(),
            of = segments.len(),
            forgotten_pages = forgotten,
            "lent the successor epoch the cache segments no request in flight continues from"
        );
        lent
    }

    /// Promotes a successor scheduler: its workers adopt the predecessor's
    /// cache flags of the units they inherit and map the segments the
    /// predecessor lent writable, its pool imports the predecessor's
    /// published pages, each unit moved to where the workers map it, and
    /// allocates the lent units as its own.
    ///
    /// Fails, changing nothing, when the epoch's KV cache limit in the
    /// trial cannot hold what the requests the predecessor hands over
    /// compute: the whole cache a request reserved at admission (one with
    /// images), which it reserves again, and of every other request all the
    /// units it held when they `recompute` from their tokens, beside a page
    /// each, the last they continue. The limit is the units its own
    /// segments hold free, those of the segments the predecessor lent, and
    /// those of the segments it has yet to create that every device's
    /// storage ledger has room for: the quota less everything booked, the
    /// predecessor's kept storage included, and, when the predecessor's
    /// processes exit at the promotion (it `grows`), with what they booked
    /// transferred to this epoch. The epoch has no more until its commit
    /// gives it the predecessor's segments, and the requests would wait for
    /// it.
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
        let own_free = cache.pool().free_units() as u32;
        // The segments the predecessor lent, in this pool's units, where the
        // workers map the predecessor's planes after the epoch's head.
        let inherited = cache.pool().inherited();
        let borrowed: Vec<std::ops::Range<u32>> = parked
            .lent
            .iter()
            .filter_map(|lent| {
                let units = lent.start.checked_add(inherited.start)?
                    ..lent.end.checked_add(inherited.start)?;
                (!units.is_empty() && inherited.start <= units.start && units.end <= inherited.end)
                    .then_some(units)
            })
            .collect();
        let lent: u32 = borrowed.iter().map(|units| units.end - units.start).sum();
        let growable = match self.cache_storage()? {
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
        let free = own_free + lent + growable;
        if needed > free {
            return Err(format!(
                "the successor epoch's KV cache holds {free} free units within its storage \
                 ledgers until its commit ({own_free} in its own segments, {lent} in the \
                 segments its predecessor lent, {growable} in the segments the ledgers have \
                 room for), and the {} requests in flight need {needed}{}{}",
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
        let index = &parked.pages;
        if inherited.is_empty() {
            return Ok(());
        }
        self.run_operation(ipc::memory::AdoptCache {
            borrowed: borrowed
                .iter()
                .map(|units| [units.start, units.end])
                .collect(),
            units: vec![[inherited.start, inherited.end]],
        })?;
        let source = self
            .worker
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
        let imported = cache.pool().import(index, inherited.start, &source);
        let borrowed = cache.pool().borrow(&borrowed);
        tracing::info!(
            pages = imported,
            offered = index.len(),
            units = inherited.end - inherited.start,
            borrowed,
            "imported the predecessor's prefix cache index"
        );
        Ok(())
    }

    /// Adopts the pages the drained predecessor epoch parked with, which lie
    /// at their own units of the planes this scheduler's workers map (see
    /// the module documentation): the workers copy the predecessor's flags
    /// and scales of those units, and the pool adopts every page none of
    /// whose units this epoch wrote. The requests held for the index are
    /// admitted from then on.
    ///
    /// `parked` holds one cache per worker group of the scheduler whose
    /// predecessor's planes held pages: a page is read where that group's
    /// worker maps it, and the groups share one unit space.
    pub(super) fn inherit(&mut self, parked: &[(String, ParkedCache)]) -> Result<(), String> {
        let inherited = self.adopt_index(parked);
        self.release_held_for_index();
        inherited
    }

    fn adopt_index(&mut self, parked: &[(String, ParkedCache)]) -> Result<(), String> {
        let Some(cache) = self.storage.cache() else {
            return Ok(());
        };
        if parked.iter().all(|(_, cache)| cache.pages.is_empty()) {
            return Ok(());
        }
        if !cache.pool().inherited().is_empty() {
            return Err(
                "a scheduler that maps its predecessor's planes behind its own \
                        imports its index at its promotion"
                    .to_owned(),
            );
        }
        // The pages whose units still hold what the predecessor left, and
        // those units, whose flags and scales the workers copy, as half-open
        // runs. Nothing allocates between the two: this thread does.
        let workers = self.worker.info().workers.clone();
        let mut adoptable = Vec::with_capacity(parked.len());
        for (group, parked) in parked {
            let source = workers
                .iter()
                .find(|(id, info)| id.to_string() == *group && info.prefix_cache.is_some())
                .map(|(_, info)| Arc::new(info.endpoint.clone()))
                .ok_or_else(|| format!("no worker of group {group} maps a cache"))?;
            adoptable.push((source, cache.pool().adoptable(&parked.pages)));
        }
        let mut units: Vec<u32> = adoptable
            .iter()
            .flat_map(|(_, pages)| pages.iter().flat_map(|page| page.units.iter().copied()))
            .collect();
        units.sort_unstable();
        units.dedup();
        let mut runs: Vec<[u32; 2]> = Vec::new();
        for unit in units {
            match runs.last_mut() {
                Some(run) if run[1] == unit => run[1] += 1,
                _ => runs.push([unit, unit + 1]),
            }
        }
        let adopted: Vec<ipc::memory::CacheAdopted> =
            self.run_operation(ipc::memory::AdoptCache {
                borrowed: Vec::new(),
                units: runs,
            })?;
        let Some(cache) = self.storage.cache() else {
            return Ok(());
        };
        let imported: usize = adoptable
            .iter()
            .map(|(source, pages)| cache.pool().adopt(pages, source))
            .sum();
        tracing::info!(
            pages = imported,
            offered = parked
                .iter()
                .map(|(_, cache)| cache.pages.len())
                .sum::<usize>(),
            copied_units = adopted.iter().map(|rank| rank.inherited).max().unwrap_or(0),
            "imported the predecessor's prefix cache index"
        );
        Ok(())
    }

    /// Commits a successor scheduler: its workers take the predecessor's
    /// segments they mapped read-only, and its pool allocates their units.
    /// Returns what each worker reported.
    pub(super) fn commit(&mut self) -> Result<Vec<String>, String> {
        let draining = std::mem::replace(&mut self.draining, true);
        while !(self.inflight.pending_batches.is_empty()
            && self.inflight.pending_submissions.is_empty())
        {
            if self.failed() {
                self.draining = draining;
                return Err("the engine failed before its commit".to_owned());
            }
            if !self.step_nonblocking() {
                self.park_for_progress();
            }
        }
        self.draining = draining;
        let results = self.worker.commit().map_err(|error| format!("{error:#}"))?;
        if let Some(cache) = self.storage.cache() {
            cache.pool().take_inherited();
        }
        self.trial = false;
        tracing::info!(
            units = self.storage.usable_units(),
            "the scheduler's epoch is committed"
        );
        Ok(results)
    }

    /// The KV cache every worker group holds now, as a successor maps it;
    /// the ranks of a group agree on it.
    pub(super) fn describe_cache(&mut self) -> Result<Vec<GroupCache>, String> {
        let units: BTreeMap<String, u32> = self
            .worker
            .info()
            .workers
            .iter()
            .filter_map(|(group, info)| {
                info.prefix_cache
                    .as_ref()
                    .map(|cache| (group.to_string(), cache.num_units))
            })
            .collect();
        let draining = std::mem::replace(&mut self.draining, true);
        while !(self.inflight.pending_batches.is_empty()
            && self.inflight.pending_submissions.is_empty())
        {
            if self.failed() {
                self.draining = draining;
                return Err("the engine failed before it described its cache".to_owned());
            }
            if !self.step_nonblocking() {
                self.park_for_progress();
            }
        }
        self.draining = draining;
        let reports = self
            .worker
            .operate_groups(&ipc::memory::CacheStorage.into())
            .map_err(|error| format!("{error:#}"))?;
        let mut groups = Vec::new();
        for (group, results) in reports {
            let Some(units) = units.get(&group.to_string()).copied() else {
                continue;
            };
            let mut described: Option<ipc::memory::CacheStorageReport> = None;
            let replies = ipc::memory::replies::<ipc::memory::CacheStorage>(&results)
                .map_err(|error| format!("a worker's cache report: {error}"))?;
            for report in replies {
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
            // The units its warmup lends are the group's report at startup,
            // which no rank reports again.
            let scratch = self
                .worker
                .info()
                .workers
                .iter()
                .find(|(id, _)| *id == group)
                .and_then(|(_, info)| info.prefix_cache.as_ref())
                .map_or(0, |cache| cache.scratch_units);
            if let Some(report) = described {
                groups.push(GroupCache {
                    worker_group: group.to_string(),
                    segments: report.segments,
                    held: report.held,
                    units,
                    scratch,
                });
            }
        }
        Ok(groups)
    }

    /// The scheduler's prefix cache index: every page its pool publishes.
    fn index(&self) -> Vec<IndexedPage> {
        self.storage
            .cache()
            .map(|cache| cache.pool().export())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoint(incarnation: &str) -> ipc::MainProcessEndpoint {
        ipc::MainProcessEndpoint {
            incarnation: incarnation.to_owned(),
            ..ipc::MainProcessInfo::default().endpoint
        }
    }

    fn page(unit: u32, source: &ipc::MainProcessEndpoint) -> IndexedPage {
        IndexedPage {
            group: 0,
            hash: storage::Key([unit as u8; 32]),
            tokens: vec![unit; 4],
            units: vec![unit],
            resume: None,
            source: Some(Arc::new(source.clone())),
        }
    }

    /// A scheduler of two worker groups (prefill and decode) hands each
    /// group the pages its worker's planes hold, and a receiver reads each
    /// group's pages back as they were.
    #[test]
    fn an_index_names_each_page_by_the_group_whose_planes_hold_it() {
        let (prefill, decode) = (endpoint("prefill"), endpoint("decode"));
        let parked = ParkedCache {
            pages: vec![page(3, &prefill), page(5, &decode), page(7, &decode)],
            held_units: 4,
            requests: 1,
            ..ParkedCache::default()
        };
        let workers = vec![
            ("prefill".to_owned(), prefill),
            ("decode".to_owned(), decode),
        ];
        let indexes = parked.into_indexes(&workers);
        let groups: Vec<(&str, Vec<u32>)> = indexes
            .iter()
            .map(|index| {
                let units = index
                    .pages
                    .iter()
                    .flat_map(|page| page.units.clone())
                    .collect();
                (index.worker_group.as_str(), units)
            })
            .collect();
        assert_eq!(groups, [("prefill", vec![3]), ("decode", vec![5, 7])]);
        assert_eq!((indexes[0].held_units, indexes[0].requests), (4, 1));

        let decoded = ParkedCache::from_index(&indexes[1]);
        let units: Vec<u32> = decoded
            .pages
            .iter()
            .flat_map(|page| page.units.clone())
            .collect();
        assert_eq!(units, [5, 7]);
        assert!(decoded.pages.iter().all(|page| page.source.is_none()));
    }
}
