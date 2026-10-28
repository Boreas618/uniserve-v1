//! Scheduler-owned leases and destinations for background cache transfers.

use super::*;
use crate::kv_cache::{CachedTokens, PageRef, PrefixScope};
use storage::manifest::cache::{Operation, Page, Prefix, Snapshot};

#[derive(Default)]
pub(super) struct Warming {
    leased: Vec<PageRef>,
    reserved: Vec<Reserved>,
    pub(super) prefill_tokens: Option<usize>,
}

struct Reserved {
    page: Page,
    reference: PageRef,
    hash: u64,
    tokens: Vec<u32>,
    source: Arc<ipc::WorkerEndpoint>,
    resume: proto::request::CacheResume,
    published: bool,
}

impl Warming {
    /// Admission has acquired every usable page into the request's ordinary
    /// block tables, so the warming reservation no longer pins those units.
    pub(super) fn claimed(&mut self, resume: proto::request::CacheResume) {
        self.reserved.retain(|page| page.resume != resume);
    }
}

impl Scheduler {
    /// Fence the current batches, mutate page ownership, and return concrete
    /// worker storage views. Device copies run later at the Transfer Server.
    pub(super) fn warm_cache(
        &mut self,
        operation: Operation,
        together: &crate::HookBarrier,
    ) -> Result<Vec<String>, String> {
        let closing = std::mem::replace(&mut self.closing, true);
        let result = self
            .drain_batches()
            .inspect_err(|error| together.fail(error))
            .and_then(|()| together.meet())
            .and_then(|()| self.warm_pages(operation));
        if let Err(error) = &result {
            together.fail(error);
        }
        let synchronized = together.meet();
        self.closing = closing;
        result.and_then(|snapshot| {
            synchronized?;
            serde_json::to_string(&snapshot)
                .map(|value| vec![value])
                .map_err(|error| error.to_string())
        })
    }

    fn warm_pages(&mut self, operation: Operation) -> Result<Snapshot, String> {
        let prefixes = match operation {
            Operation::Snapshot { world, after } => self.snapshot_pages(world, &after)?,
            Operation::Reserve { prefixes, keep } => self.reserve_pages(prefixes, keep)?,
            Operation::Coverage { prefixes, keep } => self.cache_coverage(prefixes, keep),
            Operation::Prefill { tokens } => {
                if tokens == Some(0) {
                    return Err("a warming prefill limit must be positive".into());
                }
                self.warming.prefill_tokens = tokens.map(|tokens| tokens as usize);
                Vec::new()
            }
            Operation::Retain { requests } => {
                self.warming
                    .reserved
                    .retain(|page| requests.contains(&(page.resume.world, page.resume.request_id)));
                Vec::new()
            }
            Operation::Publish => {
                let pages: Vec<_> = self
                    .warming
                    .reserved
                    .iter()
                    .filter(|page| !page.published)
                    .map(|page| page.page.clone())
                    .collect();
                let payload = serde_json::to_string(&pages).map_err(|error| error.to_string())?;
                self.executor
                    .hook("uniserve.cache_publish", &payload)
                    .map_err(|error| format!("{error:#}"))?;
                if let Some(cache) = self.storage.cache() {
                    for page in &mut self.warming.reserved {
                        if page.published {
                            continue;
                        }
                        cache.block_pool.activate([&page.reference]);
                        cache.block_pool.publish(
                            &page.reference,
                            page.page.group,
                            page.hash,
                            &page.tokens,
                            &page.source,
                            Some(page.resume),
                        );
                        page.published = true;
                    }
                    // Sliding attention and recurrent state need only their
                    // newest window. Keep its predecessor until publication
                    // has completed, then release superseded reservation pins.
                    let mut ends = std::collections::HashMap::new();
                    for page in &self.warming.reserved {
                        let end = ends.entry((page.resume, page.page.group)).or_insert(0);
                        *end = (*end).max(page.page.start + page.page.tokens);
                    }
                    self.warming.reserved.retain(|page| {
                        cache.coordinator.groups()[page.page.group as usize]
                            .window
                            .is_none_or(|window| {
                                let end = ends[&(page.resume, page.page.group)];
                                page.page.start + page.page.tokens
                                    > end.saturating_sub(window as u32)
                            })
                    });
                }
                Vec::new()
            }
            Operation::Release => {
                self.warming.leased.clear();
                Vec::new()
            }
            Operation::Discard => {
                self.warming.reserved.clear();
                Vec::new()
            }
        };
        let pages: Vec<_> = prefixes
            .iter()
            .flat_map(|prefix| prefix.pages.iter().cloned())
            .collect();
        let ranks = if pages.is_empty() {
            Vec::new()
        } else {
            let payload = serde_json::to_string(&pages).map_err(|error| error.to_string())?;
            self.executor
                .hook("uniserve.cache_manifest", &payload)
                .map_err(|error| format!("{error:#}"))?
                .into_iter()
                .map(|value| {
                    serde_json::from_str::<Option<storage::manifest::cache::Rank>>(&value)
                        .map_err(|error| error.to_string())
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .filter(|rank| !rank.storage.storages.is_empty())
                .collect()
        };
        Ok(Snapshot {
            page_tokens: self
                .storage
                .cache()
                .and_then(|cache| {
                    cache
                        .coordinator
                        .groups()
                        .iter()
                        .map(|shape| shape.block_size as u32)
                        .max()
                })
                .unwrap_or(1),
            prefixes,
            ranks,
            nodes: self.executor.storage_nodes(),
        })
    }

    fn snapshot_pages(&mut self, world: u64, after: &[(u64, u32)]) -> Result<Vec<Prefix>, String> {
        if !self.warming.leased.is_empty() {
            return Err("the previous cache snapshot still holds page leases".into());
        }
        let Some(cache) = self.storage.cache() else {
            return Ok(Vec::new());
        };
        let mut prefixes = Vec::new();
        for (&id, state) in &self.running {
            if !state.req.is_migratable() || state.terminal_intent.is_terminal() {
                continue;
            }
            let Some(allocation) = state.allocations.as_ref() else {
                continue;
            };
            let key = RequestKey::new(self.engine_id, id, state.request_epoch);
            let Some(worker) = self
                .placement
                .affinity
                .get(&(key, DEFAULT_COMPONENT.to_owned()))
            else {
                continue;
            };
            let own = state
                .req
                .generated_token_ids
                .len()
                .min(state.generated_token_ids.len());
            let tokens: Vec<_> = state
                .req
                .prompt_token_ids
                .iter()
                .chain(&state.generated_token_ids[own..])
                .copied()
                .collect();
            let mut prefix = Prefix {
                world,
                request: id.0,
                tokens,
                computed: state.cache_visible_len,
                cached: 0,
                isolation_key: state.req.cache.isolation_key,
                state_seed: state
                    .req
                    .cache
                    .state_seed
                    .unwrap_or(cache.coordinator.seed()),
                pages: Vec::new(),
            };
            for (group, (table, shape)) in allocation
                .block_tables()
                .iter()
                .zip(cache.coordinator.groups())
                .enumerate()
            {
                if shape.draft {
                    continue;
                }
                let copied = after
                    .iter()
                    .find(|(request, _)| *request == id.0)
                    .map_or(0, |(_, tokens)| *tokens as usize);
                for index in copied / shape.block_size..prefix.computed as usize / shape.block_size
                {
                    let Some(page) = table.page(index) else {
                        continue;
                    };
                    prefix.pages.push(Page {
                        tensor: format!("{world}/{}", id.0),
                        worker_group: worker.to_string(),
                        group: group as u32,
                        start: (index * shape.block_size) as u32,
                        tokens: shape.block_size as u32,
                        total_tokens: prefix.tokens.len() as u32,
                        state: shape.state,
                        units: page.units().iter().map(|unit| unit.0).collect(),
                    });
                    self.warming.leased.push(page.clone());
                }
            }
            // Even a request that wrote no new complete page participates
            // in coverage and retains its existing target reservations.
            prefixes.push(prefix);
        }
        Ok(prefixes)
    }

    fn reserve_pages(&mut self, prefixes: Vec<Prefix>, keep: bool) -> Result<Vec<Prefix>, String> {
        let Some(cache) = self.storage.cache() else {
            return Ok(Vec::new());
        };
        let (group_name, worker) = self
            .executor
            .info()
            .workers
            .iter()
            .find(|(_, worker)| worker.prefix_cache.is_some())
            .ok_or("no cache worker owns the target pages")?;
        let endpoint = Arc::new(worker.endpoint.clone());
        let mut targets = Vec::new();
        let mut reserved = Vec::new();
        for mut prefix in prefixes {
            let resume = proto::request::CacheResume {
                world: prefix.world,
                request_id: prefix.request,
            };
            let scope = PrefixScope {
                isolation_key: prefix.isolation_key,
                state_seed: keep.then_some(prefix.state_seed),
                resume: Some(resume),
            };
            let mut hashes = vec![Vec::new(); cache.coordinator.groups().len()];
            cache.coordinator.extend_hashes(
                &mut hashes,
                CachedTokens::prompt(&prefix.tokens),
                scope,
            );
            let source_pages = std::mem::take(&mut prefix.pages);
            for (group, shape) in cache.coordinator.groups().iter().enumerate() {
                if shape.draft {
                    continue;
                }
                let complete = if shape.state {
                    // A recurrent checkpoint describes the state at exactly
                    // this boundary. Repartition its fields, never relabel an
                    // earlier or later checkpoint as the desired state.
                    source_pages
                        .iter()
                        .filter(|page| page.state)
                        .map(|page| (page.start + page.tokens) as usize)
                        .filter(|end| end % shape.block_size == 0)
                        .max()
                        .unwrap_or(0)
                        / shape.block_size
                } else {
                    prefix.computed as usize / shape.block_size
                };
                let first = shape.window.map_or(0, |window| {
                    (complete * shape.block_size).saturating_sub(window) / shape.block_size
                });
                for (index, &hash) in hashes[group].iter().enumerate().take(complete).skip(first) {
                    let start = (index * shape.block_size) as u32;
                    if self.warming.reserved.iter().any(|page| {
                        page.resume == resume
                            && page.page.group == group as u32
                            && page.page.start == start
                    }) {
                        continue;
                    }
                    let reference = cache
                        .block_pool
                        .allocate(1, shape.units_per_page)
                        .ok_or("the target cache has no room for the warming pages")?
                        .remove(0);
                    let page = Page {
                        tensor: format!("{}/{}", prefix.world, prefix.request),
                        worker_group: group_name.to_string(),
                        group: group as u32,
                        start,
                        tokens: shape.block_size as u32,
                        total_tokens: prefix.tokens.len() as u32,
                        state: shape.state,
                        units: reference.units().iter().map(|unit| unit.0).collect(),
                    };
                    reserved.push(Reserved {
                        page: page.clone(),
                        reference,
                        hash,
                        tokens: prefix.tokens
                            [index * shape.block_size..(index + 1) * shape.block_size]
                            .to_vec(),
                        source: Arc::clone(&endpoint),
                        resume,
                        published: false,
                    });
                    prefix.pages.push(page);
                }
            }
            targets.push(prefix);
        }
        self.warming.reserved.extend(reserved);
        Ok(targets)
    }

    fn cache_coverage(&self, mut prefixes: Vec<Prefix>, keep: bool) -> Vec<Prefix> {
        if let Some(cache) = self.storage.cache() {
            for prefix in &mut prefixes {
                let resume = proto::request::CacheResume {
                    world: prefix.world,
                    request_id: prefix.request,
                };
                let scope = PrefixScope {
                    isolation_key: prefix.isolation_key,
                    state_seed: keep.then_some(prefix.state_seed),
                    resume: Some(resume),
                };
                prefix.cached = self
                    .executor
                    .info()
                    .workers
                    .iter()
                    .filter(|(_, worker)| worker.prefix_cache.is_some())
                    .map(|(_, worker)| {
                        cache
                            .coordinator
                            .probe_prefix(
                                &cache.block_pool,
                                &prefix.tokens,
                                true,
                                false,
                                scope,
                                &worker.endpoint,
                            )
                            .cached_tokens as u32
                    })
                    .max()
                    .unwrap_or(0);
                prefix.pages.clear();
            }
        }
        prefixes
    }
}
