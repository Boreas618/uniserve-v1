//! Host allocations holding a successor's cache pages before device admission.

use ::storage::Key;
use ::storage::manifest::cache::Page;
use ::storage::manifest::{Binding, Chunk, Rank};
use ::storage::protocol::Placement;

use crate::storage::retained::{ExclusiveSpec, Retained};
use crate::{Error, Result};

struct Held {
    key: Key,
    page: Page,
    rank: Rank,
}

/// Retains each round's compact host allocation until its copies retire.
/// Numerical views preserve the target layout; only their backing changes.
#[derive(Default)]
pub(crate) struct Staging {
    held: Vec<Held>,
    next: u64,
}

impl Staging {
    pub(crate) fn reserve(
        &mut self,
        retained: &Retained,
        device: cuda::DeviceId,
        page: Page,
        mut rank: Rank,
    ) -> Result<Option<(Rank, Key)>> {
        // Coverage advances only at a boundary shared by all cache groups.
        // Dense pages ahead of a recurrent checkpoint can therefore appear
        // in several snapshots; their completed Host copy stays immutable.
        if self.held.iter().any(|held| held.page == page) {
            return Ok(None);
        }
        let bytes: u64 = rank
            .storages
            .iter()
            .map(|storage| storage.layout.bytes)
            .sum();
        let name = format!("cache staging/{}", self.next);
        self.next += 1;
        let key = retained.exclusive_key(
            &ExclusiveSpec {
                name: &name,
                device,
                bytes: bytes as usize,
                access: &[],
            },
            &device.uuid()?.0,
        );
        // The server admits and retains the host backing. Only the
        // Transfer Server maps it; this rank never reads its staged bytes.
        retained.call("allocating cache staging", |client| {
            client.allocate_host(key, name, bytes)
        })?;
        retained.use_keys([key]);
        let mut offset = 0;
        for storage in &mut rank.storages {
            storage.chunks = vec![Chunk {
                offset: 0,
                length: storage.layout.bytes,
                key,
                storage_offset: offset,
                binding: Binding::Local {
                    at: Placement::Host { numa_node: None },
                },
            }];
            offset += storage.layout.bytes;
        }
        self.held.push(Held {
            key,
            page,
            rank: rank.clone(),
        });
        Ok(Some((rank, key)))
    }

    pub(crate) fn source(&self, gpu: ::storage::manifest::Gpu) -> Rank {
        Rank {
            gpu: Some(gpu),
            storages: self
                .held
                .iter()
                .flat_map(|held| held.rank.storages.clone())
                .collect(),
        }
    }

    /// The caller has completed or cancelled every transfer using these bytes.
    pub(crate) fn release(&mut self, retained: &Retained, mut keys: Vec<Key>) -> Result<()> {
        // A restarted predecessor retains its rollback keys at startup;
        // their allocations were created by the exited worker process.
        keys.extend(self.held.iter().map(|held| held.key));
        retained
            .forget(&keys)
            .map_err(|error| Error::Interface(format!("releasing cache staging: {error}")))?;
        self.held.clear();
        Ok(())
    }

    pub(crate) fn allocations(&self) -> Vec<(Key, Page)> {
        self.held
            .iter()
            .map(|held| (held.key, held.page.clone()))
            .collect()
    }

    /// Retire completed requests and superseded windows after their copies
    /// completed. Full attention retains its entire prefix.
    pub(crate) fn retain(&mut self, retained: &Retained, requests: &[(u64, u64)]) -> Result<()> {
        let mut ends = std::collections::BTreeMap::new();
        for held in &self.held {
            let end = ends
                .entry((held.page.request, held.page.group))
                .or_insert(0);
            *end = (*end).max(held.page.start + held.page.tokens);
        }
        let mut released = Vec::new();
        self.held.retain(|held| {
            let page = &held.page;
            let window = page.window.or(page.state.then_some(page.tokens));
            let keep = requests.contains(&page.request)
                && window.is_none_or(|window| {
                    page.start + page.tokens
                        > ends[&(page.request, page.group)].saturating_sub(window)
                });
            if !keep {
                released.push(held.key);
            }
            keep
        });
        retained
            .forget(&released)
            .map_err(|error| Error::Interface(format!("retiring cache staging: {error}")))?;
        Ok(())
    }
}
