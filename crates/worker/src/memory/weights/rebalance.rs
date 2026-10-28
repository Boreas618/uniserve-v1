//! Expert redistribution into retained allocations, keeping graph addresses.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::os::fd::OwnedFd;
use std::sync::Arc;

use storage::manifest::weights::{Sources, Transfer};
use storage::{Key, Mode, Placement};

use super::{MANIFEST_VERSION, Planned, Published, Receipt, WeightKeys, WeightManifest, read_text};
use crate::ffi::{Backing, Handle, Interpreter, MemoryKind, integers};
use crate::memory::allocation::Storage;
use crate::memory::retained::Retained;
use crate::parallel::host::HostGroup;
use crate::{Error, Result};

/// Incoming physical allocations and the graph addresses they will replace.
/// Preparation leaves live tensors untouched. Commit requires every rank's
/// copies complete and all computations drained at the common step boundary.
pub(crate) struct Rebalance {
    keys: WeightKeys,
    targets: Vec<Arc<Storage>>,
    backing: Vec<Arc<Storage>>,
    chunks: Vec<Vec<(usize, Key)>>,
    receipt: Option<Receipt>,
    descriptors: Option<Vec<Vec<(usize, OwnedFd)>>>,
    previous: BTreeSet<Key>,
    incoming: BTreeSet<Key>,
    bound: Handle,
    publication: Published,
}

impl WeightKeys {
    /// Compute the exact copy plan before asking the serving scheduler for room.
    pub(crate) fn plan_rebalance(
        &self,
        retained: &Retained,
        planned: &Planned,
        sources: &mut Sources,
    ) -> Result<Transfer> {
        self.measure_sources(retained, sources)?;
        let (_, gpu) = self.device(0)?;
        let request = storage::manifest::Request {
            source: sources.world.clone(),
            target: storage::manifest::World {
                ranks: vec![storage::manifest::Rank {
                    gpu: Some(gpu),
                    storages: planned
                        .storages
                        .iter()
                        .map(|held| held.storage.clone())
                        .collect(),
                }],
            },
            gpus: vec![gpu],
            links: sources.links.clone(),
        };
        let manifest = storage::manifest::Manifest::plan(&request)
            .map_err(|error| Error::Interface(error.to_string()))?;
        if !manifest.missing.is_empty() || !manifest.decodes.is_empty() {
            return Err(Error::State(
                "expert source manifests must cover every requested weight",
            ));
        }
        Ok(Transfer { request, manifest })
    }

    fn retained_keys(&self, retained: &Retained) -> Result<BTreeSet<Key>> {
        let text = read_text(retained, self.manifest)?
            .ok_or(Error::State("the active weight manifest is retained"))?;
        let manifest: WeightManifest = serde_json::from_str(&text)
            .map_err(|error| Error::Interface(format!("weight manifest: {error}")))?;
        Ok(manifest
            .storages
            .iter()
            .flat_map(|storage| storage.storage.chunks.iter().map(|chunk| chunk.key))
            .chain([self.manifest])
            .collect())
    }

    /// Stage a new expert distribution using ordinary weight copy manifests.
    pub(crate) fn rebalance(
        &self,
        retained: &Retained,
        planned: Planned,
        live: &[Arc<Storage>],
        previous: &WeightKeys,
        sources: &Sources,
        transfer: Transfer,
    ) -> Result<Rebalance> {
        let targets = planned.targets(live)?;
        let chunks: Vec<Vec<(usize, Key)>> = planned
            .storages
            .iter()
            .map(|storage| {
                storage
                    .storage
                    .chunks
                    .iter()
                    .map(|chunk| (chunk.length as usize, chunk.key))
                    .collect()
            })
            .collect();
        let mut incoming: BTreeSet<Key> = chunks
            .iter()
            .flatten()
            .map(|(_, key)| *key)
            .chain([self.manifest])
            .collect();
        let text = read_text(retained, previous.manifest)?
            .ok_or(Error::State("the active weight manifest is retained"))?;
        let previous_manifest: WeightManifest = serde_json::from_str(&text)
            .map_err(|error| Error::Interface(format!("weight manifest: {error}")))?;
        let previous = previous.retained_keys(retained)?;
        let mut publication = Published {
            keys: self.clone(),
            layout: planned.layout.clone(),
            manifest: WeightManifest {
                version: MANIFEST_VERSION,
                fingerprints: self.fingerprint_digests(),
                tensors: planned.tensors.clone(),
                storages: planned.storages.clone(),
            },
            storages: targets.clone(),
            created: Vec::new(),
            shared: chunks.iter().flatten().map(|(_, key)| *key).collect(),
        };
        let host = previous_manifest
            .storages
            .into_iter()
            .filter(|storage| storage.device.is_none());
        let host_owners = live
            .iter()
            .filter(|owner| matches!(owner.device(), MemoryKind::PinnedHost | MemoryKind::Host));
        for (storage, owner) in host.zip(host_owners) {
            incoming.extend(storage.storage.chunks.iter().map(|chunk| chunk.key));
            publication
                .shared
                .extend(storage.storage.chunks.iter().map(|chunk| chunk.key));
            publication.manifest.storages.push(storage);
            publication.storages.push(Arc::clone(owner));
        }
        // Each rank plans its own receiver; the source manifests describe the
        // whole world. The engine synchronizes publication across all ranks.
        let mut description = planned.into_transfer();
        description.tensors.retain(|tensor| !tensor.host);
        let found = self.receive(
            retained,
            description,
            sources,
            &HostGroup::single(0),
            Some(transfer),
        )?;
        if !found.missing.is_empty() {
            if let Some(receipt) = found.receipt {
                receipt.cancel(retained)?;
            }
            let release = incoming.difference(&previous).copied().collect::<Vec<_>>();
            retained.forget(&release)?;
            return Err(Error::State(
                "expert source manifests must cover every requested weight",
            ));
        }
        let bound = found.bind(&BTreeSet::new())?;
        let (backing, receipt) = found.into_storages();
        Ok(Rebalance {
            keys: self.clone(),
            targets,
            backing,
            chunks,
            receipt,
            descriptors: None,
            previous,
            incoming,
            bound,
            publication,
        })
    }
}

impl Planned {
    /// Resolve the fixed addresses independently of logical expert order.
    fn targets(&self, live: &[Arc<Storage>]) -> Result<Vec<Arc<Storage>>> {
        let addresses = integers(
            Interpreter::import("uniserve.loading.retained.Layout.storage_addresses")?
                .call_values(&[self.layout.to_any()])?,
        )
        .map_err(|error| Error::numerical("storage_addresses", &error))?;
        let mut targets = vec![None; self.storages.len()];
        for (&address, &(storage, offset)) in addresses.iter().zip(&self.placed) {
            let base = address as u64 - offset;
            let target = live
                .iter()
                .find(|allocation| allocation.address() == base)
                .ok_or(Error::State(
                    "expert redistribution requires retained weight addresses",
                ))?;
            let bytes: u64 = self.storages[storage]
                .storage
                .chunks
                .iter()
                .map(|chunk| chunk.length)
                .sum();
            if target.bytes() as u64 != bytes {
                return Err(Error::Invalid(
                    "expert redistribution must preserve storage extents".into(),
                ));
            }
            targets[storage] = Some(Arc::clone(target));
        }
        targets
            .into_iter()
            .map(|target| target.ok_or(Error::State("every weight storage has a live tensor")))
            .collect()
    }

    /// Keep creating handles for active content keys before remapping their
    /// original reservations. Imported allocations have an external creator.
    pub(crate) fn allocations(
        &self,
        live: &[Arc<Storage>],
    ) -> Result<BTreeMap<Key, Arc<cuda::PhysicalAllocation>>> {
        let mut allocations = BTreeMap::new();
        for (storage, layout) in self.targets(live)?.iter().zip(&self.storages) {
            if storage.allocation().is_some() {
                // DWDP owns creating handles in its resident peer storage.
                continue;
            }
            for (index, chunk) in layout.storage.chunks.iter().enumerate() {
                if let Some(allocation) = storage.physical_allocation(index)? {
                    allocations.insert(chunk.key, allocation);
                }
            }
        }
        Ok(allocations)
    }
}

impl Rebalance {
    /// Finish background copies on a preparation thread, without blocking the
    /// serving worker's control hook. The returned tensor views borrow backing
    /// retained by this operation until the distributed owner finishes staging.
    pub(crate) fn wait(&mut self, retained: &Retained) -> Result<&Handle> {
        if let Some(receipt) = self.receipt.as_mut() {
            receipt.retain(retained, true)?;
        }
        self.receipt = None;
        self.ready(retained)?;
        Ok(&self.bound)
    }
    /// Retain completed copies and acquire their original descriptors.
    /// Imported CUDA handles cannot be re-exported as allocation descriptors.
    pub(crate) fn ready(&mut self, retained: &Retained) -> Result<bool> {
        if let Some(receipt) = self.receipt.as_mut()
            && !receipt.ready(retained)?
        {
            return Ok(false);
        }
        if let Some(receipt) = self.receipt.as_mut() {
            receipt.retain(retained, true)?;
        }
        self.receipt = None;
        if self.descriptors.is_none() {
            let (_, gpu) = self.keys.device(0)?;
            let keys = self.chunks.iter().flatten().map(|(_, key)| *key).collect();
            let got = retained.call("borrowing completed expert allocations", |client| {
                client.get(keys, Mode::Read, vec![Placement::Device { gpu }])
            })?;
            let mut descriptors = HashMap::new();
            for tensor in got.tensors {
                let descriptor = got.storages[tensor.storage as usize]
                    .1
                    .try_clone()
                    .map_err(|error| {
                        Error::transport_source("borrowing an expert allocation", error)
                    })?;
                descriptors.insert(tensor.key, descriptor);
            }
            self.descriptors = Some(
                self.chunks
                    .iter()
                    .map(|chunks| {
                        chunks
                            .iter()
                            .map(|(bytes, key)| {
                                let descriptor = descriptors
                                    .get(key)
                                    .ok_or(Error::State(
                                        "a completed expert allocation is retained",
                                    ))?
                                    .try_clone()
                                    .map_err(|error| {
                                        Error::transport_source(
                                            "borrowing an expert allocation",
                                            error,
                                        )
                                    })?;
                                Ok((*bytes, descriptor))
                            })
                            .collect::<Result<Vec<_>>>()
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        Ok(true)
    }

    /// Install allocations at the captured addresses. A driver failure after
    /// the first remap invalidates the worker; it must recover from its launch
    /// distribution before any numerical call resumes.
    pub(crate) fn commit(
        mut self,
        retained: &Retained,
        allocations: &mut BTreeMap<Key, Arc<cuda::PhysicalAllocation>>,
        resident: Option<(Handle, Vec<Arc<Storage>>)>,
    ) -> Result<(WeightKeys, Vec<Arc<Storage>>)> {
        let descriptors = self
            .descriptors
            .ok_or(Error::State("expert copies must complete before commit"))?;
        for ((storage, chunks), target) in self.backing.iter().zip(&self.chunks).zip(&self.targets)
        {
            if target.allocation().is_some() {
                continue;
            }
            for (index, (_, key)) in chunks.iter().enumerate() {
                if let Some(allocation) = storage.physical_allocation(index)? {
                    allocations.insert(*key, allocation);
                }
            }
        }
        for (target, allocations) in self.targets.iter().zip(&descriptors) {
            if target.allocation().is_some() {
                // The distributed owner remaps its complete composite views.
                continue;
            }
            let borrowed = allocations
                .iter()
                .map(|(bytes, fd)| (*bytes, fd))
                .collect::<Vec<_>>();
            target.remap(&borrowed).map_err(|error| {
                Error::Invariant(format!(
                    "expert allocations could not be installed: {error}"
                ))
            })?;
        }
        if let Some((layout, owners)) = resident {
            let staged = self.publication.replace_resident(layout, &owners)?;
            retained.forget(&staged)?;
        }
        self.incoming = self
            .publication
            .manifest
            .storages
            .iter()
            .flat_map(|storage| storage.storage.chunks.iter().map(|chunk| chunk.key))
            .chain([self.keys.manifest])
            .collect();
        let storages = self.publication.put(retained)?;
        let retired = self
            .previous
            .difference(&self.incoming)
            .copied()
            .collect::<Vec<_>>();
        retained.forget(&retired)?;
        allocations.retain(|key, _| self.incoming.contains(key));
        Ok((self.keys, storages))
    }

    /// Release only this preparation's allocations, preserving live aliases.
    pub(crate) fn abort(self, retained: &Retained) -> Result<()> {
        if let Some(receipt) = self.receipt {
            receipt.cancel(retained)?;
        }
        let release = self
            .incoming
            .difference(&self.previous)
            .copied()
            .collect::<Vec<_>>();
        retained.forget(&release)
    }
}
