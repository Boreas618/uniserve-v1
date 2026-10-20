//! The memory a worker keeps at its host's Storage Server.
//!
//! An owner whose contents must outlive the worker process takes its memory
//! here instead of allocating it: the device prefix cache planes, the buffer
//! pool and the latent banks are exclusive storages, one writer at a time,
//! keyed by what determines their shape. A restarted worker of the same
//! scope maps the memory its predecessor left, at whatever address, instead
//! of allocating it again; the server first shoots the predecessor down if
//! it still holds write access. A worker makes no Storage Server call while
//! it serves: only at startup and at its orderly close.
//!
//! The library loaded into another process (tests, harnesses) allocates the
//! same owners locally ([`LocalMemory`]); their memory ends with the
//! process.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use sha2::{Digest, Sha256};
use uniserve_cuda::{DeviceId, Protection};
use uniserve_ipc::StorageLaunch;
use uniserve_storage::protocol::Access;
use uniserve_storage::{Claim, Client, Key, Mode, Placement, Role, Scope, StorageRecord};

use crate::ffi::Backing;
use crate::storage::memory::Storage;
use crate::{Error, Result, lock};

/// One exclusive device storage an owner needs.
#[derive(Debug, Clone, Copy)]
pub struct ExclusiveSpec<'a> {
    /// The storage's name within the worker's scope, such as
    /// `prefix_cache.key.values`.
    pub name: &'a str,
    pub device: DeviceId,
    pub bytes: usize,
    /// Other local devices that address the storage.
    pub access: &'a [DeviceId],
}

/// Where an owner of exclusive device memory takes it.
pub trait ExclusiveMemory {
    /// One storage per spec, in order, mapped read-write. Contents are
    /// whatever the storage last held, or uninitialized: the owner resets
    /// them as it would fresh memory.
    fn exclusive(&mut self, specs: &[ExclusiveSpec<'_>]) -> Result<Vec<Arc<Storage>>>;
}

/// Exclusive memory a process allocates for itself, which ends with it.
pub struct LocalMemory;

impl ExclusiveMemory for LocalMemory {
    fn exclusive(&mut self, specs: &[ExclusiveSpec<'_>]) -> Result<Vec<Arc<Storage>>> {
        specs
            .iter()
            .map(|spec| Storage::exportable(spec.device, spec.bytes, spec.access))
            .collect()
    }
}

/// The worker's connection to its host's Storage Server, shared by the
/// owners that take memory and the capacity planning that counts it. Only
/// the control thread calls it.
pub(crate) struct Retained {
    scope: Scope,
    state: Mutex<State>,
}

struct State {
    client: Client,
    /// The GPU UUID of each device the worker addressed, by ordinal.
    gpus: BTreeMap<i32, [u8; 16]>,
    /// Exclusive tensors the worker holds write access to.
    written: BTreeSet<Key>,
    /// Every tensor the worker uses, which its startup retains.
    used: BTreeSet<Key>,
}

impl State {
    /// The UUID of `device`, which retained keys and records name.
    fn gpu(&mut self, device: DeviceId) -> Result<[u8; 16]> {
        if let Some(uuid) = self.gpus.get(&device.0) {
            return Ok(*uuid);
        }
        let uuid = device.uuid()?.0;
        self.gpus.insert(device.0, uuid);
        Ok(uuid)
    }
}

impl Retained {
    /// Connect as worker `rank` of group `group`, started by the group's
    /// launch `generation`, to the server `launch` names.
    pub(crate) fn connect(
        launch: &StorageLaunch,
        group: &str,
        rank: u32,
        generation: u64,
    ) -> Result<Arc<Self>> {
        let scope = Scope {
            deployment: launch.deployment.clone(),
            group: group.to_owned(),
            rank,
        };
        let client = Client::connect(
            std::path::Path::new(&launch.socket),
            Role::Worker {
                scope: scope.clone(),
                generation,
                claim: Claim(launch.claim),
            },
        )
        .map_err(|error| storage_error("connecting to the storage server", error))?;
        Ok(Arc::new(Self {
            scope,
            state: Mutex::new(State {
                client,
                gpus: BTreeMap::new(),
                written: BTreeSet::new(),
                used: BTreeSet::new(),
            }),
        }))
    }

    /// The key of an exclusive storage: the digest of its scope, name,
    /// layout and GPU. It covers its shape only, so a restart reuses the
    /// memory whatever else changed.
    fn exclusive_key(&self, spec: &ExclusiveSpec<'_>, gpu: &[u8; 16]) -> Key {
        let mut digest = Sha256::new();
        digest.update(b"uniserve-exclusive-storage\0");
        for part in [
            self.scope.deployment.as_bytes(),
            self.scope.group.as_bytes(),
            spec.name.as_bytes(),
        ] {
            digest.update((part.len() as u64).to_le_bytes());
            digest.update(part);
        }
        digest.update(self.scope.rank.to_le_bytes());
        // A byte storage: dtype uint8, shape [bytes], stride 1.
        digest.update(b"uint8");
        digest.update((spec.bytes as u64).to_le_bytes());
        digest.update(1_u64.to_le_bytes());
        digest.update(gpu);
        Key(digest.finalize().into())
    }

    /// The bytes of the scope's retained device storages on `device`,
    /// which NVML attributes to no process: capacity planning counts them
    /// as the worker's own.
    pub(crate) fn scope_bytes(&self, device: DeviceId) -> Result<u64> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(device)?;
        let listing = state
            .client
            .list()
            .map_err(|error| storage_error("listing the storage server", error))?;
        Ok(listing
            .device_bytes
            .iter()
            .filter(|entry| entry.gpu == Some(gpu) && entry.scope == self.scope)
            .map(|entry| entry.bytes)
            .sum())
    }

    /// Release the scope's tensors named like a spec of `specs` whose key
    /// is not the spec's: an owner whose shape changed frees its old memory
    /// before it allocates the new.
    fn release_superseded(&self, state: &mut State, specs: &[(String, Key)]) -> Result<()> {
        let listing = state
            .client
            .list()
            .map_err(|error| storage_error("listing the storage server", error))?;
        let mut keep = Vec::new();
        let mut superseded = 0;
        for tensor in listing
            .tensors
            .iter()
            .filter(|tensor| tensor.scopes.contains(&self.scope))
        {
            let stale = specs
                .iter()
                .any(|(name, key)| *name == tensor.name && *key != tensor.key);
            if stale {
                superseded += 1;
            } else {
                keep.push(tensor.key);
            }
        }
        if superseded > 0 {
            state
                .client
                .retain(keep)
                .map_err(|error| storage_error("releasing superseded storages", error))?;
        }
        Ok(())
    }

    /// Release every tensor of the scope the worker's startup did not use.
    pub(crate) fn retain_used(&self) -> Result<()> {
        let mut state = lock(&self.state);
        let used = state.used.iter().copied().collect();
        let released = state
            .client
            .retain(used)
            .map_err(|error| storage_error("retaining the worker's storages", error))?;
        if released > 0 {
            tracing::info!(
                released,
                "released retained storages the worker no longer uses"
            );
        }
        Ok(())
    }

    /// Give up write access at an orderly close, once the worker drained
    /// its streams: a successor takes the storages without a shoot-down.
    pub(crate) fn release(&self) -> Result<()> {
        let mut state = lock(&self.state);
        if state.written.is_empty() {
            return Ok(());
        }
        let keys: Vec<Key> = std::mem::take(&mut state.written).into_iter().collect();
        state
            .client
            .release(keys)
            .map_err(|error| storage_error("releasing write access", error))
    }
}

impl ExclusiveMemory for &Retained {
    /// Map the specs the scope retains, with write access, and create and
    /// publish the others.
    fn exclusive(&mut self, specs: &[ExclusiveSpec<'_>]) -> Result<Vec<Arc<Storage>>> {
        let retained: &Retained = self;
        let mut state = lock(&retained.state);
        let mut keyed = Vec::with_capacity(specs.len());
        for spec in specs {
            let gpu = state.gpu(spec.device)?;
            keyed.push((
                spec.name.to_owned(),
                retained.exclusive_key(spec, &gpu),
                gpu,
            ));
        }
        let names: Vec<(String, Key)> = keyed
            .iter()
            .map(|(name, key, _)| (name.clone(), *key))
            .collect();
        retained.release_superseded(&mut state, &names)?;

        let got = state
            .client
            .get(names.iter().map(|(_, key)| *key).collect(), Mode::Write)
            .map_err(|error| storage_error("taking exclusive storages", error))?;
        let mut storages: Vec<Option<(StorageRecord, std::os::fd::OwnedFd)>> =
            got.storages.into_iter().map(Some).collect();
        let mut found = BTreeMap::new();
        for tensor in &got.tensors {
            let (record, descriptor) = storages
                .get_mut(tensor.storage as usize)
                .and_then(Option::take)
                .ok_or_else(|| Error::Interface("a retained storage was named twice".into()))?;
            found.insert(tensor.key, (record, descriptor));
        }

        let mut mapped = Vec::with_capacity(specs.len());
        for (spec, (name, key, gpu)) in specs.iter().zip(&keyed) {
            let storage = match found.remove(key) {
                Some((record, descriptor)) => Storage::import(
                    descriptor,
                    record.bytes as usize,
                    spec.device,
                    spec.access,
                    Protection::ReadWrite,
                )?,
                None => {
                    let (storage, descriptor) =
                        Storage::retainable(spec.device, spec.bytes, spec.access)?;
                    let bytes = storage.bytes() as u64;
                    let record = StorageRecord {
                        placement: Placement::Device { gpu: *gpu },
                        bytes,
                    };
                    let tensor = uniserve_storage::TensorRecord {
                        key: *key,
                        name: name.clone(),
                        storage: 0,
                        offset: 0,
                        dtype: "uint8".into(),
                        shape: vec![bytes],
                        strides: vec![1],
                        access: Access::Exclusive,
                    };
                    state
                        .client
                        .put(
                            vec![tensor],
                            &[(record, std::os::fd::AsFd::as_fd(&descriptor))],
                        )
                        .map_err(|error| storage_error("publishing an exclusive storage", error))?;
                    storage
                }
            };
            state.written.insert(*key);
            state.used.insert(*key);
            mapped.push(storage);
        }
        tracing::debug!(
            reused = specs.len() - got.missing.len(),
            created = got.missing.len(),
            "took exclusive storages"
        );
        Ok(mapped)
    }
}

/// A Storage Server failure while the worker was doing `what`.
fn storage_error(what: &str, error: uniserve_storage::Error) -> Error {
    Error::transport_source(what, error)
}
