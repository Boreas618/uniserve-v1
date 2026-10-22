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

use cuda::{DeviceId, Protection};
use ipc::StorageLaunch;
use sha2::{Digest, Sha256};
use storage::protocol::Access;
use storage::{Claim, Client, Key, Mode, Placement, Role, Scope, StorageRecord};

use crate::ffi::Backing;
use crate::memory::allocation::Storage;
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

/// An exclusive storage an owner took.
pub struct Exclusive {
    pub storage: Arc<Storage>,
    /// Whether the memory was created for the owner: its contents are
    /// uninitialized, and the owner resets them. A storage a predecessor
    /// left holds the finite values it last wrote, which its owner keeps:
    /// a standby worker's predecessor still serves from them, and a
    /// successor engine of a handover reads the pages they hold.
    pub created: bool,
}

/// One exclusive device storage of segments, each retained on its own, so
/// that part of its memory is given back and created again at unchanged
/// addresses (`Storage::segmented`).
#[derive(Debug, Clone, Copy)]
pub struct SegmentedSpec<'a> {
    /// The storage's name within the worker's scope; segment `i` is
    /// retained as `{name}#{i}`.
    pub name: &'a str,
    pub device: DeviceId,
    /// Each segment's length, a multiple of the allocation granularity.
    pub segments: &'a [usize],
    /// Other local devices that address the storage.
    pub access: &'a [DeviceId],
}

/// A segmented storage an owner took.
pub struct SegmentedExclusive {
    pub storage: Arc<Storage>,
    /// For each segment, whether it was created for the owner: its contents
    /// are uninitialized.
    pub created: Vec<bool>,
    /// For each segment, whether it is absent: a standby finds none for a
    /// segment its predecessor gave back, and leaves it unmapped.
    pub absent: Vec<bool>,
}

/// Where an owner of exclusive device memory takes it.
pub trait ExclusiveMemory {
    /// One storage per spec, in order, mapped read-write.
    fn exclusive(&mut self, specs: &[ExclusiveSpec<'_>]) -> Result<Vec<Exclusive>>;

    /// A segmented storage of `spec`, its segments mapped read-write.
    fn segmented(&mut self, spec: &SegmentedSpec<'_>) -> Result<SegmentedExclusive>;

    /// Gives back the memory of `segments` of the storage `spec` took: they
    /// are unmapped here at once, and their memory is freed once no other
    /// process maps it. No device work may access them any more.
    fn release_segments(
        &mut self,
        spec: &SegmentedSpec<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()>;

    /// Creates `segments` of the storage `spec` took again, at their
    /// addresses; their contents are uninitialized.
    fn restore_segments(
        &mut self,
        spec: &SegmentedSpec<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()>;
}

/// The device bytes a worker's scope retains on one GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScopeBytes {
    /// Storages the worker uses: its weights and the exclusive storages it
    /// took.
    pub held: u64,
    /// Exclusive storages a predecessor left that the worker has not taken
    /// yet: its owners take them again, or release them when their shape
    /// changed, before allocating their own memory. A storage no owner
    /// asks for any more is released once the worker started.
    pub reclaimable: u64,
}

/// Exclusive memory a process allocates for itself, which ends with it.
pub struct LocalMemory;

impl ExclusiveMemory for LocalMemory {
    fn exclusive(&mut self, specs: &[ExclusiveSpec<'_>]) -> Result<Vec<Exclusive>> {
        specs
            .iter()
            .map(|spec| {
                Ok(Exclusive {
                    storage: Storage::exportable(spec.device, spec.bytes, spec.access)?,
                    created: true,
                })
            })
            .collect()
    }

    fn segmented(&mut self, spec: &SegmentedSpec<'_>) -> Result<SegmentedExclusive> {
        let storage = Storage::segmented(spec.device, spec.segments, spec.access)?;
        for index in 0..spec.segments.len() {
            storage.create_segment(index)?;
        }
        Ok(SegmentedExclusive {
            storage,
            created: vec![true; spec.segments.len()],
            absent: vec![false; spec.segments.len()],
        })
    }

    fn release_segments(
        &mut self,
        _spec: &SegmentedSpec<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()> {
        for index in segments {
            storage.release_segment(*index)?;
        }
        Ok(())
    }

    fn restore_segments(
        &mut self,
        _spec: &SegmentedSpec<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()> {
        for index in segments {
            storage.create_segment(*index)?;
        }
        Ok(())
    }
}

/// The worker's connection to its host's Storage Server, shared by the
/// owners that take memory and the capacity planning that counts it. Only
/// the control thread calls it.
pub(crate) struct Retained {
    scope: Scope,
    state: Mutex<State>,
}

/// A standby worker's exclusive storages: their keys, which promotion
/// takes write access to, and their mappings, read-only while it parks.
#[derive(Default)]
struct Standby {
    keys: Vec<Key>,
    storages: Vec<Arc<Storage>>,
}

struct State {
    client: Client,
    /// The GPU UUID of each device the worker addressed, by ordinal.
    gpus: BTreeMap<i32, [u8; 16]>,
    /// Exclusive tensors the worker holds write access to.
    written: BTreeSet<Key>,
    /// Every tensor the worker uses, which its startup retains.
    used: BTreeSet<Key>,
    /// A standby worker's exclusive storages, until it is promoted; `None`
    /// for a worker that takes write access at once.
    standby: Option<Standby>,
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
    /// launch `generation`, to the server `launch` names. A standby worker
    /// (`StorageLaunch::standby`) maps exclusive storages without taking
    /// write access.
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
                standby: launch.standby.then(Standby::default),
            }),
        }))
    }

    /// The key of an exclusive storage: the digest of its scope, name,
    /// layout and GPU. It covers its shape only, so a restart reuses the
    /// memory whatever else changed.
    fn exclusive_key(&self, spec: &ExclusiveSpec<'_>, gpu: &[u8; 16]) -> Key {
        self.storage_key(spec.name, spec.bytes, gpu)
    }

    /// The key of segment `index` of a segmented storage, as the exclusive
    /// storage `{name}#{index}` of the segment's length.
    fn segment_key(&self, spec: &SegmentedSpec<'_>, index: usize, gpu: &[u8; 16]) -> Key {
        self.storage_key(&format!("{}#{index}", spec.name), spec.segments[index], gpu)
    }

    fn storage_key(&self, name: &str, bytes: usize, gpu: &[u8; 16]) -> Key {
        let mut digest = Sha256::new();
        digest.update(b"uniserve-exclusive-storage\0");
        for part in [
            self.scope.deployment.as_bytes(),
            self.scope.group.as_bytes(),
            name.as_bytes(),
        ] {
            digest.update((part.len() as u64).to_le_bytes());
            digest.update(part);
        }
        digest.update(self.scope.rank.to_le_bytes());
        // A byte storage: dtype uint8, shape [bytes], stride 1.
        digest.update(b"uint8");
        digest.update((bytes as u64).to_le_bytes());
        digest.update(1_u64.to_le_bytes());
        digest.update(gpu);
        Key(digest.finalize().into())
    }

    /// Publishes the memory of segment `index`, which `descriptor` maps, as
    /// an exclusive tensor the worker writes.
    fn publish_segment(
        state: &mut State,
        spec: &SegmentedSpec<'_>,
        index: usize,
        key: Key,
        gpu: [u8; 16],
        descriptor: &std::os::fd::OwnedFd,
    ) -> Result<()> {
        let bytes = spec.segments[index] as u64;
        let record = StorageRecord {
            placement: Placement::Device { gpu },
            bytes,
        };
        let tensor = storage::TensorRecord {
            key,
            name: format!("{}#{index}", spec.name),
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
                &[(record, std::os::fd::AsFd::as_fd(descriptor))],
            )
            .map_err(|error| storage_error("publishing a segment", error))
    }

    /// The UUID of `device`, which retained keys and records name.
    pub(crate) fn gpu(&self, device: DeviceId) -> Result<[u8; 16]> {
        lock(&self.state).gpu(device)
    }

    /// Make one call on the connection, as the worker doing `what`.
    pub(crate) fn call<T>(
        &self,
        what: &str,
        request: impl FnOnce(&mut Client) -> storage::Result<T>,
    ) -> Result<T> {
        request(&mut lock(&self.state).client).map_err(|error| storage_error(what, error))
    }

    /// Note tensors the worker uses, which its startup retains.
    pub(crate) fn use_keys(&self, keys: impl IntoIterator<Item = Key>) {
        lock(&self.state).used.extend(keys);
    }

    /// Move the worker's read-only tensors to new keys, each `(old, new)`,
    /// as their bytes were rewritten in place (`Request::Rekey`).
    pub(crate) fn rekey(&self, moves: Vec<(Key, Key)>) -> Result<()> {
        if moves.is_empty() {
            return Ok(());
        }
        let mut state = lock(&self.state);
        state
            .client
            .rekey(moves.clone())
            .map_err(|error| storage_error("renaming retained weights", error))?;
        for (old, new) in moves {
            if state.used.remove(&old) {
                state.used.insert(new);
            }
        }
        Ok(())
    }

    /// Release the scope's read-only tensors the worker does not bind
    /// before it loads the rest of its weights, so the memory of weights it
    /// no longer binds, such as another checkpoint's of the same model,
    /// returns before loading needs it. Exclusive storages stay.
    pub(crate) fn release_read_only(&self) -> Result<()> {
        let mut state = lock(&self.state);
        let listing = state
            .client
            .list()
            .map_err(|error| storage_error("listing the storage server", error))?;
        let keep: Vec<Key> = listing
            .tensors
            .iter()
            .filter(|tensor| {
                tensor.scopes.contains(&self.scope)
                    && (tensor.access != Access::ReadOnly || state.used.contains(&tensor.key))
            })
            .map(|tensor| tensor.key)
            .collect();
        let released = state
            .client
            .retain(keep)
            .map_err(|error| storage_error("releasing retained weights", error))?;
        if released > 0 {
            tracing::info!(released, "released retained weights before loading");
        }
        Ok(())
    }

    /// The scope's retained device bytes on `device`, which NVML attributes
    /// to no process: capacity planning counts the held ones as the
    /// worker's own and the reclaimable ones as free.
    pub(crate) fn scope_bytes(&self, device: DeviceId) -> Result<ScopeBytes> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(device)?;
        let listing = state
            .client
            .list()
            .map_err(|error| storage_error("listing the storage server", error))?;
        let total: u64 = listing
            .device_bytes
            .iter()
            .filter(|entry| entry.gpu == Some(gpu) && entry.scope == self.scope)
            .map(|entry| entry.bytes)
            .sum();
        // Every exclusive tensor is a storage of its own.
        let reclaimable: u64 = listing
            .tensors
            .iter()
            .filter(|tensor| {
                tensor.access == Access::Exclusive
                    && tensor.placement == Placement::Device { gpu }
                    && tensor.scopes.contains(&self.scope)
                    && !state.written.contains(&tensor.key)
            })
            .map(|tensor| tensor.storage_bytes)
            .sum();
        Ok(ScopeBytes {
            held: total.saturating_sub(reclaimable),
            reclaimable,
        })
    }

    /// Release the exclusive storages a predecessor left that no owner of
    /// this worker took and that `taking` does not take now. An owner that
    /// allocates new memory calls this first: capacity planning counted
    /// those storages as free (`ScopeBytes::reclaimable`), and an owner
    /// whose shape changed, or that this worker no longer has, leaves them
    /// to no one.
    fn release_untaken(&self, state: &mut State, taking: &BTreeSet<Key>) -> Result<()> {
        let listing = state
            .client
            .list()
            .map_err(|error| storage_error("listing the storage server", error))?;
        let mut keep = Vec::new();
        let mut untaken = 0;
        for tensor in listing
            .tensors
            .iter()
            .filter(|tensor| tensor.scopes.contains(&self.scope))
        {
            if tensor.access == Access::Exclusive
                && !state.written.contains(&tensor.key)
                && !taking.contains(&tensor.key)
            {
                untaken += 1;
            } else {
                keep.push(tensor.key);
            }
        }
        if untaken > 0 {
            state
                .client
                .retain(keep)
                .map_err(|error| storage_error("releasing untaken storages", error))?;
            tracing::info!(
                released = untaken,
                "released exclusive storages no owner takes again"
            );
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

    /// Park a standby worker once it warmed up: its exclusive storages
    /// become read-only, so it writes none of them until it is promoted.
    pub(crate) fn park(&self) -> Result<()> {
        let state = lock(&self.state);
        let Some(standby) = &state.standby else {
            return Ok(());
        };
        for storage in &standby.storages {
            storage.protect(Protection::ReadOnly)?;
        }
        tracing::info!(
            storages = standby.storages.len(),
            "the standby worker parked"
        );
        Ok(())
    }

    /// Promote a parked standby worker: take write access to its exclusive
    /// storages once their previous writer is out (released, or shot down
    /// by the server), and make the mappings writable at unchanged
    /// addresses, which its captured graphs address. Fails for a worker
    /// that is not a standby.
    pub(crate) fn promote(&self) -> Result<()> {
        let mut state = lock(&self.state);
        let standby = state
            .standby
            .take()
            .ok_or_else(|| Error::State("only a standby worker is promoted"))?;
        state
            .client
            .promote(standby.keys.clone())
            .map_err(|error| storage_error("promoting the standby worker", error))?;
        for storage in &standby.storages {
            storage.protect(Protection::ReadWrite)?;
        }
        state.written.extend(standby.keys.iter().copied());
        tracing::info!(
            storages = standby.storages.len(),
            "the standby worker was promoted"
        );
        Ok(())
    }

    /// Whether the worker stands by, not yet promoted.
    pub(crate) fn standing_by(&self) -> bool {
        lock(&self.state).standby.is_some()
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
    fn exclusive(&mut self, specs: &[ExclusiveSpec<'_>]) -> Result<Vec<Exclusive>> {
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
        let keys: Vec<Key> = keyed.iter().map(|(_, key, _)| *key).collect();
        // A standby maps what its predecessor holds, and creates nothing.
        let standby = state.standby.is_some();
        let mode = if standby { Mode::Standby } else { Mode::Write };
        let got = state
            .client
            .get(keys, mode)
            .map_err(|error| storage_error("taking exclusive storages", error))?;
        if standby && let Some(missing) = got.missing.first() {
            let name = keyed
                .iter()
                .find(|(_, key, _)| key == missing)
                .map_or("an exclusive storage", |(name, _, _)| name.as_str());
            return Err(Error::Invalid(format!(
                "the standby worker found no {name} of its predecessor's shape"
            )));
        }
        if !got.missing.is_empty() {
            let taking = got.tensors.iter().map(|tensor| tensor.key).collect();
            retained.release_untaken(&mut state, &taking)?;
        }
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
            let created = !found.contains_key(key);
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
                    let tensor = storage::TensorRecord {
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
            state.used.insert(*key);
            match &mut state.standby {
                Some(standby) => {
                    standby.keys.push(*key);
                    standby.storages.push(Arc::clone(&storage));
                }
                None => {
                    state.written.insert(*key);
                }
            }
            mapped.push(Exclusive { storage, created });
        }
        tracing::debug!(
            reused = specs.len() - got.missing.len(),
            created = got.missing.len(),
            "took exclusive storages"
        );
        Ok(mapped)
    }

    fn segmented(&mut self, spec: &SegmentedSpec<'_>) -> Result<SegmentedExclusive> {
        self.take_segmented(spec)
    }

    fn release_segments(
        &mut self,
        spec: &SegmentedSpec<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()> {
        self.give_back_segments(spec, storage, segments)
    }

    fn restore_segments(
        &mut self,
        spec: &SegmentedSpec<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()> {
        self.create_segments(spec, storage, segments)
    }
}

impl Retained {
    /// Takes the segments of a segmented storage: as [`ExclusiveMemory::exclusive`]
    /// takes whole ones, except that a standby leaves a segment its
    /// predecessor gave back absent.
    fn take_segmented(&self, spec: &SegmentedSpec<'_>) -> Result<SegmentedExclusive> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(spec.device)?;
        let keys: Vec<Key> = (0..spec.segments.len())
            .map(|index| self.segment_key(spec, index, &gpu))
            .collect();
        let standby = state.standby.is_some();
        let mode = if standby { Mode::Standby } else { Mode::Write };
        let got = state
            .client
            .get(keys.clone(), mode)
            .map_err(|error| storage_error("taking a segmented storage", error))?;
        if !standby && !got.missing.is_empty() {
            let taking = got.tensors.iter().map(|tensor| tensor.key).collect();
            self.release_untaken(&mut state, &taking)?;
        }
        let mut storages: Vec<Option<(StorageRecord, std::os::fd::OwnedFd)>> =
            got.storages.into_iter().map(Some).collect();
        let mut found = BTreeMap::new();
        for tensor in &got.tensors {
            let (record, descriptor) = storages
                .get_mut(tensor.storage as usize)
                .and_then(Option::take)
                .ok_or_else(|| Error::Interface("a retained storage was named twice".into()))?;
            if record.bytes as usize
                != spec.segments[keys
                    .iter()
                    .position(|key| *key == tensor.key)
                    .ok_or_else(|| Error::Interface("an unasked segment was found".into()))?]
            {
                return Err(Error::Interface(
                    "a retained segment has another length".into(),
                ));
            }
            found.insert(tensor.key, descriptor);
        }
        let storage = Storage::segmented(spec.device, spec.segments, spec.access)?;
        let mut created = vec![false; keys.len()];
        let mut absent = vec![false; keys.len()];
        for (index, key) in keys.iter().enumerate() {
            match found.remove(key) {
                Some(descriptor) => storage.import_segment(index, &descriptor)?,
                None if standby => {
                    absent[index] = true;
                    continue;
                }
                None => {
                    let descriptor = storage.create_segment(index)?;
                    Self::publish_segment(&mut state, spec, index, *key, gpu, &descriptor)?;
                    created[index] = true;
                }
            }
            state.used.insert(*key);
            match &mut state.standby {
                Some(standby) => standby.keys.push(*key),
                None => {
                    state.written.insert(*key);
                }
            }
        }
        if let Some(standby) = &mut state.standby {
            standby.storages.push(Arc::clone(&storage));
        }
        Ok(SegmentedExclusive {
            storage,
            created,
            absent,
        })
    }

    /// Unmaps `segments` and has the Storage Server drop them.
    fn give_back_segments(
        &self,
        spec: &SegmentedSpec<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(spec.device)?;
        for index in segments {
            storage.release_segment(*index)?;
            let key = self.segment_key(spec, *index, &gpu);
            state.used.remove(&key);
            state.written.remove(&key);
        }
        let used: Vec<Key> = state.used.iter().copied().collect();
        state
            .client
            .retain(used)
            .map_err(|error| storage_error("giving back segments", error))?;
        Ok(())
    }

    /// Creates `segments` again and publishes them.
    fn create_segments(
        &self,
        spec: &SegmentedSpec<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(spec.device)?;
        for index in segments {
            let key = self.segment_key(spec, *index, &gpu);
            let descriptor = storage.create_segment(*index)?;
            Self::publish_segment(&mut state, spec, *index, key, gpu, &descriptor)?;
            state.used.insert(key);
            state.written.insert(key);
        }
        Ok(())
    }
}

/// A Storage Server failure while the worker was doing `what`.
fn storage_error(what: &str, error: storage::Error) -> Error {
    Error::transport_source(what, error)
}
