//! The rank's client of its host's Storage Server.
//!
//! Everything a rank would lose by exiting, and that costs time or memory to rebuild, is storage
//! the host's Storage Server retains: the weights, which every process that maps them shares
//! read-only, and the exclusive storages of the rank's scope (the cache planes, the memory of a
//! buffer pool region). A [`StorageClient`] is the rank's one connection to that server. The
//! owners of exclusive storage take it through the client ([`ExclusiveStorage`]), the weight
//! store publishes and maps weights through it
//! ([`WeightStore`](crate::memory::weights::WeightStore)), and the rank's share of its GPU's
//! quota is stated and read through it.
//!
//! ## A region's storage across processes and epochs
//!
//! A cache plane's storage is keyed by its region's memory and what fixes its shape: a segment's
//! part by the segment the region's segment table names (the epoch that created it and its
//! ordinal), and a storage that is not split into segments by the epoch of the table's first
//! segment ([`ExclusiveConfig::epoch`]). Every epoch that names the region, a standby and a
//! restarted engine of the same scope map the same storages, writable, with an access that
//! records no writer (`Mode::Region`): the buffer pool, not a writer, keeps the blocks they
//! write apart, so nothing of the cache is copied or handed over between them.
//!
//! A rank makes no Storage Server call while it serves: only at startup, when the segments of
//! its planes are given back or created again, and at its orderly close.

use std::collections::{BTreeMap, BTreeSet};
use std::os::fd::{AsFd, OwnedFd};
use std::sync::{Arc, Mutex};

use ipc::StorageLaunch;
use platform::{DeviceId, Protection};
use storage::protocol::{Access, ErrorKind, TensorListing};
use storage::{
    Claim, Client, HostStorage, Item, Key, Location, Mode, Quota, Role, Scope, Segment,
    StorageRecord, TensorRecord,
};

use crate::memory::allocation::Storage;
use crate::memory::exclusive::{ExclusiveConfig, ExclusiveStorage, SegmentPart, SegmentedConfig};
use crate::{Error, Result, lock};

/// A deployment's limit of a GPU of `total` bytes: `fraction` of the whole of it
/// (`mem_fraction_static`), rounded down, as the Storage Server's quota states it.
pub fn quota_limit(total: u64, fraction: f64) -> u64 {
    (total as f64 * fraction) as u64
}

/// The rank's connection to its host's Storage Server: what the rank retains there, and its
/// share of each GPU's quota.
///
/// One client serves every owner of the rank, behind one lock, so two owners' requests never
/// interleave on the connection.
pub struct StorageClient {
    scope: Scope,
    /// The epoch the rank belongs to, which retains what the rank uses and keys the storages
    /// of no segmented region.
    epoch: u64,
    state: Mutex<State>,
}

/// What the connection holds between requests.
struct State {
    client: Client,
    /// The GPU UUID of each device the rank addressed, by ordinal.
    gpus: BTreeMap<i32, [u8; 16]>,
    /// The exclusive storages the rank took and maps.
    taken: BTreeSet<Key>,
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

    /// Where the rank's retained storage lies: the GPUs it addressed, and host storage. A key
    /// has a copy at each location that holds it, so the rank retains and releases the copies
    /// at its own locations and leaves other ranks' copies to them.
    fn locations(&self) -> Vec<Location> {
        self.gpus
            .values()
            .map(|gpu| Location::Device { gpu: *gpu })
            .chain(std::iter::once(Location::Host { numa_node: None }))
            .collect()
    }
}

impl StorageClient {
    /// Connects as rank `rank` of worker group `group`, started by the group's launch
    /// `generation`, to the server `launch` names.
    ///
    /// # Errors
    ///
    /// When the server cannot be reached or refuses the rank.
    pub fn connect(
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
                epoch: launch.epoch,
            },
        )
        .map_err(|error| storage_error("connecting to the storage server", error))?;

        Ok(Arc::new(Self {
            scope,
            epoch: launch.epoch,
            state: Mutex::new(State {
                client,
                gpus: BTreeMap::new(),
                taken: BTreeSet::new(),
            }),
        }))
    }

    /// The rank's scope, which its epoch's quota reservations name.
    pub fn scope(&self) -> &Scope {
        &self.scope
    }

    /// The rank's epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Makes one request on the connection, as the rank doing `what`.
    ///
    /// # Errors
    ///
    /// The server's refusal or the connection's failure, stated with `what`.
    pub fn call<T>(
        &self,
        what: &str,
        request: impl FnOnce(&mut Client) -> storage::Result<T>,
    ) -> Result<T> {
        request(&mut lock(&self.state).client).map_err(|error| storage_error(what, error))
    }

    /// The UUID of `device`, which the keys and records of its storage name.
    pub(super) fn gpu(&self, device: DeviceId) -> Result<[u8; 16]> {
        lock(&self.state).gpu(device)
    }

    /// Ends the epoch's retention of `keys` at the rank's locations, as the rank doing `doing`.
    pub(super) fn end_retention(&self, keys: &BTreeSet<Key>, doing: &str) -> Result<()> {
        self.release_keys(&mut lock(&self.state), keys, doing)
    }

    /// The key of an exclusive storage of the rank: the digest of its scope, the epoch of its
    /// region's memory (the rank's own for a storage of no segmented region), its name, its
    /// length and its GPU. It covers the storage's shape only, so every epoch that names the
    /// region, and a restart of the epoch, maps the storage again whatever else changed.
    fn exclusive_key(&self, config: &ExclusiveConfig<'_>, gpu: &[u8; 16]) -> Key {
        storage::key::exclusive(
            &self.scope,
            config.name,
            config.bytes as u64,
            gpu,
            config.epoch.unwrap_or(self.epoch),
        )
    }

    /// The key of one part of a segmented storage, as the exclusive storage
    /// `{name}#{column}/{ordinal}` of the part's length that the segment's epoch created.
    fn part_key(&self, name: &str, part: &SegmentPart, gpu: &[u8; 16]) -> Key {
        storage::key::exclusive(
            &self.scope,
            &segment_name(name, part),
            part.length as u64,
            gpu,
            part.segment.epoch,
        )
    }

    /// The scope's retained device bytes on `device`, which NVML attributes to no process: the
    /// bytes of the storages the rank uses (its weights and the exclusive storages it took),
    /// and the bytes of the exclusive storages a predecessor left that the rank has not taken.
    ///
    /// The owners take the second again, or release them when their shape changed, before
    /// allocating storage of their own, so a rank sizing its memory counts the first as its
    /// own and the second as free.
    pub fn scope_bytes(&self, device: DeviceId) -> Result<(u64, u64)> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(device)?;
        let listing = state
            .client
            .list()
            .map_err(|error| storage_error("listing the storage server", error))?;

        let total: u64 = listing
            .device_bytes
            .iter()
            .filter(|entry| {
                entry.gpu == Some(gpu) && entry.scope == self.scope && entry.epoch == self.epoch
            })
            .map(|entry| entry.bytes)
            .sum();

        // Every exclusive tensor is a storage of its own. Those another epoch uses, as the
        // storages of a region two epochs name, are not free.
        let untaken: u64 = listing
            .tensors
            .iter()
            .filter(|tensor| {
                tensor.access == Access::Exclusive
                    && tensor.location == Location::Device { gpu }
                    && self.retains(tensor)
                    && tensor
                        .retainers
                        .iter()
                        .all(|retainer| retainer.epoch == self.epoch)
                    && !state.taken.contains(&tensor.key)
            })
            .map(|tensor| tensor.storage_bytes)
            .sum();

        Ok((total.saturating_sub(untaken), untaken))
    }

    /// Releases the exclusive storages a predecessor left that no owner of this rank took and
    /// that `taking` does not take now. An owner that allocates new storage calls this first:
    /// [`Self::scope_bytes`] counted those storages as free, and an owner whose shape changed,
    /// or that this rank does not have, leaves them to no one.
    fn release_untaken(&self, state: &mut State, taking: &BTreeSet<Key>) -> Result<()> {
        let listing = state
            .client
            .list()
            .map_err(|error| storage_error("listing the storage server", error))?;

        let mut keep = Vec::new();
        let mut untaken = 0;
        for tensor in listing.tensors.iter().filter(|tensor| self.retains(tensor)) {
            if tensor.access == Access::Exclusive
                && !state.taken.contains(&tensor.key)
                && !taking.contains(&tensor.key)
            {
                untaken += 1;
            } else {
                keep.push(tensor.key);
            }
        }

        if untaken > 0 {
            let at = state.locations();
            state
                .client
                .retain(keep, at)
                .map_err(|error| storage_error("releasing untaken storages", error))?;
            tracing::info!(
                released = untaken,
                "released exclusive storages no owner takes again"
            );
        }
        Ok(())
    }

    /// Ends the epoch's retention of `keys` alone. The server's retention is the scope's in the
    /// epoch, which every worker of the scope shares, and the rank's owners take their storages
    /// one after another in its startup, so every other storage the scope retains stays.
    fn release_keys(&self, state: &mut State, keys: &BTreeSet<Key>, doing: &str) -> Result<()> {
        if keys.is_empty() {
            return Ok(());
        }

        let listing = state
            .client
            .list()
            .map_err(|error| storage_error("listing the storage server", error))?;
        let keep: Vec<Key> = listing
            .tensors
            .iter()
            .filter(|tensor| self.retains(tensor) && !keys.contains(&tensor.key))
            .map(|tensor| tensor.key)
            .collect();

        let at = state.locations();
        state
            .client
            .retain(keep, at)
            .map_err(|error| storage_error(doing, error))?;
        Ok(())
    }

    /// Whether the rank's scope retains `tensor` in the rank's epoch.
    fn retains(&self, tensor: &TensorListing) -> bool {
        tensor
            .retainers
            .iter()
            .any(|retainer| retainer.scope == self.scope && retainer.epoch == self.epoch)
    }

    /// States this epoch's quota of the GPU of `device`, `fraction` of the whole of it: every
    /// storage retained there from now on fits it or is refused, and the cache's segments are
    /// given back against it. What this rank holds already stays.
    pub fn state_quota(&self, device: DeviceId, fraction: f64) -> Result<()> {
        let total = device.memory_info()?.total;

        self.reserve(device, total, quota_limit(total, fraction), Vec::new())
            .map(drop)
    }

    /// The deployment's quota of the GPU of `device`, and what the host admits of new host
    /// storage.
    pub fn usage(&self, device: DeviceId) -> Result<(Quota, HostStorage)> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(device)?;
        let usage = state
            .client
            .usage(vec![gpu])
            .map_err(|error| storage_error("reading the quota", error))?;

        Ok((self.of_deployment(gpu, usage.quotas), usage.host))
    }

    /// Reserves `items` of the rank on the GPU of `device` of `total` bytes, stating the
    /// epoch's limit `quota` there, and returns the deployment's quota of the GPU after it.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] with the server's refusal, which names every reservation of the
    /// quota, when the reservation would take the GPU's usage past its limit.
    pub fn reserve(
        &self,
        device: DeviceId,
        total: u64,
        quota: u64,
        items: Vec<(Item, u64)>,
    ) -> Result<Quota> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(device)?;
        let quotas = state
            .client
            .reserve(gpu, total, Some(quota), items)
            .map_err(|error| match error {
                storage::Error::Server {
                    kind: ErrorKind::Quota,
                    message,
                } => Error::Unsupported(format!(
                    "the device storage the rank holds on {device} does not fit its quota: \
                     {message}"
                )),
                error => storage_error("reserving device storage", error),
            })?;

        Ok(self.of_deployment(gpu, quotas))
    }

    /// The quota of the rank's deployment among `quotas` of `gpu`; an empty one when the
    /// server keeps none yet.
    fn of_deployment(&self, gpu: [u8; 16], quotas: Vec<Quota>) -> Quota {
        quotas
            .into_iter()
            .find(|quota| quota.deployment == self.scope.deployment)
            .unwrap_or(Quota {
                gpu,
                deployment: self.scope.deployment.clone(),
                total: 0,
                limit: None,
                reservations: Vec::new(),
                host_link_bytes_per_second: None,
                transfer_context: 0,
            })
    }

    /// Publishes the storage of part `index` of `config`, which `descriptor` maps, as an
    /// exclusive tensor of the rank's scope.
    fn publish_segment(
        state: &mut State,
        config: &SegmentedConfig<'_>,
        index: usize,
        key: Key,
        gpu: [u8; 16],
        descriptor: &OwnedFd,
    ) -> Result<()> {
        let part = &config.parts[index];
        let bytes = part.length as u64;

        let record = StorageRecord {
            location: Location::Device { gpu },
            bytes,
            // The quota reserves the storages of one segment together, as the cache's.
            segment: Some(Segment {
                epoch: part.segment.epoch,
                ordinal: part.segment.ordinal,
            }),
            fabric: None,
        };
        let tensor = byte_tensor(
            key,
            segment_name(config.name, part),
            0,
            bytes,
            Access::Exclusive,
        );

        state
            .client
            .put(vec![tensor], &[(record, descriptor.as_fd())])
            .map_err(|error| storage_error("publishing a segment", error))
    }
}

/// The storages of a cache's planes are the memory of a buffer pool region: they are taken
/// with `Mode::Region`, which records no writer and shoots no other process down, so every
/// epoch that names the region maps them writable at once (`specs/buffer-pool.md` §14.1).
impl ExclusiveStorage for StorageClient {
    /// Maps the storages the server holds under the configs' keys, and creates and publishes
    /// the others.
    fn exclusive(&self, configs: &[ExclusiveConfig<'_>]) -> Result<Vec<(Arc<Storage>, bool)>> {
        let mut state = lock(&self.state);
        let state = &mut *state;

        let mut keyed = Vec::with_capacity(configs.len());
        for config in configs {
            let gpu = state.gpu(config.device)?;
            keyed.push((self.exclusive_key(config, &gpu), gpu));
        }
        let keys: Vec<Key> = keyed.iter().map(|(key, _)| *key).collect();

        let (got, descriptors) = state
            .client
            .get(keys, Mode::Region, Vec::new())
            .map_err(|error| storage_error("taking exclusive storages", error))?;
        if !got.missing.is_empty() {
            let taking = got.tensors.iter().map(|tensor| tensor.key).collect();
            self.release_untaken(state, &taking)?;
        }

        let mut storages: Vec<Option<(StorageRecord, OwnedFd)>> = got
            .storages
            .into_iter()
            .zip(descriptors)
            .map(Some)
            .collect();
        let mut found = BTreeMap::new();
        for tensor in &got.tensors {
            let storage = storages
                .get_mut(tensor.storage as usize)
                .and_then(Option::take)
                .ok_or_else(|| Error::Interface("a retained storage was named twice".into()))?;
            found.insert(tensor.key, storage);
        }

        let mut taken = Vec::with_capacity(configs.len());
        for (config, (key, gpu)) in configs.iter().zip(&keyed) {
            let (storage, created) = match found.remove(key) {
                Some((record, descriptor)) => {
                    let storage = match record.location {
                        Location::Host { .. } => Storage::map_host(
                            descriptor,
                            record.bytes as usize,
                            config.device,
                            true,
                        )?,
                        Location::Device { .. } => Storage::map_device(
                            &descriptor,
                            record.bytes as usize,
                            config.device,
                            config.access,
                            Protection::ReadWrite,
                        )?,
                    };
                    (storage, false)
                }

                None => {
                    let (storage, descriptor) =
                        Storage::retained(config.device, config.bytes, config.access)?;
                    let bytes = storage.bytes() as u64;
                    let record = StorageRecord {
                        location: Location::Device { gpu: *gpu },
                        segment: None,
                        bytes,
                        fabric: None,
                    };
                    let tensor =
                        byte_tensor(*key, config.name.to_owned(), 0, bytes, Access::Exclusive);

                    state
                        .client
                        .put(vec![tensor], &[(record, descriptor.as_fd())])
                        .map_err(|error| storage_error("publishing an exclusive storage", error))?;
                    (storage, true)
                }
            };

            state.taken.insert(*key);
            taken.push((storage, created));
        }

        tracing::debug!(
            mapped = configs.len() - got.missing.len(),
            created = got.missing.len(),
            "took exclusive storages"
        );
        Ok(taken)
    }

    fn key(&self, config: &ExclusiveConfig<'_>) -> Result<Option<(Key, [u8; 16])>> {
        let gpu = self.gpu(config.device)?;

        Ok(Some((self.exclusive_key(config, &gpu), gpu)))
    }

    /// Takes the parts of a segmented storage as [`Self::exclusive`] takes whole storages,
    /// except that a missing part whose `create` is unset (a segment given back) is left
    /// unmapped.
    fn segmented(&self, config: &SegmentedConfig<'_>) -> Result<(Arc<Storage>, Vec<bool>)> {
        let mut state = lock(&self.state);
        let state = &mut *state;

        let gpu = state.gpu(config.device)?;
        let keys: Vec<Key> = config
            .parts
            .iter()
            .map(|part| self.part_key(config.name, part, &gpu))
            .collect();

        let (got, descriptors) = state
            .client
            .get(keys.clone(), Mode::Region, Vec::new())
            .map_err(|error| storage_error("taking a segmented storage", error))?;

        // The storages a predecessor left that no owner takes again are released before a
        // missing part is created in their place. Parts left unmapped (segments given back)
        // create nothing, and the storages not taken yet include other planes' segments, which
        // their owner takes next: the value planes' after the key planes'.
        let creates = got.missing.iter().any(|missing| {
            keys.iter()
                .position(|key| key == missing)
                .is_some_and(|index| config.parts[index].create)
        });
        if creates {
            let taking = got.tensors.iter().map(|tensor| tensor.key).collect();
            self.release_untaken(state, &taking)?;
        }

        let mut storages: Vec<Option<(StorageRecord, OwnedFd)>> = got
            .storages
            .into_iter()
            .zip(descriptors)
            .map(Some)
            .collect();
        let mut found = BTreeMap::new();
        for tensor in &got.tensors {
            let (record, descriptor) = storages
                .get_mut(tensor.storage as usize)
                .and_then(Option::take)
                .ok_or_else(|| Error::Interface("a retained storage was named twice".into()))?;
            let index = keys
                .iter()
                .position(|key| *key == tensor.key)
                .ok_or_else(|| Error::Interface("an unasked segment was found".into()))?;

            if record.bytes as usize != config.parts[index].length {
                return Err(Error::Interface(
                    "a retained segment has another length".into(),
                ));
            }
            found.insert(tensor.key, descriptor);
        }

        let storage = Storage::segmented(config.device, &config.lengths(), config.access)?;
        let mut created = vec![false; keys.len()];
        for (index, key) in keys.iter().enumerate() {
            match found.remove(key) {
                Some(descriptor) => storage.import_segment(index, &descriptor)?,
                // A part nobody holds stays unmapped when it was given back.
                None if !config.parts[index].create => continue,
                None => {
                    let descriptor = storage.create_segment(index)?;
                    Self::publish_segment(state, config, index, *key, gpu, &descriptor)?;
                    created[index] = true;
                }
            }
            state.taken.insert(*key);
        }

        Ok((storage, created))
    }

    fn segment_key(
        &self,
        config: &SegmentedConfig<'_>,
        index: usize,
    ) -> Result<Option<(Key, [u8; 16])>> {
        let part = config
            .parts
            .get(index)
            .ok_or_else(|| Error::Invalid(format!("{} has no part {index}", config.name)))?;
        let gpu = self.gpu(config.device)?;

        Ok(Some((self.part_key(config.name, part, &gpu), gpu)))
    }

    /// Unmaps the parts `segments` and has the Storage Server drop them.
    fn release_segments(
        &self,
        config: &SegmentedConfig<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(config.device)?;

        let mut given = BTreeSet::new();
        for &index in segments {
            storage.release_segment(index)?;

            let key = self.part_key(config.name, &config.parts[index], &gpu);
            state.taken.remove(&key);
            given.insert(key);
        }

        self.release_keys(&mut state, &given, "giving back segments")
    }

    /// Creates the parts `segments` again, as segments of the rank's own, and publishes them.
    fn restore_segments(
        &self,
        config: &SegmentedConfig<'_>,
        storage: &Storage,
        segments: &[usize],
    ) -> Result<()> {
        let mut state = lock(&self.state);
        let gpu = state.gpu(config.device)?;

        for &index in segments {
            let key = self.part_key(config.name, &config.parts[index], &gpu);
            let descriptor = storage.create_segment(index)?;
            Self::publish_segment(&mut state, config, index, key, gpu, &descriptor)?;

            state.taken.insert(key);
        }
        Ok(())
    }
}

/// The record of the one-dimensional byte tensor of `bytes` bytes at byte `offset` of the
/// storage a `put` carries with it. The server never interprets a tensor's bytes, so every
/// storage this crate publishes is stated as bytes; its owner knows what they hold.
pub(super) fn byte_tensor(
    key: Key,
    name: String,
    offset: u64,
    bytes: u64,
    access: Access,
) -> TensorRecord {
    TensorRecord {
        key,
        name,
        storage: 0,
        offset,
        dtype: "uint8".into(),
        shape: vec![bytes],
        strides: vec![1],
        access,
    }
}

/// The name a part of a segmented storage is retained under.
fn segment_name(name: &str, part: &SegmentPart) -> String {
    format!("{name}#{}/{}", part.column, part.segment.ordinal)
}

/// A Storage Server failure while the rank was doing `what`.
fn storage_error(what: &str, error: storage::Error) -> Error {
    Error::transport_source(what, error)
}
