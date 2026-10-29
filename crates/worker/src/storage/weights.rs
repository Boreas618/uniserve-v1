//! The model weights a worker keeps at its host's Storage Server.
//!
//! Every device tensor of a loaded model, each parameter and buffer by its
//! fields (a dense tensor's data, or a quantized tensor's encoded fields),
//! lies in a read-only retained storage: one per torch storage loading
//! made, the storages below the allocation granularity packed into one
//! storage per device. Each storage's storage is allocated in chunks, each a
//! CUDA VMM allocation of its own mapped back to back with the next
//! (`::storage::manifest::layout`): wherever another parallel configuration
//! of the parallel degrees in use would cut the logical tensors the storage
//! holds, rounded to the allocation granularity. A chunk is retained under
//! the key of what it holds: the regions of logical tensors its bytes are,
//! where in it they lie, and the fingerprint of what determines their
//! values and encoding (`uniserve.loading.retained.fingerprints`: the
//! tensor's checkpoint inputs, assignment rectangles, numerical formats,
//! declared loading transformations and what
//! selects the providers whose layouts encoded weights are stored in). The
//! key names neither the worker's rank nor its GPU: a chunk lies on a GPU,
//! the Storage Server keeping one copy of a key per GPU, so a worker of any
//! world and any parallel configuration that needs a region on a GPU finds
//! the chunk that holds it there.
//!
//! A manifest, host bytes keyed by the tensor identities, structure and rank's
//! position in its parallel configuration, names each tensor's source and
//! representation, and each storage's fields, their logical parts and
//! encodings, and its chunks. A worker of that position binds the tensors
//! whose storages' chunks are all retained on its GPUs instead of reading
//! the checkpoint. The worker binds only fields with matching loading identities,
//! using the recorded keys of their retained chunks. Unchanged fields of a
//! packed chunk remain reusable when neighboring fields change.
//!
//! Loading is collective, so every worker that loads with this one (its
//! worker group, or its expert union) agrees on what to read from the
//! checkpoint: the tensors any of them misses, or everything when one of
//! them has no manifest ([`Load`]). A worker binds the rest. A change to one
//! checkpoint source's files thus reloads only that source's tensors. A
//! worker that read tensors publishes them afterwards: a chunk retained
//! on its GPU under the same key, as another world's, is mapped instead of
//! allocated, and the worker copies its bytes into the chunks it creates;
//! every chunk is then mapped read-only, so any rewrite of a weight faults.
//! A chunk lives while any world retains it.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::ops::Range;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use ::storage::manifest::weights::RetainedStorage;
use ::storage::manifest::{self, Field, StorageLayout};
use ::storage::protocol::Access;
use ::storage::{Key, Mode, Placement, StorageRecord, TensorRecord};
use cuda::{DeviceAllocation, DeviceId, Protection};
use sha2::{Digest, Sha256};
use tvm_ffi::Any;

mod conversion;

use super::allocation::{Storage, StorageView};
use super::retained::Retained;

mod rebalance;
mod received;
use crate::config::WorkerLaunch;
use crate::ffi::{
    Backing, DType, Handle, Interpreter, integer_list, integers, list, strings, texts,
};
use crate::{Error, Result};
pub(crate) use rebalance::Rebalance;
pub(crate) use received::Receipt;

/// The library module that describes, publishes and binds retained weights.
const RETAINED: &str = "uniserve.loading.retained";

/// Byte alignment of each torch storage packed into a shared storage, the
/// alignment torch's allocator gives a storage.
const PACK_ALIGNMENT: u64 = 512;

/// The version of the manifest this worker writes and binds
/// (`::storage::manifest::weights`); a manifest of another version names no
/// tensor this worker binds. Its `tensors` are those
/// `uniserve.loading.retained.Layout.manifest` writes, of the same version.
use ::storage::manifest::weights::VERSION as MANIFEST_VERSION;

/// The name a rank's manifest is retained under, by which the manifests a
/// world retains are found.
use ::storage::manifest::weights::NAME as MANIFEST_NAME;

/// The identity of what `model`'s forwards compute: its components'
/// functions and formats (`uniserve.loading.rank.identity`), which keys
/// the cache pages they write.
pub(crate) fn model_identity(model: &Handle) -> Result<String> {
    let identity: tvm_ffi::String = Interpreter::import("uniserve.loading.rank.identity")?
        .call_values(&[model.to_any()])?
        .try_into()
        .map_err(|error| Error::numerical("identity", &error))?;
    Ok(identity.as_str().to_owned())
}

/// What determines a rank's retained bytes besides the regions they are,
/// as `uniserve.loading.retained.fingerprints` states it.
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct Fingerprints {
    /// Tensor shapes and representations, used to find the rank's manifest.
    pub structure: String,
    /// Each tensor's loading identity, keyed by its library tensor name.
    pub tensors: BTreeMap<String, String>,
    /// What determines the weights' values, regardless of the layouts their
    /// bytes take: it keys what forwards compute from them.
    pub values: String,
    /// Each cache layer's upstream computation and checkpoint inputs,
    /// described before parallel binding removes nonresident modules.
    pub cache: BTreeMap<String, String>,
}

impl Fingerprints {
    /// Upstream component keys for each physical group of cache layers.
    /// Grouping combines only resident layers; host page layouts add their
    /// head regions, encoding and page dimensions separately.
    pub(crate) fn cache_groups(&self, plan: &crate::storage::CachePlan) -> Result<Vec<String>> {
        cache_groups(&self.cache, plan)
    }
}

pub(crate) fn cache_groups(
    cache: &BTreeMap<String, String>,
    plan: &crate::storage::CachePlan,
) -> Result<Vec<String>> {
    plan.groups
        .iter()
        .map(|group| {
            let mut identities = group
                .layers
                .iter()
                .map(|layer| {
                    let identity = cache.get(layer).ok_or_else(|| {
                        Error::Interface(format!(
                            "cache layer {layer} declares no upstream identity"
                        ))
                    })?;
                    Ok(format!("{layer}:{identity}"))
                })
                .collect::<Result<Vec<_>>>()?;
            identities.sort();
            Ok(identities.join("|"))
        })
        .collect()
}

impl Fingerprints {
    /// The digest that keys the model's cache pages: what determines the
    /// weights' values and the identity of what its forwards compute
    /// (`uniserve.numerics.identity`, [`model_identity`]); neither the
    /// layouts the weights' bytes take nor the code that computes them.
    pub(crate) fn pages(&self, identity: &str) -> String {
        format!("{}:{identity}", self.values)
    }
}

/// The digest of a fingerprint's text, which chunk keys include.
fn fingerprint_digest(fingerprint: &str) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(b"uniserve-weight-fingerprint\0");
    digest.update(fingerprint.as_bytes());
    digest.finalize().into()
}

/// The keys of one worker's retained weights.
#[derive(Clone)]
pub(crate) struct WeightKeys {
    /// The digest of each tensor's own loading inputs and encoding.
    tensors: BTreeMap<String, [u8; 32]>,
    /// The manifest's key: the structure and the rank's position in its
    /// parallel configuration.
    manifest: Key,
    /// The rank's position ([`position`]).
    position: Vec<u8>,
    /// The worker's devices, its primary device first, with their GPU
    /// UUIDs: a manifest names a storage's device by its index here.
    devices: Vec<(DeviceId, [u8; 16])>,
    /// The tensor- and expert-parallel degrees weight storage is chunked
    /// for (`load_config.parallel_degrees`).
    degrees: Vec<u64>,
}

/// What a worker reads from the checkpoint, as the workers that load with
/// it agree: every tensor, or only the named tensors, which the others'
/// retained ones complete.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Load {
    All,
    Tensors(BTreeSet<String>),
}

impl Load {
    /// What this worker needs read: everything without a manifest, or when a
    /// tensor of a stack streamed from host storage is missing (a stack's
    /// host region is retained whole); otherwise its missing tensors.
    pub(crate) fn of(found: Option<&Found>) -> Self {
        match found {
            Some(found) if !found.host_missing => Self::Tensors(found.missing.clone()),
            _ => Self::All,
        }
    }

    /// What every worker reads: the union of their needs.
    pub(crate) fn union(needs: impl IntoIterator<Item = Self>) -> Self {
        let mut tensors = BTreeSet::new();
        for need in needs {
            match need {
                Self::All => return Self::All,
                Self::Tensors(names) => tensors.extend(names),
            }
        }
        Self::Tensors(tensors)
    }
}

/// The manifest of a rank's retained weights.
#[derive(serde::Serialize, serde::Deserialize)]
struct WeightManifest {
    version: u64,
    /// Each tensor's loading digest in hex. Readers bind or copy only fields
    /// whose owning tensor has the same digest.
    fingerprints: BTreeMap<String, String>,
    /// Each tensor's identity, checkpoint source, whether it streams from
    /// host storage, representation and field names, as the library writes
    /// them.
    tensors: Vec<ManifestTensor>,
    storages: Vec<RetainedStorage>,
}

/// One tensor a manifest names.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct ManifestTensor {
    id: String,
    /// The checkpoint source the tensor is loaded from; empty for a tensor
    /// no source loads.
    source: String,
    /// Whether the tensor lies in the host region of a streamed stack.
    host: bool,
    representation: serde_json::Value,
    fields: Vec<String>,
}

/// Weights a worker moved into retained storage and has not put yet: the
/// host regions of layers streamed from host storage join them
/// ([`Published::add_host`]) before [`Published::put`].
pub(crate) struct Published {
    keys: WeightKeys,
    /// The library's layout of the loaded model.
    layout: Handle,
    manifest: WeightManifest,
    /// The storages, mapped, in manifest order.
    storages: Vec<Arc<Storage>>,
    /// The chunks the worker created, with their records and descriptors.
    created: Vec<(TensorRecord, StorageRecord, OwnedFd)>,
    /// The keys of the chunks the worker maps from another world.
    shared: Vec<Key>,
}

/// Retained weights a worker found, mapped read-only.
pub(crate) struct Found {
    manifest: String,
    /// The storages whose chunks are all retained, mapped, by manifest
    /// index.
    storages: Vec<Arc<Storage>>,
    /// Each field's identity and where it lies: the index into `storages`,
    /// its byte offset, element type, shape and strides.
    fields: Vec<(String, FieldRecord)>,
    /// The tensors the manifest names with a field missing.
    missing: BTreeSet<String>,
    /// Whether one of them lies in the host region of a streamed stack.
    host_missing: bool,
    /// The manifest's description of each mapped storage, in `storages`
    /// order, which the manifest a worker publishes beside them keeps.
    kept: Vec<RetainedStorage>,
    /// The weights the worker receives into chunks of `storages` from
    /// other worlds; `None` when every chunk was retained already.
    receipt: Option<Receipt>,
}

/// Where one bound field lies.
struct FieldRecord {
    storage: u32,
    offset: u64,
    dtype: String,
    shape: Vec<u64>,
    strides: Vec<u64>,
}

impl WeightKeys {
    /// Republish a bound rank when a streaming owner replaces its resident
    /// fields. Existing ordinary and host storages stay in the same manifest.
    pub(crate) fn borrowed(
        &self,
        model: &Handle,
        host: &[String],
        found: &Found,
    ) -> Result<Published> {
        let layout = self.plan(model, host, None)?.layout;
        Ok(Published {
            keys: self.clone(),
            layout,
            manifest: WeightManifest {
                version: MANIFEST_VERSION,
                fingerprints: self.fingerprint_digests(),
                tensors: manifest_tensors(&found.manifest)?,
                storages: found.kept.clone(),
            },
            storages: found.storages.clone(),
            created: Vec::new(),
            shared: found
                .kept
                .iter()
                .flat_map(|storage| storage.storage.chunks.iter().map(|chunk| chunk.key))
                .collect(),
        })
    }
    /// The keys of weights of a model with `fingerprints` at `position` in
    /// its parallel configuration ([`position`]), on `devices` (the primary
    /// device first), chunked for the parallel `degrees`.
    pub(crate) fn new(
        retained: &Retained,
        fingerprints: &Fingerprints,
        position: &[u8],
        devices: &[DeviceId],
        degrees: &[u64],
    ) -> Result<Self> {
        let devices = devices
            .iter()
            .map(|device| Ok((*device, retained.gpu(*device)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::keyed(
            fingerprints,
            position.to_vec(),
            devices,
            degrees.to_vec(),
        ))
    }

    /// The keys [`WeightKeys::new`] gives the same rank, for a rank that
    /// plans its weights before its world starts and holds no connection to
    /// a Storage Server: each device's GPU is read from the driver.
    pub(crate) fn planning(
        fingerprints: &Fingerprints,
        position: &[u8],
        devices: &[DeviceId],
        degrees: &[u64],
    ) -> Result<Self> {
        let devices = devices
            .iter()
            .map(|device| Ok((*device, device.uuid()?.0)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self::keyed(
            fingerprints,
            position.to_vec(),
            devices,
            degrees.to_vec(),
        ))
    }

    /// The storages `planned` (this rank's, [`WeightKeys::plan`]) takes on
    /// the rank's one GPU, as a target rank of a copy manifest; `None` when
    /// they span devices or include a stack's host region, which a rank
    /// does not receive.
    pub(crate) fn planned_storages(&self, planned: &Planned) -> Option<Vec<manifest::Storage>> {
        (self.devices.len() == 1
            && planned
                .storages
                .iter()
                .all(|storage| storage.device == Some(0)))
        .then(|| {
            planned
                .storages
                .iter()
                .map(|storage| storage.storage.clone())
                .collect()
        })
    }

    fn keyed(
        fingerprints: &Fingerprints,
        position: Vec<u8>,
        devices: Vec<(DeviceId, [u8; 16])>,
        degrees: Vec<u64>,
    ) -> Self {
        let structure = fingerprint_digest(&fingerprints.structure);
        let mut digest = Sha256::new();
        digest.update(b"uniserve-weights-manifest\0");
        digest.update(structure);
        digest.update((position.len() as u64).to_le_bytes());
        digest.update(&position);
        // Manifests are immutable content-addressed records too. A changed
        // tensor must not publish new metadata under an existing manifest key.
        for (tensor, fingerprint) in &fingerprints.tensors {
            digest.update((tensor.len() as u64).to_le_bytes());
            digest.update(tensor.as_bytes());
            digest.update(fingerprint_digest(fingerprint));
        }
        Self {
            tensors: fingerprints
                .tensors
                .iter()
                .map(|(source, fingerprint)| (source.clone(), fingerprint_digest(fingerprint)))
                .collect(),
            manifest: Key(digest.finalize().into()),
            position,
            devices,
            degrees,
        }
    }

    /// The keys of the same rank's weights loaded from the checkpoint of
    /// `fingerprints`, as an in-place weight update rewrites them.
    pub(crate) fn under(&self, fingerprints: &Fingerprints) -> Self {
        Self::keyed(
            fingerprints,
            self.position.clone(),
            self.devices.clone(),
            self.degrees.clone(),
        )
    }

    /// The loading identity of the tensor owning one encoded field.
    fn fingerprint(&self, field: &str) -> Option<[u8; 32]> {
        let (tensor, _) = field.rsplit_once('/')?;
        self.tensors.get(tensor).copied()
    }

    /// Where device `index` of the worker lies.
    fn device(&self, index: usize) -> Result<(DeviceId, [u8; 16])> {
        self.devices.get(index).copied().ok_or_else(|| {
            Error::Interface(format!(
                "a retained weight lies on device {index} of {}",
                self.devices.len()
            ))
        })
    }

    /// The index of CUDA device ordinal `ordinal` among the worker's
    /// devices.
    fn device_index(&self, ordinal: i64) -> Result<usize> {
        self.devices
            .iter()
            .position(|(device, _)| i64::from(device.0) == ordinal)
            .ok_or_else(|| {
                Error::Invariant(format!(
                    "a loaded weight lies on device {ordinal}, which the worker does not address"
                ))
            })
    }

    fn first_device(&self) -> Result<DeviceId> {
        self.devices
            .first()
            .map(|(device, _)| *device)
            .ok_or_else(|| Error::Invariant("a worker retaining weights addresses a GPU".into()))
    }

    /// Where storage `storage` of a manifest lies.
    fn location(&self, storage: &RetainedStorage) -> Result<Placement> {
        match storage.device {
            Some(index) => Ok(Placement::Device {
                gpu: self.device(index)?.1,
            }),
            None => Ok(Placement::Host { numa_node: None }),
        }
    }

    /// The retained manifest's text; `None` without one.
    fn read_manifest(&self, retained: &Retained) -> Result<Option<String>> {
        read_text(retained, self.manifest)
    }

    /// The manifest this worker reads, with only fields whose loading
    /// identities match. Chunk keys continue to name the bytes actually
    /// retained, including when a packed chunk contains changed neighbors.
    fn manifest(&self, retained: &Retained) -> Result<Option<(String, WeightManifest)>> {
        let Some(text) = self.read_manifest(retained)? else {
            return Ok(None);
        };
        let version: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
            Error::Interface(format!("the retained weight manifest is not JSON: {error}"))
        })?;
        if version["version"] != MANIFEST_VERSION {
            tracing::info!("the retained weight manifest is of another version; loading");
            return Ok(None);
        }
        let mut manifest: WeightManifest = serde_json::from_str(&text).map_err(|error| {
            Error::Interface(format!(
                "the retained weight manifest is malformed: {error}"
            ))
        })?;
        let wanted = self.fingerprint_digests();
        for storage in &mut manifest.storages {
            storage.storage.layout.fields.retain(|field| {
                field.tensor.rsplit_once('/').is_some_and(|(tensor, _)| {
                    wanted
                        .get(tensor)
                        .is_some_and(|digest| manifest.fingerprints.get(tensor) == Some(digest))
                })
            });
        }
        manifest
            .storages
            .retain(|storage| !storage.storage.layout.fields.is_empty());
        Ok(Some((text, manifest)))
    }

    /// Rename rewritten chunks and publish their current tensor identities.
    /// The caller has completed the in-place numerical update while serving
    /// is paused; future readers must see both the new keys and their inputs.
    pub(crate) fn rename(&self, to: &WeightKeys, retained: &Retained) -> Result<usize> {
        let (_, mut manifest) = self
            .manifest(retained)?
            .ok_or_else(|| Error::State("the worker's weights have no retained manifest"))?;
        let mut moves = Vec::new();
        let mut chunks = Vec::new();
        for storage in &mut manifest.storages {
            let renamed = rekeyed(&storage.storage, &|source| to.fingerprint(source))?;
            for (chunk, key) in storage.storage.chunks.iter_mut().zip(renamed) {
                chunks.push((chunk.key, key));
                chunk.key = key;
            }
        }
        // Only the chunks the server retains on the worker's GPUs move.
        let at = self.locations();
        let got = retained.call("finding retained weights", |client| {
            client.get(chunks.iter().map(|(old, _)| *old).collect(), Mode::Read, at)
        })?;
        let held: BTreeSet<Key> = got.tensors.iter().map(|record| record.key).collect();
        moves.extend(
            chunks
                .into_iter()
                .filter(|(old, new)| old != new && held.contains(old)),
        );
        moves.sort_unstable();
        moves.dedup();
        let moved = moves.len();
        retained.rekey(moves)?;
        manifest.fingerprints = to.fingerprint_digests();
        let (tensor, record, descriptor) = manifest_record(to.manifest, &manifest)?;
        retained.call("publishing updated weight identities", |client| {
            client.put(vec![tensor], &[(record, descriptor.as_fd())])
        })?;
        retained.use_keys([to.manifest]);
        if self.manifest != to.manifest {
            retained.forget(&[self.manifest])?;
        }
        Ok(moved)
    }

    /// Where the worker's retained weights lie: its GPUs and host storage.
    fn locations(&self) -> Vec<Placement> {
        self.devices
            .iter()
            .map(|(_, gpu)| Placement::Device { gpu: *gpu })
            .chain(std::iter::once(Placement::Host { numa_node: None }))
            .collect()
    }

    /// Map the weights retained for this worker's position: every tensor
    /// its manifest names whose storages' chunks are all retained on its
    /// devices. `None` without a manifest.
    pub(crate) fn find(&self, retained: &Retained) -> Result<Option<Found>> {
        let started = std::time::Instant::now();
        let Some((text, manifest)) = self.manifest(retained)? else {
            return Ok(None);
        };
        let keys: Vec<Key> = manifest
            .storages
            .iter()
            .flat_map(|storage| storage.storage.chunks.iter().map(|chunk| chunk.key))
            .collect();
        let got = retained.call("finding retained weights", |client| {
            client.get(keys, Mode::Read, self.locations())
        })?;
        // The descriptor of each chunk found, by key and where it lies.
        let mut descriptors: HashMap<(Key, Placement), (u64, OwnedFd)> = HashMap::new();
        let mut records: Vec<Option<(StorageRecord, OwnedFd)>> =
            got.storages.into_iter().map(Some).collect();
        for record in &got.tensors {
            let Some((storage, descriptor)) = records
                .get(record.storage as usize)
                .and_then(|entry| entry.as_ref())
                .map(|(storage, descriptor)| (*storage, descriptor.try_clone()))
            else {
                continue;
            };
            let descriptor = descriptor.map_err(|error| {
                Error::transport_source("duplicating a retained descriptor", error)
            })?;
            descriptors.insert((record.key, storage.placement), (storage.bytes, descriptor));
        }
        records.clear();

        // Map every storage whose chunks are all found.
        let mut storages = Vec::new();
        let mut fields = Vec::new();
        let mut kept = Vec::new();
        let mut used = vec![self.manifest];
        for retained_storage in &manifest.storages {
            let location = self.location(retained_storage)?;
            let chunks = &retained_storage.storage.chunks;
            let present: Option<Vec<&(u64, OwnedFd)>> = chunks
                .iter()
                .map(|chunk| descriptors.get(&(chunk.key, location)))
                .collect();
            let Some(present) = present else {
                continue;
            };
            let storage = match retained_storage.device {
                Some(device) => {
                    let (device, _) = self.device(device)?;
                    let lengths: Vec<usize> =
                        chunks.iter().map(|chunk| chunk.length as usize).collect();
                    let storage = Storage::segmented(device, &lengths, &[])?;
                    for (segment, (_, descriptor)) in present.iter().enumerate() {
                        storage.import_segment_read_only(segment, descriptor)?;
                    }
                    storage
                }
                None => {
                    let (bytes, descriptor) = present.first().ok_or_else(|| {
                        Error::Interface("a retained host region has no chunk".into())
                    })?;
                    let descriptor = descriptor.try_clone().map_err(|error| {
                        Error::transport_source("duplicating a retained descriptor", error)
                    })?;
                    Storage::import_host(descriptor, *bytes as usize, self.first_device()?)?
                }
            };
            used.extend(chunks.iter().map(|chunk| chunk.key));
            kept.push(retained_storage.clone());
            let mapped = storages.len() as u32;
            storages.push(storage);
            for field in &retained_storage.storage.layout.fields {
                fields.push((field.tensor.clone(), field_record(mapped, field)));
            }
        }
        // A tensor is bound when every field of it lies in a mapped storage.
        let bound: BTreeSet<&str> = fields.iter().map(|(id, _)| id.as_str()).collect();
        let mut missing = BTreeSet::new();
        let mut host_missing = false;
        for tensor in &manifest.tensors {
            let whole = tensor
                .fields
                .iter()
                .all(|field| bound.contains(format!("{}/{field}", tensor.id).as_str()));
            if !whole {
                missing.insert(tensor.id.clone());
                host_missing |= tensor.host;
            }
        }
        fields.retain(|(id, _)| {
            id.rsplit_once('/')
                .is_none_or(|(tensor, _)| !missing.contains(tensor))
        });
        retained.use_keys(used);
        tracing::info!(
            storages = storages.len(),
            bytes = storages
                .iter()
                .map(|storage| storage.bytes())
                .sum::<usize>(),
            missing = missing.len(),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "mapped the retained weights"
        );
        Ok(Some(Found {
            manifest: text,
            storages,
            fields,
            missing,
            host_missing,
            kept,
            receipt: None,
        }))
    }

    /// Move a loaded model's device weights into retained chunks (`model`
    /// is `rank.Loaded`): every tensor, or only the tensors `only` names,
    /// which the worker read while it bound the others. A chunk the server
    /// retains on the same GPU under the same key is mapped instead of
    /// created. The projection matrices of the stacks under `host_paths`
    /// stream from host storage: their host regions join the published
    /// weights once allocated ([`Published::add_host`]).
    pub(crate) fn publish(
        &self,
        retained: &Retained,
        model: &Handle,
        host_paths: &[String],
        only: Option<&BTreeSet<String>>,
    ) -> Result<Published> {
        let Planned {
            layout,
            tensors,
            sizes,
            storages: retained_storages,
            placed,
        } = self.plan(model, host_paths, only)?;

        // Chunks another world retains on the same GPU are mapped.
        let keys: Vec<Key> = retained_storages
            .iter()
            .flat_map(|storage| storage.storage.chunks.iter().map(|chunk| chunk.key))
            .collect();
        let at: Vec<Placement> = self
            .devices
            .iter()
            .map(|(_, gpu)| Placement::Device { gpu: *gpu })
            .collect();
        let got = retained.call("finding retained weight chunks", |client| {
            client.get(keys, Mode::Read, at)
        })?;
        let mut existing: HashMap<(Key, Placement), OwnedFd> = HashMap::new();
        for record in &got.tensors {
            if let Some((storage, descriptor)) = got.storages.get(record.storage as usize) {
                let descriptor = descriptor.try_clone().map_err(|error| {
                    Error::transport_source("duplicating a retained descriptor", error)
                })?;
                existing.insert((record.key, storage.placement), descriptor);
            }
        }
        self.materialize(layout, tensors, sizes, retained_storages, placed, existing)
    }

    /// The retained storages a loaded model's device weights become
    /// (`model` is `rank.Loaded`), chunked and keyed, before any storage is
    /// allocated: every tensor, or only those `only` names; the stacks
    /// under `host_paths` stream from host storage.
    pub(crate) fn plan(
        &self,
        model: &Handle,
        host_paths: &[String],
        only: Option<&BTreeSet<String>>,
    ) -> Result<Planned> {
        let layout = Handle::new(
            Interpreter::import(&format!("{RETAINED}.describe"))?.call_values(&[
                model.to_any(),
                Any::from(strings(host_paths)),
                only.map(|names| Any::from(strings(names)))
                    .unwrap_or_default(),
            ])?,
        )?;
        self.plan_layout(layout)
    }

    /// Chunk a numerical layout supplied by loading or expert redistribution.
    pub(crate) fn plan_layout(&self, layout: Handle) -> Result<Planned> {
        let method = |name: &str| Interpreter::import(&format!("{RETAINED}.Layout.{name}"));
        let numbers = |name: &str| -> Result<Vec<i64>> {
            integers(method(name)?.call_values(&[layout.to_any()])?)
                .map_err(|error| Error::numerical(name, &error))
        };
        let names = |name: &str| -> Result<Vec<String>> {
            texts(method(name)?.call_values(&[layout.to_any()])?)
                .map_err(|error| Error::numerical(name, &error))
        };
        let tensors: tvm_ffi::String = method("manifest")?.call_as(&[layout.view()])?;
        let tensors = manifest_tensors(tensors.as_str())?;
        let devices = numbers("storage_devices")?;
        let sizes = numbers("storage_bytes")?;
        let fields = FieldColumns {
            ids: names("field_ids")?,
            sources: names("field_sources")?,
            storages: numbers("field_storages")?,
            offsets: numbers("field_offsets")?,
            dtypes: names("field_dtypes")?,
            ranks: numbers("field_ranks")?,
            shapes: numbers("field_shapes")?,
            strides: numbers("field_strides")?,
            parts: names("field_parts")?,
        }
        .fields()?;

        let devices = devices
            .iter()
            .map(|&ordinal| self.device_index(ordinal))
            .collect::<Result<Vec<_>>>()?;
        let granularity = self
            .devices
            .iter()
            .map(|(device, _)| Ok(DeviceAllocation::granularity(*device)? as u64))
            .collect::<Result<Vec<_>>>()?;
        let (retained_storages, placed) = retained_storages(
            &devices,
            &sizes,
            fields,
            &granularity,
            &self.degrees,
            &|source| self.fingerprint(source),
        )?;
        Ok(Planned {
            layout,
            tensors,
            sizes,
            storages: retained_storages,
            placed,
        })
    }

    /// Allocate `retained_storages` and move the loaded bytes `layout`
    /// describes into them: a chunk of `existing` (by key and location) is
    /// mapped read-only, every other chunk created and written.
    fn materialize(
        &self,
        layout: Handle,
        tensors: Vec<ManifestTensor>,
        sizes: Vec<i64>,
        retained_storages: Vec<RetainedStorage>,
        placed: Members,
        existing: HashMap<(Key, Placement), OwnedFd>,
    ) -> Result<Published> {
        let method = |name: &str| Interpreter::import(&format!("{RETAINED}.Layout.{name}"));
        let mut storages = Vec::with_capacity(retained_storages.len());
        let mut created = Vec::new();
        let mut shared = Vec::new();
        // The byte ranges of each retained storage the worker writes.
        let mut written: Vec<Vec<Range<u64>>> = Vec::with_capacity(retained_storages.len());
        for retained_storage in &retained_storages {
            let (device, gpu) = self.device(retained_storage.device.unwrap_or(0))?;
            let location = Placement::Device { gpu };
            let chunks = &retained_storage.storage.chunks;
            let lengths: Vec<usize> = chunks.iter().map(|chunk| chunk.length as usize).collect();
            let storage = Storage::segmented(device, &lengths, &[])?;
            let mut ranges = Vec::new();
            for (segment, chunk) in chunks.iter().enumerate() {
                if let Some(descriptor) = existing.get(&(chunk.key, location)) {
                    storage.import_segment_read_only(segment, descriptor)?;
                    shared.push(chunk.key);
                    continue;
                }
                let descriptor = storage.create_segment(segment)?;
                created.push((
                    TensorRecord {
                        key: chunk.key,
                        name: format!("weights {} chunk {segment}", storage_name(retained_storage)),
                        storage: 0,
                        offset: 0,
                        dtype: "uint8".into(),
                        shape: vec![chunk.length],
                        strides: vec![1],
                        access: Access::ReadOnly,
                    },
                    StorageRecord {
                        placement: location,
                        bytes: chunk.length,
                        segment: None,
                        fabric: storage.segment_fabric(segment)?,
                    },
                    descriptor,
                ));
                ranges.push(chunk.offset..chunk.offset + chunk.length);
            }
            storages.push(storage);
            written.push(ranges);
        }

        // Copy each torch storage's bytes into the chunks the worker
        // created; the chunks it maps hold them already.
        let publish = method("publish")?;
        let views = storages
            .iter()
            .map(|storage| StorageView::whole(Arc::clone(storage) as Arc<dyn Backing>, DType::U8))
            .collect::<Result<Vec<_>>>()?;
        for (index, &(storage, offset)) in placed.iter().enumerate() {
            let size = sizes[index] as u64;
            // The created chunks' bytes of this torch storage, relative to
            // its start, one range per chunk.
            let ranges: Vec<i64> = written[storage]
                .iter()
                .filter_map(|range| {
                    let start = range.start.max(offset);
                    let end = range.end.min(offset + size);
                    (start < end).then(|| [(start - offset) as i64, (end - offset) as i64])
                })
                .flatten()
                .collect();
            publish.call_values(&[
                layout.to_any(),
                Any::from(index as i64),
                views[storage].handle()?.to_any(),
                Any::from(offset as i64),
                Any::from(integer_list(ranges)),
            ])?;
        }
        drop(views);
        Ok(Published {
            keys: self.clone(),
            layout,
            manifest: WeightManifest {
                version: MANIFEST_VERSION,
                fingerprints: self.fingerprint_digests(),
                tensors,
                storages: retained_storages,
            },
            storages,
            created,
            shared,
        })
    }

    /// The fingerprint digests a manifest of this worker names, in hex
    /// (`WeightManifest::fingerprints`).
    pub(crate) fn fingerprint_digests(&self) -> BTreeMap<String, String> {
        self.tensors
            .iter()
            .map(|(tensor, digest)| (tensor.clone(), hex(digest)))
            .collect()
    }
}

/// A digest in hex.
fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A rank's weights planned for retention ([`WeightKeys::plan`]): the
/// library's layout of the loaded model, the manifest's tensors, each torch
/// storage's bytes, the retained storages they become, chunked and keyed,
/// and each torch storage's retained storage and offset there.
pub(crate) struct Planned {
    layout: Handle,
    tensors: Vec<ManifestTensor>,
    sizes: Vec<i64>,
    storages: Vec<RetainedStorage>,
    placed: Members,
}

/// A transfer's numerical description, independent of the model that supplied it.
pub(crate) struct TransferLayout {
    tensors: Vec<ManifestTensor>,
    storages: Vec<RetainedStorage>,
}

impl Planned {
    /// Release the borrowed Python layout before allocating transfer targets.
    pub(crate) fn into_transfer(self) -> TransferLayout {
        TransferLayout {
            tensors: self.tensors,
            storages: self.storages,
        }
    }
}

/// Each torch storage's retained storage and its byte offset there.
type Members = Vec<(usize, u64)>;

/// The retained storages a rank's torch storages become, chunked and
/// keyed, and each torch storage's retained storage and byte offset there.
///
/// `devices[i]` and `sizes[i]` are torch storage `i`'s device (an index
/// into `granularity`, the allocation granularity of each of the rank's
/// devices) and bytes; `fields` the fields with the torch storage each
/// views. A torch storage of at least the granularity is a retained
/// storage of its own; the smaller ones of a device are packed into one,
/// each at the next multiple of [`PACK_ALIGNMENT`]. Each retained storage
/// is chunked for the parallel `degrees` and keyed with `fingerprint`
/// (`::storage::manifest::Storage::chunked`).
fn retained_storages(
    devices: &[usize],
    sizes: &[i64],
    fields: Vec<(usize, Field)>,
    granularity: &[u64],
    degrees: &[u64],
    fingerprint: &dyn Fn(&str) -> Option<[u8; 32]>,
) -> Result<(Vec<RetainedStorage>, Members)> {
    let granule = |device: usize| {
        granularity.get(device).copied().ok_or_else(|| {
            Error::Interface(format!("a torch storage lies on unknown device {device}"))
        })
    };
    let mut by_storage: Vec<Vec<Field>> = vec![Vec::new(); sizes.len()];
    for (storage, field) in fields {
        by_storage
            .get_mut(storage)
            .ok_or_else(|| Error::Interface(format!("field {} views no storage", field.tensor)))?
            .push(field);
    }
    let mut layouts: Vec<(usize, StorageLayout)> = Vec::new();
    let mut packs: BTreeMap<usize, usize> = BTreeMap::new();
    let mut placed = Vec::with_capacity(sizes.len());
    for ((&device, &size), storage_fields) in devices.iter().zip(sizes).zip(by_storage) {
        let size = size as u64;
        let target = if size < granule(device)? {
            let pack = *packs.entry(device).or_insert_with(|| {
                layouts.push((
                    device,
                    StorageLayout {
                        bytes: 0,
                        fields: Vec::new(),
                    },
                ));
                layouts.len() - 1
            });
            let offset = layouts[pack].1.bytes.next_multiple_of(PACK_ALIGNMENT);
            layouts[pack].1.bytes = offset + size;
            (pack, offset)
        } else {
            layouts.push((
                device,
                StorageLayout {
                    bytes: size,
                    fields: Vec::new(),
                },
            ));
            (layouts.len() - 1, 0)
        };
        placed.push(target);
        for mut field in storage_fields {
            field.offset += target.1;
            layouts[target.0].1.fields.push(field);
        }
    }
    let storages = layouts
        .into_iter()
        .map(|(device, layout)| {
            let storage =
                manifest::Storage::chunked(layout, granule(device)?, degrees, fingerprint)
                    .map_err(|error| Error::Interface(error.to_string()))?;
            Ok(RetainedStorage {
                device: Some(device),
                storage,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((storages, placed))
}

/// The retained storages of a rank described by the library's layout
/// columns, as JSON: what [`WeightKeys::publish`] retains of the same
/// layout, for a planner that computes a world's chunks without its
/// workers (`uniserve.loading.retained.Layout.columns`).
///
/// The description is a JSON object of the layout's `storage_devices`,
/// `storage_bytes` and `field_*` lists, each device's allocation
/// `granularity` (by device ordinal, as a list indexed by the devices in
/// order of first use), the parallel `degrees`, and the model's
/// `fingerprints` (`uniserve.loading.retained.fingerprints`). Returns
/// `{"storages": [...], "members": [[storage, offset], ...]}`: each
/// retained storage's device index, layout and chunks, and each torch
/// storage's retained storage and offset.
pub(crate) fn plan_storages(description: &str) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct Description {
        storage_devices: Vec<i64>,
        storage_bytes: Vec<i64>,
        field_ids: Vec<String>,
        field_sources: Vec<String>,
        field_storages: Vec<i64>,
        field_offsets: Vec<i64>,
        field_dtypes: Vec<String>,
        field_ranks: Vec<i64>,
        field_shapes: Vec<i64>,
        field_strides: Vec<i64>,
        field_parts: Vec<String>,
        granularity: Vec<u64>,
        degrees: Vec<u64>,
        fingerprints: Fingerprints,
    }
    let description: Description = serde_json::from_str(description)
        .map_err(|error| Error::Invalid(format!("a weight layout description: {error}")))?;
    // Devices in order of first use, as a worker's devices list them.
    let mut ordinals: Vec<i64> = Vec::new();
    let devices: Vec<usize> = description
        .storage_devices
        .iter()
        .map(|ordinal| {
            ordinals
                .iter()
                .position(|seen| seen == ordinal)
                .unwrap_or_else(|| {
                    ordinals.push(*ordinal);
                    ordinals.len() - 1
                })
        })
        .collect();
    let keys = WeightKeys::keyed(
        &description.fingerprints,
        Vec::new(),
        Vec::new(),
        description.degrees.clone(),
    );
    let fields = FieldColumns {
        ids: description.field_ids,
        sources: description.field_sources,
        storages: description.field_storages,
        offsets: description.field_offsets,
        dtypes: description.field_dtypes,
        ranks: description.field_ranks,
        shapes: description.field_shapes,
        strides: description.field_strides,
        parts: description.field_parts,
    }
    .fields()?;
    let (storages, members) = retained_storages(
        &devices,
        &description.storage_bytes,
        fields,
        &description.granularity,
        &description.degrees,
        &|source| keys.fingerprint(source),
    )?;
    serde_json::to_string(&serde_json::json!({
        "storages": storages,
        "members": members,
    }))
    .map_err(|error| Error::Invalid(format!("encoding retained storages: {error}")))
}

/// A readable name of a retained storage: its first field's tensor.
fn storage_name(storage: &RetainedStorage) -> String {
    storage
        .storage
        .layout
        .fields
        .first()
        .map_or_else(String::new, |field| field.tensor.clone())
}

/// The keys of `storage`'s chunks under other fingerprints.
fn rekeyed(
    storage: &manifest::Storage,
    fingerprint: &dyn Fn(&str) -> Option<[u8; 32]>,
) -> Result<Vec<Key>> {
    storage
        .chunks
        .iter()
        .map(|chunk| {
            storage
                .layout
                .chunk_key(&(chunk.offset..chunk.offset + chunk.length), fingerprint)
                .map_err(|error| Error::Interface(error.to_string()))
        })
        .collect()
}

/// The record binding reads of `field`, which lies in mapped storage
/// `storage`.
fn field_record(storage: u32, field: &Field) -> FieldRecord {
    FieldRecord {
        storage,
        offset: field.offset,
        dtype: field.dtype.clone(),
        shape: field.shape.clone(),
        strides: field.strides.clone(),
    }
}

/// The library's flat lists over a layout's fields.
struct FieldColumns {
    ids: Vec<String>,
    sources: Vec<String>,
    storages: Vec<i64>,
    offsets: Vec<i64>,
    dtypes: Vec<String>,
    ranks: Vec<i64>,
    shapes: Vec<i64>,
    strides: Vec<i64>,
    /// Each field's logical part and encoding, as JSON.
    parts: Vec<String>,
}

/// A field's logical part and encoding as the library declares it
/// (`uniserve.loading.retained.field_parts`).
#[derive(serde::Deserialize)]
struct FieldPart {
    element: u64,
    logical: manifest::Logical,
    encoding: manifest::Encoding,
}

impl FieldColumns {
    /// Each field with the index of the torch storage it views.
    fn fields(self) -> Result<Vec<(usize, Field)>> {
        let mut fields = Vec::with_capacity(self.ids.len());
        let mut position = 0;
        for (index, id) in self.ids.iter().enumerate() {
            let rank = self.ranks[index] as usize;
            let part: FieldPart = serde_json::from_str(&self.parts[index]).map_err(|error| {
                Error::Interface(format!("the logical part of field {id}: {error}"))
            })?;
            fields.push((
                self.storages[index] as usize,
                Field {
                    tensor: id.clone(),
                    source: self.sources[index].clone(),
                    dtype: self.dtypes[index].clone(),
                    element: part.element,
                    offset: self.offsets[index] as u64,
                    shape: self.shapes[position..position + rank]
                        .iter()
                        .map(|&extent| extent as u64)
                        .collect(),
                    strides: self.strides[position..position + rank]
                        .iter()
                        .map(|&stride| stride as u64)
                        .collect(),
                    logical: part.logical,
                    encoding: part.encoding,
                },
            ));
            position += rank;
        }
        Ok(fields)
    }
}

impl Published {
    /// Replace numerical fields with the allocations their streaming owner
    /// actually keeps resident. Fields in ordinary or host storage remain
    /// retained; superseded shard allocations are released with this owner.
    /// Returns keys the caller can release after publishing the replacement.
    pub(crate) fn replace_resident(
        &mut self,
        layout: Handle,
        live: &[Arc<Storage>],
    ) -> Result<Vec<Key>> {
        let method = |name: &str| Interpreter::import(&format!("{RETAINED}.Layout.{name}"));
        let numbers = |name: &str| -> Result<Vec<i64>> {
            integers(method(name)?.call_values(&[layout.to_any()])?)
                .map_err(|error| Error::numerical(name, &error))
        };
        let names = |name: &str| -> Result<Vec<String>> {
            texts(method(name)?.call_values(&[layout.to_any()])?)
                .map_err(|error| Error::numerical(name, &error))
        };
        let tensors: tvm_ffi::String = method("manifest")?.call_as(&[layout.view()])?;
        let tensors = manifest_tensors(tensors.as_str())?;
        let replaced: BTreeSet<_> = tensors.iter().map(|tensor| tensor.id.clone()).collect();
        let fields = FieldColumns {
            ids: names("field_ids")?,
            sources: names("field_sources")?,
            storages: numbers("field_storages")?,
            offsets: numbers("field_offsets")?,
            dtypes: names("field_dtypes")?,
            ranks: numbers("field_ranks")?,
            shapes: numbers("field_shapes")?,
            strides: numbers("field_strides")?,
            parts: names("field_parts")?,
        }
        .fields()?;
        let addresses = numbers("storage_addresses")?;
        let mut incoming = Vec::new();
        let mut replaced_fields = BTreeSet::new();
        for (index, address) in addresses.into_iter().enumerate() {
            let Some(owner) = live.iter().find(|owner| owner.address() == address as u64) else {
                // Scalar quantization constants retain their existing owner;
                // only fields transferred into distributed pages are replaced.
                continue;
            };
            let allocation = owner.allocation().ok_or(Error::State(
                "distributed resident fields use exportable allocations",
            ))?;
            let device = self.keys.device_index(allocation.device().0 as i64)?;
            let (_, gpu) = self.keys.device(device)?;
            let location = Placement::Device { gpu };
            let storage_layout = StorageLayout {
                bytes: owner.bytes() as u64,
                fields: fields
                    .iter()
                    .filter(|(storage, _)| *storage == index)
                    .map(|(_, field)| field.clone())
                    .collect(),
            };
            replaced_fields.extend(
                storage_layout
                    .fields
                    .iter()
                    .map(|field| field.tensor.clone()),
            );
            let length = storage_layout.bytes;
            let key = storage_layout
                .chunk_key(&(0..length), &|source| self.keys.fingerprint(source))
                .map_err(|error| Error::Interface(error.to_string()))?;
            let descriptor = allocation.export_descriptor()?.ok_or(Error::State(
                "resident weights export a retained descriptor",
            ))?;
            let fabric = match allocation.export()? {
                Some(cuda::ShareableHandle::Fabric(bytes)) => {
                    Some(::storage::protocol::FabricHandle(bytes))
                }
                _ => None,
            };
            self.created.push((
                TensorRecord {
                    key,
                    name: "resident expert weights".into(),
                    storage: 0,
                    offset: 0,
                    dtype: "uint8".into(),
                    shape: vec![length],
                    strides: vec![1],
                    access: Access::ReadOnly,
                },
                StorageRecord {
                    placement: location,
                    bytes: length,
                    segment: None,
                    fabric,
                },
                descriptor,
            ));
            incoming.push((
                Arc::clone(owner),
                RetainedStorage {
                    device: Some(device),
                    storage: manifest::Storage {
                        layout: storage_layout,
                        chunks: vec![manifest::Chunk {
                            offset: 0,
                            length,
                            key,
                            storage_offset: 0,
                            binding: manifest::Binding::Local { at: location },
                        }],
                    },
                },
            ));
        }
        let mut retired = Vec::new();
        let mut kept = Vec::new();
        let mut kept_owners = Vec::new();
        for (owner, mut storage) in self
            .storages
            .drain(..)
            .zip(self.manifest.storages.drain(..))
        {
            storage
                .storage
                .layout
                .fields
                .retain(|field| !replaced_fields.contains(&field.tensor));
            if !storage.storage.layout.fields.is_empty() {
                kept.push(storage);
                kept_owners.push(owner);
            } else {
                retired.extend(storage.storage.chunks.iter().map(|chunk| chunk.key));
            }
        }
        for (owner, storage) in incoming {
            kept_owners.push(owner);
            kept.push(storage);
        }
        self.storages = kept_owners;
        self.manifest.storages = kept;
        self.manifest
            .tensors
            .retain(|tensor| !replaced.contains(&tensor.id));
        self.manifest.tensors.extend(tensors);
        let current: BTreeSet<_> = self
            .manifest
            .storages
            .iter()
            .flat_map(|storage| storage.storage.chunks.iter().map(|chunk| chunk.key))
            .collect();
        self.created
            .retain(|(tensor, _, _)| current.contains(&tensor.key));
        self.shared
            .retain(|key| current.contains(key) && !retired.contains(key));
        Ok(retired)
    }

    /// Keep in the manifest the storages the worker bound from `found`,
    /// beside those it publishes: a later worker binds them too. Packed
    /// storage may hold both retained and reloaded tensors; only reloaded
    /// fields leave its manifest. Its owner must remain available for access
    /// changes and future updates while any retained field still uses it.
    pub(crate) fn keep(&mut self, found: &Found, load: &BTreeSet<String>) {
        for (storage, owner) in found.kept.iter().zip(&found.storages) {
            let mut storage = storage.clone();
            storage.storage.layout.fields.retain(|field| {
                field
                    .tensor
                    .rsplit_once('/')
                    .is_none_or(|(tensor, _)| !load.contains(tensor))
            });
            if storage.storage.layout.fields.is_empty() {
                continue;
            }
            self.shared
                .extend(storage.storage.chunks.iter().map(|chunk| chunk.key));
            self.manifest.storages.push(storage);
            self.storages.push(Arc::clone(owner));
        }
    }

    /// Add the host region of the layer stack under `path`, which
    /// `storage` holds, before the stack is bound to its slots (`model` is
    /// `rank.Loaded`). The region is retained whole, under the key of what
    /// its fields hold.
    pub(crate) fn add_host(
        &mut self,
        model: &Handle,
        path: &str,
        storage: Arc<Storage>,
        descriptor: OwnedFd,
    ) -> Result<()> {
        let function = format!("{RETAINED}.Layout.host_fields");
        let invalid = |error: tvm_ffi::Error| Error::numerical(&function, &error);
        let fields = Interpreter::import(&function)?.call_values(&[
            self.layout.to_any(),
            model.to_any(),
            Any::from(tvm_ffi::String::from(path)),
        ])?;
        let columns: tvm_ffi::Array<Any> = tvm_ffi::Array::try_from(fields).map_err(invalid)?;
        let column = |index: usize| columns.get(index).map_err(invalid);
        let fields = FieldColumns {
            ids: texts(column(0)?).map_err(invalid)?,
            offsets: integers(column(1)?).map_err(invalid)?,
            dtypes: texts(column(2)?).map_err(invalid)?,
            ranks: integers(column(3)?).map_err(invalid)?,
            shapes: integers(column(4)?).map_err(invalid)?,
            strides: integers(column(5)?).map_err(invalid)?,
            sources: texts(column(6)?).map_err(invalid)?,
            parts: texts(column(7)?).map_err(invalid)?,
            storages: Vec::new(),
        };
        let count = fields.ids.len();
        let fields = FieldColumns {
            storages: vec![0; count],
            ..fields
        }
        .fields()?;
        let layout = StorageLayout {
            bytes: storage.bytes() as u64,
            fields: fields.into_iter().map(|(_, field)| field).collect(),
        };
        let window = 0..layout.bytes;
        let key = layout
            .chunk_key(&window, &|source| self.keys.fingerprint(source))
            .map_err(|error| Error::Interface(error.to_string()))?;
        let location = Placement::Host { numa_node: None };
        self.created.push((
            TensorRecord {
                key,
                name: format!("weights host region {path}"),
                storage: 0,
                offset: 0,
                dtype: "uint8".into(),
                shape: vec![layout.bytes],
                strides: vec![1],
                access: Access::ReadOnly,
            },
            StorageRecord {
                placement: location,
                bytes: layout.bytes,
                segment: None,
                fabric: None,
            },
            descriptor,
        ));
        self.manifest.storages.push(RetainedStorage {
            device: None,
            storage: manifest::Storage {
                chunks: vec![manifest::Chunk {
                    offset: 0,
                    length: layout.bytes,
                    key,
                    storage_offset: 0,
                    binding: manifest::Binding::Local {
                        at: Placement::Host { numa_node: None },
                    },
                }],
                layout,
            },
        });
        self.storages.push(storage);
        Ok(())
    }

    /// Put the chunks the worker created and the manifest, and make every
    /// retained mapping read-only: no worker writes the weights again.
    /// Returns the retained storages, which the worker holds for its
    /// lifetime.
    pub(crate) fn put(self, retained: &Retained) -> Result<Vec<Arc<Storage>>> {
        let Self {
            keys,
            layout,
            manifest,
            storages,
            created,
            shared,
        } = self;
        drop(layout);
        let (mut manifest_tensor, manifest_record, manifest_storage) =
            manifest_record(keys.manifest, &manifest)?;
        let mut tensors: Vec<TensorRecord> = Vec::with_capacity(created.len() + 1);
        let mut records: Vec<(StorageRecord, std::os::fd::BorrowedFd<'_>)> =
            Vec::with_capacity(created.len() + 1);
        for (index, (tensor, storage, descriptor)) in created.iter().enumerate() {
            tensors.push(TensorRecord {
                storage: index as u32,
                ..tensor.clone()
            });
            records.push((*storage, descriptor.as_fd()));
        }
        manifest_tensor.storage = records.len() as u32;
        tensors.push(manifest_tensor);
        records.push((manifest_record, manifest_storage.as_fd()));
        let used: Vec<Key> = tensors
            .iter()
            .map(|tensor| tensor.key)
            .chain(shared.iter().copied())
            .collect();
        // The transport carries a message's descriptors however many.
        retained.call("publishing weights", |client| client.put(tensors, &records))?;
        retained.use_keys(used);
        drop(records);

        let (mut device_bytes, mut host_bytes) = (0, 0);
        for storage in &storages {
            storage.protect(Protection::ReadOnly)?;
            match storage.device() {
                crate::ffi::MemoryKind::Cuda(_) => device_bytes += storage.bytes(),
                _ => host_bytes += storage.bytes(),
            }
        }
        tracing::info!(
            storages = storages.len(),
            created = created.len(),
            shared = shared.len(),
            device_bytes,
            host_bytes,
            "published the loaded weights"
        );
        Ok(storages)
    }
}

impl Found {
    /// The library's `Retained` tensors over the mapped storages, which
    /// loading installs instead of reading the checkpoint, but for the
    /// tensors `load` names, which it reads.
    pub(crate) fn bind(&self, load: &BTreeSet<String>) -> Result<Handle> {
        let views = self
            .storages
            .iter()
            .map(|storage| StorageView::whole(Arc::clone(storage) as Arc<dyn Backing>, DType::U8))
            .collect::<Result<Vec<_>>>()?;
        let tensors = views
            .iter()
            .map(|view| Ok(view.handle()?.to_any()))
            .collect::<Result<Vec<Any>>>()?;
        // The fields of the tensors loading reads are left out.
        let bound: Vec<&(String, FieldRecord)> = self
            .fields
            .iter()
            .filter(|(id, _)| {
                id.rsplit_once('/')
                    .is_none_or(|(tensor, _)| !load.contains(tensor))
            })
            .collect();
        let records = || bound.iter().map(|(_, record)| record);
        let build = Interpreter::import(&format!("{RETAINED}.Retained.build"))?;
        Handle::new(build.call_values(&[
            Any::from(tvm_ffi::String::from(self.manifest.as_str())),
            Any::from(list(tensors)),
            Any::from(strings(bound.iter().map(|(id, _)| id))),
            Any::from(strings(load)),
            Any::from(integer_list(
                records().map(|record| i64::from(record.storage)),
            )),
            Any::from(integer_list(records().map(|record| record.offset as i64))),
            Any::from(strings(records().map(|record| &record.dtype))),
            Any::from(integer_list(
                records().map(|record| record.shape.len() as i64),
            )),
            Any::from(integer_list(records().flat_map(|record| {
                record.shape.iter().map(|&extent| extent as i64)
            }))),
            Any::from(integer_list(records().flat_map(|record| {
                record.strides.iter().map(|&stride| stride as i64)
            }))),
        ])?)
    }

    /// The mapped storages, which the worker holds for its lifetime.
    /// The mapped storages, and the weights the worker still receives
    /// into some of them from other worlds ([`WeightKeys::receive`]),
    /// which it retains once they arrived ([`Receipt::retain`]).
    pub(crate) fn into_storages(self) -> (Vec<Arc<Storage>>, Option<Receipt>) {
        (self.storages, self.receipt)
    }
}

/// The rank's position in its parallel configuration, which keys its
/// manifest: the component configuration and the rank's place in it, its
/// role, whether a tower lies on a generation device, its expert union and
/// whether its tensor-parallel group divides the experts by expert.
/// The devices it runs on are not part of it: a rank at the same position
/// on other GPUs holds the same regions.
pub(crate) fn position(launch: &WorkerLaunch) -> Result<Vec<u8>> {
    let execution = &launch.execution;
    let union = launch.expert_parallel.as_ref().map(|union| {
        serde_json::json!({
            "rank": union.rank,
            "size": union.size,
            "attention_ranks": union.attention_ranks,
            "exchange": union.exchange,
        })
    });
    let components = serde_json::to_value(&launch.components)
        .map_err(|error| Error::Invalid(format!("encoding the components: {error}")))?;
    serde_json::to_vec(&serde_json::json!({
        "components": components,
        "rank": execution.rank,
        "world_size": execution.world_size,
        "role": execution.role,
        "generation_device": execution.generation_device.is_some(),
        "expert_union": union,
        "tensor_experts": launch.tensor_experts,
        "expert_order": launch.expert_order,
    }))
    .map_err(|error| Error::Invalid(format!("encoding the rank's position: {error}")))
}

/// The text of the retained host bytes `key` (a weight manifest); `None`
/// when the server retains no such bytes.
fn read_text(retained: &Retained, key: Key) -> Result<Option<String>> {
    let got = retained.call("finding retained weights", |client| {
        client.get(
            vec![key],
            Mode::Read,
            vec![Placement::Host { numa_node: None }],
        )
    })?;
    let (Some(record), Some((storage, descriptor))) =
        (got.tensors.first(), got.storages.into_iter().next())
    else {
        return Ok(None);
    };
    if !matches!(storage.placement, Placement::Host { .. }) || record.shape.len() != 1 {
        return Err(Error::Interface(
            "a retained weight manifest is not host bytes".into(),
        ));
    }
    let mut bytes = vec![0; record.shape[0] as usize];
    std::fs::File::from(descriptor)
        .read_exact_at(&mut bytes, record.offset)
        .map_err(|error| Error::transport_source("reading the retained weight manifest", error))?;
    String::from_utf8(bytes)
        .map(Some)
        .map_err(|_| Error::Interface("the retained weight manifest is not text".into()))
}

/// The immutable host record of a rank's current tensor identities and chunks.
fn manifest_record(
    key: Key,
    manifest: &WeightManifest,
) -> Result<(TensorRecord, StorageRecord, OwnedFd)> {
    let text = serde_json::to_string(manifest)
        .map_err(|error| Error::Invalid(format!("encoding the weight manifest: {error}")))?;
    Ok((
        TensorRecord {
            key,
            name: MANIFEST_NAME.into(),
            storage: 0,
            offset: 0,
            dtype: "uint8".into(),
            shape: vec![text.len() as u64],
            strides: vec![1],
            access: Access::ReadOnly,
        },
        StorageRecord {
            placement: Placement::Host { numa_node: None },
            bytes: text.len() as u64,
            segment: None,
            fabric: None,
        },
        host_bytes(text.as_bytes())?,
    ))
}

/// A host memfd holding `bytes`.
fn host_bytes(bytes: &[u8]) -> Result<OwnedFd> {
    let descriptor =
        rustix::fs::memfd_create("uniserve-weights-manifest", rustix::fs::MemfdFlags::CLOEXEC)
            .map_err(|error| {
                Error::transport_source(
                    "creating the manifest storage",
                    std::io::Error::from(error),
                )
            })?;
    let mut file = std::fs::File::from(descriptor);
    file.write_all(bytes)
        .map_err(|error| Error::transport_source("writing the manifest", error))?;
    Ok(file.into())
}

/// The tensors of the library's manifest text.
fn manifest_tensors(text: &str) -> Result<Vec<ManifestTensor>> {
    #[derive(serde::Deserialize)]
    struct Described {
        version: u64,
        tensors: Vec<ManifestTensor>,
    }
    let described: Described = serde_json::from_str(text).map_err(|error| {
        Error::Interface(format!(
            "the library's weight manifest is malformed: {error}"
        ))
    })?;
    if described.version != MANIFEST_VERSION {
        return Err(Error::Interface(format!(
            "the library writes weight manifest version {}, the worker binds {MANIFEST_VERSION}",
            described.version
        )));
    }
    Ok(described.tensors)
}
