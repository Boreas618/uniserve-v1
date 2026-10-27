//! The model weights a worker keeps at its host's Storage Server.
//!
//! Every device tensor of a loaded model, each parameter and buffer by its
//! fields (a dense tensor's data, or a quantized tensor's encoded fields),
//! is a read-only retained tensor. Its key is the digest of what determines
//! its bytes (`uniserve.loading.retained.fingerprints`), the worker's
//! placement, the UUID of each GPU it addresses, and the field's identity.
//! What determines the bytes is the fingerprint of the checkpoint source the
//! tensor is loaded from (its files, with the structure below), or, for a
//! tensor no source loads, the structure fingerprint: the normalized
//! configuration, the weight choices, the revisions of how the weights
//! derive from the checkpoint and what selects the providers whose layouts
//! encoded weights are stored in. A manifest, a host storage keyed by the
//! structure, names each tensor's source and representation, from which
//! binding rebuilds it.
//!
//! Loading is collective, so every worker that loads with this one (its
//! worker group, or its expert union) agrees on what to read from the
//! checkpoint: the tensors any of them misses, or everything when one of
//! them has no manifest ([`Load`]). A worker binds the rest. A change to one
//! checkpoint source's files thus reloads only that source's tensors. A
//! worker that read tensors publishes them afterwards: each torch storage
//! moves into retained device memory of its own, or, below the allocation
//! granularity, into one allocation shared with the other small ones, so
//! branches that loading coalesced stay adjacent; the mappings are then
//! downgraded to read-only, so any rewrite of a weight faults.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use cuda::{DeviceAllocation, DeviceId, Protection};
use sha2::{Digest, Sha256};
use storage::protocol::Access;
use storage::{Key, Mode, Placement, StorageRecord, TensorRecord};
use tvm_ffi::Any;

use super::allocation::{Storage, StorageView};
use super::retained::Retained;
use crate::config::WorkerLaunch;
use crate::ffi::{
    Backing, DType, Handle, Interpreter, integer_list, integers, list, strings, texts,
};
use crate::{Error, Result};

/// The library module that describes, publishes and binds retained weights.
const RETAINED: &str = "uniserve.loading.retained";

/// Byte alignment of each torch storage packed into a shared allocation,
/// the alignment torch's allocator gives a storage.
const PACK_ALIGNMENT: usize = 512;

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

/// What determines a rank's retained bytes besides its placement and GPUs,
/// as `uniserve.loading.retained.fingerprints` states it.
#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct Fingerprints {
    /// Everything but the checkpoint's files: it keys the manifest and the
    /// tensors no checkpoint source loads.
    pub structure: String,
    /// Each checkpoint source's fingerprint, by source name: it keys the
    /// tensors loaded from that source.
    pub sources: BTreeMap<String, String>,
    /// What determines the weights' values, regardless of the layouts their
    /// bytes take: it keys what forwards compute from them.
    pub values: String,
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

/// The keys of one worker's retained weights.
#[derive(Clone)]
pub(crate) struct WeightKeys {
    /// The base of the manifest's key and of the tensors no source loads.
    structure: Key,
    /// The base of each checkpoint source's tensors' keys.
    sources: BTreeMap<String, Key>,
    /// The GPU UUID of each device ordinal the worker addresses.
    gpus: BTreeMap<i32, [u8; 16]>,
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
    /// tensor of a stack streamed from host memory is missing (a stack's
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

/// One tensor a manifest names.
struct ManifestTensor {
    id: String,
    /// The checkpoint source the tensor is loaded from; empty for a tensor
    /// no source loads.
    source: String,
    /// Whether the tensor lies in the host region of a streamed stack.
    host: bool,
    fields: Vec<String>,
}

/// Weights a worker moved into retained memory and has not put yet: the
/// host regions of layers streamed from host memory join them
/// ([`Published::add_host`]) before [`Published::put`].
pub(crate) struct Published {
    keys: WeightKeys,
    /// The library's layout of the loaded model.
    layout: Handle,
    manifest: String,
    /// One record per field, naming its storage by index into `storages`.
    tensors: Vec<TensorRecord>,
    /// The retained storages, with their descriptors and placements.
    storages: Vec<(Arc<Storage>, OwnedFd, Placement)>,
}

/// Retained weights a worker found, mapped read-only.
pub(crate) struct Found {
    manifest: String,
    storages: Vec<Arc<Storage>>,
    /// Each field's identity and record, in manifest order, for every
    /// tensor whose fields are all retained.
    fields: Vec<(String, TensorRecord)>,
    /// The tensors the manifest names with a field missing.
    missing: BTreeSet<String>,
    /// Whether one of them lies in the host region of a streamed stack.
    host_missing: bool,
}

impl WeightKeys {
    /// The keys of weights of a model with `fingerprints` placed as
    /// `placement` describes on `devices`.
    pub(crate) fn new(
        retained: &Retained,
        fingerprints: &Fingerprints,
        placement: &[u8],
        devices: &[DeviceId],
    ) -> Result<Self> {
        let mut gpus = BTreeMap::new();
        for device in devices {
            gpus.insert(device.0, retained.gpu(*device)?);
        }
        let base = |fingerprint: &str| {
            let mut digest = Sha256::new();
            digest.update(b"uniserve-weights\0");
            for part in [fingerprint.as_bytes(), placement] {
                digest.update((part.len() as u64).to_le_bytes());
                digest.update(part);
            }
            for (device, gpu) in &gpus {
                digest.update(device.to_le_bytes());
                digest.update(gpu);
            }
            Key(digest.finalize().into())
        };
        Ok(Self {
            structure: base(&fingerprints.structure),
            sources: fingerprints
                .sources
                .iter()
                .map(|(source, fingerprint)| (source.clone(), base(fingerprint)))
                .collect(),
            gpus,
        })
    }

    fn derive(base: &Key, part: &[u8]) -> Key {
        let mut digest = Sha256::new();
        digest.update(base.0);
        digest.update((part.len() as u64).to_le_bytes());
        digest.update(part);
        Key(digest.finalize().into())
    }

    fn manifest(&self) -> Key {
        Self::derive(&self.structure, b"manifest")
    }

    /// The key of field `id` of a tensor loaded from checkpoint `source`
    /// (empty for none); `None` for a source the model does not have.
    fn field(&self, source: &str, id: &str) -> Option<Key> {
        let base = if source.is_empty() {
            &self.structure
        } else {
            self.sources.get(source)?
        };
        Some(Self::derive(base, id.as_bytes()))
    }

    /// The key of a field the worker publishes, whose source the model has.
    fn published_key(&self, source: &str, id: &str) -> Result<Key> {
        self.field(source, id).ok_or_else(|| {
            Error::Invariant(format!(
                "published weight {id} is loaded from checkpoint source {source}, \
                 which the model's fingerprints do not name"
            ))
        })
    }

    fn device(&self, gpu: &[u8; 16]) -> Result<DeviceId> {
        self.gpus
            .iter()
            .find(|(_, uuid)| *uuid == gpu)
            .map(|(device, _)| DeviceId(*device))
            .ok_or_else(|| {
                Error::Interface(
                    "a retained weight lies on a GPU the worker does not address".into(),
                )
            })
    }

    fn first_device(&self) -> Result<DeviceId> {
        self.gpus
            .keys()
            .next()
            .map(|device| DeviceId(*device))
            .ok_or_else(|| Error::Invariant("a worker retaining weights addresses a GPU".into()))
    }

    fn gpu(&self, device: i64) -> Result<[u8; 16]> {
        i32::try_from(device)
            .ok()
            .and_then(|device| self.gpus.get(&device))
            .copied()
            .ok_or_else(|| {
                Error::Invariant(format!(
                    "a loaded weight lies on device {device}, which the worker does not address"
                ))
            })
    }

    /// The retained manifest's text; `None` without one.
    fn read_manifest(&self, retained: &Retained) -> Result<Option<String>> {
        let manifest_key = self.manifest();
        let got = retained.call("finding retained weights", |client| {
            client.get(vec![manifest_key], Mode::Read)
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
            .map_err(|error| {
                Error::transport_source("reading the retained weight manifest", error)
            })?;
        String::from_utf8(bytes)
            .map(Some)
            .map_err(|_| Error::Interface("the retained weight manifest is not text".into()))
    }

    /// The moves that rename this worker's retained weights to their keys
    /// under `to`, the keys of the same tensors loaded from another
    /// checkpoint: the manifest's and every retained field's whose key
    /// changes.
    pub(crate) fn moves(&self, to: &WeightKeys, retained: &Retained) -> Result<Vec<(Key, Key)>> {
        let manifest = self
            .read_manifest(retained)?
            .ok_or_else(|| Error::State("the worker's weights have no retained manifest"))?;
        let tensors = manifest_tensors(&manifest)?.ok_or_else(|| {
            Error::Interface("the retained weight manifest is of another version".into())
        })?;
        let mut moves = Vec::new();
        if self.manifest() != to.manifest() {
            moves.push((self.manifest(), to.manifest()));
        }
        let mut fields = Vec::new();
        for tensor in &tensors {
            for field in &tensor.fields {
                let id = format!("{}/{field}", tensor.id);
                let (Some(old), Some(new)) = (
                    self.field(&tensor.source, &id),
                    to.field(&tensor.source, &id),
                ) else {
                    return Err(Error::Interface(format!(
                        "weight {id} is loaded from checkpoint source {}, which one of the \
                         checkpoints does not have",
                        tensor.source
                    )));
                };
                fields.push((old, new));
            }
        }
        // Only the fields the server retains move.
        let got = retained.call("finding retained weights", |client| {
            client.get(fields.iter().map(|(old, _)| *old).collect(), Mode::Read)
        })?;
        let held: BTreeSet<Key> = got.tensors.iter().map(|record| record.key).collect();
        moves.extend(
            fields
                .into_iter()
                .filter(|(old, new)| old != new && held.contains(old)),
        );
        Ok(moves)
    }

    /// Map the weights a predecessor retained: every tensor its manifest
    /// names whose fields are all retained. `None` without a manifest.
    pub(crate) fn find(&self, retained: &Retained) -> Result<Option<Found>> {
        let started = std::time::Instant::now();
        let manifest_key = self.manifest();
        let Some(manifest) = self.read_manifest(retained)? else {
            return Ok(None);
        };
        let Some(tensors) = manifest_tensors(&manifest)? else {
            tracing::info!("the retained weight manifest is of another version; loading");
            return Ok(None);
        };
        // Every field of every tensor, keyed by the tensor's source.
        let fields: Vec<(usize, String, Option<Key>)> = tensors
            .iter()
            .enumerate()
            .flat_map(|(index, tensor)| {
                tensor.fields.iter().map(move |field| {
                    let id = format!("{}/{field}", tensor.id);
                    let key = self.field(&tensor.source, &id);
                    (index, id, key)
                })
            })
            .collect();
        let keys: Vec<Key> = fields.iter().filter_map(|(_, _, key)| *key).collect();
        let got = retained.call("finding retained weights", |client| {
            client.get(keys, Mode::Read)
        })?;
        let mut records: HashMap<Key, TensorRecord> = got
            .tensors
            .into_iter()
            .map(|record| (record.key, record))
            .collect();
        let mut complete = vec![true; tensors.len()];
        for (index, _, key) in &fields {
            if key.is_none_or(|key| !records.contains_key(&key)) {
                complete[*index] = false;
            }
        }
        let mut found = Vec::new();
        for (index, id, key) in fields {
            if complete[index]
                && let Some(record) = key.and_then(|key| records.remove(&key))
            {
                found.push((id, record));
            }
        }
        let missing: BTreeSet<String> = tensors
            .iter()
            .zip(&complete)
            .filter(|(_, complete)| !**complete)
            .map(|(tensor, _)| tensor.id.clone())
            .collect();
        let host_missing = tensors
            .iter()
            .zip(&complete)
            .any(|(tensor, complete)| tensor.host && !complete);

        let storages = got
            .storages
            .into_iter()
            .map(|(record, descriptor)| match record.placement {
                Placement::Device { gpu } => Storage::import(
                    descriptor,
                    record.bytes as usize,
                    self.device(&gpu)?,
                    &[],
                    Protection::ReadOnly,
                ),
                // The host region of a layer stack streamed from host
                // memory, registered with the worker's first device.
                Placement::Host { .. } => {
                    Storage::import_host(descriptor, record.bytes as usize, self.first_device()?)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        retained.use_keys(
            std::iter::once(manifest_key).chain(found.iter().map(|(_, record)| record.key)),
        );
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
            manifest,
            storages,
            fields: found,
            missing,
            host_missing,
        }))
    }

    /// Move a loaded model's device weights into retained memory (`model`
    /// is `rank.Loaded`): every tensor, or only the tensors `only` names,
    /// which the worker read while it bound the others. The projection
    /// matrices of the stacks under `host_paths` stream from host memory:
    /// their host regions join the published weights once allocated
    /// ([`Published::add_host`]).
    pub(crate) fn publish(
        &self,
        model: &Handle,
        host_paths: &[String],
        only: Option<&BTreeSet<String>>,
    ) -> Result<Published> {
        let layout = Handle::new(
            Interpreter::import(&format!("{RETAINED}.describe"))?.call_values(&[
                model.to_any(),
                Any::from(strings(host_paths)),
                only.map(|names| Any::from(strings(names)))
                    .unwrap_or_default(),
            ])?,
        )?;
        let method = |name: &str| Interpreter::import(&format!("{RETAINED}.Layout.{name}"));
        let numbers = |name: &str| -> Result<Vec<i64>> {
            integers(method(name)?.call_values(&[layout.to_any()])?)
                .map_err(|error| Error::numerical(name, &error))
        };
        let names = |name: &str| -> Result<Vec<String>> {
            texts(method(name)?.call_values(&[layout.to_any()])?)
                .map_err(|error| Error::numerical(name, &error))
        };
        let manifest: tvm_ffi::String = method("manifest")?.call_as(&[layout.view()])?;
        let devices = numbers("storage_devices")?;
        let sizes = numbers("storage_bytes")?;
        let ids = names("field_ids")?;
        let sources = names("field_sources")?;
        let field_storages = numbers("field_storages")?;
        let offsets = numbers("field_offsets")?;
        let dtypes = names("field_dtypes")?;
        let ranks = numbers("field_ranks")?;
        let shapes = numbers("field_shapes")?;
        let strides = numbers("field_strides")?;

        // Where each torch storage moves: an allocation of its own at
        // offset zero, or its device's shared allocation of small storages.
        let mut granularity = BTreeMap::new();
        let mut packs: BTreeMap<i64, usize> = BTreeMap::new();
        let mut packed = vec![None; sizes.len()];
        for (index, (&device, &size)) in devices.iter().zip(&sizes).enumerate() {
            let page = match granularity.get(&device) {
                Some(&page) => page,
                None => {
                    let page = DeviceAllocation::granularity(DeviceId(device as i32))?;
                    granularity.insert(device, page);
                    page
                }
            };
            if (size as usize) < page {
                let total = packs.entry(device).or_default();
                let offset = total.next_multiple_of(PACK_ALIGNMENT);
                *total = offset + size as usize;
                packed[index] = Some(offset);
            }
        }

        let publish = method("publish")?;
        // Allocations in creation order, with their descriptors and views.
        let mut allocations: Vec<(Arc<Storage>, OwnedFd, StorageView)> = Vec::new();
        let mut pack_index: BTreeMap<i64, usize> = BTreeMap::new();
        // Each torch storage's allocation and byte offset in it.
        let mut placed = Vec::with_capacity(sizes.len());
        for (index, (&device, &size)) in devices.iter().zip(&sizes).enumerate() {
            let target = DeviceId(device as i32);
            let (allocation, offset) = match packed[index] {
                Some(offset) => {
                    let allocation = match pack_index.get(&device) {
                        Some(&allocation) => allocation,
                        None => {
                            allocations.push(allocate(target, packs[&device])?);
                            pack_index.insert(device, allocations.len() - 1);
                            allocations.len() - 1
                        }
                    };
                    (allocation, offset)
                }
                None => {
                    allocations.push(allocate(target, size as usize)?);
                    (allocations.len() - 1, 0)
                }
            };
            let memory = allocations[allocation].2.handle()?.to_any();
            publish.call_values(&[
                layout.to_any(),
                Any::from(index as i64),
                memory,
                Any::from(offset as i64),
            ])?;
            placed.push((allocation, offset));
        }

        // One record per field, naming its allocation and byte offset there.
        let mut tensors = Vec::with_capacity(ids.len() + 1);
        let mut position = 0;
        for (field, id) in ids.iter().enumerate() {
            let rank = ranks[field] as usize;
            let (allocation, base) = placed[field_storages[field] as usize];
            tensors.push(TensorRecord {
                key: self.published_key(&sources[field], id)?,
                name: format!("weights {id}"),
                storage: allocation as u32,
                offset: (base as i64 + offsets[field]) as u64,
                dtype: dtypes[field].clone(),
                shape: shapes[position..position + rank]
                    .iter()
                    .map(|&extent| extent as u64)
                    .collect(),
                strides: strides[position..position + rank]
                    .iter()
                    .map(|&stride| stride as u64)
                    .collect(),
                access: Access::ReadOnly,
            });
            position += rank;
        }
        let storages = allocations
            .into_iter()
            .map(|(storage, descriptor, _)| {
                let crate::ffi::MemoryKind::Cuda(device) = storage.device() else {
                    return Err(Error::Invariant(
                        "retained weights are device memory".into(),
                    ));
                };
                let gpu = self.gpu(i64::from(device.0))?;
                Ok((storage, descriptor, Placement::Device { gpu }))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Published {
            keys: self.clone(),
            layout,
            manifest: manifest.as_str().to_owned(),
            tensors,
            storages,
        })
    }
}

impl Published {
    /// Add the host region of the layer stack under `path`, which
    /// `storage` holds, before the stack is bound to its slots (`model` is
    /// `rank.Loaded`).
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
        let ids = texts(column(0)?).map_err(invalid)?;
        let offsets = integers(column(1)?).map_err(invalid)?;
        let dtypes = texts(column(2)?).map_err(invalid)?;
        let ranks = integers(column(3)?).map_err(invalid)?;
        let shapes = integers(column(4)?).map_err(invalid)?;
        let strides = integers(column(5)?).map_err(invalid)?;
        let sources = texts(column(6)?).map_err(invalid)?;

        let index = self.storages.len() as u32;
        let mut position = 0;
        for (field, id) in ids.iter().enumerate() {
            let rank = ranks[field] as usize;
            self.tensors.push(TensorRecord {
                key: self.keys.published_key(&sources[field], id)?,
                name: format!("weights {id}"),
                storage: index,
                offset: offsets[field] as u64,
                dtype: dtypes[field].clone(),
                shape: shapes[position..position + rank]
                    .iter()
                    .map(|&extent| extent as u64)
                    .collect(),
                strides: strides[position..position + rank]
                    .iter()
                    .map(|&stride| stride as u64)
                    .collect(),
                access: Access::ReadOnly,
            });
            position += rank;
        }
        self.storages
            .push((storage, descriptor, Placement::Host { numa_node: None }));
        Ok(())
    }

    /// Put the weights and their manifest, and make every retained mapping
    /// read-only: no worker writes the weights again. Returns the retained
    /// storages, which the worker holds for its lifetime.
    pub(crate) fn put(self, retained: &Retained) -> Result<Vec<Arc<Storage>>> {
        let Self {
            keys,
            layout,
            manifest,
            mut tensors,
            storages,
        } = self;
        drop(layout);
        let manifest_bytes = manifest.as_bytes();
        let manifest_memory = host_bytes(manifest_bytes)?;
        tensors.push(TensorRecord {
            key: keys.manifest(),
            name: "weights manifest".into(),
            storage: storages.len() as u32,
            offset: 0,
            dtype: "uint8".into(),
            shape: vec![manifest_bytes.len() as u64],
            strides: vec![1],
            access: Access::ReadOnly,
        });
        let mut records: Vec<(StorageRecord, std::os::fd::BorrowedFd<'_>)> = storages
            .iter()
            .map(|(storage, descriptor, placement)| {
                (
                    StorageRecord {
                        placement: *placement,
                        bytes: storage.bytes() as u64,
                    },
                    descriptor.as_fd(),
                )
            })
            .collect();
        records.push((
            StorageRecord {
                placement: Placement::Host { numa_node: None },
                bytes: manifest_bytes.len() as u64,
            },
            manifest_memory.as_fd(),
        ));
        let keys: Vec<Key> = tensors.iter().map(|tensor| tensor.key).collect();
        retained.call("publishing weights", |client| client.put(tensors, &records))?;
        retained.use_keys(keys);
        drop(records);

        let mut published = Vec::with_capacity(storages.len());
        let (mut device_bytes, mut host_bytes) = (0, 0);
        for (storage, _, placement) in storages {
            storage.protect(Protection::ReadOnly)?;
            match placement {
                Placement::Device { .. } => device_bytes += storage.bytes(),
                Placement::Host { .. } => host_bytes += storage.bytes(),
            }
            published.push(storage);
        }
        tracing::info!(
            storages = published.len(),
            device_bytes,
            host_bytes,
            "published the loaded weights"
        );
        Ok(published)
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
        let bound: Vec<&(String, TensorRecord)> = self
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
    pub(crate) fn into_storages(self) -> Vec<Arc<Storage>> {
        self.storages
    }
}

/// What of a launch places the rank's weights: the component placement and
/// the rank's place in it, its role, its devices and its expert union.
pub(crate) fn placement(launch: &WorkerLaunch) -> Result<Vec<u8>> {
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
        .map_err(|error| Error::Invalid(format!("encoding the component placement: {error}")))?;
    serde_json::to_vec(&serde_json::json!({
        "components": components,
        "rank": execution.rank,
        "world_size": execution.world_size,
        "role": execution.role,
        "device": execution.device.to_string(),
        "generation_device": execution.generation_device.map(|device| device.to_string()),
        "expert_union": union,
    }))
    .map_err(|error| Error::Invalid(format!("encoding the weight placement: {error}")))
}

/// Retained device memory for `bytes` on `device`, with its descriptor and
/// a byte view of the whole allocation.
fn allocate(device: DeviceId, bytes: usize) -> Result<(Arc<Storage>, OwnedFd, StorageView)> {
    let (storage, descriptor) = Storage::retainable(device, bytes, &[])?;
    let view = StorageView::whole(Arc::clone(&storage) as Arc<dyn Backing>, DType::U8)?;
    Ok((storage, descriptor, view))
}

/// A host memfd holding `bytes`.
fn host_bytes(bytes: &[u8]) -> Result<OwnedFd> {
    let descriptor =
        rustix::fs::memfd_create("uniserve-weights-manifest", rustix::fs::MemfdFlags::CLOEXEC)
            .map_err(|error| {
                Error::transport_source("creating the manifest memory", std::io::Error::from(error))
            })?;
    let mut file = std::fs::File::from(descriptor);
    file.write_all(bytes)
        .map_err(|error| Error::transport_source("writing the manifest", error))?;
    Ok(file.into())
}

/// The version of the manifest `uniserve.loading.retained` writes; a
/// manifest of another version names no tensor this worker binds.
const MANIFEST_VERSION: u64 = 4;

/// The tensors a manifest names, in order; `None` for a manifest of
/// another version.
fn manifest_tensors(manifest: &str) -> Result<Option<Vec<ManifestTensor>>> {
    let invalid = |what: &str| Error::Interface(format!("the retained weight manifest {what}"));
    let manifest: serde_json::Value = serde_json::from_str(manifest)
        .map_err(|error| invalid(&format!("is not JSON: {error}")))?;
    if manifest["version"] != MANIFEST_VERSION {
        return Ok(None);
    }
    let tensors = manifest["tensors"]
        .as_array()
        .ok_or_else(|| invalid("lists no tensors"))?;
    let mut named = Vec::with_capacity(tensors.len());
    for tensor in tensors {
        let id = tensor["id"]
            .as_str()
            .ok_or_else(|| invalid("names a tensor without an id"))?;
        let source = tensor["source"]
            .as_str()
            .ok_or_else(|| invalid("names a tensor without a source"))?;
        let fields = tensor["fields"]
            .as_array()
            .ok_or_else(|| invalid("names a tensor without fields"))?
            .iter()
            .map(|field| {
                field
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| invalid("names a field that is not text"))
            })
            .collect::<Result<Vec<_>>>()?;
        named.push(ManifestTensor {
            id: id.to_owned(),
            source: source.to_owned(),
            host: tensor["host"].as_bool().unwrap_or(false),
            fields,
        });
    }
    Ok(Some(named))
}
