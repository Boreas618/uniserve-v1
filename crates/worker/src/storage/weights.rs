//! The model weights a worker keeps at its host's Storage Server.
//!
//! Every device tensor of a loaded model, each parameter and buffer by its
//! fields (a dense tensor's data, or a quantized tensor's encoded fields),
//! is a read-only retained tensor. Its key is the digest of what determines
//! its bytes: the model fingerprint (`uniserve.loading.retained.
//! fingerprint`: the checkpoint, the normalized configuration, the weight
//! choices and the numerics of the code that loads and lays them out), the
//! worker's placement, the UUID of each GPU it addresses, and the field's
//! identity. A manifest, a host storage beside them, names each tensor's
//! representation, from which binding rebuilds it.
//!
//! A worker binds retained weights only when every worker that loads with
//! it (its worker group, or its expert union) finds all of its own:
//! loading is collective, so either all of them load or none does. A worker
//! that loads publishes afterwards: each torch storage of the loaded model
//! moves into retained device memory of its own, or, below the allocation
//! granularity, into one allocation shared with the other small ones, so
//! branches that loading coalesced stay adjacent; the mappings are then
//! downgraded to read-only, so any rewrite of a weight faults.

use std::collections::BTreeMap;
use std::io::Write;
use std::os::fd::{AsFd, OwnedFd};
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use sha2::{Digest, Sha256};
use tvm_ffi::Any;
use uniserve_cuda::{DeviceAllocation, DeviceId, Protection};
use uniserve_storage::protocol::Access;
use uniserve_storage::{Key, Mode, Placement, StorageRecord, TensorRecord};

use super::memory::{Storage, StorageView};
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

/// The keys of one worker's retained weights.
pub(crate) struct WeightKeys {
    key: Key,
    /// The GPU UUID of each device ordinal the worker addresses.
    gpus: BTreeMap<i32, [u8; 16]>,
}

/// Retained weights a worker found, mapped read-only.
pub(crate) struct Found {
    manifest: String,
    storages: Vec<Arc<Storage>>,
    /// Each field's identity and record, in manifest order.
    fields: Vec<(String, TensorRecord)>,
}

impl WeightKeys {
    /// The keys of weights of model `fingerprint` placed as `placement`
    /// describes on `devices`.
    pub(crate) fn new(
        retained: &Retained,
        fingerprint: &str,
        placement: &[u8],
        devices: &[DeviceId],
    ) -> Result<Self> {
        let mut gpus = BTreeMap::new();
        for device in devices {
            gpus.insert(device.0, retained.gpu(*device)?);
        }
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
        Ok(Self {
            key: Key(digest.finalize().into()),
            gpus,
        })
    }

    fn derive(&self, part: &[u8]) -> Key {
        let mut digest = Sha256::new();
        digest.update(self.key.0);
        digest.update((part.len() as u64).to_le_bytes());
        digest.update(part);
        Key(digest.finalize().into())
    }

    fn manifest(&self) -> Key {
        self.derive(b"manifest")
    }

    fn field(&self, id: &str) -> Key {
        self.derive(id.as_bytes())
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

    /// Map the weights a predecessor retained, when every one of them is
    /// retained; `None` otherwise.
    pub(crate) fn find(&self, retained: &Retained) -> Result<Option<Found>> {
        let started = std::time::Instant::now();
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
        let manifest = String::from_utf8(bytes)
            .map_err(|_| Error::Interface("the retained weight manifest is not text".into()))?;
        let ids = field_ids(&manifest)?;

        let keys: Vec<Key> = ids.iter().map(|id| self.field(id)).collect();
        let got = retained.call("finding retained weights", |client| {
            client.get(keys, Mode::Read)
        })?;
        if !got.missing.is_empty() {
            tracing::info!(
                missing = got.missing.len(),
                "retained weights are incomplete; loading"
            );
            return Ok(None);
        }
        if got.tensors.len() != ids.len() {
            return Err(Error::Interface(
                "retained weights disagree with their manifest".into(),
            ));
        }
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
                Placement::Host { .. } => Err(Error::Interface(
                    "a retained weight lies in host memory".into(),
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        retained.use_keys(
            std::iter::once(manifest_key).chain(got.tensors.iter().map(|tensor| tensor.key)),
        );
        tracing::info!(
            storages = storages.len(),
            bytes = storages
                .iter()
                .map(|storage| storage.bytes())
                .sum::<usize>(),
            elapsed_ms = started.elapsed().as_millis() as u64,
            "mapped the retained weights"
        );
        Ok(Some(Found {
            manifest,
            storages,
            fields: ids.into_iter().zip(got.tensors).collect(),
        }))
    }

    /// Publish a loaded model's weights (`model` is `rank.Loaded`): move
    /// them into retained memory, put them, and downgrade the mappings to
    /// read-only. Returns the retained storages, which the worker holds
    /// for its lifetime.
    pub(crate) fn publish(&self, retained: &Retained, model: &Handle) -> Result<Vec<Arc<Storage>>> {
        let layout = Handle::new(
            Interpreter::import(&format!("{RETAINED}.describe"))?.call_values(&[model.to_any()])?,
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
        drop(layout);

        // One record per field, naming its allocation and byte offset there.
        let mut tensors = Vec::with_capacity(ids.len() + 1);
        let mut position = 0;
        for (field, id) in ids.iter().enumerate() {
            let rank = ranks[field] as usize;
            let (allocation, base) = placed[field_storages[field] as usize];
            tensors.push(TensorRecord {
                key: self.field(id),
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
        let manifest_bytes = manifest.as_str().as_bytes();
        let manifest_memory = host_bytes(manifest_bytes)?;
        tensors.push(TensorRecord {
            key: self.manifest(),
            name: "weights manifest".into(),
            storage: allocations.len() as u32,
            offset: 0,
            dtype: "uint8".into(),
            shape: vec![manifest_bytes.len() as u64],
            strides: vec![1],
            access: Access::ReadOnly,
        });
        let mut storages = Vec::with_capacity(allocations.len() + 1);
        for (storage, descriptor, _) in &allocations {
            let device = match storage.device() {
                crate::ffi::MemoryKind::Cuda(device) => device,
                _ => {
                    return Err(Error::Invariant(
                        "retained weights are device memory".into(),
                    ));
                }
            };
            storages.push((
                StorageRecord {
                    placement: Placement::Device {
                        gpu: self.gpu(i64::from(device.0))?,
                    },
                    bytes: storage.bytes() as u64,
                },
                descriptor.as_fd(),
            ));
        }
        storages.push((
            StorageRecord {
                placement: Placement::Host { numa_node: None },
                bytes: manifest_bytes.len() as u64,
            },
            manifest_memory.as_fd(),
        ));
        let keys: Vec<Key> = tensors.iter().map(|tensor| tensor.key).collect();
        retained.call("publishing weights", |client| {
            client.put(tensors, &storages)
        })?;
        retained.use_keys(keys);

        // Published: no worker writes the weights again.
        let mut retained_storages = Vec::with_capacity(allocations.len());
        let mut total = 0;
        for (storage, _, _) in allocations {
            storage.protect(Protection::ReadOnly)?;
            total += storage.bytes();
            retained_storages.push(storage);
        }
        tracing::info!(
            storages = retained_storages.len(),
            bytes = total,
            "published the loaded weights"
        );
        Ok(retained_storages)
    }
}

impl Found {
    /// The library's `Retained` tensors over the mapped storages, which
    /// loading installs instead of reading the checkpoint.
    pub(crate) fn bind(&self) -> Result<Handle> {
        let views = self
            .storages
            .iter()
            .map(|storage| StorageView::whole(Arc::clone(storage) as Arc<dyn Backing>, DType::U8))
            .collect::<Result<Vec<_>>>()?;
        let tensors = views
            .iter()
            .map(|view| Ok(view.handle()?.to_any()))
            .collect::<Result<Vec<Any>>>()?;
        let records = || self.fields.iter().map(|(_, record)| record);
        let build = Interpreter::import(&format!("{RETAINED}.Retained.build"))?;
        Handle::new(build.call_values(&[
            Any::from(tvm_ffi::String::from(self.manifest.as_str())),
            Any::from(list(tensors)),
            Any::from(strings(self.fields.iter().map(|(id, _)| id))),
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

/// The field identities a manifest names, `<tensor id>/<field>`, in order.
fn field_ids(manifest: &str) -> Result<Vec<String>> {
    let invalid = |what: &str| Error::Interface(format!("the retained weight manifest {what}"));
    let manifest: serde_json::Value = serde_json::from_str(manifest)
        .map_err(|error| invalid(&format!("is not JSON: {error}")))?;
    if manifest["version"] != 1 {
        return Err(invalid("is of another version"));
    }
    let tensors = manifest["tensors"]
        .as_array()
        .ok_or_else(|| invalid("lists no tensors"))?;
    let mut ids = Vec::new();
    for tensor in tensors {
        let id = tensor["id"]
            .as_str()
            .ok_or_else(|| invalid("names a tensor without an id"))?;
        let fields = tensor["fields"]
            .as_array()
            .ok_or_else(|| invalid("names a tensor without fields"))?;
        for field in fields {
            let field = field
                .as_str()
                .ok_or_else(|| invalid("names a field that is not text"))?;
            ids.push(format!("{id}/{field}"));
        }
    }
    Ok(ids)
}
