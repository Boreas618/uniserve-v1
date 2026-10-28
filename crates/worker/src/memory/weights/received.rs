//! Weights a worker receives from the other worlds on its host.
//!
//! A worker whose position has no manifest of its own, such as a rank of a
//! world that changes the deployment's parallel configuration, finds the
//! weights the deployment's other worlds retain on the host's GPUs: every
//! manifest the Storage Server retains, of this version and of the same
//! fingerprints, names storages whose chunks lie on the GPUs that hold
//! them ([`WeightKeys::sources`]). The worker plans the storages its own
//! weights take without reading the checkpoint ([`WeightKeys::plan`] over
//! an unread model), and the ranks that load together compute one copy
//! manifest from those sources to all of their storages
//! (`storage::manifest`), each verifying that every rank computed the same
//! digest before any byte moves ([`WeightKeys::receive`]).
//!
//! Each rank then allocates its storages at fixed addresses: a chunk whose
//! key a source holds on its GPU is mapped read-only, every other chunk is
//! created and retained exclusive under a receiving key, which the Transfer
//! Server writes the manifest's copies into as one task at the
//! `Background` priority, across the links it measured. Padding the
//! manifest names is zeroed on the rank's GPU, and the tensors of the
//! regions no source holds are read from the checkpoint as a load of only
//! those tensors. The rank binds its model to the storages while the bytes
//! arrive; once the transfer ended ([`Receipt::retain`]), the received
//! chunks are retained read-only under their keys, like chunks a load
//! writes, the receiving keys are released and the storages become
//! read-only.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::os::fd::{AsFd, OwnedFd};
use std::sync::Arc;

use cuda::{Protection, Stream};
use sha2::{Digest, Sha256};
use storage::manifest::{self, Gpu, Manifest, Space};
use storage::protocol::{Access, LinkBandwidth, ListedTensor, Priority, Region, Span};
use storage::{Key, Mode, Placement, StorageRecord, TensorRecord};

use super::{
    Found, MANIFEST_NAME, MANIFEST_VERSION, Planned, WeightKeys, WeightManifest, field_record,
    host_bytes, read_text, storage_name,
};
use crate::ffi::{Backing, Interpreter};
use crate::memory::allocation::Storage;
use crate::memory::retained::Retained;
use crate::parallel::host::HostGroup;
use crate::{Error, Result};

/// The weights the deployment's worlds retain on the host: their ranks'
/// storages, each on a GPU that holds all of its chunks, and the links the
/// Transfer Server measured between the GPUs.
pub(crate) struct Sources {
    world: manifest::World,
    links: Vec<LinkBandwidth>,
}

/// Weights a worker receives into chunks it created, until the transfer
/// ends and they are retained read-only ([`Receipt::retain`]).
pub(crate) struct Receipt {
    /// The transfer task writing the chunks; `None` when the manifest
    /// copies nothing to the rank.
    task: Option<u64>,
    /// Each created chunk's read-only record under its key, with its
    /// descriptor.
    created: Vec<(TensorRecord, StorageRecord, OwnedFd)>,
    /// The keys the created chunks are retained under while they receive.
    receiving: Vec<Key>,
    /// The chunks mapped from other worlds, which the rank retains too.
    shared: Vec<Key>,
    /// The rank's manifest and its key, retained beside the chunks when no
    /// publication of tensors the rank read retains it.
    manifest: (Key, String),
    /// The rank's storages, read-only once the bytes arrived.
    storages: Vec<Arc<Storage>>,
}

impl WeightKeys {
    /// The weights the other worlds of the worker's deployment retain on
    /// the host of the same model, which this worker may receive: `None`
    /// when they retain none, or no Transfer Server runs to copy them.
    pub(crate) fn sources(&self, retained: &Retained) -> Result<Option<Sources>> {
        // A rank whose weights span devices (a generation tower on a device
        // of its own) loads them.
        if self.devices.len() != 1 {
            return Ok(None);
        }
        let listing = retained.call("listing the storage server", |client| client.list())?;
        let Some(links) = listing
            .transfer
            .map(|transfer| transfer.links)
            .filter(|links| !links.is_empty())
        else {
            return Ok(None);
        };
        let deployment = retained.scope().deployment.as_str();
        let ours = |tensor: &ListedTensor| {
            tensor.access == Access::ReadOnly
                && tensor
                    .retainers
                    .iter()
                    .any(|retainer| retainer.scope.deployment == deployment)
        };
        // The GPUs holding each chunk, and the manifests.
        let mut held: HashMap<Key, BTreeSet<Gpu>> = HashMap::new();
        let mut manifests = BTreeSet::new();
        for tensor in listing.tensors.iter().filter(|tensor| ours(tensor)) {
            match tensor.placement {
                Placement::Device { gpu } => {
                    held.entry(tensor.key).or_default().insert(gpu);
                }
                Placement::Host { .. } if tensor.name == MANIFEST_NAME => {
                    manifests.insert(tensor.key);
                }
                Placement::Host { .. } => {}
            }
        }

        let digests = self.fingerprint_digests();
        // Each source storage once per GPU, whatever manifests name it.
        let mut by_gpu: BTreeMap<Gpu, BTreeMap<Vec<u8>, manifest::Storage>> = BTreeMap::new();
        for key in manifests {
            let Some(text) = read_text(retained, key)? else {
                continue;
            };
            // A manifest of another version or format names no source.
            let Ok(manifest) = serde_json::from_str::<WeightManifest>(&text) else {
                continue;
            };
            if manifest.version != MANIFEST_VERSION {
                continue;
            }
            for storage in &manifest.storages {
                // Only the fields of sources with this worker's digests
                // hold the bytes it needs.
                let same = storage.device.is_some()
                    && storage.storage.layout.fields.iter().all(|field| {
                        manifest.fingerprints.get(&field.source) == digests.get(&field.source)
                    });
                if !same {
                    continue;
                }
                let mut gpus: Option<BTreeSet<Gpu>> = None;
                for chunk in &storage.storage.chunks {
                    let on = held.get(&chunk.key).cloned().unwrap_or_default();
                    gpus = Some(match gpus {
                        None => on,
                        Some(gpus) => gpus.intersection(&on).copied().collect(),
                    });
                }
                let identity = serde_json::to_vec(&storage.storage).map_err(|error| {
                    Error::Invalid(format!("encoding a source storage: {error}"))
                })?;
                for gpu in gpus.unwrap_or_default() {
                    by_gpu
                        .entry(gpu)
                        .or_default()
                        .insert(identity.clone(), storage.storage.clone());
                }
            }
        }
        // One source rank per GPU, its storages in a fixed order, so that
        // every worker that lists the same server plans the same manifest.
        let ranks: Vec<manifest::Rank> = by_gpu
            .into_iter()
            .map(|(gpu, storages)| manifest::Rank {
                gpu: Some(gpu),
                storages: storages.into_values().collect(),
            })
            .collect();
        if ranks.is_empty() {
            return Ok(None);
        }
        Ok(Some(Sources {
            world: manifest::World { ranks },
            links,
        }))
    }

    /// Receive this rank's `planned` storages from `sources`, together
    /// with the ranks of `agreement` (the ranks that load together, every
    /// one of which calls this): the ranks plan one manifest from the
    /// sources to all their storages and verify they computed the same
    /// digest; then each maps the chunks a source holds on its GPU,
    /// creates the rest, submits the copies into them to the Transfer
    /// Server and zeroes the padding the manifest names. Returns the
    /// storages as found weights: the tensors of the regions no source
    /// holds are missing, for the rank to read from the checkpoint, and
    /// the receipt retains the received chunks once they arrived.
    pub(crate) fn receive(
        &self,
        retained: &Retained,
        planned: Planned,
        sources: &Sources,
        agreement: &HostGroup,
    ) -> Result<Found> {
        let started = std::time::Instant::now();
        let Planned {
            layout,
            tensors,
            storages,
            ..
        } = planned;
        drop(layout);
        // The unread model's device memory returns before the storages are
        // allocated: its modules may hold each other in cycles, which only
        // the cycle collector frees, and the caching allocator keeps freed
        // blocks until emptied.
        Interpreter::import("gc.collect")?.call_values(&[])?;
        Interpreter::import("torch.cuda.empty_cache")?.call_values(&[])?;
        let (device, gpu) = self.device(0)?;
        if storages.iter().any(|storage| storage.device != Some(0)) {
            return Err(Error::Unsupported(
                "a rank receives weights only on its one GPU".into(),
            ));
        }

        // Every rank's storages as a target rank on its GPU.
        let own = manifest::Rank {
            gpu: Some(gpu),
            storages: storages
                .iter()
                .map(|storage| storage.storage.clone())
                .collect(),
        };
        let encoded = serde_json::to_vec(&own)
            .map_err(|error| Error::Invalid(format!("encoding a target rank: {error}")))?;
        let freed = started.elapsed();
        let targets = agreement
            .all_gather_bytes(&encoded)?
            .iter()
            .map(|bytes| {
                serde_json::from_slice::<manifest::Rank>(bytes)
                    .map_err(|error| Error::Interface(format!("a rank's target storages: {error}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let rank = agreement.index();
        let request = manifest::Request {
            source: sources.world.clone(),
            gpus: targets.iter().filter_map(|target| target.gpu).collect(),
            target: manifest::World { ranks: targets },
            links: sources.links.clone(),
        };
        let gathered = started.elapsed();
        let manifest =
            Manifest::plan(&request).map_err(|error| Error::Interface(error.to_string()))?;
        // Every participating rank verifies the digest before any byte
        // moves: the ranks computed the same manifest from the same inputs.
        let planned = started.elapsed();
        let digests = agreement.all_gather_bytes(&manifest.digest)?;
        if !manifest.verify(&manifest.digest)
            || digests
                .iter()
                .any(|digest| digest.as_slice() != manifest.digest)
        {
            return Err(Error::Invariant(
                "the ranks receiving weights computed different copy manifests".into(),
            ));
        }
        let verified = started.elapsed();
        let own_rank = rank as u32;

        // The tensors of regions no source holds, and of fields that move
        // only through a conversion of their encoding, are read.
        let mut missing: BTreeSet<String> = manifest
            .missing
            .iter()
            .filter(|missing| missing.rank == own_rank)
            .map(|missing| tensor_of(&missing.tensor).to_owned())
            .collect();
        for decode in &manifest.decodes {
            let staging = &manifest.stagings[decode.staging as usize];
            if staging.rank == own_rank {
                missing.extend(
                    decode
                        .fields
                        .iter()
                        .map(|(_, field)| tensor_of(field).to_owned()),
                );
            }
        }

        // Map the chunks a source holds on this GPU, create the rest.
        let mapped: BTreeSet<(u32, u32)> = manifest
            .mappings
            .iter()
            .filter(|mapping| mapping.rank == own_rank)
            .map(|mapping| (mapping.storage, mapping.chunk))
            .collect();
        let mapped_keys: Vec<Key> = manifest
            .mappings
            .iter()
            .filter(|mapping| mapping.rank == own_rank)
            .map(|mapping| mapping.key)
            .collect();
        let location = Placement::Device { gpu };
        let got = retained.call("finding retained weight chunks", |client| {
            client.get(mapped_keys.clone(), Mode::Read, vec![location])
        })?;
        let mut descriptors: HashMap<Key, OwnedFd> = HashMap::new();
        for record in &got.tensors {
            if let Some((_, descriptor)) = got.storages.get(record.storage as usize) {
                let descriptor = descriptor.try_clone().map_err(|error| {
                    Error::transport_source("duplicating a retained descriptor", error)
                })?;
                descriptors.insert(record.key, descriptor);
            }
        }
        let mut mapped_storages = Vec::with_capacity(storages.len());
        let mut created = Vec::new();
        let mut receiving_records = Vec::new();
        let mut receiving: HashMap<Key, Key> = HashMap::new();
        let mut shared = Vec::new();
        // Each storage's created byte ranges, which fills may write.
        let mut writable: Vec<Vec<(u64, u64)>> = Vec::with_capacity(storages.len());
        for (index, retained_storage) in storages.iter().enumerate() {
            let chunks = &retained_storage.storage.chunks;
            let lengths: Vec<usize> = chunks.iter().map(|chunk| chunk.length as usize).collect();
            let storage = Storage::segmented(device, &lengths, &[])?;
            let mut ranges = Vec::new();
            for (segment, chunk) in chunks.iter().enumerate() {
                if mapped.contains(&(index as u32, segment as u32)) {
                    let descriptor = descriptors.get(&chunk.key).ok_or_else(|| {
                        Error::State("a chunk the manifest maps is no longer retained")
                    })?;
                    storage.import_segment_read_only(segment, descriptor)?;
                    shared.push(chunk.key);
                    continue;
                }
                let descriptor = storage.create_segment(segment)?;
                let into = receiving_key(retained.world(), &chunk.key);
                if receiving.insert(chunk.key, into).is_some() {
                    return Err(Error::Unsupported(format!(
                        "the rank holds two chunks of one key in {}",
                        storage_name(retained_storage)
                    )));
                }
                let name = format!("weights {} chunk {segment}", storage_name(retained_storage));
                let record = StorageRecord {
                    placement: location,
                    bytes: chunk.length,
                    segment: None,
                };
                let tensor = TensorRecord {
                    key: chunk.key,
                    name,
                    storage: 0,
                    offset: 0,
                    dtype: "uint8".into(),
                    shape: vec![chunk.length],
                    strides: vec![1],
                    access: Access::ReadOnly,
                };
                receiving_records.push((
                    TensorRecord {
                        key: into,
                        name: format!("{} receiving", tensor.name),
                        access: Access::Exclusive,
                        ..tensor.clone()
                    },
                    record,
                    descriptor.try_clone().map_err(|error| {
                        Error::transport_source("duplicating a chunk descriptor", error)
                    })?,
                ));
                created.push((tensor, record, descriptor));
                ranges.push((chunk.offset, chunk.offset + chunk.length));
            }
            mapped_storages.push(storage);
            writable.push(ranges);
        }

        let allocated = started.elapsed();
        // The created chunks receive under their receiving keys: the
        // Transfer Server writes only exclusive tensors.
        if !receiving_records.is_empty() {
            let tensors: Vec<TensorRecord> = receiving_records
                .iter()
                .enumerate()
                .map(|(index, (tensor, _, _))| TensorRecord {
                    storage: index as u32,
                    ..tensor.clone()
                })
                .collect();
            let records: Vec<(StorageRecord, std::os::fd::BorrowedFd<'_>)> = receiving_records
                .iter()
                .map(|(_, record, descriptor)| (*record, descriptor.as_fd()))
                .collect();
            retained.call("retaining receiving weight chunks", |client| {
                client.put(tensors, &records)
            })?;
        }
        drop(receiving_records);

        // The chunks the rank created start zeroed, the padding the
        // manifest names among them.
        let registered = started.elapsed();
        zero_created(&mapped_storages, &writable, device)?;
        let zeroed = started.elapsed();

        // The copies into this rank's storages, as one transfer task;
        // copies through staging spaces move encoded fields, which the rank
        // reads instead.
        let mut copies = manifest.clone();
        copies.copies.retain(|copy| {
            copy.target.rank == own_rank
                && matches!(copy.source.space, Space::Storage { .. })
                && matches!(copy.target.space, Space::Storage { .. })
        });
        let spans = copies
            .spans(&request)
            .map(|span| {
                let mut span = span.map_err(|error| Error::Interface(error.to_string()))?;
                if let Region::Tensor { key, .. } = &mut span.destination {
                    *key = *receiving.get(key).ok_or_else(|| {
                        Error::Invariant("a copy writes a chunk the rank did not create".into())
                    })?;
                }
                Ok(span)
            })
            .collect::<Result<Vec<_>>>()?;
        let bytes: u64 = spans.iter().map(Span::copied).sum();
        let spans_count = spans.len();
        let listed = started.elapsed();
        let task = if spans.is_empty() {
            None
        } else {
            Some(retained.transfer(spans, Priority::Background, retained.world())?)
        };

        // The manifest the rank retains for its position, and the fields
        // it binds: all but those of the tensors it reads.
        let submitted = started.elapsed();
        let manifest_text = serde_json::to_string(&WeightManifest {
            version: MANIFEST_VERSION,
            fingerprints: self.fingerprint_digests(),
            tensors,
            storages: storages.clone(),
        })
        .map_err(|error| Error::Invalid(format!("encoding the weight manifest: {error}")))?;
        let mut fields = Vec::new();
        for (index, retained_storage) in storages.iter().enumerate() {
            for field in &retained_storage.storage.layout.fields {
                if !missing.contains(tensor_of(&field.tensor)) {
                    fields.push((field.tensor.clone(), field_record(index as u32, field)));
                }
            }
        }
        // The receiving keys stay retained, and booked, until the bytes
        // arrived ([`Receipt::retain`]).
        retained.use_keys(shared.iter().copied().chain(receiving.values().copied()));
        tracing::info!(
            rank,
            mapped = shared.len(),
            created = created.len(),
            copied_bytes = bytes,
            spans = spans_count,
            missing = missing.len(),
            digest = %super::hex(&manifest.digest),
            // Each phase's end, from the start: the unread model freed,
            // the ranks' storages gathered, the manifest planned and
            // verified, the chunks mapped and created, the receiving keys
            // retained, the padding zeroed, the spans listed and the
            // transfer submitted.
            freed_ms = freed.as_millis() as u64,
            gathered_ms = gathered.as_millis() as u64,
            planned_ms = planned.as_millis() as u64,
            verified_ms = verified.as_millis() as u64,
            allocated_ms = allocated.as_millis() as u64,
            registered_ms = registered.as_millis() as u64,
            zeroed_ms = zeroed.as_millis() as u64,
            listed_ms = listed.as_millis() as u64,
            submitted_ms = submitted.as_millis() as u64,
            "receiving the rank's weights from the host's other worlds"
        );
        Ok(Found {
            manifest: manifest_text.clone(),
            storages: mapped_storages.clone(),
            fields,
            missing,
            host_missing: false,
            kept: storages,
            receipt: Some(Receipt {
                task,
                created,
                receiving: receiving.into_values().collect(),
                shared,
                manifest: (self.manifest, manifest_text),
                storages: mapped_storages,
            }),
        })
    }
}

impl Receipt {
    /// Wait until the received bytes arrived, then retain the created
    /// chunks read-only under their keys, and the rank's manifest unless
    /// `published` (a publication of the tensors the rank read retains
    /// it), release the receiving keys and make the storages read-only.
    pub(crate) fn retain(self, retained: &Retained, published: bool) -> Result<()> {
        let started = std::time::Instant::now();
        if let Some(task) = self.task {
            retained.transferred(task)?;
        }
        let waited = started.elapsed();
        let manifest_memory = host_bytes(self.manifest.1.as_bytes())?;
        let mut tensors = Vec::with_capacity(self.created.len() + 1);
        let mut records = Vec::with_capacity(self.created.len() + 1);
        for (index, (tensor, record, descriptor)) in self.created.iter().enumerate() {
            tensors.push(TensorRecord {
                storage: index as u32,
                ..tensor.clone()
            });
            records.push((*record, descriptor.as_fd()));
        }
        if !published {
            tensors.push(TensorRecord {
                key: self.manifest.0,
                name: MANIFEST_NAME.into(),
                storage: records.len() as u32,
                offset: 0,
                dtype: "uint8".into(),
                shape: vec![self.manifest.1.len() as u64],
                strides: vec![1],
                access: Access::ReadOnly,
            });
            records.push((
                StorageRecord {
                    placement: Placement::Host { numa_node: None },
                    bytes: self.manifest.1.len() as u64,
                    segment: None,
                },
                manifest_memory.as_fd(),
            ));
        }
        let used: Vec<Key> = tensors
            .iter()
            .map(|tensor| tensor.key)
            .chain(self.shared.iter().copied())
            .collect();
        // The receiving keys end before the same storage is retained under
        // the chunks' keys: the GPU's ledger books each storage the server
        // retains, and the receiving storages' booking gives way to the
        // weights' rather than counting the storage twice.
        retained.forget(&self.receiving)?;
        if !tensors.is_empty() {
            retained.call("retaining received weights", |client| {
                client.put(tensors, &records)
            })?;
        }
        retained.use_keys(used);
        for storage in &self.storages {
            storage.protect(Protection::ReadOnly)?;
        }
        tracing::info!(
            chunks = self.created.len(),
            waited_ms = waited.as_millis() as u64,
            "retained the received weights"
        );
        Ok(())
    }
}

/// The tensor a field identity (`<tensor id>/<field>`) belongs to.
fn tensor_of(field: &str) -> &str {
    field.rsplit_once('/').map_or(field, |(tensor, _)| tensor)
}

/// The key a chunk of `key` is retained under in world `world` while it
/// receives its bytes.
fn receiving_key(world: u64, key: &Key) -> Key {
    let mut digest = Sha256::new();
    digest.update(b"uniserve-weights-receiving\0");
    digest.update(world.to_le_bytes());
    digest.update(key.0);
    Key(digest.finalize().into())
}

/// Zero every chunk the rank created (`created`, byte ranges per
/// storage). The padding the manifest's fills name lies in them (a mapped
/// chunk holds its source's zeros already), and until the transfer ends
/// every byte of a weight is either its value or zero: the forwards that
/// warm the rank up and capture its graphs meanwhile read finite numbers.
fn zero_created(
    storages: &[Arc<Storage>],
    created: &[Vec<(u64, u64)>],
    device: cuda::DeviceId,
) -> Result<()> {
    let stream = Stream::new(device)?;
    for (storage, ranges) in storages.iter().zip(created) {
        let base = storage.address();
        for &(first, last) in ranges {
            // SAFETY: the bytes lie in a chunk this rank created and nothing
            // reads or writes them yet; the stream is synchronized below.
            unsafe { cuda::memset_d8(base + first, 0, (last - first) as usize, &stream) }?;
        }
    }
    stream.synchronize()?;
    Ok(())
}
