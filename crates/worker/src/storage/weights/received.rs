//! Weights a worker receives from the deployment's other worlds.
//!
//! A worker whose position has no manifest of its own, such as a rank of a
//! world that changes the deployment's parallel configuration, finds the
//! weights the deployment's other worlds retain on the host's GPUs and on
//! the GPUs of the deployment's other nodes: every manifest a Storage
//! Server retains, of this version and of the same fingerprints, names
//! storages whose chunks lie on the GPUs that hold them
//! ([`WeightKeys::sources`]). The host's server is listed and read through
//! the worker's own connection; another node's over TCP as a peer, whose
//! manifests the worker reads inline (`Request::Read`). The worker plans
//! the storages its own weights take without reading the checkpoint
//! ([`WeightKeys::plan`] over an unread model), and the ranks that load
//! together consume the daemon's copy manifest. Cold starts without such
//! a plan compute it from the ranks' target layouts ([`WeightKeys::receive`]).
//!
//! A manifest's plan weighs every block by the bandwidth of its link, so
//! each rank first has its host's Transfer Server measure the link from
//! every source GPU of another node into its GPU that it has not measured
//! (two 64 MiB windows of READs or NVLink copies, repeating a smaller
//! retained chunk when necessary), and the ranks plan from
//! the links all of them measured, gathered across their hosts.
//!
//! Each rank then allocates its storages at fixed addresses: a chunk whose
//! key a source holds on its GPU is mapped read-only, every other chunk is
//! created and retained exclusive under a receiving key, which the Transfer
//! Server writes the manifest's copies into as one task at the
//! `Background` priority, across the links it measured; a copy from a GPU
//! of another node names that node's region (`Region::Remote`), which the
//! Transfer Server copies over the NVLink fabric when the source chunk's
//! fabric handle imports, and READs otherwise. Padding the manifest names
//! is zeroed on the rank's GPU, and the tensors of the regions no source
//! holds are read from the checkpoint as a load of only those tensors. The
//! rank binds its model to the storages while the bytes arrive; once the
//! transfer ended ([`Receipt::retain`]), the received chunks are retained
//! read-only under their keys, like chunks a load writes, the receiving
//! keys are released and the storages become read-only.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::os::fd::{AsFd, OwnedFd};
use std::sync::Arc;

use ::storage::manifest::weights::{Sources, Transfer};
use ::storage::manifest::{self, Manifest, Space};
use ::storage::protocol::{Access, LinkBandwidth, Priority, Region, Span};
use ::storage::{Client, Key, Mode, Placement, StorageRecord, TensorRecord};
use cuda::{Protection, Stream};
use sha2::{Digest, Sha256};

use super::{
    Found, MANIFEST_NAME, MANIFEST_VERSION, TransferLayout, WeightKeys, WeightManifest,
    field_record, host_bytes, storage_name,
};
use crate::ffi::Backing;
use crate::parallel::host::HostGroup;
use crate::storage::allocation::Storage;
use crate::storage::retained::Retained;
use crate::{Error, Result};

/// How long another node's Storage Server may take to accept the worker's
/// connection and answer one request: a listing and a manifest are small.
const PEER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

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
    /// the host and on the deployment's other nodes (their Storage Servers
    /// at `peers`), of the same model, which this worker may receive:
    /// `None` when they retain none, or no Transfer Server runs to copy
    /// them. Failure to reach a peer is reported to the caller.
    pub(crate) fn sources(
        &self,
        retained: &Retained,
        peers: &[String],
        worlds: Option<&[u64]>,
    ) -> Result<Option<Sources>> {
        if self.devices.len() != 1 {
            return Ok(None);
        }
        let deployment = &retained.scope().deployment;
        let fingerprints = self.fingerprint_digests();
        let listing = retained.call("listing the storage server", |client| client.list())?;
        let mut sources =
            manifest::weights::sources(&listing, deployment, worlds, &fingerprints, |key| {
                let bytes = retained.call("reading a weight manifest", |client| {
                    client.read(deployment.clone(), key)
                })?;
                String::from_utf8(bytes)
                    .map(Some)
                    .map_err(|error| Error::Interface(error.to_string()))
            })?
            .unwrap_or_else(|| Sources {
                world: manifest::World { ranks: Vec::new() },
                links: listing
                    .transfer
                    .map(|transfer| transfer.links)
                    .unwrap_or_default(),
            });
        for endpoint in peers {
            let mut peer = Client::connect_peer(endpoint, PEER_TIMEOUT)
                .map_err(|error| storage_error_at(endpoint, "connecting", &error))?;
            let listing = peer
                .list()
                .map_err(|error| storage_error_at(endpoint, "listing", &error))?;
            let read = |key| {
                let bytes = peer.read(deployment.clone(), key).map_err(|error| {
                    storage_error_at(endpoint, "reading a weight manifest", &error)
                })?;
                String::from_utf8(bytes)
                    .map(Some)
                    .map_err(|error| Error::Interface(error.to_string()))
            };
            if let Some(mut remote) =
                manifest::weights::sources(&listing, deployment, worlds, &fingerprints, read)?
            {
                for rank in &mut remote.world.ranks {
                    for storage in &mut rank.storages {
                        for chunk in &mut storage.chunks {
                            let at = match chunk.binding {
                                manifest::Binding::Rank => Placement::Device {
                                    gpu: rank.gpu.ok_or_else(|| {
                                        Error::Invariant(
                                            "retained device weights name no GPU".into(),
                                        )
                                    })?,
                                },
                                manifest::Binding::Local { at }
                                | manifest::Binding::Remote { at, .. } => at,
                            };
                            chunk.binding = manifest::Binding::Remote {
                                endpoint: endpoint.clone(),
                                at,
                            };
                        }
                    }
                }
                sources.world.ranks.extend(remote.world.ranks);
                sources.links.extend(remote.links);
            }
        }
        sources.world.ranks.sort_by_key(|rank| rank.gpu);
        Ok((!sources.world.ranks.is_empty()).then_some(sources))
    }

    /// Measures source links before the daemon assigns ranks or books storage.
    pub(crate) fn measure_sources(&self, retained: &Retained, sources: &mut Sources) -> Result<()> {
        let (device, gpu) = self.device(0)?;
        sources.links =
            crate::transfer::links::measure(retained, sources, device, Placement::Device { gpu })?;
        Ok(())
    }

    /// Receive this rank's `planned` storages from `sources`, together
    /// with the ranks of `agreement` (the ranks that load together, every
    /// one of which calls this): the ranks plan one manifest from the
    /// sources to all their storages when no daemon plan is supplied;
    /// then each maps the chunks a source holds on its GPU,
    /// creates the rest, submits the copies into them to the Transfer
    /// Server and zeroes the padding the manifest names. Returns the
    /// storages as found weights: the tensors of the regions no source
    /// holds are missing, for the rank to read from the checkpoint, and
    /// the receipt retains the received chunks once they arrived.
    pub(crate) fn receive(
        &self,
        retained: &Retained,
        planned: TransferLayout,
        sources: &Sources,
        agreement: &HostGroup,
        transfer: Option<Transfer>,
    ) -> Result<Found> {
        let started = std::time::Instant::now();
        let TransferLayout { tensors, storages } = planned;
        let (device, gpu) = self.device(0)?;
        if storages.iter().any(|storage| storage.device != Some(0)) {
            return Err(Error::Unsupported(
                "a rank receives weights only on its one GPU".into(),
            ));
        }

        let freed = started.elapsed();
        let rank = agreement.index();
        let Transfer { request, manifest } = match transfer {
            Some(transfer) => transfer,
            None => {
                // Cold starts have no daemon redistribution plan. The ranks
                // describe their target storages through their existing group.
                let own = manifest::Rank {
                    gpu: Some(gpu),
                    storages: storages
                        .iter()
                        .map(|storage| storage.storage.clone())
                        .collect(),
                };
                let encoded =
                    serde_json::to_vec(&own).map_err(|error| Error::Invalid(error.to_string()))?;
                let links = crate::transfer::links::measure(
                    retained,
                    sources,
                    device,
                    Placement::Device { gpu },
                )?;
                let links = gather_links(agreement, &links)?;
                let targets = agreement
                    .all_gather_bytes(&encoded)?
                    .iter()
                    .map(|bytes| {
                        serde_json::from_slice::<manifest::Rank>(bytes)
                            .map_err(|error| Error::Interface(error.to_string()))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let request = manifest::Request {
                    source: sources.world.clone(),
                    gpus: targets.iter().filter_map(|target| target.gpu).collect(),
                    target: manifest::World { ranks: targets },
                    links,
                };
                let manifest = Manifest::plan(&request)
                    .map_err(|error| Error::Interface(error.to_string()))?;
                Transfer { request, manifest }
            }
        };
        let planned = started.elapsed();
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
                    fabric: storage.segment_fabric(segment)?,
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
        // The chunks the rank created start zeroed, the padding the
        // manifest names among them.
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
            .bind(&request, &sources.world)
            .spans()
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
        // The manifest the rank retains for its position, and the fields
        // it binds: all but those of the tensors it reads.
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
        // Finish all local preparation before publishing receiving chunks.
        // A failed transfer submission then has only these records to release.
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
        retained.use_keys(shared.iter().copied().chain(receiving.values().copied()));
        let registered = started.elapsed();
        let task = if spans.is_empty() {
            None
        } else {
            match retained.transfer(spans, Priority::Background, retained.world()) {
                Ok(task) => Some(task),
                Err(error) => {
                    retained.forget(&receiving.values().copied().collect::<Vec<_>>())?;
                    return Err(error);
                }
            }
        };
        let submitted = started.elapsed();
        tracing::info!(
            rank,
            mapped = shared.len(),
            created = created.len(),
            copied_bytes = bytes,
            spans = spans_count,
            missing = missing.len(),
            digest = %super::hex(&manifest.digest),
            // Each phase's end, from the start: the unread model freed,
            // the links from other nodes measured and gathered, the ranks'
            // storages gathered, the manifest selected, the chunks
            // mapped and created, the padding zeroed, the spans listed,
            // the receiving keys retained and the
            // transfer submitted.
            freed_ms = freed.as_millis() as u64,
            planned_ms = planned.as_millis() as u64,
            allocated_ms = allocated.as_millis() as u64,
            registered_ms = registered.as_millis() as u64,
            zeroed_ms = zeroed.as_millis() as u64,
            listed_ms = listed.as_millis() as u64,
            submitted_ms = submitted.as_millis() as u64,
            "receiving the rank's weights from the deployment's other worlds"
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
    /// Poll the transfer once; completion is consumed exactly once.
    pub(crate) fn ready(&mut self, retained: &Retained) -> Result<bool> {
        if let Some(task) = self.task {
            let completion =
                retained.call("polling expert copies", |client| client.poll_transfer(task))?;
            let Some((bytes, end)) = completion else {
                return Ok(false);
            };
            self.task = None;
            match end {
                ::storage::protocol::TransferEnd::Complete => {}
                ::storage::protocol::TransferEnd::Cancelled => {
                    return Err(Error::Interface(format!(
                        "transfer {task} was cancelled after {bytes} bytes"
                    )));
                }
                ::storage::protocol::TransferEnd::Failed { message } => {
                    return Err(Error::Interface(format!(
                        "transfer {task} failed after {bytes} bytes: {message}"
                    )));
                }
            }
        }
        Ok(true)
    }

    /// Cancel pending copies before releasing their exclusive destinations.
    pub(crate) fn cancel(self, retained: &Retained) -> Result<()> {
        if let Some(task) = self.task {
            retained.cancel_transfer(task)?;
        }
        retained.forget(&self.receiving)
    }

    /// Wait until the received bytes arrived, then retain the created
    /// chunks read-only under their keys, and the rank's manifest unless
    /// `published` (a publication of the tensors the rank read retains
    /// it), release the receiving keys and make the storages read-only.
    pub(crate) fn retain(&mut self, retained: &Retained, published: bool) -> Result<()> {
        let started = std::time::Instant::now();
        if let Some(task) = self.task {
            let transferred = retained.transferred(task);
            self.task = None;
            transferred?;
        }
        let waited = started.elapsed();
        let manifest_storage = host_bytes(self.manifest.1.as_bytes())?;
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
                    fabric: None,
                },
                manifest_storage.as_fd(),
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

/// Every link the ranks of `agreement` measured, each rank giving `links`,
/// in one order on every rank: a link measured by several (each rank of a
/// host lists the same) takes the highest bandwidth.
fn gather_links(agreement: &HostGroup, links: &[LinkBandwidth]) -> Result<Vec<LinkBandwidth>> {
    let encoded = serde_json::to_vec(links)
        .map_err(|error| Error::Invalid(format!("encoding measured links: {error}")))?;
    let mut merged: BTreeMap<Vec<u8>, LinkBandwidth> = BTreeMap::new();
    for bytes in agreement.all_gather_bytes(&encoded)? {
        let links: Vec<LinkBandwidth> = serde_json::from_slice(&bytes)
            .map_err(|error| Error::Interface(format!("a rank's measured links: {error}")))?;
        for measured in links {
            let identity = serde_json::to_vec(&measured.link)
                .map_err(|error| Error::Invalid(format!("encoding a link: {error}")))?;
            let entry = merged.entry(identity).or_insert_with(|| measured.clone());
            entry.bytes_per_second = entry.bytes_per_second.max(measured.bytes_per_second);
        }
    }
    Ok(merged.into_values().collect())
}

/// A failure of a request to another node's Storage Server at `endpoint`.
fn storage_error_at(endpoint: &str, what: &str, error: &::storage::Error) -> Error {
    Error::Interface(format!(
        "{what} at another node's storage server {endpoint}: {error}"
    ))
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
