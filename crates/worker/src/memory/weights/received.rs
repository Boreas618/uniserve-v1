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
//! together compute one copy manifest from those sources to all of their
//! storages (`storage::manifest`), each verifying that every rank computed
//! the same digest before any byte moves ([`WeightKeys::receive`]).
//!
//! A manifest's plan weighs every block by the bandwidth of its link, so
//! each rank first has its host's Transfer Server measure the link from
//! every source GPU of another node into its GPU that it has not measured
//! (two READs, or copies over the NVLink fabric, of 64 MiB of a chunk that
//! GPU holds into a scratch storage of the rank's), and the ranks plan from
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

use cuda::{Protection, Stream};
use sha2::{Digest, Sha256};
use storage::manifest::{self, Gpu, Manifest, Space};
use storage::protocol::{
    Access, LinkBandwidth, ListedTensor, Listing, Memory, Priority, Region, Span,
};
use storage::{Client, Key, Mode, Placement, StorageRecord, TensorRecord};

use super::{
    Found, MANIFEST_NAME, MANIFEST_VERSION, Planned, WeightKeys, WeightManifest, field_record,
    host_bytes, read_text, storage_name,
};
use crate::ffi::{Backing, Interpreter};
use crate::memory::allocation::Storage;
use crate::memory::retained::Retained;
use crate::parallel::host::HostGroup;
use crate::{Error, Result};

/// The weights the deployment's worlds retain on the host and on its other
/// nodes: their ranks' storages, each on a GPU that holds all of its
/// chunks, the links the host's Transfer Server measured, and the server
/// of each source GPU of another node.
pub(crate) struct Sources {
    world: manifest::World,
    links: Vec<LinkBandwidth>,
    remote: BTreeMap<Gpu, String>,
}

/// How long another node's Storage Server may take to accept the worker's
/// connection and answer one request: a listing and a manifest are small.
const PEER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

/// The bytes of one timed copy of a link measurement: the Transfer Server
/// measures a link from another host by its contiguous chunks of 64 MiB,
/// the faster of the first two giving its bandwidth (its lanes'
/// `MEASURE_BYTES`).
const LINK_SAMPLE: u64 = 64 << 20;

/// Every source storage a listing's manifests name, by GPU, once per GPU
/// however many manifests name it.
type ByGpu = BTreeMap<Gpu, BTreeMap<Vec<u8>, manifest::Storage>>;

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
    pub(crate) fn sources(&self, retained: &Retained, peers: &[String]) -> Result<Option<Sources>> {
        // A rank whose weights span devices (a generation tower on a device
        // of its own) loads them.
        if self.devices.len() != 1 {
            return Ok(None);
        }
        let listing = retained.call("listing the storage server", |client| client.list())?;
        let Some(links) = listing
            .transfer
            .as_ref()
            .map(|transfer| transfer.links.clone())
            .filter(|links| !links.is_empty())
        else {
            return Ok(None);
        };
        let deployment = retained.scope().deployment.clone();
        let mut by_gpu = ByGpu::new();
        self.collect(
            &listing,
            &deployment,
            |key| read_text(retained, key),
            &mut by_gpu,
        )?;
        let mut remote = BTreeMap::new();
        for endpoint in peers {
            let mut peer = Client::connect_peer(endpoint, PEER_TIMEOUT)
                .map_err(|error| storage_error_at(endpoint, "connecting", &error))?;
            let listing = peer
                .list()
                .map_err(|error| storage_error_at(endpoint, "listing", &error))?;
            let read = |key: Key| {
                let bytes = peer.read(deployment.clone(), key).map_err(|error| {
                    storage_error_at(endpoint, "reading a weight manifest", &error)
                })?;
                String::from_utf8(bytes).map(Some).map_err(|_| {
                    Error::Interface(format!("a weight manifest at {endpoint} is not text"))
                })
            };
            for gpu in self.collect(&listing, &deployment, read, &mut by_gpu)? {
                remote.entry(gpu).or_insert_with(|| endpoint.clone());
            }
        }
        // One source rank per GPU, its storages in a fixed order, so that
        // every worker that lists the same servers plans the same manifest.
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
            remote,
        }))
    }

    /// Adds every source storage that `listing`'s manifests name to
    /// `by_gpu`, reading each manifest with `read`: the storages of the
    /// fields of this worker's fingerprints whose chunks all lie on one GPU
    /// of the listing, read-only and retained by the deployment. Returns
    /// the GPUs it added storages of.
    fn collect(
        &self,
        listing: &Listing,
        deployment: &str,
        mut read: impl FnMut(Key) -> Result<Option<String>>,
        by_gpu: &mut ByGpu,
    ) -> Result<BTreeSet<Gpu>> {
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
        let mut added = BTreeSet::new();
        for key in manifests {
            let Some(text) = read(key)? else {
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
                    added.insert(gpu);
                    by_gpu
                        .entry(gpu)
                        .or_default()
                        .insert(identity.clone(), storage.storage.clone());
                }
            }
        }
        Ok(added)
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
        // Every rank plans from the links all of them measured: a rank of
        // another host measured other links.
        let links = measure_links(retained, sources, device, gpu)?;
        let links = gather_links(agreement, &links)?;
        let measured = started.elapsed();
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
            links,
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
                // A source chunk on a GPU of another node is that node's
                // region.
                if let Region::Tensor {
                    at: Some(Placement::Device { gpu }),
                    ..
                } = &span.source
                    && let Some(endpoint) = sources.remote.get(gpu)
                {
                    span.source = Region::Remote {
                        endpoint: endpoint.clone(),
                        region: Box::new(span.source),
                    };
                }
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
            remote_sources = sources.remote.len(),
            // Each phase's end, from the start: the unread model freed,
            // the links from other nodes measured and gathered, the ranks'
            // storages gathered, the manifest planned and
            // verified, the chunks mapped and created, the receiving keys
            // retained, the padding zeroed, the spans listed and the
            // transfer submitted.
            freed_ms = freed.as_millis() as u64,
            measured_ms = measured.as_millis() as u64,
            gathered_ms = gathered.as_millis() as u64,
            planned_ms = planned.as_millis() as u64,
            verified_ms = verified.as_millis() as u64,
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
                    fabric: None,
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

/// The links of `sources` into `gpu`, this rank's GPU on `device`, as the
/// host's Transfer Server measured them, after it measured the link from
/// every source GPU of another node it had not: for each, two copies of
/// [`LINK_SAMPLE`] bytes from the longest chunk that GPU holds into a
/// scratch storage the rank creates and releases, one task per GPU since
/// the copies write the same scratch. A GPU that holds no chunk that long
/// stays unmeasured, so a plan whose blocks only it holds fails.
fn measure_links(
    retained: &Retained,
    sources: &Sources,
    device: cuda::DeviceId,
    gpu: Gpu,
) -> Result<Vec<LinkBandwidth>> {
    let into = Memory::Device { gpu };
    let reaches = |links: &[LinkBandwidth], source: &Gpu| {
        links.iter().any(|measured| {
            measured.link.destination == into
                && match &measured.link.source {
                    Memory::Device { gpu: from }
                    | Memory::Remote {
                        gpu: Some(from), ..
                    } => from == source,
                    _ => false,
                }
        })
    };
    let mut unmeasured: Vec<(Gpu, String, Key)> = Vec::new();
    for (source, endpoint) in &sources.remote {
        if reaches(&sources.links, source) {
            continue;
        }
        let longest = sources
            .world
            .ranks
            .iter()
            .filter(|rank| rank.gpu == Some(*source))
            .flat_map(|rank| rank.storages.iter().flat_map(|storage| &storage.chunks))
            .filter(|chunk| chunk.length >= LINK_SAMPLE)
            .max_by_key(|chunk| (chunk.length, chunk.key));
        match longest {
            Some(chunk) => unmeasured.push((*source, endpoint.clone(), chunk.key)),
            None => tracing::warn!(
                gpu = %cuda::DeviceUuid(*source),
                endpoint,
                "another node's GPU holds no chunk long enough to measure its link"
            ),
        }
    }
    if unmeasured.is_empty() {
        return Ok(sources.links.clone());
    }

    let started = std::time::Instant::now();
    let bytes = 2 * LINK_SAMPLE;
    let scratch = Storage::segmented(device, &[bytes as usize], &[])?;
    let descriptor = scratch.create_segment(0)?;
    let key = measuring_key(retained.world(), &gpu);
    retained.call("retaining a link measurement's scratch", |client| {
        client.put(
            vec![TensorRecord {
                key,
                name: "link measurement scratch".into(),
                storage: 0,
                offset: 0,
                dtype: "uint8".into(),
                shape: vec![bytes],
                strides: vec![1],
                access: Access::Exclusive,
            }],
            &[(
                StorageRecord {
                    placement: Placement::Device { gpu },
                    bytes,
                    segment: None,
                    fabric: None,
                },
                descriptor.as_fd(),
            )],
        )
    })?;
    retained.use_keys([key]);
    let copied = unmeasured.iter().try_for_each(|(source, endpoint, chunk)| {
        let spans = (0..2)
            .map(|half| Span {
                source: Region::Remote {
                    endpoint: endpoint.clone(),
                    region: Box::new(Region::Tensor {
                        key: *chunk,
                        at: Some(Placement::Device { gpu: *source }),
                    }),
                },
                source_offset: 0,
                destination: Region::Tensor { key, at: None },
                destination_offset: half * LINK_SAMPLE,
                bytes: LINK_SAMPLE,
                rows: 1,
                source_pitch: 0,
                destination_pitch: 0,
            })
            .collect();
        let task = retained.transfer(spans, Priority::Background, retained.world())?;
        retained.transferred(task)
    });
    retained.forget(&[key])?;
    drop(scratch);
    copied?;

    // The Transfer Server reports a link it measured before the end of the
    // task that measured it.
    let listing = retained.call("listing the storage server", |client| client.list())?;
    let links = listing
        .transfer
        .map(|transfer| transfer.links)
        .unwrap_or_default();
    let missing: Vec<String> = unmeasured
        .iter()
        .filter(|(source, ..)| !reaches(&links, source))
        .map(|(source, ..)| cuda::DeviceUuid(*source).to_string())
        .collect();
    if !missing.is_empty() {
        tracing::warn!(
            ?missing,
            "the links from these GPUs of another node stay unmeasured"
        );
    }
    tracing::info!(
        gpus = unmeasured.len(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "measured the links from another node's GPUs"
    );
    Ok(links)
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

/// The key a rank's link measurement scratch on `gpu` is retained under in
/// world `world`.
fn measuring_key(world: u64, gpu: &Gpu) -> Key {
    // Scratch is named by its owner, not by the bytes the transfer writes.
    let mut key = [0; 32];
    key[..8].copy_from_slice(b"linkbw\0\0");
    key[8..16].copy_from_slice(&world.to_le_bytes());
    key[16..].copy_from_slice(gpu);
    Key(key)
}

/// A failure of a request to another node's Storage Server at `endpoint`.
fn storage_error_at(endpoint: &str, what: &str, error: &storage::Error) -> Error {
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
