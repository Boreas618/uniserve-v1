//! The weight manifests ranks retain, and the source world they make.
//!
//! A rank that retains its weights at its host's Storage Server retains,
//! beside its chunks, a manifest: host bytes of JSON text under the name
//! [`NAME`] that state the manifest's [`VERSION`], the digest of what
//! determines the bytes of each checkpoint source's tensors
//! (`fingerprints`, by source name, the empty name for the tensors no
//! source loads), the rank's tensors as its library describes them, and
//! its storages ([`RetainedStorage`]): each one's device among the rank's
//! devices, layout and chunks. The worker writes it; a world that receives
//! weights and the daemon that plans its preparation read the storages of
//! every manifest a world retains as the source world of a copy manifest
//! ([`sources`]), both from the same listing and texts, so both compute the
//! same source world.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use super::{Gpu, Rank, Storage, World};
use crate::Key;
use crate::protocol::{Access, LinkBandwidth, Listing, Placement};

/// The name a rank's manifest is retained under, by which the manifests a
/// world retains are found.
pub const NAME: &str = "weights manifest";

/// The version of the manifest the worker writes and reads; a manifest of
/// another version names no storage.
pub const VERSION: u64 = 6;

/// One retained storage of a manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetainedStorage {
    /// The index of the storage's device among the rank's devices; `None`
    /// for the host region of a stack streamed from host storage.
    pub device: Option<usize>,
    #[serde(flatten)]
    pub storage: Storage,
}

/// What a rank plans of its weights before its world starts (`uniserve
/// worker --plan`), without reading the checkpoint or retaining anything:
/// a daemon plans a change of the parallel configuration from every target
/// rank's.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlannedRank {
    /// The GPU the rank planned on, and its total and free bytes before
    /// the rank's model was materialized there.
    pub gpu: Gpu,
    pub total_bytes: u64,
    pub free_bytes: u64,
    /// The digest of what determines the bytes of the rank's tensors, by
    /// checkpoint source, in hex as a manifest states them: the storages it
    /// receives are of sources with the same digests ([`sources`]).
    pub fingerprints: BTreeMap<String, String>,
    /// The storages its weights take on its GPU, chunked and keyed as the
    /// rank retains them; `None` for a rank that does not receive its
    /// weights (it retains none, or they span devices or stream from host
    /// storage), which loads them.
    pub storages: Option<Vec<Storage>>,
    /// Retained source weights and measured links on the deployment's nodes.
    pub sources: Option<Sources>,
}

/// What a reader takes from a manifest's text: its version, fingerprints
/// and storages.
#[derive(Deserialize)]
struct Read {
    version: u64,
    fingerprints: BTreeMap<String, String>,
    storages: Vec<RetainedStorage>,
}

/// The weights one world retains on a host, as the source world of a copy
/// manifest, and the links its Transfer Server measured between the GPUs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sources {
    pub world: World,
    pub links: Vec<LinkBandwidth>,
    /// Storage endpoint for each source GPU outside the reader's host.
    pub remote: Vec<(Gpu, String)>,
}

/// A daemon's executable weight redistribution plan. Workers consume this
/// exact request and manifest; they do not independently choose another plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transfer {
    pub request: super::Request,
    pub manifest: super::Manifest,
}

/// The weights world `world` of `deployment` retains on the host `listing`
/// lists, for a rank whose checkpoint sources have the digests
/// `fingerprints` (hex, by source name, as a manifest states them).
///
/// Every manifest the world retains names storages; a storage on a device
/// whose fields' sources all have this rank's digests holds bytes the rank
/// may need, on every GPU where the world retains all of its chunks. Each
/// GPU is one source rank, its storages once each in the order of their
/// serialized form, so every reader of the same listing and texts computes
/// the same world. `read` returns the text of the host bytes of a key
/// (`None` when the server no longer retains them). Returns `None` when the
/// world retains no such storage, or no Transfer Server runs to copy them.
///
/// # Errors
///
/// Fails when `read` fails.
pub fn sources<E>(
    listing: &Listing,
    deployment: &str,
    world: Option<u64>,
    fingerprints: &BTreeMap<String, String>,
    mut read: impl FnMut(Key) -> Result<Option<String>, E>,
) -> Result<Option<Sources>, E> {
    let Some(links) = listing
        .transfer
        .as_ref()
        .map(|transfer| transfer.links.clone())
        .filter(|links| !links.is_empty())
    else {
        return Ok(None);
    };
    let retains = |tensor: &crate::protocol::ListedTensor| {
        tensor.access == Access::ReadOnly
            && tensor.retainers.iter().any(|retainer| {
                retainer.scope.deployment == deployment
                    && world.is_none_or(|world| retainer.world == world)
            })
    };
    // The GPUs holding each chunk, and the manifests.
    let mut held: HashMap<Key, BTreeSet<Gpu>> = HashMap::new();
    let mut manifests = BTreeSet::new();
    for tensor in listing.tensors.iter().filter(|tensor| retains(tensor)) {
        match tensor.placement {
            Placement::Device { gpu } => {
                held.entry(tensor.key).or_default().insert(gpu);
            }
            Placement::Host { .. } if tensor.name == NAME => {
                manifests.insert(tensor.key);
            }
            Placement::Host { .. } => {}
        }
    }

    // Each source storage once per GPU, whatever manifests name it.
    let mut by_gpu: BTreeMap<Gpu, BTreeMap<Vec<u8>, Storage>> = BTreeMap::new();
    for key in manifests {
        let Some(text) = read(key)? else {
            continue;
        };
        // A manifest of another version or format names no source.
        let Ok(manifest) = serde_json::from_str::<Read>(&text) else {
            continue;
        };
        if manifest.version != VERSION {
            continue;
        }
        for storage in &manifest.storages {
            // Only the fields of sources with the rank's digests hold the
            // bytes it needs.
            let same = storage.device.is_some()
                && storage.storage.layout.fields.iter().all(|field| {
                    manifest.fingerprints.get(&field.source) == fingerprints.get(&field.source)
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
            // Serializing a storage of strings and numbers cannot fail.
            #[expect(
                clippy::expect_used,
                reason = "storage contains only JSON-compatible values"
            )]
            let identity =
                serde_json::to_vec(&storage.storage).expect("storage contains JSON values");
            for gpu in gpus.unwrap_or_default() {
                by_gpu
                    .entry(gpu)
                    .or_default()
                    .insert(identity.clone(), storage.storage.clone());
            }
        }
    }
    let ranks: Vec<Rank> = by_gpu
        .into_iter()
        .map(|(gpu, storages)| Rank {
            gpu: Some(gpu),
            storages: storages.into_values().collect(),
        })
        .collect();
    if ranks.is_empty() {
        return Ok(None);
    }
    Ok(Some(Sources {
        world: World { ranks },
        links,
        remote: Vec::new(),
    }))
}
