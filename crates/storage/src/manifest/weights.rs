//! The weight manifests ranks retain, and the source world they make.
//!
//! A rank that retains its weights at its host's Storage Server retains,
//! beside its chunks, a manifest: host bytes of JSON text under the name
//! [`NAME`] that state the manifest's [`VERSION`], the digest of what
//! determines each tensor's bytes (`fingerprints`, by tensor name),
//! the rank's tensors as its library describes them, and
//! its storages ([`RetainedStorage`]): each one's device among the rank's
//! devices, layout and chunks. The worker writes it; the ranks of a world
//! that receives weights and the ranks a daemon plans its preparation with
//! read the storages of every manifest a world retains as the source world
//! of a copy manifest ([`sources`]), all from the same listing and texts and
//! by the fingerprints of the whole model, so all compute the same source
//! world.

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
pub const VERSION: u64 = 7;

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
    /// Allocation unit for retained conversion scratch on this GPU.
    pub allocation_granularity: u64,
    /// Resident deployment storage and the Transfer Server context needed
    /// beside this rank, measured from the device ledger.
    pub resident_bytes: u64,
    pub device_quota: Option<u64>,
    /// Target numerical capacity, independent of the source rank's layout.
    pub budget: Option<RankBudget>,
    /// The digest of what determines the bytes of the rank's tensors, by
    /// tensor name, in hex as a manifest states them: received fields have
    /// the same digests ([`sources`]).
    pub fingerprints: BTreeMap<String, String>,
    /// The storages its weights take on its GPU, chunked and keyed as the
    /// rank retains them; `None` for a rank that does not receive its
    /// weights (it retains none, or they span devices or stream from host
    /// storage), which loads them.
    pub storages: Option<Vec<Storage>>,
    /// Retained source weights and measured links on the deployment's nodes.
    pub sources: Option<Sources>,
}

/// What a successor rank must hold on its GPU before it can serve.
/// Weight bytes are filled from the concrete transfer manifest; the other
/// quantities come from the target rank's numerical and executor layouts.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankBudget {
    pub quota: u64,
    pub process: u64,
    pub graphs: u64,
    pub fixed: u64,
    pub cache: u64,
    pub cache_capacity: u64,
    /// Bytes of one target cache tail segment, its physical allocation unit.
    pub cache_segment: u64,
    pub weights: u64,
    pub pages: Vec<CachePage>,
    /// A complete cross-world request reservation, when the caller routes
    /// requests from multiple predecessor engines onto this target GPU.
    pub in_flight: Option<CacheReservation>,
}

/// Value bytes retained for accepted prefixes and the device segments
/// required to materialize them. Host copies need only the value bytes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheReservation {
    pub bytes: u64,
    pub mapped: u64,
}

/// Value storage added by a target cache group's logical page, including its
/// resident layers, heads and numerical encoding. Full-pool metadata and unit
/// index columns are already charged to the startup cache budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePage {
    pub tokens: u32,
    pub bytes: u64,
    pub window: Option<u32>,
    pub state: bool,
}

impl RankBudget {
    /// Storage the new rank adds beside already retained chunks.
    pub fn required(&self) -> u64 {
        self.process
            + self.graphs
            + self.fixed
            + self.weights
            + (self.cache + self.in_flight.map_or(0, |cache| cache.mapped)).max(self.cache_capacity)
    }

    /// The exact value demand and segment-rounded device demand of this rank.
    pub fn reserve_requests(&self, tokens: &[u32]) -> CacheReservation {
        let bytes = self.request_bytes(tokens);
        let segment = self.cache_segment.max(1);
        CacheReservation {
            bytes,
            mapped: bytes.div_ceil(segment) * segment,
        }
    }

    /// Writable cache needed to resume accepted request prefixes at this
    /// target layout. Windowed attention keeps the pages intersecting its
    /// history; recurrent layers keep an input and an output checkpoint.
    pub fn request_bytes(&self, tokens: &[u32]) -> u64 {
        tokens
            .iter()
            .filter(|&&tokens| tokens > 0)
            .map(|&tokens| {
                self.pages
                    .iter()
                    .map(|page| {
                        let count = if page.state {
                            2
                        } else {
                            let first = page
                                .window
                                .map_or(0, |window| tokens.saturating_sub(window) / page.tokens);
                            tokens.div_ceil(page.tokens) - first
                        };
                        u64::from(count) * page.bytes
                    })
                    .sum::<u64>()
            })
            .sum()
    }
}

#[cfg(test)]
mod budget_tests {
    use super::{CachePage, RankBudget};

    #[test]
    fn request_storage_follows_target_pages_windows_and_state() {
        let budget = RankBudget {
            pages: vec![
                CachePage {
                    tokens: 8,
                    bytes: 128,
                    window: None,
                    state: false,
                },
                CachePage {
                    tokens: 4,
                    bytes: 32,
                    window: Some(5),
                    state: false,
                },
                CachePage {
                    tokens: 8,
                    bytes: 16,
                    window: None,
                    state: true,
                },
            ],
            ..Default::default()
        };
        // Eight tokens fit one full-attention page, two window pages,
        // and the input/output recurrent checkpoints. Token nine adds
        // another full-attention page while the window still spans two.
        assert_eq!(budget.request_bytes(&[0]), 0);
        assert_eq!(budget.request_bytes(&[8]), 224);
        assert_eq!(budget.request_bytes(&[9]), 352);
        assert_eq!(budget.request_bytes(&[8, 9]), 576);
    }
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
}

/// A daemon's executable weight redistribution plan. Workers consume this
/// exact request and manifest; they do not independently choose another plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transfer {
    pub request: super::Request,
    pub manifest: super::Manifest,
}

/// The weights the selected `worlds` of `deployment` retain on the host `listing`
/// lists, for a model whose tensors have the digests `fingerprints` (hex,
/// by tensor name, as a manifest states them).
///
/// Every manifest the selected worlds retain names storages; the fields of
/// a storage on a device whose tensors have the model's digests hold bytes
/// a rank of the model may need, on every GPU where the world retains all
/// of the storage's chunks. Each GPU is one source rank, its storages once
/// each in the order of their serialized form, so every reader of the same
/// listing and texts with the same `fingerprints` computes the same world.
/// The ranks that receive one copy manifest bind the world it was planned
/// from, so `fingerprints` must name every tensor of the model, not only
/// those the reading rank holds: a rank that leaves out tensors another
/// rank holds selects fewer storages, at other positions. `read` returns
/// the text of the host bytes of a key (`None` when the server no longer
/// retains them). Returns `None` when the world retains no such storage,
/// or no Transfer Server runs to copy them.
///
/// # Errors
///
/// Fails when `read` fails.
pub fn sources<E>(
    listing: &Listing,
    deployment: &str,
    worlds: Option<&[u64]>,
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
                    && worlds.is_none_or(|worlds| worlds.contains(&retainer.world))
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
            if storage.device.is_none() {
                continue;
            }
            // Packed storage can hold independently keyed tensors. Keep
            // each matching field available to the transfer planner; the
            // chunks still name the original backing bytes.
            let mut source = storage.storage.clone();
            source.layout.fields.retain(|field| {
                field.tensor.rsplit_once('/').is_some_and(|(tensor, _)| {
                    fingerprints
                        .get(tensor)
                        .is_some_and(|wanted| manifest.fingerprints.get(tensor) == Some(wanted))
                })
            });
            if source.layout.fields.is_empty() {
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
            let identity = serde_json::to_vec(&source).expect("storage contains JSON values");
            for gpu in gpus.unwrap_or_default() {
                by_gpu
                    .entry(gpu)
                    .or_default()
                    .insert(identity.clone(), source.clone());
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
    }))
}
