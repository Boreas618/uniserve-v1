//! The daemon plans retained weights, rank placement and device memory before
//! a successor starts. Planning ranks run on their actual nodes through the
//! deployment process interface; each describes its target layout and measures
//! the links to retained sources. The daemon caches and sends executable copy
//! manifests, so admission and transfer use the same assignment and byte counts.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::Context as _;
use engine::launches::Agreement;
use proto::request::wire;
use storage::manifest::weights::{Sources, Transfer};
use storage::manifest::{self, Gpu, Manifest};

/// What a change of the parallel configuration does with the weights, as
/// the daemon planned it.
pub(crate) struct Redistribution {
    /// Each target rank's CUDA device index on its host, in rank order
    /// (`placement_config.devices`): the order the daemon chose, or the one
    /// the update stated.
    pub(crate) devices: Vec<u32>,
    /// Whether the daemon chose the order.
    pub(crate) chosen: bool,
    /// Executable weight transfer for each worker group.
    pub(crate) received: Vec<wire::ReceivedWeights>,
    /// The bytes of weight chunks the successor creates on each GPU.
    pub(crate) created: BTreeMap<Gpu, u64>,
    /// What the plan reports: each GPU's mapped, copied and created bytes,
    /// each manifest's digest and link loads.
    pub(crate) report: serde_json::Value,
    /// The refusal when a GPU the serving world does not use cannot hold
    /// what the successor creates there.
    pub(crate) refusal: Option<String>,
}

/// Plans the target world from the weights `served_world` retains across
/// the deployment's nodes.
/// `open` lets the daemon choose the ranks' device order; otherwise the
/// target's order stands. Returns `None` when the successor receives no
/// weights: its ranks retain no reusable weights, or the source world
/// retains none of the required model.
///
/// # Errors
///
/// Fails when the target's launches do not resolve, a planning process
/// fails, the Storage Server cannot be read, or a manifest cannot be
/// computed.
pub(crate) async fn plan(
    manifests: &Mutex<manifest::Planner>,
    process_address: &str,
    target_world: u64,
    served_world: u64,
    target: &wire::EngineSpec,
    open: bool,
) -> anyhow::Result<Option<Redistribution>> {
    let started = Instant::now();
    let shorthand = target
        .placement
        .as_ref()
        .context("the target spec names no ranks")?;
    // The daemon orders only the shorthand's ranks, on CUDA devices.
    let open = open
        && shorthand.worker_groups.is_empty()
        && matches!(shorthand.device.as_str(), "cuda" | "gpu");
    // Preserve the declared device set. Assignment may permute its ranks,
    // but may not introduce GPUs outside the deployment's device set. The
    // planning claim belongs to the successor, preserving serving visibility.
    let planning = wire::EngineSpec {
        world: target_world,
        ..target.clone()
    };
    let launches = engine::launches::rank_launches(&planning)
        .context("resolving the target world's launches")?;
    let address = process_address.to_owned();
    let planned = tokio::task::spawn_blocking(move || {
        engine::launches::plan_ranks(&planning, &address, served_world)
    })
    .await
    .context("planning ranks did not complete")??;
    let ranks_planned = started.elapsed();
    if planned.iter().any(|rank| rank.storages.is_none()) {
        return Ok(None);
    }
    let Some(mut sources) = planned
        .iter()
        .map(|rank| rank.sources.clone())
        .collect::<Option<Vec<Sources>>>()
    else {
        return Ok(None);
    };
    // Every planning rank may measure a different destination. Use the union
    // for all assignments, keeping one observation per directed link.
    let mut links = BTreeMap::<Vec<u8>, storage::protocol::LinkBandwidth>::new();
    for source in &sources {
        for measured in &source.links {
            let key = serde_json::to_vec(&measured.link)?;
            let entry = links.entry(key).or_insert_with(|| measured.clone());
            entry.bytes_per_second = entry.bytes_per_second.max(measured.bytes_per_second);
        }
    }
    let links: Vec<_> = links.into_values().collect();
    for source in &mut sources {
        source.links = links.clone();
    }

    let sources_read = started.elapsed();
    // The GPU of each device index the ranks planned on.
    let devices: Vec<u32> = launches
        .iter()
        .map(|launch| device_index(&launch.descriptor.device))
        .collect::<anyhow::Result<_>>()?;
    let storages = |index: usize| planned[index].storages.clone().unwrap_or_default();
    let same_sources = sources.iter().all(|other| other.world == sources[0].world);
    let mut assigned: Vec<Gpu> = planned.iter().map(|rank| rank.gpu).collect();
    if open && same_sources {
        // Node placement is declared by the deployment. Permute its ranks
        // among that node's devices, retaining the deployment's host boundary.
        let mut hosts = BTreeMap::<&str, Vec<usize>>::new();
        for (index, launch) in launches.iter().enumerate() {
            hosts.entry(&launch.host).or_default().push(index);
        }
        for members in hosts.into_values() {
            let request = manifest::Request {
                source: sources[0].world.clone(),
                target: manifest::World {
                    ranks: members
                        .iter()
                        .map(|&index| manifest::Rank {
                            gpu: None,
                            storages: storages(index),
                        })
                        .collect(),
                },
                gpus: members.iter().map(|&index| planned[index].gpu).collect(),
                links: links.clone(),
            };
            let manifest = planned_manifest(manifests, &request)?;
            for (&index, &gpu) in members.iter().zip(&manifest.gpus) {
                assigned[index] = gpu;
            }
        }
    }
    let order: Vec<u32> = assigned
        .iter()
        .map(|gpu| {
            planned
                .iter()
                .position(|rank| rank.gpu == *gpu)
                .map(|index| devices[index])
                .context("the assignment names a GPU no rank planned on")
        })
        .collect::<anyhow::Result<_>>()?;

    let ranks_assigned = started.elapsed();
    // One manifest per group of ranks that loads together, as its ranks
    // compute it: in their order there, on their GPUs.
    let mut agreements: BTreeMap<&Agreement, Vec<usize>> = BTreeMap::new();
    for (index, launch) in launches.iter().enumerate() {
        agreements.entry(&launch.agreement).or_default().push(index);
    }
    let mut received = Vec::new();
    let mut per_gpu: BTreeMap<Gpu, Bytes> = BTreeMap::new();
    let mut ranks_report = Vec::new();
    let mut manifests_report = Vec::new();
    for mut members in agreements.into_values() {
        members.sort_by_key(|&index| launches[index].agreement_index);
        let first = members[0];
        let request = manifest::Request {
            source: sources[first].world.clone(),
            target: manifest::World {
                ranks: members
                    .iter()
                    .map(|&index| manifest::Rank {
                        gpu: Some(assigned[index]),
                        storages: storages(index),
                    })
                    .collect(),
            },
            gpus: members.iter().map(|&index| assigned[index]).collect(),
            links: sources[first].links.clone(),
        };
        let manifest = planned_manifest(manifests, &request)?;
        let groups: BTreeSet<&str> = members
            .iter()
            .map(|&index| launches[index].group.as_str())
            .collect();
        let encoded = serde_json::to_vec(&Transfer {
            request: request.clone(),
            manifest: manifest.as_ref().clone(),
        })?;
        received.extend(groups.iter().map(|group| wire::ReceivedWeights {
            worker_group: (*group).to_owned(),
            world: served_world,
            manifest: encoded.clone(),
        }));
        for (position, &index) in members.iter().enumerate() {
            let bytes = rank_bytes(&manifest, &request, position as u32);
            let launch = &launches[index];
            ranks_report.push(serde_json::json!({
                "group": launch.group,
                "rank": launch.rank,
                "device": order[index],
                "gpu": storage::protocol::gpu_name(&assigned[index]),
                "mapped_bytes": bytes.mapped,
                "copied_bytes": bytes.copied,
                "created_bytes": bytes.created,
                "missing_tensors": bytes.missing,
            }));
            per_gpu.entry(assigned[index]).or_default().add(&bytes);
        }
        manifests_report.push(serde_json::json!({
            "groups": groups,
            "digest": hex(&manifest.digest),
            "copied_bytes": manifest.copied_bytes(),
            "links": manifest.links.iter().map(|link| serde_json::json!({
                "source": storage::protocol::gpu_name(&link.source),
                "target": storage::protocol::gpu_name(&link.target),
                "bytes": link.bytes,
                "seconds": link.seconds,
            })).collect::<Vec<_>>(),
        }));
    }

    // A GPU the serving world holds no weights on has none of its KV cache
    // to give back: what the successor creates there must fit in its free
    // storage.
    let serving: BTreeSet<Gpu> = sources
        .iter()
        .flat_map(|sources| sources.world.ranks.iter().filter_map(|rank| rank.gpu))
        .collect();
    let mut refusals = Vec::new();
    let mut gpus_report = Vec::new();
    for (gpu, bytes) in &per_gpu {
        let free = planned
            .iter()
            .find(|rank| rank.gpu == *gpu)
            .map(|rank| rank.free_bytes);
        let serves = serving.contains(gpu);
        if !serves && let Some(free) = free.filter(|free| bytes.created > *free) {
            refusals.push(format!(
                "GPU {} holds none of the serving world's weights and has {free} bytes free, \
                 but the successor creates {} bytes of weights there",
                storage::protocol::gpu_name(gpu),
                bytes.created
            ));
        }
        gpus_report.push(serde_json::json!({
            "gpu": storage::protocol::gpu_name(gpu),
            "serving": serves,
            "free_bytes": free,
            "mapped_bytes": bytes.mapped,
            "copied_bytes": bytes.copied,
            "created_bytes": bytes.created,
        }));
    }
    let chosen = open && same_sources;
    let refusal = (!refusals.is_empty()).then(|| refusals.join("; "));
    tracing::info!(
        devices = ?order,
        chosen,
        // Each phase's end, from the start: the planning processes, the
        // sources read, the ranks' GPUs assigned, and the manifests.
        planned_ms = ranks_planned.as_millis() as u64,
        sources_ms = sources_read.as_millis() as u64,
        assigned_ms = ranks_assigned.as_millis() as u64,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "planned the weights of a change of the parallel configuration"
    );
    Ok(Some(Redistribution {
        devices: order.clone(),
        chosen,
        received,
        created: per_gpu
            .iter()
            .map(|(gpu, bytes)| (*gpu, bytes.created))
            .collect(),
        report: serde_json::json!({
            "devices": order,
            "order": if chosen { "chosen" } else { "stated" },
            "ranks": ranks_report,
            "gpus": gpus_report,
            "manifests": manifests_report,
            "planning_seconds": started.elapsed().as_secs_f64(),
            "feasible": refusal.is_none(),
            "refusal": refusal,
        }),
        refusal,
    }))
}

/// The bytes one target rank maps, copies and creates, and the tensors of
/// the regions no source holds.
#[derive(Default)]
struct Bytes {
    mapped: u64,
    copied: u64,
    created: u64,
    missing: u64,
}

impl Bytes {
    fn add(&mut self, other: &Self) {
        self.mapped += other.mapped;
        self.copied += other.copied;
        self.created += other.created;
        self.missing += other.missing;
    }
}

/// What target rank `rank` of `request` maps, copies and creates under
/// `manifest`: every chunk it does not map it creates.
fn rank_bytes(manifest: &Manifest, request: &manifest::Request, rank: u32) -> Bytes {
    let mapped: BTreeSet<(u32, u32)> = manifest
        .mappings
        .iter()
        .filter(|mapping| mapping.rank == rank)
        .map(|mapping| (mapping.storage, mapping.chunk))
        .collect();
    let mut bytes = Bytes::default();
    for (storage, held) in request.target.ranks[rank as usize]
        .storages
        .iter()
        .enumerate()
    {
        for (chunk, value) in held.chunks.iter().enumerate() {
            if mapped.contains(&(storage as u32, chunk as u32)) {
                bytes.mapped += value.length;
            } else {
                bytes.created += value.length;
            }
        }
    }
    bytes.copied = manifest
        .copies
        .iter()
        .filter(|copy| copy.target.rank == rank)
        .map(manifest::Copy::bytes)
        .sum();
    bytes.missing = manifest
        .missing
        .iter()
        .filter(|missing| missing.rank == rank)
        .count() as u64;
    bytes
}

/// The manifest of `request`, computed once per request.
fn planned_manifest(
    manifests: &Mutex<manifest::Planner>,
    request: &manifest::Request,
) -> anyhow::Result<std::sync::Arc<Manifest>> {
    manifests
        .lock()
        .map_err(|_| anyhow::anyhow!("the manifest planner's lock is poisoned"))?
        .manifest(request)
        .context("planning a copy manifest")
}

/// The device index of a rank's device name (`cuda:N`).
fn device_index(device: &str) -> anyhow::Result<u32> {
    device
        .strip_prefix("cuda:")
        .and_then(|index| index.parse().ok())
        .with_context(|| format!("a planned rank's device {device:?} is no CUDA device index"))
}

/// A digest in hex.
fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
