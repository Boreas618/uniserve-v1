//! Background cache copies between live predecessor and successor owners.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;
use storage::manifest::cache::{self, Operation, Snapshot};
use storage::manifest::{Binding, Encoding, Manifest, Request, World};
use storage::protocol::{LinkBandwidth, Placement, Region};

use crate::engine::client::EngineClient;

/// How the successor consumes the predecessor's numerical state.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Reuse {
    Exact,
    Keep,
    Recompute,
}

/// Observed completed copies and the accepted prefix still to compute.
#[derive(Debug, Default, Serialize)]
pub(super) struct Report {
    pub rounds: u32,
    pub copied_bytes: u64,
    pub tail_tokens: u32,
    pub limit_tokens: u32,
    pub warmed_requests: u32,
    pub warmed_tokens: u64,
    pub recompute: Option<String>,
    pub staged: bool,
    #[serde(skip)]
    pub backup: Vec<Snapshot>,
}

/// Warm every predecessor assigned to this successor. A successful result
/// retains target pages for their continuations and leaves the temporary
/// source prefill limit in force until the caller parks those sources.
pub(super) async fn prepare(
    sources: &[Arc<EngineClient>],
    target: &Arc<EngineClient>,
    reuse: Reuse,
    tail_tokens: Option<u32>,
    prefill_tokens: Option<u32>,
    host: bool,
) -> Result<Report, String> {
    let result = rounds(sources, target, reuse, tail_tokens, prefill_tokens, host).await;
    let owners: Vec<_> = sources.iter().chain(std::iter::once(target)).collect();
    let mut errors = Vec::new();
    if result
        .as_ref()
        .map_or(true, |report| report.recompute.is_some() && !report.staged)
    {
        // Cancellation completes physical work before either scheduler may
        // reuse a borrowed unit. Failed cancellation leaves those leases held.
        for owner in &owners {
            if let Err(error) = owner.hook("uniserve.cache_cancel", "null").await {
                errors.push(error);
            }
        }
        if errors.is_empty() {
            if let Err(error) = target.cache_warm(Operation::Discard).await {
                errors.push(error);
            }
            for source in sources {
                if let Err(error) = source.cache_warm(Operation::Release).await {
                    errors.push(error);
                }
            }
            for owner in &owners {
                if let Err(error) = owner.hook("uniserve.cache_unstage", "[]").await {
                    errors.push(error);
                }
            }
        }
        if let Err(error) = resume_prefill(sources).await {
            errors.push(error);
        }
    }
    for owner in owners {
        if let Err(error) = owner.hook("uniserve.cache_session", "false").await {
            errors.push(error);
        }
    }
    if !errors.is_empty() {
        if let Err(error) = resume_prefill(sources).await {
            errors.push(error);
        }
        if let Err(error) = result {
            errors.insert(0, error);
        }
        return Err(errors.join("; "));
    }
    result
}

/// Restore the configured prefill budget after promotion or preparation failure.
pub(super) async fn resume_prefill(sources: &[Arc<EngineClient>]) -> Result<(), String> {
    let mut errors = Vec::new();
    for source in sources {
        if let Err(error) = source.cache_warm(Operation::Prefill { tokens: None }).await {
            errors.push(error);
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

async fn rounds(
    sources: &[Arc<EngineClient>],
    target: &Arc<EngineClient>,
    mut reuse: Reuse,
    tail_tokens: Option<u32>,
    prefill_tokens: Option<u32>,
    host: bool,
) -> Result<Report, String> {
    let mut report = Report {
        staged: host,
        recompute: (reuse == Reuse::Recompute)
            .then(|| "the update recomputes in-flight state".into()),
        ..Report::default()
    };
    let keep = reuse == Reuse::Keep;
    let mut copied: HashMap<(u64, u64), u32> = HashMap::new();
    // A warming operation owns the transfer contexts across its rounds.
    // Individual task completion must not tear down and recreate them.
    for owner in sources.iter().chain(std::iter::once(target)) {
        owner.hook("uniserve.cache_session", "true").await?;
    }
    // Admission reserved room for live prefixes. Map any deferred writable
    // tail segments before leasing destination units from their pool.
    if !host {
        target
            .resize_cache(proto::request::wire::CacheChange::Grow)
            .await?;
    }
    loop {
        let mut snapshots = Vec::new();
        for source in sources {
            snapshots.extend(
                source
                    .cache_warm(Operation::Snapshot {
                        world: source.world(),
                        after: copied
                            .iter()
                            .filter(|((world, _), _)| *world == source.world())
                            .map(|((_, request), &tokens)| (*request, tokens))
                            .collect(),
                    })
                    .await?,
            );
        }
        let prefixes: Vec<_> = snapshots
            .iter()
            .flat_map(|snapshot| snapshot.prefixes.clone())
            .collect();
        let targets = if reuse == Reuse::Recompute {
            Vec::new()
        } else {
            target
                .cache_warm(Operation::Reserve {
                    prefixes: prefixes.clone(),
                    keep,
                    host,
                })
                .await?
        };
        let page_tokens = snapshots
            .iter()
            .chain(&targets)
            .map(|snapshot| snapshot.page_tokens)
            .max()
            .unwrap_or(1);
        report.limit_tokens = tail_tokens.unwrap_or(page_tokens);
        if host {
            // Retain the source layout as well as the successor layout.
            // Only these completed copies permit eviction before promotion.
            for source in sources {
                let prefixes = prefixes
                    .iter()
                    .filter(|prefix| prefix.world == source.world())
                    .cloned()
                    .collect();
                let backup = source.cache_warm(Operation::Stage { prefixes }).await?;
                match copy(source, &snapshots, &backup).await? {
                    CopyResult::Complete(bytes) => report.copied_bytes += bytes,
                    CopyResult::Recompute(reason) => return Err(reason),
                }
            }
        }
        match copy(target, &snapshots, &targets).await? {
            CopyResult::Complete(bytes) => report.copied_bytes += bytes,
            CopyResult::Recompute(reason) if !keep => {
                report.recompute = Some(reason);
                if !host {
                    return Ok(report);
                }
                // No copy was submitted for an unsupported encoding. Keep
                // the completed source backup, release the unused target
                // views, and reserve GPU capacity for ordinary recomputation.
                target.cache_warm(Operation::Discard).await?;
                target.hook("uniserve.cache_unstage", "[]").await?;
                reuse = Reuse::Recompute;
            }
            CopyResult::Recompute(reason) => {
                return Err(format!("in-flight state cannot be kept: {reason}"));
            }
        }
        target.cache_warm(Operation::Publish { host }).await?;
        for source in sources {
            source.cache_warm(Operation::Release).await?;
        }
        report.rounds += 1;

        let coverage = if reuse == Reuse::Recompute {
            // The successor computes fresh state. Its predecessor still
            // needs complete rollback copies before releasing device pages.
            let mut coverage = Vec::new();
            for source in sources {
                coverage.extend(
                    source
                        .cache_warm(Operation::Coverage {
                            prefixes: prefixes
                                .iter()
                                .filter(|prefix| prefix.world == source.world())
                                .cloned()
                                .collect(),
                            keep: true,
                        })
                        .await?,
                );
            }
            coverage
        } else {
            target
                .cache_warm(Operation::Coverage { prefixes, keep })
                .await?
        };
        let cached: HashMap<_, _> = coverage
            .iter()
            .flat_map(|snapshot| &snapshot.prefixes)
            .map(|prefix| ((prefix.world, prefix.request), prefix.cached))
            .collect();
        let resumes: HashMap<_, _> = coverage
            .iter()
            .flat_map(|snapshot| &snapshot.prefixes)
            .map(|prefix| ((prefix.world, prefix.request), prefix.resume))
            .collect();
        let mut live = Vec::new();
        let mut live_sources = Vec::new();
        report.tail_tokens = 0;
        report.warmed_requests = 0;
        report.warmed_tokens = 0;
        for source in sources {
            for request in source.describe_requests().await? {
                let identity = (source.world(), request.request_id);
                live_sources.push(identity);
                if let Some(&resume) = resumes.get(&identity) {
                    live.push(resume);
                }
                let warmed = (*cached.get(&identity).unwrap_or(&0)).min(request.tokens);
                if reuse != Reuse::Recompute {
                    report.warmed_tokens += u64::from(warmed);
                    report.warmed_requests += u32::from(warmed > 0);
                }
                report.tail_tokens = report.tail_tokens.max(
                    request
                        .tokens
                        .saturating_sub(*cached.get(&identity).unwrap_or(&0)),
                );
            }
        }
        target
            .cache_warm(Operation::Retain { requests: live })
            .await?;
        if host {
            let payload =
                serde_json::to_string(&live_sources).map_err(|error| error.to_string())?;
            for owner in sources.iter().chain(std::iter::once(target)) {
                owner.hook("uniserve.cache_retain_staged", &payload).await?;
            }
            report.backup.clear();
            for source in sources {
                let ranks = staged(source).await?;
                let mut prefixes: Vec<_> = snapshots
                    .iter()
                    .flat_map(|snapshot| &snapshot.prefixes)
                    .filter(|prefix| {
                        prefix.world == source.world()
                            && live_sources.contains(&(prefix.world, prefix.request))
                    })
                    .cloned()
                    .collect();
                for prefix in &mut prefixes {
                    prefix.pages = ranks
                        .iter()
                        .flat_map(|rank| &rank.allocations)
                        .filter(|(_, page)| page.request == (prefix.world, prefix.request))
                        .map(|(_, page)| page.clone())
                        .collect();
                }
                report.backup.push(Snapshot {
                    prefixes,
                    ranks,
                    page_tokens,
                    nodes: snapshots
                        .iter()
                        .flat_map(|snapshot| snapshot.nodes.clone())
                        .collect(),
                });
            }
        }
        copied = cached;
        tracing::info!(
            world = target.world(),
            rounds = report.rounds,
            copied_bytes = report.copied_bytes,
            tail_tokens = report.tail_tokens,
            warmed_requests = report.warmed_requests,
            warmed_tokens = report.warmed_tokens,
            "cache warming round completed"
        );
        if report.tail_tokens <= report.limit_tokens {
            return Ok(report);
        }
        if report.rounds == 1 {
            for source in sources {
                source
                    .cache_warm(Operation::Prefill {
                        tokens: Some(prefill_tokens.unwrap_or(page_tokens)),
                    })
                    .await?;
            }
        }
    }
}

/// Retained host keys kept across process retirement and a rollback restart.
pub(super) fn keys(snapshots: &[Snapshot]) -> Vec<Vec<u8>> {
    snapshots
        .iter()
        .flat_map(|snapshot| &snapshot.ranks)
        .flat_map(|rank| &rank.storage.storages)
        .flat_map(|storage| &storage.chunks)
        .map(|chunk| chunk.key)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|key| key.0.to_vec())
        .collect()
}

/// Load the successor's staged layout after the predecessors evicted their
/// copied tails. Publication follows physical copy completion on every rank.
pub(super) async fn materialize(target: &Arc<EngineClient>) -> Result<(), String> {
    target
        .resize_cache(proto::request::wire::CacheChange::Grow)
        .await?;
    let destinations = target.cache_warm(Operation::Materialize).await?;
    let ranks = staged(target).await?;
    let source = Snapshot {
        ranks,
        nodes: destinations
            .iter()
            .flat_map(|snapshot| snapshot.nodes.clone())
            .collect(),
        prefixes: Vec::new(),
        page_tokens: 1,
    };
    match copy(target, &[source], &destinations).await? {
        CopyResult::Complete(_) => {}
        CopyResult::Recompute(reason) => return Err(reason),
    }
    target
        .cache_warm(Operation::Publish { host: false })
        .await?;
    target.hook("uniserve.cache_unstage", "[]").await?;
    Ok(())
}

async fn staged(target: &Arc<EngineClient>) -> Result<Vec<cache::Rank>, String> {
    Ok(target
        .hook("uniserve.cache_staged", "null")
        .await?
        .into_iter()
        .map(|value| {
            serde_json::from_str::<Option<cache::Rank>>(&value).map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect())
}

/// Restore complete source pages before releasing its held continuations.
/// The caller has retired the successor and restored the source's capacity.
pub(super) async fn restore(target: &Arc<EngineClient>, backup: &[Snapshot]) -> Result<(), String> {
    let mut prefixes = BTreeMap::new();
    for prefix in backup
        .iter()
        .flat_map(|snapshot| &snapshot.prefixes)
        .filter(|prefix| prefix.world == target.world())
    {
        prefixes.insert((prefix.world, prefix.request), prefix.clone());
    }
    if prefixes.is_empty() {
        return Ok(());
    }
    target
        .resize_cache(proto::request::wire::CacheChange::Grow)
        .await?;
    let destinations = target
        .cache_warm(Operation::Reserve {
            prefixes: prefixes.into_values().collect(),
            keep: true,
            host: false,
        })
        .await?;
    match copy(target, backup, &destinations).await? {
        CopyResult::Complete(_) => {}
        CopyResult::Recompute(reason) => return Err(reason),
    }
    target
        .cache_warm(Operation::Publish { host: false })
        .await?;
    let payload = serde_json::to_string(&keys(backup)).map_err(|error| error.to_string())?;
    target.hook("uniserve.cache_unstage", &payload).await?;
    Ok(())
}

enum CopyResult {
    Complete(u64),
    Recompute(String),
}

async fn copy(
    target: &EngineClient,
    sources: &[Snapshot],
    targets: &[Snapshot],
) -> Result<CopyResult, String> {
    let nodes: BTreeMap<_, _> = sources
        .iter()
        .chain(targets)
        .flat_map(|snapshot| snapshot.nodes.clone())
        .collect();
    let source_ranks: Vec<_> = sources
        .iter()
        .flat_map(|snapshot| snapshot.ranks.iter())
        .collect();
    let mut hosts: BTreeMap<&str, Vec<&cache::Rank>> = BTreeMap::new();
    for rank in targets.iter().flat_map(|snapshot| &snapshot.ranks) {
        hosts.entry(&rank.host).or_default().push(rank);
    }
    let views: Vec<_> = hosts
        .keys()
        .map(|host| {
            let ranks = source_ranks
                .iter()
                .map(|source| {
                    let mut rank = source.storage.clone();
                    if source.host != *host {
                        let endpoint = nodes
                            .get(&source.host)
                            .ok_or_else(|| format!("no Storage Server for {}", source.host))?;
                        for storage in &mut rank.storages {
                            for chunk in &mut storage.chunks {
                                chunk.binding = Binding::Remote {
                                    endpoint: endpoint.clone(),
                                    at: match &chunk.binding {
                                        Binding::Rank => Placement::Device {
                                            gpu: rank.gpu.ok_or("a cache source has no GPU")?,
                                        },
                                        Binding::Local { at } | Binding::Remote { at, .. } => *at,
                                    },
                                };
                            }
                        }
                    }
                    Ok(rank)
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(cache::Sources {
                host: (*host).to_owned(),
                host_destination: hosts[host]
                    .iter()
                    .flat_map(|rank| &rank.storage.storages)
                    .flat_map(|storage| &storage.chunks)
                    .any(|chunk| {
                        matches!(
                            chunk.binding,
                            Binding::Local {
                                at: Placement::Host { .. }
                            }
                        )
                    }),
                sources: storage::manifest::weights::Sources {
                    world: World { ranks },
                    links: Vec::new(),
                },
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let payload = serde_json::to_string(&views).map_err(|error| error.to_string())?;
    let links: Vec<LinkBandwidth> = target
        .hook("uniserve.cache_links", &payload)
        .await?
        .into_iter()
        .map(|value| {
            serde_json::from_str::<Vec<LinkBandwidth>>(&value).map_err(|error| error.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect();
    let mut copies: BTreeMap<(String, u32), Vec<storage::protocol::Span>> = BTreeMap::new();
    let mut bytes = 0;
    for view in views {
        let mut ranks: Vec<storage::manifest::Rank> = Vec::new();
        let mut owners = HashMap::new();
        for owner in &hosts[view.host.as_str()] {
            let index = match ranks.iter().position(|rank| rank.gpu == owner.storage.gpu) {
                Some(index) => index,
                None => {
                    ranks.push(storage::manifest::Rank {
                        gpu: owner.storage.gpu,
                        storages: Vec::new(),
                    });
                    ranks.len() - 1
                }
            };
            for storage in &owner.storage.storages {
                for chunk in &storage.chunks {
                    owners.insert(chunk.key, (owner.worker_group.clone(), owner.rank));
                }
            }
            ranks[index].storages.extend(owner.storage.storages.clone());
        }
        let request = Request {
            source: view.sources.world,
            gpus: ranks.iter().filter_map(|rank| rank.gpu).collect(),
            target: World { ranks },
            links: links.clone(),
        };
        let manifest = Manifest::plan(&request).map_err(|error| error.to_string())?;
        let mut conversion = None;
        for missing in &manifest.missing {
            let storage =
                &request.target.ranks[missing.rank as usize].storages[missing.storage as usize];
            let field = storage
                .layout
                .fields
                .iter()
                .find(|field| field.tensor == missing.tensor)
                .ok_or("a missing cache field has no layout")?;
            let encoded = !matches!(field.encoding, Encoding::Affine)
                || request
                    .source
                    .ranks
                    .iter()
                    .flat_map(|rank| &rank.storages)
                    .flat_map(|storage| &storage.layout.fields)
                    .any(|source| {
                        source.tensor == field.tensor
                            && !matches!(source.encoding, Encoding::Affine)
                    });
            if encoded {
                conversion = Some(format!(
                    "cache encoding has no exact conversion for {}",
                    missing.tensor
                ));
                continue;
            }
            return Err(format!(
                "no source holds cache field {} at {:?}",
                missing.tensor, missing.logical
            ));
        }
        if let Some(reason) = conversion {
            return Ok(CopyResult::Recompute(reason));
        }
        if !manifest.encodes.is_empty() || !manifest.decodes.is_empty() {
            return Err("cache conversion requires a registered numerical converter".into());
        }
        bytes += manifest.copied_bytes();
        for span in manifest.bind(&request, &request.source).spans() {
            let span = span.map_err(|error| error.to_string())?;
            let Region::Tensor { key, .. } = &span.destination else {
                return Err("a cache copy has no retained target".into());
            };
            let owner = owners.get(key).ok_or("a cache copy has no target owner")?;
            copies.entry(owner.clone()).or_default().push(span);
        }
    }
    let copies: Vec<_> = copies
        .into_iter()
        .map(|((worker_group, rank), spans)| cache::Copy {
            worker_group,
            rank,
            spans,
        })
        .collect();
    let payload = serde_json::to_string(&copies).map_err(|error| error.to_string())?;
    target.hook("uniserve.cache_transfer", &payload).await?;
    loop {
        let done = target
            .hook("uniserve.cache_poll", "null")
            .await?
            .into_iter()
            .map(|value| serde_json::from_str::<bool>(&value).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .all(|done| done);
        if done {
            return Ok(CopyResult::Complete(bytes));
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::client::EngineState;
    use crate::test_support::{SERVED_MODEL, sim_client, sim_core};

    #[tokio::test]
    async fn a_prepared_engine_answers_cache_commands_before_promotion() {
        let mut config = engine::EngineConfig::sim(SERVED_MODEL);
        config.predecessor = Some(BTreeMap::new());
        let client = sim_client(
            (0..2)
                .map(|_| sim_core(config.clone(), engine::SimEngine::new()))
                .collect(),
        );
        assert_eq!(client.state(), EngineState::Standby);
        let snapshots = tokio::time::timeout(
            Duration::from_secs(10),
            client.cache_warm(Operation::Reserve {
                prefixes: Vec::new(),
                keep: false,
                host: false,
            }),
        )
        .await
        .expect("prepared cache response")
        .expect("empty cache reservation");
        assert_eq!(snapshots.len(), 2);
        assert!(
            snapshots
                .iter()
                .all(|snapshot| snapshot.prefixes.is_empty())
        );
        assert_eq!(client.state(), EngineState::Standby);
        tokio::time::timeout(
            Duration::from_secs(10),
            client.promote(Vec::new(), false, false),
        )
        .await
        .expect("promotion response")
        .expect("promotion");
        assert_eq!(client.state(), EngineState::Ready);
        client.shutdown();
    }
}
