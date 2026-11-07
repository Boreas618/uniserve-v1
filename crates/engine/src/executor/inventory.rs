//! What the mechanism learns of the pools an executor serves (4.2, 7.3).
//!
//! [`ExecutorInfo`] is what the scheduler plans against: each pool's capabilities, the capacities
//! the engine launched it with, each rank's host pools in launch order, and the codec pairing that
//! keeps a media unit on the host that decoded it. A deployment launched as processes derives those
//! values from the ranks' startup reports; a deployment served in process derives them from what its
//! executor publishes through `executor::Batches::inventory`. Both call [`pool_info`] for the pool's
//! agreed report, so one deployment states one `ExecutorInfo` whichever way it is served.
//!
//! Nothing here is defaulted that the inventory could state. A pool whose facts are missing is
//! refused by name: a guessed rank count, cache geometry or sequence bound makes the host KV path
//! or the device's row addressing vanish at runtime instead of failing here.

use std::collections::BTreeMap;

use anyhow::ensure;
use ipc::MainProcessInfo;

use crate::ExecutorGroupId;
use crate::executor::codecs::video_unit_codecs;
use crate::executor::{ExecutorInfo, PoolLayout, RankEncoderPools};

impl ExecutorInfo {
    /// The capability information of the pools `pools` describes, as the mechanism reads it.
    ///
    /// Fails, naming the pool, when a pool reports no rank, states no longest sequence, states a
    /// longest sequence another pool does not, or when its layout disagrees with the ranks it
    /// reports; fails as [`ExecutorInfo::from_workers`] does when two pools share an identity or
    /// replicas of one component expose different numerical outputs.
    pub fn from_inventory(pools: &[executor::batch::Inventory]) -> anyhow::Result<Self> {
        ensure!(
            !pools.is_empty(),
            "an executor serves at least one pool, and it reports none"
        );

        let mut reports = Vec::with_capacity(pools.len());
        for pool in pools {
            ensure!(
                !pool.ranks.is_empty(),
                "pool {} reports no rank",
                named(&pool.group)
            );
            let info = pool_info(&pool.ranks)
                .map_err(|error| error.context(format!("pool {}", named(&pool.group))))?;
            reports.push((ExecutorGroupId(pool.group.clone()), info));
        }
        // Pool identities are unique and replicas of one component agree, which
        // is the same check every launch passes (`from_workers`).
        let mut info = Self::from_workers(reports)?;

        // The attention descriptor's static bound is one number for the
        // deployment: every group is launched for the same longest sequence, so
        // a pool that states another one is a deployment this cannot describe.
        let mut longest: Option<(u32, &str)> = None;
        for pool in pools {
            ensure!(
                pool.max_model_len > 0,
                "pool {} states no longest sequence",
                named(&pool.group)
            );
            longest = Some(match longest {
                Some((length, other)) => {
                    ensure!(
                        length == pool.max_model_len,
                        "pool {} was built for {} tokens and pool {} for {}",
                        named(&pool.group),
                        pool.max_model_len,
                        named(other),
                        length
                    );
                    (length, other)
                }
                None => (pool.max_model_len, pool.group.as_str()),
            });
        }
        info.max_model_len = longest.map_or(0, |(length, _)| length);

        // The capacities come from the launch, not from the ranks' reports: a
        // rank reports what it holds, not how many it was launched for.
        for pool in pools {
            let id = ExecutorGroupId(pool.group.clone());
            ensure!(
                pool.layout.ranks.max(1) == pool.ranks.len(),
                "pool {} was launched for {} ranks and reports {}",
                named(&pool.group),
                pool.layout.ranks,
                pool.ranks.len()
            );
            info.layouts.insert(
                id.clone(),
                PoolLayout {
                    queue_depth: pool.layout.queue_depth.max(1),
                    ranks: pool.ranks.len(),
                    components: pool.layout.components.clone(),
                },
            );
            // Each rank's own reports, in launch order: media buffers, the
            // prefix cache and its host page pools, and the host latent and
            // encoder pools resolve rank by rank.
            info.rank_buffer_bytes.insert(
                id.clone(),
                pool.ranks
                    .iter()
                    .map(|rank| rank.buffer_pool_bytes)
                    .collect(),
            );
            info.rank_caches.insert(
                id.clone(),
                pool.ranks
                    .iter()
                    .map(|rank| (rank.endpoint.host.clone(), rank.prefix_cache.clone()))
                    .collect(),
            );
            info.rank_latents.insert(
                id.clone(),
                pool.ranks
                    .iter()
                    .map(|rank| {
                        (
                            rank.endpoint.host.clone(),
                            rank.latent_pool.as_deref().cloned(),
                        )
                    })
                    .collect(),
            );
            info.rank_encoders.insert(
                id,
                pool.ranks
                    .iter()
                    .map(|rank| (rank.endpoint.host.clone(), rank.encoder_pool.clone()))
                    .collect::<RankEncoderPools>(),
            );
        }

        // The video graph may span pools: a pool decodes and another encodes,
        // and an encoder is a candidate for a decoder only when every media
        // unit it would encode stays on the host that decoded it.
        let routing = info.media_routing()?;
        let layouts = pools
            .iter()
            .map(|pool| {
                (
                    ExecutorGroupId(pool.group.clone()),
                    pool.ranks
                        .iter()
                        .map(|rank| rank.endpoint.host.clone())
                        .collect::<Vec<_>>(),
                    pool.layout.components.clone(),
                )
            })
            .collect::<Vec<_>>();
        info.video_codecs = video_unit_codecs(&routing, &layouts)?;
        Ok(info)
    }
}

/// The one report a pool's ranks add up to.
///
/// The ranks derive every shared capacity (block size, request slots, cache units) collectively
/// before they report, so their reports differ only in the fields reconciled here: the buffer bytes
/// and transfer tickets a pool holds are its narrowest rank's, the image facts are those the ranks
/// holding them agree on, and the prefix cache's layers and heads are their union. The fields a rank
/// reports for itself alone — its host latent and encoder pool — are left for the rank-by-rank maps.
pub(crate) fn pool_info(ranks: &[MainProcessInfo]) -> anyhow::Result<MainProcessInfo> {
    ensure!(!ranks.is_empty(), "a pool has at least one rank");

    // Rank 0's report becomes the pool's.
    let mut info = ranks[0].clone();
    info.latent_pool = None;
    info.encoder_pool = None;

    // A rank's product storage follows the components it holds, so the pool
    // reports the smallest nonzero pool, which no rank that holds products
    // falls short of.
    info.buffer_pool_bytes = ranks
        .iter()
        .map(|rank| rank.buffer_pool_bytes)
        .filter(|bytes| *bytes > 0)
        .min()
        .unwrap_or(0);
    // The pool holds what its narrowest rank does.
    info.transfer_tickets = ranks
        .iter()
        .map(|rank| rank.transfer_tickets)
        .min()
        .unwrap_or(info.transfer_tickets);

    // A rank states the shapes the engine warms, and the image denoiser's
    // image-unconditional prefix, only for what it holds: the pool's are those
    // its stating ranks agree on.
    let mut image_warmups = Vec::new();
    let mut captured_images = None;
    let mut captured_latents = None;
    let mut image_capacity = None;
    let mut diffusion_lm_step_rows = 0;
    let mut image_unconditional = None;
    let mut image_encoders = BTreeMap::new();
    for (rank, stated) in ranks.iter().enumerate() {
        for (name, output) in stated.image_encoders.iter() {
            ensure!(
                image_encoders
                    .insert(name.clone(), *output)
                    .is_none_or(|known| known == *output),
                "rank {rank} states another {name} encoder output than another rank"
            );
        }
        if let Some(tokens) = &stated.image_unconditional_token_ids {
            ensure!(
                image_unconditional
                    .as_ref()
                    .is_none_or(|known| known == tokens),
                "rank {rank} states another image-unconditional prefix than another denoiser rank"
            );
            image_unconditional = Some(tokens.clone());
        }
        if !stated.image_warmups.is_empty() {
            ensure!(
                image_warmups.is_empty() || image_warmups == stated.image_warmups,
                "rank {rank} states other image warmup shapes than another denoiser rank"
            );
            image_warmups.clone_from(&stated.image_warmups);
        }
        if let Some(shapes) = &stated.captured_images {
            ensure!(
                captured_images.as_ref().is_none_or(|known| known == shapes),
                "rank {rank} states other captured image steps than another denoiser rank"
            );
            captured_images = Some(shapes.clone());
        }
        if let Some(sizes) = &stated.captured_latents {
            ensure!(
                captured_latents.as_ref().is_none_or(|known| known == sizes),
                "rank {rank} states other captured latent image sizes than another latent encoder rank"
            );
            captured_latents = Some(sizes.clone());
        }
        if let Some(capacity) = stated.image_capacity {
            ensure!(
                image_capacity.is_none_or(|known| known == capacity),
                "rank {rank} states another image capacity than another image rank"
            );
            image_capacity = Some(capacity);
        }
        if stated.diffusion_lm_step_rows > 0 {
            ensure!(
                diffusion_lm_step_rows == 0
                    || diffusion_lm_step_rows == stated.diffusion_lm_step_rows,
                "rank {rank} states other diffusion LM step rows than another rank"
            );
            diffusion_lm_step_rows = stated.diffusion_lm_step_rows;
        }
    }
    info.image_warmups = image_warmups;
    info.captured_images = captured_images;
    info.captured_latents = captured_latents;
    info.image_capacity = image_capacity;
    info.diffusion_lm_step_rows = diffusion_lm_step_rows;
    info.image_unconditional_token_ids = image_unconditional;
    info.image_encoders = Box::new(image_encoders);

    // Every logical cache layer, numbered from zero without a gap, belongs to
    // one cache group on every rank that stores it, and each group's KV heads
    // are covered from head 0 to `total_kv_heads` by the ranks storing them.
    // Overlapping regions are allowed; a gap is not. The pool reports the union
    // of its ranks' layers, which bounds exports.
    if let Some(cache) = &mut info.prefix_cache {
        let regions: Vec<_> = ranks
            .iter()
            .filter_map(|rank| rank.prefix_cache.as_ref())
            .collect();
        // Every rank describes the same cache groups in one order; the
        // coverage below reads group `index` of each.
        ensure!(
            regions
                .iter()
                .all(|region| region.groups.len() == cache.groups.len()),
            "the pool's ranks report different cache groups"
        );
        let mut owners = BTreeMap::new();
        for (index, group) in cache.groups.iter_mut().enumerate() {
            let mut layers: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
            for region in &regions {
                let local = &region.groups[index];
                for layer in &local.layer_ids {
                    layers.entry(*layer).or_default().push((
                        local.kv_head_offset,
                        local.kv_head_offset + local.num_kv_heads,
                    ));
                }
            }
            for (layer, mut heads) in layers {
                ensure!(
                    *owners.entry(layer).or_insert(index) == index,
                    "cache layer {layer} belongs to two cache groups"
                );
                heads.sort_unstable();
                let mut covered = 0;
                for (start, end) in heads {
                    ensure!(
                        start <= covered,
                        "the pool's cache regions leave a head gap"
                    );
                    covered = covered.max(end);
                }
                ensure!(
                    covered == group.total_kv_heads,
                    "the pool's cache regions do not cover the heads of layer {layer}"
                );
            }
            group.layer_ids = owners
                .iter()
                .filter(|(_, owner)| **owner == index)
                .map(|(layer, _)| *layer)
                .collect();
        }
        ensure!(
            owners.keys().copied().eq(0..owners.len() as u32),
            "the pool's cache regions do not cover every logical cache layer"
        );
    }
    Ok(info)
}

/// A pool's name as a refusal states it.
fn named(group: &str) -> &str {
    if group.is_empty() { "<unnamed>" } else { group }
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests {
    use super::*;
    use ipc::{ComponentConfig, MainProcessEndpoint, MediaCall};
    use std::collections::BTreeSet;

    /// A rank's report, with the fields a mechanism reads distinguishable per
    /// rank so an order swapped by mistake fails a test.
    fn rank(
        host: &str,
        buffer_pool_bytes: u64,
        prefix_cache: Option<ipc::PrefixCacheInfo>,
    ) -> MainProcessInfo {
        MainProcessInfo {
            endpoint: MainProcessEndpoint {
                host: host.to_owned(),
                address_space: format!("{host}-space"),
                incarnation: format!("{host}-rank"),
            },
            buffer_pool_bytes,
            prefix_cache,
            ..MainProcessInfo::default()
        }
    }

    fn components(names: &[(&str, &[usize])]) -> BTreeMap<String, ComponentConfig> {
        names
            .iter()
            .map(|(name, ranks)| {
                (
                    (*name).to_owned(),
                    ComponentConfig {
                        ranks: ranks.to_vec(),
                        parallel_config: Default::default(),
                        units_per_rank: 1,
                        weights: Default::default(),
                        distribution: None,
                    },
                )
            })
            .collect()
    }

    fn pool(
        group: &str,
        ranks: Vec<MainProcessInfo>,
        placed_ranks: usize,
        components: BTreeMap<String, ComponentConfig>,
        max_model_len: u32,
    ) -> executor::batch::Inventory {
        executor::batch::Inventory {
            group: group.to_owned(),
            ranks,
            layout: executor::batch::PoolLayout {
                queue_depth: 2,
                ranks: placed_ranks,
                components,
            },
            max_model_len,
        }
    }

    /// A media pool that decodes and encodes on one host, so its own encoder
    /// is an admissible pairing for its decoder.
    fn media_pool(host: &str) -> executor::batch::Inventory {
        let mut info = rank(host, 64, None);
        info.media_components = BTreeMap::from([
            (MediaCall::VideoDecoding, "decode".to_owned()),
            (MediaCall::VideoEncoding, "encode".to_owned()),
        ]);
        info.components = ["decode", "encode"]
            .into_iter()
            .map(|name| ipc::ComponentInfo {
                name: name.to_owned(),
                outputs: Vec::new(),
            })
            .collect();
        pool(
            "media",
            vec![info],
            1,
            components(&[("decode", &[0]), ("encode", &[0])]),
            4096,
        )
    }

    #[test]
    fn a_pool_states_its_ranks_in_launch_order_and_its_launch_capacities() {
        let text = pool(
            "text",
            vec![rank("gpu-a", 128, None), rank("gpu-b", 96, None)],
            2,
            components(&[("text", &[0, 1])]),
            4096,
        );
        let info = ExecutorInfo::from_inventory(&[text]).expect("one pool's facts describe it");

        assert_eq!(info.workers.len(), 1);
        let id = ExecutorGroupId("text".to_owned());
        assert_eq!(info.workers[0].0, id);
        assert_eq!(
            info.rank_buffer_bytes(&id),
            vec![128, 96],
            "each rank backs its own buffer pool, in launch order"
        );
        assert_eq!(
            info.rank_caches.get(&id).map(|ranks| ranks
                .iter()
                .map(|(host, _)| host.as_str())
                .collect::<Vec<_>>()),
            Some(vec!["gpu-a", "gpu-b"]),
            "the caches are the ranks' own, in launch order"
        );
        let layout = info.layouts.get(&id).expect("the pool's launch");
        assert_eq!(layout.ranks, 2);
        assert_eq!(layout.queue_depth, 2);
        assert_eq!(layout.components, components(&[("text", &[0, 1])]));
        assert_eq!(info.max_model_len, 4096);
        assert_eq!(info.rank_count(&id), 2);
    }

    /// The attention descriptor's bound is the pool's own launch fact: a pool
    /// that states none is refused by name, rather than shaped to a bound
    /// nobody gave, and two pools of one deployment must agree on it.
    #[test]
    fn a_pools_longest_sequence_must_be_stated_and_agreed() {
        let silent = pool(
            "text",
            vec![rank("gpu-a", 128, None)],
            1,
            components(&[("text", &[0])]),
            0,
        );
        let error = ExecutorInfo::from_inventory(&[silent]).expect_err("a pool that states none");
        assert!(
            format!("{error:#}").contains("pool text states no longest sequence"),
            "{error:#}"
        );

        let narrow = pool(
            "text",
            vec![rank("gpu-a", 128, None)],
            1,
            components(&[("text", &[0])]),
            4096,
        );
        let wide = pool(
            "media",
            vec![rank("gpu-b", 64, None)],
            1,
            components(&[("text", &[0])]),
            8192,
        );
        let error = ExecutorInfo::from_inventory(&[narrow, wide]).expect_err("two pools disagree");
        let named = format!("{error:#}");
        assert!(
            named.contains("pool media") && named.contains("pool text"),
            "the disagreement names both pools: {named}"
        );
        assert!(
            named.contains("8192") && named.contains("4096"),
            "and the length each was built for: {named}"
        );
    }

    #[test]
    fn a_pools_report_is_rank_zeros_with_what_its_ranks_state_jointly() {
        let mut narrow = rank("gpu-a", 128, None);
        narrow.transfer_tickets = 4;
        let mut wide = rank("gpu-b", 96, None);
        wide.transfer_tickets = 8;
        let info = pool_info(&[narrow, wide]).expect("two ranks that agree");
        assert_eq!(
            info.buffer_pool_bytes, 96,
            "a pool holds what its narrowest rank does"
        );
        assert_eq!(info.transfer_tickets, 4);
        assert_eq!(
            info.endpoint.host, "gpu-a",
            "the pool's identity is its first rank's"
        );
    }

    #[test]
    fn a_decoder_pairs_only_with_encoders_that_stay_on_its_host() {
        let info = ExecutorInfo::from_inventory(&[media_pool("gpu-a")])
            .expect("a media pool's own encoder is admissible");
        let id = ExecutorGroupId("media".to_owned());
        assert_eq!(
            info.video_codecs.get(&id),
            Some(&BTreeSet::from([id.clone()])),
            "the encoder on the decoding host keeps every unit there"
        );
    }

    #[test]
    fn a_rank_report_its_pool_cannot_reconcile_is_refused_by_name() {
        // The refusals that name the pool are the ones `from_inventory` makes.
        for (pools, named) in [
            (
                vec![pool("no-ranks", Vec::new(), 1, components(&[]), 4096)],
                "no-ranks",
            ),
            (
                vec![pool(
                    "unbounded",
                    vec![rank("gpu-a", 64, None)],
                    1,
                    components(&[("text", &[0])]),
                    0,
                )],
                "unbounded",
            ),
            (
                vec![
                    pool(
                        "short",
                        vec![rank("gpu-a", 64, None)],
                        1,
                        components(&[("text", &[0])]),
                        4096,
                    ),
                    pool(
                        "long",
                        vec![rank("gpu-b", 64, None)],
                        1,
                        components(&[("text", &[0])]),
                        8192,
                    ),
                ],
                "long",
            ),
        ] {
            let message = ExecutorInfo::from_inventory(&pools)
                .err()
                .map(|error| error.to_string())
                .unwrap_or_default();
            assert!(
                message.contains(named),
                "the refusal names the pool {named}: {message}"
            );
        }
    }
}
