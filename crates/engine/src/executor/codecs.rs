//! The encoders that may read each video pool's decoded media units
//! ([`video_unit_codecs`]): a decoded unit is a host product named in one
//! host's namespace, so a decoder pairs with an encoder replica only when, at
//! every position of a round, the ranks that decode and encode the unit share a
//! host. Requests are routed only to admissible pairings, and a decoding pool
//! without one is refused at startup.

use crate::executor::ExecutorGroupId;

/// Returns, for each executor that decodes video, the encoder replicas that
/// keep every media unit it decodes on the host that decoded it.
///
/// A decoded media unit is a host product that its encoder reads in place
/// from shared storage, which is named in one host's namespace. A round deals
/// its units to each component's ranks in order, `units_per_rank` consecutive
/// positions each, so position `p` is decoded by the decoder's rank
/// `p / per_decoder` and encoded by the encoder's rank `p / per_encoder`. A
/// pairing is admissible when the encoder covers a whole round and those two
/// ranks share a host at every position. Requests are routed only to
/// admissible pairings, and a decoding executor without one is refused at
/// startup, naming the unit and the hosts it would cross.
///
/// Returns an empty map when the routing lacks a video decoding or a video
/// encoding component.
pub(crate) fn video_unit_codecs(
    routing: &std::collections::BTreeMap<ipc::MediaCall, String>,
    placements: &[(
        ExecutorGroupId,
        Vec<String>,
        std::collections::BTreeMap<String, crate::ComponentConfig>,
    )],
) -> anyhow::Result<
    std::collections::BTreeMap<ExecutorGroupId, std::collections::BTreeSet<ExecutorGroupId>>,
> {
    let mut pairings = std::collections::BTreeMap::new();
    let (Some(decoder), Some(encoder)) = (
        routing.get(&ipc::MediaCall::VideoDecoding),
        routing.get(&ipc::MediaCall::VideoEncoding),
    ) else {
        return Ok(pairings);
    };
    let holders = |component: &str| {
        placements
            .iter()
            .filter_map(|(id, ranks, components)| {
                components
                    .get(component)
                    .map(|config| (id.clone(), ranks, config))
            })
            .collect::<Vec<_>>()
    };
    let encoders = holders(encoder);
    for (decoder_worker, decoder_ranks, decoder_config) in holders(decoder) {
        let per_decoder = decoder_config.units_per_rank.max(1);
        let round_units = decoder_config.ranks.len() * per_decoder;
        let mut admissible = std::collections::BTreeSet::new();
        let mut refusals = Vec::new();
        for (encoder_worker, encoder_ranks, encoder_config) in &encoders {
            let per_encoder = encoder_config.units_per_rank.max(1);
            let coverage = encoder_config.ranks.len() * per_encoder;
            if coverage < round_units {
                refusals.push(format!(
                    "executor {encoder_worker} encodes {coverage} units per round and executor \
                     {decoder_worker} decodes {round_units}"
                ));
                continue;
            }
            // `coverage >= round_units` keeps `position / per_encoder` within
            // the encoder's ranks.
            let crossing = (0..round_units).find_map(|position| {
                let decoding_rank = decoder_config.ranks[position / per_decoder];
                let encoding_rank = encoder_config.ranks[position / per_encoder];
                let decoding_host = decoder_ranks
                    .get(decoding_rank)
                    .map_or("<unplaced>", String::as_str);
                let encoding_host = encoder_ranks
                    .get(encoding_rank)
                    .map_or("<unplaced>", String::as_str);
                (decoding_host != encoding_host).then(|| {
                    format!(
                        "media unit {position} is decoded by rank {decoding_rank} of executor \
                         {decoder_worker} on host {decoding_host} and would be encoded by rank \
                         {encoding_rank} of executor {encoder_worker} on host {encoding_host}"
                    )
                })
            });
            match crossing {
                Some(refusal) => refusals.push(refusal),
                None => {
                    admissible.insert(encoder_worker.clone());
                }
            }
        }
        anyhow::ensure!(
            !admissible.is_empty(),
            "no video codec keeps the media units of executor {decoder_worker} on their host, \
             where their encoder reads them through shared storage: {}",
            refusals.join("; ")
        );
        pairings.insert(decoder_worker.clone(), admissible);
    }
    Ok(pairings)
}

#[cfg(test)]
mod tests {
    use super::video_unit_codecs;
    use crate::ExecutorRank;
    use crate::executor::{ComponentConfig, ExecutorGroupId};
    use ipc::MediaCall;
    use std::collections::BTreeMap;

    fn rank(host: &str, device: &str) -> ExecutorRank {
        ExecutorRank {
            host: host.to_owned(),
            device: device.to_owned(),
        }
    }

    fn distributed(ranks: Vec<usize>, units_per_rank: usize) -> ComponentConfig {
        ComponentConfig {
            ranks,
            parallel_config: Default::default(),
            distribution: Some(ipc::ComponentDistribution::TemporalUnits),
            units_per_rank,
            weights: Default::default(),
        }
    }

    fn routing() -> BTreeMap<MediaCall, String> {
        BTreeMap::from([
            (MediaCall::VideoDecoding, "video_decoder".to_owned()),
            (MediaCall::VideoEncoding, "video_codec".to_owned()),
        ])
    }

    /// The pairing of a deployment described by its pools' placed ranks: the
    /// function itself reads only the ranks' hosts.
    fn pairing(
        placements: &[(
            &ExecutorGroupId,
            &[ExecutorRank],
            &BTreeMap<String, ComponentConfig>,
        )],
    ) -> anyhow::Result<BTreeMap<ExecutorGroupId, std::collections::BTreeSet<ExecutorGroupId>>> {
        let placed = placements
            .iter()
            .map(|(id, ranks, components)| {
                (
                    (*id).clone(),
                    ranks.iter().map(|rank| rank.host.clone()).collect(),
                    (*components).clone(),
                )
            })
            .collect::<Vec<_>>();
        video_unit_codecs(&routing(), &placed)
    }

    #[test]
    fn units_encoded_on_their_own_host_are_admitted() {
        let model = ExecutorGroupId("model".to_owned());
        let host = ExecutorGroupId("host".to_owned());
        let model_ranks = vec![
            rank("a", "cuda:0"),
            rank("a", "cuda:1"),
            rank("b", "cuda:0"),
            rank("b", "cuda:1"),
        ];
        let cpu_workers = vec![
            rank("a", "cpu"),
            rank("a", "cpu"),
            rank("b", "cpu"),
            rank("b", "cpu"),
        ];
        let model_components =
            BTreeMap::from([("video_decoder".to_owned(), distributed(vec![0, 1, 2, 3], 1))]);
        let host_components =
            BTreeMap::from([("video_codec".to_owned(), distributed(vec![0, 1, 2, 3], 1))]);
        let pairings = pairing(&[
            (&model, &model_ranks, &model_components),
            (&host, &cpu_workers, &host_components),
        ])
        .expect("each unit is encoded on its own host");
        assert_eq!(pairings[&model], [host].into());
    }

    #[test]
    fn a_decoder_dealing_several_units_per_rank_keeps_them_on_its_host() {
        // Two units per decoding rank: positions 0 to 3 come from host a's
        // decoders and go to host a's encoders, positions 4 to 7 likewise on
        // host b.
        let model = ExecutorGroupId("model".to_owned());
        let host = ExecutorGroupId("host".to_owned());
        let model_ranks = vec![
            rank("a", "cuda:0"),
            rank("a", "cuda:1"),
            rank("b", "cuda:0"),
            rank("b", "cuda:1"),
        ];
        let cpu_workers = [["a"; 4], ["b"; 4]]
            .concat()
            .into_iter()
            .map(|host| rank(host, "cpu"))
            .collect::<Vec<_>>();
        let model_components =
            BTreeMap::from([("video_decoder".to_owned(), distributed(vec![0, 1, 2, 3], 2))]);
        let host_components =
            BTreeMap::from([("video_codec".to_owned(), distributed((0..8).collect(), 1))]);
        pairing(&[
            (&model, &model_ranks, &model_components),
            (&host, &cpu_workers, &host_components),
        ])
        .expect("each host's units are encoded on it");

        // Host a holding three encoders sends its fourth unit to host b.
        let cpu_workers = [["a"; 3].as_slice(), ["b"; 5].as_slice()]
            .concat()
            .into_iter()
            .map(|host| rank(host, "cpu"))
            .collect::<Vec<_>>();
        let error = pairing(&[
            (&model, &model_ranks, &model_components),
            (&host, &cpu_workers, &host_components),
        ])
        .expect_err("the fourth unit leaves host a");
        assert!(error.to_string().contains("media unit 3"), "{error}");
    }

    #[test]
    fn every_shipped_deployment_encodes_its_units_on_their_host() {
        // Every video deployment file shipped in `configs/minimax_h3/` must
        // pass the placement checks the engine applies at startup, including
        // the pairing of every decoding executor with an encoder on its host.
        check_shipped_deployments("minimax_h3");
    }

    fn check_shipped_deployments(family: &str) {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../configs")
            .join(family);
        let mut checked = 0;
        for entry in std::fs::read_dir(&root).expect("the deployment directory exists") {
            let path = entry.expect("readable entry").path();
            let workers = crate::Deployment::read(path.to_str().expect("UTF-8 path"))
                .and_then(|deployment| deployment.worker_groups("localhost"))
                .unwrap_or_else(|error| panic!("{}: {error:#}", path.display()));
            let ids = workers
                .iter()
                .map(|worker| worker.id.clone())
                .collect::<Vec<_>>();
            let placements = workers
                .iter()
                .zip(&ids)
                .map(|(worker, id)| (id, worker.ranks.as_slice(), &worker.components))
                .collect::<Vec<_>>();
            let pairings = pairing(&placements)
                .unwrap_or_else(|error| panic!("{}: {error:#}", path.display()));
            let decoders = workers
                .iter()
                .filter(|worker| worker.components.contains_key("video_decoder"))
                .count();
            assert_eq!(pairings.len(), decoders, "{}", path.display());
            checked += 1;
        }
        assert!(checked > 0, "no {family} deployment was checked");
    }

    #[test]
    fn each_replica_decoder_pairs_with_the_encoders_on_its_host() {
        // Single-device flow replicas decode one unit per round, so every
        // round is position zero: an encoder replica is admissible only on
        // the replica's own host.
        let flows = [
            (
                ExecutorGroupId("flow-a".to_owned()),
                vec![rank("a", "cuda:0")],
            ),
            (
                ExecutorGroupId("flow-b".to_owned()),
                vec![rank("b", "cuda:0")],
            ),
        ];
        let encoders = [
            (
                ExecutorGroupId("encoder-a".to_owned()),
                vec![rank("a", "cpu")],
            ),
            (
                ExecutorGroupId("encoder-b".to_owned()),
                vec![rank("b", "cpu")],
            ),
        ];
        let decoder = BTreeMap::from([("video_decoder".to_owned(), distributed(vec![0], 1))]);
        let encoder = BTreeMap::from([("video_codec".to_owned(), distributed(vec![0], 1))]);
        let placements = flows
            .iter()
            .map(|(id, ranks)| (id, ranks.as_slice(), &decoder))
            .chain(
                encoders
                    .iter()
                    .map(|(id, ranks)| (id, ranks.as_slice(), &encoder)),
            )
            .collect::<Vec<_>>();

        let pairings = pairing(&placements).expect("every host has an encoder");

        assert_eq!(pairings[&flows[0].0], [encoders[0].0.clone()].into());
        assert_eq!(pairings[&flows[1].0], [encoders[1].0.clone()].into());

        // Without an encoder on host b, the replica there is refused by name
        // even though another replica pairs.
        let error = pairing(&placements[..3]).expect_err("host b has no encoder");
        let message = error.to_string();
        assert!(message.contains("executor flow-b"), "{message}");
        assert!(
            message.contains("host b") && message.contains("host a"),
            "{message}"
        );
    }
}
