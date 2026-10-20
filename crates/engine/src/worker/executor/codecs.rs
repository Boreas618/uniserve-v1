//! The encoders that may read each video worker's decoded media units
//! ([`video_unit_codecs`]): a decoded unit is a host product named in one
//! host's namespace, so a decoder pairs with an encoder replica only when, at
//! every position of a round, the ranks that decode and encode the unit share a
//! host. Requests are routed only to admissible pairings, and a decoding worker
//! without one is refused at startup.

use crate::executor::WorkerId;

/// Returns, for each worker that decodes video, the encoder replicas that
/// keep every media unit it decodes on the host that decoded it.
///
/// A decoded media unit is a host product that its encoder reads in place
/// from shared storage, which is named in one host's namespace. A round deals
/// its units to each component's ranks in order, `units_per_rank` consecutive
/// positions each, so position `p` is decoded by the decoder's rank
/// `p / per_decoder` and encoded by the encoder's rank `p / per_encoder`. A
/// pairing is admissible when the encoder covers a whole round and those two
/// ranks share a host at every position. Requests are routed only to
/// admissible pairings, and a decoding worker without one is refused at
/// startup, naming the unit and the hosts it would cross.
///
/// Returns an empty map when the routing lacks a video decoding or a video
/// encoding component.
pub(crate) fn video_unit_codecs(
    routing: &std::collections::BTreeMap<uniserve_ipc::MediaCall, String>,
    placements: &[(
        &WorkerId,
        &[crate::WorkerRank],
        &std::collections::BTreeMap<String, crate::ComponentConfig>,
    )],
) -> anyhow::Result<std::collections::BTreeMap<WorkerId, std::collections::BTreeSet<WorkerId>>> {
    let mut pairings = std::collections::BTreeMap::new();
    let (Some(decoder), Some(encoder)) = (
        routing.get(&uniserve_ipc::MediaCall::VideoDecoding),
        routing.get(&uniserve_ipc::MediaCall::VideoEncoding),
    ) else {
        return Ok(pairings);
    };
    let holders = |component: &str| {
        placements
            .iter()
            .filter_map(|(id, ranks, components)| {
                components
                    .get(component)
                    .map(|config| (*id, *ranks, config))
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
                    "worker {encoder_worker} encodes {coverage} units per round and worker \
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
                    .map_or("<unplaced>", |rank| rank.node.as_str());
                let encoding_host = encoder_ranks
                    .get(encoding_rank)
                    .map_or("<unplaced>", |rank| rank.node.as_str());
                (decoding_host != encoding_host).then(|| {
                    format!(
                        "media unit {position} is decoded by rank {decoding_rank} of worker \
                         {decoder_worker} on host {decoding_host} and would be encoded by rank \
                         {encoding_rank} of worker {encoder_worker} on host {encoding_host}"
                    )
                })
            });
            match crossing {
                Some(refusal) => refusals.push(refusal),
                None => {
                    admissible.insert((*encoder_worker).clone());
                }
            }
        }
        anyhow::ensure!(
            !admissible.is_empty(),
            "no video codec keeps the media units of worker {decoder_worker} on their host, \
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
    use crate::WorkerRank;
    use crate::executor::{ComponentConfig, WorkerId};
    use std::collections::BTreeMap;
    use uniserve_ipc::MediaCall;

    fn rank(node: &str, device: &str) -> WorkerRank {
        WorkerRank {
            node: node.to_owned(),
            device: device.to_owned(),
        }
    }

    fn distributed(ranks: Vec<usize>, units_per_rank: usize) -> ComponentConfig {
        ComponentConfig {
            ranks,
            parallel_config: Default::default(),
            distribution: Some(uniserve_core::ComponentDistribution::TemporalUnits),
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

    #[test]
    fn a_decoded_unit_names_the_host_worker_ranks_that_read_it() {
        // The decoder's product is read by the video codec on the host
        // worker: the encoder rank dealt this decoder rank's position is
        // named, none of the decoder's own ranks, and the host worker's
        // slots follow the model worker's run. A consumer that is not
        // distributed reads the whole product on every rank.
        use crate::executor::TransferConfig;
        use crate::worker::instance::routing::media_consumer_slots;

        let model_components =
            BTreeMap::from([("video_decoder".to_owned(), distributed(vec![0, 1, 2, 3], 1))]);
        let host_components =
            BTreeMap::from([("video_codec".to_owned(), distributed(vec![0, 1], 2))]);
        let peers = BTreeMap::from([
            ("host".to_owned(), host_components),
            ("model".to_owned(), model_components.clone()),
        ]);
        let mut transfer = TransferConfig::default();
        transfer.worker_ranks.insert("host".to_owned(), 2);
        transfer.worker_ranks.insert("model".to_owned(), 4);
        let slots = media_consumer_slots(
            &[MediaCall::VideoEncoding],
            &routing(),
            &BTreeMap::new(),
            "model",
            &model_components,
            &peers,
            &transfer,
            &[0, 1, 2, 3],
            2,
            model_components.get("video_decoder"),
        );
        // Position 2 of the round, two units per host rank: host rank 1.
        assert_eq!(slots, [transfer.acknowledgment_slot("host", 1)]);

        let routing = BTreeMap::from([
            (MediaCall::AudioDecoding, "audio_decoder".to_owned()),
            (MediaCall::AudioEncoding, "muxer".to_owned()),
        ]);
        let model_components =
            BTreeMap::from([("audio_decoder".to_owned(), distributed(vec![0, 1, 2, 3], 1))]);
        let host_components = BTreeMap::from([(
            "muxer".to_owned(),
            ComponentConfig {
                ranks: vec![0],
                parallel_config: Default::default(),
                distribution: None,
                units_per_rank: 1,
                weights: Default::default(),
            },
        )]);
        let peers = BTreeMap::from([
            ("host".to_owned(), host_components),
            ("model".to_owned(), model_components.clone()),
        ]);
        let slots = media_consumer_slots(
            &[MediaCall::AudioEncoding],
            &routing,
            &BTreeMap::new(),
            "model",
            &model_components,
            &peers,
            &transfer,
            &[0, 1, 2, 3],
            3,
            model_components.get("audio_decoder"),
        );
        assert_eq!(slots, [transfer.acknowledgment_slot("host", 0)]);
    }

    #[test]
    fn a_shared_producer_names_only_the_replica_its_request_was_routed_to() {
        // A text encoder on its own worker feeds latent preparation on two
        // denoiser replicas. A request routed to flow-1 is read there alone,
        // so its product names flow-1's reading rank and not flow-0's. A
        // component with no route yet (a call placed only when it runs) is
        // read by whichever replica serves it, so each one is named.
        use crate::executor::TransferConfig;
        use crate::worker::instance::routing::media_consumer_slots;

        let text = BTreeMap::from([(
            "text_encoder".to_owned(),
            ComponentConfig::parallel(vec![0], Default::default()),
        )]);
        let denoiser = || {
            BTreeMap::from([(
                "denoiser".to_owned(),
                ComponentConfig::parallel(vec![0], Default::default()),
            )])
        };
        let peers = BTreeMap::from([
            ("flow-0".to_owned(), denoiser()),
            ("flow-1".to_owned(), denoiser()),
            ("text".to_owned(), text.clone()),
        ]);
        let routing = BTreeMap::from([(MediaCall::LatentPreparation, "denoiser".to_owned())]);
        let mut transfer = TransferConfig::default();
        for worker in ["text", "flow-0", "flow-1"] {
            transfer.worker_ranks.insert(worker.to_owned(), 1);
        }
        let slots = |routed: &BTreeMap<String, WorkerId>| {
            media_consumer_slots(
                &[MediaCall::LatentPreparation],
                &routing,
                routed,
                "text",
                &text,
                &peers,
                &transfer,
                &[0],
                0,
                text.get("text_encoder"),
            )
        };

        let routed = BTreeMap::from([("denoiser".to_owned(), WorkerId("flow-1".to_owned()))]);
        assert_eq!(slots(&routed), [transfer.acknowledgment_slot("flow-1", 0)]);

        assert_eq!(
            slots(&BTreeMap::new()),
            [
                transfer.acknowledgment_slot("flow-0", 0),
                transfer.acknowledgment_slot("flow-1", 0),
            ]
        );
    }

    #[test]
    fn a_decoded_unit_names_only_the_encoder_on_its_route() {
        // Two hosts each hold a video codec worker. A request decoded on
        // host 0's model worker is routed to host 0's encoder, so the unit
        // names that encoder's reading rank; host 1's encoder rank at the
        // same position never reads it.
        use crate::executor::TransferConfig;
        use crate::worker::instance::routing::media_consumer_slots;

        let model = BTreeMap::from([("video_decoder".to_owned(), distributed(vec![0, 1], 1))]);
        let encoder = || BTreeMap::from([("video_codec".to_owned(), distributed(vec![0, 1], 1))]);
        let peers = BTreeMap::from([
            ("encoder-0".to_owned(), encoder()),
            ("encoder-1".to_owned(), encoder()),
            ("model-0".to_owned(), model.clone()),
        ]);
        let mut transfer = TransferConfig::default();
        for worker in ["encoder-0", "encoder-1", "model-0"] {
            transfer.worker_ranks.insert(worker.to_owned(), 2);
        }
        let routed = BTreeMap::from([("video_codec".to_owned(), WorkerId("encoder-0".to_owned()))]);

        let slots = media_consumer_slots(
            &[MediaCall::VideoEncoding],
            &routing(),
            &routed,
            "model-0",
            &model,
            &peers,
            &transfer,
            &[0, 1],
            1,
            model.get("video_decoder"),
        );

        assert_eq!(slots, [transfer.acknowledgment_slot("encoder-0", 1)]);
    }

    fn pairing(
        placements: &[(&WorkerId, &[WorkerRank], &BTreeMap<String, ComponentConfig>)],
    ) -> anyhow::Result<BTreeMap<WorkerId, std::collections::BTreeSet<WorkerId>>> {
        video_unit_codecs(&routing(), placements)
    }

    #[test]
    fn units_encoded_on_their_own_host_are_admitted() {
        let model = WorkerId("model".to_owned());
        let host = WorkerId("host".to_owned());
        let model_ranks = vec![
            rank("a", "cuda:0"),
            rank("a", "cuda:1"),
            rank("b", "cuda:0"),
            rank("b", "cuda:1"),
        ];
        let host_ranks = vec![
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
            (&host, &host_ranks, &host_components),
        ])
        .expect("each unit is encoded on its own host");
        assert_eq!(pairings[&model], [host].into());
    }

    #[test]
    fn a_decoder_dealing_several_units_per_rank_keeps_them_on_its_host() {
        // Two units per decoding rank: positions 0 to 3 come from host a's
        // decoders and go to host a's encoders, positions 4 to 7 likewise on
        // host b.
        let model = WorkerId("model".to_owned());
        let host = WorkerId("host".to_owned());
        let model_ranks = vec![
            rank("a", "cuda:0"),
            rank("a", "cuda:1"),
            rank("b", "cuda:0"),
            rank("b", "cuda:1"),
        ];
        let host_ranks = [["a"; 4], ["b"; 4]]
            .concat()
            .into_iter()
            .map(|node| rank(node, "cpu"))
            .collect::<Vec<_>>();
        let model_components =
            BTreeMap::from([("video_decoder".to_owned(), distributed(vec![0, 1, 2, 3], 2))]);
        let host_components =
            BTreeMap::from([("video_codec".to_owned(), distributed((0..8).collect(), 1))]);
        pairing(&[
            (&model, &model_ranks, &model_components),
            (&host, &host_ranks, &host_components),
        ])
        .expect("each host's units are encoded on it");

        // Host a holding three encoders sends its fourth unit to host b.
        let host_ranks = [["a"; 3].as_slice(), ["b"; 5].as_slice()]
            .concat()
            .into_iter()
            .map(|node| rank(node, "cpu"))
            .collect::<Vec<_>>();
        let error = pairing(&[
            (&model, &model_ranks, &model_components),
            (&host, &host_ranks, &host_components),
        ])
        .expect_err("the fourth unit leaves host a");
        assert!(error.to_string().contains("media unit 3"), "{error}");
    }

    #[test]
    fn every_shipped_deployment_encodes_its_units_on_their_host() {
        // Every video deployment file shipped in `configs/minimax_h3/` must
        // pass the placement checks the engine applies at startup, including
        // the pairing of every decoding worker with an encoder on its host.
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
                .and_then(|deployment| deployment.workers("localhost"))
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
            (WorkerId("flow-a".to_owned()), vec![rank("a", "cuda:0")]),
            (WorkerId("flow-b".to_owned()), vec![rank("b", "cuda:0")]),
        ];
        let encoders = [
            (WorkerId("encoder-a".to_owned()), vec![rank("a", "cpu")]),
            (WorkerId("encoder-b".to_owned()), vec![rank("b", "cpu")]),
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
        assert!(message.contains("worker flow-b"), "{message}");
        assert!(
            message.contains("host b") && message.contains("host a"),
            "{message}"
        );
    }
}
