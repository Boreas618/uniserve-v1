//! The binary's start on the meta device against a Storage Server in the test's process: one
//! engine whose scheduler grew its KV region into the segments its rank created, and whose close
//! ends the session; and one engine of a media launch. Then the generation limits an engine
//! reports, against what its groups state.
//!
//! The text model is the Qwen3-MoE fixture of the `qwen3` crate and the media model a released
//! FastH3 checkpoint (`crate::process::ranks::tests::FAST_H3`, which the repository does not
//! hold), each loaded from its shard headers alone, and the policy the packaged `fcfs`; no GPU
//! and no Build Server is touched.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::batch::Capacity;
use crate::batch::{Phase, Shape, Variant};
use crate::executor::peers::{LOST_EXIT, Liveness};
use proto::request::GenerationLimits;
use storage::runtime::GroupId;

use super::{FIRST_GENERATION, lowered, start};
use crate::memory::client::StorageClient;
use crate::process::epoch::kv_regions;
use crate::process::ranks::layout;
use crate::process::ranks::tests::{FAST_H3, loaded, video_launch};
use crate::support::{DEPLOYMENT, StorageServer, launch};

#[test]
fn a_meta_start_builds_one_engine_whose_close_ends_the_session() {
    let server = StorageServer::start();
    let mut launch = launch();
    launch.engine_config.device.device = "meta".to_owned();
    let registry = nn::family::Registry::new([models::qwen3::family::FAMILY]).unwrap();
    let storage = crate::memory::client::StorageLaunch {
        socket: server.socket.to_string_lossy().into_owned(),
        deployment: launch.engine_config.deployment.clone(),
        claim: [0; 16],
        tcp: None,
        peers: Vec::new(),
        epoch: launch.epoch,
        standby: false,
        standby_storage: Default::default(),
    };
    // An engine of one main process has no peer link to lose.
    let liveness = Liveness::new(Arc::new(|| {}), LOST_EXIT);
    // The packaged policy the launch names, with the arguments the launch pins on it.
    let parameters = crate::modules::load::parameters(&launch.modules.policy).unwrap();
    let policy = (crate::scheduler::packaged("fcfs"), parameters);

    // One scheduler over the one group of the one rank, which no routing policy chooses among,
    // assembling batches for the start's generation.
    let engine = start(&launch, &registry, &storage, None, policy, liveness).unwrap();
    assert_eq!(engine.served.load(Ordering::Acquire), FIRST_GENERATION);
    assert_eq!(engine.schedulers.len(), 1);
    assert!(engine.route.is_none());
    assert_eq!(engine.workers.len(), 1);
    assert_eq!(engine.capacities.len(), 1);
    // The launch's model states no feature; the engine serves what its group's phases do.
    assert!(
        launch
            .engine_config
            .model
            .generation_limits
            .features
            .is_empty()
    );
    assert_eq!(
        engine.limits.features,
        proto::request::GenerationFeatures::UNDERSTANDING
    );
    let (open, opened) = {
        let head = engine.head.as_ref().unwrap();
        let head = service::lock(head);
        let (open, opened) = head.session().unwrap();
        (open.clone(), opened.clone())
    };
    let regions = kv_regions(&open);
    assert_eq!(regions.len(), 1);
    let kv = regions[0];

    // A successor epoch of the layout finds the region holding every unit past the padding unit,
    // over the segments the start's rank created and its head granted.
    let mut successor = launch.clone();
    successor.epoch = 2;
    let next_open = layout(&successor, &loaded(&successor)).unwrap();
    let mut next = StorageClient::connect_head(&server.socket, DEPLOYMENT, 2).unwrap();
    let _pairs = next.open(next_open).unwrap();
    let (_, next_opened) = next.session().unwrap();
    let limit = open.regions[kv].limit;
    assert_eq!(next_opened.regions[kv], (opened.regions[kv].0, limit - 1));
    let units: u32 = next_opened.segments[kv]
        .iter()
        .map(|(_, _, units)| units)
        .sum();
    assert_eq!(units, limit);
    assert!(
        next_opened.segments[kv]
            .iter()
            .all(|(segment, _, _)| segment.epoch == launch.epoch)
    );
    next.close().unwrap();

    // The close ends the epoch's session: a head of the epoch opens it again.
    let mut outputs = Vec::new();
    engine
        .close(|scheduler, output| outputs.push((scheduler, output)))
        .unwrap();
    let mut again = StorageClient::connect_head(&server.socket, DEPLOYMENT, launch.epoch).unwrap();
    let _pairs = again.open(open).unwrap();
    again.close().unwrap();
    assert!(
        outputs.iter().all(|(scheduler, output)| *scheduler == 0
            && !matches!(output, crate::scheduler::Output::Failed(_))),
        "{outputs:?}"
    );
}

#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn a_meta_start_of_a_media_launch_builds_one_engine_whose_close_ends_the_session() {
    let server = StorageServer::start();
    let mut launch = video_launch();
    launch.engine_config.device.device = "meta".to_owned();
    let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
    let storage = crate::memory::client::StorageLaunch {
        socket: server.socket.to_string_lossy().into_owned(),
        deployment: launch.engine_config.deployment.clone(),
        claim: [0; 16],
        tcp: None,
        peers: Vec::new(),
        epoch: launch.epoch,
        standby: false,
        standby_storage: Default::default(),
    };
    let liveness = Liveness::new(Arc::new(|| {}), LOST_EXIT);
    let parameters = crate::modules::load::parameters(&launch.modules.policy).unwrap();
    let policy = (crate::scheduler::packaged("fcfs"), parameters);

    // One scheduler over the one media group of the one rank, whose worker is warm, with no
    // KV region to grow: every video phase on the group, with its steps and product sizes.
    assert!(
        std::path::Path::new(FAST_H3).is_dir(),
        "{FAST_H3} is missing"
    );
    let engine = start(&launch, &registry, &storage, None, policy, liveness).unwrap();
    assert_eq!(engine.schedulers.len(), 1);
    assert_eq!(engine.workers.len(), 1);
    let (id, capacity) = &engine.capacities[0];
    assert_eq!(id.0, 0);
    assert!(
        capacity.serves(crate::batch::Phase::Denoise) && capacity.serves(crate::batch::Phase::Mux)
    );
    assert!(capacity.denoise_steps > 0 && !capacity.media_bytes.is_empty());
    {
        let head = service::lock(engine.head.as_ref().unwrap());
        let (open, _) = head.session().unwrap();
        assert!(kv_regions(open).is_empty());
    }

    // The close ends the epoch's session.
    let mut outputs = Vec::new();
    engine
        .close(|scheduler, output| outputs.push((scheduler, output)))
        .unwrap();
    assert!(
        outputs
            .iter()
            .all(|(_, output)| !matches!(output, crate::scheduler::Output::Failed(_))),
        "{outputs:?}"
    );
}

/// What a worker group serving `phases` states, with a prefill bucket of `tokens` tokens and
/// encoder outputs of `feature_bytes` a token.
fn capacity(phases: &[Phase], tokens: u32, feature_bytes: u32) -> Capacity {
    let mut served = 0;
    let mut rows = [0; Phase::COUNT];
    for phase in phases {
        served |= 1 << phase.index();
        rows[phase.index()] = 4;
    }
    Capacity {
        phases: served,
        slots: 4,
        kv_cache: None,
        cache_groups: Vec::new(),
        seeds: Vec::new(),
        buckets: vec![
            Shape::prefill(4, tokens, Variant::PrefillOutput),
            Shape::decode(4, Variant::DecodeSampled),
        ],
        rows,
        max_model_len: 4096,
        denoise_steps: 0,
        bos_token_id: 1,
        eos_token_ids: vec![2],
        feature_bytes,
        encoder_identity: String::new(),
        latent_seed: 0,
        media_bytes: Vec::new(),
        call_window: 2,
    }
}

/// The limits a family that takes images configures: no bound of its own on their encodings.
fn unbounded() -> GenerationLimits {
    GenerationLimits {
        max_vae_grid_tokens: u32::MAX,
        max_vit_grid_tokens: u32::MAX,
        max_latent_feature_bytes: u64::MAX,
        max_vision_feature_bytes: u64::MAX,
        encoder_cache_entries: u32::MAX,
        max_logprobs: 20,
        ..GenerationLimits::default()
    }
}

#[test]
fn the_limits_an_engine_reports_are_the_lower_of_the_launch_s_and_its_groups() {
    // Groups that encode no image hold no encoding: whatever the launch states, a request with
    // an image is refused for want of a bound.
    let text = vec![(
        GroupId(0),
        capacity(&[Phase::Prefill, Phase::Decode], 2048, 0),
    )];
    let ceilings = crate::scheduler::policy::ceilings(text.iter().map(|(_, capacity)| capacity));
    let limits = lowered(unbounded(), &text, &ceilings);
    assert_eq!(
        (limits.max_vit_grid_tokens, limits.max_vision_feature_bytes),
        (0, 0)
    );
    assert_eq!(
        (limits.max_vae_grid_tokens, limits.max_latent_feature_bytes),
        (0, 0)
    );
    assert_eq!(limits.encoder_cache_entries, 0);
    assert_eq!(limits.max_logprobs, 20);

    // A group that encodes images holds a grid one prefill writes whole, and its features at
    // the bytes of a row of its encoder output.
    let vision = vec![(
        GroupId(0),
        capacity(
            &[Phase::Prefill, Phase::Decode, Phase::VisionEncode],
            2048,
            4096,
        ),
    )];
    let ceilings = crate::scheduler::policy::ceilings(vision.iter().map(|(_, capacity)| capacity));
    let limits = lowered(unbounded(), &vision, &ceilings);
    assert_eq!(limits.max_vit_grid_tokens, 2048);
    assert_eq!(limits.max_vision_feature_bytes, 2048 * 4096);
    assert_eq!(limits.max_latent_feature_bytes, 0);
    assert_eq!(limits.encoder_cache_entries, u32::MAX);

    // A launch that states less than the groups hold keeps its own.
    let stated = GenerationLimits {
        max_vit_grid_tokens: 1024,
        max_vision_feature_bytes: 1 << 20,
        ..unbounded()
    };
    let limits = lowered(stated, &vision, &ceilings);
    assert_eq!(limits.max_vit_grid_tokens, 1024);
    assert_eq!(limits.max_vision_feature_bytes, 1 << 20);
}
