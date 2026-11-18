//! The binary's start on the meta device against a Storage Server in the test's process: one
//! engine whose scheduler grew its KV region into the segments its rank created once admission
//! opened, and whose close ends the session; the regions a target of the same layout that states
//! smaller host pools shares with the serving engine, through its start, its promotion, a commit,
//! a rollback and an abandoned update; one engine of a media launch; and one engine of a model
//! with a vision tower, whose rank encodes images beside its prefills. Then the generation limits
//! an engine reports, against what its groups state.
//!
//! The text model is the Qwen3-MoE fixture of the `qwen3` crate, the media model a released
//! FastH3 checkpoint (`crate::process::ranks::tests::fast_h3`) and the model with a vision tower
//! a released Qwen3.6 checkpoint ([`qwen_vision`]), neither of which the repository holds, each
//! loaded from its shard headers alone, and the policy the packaged `arrival`; no GPU and no Build
//! Server is touched.

use std::collections::{BTreeMap, BTreeSet};
use std::num::NonZeroU64;
use std::sync::Arc;
use std::time::Instant;

use crate::batch::Capacity;
use crate::batch::{Phase, Shape, Variant};
use crate::executor::peers::Liveness;
use crate::memory::buffers::UNIT_BYTES;
use crate::memory::encoders;
use proto::config::EngineArgs;
use proto::request::GenerationLimits;
use storage::runtime::{GroupId, Open, Tier};

use super::{Engine, FIRST_GENERATION, lowered, start};
use crate::memory::client::{StorageClient, StorageLaunch};
use crate::process::epoch::{drain, kv_regions, promote};
use crate::process::ranks::layout;
use crate::process::ranks::tests::{loaded, video_launch};
use testkit::LONG;
use testkit::serving::{DEPLOYMENT, StorageServer, launch};

/// The epoch and launch generation of the sessions that read what the buffer pool holds, beside
/// the engines of a case.
const READER: u64 = 9;

/// The host pools of the serving epoch and of a target of its layout that halves them
/// (`kv_cache_config.host_gib`). A meta start numbers its KV planes at their fewest units
/// whatever `kv_cache_config.max_tokens` states, so the host pools are what such a target sizes
/// apart from the serving epoch there; the regions of either size are resized alike
/// (`crate::process::epoch::grow_regions`).
const SERVING_HOST_GIB: u64 = 2;
const TARGET_HOST_GIB: u64 = 1;

/// The Storage Server launch of `launch`'s engine, whose Storage Server is `server`.
fn storage_launch(server: &StorageServer, launch: &EngineArgs) -> StorageLaunch {
    StorageLaunch {
        socket: server.socket.to_string_lossy().into_owned(),
        deployment: launch.deployment.clone(),
        claim: [0; 16],
        tcp: None,
        peers: Vec::new(),
        epoch: launch.epoch,
        standby: false,
        standby_storage: Default::default(),
    }
}

/// `launch`'s engine, started on the meta device at `server` with the packaged policy the launch
/// names.
fn start_on_meta(server: &StorageServer, launch: &EngineArgs) -> Engine {
    let registry = nn::family::Registry::new([models::qwen3::family::FAMILY]).unwrap();
    let parameters = crate::modules::load::parameters(&launch.modules.policy).unwrap();
    let policy = (
        crate::scheduler::policy::tests::packaged("arrival"),
        parameters,
    );
    // An engine of one main process has no peer link to lose.
    let liveness = Liveness::new(Arc::new(|| {}));
    start(
        launch,
        &registry,
        &storage_launch(server, launch),
        None,
        policy,
        liveness,
    )
    .unwrap()
}

/// Opens admission on `engine` as its process does when the engine is promoted: once its start
/// ends unparked, and at each `Promote` that finds it parked, a rollback's included
/// (`MainProcess::open_admission`).
fn open_admission(engine: &Engine) {
    promote(&engine.inboxes(), engine.head.as_deref(), &engine.binding).unwrap();
}

/// Drains `engine` beside an epoch of the same layout, which keeps the regions the two share
/// (`Drain` without `shrink`), and waits until every scheduler holds no request.
fn drain_beside(engine: &Engine) {
    for reply in drain(&engine.inboxes(), Some(Instant::now() + LONG), false).unwrap() {
        reply.recv_timeout(LONG).unwrap();
    }
}

/// Ends `engine`, as its process's exit does, its session included.
fn close(engine: Engine) {
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

/// The regions `engine`'s session names, by name, each with the units the session states the
/// region serves: what the engine sizes it to once it is promoted.
fn stated(engine: &Engine) -> BTreeMap<String, u32> {
    let head = service::lock(engine.head.as_ref().unwrap());
    let (open, _) = head.session().unwrap();
    open.regions
        .iter()
        .map(|region| (region.name.clone(), region.units))
        .collect()
}

/// The units in each region of `names` that the reader's layout `open` names: what a session of
/// that layout, of epoch [`READER`], opened now at `server` is answered.
fn held(server: &StorageServer, open: &Open, names: &BTreeSet<String>) -> BTreeMap<String, u32> {
    let mut reader =
        StorageClient::connect_head(&server.socket, DEPLOYMENT, READER, READER).unwrap();
    let _pairs = reader.open(open.clone()).unwrap();
    let (_, opened) = reader.session().unwrap();
    let held = open
        .regions
        .iter()
        .zip(&opened.regions)
        .filter(|(region, _)| names.contains(&region.name))
        .map(|(region, &(_, units))| (region.name.clone(), units))
        .collect();
    reader.close().unwrap();
    held
}

/// The parts of an update of the host pools' size (`kv_cache_config.host_gib`) over the
/// fixture's layout, at `server`: the serving engine of epoch 1, of [`SERVING_HOST_GIB`], started
/// with its admission open; the launch of the target of epoch 2, of the same layout, which states
/// [`TARGET_HOST_GIB`] and parks (it succeeds epoch 1); and the layout of the reader's sessions,
/// the serving epoch's.
fn update(server: &StorageServer) -> (Engine, EngineArgs, Open) {
    let mut serving = launch();
    serving.engine_config.device.device = proto::config::DeviceKind::Meta;
    serving.engine_config.kv_cache.host_gib = NonZeroU64::new(SERVING_HOST_GIB);

    let mut target = serving.clone();
    target.epoch = 2;
    target.engine = 2;
    target.predecessor = Some(proto::request::wire::Predecessor {
        epoch: serving.epoch,
    });
    target.engine_config.kv_cache.host_gib = NonZeroU64::new(TARGET_HOST_GIB);

    let mut reader = serving.clone();
    reader.epoch = READER;
    reader.engine = READER;
    let reader = layout(&reader, &loaded(&reader)).unwrap();

    let engine = start_on_meta(server, &serving);
    open_admission(&engine);
    (engine, target, reader)
}

/// The names of the regions both engines' sessions name, which the buffer pool holds once for
/// both: the region of the KV planes and the host pools among them, the target stating half the
/// serving engine's units of each host pool.
fn shared(serving: &Engine, target: &Engine) -> BTreeSet<String> {
    let (theirs, ours) = (stated(serving), stated(target));
    let shared: BTreeSet<String> = theirs
        .keys()
        .filter(|name| ours.contains_key(*name))
        .cloned()
        .collect();

    let head = service::lock(serving.head.as_ref().unwrap());
    let (open, _) = head.session().unwrap();
    let kv = &open.regions[kv_regions(open)[0]].name;
    assert!(shared.contains(kv), "{shared:?}");
    let pools: Vec<&String> = open
        .regions
        .iter()
        .filter(|region| region.tier == Tier::Host)
        .map(|region| &region.name)
        .collect();
    assert!(!pools.is_empty(), "the launch keeps a host tier");
    for pool in pools {
        assert!(shared.contains(pool), "{shared:?}");
        assert_eq!(ours[pool], theirs[pool] / 2, "{pool}");
    }
    shared
}

/// The units `engine`'s session states of each region of `names`.
fn stated_of(engine: &Engine, names: &BTreeSet<String>) -> BTreeMap<String, u32> {
    stated(engine)
        .into_iter()
        .filter(|(name, _)| names.contains(name))
        .collect()
}

/// The cache's size is no layout: a target of the serving epoch's layout that states half its
/// host pools names the regions the serving engine's session names, and its start parks it with
/// every region the two share at the serving epoch's size, so the serving epoch keeps its
/// capacity while the target prepares. Its promotion resizes them to its own sizes, and a commit,
/// which drains the serving engine beside it and ends it, keeps them.
#[test]
fn a_target_of_the_same_layout_resizes_the_shared_regions_once_it_is_promoted() {
    let server = StorageServer::start();
    let (serving, target, reader) = update(&server);
    let target = start_on_meta(&server, &target);
    let shared = shared(&serving, &target);
    let before = stated_of(&serving, &shared);
    assert_eq!(held(&server, &reader, &shared), before);

    // The switch: the target opens admission, sized as it states.
    open_admission(&target);
    let ours = stated_of(&target, &shared);
    assert_eq!(held(&server, &reader, &shared), ours);

    // The commit: the serving engine drains beside the target and ends.
    drain_beside(&serving);
    close(serving);
    assert_eq!(held(&server, &reader, &shared), ours);
    close(target);
}

/// An update abandoned after its target started (refused, failed or cancelled before its switch)
/// changes nothing: the daemon stops the target, whose engine ends unpromoted, and every region
/// it shared with the serving engine holds what it held before the target started.
#[test]
fn a_target_abandoned_before_its_promotion_leaves_every_shared_region_at_the_serving_size() {
    let server = StorageServer::start();
    let (serving, target, reader) = update(&server);
    let everything: BTreeSet<String> = stated(&serving).into_keys().collect();
    let before = held(&server, &reader, &everything);

    let target = start_on_meta(&server, &target);
    let shared = shared(&serving, &target);
    close(target);
    let before: BTreeMap<String, u32> = before
        .into_iter()
        .filter(|(name, _)| shared.contains(name))
        .collect();
    assert_eq!(held(&server, &reader, &shared), before);
    close(serving);
}

/// A rollback runs the switch back: the serving engine keeps its sizes while the target starts,
/// the target's promotion resizes the regions the two share to its own, and the serving engine,
/// drained beside it, is promoted again and resizes them to the serving epoch's; the target then
/// drains and ends, and they stay so.
#[test]
fn a_rollback_restores_the_serving_size_of_every_shared_region() {
    let server = StorageServer::start();
    let (serving, target, reader) = update(&server);
    let target = start_on_meta(&server, &target);
    let shared = shared(&serving, &target);
    let before = stated_of(&serving, &shared);
    assert_eq!(held(&server, &reader, &shared), before);

    open_admission(&target);
    assert_eq!(held(&server, &reader, &shared), stated_of(&target, &shared));
    drain_beside(&serving);

    // The rollback: the drained serving engine is promoted again, and the target drains and ends.
    open_admission(&serving);
    assert_eq!(held(&server, &reader, &shared), before);
    drain_beside(&target);
    close(target);
    assert_eq!(held(&server, &reader, &shared), before);
    close(serving);
}

#[test]
fn a_meta_start_builds_one_engine_whose_close_ends_the_session() {
    let server = StorageServer::start();
    let mut launch = launch();
    launch.engine_config.device.device = proto::config::DeviceKind::Meta;
    let registry = nn::family::Registry::new([models::qwen3::family::FAMILY]).unwrap();
    let storage = crate::memory::client::StorageLaunch {
        socket: server.socket.to_string_lossy().into_owned(),
        deployment: launch.deployment.clone(),
        claim: [0; 16],
        tcp: None,
        peers: Vec::new(),
        epoch: launch.epoch,
        standby: false,
        standby_storage: Default::default(),
    };
    // An engine of one main process has no peer link to lose.
    let liveness = Liveness::new(Arc::new(|| {}));
    // The packaged policy the launch names, with the arguments the launch pins on it.
    let parameters = crate::modules::load::parameters(&launch.modules.policy).unwrap();
    let policy = (
        crate::scheduler::policy::tests::packaged("arrival"),
        parameters,
    );

    // One scheduler over the one group of the one rank, assembling batches for the start's
    // generation.
    let engine = start(&launch, &registry, &storage, None, policy, liveness).unwrap();
    assert_eq!(engine.served.kernels(), FIRST_GENERATION);
    assert_eq!(engine.schedulers.len(), 1);
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

    // Once admission opened, a successor epoch of the layout finds the region holding every unit
    // past the padding unit, over the segments the start's rank created and its head granted.
    open_admission(&engine);
    let mut successor = launch.clone();
    successor.epoch = 2;
    let next_open = layout(&successor, &loaded(&successor)).unwrap();
    let mut next = StorageClient::connect_head(&server.socket, DEPLOYMENT, 2, 2).unwrap();
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
    let mut again =
        StorageClient::connect_head(&server.socket, DEPLOYMENT, launch.epoch, launch.engine)
            .unwrap();
    let _pairs = again.open(open).unwrap();
    again.close().unwrap();
    assert!(
        outputs.iter().all(|(scheduler, output)| *scheduler == 0
            && !matches!(output, crate::scheduler::Output::Failed(_))),
        "{outputs:?}"
    );
}

/// The standby of the serving engine is another engine of the serving epoch (its own launch
/// generation, started parked beside it): its start opens a session of its own beside the serving
/// engine's, names the epoch's KV region and maps the segments the serving engine's rank created,
/// creating none, and keeps slot rows of its own. Each engine's close ends its own session.
#[test]
fn a_standby_of_the_serving_epoch_opens_a_session_of_its_own_and_maps_its_region() {
    let server = StorageServer::start();
    let mut launch = launch();
    launch.engine_config.device.device = proto::config::DeviceKind::Meta;
    let registry = nn::family::Registry::new([models::qwen3::family::FAMILY]).unwrap();
    let storage = crate::memory::client::StorageLaunch {
        socket: server.socket.to_string_lossy().into_owned(),
        deployment: launch.deployment.clone(),
        claim: [0; 16],
        tcp: None,
        peers: Vec::new(),
        epoch: launch.epoch,
        standby: false,
        standby_storage: Default::default(),
    };
    let parameters = crate::modules::load::parameters(&launch.modules.policy).unwrap();
    let policy = || {
        (
            crate::scheduler::policy::tests::packaged("arrival"),
            parameters.clone(),
        )
    };
    let session = |engine: &super::Engine| {
        let head = service::lock(engine.head.as_ref().unwrap());
        let (open, opened) = head.session().unwrap();
        (open.clone(), opened.clone())
    };

    let liveness = Liveness::new(Arc::new(|| {}));
    let serving = start(&launch, &registry, &storage, None, policy(), liveness).unwrap();
    let (open, opened) = session(&serving);
    let kv = kv_regions(&open)[0];

    let mut parked = launch.clone();
    parked.engine = launch.engine + 1;
    parked.predecessor = Some(proto::request::wire::Predecessor {
        epoch: launch.epoch,
    });
    let liveness = Liveness::new(Arc::new(|| {}));
    let standby = start(&parked, &registry, &storage, None, policy(), liveness).unwrap();
    let (standby_open, standby_opened) = session(&standby);
    assert_eq!(standby_open.regions[kv].name, open.regions[kv].name);
    assert_eq!(standby_opened.regions[kv].0, opened.regions[kv].0);
    assert!(!opened.segments[kv].is_empty());
    assert_eq!(standby_opened.segments[kv], opened.segments[kv]);
    let slots = |open: &storage::runtime::Open| {
        open.regions[usize::from(open.groups[0].slots)].name.clone()
    };
    assert_ne!(slots(&standby_open), slots(&open));

    let mut outputs = Vec::new();
    standby
        .close(|scheduler, output| outputs.push((scheduler, output)))
        .unwrap();
    serving
        .close(|scheduler, output| outputs.push((scheduler, output)))
        .unwrap();
    assert!(
        outputs
            .iter()
            .all(|(_, output)| !matches!(output, crate::scheduler::Output::Failed(_))),
        "{outputs:?}"
    );
}

#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn a_meta_start_of_a_media_launch_builds_one_engine_whose_close_ends_the_session() {
    let server = StorageServer::start();
    let mut launch = video_launch();
    launch.engine_config.device.device = proto::config::DeviceKind::Meta;
    let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
    let storage = crate::memory::client::StorageLaunch {
        socket: server.socket.to_string_lossy().into_owned(),
        deployment: launch.deployment.clone(),
        claim: [0; 16],
        tcp: None,
        peers: Vec::new(),
        epoch: launch.epoch,
        standby: false,
        standby_storage: Default::default(),
    };
    let liveness = Liveness::new(Arc::new(|| {}));
    let parameters = crate::modules::load::parameters(&launch.modules.policy).unwrap();
    let policy = (
        crate::scheduler::policy::tests::packaged("arrival"),
        parameters,
    );

    // One scheduler over the one media group of the one rank, whose worker is warm, with no
    // KV region to grow: every video phase on the group, with its steps and product sizes.
    let engine = start(&launch, &registry, &storage, None, policy, liveness).unwrap();
    assert_eq!(engine.schedulers.len(), 1);
    assert_eq!(engine.workers.len(), 1);
    let (id, capacity) = &engine.capacities[0];
    assert_eq!(id.0, 0);
    assert!(
        capacity.serves(crate::batch::Phase::Denoise) && capacity.serves(crate::batch::Phase::Mux)
    );
    assert!(capacity.denoise_steps > 0 && !capacity.media_bytes.is_empty());
    // The daemon binds the video processor with the denoiser the engine states.
    assert!(!engine.video.is_empty());
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

/// The released Qwen3.6 checkpoint (`Qwen3_5MoeForConditionalGeneration`) whose headers the
/// start of a model with a vision tower loads: a tower whole on every rank, which merges
/// `2 × 2` patches into a row of `2048` BF16 features over three rotary axes.
fn qwen_vision() -> std::path::PathBuf {
    testkit::assets::QWEN3_6_35B_A3B.snapshot()
}

#[test]
#[ignore = "reads the headers of a released Qwen3.6 checkpoint, which the repository does not hold"]
fn a_meta_start_of_a_model_with_a_vision_tower_encodes_images_beside_its_prefills() {
    let server = StorageServer::start();
    // The fixture's launch, one rank of one group with prefill buckets of up to 32 tokens, over
    // the checkpoint, with the limits of a family that takes images.
    let mut launch = launch();
    launch.engine_config.device.device = proto::config::DeviceKind::Meta;
    launch.engine_config.model.model = qwen_vision().to_string_lossy().into_owned();
    launch.engine_config.model.generation_limits = unbounded();
    let registry = nn::family::Registry::new([models::qwen3::family::HYBRID_FAMILY]).unwrap();
    let storage = crate::memory::client::StorageLaunch {
        socket: server.socket.to_string_lossy().into_owned(),
        deployment: launch.deployment.clone(),
        claim: [0; 16],
        tcp: None,
        peers: Vec::new(),
        epoch: launch.epoch,
        standby: false,
        standby_storage: Default::default(),
    };
    let liveness = Liveness::new(Arc::new(|| {}));
    let parameters = crate::modules::load::parameters(&launch.modules.policy).unwrap();
    let policy = (
        crate::scheduler::policy::tests::packaged("arrival"),
        parameters,
    );
    let engine = start(&launch, &registry, &storage, None, policy, liveness).unwrap();

    // The rank's one worker serves the prefills and the image encodings: its causal runner and
    // its vision runner. A row of an encoder output is a row of features and its coordinates.
    let feature_bytes: u32 = 2048 * 2 + 3 * 4;
    assert_eq!(engine.workers.len(), 1);
    for phase in [Phase::Prefill, Phase::Decode, Phase::VisionEncode] {
        assert!(engine.workers[0].serves(phase), "{phase:?}");
    }
    let (_, capacity) = &engine.capacities[0];
    assert!(capacity.serves(Phase::VisionEncode));
    assert_eq!(capacity.feature_bytes, feature_bytes);
    assert_ne!(capacity.encoder_identity, "");

    // An admitted image's block is written whole by one prefill, so it holds at most the 32
    // rows of the widest prefill bucket: the vision runner captures the grids of 32 rows and of
    // each doubling below, of `2 × 2` patches a row, and none of the tower's larger grids.
    let encodes: Vec<u32> = capacity
        .buckets
        .iter()
        .filter(|shape| shape.variant == Variant::ConditionEncode)
        .map(|shape| shape.tokens)
        .collect();
    assert_eq!(encodes, [128, 64, 32, 16, 8, 4]);

    // The limits the engine reports: an image's grid within the widest prefill bucket, which
    // writes it whole, and its features at that many rows.
    assert_eq!(engine.limits.max_vit_grid_tokens, 32);
    assert_eq!(
        engine.limits.max_vision_feature_bytes,
        32 * u64::from(feature_bytes)
    );
    assert_ne!(engine.limits.encoder_cache_entries, 0);

    // The session: after the cache groups, the class of encoder outputs, sized per request, its
    // host copies in the host region of encoder outputs; the group's region of them holds eight
    // outputs of the largest admitted image, 32 rows, in 256-byte units.
    {
        let head = service::lock(engine.head.as_ref().unwrap());
        let (open, _) = head.session().unwrap();
        let class = capacity.cache_groups.len();
        // A text deployment holds no product: the class of stored trajectories follows.
        assert_eq!(usize::from(open.trajectory), class + 1);
        let outputs = &open.classes[class];
        assert_eq!(outputs.units, 0);
        let (host, _) = outputs.host.unwrap();
        let host = &open.regions[usize::from(host)];
        assert!(host.name.starts_with("host encoders"), "{}", host.name);
        assert_eq!(
            (host.tier, host.part_bytes),
            (Tier::Host, encoders::PAGE_BYTES)
        );

        let regions = &open.groups[0].regions[class];
        assert_eq!(regions.len(), 1);
        let region = &open.regions[usize::from(regions[0])];
        assert_eq!((region.tier, region.part_bytes), (Tier::Device, UNIT_BYTES));
        let largest = 32 * u64::from(feature_bytes);
        assert_eq!(u64::from(region.units), 8 * largest.div_ceil(UNIT_BYTES));
    }

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
        token_history: 0,
        feature_bytes,
        encoder_identity: String::new(),
        latent_seed: 0,
        media_bytes: Vec::new(),
        pipeline: None,
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
    let ceilings = crate::scheduler::policy::ceilings(
        text.iter().map(|(_, capacity)| capacity),
        &proto::config::PolicyBudget::default(),
    );
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
    let ceilings = crate::scheduler::policy::ceilings(
        vision.iter().map(|(_, capacity)| capacity),
        &proto::config::PolicyBudget::default(),
    );
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
