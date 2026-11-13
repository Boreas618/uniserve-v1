//! A text rank and a media rank started on the meta device against a Storage Server in the
//! test's process, in the order a session opens before its ranks map memory
//! (`specs/buffer-pool.md` §14.1).
//!
//! The text rank: its load to its division, the session the engine opens over the layout its
//! units state, the segments the rank creates once the answer shows the region new and the head
//! grants, the executor spawned over the rank's runner, which maps the session's completions into
//! its block tables, and a successor epoch of the same layout, whose rank maps those segments.
//! Its model is the Qwen3-MoE fixture of the `qwen3` crate, loaded from its shard headers alone.
//!
//! The media rank: its load and its runners' profile run, the session's media regions and
//! classes, its opening over the answer's numbers, the executor spawned over its media runners,
//! whose tables map a product the session allocates, and what its group states. Its model is a
//! released FastH3 checkpoint, loaded from its shard headers alone; the checkpoint is not part of
//! the repository, so the test is ignored by default and fails when the checkpoint is missing.
//!
//! No GPU is touched. The server runs on a thread of the test process with a socket of its own.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

use ipc::CacheGroupKind;
use ipc::batch::{Phase, Variant};
use proto::config::{
    BlockConfig, BlockDistribution, EngineArgs, WorkerGroupConfig, WorkerGroupId, WorkerRank,
};
use storage::QueuePair;
use storage::runtime::{Completion, GroupId, OwnerId, Request, RequestId, Tier};
use worker::completion::{Outcome, Submission};
use worker::memory::buffers::UNIT_BYTES;
use worker::memory::encoders;

use super::{
    Backing, LoadedRank, Meeting, bind, capacities, layout, load_ranks, open_ranks, spawn,
};
use crate::rank::ModelRunner;
use crate::storage_client::StorageClient;
use crate::support::{DEPLOYMENT, SLOTS, StorageServer, launch};
use ipc::product::KINDS;

/// Submits `request` under `number` on the scheduler's end `pair` and returns its completion, the
/// next to arrive, within ten seconds.
fn ask(pair: &mut QueuePair, number: u64, request: &Request) -> Completion {
    let mut record = Vec::new();
    request.encode(RequestId(number), &mut record);
    assert!(
        pair.submit(&record).unwrap(),
        "the submission queue is full"
    );

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut record = Vec::new();
    loop {
        if pair.next_completion(&mut record).unwrap() {
            let (answered, completion) = Completion::decode(&record).unwrap();
            assert_eq!(
                answered,
                RequestId(number),
                "{completion:?} answers another request"
            );
            return completion;
        }
        assert!(Instant::now() < deadline, "no completion arrived");
        std::thread::sleep(Duration::from_micros(100));
    }
}

/// Waits for a job of the group's executors, which must succeed.
fn succeeded(submission: worker::Result<Submission>) {
    match submission.unwrap().wait() {
        Outcome::Success(_) => {}
        Outcome::Failed(fault) => panic!("the executors refused the job: {fault:?}"),
        Outcome::Cancelled => panic!("the executors cancelled the job"),
    }
}

/// The ranks of `launch`'s process loaded onto the meta device to their division.
pub(crate) fn loaded(launch: &EngineArgs) -> Vec<LoadedRank> {
    let registry = nn::family::Registry::new([models::qwen3::family::FAMILY]).unwrap();
    let family = crate::resolve(&registry, Path::new(&launch.engine_config.model.model)).unwrap();
    let model_runner = ModelRunner::of(launch, family).unwrap();

    load_ranks(
        model_runner.ranks().unwrap(),
        Backing::Meta,
        launch,
        Meeting::Local,
    )
    .unwrap()
}

/// The layout of the session of `ranks`, `launch`'s ranks of one group of one rank.
fn stated(launch: &EngineArgs, ranks: &[LoadedRank]) -> storage::runtime::Open {
    layout(launch, ranks).unwrap()
}

#[test]
fn a_meta_rank_creates_its_region_s_segments_once_the_session_is_open_and_a_successor_maps_them() {
    let server = StorageServer::start();
    let launch = launch();

    // The rank's load to its division. A meta rank's cache holds the fewest units it serves
    // with: the padding unit, which the region reserves, and one page of its one cache group.
    let mut ranks = loaded(&launch);
    assert_eq!(ranks.len(), 1);
    assert!(
        !ranks[0].keys.is_empty(),
        "the profile run names the kernels the rank launches"
    );
    let units = ranks[0].text().unwrap().units;
    assert_eq!(units, 2);

    // The session over the rank's layout: one scheduler of one group, whose KV region states its
    // numbering and no segment, and asks for no unit, since none of its memory exists yet.
    let open = stated(&launch, &ranks);
    assert_eq!(open.schedulers, vec![vec![GroupId(0)]]);
    let group = &open.groups[0];
    let kv = usize::from(group.regions[0][0]);
    assert!(open.regions.iter().all(|region| region.segments.is_empty()));
    assert_eq!((open.regions[kv].units, open.regions[kv].limit), (0, units));

    let mut client = StorageClient::connect(&server.socket, DEPLOYMENT, launch.epoch).unwrap();
    let mut pairs = client.open(open.clone()).unwrap();
    let opened = client.session().unwrap().1.clone();
    assert_eq!(pairs.len(), 1);
    assert!(
        opened.segments[kv].is_empty(),
        "no session named the region before"
    );
    assert_eq!(opened.regions[kv].1, 0);
    assert_eq!(opened.regions[usize::from(group.slots)].1, SLOTS);

    // The rank creates the segments that cover its region's units, back to back from unit 0 and
    // keyed by its epoch, and the head grants them.
    let created = open_ranks(&mut ranks, &open, &opened).unwrap();
    assert_eq!(created.len(), 1);
    let (region, segments) = &created[0];
    assert_eq!(*region, open.regions[kv].name);
    let mut next = 0;
    for &(segment, first, count) in segments {
        assert_eq!(segment.epoch, launch.epoch);
        assert_eq!(first, next);
        next += count;
    }
    assert_eq!(next, units);
    for &(segment, first, count) in segments {
        client.grant(region, segment, first, count).unwrap();
    }

    bind(&mut ranks, &open, &opened).unwrap();
    let (executors, spawned) = spawn(&mut ranks, &client, &pairs, Vec::new()).unwrap();
    assert_eq!(spawned.len(), 1);

    // The region grows into the granted segments: every unit past the padding unit.
    let pair = &mut pairs[0];
    let number = opened.regions[kv].0;
    let resize = Request::Resize {
        region: number,
        units: units - 1,
    };
    let resized = ask(pair, 1, &resize);
    assert_eq!(
        resized,
        Completion::Resized {
            units: units - 1,
            evicted: Vec::new(),
        }
    );

    // An owner attached to the group and a block of class 0 allocated to it, both mapped into
    // the group's block tables, whose regions are the ones the session numbered: unit 1, the one
    // unit past the padding unit.
    let attach = Request::Attach {
        owner: OwnerId(1),
        group: GroupId(0),
    };
    let attached = ask(pair, 2, &attach);
    let Completion::Attached { slot } = attached else {
        panic!("the owner is attached: {attached:?}");
    };
    succeeded(executors.map(GroupId(0), slot, RequestId(2), attach, attached));

    let allocate = Request::Allocate {
        owner: OwnerId(1),
        class: 0,
        at: 0,
        count: 1,
        units: 0,
    };
    let allocated = ask(pair, 3, &allocate);
    let Completion::Allocated { runs, .. } = &allocated else {
        panic!("a block is allocated: {allocated:?}");
    };
    assert_eq!(*runs, vec![(number, 1, 1)]);
    succeeded(executors.map(GroupId(0), slot, RequestId(3), allocate, allocated));

    // What the group's ranks state of it: prefill, refill and decode on every slot, over its one
    // full-attention cache group of 64-token pages.
    let stated_capacities = capacities(&ranks, &open).unwrap();
    assert_eq!(stated_capacities.len(), 1);
    let (id, capacity) = &stated_capacities[0];
    assert_eq!(*id, GroupId(0));
    for phase in Phase::ALL {
        let served = matches!(phase, Phase::Prefill | Phase::Decode | Phase::Refill);
        assert_eq!(capacity.serves(phase), served, "{phase:?}");
        assert_eq!(capacity.rows[phase.index()], if served { SLOTS } else { 0 });
    }
    assert_eq!(capacity.slots, SLOTS);
    assert_eq!(capacity.cache_groups, vec![(CacheGroupKind::Full, 64)]);

    // The group's one key chain roots at the seed of the page identity its cache states for the
    // weights the process serves, which a meta start loads none of.
    let identities = worker::lock(&ranks[0].text().unwrap().cache)
        .page_identities(&crate::data_plane::identity_of(&[]));
    let seeds: Vec<storage::Key> = identities
        .iter()
        .map(|identity| storage::key::page_seed(identity))
        .collect();
    assert_eq!(capacity.seeds, seeds);
    assert_eq!(capacity.seeds.len(), 1);
    assert_ne!(capacity.seeds[0], storage::Key([0; 32]));
    assert_eq!(capacity.call_window, 2);
    assert_eq!(capacity.eos_token_ids, vec![2]);

    // A successor epoch of the same layout names the same region, whose answer states the
    // segments the first epoch's rank created: its rank lays its planes out over them and
    // creates none.
    let mut successor = launch.clone();
    successor.epoch = 2;
    let mut successors = loaded(&successor);
    let next_open = stated(&successor, &successors);
    assert_eq!(next_open.regions[kv].name, open.regions[kv].name);

    let mut next_client = StorageClient::connect(&server.socket, DEPLOYMENT, 2).unwrap();
    let _next_pairs = next_client.open(next_open.clone()).unwrap();
    let next_opened = next_client.session().unwrap().1.clone();
    assert_eq!(next_opened.regions[kv], (number, units - 1));
    assert_eq!(next_opened.segments[kv], *segments);

    let none = open_ranks(&mut successors, &next_open, &next_opened).unwrap();
    assert_eq!(none, vec![(open.regions[kv].name.clone(), Vec::new())]);
    let mapped: Vec<(u64, u32, u32)> = worker::lock(&successors[0].text().unwrap().cache)
        .config()
        .segments
        .iter()
        .flatten()
        .map(|part| (part.segment.epoch, part.segment.ordinal, part.segment.units))
        .collect();
    let table: Vec<(u64, u32, u32)> = segments
        .iter()
        .map(|(segment, _, count)| (segment.epoch, segment.ordinal, *count))
        .collect();
    assert_eq!(mapped, table);
    next_client.close().unwrap();

    // The executors stop before the session closes.
    drop(executors);
    drop(spawned);
    client.close().unwrap();
}

/// The released FastH3 checkpoint the media rank loads the headers of: four denoising steps.
pub(crate) const FAST_H3: &str =
    "/workspace/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree";

/// The launch of epoch 1's one main process serving [`FAST_H3`] on one rank of one group that
/// places every block, as `configs/minimax_h3/single.toml` does, at its smallest: square videos
/// of the shortest timeline, prompts of up to 256 tokens and [`SLOTS`] request slots.
pub(crate) fn video_launch() -> EngineArgs {
    let mut launch = launch();
    let every = || BlockConfig::parallel(vec![0]);
    let dealt = || BlockConfig {
        distribution: Some(BlockDistribution::TemporalUnits),
        ..every()
    };
    let blocks = BTreeMap::from([
        ("transformer".to_owned(), every()),
        ("text_encoder".to_owned(), every()),
        ("video_decoder".to_owned(), dealt()),
        ("audio_decoder".to_owned(), dealt()),
        ("latent_encoder".to_owned(), dealt()),
        ("video_codec".to_owned(), dealt()),
        ("muxer".to_owned(), every()),
        ("media_reader".to_owned(), every()),
    ]);
    let group = WorkerGroupConfig::placed(&["localhost".to_owned()], "cuda", 1, 2, blocks);

    let config = &mut launch.engine_config;
    config.model.model = FAST_H3.to_owned();
    config.model.max_model_len = 256;
    config.parallel.worker_groups = vec![group.clone()];
    config.video = proto::config::VideoConfig {
        max_seconds: 4.0,
        sizes: vec!["768x768".to_owned()],
        text_capacities: None,
        max_condition_rows: None,
    };
    launch.group = group.id.0;
    launch
}

#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn a_meta_media_rank_opens_its_media_memory_and_spawns_its_runners() {
    assert!(
        Path::new(FAST_H3).is_dir(),
        "the media rank loads the headers of {FAST_H3}, which is missing"
    );
    let server = StorageServer::start();
    let launch = video_launch();

    // The rank's load and its runners' profile run: every phase of a video request on its one
    // rank, a slot's row holding the largest trajectory, and its group's product sizes.
    let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
    let family = crate::resolve(&registry, Path::new(FAST_H3)).unwrap();
    let model_runner = ModelRunner::of(&launch, family).unwrap();
    let mut ranks = load_ranks(
        model_runner.ranks().unwrap(),
        Backing::Meta,
        &launch,
        Meeting::Local,
    )
    .unwrap();
    assert_eq!(ranks.len(), 1);
    assert!(!ranks[0].keys.is_empty(), "the profile run names kernels");
    let video = ranks[0].video().unwrap();
    let (tensors, encoding, pages) = video.trajectory.unwrap();
    assert!(video.row > 0);
    let media_bytes = video.capacity.media_bytes.clone();
    assert!(!media_bytes.is_empty());
    assert!(
        media_bytes
            .iter()
            .all(|(_, trajectory, bytes)| bytes.len() == KINDS.len() && *trajectory > 0)
    );

    // The session: the host regions of encoder outputs and stored trajectories, the rank's
    // product buffer and the group's slot rows; one class per product kind, the conditioning's
    // host copies in the encoder region, and the class of stored trajectories in the latent one.
    let open = layout(&launch, &ranks).unwrap();
    assert_eq!(open.schedulers, vec![vec![GroupId(0)]]);
    let named = |prefix: &str| {
        let index = open
            .regions
            .iter()
            .position(|region| region.name.starts_with(prefix))
            .unwrap_or_else(|| panic!("the session states no region {prefix}"));
        (u8::try_from(index).unwrap(), &open.regions[index])
    };
    // A host region's unit is one part, the one host copy of a block one rank holds whole.
    let (encoders_index, encoders_region) = named("host encoders");
    assert_eq!(encoders_region.tier, Tier::Host);
    assert_eq!(encoders_region.part_bytes, encoders::PAGE_BYTES);
    assert_eq!(encoders_region.unit_bytes, encoders::PAGE_BYTES);
    let (latents_index, latents_region) = named("host latents");
    assert_eq!(latents_region.tier, Tier::Host);
    assert_eq!(latents_region.part_bytes, pages.page_bytes().unwrap());
    assert_eq!(latents_region.unit_bytes, latents_region.part_bytes);
    assert_eq!(
        u64::from(latents_region.units),
        worker::memory::latent::host::pool_pages(pages)
    );
    let (products_index, products) = named("products 1 0");
    assert_eq!(
        (products.tier, products.part_bytes),
        (Tier::Device, UNIT_BYTES)
    );
    assert_eq!(products.units, products.limit);
    let (slots_index, slot_rows) = named("slots 1 0");
    assert_eq!((slot_rows.reserved, slot_rows.units), (1, SLOTS));
    assert_eq!(slot_rows.part_bytes, video.row.next_multiple_of(UNIT_BYTES));
    assert_eq!(open.regions.len(), 4);

    assert_eq!(open.classes.len(), KINDS.len() + 1);
    assert_eq!(usize::from(open.trajectory), KINDS.len());
    for (kind, class) in open.classes[..KINDS.len()].iter().enumerate() {
        let name = KINDS[kind].0;
        assert_eq!(class.units, 0, "{name}");
        let conditioning = name == ipc::product::CONDITIONING;
        assert_eq!(
            class.host,
            conditioning.then_some((encoders_index, 0)),
            "{name}"
        );
    }
    let trajectory = &open.classes[KINDS.len()];
    assert_eq!(trajectory.host, Some((latents_index, 0)));
    assert_eq!(
        (trajectory.tensors, trajectory.encoding),
        (tensors, encoding)
    );

    // Every product lies in the rank's product buffer, which backs them all: a block of every
    // request slot of each kind fits it.
    let group = &open.groups[0];
    assert_eq!(group.slots, slots_index);
    assert_eq!(group.regions.len(), open.classes.len());
    for regions in &group.regions[..KINDS.len()] {
        assert_eq!(*regions, vec![products_index]);
    }
    assert!(group.regions[KINDS.len()].is_empty());
    let request_units: u64 = (0..KINDS.len())
        .map(|kind| {
            let largest = media_bytes.iter().map(|(_, _, bytes)| bytes[kind]).max();
            largest.unwrap().div_ceil(UNIT_BYTES).max(1)
        })
        .sum();
    assert!(u64::from(products.units) >= request_units * u64::from(SLOTS));

    // The session opens, the rank opens its memory over the answer's numbers and creates no
    // segment, and its executor is spawned over its media runners.
    let mut client = StorageClient::connect(&server.socket, DEPLOYMENT, launch.epoch).unwrap();
    let mut pairs = client.open(open.clone()).unwrap();
    let opened = client.session().unwrap().1.clone();
    assert!(opened.segments.iter().all(Vec::is_empty));
    let created = open_ranks(&mut ranks, &open, &opened).unwrap();
    assert!(created.is_empty());
    bind(&mut ranks, &open, &opened).unwrap();
    let (executors, spawned) = spawn(&mut ranks, &client, &pairs, Vec::new()).unwrap();
    assert_eq!(spawned.len(), 1);

    // An owner attached to the group and the video latents allocated to it: one run in the rank's
    // product buffer, mapped into the group's block tables.
    let pair = &mut pairs[0];
    let attach = Request::Attach {
        owner: OwnerId(1),
        group: GroupId(0),
    };
    let attached = ask(pair, 1, &attach);
    let Completion::Attached { slot } = attached else {
        panic!("the owner is attached: {attached:?}");
    };
    succeeded(executors.map(GroupId(0), slot, RequestId(1), attach, attached));
    let latents = u8::try_from(ipc::product::kind(ipc::product::VIDEO_LATENTS).unwrap()).unwrap();
    let allocate = Request::Allocate {
        owner: OwnerId(1),
        class: latents,
        at: 4,
        count: 1,
        units: 3,
    };
    let allocated = ask(pair, 2, &allocate);
    let Completion::Allocated { runs, .. } = &allocated else {
        panic!("a block is allocated: {allocated:?}");
    };
    assert_eq!(
        *runs,
        vec![(opened.regions[usize::from(products_index)].0, 0, 3)]
    );
    succeeded(executors.map(GroupId(0), slot, RequestId(2), allocate, allocated));

    // What the group states: every phase of a video request, one request a stage of the
    // denoiser and the decoders, the encode's slots, the model's steps, and its product sizes.
    let stated = capacities(&ranks, &open).unwrap();
    assert_eq!(stated.len(), 1);
    let (id, capacity) = &stated[0];
    assert_eq!(*id, GroupId(0));
    let served = [
        Phase::MediaRead,
        Phase::TextEncode,
        Phase::LatentPrepare,
        Phase::Denoise,
        Phase::VideoDecode,
        Phase::AudioDecode,
        Phase::VideoEncode,
        Phase::AudioEncode,
        Phase::Mux,
    ];
    for phase in Phase::ALL {
        assert_eq!(capacity.serves(phase), served.contains(&phase), "{phase:?}");
    }
    for phase in [
        Phase::LatentPrepare,
        Phase::Denoise,
        Phase::VideoDecode,
        Phase::AudioDecode,
    ] {
        assert_eq!(capacity.rows[phase.index()], 1, "{phase:?}");
    }
    assert_eq!(capacity.rows[Phase::TextEncode.index()], SLOTS);
    assert_eq!(capacity.denoise_steps, 4);
    assert_eq!(capacity.media_bytes, media_bytes);
    assert!(capacity.kv_cache.is_none() && capacity.cache_groups.is_empty());
    for variant in [
        Variant::MediaRead,
        Variant::VideoEncode,
        Variant::AudioEncode,
        Variant::Mux,
    ] {
        assert!(
            capacity
                .buckets
                .iter()
                .any(|shape| shape.variant == variant),
            "{variant:?}"
        );
    }

    // The executors stop before the session closes.
    drop(executors);
    drop(spawned);
    client.close().unwrap();
}

/// The launch of [`FAST_H3`] over two devices of one process, each the one rank of a group: the
/// `text` group encodes prompts on device 0, and the `flow` group serves every other phase on
/// device 1, as `configs/minimax_h3/fasth3-ulysses4x2.toml` splits them.
fn split_launch() -> EngineArgs {
    let mut launch = video_launch();
    let one = || BlockConfig::parallel(vec![0]);
    let dealt = || BlockConfig {
        distribution: Some(BlockDistribution::TemporalUnits),
        ..one()
    };
    let group = |id: &str, device: &str, blocks: BTreeMap<String, BlockConfig>| WorkerGroupConfig {
        id: WorkerGroupId(id.to_owned()),
        role: Default::default(),
        replicas: 1,
        ranks: vec![WorkerRank {
            host: "localhost".to_owned(),
            device: device.to_owned(),
        }],
        blocks,
        queue_depth: 2,
        mem_fraction_static: None,
    };
    let text = group(
        "text",
        "cuda:0",
        BTreeMap::from([("text_encoder".to_owned(), one())]),
    );
    let flow = group(
        "flow",
        "cuda:1",
        BTreeMap::from([
            ("transformer".to_owned(), one()),
            ("video_decoder".to_owned(), dealt()),
            ("audio_decoder".to_owned(), dealt()),
            ("latent_encoder".to_owned(), dealt()),
            ("video_codec".to_owned(), dealt()),
            ("muxer".to_owned(), one()),
            ("media_reader".to_owned(), one()),
        ]),
    );
    launch.engine_config.parallel.worker_groups = vec![text, flow];
    launch.devices = vec![0, 1];
    launch.world_size = 2;
    launch
}

#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn the_conditioning_lies_with_the_text_encoder_and_the_denoising_group_reads_it() {
    assert!(Path::new(FAST_H3).is_dir(), "{FAST_H3} is missing");
    let server = StorageServer::start();
    let launch = split_launch();

    let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
    let family = crate::resolve(&registry, Path::new(FAST_H3)).unwrap();
    let model_runner = ModelRunner::of(&launch, family).unwrap();
    let mut ranks = load_ranks(
        model_runner.ranks().unwrap(),
        Backing::Meta,
        &launch,
        Meeting::Local,
    )
    .unwrap();
    assert_eq!(ranks.len(), 2);
    assert_eq!(
        ranks[0].video().unwrap().row,
        0,
        "the text group denoises nothing"
    );
    assert!(ranks[1].video().unwrap().row > 0);

    // One scheduler of both groups. The text group writes the conditioning into its rank's
    // product buffer alone and reads the vision pixels the flow group's media read writes; the
    // flow group writes every other product into its rank's buffer and reads the conditioning
    // where the text group wrote it.
    let open = layout(&launch, &ranks).unwrap();
    assert_eq!(open.schedulers, vec![vec![GroupId(0), GroupId(1)]]);
    let region = |name: &str| {
        let index = open.regions.iter().position(|region| region.name == name);
        u8::try_from(index.unwrap_or_else(|| panic!("no region {name}"))).unwrap()
    };
    let (text_buffer, flow_buffer) = (region("products 1 0"), region("products 1 1"));
    let (text, flow) = (&open.groups[0], &open.groups[1]);
    for (kind, (name, ..)) in KINDS.iter().enumerate() {
        let (texts, flows) = match *name {
            ipc::product::CONDITIONING => (vec![text_buffer], vec![text_buffer]),
            ipc::product::VISION_PIXELS => (vec![flow_buffer], vec![flow_buffer]),
            _ => (Vec::new(), vec![flow_buffer]),
        };
        assert_eq!(text.regions[kind], texts, "{name} on the text group");
        assert_eq!(flow.regions[kind], flows, "{name} on the flow group");
    }

    // Both ranks open, each reaching the other's buffer, and an executor serves each group.
    let mut client = StorageClient::connect(&server.socket, DEPLOYMENT, launch.epoch).unwrap();
    let mut pairs = client.open(open.clone()).unwrap();
    let opened = client.session().unwrap().1.clone();
    assert!(open_ranks(&mut ranks, &open, &opened).unwrap().is_empty());
    let (executors, spawned) = spawn(&mut ranks, &client, &pairs, Vec::new()).unwrap();
    assert_eq!(spawned.len(), 2);

    // A request's conditioning, allocated for its owner on the text group in the text rank's
    // buffer, is mapped into the flow group's tables under the request's slot there.
    let pair = &mut pairs[0];
    let mut slots = Vec::new();
    for (number, (owner, group)) in [(1, (OwnerId(1), GroupId(0))), (2, (OwnerId(2), GroupId(1)))] {
        let attach = Request::Attach { owner, group };
        let attached = ask(pair, number, &attach);
        let Completion::Attached { slot } = attached else {
            panic!("the owner is attached: {attached:?}");
        };
        succeeded(executors.map(group, slot, RequestId(number), attach, attached));
        slots.push(slot);
    }
    let conditioning = ipc::product::kind(ipc::product::CONDITIONING).unwrap();
    let allocate = Request::Allocate {
        owner: OwnerId(1),
        class: u8::try_from(conditioning).unwrap(),
        at: 3,
        count: 1,
        units: 2,
    };
    let allocated = ask(pair, 3, &allocate);
    let Completion::Allocated { runs, .. } = &allocated else {
        panic!("a block is allocated: {allocated:?}");
    };
    assert_eq!(
        *runs,
        vec![(opened.regions[usize::from(text_buffer)].0, 0, 2)]
    );
    for (group, slot) in [(GroupId(0), slots[0]), (GroupId(1), slots[1])] {
        let mapped = executors.map(
            group,
            slot,
            RequestId(3),
            allocate.clone(),
            allocated.clone(),
        );
        succeeded(mapped);
    }

    drop(executors);
    drop(spawned);
    client.close().unwrap();
}

/// The launch of [`FAST_H3`] as `configs/minimax_h3/fasth3-dp8-text-tp8.toml` places it on one
/// host of eight devices: the `text` group's eight ranks encode prompts eight-way tensor
/// parallel, and eight one-rank `flow` replicas serve every other phase, replica `i` on the
/// device of text rank `i`.
fn dp8_launch() -> EngineArgs {
    let mut launch = video_launch();
    let file = crate::support::workspace().join("configs/minimax_h3/fasth3-dp8-text-tp8.toml");
    let text = std::fs::read_to_string(file).unwrap();
    launch.engine_config.parallel.worker_groups =
        WorkerGroupConfig::parse_file(&text, "localhost").unwrap();
    launch.devices = (0..8).collect();
    launch.world_size = 8;
    launch
}

#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn a_device_holds_a_rank_of_the_text_group_and_one_of_a_flow_replica() {
    assert!(Path::new(FAST_H3).is_dir(), "{FAST_H3} is missing");
    let server = StorageServer::start();
    let launch = dp8_launch();
    assert_eq!(launch.engine_config.parallel.worker_groups.len(), 9);

    // Sixteen ranks, two on each device: text rank `i` and the rank of flow replica `i`.
    let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
    let family = crate::resolve(&registry, Path::new(FAST_H3)).unwrap();
    let model_runner = ModelRunner::of(&launch, family).unwrap();
    let placed = model_runner.ranks().unwrap();
    assert_eq!(placed.len(), 16);
    for device in 0..8 {
        let on: Vec<u32> = placed
            .iter()
            .filter(|rank| rank.devices() == [platform::DeviceId(device)])
            .map(ModelRunner::rank)
            .collect();
        assert_eq!(on, [device as u32, 8 + device as u32], "device {device}");
    }
    let mut ranks = load_ranks(placed, Backing::Meta, &launch, Meeting::Local).unwrap();
    for (index, rank) in ranks.iter().enumerate() {
        let denoises = rank.video().unwrap().row > 0;
        assert_eq!(denoises, index >= 8, "rank {index}");
    }

    // One scheduler of the nine groups; a product buffer per rank; a host region's unit is one
    // part, the one host copy of a block one rank holds whole.
    let open = layout(&launch, &ranks).unwrap();
    assert_eq!(
        open.schedulers,
        vec![(0..9).map(GroupId).collect::<Vec<_>>()]
    );
    let region = |name: &str| {
        let index = open.regions.iter().position(|region| region.name == name);
        u8::try_from(index.unwrap_or_else(|| panic!("no region {name}"))).unwrap()
    };
    for rank in 0..16 {
        region(&format!("products 1 {rank}"));
    }
    for host in ["host encoders", "host latents"] {
        let stated = open
            .regions
            .iter()
            .find(|region| region.name.starts_with(host));
        let stated = stated.unwrap_or_else(|| panic!("no region {host}"));
        assert_eq!(stated.unit_bytes, stated.part_bytes, "{host}");
    }

    // The conditioning lies with text rank 0 alone, where every flow replica reads it.
    let conditioning = ipc::product::kind(ipc::product::CONDITIONING).unwrap();
    let text_buffer = region("products 1 0");
    for group in &open.groups {
        assert_eq!(group.regions[conditioning], vec![text_buffer]);
    }

    // The session opens, every rank opens its own memory, two of them on each device, and an
    // executor serves each rank.
    let mut client = StorageClient::connect(&server.socket, DEPLOYMENT, launch.epoch).unwrap();
    let mut pairs = client.open(open.clone()).unwrap();
    let opened = client.session().unwrap().1.clone();
    assert!(open_ranks(&mut ranks, &open, &opened).unwrap().is_empty());
    let (executors, spawned) = spawn(&mut ranks, &client, &pairs, Vec::new()).unwrap();
    assert_eq!(spawned.len(), 16);

    // A request's conditioning, allocated for its owner on the text group in text rank 0's
    // buffer, is mapped into the tables of the flow replica that serves the request, whose ranks
    // share text rank 0's device or not.
    let pair = &mut pairs[0];
    let mut slots = Vec::new();
    for (number, (owner, group)) in [(1, (OwnerId(1), GroupId(0))), (2, (OwnerId(2), GroupId(4)))] {
        let attach = Request::Attach { owner, group };
        let attached = ask(pair, number, &attach);
        let Completion::Attached { slot } = attached else {
            panic!("the owner is attached: {attached:?}");
        };
        succeeded(executors.map(group, slot, RequestId(number), attach, attached));
        slots.push(slot);
    }
    let allocate = Request::Allocate {
        owner: OwnerId(1),
        class: u8::try_from(conditioning).unwrap(),
        at: 3,
        count: 1,
        units: 2,
    };
    let allocated = ask(pair, 3, &allocate);
    let Completion::Allocated { runs, .. } = &allocated else {
        panic!("a block is allocated: {allocated:?}");
    };
    assert_eq!(
        *runs,
        vec![(opened.regions[usize::from(text_buffer)].0, 0, 2)]
    );
    for (group, slot) in [(GroupId(0), slots[0]), (GroupId(4), slots[1])] {
        let mapped = executors.map(
            group,
            slot,
            RequestId(3),
            allocate.clone(),
            allocated.clone(),
        );
        succeeded(mapped);
    }

    // What the groups state: the text group encodes prompts, and each replica every other phase.
    let stated = capacities(&ranks, &open).unwrap();
    assert_eq!(stated.len(), 9);
    for (id, capacity) in &stated {
        let text = *id == GroupId(0);
        assert_eq!(capacity.serves(Phase::TextEncode), text, "group {}", id.0);
        assert_eq!(capacity.serves(Phase::Denoise), !text, "group {}", id.0);
    }

    drop(executors);
    drop(spawned);
    client.close().unwrap();
}
