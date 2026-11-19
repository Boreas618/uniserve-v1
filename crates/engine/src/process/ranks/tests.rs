//! A text rank and a media rank started on the meta device against a Storage Server in the
//! test's process, in the order a session opens before its ranks map memory
//! (`specs/buffer-pool.md` §14.1).
//!
//! The text rank: its load to its division, the session the engine opens over the layout its
//! units state, the segments the rank creates once the answer shows the region new and the head
//! grants, the worker spawned over the rank's runner, which maps the session's completions into
//! its block tables, and successor epochs of the same layout, whose ranks map those segments at
//! the region's numbering whatever their own division counted.
//! Its model is the Qwen3-MoE fixture of the `qwen3` crate, loaded from its shard headers alone.
//!
//! The media rank: its load and its runners' profile run, the session's media regions and
//! classes, its opening over the answer's numbers, the worker spawned over its media runners,
//! whose tables map a product the session allocates, and what its group states. Its model is a
//! released FastH3 checkpoint, loaded from its shard headers alone; the checkpoint is not part of
//! the repository, so the test is ignored by default and fails when the checkpoint is missing.
//!
//! No GPU is touched. The server runs on a thread of the test process with a socket of its own.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::batch::cache::CacheGroupKind;
use crate::batch::{Phase, Variant};
use crate::memory::buffers::UNIT_BYTES;
use crate::memory::encoders;
use crate::worker::completion::{Outcome, Submission};
use proto::config::{
    BlockConfig, BlockDistribution, EngineArgs, WorkerGroupConfig, WorkerGroupId, WorkerRank,
};
use storage::runtime::{Completion, GroupId, OwnerId, Request, RequestId, Tier};
use storage::{Interest, QueuePair};

use super::{
    Backing, Loaded, LoadedRank, Meeting, bind, capacities, layout, load_ranks, open_ranks, spawn,
};
use crate::memory::client::StorageClient;
use crate::process::ranks::rank::ModelRunner;
use crate::process::ranks::rank::text::Keys;
use crate::process::ranks::rank::video::MediaCollective;
use nn::pipeline::{Pipeline, Product};
use testkit::LONG;
use testkit::serving::{DEPLOYMENT, SLOTS, StorageServer, launch};

/// The multiprocessors of the device each rank of the tests' starts stands in for, which a member
/// of an expert union states in its exchange's key: the count the exchange's own tests state.
const MULTIPROCESSORS: u32 = 148;

/// What the tests' starts load onto: nothing outside the process, each rank's device stating
/// [`MULTIPROCESSORS`], since the host need not hold the devices the launches name.
const META: Backing<'static> = Backing::Meta {
    multiprocessors: |_| Ok(MULTIPROCESSORS),
};

/// The pipeline of the media cases: MiniMax-H3's, as its family declares it.
fn pipeline() -> Pipeline {
    models::minimax_h3::family::pipeline()
}

/// The collective groups of `block` along `axis` (`None`: its carry communicator) among the kinds
/// a media rank forms.
fn formed<'a>(
    collectives: &'a [MediaCollective],
    block: &str,
    axis: Option<usize>,
) -> &'a super::rank::Collective {
    &collectives
        .iter()
        .find(|kind| kind.block == block && kind.axis == axis)
        .unwrap_or_else(|| panic!("no collective of block {block} along {axis:?}"))
        .collective
}

/// Submits `request` under `number` on the scheduler's end `pair` and returns its completion, the
/// next to arrive, waiting on the end's doorbell for the buffer pool's records.
fn ask(pair: &mut QueuePair, number: u64, request: &Request) -> Completion {
    let mut record = Vec::new();
    request.encode(RequestId(number), &mut record);
    assert!(
        pair.submit(&record).unwrap(),
        "the submission queue is full"
    );

    let mut record = Vec::new();
    loop {
        // The counts are taken before the look, so a record written after it rings.
        let seen = pair.signals();
        if pair.next_completion(&mut record).unwrap() {
            let (answered, completion) = Completion::decode(&record).unwrap();
            assert_eq!(
                answered,
                RequestId(number),
                "{completion:?} answers another request"
            );
            return completion;
        }
        assert!(
            pair.wait(Interest::RECORDS, seen, Some(LONG)).unwrap(),
            "no completion arrived"
        );
    }
}

/// Waits for a job of the group's workers, which must succeed.
fn succeeded(submission: crate::worker::Result<Submission>) {
    match submission.unwrap().wait() {
        Outcome::Success(_) => {}
        Outcome::Failed(fault) => panic!("the workers refused the job: {fault:?}"),
        Outcome::Cancelled => panic!("the workers cancelled the job"),
    }
}

/// The ranks of `launch`'s process loaded onto the meta device to their division.
pub(crate) fn loaded(launch: &EngineArgs) -> Vec<LoadedRank> {
    let registry = nn::family::Registry::new([models::qwen3::family::FAMILY]).unwrap();
    let family =
        crate::process::catalogue::resolve(&registry, Path::new(&launch.engine_config.model.model))
            .unwrap();
    let model_runner = ModelRunner::of(launch, family).unwrap();

    load_ranks(model_runner.ranks().unwrap(), META, launch, Meeting::Local).unwrap()
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

    let mut client =
        StorageClient::connect_head(&server.socket, DEPLOYMENT, launch.epoch, launch.engine)
            .unwrap();
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
    let created = open_ranks(&mut ranks, &open, &opened, None).unwrap();
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

    bind(&mut ranks, &open, &opened, true).unwrap();
    let (workers, spawned) = spawn(
        &mut ranks,
        (&open, &opened),
        &|group| client.completed_registers(group, &pairs),
        Vec::new(),
    )
    .unwrap();
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
    succeeded(workers.map(GroupId(0), slot, RequestId(2), attach, attached));

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
    succeeded(workers.map(GroupId(0), slot, RequestId(3), allocate, allocated));

    // What the group's ranks state of it: prefill, refill and decode on every slot, over its one
    // full-attention cache group of 64-token pages.
    let stated_capacities = capacities(&ranks, &open, (&[], Vec::new())).unwrap();
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
    let identities = service::lock(&ranks[0].text().unwrap().cache)
        .page_identities(&crate::process::binding::page_identity(&[]));
    let seeds: Vec<storage::Key> = identities.iter().map(storage::key::page_seed).collect();
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

    let mut next_client = StorageClient::connect_head(&server.socket, DEPLOYMENT, 2, 2).unwrap();
    let _next_pairs = next_client.open(next_open.clone()).unwrap();
    let next_opened = next_client.session().unwrap().1.clone();
    assert_eq!(next_opened.regions[kv], (number, units - 1));
    assert_eq!(next_opened.segments[kv], *segments);

    let none = open_ranks(&mut successors, &next_open, &next_opened, None).unwrap();
    assert_eq!(none, vec![(open.regions[kv].name.clone(), Vec::new())]);
    let mapped: Vec<(u64, u32, u32)> = service::lock(&successors[0].text().unwrap().cache)
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

    // A successor of the same layout whose own division numbers its cache otherwise, as one
    // whose figure of its GPU's quota rounds to another count of segments does, still names the
    // region, and takes the numbering the buffer pool holds for it: its rank maps the segments the
    // first epoch's rank created, creates none, and lays its cache out over the region's units.
    let mut measured = launch.clone();
    measured.epoch = 3;
    let mut others = loaded(&measured);
    {
        let Loaded::Text(text) = &mut others[0].loaded else {
            panic!("a text rank");
        };
        text.units = 2 * units;
        text.first = 2 * units;
    }
    let other_open = stated(&measured, &others);
    assert_eq!(other_open.regions[kv].name, open.regions[kv].name);
    assert_eq!(other_open.regions[kv].limit, 2 * units);

    let mut other_client = StorageClient::connect_head(&server.socket, DEPLOYMENT, 3, 3).unwrap();
    let _other_pairs = other_client.open(other_open).unwrap();
    let (held, other_opened) = {
        let (held, answered) = other_client.session().unwrap();
        (held.clone(), answered.clone())
    };
    assert_eq!(other_opened.limits[kv], units);
    assert_eq!(held.regions[kv].limit, units);
    assert_eq!(other_opened.regions[kv], (number, units - 1));
    assert_eq!(other_opened.segments[kv], *segments);

    let none = open_ranks(&mut others, &held, &other_opened, None).unwrap();
    assert_eq!(none, vec![(open.regions[kv].name.clone(), Vec::new())]);
    let cache = service::lock(&others[0].text().unwrap().cache);
    assert_eq!(cache.config().units, units);
    let mapped: Vec<(u64, u32, u32)> = cache
        .config()
        .segments
        .iter()
        .flatten()
        .map(|part| (part.segment.epoch, part.segment.ordinal, part.segment.units))
        .collect();
    assert_eq!(mapped, table);
    drop(cache);
    other_client.close().unwrap();

    // The workers stop before the session closes.
    drop(workers);
    drop(spawned);
    client.close().unwrap();
}

/// The released FastH3 checkpoint the media rank loads the headers of: four denoising steps.
pub(crate) fn fast_h3() -> PathBuf {
    testkit::assets::FAST_H3_4_STEP.snapshot()
}

/// The launch of epoch 1's one main process serving [`fast_h3`] on one rank of one group that
/// places every block at its smallest: square videos of the shortest timeline, prompts of up to 256 tokens and [`SLOTS`] request slots.
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
    let group = WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        proto::config::DeviceKind::Cuda,
        1,
        2,
        blocks,
    );

    let config = &mut launch.engine_config;
    config.model.model = fast_h3().to_string_lossy().into_owned();
    config.model.max_model_len = 256;
    config.parallel.worker_groups = vec![group.clone()];
    config.video = proto::config::VideoConfig {
        max_seconds: Some(4.0),
        sizes: Some(vec!["768x768".to_owned()]),
        ..proto::config::VideoConfig::default()
    };
    launch.group = group.id.0;
    launch
}

#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn a_meta_media_rank_opens_its_media_memory_and_spawns_its_runners() {
    let server = StorageServer::start();
    let launch = video_launch();

    // The rank's load and its runners' profile run: every phase of a video request on its one
    // rank, a slot's row holding the largest trajectory, and its group's product sizes.
    let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
    let family = crate::process::catalogue::resolve(&registry, &fast_h3()).unwrap();
    let model_runner = ModelRunner::of(&launch, family).unwrap();
    let mut ranks =
        load_ranks(model_runner.ranks().unwrap(), META, &launch, Meeting::Local).unwrap();
    assert_eq!(ranks.len(), 1);
    assert!(!ranks[0].keys.is_empty(), "the profile run names kernels");
    let video = ranks[0].video().unwrap();
    let (tensors, encoding, pages) = video.trajectory.unwrap();
    assert!(video.row > 0);
    // No media runner is shadowed, so the rank's scratch pool maps nothing beside its graph
    // pool's regions, whatever its forwards allocate.
    let shadows: Vec<u64> = video.peaks.iter().map(|(_, peaks)| peaks.shadow).collect();
    assert_eq!(video.peaks.len(), 4);
    assert_eq!(
        crate::memory::division::pool_bytes(UNIT_BYTES, &[], &shadows, 0).1,
        0
    );
    let media_bytes = video.capacity.media_bytes.clone();
    assert!(!media_bytes.is_empty());
    assert!(
        media_bytes.iter().all(
            |(_, trajectory, bytes)| bytes.len() == pipeline().products.len() && *trajectory > 0
        )
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
        crate::memory::latent::host::pool_pages(pages)
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

    assert_eq!(open.classes.len(), pipeline().products.len() + 1);
    assert_eq!(usize::from(open.trajectory), pipeline().products.len());
    for (kind, class) in open.classes[..pipeline().products.len()].iter().enumerate() {
        let name = pipeline().products[kind].product.name();
        assert_eq!(class.units, 0, "{name}");
        let conditioning = name == Product::Conditioning.name();
        assert_eq!(
            class.host,
            conditioning.then_some((encoders_index, 0)),
            "{name}"
        );
    }
    let trajectory = &open.classes[pipeline().products.len()];
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
    for regions in &group.regions[..pipeline().products.len()] {
        assert_eq!(*regions, vec![products_index]);
    }
    assert!(group.regions[pipeline().products.len()].is_empty());
    let request_units: u64 = (0..pipeline().products.len())
        .map(|kind| {
            let largest = media_bytes.iter().map(|(_, _, bytes)| bytes[kind]).max();
            largest.unwrap().div_ceil(UNIT_BYTES).max(1)
        })
        .sum();
    assert!(u64::from(products.units) >= request_units * u64::from(SLOTS));

    // The session opens, the rank opens its memory over the answer's numbers and creates no
    // segment, and its worker is spawned over its media runners.
    let mut client =
        StorageClient::connect_head(&server.socket, DEPLOYMENT, launch.epoch, launch.engine)
            .unwrap();
    let mut pairs = client.open(open.clone()).unwrap();
    let opened = client.session().unwrap().1.clone();
    assert!(opened.segments.iter().all(Vec::is_empty));
    let created = open_ranks(&mut ranks, &open, &opened, None).unwrap();
    assert!(created.is_empty());
    bind(&mut ranks, &open, &opened, true).unwrap();
    let (workers, spawned) = spawn(
        &mut ranks,
        (&open, &opened),
        &|group| client.completed_registers(group, &pairs),
        Vec::new(),
    )
    .unwrap();
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
    succeeded(workers.map(GroupId(0), slot, RequestId(1), attach, attached));
    let latents = u8::try_from(pipeline().kind(Product::VideoLatents).unwrap()).unwrap();
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
    succeeded(workers.map(GroupId(0), slot, RequestId(2), allocate, allocated));

    // What the group states: every phase of a video request, one request a stage of the
    // denoiser and the decoders, the encode's slots, the model's steps, and its product sizes.
    let stated = capacities(&ranks, &open, (&[], Vec::new())).unwrap();
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

    // The workers stop before the session closes.
    drop(workers);
    drop(spawned);
    client.close().unwrap();
}

/// A media group's request slots are the deployment's running requests, or as many as its
/// ranks' shares of their GPUs hold when fewer fit (`specs/engine-worker-port.md` §1,
/// `request_slots`), a slot taking the trajectory row and the product blocks one request holds
/// on a rank: the session's slot rows, product buffer and host latent pages and what the group
/// states follow them, and its captured buckets, which the scheduling policy's limits are stated
/// against, stay the deployment's. A share that holds no slot refuses the start, naming what one
/// slot needs.
#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn a_media_group_holds_as_many_request_slots_as_its_ranks_shares_hold() {
    let launch = video_launch();
    let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
    let family = crate::process::catalogue::resolve(&registry, &fast_h3()).unwrap();
    let model_runner = ModelRunner::of(&launch, family).unwrap();
    let load = || load_ranks(model_runner.ranks().unwrap(), META, &launch, Meeting::Local).unwrap();
    let region = |open: &storage::runtime::Open, name: &str| {
        open.regions
            .iter()
            .find(|region| region.name == name)
            .unwrap_or_else(|| panic!("the session states no region {name}"))
            .clone()
    };

    // A start backed by nothing holds the deployment's slots. What one slot takes of the rank's
    // share: its trajectory row and the product units one request's blocks take.
    let mut ranks = load();
    let open = layout(&launch, &ranks).unwrap();
    let (products, rows) = (region(&open, "products 1 0"), region(&open, "slots 1 0"));
    assert_eq!(rows.units, SLOTS);
    assert_eq!(products.units % SLOTS, 0);
    let request = products.units / SLOTS;
    let slot = u64::from(request) * UNIT_BYTES + rows.part_bytes;

    // A share that holds two slots and half a third holds two. The process holds every rank of
    // the group, whose fewest are those its ranks decide.
    let alone = |fewest: &[u32]| -> anyhow::Result<Vec<u32>> { Ok(fewest.to_vec()) };
    super::fit_media_slots(&mut ranks, |_| Ok(Some(2 * slot + slot / 2)), alone).unwrap();
    let open = layout(&launch, &ranks).unwrap();
    assert_eq!(region(&open, "slots 1 0").units, 2);
    assert_eq!(region(&open, "products 1 0").units, 2 * request);
    let video = ranks[0].video().unwrap();
    let pages = video.trajectory.unwrap().2;
    assert_eq!(pages.pages, 2 * u64::from(pages.trajectory_pages));

    // The group serves as many requests as its slots hold and encodes at most that many prompts
    // in a batch, over the buckets of the deployment's running requests.
    let stated = capacities(&ranks, &open, (&[], Vec::new())).unwrap();
    let capacity = &stated[0].1;
    assert_eq!(capacity.slots, 2);
    assert_eq!(capacity.rows[Phase::TextEncode.index()], 2);
    let encodes = capacity
        .buckets
        .iter()
        .filter(|shape| shape.variant == Variant::Encode)
        .map(|shape| shape.rows);
    assert_eq!(encodes.max(), Some(SLOTS));

    // A share one byte short of a slot refuses the start.
    let mut ranks = load();
    let refusal = super::fit_media_slots(&mut ranks, |_| Ok(Some(slot - 1)), alone).unwrap_err();
    let message = format!("{refusal:#}");
    assert!(
        message.contains(&format!("one slot needs {slot} bytes")),
        "{message}"
    );
}

/// A block's ranks are any positions of its worker group, in the order the block states them:
/// the denoiser's sequence group of a group whose transformer block holds ranks `[3, 1, 2, 0]`
/// has the group's fourth rank as its rank 0, and each rank joins it at its place in the block;
/// the video decoder's ranks `[2, 0]` form the decoder ranks' communicator the same way, which
/// the ranks outside the block do not join.
#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn a_parallel_group_takes_its_block_s_ranks_in_their_stated_order() {
    let mut launch = video_launch();
    let mut group = launch.engine_config.parallel.worker_groups[0].clone();
    group.ranks = (0..4)
        .map(|device| WorkerRank {
            host: "localhost".to_owned(),
            device: proto::config::RankDevice::cuda(device as u32),
        })
        .collect();
    let transformer = group.blocks.get_mut("transformer").unwrap();
    transformer.ranks = vec![3, 1, 2, 0];
    transformer.sequence_parallel = proto::config::SequenceParallel::Ulysses {
        ulysses_degree: 4,
        projection: Default::default(),
        projection_rows: None,
    };
    group.blocks.get_mut("video_decoder").unwrap().ranks = vec![2, 0];
    launch.engine_config.parallel.worker_groups = vec![group];
    launch.devices = vec![0, 1, 2, 3];
    launch.world_size = 4;

    let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
    let family = crate::process::catalogue::resolve(&registry, &fast_h3()).unwrap();
    let runners = ModelRunner::of(&launch, family).unwrap().ranks().unwrap();
    assert_eq!(runners.len(), 4);
    for runner in &runners {
        let collectives = runner.media_collectives().unwrap();
        let sequence = formed(&collectives, "transformer", Some(nn::family::ULYSSES_AXIS));
        let decoder = formed(&collectives, "video_decoder", None);
        assert_eq!(sequence.groups, [vec![3, 1, 2, 0]]);
        let place = [3, 1, 2, 0]
            .iter()
            .position(|rank| *rank == runner.rank() as usize);
        assert_eq!(sequence.member, Some((0, place.unwrap())));

        assert_eq!(decoder.groups, [vec![2, 0]]);
        let place = [2, 0]
            .iter()
            .position(|rank| *rank == runner.rank() as usize);
        assert_eq!(decoder.member, place.map(|place| (0, place)));
    }
}

/// A denoiser block of a tensor split two by Ulysses two over the group's ranks `[3, 1, 2, 0]`
/// forms its Ulysses groups of its consecutive places and its tensor groups of the places a
/// Ulysses degree apart, each listing its members in the order of their coordinate; a hybrid
/// of Ulysses two by all-gather two forms its all-gather groups the same way, and neither forms
/// a group of an axis of one rank.
#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn a_denoiser_block_forms_a_group_along_each_axis_of_its_partition() {
    let groups_of = |tensor: usize, sequence: proto::config::SequenceParallel| {
        let mut launch = video_launch();
        let mut group = launch.engine_config.parallel.worker_groups[0].clone();
        group.ranks = (0..4)
            .map(|device| WorkerRank {
                host: "localhost".to_owned(),
                device: proto::config::RankDevice::cuda(device as u32),
            })
            .collect();
        let transformer = group.blocks.get_mut("transformer").unwrap();
        transformer.ranks = vec![3, 1, 2, 0];
        transformer.tensor_parallel_size = tensor;
        transformer.sequence_parallel = sequence;
        launch.engine_config.parallel.worker_groups = vec![group];
        launch.devices = vec![0, 1, 2, 3];
        launch.world_size = 4;

        let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
        let family = crate::process::catalogue::resolve(&registry, &fast_h3()).unwrap();
        let runners = ModelRunner::of(&launch, family).unwrap().ranks().unwrap();
        runners
            .iter()
            .map(|runner| (runner.rank(), runner.media_collectives().unwrap()))
            .collect::<Vec<_>>()
    };

    let mixed = groups_of(
        2,
        proto::config::SequenceParallel::Ulysses {
            ulysses_degree: 2,
            projection: Default::default(),
            projection_rows: None,
        },
    );
    for (rank, collectives) in &mixed {
        let [ulysses, gather, tensor, layers] =
            [0, 1, 2, 3].map(|axis| formed(collectives, "transformer", Some(axis)));
        assert_eq!(ulysses.groups, [vec![3, 1], vec![2, 0]]);
        assert_eq!(tensor.groups, [vec![3, 2], vec![1, 0]]);
        assert!(gather.groups.is_empty() && layers.groups.is_empty());

        // Group rank 0 is the block's place 3: the second of its Ulysses pair and of its tensor
        // pair.
        if *rank == 0 {
            assert_eq!(
                (ulysses.member, tensor.member),
                (Some((1, 1)), Some((1, 1)))
            );
        }
    }

    let hybrid = groups_of(
        1,
        proto::config::SequenceParallel::Hybrid {
            ulysses_degree: 2,
            allgather_degree: 2,
        },
    );
    for (_, collectives) in &hybrid {
        let [ulysses, gather, tensor] =
            [0, 1, 2].map(|axis| formed(collectives, "transformer", Some(axis)));
        assert_eq!(ulysses.groups, [vec![3, 1], vec![2, 0]]);
        assert_eq!(gather.groups, [vec![3, 2], vec![1, 0]]);
        assert!(tensor.groups.is_empty());
    }
}

/// A denoising rank holds the condition encoders whenever the deployment places them, whichever
/// ranks their block names: in a group whose Ulysses-2 denoiser spans both ranks and whose latent
/// encoder names rank 1 alone, both denoising ranks profile the keyframe openings their latent
/// preparation encodes, and size the condition products a keyframe request reads.
#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn every_denoising_rank_encodes_the_conditions_of_its_requests() {
    let mut launch = video_launch();
    let mut group = launch.engine_config.parallel.worker_groups[0].clone();
    group.ranks = (0..2)
        .map(|device| WorkerRank {
            host: "localhost".to_owned(),
            device: proto::config::RankDevice::cuda(device as u32),
        })
        .collect();
    let transformer = group.blocks.get_mut("transformer").unwrap();
    transformer.ranks = vec![1, 0];
    transformer.sequence_parallel = proto::config::SequenceParallel::Ulysses {
        ulysses_degree: 2,
        projection: Default::default(),
        projection_rows: None,
    };
    group.blocks.get_mut("latent_encoder").unwrap().ranks = vec![1];
    launch.engine_config.parallel.worker_groups = vec![group];
    launch.devices = vec![0, 1];
    launch.world_size = 2;

    let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
    let family = crate::process::catalogue::resolve(&registry, &fast_h3()).unwrap();
    let model_runner = ModelRunner::of(&launch, family).unwrap();
    let ranks = load_ranks(model_runner.ranks().unwrap(), META, &launch, Meeting::Local).unwrap();

    let pixels = pipeline().kind(Product::ConditionPixels).unwrap();
    for rank in &ranks {
        let video = rank.video().unwrap();
        let largest = video
            .capacity
            .media_bytes
            .iter()
            .map(|(_, _, bytes)| bytes[pixels])
            .max();
        assert!(
            largest > Some(0),
            "rank {} sizes no condition pixels",
            rank.model_runner.rank()
        );
    }
}

/// The launch of [`fast_h3`] over two devices of one process, each the one rank of a group: the
/// `text` group encodes prompts on device 0, and the `flow` group serves every other phase on
/// device 1.
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
        copies: 1,
        ranks: vec![WorkerRank {
            host: "localhost".to_owned(),
            device: device.parse().unwrap(),
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

/// A group that places the latent encoder alone names no phase, since the denoising ranks'
/// latent preparation computes its work: its rank opens and its worker serves nothing, beside
/// the groups that serve every phase.
#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn a_group_of_no_phase_spawns_a_worker_that_serves_nothing() {
    let server = StorageServer::start();
    let mut launch = split_launch();
    let mut encoder = launch.engine_config.parallel.worker_groups[0].clone();
    encoder.id = WorkerGroupId("latent_encoder".to_owned());
    encoder.ranks[0].device = proto::config::RankDevice::cuda(2);
    encoder.blocks =
        BTreeMap::from([("latent_encoder".to_owned(), BlockConfig::parallel(vec![0]))]);
    launch.engine_config.parallel.worker_groups.push(encoder);
    launch.devices = vec![0, 1, 2];
    launch.world_size = 3;

    let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
    let family = crate::process::catalogue::resolve(&registry, &fast_h3()).unwrap();
    let model_runner = ModelRunner::of(&launch, family).unwrap();
    let mut ranks =
        load_ranks(model_runner.ranks().unwrap(), META, &launch, Meeting::Local).unwrap();
    assert_eq!(ranks[2].video().unwrap().capacity.phases, 0);

    let open = layout(&launch, &ranks).unwrap();
    let mut client =
        StorageClient::connect_head(&server.socket, DEPLOYMENT, launch.epoch, launch.engine)
            .unwrap();
    let pairs = client.open(open.clone()).unwrap();
    let opened = client.session().unwrap().1.clone();
    open_ranks(&mut ranks, &open, &opened, None).unwrap();
    let (_, spawned) = spawn(
        &mut ranks,
        (&open, &opened),
        &|group| client.completed_registers(group, &pairs),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(spawned.len(), 3);
    drop(spawned);
    client.close().unwrap();
}

#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn the_conditioning_lies_with_the_text_encoder_and_the_denoising_group_reads_it() {
    let server = StorageServer::start();
    let launch = split_launch();

    let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
    let family = crate::process::catalogue::resolve(&registry, &fast_h3()).unwrap();
    let model_runner = ModelRunner::of(&launch, family).unwrap();
    let mut ranks =
        load_ranks(model_runner.ranks().unwrap(), META, &launch, Meeting::Local).unwrap();
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
    for (kind, spec) in pipeline().products.iter().enumerate() {
        let name = spec.product.name();
        let (texts, flows) = match spec.product {
            Product::Conditioning => (vec![text_buffer], vec![text_buffer]),
            Product::VisionPixels => (vec![flow_buffer], vec![flow_buffer]),
            _ => (Vec::new(), vec![flow_buffer]),
        };
        assert_eq!(text.regions[kind], texts, "{name} on the text group");
        assert_eq!(flow.regions[kind], flows, "{name} on the flow group");
    }

    // Both ranks open, each reaching the other's buffer, and a worker serves each group.
    let mut client =
        StorageClient::connect_head(&server.socket, DEPLOYMENT, launch.epoch, launch.engine)
            .unwrap();
    let mut pairs = client.open(open.clone()).unwrap();
    let opened = client.session().unwrap().1.clone();
    assert!(
        open_ranks(&mut ranks, &open, &opened, None)
            .unwrap()
            .is_empty()
    );
    let (workers, spawned) = spawn(
        &mut ranks,
        (&open, &opened),
        &|group| client.completed_registers(group, &pairs),
        Vec::new(),
    )
    .unwrap();
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
        succeeded(workers.map(group, slot, RequestId(number), attach, attached));
        slots.push(slot);
    }
    let conditioning = pipeline().kind(Product::Conditioning).unwrap();
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
        let mapped = workers.map(
            group,
            slot,
            RequestId(3),
            allocate.clone(),
            allocated.clone(),
        );
        succeeded(mapped);
    }

    drop(workers);
    drop(spawned);
    client.close().unwrap();
}

/// The launch of [`fast_h3`] as `testkit::layouts::fasth3_dp8_text_tp8` places it on one host of
/// eight devices: the `text` group's eight ranks encode prompts eight-way tensor
/// parallel, and eight one-rank `flow` replicas serve every other phase, replica `i` on the
/// device of text rank `i`.
fn dp8_launch() -> EngineArgs {
    let mut launch = video_launch();
    launch.engine_config.parallel.worker_groups =
        testkit::layouts::fasth3_dp8_text_tp8("localhost");
    launch.devices = (0..8).collect();
    launch.world_size = 8;
    launch
}

#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn a_device_holds_a_rank_of_the_text_group_and_one_of_a_flow_replica() {
    let server = StorageServer::start();
    let launch = dp8_launch();
    assert_eq!(launch.engine_config.parallel.worker_groups.len(), 9);

    // Sixteen ranks, two on each device: text rank `i` and the rank of flow replica `i`.
    let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
    let family = crate::process::catalogue::resolve(&registry, &fast_h3()).unwrap();
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
    let mut ranks = load_ranks(placed, META, &launch, Meeting::Local).unwrap();
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
    let conditioning = pipeline().kind(Product::Conditioning).unwrap();
    let text_buffer = region("products 1 0");
    for group in &open.groups {
        assert_eq!(group.regions[conditioning], vec![text_buffer]);
    }

    // The session opens, every rank opens its own memory, two of them on each device, and an
    // worker serves each rank.
    let mut client =
        StorageClient::connect_head(&server.socket, DEPLOYMENT, launch.epoch, launch.engine)
            .unwrap();
    let mut pairs = client.open(open.clone()).unwrap();
    let opened = client.session().unwrap().1.clone();
    assert!(
        open_ranks(&mut ranks, &open, &opened, None)
            .unwrap()
            .is_empty()
    );
    let (workers, spawned) = spawn(
        &mut ranks,
        (&open, &opened),
        &|group| client.completed_registers(group, &pairs),
        Vec::new(),
    )
    .unwrap();
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
        succeeded(workers.map(group, slot, RequestId(number), attach, attached));
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
        let mapped = workers.map(
            group,
            slot,
            RequestId(3),
            allocate.clone(),
            allocated.clone(),
        );
        succeeded(mapped);
    }

    // What the groups state: the text group encodes prompts, and each replica every other phase.
    let stated = capacities(&ranks, &open, (&[], Vec::new())).unwrap();
    assert_eq!(stated.len(), 9);
    for (id, capacity) in &stated {
        let text = *id == GroupId(0);
        assert_eq!(capacity.serves(Phase::TextEncode), text, "group {}", id.0);
        assert_eq!(capacity.serves(Phase::Denoise), !text, "group {}", id.0);
    }

    drop(workers);
    drop(spawned);
    client.close().unwrap();
}

/// A two-host layout of `testkit::layouts`, whose leader is the host it is given.
type Layout = fn(&str) -> Vec<WorkerGroupConfig>;

/// The launch of `checkpoint` for the main process of host `host` of the two-host deployment
/// `layout`, whose hosts are `rank-0`, the scheduler node's, and `rank-1`, four devices each: the
/// process serves from the first rank its host holds, in a world of every rank of the
/// deployment.
fn two_host_launch(layout: Layout, host: &str, checkpoint: &Path) -> EngineArgs {
    let mut launch = video_launch();
    let groups = layout("rank-0");
    let ranks: Vec<&WorkerRank> = groups.iter().flat_map(|group| &group.ranks).collect();
    launch.world_size = u32::try_from(ranks.len()).unwrap();
    launch.rank = u32::try_from(ranks.iter().position(|rank| rank.host == host).unwrap()).unwrap();

    launch.engine_config.model.model = checkpoint.to_string_lossy().into_owned();
    launch.engine_config.parallel.engine_host = "rank-0".to_owned();
    launch.engine_config.parallel.worker_groups = groups;
    launch.host = host.to_owned();
    launch.devices = vec![0, 1, 2, 3];
    launch
}

/// The launch and the ranks of the scheduler node's process (host `rank-0`) of the two-host
/// deployment `file` of `checkpoint`, loaded onto the meta device while the process of host
/// `rank-1` loads its own: the two meet at one rendezvous as the deployment's host group, and
/// agree their groups' request slots over it.
fn two_host_ranks(layout: Layout, checkpoint: &Path) -> (EngineArgs, Vec<LoadedRank>) {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap().to_string();
    let load = |host: &str, listener: Option<std::net::TcpListener>| {
        let launch = two_host_launch(layout, host, checkpoint);
        let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
        let family = crate::process::catalogue::resolve(&registry, checkpoint).unwrap();
        let model_runner = ModelRunner::of(&launch, family).unwrap();
        let meeting = Meeting::World(crate::parallel::Rendezvous {
            listener,
            endpoint: address.clone(),
            epoch: launch.epoch,
            timeout: launch.engine_config.worker.join_timeout,
        });
        let ranks = load_ranks(model_runner.ranks().unwrap(), META, &launch, meeting);
        (launch, ranks.unwrap())
    };

    std::thread::scope(|scope| {
        let peer = scope.spawn(|| load("rank-1", None));
        let head = load("rank-0", Some(listener));
        peer.join().unwrap();
        head
    })
}

/// `testkit::layouts::fasth3_ulysses4x2_two_host` places each replica's text encoder and denoiser
/// on one host. The scheduler node's process holds the ranks of its own host alone, and
/// its session states every group of both hosts: every rank's product buffer and every group's
/// slot rows, each denoising replica reading the conditioning of the text encoder on its own
/// host and each text encoder the vision pixels of the media reader on its own, and the groups
/// of each host that host's memory. No product passes between the two processes.
#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn a_two_host_media_session_states_both_hosts_replicas_when_none_passes_a_product_across() {
    let (launch, ranks) = two_host_ranks(testkit::layouts::fasth3_ulysses4x2_two_host, &fast_h3());

    // Text replica 0 is ranks 0-3 and flow replica 0 ranks 8-11, on `rank-0`; replicas 1 are
    // ranks 4-7 and 12-15, on `rank-1`.
    let held: Vec<u32> = ranks.iter().map(|rank| rank.model_runner.rank()).collect();
    assert_eq!(held, [0, 1, 2, 3, 8, 9, 10, 11]);
    for rank in &ranks {
        assert_eq!(rank.video().unwrap().group_slots, vec![SLOTS; 4]);
    }

    let open = layout(&launch, &ranks).unwrap();
    assert_eq!(
        open.schedulers,
        vec![(0..4).map(GroupId).collect::<Vec<_>>()]
    );
    let region = |name: &str| {
        let index = open.regions.iter().position(|region| region.name == name);
        u8::try_from(index.unwrap_or_else(|| panic!("no region {name}"))).unwrap()
    };
    let buffers = |ranks: std::ops::Range<usize>| -> Vec<u8> {
        ranks
            .map(|rank| region(&format!("products 1 {rank}")))
            .collect()
    };
    for group in 0..4 {
        let slots = &open.regions[usize::from(region(&format!("slots 1 {group}")))];
        assert_eq!(slots.units, SLOTS, "group {group}");
    }

    // Groups text-0, text-1, flow-0 and flow-1, in that order.
    let conditioning = pipeline().kind(Product::Conditioning).unwrap();
    let vision = pipeline().kind(Product::VisionPixels).unwrap();
    let groups = &open.groups;
    assert_eq!(groups[2].regions[conditioning], buffers(0..1));
    assert_eq!(groups[3].regions[conditioning], buffers(4..5));
    assert_eq!(groups[0].regions[vision], buffers(8..12));
    assert_eq!(groups[1].regions[vision], buffers(12..16));
    assert_eq!(groups[0].memory, groups[2].memory);
    assert_eq!(groups[1].memory, groups[3].memory);
    assert_ne!(groups[0].memory, groups[1].memory);
    for group in groups {
        assert!(group.receives.iter().all(Vec::is_empty), "{}", group.id.0);
    }
}

/// The base MiniMax-H3 release, whose dense denoiser any Ulysses degree splits: the sparse
/// attention of the FastH3 exports composes rows in blocks that an eight-way split of the test's
/// one timeline does not leave whole, and refuses it in the profile run.
fn minimax_h3() -> PathBuf {
    testkit::assets::MINIMAX_H3.snapshot()
}

/// The layout the scheduler node states for a Ulysses8 two-host layout `placed`, which places the
/// `model` group's ranks 0-3 on `rank-0` and 4-7 on `rank-1`, beside a muxer (rank 8) and a media
/// reader (rank 9) on `rank-0`, each its own group of one scheduler. `wide` states whether the
/// text encoder holds all eight ranks (`testkit::layouts::ulysses8_two_host`) or the first four
/// (`testkit::layouts::ulysses8_reference_two_host`).
///
/// Nothing is refused, and every product a group reads in the other host's process is received
/// there (`p4-design.md` §1.1): the conditioning rank 0 wrote, the condition media the reader
/// wrote and, where the text encoder spans both hosts, the vision pixels, each in the product
/// buffer of rank 4, the group's first rank on `rank-1`, which holds a slot's blocks of them too;
/// the products every rank of the group backs need no crossing, nor does what the muxer reads on
/// its own host. Each product buffer lies in its host's memory. The conditioning's host copies
/// lie in part 0 on rank 0's host, and a saved trajectory's in part 0 on each host.
fn a_ulysses8_two_host_layout_receives_what_crosses(placed: Layout, wide: bool) {
    let (launch, ranks) = two_host_ranks(placed, &minimax_h3());
    let held: Vec<u32> = ranks.iter().map(|rank| rank.model_runner.rank()).collect();
    assert_eq!(held, [0, 1, 2, 3, 8, 9]);

    let open = layout(&launch, &ranks).unwrap();
    assert_eq!(
        open.schedulers,
        vec![(0..3).map(GroupId).collect::<Vec<_>>()]
    );
    let region = |rank: usize| {
        let name = format!("products 1 {rank}");
        let index = open.regions.iter().position(|region| region.name == name);
        u8::try_from(index.unwrap_or_else(|| panic!("no region {name}"))).unwrap()
    };
    let buffers = |ranks: std::ops::Range<usize>| -> Vec<u8> { ranks.map(region).collect() };

    // Each product buffer in the memory of its host's process.
    let devices = |host: &str| -> String {
        (0..4)
            .map(|device| format!("{host}/cuda:{device}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let (first, second) = (devices("rank-0"), devices("rank-1"));
    for rank in 0..10 {
        let memory = if (4..8).contains(&rank) {
            &second
        } else {
            &first
        };
        assert_eq!(
            &open.regions[usize::from(region(rank))].memory,
            memory,
            "rank {rank}"
        );
    }
    let (model, muxer, reader) = (&open.groups[0], &open.groups[1], &open.groups[2]);
    for group in [model, muxer, reader] {
        assert_eq!(group.memory, first, "group {}", group.id.0);
    }

    // Where the model group finds each product, and what rank 4 receives.
    let at = |product: Product| pipeline().kind(product).unwrap();
    let crossing = vec![region(4)];
    let vision = if wide { crossing.clone() } else { Vec::new() };
    for (name, lies, received) in [
        (Product::ConditionPixels, vec![region(9)], crossing.clone()),
        (Product::ConditionSamples, vec![region(9)], crossing.clone()),
        (Product::VisionPixels, vec![region(9)], vision),
        (Product::Conditioning, vec![region(0)], crossing.clone()),
        (Product::VideoLatents, buffers(0..8), Vec::new()),
        (Product::AudioLatents, buffers(0..8), Vec::new()),
        (Product::VideoUnits, buffers(0..8), Vec::new()),
        (Product::VideoTail, buffers(0..8), Vec::new()),
        (Product::EncodedUnits, buffers(0..8), Vec::new()),
        (Product::AudioSamples, buffers(0..8), Vec::new()),
    ] {
        let label = name.name();
        assert_eq!(model.regions[at(name)], lies, "{label} on the model group");
        assert_eq!(
            model.receives[at(name)],
            received,
            "{label} on the model group"
        );
    }

    // The muxer reads the encoded units and the track on its own host; the reader writes the
    // condition media in its rank's buffer.
    for (index, spec) in pipeline().products.iter().enumerate() {
        let name = spec.product.name();
        let muxed = [Product::EncodedUnits, Product::AudioSamples].contains(&spec.product);
        let read = [
            Product::ConditionPixels,
            Product::ConditionSamples,
            Product::VisionPixels,
        ]
        .contains(&spec.product);
        let muxes = if muxed { buffers(0..4) } else { Vec::new() };
        let reads = if read { vec![region(9)] } else { Vec::new() };
        assert_eq!(muxer.regions[index], muxes, "{name} on the muxer");
        assert_eq!(reader.regions[index], reads, "{name} on the reader");
        assert!(muxer.receives[index].is_empty() && reader.receives[index].is_empty());
    }

    // Rank 4 holds a slot's blocks of what it receives beside what it backs.
    let units = |rank: usize| open.regions[usize::from(region(rank))].units;
    assert!(units(4) > units(5), "{} and {}", units(4), units(5));

    // Host parts: the conditioning's where rank 0 stores it, a trajectory's on both hosts.
    let trajectory = usize::from(open.trajectory);
    let part = |host: &str| (host.to_owned(), 0);
    assert_eq!(model.parts[at(Product::Conditioning)], vec![part("rank-0")]);
    assert_eq!(
        model.parts[trajectory],
        vec![part("rank-0"), part("rank-1")]
    );
    for group in [muxer, reader] {
        assert!(
            group.parts.iter().all(Vec::is_empty),
            "group {}",
            group.id.0
        );
    }
}

/// `testkit::layouts::ulysses8_two_host`: the text encoder over all eight ranks.
#[test]
#[ignore = "reads the headers of the released MiniMax-H3 checkpoint, which the repository does not hold"]
fn the_ulysses8_two_host_layout_receives_what_crosses_the_hosts() {
    a_ulysses8_two_host_layout_receives_what_crosses(testkit::layouts::ulysses8_two_host, true);
}

/// `testkit::layouts::ulysses8_reference_two_host`: the text encoder over ranks 0-3, on
/// `rank-0`, whose `rank-1` ranks load no part of it.
#[test]
#[ignore = "reads the headers of the released MiniMax-H3 checkpoint, which the repository does not hold"]
fn the_ulysses8_reference_two_host_layout_receives_what_crosses_the_hosts() {
    a_ulysses8_two_host_layout_receives_what_crosses(
        testkit::layouts::ulysses8_reference_two_host,
        false,
    );
}

/// The released checkpoint deliverable E1 serves (`artifacts/buffer-pool/e1/protocol.md`).
fn qwen3_30b() -> PathBuf {
    testkit::assets::QWEN3_30B_A3B.snapshot()
}

/// The bytes of one GB200, whose quota at a fraction of 0.70 is the 138528423936 bytes E1's
/// deployment is refused or admitted against, and the granule its exportable allocations take.
const GB200_BYTES: u64 = 197_897_748_480;
const GB200_GRANULE: u64 = 2 << 20;

/// What the Storage Server measured of E1's deployment on that GPU as it admitted the rank
/// (`artifacts/buffer-pool/pf1/measure/failed/branch-416dd3b7d-serve.log`, its reservations): the
/// checkpoint's weights as the server retains them, granule-packed, and the warm process beside
/// its pools and the server's storages, measured after its warm-up, its request slots' storages
/// and its cache's tables included.
const RETAINED_WEIGHTS: u64 = 61_066_969_088;
const WARM_PROCESS: u64 = 8_604_614_656;

/// E1's launch: one rank of one GPU serving [`qwen3_30b`] as the protocol's command states it
/// (`--max-running-requests 128 --mem-fraction-static 0.70 --set
/// kv_cache_config.max_tokens=131072`), with the checkpoint's 40960-token context and the
/// daemon's defaults otherwise: 20 log-probability candidates and the default graph tiers.
fn e1_launch() -> EngineArgs {
    let mut launch = launch();
    let config = &mut launch.engine_config;

    config.model.model = qwen3_30b().to_string_lossy().into_owned();
    config.model.max_model_len = 40_960;
    config.max_running_requests = std::num::NonZeroUsize::new(128).unwrap();
    config.mem_fraction_static = 0.70;
    config.max_logprobs = 20;
    config.kv_cache.max_tokens = std::num::NonZeroU64::new(131_072);
    config.cuda_graph = Default::default();
    launch
}

/// The bytes of the tensors of the checkpoint at `directory`, as its shards' headers state them.
fn checkpoint_bytes(directory: &Path) -> u64 {
    let checkpoint = nn::Checkpoint::open(directory).unwrap();
    checkpoint
        .names()
        .map(|name| {
            let (first, end) = checkpoint.locate(name).unwrap().bytes;
            end - first
        })
        .sum()
}

/// E1's text rank started on the meta device from the released checkpoint's headers: its profile
/// run sizes its pools, and its division of a GB200's quota at 0.70 holds every reservation the
/// Storage Server admits it with: the weights, the cache's 131072 tokens in whole segments, the
/// graph pool, the scratch pool of the shadow region and the scratch page set, and the warm
/// process with its 128 request slots. Its start creates the stated capacity whole before its
/// warm-up.
#[test]
#[ignore = "reads the headers of the released Qwen3-30B-A3B checkpoint, which the repository does not hold"]
fn the_e1_deployment_starts_within_its_gpu_s_quota_at_a_fraction_of_0_70() {
    let launch = e1_launch();
    let ranks = loaded(&launch);
    assert_eq!(ranks.len(), 1);
    let rank = &ranks[0];
    let text = rank.text().unwrap();
    let layout = std::sync::Arc::clone(service::lock(&text.cache).layout());

    // The server retains the checkpoint's tensors, packed into whole granules.
    let checkpoint = checkpoint_bytes(&qwen3_30b());
    assert!(
        checkpoint <= RETAINED_WEIGHTS && RETAINED_WEIGHTS - checkpoint < 64 * GB200_GRANULE,
        "the checkpoint's {checkpoint} bytes against the {RETAINED_WEIGHTS} retained"
    );

    // The catalogue's largest bucket is 128 rows of 16384 tokens, and the scratch page set of the
    // rank's paired forwards is that bucket's pages, `ceil(16384 / 64) + 128`.
    assert_eq!(rank.model_runner.buckets().bounds(), (128, 16_384));
    assert_eq!(text.config.scratch_pages, 256 + 128);

    // The pools as the device maps them: the graph pool the runner's region, and the scratch pool
    // the shadow region beside the scratch page set, each in whole granules.
    let peaks = text.peaks;
    let scratch_set = crate::process::ranks::rank::memory::scratch_bytes(
        &text.config,
        service::lock(&text.cache).config(),
        GB200_GRANULE,
    );
    let shadow = peaks.shadow;
    let (graph, scratch) = crate::memory::division::pool_bytes(
        GB200_GRANULE,
        &[peaks.graph],
        &[peaks.shadow + scratch_set],
        layout.state_bytes(),
    );

    // The shadow region holds one paired call: the most one launch of a profiled forward carves,
    // beside the scratch page set's planes one layer's cache write writes in place, the key and
    // value planes of 384 pages of 64 tokens of four heads of 128 BF16 elements and their flags.
    let cache_writes =
        crate::runner::causal_lm::CausalLmRunner::cache_writes(&text.config, &layout);
    assert_eq!(cache_writes, 2 * 384 * 65_536 + 2 * 512);
    assert!(shadow > cache_writes);

    // The division of what the quota leaves once the weights and the warm process are held: the
    // deployment's 128 request slots, and the 2048 units of 131072 tokens in whole segments.
    let limit = crate::memory::client::quota_limit(GB200_BYTES, 0.70);
    assert_eq!(limit, 138_528_423_936);
    let unreserved = limit - RETAINED_WEIGHTS - WARM_PROCESS;
    let host = crate::parallel::HostGroup::local(1).unwrap().remove(0);
    let (slots, (units, first), states) = rank
        .model_runner
        .divide_unreserved(
            (unreserved, unreserved),
            GB200_GRANULE,
            &host,
            text.model.as_ref(),
            &layout,
            &[("graph pool", graph), ("scratch pool", scratch)],
        )
        .unwrap();
    assert_eq!(slots, 128);
    assert!(units >= 2048, "the cache holds {units} units");
    // A model that keeps no recurrent state has no state region.
    assert_eq!(states, (0, 0));

    // The start creates the stated capacity whole before its warm-up: the quota holds it beside
    // the warm process the reservations below count.
    let (_, segment) = crate::memory::cache::segments::segment_layout(
        units,
        layout.segment_plane(),
        GB200_GRANULE,
    )
    .unwrap();
    assert_eq!(first, units);

    // A segment is one granule of each column's allocation: 32 units of the 64 KiB pages of four
    // heads of 128 BF16 elements, so a region holds what the quota holds to within 32 units.
    assert_eq!(u64::from(segment) * layout.plane_bytes(), GB200_GRANULE);

    // Every reservation the Storage Server admits the rank with, and what the quota leaves. The
    // warm process holds the request slots' storages, which are counted a second time here, as
    // the division counts them beside the process it measures before its warm-up.
    let slot_storage = rank
        .model_runner
        .slot_storage(text.model.as_ref(), &layout, slots)
        .unwrap();
    let planes = u64::from(units) * layout.unit_bytes();
    let reserved = RETAINED_WEIGHTS + planes + graph + scratch + WARM_PROCESS;
    eprintln!(
        "E1 at 0.70 of a GB200: limit {limit}; weights {RETAINED_WEIGHTS}; cache {units} units, \
         {planes} bytes; graph pool {graph}; scratch pool {scratch} (shadow region {shadow}, \
         scratch page set {scratch_set}); request slots {slots}, {slot_storage} bytes (in the warm \
         process); warm process {WARM_PROCESS}; reserved {reserved}; left {}",
        limit - reserved
    );
    assert!(reserved + slot_storage <= limit);
}

/// The head of `launch`'s epoch on `server`, its session open over the layout of `ranks`, and
/// that layout with the session's answer.
fn opened_session(
    server: &StorageServer,
    launch: &EngineArgs,
    ranks: &[LoadedRank],
) -> (
    StorageClient,
    Vec<QueuePair>,
    storage::runtime::Open,
    storage::runtime::Opened,
) {
    let open = stated(launch, ranks);
    let mut head =
        StorageClient::connect_head(&server.socket, DEPLOYMENT, launch.epoch, launch.engine)
            .unwrap();
    let pairs = head.open(open).unwrap();
    let (held, opened) = {
        let (held, opened) = head.session().unwrap();
        (held.clone(), opened.clone())
    };
    (head, pairs, held, opened)
}

/// The name of the one KV region of the session `open` states.
fn kv_region(open: &storage::runtime::Open) -> &str {
    let kv = usize::from(open.groups[0].regions[0][0]);
    &open.regions[kv].name
}

/// Data-parallel replicas that each run an engine of their own (`daemon::leader::replica_configs`,
/// each replica's configuration placing its own worker group on its own device) share no device
/// memory: each names a KV region of its own, though each numbers its one scheduler 0, so two
/// replicas starting at once each create and grant their own segments.
#[test]
fn replicas_in_engines_of_their_own_name_regions_of_their_own() {
    let server = StorageServer::start();
    let groups = WorkerGroupConfig::data_parallel(
        &["localhost".to_owned()],
        proto::config::DeviceKind::Cuda,
        2,
        1,
        2,
        WorkerGroupConfig::single_block(proto::config::DEFAULT_BLOCK, 1),
    );
    let replica = |index: usize| {
        let mut launch = launch();
        launch.engine_config.parallel.worker_groups = vec![groups[index].clone()];
        launch.group = groups[index].id.0.clone();
        launch.devices = vec![index as u32];
        // The daemon numbers each replica's engine, and its epoch, apart.
        launch.engine = 1 + index as u64;
        launch.epoch = 1 + index as u64;
        launch
    };
    let launches = [replica(0), replica(1)];

    // Both replicas open their sessions before either grants, as two engines started together
    // do: neither finds the other's segments, and each creates its own.
    let mut ranks: Vec<Vec<LoadedRank>> = launches.iter().map(loaded).collect();
    let mut sessions = Vec::new();
    for (launch, ranks) in launches.iter().zip(&ranks) {
        sessions.push(opened_session(&server, launch, ranks));
    }
    let names: Vec<String> = sessions
        .iter()
        .map(|(_, _, open, _)| kv_region(open).to_owned())
        .collect();
    assert_ne!(names[0], names[1], "the replicas share a KV region");

    for (ranks, (head, _, open, opened)) in ranks.iter_mut().zip(&mut sessions) {
        let kv = usize::from(open.groups[0].regions[0][0]);
        assert!(
            opened.segments[kv].is_empty(),
            "a replica maps another's segments"
        );
        let created = open_ranks(ranks, open, opened, None).unwrap();
        assert_eq!(created.len(), 1);
        let (region, segments) = &created[0];
        assert!(!segments.is_empty());
        for &(segment, first, count) in segments {
            head.grant(region, segment, first, count).unwrap();
        }
    }
    for (mut head, ..) in sessions {
        head.close().unwrap();
    }
}

/// A target of another layout, here KV pages of another block size, which the daemon drains to with
/// `shrink` (`specs/buffer-pool.md` §14, "A change of layout"), names a KV region of its own and
/// maps none of the serving epoch's segments, while a successor of the same layout names the
/// serving epoch's region and maps them.
#[test]
fn a_target_of_another_layout_maps_none_of_the_serving_epochs_segments() {
    let server = StorageServer::start();
    let serving = launch();
    let mut ranks = loaded(&serving);
    let (mut head, _pairs, open, opened) = opened_session(&server, &serving, &ranks);
    let created = open_ranks(&mut ranks, &open, &opened, None).unwrap();
    let (region, segments) = &created[0];
    for &(segment, first, count) in segments {
        head.grant(region, segment, first, count).unwrap();
    }

    let mut same = serving.clone();
    same.epoch = 2;
    let mut other = serving.clone();
    other.epoch = 3;
    other.engine_config.kv_cache.block_size = std::num::NonZeroU32::new(32);
    assert_eq!(serving.engine_config.layout(), same.engine_config.layout());
    assert_ne!(serving.engine_config.layout(), other.engine_config.layout());

    let same_ranks = loaded(&same);
    let (mut same_head, _, same_open, same_opened) = opened_session(&server, &same, &same_ranks);
    assert_eq!(kv_region(&same_open), kv_region(&open));
    let kv = usize::from(same_open.groups[0].regions[0][0]);
    assert_eq!(same_opened.segments[kv], *segments);
    same_head.close().unwrap();

    let mut other_ranks = loaded(&other);
    let (mut other_head, _, other_open, other_opened) =
        opened_session(&server, &other, &other_ranks);
    assert_ne!(kv_region(&other_open), kv_region(&open));
    let kv = usize::from(other_open.groups[0].regions[0][0]);
    assert!(other_opened.segments[kv].is_empty());
    let other_created = open_ranks(&mut other_ranks, &other_open, &other_opened, None).unwrap();
    let (other_region, other_segments) = &other_created[0];
    for &(segment, first, count) in other_segments {
        other_head
            .grant(other_region, segment, first, count)
            .unwrap();
    }
    other_head.close().unwrap();
    head.close().unwrap();
}

/// A deployment whose experts live on a dedicated expert group (`role = "experts"`, as
/// `tests/python/e2e/http_helpers.py` places one) opens one KV region, its attention group's:
/// the expert group serves no request, the session states it nowhere, and its rank opens a cache
/// of the fewest units of its own, maps no region and creates nothing the head grants, beside
/// the attention rank's region, whose segments the head grants. Both ranks are members of the
/// expert union, whose exchange keys state the multiprocessors the start states of their devices.
#[test]
fn a_dedicated_expert_rank_maps_no_kv_region() {
    let server = StorageServer::start();
    let mut launch = launch();
    let rank = |device: &str| WorkerRank {
        host: "localhost".to_owned(),
        device: device.parse().unwrap(),
    };
    let attention = WorkerGroupConfig {
        id: WorkerGroupId("attention".to_owned()),
        role: Default::default(),
        copies: 1,
        ranks: vec![rank("cuda:0")],
        blocks: WorkerGroupConfig::single_block(proto::config::DEFAULT_BLOCK, 1),
        queue_depth: 2,
        mem_fraction_static: None,
    };
    let experts = WorkerGroupConfig {
        id: WorkerGroupId("experts".to_owned()),
        role: proto::config::WorkerGroupRole::Experts,
        copies: 1,
        ranks: vec![rank("cuda:1")],
        blocks: BTreeMap::new(),
        queue_depth: 2,
        mem_fraction_static: None,
    };
    launch.engine_config.parallel.worker_groups = vec![attention, experts];
    launch.engine_config.moe.expert_parallel = true;
    launch.group = "attention".to_owned();
    launch.devices = vec![0, 1];
    launch.world_size = 2;

    let mut ranks = loaded(&launch);
    assert_eq!(ranks.len(), 2);

    // Both members' profile runs exchange, and every exchange key they record states the
    // multiprocessors the start states of their devices.
    let stated = format!("_multiprocessors_{MULTIPROCESSORS}_");
    for rank in &ranks {
        let exchanges: Vec<&String> = rank
            .keys
            .iter()
            .filter(|(operator, _)| *operator == proto::module::operator::EXPERT_EXCHANGE)
            .map(|(_, key)| key)
            .collect();
        assert!(
            !exchanges.is_empty(),
            "rank {} records no exchange",
            rank.model_runner.rank()
        );
        assert!(
            exchanges.iter().all(|key| key.contains(&stated)),
            "{exchanges:?}"
        );
    }

    let (mut head, pairs, open, opened) = opened_session(&server, &launch, &ranks);
    assert_eq!(
        open.groups.len(),
        1,
        "the session states the attention group alone"
    );

    let created = open_ranks(&mut ranks, &open, &opened, None).unwrap();
    assert_eq!(created.len(), 1);
    let (region, segments) = &created[0];
    assert_eq!(region, kv_region(&open));
    for &(segment, first, count) in segments {
        head.grant(region, segment, first, count).unwrap();
    }
    let expert_cache = service::lock(&ranks[1].text().unwrap().cache)
        .config()
        .clone();
    assert!(expert_cache.segments.is_none());
    assert_eq!(
        expert_cache.units,
        crate::process::ranks::rank::memory::fewest_units(&expert_cache.layout)
    );

    bind(&mut ranks, &open, &opened, true).unwrap();
    let (workers, spawned) = spawn(
        &mut ranks,
        (&open, &opened),
        &|group| head.completed_registers(group, &pairs),
        Vec::new(),
    )
    .unwrap();
    assert_eq!(spawned.len(), 2);
    drop(workers);
    drop(spawned);
    head.close().unwrap();
}

/// A text deployment whose hosts hold its ranks interleaved, as the daemon launches one whose two
/// tensor-parallel groups each span both hosts (hosts ordered by their first rank, each host's
/// process launched from it): each host's process numbers its ranks by their places in the
/// deployment, each on the device that places it and at its place in its group's tensor group,
/// so the two hosts hold every rank of the deployment once.
#[test]
fn a_host_numbers_its_text_ranks_by_their_places_in_the_deployment() {
    let placed = |host: &str, device: &str| WorkerRank {
        host: host.to_owned(),
        device: device.parse().unwrap(),
    };
    let group = |id: &str, ranks: Vec<WorkerRank>| WorkerGroupConfig {
        id: WorkerGroupId(id.to_owned()),
        role: Default::default(),
        copies: 1,
        blocks: WorkerGroupConfig::single_block(proto::config::DEFAULT_BLOCK, ranks.len()),
        ranks,
        queue_depth: 2,
        mem_fraction_static: None,
    };
    // Ranks 0 and 1 are group `a`'s, on each host's device 0; ranks 2 and 3 group `b`'s, on each
    // host's device 1.
    let groups = vec![
        group("a", vec![placed("h0", "cuda:0"), placed("h1", "cuda:0")]),
        group("b", vec![placed("h0", "cuda:1"), placed("h1", "cuda:1")]),
    ];

    let mut numbered = Vec::new();
    for (host, first) in [("h0", 0), ("h1", 1)] {
        let mut launch = launch();
        launch.engine_config.parallel.worker_groups = groups.clone();
        launch.engine_config.parallel.tensor_parallel_size =
            std::num::NonZeroUsize::new(2).unwrap();
        launch.engine_config.parallel.data_parallel_size = std::num::NonZeroUsize::new(2).unwrap();
        launch.host = host.to_owned();
        launch.rank = first;
        launch.devices = vec![0, 1];
        launch.world_size = 4;

        let registry = nn::family::Registry::new([models::qwen3::family::FAMILY]).unwrap();
        let family = crate::process::catalogue::resolve(
            &registry,
            Path::new(&launch.engine_config.model.model),
        )
        .unwrap();
        let ranks = ModelRunner::of(&launch, family).unwrap().ranks().unwrap();

        let stated: Vec<(u32, String, u32, i32)> = ranks
            .iter()
            .map(|rank| {
                (
                    rank.rank(),
                    rank.group().to_owned(),
                    rank.part(),
                    rank.devices()[0].0,
                )
            })
            .collect();
        let position = u32::from(host == "h1");
        assert_eq!(
            stated,
            vec![
                (first, "a".to_owned(), position, 0),
                (first + 2, "b".to_owned(), position, 1),
            ],
            "the ranks host {host} holds"
        );
        numbered.extend(stated.into_iter().map(|(rank, ..)| rank));
    }
    numbered.sort_unstable();
    assert_eq!(numbered, vec![0, 1, 2, 3]);
}

/// A model that keeps a recurrent state copies it between its request slots and the units of its
/// state region with kernels whose keys carry the region's units (the units' rows are the
/// checkpoint planes times the units), which the load's profile run, over the fewest units, does
/// not name. Once the session numbers the state region, the rank's opening asks the Build Server,
/// for the rank's device, for the kernels its state launches over that numbering that its
/// generation lacks, and the rank's kernels name them. A dense model's cache launches no such
/// kernel and asks for none.
#[test]
fn a_rank_opening_resolves_the_kernels_of_its_cache_over_its_region_s_units() {
    let fixtures = [
        (
            models::qwen3::family::HYBRID_FAMILY,
            models::fixtures::qwen3_5_moe(),
            true,
        ),
        (
            models::qwen3::family::FAMILY,
            models::fixtures::qwen3_moe(),
            false,
        ),
    ];
    for (family, fixture, keeps_state) in fixtures {
        let server = StorageServer::start();
        let mut launch = launch();
        launch.engine_config.model.model = fixture.to_string_lossy().into_owned();
        let registry = nn::family::Registry::new([family]).unwrap();
        let family = crate::process::catalogue::resolve(
            &registry,
            Path::new(&launch.engine_config.model.model),
        )
        .unwrap();
        let model_runner = ModelRunner::of(&launch, family).unwrap();
        let mut ranks =
            load_ranks(model_runner.ranks().unwrap(), META, &launch, Meeting::Local).unwrap();

        // The regions numbered for more units than the fewest, in whole segments, as a division
        // of a device's quota numbers them.
        let (loaded, loaded_states) = {
            let text = ranks[0].text().unwrap();
            (text.units, text.states.0)
        };
        {
            let Loaded::Text(text) = &mut ranks[0].loaded else {
                panic!("a text rank");
            };
            text.units = 40 * loaded;
            text.first = text.units;
            text.states = (40 * loaded_states, 40 * loaded_states);
        }
        let (mut head, _pairs, open, opened) = opened_session(&server, &launch, &ranks);

        let asked: std::sync::Mutex<Vec<(platform::DeviceId, Keys)>> = Default::default();
        let resolve = |device: platform::DeviceId, keys: &Keys| -> anyhow::Result<Vec<_>> {
            asked.lock().unwrap().push((device, keys.clone()));
            Ok(Vec::new())
        };
        open_ranks(&mut ranks, &open, &opened, Some(&resolve)).unwrap();

        let text = ranks[0].text().unwrap();
        let cache = service::lock(&text.cache);
        let (units, states, layout) = (
            cache.config().units,
            cache.config().states.0,
            cache.layout(),
        );
        assert_eq!(units, 40 * loaded);
        assert_eq!(states, 40 * loaded_states);
        let shape = |units: u32| {
            format!(
                "shape_{}x{}_",
                u64::from(layout.checkpoint_planes()) * u64::from(units),
                layout.plane_bytes()
            )
        };
        let copies = |keys: &Keys, units: u32| -> Vec<&'static str> {
            keys.iter()
                .filter(|(_, key)| key.contains(&shape(units)))
                .map(|(operator, _)| *operator)
                .collect()
        };

        let asked = asked.into_inner().unwrap();
        if !keeps_state {
            assert!(asked.is_empty(), "a dense cache asks for {asked:?}");
            drop(cache);
            head.close().unwrap();
            continue;
        }
        assert_eq!(asked.len(), 1, "{asked:?}");
        let (device, keys) = &asked[0];
        assert_eq!(*device, ranks[0].model_runner.devices()[0]);
        assert_eq!(
            copies(keys, states),
            ["cache_block_scatter", "cache_block_snapshot"]
        );
        assert!(copies(keys, loaded_states).is_empty(), "{keys:?}");
        assert!(keys.is_subset(&ranks[0].keys));
        drop(cache);
        head.close().unwrap();
    }
}

/// A media group whose warm ranks keep fewer request slots than they warmed up with states them,
/// and the session's regions of the group serve them: its slot rows the slots kept, and the
/// product buffer of each of its ranks the parts of those slots, a slot's part being the buffer's
/// units over the slots it was opened with.
#[test]
#[ignore = "reads the headers of a released FastH3 checkpoint, which the repository does not hold"]
fn the_regions_of_a_media_group_serve_the_slots_it_kept() {
    let launch = video_launch();
    let registry = nn::family::Registry::new([models::minimax_h3::family::FAMILY]).unwrap();
    let family = crate::process::catalogue::resolve(&registry, &fast_h3()).unwrap();
    let model_runner = ModelRunner::of(&launch, family).unwrap();
    let ranks = load_ranks(model_runner.ranks().unwrap(), META, &launch, Meeting::Local).unwrap();
    let open = layout(&launch, &ranks).unwrap();
    let group = &open.groups[0];
    let slot_region = usize::from(group.slots);
    let products = open
        .regions
        .iter()
        .position(|region| region.name.starts_with("products "))
        .unwrap();
    assert_eq!(open.regions[slot_region].units, SLOTS);
    let per_slot = open.regions[products].units / SLOTS;

    // Every slot the group warmed up with is kept: no region changes.
    let mut stated = capacities(&ranks, &open, (&[], Vec::new())).unwrap();
    assert!(
        super::kept_regions(&launch, &ranks, &open, &stated)
            .unwrap()
            .is_empty()
    );

    // One slot kept.
    stated[0].1.slots = 1;
    let mut kept = super::kept_regions(&launch, &ranks, &open, &stated).unwrap();
    kept.sort_unstable();
    let mut expected = vec![(slot_region, 1), (products, per_slot)];
    expected.sort_unstable();
    assert_eq!(kept, expected);
}

/// The launch of one main process serving the Qwen3-MoE fixture (two decoder layers of four
/// query heads over two key/value heads) on one group of `devices` ranks listed in descending
/// device order, whose `model` block lists them in descending order too and states its degrees
/// as `degrees` sets them, as `tests/python/e2e/test_parallel_ranks_http.py` places its group.
fn parallel_launch(devices: u32, degrees: impl FnOnce(&mut BlockConfig)) -> EngineArgs {
    let mut launch = launch();
    let ranks: Vec<WorkerRank> = (0..devices)
        .rev()
        .map(|device| WorkerRank {
            host: "localhost".to_owned(),
            device: proto::config::RankDevice::cuda(device),
        })
        .collect();
    let mut block = BlockConfig::parallel((0..devices as usize).rev().collect());
    degrees(&mut block);
    let group = WorkerGroupConfig {
        id: WorkerGroupId("model".to_owned()),
        role: Default::default(),
        copies: 1,
        blocks: BTreeMap::from([(proto::config::DEFAULT_BLOCK.to_owned(), block)]),
        ranks,
        queue_depth: 2,
        mem_fraction_static: None,
    };

    launch.engine_config.parallel.worker_groups = vec![group.clone()];
    launch.group = group.id.0;
    launch.devices = (0..devices).collect();
    launch.world_size = devices;
    launch
}

/// A group of two pipeline stages and a group of two Ulysses members start on the meta device as
/// one group of the session (`specs/engine-worker-port.md` §7): each rank takes the position its
/// `model` block lists it at, so the group's second rank, on `cuda:0`, is position 0; a stage
/// holds one of the fixture's two layers with every key/value head, a member every layer with
/// one of the two heads; both ranks lay their part of a unit out alike, and the session states
/// one KV region whose unit is both parts, with a host part per position. The ranks' profile
/// runs, over the group's split, pass their rows over the group, and the group states what one
/// rank of it serves.
#[test]
fn a_pipeline_and_a_sequence_group_start_as_one_group_of_the_session() {
    let ulysses = |projection| proto::config::SequenceParallel::Ulysses {
        ulysses_degree: 2,
        projection,
        projection_rows: None,
    };
    let local = proto::config::SequenceParallel::Local;
    let layouts = [
        ("pipeline", 2, local, (1, 2)),
        (
            "gathered sequence",
            1,
            ulysses(proto::config::HeadProjection::Gather),
            (2, 1),
        ),
        (
            "exchanged sequence",
            1,
            ulysses(proto::config::HeadProjection::Exchange),
            (2, 1),
        ),
    ];

    for (axis, stages, sequence, held) in layouts {
        let server = StorageServer::start();
        let launch = parallel_launch(2, |block| {
            block.pipeline_parallel_size = stages;
            block.sequence_parallel = sequence;
        });
        let mut ranks = loaded(&launch);

        let placed: Vec<(u32, u32, i32)> = ranks
            .iter()
            .map(|rank| {
                let runner = &rank.model_runner;
                (runner.rank(), runner.part(), runner.devices()[0].0)
            })
            .collect();
        assert_eq!(placed, [(0, 1, 1), (1, 0, 0)], "{axis}");

        let shapes: Vec<(u32, u32)> = ranks
            .iter()
            .map(|rank| {
                // The layers and the key heads of the model's key field.
                let fields = rank.text().unwrap().model.cache();
                let key = fields
                    .iter()
                    .find(|field| field.name == nn::family::KEY)
                    .unwrap();
                (key.owners.len() as u32, key.record.heads)
            })
            .collect();
        assert_eq!(shapes, [held, held], "{axis}");

        let units: Vec<u64> = ranks
            .iter()
            .map(|rank| {
                service::lock(&rank.text().unwrap().cache)
                    .layout()
                    .unit_bytes()
            })
            .collect();
        assert_eq!(units[0], units[1], "{axis}");

        // One KV region of the group's units, each the two ranks' parts; a host part per
        // position, each on the host of the rank at that position.
        let open = stated(&launch, &ranks);
        assert_eq!(open.schedulers, vec![vec![GroupId(0)]], "{axis}");
        let group = &open.groups[0];
        let kv = usize::from(group.regions[0][0]);
        assert_eq!(
            (open.regions[kv].part_bytes, open.regions[kv].unit_bytes),
            (units[0], 2 * units[0]),
            "{axis}"
        );
        let localhost = |part| ("localhost".to_owned(), part);
        assert_eq!(group.parts[0], [localhost(0), localhost(1)], "{axis}");

        let mut head =
            StorageClient::connect_head(&server.socket, DEPLOYMENT, launch.epoch, launch.engine)
                .unwrap();
        let pairs = head.open(open.clone()).unwrap();
        let opened = head.session().unwrap().1.clone();
        let created = open_ranks(&mut ranks, &open, &opened, None).unwrap();
        for (region, segments) in &created {
            for &(segment, first, count) in segments {
                head.grant(region, segment, first, count).unwrap();
            }
        }
        bind(&mut ranks, &open, &opened, true).unwrap();
        let (workers, spawned) = spawn(
            &mut ranks,
            (&open, &opened),
            &|group| head.completed_registers(group, &pairs),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(spawned.len(), 2, "{axis}");

        // The group serves a text group's phases on every slot, over its one cache group.
        let stated_capacities = capacities(&ranks, &open, (&[], Vec::new())).unwrap();
        assert_eq!(stated_capacities.len(), 1, "{axis}");
        let (id, capacity) = &stated_capacities[0];
        assert_eq!(*id, GroupId(0));
        for phase in [Phase::Prefill, Phase::Decode] {
            assert!(capacity.serves(phase), "{axis} {phase:?}");
            assert_eq!(capacity.rows[phase.index()], SLOTS, "{axis} {phase:?}");
        }
        assert_eq!(capacity.cache_groups, vec![(CacheGroupKind::Full, 64)]);
        assert_eq!(capacity.seeds.len(), 1);

        drop(workers);
        drop(spawned);
        head.close().unwrap();
    }
}

/// A text group splits its model along one axis, a Ulysses sequence's every row of a projection at
/// once, a pipeline's stages on one host, and every serving group of a deployment alike: a block
/// of two degrees above one, a sequence split of another kind, an overlapped projection step, a
/// pipeline across hosts and two groups split otherwise are refused at start, each naming what it
/// states.
#[test]
fn a_text_group_splits_its_model_along_one_axis_and_every_group_alike() {
    let registry = nn::family::Registry::new([models::qwen3::family::FAMILY]).unwrap();
    let refused = |launch: &EngineArgs| -> String {
        let family = crate::process::catalogue::resolve(
            &registry,
            Path::new(&launch.engine_config.model.model),
        )
        .unwrap();
        let ranks = ModelRunner::of(launch, family).unwrap().ranks();
        format!("{:#}", ranks.err().expect("the deployment is refused"))
    };

    let two_axes = parallel_launch(4, |block| {
        block.tensor_parallel_size = 2;
        block.pipeline_parallel_size = 2;
    });
    assert!(refused(&two_axes).contains("one axis"));

    let gathered = parallel_launch(2, |block| {
        block.sequence_parallel = proto::config::SequenceParallel::Allgather {
            allgather_degree: 2,
        };
    });
    assert!(refused(&gathered).contains("Ulysses"));

    let stepped = parallel_launch(2, |block| {
        block.sequence_parallel = proto::config::SequenceParallel::Ulysses {
            ulysses_degree: 2,
            projection: proto::config::HeadProjection::Exchange,
            projection_rows: Some(64),
        };
    });
    assert!(refused(&stepped).contains("projection_rows"));

    // The stages of a pipeline meet at a barrier of one process.
    let mut spread = parallel_launch(2, |block| block.pipeline_parallel_size = 2);
    spread.engine_config.parallel.worker_groups[0].ranks[0].host = "elsewhere".to_owned();
    assert!(refused(&spread).contains("more than one host"));

    // A prefill group of two pipeline stages beside a decode group of two tensor ranks.
    let mut split = parallel_launch(4, |_| {});
    let group = |id: &str, role, devices: [&str; 2], block: BlockConfig| WorkerGroupConfig {
        id: WorkerGroupId(id.to_owned()),
        role,
        copies: 1,
        blocks: BTreeMap::from([(proto::config::DEFAULT_BLOCK.to_owned(), block)]),
        ranks: devices
            .iter()
            .map(|device| WorkerRank {
                host: "localhost".to_owned(),
                device: device.parse().unwrap(),
            })
            .collect(),
        queue_depth: 2,
        mem_fraction_static: None,
    };
    split.engine_config.parallel.worker_groups = vec![
        group(
            "prefill",
            proto::config::WorkerGroupRole::Model,
            ["cuda:0", "cuda:1"],
            BlockConfig {
                pipeline_parallel_size: 2,
                ..BlockConfig::parallel(vec![0, 1])
            },
        ),
        group(
            "decode",
            proto::config::WorkerGroupRole::Model,
            ["cuda:2", "cuda:3"],
            BlockConfig {
                tensor_parallel_size: 2,
                ..BlockConfig::parallel(vec![0, 1])
            },
        ),
    ];
    split.group = "prefill".to_owned();
    let refusal = refused(&split);
    assert!(
        refusal.contains("prefill") && refusal.contains("decode"),
        "{refusal}"
    );
}

/// The key of the one launch of `operator` the profile runs of `ranks` name, the same on every
/// rank.
fn rank_key(ranks: &[LoadedRank], operator: &str) -> String {
    let keys: std::collections::BTreeSet<&String> = ranks
        .iter()
        .flat_map(|rank| rank.keys.iter())
        .filter(|(launched, _)| *launched == operator)
        .map(|(_, key)| key)
        .collect();
    assert_eq!(keys.len(), 1, "{operator}: {keys:?}");
    keys.into_iter().next().unwrap().clone()
}

/// A deployment of one scheduler's group of two tensor-parallel ranks over the fixture, with
/// expert parallelism (`moe_config.expert_parallel`) when `expert_parallel` holds.
fn tensor_group(expert_parallel: bool) -> EngineArgs {
    let mut launch = parallel_launch(2, |block| block.tensor_parallel_size = 2);
    launch.engine_config.moe.expert_parallel = expert_parallel;
    launch
}

/// With expert parallelism a scheduler's group of two tensor-parallel ranks shards the fixture's
/// four experts over its ranks (`docs/expert_parallelism.md`): each rank's profile run names the
/// routed products of two whole experts at the full eight intermediate columns and the sort of a
/// window of two, where the same group without the flag holds every expert at half its width.
/// Neither runs an exchange: every rank holds every token.
#[test]
fn a_tensor_group_with_expert_parallelism_runs_whole_experts_over_every_token() {
    for expert_parallel in [true, false] {
        let ranks = loaded(&tensor_group(expert_parallel));
        assert_eq!(ranks.len(), 2);

        let (local, intermediate, experts) = if expert_parallel {
            (2, 8, 2)
        } else {
            (4, 4, 4)
        };
        assert!(
            rank_key(&ranks, "moe_sort").starts_with(&format!("experts_4_top_2_local_{local}_")),
            "expert parallelism {expert_parallel}"
        );
        for operator in ["gather_gemm", "route_gemm"] {
            assert!(
                rank_key(&ranks, operator).starts_with(&format!(
                    "hidden_16_intermediate_{intermediate}_experts_{experts}_top_2_"
                )),
                "{operator}, expert parallelism {expert_parallel}"
            );
        }
        assert!(
            ranks.iter().all(|rank| rank
                .keys
                .iter()
                .all(|(operator, _)| *operator != "expert_exchange")),
            "expert parallelism {expert_parallel}"
        );
    }
}

/// The placement each rank hands its model under expert parallelism: a group of two
/// tensor-parallel ranks holds whole experts of the group's slots, each rank at its tensor
/// position, and no expert union; two schedulers of one rank each form an expert union, each
/// keeping its scheduler's half of the experts, which their exchange reaches.
#[test]
fn expert_parallelism_places_whole_experts_on_a_tensor_group_and_shares_them_in_a_union() {
    let registry = nn::family::Registry::new([models::qwen3::family::FAMILY]).unwrap();
    let placements = |launch: &EngineArgs| -> Vec<nn::family::TextPlacement> {
        let family = crate::process::catalogue::resolve(
            &registry,
            Path::new(&launch.engine_config.model.model),
        )
        .unwrap();
        let ranks = ModelRunner::of(launch, family).unwrap().ranks().unwrap();
        ranks
            .iter()
            .map(|rank| rank.text_placement().unwrap())
            .collect()
    };
    let half = |rank| nn::Split::new(2, rank).unwrap();

    for expert_parallel in [true, false] {
        let tensor: Vec<(nn::Split, nn::Split, bool)> = placements(&tensor_group(expert_parallel))
            .iter()
            .map(|placement| {
                (
                    placement.tensor,
                    placement.experts,
                    placement.tensor_experts,
                )
            })
            .collect();
        // The group's ranks are listed in descending device order, so its first rank is
        // position one.
        assert_eq!(
            tensor,
            [
                (half(1), nn::Split::WHOLE, expert_parallel),
                (half(0), nn::Split::WHOLE, expert_parallel)
            ],
            "expert parallelism {expert_parallel}"
        );
    }

    let mut union = launch();
    let scheduler = |index: usize| WorkerGroupConfig {
        id: WorkerGroupId(format!("model-{index}")),
        role: Default::default(),
        copies: 1,
        ranks: vec![WorkerRank {
            host: "localhost".to_owned(),
            device: proto::config::RankDevice::cuda(index as u32),
        }],
        blocks: WorkerGroupConfig::single_block(proto::config::DEFAULT_BLOCK, 1),
        queue_depth: 2,
        mem_fraction_static: None,
    };
    union.engine_config.parallel.worker_groups = vec![scheduler(0), scheduler(1)];
    union.engine_config.parallel.data_parallel_size = std::num::NonZeroUsize::new(2).unwrap();
    union.engine_config.moe.expert_parallel = true;
    union.group = "model-0".to_owned();
    union.devices = vec![0, 1];
    union.world_size = 2;
    let shared: Vec<(nn::Split, nn::Split, bool)> = placements(&union)
        .iter()
        .map(|placement| {
            (
                placement.tensor,
                placement.experts,
                placement.tensor_experts,
            )
        })
        .collect();
    assert_eq!(
        shared,
        [
            (nn::Split::WHOLE, half(0), false),
            (nn::Split::WHOLE, half(1), false)
        ]
    );
}

/// Expert parallelism outside an expert union shards a group's experts over its tensor-parallel
/// ranks, so a group split over pipeline stages or sequence members with the flag is refused at
/// start rather than served with every expert on every rank.
#[test]
fn a_pipeline_or_sequence_group_with_expert_parallelism_is_refused() {
    let registry = nn::family::Registry::new([models::qwen3::family::FAMILY]).unwrap();
    let stages = |block: &mut BlockConfig| block.pipeline_parallel_size = 2;
    let members = |block: &mut BlockConfig| {
        block.sequence_parallel = proto::config::SequenceParallel::Ulysses {
            ulysses_degree: 2,
            projection: proto::config::HeadProjection::Gather,
            projection_rows: None,
        };
    };
    for mut launch in [parallel_launch(2, stages), parallel_launch(2, members)] {
        launch.engine_config.moe.expert_parallel = true;
        let family = crate::process::catalogue::resolve(
            &registry,
            Path::new(&launch.engine_config.model.model),
        )
        .unwrap();
        let ranks = ModelRunner::of(&launch, family).unwrap().ranks();
        let refusal = format!("{:#}", ranks.err().expect("the deployment is refused"));
        assert!(refusal.contains("expert_parallel"), "{refusal}");
    }
}
