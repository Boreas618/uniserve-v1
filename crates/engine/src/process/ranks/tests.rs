//! A text rank started on the meta device against a Storage Server in the test's process, in the
//! order a session opens before its ranks map memory (`specs/buffer-pool.md` §14.1): the rank's
//! load to its division, the session the engine opens over the layout its units state, the
//! segments the rank creates once the answer shows the region new and the head grants, the
//! executor spawned over the rank's runner, which maps the session's completions into its block
//! tables, and a successor epoch of the same layout, whose rank maps those segments.
//!
//! The model is the Qwen3-MoE fixture of the `qwen3` crate, loaded from its shard headers alone;
//! no GPU is touched. The server runs on a thread of the test process with a socket of its own.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ipc::CacheGroupKind;
use ipc::batch::Phase;
use proto::config::EngineArgs;
use storage::QueuePair;
use storage::runtime::{Completion, GroupId, OwnerId, Request, RequestId};
use worker::completion::{Outcome, Submission};

use super::{
    Backing, LoadedRank, Meeting, bind, capacities, layout, load_ranks, open_ranks, spawn,
};
use crate::rank::ModelRunner;
use crate::storage_client::StorageClient;
use crate::support::{DEPLOYMENT, SLOTS, StorageServer, launch};

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
    let registry = nn::family::Registry::new([qwen3::family::FAMILY]).unwrap();
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
    let rank = &ranks[0];
    let cache = Arc::clone(worker::lock(&rank.cache).layout());
    let slot_bytes = worker::CausalLmRunner::slot_bytes(&rank.config, rank.model.as_ref()).unwrap();

    layout(launch, &cache, rank.units, 1, slot_bytes / u64::from(SLOTS)).unwrap()
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
    let units = ranks[0].units;
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
    let created = open_ranks(&mut ranks, &open, &opened.segments).unwrap();
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

    bind(&ranks, &open, &opened).unwrap();
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
    let identities =
        worker::lock(&ranks[0].cache).page_identities(&crate::data_plane::identity_of(&[]));
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

    let none = open_ranks(&mut successors, &next_open, &next_opened.segments).unwrap();
    assert_eq!(none, vec![(open.regions[kv].name.clone(), Vec::new())]);
    let mapped: Vec<(u64, u32, u32)> = worker::lock(&successors[0].cache)
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
