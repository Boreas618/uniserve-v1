//! A text rank started on the meta device against a Storage Server in the test's process: the
//! rank's load through its profile run, the session the engine opens with the buffer pool over
//! the rank's layout, and the executor spawned over the rank's runner, which maps the session's
//! completions into its block tables.
//!
//! The model is the Qwen3-MoE fixture of the `qwen3` crate, loaded from its shard headers alone;
//! no GPU is touched. The server runs on a thread of the test process with a socket of its own.

use std::num::{NonZeroU32, NonZeroUsize};
use std::os::fd::{AsFd as _, AsRawFd as _, FromRawFd as _, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ipc::CacheGroupKind;
use ipc::batch::Phase;
use proto::config::{EngineArgs, WorkerGroupConfig};
use storage::QueuePair;
use storage::runtime::{Completion, GroupId, OwnerId, Request, RequestId};
use worker::completion::{Outcome, Submission};

use super::{Backing, Meeting, bind, capacities, layout, load_ranks, spawn};
use crate::rank::ModelRunner;
use crate::storage_client::StorageClient;

/// The request slots the rank holds.
const SLOTS: u32 = 4;

/// A Storage Server on a thread of the test process, stopped when dropped.
struct Server {
    socket: PathBuf,
    stop: Arc<AtomicBool>,
    wake: OwnedFd,
    thread: Option<std::thread::JoinHandle<()>>,
    _directory: tempfile::TempDir,
}

impl Server {
    fn start() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("storage.sock");
        let (listener, lock) = service::bind(&socket).unwrap().unwrap();

        let mut ends = [0; 2];
        // SAFETY: pipe2 fills the two descriptors it creates.
        let created = unsafe { libc::pipe2(ends.as_mut_ptr(), libc::O_NONBLOCK) };
        assert_eq!(created, 0);
        // SAFETY: two new descriptors this test owns.
        let (reader, writer) =
            unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };

        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let thread = std::thread::spawn(move || {
            let _lock = lock;
            let mut server = storage_server::Server::new(listener, 64, 0, None, None).unwrap();
            server
                .run(reader.as_fd(), &|| stopped.load(Ordering::Acquire))
                .unwrap();
        });

        Self {
            socket,
            stop,
            wake: writer,
            thread: Some(thread),
            _directory: directory,
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        // SAFETY: a write to a pipe this test owns.
        unsafe { libc::write(self.wake.as_raw_fd(), b"x".as_ptr().cast(), 1) };
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

/// The launch of epoch 1's one main process, serving the Qwen3-MoE fixture on one rank of one
/// group: decode buckets of one and two rows, prefill buckets of up to 32 tokens.
fn launch() -> EngineArgs {
    let checkpoint = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../models/qwen3/tests/data/qwen3_moe/model")
        .to_string_lossy()
        .into_owned();
    let group = WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        "cuda",
        1,
        2,
        WorkerGroupConfig::single_component(proto::config::DEFAULT_BLOCK, 1),
    );
    let package = |backend: &str| proto::module::PackageRevision {
        backend: backend.to_owned(),
        revision: 1,
        arguments: Default::default(),
    };
    let modules = proto::module::ModuleSet {
        engine: package("uniserve"),
        policy: package("fcfs"),
        kernels: Default::default(),
    };
    let sizes = |sizes: &[u32]| {
        Some(
            sizes
                .iter()
                .map(|size| NonZeroU32::new(*size).unwrap())
                .collect(),
        )
    };

    let config = proto::config::EngineConfig {
        executable: proto::BUILD.to_owned(),
        model: proto::config::ModelConfig {
            model: checkpoint,
            base: None,
            lora_path: None,
            max_model_len: 64,
            dtype: proto::request::ModelDtype::BFloat16,
            quantization: Default::default(),
            generation_limits: Default::default(),
            bos: 1,
            eos: vec![2],
        },
        parallel: proto::config::ParallelConfig {
            tensor_parallel_size: NonZeroUsize::MIN,
            data_parallel_size: NonZeroUsize::MIN,
            engine_host: "localhost".to_owned(),
            worker_groups: vec![group.clone()],
        },
        deployment: "model".to_owned(),
        max_running_requests: NonZeroUsize::new(SLOTS as usize).unwrap(),
        max_num_batched_tokens: None,
        max_logprobs: 2,
        attention_backend: String::new(),
        prefill_attention_backend: None,
        decode_attention_backend: None,
        moe_runner_backend: None,
        device: Default::default(),
        scheduler: Default::default(),
        media: Default::default(),
        mem_fraction_static: 0.9,
        load: Default::default(),
        kv_cache: Default::default(),
        moe: Default::default(),
        cuda_graph: proto::config::CudaGraphConfig {
            decode_batch_sizes: sizes(&[1, 2]),
            prefill_token_sizes: sizes(&[16, 32]),
            ..Default::default()
        },
        worker: Default::default(),
        speculative: Default::default(),
        video: Default::default(),
        profiling: Default::default(),
    };

    let mut launch = EngineArgs::new(config, modules);
    launch.epoch = 1;
    launch.host = "localhost".to_owned();
    launch.devices = vec![0];
    launch.group = group.id.0.clone();
    launch.world_size = 1;
    launch.queue_depth = 2;
    launch
}

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

#[test]
fn a_text_rank_starts_on_the_meta_device_and_its_executor_maps_the_session_s_completions() {
    let server = Server::start();
    let launch = launch();
    let registry = nn::family::Registry::new([qwen3::family::FAMILY]).unwrap();
    let family = crate::resolve(&registry, Path::new(&launch.engine_config.model.model)).unwrap();

    // The rank's start through its profile run.
    let model_runner = ModelRunner::of(&launch, family).unwrap();
    let mut ranks = load_ranks(
        model_runner.ranks().unwrap(),
        Backing::Meta,
        &launch,
        Meeting::Local,
    )
    .unwrap();
    assert_eq!(ranks.len(), 1);
    assert!(
        !ranks[0].keys.is_empty(),
        "the profile run names the kernels the rank launches"
    );

    // The session over the rank's layout: one scheduler of one group, whose KV region holds the
    // cache's units beside the padding unit and whose slot region its request slots.
    let open = layout(&ranks, &launch).unwrap();
    let mut client = StorageClient::connect(&server.socket, "model", launch.epoch).unwrap();
    let mut pairs = client.open(open.clone()).unwrap();
    let opened = client.session().unwrap().1.clone();
    assert_eq!(open.schedulers, vec![vec![GroupId(0)]]);
    assert_eq!(pairs.len(), 1);

    let group = &open.groups[0];
    let kv = usize::from(group.regions[0][0]);
    let units = worker::lock(&ranks[0].cache).config().units;
    assert_eq!(units, 2);
    assert_eq!(opened.regions[kv].1, units - 1);
    assert_eq!(opened.regions[usize::from(group.slots)].1, SLOTS);

    bind(&ranks, &open, &opened).unwrap();
    let (executors, spawned) = spawn(&mut ranks, &client, &pairs, Vec::new()).unwrap();
    assert_eq!(spawned.len(), 1);

    // An owner attached to the group and a block of class 0 allocated to it, both mapped into
    // the group's block tables, whose regions are the ones the session numbered. A meta rank's
    // cache holds the fewest units it serves with: the padding unit, which the region reserves,
    // and one page of its one cache group, unit 1.
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

    let allocate = Request::Allocate {
        owner: OwnerId(1),
        class: 0,
        at: 0,
        count: 1,
        units: 0,
    };
    let allocated = ask(pair, 2, &allocate);
    let Completion::Allocated { runs, .. } = &allocated else {
        panic!("a block is allocated: {allocated:?}");
    };
    assert_eq!(*runs, vec![(opened.regions[kv].0, 1, 1)]);
    succeeded(executors.map(GroupId(0), slot, RequestId(2), allocate, allocated));

    // What the group's ranks state of it: prefill, refill and decode on every slot, over its one
    // full-attention cache group of 64-token pages.
    let stated = capacities(&ranks, &open).unwrap();
    assert_eq!(stated.len(), 1);
    let (id, capacity) = &stated[0];
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

    // The executors stop before the session closes.
    drop(executors);
    drop(spawned);
    client.close().unwrap();
}
