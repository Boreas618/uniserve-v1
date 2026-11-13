//! What the engine's tests start from, shared by the library's unit tests (`crate::support`) and
//! the integration tests (`mod support`): a Storage Server on a thread of the test process, and
//! the launch of one main process serving the Qwen3-MoE fixture of the `qwen3` crate.

use std::num::{NonZeroU32, NonZeroUsize};
use std::os::fd::{AsFd as _, AsRawFd as _, FromRawFd as _, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use proto::config::{EngineArgs, WorkerGroupConfig};

/// The deployment the launch serves, and the one a test's sessions are of.
pub(crate) const DEPLOYMENT: &str = "model";

/// The request slots the launch's one rank holds.
pub(crate) const SLOTS: u32 = 4;

/// A Storage Server on a thread of the test process, listening on a socket of its own directory;
/// stopped when dropped.
pub(crate) struct StorageServer {
    pub(crate) socket: PathBuf,
    stop: Arc<AtomicBool>,
    wake: OwnedFd,
    thread: Option<std::thread::JoinHandle<()>>,
    _directory: tempfile::TempDir,
}

impl StorageServer {
    pub(crate) fn start() -> Self {
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
            let mut server = storage_server::Server::new(listener, 0, None, None).unwrap();
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

impl Drop for StorageServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        // SAFETY: a write to a pipe this test owns.
        unsafe { libc::write(self.wake.as_raw_fd(), b"x".as_ptr().cast(), 1) };
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

/// The workspace the tests were built from.
pub(crate) fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The launch of epoch 1's one main process, engine 1, serving the Qwen3-MoE fixture on one rank of one
/// group of [`SLOTS`] request slots, on the device the configuration defaults to: decode buckets
/// of one and two rows, prefill buckets of up to 32 tokens, and the `fcfs` policy package, whose
/// limits the deployment pins to what those buckets serve.
pub(crate) fn launch() -> EngineArgs {
    let checkpoint = workspace()
        .join("crates/models/tests/it/qwen3/data/qwen3_moe/model")
        .to_string_lossy()
        .into_owned();
    let group = WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        "cuda",
        1,
        2,
        WorkerGroupConfig::single_component(proto::config::DEFAULT_BLOCK, 1),
    );
    let package = |backend: &str, arguments: &str| proto::module::PackageRevision {
        backend: backend.to_owned(),
        revision: 1,
        arguments: toml::from_str(arguments).unwrap(),
    };
    let modules = proto::module::ModuleSet {
        engine: package("uniserve", ""),
        // A prefill of up to 32 tokens and a batch of up to two rows: what the buckets capture.
        policy: package(
            "fcfs",
            "chunked_prefill_size = 32\nmax_batch = 2\nmax_num_batched_tokens = 32",
        ),
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
        deployment: DEPLOYMENT.to_owned(),
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
    launch.engine = 1;
    launch.host = "localhost".to_owned();
    launch.devices = vec![0];
    launch.group = group.id.0.clone();
    launch.world_size = 1;
    launch.queue_depth = 2;
    launch
}
