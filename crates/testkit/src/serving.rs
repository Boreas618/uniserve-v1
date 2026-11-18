//! What the engine's own tests serve from: a Storage Server on a thread of the test's process, the
//! launch of one main process serving the Qwen3-MoE fixture of the `models` crate, the kernel a
//! generation holds that no test launches, and the bound of a test's waits.
//!
//! The engine's unit tests and its integration tests share these, so they live in this
//! development crate rather than in the engine's test tree. None of them names an engine type:
//! the engine's unit tests link this crate beside their own build of the engine.

use std::num::{NonZeroU32, NonZeroUsize};
use std::os::fd::{AsFd as _, OwnedFd};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use proto::config::{EngineArgs, WorkerGroupConfig};

/// The bound of a test's wait that only a defect reaches: a passing test's waits end on the event
/// they wait for, long before it.
pub const LONG: Duration = Duration::from_secs(60);

/// A kernel a test's generation holds and never launches: the C library's `getpid`, which takes
/// no argument and touches no state, stands in for a package's launch entry, so a runner's
/// generation, its graphs' bindings and their comparisons hold entries without a device or a
/// Build Server.
///
/// # Panics
///
/// When the process's C library cannot be opened by its soname.
#[expect(clippy::unwrap_used, reason = "test support fails the test that asks")]
pub fn unlaunched_kernel() -> Arc<ops::Kernel> {
    let library = platform::NativeLibrary::open("libc.so.6").unwrap();
    Arc::new(ops::Kernel::load_native(library, "getpid").unwrap())
}

/// The deployment [`launch`] serves, and the one a test's sessions are of.
pub const DEPLOYMENT: &str = "model";

/// The request slots [`launch`]'s one rank holds.
pub const SLOTS: u32 = 4;

/// A Storage Server on a thread of the test's process, serving on a socket of its own directory
/// and publishing its readiness record beside it as a host's server does; stopped when dropped.
pub struct StorageServer {
    /// The socket the server serves on.
    pub socket: PathBuf,
    stop: Arc<AtomicBool>,
    /// The write end of the pipe whose readability wakes the server's loop.
    wake: OwnedFd,
    thread: Option<std::thread::JoinHandle<()>>,
    _directory: tempfile::TempDir,
}

impl StorageServer {
    /// Starts the server, which leaves none of the host's available storage to other processes
    /// and runs neither the Mooncake connector nor the Transfer Server. Its socket is bound and
    /// its record published before this returns, so a client connects at once.
    ///
    /// # Panics
    ///
    /// When the directory, the socket, the record or the wake pipe cannot be made.
    #[expect(
        clippy::unwrap_used,
        reason = "test support fails the test that starts it"
    )]
    pub fn start() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("storage.sock");
        let (listener, starting) = service::host::bind::<storage::Settings>(&socket)
            .unwrap()
            .owned(&socket)
            .unwrap();
        let lock = starting
            .finish(&service::Readiness::of_process(
                service::State::Serving,
                None,
                service::Endpoints::default(),
                storage::Settings::default(),
            ))
            .unwrap();
        let (reader, wake) = rustix::pipe::pipe_with(
            rustix::pipe::PipeFlags::NONBLOCK | rustix::pipe::PipeFlags::CLOEXEC,
        )
        .unwrap();

        let stop = Arc::new(AtomicBool::new(false));
        let stopped = Arc::clone(&stop);
        let thread = std::thread::spawn(move || {
            // The socket's lock is held for as long as the server serves.
            let _lock = lock;
            // A server of no children that leaves the host's storage to no other process.
            let settings = ::storage::Settings {
                server: ::storage::ServerSettings {
                    host_reserve_gib: 0,
                    ..::storage::ServerSettings::default()
                },
                connector: None,
            };
            let mut server = storage_server::Server::new(listener, settings, None).unwrap();
            server
                .run(reader.as_fd(), &|| stopped.load(Ordering::Acquire))
                .unwrap();
        });

        Self {
            socket,
            stop,
            wake,
            thread: Some(thread),
            _directory: directory,
        }
    }
}

impl Drop for StorageServer {
    #[expect(
        clippy::unwrap_used,
        reason = "test support fails the test that stops it"
    )]
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        rustix::io::write(&self.wake, b"x").unwrap();

        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

/// A revision of the workspace's default policy package (`crate::policy::package`) whose arguments
/// are `stated` (TOML) over the package's declared defaults, as the Build Server states a
/// revision's arguments to the daemon that launches it.
///
/// # Panics
///
/// When the package's manifest cannot be read, or `stated` is not a TOML table.
#[expect(clippy::unwrap_used, reason = "test support fails the test that asks")]
pub fn policy_revision(stated: &str) -> proto::module::PackageRevision {
    let manifest = build_server::manifest::Manifest::load(&crate::policy::package(
        proto::config::DEFAULT_SCHEDULING,
    ))
    .unwrap();
    let mut arguments: toml::Table = manifest.parameters.into_iter().collect();
    arguments.extend(toml::from_str::<toml::Table>(stated).unwrap());
    proto::module::PackageRevision {
        backend: manifest.backend,
        revision: 1,
        arguments,
    }
}

/// The launch of epoch 1's one main process, engine 1, serving the Qwen3-MoE fixture
/// (`models::fixtures::qwen3_moe`) on one rank of one group of [`SLOTS`] request slots, on the
/// device the configuration defaults to: decode buckets of one and two rows, prefill buckets of
/// up to 32 tokens, and the `arrival` policy package, whose limits the deployment pins to what
/// those buckets serve.
///
/// # Panics
///
/// Never: the policy's arguments, the slots and the bucket sizes are positive constants.
#[expect(
    clippy::unwrap_used,
    reason = "constant arguments, slots and bucket sizes"
)]
pub fn launch() -> EngineArgs {
    let checkpoint = models::fixtures::qwen3_moe().to_string_lossy().into_owned();
    let group = WorkerGroupConfig::placed(
        &["localhost".to_owned()],
        proto::config::DeviceKind::Cuda,
        1,
        2,
        WorkerGroupConfig::single_block(proto::config::DEFAULT_BLOCK, 1),
    );
    let package = |backend: &str, arguments: &str| proto::module::PackageRevision {
        backend: backend.to_owned(),
        revision: 1,
        arguments: toml::from_str(arguments).unwrap(),
    };
    let modules = proto::module::ModuleSet {
        engine: package("uniserve", ""),
        policy: policy_revision(""),
        kernels: Default::default(),
        programs: Default::default(),
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
        max_running_requests: NonZeroUsize::new(SLOTS as usize).unwrap(),
        max_num_batched_tokens: None,
        max_logprobs: 2,
        attention_backend: None,
        prefill_attention_backend: None,
        decode_attention_backend: None,
        moe_runner_backend: None,
        device: Default::default(),
        // A prefill of up to 32 tokens and a batch of up to two rows: what the buckets capture.
        scheduler: proto::config::SchedulerConfig {
            chunked_prefill_size: NonZeroUsize::new(32),
            max_batch: NonZeroUsize::new(2).unwrap(),
            max_num_batched_tokens: 32,
            ..Default::default()
        },
        media: Default::default(),
        mem_fraction_static: 0.9,
        load: Default::default(),
        kv_cache: Default::default(),
        moe: Default::default(),
        cuda_graph: proto::config::CudaGraphConfig {
            decode_batch_sizes: sizes(&[1, 2]),
            prefill_token_sizes: sizes(&[16, 32]),
        },
        worker: Default::default(),
        speculative: Default::default(),
        video: Default::default(),
        profiling: Default::default(),
    };

    let mut launch = EngineArgs::new(config, modules);
    launch.deployment = DEPLOYMENT.to_owned();
    launch.epoch = 1;
    launch.engine = 1;
    launch.host = "localhost".to_owned();
    launch.devices = vec![0];
    launch.group = group.id.0.clone();
    launch.world_size = 1;
    launch.queue_depth = 2;
    launch
}
