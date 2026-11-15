//! `uniserve-engine`: a node's main process for one epoch, as its own
//! executable.
//!
//! The engine is a package: this crate with the crates it depends on, built by
//! the Build Server into this executable (`package.toml`). A node's daemon
//! resolves the revision its epoch names and starts it with one launch
//! (`--launch`, an inherited descriptor of a sealed file holding its
//! [`proto::config::EngineArgs`]) and its lifeline (`--lifeline`);
//! the process serves until its daemon commands it down, the epoch fails, or
//! the daemon is gone ([`follow_lifeline`]). The process states the settings its
//! CUDA and NCCL libraries read at initialization itself ([`RUNTIME_SETTINGS`]).
//!
//! With `--keys` the process runs the same start up to the profile run of
//! every device's runners, prints the kernel keys they recorded as the key
//! list `uniserve build-server build --keys` reads, and serves nothing.
//!
//! The engine names no concrete model: the executable loads the families
//! `models` registers ([`models::FAMILIES`]), and the library dispatches by
//! capability.

use std::io::Write as _;
use std::process::ExitCode;

use anyhow::{Context as _, Result};
use clap::Parser;
use service::logging;

// mimalloc is built with `no_thp` (workspace `Cargo.toml`): its arenas carry
// no transparent-huge-page advice, so khugepaged never collapses the engine's
// heap while the scheduler thread runs.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// NVTX's static injection: the strong definition of the weak
/// `InitializeInjectionNvtx2_fnptr` the NVTX loader compiled into this
/// executable declares, which NVTX calls at the process's first NVTX call to
/// hand its ranges to the CUPTI the process loads. NVTX consults it after a
/// library named in its environment, so a profiler that injects its own
/// (Nsight Systems) keeps the ranges.
#[unsafe(no_mangle)]
#[used]
#[allow(non_upper_case_globals)]
static InitializeInjectionNvtx2_fnptr: unsafe extern "C" fn(
    *const std::ffi::c_void,
) -> std::ffi::c_int = platform::cupti::initialize_nvtx_injection;

/// The engine's command line: a main process with `--launch`.
#[derive(Debug, Parser)]
#[command(
    name = "uniserve-engine",
    about = "A node's main process for one epoch of a UniServe deployment."
)]
struct Cli {
    /// The descriptor of the launch the daemon passed (`proto::LAUNCH_OPTION`): a sealed file
    /// that holds this process's `proto::config::EngineArgs`.
    #[arg(long, value_name = "FD")]
    launch: i32,
    /// The descriptor of the lifeline the supervising daemon passed
    /// (`proto::LIFELINE_OPTION`): the read end of a pipe whose write end the
    /// daemon holds. The process ends once the pipe does.
    #[arg(long, value_name = "FD")]
    lifeline: Option<i32>,
    /// Print the kernel keys this launch's profile runs record
    /// (`operator<TAB>key` lines, sorted, each once) and exit, serving
    /// nothing.
    #[arg(long)]
    keys: bool,
    /// Log level of every target (tracing's names: `off`, `error`, `warn`, `info`, `debug`,
    /// `trace`), `info` when unset; the daemon passes its own.
    #[arg(long)]
    log_level: Option<tracing::level_filters::LevelFilter>,
}

fn main() -> Result<ExitCode> {
    // A supervised process's standard error is its supervisor's pipe, and the
    // supervisor that started it may be gone: after a hot restart the
    // successor adopts the process without its predecessor's pipe, so a panic
    // written only to standard error is invisible in the run's log. Standard
    // output is the log every process of the deployment shares, so a panic is
    // stated there as well.
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic| {
        println!("a process panicked: {panic}");
        default(panic);
    }));
    // Before any thread, CUDA or NCCL exists.
    runtime_settings();
    let cli = Cli::parse();
    let lifeline = cli.lifeline.map(take_lifeline);
    let launch = read_launch(cli.launch)?;
    let registry = nn::family::Registry::new(models::FAMILIES.iter().copied())?;

    if cli.keys {
        // The key list is this command's output, so its logs go to standard
        // error.
        logging::init(
            "Engine",
            &logging::Levels::new(cli.log_level),
            logging::Output::Stderr,
        );
        let recorded = engine::record(&launch, &registry)?;
        let list = build_server::build::key_list(&recorded)?;
        let mut stdout = std::io::stdout().lock();
        stdout
            .write_all(list.as_bytes())
            .and_then(|()| stdout.flush())
            .context("writing the key list")?;
        return Ok(ExitCode::SUCCESS);
    }

    follow_lifeline(lifeline);
    // A rank receives and creates a descriptor per part of its KV region's segments.
    service::raise_descriptor_limit().context("raising the descriptor limit")?;
    logging::init(
        "Engine",
        &logging::Levels::new(cli.log_level),
        logging::Output::Stdout,
    );
    engine::run(launch, &registry).map(|()| ExitCode::SUCCESS)
}

/// Settings this process's CUDA and NCCL libraries read from the process environment when they
/// initialize, stated by the process itself before either initializes.
///
/// - `CUDA_MODULE_LOADING=EAGER`: lazy module loading synchronizes the context when a kernel is
///   first used, which can deadlock against a collective another stream is waiting in for a peer.
/// - `NCCL_RUNTIME_CONNECT=0`: deferred peer handshakes would allocate connection buffers after the
///   process reserved its device storage.
/// - `NCCL_GRAPH_REGISTER=0`: a collective captured in a graph registers its buffers, which lie in
///   the rank's graph pool, with its peers, which map that storage themselves. A sleep, and a
///   standby that parks with its storage suspended, give the graph pool's storage back and map new
///   storage at the same addresses when they wake, which the peers' mappings do not follow: every
///   collective after the wake would move data through storage no rank reads. Unregistered, a
///   captured collective copies through NCCL's own buffers, which stay mapped.
const RUNTIME_SETTINGS: [(&str, &str); 3] = [
    ("CUDA_MODULE_LOADING", "EAGER"),
    ("NCCL_RUNTIME_CONNECT", "0"),
    ("NCCL_GRAPH_REGISTER", "0"),
];

/// States [`RUNTIME_SETTINGS`] in this process's environment. Runs first in `main`, before any
/// thread starts.
fn runtime_settings() {
    for (name, value) in RUNTIME_SETTINGS {
        // SAFETY: no other thread exists yet to read the environment.
        unsafe { std::env::set_var(name, value) };
    }
}

/// The launch the daemon passed at the inherited descriptor `descriptor`, read whole; the
/// descriptor closes once it is read.
///
/// # Errors
///
/// When the descriptor cannot be read or holds no launch.
fn read_launch(descriptor: i32) -> Result<proto::config::EngineArgs> {
    use std::io::Read as _;
    use std::os::fd::FromRawFd as _;

    // SAFETY: the daemon passed this descriptor for the launch, and nothing else in this process
    // owns it; the file takes it over and closes it.
    let mut file = std::fs::File::from(unsafe { std::os::fd::OwnedFd::from_raw_fd(descriptor) });
    let mut text = Vec::new();
    file.read_to_end(&mut text)
        .with_context(|| format!("reading the launch at descriptor {descriptor}"))?;
    serde_json::from_slice(&text)
        .with_context(|| format!("reading the launch at descriptor {descriptor}"))
}

/// Takes the lifeline at `descriptor`, which this process inherited from the daemon that
/// supervises it (`--lifeline`): the descriptor turns close-on-exec, so no process this one starts
/// takes it for its own.
fn take_lifeline(descriptor: i32) -> i32 {
    // SAFETY: fcntl takes integers only; the descriptor is this process's.
    unsafe { libc::fcntl(descriptor, libc::F_SETFD, libc::FD_CLOEXEC) };
    descriptor
}

/// Ends this process once its daemon is gone: the end of the lifeline's
/// pipe arrives once the daemon, and every successor a hot restart handed
/// its processes to, ended, however they ended.
fn follow_lifeline(lifeline: Option<i32>) {
    let Some(descriptor) = lifeline else {
        return;
    };
    let watching = std::thread::Builder::new()
        .name("lifeline".to_owned())
        .spawn(move || {
            let mut byte = [0_u8; 1];
            loop {
                // SAFETY: reads into a local byte from this process's
                // descriptor.
                let read = unsafe { libc::read(descriptor, byte.as_mut_ptr().cast(), 1) };
                if read == 0 {
                    eprintln!("the daemon that supervised this process is gone; exiting");
                    // SAFETY: ends the process at once, as a signal would.
                    unsafe { libc::_exit(1) };
                }
                if read < 0
                    && std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted
                {
                    return;
                }
            }
        });
    if let Err(error) = watching {
        eprintln!("the lifeline's watch did not start: {error}");
    }
}
