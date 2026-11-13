//! `uniserve-engine`: a node's main process for one epoch, as its own
//! executable.
//!
//! The engine is a package: this crate with the crates it depends on, built by
//! the Build Server into this executable (`package.toml`). A node's daemon
//! resolves the revision its epoch names and starts it with one launch file
//! (`--launch`, [`proto::config::EngineArgs`]); the process serves until its
//! daemon commands it down, the epoch fails, or the daemon is gone
//! ([`follow_lifeline`]).
//!
//! With `--keys` the process runs the same start up to the profile run of
//! every device's runners, prints the kernel keys they recorded as the key
//! list `uniserve build-server build --keys` reads, and serves nothing.
//!
//! `uniserve-engine precompile --wasm <raw> --out <module>` turns a scheduling
//! policy package's wasm32 module into the precompiled artifact the Build
//! Server stores, under the wasmtime configuration the engine loads it with
//! (`engine::scheduler::policy`), and `uniserve-engine precompile --version`
//! prints the wasmtime version a policy package must declare as its last
//! word. The Build Server runs both (`--wasm` names this executable).
//!
//! This is the one place the engine names a concrete model: the families it
//! loads are registered in [`FAMILIES`], and the library dispatches by
//! capability.

use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context as _, Result};
use clap::{Parser, Subcommand};
use engine::scheduler::policy;
use service::logging;

// mimalloc is built with `no_thp` (workspace `Cargo.toml`): its arenas carry
// no transparent-huge-page advice, so khugepaged never collapses the engine's
// heap while the scheduler thread runs.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// The model families this executable loads. A model is added with its module
/// of `models` and one entry here, and its request-processing side in
/// `processing::FAMILIES`, which the daemon serves.
const FAMILIES: [nn::family::Family; 4] = [
    models::qwen3::family::FAMILY,
    models::qwen3::family::HYBRID_FAMILY,
    models::qwen4_exp::family::FAMILY,
    models::minimax_h3::family::FAMILY,
];

/// The engine's command line: a main process with `--launch`, or one of its
/// commands.
#[derive(Debug, Parser)]
#[command(
    name = "uniserve-engine",
    about = "A node's main process for one epoch of a UniServe deployment.",
    subcommand_negates_reqs = true,
    args_conflicts_with_subcommands = true
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
    /// The launch file the daemon wrote for this process
    /// (`proto::config::EngineArgs`).
    #[arg(long, value_name = "FILE", required = true)]
    launch: Option<PathBuf>,
    /// Print the kernel keys this launch's profile runs record
    /// (`operator<TAB>key` lines, sorted, each once) and exit, serving
    /// nothing.
    #[arg(long)]
    keys: bool,
    /// Log level of UniServe's own code, such as `info` or `debug`.
    #[arg(long)]
    log_level: Option<String>,
    /// Log level of the HTTP client libraries.
    #[arg(long)]
    log_level_http: Option<String>,
}

/// What the executable does besides serving an epoch.
#[derive(Debug, Subcommand)]
enum Command {
    /// Precompile a scheduling policy package's wasm32 module for this host's
    /// architecture, as the Build Server stores it.
    Precompile(Precompile),
}

/// The arguments of `precompile`.
#[derive(Debug, clap::Args)]
struct Precompile {
    /// Print the wasmtime version modules are precompiled with, the one a
    /// policy package declares, and exit.
    #[arg(long, exclusive = true)]
    version: bool,
    /// The raw wasm32 module the policy crate compiled to.
    #[arg(long, value_name = "FILE", required_unless_present = "version")]
    wasm: Option<PathBuf>,
    /// Where the precompiled module is written; its directory is created.
    #[arg(long, value_name = "FILE", required_unless_present = "version")]
    out: Option<PathBuf>,
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
    let lifeline = take_lifeline();
    let cli = Cli::parse();
    if let Some(Command::Precompile(arguments)) = cli.command {
        return precompile(&arguments).map(|()| ExitCode::SUCCESS);
    }
    let launch_file = cli.launch.context("--launch names the launch file")?;
    let registry = nn::family::Registry::new(FAMILIES)?;

    if cli.keys {
        // The key list is this command's output, so its logs go to standard
        // error.
        logging::init(
            "Engine",
            cli.log_level.as_deref(),
            cli.log_level_http.as_deref(),
            logging::Output::Stderr,
        );
        let launch: proto::config::EngineArgs = std::fs::read(&launch_file)
            .map_err(anyhow::Error::new)
            .and_then(|text| serde_json::from_slice(&text).map_err(anyhow::Error::new))
            .with_context(|| format!("reading the launch file {}", launch_file.display()))?;
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
        cli.log_level.as_deref(),
        cli.log_level_http.as_deref(),
        logging::Output::Stdout,
    );
    engine::run(&launch_file, &registry).map(|()| ExitCode::SUCCESS)
}

/// `precompile`: prints the wasmtime version, or precompiles `--wasm` into
/// `--out`. A failure names the file it concerns; the process then exits 1,
/// and with 2 for a command line clap refused.
fn precompile(arguments: &Precompile) -> Result<()> {
    if arguments.version {
        println!("wasmtime {}", policy::WASMTIME_VERSION);
        return Ok(());
    }
    let (Some(wasm), Some(out)) = (&arguments.wasm, &arguments.out) else {
        anyhow::bail!("precompile needs --wasm and --out");
    };

    let bytes = std::fs::read(wasm).with_context(|| format!("cannot read {}", wasm.display()))?;
    let module = policy::precompile(&bytes)
        .with_context(|| format!("cannot precompile {}", wasm.display()))?;

    let parent = out.parent().filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    std::fs::write(out, &module).with_context(|| format!("cannot write {}", out.display()))
}

/// The lifeline this process inherited from the daemon that supervises it
/// ([`proto::LIFELINE_ENV`]): the read end of a pipe whose write end the
/// daemon holds. The variable leaves the environment and the descriptor turns
/// close-on-exec, so no process this one starts takes either for its own.
/// Runs before any thread starts.
fn take_lifeline() -> Option<i32> {
    let descriptor = std::env::var(proto::LIFELINE_ENV)
        .ok()?
        .parse::<i32>()
        .ok()?;
    // SAFETY: no other thread exists yet to read the environment.
    unsafe { std::env::remove_var(proto::LIFELINE_ENV) };
    // SAFETY: fcntl takes integers only; the descriptor is this process's.
    unsafe { libc::fcntl(descriptor, libc::F_SETFD, libc::FD_CLOEXEC) };
    Some(descriptor)
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
