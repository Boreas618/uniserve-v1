//! `scheduling-precompile`: a policy package's raw wasm32 module, in the
//! precompiled form the Program Server serves (§5.1, §5.2, §5.3).
//!
//! A policy package declares `[module] form = wasm32 precompiled`; the Program
//! Server builds the program with this binary, and the engine loads the result
//! with the same `wasmtime::Config` it precompiled under — a precompiled module
//! carries wasmtime's artifact format and configuration, so any other producer
//! makes a module the main process rejects.
//!
//! ```text
//! scheduling-precompile --wasm policy.wasm --out policy.module.wasm
//! scheduling-precompile --version
//! ```
//!
//! `--version` prints `scheduling-precompile <wasmtime version>`; the last
//! whitespace-separated token is the version a package's `[toolchain]` must
//! declare, so the Program Server reads it from the binary it will build with.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use scheduling::{Engine, WASMTIME_VERSION};

/// What the binary accepts.
const USAGE: &str = "\
usage: scheduling-precompile --wasm <raw.wasm> --out <module.wasm>
       scheduling-precompile --version";

/// Why the run ended.
enum Problem {
    /// The command line was not understood.
    Usage(String),
    /// Reading, compiling or writing failed.
    Failed(String),
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(Problem::Usage(message)) => {
            eprintln!("scheduling-precompile: {message}");
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
        Err(Problem::Failed(message)) => {
            eprintln!("scheduling-precompile: {message}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<(), Problem> {
    match Arguments::parse(std::env::args().skip(1))? {
        Arguments::Version => {
            println!("scheduling-precompile {WASMTIME_VERSION}");
            Ok(())
        }
        Arguments::Precompile { wasm, out } => precompile(&wasm, &out),
    }
}

/// The two things this binary does.
enum Arguments {
    /// Print the wasmtime version this binary precompiles with.
    Version,
    /// Precompile `wasm` into `out`.
    Precompile { wasm: PathBuf, out: PathBuf },
}

impl Arguments {
    /// Reads `arguments`, one flag and its value at a time.
    fn parse(arguments: impl Iterator<Item = String>) -> Result<Self, Problem> {
        let mut wasm = None;
        let mut out = None;
        let mut arguments = arguments;
        while let Some(flag) = arguments.next() {
            match flag.as_str() {
                "--version" => return Ok(Self::Version),
                "--wasm" => wasm = Some(value(&mut arguments, "--wasm")?),
                "--out" => out = Some(value(&mut arguments, "--out")?),
                other => return Err(Problem::Usage(format!("unknown argument {other}"))),
            }
        }
        match (wasm, out) {
            (Some(wasm), Some(out)) => Ok(Self::Precompile { wasm, out }),
            (None, _) => Err(Problem::Usage("--wasm is required".into())),
            (_, None) => Err(Problem::Usage("--out is required".into())),
        }
    }
}

/// The value of a flag.
fn value(arguments: &mut impl Iterator<Item = String>, flag: &str) -> Result<PathBuf, Problem> {
    arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| Problem::Usage(format!("{flag} needs a path")))
}

/// Compiles `wasm` with the engine's own configuration and writes the module to
/// `out`, creating the directory it is written into.
fn precompile(wasm: &Path, out: &Path) -> Result<(), Problem> {
    let bytes = fs::read(wasm)
        .map_err(|error| Problem::Failed(format!("cannot read {}: {error}", wasm.display())))?;
    let engine = Engine::new()
        .map_err(|error| Problem::Failed(format!("cannot start the engine: {error}")))?;
    let module = engine.precompile(&bytes).map_err(|error| {
        Problem::Failed(format!("cannot precompile {}: {error}", wasm.display()))
    })?;
    let parent = out.parent().filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        fs::create_dir_all(parent).map_err(|error| {
            Problem::Failed(format!("cannot create {}: {error}", parent.display()))
        })?;
    }
    fs::write(out, &module)
        .map_err(|error| Problem::Failed(format!("cannot write {}: {error}", out.display())))
}
