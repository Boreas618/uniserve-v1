//! `buffer-pool-verify`: the checks of the Rust buffer pool against its design model
//! (`specs/buffer-pool.md` §18-19, `verification/buffer-pool`), built with the crate's
//! `verification` feature.
//!
//! - `enumerate` (L1) enumerates every request sequence of the model's small configurations on
//!   the Rust buffer pool, checks the pool's invariants after every step, and writes each
//!   configuration's states and transitions in the model's terms for `agreement.sh` and
//!   `agreement_steps.sh`.
//! - `replay` replays the traces a Storage Server recorded (`storage_config.buffer_pool_trace`)
//!   on new buffer pools, checks every completion against the run's and the pool's invariants
//!   after every input, and writes the states each passes through for `replay.sh`.
//!
//! Each prints one line per configuration or trace and exits non-zero at the first failure.

use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::Parser as _;
use storage_server::buffer_pool::{enumeration, replay};

#[derive(Debug, clap::Parser)]
#[command(name = "buffer-pool-verify")]
enum Verify {
    /// Enumerate every request sequence of the named configurations (every one when none is
    /// named: core, copies, transfer, tiers, cross, parts, trial).
    Enumerate {
        /// Write each configuration's states (`<name>.txt`), its transitions with their inputs
        /// (`<name>.steps`) and a transition the model does not allow (`forged.steps`) here.
        #[arg(long)]
        states: Option<PathBuf>,
        configurations: Vec<String>,
    },
    /// Replay every trace (`<deployment>.inputs` beside its `<deployment>.completions`) in a
    /// directory.
    Replay {
        /// The directory the Storage Server recorded its traces in.
        traces: PathBuf,
        /// Write the states each replay passes through (`<deployment>.steps`) here.
        #[arg(long)]
        states: Option<PathBuf>,
    },
}

fn main() -> ExitCode {
    let outcome = match Verify::parse() {
        Verify::Enumerate {
            states,
            configurations,
        } => enumerate(&configurations, states.as_deref()),
        Verify::Replay { traces, states } => replay_all(&traces, states.as_deref()),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(failure) => {
            eprintln!("{failure}");
            ExitCode::FAILURE
        }
    }
}

fn enumerate(configurations: &[String], states: Option<&Path>) -> Result<(), String> {
    for reached in enumeration::enumerate(configurations, states)? {
        println!(
            "{}: {} states, {} transitions",
            reached.shape, reached.states, reached.transitions
        );
    }
    Ok(())
}

/// Replays each trace of `traces` in the order of their names.
fn replay_all(traces: &Path, states: Option<&Path>) -> Result<(), String> {
    let listed =
        std::fs::read_dir(traces).map_err(|error| format!("{}: {error}", traces.display()))?;
    let mut found = Vec::new();
    for entry in listed {
        let path = entry
            .map_err(|error| format!("{}: {error}", traces.display()))?
            .path();
        let name = path
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned());
        if let Some(name) = name
            && path
                .extension()
                .is_some_and(|extension| extension == "inputs")
        {
            found.push((name, path));
        }
    }
    found.sort();
    if found.is_empty() {
        return Err(format!("{} holds no trace", traces.display()));
    }

    for (name, inputs) in found {
        let mut written = match states {
            Some(directory) => {
                let path = directory.join(format!("{name}.steps"));
                let file =
                    File::create(&path).map_err(|error| format!("{}: {error}", path.display()))?;
                Some(BufWriter::new(file))
            }
            None => None,
        };
        let replayed = replay::replay(
            &inputs,
            written.as_mut().map(|file| file as &mut dyn std::io::Write),
        )
        .map_err(|difference| format!("{name}: {difference}"))?;
        if let Some(file) = &mut written {
            file.flush().map_err(|error| format!("{name}: {error}"))?;
        }
        println!(
            "replayed {name}: {} inputs ({} requests), every completion equal and every check clean",
            replayed.inputs, replayed.requests
        );
    }
    Ok(())
}
