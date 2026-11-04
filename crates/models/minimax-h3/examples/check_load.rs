//! Run a deployment's load path on the host and report what each component's load read.
//!
//!     cargo run -p minimax-h3 --example check_load -- --root <export> [--root <ref export>] \
//!         [--task t2va|fl2va|ref2va] [--component <name> ...]
//!
//! This is the loading boundary's own check, and it needs no device: `nn::HostStore` keeps the
//! weights in host memory, so `load` runs its real path - sidecars, the fold, the encoders'
//! attach, the tables - while every layer it builds is built for enumeration and never executes.
//! What it answers is the question an index walk cannot: which of a component's tensors the load
//! actually asked for (`Checkpoint::reading`).
//!
//! Two failures it catches and nothing else does:
//!
//! - a load that reaches into a directory its component does not declare (a `ref2va` deployment
//!   that opened `transformer/` while placing `transformer_ref/`): the declared directory reports
//!   **zero** tensors read, and that is this program's error;
//! - a component whose loader silently skips a group of names it needs: those names appear in the
//!   unread column, which is a report rather than a refusal, because a component of a shared
//!   export legitimately leaves its sibling's tensors alone.
//!
//! **Host memory.** One component is loaded at a time and its own tensor bytes are printed before
//! it loads, so a caller can pick something that fits the machine: a DiT partition is about 63 GB,
//! the text encoder about 64 GB, the video VAE about 10 GB, and a VAE encoder well under 1 GB.
//! Loading several components in one invocation sums their bytes, because the store keeps them.

use std::path::PathBuf;

use nn::VideoTask;
use nn::weights::{HostStore, WeightLoader};

use minimax_h3::checkpoint::{Layout, Sidecars};
use minimax_h3::config::Config;
use minimax_h3::load;
use minimax_h3::partition::{Component, Components, Partition};

/// The task a deployment serves, by the name its deployment file gives it.
fn task(name: &str) -> Result<VideoTask, String> {
    match name {
        "t2va" => Ok(VideoTask::TextToVideoAudio),
        "fl2va" => Ok(VideoTask::FirstLastToVideoAudio),
        "ref2va" => Ok(VideoTask::ReferenceToVideoAudio),
        other => Err(format!(
            "`{other}` is not a MiniMax-H3 task; the deployment serves t2va, fl2va or ref2va"
        )),
    }
}

/// The bytes a directory's tensors hold, which is the most host memory one component of it costs.
fn directory_bytes(
    checkpoint: &nn::Checkpoint,
    directory: &str,
) -> Result<u64, Box<dyn std::error::Error>> {
    let source = checkpoint.component(directory)?;
    let mut bytes = 0_u64;
    for name in source.names() {
        if let Some(location) = source.tensor(name) {
            bytes += location.bytes.1 - location.bytes.0;
        }
    }
    Ok(bytes)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut requested: Vec<String> = Vec::new();
    let mut task_name = "t2va".to_owned();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--root" => {
                roots.push(arguments.next().ok_or("--root needs a path")?.into());
            }
            "--component" => requested.push(arguments.next().ok_or("--component needs a name")?),
            "--task" => {
                task_name = arguments.next().ok_or("--task needs a name")?;
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }
    if roots.is_empty() {
        return Err(
            "usage: check_load --root <export> [--root <reference export>] \
             [--task t2va|fl2va|ref2va] [--component <name> ...]"
                .into(),
        );
    }
    let task = task(&task_name)?;

    // `ref2va` continues a base release, so more than one root is a union rather than a mistake:
    // the earlier root shadows the later one per tensor name, and a component keeps both roots'
    // shards.
    let paths = roots.iter().map(PathBuf::as_path).collect::<Vec<_>>();
    let checkpoint = nn::Checkpoint::open_roots(&paths)?;
    println!(
        "roots: {} ({} components: {})",
        roots
            .iter()
            .map(|root| root.display().to_string())
            .collect::<Vec<_>>()
            .join(" + "),
        checkpoint.components().count(),
        checkpoint.components().collect::<Vec<_>>().join(", ")
    );

    let layout = Layout::detect(&roots[0])?;
    let sidecars = Sidecars::read(&roots[0], None, &layout)?;
    let config = Config::normalize(&layout, &sidecars)?;

    // The deployment's own placement decides which blocks exist and which directory each reads;
    // `--component` narrows to a subset, which is how a caller keeps one load inside host memory.
    let serving = Partition::serving(&config, task, 1, 1, 0, 0)?;
    let placed = serving.blocks.iter().collect::<Vec<_>>();
    println!(
        "{task_name} places {}",
        placed
            .iter()
            .map(|component| component.name())
            .collect::<Vec<_>>()
            .join(", ")
    );
    let chosen = if requested.is_empty() {
        placed.clone()
    } else {
        requested
            .iter()
            .map(|name| {
                Component::named(name)
                    .ok_or_else(|| format!("`{name}` is not a component of this deployment"))
            })
            .collect::<Result<Vec<_>, _>>()?
    };
    for component in &chosen {
        if !serving.blocks.contains(*component) {
            return Err(format!(
                "`{}` is not placed by {task_name}; the deployment places {}",
                component.name(),
                placed
                    .iter()
                    .map(|placed| placed.name())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
            .into());
        }
    }

    let mut failures = Vec::new();
    for component in chosen {
        let directories = component.directories();
        println!("component {} ({})", component.name(), component.block());
        let mut budget = 0_u64;
        for directory in directories {
            match directory_bytes(&checkpoint, directory) {
                Ok(bytes) => {
                    budget += bytes;
                    println!("  {directory}: {} MiB of tensors", bytes / (1 << 20));
                }
                Err(error) => println!("  {directory}: not openable here ({error})"),
            }
        }
        println!("  at most {} MiB of host memory", budget / (1 << 20));

        // One component, one partition: the load path is the deployment's, and what it places is
        // narrowed so that one component's weights are what the store holds.
        let partition = serving.placing(Components::of([component]));
        let mut store = HostStore::new();
        {
            let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
            if let Err(error) = load(&checkpoint, partition, &mut loader) {
                println!("  load refused: {error}");
                failures.push(format!("{} refused to load: {error}", component.name()));
                continue;
            }
        }
        println!("  {} weight regions published", store.len());

        // The claims are per component, so nothing is merged by bare name: a base root and a
        // reference root may both carry `transformer_blocks.0.*`, and each is reported against the
        // directory the block declared.
        for directory in directories {
            let source = match checkpoint.component(directory) {
                Ok(source) => source,
                Err(_) => continue,
            };
            let reading = source.reading();
            let total = reading.read.len() + reading.unread.len();
            println!(
                "  {directory}: {total} tensors, {} read, {} unread",
                reading.read.len(),
                reading.unread.len()
            );
            if reading.read.is_empty() {
                // The load opened this directory and asked for none of its tensors: either the
                // component reads somewhere else or its loader named names this export does not
                // hold. Both are findings, and neither is visible from an index walk.
                failures.push(format!(
                    "{} declares {directory} and the load read none of its {total} tensors",
                    component.name()
                ));
            }
            for name in reading.unread.iter().take(8) {
                println!("    unread {name}");
            }
            if reading.unread.len() > 8 {
                println!("    ... {} more unread", reading.unread.len() - 8);
            }
        }
    }

    if !failures.is_empty() {
        for failure in &failures {
            eprintln!("finding: {failure}");
        }
        return Err(format!("{} load findings", failures.len()).into());
    }
    println!("every component read from every directory it declares");
    Ok(())
}
