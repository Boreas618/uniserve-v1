//! Run a deployment's load path on the host and report what each component's load read.
//!
//!     cargo run -p minimax-h3 --example check_load -- --root <export> [--root <ref export>] \
//!         [--task t2va|fl2va|ref2va] [--dtype auto|bf16|f16|f32] [--component <name> ...]
//!
//! This is the loading boundary's own check, and it needs no device: `nn::HostStore` keeps the
//! weights in host memory, so `load` runs its real path - sidecars, the fold, the encoders'
//! attach, the tables - while every layer it builds is built for enumeration and never executes.
//! What it answers is the question an index walk cannot: whether a component's load **publishes**
//! the weights it says it reads, and which tensors of its declared directories it left alone.
//!
//! - The verdict is the number of weight regions the store published. A component that publishes
//!   none asked the checkpoint for nothing, whatever its key list says.
//! - The published types are listed per component. The key of a retained region carries the type it
//!   was retained at, so a per-tensor pin shows here as the type the model asked for rather than the
//!   one the file holds, and a stated deployment converting its weights shows as one type only.
//! - The per-name `read`/`unread` columns come from `Checkpoint::reading`, which records a claim the
//!   loader takes per name; this loader reads bytes without claiming, so the columns are printed as
//!   information and are not a verdict. When the claims land, the unread column becomes the second
//!   finding: a component whose loader silently skips a group of names it needs.
//!
//! **Host memory.** One component is loaded at a time and its own tensor bytes are printed before
//! it loads, so a caller can pick something that fits the machine: a DiT partition is about 63 GB,
//! the text encoder about 64 GB, the video VAE about 10 GB, and a VAE encoder well under 1 GB.
//! Loading several components in one invocation sums their bytes, because the store keeps them.

use std::path::PathBuf;

use nn::weights::{HostStore, WeightLoader};
use nn::{DType, VideoTask};

use minimax_h3::checkpoint::{Layout, Sidecars};
use minimax_h3::config::Config;
use minimax_h3::denoiser::ConditionEncoders;
use minimax_h3::load;
use minimax_h3::load::stored_types;
use minimax_h3::partition::{Component, Components, Partition, Ulysses};

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

fn parse_dtype(text: &str) -> Result<Option<DType>, String> {
    match text {
        "auto" => Ok(None),
        "bf16" => Ok(Some(DType::BF16)),
        "f16" => Ok(Some(DType::F16)),
        "f32" => Ok(Some(DType::F32)),
        other => Err(format!(
            "`{other}` is not an element type; write auto, bf16, f16 or f32"
        )),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut roots: Vec<PathBuf> = Vec::new();
    let mut requested: Vec<String> = Vec::new();
    let mut stated: Option<DType> = None;
    let mut task_name = "t2va".to_owned();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--root" => {
                roots.push(arguments.next().ok_or("--root needs a path")?.into());
            }
            "--component" => requested.push(arguments.next().ok_or("--component needs a name")?),
            "--dtype" => stated = parse_dtype(&arguments.next().ok_or("--dtype needs a type")?)?,
            "--task" => {
                task_name = arguments.next().ok_or("--task needs a name")?;
            }
            other => return Err(format!("unknown argument `{other}`").into()),
        }
    }
    if roots.is_empty() {
        return Err(
            "usage: check_load --root <export> [--root <reference export>] \
             [--task t2va|fl2va|ref2va] [--dtype auto|bf16|f16|f32] [--component <name> ...]"
                .into(),
        );
    }
    let task = task(&task_name)?;

    // `ref2va` continues a base release, so more than one root is a union rather than a mistake:
    // the earlier root shadows the later one per tensor name, and a component keeps both roots'
    // shards.
    let paths = roots.iter().map(PathBuf::as_path).collect::<Vec<_>>();
    let checkpoint = match nn::Checkpoint::open_roots(&paths) {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            return Err(format!(
                "{error}\nnote: this instrument loads weights, so it needs the export's shards and \
                 not only its indexes and sidecars. An export of `*.index.json` with no \
                 `*.safetensors` beside it cannot be loaded on this host at all; run this where the \
                 weights are."
            )
            .into());
        }
    };
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

    // The deployment's own rank layout decides which blocks exist and which directory each reads;
    // `--component` narrows to a subset, which is how a caller keeps one load inside host memory.
    let serving = Partition::serving(&config, task, Ulysses::single(), 1, 0, 0)?;
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
        // **What the load deliberately leaves alone, printed with its reason.** A tensor the
        // checkpoint holds, the load does not ask for and nobody has named is the finding this
        // column exists to make visible; a named exclusion is a decision, and a reader who cannot
        // see it cannot tell the two apart.
        for exclusion in load::excluded_names(component.name()) {
            println!("  not asked for: {} ({})", exclusion.name, exclusion.reason);
        }
        let mut budget = 0_u64;
        for directory in directories {
            match directory_bytes(&checkpoint, directory) {
                Ok(bytes) => {
                    budget += bytes;
                    println!("  {directory}: {} MiB of tensors", bytes / (1 << 20));
                }
                Err(error) => println!("  {directory}: not openable here ({error})"),
            }
            // `dtype = "auto"` converts nothing, so the element type the deployment loads this
            // component in is the one its tensors are stored in. Printing it next to the count is
            // what makes a store list's dtype auditable against the export it will run on.
            match stored_types(&checkpoint, directory) {
                Ok(types) => {
                    // `auto` converts nothing, so what it loads is exactly what the file holds -
                    // one type or several. A component of several is not a defect; the pins are the
                    // model's requests, and a mixed cell is refused at its launch, by name.
                    let listed: Vec<String> = types
                        .iter()
                        .map(|(dtype, example)| format!("{dtype} (such as `{example}`)"))
                        .collect();
                    let what = if types.len() == 1 {
                        "which is what `auto` loads"
                    } else {
                        "which `auto` loads as stored, weight by weight"
                    };
                    println!("  {directory}: {}, {what}", listed.join(", "));
                }
                Err(error) => println!("  {directory}: not answerable here ({error})"),
            }
        }
        println!("  at most {} MiB of host memory", budget / (1 << 20));

        // One component, one partition: the load path is the deployment's, and what it places is
        // narrowed so that one component's weights are what the store holds.
        //
        // The condition encoders are the one exception, and it is why `--component latent_encoder`
        // exists at all: they are the denoiser's conditioning half, so a partition holding them
        // alone is not a deployment and `Partition::validate` refuses it by name. What this probe
        // asks is whether their names answer the export, and `ConditionEncoders::load` is the call
        // the deployment loader makes for them in a pass of its own - so it is called directly.
        let partition = serving.placing(Components::of([component]));
        let mut store = HostStore::new();
        let outcome = {
            // `--dtype` is the deployment's statement, and it reaches both the loader (one
            // conversion per weight, named in the retained key) and the boundary (`auto` refuses a
            // component of two floating-point types rather than mixing them in a forward).
            let mut loader = WeightLoader::new(&checkpoint, &mut store, stated, None);
            if matches!(component, Component::LatentEncoder) {
                ConditionEncoders::load(&config, &checkpoint, &mut loader).map(|_| ())
            } else {
                let tile = minimax_h3::video_vae::TILE_PIXELS;
                load(&checkpoint, partition, stated, &[(tile, tile)], &mut loader).map(|_| ())
            }
        };
        if let Err(error) = outcome {
            println!("  load refused: {error}");
            failures.push(format!("{} refused to load: {error}", component.name()));
            continue;
        }
        println!("  {} weight regions published", store.len());

        // **The two-sided check.** A load need not read every tensor a file holds; what it owes is to
        // name the ones it does not. The published regions carry the checkpoint tensors' names, so
        // this subtracts three sets - what the checkpoint holds, what the load asked for, and what
        // the component names as excluded - and a name in the remainder is a weight the model
        // silently skips. `1 087 tensors, two of them the spectral loss head` is a fact about this
        // component; this line is what makes it a decision rather than a claim.
        let mut asked: Vec<String> = store
            .keys()
            .flat_map(|key| key.region.tensors.iter().cloned())
            .collect();
        asked.sort_unstable();
        asked.dedup();
        for directory in directories {
            let Ok(source) = checkpoint.component(directory) else {
                continue;
            };
            let held: Vec<String> = source.names().map(str::to_owned).collect();
            // The names are relative to the checkpoint's own directory, so the table is keyed by
            // it: a component may hold two directories (`latent_encoder` holds `vae` and `audio_vae`).
            let missing = load::unasked(directory, &asked, &held);
            if missing.is_empty() {
                println!("  {directory}: every tensor is asked for or named as excluded");
            } else {
                println!(
                    "  {directory}: {} tensor(s) neither asked for nor named as excluded: {}",
                    missing.len(),
                    missing
                        .iter()
                        .map(|name| format!("`{name}`"))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                failures.push(format!(
                    "{} leaves {} tensor(s) of {directory} unaccounted: {}",
                    component.name(),
                    missing.len(),
                    missing.join(", ")
                ));
            }
        }

        // The types the load actually published, which is the statement a per-tensor pin makes: the
        // key of a retained region carries the type it was retained at, so a pin shows up here as a
        // type the component does not store (a conversion) or as two types side by side where the
        // reference pins two families differently. A deployment that stated one type should show one
        // and nothing else.
        let mut types: Vec<(DType, usize)> = Vec::new();
        for key in store.keys() {
            match types.iter_mut().find(|(dtype, _)| *dtype == key.dtype) {
                Some((_, count)) => *count += 1,
                None => types.push((key.dtype, 1)),
            }
        }
        let published: Vec<String> = types
            .iter()
            .map(|(dtype, count)| format!("{count} at {dtype}"))
            .collect();
        println!("  published: {}", published.join(", "));
        if let Some(stated) = stated
            && types.iter().any(|(dtype, _)| *dtype != stated)
        {
            let other: Vec<String> = types
                .iter()
                .filter(|(dtype, _)| *dtype != stated)
                .map(|(dtype, count)| format!("{count} at {dtype}"))
                .collect();
            println!(
                "  note: the deployment states {stated} and the load published {} - a pinned tensor \
                 keeps its pin, and a component that is not pinned must not",
                other.join(", ")
            );
        }
        // What the load published is the load's own statement of what it asked for. A component
        // that publishes nothing read nothing, whatever its keys say, and that is a finding.
        if store.is_empty() {
            failures.push(format!(
                "{} published no weight region: its load asked the checkpoint for no tensor",
                component.name()
            ));
        }

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
                // The reading record is kept by a claim the loader takes per name, and a loader
                // that reads bytes without claiming leaves it empty - so this is printed and not
                // failed on: the published-region count above is the load's own statement. What
                // the column answers, when it is filled, is which tensors a component left alone.
                println!("    (no per-name claims recorded; the published count is the verdict)");
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
