//! The deployment sizes, printed from the rows `notes/h3/deployments.md` renders.
//!
//! The table is the model's own arithmetic and the measurements around it, held here in one place so
//! the note and the tool cannot drift: every figure the note states for a stage is a field below,
//! and every total the note states is a sum this program computes rather than a second literal.
//!
//! Runs host-only: no checkpoint, no GPU, no reference Python. `--root <checkpoint>` additionally
//! recomputes the derived quantities from that checkpoint's sidecars.

use std::path::PathBuf;

/// One stage of one deployment, in MiB of device memory and count of state tables.
///
/// `weights` is what the stage holds resident, `constants` what it derives once at load, `arena` the
/// peak of one call, and `tables` the state tables a request uploads.
struct Stage {
    deployment: &'static str,
    stage: &'static str,
    weights: u64,
    constants: u64,
    arena: u64,
    tables: u32,
}

#[rustfmt::skip]
const STAGES: &[Stage] = &[
    Stage { deployment: "t2va dense", stage: "text_encoder", weights: 49121, constants: 0, arena: 2, tables: 7 },
    Stage { deployment: "t2va dense", stage: "denoiser (base DiT + refiner)", weights: 41953, constants: 1946, arena: 10129, tables: 12 },
    Stage { deployment: "t2va dense", stage: "video_decoder", weights: 4618, constants: 0, arena: 7591, tables: 1 },
    Stage { deployment: "t2va dense", stage: "audio_decoder", weights: 123, constants: 0, arena: 1178, tables: 0 },
    Stage { deployment: "t2va dense", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
    Stage { deployment: "fl2va dense", stage: "text_encoder", weights: 49121, constants: 0, arena: 2, tables: 7 },
    // The encoder set and the conditioner's six buffers are a **bump released before the layout is
    // captured**: the conditioner runs once per request, writes the condition rows, and its buffers
    // are dead when the replays reserve. They never coexist with the captured plan, so the arena is
    // the plan's own 10,129 MiB and not the sum - the runner's order is prepare -> condition in the
    // bump -> release -> capture -> replay, and a runner that captures first holds 17,041.
    Stage { deployment: "fl2va dense", stage: "denoiser (base DiT + refiner + encoders)", weights: 42453, constants: 1946, arena: 10129, tables: 12 },
    Stage { deployment: "fl2va dense", stage: "video_decoder", weights: 4618, constants: 0, arena: 7591, tables: 1 },
    Stage { deployment: "fl2va dense", stage: "audio_decoder", weights: 123, constants: 0, arena: 1178, tables: 0 },
    Stage { deployment: "fl2va dense", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "text_encoder", weights: 49121, constants: 0, arena: 2, tables: 7 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "denoiser (OmniRef DiT + refiner + encoders)", weights: 42453, constants: 1946, arena: 10129, tables: 14 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "video_decoder", weights: 4618, constants: 0, arena: 7591, tables: 1 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "audio_decoder", weights: 123, constants: 0, arena: 1178, tables: 0 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
    Stage { deployment: "t2va + DMD student", stage: "text_encoder", weights: 49121, constants: 0, arena: 2, tables: 7 },
    Stage { deployment: "t2va + DMD student", stage: "denoiser (student + refiner)", weights: 41953, constants: 343, arena: 10129, tables: 12 },
    Stage { deployment: "t2va + DMD student", stage: "video_decoder", weights: 4618, constants: 0, arena: 7591, tables: 1 },
    Stage { deployment: "t2va + DMD student", stage: "audio_decoder", weights: 123, constants: 0, arena: 1178, tables: 0 },
    Stage { deployment: "t2va + DMD student", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
    Stage { deployment: "t2va dense, Ulysses 2", stage: "text_encoder", weights: 49121, constants: 0, arena: 2, tables: 7 },
    // The DiT's weights are not sliced by the Ulysses degree: every rank reads the whole 638-name
    // set at the global head width (`Attention::load` projects `heads * head_dim`, and `projection`
    // takes no rank), so this row is the one-rank row plus the four exchange entries' arena.
    Stage { deployment: "t2va dense, Ulysses 2", stage: "denoiser (base DiT + refiner)", weights: 41953, constants: 1946, arena: 11169, tables: 12 },
    Stage { deployment: "t2va dense, Ulysses 2", stage: "video_decoder", weights: 4618, constants: 0, arena: 7591, tables: 1 },
    Stage { deployment: "t2va dense, Ulysses 2", stage: "audio_decoder", weights: 123, constants: 0, arena: 1178, tables: 0 },
    Stage { deployment: "t2va dense, Ulysses 2", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
    Stage { deployment: "t2va dense, Ulysses 2 x tensor 2", stage: "text_encoder", weights: 25129, constants: 0, arena: 1, tables: 7 },
    Stage { deployment: "t2va dense, Ulysses 2 x tensor 2", stage: "denoiser (base DiT + refiner)", weights: 41953, constants: 1946, arena: 11169, tables: 12 },
    Stage { deployment: "t2va dense, Ulysses 2 x tensor 2", stage: "video_decoder", weights: 4618, constants: 0, arena: 7591, tables: 1 },
    Stage { deployment: "t2va dense, Ulysses 2 x tensor 2", stage: "audio_decoder", weights: 123, constants: 0, arena: 1178, tables: 0 },
    Stage { deployment: "t2va dense, Ulysses 2 x tensor 2", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
];

/// The deployments, in the note's order.
///
/// The last two state their degrees because the two shard different stages: the Ulysses degree cuts
/// the denoiser's heads and its plan grows with it, while the tensor degree shards the text encoder's
/// projections and leaves the DiT's plan exactly where it was (`2d44d7da8`, measured).
const DEPLOYMENTS: [&str; 6] = [
    "t2va dense",
    "fl2va dense",
    "ref2va dense (tile 128)",
    "t2va + DMD student",
    "t2va dense, Ulysses 2",
    "t2va dense, Ulysses 2 x tensor 2",
];

fn mib(bytes: u64) -> String {
    if bytes >= 1024 {
        format!("{:.2} GiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} MiB")
    }
}

/// Report a plan that names one buffer twice, and remember to fail the run.
///
/// The executor keys the arena's storage by these names, so a repeated name aliases two regions: the
/// plan stops saying what the block reserves while its arithmetic still adds up, which is a defect
/// no size a reader checks can find. One plan carried it (`audio.wide`, a patch widening and a head
/// widening, `f2e1ef10c`), so every plan this example builds is scanned.
fn distinct_names(label: &str, plan: &[minimax_h3::decoding::Buffer]) {
    let mut seen = std::collections::HashSet::new();
    for buffer in plan {
        if !seen.insert(buffer.name.as_str()) {
            println!("  PLAN COLLISION: {label} names `{}` twice", buffer.name);
            PLAN_COLLISIONS.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

static PLAN_COLLISIONS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut root: Option<PathBuf> = None;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--root" {
            root = arguments.next().map(PathBuf::from);
        }
    }

    for deployment in DEPLOYMENTS {
        let stages: Vec<&Stage> = STAGES
            .iter()
            .filter(|stage| stage.deployment == deployment)
            .collect();
        println!("\n### {deployment}\n");
        println!("| stage | weights | constants | arena | tables |");
        println!("| --- | --- | --- | --- | --- |");
        for stage in &stages {
            println!(
                "| {} | {} | {} | {} | {} |",
                stage.stage,
                mib(stage.weights),
                mib(stage.constants),
                mib(stage.arena),
                stage.tables
            );
        }
        // The totals are sums of the rows above, so a stage's change cannot leave them stale.
        let weights: u64 = stages.iter().map(|stage| stage.weights).sum();
        let constants: u64 = stages.iter().map(|stage| stage.constants).sum();
        // A stage alone on its rank holds its own weights, constants and arena; the peak over the
        // stages is what any one of them needs.
        let largest = stages
            .iter()
            .map(|stage| stage.weights + stage.constants)
            .max()
            .unwrap_or(0);
        println!(
            "\nall stages in one place: {} of weights, {} of constants",
            mib(weights),
            mib(constants)
        );
        // The note's "largest stage" is weights + constants + arena; this line says which of the three
        // it is summing so the two cannot drift apart quietly.
        println!(
            "largest stage by weights + constants: {} (the note's largest stage counts the arena too, and the \
             table above carries it)",
            mib(largest)
        );
        // The note's "largest stage" is weights + constants + arena for the stage that needs most, and
        // that is the number a deployment is bought against; it is printed in MiB because the note adds
        // the long request's arena to it and a rounded GiB would not add up.
        if let Some(peak) = stages
            .iter()
            .max_by_key(|stage| stage.weights + stage.constants + stage.arena)
        {
            println!(
                "largest stage, weights + constants + arena: {} MiB ({})",
                peak.weights + peak.constants + peak.arena,
                peak.stage
            );
        }
    }

    if let Some(root) = root {
        derived(&root)?;
    } else {
        println!(
            "\n(recomputed row bounds and decode plans need `--root <checkpoint>`; the stage table \
             above is the note's, and needs neither a checkpoint nor a GPU)"
        );
    }
    if PLAN_COLLISIONS.load(std::sync::atomic::Ordering::Relaxed) {
        return Err("a plan names one buffer twice, so two regions would alias".into());
    }
    Ok(())
}

/// Recompute the quantities the note marks as arithmetic, from a checkpoint's sidecars.
fn derived(root: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    use minimax_h3::Dit;
    use minimax_h3::checkpoint::{Layout, Sidecars};
    use minimax_h3::config::{Config, NAMED_CANVASES};
    use minimax_h3::partition::Capacity;

    let layout = Layout::detect(root)?;
    let sidecars = Sidecars::read(root, None, &layout)?;
    let config = Config::normalize(&layout, &sidecars)?;
    let denoiser = &config.denoisers[&Dit::Base];
    let width = u64::from(denoiser.transformer.hidden_size);

    println!("\n## recomputed from {}\n", root.display());
    for frames in [124_u32, 362] {
        let capacity = Capacity {
            text_rows: 128,
            condition_rows: 192,
            frames,
        };
        let rows = u64::from(capacity.rows(denoiser)?);
        println!(
            "{frames} frames: {rows} rows, packed hidden {} (rows x {width} x 2 B)",
            mib(rows * width * 2 / (1 << 20))
        );
        if let minimax_h3::config::AttentionKind::Sparse { tile, .. } = denoiser.attention {
            // The tile bound is the deployment's own maximum over every layout it serves, and it
            // is the model's (`Layouts::tiles_bound`) - a scratch figure derived here from one
            // request's rows would be a second statement of it and could size the sparse stack's
            // buffer short.
            let arrangement = minimax_h3::partition::Partition::single().arrangement(denoiser)?;
            let bound =
                minimax_h3::denoiser::Layouts::new(denoiser.clone(), arrangement)?.tiles_bound()?;
            let heads = u64::from(denoiser.transformer.num_attention_heads);
            let bound = u64::from(bound);
            println!(
                "  sparse scratch {} at the deployment's bound (tiles^2 x heads x 2 B, {bound} tiles)",
                mib(bound * bound * heads * 2 / (1 << 20))
            );
            // This request's own tile count, for comparison: the buffer is sized by the bound, so a
            // request under it must not be read as the size.
            println!(
                "    this request's rows span {} tiles of {} rows",
                rows.div_ceil(u64::from(tile)),
                tile
            );
        } else {
            println!("  sparse scratch: this DiT attends densely, so no tile bound applies");
        }
    }

    // The stage table's weight column, from the checkpoint's own index rather than by hand.
    //
    // A component's `total_size` is the index's sum over every tensor it names, so a row cannot
    // drift from the checkpoint it describes - which is how the three hand-carried rows of the
    // previous round drifted. The index is read as a small file: the shards themselves are not
    // needed to know how many bytes they hold. Two components hold a block's encoder and its
    // decoder in one directory, and the table's split of those into two stages is the loaders' key
    // sets, so they are printed whole with the table's two rows beside them.
    let mut component_bytes = std::collections::BTreeMap::new();
    println!();
    for component in minimax_h3::partition::Component::ALL {
        let Some(directory) = component.directories().first().copied() else {
            continue;
        };
        // The two VAEs are each named by two components (a decoder's directory and the condition
        // encoder that reads its other half), and the byte figure belongs to the directory.
        if component_bytes.contains_key(directory) {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(root.join(directory)) else {
            println!("weights {directory}: no such directory, so the table's figure stands alone");
            continue;
        };
        let mut named = None;
        let mut single = None;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".index.json") {
                named = Some(entry.path());
            } else if name.ends_with(".safetensors") {
                single = Some(entry.path());
            }
        }
        // A sharded export states its own total in the index's metadata; an export of one file
        // states nothing, and its data section is the file less its eight-byte length and header.
        if let Some(path) = named {
            let text = std::fs::read_to_string(path)?;
            let index: serde_json::Value = serde_json::from_str(&text)?;
            let total = index["metadata"]["total_size"]
                .as_u64()
                .or_else(|| index["total_size"].as_u64())
                .unwrap_or(0);
            let count = index["weight_map"].as_object().map_or(0, |map| map.len());
            component_bytes.insert(directory.to_owned(), total);
            println!(
                "weights {directory}: {count} tensors in the index, {} MiB",
                total / (1 << 20)
            );
        } else if let Some(path) = single {
            let bytes = std::fs::metadata(&path)?.len();
            let mut header = [0_u8; 8];
            let mut file = std::fs::File::open(&path)?;
            std::io::Read::read_exact(&mut file, &mut header)?;
            let header_len = u64::from_le_bytes(header);
            let data = bytes.saturating_sub(8 + header_len);
            component_bytes.insert(directory.to_owned(), data);
            println!(
                "weights {directory}: one file, {data} bytes of data = {} MiB (its header states no \
                 count)",
                data / (1 << 20)
            );
        } else {
            println!(
                "weights {directory}: no index and no shard beside it here, so the table's figure \
                 stands alone"
            );
        }
    }

    // The table's weight column is the **kept** tensors of each block, and a component's total is
    // not that: the text encoder retains 50 of its 64 layers, the denoiser folds `adaln_proj` away,
    // and the two VAEs are told apart by their key prefixes. Which tensors a block keeps is the
    // loaders' selection, and restating it here would be a second statement of one rule - so the
    // index is a bound the table must sit under, not a figure that replaces it. A row that
    // overstates a block is what the bound catches; one that understates it is caught by
    // `inspect_checkpoint`, which proves the key groups exist.
    for deployment in DEPLOYMENTS {
        // A rank places one DiT partition, so the other one's directory is not part of the bound.
        let placed = if deployment.contains("ref2va") {
            Dit::Reference
        } else {
            Dit::Base
        };
        let unplaced = match placed {
            Dit::Base => "transformer_ref",
            Dit::Reference => "transformer",
        };
        let placed_directories = component_bytes
            .iter()
            .filter(|(directory, _)| directory.as_str() != unplaced)
            .map(|(directory, _)| directory.as_str())
            .collect::<Vec<_>>();
        let all: u64 = component_bytes
            .iter()
            .filter(|(directory, _)| directory.as_str() != unplaced)
            .map(|(_, bytes)| bytes / (1 << 20))
            .sum();
        let table: u64 = STAGES
            .iter()
            .filter(|stage| stage.deployment == deployment)
            .map(|stage| stage.weights)
            .sum();
        if table > all {
            return Err(format!(
                "{deployment} states {table} MiB of weights and the components it places hold \
                 {all} MiB"
            )
            .into());
        }
        println!(
            "weights {deployment}: table {table} MiB, under the {all} MiB that {} hold",
            placed_directories.join(", ")
        );
    }

    // The DiT's plans, from the crate's own function, one line per deployment shape: the base DiT on
    // one rank and on two, and the reference DiT a `ref2va` runs. Each is one evaluation's reservations.
    let ratio = config.video_vae.spatial_compression();
    // (name, DiT, task, Ulysses degree, text-tensor degree). The two degrees are stated apart
    // because they shard different things: the denoiser's heads follow the Ulysses degree, while the
    // text encoder's shards follow the tensor degree and leave the DiT's plan untouched. A row that
    // named one degree while measuring the other would size a deployment it never planned.
    for (name, dit, task, ulysses, tensor) in [
        (
            "t2va / fl2va, 1 rank",
            Dit::Base,
            nn::capability::VideoTask::TextToVideoAudio,
            1_usize,
            1_usize,
        ),
        (
            "t2va, Ulysses 2 (the text encoder unsharded)",
            Dit::Base,
            nn::capability::VideoTask::TextToVideoAudio,
            2,
            1,
        ),
        (
            "t2va, Ulysses 2 x tensor 2",
            Dit::Base,
            nn::capability::VideoTask::TextToVideoAudio,
            2,
            2,
        ),
        (
            "ref2va (OmniRef DiT), 1 rank",
            Dit::Reference,
            nn::capability::VideoTask::ReferenceToVideoAudio,
            1,
            1,
        ),
    ] {
        let Some(rules) = config.denoisers.get(&dit) else {
            println!("DiT plan {name}: this checkpoint has no such DiT");
            continue;
        };
        let partition =
            minimax_h3::partition::Partition::serving(&config, task, ulysses, tensor, 0, 0)?;
        let arrangement = partition.arrangement(rules)?;
        for frames in [124_u32, 362] {
            let conditioned = !matches!(task, nn::capability::VideoTask::TextToVideoAudio);
            let request = nn::capability::Layout {
                video: nn::capability::LatentShape {
                    channels: rules.transformer.video_channels,
                    frames: minimax_h3::packing::video_latent_frames(frames)?,
                    height: 672 / ratio,
                    width: 1536 / ratio,
                },
                audio: Some(nn::capability::LatentShape {
                    channels: minimax_h3::config::AUDIO_CHANNELS * rules.transformer.audio_channels,
                    frames: minimax_h3::packing::audio_latent_frames(frames),
                    height: 1,
                    width: 1,
                }),
                text_rows: arrangement.text_rows,
                condition_rows: arrangement.condition_rows,
                condition_elements: 0,
            };
            let plan = minimax_h3::denoiser::denoise_arena(
                rules,
                &arrangement,
                minimax_h3::modulation::Groups::of(conditioned).count(),
                nn::DType::BF16,
                &request,
            )?;
            distinct_names(&format!("the DiT plan at {frames} frames"), &plan);
            let bytes: u64 = plan.iter().map(minimax_h3::decoding::Buffer::bytes).sum();
            println!(
                "DiT plan {name}, {frames} frames: {} entries, {} = {} MiB ({} bytes)",
                plan.len(),
                mib(bytes / (1 << 20)),
                bytes / (1 << 20),
                bytes
            );
            // The entries in the order the plan reserves them, which is the order the arena's
            // offsets follow: the note describes this list by name and shape, so it must be
            // printable from the same call that sizes it.
            if std::env::var("H3_DIT_BUFFERS").is_ok() {
                for buffer in &plan {
                    println!(
                        "    {:28} {:5} {:>12} elements {:>12} B",
                        buffer.name,
                        buffer.element,
                        buffer.elements,
                        buffer.bytes()
                    );
                }
            }
        }
    }

    // The audio encoder's plan, which a ref2va's conditioning pass reserves beside the video
    // encoder's set: `samples` is the reference's sample count at the request's frame count.
    for frames in [124_u32, 362] {
        let samples = u64::from(minimax_h3::packing::audio_latent_frames(frames)) * 800;
        let plan = minimax_h3::encoding::audio_encoder_arena(&config.audio_vae, 2, samples)?;
        distinct_names("the audio encoder plan", &plan);
        let bytes: u64 = plan.iter().map(minimax_h3::decoding::Buffer::bytes).sum();
        println!(
            "audio encoder plan {frames} frames ({samples} samples): {} buffers, {}",
            plan.len(),
            mib(bytes / (1 << 20))
        );
    }
    // The conditioning pass's own arena, which a conditioned deployment reserves once per condition
    // beside the encoder's set. Two anchors, because the tile buffer is in pixels and the rest are in
    // latents: a still and a 25-frame reference at the same raster.
    // The latent raster the packing names: 84 columns by 48 rows, the reference's own.
    let latent = minimax_h3::encoding::Extent {
        frames: 1,
        height: 48,
        width: 84,
    };
    for (frames, extent) in [
        (1_u32, latent),
        (
            // 25 frames of a reference are 7 latent frames by the encoder's own rule, and the plan
            // refuses a caller that hands it the wrong one of the two counts.
            25,
            minimax_h3::encoding::Extent {
                frames: 7,
                ..latent
            },
        ),
    ] {
        let plan = minimax_h3::encoding::conditioning_arena(&config.video_vae, frames, extent)?;
        distinct_names("the conditioning plan", &plan);
        let bytes: u64 = plan.iter().map(minimax_h3::decoding::Buffer::bytes).sum();
        println!(
            "conditioning plan {frames} frames ({} latent): {} buffers, {} bytes = {} KiB",
            extent.frames,
            plan.len(),
            bytes,
            bytes / (1 << 10)
        );
    }

    // The splice's plan: one buffer, one clip's raster at the canvas's pixel extents. It does not
    // grow with the grid or the clip count, because a splice blends one clip at a time.
    for canvas in NAMED_CANVASES {
        let clip = minimax_h3::video_vae::Canvas::clip(
            &config.video_vae,
            canvas.height / ratio,
            canvas.width / ratio,
        );
        let plan = minimax_h3::output::splice_arena(&config.video_vae, &clip, nn::DType::F16)?;
        distinct_names("the splice plan", &plan);
        let bytes: u64 = plan.iter().map(minimax_h3::decoding::Buffer::bytes).sum();
        println!(
            "splice plan {}x{}: {} buffers, {} bytes = {} MiB",
            canvas.width,
            canvas.height,
            plan.len(),
            bytes,
            bytes / (1 << 20)
        );
    }
    for canvas in NAMED_CANVASES {
        let latent = minimax_h3::video_vae::Canvas::clip(
            &config.video_vae,
            canvas.height / ratio,
            canvas.width / ratio,
        );
        let decode = minimax_h3::decoding::video_arena(&config.video_vae, &latent)?;
        distinct_names(
            &format!("the decode plan at {}x{}", canvas.width, canvas.height),
            &decode,
        );
        let arena: u64 = decode.iter().map(|buffer| buffer.bytes()).sum();
        if std::env::var("H3_ARENA_BUFFERS").is_ok() && canvas.width == 1536 {
            for buffer in minimax_h3::decoding::video_arena(&config.video_vae, &latent)? {
                let bytes = buffer.bytes();
                println!("    {} {} {} B", buffer.name, buffer.element, bytes);
            }
        }
        println!(
            "decode plan {}x{}: {} bytes = {} MiB",
            canvas.width,
            canvas.height,
            arena,
            arena / (1 << 20)
        );
    }
    for frames in [124_u32, 362] {
        // `audio_arena` takes audio latent steps, not video frames: one latent step is 800 samples and
        // the request's frame count reaches them through the packing rule.
        let plan = minimax_h3::audio_vae::audio_arena(
            &config.audio_vae,
            minimax_h3::packing::audio_latent_frames(frames),
        )?;
        distinct_names("the audio decoder plan", &plan);
        let bytes: u64 = plan.iter().map(|buffer| buffer.bytes()).sum();
        println!(
            "audio plan {frames} frames ({} latent steps): {} buffers, {} bytes = {} MiB",
            minimax_h3::packing::audio_latent_frames(frames),
            plan.len(),
            bytes,
            bytes / (1 << 20)
        );
        if std::env::var("H3_PLAN_BUFFERS").is_ok() && frames == 124 {
            for buffer in &plan {
                let size = buffer.bytes();
                println!(
                    "    {:32} {} {:>12} elements {:>10} B",
                    buffer.name, buffer.element, buffer.elements, size
                );
            }
        }
    }
    // A reference clip's tile grid, so the loop's multiplication is visible beside the per-tile plan.
    for (height, width) in [(768_u32, 1344_u32), (672, 1536)] {
        let encoder = minimax_h3::tiled_encode::ClipEncoder::new(height, width)?;
        let plan = encoder.plan();
        let ratio = config.video_vae.spatial_compression();
        let mut total = 0_u64;
        let mut largest = 0_u64;
        for tile in &plan.tiles {
            let extent = minimax_h3::encoding::Extent {
                frames: minimax_h3::decoding::CLIP_OUTPUT_FRAMES,
                height: tile.rows.1 * ratio,
                width: tile.columns.1 * ratio,
            };
            let bytes: u64 =
                minimax_h3::encoding::video_encoder_tile_arena(&config.video_vae, extent)?
                    .iter()
                    .map(|buffer| buffer.bytes())
                    .sum();
            total += bytes;
            if largest < bytes {
                largest = bytes;
                println!(
                    "    largest tile: latent {}x{}, pixels {}x{}",
                    tile.rows.1, tile.columns.1, extent.height, extent.width
                );
            }
        }
        // After the hoist the run reserves one set per *distinct* tile extent, which is what
        // `VideoEncoder::reserve` keeps; the extents are the plan's, so the sum is the rank's cost.
        let mut distinct: Vec<(u32, u32)> = Vec::new();
        for tile in &plan.tiles {
            let extent = (tile.rows.1 * ratio, tile.columns.1 * ratio);
            if !distinct.contains(&extent) {
                distinct.push(extent);
            }
        }
        let mut hoisted = 0_u64;
        for (height, width) in &distinct {
            let extent = minimax_h3::encoding::Extent {
                frames: minimax_h3::decoding::CLIP_OUTPUT_FRAMES,
                height: *height,
                width: *width,
            };
            hoisted += minimax_h3::encoding::video_encoder_tile_arena(&config.video_vae, extent)?
                .iter()
                .map(|buffer| buffer.bytes())
                .sum::<u64>();
        }
        if std::env::var("H3_PLAN_TILES").is_ok() && width == 1344 {
            for tile in &plan.tiles {
                println!(
                    "    tile r{} c{}: latent {}x{} at ({},{}) overlaps {}/{}",
                    tile.row,
                    tile.column,
                    tile.rows.1,
                    tile.columns.1,
                    tile.rows.0,
                    tile.columns.0,
                    tile.above,
                    tile.left
                );
            }
        }
        println!(
            "encoder grid {width}x{height}: {} tiles, {} distinct extents, one set {} MiB, if-reserved-per-tile {} MiB, reserved once {} MiB",
            plan.tiles.len(),
            distinct.len(),
            largest / (1 << 20),
            total / (1 << 20),
            hoisted / (1 << 20)
        );
    }
    // The video encoder reads one tile per call, so its plan is stated per tile; the driver's loop
    // must hoist it or the arena is reserved once per tile.
    for (width, height) in [(256_u32, 256_u32), (512, 128)] {
        let raster = minimax_h3::encoding::Extent {
            frames: 25,
            height,
            width,
        };
        let plan = minimax_h3::encoding::video_encoder_tile_arena(&config.video_vae, raster)?;
        distinct_names("the video encoder tile plan", &plan);
        let bytes: u64 = plan.iter().map(|buffer| buffer.bytes()).sum();
        println!(
            "encoder tile {width}x{height} px: {} buffers, {} bytes = {} MiB",
            plan.len(),
            bytes,
            bytes / (1 << 20)
        );
    }
    let tile = minimax_h3::video_vae::Canvas::clip(&config.video_vae, 16, 16);
    let one: u64 = minimax_h3::decoding::video_arena(&config.video_vae, &tile)?
        .iter()
        .map(|buffer| buffer.bytes())
        .sum();
    println!("one tile: {}", mib(one / (1 << 20)));
    Ok(())
}
