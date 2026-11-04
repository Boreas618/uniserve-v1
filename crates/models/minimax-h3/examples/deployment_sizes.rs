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
    Stage { deployment: "t2va dense", stage: "denoiser (base DiT + refiner)", weights: 41953, constants: 1946, arena: 1470, tables: 12 },
    Stage { deployment: "t2va dense", stage: "video_decoder", weights: 4618, constants: 0, arena: 7591, tables: 1 },
    Stage { deployment: "t2va dense", stage: "audio_decoder", weights: 123, constants: 0, arena: 1178, tables: 0 },
    Stage { deployment: "t2va dense", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
    Stage { deployment: "fl2va dense", stage: "text_encoder", weights: 49121, constants: 0, arena: 2, tables: 7 },
    Stage { deployment: "fl2va dense", stage: "denoiser (base DiT + refiner)", weights: 41953, constants: 1946, arena: 1470, tables: 12 },
    Stage { deployment: "fl2va dense", stage: "video_decoder", weights: 4618, constants: 0, arena: 7591, tables: 1 },
    Stage { deployment: "fl2va dense", stage: "audio_decoder", weights: 123, constants: 0, arena: 1178, tables: 0 },
    Stage { deployment: "fl2va dense", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
    Stage { deployment: "fl2va dense", stage: "latent_encoder", weights: 512, constants: 0, arena: 189582, tables: 0 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "text_encoder", weights: 49121, constants: 0, arena: 2, tables: 7 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "denoiser (OmniRef DiT + refiner)", weights: 41953, constants: 1946, arena: 1550, tables: 14 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "video_decoder", weights: 4618, constants: 0, arena: 7591, tables: 1 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "audio_decoder", weights: 123, constants: 0, arena: 1178, tables: 0 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "latent_encoder", weights: 512, constants: 0, arena: 189582, tables: 0 },
    Stage { deployment: "t2va + DMD student", stage: "text_encoder", weights: 49121, constants: 0, arena: 2, tables: 7 },
    Stage { deployment: "t2va + DMD student", stage: "denoiser (student + refiner)", weights: 41953, constants: 343, arena: 1470, tables: 12 },
    Stage { deployment: "t2va + DMD student", stage: "video_decoder", weights: 4618, constants: 0, arena: 7591, tables: 1 },
    Stage { deployment: "t2va + DMD student", stage: "audio_decoder", weights: 123, constants: 0, arena: 1178, tables: 0 },
    Stage { deployment: "t2va + DMD student", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
    Stage { deployment: "t2va dense, TP2", stage: "text_encoder", weights: 25129, constants: 0, arena: 1, tables: 7 },
    Stage { deployment: "t2va dense, TP2", stage: "denoiser (base DiT + refiner)", weights: 21299, constants: 1946, arena: 735, tables: 12 },
    Stage { deployment: "t2va dense, TP2", stage: "video_decoder", weights: 4618, constants: 0, arena: 7591, tables: 1 },
    Stage { deployment: "t2va dense, TP2", stage: "audio_decoder", weights: 123, constants: 0, arena: 1178, tables: 0 },
    Stage { deployment: "t2va dense, TP2", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
];

/// The deployments, in the note's order.
const DEPLOYMENTS: [&str; 5] = [
    "t2va dense",
    "fl2va dense",
    "ref2va dense (tile 128)",
    "t2va + DMD student",
    "t2va dense, TP2",
];

fn mib(bytes: u64) -> String {
    if bytes >= 1024 {
        format!("{:.2} GiB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} MiB")
    }
}

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
    }

    if let Some(root) = root {
        derived(&root)?;
    } else {
        println!(
            "\n(recomputed row bounds and decode plans need `--root <checkpoint>`; the stage table \
             above is the note's, and needs neither a checkpoint nor a GPU)"
        );
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
            let heads = u64::from(denoiser.transformer.num_attention_heads);
            let tiles = rows.div_ceil(u64::from(tile));
            println!(
                "  sparse scratch {} (tiles^2 x heads x 2 B, {tiles} tiles)",
                mib(tiles * tiles * heads * 2 / (1 << 20))
            );
        }
    }

    let ratio = config.video_vae.spatial_compression();
    for canvas in NAMED_CANVASES {
        let latent = minimax_h3::video_vae::Canvas::clip(
            &config.video_vae,
            canvas.height / ratio,
            canvas.width / ratio,
        );
        let arena: u64 = minimax_h3::decoding::video_arena(&config.video_vae, &latent)?
            .iter()
            .map(|buffer| buffer.elements * if buffer.element == "f16" { 2 } else { 4 })
            .sum();
        if std::env::var("H3_ARENA_BUFFERS").is_ok() && canvas.width == 1536 {
            for buffer in minimax_h3::decoding::video_arena(&config.video_vae, &latent)? {
                let bytes = buffer.elements * if buffer.element == "f16" { 2 } else { 4 };
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
        let bytes: u64 = plan
            .iter()
            .map(|buffer| buffer.elements * if buffer.element == "f16" { 2 } else { 4 })
            .sum();
        println!(
            "audio plan {frames} frames ({} latent steps): {} buffers, {} bytes = {} MiB",
            minimax_h3::packing::audio_latent_frames(frames),
            plan.len(),
            bytes,
            bytes / (1 << 20)
        );
        if std::env::var("H3_PLAN_BUFFERS").is_ok() && frames == 124 {
            for buffer in &plan {
                let size = buffer.elements * if buffer.element == "f16" { 2 } else { 4 };
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
                    .map(|buffer| buffer.elements * 4)
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
                .map(|buffer| buffer.elements * 4)
                .sum::<u64>();
        }
        println!(
            "encoder grid {width}x{height}: {} tiles, {} distinct extents, largest tile {} MiB, loop-today {} MiB, hoisted {} MiB",
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
        let bytes: u64 = plan.iter().map(|buffer| buffer.elements * 4).sum();
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
        .map(|buffer| buffer.elements * if buffer.element == "f16" { 2 } else { 4 })
        .sum();
    println!("one tile: {}", mib(one / (1 << 20)));
    Ok(())
}
