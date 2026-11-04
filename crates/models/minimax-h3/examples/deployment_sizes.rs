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
    Stage { deployment: "t2va dense", stage: "video_decoder", weights: 4618, constants: 0, arena: 7757, tables: 3 },
    Stage { deployment: "t2va dense", stage: "audio_decoder", weights: 123, constants: 0, arena: 2, tables: 0 },
    Stage { deployment: "t2va dense", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
    Stage { deployment: "fl2va dense", stage: "text_encoder", weights: 49121, constants: 0, arena: 2, tables: 7 },
    Stage { deployment: "fl2va dense", stage: "denoiser (base DiT + refiner)", weights: 41953, constants: 1946, arena: 1470, tables: 12 },
    Stage { deployment: "fl2va dense", stage: "video_decoder", weights: 4618, constants: 0, arena: 7757, tables: 3 },
    Stage { deployment: "fl2va dense", stage: "audio_decoder", weights: 123, constants: 0, arena: 2, tables: 0 },
    Stage { deployment: "fl2va dense", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
    Stage { deployment: "fl2va dense", stage: "latent_encoder", weights: 512, constants: 0, arena: 2, tables: 0 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "text_encoder", weights: 49121, constants: 0, arena: 2, tables: 7 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "denoiser (OmniRef DiT + refiner)", weights: 41953, constants: 1946, arena: 1550, tables: 14 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "video_decoder", weights: 4618, constants: 0, arena: 7757, tables: 3 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "audio_decoder", weights: 123, constants: 0, arena: 2, tables: 0 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
    Stage { deployment: "ref2va dense (tile 128)", stage: "latent_encoder", weights: 512, constants: 0, arena: 2, tables: 0 },
    Stage { deployment: "t2va + DMD student", stage: "text_encoder", weights: 49121, constants: 0, arena: 2, tables: 7 },
    Stage { deployment: "t2va + DMD student", stage: "denoiser (student + refiner)", weights: 41953, constants: 343, arena: 1470, tables: 12 },
    Stage { deployment: "t2va + DMD student", stage: "video_decoder", weights: 4618, constants: 0, arena: 7757, tables: 3 },
    Stage { deployment: "t2va + DMD student", stage: "audio_decoder", weights: 123, constants: 0, arena: 2, tables: 0 },
    Stage { deployment: "t2va + DMD student", stage: "video_postprocessor (splice)", weights: 0, constants: 0, arena: 148, tables: 0 },
    Stage { deployment: "t2va dense, TP2", stage: "text_encoder", weights: 25129, constants: 0, arena: 1, tables: 7 },
    Stage { deployment: "t2va dense, TP2", stage: "denoiser (base DiT + refiner)", weights: 21299, constants: 1946, arena: 735, tables: 12 },
    Stage { deployment: "t2va dense, TP2", stage: "video_decoder", weights: 4618, constants: 0, arena: 7757, tables: 3 },
    Stage { deployment: "t2va dense, TP2", stage: "audio_decoder", weights: 123, constants: 0, arena: 2, tables: 0 },
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
        println!("largest stage: {} of weights + constants", mib(largest));
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
    let tile = minimax_h3::video_vae::Canvas::clip(&config.video_vae, 16, 16);
    let one: u64 = minimax_h3::decoding::video_arena(&config.video_vae, &tile)?
        .iter()
        .map(|buffer| buffer.elements * if buffer.element == "f16" { 2 } else { 4 })
        .sum();
    println!("one tile: {}", mib(one / (1 << 20)));
    Ok(())
}
