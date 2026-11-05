//! Price the video encoder's arena from the plan's own functions, with no device and no weights.
//!
//! ```text
//! cargo run -p minimax-h3 --example encoder_plan -- \
//!     --root /opt/uv/models/MiniMax-H3 --frames 17 --height 768 --width 768
//! ```
//!
//! ## Why this exists
//!
//! `vae_decode --stage raster` reserves `video_encoder_run_arena` plus the conditioning plan plus its
//! own two buffers, and a device run asked for more than that: `323767296 bytes requested at offset
//! 4437654016 of a 4500302336-byte region` at 17 frames of 768x768, 16 tiles. The region is the plan;
//! the requests are the forward's. **A plan that does not read what the forward asks cannot be
//! audited from the failure alone**, because the arena's message names the request that did not fit
//! and not the buffer the plan never counted.
//!
//! So this prints the plan entry by entry - name, element count and bytes, then the aligned
//! reservation - and the reader compares the sum against what the run asked for. It launches nothing
//! and loads nothing: the plan functions are arithmetic over the checkpoint's config.

use std::path::PathBuf;

use minimax_h3::VideoVaeConfig;
use minimax_h3::decoding::{Buffer, arena_reservation};
use minimax_h3::encoding::{Extent, conditioning_arena, video_encoder_run_arena};

const USAGE: &str = "\
usage: encoder_plan [--root <checkpoint>] [--frames <n>] [--height <n>] [--width <n>]

  --root    the checkpoint's small files (default /opt/uv/models/MiniMax-H3)
  --frames  the clip's raster frames (default 17, the decoder's clip body)
  --height  the raster's pixel height (default 768)
  --width   the raster's pixel width (default 768)";

fn main() {
    if let Err(report) = run() {
        eprintln!("{report}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut root = PathBuf::from("/opt/uv/models/MiniMax-H3");
    let mut frames = 17_u32;
    let mut height = 768_u32;
    let mut width = 768_u32;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} takes a value\n{USAGE}"))
        };
        match argument.as_str() {
            "--root" => root = PathBuf::from(value()?),
            "--frames" => frames = count(&value()?, "--frames")?,
            "--height" => height = count(&value()?, "--height")?,
            "--width" => width = count(&value()?, "--width")?,
            "--help" | "-h" => {
                println!("{USAGE}");
                return Ok(());
            }
            other => return Err(format!("{other} is not an argument\n{USAGE}")),
        }
    }

    let config = video_config(&root)?;

    // The tiles the encoder's own grid splits the raster into - the same grid the driver executes, so
    // the extents here are the ones a run reserves. A hand-cut guess at the tile size gets a different
    // count and therefore a different plan.
    let clip = minimax_h3::tiled_encode::ClipEncoder::new(height, width)
        .map_err(|error| error.to_string())?;
    let plan = clip.plan();
    let extents: Vec<Extent> = plan.tile_extents(frames);
    println!(
        "raster {frames} frames of {height} x {width} -> {} tile(s), {} distinct extent(s)",
        extents.len(),
        {
            let mut distinct = extents.clone();
            distinct.sort_by_key(|extent| (extent.frames, extent.height, extent.width));
            distinct.dedup_by_key(|extent| (extent.frames, extent.height, extent.width));
            distinct.len()
        }
    );

    let tile_plan =
        video_encoder_run_arena(&config, &extents).map_err(|error| error.to_string())?;
    // The conditioning runs once per request over the *latent* grid.
    let latent_frames = minimax_h3::encoding::Stack::of(&config)
        .map_err(|error| error.to_string())?
        .clip
        .latent_frames(frames)
        .map_err(|error| error.to_string())?;
    let latent = Extent {
        frames: latent_frames,
        height: height / 16,
        width: width / 16,
    };
    let conditioning = conditioning_arena(&config, frames, latent).map_err(|e| e.to_string())?;

    let mut plan: Vec<Buffer> = tile_plan.clone();
    plan.extend(conditioning.clone());
    // The harness's own buffers, which the model's plan deliberately does not state: one pixel buffer
    // per tile, because the scene reads every tile's pixels in one call, and the latent it reads back.
    for extent in &extents {
        plan.push(Buffer {
            name: "raster pixels".to_owned(),
            element: "f32",
            elements: u64::from(minimax_h3::output::Rgb::CHANNELS)
                * u64::from(extent.frames)
                * u64::from(extent.height)
                * u64::from(extent.width),
        });
    }
    plan.push(Buffer {
        name: "raster latent".to_owned(),
        element: "f32",
        elements: u64::from(config.latent_channels)
            * u64::from(latent.frames)
            * u64::from(latent.height)
            * u64::from(latent.width),
    });

    let mut total = 0_u64;
    for (label, part) in [("tile", &tile_plan), ("conditioning", &conditioning)] {
        println!("\n{label} plan: {} buffer(s)", part.len());
        for buffer in part.iter() {
            let bytes = buffer.bytes();
            total += bytes;
            println!(
                "  {:32} {:>14} elements {:>12} B",
                buffer.name, buffer.elements, bytes
            );
        }
    }
    let reserved = arena_reservation(&plan);
    println!(
        "\n{} buffer(s) in the whole plan, {total} B raw, {reserved} B at the arena's alignment",
        plan.len()
    );
    println!(
        "a device run at this canvas reserved 4500302336 B for 54 buffer(s) and then asked for\n\
         4761421312 B: the region is this plan and the requests are the forward's, so the shortfall\n\
         is {} B of requests this list does not state",
        4761421312_u64.saturating_sub(reserved)
    );
    Ok(())
}

fn count(text: &str, flag: &str) -> Result<u32, String> {
    text.parse()
        .map_err(|_| format!("{flag} {text} is not a count"))
}

fn video_config(root: &std::path::Path) -> Result<VideoVaeConfig, String> {
    let path = root.join("vae/config.json");
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    VideoVaeConfig::parse(&values).map_err(|error| error.to_string())
}
