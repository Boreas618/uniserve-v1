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

use std::collections::BTreeMap;

use nn::weights::WeightLoader;
use nn::{Arena, DType, Forward, Shape};

use minimax_h3::VideoVaeConfig;
use minimax_h3::decoding::{Buffer, arena_reservation};
use minimax_h3::encoding::{Extent, conditioning_arena, video_encoder_run_arena};

const USAGE: &str = "\
usage: encoder_plan [--root <checkpoint>] [--frames <n>] [--height <n>] [--width <n>] [--extents <json>]

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
    let mut extents_path: Option<PathBuf> = None;
    let mut arena_bytes: Option<u64> = None;
    let mut check = false;
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
            "--extents" => extents_path = Some(PathBuf::from(value()?)),
            "--check" => check = true,
            "--arena" => {
                arena_bytes = Some(
                    value()?
                        .parse()
                        .map_err(|_| "--arena takes bytes".to_owned())?,
                )
            }
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
        "\nthe tile plan alone reserves {} B; the conditioning {} B; the harness's own buffers {} B",
        arena_reservation(&tile_plan),
        arena_reservation(&conditioning),
        reserved - arena_reservation(&tile_plan) - arena_reservation(&conditioning)
    );

    // What the **forward** asks, measured on the host: a recording forward launches nothing, and the
    // arena is bookkeeping over an address nothing dereferences, so every allocation the encode would
    // make is made here and `used()` counts it. The reservation the plan states is compared against
    // that, which is the only way a short plan can be named rather than inferred from a device failure.
    if let Some(path) = &extents_path {
        let table = weight_extents(path)?;
        let mut loader = WeightLoader::stated(DType::F32, Some(&table));
        let encoder = minimax_h3::encoding::VideoEncoder::load(&config, &mut loader)
            .map_err(|error| error.to_string())?;
        let arena = Arena::new(ops::DevicePtr::new(0), arena_bytes.unwrap_or(1_u64 << 42));
        let forward = Forward::recording(&arena);

        let before = arena.used();
        let scratch = encoder
            .reserve(&forward, &extents)
            .map_err(|error| error.to_string())?;
        let set_bytes = arena.used() - before;
        println!(
            "\ntrace: `reserve` asks {set_bytes} B for {} distinct extent(s)",
            {
                let mut distinct = extents.clone();
                distinct.sort_by_key(|extent| (extent.frames, extent.height, extent.width));
                distinct.dedup_by_key(|extent| (extent.frames, extent.height, extent.width));
                distinct.len()
            }
        );

        // One tile, exactly as the driver runs it: the caller's pixels and the latent it reads back
        // come out of this arena here because the harness has one pointer to give.
        let tile = extents[0];
        let pixels = arena
            .tensor(
                DType::F32,
                Shape::new(&[
                    1,
                    u64::from(config.in_channels),
                    u64::from(tile.frames),
                    u64::from(tile.height),
                    u64::from(tile.width),
                ]),
            )
            .map_err(|error| error.to_string())?;
        let latent = minimax_h3::encoding::Stack::of(&config)
            .map_err(|error| error.to_string())?
            .clip
            .latent_frames(tile.frames)
            .map_err(|error| error.to_string())?;
        let out = arena
            .tensor(
                DType::F32,
                Shape::new(&[
                    1,
                    2 * u64::from(config.latent_channels),
                    u64::from(latent),
                    u64::from(tile.height / 16),
                    u64::from(tile.width / 16),
                ]),
            )
            .map_err(|error| error.to_string())?;
        let before_tile = arena.used();
        encoder
            .encode_tile(&forward, &scratch, pixels, out)
            .map_err(|error| error.to_string())?;
        let tile_bytes = arena.used() - before_tile;
        println!(
            "trace: one tile asks {tile_bytes} B beyond the run's set; {} tile(s) in the grid is {} B",
            extents.len(),
            tile_bytes * u64::try_from(extents.len()).unwrap_or(1)
        );
        let asked = set_bytes + tile_bytes * u64::try_from(extents.len()).unwrap_or(1);
        println!(
            "trace: the forward asks {asked} B against the plan's {reserved} B - {}",
            match asked.cmp(&reserved) {
                std::cmp::Ordering::Greater => format!("SHORT by {} B", asked - reserved),
                std::cmp::Ordering::Equal => "exactly the plan".to_owned(),
                std::cmp::Ordering::Less => format!("{} B of slack", reserved - asked),
            }
        );
        // The set is the encoder's own share: the caller's pixels and latent are planned beside it, so
        // the comparison is `reserve` against the tile plan alone. A plan is an upper bound - every
        // buffer's bytes are rounded up to the arena's alignment and the last one's trailing pad is
        // never read - so up to `ALIGNMENT - 1` bytes of slack are expected. Anything shorter is a
        // request the plan cannot fund, and any allocation a tile makes beyond the set is the class of
        // defect the plan exists to prevent.
        let set_plan = arena_reservation(&tile_plan);
        if set_bytes > set_plan {
            return Err(format!(
                "`reserve` asks {set_bytes} B of a {set_plan}-byte tile plan: {} B of buffers the plan does not state",
                set_bytes - set_plan
            ));
        }
        if tile_bytes != 0 {
            return Err(format!(
                "one tile allocates {tile_bytes} B beyond the set it was reserved: a layer is taking buffers from the arena inside a forward"
            ));
        }
        if check && set_plan - set_bytes > nn::forward::ALIGNMENT - 1 {
            return Err(format!(
                "the tile plan reserves {} B more than `reserve` takes, past the {} B an alignment step explains: a buffer nothing reads",
                set_plan - set_bytes,
                nn::forward::ALIGNMENT - 1
            ));
        }
        let _ = &asked;
    }

    println!(
        "\n{} buffer(s) in the whole plan, {total} B raw, {reserved} B at the arena's alignment",
        plan.len()
    );
    println!(
        "the region a run allocates is this plan and the requests are the forward's, so the two are one\n\
         list: `--extents` measures them against each other, and `--check` exits non-zero when they\n\
         disagree - a plan short by any request, or a tile that allocates inside its forward."
    );
    Ok(())
}

/// The checkpoint's tensor extents for the video VAE, from the reference's own table.
fn weight_extents(path: &std::path::Path) -> Result<BTreeMap<String, (DType, Shape)>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let table = values.get("vae").unwrap_or(&values);
    let object = table
        .as_object()
        .ok_or_else(|| format!("{} is not an object", path.display()))?;
    let mut extents = BTreeMap::new();
    for (name, entry) in object {
        let dtype = match entry.get("dtype").and_then(|value| value.as_str()) {
            Some("F32") => DType::F32,
            Some("F16") => DType::F16,
            Some("BF16") => DType::BF16,
            other => return Err(format!("{name}: {other:?} is not an element type")),
        };
        let dims = entry
            .get("shape")
            .and_then(|value| value.as_array())
            .ok_or_else(|| format!("{name}: the entry states no shape"))?
            .iter()
            .map(|dim| {
                dim.as_u64()
                    .ok_or_else(|| format!("{name}: a dimension is not a count"))
            })
            .collect::<Result<Vec<u64>, String>>()?;
        extents.insert(name.clone(), (dtype, Shape::new(&dims)));
    }
    Ok(extents)
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
