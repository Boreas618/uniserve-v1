//! Price the audio decoder's arena, and check that its plan is the forward's request list.
//!
//! ```text
//! cargo run -p minimax-h3 --example audio_plan -- \
//!     --root /opt/uv/models/MiniMax-H3 --extents artifacts/h3-reference/component-tensors.json
//! ```
//!
//! The same instrument the video encoder and the video decoder have: a recording forward launches
//! nothing, and an arena is bookkeeping over an address nothing dereferences, so the whole decode runs
//! on a host with no device and `Arena::used()` counts what it took.
//!
//! **Two numbers matter and they are different questions.** Sizing the arena to *exactly* the plan's
//! bytes and completing proves the plan covers every request the forward makes - a plan short by one
//! buffer fails here, by name, naming the request's size. Comparing `used()` with the plan after the
//! fact then says whether the plan is exact or reserves something nothing takes; slack is not a
//! defect, but a number a runner sizes storage from should be a number the forward fills.

use std::collections::BTreeMap;
use std::path::PathBuf;

use nn::weights::WeightLoader;
use nn::{Arena, DType, Forward, Shape};

use minimax_h3::audio_vae::{BigVgan, audio_arena, track_samples};
use minimax_h3::decoding::arena_reservation;
use minimax_h3::encoding::{AudioEncoder, audio_encoder_arena};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let mut root = PathBuf::from("/opt/uv/models/MiniMax-H3");
    let mut extents_path: Option<PathBuf> = None;
    let mut lengths: Vec<u32> = vec![582, 1164, 3492];
    let mut check = false;
    while let Some(flag) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "--root" => root = PathBuf::from(value()?),
            "--extents" => extents_path = Some(PathBuf::from(value()?)),
            "--check" => check = true,
            "--latent-frames" => {
                lengths = value()?
                    .split(',')
                    .map(|text| {
                        text.trim()
                            .parse::<u32>()
                            .map_err(|_| format!("{text} is not a latent frame count"))
                    })
                    .collect::<Result<Vec<u32>, String>>()?
            }
            other => return Err(format!("`{other}` is not a flag of this tool")),
        }
    }
    let path = extents_path.ok_or_else(|| {
        format!(
            "--extents is required: the weight-normalized convolutions are folded at load and their shapes are the checkpoint's. The component's table is {}",
            root.join("audio_vae/config.json").display()
        )
    })?;
    let extents = weight_extents(&path)?;

    let values = config_values(&root)?;
    let config = minimax_h3::AudioVaeConfig::parse(&values).map_err(|error| error.to_string())?;
    let mut exact = 0usize;
    let mut slack = 0i128;
    for length in lengths {
        let plan = audio_arena(&config, length).map_err(|error| error.to_string())?;
        let reserved = arena_reservation(&plan);
        // The forward's own buffers come out of a second region: the plan is what the decode may take,
        // not what the caller holds.
        let arena = Arena::new(ops::DevicePtr::new(0), reserved);
        let caller = Arena::new(ops::DevicePtr::new(0), 1_u64 << 40);
        let f = Forward::recording(&arena);
        let mut loader = WeightLoader::stated(DType::F32, Some(&extents));
        let decoder = BigVgan::load(&config, &mut loader).map_err(|error| error.to_string())?;
        let latents = caller
            .tensor(
                DType::F32,
                Shape::new(&[2, u64::from(config.latent_channels), u64::from(length)]),
            )
            .map_err(|error| error.to_string())?;
        let samples = track_samples(&config, length);
        // The interleave writes the wire format, so the decode's last view is i16 rather than FP32.
        let out = caller
            .tensor(DType::I16, Shape::new(&[samples, 2]))
            .map_err(|error| error.to_string())?;
        decoder
            .forward(&f, latents, out)
            .map_err(|error| format!("{length} latent frames: {error}"))?;
        let used = arena.used();
        if used > reserved {
            return Err(format!(
                "{length} latent frames: the decode took {used} B of a {reserved}-byte plan"
            ));
        }
        exact += usize::from(used == reserved);
        if check && reserved - used > nn::forward::ALIGNMENT - 1 {
            return Err(format!(
                "the plan reserves {} B more than the encode takes, past the {} B an alignment step explains",
                reserved - used,
                nn::forward::ALIGNMENT - 1
            ));
        }
        if check && reserved - used > nn::forward::ALIGNMENT - 1 {
            return Err(format!(
                "the plan reserves {} B more than the decode takes, past the {} B an alignment step explains - a buffer nothing reads",
                reserved - used,
                nn::forward::ALIGNMENT - 1
            ));
        }
        slack += i128::try_from(reserved - used).unwrap_or(i128::MAX);
        println!(
            "{length:>5} latent frames ({samples} samples): {} buffer(s), {reserved} B planned, {used} B taken, {}",
            plan.len(),
            match reserved.cmp(&used) {
                std::cmp::Ordering::Equal => "exact".to_owned(),
                std::cmp::Ordering::Greater => format!("{} B of slack", reserved - used),
                std::cmp::Ordering::Less => unreachable!(),
            }
        );
    }
    println!(
        "# {} of the lengths decodes exactly its plan; {slack} B of slack over all of them",
        exact
    );
    println!(
        "# a plan the forward fills is a number a runner may size storage from; slack is headroom nobody uses."
    );
    println!(
        "# Up to {} B of slack is expected: each buffer's bytes are rounded up to the arena's {}-byte alignment, and the last buffer's trailing pad is never read.",
        nn::forward::ALIGNMENT - 1,
        nn::forward::ALIGNMENT
    );

    // The encoder half of the same question, since both plans are this crate's: it reads a padded
    // track and writes one latent frame's moments per hop.
    let mut exact = 0usize;
    for track in [362_u64, 725] {
        let mut loader = WeightLoader::stated(DType::F32, Some(&extents));
        let encoder =
            AudioEncoder::load(&config, &mut loader).map_err(|error| error.to_string())?;
        let length = minimax_h3::encoding::padded_samples(track, encoder.hop())
            .map_err(|e| e.to_string())?;
        // The encoder reads the stereo pair the wire carries, two identical rows for a mono track.
        let plan = audio_encoder_arena(&config, 2, length).map_err(|error| error.to_string())?;
        let reserved = arena_reservation(&plan);
        let arena = Arena::new(ops::DevicePtr::new(0), reserved);
        let caller = Arena::new(ops::DevicePtr::new(0), 1_u64 << 40);
        let f = Forward::recording(&arena);
        let samples = caller
            .tensor(DType::F32, Shape::new(&[2, 1, length]))
            .map_err(|error| error.to_string())?;
        let frames = encoder.frames(length).map_err(|error| error.to_string())?;
        // The latent is the stereo pair's rows, channel-major: `[batch * frames, latent_channels]`,
        // which is the form the DiT's audio conditioning reads.
        let out = caller
            .tensor(
                DType::F32,
                Shape::new(&[2 * frames, u64::from(config.latent_channels)]),
            )
            .map_err(|error| error.to_string())?;
        encoder
            .encode(&f, samples, out)
            .map_err(|error| format!("audio encoder track {track}: {error}"))?;
        let used = arena.used();
        if used > reserved {
            return Err(format!(
                "audio encoder track {track}: the encode took {used} B of a {reserved}-byte plan"
            ));
        }
        exact += usize::from(used == reserved);
        println!(
            "audio encoder, {track} samples ({length} padded): {} buffer(s), {reserved} B planned, {used} B taken, {}",
            plan.len(),
            match reserved.cmp(&used) {
                std::cmp::Ordering::Equal => "exact".to_owned(),
                std::cmp::Ordering::Greater => format!("{} B of slack", reserved - used),
                std::cmp::Ordering::Less => unreachable!(),
            }
        );
    }
    println!("# {exact} of the two encodes fills its plan exactly");
    Ok(())
}

/// The checkpoint's tensor extents for the audio VAE, from the reference's own table.
fn weight_extents(path: &std::path::Path) -> Result<BTreeMap<String, (DType, Shape)>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let table = values.get("audio_vae").unwrap_or(&values);
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

/// The checkpoint's configuration values, as `AudioVaeConfig::parse` reads them.
fn config_values(root: &std::path::Path) -> Result<serde_json::Value, String> {
    let path = root.join("audio_vae/config.json");
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
}
