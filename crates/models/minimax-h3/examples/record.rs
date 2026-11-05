//! A model's programs, recorded from the model's own `forward`.
//!
//! ```text
//! cargo run -p minimax-h3 --example record -- --root /opt/uv/models/MiniMax-H3 --component all \
//!     --clip 260 --raster 672x1536 --extents h3-audio-extents.json \
//!     --against notes/programs/minimax-h3-keys.txt
//! ```
//!
//! ## Why this exists beside `examples/programs.rs`
//!
//! `programs.rs` enumerates a model's programs in a traversal written by hand beside each `forward`, and on one
//! day that traversal was wrong twice and incomplete once (`silu`'s element type, `act_and_mul`'s packed
//! stride, and four `scaled_add` crossings of the sample/FP32 boundary that no row stated). Each was found by
//! comparing it with a second implementation, none by a test, and the missing four were invisible to
//! `key_coverage`: a row neither the fixture nor the store holds answers "every key is covered".
//!
//! This drives the model's **own** code instead: `WeightLoader::shapes` states every weight's extents without
//! reading a byte, `Forward::recording` reports each launch to the capture sink instead of launching it, and the
//! component's `forward` runs unchanged. No shard, no device. The row set is what the model launches, by
//! construction, and the hand-written walk becomes the thing being checked rather than the source.
//!
//! ## What a refusal is
//!
//! A harness that cannot run prints its name and its reason and the recorder **keeps going**: a recorder that
//! stops at the first refusal reports a small number that looks exactly like agreement, which is the failure
//! this file exists to remove. The summary line names every refusal.
//!
//! ## What the output is
//!
//! The header states the **deployment** it recorded — element type, clip length, raster, packed rows, and the
//! attention backend the checkpoint selects — before any row, because a recording that does not say what it is a
//! recording of is the same object as a fixture that carries more than the walk.
//!
//! ## The extents table
//!
//! A weight a module merely reads takes its extents from the request it asks with. A weight **computed at
//! load** does not: the folded form of a weight-normalized convolution is `v · g / ‖v‖`, and `v`'s extents are a
//! fact of the checkpoint's shard header. `--extents` supplies them as `{"tensor": {"dtype": "F32", "shape":
//! [..]}}`, produced where the weights are (`--list` prints the names the load asks for) and used where they are
//! not.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use minimax_h3::audio_vae::BigVgan;
use minimax_h3::checkpoint::Dit;
use minimax_h3::denoiser::{Arrangement, Denoiser, Layouts};
use minimax_h3::encoding::{AudioEncoder, VideoEncoder};
use minimax_h3::video_vae::Tiles;
// The capability trait, for the request's noise draw: it is a method of the trait, not of the struct.
use minimax_h3::AttentionKind;
use minimax_h3::Partition;
use minimax_h3::output::{Splice, VideoPostprocessor};
use minimax_h3::text_encoder::TextEncoder;
use minimax_h3::video_vae::{Canvas, VideoDecoder};
use nn::capability::Denoiser as _;
use nn::capability::{
    ConditionRole, ConditionShape, Encoder, EncoderInputs, EncoderOutputs, ImageInput, ImageShape,
    TokenBatch, VideoRequest, VideoTask,
};
use nn::{PhaseKind, TensorView};

use nn::forward::Arena;
use nn::layers::{Linear, WeightOrder};
use nn::tensor::DevicePtr;
use nn::weights::WeightLoader;
use nn::weights::WeightRequest;
use nn::{DType, Forward, Shape};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(problem) => {
            eprintln!("error: {problem}");
            ExitCode::FAILURE
        }
    }
}

/// One component's harness: its rows, or the reason it could not run.
type Harness = fn(&Request) -> Result<Vec<(String, String)>, String>;

/// What the recorder was asked to record — the deployment, stated so the output can say what it is.
#[derive(Clone)]
struct Request {
    root: PathBuf,
    clip: u32,
    raster: (u32, u32),
    /// **Every canvas the recording claims.** One canvas certifies one canvas: the encoder's pad and
    /// reduction classes are derived from a frame's extent, and a harness that drives the largest
    /// raster alone leaves the smallest classes unexercised - and the classes the model launches at
    /// them invisible. The first entry is `raster`, which the header and the other harnesses read.
    rasters: Vec<(u32, u32)>,
    /// The deployment's own bounded packed-row count (`Layouts::rows_bound`), which serving reads and
    /// the attention is compiled for. Derived once here so the header and the harness cannot differ.
    bound: u32,
    /// The deployment's prompt capacity: the arrangement's text rows and the request's own text
    /// length, which key the packed stream's offsets.
    tokens: u32,
    /// The deployment's audio track length in samples, which keys the encoder's rows.
    track: u32,
    /// The trajectory's element count as the deployment states it. No row is keyed by it - the
    /// solver's residual is one column - so it is carried into the deployment line only, where a
    /// reader comparing two deployments sees what each claimed.
    elements: u64,
    /// The caller's tensor table, when one was given: the four components' stored element types.
    types: Option<PathBuf>,
    extents: BTreeMap<String, (DType, Shape)>,
    /// The sparse deployment's served tile and keep rate, absent for a dense one.
    sparse: Option<(u32, f64, Option<f64>)>,
}

/// The rows a recording pass collected, in the order the model launched them.
#[derive(Default)]
struct Recorded(Vec<(String, String)>);

impl ops::capture::Sink for Recorded {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.push((operator.to_owned(), key.to_owned()));
    }
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let mut root = PathBuf::from("/opt/uv/models/MiniMax-H3");
    let mut component = String::from("all");
    // A clip is `17n + 5` frames (the packing's rule, enforced below) and its soundtrack follows from it:
    // 260 frames are 433 audio latent frames of 800 samples, so the two defaults are one deployment.
    let mut clip = 260u32;
    // The first canvas the deployment offers: every harness that reads one raster (the denoiser's,
    // the decoder's) reads this one, and the others are added by `--raster`.
    let mut raster = {
        let first = minimax_h3::config::offered_canvases()
            .first()
            .map(|canvas| (canvas.height, canvas.width));
        first.unwrap_or((672, 1536))
    };
    // **One canvas by default, and every `--raster` given is claimed.** A recorder over one canvas
    // certifies one canvas, so the output states which ones it drove; a caller that wants the
    // deployment's extremes passes both, and the classes each one reaches are then visible.
    let mut claimed: Vec<(u32, u32)> = Vec::new();
    // No default: the bounded count is the deployment's own (`Layouts::rows_bound`) and serving
    // reads it. A stated value is a claim the denoiser harness checks against it.
    let mut rows: Option<u32> = None;
    // The deployment's prompt capacity and audio track, which the walk states as `--tokens` and
    // `--track`. They key the packed stream's offsets and the audio encoder's widths, so a
    // comparison of two deploments must pass the same pair to both enumerations.
    let mut tokens = 4096u32;
    let mut track = 346_400u32;
    let mut elements = 0u64;
    let mut against: Option<PathBuf> = None;
    let mut extents: Option<PathBuf> = None;
    let mut types: Option<PathBuf> = None;
    let mut sparse: Option<(u32, f64, Option<f64>)> = None;
    // Which deployment this recording states: the walk's own flag and the walk's own default.
    let mut conditioned_deployment = true;
    let mut list = false;
    let mut names = false;
    while let Some(argument) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{argument} needs a value"))
        };
        match argument.as_str() {
            "--root" => root = PathBuf::from(value()?),
            "--component" => component = value()?,
            "--clip" => {
                let frames: u32 = value()?.parse().map_err(|_| "clip is a count of frames")?;
                // A clip is `17n + 5` frames and the packing accepts no other, so a recorder driven at
                // one would record a deployment the sampler cannot be handed. The rule is
                // `decoding::clips`' and this calls it rather than restating it: the walk refuses by
                // the same function, which is what keeps the two enumerations at one invocation.
                minimax_h3::decoding::clips(frames).map_err(|error| error.to_string())?;
                clip = frames;
            }
            "--raster" => {
                let stated = value()?;
                // **The walk's own order: height first.** `programs.rs --rasters` splits `HxW`, and
                // the first axis here is the one the pixel tensor's rows are (`--raster`'s message
                // said the opposite for a while, which is how a portrait canvas got recorded against
                // a landscape walk).
                let (h, w) = stated
                    .split_once('x')
                    .ok_or("--raster is <height>x<width>, as the walk's --rasters")?;
                let pair = (
                    w.parse().map_err(|_| "--raster's width is a count")?,
                    h.parse().map_err(|_| "--raster's height is a count")?,
                );
                let pair = (pair.1, pair.0);
                claimed.push(pair);
                if claimed.len() == 1 {
                    raster = pair;
                }
            }
            "--rows" => rows = Some(value()?.parse().map_err(|_| "--rows is a count")?),
            "--tokens" => tokens = value()?.parse().map_err(|_| "--tokens is a count")?,
            "--track" => {
                track = value()?
                    .parse()
                    .map_err(|_| "--track is a count of samples")?
            }
            "--elements" => elements = value()?.parse().map_err(|_| "--elements is a count")?,
            "--against" => against = Some(PathBuf::from(value()?)),
            "--conditioned" => {
                conditioned_deployment = match value()?.as_str() {
                    "true" => true,
                    "false" => false,
                    other => {
                        return Err(format!("--conditioned takes true or false, not {other}"));
                    }
                };
            }
            "--extents" => extents = Some(PathBuf::from(value()?)),
            // The caller's `component-tensors.json`: the four components' stored element types, which
            // no shardless host can read for itself.
            "--types" => types = Some(PathBuf::from(value()?)),
            // `tile,sparsity[,reference_keep]`: the bound the compression reduces over is the
            // deployment's own, computed from its layouts at load, so it is not a flag here.
            "--sparse" => {
                let stated = value()?;
                let parts: Vec<&str> = stated.split(',').collect();
                if parts.len() != 2 && parts.len() != 3 {
                    return Err(
                        "--sparse is `tile,sparsity[,reference_keep]` here: the recorder derives the \
                         tile bound (`Layouts::tiles_bound`) as the deployment's own, so unlike the \
                         walk's `tile,sparsity,tiles[,reference_keep]` there is no tile count to \
                         state - and a third value is the keep rate, which must lie in (0, 1]"
                            .to_owned(),
                    );
                }
                sparse = Some((
                    parts[0]
                        .trim()
                        .parse()
                        .map_err(|_| "--sparse's tile is a row count")?,
                    parts[1]
                        .trim()
                        .parse()
                        .map_err(|_| "--sparse's sparsity is a fraction")?,
                    match parts.get(2) {
                        Some(keep) => Some(
                            keep.trim()
                                .parse()
                                .map_err(|_| "--sparse's keep rate is a fraction")?,
                        ),
                        None => None,
                    },
                ));
            }
            "--list" => list = true,
            "--names" => names = true,
            "--check-extents" => {
                return check_extents(&root).map_err(|error| error.to_string());
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }

    if names {
        return list_names(&root, &component);
    }
    if list {
        return list_extents(&root, &component);
    }

    // The caller's tensor table is an artifact of an export, not a file in the tree, so it is stated
    // when it is at hand and said to be unstated when it is not: requiring it would make a recording
    // refuse on a host that simply has not exported one.
    let table = match extents.as_deref() {
        Some(path) => read_extents(path)?,
        None => BTreeMap::new(),
    };
    let bound = deployment_bound(&root, sparse)?;
    if let Some(stated) = rows
        && stated != bound
    {
        return Err(format!(
            "--rows states {stated} packed rows and the deployment's own bound is {bound}: the \
             attention is compiled for the bound it serves, and a recording of a smaller one \
             describes a deployment the model cannot be asked to be"
        ));
    }
    let request = Request {
        root,
        bound,
        clip,
        raster,
        rasters: {
            // The deployment's own canvases, and a stated one must be among them: recording a canvas
            // the engine cannot be asked for certifies a class no request reaches.
            let offered: Vec<(u32, u32)> = minimax_h3::config::offered_canvases()
                .iter()
                .map(|canvas| (canvas.height, canvas.width))
                .collect();
            if claimed.is_empty() {
                offered
            } else {
                for pair in &claimed {
                    if !offered.contains(pair) {
                        return Err(format!(
                            "--raster {}x{} is not one of the deployment's offered canvases {offered:?}",
                            pair.0, pair.1
                        ));
                    }
                }
                claimed
            }
        },
        tokens,
        track,
        elements,
        types,
        extents: table,
        sparse,
    };

    let harnesses: Vec<(&str, Harness)> = match component.as_str() {
        // `all` is one *deployment*, not both: the plain denoiser runs a `t2va` step and the
        // conditioned one runs a `fl2va` step, so a row set holding both describes two deployments and
        // cannot be compared with either walk. `--conditioned` selects which, and it is the walk's own
        // flag with the walk's own default, because the comparison is only meaningful when the two
        // enumerations were invoked identically.
        "all" => {
            let mut harnesses: Vec<(&str, Harness)> = vec![
                ("audio-encoder", audio_encoder as Harness),
                ("audio-decoder", audio_decoder as Harness),
                ("video-encoder", video_encoder as Harness),
                ("video-decoder", video_decoder as Harness),
                ("video-postprocessor", video_postprocessor as Harness),
                ("text-encoder", text_encoder as Harness),
                ("vision-tower", vision_tower as Harness),
            ];
            if conditioned_deployment {
                harnesses.push(("conditioned", conditioned as Harness));
            } else {
                harnesses.push(("denoiser", denoiser as Harness));
            }
            harnesses
        }
        // The order above is the one a caller reads: the decoders, then the network, then the prompt's
        // path. `all` is every harness this build can drive, and the audio halves answer by name while
        // their extents table is missing rather than silently leaving the run short.
        "audio-encoder" => vec![("audio-encoder", audio_encoder as Harness)],
        "audio-decoder" => vec![("audio-decoder", audio_decoder as Harness)],
        "video-encoder" => vec![("video-encoder", video_encoder as Harness)],
        "video-decoder" => vec![("video-decoder", video_decoder as Harness)],
        "denoiser" => vec![("denoiser", denoiser as Harness)],
        "video-postprocessor" => vec![("video-postprocessor", video_postprocessor as Harness)],
        "conditioned" => vec![("conditioned", conditioned as Harness)],
        "text-encoder" => vec![("text-encoder", text_encoder as Harness)],
        "vision-tower" => vec![("vision-tower", vision_tower as Harness)],
        other => return Err(format!("unknown component {other}")),
    };

    // The row count is printed once every row is read, so no two readings of this print can disagree
    // about it: the rows go out deduplicated, in the order a fixture holds them, and the summary below
    // is the count of exactly what was printed.
    let mut all: Vec<(String, String)> = Vec::new();
    let mut refusals: Vec<(String, String)> = Vec::new();
    // **Every canvas the recording claims is driven, not only the first.** The walk states a
    // deployment's rows over each of its offered canvases, and a harness that reads the single
    // `raster` certifies one canvas while the header names six - the frame operators' classes are
    // derived from a frame's extent, so a mean canvas hides the extreme ones. Driving the harness
    // list once per raster is the one place the scope can be got wrong, so it is stated here rather
    // than in each harness: rows deduplicate, and a component that refuses on one canvas is named
    // with the canvas it refused on.
    // **The track is the clip's, not an independent quantity, and the model's own arithmetic says so.**
    // The walk derives `--track` from `--samples` (`packing::audio_latent_frames` then
    // `audio_vae::track_samples`) and refuses a stated one that disagrees, so a recorder stating its own
    // length records a deployment the walk cannot state: the audio encoder's residual widths and the
    // conditioned stream's offsets are both keyed by it, and the two enumerations are only comparable
    // when they were invoked identically. The rule is called here rather than restated.
    {
        let config =
            minimax_h3::Config::read(&request.root, None).map_err(|error| error.to_string())?;
        let latents = minimax_h3::packing::audio_latent_frames(request.clip);
        let implied = minimax_h3::audio_vae::track_samples(&config.audio_vae, latents);
        if u64::from(request.track) != implied {
            return Err(format!(
                "--track states {} samples and a {}-frame clip's soundtrack is {} samples \
                 ({} latent frames of {}): the track is the clip's, and the walk derives it",
                request.track,
                request.clip,
                implied,
                latents,
                minimax_h3::audio_vae::samples_per_latent(&config.audio_vae),
            ));
        }
    }

    let canvases = if request.rasters.is_empty() {
        vec![request.raster]
    } else {
        request.rasters.clone()
    };
    let many = canvases.len() > 1;
    for (height, width) in canvases {
        let per = Request {
            raster: (height, width),
            ..request.clone()
        };
        for (name, harness) in &harnesses {
            match harness(&per) {
                Ok(rows) if rows.is_empty() => {
                    // An empty record is not a pass: `0 only recorded` over a harness that drove
                    // nothing reads exactly like a deployment that launches nothing. A harness
                    // that needs an argument it was not given refuses by name above; one that
                    // returns no rows at all is refusing silently, and this is where it is named.
                    let why = format!(
                        "recorded no launch: nothing was driven for this harness, and an empty \
                         record cannot be read as a deployment that launches nothing{}",
                        if many {
                            format!(" (at {width}x{height})")
                        } else {
                            String::new()
                        }
                    );
                    println!("# refused: {name}: {why}");
                    eprintln!("# refused: {name}: {why}");
                    refusals.push((name.to_string(), why));
                }
                Ok(rows) => {
                    eprintln!("# {name} at {width}x{height}: {} rows", rows.len());
                    all.extend(rows);
                }
                Err(reason) => {
                    // A refusal is a row of the output, not an exit.
                    let why = if many {
                        format!("{reason} (at {width}x{height})")
                    } else {
                        reason
                    };
                    println!("# refused: {name}: {why}");
                    eprintln!("# refused: {name}: {why}");
                    refusals.push((name.to_string(), why));
                }
            }
        }
    }
    all.sort();
    all.dedup();
    // The deployment line is the one `h3_fixture` copies into a fixture, and it is the walk's own line
    // with this recorder's numbers in it: a print from here is a fixture's source on a machine with no
    // device and no shard, which is what makes the walk the second enumeration rather than the first.
    println!(
        "# MiniMax-H3 programs for these clip lengths: [{}], raster sizes: [{}], prompt lengths: [{}], \
         packed rows: [{}], trajectory elements: [{}], element types: {}, sparse attention: \
         {:?}, tensor-parallel degree 1, ulysses degree 1 - recorded from the model's own \
         forward at {}",
        request.clip,
        // **The canvases this recording claims**, not the one the other harnesses read: a count is a
        // statement about the scope beside it, and one canvas certifies one canvas.
        request
            .rasters
            .iter()
            .map(|(width, height)| format!("{width}x{height}"))
            .collect::<Vec<_>>()
            .join(", "),
        request.tokens,
        request.bound,
        request.elements,
        request.types.as_deref().map_or_else(
            || format!("[{}]", minimax_h3::load::UNSTATED),
            deployment_types,
        ),
        request.sparse,
        request.root.display()
    );
    for row in &all {
        println!("{}\t{}", row.0, row.1);
    }
    let mut operators: Vec<&str> = all.iter().map(|(name, _)| name.as_str()).collect();
    operators.sort_unstable();
    operators.dedup();
    // A component that refused on more than one canvas is one component refused: the count is of
    // components, and each refusal line above names the canvas it happened on.
    let mut refused_names: Vec<&str> = refusals.iter().map(|(name, _)| name.as_str()).collect();
    refused_names.sort_unstable();
    refused_names.dedup();
    eprintln!(
        "# {} data rows, {} operators, {} of {} components refused{}",
        all.len(),
        operators.len(),
        refused_names.len(),
        harnesses.len(),
        if refused_names.is_empty() {
            String::new()
        } else {
            format!(": {}", refused_names.join(", "))
        }
    );

    if let Some(walk) = against {
        compare(&all, &walk)?;
    }
    Ok(())
}

/// The rows recorded and the rows a walk states, in both directions.
///
/// `recorded, not stated` is the severe one: the walk is missing a program the model launches, which is the
/// failure that cost a caller run and four rows in one day. `stated, not recorded` is the other direction and is
/// its own defect, because a store then holds a cell nothing reads.
fn compare(recorded: &[(String, String)], walk: &Path) -> Result<(), String> {
    let stated =
        std::fs::read_to_string(walk).map_err(|error| format!("{}: {error}", walk.display()))?;
    let stated: std::collections::BTreeSet<(String, String)> = stated
        .lines()
        .filter(|line| !line.starts_with('#') && line.contains('\t'))
        .map(|line| {
            let (operator, key) = line.split_once('\t').unwrap_or((line, ""));
            (operator.to_owned(), key.to_owned())
        })
        .collect();
    let recorded: std::collections::BTreeSet<(String, String)> = recorded.iter().cloned().collect();
    let only_recorded: Vec<_> = recorded.difference(&stated).collect();
    let only_stated: Vec<_> = stated.difference(&recorded).collect();
    for (operator, key) in &only_recorded {
        println!("recorded, not stated:\t{operator}\t{key}");
    }
    for (operator, key) in &only_stated {
        println!("stated, not recorded:\t{operator}\t{key}");
    }
    eprintln!(
        "# {} recorded, {} stated, {} only recorded, {} only stated",
        recorded.len(),
        stated.len(),
        only_recorded.len(),
        only_stated.len()
    );
    Ok(())
}

/// The audio VAE's encoder, as one request: a track of whole latent frames in FP32.
///
/// It is the first harness because its `forward` takes two views and nothing else — the track and the rows it
/// writes — so the operand construction a harness owes is visible here in full.
fn audio_encoder(request: &Request) -> Result<Vec<(String, String)>, String> {
    let config = audio_config(&request.root)?;

    // The shapes-only loader: no checkpoint, no shard, no device. A load that reads bytes refuses by name,
    // and a computed weight is answered from the extents table.
    let mut loader = WeightLoader::stated(DType::F32, Some(&request.extents));
    let encoder = AudioEncoder::load(&config, &mut loader).map_err(|error| error.to_string())?;

    // The arena is a null-backed region of a size no reservation will reach; nothing reads it.
    let arena = Arena::new(DevicePtr::new(0), 1 << 40);
    let forward = Forward::recording(&arena);

    // The deployment's own track length, so the widths this harness records are the ones the walk
    // states for the same deployment rather than one window of 64 hops.
    // The deployment states its track's length; the network takes it right-padded to a whole number of
    // latent frames, which is the caller's own rule (`padded_samples`) and not a rounding here.
    let length = minimax_h3::encoding::padded_samples(u64::from(request.track), encoder.hop())
        .map_err(|error| error.to_string())?;
    // The encoder reads the stereo pair: two batch rows, one channel each, which is the shape
    // `unpack_samples` hands it after a mono track has been expanded to two identical channels.
    // One row encodes half a track of latent frames and every count downstream accepts it, so the
    // shape is stated here rather than left to the caller's convenience.
    let samples = arena
        .tensor(
            DType::F32,
            Shape::new(&[u64::from(minimax_h3::config::AUDIO_CHANNELS), 1, length]),
        )
        .map_err(|error| error.to_string())?;
    let frames = encoder.frames(length).map_err(|error| error.to_string())?;
    let channels = u64::from(config.latent_channels);
    // The latents are channel-major: both channels' frames as `2 * frames` rows, the stereo pair
    // named in the row count rather than a batch dimension.
    let out = arena
        .tensor(
            DType::F32,
            Shape::new(&[
                u64::from(minimax_h3::config::AUDIO_CHANNELS) * frames,
                channels,
            ]),
        )
        .map_err(|error| error.to_string())?;

    let mut recorded = Recorded::default();
    ops::capture::set_sink(Some(&mut recorded));
    let run = encoder.encode(&forward, samples, out);
    ops::capture::set_sink(None);
    run.map_err(|error| error.to_string())?;
    Ok(recorded.0)
}

/// The audio VAE's decoder, as one request: 64 latent frames in FP32, decoded to interleaved PCM.
///
/// The decoder's `forward` takes the latents `[2, latent_channels, frames]` and writes `[samples, 2]`, and every
/// intermediate is FP32 — the audio VAE's arithmetic is stated in that type whatever the deployment's sample
/// type is. The frames are the bucket a decode of a 64-frame window serves, which is the one `BigVgan`'s
/// own `programs` states its per-row launches at.
fn audio_decoder(request: &Request) -> Result<Vec<(String, String)>, String> {
    let config = audio_config(&request.root)?;

    // The same extents table the encoder needs: the decoder's stages carry weight-normalized convolutions too,
    // and a fold reads its direction tensor's extents rather than its own.
    let mut loader = WeightLoader::stated(DType::F32, Some(&request.extents));
    let decoder = BigVgan::load(&config, &mut loader).map_err(|error| error.to_string())?;

    let arena = Arena::new(DevicePtr::new(0), 1 << 40);
    let forward = Forward::recording(&arena);

    // A clip's video frame count is not the decoder's latent frame count: the audio track is packed at
    // its own rate, so the latent frames come from the packing's arithmetic and every stage's length
    // follows from them. Handing the clip's frames in as latent frames launches a ladder no decode
    // makes - the same quantity stated twice, which is what the walk and this harness must not do.
    let frames = minimax_h3::packing::audio_latent_frames(request.clip);
    let latent_channels = u64::from(config.latent_channels);
    let latents = arena
        .tensor(
            DType::F32,
            Shape::new(&[2, latent_channels, u64::from(frames)]),
        )
        .map_err(|error| error.to_string())?;
    let samples = minimax_h3::audio_vae::track_samples(&config, frames);
    // The interleave is the one operator whose output is the wire format: the track is PCM, so the decode's
    // last view is i16 rather than FP32 and the type is the op's own contract.
    let out = arena
        .tensor(DType::I16, Shape::new(&[samples, 2]))
        .map_err(|error| error.to_string())?;

    let mut recorded = Recorded::default();
    ops::capture::set_sink(Some(&mut recorded));
    let run = decoder.forward(&forward, latents, out);
    ops::capture::set_sink(None);
    run.map_err(|error| error.to_string())?;
    Ok(recorded.0)
}

/// The video VAE's encoder: one still, through the tile walk the encoder's own `encode` reserves.
///
/// Its weights are stored plainly — no weight-normalized pair — so this harness needs no extents table,
/// which is why it can run on a host with no shard while the audio VAE cannot.
fn video_encoder(request: &Request) -> Result<Vec<(String, String)>, String> {
    let config = video_config(&request.root)?;
    // The video VAE's encoder computes in FP32 at every deployment, so its loader states FP32: the
    // element type a module carries is the one its arithmetic runs in, not the sample's.
    let mut loader = WeightLoader::stated(DType::F32, Some(&request.extents));
    let encoder = VideoEncoder::load(&config, &mut loader).map_err(|error| error.to_string())?;

    let arena = Arena::new(DevicePtr::new(0), 1 << 40);
    let forward = Forward::recording(&arena);

    // `--raster <height>x<width>` is the walk's own convention (`programs.rs --rasters`), so the pixel
    // tensor's two spatial axes are the pair in that order.
    // **Both frame counts and every canvas the recording claims.** The encoder reads one still or one clip, and its
    // temporal stack's levels exist only for the clip: a harness that drives a single frame records
    // no launch of them, which is how the walk's `for frames in [1, clip.frames]` came to state
    // levels nothing confirmed. The stack states the clip's length.
    let stack = minimax_h3::encoding::Stack::of(&config).map_err(|error| error.to_string())?;
    let mut recorded = Recorded::default();
    for (height, width) in request.rasters.clone() {
        for frames in [1, stack.clip.frames] {
            let latency = encoder
                .latent_frames(frames)
                .map_err(|error| error.to_string())?;
            // The encoder's arithmetic is FP32 whatever the deployment's sample type is (`nn` refuses a
            // bias and an output that disagree, which is what a mixed operand reads as).
            let pixels = arena
                .tensor(
                    DType::F32,
                    Shape::new(&[1, 3, u64::from(frames), u64::from(height), u64::from(width)]),
                )
                .map_err(|error| error.to_string())?;
            let compression = u64::from(config.spatial_compression());
            let moments = arena
                .tensor(
                    DType::F32,
                    Shape::new(&[
                        2 * u64::from(config.latent_channels),
                        u64::from(latency),
                        u64::from(height) / compression,
                        u64::from(width) / compression,
                    ]),
                )
                .map_err(|error| error.to_string())?;

            ops::capture::set_sink(Some(&mut recorded));
            let run = encoder.encode(&forward, pixels, moments);
            ops::capture::set_sink(None);
            run.map_err(|error| error.to_string())?;
        }
    }
    Ok(recorded.0)
}

/// The video decoder's own forward, at one clipped latent raster.
///
/// The decoder takes its input projection as an operand rather than loading it — the model's patch
/// projection is a layer the runner owns — so the harness states one here: a shapes-only weight, whose
/// extents are the loader's to name and never a byte of a checkpoint.
fn video_decoder(request: &Request) -> Result<Vec<(String, String)>, String> {
    let config = video_config(&request.root)?;
    let mut loader = WeightLoader::stated(DType::F32, Some(&request.extents));
    let (height, width) = request.raster;
    // The latent raster of one decode call is the clip's own: the latent frames a clip decodes, and the
    // pixel raster over the VAE's spatial compression.
    let ratio = config.spatial_compression();
    let latent_height = height / ratio;
    let latent_width = width / ratio;
    let clip_canvas = Canvas::clip(&config, latent_height, latent_width);
    // **The decoder decodes one tile, not a clip.** Its scratch is sized for a tile's rows and every
    // buffer it reads is the tile's own canvas (`Tiles::tiles` states one per tile), so a harness that
    // hands it the clip's canvas asks for a decode of 10 920 rows into scratch sized for 1 792 - which
    // is exactly the refusal this harness reported before the tile existed here.
    let tiles = Tiles::of(&config, &clip_canvas).map_err(|error| error.to_string())?;
    let canvas = tiles
        .tiles(&config, &clip_canvas)
        .first()
        .copied()
        .ok_or("a clip with no tiles to decode")?
        .canvas;
    let model_width = config.decoder_num_attention_heads * config.decoder_attention_head_dim;

    let projection = loader
        .load(&WeightRequest::region(
            "virtual.proj_in.weight",
            0,
            Shape::new(&[u64::from(model_width), u64::from(config.latent_channels)]),
        ))
        .map_err(|error| error.to_string())?;
    // The checkpoint's first affine carries a bias (`decoder.proj_in.bias`), which the loader reads
    // beside the weight; a harness without it records the no-epilogue gemm the model never launches.
    let bias = loader
        .load(&WeightRequest::region(
            "virtual.proj_in.bias",
            0,
            Shape::new(&[u64::from(model_width)]),
        ))
        .map_err(|error| error.to_string())?;
    let input =
        Linear::new(projection, Some(bias), WeightOrder::NxK).map_err(|error| error.to_string())?;
    let decoder = VideoDecoder::load(
        &config,
        canvas,
        AttentionKind::Dense,
        DType::F32,
        input,
        &mut loader,
    )
    .map_err(|error| error.to_string())?;

    let arena = Arena::new(DevicePtr::new(0), 1 << 40);
    let forward = Forward::recording(&arena);
    let scratch = decoder
        .scratch(&forward)
        .map_err(|error| error.to_string())?;
    let latents = arena
        .tensor(
            DType::F32,
            Shape::new(&[canvas.voxels(), u64::from(config.latent_channels)]),
        )
        .map_err(|error| error.to_string())?;
    // The layout declares `video_positions` as FP32 (`video_vae::table_specs`), so the decode step
    // reads the table at that type; a harness that hands it i32 records a rotation nothing launches.
    let positions = arena
        .tensor(DType::F32, Shape::new(&[3, canvas.rows()]))
        .map_err(|error| error.to_string())?;
    // The decoder writes one row per token, and a row is the pixels and frames its final projection
    // produces: three channels over the temporal and both spatial upsamples, not the model's width.
    let out_channels = u64::from(config.out_channels)
        * u64::from(config.temporal_compression())
        * u64::from(config.spatial_compression())
        * u64::from(config.spatial_compression());
    let out = arena
        .tensor(DType::F32, Shape::new(&[canvas.rows(), out_channels]))
        .map_err(|error| error.to_string())?;

    let mut recorded = Recorded::default();
    ops::capture::set_sink(Some(&mut recorded));
    let run = decoder.decode(&forward, latents, positions, out, &scratch);
    ops::capture::set_sink(None);
    run.map_err(|error| error.to_string())?;

    // **The stitch, over the clip's whole grid.** A one-tile decode launches no seam blend, so a
    // harness that stops at `decode` records none of the `cross_fade` rows the walk states - and the
    // blends are what the decoder cannot skip: every offered canvas is multi-column and multi-row.
    // `decode_clip` decodes each tile again through the same decoder and writes it into the clip's
    // raster, blending it with the tiles above and to its left in `Tile` order.
    let stitches = minimax_h3::decoding::ClipDecoder::new(&config, DType::F32)
        .map_err(|error| error.to_string())?;
    let grid = tiles.tiles(&config, &clip_canvas);
    let clip_voxels: u64 = grid.iter().map(|tile| tile.canvas.voxels()).sum();
    let clip_latents = arena
        .tensor(
            DType::F32,
            Shape::new(&[clip_voxels, u64::from(config.latent_channels)]),
        )
        .map_err(|error| error.to_string())?;
    let clip_positions = arena
        .tensor(DType::F32, Shape::new(&[3, canvas.rows()]))
        .map_err(|error| error.to_string())?;
    // The clip's raster in pixels: three channels over the clip's frames and the request's raster.
    let clip_out = arena
        .tensor(
            DType::F32,
            Shape::new(&[
                u64::from(minimax_h3::output::Rgb::CHANNELS),
                // The raster's frame axis is in **pixels**: each latent frame decodes to the VAE's
                // temporal factor. The clip's own latent count is what the latents carry; the raster
                // the tiles are stitched into is in decoded frames, rows and columns.
                u64::from(clip_canvas.frames) * u64::from(config.temporal_compression()),
                u64::from(height),
                u64::from(width),
            ]),
        )
        .map_err(|error| error.to_string())?;
    ops::capture::set_sink(Some(&mut recorded));
    let stitched = stitches.decode_clip(
        &forward,
        &decoder,
        &grid,
        clip_latents,
        clip_positions,
        clip_out,
    );
    ops::capture::set_sink(None);
    stitched.map_err(|error| error.to_string())?;
    Ok(recorded.0)
}

/// The text encoder's own prefill: one prompt of the deployment's length, no images.
///
/// The tower is a separate component only when a request carries one, so the harness states the text-only
/// path — the same one a prompt without media takes — and the tower is the row set still owed.
fn text_encoder(request: &Request) -> Result<Vec<(String, String)>, String> {
    encoder_prefill(request, false)
}

/// The text encoder over a prompt that carries one image: the vision tower's own path.
fn vision_tower(request: &Request) -> Result<Vec<(String, String)>, String> {
    encoder_prefill(request, true)
}

fn encoder_prefill(request: &Request, with_image: bool) -> Result<Vec<(String, String)>, String> {
    let config = text_config(&request.root)?;
    let partition = Partition {
        tensor_parallel: 1,
        tensor_rank: 0,
        ulysses_degree: 1,
        ..Partition::single()
    };
    let text_type = component_type(request, "text_encoder")?;
    let mut loader = WeightLoader::stated(text_type, Some(&request.extents));
    let encoder = TextEncoder::load(&mut loader, &config, &partition, text_type)
        .map_err(|error| error.to_string())?;

    let arena = Arena::new(DevicePtr::new(0), 1 << 40);
    let forward = Forward::recording(&arena);
    let tokens = 4096u32;
    let count = u64::from(tokens);
    let ints = |values: Vec<i32>| -> Result<TensorView, String> {
        let shape = Shape::new(&[values.len() as u64]);
        let view = arena
            .tensor(DType::I32, shape)
            .map_err(|error| error.to_string())?;
        let _ = values;
        Ok(view)
    };
    let ids = arena
        .tensor(DType::I32, Shape::new(&[count]))
        .map_err(|error| error.to_string())?;
    let batch = TokenBatch {
        positions: ints((0..tokens as i32).collect())?,
        slot_mapping: ints((0..tokens as i32).collect())?,
        seq_lens: ints(vec![tokens as i32])?,
        query_offsets: ints(vec![0, tokens as i32])?,
        logits_rows: ints(vec![tokens as i32 - 1])?,
        phase: PhaseKind::Prefill,
        page_size: 64,
        num_tokens: tokens,
        max_kv_len: tokens,
        draft_lens: None,
    };
    // One image at the deployment's patch size when the harness asks for the tower: the tower's rows
    // are the token count's, and the grid only says how many patches the prompt carries.
    let shapes = if with_image {
        vec![ImageShape { grid: [1, 28, 28] }]
    } else {
        Vec::new()
    };
    let pixels = if with_image {
        Some(
            arena
                .tensor(DType::F32, Shape::new(&[784, 1536]))
                .map_err(|error| error.to_string())?,
        )
    } else {
        None
    };
    let images: Vec<ImageInput> = pixels
        .iter()
        .map(|pixels| ImageInput { pixels: *pixels })
        .collect();
    let layout = Encoder::layout(&encoder, &batch, &shapes);
    let specs = Encoder::state_tables(&encoder, &layout);
    let mut tables = Vec::with_capacity(specs.len());
    for spec in &specs {
        tables.push(
            arena
                .tensor(spec.dtype, spec.shape)
                .map_err(|error| error.to_string())?,
        );
    }
    let hidden = arena
        .tensor(
            text_type,
            Shape::new(&[count, u64::from(config.hidden_size)]),
        )
        .map_err(|error| error.to_string())?;

    let mut recorded = Recorded::default();
    ops::capture::set_sink(Some(&mut recorded));
    let run = encoder.encode(
        &forward,
        EncoderInputs {
            ids,
            batch,
            images: &images,
            layout: &layout,
            state: &tables,
            diagnostics: &[],
        },
        EncoderOutputs { hidden },
    );
    ops::capture::set_sink(None);
    run.map_err(|error| error.to_string())?;
    Ok(recorded.0)
}

/// The post-processor's own blend: the clip's raster into the timeline's bytes.
///
/// It is the one decoded component with no checkpoint of its own — its two constants are the model's — so it
/// measures whether the walk's rows are the launch's, not whether a store has them.
fn video_postprocessor(request: &Request) -> Result<Vec<(String, String)>, String> {
    // The post-processor runs in the VAE container's type, which is the type its rows key.
    let vae_type = component_type(request, "vae")?;
    let mut loader = WeightLoader::stated(DType::F32, Some(&request.extents));
    let bytes = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let mean = loader
        .constant(
            "video_postprocessor.pixel_mean",
            DType::F32,
            Shape::new(&[3]),
            &bytes(&minimax_h3::output::PIXEL_MEAN),
            "minimax-h3",
        )
        .map_err(|error| error.to_string())?;
    let std = loader
        .constant(
            "video_postprocessor.pixel_std",
            DType::F32,
            Shape::new(&[3]),
            &bytes(&minimax_h3::output::PIXEL_STD),
            "minimax-h3",
        )
        .map_err(|error| error.to_string())?;
    let post = VideoPostprocessor::new(vae_type, mean, std).map_err(|error| error.to_string())?;

    let arena = Arena::new(DevicePtr::new(0), 1 << 40);
    let forward = Forward::recording(&arena);
    let splice = Splice::CLIP;
    // **Every canvas the recording claims.** The seam blend's rows are per pixel count, so one
    // raster records one canvas's blend and leaves the others' stated and unrecorded.
    let mut recorded = Recorded::default();
    for (height, width) in request.rasters.clone() {
        let clip = arena
            .tensor(
                vae_type,
                Shape::new(&[
                    3,
                    u64::from(splice.overlap.1),
                    u64::from(height),
                    u64::from(width),
                ]),
            )
            .map_err(|error| error.to_string())?;
        // A timeline's second clip blends against its predecessor's tail; the first has nothing to fade
        // against and skips the seam, so the harness states the predecessor and the blend is recorded.
        let previous = arena
            .tensor(
                vae_type,
                Shape::new(&[
                    3,
                    u64::from(splice.overlap.1 - splice.overlap.0),
                    u64::from(height),
                    u64::from(width),
                ]),
            )
            .map_err(|error| error.to_string())?;
        let out = arena
            .tensor(
                DType::U8,
                Shape::new(&[
                    u64::from(splice.frames()),
                    u64::from(height),
                    u64::from(width),
                    3,
                ]),
            )
            .map_err(|error| error.to_string())?;

        ops::capture::set_sink(Some(&mut recorded));
        let run = post.forward(&forward, Some(previous), clip, &splice, out);
        ops::capture::set_sink(None);
        run.map_err(|error| error.to_string())?;
    }
    Ok(recorded.0)
}

/// The denoiser's own step, at one canvas and one schedule position.
///
/// The network's own `denoise` reads a `Step`, the request's latents, and the conditioning: text rows,
/// the request's own conditions, and the tables the layout names. All of them are statements of
/// shapes here — the arena is null-backed and the forward records instead of launching — so a step
/// costs no device and no checkpoint byte.
fn denoiser(request: &Request) -> Result<Vec<(String, String)>, String> {
    let dit_type = component_type(request, "transformer")?;
    let config =
        minimax_h3::Config::read(&request.root, None).map_err(|error| error.to_string())?;
    let network = config
        .denoisers
        .get(&Dit::Base)
        .ok_or("this checkpoint holds no base network")?
        .clone();
    // A sparse deployment states its served tile and keep rate, and the tile bound its compression
    // reduces over is the model's own: it is computed at load from these layouts, not stated here.
    let mut network = network;
    if let Some((tile, sparsity, reference_keep)) = request.sparse {
        network.attention = AttentionKind::sparse(tile, sparsity, reference_keep)
            .map_err(|error| error.to_string())?;
    }
    let config_hidden = network.transformer.hidden_size;
    // The two prefix capacities are the request's own: a prompt of the walk's length and one unit of
    // condition rows, which a `t2va` request leaves empty. The bound is the deployment's.
    let arrangement =
        Arrangement::new(&network, request.tokens, 64, 1).map_err(|error| error.to_string())?;
    // The layout rule is the same statement the network loads with: the tables a step reads are the
    // rule's, and a harness that built its own would be a second source of truth for them.
    let rules = Layouts::new(network.clone(), arrangement).map_err(|error| error.to_string())?;
    let mut loader = WeightLoader::stated(
        component_type(request, "transformer")?,
        Some(&request.extents),
    );
    let partition = Partition {
        tensor_parallel: 1,
        tensor_rank: 0,
        ulysses_degree: 1,
        ..Partition::single()
    };
    let denoiser = Denoiser::load(
        network,
        arrangement,
        request.bound,
        &partition,
        &mut loader,
        None,
    )
    .map_err(|error| error.to_string())?;

    let (height, width) = request.raster;
    let asked = VideoRequest {
        task: VideoTask::TextToVideoAudio,
        width,
        height,
        frames: request.clip,
        fps: 24,
        first_frames: 0,
        last_frames: 0,
        references: 0,
        text_tokens: request.tokens,
        conditions: &[],
        vision_spans: &[],
    };
    let (layout, packing) = rules.layout(&asked).map_err(|error| error.to_string())?;

    let arena = Arena::new(DevicePtr::new(0), 1 << 40);
    let forward = Forward::recording(&arena);
    // Every table the layout names, at the shape and type the model declares for it.
    let specs = rules
        .table_specs(&layout)
        .map_err(|error| error.to_string())?;
    let mut tables = Vec::with_capacity(specs.len());
    for spec in &specs {
        tables.push(
            arena
                .tensor(spec.dtype, spec.shape)
                .map_err(|error| error.to_string())?,
        );
    }
    let hidden = u64::from(config_hidden);
    let text = arena
        .tensor(
            dit_type,
            Shape::new(&[u64::from(packing.text_rows), hidden]),
        )
        .map_err(|error| error.to_string())?;
    // The latents are the trajectory's, and a trajectory is FP32 whatever the stack's own type is:
    // the solver accumulates into it, and the two cross at the stack's boundary (the four crossings the
    // walk states).
    let video = arena
        .tensor(
            ops::DType::F32,
            Shape::new(&[
                u64::from(config.video_vae.latent_channels),
                u64::from(layout.video.frames),
                u64::from(layout.video.height),
                u64::from(layout.video.width),
            ]),
        )
        .map_err(|error| error.to_string())?;
    let step = nn::Step {
        index: 0,
        count: 1,
        sigma: 1.0,
        sigma_next: 0.0,
    };
    // The audio trajectory: the layout's audio rows are two channels of stereo, so its steps are half
    // of them.
    let steps = u64::from(packing.audio_rows) / 2;
    let audio = arena
        .tensor(ops::DType::F32, Shape::new(&[2, steps]))
        .map_err(|error| error.to_string())?;
    let latents = nn::Latents {
        video,
        audio: Some(audio),
    };
    let cond = nn::Conditioning {
        text,
        source: None,
        media: &[],
        images: None,
        audio: None,
        state: &tables,
        step: None,
    };

    let mut recorded = Recorded::default();
    ops::capture::set_sink(Some(&mut recorded));
    // The request's own noise draw runs first: one Philox stream per request, the conditions' draws
    // then the video's then the audio's. A harness that hands the network pre-made latents never
    // launches it, which is why the row was stated and unlaunched until now.
    // The AdaLN cache is filled once after load - the timestep pair, its activation and the
    // projections - and a harness that only calls `denoise` reads a cache nothing wrote, so its
    // launches (the time embedder's `silu` among them) are stated and unrecorded.
    denoiser
        .prepare_constants(&forward, &[])
        .map_err(|error| error.to_string())?;
    denoiser
        .draw_noise(&forward, nn::capability::Seed(0), layout, latents)
        .map_err(|error| error.to_string())?;
    // The prompt's refinement runs before the first step, over the layout's own text rows: the
    // encoder's features at their width into the network's rows.
    let text_width = u64::from(config.text_encoder.hidden_size);
    let run = arena
        .tensor(
            dit_type,
            Shape::new(&[u64::from(packing.text_rows), text_width]),
        )
        .map_err(|error| error.to_string())
        .and_then(|source| {
            denoiser
                .refine_conditioning(&forward, source, text, &tables, None)
                .map_err(|error| error.to_string())
        })
        .and_then(|()| {
            denoiser
                .denoise(&forward, &step, latents, cond)
                .map_err(|error| error.to_string())
        });
    ops::capture::set_sink(None);
    run?;
    Ok(recorded.0)
}

/// The conditioned pass: one reference still, through the model's own condition path.
///
/// A `t2va` request launches neither the condition encoders nor the visual conditioner, so its rows
/// say nothing about them. This harness builds what the runner builds for a reference - the pixels,
/// the anchoring draw, the rows the request's own layout counts - and drives
/// `prepare_conditioning`, the capability call the deployment serves: it encodes the media and
/// refines the prompt in place. That is where `posterior`, `frame_pad` and the encoders' launches
/// happen, and nothing else reaches them without a device.
fn conditioned(request: &Request) -> Result<Vec<(String, String)>, String> {
    let text_type = component_type(request, "text_encoder")?;
    let dit_type = component_type(request, "transformer")?;
    let config =
        minimax_h3::Config::read(&request.root, None).map_err(|error| error.to_string())?;
    let network = config
        .denoisers
        .get(&Dit::Base)
        .ok_or("this checkpoint holds no base network")?
        .clone();
    // **The sparse deployment's attention kind, as `denoiser` applies it.** Without this the
    // conditioned harness drove the dense stack whatever `--sparse` said, so a recording of the
    // conditioned deployment stated no `vsa_*` row at all while the plain harness launched six of
    // them: the model wires VSA and the instrument was reading the other backend.
    let mut network = network;
    if let Some((tile, sparsity, reference_keep)) = request.sparse {
        network.attention = AttentionKind::sparse(tile, sparsity, reference_keep)
            .map_err(|error| error.to_string())?;
    }
    let config_hidden = network.transformer.hidden_size;
    let (height, width) = request.raster;
    let partition = Partition {
        tensor_parallel: 1,
        tensor_rank: 0,
        ulysses_degree: 1,
        ..Partition::single()
    };
    // The reference's own raster is the request's canvas at the VAE's spatial compression, and its
    // packed rows are one per 2x2 patch of its single latent frame. Both are the model's arithmetic:
    // the canvas comes from the layout rule, and the rows from the packing's own count.
    let probe = VideoRequest {
        task: VideoTask::ReferenceToVideoAudio,
        width,
        height,
        frames: request.clip,
        fps: 24,
        first_frames: 0,
        last_frames: 0,
        references: 1,
        text_tokens: request.tokens,
        conditions: &[],
        vision_spans: &[],
    };
    let unit = network.attention.capacity_rows();
    let probe_rules = Layouts::new(
        network.clone(),
        Arrangement::new(&network, request.tokens, unit, 1).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    // The reference's raster is the request's own latent raster, which the layout states for the video
    // it generates: the canvas the rule returns is the decoder's pixel canvas, not this count.
    let (probe_layout, _) = probe_rules
        .layout(&probe)
        .map_err(|error| error.to_string())?;
    let raster = minimax_h3::packing::LatentRaster {
        height: probe_layout.video.height,
        width: probe_layout.video.width,
    };
    // **The reference is a clip, not a still.** An image reference carries no soundtrack in H3's
    // packing (a one-frame reference is an image, and the packing refuses an image with audio), so
    // the request that reaches the audio condition is a `ref2va` one whose reference is decoded to a
    // latent frame or more - one encoder clip, which is the shortest reference the model serves.
    let piece_frames = 17u32;
    let piece_latent = network
        .clip
        .latent_frames(piece_frames)
        .map_err(|error| error.to_string())?;
    // The capacity this request needs is the packing's own count for its condition - the reference's
    // visual rows **and its soundtrack's** - so it is asked of the packing rather than restated: a
    // capacity that held only the visual rows refused the load by 80 rows on the first canvas.
    let target_canvas = minimax_h3::config::offered_canvases()
        .iter()
        .find(|canvas| canvas.height == height && canvas.width == width)
        .copied()
        .ok_or("the request's raster is not one of the deployment's canvases")?;
    let piece_condition = minimax_h3::packing::Condition {
        role: minimax_h3::packing::Role::Reference,
        frame: Some(minimax_h3::packing::Frame {
            canvas: target_canvas,
            num_frames: piece_frames,
        }),
        audio_samples: request.track,
    };
    let piece_segments = minimax_h3::packing::condition_segments(
        std::slice::from_ref(&piece_condition),
        &target_canvas,
        network.clip,
    )
    .map_err(|error| error.to_string())?;
    let rows: u32 = piece_segments
        .iter()
        .map(|segment| segment.video_rows() + segment.audio_rows())
        .sum();
    // The condition capacity is the deployment's and holds this request's rows, in whole capacity
    // units: a capacity the packing cannot serve is refused at load rather than at the first write.
    let capacity = rows.div_ceil(unit) * unit;

    let arrangement = Arrangement::new(&network, request.tokens, capacity, 1)
        .map_err(|error| error.to_string())?;
    let rules = Layouts::new(network.clone(), arrangement).map_err(|error| error.to_string())?;
    let mut loader = WeightLoader::stated(
        // The network's own element type. The VAE encoders beside it compute in FP32 at every
        // deployment and get their own statement below; loading the DiT through this one made a
        // conditioned harness read every DiT weight as FP32, so its attended rows were written FP32
        // and two launch keys stated a type the checkpoint does not hold.
        dit_type,
        Some(&request.extents),
    );
    // `ConditionEncoders` reads its weights from the VAE directories, as the decoder halves do, and
    // the stated loader answers every name from the extents table on a host with no shard.
    // The encoders compute in FP32 at every deployment, so they get their own statement. They are
    // built here rather than through `ConditionEncoders::load` because that path reborrows the
    // loader into the checkpoint's component, and a reborrow publishes into a retained store: a host
    // with no shard has none, and a recording states shapes instead of reading bytes.
    let mut encoder_loader = WeightLoader::stated(DType::F32, Some(&request.extents));
    let encoders = minimax_h3::denoiser::ConditionEncoders {
        video: VideoEncoder::load(&config.video_vae, &mut encoder_loader)
            .map_err(|error| error.to_string())?,
        audio: AudioEncoder::load(&config.audio_vae, &mut encoder_loader)
            .map_err(|error| error.to_string())?,
    };

    let conditions = [ConditionShape {
        role: ConditionRole::Reference,
        frames: piece_frames,
        // The condition's own raster in **pixels**: the model reduces it to latents itself
        // (`canvas_of_raster` then `latent_raster`), so a harness that states the latent raster here
        // is divided twice and the packing sees a raster the request never had.
        height,
        width,
        // A reference condition that carries audio: the audio projection and its narrow crossing
        // (`target_bf16_source_f32 columns 2688`, the audio modulation at 56) exist only for a
        // request with an audio condition, and a harness that states `0` never reaches them.
        audio_samples: request.track,
    }];
    let asked = VideoRequest {
        conditions: &conditions,
        ..probe
    };
    let (layout, packing) = rules.layout(&asked).map_err(|error| error.to_string())?;
    // **The bound is the deployment's packed row count, not this request's.** Passing the request's
    // own total loaded the attention for exactly that one layout, so the first launch of a request
    // with one more condition row refused by name - and raising the constant would have been the
    // same defect with a luckier number. `request.bound` is `Layouts::rows_bound`, which serving
    // reads and which the walk takes as its only source, so the harness cannot state a deployment
    // the model cannot be asked to be. The launch still states this request's own packed rows, and
    // they are inside the bound by construction.
    let _ = packing.padded_tokens;
    let mut denoiser = Denoiser::load(
        network,
        arrangement,
        request.bound,
        &partition,
        &mut loader,
        None,
    )
    .map_err(|error| error.to_string())?;
    denoiser
        .attach_encoders(encoders)
        .map_err(|error| error.to_string())?;

    let arena = Arena::new(DevicePtr::new(0), 1 << 40);
    let forward = Forward::recording(&arena);
    let specs = rules
        .table_specs(&layout)
        .map_err(|error| error.to_string())?;
    let mut tables = Vec::with_capacity(specs.len());
    for spec in &specs {
        tables.push(
            arena
                .tensor(spec.dtype, spec.shape)
                .map_err(|error| error.to_string())?,
        );
    }
    let hidden = u64::from(config_hidden);
    // The prefix the conditioner writes into: the prompt's rows, the condition's rows beneath them,
    // and the zero row the layout reserves after both.
    let prefix = u64::from(packing.text_rows + packing.condition_rows + 1);
    // The conditioner's input is the text encoder's own features, at that encoder's width; its output
    // is the denoiser's rows, at the network's - two widths, and a source at the wrong one is refused.
    let text_width = u64::from(config.text_encoder.hidden_size);
    let source = arena
        .tensor(text_type, Shape::new(&[prefix, text_width]))
        .map_err(|error| error.to_string())?;
    let text = arena
        .tensor(dit_type, Shape::new(&[prefix, hidden]))
        .map_err(|error| error.to_string())?;
    // **The refiner reads the prompt, not the prefix.** Its attention is loaded for the deployment's
    // text capacity (`Arrangement::text_rows`, which is also the bound its own `refiner_cu_seq_lens_k`
    // table states), and the reference's `TokenRefiner.forward` takes `[documents, rows, text_dim]` -
    // the prompt alone. Handing it the whole prefix buffer gave its key projection every condition
    // row and the anchor row, so its keys outran the bound it was loaded for by
    // `condition_rows + 1` and the first conditioned launch refused by name. The conditioner writes
    // below the prompt in the same buffer, so both views name the same allocation and no copy moves.
    let prompt = |view: TensorView| {
        let columns = view.shape.dim(1);
        TensorView::with_strides(
            view.offset_elements(0),
            view.dtype,
            Shape::new(&[u64::from(arrangement.text_rows), columns]),
            nn::Strides::new(&[columns, 1]),
        )
    };
    let source = prompt(source);
    let text = prompt(text);
    // The condition's pixels at the canvas the request names, in the encoder's own channel order,
    // and the anchoring draw at its latent raster: the runner owns both, and the conditioner
    // patchifies the draw rather than reading rows from it.
    let pixels = arena
        .tensor(
            ops::DType::F32,
            Shape::new(&[
                1,
                3,
                u64::from(piece_frames),
                u64::from(height),
                u64::from(width),
            ]),
        )
        .map_err(|error| error.to_string())?;
    let noise = arena
        .tensor(
            ops::DType::F32,
            Shape::new(&[
                u64::from(config.video_vae.latent_channels),
                u64::from(piece_latent),
                u64::from(raster.height),
                u64::from(raster.width),
            ]),
        )
        .map_err(|error| error.to_string())?;
    // The reference's soundtrack is a condition of the request the layout has already packed, so its
    // rows are the model's own count - the denoiser's segment for it - and not a second statement of
    // the arithmetic.
    let piece_segment = denoiser
        .segments(&asked)
        .map_err(|error| error.to_string())?
        .into_iter()
        .find(|segment| segment.audio_rows() > 0 || segment.video_rows() > 0)
        .ok_or("the request's reference packs into no segment")?;
    let piece_audio_rows = piece_segment.audio_rows();
    let piece_rows = piece_segment.video_rows();
    // The piece's soundtrack, at the samples the request's condition states: a reference that
    // carries audio is the deployment's own case (`fl2va` fits every `ref2va` request that has one),
    // and a harness that hands the conditioner `None` records a conditioner that never encodes one.
    let piece_track = arena
        .tensor(
            ops::DType::F32,
            // The encoder reads one mono track per batch row, so a stereo soundtrack is its two
            // channels - the same form the generated track takes, and the one the packing's
            // channel-major rows are counted for.
            Shape::new(&[
                u64::from(minimax_h3::config::AUDIO_CHANNELS),
                1,
                // The encoder convolves whole 800-sample latent frames, so the request's track is
                // handed to it **padded** - the padding is a request buffer's tail, which the
                // reference's `unpack_samples` fills and which is the caller's here.
                minimax_h3::encoding::padded_samples(
                    u64::from(request.track),
                    u64::from(config.audio_vae.hop()),
                )
                .map_err(|error| error.to_string())?,
            ]),
        )
        .map_err(|error| error.to_string())?;
    let pieces = [nn::capability::MediaInput {
        role: nn::capability::MediaRole::Reference,
        rows: piece_rows,
        audio_rows: piece_audio_rows,
        pixels: Some(pixels),
        audio: Some(piece_track),
        noise: Some(noise),
    }];
    let video = arena
        .tensor(
            ops::DType::F32,
            Shape::new(&[
                u64::from(config.video_vae.latent_channels),
                u64::from(layout.video.frames),
                u64::from(layout.video.height),
                u64::from(layout.video.width),
            ]),
        )
        .map_err(|error| error.to_string())?;
    let steps = u64::from(packing.audio_rows) / 2;
    let audio = arena
        .tensor(ops::DType::F32, Shape::new(&[2, steps]))
        .map_err(|error| error.to_string())?;
    let latents = nn::Latents {
        video,
        audio: Some(audio),
    };
    let cond = nn::Conditioning {
        text,
        source: Some(source),
        media: &pieces,
        images: None,
        audio: None,
        state: &tables,
        step: None,
    };
    let step = nn::Step {
        index: 0,
        count: 1,
        sigma: 1.0,
        sigma_next: 0.0,
    };

    let mut recorded = Recorded::default();
    ops::capture::set_sink(Some(&mut recorded));
    let run = denoiser
        .prepare_constants(&forward, &[])
        .map_err(|error| error.to_string())
        .and_then(|()| {
            denoiser
                .draw_noise(&forward, nn::capability::Seed(0), layout, latents)
                .map_err(|error| error.to_string())
        })
        // The capability call the runner makes: it encodes the conditions into the prefix rows and
        // refines the prompt in place, and it is refused when this block holds no encoder.
        .and_then(|()| {
            minimax_h3::denoiser::Denoiser::prepare_conditioning(&denoiser, &forward, cond)
                .map_err(|error| error.to_string())
        })
        .and_then(|()| {
            denoiser
                .denoise(&forward, &step, latents, cond)
                .map_err(|error| error.to_string())
        });
    ops::capture::set_sink(None);
    run?;
    Ok(recorded.0)
}

/// The text encoder's statement: its own `config.json` and the processor's preprocessor config.
fn text_config(root: &Path) -> Result<minimax_h3::TextEncoderConfig, String> {
    let read = |path: PathBuf| -> Result<serde_json::Value, String> {
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("{}: {error}", path.display()))?;
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
    };
    let metadata = read(root.join("text_encoder/config.json"))?;
    let processor = read(root.join("text_encoder/preprocessor_config.json"))?;
    minimax_h3::TextEncoderConfig::parse(&metadata, &processor).map_err(|error| error.to_string())
}

fn video_config(root: &Path) -> Result<minimax_h3::VideoVaeConfig, String> {
    let path = root.join("vae/config.json");
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    minimax_h3::VideoVaeConfig::parse(&values).map_err(|error| error.to_string())
}

fn audio_config(root: &Path) -> Result<minimax_h3::AudioVaeConfig, String> {
    let path = root.join("audio_vae/config.json");
    let text =
        std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    minimax_h3::AudioVaeConfig::parse(&values).map_err(|error| error.to_string())
}

/// The four components' stored element types, as the caller's tensor table states them.
///
/// One function with two callers: the walk and this recorder state the same field from the same
/// artifact, so their deployment lines cannot disagree about what a deployment is. The types are not
/// derivable on a host with no shard (`nn::Checkpoint::open` reads a shard's header and `unopened`
/// carries no names), which is why they arrive as a table rather than being read here.
fn deployment_types(path: &Path) -> String {
    match minimax_h3::load::ElementTypes::read(path) {
        Ok(stated) => format!("[{}] from {}", stated.field(), path.display()),
        Err(error) => format!("unreadable ({error})"),
    }
}

/// The **store's** bounded packed-row count: the widest over the tasks one store serves, from the
/// model's own rule.
///
/// Serving sizes its attention from `Layouts::rows_bound` of the arrangement `Partition::serving`
/// builds, so the recorder asks that same function rather than a flag - and it asks it for every task,
/// because `max_kv_len` is exact and part of a dense attention program's key while one store serves
/// all three. A layout built from a request's own text length, or from a single condition unit, keys
/// attention at a bound no store holds: the cell is missing, the refusal names a key rather than the
/// request, and the recording's own row count does not change - which is how a bound that follows a
/// flag survives inspection.
///
/// `--tokens` is deliberately **not** an argument here. It states the prompt a recording drives,
/// which is a request's fact and belongs with the request; the prompt length is not in the key.
fn deployment_bound(root: &Path, sparse: Option<(u32, f64, Option<f64>)>) -> Result<u32, String> {
    let config = minimax_h3::Config::read(root, None).map_err(|error| error.to_string())?;
    let mut bound = 0u32;
    for (task, dit) in minimax_h3::checkpoint::TASK_DITS {
        let mut network = config
            .denoisers
            .get(&dit)
            .ok_or_else(|| format!("this checkpoint holds no {} network", dit.directory()))?
            .clone();
        if let Some((tile, sparsity, reference_keep)) = sparse {
            network.attention = AttentionKind::sparse(tile, sparsity, reference_keep)
                .map_err(|error| error.to_string())?;
        }
        // The partition is the store's statement of the two capacities; the arrangement is the only
        // place they become rows, so both come from here rather than from an argument.
        let partition =
            Partition::serving(&config, task, 1, 1, 0, 0).map_err(|error| error.to_string())?;
        let arrangement = partition
            .arrangement(&network)
            .map_err(|error| error.to_string())?;
        let rules = Layouts::new(network, arrangement).map_err(|error| error.to_string())?;
        bound = bound.max(rules.rows_bound().map_err(|error| error.to_string())?);
    }
    Ok(bound)
}

/// One component's stored element type, from the caller's table where it is at hand and the
/// checkpoint's own shard headers where they are.
///
/// The harnesses state their weights' shapes (this host holds no shard), so the type a weight is
/// stated in is the component's own - the same fact the walk keys its rows from. A component of two
/// stored types answers with its bulk, and the individual crossings the extents table states keep
/// their own type, because the table's entry is the more particular fact.
fn component_type(request: &Request, component: &str) -> Result<DType, String> {
    let stated = match request.types.as_deref() {
        Some(table) => {
            Some(minimax_h3::load::ElementTypes::read(table).map_err(|error| error.to_string())?)
        }
        None => None,
    };
    let checkpoint = nn::Checkpoint::open(&request.root).ok();
    minimax_h3::load::deployment_dtype(stated.as_ref(), checkpoint.as_ref(), component)
        .map_err(|error| error.to_string())
}

/// Every checkpoint tensor the chosen component's load asks for, in the load's order.
///
/// This is the other direction of the extents table and the **completeness check of a load**: a parameter the
/// model never asks for is a tensor the checkpoint holds and no layer reads, and a parameter it asks for and the
/// checkpoint does not hold fails the load. Neither shows up in a shape, a count or a launch, so the two lists
/// are compared instead - here, on a host with no shard, because a name is not a weight.
fn requested_names(root: &Path, component: &str) -> Result<Vec<String>, String> {
    let config = audio_config(root)?;
    let names = match component {
        "all" => {
            let mut names = AudioEncoder::keys(&config).map_err(|error| error.to_string())?;
            names.extend(BigVgan::keys(&config).map_err(|error| error.to_string())?);
            names
        }
        "audio-encoder" => AudioEncoder::keys(&config).map_err(|error| error.to_string())?,
        "audio-decoder" => BigVgan::keys(&config).map_err(|error| error.to_string())?,
        other => return Err(format!("component {other} asks for no names here")),
    };
    Ok(names)
}

/// Every name the chosen component's load asks for, one per line.
///
/// `--list` states the extents a **computed** weight needs; this states the parameter set itself, which is what a
/// checkpoint's own key list is diffed against.
fn list_names(root: &Path, component: &str) -> Result<(), String> {
    let mut names = requested_names(root, component)?;
    names.sort();
    names.dedup();
    for name in &names {
        println!("{name}");
    }
    eprintln!("# {} checkpoint tensors asked for", names.len());
    Ok(())
}

/// Check the extents table against the checkpoint's own headers, name by name.
///
/// The recorder's rows are only as right as the extents table it builds every tensor from: a table
/// entry stating a type the checkpoint does not store would key a row the model never launches, and
/// it would agree with itself exactly as the walk did before the recorder caught it. So the table is
/// diffed against the checkpoint where the shards are - the one direction a fixture cannot check
/// about itself.
///
/// Absent names and differing types or extents are both failures: the first is a row for a tensor
/// that does not exist, the second is a row keyed from the wrong type. A checkpoint tensor the table
/// does not name is not a failure - the table holds the tensors the harnesses touch, not every weight.
fn check_extents(root: &Path) -> Result<(), String> {
    let path = PathBuf::from("notes/programs/h3-extents.json");
    let stated = read_extents(&path)?;
    // `open` requires every shard the index names; a host holding some of them still has headers to
    // check against, which is what `unopened` gives - and a host holding none answers zero tensors,
    // which the summary states rather than pretending the table passed.
    let checkpoint = match nn::Checkpoint::open(root) {
        Ok(opened) => opened,
        Err(_) => nn::Checkpoint::unopened(root).map_err(|error| error.to_string())?,
    };
    let mut absent = Vec::new();
    let mut differ = Vec::new();
    let mut matched = 0_u32;
    for (name, (dtype, shape)) in &stated {
        let found = if checkpoint.is_pipeline() {
            checkpoint
                .components()
                .filter_map(|component| checkpoint.component(component).ok())
                .find_map(|component| component.tensor(name))
        } else {
            checkpoint.tensor(name)
        };
        match found {
            None => absent.push(name.clone()),
            Some(location) => {
                if location.dtype != *dtype || location.shape != *shape {
                    differ.push(format!(
                        "{name}: the table states {} {:?} and the checkpoint stores {} {:?}",
                        dtype,
                        shape.dims(),
                        location.dtype,
                        location.shape.dims()
                    ));
                } else {
                    matched += 1;
                }
            }
        }
    }
    println!(
        "# {} extents entries, {matched} match the checkpoint, {} absent here, {} differ",
        stated.len(),
        absent.len(),
        differ.len()
    );
    // An absent name is not a failure on every host: this one holds only the `text_encoder` shard, so
    // the VAEs' and the DiT's names have nothing to agree with. What is always a failure is a name
    // the checkpoint holds at another type or another shape - that is a row keyed from a tensor the
    // model does not have. A host with every shard sees no absent name, which is the run that proves
    // the table.
    for name in &absent {
        println!("absent\t{name}");
    }
    for line in &differ {
        println!("differ\t{line}");
    }
    // **Nothing checked is not a pass.** A table whose every name is absent describes either another
    // component or another host, and in both cases the check ran over nothing - so it refuses rather
    // than printing a summary that reads like agreement.
    if matched == 0 {
        return Err(format!(
            "{}: {} extents entries and not one was found in the checkpoint ({} tensors there), so \
             nothing was checked: this host holds no shard for these names",
            root.display(),
            stated.len(),
            checkpoint.len()
        ));
    }
    if differ.is_empty() {
        Ok(())
    } else {
        Err(
            "the extents table and the checkpoint disagree; a fixture cut from it keys rows the \
             model does not launch"
                .to_owned(),
        )
    }
}

/// The checkpoint tensors' extents, as the walk states them: `{"name": {"dtype": "F32", "shape": [...]}}`.
///
/// This is the checkpoint's shard-header content, not a model fact: it is produced where the weights are and
/// used where they are not.
fn read_extents(path: &Path) -> Result<BTreeMap<String, (DType, Shape)>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let object = values
        .as_object()
        .ok_or_else(|| format!("{} is not an object", path.display()))?;
    let mut extents = BTreeMap::new();
    for (name, entry) in object {
        let dtype = entry
            .get("dtype")
            .and_then(|value| value.as_str())
            .and_then(dtype_of)
            .ok_or_else(|| format!("{name}: the entry states no element type"))?;
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

fn dtype_of(name: &str) -> Option<DType> {
    match name {
        "F32" => Some(DType::F32),
        "F16" => Some(DType::F16),
        "BF16" => Some(DType::BF16),
        "I32" => Some(DType::I32),
        _ => None,
    }
}

/// Print the tensors whose extents a request cannot state, as a skeleton of the extents table.
///
/// Only the weight-normalized tensors: a weight a module merely reads takes its extents from the request it
/// asks with, so listing those too would hand the host a file larger than the recorder reads — the same defect
/// as a fixture that carries more than the walk. The names come from the component's own signature
/// (`AudioEncoder::keys`), so the skeleton cannot drift from what the load binds against.
fn list_extents(root: &Path, component: &str) -> Result<(), String> {
    let names = requested_names(root, component)?;
    let normalized: Vec<&String> = names
        .iter()
        .filter(|name| name.ends_with("weight_v") || name.ends_with("weight_g"))
        .collect();
    for name in &normalized {
        println!("{name:?}: {{\"dtype\": \"F32\", \"shape\": []}}");
    }
    // The components this file does not ask for are named, so a listing cannot read as complete.
    eprintln!(
        "# {} of {} tensors need an extents entry: the weight-normalized pairs, whose folded form takes \
         nothing from a request",
        normalized.len(),
        names.len()
    );
    Ok(())
}
