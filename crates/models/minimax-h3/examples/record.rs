//! A model's programs, recorded from the model's own `forward`.
//!
//! ```text
//! cargo run -p minimax-h3 --example record -- --root /opt/uv/models/MiniMax-H3 --component all \
//!     --dtype f16 --clip 256 --raster 480x832 --rows 32000 --extents h3-audio-extents.json \
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

use minimax_h3::audio_vae::AudioDecoder;
use minimax_h3::encoding::{AudioEncoder, VideoEncoder};
use minimax_h3::checkpoint::Dit;
use minimax_h3::denoiser::{Arrangement, Denoiser, Layouts};
// The capability trait, for the request's noise draw: it is a method of the trait, not of the struct.
use nn::capability::Denoiser as _;
use nn::capability::{
    Encoder, EncoderInputs, EncoderOutputs, ImageInput, ImageShape, TokenBatch, VideoRequest,
    VideoTask,
};
use nn::{PhaseKind, TensorView};
use minimax_h3::AttentionKind;
use minimax_h3::output::{Splice, VideoPostprocessor};
use minimax_h3::text_encoder::TextEncoder;
use minimax_h3::video_vae::{Canvas, VideoDecoder};
use minimax_h3::Partition;
use nn::layers::{Linear, WeightOrder};
use nn::weights::WeightRequest;
use nn::forward::Arena;
use nn::tensor::DevicePtr;
use nn::weights::WeightLoader;
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
struct Request {
    root: PathBuf,
    dtype: DType,
    clip: u32,
    raster: (u32, u32),
    rows: u32,
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
    let mut dtype = DType::F16;
    let mut clip = 256u32;
    let mut raster = (480u32, 832u32);
    let mut rows = 32_000u32;
    let mut against: Option<PathBuf> = None;
    let mut extents: Option<PathBuf> = None;
    let mut sparse: Option<(u32, f64, Option<f64>)> = None;
    let mut list = false;
    let mut names = false;
    while let Some(argument) = arguments.next() {
        let mut value = || arguments.next().ok_or_else(|| format!("{argument} needs a value"));
        match argument.as_str() {
            "--root" => root = PathBuf::from(value()?),
            "--component" => component = value()?,
            "--dtype" => {
                dtype = match value()?.as_str() {
                    "f16" => DType::F16,
                    "bf16" => DType::BF16,
                    "f32" => DType::F32,
                    other => return Err(format!("unknown dtype {other}")),
                }
            }
            "--clip" => clip = value()?.parse().map_err(|_| "clip is a count of frames")?,
            "--raster" => {
                let stated = value()?;
                let (w, h) = stated.split_once('x').ok_or("--raster is <width>x<height>")?;
                raster = (
                    w.parse().map_err(|_| "--raster's width is a count")?,
                    h.parse().map_err(|_| "--raster's height is a count")?,
                );
            }
            "--rows" => rows = value()?.parse().map_err(|_| "--rows is a count")?,
            "--against" => against = Some(PathBuf::from(value()?)),
            "--extents" => extents = Some(PathBuf::from(value()?)),
            // `tile,sparsity[,reference_keep]`: the bound the compression reduces over is the
            // deployment's own, computed from its layouts at load, so it is not a flag here.
            "--sparse" => {
                let stated = value()?;
                let parts: Vec<&str> = stated.split(',').collect();
                if parts.len() != 2 && parts.len() != 3 {
                    return Err("--sparse is `tile,sparsity[,reference_keep]`".to_owned());
                }
                sparse = Some((
                    parts[0].trim().parse().map_err(|_| "--sparse's tile is a row count")?,
                    parts[1].trim().parse().map_err(|_| "--sparse's sparsity is a fraction")?,
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
            other => return Err(format!("unknown argument {other}")),
        }
    }

    if names {
        return list_names(&root, &component);
    }
    if list {
        return list_extents(&root, &component);
    }

    let table = match extents.as_deref() {
        Some(path) => read_extents(path)?,
        None => BTreeMap::new(),
    };
    let request = Request {
        root,
        dtype,
        clip,
        raster,
        rows,
        extents: table,
        sparse,
    };

    let harnesses: Vec<(&str, Harness)> = match component.as_str() {
        "all" => vec![
            ("audio-encoder", audio_encoder as Harness),
            ("audio-decoder", audio_decoder as Harness),
            ("video-encoder", video_encoder as Harness),
            ("video-decoder", video_decoder as Harness),
            ("denoiser", denoiser as Harness),
            ("video-postprocessor", video_postprocessor as Harness),
            ("text-encoder", text_encoder as Harness),
            ("vision-tower", vision_tower as Harness),
        ],
        // The order above is the one a caller reads: the decoders, then the network, then the prompt's
        // path. `all` is every harness this build can drive, and the audio halves answer by name while
        // their extents table is missing rather than silently leaving the run short.
        "audio-encoder" => vec![("audio-encoder", audio_encoder as Harness)],
        "audio-decoder" => vec![("audio-decoder", audio_decoder as Harness)],
        "video-encoder" => vec![("video-encoder", video_encoder as Harness)],
        "video-decoder" => vec![("video-decoder", video_decoder as Harness)],
        "denoiser" => vec![("denoiser", denoiser as Harness)],
        "video-postprocessor" => vec![("video-postprocessor", video_postprocessor as Harness)],
        "text-encoder" => vec![("text-encoder", text_encoder as Harness)],
        "vision-tower" => vec![("vision-tower", vision_tower as Harness)],
        other => return Err(format!("unknown component {other}")),
    };

    println!(
        "# recorded from the model's own forward, at {}: dtype {}, clip {}, raster {}x{}, packed rows {}, sparse {:?}",
        request.root.display(),
        request.dtype,
        request.clip,
        request.raster.0,
        request.raster.1,
        request.rows,
        request.sparse
    );
    let mut all: Vec<(String, String)> = Vec::new();
    let mut refusals: Vec<(&str, String)> = Vec::new();
    for (name, harness) in &harnesses {
        match harness(&request) {
            Ok(rows) => {
                eprintln!("# {name}: {} rows", rows.len());
                for row in rows {
                    println!("{}\t{}", row.0, row.1);
                    all.push(row);
                }
            }
            Err(reason) => {
                // A refusal is a row of the output, not an exit.
                println!("# refused: {name}: {reason}");
                eprintln!("# refused: {name}: {reason}");
                refusals.push((name, reason));
            }
        }
    }
    all.sort();
    all.dedup();
    let mut operators: Vec<&str> = all.iter().map(|(name, _)| name.as_str()).collect();
    operators.sort_unstable();
    operators.dedup();
    eprintln!(
        "# {} data rows, {} operators, {} of {} components refused{}",
        all.len(),
        operators.len(),
        refusals.len(),
        harnesses.len(),
        if refusals.is_empty() {
            String::new()
        } else {
            format!(
                ": {}",
                refusals
                    .iter()
                    .map(|(name, _)| *name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
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
    let stated = std::fs::read_to_string(walk).map_err(|error| format!("{}: {error}", walk.display()))?;
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

    let hop = encoder.hop();
    let length = hop * 64;
    let samples = arena
        .tensor(DType::F32, Shape::new(&[1, 1, length]))
        .map_err(|error| error.to_string())?;
    let frames = encoder.frames(length).map_err(|error| error.to_string())?;
    let channels = u64::from(config.latent_channels);
    let out = arena
        .tensor(DType::F32, Shape::new(&[frames, channels]))
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
/// type is. The frames are the bucket a decode of a 64-frame window serves, which is the one `AudioDecoder`'s
/// own `programs` states its per-row launches at.
fn audio_decoder(request: &Request) -> Result<Vec<(String, String)>, String> {
    let config = audio_config(&request.root)?;

    // The same extents table the encoder needs: the decoder's stages carry weight-normalized convolutions too,
    // and a fold reads its direction tensor's extents rather than its own.
    let mut loader = WeightLoader::stated(DType::F32, Some(&request.extents));
    let decoder = AudioDecoder::load(&config, &mut loader).map_err(|error| error.to_string())?;

    let arena = Arena::new(DevicePtr::new(0), 1 << 40);
    let forward = Forward::recording(&arena);

    let frames = 64u32;
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
    let (height, width) = request.raster;
    let frames = 1u32;
    let latency = encoder.latent_frames(frames).map_err(|error| error.to_string())?;
    // The encoder's arithmetic is FP32 whatever the deployment's sample type is (`nn` refuses a bias and an
    // output that disagree, which is what a mixed operand reads as).
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

    let mut recorded = Recorded::default();
    ops::capture::set_sink(Some(&mut recorded));
    let run = encoder.encode(&forward, pixels, moments);
    ops::capture::set_sink(None);
    run.map_err(|error| error.to_string())?;
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
    let canvas = Canvas::clip(&config, latent_height, latent_width);
    let model_width =
        config.decoder_num_attention_heads * config.decoder_attention_head_dim;

    let projection = loader
        .load(&WeightRequest::region(
            "virtual.proj_in.weight",
            0,
            Shape::new(&[
                u64::from(model_width),
                u64::from(config.latent_channels),
            ]),
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
    let scratch = decoder.scratch(&forward).map_err(|error| error.to_string())?;
    let latents = arena
        .tensor(
            DType::F32,
            Shape::new(&[canvas.voxels(), u64::from(config.latent_channels)]),
        )
        .map_err(|error| error.to_string())?;
    // The layout declares `video_positions` as FP32 (`video_vae::table_specs`), so the decode step
    // reads the table at that type; a harness that hands it i32 records a rotation nothing launches.
    let positions = arena
        .tensor(
            DType::F32,
            Shape::new(&[3, canvas.rows()]),
        )
        .map_err(|error| error.to_string())?;
    // The decoder writes one row per token, and a row is the pixels and frames its final projection
    // produces: three channels over the temporal and both spatial upsamples, not the model's width.
    let out_channels = u64::from(config.out_channels)
        * u64::from(config.temporal_compression())
        * u64::from(config.spatial_compression())
        * u64::from(config.spatial_compression());
    let out = arena
        .tensor(
            DType::F32,
            Shape::new(&[canvas.rows(), out_channels]),
        )
        .map_err(|error| error.to_string())?;

    let mut recorded = Recorded::default();
    ops::capture::set_sink(Some(&mut recorded));
    let run = decoder.decode(&forward, latents, positions, out, &scratch);
    ops::capture::set_sink(None);
    run.map_err(|error| error.to_string())?;
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
    let mut loader = WeightLoader::stated(request.dtype, Some(&request.extents));
    let encoder = TextEncoder::load(&mut loader, &config, &partition, request.dtype)
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
    let images: Vec<ImageInput> = pixels.iter().map(|pixels| ImageInput { pixels: *pixels }).collect();
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
            request.dtype,
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
    let mut loader = WeightLoader::stated(DType::F32, Some(&request.extents));
    let bytes = |values: &[f32]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
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
    let post = VideoPostprocessor::new(request.dtype, mean, std).map_err(|error| error.to_string())?;

    let arena = Arena::new(DevicePtr::new(0), 1 << 40);
    let forward = Forward::recording(&arena);
    let splice = Splice::CLIP;
    let (height, width) = request.raster;
    let clip = arena
        .tensor(
            request.dtype,
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
            request.dtype,
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

    let mut recorded = Recorded::default();
    ops::capture::set_sink(Some(&mut recorded));
    let run = post.forward(&forward, Some(previous), clip, &splice, out);
    ops::capture::set_sink(None);
    run.map_err(|error| error.to_string())?;
    Ok(recorded.0)
}

/// The denoiser's own step, at one canvas and one schedule position.
///
/// The network's own `denoise` reads a `Step`, the request's latents, and the conditioning: text rows,
/// the request's own conditions, and the tables the layout names. All of them are statements of
/// shapes here — the arena is null-backed and the forward records instead of launching — so a step
/// costs no device and no checkpoint byte.
fn denoiser(request: &Request) -> Result<Vec<(String, String)>, String> {
    let config = minimax_h3::Config::read(&request.root, None).map_err(|error| error.to_string())?;
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
    let arrangement = Arrangement::new(&network, 4096, 64, 1)
        .map_err(|error| error.to_string())?;
    // The layout rule is the same statement the network loads with: the tables a step reads are the
    // rule's, and a harness that built its own would be a second source of truth for them.
    let rules = Layouts::new(network.clone(), arrangement).map_err(|error| error.to_string())?;
    let mut loader = WeightLoader::stated(request.dtype, Some(&request.extents));
    let partition = Partition {
        tensor_parallel: 1,
        tensor_rank: 0,
        ulysses_degree: 1,
        ..Partition::single()
    };
    let denoiser = Denoiser::load(
        network,
        arrangement,
        request.rows,
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
        text_tokens: 4096,
        conditions: &[],
        vision_spans: &[],
    };
    let (layout, packing) = rules.layout(&asked).map_err(|error| error.to_string())?;

    let arena = Arena::new(DevicePtr::new(0), 1 << 40);
    let forward = Forward::recording(&arena);
    // Every table the layout names, at the shape and type the model declares for it.
    let specs = rules.table_specs(&layout).map_err(|error| error.to_string())?;
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
            request.dtype,
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
    denoiser
        .draw_noise(&forward, nn::capability::Seed(0), layout, latents)
        .map_err(|error| error.to_string())?;
    // The prompt's refinement runs before the first step, over the layout's own text rows: the
    // encoder's features at their width into the network's rows.
    let text_width = u64::from(config.text_encoder.hidden_size);
    let run = arena
        .tensor(
            request.dtype,
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

/// The text encoder's statement: its own `config.json` and the processor's preprocessor config.
fn text_config(root: &Path) -> Result<minimax_h3::TextEncoderConfig, String> {
    let read = |path: PathBuf| -> Result<serde_json::Value, String> {
        let text = std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))
    };
    let metadata = read(root.join("text_encoder/config.json"))?;
    let processor = read(root.join("text_encoder/preprocessor_config.json"))?;
    minimax_h3::TextEncoderConfig::parse(&metadata, &processor).map_err(|error| error.to_string())
}

fn video_config(root: &Path) -> Result<minimax_h3::VideoVaeConfig, String> {
    let path = root.join("vae/config.json");
    let text = std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    minimax_h3::VideoVaeConfig::parse(&values).map_err(|error| error.to_string())
}

fn audio_config(root: &Path) -> Result<minimax_h3::AudioVaeConfig, String> {
    let path = root.join("audio_vae/config.json");
    let text = std::fs::read_to_string(&path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    minimax_h3::AudioVaeConfig::parse(&values).map_err(|error| error.to_string())
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
            names.extend(AudioDecoder::keys(&config).map_err(|error| error.to_string())?);
            names
        }
        "audio-encoder" => AudioEncoder::keys(&config).map_err(|error| error.to_string())?,
        "audio-decoder" => AudioDecoder::keys(&config).map_err(|error| error.to_string())?,
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

/// The checkpoint tensors' extents, as the walk states them: `{"name": {"dtype": "F32", "shape": [...]}}`.
///
/// This is the checkpoint's shard-header content, not a model fact: it is produced where the weights are and
/// used where they are not.
fn read_extents(path: &Path) -> Result<BTreeMap<String, (DType, Shape)>, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
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
            .map(|dim| dim.as_u64().ok_or_else(|| format!("{name}: a dimension is not a count")))
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
