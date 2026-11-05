//! The descriptor keys the H3 deployment compiles, one `operator<TAB>key` line each.
//!
//! Run it with
//!
//! ```text
//! cargo run -p minimax-h3 --example minimax_h3_keys [-- --root <checkpoint>] [--rasters HxW,...]
//! ```
//!
//! The output is `notes/programs/minimax-h3-deployment-keys.txt`: **the store's list** - every row
//! every component this build loads states, over both element types - so no package has to guess a
//! variant.
//!
//! ## The two flags, and why the canvases are one
//!
//! `--root` names the checkpoint the video encoder's rows are derived from (default
//! `/opt/uv/models/MiniMax-H3`); its `vae/config.json` is the geometry, and the header prints the
//! fields that decided the rows. `--rasters` names the canvases those rows cover, in `HxW`, defaulting
//! to `NAMED_CANVASES` - **the six the model offers**, because a store built for one canvas stops at
//! `no kernel` on any other.
//!
//! A deployment that serves a canvas the model does not name builds **its own store** rather than
//! adding a row to this one: a row is a cell every store built from this list must compile, so a
//! canvas outside the offered set belongs to a list generated with `--rasters` naming it. The two
//! lists are then two stores, and neither claims the other's canvases.
//!
//! It is not the fixture. `notes/programs/minimax-h3-keys.txt` is **one deployment's** rows, copied
//! from that deployment's walk (`examples/programs.rs` then `examples/h3_fixture.rs`), and it is a
//! subset in the sense that matters: a store built from this file serves a deployment, and a fixture
//! is what a walk bound. Two lists, two questions - `notes/h3/fixture-delta.md` is the two-way check
//! between them, and the difference between them is the finding rather than the error.
//!
//! ## Regenerating the list, and its second opinion
//!
//! The list and the check that reads it back are one recipe, because a list regenerated without its
//! check is a list nobody has asked whether it is complete:
//!
//! ```text
//! cargo run -q -p minimax-h3 --example minimax_h3_keys > notes/programs/minimax-h3-deployment-keys.txt
//! .venv/bin/python crates/models/minimax-h3/tools/check_encoder_manifests.py \
//!     --against notes/programs/minimax-h3-deployment-keys.txt
//! ```
//!
//! The second command is a host-side set difference in one direction only: it asks whether every key
//! the encoders' descriptor calls name is in the list, and names the ones that are not. Rows the
//! encoders do not call belong to other stages and are not failures. It also states how many calls
//! it could express, so a green cannot be read as complete coverage of a stage whose field sets the
//! tool does not model.
//!
//! ## Where the numbers come from
//!
//! Every geometry below is the checkpoint's, read from
//! `/opt/uv/models/MiniMax-H3/{transformer,audio_vae,vae}/config.json`, never inferred from prose:
//!
//! - transformer: `hidden_size 5376`, `num_attention_heads 56`, `attention_head_dim 128` (so
//!   attention is 56 heads of 128, not grouped), `ffn_dim 14336`, `num_layers 50`, `norm_eps` and
//!   `qk_norm_eps` 1e-5, `rope_freq_dim 16` per axis over the three axes with the leading 96
//!   channels of each head rotated and the trailing 32 left as they are
//!   (`uniserve/models/minimax_h3/attention.py`), `in_channels 24`, `audio_in_channels 32`,
//!   `text_dim 5120`, `patch_size [1, 2, 2]`;
//! - audio VAE: `latent_dim 2048`, `latent_channels 32`, `encoder_dim 64`, `encoder_rates
//!   [2, 4, 4, 5, 5]`, `decoder_dim 1024`, `decoder_rates [5, 5, 2, 2, 2, 2, 2]`,
//!   `decoder_kernel_sizes [9, 9, 4, 4, 4, 4, 4]`, `num_attention_heads 8` over the 2048-wide latent
//!   (so 8 heads of 256, FP32);
//! - video VAE: `in_channels 3`, `latent_channels 24`, `block_out_channels [128, 256, 256, 512,
//!   512, 1024]`, `norm_num_groups 32`, `norm_eps 1e-6`, spatial padding `reflect`,
//!   `decoder_num_attention_heads 32` of `decoder_attention_head_dim 64`.
//!
//! The BigVGAN decoder's channel schedule follows from `decoder_dim` and `decoder_rates`: the
//! upsamplers halve the width at each of the seven stages (`1024 // 2**(i+1)`), and the AMP blocks,
//! their activations and `activation_post` all sit at those widths
//! (`diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py:457-470`). The encoder doubles
//! from `encoder_dim` at each of its five rates, so its `Snake` runs at 128 through 2048.
//!
//! The tile and block constants belong to the layer crate, and this program reads them: it lives in
//! the model crate so that `nn` is a dependency, and every tile below is `nn::layers::tiles::*` or
//! an `ops::*_BLOCK` the package owns. A constant restated here would make the fixture agree with
//! this file and leave the layers free to disagree with both.
//!
//! ## What is deliberately not here yet
//!
//! Four families depend on numbers only `crates/models/minimax-h3` will fix, and a guess would cost
//! a package a wrong variant:
//!
//! - the attention caps: `max_kv_len` for the dense variants and the paged KV length. H3 packs
//!   frames × latent rows × (height/16) × (width/16) for video plus text and audio tokens, so the
//!   sequence is tens of thousands of rows at the deployment's resolutions — the cap is the model's
//!   scheduling decision, not a constant of the checkpoint;
//! - `vsa_select_tiles`, whose `block` is the number of key tiles one query tile may select;
//! - `gated_residual`, whose `elements` bakes `rows × width` into the descriptor and therefore has
//!   one variant per token count — the deployment's buckets are the model's;
//! - the `frame` operators, whose tile rule (`FramePadDescriptor::tiles`) follows the decoding
//!   resolution's pixel extents.
//!
//! They are added to this generator the hour the model names those numbers; nothing else waits.

use minimax_h3::SPLIT_HALF;
use nn::layers::tiles::{
    ELEMENTWISE_BLOCK, EMBEDDING_TOKENS, LOGITS_BLOCK, PDL, POSITION_BLOCK, QK_NORM_TOKENS,
    ROTARY_BLOCK, ROTATE_BLOCK, ROWS_BLOCK, UNPATCHIFY_BLOCK, row_tile,
};
use ops::DType;
use ops::modulation::ModulatedRmsNormDescriptor;
use ops::noise::NoiseDescriptor;
use ops::residual::{ScaledResidualDescriptor, WeightedRmsNormDescriptor};

/// Samples one `audio_resample` program produces: the package's `block` dimension, which the
/// reference's resampler takes as its hop (`crates/ops/kernels/audio_resample/*/package.toml`).
const RESAMPLE_BLOCK: u32 = 256;

/// The audio VAE decoder's initial width (`decoder_dim`).
const DECODER_DIM: u32 = 1024;

/// The audio VAE decoder's upsampling rates: seven stages of the 800× total.
const DECODER_RATES: [u32; 7] = [5, 5, 2, 2, 2, 2, 2];

/// The audio VAE encoder's initial width (`encoder_dim`).
const ENCODER_DIM: u32 = 64;

/// The audio VAE encoder's strides: five blocks.
const ENCODER_RATES: [u32; 5] = [2, 4, 4, 5, 5];

/// The alias-free activation wrapper's upsampling ratio (`Activation1d(ratio=2)`).
const ACTIVATION_RATIO: u32 = 2;

/// The alias-free activation wrapper's Kaiser-sinc filter length (`Activation1d(kernel_size=12)`).
const ACTIVATION_KERNEL: u32 = 12;

/// The transformer's hidden width (`hidden_size`).
const HIDDEN: u32 = 5376;
/// The video latent's channels, `vae/config.json`'s `latent_channels` four to a token: the DiT's
/// input projection reads `4 * VIDEO_CHANNELS` = 96 features per patch, and the solver's step is one
/// gated residual over the whole trajectory, so that width is both an input and a residual.
const VIDEO_CHANNELS: u32 = 24;

/// The DiT's timestep embedding width, `time_dim` (`checkpoint/config.json`): the AdaLN cache's
/// tables and the one cast that narrows the timestep activation into them are this wide.
const TIME_DIM: u32 = 2688;

/// The rotary geometry: 16 frequencies per position axis over three axes (time, height, width),
/// so the leading 96 channels of each 128-channel head rotate and the trailing 32 do not.
const ROPE_FREQ_DIM: u32 = 16;
const ROPE_AXES: u32 = 3;
const ROTATED: u32 = ROPE_FREQ_DIM * ROPE_AXES * 2;

/// The video VAE decoder's token width: `decoder_num_attention_heads · decoder_attention_head_dim`.
const VAE_WIDTH: u32 = 32 * 64;

/// The video VAE's spatial compression: 256 output pixels per 16 latent columns.
const VAE_SPATIAL_COMPRESSION: u32 = 16;

/// The video decoder's rotated width: three quarters of its 64-wide attention head (`vae/config.json`:
/// `decoder_attention_head_dim 64`, `decoder_rope_dim_ratio 0.75`), which is 48 elements - a 24-column
/// table, split evenly over the three coordinate axes (`video_vae::rotary`).
const VAE_DECODER_ROTATED: u32 = 48;

/// The video VAE decoder feed-forward's packed width (`decoders.md`: the `mlp` buffer is
/// `[rows, 8192]`).
const VAE_MLP: u32 = 8192;

/// The attention projections' inner width, `num_attention_heads · attention_head_dim = 7168`: the
/// contraction length of the output projection (`attention.py`'s `RowParallelLinear(inner, hidden)`).
const HEADS_TIMES_HEAD_DIM: u32 = 7168;

/// The attention geometry: 56 heads of 128 features (`num_attention_heads`, `attention_head_dim`).
const HEADS: u32 = 56;
const HEAD_DIM: u32 = 128;

/// Rows of one VSA tile (`vsa_tile_size` in `fastvideo_inference.json`: 64 for the FastH3 students,
/// 128 for OmniRef). `uniserve/models/minimax_h3/packing.py` binds a 64-row tile to a `4×4×4`
/// spatiotemporal block and refuses any size it does not pack, so this is the checkpoint's number,
/// not a scheduling choice. The `vsa_sparsity: 0.8` beside it gives the selection's keep count, which
/// is a launch value: it follows the clip's tile count, so it is in no key.
const VSA_TILE_ROWS: u32 = 64;

/// Rows one VSA program handles (`_BLOCK_ROWS` in `uniserve_kernels/attention/vsa_segments.py`): the
/// kernel's own tiling inside a tile, compiled in.
const VSA_BLOCK_ROWS: u32 = 16;

/// The **class** of the deployment's longest sparse reduction: the key field, not the live bound.
///
/// The bound itself is the model's, computed once at load from the canvas rule
/// (`denoiser.rs::Layouts::tiles_bound`): the maximum of `segment_packing(...).extent.video_tiles`
/// over every canvas the deployment offers and every clip length it serves, which is **5400** tiles
/// for the released base network at `VSA_TILE_ROWS` (printed by `examples/deployment_sizes.rs
/// --root <checkpoint>`). Both descriptors that read it key
/// `next_power_of_two()` of it - `CompressPooledDescriptor`'s `max_tiles` and
/// `SelectTilesDescriptor`'s `block` (which `VsaTiles::select` states from the score matrix's own
/// key extent) - because the package compiles the reduction to a power of two, so two bounds inside
/// one class are one module and a key that told them apart would demand a compilation per tile count
/// a deployment happens to request. **So this constant is 8192 while the launch passes 5400**: a
/// reader who assumes the key field holds the number they can compute is the reader this comment is
/// for.
const VSA_MAX_TILES: u32 = 8192;

/// The transformer's feed-forward width (`ffn_dim`).
const FFN: u32 = 14336;

/// The audio encoder's attention: 8 heads over the 2048-wide latent, so 256 features per head.
/// The clip lengths H3 is driven at, in video frames - **the deployment's statement**, not a property
/// of the model: the model's frame rule says which lengths a request can be snapped to, and no source
/// says which of those this deployment drives. The delivery's stages and the model's bound are the
/// list; `--frames` states another, and it may not omit one of these.
const DEPLOYMENT_FRAMES: [u32; 6] = [5, 17, 25, 49, 121, minimax_h3::config::MAX_FRAMES];

const AUDIO_ATTENTION_HEADS: u32 = 8;
const AUDIO_LATENT_DIM: u32 = 2048;

/// The audio VAE's latent channels (`latent_channels`): the encoder's narrow width.
const AUDIO_LATENT_CHANNELS: u32 = 32;

/// The modulation table's chunks per row: shift, scale and gate for each of the two sublayers
/// (`transformer.rs`'s `chunks` refuses a table wider or narrower than `6 · width`).
const MODULATION_CHUNKS: u32 = 6;

/// The transformer's normalizer epsilon (`norm_eps`, `qk_norm_eps`).
const NORM_EPSILON: f32 = 1e-5;

/// The activation element types a deployment may state. `dtype = "auto"` means the checkpoint's own
/// representation (BF16 for the DiT, the text stack, the conditioner and the sampler), and a
/// deployment that states FP32 makes every one of those programs a different key - the descriptor
/// carries the element type, and a store holding one set cannot serve the other.
const DEPLOYMENTS: [DType; 2] = [DType::BF16, DType::F32];

/// The element types a video VAE may be stored and run in: FP16 at `dtype = "auto"`, FP32 where the
/// deployment states it. Its programs are the ones the *checkpoint's* representation decides, which is
/// the opposite of the stack above - so these rows carry both and nothing else about them moves.
const VAES: [DType; 2] = [DType::F16, DType::F32];

/// The value a flag was given, or a message naming the flag that was given none.
///
/// The generator's flags are the deployment's, so a missing one is a caller's mistake rather than a
/// condition to recover from; naming the flag is the whole of the help it needs.
fn value(arguments: &mut impl Iterator<Item = String>, flag: &str) -> String {
    match arguments.next() {
        Some(value) => value,
        None => panic!("{flag} needs a value"),
    }
}

/// One component of a `HxW` raster, or a message naming the raster and the part that is not a number.
fn parse(part: &str, raster: &str, what: &str) -> u32 {
    match part.parse() {
        Ok(value) => value,
        Err(_) => panic!("{raster}: {part} is not {what}"),
    }
}

fn main() {
    // The checkpoint states the geometry the derived rows are built from, so the generator reads it
    // rather than restating it: `--root`, or the deployment's own path. A generator that transcribed
    // these numbers could not be checked against the checkpoint at all.
    let mut arguments = std::env::args().skip(1);
    let mut root = std::path::PathBuf::from("/opt/uv/models/MiniMax-H3");
    let mut rasters: Vec<(u32, u32)> = Vec::new();
    // The clip lengths the deployment serves, in video frames: the audio rows are stated once per
    // track length, so this is a deployment statement like `--rasters` and not a property of a model.
    //
    // The default is the lengths the delivery's stages drive and the model's own bound. The rows are
    // transient by construction - when the residual add's extent stops being part of its key they
    // collapse to one per add - so the ladder ships complete now and gets smaller later, and a
    // request naming a length outside it is refused by name at its first audio row program.
    let mut clip_frames: Vec<u32> = Vec::new();
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--root" => root = std::path::PathBuf::from(value(&mut arguments, "--root")),
            "--frames" => {
                for frames in value(&mut arguments, "--frames").split(',') {
                    clip_frames.push(parse(frames.trim(), frames, "a clip length"));
                }
            }
            "--rasters" => {
                for raster in value(&mut arguments, "--rasters").split(',') {
                    let Some((height, width)) = raster.trim().split_once('x') else {
                        panic!("{raster} is not HxW");
                    };
                    rasters.push((
                        parse(height, raster, "a height"),
                        parse(width, raster, "a width"),
                    ));
                }
            }
            other => panic!("unknown flag {other}"),
        }
    }
    if rasters.is_empty() {
        rasters = minimax_h3::NAMED_CANVASES
            .iter()
            .map(|canvas| (canvas.height, canvas.width))
            .collect();
    }
    let config = match minimax_h3::config::Config::read(&root, None) {
        Ok(config) => config,
        Err(error) => panic!(
            "the checkpoint at {} has no configuration to read: {error}",
            root.display()
        ),
    };
    // One pass per deployment dtype, then the union: a row whose key the deployment does not decide -
    // a table that is FP32 by the reference's own arithmetic, a widening, a convolution - is emitted
    // by both passes and appears once after the dedup below. A row that *is* the deployment's appears
    // twice, once per element type, which is the point.
    let mut lines: Vec<(String, String)> = Vec::new();
    for deployment in DEPLOYMENTS {
        lines.extend(rows(deployment));
    }
    // ---- The video encoder's rows, **derived rather than written**.
    //
    // Its frame operators key on classes the raster derives - `frame_pad`'s tile is a function of the
    // output's columns and the levels' `scaled_add` widths are latent-space extents - so a written set
    // covers one canvas and leaves every level of every other one without a program. The encoder
    // itself states what it launches, and **every canvas the model offers is one the store must
    // serve**: a store with one canvas's rows stops at `no kernel` on any other.
    //
    // This is the one place the generator reads the deployment rather than restating it, and it is
    // deliberate: a twenty-two field literal of the video VAE's config would be the largest written
    // copy in this file, and the copies are what have been wrong all night. `MINIMAX_H3_CHECKPOINT`
    // names another root; regenerating the list therefore needs the checkpoint, which a store build
    // has by construction.
    let root = std::env::var("MINIMAX_H3_CHECKPOINT")
        .unwrap_or_else(|_| "/opt/uv/models/MiniMax-H3".to_owned());
    let sidecar = format!("{root}/vae/config.json");
    let values: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&sidecar)
            .unwrap_or_else(|error| panic!("the video VAE's sidecar at {sidecar}: {error}")),
    )
    .unwrap_or_else(|error| panic!("{sidecar} is not JSON: {error}"));
    let video_vae = minimax_h3::config::VideoVaeConfig::parse(&values)
        .unwrap_or_else(|error| panic!("{sidecar}: {error}"));
    // **The canvases the deployment serves**, which are the ones the model offers - not the raster a
    // test case happens to use. The encoder's frame operators key on the canvas, so this is the whole
    // reason the rows are derived: a written set covers one canvas and leaves the others without a
    // program. `NAMED_CANVASES` is the offered set - it is what `--rasters` defaults to above, so a
    // caller who narrows the canvases narrows the rows, and the two cannot say different things.
    lines.extend(
        match minimax_h3::encoding::VideoEncoder::programs(&video_vae, &rasters) {
            Ok(rows) => rows,
            Err(error) => panic!("the video encoder's programs: {error}"),
        },
    );

    // ---- The video decoder and the postprocessor: derived, like the encoder, and for the same
    // reason.
    //
    // Neither was called here, so the decoder's own rows were **launched and never listed** - twelve
    // `scaled_add` widths at the decoder's stack, which a walk printed at every canvas and a check
    // read as a disagreement about the canvas because the same shape survived a change of one. The
    // decoder runs at the export's own element type, so both of the VAE's are stated
    // (`VideoDecoder::programs` takes the type; the postprocessor takes the rasters too, because its
    // `scaled_add` covers a whole frame).
    //
    // A component the generator does not call is a component whose rows are missing from every store
    // built from this list, and nothing in the list can say so: the file reads complete.
    for dtype in VAES {
        match minimax_h3::video_vae::VideoDecoder::programs(&config.video_vae, dtype) {
            Ok(rows) => lines.extend(rows),
            Err(error) => panic!("the video decoder states no programs at {dtype:?}: {error}"),
        }
        lines.extend(minimax_h3::output::VideoPostprocessor::programs(
            dtype, &rasters,
        ));
    }

    // ---- The audio decoder's and encoder's rows: derived from the clip lengths, for the reason the
    // video decoder's are derived, and with one more.
    //
    // Their per-row programs carry `columns = inner × samples` in the **key**, so a track length is its
    // own row set: a written set would be one length's and every other length would stop at a missing
    // program. The conversion from video frames to sample counts is the crate's own arithmetic
    // (`audio_latent_frames`, `track_samples`) rather than a constant here, because 44.1 kHz was
    // written into a measurement once tonight and produced rows for a rate this checkpoint does not
    // use.
    // The audio operators state one row set per track length, because their residual adds carry the
    // track's extent in the key: the ladder above is the deployment's clip lengths, converted here
    // through the crate's own arithmetic so no sample count is written down twice.
    // **The deployment's ladder, stated once, and `--frames` must not leave one of its rungs out:**
    // a run with no flag and a run stating this list must produce one row set, and the check fails
    // when a stated ladder and the deployment's part company - the one failure of this file that
    // nothing else would notice. It covers **default against stated, not stated against legal**: the
    // frame rule (`align_num_frames`) says which lengths a request snaps to, and the deployment says
    // which of those it drives.
    if clip_frames.is_empty() {
        clip_frames = DEPLOYMENT_FRAMES.to_vec();
    } else {
        for frames in DEPLOYMENT_FRAMES {
            if !clip_frames.contains(&frames) {
                panic!(
                    "the run states the ladder {clip_frames:?}, which omits {frames} - a clip the \
                     deployment drives; state a superset of {DEPLOYMENT_FRAMES:?} or no --frames"
                );
            }
        }
    }
    let audio = &config.audio_vae;
    // A decode starts from a track's **latent frames** and the encoder's residual units are sized in
    // **samples**: the two operators take the same clip in different units, and passing one where the
    // other is meant was a whole ladder of wrong widths (the tallest 1.4e9 instead of 1,365,600).
    let latents: Vec<u32> = clip_frames
        .iter()
        .map(|frames| minimax_h3::packing::audio_latent_frames(*frames))
        .collect();
    let samples: Vec<u32> = latents
        .iter()
        .map(
            |latent| match u32::try_from(minimax_h3::audio_vae::track_samples(audio, *latent)) {
                Ok(samples) => samples,
                Err(_) => {
                    panic!("a track of {latent} latent frames is longer than a u32 sample count")
                }
            },
        )
        .collect();
    if !clip_frames.is_empty() {
        match minimax_h3::audio_vae::AudioDecoder::programs(audio, &latents) {
            Ok(rows) => lines.extend(rows),
            Err(error) => panic!("the audio decoder states no programs: {error}"),
        }
        match minimax_h3::encoding::AudioEncoder::programs(audio, &samples) {
            Ok(rows) => lines.extend(rows),
            Err(error) => panic!("the audio encoder states no programs: {error}"),
        }
    }

    // One key per line, the same key once: a variant two layers launch is packaged once.
    lines.sort();
    lines.dedup();
    header(&lines, &config.video_vae, &rasters);
    for (operator, descriptor) in lines {
        println!("{operator}\t{descriptor}");
    }
}

/// What the rows below are: which deployment, which checkpoint, and which program produced them.
///
/// A file of bare descriptors cannot be traced when a row is questioned, and tonight every row was
/// questioned at least once. Every number here is computed from the rows themselves, so the header
/// cannot disagree with them; `build_fixtures` reads only the rows, so a line starting `#` is a
/// reader's and never a cell.
fn header(
    lines: &[(String, String)],
    video_vae: &minimax_h3::config::VideoVaeConfig,
    rasters: &[(u32, u32)],
) {
    let operators = lines
        .iter()
        .map(|(operator, _)| operator.as_str())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    // The deployments the two loops above build: the stack's stated type, the checkpoint's own, and
    // the video VAE's container type, which the checkpoint states rather than the deployment.
    println!(
        "# The program rows a MiniMax-H3 deployment launches, one `operator<TAB>key` line each."
    );
    println!("# {} data rows, {operators} operators", lines.len());
    // The rule the row set is grown and shrunk by, stated where a reader meets the count. A row was
    // dropped once for having no launcher and a device run stopped on it in its first four launches
    // (`rotary_factors ... sections_16-16-16`, the video decoder's), so the asymmetry is written down:
    // **a row costs a compile and a missing row costs a run**, and a removal has to be argued from the
    // construction rather than from an absence of launches.
    println!(
        "# rows are here because a launcher reaches them or because the construction requires them; \
         a row is removed only when the construction contradicts it, never because no walk has \
         produced it yet (`tools/walk_gaps.py --by-shape` reports a walk's shape the file lacks)."
    );
    println!(
        "# deployment: **a union, not one deployment's rows** - the stack at {} and the video VAE at \
         {}; a run states which deployment it serves and a store built from this file serves all of \
         them, which is why one operator appears at more than one element type.",
        DEPLOYMENTS
            .iter()
            .map(|dtype| format!("{dtype:?}"))
            .collect::<Vec<_>>()
            .join(" and "),
        VAES.iter()
            .map(|dtype| format!("{dtype:?}"))
            .collect::<Vec<_>>()
            .join(" and "),
    );
    println!(
        "# checkpoint: MiniMax-H3. Its two DiTs (`transformer`, `transformer_ref`) are the same \
         config, so one set of rows serves the base network and the reference DiT."
    );
    println!("# generator: cargo run -q -p minimax-h3 --example minimax_h3_keys");
    // The four fields that decide the derived rows, printed from the checkpoint that decided them. A
    // reader asking "which canvas is this file for" gets the answer beside the count, and the answer
    // cannot drift from the rows: both come from one read of `--root`.
    println!(
        "# derived: the video encoder's canvas-class rows, from `--root`'s own config (default \
         /opt/uv/models/MiniMax-H3). block_out_channels {:?}, spatial_downsample_factors {:?}, \
         temporal_downsample_factors {:?}, latent_channels {}; rasters (height, width) {:?}.",
        video_vae.block_out_channels,
        video_vae.spatial_downsample_factors,
        video_vae.temporal_downsample_factors,
        video_vae.latent_channels,
        rasters,
    );
    // The second category of a row that varies with the request, distinct from a deployment's type.
    // A store built from this file serves the canvases and lengths its rows describe; a key derived
    // from the request's own numbers cannot be enumerated here at all, and a *value* that varies with
    // a layout is not a program key but a table the executor uploads - the distinction matters,
    // because only one of the two belongs in a list of programs.
    println!(
        "# generator needs the checkpoint: the video encoder's rows are derived from `vae/config.json` (MINIMAX_H3_CHECKPOINT names another root), because its frame operators key on classes the raster derives and every canvas the model offers must be served."
    );
    println!(
        "# request-derived: a key computed from the request's own numbers cannot be a row here. \
         `frame_pad`'s tile and the `scaled_add` latent widths follow the canvas, and \
         `scaled_residual ... columns_<elements>` follows the trajectory's element count, so a store \
         built from this file serves the canvases and lengths its rows describe \
         (`tools/walk_gaps.py --by-shape` prints both sides' statements). A value that varies with a \
         layout and is *not* a program key - the vision tower's position tables - is uploaded per \
         request and appears in no list of programs."
    );
    println!(
        "# sparse: the `vsa_*` and `sparse_attention` rows belong to the video-sparse deployment and \
         are the only group no device run has exercised; their tile fields name the class the \
         deployment's own tile bound lands in, never a request's count."
    );
}

/// Every row one deployment's dtype asks for.
fn rows(deployment: DType) -> Vec<(String, String)> {
    let mut lines: Vec<(String, String)> = Vec::new();
    let mut push = |operator: &str, descriptor: String| {
        lines.push((operator.to_owned(), descriptor));
    };

    // ---- Audio: the BigVGAN decoder's activations, at every width the seven stages produce.
    //
    // `snake_beta` is the decoder's: `MiniMaxH3AudioAMPBlock` and `activation_post` wrap
    // `MiniMaxH3AudioSnakeBeta` (`autoencoder_kl_minimax_h3_audio.py:412`, `:470`), whose two
    // parameters are log-space.
    for stage in 0..DECODER_RATES.len() as u32 {
        let channels = DECODER_DIM >> (stage + 1);
        push(
            "snake_beta",
            ops::snake_beta::Descriptor {
                dtype: DType::F32,
                channels,
                block: ELEMENTWISE_BLOCK,
                pdl: PDL,
            }
            .to_string(),
        );
    }

    // `snake` is the encoder's: one linear parameter, used by the DAC residual units and encoder
    // blocks (`:229`, `:231`, `:252`, `:278`, and `audio_vae.py:321`-`471`). **A stage runs its
    // Snake at the width it takes in and doubles on the way out** (`EncoderStage.inner` is "half the
    // stage's output width"), and the network ends with one more Snake at the doubled width - six
    // widths over five rates, not five. Doubling before the push lost the narrowest one, and with it
    // the encoder's first stage.
    let mut encoder_channels = ENCODER_DIM;
    for _ in ENCODER_RATES {
        push(
            "snake",
            ops::snake::Descriptor {
                dtype: DType::F32,
                channels: encoder_channels,
                block: ELEMENTWISE_BLOCK,
                pdl: PDL,
            }
            .to_string(),
        );
        encoder_channels *= 2;
    }
    push(
        "snake",
        ops::snake::Descriptor {
            dtype: DType::F32,
            channels: encoder_channels,
            block: ELEMENTWISE_BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );

    // The decoder's upsamplers are the reference's `Activation1d`: a replicate pad, a transposed
    // convolution of stride 2, a `SnakeBeta`, then a low-pass of the same ratio — which is our
    // `audio_resample` in both directions at the same width, ratio 2 and a 12-tap filter.
    for stage in 0..DECODER_RATES.len() as u32 {
        let channels = DECODER_DIM >> (stage + 1);
        for mode in [
            ops::audio_resample::Mode::Upsample,
            ops::audio_resample::Mode::Downsample,
        ] {
            push(
                "audio_resample",
                ops::audio_resample::Descriptor {
                    mode,
                    dtype: DType::F32,
                    channels,
                    kernel: ACTIVATION_KERNEL,
                    ratio: ACTIVATION_RATIO,
                    block: RESAMPLE_BLOCK,
                    pdl: PDL,
                }
                .to_string(),
            );
        }
    }

    // ---- Media: the two programs whose variants do not depend on a resolution.
    //
    // Padding is not one of them: the video VAE's replicate and reflect pads are `frame_pad`'s
    // `reflect` flag (`crates/ops/src/frame.rs`), and the audio decoder's pads are fused into
    // `audio_resample` and its convolutions. A general ATen `F.pad(mode="replicate")` over trailing
    // axes has no caller in the reference, so it is not declared.
    //
    // The video path's antialiased downscale of a packed uint8 frame, three colour channels.
    push(
        "antialiased_resize_uint8",
        ops::resize::Descriptor {
            mode: ops::resize::Mode::Bicubic,
            channels: 3,
            block: ELEMENTWISE_BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );

    // ---- The DiT's projections: one key per distinct contraction length.
    //
    // The transformer stores its weights as Hugging Face checkpoints do (`[n, k]`, `nxk`), contracts
    // in FP32 and adds no bias: `MergedColumnParallelLinear(..., bias=False)` for the fused q/k/v and
    // `RowParallelLinear` for the output, `ffn_dim` for the feed-forward (`attention.py`,
    // `transformer.py`). The three lengths are the attention projections' `hidden_size`,
    // the output projection's `heads · head_dim = 7168`, and the feed-forward's `ffn_dim = 14336`.
    for k in [HIDDEN, HEADS_TIMES_HEAD_DIM, FFN] {
        for epilogue in [ops::gemm::Epilogue::None, ops::gemm::Epilogue::Bias] {
            push(
                "gemm",
                ops::gemm::Descriptor {
                    input: deployment,
                    weight: deployment,
                    accumulate: DType::F32,
                    output: deployment,
                    order: ops::gemm::WeightOrder::NxK,
                    epilogue,
                    k,
                }
                .to_string(),
            );
        }
    }

    // The sampler's input projection at a stated FP32 is the diagonal below (`k = HIDDEN`, all
    // three types FP32). At `dtype = "auto"` it asks for an FP32 activation over a BF16 weight, and
    // that cell has no row: no program can compute it. cuBLASLt takes one type per operand and a
    // mixed pair is not among its legal combinations, so the cell built (this backend's manifest
    // declares `input` and `weight` as independent lists and its build script never compiles the
    // element type) and would be refused by `cublasLtMatmulAlgoGetHeuristic` before any pointer was
    // read. `ops::gemm::Descriptor` lets the two types differ, which is why the row existed.
    //
    // The deployment's crossings are a named gap, not a row: `auto`'s sample is FP32 and its stack
    // BF16, so a sampler run at `auto` needs an element-type crossing at `patch` and at
    // `unpatchify_video` (one doc in `notes/h3/kernel-cells-audit.md` §5), and until those carry an
    // input and an output type, `auto` stops at the first crossing by name. A stated FP32 deployment
    // is complete.

    // The audio encoder is FP32 throughout (`Model.forward` casts, and the whole autoencoder runs in
    // FP32): its pre-block contracts at the latent width for the fused q/k/v and the shortcut, at
    // the head width for the attended rows, and at the latent channels for the GeGLU's two halves
    // and its output (`audio_vae.py:403-440`: `Linear(width, 3·width)`, `Linear(width, channels)`,
    // `Linear(channels, channels)`, `GeGLU(channels, 2·channels)`).
    for k in [
        AUDIO_LATENT_DIM,
        AUDIO_LATENT_DIM / AUDIO_ATTENTION_HEADS,
        AUDIO_LATENT_CHANNELS,
        2 * AUDIO_LATENT_CHANNELS,
    ] {
        for epilogue in [ops::gemm::Epilogue::None, ops::gemm::Epilogue::Bias] {
            push(
                "gemm",
                ops::gemm::Descriptor {
                    input: DType::F32,
                    weight: DType::F32,
                    accumulate: DType::F32,
                    output: DType::F32,
                    order: ops::gemm::WeightOrder::NxK,
                    epilogue,
                    k,
                }
                .to_string(),
            );
        }
    }

    // The DiT's rotation: three position axes of 32 rotated elements each, cut into three sections
    // of the head's rotated columns, and the per-head normalization fused with it.
    //
    // `attention.py` normalizes every query and key head with a *learned* RMSNorm (`affine`) and
    // rotates the leading 96 channels over (time, height, width) **split-half** - its own words
    // ("its leading 96 channels split-half", `attention.py:94-96`), and what the fused operator does
    // (`_rope.py:157`, `:258`: `_rotate(part, cosine, sine, "split")`). A neox pairing would rotate
    // adjacent pairs and give different numbers from the same table, which no shape refuses. The
    // table lists one row per partition: the whole head count, and the Ulysses-2 shard the same
    // block binds at half the query and key heads.
    for heads in [HEADS, HEADS / 2] {
        push(
            "qk_norm_rope",
            ops::rope::QkNormRopeDescriptor {
                dtype: deployment,
                output: deployment,
                head_dim: HEAD_DIM,
                q_heads: heads,
                k_heads: heads,
                // The model's own convention, not a restatement of it: coordinate `i` pairs with
                // `i + rotated / 2` (`_rope.py:48`).
                rotation: SPLIT_HALF,
                epsilon: ops::Float::from(NORM_EPSILON),
                rotated: ROTATED,
                rows: QK_NORM_TOKENS,
                block: row_tile(HEAD_DIM).min(1024),
                affine: true,
                pdl: PDL,
            }
            .to_string(),
        );
    }
    // The factors those two operators read, and the two are not the same cut:
    //
    // - **The DiT's** is three equal blocks of 32 rotated elements (96 in all, a 48-column table).
    //   The reference hands its rotary **one axis spanning the whole head** (`attention.py:94-105`,
    //   `axis_dims=(head_dim,)`), so the same `rope_freq_dim` frequencies serve each coordinate
    //   axis and the three axes' factors are contiguous blocks - box 0 is time, box 1 height, box 2
    //   width. A cycle would give the second and third axes the first axis's frequencies.
    // - **The text encoder's** is Qwen3-VL's M-RoPE: three unequal sections (48, 40, 40 of the
    //   head's 128 rotated elements) whose frequencies cycle over (time, height, width), a
    //   64-column table.
    //
    // - **The video decoder's** is a third cut again, and it is not a head of the stack: its head is
    //   64 wide and three quarters of it rotate (`vae/config.json`: `decoder_attention_head_dim 64`,
    //   `decoder_rope_dim_ratio 0.75`), so 48 elements split evenly over three axes - a 24-column
    //   table - and its **coordinates are F32 whatever the decode's element type is**, which is the
    //   same fact that makes the decoder plan's `cosine`/`sine` entries F32
    //   (`video_vae::rotary`). The stage-2 walk asked for exactly this row -
    //   `rotary_factors dtype_f32_positions_f32_columns_24_..._sections_16-16-16` - in the first four
    //   launches the decoder ever made, and the file did not hold it because no launcher had reached
    //   it when the row was dropped. A row costs a compile; a missing row costs a run.
    for (rotated, per_axis, interleaved, positions) in [
        (ROTATED, [ROTATED / ROPE_AXES; 3], false, DType::I32),
        (128, [48, 40, 40], true, DType::I32),
        (
            VAE_DECODER_ROTATED,
            [VAE_DECODER_ROTATED / ROPE_AXES; 3],
            false,
            DType::F32,
        ),
    ] {
        push(
            "rotary_factors",
            ops::rope::FactorsDescriptor {
                dtype: DType::F32,
                positions,
                columns: rotated / 2,
                block: ROTARY_BLOCK,
                // The layer's scale, not the rope base: every H3 `Rotary` is built with 1.0 and the
                // 1e4 base lives in the frequencies table the host fills.
                scale: ops::Float::from(1.0_f32),
                pdl: PDL,
                axes: ROPE_AXES,
                // A section is *rotated elements* per axis, not frequencies: three axes of 32
                // elements sum to the 96 the DiT's head rotates, and passing the frequency count
                // (16) would describe a different head - the descriptor `attention.rs` builds says
                // 32 each. The encoder's sections are unequal for the same reason it uses M-RoPE.
                sections: {
                    let mut sections = [0; ops::rope::MAX_AXES];
                    sections[0] = per_axis[0];
                    sections[1] = per_axis[1];
                    sections[2] = per_axis[2];
                    sections
                },
                // The DiT cuts the spectrum into blocks; the encoder's M-RoPE cycles it.
                interleaved,
            }
            .to_string(),
        );
    }

    // ---- The row gathers: `nn::layers::gather_rows` binds `logits_gather` with the gathered width
    // in its key, so every place the DiT gathers whole rows is a cell of this operator.
    //
    // The video's and the audio's hidden rows (four times per evaluation), the conditioner's prefix
    // rows once per request, the per-layer modulation table at `18 * hidden`, the final norm's at
    // `2 * hidden`, and the three narrow i32 gathers: each row's group tag, the coefficient, and the
    // step-to-entry table.
    // The two modulation tables are the **deployment's** type, not FP32: at `dtype = "auto"` the AdaLN
    // cache narrows the FP32 timestep pair into the modulation projections' type once per evaluation
    // (`scaled_add` at `time_dim`), so the tables those projections build are the stack's own BF16 -
    // that is the whole point of the narrowing. Only the two i32 gathers and the solver's coefficient
    // are pinned.
    for (dtype, columns) in [
        (deployment, HIDDEN),
        (deployment, 18 * HIDDEN),
        (deployment, 2 * HIDDEN),
        // The solver's step coefficient: `[steps, 1]` FP32, gathered at the evaluation the
        // executor's `step` table names, so one staging word serves every step (`solver.rs`).
        (DType::F32, 1),
        (DType::I32, 1),
        (DType::I32, 2),
        (DType::I32, 4),
    ] {
        push(
            "logits_gather",
            ops::logits::GatherDescriptor {
                dtype,
                columns,
                block: LOGITS_BLOCK,
                pdl: PDL,
            }
            .to_string(),
        );
    }

    // ---- The transformer's per-row programs, at `hidden_size`.
    //
    // The DiT's three modulated norms, each from its own call site in `transformer.rs`. The six
    // modulation vectors are cut from a `[rows, 18 · hidden]` table, so a vector's row stride is
    // `6 · hidden` (32256); the attention norm folds no gate and passes no update, the MLP norm
    // folds the sixth vector's gate, the block's update **and** retains the gated sum (the block's
    // second residual reads it after the down projection has overwritten `update`), and the output
    // norm reads a `[rows, 2 · hidden]` table instead, stride `2 · hidden` (10752) with neither.
    for (gate_stride, shift_stride, scale_stride, update, retain) in [
        (0_u32, 6 * HIDDEN, 6 * HIDDEN, false, false),
        (6 * HIDDEN, 6 * HIDDEN, 6 * HIDDEN, true, true),
        (0, 2 * HIDDEN, 2 * HIDDEN, false, false),
    ] {
        push(
            "modulated_rms_norm",
            ModulatedRmsNormDescriptor {
                dtype: deployment,
                output: deployment,
                indices: DType::I32,
                width: HIDDEN,
                block: row_tile(HIDDEN),
                epsilon: ops::Float::from(NORM_EPSILON),
                gate_stride: u64::from(gate_stride),
                shift_stride: u64::from(shift_stride),
                scale_stride: u64::from(scale_stride),
                update,
                retain,
            }
            .to_string(),
        );
    }
    // The gated residual that closes each block: `hidden + gate[i] · update`. The block's
    // modulation table is `[rows, 6 · hidden]` and the gate is the sixth chunk of a row
    // (`transformer.rs`'s `chunks(modulation, width, 3)`, which takes the last of the three MLP
    // chunks), so the gate's rows sit `6 · hidden` elements apart. The row count is the launch's,
    // not the key's: it changes with the clip length.
    push(
        "gated_residual",
        ops::modulation::GatedResidualDescriptor {
            dtype: deployment,
            indices: DType::I32,
            width: HIDDEN,
            gate_stride: u64::from(MODULATION_CHUNKS) * u64::from(HIDDEN),
        }
        .to_string(),
    );
    // The solver's step flattens the latent to `[elements, 1]` and steps it by one scalar, so the
    // residual's width there is one column and the kernel's `WIDTH` is one. **The step is the
    // sample's, not the stack's**: `Denoiser::programs` keys it from `SAMPLE_DTYPE`, and the forward
    // refuses a sample that is not that type - so FP32 is the only type at which a one-column
    // residual is launched, and a deployment-typed row here would be a cell nothing reads.
    //
    // The refiner's identity residual is the model's other one-column launcher, and it has its own
    // owner: `TokenRefiner::programs` states it from the scale it loads, so this file does not
    // restate a width the refiner already states (`specs/worker.md`: one object states a key).
    push(
        "scaled_residual",
        ScaledResidualDescriptor {
            dtype: DType::F32,
            columns: 1,
            block: ops::residual::RESIDUAL_BLOCK,
            update_bias: false,
        }
        .to_string(),
    );
    push(
        "scaled_residual",
        ScaledResidualDescriptor {
            dtype: deployment,
            columns: HIDDEN,
            block: ops::residual::RESIDUAL_BLOCK,
            update_bias: false,
        }
        .to_string(),
    );
    push(
        "weighted_rms_norm",
        WeightedRmsNormDescriptor {
            dtype: deployment,
            columns: HIDDEN,
            tile: row_tile(HIDDEN),
            absmax: false,
        }
        .to_string(),
    );
    // The DiT's output norm, `norm_out.norm`, is a **plain** `RmsNorm` over the hidden state
    // (`transformer.rs`'s `load`, `RmsNorm::new(gain(...), eps)`) - the stack's own final
    // normalization, not the weighted one every block wears and not the modulated one it consumes.
    // It had no row at this width in either type, and a walk of the stack launches it once per
    // forward: the last normalization before the output projection would have been the first
    // `no kernel` of a run that had already done all its work.
    push(
        "rms_norm",
        ops::norm::RmsNormDescriptor {
            dtype: deployment,
            columns: HIDDEN,
            tile: row_tile(HIDDEN),
            pdl: PDL,
        }
        .to_string(),
    );
    // The DiT's attention: one dense program over the packed sequence. `max_kv_len` is a declared
    // bound outside the key (a padded-token bucket up to 174,598 for the widest layout), so the key
    // names only the geometry. The batch path carries `[batch + 1]` offsets.
    push(
        "dense_attention",
        ops::attention::dense::DenseDescriptor {
            query: deployment,
            key_value: deployment,
            output: deployment,
            head_dim: HEAD_DIM,
            q_heads: HEADS,
            kv_heads: HEADS,
            mask: ops::attention::Mask::Dense,
            windowed: false,
            max_kv_len: 174_598,
            cu_seq_lens: true,
        }
        .to_string(),
    );
    // The VSA branch's own attention: one program per tile size, and the key extent (tiles per
    // segment per layout) is a declared bound outside the key.
    //
    // The FP32 form is not listed, and the reason is a programs gap named in
    // `notes/h3/kernel-cells-audit.md`: every `sparse_attention` backend refuses it at its entry -
    // `sm90` and `sm100` accept BF16 only (`sm100`'s `attention.cu` refuses and then casts its
    // operands to `__nv_bfloat16*`), and `sm120` accepts BF16 and FP16 - because the kernel is the
    // vendor's block-sparse pass, written for 16-bit operands. A row for it would build (the element
    // type never reaches those build scripts) and no launch could use it, so a store built from this
    // list would fail on a key no task binds: `t2va`, `fl2va` and `ref2va` all attend densely. A
    // sparse deployment that states FP32 stops at its launch with `no kernel for operator
    // sparse_attention descriptor ...`, which is the same fact at the moment it can be acted on.
    // One row: the BF16 pass's. See the note above for why the FP32 pass emits none.
    if deployment == DType::BF16 {
        push(
            "sparse_attention",
            ops::attention::sparse::SparseDescriptor {
                // The checkpoint's `vsa_tile_size`, read by the model's sparse launch; the layer
                // takes the tile as a construction argument, so no crate owns it as a constant.
                block: 64,
                dtype: deployment,
                output: deployment,
                head_dim: HEAD_DIM,
                lse: false,
                keys: 65_536,
            }
            .to_string(),
        );
    }
    // The DiT's row mapping: one key for every raster, because the program reads the canvas as a
    // launch word. A canvas is a multiple of the model's own step and the VAE compresses it 16x, so
    // every latent side a served canvas produces is even — that check is the operator's, at launch.
    // Two element types: the stack's own BF16 rows, and the FP32 latent the sampler patchifies -
    // `Denoiser::denoise` holds its sample in the latent's type for the whole trajectory, so the
    // first patch and the last unpatchify of every evaluation cross at FP32.
    for dtype in [deployment, DType::F32] {
        push(
            "patch",
            ops::patch::Descriptor {
                dtype,
                block: ops::patch::BLOCK,
                pdl: PDL,
            }
            .to_string(),
        );
    }
    // The two `unpatchify_video` forms the decoder needs: the latents, whose per-channel scale and
    // bias denormalize, and the decoder's own output, which does not. The token grid, the channels
    // and the patch are the request's and travel at launch, so the two forms are the whole set.
    for vae in VAES {
        for affine in [true, false] {
            push(
                "unpatchify_video",
                ops::unpatchify::Descriptor {
                    dtype: vae,
                    affine,
                    block: UNPATCHIFY_BLOCK,
                    pdl: PDL,
                }
                .to_string(),
            );
        }
    }
    // The sampler's own crossing: the velocity is FP32 like the sample it advances, so the head's
    // output unpatchifies without the decoder's per-channel denormalization.
    push(
        "unpatchify_video",
        ops::unpatchify::Descriptor {
            dtype: DType::F32,
            affine: false,
            block: UNPATCHIFY_BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );
    // The register prefix and the zero tail around the caller's rows.
    push(
        "sequence_append",
        ops::sequence_append::Descriptor {
            dtype: deployment,
            width: HIDDEN,
            registers: 4,
            tail: 12,
            block: ops::sequence_append::BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );
    // The seam between two decoded tiles - **every width the model can hand the operator**, on both
    // axes, because both are compiled loop bounds rather than launch values.
    //
    // `decoding.rs`'s `ClipDecoder::new` builds one blend per overlap its tile grid can produce:
    // `extent` starts at `video_vae::MINIMUM_OVERLAP` (64 output pixels) and grows by the alignment
    // (`spatial_compression`, 16) while it is under `TILE_PIXELS` (256), for axes 2 (height) and 3
    // (width). That is twelve widths on two axes per element type, and a seam only appears where a
    // canvas is larger than one tile - which is why a row set written from the one-tile case held a
    // single `extent 5`, a width this code has not built since the blend moved from latent voxels to
    // output pixels. A 768 x 1344 clip decodes as 3 x 6 tiles and would have stopped on the first seam.
    for vae in VAES {
        let mut extent = minimax_h3::video_vae::MINIMUM_OVERLAP;
        while extent < minimax_h3::video_vae::TILE_PIXELS {
            for axis in [2_u32, 3] {
                push(
                    "cross_fade",
                    ops::cross_fade::Descriptor {
                        dtype: vae,
                        extent,
                        axis,
                        block: ops::cross_fade::BLOCK,
                        pdl: PDL,
                    }
                    .to_string(),
                );
            }
            // The step is the config's spatial compression (16): a seam widens by one alignment unit.
            extent += VAE_SPATIAL_COMPRESSION;
        }
    }
    // The **video encoder's** seam blends, derived from the code rather than written down: the tile grid
    // of every canvas the model serves (`NAMED_CANVASES`) and the latent-space widths its seams produce.
    //
    // The encoder works on the latent grid, so its blend width is a decoder-space overlap divided by the
    // spatial compression (`Grid::blends` over `Seams::split`'s overlaps, `encoding.rs`). Asking the grid
    // itself for those widths is the point: the numbers depend on the raster, a canvas the config gains
    // later would add one, and a transcription would go stale silently - a blend's extent divides its
    // weights, so a wrong width fades a seam by the wrong ramp and no shape check sees it.
    //
    // One extent-1 layer is built unconditionally for the placement of a tile that has no neighbour
    // (`tiled_encode.rs`), so it belongs in the set with the derived ones.
    {
        let mut widths: Vec<u32> = vec![1];
        for canvas in minimax_h3::NAMED_CANVASES {
            // A named canvas is a legal raster, so its grid exists; a refusal here is a defect in the
            // canvas list rather than in this list, and it must not be silently skipped.
            let Ok(grid) = minimax_h3::encoding::Grid::of(canvas.height, canvas.width) else {
                panic!(
                    "the named canvas {}x{} has no tile grid",
                    canvas.height, canvas.width
                );
            };
            for overlap in minimax_h3::encoding::Grid::blends(&grid.rows)
                .into_iter()
                .chain(minimax_h3::encoding::Grid::blends(&grid.columns))
            {
                // `Grid::blends` answers the latent-space widths itself (it divides the pixel overlaps
                // by the spatial compression); a zero is an axis with no seam.
                if overlap != 0 {
                    widths.push(overlap);
                }
            }
        }
        widths.sort_unstable();
        widths.dedup();
        for extent in widths {
            for axis in [2_u32, 3] {
                push(
                    "cross_fade",
                    ops::cross_fade::Descriptor {
                        dtype: DType::F32,
                        extent,
                        axis,
                        block: ops::cross_fade::BLOCK,
                        pdl: PDL,
                    }
                    .to_string(),
                );
            }
        }
    }
    // The video decoder's per-head normalization fused with its rotation - **no learned gain**, which
    // is what `affine_false` states and what makes this a different program from the DiT's row.
    //
    // Every number here is the export's, and none is a convention of this crate: `vae/config.json`
    // gives the decoder 32 heads of 64 elements (`decoder_num_attention_heads`,
    // `decoder_attention_head_dim`) with `decoder_rope_dim_ratio 0.75`, so **48 of the 64 columns
    // rotate**, split-half; and `decoder_norm_eps 1e-5` is the RMS floor. `VideoDecoder::load` builds
    // its `Rotary` with the identity gain because the checkpoint stores no normalization weight for
    // that attention (`vae/` has no `decoder.*.attn.norm.*`), and `ops::rope` states a gainless
    // normalization as the `_affine_false` key.
    //
    // Two rows, because the VAE's element type is the export's: F16 for a checkpoint that stores it
    // half, F32 for one that stores it whole. Without them the store holds no cell for the first
    // operator a decode launches past its input projection, and stage 2 would stop on this row as if
    // it were a new defect - which is the class this list has now had three times in one night.
    for vae in VAES {
        push(
            "qk_norm_rope",
            ops::rope::QkNormRopeDescriptor {
                dtype: vae,
                output: vae,
                head_dim: 64,
                q_heads: 32,
                k_heads: 32,
                rotation: SPLIT_HALF,
                // `f32` on purpose: the config's epsilon reaches the descriptor as a 32-bit float,
                // and a 64-bit literal would name a different key.
                epsilon: ops::Float::from(1.0e-5_f32),
                rotated: 48,
                rows: QK_NORM_TOKENS,
                // `head_dim.next_power_of_two().min(1024)` - the row tile `nn`'s `Rotary` builds.
                block: 64,
                affine: false,
                pdl: PDL,
            }
            .to_string(),
        );
    }
    // The latent reorder: the video decoder writes channel-first and the media stage reads
    // channel-last, so one strided copy moves the block. Two rows, because the two consumers work in
    // different precisions — the VAE's latent is FP32 (its native precision) and the VSA compose's
    // pooled block is BF16 — and the key names the element type and nothing else: the strides are
    // launch values, so one program serves every clip geometry.
    for dtype in [DType::F32, deployment] {
        push(
            "copy_block",
            ops::copy_block::Descriptor {
                source: dtype,
                out: dtype,
            }
            .to_string(),
        );
    }
    // The condition encoder's last numerical step: the VAE encoder's moments to the sample a
    // conditioner reads, through its FP16 round trip. One row: the chain is fixed and in f32 by the
    // reference, so the only thing the key can name is the operator's own tiling -- and the two
    // launches the conditioned path makes (it does not use programmatic dependent launch) share it.
    push(
        "posterior",
        ops::posterior::Descriptor {
            block: ops::posterior::BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );
    // The audio decoder's fused BigVGAN block, at the decoder's own rate and taps.
    //
    // One launch per block activation, and each stage's blocks run at the stage's own width - the
    // decoder's schedule halving per stage, `1024 >> (stage + 1)`, so 512 down to 8 over the seven
    // rates. The taps, the ratio and the program's block are the same at every stage, so the key
    // differs only in its width, and a single row at the first width would describe one launch and
    // leave the rest without a program.
    //
    // **The widths are the schedule's, computed, not a list.** A written list held 1024 - a width the
    // schedule never produces, since the first stage halves the decoder's widest - and stopped at 16,
    // missing the narrowest stage's 8: one row with no launcher and one launch with no row, from the
    // same transcription, and the deployment's audio decoder would have stopped at its last block.
    for stage in 0..DECODER_RATES.len() as u32 {
        let channels = DECODER_DIM >> (stage + 1);
        push(
            "audio_activation",
            ops::audio_activation::Descriptor {
                dtype: DType::F32,
                channels,
                kernel: ACTIVATION_KERNEL,
                ratio: ACTIVATION_RATIO,
                // The model's own program width, read from it rather than restated: the launcher
                // passes `audio_vae::ACTIVATION_BLOCK` to `AudioActivation::new`, so a copy of the
                // number here is a second answer to a question the launch already answers - and this
                // row said 1024 for seven launches the device makes at 256.
                block: minimax_h3::audio_vae::ACTIVATION_BLOCK,
                pdl: PDL,
            }
            .to_string(),
        );
    }
    // The timestep embedding's activation, over the transformer's hidden width, at **the stack's own
    // element type**: `transformer.rs` launches `Silu::descriptor_of(dtype, config.time_hidden_dim)`
    // from the stack's `dtype`, and the stage-5 walk launched it at `f16` for an f16 deployment. A row
    // at FP32 alone was this file's defect: the AdaLN *pair* downstream is FP32, but the activation
    // that feeds it is not, and a BF16 deployment would have stopped on this key.
    push(
        "silu",
        ops::activation::SiluDescriptor {
            dtype: deployment,
            columns: HIDDEN,
            block: ELEMENTWISE_BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );
    // The solver's Euler step: an FP32 source added into a BF16 running sample.
    push(
        "scaled_add",
        ops::residual::ScaledAddDescriptor {
            target: deployment,
            source: DType::F32,
            columns: HIDDEN,
            block: ops::residual::RESIDUAL_BLOCK,
        }
        .to_string(),
    );
    // **The widening**: `Transformer::wide` turns the stack's own BF16 rows into the FP32 the head's
    // floor computes in - `target <- 1 * source + 0 * target`, so the target is never read and an
    // uninitialised buffer is the right operand (`transformer.rs`, `denoise_arena`'s `*.wide`).
    // Three widths: the patchified rows before the video input projection (96 columns), the hidden
    // rows before the output projection (5376), and the audio latent's own rows before the audio
    // input projection (32) - the one projection whose input is narrower than a patch.
    // The AdaLN cache's narrowing at `dtype = "auto"`: the reference casts the timestep activation to
    // the weight's type before the modulation projections (`weights.py:210-224`), so the cache is BF16
    // while the timestep pair arrives FP32 - one cast of `[groups, time_dim = 2688]`, and a
    // `ScaledAdd` with `alpha = 1, beta = 0` is that cast. Nothing at a stated FP32 deployment
    // narrows here, so only the pass whose deployment is narrower emits the row.
    if deployment != DType::F32 {
        push(
            "scaled_add",
            ops::residual::ScaledAddDescriptor {
                target: deployment,
                source: DType::F32,
                columns: TIME_DIM,
                block: ops::residual::RESIDUAL_BLOCK,
            }
            .to_string(),
        );
    }
    for columns in [32, 96, HIDDEN] {
        push(
            "scaled_add",
            ops::residual::ScaledAddDescriptor {
                target: DType::F32,
                source: deployment,
                columns,
                block: ops::residual::RESIDUAL_BLOCK,
            }
            .to_string(),
        );
    }
    // **The conditioner's per-head normalization.** `TokenRefiner::programs` states two norm widths -
    // the hidden state's and **one head's**, `config.head_dim` = 128 - because its attention
    // normalizes each query and key over a head rather than through the fused `qk_norm_rope` the DiT's
    // blocks use. The hidden width was a row; the head's was not, so the conditioner's attention would
    // have stopped at its first normalization. It is the conditioner's dtype - BF16 at `auto`, like
    // the rest of that path - and the tile is the operator's for 128 columns.
    push(
        "rms_norm",
        ops::norm::RmsNormDescriptor {
            dtype: deployment,
            columns: HEAD_DIM,
            tile: HEAD_DIM.next_power_of_two(),
            pdl: PDL,
        }
        .to_string(),
    );
    // The conditioner's anchoring mix runs over `[condition rows, video_channels * 4]`, which is
    // narrower than any hidden width, in the conditioner's own dtype and in the trace path's.
    for (target, source) in [(DType::F32, DType::F32), (deployment, deployment)] {
        push(
            "scaled_add",
            ops::residual::ScaledAddDescriptor {
                target,
                source,
                columns: 4 * VIDEO_CHANNELS,
                block: ops::residual::RESIDUAL_BLOCK,
            }
            .to_string(),
        );
    }
    // **The solver's step over the whole trajectory, at the sample's own width and type.** The sample
    // is `[tokens, 4 * video_channels]` for the video latent and the audio latent's own width for its
    // own trajectory, and `Solver::step` runs one gated residual over all of it - so the width is the
    // latent's, not the stack's, and the type is `SAMPLE_DTYPE`, which the forward refuses to depart
    // from. Keyed from the stack these would name a row no step launches, which is how the 96-column
    // one came to be absent while the same 96 was in the list as a `scaled_add`.
    push(
        "scaled_residual",
        ScaledResidualDescriptor {
            dtype: minimax_h3::denoiser::SAMPLE_DTYPE,
            columns: 4 * VIDEO_CHANNELS,
            block: ops::residual::RESIDUAL_BLOCK,
            update_bias: false,
        }
        .to_string(),
    );
    // The audio decoder's last step: two float channels to interleaved 16-bit PCM. The clip's length
    // is a launch value, so the key holds only the element type and the block.
    push(
        "pcm_interleave",
        ops::pcm::Descriptor {
            dtype: DType::F32,
            block: ops::pcm::BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );
    // The initial latent noise, in both element types: the request's draw is FP32 because the
    // reference's sample is (it refuses any other), and the conditioning path draws its own
    // conditions in the trace type.
    for dtype in [deployment, DType::F32] {
        push(
            "philox_noise",
            NoiseDescriptor {
                dtype,
                block: ops::noise::NOISE_BLOCK,
                // One standard normal sample per element, its stream keyed by the request's seed as
                // a launch scalar (the model has no key view to upload).
                kind: ops::noise::NoiseKind::Normal,
                source: ops::noise::KeySource::Scalar,
            }
            .to_string(),
        );
    }

    // ---- The audio encoder's head pool, and the video path's last step.
    //
    // The pool is the projection's two fused reductions: `heads = num_attention_heads 8`, the head
    // width `latent_dim / heads = 256`, pooled to `latent_channels = 32` (`audio_vae.py:431-440`,
    // `audio_vae/config.json`).
    push(
        "adaptive_pool",
        ops::adaptive_pool::Descriptor {
            dtype: DType::F32,
            heads: AUDIO_ATTENTION_HEADS,
            width: AUDIO_LATENT_DIM / AUDIO_ATTENTION_HEADS,
            output: AUDIO_LATENT_CHANNELS,
            block: ELEMENTWISE_BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );
    // The audio encoder's projection block: a **causal** attention and a **tanh-GELU** gated unit.
    //
    // The geometry is the export's (`audio_vae/config.json`): `latent_dim 2048` over
    // `num_attention_heads 8` heads of 256, with as many key heads as query heads. The mask is
    // `causal`: a frame depends only on the audio before it (`audio_vae.py:433`), and **the object
    // the load builds is the authority** - `Projection::load` constructs
    // `Attention::dense(..., true, ...)`. This list once said `dense` on the strength of a
    // `programs` parameter, which is a statement about the load rather than the load; the file's row
    // is the one that follows the constructed layer. The gated unit is
    // `GatedActivation::new(GeluTanh)` over the
    // latent channels, whose gate and value are **two separate** `[frames, 2 · 32]` buffers rather
    // than one packed one - so the row stride is the width, and this row is the `gelu_tanh`
    // activation the DiT's `silu` rows cannot stand in for.
    push(
        "dense_attention",
        ops::attention::DenseDescriptor {
            query: DType::F32,
            key_value: DType::F32,
            output: DType::F32,
            head_dim: AUDIO_LATENT_DIM / AUDIO_ATTENTION_HEADS,
            q_heads: AUDIO_ATTENTION_HEADS,
            kv_heads: AUDIO_ATTENTION_HEADS,
            mask: ops::attention::Mask::Causal,
            windowed: false,
            // A launch bound outside the key: the encoder attends one track's frames as one sequence.
            max_kv_len: 4096,
            cu_seq_lens: false,
        }
        .to_string(),
    );
    let audio_hidden = 2 * AUDIO_LATENT_CHANNELS;
    push(
        "act_and_mul",
        ops::activation::ActAndMulDescriptor {
            dtype: DType::F32,
            width: audio_hidden,
            block: ELEMENTWISE_BLOCK,
            activation: ops::activation::Activation::GeluTanh,
            gate_stride: u64::from(audio_hidden),
            value_stride: u64::from(audio_hidden),
            wide: false,
            pdl: PDL,
            // `GatedActivation::new` is the default (non-stepwise) form; the DiT's MLP declares the
            // stepwise one because the reference rounds each row's output as it goes.
            stepwise: false,
        }
        .to_string(),
    );
    // The projection's MLP normalizes over the latent channels (`projection.mlp.norm`,
    // `LayerNorm` at 32 with its bias), a width no other row of this list has.
    push(
        "layer_norm",
        ops::norm::LayerNormDescriptor {
            dtype: DType::F32,
            columns: AUDIO_LATENT_CHANNELS,
            tile: row_tile(AUDIO_LATENT_CHANNELS),
            bias: true,
            pdl: PDL,
        }
        .to_string(),
    );

    // The video path's last step: the VAE's raster is `float16`
    // (`uniserve/models/minimax_h3/output.py`'s own state buffer), three channels, packed bytes.
    for vae in VAES {
        push(
            "rgb_bytes",
            ops::rgb::Descriptor {
                dtype: vae,
                channels: 3,
                block: ops::rgb::BLOCK,
                pdl: PDL,
            }
            .to_string(),
        );
    }

    // ---- The text encoder (Qwen3-VL, `text_encoder/config.json`): its norms and its two audited
    // activations. `hidden_size 5120` for the language model, `intermediate_size` 25600 gated, and
    // the vision tower's `gelu_pytorch_tanh` over its own `intermediate_size 4304`.
    //
    // `rms_norm`/`add_rms_norm` are compiled per column count and tile, the tile being the layer's
    // `next_power_of_two(columns)`; the element type is the checkpoint's `bfloat16`.
    let text_columns = 5120_u32;
    push(
        "rms_norm",
        ops::norm::RmsNormDescriptor {
            dtype: deployment,
            columns: text_columns,
            tile: text_columns.next_power_of_two(),
            pdl: PDL,
        }
        .to_string(),
    );
    push(
        "add_rms_norm",
        ops::norm::AddRmsNormDescriptor {
            dtype: deployment,
            columns: text_columns,
            tile: text_columns.next_power_of_two(),
            pdl: PDL,
        }
        .to_string(),
    );
    // The audio encoder's GeGLU applies tanh-GELU over its hidden width, `2 · latent_channels`
    // (`audio_vae.py:417`, `GeGLU(channels, 2 * channels)`), in FP32.
    push(
        "gelu",
        ops::gelu::Descriptor {
            dtype: DType::F32,
            columns: AUDIO_LATENT_CHANNELS * 2,
            block: ELEMENTWISE_BLOCK,
            approximate: ops::gelu::Approximation::Tanh,
            pdl: PDL,
        }
        .to_string(),
    );
    // Its norms are `[patches, 1152]`, so the tile is the layer's `next_power_of_two`.
    push(
        "layer_norm",
        ops::norm::LayerNormDescriptor {
            dtype: deployment,
            columns: 1152,
            tile: 1152_u32.next_power_of_two(),
            bias: true,
            pdl: PDL,
        }
        .to_string(),
    );
    // **The mergers' norm, at a merged row's width** - the same operator one width over, and it was
    // missing: `vision.rs` launches `LayerNorm::descriptor_of(dtype, merged, true)` beside the
    // blocks' own, with `merged = hidden_size · spatial_merge_size²` (`BlockRows::merged_width`,
    // 1152 · 4 = **4608**). The mergers' two `gemm`s at that width were already rows, so the family
    // was in the file while the normalization that feeds them was not - the encode path would have
    // stopped between the merger's norm and its own projection.
    push(
        "layer_norm",
        ops::norm::LayerNormDescriptor {
            dtype: deployment,
            columns: 1152 * 4,
            tile: (1152 * 4_u32).next_power_of_two(),
            bias: true,
            pdl: PDL,
        }
        .to_string(),
    );
    // The video encoder's four levels normalize in FP32 at their own channel counts
    // (`encoding.rs`: 128, 256, 512 and 1024), each with the tile its width's next power of two is.
    // These are `ref2va`'s keys - the encoder block is the one path a `t2va` request never touches -
    // and they are listed because a deployment builds one store for every task it serves.
    for columns in [128_u32, 256, 512, 1024] {
        push(
            "layer_norm",
            ops::norm::LayerNormDescriptor {
                dtype: DType::F32,
                columns,
                tile: columns.next_power_of_two(),
                bias: true,
                pdl: PDL,
            }
            .to_string(),
        );
    }
    // Its 2-D rotary turns each head's leading half against the trailing half (`Rotation::Half`,
    // `vision.rs`'s `Rotate::new(head_dim, heads, Rotation::Half)`), 8 rows and 128 lanes per program
    // (`nn::layers::rotate`'s own constants); the whole 72-element head rotates, so no narrower
    // `rotated` is appended.
    push(
        "rope_apply",
        ops::rope::RopeApplyDescriptor {
            dtype: deployment,
            output: deployment,
            head_dim: 72,
            heads: 16,
            rotation: ops::rope::Rotation::Half,
            rows: 8,
            block: ROTATE_BLOCK,
            pdl: PDL,
            rotated: 72,
        }
        .to_string(),
    );
    // Its learned position table, bilinearly resampled from the 48 × 48 it was trained on to the
    // patch grid a request's frame count produces. `channels` and `in_side` are compiled in; the
    // output's grid is the args' extent.
    push(
        "position_interpolate",
        ops::position::Descriptor {
            dtype: deployment,
            channels: 1152,
            in_side: 48,
            block: POSITION_BLOCK,
            pdl: PDL,
        }
        .to_string(),
    );

    // The vision tower is `Qwen3VLVisionModel` (`text_encoder/config.json`'s `vision_config`):
    // `hidden_size 1152`, `num_heads 16` — so `head_dim 72` — 27 blocks, `num_position_embeddings
    // 2304` (a 48 × 48 learned table), `patch_size 16`, `temporal_patch_size 2`, and both weight and
    // bias on every LayerNorm (the checkpoint's `model.visual.*.norm{1,2}.bias` entries). Its MLP
    // **Two activations, not one.** A block's MLP takes `gelu_pytorch_tanh` over
    // `intermediate_size` 4304; a patch merger takes a plain `nn.GELU()`, the exact `erf` form, over
    // the merged row `hidden_size · spatial_merge_size²` = 1152 · 4 = 4608 (`vision.rs` builds
    // `Gelu::tanh()` and `Gelu::exact()` for those two sites, and its own header says so). The rows
    // differ in `columns` *and* in `approximate`, so one row cannot cover both.
    for (columns, approximate) in [
        (4304_u32, ops::gelu::Approximation::Tanh),
        (4608_u32, ops::gelu::Approximation::Exact),
    ] {
        push(
            "gelu",
            ops::gelu::Descriptor {
                dtype: deployment,
                columns,
                block: ELEMENTWISE_BLOCK,
                approximate,
                pdl: PDL,
            }
            .to_string(),
        );
    }

    // ---- The vision tower's products, all BF16 against FP32 accumulation with no epilogue.
    //
    // The key names the contraction, not the output width, so four rows cover nine call sites: the
    // patch projection reads the flat tubelet (`in_channels · temporal_patch_size · patch²` = 3 · 2 ·
    // 16² = 1536, `vision.rs`'s `Geometry::tubelet`); the fused qkv branches and the attention
    // projection contract over `hidden_size` 1152; the MLP's first matrix does too and its second
    // contracts over `intermediate_size` 4304; and both merger matrices read the merged row
    // (`hidden_size · spatial_merge_size²` = 4 · 1152 = 4608, `Geometry::merged_width`), the deepstack
    // mergers included.
    for k in [1536_u32, 1152, 4304, 4608] {
        push(
            "gemm",
            ops::gemm::Descriptor {
                input: deployment,
                weight: deployment,
                accumulate: DType::F32,
                output: deployment,
                order: ops::gemm::WeightOrder::NxK,
                epilogue: ops::gemm::Epilogue::None,
                k,
            }
            .to_string(),
        );
    }

    // Its attention and its residual add, at the tower's own geometry: 16 heads of 72, a dense mask
    // with no window, and **one sequence per time step** — `vision.rs` calls `attend_dense` with the
    // offsets the grid's time steps give it, so the key carries no `cu_seq_lens` suffix (that name is
    // written only for the form without offsets).
    push(
        "dense_attention",
        ops::attention::DenseDescriptor {
            query: deployment,
            key_value: deployment,
            output: deployment,
            head_dim: 72,
            q_heads: 16,
            kv_heads: 16,
            mask: ops::attention::Mask::Dense,
            windowed: false,
            // A declared bound outside the key, sized for the tower's longest sequence: its learned
            // position table is `num_position_embeddings 2304`, and no attention sequence is longer
            // than the patches that table covers.
            max_kv_len: 2304,
            cu_seq_lens: true,
        }
        .to_string(),
    );
    // Two residual streams, two widths, and neither covers the other: the tower's is
    // `[patches, 1152]` and the text encoder's is `[tokens, 5120]` - the language stack's hidden,
    // which `text_encoder.rs:1140` writes each retained layer's down projection into before the next
    // layer's norm. Both are a plain scale with no bias update, and the block is the residual
    // operator's own (`RESIDUAL_BLOCK`, 1024) rather than the elementwise one that happens to equal
    // it.
    for columns in [1152_u32, 5120] {
        push(
            "scaled_residual",
            ops::residual::ScaledResidualDescriptor {
                dtype: deployment,
                columns,
                block: ops::residual::RESIDUAL_BLOCK,
                update_bias: false,
            }
            .to_string(),
        );
    }

    // ---- The video VAE decoder's own stack, from `vae/config.json` and `video_vae.rs`'s composition.
    //
    // The decoder is 36 blocks of one width - `decoder_num_attention_heads 32 ·
    // decoder_attention_head_dim 64 = 2048` - with a feed-forward four times as wide
    // (`decoder_ffn_mult 4` -> 8192), and every shape below is one no other component of this model
    // asks for. That is exactly why none of them was here: a row set written from the shapes that
    // have run drifts from the shapes that have not, and stage 2 has never decoded.
    //
    // Each family is pushed once per element type the export may store the VAE in: F16 for a
    // checkpoint that keeps it half (the default, `dtype = "auto"`, which converts nothing), F32 for
    // one that keeps it whole.
    for vae in VAES {
        // `norm1`/`norm2` - `RmsNorm::new(weight, decoder_norm_eps)` over the width.
        push(
            "rms_norm",
            ops::norm::RmsNormDescriptor {
                dtype: vae,
                columns: VAE_WIDTH,
                tile: row_tile(VAE_WIDTH),
                pdl: PDL,
            }
            .to_string(),
        );
        // `decoder.norm_out` - the one LayerNorm of the stack, and it has its bias.
        push(
            "layer_norm",
            ops::norm::LayerNormDescriptor {
                dtype: vae,
                columns: VAE_WIDTH,
                tile: row_tile(VAE_WIDTH),
                bias: true,
                pdl: PDL,
            }
            .to_string(),
        );
        // Both residual adds carry a projection's bias (`ScaledResidual::new(scale, Some(bias))`): the
        // attention's `to_out` bias and the feed-forward's `ff.net.2` bias belong to the residuals, as
        // the reference groups them, so `update_bias` is true - a different program from the DiT's.
        push(
            "scaled_residual",
            ScaledResidualDescriptor {
                dtype: vae,
                columns: VAE_WIDTH,
                block: ops::residual::RESIDUAL_BLOCK,
                update_bias: true,
            }
            .to_string(),
        );
        // The packed gate projection's bias is the activation's (`PackedSwiGlu::new(Some(bias))`), and
        // its two halves sit beside each other, so the row stride is twice the width. The DiT's row is
        // `bias_false`; this one is not interchangeable with it.
        push(
            "value_first_swiglu",
            ops::activation::ValueFirstSwigluDescriptor {
                dtype: vae,
                width: VAE_MLP,
                block: ELEMENTWISE_BLOCK,
                input_stride: 2 * u64::from(VAE_MLP),
                bias: true,
                absmax: false,
            }
            .to_string(),
        );
        // The attention: 32 heads of 64 with as many key/value heads, a dense mask, no window.
        //
        // **The form follows the element type**, because the two programs serve different shapes and
        // each key names its own. The FP32 program is a GEMM and a softmax over one sequence and
        // reads no offsets, so its key carries `cu_seq_lens_false`; FlashAttention-4's kernel serves
        // the packed form, takes the offsets table and refuses a null one, so its key carries no
        // suffix at all. One sequence makes them arithmetically identical, and the layer states the
        // same predicate (`video_vae.rs`'s `block_attention` and `packed`), so the element type alone
        // decides - a clip decoded at the checkpoint's own FP16 must not ask the FP32 package for a
        // row, nor the flash kernel for one at f32.
        push(
            "dense_attention",
            ops::attention::DenseDescriptor {
                query: vae,
                key_value: vae,
                output: vae,
                head_dim: 64,
                q_heads: 32,
                kv_heads: 32,
                mask: ops::attention::Mask::Dense,
                windowed: false,
                // A launch bound outside the key: one tile's rows are `voxels + registers + 1` for
                // a 256 x 256 x 5 tile, and 2048 covers the largest tile the plan decodes.
                max_kv_len: 2048,
                cu_seq_lens: vae != DType::F32,
            }
            .to_string(),
        );
        // Its projections, at the VAE's own type: the input projection, the three q/k/v projections
        // and `proj_out` carry a bias, while `to_out` and `ff.net.2` fold theirs into the residuals.
        for epilogue in [ops::gemm::Epilogue::None, ops::gemm::Epilogue::Bias] {
            push(
                "gemm",
                ops::gemm::Descriptor {
                    input: vae,
                    weight: vae,
                    accumulate: DType::F32,
                    output: vae,
                    order: ops::gemm::WeightOrder::NxK,
                    epilogue,
                    k: VAE_WIDTH,
                }
                .to_string(),
            );
        }
    }

    // ---- The video VAE encoder's residual adds.
    //
    // Every block of the encoder adds its convolution pair's output back into its input
    // (`ScaledAdd.forward(f, rows(out), rows(residual), 1.0, 1.0)`) at that stage's own channel count -
    // `block_out_channels [128, 256, 256, 512, 512, 1024]`, so four distinct widths, and the encoder is
    // FP32 throughout. `scaled_add` is the operator that widens and narrows in the diffusion path, and
    // no row of it existed at any of these four widths: the encoder's residual adds are the only
    // consumer of that family at a VAE channel count.
    for columns in [128_u32, 256, 512, 1024] {
        push(
            "scaled_add",
            ops::residual::ScaledAddDescriptor {
                target: DType::F32,
                source: DType::F32,
                columns,
                block: ops::residual::RESIDUAL_BLOCK,
            }
            .to_string(),
        );
    }

    // ---- The video VAE's encoder, `vae/config.json` and `encoding.rs`'s own construction.
    //
    // The encoder is FP32 and every convolution passes `tf32 = true` at `Conv3d::new`, so the flag is
    // the model's declaration rather than a preset's inference. Its geometry is cubic with zero
    // padding and unit dilation (`encoding.rs::geometry`), and the downsampler of a stage strides by
    // that stage's factors: `spatial_downsample_factors [2,2,2,2,1,1]` against
    // `temporal_downsample_factors [1,2,2,1,1,1]` gives three distinct pairs — `[1,2,2]`, `[2,2,2]`
    // and the two stages whose product is 1, which carry no downsampler at all — so four rows cover
    // every convolution of the encoder and the whole of `quant_conv`.
    for (kernel, stride) in [
        ([3_u32, 3, 3], [1_u32, 1, 1]),
        ([3, 3, 3], [1, 2, 2]),
        ([3, 3, 3], [2, 2, 2]),
        ([1, 1, 1], [1, 1, 1]),
    ] {
        push(
            "conv3d",
            ops::conv::Descriptor {
                input: DType::F32,
                weight: DType::F32,
                output: DType::F32,
                geometry: ops::conv::Geometry {
                    kernel,
                    stride,
                    padding: [0, 0, 0],
                    dilation: [1, 1, 1],
                    groups: 1,
                },
                bias: true,
                tf32: true,
                rank: 3,
                transposed: false,
            }
            .to_string(),
        );
    }
    // Its per-frame group statistics over `norm_num_groups 32` with `norm_eps 1e-6`.
    push(
        "frame_moments",
        ops::frame::FrameMomentsDescriptor {
            dtype: DType::F32,
            // The encoder's widest frame: four channels per group over a 1344x768 raster. The class
            // is a declared bound and not a compilation, so it does not reach the key - the launch
            // refuses a frame outside it.
            reduction: ops::frame::frame_reduction_class(4, 1344 * 768),
            epsilon: ops::descriptor::Float::from(1e-6_f32),
        }
        .to_string(),
    );
    // Its normalize-then-pad, with the affine parameters the checkpoint carries (`centered` is the
    // form without them) and `spatial_padding_mode reflect`. `single_pixel` is the descriptor's form
    // for a one-pixel raster, which needs no padding at all: both are listed because which one a
    // request takes depends on the clip's latent grid, not on a compiled choice.
    for single_pixel in [false, true] {
        push(
            "frame_norm_pad",
            ops::frame::FrameNormPadDescriptor {
                dtype: DType::F32,
                reduction: ops::frame::frame_reduction_class(4, 1344 * 768),
                channels_last: false,
                centered: false,
                single_pixel,
            }
            .to_string(),
        );
    }

    // ---- The text stack's attention, at its own geometry.
    //
    // The language layers are a 64-head network with 8 key/value heads (GQA) of 128 elements each,
    // and the fused normalization-plus-rotation runs at those same head counts: `nn`'s `Attention`
    // builds both descriptors from the heads the partition gives it, so the whole count and the
    // Ulysses-2 shard are two calls of the same block rather than two code paths. Every key here is
    // bound by a `t2va` request's first stage, which is why they belong in the list even though no
    // DiT stage binds them.
    //
    // **The mask is `causal`, and the tower's is `dense`: one stage reads both.** The base text
    // encoder hands the language stack `VarlenInput(lengths, lengths, (True,) * len(counts))`
    // (`uniserve/models/encoder.py:482`) - a prompt is a causal sequence - while the vision tower
    // hands its rows `(False,)` (`uniserve/models/qwen3_vl/vision.py:281`), because a patch sees the
    // whole image. This table said `dense` for the text stack until a device asked for
    // `mask_causal` by name on the first block of stage 4, which is the only instrument that could
    // have told us: both keys have the same shape, the same heads and the same dtypes.
    for heads in [64_u32, 32] {
        push(
            "dense_attention",
            ops::attention::dense::DenseDescriptor {
                query: deployment,
                key_value: deployment,
                output: deployment,
                head_dim: 128,
                q_heads: heads,
                kv_heads: heads / 8,
                mask: ops::attention::Mask::Causal,
                windowed: false,
                // A prompt's token count is a launch value, not part of the key: the bound below is
                // the text capacity's own, and the reference masks query rows past the prompt.
                max_kv_len: 4096,
                cu_seq_lens: true,
            }
            .to_string(),
        );
        push(
            "qk_norm_rope",
            ops::rope::QkNormRopeDescriptor {
                dtype: deployment,
                output: deployment,
                head_dim: 128,
                q_heads: heads,
                k_heads: heads / 8,
                // Qwen3-VL pairs `x[i]` with `x[i + d/2]` over the whole head, and its RMS floor is
                // the network's own 1e-6 (`config.json`'s `rms_norm_eps`), unlike the DiT's.
                rotation: ops::rope::Rotation::Half,
                // The literal is `f32` on purpose: the config's `rms_norm_eps` reaches the
                // descriptor as a 32-bit float, and a 64-bit literal would name a different key.
                epsilon: ops::Float::from(1.0e-6_f32),
                rotated: 128,
                rows: 1,
                block: ROTATE_BLOCK,
                affine: true,
                pdl: PDL,
            }
            .to_string(),
        );
    }
    // ---- The text encoder's MLP and its token embedding, and the two packed gated activations.
    //
    // The language layers' MLP is a `GatedActivation` over two *separate* contiguous projections —
    // `text_encoder.rs` runs its own `gate` and `up` Linears into their own `[tokens, intermediate]`
    // buffers before the activation — so both row strides are the intermediate width, 25600. The
    // activation is `silu` (`hidden_act`), the layer's block is 1024, and `wide` is false because the
    // output is in bytes far below the u32 element count.
    push(
        "act_and_mul",
        ops::activation::ActAndMulDescriptor {
            dtype: deployment,
            width: 25600,
            block: ELEMENTWISE_BLOCK,
            activation: ops::activation::Activation::Silu,
            gate_stride: 25600,
            value_stride: 25600,
            wide: false,
            pdl: PDL,
            stepwise: false,
        }
        .to_string(),
    );
    // The DiT's block MLP, whose halves are packed beside each other in one `[rows, 2 · 14336]`
    // projection: both row strides are twice the width, and the reference rounds each row's output
    // to the activation's dtype as it goes, so the site declares the stepwise form.
    push(
        "act_and_mul",
        ops::activation::ActAndMulDescriptor {
            dtype: deployment,
            width: 14336,
            block: ELEMENTWISE_BLOCK,
            activation: ops::activation::Activation::Silu,
            gate_stride: 2 * 14336,
            value_stride: 2 * 14336,
            wide: false,
            pdl: PDL,
            stepwise: true,
        }
        .to_string(),
    );
    // **The refiner's feed-forward**: the same width and the same activation as the DiT's block MLP
    // and a different program, because `conditioning.rs` runs its gate and up projections into two
    // *separate* contiguous buffers - both row strides are the intermediate width, while the DiT
    // reads the halves of one packed projection at twice it. A walk of this tree launches both, so
    // both are rows; one row at the packed stride would leave the refiner's every conditioned
    // request without a program.
    push(
        "act_and_mul",
        ops::activation::ActAndMulDescriptor {
            dtype: deployment,
            width: 14336,
            block: ELEMENTWISE_BLOCK,
            activation: ops::activation::Activation::Silu,
            gate_stride: 14336,
            value_stride: 14336,
            wide: false,
            pdl: PDL,
            stepwise: true,
        }
        .to_string(),
    );
    // The two packed forms: the DiT's block MLP and the conditioning projection both contract to
    // `ffn_dim 14336` in BF16 from a packed `[rows, 2 · width]` row, and the video VAE decoder's to
    // its own `mlp` width 8192 in FP16 (`decoders.md`'s arena: `mlp [rows, 8192]`). The row stride is
    // twice the width because the two halves are packed beside each other, and no call site passes a
    // bias.
    push(
        "value_first_swiglu",
        ops::activation::ValueFirstSwigluDescriptor {
            dtype: deployment,
            width: 14336,
            block: ELEMENTWISE_BLOCK,
            input_stride: 2 * 14336,
            bias: false,
            absmax: false,
        }
        .to_string(),
    );
    for vae in VAES {
        push(
            "value_first_swiglu",
            ops::activation::ValueFirstSwigluDescriptor {
                dtype: vae,
                width: VAE_MLP,
                block: ELEMENTWISE_BLOCK,
                input_stride: 2 * u64::from(VAE_MLP),
                bias: false,
                absmax: false,
            }
            .to_string(),
        );
    }
    // The text encoder's token embedding, at the single-partition shape: the whole vocabulary
    // (151936 rows of 5120) with i32 ids, eight tokens per program. A sharded deployment re-keys it,
    // because `first` and `rows` are the rank's slice of the vocabulary.
    push(
        "embedding",
        ops::embedding::Descriptor {
            dtype: deployment,
            ids: DType::I32,
            hidden: 5120,
            first: 0,
            rows: 151936,
            block: EMBEDDING_TOKENS,
            pdl: PDL,
        }
        .to_string(),
    );

    // The encoder's causal frame padding, in FP32 with reflect edges (`spatial_padding_mode`).
    //
    // **Its tile is a function of the raster's width**, not a constant: `FramePadDescriptor::tiles`
    // takes `min(ROW_PIXELS 512, next_power_of_two(columns))` pixels per program and divides
    // `ROW_ELEMENTS 2048` by it, so a raster at least 512 pixels wide gives the 512-pixel tile with
    // four channels per program — every realistic clip width — and a narrower one re-keys to its own
    // pixel block. This row is the wide case; a request whose width is under 512 compiles another.
    push(
        "frame_pad",
        ops::frame::FramePadDescriptor {
            dtype: DType::F32,
            reflect: true,
            block_pixels: 512,
            block_channels: 4,
        }
        .to_string(),
    );

    // ---- The audio decoder's and encoder's convolutions, in FP32 with TF32 declared on.
    //
    // The decoder's seven upsamplers are `ConvTranspose1d` with `kernel = decoder_kernel_sizes`,
    // `stride = decoder_rates`, `padding = (kernel - rate) / 2` — two distinct geometries for the
    // seven stages (`autoencoder_kl_minimax_h3_audio.py:447-459`): `(9, 5, 2)` twice and `(4, 2, 1)`
    // five times. The key names the geometry, not the channels, so two rows cover all seven.
    // The key builder: the operator's name and the descriptor, pushed by the caller so that one
    // closure does not hold `push` for the rest of the section.
    let convolution = |kernel: [u32; 3],
                       stride: [u32; 3],
                       padding: [u32; 3],
                       dilation: [u32; 3],
                       transposed: bool|
     -> (&'static str, String) {
        (
            if transposed {
                "conv_transpose1d"
            } else {
                "conv1d"
            },
            ops::conv::Descriptor {
                input: DType::F32,
                weight: DType::F32,
                output: DType::F32,
                geometry: ops::conv::Geometry {
                    kernel,
                    stride,
                    padding,
                    dilation,
                    groups: 1,
                },
                bias: true,
                // The preset's declaration for a model convolution in FP32 (the video VAE keeps it
                // on, and the audio VAEs are the same FP32 numerical path).
                tf32: true,
                rank: 1,
                transposed,
            }
            .to_string(),
        )
    };
    let (operator, key) = convolution([9, 1, 1], [5, 1, 1], [2, 1, 1], [1, 1, 1], true);
    push(operator, key);
    let (operator, key) = convolution([4, 1, 1], [2, 1, 1], [1, 1, 1], [1, 1, 1], true);
    push(operator, key);
    // The decoder's entry and exit convolutions, and the pointwise narrowing inside a residual unit.
    let (operator, key) = convolution([7, 1, 1], [1, 1, 1], [3, 1, 1], [1, 1, 1], false);
    push(operator, key);
    let (operator, key) = convolution([1, 1, 1], [1, 1, 1], [0, 1, 1], [1, 1, 1], false);
    push(operator, key);
    // The decoder's residual units: the three resblock kernels, each at the three dilations
    // (`resblock_kernel_sizes [3, 7, 11]`, `resblock_dilation_sizes [1, 3, 5]`).
    for kernel in [3_u32, 7, 11] {
        for dilation in [1_u32, 3, 5] {
            push(
                "conv1d",
                ops::conv::Descriptor {
                    input: DType::F32,
                    weight: DType::F32,
                    output: DType::F32,
                    geometry: ops::conv::Geometry {
                        kernel: [kernel, 1, 1],
                        stride: [1, 1, 1],
                        padding: [(dilation * (kernel - 1)) / 2, 1, 1],
                        dilation: [dilation, 1, 1],
                        groups: 1,
                    },
                    bias: true,
                    tf32: true,
                    rank: 1,
                    transposed: false,
                }
                .to_string(),
            );
        }
    }
    // The encoder's residual units are a seven-tap convolution at dilations 1, 3 and 9
    // (`audio_vae.py:352-354`, padding `3 · dilation`), and its exit convolution is three taps at
    // padding 1. Two of these rows coincide with the decoder's seven-tap rows above.
    for dilation in [1_u32, 3, 9] {
        let (operator, key) = convolution(
            [7, 1, 1],
            [1, 1, 1],
            [3 * dilation, 1, 1],
            [dilation, 1, 1],
            false,
        );
        push(operator, key);
    }
    let (operator, key) = convolution([3, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], false);
    push(operator, key);
    // The encoder's strided convolutions: `kernel = 2 · stride`, `padding = ceil(stride / 2)`
    // (`audio_vae.py:355-363`), for the three distinct rates of `encoder_rates [2, 4, 4, 5, 5]`.
    for stride in [2_u32, 4, 5] {
        let (operator, key) = convolution(
            [2 * stride, 1, 1],
            [stride, 1, 1],
            [stride.div_ceil(2), 1, 1],
            [1, 1, 1],
            false,
        );
        push(operator, key);
    }

    // ---- The VSA selection's pooled-score product: the stage-1 video attention's per-head tile
    // similarity, over the transformer's own head geometry (56 heads of 128). The operands are the
    // pooled means in FP32 and so are the scores, so no element type is in the key; the lanes one
    // program uses for the contraction are the head width, and the pooling itself (the tile
    // reduction and its stride) is `vsa_prepare_tiles`' launch values.
    push(
        "vsa_scores",
        ops::segments::ScoresDescriptor::new(HEADS, HEAD_DIM, HEAD_DIM.next_power_of_two(), PDL)
            .to_string(),
    );

    // The compression the gate adds to every query row: one softmax over the key tiles. `max_tiles`
    // is the deployment's longest sparse reduction — 40960 keys at the 64-key tile — because it is
    // the block the softmax reduces in, and so the reduction length the tolerance is written
    // against. The pooled scores and values are FP32, so the result's type is the whole of `dtype`.
    push(
        "vsa_compress_pooled",
        ops::segments::CompressPooledDescriptor::new(
            deployment,
            HEADS,
            HEAD_DIM,
            VSA_MAX_TILES,
            HEAD_DIM.next_power_of_two(),
            PDL,
        )
        .to_string(),
    );

    // The pooling that feeds it, and the two tile passes around the attention, all at the tile the
    // contract names. Q, K and V are contiguous `[rows, 56, 128]` views of one projected interval, so
    // a row is 7168 elements and a head 128; stage-1 Q and K are normalized and rotated over the
    // leading 96 of their 128 channels (`attention.py`'s `vsa.NormRope`), which is `ROTATED`.
    push(
        "vsa_prepare_tiles",
        ops::segments::PrepareTilesDescriptor {
            dtype: deployment,
            heads: HEADS,
            width: HEAD_DIM,
            rotated: ROTATED,
            tile_rows: VSA_TILE_ROWS,
            block_rows: VSA_BLOCK_ROWS,
            epsilon: ops::descriptor::Float::from(NORM_EPSILON),
            norm: true,
            query_row_stride: u64::from(HEADS_TIMES_HEAD_DIM),
            query_head_stride: u64::from(HEAD_DIM),
            key_row_stride: u64::from(HEADS_TIMES_HEAD_DIM),
            key_head_stride: u64::from(HEAD_DIM),
            value_row_stride: u64::from(HEADS_TIMES_HEAD_DIM),
            value_head_stride: u64::from(HEAD_DIM),
            // A split-half factor table holds `rotated / 2` values a row.
            factor_stride: u64::from(ROTATED / 2),
        }
        .to_string(),
    );
    // A tile's rows past its valid size hold zeros, on both sides of the attention.
    push(
        "vsa_zero_padding",
        ops::segments::ZeroPaddingDescriptor {
            dtype: deployment,
            heads: HEADS,
            tile_rows: VSA_TILE_ROWS,
            width: HEAD_DIM,
            block_rows: VSA_BLOCK_ROWS,
        }
        .to_string(),
    );
    // The selection, one launch per sparse layer over the same tile extents the scores hold: those
    // extents are the deployment's tile *bound* (what the sparse scratch allocates for the scores,
    // the indices and the counts), and the sparsity only decides which tiles the keep table names.
    // So `block` is a class like the compression's `max_tiles`, not a per-request count.
    push(
        "vsa_select_tiles",
        ops::segments::SelectTilesDescriptor {
            block: VSA_MAX_TILES.next_power_of_two(),
        }
        .to_string(),
    );
    // The compressed gated tiles that give back the sparse rows.
    push(
        "vsa_add_gated_tiles",
        ops::segments::AddGatedTilesDescriptor {
            dtype: deployment,
            tile_rows: VSA_TILE_ROWS,
            width: HEAD_DIM,
            block_rows: VSA_BLOCK_ROWS,
        }
        .to_string(),
    );

    // ---- The audio encoder's frame/channel swap, in the audio VAEs' own FP32.
    //
    // `AudioVAEEncoder` holds `[batch, channels, frames]` and its attention attends over frames, so
    // frames move to the leading axis before the qkv projection and back after it. Both extents are
    // launch values, so one row covers every clip length.
    push(
        "transpose",
        ops::transpose::Descriptor {
            dtype: DType::F32,
            block: ops::transpose::TILE,
            pdl: PDL,
        }
        .to_string(),
    );

    // Placing the rows a tower contributes into the sequence that reserved them. Four call sites
    // exist and each is in a layer of `crates/models/minimax-h3`, all of them at the layer crate's
    // one element block (its `rows.rs`'s `BLOCK`, 1024) and none on a dependent launch:
    //
    // - the text encoder's prompt reserves placeholder rows for its images and the tower's vision
    //   rows *replace* them (`text_encoder.rs`, `Rows::replace`), at the language width;
    // - a DeepStack tap *adds* a tower layer's rows into rows that already hold a residual
    //   (`text_encoder.rs`, `Rows::add`), at the same width;
    // - the denoiser scatters its packed video rows and its audio rows into the one sequence
    //   (`denoiser.rs`, `Rows::replace`), at the transformer's hidden width — one key for both
    //   modalities, because the operator reads only rows, columns and the index view;
    // - the video VAE decoder starts its stream by placing the projected voxels, its register rows
    //   and its zero row into the token rows (`video_vae.rs`, `Rows::replace`), at the decoder's
    //   `decoder_num_attention_heads · decoder_attention_head_dim = 32 · 64` and in FP16.
    //
    // The merge that makes a language-width token out of a `2x2` block of patches is *not* here: it
    // is a reshape of the contiguous `[patches, 1152]` hidden state into `[patches/4, 4608]`, so no
    // rows are gathered and no program reads them.
    for (dtype, mode, columns) in [
        (deployment, ops::rows::Mode::Replace, text_columns),
        (deployment, ops::rows::Mode::Add, text_columns),
        (deployment, ops::rows::Mode::Replace, HIDDEN),
        // The voxel rows' own width, at the VAE's representation rather than the stack's.
        (VAES[0], ops::rows::Mode::Replace, VAE_WIDTH),
        (VAES[1], ops::rows::Mode::Replace, VAE_WIDTH),
    ] {
        push(
            "rows",
            ops::rows::Descriptor {
                dtype,
                mode,
                columns,
                block: ROWS_BLOCK,
                pdl: PDL,
            }
            .to_string(),
        );
    }

    // Gaps this file does not fill, named rather than invented. Each is a number the deployment
    // owns, not one this file could read off a config:
    // - `bias_add`: the audio decoder's channels are derivable and the video VAE's are not yet, and
    //   in both cases the two flags belong to the layer that calls it. The audio decoder halves
    //   `decoder_dim` once per upsample (`diffusers`' `MiniMaxH3AudioBigVGANDecoder.ups`:
    //   `upsample_initial_channel // 2**i` into `// 2**(i+1)`, and its resblocks sit at the output
    //   width), so its call sites are 1024, 512, 256, 128, 64, 32, 16 and 8; the video VAE's widths
    //   come from its own stage list once that decoder is composed. `residual` and `residual_bias`
    //   decide the key, and they are how the block folds its residual — the layer's, not the config's.
    // - `paged_attention`: the decode attention is trtllm-gen, and its key follows the executor's
    //   page geometry (page size, pages per sequence), which is scheduling and not the model's.
    // - `vsa_select_tiles` is above with the other four tile passes: its block is the tile bound's
    //   power of two, which the deployment allocates, and `attention_sparse.rs` launches it once per
    //   sparse layer of every evaluation.
    // - `softmax`, `swiglu`, `scaled_residual_rms_norm`, `scaled_residual_layer_norm`,
    //   `vsa_compose_gated_rows`: declared, and nothing in either model's code calls them yet, so
    //   there is no geometry to key. They are named here rather than left to be found: the first two
    //   belong to the sampler's chain and the Qwen3 slice, the two residuals to a stream that folds
    //   its normalization in one launch, and the composition to the VSA chain's last step.

    lines
}
