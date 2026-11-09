//! The widths and shapes one denoising step crosses, walked on the host.
//!
//! A device run of the denoise stage costs a checkpoint load and a capture, and the mismatches it
//! finds are almost always a **width or a row count stated twice**: the text encoder's features and
//! the denoiser's rows are two widths (`text_dim` and `hidden_size`, joined by the refiner's input
//! projection), a layout's prefix is a capacity rather than a prompt, and the plan's buffers are
//! counted in the stack's element type. Each of those was found on a device once; this walks the
//! same numbers from the configuration, the layout and the plan, with no device and no weights.
//!
//! Every line names the statement it comes from - the config field, the layout's count, or the plan
//! entry - so a disagreement points at one of them rather than at a run.
//!
//! ```text
//! cargo run -p minimax-h3 --example denoise_shapes -- \
//!     --root MiniMaxAI/MiniMax-H3 --width 768 --height 768 --frames 124 --text-tokens 1024
//! ```
//!
//! It is a diagnosis, never a gate (`specs/worker.md` 10): the numbers it prints are the model's own
//! statements, and the decision it records is whether they agree with each other.

use std::path::PathBuf;
use std::process::ExitCode;

use minimax_h3::config::Config;
use minimax_h3::denoiser::{self, Layouts};
use minimax_h3::partition::{Partition, Ulysses};
use nn::capability::{ConditionRole, ConditionShape, VideoRequest};
use nn::{DType, VideoTask};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    root: PathBuf,
    width: u32,
    height: u32,
    frames: u32,
    /// The prompt capacity the deployment prepares, which is the layout's text rows.
    text_tokens: Option<u32>,
    /// The width the runner sizes its conditioning columns at.
    ///
    /// One number for both buffers is what `DenoiserShape::width` gives a runner today: the text
    /// encoder's hidden width. The served walk's stop - `gather_rows: out is [rows, 5376],
    /// [rows, 5120] expected` - is this width against the rows the denoiser reads, so the walk takes
    /// it as an input and refuses the pair when they disagree.
    column_width: Option<u32>,
    ulysses: usize,
    dtype: DType,
    /// The task the request is served for: `t2va`, `fl2va` or `ref2va`.
    task: VideoTask,
}

fn usage() -> ! {
    eprintln!(
        "usage: denoise_shapes --root <checkpoint> [--width W] [--height H] [--frames F] \
         [--text-tokens N] [--ulysses D] [--dtype bf16|f16|f32] [--task t2va|fl2va|ref2va]"
    );
    std::process::exit(2)
}

fn run() -> Result<(), String> {
    let mut args = Args {
        root: PathBuf::new(),
        width: 768,
        height: 768,
        frames: 124,
        text_tokens: None,
        column_width: None,
        ulysses: 1,
        dtype: DType::BF16,
        task: VideoTask::TextToVideoAudio,
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = |_name: &str| arguments.next().unwrap_or_else(|| usage()).to_string();
        match flag.as_str() {
            "--root" => args.root = PathBuf::from(value("--root")),
            "--width" => args.width = number(&value("--width"), "--width")?,
            "--height" => args.height = number(&value("--height"), "--height")?,
            "--frames" => args.frames = number(&value("--frames"), "--frames")?,
            "--text-tokens" => {
                args.text_tokens = Some(number(&value("--text-tokens"), "--text-tokens")?)
            }
            "--column-width" => {
                args.column_width = Some(number(&value("--column-width"), "--column-width")?)
            }
            "--ulysses" => args.ulysses = number(&value("--ulysses"), "--ulysses")? as usize,
            "--task" => {
                args.task = match value("--task").as_str() {
                    "t2va" => VideoTask::TextToVideoAudio,
                    "fl2va" => VideoTask::FirstLastToVideoAudio,
                    "ref2va" => VideoTask::ReferenceToVideoAudio,
                    other => {
                        return Err(format!("--task takes t2va, fl2va or ref2va, not `{other}`"));
                    }
                }
            }
            "--dtype" => {
                let stated = value("--dtype").to_ascii_lowercase();
                args.dtype = match stated.as_str() {
                    "bf16" => DType::BF16,
                    "f16" => DType::F16,
                    "f32" => DType::F32,
                    other => return Err(format!("--dtype takes bf16, f16 or f32, not `{other}`")),
                };
            }
            other => return Err(format!("unknown flag `{other}`")),
        }
    }
    if args.root.as_os_str().is_empty() {
        usage();
    }

    let config = Config::read(&args.root, None).map_err(|error| error.to_string())?;
    let task = args.task;
    let serving = Partition::serving(
        &config,
        task,
        Ulysses::of(args.ulysses, minimax_h3::partition::Projection::Exchange),
        1,
        0,
        0,
    )
    .map_err(|error| error.to_string())?;
    let text_rows = args.text_tokens.unwrap_or(serving.text_rows);
    let partition = serving.sized(text_rows, serving.condition_rows);
    let denoiser = config
        .denoisers
        .get(&minimax_h3::checkpoint::Dit::Base)
        .ok_or_else(|| {
            format!(
                "the checkpoint at {} holds no base DiT",
                args.root.display()
            )
        })?
        .clone();
    let arrangement = partition
        .arrangement(&denoiser)
        .map_err(|error| error.to_string())?;
    let layouts = Layouts::new(denoiser.clone(), arrangement).map_err(|error| error.to_string())?;
    // A conditioned task states its conditions and the counts that name them: a first-and-last-frame
    // request anchors two frames of the canvas, and a reference is a clip of its own raster. The
    // condition rows they produce are the layout's, and their offsets are what `write_state` writes.
    let conditions: Vec<ConditionShape> = match task {
        VideoTask::TextToVideoAudio => Vec::new(),
        VideoTask::FirstLastToVideoAudio => vec![
            ConditionShape {
                role: ConditionRole::FirstFrame,
                frames: 1,
                height: args.height,
                width: args.width,
                audio_samples: 0,
            },
            ConditionShape {
                role: ConditionRole::LastFrame,
                frames: 1,
                height: args.height,
                width: args.width,
                audio_samples: 0,
            },
        ],
        VideoTask::ReferenceToVideoAudio => vec![ConditionShape {
            role: ConditionRole::Reference,
            frames: 8,
            height: args.height,
            width: args.width,
            audio_samples: 0,
        }],
    };
    let first_frames = conditions
        .iter()
        .filter(|shape| shape.role == ConditionRole::FirstFrame)
        .count() as u32;
    let last_frames = conditions
        .iter()
        .filter(|shape| shape.role == ConditionRole::LastFrame)
        .count() as u32;
    let references = conditions
        .iter()
        .filter(|shape| shape.role == ConditionRole::Reference)
        .count() as u32;
    let request = VideoRequest {
        task,
        width: args.width,
        height: args.height,
        frames: args.frames,
        fps: minimax_h3::config::FPS,
        first_frames,
        last_frames,
        references,
        text_tokens: text_rows,
        conditions: &conditions,
        vision_spans: &[],
        condition_rows: None,
    };
    let layout = layouts.cover(&request);

    // 1. The widths the checkpoint states, and the one the model adds to them.
    let mut disagreements: Vec<String> = Vec::new();
    let transformer = &denoiser.transformer;
    let hidden = u64::from(transformer.hidden_size);
    let text_dim = u64::from(transformer.text_dim);
    println!("widths");
    println!("  hidden_size          {hidden}   (the packed rows, and the stack's own width)");
    println!("  text_dim             {text_dim}   (the text encoder's features)");
    println!(
        "  refiner input        [{hidden}, {text_dim}]   (context_embedder: the widening layer)"
    );
    println!(
        "  intermediate_size    {}   (the feed-forward's packed projection)",
        transformer.intermediate_size
    );
    println!(
        "  heads x head_dim     {} x {} = {}",
        transformer.num_attention_heads,
        transformer.head_dim,
        u64::from(transformer.num_attention_heads) * u64::from(transformer.head_dim)
    );
    println!("  audio_channels       {}", transformer.audio_channels);
    // The conditioning contract, which is the one width pair a runner owns. The loaded model states
    // it (`nn::capability::Denoiser::text_width`, H3 answering `hidden_size`); from a configuration
    // the two halves are the same two numbers.
    println!("conditioning");
    println!(
        "  text   [rows, {hidden}]   what the denoiser reads (prepare_conditioning writes it)"
    );
    println!("  source [rows, {text_dim}]   what the text encoder wrote");
    if text_dim == hidden {
        println!("  one width: the refiner does not widen, so a runner may size both alike");
    } else {
        println!(
            "  **two widths**: a runner that sizes both from {text_dim} hands the refiner a buffer \
             too narrow for its own output"
        );
    }
    // What the runner actually sizes its two columns at, which is the input the served walk met.
    // One number for both is `DenoiserShape::width`'s answer: the encoder's.
    let column = u64::from(args.column_width.unwrap_or(transformer.text_dim));
    println!("  runner columns       text [{column}], source [{text_dim}]");
    if column != hidden {
        disagreements.push(format!(
            "the runner sizes the text column at {column} and the denoiser reads rows {hidden} \
             wide: `gather_rows` will refuse `out is [rows, {hidden}], [rows, {column}] expected` \
             - the runner's `text` column takes `Denoiser::text_width`, and the encoder's product \
             fills `source`"
        ));
    }

    // 2. The layout: a capacity, not a prompt.
    println!("layout of {task:?}");
    println!(
        "  text_rows {}  condition_rows {}  condition_elements {}",
        layout.text_rows, layout.condition_rows, layout.condition_elements
    );
    println!(
        "  padded_tokens {}  (from `denoise::packed_rows`, the forward's own packing)",
        denoiser::packed_rows(&denoiser, &arrangement, &layout)
            .map_err(|error| error.to_string())?
    );
    println!(
        "  video latent [channels {}, frames {}, height {}, width {}]",
        layout.video.channels, layout.video.frames, layout.video.height, layout.video.width
    );
    if let Some(audio) = layout.audio {
        println!(
            "  audio latent [channels {}, frames {}]",
            audio.channels, audio.frames
        );
    }

    // 3. The checks that are statements about statements.
    let tokens = u64::from(
        denoiser::packed_rows(&denoiser, &arrangement, &layout)
            .map_err(|error| error.to_string())?,
    );
    if layout.text_rows != arrangement.text_rows {
        disagreements.push(format!(
            "the layout holds {} text rows and the arrangement states {}",
            layout.text_rows, arrangement.text_rows
        ));
    }
    if u64::from(layout.text_rows) + u64::from(layout.condition_rows) > tokens {
        disagreements.push(format!(
            "the prefix alone ({} + {} rows) is longer than the packed rows ({tokens})",
            layout.text_rows, layout.condition_rows
        ));
    }
    if request.text_tokens > layout.text_rows {
        disagreements.push(format!(
            "a request of {} prompt rows does not fit the layout's {}",
            request.text_tokens, layout.text_rows
        ));
    }

    // 4. The plan: every buffer one evaluation reserves, in the forward's own order.
    let plan = denoiser::denoise_arena(&denoiser, &arrangement, 1, args.dtype, &layout)
        .map_err(|error| error.to_string())?;
    let total: u64 = plan.iter().map(|buffer| buffer.elements).sum();
    println!("plan  {} buffers", plan.len());
    for buffer in &plan {
        let width = if tokens > 0 && buffer.elements % tokens == 0 {
            format!(" = {tokens} rows x {}", buffer.elements / tokens)
        } else {
            String::new()
        };
        println!(
            "  {:<28} {:<5} {:>12} elements{width}",
            buffer.name, buffer.element, buffer.elements
        );
        if buffer.elements == 0 {
            disagreements.push(format!("plan entry `{}` holds no elements", buffer.name));
        }
    }
    println!("  total {total} elements at {}", args.dtype);

    // 6. The stage's floor, which is the LARGER of the evaluation and the prompt's fold: the two
    // occupy the same offsets, because the capture walk runs the evaluation alone from the region's
    // base and the fold runs only on the live path, before the first replay.
    let fold_rows = layout.text_rows.saturating_add(layout.condition_rows);
    let fold = minimax_h3::conditioning::fold_bytes(&denoiser.transformer, fold_rows, args.dtype);
    // The plan's own reservation, not the element count times one width: a plan carries `f32`,
    // `bf16` and `i32` entries, and the worker sums each entry's own width.
    let evaluation = minimax_h3::decoding::arena_reservation(&plan);
    println!(
        "floor  max(evaluation {} MiB, fold {} MiB) = {} MiB",
        evaluation / (1 << 20),
        fold / (1 << 20),
        evaluation.max(fold) / (1 << 20)
    );
    if fold > evaluation {
        disagreements.push(format!(
            "the prompt's fold reserves {} MiB and one evaluation {evaluation} MiB: the floor is \
             the fold here, and a deployment sized by the evaluation alone would stop at its first \
             request",
            fold / (1 << 20)
        ));
    }

    // 5. The state tables the runner writes from this layout, with their values.
    //
    // `prepare_state` is the worker's per-request call and `write_state` is a host function, so
    // this runs the same code the served walk runs before its first launch - and it is where
    // `write_segment_state` refuses a table whose rows are not the packing's own.
    let specs = layouts
        .table_specs(&layout)
        .map_err(|error| error.to_string())?;
    let mut written: Vec<Vec<u8>> = Vec::with_capacity(specs.len());
    for spec in &specs {
        let elements = spec.shape.dims().iter().product::<u64>();
        let width = spec
            .dtype
            .bytes()
            .ok_or_else(|| format!("the state table `{}` is {}", spec.name, spec.dtype))?;
        let bytes = usize::try_from(elements * width)
            .map_err(|_| format!("the state table `{}` is larger than memory", spec.name))?;
        written.push(vec![0u8; bytes]);
    }
    layouts
        .write_state(&request, &layout, &mut written)
        .map_err(|error| error.to_string())?;
    println!("state tables  {} written", specs.len());
    for (spec, bytes) in specs.iter().zip(&written) {
        let non_zero = bytes.iter().filter(|byte| **byte != 0).count();
        println!(
            "  {:<20} {:<4} {:?}  {} byte(s), {non_zero} non-zero",
            spec.name,
            spec.dtype.to_string(),
            spec.shape.dims(),
            bytes.len()
        );
    }

    if disagreements.is_empty() {
        println!(
            "denoise shapes: {} statement(s) agree",
            plan.len() + 6 + specs.len()
        );
        Ok(())
    } else {
        for disagreement in &disagreements {
            println!("disagreement: {disagreement}");
        }
        Err(format!("{} disagreement(s)", disagreements.len()))
    }
}

fn number(value: &str, flag: &str) -> Result<u32, String> {
    value
        .parse()
        .map_err(|_| format!("{flag} takes a whole number, not `{value}`"))
}
