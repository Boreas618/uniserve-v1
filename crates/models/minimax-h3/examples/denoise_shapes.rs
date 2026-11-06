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
//!     --root /opt/uv/models/MiniMax-H3 --width 768 --height 768 --frames 124 --text-tokens 1024
//! ```
//!
//! It is a diagnosis, never a gate (`specs/worker.md` 10): the numbers it prints are the model's own
//! statements, and the decision it records is whether they agree with each other.

use std::path::PathBuf;
use std::process::ExitCode;

use minimax_h3::config::Config;
use minimax_h3::denoiser::{self, Layouts};
use minimax_h3::partition::{Partition, Ulysses};
use nn::capability::VideoRequest;
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
}

fn usage() -> ! {
    eprintln!(
        "usage: denoise_shapes --root <checkpoint> [--width W] [--height H] [--frames F] \
         [--text-tokens N] [--ulysses D] [--dtype bf16|f16|f32]"
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
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = |_name: &str| arguments.next().unwrap_or_else(|| usage()).to_string();
        match flag.as_str() {
            "--root" => args.root = PathBuf::from(value("--root")),
            "--width" => args.width = number(&value("--width"), "--width")?,
            "--height" => args.height = number(&value("--height"), "--height")?,
            "--frames" => args.frames = number(&value("--frames"), "--frames")?,
            "--text-tokens" => args.text_tokens = Some(number(&value("--text-tokens"), "--text-tokens")?),
            "--column-width" => {
                args.column_width = Some(number(&value("--column-width"), "--column-width")?)
            }
            "--ulysses" => args.ulysses = number(&value("--ulysses"), "--ulysses")? as usize,
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
    let task = VideoTask::TextToVideoAudio;
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
    let request = VideoRequest {
        task,
        width: args.width,
        height: args.height,
        frames: args.frames,
        fps: minimax_h3::config::FPS,
        first_frames: 0,
        last_frames: 0,
        references: 0,
        text_tokens: text_rows,
        conditions: &[],
        vision_spans: &[],
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
    println!("  text   [rows, {hidden}]   what the denoiser reads (prepare_conditioning writes it)");
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
    println!("  runner columns       text [{column}], source [{column}]");
    if column != hidden {
        disagreements.push(format!(
            "the runner sizes the text column at {column} and the denoiser reads rows {hidden} \
             wide: `gather_rows` will refuse `out is [rows, {hidden}], [rows, {column}] expected` \
             - the runner's `text` column takes `Denoiser::text_width`, and the encoder's product \
             fills `source`"
        ));
    }

    // 2. The layout: a capacity, not a prompt.
    println!("layout");
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
        println!("  audio latent [channels {}, frames {}]", audio.channels, audio.frames);
    }

    // 3. The checks that are statements about statements.
    let tokens = u64::from(
        denoiser::packed_rows(&denoiser, &arrangement, &layout).map_err(|error| error.to_string())?,
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

    if disagreements.is_empty() {
        println!("denoise shapes: {} statement(s) agree", plan.len() + 6);
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
