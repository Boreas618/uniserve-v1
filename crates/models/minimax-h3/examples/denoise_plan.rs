//! Price the denoiser's arena at a layout, and check that its plan is the forward's request list.
//!
//! ```text
//! cargo run -p minimax-h3 --example denoise_plan -- \
//!     --root $UNISERVE_H3_MODEL --extents artifacts/h3-reference/component-tensors.json
//! ```
//!
//! The same instrument the two video stages and the audio decoder have: a recording forward launches
//! nothing, and an arena is bookkeeping over an address nothing dereferences, so one evaluation of the
//! denoiser runs on a host with no device and `Arena::used()` counts what it took.
//!
//! **Two numbers matter and they are different questions.** Sizing the arena to *exactly* the plan's
//! bytes and completing proves the plan covers every request the forward makes - a plan short by one
//! buffer fails here, by name, naming the request's size. Comparing `used()` with the plan afterwards
//! then says whether the plan is exact or reserves something nothing takes; slack is not a defect, but
//! a number a runner sizes storage from should be a number the forward fills.
//!
//! **The preparation is not in the plan and is not measured against it.** `prepare_constants`,
//! `draw_noise` and `refine_conditioning` run once per request from a bump the runner releases before
//! it captures, which is why a conditioned deployment's stage is the plan's own arena and not the sum.
//! So they run here on the caller's arena and the measurement is `denoise` alone.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use nn::capability::Denoiser as _;
use nn::weights::WeightLoader;
use nn::{Arena, Conditioning, DType, Forward, Latents, Shape, Step};

use minimax_h3::Dit;
use minimax_h3::config::AttentionKind;
use minimax_h3::decoding::arena_reservation;
use minimax_h3::denoiser::{Arrangement, Denoiser, Layouts};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

/// One layout to price: a pixel canvas, a clip's frames and a prompt's rows.
struct Case {
    height: u32,
    width: u32,
    frames: u32,
    tokens: u32,
}

fn run() -> Result<(), String> {
    let mut arguments = std::env::args().skip(1);
    let mut root = PathBuf::new();
    let mut extents_path: Option<PathBuf> = None;
    // No `--case` prices the deployment: every canvas the network offers, at the request shape the
    // flags state. A literal raster here would be a second statement of a deployment, which is the
    // error this lane has already paid for twice.
    let mut cases: Option<Vec<Case>> = None;
    let mut clip = 124u32;
    let mut tokens = 512u32;
    let mut sparse: Option<(u32, f64)> = None;
    let mut dtype = DType::BF16;
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
            // `HxW:frames:tokens`, repeated: one case per layout this deployment will serve. The
            // plan is per layout, so a tool that printed one number would say nothing about the
            // canvas a request actually asks for.
            "--case" => {
                let text = value()?;
                cases = Some(
                    text.split(',')
                        .map(|case| case_of(case.trim()))
                        .collect::<Result<Vec<Case>, String>>()?,
                );
            }
            "--clip" => {
                clip = value()?
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| "--clip is not a frame count".to_owned())?
            }
            "--tokens" => {
                tokens = value()?
                    .trim()
                    .parse::<u32>()
                    .map_err(|_| "--tokens is not a row count".to_owned())?
            }
            "--sparse" => {
                let text = value()?;
                sparse = Some(sparse_of(&text)?);
            }
            "--dtype" => {
                dtype = match value()?.to_ascii_lowercase().as_str() {
                    "bf16" => DType::BF16,
                    "f16" => DType::F16,
                    "f32" => DType::F32,
                    other => return Err(format!("`{other}` is not an element type")),
                }
            }
            // `--check` is the acceptance: a plan short of the forward exits non-zero, so a runner
            // that sizes storage from this number cannot ship a short one.
            "--check" => check = true,
            other => return Err(format!("`{other}` is not a flag of this tool")),
        }
    }
    if root.as_os_str().is_empty() {
        root = std::env::var_os("UNISERVE_H3_MODEL")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                eprintln!("name the checkpoint with --root or UNISERVE_H3_MODEL");
                std::process::exit(2)
            });
    }
    let path = extents_path.ok_or_else(|| {
        format!(
            "--extents is required: the weights whose extents are computed at load are the checkpoint's. The table is {}",
            root.join("transformer/config.json").display()
        )
    })?;
    let extents = read_extents(&path)?;

    let config = minimax_h3::Config::read(&root, None).map_err(|error| error.to_string())?;
    let network = config
        .denoisers
        .get(&Dit::Base)
        .ok_or("this checkpoint holds no base network")?
        .clone();
    let mut network = network;
    if let Some((tile, sparsity)) = sparse {
        network.attention =
            AttentionKind::sparse(tile, sparsity, None).map_err(|error| error.to_string())?;
    }
    let hidden = u64::from(network.transformer.hidden_size);
    let text_width = u64::from(config.text_encoder.hidden_size);
    let channels = u64::from(config.video_vae.latent_channels);
    let unit = network.attention.capacity_rows();
    let cases = match cases {
        Some(cases) => cases,
        None => minimax_h3::config::offered_canvases()
            .iter()
            .map(|canvas| Case {
                height: canvas.height,
                width: canvas.width,
                frames: clip,
                tokens,
            })
            .collect(),
    };
    println!(
        "# base network, {} activations, {clip} frames, {tokens} prompt rows, {} layout(s), {}",
        match dtype {
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::F32 => "f32",
            _ => "other",
        },
        cases.len(),
        match sparse {
            Some((tile, sparsity)) => format!("sparse {tile}, {sparsity}"),
            None => "dense attention".to_owned(),
        }
    );

    let mut short = 0usize;
    let mut exact = 0usize;
    let mut slack = 0i128;
    for case in &cases {
        let offered = minimax_h3::config::offered_canvases()
            .iter()
            .any(|canvas| canvas.height == case.height && canvas.width == case.width);
        if !offered {
            return Err(format!(
                "{}x{} is not one of the deployment's canvases",
                case.height, case.width
            ));
        }
        // A `t2va` request carries no condition, so the condition capacity is one unit and holds no
        // rows: the layout's own packed count is what the plan is priced at.
        let arrangement =
            Arrangement::new(&network, case.tokens, unit, 1).map_err(|error| error.to_string())?;
        let rules =
            Layouts::new(network.clone(), arrangement).map_err(|error| error.to_string())?;
        let request = nn::capability::VideoRequest {
            width: case.width,
            height: case.height,
            frames: case.frames,
            fps: 24,
            first_frames: 0,
            last_frames: 0,
            references: 0,
            text_tokens: case.tokens,
            conditions: &[],
            vision_spans: &[],
            condition_rows: None,
        };
        let (layout, packing) = rules.layout(&request).map_err(|error| error.to_string())?;
        let bound = rules.rows_bound().map_err(|error| error.to_string())?;

        let mut loader = WeightLoader::stated(dtype, Some(&extents));
        let denoiser = Denoiser::load(
            network.clone(),
            arrangement,
            bound,
            &minimax_h3::Partition::single(),
            &mut loader,
            None,
        )
        .map_err(|error| error.to_string())?;

        let plan = denoiser
            .denoise_arena(&layout)
            .map_err(|error| error.to_string())?;
        let reserved = arena_reservation(&plan);
        // **Two routes to one number**, which is the only kind of agreement worth having: the plan
        // this tool sums, and the answer the worker's capability call gives. A capability no one
        // calls is a declaration, so the tool calls it here and refuses when the two disagree.
        let stated = Denoiser::arena_bytes(&denoiser, &request)
            .map_err(|error| format!("{}x{}: {error}", case.height, case.width))?;
        if stated != reserved {
            return Err(format!(
                "{}x{} at {} frames: the capability states {stated} B and the plan sums to {reserved} B",
                case.height, case.width, case.frames
            ));
        }

        // The plan's region and the caller's are two: the plan is what one evaluation may take, not
        // what the request holds across its steps.
        let arena = Arena::new(ops::DevicePtr::new(0), reserved);
        let caller = Arena::new(ops::DevicePtr::new(0), 1_u64 << 40);
        let specs = rules
            .table_specs(&layout)
            .map_err(|error| error.to_string())?;
        let mut tables = Vec::with_capacity(specs.len());
        for spec in &specs {
            tables.push(
                caller
                    .tensor(spec.dtype, spec.shape)
                    .map_err(|error| error.to_string())?,
            );
        }
        let text_rows = u64::from(packing.text_rows);
        let source = caller
            .tensor(dtype, Shape::new(&[text_rows, text_width]))
            .map_err(|error| error.to_string())?;
        let text = caller
            .tensor(dtype, Shape::new(&[text_rows, hidden]))
            .map_err(|error| error.to_string())?;
        // The trajectory is FP32 whatever the stack's type is: the solver accumulates into it.
        let video = caller
            .tensor(
                DType::F32,
                Shape::new(&[
                    channels,
                    u64::from(layout.video.frames),
                    u64::from(layout.video.height),
                    u64::from(layout.video.width),
                ]),
            )
            .map_err(|error| error.to_string())?;
        // The audio trajectory is the layout's audio rows as two channels of stereo.
        let audio = caller
            .tensor(
                DType::F32,
                Shape::new(&[2, u64::from(packing.audio_rows) / 2]),
            )
            .map_err(|error| error.to_string())?;
        let latents = Latents {
            video,
            audio: Some(audio),
        };
        let cond = Conditioning {
            text,
            source: None,
            media: &[],
            images: None,
            audio: None,
            state: &tables,
            step: None,
        };
        let step = Step {
            index: 0,
            count: 1,
            sigma: 1.0,
            sigma_next: 0.0,
        };

        let prepared = Forward::recording(&caller);
        denoiser
            .prepare_constants(&prepared, &[])
            .map_err(|error| error.to_string())?;
        denoiser
            .draw_noise(&prepared, nn::capability::Seed(0), layout, latents)
            .map_err(|error| error.to_string())?;
        denoiser
            .refine_conditioning(&prepared, source, text, &tables, None)
            .map_err(|error| error.to_string())?;

        let forward = Forward::recording(&arena);
        denoiser
            .denoise(&forward, &step, latents, cond)
            .map_err(|error| format!("{}x{}: {error}", case.height, case.width))?;
        let used = arena.used();
        if used > reserved {
            short += 1;
            println!(
                "{:>5}x{:<5} {:>4} frames {:>4} tokens: {} buffer(s), {reserved} B planned, {used} B taken, SHORT by {} B",
                case.height,
                case.width,
                case.frames,
                case.tokens,
                plan.len(),
                used - reserved
            );
            if check {
                return Err(format!(
                    "{}x{} at {} frames: the forward takes {used} B of a {reserved}-byte plan, short by {} B",
                    case.height,
                    case.width,
                    case.frames,
                    used - reserved
                ));
            }
            continue;
        }
        exact += usize::from(used == reserved);
        // `u64 -> i128` is infallible: the slack of a plan is not a place to hide a `MAX`.
        slack += i128::from(reserved - used);
        println!(
            "{:>5}x{:<5} {:>4} frames {:>4} tokens: {} buffer(s), {reserved} B planned, {used} B taken, {}",
            case.height,
            case.width,
            case.frames,
            case.tokens,
            plan.len(),
            match reserved.cmp(&used) {
                std::cmp::Ordering::Equal => "exact".to_owned(),
                std::cmp::Ordering::Greater => format!("{} B of slack", reserved - used),
                std::cmp::Ordering::Less => unreachable!(),
            }
        );
    }
    println!(
        "# {} of the {} layouts take exactly their plan; {slack} B of slack over all of them; {short} short",
        exact,
        cases.len()
    );
    println!(
        "# a plan the forward fills is a number a runner may size storage from; slack is headroom nobody uses."
    );
    if check && short > 0 {
        return Err(format!(
            "{short} layout(s) ask more than their plan reserves"
        ));
    }
    Ok(())
}

/// `HxW:frames:tokens`, the order a request states them in.
fn case_of(text: &str) -> Result<Case, String> {
    let mut parts = text.split(':');
    let raster = parts
        .next()
        .ok_or_else(|| format!("`{text}` states no raster"))?;
    let (height, width) = raster_of(raster)?;
    let frames = parts
        .next()
        .ok_or_else(|| format!("`{text}` states no frame count"))?
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("`{text}`: the frame count is not a count"))?;
    let tokens = parts
        .next()
        .ok_or_else(|| format!("`{text}` states no prompt width"))?
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("`{text}`: the prompt width is not a count"))?;
    if parts.next().is_some() {
        return Err(format!(
            "`{text}` states more than a raster, frames and tokens"
        ));
    }
    Ok(Case {
        height,
        width,
        frames,
        tokens,
    })
}

/// `HxW`, the raster a request states.
fn raster_of(text: &str) -> Result<(u32, u32), String> {
    let (height, width) = text
        .split_once('x')
        .ok_or_else(|| format!("`{text}` is not a raster: state it as HxW"))?;
    Ok((
        height
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("`{text}`: the height is not a count"))?,
        width
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("`{text}`: the width is not a count"))?,
    ))
}

/// `tile,sparsity`, the deployment's own grammar.
fn sparse_of(text: &str) -> Result<(u32, f64), String> {
    let mut parts = text.split(',');
    let tile = parts
        .next()
        .ok_or_else(|| format!("`{text}` states no tile"))?
        .trim()
        .parse::<u32>()
        .map_err(|_| format!("`{text}`: the tile is not a count"))?;
    let sparsity = parts
        .next()
        .ok_or_else(|| format!("`{text}` states no sparsity"))?
        .trim()
        .parse::<f64>()
        .map_err(|_| format!("`{text}`: the sparsity is not a number"))?;
    if parts.next().is_some() {
        return Err(format!(
            "`{text}` states more than a tile and a sparsity: the reference keep rate is the deployment's"
        ));
    }
    Ok((tile, sparsity))
}

/// The checkpoint's tensor extents, from the reference's own table.
fn read_extents(path: &Path) -> Result<BTreeMap<String, (DType, Shape)>, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let values: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let object = values
        .as_object()
        .ok_or_else(|| format!("{} is not an object", path.display()))?;
    // The table is one object per component, and a component's own tensors are named without the
    // component prefix: this network is the base one, whose entries stand under `transformer`. A
    // deployment's own directory (`Ref2VA/transformer`) holds the same names with its own extents,
    // and taking it here would price the wrong network.
    let object = object
        .get("transformer")
        .and_then(|entry| entry.as_object())
        .ok_or_else(|| format!("{} states no `transformer` component", path.display()))?;
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
