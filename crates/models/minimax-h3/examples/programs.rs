//! The programs this deployment compiles, one `operator<TAB>key` line each.
//!
//! ```text
//! cargo run -p minimax-h3 --example programs -- /opt/uv/models/MiniMax-H3 [samples ...]
//! ```
//!
//! The rows are `notes/programs/minimax-h3-keys.txt`: what a kernel package must provide for
//! `crates/models/minimax-h3`, stated by the model from its own configuration and the same
//! construction its layers launch with, so no package and no fixture has to guess a variant.
//!
//! `samples` are the clip lengths the deployment serves. A row program's descriptor carries its
//! width, so the set depends on them; the default is the one the caller's requests use.
//!
//! ## What is walked, and what is not
//!
//! Each component states its own launches through [`minimax_h3::programs::Programs`]. A component
//! that does not yet state its programs is named on stderr rather than omitted silently: a fixture
//! missing a component's rows reads as complete and refuses at the first launch of that component.

use std::path::PathBuf;

use minimax_h3::audio_vae::BigVgan;
use minimax_h3::config::{AttentionKind, AudioVaeConfig, TransformerConfig, VideoVaeConfig};
use minimax_h3::denoiser::Denoiser;
use minimax_h3::encoding::{AudioEncoder, VideoEncoder};
use minimax_h3::modulation::Groups;
use minimax_h3::output::VideoPostprocessor;
use minimax_h3::partition::Partition;
use minimax_h3::programs::{Deployment, Types};
use minimax_h3::tiled_encode::ClipEncoder;
use minimax_h3::video_vae::VideoDecoder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let root: PathBuf = arguments
        .next()
        .ok_or("usage: programs <pipeline root> --samples a,b --rasters HxW,...")?
        .into();

    // The buckets are the deployment's, never the model's: a row program's key carries its width, so
    // a fixture is only valid for the shapes it was generated at. Nothing here has a default — a walk
    // that invented one would describe a deployment nobody runs.
    let mut samples: Vec<u32> = Vec::new();
    let mut rasters: Vec<(u32, u32)> = Vec::new();
    // The video decoder's precision is the checkpoint's, not the config's: the design converts
    // nothing at load, so the deployment states the element type its weights are in.
    // The four components' stored element types, as the caller's tensor table states them. A
    // deployment's element type is its own per component (the design's precision rule), so a single
    // `--dtype` handed to every component is the defect this flag replaces: the video VAE asked for
    // the stack's type states rows no launch asks for.
    // The task the deployment serves. It is not cosmetic: the serving partition derives its capacity
    // from it (`Partition::serving`), and the packed row bound - hence the gated activation's `wide`
    // form and the sparse tile class - follows from that capacity. A walk that built the partition
    // from `Partition::single()` stated a bound serving never asks for.
    let mut task: Option<nn::VideoTask> = None;
    let mut types: Option<PathBuf> = None;
    // The prompt lengths the deployment serves, and its tensor-parallel degree: both are the
    // deployment's, and the degree decides the shard widths the store must hold.
    let mut tokens: Vec<u32> = Vec::new();
    let mut tensor_parallel: Option<usize> = None;
    let mut ulysses: Option<usize> = None;
    // The packed row count the DiT partitions are compiled for: a deployment's layout statement, not
    // a request's count.
    let mut packed: Vec<u32> = Vec::new();
    // The element count of the largest trajectory the solver steps: its residual is one operator over
    // a whole sample, so its width is the deployment's canvas and not a constant.
    let mut elements: Vec<u64> = Vec::new();
    // The longest audio track the deployment encodes: the audio encoder's residual units are one
    // operator each over their own output, so their key carries its channel count times these samples.
    let mut track: Vec<u32> = Vec::new();
    // A sparse deployment's attention: the segment tile, the sparsity rate, and the tile bound the
    // selection's compression reduces over — a deployment fact, stated by whoever sizes the store.
    let mut sparse: Option<(u32, f64, u32, Option<f64>)> = None;
    // Whether the deployment serves conditioned tasks (`fl2va`/`ref2va`) or only `t2va`: it decides the
    // modulation cache's group count, which is a program's width.
    let mut conditioned: Option<bool> = None;
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--samples" => {
                for sample in value.split(',') {
                    samples.push(sample.trim().parse()?);
                }
            }
            "--rasters" => {
                for raster in value.split(',') {
                    let (height, width) = raster
                        .trim()
                        .split_once('x')
                        .ok_or_else(|| format!("{raster} is not HxW"))?;
                    rasters.push((height.parse()?, width.parse()?));
                }
            }
            "--tokens" => {
                for token in value.split(',') {
                    tokens.push(token.trim().parse()?);
                }
            }
            "--rows" => {
                for row in value.split(',') {
                    packed.push(row.trim().parse()?);
                }
            }
            "--conditioned" => {
                conditioned = Some(match value.as_str() {
                    "true" => true,
                    "false" => false,
                    other => return Err(format!("{other} is not a boolean").into()),
                });
            }
            "--elements" => {
                for count in value.split(',') {
                    elements.push(count.trim().parse()?);
                }
            }
            "--sparse" => {
                // `tile,sparsity,tiles[,reference_keep]`: the segment tile rows, the fraction of key
                // tiles dropped, the bound on the key tiles one compression reduces over, and — for a
                // student with reference videos — the fraction of its own key tiles a reference query
                // keeps. The three are the checkpoint's own contract (`fastvideo_inference.json`), and
                // `Config` reads them there; this flag states them for a base release whose contract
                // lives outside the tree the walk is given.
                let parts: Vec<&str> = value.split(',').collect();
                if parts.len() != 3 && parts.len() != 4 {
                    return Err("--sparse is `tile,sparsity,tiles[,reference_keep]`".into());
                }
                let reference_keep = match parts.get(3) {
                    Some(keep) => Some(keep.trim().parse()?),
                    None => None,
                };
                sparse = Some((
                    parts[0].trim().parse()?,
                    parts[1].trim().parse()?,
                    parts[2].trim().parse()?,
                    reference_keep,
                ));
            }
            "--track" => {
                for samples in value.split(',') {
                    track.push(samples.trim().parse()?);
                }
            }
            "--tp" => tensor_parallel = Some(value.parse()?),
            // The denoiser's Ulysses degree, which `--tp` does not touch: the text encoder's tensor
            // shard and the denoiser's sequence shard are independent groups.
            "--ulysses" => ulysses = Some(value.parse()?),
            "--task" => {
                task = Some(match value.as_str() {
                    "t2va" => nn::VideoTask::TextToVideoAudio,
                    "fl2va" => nn::VideoTask::FirstLastToVideoAudio,
                    "ref2va" => nn::VideoTask::ReferenceToVideoAudio,
                    other => {
                        return Err(
                            format!("{other} is not an H3 task (t2va, fl2va, ref2va)").into()
                        );
                    }
                })
            }
            "--types" => types = Some(PathBuf::from(value.as_str())),
            // Refused, not ignored: a run that states an element type believes it chose one, and the
            // type is each component's own - the checkpoint's, or the table `--types` names.
            "--dtype" => {
                return Err(format!(
                    "--dtype {value} states one element type for four components; each component's \
                     type is its own, so state --types <table> or let the checkpoint's shard headers \
                     answer"
                )
                .into());
            }
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if samples.is_empty() {
        return Err("--samples is the deployment's and has no default".into());
    }
    // **The canvases are the deployment's, not the caller's.** A row keyed for a raster the engine
    // can never be asked for is a cell no launch reads, and a stated raster outside the served set
    // was exactly that: `480x832` is the encoder's own test raster, and its `scaled_add columns
    // 399360` is the seam blend over a canvas this deployment does not serve.
    let offered: Vec<(u32, u32)> = minimax_h3::config::offered_canvases()
        .iter()
        .map(|canvas| (canvas.height, canvas.width))
        .collect();
    if rasters.is_empty() {
        rasters = offered.clone();
    } else {
        for (height, width) in &rasters {
            if !offered.contains(&(*height, *width)) {
                return Err(format!(
                    "--rasters {height}x{width} is not one of the deployment's offered canvases \
                     {offered:?}: a row keyed for a canvas no request can name is a cell no launch \
                     reads"
                )
                .into());
            }
        }
    }
    // **`--dtype` is gone.** A deployment's element type is its own per component (the design's
    // precision rule: BF16 text encoder and DiT, FP32 video VAE), so one flag could only ever state
    // three components wrong. A stated `--dtype` is refused rather than ignored: a run that thinks it
    // chose a type should learn that it did not.
    if tokens.is_empty() {
        return Err("--tokens is the deployment's and has no default".into());
    }
    tokens.sort_unstable();
    tokens.dedup();
    // `--rows` has no default and no authority: the deployment derives this number
    // (`Layouts::rows_bound`), so a stated value is checked against it below rather than believed.
    packed.sort_unstable();
    packed.dedup();
    if track.is_empty() {
        return Err("--track is the deployment's and has no default".into());
    }
    track.sort_unstable();
    track.dedup();
    if elements.is_empty() {
        return Err("--elements is the deployment's and has no default".into());
    }
    elements.sort_unstable();
    elements.dedup();
    // The checkpoint states how it must run: its `fastvideo_inference.json` names the attention
    // backend and, when it is sparse, the tile and the keep rate. That statement is the deployment's,
    // not this tool's, so it is read here as the model reads it — a caller's flag may not override it.
    let stated = minimax_h3::Contract::read(&root)?.and_then(|contract| contract.attention);
    let stated_attention = stated;
    if stated.is_some() && sparse.is_some() {
        return Err(
            "this checkpoint states its own contract: --sparse states a deployment it already \
                    states"
                .into(),
        );
    }
    let conditioned = conditioned.ok_or("--conditioned is the deployment's and has no default")?;
    let degree = tensor_parallel.ok_or("--tp is the deployment's and has no default")?;
    let task = task.ok_or("--task is the deployment's and has no default")?;
    // A task that conditions cannot be walked unconditioned, and the reverse: the capacity, the
    // modulation cache's groups and the blocks a deployment holds are three statements of one fact.
    let conditioned_task = !matches!(task, nn::VideoTask::TextToVideoAudio);
    if conditioned_task != conditioned {
        return Err(format!(
            "--task {task:?} and --conditioned {conditioned} disagree: a task either holds the \
             conditioner's cache or it does not"
        )
        .into());
    }
    let config = minimax_h3::Config::read(&root, None)?;
    let partition = Partition::serving(&config, task, ulysses.unwrap_or(1), degree, 0, 0)?;
    // The sparse chain composes each layer's compression across the exchange's members, and that
    // exchange is not written: the model refuses it by name at load, and the walk refuses the same
    // deployment here rather than printing rows that read as served.
    if let Some(tile) = sparse
        .map(|(tile, _, _, _)| tile)
        .or(stated_attention.map(|sparse| sparse.tile))
        && partition.ulysses_degree > 1
    {
        return Err(format!(
            "the sparse attention (tile {tile}) composes its compression across {} ulysses ranks, \
             whose exchange is not written",
            partition.ulysses_degree
        )
        .into());
    }
    samples.sort_unstable();
    samples.dedup();
    // A stated clip is checked against the rule the model produces clips by, and a stated track
    // against the clip it belongs to: `--samples` and `--track` are two statements of one clip, and
    // nothing compared them - so a walk could state an audio decoder at one clip and an audio encoder
    // at another, and every row would be a real program at a length no request names.
    let generated = minimax_h3::programs::Deployment::generated();
    for frames in &samples {
        if !generated.contains(frames) {
            return Err(format!(
                "--samples {frames} is not a clip this model produces: the packing accepts 17n + 5 \
                 over 22 to 1008, and a clip outside it is a request the sampler cannot be handed"
            )
            .into());
        }
    }
    for stated_track in &track {
        let named = samples
            .iter()
            .find(|frames| minimax_h3::packing::audio_samples(**frames) == *stated_track);
        if named.is_none() {
            let implied: Vec<u32> = samples
                .iter()
                .map(|frames| minimax_h3::packing::audio_samples(*frames))
                .collect();
            return Err(format!(
                "--track {stated_track} is not the track of any stated clip: --samples {samples:?} \
                 imply tracks {implied:?}, and the audio encoder would be walked at a clip the \
                 audio decoder is not"
            )
            .into());
        }
    }
    rasters.sort_unstable();
    rasters.dedup();

    let mut rows: Vec<(String, String)> = Vec::new();

    // Each component's stored element type: the caller's table where it is at hand, the checkpoint's
    // own shard headers where they are, and a refusal naming both where neither is. The video and
    // audio VAEs are the components whose type is *not* the stack's, and handing them `--dtype` was
    // a measured defect: seven decoder rows and two postprocessor rows appear at the wrong type.
    let stated = match types.as_deref() {
        Some(table) => Some(minimax_h3::load::ElementTypes::read(table)?),
        None => None,
    };
    let checkpoint = nn::Checkpoint::open(&root).ok();
    let text_type =
        minimax_h3::load::deployment_dtype(stated.as_ref(), checkpoint.as_ref(), "text_encoder")?;
    let stack_type =
        minimax_h3::load::deployment_dtype(stated.as_ref(), checkpoint.as_ref(), "transformer")?;
    // **The transformer's second element type, where the checkpoint stores one**: the four boundary
    // projections and the timestep embedder's pair. `None` means the component is single-typed and
    // the stack's type is the only one, which is what every other component's answer is.
    let boundary_type =
        minimax_h3::load::boundary_dtype(stated.as_ref(), checkpoint.as_ref(), "transformer")?
            .unwrap_or(stack_type);
    let vae_type = minimax_h3::load::deployment_dtype(stated.as_ref(), checkpoint.as_ref(), "vae")?;
    let audio_type =
        minimax_h3::load::deployment_dtype(stated.as_ref(), checkpoint.as_ref(), "audio_vae")?;

    // **The deployment states its own facts.** `--rows` was an input and the deployment derives that
    // number (`Layouts::rows_bound`), so a walk cut at a stated bound could describe a deployment no
    // request reaches - and every count it printed was about that request. The value is asked here
    // and a stated `--rows` is checked against it.
    let deployment = Deployment::new(
        config.clone(),
        Types {
            text_encoder: text_type,
            transformer: stack_type,
            transformer_boundary: boundary_type,
            video: vae_type,
            audio: audio_type,
        },
        samples.clone(),
        // The tasks the checkpoint serves, which is what the deployment holds. **`--task` is not this**:
        // it selects the serving partition the bound is derived at, while a task states the modulation
        // cache's groups and a cache width is a program key. A walk narrowed to one task leaves another
        // task's requests without a cell, and nothing in a fixture would show it.
        Deployment::checkpoint_tasks(&root)?,
    )?;
    let attention = match (sparse, stated_attention) {
        (_, Some(contract)) => {
            AttentionKind::sparse(contract.tile, contract.sparsity, contract.reference_keep)?
        }
        (Some((tile, sparsity, _, reference_keep)), None) => {
            AttentionKind::sparse(tile, sparsity, reference_keep)?
        }
        (None, None) => AttentionKind::Dense,
    };
    let bound = deployment.bound(&partition, Some(attention))?;
    let rows_bound = match packed.iter().copied().max() {
        Some(stated) if stated != bound => {
            return Err(format!(
                "--rows states {stated} packed rows and the deployment's own bound is {bound}: a \
                 walk that stamped another number describes a deployment it cannot be asked to be"
            )
            .into());
        }
        _ => bound,
    };
    eprintln!(
        "the deployment's packed row bound is {rows_bound}: {}",
        if packed.is_empty() {
            "stated by the deployment, no --rows given"
        } else {
            "--rows agrees with it"
        }
    );

    let audio = root.join("audio_vae/config.json");
    if audio.exists() {
        // The two audio VAEs state F32 rows by the design's own table, and neither component takes a
        // type - so a checkpoint that stored the audio VAE in another type would be walked at F32 and
        // read as served. It is refused here rather than stated, because making the components take a
        // type is their owner's change and a silent F32 is the defect this flag exists to end.
        if audio_type != ops::DType::F32 {
            return Err(format!(
                "this checkpoint stores the audio VAE in {audio_type:?}: the audio components state \
                 F32 rows by design and take no type, so a walk here would state rows no launch asks"
            )
            .into());
        }
        let config = AudioVaeConfig::parse(&read(&audio)?)?;
        // `--samples` is the deployment's clip lengths in **frames**; a decode starts from the
        // track's **latent frames** and `BigVgan::programs` takes that quantity. Passing the
        // frames stated every stage's width a factor too wide - at `--samples 5` the seven residual
        // rows landed at `5 x {5,25,50,100,200,400,800}` instead of `8 x {...}`, where 8 is the
        // clip's latent frames - and the rows matched no store the list builds.
        let latents: Vec<u32> = samples
            .iter()
            .map(|frames| minimax_h3::packing::audio_latent_frames(*frames))
            .collect();
        rows.extend(BigVgan::programs(&config, &latents)?);
        // The audio encoder runs FP32 by the design's own rows, so it takes no dtype.
        rows.extend(AudioEncoder::programs(&config, &track)?);
    } else {
        eprintln!("not walked: {} has no audio VAE config", root.display());
    }

    // The post-processor is weightless: its programs follow from the raster sizes and the decoder's
    // element type, which the video VAE's config states.
    let vae = root.join("vae/config.json");
    if vae.exists() {
        let config = VideoVaeConfig::parse(&read(&vae)?)?;
        rows.extend(VideoDecoder::programs(&config, vae_type)?);
        rows.extend(VideoPostprocessor::programs(vae_type, &rasters));
        // The encoder runs FP32 by the design's own row, and its frame operators key on classes the
        // raster derives — the one component of this model where the raster is load-bearing.
        rows.extend(VideoEncoder::programs(&config, &rasters)?);
        // **The encoder's tiled driver.** A canvas larger than one tile is encoded by tiles that
        // cross-fade where they meet, and `ClipEncoder` states the blends the grid produces and the
        // frame operators at each tile's extent - rows no whole-canvas call describes. Without it the
        // walk described one of the encoder's two drivers, which is how thirteen rows a launch takes
        // came to be absent from every walk.
        for frames in &samples {
            rows.extend(ClipEncoder::programs(&config, &rasters, *frames)?);
        }
    } else {
        eprintln!("not walked: {} has no video VAE config", root.display());
    }

    // The text encoder and its tower, from the deployment rather than from the component directory:
    // a walk that parsed `text_encoder/config.json` itself was the last place this tool answered a
    // deployment question with a second parser.
    rows.extend(deployment.text_rows(&partition, &tokens)?);

    // The two DiT partitions, each stated from its own `config.json`. The attention kind is a
    // deployment's contract (a sparse contract is a student's), so this walks the base form and the
    // walk itself refuses a sparse partition by name.
    for (component, directory) in [
        ("transformer", "transformer"),
        ("transformer_ref", "transformer_ref"),
    ] {
        let path = root.join(directory).join("config.json");
        if !path.exists() {
            eprintln!("not walked: {component}");
            continue;
        }
        let values = read(&path)?;
        let values = values
            .as_object()
            .ok_or_else(|| format!("{component}/config.json is not an object"))?;
        let config = TransformerConfig::parse(values, &format!("{component}/config.json"))?;
        let max = rows_bound;
        // The group count is the task set's: a conditioned deployment's modulation cache is another
        // program at the entries' width. The denoiser states its two stacks and the orchestration
        // around them, so the walk asks it rather than a stack.
        // The bound on the key tiles one compression reduces over: the deployment's packed rows over
        // the tile's rows, rounded up to the next power of two because the bound is a compiled class
        // rather than a request count. A caller's stated tile count is not an input either - the
        // deployment's own bound decides the class.
        let tiles = match (attention, sparse) {
            (AttentionKind::Dense, _) => 0,
            (sparse_kind, Some((tile, _, 0, _))) => {
                let _ = (sparse_kind, tile);
                rows_bound.div_ceil(tile).next_power_of_two()
            }
            (_, Some((tile, _, stated_tiles, _))) => {
                let derived = rows_bound.div_ceil(tile).next_power_of_two();
                if stated_tiles != derived {
                    return Err(format!(
                        "--sparse states {stated_tiles} tiles and the deployment's bound \
                         {rows_bound} over a {tile}-row tile is {derived}"
                    )
                    .into());
                }
                derived
            }
            (_, None) => {
                let Some(contract) = stated_attention else {
                    unreachable!("a dense deployment states its attention contract")
                };
                rows_bound.div_ceil(contract.tile).next_power_of_two()
            }
        };
        // **The groups are the served tasks', not this walk's.** A task fixes how many timestep
        // groups the modulation cache holds and the cache's width is a program key, so stating one
        // task's cache leaves another task's requests with no cell - `logits_gather columns_2` is that
        // row, and nothing but iterating the served set states it.
        for groups in [Groups::of(false), Groups::of(true)] {
            rows.extend(Denoiser::programs(
                &config,
                attention,
                stack_type,
                boundary_type,
                max,
                groups,
                &partition,
                &elements,
                tiles,
            )?);
        }
    }

    // Deduplicate first: a count of the launches before deduplication is a count of something else
    // (163 against 134 here), and the header's number must be the number of rows below it.
    rows.sort();
    rows.dedup();

    // The header states its own size, printed from the vector that produced the rows rather than
    // counted afterwards: a reader comparing two fixtures sees the count, the command and the reasons
    // below without running anything, and no two readings of the file can disagree about it.
    let operators = {
        let mut names: Vec<&str> = rows.iter().map(|(name, _)| name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        names.len()
    };
    println!("# {} data rows, {operators} operators", rows.len());
    println!(
        "# MiniMax-H3 programs for these clip lengths: {samples:?}, raster sizes: {rasters:?}, prompt lengths: {tokens:?}, packed rows: [{rows_bound}] (the deployment's own bound), trajectory elements: {elements:?}, audio tracks: {track:?}, sparse attention: {sparse:?}, tensor-parallel degree {degree}, ulysses degree {}",
        partition.ulysses_degree
    );
    println!(
        "# generated by `cargo run -p minimax-h3 --example programs -- <pipeline root> --types <table> --samples {} --rasters {} --tokens {} --elements {} --track {} --sparse {} --tp {degree}`",
        samples
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
        rasters
            .iter()
            .map(|(height, width)| format!("{height}x{width}"))
            .collect::<Vec<_>>()
            .join(","),
        tokens
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
        elements
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(","),
        track
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(","),
        sparse.map_or(
            "-".to_owned(),
            |(tile, sparsity, tiles, reference_keep)| format!(
                "{tile},{sparsity},{tiles}{}",
                reference_keep.map_or(String::new(), |keep| format!(",{keep}"))
            )
        ),
    );
    for (operator, key) in &rows {
        println!("{operator}\t{key}");
    }
    eprintln!("{} rows", rows.len());
    Ok(())
}

fn read(path: &std::path::Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
}
