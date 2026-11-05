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

use minimax_h3::audio_vae::AudioDecoder;
use minimax_h3::encoding::{AudioEncoder, VideoEncoder};
use minimax_h3::config::{
    AttentionKind, AudioVaeConfig, TextEncoderConfig, TransformerConfig, VideoVaeConfig,
};
use minimax_h3::partition::Partition;
use minimax_h3::output::VideoPostprocessor;
use minimax_h3::text_encoder::TextEncoder;
use minimax_h3::denoiser::{Arrangement, Denoiser, Layouts};
use minimax_h3::modulation::Groups;
use minimax_h3::video_vae::VideoDecoder;
use minimax_h3::vision::VisionTower;

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
    let mut dtype: Option<ops::DType> = None;
    // The caller's `component-tensors.json`, which states the four components' stored element types.
    let mut types: Option<PathBuf> = None;
    // The VAE container's element type is the checkpoint's, not the deployment's: the export stores its
    // VAE weights in one type (`dtype = "auto"` converts nothing), and both VAE halves compute in it.
    // FP32 is what this export's VAE is stored in; another export is another value, stated here.
    let mut vae_dtype = ops::DType::F32;
    // The prompt lengths the deployment serves, and its tensor-parallel degree: both are the
    // deployment's, and the degree decides the shard widths the store must hold.
    let mut tokens: Vec<u32> = Vec::new();
    let mut tensor_parallel: Option<usize> = None;    let mut ulysses: Option<usize> = None;
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
            "--vae-dtype" => {
                vae_dtype = match value.as_str() {
                    "f16" => ops::DType::F16,
                    "bf16" => ops::DType::BF16,
                    "f32" => ops::DType::F32,
                    other => return Err(format!("{other} is not a float type").into()),
                };
            }
            // The four components' stored element types, from the caller's tensor table: a row is keyed
            // by an element type, and no shardless host can read them from the checkpoint, so a
            // deployment states them or the fixture cannot be attributed.
            "--types" => types = Some(PathBuf::from(value.as_str())),
            "--dtype" => {
                dtype = Some(match value.as_str() {
                    "f16" => ops::DType::F16,
                    "bf16" => ops::DType::BF16,
                    "f32" => ops::DType::F32,
                    other => return Err(format!("{other} is not a float type").into()),
                });
            }
            other => return Err(format!("unknown flag {other}").into()),
        }
    }
    if samples.is_empty() || rasters.is_empty() {
        return Err("both --samples and --rasters are the deployment's and have no default".into());
    }
    let dtype = dtype.ok_or("--dtype is the checkpoint's and has no default")?;
    // The four types are stated when the caller's table is at hand, and said to be unstated when it is
    // not: the table is an artifact of an export (`artifacts/` is not in the tree), so requiring it
    // would make the walk refuse on a host that simply has not exported one. A row set without its
    // element types is still unattributable - that is a statement about the header, and the gate is
    // the fixture's generator rather than this diagnostic.
    let element_types = match types.as_deref() {
        Some(table) => {
            let stated = minimax_h3::load::ElementTypes::read(table)?;
            // Checked where it can be: a host with the checkpoint's shards compares the statement
            // with them and refuses a disagreement by name, and a host with none states it unchecked.
            if let Ok(checkpoint) = nn::Checkpoint::open(&root) {
                stated.verify(&checkpoint)?;
            }
            format!("[{}] from {}", stated.field(), table.display())
        }
        None => format!("[{}]", minimax_h3::load::UNSTATED),
    };
    if tokens.is_empty() {
        return Err("--tokens is the deployment's and has no default".into());
    }
    tokens.sort_unstable();
    tokens.dedup();
    if packed.is_empty() {
        return Err("--rows is the deployment's and has no default".into());
    }
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
        return Err("this checkpoint states its own contract: --sparse states a deployment it already \
                    states"
            .into());
    }
    let conditioned = conditioned.ok_or("--conditioned is the deployment's and has no default")?;
    let degree = tensor_parallel.ok_or("--tp is the deployment's and has no default")?;
    let partition = Partition {
        tensor_parallel: degree,
        tensor_rank: 0,
        ulysses_degree: ulysses.unwrap_or(1),
        ..Partition::single()
    };
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
    rasters.sort_unstable();
    rasters.dedup();

    let mut rows: Vec<(String, String)> = Vec::new();

    let audio = root.join("audio_vae/config.json");
    if audio.exists() {
        let config = AudioVaeConfig::parse(&read(&audio)?)?;
        rows.extend(AudioDecoder::programs(&config, &samples)?);
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
        // The container's type, not the deployment's: the decoder's tensor space is the checkpoint's
        // and the recorder measures it at that type (M5319).
        rows.extend(VideoDecoder::programs(&config, vae_dtype)?);
        rows.extend(VideoPostprocessor::programs(dtype, &rasters));
        // The encoder runs FP32 by the design's own row, and its frame operators key on classes the
        // raster derives — the one component of this model where the raster is load-bearing.
        rows.extend(VideoEncoder::programs(&config, &rasters)?);
    } else {
        eprintln!("not walked: {} has no video VAE config", root.display());
    }

    let encoder = root.join("text_encoder/config.json");
    if encoder.exists() {
        let metadata = read(&encoder)?;
        let processor = read(&root.join("text_encoder/preprocessor_config.json"))?;
        let config = TextEncoderConfig::parse(&metadata, &processor)?;
        // One row per prompt length: the attention's key bound is the deployment's, and the gated
        // activation's `wide` form follows from it.
        for tokens in &tokens {
            rows.extend(TextEncoder::programs(&config, &partition, dtype, *tokens)?);
        }
        // The tower attends grids of the deployment's largest image; its programs do not depend on
        // them beyond the bound the module is accepted for.
        let sequence = tokens.iter().copied().max().unwrap_or(0);
        rows.extend(VisionTower::programs(&config.vision, dtype, sequence)?);
    } else {
        eprintln!("not walked: text_encoder");
    }

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
        let config =
            TransformerConfig::parse(values, &format!("{component}/config.json"))?;
        let max = packed.iter().copied().max().unwrap_or(0);
        // The group count is the task set's: a conditioned deployment's modulation cache is another
        // program at the entries' width. The denoiser states its two stacks and the orchestration
        // around them, so the walk asks it rather than a stack.
        let (attention, tiles) = match (sparse, stated_attention) {
            (_, Some(sparse_contract)) => (
                AttentionKind::sparse(
                    sparse_contract.tile,
                    sparse_contract.sparsity,
                    sparse_contract.reference_keep,
                )?,
                // The bound on the key tiles one compression reduces over: the deployments' packed
                // rows over the tile's rows, rounded up to the next power of two because the bound is
                // a compiled class rather than a request count.
                (packed.iter().copied().max().unwrap_or(0).div_ceil(sparse_contract.tile))
                    .next_power_of_two(),
            ),
            (Some((tile, sparsity, stated_tiles, reference_keep)), None) => {
                let attention = AttentionKind::sparse(tile, sparsity, reference_keep)?;
                // The tile bound is the deployment's own rule and not a caller's number: it caps every
                // canvas and frame count the deployment offers, and the launch keys its next power of
                // two. A stated bound that is another class is refused by name rather than walked,
                // because a fixture row naming the wrong class is a module no launch asks for.
                let mut network = minimax_h3::Config::read(&root, None)?.denoisers;
                let mut base = network
                    .remove(&minimax_h3::Dit::Base)
                    .ok_or("this checkpoint holds no base network")?;
                base.attention = attention;
                let tokens_max = tokens.iter().copied().max().unwrap_or(0);
                let arrangement = Arrangement::new(&base, tokens_max, 64, 1)?;
                let computed = Layouts::new(base, arrangement)?.tiles_bound()?;
                if computed.next_power_of_two() != stated_tiles.next_power_of_two() {
                    return Err(format!(
                        "--sparse states {stated_tiles} tiles and the deployment's own bound is \
                         {computed}: the launch keys the bound's next power of two \
                         ({}, not {}), and a row naming another class is a module no launch asks for",
                        computed.next_power_of_two(),
                        stated_tiles.next_power_of_two()
                    )
                    .into());
                }
                (attention, computed)
            }
            (None, None) => (AttentionKind::Dense, 0),
        };
        rows.extend(Denoiser::programs(
            &config,
            attention,
            dtype,
            max,
            Groups::of(conditioned),
            &partition,
            tiles,
        )?);
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
    println!("# the VAE container runs {vae_dtype}");
    println!(
        "# MiniMax-H3 programs for these clip lengths: {samples:?}, raster sizes: {rasters:?}, prompt lengths: {tokens:?}, packed rows: {packed:?}, trajectory elements: {elements:?}, element types: {element_types}, audio tracks: {track:?}, sparse attention: {sparse:?}, tensor-parallel degree {degree}, ulysses degree {}", partition.ulysses_degree
    );
    println!(
        "# generated by `cargo run -p minimax-h3 --example programs -- <pipeline root> --dtype {dtype} --samples {} --rasters {} --tokens {} --rows {} --elements {} --track {} --sparse {} --tp {degree}`",
        samples.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
        rasters.iter().map(|(height, width)| format!("{height}x{width}")).collect::<Vec<_>>().join(","),
        tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
        packed.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
        elements.iter().map(u64::to_string).collect::<Vec<_>>().join(","),
        track.iter().map(u32::to_string).collect::<Vec<_>>().join(","),
        sparse.map_or("-".to_owned(), |(tile, sparsity, tiles, reference_keep)| format!(
            "{tile},{sparsity},{tiles}{}",
            reference_keep.map_or(String::new(), |keep| format!(",{keep}"))
        )),
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
