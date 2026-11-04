//! Run the encoder stage: the text encoder's own output for one request.
//!
//!     cargo run -p minimax-h3 --example encoder -- --stage weights --root <export>
//!     cargo run -p minimax-h3 --example encoder -- --stage encode --root <export> \
//!         --device 0 --store <store> --out artifacts/h3-stage4
//!
//! A served request's first arrow is prompt -> text encoder -> the denoiser's conditioner, and this
//! is the only stage that exercises it. `--stage weights` is the host half: the language stack's and
//! the tower's names against the export's index, then the real load into a store that checks every
//! element type and shape as it is handed over. `--stage encode` is the device half: one prompt on
//! one rank, decoded to the rows the reference probe (`tools/dump_encoder.py --probe prompt_hidden`)
//! reads back for the same case.
//!
//! The case carries the prompt twice on purpose: `prompt` is the text and `prompt_ids` the ids a
//! front end tokenized it into. The crates have no tokenizer, so ids are data; the string is there
//! so the reference probe can tokenize it with the checkpoint's own tokenizer and refuse a
//! disagreement at the first index where the two lists part, instead of hiding it inside every row.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use nn::capability::{Encoder as _, EncoderInputs, EncoderOutputs, EncoderPrompt, TokenBatch};
use nn::weights::{HostStore, WeightLoader};
use nn::{
    Arena, Backing, Checkpoint, DType, DevicePtr, Forward, RetainedWeights, Shape, TensorView,
    WeightKey,
};
use platform::{DeviceId, PoolKind};

use minimax_h3::config::Config;
use minimax_h3::load;
use minimax_h3::partition::Partition;
use minimax_h3::text_encoder::TextEncoder;
use minimax_h3::vision::VisionTower;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args = Args::parse()?;
    match args.stage {
        Stage::Weights => stage_weights(&args),
        Stage::Encode => stage_encode(&args),
    }
}

/// One failure line, as the series' examples report it.
#[derive(Debug)]
struct Failure(String);

impl Failure {
    fn message(text: impl Into<String>) -> Self {
        Self(text.into())
    }
}

/// Every function here reports one `Failure`, the series' shape.
type Result<T> = std::result::Result<T, Failure>;

impl std::fmt::Display for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl From<minimax_h3::Error> for Failure {
    fn from(error: minimax_h3::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<nn::Error> for Failure {
    fn from(error: nn::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<platform::Error> for Failure {
    fn from(error: platform::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Stage {
    Weights,
    Encode,
}

struct Args {
    stage: Stage,
    root: PathBuf,
    device: i32,
    store: Option<PathBuf>,
    out: PathBuf,
    /// The prompt as text, and the ids a front end tokenized it into.
    prompt: String,
    prompt_ids: Vec<u32>,
    /// The element type to compute in: `auto` is the checkpoint's own, anything else is the
    /// deployment's statement - one conversion at load, printed, and carried in the case so the
    /// reference half loads at the same width.
    dtype: Option<String>,
}

impl Args {
    fn parse() -> Result<Self> {
        let mut stage = Stage::Weights;
        let mut root = None;
        let mut device = 0;
        let mut store = None;
        let mut out = PathBuf::from("artifacts/h3-stage4");
        let mut prompt = None;
        let mut prompt_ids = None;
        let mut dtype = None;
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--stage" => {
                    let value = arguments
                        .next()
                        .ok_or_else(|| Failure("--stage needs `weights` or `encode`".into()))?;
                    stage = match value.as_str() {
                        "weights" => Stage::Weights,
                        "encode" => Stage::Encode,
                        other => {
                            return Err(Failure(format!(
                                "--stage takes `weights` or `encode`, not `{other}`"
                            )));
                        }
                    };
                }
                "--root" => {
                    root = Some(PathBuf::from(
                        arguments
                            .next()
                            .ok_or_else(|| Failure("--root needs a path".into()))?,
                    ));
                }
                "--device" => {
                    device = arguments
                        .next()
                        .ok_or_else(|| Failure("--device needs a number".into()))?
                        .parse()
                        .map_err(|error| Failure(format!("--device: {error}")))?;
                }
                "--store" => {
                    store = Some(PathBuf::from(
                        arguments
                            .next()
                            .ok_or_else(|| Failure("--store needs a path".into()))?,
                    ));
                }
                "--out" => {
                    out = PathBuf::from(
                        arguments
                            .next()
                            .ok_or_else(|| Failure("--out needs a path".into()))?,
                    );
                }
                "--dtype" => {
                    dtype = Some(arguments.next().ok_or_else(|| {
                        Failure("--dtype needs `auto` or an element type".into())
                    })?);
                }
                "--prompt" => {
                    prompt = Some(
                        arguments
                            .next()
                            .ok_or_else(|| Failure("--prompt needs text".into()))?,
                    );
                }
                "--prompt-ids" => {
                    let value = arguments.next().ok_or_else(|| {
                        Failure("--prompt-ids needs a comma-separated list".into())
                    })?;
                    if value.trim().is_empty() {
                        prompt_ids = Some(Vec::new());
                        continue;
                    }
                    prompt_ids = Some(
                        value
                            .split(',')
                            .map(|field| {
                                field.trim().parse::<u32>().map_err(|error| {
                                    Failure(format!("--prompt-ids: {field:?}: {error}"))
                                })
                            })
                            .collect::<Result<Vec<u32>>>()?,
                    );
                }
                other => {
                    return Err(Failure(format!(
                        "unknown argument {other}; the flags are --stage, --root, --device, \
                         --store, --out, --prompt and --prompt-ids"
                    )));
                }
            }
        }
        let root = root.ok_or_else(|| Failure("--root is required".into()))?;
        // The ids are the front end's answer, not this harness's: a prompt given without them would
        // make the case carry a string and a list that nothing relates, and the reference probe's
        // cross-check would report that silence as a tokenizer disagreement.
        let (prompt, prompt_ids) = match (prompt, prompt_ids) {
            (Some(prompt), Some(ids)) => (prompt, ids),
            (Some(_), None) => {
                return Err(Failure(
                    "--prompt given without --prompt-ids: the ids are the front end's tokenizer's, \
                     and this harness has none of its own"
                        .into(),
                ));
            }
            (None, Some(ids)) => (DEFAULT_PROMPT.to_owned(), ids),
            (None, None) => (DEFAULT_PROMPT.to_owned(), DEFAULT_IDS.to_vec()),
        };
        Ok(Self {
            stage,
            root,
            device,
            store,
            out,
            prompt,
            prompt_ids,
            dtype,
        })
    }
}

/// The ids the checkpoint's tokenizer gives the default prompt.
///
/// The export's text encoder component: the language stack's weights and the tower's.
const TEXT_ENCODER: &str = "text_encoder";

/// The prompt the harness encodes when asked for no particular one.
const DEFAULT_PROMPT: &str = "a cat walking on the grass";

/// The ids the checkpoint's own tokenizer gives [`DEFAULT_PROMPT`], measured on the host with
/// `Qwen2Tokenizer`: a default case still has to say what it encoded, and these are the checkpoint's
/// answer rather than a guess - the reference probe re-tokenizes the string and refuses a wrong list
/// at the first index where the two part.
const DEFAULT_IDS: [u32; 6] = [64, 8251, 11435, 389, 279, 16359];

/// The one element type the export's `text_encoder` holds.
///
/// `dtype = auto` is the checkpoint's own representation and converts nothing, so the width of every
/// page this stage writes follows the export rather than a harness constant. Two types in one
/// component is a deployment question, not a diagnosis one: it is refused, by name and with both
/// types, rather than answered by picking one.
fn compute_dtype(checkpoint: &Checkpoint, component: &str, stated: Option<DType>) -> Result<DType> {
    let own = load::stored_dtype(checkpoint, component)?;
    match stated {
        None => {
            println!("dtype: {own}, the checkpoint's own (`auto`)");
            Ok(own)
        }
        Some(dtype) => {
            println!(
                "dtype: {dtype}, the deployment's statement; the component holds {own}, so this \
                 is one conversion at load"
            );
            Ok(dtype)
        }
    }
}

/// The element type `--dtype` states, or `None` for `auto`.
///
/// `None` is the design's rule - the checkpoint's own representation and no conversion - and a
/// stated type is the deployment's decision, applied once by the loader. Both stages resolve the
/// flag through this one function, so a stage cannot honour it in its plan and forget it at the
/// weights: a weight that is never asked for at the stated type is the mixed-cell signature.
fn stated_dtype(args: &Args) -> Result<Option<DType>> {
    match args.dtype.as_deref() {
        None | Some("auto") => Ok(None),
        Some("bf16") => Ok(Some(DType::BF16)),
        Some("f16") => Ok(Some(DType::F16)),
        Some("f32") => Ok(Some(DType::F32)),
        Some(other) => Err(Failure::message(format!(
            "--dtype takes `auto`, `bf16`, `f16` or `f32`, not `{other}`"
        ))),
    }
}

/// The names the export's `text_encoder` index holds.
fn index_names(root: &Path) -> Result<Vec<String>> {
    let path = root
        .join("text_encoder")
        .join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|error| Failure(format!("{}: {error}", path.display())))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| Failure(format!("{}: {error}", path.display())))?;
    let map = value
        .get("weight_map")
        .and_then(|map| map.as_object())
        .ok_or_else(|| Failure(format!("{}: no weight_map", path.display())))?;
    Ok(map.keys().cloned().collect())
}

/// Stage 1: the names, then the load, with no device.
///
/// Two checks in one pass, each naming its own failure: the language stack's 551 names and the
/// tower's 351 against the export's index, then the same load the device path performs into a store
/// that validates every tensor's element type and shape as the loader hands it over. A checkpoint
/// that lacks a name is refused by name before a byte is read.
fn stage_weights(args: &Args) -> Result<()> {
    let config = Config::read(&args.root, None)?;
    let text = &config.text_encoder;
    let language = TextEncoder::keys(text);
    let tower = VisionTower::keys(&text.vision);
    let index = index_names(&args.root)?;

    for (half, asked) in [("language stack", &language), ("vision tower", &tower)] {
        let missing: Vec<&String> = asked.iter().filter(|name| !index.contains(name)).collect();
        println!(
            "names: {half} {} asked, {} in the index, {} missing",
            asked.len(),
            index.len(),
            missing.len()
        );
        for name in &missing {
            println!("  missing {name}");
        }
        if !missing.is_empty() {
            return Err(Failure(format!(
                "the encoders ask {} {half} tensors and {} are not in the export",
                asked.len(),
                missing.len()
            )));
        }
    }

    let checkpoint = Checkpoint::open(&args.root)?;
    let source = checkpoint.component("text_encoder")?;
    let stated = stated_dtype(&args)?;
    let dtype = compute_dtype(&checkpoint, TEXT_ENCODER, stated)?;
    let mut store = HostStore::new();
    let mut loader = WeightLoader::new(source, &mut store, stated, None);
    let encoder = TextEncoder::load(&mut loader, text, &Partition::single(), dtype)?;
    println!("loaded: text encoder at {}", encoder.dtype());
    println!(
        "load: published {} weight regions on the host, {} retained layers",
        store.len(),
        text.num_retained_layers
    );
    drop(encoder);
    Ok(())
}

// --- the device scaffolding the encode path shares with `dit_layer`'s stage: the same store, the
// same weight bookkeeping, and the same tensor container writer.

/// A store with no ceiling: this example is the only user of the device.
struct NoLimit;

fn allocate_tables(arena: &Arena, specs: &[nn::capability::StateTable]) -> Result<Vec<TensorView>> {
    specs
        .iter()
        .map(|spec| Ok(arena.tensor(spec.dtype, spec.shape)?))
        .collect()
}

fn host_tables(specs: &[nn::capability::StateTable]) -> Result<Vec<Vec<u8>>> {
    specs
        .iter()
        .map(|spec| {
            let width = spec.dtype.bytes().ok_or_else(|| {
                Failure::message(format!(
                    "the table `{}` is {} and has no byte width",
                    spec.name, spec.dtype
                ))
            })?;
            let bytes = usize::try_from(spec.shape.numel() * width).map_err(|_| {
                Failure::message(format!(
                    "the table `{}` is {:?} and does not fit a host allocation",
                    spec.name,
                    spec.shape.dims()
                ))
            })?;
            Ok(vec![0_u8; bytes])
        })
        .collect()
}

fn upload_bytes(stream: &platform::Stream, view: TensorView, bytes: &[u8]) -> Result<()> {
    let width = view
        .dtype
        .bytes()
        .ok_or_else(|| Failure::message(format!("a {} view has no byte width", view.dtype)))?;
    let expected = usize::try_from(view.shape.numel() * width).unwrap_or(usize::MAX);
    if bytes.len() != expected {
        return Err(Failure::message(format!(
            "{} bytes for a {} view of {:?}, which holds {expected}",
            bytes.len(),
            view.dtype,
            view.shape.dims()
        )));
    }
    // SAFETY: `bytes` outlives the copy, which completes before the wait below.
    unsafe {
        platform::memcpy_async(
            view.ptr.address(),
            bytes.as_ptr() as u64,
            bytes.len(),
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(())
}

fn write_h3t(path: &Path, shape: &[u64], dtype: u8, payload: &[u8]) -> Result<()> {
    let mut bytes = Vec::with_capacity(6 + shape.len() * 8 + payload.len());
    bytes.extend_from_slice(b"H3D1");
    bytes.push(dtype);
    bytes.push(u8::try_from(shape.len()).unwrap_or(0));
    for extent in shape {
        bytes.extend_from_slice(&extent.to_le_bytes());
    }
    bytes.extend_from_slice(payload);
    std::fs::write(path, bytes)
        .map_err(|error| Failure::message(format!("{}: {error}", path.display())))?;
    Ok(())
}

struct DeviceWeights {
    pools: platform::Allocator,
    buffers: Vec<platform::Buffer>,
    bytes: u64,
}

impl DeviceWeights {
    fn new(device: &platform::Device, requested: u64) -> Result<Self> {
        let budget = std::sync::Arc::new(NoLimit);
        let mut pools = platform::Allocator::new(device, budget)?;
        let mut wanted = requested.max(8 << 30);
        let mut reserved = None;
        while wanted >= (1 << 30) {
            if pools
                .reserve(
                    platform::PoolKind::Resident,
                    usize::try_from(wanted).unwrap_or(usize::MAX),
                )
                .is_ok()
            {
                reserved = Some(wanted);
                break;
            }
            wanted /= 2;
        }
        let reserved = reserved.ok_or_else(|| {
            Failure::message(
                "the driver refused every resident reservation down to 1 GiB: the loaded layers' \
                 weights cannot be held on this device",
            )
        })?;
        println!("resident pool: {} GiB reserved", reserved >> 30);
        Ok(Self {
            pools,
            buffers: Vec::new(),
            bytes: 0,
        })
    }

    fn allocator(&mut self) -> &mut platform::Allocator {
        &mut self.pools
    }
}

impl platform::Budget for NoLimit {
    fn book(
        &self,
        _device: platform::DeviceId,
        _pool: platform::PoolKind,
        _bytes: u64,
    ) -> platform::Result<()> {
        Ok(())
    }
}

impl RetainedWeights for DeviceWeights {
    fn retained(&mut self, _key: &WeightKey) -> nn::Result<Option<Backing>> {
        Ok(None)
    }

    fn publish(
        &mut self,
        key: &WeightKey,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
    ) -> nn::Result<Backing> {
        let buffer = self
            .pools
            .allocate(platform::PoolKind::Resident, bytes.len())?;
        // The publish is synchronous with respect to the host: the copy is issued on the device's
        // default stream and waited on before the backing is handed out, because the loader keeps no
        // stream and a later launch may use another one.
        let stream = platform::Stream::new(buffer.device())?;
        // SAFETY: `bytes` outlives the copy, which completes before the wait below.
        unsafe {
            platform::memcpy_async(
                buffer.address(),
                bytes.as_ptr() as u64,
                bytes.len(),
                &stream,
            )?;
        }
        stream.synchronize()?;
        let backing = Backing::new(DevicePtr::new(buffer.address()), bytes.len() as u64);
        backing.view(dtype, shape)?;
        self.bytes += bytes.len() as u64;
        self.buffers.push(buffer);
        let _ = key;
        Ok(backing)
    }
}

fn json_field(text: &str, field: &str) -> Option<String> {
    let needle = format!("\"{field}\"");
    let at = text.find(&needle)? + needle.len();
    let rest = text[at..].trim_start().strip_prefix(':')?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(rest[..end].to_owned())
}

fn load_generation(device: &platform::Device, root: &Path) -> Result<ops::Generation> {
    if !root.is_dir() {
        return Err(Failure::message(format!(
            "the store root {} is not a directory: nothing is installed there, so no stage past \
             `weights` can launch",
            root.display()
        )));
    }
    let mut entries = Vec::new();
    let mut installed = 0_usize;
    let mut operators: Vec<String> = std::fs::read_dir(root)
        .map_err(|error| Failure::message(format!("{}: {error}", root.display())))?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .filter_map(|path| path.file_name()?.to_str().map(str::to_owned))
        .collect();
    operators.sort();
    for operator in &operators {
        for (descriptor, directory) in committed(&root.join(operator)) {
            entries.push(install(device, operator, &descriptor, &directory)?);
            installed += 1;
        }
    }
    println!(
        "store: {installed} committed modules of {} operators",
        operators.len()
    );
    Ok(ops::Generation::new(1, entries))
}

fn committed(op_root: &Path) -> Vec<(String, PathBuf)> {
    let mut found = Vec::new();
    let backends = std::fs::read_dir(op_root).into_iter().flatten().flatten();
    for backend in backends {
        let revisions = std::fs::read_dir(backend.path())
            .into_iter()
            .flatten()
            .flatten();
        for revision in revisions {
            let targets = std::fs::read_dir(revision.path())
                .into_iter()
                .flatten()
                .flatten();
            for target in targets {
                let descriptors = std::fs::read_dir(target.path())
                    .into_iter()
                    .flatten()
                    .flatten();
                for descriptor in descriptors {
                    if descriptor.path().join(".committed").is_file() {
                        found.push((
                            descriptor.file_name().to_string_lossy().into_owned(),
                            descriptor.path(),
                        ));
                    }
                }
            }
        }
    }
    found.sort_by(|left, right| left.0.cmp(&right.0));
    found
}

fn install(
    device: &platform::Device,
    operator: &str,
    descriptor: &str,
    directory: &Path,
) -> Result<ops::KernelEntry> {
    let metadata_path = directory.join("metadata.json");
    let metadata = std::fs::read(&metadata_path).ok();
    let launch = std::fs::read_to_string(directory.join("launch.json")).unwrap_or_default();
    let entry = metadata
        .as_deref()
        .and_then(|bytes| json_field(&String::from_utf8_lossy(bytes), "name"))
        .or_else(|| json_field(&launch, "name"))
        .unwrap_or_else(|| format!("uniserve_{operator}"));
    let cubin = directory.join("module.cubin");
    if cubin.is_file() {
        let image = std::fs::read(&cubin)
            .map_err(|error| Failure::message(format!("reading {}: {error}", cubin.display())))?;
        let kernel = ops::install::install(
            device,
            operator,
            descriptor,
            ops::install::Module::Cubin {
                image: &image,
                metadata: metadata.as_deref(),
                entry: &entry,
            },
        )
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
        return kernel.ok_or_else(|| {
            Failure::message(format!(
                "{}: this build of `ops` has no operator `{operator}`",
                directory.display()
            ))
        });
    }
    let library = std::fs::read_dir(directory)
        .into_iter()
        .flatten()
        .flatten()
        .map(|file| file.path())
        .find(|path| path.extension().is_some_and(|kind| kind == "so"))
        .ok_or_else(|| {
            Failure::message(format!(
                "{}: neither module.cubin nor a shared object",
                directory.display()
            ))
        })?;
    let installed = ops::install::install(
        device,
        operator,
        descriptor,
        ops::install::Module::Native {
            path: &library,
            entry: &entry,
        },
    )
    .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    installed.ok_or_else(|| {
        Failure::message(format!(
            "{}: this build of `ops` has no operator `{operator}`",
            directory.display()
        ))
    })
}

fn upload_i32(
    stream: &platform::Stream,
    arena: &Arena,
    shape: Shape,
    values: &[i32],
) -> Result<TensorView> {
    let tensor = arena.tensor(DType::I32, shape)?;
    if tensor.shape.numel() != u64::try_from(values.len()).unwrap_or(u64::MAX) {
        return Err(Failure::message(format!(
            "{} values for the extents {:?}",
            values.len(),
            shape.dims()
        )));
    }
    // SAFETY: `values` outlives the copy, which completes before the wait below.
    unsafe {
        platform::memcpy_async(
            tensor.ptr.address(),
            values.as_ptr() as u64,
            std::mem::size_of_val(values),
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(tensor)
}

/// The container's element-type code (`.h3t`, from `tools/dump_tables.py`'s table).
///
/// The page is written at the checkpoint's own element type, so its code is derived from the dtype
/// the load used: a hard-coded code would describe a tensor of another width, which is the one
/// defect a reader of the page cannot see.
fn code_of(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 1,
        DType::I32 => 3,
        DType::BF16 => 4,
        DType::F16 => 5,
        _ => 4,
    }
}

/// `elements` of `view`'s own width, copied back from the device.
fn download(stream: &platform::Stream, view: TensorView, elements: usize) -> Result<Vec<u8>> {
    let width = usize::try_from(view.dtype.bytes().unwrap_or(u64::MAX)).map_err(|_| {
        Failure::message("an element type wider than this host addresses".to_owned())
    })?;
    let mut bytes = vec![0_u8; elements * width];
    // SAFETY: `bytes` is written by the copy, which completes before the wait below.
    unsafe {
        platform::memcpy_async(
            bytes.as_mut_ptr() as u64,
            view.ptr.address(),
            bytes.len(),
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(bytes)
}

struct Walk(Vec<(&'static str, String)>);

impl ops::capture::Sink for Walk {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.push((operator, key.to_owned()));
    }
}

struct SinkGuard;

impl Drop for SinkGuard {
    fn drop(&mut self) {
        ops::capture::set_sink(None);
    }
}

impl Walk {
    /// The walk as the store's key list is written: one `operator<TAB>key` per launch, in order.
    fn text(&self) -> String {
        let mut text = String::new();
        for (operator, key) in &self.0 {
            text.push_str(operator);
            text.push('\t');
            text.push_str(key);
            text.push('\n');
        }
        text
    }

    /// How many distinct programs the walk bound.
    fn distinct(&self) -> usize {
        let mut keys: Vec<(&str, &str)> = self
            .0
            .iter()
            .map(|(operator, key)| (*operator, key.as_str()))
            .collect();
        keys.sort_unstable();
        keys.dedup();
        keys.len()
    }
}

/// Stage 2: one prompt on one rank, through the device.
///
/// Not written yet. It says what it will do rather than pretending, because a case that those
/// numbers describe is what the reference probe reads back - and a harness that wrote a case without
/// running would make a page look like a disagreement.
fn stage_encode(args: &Args) -> Result<()> {
    let config = Config::read(&args.root, None)?;
    let text = &config.text_encoder;
    let ids = &args.prompt_ids;
    let tokens = u32::try_from(ids.len())
        .map_err(|_| Failure::message("a prompt of more than four billion ids"))?;
    if tokens == 0 {
        return Err(Failure::message(
            "the prompt is empty: an encode needs at least one token id",
        ));
    }
    let store = args.store.clone().ok_or_else(|| {
        Failure::message("--store is required for `encode`: the walk's programs are read from it")
    })?;
    // The flag, the checkpoint and the element type are host checks: they run before a device is
    // opened, so a typo costs a second rather than a device.
    let stated = stated_dtype(&args)?;
    let checkpoint = Checkpoint::open(&args.root)?;
    let source = checkpoint.component("text_encoder")?;
    let dtype = compute_dtype(&checkpoint, TEXT_ENCODER, stated)?;

    // 1. The device and the generation the compiled programs were loaded for.
    let device = platform::Device::new(DeviceId(args.device))?;
    let _guard = device.enter()?;
    let stream = device.stream()?;
    let generation = load_generation(&device, &store)?;

    // 2. The weights, through the loader's own boundary: every tensor is checked for its element
    //    type and shape as it is handed over, so a checkpoint this encoder cannot read is refused
    //    here rather than at the first launch.
    //
    let mut weights = DeviceWeights::new(&device, 0)?;
    let encoder = {
        let mut loader = WeightLoader::new(source, &mut weights, stated, None);
        TextEncoder::load(&mut loader, text, &Partition::single(), dtype)?
    };
    // The type the weights came back in, beside the one asked for: a loader that quietly widened or
    // narrowed a component would show here rather than in a page nobody can attribute.
    println!(
        "loaded: text encoder at {}, {} requested",
        encoder.dtype(),
        dtype
    );

    // 3. The arena. The encoder's own temporaries are reserved inside the forward, so this region
    //    only has to be large enough: a wide factor over the prompt's own rows, and the forward's
    //    reservation refuses by name if the estimate is ever short.
    let hidden = u64::from(text.hidden_size);
    let rows_bound = 4096_u64.max(u64::from(tokens) * 2);
    let bytes = rows_bound * hidden * 2 * 32 + (1 << 30);
    let allocator = weights.allocator();
    allocator.reserve(
        PoolKind::Graph,
        usize::try_from(bytes).unwrap_or(usize::MAX),
    )?;
    let region = allocator.allocate(
        PoolKind::Graph,
        usize::try_from(bytes).unwrap_or(usize::MAX),
    )?;
    let arena = Arena::new(DevicePtr::new(region.address()), bytes);
    // Said out loud because the number is not a measurement: a deployment row must come from a plan
    // function or from a run, and a harness's headroom that looked like a plan has already written
    // one row wrong.
    println!(
        "arena: {rows_bound} rows bound, {} MiB reserved - the harness's over-reserve, not a \
         measured plan",
        bytes >> 20
    );
    let forward = Forward::new(&stream, &generation, &arena);

    // 4. The batch: one unpadded sequence of `tokens` rows, no cache, no draft.
    //
    // The encoder reads the row table and the key bound; the rest of the batch is the runner's
    // shape, and it is filled with the values that mean "nothing": no cache slot written, no row
    // whose logits are produced, one sequence over the whole prompt.
    let tokens64 = u64::from(tokens);
    let positions = upload_i32(
        &stream,
        &arena,
        Shape::new(&[tokens64]),
        &vec![0_i32; tokens as usize],
    )?;
    let slot_mapping = upload_i32(
        &stream,
        &arena,
        Shape::new(&[tokens64]),
        &vec![-1_i32; tokens as usize],
    )?;
    let seq_lens = upload_i32(&stream, &arena, Shape::new(&[1]), &[tokens as i32])?;
    let query_offsets = upload_i32(&stream, &arena, Shape::new(&[2]), &[0, tokens as i32])?;
    let logits_rows = upload_i32(&stream, &arena, Shape::new(&[1]), &[0])?;
    let batch = TokenBatch {
        positions,
        slot_mapping,
        seq_lens,
        query_offsets,
        logits_rows,
        phase: ops::PhaseKind::Encode,
        page_size: 1,
        num_tokens: tokens,
        max_kv_len: tokens,
        draft_lens: None,
    };

    // 5. The layout, its tables, and their host bytes: the model's own request arithmetic, written
    //    where a captured call cannot reach it.
    let layout = encoder.layout(&batch, &[]);
    let specs = encoder.state_tables(&layout);
    let mut host = host_tables(&specs)?;
    encoder.prepare_state(&EncoderPrompt { ids }, &layout, &mut host)?;
    let state = allocate_tables(&arena, &specs)?;
    for (view, bytes) in state.iter().zip(host.iter()) {
        upload_bytes(&stream, *view, bytes)?;
    }
    println!(
        "layout: {} rows for {tokens} tokens, {} state tables of {} bytes",
        layout.rows,
        specs.len(),
        host.iter().map(Vec::len).sum::<usize>()
    );

    // 6. The ids and the output rows the encode fills.
    let signed = ids
        .iter()
        .map(|id| {
            i32::try_from(*id).map_err(|_| Failure::message(format!("id {id} is beyond i32")))
        })
        .collect::<Result<Vec<i32>>>()?;
    let ids_view = upload_i32(&stream, &arena, Shape::new(&[tokens64]), &signed)?;
    let output = arena.tensor(dtype, Shape::new(&[u64::from(layout.rows), hidden]))?;

    // 7. The encode, with the executor's own recorder installed for its duration: the walk is what
    //    proves every program the forward binds is in the store's list.
    let mut walk = Walk(Vec::new());
    ops::capture::set_sink(Some(&mut walk));
    let sink = SinkGuard;
    encoder.encode(
        &forward,
        EncoderInputs {
            ids: ids_view,
            batch,
            images: &[],
            layout: &layout,
            state: &state,
            diagnostics: &[],
        },
        EncoderOutputs { hidden: output },
    )?;
    drop(sink);

    // 8. The page: the rows, the case that says what they are, and the walk.
    let rows = u64::from(layout.rows);
    let elements = usize::try_from(rows * hidden)
        .map_err(|_| Failure::message("the output is larger than this host addresses"))?;
    let payload = download(&stream, output, elements)?;
    let walked = walk.text();
    println!(
        "walk: {} launches, {} distinct programs",
        walk.0.len(),
        walk.distinct()
    );

    let directory = args.out.join("text_encoder");
    std::fs::create_dir_all(&directory)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    write_h3t(
        &directory.join("prompt_hidden.h3t"),
        &[rows, hidden],
        code_of(dtype),
        &payload,
    )?;
    std::fs::write(directory.join("walk.txt"), walked)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    let ids_json = ids
        .iter()
        .map(u32::to_string)
        .collect::<Vec<String>>()
        .join(", ");
    let case = format!(
        "{{\n  \"task\": \"t2va\",\n  \"text_tokens\": {tokens},\n  \"hidden\": {hidden},\n  \
         \"prompt\": {prompt},\n  \"prompt_ids\": [{ids_json}],\n  \"vision\": \"none\",\n  \
         \"vocab_size\": {vocab},\n  \"dtype\": \"{element}\",\n  \
         \"checkpoint_dtype\": \"{own}\"\n}}\n",
        prompt = json_string(&args.prompt),
        vocab = text.vocab_size,
        element = dtype,
        own = load::stored_dtype(&checkpoint, TEXT_ENCODER)?,
    );
    std::fs::write(directory.join("case.json"), case)
        .map_err(|error| Failure::message(format!("{}: {error}", directory.display())))?;
    println!(
        "wrote   {} and case.json: {rows} rows of {hidden}, {} bytes",
        directory.join("prompt_hidden.h3t").display(),
        payload.len()
    );
    Ok(())
}

/// A prompt as a JSON string literal.
///
/// The case carries the prompt as text so the reference probe can tokenize it with the checkpoint's
/// own tokenizer; a prompt with a quote or a backslash in it must not produce a case that fails to
/// parse, which is what `format!` alone would do.
fn json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}
