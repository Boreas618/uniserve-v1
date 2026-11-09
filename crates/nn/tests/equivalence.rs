//! Bitwise equivalence of the forms the shared layers are about to merge.
//!
//! Each case runs two forms of one computation on the same inputs and reports how far their output
//! bytes are apart, so the merge each case gates either collapses the two forms into one or keeps a
//! named parameter of the shared layer. A case is a measurement, not a gate: it prints one line per
//! comparison and fails only when the harness itself cannot run.
//!
//! ```text
//! equivalence <name>: identical
//! equivalence <name>: differs <n> elements, max <k> ulp
//! ```
//!
//! `ulp` is the distance between the two values in the output's own element type (adjacent
//! representable values are one apart; `+0` and `-0` differ by zero). The refiner check is a
//! statement about configs, so it can also answer `absent`.
//!
//! | Case | Forms | Kernels |
//! | --- | --- | --- |
//! | `device::ones_scaled_residual` | a channel-scaled `Residual` over an all-ones scale vs `Axpby(1, 1)` and vs one rounding of the sum | `scaled_residual`, `scaled_add` |
//! | `device::channel_affine` | the diagonal-matrix products vs `x·s+m` and `(x-m)/s` on the host | `gemm`, `conv1d`, `conv3d` |
//! | `device::frame_blend` | the `Axpby` frame loop vs `CrossFade` | `scaled_add`, `cross_fade` |
//! | `device::tile_stitch_corners` | `TileStitch` vs `CrossFade` vertical then horizontal, neighbours blended in place | `tile_stitch`, `cross_fade` |
//! | `device::bias_after_identity_gemm` | the identity-matrix product with a bias vs `bias_add` | `gemm`, `bias_add` |
//! | `host::refiner_final_norm_eps` | `final_norm_eps` vs `norm_eps` in each shipped H3 transformer config | none |
//!
//! Shapes, element types and parameters are the ones the shipped forwards call with
//! (`/workspace/models/Qwen3-30B-A3B`, `MiniMax-H3`, `FastVideo-FastH3-8-Step-V2`); the norm gains and
//! the post-quant bias are read from those checkpoints, and every activation is drawn from a
//! fixed-seed distribution over the activation's range.
//!
//! ## Running
//!
//! The host cases read the checkpoints' configs (`UNISERVE_MODEL_ROOT`, default `/workspace/models`):
//!
//! ```text
//! cargo test --release -p nn --test equivalence -- --ignored --nocapture host::
//! ```
//!
//! The device cases take one GPU (`CUDA_VISIBLE_DEVICES`) and a running Program Server whose socket
//! `UNISERVE_PROGRAM_SERVER_SOCKET` names; the server builds any module the store lacks on first use.
//! With the model-library store:
//!
//! ```text
//! A=/workspace/dev/uniserve-v1/artifacts/model-library/phase2
//! $A/bin/uniserve program-server serve --daemon \
//!     --socket $A/store/program-server.sock \
//!     --package-root $A/store/packages --store-root $A/store/modules \
//!     --target sm_100a --wasm $A/bin/scheduling-precompile \
//!     --python /workspace/dev/uniserve-v1/crates/program-server/python/.venv/bin/python \
//!     --max-jobs 16
//! UNISERVE_PROGRAM_SERVER_SOCKET=$A/store/program-server.sock CUDA_VISIBLE_DEVICES=0 \
//!     cargo test --release -p nn --test equivalence -- --ignored --nocapture --test-threads=1 device::
//! ```

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use nn::layers::{
    Axpby, Conv1d, Conv3d, ConvSpec, CrossFade, Linear, Residual, Scale, TileGrid, TileStitch,
    WeightOrder,
};
use nn::{Arena, Backing, DType, DevicePtr, Forward, Shape, Strides, TensorView, Weight};
use ops::values::{f16_to_f32, f32_to_bf16, f32_to_f16};

/// Every harness failure: a missing asset, a refused launch, a device or Program Server error.
type Outcome<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// The directory holding the shipped checkpoints.
fn models() -> PathBuf {
    std::env::var_os("UNISERVE_MODEL_ROOT")
        .map_or_else(|| PathBuf::from("/workspace/models"), PathBuf::from)
}

const H3: &str = "MiniMax-H3";
const FAST_H3: &str = "FastVideo-FastH3-8-Step-V2";

/// A config file as JSON.
fn config(path: &Path) -> Outcome<serde_json::Value> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

/// The number at `pointer` (`/text_config/rope_theta`) of `value`.
fn number(value: &serde_json::Value, pointer: &str) -> Outcome<f64> {
    value
        .pointer(pointer)
        .and_then(serde_json::Value::as_f64)
        .ok_or_else(|| format!("the config states no number at {pointer}").into())
}

/// The numbers of the array at `pointer`, as the model configs read them: to `f32`.
fn numbers(value: &serde_json::Value, pointer: &str) -> Outcome<Vec<f32>> {
    value
        .pointer(pointer)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("the config states no array at {pointer}"))?
        .iter()
        .map(|item| {
            item.as_f64()
                .map(|number| number as f32)
                .ok_or_else(|| format!("{pointer} holds a non-number").into())
        })
        .collect()
}

// ---- Fixed-seed inputs ----------------------------------------------------------------------

/// A SplitMix64 stream: the same seed draws the same inputs on every host.
struct Draw(u64);

impl Draw {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut mixed = self.0;
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^ (mixed >> 31)
    }

    /// A uniform value in `(0, 1]`, from the top 53 bits.
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) + 1) as f64 / (1_u64 << 53) as f64
    }

    /// `count` values uniform in `[low, high)`.
    fn uniform(&mut self, count: usize, low: f32, high: f32) -> Vec<f32> {
        (0..count)
            .map(|_| low + (high - low) * (1.0 - self.unit()) as f32)
            .collect()
    }

    /// One standard normal value (Box-Muller, one of each pair).
    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * self.unit().ln()).sqrt();
        radius * (std::f64::consts::TAU * self.unit()).cos()
    }

    /// `count` values of `N(0, std²)`.
    fn normals(&mut self, count: usize, std: f32) -> Vec<f32> {
        (0..count).map(|_| (self.normal() as f32) * std).collect()
    }
}

// ---- Element bytes ----------------------------------------------------------------------------

/// `values` rounded to `dtype` (round to nearest even), as little-endian bytes.
fn encode(dtype: DType, values: &[f32]) -> Outcome<Vec<u8>> {
    Ok(match dtype {
        DType::F32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        DType::F16 => values
            .iter()
            .flat_map(|value| f32_to_f16(*value).to_le_bytes())
            .collect(),
        DType::BF16 => values
            .iter()
            .flat_map(|value| f32_to_bf16(*value).to_le_bytes())
            .collect(),
        other => return Err(format!("no float encoding for {other}").into()),
    })
}

/// The values of little-endian `bytes` of `dtype`, widened to `f32` exactly.
fn decode(dtype: DType, bytes: &[u8]) -> Outcome<Vec<f32>> {
    Ok(match dtype {
        DType::F32 => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect(),
        DType::F16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|half| f16_to_f32(u16::from_le_bytes(*half)))
            .collect(),
        DType::BF16 => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|half| f32::from_bits(u32::from(u16::from_le_bytes(*half)) << 16))
            .collect(),
        other => return Err(format!("no float decoding for {other}").into()),
    })
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

// ---- Comparison ---------------------------------------------------------------------------------

/// The bits of element `index` of `bytes`, whose elements are `width` bytes.
fn bits(bytes: &[u8], width: usize, index: usize) -> u32 {
    let mut word = [0_u8; 4];
    word[..width].copy_from_slice(&bytes[index * width..(index + 1) * width]);
    u32::from_le_bytes(word)
}

/// A float's bits on a line where adjacent representable values are one apart and the signed
/// zeros meet, so the distance of two values is their ulp distance.
fn ordered(bits: u32, width: usize) -> i64 {
    let sign = 1_u32 << (width * 8 - 1);
    if bits & sign == 0 {
        i64::from(bits)
    } else {
        -i64::from(bits & !sign)
    }
}

/// How two outputs of `dtype` differ: the elements whose bits differ, and the largest distance.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Difference {
    elements: u64,
    ulp: u64,
}

fn difference(dtype: DType, left: &[u8], right: &[u8]) -> Outcome<Difference> {
    let width = dtype.bytes().ok_or("an element type without a width")? as usize;
    if left.len() != right.len() || !left.len().is_multiple_of(width) {
        return Err(format!(
            "outputs of {} and {} bytes are not two tensors of {dtype}",
            left.len(),
            right.len()
        )
        .into());
    }
    let mut found = Difference::default();
    for index in 0..left.len() / width {
        let (a, b) = (bits(left, width, index), bits(right, width, index));
        if a != b {
            found.elements += 1;
            found.ulp = found.ulp.max(ordered(a, width).abs_diff(ordered(b, width)));
        }
    }
    Ok(found)
}

/// Prints the one result line of comparison `name`.
fn report(name: &str, found: Difference) {
    if found.elements == 0 {
        println!("equivalence {name}: identical");
    } else {
        println!(
            "equivalence {name}: differs {} elements, max {} ulp",
            found.elements, found.ulp
        );
    }
}

/// Measures and reports two outputs of `dtype`.
fn measure(name: &str, dtype: DType, left: &[u8], right: &[u8]) -> Outcome {
    report(name, difference(dtype, left, right)?);
    Ok(())
}

// ---- Checkpoint tensors -------------------------------------------------------------------------

/// The safetensors file of `directory` that holds `name`: the one the index maps it to, or the
/// directory's single file.
fn shard(directory: &Path, name: &str) -> Outcome<PathBuf> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(directory)? {
        let path = entry?.path();
        let file = path
            .file_name()
            .and_then(|file| file.to_str())
            .unwrap_or_default();
        if file.ends_with(".safetensors.index.json") {
            let index = config(&path)?;
            let shard = index
                .pointer(&format!(
                    "/weight_map/{}",
                    name.replace('~', "~0").replace('/', "~1")
                ))
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| format!("{} maps no {name}", path.display()))?;
            return Ok(directory.join(shard));
        }
        if file.ends_with(".safetensors") {
            files.push(path);
        }
    }
    match files.as_slice() {
        [only] => Ok(only.clone()),
        _ => Err(format!(
            "{} has no index and {} shards",
            directory.display(),
            files.len()
        )
        .into()),
    }
}

/// The checkpoint tensor `name` of `directory`, widened to `f32`.
fn tensor(directory: &Path, name: &str) -> Outcome<Vec<f32>> {
    let path = shard(directory, name)?;
    let file = std::fs::File::open(&path)?;
    // SAFETY: the shard is a read-only checkpoint file nothing writes while the harness runs, and
    // the mapping is dropped before this function returns.
    let mapped = unsafe { memmap2::Mmap::map(&file)? };
    let tensors = safetensors::SafeTensors::deserialize(&mapped)?;
    let view = tensors.tensor(name)?;
    let dtype = match view.dtype() {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        other => return Err(format!("{name} is stored as {other:?}").into()),
    };
    decode(dtype, view.data())
}

// ---- Video tile grid ----------------------------------------------------------------------------

/// The checkpoint's decode tile and minimum seam, in pixels (`minimax-h3` `video_vae::Tiles::of`),
/// and the decoder's spatial compression and patch frames (`vae/config.json`'s downsample factors).
const TILE_PIXELS: u32 = 256;
const MINIMUM_OVERLAP: u32 = 64;
const SPATIAL: u32 = 16;
const PATCH_FRAMES: u32 = 4;

/// One axis of a canvas cut into overlapping tiles: where each starts, and each seam's width.
struct Seams {
    starts: Vec<u32>,
    overlaps: Vec<u32>,
}

/// The reference's `split_tiles` as `minimax-h3` `video_vae::Seams::split` computes it: the fewest
/// tiles whose minimum-overlap coverage reaches `length`, the slack spread over the seams in whole
/// alignment steps.
fn split(length: u32, tile: u32, overlap: u32, alignment: u32) -> Seams {
    if tile >= length {
        return Seams {
            starts: vec![0],
            overlaps: Vec::new(),
        };
    }
    let mut count = length.div_ceil(tile);
    while tile * count - overlap * (count - 1) < length {
        count += 1;
    }
    let mut overlaps = vec![overlap; (count - 1) as usize];
    let slack = tile * count - overlap * (count - 1) - length;
    for index in 0..(slack / alignment) as usize {
        let seam = index % overlaps.len();
        overlaps[seam] += alignment;
    }
    let mut starts = vec![0];
    for overlap in &overlaps {
        starts.push(starts[starts.len() - 1] + tile - overlap);
    }
    Seams { starts, overlaps }
}

/// The stitch table of one axis (`minimax-h3` `decoding::stitch_axis`): per position, the tile
/// that keeps it, the position inside that tile, and the seam it lies in (zero outside one).
fn stitch_axis(seams: &Seams, length: u32) -> Vec<[i32; 3]> {
    let mut rows = Vec::with_capacity(length as usize);
    let mut tile = 0_usize;
    for position in 0..length {
        while tile + 1 < seams.starts.len() && seams.starts[tile + 1] <= position {
            tile += 1;
        }
        let inside = position - seams.starts[tile];
        let seam = tile
            .checked_sub(1)
            .and_then(|previous| seams.overlaps.get(previous).copied())
            .filter(|overlap| inside < *overlap)
            .unwrap_or(0);
        rows.push([tile as i32, inside as i32, seam as i32]);
    }
    rows
}

mod host {
    use super::*;

    /// Case 7: whether each shipped H3 transformer config states a refiner final-norm epsilon
    /// distinct from `norm_eps`. `identical` means the two fields hold the same number; `ulp` is
    /// their distance as the f32 the config reader keeps.
    #[test]
    #[ignore = "reads the shipped checkpoints' configs under UNISERVE_MODEL_ROOT"]
    fn refiner_final_norm_eps() -> Outcome {
        for checkpoint in [H3, FAST_H3] {
            let root = models().join(checkpoint);
            let mut configs = BTreeSet::new();
            let mut pending = vec![(root.clone(), 0)];
            while let Some((directory, depth)) = pending.pop() {
                for entry in std::fs::read_dir(&directory)? {
                    let path = entry?.path();
                    if path.is_dir() && depth < 2 {
                        pending.push((path, depth + 1));
                    } else if path.file_name().is_some_and(|name| name == "config.json") {
                        configs.insert(path);
                    }
                }
            }
            for path in configs {
                let value = config(&path)?;
                let Some(norm) = value.get("norm_eps").and_then(serde_json::Value::as_f64) else {
                    continue;
                };
                // Both transformer schemas the checkpoints ship name the refiner's depth.
                let refiner = ["num_refiner_layers", "token_refiner_num_layers"];
                if !refiner.iter().any(|key| value.get(key).is_some()) {
                    // A config with a `norm_eps` but no refiner is not a transformer's.
                    continue;
                }
                let relative = path.strip_prefix(models())?.display().to_string();
                let name = format!("final_norm_eps_vs_norm_eps_{relative}");
                match value
                    .get("final_norm_eps")
                    .and_then(serde_json::Value::as_f64)
                {
                    None => println!("equivalence {name}: absent"),
                    Some(last) => measure(
                        &name,
                        DType::F32,
                        &encode(DType::F32, &[last as f32])?,
                        &encode(DType::F32, &[norm as f32])?,
                    )?,
                }
            }
        }
        Ok(())
    }
}

mod device {
    use std::io::Read as _;
    use std::os::fd::AsRawFd as _;

    use ops::{Generation, KernelEntry};
    use platform::{DeviceAllocation, DeviceId, Stream};
    use program_server::{
        Client, Descriptors, ModuleForm, PrepareRequest, ProgramRequest, ProgramSlot, Selector,
        Target,
    };

    use super::*;

    const DEVICE: DeviceId = DeviceId(0);

    /// Bytes of the arena a forward may reserve scratch from.
    const ARENA_BYTES: usize = 64 << 20;

    /// The programs a recording walk named, as `(operator, key)` pairs.
    #[derive(Default)]
    struct Launches(Vec<(&'static str, String)>);

    impl ops::capture::Sink for Launches {
        fn launched(&mut self, operator: &'static str, key: &str) {
            self.0.push((operator, key.to_owned()));
        }
    }

    /// One device, its stream, the Program Server that supplies its kernels, and every buffer a
    /// case allocated, freed when the bench drops.
    struct Bench {
        device: platform::Device,
        stream: Stream,
        client: Client,
        target: Target,
        /// The backend each installed operator's package is, from the store's listing.
        backends: BTreeMap<String, Vec<String>>,
        /// Every kernel installed so far, by operator and key: each case asks only for the ones
        /// it has not met.
        kernels: HashMap<(&'static str, String), KernelEntry>,
        arena: Arena,
        buffers: Vec<DeviceAllocation>,
    }

    impl Bench {
        fn open() -> Outcome<Self> {
            let socket = std::env::var_os("UNISERVE_PROGRAM_SERVER_SOCKET").ok_or(
                "UNISERVE_PROGRAM_SERVER_SOCKET names no Program Server: start one with \
                 `uniserve program-server serve` (see this file's header)",
            )?;
            let device = platform::Device::new(DEVICE)?;
            let stream = Stream::new(DEVICE)?;
            let target = Target::of(DEVICE)?;
            let mut client = Client::connect(Path::new(&socket))?;
            let mut backends: BTreeMap<String, Vec<String>> = BTreeMap::new();
            for package in client.list()?.packages {
                let known = backends.entry(package.op).or_default();
                if !known.contains(&package.backend) {
                    known.push(package.backend);
                }
            }
            let scratch = DeviceAllocation::linear(DEVICE, ARENA_BYTES)?;
            let arena = Arena::new(DevicePtr::new(scratch.address()), ARENA_BYTES as u64);
            Ok(Self {
                device,
                stream,
                client,
                target,
                backends,
                kernels: HashMap::new(),
                arena,
                buffers: vec![scratch],
            })
        }

        /// A new device tensor of `dtype` and `dims`, row-major, its contents undefined.
        fn tensor(&mut self, dtype: DType, dims: &[u64]) -> Outcome<TensorView> {
            let shape = Shape::new(dims);
            let bytes = shape.numel() * dtype.bytes().ok_or("an element type without a width")?;
            let buffer = DeviceAllocation::linear(DEVICE, bytes.max(1) as usize)?;
            let view = TensorView::new(DevicePtr::new(buffer.address()), dtype, shape);
            self.buffers.push(buffer);
            Ok(view)
        }

        /// Copies `bytes` into the device at `address`.
        fn write(&self, address: DevicePtr, bytes: &[u8]) -> Outcome {
            // SAFETY: the destination is a live allocation of at least `bytes.len()` bytes, the
            // source is this slice, and the stream is synchronized before the slice can go.
            unsafe {
                platform::memcpy_async(
                    address.address(),
                    bytes.as_ptr() as u64,
                    bytes.len(),
                    &self.stream,
                )?;
            }
            self.stream.synchronize()?;
            Ok(())
        }

        /// A device tensor holding `bytes`.
        fn upload(&mut self, dtype: DType, dims: &[u64], bytes: &[u8]) -> Outcome<TensorView> {
            let view = self.tensor(dtype, dims)?;
            if view.byte_len()? != bytes.len() as u64 {
                return Err(
                    format!("{} bytes for a {dtype} tensor of {dims:?}", bytes.len()).into(),
                );
            }
            self.write(view.ptr, bytes)?;
            Ok(view)
        }

        /// `values` rounded to `dtype` in a new device tensor.
        fn values(&mut self, dtype: DType, dims: &[u64], values: &[f32]) -> Outcome<TensorView> {
            let bytes = encode(dtype, values)?;
            self.upload(dtype, dims, &bytes)
        }

        /// A device weight holding `values` rounded to `dtype`.
        fn weight(&mut self, dtype: DType, dims: &[u64], values: &[f32]) -> Outcome<Weight> {
            let view = self.values(dtype, dims, values)?;
            Ok(Weight {
                backing: Backing::new(view.ptr, view.byte_len()?),
                view,
            })
        }

        /// The bytes of a row-major device tensor, after everything queued before.
        fn download(&self, view: TensorView) -> Outcome<Vec<u8>> {
            if !view.is_row_major() {
                return Err("a strided view is read through the tensor it lies in".into());
            }
            let mut bytes = vec![0_u8; view.byte_len()? as usize];
            // SAFETY: the source is a live device range of `bytes.len()` bytes and the destination
            // this vector; the stream is synchronized before the vector is read.
            unsafe {
                platform::memcpy_async(
                    bytes.as_mut_ptr() as u64,
                    view.ptr.address(),
                    bytes.len(),
                    &self.stream,
                )?;
            }
            self.stream.synchronize()?;
            Ok(bytes)
        }

        /// Runs `walk` on the device: a recording forward names its programs, the Program Server
        /// prepares and resolves the ones not yet installed, and a launching forward runs it.
        fn run(&mut self, walk: &dyn Fn(&Forward<'_>) -> nn::Result<()>) -> Outcome {
            let counting = Arena::counting();
            let mut launches = Launches::default();
            ops::capture::set_sink(Some(&mut launches));
            let recorded = walk(&Forward::recording(&counting));
            ops::capture::set_sink(None);
            recorded?;

            let mut missing: BTreeMap<&'static str, BTreeSet<String>> = BTreeMap::new();
            for (operator, key) in launches.0 {
                if !self.kernels.contains_key(&(operator, key.clone())) {
                    missing.entry(operator).or_default().insert(key);
                }
            }
            if !missing.is_empty() {
                self.install(&missing)?;
            }

            let generation = Generation::new(1, self.kernels.values().cloned());
            walk(&Forward::new(&self.stream, &generation, &self.arena))?;
            self.stream.synchronize()?;
            Ok(())
        }

        /// Prepares every missing program and installs each key's module into this device.
        fn install(&mut self, missing: &BTreeMap<&'static str, BTreeSet<String>>) -> Outcome {
            let mut programs = Vec::new();
            for operator in missing.keys() {
                let backend = match self.backends.get(*operator).map(Vec::as_slice) {
                    Some([only]) => only.clone(),
                    Some(several) => {
                        return Err(format!(
                            "{operator} has the packages {several:?}: name the one to measure"
                        )
                        .into());
                    }
                    None => return Err(format!("the store has no package of {operator}").into()),
                };
                programs.push(ProgramRequest {
                    slot: ProgramSlot::all(*operator),
                    selector: Selector::Latest {
                        backend,
                        arguments: Default::default(),
                    },
                });
            }
            let request = PrepareRequest {
                targets: vec![self.target.clone()],
                programs,
                descriptors: missing
                    .iter()
                    .map(|(operator, keys)| Descriptors {
                        op: (*operator).to_owned(),
                        keys: keys.iter().cloned().collect(),
                    })
                    .collect(),
            };
            let prepared = self.client.prepare(&request, &mut |_| {})?;
            for chosen in prepared.programs {
                let operator = ops::operator_name(&chosen.slot.op)
                    .ok_or_else(|| format!("{} is no operator of this build", chosen.slot.op))?;
                for key in missing.get(operator).into_iter().flatten() {
                    let module = self.client.resolve(
                        operator,
                        &chosen.program,
                        &self.target,
                        key,
                        &mut |_| {},
                    )?;
                    let entry = load(&self.device, operator, key, module)?;
                    self.kernels.insert((operator, key.clone()), entry);
                }
            }
            Ok(())
        }
    }

    /// Loads one resolved module into `device` as the kernel of `operator` for `key`.
    fn load(
        device: &platform::Device,
        operator: &str,
        key: &str,
        module: program_server::Module,
    ) -> Outcome<KernelEntry> {
        let read = |file: &program_server::ModuleFile| -> Outcome<Vec<u8>> {
            let mut bytes = Vec::with_capacity(file.size as usize);
            (&file.file).read_to_end(&mut bytes)?;
            Ok(bytes)
        };
        let entry = match module.form {
            ModuleForm::Cubin => {
                let mut image = None;
                let mut metadata = None;
                for file in &module.files {
                    if module.metadata.as_deref() == Some(file.name.as_str()) {
                        metadata = Some(read(file)?);
                    } else {
                        image = Some(read(file)?);
                    }
                }
                let image = image.ok_or("a cubin module carries no image")?;
                // The kernel's symbol is its metadata's name; without metadata `ops` takes the
                // module's single entry.
                let symbol = match &metadata {
                    Some(metadata) => ops::CubinMetadata::parse(metadata)?.name,
                    None => String::new(),
                };
                ops::install(
                    device,
                    operator,
                    key,
                    ops::Module::Cubin {
                        image: &image,
                        metadata: metadata.as_deref(),
                        entry: &symbol,
                    },
                )?
            }
            ModuleForm::Native => {
                let symbol = module
                    .entries
                    .get(program_server::manifest::LAUNCH_ENTRY)
                    .ok_or("a native module states no launch entry")?;
                // The library is the file that defines the entry; it is loaded where the store
                // keeps it, which the handle the server sent still names.
                let mut library = None;
                for file in &module.files {
                    let defines = platform::NativeLibrary::defined_functions(&read(file)?)
                        .is_some_and(|names| names.iter().any(|name| name == symbol));
                    if defines || module.files.len() == 1 {
                        library = Some(std::fs::read_link(format!(
                            "/proc/self/fd/{}",
                            file.file.as_raw_fd()
                        ))?);
                    }
                }
                let path = library.ok_or("no file of the native module defines its entry")?;
                ops::install(
                    device,
                    operator,
                    key,
                    ops::Module::Native {
                        path: &path,
                        entry: symbol,
                    },
                )?
            }
            other => return Err(format!("a {other:?} module is not a kernel").into()),
        };
        entry.ok_or_else(|| format!("{operator} is no operator of this build").into())
    }

    // ---- Case 2 -------------------------------------------------------------------------------

    /// Case 2: a channel-scaled `Residual` over the all-ones scale the H3 refiner, text encoder and vision
    /// tower publish, against a plain add: `Axpby` with both coefficients one, and the sum of
    /// the two BF16 operands rounded once (what a `Residual` with no scale computes).
    #[test]
    #[ignore = "requires a CUDA device and a Program Server (UNISERVE_PROGRAM_SERVER_SOCKET)"]
    fn ones_scaled_residual() -> Outcome {
        let root = models();
        let dit = config(&root.join(H3).join("transformer/config.json"))?;
        let text = config(&root.join(H3).join("text_encoder/config.json"))?;
        let mut bench = Bench::open()?;
        let mut draw = Draw::new(21);
        let dtype = DType::BF16;

        // Rows: a refiner request's 512 packed prompt rows, a 1024-token prompt for the text
        // encoder, and the 4096 patches of a 1024 x 1024 image for the vision tower.
        let sites = [
            ("refiner", number(&dit, "/hidden_size")? as u64, 512_u64),
            (
                "text_encoder",
                number(&text, "/text_config/hidden_size")? as u64,
                1024,
            ),
            (
                "vision",
                number(&text, "/vision_config/hidden_size")? as u64,
                4096,
            ),
        ];
        for (name, width, rows) in sites {
            let count = (width * rows) as usize;
            // The residual stream has grown past the update it receives: σ 4 against σ 1.
            let hidden = encode(dtype, &draw.normals(count, 4.0))?;
            let update = encode(dtype, &draw.normals(count, 1.0))?;
            let scaled = bench.upload(dtype, &[rows, width], &hidden)?;
            let added = bench.upload(dtype, &[rows, width], &hidden)?;
            let update_view = bench.upload(dtype, &[rows, width], &update)?;
            let ones = bench.weight(dtype, &[width], &vec![1.0; width as usize])?;
            let residual = Residual::new(Scale::Channel(ones), None)?;

            bench.run(&|f| {
                residual.forward(f, scaled, update_view, None, scaled)?;
                Axpby.forward(f, added, update_view, 1.0, 1.0)
            })?;

            let scaled = bench.download(scaled)?;
            measure(
                &format!("scaled_residual_ones_vs_scaled_add_{name}"),
                dtype,
                &scaled,
                &bench.download(added)?,
            )?;
            let sums: Vec<f32> = decode(dtype, &hidden)?
                .iter()
                .zip(decode(dtype, &update)?)
                .map(|(hidden, update)| hidden + update)
                .collect();
            measure(
                &format!("scaled_residual_ones_vs_rounded_sum_{name}"),
                dtype,
                &scaled,
                &encode(dtype, &sums)?,
            )?;
        }
        Ok(())
    }

    // ---- Case 4 -------------------------------------------------------------------------------

    /// Case 4: the latent space's per-channel affine, computed today as a product with a diagonal
    /// matrix and a bias, against the two host forms a `ChannelAffine` can compute: one fused
    /// multiply-add of the same tables, and the reference's separate operations (`x·s` then `+m`
    /// for the decoders' restore, `(x-m)/s` for the encoders' normalization). All FP32.
    #[test]
    #[ignore = "requires a CUDA device and a Program Server (UNISERVE_PROGRAM_SERVER_SOCKET)"]
    fn channel_affine() -> Outcome {
        let root = models();
        let vae = config(&root.join(H3).join("vae/config.json"))?;
        let audio = config(&root.join(H3).join("audio_vae/config.json"))?;
        let video_stats = (
            numbers(&vae, "/latents_mean")?,
            numbers(&vae, "/latents_std")?,
        );
        let audio_stats = (
            numbers(&audio, "/latents_mean")?,
            numbers(&audio, "/latents_std")?,
        );
        let mut bench = Bench::open()?;
        let mut draw = Draw::new(41);

        // The decoders restore what a denoiser published, unit-variance latents: the video
        // decoder's `[voxels, 24]` rows for the 28 tiles of a 720 x 1280 clip (7 x 16 x 16 voxels
        // a tile), the audio decoder's `[2, 32, 400]` stereo pair for ten seconds.
        let (mean, std) = &video_stats;
        let channels = mean.len() as u64;
        let voxels = 28 * 7 * 16 * 16_u64;
        let latents = draw.normals((voxels * channels) as usize, 1.0);
        let restore = Linear::new(
            bench.weight(DType::F32, &[channels, channels], &diagonal(std))?,
            Some(bench.weight(DType::F32, &[channels], mean)?),
            WeightOrder::NxK,
        )?;
        let input = bench.values(DType::F32, &[voxels, channels], &latents)?;
        let out = bench.tensor(DType::F32, &[voxels, channels])?;
        bench.run(&|f| restore.forward(f, input, out))?;
        restored(
            "video_decoder_restore_gemm",
            &bench.download(out)?,
            &latents,
            |index| index % channels as usize,
            std,
            mean,
        )?;

        let (mean, std) = &audio_stats;
        let channels = mean.len() as u64;
        let frames = 400_u64;
        let latents = draw.normals((2 * channels * frames) as usize, 1.0);
        let denormalize = Conv1d::new(
            bench.weight(DType::F32, &[channels, channels, 1], &diagonal(std))?,
            Some(bench.weight(DType::F32, &[channels], mean)?),
            one_by_one(channels, false),
        )?;
        let input = bench.values(DType::F32, &[2, channels, frames], &latents)?;
        let out = bench.tensor(DType::F32, &[2, channels, frames])?;
        bench.run(&|f| denormalize.forward(f, input, out))?;
        restored(
            "audio_decoder_restore_conv1d",
            &bench.download(out)?,
            &latents,
            |index| index / frames as usize % channels as usize,
            std,
            mean,
        )?;

        // The encoders normalize a posterior sample, which lies about each channel's statistics:
        // the video encoder's `[1, 24, 7, 45, 80]` condition raster (a 720 x 1280 clip), with the
        // TF32 flag its convolution is built with, and the audio encoder's `[1, 32, 400]`.
        let (mean, std) = &video_stats;
        let channels = mean.len() as u64;
        let positions = 7 * 45 * 80_u64;
        let samples = sample(&mut draw, mean, std, positions as usize);
        let (scale, shift) = normalize_tables(mean, std);
        let normalize = Conv3d::new(
            bench.weight(DType::F32, &[channels, channels, 1, 1, 1], &scale)?,
            Some(bench.weight(DType::F32, &[channels], &shift)?),
            one_by_one(channels, true),
        )?;
        let input = bench.values(DType::F32, &[1, channels, 7, 45, 80], &samples)?;
        let out = bench.tensor(DType::F32, &[1, channels, 7, 45, 80])?;
        bench.run(&|f| normalize.forward(f, input, out))?;
        normalized(
            "video_encoder_normalize_conv3d",
            &bench.download(out)?,
            &samples,
            positions as usize,
            mean,
            std,
        )?;

        let (mean, std) = &audio_stats;
        let channels = mean.len() as u64;
        let samples = sample(&mut draw, mean, std, frames as usize);
        let (scale, shift) = normalize_tables(mean, std);
        let normalize = Conv1d::new(
            bench.weight(DType::F32, &[channels, channels, 1], &scale)?,
            Some(bench.weight(DType::F32, &[channels], &shift)?),
            one_by_one(channels, false),
        )?;
        let input = bench.values(DType::F32, &[1, channels, frames], &samples)?;
        let out = bench.tensor(DType::F32, &[1, channels, frames])?;
        bench.run(&|f| normalize.forward(f, input, out))?;
        normalized(
            "audio_encoder_normalize_conv1d",
            &bench.download(out)?,
            &samples,
            frames as usize,
            mean,
            std,
        )?;
        Ok(())
    }

    /// `[channels, channels]` with `values` on the diagonal: the restore's weight.
    fn diagonal(values: &[f32]) -> Vec<f32> {
        let channels = values.len();
        let mut matrix = vec![0.0; channels * channels];
        for (channel, value) in values.iter().enumerate() {
            matrix[channel * channels + channel] = *value;
        }
        matrix
    }

    /// The encoders' diagonal and shift (`minimax-h3` `encoding::normalize_tables`): `1/s` on the
    /// diagonal and `-m/s`, each one f32 operation.
    fn normalize_tables(mean: &[f32], std: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let reciprocals: Vec<f32> = std.iter().map(|deviation| 1.0 / deviation).collect();
        let shift = mean
            .iter()
            .zip(std)
            .map(|(mean, deviation)| -mean / deviation)
            .collect();
        (diagonal(&reciprocals), shift)
    }

    /// A one-by-one convolution of `channels`, as the H3 encoders and decoders build theirs.
    fn one_by_one<const N: usize>(channels: u64, tf32: bool) -> ConvSpec<N> {
        ConvSpec {
            tf32,
            ..ConvSpec::new(channels as u32, channels as u32, [1; N])
        }
    }

    /// A channel-major sample of `positions` values per channel, each channel `N(mean, std²)`.
    fn sample(draw: &mut Draw, mean: &[f32], std: &[f32], positions: usize) -> Vec<f32> {
        mean.iter()
            .zip(std)
            .flat_map(|(mean, deviation)| {
                draw.normals(positions, *deviation)
                    .into_iter()
                    .map(move |value| value + mean)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Reports a device restore against `fma(x, s, m)` and `x·s + m` with two roundings.
    fn restored(
        name: &str,
        device: &[u8],
        latents: &[f32],
        channel: impl Fn(usize) -> usize,
        std: &[f32],
        mean: &[f32],
    ) -> Outcome {
        let fused: Vec<f32> = (0..latents.len())
            .map(|index| latents[index].mul_add(std[channel(index)], mean[channel(index)]))
            .collect();
        let separate: Vec<f32> = (0..latents.len())
            .map(|index| latents[index] * std[channel(index)] + mean[channel(index)])
            .collect();
        measure(
            &format!("channel_affine_{name}_vs_fma"),
            DType::F32,
            device,
            &encode(DType::F32, &fused)?,
        )?;
        measure(
            &format!("channel_affine_{name}_vs_multiply_then_add"),
            DType::F32,
            device,
            &encode(DType::F32, &separate)?,
        )
    }

    /// Reports a device normalization of a channel-major sample against the reference's
    /// `(x - m) / s` and against `fma(x, 1/s, -m/s)` over the code's tables.
    fn normalized(
        name: &str,
        device: &[u8],
        samples: &[f32],
        positions: usize,
        mean: &[f32],
        std: &[f32],
    ) -> Outcome {
        let channel = |index: usize| index / positions;
        let divided: Vec<f32> = (0..samples.len())
            .map(|index| (samples[index] - mean[channel(index)]) / std[channel(index)])
            .collect();
        let fused: Vec<f32> = (0..samples.len())
            .map(|index| {
                let (mean, deviation) = (mean[channel(index)], std[channel(index)]);
                samples[index].mul_add(1.0 / deviation, -mean / deviation)
            })
            .collect();
        measure(
            &format!("channel_affine_{name}_vs_subtract_then_divide"),
            DType::F32,
            device,
            &encode(DType::F32, &divided)?,
        )?;
        measure(
            &format!("channel_affine_{name}_vs_fma_of_reciprocal"),
            DType::F32,
            device,
            &encode(DType::F32, &fused)?,
        )
    }

    // ---- Case 5 -------------------------------------------------------------------------------

    /// Case 5, clips: the video postprocessor's fade of a clip's first five frames against its
    /// predecessor's tail (`minimax-h3` `output::VideoPostprocessor::blend`: three `Axpby`
    /// launches a frame, the ramp `i/5` and its complement rounded to FP16 on the host) against
    /// `CrossFade` over the same frames, read as a band of five rows of one row per frame.
    #[test]
    #[ignore = "requires a CUDA device and a Program Server (UNISERVE_PROGRAM_SERVER_SOCKET)"]
    fn frame_blend() -> Outcome {
        let mut bench = Bench::open()?;
        let mut draw = Draw::new(51);
        let dtype = DType::F16;
        let (channels, overlap, frames, height, width) = (3_u64, 5_u64, 25_u64, 720_u64, 1280_u64);
        let pixels = height * width;

        // A decoded raster lies in [-1, 1].
        let previous = bench.values(
            dtype,
            &[channels, overlap, height, width],
            &draw.uniform((channels * overlap * pixels) as usize, -1.0, 1.0),
        )?;
        let clip = bench.values(
            dtype,
            &[channels, frames, height, width],
            &draw.uniform((channels * frames * pixels) as usize, -1.0, 1.0),
        )?;
        let looped = bench.tensor(dtype, &[channels, overlap, height, width])?;
        let faded = bench.tensor(dtype, &[channels, overlap, height, width])?;
        let term = bench.tensor(dtype, &[1, pixels])?;
        let rounded = |value: f32| f16_to_f32(f32_to_f16(value));

        // `[channels, 1, overlap, pixels]`: the frames as the rows `CrossFade` blends along.
        let band = |view: TensorView, frames: u64| {
            TensorView::with_strides(
                view.ptr,
                dtype,
                Shape::new(&[channels, 1, overlap, pixels]),
                Strides::new(&[frames * pixels, frames * pixels, pixels, 1]),
            )
        };
        let fade = CrossFade::new(dtype, overlap as u32, 2)?;

        bench.run(&|f| {
            for channel in 0..channels {
                for frame in 0..overlap {
                    let weight = rounded(frame as f32 / overlap as f32);
                    let complement = rounded(1.0 - weight);
                    let row = |view: TensorView| {
                        TensorView::new(
                            view.element(&[channel, frame, 0, 0]),
                            dtype,
                            Shape::new(&[1, pixels]),
                        )
                    };
                    Axpby.forward(f, row(looped), row(previous), complement, 0.0)?;
                    Axpby.forward(f, term, row(clip), weight, 0.0)?;
                    Axpby.forward(f, row(looped), term, 1.0, 1.0)?;
                }
            }
            fade.forward(
                f,
                Some(band(previous, overlap)),
                band(clip, frames),
                band(faded, overlap),
            )
        })?;

        measure(
            "blend_frames_scaled_add_loop_vs_cross_fade",
            dtype,
            &bench.download(looped)?,
            &bench.download(faded)?,
        )
    }

    /// Case 5, tiles: `TileStitch` over a 720 x 1280 clip's decode tiles, which fades every pixel
    /// against its neighbours as they were decoded, against the sequential stitch: tiles in raster
    /// order, each blended in place with `CrossFade` against the tile above and then the tile to its
    /// left as those were already blended, then cropped. Corners (in a row and a column seam),
    /// edges (in one) and the interior are reported separately.
    #[test]
    #[ignore = "requires a CUDA device and a Program Server (UNISERVE_PROGRAM_SERVER_SOCKET)"]
    fn tile_stitch_corners() -> Outcome {
        let mut bench = Bench::open()?;
        let mut draw = Draw::new(52);
        let dtype = DType::F16;
        let (channels, height, width) = (3_u32, 720_u32, 1280_u32);
        // Two latent frames' tokens: eight frames of every tile.
        let frames = 2 * PATCH_FRAMES;
        let rows = split(height, TILE_PIXELS, MINIMUM_OVERLAP, SPATIAL);
        let columns = split(width, TILE_PIXELS, MINIMUM_OVERLAP, SPATIAL);
        let (grid_rows, grid_columns) = (rows.starts.len(), columns.starts.len());
        let tiles = grid_rows * grid_columns;
        let side = TILE_PIXELS as usize;
        let tile_elements = channels as usize * frames as usize * side * side;

        // Every tile's decoded raster, `[channels, frames, 256, 256]` in [-1, 1].
        let rasters = draw.uniform(tiles * tile_elements, -1.0, 1.0);

        // The same values as the decoder leaves them: each tile's token rows (voxels, then four
        // register rows and the zero row), each token one patch of `channels · 4 · 16 · 16`.
        let cells = side / SPATIAL as usize;
        let voxels = (frames / PATCH_FRAMES) as usize * cells * cells;
        let token_rows = voxels + 5;
        let patch = channels as usize * PATCH_FRAMES as usize * (SPATIAL * SPATIAL) as usize;
        let mut tokens = vec![0.0_f32; tiles * token_rows * patch];
        let spatial = SPATIAL as usize;
        for tile in 0..tiles {
            for channel in 0..channels as usize {
                for frame in 0..frames as usize {
                    for row in 0..side {
                        for column in 0..side {
                            let (token_frame, inner) = (frame / 4, frame % 4);
                            let token =
                                (token_frame * cells + row / spatial) * cells + column / spatial;
                            let inside = ((channel * 4 + inner) * spatial + row % spatial)
                                * spatial
                                + column % spatial;
                            tokens[(tile * token_rows + token) * patch + inside] = rasters[tile
                                * tile_elements
                                + ((channel * frames as usize + frame) * side + row) * side
                                + column];
                        }
                    }
                }
            }
        }
        let tokens = bench.values(dtype, &[(tiles * token_rows) as u64, patch as u64], &tokens)?;
        let row_table = stitch_axis(&rows, height);
        let column_table = stitch_axis(&columns, width);
        let flatten = |table: &[[i32; 3]]| table.iter().flatten().copied().collect::<Vec<i32>>();
        let row_view = bench.upload(
            DType::I32,
            &[u64::from(height), 3],
            &i32_bytes(&flatten(&row_table)),
        )?;
        let column_view = bench.upload(
            DType::I32,
            &[u64::from(width), 3],
            &i32_bytes(&flatten(&column_table)),
        )?;
        let raster_dims = [
            u64::from(channels),
            u64::from(frames),
            u64::from(height),
            u64::from(width),
        ];
        let stitched = bench.tensor(dtype, &raster_dims)?;
        let tile_dims = [
            u64::from(channels),
            u64::from(frames),
            side as u64,
            side as u64,
        ];
        let blended = bench.values(
            dtype,
            &[&[tiles as u64][..], &tile_dims[..]].concat(),
            &rasters,
        )?;

        let stitch = TileStitch::new(dtype, channels, PATCH_FRAMES, SPATIAL)?;
        let grid = TileGrid {
            columns: grid_columns as u32,
            token_rows: token_rows as u32,
            height: TILE_PIXELS,
            width: TILE_PIXELS,
        };
        // A band of one tile: `extent` rows or columns starting at `start` along `axis`.
        let band = |tile: usize, axis: usize, start: u64, extent: u64| {
            let side = side as u64;
            let base = TensorView::new(
                blended.offset_elements((tile * tile_elements) as u64),
                dtype,
                Shape::new(&tile_dims),
            );
            let (dims, origin) = if axis == 2 {
                ([tile_dims[0], tile_dims[1], extent, side], [0, 0, start, 0])
            } else {
                ([tile_dims[0], tile_dims[1], side, extent], [0, 0, 0, start])
            };
            TensorView::with_strides(
                base.element(&origin),
                dtype,
                Shape::new(&dims),
                Strides::row_major(Shape::new(&tile_dims)),
            )
        };
        let mut fades = BTreeMap::new();
        for overlap in rows.overlaps.iter().chain(&columns.overlaps) {
            for axis in [2_u32, 3] {
                fades.insert((*overlap, axis), CrossFade::new(dtype, *overlap, axis)?);
            }
        }

        bench.run(&|f| {
            stitch.forward(f, tokens, row_view, column_view, grid, stitched)?;
            for row in 0..grid_rows {
                for column in 0..grid_columns {
                    let tile = row * grid_columns + column;
                    let side = side as u64;
                    if row > 0 {
                        let extent = rows.overlaps[row - 1];
                        let current = band(tile, 2, 0, u64::from(extent));
                        let above = band(
                            tile - grid_columns,
                            2,
                            side - u64::from(extent),
                            u64::from(extent),
                        );
                        fades[&(extent, 2)].forward(f, Some(above), current, current)?;
                    }
                    if column > 0 {
                        let extent = columns.overlaps[column - 1];
                        let current = band(tile, 3, 0, u64::from(extent));
                        let left = band(tile - 1, 3, side - u64::from(extent), u64::from(extent));
                        fades[&(extent, 3)].forward(f, Some(left), current, current)?;
                    }
                }
            }
            Ok(())
        })?;

        // The sequential stitch's crop: each pixel from the tile the table says keeps it.
        let blended = bench.download(blended)?;
        let stitched = bench.download(stitched)?;
        let element = 2;
        let mut parts: [(Vec<u8>, Vec<u8>); 3] = Default::default();
        for channel in 0..channels as usize {
            for frame in 0..frames as usize {
                for (y, &[tile_row, inside_row, row_seam]) in row_table.iter().enumerate() {
                    for (x, &[tile_column, inside_column, column_seam]) in
                        column_table.iter().enumerate()
                    {
                        let tile = tile_row as usize * grid_columns + tile_column as usize;
                        let source = tile * tile_elements
                            + ((channel * frames as usize + frame) * side + inside_row as usize)
                                * side
                            + inside_column as usize;
                        let target = ((channel * frames as usize + frame) * height as usize + y)
                            * width as usize
                            + x;
                        let part = usize::from(row_seam > 0) + usize::from(column_seam > 0);
                        // 2: corner, 1: edge, 0: interior.
                        let (left, right) = &mut parts[2 - part];
                        left.extend_from_slice(&stitched[target * element..(target + 1) * element]);
                        right.extend_from_slice(&blended[source * element..(source + 1) * element]);
                    }
                }
            }
        }
        for ((left, right), region) in parts.iter().zip(["corners", "edges", "interior"]) {
            measure(
                &format!("tile_stitch_vs_sequential_blend_{region}"),
                dtype,
                left,
                right,
            )?;
        }
        Ok(())
    }

    // ---- Case 6 -------------------------------------------------------------------------------

    /// Case 6: the video decoder's post-quant bias, added today as an FP16 product with the
    /// identity matrix whose epilogue adds the checkpoint's bias, against `bias_add` launched
    /// through `ops` with the descriptor the decoder's 24 latent channels need. The input is the
    /// post-quant product of the 28 tiles of a 720 x 1280 clip, rounded to FP16.
    #[test]
    #[ignore = "requires a CUDA device and a Program Server (UNISERVE_PROGRAM_SERVER_SOCKET)"]
    fn bias_after_identity_gemm() -> Outcome {
        let directory = models().join(H3).join("vae");
        let bias = tensor(&directory, "post_quant_conv.bias")?;
        let mut bench = Bench::open()?;
        let mut draw = Draw::new(61);
        let dtype = DType::F16;
        let channels = bias.len() as u64;
        let voxels = 28 * 7 * 16 * 16_u64;

        let identity = diagonal(&vec![1.0; channels as usize]);
        let bias = bench.weight(dtype, &[channels], &bias)?;
        let gemm = Linear::new(
            bench.weight(dtype, &[channels, channels], &identity)?,
            Some(bias),
            WeightOrder::NxK,
        )?;
        let input = bench.values(
            dtype,
            &[voxels, channels],
            &draw.normals((voxels * channels) as usize, 1.0),
        )?;
        let multiplied = bench.tensor(dtype, &[voxels, channels])?;
        let added = bench.tensor(dtype, &[voxels, channels])?;
        // `[voxels, channels, 1]`: the rows as a batch of channel-first values of one position.
        let channel_first = |view: TensorView| view.reshape(Shape::new(&[voxels, channels, 1]));
        let descriptor = ops::bias::Descriptor {
            dtype,
            channels: channels as u32,
            block: 1024,
            residual: false,
            residual_bias: false,
            pdl: nn::layers::launch::PDL,
        };

        bench.run(&|f| {
            gemm.forward(f, input, multiplied)?;
            f.operate::<ops::bias::BiasAdd>(
                &descriptor,
                ops::bias::Args {
                    values: channel_first(input),
                    bias: bias.view,
                    // An absent residual carries the values and the bias (`ops::bias::Args`).
                    residual: channel_first(input),
                    residual_bias: bias.view,
                    out: channel_first(added),
                },
            )
        })?;

        measure(
            "bias_identity_gemm_vs_bias_add_video_decoder_post_quant",
            dtype,
            &bench.download(multiplied)?,
            &bench.download(added)?,
        )
    }
}
