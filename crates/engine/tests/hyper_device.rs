//! The kernels of the hyper-connected residual (mHC) and of the hashed n-gram memory on a device,
//! against their CPU formulas: each `nn` function computed on GPU 0 with the kernels the Build
//! Server builds for its keys, and on the CPU from the same bytes, at the shapes of
//! DeepSeek-V4.1-Flash (four streams of 5120, 1 to 16384 tokens), within the operator's own
//! bound (`ops::Op::tolerance`); the boundary also over the released checkpoint's connections of
//! one layer.
//!
//! Every case needs GPU 0 and this host's Build Server (`uniserve build-server serve
//! --deployment device-cases`). The CPU formulas of the largest calls are long in an unoptimized
//! build, so run them optimized:
//!
//! ```text
//! cargo test --release -p engine --test hyper_device -- --ignored --test-threads 1
//! ```

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

mod device;

use std::collections::BTreeSet;
use std::sync::Arc;

use nn::functional::{self, MhcMixes, MhcMixing};
use ops::tolerance::Tolerance;
use ops::{Context, Generation, Op};
use platform::{DeviceAllocation, DeviceRange, Stream};
use tensor::{DType, Device, Tensor};

use device::{DEVICE, Kernels, Outcome};

/// The model's residual: four streams of 5120 elements, twenty Sinkhorn iterations.
const STREAMS: usize = 4;
const HIDDEN: usize = 5120;
const MIXES: usize = STREAMS * (STREAMS + 2);

/// The token counts of a decode, a verification, the boundary between the two token blocks a
/// boundary program holds, and prefill chunks up to the largest.
const TOKENS: [usize; 7] = [1, 16, 64, 512, 513, 4096, 16384];

/// The bytes of the region the device calls take their outputs from.
const REGION: usize = 8 << 30;

/// A fixed sequence of `count` values in `(-1, 1)` for `seed` (Numerical Recipes' generator).
fn sequence(count: usize, seed: u32) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32.wrapping_mul(seed.wrapping_add(1));
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 23) as f32 - 1.0
        })
        .collect()
}

/// One input, as its bytes on both devices.
struct Input {
    bytes: Vec<u8>,
    dims: Vec<usize>,
    dtype: DType,
}

impl Input {
    fn floats(values: &[f32], dims: &[usize], dtype: DType) -> Self {
        Self {
            bytes: ops::values::encode(dtype, values).unwrap(),
            dims: dims.to_vec(),
            dtype,
        }
    }

    fn raw(bytes: Vec<u8>, dims: &[usize], dtype: DType) -> Self {
        Self {
            bytes,
            dims: dims.to_vec(),
            dtype,
        }
    }

    /// The host tensor of the bytes.
    fn host(&self) -> Tensor {
        Tensor::from_raw_buffer(&self.bytes, self.dtype, &self.dims, &Device::Cpu).unwrap()
    }

    /// The device tensor of the bytes, over an allocation the caller keeps.
    fn device(&self, stream: &Stream) -> Outcome<(DeviceAllocation, Tensor)> {
        if self.dtype == DType::F32 {
            let values = ops::values::decode(DType::F32, &self.bytes)?;
            return device::weight(stream, &values, &self.dims);
        }
        let allocation = DeviceAllocation::linear(DEVICE, self.bytes.len().max(1))?;
        // SAFETY: the destination is the allocation just made for these bytes, the source this
        // input's vector, and the stream is waited for before the vector can go.
        unsafe {
            platform::memcpy_async(
                allocation.address(),
                self.bytes.as_ptr() as u64,
                self.bytes.len(),
                stream,
            )?;
        }
        stream.synchronize()?;
        let tensor = ops::retained(
            allocation.address(),
            self.dims.as_slice(),
            self.dtype,
            &ops::cuda(DEVICE.0 as usize),
        )?;
        Ok((allocation, tensor))
    }
}

/// `tensor`, a contiguous device tensor, copied to the host once the stream is done.
fn download(tensor: &Tensor, stream: &Stream) -> Outcome<Tensor> {
    stream.synchronize()?;
    let bytes = tensor.dtype().storage_bytes(tensor.elem_count()).unwrap();
    let mut host = vec![0_u8; bytes];
    // SAFETY: the source is the tensor's storage, contiguous, and the destination holds its
    // bytes; the stream is waited for before the vector is read.
    unsafe {
        platform::memcpy_async(
            host.as_mut_ptr() as u64,
            ops::operand(tensor)?.ptr.address(),
            bytes,
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(Tensor::from_raw_buffer(
        &host,
        tensor.dtype(),
        tensor.dims(),
        &Device::Cpu,
    )?)
}

/// The elements of `tensor` as FP32 (integers exactly below 2²⁴, as the rows compared are).
fn elements(tensor: &Tensor) -> Vec<f64> {
    if tensor.dtype() == DType::I64 {
        return tensor
            .flatten_all()
            .unwrap()
            .to_vec1::<i64>()
            .unwrap()
            .into_iter()
            .map(|value| value as f64)
            .collect();
    }
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .into_iter()
        .map(f64::from)
        .collect()
}

/// Asserts `device` within `tolerance` of `host` element by element, and prints the largest
/// difference.
fn assert_within(what: &str, device: &Tensor, host: &Tensor, tolerance: Tolerance) {
    let (device, host) = (elements(device), elements(host));
    assert_eq!(
        device.len(),
        host.len(),
        "{what}: the element counts differ"
    );
    let peak = host.iter().fold(0_f64, |peak, value| peak.max(value.abs())) as f32;
    let mut worst = 0_f64;
    for (index, (&got, &expected)) in device.iter().zip(&host).enumerate() {
        assert!(
            tolerance.allows_within(peak, expected as f32, got as f32),
            "{what}: element {index} is {got} on the device and {expected} on the host (peak \
             {peak}), beyond {tolerance:?}"
        );
        worst = worst.max((got - expected).abs());
    }
    eprintln!("{what}: largest difference {worst:.3e} (peak {peak:.3e}), within {tolerance:?}");
}

/// The keys `forward` launches, from its profile run on the meta device.
fn keys<T>(forward: impl FnOnce() -> nn::Result<T>) -> BTreeSet<(&'static str, String)> {
    let context = Context::meta(None);
    context.record_launches();
    {
        let _guard = context.enter();
        forward().unwrap();
    }
    context
        .take_launches()
        .into_iter()
        .map(|(operator, key, _, _)| (operator, key))
        .collect()
}

/// A launching context on GPU 0 over `kernels`, with the region its outputs take, which the
/// caller keeps.
fn context(stream: &Stream, kernels: Arc<Generation>) -> Outcome<(DeviceAllocation, Context)> {
    let region = DeviceAllocation::linear(DEVICE, REGION)?;
    let range = DeviceRange::new(
        platform::DevicePtr::from_address(region.address()),
        region.len(),
    );
    Ok((
        region,
        Context::new(DEVICE.0 as usize, stream.clone(), kernels, range),
    ))
}

/// A meta tensor of `dims`.
fn meta(dims: &[usize], dtype: DType) -> Tensor {
    ops::meta_tensor(dims, dtype).unwrap()
}

/// A connection's parameters at the checkpoint's magnitudes, and a pending output's.
struct Boundary {
    streams: Input,
    output: Input,
    pre: Input,
    post: Input,
    comb: Input,
    projection: Input,
    scale: Input,
    base: Input,
    norm: Input,
}

impl Boundary {
    fn new(tokens: usize, seed: u32) -> Self {
        let scaled = |count, seed, scale: f32, shift: f32| -> Vec<f32> {
            sequence(count, seed)
                .into_iter()
                .map(|x| shift + scale * x)
                .collect()
        };
        // A doubly stochastic matrix near the ones the checkpoint's connections compute.
        let comb: Vec<f32> = (0..tokens)
            .flat_map(|_| {
                (0..STREAMS * STREAMS).map(|index| {
                    if index / STREAMS == index % STREAMS {
                        0.85
                    } else {
                        0.05
                    }
                })
            })
            .collect();
        Self {
            streams: Input::floats(
                &scaled(tokens * STREAMS * HIDDEN, seed, 2.0, 0.0),
                &[tokens, STREAMS, HIDDEN],
                DType::BF16,
            ),
            output: Input::floats(
                &scaled(tokens * HIDDEN, seed + 1, 1.0, 0.0),
                &[tokens, HIDDEN],
                DType::BF16,
            ),
            pre: Input::floats(
                &scaled(tokens * STREAMS, seed + 2, 0.5, 0.5),
                &[tokens, STREAMS],
                DType::F32,
            ),
            post: Input::floats(
                &scaled(tokens * STREAMS, seed + 3, 1.0, 1.0),
                &[tokens, STREAMS],
                DType::F32,
            ),
            comb: Input::floats(&comb, &[tokens, STREAMS, STREAMS], DType::F32),
            projection: Input::floats(
                &scaled(MIXES * STREAMS * HIDDEN, seed + 4, 0.3, 0.0),
                &[MIXES, STREAMS * HIDDEN],
                DType::F32,
            ),
            scale: Input::floats(&[0.07, 0.02, 0.17], &[3], DType::F32),
            base: Input::floats(&scaled(MIXES, seed + 5, 3.0, 0.0), &[MIXES], DType::F32),
            norm: Input::floats(
                &scaled(HIDDEN, seed + 6, 0.01, 0.03),
                &[HIDDEN],
                DType::BF16,
            ),
        }
    }
}

/// The boundary's call over `tensors` (streams, output, pre, post, comb, projection, scale, base,
/// norm), with the pending output written back when `post`.
fn boundary(tensors: &[Tensor], post: bool) -> nn::Result<(Tensor, Tensor, MhcMixes)> {
    let pending = MhcMixes {
        pre: tensors[2].clone(),
        post: tensors[3].clone(),
        comb: tensors[4].clone(),
    };
    let mixing = MhcMixing {
        projection: &tensors[5],
        scale: &tensors[6],
        base: &tensors[7],
        iterations: 20,
        rms_eps: 1e-20,
        eps: 1e-6,
    };
    functional::mhc_boundary(
        &tensors[0],
        post.then_some((&tensors[1], &pending)),
        &tensors[2],
        mixing,
        &tensors[8],
        1e-20,
    )
}

/// Computes the boundary over `inputs` on the device with the kernels the Build Server selects
/// for its keys and on the CPU, and holds every output to the operator's bound.
fn assert_boundary(
    kernels: &mut Kernels,
    stream: &Stream,
    inputs: &Boundary,
    tokens: usize,
    post: bool,
    what: &str,
) -> Outcome {
    let all = [
        &inputs.streams,
        &inputs.output,
        &inputs.pre,
        &inputs.post,
        &inputs.comb,
        &inputs.projection,
        &inputs.scale,
        &inputs.base,
        &inputs.norm,
    ];
    let metas: Vec<Tensor> = all
        .iter()
        .map(|input| meta(&input.dims, input.dtype))
        .collect();
    let generation = kernels.generation(1, &keys(|| boundary(&metas, post)))?;

    let host: Vec<Tensor> = all.iter().map(|input| input.host()).collect();
    let expected = boundary(&host, post)?;

    let placed: Vec<(DeviceAllocation, Tensor)> = all
        .iter()
        .map(|input| input.device(stream))
        .collect::<Outcome<_>>()?;
    let tensors: Vec<Tensor> = placed.iter().map(|(_, tensor)| tensor.clone()).collect();
    let (_region, context) = context(stream, generation)?;
    let (reached, input, mixes) = {
        let _guard = context.enter();
        boundary(&tensors, post)?
    };

    let tolerance = ops::mhc::MhcBoundary::tolerance(&ops::mhc::BoundaryDescriptor::new(
        DType::BF16,
        STREAMS as u32,
        HIDDEN as u32,
        20,
        post,
        tokens as u64,
    ));
    let case = |output: &str| format!("{what}, {tokens} tokens, post {post}: {output}");
    if post {
        assert_within(
            &case("the written streams"),
            &download(&reached, stream)?,
            &expected.0,
            tolerance,
        );
    }
    assert_within(
        &case("the input"),
        &download(&input, stream)?,
        &expected.1,
        tolerance,
    );
    assert_within(
        &case("the pre-mix"),
        &download(&mixes.pre, stream)?,
        &expected.2.pre,
        tolerance,
    );
    assert_within(
        &case("the post-mix"),
        &download(&mixes.post, stream)?,
        &expected.2.post,
        tolerance,
    );
    assert_within(
        &case("the combination"),
        &download(&mixes.comb, stream)?,
        &expected.2.comb,
        tolerance,
    );
    Ok(())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn the_boundary_kernel_computes_its_cpu_formula() -> Outcome {
    let stream = Stream::new(DEVICE)?;
    let mut kernels = Kernels::open()?;

    for (index, &tokens) in TOKENS.iter().enumerate() {
        for post in [false, true] {
            let inputs = Boundary::new(tokens, 10 * index as u32);
            assert_boundary(&mut kernels, &stream, &inputs, tokens, post, "synthetic")?;
        }
    }
    Ok(())
}

/// The released checkpoint's connections of layer 3 (its attention's and its FFN's projection,
/// scales, biases and input norm) over synthetic streams: the mixes are computed from the
/// projection's own magnitudes, which bound what reading it in TF32 costs.
#[test]
#[ignore = "requires a GPU, a Build Server and the released checkpoint"]
fn the_boundary_kernel_computes_its_cpu_formula_over_released_connections() -> Outcome {
    let stream = Stream::new(DEVICE)?;
    let mut kernels = Kernels::open()?;
    let checkpoint = testkit::assets::DEEPSEEK_V41_FLASH.checkpoint();
    let read = |name: &str, dims: &[usize], dtype: DType| {
        Input::raw(checkpoint.read(name).unwrap(), dims, dtype)
    };

    for (connection, norm) in [("hc_attn", "attn_norm"), ("hc_ffn", "ffn_norm")] {
        for (index, &tokens) in [1, 6, 64, 513, 4096].iter().enumerate() {
            let inputs = Boundary {
                projection: read(
                    &format!("layers.3.{connection}_fn"),
                    &[MIXES, STREAMS * HIDDEN],
                    DType::F32,
                ),
                scale: read(&format!("layers.3.{connection}_scale"), &[3], DType::F32),
                base: read(&format!("layers.3.{connection}_base"), &[MIXES], DType::F32),
                norm: read(&format!("layers.3.{norm}.weight"), &[HIDDEN], DType::BF16),
                ..Boundary::new(tokens, 70 + 10 * index as u32)
            };
            assert_boundary(&mut kernels, &stream, &inputs, tokens, true, connection)?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn the_write_and_read_kernels_compute_their_cpu_formulas() -> Outcome {
    let stream = Stream::new(DEVICE)?;
    let mut kernels = Kernels::open()?;

    for (index, &tokens) in TOKENS.iter().enumerate() {
        let inputs = Boundary::new(tokens, 100 + 10 * index as u32);
        let all = [
            &inputs.streams,
            &inputs.output,
            &inputs.pre,
            &inputs.post,
            &inputs.comb,
            &inputs.norm,
        ];
        let call = |tensors: &[Tensor]| -> nn::Result<(Tensor, Tensor)> {
            let mixes = MhcMixes {
                pre: tensors[2].clone(),
                post: tensors[3].clone(),
                comb: tensors[4].clone(),
            };
            Ok((
                functional::mhc_post(&tensors[0], &tensors[1], &mixes)?,
                functional::mhc_pre(&tensors[0], &tensors[2], &tensors[5], 1e-20)?,
            ))
        };
        let metas: Vec<Tensor> = all
            .iter()
            .map(|input| meta(&input.dims, input.dtype))
            .collect();
        let generation = kernels.generation(1, &keys(|| call(&metas)))?;
        let host: Vec<Tensor> = all.iter().map(|input| input.host()).collect();
        let (written, read) = call(&host)?;

        let placed: Vec<(DeviceAllocation, Tensor)> = all
            .iter()
            .map(|input| input.device(&stream))
            .collect::<Outcome<_>>()?;
        let tensors: Vec<Tensor> = placed.iter().map(|(_, tensor)| tensor.clone()).collect();
        let (_region, context) = context(&stream, generation)?;
        let (device_written, device_read) = {
            let _guard = context.enter();
            call(&tensors)?
        };

        let post = ops::mhc::MhcPost::tolerance(&ops::mhc::PostDescriptor::new(
            DType::BF16,
            4,
            HIDDEN as u32,
        ));
        let pre = ops::mhc::MhcPre::tolerance(&ops::mhc::PreDescriptor::new(
            DType::BF16,
            4,
            HIDDEN as u32,
        ));
        assert_within(
            &format!("{tokens} tokens: the written streams"),
            &download(&device_written, &stream)?,
            &written,
            post,
        );
        assert_within(
            &format!("{tokens} tokens: the read row"),
            &download(&device_read, &stream)?,
            &read,
            pre,
        );
    }
    Ok(())
}

/// The table rows of the n-gram memory's calls: `(normalize, multipliers, moduli, offsets)` of
/// the model's shape (contexts of four tokens, eight heads of each length, a vocabulary of
/// 129,280 normalized into 99,092, primes above sixteen million).
fn hashing() -> [Input; 4] {
    let normalize: Vec<u8> = (0..129_280_i32)
        .flat_map(|id| (id * 7919 % 99_092).to_le_bytes())
        .collect();
    let multipliers: Vec<u8> = [
        76_632_096_046_245_i64,
        4_839_876_093_313,
        35_959_672_319_349,
        73_987_337_458_391,
    ]
    .iter()
    .flat_map(|value| value.to_le_bytes())
    .collect();
    let moduli: Vec<i64> = (0..24).map(|index| 16_000_057 + 22 * index).collect();
    let offsets: Vec<i64> = moduli
        .iter()
        .scan(0, |next, modulus| {
            let first = *next;
            *next += modulus;
            Some(first)
        })
        .collect();
    let bytes = |values: &[i64]| {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    };
    [
        Input::raw(normalize, &[129_280], DType::I32),
        Input::raw(multipliers, &[4], DType::I64),
        Input::raw(bytes(&moduli), &[3, 8], DType::I64),
        Input::raw(bytes(&offsets), &[3, 8], DType::I64),
    ]
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn the_ngram_kernels_compute_their_cpu_formulas() -> Outcome {
    let stream = Stream::new(DEVICE)?;
    let mut kernels = Kernels::open()?;
    let [normalize, multipliers, moduli, offsets] = hashing();

    // A table shard of a million rows that opens at row 1000, read by ids inside and outside it.
    const ROWS: usize = 1 << 20;
    const FIRST: u64 = 1000;
    let codes: Vec<u8> = sequence(ROWS * 256, 7)
        .into_iter()
        .map(|x| {
            let byte = ((x + 1.0) * 127.5) as u8;
            if byte & 0x7F == 0x7F { byte - 1 } else { byte }
        })
        .collect();
    let exponents: Vec<u8> = sequence(ROWS * 8, 8)
        .into_iter()
        .map(|x| (120.0 + 8.0 * x) as u8)
        .collect();
    let values = Input::raw(codes, &[ROWS, 256], DType::F8E4M3);
    let scales = Input::raw(exponents, &[ROWS, 8], DType::U8);

    for (index, &tokens) in TOKENS.iter().enumerate() {
        let seed = 200 + 10 * index as u32;
        let history: Vec<u8> = sequence(tokens * 4, seed)
            .into_iter()
            .flat_map(|x| (((x + 1.0) * 64_640.0) as u32).to_le_bytes())
            .collect();
        let history = Input::raw(history, &[tokens, 4], DType::U32);
        let ids: Vec<u8> = sequence(tokens * 24, seed + 1)
            .into_iter()
            .flat_map(|x| (((x + 1.0) * (ROWS as f32 / 2.0 + 600.0)) as i64).to_le_bytes())
            .collect();
        let ids = Input::raw(ids, &[tokens * 24], DType::I64);
        let streams = Input::floats(
            &sequence(tokens * STREAMS * HIDDEN, seed + 2),
            &[tokens, STREAMS, HIDDEN],
            DType::BF16,
        );
        let projected = Input::floats(
            &sequence(tokens * (STREAMS + 1) * HIDDEN, seed + 3),
            &[tokens, (STREAMS + 1) * HIDDEN],
            DType::BF16,
        );
        let gains = |seed| -> Vec<f32> {
            sequence(STREAMS * HIDDEN, seed)
                .into_iter()
                .map(|x| 1.0 + 0.3 * x)
                .collect()
        };
        let query = Input::floats(&gains(seed + 4), &[STREAMS, HIDDEN], DType::BF16);
        let key = Input::floats(&gains(seed + 5), &[STREAMS, HIDDEN], DType::BF16);

        let all = [
            &history,
            &normalize,
            &multipliers,
            &moduli,
            &offsets,
            &ids,
            &values,
            &scales,
            &streams,
            &projected,
            &query,
            &key,
        ];
        let call = |t: &[Tensor]| -> nn::Result<(Tensor, Tensor, Tensor)> {
            Ok((
                functional::ngram_hash(&t[0], &t[1], &t[2], &t[3], &t[4])?,
                functional::embedding_mxfp8(&t[5], &t[6], &t[7], FIRST, DType::BF16)?,
                functional::ngram_gate(&t[8], &t[9], &t[10], &t[11], 1e-20)?,
            ))
        };
        let metas: Vec<Tensor> = all
            .iter()
            .map(|input| meta(&input.dims, input.dtype))
            .collect();
        let generation = kernels.generation(1, &keys(|| call(&metas)))?;
        let host: Vec<Tensor> = all.iter().map(|input| input.host()).collect();
        let (rows, recalled, gated) = call(&host)?;

        let placed: Vec<(DeviceAllocation, Tensor)> = all
            .iter()
            .map(|input| input.device(&stream))
            .collect::<Outcome<_>>()?;
        let tensors: Vec<Tensor> = placed.iter().map(|(_, tensor)| tensor.clone()).collect();
        let (_region, context) = context(&stream, generation)?;
        let (device_rows, device_recalled, device_gated) = {
            let _guard = context.enter();
            call(&tensors)?
        };

        let exact = Tolerance::exact();
        let gate = ops::ngram::NgramGate::tolerance(&ops::ngram::GateDescriptor::new(
            DType::BF16,
            4,
            HIDDEN as u32,
        ));
        assert_within(
            &format!("{tokens} tokens: the rows"),
            &download(&device_rows, &stream)?,
            &rows,
            exact,
        );
        assert_within(
            &format!("{tokens} tokens: the recalled rows"),
            &download(&device_recalled, &stream)?,
            &recalled,
            exact,
        );
        assert_within(
            &format!("{tokens} tokens: the gated streams"),
            &download(&device_gated, &stream)?,
            &gated,
            gate,
        );
    }
    Ok(())
}
