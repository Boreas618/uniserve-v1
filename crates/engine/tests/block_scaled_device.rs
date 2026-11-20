//! A block-scaled release on a device: DeepSeek-V4.1-Flash's layer-3 projections, its grouped
//! output projection and one of its routed experts, loaded through `nn::WeightLoader` over a GPU's weight store as a rank loads
//! them and computed by the MXFP8 and MXFP4 kernels, against the products of the checkpoint's
//! values dequantized on the host.
//!
//! The checkpoint stores its dense weights as E4M3 codes under one UE8M0 scale per 32 × 32 block
//! and its routed experts as packed E2M1 codes under one UE8M0 scale per 32 elements of a row;
//! its activations are encoded per 32 elements into E4M3 under a UE8M0 scale at run time. Each
//! case decodes the stored bytes itself, encodes the same BF16 rows by that rule, and holds the
//! device's output to the exact product of the two in FP64 under the operator's bound: the
//! products' FP32 accumulation and their output's rounding, and, where the device encodes an
//! intermediate, half a code step of that encoding under the rule the operator states.
//!
//! Every case needs GPU 0, this host's Build Server (`uniserve build-server serve --deployment
//! device-cases`) and the release in the Hugging Face cache ([`DEEPSEEK_V41_FLASH`]). Run them
//! on a GPU host:
//!
//! ```text
//! cargo test -p engine --test block_scaled_device -- --ignored --test-threads 1
//! ```

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

#[expect(
    dead_code,
    reason = "the cases place their inputs by their bytes, not as FP32 values"
)]
mod device;

use std::collections::BTreeSet;
use std::sync::Arc;

use engine::memory::client::{StorageClient, StorageLaunch};
use engine::memory::weights::WeightStore;
use nn::functional;
use nn::modules::{GroupedLinear, Linear};
use nn::quantization::Declaration;
use nn::{Checkpoint, MetaWeights, RetainedWeights, WeightLoader};
use ops::Op;
use ops::moe::mxfp4::{
    GatherGemmMxfp4Descriptor, RouteGemmMxfp4, RouteGemmMxfp4Descriptor, WEIGHT_SCALE_LAYOUT,
};
use ops::mx::{GemmDescriptor, Mxfp8Gemm};
use platform::{DeviceAllocation, DevicePtr, DeviceRange, Stream};
use storage::{Claimant, Client, Role, Scope};
use tensor::{DType, Tensor};
use testkit::StorageServer;
use testkit::assets::DEEPSEEK_V41_FLASH;

use device::{DEVICE, Kernels, Outcome};

/// The release's model width, its query and key-value compressions, and its experts' width.
const HIDDEN: usize = 5120;
const QUERY_RANK: usize = 1280;
const KEY_VALUE: usize = 512;
const INTERMEDIATE: usize = 2304;

/// The release's grouped output projection `wo_a`: eight groups, each of the outputs of eight
/// heads of 512 (4096) to a rank of 1024.
const OUTPUT_GROUPS: usize = 8;
const GROUP_WIDTH: usize = 4096;
const OUTPUT_RANK: usize = 1024;

/// The layer the cases read, and the routed expert of it.
const LAYER: &str = "layers.3";
const EXPERT: u32 = 7;

/// The release's SwiGLU clamp (`swiglu_limit`).
const LIMIT: f64 = 10.0;

/// The token rows each case computes.
const TOKENS: usize = 5;

/// The deployment, worker group and epoch of the cases' rank at its Storage Server.
const DEPLOYMENT: &str = "block-scaled";
const GROUP: &str = "text";
const EPOCH: u64 = 1;

/// The value of a UE8M0 scale byte: `2^(byte − 127)`.
fn ue8m0(byte: u8) -> f64 {
    2f64.powi(i32::from(byte) - 127)
}

/// The value of an E4M3 byte: a sign, four exponent bits biased by seven and three mantissa bits,
/// subnormal below the first exponent.
fn e4m3(byte: u8) -> f64 {
    let (exponent, mantissa) = (i32::from((byte >> 3) & 0x0f), f64::from(byte & 0x07));
    let magnitude = if exponent == 0 {
        mantissa / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + mantissa / 8.0) * 2f64.powi(exponent - 7)
    };
    if byte & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// The value of an E2M1 code, the low four bits of `code`: a sign and the magnitudes
/// `0, 0.5, 1, 1.5, 2, 3, 4, 6`.
fn e2m1(code: u8) -> f64 {
    const MAGNITUDES: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let magnitude = MAGNITUDES[usize::from(code & 0x07)];
    if code & 0x08 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

/// The nearest E4M3 value to `x` within ±448, a tie to the even code.
fn nearest_e4m3(x: f64) -> f64 {
    let magnitude = x.abs().min(448.0);
    let (mut best, mut best_code) = (0.0, 0_u8);
    for code in 0..=0x7e_u8 {
        let value = e4m3(code);
        let (distance, best_distance) = ((value - magnitude).abs(), (best - magnitude).abs());
        if distance < best_distance || (distance == best_distance && code % 2 == 0) {
            (best, best_code) = (value, code);
        }
    }
    debug_assert!(best_code <= 0x7e);
    best.copysign(x)
}

/// The values `rows` BF16 rows state after their MXFP8 encoding: per 32 elements the least power
/// of two at or above `fl(max(amax, 10⁻⁴) · fl(1/448))`, evaluated in FP32, scales the E4M3
/// codes of the quotients.
fn encoded(rows: &[f32]) -> Vec<f64> {
    rows.chunks(32)
        .flat_map(|block| {
            let amax = block.iter().fold(1e-4_f32, |amax, x| amax.max(x.abs()));
            let unit = amax * (1.0_f32 / 448.0);
            // The exponent of the power of two at or above `unit`: its own exponent, one more
            // when any mantissa bit is set.
            let bits = unit.to_bits();
            let exponent = ((bits >> 23) & 0xff) as i32 - 127 + i32::from(bits & 0x7f_ffff != 0);
            let scale = 2f64.powi(exponent);
            block
                .iter()
                .map(move |x| nearest_e4m3(f64::from(*x) / scale) * scale)
        })
        .collect()
}

/// `count` BF16 values of a normalized hidden state's spread, as FP32 and as the BF16 bytes a
/// device reads: a fixed sequence of sums of four uniform draws, rounded to the nearest BF16.
fn hidden_rows(count: usize, seed: u64) -> (Vec<f32>, Vec<u8>) {
    let mut state = seed;
    let mut uniform = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 40) as f32 / (1_u64 << 24) as f32 - 0.5
    };
    let values: Vec<f32> = (0..count)
        .map(|_| {
            let sum = uniform() + uniform() + uniform() + uniform();
            let bits = (sum * 1.7).to_bits();
            // Round to nearest even on the sixteen bits BF16 drops.
            let rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
            f32::from_bits(rounded)
        })
        .collect();
    let bytes = values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    (values, bytes)
}

/// The values of the checkpoint's E4M3 weight `name` under its 32 × 32 block scales, row-major.
fn fp8_weight(checkpoint: &Checkpoint, name: &str, rows: usize, columns: usize) -> Vec<f64> {
    let codes = checkpoint.read(&format!("{name}.weight")).unwrap();
    let scales = checkpoint.read(&format!("{name}.scale")).unwrap();
    assert_eq!(
        (codes.len(), scales.len()),
        (rows * columns, rows.div_ceil(32) * columns.div_ceil(32)),
        "{name}"
    );
    (0..rows * columns)
        .map(|index| {
            let (row, column) = (index / columns, index % columns);
            e4m3(codes[index]) * ue8m0(scales[(row / 32) * columns.div_ceil(32) + column / 32])
        })
        .collect()
}

/// The values of the checkpoint's MXFP4 weight `name`: two E2M1 codes a byte, the low nibble
/// first, under one scale per 32 elements of a row; row-major.
fn fp4_weight(checkpoint: &Checkpoint, name: &str, rows: usize, columns: usize) -> Vec<f64> {
    let packed = checkpoint.read(&format!("{name}.weight")).unwrap();
    let scales = checkpoint.read(&format!("{name}.scale")).unwrap();
    assert_eq!(
        (packed.len(), scales.len()),
        (rows * columns / 2, rows * columns / 32),
        "{name}"
    );
    (0..rows * columns)
        .map(|element| {
            let byte = packed[element / 2];
            let code = if element % 2 == 0 { byte } else { byte >> 4 };
            e2m1(code) * ue8m0(scales[element / 32])
        })
        .collect()
}

/// `matrix [rows, columns] · x` and `|matrix| · |x|`, accumulated in FP64.
fn products(matrix: &[f64], columns: usize, x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    matrix
        .chunks(columns)
        .map(|row| {
            row.iter().zip(x).fold((0.0, 0.0), |(sum, terms), (w, x)| {
                (sum + w * x, terms + (w * x).abs())
            })
        })
        .unzip()
}

/// The release, opened by its shard set.
fn release() -> Checkpoint {
    DEEPSEEK_V41_FLASH.checkpoint()
}

/// A loader over `store` that reads the release's quantization config.
fn loader<'a>(checkpoint: &'a Checkpoint, store: &'a dyn RetainedWeights) -> WeightLoader<'a> {
    let mut loader = WeightLoader::new(checkpoint, store, None);
    loader
        .declare(&Declaration::read(checkpoint).unwrap())
        .unwrap();
    loader
}

/// Rank 0 of the cases' worker group at `server`, connected under the claim of its deployment's
/// head; the head's connection is returned with it, because the claim lasts while it is open.
fn rank(server: &StorageServer) -> (Client, Arc<StorageClient>) {
    let scope = Scope {
        deployment: DEPLOYMENT.into(),
        group: GROUP.into(),
        rank: 0,
    };
    let mut head = Client::connect(
        &server.socket,
        Role::Head {
            deployment: DEPLOYMENT.into(),
            epoch: EPOCH,
            generation: 0,
        },
    )
    .unwrap();
    let claim = head
        .claim(Claimant::Head, EPOCH, vec![scope], Vec::new())
        .unwrap();

    let launch = StorageLaunch {
        socket: server.socket.to_string_lossy().into_owned(),
        deployment: DEPLOYMENT.into(),
        claim: claim.0,
        tcp: None,
        peers: Vec::new(),
        epoch: EPOCH,
        standby: false,
        standby_storage: proto::request::StandbyStorage::default(),
    };
    let client = StorageClient::connect_rank(&launch, GROUP, 0, 1).unwrap();
    (head, client)
}

/// The bytes of the contiguous device tensor `tensor`, once `stream` has completed the work
/// before this call.
fn host_bytes(stream: &Stream, tensor: &Tensor) -> Outcome<Vec<u8>> {
    let size = tensor
        .dtype()
        .size_in_bytes()
        .ok_or("a packed element type")?;
    let mut bytes = vec![0_u8; tensor.elem_count() * size];

    // SAFETY: the source is the tensor's own elements, contiguous from its first, and the
    // destination holds exactly their bytes; the stream is waited for before they are read.
    unsafe {
        platform::memcpy_async(
            bytes.as_mut_ptr() as u64,
            ops::operand(tensor)?.ptr.address(),
            bytes.len(),
            stream,
        )?;
    }
    stream.synchronize()?;
    Ok(bytes)
}

/// The values of BF16 bytes.
fn bf16_values(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f64::from(f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16)))
        .collect()
}

/// One input of a forward: its bytes, extents and element type.
type Input<'a> = (&'a [u8], &'a [usize], DType);

/// Runs `forward` once over the weights `load` reads from the release: first over the meta
/// store under a meta context, which states the kernels and the device bytes it needs, then over
/// a GPU's weight store under a launching context of those kernels. `forward` is handed the
/// weights and the `inputs` as tensors, and `read` the stream it ran on and its outputs.
fn on_release<W, R>(
    load: impl Fn(&WeightLoader<'_>) -> nn::Result<W>,
    inputs: &[Input<'_>],
    forward: impl Fn(&W, &[Tensor]) -> nn::Result<Vec<Tensor>>,
    read: impl FnOnce(&Stream, &[Tensor]) -> Outcome<R>,
) -> Outcome<R> {
    let checkpoint = release();

    // The meta run states the kernels and the peak before anything touches the device.
    let stated = MetaWeights::new();
    let weights = load(&loader(&checkpoint, &stated))?;
    let meta = ops::Context::meta(None);
    meta.record_launches();
    {
        let _guard = meta.enter();
        let stated: Vec<Tensor> = inputs
            .iter()
            .map(|(_, dims, dtype)| ops::meta_tensor(*dims, *dtype))
            .collect::<ops::Result<_>>()?;
        forward(&weights, &stated)?;
    }
    let keys: BTreeSet<(&'static str, String)> = meta
        .take_launches()
        .into_iter()
        .map(|(operator, key, _, _)| (operator, key))
        .collect();
    let mut kernels = Kernels::open()?;
    let generation = kernels.generation(1, &keys)?;

    // The rank's load: published to the device through the host's Storage Server.
    let server = StorageServer::start();
    let (_head, client) = rank(&server);
    let store = WeightStore::new(client, DEVICE)?;
    let weights = load(&loader(&checkpoint, &store))?;
    store.finish()?;

    let stream = Stream::new(DEVICE)?;
    let region = DeviceAllocation::linear(DEVICE, (meta.peak() as usize).max(1 << 20))?;
    let context = ops::Context::new(
        DEVICE.0 as usize,
        stream.clone(),
        generation,
        DeviceRange::new(DevicePtr::from_address(region.address()), region.len()),
    );
    let (_memory, tensors): (Vec<_>, Vec<_>) = inputs
        .iter()
        .map(|(bytes, dims, dtype)| device::tensor(&stream, bytes, dims, *dtype))
        .collect::<Outcome<Vec<_>>>()?
        .into_iter()
        .unzip();
    let outputs = {
        let _guard = context.enter();
        forward(&weights, &tensors)?
    };
    stream.synchronize()?;
    read(&stream, &outputs)
}

/// The release's merged query and key-value compression (`wq_a` and `wkv`, 5120 → 1280 + 512)
/// is the product of its block-scaled weights with the MXFP8 encoding of its input, to the
/// accumulation and output rounding of a 5120-term FP32 dot product.
#[test]
#[ignore = "requires a GPU, a Build Server and the released checkpoint"]
fn a_released_fp8_projection_is_the_product_of_its_dequantized_weights() -> Outcome {
    let (hidden, bytes) = hidden_rows(TOKENS * HIDDEN, 41);
    let attention = format!("{LAYER}.attn");

    let got = on_release(
        |loader| {
            Linear::load_merged(
                loader.root().pp(&attention),
                &["wq_a", "wkv"],
                HIDDEN as u32,
                &[QUERY_RANK as u32, KEY_VALUE as u32],
                false,
            )
        },
        &[(&bytes, &[TOKENS, HIDDEN], DType::BF16)],
        |linear, inputs| Ok(vec![linear.forward(&inputs[0])?]),
        |stream, outputs| Ok(bf16_values(&host_bytes(stream, &outputs[0])?)),
    )?;

    let checkpoint = release();
    let mut weight = fp8_weight(
        &checkpoint,
        &format!("{attention}.wq_a"),
        QUERY_RANK,
        HIDDEN,
    );
    weight.extend(fp8_weight(
        &checkpoint,
        &format!("{attention}.wkv"),
        KEY_VALUE,
        HIDDEN,
    ));
    let outputs = QUERY_RANK + KEY_VALUE;
    let want: Vec<f64> = hidden
        .chunks(HIDDEN)
        .flat_map(|row| products(&weight, HIDDEN, &encoded(row)).0)
        .collect();

    // The product's own bound (`Mxfp8Gemm::tolerance`).
    let tolerance = Mxfp8Gemm::tolerance(&GemmDescriptor::new(
        TOKENS as u64,
        outputs as u32,
        HIDDEN as u32,
        1,
    ));
    let peak = want
        .iter()
        .fold(0.0_f64, |peak, value| peak.max(value.abs())) as f32;
    let outside: Vec<usize> = (0..want.len())
        .filter(|&index| !tolerance.allows_within(peak, want[index] as f32, got[index] as f32))
        .collect();
    assert!(
        outside.is_empty(),
        "{} of {} outputs outside {tolerance:?} of the peak {peak}; first [{}, {}]: {} for {}",
        outside.len(),
        want.len(),
        outside[0] / outputs,
        outside[0] % outputs,
        got[outside[0]],
        want[outside[0]],
    );
    Ok(())
}

/// The release's grouped output projection (`wo_a`, eight groups of 4096 → 1024) is, group for
/// group, the product of the group's rows of its block-scaled weight with the MXFP8 encoding of
/// the group's slice of the attention's output, to the accumulation and output rounding of a
/// 4096-term FP32 dot product, the groups' outputs side by side.
#[test]
#[ignore = "requires a GPU, a Build Server and the released checkpoint"]
fn a_released_grouped_projection_is_each_groups_product_of_its_dequantized_weights() -> Outcome {
    let width = OUTPUT_GROUPS * GROUP_WIDTH;
    let (attended, bytes) = hidden_rows(TOKENS * width, 43);
    let attention = format!("{LAYER}.attn");

    let got = on_release(
        |loader| {
            GroupedLinear::load(
                GROUP_WIDTH as u32,
                OUTPUT_RANK as u32,
                OUTPUT_GROUPS as u32,
                loader.root().pp(&attention).pp("wo_a"),
            )
        },
        &[(&bytes, &[TOKENS, width], DType::BF16)],
        |projection, inputs| Ok(vec![projection.forward(&inputs[0])?]),
        |stream, outputs| Ok(bf16_values(&host_bytes(stream, &outputs[0])?)),
    )?;

    let checkpoint = release();
    let weight = fp8_weight(
        &checkpoint,
        &format!("{attention}.wo_a"),
        OUTPUT_GROUPS * OUTPUT_RANK,
        GROUP_WIDTH,
    );
    let outputs = OUTPUT_GROUPS * OUTPUT_RANK;
    let want: Vec<f64> = attended
        .chunks(width)
        .flat_map(|row| {
            // A row's groups are whole 32-element blocks of it, so their encoding is the row's.
            let encoded = encoded(row);
            (0..OUTPUT_GROUPS)
                .flat_map(|group| {
                    let rows =
                        group * OUTPUT_RANK * GROUP_WIDTH..(group + 1) * OUTPUT_RANK * GROUP_WIDTH;
                    let slice = &encoded[group * GROUP_WIDTH..(group + 1) * GROUP_WIDTH];
                    products(&weight[rows], GROUP_WIDTH, slice).0
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // The product's own bound (`Mxfp8Gemm::tolerance`).
    let tolerance = Mxfp8Gemm::tolerance(&GemmDescriptor::new(
        TOKENS as u64,
        OUTPUT_RANK as u32,
        GROUP_WIDTH as u32,
        OUTPUT_GROUPS as u32,
    ));
    let peak = want
        .iter()
        .fold(0.0_f64, |peak, value| peak.max(value.abs())) as f32;
    let outside: Vec<usize> = (0..want.len())
        .filter(|&index| !tolerance.allows_within(peak, want[index] as f32, got[index] as f32))
        .collect();
    assert!(
        outside.is_empty(),
        "{} of {} outputs outside {tolerance:?} of the peak {peak}; first [{}, {}]: {} for {}",
        outside.len(),
        want.len(),
        outside[0] / outputs,
        outside[0] % outputs,
        got[outside[0]],
        want[outside[0]],
    );
    Ok(())
}

/// The weights of one routed expert as the cases load it: its `[up; gate]` and down stacks and
/// its clamp.
struct Expert {
    up_gate: nn::quantization::Mxfp4Stack,
    down: nn::quantization::Mxfp4Stack,
    limit: Tensor,
}

/// A routed expert of the release (expert 7 of layer 3), every token routed to it alone: its
/// gated rows are the clamped SwiGLU of its MXFP4 projections of the encoded input, encoded into
/// MXFP8 again by the OCP rule (each block's largest exponent less E4M3's, the largest quotients
/// saturating) within half a code step, and the block's output is its down projection of those
/// very rows, to the accumulation and output rounding of a 2304-term FP32 dot product.
#[test]
#[ignore = "requires a GPU, a Build Server and the released checkpoint"]
fn a_released_mxfp4_expert_is_the_product_of_its_dequantized_weights() -> Outcome {
    let (hidden, bytes) = hidden_rows(TOKENS * HIDDEN, 43);
    let experts = format!("{LAYER}.ffn.experts");
    // Every token's one route is to the stack's only expert, at weight one.
    let ids = vec![0_u8; TOKENS * 4];
    let route_weights: Vec<u8> = std::iter::repeat_n(1.0_f32.to_le_bytes(), TOKENS)
        .flatten()
        .collect();

    let (gated_codes, gated_scales, got) = on_release(
        |loader| {
            let stack = |name: &str, projections: &[&str], layout| {
                loader.mxfp4_stack(name, &experts, projections, &[EXPERT], layout)
            };
            let up_gate = stack(
                "experts.up_gate",
                &["w3", "w1"],
                (
                    GatherGemmMxfp4Descriptor::WEIGHT_VALUES,
                    WEIGHT_SCALE_LAYOUT,
                ),
            )?;
            let down = stack(
                "experts.down",
                &["w2"],
                (RouteGemmMxfp4Descriptor::WEIGHT_VALUES, WEIGHT_SCALE_LAYOUT),
            )?;
            let limit = loader.constant(
                "experts.limit",
                DType::F32,
                &nn::Shape::from(1),
                &(LIMIT as f32).to_le_bytes(),
                "engine::tests",
            )?;
            Ok(Expert {
                up_gate,
                down,
                limit,
            })
        },
        &[
            (&bytes, &[TOKENS, HIDDEN], DType::BF16),
            (&ids, &[TOKENS, 1], DType::U32),
            (&route_weights, &[TOKENS, 1], DType::F32),
        ],
        |expert, inputs| {
            let [x, ids, weights] = inputs else {
                unreachable!("three inputs")
            };
            let maps = functional::moe_sort(ids, weights, 1, 0..1, None)?;
            let (codes, scales) =
                functional::gather_gemm_mxfp4(x, &expert.up_gate, &expert.limit, &maps, 1)?;
            let routed = functional::route_gemm_mxfp4(&codes, &scales, &expert.down, &maps, 1)?;
            let out = functional::moe_unpermute(&routed, &maps, weights, 1, None)?;
            Ok(vec![codes, scales, out])
        },
        |stream, outputs| {
            let scales = (outputs[1].dims().to_vec(), host_bytes(stream, &outputs[1])?);
            Ok((
                host_bytes(stream, &outputs[0])?,
                scales,
                bf16_values(&host_bytes(stream, &outputs[2])?),
            ))
        },
    )?;

    let checkpoint = release();
    let weight = |name: &str, rows, columns| {
        fp4_weight(
            &checkpoint,
            &format!("{experts}.{EXPERT}.{name}"),
            rows,
            columns,
        )
    };
    let (gate, up) = (
        weight("w1", INTERMEDIATE, HIDDEN),
        weight("w3", INTERMEDIATE, HIDDEN),
    );
    let down = weight("w2", HIDDEN, INTERMEDIATE);

    // The gated rows' scales, from the 128 × 4 atom: row `r`, block `b` of a table of
    // `padded` blocks a row.
    let (dims, table) = gated_scales;
    let padded = dims[1];
    let blocks = INTERMEDIATE / 32;
    let scale = |row: usize, block: usize| {
        table[(block % 4)
            + (block / 4) * 512
            + (row % 32) * 16
            + ((row % 128) / 32) * 4
            + (row / 128) * 128 * padded]
    };

    // 1. The gated rows. A route's permuted row is its token's: the sort lays one expert's
    // routes out from the first row in route order.
    let gamma = ops::tolerance::gamma(HIDDEN as u64, DType::F32);
    let mut decoded = vec![0.0_f64; TOKENS * INTERMEDIATE];
    let (mut outside, mut rescaled) = (Vec::new(), Vec::new());
    for (token, row) in hidden.chunks(HIDDEN).enumerate() {
        let x = encoded(row);
        let (gate, gate_terms) = products(&gate, HIDDEN, &x);
        let (up, up_terms) = products(&up, HIDDEN, &x);

        // The exact gated row, and how far each element's FP32 accumulation may move it.
        let mut exact = vec![0.0_f64; INTERMEDIATE];
        let mut moved = vec![0.0_f64; INTERMEDIATE];
        for column in 0..INTERMEDIATE {
            let clamped_gate = gate[column].min(LIMIT);
            let activated = clamped_gate / (1.0 + (-clamped_gate).exp());
            let clamped_up = up[column].clamp(-LIMIT, LIMIT);
            exact[column] = activated * clamped_up;
            // Each projection's accumulation carried through the SwiGLU, whose derivative in
            // the gate stays below 1.1.
            moved[column] = 2.0
                * gamma
                * (activated.abs() * up_terms[column]
                    + 1.1 * clamped_up.abs() * gate_terms[column]);
        }

        for (block, values) in exact.chunks(32).enumerate() {
            // The device's scale is the OCP rule's of the block's largest magnitude, unless the
            // accumulation may have moved that magnitude across a power of two.
            let unit = ue8m0(scale(token, block));
            let (index, amax) =
                values
                    .iter()
                    .enumerate()
                    .fold((0, 0.0_f64), |(at, amax), (index, value)| {
                        if value.abs() > amax {
                            (index, value.abs())
                        } else {
                            (at, amax)
                        }
                    });
            let ocp = 2f64.powf(amax.log2().floor() - 8.0);
            let slack = moved[block * 32 + index];
            let crossing = 2f64.powf(amax.log2().round());
            if unit != ocp && (amax - crossing).abs() > slack {
                rescaled.push((token, block, amax, unit.log2(), ocp.log2()));
            }

            for (within, want) in values.iter().enumerate() {
                let column = block * 32 + within;
                let got = e4m3(gated_codes[token * INTERMEDIATE + column]) * unit;
                decoded[token * INTERMEDIATE + column] = got;

                // The encoding under the device's scale: quotients above 448 saturate, and a code
                // is within half a step of its quotient, 2⁻⁴ of a normal value or 2⁻¹⁰ of the
                // scale for a subnormal one.
                let saturated = want.clamp(-448.0 * unit, 448.0 * unit);
                let bound = 2f64.powi(-4) * saturated.abs() + 2f64.powi(-10) * unit + moved[column];
                if (got - saturated).abs() > bound {
                    outside.push((token, column, got, *want, bound));
                }
            }
        }
    }
    assert!(
        rescaled.is_empty(),
        "{} blocks of {} under another scale than the OCP rule's; first (token, block, amax, \
         log2 of the device's scale, the rule's) {:?}",
        rescaled.len(),
        TOKENS * blocks,
        &rescaled[..rescaled.len().min(6)]
    );
    assert!(
        outside.is_empty(),
        "{} of {} gated elements outside the encoding's bound; first (token, column, device, \
         exact, bound) {:?}",
        outside.len(),
        TOKENS * INTERMEDIATE,
        &outside[..outside.len().min(6)]
    );
    assert!(blocks <= padded);

    // 2. The down projection of those very rows, combined at weight one.
    // The product's own bound (`RouteGemmMxfp4::tolerance`); the combine at weight one is exact.
    let tolerance = RouteGemmMxfp4::tolerance(&RouteGemmMxfp4Descriptor::new(
        HIDDEN as u32,
        INTERMEDIATE as u32,
        1,
        1,
    ));
    let want: Vec<f64> = decoded
        .chunks(INTERMEDIATE)
        .flat_map(|row| products(&down, INTERMEDIATE, row).0)
        .collect();
    let peak = want
        .iter()
        .fold(0.0_f64, |peak, value| peak.max(value.abs())) as f32;
    let outside: Vec<usize> = (0..want.len())
        .filter(|&index| !tolerance.allows_within(peak, want[index] as f32, got[index] as f32))
        .collect();
    assert!(
        outside.is_empty(),
        "{} of {} outputs outside {tolerance:?} of the peak {peak}; first [{}, {}]: {} for {}",
        outside.len(),
        want.len(),
        outside[0] / HIDDEN,
        outside[0] % HIDDEN,
        got[outside[0]],
        want[outside[0]],
    );
    Ok(())
}
