//! The initial noise of a diffusion request, drawn where the latents are.
//!
//! A request's noise is a function of the request's key and the element's position, so the same
//! request draws the same latents again without the host holding a sample. Element `e` of a
//! request's stream is half of one Philox word pair:
//!
//! ```text
//! counter = [pair_low, pair_high, stage, draw]      pair = e / 2
//! words   = philox4x32_10(counter, [key_low, key_high])
//! u1      = 1 − (words[0] >> 8) / 2²⁴               in (0, 1]
//! u2      = (words[1] >> 8) / 2²⁴                   in [0, 1)
//! r       = sqrt(−2 · ln u1)
//! out[e]  = r · cos(2π · u2) for an even e,  r · sin(2π · u2) for an odd one
//! ```
//!
//! The normal's arithmetic is FP32 and the result rounds once into the output's element type.
//! `stage` and `draw` place a draw within the request's schedule so that two draws never
//! coincide.
//!
//! A draw may start at an element `offset` into the stream: a model draws its conditions, its
//! video latent and its audio latent through one stream, and a rank that holds part of a latent
//! draws its own slice, so two draws at adjacent offsets are one longer draw. The key is one seed
//! for the whole buffer, addressed flattened ([`philox_normal`]), or one key per row, each row
//! its own stream ([`philox_normal_rows`]).
//!
//! The destination is execution's: latents it owns and lends, written in place. A draw no
//! buffer of execution holds is returned as a new tensor ([`philox_normal_like`]).

use ops::noise::{KeySource, NoiseArgs, NoiseDescriptor, NoiseKind, PhiloxNoise};
use tensor::{DType, Device, Tensor};

use super::on_cpu;
use crate::error::{Error, Result};

/// The multipliers and the Weyl key increments of Philox4x32.
const MULTIPLIERS: [u32; 2] = [0xD251_1F53, 0xCD9E_8D57];
const KEY_INCREMENTS: [u32; 2] = [0x9E37_79B9, 0xBB67_AE85];

/// The scale of the 24 bits a uniform keeps of its word: they and the product are exact in FP32.
const UNIFORM_SCALE: f32 = 1.0 / 16_777_216.0;

/// The ten-round Philox4x32 bijection of `counter` under `key`.
fn philox4x32_10(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    for round in 0..10 {
        // Round zero consumes the caller's key; every later round advances both key words first.
        if round > 0 {
            key = [
                key[0].wrapping_add(KEY_INCREMENTS[0]),
                key[1].wrapping_add(KEY_INCREMENTS[1]),
            ];
        }

        let first = u64::from(MULTIPLIERS[0]) * u64::from(counter[0]);
        let second = u64::from(MULTIPLIERS[1]) * u64::from(counter[2]);
        counter = [
            (second >> 32) as u32 ^ counter[1] ^ key[0],
            second as u32,
            (first >> 32) as u32 ^ counter[3] ^ key[1],
            first as u32,
        ];
    }
    counter
}

/// `count` standard normal samples of the stream of `key` from element `offset` on, in FP32.
fn normal_stream(key: u64, stage: u32, draw: u32, offset: u64, count: usize) -> Vec<f32> {
    (0..count as u64)
        .map(|index| {
            let (pair, half) = ops::noise::word_and_half(offset, index);
            let words = philox4x32_10(
                [pair as u32, (pair >> 32) as u32, stage, draw],
                [key as u32, (key >> 32) as u32],
            );

            let first = (words[0] >> 8) as f32 * UNIFORM_SCALE;
            let second = (words[1] >> 8) as f32 * UNIFORM_SCALE;
            let radius = (-2.0 * (1.0 - first).ln()).sqrt();
            let angle = std::f32::consts::TAU * second;

            if half == 0 {
                radius * angle.cos()
            } else {
                radius * angle.sin()
            }
        })
        .collect()
}

/// The descriptor of a normal draw into `dtype` elements keyed by `source`.
fn descriptor(dtype: DType, source: KeySource) -> NoiseDescriptor {
    NoiseDescriptor::new(dtype, NoiseKind::Normal, source)
}

/// Draws standard normal noise into `out`, in place and in its element type, from one `seed`
/// over the flattened buffer: element `i` is element `offset + i` of the seed's stream, whatever
/// shape `out` has.
///
/// A caller with a whole buffer of its own passes an offset of zero; an odd offset is legal and
/// takes the second half of a word first.
///
/// # Errors
///
/// When `out` has no axis or is a strided view, or the operator refuses its arguments.
pub fn philox_normal(out: &Tensor, seed: u64, stage: u32, draw: u32, offset: u64) -> Result<()> {
    const OWNER: &str = "philox_normal";

    // The draw addresses the buffer flattened, so where its axes fold changes no value: every
    // axis but the last folds into the rows the operator reads.
    let Some(width) = out.dims().last().copied().filter(|_| out.is_contiguous()) else {
        return Err(Error::shape(
            OWNER,
            format!(
                "a buffer of {:?} is drawn in place through its packed elements",
                out.dims()
            ),
        ));
    };
    let elements = out.elem_count();

    if on_cpu(out) {
        let samples = normal_stream(seed, stage, draw, offset, elements);
        let drawn = Tensor::from_vec(samples, out.shape(), &Device::Cpu)?.to_dtype(out.dtype())?;

        return Ok(out.slice_set(&drawn, 0, 0)?);
    }

    let rows = out.reshape((elements / width.max(1), width))?;
    ops::launch::<PhiloxNoise>(
        &descriptor(out.dtype(), KeySource::Scalar),
        NoiseArgs {
            out: ops::operand(&rows)?,
            keys: None,
            key: seed,
            stage,
            draw,
            offset,
        },
    )?;
    Ok(())
}

/// Standard normal noise of `like`'s shape and element type, on `like`'s device: a new tensor
/// whose element `i`, counted flattened, is element `offset + i` of `seed`'s stream
/// ([`philox_normal`]).
///
/// The draw of a model's own intermediate, which no buffer of execution holds: the noise a VAE
/// encoder samples its posterior with. `like` states the shape, the type and the device only;
/// its elements are not read.
///
/// # Errors
///
/// When `like` has no axis, the tensor cannot be allocated, or the operator refuses its
/// arguments.
pub fn philox_normal_like(
    like: &Tensor,
    seed: u64,
    stage: u32,
    draw: u32,
    offset: u64,
) -> Result<Tensor> {
    let out = if on_cpu(like) {
        Tensor::zeros(like.shape(), like.dtype(), &Device::Cpu)?
    } else {
        ops::empty(like.shape(), like.dtype())?
    };

    philox_normal(&out, seed, stage, draw, offset)?;
    Ok(out)
}

/// Draws standard normal noise into `out`, `[rows, elements]`, in place and in its element type,
/// one independent stream per row: row `r` draws from `keys[r]`, and its element `i` is element
/// `offset + i` of that key's stream.
///
/// `keys` is `[rows]` `I64`, each row's sampling key as its bit pattern.
///
/// # Errors
///
/// When `out` is not a packed matrix, `keys` is not one `I64` key per row, or the operator
/// refuses its arguments.
pub fn philox_normal_rows(
    out: &Tensor,
    keys: &Tensor,
    stage: u32,
    draw: u32,
    offset: u64,
) -> Result<()> {
    const OWNER: &str = "philox_normal_rows";

    let &[rows, elements] = out.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("the noise is [rows, elements], not {:?}", out.dims()),
        ));
    };
    if keys.dims() != [rows] || keys.dtype() != DType::I64 || !out.is_contiguous() {
        return Err(Error::shape(
            OWNER,
            format!(
                "{} sampling keys of {:?} for {rows} packed rows of one i64 key each",
                ops::abi::name(keys.dtype()),
                keys.dims()
            ),
        ));
    }

    if on_cpu(out) {
        let keys: Vec<i64> = keys.to_vec1()?;
        let samples: Vec<f32> = keys
            .iter()
            .flat_map(|key| normal_stream(*key as u64, stage, draw, offset, elements))
            .collect();
        let drawn = Tensor::from_vec(samples, out.shape(), &Device::Cpu)?.to_dtype(out.dtype())?;

        return Ok(out.slice_set(&drawn, 0, 0)?);
    }

    ops::launch::<PhiloxNoise>(
        &descriptor(out.dtype(), KeySource::Rows),
        NoiseArgs {
            out: ops::operand(out)?,
            keys: Some(ops::operand(keys)?),
            key: 0,
            stage,
            draw,
            offset,
        },
    )?;
    Ok(())
}
