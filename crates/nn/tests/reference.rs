//! Shared layers and the kernels they launch, run on a device against a host reference.
//!
//! Each case draws fixed-seed inputs, rounds them to the layer's element type, runs the layer's
//! forward on the device with the kernels a Program Server builds for it, and compares every output
//! element with the same computation evaluated in `f64` on the host from the rounded inputs. The
//! bound is the operator's own tolerance (`ops::Op::tolerance`, written from the reduction length,
//! the accumulation type and the output type; `ops::tolerance` states its derivation), which bounds
//! the distance between two correct implementations, so a host reference evaluated more exactly than
//! either lies inside it.
//!
//! The cases need one GPU (`CUDA_VISIBLE_DEVICES`) and a running Program Server whose socket
//! `UNISERVE_PROGRAM_SERVER_SOCKET` names; the server builds any module its store lacks on first
//! use:
//!
//! ```text
//! uniserve program-server serve --daemon --socket <socket> \
//!     --package-root <store>/packages --store-root <store>/modules --target sm_100a \
//!     --wasm <bin>/scheduling-precompile --python crates/program-server/python/.venv/bin/python \
//!     --max-jobs 16
//! UNISERVE_PROGRAM_SERVER_SOCKET=<socket> CUDA_VISIBLE_DEVICES=0 \
//!     cargo test -p nn --test reference -- --ignored --test-threads=1
//! ```

mod support;

use nn::layers::{
    Epilogue, Gain, GateBias, GatedActivation, Halves, Linear, ResidualStream, RmsNorm, Rounding,
    WeightOrder,
};
use nn::{DType, TensorView};
use ops::Tolerance;
use ops::activation::Activation;

use support::device::Bench;
use support::{Draw, Outcome, decode, encode};

/// `values` rounded to `dtype` and widened back: the inputs exactly as the device holds them.
fn rounded(dtype: DType, values: &[f32]) -> Outcome<Vec<f32>> {
    decode(dtype, &encode(dtype, values)?)
}

/// Every element of `got` within `tolerance` of `reference`, or the first one that is not.
///
/// A bound that [`Tolerance::cancels`] is taken against the largest magnitude of the reference
/// row the element belongs to (`width` elements a row), as the operator's comparison is.
fn compare(
    name: &str,
    tolerance: Tolerance,
    width: usize,
    reference: &[f64],
    got: &[f32],
) -> Outcome {
    if reference.len() != got.len() {
        return Err(format!(
            "{name}: {} outputs, {} expected",
            got.len(),
            reference.len()
        )
        .into());
    }
    let mut outside = 0_usize;
    let mut first = None;
    for (row, (expected, candidate)) in reference.chunks(width).zip(got.chunks(width)).enumerate() {
        let peak = expected
            .iter()
            .fold(0.0_f64, |peak, value| peak.max(value.abs()));
        for (column, (expected, candidate)) in expected.iter().zip(candidate).enumerate() {
            if !tolerance.allows_within(peak as f32, *expected as f32, *candidate) {
                outside += 1;
                first.get_or_insert((row, column, *expected, *candidate));
            }
        }
    }
    match first {
        None => Ok(()),
        Some((row, column, expected, candidate)) => Err(format!(
            "{name}: {outside} of {} elements outside {tolerance:?}; first at row {row}, column \
             {column}: {candidate} against {expected}",
            got.len()
        )
        .into()),
    }
}

/// A BF16 `Linear` with a bias, its weight stored `[n, k]`, against the host product
/// `out[r, j] = Σ_i x[r, i] · w[j, i] + b[j]`. The operator's bound is the mixed-sign dot product's
/// over `k` terms accumulated in FP32 and rounded once to BF16.
#[test]
#[ignore = "requires a CUDA device and a Program Server (UNISERVE_PROGRAM_SERVER_SOCKET)"]
fn a_linear_layer_matches_the_host_product() -> Outcome {
    let (rows, k, n) = (48_usize, 512_usize, 320_usize);
    let dtype = DType::BF16;
    let mut bench = Bench::open()?;
    let mut draw = Draw::new(7);

    let x = rounded(dtype, &draw.normals(rows * k, 1.0))?;
    let w = rounded(dtype, &draw.normals(n * k, 1.0 / (k as f32).sqrt()))?;
    let b = rounded(dtype, &draw.normals(n, 0.5))?;

    let layer = Linear::new(
        bench.weight(dtype, &[n as u64, k as u64], &w)?,
        Some(bench.weight(dtype, &[n as u64], &b)?),
        WeightOrder::NxK,
    )?;
    let input = bench.values(dtype, &[rows as u64, k as u64], &x)?;
    let out = bench.run(&|f| layer.forward(f, input))?;

    let mut reference = vec![0.0_f64; rows * n];
    for row in 0..rows {
        for column in 0..n {
            let products: f64 = (0..k)
                .map(|index| f64::from(x[row * k + index]) * f64::from(w[column * k + index]))
                .sum();
            reference[row * n + column] = products + f64::from(b[column]);
        }
    }
    compare(
        "linear",
        Tolerance::dot(k as u64, DType::F32, dtype),
        n,
        &reference,
        &decode(dtype, &bench.download(out)?)?,
    )
}

/// The host RMS normalization `x · rsqrt(mean(x²) + eps) · g`, `g` the weight or one plus it.
fn rms_reference(x: &[f32], weight: &[f32], eps: f64, gain: Gain) -> Vec<f64> {
    let width = weight.len();
    x.chunks(width)
        .flat_map(|row| {
            let mean = row
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>()
                / width as f64;
            let inverse = 1.0 / (mean + eps).sqrt();
            row.iter().zip(weight).map(move |(value, weight)| {
                let gain = match gain {
                    Gain::Plain => f64::from(*weight),
                    Gain::Centred => 1.0 + f64::from(*weight),
                };
                f64::from(*value) * inverse * gain
            })
        })
        .collect()
}

/// A BF16 `RmsNorm` against the host normalization, with the stored and the zero-centred gain,
/// over rows of a width that is not a power of two. The operator's bound is a scale-type reduction
/// of `columns` squares in FP32, rounded once to BF16.
#[test]
#[ignore = "requires a CUDA device and a Program Server (UNISERVE_PROGRAM_SERVER_SOCKET)"]
fn an_rms_norm_matches_the_host_normalization() -> Outcome {
    let (rows, width) = (33_usize, 1536_usize);
    let dtype = DType::BF16;
    let eps = 1e-6_f32;
    let mut bench = Bench::open()?;
    let mut draw = Draw::new(11);

    for gain in [Gain::Plain, Gain::Centred] {
        // Rows of several scales, so the normalization is not checked at one magnitude only.
        let x: Vec<f32> = draw
            .normals(rows * width, 1.0)
            .chunks(width)
            .enumerate()
            .flat_map(|(row, values)| {
                let scale = 0.25 * (1 + row % 7) as f32;
                values
                    .iter()
                    .map(move |value| value * scale)
                    .collect::<Vec<_>>()
            })
            .collect();
        let x = rounded(dtype, &x)?;
        let weight = rounded(dtype, &draw.uniform(width, -0.5, 1.5))?;

        let norm = RmsNorm::new(
            bench.weight(dtype, &[width as u64], &weight)?,
            eps,
            Epilogue::None,
            gain,
            dtype,
        )?;
        let input = bench.values(dtype, &[rows as u64, width as u64], &x)?;
        let out = bench.run(&|f| norm.forward(f, input, None))?;

        compare(
            &format!("rms_norm {gain:?}"),
            Tolerance::rounding(width as u64, DType::F32, dtype),
            width,
            &rms_reference(&x, &weight, f64::from(eps), gain),
            &decode(dtype, &bench.download(out)?)?,
        )?;
    }
    Ok(())
}

/// A BF16 `RmsNorm` after a residual add, out of place: the stream after the add is `x + r`
/// rounded once, and the normalized row is that of the unrounded sum. The operator's bound is the
/// row's squares and the one add, in FP32, rounded once to BF16.
#[test]
#[ignore = "requires a CUDA device and a Program Server (UNISERVE_PROGRAM_SERVER_SOCKET)"]
fn a_residual_rms_norm_matches_the_host_add_and_normalization() -> Outcome {
    let (rows, width) = (17_usize, 2048_usize);
    let dtype = DType::BF16;
    let eps = 1e-5_f32;
    let mut bench = Bench::open()?;
    let mut draw = Draw::new(13);

    let x = rounded(dtype, &draw.normals(rows * width, 1.0))?;
    let residual = rounded(dtype, &draw.normals(rows * width, 4.0))?;
    let weight = rounded(dtype, &draw.uniform(width, 0.5, 1.5))?;

    let norm = RmsNorm::new(
        bench.weight(dtype, &[width as u64], &weight)?,
        eps,
        Epilogue::None,
        Gain::Plain,
        dtype,
    )?;
    let dims = [rows as u64, width as u64];
    let input = bench.values(dtype, &dims, &x)?;
    let stream_in = bench.values(dtype, &dims, &residual)?;
    let stream_out = bench.tensor(dtype, &dims)?;
    let out = bench.run(&|f| {
        norm.forward(
            f,
            input,
            Some(ResidualStream {
                input: stream_in,
                output: stream_out,
            }),
        )
    })?;

    let sums: Vec<f32> = x
        .iter()
        .zip(&residual)
        .map(|(x, residual)| (f64::from(*x) + f64::from(*residual)) as f32)
        .collect();
    let tolerance = Tolerance::rounding(width as u64 + 1, DType::F32, dtype);
    compare(
        "add_rms_norm stream",
        tolerance,
        width,
        &sums
            .iter()
            .map(|value| f64::from(*value))
            .collect::<Vec<_>>(),
        &decode(dtype, &bench.download(stream_out)?)?,
    )?;
    compare(
        "add_rms_norm output",
        tolerance,
        width,
        &rms_reference(&sums, &weight, f64::from(eps), Gain::Plain),
        &decode(dtype, &bench.download(out)?)?,
    )
}

/// `silu(g) · v` on the host.
fn gated_silu(gate: &[f32], value: &[f32]) -> Vec<f64> {
    gate.iter()
        .zip(value)
        .map(|(gate, value)| {
            let gate = f64::from(*gate);
            gate / (1.0 + (-gate).exp()) * f64::from(*value)
        })
        .collect()
}

/// A BF16 gated SiLU in each of its three rounding orders, over two halves of one row, against the
/// host product. The operators' bound is a few roundings of the output type and no reduction: three
/// for `act_and_mul`'s SiLU, four for the SwiGLU order.
#[test]
#[ignore = "requires a CUDA device and a Program Server (UNISERVE_PROGRAM_SERVER_SOCKET)"]
fn a_gated_silu_matches_the_host_product_in_every_rounding_order() -> Outcome {
    let (rows, width) = (24_usize, 1408_usize);
    let dtype = DType::BF16;
    let mut bench = Bench::open()?;
    let mut draw = Draw::new(17);

    let gate = rounded(dtype, &draw.normals(rows * width, 3.0))?;
    let value = rounded(dtype, &draw.normals(rows * width, 1.0))?;
    let reference = gated_silu(&gate, &value);
    let dims = [rows as u64, width as u64];
    let unit = dtype.unit_roundoff() as f32;

    for (rounding, roundings) in [
        (Rounding::Fused, 3.0),
        (Rounding::Stepwise, 3.0),
        (Rounding::ValueProduct, 4.0),
    ] {
        let layer = GatedActivation::new(Activation::Silu, rounding, GateBias::None)?;
        let gate_view: TensorView = bench.values(dtype, &dims, &gate)?;
        let value_view = bench.values(dtype, &dims, &value)?;
        let out = bench.run(&|f| {
            layer.forward(
                f,
                Halves::Split {
                    gate: gate_view,
                    value: value_view,
                },
            )
        })?;
        compare(
            &format!("gated silu {rounding:?}"),
            Tolerance::new(0.0, roundings * unit),
            width,
            &reference,
            &decode(dtype, &bench.download(out)?)?,
        )?;
    }
    Ok(())
}
