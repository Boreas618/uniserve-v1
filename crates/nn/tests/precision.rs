//! Dense projections converted at load into FP8 or NVFP4, their activations scaled per call
//! (`Scheme::DynamicFp8`, `Scheme::DynamicNvfp4`), held to the error their formats allow.
//!
//! The bound a converted product is held to follows from the formats alone. An E4M3 element
//! rounds to nearest within half a unit of its three-bit mantissa, `2⁻⁴` of its value, or of the
//! smallest subnormal step `2⁻⁹` of its scale `s`: `e ≤ 2⁻⁴|x| + 2⁻¹⁰ s`. An NVFP4 element rounds
//! within half the widest E2M1 step (2, between 4 and 6) of its block's scale, and that scale is
//! the block's largest magnitude `b` over 6, rounded to E4M3 within `2⁻⁴` (or below the smallest
//! subnormal, `2⁻¹⁰ s` absolute, where a block that encodes to zeros errs by at most its own
//! magnitude, below `6 · 2⁻¹⁰ s`): `e ≤ (17/96) b + 6 · 2⁻¹⁰ s`. A product of encoded operands
//! then differs from the exact dense product by at most
//!
//! ```text
//! Σ_k (e_x |w| + |x| e_w + e_x e_w)              the operands' encodings
//! + (k + 8) u₃₂ Σ_k (|x| + e_x)(|w| + e_w)       the FP32 accumulation and the factors' roundings
//! + u_out (|y| + the two terms above)            the one rounding into the output type
//! ```

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use std::path::Path;

use nn::collective::Split;
use nn::modules::{Linear, LinearConfig, Parallel};
use nn::quantization::{Declaration, Scheme};
use nn::{Checkpoint, CpuWeights, DType, MetaWeights, WeightLoader};
use nn_testing::{
    Recording, encode, meta, operators, record, sequence, tensor, values, write_checkpoint,
};

/// The FP32 unit roundoff, and BF16's: the accumulation's and the output's.
const U_F32: f64 = 1.0 / (1u64 << 24) as f64;
const U_BF16: f64 = 1.0 / 256.0;

/// The dense checkpoint at `directory` holding the BF16 matrix `proj.weight`, `[n, k]`.
fn dense(directory: &Path, n: usize, k: usize, weight: &[f32]) -> Checkpoint {
    write_checkpoint(
        directory,
        &[&[(
            "proj.weight",
            DType::BF16,
            vec![n, k],
            encode(DType::BF16, weight),
        )]],
    );
    Checkpoint::open(directory).unwrap()
}

/// The declaration converting `proj` to `scheme`.
fn converting(scheme: Scheme) -> Declaration {
    Declaration::converted(vec![("proj".to_owned(), scheme)]).unwrap()
}

/// The largest magnitude of `values`, in FP64.
fn amax(values: &[f64]) -> f64 {
    values.iter().fold(0.0, |amax, value| amax.max(value.abs()))
}

/// The bound each element of `values` encodes within under `scheme` (module formulas): FP8 one
/// scale for the tensor, NVFP4 one per sixteen elements of a row of `k`.
fn encoding_bounds(values: &[f64], k: usize, scheme: Scheme) -> Vec<f64> {
    let tensor = amax(values);
    match scheme {
        Scheme::DynamicFp8 => {
            let scale = tensor / 448.0;
            values
                .iter()
                .map(|value| value.abs() / 16.0 + scale / 1024.0)
                .collect()
        }
        _ => {
            let scale = tensor / 2688.0;
            // Rows are whole blocks, so the blocks of the flattened matrix are its rows'.
            assert_eq!(k % 16, 0);
            values
                .chunks(16)
                .flat_map(|block| {
                    let largest = amax(block);
                    block
                        .iter()
                        .map(move |_| largest * 17.0 / 96.0 + 6.0 * scale / 1024.0)
                })
                .collect()
        }
    }
}

/// Every element of the converted product `actual` of `x [m, k]` and `weight [n, k]` lies within
/// its formats' bound (module documentation) of the exact dense product.
fn assert_within_format(actual: &[f32], x: &[f64], weight: &[f64], k: usize, scheme: Scheme) {
    let (e_x, e_w) = (
        encoding_bounds(x, k, scheme),
        encoding_bounds(weight, k, scheme),
    );
    let (m, n) = (x.len() / k, weight.len() / k);

    for row in 0..m {
        for column in 0..n {
            let (mut exact, mut encoding, mut magnitude) = (0.0, 0.0, 0.0);
            for depth in 0..k {
                let (a, b) = (x[row * k + depth], weight[column * k + depth]);
                let (da, db) = (e_x[row * k + depth], e_w[column * k + depth]);

                exact += a * b;
                encoding += da * b.abs() + a.abs() * db + da * db;
                magnitude += (a.abs() + da) * (b.abs() + db);
            }
            let accumulation = (k as f64 + 8.0) * U_F32 * magnitude;
            let bound = encoding + accumulation + U_BF16 * (exact.abs() + encoding + accumulation);

            let value = f64::from(actual[row * n + column]);
            assert!(
                (value - exact).abs() <= bound,
                "{scheme:?} [{row}, {column}]: {value} against {exact}, beyond {bound}"
            );
        }
    }
}

/// The values of `values` once rounded to BF16, in FP64: what the checkpoint and the activation
/// hold.
fn as_bf16(elements: &[f32]) -> Vec<f64> {
    values(&tensor(elements, elements.len(), DType::BF16))
        .into_iter()
        .map(f64::from)
        .collect()
}

/// A converted projection's product lies within its formats' bound of the dense product, for
/// operands of either sign and for positive ones, whose products cancel nothing so that the
/// bound is a fraction of the result, and with an activation outlier that sets the tensor's
/// scale far above its other elements.
#[test]
fn a_converted_product_is_within_its_formats_bound_of_the_dense_product() {
    let (m, n, k) = (5, 256, 256);

    for scheme in [Scheme::DynamicFp8, Scheme::DynamicNvfp4] {
        for positive in [false, true] {
            let shift = |values: Vec<f32>| -> Vec<f32> {
                if positive {
                    values.iter().map(|value| value.abs() + 0.125).collect()
                } else {
                    values
                }
            };
            let weight: Vec<f32> = shift(sequence(n * k, 3))
                .iter()
                .map(|value| value * 0.0625)
                .collect();
            let mut x = shift(sequence(m * k, 4));
            x[k + 7] = 24.0;

            let directory = tempfile::tempdir().unwrap();
            let checkpoint = dense(directory.path(), n, k, &weight);
            let store = CpuWeights::new();
            let mut loader = WeightLoader::new(&checkpoint, &store, None);
            loader.declare(&converting(scheme)).unwrap();
            let linear = Linear::load(
                &LinearConfig {
                    inputs: k as u32,
                    outputs: n as u32,
                    bias: false,
                    parallel: Parallel::None,
                },
                loader.root().pp("proj"),
            )
            .unwrap();
            assert_eq!(linear.dtype(), Some(DType::BF16));
            assert_eq!(linear.is_packed(), scheme == Scheme::DynamicNvfp4);

            let out = linear.forward(&tensor(&x, (m, k), DType::BF16)).unwrap();
            assert_eq!((out.dims(), out.dtype()), (&[m, n][..], DType::BF16));
            assert_within_format(&values(&out), &as_bf16(&x), &as_bf16(&weight), k, scheme);
        }
    }
}

/// A tensor-parallel share is the part of the whole tensor's encoding it holds: a column share
/// computes exactly the whole product's columns, and a row share holds exactly the whole
/// encoding's columns, whatever the share's own magnitudes.
#[test]
fn a_share_is_its_part_of_the_whole_tensors_encoding() {
    let (n, k) = (512, 512);
    let mut weight: Vec<f32> = sequence(n * k, 11)
        .iter()
        .map(|value| value * 0.1)
        .collect();
    // The largest magnitude lies in the first quarter of rows and of columns, so no share but
    // the first holds the tensor's scale.
    weight[3] = 2.0;
    let x = tensor(&sequence(3 * k, 12), (3, k), DType::BF16);

    for scheme in [Scheme::DynamicFp8, Scheme::DynamicNvfp4] {
        let directory = tempfile::tempdir().unwrap();
        let checkpoint = dense(directory.path(), n, k, &weight);
        let store = CpuWeights::new();
        let mut loader = WeightLoader::new(&checkpoint, &store, None);
        loader.declare(&converting(scheme)).unwrap();
        let load = |parallel| {
            Linear::load(
                &LinearConfig {
                    inputs: k as u32,
                    outputs: n as u32,
                    bias: false,
                    parallel,
                },
                loader.root().pp("proj"),
            )
            .unwrap()
        };

        let whole = load(Parallel::None);
        let column = load(Parallel::Column(Split { size: 2, rank: 1 }));
        let row = load(Parallel::Row(Split { size: 2, rank: 1 }));

        let all = values(&whole.forward(&x).unwrap());
        let share = values(&column.forward(&x).unwrap());
        for (index, value) in share.iter().enumerate() {
            let (row, column) = (index / (n / 2), index % (n / 2));
            assert_eq!(
                *value,
                all[row * n + n / 2 + column],
                "{scheme:?} [{row}, {column}]"
            );
        }

        // A row share's stored elements are the whole encoding's columns of its block: one
        // element a byte for FP8, two for NVFP4.
        let per_byte = if scheme == Scheme::DynamicNvfp4 { 2 } else { 1 };
        let (stored, held) = (values(whole.weight()), values(row.weight()));
        let (width, half) = (k / per_byte, k / per_byte / 2);
        for (index, value) in held.iter().enumerate() {
            let (r, c) = (index / half, index % half);
            assert_eq!(
                *value,
                stored[r * width + half + c],
                "{scheme:?} [{r}, {c}]"
            );
        }
    }
}

/// A converted projection's forward measures its activation and multiplies on the device: each
/// row's largest magnitude (`reduce`'s `absmax`), the two factors (`dynamic_scale`), then the
/// encoded product. A meta load and forward state those kernels and nothing else.
#[test]
fn a_converted_projection_states_its_measurement_and_its_product() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = dense(directory.path(), 256, 512, &sequence(256 * 512, 5));

    for (scheme, expected) in [
        (
            Scheme::DynamicFp8,
            &["reduce", "dynamic_scale", "fp8_quantize", "fp8_gemm"][..],
        ),
        (
            Scheme::DynamicNvfp4,
            &["reduce", "dynamic_scale", "nvfp4_quantize", "nvfp4_gemm"][..],
        ),
    ] {
        let store = MetaWeights::new();
        let mut loader = WeightLoader::new(&checkpoint, &store, None);
        loader.declare(&converting(scheme)).unwrap();
        let linear = Linear::load(
            &LinearConfig {
                inputs: 512,
                outputs: 256,
                bias: false,
                parallel: Parallel::None,
            },
            loader.root().pp("proj"),
        )
        .unwrap();

        let (output, Recording { launches, .. }) =
            record(&[], || linear.forward(&meta((40, 512), DType::BF16)));
        assert_eq!(output.unwrap().dims(), [40, 256]);
        assert_eq!(operators(&launches), expected, "{scheme:?}");
        assert_eq!(
            launches[..2],
            [
                (
                    "reduce",
                    "function_absmax_dtype_bf16_kept_1_reduced_1_tile_512_block_1024_pdl_false"
                        .to_owned()
                ),
                ("dynamic_scale", "dtype_bf16_tile_1024_pdl_false".to_owned()),
            ],
            "{scheme:?}"
        );
    }
}

/// A converted weight is read through its encoded product only: a module that would read it
/// dense is refused, so a policy is never served in another precision, and an NVFP4 share that
/// is not whole tiles of the product is refused at load, naming the projection.
#[test]
fn a_converted_weight_is_refused_where_no_encoded_product_reads_it() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = dense(directory.path(), 256, 512, &sequence(256 * 512, 6));

    let store = MetaWeights::new();

    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    loader.declare(&converting(Scheme::DynamicFp8)).unwrap();
    let dense_read = loader.root().pp("proj").get((256, 512), "weight");
    let message = dense_read.unwrap_err().to_string();
    assert!(
        message.contains("converts this weight at load"),
        "{message}"
    );

    loader.declare(&converting(Scheme::DynamicNvfp4)).unwrap();
    let narrow = Linear::load(
        &LinearConfig {
            inputs: 512,
            outputs: 256,
            bias: false,
            parallel: Parallel::Column(Split { size: 2, rank: 0 }),
        },
        loader.root().pp("proj"),
    );
    let message = narrow.err().unwrap().to_string();
    assert!(
        message.contains("proj.weight") && message.contains("not whole tiles of 256"),
        "{message}"
    );
}
