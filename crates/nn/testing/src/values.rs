//! CPU tensors of stated values, their elements and bytes, and comparisons of them.

use nn::{DType, Device, Shape, Tensor};

/// `values` as the little-endian bytes of `dtype`, a floating-point type, rounded to nearest
/// even as the tensor library rounds (`ops::values::encode`).
pub fn encode(dtype: DType, values: &[f32]) -> Vec<u8> {
    ops::values::encode(dtype, values)
        .unwrap_or_else(|error| panic!("{values:?} as {dtype:?}: {error}"))
}

/// The values the little-endian bytes `bytes` of `dtype`, a floating-point type, hold, widened
/// to FP32: the inverse of [`encode`].
pub fn decode(dtype: DType, bytes: &[u8]) -> Vec<f32> {
    ops::values::decode(dtype, bytes)
        .unwrap_or_else(|error| panic!("{} bytes as {dtype:?}: {error}", bytes.len()))
}

/// The CPU tensor of `values` in `shape`, rounded into `dtype`.
pub fn tensor(values: &[f32], shape: impl Into<Shape>, dtype: DType) -> Tensor {
    let shape = shape.into();

    Tensor::from_slice(values, shape.clone(), &Device::Cpu)
        .and_then(|tensor| tensor.to_dtype(dtype))
        .unwrap_or_else(|error| panic!("{} values in {shape:?}: {error}", values.len()))
}

/// The elements of `tensor`, widened to FP32, in row-major order.
pub fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .and_then(|tensor| tensor.flatten_all())
        .and_then(|tensor| tensor.to_vec1())
        .unwrap_or_else(|error| panic!("the elements of {:?}: {error}", tensor.dims()))
}

/// Asserts that `actual` is `expected` to within `tolerance` of each element's magnitude, or of
/// one for elements below it.
pub fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "the element counts differ");

    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let bound = tolerance * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= bound,
            "element {index}: {actual} against {expected}, beyond {bound}"
        );
    }
}

/// A fixed sequence of `count` values in `(-1, 1)`, different for each `seed`: an input whose
/// elements are unlike each other and unlike their positions.
///
/// The sequence is a 32-bit linear congruential generator (Numerical Recipes' multiplier and
/// increment) seeded by the golden-ratio constant times `seed + 1`; each value is its state's
/// top 24 bits scaled to `[0, 2)` and shifted by one.
pub fn sequence(count: usize, seed: u32) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32.wrapping_mul(seed.wrapping_add(1));

    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 23) as f32 - 1.0
        })
        .collect()
}
