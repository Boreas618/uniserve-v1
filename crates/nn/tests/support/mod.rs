//! What nn's integration tests share: safetensors shards written by hand, CPU tensors of stated
//! values, and forwards run under a meta context for their launch tables.
//!
//! Each test file that declares `mod support;` compiles its own copy and uses part of it.
#![allow(dead_code, reason = "each test crate uses part of the shared support")]

use std::path::{Path, PathBuf};

use nn::{DType, Device, Shape, Tensor};

/// The format's name for an element type: the header states the name, not a number.
fn format_name(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "U8",
        DType::I16 => "I16",
        DType::I32 => "I32",
        DType::I64 => "I64",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::F32 => "F32",
        DType::F8E4M3 => "F8_E4M3",
        DType::F4 => "F4",
        other => panic!("the fixtures write no {other:?} tensor"),
    }
}

/// Writes the safetensors file `name` in `directory` with `header`, the JSON body of the
/// format's header, and `payload`, and returns its path.
///
/// The file is composed from the format's own grammar (an eight-byte length, the JSON body padded
/// to eight bytes as every exporter writes it, then the payload), so a fixture states the format
/// without going through a writer.
pub(crate) fn shard_with_header(
    directory: &Path,
    name: &str,
    header: &str,
    payload: &[u8],
) -> PathBuf {
    let mut body = header.to_owned();
    let padded = body.len().div_ceil(8) * 8;
    body.push_str(&" ".repeat(padded - body.len()));

    let mut file = (body.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(body.as_bytes());
    file.extend_from_slice(payload);

    let path = directory.join(name);
    std::fs::write(&path, file).unwrap();
    path
}

/// Writes one shard holding `tensors` (name, element type, extents, bytes) and returns its
/// path. A shape and a byte count that disagree are a bug in the fixture and fail here.
pub(crate) fn shard(
    directory: &Path,
    name: &str,
    tensors: &[(&str, DType, Vec<usize>, Vec<u8>)],
) -> PathBuf {
    shard_stating(directory, name, &[], tensors)
}

/// [`shard`], with `metadata` as the header's `__metadata__` block: what the file's writer
/// states about it.
pub(crate) fn shard_stating(
    directory: &Path,
    name: &str,
    metadata: &[(&str, &str)],
    tensors: &[(&str, DType, Vec<usize>, Vec<u8>)],
) -> PathBuf {
    let mut header = String::from("{");
    if !metadata.is_empty() {
        let entries = metadata
            .iter()
            .map(|(key, value)| format!(r#""{key}":"{value}""#))
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(r#""__metadata__":{{{entries}}},"#));
    }
    let mut payload: Vec<u8> = Vec::new();

    for (tensor, dtype, shape, bytes) in tensors {
        let expected = shape.iter().product::<usize>() * dtype.size_in_bytes();
        assert_eq!(
            expected,
            bytes.len(),
            "the fixture's `{tensor}` says {shape:?} of {dtype:?}, which is {expected} bytes"
        );

        let start = payload.len();
        payload.extend_from_slice(bytes);
        let extents: Vec<String> = shape.iter().map(usize::to_string).collect();
        header.push_str(&format!(
            r#""{tensor}":{{"dtype":"{}","shape":[{}],"data_offsets":[{start},{}]}},"#,
            format_name(*dtype),
            extents.join(","),
            payload.len()
        ));
    }
    header.pop();
    header.push('}');

    shard_with_header(directory, name, &header, &payload)
}

/// `values` as the little-endian bytes of `dtype`, a floating-point type, rounded to nearest
/// even.
pub(crate) fn encode(dtype: DType, values: &[f32]) -> Vec<u8> {
    ops::values::encode(dtype, values).unwrap()
}

/// `values` as little-endian FP32 bytes.
pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    encode(DType::F32, values)
}

/// The CPU tensor of `values` in `shape`, rounded into `dtype`.
pub(crate) fn tensor(values: &[f32], shape: impl Into<Shape>, dtype: DType) -> Tensor {
    Tensor::from_slice(values, shape.into(), &Device::Cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap()
}

/// The elements of `tensor`, widened to FP32, in row-major order.
pub(crate) fn values(tensor: &Tensor) -> Vec<f32> {
    tensor
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// Asserts that `actual` is `expected` to within `tolerance` of each element's magnitude, or of
/// one for elements below it.
pub(crate) fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
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
pub(crate) fn sequence(count: usize, seed: u32) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32.wrapping_mul(seed.wrapping_add(1));

    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1 << 23) as f32 - 1.0
        })
        .collect()
}

/// A meta tensor of `shape` elements of `dtype`: an input of a profile run, which takes no block
/// of the context's allocator. Each call gives another address, as two inputs of a forward have.
pub(crate) fn meta(shape: impl Into<Shape>, dtype: DType) -> Tensor {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(1 << 40);
    let address = NEXT.fetch_add(1 << 24, Ordering::Relaxed);

    ops::retained(address, shape, dtype, &ops::meta()).unwrap()
}

/// Runs `forward` under a new meta context while it records, and returns what it returned with
/// the launch table as `(operator, key)` in launch order.
pub(crate) fn recorded<T>(
    forward: impl FnOnce() -> nn::Result<T>,
) -> (nn::Result<T>, Vec<(&'static str, String)>) {
    recorded_split(None, forward)
}

/// [`recorded`], with the context stating this rank's place `(rank, ranks)` in a group.
pub(crate) fn recorded_split<T>(
    split: Option<(u64, u64)>,
    forward: impl FnOnce() -> nn::Result<T>,
) -> (nn::Result<T>, Vec<(&'static str, String)>) {
    let context = ops::Context::meta(None);
    if let Some((rank, ranks)) = split {
        context.set_group(None, rank, ranks).unwrap();
    }
    context.record_launches();

    let output = {
        let _guard = context.enter();
        forward()
    };
    let launches = context
        .take_launches()
        .into_iter()
        .map(|(operator, key, ..)| (operator, key))
        .collect();
    (output, launches)
}

/// The operators of a launch table, in launch order.
pub(crate) fn operators(launches: &[(&'static str, String)]) -> Vec<&'static str> {
    launches.iter().map(|(operator, _)| *operator).collect()
}
