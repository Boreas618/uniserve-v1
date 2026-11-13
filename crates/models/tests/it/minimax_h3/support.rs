//! What the family's tests share: safetensors shards written by hand, CPU tensors of
//! stated values, meta tensors, and forwards run under a meta context for their launch tables.

use std::collections::{BTreeMap, BTreeSet};
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
    let mut header = String::from("{");
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
    if header.len() > 1 {
        header.pop();
    }
    header.push('}');

    shard_with_header(directory, name, &header, &payload)
}

/// Writes the shard `name` in `directory` holding FP32 `tensors` (name, extents, values) and
/// returns its path: a tiny checkpoint of stated values.
pub(crate) fn f32_shard(
    directory: &Path,
    name: &str,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
) -> PathBuf {
    let typed: Vec<(&str, DType, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(tensor, shape, values)| {
            (
                tensor.as_str(),
                DType::F32,
                shape.clone(),
                f32_bytes(values),
            )
        })
        .collect();

    shard(directory, name, &typed)
}

/// `values` as little-endian FP32 bytes, the encoding a safetensors payload holds.
pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// The FP32 values `bytes` holds, little-endian: the inverse of [`f32_bytes`].
pub(crate) fn f32_values(bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len().is_multiple_of(4),
        "{} bytes are not whole FP32 elements",
        bytes.len()
    );

    let (elements, _) = bytes.as_chunks::<4>();
    elements
        .iter()
        .map(|element| f32::from_le_bytes(*element))
        .collect()
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
    let address = NEXT.fetch_add(1 << 32, Ordering::Relaxed);

    ops::retained(address, shape, dtype, &ops::meta()).unwrap()
}

/// What a forward run under a meta context states: its launches, as `(operator, key)` in launch
/// order, and the peak of the context's allocator in bytes.
pub(crate) type Recording = (Vec<(&'static str, String)>, u64);

/// Runs `forward` under a new meta context while it records, and returns what it returned with
/// the context's [`Recording`].
pub(crate) fn recorded<T, E>(forward: impl FnOnce() -> Result<T, E>) -> (Result<T, E>, Recording) {
    let context = ops::Context::meta(None);
    context.record_launches();

    let output = {
        let _guard = context.enter();
        forward()
    };
    let launches = context
        .take_launches()
        .into_iter()
        .map(|(operator, key, _)| (operator, key))
        .collect();

    (output, (launches, context.peak()))
}

/// The operators of the device backend's generic tensor primitives (`ops::primitives`);
/// every other operator a forward launches is a fused operator of `nn`.
pub(crate) const PRIMITIVES: [&str; 8] = [
    "cast",
    "copy_strided",
    "elementwise_binary",
    "elementwise_unary",
    "elementwise_where",
    "fill",
    "index",
    "reduce",
];

/// Checks that every launch of `recording` names an operator and a key, and prints the run's
/// launches, kernel keys, operators by kind and allocator peak.
pub(crate) fn report(run: &str, (launches, peak): &Recording) {
    assert!(!launches.is_empty(), "{run} launched nothing");
    for (index, (operator, key)) in launches.iter().enumerate() {
        assert!(
            !operator.is_empty() && !key.is_empty(),
            "{run}: launch {index} states operator {operator:?} and key {key:?}"
        );
    }

    // Each operator's launches and its distinct keys.
    let mut operators: BTreeMap<&str, (usize, BTreeSet<&str>)> = BTreeMap::new();
    for (operator, key) in launches {
        let entry = operators.entry(operator).or_default();
        entry.0 += 1;
        entry.1.insert(key);
    }
    let keys: usize = operators.values().map(|(_, keys)| keys.len()).sum();

    println!(
        "{run}: {} launches, {keys} kernel keys, allocator peak {peak} bytes ({:.1} MiB)",
        launches.len(),
        *peak as f64 / (1 << 20) as f64
    );
    for (kind, generic) in [("generic primitives", true), ("fused operators", false)] {
        let listed: Vec<String> = operators
            .iter()
            .filter(|(operator, _)| PRIMITIVES.contains(operator) == generic)
            .map(|(operator, (count, keys))| format!("{operator} {count}/{}", keys.len()))
            .collect();

        println!("  {kind} (launches/keys): {}", listed.join(", "));
    }
    for (operator, (_, keys)) in &operators {
        for key in keys {
            println!("    {operator}: {key}");
        }
    }
}
