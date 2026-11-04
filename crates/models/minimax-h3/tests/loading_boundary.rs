//! The loading boundary's answer to "which representation does a deployment load in".
//!
//! `dtype = "auto"` converts nothing, so the element type a component's weights keep is the one the
//! checkpoint stores them in. These tests write their own shards - header and payload by hand - so a
//! component's stored types are a fixture rather than a claim, and no device or checkpoint is needed.

#![cfg(test)]

use std::io::Write;
use std::path::Path;

use minimax_h3::load::stored_dtype;
use nn::{Checkpoint, DType};

/// The format's name for an element type; a header states the name, not a number.
fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "U8",
        DType::I16 => "I16",
        DType::I32 => "I32",
        DType::I64 => "I64",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::F32 => "F32",
        DType::F8E4M3 => "F8_E4M3",
        DType::F4E2M1 => "F4_E2M1",
    }
}

/// Write one component directory holding `tensors` - name, element type, extents - and answer it.
///
/// Every tensor is a small vector of its own type, so a component's mix of types is exactly what the
/// fixture says it is and the byte count follows from the extents.
fn component(root: &Path, name: &str, tensors: &[(&str, DType, u64)]) -> std::path::PathBuf {
    let directory = root.join(name);
    std::fs::create_dir_all(&directory).unwrap();
    let (mut header, mut payload) = (String::from("{"), Vec::new());
    for (tensor, dtype, elements) in tensors {
        let bytes = (*elements as usize) * dtype.bytes().unwrap() as usize;
        let start = payload.len() as u64;
        payload.extend(std::iter::repeat_n(0u8, bytes));
        header.push_str(&format!(
            r#""{tensor}":{{"dtype":"{}","shape":[{elements}],"data_offsets":[{start},{}]}},"#,
            dtype_name(*dtype),
            payload.len()
        ));
    }
    header.pop();
    header.push('}');
    let mut file = std::fs::File::create(directory.join("model.safetensors")).unwrap();
    // The safetensors header is its length as a little-endian `u64`, then the header, then the data.
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(&payload).unwrap();
    directory
}

/// One component of one float type answers with it, whichever it is.
#[test]
fn a_single_representation_is_the_components_own() {
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let root = tempfile::tempdir().unwrap();
        component(
            root.path(),
            "text_encoder",
            &[("embed.weight", dtype, 7), ("norm.bias", dtype, 3)],
        );
        let checkpoint = Checkpoint::open(root.path()).unwrap();
        assert_eq!(
            stored_dtype(&checkpoint, "text_encoder").unwrap(),
            dtype,
            "a component stored in {dtype} loads in {dtype}: `auto` converts nothing"
        );
    }
}

/// An integer table is not a weight: it does not make the component ambiguous.
#[test]
fn integer_tensors_do_not_count_as_a_representation() {
    let root = tempfile::tempdir().unwrap();
    component(
        root.path(),
        "transformer",
        &[
            ("proj_in.weight", DType::F32, 5),
            ("positions", DType::I64, 5),
            ("counts", DType::I32, 5),
        ],
    );
    let checkpoint = Checkpoint::open(root.path()).unwrap();
    assert_eq!(
        stored_dtype(&checkpoint, "transformer").unwrap(),
        DType::F32
    );
}

/// Two floating-point types in one component have no single representation, and the refusal names
/// both, each with a tensor that carries it.
#[test]
fn two_representations_are_refused_by_name() {
    let root = tempfile::tempdir().unwrap();
    component(
        root.path(),
        "transformer",
        &[
            ("proj_in.weight", DType::BF16, 5),
            ("adaln.table", DType::F32, 5),
        ],
    );
    let checkpoint = Checkpoint::open(root.path()).unwrap();
    let message = stored_dtype(&checkpoint, "transformer")
        .expect_err("a component of two float types has no answer")
        .to_string();
    for named in [
        "BF16",
        "F32",
        "proj_in.weight",
        "adaln.table",
        "transformer",
    ] {
        assert!(
            message.contains(named),
            "the refusal must name {named}, and it says: {message}"
        );
    }
}

/// A component with no floating-point tensor at all is refused rather than answered with a type.
#[test]
fn no_representation_is_refused() {
    let root = tempfile::tempdir().unwrap();
    component(root.path(), "scheduler", &[("steps", DType::I64, 4)]);
    let checkpoint = Checkpoint::open(root.path()).unwrap();
    let error = stored_dtype(&checkpoint, "scheduler").expect_err("no float type to keep");
    assert!(error.to_string().contains("no floating-point tensor"));
}
