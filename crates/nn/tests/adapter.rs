//! An adapter fused at the loading boundary: every read of a tensor it edits returns the fused
//! value, a tensor it assigns joins the checkpoint, every other tensor reads as without it, and
//! a fused weight is retained under a key of its own.
//!
//! The shards are written here, header and payload, so no checkpoint and no device is needed.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

mod support;

use std::path::Path;

use nn::{Adapter, Checkpoint, CpuWeights, DType, Shape, WeightLayout, WeightLoader, WeightSource};
use support::{encode, shard_stating, values};

/// Writes the BF16 shard `name` of `tensors` in `directory`, stating `metadata`; every value
/// here is exactly representable.
fn shard(
    directory: &Path,
    name: &str,
    metadata: &[(&str, &str)],
    tensors: &[(&str, &[usize], &[f32])],
) {
    let typed: Vec<(&str, DType, Vec<usize>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, shape, values)| {
            (
                *name,
                DType::BF16,
                shape.to_vec(),
                encode(DType::BF16, values),
            )
        })
        .collect();
    shard_stating(directory, name, metadata, &typed);
}

/// A pipeline root whose `transformer` holds a projection, its bias, a norm and an untouched
/// matrix.
fn base(root: &Path) {
    let transformer = root.join("transformer");
    std::fs::create_dir_all(&transformer).unwrap();
    shard(
        &transformer,
        "model.safetensors",
        &[],
        &[
            ("proj.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("proj.bias", &[2], &[1.0, 1.0]),
            ("norm.weight", &[3], &[1.0, 1.0, 1.0]),
            ("other.weight", &[2], &[7.0, 8.0]),
        ],
    );
}

const COUNTS: [(&str, &str); 5] = [
    ("format", "fastvideo-lora-v2"),
    ("rank", "1"),
    ("low_rank_tensors", "2"),
    ("diff_tensors", "2"),
    ("set_weight_tensors", "1"),
];

/// A rank-one adapter of `proj`, deltas of its bias and of the norm, and an assigned gate.
fn adapter(directory: &Path, metadata: &[(&str, &str)]) {
    shard(
        directory,
        "adapter_model.safetensors",
        metadata,
        &[
            ("proj.lora_A.weight", &[1, 3], &[1.0, 2.0, 3.0]),
            ("proj.lora_B.weight", &[2, 1], &[0.5, -1.0]),
            ("proj.diff_b", &[2], &[0.25, 0.25]),
            ("norm.diff", &[3], &[0.5, 0.0, -0.5]),
            ("gate.set_weight", &[2, 2], &[9.0, 10.0, 11.0, 12.0]),
        ],
    );
}

/// The BF16 values of the tensor `name` of `checkpoint`, as a read returns them.
fn read(checkpoint: &Checkpoint, name: &str) -> Vec<f32> {
    ops::values::decode(DType::BF16, &checkpoint.read(name).unwrap()).unwrap()
}

#[test]
fn an_adapter_is_fused_into_every_read_of_what_it_edits() {
    let root = tempfile::tempdir().unwrap();
    base(root.path());
    let adapters = tempfile::tempdir().unwrap();
    adapter(adapters.path(), &COUNTS);

    let plain = Checkpoint::open(root.path()).unwrap();
    let adapted = Checkpoint::open(root.path())
        .unwrap()
        .attach(Some("transformer"), Adapter::open(adapters.path()).unwrap())
        .unwrap();
    let transformer = adapted.stage("transformer").unwrap();
    let fusion = transformer.adapter().expect("the stage holds the adapter");
    assert!(fusion.edits("proj.weight") && fusion.edits("proj.bias"));
    assert!(fusion.assigns("gate.weight") && !fusion.edits("other.weight"));

    // `W + lora_B @ lora_A`: rows scaled by 0.5 and -1 of [1, 2, 3].
    assert_eq!(
        read(transformer, "proj.weight"),
        [1.5, 3.0, 4.5, 3.0, 3.0, 3.0]
    );
    assert_eq!(read(transformer, "proj.bias"), [1.25, 1.25]);
    assert_eq!(read(transformer, "norm.weight"), [1.5, 1.0, 0.5]);
    assert_eq!(read(transformer, "gate.weight"), [9.0, 10.0, 11.0, 12.0]);
    assert_eq!(read(transformer, "other.weight"), [7.0, 8.0]);

    // A row-parallel shard gathers its columns of the fused value, one span a row.
    let column = WeightSource::Strided {
        tensor: "proj.weight".to_owned(),
        start: 1,
        shape: Shape::from((2, 2)),
        row_stride: 3,
    };
    let other = WeightSource::Region {
        tensor: "other.weight".to_owned(),
        start: 0,
        shape: Shape::from(2),
    };
    let load = |loader: &WeightLoader<'_>, source: &WeightSource, shape, pinned| {
        loader
            .load(
                "weight",
                source,
                WeightLayout::ROW_MAJOR,
                &Shape::from(shape),
                pinned,
                false,
            )
            .unwrap()
    };

    let store = CpuWeights::new();
    let loader = WeightLoader::new(transformer, &store, None);
    let fused = load(&loader, &column, vec![2, 2], None);
    assert_eq!(values(&fused), [3.0, 4.5, 3.0, 3.0]);
    let untouched = load(&loader, &other, vec![2], None);

    // A pinned read converts the fused value as it would a stored one.
    let wide = load(&loader, &column, vec![2, 2], Some(DType::F32));
    assert_eq!(
        (wide.dtype(), values(&wide)),
        (DType::F32, vec![3.0, 4.5, 3.0, 3.0])
    );

    // The base checkpoint's own columns, for comparison.
    let base_store = CpuWeights::new();
    let base_loader = WeightLoader::new(plain.stage("transformer").unwrap(), &base_store, None);
    assert_eq!(
        values(&load(&base_loader, &column, vec![2, 2], None)),
        [2.0, 3.0, 5.0, 6.0]
    );
    assert_eq!(
        values(&load(&base_loader, &other, vec![2], None)),
        values(&untouched)
    );

    // The fused weight is retained under a key that names both files, the base shard's and the
    // adapter's; an untouched one under the base shard's alone.
    let shard = transformer.files()[0].identity().clone();
    let keys = store.keys();
    let key = |tensor: &str| {
        keys.iter()
            .find(|key| key.tensors == [tensor] && key.dtype == DType::BF16)
            .unwrap()
    };
    assert_eq!(key("other.weight").files, vec![shard.clone()]);
    assert_ne!(key("proj.weight").files, vec![shard]);
}

#[test]
fn an_adapter_is_held_to_its_own_statement_and_to_the_checkpoint() {
    let root = tempfile::tempdir().unwrap();
    base(root.path());
    let adapters = tempfile::tempdir().unwrap();
    let file = adapters.path().join("adapter_model.safetensors");

    // One low-rank tensor fewer stated than the file holds: a truncated file is not served.
    let mut short = COUNTS;
    short[2] = ("low_rank_tensors", "1");
    adapter(adapters.path(), &short);
    let error = Adapter::open(&file).unwrap_err().to_string();
    assert!(error.contains("low_rank_tensors"), "{error}");

    // Another container is not read as this one.
    let mut foreign = COUNTS;
    foreign[0] = ("format", "peft");
    adapter(adapters.path(), &foreign);
    let error = Adapter::open(&file).unwrap_err().to_string();
    assert!(error.contains("peft"), "{error}");

    // An adapter of another network edits what this one does not hold.
    adapter(adapters.path(), &COUNTS);
    let other = tempfile::tempdir().unwrap();
    let transformer = other.path().join("transformer");
    std::fs::create_dir_all(&transformer).unwrap();
    shard(
        &transformer,
        "model.safetensors",
        &[],
        &[("unrelated.weight", &[2], &[1.0, 2.0])],
    );
    let error = Checkpoint::open(other.path())
        .unwrap()
        .attach(Some("transformer"), Adapter::open(&file).unwrap())
        .err()
        .expect("refused")
        .to_string();
    assert!(error.contains("does not hold"), "{error}");

    // An assigned parameter the checkpoint already holds is not overwritten.
    shard(
        &transformer,
        "model.safetensors",
        &[],
        &[
            ("proj.weight", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("proj.bias", &[2], &[1.0, 1.0]),
            ("norm.weight", &[3], &[1.0, 1.0, 1.0]),
            ("gate.weight", &[2, 2], &[0.0, 0.0, 0.0, 0.0]),
        ],
    );
    let error = Checkpoint::open(other.path())
        .unwrap()
        .attach(Some("transformer"), Adapter::open(&file).unwrap())
        .err()
        .expect("refused")
        .to_string();
    assert!(error.contains("already holds"), "{error}");
}
