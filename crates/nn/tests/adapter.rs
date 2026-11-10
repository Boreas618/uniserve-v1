//! An adapter fused at the loading boundary: every read of a tensor it edits answers the fused
//! value, a tensor it assigns joins the checkpoint, every other tensor reads as without it, and a
//! fused weight is retained under a key of its own.
//!
//! The shards are written here, header and payload, so no checkpoint and no device is needed.

#![cfg(test)]
use std::path::Path;

use nn::weights::{RetainedWeights, WeightRequest};
use nn::{Adapter, Checkpoint, DType, DevicePtr, Shape, WeightKey, WeightLoader};

/// The bytes of `values` as BF16; every value here is exactly representable.
fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn from_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
        .collect()
}

/// Write one BF16 shard of `tensors` with the header metadata `metadata`.
fn shard(path: &Path, metadata: &[(&str, &str)], tensors: &[(&str, &[u64], &[f32])]) {
    let mut header = String::from("{");
    if !metadata.is_empty() {
        let entries = metadata
            .iter()
            .map(|(key, value)| format!(r#""{key}":"{value}""#))
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(r#""__metadata__":{{{entries}}},"#));
    }
    let mut payload = Vec::new();
    for (name, shape, values) in tensors {
        let start = payload.len();
        payload.extend(bf16(values));
        let extents = shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            r#""{name}":{{"dtype":"BF16","shape":[{extents}],"data_offsets":[{start},{}]}},"#,
            payload.len()
        ));
    }
    header.pop();
    header.push('}');
    let padded = header.len().div_ceil(8) * 8;
    header.push_str(&" ".repeat(padded - header.len()));
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&payload);
    std::fs::write(path, file).unwrap();
}

/// A pipeline root whose `transformer` holds a projection, its bias, a norm and an untouched
/// matrix.
fn base(root: &Path) {
    let transformer = root.join("transformer");
    std::fs::create_dir_all(&transformer).unwrap();
    shard(
        &transformer.join("model.safetensors"),
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
fn adapter(path: &Path, metadata: &[(&str, &str)]) {
    shard(
        path,
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

#[derive(Default)]
struct Recording {
    published: Vec<(WeightKey, Vec<u8>)>,
}

impl RetainedWeights for Recording {
    fn retained(&mut self, key: &WeightKey) -> nn::Result<Option<nn::Backing>> {
        Ok(self
            .published
            .iter()
            .position(|(published, _)| published == key)
            .map(|index| {
                nn::Backing::new(
                    DevicePtr::new(0x1_0000 * (index as u64 + 1)),
                    self.published[index].1.len() as u64,
                )
            }))
    }

    fn publish(
        &mut self,
        key: &WeightKey,
        _dtype: DType,
        _shape: Shape,
        bytes: &[u8],
    ) -> nn::Result<nn::Backing> {
        self.published.push((key.clone(), bytes.to_vec()));
        Ok(nn::Backing::new(
            DevicePtr::new(0x1_0000 * self.published.len() as u64),
            bytes.len() as u64,
        ))
    }
}

#[test]
fn an_adapter_is_fused_into_every_read_of_what_it_edits() {
    let root = tempfile::tempdir().unwrap();
    base(root.path());
    let adapters = tempfile::tempdir().unwrap();
    adapter(&adapters.path().join("adapter_model.safetensors"), &COUNTS);

    let plain = Checkpoint::open(root.path()).unwrap();
    let adapted = Checkpoint::open(root.path())
        .unwrap()
        .attach(Some("transformer"), Adapter::open(adapters.path()).unwrap())
        .unwrap();
    let transformer = adapted.component("transformer").unwrap();
    let fusion = transformer
        .adapter()
        .expect("the component holds the adapter");
    assert!(fusion.edits("proj.weight") && fusion.edits("proj.bias"));
    assert!(fusion.assigns("gate.weight") && !fusion.edits("other.weight"));

    // `W + lora_B @ lora_A`: rows scaled by 0.5 and -1 of [1, 2, 3].
    let read = |name: &str| from_bf16(&transformer.read(name).unwrap().bytes);
    assert_eq!(read("proj.weight"), vec![1.5, 3.0, 4.5, 3.0, 3.0, 3.0]);
    assert_eq!(read("proj.bias"), vec![1.25, 1.25]);
    assert_eq!(read("norm.weight"), vec![1.5, 1.0, 0.5]);
    assert_eq!(read("gate.weight"), vec![9.0, 10.0, 11.0, 12.0]);
    assert_eq!(read("other.weight"), vec![7.0, 8.0]);

    // A row-parallel shard gathers its columns of the fused value, one span a row.
    let mut store = Recording::default();
    let column = WeightRequest::strided("proj.weight", 1, Shape::new(&[2, 2]), 3);
    let other = WeightRequest::region("other.weight", 0, Shape::new(&[2]));
    {
        let mut loader = WeightLoader::new(transformer, &mut store, None, None);
        loader.load(&column).unwrap();
        loader.load(&other).unwrap();
        // A pinned read converts the fused value as it would a stored one.
        let wide = loader.load(&column.clone().pinned(DType::F32)).unwrap();
        assert_eq!(wide.dtype(), DType::F32);
    }
    assert_eq!(from_bf16(&store.published[0].1), vec![3.0, 4.5, 3.0, 3.0]);
    let wide = store.published[2]
        .1
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    assert_eq!(wide, vec![3.0, 4.5, 3.0, 3.0]);

    // The fused weight is retained under a key of its own; an untouched one under the base's.
    let mut base_store = Recording::default();
    {
        let mut loader = WeightLoader::new(
            plain.component("transformer").unwrap(),
            &mut base_store,
            None,
            None,
        );
        loader.load(&column).unwrap();
        loader.load(&other).unwrap();
    }
    assert_eq!(
        from_bf16(&base_store.published[0].1),
        vec![2.0, 3.0, 5.0, 6.0]
    );
    // The base shard's identity is its own read's here (the fixture was written a moment ago),
    // so the keys are compared within the adapted checkpoint.
    let shard = transformer.files()[0].identity().clone();
    assert_eq!(store.published[1].0.files, vec![shard.clone()]);
    assert_eq!(store.published[1].1, base_store.published[1].1);
    assert_ne!(store.published[0].0.files, vec![shard]);
}

#[test]
fn an_adapter_is_held_to_its_own_statement_and_to_the_checkpoint() {
    let root = tempfile::tempdir().unwrap();
    base(root.path());
    let adapters = tempfile::tempdir().unwrap();
    let file = adapters.path().join("adapter_model.safetensors");

    // One low-rank tensor fewer stated than carried: a truncated file is not served.
    let mut short = COUNTS;
    short[2] = ("low_rank_tensors", "1");
    adapter(&file, &short);
    let error = Adapter::open(&file).unwrap_err().to_string();
    assert!(error.contains("low_rank_tensors"), "{error}");

    // Another container is not read as this one.
    let mut foreign = COUNTS;
    foreign[0] = ("format", "peft");
    adapter(&file, &foreign);
    let error = Adapter::open(&file).unwrap_err().to_string();
    assert!(error.contains("peft"), "{error}");

    // An adapter of another network edits what this one does not hold.
    adapter(&file, &COUNTS);
    let other = tempfile::tempdir().unwrap();
    let transformer = other.path().join("transformer");
    std::fs::create_dir_all(&transformer).unwrap();
    shard(
        &transformer.join("model.safetensors"),
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
        &transformer.join("model.safetensors"),
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
