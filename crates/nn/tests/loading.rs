//! The loading boundary's behaviour: a checkpoint read through its index, its file identities, and
//! the loader's conversion, fusion and key reuse.
//!
//! These tests build their own shards with the safetensors crate, so they need no checkpoint and no
//! device: a weight is published to a stub store that records what it was given.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]
use std::path::Path;

use nn::weights::{RetainedWeights, WeightRequest};
use nn::{Checkpoint, DType, DevicePtr, Shape, WeightKey, WeightLayout, WeightLoader};
use safetensors::tensor::{Dtype, TensorView};

/// The bytes of `values` as BF16.
fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// Write one shard holding `tensors` and answer the path.
fn shard(dir: &Path, name: &str, tensors: &[(&str, Vec<u64>, Vec<u8>)]) -> std::path::PathBuf {
    let path = dir.join(name);
    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(tensor, shape, bytes)| {
            let shape: Vec<usize> = shape.iter().map(|&extent| extent as usize).collect();
            (
                (*tensor).to_owned(),
                TensorView::new(Dtype::BF16, shape, bytes).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, &None, &path).unwrap();
    path
}

/// A store that keeps what it was given, in host memory.
#[derive(Default)]
struct Recording {
    published: Vec<(WeightKey, Vec<u8>, DType, Shape, u64)>,
    next: u64,
}

impl RetainedWeights for Recording {
    fn retained(&mut self, key: &WeightKey) -> nn::Result<Option<nn::Backing>> {
        Ok(self
            .published
            .iter()
            .find(|(published, ..)| published == key)
            .map(|(_, bytes, _, _, address)| {
                nn::Backing::new(DevicePtr::new(*address), bytes.len() as u64)
            }))
    }

    fn publish(
        &mut self,
        key: &WeightKey,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
    ) -> nn::Result<nn::Backing> {
        self.next += 0x1_0000;
        let address = self.next;
        self.published
            .push((key.clone(), bytes.to_vec(), dtype, shape, address));
        Ok(nn::Backing::new(
            DevicePtr::new(address),
            bytes.len() as u64,
        ))
    }
}

#[test]
fn a_checkpoint_is_read_through_its_index() {
    let dir = tempfile::tempdir().unwrap();
    let values = [1.0_f32, 2.0, 3.0, 4.0];
    let bytes = bf16(&values);
    let first = shard(
        dir.path(),
        "model-00001-of-00002.safetensors",
        &[("model.embed_tokens.weight", vec![2, 2], bytes.clone())],
    );
    let second = shard(
        dir.path(),
        "model-00002-of-00002.safetensors",
        &[("lm_head.weight", vec![2, 2], bytes)],
    );
    let index = format!(
        r#"{{"metadata": {{}}, "weight_map": {{"model.embed_tokens.weight": "{}", "lm_head.weight": "{}"}}}}"#,
        first.file_name().unwrap().to_string_lossy(),
        second.file_name().unwrap().to_string_lossy(),
    );
    std::fs::write(dir.path().join("model.safetensors.index.json"), index).unwrap();

    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    assert_eq!(checkpoint.len(), 2);
    assert_eq!(checkpoint.files().len(), 2);

    let tensor = checkpoint.read("model.embed_tokens.weight").unwrap();
    assert_eq!(tensor.dtype, DType::BF16);
    assert_eq!(tensor.shape, Shape::new(&[2, 2]));
    assert_eq!(tensor.bytes, bf16(&values));
    assert!(checkpoint.tensor("lm_head.weight").is_some());
    assert!(checkpoint.tensor("absent").is_none());

    // A file's identity is its device, inode, times, or the blob name of a cache; two files of one
    // checkpoint have different identities.
    let first_identity = checkpoint.identity("model.embed_tokens.weight").unwrap();
    let second_identity = checkpoint.identity("lm_head.weight").unwrap();
    assert_ne!(first_identity, second_identity);
    assert!(first_identity.as_str().contains(':') || first_identity.as_str().starts_with("blob:"));
}

#[test]
fn a_region_is_converted_once_and_reused_by_its_key() {
    let dir = tempfile::tempdir().unwrap();
    let values: Vec<f32> = (0..8).map(|index| index as f32).collect();
    shard(
        dir.path(),
        "model.safetensors",
        &[("weight", vec![2, 4], bf16(&values))],
    );
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let mut store = Recording::default();

    let request = WeightRequest::region("weight", 0, Shape::new(&[2, 4]));
    let first = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, Some(DType::F32), None);
        loader.load(&request).unwrap()
    };
    assert_eq!(first.dtype(), DType::F32);
    assert_eq!(first.shape(), Shape::new(&[2, 4]));
    assert_eq!(store.published.len(), 1);
    // The conversion is exact for these values: they are small integers.
    let published = &store.published[0].1;
    let converted: Vec<f32> = published
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    assert_eq!(converted, values);

    // The same request is served from retained storage: no second conversion, no second publish.
    let second = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, Some(DType::F32), None);
        loader.load(&request).unwrap()
    };
    assert_eq!(store.published.len(), 1);
    assert_eq!(second.view().ptr, first.view().ptr);
    assert_eq!(second.view().shape, first.view().shape);
}

#[test]
fn fused_parts_are_joined_in_the_order_the_operator_reads() {
    let dir = tempfile::tempdir().unwrap();
    let gate: Vec<f32> = (0..4).map(|index| 10.0 + index as f32).collect();
    let up: Vec<f32> = (0..4).map(|index| 20.0 + index as f32).collect();
    shard(
        dir.path(),
        "model.safetensors",
        &[
            ("gate", vec![2, 2], bf16(&gate)),
            ("up", vec![2, 2], bf16(&up)),
        ],
    );
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let mut store = Recording::default();

    // `up` before `gate`, as `WeightLayout::UP_GATE` declares.
    let request = WeightRequest::parts(
        "experts.up_gate",
        vec!["up".to_owned(), "gate".to_owned()],
        Shape::new(&[4, 2]),
        WeightLayout::UP_GATE,
    );
    let weight = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, Some(DType::F32), None);
        loader.load(&request).unwrap()
    };
    assert_eq!(weight.shape(), Shape::new(&[4, 2]));
    let published = &store.published[0].1;
    let converted: Vec<f32> = published
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect();
    assert_eq!(converted, [up, gate].concat());
    assert!(store.published[0].0.to_string().contains("up_gate"));
    assert!(store.published[0].0.to_string().contains("up+gate"));
}

#[test]
fn a_layout_a_single_region_cannot_produce_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    shard(
        dir.path(),
        "model.safetensors",
        &[("weight", vec![2, 2], bf16(&[1.0, 2.0, 3.0, 4.0]))],
    );
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let mut store = Recording::default();
    let request =
        WeightRequest::region("weight", 0, Shape::new(&[2, 2])).with_layout(WeightLayout::UP_GATE);
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    assert!(loader.load(&request).is_err());
    assert!(store.published.is_empty());
}
