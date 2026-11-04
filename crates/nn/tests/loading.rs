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

/// Write one shard whose tensors are byte buffers of a stated type, and answer the path.
///
/// The dtype is stated per tensor because an encoded expert's fields are not all bytes: its packed
/// FP4 values and its swizzled scales are `U8`, and its per-expert global scale is an FP32 scalar
/// stored as four bytes. A shape and a byte count that disagree are a bug in the test's fixture, so
/// the view refuses them here rather than in the loader.
fn shard_typed(
    dir: &Path,
    name: &str,
    tensors: &[(&str, Dtype, Vec<u64>, Vec<u8>)],
) -> std::path::PathBuf {
    let path = dir.join(name);
    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(tensor, dtype, shape, bytes)| {
            let shape: Vec<usize> = shape.iter().map(|&extent| extent as usize).collect();
            (
                (*tensor).to_owned(),
                TensorView::new(*dtype, shape, bytes).unwrap(),
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

#[test]
fn a_computed_constant_is_published_once_and_needs_its_own_name() {
    let dir = tempfile::tempdir().unwrap();
    shard(
        dir.path(),
        "model.safetensors",
        &[("weight", vec![1], bf16(&[1.0]))],
    );
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let mut store = Recording::default();
    let table: Vec<u8> = [0.5_f32, 0.25]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    let first = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .constant("rotary.inv_freq", DType::F32, Shape::new(&[2]), &table)
            .unwrap()
    };
    let second = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .constant("rotary.inv_freq", DType::F32, Shape::new(&[2]), &table)
            .unwrap()
    };
    assert_eq!(store.published.len(), 1);
    assert_eq!(first.view().ptr, second.view().ptr);
    assert!(store.published[0].0.files.is_empty());

    // The same bytes under another name are another constant.
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    loader
        .constant("other.inv_freq", DType::F32, Shape::new(&[2]), &table)
        .unwrap();
    assert_eq!(store.published.len(), 2);

    // A byte count that does not match the extents is refused before anything is published.
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    assert!(
        loader
            .constant("short.inv_freq", DType::F32, Shape::new(&[2]), &table[..4])
            .is_err()
    );
    assert_eq!(store.published.len(), 2);
}
/// The `f32` values of BF16 bytes.
fn from_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| f32::from_bits(u32::from(u16::from_le_bytes(*chunk)) << 16))
        .collect()
}

#[test]
fn a_weight_normalized_convolution_is_computed_once_at_load() {
    let dir = tempfile::tempdir().unwrap();
    shard(
        dir.path(),
        "model.safetensors",
        &[
            // Two output channels of two elements: v = [[3, 4], [0, 1]].
            ("conv.weight_v", vec![2, 2], bf16(&[3.0, 4.0, 0.0, 1.0])),
            // PyTorch keeps the gain as [channels, 1, 1].
            ("conv.weight_g", vec![2, 1, 1], bf16(&[5.0, 10.0])),
        ],
    );
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let mut store = Recording::default();
    let weight = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .weight_norm("conv.weight", "conv.weight_v", "conv.weight_g")
            .unwrap()
    };

    // v · g / ‖v‖ per channel: [3, 4] · 5/5 and [0, 1] · 10/1.
    let (_, bytes, dtype, shape, _) = &store.published[0];
    assert_eq!(*dtype, DType::BF16);
    assert_eq!(shape.dims(), &[2, 2]);
    assert_eq!(from_bf16(bytes), vec![3.0, 4.0, 0.0, 10.0]);
    assert_eq!(weight.shape().dims(), &[2, 2]);
    assert_eq!(store.published.len(), 1);

    // A second load reuses the retained region.
    let again = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .weight_norm("conv.weight", "conv.weight_v", "conv.weight_g")
            .unwrap()
    };
    assert_eq!(again.view().ptr, weight.view().ptr);
    assert_eq!(store.published.len(), 1);
}

#[test]
fn a_weight_normalized_gain_must_match_the_output_channels() {
    let dir = tempfile::tempdir().unwrap();
    shard(
        dir.path(),
        "model.safetensors",
        &[
            ("conv.weight_v", vec![2, 2], bf16(&[3.0, 4.0, 0.0, 1.0])),
            ("conv.weight_g", vec![3, 1, 1], bf16(&[5.0, 10.0, 1.0])),
        ],
    );
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let mut store = Recording::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    assert!(
        loader
            .weight_norm("conv.weight", "conv.weight_v", "conv.weight_g")
            .is_err()
    );
    assert!(store.published.is_empty());
}

#[test]
fn a_strided_row_parallel_shard_gathers_its_columns() {
    let dir = tempfile::tempdir().unwrap();
    // Four rows of six elements: row r holds 6r .. 6r+5.
    let values: Vec<f32> = (0..24).map(|value| value as f32).collect();
    shard(
        dir.path(),
        "model.safetensors",
        &[("o_proj.weight", vec![4, 6], bf16(&values))],
    );
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let mut store = Recording::default();

    let request = WeightRequest::strided("o_proj.weight", 0, Shape::new(&[4, 3]), 6);
    let weight = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader.load(&request).unwrap()
    };
    assert_eq!(weight.shape().dims(), &[4, 3]);
    let (_, bytes, _, _, _) = &store.published[0];
    assert_eq!(
        from_bf16(bytes),
        vec![
            0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 12.0, 13.0, 14.0, 18.0, 19.0, 20.0
        ],
        "columns 0..2 of every row, rows contiguous"
    );

    // The same request reuses the retained region.
    {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader.load(&request).unwrap();
    }
    assert_eq!(store.published.len(), 1);

    // A second column block of the same extents is another key.
    {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .load(&WeightRequest::strided(
                "o_proj.weight",
                3,
                Shape::new(&[4, 3]),
                6,
            ))
            .unwrap();
    }
    assert_eq!(store.published.len(), 2);
    assert_ne!(store.published[0].0, store.published[1].0);
}

/// A calibrated NVFP4 matrix is its two byte fields and its tensor scale, published under keys that
/// name the scale layout; nothing is quantized at load.
#[test]
fn a_calibrated_nvfp4_matrix_is_loaded_as_its_three_fields() {
    let directory = tempfile::tempdir().unwrap();
    let directory = directory.path();
    // `[4, 64]` encoded: 32 bytes a row of packed pairs, four block scales a row.
    let values: Vec<u8> = (0..128).map(|byte| byte as u8).collect();
    let scales: Vec<u8> = (0..16).map(|byte| 0x30 + byte).collect();
    shard_typed(
        directory,
        "model.safetensors",
        &[
            ("w", Dtype::U8, vec![4, 32], values.clone()),
            ("w_scale", Dtype::U8, vec![4, 4], scales.clone()),
        ],
    );

    let checkpoint = Checkpoint::open(directory).unwrap();
    let mut store = Recording::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let quantized = loader
        .nvfp4(
            &WeightRequest::region("w", 0, Shape::new(&[4, 32])),
            &WeightRequest::region("w_scale", 0, Shape::new(&[4, 4])),
            ops::fp4::ScaleLayout::Linear,
            0.25,
        )
        .unwrap();

    assert_eq!(quantized.packed.shape().dims(), &[4, 32]);
    assert_eq!(quantized.packed.dtype(), DType::U8);
    assert_eq!(quantized.scales.shape().dims(), &[4, 4]);
    assert_eq!(quantized.scales.dtype(), DType::U8);
    assert_eq!(quantized.global_scale, 0.25);
    assert_eq!(store.published.len(), 2);
    assert_eq!(store.published[0].1, values);
    assert_eq!(store.published[1].1, scales);
    // The layout names the arrangement, so the two fields never share a key with a plain region.
    assert_eq!(store.published[0].0.layout, WeightLayout::NVFP4_VALUES);
    assert_eq!(
        store.published[1].0.layout,
        WeightLayout::NVFP4_SCALES_LINEAR
    );
}

/// The block scales are the shape the requested layout names; another layout's shape is refused
/// rather than reinterpreted, because the two arrangements index the same bytes differently.
#[test]
fn the_block_scales_must_have_the_layout_s_shape() {
    let directory = tempfile::tempdir().unwrap();
    let directory = directory.path();
    let values: Vec<u8> = vec![0; 128];
    let scales: Vec<u8> = vec![0x38; 16];
    shard_typed(
        directory,
        "model.safetensors",
        &[
            ("w", Dtype::U8, vec![4, 32], values),
            ("w_scale", Dtype::U8, vec![4, 4], scales),
        ],
    );

    let checkpoint = Checkpoint::open(directory).unwrap();
    let mut store = Recording::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let error = loader
        .nvfp4(
            &WeightRequest::region("w", 0, Shape::new(&[4, 32])),
            &WeightRequest::region("w_scale", 0, Shape::new(&[4, 4])),
            ops::fp4::ScaleLayout::Swizzled128X4,
            0.25,
        )
        .expect_err("a linear scale field is not the swizzled layout's field");
    assert!(format!("{error}").contains("swizzled_128x4"), "{error}");
    assert!(store.published.is_empty());
}

/// An expert stack is its experts' encoded tensors concatenated in the operator's reading order —
/// expert-major, up before gate — and its tensor scales become one reciprocal per expert, because a
/// stack's scale varies per expert where a single matrix has one.
#[test]
fn a_calibrated_expert_stack_is_joined_expert_major_and_projection_by_projection() {
    let directory = tempfile::tempdir().unwrap();
    let directory = directory.path();
    // Two experts, `[2, 64]` packed (32 bytes a row), two projections each, four block scales a row.
    let mut tensors: Vec<(String, Dtype, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut expected = Vec::new();
    for expert in 0..2u8 {
        for (index, projection) in ["up", "gate"].iter().enumerate() {
            let stem = format!("experts.{expert}.{projection}_proj");
            let values: Vec<u8> = (0..64)
                .map(|byte| byte as u8 + expert * 64 + index as u8)
                .collect();
            let scales: Vec<u8> = (0..8)
                .map(|byte| 0x20 + byte as u8 + expert * 8 + index as u8)
                .collect();
            expected.extend_from_slice(&values);
            tensors.push((format!("{stem}.weight"), Dtype::U8, vec![2, 32], values));
            tensors.push((
                format!("{stem}.weight_scale"),
                Dtype::U8,
                vec![2, 4],
                scales.clone(),
            ));
            tensors.push((
                format!("{stem}.weight_scale_2"),
                Dtype::F32,
                vec![1],
                (0.5f32 * (expert + 1) as f32).to_le_bytes().to_vec(),
            ));
        }
    }
    let views: Vec<(&str, Dtype, Vec<u64>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| (name.as_str(), *dtype, shape.clone(), bytes.clone()))
        .collect();
    shard_typed(directory, "model.safetensors", &views);

    let checkpoint = Checkpoint::open(directory).unwrap();
    let mut store = Recording::default();
    let stack = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .nvfp4_stack(
                "layer.0.mlp.experts.up_gate",
                "experts",
                &["up", "gate"],
                2,
                ops::fp4::ScaleLayout::Linear,
            )
            .unwrap()
    };

    assert_eq!(stack.values.shape().dims(), &[2, 4, 32]);
    assert_eq!(stack.values.dtype(), DType::U8);
    assert_eq!(stack.scales.shape().dims(), &[2, 4, 4]);
    assert_eq!(stack.global_scale.shape().dims(), &[2]);
    assert_eq!(stack.global_scale.dtype(), DType::F32);
    // The store received the values in reading order, then the scales, then the reciprocals.
    assert_eq!(store.published.len(), 3);
    assert_eq!(store.published[0].1, expected);
    assert_eq!(store.published[0].0.layout, WeightLayout::NVFP4_VALUES);
    assert_eq!(store.published[2].0.layout, WeightLayout::ROW_MAJOR);
    let reciprocals: Vec<f32> = store.published[2]
        .1
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    assert_eq!(reciprocals, [2.0, 1.0], "1 / (0.5 · (e + 1))");

    // A second load of the same checkpoint publishes nothing: the three keys hit the store.
    let again = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .nvfp4_stack(
                "layer.0.mlp.experts.up_gate",
                "experts",
                &["up", "gate"],
                2,
                ops::fp4::ScaleLayout::Linear,
            )
            .unwrap()
    };
    assert_eq!(again.values.shape().dims(), stack.values.shape().dims());
    assert_eq!(store.published.len(), 3, "the retained regions were reused");
}

/// A stack whose experts disagree on their geometry is refused rather than read as one stack.
#[test]
fn a_stack_needs_every_expert_to_agree_on_its_geometry() {
    let directory = tempfile::tempdir().unwrap();
    let directory = directory.path();
    let byte = |values: usize| vec![0u8; values];
    shard_typed(
        directory,
        "model.safetensors",
        &[
            ("experts.0.up_proj.weight", Dtype::U8, vec![2, 32], byte(64)),
            (
                "experts.0.up_proj.weight_scale",
                Dtype::U8,
                vec![2, 4],
                byte(8),
            ),
            (
                "experts.0.up_proj.weight_scale_2",
                Dtype::F32,
                vec![1],
                byte(4),
            ),
            (
                "experts.1.up_proj.weight",
                Dtype::U8,
                vec![4, 32],
                byte(128),
            ),
            (
                "experts.1.up_proj.weight_scale",
                Dtype::U8,
                vec![4, 4],
                byte(16),
            ),
            (
                "experts.1.up_proj.weight_scale_2",
                Dtype::F32,
                vec![1],
                byte(4),
            ),
        ],
    );
    let checkpoint = Checkpoint::open(directory).unwrap();
    let mut store = Recording::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let error = loader
        .nvfp4_stack(
            "layer.0.mlp.experts.up",
            "experts",
            &["up"],
            2,
            ops::fp4::ScaleLayout::Linear,
        )
        .expect_err("two experts of different shapes are not one stack");
    assert!(
        format!("{error}").contains("against the stack's"),
        "{error}"
    );
    assert!(store.published.is_empty());
}

/// A pipeline's components keep their indexes apart; one loader reborrows another checkpoint and
/// keeps the store, so the two components share the conversion path and its retained weights.
#[test]
fn a_loader_borrows_another_component_s_checkpoint_and_keeps_the_store() {
    let root = tempfile::tempdir().unwrap();
    let transformer = root.path().join("transformer");
    let vae = root.path().join("vae");
    std::fs::create_dir_all(&transformer).unwrap();
    std::fs::create_dir_all(&vae).unwrap();
    let scale = bf16(&[1.0, 2.0]);
    shard(
        &transformer,
        "model.safetensors",
        &[("weight", vec![2], scale.clone())],
    );
    shard(&vae, "model.safetensors", &[("weight", vec![2], scale)]);

    let transformer = Checkpoint::open(&transformer).unwrap();
    let vae = Checkpoint::open(&vae).unwrap();
    let mut store = Recording::default();
    let (first_ptr, again_ptr) = {
        let mut loader = WeightLoader::new(&transformer, &mut store, Some(DType::F32), None);
        let first = loader
            .load(&WeightRequest::region("weight", 0, Shape::new(&[2])))
            .unwrap();
        assert_eq!(first.dtype(), DType::F32);
        {
            let mut vae_loader = loader.reborrow(&vae).unwrap();
            let second = vae_loader
                .load(&WeightRequest::region("weight", 0, Shape::new(&[2])))
                .unwrap();
            assert_eq!(second.dtype(), DType::F32);
        }
        // Read again through the first loader: the store answers, nothing is published a third
        // time.
        let again = loader
            .load(&WeightRequest::region("weight", 0, Shape::new(&[2])))
            .unwrap();
        (first.view().ptr, again.view().ptr)
    };

    // Same tensor name, same bytes, two components: the keys differ by the file that holds them,
    // and both are retained by the one store.
    assert_eq!(store.published.len(), 2);
    assert_ne!(store.published[0].0.files, store.published[1].0.files);
    assert_eq!(store.published[0].1, store.published[1].1);
    assert_eq!(again_ptr, first_ptr);
}

/// A pipeline root holds one component per subdirectory, each with its own index and its own file
/// identities, and one loader reads them all through one store.
#[test]
fn a_pipeline_root_holds_its_components() {
    let root = tempfile::tempdir().unwrap();
    let transformer = root.path().join("transformer");
    let vae = root.path().join("vae");
    std::fs::create_dir_all(&transformer).unwrap();
    std::fs::create_dir_all(&vae).unwrap();
    shard(
        &transformer,
        "model.safetensors",
        &[("weight", vec![2], bf16(&[1.0, 2.0]))],
    );
    shard(
        &vae,
        "model.safetensors",
        &[("weight", vec![2], bf16(&[3.0, 4.0]))],
    );
    // A file that is not a shard set is not a component.
    std::fs::write(root.path().join("config.json"), b"{}").unwrap();

    let pipeline = Checkpoint::open(root.path()).unwrap();
    assert!(pipeline.is_pipeline());
    assert_eq!(
        pipeline.components().collect::<Vec<_>>(),
        vec!["transformer", "vae"]
    );

    let transformer = pipeline.component("transformer").unwrap();
    let vae = pipeline.component("vae").unwrap();
    let transformer_component = pipeline.component("transformer").unwrap();
    assert_eq!(transformer.read("weight").unwrap().bytes, bf16(&[1.0, 2.0]));
    assert_eq!(vae.read("weight").unwrap().bytes, bf16(&[3.0, 4.0]));
    // The identities name the file each component's tensors came from, not the root's directory.
    assert_ne!(
        transformer.identity("weight").unwrap(),
        vae.identity("weight").unwrap()
    );
    // The root itself holds no tensor: a component's tensors are read from that component.
    assert!(pipeline.tensor("weight").is_none());
    assert!(pipeline.component("audio_vae").is_err());

    // The loader's policy is kept across the components, and a policy the component does not
    // describe is refused rather than read as something else.
    let mut store = Recording::default();
    let mut loader = WeightLoader::new(vae, &mut store, None, None);
    assert!(
        loader.reborrow(transformer_component).is_ok(),
        "the same policy, another component"
    );
    let mut fp8 = WeightLoader::new(vae, &mut store, None, Some(nn::Quantization::Fp8));
    let error = match fp8.reborrow(transformer_component) {
        Ok(_) => panic!("a BF16 component is not an FP8 checkpoint"),
        Err(error) => error,
    };
    assert!(format!("{error}").contains("fp8"), "{error}");
}
