//! The loading boundary's behaviour: a checkpoint read through its index, its file identities, and
//! the loader's conversion, fusion and key reuse.
//!
//! These tests write their own shards — the header and the payload, by hand — so they need no
//! checkpoint and no device: a weight is published to a stub store that records what it was given.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]
use std::path::Path;

use nn::weights::{Nvfp4Names, RetainedWeights, WeightRequest};
use nn::{Checkpoint, DType, DevicePtr, Shape, WeightKey, WeightLayout, WeightLoader};

/// The bytes of `values` as BF16.
fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// The format's name for an element type: the header states the name, not a number.
fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "U8",
        DType::U32 => "U32",
        DType::I16 => "I16",
        DType::I32 => "I32",
        DType::I64 => "I64",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::F32 => "F32",
        DType::F64 => "F64",
        DType::F8E4M3 => "F8_E4M3",
        DType::F4 => "F4_E2M1",
        other => panic!("the fixtures write no {other:?} tensor"),
    }
}

/// Write one shard holding `tensors` — name, element type, extents, bytes — and answer the path.
///
/// The header is written here rather than by a crate: an encoded expert's fields are not all bytes
/// (its packed FP4 values and block scales are one byte each, its tensor scale an FP32 scalar), and
/// a calibrated export's block scales are `F8_E4M3`, which the safetensors crate's own `DType`
/// cannot express — the same gap the library's reader exists to close. A shape and a byte count that
/// disagree are a bug in the fixture, so they are refused here rather than in the loader.
fn shard_typed(
    dir: &Path,
    name: &str,
    tensors: &[(&str, DType, Vec<u64>, Vec<u8>)],
) -> std::path::PathBuf {
    let path = dir.join(name);
    let mut header = String::from("{");
    let mut payload: Vec<u8> = Vec::new();
    for (tensor, dtype, shape, bytes) in tensors {
        let expected: u64 = shape.iter().product::<u64>() * ops::abi::bytes(*dtype).unwrap();
        assert_eq!(
            expected,
            bytes.len() as u64,
            "the fixture's `{tensor}` says {shape:?} of {dtype}, which is {expected} bytes, not {}",
            bytes.len(),
            dtype = ops::abi::name(*dtype)
        );
        let start = payload.len() as u64;
        payload.extend_from_slice(bytes);
        let extents: Vec<String> = shape.iter().map(u64::to_string).collect();
        header.push_str(&format!(
            r#""{tensor}":{{"dtype":"{}","shape":[{}],"data_offsets":[{start},{}]}},"#,
            dtype_name(*dtype),
            extents.join(","),
            payload.len()
        ));
    }
    header.pop();
    header.push('}');
    shard_with_header(dir, name, &header, &payload);
    path
}

/// Write one shard holding BF16 tensors and answer the path.
fn shard(dir: &Path, name: &str, tensors: &[(&str, Vec<u64>, Vec<u8>)]) -> std::path::PathBuf {
    let typed: Vec<(&str, DType, Vec<u64>, Vec<u8>)> = tensors
        .iter()
        .map(|(tensor, shape, bytes)| (*tensor, DType::BF16, shape.clone(), bytes.clone()))
        .collect();
    shard_typed(dir, name, &typed)
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

/// A computed weight is named by what computed it, not by the label its model chose.
///
/// Two deployments that both call their fused output heads
/// `denoiser.video.student_heads.weight` and compute different numbers are two weights. The label
/// and the extent are identical, so a key holding only those makes the second deployment map the
/// first one's bytes - which is the silent cross-deployment corruption this test exists against.
#[test]
fn a_computed_weight_is_named_by_its_provenance_and_not_by_its_label_alone() {
    let dir = tempfile::tempdir().unwrap();
    shard(
        dir.path(),
        "model.safetensors",
        &[("weight", vec![1], bf16(&[1.0]))],
    );
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let mut store = Recording::default();
    let shape = Shape::new(&[2]);
    let name = "denoiser.video.student_heads.weight";
    let base: Vec<u8> = [0.5_f32, 0.25]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let reference: Vec<u8> = [1.0_f32, 2.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();

    let first = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .constant(name, DType::F32, shape, &base, "transformer")
            .unwrap()
    };
    let second = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .constant(name, DType::F32, shape, &reference, "transformer_ref")
            .unwrap()
    };
    assert_ne!(
        store.published[0].0, store.published[1].0,
        "two producers computing different tables under one label are two keys"
    );
    assert_ne!(store.published[0].1, store.published[1].1);
    assert_ne!(first.view().ptr, second.view().ptr);
    assert!(store.published[0].0.to_string().contains("computed:"));

    // The same numbers by the same class are one weight: a restart after a key change maps what
    // the first load published instead of computing it again.
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    loader
        .constant(name, DType::F32, shape, &base, "transformer")
        .unwrap();
    assert_eq!(store.published.len(), 2);

    // The same numbers by another class are another weight: the producer is part of the
    // provenance even when the bytes agree, because nothing else states which interpretation the
    // table belongs to.
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    loader
        .constant(name, DType::F32, shape, &base, "transformer_ref")
        .unwrap();
    assert_eq!(store.published.len(), 3);
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
            .constant(
                "rotary.inv_freq",
                DType::F32,
                Shape::new(&[2]),
                &table,
                "test",
            )
            .unwrap()
    };
    let second = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .constant(
                "rotary.inv_freq",
                DType::F32,
                Shape::new(&[2]),
                &table,
                "test",
            )
            .unwrap()
    };
    assert_eq!(store.published.len(), 1);
    assert_eq!(first.view().ptr, second.view().ptr);
    assert!(store.published[0].0.files.is_empty());

    // The same bytes under another name are another constant.
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    loader
        .constant(
            "other.inv_freq",
            DType::F32,
            Shape::new(&[2]),
            &table,
            "test",
        )
        .unwrap();
    assert_eq!(store.published.len(), 2);

    // A byte count that does not match the extents is refused before anything is published.
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    assert!(
        loader
            .constant(
                "short.inv_freq",
                DType::F32,
                Shape::new(&[2]),
                &table[..4],
                "test"
            )
            .is_err()
    );
    assert_eq!(store.published.len(), 2);
}

#[test]
fn a_scope_names_its_constants_by_its_module_path_and_encodes_their_type() {
    let dir = tempfile::tempdir().unwrap();
    shard(
        dir.path(),
        "model.safetensors",
        &[("weight", vec![1], bf16(&[1.0]))],
    );
    let checkpoint = Checkpoint::open(dir.path()).unwrap();
    let mut store = Recording::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let mut scope = loader.scope("model.rotary_emb");

    let table = scope
        .constant("inv_freq", DType::BF16, &[2], &[0.5, 0.25])
        .unwrap();
    assert_eq!(table.dtype(), DType::BF16);
    let rows = scope.indices("rows", 3, 2).unwrap();
    assert_eq!(rows.dtype(), DType::I32);
    scope.zeros("pad", DType::F32, &[1, 3]).unwrap();
    // A value count that is not the shape's is refused, as is a range that leaves I32.
    assert!(scope.constant_f32("short", &[3], &[1.0]).is_err());
    assert!(scope.indices("far", i32::MAX, 2).is_err());

    let published: Vec<(&str, &[u8])> = store
        .published
        .iter()
        .map(|(key, bytes, ..)| (key.region.tensors[0].as_str(), bytes.as_slice()))
        .collect();
    let indices: Vec<u8> = [3_i32, 4]
        .iter()
        .flat_map(|row| row.to_ne_bytes())
        .collect();
    assert_eq!(
        published,
        [
            ("model.rotary_emb.inv_freq", bf16(&[0.5, 0.25]).as_slice()),
            ("model.rotary_emb.rows", indices.as_slice()),
            ("model.rotary_emb.pad", [0_u8; 12].as_slice()),
        ]
    );
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
            ("w", DType::U8, vec![4, 32], values.clone()),
            ("w_scale", DType::U8, vec![4, 4], scales.clone()),
        ],
    );

    let checkpoint = Checkpoint::open(directory).unwrap();
    let mut store = Recording::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let quantized = loader
        .nvfp4(
            &WeightRequest::region("w", 0, Shape::new(&[4, 32])),
            &WeightRequest::region("w_scale", 0, Shape::new(&[4, 4])),
            None,
            Nvfp4Names::vllm(),
            nn::ScaleLayouts {
                file: ops::fp4::ScaleLayout::Linear,
                product: ops::fp4::ScaleLayout::Linear,
            },
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
            ("w", DType::U8, vec![4, 32], values),
            ("w_scale", DType::U8, vec![4, 4], scales),
        ],
    );

    let checkpoint = Checkpoint::open(directory).unwrap();
    let mut store = Recording::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let error = loader
        .nvfp4(
            &WeightRequest::region("w", 0, Shape::new(&[4, 32])),
            &WeightRequest::region("w_scale", 0, Shape::new(&[4, 4])),
            None,
            Nvfp4Names::vllm(),
            nn::ScaleLayouts {
                file: ops::fp4::ScaleLayout::Swizzled128X4,
                product: ops::fp4::ScaleLayout::Swizzled128X4,
            },
            0.25,
        )
        .expect_err("a linear scale field is not the swizzled layout's field");
    assert!(format!("{error}").contains("swizzled_128x4"), "{error}");
    assert!(store.published.is_empty());
}

/// An expert stack is its experts' encoded tensors concatenated in the operator's reading order —
/// expert-major, up before gate — and its tensor scales become one reciprocal per expert **and
/// projection**, because the checkpoint gives each expert's up and gate their own `weight_scale_2`.
/// The scales are published the other way round: one packed table per projection, which is what the
/// batched-GEMM entry reads and what the checkpoint itself says a tensor at a time.
#[test]
fn a_calibrated_expert_stack_is_joined_expert_major_and_projection_by_projection() {
    let directory = tempfile::tempdir().unwrap();
    let directory = directory.path();
    // Two experts, `[2, 64]` packed (32 bytes a row), two projections each, four block scales a row.
    let mut tensors: Vec<(String, DType, Vec<u64>, Vec<u8>)> = Vec::new();
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
            tensors.push((format!("{stem}.weight"), DType::U8, vec![2, 32], values));
            tensors.push((
                format!("{stem}.weight_scale"),
                DType::U8,
                vec![2, 4],
                scales.clone(),
            ));
            tensors.push((
                format!("{stem}.weight_scale_2"),
                DType::F32,
                vec![1],
                (0.5f32 * (expert + 1) as f32).to_le_bytes().to_vec(),
            ));
        }
    }
    let views: Vec<(&str, DType, Vec<u64>, Vec<u8>)> = tensors
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
                nn::ExpertStack {
                    prefix: "experts",
                    projections: &["up_proj", "gate_proj"],
                    slots: &(0..2).collect::<Vec<u32>>(),
                    vocabulary: 2,
                },
                Nvfp4Names::vllm(),
                nn::StackLayout {
                    values: nn::ValueLayout::Plain,
                    scales: ops::fp4::ScaleLayout::Linear,
                },
                false,
            )
            .map(|(stack, _)| stack)
            .unwrap()
    };

    assert_eq!(stack.values.shape().dims(), &[2, 4, 32]);
    assert_eq!(stack.values.dtype(), DType::U8);
    assert_eq!(stack.scales.shape().dims(), &[2, 4, 4]);
    assert_eq!(
        stack.global_scale.shape().dims(),
        &[2, 2],
        "one packed table per projection, of two experts each"
    );
    assert_eq!(stack.global_scale.dtype(), DType::F32);
    // The store received the values in reading order, then the scales, then the tensor scales, then
    // the stack's encode scale.
    assert_eq!(store.published.len(), 4);
    assert_eq!(store.published[0].1, expected);
    assert_eq!(store.published[0].0.layout, WeightLayout::NVFP4_VALUES);
    assert_eq!(store.published[2].0.layout, WeightLayout::ROW_MAJOR);
    let tensor_scales: Vec<f32> = store.published[2]
        .1
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    assert_eq!(
        tensor_scales,
        [0.5, 1.0, 0.5, 1.0],
        "0.5 · (e + 1) as one packed row per projection: up's two experts, then gate's"
    );
    // The neutral encode scale is one value per stack, published so the encoder reads a device
    // operand: a stack that encodes nothing multiplies by one.
    let (key, bytes, dtype, shape, _) = &store.published[3];
    assert_eq!(key.layout, WeightLayout::ROW_MAJOR);
    assert_eq!(*dtype, DType::F32);
    assert_eq!(shape.dims(), &[1]);
    assert_eq!(bytes, &1.0_f32.to_le_bytes());

    // A second load of the same checkpoint publishes nothing: the four keys hit the store.
    let again = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .nvfp4_stack(
                "layer.0.mlp.experts.up_gate",
                nn::ExpertStack {
                    prefix: "experts",
                    projections: &["up_proj", "gate_proj"],
                    slots: &(0..2).collect::<Vec<u32>>(),
                    vocabulary: 2,
                },
                Nvfp4Names::vllm(),
                nn::StackLayout {
                    values: nn::ValueLayout::Plain,
                    scales: ops::fp4::ScaleLayout::Linear,
                },
                false,
            )
            .map(|(stack, _)| stack)
            .unwrap()
    };
    assert_eq!(again.values.shape().dims(), stack.values.shape().dims());
    assert_eq!(store.published.len(), 4, "the retained regions were reused");
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
            ("experts.0.up_proj.weight", DType::U8, vec![2, 32], byte(64)),
            (
                "experts.0.up_proj.weight_scale",
                DType::U8,
                vec![2, 4],
                byte(8),
            ),
            (
                "experts.0.up_proj.weight_scale_2",
                DType::F32,
                vec![1],
                1.0f32.to_le_bytes().to_vec(),
            ),
            (
                "experts.1.up_proj.weight",
                DType::U8,
                vec![4, 32],
                byte(128),
            ),
            (
                "experts.1.up_proj.weight_scale",
                DType::U8,
                vec![4, 4],
                byte(16),
            ),
            (
                "experts.1.up_proj.weight_scale_2",
                DType::F32,
                vec![1],
                1.0f32.to_le_bytes().to_vec(),
            ),
        ],
    );
    let checkpoint = Checkpoint::open(directory).unwrap();
    let mut store = Recording::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let error = loader
        .nvfp4_stack(
            "layer.0.mlp.experts.up",
            nn::ExpertStack {
                prefix: "experts",
                projections: &["up_proj"],
                slots: &(0..2).collect::<Vec<u32>>(),
                vocabulary: 2,
            },
            Nvfp4Names::vllm(),
            nn::StackLayout {
                values: nn::ValueLayout::Plain,
                scales: ops::fp4::ScaleLayout::Linear,
            },
            false,
        )
        .map(|(stack, _)| stack)
        .expect_err("two experts of different shapes are not one stack");
    assert!(
        format!("{error}").contains("against the stack's"),
        "{error}"
    );
    assert!(store.published.is_empty());
}

/// A pipeline's stages keep their indexes apart; one loader reborrows another checkpoint and
/// keeps the store, so the two stages share the conversion path and its retained weights.
#[test]
fn a_loader_borrows_another_stage_s_checkpoint_and_keeps_the_store() {
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

    // Same tensor name, same bytes, two stages: the keys differ by the file that holds them,
    // and both are retained by the one store.
    assert_eq!(store.published.len(), 2);
    assert_ne!(store.published[0].0.files, store.published[1].0.files);
    assert_eq!(store.published[0].1, store.published[1].1);
    assert_eq!(again_ptr, first_ptr);
}

/// A pipeline root holds one stage per subdirectory, each with its own index and its own file
/// identities, and one loader reads them all through one store.
#[test]
fn a_pipeline_root_holds_its_stages() {
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
    // A file that is not a shard set is not a stage.
    std::fs::write(root.path().join("config.json"), b"{}").unwrap();

    let pipeline = Checkpoint::open(root.path()).unwrap();
    assert!(pipeline.is_pipeline());
    assert_eq!(
        pipeline.stages().collect::<Vec<_>>(),
        vec!["transformer", "vae"]
    );

    let transformer = pipeline.stage("transformer").unwrap();
    let vae = pipeline.stage("vae").unwrap();
    let transformer_stage = pipeline.stage("transformer").unwrap();
    assert_eq!(transformer.read("weight").unwrap().bytes, bf16(&[1.0, 2.0]));
    assert_eq!(vae.read("weight").unwrap().bytes, bf16(&[3.0, 4.0]));
    // The identities name the file each stage's tensors came from, not the root's directory.
    assert_ne!(
        transformer.identity("weight").unwrap(),
        vae.identity("weight").unwrap()
    );
    // The root itself holds no tensor: a stage's tensors are read from that stage.
    assert!(pipeline.tensor("weight").is_none());
    assert!(pipeline.stage("audio_vae").is_err());

    // The loader's policy is kept across the stages, and a policy the stage does not
    // describe is refused rather than read as something else.
    let mut store = Recording::default();
    let mut loader = WeightLoader::new(vae, &mut store, None, None);
    assert!(
        loader.reborrow(transformer_stage).is_ok(),
        "the same policy, another stage"
    );
    let mut fp8 = WeightLoader::new(vae, &mut store, None, Some(nn::Quantization::Fp8));
    let error = match fp8.reborrow(transformer_stage) {
        Ok(_) => panic!("a BF16 stage is not an FP8 checkpoint"),
        Err(error) => error,
    };
    assert!(format!("{error}").contains("fp8"), "{error}");
}

/// An export over a base is one checkpoint whose file set spans both: the union is per tensor, the
/// export's own tensors shadow the base's, a stage only the base carries still answers, and a
/// tensor only the base carries inside a unioned stage comes from the base.
#[test]
fn an_export_over_a_base_shadows_per_tensor_across_roots() {
    let root = tempfile::tempdir().unwrap();
    let root = root.path();
    let base = root.join("base");
    let export = root.join("export");
    for (dir, files) in [
        (&base, vec!["transformer", "text_encoder"]),
        (&export, vec!["transformer"]),
    ] {
        for stage in files {
            std::fs::create_dir_all(dir.join(stage)).unwrap();
        }
    }
    // The base: `transformer` with two tensors, and a `text_encoder` the export does not carry.
    shard(
        &base.join("transformer"),
        "model.safetensors",
        &[
            ("shared", vec![2], bf16(&[1.0, 2.0])),
            ("base_only", vec![1], bf16(&[3.0])),
        ],
    );
    shard(
        &base.join("text_encoder"),
        "model.safetensors",
        &[("weight", vec![1], bf16(&[4.0]))],
    );
    // The export overrides `shared` and says nothing about the other two.
    shard(
        &export.join("transformer"),
        "model.safetensors",
        &[("shared", vec![2], bf16(&[9.0, 9.0]))],
    );

    let unioned = Checkpoint::open_roots(&[&export, &base]).unwrap();

    // The export's root is the one a log names.
    assert_eq!(unioned.root(), export.as_path());
    // The override wins, and it is the export's *bytes*, not a duplicate-key refusal.
    let transformer = unioned.stage("transformer").unwrap();
    assert_eq!(
        transformer.read("shared").unwrap().bytes,
        bf16(&[9.0, 9.0]),
        "the export shadows the base's tensor of the same name"
    );
    // The tensor only the base carries inside the unioned stage comes from the base.
    assert_eq!(
        transformer.read("base_only").unwrap().bytes,
        bf16(&[3.0]),
        "a unioned stage is not all-or-nothing"
    );
    // A stage the export does not carry is the base's.
    assert_eq!(
        unioned
            .stage("text_encoder")
            .unwrap()
            .read("weight")
            .unwrap()
            .bytes,
        bf16(&[4.0]),
        "a stage only the base carries still answers"
    );
    assert_eq!(
        unioned.stages().collect::<Vec<_>>(),
        ["text_encoder", "transformer"]
    );
    // One root is what `open` already means.
    let single = Checkpoint::open_roots(&[&export]).unwrap();
    assert_eq!(single.root(), export.as_path());
    assert!(single.stage("text_encoder").is_err());
}

/// A name neither root holds is refused with the tensor and the roots, in precedence order: in a
/// union spanning an export and a base, a bare "not found" is the report that costs an hour.
#[test]
fn a_missing_tensor_in_a_union_names_the_roots_it_searched() {
    let root = tempfile::tempdir().unwrap();
    let root = root.path();
    let base = root.join("base");
    let export = root.join("export");
    for dir in [&base, &export] {
        std::fs::create_dir_all(dir.join("transformer")).unwrap();
        shard(
            &dir.join("transformer"),
            "model.safetensors",
            &[("weight", vec![1], bf16(&[1.0]))],
        );
    }

    let unioned = Checkpoint::open_roots(&[&export, &base]).unwrap();
    let text = match unioned.read("layers.0.mlp.experts.0.up_proj.weight") {
        Ok(_) => panic!("neither root holds that tensor"),
        Err(error) => format!("{error}"),
    };
    assert!(
        text.contains("layers.0.mlp.experts.0.up_proj.weight"),
        "{text}"
    );
    let export_at = text.find("export").expect("the export is named");
    let base_at = text.find("base").expect("the base is named");
    assert!(
        export_at < base_at,
        "precedence order is the export's first: {text}"
    );
}

/// Write a shard whose header is written by hand, so the test can name a dtype the safetensors
/// crate's own `DType` cannot express — which is the whole reason the library reads headers itself.
fn shard_with_header(dir: &Path, name: &str, header: &str, payload: &[u8]) -> std::path::PathBuf {
    let path = dir.join(name);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(payload);
    std::fs::write(&path, bytes).unwrap();
    path
}

/// A calibrated checkpoint's header names FP8 and FP4 dtypes, which `safetensors` 0.3.3's `DType`
/// has no variants for: the library reads the header itself and maps the format's names onto the
/// types our operators speak, with the offsets the payload actually has.
#[test]
fn a_header_naming_fp8_and_fp4_opens_with_those_types() {
    let root = tempfile::tempdir().unwrap();
    let root = root.path();
    // Four tensors, laid out back to back: 8 bytes of sixteen codes, 4 block scales, 2 bytes, 2 bf16s.
    let payload: Vec<u8> = (0..18).map(|byte| byte as u8).collect();
    let header = concat!(
        r#"{"__metadata__":{"format":"pt"},"#,
        r#""packed":{"dtype":"F4","shape":[2,8],"data_offsets":[0,8]},"#,
        r#""scales":{"dtype":"F8_E4M3","shape":[4],"data_offsets":[8,12]},"#,
        r#""pair":{"dtype":"U8","shape":[2],"data_offsets":[12,14]},"#,
        r#""weight":{"dtype":"BF16","shape":[2],"data_offsets":[14,18]}}"#
    );
    shard_with_header(root, "model.safetensors", header, &payload);

    let checkpoint = Checkpoint::open(root).unwrap();
    let packed = checkpoint.read("packed").unwrap();
    assert_eq!(packed.dtype, DType::F4);
    assert_eq!(packed.shape.dims(), &[2, 8]);
    assert_eq!(packed.bytes, payload[0..8]);
    let scales = checkpoint.read("scales").unwrap();
    assert_eq!(scales.dtype, DType::F8E4M3);
    assert_eq!(scales.bytes, payload[8..12]);
    assert_eq!(checkpoint.read("pair").unwrap().dtype, DType::U8);
    let weight = checkpoint.read("weight").unwrap();
    assert_eq!(weight.dtype, DType::BF16);
    assert_eq!(weight.bytes, payload[14..18]);
}

/// A dtype name the operator ABI has no type for is refused by name at open, not read as something
/// it is not, and neither is an offset the payload cannot hold.
#[test]
fn a_header_the_abi_cannot_express_is_refused_by_name() {
    let root = tempfile::tempdir().unwrap();
    let root = root.path();
    let payload = vec![0u8; 1];
    shard_with_header(
        root,
        "model.safetensors",
        r#"{"flag":{"dtype":"BOOL","shape":[1],"data_offsets":[0,1]}}"#,
        &payload,
    );
    let text = match Checkpoint::open(root) {
        Ok(_) => panic!("BOOL has no operator ABI type"),
        Err(error) => format!("{error}"),
    };
    assert!(text.contains("BOOL"), "{text}");
    assert!(text.contains("flag"), "the tensor is named too: {text}");

    let root = tempfile::tempdir().unwrap();
    let root = root.path();
    shard_with_header(
        root,
        "model.safetensors",
        r#"{"weight":{"dtype":"BF16","shape":[32],"data_offsets":[0,64]}}"#,
        &payload,
    );
    let text = match Checkpoint::open(root) {
        Ok(_) => panic!("the payload is one byte"),
        Err(error) => format!("{error}"),
    };
    assert!(text.contains("not fully covered"), "{text}");
}

/// A real NVFP4 export stores its block scales as `F8_E4M3`, one byte per sixteen elements: the
/// loader publishes exactly that type and does not require `U8`, which was a fixture's habit.
#[test]
fn a_stack_keeps_the_checkpoint_s_e4m3_block_scales() {
    let root = tempfile::tempdir().unwrap();
    let root = root.path();
    let mut payload: Vec<u8> = Vec::new();
    let mut header = String::from("{");
    let mut offset = 0_u64;
    for expert in 0..2 {
        for projection in ["up", "gate"] {
            let stem = format!("experts.{expert}.{projection}_proj");
            for (suffix, dtype, bytes) in [
                (".weight", "U8", 64_u64),
                (".weight_scale", "F8_E4M3", 8),
                (".weight_scale_2", "F32", 4),
            ] {
                payload.extend(std::iter::repeat_n(0x3c_u8, bytes as usize));
                header.push_str(&format!(
                    r#""{stem}{suffix}":{{"dtype":"{dtype}","shape":{},"data_offsets":[{offset},{}]}},"#,
                    match suffix {
                        ".weight" => "[2, 32]".to_owned(),
                        ".weight_scale" => "[2, 4]".to_owned(),
                        _ => "[1]".to_owned(),
                    },
                    offset + bytes,
                ));
                offset += bytes;
                if suffix == ".weight_scale_2" {
                    // A positive tensor scale: the fixture writes `0x3c3c3c3c`, a small positive f32.
                    let start = payload.len() - 4;
                    payload[start..].copy_from_slice(&0.5f32.to_le_bytes());
                }
            }
        }
    }
    header.pop();
    header.push('}');
    shard_with_header(root, "model.safetensors", &header, &payload);

    let checkpoint = Checkpoint::open(root).unwrap();
    let mut store = Recording::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let stack = loader
        .nvfp4_stack(
            "layer.0.mlp.experts.up_gate",
            nn::ExpertStack {
                prefix: "experts",
                projections: &["up_proj", "gate_proj"],
                slots: &(0..2).collect::<Vec<u32>>(),
                vocabulary: 2,
            },
            Nvfp4Names::vllm(),
            nn::StackLayout {
                values: nn::ValueLayout::Plain,
                scales: ops::fp4::ScaleLayout::Linear,
            },
            false,
        )
        .map(|(stack, _)| stack)
        .unwrap();
    assert_eq!(
        stack.scales.dtype(),
        DType::F8E4M3,
        "the checkpoint's scale type is published unchanged"
    );
    assert_eq!(stack.values.dtype(), DType::U8);
    assert_eq!(stack.scales.shape().dims(), &[2, 4, 4]);
    assert_eq!(stack.global_scale.shape().dims(), &[2, 2]);
}

/// A calibrated projection is read in the spelling its export wrote: the same numbers are `weight`
/// and `weight_scale_2` in one toolchain's checkpoint and `weight_packed` and `weight_tensor_scale`
/// in another's, so the vocabulary is the loading boundary's statement about its own checkpoint.
#[test]
fn a_calibrated_projection_is_read_in_the_spelling_its_export_wrote() {
    let directory = tempfile::tempdir().unwrap();
    let directory = directory.path();
    let values: Vec<u8> = (0..128).map(|byte| byte as u8).collect();
    let scales: Vec<u8> = (0..16).map(|byte| 0x30 + byte).collect();
    shard_typed(
        directory,
        "model.safetensors",
        &[
            ("p.up_proj.weight", DType::U8, vec![4, 32], values),
            ("p.up_proj.weight_scale", DType::U8, vec![4, 4], scales),
        ],
    );
    let checkpoint = Checkpoint::open(directory).unwrap();
    let mut store = Recording::default();

    // vLLM's spelling: the packed values are `weight` and the block scales `weight_scale`.
    {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        let quantized = loader
            .nvfp4(
                &WeightRequest::region("p.up_proj.weight", 0, Shape::new(&[4, 32])),
                &WeightRequest::region("p.up_proj.weight_scale", 0, Shape::new(&[4, 4])),
                None,
                Nvfp4Names::vllm(),
                nn::ScaleLayouts {
                    file: ops::fp4::ScaleLayout::Linear,
                    product: ops::fp4::ScaleLayout::Linear,
                },
                0.25,
            )
            .unwrap();
        assert_eq!(quantized.packed.dtype(), DType::U8);
    }

    // The same checkpoint read as the other export's spelling: the name it asks for is the one it
    // says it expected, so a wrong vocabulary is a loading-boundary error and not a silent zero.
    {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        let error = loader
            .nvfp4(
                &WeightRequest::region("p.up_proj.weight_packed", 0, Shape::new(&[4, 32])),
                &WeightRequest::region("p.up_proj.weight_scale", 0, Shape::new(&[4, 4])),
                None,
                Nvfp4Names::diffusers(),
                nn::ScaleLayouts {
                    file: ops::fp4::ScaleLayout::Linear,
                    product: ops::fp4::ScaleLayout::Linear,
                },
                0.25,
            )
            .expect_err("a checkpoint in the other export's spelling");
        let text = format!("{error}");
        assert!(text.contains("weight_packed"), "{text}");
        assert!(text.contains("diffusers ModelOpt"), "{text}");
    }
}

/// A stack names its fields with the export's own suffixes: the same experts read as one stack in
/// each spelling, and the vocabulary is not a property of the layout the scales arrive in.
#[test]
fn a_calibrated_stack_names_its_fields_with_its_exports_suffixes() {
    let directory = tempfile::tempdir().unwrap();
    let directory = directory.path();
    // A `[4, 32]` packed weight is 128 bytes — two E2M1 codes a byte along a contraction of 64 —
    // and its linear block scales are one byte per sixteen elements, sixteen of them.
    let values: Vec<u8> = (0..128).map(|byte| byte as u8).collect();
    let scales: Vec<u8> = (0..16).map(|byte| 0x30 + byte).collect();
    let tensor_scale: Vec<u8> = 1.0_f32.to_le_bytes().to_vec();
    let mut tensors: Vec<(String, DType, Vec<u64>, Vec<u8>)> = Vec::new();
    for expert in 0..2 {
        for (field, dtype, shape, bytes) in [
            ("weight_packed", DType::U8, vec![4, 32], values.clone()),
            ("weight_scale", DType::U8, vec![4, 4], scales.clone()),
            (
                "weight_tensor_scale",
                DType::F32,
                vec![1],
                tensor_scale.clone(),
            ),
        ] {
            tensors.push((
                format!("experts.{expert}.up_proj.{field}"),
                dtype,
                shape,
                bytes,
            ));
        }
    }
    let borrowed: Vec<(&str, DType, Vec<u64>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| (name.as_str(), *dtype, shape.clone(), bytes.clone()))
        .collect();
    shard_typed(directory, "model.safetensors", &borrowed);
    let checkpoint = Checkpoint::open(directory).unwrap();
    let mut store = Recording::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);

    let stack = loader
        .nvfp4_stack(
            "layer.0.mlp.experts.up",
            nn::ExpertStack {
                prefix: "experts",
                projections: &["up_proj"],
                slots: &(0..2).collect::<Vec<u32>>(),
                vocabulary: 2,
            },
            Nvfp4Names::diffusers(),
            nn::StackLayout {
                values: nn::ValueLayout::Plain,
                scales: ops::fp4::ScaleLayout::Linear,
            },
            false,
        )
        .map(|(stack, _)| stack)
        .expect("an export that states weight_packed and weight_tensor_scale loads as such");
    assert_eq!(stack.values.shape().dims(), &[2, 4, 32]);
    assert_eq!(stack.global_scale.shape().dims(), &[1, 2]);

    // Read as the other spelling the stack asks for names this checkpoint does not have, and the
    // refusal names the field and the export it expected.
    let other = {
        let mut store = Recording::default();
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .nvfp4_stack(
                "layer.0.mlp.experts.up",
                nn::ExpertStack {
                    prefix: "experts",
                    projections: &["up_proj"],
                    slots: &(0..2).collect::<Vec<u32>>(),
                    vocabulary: 2,
                },
                Nvfp4Names::vllm(),
                nn::StackLayout {
                    values: nn::ValueLayout::Plain,
                    scales: ops::fp4::ScaleLayout::Linear,
                },
                false,
            )
            .map(|(stack, _)| stack)
            .expect_err("a diffusers checkpoint read as vLLM's spelling")
    };
    let text = format!("{other}");
    assert!(text.contains("experts.0.up_proj.weight"), "{text}");
    assert!(text.contains("vLLM ModelOpt"), "{text}");
}

/// A load that reads part of a checkpoint reports the rest by name, so a wrong number can be traced
/// to the tensors nobody asked for rather than to the ones that loaded. A pipeline qualifies its
/// stages, because two stages may hold one name each.
#[test]
fn a_reading_names_the_tensors_a_load_never_asked_for() {
    let directory = tempfile::tempdir().unwrap();
    shard(
        directory.path(),
        "model.safetensors",
        &[
            ("a", vec![2], bf16(&[1.0, 2.0])),
            ("b", vec![2], bf16(&[3.0, 4.0])),
            ("c", vec![2], bf16(&[5.0, 6.0])),
        ],
    );
    let checkpoint = Checkpoint::open(directory.path()).unwrap();
    checkpoint.read("a").unwrap();
    let reading = checkpoint.reading();
    assert_eq!(reading.read, ["a"]);
    assert_eq!(reading.unread, ["b", "c"]);
    assert!(!reading.is_complete());

    // Reading the rest completes it: the report is the claims' complement, not a snapshot of one
    // moment of the load.
    checkpoint.read("b").unwrap();
    checkpoint.read("c").unwrap();
    assert!(checkpoint.reading().is_complete());

    // A pipeline root answers with its stages' names qualified, and a stage that was not
    // read is reported whole.
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
    let pipeline = Checkpoint::open(root.path()).unwrap();
    pipeline
        .stage("transformer")
        .unwrap()
        .read("weight")
        .unwrap();
    let reading = pipeline.reading();
    assert_eq!(reading.read, ["transformer/weight"]);
    assert_eq!(reading.unread, ["vae/weight"]);
    assert!(!reading.is_complete());
}

/// A shuffled stack's values are the shard's rows permuted inside 32-row blocks — the reference's
/// `srcToDstBlk32RowMap` in closed form — applied once at load, with the published layout naming the
/// arrangement so a product cannot read one for the other.
#[test]
fn a_shuffled_stack_permutes_its_values_rows_at_load() {
    let directory = tempfile::tempdir().unwrap();
    let directory = directory.path();
    // One expert, 64 rows by 32 columns — two 32-row blocks, so the block boundary is covered — with
    // 16 packed bytes and two block scales a row.
    let expert = |prefix: &str| {
        let mut values = Vec::new();
        for row in 0..64u8 {
            values.extend(std::iter::repeat_n(row, 16));
        }
        vec![
            (
                format!("{prefix}.weight_packed"),
                DType::U8,
                vec![64, 16],
                values,
            ),
            (
                format!("{prefix}.weight_scale"),
                DType::F8E4M3,
                vec![64, 2],
                vec![0x38; 128],
            ),
            (
                format!("{prefix}.weight_tensor_scale"),
                DType::F32,
                vec![1],
                0.5f32.to_le_bytes().to_vec(),
            ),
        ]
    };
    let mut tensors = expert("experts.0.up_proj");
    tensors.extend(expert("experts.0.gate_proj"));
    let views: Vec<(&str, DType, Vec<u64>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| (name.as_str(), *dtype, shape.clone(), bytes.clone()))
        .collect();
    shard_typed(directory, "model.safetensors", &views);

    let checkpoint = Checkpoint::open(directory).unwrap();
    let mut store = Recording::default();
    {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .nvfp4_stack(
                "layer.0.mlp.experts.up_gate",
                nn::ExpertStack {
                    prefix: "experts",
                    projections: &["up_proj", "gate_proj"],
                    slots: &(0..1).collect::<Vec<u32>>(),
                    vocabulary: 1,
                },
                Nvfp4Names::diffusers(),
                nn::StackLayout {
                    values: nn::ValueLayout::Shuffled {
                        epilogue_tile_m: 128,
                    },
                    scales: ops::fp4::ScaleLayout::Linear,
                },
                false,
            )
            .map(|(stack, _)| stack)
            .unwrap();
    }
    assert_eq!(
        store.published[0].0.layout,
        WeightLayout::NVFP4_VALUES_SHUFFLED,
        "the published values name the arrangement they are in"
    );

    // Destination row `d` holds source row `(d / 32) · 32 + (d % 8) · 4 + (d % 32) / 8`: within its
    // block the interleave of four eight-row groups, and the block wherever it was.
    let published = &store.published[0].1;
    for destination in 0..64usize {
        let within = destination % 32;
        let source = (destination / 32) * 32 + (within % 8) * 4 + within / 8;
        assert_eq!(
            published[destination * 16],
            source as u8,
            "row {destination} should hold source row {source}"
        );
    }

    // A row count that is not a whole number of 32-row blocks has no such arrangement, and the
    // refusal names both numbers.
    let short = tempfile::tempdir().unwrap();
    let short_directory = short.path();
    let rows = 30u8;
    let mut tensors = Vec::new();
    for projection in ["up_proj", "gate_proj"] {
        let mut values = Vec::new();
        for row in 0..rows {
            values.extend(std::iter::repeat_n(row, 16));
        }
        tensors.push((
            format!("experts.0.{projection}.weight_packed"),
            DType::U8,
            vec![u64::from(rows), 16],
            values,
        ));
        tensors.push((
            format!("experts.0.{projection}.weight_scale"),
            DType::F8E4M3,
            vec![u64::from(rows), 2],
            vec![0x38; usize::from(rows) * 2],
        ));
        tensors.push((
            format!("experts.0.{projection}.weight_tensor_scale"),
            DType::F32,
            vec![1],
            0.5f32.to_le_bytes().to_vec(),
        ));
    }
    let views: Vec<(&str, DType, Vec<u64>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| (name.as_str(), *dtype, shape.clone(), bytes.clone()))
        .collect();
    shard_typed(short_directory, "model.safetensors", &views);
    let short_checkpoint = Checkpoint::open(short_directory).unwrap();
    let error = {
        let mut store = Recording::default();
        let mut loader = WeightLoader::new(&short_checkpoint, &mut store, None, None);
        loader
            .nvfp4_stack(
                "layer.0.mlp.experts.up_gate",
                nn::ExpertStack {
                    prefix: "experts",
                    projections: &["up_proj", "gate_proj"],
                    slots: &(0..1).collect::<Vec<u32>>(),
                    vocabulary: 1,
                },
                Nvfp4Names::diffusers(),
                nn::StackLayout {
                    values: nn::ValueLayout::Shuffled {
                        epilogue_tile_m: 128,
                    },
                    scales: ops::fp4::ScaleLayout::Linear,
                },
                false,
            )
            .map(|(stack, _)| stack)
            .expect_err("a 30-row matrix has no 32-row arrangement")
    };
    assert!(format!("{error}").contains("32-row blocks"), "{error}");
}

/// The products read their weight scales swizzled, and an export holds them linear: the loader
/// builds the 128x4 table once at load — rows padded to 128, scale columns to 4, each tile laid out
/// so a thread block's scales are contiguous — rather than a step paying for it.
#[test]
fn a_swizzled_scale_table_is_built_from_the_shards_linear_one() {
    let directory = tempfile::tempdir().unwrap();
    let directory = directory.path();
    // One expert, 128 rows by 32 columns: sixteen packed bytes and two block scales a row.
    let mut values = Vec::new();
    let mut scales = Vec::new();
    for row in 0..128u8 {
        values.extend(std::iter::repeat_n(row, 16));
        scales.extend_from_slice(&[row, row.wrapping_add(1)]);
    }
    let views: Vec<(&str, DType, Vec<u64>, Vec<u8>)> = vec![
        (
            "experts.0.down_proj.weight_packed",
            DType::U8,
            vec![128, 16],
            values,
        ),
        (
            "experts.0.down_proj.weight_scale",
            DType::F8E4M3,
            vec![128, 2],
            scales,
        ),
        (
            "experts.0.down_proj.weight_tensor_scale",
            DType::F32,
            vec![1],
            0.5f32.to_le_bytes().to_vec(),
        ),
    ];
    shard_typed(directory, "model.safetensors", &views);

    let checkpoint = Checkpoint::open(directory).unwrap();
    let mut store = Recording::default();
    {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        loader
            .nvfp4_stack(
                "layer.0.mlp.experts.down",
                nn::ExpertStack {
                    prefix: "experts",
                    projections: &["down_proj"],
                    slots: &(0..1).collect::<Vec<u32>>(),
                    vocabulary: 1,
                },
                Nvfp4Names::diffusers(),
                nn::StackLayout {
                    values: nn::ValueLayout::Shuffled {
                        epilogue_tile_m: 128,
                    },
                    scales: ops::fp4::ScaleLayout::Swizzled128X4,
                },
                false,
            )
            .map(|(stack, _)| stack)
            .unwrap();
    }
    // Two scale columns pad to four, and 128 rows are one tile: 128 × 4 bytes.
    let (key, bytes, ..) = &store.published[1];
    assert_eq!(key.layout, WeightLayout::NVFP4_SCALES_SWIZZLED);
    assert_eq!(bytes.len(), 128 * 4);
    // A source row moves twice, and the table must show both moves composed. The values are
    // published shuffled, and the scales are read beside the row, so the same arrangement applies to
    // them: within its block of thirty-two rows a source row goes to `(r % 4) * 8 + (r % 32) / 4`
    // — sixteen lanes of eight rows, the reference's `srcToDstBlk32RowMap` — and the artifact's
    // `computeSFIndex` then puts that row's scale group at
    // `(row % 32) * 16 + ((row % 128) / 32) * 4`. Asserting the composite over every row states
    // both formulas at once; asserting a probe would go green on either one alone.
    for source in 0..128_u64 {
        let within = source % 32;
        let row = (source / 32) * 32 + (within % 4) * 8 + within / 4;
        let at = ((row % 32) * 16 + ((row % 128) / 32) * 4) as usize;
        assert_eq!(bytes[at], source as u8, "row {source}'s first scale");
        assert_eq!(
            bytes[at + 1],
            source.wrapping_add(1) as u8,
            "row {source}'s second scale"
        );
    }
    // Columns 2 and 3 of the tile are the padding: no block exists there, and a zero scale on a
    // block that does not exist is what the artifact's own interleave writes.
    for (at, value) in bytes.iter().enumerate() {
        if at % 4 >= 2 {
            assert_eq!(*value, 0, "a padding column is a zero scale at {at}");
        }
    }
}

/// One weight's element type and extents: what a meta load owes a caller.
type Answer = (DType, Vec<u64>);

/// The answer of `weight`.
fn answer(weight: &nn::Weight) -> Answer {
    (weight.dtype(), weight.shape().dims().to_vec())
}

/// A dense checkpoint: a projection, a row-parallel matrix, two fusable parts and a
/// weight-normalized pair, all BF16 with values in every byte.
fn dense_checkpoint(directory: &Path) -> Checkpoint {
    let values = |count: usize| bf16(&(1..=count).map(|value| value as f32).collect::<Vec<_>>());
    shard(
        directory,
        "model.safetensors",
        &[
            ("proj.weight", vec![4, 8], values(32)),
            ("rows.weight", vec![4, 8], values(32)),
            ("experts.0.up", vec![2, 8], values(16)),
            ("experts.0.gate", vec![2, 8], values(16)),
            ("conv.weight_v", vec![2, 3, 1], values(6)),
            ("conv.weight_g", vec![2, 1, 1], values(2)),
        ],
    );
    Checkpoint::open(directory).unwrap()
}

/// Every kind of dense load, in one order: a pinned region, a strided column block, fused parts, a
/// tensor-parallel block of each part, a weight-normalized fold and a computed constant.
fn dense_loads(loader: &mut WeightLoader<'_>) -> Vec<nn::Weight> {
    let parts = vec!["experts.0.up".to_owned(), "experts.0.gate".to_owned()];
    vec![
        loader
            .load(&WeightRequest::region("proj.weight", 0, Shape::new(&[4, 8])).pinned(DType::F32))
            .unwrap(),
        loader
            .load(&WeightRequest::strided(
                "rows.weight",
                4,
                Shape::new(&[4, 4]),
                8,
            ))
            .unwrap(),
        loader
            .load(&WeightRequest::parts(
                "experts.0.up_gate",
                parts.clone(),
                Shape::new(&[4, 8]),
                WeightLayout::UP_GATE,
            ))
            .unwrap(),
        loader
            .load(&WeightRequest::part_blocks(
                "experts.0.up_gate.rank1",
                parts,
                Shape::new(&[4, 4]),
                WeightLayout::UP_GATE,
                nn::weights::PartBlock {
                    start: 4,
                    rows: 2,
                    columns: 4,
                    row_stride: 8,
                },
            ))
            .unwrap(),
        loader
            .weight_norm("conv.weight", "conv.weight_v", "conv.weight_g")
            .unwrap(),
        loader
            .constant(
                "table",
                DType::I32,
                Shape::new(&[2]),
                &[1, 0, 0, 0, 2, 0, 0, 0],
                "test",
            )
            .unwrap(),
    ]
}

/// A meta load of a dense checkpoint answers the element type and extents a device load answers for
/// every kind of request, at aligned, non-null addresses that are distinct per weight and the same
/// for a weight asked for twice — and it reads no tensor of the checkpoint.
#[test]
fn a_meta_load_answers_what_a_device_load_answers_for_a_dense_checkpoint() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = dense_checkpoint(directory.path());

    let mut store = Recording::default();
    let device = dense_loads(&mut WeightLoader::new(&checkpoint, &mut store, None, None));
    let reading = checkpoint.reading();

    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let meta = dense_loads(&mut loader);
    assert_eq!(
        meta.iter().map(answer).collect::<Vec<_>>(),
        device.iter().map(answer).collect::<Vec<_>>(),
        "every view has the device load's element type and extents"
    );
    assert_eq!(
        checkpoint.reading(),
        reading,
        "the meta load asked the checkpoint for no tensor"
    );

    let addresses: Vec<u64> = meta
        .iter()
        .map(|weight| weight.view().ptr.address())
        .collect();
    for address in &addresses {
        assert!(*address != 0 && address % 256 == 0, "{address:#x}");
    }
    let mut distinct = addresses.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), addresses.len(), "one address per weight");

    let again = loader
        .load(&WeightRequest::region("proj.weight", 0, Shape::new(&[4, 8])).pinned(DType::F32))
        .unwrap();
    assert_eq!(
        again.view().ptr.address(),
        addresses[0],
        "a key answered once is answered again with the same region"
    );
}

/// A calibrated NVFP4 checkpoint in the vLLM spelling: one projection and a two-expert up/gate
/// stack, E4M3 block scales, FP32 tensor and activation scales. `zero` writes every value as zero,
/// which a device load refuses and a meta load never sees.
fn nvfp4_checkpoint(directory: &Path, zero: bool) -> Checkpoint {
    let byte = |value: u8| if zero { 0 } else { value };
    let scale = |value: f32| (if zero { 0.0 } else { value }).to_le_bytes().to_vec();
    let mut tensors: Vec<(String, DType, Vec<u64>, Vec<u8>)> = Vec::new();
    let mut projection = |stem: String, rows: u64| {
        tensors.push((
            format!("{stem}.weight"),
            DType::U8,
            vec![rows, 32],
            vec![byte(0x21); rows as usize * 32],
        ));
        tensors.push((
            format!("{stem}.weight_scale"),
            DType::F8E4M3,
            vec![rows, 4],
            vec![byte(0x38); rows as usize * 4],
        ));
        tensors.push((
            format!("{stem}.weight_scale_2"),
            DType::F32,
            vec![1],
            scale(0.5),
        ));
        tensors.push((
            format!("{stem}.input_scale"),
            DType::F32,
            vec![1],
            scale(0.25),
        ));
    };
    projection("layer.q_proj".to_owned(), 4);
    for expert in 0..2 {
        for name in ["up", "gate"] {
            projection(format!("experts.{expert}.{name}_proj"), 2);
        }
    }
    let views: Vec<(&str, DType, Vec<u64>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| (name.as_str(), *dtype, shape.clone(), bytes.clone()))
        .collect();
    shard_typed(directory, "model.safetensors", &views);
    Checkpoint::open(directory).unwrap()
}

/// What an NVFP4 projection and stack load answers, field by field, in a fixed order.
fn nvfp4_loads(loader: &mut WeightLoader<'_>) -> nn::Result<Vec<Option<Answer>>> {
    let global_scale = loader.tensor_scale(&WeightRequest::region(
        "layer.q_proj.weight_scale_2",
        0,
        Shape::new(&[1]),
    ))?;
    // The file's linear table is permuted into the product's swizzled one, which is the path a
    // device load reads the block scales' payload on.
    let quantized = loader.nvfp4(
        &WeightRequest::region("layer.q_proj.weight", 0, Shape::new(&[4, 32])),
        &WeightRequest::region("layer.q_proj.weight_scale", 0, Shape::new(&[4, 4])),
        Some(&WeightRequest::region(
            "layer.q_proj.input_scale",
            0,
            Shape::new(&[1]),
        )),
        Nvfp4Names::vllm(),
        nn::ScaleLayouts {
            file: ops::fp4::ScaleLayout::Linear,
            product: ops::fp4::ScaleLayout::Swizzled128X4,
        },
        global_scale,
    )?;
    let stack = loader
        .nvfp4_stack(
            "experts.up_gate",
            nn::ExpertStack {
                prefix: "experts",
                projections: &["up_proj", "gate_proj"],
                slots: &(0..2).collect::<Vec<u32>>(),
                vocabulary: 2,
            },
            Nvfp4Names::vllm(),
            nn::StackLayout {
                values: nn::ValueLayout::Plain,
                scales: ops::fp4::ScaleLayout::Swizzled128X4,
            },
            true,
        )
        .map(|(stack, _)| stack)?;
    Ok(vec![
        Some(answer(&quantized.packed)),
        Some(answer(&quantized.scales)),
        quantized.input_scale.as_ref().map(answer),
        Some(answer(&stack.values)),
        Some(answer(&stack.scales)),
        Some(answer(&stack.global_scale)),
        Some(answer(&stack.encode_scale)),
        stack.input_scale.as_ref().map(answer),
    ])
}

/// A meta load of a calibrated NVFP4 checkpoint decides the representation a device load decides:
/// the packed values as U8, the checkpoint's own E4M3 block scales in the product's layout, the
/// activation scales present, and the stack's tables in their published extents.
#[test]
fn a_meta_load_answers_what_a_device_load_answers_for_an_nvfp4_checkpoint() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = nvfp4_checkpoint(directory.path(), false);

    let mut store = Recording::default();
    let device = nvfp4_loads(&mut WeightLoader::new(&checkpoint, &mut store, None, None)).unwrap();
    let meta = nvfp4_loads(&mut WeightLoader::meta(&checkpoint, None, None)).unwrap();
    assert_eq!(meta, device);
    assert_eq!(
        meta[1],
        Some((DType::F8E4M3, vec![128, 4])),
        "the checkpoint's own scale type, padded to the swizzled table"
    );
}

/// A meta load reads no payload: a checkpoint whose values a device load refuses — all-zero packed
/// values and block scales, zero tensor scales, an all-zero matrix asked to hold a value — loads,
/// and no tensor is read.
#[test]
fn a_meta_load_reads_no_payload() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = nvfp4_checkpoint(directory.path(), true);

    let mut store = Recording::default();
    let refused = nvfp4_loads(&mut WeightLoader::new(&checkpoint, &mut store, None, None))
        .expect_err("a device load reads the zeros and refuses them");
    assert!(
        refused.to_string().contains("not a positive number"),
        "{refused}"
    );
    let before = checkpoint.reading();

    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    nvfp4_loads(&mut loader).expect("a meta load never sees the values");
    loader
        .load(&WeightRequest::region("layer.q_proj.weight", 0, Shape::new(&[4, 32])).nonzero())
        .expect("nor the zeros of a matrix asked to hold a value");
    assert_eq!(checkpoint.reading(), before, "no tensor was read");
}

/// A request the header contradicts is refused by a meta load with the device load's own refusal:
/// a region past its tensor, parts of different tails, a gain of the wrong length and an expert
/// stack whose block scales are not the linear table of its values.
#[test]
fn a_meta_load_refuses_what_the_header_contradicts() {
    let dense = tempfile::tempdir().unwrap();
    let checkpoint = dense_checkpoint(dense.path());
    let mut store = Recording::default();
    let mut device = WeightLoader::new(&checkpoint, &mut store, None, None);
    let mut meta = WeightLoader::meta(&checkpoint, None, None);

    let past = WeightRequest::region("proj.weight", 0, Shape::new(&[8, 8]));
    let tails = WeightRequest::parts(
        "mixed",
        vec!["experts.0.up".to_owned(), "conv.weight_v".to_owned()],
        Shape::new(&[4, 8]),
        WeightLayout::UP_GATE,
    );
    for request in [&past, &tails] {
        let expected = device.load(request).expect_err("a device load refuses");
        let refused = meta.load(request).expect_err("a meta load refuses");
        assert_eq!(refused.to_string(), expected.to_string());
    }
    let expected = device
        .weight_norm("conv.weight", "conv.weight_v", "proj.weight")
        .expect_err("a gain of 32 for 2 channels");
    let refused = meta
        .weight_norm("conv.weight", "conv.weight_v", "proj.weight")
        .expect_err("a meta load reads the gain's extents from the header");
    assert_eq!(refused.to_string(), expected.to_string());

    // A stack whose block scales are stated as wide as the packed values: the swizzled table is
    // the loader's to build, so the header alone refuses it.
    let encoded = tempfile::tempdir().unwrap();
    let mut tensors = Vec::new();
    for projection in ["up", "gate"] {
        let stem = format!("experts.0.{projection}_proj");
        tensors.push((
            format!("{stem}.weight"),
            DType::U8,
            vec![2, 32],
            vec![1; 64],
        ));
        tensors.push((
            format!("{stem}.weight_scale"),
            DType::U8,
            vec![2, 32],
            vec![1; 64],
        ));
        tensors.push((
            format!("{stem}.weight_scale_2"),
            DType::F32,
            vec![1],
            1.0_f32.to_le_bytes().to_vec(),
        ));
    }
    let views: Vec<(&str, DType, Vec<u64>, Vec<u8>)> = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| (name.as_str(), *dtype, shape.clone(), bytes.clone()))
        .collect();
    shard_typed(encoded.path(), "model.safetensors", &views);
    let checkpoint = Checkpoint::open(encoded.path()).unwrap();
    let stack = |loader: &mut WeightLoader<'_>| {
        loader
            .nvfp4_stack(
                "experts.up_gate",
                nn::ExpertStack {
                    prefix: "experts",
                    projections: &["up_proj", "gate_proj"],
                    slots: &(0..1).collect::<Vec<u32>>(),
                    vocabulary: 1,
                },
                Nvfp4Names::vllm(),
                nn::StackLayout {
                    values: nn::ValueLayout::Plain,
                    scales: ops::fp4::ScaleLayout::Linear,
                },
                false,
            )
            .map(|(stack, _)| stack)
            .map(|_| ())
    };
    let mut store = Recording::default();
    let expected = stack(&mut WeightLoader::new(&checkpoint, &mut store, None, None))
        .expect_err("a device load refuses");
    let refused =
        stack(&mut WeightLoader::meta(&checkpoint, None, None)).expect_err("a meta load refuses");
    assert_eq!(refused.to_string(), expected.to_string());
    assert!(refused.to_string().contains("linear"), "{refused}");
}

/// One stage of `dtype`-typed vectors, `elements` each, under `root/name`.
fn stage(root: &Path, name: &str, tensors: &[(&str, DType, u64)]) {
    let directory = root.join(name);
    std::fs::create_dir_all(&directory).unwrap();
    let typed: Vec<(&str, DType, Vec<u64>, Vec<u8>)> = tensors
        .iter()
        .map(|(tensor, dtype, elements)| {
            let bytes = vec![0u8; (*elements * ops::abi::bytes(*dtype).unwrap()) as usize];
            (*tensor, *dtype, vec![*elements], bytes)
        })
        .collect();
    shard_typed(&directory, "model.safetensors", &typed);
}

/// A checkpoint of one floating-point type answers with it, whichever it is: `auto` converts
/// nothing, so that type is what its weights keep.
#[test]
fn a_checkpoint_of_one_float_type_is_stored_in_it() {
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let root = tempfile::tempdir().unwrap();
        stage(
            root.path(),
            "text_encoder",
            &[("embed.weight", dtype, 7), ("norm.bias", dtype, 3)],
        );
        let checkpoint = Checkpoint::open(root.path()).unwrap();
        let text_encoder = checkpoint.stage("text_encoder").unwrap();
        assert_eq!(text_encoder.stored_dtype().unwrap(), dtype);
    }
}

/// An integer table is not a weight a precision describes, so it does not make the checkpoint
/// ambiguous, and a checkpoint of integers alone has no stored type to answer.
#[test]
fn integer_tensors_are_not_a_stored_type() {
    let root = tempfile::tempdir().unwrap();
    stage(
        root.path(),
        "transformer",
        &[
            ("proj_in.weight", DType::F32, 5),
            ("positions", DType::I64, 5),
            ("counts", DType::I32, 5),
        ],
    );
    stage(root.path(), "scheduler", &[("steps", DType::I64, 4)]);
    let checkpoint = Checkpoint::open(root.path()).unwrap();
    assert_eq!(
        checkpoint
            .stage("transformer")
            .unwrap()
            .stored_dtype()
            .unwrap(),
        DType::F32
    );
    let error = checkpoint
        .stage("scheduler")
        .unwrap()
        .stored_dtype()
        .expect_err("no float type to keep");
    assert!(
        error.to_string().contains("no floating-point tensor"),
        "{error}"
    );
}

/// A checkpoint of two float types lists each with a tensor that carries it, and a caller that asks
/// for one type is refused with both named.
#[test]
fn two_stored_types_are_listed_and_refused_as_one() {
    let root = tempfile::tempdir().unwrap();
    stage(
        root.path(),
        "transformer",
        &[
            ("audio_proj_in.bias", DType::F32, 3),
            ("context_embedder.bias", DType::BF16, 3),
            ("proj_in.weight", DType::BF16, 3),
            ("positions", DType::I32, 3),
        ],
    );
    let checkpoint = Checkpoint::open(root.path()).unwrap();
    let transformer = checkpoint.stage("transformer").unwrap();
    let types = transformer.stored_types();
    assert_eq!(
        types.iter().map(|(dtype, _)| *dtype).collect::<Vec<_>>(),
        vec![DType::F32, DType::BF16],
        "two floating-point types, and the integer table is not one of them"
    );
    assert_eq!(types[0].1, "audio_proj_in.bias");

    let message = transformer
        .stored_dtype()
        .expect_err("two float types are not one")
        .to_string();
    let folded = message.to_lowercase();
    for named in [
        "bf16",
        "f32",
        "audio_proj_in.bias",
        "context_embedder.bias",
        "transformer",
    ] {
        assert!(
            folded.contains(named),
            "the refusal names {named}: {message}"
        );
    }
}

/// A checkpoint's JSON documents are read from the first of its roots that holds each: an export's
/// own before the base's, a document only the base holds from the base, and one no root holds is
/// absent rather than an error.
#[test]
fn a_sidecar_is_read_from_the_first_root_that_holds_it() {
    let export = tempfile::tempdir().unwrap();
    let base = tempfile::tempdir().unwrap();
    for root in [export.path(), base.path()] {
        stage(root, "transformer", &[("proj_in.weight", DType::BF16, 2)]);
    }
    std::fs::write(
        export.path().join("transformer/config.json"),
        r#"{"from": "export"}"#,
    )
    .unwrap();
    std::fs::write(
        base.path().join("transformer/config.json"),
        r#"{"from": "base"}"#,
    )
    .unwrap();
    std::fs::create_dir_all(base.path().join("processor")).unwrap();
    std::fs::write(
        base.path().join("processor/preprocessor_config.json"),
        r#"{"from": "base"}"#,
    )
    .unwrap();
    std::fs::write(export.path().join("broken.json"), "{").unwrap();

    let checkpoint = Checkpoint::open_roots(&[export.path(), base.path()]).unwrap();
    let own = checkpoint.sidecar("transformer/config.json").unwrap();
    assert_eq!(own.unwrap()["from"], "export");
    let inherited = checkpoint
        .sidecar("processor/preprocessor_config.json")
        .unwrap();
    assert_eq!(inherited.unwrap()["from"], "base");
    assert!(
        checkpoint
            .sidecar("processor/video_preprocessor_config.json")
            .unwrap()
            .is_none()
    );
    let error = checkpoint
        .sidecar("broken.json")
        .expect_err("a document that is not JSON is refused");
    assert!(error.to_string().contains("broken.json"), "{error}");
}
