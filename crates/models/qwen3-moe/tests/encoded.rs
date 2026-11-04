//! An encoded (NVFP4) expert stack: the loader joins it expert-major from the fields a checkpoint
//! carries, and refuses a stack whose geometry or dtypes do not agree.
//!
//! The shard is written here, by hand, rather than through the `safetensors` crate's writer: the
//! format is a header JSON plus a payload, which is exactly what our loader reads, while the crate's
//! `Dtype` vocabulary is its writer's and (at the pinned 0.3) has no FP8 element at all. Writing the
//! header ourselves keeps the workspace's pin untouched and lets a fixture state exactly the bytes a
//! real NVFP4 checkpoint holds.

use std::path::Path;

use nn::weights::{Nvfp4Names, RetainedWeights, WeightKey};
use nn::{
    Backing, Checkpoint, DType, DevicePtr, Shape, WeightLayout, WeightLoader,
    weights::WeightRequest,
};
use ops::fp4::ScaleLayout;
use qwen3_moe::weights::ExpertSource;
use tempfile::TempDir;

/// Two experts, a `4 × 32` contraction: two E2M1 codes a byte along the contraction (a 16-byte
/// packed row) and two scale blocks of sixteen per row.
const EXPERTS: u32 = 2;
const ROWS: u64 = 4;
const CONTRACTION: u64 = 32;
const BLOCK_ELEMENTS: u64 = 16;

/// One tensor of the hand-written shard: name, extents, element type, payload.
struct Tensor<'a> {
    name: &'a str,
    shape: Vec<u64>,
    dtype: &'a str,
    bytes: Vec<u8>,
}

impl Tensor<'_> {
    fn elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Bytes one element occupies, from the safetensors format's own table.
    fn width(dtype: &str) -> u64 {
        match dtype {
            "U8" => 1,
            "F8_E4M3" => 1,
            "F32" => 4,
            _ => panic!("the fixture uses U8, F8_E4M3 and F32 only"),
        }
    }
}

/// Write `tensors` as one safetensors file: an 8-byte header length, the JSON header, the payload.
fn write_shard(dir: &Path, tensors: &[Tensor<'_>]) -> std::path::PathBuf {
    let mut header = String::from("{");
    let mut offset = 0_u64;
    let mut payload = Vec::new();
    for (index, tensor) in tensors.iter().enumerate() {
        let bytes = tensor.elements() * Tensor::width(tensor.dtype);
        // The payload is exactly what the header declares: a fixture that deliberately mis-declares
        // a field's element type is padded to that type's width, so the dtype is what the loader
        // objects to and not the length.
        let mut payload_tensor = tensor.bytes.clone();
        payload_tensor.resize(bytes as usize, 0);
        if index > 0 {
            header.push(',');
        }
        header.push_str(&format!(
            r#""{}":{{"dtype":"{}","shape":{:?},"data_offsets":[{},{}]}}"#,
            tensor.name,
            tensor.dtype,
            tensor.shape,
            offset,
            offset + bytes
        ));
        offset += bytes;
        payload.extend_from_slice(&payload_tensor);
    }
    header.push('}');

    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&payload);
    let path = dir.join("model.safetensors");
    std::fs::write(&path, file)
        .map_err(|error| format!("the fixture shard: {error}"))
        .unwrap_or_else(|error| panic!("{error}"));
    path
}

fn packed(seed: u8) -> Vec<u8> {
    let count = (ROWS * (CONTRACTION / 2)) as usize;
    (0..count)
        .map(|index| seed.wrapping_add((index as u8).wrapping_mul(37)))
        .collect()
}

fn scales(seed: u8) -> Vec<u8> {
    // One E4M3 scale per sixteen contraction elements, one row per output row.
    let count = (ROWS * (CONTRACTION / BLOCK_ELEMENTS)) as usize;
    (0..count)
        .map(|index| seed.wrapping_add(index as u8))
        .collect()
}

/// One expert's tensor scale: a scalar in the checkpoint, one f32, which the loader turns into a
/// `[experts]` reciprocal table — a weight, so no layer reads a device scalar to build a launch.
fn tensor_scale(seed: u8) -> Vec<u8> {
    f32::from(seed).to_le_bytes().to_vec()
}

/// A shard holding `projections` of every expert of `layers.0.mlp.experts`, in both encodings, with
/// one scale element type for every expert.
fn encoded_shard(projections: &[&str], packed_dtype: &str, scale_dtype: &str) -> TempDir {
    encoded_shard_with(projections, packed_dtype, |_| scale_dtype.to_owned())
}

/// The same shard, with the block-scale element type chosen per expert: a checkpoint whose experts
/// disagree is a misread stack, and one case below needs to write exactly that.
fn encoded_shard_with(
    projections: &[&str],
    packed_dtype: &str,
    scale_dtype: impl Fn(usize) -> String,
) -> TempDir {
    let dir = TempDir::new().unwrap_or_else(|error| panic!("a temporary directory: {error}"));
    let mut owned: Vec<(String, Vec<u64>, String, Vec<u8>)> = Vec::new();
    for expert in 0..EXPERTS {
        for (index, projection) in projections.iter().enumerate() {
            let stem = format!("model.layers.0.mlp.experts.{expert}.{projection}_proj");
            let seed = 1 + (expert as u8) * 16 + index as u8;
            owned.push((
                format!("{stem}.weight"),
                vec![ROWS, CONTRACTION / 2],
                packed_dtype.to_owned(),
                packed(seed),
            ));
            owned.push((
                format!("{stem}.weight_scale"),
                vec![ROWS, CONTRACTION / BLOCK_ELEMENTS],
                scale_dtype(expert as usize),
                scales(seed),
            ));
            owned.push((
                format!("{stem}.weight_scale_2"),
                Vec::new(),
                "F32".to_owned(),
                tensor_scale(seed),
            ));
        }
    }
    let borrowed: Vec<Tensor<'_>> = owned
        .iter()
        .map(|(name, shape, dtype, bytes)| Tensor {
            name,
            shape: shape.clone(),
            dtype,
            bytes: bytes.clone(),
        })
        .collect();
    write_shard(dir.path(), &borrowed);
    dir
}

/// A store that keeps what it was given, in host memory, the way a layer's store does.
#[derive(Default)]
struct Retaining {
    published: Vec<(WeightKey, Vec<u8>, DType, Shape, u64)>,
    next: u64,
}

impl RetainedWeights for Retaining {
    fn retained(&mut self, key: &WeightKey) -> nn::Result<Option<Backing>> {
        Ok(self
            .published
            .iter()
            .find(|(published, ..)| published == key)
            .map(|(_, bytes, _, _, address)| {
                Backing::new(DevicePtr::new(*address), bytes.len() as u64)
            }))
    }

    fn publish(
        &mut self,
        key: &WeightKey,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
    ) -> nn::Result<Backing> {
        self.next += 0x1_0000;
        let address = self.next;
        self.published
            .push((key.clone(), bytes.to_vec(), dtype, shape, address));
        Ok(Backing::new(DevicePtr::new(address), bytes.len() as u64))
    }
}

#[test]
fn a_two_expert_encoded_stack_is_joined_expert_major_with_its_three_fields() {
    let dir = encoded_shard(&["up", "gate"], "U8", "F8_E4M3");
    let checkpoint = Checkpoint::open(dir.path()).expect("the shard opens");
    let mut store = Retaining::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);

    let stack = loader
        .nvfp4_stack(
            "layers.0.mlp.experts.up",
            "model.layers.0.mlp.experts",
            &["up", "gate"],
            EXPERTS,
            Nvfp4Names::vllm(),
            nn::weights::ValueLayout::Shuffled {
                epilogue_tile_m: 128,
            },
            ScaleLayout::Linear,
            false,
        )
        .expect("a two-expert encoded stack loads");

    // The packed field: every expert's every projection, expert-major, `[experts · projections ·
    // rows, columns/2]` — the row order the routed product reads.
    assert_eq!(stack.values.view.dtype, DType::U8);
    assert_eq!(
        stack.values.view.shape.dims(),
        &[u64::from(EXPERTS), 2 * ROWS, CONTRACTION / 2]
    );
    // The block scales keep the layout the loader was told to read, and one reciprocal per expert.
    assert_eq!(stack.layout, ScaleLayout::Linear);
    // The block-scale field is the shard's own element type and is published unchanged: a real
    // ModelOpt export stores it as E4M3, and a loader that relabelled it U8 would hide which byte
    // the kernel decodes.
    assert_eq!(stack.scales.view.dtype, DType::F8E4M3);
    // One tensor scale per (expert, projection), in the reading order the stack fuses: a checkpoint
    // carries one `weight_scale_2` per projection tensor, so the two halves of `up_gate` generally
    // differ and a kernel that used one for both would be wrong by their ratio.
    assert_eq!(
        stack.global_scale.view.shape.dims(),
        &[u64::from(EXPERTS), 2]
    );
    let published = store
        .published
        .iter()
        .find(|(_, _, dtype, shape, _)| *dtype == DType::F32 && shape.rank() == 2)
        .expect("the loader publishes the reciprocal table");
    let reciprocals: Vec<f32> = published
        .1
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    // The fixture's tensor scales, expert-major and projection-minor: 1, 2 for expert 0's up and
    // gate, then 17, 18 for expert 1's — published as their reciprocals.
    let expected: Vec<f32> = [1.0_f32, 2.0, 17.0, 18.0].iter().map(|s| 1.0 / s).collect();
    assert_eq!(reciprocals.len(), expected.len());
    for (found, want) in reciprocals.iter().zip(&expected) {
        assert!(
            (found - want).abs() <= 1e-6,
            "{reciprocals:?} against {expected:?}"
        );
    }
}

#[test]
fn a_packed_field_that_is_not_u8_is_refused_by_name() {
    // The packed values declared as F32: the field holds codes, so this is a misread checkpoint.
    let dir = encoded_shard(&["up"], "F32", "U8");
    let checkpoint = Checkpoint::open(dir.path()).expect("the shard opens");
    let mut store = Retaining::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let error = loader
        .nvfp4_stack(
            "layers.0.mlp.experts.up",
            "model.layers.0.mlp.experts",
            &["up"],
            EXPERTS,
            Nvfp4Names::vllm(),
            nn::weights::ValueLayout::Shuffled {
                epilogue_tile_m: 128,
            },
            ScaleLayout::Linear,
            false,
        )
        .unwrap_err();
    assert!(format!("{error}").contains("up_proj.weight"), "{error}");
}

#[test]
fn a_stack_whose_experts_declare_different_scale_types_is_refused() {
    // One set of views serves every expert, so the block-scale element type is the stack's, not the
    // projection's: a checkpoint where one expert says E4M3 and the next says U8 is a misread stack,
    // and reading it would decode one expert's scales with the other's rule.
    let dir = encoded_shard_with(&["up"], "U8", |expert| match expert {
        0 => "F8_E4M3".to_owned(),
        _ => "U8".to_owned(),
    });
    let checkpoint = Checkpoint::open(dir.path()).expect("the shard opens");
    let mut store = Retaining::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let error = loader
        .nvfp4_stack(
            "layers.0.mlp.experts.up",
            "model.layers.0.mlp.experts",
            &["up"],
            EXPERTS,
            Nvfp4Names::vllm(),
            nn::weights::ValueLayout::Shuffled {
                epilogue_tile_m: 128,
            },
            ScaleLayout::Linear,
            false,
        )
        .unwrap_err();
    let message = format!("{error}");
    assert!(message.contains("weight_scale"), "{message}");
}

#[test]
fn a_projection_the_shard_does_not_hold_is_refused_by_name() {
    let dir = encoded_shard(&["up"], "U8", "U8");
    let checkpoint = Checkpoint::open(dir.path()).expect("the shard opens");
    let mut store = Retaining::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let error = loader
        .nvfp4_stack(
            "layers.0.mlp.experts.down",
            "model.layers.0.mlp.experts",
            &["down"],
            EXPERTS,
            Nvfp4Names::vllm(),
            nn::weights::ValueLayout::Shuffled {
                epilogue_tile_m: 128,
            },
            ScaleLayout::Linear,
            false,
        )
        .unwrap_err();
    assert!(format!("{error}").contains("down"), "{error}");
}

#[test]
fn the_dense_request_names_one_tensor_and_the_encoded_one_names_three() {
    // The sentence this test exists for: both paths read the same stem, so a quantized checkpoint
    // read through the dense request would take the packed field for BF16 weights. That is why the
    // encoded path exists and why the dense one must never be pointed at it.
    let module = ExpertSource {
        module: "layers.0.mlp.experts".to_owned(),
        checkpoint_prefix: "model.layers.0.mlp.experts".to_owned(),
    };
    let dense = WeightRequest::parts(
        "layers.0.mlp.experts.up".to_owned(),
        vec![module.tensors(0).up],
        Shape::new(&[1, ROWS, CONTRACTION]),
        WeightLayout::UP_GATE,
    );
    let nn::weights::WeightSource::Parts { parts, .. } = &dense.source else {
        panic!("the dense request fuses the experts' projections");
    };
    assert_eq!(parts.len(), 1);

    let encoded = module.encoded(0);
    assert_eq!(&encoded.up.values, parts.first().unwrap());
    assert_eq!(
        encoded.up.scales,
        "model.layers.0.mlp.experts.0.up_proj.weight_scale"
    );
    assert_eq!(
        encoded.up.global_scale,
        "model.layers.0.mlp.experts.0.up_proj.weight_scale_2"
    );
}
