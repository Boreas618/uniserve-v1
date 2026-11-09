//! The projections a model loads through its scopes: a checkpoint's quantization declaration, the
//! representation each projection is read in, and the shards a tensor-parallel rank holds.
//!
//! The tests write their own shards (header and payload by hand) and publish to a host store that
//! keeps every published byte, so what a rank read is checked against the checkpoint's own values.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]
use std::path::Path;

use nn::family::Split;
use nn::layers::{
    Bias, Linear, MergedColumnLinear, Parallel, ParallelLMHead, QkvBias, QkvLinear, QkvNames,
    QkvSpec, VocabParallelEmbedding,
};
use nn::quantization::{Declaration, KvCache, Scheme};
use nn::weights::RetainedWeights;
use nn::{Checkpoint, DType, DevicePtr, Shape, TensorView, WeightKey, WeightLoader};

/// The format's name for an element type.
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

/// Write `directory/model.safetensors` holding `tensors`: name, element type, extents, bytes.
fn shard(directory: &Path, tensors: &[(&str, DType, Vec<u64>, Vec<u8>)]) {
    let mut header = String::from("{");
    let mut payload: Vec<u8> = Vec::new();
    for (tensor, dtype, shape, bytes) in tensors {
        let start = payload.len();
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
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&payload);
    std::fs::write(directory.join("model.safetensors"), file).unwrap();
}

/// A `[rows, columns]` F32 matrix whose element `(r, c)` is `r * 100 + c`, so a read region names
/// itself.
fn matrix(rows: u64, columns: u64) -> Vec<u8> {
    (0..rows)
        .flat_map(|row| (0..columns).map(move |column| (row * 100 + column) as f32))
        .flat_map(f32::to_le_bytes)
        .collect()
}

/// A store that keeps every published region's bytes at a synthetic address.
#[derive(Default)]
struct Store {
    published: Vec<(WeightKey, Vec<u8>, u64)>,
    next: u64,
}

impl RetainedWeights for Store {
    fn retained(&mut self, key: &WeightKey) -> nn::Result<Option<nn::Backing>> {
        Ok(self
            .published
            .iter()
            .find(|(held, ..)| held == key)
            .map(|(_, bytes, address)| {
                nn::Backing::new(DevicePtr::new(*address), bytes.len() as u64)
            }))
    }

    fn publish(
        &mut self,
        key: &WeightKey,
        _: DType,
        _: Shape,
        bytes: &[u8],
    ) -> nn::Result<nn::Backing> {
        self.next += 0x1_0000;
        self.published
            .push((key.clone(), bytes.to_vec(), self.next));
        Ok(nn::Backing::new(
            DevicePtr::new(self.next),
            bytes.len() as u64,
        ))
    }
}

impl Store {
    /// The F32 values a view addresses.
    fn values(&self, view: TensorView) -> Vec<f32> {
        let (_, bytes, _) = self
            .published
            .iter()
            .find(|(_, _, address)| *address == view.ptr.address())
            .expect("the view addresses a published region");
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_le_bytes(*word))
            .collect()
    }
}

/// The checkpoint at `directory`, with `config` as its `config.json`.
fn checkpoint(directory: &Path, config: &str) -> Checkpoint {
    std::fs::write(directory.join("config.json"), config).unwrap();
    Checkpoint::open(directory).unwrap()
}

/// The values `(r, c)` of rows `rows` and columns `columns` of a [`matrix`].
fn expected(rows: std::ops::Range<u64>, columns: std::ops::Range<u64>) -> Vec<f32> {
    rows.flat_map(|row| {
        columns
            .clone()
            .map(move |column| (row * 100 + column) as f32)
    })
    .collect()
}

#[test]
fn a_column_rank_reads_its_rows_and_a_row_rank_its_columns() {
    let directory = tempfile::tempdir().unwrap();
    shard(
        directory.path(),
        &[
            ("layer.up.weight", DType::F32, vec![4, 6], matrix(4, 6)),
            ("layer.up.bias", DType::F32, vec![4], matrix(1, 4)),
            ("layer.down.weight", DType::F32, vec![6, 4], matrix(6, 4)),
        ],
    );
    let checkpoint = checkpoint(directory.path(), "{}");
    let mut store = Store::default();
    let rank = Split { size: 2, rank: 1 };
    let (up, down) = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        let mut scope = loader.scope("layer");
        let up = Linear::load(
            &mut scope.scope("up"),
            6,
            4,
            Bias::Present,
            Parallel::Column(rank),
        )
        .unwrap();
        let down = Linear::load(
            &mut scope.scope("down"),
            4,
            6,
            Bias::Absent,
            Parallel::Row(rank),
        )
        .unwrap();
        (up, down)
    };
    // Rank 1 of 2 computes output channels 2..4 of the up projection: rows 2..4 of its matrix.
    assert_eq!((up.k(), up.n()), (6, 2));
    assert_eq!(store.values(up.weight()), expected(2..4, 0..6));
    // And contracts columns 2..4 of every row of the down projection.
    assert_eq!((down.k(), down.n()), (2, 6));
    assert_eq!(store.values(down.weight()), expected(0..6, 2..4));

    // A split that does not divide the axis is refused by the module's name.
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let refused = Linear::load(
        &mut loader.scope("layer.up"),
        6,
        4,
        Bias::Absent,
        Parallel::Column(Split { size: 3, rank: 0 }),
    )
    .err()
    .expect("three ranks do not split four channels");
    assert!(refused.to_string().contains("layer.up.weight"), "{refused}");
}

#[test]
fn a_projection_of_separate_or_fused_modules_reads_each_head_window() {
    // Four query heads and two key/value heads of width two over a hidden width of three.
    let directory = tempfile::tempdir().unwrap();
    shard(
        directory.path(),
        &[
            ("attn.q_proj.weight", DType::F32, vec![8, 3], matrix(8, 3)),
            ("attn.k_proj.weight", DType::F32, vec![4, 3], matrix(4, 3)),
            ("attn.v_proj.weight", DType::F32, vec![4, 3], matrix(4, 3)),
            ("fused.qkv.weight", DType::F32, vec![16, 3], matrix(16, 3)),
            ("fused.q_bias", DType::F32, vec![8], matrix(1, 8)),
            ("fused.k_bias", DType::F32, vec![4], matrix(1, 4)),
            ("fused.v_bias", DType::F32, vec![4], matrix(1, 4)),
        ],
    );
    let checkpoint = checkpoint(directory.path(), "{}");
    let spec = QkvSpec {
        hidden: 3,
        heads: 4,
        kv_heads: 2,
        head_dim: 2,
    };
    let mut store = Store::default();
    let load = |store: &mut Store, module: &str, names, bias, split| {
        let mut loader = WeightLoader::new(&checkpoint, store, None, None);
        QkvLinear::load(&mut loader.scope(module), spec, names, bias, split).unwrap()
    };

    // Two ranks: rank 1 holds query heads 2..4 and key/value head 1.
    let separate = load(
        &mut store,
        "attn",
        QkvNames::Separate(["q_proj", "k_proj", "v_proj"]),
        QkvBias::Absent,
        Split { size: 2, rank: 1 },
    );
    assert_eq!(
        store.values(separate.query().weight()),
        expected(4..8, 0..3)
    );
    assert_eq!(store.values(separate.key().weight()), expected(2..4, 0..3));

    // Four ranks over two key/value heads: each head is replicated on two consecutive ranks, so
    // rank 3 reads query head 3 and key/value head 1.
    let replicated = load(
        &mut store,
        "attn",
        QkvNames::Separate(["q_proj", "k_proj", "v_proj"]),
        QkvBias::Absent,
        Split { size: 4, rank: 3 },
    );
    assert_eq!(
        store.values(replicated.query().weight()),
        expected(6..8, 0..3)
    );
    assert_eq!(
        store.values(replicated.value().weight()),
        expected(2..4, 0..3)
    );

    // One fused module: the query rows, then the key rows, then the value rows.
    let fused = load(
        &mut store,
        "fused",
        QkvNames::Fused("qkv"),
        QkvBias::Windows(["q_bias", "k_bias", "v_bias"]),
        Split::WHOLE,
    );
    assert_eq!(store.values(fused.query().weight()), expected(0..8, 0..3));
    assert_eq!(store.values(fused.key().weight()), expected(8..12, 0..3));
    assert_eq!(store.values(fused.value().weight()), expected(12..16, 0..3));

    // A split wider than the key/value heads that does not replicate them evenly is refused.
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let wide = QkvSpec { heads: 6, ..spec };
    assert!(
        QkvLinear::load(
            &mut loader.scope("attn"),
            wide,
            QkvNames::Separate(["q_proj", "k_proj", "v_proj"]),
            QkvBias::Absent,
            Split { size: 3, rank: 0 },
        )
        .is_err()
    );
}

#[test]
fn merged_modules_are_one_product_in_the_order_named() {
    let directory = tempfile::tempdir().unwrap();
    shard(
        directory.path(),
        &[
            ("attn.to_q.weight", DType::F32, vec![2, 2], matrix(2, 2)),
            ("attn.to_k.weight", DType::F32, vec![1, 2], matrix(1, 2)),
        ],
    );
    let checkpoint = checkpoint(directory.path(), "{}");
    let mut store = Store::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    let merged = MergedColumnLinear::load(
        &mut loader.scope("attn"),
        &["to_q", "to_k"],
        2,
        &[2, 1],
        Bias::Absent,
    )
    .unwrap();
    assert_eq!(merged.n(), 3);
}

/// One calibrated checkpoint: a packed `q_proj` beside a dense router, both declared in
/// `config.json`'s compressed-tensors block.
fn calibrated(directory: &Path) -> Checkpoint {
    shard(
        directory,
        &[
            (
                "m.q_proj.weight",
                DType::U8,
                vec![128, 16],
                vec![0x11; 128 * 16],
            ),
            (
                "m.q_proj.weight_scale",
                DType::F8E4M3,
                vec![128, 2],
                vec![0x38; 256],
            ),
            (
                "m.q_proj.weight_scale_2",
                DType::F32,
                vec![1],
                0.5_f32.to_le_bytes().to_vec(),
            ),
            (
                "m.q_proj.input_scale",
                DType::F32,
                vec![1],
                0.25_f32.to_le_bytes().to_vec(),
            ),
            ("m.gate.weight", DType::F32, vec![2, 32], matrix(2, 32)),
            ("m.packed_norm.weight", DType::U8, vec![16], vec![1; 16]),
            ("m.packed_norm.weight_scale", DType::U8, vec![1], vec![1]),
        ],
    );
    checkpoint(
        directory,
        r#"{"quantization_config": {
            "config_groups": {"group_0": {
                "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                "targets": ["Linear"]}},
            "ignore": ["m.gate"],
            "kv_cache_scheme": {"num_bits": 8, "type": "float"}}}"#,
    )
}

#[test]
fn a_declaration_is_resolved_against_the_shards() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = calibrated(directory.path());
    let declaration = Declaration::read(&checkpoint).unwrap();
    assert!(declaration.is_encoded());
    assert_eq!(declaration.kv_cache(), KvCache::Fp8);
    // Before the shards are read, the declaration's intent: packed unless ignored.
    assert_eq!(declaration.scheme("m.q_proj.weight"), Scheme::Nvfp4);
    assert_eq!(declaration.scheme("m.gate.weight"), Scheme::Dense);
    assert_eq!(declaration.scheme("m.o_proj.weight"), Scheme::Nvfp4);
    // After: a weight is packed exactly when its shard stores a block scale beside it.
    let resolved = declaration.resolve(&checkpoint);
    assert_eq!(resolved.scheme("m.q_proj.weight"), Scheme::Nvfp4);
    assert_eq!(resolved.scheme("m.o_proj.weight"), Scheme::Dense);
    assert_eq!(resolved.scheme("m.gate.weight"), Scheme::Dense);
}

#[test]
fn a_packed_projection_loads_as_nvfp4_and_a_packed_tensor_elsewhere_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = calibrated(directory.path());
    let declaration = Declaration::read(&checkpoint).unwrap();
    let mut store = Store::default();
    let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
    loader.declare(&declaration);
    let mut scope = loader.scope("m");
    let query = Linear::load(
        &mut scope.scope("q_proj"),
        32,
        128,
        Bias::Absent,
        Parallel::None,
    )
    .unwrap();
    assert!(query.is_packed());
    assert_eq!((query.k(), query.n()), (32, 128));
    // A packed projection with a bias is not an export this reads.
    assert!(
        Linear::load(
            &mut scope.scope("q_proj"),
            32,
            128,
            Bias::Present,
            Parallel::None
        )
        .is_err()
    );
    // The router is dense, and a dense layer reads it.
    let router = scope.scope("gate").tensor("weight", &[2, 32]).unwrap();
    assert_eq!(router.dtype(), DType::F32);
    // A norm the shards store packed is not a weight a norm can read.
    let refused = scope
        .scope("packed_norm")
        .tensor("weight", &[16])
        .expect_err("a packed norm");
    assert!(
        refused.to_string().contains("m.packed_norm.weight"),
        "{refused}"
    );
}

#[test]
fn a_declaration_this_library_does_not_serve_is_refused_by_name() {
    for (config, named) in [
        (
            r#"{"quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}}"#,
            "FP8",
        ),
        (
            r#"{"quantization_config": {"quant_method": "awq", "quant_algo": "NVFP4"}}"#,
            "awq",
        ),
        (
            r#"{"quantization_config": {"config_groups": {"g": {"weights":
                {"num_bits": 8, "type": "float", "group_size": 16}}}}}"#,
            "NVFP4",
        ),
        (
            r#"{"quantization_config": {"quant_algo": "NVFP4", "kv_cache_quant_algo": "INT8"}}"#,
            "INT8",
        ),
    ] {
        let directory = tempfile::tempdir().unwrap();
        shard(directory.path(), &[("w", DType::F32, vec![1], vec![0; 4])]);
        let checkpoint = checkpoint(directory.path(), config);
        let refused = Declaration::read(&checkpoint).expect_err("an unserved declaration");
        assert!(refused.to_string().contains(named), "{refused}");
    }

    // The export's own file states the same algorithm or the checkpoint is refused.
    let directory = tempfile::tempdir().unwrap();
    shard(directory.path(), &[("w", DType::F32, vec![1], vec![0; 4])]);
    std::fs::write(
        directory.path().join("hf_quant_config.json"),
        r#"{"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "NVFP4"}}"#,
    )
    .unwrap();
    let agreeing = checkpoint(
        directory.path(),
        r#"{"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}"#,
    );
    assert!(Declaration::read(&agreeing).unwrap().is_encoded());
    let disagreeing = checkpoint(
        directory.path(),
        r#"{"quantization_config": {"quant_method": "modelopt", "quant_algo": "FP8"}}"#,
    );
    let refused = Declaration::read(&disagreeing).expect_err("two different quantizations");
    assert!(refused.to_string().contains("two different"), "{refused}");

    // A checkpoint that declares nothing is dense.
    let dense = tempfile::tempdir().unwrap();
    shard(dense.path(), &[("w", DType::F32, vec![1], vec![0; 4])]);
    let declaration = Declaration::read(&checkpoint(dense.path(), "{}")).unwrap();
    assert!(!declaration.is_encoded());
    assert_eq!(declaration.scheme("w"), Scheme::Dense);
}

/// The embedding and the head shard the vocabulary by one rule: a rank's rows of each table are
/// the same ids, and a tied head reads the embedding's own rows.
#[test]
fn the_embedding_and_the_head_hold_the_same_vocabulary_rows() {
    let directory = tempfile::tempdir().unwrap();
    shard(
        directory.path(),
        &[
            ("embed.weight", DType::F32, vec![32, 3], matrix(32, 3)),
            ("head.weight", DType::F32, vec![32, 3], matrix(32, 3)),
        ],
    );
    let checkpoint = checkpoint(directory.path(), "{}");
    let mut store = Store::default();
    let rank = Split { size: 2, rank: 1 };
    let (embedding, head) = {
        let mut loader = WeightLoader::new(&checkpoint, &mut store, None, None);
        let embedding =
            VocabParallelEmbedding::load(&mut loader.scope("embed"), 32, 3, rank).unwrap();
        let head = ParallelLMHead::load(&mut loader.scope("head"), 32, 3, rank).unwrap();
        (embedding, head)
    };
    assert_eq!(embedding.shard(), (16, 16));
    assert_eq!(
        store.values(embedding.weight().view()),
        expected(16..32, 0..3)
    );
    assert_eq!(head.padded(), 16);
    assert_eq!(ParallelLMHead::tied(&embedding).unwrap().padded(), 16);
}
