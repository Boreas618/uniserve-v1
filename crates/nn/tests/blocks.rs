//! What the shared blocks launch and reserve: each block is loaded by a meta loader from a
//! checkpoint of zero tensors and walked on a recording forward, so the programs it states and the
//! arena it leaves behind are observable without a device.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

use nn::blocks::{GateUp, GatedMlp, GatedMlpSpec};
use nn::family::Split;
use nn::layers::Rounding;
use nn::{Arena, Checkpoint, DType, Forward, Shape, TensorView, WeightLoader};
use ops::activation::Activation;

/// Runs `walk` with a capture sink installed and answers the operators it reported, in order.
fn launched<T>(walk: impl FnOnce() -> T) -> (T, Vec<String>) {
    #[derive(Default)]
    struct Launches(Vec<String>);
    impl ops::capture::Sink for Launches {
        fn launched(&mut self, operator: &'static str, _key: &str) {
            self.0.push(operator.to_owned());
        }
    }
    let mut launches = Launches::default();
    ops::capture::set_sink(Some(&mut launches));
    let answer = walk();
    ops::capture::set_sink(None);
    (answer, launches.0)
}

/// A checkpoint in `directory` holding BF16 zero tensors of the given names and extents.
fn checkpoint(directory: &std::path::Path, tensors: &[(&str, &[u64])]) -> Checkpoint {
    let mut header = String::from("{");
    let mut offset = 0_u64;
    for (name, shape) in tensors {
        let bytes = shape.iter().product::<u64>() * 2;
        let extents: Vec<String> = shape.iter().map(u64::to_string).collect();
        header.push_str(&format!(
            r#""{name}":{{"dtype":"BF16","shape":[{}],"data_offsets":[{offset},{}]}},"#,
            extents.join(","),
            offset + bytes
        ));
        offset += bytes;
    }
    header.pop();
    header.push('}');
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.resize(file.len() + offset as usize, 0);
    std::fs::write(directory.join("model.safetensors"), file).unwrap();
    Checkpoint::open(directory).unwrap()
}

/// A view of `dims` BF16 elements reserved from `arena`.
fn view(arena: &Arena, dims: &[u64]) -> TensorView {
    arena.tensor(DType::BF16, Shape::new(dims)).unwrap()
}

const HIDDEN: u64 = 64;
const INTERMEDIATE: u64 = 128;

fn mlp_spec(rounding: Rounding, gate_up: GateUp<'_>) -> GatedMlpSpec<'_> {
    GatedMlpSpec {
        hidden: HIDDEN as u32,
        intermediate: INTERMEDIATE as u32,
        activation: Activation::Silu,
        rounding,
        gate_up,
        down: "down_proj",
    }
}

const SPLIT: GateUp<'static> = GateUp::Split {
    gate: "gate_proj",
    up: "up_proj",
};

/// A split gated MLP projects twice, gates at its rounding, projects back, and leaves the arena
/// as it found it: its intermediates are its own.
#[test]
fn a_split_gated_mlp_gates_at_its_rounding_and_releases_its_intermediates() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("mlp.gate_proj.weight", &[INTERMEDIATE, HIDDEN]),
            ("mlp.up_proj.weight", &[INTERMEDIATE, HIDDEN]),
            ("mlp.down_proj.weight", &[HIDDEN, INTERMEDIATE]),
        ],
    );
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let (x, out) = (view(&arena, &[5, HIDDEN]), view(&arena, &[5, HIDDEN]));

    for (rounding, gate) in [
        (Rounding::ValueProduct, "swiglu"),
        (Rounding::Fused, "act_and_mul"),
    ] {
        let mlp = GatedMlp::load(
            &mut loader.scope("mlp"),
            mlp_spec(rounding, SPLIT),
            Split::WHOLE,
        )
        .unwrap();
        let used = arena.used();
        let (result, launches) = launched(|| mlp.forward(&f, x, out));
        result.unwrap();
        assert_eq!(launches, ["gemm", "gemm", gate, "gemm"], "{rounding:?}");
        assert_eq!(arena.used(), used, "the intermediates are released");
    }
}

/// A value-first projection is read once and gated from its two halves.
#[test]
fn a_value_first_gated_mlp_reads_one_packed_projection() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("ff.proj.weight", &[2 * INTERMEDIATE, HIDDEN]),
            ("ff.down_proj.weight", &[HIDDEN, INTERMEDIATE]),
        ],
    );
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let mlp = GatedMlp::load(
        &mut loader.scope("ff"),
        mlp_spec(Rounding::Stepwise, GateUp::ValueFirst("proj")),
        Split::WHOLE,
    )
    .unwrap();
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let x = view(&arena, &[3, HIDDEN]);
    let (out, launches) = launched(|| mlp.apply(&f, x));
    assert_eq!(out.unwrap().shape.dims(), &[3, HIDDEN]);
    assert_eq!(launches, ["gemm", "act_and_mul", "gemm"]);
}

/// Over a tensor-parallel group the block holds its rank's share of the gated width and reduces in
/// its down projection, which a forward without a group refuses.
#[test]
fn a_tensor_parallel_gated_mlp_reduces_in_its_down_projection() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("mlp.gate_proj.weight", &[INTERMEDIATE, HIDDEN]),
            ("mlp.up_proj.weight", &[INTERMEDIATE, HIDDEN]),
            ("mlp.down_proj.weight", &[HIDDEN, INTERMEDIATE]),
        ],
    );
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let mlp = GatedMlp::load(
        &mut loader.scope("mlp"),
        mlp_spec(Rounding::ValueProduct, SPLIT),
        Split { size: 2, rank: 1 },
    )
    .unwrap();
    let arena = Arena::counting();
    let (x, out) = (view(&arena, &[4, HIDDEN]), view(&arena, &[4, HIDDEN]));

    let split = Forward::recording(&arena).with_split(1, 2).unwrap();
    assert!(mlp.forward(&split, x, out).is_ok());
    assert!(
        mlp.forward(&Forward::recording(&arena), x, out).is_err(),
        "a row-parallel projection with no group to reduce over"
    );

    // A split that does not divide the gated width is refused at load.
    assert!(
        GatedMlp::load(
            &mut loader.scope("mlp"),
            mlp_spec(Rounding::ValueProduct, SPLIT),
            Split { size: 3, rank: 0 },
        )
        .is_err()
    );
}

// ---- QKV attention ------------------------------------------------------------------------------

use nn::blocks::{AttentionKind, Keys, QkNormNames, QkvAttention, QkvAttentionSpec, RopeFactors};
use nn::layers::{
    Bias, Precision, QkNorm, QkvBias, QkvNames, QkvSpec, Rotary, RotarySpec, Spectrum, Varlen,
};
use nn::{KvGroup, KvPlanes, KvView, PhaseKind, TokenBatch};

const HEADS: u64 = 4;
const KV_HEADS: u64 = 2;
const HEAD_DIM: u64 = 32;

/// The tensors of an attention module `attn` with separate projections and per-head norm gains.
fn attention_checkpoint(directory: &std::path::Path) -> Checkpoint {
    checkpoint(
        directory,
        &[
            ("attn.q_proj.weight", &[HEADS * HEAD_DIM, HIDDEN]),
            ("attn.k_proj.weight", &[KV_HEADS * HEAD_DIM, HIDDEN]),
            ("attn.v_proj.weight", &[KV_HEADS * HEAD_DIM, HIDDEN]),
            ("attn.o_proj.weight", &[HIDDEN, HEADS * HEAD_DIM]),
            ("attn.q_norm.weight", &[HEAD_DIM]),
            ("attn.k_norm.weight", &[HEAD_DIM]),
        ],
    )
}

fn attention_spec(qk_norm: QkNormNames<'static>, kind: AttentionKind) -> QkvAttentionSpec<'static> {
    QkvAttentionSpec {
        qkv: QkvSpec {
            hidden: HIDDEN as u32,
            heads: HEADS as u32,
            kv_heads: KV_HEADS as u32,
            head_dim: HEAD_DIM as u32,
        },
        names: QkvNames::Separate(["q_proj", "k_proj", "v_proj"]),
        bias: QkvBias::Absent,
        output: "o_proj",
        output_bias: Bias::Absent,
        qk_norm,
        kind,
        scale: 0.125,
    }
}

const DENSE: AttentionKind = AttentionKind::Dense {
    max_keys: 1024,
    causal: false,
    window: None,
};

const FUSED: QkNormNames<'static> = QkNormNames::Fused {
    query: "q_norm",
    key: "k_norm",
};

/// The stack's rotary over the block's heads, fused with the q/k norm or not.
fn rotary(loader: &mut WeightLoader<'_>, qk_norm: QkNorm) -> Rotary {
    Rotary::load(
        &mut loader.scope("rotary_emb"),
        Spectrum::Default {
            theta: 10_000.0,
            dim: HEAD_DIM as u32,
            precision: Precision::F64,
        },
        RotarySpec {
            head_dim: HEAD_DIM as u32,
            heads: HEADS as u32,
            kv_heads: KV_HEADS as u32,
            qk_norm,
        },
    )
    .unwrap()
}

/// Each q/k-norm form launches its own programs between the projections and the attention: the
/// gains fused into the rotation, a per-head RmsNorm of each projection, or the rotation alone.
#[test]
fn each_qk_norm_form_launches_its_own_programs() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = attention_checkpoint(directory.path());
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let fused = rotary(&mut loader, QkNorm::Fused { epsilon: 1e-6 });
    let bare = rotary(&mut loader, QkNorm::None);
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let (x, out) = (view(&arena, &[6, HIDDEN]), view(&arena, &[6, HIDDEN]));
    let factors = || {
        arena
            .tensor(DType::F32, Shape::new(&[6, HEAD_DIM / 2]))
            .unwrap()
    };
    let (cosine, sine) = (factors(), factors());
    let rope = |rotary| {
        Some(RopeFactors {
            rotary,
            cosine,
            sine,
        })
    };

    let cases = [
        (
            FUSED,
            rope(&fused),
            &[
                "gemm",
                "gemm",
                "gemm",
                "qk_norm_rope",
                "dense_attention",
                "gemm",
            ][..],
        ),
        (
            QkNormNames::Separate {
                query: "q_norm",
                key: "k_norm",
                epsilon: 1e-6,
            },
            None,
            &[
                "gemm",
                "gemm",
                "gemm",
                "rms_norm",
                "rms_norm",
                "dense_attention",
                "gemm",
            ][..],
        ),
        (
            QkNormNames::None,
            rope(&bare),
            &[
                "gemm",
                "gemm",
                "gemm",
                "rope_apply",
                "rope_apply",
                "dense_attention",
                "gemm",
            ][..],
        ),
    ];
    for (qk_norm, rope, expected) in cases {
        let block = QkvAttention::load(
            &mut loader.scope("attn"),
            attention_spec(qk_norm, DENSE),
            Split::WHOLE,
        )
        .unwrap();
        let used = arena.used();
        let (result, launches) = launched(|| block.forward(&f, x, rope, Keys::Dense(None), out));
        result.unwrap();
        assert_eq!(launches, expected, "{qk_norm:?}");
        assert_eq!(arena.used(), used, "the intermediates are released");
    }

    // A norm fused into the rotation needs the rotation.
    let block = QkvAttention::load(
        &mut loader.scope("attn"),
        attention_spec(FUSED, DENSE),
        Split::WHOLE,
    )
    .unwrap();
    assert!(block.forward(&f, x, None, Keys::Dense(None), out).is_err());
    // Packed sequences name their tables and their bounds.
    let offsets = arena.tensor(DType::I32, Shape::new(&[3])).unwrap();
    let varlen = Varlen {
        cu_q: offsets,
        cu_k: offsets,
        max_q: Some(3),
        max_k: 6,
    };
    assert!(
        block
            .forward(&f, x, rope(&fused), Keys::Dense(Some(varlen)), out)
            .is_ok()
    );
}

/// A paged block writes its keys and values into its layer's planes before it reads them, and
/// refuses keys of the other kind.
#[test]
fn a_paged_block_writes_its_layers_planes_then_reads_them() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = attention_checkpoint(directory.path());
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let fused = rotary(&mut loader, QkNorm::Fused { epsilon: 1e-6 });
    let paged = AttentionKind::Paged {
        page_size: 64,
        max_kv_len: 4096,
        window: None,
    };
    let block = QkvAttention::load(
        &mut loader.scope("attn"),
        attention_spec(FUSED, paged),
        Split::WHOLE,
    )
    .unwrap();

    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let tensor = |dtype: DType, dims: &[u64]| arena.tensor(dtype, Shape::new(dims)).unwrap();
    let tokens = 4;
    let batch = TokenBatch {
        positions: tensor(DType::I32, &[tokens]),
        slot_mapping: tensor(DType::I32, &[tokens]),
        seq_lens: tensor(DType::I32, &[1]),
        query_offsets: tensor(DType::I32, &[2]),
        logits_rows: tensor(DType::I32, &[1]),
        phase: PhaseKind::Prefill,
        page_size: 64,
        num_tokens: tokens as u32,
        max_kv_len: 64,
        draft_lens: None,
    };
    let pages = [1, 64, KV_HEADS, HEAD_DIM];
    let planes = [KvPlanes {
        key: tensor(DType::BF16, &pages),
        value: tensor(DType::BF16, &pages),
        key_initialized: tensor(DType::U8, &[1, 1, 1, 1]),
        value_initialized: tensor(DType::U8, &[1, 1, 1, 1]),
        key_scale: None,
        value_scale: None,
    }];
    let groups = [KvGroup {
        block_tables: tensor(DType::I32, &[1, 1]),
        page_size: 64,
        dtype: DType::BF16,
        layers: &planes,
    }];
    let cache = KvView { groups: &groups };
    let (x, out) = (
        view(&arena, &[tokens, HIDDEN]),
        view(&arena, &[tokens, HIDDEN]),
    );
    let (cosine, sine) = (
        tensor(DType::F32, &[tokens, HEAD_DIM / 2]),
        tensor(DType::F32, &[tokens, HEAD_DIM / 2]),
    );
    let rope = RopeFactors {
        rotary: &fused,
        cosine,
        sine,
    };
    let keys = Keys::Paged {
        batch: &batch,
        cache: &cache,
        group: 0,
        layer: 0,
        workspace: tensor(DType::U8, &[1 << 20]),
    };
    let (result, launches) = launched(|| block.forward(&f, x, Some(rope), keys, out));
    result.unwrap();
    assert_eq!(
        launches,
        [
            "gemm",
            "gemm",
            "gemm",
            "qk_norm_rope",
            "kv_write",
            "paged_attention",
            "gemm"
        ]
    );
    assert!(
        block
            .forward(&f, x, Some(rope), Keys::Dense(None), out)
            .is_err()
    );
}
