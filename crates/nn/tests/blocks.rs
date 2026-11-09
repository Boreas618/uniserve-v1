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
        bias: nn::layers::Bias::Absent,
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
    let cache = KvView {
        groups: &groups,
        workspace: tensor(DType::U8, &[1 << 20]),
    };
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

// ---- Timestep embedding -------------------------------------------------------------------------

/// The timestep embedding projects, activates and projects, and leaves the arena as it found it.
#[test]
fn a_timestep_embedding_projects_activates_and_projects() {
    use nn::timestep::{TimestepEmbedding, TimestepSpec};

    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("time.linear_1.weight", &[HIDDEN, 16]),
            ("time.linear_1.bias", &[HIDDEN]),
            ("time.linear_2.weight", &[INTERMEDIATE, HIDDEN]),
            ("time.linear_2.bias", &[INTERMEDIATE]),
        ],
    );
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let embedding = TimestepEmbedding::load(
        &mut loader.scope("time"),
        TimestepSpec {
            frequency_dim: 16,
            hidden: HIDDEN as u32,
            out: INTERMEDIATE as u32,
        },
    )
    .unwrap();
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let (x, out) = (view(&arena, &[2, 16]), view(&arena, &[2, INTERMEDIATE]));
    let used = arena.used();
    let (result, launches) = launched(|| embedding.forward(&f, x, out));
    result.unwrap();
    assert_eq!(launches.len(), 3, "{launches:?}");
    assert_eq!(
        (launches[0].as_str(), launches[2].as_str()),
        ("gemm", "gemm")
    );
    assert_eq!(arena.used(), used);
}

// ---- Step-indexed projection --------------------------------------------------------------------

/// A step-indexed projection gathers its step's weight and bias rows, projects with them, and
/// releases the gathered rows; tables of another extent or type are refused.
#[test]
fn a_step_indexed_projection_gathers_its_steps_row_then_projects() {
    use nn::Weight;
    use nn::layers::StepIndexed;

    let arena = Arena::counting();
    let table = |dtype: DType, dims: &[u64]| {
        Weight::borrowed(arena.tensor(dtype, Shape::new(dims)).unwrap()).unwrap()
    };
    let (steps, inputs, outputs) = (4, 32, 8);
    let projection = StepIndexed::new(
        table(DType::F32, &[steps, outputs * inputs]),
        table(DType::F32, &[steps, outputs]),
        inputs as u32,
        outputs as u32,
    )
    .unwrap();
    assert_eq!(projection.steps(), steps);

    let f = Forward::recording(&arena);
    let step = arena.tensor(DType::I32, Shape::new(&[1])).unwrap();
    let input = arena.tensor(DType::F32, Shape::new(&[3, inputs])).unwrap();
    let out = arena.tensor(DType::F32, Shape::new(&[3, outputs])).unwrap();
    let used = arena.used();
    let (result, launches) = launched(|| projection.forward(&f, step, input, out));
    result.unwrap();
    assert_eq!(launches.len(), 3, "{launches:?}");
    assert_eq!(launches[2], "gemm");
    assert_eq!(launches[0], launches[1], "one gather per table");
    assert_eq!(arena.used(), used);

    assert!(
        StepIndexed::new(
            table(DType::F32, &[steps, outputs * inputs]),
            table(DType::BF16, &[steps, outputs]),
            inputs as u32,
            outputs as u32,
        )
        .is_err()
    );
    assert!(
        StepIndexed::new(
            table(DType::F32, &[steps, outputs * inputs]),
            table(DType::F32, &[steps - 1, outputs]),
            inputs as u32,
            outputs as u32,
        )
        .is_err()
    );
}

// ---- AdaLN block, final norm and timestep table ---------------------------------------------------

use nn::blocks::{AdaLnBlock, AdaLnOut, AdaLnSpec, ModulationTable};
use nn::diffusion::{TimestepTable, TimestepTableSpec};

/// The step's gathered modulation of `parameters` rows of `vectors` vectors each.
fn modulation(arena: &Arena, parameters: u64, vectors: u64) -> TensorView {
    view(arena, &[parameters, vectors * HIDDEN])
}

/// A block's table is read by stage, attention then MLP, each vector a row-strided view of the
/// table; the final norm's table holds a shift and a scale; each refuses the other's reading.
#[test]
fn a_modulation_table_cuts_shift_scale_and_gate_in_storage_order() {
    let arena = Arena::counting();
    let table = modulation(&arena, 4, 6);
    let block = ModulationTable::block(table, HIDDEN).unwrap();
    let address = |weight: nn::Weight| weight.view().ptr.address() - table.ptr.address();
    let row = 2 * HIDDEN;
    let (attention, mlp) = (block.attention().unwrap(), block.mlp().unwrap());
    let offsets = [
        attention.shift,
        attention.scale,
        attention.gate,
        mlp.shift,
        mlp.scale,
        mlp.gate,
    ]
    .map(address);
    assert_eq!(offsets, [0, 1, 2, 3, 4, 5].map(|vector| vector * row));
    let shift = attention.shift.view();
    assert_eq!(shift.shape.dims(), &[4, HIDDEN]);
    assert_eq!(
        shift.strides.get(0),
        6 * HIDDEN,
        "a vector strides over the whole row"
    );
    assert!(block.shift_scale().is_err());

    let output = ModulationTable::output(modulation(&arena, 4, 2), HIDDEN).unwrap();
    let (shift, scale) = output.shift_scale().unwrap();
    assert_eq!(scale.view().ptr.address() - shift.view().ptr.address(), row);
    assert!(output.attention().is_err());

    assert!(ModulationTable::block(modulation(&arena, 4, 2), HIDDEN).is_err());
    assert!(ModulationTable::output(view(&arena, &[4, 2, 2 * HIDDEN]), HIDDEN).is_err());
}

fn adaln_checkpoint(directory: &std::path::Path) -> Checkpoint {
    checkpoint(
        directory,
        &[
            ("block.norm1.weight", &[HIDDEN]),
            ("block.norm2.weight", &[HIDDEN]),
            ("block.ff.proj.weight", &[2 * INTERMEDIATE, HIDDEN]),
            ("block.ff.down_proj.weight", &[HIDDEN, INTERMEDIATE]),
            ("norm_out.weight", &[HIDDEN]),
        ],
    )
}

/// The block normalizes the attention's input under its modulation, runs the caller's attention
/// on it, folds the gated update into one modulated norm for the MLP, and joins the MLP's output
/// through its gate.
#[test]
fn an_adaln_block_modulates_both_inputs_and_gates_both_updates() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = adaln_checkpoint(directory.path());
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let block = AdaLnBlock::load(
        &mut loader.scope("block"),
        AdaLnSpec {
            hidden: HIDDEN as u32,
            eps: 1e-6,
            norm1: "norm1",
            norm2: "norm2",
            mlp: GatedMlpSpec {
                down: "ff.down_proj",
                ..mlp_spec(Rounding::Stepwise, GateUp::ValueFirst("ff.proj"))
            },
        },
        Split::WHOLE,
    )
    .unwrap();
    assert_eq!(block.dtype(), DType::BF16);

    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let tokens = 5;
    let table = ModulationTable::block(modulation(&arena, 6, 6), HIDDEN).unwrap();
    let rows = arena.tensor(DType::I32, Shape::new(&[tokens])).unwrap();
    let [hidden, normalized, update, residual] = [0; 4].map(|_| view(&arena, &[tokens, HIDDEN]));

    let mut attended = None;
    let (result, launches) = launched(|| {
        block.forward(
            &f,
            &table,
            rows,
            hidden,
            normalized,
            update,
            residual,
            |input, output| {
                attended = Some((input.ptr, output.ptr));
                Ok(())
            },
        )
    });
    result.unwrap();
    assert_eq!(attended, Some((normalized.ptr, update.ptr)));
    assert_eq!(
        launches,
        [
            "modulated_rms_norm",
            "modulated_rms_norm",
            "gemm",
            "act_and_mul",
            "gemm",
            "gated_residual",
        ]
    );

    // A final norm's table carries no gated stage.
    let output = ModulationTable::output(modulation(&arena, 6, 2), HIDDEN).unwrap();
    assert!(
        block
            .attention_input(&f, &output, rows, hidden, normalized)
            .is_err()
    );
}

/// The final norm is one modulated launch into a buffer of its input's shape.
#[test]
fn the_final_adaln_norm_modulates_by_shift_and_scale() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = adaln_checkpoint(directory.path());
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let norm = AdaLnOut::load(&mut loader.scope("norm_out"), HIDDEN as u32, 1e-6).unwrap();
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let table = ModulationTable::output(modulation(&arena, 2, 2), HIDDEN).unwrap();
    let rows = arena.tensor(DType::I32, Shape::new(&[3])).unwrap();
    let hidden = view(&arena, &[3, HIDDEN]);
    let (out, launches) = launched(|| norm.apply(&f, &table, rows, hidden));
    let out = out.unwrap();
    assert_eq!((out.shape, out.dtype), (hidden.shape, hidden.dtype));
    assert_eq!(launches, ["modulated_rms_norm"]);

    let block_table = ModulationTable::block(modulation(&arena, 2, 6), HIDDEN).unwrap();
    assert!(norm.forward(&f, &block_table, rows, hidden, out).is_err());
}

/// The table holds one row per distinct level and the entry of each step's groups, and states the
/// modulation tables the runtime allocates; bound to them, a step gathers its entries, then each
/// block's rows at them.
#[test]
fn a_timestep_table_holds_each_distinct_level_once() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = adaln_checkpoint(directory.path());
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    // Three steps of two groups; the last step repeats the first's levels.
    let levels = [1.0, 0.75, 0.5, 0.25, 1.0, 0.75];
    let table = TimestepTable::load(
        &mut loader.scope("adaln"),
        &levels,
        2,
        TimestepTableSpec {
            frequency_dim: 16,
            layers: 2,
            layer_columns: 6 * HIDDEN,
            output_columns: 2 * HIDDEN,
            dtype: DType::BF16,
        },
    )
    .unwrap();
    assert_eq!((table.groups(), table.steps()), (2, 3));
    assert_eq!(table.frequencies().shape.dims(), &[4, 16]);
    let tables = table.tables();
    assert_eq!(tables.len(), 2);
    assert_eq!(tables[0].shape.dims(), &[2, 4, 6 * HIDDEN]);
    assert_eq!(tables[1].shape.dims(), &[4, 2 * HIDDEN]);

    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let views: Vec<_> = tables
        .iter()
        .map(|stated| arena.tensor(stated.dtype, stated.shape).unwrap())
        .collect();
    let bound = table.bind(&views).unwrap();
    assert_eq!(bound.layer_table(1).unwrap().shape.dims(), &[4, 6 * HIDDEN]);
    assert!(table.bind(&views[..1]).is_err(), "both tables are bound");
    let step = arena.tensor(DType::I32, Shape::new(&[1])).unwrap();
    let entries = arena.tensor(DType::I32, Shape::new(&[1, 2])).unwrap();
    let rows = modulation(&arena, 2, 6);
    let (result, launches) = launched(|| {
        table.entries_of(&f, step, entries)?;
        bound.layer(&f, 1, entries, rows)
    });
    result.unwrap();
    assert_eq!(launches.len(), 2);
    assert!(bound.layer(&f, 2, entries, rows).is_err(), "two blocks");

    let (times, indices) = TimestepTable::entries(&levels, 2).unwrap();
    assert_eq!(times, [1.0, 0.75, 0.5, 0.25]);
    assert_eq!(indices, [0, 1, 2, 3, 0, 1]);
    assert!(
        TimestepTable::entries(&levels, 4).is_err(),
        "not whole steps"
    );
}

// ---- Patch embedding -----------------------------------------------------------------------------

/// A patch embedding packs `1x2x2` patch tokens of every channel, projects them into its width,
/// and unpacks tokens back into a raster; its projection is the module at its scope.
#[test]
fn a_patch_embedding_packs_projects_and_unpacks_its_tokens() {
    use nn::layers::{PatchEmbed, PatchEmbedSpec};

    let (channels, width) = (8_u64, 32_u64);
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("proj_in.weight", &[width, channels * 4]),
            ("proj_in.bias", &[width]),
        ],
    );
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let spec = PatchEmbedSpec {
        channels: channels as u32,
        temporal: 1,
        spatial: 2,
        hidden: width as u32,
    };
    let embed = PatchEmbed::load(&mut loader.scope("proj_in"), spec).unwrap();
    assert_eq!(embed.token_width(), channels * 4);
    assert_eq!(embed.weight().shape.dims(), &[width, channels * 4]);

    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    // Two frames of a 4x6 raster: 2 · 2 · 3 tokens.
    let latent = view(&arena, &[channels, 2, 4, 6]);
    let tokens = view(&arena, &[12, channels * 4]);
    let hidden = view(&arena, &[12, width]);
    let (result, launches) = launched(|| {
        embed.patchify(&f, latent, tokens)?;
        embed.project(&f, tokens, hidden)?;
        embed.unpatchify(&f, tokens, latent)
    });
    result.unwrap();
    assert_eq!(launches, ["patch", "gemm", "unpatchify_video"]);

    // A patch of no rows is refused before any tensor is read.
    let empty = PatchEmbedSpec { spatial: 0, ..spec };
    assert!(PatchEmbed::load(&mut loader.scope("proj_in"), empty).is_err());
}

// ---- Pre-norm block ------------------------------------------------------------------------------

use nn::blocks::{Pending, PreNormBlock, PreNormSpec, ResidualForm};

fn prenorm_spec(residual: ResidualForm, layer_scale: Option<[&str; 2]>) -> PreNormSpec<'_> {
    PreNormSpec {
        hidden: HIDDEN as u32,
        eps: 1e-6,
        norm1: "norm1",
        norm2: "norm2",
        layer_scale,
        norm_dtype: Some(DType::F32),
        residual,
        mlp: GatedMlpSpec {
            down: "ff.down_proj",
            ..mlp_spec(Rounding::Stepwise, GateUp::ValueFirst("ff.proj"))
        },
    }
}

/// Two blocks and a final join: a separate block joins each update in its own launch before the
/// next norm, a fused block folds it into that norm (under the layer scale when it has one), and
/// the attention reads the first norm's output and writes the update the second norm joins.
#[test]
fn a_pre_norm_block_joins_each_update_in_its_residual_form() {
    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(
        directory.path(),
        &[
            ("block.norm1.weight", &[HIDDEN]),
            ("block.norm2.weight", &[HIDDEN]),
            ("block.scale1", &[HIDDEN]),
            ("block.scale2", &[HIDDEN]),
            ("block.ff.proj.weight", &[2 * INTERMEDIATE, HIDDEN]),
            ("block.ff.down_proj.weight", &[HIDDEN, INTERMEDIATE]),
        ],
    );
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let tokens = 4;
    let stream = arena
        .tensor(DType::F32, Shape::new(&[tokens, HIDDEN]))
        .unwrap();
    let [normalized, update] = [0; 2].map(|_| view(&arena, &[tokens, HIDDEN]));

    // The stream is FP32 and the norms write the BF16 rows the projections read, so a norm that
    // reads the stream alone is the narrowing one.
    let cases: [(_, _, &[&str]); 2] = [
        (
            ResidualForm::Separate,
            None,
            &["scaled_add", "weighted_rms_norm"],
        ),
        (
            ResidualForm::Fused,
            Some(["scale1", "scale2"]),
            &["scaled_residual_rms_norm"],
        ),
    ];
    for (residual, layer_scale, joined) in cases {
        let block = PreNormBlock::load(
            &mut loader.scope("block"),
            prenorm_spec(residual, layer_scale),
            Split::WHOLE,
        )
        .unwrap();
        let mut attended = Vec::new();
        let (last, launches) = launched(|| -> nn::Result<Pending> {
            let mut pending = None;
            for _ in 0..2 {
                let attend = |input: TensorView, output: TensorView| {
                    attended.push((input.ptr, output.ptr));
                    Ok(())
                };
                pending = Some(block.forward(&f, stream, pending, normalized, update, attend)?);
            }
            Ok(pending.unwrap())
        });
        let last = last.unwrap();
        assert_eq!(attended, [(normalized.ptr, update.ptr); 2]);
        assert_eq!(last.update.ptr, update.ptr);
        assert_eq!(last.scale.is_some(), layer_scale.is_some());

        // The first norm reads the FP32 stream alone and narrows it; every later norm joins the
        // update before it.
        let mlp = ["gemm", "act_and_mul", "gemm"];
        let mut expected = vec!["weighted_rms_norm"];
        expected.extend(joined);
        expected.extend(mlp);
        expected.extend(joined);
        expected.extend(joined);
        expected.extend(mlp);
        assert_eq!(launches, expected, "{residual:?}");
    }

    // The separate join of a stack's last update is one plain add into the stream.
    let last = Pending {
        update,
        scale: None,
    };
    let (result, launches) = launched(|| last.join(&f, stream));
    result.unwrap();
    assert_eq!(launches, ["scaled_add"]);
}

// ---- Noising -------------------------------------------------------------------------------------

/// A clean sample is noised in place in one launch, from a noise of its own shape only.
#[test]
fn add_noise_mixes_a_sample_with_noise_of_its_shape_in_place() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    let sample = arena.tensor(DType::F32, Shape::new(&[6, HIDDEN])).unwrap();
    let noise = arena.tensor(DType::F32, Shape::new(&[6, HIDDEN])).unwrap();
    let (result, launches) = launched(|| nn::diffusion::add_noise(&f, sample, noise, 0.999));
    result.unwrap();
    assert_eq!(launches, ["scaled_add"]);

    let short = arena.tensor(DType::F32, Shape::new(&[5, HIDDEN])).unwrap();
    assert!(nn::diffusion::add_noise(&f, sample, short, 0.999).is_err());
}
