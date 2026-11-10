//! The Qwen3.5-MoE family against a released checkpoint: it opens through the catalogue's family,
//! loads every module the checkpoint names at the extents the configuration states (its drafter's
//! among them), and its prefill, decode and verification forwards and its drafter's forward compose
//! over a cache that carries the recurrent state.
//!
//! The load is a meta load and the forwards are recording forwards: they read the checkpoint's
//! headers and launch nothing, so the test needs the checkpoint's files and no device.

use std::collections::BTreeSet;
use std::path::PathBuf;

use nn::PhaseKind;
use nn::capability::{KvGroup, KvPlanes, KvView, StateLayer, StateView, TokenBatch};
use nn::family::{CacheShape, Model, Split, TextModel, TextPlacement};
use nn::{Checkpoint, DType, Shape, TensorView, WeightLoader};

/// The checkpoint the test reads, from `UNISERVE_QWEN36_MODEL`.
fn checkpoint_root() -> PathBuf {
    std::env::var_os("UNISERVE_QWEN36_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("UNISERVE_QWEN36_MODEL names a Qwen3.6-35B-A3B checkpoint"))
}

/// The programs a launch reported, as `(operator, key)` pairs.
#[derive(Default)]
struct Launches(BTreeSet<(String, String)>);

impl ops::capture::Sink for Launches {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.insert((operator.to_owned(), key.to_owned()));
    }
}

/// How a recorded forward's state rows lie over its attention rows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StateRows {
    /// One state row per attention row: a prefill or a decode.
    Attention,
    /// A verification: one attention row per token, `calls` state rows walking them, and a trail.
    Verification { calls: u64 },
}

/// The programs one forward of `rows` rows of `tokens` tokens in `phase` launches, walked on a
/// recording forward over a cache of `slots` request slots, its state rows laid out as `state`
/// says.
fn recorded_forward(
    model: &dyn TextModel,
    phase: PhaseKind,
    rows: u64,
    tokens: u64,
    state: StateRows,
) -> BTreeSet<(String, String)> {
    let arena = nn::Arena::counting();
    let f = nn::Forward::recording(&arena);
    let tensor = |dtype: DType, dims: &[u64]| -> TensorView {
        arena
            .tensor(dtype, Shape::new(dims))
            .unwrap_or_else(|error| panic!("a counting arena holds any reservation: {error}"))
    };
    let page_size = 64_u32;
    let row_pages = 4_u64;
    let pages = rows * row_pages;
    let batch = TokenBatch {
        positions: tensor(DType::I32, &[tokens]),
        slot_mapping: tensor(DType::I32, &[tokens]),
        seq_lens: tensor(DType::I32, &[rows]),
        query_offsets: tensor(DType::I32, &[rows + 1]),
        logits_rows: tensor(DType::I32, &[rows]),
        phase,
        page_size,
        num_tokens: u32::try_from(tokens).unwrap_or_else(|_| panic!("{tokens} tokens")),
        max_kv_len: page_size * row_pages as u32,
        draft_lens: None,
    };
    let cache = model.cache();
    let planes_shape = [
        pages,
        u64::from(page_size),
        u64::from(cache.kv_heads),
        u64::from(cache.head_dim),
    ];
    let layers: Vec<KvPlanes> = (0..cache.layers)
        .map(|_| KvPlanes {
            key: tensor(DType::BF16, &planes_shape),
            value: tensor(DType::BF16, &planes_shape),
            key_initialized: tensor(DType::U8, &[pages, 1, 1, 1]),
            value_initialized: tensor(DType::U8, &[pages, 1, 1, 1]),
            key_scale: None,
            value_scale: None,
        })
        .collect();
    let groups = [KvGroup {
        block_tables: tensor(DType::I32, &[rows, row_pages]),
        page_size,
        dtype: DType::BF16,
        layers: &layers,
    }];

    // Every recurrent layer's live and checkpoint banks, as the infrastructure binds them.
    let shape = model
        .state()
        .unwrap_or_else(|| panic!("a hybrid model keeps a recurrent state"));
    let slots = 8_u64;
    let bank = |field: &nn::family::StateField| {
        let mut dims = vec![slots];
        dims.extend(&field.dims);
        tensor(field.dtype, &dims)
    };
    let banks: Vec<(Vec<TensorView>, Vec<TensorView>)> = (0..shape.layers)
        .map(|_| {
            (
                shape.fields.iter().map(bank).collect(),
                shape.fields.iter().map(bank).collect(),
            )
        })
        .collect();
    // A verification keeps a trail: the state after every token, per field of every layer.
    let trail_bank = |field: &nn::family::StateField| {
        let mut dims = vec![tokens];
        dims.extend(&field.dims);
        tensor(field.dtype, &dims)
    };
    let trails: Vec<Vec<TensorView>> = (0..shape.layers)
        .map(|_| shape.fields.iter().map(trail_bank).collect())
        .collect();
    let state_rows = match state {
        StateRows::Attention => rows,
        StateRows::Verification { calls } => calls,
    };
    let state_layers: Vec<StateLayer<'_>> = banks
        .iter()
        .zip(&trails)
        .map(|((live, checkpoint), trail)| StateLayer {
            live,
            checkpoint,
            trail: matches!(state, StateRows::Verification { .. }).then_some(trail.as_slice()),
        })
        .collect();
    let view = KvView {
        groups: &groups,
        workspace: tensor(DType::U8, &[1 << 20]),
        state: Some(StateView {
            query_offsets: tensor(DType::I32, &[state_rows + 1]),
            seq_lens: tensor(DType::I32, &[state_rows]),
            slots: tensor(DType::I32, &[state_rows]),
            checkpoints: tensor(DType::I32, &[state_rows]),
            layers: &state_layers,
        }),
    };
    let hidden = tensor(DType::BF16, &[tokens, u64::from(model.hidden())]);

    let mut launches = Launches::default();
    ops::capture::set_sink(Some(&mut launches));
    let walked = model.forward(&f, &batch, &view, hidden);
    ops::capture::set_sink(None);
    walked.unwrap_or_else(|error| panic!("the recorded {phase:?} forward: {error}"));
    launches.0
}

/// The planes of `cache`'s layers for `rows` rows of `row_pages` pages of `page_size` tokens.
fn planes(
    tensor: &dyn Fn(DType, &[u64]) -> TensorView,
    cache: CacheShape,
    rows: u64,
    page_size: u32,
    row_pages: u64,
) -> (TensorView, Vec<KvPlanes>) {
    let pages = rows * row_pages;
    let shape = [
        pages,
        u64::from(page_size),
        u64::from(cache.kv_heads),
        u64::from(cache.head_dim),
    ];
    let layers = (0..cache.layers)
        .map(|_| KvPlanes {
            key: tensor(DType::BF16, &shape),
            value: tensor(DType::BF16, &shape),
            key_initialized: tensor(DType::U8, &[pages, 1, 1, 1]),
            value_initialized: tensor(DType::U8, &[pages, 1, 1, 1]),
            key_scale: None,
            value_scale: None,
        })
        .collect();
    (tensor(DType::I32, &[rows, row_pages]), layers)
}

/// The programs one drafter forward of `rows` rows of `tokens` tokens in `phase` launches over a
/// draft cache of the drafter's own layers.
fn recorded_draft(
    model: &dyn TextModel,
    phase: PhaseKind,
    rows: u64,
    tokens: u64,
) -> BTreeSet<(String, String)> {
    let drafter = model
        .drafter()
        .unwrap_or_else(|| panic!("the model loaded its drafter"));
    let arena = nn::Arena::counting();
    let f = nn::Forward::recording(&arena);
    let tensor = |dtype: DType, dims: &[u64]| -> TensorView {
        arena
            .tensor(dtype, Shape::new(dims))
            .unwrap_or_else(|error| panic!("a counting arena holds any reservation: {error}"))
    };
    let page_size = 64_u32;
    let row_pages = 4_u64;
    let batch = TokenBatch {
        positions: tensor(DType::I32, &[tokens]),
        slot_mapping: tensor(DType::I32, &[tokens]),
        seq_lens: tensor(DType::I32, &[rows]),
        query_offsets: tensor(DType::I32, &[rows + 1]),
        logits_rows: tensor(DType::I32, &[rows]),
        phase,
        page_size,
        num_tokens: u32::try_from(tokens).unwrap_or_else(|_| panic!("{tokens} tokens")),
        max_kv_len: page_size * row_pages as u32,
        draft_lens: None,
    };
    let (tables, layers) = planes(&tensor, drafter.cache(), rows, page_size, row_pages);
    let groups = [KvGroup {
        block_tables: tables,
        page_size,
        dtype: DType::BF16,
        layers: &layers,
    }];
    let view = KvView {
        groups: &groups,
        workspace: tensor(DType::U8, &[1 << 20]),
        state: None,
    };
    let hidden = |_: ()| tensor(DType::BF16, &[tokens, u64::from(model.hidden())]);
    let (embedded, states) = (hidden(()), hidden(()));

    let mut launches = Launches::default();
    ops::capture::set_sink(Some(&mut launches));
    let walked = drafter.forward(&f, &batch, &view, embedded, states);
    ops::capture::set_sink(None);
    walked.unwrap_or_else(|error| panic!("the recorded {phase:?} drafter forward: {error}"));
    launches.0
}

/// The released checkpoint loads every module at its stated extents and serves both phases.
#[test]
#[ignore = "requires UNISERVE_QWEN36_MODEL naming a Qwen3.6-35B-A3B checkpoint"]
fn a_released_hybrid_checkpoint_loads_and_composes_both_phases() {
    let Model::Text(family) = qwen3::family::HYBRID_FAMILY.model else {
        panic!("Qwen3.5-MoE is a text family");
    };
    let checkpoint =
        Checkpoint::open(&checkpoint_root()).unwrap_or_else(|error| panic!("the shards: {error}"));
    let opened = family
        .open(&checkpoint)
        .unwrap_or_else(|error| panic!("the configuration: {error}"));
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let placement = TextPlacement {
        tensor: Split::WHOLE,
        experts: Split::WHOLE,
        drafter: true,
    };
    let model = opened
        .load(
            &checkpoint,
            &mut loader,
            &placement,
            &nn::family::ExpertOrder::new(),
        )
        .unwrap_or_else(|error| panic!("the meta load: {error}"));

    // Ten attention layers keep keys and values; thirty linear layers keep a convolution window of
    // three inputs of 8192 channels and a 128 × 128 FP32 state per value head.
    let cache = model.cache();
    assert_eq!((cache.layers, cache.kv_heads, cache.head_dim), (10, 2, 256));
    let state = model.state().unwrap_or_else(|| panic!("a recurrent state"));
    assert_eq!(state.layers, 30);
    assert_eq!(state.fields[0].dims, vec![3, 8192]);
    assert_eq!(state.fields[1].dims, vec![32, 128, 128]);
    assert_eq!(state.fields[1].dtype, DType::F32);

    let prefill = recorded_forward(&*model, PhaseKind::Prefill, 2, 200, StateRows::Attention);
    let decode = recorded_forward(&*model, PhaseKind::Decode, 4, 4, StateRows::Attention);
    // A verification of two calls of four tokens each: eight one-token attention rows, two state
    // rows walking them, and the trail.
    let verification = recorded_forward(
        &*model,
        PhaseKind::Decode,
        8,
        8,
        StateRows::Verification { calls: 2 },
    );
    let operators = |launches: &BTreeSet<(String, String)>| -> BTreeSet<String> {
        launches
            .iter()
            .map(|(operator, _)| operator.clone())
            .collect()
    };
    for (phase, launches) in [
        ("prefill", &prefill),
        ("decode", &decode),
        ("verification", &verification),
    ] {
        let named = operators(launches);
        for operator in [
            "causal_conv1d",
            "gated_delta_rule",
            "gated_rms_norm",
            "sigmoid_gated_add",
            "qk_norm_rope",
            "act_and_mul",
            "paged_attention",
        ] {
            assert!(
                named.contains(operator),
                "{phase} launches no {operator}: {named:?}"
            );
        }
    }
    let rule = |launches: &BTreeSet<(String, String)>| -> Vec<String> {
        launches
            .iter()
            .filter(|(operator, _)| operator == "gated_delta_rule")
            .map(|(_, key)| key.clone())
            .collect()
    };
    assert!(
        rule(&prefill)
            .iter()
            .all(|key| key.contains("form_chunked"))
    );
    assert!(
        rule(&decode)
            .iter()
            .all(|key| key.contains("form_recurrent"))
    );
    // The trail is a launch operand and not a program: a verification walks the decode's programs.
    assert_eq!(verification, decode);

    // One drafter layer over a draft cache of its own: two heads of 256 per layer, and both forms
    // it runs, a prompt's entries and a drafted token's, compose through the model's gated
    // attention and routed block.
    let drafter = model.drafter().unwrap_or_else(|| panic!("the drafter"));
    let draft_cache = drafter.cache();
    assert_eq!(
        (
            draft_cache.layers,
            draft_cache.kv_heads,
            draft_cache.head_dim
        ),
        (1, 2, 256)
    );
    let entries = recorded_draft(&*model, PhaseKind::Prefill, 2, 200);
    let drafted = recorded_draft(&*model, PhaseKind::Decode, 4, 4);
    for (phase, launches) in [("entries", &entries), ("drafted", &drafted)] {
        let named = operators(launches);
        for operator in [
            "rms_norm",
            "copy_block",
            "qk_norm_rope",
            "paged_attention",
            "act_and_mul",
        ] {
            assert!(
                named.contains(operator),
                "the drafter's {phase} launch no {operator}: {named:?}"
            );
        }
        assert!(!named.contains("gated_delta_rule"), "{named:?}");
    }
    for (operator, key) in prefill.union(&decode).chain(entries.union(&drafted)) {
        println!("{operator} {key}");
    }
}

/// The calibrated mixed-precision export loads its attention and delta-network projections as
/// per-tensor FP8 products and its experts, shared experts and head as the dense matrices their
/// weight-only NVFP4 fields state, and serves both phases with no encoded-activation program.
#[test]
#[ignore = "requires UNISERVE_QWEN36_NVFP4_MODEL naming a Qwen3.6-35B-A3B-NVFP4 checkpoint"]
fn a_mixed_precision_hybrid_checkpoint_loads_fp8_and_weight_only_nvfp4() {
    let Model::Text(family) = qwen3::family::HYBRID_FAMILY.model else {
        panic!("Qwen3.5-MoE is a text family");
    };
    let root = std::env::var_os("UNISERVE_QWEN36_NVFP4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("UNISERVE_QWEN36_NVFP4_MODEL names the NVFP4 checkpoint"));
    let checkpoint = Checkpoint::open(&root).unwrap_or_else(|error| panic!("the shards: {error}"));
    let opened = family
        .open(&checkpoint)
        .unwrap_or_else(|error| panic!("the configuration: {error}"));
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    let model = opened
        .load(
            &checkpoint,
            &mut loader,
            &TextPlacement {
                tensor: Split::WHOLE,
                experts: Split::WHOLE,
                drafter: false,
            },
            &nn::family::ExpertOrder::new(),
        )
        .unwrap_or_else(|error| panic!("the meta load: {error}"));
    // The experts are dense once decoded: no step encodes activations for them.
    let experts = model.experts().unwrap_or_else(|| panic!("routed experts"));
    assert!(!experts.encoded);

    let mut launches = recorded_forward(&*model, PhaseKind::Prefill, 2, 200, StateRows::Attention);
    launches.extend(recorded_forward(
        &*model,
        PhaseKind::Decode,
        4,
        4,
        StateRows::Attention,
    ));
    let named: BTreeSet<&str> = launches
        .iter()
        .map(|(operator, _)| operator.as_str())
        .collect();
    assert!(named.contains("fp8_gemm"), "{named:?}");
    for encoded in [
        "nvfp4_quantize",
        "nvfp4_gemm",
        "gather_gemm_nvfp4",
        "route_gemm_nvfp4",
    ] {
        assert!(!named.contains(encoded), "{encoded} runs: {named:?}");
    }
    for (operator, key) in &launches {
        println!("{operator} {key}");
    }
}
