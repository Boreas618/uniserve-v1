//! The Qwen3.5-MoE family against a released checkpoint: it opens through the catalogue's family,
//! loads every module the checkpoint names at the extents the configuration states, and its
//! prefill and decode forwards compose over a cache that carries the recurrent state.
//!
//! The load is a meta load and the forwards are recording forwards: they read the checkpoint's
//! headers and launch nothing, so the test needs the checkpoint's files and no device.

use std::collections::BTreeSet;
use std::path::PathBuf;

use nn::PhaseKind;
use nn::capability::{KvGroup, KvPlanes, KvView, StateLayer, StateView, TokenBatch};
use nn::family::{Model, Split, TextModel, TextPlacement};
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

/// The programs one forward of `rows` rows of `tokens` tokens in `phase` launches, walked on a
/// recording forward over a cache of `slots` request slots.
fn recorded_forward(
    model: &dyn TextModel,
    phase: PhaseKind,
    rows: u64,
    tokens: u64,
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
    let state_layers: Vec<StateLayer<'_>> = banks
        .iter()
        .map(|(live, checkpoint)| StateLayer { live, checkpoint })
        .collect();
    let view = KvView {
        groups: &groups,
        workspace: tensor(DType::U8, &[1 << 20]),
        state: Some(StateView {
            slots: tensor(DType::I32, &[rows]),
            checkpoints: tensor(DType::I32, &[rows]),
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

    let prefill = recorded_forward(&*model, PhaseKind::Prefill, 2, 200);
    let decode = recorded_forward(&*model, PhaseKind::Decode, 4, 4);
    let operators = |launches: &BTreeSet<(String, String)>| -> BTreeSet<String> {
        launches
            .iter()
            .map(|(operator, _)| operator.clone())
            .collect()
    };
    for (phase, launches) in [("prefill", &prefill), ("decode", &decode)] {
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
    for (operator, key) in prefill.union(&decode) {
        println!("{operator} {key}");
    }
}
