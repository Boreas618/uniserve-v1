//! Profile runs on the meta device over the released checkpoints' headers.
//!
//! A meta load reads a checkpoint's shard headers and no tensor data, and a forward over meta
//! tensors under a meta context launches nothing: it checks every operand, states the kernel
//! each launch would run and counts what the forward allocates. Each test loads a released
//! checkpoint through its family, runs one step of a decode bucket and of a prefill bucket
//! (the embedding, the decoder stack and the logits), and asserts the shapes of the outputs,
//! that every launch names an operator, that no kernel key names a model family, and that a
//! second run of the same bucket states the same keys and reaches the same allocator peak. It
//! prints the keys and the peaks.
//!
//! The tests read the releases of [`crate::releases`] through the Hugging Face cache and need
//! no device; they are ignored by default and fail, naming the release, when asked for without
//! it:
//!
//! ```text
//! cargo test -p models --test it qwen3::profile -- --ignored --nocapture
//! ```

use std::collections::BTreeSet;
use std::sync::Arc;

use models::qwen3::{Config, Qwen3VlModel};
use nn::config::Fields;
use nn::family::{CacheField, Family, Projection, SequenceSplit, TextModel, TextPlacement};
use nn::{
    DType, Encoder, ExpertExchange, KvCache, MetaWeights, Phase, RecurrentState, Shape, Split,
    WeightLoader,
};
use nn_testing::text::{self, WORKSPACE_BYTES, key_value_shape};
use nn_testing::{
    PAGE_TOKENS, Recording, Release, assert_generic, assert_operators, meta, record, report,
};
use processing::qwen_vl::ProcessorConfig;

use crate::releases::{
    MINIMAX_H3, QWEN3_6_35B_A3B, QWEN3_6_35B_A3B_NVFP4, QWEN3_30B_A3B, QWEN3_30B_A3B_NVFP4,
};

/// The families the profiled checkpoints declare.
const FAMILIES: [Family; 2] = [
    models::qwen3::family::FAMILY,
    models::qwen3::family::HYBRID_FAMILY,
];

/// Request slots of the recurrent state's buffers.
const SLOTS: usize = 8;

/// The kernels a forward stated: `(operator, key)` pairs.
type Keys = BTreeSet<(&'static str, String)>;

/// The model of `release`, meta-loaded for `placement`.
fn load(release: Release, placement: &TextPlacement) -> Arc<dyn TextModel> {
    let checkpoint = release.checkpoint();
    let store = MetaWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);

    text::load(&FAMILIES, &checkpoint, &mut loader, placement)
        .unwrap_or_else(|error| panic!("{release}: {error}"))
}

/// The distinct kernels of `recording`.
fn keys(recording: &Recording) -> Keys {
    recording.launches.iter().cloned().collect()
}

/// The index columns of a meta batch of `rows` rows of `tokens` tokens in all, whose rows may
/// each read `row_pages` pages.
fn batch(phase: Phase, rows: usize, tokens: usize, row_pages: usize) -> nn::TokenBatch {
    text::batch(
        phase,
        [
            meta(tokens, DType::I32),
            meta(rows, DType::U32),
            meta(rows + 1, DType::U32),
            meta(rows, DType::U32),
        ],
        None,
        PAGE_TOKENS,
        PAGE_TOKENS * row_pages as u32,
    )
}

/// The meta cache group of `fields` in `dtype` over `rows` rows of `row_pages` pages each, lent
/// to a forward of `tokens` tokens.
fn cache(
    fields: &[CacheField],
    dtype: DType,
    (rows, row_pages): (usize, usize),
    tokens: usize,
) -> KvCache {
    text::cache(
        fields,
        dtype,
        (rows * row_pages, PAGE_TOKENS),
        (
            meta((rows, row_pages), DType::U32),
            meta(tokens, DType::U32),
            meta(WORKSPACE_BYTES, DType::U8),
        ),
        |shape, dtype| Ok(meta(shape, dtype)),
    )
    .expect("a meta cache group")
}

/// One step of `model` over the bucket `batch`, at this rank's place `group` of its deployment
/// and with the expert group's `exchange` when the experts are split: what it recorded, each
/// launch checked ([`assert_generic`]).
fn step(
    model: &dyn TextModel,
    batch: &nn::TokenBatch,
    group: &[Split],
    exchange: Option<&ExpertExchange>,
) -> Recording {
    let (rows, tokens, phase) = (batch.rows(), batch.tokens(), batch.phase);
    let row_pages = (batch.max_kv_len / PAGE_TOKENS) as usize;

    let (stepped, recording) = record(group, || {
        let cache = cache(&model.cache(), model.dtype(), (rows, row_pages), tokens);
        let runs = model.state_layers();
        let state = (!runs.is_empty()).then(|| {
            let buffers = || {
                RecurrentState::buffers(&runs, |_| SLOTS, |shape, dtype| Ok(meta(shape, dtype)))
                    .expect("the state buffers")
            };
            text::state(
                [
                    meta(rows + 1, DType::U32),
                    meta(rows, DType::U32),
                    meta(rows, DType::U32),
                    meta(rows, DType::U32),
                ],
                &buffers(),
                &buffers(),
                None,
            )
        });

        let embedded = model
            .embed_input_ids(&meta(tokens, DType::U32))
            .expect("the embeddings");
        assert_eq!(embedded.dims(), [tokens, model.hidden() as usize]);
        assert_eq!(embedded.dtype(), model.dtype());

        let hidden = model
            .forward(batch, &[cache], state.as_ref(), exchange, &embedded)
            .unwrap_or_else(|error| panic!("the {phase:?} forward: {error}"));
        assert_eq!(hidden.dims(), [tokens, model.hidden() as usize]);
        assert_eq!(hidden.dtype(), model.dtype());

        // Execution gathers the rows a batch asks logits for: one per row.
        let logits = model
            .compute_logits(&hidden.narrow(0, 0, rows).expect("one hidden row per row"))
            .expect("the logits");
        assert_eq!(logits.dims(), [rows, model.vocabulary() as usize]);
        assert_eq!(logits.dtype(), DType::F32);
        Ok(())
    });
    stepped.unwrap_or_else(|error| panic!("the {phase:?} step: {error}"));

    assert_generic(&format!("a {phase:?} step"), &recording.launches);
    recording
}

/// Runs the decode bucket (eight rows of one token) and the prefill bucket (two rows of 256
/// tokens) of `model` twice each, asserts that both runs of a bucket state the same kernels
/// and reach the same peak, reports them under `name`, and returns the kernels of the two
/// buckets, decode first.
fn buckets(
    name: &str,
    model: &dyn TextModel,
    group: &[Split],
    exchange: &dyn Fn(usize) -> Option<ExpertExchange>,
) -> [Keys; 2] {
    let run = |phase, bucket: (usize, usize)| {
        // Each row may read its own tokens, in whole pages.
        let row_pages = (bucket.1 / bucket.0).div_ceil(PAGE_TOKENS as usize);
        let batch = batch(phase, bucket.0, bucket.1, row_pages);

        let exchange = exchange(bucket.1);
        let first = step(model, &batch, group, exchange.as_ref());
        let second = step(model, &batch, group, exchange.as_ref());
        assert_eq!(
            (keys(&first), first.peak),
            (keys(&second), second.peak),
            "two runs of one {phase:?} bucket differ"
        );

        report(
            &format!("{name} {phase:?} {}x{}", bucket.0, bucket.1 / bucket.0),
            &first,
        );
        assert!(first.peak > 0, "a forward allocates its activations");

        keys(&first)
    };

    [run(Phase::Decode, (8, 8)), run(Phase::Prefill, (2, 512))]
}

/// The operators of `keys`.
fn operators(keys: &Keys) -> BTreeSet<&'static str> {
    keys.iter().map(|(operator, _)| *operator).collect()
}

/// No expert group: the experts are this rank's own.
fn no_exchange(_: usize) -> Option<ExpertExchange> {
    None
}

/// The operators the dense experts of a routed block launch, and the ones packed experts do.
const DENSE_EXPERTS: [&str; 2] = ["gather_gemm", "route_gemm"];
const PACKED_EXPERTS: [&str; 2] = ["gather_gemm_nvfp4", "route_gemm_nvfp4"];

/// Qwen3-30B-A3B in BF16: 48 attention layers, every layer routed over dense experts.
#[test]
#[ignore = "reads the Hugging Face release Qwen/Qwen3-30B-A3B"]
fn qwen3_moe_profiles_a_decode_and_a_prefill_bucket() {
    let model = load(QWEN3_30B_A3B, &TextPlacement::WHOLE);

    assert_eq!(key_value_shape(&model.cache()), (48, 4, 128));
    assert!(model.state_layers().is_empty());
    assert!(model.drafter().is_none() && model.vision().is_none());
    let experts = model.experts().expect("routed experts");
    assert_eq!(
        (experts.experts, experts.top, experts.encoded),
        (128, 8, false)
    );

    for keys in buckets("Qwen3-30B-A3B", &*model, &[], &no_exchange) {
        assert_operators(
            "a dense Qwen3-MoE step",
            &operators(&keys),
            &[
                "qk_norm_rope",
                "paged_attention",
                "topk_softmax",
                "moe_sort",
            ],
            &PACKED_EXPERTS,
        );
        assert_operators(
            "a dense Qwen3-MoE step",
            &operators(&keys),
            &DENSE_EXPERTS,
            &[],
        );
    }
}

/// Qwen3-30B-A3B over two pipeline stages and over two Ulysses members of either projection
/// form: a stage holds 24 of the 48 layers with every key/value head and passes its rows over the
/// group, a member holds every layer with two of the four key/value heads and attends its heads
/// of every row of the batch. Each returns the whole batch's rows, also for a batch of one token,
/// whose second member holds a padding row alone.
#[test]
#[ignore = "reads the Hugging Face release Qwen/Qwen3-30B-A3B"]
fn qwen3_moe_profiles_pipeline_stages_and_sequence_members() {
    for rank in 0..2 {
        let placement = TextPlacement {
            pipeline: Split::new(2, rank).unwrap(),
            ..TextPlacement::WHOLE
        };
        let model = load(QWEN3_30B_A3B, &placement);
        let (layers, kv_heads, _) = key_value_shape(&model.cache());
        assert_eq!((layers, kv_heads), (24, 4));

        let name = format!("Qwen3-30B-A3B pipeline stage {rank} of 2");
        buckets(
            &name,
            &*model,
            &[Split::new(2, rank).unwrap()],
            &no_exchange,
        );
    }

    for projection in [Projection::Gather, Projection::Exchange] {
        for rank in 0..2 {
            let placement = TextPlacement {
                sequence: SequenceSplit {
                    split: Split::new(2, rank).unwrap(),
                    projection,
                },
                ..TextPlacement::WHOLE
            };
            let model = load(QWEN3_30B_A3B, &placement);
            let (layers, kv_heads, _) = key_value_shape(&model.cache());
            assert_eq!((layers, kv_heads), (48, 2));

            let group = [Split::new(2, rank).unwrap()];
            let name = format!("Qwen3-30B-A3B {projection:?} sequence member {rank} of 2");
            buckets(&name, &*model, &group, &no_exchange);
            step(&*model, &batch(Phase::Decode, 1, 1, 1), &group, None);
        }
    }
}

/// Qwen3-30B-A3B over two tensor-parallel ranks and over two expert-parallel ranks: each rank
/// holds half the heads, or half the experts with the group's exchange.
#[test]
#[ignore = "reads the Hugging Face release Qwen/Qwen3-30B-A3B"]
fn qwen3_moe_profiles_tensor_and_expert_parallel_ranks() {
    for rank in 0..2 {
        let placement = TextPlacement {
            tensor: Split::new(2, rank).unwrap(),
            ..TextPlacement::WHOLE
        };
        let model = load(QWEN3_30B_A3B, &placement);
        assert_eq!(key_value_shape(&model.cache()).1, 2);

        let name = format!("Qwen3-30B-A3B tensor rank {rank} of 2");
        buckets(
            &name,
            &*model,
            &[Split::new(2, rank).unwrap()],
            &no_exchange,
        );
    }

    // A deployment that splits neither the heads nor the experts evenly is refused at load.
    let checkpoint = QWEN3_30B_A3B.checkpoint();
    for (tensor, experts) in [(3, 1), (1, 3)] {
        let config = models::qwen3::Config::open(&checkpoint).expect("the configuration");
        let split = |size| Split::new(size, 0).unwrap();
        assert!(config.check_splits(split(tensor), split(experts)).is_err());
    }

    for rank in 0..2 {
        let placement = TextPlacement {
            experts: Split::new(2, rank).unwrap(),
            ..TextPlacement::WHOLE
        };
        let model = load(QWEN3_30B_A3B, &placement);
        let shape = model.experts().expect("routed experts");

        // The rows of the group's exchange, as execution lends them: each rank may send
        // every token of a bucket.
        let exchange = |tokens: usize| {
            let rank_rows = tokens as u32;
            let rows = 2 * tokens;
            let (hidden, top) = (model.hidden() as usize, shape.top as usize);

            Some(ExpertExchange {
                ranks: 2,
                attention: 0,
                rank_rows,
                step_rows: rank_rows,
                multiprocessors: 148,
                states: meta((rows, hidden), model.dtype()),
                ids: meta((rows, top), DType::U32),
                weights: meta((rows, top), DType::F32),
                combined: meta((rows, hidden), model.dtype()),
                live: meta(1, DType::U32),
                packed: meta((tokens, hidden / 2), DType::U8),
                packed_scales: meta((tokens, hidden / 16), DType::F8E4M3),
                fused: false,
                source_only: false,
            })
        };

        let group = [Split::new(2, rank).unwrap()];
        let name = format!("Qwen3-30B-A3B expert rank {rank} of 2");
        for keys in buckets(&name, &*model, &group, &exchange) {
            assert_operators(
                "an expert-parallel step",
                &operators(&keys),
                &["expert_exchange"],
                &[],
            );
        }

        // A rank with no tokens of its own joins the group's step.
        let (joined, recording) = record(&group, || {
            let lent = exchange(8).expect("an exchange");
            model.join_expert_step(&lent)
        });
        joined.expect("the group's step joined");
        assert_generic("a joined step", &recording.launches);
        assert_operators(
            "a joined step",
            &operators(&keys(&recording)),
            &["expert_exchange"],
            &["paged_attention"],
        );
    }
}

/// The Qwen3 family states no precision mode and no module selection: the BF16 checkpoint and
/// its NVFP4 export each open in their own representation under an empty policy, and a stated
/// policy is refused by name for either, even one that names their own representation.
#[test]
#[ignore = "reads the Hugging Face releases Qwen/Qwen3-30B-A3B and nvidia/Qwen3-30B-A3B-NVFP4"]
fn qwen3_moe_takes_no_precision_policy() {
    use nn::family::Model;
    use proto::config::{PrecisionMode, Quantization, WeightFormat};

    let Model::Text(family) = models::qwen3::family::FAMILY.model else {
        panic!("Qwen3-MoE is a text family");
    };
    let stated = [
        Quantization {
            mode: Some(PrecisionMode::Quality),
            modules: Default::default(),
        },
        Quantization {
            mode: None,
            modules: [("mlp".to_owned(), WeightFormat::Bf16)].into(),
        },
    ];

    for release in [QWEN3_30B_A3B, QWEN3_30B_A3B_NVFP4] {
        let checkpoint = release.checkpoint();
        assert!(
            family.open(&checkpoint, &Quantization::default()).is_ok(),
            "{release}"
        );
        for policy in &stated {
            let error = family
                .open(&checkpoint, policy)
                .err()
                .unwrap_or_else(|| panic!("{release} took {policy}"));
            assert!(
                error
                    .to_string()
                    .contains("the qwen3 family states no precision mode"),
                "{release}: {error}"
            );
        }
    }
}

/// Qwen3-30B-A3B in calibrated NVFP4: packed projections and packed experts, activations
/// encoded before each packed product.
#[test]
#[ignore = "reads the Hugging Face release nvidia/Qwen3-30B-A3B-NVFP4"]
fn qwen3_moe_nvfp4_profiles_a_decode_and_a_prefill_bucket() {
    let model = load(QWEN3_30B_A3B_NVFP4, &TextPlacement::WHOLE);

    assert_eq!(key_value_shape(&model.cache()), (48, 4, 128));
    assert!(model.experts().expect("routed experts").encoded);

    for keys in buckets("Qwen3-30B-A3B-NVFP4", &*model, &[], &no_exchange) {
        assert_operators(
            "a packed Qwen3-MoE step",
            &operators(&keys),
            &["qk_norm_rope", "paged_attention", "nvfp4_quantize"],
            &DENSE_EXPERTS,
        );
        assert_operators(
            "a packed Qwen3-MoE step",
            &operators(&keys),
            &PACKED_EXPERTS,
            &[],
        );
    }
}

/// The operators every step of a hybrid model launches.
const HYBRID: [&str; 7] = [
    "causal_conv1d",
    "gated_delta_rule",
    "gated_rms_norm",
    "sigmoid_gated_add",
    "qk_norm_rope",
    "act_and_mul",
    "paged_attention",
];

/// Asserts what a Qwen3.6-35B-A3B model states of its cache and state: ten attention layers
/// keep keys and values; thirty linear layers keep a convolution window of three inputs of
/// 8192 channels and a 128 × 128 FP32 state per value head.
fn assert_hybrid_shapes(model: &dyn TextModel) {
    assert_eq!(key_value_shape(&model.cache()), (10, 2, 256));

    let [state] = <[_; 1]>::try_from(model.state_layers()).expect("one run of recurrent layers");
    assert_eq!(state.layers, 30);
    assert_eq!(state.fields[0].dims, [3, 8192]);
    assert_eq!(state.fields[0].dtype, model.dtype());
    assert_eq!(state.fields[1].dims, [32, 128, 128]);
    assert_eq!(state.fields[1].dtype, DType::F32);
}

/// The keys of `operator` among `keys`.
fn keys_of<'k>(keys: &'k Keys, operator: &str) -> Vec<&'k str> {
    keys.iter()
        .filter(|(named, _)| *named == operator)
        .map(|(_, key)| key.as_str())
        .collect()
}

/// Qwen3.6-35B-A3B in BF16, with its drafter and its vision tower: the model's two buckets,
/// the drafter's forward over a draft cache of its own, and one image through the tower.
#[test]
#[ignore = "reads the Hugging Face release Qwen/Qwen3.6-35B-A3B"]
fn qwen3_6_profiles_its_buckets_its_drafter_and_its_tower() {
    let placement = TextPlacement {
        drafter: true,
        vision: true,
        ..TextPlacement::WHOLE
    };
    let model = load(QWEN3_6_35B_A3B, &placement);
    assert_hybrid_shapes(&*model);
    assert!(!model.experts().expect("routed experts").encoded);

    let [decode, prefill] = buckets("Qwen3.6-35B-A3B", &*model, &[], &no_exchange);
    for keys in [&decode, &prefill] {
        assert_operators("a hybrid step", &operators(keys), &HYBRID, &PACKED_EXPERTS);
        assert_operators("a hybrid step", &operators(keys), &DENSE_EXPERTS, &[]);
    }

    // A prefill advances each row's state chunk by chunk and a decode token by token.
    let rule = |keys| keys_of(keys, "gated_delta_rule");
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

    // A prefill that holds an image's rows states three coordinates per token and turns by
    // the sectioned spectrum: the same kernels, the factors of three axes.
    let mut with_image = batch(Phase::Prefill, 1, 512, 8);
    with_image.rope_positions = Some(meta((3, 512), DType::I32));
    let image_keys = keys(&step(&*model, &with_image, &[], None));
    assert_eq!(operators(&image_keys), operators(&prefill));
    assert_ne!(
        keys_of(&image_keys, "rotary_factors"),
        keys_of(&prefill, "rotary_factors")
    );

    // The drafter: one layer over a draft cache of two heads of 256, through the model's gated
    // attention and routed block, with no recurrent layer.
    let drafter = model.drafter().expect("the drafter");
    let draft = drafter.cache();
    assert_eq!(key_value_shape(&draft), (1, 2, 256));

    for (phase, rows, tokens) in [(Phase::Decode, 8, 8), (Phase::Prefill, 2, 512)] {
        let row_pages = tokens / rows / PAGE_TOKENS as usize + 1;
        let (drafted, recording) = record(&[], || {
            let cache = cache(&draft, model.dtype(), (rows, row_pages), tokens);
            let rows_of = || meta((tokens, model.hidden() as usize), model.dtype());

            drafter.forward(
                &batch(phase, rows, tokens, row_pages),
                &cache,
                None,
                &rows_of(),
                &rows_of(),
            )
        });
        let drafted =
            drafted.unwrap_or_else(|error| panic!("the drafter's {phase:?} forward: {error}"));
        assert_eq!(drafted.dims(), [tokens, model.hidden() as usize]);

        report(&format!("Qwen3.6-35B-A3B drafter {phase:?}"), &recording);
        assert_operators(
            "the drafter",
            &operators(&keys(&recording)),
            &["rms_norm", "qk_norm_rope", "paged_attention", "act_and_mul"],
            &["gated_delta_rule", "causal_conv1d"],
        );
    }

    // The tower: an image of 448 x 672 pixels is sized to a 28 x 42 patch grid, which merges
    // to 294 rows of the model's width at three M-RoPE axes.
    let tower = model.vision().expect("the tower");
    let grid = tower.image_grid(448, 672).expect("an image's grid");
    assert_eq!(grid, [1, 28, 42]);
    let rows = tower.image_rows(grid).expect("whole merge blocks") as usize;
    assert_eq!((rows, tower.position_axes()), (294, 3));

    let format = tower.patch_format();
    assert_eq!((format.width(), format.dtype), (1536, model.dtype()));
    let positions = tower
        .image_positions(grid, 7)
        .expect("the block's positions");
    assert_eq!(positions.dims(), [3, rows]);

    let (encoded, recording) = record(&[], || {
        let pixels = meta((28 * 42, 1536), format.dtype);
        let features = meta((rows, model.hidden() as usize), model.dtype());

        tower.encode_image(grid, &pixels, &features)
    });
    encoded.expect("the image's features");
    report("Qwen3.6-35B-A3B tower 28x42", &recording);
    assert_operators(
        "the tower",
        &operators(&keys(&recording)),
        &["layer_norm", "dense_attention", "rotary_factors"],
        &[],
    );
}

/// The calibrated mixed-precision export of Qwen3.6-35B-A3B stores its experts as weight-only
/// NVFP4 (`W4A16_NVFP4`), which no operator of this build multiplies: the family refuses the
/// checkpoint when it opens it, naming the representation, and the checkpoint is refused until
/// an operator serves that representation (decision D12).
#[test]
#[ignore = "reads the Hugging Face release nvidia/Qwen3.6-35B-A3B-NVFP4"]
fn qwen3_6_nvfp4_is_refused_by_its_experts_representation() {
    let release = QWEN3_6_35B_A3B_NVFP4;
    let refused = text::open(&FAMILIES, &release.checkpoint())
        .err()
        .unwrap_or_else(|| panic!("{release} opened"));

    assert!(refused.to_string().contains("W4A16_NVFP4"), "{refused}");
}

/// The Qwen3-VL encoder over the MiniMax-H3 text encoder's headers: fifty of its layers before
/// the final norm, its tower and DeepStack mergers, encoding a prompt that presents an image.
#[test]
#[ignore = "reads the Hugging Face release MiniMaxAI/MiniMax-H3"]
fn the_vl_encoder_profiles_a_prompt_with_an_image() {
    // The pipeline keeps the encoder's weights and its processor files in two directories, as
    // the family composes them.
    let (pipeline, _) = models::minimax_h3::family::compose(&MINIMAX_H3.sources())
        .unwrap_or_else(|error| panic!("{MINIMAX_H3}: {error}"));
    let checkpoint = pipeline.stage("text_encoder").expect("the text encoder");
    let config = Config::open(checkpoint).expect("the configuration");
    let sidecar = |file: &str| {
        pipeline
            .sidecar(&format!("processor/{file}"))
            .expect("a readable processor file")
            .unwrap_or_else(|| panic!("the pipeline holds no processor/{file}"))
    };
    let [image_file, video_file] = ProcessorConfig::FILES;
    let (image, video) = (sidecar(image_file), sidecar(video_file));
    let images = (
        ProcessorConfig::from_values(&image, &video).expect("the processor"),
        Fields::of(image_file, &image)
            .and_then(|fields| fields.pixel_norm())
            .expect("the pixel normalization"),
    );

    let store = MetaWeights::new();
    let loader = WeightLoader::new(checkpoint, &store, None);
    let model = Qwen3VlModel::load(&config, loader.root(), images, Split::WHOLE, 50, false)
        .expect("the meta load");
    let vision = config.vision.as_ref().expect("a tower");
    let hidden = config.text.hidden_size as usize;

    // Three text tokens, one 24 x 36-patch image (216 merged rows), two text tokens.
    let grid = [1, 24, 36];
    let (patches, merged) = (24 * 36, 24 * 36 / 4);
    let ids = [
        vec![10, 11, 12],
        vec![vision.image_token_id; merged],
        vec![13, 14],
    ]
    .concat();
    let tokens = ids.len();

    let encode = |prompt: &[u32]| -> (nn::Result<()>, Recording) {
        record(&[], || {
            let batch = batch(
                Phase::Prefill,
                1,
                tokens,
                tokens.div_ceil(PAGE_TOKENS as usize),
            );
            let layout = model.layout(&batch, &[grid]);
            assert_eq!((layout.rows, layout.tokens), (tokens as u32, tokens as u32));

            let pixels = meta((patches, vision.tubelet() as usize), DType::BF16);
            model.encode(
                prompt,
                &meta(tokens, DType::U32),
                &batch,
                &[pixels],
                &layout,
                &meta(Shape::from((tokens, hidden)), DType::BF16),
            )
        })
    };

    let (encoded, recording) = encode(&ids);
    encoded.expect("the encode");
    let keys = keys(&recording);
    assert_eq!(
        self::keys(&encode(&ids).1),
        keys,
        "two encodes of one layout differ"
    );
    report(
        &format!("MiniMax-H3 text encoder {tokens} tokens, 24x36 image"),
        &recording,
    );

    assert_operators(
        "an encode",
        &operators(&keys),
        &[
            "embedding",
            "index",
            "layer_norm",
            "dense_attention",
            "qk_norm_rope",
            "act_and_mul",
        ],
        &["paged_attention", "kv_write"],
    );

    // The tower's repeated 2-D spectrum and the language model's sectioned M-RoPE are two
    // kernels.
    assert_eq!(keys_of(&keys, "rotary_factors").len(), 2);

    // The prompt's M-RoPE coordinates and placeholder rows, and the frame's taps, weights,
    // rotary coordinates and offsets, are the host values the encode moves to the device.
    let host = 4 * (3 * tokens + merged) + patches * (4 * 4 + 4 * 4 + 2 * 4) + 8;
    assert!(
        recording.host_bytes >= host as u64,
        "{} bytes of host values, at least {host} expected",
        recording.host_bytes
    );

    // A prompt whose placeholders do not hold the image's merged rows is refused.
    let mut short = ids.clone();
    short[4] = 15;
    assert!(encode(&short).0.is_err());
}
