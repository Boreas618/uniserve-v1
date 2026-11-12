//! Profile runs on the meta device over the released checkpoints' headers.
//!
//! A meta load reads a checkpoint's shard headers and no tensor data, and a forward over meta
//! tensors under a meta context launches nothing: it checks every operand, states the kernel
//! each launch would run and counts what the forward allocates. Each test loads a released
//! checkpoint through its family, runs one step of a decode bucket and of a prefill bucket
//! (the embedding, the decoder stack and the logits), and asserts the shapes of the outputs,
//! that every launch names an operator, that no kernel key names a model, and that a second
//! run of the same bucket states the same keys and reaches the same allocator peak. It prints
//! the keys and the peaks.
//!
//! The tests need the checkpoints' files under [`MODELS`] and no device; they are ignored by
//! default and fail when asked for without the files.

#![cfg(test)]

mod support;

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use nn::config::Fields;
use nn::family::{TextModel, TextPlacement};
use nn::{DType, Encoder, ExpertExchange, PhaseKind, Shape, Split, Tensor, WeightLoader};
use processing::qwen_vl::ProcessorConfig;
use qwen3::{Config, Qwen3VlModel};

/// Where the released checkpoints are.
const MODELS: &str = "/workspace/models";

/// Tokens of one cache page.
const PAGE: u32 = 64;

/// Request slots of the recurrent state's buffers.
const SLOTS: usize = 8;

/// The kernels a forward stated: `(operator, key)` pairs.
type Keys = BTreeSet<(&'static str, String)>;

/// A meta tensor of `shape` elements of `dtype`: an input of a profile run, which takes no
/// block of the context's allocator. Each call gives a range of addresses no other has.
fn meta(shape: Shape, dtype: DType) -> Tensor {
    static NEXT: AtomicU64 = AtomicU64::new(1 << 40);

    let bytes = (shape.elem_count() * dtype.size_in_bytes()) as u64;
    let address = NEXT.fetch_add(
        bytes.next_multiple_of(1 << 16).max(1 << 16),
        Ordering::Relaxed,
    );

    backend::retained(address, shape, dtype, &backend::meta()).expect("a meta tensor")
}

/// The model of the released checkpoint `name`, meta-loaded for `placement`.
fn load(name: &str, placement: &TextPlacement) -> Arc<dyn TextModel> {
    let root = Path::new(MODELS).join(name);
    assert!(
        root.is_dir(),
        "{} is not there: this test reads the released checkpoint's headers",
        root.display()
    );

    let checkpoint = support::checkpoint(&root);
    let mut loader = WeightLoader::meta(&checkpoint, None, None);
    support::load(&checkpoint, &mut loader, placement)
}

/// A meta context, stating this rank's place `(rank, ranks)` in a group when it has one, that
/// records the launches of what runs under it.
fn context(group: Option<(u64, u64)>) -> backend::Context {
    let context = backend::Context::meta(None);
    if let Some((rank, ranks)) = group {
        context
            .set_group(None, rank, ranks)
            .expect("a group of a meta context");
    }
    context.record_launches();
    context
}

/// The kernels `context` recorded, each checked: every launch names its operator and its key,
/// and no key names a model.
fn keys(context: &backend::Context) -> Keys {
    context
        .take_launches()
        .into_iter()
        .map(|(operator, key, _)| {
            assert!(!operator.is_empty(), "a launch of no operator: {key}");
            assert!(!key.is_empty(), "{operator} states no key");
            assert!(
                !key.to_lowercase().contains("qwen") && !operator.to_lowercase().contains("qwen"),
                "{operator} {key} names a model"
            );

            (operator, key)
        })
        .collect()
}

/// The index columns of a meta batch of `rows` rows of `tokens` tokens in all, whose rows may
/// each read `row_pages` pages.
fn batch(phase: PhaseKind, rows: usize, tokens: usize, row_pages: usize) -> nn::TokenBatch {
    support::batch(
        phase,
        [
            meta(Shape::from(tokens), DType::I32),
            meta(Shape::from(tokens), DType::U32),
            meta(Shape::from(rows), DType::U32),
            meta(Shape::from(rows + 1), DType::U32),
            meta(Shape::from(rows), DType::U32),
        ],
        PAGE,
        PAGE * row_pages as u32,
    )
}

/// One step of `model` over the bucket `batch`, at this rank's place `group` of its deployment
/// and with the expert group's `exchange` when the experts are split: the kernels it states
/// and the bytes its allocator reaches.
fn step(
    model: &dyn TextModel,
    batch: &nn::TokenBatch,
    group: Option<(u64, u64)>,
    exchange: Option<&ExpertExchange>,
) -> (Keys, u64) {
    let (rows, tokens, phase) = (batch.rows(), batch.tokens(), batch.phase);
    let row_pages = (batch.max_kv_len / PAGE) as usize;

    let context = context(group);
    {
        let _guard = context.enter();

        let cache = support::cache(
            model.cache(),
            model.dtype(),
            (rows * row_pages, PAGE),
            meta(Shape::from((rows, row_pages)), DType::U32),
            &meta,
        );
        let state = model.state().map(|shape| {
            support::state(
                [
                    meta(Shape::from(rows + 1), DType::U32),
                    meta(Shape::from(rows), DType::U32),
                    meta(Shape::from(rows), DType::U32),
                    meta(Shape::from(rows), DType::U32),
                ],
                &support::state_buffers(&shape, SLOTS, &meta),
                &support::state_buffers(&shape, SLOTS, &meta),
            )
        });

        let embedded = model
            .embed_input_ids(&meta(Shape::from(tokens), DType::U32))
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
    }

    (keys(&context), context.peak())
}

/// Runs the decode bucket (eight rows of one token) and the prefill bucket (two rows of 256
/// tokens) of `model` twice each, asserts that both runs of a bucket state the same kernels
/// and reach the same peak, prints them under `name`, and returns the kernels of the two
/// buckets, decode first.
fn buckets(
    name: &str,
    model: &dyn TextModel,
    group: Option<(u64, u64)>,
    exchange: &dyn Fn(usize) -> Option<ExpertExchange>,
) -> [Keys; 2] {
    let run = |phase, bucket: (usize, usize)| {
        // Each row may read its own tokens, in whole pages.
        let row_pages = (bucket.1 / bucket.0).div_ceil(PAGE as usize);
        let batch = batch(phase, bucket.0, bucket.1, row_pages);

        let exchange = exchange(bucket.1);
        let first = step(model, &batch, group, exchange.as_ref());
        let second = step(model, &batch, group, exchange.as_ref());
        assert_eq!(first, second, "two runs of one {phase:?} bucket differ");

        let (keys, peak) = first;
        let operators: BTreeSet<&str> = keys.iter().map(|(operator, _)| *operator).collect();
        println!(
            "{name} {phase:?} {}x{}: {} kernel keys of {} operators, allocator peak {peak} bytes",
            bucket.0,
            bucket.1 / bucket.0,
            keys.len(),
            operators.len()
        );
        println!("{name} {phase:?} operators: {operators:?}");
        for (operator, key) in &keys {
            println!("{name} {phase:?} key: {operator} {key}");
        }
        assert!(peak > 0, "a forward allocates its activations");

        keys
    };

    [
        run(PhaseKind::Decode, (8, 8)),
        run(PhaseKind::Prefill, (2, 512)),
    ]
}

/// The operators of `keys`.
fn operators(keys: &Keys) -> BTreeSet<&'static str> {
    keys.iter().map(|(operator, _)| *operator).collect()
}

/// Asserts that `keys` states every operator of `expected` and none of `absent`.
fn assert_operators(what: &str, keys: &Keys, expected: &[&str], absent: &[&str]) {
    let named = operators(keys);

    for operator in expected {
        assert!(
            named.contains(operator),
            "{what} launches no {operator}: {named:?}"
        );
    }
    for operator in absent {
        assert!(
            !named.contains(operator),
            "{what} launches {operator}: {named:?}"
        );
    }
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
#[ignore = "requires /workspace/models/Qwen3-30B-A3B"]
fn qwen3_moe_profiles_a_decode_and_a_prefill_bucket() {
    let model = load("Qwen3-30B-A3B", &support::whole(false));

    let cache = model.cache();
    assert_eq!((cache.layers, cache.kv_heads, cache.head_dim), (48, 4, 128));
    assert!(model.state().is_none());
    assert!(model.drafter().is_none() && model.vision().is_none());
    let experts = model.experts().expect("routed experts");
    assert_eq!(
        (experts.experts, experts.top, experts.encoded),
        (128, 8, false)
    );

    for keys in buckets("Qwen3-30B-A3B", &*model, None, &no_exchange) {
        assert_operators(
            "a dense Qwen3-MoE step",
            &keys,
            &[
                "qk_norm_rope",
                "paged_attention",
                "topk_softmax",
                "moe_sort",
            ],
            &PACKED_EXPERTS,
        );
        assert_operators("a dense Qwen3-MoE step", &keys, &DENSE_EXPERTS, &[]);
    }
}

/// Qwen3-30B-A3B over two tensor-parallel ranks and over two expert-parallel ranks: each rank
/// holds half the heads, or half the experts with the group's exchange.
#[test]
#[ignore = "requires /workspace/models/Qwen3-30B-A3B"]
fn qwen3_moe_profiles_tensor_and_expert_parallel_ranks() {
    for rank in 0..2 {
        let placement = TextPlacement {
            tensor: Split { size: 2, rank },
            ..support::whole(false)
        };
        let model = load("Qwen3-30B-A3B", &placement);
        assert_eq!(model.cache().kv_heads, 2);

        let name = format!("Qwen3-30B-A3B tensor rank {rank} of 2");
        buckets(&name, &*model, Some((u64::from(rank), 2)), &no_exchange);
    }

    // A deployment that splits neither the heads nor the experts evenly is refused at load.
    let checkpoint = support::checkpoint(&Path::new(MODELS).join("Qwen3-30B-A3B"));
    for (tensor, experts) in [(3, 1), (1, 3)] {
        let config = qwen3::Config::open(&checkpoint).expect("the configuration");
        let split = |size| Split { size, rank: 0 };
        assert!(config.check_splits(split(tensor), split(experts)).is_err());
    }

    for rank in 0..2 {
        let placement = TextPlacement {
            experts: Split { size: 2, rank },
            ..support::whole(false)
        };
        let model = load("Qwen3-30B-A3B", &placement);
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
                states: meta(Shape::from((rows, hidden)), model.dtype()),
                ids: meta(Shape::from((rows, top)), DType::U32),
                weights: meta(Shape::from((rows, top)), DType::F32),
                combined: meta(Shape::from((rows, hidden)), model.dtype()),
                live: meta(Shape::from(1), DType::U32),
                packed: meta(Shape::from((tokens, hidden / 2)), DType::U8),
                packed_scales: meta(Shape::from((tokens, hidden / 16)), DType::F8E4M3),
                fused: false,
                source_only: false,
            })
        };

        let name = format!("Qwen3-30B-A3B expert rank {rank} of 2");
        for keys in buckets(&name, &*model, Some((u64::from(rank), 2)), &exchange) {
            assert_operators("an expert-parallel step", &keys, &["expert_exchange"], &[]);
        }

        // A rank with no tokens of its own joins the group's step.
        let context = context(Some((u64::from(rank), 2)));
        {
            let _guard = context.enter();
            let lent = exchange(8).expect("an exchange");
            model
                .join_expert_step(&lent)
                .expect("the group's step joined");
        }
        assert_operators(
            "a joined step",
            &keys(&context),
            &["expert_exchange"],
            &["paged_attention"],
        );
    }
}

/// Qwen3-30B-A3B in calibrated NVFP4: packed projections and packed experts, activations
/// encoded before each packed product.
#[test]
#[ignore = "requires /workspace/models/Qwen3-30B-A3B-NVFP4"]
fn qwen3_moe_nvfp4_profiles_a_decode_and_a_prefill_bucket() {
    let model = load("Qwen3-30B-A3B-NVFP4", &support::whole(false));

    let cache = model.cache();
    assert_eq!((cache.layers, cache.kv_heads, cache.head_dim), (48, 4, 128));
    assert!(model.experts().expect("routed experts").encoded);

    for keys in buckets("Qwen3-30B-A3B-NVFP4", &*model, None, &no_exchange) {
        assert_operators(
            "a packed Qwen3-MoE step",
            &keys,
            &["qk_norm_rope", "paged_attention", "nvfp4_quantize"],
            &DENSE_EXPERTS,
        );
        assert_operators("a packed Qwen3-MoE step", &keys, &PACKED_EXPERTS, &[]);
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
    let cache = model.cache();
    assert_eq!((cache.layers, cache.kv_heads, cache.head_dim), (10, 2, 256));

    let state = model.state().expect("a recurrent state");
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
#[ignore = "requires /workspace/models/Qwen3.6-35B-A3B"]
fn qwen3_6_profiles_its_buckets_its_drafter_and_its_tower() {
    let model = load("Qwen3.6-35B-A3B", &support::whole(true));
    assert_hybrid_shapes(&*model);
    assert!(!model.experts().expect("routed experts").encoded);

    let [decode, prefill] = buckets("Qwen3.6-35B-A3B", &*model, None, &no_exchange);
    for keys in [&decode, &prefill] {
        assert_operators("a hybrid step", keys, &HYBRID, &PACKED_EXPERTS);
        assert_operators("a hybrid step", keys, &DENSE_EXPERTS, &[]);
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
    let mut with_image = batch(PhaseKind::Prefill, 1, 512, 8);
    with_image.rope_positions = Some(meta(Shape::from((3, 512)), DType::I32));
    let (image_keys, _) = step(&*model, &with_image, None, None);
    assert_eq!(operators(&image_keys), operators(&prefill));
    assert_ne!(
        keys_of(&image_keys, "rotary_factors"),
        keys_of(&prefill, "rotary_factors")
    );

    // The drafter: one layer over a draft cache of two heads of 256, through the model's gated
    // attention and routed block, with no recurrent layer.
    let drafter = model.drafter().expect("the drafter");
    let draft = drafter.cache();
    assert_eq!((draft.layers, draft.kv_heads, draft.head_dim), (1, 2, 256));

    for (phase, rows, tokens) in [(PhaseKind::Decode, 8, 8), (PhaseKind::Prefill, 2, 512)] {
        let context = context(None);
        {
            let _guard = context.enter();
            let row_pages = tokens / rows / PAGE as usize + 1;
            let cache = support::cache(
                draft,
                model.dtype(),
                (rows * row_pages, PAGE),
                meta(Shape::from((rows, row_pages)), DType::U32),
                &meta,
            );
            let rows_of = || {
                meta(
                    Shape::from((tokens, model.hidden() as usize)),
                    model.dtype(),
                )
            };

            let drafted = drafter
                .forward(
                    &batch(phase, rows, tokens, row_pages),
                    &cache,
                    &rows_of(),
                    &rows_of(),
                )
                .unwrap_or_else(|error| panic!("the drafter's {phase:?} forward: {error}"));
            assert_eq!(drafted.dims(), [tokens, model.hidden() as usize]);
        }

        let keys = keys(&context);
        println!(
            "Qwen3.6-35B-A3B drafter {phase:?}: {} kernel keys, allocator peak {} bytes",
            keys.len(),
            context.peak()
        );
        assert_operators(
            "the drafter",
            &keys,
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

    let context = context(None);
    {
        let _guard = context.enter();
        let pixels = meta(Shape::from((28 * 42, 1536)), format.dtype);
        let features = meta(Shape::from((rows, model.hidden() as usize)), model.dtype());

        tower
            .encode_image(grid, &pixels, &features)
            .expect("the image's features");
    }
    let keys = keys(&context);
    println!(
        "Qwen3.6-35B-A3B tower 28x42: {} kernel keys of operators {:?}, allocator peak {} bytes",
        keys.len(),
        operators(&keys),
        context.peak()
    );
    assert_operators(
        "the tower",
        &keys,
        &["layer_norm", "dense_attention", "rotary_factors"],
        &[],
    );
}

/// The calibrated mixed-precision export of Qwen3.6-35B-A3B loads its attention and
/// delta-network projections as per-tensor FP8 products and its experts, shared experts and
/// head as the dense matrices their weight-only NVFP4 fields state, so no step encodes an
/// activation.
#[test]
#[ignore = "requires /workspace/models/Qwen3.6-35B-A3B-NVFP4"]
fn qwen3_6_nvfp4_profiles_a_decode_and_a_prefill_bucket() {
    let model = load("Qwen3.6-35B-A3B-NVFP4", &support::whole(false));
    assert_hybrid_shapes(&*model);

    // The experts are dense once decoded: no step encodes activations for them.
    assert!(!model.experts().expect("routed experts").encoded);

    for keys in buckets("Qwen3.6-35B-A3B-NVFP4", &*model, None, &no_exchange) {
        assert_operators("a mixed-precision hybrid step", &keys, &HYBRID, &[]);
        assert_operators(
            "a mixed-precision hybrid step",
            &keys,
            &["fp8_gemm"],
            &["nvfp4_quantize", "nvfp4_gemm"],
        );
        assert_operators(
            "a mixed-precision hybrid step",
            &keys,
            &DENSE_EXPERTS,
            &PACKED_EXPERTS,
        );
    }
}

/// The Qwen3-VL encoder over the MiniMax-H3 text encoder's headers: fifty of its layers before
/// the final norm, its tower and DeepStack mergers, encoding a prompt that presents an image.
#[test]
#[ignore = "requires /workspace/models/MiniMax-H3"]
fn the_vl_encoder_profiles_a_prompt_with_an_image() {
    let root = Path::new(MODELS).join("MiniMax-H3");
    assert!(
        root.is_dir(),
        "{} is not there: this test reads the released checkpoint's headers",
        root.display()
    );

    // The pipeline keeps the encoder's weights and its processor files in two directories.
    let pipeline = support::checkpoint(&root);
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
        qwen3::config::pixel_norm(&Fields::of(image_file, &image).expect("an object"))
            .expect("the pixel normalization"),
    );

    let loader = WeightLoader::meta(checkpoint, None, None);
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

    let encode = |prompt: &[u32]| -> (nn::Result<()>, Keys, u64, u64) {
        let context = context(None);
        let encoded = {
            let _guard = context.enter();
            let batch = batch(
                PhaseKind::Prefill,
                1,
                tokens,
                tokens.div_ceil(PAGE as usize),
            );
            let layout = model.layout(&batch, &[grid]);
            assert_eq!((layout.rows, layout.tokens), (tokens as u32, tokens as u32));

            let pixels = meta(
                Shape::from((patches, vision.tubelet() as usize)),
                DType::BF16,
            );
            model.encode(
                prompt,
                &meta(Shape::from(tokens), DType::U32),
                &batch,
                &[pixels],
                &layout,
                &meta(Shape::from((tokens, hidden)), DType::BF16),
            )
        };

        (
            encoded,
            keys(&context),
            context.peak(),
            context.input_peak(),
        )
    };

    let (encoded, keys, peak, copied) = encode(&ids);
    encoded.expect("the encode");
    assert_eq!(encode(&ids).1, keys, "two encodes of one layout differ");
    println!(
        "MiniMax-H3 text encoder {tokens} tokens, 24x36 image: {} kernel keys of operators {:?}, \
         allocator peak {peak} bytes, {copied} bytes of host values",
        keys.len(),
        operators(&keys)
    );
    for (operator, key) in &keys {
        println!("MiniMax-H3 text encoder key: {operator} {key}");
    }

    assert_operators(
        "an encode",
        &keys,
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
        copied >= host as u64,
        "{copied} bytes of host values, at least {host} expected"
    );

    // A prompt whose placeholders do not hold the image's merged rows is refused.
    let mut short = ids.clone();
    short[4] = 15;
    assert!(encode(&short).0.is_err());
}
