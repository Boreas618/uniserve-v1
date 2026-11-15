//! Profile runs on the meta device over the released checkpoints' headers.
//!
//! A meta load reads a checkpoint's shard headers and no tensor data, and a forward over meta
//! tensors under a meta context launches nothing: it checks every operand, states the kernel
//! each launch would run and counts what the forward allocates. Each test loads a released
//! checkpoint through its family, runs verify buckets (one and eight requests of four rows:
//! the committed token and three drafts) and a prefill bucket (the embedding, the decoder
//! stack and the logits), at tensor parallelism one and as a rank of four, and asserts the
//! shapes of the outputs, that every launch names an operator, that no kernel key names a
//! model, and that a second run of the same bucket states the same keys and reaches the same
//! allocator peak. It prints the keys, the peaks, the bytes of host values a forward moves to
//! the device, and the launches of one layer of each kind.
//!
//! The tests need the checkpoints' files under [`MODELS`] and no device; they are ignored by
//! default and fail when asked for without the files.

use super::support;

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use nn::family::{TextModel, TextPlacement};
use nn::{DType, Phase, Shape, Split, Tensor, WeightLoader};

/// Where the released checkpoints are.
const MODELS: &str = "/workspace/models";

/// Tokens of one cache page: whole chunks of the gated delta rule and whole groups of the
/// indexer.
const PAGE: u32 = 64;

/// Request slots of the recurrent state's buffers.
const SLOTS: usize = 64;

/// Tokens a step of the profiled deployment drafts, and the rows a request has in a step.
const DRAFTS: u32 = 3;
const STEP_ROWS: usize = DRAFTS as usize + 1;

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

    ops::retained(address, shape, dtype, &ops::meta()).expect("a meta tensor")
}

/// The released checkpoint `name`, its headers read.
fn checkpoint(name: &str) -> nn::Checkpoint {
    let root = Path::new(MODELS).join(name);
    assert!(
        root.is_dir(),
        "{} is not there: this test reads the released checkpoint's headers",
        root.display()
    );

    support::checkpoint(&root)
}

/// The model of the released checkpoint `name`, meta-loaded for `placement`, with the
/// checkpoint tensors the load named none of its weights by.
///
/// A block scale beside a weight decoded at load (`weight_scale_inv`) is read with the weight
/// and is the weight's.
fn load(name: &str, placement: &TextPlacement) -> (Arc<dyn TextModel>, Vec<String>) {
    let checkpoint = checkpoint(name);
    let mut loader = WeightLoader::meta(&checkpoint, None);
    let model = support::load(&checkpoint, &mut loader, placement)
        .unwrap_or_else(|error| panic!("{name}: {error}"));

    let named = loader.named().expect("a meta loader");
    let unnamed = checkpoint
        .names()
        .filter(|tensor| {
            let weight = tensor.strip_suffix("_scale_inv").unwrap_or(tensor);
            !named.contains(weight)
        })
        .map(str::to_owned)
        .collect();
    (model, unnamed)
}

/// A meta context, stating this rank's place `(rank, ranks)` in a group when it has one, that
/// records the launches of what runs under it.
fn context(group: Option<(u64, u64)>) -> ops::Context {
    let context = ops::Context::meta(None);
    if let Some((rank, ranks)) = group {
        context
            .set_group(None, rank, ranks)
            .expect("a group of a meta context");
    }
    context.record_launches();
    context
}

/// The launches `context` recorded, in order, each checked: every launch names its operator
/// and its key, and no key names a model.
fn launches(context: &ops::Context) -> Vec<(&'static str, String)> {
    context
        .take_launches()
        .into_iter()
        .map(|(operator, key, ..)| {
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

/// What one step of a bucket stated: its kernel keys, the launches of one gated delta network
/// layer and of one attention layer by operator, the allocator's peak and the bytes of host
/// values moved to the device.
#[derive(Debug, PartialEq, Eq)]
struct Profile {
    keys: Keys,
    launches: usize,
    linear_layer: BTreeMap<&'static str, usize>,
    attention_layer: BTreeMap<&'static str, usize>,
    peak: u64,
    host_bytes: u64,
}

/// One step of `model` over a bucket of `rows` attention rows of `tokens` tokens in all, whose
/// `requests` state rows hold them, at this rank's place `group` of its deployment.
fn step(
    model: &dyn TextModel,
    phase: Phase,
    (rows, tokens, requests): (usize, usize, usize),
    group: Option<(u64, u64)>,
) -> Profile {
    let row_pages = (tokens / rows).div_ceil(PAGE as usize).max(2);
    let column = |length: usize, dtype: DType| meta(Shape::from(length), dtype);

    let context = context(group);
    {
        let _guard = context.enter();

        let batch = support::batch(
            phase,
            [
                column(tokens, DType::I32),
                column(tokens, DType::U32),
                column(rows, DType::U32),
                column(rows + 1, DType::U32),
                column(rows, DType::U32),
            ],
            Some((
                column(tokens, DType::U32),
                meta(Shape::from((rows, 2)), DType::U32),
            )),
            PAGE,
            PAGE * row_pages as u32,
        );
        let cache = support::cache(
            model.cache(),
            model.dtype(),
            (rows * row_pages, PAGE),
            meta(Shape::from((rows, row_pages)), DType::U32),
            &meta,
        );
        let shapes = model.state_layers();
        let state = support::state(
            [
                column(requests + 1, DType::U32),
                column(requests, DType::U32),
                column(requests, DType::U32),
                column(requests, DType::U32),
            ],
            &support::state_buffers(&shapes, SLOTS, &meta),
            &support::state_buffers(&shapes, SLOTS, &meta),
            Some((
                meta(Shape::from((requests, STEP_ROWS)), DType::U32),
                column(requests, DType::U32),
            )),
        );

        let embedded = model
            .embed_input_ids(&column(tokens, DType::U32))
            .expect("the embeddings");
        assert_eq!(embedded.dims(), [tokens, model.hidden() as usize]);
        assert_eq!(embedded.dtype(), model.dtype());

        let streams = model
            .forward(&batch, &[cache], Some(&state), None, &embedded)
            .unwrap_or_else(|error| panic!("the {phase:?} forward: {error}"));
        assert_eq!(streams.dims(), [tokens, model.stream_width() as usize]);
        assert_eq!(streams.dtype(), model.dtype());

        // Execution gathers the rows a batch asks logits for: one per row.
        let logits = model
            .compute_logits(
                &streams
                    .narrow(0, 0, rows)
                    .expect("one row of streams per row"),
            )
            .expect("the logits");
        assert_eq!(logits.dims(), [rows, model.vocabulary() as usize]);
        assert_eq!(logits.dtype(), DType::F32);
    }

    let launches = launches(&context);

    // The grouped norm opens each read of the streams: two a layer, and three more in the
    // layer that holds the PLE layer (the stack's second). The launches from one layer's
    // first read to the next layer's are one layer's chain.
    let reads: Vec<usize> = launches
        .iter()
        .enumerate()
        .filter(|(_, (operator, key))| *operator == "rms_norm" && key.contains("_groups_"))
        .map(|(index, _)| index)
        .collect();
    let first_read = |layer: usize| reads[7 + 2 * (layer - 2)];
    let layer = |layer: usize| -> BTreeMap<&'static str, usize> {
        let mut operators = BTreeMap::new();
        for (operator, _) in &launches[first_read(layer)..first_read(layer + 1)] {
            *operators.entry(*operator).or_default() += 1;
        }
        operators
    };

    Profile {
        launches: launches.len(),
        // The stack's third layer is a gated delta network and its fourth attends.
        linear_layer: layer(2),
        attention_layer: layer(3),
        keys: launches.into_iter().collect(),
        peak: context.peak(),
        host_bytes: context.input_peak(),
    }
}

/// Runs the two verify buckets and the prefill bucket of `model` twice each, asserts that both
/// runs of a bucket state the same kernels and reach the same peak, and prints them under
/// `name`.
fn buckets(name: &str, model: &dyn TextModel, group: Option<(u64, u64)>) -> Vec<Profile> {
    [
        (Phase::Decode, (STEP_ROWS, STEP_ROWS, 1)),
        (Phase::Decode, (8 * STEP_ROWS, 8 * STEP_ROWS, 8)),
        (Phase::Prefill, (2, 512, 2)),
    ]
    .into_iter()
    .map(|(phase, bucket)| {
        let first = step(model, phase, bucket, group);
        let second = step(model, phase, bucket, group);
        assert_eq!(first, second, "two runs of one {phase:?} bucket differ");

        let operators: BTreeSet<&str> = first.keys.iter().map(|(operator, _)| *operator).collect();
        println!(
            "{name} {phase:?} {} rows, {} tokens: {} launches, {} kernel keys of {} operators, \
             {} launches a gated delta layer, {} an attention layer, allocator peak {} bytes, \
             {} bytes of host values",
            bucket.0,
            bucket.1,
            first.launches,
            first.keys.len(),
            operators.len(),
            first.linear_layer.values().sum::<usize>(),
            first.attention_layer.values().sum::<usize>(),
            first.peak,
            first.host_bytes
        );
        println!(
            "{name} {phase:?} a gated delta layer: {:?}",
            first.linear_layer
        );
        println!(
            "{name} {phase:?} an attention layer: {:?}",
            first.attention_layer
        );
        println!("{name} {phase:?} operators: {operators:?}");
        for (operator, key) in &first.keys {
            println!("{name} {phase:?} key: {operator} {key}");
        }
        assert!(first.peak > 0, "a forward allocates its activations");

        first
    })
    .collect()
}

/// Asserts that `profile` states every operator of `expected` and none of `absent`.
fn assert_operators(what: &str, profile: &Profile, expected: &[&str], absent: &[&str]) {
    let named: BTreeSet<&str> = profile.keys.iter().map(|(operator, _)| *operator).collect();

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

/// The operators every step of the model launches.
const STEP: [&str; 14] = [
    "rms_norm",
    "gemm",
    "causal_conv1d",
    "gated_delta_rule",
    "gated_rms_norm",
    "qk_norm_rope",
    "rope_apply",
    "kv_write",
    "rows",
    "index_scores",
    "top_k",
    "indexed_attention",
    "topk_softmax",
    "sigmoid_gated_add",
];

/// Asserts what a loaded model states of its cache, its state and its experts at `tensor`'s
/// rank: twelve attention layers with index keys of 128 per four tokens; thirty-six gated
/// delta networks with a window of the history and the drafts and a state of value rows; one
/// PLE window; twelve rings of eight raw keys.
fn assert_shapes(model: &dyn TextModel, tensor: Split) {
    let ranks = tensor.size as usize;
    let cache = model.cache();
    assert_eq!((cache.layers, cache.head_dim), (12, 256));
    assert_eq!(cache.kv_heads as usize, (2 / ranks).max(1));
    let keys = cache.index_keys.expect("index keys");
    assert_eq!((keys.width, keys.ratio), (128, 4));

    let [linear, ple, ring] = model
        .state_layers()
        .try_into()
        .expect("three kinds of state");
    assert_eq!((linear.layers, ple.layers, ring.layers), (36, 1, 12));
    assert_eq!(linear.fields[0].dims, [3 + DRAFTS as usize, 10_240 / ranks]);
    assert_eq!(linear.fields[1].dims, [48 / ranks, 128, 128]);
    assert_eq!(linear.fields[1].dtype, DType::F32);
    assert_eq!(ple.fields[0].dims, [9 + DRAFTS as usize, 10_240]);
    assert_eq!(ring.fields[0].dims, [8, 128]);
    assert_eq!(
        (ring.fields[1].dims.clone(), ring.fields[1].dtype),
        (vec![8, 3], DType::I32)
    );

    assert_eq!((model.hidden(), model.stream_width()), (2560, 10_240));
    let experts = model.experts().expect("routed experts");
    assert_eq!(
        (experts.experts, experts.top, experts.intermediate),
        (512, 10, 640)
    );
}

/// The profile of a checkpoint whose experts are NVFP4: whole, and as rank one of four.
fn profile_packed(name: &str) {
    for tensor in [Split::WHOLE, Split { size: 4, rank: 1 }] {
        let (model, unread) = load(name, &support::placement(tensor, DRAFTS));
        assert_shapes(&*model, tensor);
        assert!(model.experts().expect("routed experts").encoded);

        // Every tensor of the checkpoint is a module's: the load of the whole model names
        // each, and a rank names all but the shards of the n-gram table other ranks hold.
        let stray: Vec<&String> = unread
            .iter()
            .filter(|name| tensor.size == 1 || !name.contains("ngram_embedding.shard_"))
            .collect();
        assert!(
            stray.is_empty(),
            "{name}: {} tensors no module names, the first {:?}",
            stray.len(),
            &stray[..stray.len().min(12)]
        );

        let label = format!("{name} tensor rank {} of {}", tensor.rank, tensor.size);
        let group = (tensor.size > 1).then_some((u64::from(tensor.rank), u64::from(tensor.size)));
        for profile in buckets(&label, &*model, group) {
            assert_operators("a step", &profile, &STEP, &["gather_gemm", "route_gemm"]);
            assert_operators(
                "a step",
                &profile,
                &["nvfp4_quantize", "gather_gemm_nvfp4", "route_gemm_nvfp4"],
                &[],
            );
        }

        drafter_and_tower(&label, &*model, group);
    }
}

/// The drafter's forward over a draft cache and a ring of its own, and one image through the
/// tower.
fn drafter_and_tower(name: &str, model: &dyn TextModel, group: Option<(u64, u64)>) {
    let drafter = model.drafter().expect("the drafter");
    let draft = drafter.cache();
    assert_eq!((draft.layers, draft.head_dim), (1, 256));
    let shapes = drafter.state();
    assert_eq!(shapes.len(), 1);

    for (rows, tokens, requests) in [(STEP_ROWS, STEP_ROWS, 1), (1, 1, 1)] {
        let context = context(group);
        {
            let _guard = context.enter();
            let column = |length: usize, dtype: DType| meta(Shape::from(length), dtype);
            let batch = support::batch(
                Phase::Decode,
                [
                    column(tokens, DType::I32),
                    column(tokens, DType::U32),
                    column(rows, DType::U32),
                    column(rows + 1, DType::U32),
                    column(rows, DType::U32),
                ],
                None,
                PAGE,
                PAGE * 2,
            );
            let cache = support::cache(
                draft,
                model.dtype(),
                (rows * 2, PAGE),
                meta(Shape::from((rows, 2)), DType::U32),
                &meta,
            );
            let state = support::state(
                [
                    column(requests + 1, DType::U32),
                    column(requests, DType::U32),
                    column(requests, DType::U32),
                    column(requests, DType::U32),
                ],
                &support::state_buffers(&shapes, SLOTS, &meta),
                &support::state_buffers(&shapes, SLOTS, &meta),
                None,
            );

            let drafted = drafter
                .forward_stateful(
                    &batch,
                    &cache,
                    Some(&state),
                    &meta(
                        Shape::from((tokens, model.hidden() as usize)),
                        model.dtype(),
                    ),
                    &meta(
                        Shape::from((tokens, model.stream_width() as usize)),
                        model.dtype(),
                    ),
                )
                .unwrap_or_else(|error| panic!("the drafter's forward: {error}"));
            assert_eq!(drafted.dims(), [tokens, model.stream_width() as usize]);

            let rows_read = drafter
                .logits_rows(&drafted)
                .expect("the drafter's read-out");
            assert_eq!(rows_read.dims(), [tokens, model.hidden() as usize]);
            let logits = model.compute_logits(&rows_read).expect("the draft logits");
            assert_eq!(logits.dims(), [tokens, model.vocabulary() as usize]);
        }

        let launched = launches(&context);
        let operators: BTreeSet<&str> = launched.iter().map(|(operator, _)| *operator).collect();
        println!(
            "{name} drafter {tokens} rows: {} launches, allocator peak {} bytes, operators \
             {operators:?}",
            launched.len(),
            context.peak()
        );
        for operator in [
            "index_scores",
            "indexed_attention",
            "kv_write",
            "topk_softmax",
        ] {
            assert!(
                operators.contains(operator),
                "the drafter launches no {operator}"
            );
        }
        assert!(!operators.contains("gated_delta_rule"));
    }

    // The tower: an image of 448 x 672 pixels is sized to a 28 x 42 patch grid, which merges
    // to 294 rows of the model's width at three M-RoPE axes.
    let tower = model.vision().expect("the tower");
    let grid = tower.image_grid(448, 672).expect("an image's grid");
    assert_eq!(grid, [1, 28, 42]);
    let rows = tower.image_rows(grid).expect("whole merge blocks") as usize;
    assert_eq!((rows, tower.position_axes()), (294, 3));

    let context = context(None);
    {
        let _guard = context.enter();
        let format = tower.patch_format();
        let pixels = meta(
            Shape::from((28 * 42, format.width() as usize)),
            format.dtype,
        );
        let features = meta(Shape::from((rows, model.hidden() as usize)), model.dtype());

        tower
            .encode_image(grid, &pixels, &features)
            .expect("the image's features");
    }
    let launched = launches(&context);
    println!(
        "{name} tower 28x42: {} launches, allocator peak {} bytes",
        launched.len(),
        context.peak()
    );
    assert!(
        launched
            .iter()
            .any(|(operator, _)| *operator == "dense_attention")
    );
}

/// The NVIDIA export: NVFP4 experts in every decoder layer, the drafter's experts
/// block-scaled FP8 (decoded dense at load), the n-gram table FP8.
#[test]
#[ignore = "requires /workspace/models/Qwen3.8-Flash-Next-NVFP4-nvidia"]
fn the_nvidia_nvfp4_export_profiles_its_buckets() {
    profile_packed("Qwen3.8-Flash-Next-NVFP4-nvidia");
}

/// The RadixArk export: NVFP4 experts in every decoder layer, the drafter's experts stacked
/// BF16.
#[test]
#[ignore = "requires /workspace/models/Qwen3.8-Flash-Next-NVFP4-RadixArk"]
fn the_radixark_nvfp4_export_profiles_its_buckets() {
    profile_packed("Qwen3.8-Flash-Next-NVFP4-RadixArk");
}

/// The FP8 export stores the experts of every decoder layer as block-scaled FP8, which no
/// operator of this build multiplies and no load decodes: its configuration opens and its load
/// is refused by the representation, before any tensor is read.
#[test]
#[ignore = "requires /workspace/models/Qwen3.8-Flash-Next-FP8"]
fn the_fp8_export_is_refused_by_its_experts_representation() {
    let name = "Qwen3.8-Flash-Next-FP8";
    let checkpoint = checkpoint(name);

    let config = models::qwen4_exp::Config::open(&checkpoint).expect("the configuration");
    assert_eq!(config.text.num_hidden_layers, 48);
    assert_eq!(config.ngram.table_rows(), 320_001_536);

    for tensor in [Split::WHOLE, Split { size: 4, rank: 0 }] {
        let mut loader = WeightLoader::meta(&checkpoint, None);
        let refused = support::load(
            &checkpoint,
            &mut loader,
            &support::placement(tensor, DRAFTS),
        )
        .err()
        .unwrap_or_else(|| panic!("{name} loaded"));

        let message = refused.to_string();
        println!("{name}: {message}");
        assert!(
            matches!(refused, nn::Error::Model { .. }) && message.contains("weight_scale_inv"),
            "{message}"
        );
    }
}
