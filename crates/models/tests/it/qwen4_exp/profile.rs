//! Profile runs on the meta device over the released checkpoints' headers.
//!
//! A meta load reads a checkpoint's shard headers and no tensor data, and a forward over meta
//! tensors under a meta context launches nothing: it checks every operand, states the kernel
//! each launch would run and counts what the forward allocates. Each test loads a released
//! checkpoint through its family, runs verify buckets (one and eight requests of four rows:
//! the committed token and three drafts) and a prefill bucket (the embedding, the decoder
//! stack and the logits), at tensor parallelism one and as a rank of four, and asserts the
//! shapes of the outputs, that every launch names an operator, that no kernel key names a
//! model family, and that a second run of the same bucket states the same keys and reaches the
//! same allocator peak. It prints the keys, the peaks, the bytes of host values a forward moves
//! to the device, and the launches of one layer of each kind.
//!
//! The tests read the releases of [`crate::releases`] through the Hugging Face cache and need
//! no device; they are ignored by default and fail, naming the release, when asked for without
//! it:
//!
//! ```text
//! cargo test -p models --test it qwen4_exp::profile -- --ignored --nocapture
//! ```

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use nn::family::{CacheShape, Family, TextModel, TextPlacement};
use nn::{DType, KvCache, MetaWeights, Phase, RecurrentState, Split, WeightLoader};
use nn_testing::text::{self, WORKSPACE_BYTES};
use nn_testing::{
    Launch, PAGE_TOKENS, Recording, Release, assert_generic, assert_operators, meta, record, report,
};

use crate::releases::{
    QWEN3_8_FLASH_NEXT_FP8, QWEN3_8_FLASH_NEXT_NVFP4, QWEN3_8_FLASH_NEXT_NVFP4_RADIXARK,
};

/// The family the profiled checkpoints declare.
const FAMILIES: [Family; 1] = [models::qwen4_exp::family::FAMILY];

/// Request slots of the recurrent state's buffers.
const SLOTS: usize = 64;

/// Tokens a step of the profiled deployment drafts, and the rows a request has in a step.
const DRAFTS: u32 = 3;
const STEP_ROWS: usize = DRAFTS as usize + 1;

/// The kernels a forward stated: `(operator, key)` pairs.
type Keys = BTreeSet<(&'static str, String)>;

/// The profiled deployment at `tensor`'s rank: it drafts [`DRAFTS`] tokens a step with the
/// checkpoint's drafter.
fn placement(tensor: Split) -> TextPlacement {
    TextPlacement {
        tensor,
        drafter: true,
        draft_tokens: DRAFTS,
        ..TextPlacement::WHOLE
    }
}

/// The model of `release`, meta-loaded for `placement`, with the checkpoint tensors the load
/// consumed none of: neither a weight's tensor the meta store holds a key of, nor a value the
/// loader read (a tensor scale).
fn load(release: Release, placement: &TextPlacement) -> (Arc<dyn TextModel>, Vec<String>) {
    let checkpoint = release.checkpoint();
    let store = MetaWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    let model = text::load(&FAMILIES, &checkpoint, &mut loader, placement)
        .unwrap_or_else(|error| panic!("{release}: {error}"));

    let (read, _) = checkpoint.reading();
    let mut consumed = store.tensors();
    consumed.extend(read);
    let unnamed = checkpoint
        .names()
        .filter(|tensor| !consumed.contains(*tensor))
        .map(str::to_owned)
        .collect();
    (model, unnamed)
}

/// The meta cache group of `shape` in `dtype` over `rows` rows of `row_pages` pages each.
fn cache(shape: CacheShape, dtype: DType, rows: usize, row_pages: usize) -> KvCache {
    KvCache::allocate(
        shape,
        dtype,
        rows * row_pages,
        PAGE_TOKENS,
        meta((rows, row_pages), DType::U32),
        meta(WORKSPACE_BYTES, DType::U8),
        |shape, dtype| Ok(meta(shape, dtype)),
    )
    .expect("a meta cache group")
}

/// Meta buffers of every layer of `shapes`, [`SLOTS`] rows each.
fn buffers(shapes: &[nn::family::StateShape]) -> Vec<Vec<nn::Tensor>> {
    RecurrentState::buffers(shapes, |_| SLOTS, |shape, dtype| Ok(meta(shape, dtype)))
        .expect("the state buffers")
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
/// `requests` state rows hold them, at this rank's place `group` of its deployment: what it
/// recorded, each launch checked ([`assert_generic`]).
fn step(
    model: &dyn TextModel,
    phase: Phase,
    (rows, tokens, requests): (usize, usize, usize),
    group: &[Split],
) -> Recording {
    let row_pages = (tokens / rows).div_ceil(PAGE_TOKENS as usize).max(2);

    let (stepped, recording) = record(group, || {
        let batch = text::batch(
            phase,
            [
                meta(tokens, DType::I32),
                meta(tokens, DType::U32),
                meta(rows, DType::U32),
                meta(rows + 1, DType::U32),
                meta(rows, DType::U32),
            ],
            Some((meta(tokens, DType::U32), meta((rows, 2), DType::U32))),
            PAGE_TOKENS,
            PAGE_TOKENS * row_pages as u32,
        );
        let cache = cache(model.cache(), model.dtype(), rows, row_pages);
        let shapes = model.state_layers();
        let state = text::state(
            [
                meta(requests + 1, DType::U32),
                meta(requests, DType::U32),
                meta(requests, DType::U32),
                meta(requests, DType::U32),
            ],
            &buffers(&shapes),
            &buffers(&shapes),
            Some((
                meta((requests, STEP_ROWS), DType::U32),
                meta(requests, DType::U32),
            )),
        );

        let embedded = model
            .embed_input_ids(&meta(tokens, DType::U32))
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
        Ok(())
    });
    stepped.unwrap_or_else(|error| panic!("the {phase:?} step: {error}"));

    assert_generic(&format!("a {phase:?} step"), &recording.launches);
    recording
}

/// The profile of `recording`, one step of a bucket.
fn profile(recording: &Recording) -> Profile {
    let launches: &[Launch] = &recording.launches;

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
        keys: launches.iter().cloned().collect(),
        peak: recording.peak,
        host_bytes: recording.host_bytes,
    }
}

/// Runs the two verify buckets and the prefill bucket of `model` twice each, asserts that both
/// runs of a bucket state the same kernels and reach the same peak, and reports them under
/// `name`.
fn buckets(name: &str, model: &dyn TextModel, group: &[Split]) -> Vec<Profile> {
    [
        (Phase::Decode, (STEP_ROWS, STEP_ROWS, 1)),
        (Phase::Decode, (8 * STEP_ROWS, 8 * STEP_ROWS, 8)),
        (Phase::Prefill, (2, 512, 2)),
    ]
    .into_iter()
    .map(|(phase, bucket)| {
        let recording = step(model, phase, bucket, group);
        let first = profile(&recording);
        let second = profile(&step(model, phase, bucket, group));
        assert_eq!(first, second, "two runs of one {phase:?} bucket differ");

        let run = format!("{name} {phase:?} {} rows, {} tokens", bucket.0, bucket.1);
        report(&run, &recording);
        println!(
            "{run}: {} launches a gated delta layer, {} an attention layer",
            first.linear_layer.values().sum::<usize>(),
            first.attention_layer.values().sum::<usize>(),
        );
        println!("{run} a gated delta layer: {:?}", first.linear_layer);
        println!("{run} an attention layer: {:?}", first.attention_layer);
        assert!(first.peak > 0, "a forward allocates its activations");

        first
    })
    .collect()
}

/// The operators of `profile`.
fn operators(profile: &Profile) -> BTreeSet<&'static str> {
    profile.keys.iter().map(|(operator, _)| *operator).collect()
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
    let ranks = tensor.size() as usize;
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
fn profile_packed(release: Release) {
    for tensor in [Split::WHOLE, Split::new(4, 1).unwrap()] {
        let (model, unread) = load(release, &placement(tensor));
        assert_shapes(&*model, tensor);
        assert!(model.experts().expect("routed experts").encoded);

        // Every tensor of the checkpoint is a module's: the load of the whole model names
        // each, and a rank names all but the shards of the n-gram table other ranks hold.
        let stray: Vec<&String> = unread
            .iter()
            .filter(|name| tensor.size() == 1 || !name.contains("ngram_embedding.shard_"))
            .collect();
        assert!(
            stray.is_empty(),
            "{release}: {} tensors no module names, the first {:?}",
            stray.len(),
            &stray[..stray.len().min(12)]
        );

        let label = format!(
            "{} tensor rank {} of {}",
            release.repository,
            tensor.rank(),
            tensor.size()
        );
        let group: &[Split] = if tensor.size() > 1 { &[tensor] } else { &[] };
        for profile in buckets(&label, &*model, group) {
            assert_operators(
                "a step",
                &operators(&profile),
                &STEP,
                &["gather_gemm", "route_gemm"],
            );
            assert_operators(
                "a step",
                &operators(&profile),
                &["nvfp4_quantize", "gather_gemm_nvfp4", "route_gemm_nvfp4"],
                &[],
            );
        }

        drafter_and_tower(&label, &*model, group);
    }
}

/// The drafter's forward over a draft cache and a ring of its own, and one image through the
/// tower.
fn drafter_and_tower(name: &str, model: &dyn TextModel, group: &[Split]) {
    let drafter = model.drafter().expect("the drafter");
    let draft = drafter.cache();
    assert_eq!((draft.layers, draft.head_dim), (1, 256));
    let shapes = drafter.state();
    assert_eq!(shapes.len(), 1);

    for (rows, tokens, requests) in [(STEP_ROWS, STEP_ROWS, 1), (1, 1, 1)] {
        let (drafted, recording) = record(group, || {
            let batch = text::batch(
                Phase::Decode,
                [
                    meta(tokens, DType::I32),
                    meta(tokens, DType::U32),
                    meta(rows, DType::U32),
                    meta(rows + 1, DType::U32),
                    meta(rows, DType::U32),
                ],
                None,
                PAGE_TOKENS,
                PAGE_TOKENS * 2,
            );
            let cache = cache(draft, model.dtype(), rows, 2);
            let state = text::state(
                [
                    meta(requests + 1, DType::U32),
                    meta(requests, DType::U32),
                    meta(requests, DType::U32),
                    meta(requests, DType::U32),
                ],
                &buffers(&shapes),
                &buffers(&shapes),
                None,
            );

            let drafted = drafter
                .forward_stateful(
                    &batch,
                    &cache,
                    Some(&state),
                    &meta((tokens, model.hidden() as usize), model.dtype()),
                    &meta((tokens, model.stream_width() as usize), model.dtype()),
                )
                .unwrap_or_else(|error| panic!("the drafter's forward: {error}"));
            assert_eq!(drafted.dims(), [tokens, model.stream_width() as usize]);

            let rows_read = drafter
                .logits_rows(&drafted)
                .expect("the drafter's read-out");
            assert_eq!(rows_read.dims(), [tokens, model.hidden() as usize]);
            let logits = model.compute_logits(&rows_read).expect("the draft logits");
            assert_eq!(logits.dims(), [tokens, model.vocabulary() as usize]);
            Ok(())
        });
        drafted.unwrap_or_else(|error| panic!("the drafter's step: {error}"));

        report(&format!("{name} drafter {tokens} rows"), &recording);
        let operators: BTreeSet<&str> = recording
            .launches
            .iter()
            .map(|(operator, _)| *operator)
            .collect();
        assert_operators(
            "the drafter",
            &operators,
            &[
                "index_scores",
                "indexed_attention",
                "kv_write",
                "topk_softmax",
            ],
            &["gated_delta_rule"],
        );
    }

    // The tower: an image of 448 x 672 pixels is sized to a 28 x 42 patch grid, which merges
    // to 294 rows of the model's width at three M-RoPE axes.
    let tower = model.vision().expect("the tower");
    let grid = tower.image_grid(448, 672).expect("an image's grid");
    assert_eq!(grid, [1, 28, 42]);
    let rows = tower.image_rows(grid).expect("whole merge blocks") as usize;
    assert_eq!((rows, tower.position_axes()), (294, 3));

    let (encoded, recording) = record(&[], || {
        let format = tower.patch_format();
        let pixels = meta((28 * 42, format.width() as usize), format.dtype);
        let features = meta((rows, model.hidden() as usize), model.dtype());

        tower.encode_image(grid, &pixels, &features)
    });
    encoded.expect("the image's features");
    report(&format!("{name} tower 28x42"), &recording);
    assert!(
        recording
            .launches
            .iter()
            .any(|(operator, _)| *operator == "dense_attention")
    );
}

/// The NVIDIA export stores its drafter's experts as weight-only block-scaled FP8
/// (`FP8_PB_WO`), which no operator of this build multiplies: the family refuses the checkpoint
/// when it opens it, naming the representation, and the checkpoint is refused until an
/// operator serves that representation (decision D12).
#[test]
#[ignore = "reads the Hugging Face release nvidia/Qwen3.8-Flash-Next-NVFP4"]
fn the_nvidia_nvfp4_export_is_refused_by_its_drafters_experts() {
    let release = QWEN3_8_FLASH_NEXT_NVFP4;
    let refused = text::open(&FAMILIES, &release.checkpoint())
        .err()
        .unwrap_or_else(|| panic!("{release} opened"));

    assert!(refused.to_string().contains("FP8_PB_WO"), "{refused}");
}

/// The RadixArk export: NVFP4 experts in every decoder layer, the drafter's experts stacked
/// BF16.
#[test]
#[ignore = "reads the Hugging Face release RadixArk/Qwen3.8-Flash-Next-NVFP4"]
fn the_radixark_nvfp4_export_profiles_its_buckets() {
    profile_packed(QWEN3_8_FLASH_NEXT_NVFP4_RADIXARK);
}

/// The FP8 export declares the producer `fp8`, whose block-scaled FP8 experts no operator of
/// this build multiplies: the family refuses the checkpoint when it opens it, naming the
/// producer, and the checkpoint is refused until an operator serves its representation
/// (decision D12).
#[test]
#[ignore = "reads the Hugging Face release Qwen/Qwen3.8-Flash-Next-FP8"]
fn the_fp8_export_is_refused_by_its_producer() {
    let release = QWEN3_8_FLASH_NEXT_FP8;
    let refused = text::open(&FAMILIES, &release.checkpoint())
        .err()
        .unwrap_or_else(|| panic!("{release} opened"));

    assert!(refused.to_string().contains("`fp8`"), "{refused}");
}
