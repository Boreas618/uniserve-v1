//! Reference vectors on the CPU device: each architecture's tiny checkpoint, loaded through
//! the family's own loading path, computes what the reference implementation computed from the
//! same weights in PyTorch (the `qwen3-fixtures` generator, `crates/models/python`).
//!
//! | Fixture | Reference | What is compared |
//! | --- | --- | --- |
//! | `qwen3_moe` | `Qwen3MoeForCausalLM` | hidden states and logits of two rows |
//! | `qwen3_5_moe` | `Qwen3_5MoeForConditionalGeneration` | the same through gated delta networks, the drafter, an image's features, a prompt holding the image |
//! | `qwen3_vl` | `Qwen3VLModel` | the tower's features with DeepStack, a prompt with an image and a video, whole and truncated |
//!
//! Every causal model also runs a prefill followed by decode steps through the key/value cache
//! and the recurrent state, which must compute what the one prefill of the whole rows did.
//!
//! ## The tolerance
//!
//! The fixtures are FP32 and both sides evaluate the same formulas in FP32, in different
//! orders: PyTorch's matrix products and reductions are blocked and vectorized, the tensor
//! library's are its own, and a fused function rounds where a composition rounds twice. A
//! reduction of `n` terms evaluated in FP32 differs from its exact value by at most
//! `n · u · Σ|terms|` to first order, `u = 2⁻²⁴` being the unit roundoff, so two evaluations
//! differ by at most twice that, and the errors of consecutive stages add to first order. The
//! bound of an output is therefore `2 · u · terms`, where `terms` counts the terms of every
//! reduction on the path from the inputs ([`layer_terms`], [`tower_terms`]), relative to the
//! magnitude of the values reduced, which the largest magnitude of the compared tensor stands
//! for: a reduction's bound is relative to the sum of its terms' magnitudes, not to a result
//! they may cancel to. The count is a function of the fixture's widths and of nothing
//! measured.
//!
//! Nothing here touches a GPU.

use std::ops::Range;
use std::sync::Arc;

use models::qwen3::config::VisionConfig;
use models::qwen3::{Config, Qwen3VlModel, positions};
use nn::family::{CacheShape, Family, TextModel, TextPlacement};
use nn::{
    CpuWeights, DType, Device, Encoder, KvCache, Phase, RecurrentState, Shape, Split, Tensor,
    TokenBatch, WeightLoader,
};
use nn_testing::text::{self, WORKSPACE_BYTES};
use nn_testing::{Fixture, PAGE_TOKENS};

/// The families the fixtures' checkpoints declare.
const FAMILIES: [Family; 2] = [
    models::qwen3::family::FAMILY,
    models::qwen3::family::HYBRID_FAMILY,
];

/// The pages of each of two rows, in the order the row fills them, out of [`PAGES`]: a row of
/// seventy tokens spans two pages of [`PAGE_TOKENS`].
const TABLES: [[u32; 2]; 2] = [[3, 0], [1, 2]];

/// Pages of a cache group.
const PAGES: usize = 4;

/// Request slots of the recurrent state's buffers, and the slot each row holds.
const SLOTS: usize = 3;
const ROW_SLOTS: [u32; 2] = [2, 0];

/// The unit roundoff of FP32.
const UNIT_ROUNDOFF: f32 = f32::EPSILON / 2.0;

/// The bound two FP32 evaluations of one formula differ by when their reductions sum `terms`
/// terms in all (module documentation).
fn tolerance(terms: u32) -> f32 {
    2.0 * UNIT_ROUNDOFF * terms as f32
}

/// The terms the reductions of one decoder layer of `config` sum for a token that reads
/// `keys` keys or earlier tokens.
fn layer_terms(config: &Config, linear: bool, keys: u32) -> u32 {
    let text = &config.text;
    let hidden = text.hidden_size;

    let mixer = match (linear, text.gated_delta_net) {
        // The projections of the row, the convolution's taps, the normalization of each
        // query and key, the state's decay and update by every earlier token, its read, the
        // output norm and the output projection.
        (true, Some(network)) => {
            hidden
                + network.conv_width
                + 2 * network.key_dim
                + keys * 2 * network.key_dim
                + network.key_dim
                + network.value_dim
                + network.value_heads * network.value_dim
        }
        // The projections of the row, the norm of each head, its rotation, the scores'
        // products, the softmax and the weighted sum over the keys, the gate and the output
        // projection.
        _ => {
            hidden
                + text.head_dim
                + 2
                + text.head_dim
                + 2 * keys
                + 1
                + text.num_attention_heads * text.head_dim
        }
    };

    let feed_forward = match &config.moe {
        // The router, its softmax over the experts and the renormalization of the chosen
        // ones, each expert's two products and the weighted sum of the routes; then the
        // shared expert's products, its gate and its sum into the routed output.
        Some(moe) => {
            let routed = hidden
                + moe.num_experts
                + moe.num_experts_per_tok
                + hidden
                + 1
                + moe.moe_intermediate_size
                + moe.num_experts_per_tok;
            let shared = moe
                .shared_expert_intermediate_size
                .map_or(0, |intermediate| hidden + 1 + intermediate + hidden + 2);

            routed + shared
        }
        None => hidden + 1 + text.intermediate_size,
    };

    // The two pre-norms' mean squares and the two sums into the stream.
    2 * hidden + 2 + mixer + feed_forward
}

/// The terms of the first `layers` decoder layers of `config` over rows of `keys` tokens, and
/// of the final norm.
fn stack_terms(config: &Config, layers: usize, keys: u32) -> u32 {
    let stack: u32 = config.text.linear_layers[..layers]
        .iter()
        .map(|linear| layer_terms(config, *linear, keys))
        .sum();

    stack + config.text.hidden_size
}

/// The terms of the vision tower of `vision` over frames of `frame` patches: the patch
/// embedding, the position blend, each block's two normalizations (a mean and a variance),
/// projections, rotation, attention over the frame and MLP, and a merger.
fn tower_terms(vision: &VisionConfig, frame: u32) -> u32 {
    let hidden = vision.hidden_size;
    let block = 4 * hidden
        + hidden
        + 2
        + vision.head_dim()
        + 2 * frame
        + hidden
        + hidden
        + 1
        + vision.intermediate_size
        + 2;
    let merger = 2 * vision.merged_width() + vision.merged_width() + 1 + vision.merged_width();

    vision.tubelet() + 4 + vision.depth * block + merger
}

/// The `Qwen3MoeForCausalLM` fixture: its tiny checkpoint under `model` and the reference's
/// vectors in `reference.safetensors`.
fn qwen3_moe() -> Fixture {
    nn_testing::fixture!("data/qwen3_moe": [
        "model/config.json",
        "model/model.safetensors",
        "reference.safetensors",
    ])
}

/// The `Qwen3_5MoeForConditionalGeneration` fixture, with the processor files of its tower.
fn qwen3_5_moe() -> Fixture {
    nn_testing::fixture!("data/qwen3_5_moe": [
        "model/config.json",
        "model/model.safetensors",
        "model/preprocessor_config.json",
        "model/video_preprocessor_config.json",
        "reference.safetensors",
    ])
}

/// The `Qwen3VLModel` fixture, with the processor files of its tower.
fn qwen3_vl() -> Fixture {
    nn_testing::fixture!("data/qwen3_vl": [
        "model/config.json",
        "model/model.safetensors",
        "model/preprocessor_config.json",
        "model/video_preprocessor_config.json",
        "reference.safetensors",
    ])
}

/// The token ids the reference stored as `ids`.
fn ids(ids: &Tensor) -> Vec<u32> {
    let ids: Vec<i32> = ids.to_vec1().expect("a column of ids");
    ids.into_iter().map(|id| id as u32).collect()
}

/// A CPU column of U32 `values`.
fn column(values: &[u32]) -> Tensor {
    Tensor::new(values, &Device::Cpu).expect("a column")
}

/// A CPU tensor of zeros.
fn zeros(shape: Shape, dtype: DType) -> Tensor {
    Tensor::zeros(shape, dtype, &Device::Cpu).expect("zeros")
}

/// The causal language model of `fixture`, loaded on the CPU device whole, with its drafter
/// when `drafter` asks for it.
fn model(fixture: &Fixture, drafter: bool) -> (Arc<dyn TextModel>, Config) {
    let checkpoint = fixture.checkpoint("model");
    let store = CpuWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    let placement = TextPlacement {
        drafter,
        vision: true,
        ..TextPlacement::WHOLE
    };

    (
        text::load(&FAMILIES, &checkpoint, &mut loader, &placement).expect("the load"),
        Config::open(&checkpoint).expect("the configuration"),
    )
}

/// The batch of a call in which row `r` presents its tokens at positions `ranges[r]`, their
/// cache slots on the pages [`TABLES`] gives the row.
fn batch(phase: Phase, ranges: &[Range<u32>]) -> TokenBatch {
    let positions: Vec<i32> = ranges
        .iter()
        .flat_map(|range| range.clone().map(|position| position as i32))
        .collect();
    let slots: Vec<u32> = ranges
        .iter()
        .zip(TABLES)
        .flat_map(|(range, pages)| {
            range.clone().map(move |position| {
                pages[(position / PAGE_TOKENS) as usize] * PAGE_TOKENS + position % PAGE_TOKENS
            })
        })
        .collect();
    let seq_lens: Vec<u32> = ranges.iter().map(|range| range.end).collect();

    let mut offsets = vec![0];
    for range in ranges {
        offsets.push(offsets[offsets.len() - 1] + range.len() as u32);
    }
    let last: Vec<u32> = offsets[1..].iter().map(|end| end - 1).collect();

    text::batch(
        phase,
        [
            Tensor::new(positions.as_slice(), &Device::Cpu).expect("positions"),
            column(&slots),
            column(&seq_lens),
            column(&offsets),
            column(&last),
        ],
        None,
        PAGE_TOKENS,
        PAGE_TOKENS * TABLES[0].len() as u32,
    )
}

/// An empty cache group of `shape` for `rows` rows on the pages of [`TABLES`].
fn cache(shape: CacheShape, rows: usize) -> KvCache {
    let tables = column(&TABLES[..rows].concat())
        .reshape((rows, TABLES[0].len()))
        .expect("a block table");

    KvCache::allocate(
        shape,
        DType::F32,
        PAGES,
        PAGE_TOKENS,
        tables,
        zeros(Shape::from(WORKSPACE_BYTES), DType::U8),
        |shape, dtype| Ok(zeros(shape, dtype)),
    )
    .expect("a cache group")
}

/// The state buffers of `model`, zeroed, when it keeps a state.
fn buffers(model: &dyn TextModel) -> Option<Vec<Vec<Tensor>>> {
    let runs = model.state_layers();
    (!runs.is_empty()).then(|| {
        RecurrentState::buffers(&runs, |_| SLOTS, |shape, dtype| Ok(zeros(shape, dtype)))
            .expect("the state buffers")
    })
}

/// One forward of `model` over the tokens `ranges` of the rows' `ids`: the final hidden
/// states of those tokens, row after row. `state` holds the live and checkpoint buffers of a
/// model that keeps a state.
fn forward(
    model: &dyn TextModel,
    phase: Phase,
    ids: &[Vec<u32>],
    ranges: &[Range<u32>],
    cache: &KvCache,
    state: Option<&[Vec<Vec<Tensor>>; 2]>,
) -> Tensor {
    let batch = batch(phase, ranges);
    let tokens: Vec<u32> = ids
        .iter()
        .zip(ranges)
        .flat_map(|(ids, range)| ids[range.start as usize..range.end as usize].to_vec())
        .collect();
    let embedded = model
        .embed_input_ids(&column(&tokens))
        .expect("the embeddings");

    // Each row advances its own slot from where its earlier tokens left it.
    let state = state.map(|[live, checkpoint]| {
        text::state(
            [
                batch.query_offsets.clone(),
                batch.seq_lens.clone(),
                column(&ROW_SLOTS[..ranges.len()]),
                column(&vec![0; ranges.len()]),
            ],
            live,
            checkpoint,
            None,
        )
    });

    model
        .forward(
            &batch,
            std::slice::from_ref(cache),
            state.as_ref(),
            None,
            &embedded,
        )
        .unwrap_or_else(|error| panic!("the {phase:?} forward: {error}"))
}

/// Asserts that `actual` is `expected` to within `tolerance` of the largest magnitude of
/// `expected`, and prints the largest difference found.
fn assert_close(what: &str, actual: &Tensor, expected: &Tensor, tolerance: f32) {
    assert_eq!(actual.dims(), expected.dims(), "{what}: the shapes differ");
    let flat = |tensor: &Tensor| -> Vec<f32> {
        tensor
            .to_dtype(DType::F32)
            .and_then(|tensor| tensor.flatten_all())
            .and_then(|tensor| tensor.to_vec1())
            .expect("a tensor's elements")
    };
    let (actual, expected) = (flat(actual), flat(expected));

    let scale = expected.iter().fold(0.0_f32, |scale, x| scale.max(x.abs()));
    let worst = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f32, f32::max);
    println!(
        "{what}: differs by {:.2e} of its magnitude, bound {tolerance:.2e}",
        worst / scale
    );

    for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance * scale,
            "{what}, element {index}: {actual} against {expected}, beyond {tolerance} of {scale}"
        );
    }
}

/// The rows `rows` of `tensor`, one after another.
fn rows(tensor: &Tensor, rows: &[Range<u32>]) -> Tensor {
    let parts: Vec<Tensor> = rows
        .iter()
        .map(|range| {
            tensor
                .narrow(0, range.start as usize, range.len())
                .expect("rows of the tensor")
        })
        .collect();

    Tensor::cat(&parts, 0).expect("the rows")
}

/// The two rows of the fixture `name`, `fixture`, through its causal language model: one
/// prefill of both rows against the reference's hidden states and logits, then a prefill of
/// each row's opening tokens and decode steps through the cache and the state against the one
/// prefill.
fn check_causal_lm(name: &str, fixture: &Fixture) {
    let (model, config) = model(fixture, false);
    let reference = fixture.tensors("reference.safetensors");
    let model = &*model;

    let ids: Vec<Vec<u32>> = (0..2)
        .map(|row| ids(&reference[&format!("rows.{row}.ids")]))
        .collect();
    let lengths: Vec<u32> = ids.iter().map(|ids| ids.len() as u32).collect();
    let expected = |name: &str| {
        let parts: Vec<&Tensor> = (0..2)
            .map(|row| &reference[&format!("rows.{row}.{name}")])
            .collect();
        Tensor::cat(&parts, 0).expect("the two rows")
    };

    let layers = config.text.num_hidden_layers as usize;
    let hidden_bound = tolerance(stack_terms(&config, layers, lengths[0]));
    let logits_bound = tolerance(stack_terms(&config, layers, lengths[0]) + model.hidden());
    let state = || buffers(model).map(|live| [live, buffers(model).expect("a state")]);

    // 1. Both rows in one prefill.
    let whole: Vec<Range<u32>> = lengths.iter().map(|length| 0..*length).collect();
    let hidden = forward(
        model,
        Phase::Prefill,
        &ids,
        &whole,
        &cache(model.cache(), 2),
        state().as_ref(),
    );
    assert_close(
        &format!("{name} hidden states"),
        &hidden,
        &expected("hidden"),
        hidden_bound,
    );

    let logits = model.compute_logits(&hidden).expect("the logits");
    assert_eq!(
        logits.dims(),
        [hidden.dims()[0], model.vocabulary() as usize]
    );
    assert_close(
        &format!("{name} logits"),
        &logits,
        &expected("logits"),
        logits_bound,
    );

    // 2. Each row's opening tokens in a prefill, then its last four tokens one decode step
    //    each, over one cache and one state.
    const STEPS: u32 = 4;
    let (group, state) = (cache(model.cache(), 2), state());
    let opened: Vec<Range<u32>> = lengths.iter().map(|length| 0..length - STEPS).collect();
    let starts = [0, lengths[0]];
    let of_whole = |ranges: &[Range<u32>]| -> Tensor {
        let placed: Vec<Range<u32>> = ranges
            .iter()
            .zip(starts)
            .map(|(range, start)| start + range.start..start + range.end)
            .collect();
        rows(&hidden, &placed)
    };

    let prefilled = forward(model, Phase::Prefill, &ids, &opened, &group, state.as_ref());
    assert_close(
        &format!("{name} opening prefill"),
        &prefilled,
        &of_whole(&opened),
        hidden_bound,
    );

    for step in 0..STEPS {
        let token: Vec<Range<u32>> = opened
            .iter()
            .map(|range| range.end + step..range.end + step + 1)
            .collect();
        let decoded = forward(model, Phase::Decode, &ids, &token, &group, state.as_ref());

        assert_close(
            &format!("{name} decode step {step}"),
            &decoded,
            &of_whole(&token),
            hidden_bound,
        );
    }
}

#[test]
fn qwen3_moe_computes_the_references_hidden_states_and_logits() {
    check_causal_lm("qwen3_moe", &qwen3_moe());
}

#[test]
fn qwen3_5_moe_computes_the_references_hidden_states_and_logits() {
    check_causal_lm("qwen3_5_moe", &qwen3_5_moe());
}

/// The drafter's entries of a row are the reference's, in one prefill and as a prefill
/// followed by decode steps over the draft cache.
#[test]
fn the_drafter_computes_the_references_hidden_states() {
    let fixture = qwen3_5_moe();
    let (model, config) = model(&fixture, true);
    let reference = fixture.tensors("reference.safetensors");
    let drafter = model.drafter().expect("the drafter");

    let shape = drafter.cache();
    assert_eq!((shape.layers, shape.kv_heads, shape.head_dim), (1, 1, 16));

    for row in 0..2 {
        let ids = ids(&reference[&format!("rows.{row}.ids")]);
        let entries = ids.len() as u32 - 1;

        // The fused projection of the two inputs, one attention layer over the entries, and
        // the final norm.
        let bound = tolerance(
            2 * config.text.hidden_size
                + 2 * config.text.hidden_size
                + layer_terms(&config, false, entries)
                + config.text.hidden_size,
        );

        // Entry `i` reads the model's hidden state at `i` and the embedding of token `i + 1`.
        let embedded = model
            .embed_input_ids(&column(&ids[1..]))
            .expect("the embeddings");
        let hidden = reference[&format!("rows.{row}.hidden")]
            .narrow(0, 0, entries as usize)
            .expect("the model's hidden states");
        let draft = |phase, range: Range<u32>, cache: &KvCache| -> Tensor {
            let (first, count) = (range.start as usize, range.len());
            drafter
                .forward(
                    &batch(phase, std::slice::from_ref(&range)),
                    cache,
                    None,
                    &embedded.narrow(0, first, count).expect("the entries"),
                    &hidden.narrow(0, first, count).expect("the entries"),
                )
                .unwrap_or_else(|error| panic!("the drafter's {phase:?} forward: {error}"))
        };

        let drafted = draft(Phase::Prefill, 0..entries, &cache(shape, 1));
        assert_close(
            &format!("row {row} drafted"),
            &drafted,
            &reference[&format!("rows.{row}.drafted")],
            bound,
        );

        // The first entries in a prefill and the rest one decode step each.
        let group = cache(shape, 1);
        let opened = entries - 3;
        let prefilled = draft(Phase::Prefill, 0..opened, &group);
        assert_close(
            &format!("row {row} opening entries"),
            &prefilled,
            &rows(&drafted, std::slice::from_ref(&(0..opened))),
            bound,
        );
        for entry in opened..entries {
            let decoded = draft(Phase::Decode, entry..entry + 1, &group);
            assert_close(
                &format!("row {row} entry {entry}"),
                &decoded,
                &rows(&drafted, std::slice::from_ref(&(entry..entry + 1))),
                bound,
            );
        }
    }
}

/// The hybrid model's tower encodes an image to the reference's features, and a prefill that
/// holds them at the image's rows, under the image's M-RoPE coordinates, computes the
/// reference's hidden states.
#[test]
fn a_prompt_with_an_image_computes_the_references_hidden_states() {
    let fixture = qwen3_5_moe();
    let (model, config) = model(&fixture, false);
    let reference = fixture.tensors("reference.safetensors");
    let model = &*model;
    let tower = model.vision().expect("the tower");
    let vision = config.vision.as_ref().expect("a tower's configuration");

    // The image alone.
    let grid = [1, 4, 6];
    let image_rows = tower.image_rows(grid).expect("whole merge blocks") as usize;
    assert_eq!((image_rows, tower.position_axes()), (6, 3));

    let features = zeros(
        Shape::from((image_rows, model.hidden() as usize)),
        DType::F32,
    );
    tower
        .encode_image(grid, &reference["image.pixels"], &features)
        .expect("the image's features");
    let features_bound = tolerance(tower_terms(vision, 24));
    assert_close(
        "image features",
        &features,
        &reference["image.features"],
        features_bound,
    );

    // The prompt: the features at the image's rows, each row at the coordinates its block
    // states from the position the text before it reached.
    let ids = ids(&reference["prompt.ids"]);
    let tokens = ids.len() as u32;
    let first = ids
        .iter()
        .position(|id| *id == vision.image_token_id)
        .expect("an image");
    let coordinates = positions::rope_index(
        &ids,
        &[grid],
        vision.image_token_id,
        vision.video_token_id,
        vision.spatial_merge_size,
    )
    .expect("the prompt's coordinates");
    assert_eq!(
        tower
            .image_positions(grid, first as u32)
            .expect("the block's coordinates")
            .to_vec2::<i32>()
            .expect("coordinates"),
        coordinates
            .narrow(1, first, image_rows)
            .expect("the block's run")
            .to_vec2::<i32>()
            .expect("coordinates"),
    );

    let embedded = model
        .embed_input_ids(&column(&ids))
        .expect("the embeddings");
    embedded
        .slice_set(&features, 0, first)
        .expect("the image's rows");

    let mut batch = batch(Phase::Prefill, std::slice::from_ref(&(0..tokens)));
    batch.rope_positions = Some(coordinates);
    let state = buffers(model).map(|live| {
        text::state(
            [
                batch.query_offsets.clone(),
                batch.seq_lens.clone(),
                column(&ROW_SLOTS[..1]),
                column(&[0]),
            ],
            &live,
            &buffers(model).expect("a state"),
            None,
        )
    });
    let hidden = model
        .forward(
            &batch,
            &[cache(model.cache(), 1)],
            state.as_ref(),
            None,
            &embedded,
        )
        .expect("the prefill");

    let layers = config.text.num_hidden_layers as usize;
    assert_close(
        "prompt hidden states",
        &hidden,
        &reference["prompt.hidden"],
        features_bound + tolerance(stack_terms(&config, layers, tokens)),
    );
}

/// The Qwen3-VL tower returns each merged row's features beside its DeepStack features, and
/// the encoder computes the reference's states of a prompt with an image and a video: whole
/// with the final norm, and over a prefix of its layers without it.
#[test]
fn the_vl_encoder_computes_the_references_states() {
    let fixture = qwen3_vl();
    let checkpoint = fixture.checkpoint("model");
    let config = Config::open(&checkpoint).expect("the configuration");
    let reference = fixture.tensors("reference.safetensors");
    let vision = config.vision.as_ref().expect("a tower's configuration");
    let images = models::qwen3::config::processor(&checkpoint).expect("the processor files");

    let store = CpuWeights::new();
    let loader = WeightLoader::new(&checkpoint, &store, None);
    let load = |layers, final_norm| {
        Qwen3VlModel::load(
            &config,
            loader.root(),
            images,
            Split::WHOLE,
            layers,
            final_norm,
        )
        .expect("the encoder")
    };

    // The tower over the image: six merged rows, each its features and two DeepStack
    // features.
    let (image_grid, video_grid) = ([1, 4, 6], [2, 2, 4]);
    let whole = load(config.text.num_hidden_layers, true);
    let features = whole
        .tower()
        .forward(image_grid, &reference["image.pixels"])
        .expect("the tower");
    let features_bound = tolerance(tower_terms(vision, 24));
    assert_eq!(features.dims(), [6, 3 * 16]);
    assert_close(
        "image and DeepStack features",
        &features,
        &reference["image.features"],
        features_bound,
    );

    // The prompt: one sequence of every token, the image's and the video's rows in prompt
    // order.
    let ids = ids(&reference["prompt.ids"]);
    let tokens = ids.len() as u32;
    let batch = batch(Phase::Prefill, std::slice::from_ref(&(0..tokens)));
    let pixels = [
        reference["image.pixels"].clone(),
        reference["video.pixels"].clone(),
    ];

    for (name, layers, encoder) in [
        ("prompt.encoded", 3, &whole),
        ("prompt.truncated", 2, &load(2, false)),
    ] {
        let layout = encoder.layout(&batch, &[image_grid, video_grid]);
        assert_eq!((layout.rows, layout.tokens), (tokens, tokens));

        let out = zeros(
            Shape::from((tokens as usize, config.text.hidden_size as usize)),
            DType::F32,
        );
        encoder
            .encode(&ids, &column(&ids), &batch, &pixels, &layout, &out)
            .expect("the encode");

        // The tower's rows enter the stream and each DeepStack feature a layer's output.
        assert_close(
            name,
            &out,
            &reference[name],
            features_bound + tolerance(stack_terms(&config, layers, tokens)),
        );
    }

    // A prompt whose placeholders do not hold its grids is refused before anything is
    // written.
    let layout = whole.layout(&batch, &[image_grid]);
    let out = zeros(Shape::from((tokens as usize, 16)), DType::F32);
    let refused = whole.encode(&ids, &column(&ids), &batch, &pixels[..1], &layout, &out);
    assert!(refused.is_err());
}
