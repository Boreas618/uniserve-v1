//! Reference vectors on the CPU device: the family's modules, loaded through the real loading
//! path from a tiny checkpoint, compute what the reference implementations computed from the
//! same weights in PyTorch (`generate_reference.py`), and the whole model computes the
//! same streams, caches and state whether it is prefilled at once or stepped.
//!
//! | Test | Reference | What is compared |
//! | --- | --- | --- |
//! | n-gram ids | `embed_ple.ngram_ids` | the table rows of every head, exactly |
//! | PLE layer | `embed_ple.ref_kernel` | the streams, the window of a verify step |
//! | gated delta layer | `gdn_moe_layer.ref_kernel` | streams, experts' output, combine logits, the window and the state slots |
//! | attention layer | `qsa_moe_layer.ref_kernel` | the same, the new keys and values, the compressed keys, the ring |
//! | read-out and head | `lm_head.ref_kernel` | the folded streams, the logits of shuffled rows |
//! | drafter | `mtp_draft.fuse`, then the layer and head references | the drafter's streams and logits |
//! | drafter over two rounds | the same, its step split in two forwards over the drafter's state | the streams, logits, caches and ring |
//! | block-scaled FP8 | `mtp_draft.block_quantize`, `dequantize` | the matrix decoded at load, exactly |
//! | indexer and attention functions | vLLM's `test_qsa_reference.py` | scores, the selected groups, the attended heads |
//! | prefill against steps | the model itself | streams, keys and values, index keys, ring, windows, state |
//!
//! ## The bounds
//!
//! The challenge's references compute in FP32 with the precision of serving at every stored
//! tensor (BF16 activations, E4M3 caches and index keys, NVFP4 experts and expert
//! activations), so a comparison with them is at reduced precision and is held to the bound
//! the challenge's `TASK.md` states for the output: relative L2 over the tensor within the
//! bound, and no row worse than five times it, a row's norm floored at a tenth of the mean
//! (`_common/checks.py:11-23`). Where this crate's caches are BF16 and the reference's E4M3
//! (the keys and values a step writes, the compressed keys), the difference is the E4M3
//! rounding of each element, at most one sixteenth of it, inside the 10% the challenge
//! states for new cache rows. No bound is chosen here.
//!
//! The comparisons in FP32 (vLLM's functions, and the model against itself on the FP32
//! fixture) evaluate the same formulas in different orders. A reduction of `n` terms evaluated
//! in FP32 differs from its exact value by at most `n · u · Σ|terms|` to first order,
//! `u = 2⁻²⁴`, so two evaluations differ by at most twice that, and the errors of consecutive
//! stages add to first order: the bound of an output is `2 · u · terms`, `terms` the terms of
//! every reduction on its path ([`model_terms`]), relative to the largest magnitude of the
//! compared tensor.
//!
//! Nothing here touches a GPU.

use std::collections::HashMap;

use models::qwen4_exp::causal_lm::{coordinates, ngram_context, token_columns};
use models::qwen4_exp::{Config, DecoderLayer, HyperConnection, NgramEmbedding, PleLayer};
use nn::family::{Family, TextModel, TextPlacement};
use nn::modules::Rotary;
use nn::quantization::Declaration;
use nn::{
    CpuWeights, DType, Device, KvCache, KvPlanes, Phase, RecurrentState, Shape, Split, Tensor,
    TokenBatch, VarBuilder, WeightLoader, functional,
};
use nn_testing::text::{self, WORKSPACE_BYTES};
use nn_testing::{Fixture, PAGE_TOKENS};

/// The family the fixtures' checkpoints declare.
const FAMILIES: [Family; 1] = [models::qwen4_exp::family::FAMILY];

/// The tokens a step of the checked deployment drafts, and the rows a request has in a step.
const DRAFTS: u32 = 3;
const STEP: usize = DRAFTS as usize + 1;

/// The checked deployment: one rank that drafts [`DRAFTS`] tokens a step with the checkpoint's
/// drafter. The fixture declares the release's vision tower and holds none of it, so the
/// deployment serves none.
const DEPLOYMENT: TextPlacement = TextPlacement {
    drafter: true,
    vision: false,
    draft_tokens: DRAFTS,
    ..TextPlacement::WHOLE
};

/// The module path of the language model.
const LANGUAGE: &str = "model.language_model";

/// The unit roundoff of FP32.
const UNIT_ROUNDOFF: f32 = f32::EPSILON / 2.0;

/// The reference fixture: a tiny NVFP4 export of the family under `model` and the
/// references' vectors in `reference.safetensors`.
fn reference() -> Fixture {
    nn_testing::fixture!("data/reference": [
        "model/config.json",
        "model/model.safetensors",
        "reference.safetensors",
    ])
}

/// The consistency fixture: a tiny FP32 checkpoint of the family under `model`.
fn consistency() -> Fixture {
    nn_testing::fixture!("data/consistency": ["model/config.json", "model/model.safetensors"])
}

/// The vector `name`, which the generator wrote.
fn vector(vectors: &HashMap<String, Tensor>, name: &str) -> Tensor {
    vectors
        .get(name)
        .unwrap_or_else(|| panic!("the fixture holds no {name}"))
        .clone()
}

/// The elements of an integer vector as U32.
fn u32s(tensor: &Tensor) -> Vec<u32> {
    tensor
        .to_dtype(DType::U32)
        .and_then(|tensor| tensor.flatten_all())
        .and_then(|tensor| tensor.to_vec1())
        .expect("an integer vector")
}

/// A CPU column of U32 `values`.
fn column(values: &[u32]) -> Tensor {
    Tensor::new(values, &Device::Cpu).expect("a column")
}

/// A CPU tensor of zeros.
fn zeros(shape: Shape, dtype: DType) -> Tensor {
    Tensor::zeros(shape, dtype, &Device::Cpu).expect("zeros")
}

/// Runs `check` with the configuration of `fixture`'s checkpoint and a builder of its root, its
/// weights loaded onto the CPU device as its export declares them.
fn with_fixture<T>(fixture: &Fixture, check: impl FnOnce(&Config, VarBuilder<'_, '_>) -> T) -> T {
    let checkpoint = fixture.checkpoint("model");
    let store = CpuWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    let config = Config::open(&checkpoint).expect("the configuration");
    loader
        .declare(&Declaration::read(&checkpoint).expect("the declaration"))
        .expect("the declaration agrees with the shards");

    check(&config, loader.root())
}

/// The relative L2 difference of `actual` from `expected` over the whole tensor and of its
/// worst row, a row's norm floored at a tenth of the mean row norm
/// (`qwen-kernel-challenge:packages/_common/checks.py:11-23`).
fn differences(actual: &Tensor, expected: &Tensor) -> (f64, f64) {
    assert_eq!(actual.dims(), expected.dims(), "the shapes differ");
    let rows = expected.dims()[0];
    let flat = |tensor: &Tensor| -> Vec<Vec<f32>> {
        tensor
            .to_dtype(DType::F32)
            .and_then(|tensor| tensor.reshape((rows, tensor.elem_count() / rows)))
            .and_then(|tensor| tensor.to_vec2())
            .expect("a tensor's rows")
    };
    let (actual, expected) = (flat(actual), flat(expected));
    let norm = |row: &[f64]| row.iter().map(|x| x * x).sum::<f64>().sqrt();

    let (mut errors, mut norms) = (Vec::new(), Vec::new());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert!(
            actual.iter().all(|x| x.is_finite()),
            "a value is not finite"
        );
        let difference: Vec<f64> = actual
            .iter()
            .zip(expected)
            .map(|(a, e)| f64::from(*a) - f64::from(*e))
            .collect();
        let reference: Vec<f64> = expected.iter().map(|e| f64::from(*e)).collect();
        errors.push(norm(&difference));
        norms.push(norm(&reference));
    }

    let total = norm(&errors) / norm(&norms).max(1e-12);
    let floor = (0.1 * norms.iter().sum::<f64>() / norms.len() as f64).max(1e-12);
    let worst = errors
        .iter()
        .zip(&norms)
        .map(|(error, norm)| error / norm.max(floor))
        .fold(0.0, f64::max);
    (total, worst)
}

/// Asserts that `actual` is `expected` within the challenge's bound `bound` (module
/// documentation), and prints the difference found.
fn assert_within(what: &str, actual: &Tensor, expected: &Tensor, bound: f64) {
    let (total, worst) = differences(actual, expected);
    println!("{what}: relative L2 {total:.3e}, worst row {worst:.3e}, bound {bound}");

    assert!(
        total <= bound && worst <= 5.0 * bound,
        "{what}: relative L2 {total:.3e} and worst row {worst:.3e} against {bound}"
    );
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

    let scale = expected
        .iter()
        .filter(|x| x.is_finite())
        .fold(0.0_f32, |scale, x| scale.max(x.abs()));
    let worst = actual
        .iter()
        .zip(&expected)
        .map(|(actual, expected)| {
            if expected.is_finite() {
                (actual - expected).abs()
            } else if actual == expected {
                0.0
            } else {
                f32::INFINITY
            }
        })
        .fold(0.0_f32, f32::max);
    println!(
        "{what}: differs by {:.2e} of its magnitude, bound {tolerance:.2e}",
        worst / scale
    );
    assert!(
        worst <= tolerance * scale,
        "{what}: differs by {worst:.3e} of a magnitude of {scale:.3e}, bound {tolerance:.2e}"
    );
}

/// The rows `rows` of `tensor`, its first axis.
fn rows_of(tensor: &Tensor, rows: &[u32]) -> Tensor {
    tensor
        .index_select(&column(rows), 0)
        .expect("the rows of a tensor")
}

/// A batch whose rows are `requests` requests of [`STEP`] tokens each, the last of them the
/// request's token `seen − 1`: the shape a verify step's recurrent layers read.
fn request_batch(requests: usize, seen: u32, ngram: Option<(Tensor, Tensor)>) -> TokenBatch {
    let tokens = requests * STEP;
    let positions: Vec<i32> = (0..tokens)
        .map(|token| (seen as usize - STEP + token % STEP) as i32)
        .collect();
    let offsets: Vec<u32> = (0..=requests).map(|row| (row * STEP) as u32).collect();

    text::batch(
        Phase::Decode,
        [
            Tensor::new(positions.as_slice(), &Device::Cpu).expect("positions"),
            column(&vec![0; tokens]),
            column(&vec![seen; requests]),
            column(&offsets),
            column(&offsets[1..].iter().map(|end| end - 1).collect::<Vec<_>>()),
        ],
        ngram,
        PAGE_TOKENS,
        PAGE_TOKENS * 2,
    )
}

/// The recurrent state of a step of `requests` requests of [`STEP`] tokens that have seen
/// `seen` tokens each, in the slots `state_indices` (`[requests, STEP]`) with `num_accepted`
/// standing tokens, over the live buffers `live` of its layers.
fn request_state(
    seen: &[u32],
    state_indices: &Tensor,
    num_accepted: &Tensor,
    live: Vec<Vec<Tensor>>,
) -> RecurrentState {
    let requests = seen.len();
    let offsets: Vec<u32> = (0..=requests).map(|row| (row * STEP) as u32).collect();
    let indices = state_indices.to_dtype(DType::U32).expect("slots");
    let first = indices
        .narrow(1, 0, 1)
        .and_then(|first| first.flatten_all())
        .expect("each request's first slot");
    let checkpoint = live
        .iter()
        .map(|fields| {
            fields
                .iter()
                .map(|field| field.zeros_like().unwrap())
                .collect()
        })
        .collect::<Vec<Vec<Tensor>>>();

    text::state(
        [
            column(&offsets),
            column(seen),
            first,
            column(&vec![0; requests]),
        ],
        &live,
        &checkpoint,
        Some((indices, num_accepted.to_dtype(DType::U32).expect("counts"))),
    )
}

/// The table rows every head of the n-gram embedding reads are the reference's, exactly:
/// the wrapping products, the end-of-text rule, the non-negative remainder and the offsets.
#[test]
fn the_ngram_ids_are_the_references() {
    let fixture = reference();
    let vectors = fixture.tensors("reference.safetensors");

    with_fixture(&fixture, |config, vb| {
        let embedding = NgramEmbedding::load(
            config,
            vb.pp(LANGUAGE).pp("layers.1.ple.ple_embedding"),
            Split::WHOLE,
        )
        .expect("the embedding");

        for case in 0..3 {
            let name = |tensor: &str| format!("ple.{case}.{tensor}");
            let ids = vector(&vectors, &name("input_ids"));
            let before = vector(&vectors, &name("ngram_context"));
            let requests = before.dims()[0];

            let batch = request_batch(
                requests,
                54,
                Some((
                    ids.to_dtype(DType::U32).unwrap(),
                    before.to_dtype(DType::U32).unwrap(),
                )),
            );
            let slots = Tensor::zeros((requests, STEP), DType::U32, &Device::Cpu).unwrap();
            let state = request_state(
                &vec![54; requests],
                &slots,
                &column(&vec![1; requests]),
                Vec::new(),
            );
            let columns = token_columns(&batch, &state).expect("the token columns");
            let context = ngram_context(&batch, &columns).expect("the n-gram context");

            let rows: Vec<Vec<i64>> = embedding.ids(&context).unwrap().to_vec2().unwrap();
            let expected: Vec<Vec<i64>> = vector(&vectors, &name("ids")).to_vec2().unwrap();
            assert_eq!(rows, expected, "case {case}");
        }
    });
}

/// The PLE layer over a verify step: the streams it returns and the window it leaves, for
/// every count of standing tokens the reference's cases state.
#[test]
fn the_ple_layer_is_the_references() {
    let fixture = reference();
    let vectors = fixture.tensors("reference.safetensors");

    with_fixture(&fixture, |config, vb| {
        let model = vb.pp(LANGUAGE);
        let layer =
            PleLayer::load(config, model.pp("layers.1.ple"), Split::WHOLE).expect("the layer");
        let embed = nn::modules::Embedding::load(
            &nn::modules::EmbeddingConfig {
                vocabulary: config.text.vocab_size,
                hidden: config.text.hidden_size,
                split: Split::WHOLE,
            },
            model.pp("embed_tokens"),
        )
        .expect("the embedding");
        let streams = config.hyper_connection.streams as usize;

        for case in 0..3 {
            let name = |tensor: &str| format!("ple.{case}.{tensor}");
            let ids = vector(&vectors, &name("input_ids"))
                .to_dtype(DType::U32)
                .unwrap();
            let before = vector(&vectors, &name("ngram_context"));
            let slots = u32s(&vector(&vectors, &name("state_indices")));
            let requests = slots.len();

            // The embedding opens every stream: a copy of table rows, which is exact.
            let embedded = embed.forward(&ids).expect("the embeddings");
            let opened = Tensor::cat(&vec![&embedded; streams], 1).unwrap();
            assert_close(
                "streams",
                &opened,
                &vector(&vectors, &name("out.streams")),
                0.0,
            );

            let folded = HyperConnection::combine(
                &opened,
                &vector(&vectors, &name("block_output")),
                Some(&vector(&vectors, &name("injection"))),
                streams,
            )
            .expect("the folded streams");

            let batch = request_batch(
                requests,
                54,
                Some((ids, before.to_dtype(DType::U32).unwrap())),
            );
            // A request's window is in its first slot; its further slots are not the PLE's.
            let indices: Vec<u32> = slots.iter().flat_map(|slot| [*slot; STEP]).collect();
            let window = vector(&vectors, &name("conv_state")).copy().unwrap();
            let state = request_state(
                &vec![54; requests],
                &column(&indices).reshape((requests, STEP)).unwrap(),
                &vector(&vectors, &name("num_accepted")),
                vec![vec![window.clone()]],
            );
            let columns = token_columns(&batch, &state).expect("the token columns");
            let context = ngram_context(&batch, &columns).expect("the n-gram context");

            let added = layer
                .forward(&folded, &context, &batch, &state, 0)
                .expect("the PLE layer");
            let sum = (folded.to_dtype(DType::F32).unwrap() + added.to_dtype(DType::F32).unwrap())
                .unwrap()
                .to_dtype(folded.dtype())
                .unwrap();

            // `embed_ple/TASK.md`: hidden 2%, the window of the written slots 2%.
            assert_within(
                "PLE hidden",
                &sum,
                &vector(&vectors, &name("out.hidden")),
                0.02,
            );
            assert_within(
                "PLE window",
                &rows_of(&window, &slots),
                &rows_of(&vector(&vectors, &name("out.conv_state")), &slots),
                0.02,
            );
        }
    });
}

/// One gated delta network layer with its NVFP4 experts over a verify step, from every
/// standing count the reference's cases state: the streams, the experts' output, the combine
/// logits, the convolution window and the state slots.
#[test]
fn the_gated_delta_layer_is_the_references() {
    let fixture = reference();
    let vectors = fixture.tensors("reference.safetensors");

    with_fixture(&fixture, |config, vb| {
        let model = vb.pp(LANGUAGE);
        let layer =
            DecoderLayer::load(config, model.pp("layers.0"), true, false, &DEPLOYMENT, None)
                .expect("the layer");
        let rotary = Rotary::load(&config.text.rope, model.pp("rotary_emb")).expect("the rotary");

        for case in 0..3 {
            let name = |tensor: &str| format!("gdn.{case}.{tensor}");
            let indices = vector(&vectors, &name("state_indices"));
            let requests = indices.dims()[0];
            let window = vector(&vectors, &name("conv_state")).copy().unwrap();
            let memory = vector(&vectors, &name("ssm_state")).copy().unwrap();

            let batch = request_batch(requests, 104, None);
            let state = request_state(
                &vec![104; requests],
                &indices,
                &vector(&vectors, &name("num_accepted")),
                vec![vec![window.clone(), memory.clone()]],
            );
            let coordinates = coordinates(&batch).unwrap();
            let (cos, sin) = rotary.forward(&coordinates).unwrap();
            let columns = token_columns(&batch, &state).unwrap();

            let (streams, output, logits) = layer
                .forward(
                    &vector(&vectors, &name("hidden")),
                    Some((
                        &vector(&vectors, &name("block_output")),
                        &vector(&vectors, &name("injection")),
                    )),
                    (&cos, &sin),
                    &rotary,
                    &coordinates,
                    &columns,
                    None,
                    &batch,
                    None,
                    &state,
                    None,
                    0,
                    (0, 0),
                )
                .expect("the layer's forward");

            // `gdn_moe_layer/TASK.md`: hidden 2%, injection 8%, block output 20%, the
            // window and the state of the written slots 5%.
            let expected = |tensor: &str| vector(&vectors, &name(&format!("out.{tensor}")));
            assert_within("GDN hidden", &streams, &expected("hidden"), 0.02);
            assert_within("GDN injection", &logits, &expected("injection"), 0.08);
            assert_within("GDN block output", &output, &expected("block_output"), 0.2);

            let slots = u32s(&indices);
            assert_within(
                "GDN state",
                &rows_of(&memory, &slots),
                &rows_of(&expected("ssm_state"), &slots),
                0.05,
            );
            assert_within(
                "GDN window",
                &rows_of(&window, &slots),
                &rows_of(&expected("conv_state"), &slots),
                0.05,
            );
        }
    });
}

/// A verify step of an attention layer as the reference's inputs state it: every token its
/// own attention row over its request's pages, the requests' rings in the slots after the
/// first.
struct AttentionStep {
    batch: TokenBatch,
    cache: KvCache,
    state: RecurrentState,
    /// The cache slot of each token, the plane rows the step writes.
    slots: Vec<u32>,
    /// The index-key rows of the groups the step completes.
    completed: Vec<u32>,
    /// The ring rows the step writes.
    ring_rows: Vec<u32>,
}

/// The step of the fixture vectors under `prefix`.
fn attention_step(vectors: &HashMap<String, Tensor>, prefix: &str) -> AttentionStep {
    let name = |tensor: &str| format!("{prefix}.{tensor}");
    let start = u32s(&vector(vectors, &name("start")));
    let table: Vec<Vec<u32>> = vector(vectors, &name("page_table"))
        .to_dtype(DType::U32)
        .unwrap()
        .to_vec2()
        .unwrap();
    let (requests, tokens) = (start.len(), start.len() * STEP);
    let key = vector(vectors, &name("key")).copy().unwrap();
    let ring_keys = vector(vectors, &name("ring_keys")).copy().unwrap();
    let (pages, capacity) = (key.dims()[0], ring_keys.dims()[1] as u32);
    let groups = PAGE_TOKENS / 4;

    let positions: Vec<u32> = start
        .iter()
        .flat_map(|first| (0..STEP as u32).map(move |row| first + row))
        .collect();
    let request_of = |token: usize| token / STEP;
    let slots: Vec<u32> = positions
        .iter()
        .enumerate()
        .map(|(token, position)| {
            table[request_of(token)][(position / PAGE_TOKENS) as usize] * PAGE_TOKENS
                + position % PAGE_TOKENS
        })
        .collect();
    let completed = positions
        .iter()
        .enumerate()
        .filter(|(_, position)| (**position + 1).is_multiple_of(4))
        .map(|(token, position)| {
            let group = position / 4;
            table[request_of(token)][(group / groups) as usize] * groups + group % groups
        })
        .collect();
    let ring_rows = positions
        .iter()
        .enumerate()
        .map(|(token, position)| (request_of(token) as u32 + 1) * capacity + position % capacity)
        .collect();

    let signed: Vec<i32> = positions.iter().map(|position| *position as i32).collect();
    let offsets: Vec<u32> = (0..=tokens as u32).collect();
    let batch = text::batch(
        Phase::Decode,
        [
            Tensor::new(signed.as_slice(), &Device::Cpu).unwrap(),
            column(&slots),
            column(
                &positions
                    .iter()
                    .map(|position| position + 1)
                    .collect::<Vec<_>>(),
            ),
            column(&offsets),
            column(&offsets[..tokens]),
        ],
        None,
        PAGE_TOKENS,
        PAGE_TOKENS * table[0].len() as u32,
    );

    let row_tables: Vec<u32> = (0..tokens)
        .flat_map(|token| table[request_of(token)].clone())
        .collect();
    let flags = Tensor::ones((pages, 1, 1, 1), DType::U8, &Device::Cpu).unwrap();
    let cache = KvCache {
        block_tables: column(&row_tables)
            .reshape((tokens, table[0].len()))
            .unwrap(),
        page_size: PAGE_TOKENS,
        layers: vec![KvPlanes {
            key,
            value: vector(vectors, &name("value")).copy().unwrap(),
            key_initialized: flags.clone(),
            value_initialized: flags,
            key_scale: None,
            value_scale: None,
            index_keys: Some(vector(vectors, &name("index_keys")).copy().unwrap()),
        }],
        workspace: zeros(Shape::from(1 << 10), DType::U8),
    };

    let request_slots: Vec<u32> = (1..=requests as u32).collect();
    let seen: Vec<u32> = start.iter().map(|first| first + STEP as u32).collect();
    let state_offsets: Vec<u32> = (0..=requests).map(|row| (row * STEP) as u32).collect();
    let live = vec![vec![
        ring_keys,
        vector(vectors, &name("ring_positions")).copy().unwrap(),
    ]];
    let checkpoint: Vec<Vec<Tensor>> = live
        .iter()
        .map(|fields| {
            fields
                .iter()
                .map(|field| field.zeros_like().unwrap())
                .collect()
        })
        .collect();
    let state = text::state(
        [
            column(&state_offsets),
            column(&seen),
            column(&request_slots),
            column(&vec![0; requests]),
        ],
        &live,
        &checkpoint,
        None,
    );

    AttentionStep {
        batch,
        cache,
        state,
        slots,
        completed,
        ring_rows,
    }
}

/// Asserts the caches and the ring a step of an attention layer left against the reference's
/// under `prefix`: the keys and values of the step's tokens, the compressed keys of the groups
/// it completed and the raw keys of its rows.
fn assert_attention_caches(vectors: &HashMap<String, Tensor>, prefix: &str, step: &AttentionStep) {
    let expected = |tensor: &str| vector(vectors, &format!("{prefix}.out.{tensor}"));
    let rows = |plane: &Tensor, rows: &[u32]| {
        let (leading, width) = (
            plane.dims()[0] * plane.dims()[1],
            plane.dims()[2..].iter().product::<usize>(),
        );
        rows_of(&plane.reshape((leading, width)).unwrap(), rows)
    };
    let planes = &step.cache.layers[0];

    // `qsa_moe_layer/TASK.md`: new K/V rows 10%. The reference's caches are E4M3 where these
    // are BF16, and so are its compressed keys (module documentation).
    for (what, plane, reference) in [
        ("new keys", &planes.key, expected("key")),
        ("new values", &planes.value, expected("value")),
    ] {
        assert_within(
            what,
            &rows(plane, &step.slots),
            &rows(&reference, &step.slots),
            0.1,
        );
    }
    if !step.completed.is_empty() {
        let index_keys = planes.index_keys.as_ref().expect("index keys");
        assert_within(
            "compressed keys",
            &rows(index_keys, &step.completed),
            &rows(&expected("index_keys"), &step.completed),
            0.1,
        );
    }

    // The ring's raw keys are activations: the bound of the streams.
    assert_within(
        "ring",
        &rows(&step.state.live[0][0], &step.ring_rows),
        &rows(&expected("ring_keys"), &step.ring_rows),
        0.02,
    );
}

/// One indexed attention layer with its NVFP4 experts over a verify step: where every visible
/// group is selected (the first two cases), and where the indexer chooses among more groups
/// than a row may keep (the third).
#[test]
fn the_indexed_attention_layer_is_the_references() {
    let fixture = reference();
    let vectors = fixture.tensors("reference.safetensors");

    with_fixture(&fixture, |config, vb| {
        let model = vb.pp(LANGUAGE);
        let layer = DecoderLayer::load(
            config,
            model.pp("layers.1"),
            false,
            false,
            &DEPLOYMENT,
            None,
        )
        .expect("the layer");
        let rotary = Rotary::load(&config.text.rope, model.pp("rotary_emb")).expect("the rotary");

        for case in 0..3 {
            let prefix = format!("qsa.{case}");
            let name = |tensor: &str| format!("{prefix}.{tensor}");
            let step = attention_step(&vectors, &prefix);
            let coordinates = coordinates(&step.batch).unwrap();
            let (cos, sin) = rotary.forward(&coordinates).unwrap();
            let columns = token_columns(&step.batch, &step.state).unwrap();

            let (streams, output, logits) = layer
                .forward(
                    &vector(&vectors, &name("hidden")),
                    Some((
                        &vector(&vectors, &name("block_output")),
                        &vector(&vectors, &name("injection")),
                    )),
                    (&cos, &sin),
                    &rotary,
                    &coordinates,
                    &columns,
                    None,
                    &step.batch,
                    Some(&step.cache),
                    &step.state,
                    None,
                    0,
                    (0, 0),
                )
                .expect("the layer's forward");

            // `qsa_moe_layer/TASK.md`: hidden 2%, injection 12%, block output 25%.
            let expected = |tensor: &str| vector(&vectors, &name(&format!("out.{tensor}")));
            assert_within("QSA hidden", &streams, &expected("hidden"), 0.02);
            assert_within("QSA injection", &logits, &expected("injection"), 0.12);
            assert_within("QSA block output", &output, &expected("block_output"), 0.25);
            assert_attention_caches(&vectors, &prefix, &step);
        }
    });
}

/// The whole model of the reference fixture `fixture`, loaded through its family with its
/// drafter.
fn reference_model(fixture: &Fixture) -> std::sync::Arc<dyn TextModel> {
    let checkpoint = fixture.checkpoint("model");
    let store = CpuWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);

    text::load(&FAMILIES, &checkpoint, &mut loader, &DEPLOYMENT).expect("the load")
}

/// The fixture declares the release's vision tower and holds neither its processor files nor
/// its weights: a deployment that serves the tower is refused, naming the file, and one that
/// does not loads the model without a tower.
#[test]
fn a_declared_tower_the_checkpoint_does_not_hold_is_refused_where_it_is_served() {
    let fixture = reference();
    let checkpoint = fixture.checkpoint("model");
    let store = CpuWeights::new();
    let load = |vision| {
        let mut loader = WeightLoader::new(&checkpoint, &store, None);
        text::load(
            &FAMILIES,
            &checkpoint,
            &mut loader,
            &TextPlacement {
                vision,
                ..DEPLOYMENT
            },
        )
    };

    let error = load(true)
        .err()
        .expect("a served tower the checkpoint lacks")
        .to_string();
    assert!(error.contains("preprocessor_config.json"), "{error}");
    assert!(load(false).expect("the load").vision().is_none());
}

/// The read-out and the head: the last layer's output folded into the streams, and the logits
/// of shuffled rows through the model's `compute_logits`.
#[test]
fn the_read_out_and_the_head_are_the_references() {
    let fixture = reference();
    let vectors = fixture.tensors("reference.safetensors");
    let model = reference_model(&fixture);
    assert_eq!((model.hidden(), model.stream_width()), (128, 512));

    let folded = HyperConnection::combine(
        &vector(&vectors, "head.hidden"),
        &vector(&vectors, "head.block_output"),
        Some(&vector(&vectors, "head.injection")),
        4,
    )
    .expect("the folded streams");
    // `lm_head/TASK.md`: hidden 2%, logits 2%.
    assert_within(
        "head hidden",
        &folded,
        &vector(&vectors, "head.out.hidden"),
        0.02,
    );

    let rows = rows_of(&folded, &u32s(&vector(&vectors, "head.logits_indices")));
    let logits = model.compute_logits(&rows).expect("the logits");
    assert_eq!(logits.dtype(), DType::F32);
    assert_within(
        "head logits",
        &logits,
        &vector(&vectors, "head.out.logits"),
        0.02,
    );
}

/// The drafter: the input fusion, its attention layer over a cache and a ring of its own, and
/// its logits through its own read-out and the model's head.
#[test]
fn the_drafter_is_the_references() {
    let fixture = reference();
    let vectors = fixture.tensors("reference.safetensors");
    let model = reference_model(&fixture);
    let drafter = model.drafter().expect("the drafter");
    assert_eq!(drafter.state_layers().len(), 1);

    let step = attention_step(&vectors, "mtp");
    let tokens = vector(&vectors, "mtp.tokens").to_dtype(DType::U32).unwrap();
    let embedded = model.embed_input_ids(&tokens).expect("the embeddings");

    // A forward lent no ring is refused: the layer keeps one per request.
    assert!(
        drafter
            .forward(
                &step.batch,
                &step.cache,
                None,
                &embedded,
                &vector(&vectors, "mtp.streams")
            )
            .is_err()
    );
    let drafted = drafter
        .forward(
            &step.batch,
            &step.cache,
            Some(&step.state),
            &embedded,
            &vector(&vectors, "mtp.streams"),
        )
        .expect("the drafter's forward");

    // `mtp_draft/TASK.md`: hidden 5%, which the logits read.
    assert_within(
        "draft streams",
        &drafted,
        &vector(&vectors, "mtp.out.streams"),
        0.05,
    );
    let logits = model
        .compute_logits(&drafter.logits_rows(&drafted).expect("the read-out"))
        .expect("the draft logits");
    assert_within(
        "draft logits",
        &logits,
        &vector(&vectors, "mtp.out.logits"),
        0.05,
    );
    assert_attention_caches(&vectors, "mtp", &step);
}

/// The drafter's state carries its step across two forwards: the drafter over the first half
/// of each request's tokens, then over the second half, which pools a group whose first tokens
/// the first round left in the ring, computes the reference's streams, logits, caches and
/// ring. The state is lent as execution lends a drafter's (`nn::Drafter::state_layers`): the live
/// ring alone, which no checkpoint keeps.
#[test]
fn the_drafter_over_two_rounds_is_the_references() {
    let fixture = reference();
    let vectors = fixture.tensors("reference.safetensors");
    let model = reference_model(&fixture);
    let drafter = model.drafter().expect("the drafter");

    let step = attention_step(&vectors, "mtp");
    let tokens = vector(&vectors, "mtp.tokens").to_dtype(DType::U32).unwrap();
    let embedded = model.embed_input_ids(&tokens).expect("the embeddings");
    let streams = vector(&vectors, "mtp.streams");
    let start = u32s(&vector(&vectors, "mtp.start"));
    let requests = start.len();
    let half = STEP / 2;

    // Request 0 completes group 5 (positions 20 to 23) in the second round.
    assert_eq!(start[0] % 4, 0);

    let (mut drafted, mut order) = (Vec::new(), Vec::new());
    for round in [0..half, half..STEP] {
        // The round's tokens of every request, each its own attention row as in the step, and
        // one state row a request.
        let picked: Vec<u32> = (0..requests)
            .flat_map(|request| round.clone().map(move |row| (request * STEP + row) as u32))
            .collect();
        let pick = |tensor: &Tensor| rows_of(tensor, &picked);
        let offsets: Vec<u32> = (0..=picked.len() as u32).collect();

        let batch = text::batch(
            Phase::Decode,
            [
                pick(&step.batch.positions),
                pick(&step.batch.slot_mapping),
                pick(&step.batch.seq_lens),
                column(&offsets),
                column(&offsets[..picked.len()]),
            ],
            None,
            PAGE_TOKENS,
            step.batch.max_kv_len,
        );
        let cache = KvCache {
            block_tables: pick(&step.cache.block_tables),
            ..step.cache.clone()
        };

        let state_offsets: Vec<u32> = (0..=requests)
            .map(|request| (request * round.len()) as u32)
            .collect();
        let seen: Vec<u32> = start.iter().map(|first| first + round.end as u32).collect();
        let slots: Vec<u32> = (1..=requests as u32).collect();
        let state = text::state(
            [
                column(&state_offsets),
                column(&seen),
                column(&slots),
                column(&vec![0; requests]),
            ],
            &step.state.live,
            &[],
            None,
        );

        drafted.push(
            drafter
                .forward(
                    &batch,
                    &cache,
                    Some(&state),
                    &pick(&embedded),
                    &pick(&streams),
                )
                .expect("a round of the drafter"),
        );
        order.extend(picked);
    }
    let drafted = Tensor::cat(&drafted, 0).unwrap();

    // `mtp_draft/TASK.md`: hidden 5%, which the logits read.
    assert_within(
        "draft streams over two rounds",
        &drafted,
        &rows_of(&vector(&vectors, "mtp.out.streams"), &order),
        0.05,
    );
    let logits = model
        .compute_logits(&drafter.logits_rows(&drafted).expect("the read-out"))
        .expect("the draft logits");
    assert_within(
        "draft logits over two rounds",
        &logits,
        &rows_of(&vector(&vectors, "mtp.out.logits"), &order),
        0.05,
    );
    assert_attention_caches(&vectors, "mtp", &step);
}

/// The bound two FP32 evaluations of one formula differ by when their reductions sum `terms`
/// terms in all (module documentation).
fn tolerance(terms: u32) -> f32 {
    2.0 * UNIT_ROUNDOFF * terms as f32
}

/// The indexer's scores, its selection and the attention over a selection against vLLM's own
/// reference functions, in FP32.
#[test]
fn the_indexed_functions_are_vllms_references() {
    let fixture = reference();
    let vectors = fixture.tensors("reference.safetensors");
    let named = |tensor: &str| vector(&vectors, &format!("vllm.{tensor}"));
    let table = named("page_table").to_dtype(DType::U32).unwrap();
    let rows = named("rows").to_dtype(DType::U32).unwrap();
    let plane = |tensor: Tensor, index_keys: Option<Tensor>| KvPlanes {
        key: tensor.clone(),
        value: tensor,
        key_initialized: zeros(Shape::from(1), DType::U8),
        value_initialized: zeros(Shape::from(1), DType::U8),
        key_scale: None,
        value_scale: None,
        index_keys,
    };

    // Scores: two heads of sixteen elements against each visible key.
    let index_keys = named("index_keys");
    let keys = functional::Plane {
        records: &index_keys,
        table: &table,
        format: functional::Format::Bf16,
        layout: functional::Layout::Page,
    };
    let visible = named("visible").to_dtype(DType::U32).unwrap();
    let scores = functional::index_scores(
        &named("index_query"),
        keys,
        &rows,
        &visible,
        functional::Scoring::default(),
    )
    .expect("the scores");
    assert_close(
        "index scores",
        &scores,
        &named("out.scores"),
        tolerance(2 * 16),
    );

    // The best groups of each row, as a set: the order among them is free.
    let (_, best) = functional::top_k(&scores, 8).unwrap();
    let best: Vec<Vec<i32>> = best.view(DType::I32).unwrap().to_vec2().unwrap();
    let expected: Vec<Vec<i32>> = named("out.best").to_vec2().unwrap();
    for (row, (best, expected)) in best.iter().zip(&expected).enumerate() {
        let mut kept: Vec<i32> = expected
            .iter()
            .copied()
            .filter(|group| *group >= 0)
            .collect();
        let mut chosen = best[..kept.len()].to_vec();
        kept.sort_unstable();
        chosen.sort_unstable();
        assert_eq!(chosen, kept, "row {row}");
    }

    // Attention over a selection: a softmax over at most eleven scores of eight terms each
    // and the weighted sum of the values.
    let mut attention_plane = plane(named("key"), None);
    attention_plane.value = named("value");
    let attention_cache = KvCache {
        block_tables: table,
        page_size: 16,
        layers: vec![attention_plane],
        workspace: zeros(Shape::from(1), DType::U8),
    };
    let attended = functional::indexed_attention(
        &named("query"),
        &attention_cache,
        0,
        &rows,
        &named("selection"),
        (8.0_f32).sqrt().recip(),
    )
    .expect("the attention");
    assert_close(
        "indexed attention",
        &attended,
        &named("out.attended"),
        tolerance(8 + 3 * 11),
    );
}

// ---- The model against itself: a prefill and steps -------------------------------------------

/// The terms the reductions of one forward of the consistency fixture's model sum on the path
/// of a token that reads `keys` earlier tokens.
fn model_terms(config: &Config, keys: u32) -> u32 {
    let text = &config.text;
    let (hidden, streams, rank) = (
        text.hidden_size,
        config.hyper_connection.streams,
        config.hyper_connection.rank,
    );
    let network = text.gated_delta_net;

    // One read of the streams: the combine, the norm of each stream, the down and up
    // products and the gated mean.
    let read = 1 + hidden + streams * hidden + rank + streams;
    // The projections of the row, the convolution's taps, the normalization of each query
    // and key, the state's decay and update by every earlier token, its read, the output
    // norm and the output projection.
    let linear = hidden
        + network.conv_width
        + 2 * network.key_dim
        + keys * 2 * network.key_dim
        + network.key_dim
        + network.value_dim
        + network.value_heads * network.value_dim;
    // The projections, the norm of each head and its rotation, the indexer's projection,
    // norm, pooling and scores, the scores over the selected keys, the softmax and the
    // weighted sum, the gate and the output projection.
    let indexer = config.indexer;
    let attention = hidden
        + text.head_dim
        + 2
        + hidden
        + indexer.head_dim
        + indexer.compress_ratio
        + indexer.heads * indexer.head_dim
        + text.head_dim
        + 2 * keys.min(indexer.selection())
        + 1
        + text.num_attention_heads * text.head_dim;
    // The router, its softmax and renormalization, each expert's two products, the weighted
    // sum of the routes, and the shared expert with its gate.
    let moe = &config.moe;
    let experts = hidden
        + moe.num_experts
        + moe.num_experts_per_tok
        + hidden
        + moe.moe_intermediate_size
        + moe.num_experts_per_tok
        + 2 * hidden
        + moe.moe_intermediate_size
        + 1;
    // The n-gram rows' scale, the two projections, three norms, the agreement of a key with
    // a stream and the convolution's taps.
    let ple = 1 + config.ngram.embed_dim + 3 * hidden + hidden + config.ngram.conv_kernel_size + 1;

    let layers = text.linear_layers.len() as u32;
    let linear_layers = text.linear_layers.iter().filter(|linear| **linear).count() as u32;
    layers * (2 * read + experts)
        + linear_layers * linear
        + (layers - linear_layers) * attention
        + config.ngram.layers.len() as u32 * ple
        + read
}

/// One request of the consistency checks: its tokens, the pages its row fills in order, and
/// the first of the [`STEP`] slots it owns.
struct Request {
    ids: Vec<u32>,
    pages: Vec<u32>,
    slot: u32,
}

/// The caches and the state buffers one run of the consistency checks advances.
struct Memory {
    cache: KvCache,
    live: Vec<Vec<Tensor>>,
    checkpoint: Vec<Vec<Tensor>>,
}

/// Pages of the consistency checks' cache group and request slots of their state.
const PAGES: usize = 4;
const SLOTS: usize = 2 * STEP;

/// The vocabulary's end-of-text token in the consistency fixture.
const EOS: u32 = 60;

impl Memory {
    /// Zeroed memory for `model`.
    fn new(model: &dyn TextModel) -> Self {
        let buffers = || {
            RecurrentState::buffers(
                &model.state_layers(),
                |_| SLOTS,
                |shape, dtype| Ok(zeros(shape, dtype)),
            )
            .expect("the state buffers")
        };

        Self {
            cache: KvCache::allocate(
                model.cache(),
                model.dtype(),
                PAGES,
                PAGE_TOKENS,
                zeros(Shape::from((1, 1)), DType::U32),
                zeros(Shape::from(WORKSPACE_BYTES), DType::U8),
                |shape, dtype| Ok(zeros(shape, dtype)),
            )
            .expect("a cache group"),
            live: buffers(),
            checkpoint: buffers(),
        }
    }
}

/// One forward of `model` over `parts`, one per request in the batch: the request, the first
/// position it presents, the tokens it presents there (its own, or drafts) and the standing
/// tokens of its previous step. A prefill presents each request as one row; a decode-phase
/// forward presents every token as an attention row of its own while one state row covers a
/// request's tokens.
fn forward(
    model: &dyn TextModel,
    phase: Phase,
    parts: &[(&Request, usize, Vec<u32>, u32)],
    memory: &Memory,
) -> Tensor {
    let per_token = phase == Phase::Decode;
    let (mut positions, mut slots, mut ids) = (Vec::new(), Vec::new(), Vec::new());
    let (mut seq_lens, mut offsets, mut tables, mut context) =
        (Vec::new(), vec![0_u32], Vec::new(), Vec::new());
    let (mut state_offsets, mut state_seen, mut state_slots) =
        (vec![0_u32], Vec::new(), Vec::new());
    let (mut indices, mut accepted) = (Vec::new(), Vec::new());

    for (request, start, fed, standing) in parts {
        // What the request has presented up to each fed token: its own tokens before the
        // step, then the step's.
        let history: Vec<u32> = request.ids[..*start].iter().chain(fed).copied().collect();
        let before = |token: usize, back: usize| {
            if token >= back {
                history[token - back]
            } else {
                EOS
            }
        };

        for (row, id) in fed.iter().enumerate() {
            let position = (start + row) as u32;
            positions.push(position as i32);
            slots.push(
                request.pages[(position / PAGE_TOKENS) as usize] * PAGE_TOKENS
                    + position % PAGE_TOKENS,
            );
            ids.push(*id);

            if per_token {
                seq_lens.push(position + 1);
                offsets.push(offsets[offsets.len() - 1] + 1);
                tables.extend_from_slice(&request.pages);
                context.extend([before(start + row, 2), before(start + row, 1)]);
            }
        }
        if !per_token {
            seq_lens.push((start + fed.len()) as u32);
            offsets.push(offsets[offsets.len() - 1] + fed.len() as u32);
            tables.extend_from_slice(&request.pages);
            context.extend([before(*start, 2), before(*start, 1)]);
        }

        state_offsets.push(state_offsets[state_offsets.len() - 1] + fed.len() as u32);
        state_seen.push((start + fed.len()) as u32);
        state_slots.push(request.slot);
        indices.extend((0..STEP as u32).map(|row| request.slot + row));
        accepted.push(*standing);
    }

    let rows = seq_lens.len();
    let row_pages = parts[0].0.pages.len();
    let last: Vec<u32> = offsets[1..].iter().map(|end| end - 1).collect();
    let batch = text::batch(
        phase,
        [
            Tensor::new(positions.as_slice(), &Device::Cpu).unwrap(),
            column(&slots),
            column(&seq_lens),
            column(&offsets),
            column(&last),
        ],
        Some((column(&ids), column(&context).reshape((rows, 2)).unwrap())),
        PAGE_TOKENS,
        PAGE_TOKENS * row_pages as u32,
    );
    let cache = KvCache {
        block_tables: column(&tables).reshape((rows, row_pages)).unwrap(),
        ..memory.cache.clone()
    };
    let state = text::state(
        [
            column(&state_offsets),
            column(&state_seen),
            column(&state_slots),
            column(&vec![0; parts.len()]),
        ],
        &memory.live,
        &memory.checkpoint,
        Some((
            column(&indices).reshape((parts.len(), STEP)).unwrap(),
            column(&accepted),
        )),
    );

    let embedded = model
        .embed_input_ids(&column(&ids))
        .expect("the embeddings");
    model
        .forward(&batch, &[cache], Some(&state), None, &embedded)
        .unwrap_or_else(|error| panic!("the {phase:?} forward: {error}"))
}

/// A prefill of whole rows computes what a shorter prefill followed by verify steps computes:
/// the streams of every standing token, the keys, values and index keys of the cache, the
/// ring, the convolution and PLE windows and the delta rule's state. Each step presents four
/// rows a request, of which the first few stand and the rest are wrong drafts whose rows,
/// slots, cache rows and compressed keys the next step overwrites or never reads.
#[test]
fn a_prefill_and_verify_steps_compute_what_one_prefill_does() {
    let fixture = consistency();
    let checkpoint = fixture.checkpoint("model");
    let store = CpuWeights::new();
    let mut loader = WeightLoader::new(&checkpoint, &store, None);
    let config = Config::open(&checkpoint).expect("the configuration");
    let placement = TextPlacement {
        drafter: false,
        ..DEPLOYMENT
    };
    let model = text::load(&FAMILIES, &checkpoint, &mut loader, &placement).expect("the load");
    assert_eq!(model.dtype(), DType::F32);

    // Two requests of tokens below the vocabulary's specials, with an end of text in each.
    let ids = |length: usize, seed: u32| -> Vec<u32> {
        (0..length as u32)
            .map(|token| match (token * 7 + seed) % 11 {
                0 => EOS,
                _ => (token * token * 31 + token * 17 + seed * 13) % 59,
            })
            .collect()
    };
    let requests = [
        Request {
            ids: ids(41, 3),
            pages: vec![2, 0],
            slot: 0,
        },
        Request {
            ids: ids(27, 8),
            pages: vec![1, 3],
            slot: STEP as u32,
        },
    ];
    let bound = tolerance(model_terms(&config, 41));

    // The whole rows at once.
    let whole = Memory::new(&*model);
    let parts: Vec<_> = requests
        .iter()
        .map(|request| (request, 0, request.ids.clone(), 1))
        .collect();
    let expected = forward(&*model, Phase::Prefill, &parts, &whole);
    let row_of = |request: usize, position: usize| -> usize {
        requests[..request]
            .iter()
            .map(|r| r.ids.len())
            .sum::<usize>()
            + position
    };

    // A prefill of each row's first tokens, then steps.
    let stepped = Memory::new(&*model);
    let mut computed = [20_usize, 13];
    let parts: Vec<_> = requests
        .iter()
        .zip(computed)
        .map(|(request, split)| (request, 0, request.ids[..split].to_vec(), 1))
        .collect();
    let opened = forward(&*model, Phase::Prefill, &parts, &stepped);
    let mut first = 0;
    for (request, split) in computed.iter().enumerate() {
        assert_close(
            "a shorter prefill",
            &opened.narrow(0, first, *split).unwrap(),
            &expected.narrow(0, row_of(request, 0), *split).unwrap(),
            bound,
        );
        first += split;
    }

    // Each step: the rows that stand are the request's own tokens, the rest wrong drafts.
    let pattern = [2_usize, 4, 1, 3];
    let mut standing = [1_u32, 1];
    let mut last_rows = [0_usize, 0];
    let mut step = 0;
    while requests
        .iter()
        .zip(computed)
        .any(|(request, done)| done < request.ids.len())
    {
        let mut parts = Vec::new();
        let mut kept = Vec::new();
        for (index, request) in requests.iter().enumerate() {
            let remaining = request.ids.len() - computed[index];
            if remaining == 0 {
                continue;
            }

            let rows = remaining.min(STEP);
            let stand = if remaining <= STEP {
                rows
            } else {
                pattern[(step + index) % pattern.len()].min(rows)
            };
            let fed: Vec<u32> = (0..rows)
                .map(|row| {
                    let own = request.ids[computed[index] + row];
                    if row < stand { own } else { (own + 7) % 59 }
                })
                .collect();
            parts.push((request, computed[index], fed, standing[index]));
            kept.push((index, rows, stand));
        }

        let streams = forward(&*model, Phase::Decode, &parts, &stepped);
        let mut first = 0;
        for (index, rows, stand) in kept {
            assert_close(
                "the standing rows of a step",
                &streams.narrow(0, first, stand).unwrap(),
                &expected
                    .narrow(0, row_of(index, computed[index]), stand)
                    .unwrap(),
                bound,
            );
            first += rows;
            computed[index] += stand;
            standing[index] = stand as u32;
            last_rows[index] = rows;
        }
        step += 1;
    }

    // What the two runs left: the cache rows of every token, the index keys of every
    // complete group, the ring's last rows, and the state where the next step would read it.
    let linear_layers = config.linear_layers() as usize;
    let ple_layers = config.ngram.layers.len();
    let ratio = config.indexer.compress_ratio;
    let capacity = config.indexer.ring_capacity(DRAFTS);
    let flat = |plane: &Tensor| {
        let (leading, width) = (
            plane.dims()[0] * plane.dims()[1],
            plane.dims()[2..].iter().product::<usize>(),
        );
        plane.reshape((leading, width)).unwrap()
    };

    for (index, request) in requests.iter().enumerate() {
        let length = request.ids.len() as u32;
        let slot = |position: u32| {
            request.pages[(position / PAGE_TOKENS) as usize] * PAGE_TOKENS + position % PAGE_TOKENS
        };
        let token_rows: Vec<u32> = (0..length).map(slot).collect();
        let groups = PAGE_TOKENS / ratio;
        let group_rows: Vec<u32> = (0..length / ratio)
            .map(|group| request.pages[(group / groups) as usize] * groups + group % groups)
            .collect();
        let ring_rows: Vec<u32> = (length - capacity..length)
            .map(|position| (request.slot) * capacity + position % capacity)
            .collect();

        for layer in 0..config.attention_layers() as usize {
            let (ours, theirs) = (&stepped.cache.layers[layer], &whole.cache.layers[layer]);
            let planes = [
                ("keys", &ours.key, &theirs.key, &token_rows),
                ("values", &ours.value, &theirs.value, &token_rows),
                (
                    "index keys",
                    ours.index_keys.as_ref().unwrap(),
                    theirs.index_keys.as_ref().unwrap(),
                    &group_rows,
                ),
            ];
            for (what, ours, theirs, rows) in planes {
                assert_close(
                    what,
                    &rows_of(&flat(ours), rows),
                    &rows_of(&flat(theirs), rows),
                    bound,
                );
            }

            let ring = linear_layers + ple_layers + layer;
            assert_close(
                "the ring",
                &rows_of(&flat(&stepped.live[ring][0]), &ring_rows),
                &rows_of(&flat(&whole.live[ring][0]), &ring_rows),
                bound,
            );
            let positions = |memory: &Memory| -> Vec<Vec<i32>> {
                rows_of(&flat(&memory.live[ring][1]), &ring_rows)
                    .to_vec2()
                    .unwrap()
            };
            assert_eq!(
                positions(&stepped),
                positions(&whole),
                "the ring's positions"
            );
        }

        // The last step's rows all stand: the state after them is in the slot of its last
        // row, and the history before the next token begins at the window row of that row.
        let last = last_rows[index];
        let window = |memory: &Memory, layer: usize, first: usize, history: usize| {
            memory.live[layer][0]
                .get(request.slot as usize)
                .and_then(|window| window.narrow(0, first, history))
                .unwrap()
        };
        for layer in 0..linear_layers {
            let history = config.text.gated_delta_net.conv_width as usize - 1;
            assert_close(
                "the convolution window",
                &window(&stepped, layer, last - 1, history),
                &window(&whole, layer, 0, history),
                bound,
            );
            assert_close(
                "the recurrent state",
                &stepped.live[layer][1]
                    .get(request.slot as usize + last - 1)
                    .unwrap(),
                &whole.live[layer][1].get(request.slot as usize).unwrap(),
                bound,
            );
        }
        for layer in linear_layers..linear_layers + ple_layers {
            let ngram = &config.ngram;
            let history = ((ngram.conv_kernel_size - 1) * ngram.ngram_size) as usize;
            assert_close(
                "the PLE window",
                &window(&stepped, layer, last - 1, history),
                &window(&whole, layer, 0, history),
                bound,
            );
        }
    }
}
