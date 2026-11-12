//! The recurrent mixers of a gated delta network: the causal convolution and the gated delta
//! rule on CPU tensors against their recurrences written out in loops, the state they leave in
//! the slots' buffers, and the kernels they and the `GatedDeltaNet` module state under a meta
//! context.
//!
//! Nothing here touches a GPU.

// The helpers below unwrap; clippy's test detection needs the crate to say it is a test crate.
#![cfg(test)]

mod support;

use std::path::Path;

use nn::collective::Split;
use nn::family::StateField;
use nn::functional::{self, WeightOrder};
use nn::modules::{GatedDeltaNet, GatedDeltaNetConfig};
use nn::{
    Checkpoint, CpuWeights, DType, Device, NO_SLOT, PhaseKind, RecurrentState, Tensor, TokenBatch,
    WeightLoader,
};
use support::{assert_close, encode, meta, operators, recorded, sequence, shard, tensor, values};

/// The tolerance of an FP32 result against a reference accumulated in another order.
const F32_TOLERANCE: f32 = 2e-5;

/// A CPU column of U32 `values`.
fn column(values: &[u32]) -> Tensor {
    Tensor::new(values, &Device::Cpu).unwrap()
}

/// A CPU state of one recurrent layer over `slots` slots: `rows` are the state rows' offsets,
/// lengths, slots and checkpoints, `fields` the element type and extents of each field, and
/// `trail` the tokens of the trail the batch keeps, when it keeps one. Every buffer is zeros.
fn state(
    rows: [&[u32]; 4],
    fields: &[(DType, Vec<usize>)],
    slots: usize,
    trail: Option<usize>,
) -> RecurrentState {
    let buffers = |leading: usize| -> Vec<Vec<Tensor>> {
        let layer = fields
            .iter()
            .map(|(dtype, dims)| {
                let shape = [&[leading][..], dims.as_slice()].concat();
                Tensor::zeros(shape, *dtype, &Device::Cpu).unwrap()
            })
            .collect();
        vec![layer]
    };

    RecurrentState {
        query_offsets: column(rows[0]),
        seq_lens: column(rows[1]),
        slots: column(rows[2]),
        checkpoints: column(rows[3]),
        live: buffers(slots),
        checkpoint: buffers(slots),
        trail: trail.map(buffers),
    }
}

/// A batch of `phase` over pages of `page_size` tokens, on the CPU or of meta tensors; the
/// recurrent functions read nothing else of it.
fn batch(phase: PhaseKind, page_size: u32, on_meta: bool) -> TokenBatch {
    let column = |dtype| {
        if on_meta {
            meta(1, dtype)
        } else {
            Tensor::zeros(1, dtype, &Device::Cpu).unwrap()
        }
    };

    TokenBatch {
        positions: column(DType::I32),
        slot_mapping: column(DType::U32),
        seq_lens: column(DType::U32),
        query_offsets: column(DType::U32),
        logits_rows: column(DType::U32),
        phase,
        page_size,
        num_tokens: 1,
        max_kv_len: 4096,
        draft_lens: None,
        rope_positions: None,
        live_tokens: None,
    }
}

/// Writes `values` into row `row` of the CPU buffer `buffer`.
fn fill(buffer: &Tensor, row: usize, values: &[f32]) {
    let mut dims = buffer.dims().to_vec();
    dims[0] = 1;
    buffer
        .slice_set(&tensor(values, dims, buffer.dtype()), 0, row)
        .unwrap();
}

/// Row `row` of `buffer`, flat.
fn row_of(buffer: &Tensor, row: usize) -> Vec<f32> {
    values(&buffer.get(row).unwrap())
}

// ---- The causal convolution -------------------------------------------------------------------

/// `silu(Σ_j w[c][j] · window[t + j][c])` for each of the window's last tokens: `window` is
/// `[width − 1 + tokens][channels]` flat, the inputs the taps read, and `weight`
/// `[channels][width]` flat.
fn convolve(window: &[f32], weight: &[f32], channels: usize, width: usize) -> Vec<f32> {
    let tokens = window.len() / channels - (width - 1);

    (0..tokens)
        .flat_map(|token| {
            (0..channels).map(move |channel| {
                let sum: f32 = (0..width)
                    .map(|tap| {
                        weight[channel * width + tap] * window[(token + tap) * channels + channel]
                    })
                    .sum();
                sum / (1.0 + (-sum).exp())
            })
        })
        .collect()
}

#[test]
fn causal_conv1d_continues_each_row_from_its_slots_last_inputs() {
    let (channels, width) = (6, 4);
    let kept = width - 1;

    // Seven tokens: a row of three that starts its sequence in slot 1 and keeps a checkpoint
    // after its second token, a row of two that continues slot 0 at position seven, a padding
    // row of one, and the bucket's last token, which no row names.
    let state = state(
        [&[0, 3, 5, 6], &[3, 9, 1], &[1, 0, NO_SLOT], &[2, 0, 0]],
        &[(DType::F32, vec![kept, channels])],
        3,
        Some(7),
    );
    let held = sequence(kept * channels, 1);
    fill(&state.live[0][0], 0, &held);

    let (x, weight) = (sequence(7 * channels, 2), sequence(channels * width, 3));
    let out = functional::causal_conv1d(
        &tensor(&x, (7, channels), DType::F32),
        &tensor(&weight, (channels, width), DType::F32),
        &state,
        0,
        0,
    )
    .unwrap();
    assert_eq!(out.dims(), [7, channels]);

    // The inputs each row's taps read: zeros or the slot's last inputs, then the row's own.
    let inputs =
        |tokens: std::ops::Range<usize>| x[tokens.start * channels..tokens.end * channels].to_vec();
    let first = [vec![0.0; kept * channels], inputs(0..3)].concat();
    let second = [held.clone(), inputs(3..5)].concat();

    let mut expected = convolve(&first, &weight, channels, width);
    expected.extend(convolve(&second, &weight, channels, width));
    expected.extend(vec![0.0; 2 * channels]);
    assert_close(&values(&out), &expected, F32_TOLERANCE);

    // Each slot holds its row's last inputs, the padding row's none; the checkpoint is the
    // window after the first row's second token.
    let window =
        |inputs: &[f32], seen: usize| inputs[seen * channels..][..kept * channels].to_vec();
    let (live, checkpoint) = (&state.live[0][0], &state.checkpoint[0][0]);
    assert_eq!(row_of(live, 1), window(&first, 3));
    assert_eq!(row_of(live, 0), window(&second, 2));
    assert_eq!(row_of(live, 2), vec![0.0; kept * channels]);
    assert_eq!(row_of(checkpoint, 1), window(&first, 2));
    assert_eq!(row_of(checkpoint, 0), vec![0.0; kept * channels]);

    // The trail holds the window after every token of a row.
    let trail = &state.trail.as_ref().unwrap()[0][0];
    for token in 0..3 {
        assert_eq!(row_of(trail, token), window(&first, token + 1));
    }
    for token in 0..2 {
        assert_eq!(row_of(trail, 3 + token), window(&second, token + 1));
    }

    // A state of another extent, and a field the state does not hold.
    let narrow = tensor(&weight[..channels * 3], (channels, 3), DType::F32);
    let input = tensor(&x, (7, channels), DType::F32);
    let refused = functional::causal_conv1d(&input, &narrow, &state, 0, 0);
    assert!(refused.unwrap_err().to_string().contains("live buffer"));

    let weight = tensor(&weight, (channels, width), DType::F32);
    assert!(functional::causal_conv1d(&input, &weight, &state, 0, 1).is_err());
    assert!(functional::causal_conv1d(&input, &weight, &state, 1, 0).is_err());
}

// ---- The gated delta rule ---------------------------------------------------------------------

/// The heads of the delta-rule tests: two key heads of four elements, each serving two value
/// heads of three.
const KEY_HEADS: usize = 2;
const VALUE_HEADS: usize = 4;
const KEY_DIM: usize = 4;
const VALUE_DIM: usize = 3;

/// The per-head parameters and one token's inputs of the rule.
struct Step<'a> {
    query: &'a [f32],
    key: &'a [f32],
    value: &'a [f32],
    a: &'a [f32],
    b: &'a [f32],
}

/// One token of the gated delta rule, written out: decays `memory`
/// (`[VALUE_HEADS][KEY_DIM][VALUE_DIM]` flat), steps it toward the token's value along its
/// key, and returns what the query reads, `[VALUE_HEADS][VALUE_DIM]` flat.
fn advance(memory: &mut [f32], step: &Step<'_>, a_log: &[f32], dt_bias: &[f32]) -> Vec<f32> {
    let group = VALUE_HEADS / KEY_HEADS;
    let unit = |x: &[f32]| -> Vec<f32> {
        let length = (x.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
        x.iter().map(|x| x / length).collect()
    };
    let mut out = Vec::new();

    for head in 0..VALUE_HEADS {
        let key_head = head / group;
        let key = unit(&step.key[key_head * KEY_DIM..][..KEY_DIM]);
        let query: Vec<f32> = unit(&step.query[key_head * KEY_DIM..][..KEY_DIM])
            .iter()
            .map(|q| q / (KEY_DIM as f32).sqrt())
            .collect();
        let value = &step.value[head * VALUE_DIM..][..VALUE_DIM];

        let rate = step.a[head] + dt_bias[head];
        let softplus = if rate > 20.0 {
            rate
        } else {
            (1.0 + rate.exp()).ln()
        };
        let decay = (-a_log[head].exp() * softplus).exp();
        let beta = 1.0 / (1.0 + (-step.b[head]).exp());

        let memory = &mut memory[head * KEY_DIM * VALUE_DIM..][..KEY_DIM * VALUE_DIM];
        for element in memory.iter_mut() {
            *element *= decay;
        }
        for column in 0..VALUE_DIM {
            let recalled: f32 = (0..KEY_DIM)
                .map(|row| key[row] * memory[row * VALUE_DIM + column])
                .sum();
            let error = beta * (value[column] - recalled);
            for row in 0..KEY_DIM {
                memory[row * VALUE_DIM + column] += key[row] * error;
            }
        }
        for column in 0..VALUE_DIM {
            out.push(
                (0..KEY_DIM)
                    .map(|row| query[row] * memory[row * VALUE_DIM + column])
                    .sum(),
            );
        }
    }
    out
}

/// The inputs of `tokens` tokens of the rule, different for each `seed`.
struct Inputs {
    query: Vec<f32>,
    key: Vec<f32>,
    value: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl Inputs {
    fn new(tokens: usize, seed: u32) -> Self {
        Self {
            query: sequence(tokens * KEY_HEADS * KEY_DIM, seed),
            key: sequence(tokens * KEY_HEADS * KEY_DIM, seed + 1),
            value: sequence(tokens * VALUE_HEADS * VALUE_DIM, seed + 2),
            // Decay inputs on both sides of the softplus's linear threshold.
            a: sequence(tokens * VALUE_HEADS, seed + 3)
                .iter()
                .map(|a| a * 30.0)
                .collect(),
            b: sequence(tokens * VALUE_HEADS, seed + 4),
        }
    }

    /// Token `token`'s inputs.
    fn step(&self, token: usize) -> Step<'_> {
        Step {
            query: &self.query[token * KEY_HEADS * KEY_DIM..][..KEY_HEADS * KEY_DIM],
            key: &self.key[token * KEY_HEADS * KEY_DIM..][..KEY_HEADS * KEY_DIM],
            value: &self.value[token * VALUE_HEADS * VALUE_DIM..][..VALUE_HEADS * VALUE_DIM],
            a: &self.a[token * VALUE_HEADS..][..VALUE_HEADS],
            b: &self.b[token * VALUE_HEADS..][..VALUE_HEADS],
        }
    }

    /// The rule over these inputs' `tokens` tokens, the rows and buffers `state`'s.
    fn rule(
        &self,
        tokens: usize,
        (a_log, dt_bias): (&[f32], &[f32]),
        batch: &TokenBatch,
        state: &RecurrentState,
    ) -> nn::Result<Tensor> {
        let heads = |x: &[f32], heads, dim| tensor(x, (tokens, heads, dim), DType::F32);
        let gate = |x: &[f32]| tensor(x, (tokens, VALUE_HEADS), DType::F32);

        functional::gated_delta_rule(
            &heads(&self.query, KEY_HEADS, KEY_DIM),
            &heads(&self.key, KEY_HEADS, KEY_DIM),
            &heads(&self.value, VALUE_HEADS, VALUE_DIM),
            &gate(&self.a),
            &gate(&self.b),
            &tensor(a_log, VALUE_HEADS, DType::F32),
            &tensor(dt_bias, VALUE_HEADS, DType::F32),
            batch,
            state,
            0,
            0,
        )
    }
}

/// The extents of one slot's delta-rule state.
fn memory_field() -> (DType, Vec<usize>) {
    (DType::F32, vec![VALUE_HEADS, KEY_DIM, VALUE_DIM])
}

/// The rule over several tokens of several rows is the recurrence computed token by token:
/// a row that starts its sequence from zeros, a row that continues its slot's state, the
/// checkpoint a row keeps and the trail of every token.
#[test]
fn the_gated_delta_rule_is_its_recurrence_over_each_rows_tokens() {
    let memory = VALUE_HEADS * KEY_DIM * VALUE_DIM;
    let (a_log, dt_bias) = (sequence(VALUE_HEADS, 1), sequence(VALUE_HEADS, 2));

    // Six tokens: a row of three that starts its sequence in slot 0 and keeps a checkpoint
    // after its second token, a row of two that continues slot 2, and a padding row of one.
    let state = state(
        [&[0, 3, 5, 6], &[3, 7, 1], &[0, 2, NO_SLOT], &[2, 0, 0]],
        &[memory_field()],
        3,
        Some(6),
    );
    let held = sequence(memory, 3);
    fill(&state.live[0][0], 2, &held);

    let inputs = Inputs::new(6, 4);
    let decode = batch(PhaseKind::Decode, 64, false);
    let out = inputs.rule(6, (&a_log, &dt_bias), &decode, &state).unwrap();
    assert_eq!(out.dims(), [6, VALUE_HEADS, VALUE_DIM]);

    // The recurrence, token by token, with the state after each.
    let mut expected = Vec::new();
    let mut after = Vec::new();
    for (tokens, mut memory) in [(0..3, vec![0.0; memory]), (3..5, held)] {
        for token in tokens {
            expected.extend(advance(&mut memory, &inputs.step(token), &a_log, &dt_bias));
            after.push(memory.clone());
        }
    }
    expected.extend(vec![0.0; VALUE_HEADS * VALUE_DIM]);
    assert_close(&values(&out), &expected, F32_TOLERANCE);

    let (live, checkpoint) = (&state.live[0][0], &state.checkpoint[0][0]);
    assert_close(&row_of(live, 0), &after[2], F32_TOLERANCE);
    assert_close(&row_of(live, 2), &after[4], F32_TOLERANCE);
    assert_eq!(row_of(live, 1), vec![0.0; memory]);
    assert_close(&row_of(checkpoint, 0), &after[1], F32_TOLERANCE);
    assert_eq!(row_of(checkpoint, 2), vec![0.0; memory]);

    let trail = &state.trail.as_ref().unwrap()[0][0];
    for (token, after) in after.iter().enumerate() {
        assert_close(&row_of(trail, token), after, F32_TOLERANCE);
    }
}

/// A prompt advanced in one batch and a token appended by a decode read the same state: the
/// decode continues from what the prompt left in the slot.
#[test]
fn a_decode_continues_the_state_its_prompt_left() {
    let memory = VALUE_HEADS * KEY_DIM * VALUE_DIM;
    let (a_log, dt_bias) = (sequence(VALUE_HEADS, 5), sequence(VALUE_HEADS, 6));
    let prompt = Inputs::new(5, 7);
    let next = Inputs::new(1, 20);

    let rows = |offsets: &'static [u32], seen: &'static [u32]| {
        state([offsets, seen, &[1], &[0]], &[memory_field()], 2, None)
    };
    let prefill = rows(&[0, 5], &[5]);
    prompt
        .rule(
            5,
            (&a_log, &dt_bias),
            &batch(PhaseKind::Prefill, 64, false),
            &prefill,
        )
        .unwrap();

    // The decode's state is the prompt's buffers: one more token of the same slot.
    let mut decode = rows(&[0, 1], &[6]);
    decode.live = prefill.live.clone();
    let out = next
        .rule(
            1,
            (&a_log, &dt_bias),
            &batch(PhaseKind::Decode, 64, false),
            &decode,
        )
        .unwrap();

    let mut state = vec![0.0; memory];
    for token in 0..5 {
        advance(&mut state, &prompt.step(token), &a_log, &dt_bias);
    }
    let expected = advance(&mut state, &next.step(0), &a_log, &dt_bias);
    assert_close(&values(&out), &expected, F32_TOLERANCE);
    assert_close(&row_of(&decode.live[0][0], 1), &state, F32_TOLERANCE);
}

#[test]
fn the_rule_refuses_what_its_form_cannot_keep() {
    let (a_log, dt_bias) = (sequence(VALUE_HEADS, 1), sequence(VALUE_HEADS, 2));
    let inputs = Inputs::new(2, 3);
    let rows: [&[u32]; 4] = [&[0, 2], &[2], &[0], &[0]];

    // A batch that advances in chunks keeps no trail, and its pages are whole chunks.
    let trailed = state(rows, &[memory_field()], 1, Some(2));
    let refused = inputs.rule(
        2,
        (&a_log, &dt_bias),
        &batch(PhaseKind::Prefill, 64, false),
        &trailed,
    );
    assert!(refused.unwrap_err().to_string().contains("no trail"));

    let plain = state(rows, &[memory_field()], 1, None);
    let refused = inputs.rule(
        2,
        (&a_log, &dt_bias),
        &batch(PhaseKind::Prefill, 48, false),
        &plain,
    );
    assert!(refused.unwrap_err().to_string().contains("whole number"));

    // The state is FP32 matrices of the heads' extents, one per value head.
    let narrow = state(
        rows,
        &[(DType::F32, vec![VALUE_HEADS, KEY_DIM, 2])],
        1,
        None,
    );
    let refused = inputs.rule(
        2,
        (&a_log, &dt_bias),
        &batch(PhaseKind::Decode, 64, false),
        &narrow,
    );
    assert!(refused.unwrap_err().to_string().contains("live buffer"));
}

/// A meta state of one layer of a network of `config`, over four slots, for `tokens` tokens;
/// `trail` says whether the batch keeps one.
fn meta_state(config: &GatedDeltaNetConfig, tokens: usize, trail: bool) -> RecurrentState {
    let buffers = |leading: usize| -> Vec<Vec<Tensor>> {
        let layer = config
            .state_fields(DType::BF16)
            .iter()
            .map(|field| {
                meta(
                    [&[leading][..], field.dims.as_slice()].concat(),
                    field.dtype,
                )
            })
            .collect();
        vec![layer]
    };

    RecurrentState {
        query_offsets: meta(2, DType::U32),
        seq_lens: meta(1, DType::U32),
        slots: meta(1, DType::U32),
        checkpoints: meta(1, DType::U32),
        live: buffers(4),
        checkpoint: buffers(4),
        trail: trail.then(|| buffers(tokens)),
    }
}

/// The widths of the module tests: two key heads and four value heads of `value_dim`.
fn config(value_dim: u32) -> GatedDeltaNetConfig {
    GatedDeltaNetConfig {
        hidden: 16,
        key_heads: 2,
        value_heads: 4,
        key_dim: 4,
        value_dim,
        conv_width: 4,
        eps: 1e-6,
        split: Split::WHOLE,
    }
}

#[test]
fn the_recurrent_functions_state_their_kernels() {
    let config = config(4);
    let channels = config.conv_channels() as usize;
    let tokens = 8;

    // The convolution's key names its channels and taps.
    let state = meta_state(&config, tokens, false);
    let (mixed, launches) = recorded(|| {
        functional::causal_conv1d(
            &meta((tokens, channels), DType::BF16),
            &meta((channels, 4), DType::BF16),
            &state,
            0,
            0,
        )
    });
    assert_eq!(mixed.unwrap().dims(), [tokens, channels]);
    assert_eq!(
        launches,
        [(
            "causal_conv1d",
            "dtype_bf16_channels_32_width_4_block_32".to_owned()
        )]
    );

    // A decode advances token by token and any other batch in chunks: two kernels. The
    // queries, keys and values are three runs of heads of one convolved row.
    let rule = |phase, state: &RecurrentState| {
        let batch = batch(phase, 64, true);
        let row = meta((tokens, channels), DType::BF16)
            .reshape((tokens, channels / 4, 4))
            .unwrap();
        let heads = |first, heads| row.narrow(1, first, heads).unwrap();

        recorded(|| {
            functional::gated_delta_rule(
                &heads(0, 2),
                &heads(2, 2),
                &heads(4, 4),
                &meta((tokens, 4), DType::BF16),
                &meta((tokens, 4), DType::BF16),
                &meta(4, DType::BF16),
                &meta(4, DType::BF16),
                &batch,
                state,
                0,
                1,
            )
        })
    };
    let base = "dtype_bf16_key_heads_2_value_heads_4_key_dim_4_value_dim_4_block_4";

    let (out, launches) = rule(PhaseKind::Decode, &meta_state(&config, tokens, true));
    assert_eq!(out.unwrap().dims(), [tokens, 4, 4]);
    assert_eq!(
        launches,
        [("gated_delta_rule", format!("{base}_form_recurrent_chunk_1"))]
    );

    let (out, launches) = rule(PhaseKind::Prefill, &state);
    out.unwrap();
    assert_eq!(
        launches,
        [("gated_delta_rule", format!("{base}_form_chunked_chunk_64"))]
    );
}

// ---- The module -------------------------------------------------------------------------------

/// The weights of a gated delta network `mixer` of `config`.
fn weights(config: &GatedDeltaNetConfig) -> Vec<(&'static str, Vec<usize>, Vec<f32>)> {
    let hidden = config.hidden as usize;
    let channels = config.conv_channels() as usize;
    let heads = config.value_heads as usize;
    let value_width = heads * config.value_dim as usize;
    let filled = |shape: Vec<usize>, seed| {
        let count = shape.iter().product();
        (shape, sequence(count, seed))
    };
    let named = |name, (shape, values): (Vec<usize>, Vec<f32>)| (name, shape, values);

    vec![
        named(
            "mixer.in_proj_qkv.weight",
            filled(vec![channels, hidden], 1),
        ),
        named(
            "mixer.in_proj_z.weight",
            filled(vec![value_width, hidden], 2),
        ),
        named("mixer.in_proj_b.weight", filled(vec![heads, hidden], 3)),
        named("mixer.in_proj_a.weight", filled(vec![heads, hidden], 4)),
        named(
            "mixer.conv1d.weight",
            filled(vec![channels, 1, config.conv_width as usize], 5),
        ),
        named("mixer.A_log", filled(vec![heads], 6)),
        named("mixer.dt_bias", filled(vec![heads], 7)),
        named(
            "mixer.norm.weight",
            filled(vec![config.value_dim as usize], 8),
        ),
        named(
            "mixer.out_proj.weight",
            filled(vec![hidden, value_width], 9),
        ),
    ]
}

/// The checkpoint at `directory` holding `weights` as `dtype`.
fn checkpoint(
    directory: &Path,
    dtype: DType,
    weights: &[(&'static str, Vec<usize>, Vec<f32>)],
) -> Checkpoint {
    let typed: Vec<(&str, DType, Vec<usize>, Vec<u8>)> = weights
        .iter()
        .map(|(name, shape, values)| (*name, dtype, shape.clone(), encode(dtype, values)))
        .collect();
    shard(directory, "model.safetensors", &typed);

    Checkpoint::open(directory).unwrap()
}

/// The weight `name` of `weights`, as an FP32 CPU tensor of `shape`.
fn weight(
    weights: &[(&'static str, Vec<usize>, Vec<f32>)],
    name: &str,
    shape: impl Into<nn::Shape>,
) -> Tensor {
    let (_, _, values) = weights.iter().find(|(stored, ..)| *stored == name).unwrap();
    tensor(values, shape, DType::F32)
}

/// The module is its functions in order, over the two fields of its layer's state: the
/// projections, the convolution, the rule over the convolved heads, the gated norm per value
/// head and the output projection. Value heads twice as wide as the key heads are read out of
/// the convolved row all the same.
#[test]
fn a_gated_delta_net_mixes_its_rows_through_the_slots_state() {
    for value_dim in [4, 8] {
        let config = config(value_dim);
        let directory = tempfile::tempdir().unwrap();
        let weights = weights(&config);
        let checkpoint = checkpoint(directory.path(), DType::F32, &weights);
        let store = CpuWeights::new();
        let loader = WeightLoader::new(&checkpoint, &store, None, None);
        let network = GatedDeltaNet::load(&config, loader.root().pp("mixer")).unwrap();
        assert_eq!(network.output().n(), 16);

        // Five tokens of one row that starts its sequence in slot 1, and keeps a trail.
        let fields: Vec<(DType, Vec<usize>)> = config
            .state_fields(DType::F32)
            .into_iter()
            .map(|field| (field.dtype, field.dims))
            .collect();
        let fresh = || state([&[0, 5], &[5], &[1], &[4]], &fields, 2, Some(5));
        let decode = batch(PhaseKind::Decode, 64, false);
        let tokens = 5;
        let hidden = tensor(&sequence(tokens * 16, 10), (tokens, 16), DType::F32);

        let mixed_state = fresh();
        let out = network.forward(&hidden, &decode, &mixed_state, 0).unwrap();
        assert_eq!(out.dims(), [tokens, 16]);

        // The same functions, in order, over a state of their own.
        let (channels, value_width) = (config.conv_channels() as usize, 4 * value_dim as usize);
        let project = |x: &Tensor, name: &str, shape: (usize, usize)| {
            functional::linear(x, &weight(&weights, name, shape), None, WeightOrder::NxK).unwrap()
        };
        let reference = fresh();
        let convolved = functional::causal_conv1d(
            &project(&hidden, "mixer.in_proj_qkv.weight", (channels, 16)),
            &weight(&weights, "mixer.conv1d.weight", (channels, 4)),
            &reference,
            0,
            0,
        )
        .unwrap();
        let heads = |first: usize, heads: usize, dim: usize| {
            convolved
                .narrow(1, first, heads * dim)
                .unwrap()
                .reshape((tokens, heads, dim))
                .unwrap()
        };
        let core = functional::gated_delta_rule(
            &heads(0, 2, 4),
            &heads(8, 2, 4),
            &heads(16, 4, value_dim as usize),
            &project(&hidden, "mixer.in_proj_a.weight", (4, 16)),
            &project(&hidden, "mixer.in_proj_b.weight", (4, 16)),
            &weight(&weights, "mixer.A_log", 4),
            &weight(&weights, "mixer.dt_bias", 4),
            &decode,
            &reference,
            0,
            1,
        )
        .unwrap();
        let normed = functional::gated_rms_norm(
            &core,
            &project(&hidden, "mixer.in_proj_z.weight", (value_width, 16))
                .reshape((tokens, 4, value_dim as usize))
                .unwrap(),
            &weight(&weights, "mixer.norm.weight", value_dim as usize),
            1e-6,
        )
        .unwrap();
        let expected = project(
            &normed.reshape((tokens, value_width)).unwrap(),
            "mixer.out_proj.weight",
            (16, value_width),
        );
        assert_eq!(
            values(&out),
            values(&expected),
            "value heads of {value_dim}"
        );

        // Both fields advanced as the functions' own state did: the live state, the checkpoint
        // after the fourth token and the trail.
        let trails = |state: &RecurrentState| state.trail.clone().unwrap();
        for field in 0..2 {
            for (mixed, reference) in [
                (&mixed_state.live, &reference.live),
                (&mixed_state.checkpoint, &reference.checkpoint),
                (&trails(&mixed_state), &trails(&reference)),
            ] {
                assert_eq!(values(&mixed[0][field]), values(&reference[0][field]));
            }
        }
        assert!(
            row_of(&mixed_state.checkpoint[0][1], 1)
                .iter()
                .any(|x| *x != 0.0)
        );
    }
}

#[test]
fn a_gated_delta_net_states_its_fields_and_its_kernels() {
    let config = config(4);
    assert_eq!(config.conv_channels(), 32);
    assert_eq!(
        config.state_fields(DType::BF16),
        [
            StateField {
                name: "conv",
                dtype: DType::BF16,
                dims: vec![3, 32],
            },
            StateField {
                name: "recurrent",
                dtype: DType::F32,
                dims: vec![4, 4, 4],
            },
        ]
    );

    let directory = tempfile::tempdir().unwrap();
    let checkpoint = checkpoint(directory.path(), DType::BF16, &weights(&config));
    let loader = WeightLoader::meta(&checkpoint, None, None);
    let network = GatedDeltaNet::load(&config, loader.root().pp("mixer")).unwrap();

    // The heads of the convolved row are views of it, so the forward is the eight launches of
    // its stages and nothing between them.
    let tokens = 8;
    let hidden = meta((tokens, 16), DType::BF16);
    let prefill = batch(PhaseKind::Prefill, 64, true);
    let state = meta_state(&config, tokens, false);
    let (out, launches) = recorded(|| network.forward(&hidden, &prefill, &state, 0));
    assert_eq!(out.unwrap().dims(), [tokens, 16]);
    assert_eq!(
        operators(&launches),
        [
            "gemm",
            "gemm",
            "gemm",
            "gemm",
            "causal_conv1d",
            "gated_delta_rule",
            "gated_rms_norm",
            "gemm"
        ]
    );
    assert_eq!(
        launches[5].1,
        "dtype_bf16_key_heads_2_value_heads_4_key_dim_4_value_dim_4_block_4_form_chunked_chunk_64"
    );

    // A state whose layer holds another number of fields is not this network's.
    let mut short = meta_state(&config, tokens, false);
    short.live[0].pop();
    let (refused, _) = recorded(|| network.forward(&hidden, &prefill, &short, 0));
    assert!(refused.unwrap_err().to_string().contains("two expected"));

    // A gated delta network is served on one rank, and its value heads group over its key
    // heads.
    let split = GatedDeltaNetConfig {
        split: Split { size: 2, rank: 0 },
        ..config
    };
    let refused = GatedDeltaNet::load(&split, loader.root().pp("mixer"));
    assert!(
        refused
            .err()
            .unwrap()
            .to_string()
            .contains("one tensor rank")
    );

    let ungrouped = GatedDeltaNetConfig {
        value_heads: 3,
        ..config
    };
    assert!(GatedDeltaNet::load(&ungrouped, loader.root().pp("mixer")).is_err());

    // The convolved row is whole heads of both widths, so each run of heads is a view of it.
    let uneven = GatedDeltaNetConfig {
        value_dim: 6,
        ..config
    };
    let refused = GatedDeltaNet::load(&uneven, loader.root().pp("mixer"));
    assert!(
        refused
            .err()
            .unwrap()
            .to_string()
            .contains("do not both divide")
    );
}
