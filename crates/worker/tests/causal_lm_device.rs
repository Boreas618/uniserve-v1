//! `CausalLmRunner` on a device: a prefill commits its first token into its slot and a decode
//! step continues from it; a slot whose first step is a decode starts from its batch's row;
//! a batch's page work is issued around the replay.
//!
//! Every case needs GPU 0 and a running Build Server (`UNISERVE_BUILD_SERVER_SOCKET`). Run
//! them on a GPU host:
//!
//! ```text
//! cargo test -p worker --test causal_lm_device -- --ignored --test-threads 1
//! ```

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

mod support;

use std::sync::{Arc, Mutex};

use ipc::execution::{
    Batch, BatchColumns, BatchId, BatchRow, BatchRows, CallId, FreshSlot, RowKind,
    Shape as BatchShape, StateCopy, StateDirection, Variant,
};
use ipc::philox::DrawAddress;
use nn::capability::Drafter;
use nn::family::{CacheShape, ExpertShape, StateShape, TextModel};
use nn::functional::{WeightOrder, embedding, rms_norm};
use nn::modules::Linear;
use nn::{CausalLm, ExpertExchange, KvCache, RecurrentState, TokenBatch};
use ops::Generation;
use platform::{DeviceAllocation, DeviceRange, Stream};
use proto::module::commit::{COMMIT_FIELDS, column, mode};
use proto::module::sampler::{PARAMETER_COLUMNS, TOP_P};
use uniserve_tensor::{DType, Tensor};
use worker::buckets::Buckets;
use worker::runners::causal_lm::{CausalLmConfig, PageWork};
use worker::{CausalLmRunner, Graphs, Runner};

use support::{DEVICE, Kernels, Outcome};

const VOCABULARY: usize = 32;
const HIDDEN: usize = 16;
const SLOTS: u32 = 8;
const PAGE: u32 = 16;
const MAX_KV_LEN: u32 = 64;

/// The token the model continues `id` with.
///
/// The embedding of `id` is the unit vector of column `id mod 16`, which the normalization
/// keeps a multiple of, and the head's row `t` is the unit vector of column `(t + 1) mod 16`.
/// The largest logits are therefore those of the tokens `t` with `t + 1 ≡ id (mod 16)`, and a
/// greedy selection takes the lowest of them.
const fn next(id: u32) -> u32 {
    (id + 15) % 16
}

/// The `[rows, HIDDEN]` matrix whose row `r` is the unit vector of column `column(r)`.
fn units(rows: usize, column: impl Fn(usize) -> usize) -> Vec<f32> {
    let mut values = vec![0.0; rows * HIDDEN];
    for row in 0..rows {
        values[row * HIDDEN + column(row)] = 1.0;
    }
    values
}

/// A causal language model with no attention layer: an embedding, a normalization and a head.
struct Tiny {
    table: Tensor,
    gain: Tensor,
    head: Linear,
    /// The weights' device memory.
    _memory: Vec<DeviceAllocation>,
}

impl Tiny {
    /// The model over meta weights: what a profile run computes on.
    fn meta() -> Outcome<Self> {
        let weight = |address: u64, dims: &[usize]| {
            backend::retained(address, dims, DType::F32, &backend::meta())
        };

        Ok(Self {
            table: weight(0x1_0000, &[VOCABULARY, HIDDEN])?,
            gain: weight(0x2_0000, &[HIDDEN])?,
            head: Linear::new(
                weight(0x3_0000, &[VOCABULARY, HIDDEN])?,
                None,
                WeightOrder::NxK,
            )?,
            _memory: Vec::new(),
        })
    }

    /// The model over weights on the device.
    fn on_device(stream: &Stream) -> Outcome<Self> {
        let table = units(VOCABULARY, |id| id % HIDDEN);
        let head = units(VOCABULARY, |token| (token + 1) % HIDDEN);

        let (table_memory, table) = support::weight(stream, &table, &[VOCABULARY, HIDDEN])?;
        let (gain_memory, gain) = support::weight(stream, &[1.0; HIDDEN], &[HIDDEN])?;
        let (head_memory, head) = support::weight(stream, &head, &[VOCABULARY, HIDDEN])?;

        Ok(Self {
            table,
            gain,
            head: Linear::new(head, None, WeightOrder::NxK)?,
            _memory: vec![table_memory, gain_memory, head_memory],
        })
    }
}

impl CausalLm for Tiny {
    fn embed_input_ids(&self, ids: &Tensor) -> nn::Result<Tensor> {
        embedding(ids, &self.table, 0)
    }

    fn forward(
        &self,
        _batch: &TokenBatch,
        _cache: &[KvCache],
        _state: Option<&RecurrentState>,
        _exchange: Option<&ExpertExchange>,
        hidden: &Tensor,
    ) -> nn::Result<Tensor> {
        rms_norm(hidden, &self.gain, 1e-6, false, hidden.dtype())
    }

    fn compute_logits(&self, hidden: &Tensor) -> nn::Result<Tensor> {
        self.head.forward(hidden)
    }
}

impl TextModel for Tiny {
    fn hidden(&self) -> u32 {
        HIDDEN as u32
    }

    fn vocabulary(&self) -> u32 {
        VOCABULARY as u32
    }

    fn dtype(&self) -> DType {
        DType::F32
    }

    fn cache(&self) -> CacheShape {
        CacheShape {
            layers: 0,
            kv_heads: 0,
            head_dim: 0,
        }
    }

    fn state(&self) -> Option<StateShape> {
        None
    }

    fn experts(&self) -> Option<ExpertShape> {
        None
    }

    fn drafter(&self) -> Option<&dyn Drafter> {
        None
    }
}

/// Decode buckets of two and four rows and prefill buckets of four rows over eight tokens.
fn config() -> CausalLmConfig {
    CausalLmConfig {
        phases: vec![ipc::PhaseId(0)],
        buckets: Buckets::causal_lm(&[2, 4], &[8], true),
        slots: SLOTS,
        page_size: PAGE,
        max_kv_len: MAX_KV_LEN,
        candidates: 2,
        scratch_pages: 0,
    }
}

/// A runner on the device with every bucket captured, and the memory its graphs address.
struct Bench {
    runner: CausalLmRunner,
    _kernels: Kernels,
    _region: DeviceAllocation,
}

impl Bench {
    fn open() -> Outcome<Self> {
        let config = config();
        let inputs = CausalLmRunner::input_bytes(&config)?;

        // The profile run states the keys and the peaks before anything touches the device.
        let profiled = CausalLmRunner::new(
            Arc::new(Tiny::meta()?),
            config.clone(),
            Graphs::meta(Arc::new(Generation::new(0, [])), inputs),
            None,
            Vec::new(),
            None,
        )?;
        profiled.graphs().profile_run(&profiled)?;

        let mut kernels = Kernels::open()?;
        let serving = kernels.generation(1, &profiled.graphs().keys())?;

        let stream = Stream::new(DEVICE)?;
        let region = DeviceAllocation::linear(DEVICE, profiled.graphs().peak() as usize)?;
        let range = DeviceRange::new(
            platform::DevicePtr::from_address(region.address()),
            region.len(),
        );
        let graphs = Graphs::new(
            stream.clone(),
            serving,
            range,
            inputs,
            profiled.graphs().input_peak().max(256),
            2,
        )?;

        let runner = CausalLmRunner::new(
            Arc::new(Tiny::on_device(&stream)?),
            config,
            graphs,
            Some(stream),
            Vec::new(),
            None,
        )?;
        runner.graphs().profile_run(&runner)?;
        for bucket in runner.buckets() {
            runner.graphs().capture(&runner, &bucket)?;
        }

        Ok(Self {
            runner,
            _kernels: kernels,
            _region: region,
        })
    }
}

/// A greedy row's sampling parameters.
fn greedy() -> Vec<f32> {
    let mut row = vec![0.0; PARAMETER_COLUMNS as usize];
    row[TOP_P as usize] = 1.0;
    row
}

/// The batch of one call of `kind` on `slot` over `columns`, which state its tokens.
fn batch(kind: RowKind, variant: Variant, slot: u32, mut columns: BatchColumns) -> Batch {
    let tokens = columns.input_ids.len() as u32;

    columns.rows.calls = vec![BatchRow {
        request: slot,
        kind,
        cache_group: 0,
        rows: 0..1,
    }];
    columns.rows.slot_mapping = vec![-1; tokens as usize];
    columns.rows.query_offsets = vec![0, tokens as i32];
    columns.rows.logits_rows = vec![0];
    columns.slots = vec![slot as i32];
    columns.parameters = greedy();
    columns.page_size = PAGE;
    columns.max_kv_len = MAX_KV_LEN;

    Batch {
        id: BatchId(1),
        phase: ipc::PhaseId(0),
        group: "text".to_owned(),
        calls: vec![CallId(1)],
        shape: BatchShape::prefill(1, tokens, variant),
        columns,
        generation: 1,
        cache_units: Vec::new(),
        block_tables: Vec::new(),
        page_loads: Vec::new(),
        page_offloads: Vec::new(),
        latent_restores: Vec::new(),
        host_outputs: Vec::new(),
        images: Vec::new(),
        frees: Vec::new(),
        media: None,
    }
}

/// The one-row commit table of `slot` in `mode`, whose row samples task zero.
fn commit(slot: u32, mode: i64) -> Vec<i64> {
    let mut table = vec![0; COMMIT_FIELDS];
    table[column::SLOT] = i64::from(slot);
    table[column::MODE] = mode;
    table
}

/// A prompt of `ids` on `slot`, a fresh slot, whose last token ends the prompt.
fn prompt(slot: u32, ids: &[i32]) -> Batch {
    let tokens = ids.len() as i32;

    let mut table = commit(slot, mode::EXPLICIT);
    table[column::CACHE_LENGTH] = i64::from(tokens);
    table[column::LOGICAL] = i64::from(tokens);
    table[column::SAMPLING] = 1;

    let columns = BatchColumns {
        rows: BatchRows {
            positions: (0..tokens).collect(),
            seq_lens: vec![tokens],
            ..BatchRows::default()
        },
        input_ids: ids.to_vec(),
        draws: vec![DrawAddress::default()],
        commit: table,
        fresh_slots: vec![FreshSlot {
            slot,
            history: Vec::new(),
        }],
        ..BatchColumns::default()
    };
    batch(RowKind::Prefill, Variant::PrefillGreedy, slot, columns)
}

/// One decode step of `slot` at `position`, whose input token the host states as `id`.
fn step(slot: u32, id: i32, position: i32) -> Batch {
    let columns = BatchColumns {
        rows: BatchRows {
            positions: vec![position],
            seq_lens: vec![position + 1],
            ..BatchRows::default()
        },
        input_ids: vec![id],
        draws: vec![DrawAddress {
            key: 9,
            position: 3,
        }],
        commit: commit(slot, mode::DECODE),
        ..BatchColumns::default()
    };
    batch(RowKind::Decode, Variant::DecodeGreedy, slot, columns)
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_prefill_commits_its_first_token_and_a_decode_step_continues_from_it() -> Outcome {
    let mut bench = Bench::open()?;

    // The prompt's last token is 7, and the commit states the slot's positions.
    let output = bench.runner.execute(&prompt(2, &[5, 6, 7]))?;
    let first = &output.calls[0];
    assert_eq!(first.tokens, [next(7)]);
    assert_eq!((first.cache_computed_len, first.step_index), (3, 1));

    // The token is the most likely of the raw distribution, and its candidates are reported.
    let scores = &first.token_logprobs[0];
    assert_eq!(scores.len(), 1 + 2);
    assert_eq!((scores[0].token_id, scores[0].rank), (next(7), 1));
    assert!(scores[0].logprob < 0.0);

    // The decode reads the committed token from the slot: the id the host states is not it.
    let output = bench.runner.execute(&step(2, 0, 3))?;
    let second = &output.calls[0];
    assert_eq!(second.tokens, [next(next(7))]);
    assert_eq!((second.cache_computed_len, second.step_index), (4, 2));

    // And the step after it continues from that one.
    let output = bench.runner.execute(&step(2, 0, 4))?;
    assert_eq!(output.calls[0].tokens, [next(next(next(7)))]);
    assert!(output.timing.device_us > 0);
    Ok(())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_slot_whose_first_step_is_a_decode_starts_from_its_batch_s_row() -> Outcome {
    let mut bench = Bench::open()?;

    // The request generated two tokens elsewhere: its slot starts at the row's input token,
    // position and draw position.
    let mut batch = step(4, 9, 6);
    batch.columns.fresh_slots = vec![FreshSlot {
        slot: 4,
        history: vec![9, 9],
    }];

    let output = bench.runner.execute(&batch)?;
    let call = &output.calls[0];
    assert_eq!(call.tokens, [next(9)]);
    assert_eq!((call.cache_computed_len, call.step_index), (7, 4));
    Ok(())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_batch_s_page_work_is_issued_before_and_after_its_replay() -> Outcome {
    let mut bench = Bench::open()?;
    let calls = Arc::new(Mutex::new(Vec::new()));

    let seen = Arc::clone(&calls);
    bench.runner.set_page_work(Box::new(move |_, work| {
        seen.lock().unwrap().push(work);
        Ok(())
    }));

    let mut batch = prompt(2, &[5, 6, 7]);
    batch.columns.state_copies = vec![StateCopy {
        slot: 2,
        units: vec![ipc::UnitId(0)],
        direction: StateDirection::Restore,
    }];
    bench.runner.execute(&batch)?;

    assert_eq!(
        *calls.lock().unwrap(),
        [PageWork::BeforeReplay, PageWork::AfterReplay]
    );
    Ok(())
}
