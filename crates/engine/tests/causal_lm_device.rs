//! `CausalLmRunner` on a device: a prefill commits its first token into its slot and a decode
//! step continues from it, also when it is launched while the step before it is in flight; a
//! slot whose first step is a decode starts from its batch's row;
//! a batch's page work is issued around the replay; a runner that speculates commits the
//! prefix a verification's samples agree with and proposes the drafts after it; a kernel
//! generation of other graph options (candidates, buckets, drafts) serves its shape from its
//! switch on, the slots' state carried across, and the kept shape again after a rollback.
//!
//! Every case needs GPU 0 and this host's Build Server (`uniserve build-server serve`). Run them
//! on a GPU host:
//!
//! ```text
//! cargo test -p engine --test causal_lm_device -- --ignored --test-threads 1
//! ```

// The helpers below unwrap, as tests do (`clippy.toml`).
#![cfg(test)]

mod device;

use std::sync::{Arc, Mutex};

use engine::batch::philox::DrawAddress;
use engine::batch::{
    Batch, BatchId, Columns, DraftWork, NewSlot, Phase, Row, Rows, Shape as BatchShape, StateCopy,
    StateDirection, Variant,
};
use engine::memory::tables::BlockTables;
use engine::runner::buckets::Buckets;
use engine::runner::causal_lm::speculation::DraftPool;
use engine::runner::causal_lm::{CausalLmConfig, PageWork};
use engine::runner::graphs::Graphs;
use engine::runner::{CausalLmRunner, Forward, GraphOptions, Runner};
use nn::capability::Drafter;
use nn::family::{CacheShape, ExpertShape, StateShape, TextModel, TokenHistory};
use nn::functional::{WeightOrder, embedding, rms_norm};
use nn::modules::Linear;
use nn::{CausalLm, ExpertExchange, KvCache, RecurrentState, TokenBatch};
use ops::Generation;
use platform::{DeviceAllocation, DeviceRange, Stream};
use proto::module::commit::{COMMIT_FIELDS, column, mode};
use proto::module::sampler::{PARAMETER_COLUMNS, TOP_P};
use proto::request::RequestId;
use storage::runtime::GroupId;
use tensor::{DType, Tensor};

use device::{DEVICE, Kernels, Outcome};

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
    /// The model's drafter, in a deployment that speculates.
    drafter: Option<TinyDrafter>,
    /// The weights' device memory.
    _memory: Vec<DeviceAllocation>,
}

/// A drafter with no attention layer: the normalized embedding of the token after its entry,
/// which the model's head continues as the model continues that token. The hidden row it is
/// lent enters the sum and leaves it again, so the row is read and the proposal does not
/// depend on it.
struct TinyDrafter {
    gain: Tensor,
}

impl Drafter for TinyDrafter {
    fn cache(&self) -> CacheShape {
        CacheShape {
            index_keys: None,
            layers: 0,
            kv_heads: 0,
            head_dim: 0,
        }
    }

    fn state_layers(&self) -> Vec<StateShape> {
        Vec::new()
    }

    fn forward(
        &self,
        _batch: &TokenBatch,
        _cache: &KvCache,
        _state: Option<&RecurrentState>,
        embedded: &Tensor,
        hidden: &Tensor,
    ) -> nn::Result<Tensor> {
        let next = embedded.add(hidden)?.sub(hidden)?;

        rms_norm(&next, &self.gain, 1e-6, false, next.dtype())
    }
}

impl Tiny {
    /// The model over meta weights, with its drafter when `drafting`: what a profile run
    /// computes on.
    fn meta(drafting: bool) -> Outcome<Self> {
        let weight =
            |address: u64, dims: &[usize]| ops::retained(address, dims, DType::F32, &ops::meta());

        Ok(Self {
            table: weight(0x1_0000, &[VOCABULARY, HIDDEN])?,
            gain: weight(0x2_0000, &[HIDDEN])?,
            head: Linear::new(
                weight(0x3_0000, &[VOCABULARY, HIDDEN])?,
                None,
                WeightOrder::NxK,
            )?,
            drafter: match drafting {
                true => Some(TinyDrafter {
                    gain: weight(0x4_0000, &[HIDDEN])?,
                }),
                false => None,
            },
            _memory: Vec::new(),
        })
    }

    /// The model over weights on the device, with its drafter when `drafting`.
    fn on_device(stream: &Stream, drafting: bool) -> Outcome<Self> {
        let table = units(VOCABULARY, |id| id % HIDDEN);
        let head = units(VOCABULARY, |token| (token + 1) % HIDDEN);

        let (table_memory, table) = device::weight(stream, &table, &[VOCABULARY, HIDDEN])?;
        let (gain_memory, gain) = device::weight(stream, &[1.0; HIDDEN], &[HIDDEN])?;
        let (head_memory, head) = device::weight(stream, &head, &[VOCABULARY, HIDDEN])?;
        let mut memory = vec![table_memory, gain_memory, head_memory];

        let drafter = match drafting {
            true => {
                let (gain_memory, gain) = device::weight(stream, &[1.0; HIDDEN], &[HIDDEN])?;
                memory.push(gain_memory);

                Some(TinyDrafter { gain })
            }
            false => None,
        };

        Ok(Self {
            table,
            gain,
            head: Linear::new(head, None, WeightOrder::NxK)?,
            drafter,
            _memory: memory,
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
            index_keys: None,
            layers: 0,
            kv_heads: 0,
            head_dim: 0,
        }
    }

    fn state_layers(&self) -> Vec<StateShape> {
        Vec::new()
    }

    fn experts(&self) -> Option<ExpertShape> {
        None
    }

    fn drafter(&self) -> Option<&dyn Drafter> {
        self.drafter.as_ref().map(|drafter| drafter as &dyn Drafter)
    }

    fn token_history(&self) -> Option<TokenHistory> {
        None
    }
}

/// The draft pool of the speculating bench: draft pages of one run of a page's tokens over
/// four units. The test drafter reads no cache, so no page of it is written.
const POOL: DraftPool = DraftPool {
    group: 1,
    span: 1,
    page: PAGE,
    units: 4,
};

/// Decode buckets of two and four rows and prefill buckets of four rows over eight tokens,
/// for a runner that verifies at most `drafts` drafts a row.
fn config(drafts: u32) -> CausalLmConfig {
    CausalLmConfig {
        phases: vec![Phase::Prefill, Phase::Decode, Phase::Verify],
        buckets: Buckets::causal_lm(&[2, 4], &[8], true),
        slots: SLOTS,
        page_size: PAGE,
        max_kv_len: MAX_KV_LEN,
        candidates: 2,
        scratch_pages: 0,
        drafts,
        draft_pool: (drafts > 0).then_some(POOL),
    }
}

/// A runner on the device with every bucket captured, and the memory its graphs address.
struct Bench {
    runner: CausalLmRunner,
    kernels: Kernels,
    region: DeviceAllocation,
    /// The memory of the tensors the runner was lent, and the regions of the shapes it was
    /// given beside its own.
    memory: Vec<DeviceAllocation>,
}

impl Bench {
    /// A runner that speculates nothing.
    fn open() -> Outcome<Self> {
        Self::speculating(0)
    }

    /// A runner that verifies at most `drafts` drafts a row with the model's drafter, and
    /// none for zero.
    fn speculating(drafts: u32) -> Outcome<Self> {
        let config = config(drafts);
        let inputs = CausalLmRunner::input_bytes(&config, &Tiny::meta(drafts > 0)?)?;

        // The draft cache of a drafter that reads none: no layer, and tensors nothing reads.
        let draft = |unread: Tensor| KvCache {
            block_tables: unread.clone(),
            page_size: POOL.page,
            layers: Vec::new(),
            workspace: unread,
        };

        // The profile run states the keys and the peaks before anything touches the device.
        let mut profiled = CausalLmRunner::new(
            Arc::new(Tiny::meta(drafts > 0)?),
            config.clone(),
            Graphs::meta(Arc::new(Generation::new(0, [])), inputs),
            None,
            Vec::new(),
            None,
        )?;
        if drafts > 0 {
            let unread = ops::retained(0x5_0000, &[4], DType::F32, &ops::meta())?;
            profiled.set_draft_cache(draft(unread), None, None)?;
        }
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

        let mut memory = Vec::new();
        let mut runner = CausalLmRunner::new(
            Arc::new(Tiny::on_device(&stream, drafts > 0)?),
            config,
            graphs,
            Some(stream.clone()),
            Vec::new(),
            None,
        )?;
        if drafts > 0 {
            let (unread_memory, unread) = device::weight(&stream, &[0.0; 4], &[4])?;
            memory.push(unread_memory);
            runner.set_draft_cache(draft(unread), None, None)?;
        }
        runner.graphs().profile_run(&runner)?;
        // The warm-up, as the worker's capture thread runs it: each bucket is rehearsed over a
        // rehearsal instance of the runner's forward and captured over one beside its graphs.
        let serving = runner.graphs().kernels();
        let capture = runner.capturer(runner.graphs().beside()?)?;
        let rehearsal = runner.capturer(runner.graphs().rehearsal()?)?;
        for bucket in runner.buckets() {
            let plan = capture
                .graphs()
                .examine(&*capture, &bucket, None, None, &serving)?;
            if plan.rehearse {
                rehearsal
                    .graphs()
                    .rehearse(&*rehearsal, &bucket, &serving)?;
            }
            let graph = capture.graphs().capture(&*capture, &bucket, &serving)?;
            runner
                .graphs()
                .install(&bucket, serving.number(), Some(graph))?;
        }
        rehearsal.graphs().release_rehearsal();

        Ok(Self {
            runner,
            kernels,
            region,
            memory,
        })
    }

    /// Prepares the runner's shape under `options` for generation `number`, as a worker
    /// prepares a kernel generation of other graph options: the shape is measured over graphs
    /// that hold no device, its kernels are taken from the Build Server for the keys it
    /// launches, it computes in the runner's region when that holds its peak and in a region of
    /// its own otherwise, and every bucket of it is captured beside the serving graphs and
    /// installed in its graphs. Nothing switches.
    fn reshape(&mut self, number: u64, options: &GraphOptions) -> Outcome {
        let measure = self
            .runner
            .measure(options)?
            .ok_or("a causal runner takes graph options")?;
        let instance = &*measure.instance;
        instance.graphs().profile_run(instance)?;
        let (peak, values) = (instance.graphs().peak(), instance.graphs().input_peak());
        let next = self.kernels.generation(number, &instance.graphs().keys())?;

        let region = if peak as usize > self.region.len() {
            let own = DeviceAllocation::linear(DEVICE, peak as usize)?;
            let range =
                DeviceRange::new(platform::DevicePtr::from_address(own.address()), own.len());
            self.memory.push(own);
            Some(range)
        } else {
            None
        };
        self.runner.reshape(
            number,
            options,
            Arc::clone(&next),
            (region, values.max(256)),
        )?;

        let (_, graphs) = self
            .runner
            .reshaped()
            .ok_or("the runner prepared no shape")?;
        let capture = self.runner.reshaped_capturer(graphs.beside()?)?;
        let rehearsal = self.runner.reshaped_capturer(graphs.rehearsal()?)?;
        for bucket in capture.buckets() {
            let plan = capture
                .graphs()
                .examine(&*capture, &bucket, None, None, &next)?;
            if plan.rehearse {
                rehearsal.graphs().rehearse(&*rehearsal, &bucket, &next)?;
            }
            let graph = capture.graphs().capture(&*capture, &bucket, &next)?;
            graphs.install(&bucket, number, Some(graph))?;
        }
        rehearsal.graphs().release_rehearsal();
        Ok(())
    }
}

/// A greedy row's sampling parameters.
fn greedy() -> Vec<f32> {
    let mut row = vec![0.0; PARAMETER_COLUMNS as usize];
    row[TOP_P as usize] = 1.0;
    row
}

/// The batch of one call of `kind` on `slot` over `columns`, which state its tokens. The model
/// keeps no cache, so no token writes one.
fn batch(kind: Phase, variant: Variant, slot: u32, mut columns: Columns) -> Batch {
    let tokens = columns.input_ids.len() as u32;

    columns.rows.calls = vec![Row {
        slot,
        kind,
        cache_group: 0,
        rows: 0..1,
    }];
    columns.rows.cache_positions = vec![-1; tokens as usize];
    columns.rows.query_offsets = vec![0, tokens as i32];
    columns.rows.logits_rows = vec![0];
    columns.slots = vec![slot as i32];
    columns.parameters = greedy();
    columns.page_size = PAGE;
    columns.max_kv_len = MAX_KV_LEN;

    Batch {
        id: BatchId(1),
        group: GroupId(0),
        phase: kind,
        kernels: 1,
        policy: 1,
        revision: 0,
        copy_budget: proto::config::DEFAULT_COPY_BUDGET,
        shape: BatchShape::prefill(1, tokens, variant),
        requests: vec![RequestId(1)],
        columns,
        media: Vec::new(),
        new_slots: Vec::new(),
    }
}

/// A request started on `slot` with `history`, as a batch states it.
fn new_slot(slot: u32, history: Vec<u32>) -> NewSlot {
    NewSlot {
        slot,
        history,
        bias: Vec::new(),
        allowed: None,
        context: Vec::new(),
    }
}

/// The device block table of a rank whose requests hold no block: the model keeps no cache.
fn tables() -> BlockTables {
    BlockTables::new(GroupId(0), &[], &[])
}

/// The one-row commit table of `slot` in `mode`, whose row samples task zero.
fn commit(slot: u32, mode: i64) -> Vec<i64> {
    let mut table = vec![0; COMMIT_FIELDS];
    table[column::SLOT] = i64::from(slot);
    table[column::MODE] = mode;
    table
}

/// A prompt of `ids` on `slot`, a new slot, whose last token ends the prompt.
fn prompt(slot: u32, ids: &[i32]) -> Batch {
    let tokens = ids.len() as i32;

    let mut table = commit(slot, mode::EXPLICIT);
    table[column::CACHE_LENGTH] = i64::from(tokens);
    table[column::LOGICAL] = i64::from(tokens);
    table[column::SAMPLING] = 1;

    let columns = Columns {
        rows: Rows {
            positions: (0..tokens).collect(),
            seq_lens: vec![tokens],
            ..Rows::default()
        },
        input_ids: ids.to_vec(),
        draws: vec![DrawAddress::default()],
        commit: table,
        ..Columns::default()
    };
    let mut batch = batch(Phase::Prefill, Variant::PrefillGreedy, slot, columns);
    batch.new_slots = vec![new_slot(slot, Vec::new())];
    batch
}

/// One decode step of `slot` at `position`, whose input token the host states as `id`.
fn step(slot: u32, id: i32, position: i32) -> Batch {
    let columns = Columns {
        rows: Rows {
            positions: vec![position],
            seq_lens: vec![position + 1],
            ..Rows::default()
        },
        input_ids: vec![id],
        draws: vec![DrawAddress {
            key: 9,
            position: 3,
        }],
        commit: commit(slot, mode::DECODE),
        ..Columns::default()
    };
    batch(Phase::Decode, Variant::DecodeGreedy, slot, columns)
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_prefill_commits_its_first_token_and_a_decode_step_continues_from_it() -> Outcome {
    let mut bench = Bench::open()?;

    // The prompt's last token is 7, and the commit states the slot's positions.
    let output = bench.runner.execute(&prompt(2, &[5, 6, 7]), &tables())?;
    let first = &output.calls[0];
    assert_eq!(first.tokens, [next(7)]);
    assert_eq!((first.cache_computed_len, first.step_index), (3, 1));

    // The token is the most likely of the raw distribution, and its candidates are reported.
    let scores = &first.token_logprobs[0];
    assert_eq!(scores.len(), 1 + 2);
    assert_eq!((scores[0].token_id, scores[0].rank), (next(7), 1));
    assert!(scores[0].logprob < 0.0);

    // The decode reads the committed token from the slot: the id the host states is not it.
    let output = bench.runner.execute(&step(2, 0, 3), &tables())?;
    let second = &output.calls[0];
    assert_eq!(second.tokens, [next(next(7))]);
    assert_eq!((second.cache_computed_len, second.step_index), (4, 2));

    // And the step after it continues from that one.
    let output = bench.runner.execute(&step(2, 0, 4), &tables())?;
    assert_eq!(output.calls[0].tokens, [next(next(next(7)))]);
    assert!(output.timing.device_us > 0);
    Ok(())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_decode_launched_while_the_step_before_is_in_flight_continues_from_its_token() -> Outcome {
    let mut bench = Bench::open()?;
    bench.runner.execute(&prompt(2, &[5, 6, 7]), &tables())?;

    // Two decode steps are launched before either lands; the second reads the token the first
    // commits on the device. The device completes them in order, and each wakes its landing.
    let (landed, lands) = std::sync::mpsc::channel();
    let mut first = step(2, 0, 3);
    first.id = BatchId(2);
    let mut second = step(2, 0, 4);
    second.id = BatchId(3);
    for batch in [&first, &second] {
        let (landed, id) = (landed.clone(), batch.id.0);
        let resume = Arc::new(move || landed.send(id).unwrap());
        assert!(bench.runner.start(batch, &tables(), resume)?);
    }
    let wait = std::time::Duration::from_secs(30);
    assert_eq!(
        (lands.recv_timeout(wait)?, lands.recv_timeout(wait)?),
        (2, 3)
    );

    let output = bench.runner.execute(&first, &tables())?;
    let call = &output.calls[0];
    assert_eq!(call.tokens, [next(next(7))]);
    assert_eq!((call.cache_computed_len, call.step_index), (4, 2));

    let output = bench.runner.execute(&second, &tables())?;
    let call = &output.calls[0];
    assert_eq!(call.tokens, [next(next(next(7)))]);
    assert_eq!((call.cache_computed_len, call.step_index), (5, 3));
    Ok(())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_slot_whose_first_step_is_a_decode_starts_from_its_batch_s_row() -> Outcome {
    let mut bench = Bench::open()?;

    // The request generated two tokens elsewhere: its slot starts at the row's input token,
    // position and draw position.
    let mut batch = step(4, 9, 6);
    batch.new_slots = vec![new_slot(4, vec![9, 9])];

    let output = bench.runner.execute(&batch, &tables())?;
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
    bench.runner.set_page_work(Box::new(move |work| {
        let part = match work {
            PageWork::BeforeReplay(..) => "before replay",
            PageWork::AfterReplay => "after replay",
            PageWork::Ended => "ended",
            _ => "other",
        };
        seen.lock().unwrap().push(part);
        Ok(())
    }));

    let mut batch = prompt(2, &[5, 6, 7]);
    batch.columns.state_copies = vec![StateCopy {
        slot: 2,
        position: 0,
        direction: StateDirection::Restore,
    }];
    bench.runner.execute(&batch, &tables())?;

    assert_eq!(
        *calls.lock().unwrap(),
        ["before replay", "after replay", "ended"]
    );
    Ok(())
}

/// A verification of `slot`, a new slot: the committed token `ids[0]` at `position`, where
/// the request has drawn `position` tokens, followed by its drafts; the call proposes two
/// drafts after the tokens it commits.
fn verification(slot: u32, ids: &[i32], position: i32) -> Batch {
    let tokens = ids.len() as i32;

    let mut table = commit(slot, mode::VERIFY);
    table[column::LOGICAL] = i64::from(position);
    table[column::SAMPLING] = i64::from(position);
    table[column::VISIBLE] = i64::from(position);

    let columns = Columns {
        rows: Rows {
            positions: (position..position + tokens).collect(),
            seq_lens: vec![position + tokens],
            ..Rows::default()
        },
        input_ids: ids.to_vec(),
        draws: vec![DrawAddress {
            key: 9,
            position: (position + tokens) as u64,
        }],
        commit: table,
        drafts: vec![DraftWork {
            proposals: 2,
            ..DraftWork::default()
        }],
        ..Columns::default()
    };
    let mut batch = batch(Phase::Verify, Variant::DecodeGreedy, slot, columns);
    batch.new_slots = vec![new_slot(slot, Vec::new())];
    batch
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_verification_commits_the_drafts_its_samples_agree_with_and_proposes_the_next() -> Outcome {
    let mut bench = Bench::speculating(2)?;

    // The model continues 7 with 6 and 6 with 5: both drafts agree, and the sample after
    // them stands as well. The drafter proposes what follows the last committed token.
    let output = bench
        .runner
        .execute(&verification(2, &[7, 6, 5], 3), &tables())?;
    let call = &output.calls[0];
    assert_eq!(call.tokens, [next(7), next(6), next(5)]);
    assert_eq!(call.token_logprobs.len(), 3);
    assert_eq!(call.draft_token_ids, [next(next(5)), next(next(next(5)))]);

    // The slot stands behind the three tokens the verification committed.
    assert_eq!((call.cache_computed_len, call.step_index), (6, 6));

    // The second draft differs from the sample at its position: the first draft and that
    // sample are committed, and the drafts continue from the sample.
    let output = bench
        .runner
        .execute(&verification(3, &[7, 6, 9], 3), &tables())?;
    let call = &output.calls[0];
    assert_eq!(call.tokens, [next(7), next(6)]);
    assert_eq!(call.draft_token_ids, [next(next(6)), next(next(next(6)))]);
    assert_eq!((call.cache_computed_len, call.step_index), (5, 5));

    // No draft agrees: the verification commits the token a decode at its position would.
    let output = bench
        .runner
        .execute(&verification(4, &[7, 3, 2], 3), &tables())?;
    let call = &output.calls[0];
    assert_eq!(call.tokens, [next(7)]);
    assert_eq!(call.draft_token_ids, [next(next(7)), next(next(next(7)))]);
    assert_eq!((call.cache_computed_len, call.step_index), (4, 4));
    Ok(())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_prompt_of_a_speculating_runner_proposes_the_drafts_after_its_first_token() -> Outcome {
    let mut bench = Bench::speculating(2)?;

    // The prompt samples the token after it, and the drafter continues from that token on
    // the device: nothing is read back between the two graphs.
    let mut batch = prompt(2, &[5, 6, 7]);
    batch.columns.drafts = vec![DraftWork {
        proposals: 2,
        ..DraftWork::default()
    }];

    let output = bench.runner.execute(&batch, &tables())?;
    let call = &output.calls[0];
    assert_eq!(call.tokens, [next(7)]);
    assert_eq!(call.draft_token_ids, [next(next(7)), next(next(next(7)))]);

    // The verification of those drafts accepts both.
    let drafts: Vec<i32> = call.draft_token_ids.iter().map(|id| *id as i32).collect();
    let mut verified = verification(2, &[next(7) as i32, drafts[0], drafts[1]], 3);
    verified.new_slots.clear();

    let output = bench.runner.execute(&verified, &tables())?;
    assert_eq!(output.calls[0].tokens.len(), 3);
    Ok(())
}

/// The options of [`config`]'s runner with `candidates` and `drafts`, over decodes of `decode`
/// rows and prefills of `prefill` tokens.
fn options(decode: &[u32], prefill: &[u32], candidates: u32, drafts: u32) -> GraphOptions {
    GraphOptions {
        buckets: Buckets::causal_lm(decode, prefill, true),
        candidates,
        drafts,
    }
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_generation_of_other_candidates_reports_them_from_its_switch_on() -> Outcome {
    let mut bench = Bench::open()?;
    let output = bench.runner.execute(&prompt(2, &[5, 6, 7]), &tables())?;
    assert_eq!(output.calls[0].token_logprobs[0].len(), 1 + 2);

    // Prepared and captured, the shape of four candidates serves nothing until the switch.
    bench.reshape(2, &options(&[2, 4], &[8], 4, 0))?;
    let output = bench.runner.execute(&step(2, 0, 3), &tables())?;
    assert_eq!(output.calls[0].tokens, [next(next(7))]);
    assert_eq!(output.calls[0].token_logprobs[0].len(), 1 + 2);

    // From the switch on a scored row reports four candidates, and the slot continues from the
    // token the serving shape committed into it.
    assert!(bench.runner.switch_shape(2)?);
    let output = bench.runner.execute(&step(2, 0, 4), &tables())?;
    let call = &output.calls[0];
    assert_eq!(call.tokens, [next(next(next(7)))]);
    assert_eq!(call.token_logprobs[0].len(), 1 + 4);
    // Tokens `t` and `t + 16` have one logit, so the candidates rank in tied pairs.
    let ranks: Vec<u32> = call.token_logprobs[0][1..]
        .iter()
        .map(|entry| entry.rank)
        .collect();
    assert_eq!(ranks, [1, 1, 3, 3]);
    assert_eq!((call.cache_computed_len, call.step_index), (5, 3));

    // A rollback serves the kept shape again, over the same slot.
    assert!(bench.runner.rollback_shape(2)?);
    let output = bench.runner.execute(&step(2, 0, 5), &tables())?;
    assert_eq!(output.calls[0].tokens, [next(next(next(next(7))))]);
    assert_eq!(output.calls[0].token_logprobs[0].len(), 1 + 2);
    Ok(())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_generation_of_other_buckets_serves_them_whole_from_its_switch_on() -> Outcome {
    let mut bench = Bench::open()?;
    let long: Vec<i32> = (3..15).collect();

    // No bucket of eight tokens holds a prompt of twelve.
    let refused = bench.runner.execute(&prompt(2, &long), &tables());
    assert!(refused.is_err(), "{:?}", refused.map(|output| output.calls));

    // The catalog of decodes of one and eight rows and prefills of up to sixteen tokens
    // computes in a region of its own when the runner's cannot hold its peak.
    let catalog = options(&[1, 8], &[8, 16], 2, 0);
    bench.reshape(2, &catalog)?;
    assert!(bench.runner.switch_shape(2)?);
    let decodes = |rows: u32| {
        bench
            .runner
            .buckets()
            .iter()
            .any(|bucket| bucket.variant == Variant::DecodeGreedy && bucket.rows == rows)
    };
    assert!(decodes(1) && decodes(8) && !decodes(2) && !decodes(4));

    let output = bench.runner.execute(&prompt(2, &long), &tables())?;
    assert_eq!(output.calls[0].tokens, [next(14)]);
    let output = bench.runner.execute(&step(2, 0, 12), &tables())?;
    assert_eq!(output.calls[0].tokens, [next(next(14))]);

    // Committed, the shape serves on alone.
    assert!(bench.runner.commit_shape(2)?);
    let output = bench.runner.execute(&step(2, 0, 13), &tables())?;
    assert_eq!(output.calls[0].tokens, [next(next(next(14)))]);
    Ok(())
}

#[test]
#[ignore = "requires a GPU and a Build Server"]
fn a_generation_of_other_drafts_verifies_them_from_its_switch_on() -> Outcome {
    let mut bench = Bench::speculating(2)?;

    // A verification of three drafts is held by no bucket of a runner of two.
    let three = verification(2, &[7, 6, 5, 4], 3);
    assert!(bench.runner.execute(&three, &tables()).is_err());

    bench.reshape(2, &options(&[2, 4], &[8], 2, 3))?;
    assert!(bench.runner.switch_shape(2)?);
    let output = bench
        .runner
        .execute(&verification(3, &[7, 6, 5, 4], 3), &tables())?;
    let call = &output.calls[0];
    assert_eq!(call.tokens, [next(7), next(6), next(5), next(4)]);
    assert_eq!(call.token_logprobs.len(), 4);
    assert_eq!(call.draft_token_ids, [next(next(4)), next(next(next(4)))]);
    assert_eq!((call.cache_computed_len, call.step_index), (7, 7));

    // Rolled back, the runner verifies two drafts again.
    assert!(bench.runner.rollback_shape(2)?);
    let output = bench
        .runner
        .execute(&verification(4, &[7, 6, 5], 3), &tables())?;
    assert_eq!(output.calls[0].tokens, [next(7), next(6), next(5)]);
    Ok(())
}
