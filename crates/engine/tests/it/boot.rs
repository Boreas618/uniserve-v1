//! The in-process boot: a main process that holds its own devices reads the
//! facts its launch states, derives its configuration from them
//! (`EngineConfig::of`), and serves over the executor's own batch path
//! (`Submissions::over_batches`) — with no executor group launched and no device.
//!
//! `SchedulerHandle::boot` is that one composition. This test states what it
//! has to produce: a handle the epoch's policy commands reach, over a pool whose
//! inventory is the only statement of what it serves, and one request served
//! through it to a token the pool itself sampled.
//!
//! The pool stands in for the executor only: it is handed the mechanism's batch
//! and answers its calls. What the boot derives — the group, the budgets, the
//! cache geometry, the tokenizer's bounds — is the code under test.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use engine::{
    BatchCapacity, Event, GenerationConstraint, GenerationRequest, RankFacts, SchedulerHandle,
    Stage,
};
use executor::batch::{
    Batch, BatchOutput, Batches, CallOutput, Inventory, Placement, RowState, Submission,
    SubmissionId,
};
use executor::completion::Outcome;
use ipc::{
    ComponentConfig, ImageParams, SamplingParams, TimingCounters, MainProcessEndpoint, MainProcessInfo,
};

/// How long a command answer may take.
///
/// Generous because the first one is the whole installation of a policy: the
/// tree's `fcfs` module is built for `wasm32-unknown-unknown` on first use and
/// then loaded and instantiated, on a machine that runs this suite's tests in
/// parallel. The bound is the harness's, not the mechanism's; a mechanism that
/// stopped answering would still fail here, only later.
const ANSWER: Duration = Duration::from_secs(60);

/// How long a request may take to emit its first token.
const FIRST_TOKEN: Duration = Duration::from_secs(20);

/// The token this pool samples for the `index`th call of every batch.
const SAMPLED: u32 = 1000;

/// Blocks until a command answers, and answers its result.
fn answer<T>(receiver: crossbeam_channel::Receiver<Result<T, String>>) -> Result<T, String> {
    match receiver.recv_timeout(ANSWER) {
        Ok(result) => result,
        Err(error) => panic!("the scheduler did not answer: {error}"),
    }
}

/// One rank of a text pool: prefill and decode, eight request rows, and the
/// batch bounds the launch stated.
fn rank(host: &str) -> MainProcessInfo {
    MainProcessInfo {
        endpoint: MainProcessEndpoint {
            host: host.to_owned(),
            address_space: format!("{host}-space"),
            incarnation: format!("{host}-rank"),
        },
        supported_calls: vec![
            ipc::CallKind::Forward(ipc::ForwardMode::Prefill),
            ipc::CallKind::Forward(ipc::ForwardMode::Decode),
        ],
        request_slots: 8,
        max_batch_calls: 16,
        max_batch_tokens: 512,
        // A text pool serves token generation, so it advertises the paged KV
        // pool every such executor has (`MainProcessInfo::default`): an executor with no
        // prefix cache is refused a token request at submission, which is the
        // rule for a media-only rank and not for this one.
        buffer_pool_bytes: 1 << 20,
        ..MainProcessInfo::default()
    }
}

/// A text pool of one rank named `model`: what the executor reports of itself.
fn pool() -> Inventory {
    Inventory {
        group: "model".to_owned(),
        ranks: vec![rank("gpu-a")],
        placement: Placement {
            queue_depth: 1,
            ranks: 1,
            components: std::collections::BTreeMap::from([(
                ipc::DEFAULT_COMPONENT.to_owned(),
                ComponentConfig {
                    ranks: vec![0],
                    parallel_config: Default::default(),
                    units_per_rank: 1,
                    weights: Default::default(),
                    distribution: None,
                },
            )]),
        },
        max_model_len: 4096,
    }
}

/// The facts one main process's launch file states for a text deployment of one
/// rank: the launch's own values, and the model's bounds for a tokenizer whose
/// end of text is the simulation's.
fn facts() -> RankFacts {
    RankFacts {
        model: Some("boot-model".to_owned()),
        dtype: Some(engine::ModelDtype::BFloat16),
        world: Some(0),
        group: Some("model".to_owned()),
        rank: Some(0),
        world_size: Some(1),
        devices: Some(vec!["cpu".to_owned()]),
        hosts: Some(vec!["localhost".to_owned()]),
        max_running_requests: Some(4),
        // The budgets the pool below is built for, stated rather than left to a
        // default: a plan may not exceed what the epoch was built for, so a
        // budget tighter than the pool's would refuse the very batch a request
        // needs and read as "the policy served nothing".
        max_batch: Some(16),
        max_num_batched_tokens: Some(512),
        block_size: Some(64),
        kv_cache_dtype: Some(ipc::PrefixCacheDtype::BFloat16),
        graph_tiers: Some(vec![1, 8]),
        bos: Some(0),
        eos: Some(vec![151645]),
        end_of_image: Some(0),
        generation_limits: Some(engine::EngineConfig::sim("boot").generation_limits.clone()),
        max_model_len: Some(4096),
        ..RankFacts::default()
    }
}

/// A greedy text request of `prompt` tokens that generates at most `max_tokens`.
fn text_request(id: u64, prompt: usize, max_tokens: usize) -> GenerationRequest {
    GenerationRequest {
        request_id: ipc::RequestId(id),
        prompt_token_ids: vec![7; prompt],
        multimodal_inputs: Default::default(),
        negative_prompt_token_ids: Vec::new(),
        constraint: GenerationConstraint::UndOnly,
        sampling: SamplingParams::default(),
        image: ImageParams::default(),
        max_images: 1,
        max_und_tokens: max_tokens,
        include_stop_token: false,
        stop_strings: Vec::new(),
        stop_token_ids: Vec::new(),
        priority: 0,
        cache: Default::default(),
        image_generation: Default::default(),
        readout: Vec::new(),
        generated_token_ids: Vec::new(),
        adapter: 0,
        generated_images: Vec::new(),
        trajectory: Default::default(),
    }
}

/// A pool that serves every batch with the values the mechanism reads back.
struct Serving {
    inventory: Vec<Inventory>,
    /// The batches the mechanism handed over, so a test can see the boot got as
    /// far as executing work.
    submitted: Mutex<usize>,
    next: Mutex<u64>,
}

impl Serving {
    fn new(inventory: Vec<Inventory>) -> Self {
        Self {
            inventory,
            submitted: Mutex::new(0),
            next: Mutex::new(1),
        }
    }

    /// The batch's calls, each answered as a completed call that sampled
    /// [`SAMPLED`] and stands one token past the cache extent the mechanism
    /// wrote into the batch's own rows.
    fn serve(&self, batch: Batch) -> Submission {
        let calls = batch
            .calls
            .iter()
            .enumerate()
            .map(|(index, call)| {
                let seen = batch
                    .columns
                    .rows
                    .seq_lens
                    .get(index)
                    .copied()
                    .unwrap_or(0)
                    .max(0) as u32;
                CallOutput::completed(
                    *call,
                    vec![SAMPLED + index as u32],
                    RowState {
                        step_index: seen + 1,
                        cache_computed_len: seen + 1,
                        cache_visible_len: seen + 1,
                    },
                )
            })
            .collect::<Vec<_>>();
        let mut next = self.next.lock().expect("the pool's submissions");
        let submission = Submission::completed(
            SubmissionId(*next),
            Outcome::Success(BatchOutput {
                calls,
                timing: TimingCounters::default(),
            }),
        );
        *next += 1;
        submission
    }
}

impl Batches for Serving {
    fn submit(&self, batch: Batch) -> Result<Submission, executor::Error> {
        *self.submitted.lock().expect("the pool's batches") += 1;
        Ok(self.serve(batch))
    }

    fn shadow(&self, batch: Batch) -> Result<Submission, executor::Error> {
        Ok(self.serve(batch))
    }

    fn inventory(&self) -> &[Inventory] {
        &self.inventory
    }
}

/// The `fcfs` program, instantiated for the capacity this pool was built for:
/// a policy declares its own limits and the mechanism refuses a declaration
/// above the ceilings the ranks were started for.
fn policy_program(pool: &Serving) -> engine::PolicyProgram {
    let rank = pool
        .inventory()
        .first()
        .and_then(|pool| pool.ranks.first())
        .expect("the pool states its ranks");
    let ceilings = engine::ceilings_of(BatchCapacity::of(rank));
    engine::PolicyProgram {
        backend: "fcfs".to_owned(),
        revision: 1,
        arguments: engine::Parameters {
            chunked_prefill_size: Some(ceilings.chunked_prefill_size),
            max_queued_requests: ceilings.max_queued_requests,
            max_batch: Some(ceilings.max_batch),
            max_num_batched_tokens: Some(ceilings.max_num_batched_tokens),
            prefetch_min_tokens: Some(ceilings.prefetch_min_tokens),
            timeout: Some(ceilings.timeout),
            ..Default::default()
        },
        stated: Default::default(),
    }
}

/// The epoch's first policy: prepared, then switched in on the handle's own
/// command surface, which the launched path's is.
fn install(handle: &SchedulerHandle, pool: &Serving) {
    let source: Arc<dyn engine::PolicySource + Send + Sync> =
        Arc::new(engine::PolicyBytes::new(crate::support::policy_module()));
    answer(handle.precopy_policy(policy_program(pool), source)).expect("the policy is precopied");
    answer(handle.stage(Stage::Switchover)).expect("the policy serves");
}

#[test]
fn the_facts_boot_a_main_process_that_serves_a_request() {
    let serving = Arc::new(Serving::new(vec![pool()]));
    let batches: Arc<dyn Batches> = serving.clone();
    // The boot's whole shape: the facts of a launch file, the executor's batch
    // path, one handle. Nothing is launched: the pool is the rank.
    let handle = SchedulerHandle::boot(&facts(), batches).expect("the facts boot a scheduler");

    // The handle is the launched path's: its commands answer, and a stage waits
    // for a boundary.
    let progress = handle
        .progress()
        .recv_timeout(ANSWER)
        .expect("the scheduler answers with its progress");
    assert_eq!(progress.step, 0, "no step ran before a policy served");

    install(&handle, &serving);
    // The kernels this epoch serves on, as the control thread's switchover arm
    // publishes it.
    handle
        .generation()
        .store(3, std::sync::atomic::Ordering::Relaxed);

    let mut events = handle
        .submit(text_request(1, 8, 4))
        .expect("the request is submitted");

    // A token arrives: the booted scheduler admitted the request, the policy
    // planned it, the batch path served it, and the report reached the request.
    let deadline = Instant::now() + FIRST_TOKEN;
    let mut first_token = None;
    while first_token.is_none() && Instant::now() < deadline {
        match events.try_recv() {
            Ok(Event::TextToken { id, .. }) => first_token = Some(id),
            Ok(Event::TextTokens { ids }) => first_token = ids.first().copied(),
            Ok(_) => {}
            Err(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    let first = first_token.expect("the booted scheduler served a token");
    assert_eq!(
        first, SAMPLED,
        "the token is the one this pool sampled, not an invented one"
    );
    assert!(
        *serving.submitted.lock().expect("the pool's batches") > 0,
        "the executor was handed the work the facts' budgets admit"
    );
}
