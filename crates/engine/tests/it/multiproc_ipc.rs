//! Multiprocess launch, request lifecycle, token execution, close and rank
//! recovery.
//!
//! These tests launch real executor processes (`uniserve executor` with the
//! weightless stub model on CPU) and drive them through the engine's
//! `ExecutorGroup`, `Dispatcher` or `SchedulerHandle`, or through a raw
//! `Channel` when the executor's own protocol handling is under test.
//! Lifecycle batches (Start, Finish, Free) exercise registration, startup,
//! Info, admission, retirement, close and signals; token batches run the
//! stub's text runtime (prefill, decode, verification, sampling, logprobs and
//! device token relays). The executable comes from
//! `UNISERVE_WORKER_EXECUTABLE`, the repository environment's
//! `.venv/bin/uniserve` or the target directory's own build of it. Several
//! tests replace that executable with a
//! small wrapper script that records an executor's process id, exits a chosen rank, starts a relaunched rank as a
//! stand-in that reports changed executor info, registers under another build
//! or for an earlier launch, or holds a rank's exit until its peers have
//! closed.
//!
//! Faults are injected at the OS boundary: `/proc/thread-self/children` and
//! `/proc/<pid>/cmdline` locate a rank process, and `PausedProcess` stops,
//! resumes or kills it with signals. These are Linux interfaces, and the file
//! compiles only on Linux.

#![cfg(target_os = "linux")]

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::Context as _;
use engine::{
    CallReaders, Executor, ExecutorGroup, ExecutorGroupConfig, ExecutorProcessArgs, RankBuffer,
};
use ipc::{
    Batch, BlockTable, Bounds, BufferAllocation, CacheUnitAllocation, Call, CallCoordinates,
    CallId, CallKind, CallStatus, DType, ForwardBatch, ForwardMode, GenerationParams, NewRequest,
    RequestKey, ShapeBound, TensorRef, TransferMode,
};
use ipc::{RequestId, SamplingParams, UnitId};

const WORLD_SIZE: usize = 2;
const QUEUE_DEPTH: usize = 2;
/// The stub model's image start token, the successor of 1001.
const STUB_IMAGE_START: u32 = 151_670;
/// Waits up to 30 seconds in total for the executor's next partial or
/// terminal batch result, polling again after progress-only wakes.
fn poll_logical(executor: &mut impl Executor) -> anyhow::Result<Option<engine::BatchResult>> {
    let deadline = std::time::Instant::now() + Duration::from_secs(30);
    loop {
        let Some(remaining) = deadline.checked_duration_since(std::time::Instant::now()) else {
            return Ok(None);
        };
        if let Some(result) = executor.poll(remaining)? {
            return Ok(Some(result));
        }
    }
}

/// Builds a batch with one token call of kind `mode` on the `model` component.
///
/// The call reads `tokens` after `prefix_length` cached positions and writes
/// its sampled token to an `I64` token output produced by `call_id`, with
/// generation `4 * call_id.batch_id + 1` (saturating). `admission`, when
/// given, becomes the batch's `Start` command and supplies the request pool
/// slot; without it the call targets slot 1. The block table maps the request
/// to the single page `unit`, which the batch allocates only when
/// `prefix_length` is zero; a continuation passes the same page and a nonzero
/// prefix.
#[expect(
    clippy::too_many_arguments,
    reason = "every field of the one-call batch a test sends"
)]
fn token_batch(
    batch_id: u64,
    collective_seq: u64,
    request_key: RequestKey,
    admission: Option<NewRequest>,
    call_id: CallId,
    mode: CallKind,
    tokens: &[u32],
    unit: UnitId,
    prefix_length: u32,
) -> Batch {
    let request_pool_idx = admission.as_ref().map_or(1, |value| value.request_pool_idx);
    let token_output = TensorRef {
        request_key,
        producer_call_id: call_id,
        output_index: 0,
        generation: (call_id.batch_id as u32)
            .saturating_mul(4)
            .saturating_add(1),
        dtype: DType::I64,
        shape_bound: ShapeBound::default(),
    };
    let mut call = Call::new(request_key, call_id, mode);
    // These requests hold no cached prefix beyond their own tokens, so the
    // logical sequence and the initialized cache advance together.
    call.coordinates = CallCoordinates {
        logical_position: prefix_length,
        cache_visible_len: prefix_length,
        cache_computed_len: prefix_length,
    };
    call.token_output = Some(token_output);
    call.input_token_ids = tokens.to_vec();
    call.component = "model".into();
    call.bounds = Bounds {
        max_tokens: tokens.len().max(1) as u32,
        ..Bounds::default()
    };
    let input_length = tokens.len() as u32;
    let mut batch = Batch::new(batch_id, admission.into_iter().collect(), vec![call]);
    batch.collective_seq = collective_seq;
    batch.block_tables = vec![BlockTable {
        request_pool_idx,
        group_id: 0,
        start_page: 0,
        unit_ids: vec![unit],
        allocated_tokens: prefix_length + input_length,
    }];
    batch.new_cache_units = if prefix_length == 0 {
        vec![CacheUnitAllocation {
            request_pool_idx,
            group_id: 0,
            unit_ids: vec![unit],
            copies: Vec::new(),
        }]
    } else {
        Vec::new()
    };
    batch.forward = ForwardBatch {
        call_indices: vec![0],
        request_pool_indices: vec![request_pool_idx],
        seq_lens: vec![prefix_length + input_length.max(1)],
        query_lens: vec![input_length.max(1)],
        write_cache: vec![true],
    };
    batch
}

/// Encodes one registration frame per rank of `args`'s group, as `stated`
/// gives it for each rank, in the hexadecimal form a wrapper script embeds.
fn registration_frames(
    args: &ExecutorProcessArgs,
    stated: impl Fn(u32) -> ipc::Registration,
) -> anyhow::Result<Vec<String>> {
    (0..args.ranks.len())
        .map(|rank| {
            let frame = ipc::Frame::encode(&stated(rank as u32))?;
            Ok(frame
                .as_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>())
        })
        .collect()
}

/// The engine and its ranks must be one build, since the message types are the
/// only definition of their protocol and their encoding is positional. A rank
/// states its build when it registers, and a launch fails by naming a rank of
/// another one, including another build of the same package version.
#[test]
fn launch_refuses_a_worker_of_another_build() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    // The stand-in registers as the rank it was started as, for the launch
    // it was started for, under a build of the engine's package version that
    // the engine is not, and then waits for the engine to close.
    let other = format!("{}+0123456789abcdef", env!("CARGO_PKG_VERSION"));
    anyhow::ensure!(
        other != ipc::VERSION,
        "the stand-in's build is the engine's"
    );
    let mut args = worker_group_args();
    let frames = registration_frames(&args, |rank| ipc::Registration::Worker {
        worker_id: args.worker_id.clone(),
        rank,
        generation: 1,
        version: other.clone(),
    })?;
    let directory = tempfile::tempdir()?;
    let wrapper = directory.path().join("worker");
    std::fs::write(
        &wrapper,
        format!(
            "#!/usr/bin/env python3\nimport socket, sys\nframes = {frames:?}\nrank = int(sys.argv[sys.argv.index('--rank') + 1])\nengine = sys.argv[sys.argv.index('--engine') + 1]\nconnection = socket.socket(socket.AF_UNIX)\nconnection.connect(engine.removeprefix('unix:'))\nconnection.sendall(bytes.fromhex(frames[rank]))\nconnection.settimeout(60)\nwhile connection.recv(4096):\n    pass\n"
        ),
    )?;
    std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;

    args.executable = wrapper;
    let reported = match ExecutorGroup::spawn(args) {
        Ok(_) => anyhow::bail!("a rank of another build was accepted"),
        Err(error) => format!("{error:#}"),
    };
    assert!(
        reported.contains(&other) && reported.contains(ipc::VERSION),
        "launch failure did not name both builds: {reported}"
    );
    Ok(())
}


/// A rank placed on another host is started by that host's daemon, which
/// joined the leader; with no daemon there, the launch fails naming the host
/// whose ranks cannot start rather than waiting for them or starting them in
/// the wrong place. What a placement computes is asserted without a launch
/// (`core::tests::a_placement_shares_ranks_across_the_named_hosts`).
#[test]
fn a_launch_names_the_host_whose_daemon_is_absent() {
    let spawned = ExecutorGroup::spawn(ExecutorProcessArgs {
        executable: worker_executable(),
        daemon: crate::support::daemon(),
        model: String::new(),
        host: "compute-0".into(),
        ranks: ExecutorGroupConfig::placed(
            &["compute-1".to_owned()],
            "cpu",
            WORLD_SIZE,
            2,
            ExecutorGroupConfig::single_component("model", WORLD_SIZE),
        )
        .ranks,
        components: ExecutorGroupConfig::single_component("model", WORLD_SIZE),
        stub: true,
        ..ExecutorProcessArgs::default()
    });
    let Err(error) = spawned else {
        panic!("a rank on a host no daemon runs on must not launch");
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("compute-1"),
        "the refusal must name the host no daemon runs on: {message}"
    );
}

/// Launch arguments for a two-rank stub group on the local host's CPU.
///
/// Both ranks hold the single `model` component, with `QUEUE_DEPTH` batches
/// in flight. Tests adjust the returned value (for example components, executable or transfer edges)
/// before spawning.
///
/// Each group has a name of its own. The tests share one daemon, which files
/// a process by its group, rank and launch: two groups of one name would be
/// one group to it, where a launch replaces the other's ranks and a stop ends
/// them.
fn worker_group_args() -> ExecutorProcessArgs {
    static GROUPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let group = GROUPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let executable = worker_executable();
    let config = ExecutorProcessArgs {
        stub: true,
        // The stub rank encodes many small images, so it keeps a larger
        // encoder cache than a served model's default.
        encoder_cache_entries: 1024,
        ..ExecutorProcessArgs::default()
    };
    ExecutorProcessArgs {
        executable,
        worker_id: format!("group-{group}"),
        daemon: crate::support::daemon(),
        model: String::new(),
        ranks: ExecutorGroupConfig::placed(
            &["localhost".to_owned()],
            "cpu",
            WORLD_SIZE,
            2,
            ExecutorGroupConfig::single_component("model", WORLD_SIZE),
        )
        .ranks,
        components: ExecutorGroupConfig::single_component("model", WORLD_SIZE),
        queue_depth: QUEUE_DEPTH,
        max_total_tokens: Some(4096),
        block_size: Some(16),
        max_batch_calls: 256,
        max_batch_tokens: 256,
        attention_backend: "torch".into(),
        transfer: Default::default(),
        ..config
    }
}

fn worker_executable() -> PathBuf {
    crate::support::uniserve_executable()
}

/// Places a batch's buffer spans at the same offsets on every rank of the
/// groups these tests launch, which share one layout of their pools.
fn on_every_rank(allocations: Vec<BufferAllocation>) -> Vec<RankBuffer> {
    RankBuffer::on_every_rank(&allocations, WORLD_SIZE as u32)
}

/// Submission of a batch that carries its buffer spans, which a group takes
/// beside the batch as a rank-by-rank placement.
trait SubmitPlaced {
    fn submit_placed(&mut self, batch: Batch) -> Result<(), engine::BatchSubmitError>;
}

impl SubmitPlaced for ExecutorGroup {
    fn submit_placed(&mut self, mut batch: Batch) -> Result<(), engine::BatchSubmitError> {
        let spans = on_every_rank(std::mem::take(&mut batch.buffer_allocations));
        self.submit_batch(batch, &CallReaders::new(), &spans)
    }
}

/// A greedy text admission for engine 1 that ignores EOS, starting at
/// position zero in request pool slot `request_pool_idx`.
fn text_admission(
    request_id: u64,
    request_epoch: u64,
    request_pool_idx: u32,
) -> anyhow::Result<NewRequest> {
    Ok(NewRequest::new(
        RequestKey::new(1, RequestId(request_id), request_epoch),
        request_pool_idx,
        Some(GenerationParams {
            sampling: SamplingParams {
                temperature: 0.0,
                ignore_eos: true,
                ..SamplingParams::default()
            },
            negative_token_ids: Vec::new(),
            finish_token_ids: Vec::new(),
            initial_position: 0,
            generated_token_ids: Vec::new(),
            adapter: 0,
        }),
        None,
        0,
    )?)
}

// Products and transfers: tensor products published by one component or
// executor and imported by another, cache exports, and the retirement of work
// that reads a failed or unreachable product.

/// Device products cross executor processes over CUDA VMM with their values:
/// the source publishes two requests' transition flags (one false, one
/// true), and target calls gated by them run exactly when the imported flag
/// is true, whether they read the import directly or a copy a tensor
/// transfer made into the target's own relay.
#[test]
#[ignore = "requires a CUDA device"]
fn device_products_cross_workers_over_cuda_vmm() -> anyhow::Result<()> {
    use engine::{Dispatcher, ExecutorGroupId};

    let retire = Retire::default();
    let edges = "source->target=cuda_vmm";
    let mut executor = Dispatcher::try_new(
        vec![
            (
                ExecutorGroupId("source".into()),
                cuda_worker(&retire, "source", edges, None)?,
            ),
            (
                ExecutorGroupId("target".into()),
                cuda_worker(&retire, "target", edges, None)?,
            ),
        ],
        engine::TransferConfig::parse(edges)?,
    )?;

    // The stub's head maps 7 to 1000 and 1001 to its image start token,
    // which these calls declare a transition: request 95's flag is false,
    // request 96's true.
    let mut flags = Vec::new();
    for (batch_id, request_id, prompt) in [(1, 95, 7), (3, 96, 1001)] {
        let admission = text_admission(request_id, 1, request_id as u32 - 94)?;
        let mut prefill = token_batch(
            batch_id,
            batch_id,
            admission.request_key,
            Some(admission),
            CallId::new(batch_id, 0),
            CallKind::Forward(ForwardMode::Prefill),
            &[prompt],
            UnitId(request_id as u32 - 94),
            0,
        );
        let flag = scalar_flag(prefill.calls[0].request_key, CallId::new(batch_id, 0), 3);
        prefill.calls[0].transition_output = Some(flag.clone());
        prefill.calls[0].sampling_state = Some(ipc::SamplingState {
            transition_token_ids: vec![STUB_IMAGE_START],
            ..Default::default()
        });
        executor.submit(place(prefill, "source", Vec::new()))?;
        let prefilled = poll_logical(&mut executor)?.context("source prefill did not complete")?;
        assert_eq!(prefilled.results[0].output.status, CallStatus::Ok);

        // The source republishes the flag as a product other executors read;
        // the target imports it over the source's one edge.
        let export = TensorRef {
            producer_call_id: CallId::new(batch_id + 1, 0),
            generation: 2,
            ..flag.clone()
        };
        let publish = product_transfer(
            flag.request_key,
            CallId::new(batch_id + 1, 0),
            "model",
            &flag,
            &export,
        );
        let mut publish = Batch::new(batch_id + 1, Vec::new(), vec![publish]);
        publish.buffer_allocations.push(BufferAllocation {
            buffer: export.buffer_id(),
            offset: 256 * batch_id,
            bytes: export.max_bytes(),
        });
        executor.submit(place(publish, "source", Vec::new()))?;
        let published = poll_logical(&mut executor)?.context("flag export did not complete")?;
        assert_eq!(published.results[0].output.status, CallStatus::Ok);
        flags.push(export);
    }

    // The same requests on the target, gated directly by their imported
    // flags.
    for (batch_id, (request_id, flag, expected)) in [
        (95, &flags[0], CallStatus::Predicated),
        (96, &flags[1], CallStatus::Ok),
    ]
    .into_iter()
    .enumerate()
    {
        let admission = text_admission(request_id, 1, request_id as u32 - 94)?;
        let status = gated_prefill(
            &mut executor,
            21 + batch_id as u64,
            admission.request_key,
            Some(admission),
            flag,
            0,
        )?;
        assert_eq!(status, expected, "request {request_id}");
    }

    // The true flag copied into the target's relay of its request gates the
    // request's next call the same way.
    let owner = flags[1].request_key;
    let copy = TensorRef {
        producer_call_id: CallId::new(23, 0),
        generation: 3,
        ..flags[1].clone()
    };
    let transfer = product_transfer(owner, CallId::new(23, 0), "model", &flags[1], &copy);
    let mut transfer = Batch::new(23, Vec::new(), vec![transfer]);
    transfer.buffer_allocations.push(BufferAllocation {
        buffer: copy.buffer_id(),
        offset: 0,
        bytes: copy.max_bytes(),
    });
    executor.submit(place(transfer, "target", Vec::new()))?;
    let copied = poll_logical(&mut executor)?.context("target transfer did not complete")?;
    assert_eq!(copied.results[0].output.status, CallStatus::Ok);
    assert_eq!(
        gated_prefill(&mut executor, 24, owner, None, &copy, 1)?,
        CallStatus::Ok
    );
    executor.close()?;
    Ok(())
}

/// A request prefilled on an executor with a BF16 cache continues on an executor
/// with an FP16 cache: its cache crosses the processes over CUDA VMM and is
/// converted into the target's representation by the cache's conversion.
#[test]
#[ignore = "requires a CUDA device"]
fn a_converting_install_copies_the_cache_another_worker_exported() -> anyhow::Result<()> {
    use engine::{Dispatcher, ExecutorGroupId};

    let retire = Retire::default();
    let edges = "source->target=cuda_vmm";
    let mut executor = Dispatcher::try_new(
        vec![
            (
                ExecutorGroupId("source".into()),
                cuda_worker(&retire, "source", edges, None)?,
            ),
            (
                ExecutorGroupId("target".into()),
                cuda_worker(
                    &retire,
                    "target",
                    edges,
                    Some(ipc::PrefixCacheDtype::Float16),
                )?,
            ),
        ],
        engine::TransferConfig::parse(edges)?,
    )?;
    let admission = text_admission(97, 1, 1)?;
    let key = admission.request_key;
    let prefill = token_batch(
        1,
        1,
        key,
        Some(admission.clone()),
        CallId::new(1, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7, 8, 9],
        UnitId(1),
        0,
    );
    let tables = prefill.block_tables.clone();
    executor.submit(place(prefill, "source", Vec::new()))?;
    let prefilled = poll_logical(&mut executor)?.context("source prefill did not complete")?;
    let exported = ipc::BufferId {
        owner: key,
        producer_call_id: CallId::new(2, 0),
        output_index: 0,
        generation: 2,
    };
    let mut export = Call::new(
        key,
        CallId::new(2, 0),
        CallKind::Transfer(TransferMode::CacheExport),
    );
    export.coordinates = coordinates_after(&prefilled.results[0].output);
    export.cache_output = Some(exported);
    export.bounds.max_transfer_bytes = 1 << 20;
    let mut publish = Batch::new(2, Vec::new(), vec![export]);
    publish.block_tables = tables;
    executor.submit(place(publish, "source", Vec::new()))?;
    let published = poll_logical(&mut executor)?.context("cache export did not complete")?;
    assert_eq!(published.results[0].output.status, CallStatus::Ok);
    let transfer = published.results[0]
        .output
        .cache_output
        .clone()
        .context("cache export has no transfer")?;
    assert_eq!(transfer.exported_extent, 3);
    assert!(transfer.tensors().all(|tensor| {
        tensor
            .locations
            .iter()
            .any(|location| matches!(location.transport, ipc::TransferTransport::CudaVmm { .. }))
    }));

    let mut install = Call::new(
        key,
        CallId::new(3, 0),
        CallKind::Transfer(TransferMode::CacheInstall),
    );
    install.cache_input = Some(exported);
    install.cache_output = Some(ipc::BufferId {
        owner: key,
        producer_call_id: CallId::new(3, 0),
        output_index: 0,
        generation: 3,
    });
    install.bounds.max_transfer_bytes = 1 << 20;
    let mut target = Batch::new(3, vec![admission], vec![install]);
    target.block_tables = vec![BlockTable {
        request_pool_idx: 1,
        group_id: 0,
        start_page: 0,
        unit_ids: vec![UnitId(6)],
        allocated_tokens: 3,
    }];
    target.new_cache_units = vec![CacheUnitAllocation {
        request_pool_idx: 1,
        group_id: 0,
        unit_ids: vec![UnitId(6)],
        copies: Vec::new(),
    }];
    executor.submit(place(target, "target", Vec::new()))?;
    let installation = poll_logical(&mut executor)?.context("cache install did not complete")?;
    assert_eq!(installation.results[0].output.status, CallStatus::Ok);
    assert_eq!(installation.results[0].output.cache_visible_len, 3);

    let mut decode = token_batch(
        4,
        4,
        key,
        None,
        CallId::new(4, 0),
        CallKind::Forward(ForwardMode::Decode),
        &[1000],
        UnitId(6),
        3,
    );
    decode.calls[0].coordinates.logical_position = 3;
    executor.submit(place(decode, "target", Vec::new()))?;
    let decoded = poll_logical(&mut executor)?.context("target decode did not complete")?;
    assert_eq!(decoded.results[0].output.status, CallStatus::Ok);
    assert_eq!(decoded.results[0].output.committed_tokens, vec![1001]);
    assert_eq!(decoded.results[0].output.cache_visible_len, 4);
    executor.close()?;
    Ok(())
}

/// A real model's request continues on another executor from the prefix cache it
/// installed and the token the source sampled: the target's greedy
/// continuation of a prompt equals the source's token for token, since the
/// installed pages hold the source's bytes. Cache and token cross over shared
/// storage and over CUDA VMM. Over CUDA VMM the source's pages are read where
/// they lie, and the token, a relay scalar outside exportable storage, is
/// copied into the source's export pool beside its cache pool of a served
/// model's scale. `UNISERVE_KV_INSTALL_MODEL` names the checkpoint and
/// `UNISERVE_KV_INSTALL_PROMPT` the prompt's token ids (a JSON list).
#[test]
#[ignore = "requires a CUDA device and the checkpoint UNISERVE_KV_INSTALL_MODEL names"]
fn a_model_continues_greedily_from_installed_cache() -> anyhow::Result<()> {
    // One mechanism at a time: each pair of executors holds two copies of the
    // model on the device.
    for edges in ["source->target=posix_shm", "source->target=cuda_vmm"] {
        continue_greedily_from_installed_cache(edges).with_context(|| edges.to_owned())?;
    }
    Ok(())
}

/// Prefill a prompt on the source, export its cache and its sampled token to
/// the target over `edges`, install both there, and compare both executors'
/// greedy continuations, the target's starting from the imported token.
fn continue_greedily_from_installed_cache(edges: &str) -> anyhow::Result<()> {
    use engine::{Dispatcher, ExecutorGroupId};

    let model = std::env::var("UNISERVE_KV_INSTALL_MODEL")?;
    let prompt: Vec<u32> = serde_json::from_str(&std::env::var("UNISERVE_KV_INSTALL_PROMPT")?)?;
    let retire = Retire::default();
    let spawn = |worker_id: &str| -> anyhow::Result<ExecutorGroup> {
        let mut args = single_worker_args(worker_id, QUEUE_DEPTH);
        for rank in &mut args.ranks {
            rank.device = "cuda:0".into();
        }
        // The model's own block size, attention selection and component.
        args.stub = false;
        args.model = model.clone();
        args.block_size = None;
        args.max_batch_tokens = 8192;
        args.attention_backend = "auto".into();
        args.components = ExecutorGroupConfig::single_component("model", 1);
        args.transfer = engine::TransferConfig::parse(edges)?;
        retire.add(&args);
        ExecutorGroup::spawn(args)
    };
    let mut executor = Dispatcher::try_new(
        vec![
            (ExecutorGroupId("source".into()), spawn("source")?),
            (ExecutorGroupId("target".into()), spawn("target")?),
        ],
        engine::TransferConfig::parse(edges)?,
    )?;
    let admission = text_admission(101, 1, 1)?;
    let key = admission.request_key;
    let n = prompt.len() as u32;
    let prefill = token_batch(
        1,
        1,
        key,
        Some(admission.clone()),
        CallId::new(1, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &prompt,
        UnitId(1),
        0,
    );
    let tables = prefill.block_tables.clone();
    executor.submit(place(prefill, "source", Vec::new()))?;
    let prefilled = poll_logical(&mut executor)?.context("source prefill did not complete")?;
    let first = prefilled.results[0].output.committed_tokens[0];

    let exported = ipc::BufferId {
        owner: key,
        producer_call_id: CallId::new(2, 0),
        output_index: 0,
        generation: 2,
    };
    let mut export = Call::new(
        key,
        CallId::new(2, 0),
        CallKind::Transfer(TransferMode::CacheExport),
    );
    export.coordinates = coordinates_after(&prefilled.results[0].output);
    export.cache_output = Some(exported);
    export.bounds.max_transfer_bytes = 1 << 30;
    let mut publish = Batch::new(2, Vec::new(), vec![export]);
    publish.block_tables = tables;
    executor.submit(place(publish, "source", Vec::new()))?;
    let published = poll_logical(&mut executor)?.context("cache export did not complete")?;
    assert_eq!(published.results[0].output.status, CallStatus::Ok);

    let mut install = Call::new(
        key,
        CallId::new(3, 0),
        CallKind::Transfer(TransferMode::CacheInstall),
    );
    install.cache_input = Some(exported);
    install.cache_output = Some(ipc::BufferId {
        owner: key,
        producer_call_id: CallId::new(3, 0),
        output_index: 0,
        generation: 3,
    });
    install.bounds.max_transfer_bytes = 1 << 30;
    let mut target = Batch::new(3, vec![admission], vec![install]);
    target.block_tables = vec![BlockTable {
        request_pool_idx: 1,
        group_id: 0,
        start_page: 0,
        unit_ids: vec![UnitId(7)],
        allocated_tokens: n,
    }];
    target.new_cache_units = vec![CacheUnitAllocation {
        request_pool_idx: 1,
        group_id: 0,
        unit_ids: vec![UnitId(7)],
        copies: Vec::new(),
    }];
    executor.submit(place(target, "target", Vec::new()))?;
    let installed = poll_logical(&mut executor)?.context("cache install did not complete")?;
    assert_eq!(installed.results[0].output.status, CallStatus::Ok);

    // Greedy continuations of the same prefix on both executors.
    let mut tokens = std::collections::BTreeMap::new();
    let mut last = None;
    for (worker, unit, base) in [("source", 1, 10), ("target", 7, 20)] {
        let mut token = first;
        let mut sequence = Vec::new();
        for step in 0..4_u32 {
            let batch_id = u64::from(base + step);
            let (decoded, sampled) =
                decode_step(&mut executor, worker, key, batch_id, token, unit, n + step)?;
            token = decoded.committed_tokens[0];
            sequence.push(token);
            if worker == "source" {
                last = Some((token, sampled));
            }
        }
        tokens.insert(worker, sequence);
    }
    let (last, sampled) = last.context("the source decoded")?;

    // The source publishes its last token and the target copies it into its
    // own relay; one more step on each executor continues from that token, the
    // target's reading it from the device.
    let published = TensorRef {
        producer_call_id: CallId::new(14, 0),
        generation: 14 * 4 + 1,
        ..sampled.clone()
    };
    let publish = token_transfer(key, CallId::new(14, 0), "model", &sampled, &published);
    executor.submit(place(
        Batch::new(14, Vec::new(), vec![publish]),
        "source",
        Vec::new(),
    ))?;
    let exported = poll_logical(&mut executor)?.context("token export did not complete")?;
    assert_eq!(exported.results[0].output.status, CallStatus::Ok);
    let imported = TensorRef {
        producer_call_id: CallId::new(24, 0),
        generation: 24 * 4 + 1,
        ..published.clone()
    };
    let copy = token_transfer(key, CallId::new(24, 0), "model", &published, &imported);
    executor.submit(place(
        Batch::new(24, Vec::new(), vec![copy]),
        "target",
        Vec::new(),
    ))?;
    let copied = poll_logical(&mut executor)?.context("token import did not complete")?;
    assert_eq!(copied.results[0].output.status, CallStatus::Ok);

    let position = n + 4;
    let (source, _) = decode_step(&mut executor, "source", key, 15, last, 1, position)?;
    let mut decode = decode_batch(key, 25, 0, 7, position);
    decode.calls[0].input_token_ids.clear();
    decode.calls[0].predicate = Some(imported);
    executor.submit(place(decode, "target", Vec::new()))?;
    let target = poll_logical(&mut executor)?.context("decode did not complete")?;
    assert_eq!(target.results[0].output.status, CallStatus::Ok);
    for (worker, output) in [("source", &source), ("target", &target.results[0].output)] {
        if let Some(sequence) = tokens.get_mut(worker) {
            sequence.push(output.committed_tokens[0]);
        }
    }
    assert_eq!(tokens["source"], tokens["target"]);
    executor.close()?;
    Ok(())
}

/// A one-token decode of `key` at `position` on `executor`, in batch and call
/// `batch_id`, over the request's unit `unit`.
fn decode_batch(key: RequestKey, batch_id: u64, token: u32, unit: u32, position: u32) -> Batch {
    let mut decode = token_batch(
        batch_id,
        batch_id,
        key,
        None,
        CallId::new(batch_id, 0),
        CallKind::Forward(ForwardMode::Decode),
        &[token],
        UnitId(unit),
        position,
    );
    decode.calls[0].coordinates.logical_position = position;
    decode
}

/// Run [`decode_batch`] on `executor` and return its output and the token
/// product it published.
fn decode_step(
    executor: &mut engine::Dispatcher,
    worker: &str,
    key: RequestKey,
    batch_id: u64,
    token: u32,
    unit: u32,
    position: u32,
) -> anyhow::Result<(ipc::CallOutput, TensorRef)> {
    let decode = decode_batch(key, batch_id, token, unit, position);
    let sampled = decode.calls[0]
        .token_output
        .clone()
        .context("a decode samples a token")?;
    executor.submit(place(decode, worker, Vec::new()))?;
    let decoded = poll_logical(executor)?.context("decode did not complete")?;
    assert_eq!(decoded.results[0].output.status, CallStatus::Ok);
    Ok((decoded.results[0].output.clone(), sampled))
}

/// Launch arguments of a one-rank stub executor on the first visible CUDA
/// device, with a cache of `prefix_cache_dtype` (the model's dtype by default)
/// and the transfer `edges`.
fn cuda_worker(
    retire: &Retire,
    worker_id: &str,
    edges: &str,
    prefix_cache_dtype: Option<ipc::PrefixCacheDtype>,
) -> anyhow::Result<ExecutorGroup> {
    let mut args = single_worker_args(worker_id, QUEUE_DEPTH);
    for rank in &mut args.ranks {
        rank.device = "cuda:0".into();
    }
    args.kv_cache_dtype = prefix_cache_dtype;
    args.transfer = engine::TransferConfig::parse(edges)?;
    retire.add(&args);
    ExecutorGroup::spawn(args)
}

/// The deployments of a test's CUDA executors, retired at the host's Storage
/// Server once the test's executors are gone: their device storage outlives
/// them there, and no later launch claims a test's deployment again. A test
/// declares it before its executors, so that it drops after them.
#[derive(Default)]
struct Retire(std::sync::Mutex<Vec<String>>);

impl Retire {
    fn add(&self, args: &ExecutorProcessArgs) {
        self.0.lock().unwrap().push(args.deployment.clone());
    }
}

impl Drop for Retire {
    fn drop(&mut self) {
        let socket = storage::socket::default_socket();
        for deployment in self.0.lock().unwrap().drain(..) {
            // The server notices the head's disconnect on its next turn.
            for _ in 0..500 {
                let retired = storage::Client::connect(&socket, storage::Role::Operator)
                    .and_then(|mut client| client.retire(deployment.clone()));
                if retired.is_ok() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        }
    }
}

/// A one-element U8 flag that `producer` publishes as output `index`.
fn scalar_flag(request_key: RequestKey, producer: CallId, index: u16) -> TensorRef {
    TensorRef {
        request_key,
        producer_call_id: producer,
        output_index: index,
        generation: 1,
        dtype: DType::U8,
        shape_bound: ShapeBound::default(),
    }
}

/// Prefill one prompt token of `key` on the target after its first
/// `prefix` tokens, gated by `flag`, admitting the request with
/// `admission` when given; returns the call's status.
fn gated_prefill(
    executor: &mut engine::Dispatcher,
    batch_id: u64,
    key: RequestKey,
    admission: Option<NewRequest>,
    flag: &TensorRef,
    prefix: u32,
) -> anyhow::Result<CallStatus> {
    let slot = admission
        .as_ref()
        .map_or(2, |admission| admission.request_pool_idx);
    let mut prefill = token_batch(
        batch_id,
        batch_id,
        key,
        admission,
        CallId::new(batch_id, 0),
        CallKind::Forward(ForwardMode::Prefill),
        &[7],
        UnitId(slot + 10),
        prefix,
    );
    prefill.calls[0].predicate = Some(flag.clone());
    prefill.block_tables[0].request_pool_idx = slot;
    prefill.forward.request_pool_indices = vec![slot];
    for allocation in &mut prefill.new_cache_units {
        allocation.request_pool_idx = slot;
    }
    executor.submit(place(prefill, "target", Vec::new()))?;
    let result = poll_logical(executor)?.context("gated prefill did not complete")?;
    let output = &result.results[0].output;
    if output.status == CallStatus::Ok {
        assert_eq!(output.committed_tokens, vec![1000]);
    }
    Ok(output.status)
}

/// Environment of a rank process `media_worker` serves: the launch arguments
/// (JSON), the executor info template (a path), the media case, the path
/// that receives the published media name, and, in the `planes` case, the
/// directory that receives the cache table each launch descriptor carried.
const MEDIA_WORKER_ARGS: &str = "UNISERVE_TEST_MEDIA_WORKER_ARGS";
const MEDIA_WORKER_INFO: &str = "UNISERVE_TEST_MEDIA_WORKER_INFO";
const MEDIA_WORKER_CASE: &str = "UNISERVE_TEST_MEDIA_RESPONSE";
const MEDIA_WORKER_NAME: &str = "UNISERVE_TEST_MEDIA_NAME";
const MEDIA_WORKER_TABLES: &str = "UNISERVE_TEST_MEDIA_TABLES";
/// The bytes a `media_worker` process publishes.
const MEDIA_PAYLOAD: &[u8] = b"generated media content";

#[test]
fn media_storage_is_owned_through_worker_result_validation() -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt as _;

    // The ranks are `media_worker` processes speaking the rank protocol. Each
    // answers Info with the simulated rank's executor info, completes every
    // call it receives and attaches a shared-memory media payload to a
    // completion, selected by case: a valid payload from rank 0
    // ("retained"), a payload on a call identity the batch never carried
    // ("unknown-call"), a payload from rank 1 ("rank-output"), and a payload
    // declaring one byte more than it stores ("short-storage").
    let template = tempfile::tempdir()?;
    let info_path = template.path().join("info.json");
    std::fs::write(
        &info_path,
        serde_json::to_vec(engine::SimEngine::new().mut_info_for_test())?,
    )?;
    let executable = serde_json::to_string(&std::env::current_exe()?)?;
    for case in ["retained", "unknown-call", "rank-output", "short-storage"] {
        let directory = tempfile::tempdir()?;
        let wrapper = directory.path().join("worker");
        let name_path = directory.path().join("media-name");
        // `ExecutorGroup` runs `<executable> <args>`, so the wrapper runs
        // this test binary's `media_worker` with those arguments.
        std::fs::write(
            &wrapper,
            format!(
                "#!/usr/bin/env python3\nimport json, os, sys\nenv = dict(os.environ, {MEDIA_WORKER_ARGS}=json.dumps(sys.argv[1:]), {MEDIA_WORKER_INFO}={info}, {MEDIA_WORKER_CASE}={case:?}, {MEDIA_WORKER_NAME}={name})\nos.execve({executable}, [{executable}, '--exact', 'multiproc_ipc::media_worker', '--ignored', '--nocapture', '--test-threads', '1'], env)\n",
                info = serde_json::to_string(&info_path)?,
                name = serde_json::to_string(&name_path)?,
            ),
        )?;
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700))?;
        let mut args = worker_group_args();
        args.executable = wrapper;
        let mut worker = ExecutorGroup::spawn(args)?;
        let admission = text_admission(81, 1, 1)?;
        let batch = token_batch(
            1,
            1,
            admission.request_key,
            Some(admission),
            CallId::new(1, 0),
            CallKind::Forward(ForwardMode::Prefill),
            &[7, 8],
            UnitId(1),
            0,
        );
        worker.submit_placed(batch)?;

        // A correlation or rank-ownership failure may fail the poll and must
        // deliver no result. A short extent still delivers the call's
        // result, with its `media` failed.
        let rejected = matches!(case, "unknown-call" | "rank-output");
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut terminal = false;
        let mut results = Vec::new();
        while std::time::Instant::now() < deadline && !terminal {
            match worker.poll_batch(Duration::from_millis(100)) {
                Ok(Some(report)) => {
                    terminal = true;
                    results.extend(report.results);
                }
                Ok(None) => {}
                Err(error) => {
                    anyhow::ensure!(
                        rejected,
                        "unexpected media result failure in {case}: {error:#}"
                    );
                    terminal = true;
                }
            }
        }
        assert!(terminal, "media result did not retire in {case}");
        let published_name = std::fs::read_to_string(&name_path)?;
        // The receiver claims the POSIX name even when correlation, rank
        // ownership, or extent validation rejects the result. Retained
        // mappings keep the bytes.
        assert!(
            // SAFETY: the rank handed off an immutable, completed export.
            unsafe { ipc::SharedMedia::open(published_name.trim(), 1) }.is_err(),
            "unclaimed media in {case}"
        );
        if rejected {
            assert!(
                results.is_empty(),
                "invalid rank output was delivered in {case}"
            );
        } else {
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].output.status, CallStatus::Ok);
            if case == "short-storage" {
                assert!(results[0].media.is_err());
            } else {
                let media = std::sync::Arc::clone(
                    results[0]
                        .media
                        .as_ref()
                        .map_err(|error| anyhow::anyhow!(error.clone()))?
                        .as_ref()
                        .context("media result has no storage")?,
                );
                // The claimed mapping outlives both the rank group and the
                // result that delivered it.
                worker.close()?;
                drop(results);
                assert_eq!(media.as_bytes(), MEDIA_PAYLOAD);
                continue;
            }
        }
        worker.close()?;
    }
    Ok(())
}

/// A rank process of `media_storage_is_owned_through_worker_result_validation`:
/// it registers its channel endpoint like an executor, answers Info with
/// the template executor info under its own identity, completes every call it
/// receives as the stub's prefill of a request's first prompt tokens would,
/// and attaches the case's media payload to a completion.
///
/// In the `planes` case it stands in for a rank whose cache planes split
/// into segments instead, and attaches no media: it records the cache table
/// its launch descriptor carried (`<generation>-<rank>.json` under
/// `MEDIA_WORKER_TABLES`), holds the leading segments the table names, as a
/// relaunched executor lays its planes out, and answers
/// `uniserve.release_cache` by giving back the segments from the one it
/// names, with the count it then holds.
#[test]
#[ignore = "runs as a rank process of the media result test"]
fn media_worker() -> anyhow::Result<()> {
    use ipc::{
        ArtifactHandle, BatchOutput, CallOutput, Channel, LaunchDescriptor, MediaOutput,
        Registration, MainProcessRequest, MainProcessResponse,
    };

    let Ok(arguments) = std::env::var(MEDIA_WORKER_ARGS) else {
        return Ok(());
    };
    let arguments: Vec<String> = serde_json::from_str(&arguments)?;
    let argument = |name: &str| {
        arguments
            .iter()
            .position(|argument| argument == name)
            .and_then(|index| arguments.get(index + 1))
            .with_context(|| format!("a rank launch names {name}"))
    };

    // Register with the engine as the rank this process was started as, and
    // take the launch descriptor it answers with.
    let mut server = Channel::connect(argument("--engine")?)?;
    server.send(&Registration::Worker {
        worker_id: argument("--executor-id")?.clone(),
        rank: argument("--rank")?.parse()?,
        generation: argument("--generation")?.parse()?,
        version: ipc::VERSION.into(),
    })?;
    let descriptor: LaunchDescriptor = loop {
        if let Some(descriptor) = server.try_recv()? {
            break descriptor;
        }
        server.wait(Duration::from_secs(60))?;
    };
    let case = std::env::var(MEDIA_WORKER_CASE)?;
    let mut info: ipc::MainProcessInfo =
        serde_json::from_slice(&std::fs::read(std::env::var(MEDIA_WORKER_INFO)?)?)?;
    info.endpoint.address_space = format!("rust:{}", std::process::id());
    info.endpoint.incarnation = format!("media-rank-{}-{}", descriptor.rank, std::process::id());
    let planes = case == "planes";
    if planes {
        std::fs::write(
            Path::new(&std::env::var(MEDIA_WORKER_TABLES)?).join(format!(
                "{}-{}.json",
                argument("--generation")?,
                descriptor.rank
            )),
            serde_json::to_vec(&descriptor.segments)?,
        )?;
        if let (Some(table), Some(cache)) = (&descriptor.segments, info.prefix_cache.as_mut()) {
            cache.held_segments = table.held;
        }
    }

    // Media attaches to rank 1's completion only in "rank-output".
    let media_rank = if planes {
        u32::MAX
    } else {
        u32::from(case == "rank-output")
    };
    loop {
        let request = loop {
            if let Some(request) = server.try_recv::<MainProcessRequest>()? {
                break request;
            }
            server.wait(Duration::from_secs(1))?;
        };
        let response = match request {
            MainProcessRequest::Info { message_id } => MainProcessResponse::Info {
                message_id,
                info: info.clone(),
            },
            MainProcessRequest::Close { message_id } => {
                server.send(&MainProcessResponse::Ok { message_id })?;
                return Ok(());
            }
            MainProcessRequest::Promote { message_id } => MainProcessResponse::Ok { message_id },
            MainProcessRequest::Hook {
                message_id,
                name,
                payload,
            } if planes && name == "uniserve.release_cache" => {
                let payload: serde_json::Value = serde_json::from_str(&payload)?;
                let from = payload["from"]
                    .as_u64()
                    .context("release_cache names no segment")?;
                let cache = info
                    .prefix_cache
                    .as_mut()
                    .context("the template reports no KV cache")?;
                cache.held_segments = cache.held_segments.min(u32::try_from(from)?);
                MainProcessResponse::Hooked {
                    message_id,
                    result: serde_json::json!({ "held": cache.held_segments }).to_string(),
                }
            }
            MainProcessRequest::Hook { message_id, .. } => MainProcessResponse::Error {
                message_id,
                error: ipc::MainProcessResponseError {
                    message: "this worker runs no hooks".into(),
                    code: None,
                    fatal: false,
                    phase: None,
                    route: None,
                    calls: Vec::new(),
                },
            },
            MainProcessRequest::Submit { message_id, batch } => {
                let mut completions: Vec<CallOutput> = batch
                    .calls
                    .iter()
                    .map(|call| {
                        let length = call.input_token_ids.len() as u32;
                        CallOutput {
                            request_key: call.request_key,
                            call_id: call.call_id,
                            status: CallStatus::Ok,
                            error_code: None,
                            error_message: None,
                            timing_counters: Default::default(),
                            cache_visible_len: call.coordinates.cache_visible_len + length,
                            cache_computed_len: call.coordinates.cache_computed_len + length,
                            step_index: 0,
                            committed_tokens: vec![1000],
                            token_logprobs: Vec::new(),
                            prompt_logprobs: Vec::new(),
                            candidate_logprobs: Vec::new(),
                            finish_flags: Default::default(),
                            media_output: None,
                            cache_output: None,
                            draft_token_ids: Vec::new(),
                            latent_layouts: Vec::new(),
                        }
                    })
                    .collect();
                if descriptor.rank == media_rank
                    && let Some(completion) = completions.first_mut()
                {
                    let locator = ipc::MediaSource::publish(MEDIA_PAYLOAD)?.into_locator();
                    std::fs::write(std::env::var(MEDIA_WORKER_NAME)?, &locator.name)?;
                    completion.media_output = Some(MediaOutput {
                        handle: ArtifactHandle::PosixShm { name: locator.name },
                        bytes: locator.bytes + u64::from(case == "short-storage"),
                    });
                    if case == "unknown-call" {
                        completion.call_id.request_index = 1000;
                    }
                }
                MainProcessResponse::Result {
                    message_id,
                    result: BatchOutput {
                        batch_id: batch.batch_id,
                        completions,
                        products: Vec::new(),
                        worker_exec_us: None,
                        forward_stats: None,
                    },
                }
            }
        };
        server.send(&response)?;
    }
}

/// Launch arguments for a one-executor stub group `worker_id` that holds
/// every stub component, with `queue_depth` batches in flight.
fn single_worker_args(worker_id: &str, queue_depth: usize) -> ExecutorProcessArgs {
    let mut args = worker_group_args();
    let components = std::collections::BTreeMap::from([
        (
            "model".into(),
            ipc::ComponentConfig::parallel(vec![0], Default::default()),
        ),
        (
            "vision_encoder".into(),
            ipc::ComponentConfig::parallel(vec![0], Default::default()),
        ),
    ]);
    let binding =
        ExecutorGroupConfig::placed(&["localhost".to_owned()], "cpu", 1, queue_depth, components);
    args.ranks = binding.ranks.clone();
    args.components = binding.components;
    args.queue_depth = queue_depth;
    args.worker_id = worker_id.into();
    args
}

/// Lifts a single-call executor batch into the logical form the executor
/// accepts, placing it on `executor` with the batch's tables, pages, buffers
/// and the external `inputs` it reads.
fn place(mut batch: Batch, worker: &str, inputs: Vec<ipc::TensorExport>) -> engine::ExecutionBatch {
    use engine::{ExecutionBatch, RequestPlacement, ExecutorGroupId};

    assert_eq!(batch.calls.len(), 1);
    let call = batch.calls.remove(0);
    let mut inputs = inputs;
    inputs.extend(batch.input_products);
    ExecutionBatch::new(
        batch.batch_id,
        vec![(
            call,
            RequestPlacement {
                worker: ExecutorGroupId(worker.into()),
                request_pool_idx: None,
                block_tables: batch.block_tables,
                new_cache_units: batch.new_cache_units,
                forward: batch.forward,
                latent: batch.latent_params.into_iter().next(),
                decode: batch.decode_ranges.into_iter().next(),
                buffers: on_every_rank(batch.buffer_allocations),
                readers: Default::default(),
                page_loads: Vec::new(),
            },
        )],
        batch.commands,
        inputs,
    )
}

/// The coordinates a call entering after `output` executes at, for a request
/// whose completions advance its logical position with its visible cache.
fn coordinates_after(output: &ipc::CallOutput) -> CallCoordinates {
    CallCoordinates {
        logical_position: output.cache_visible_len,
        cache_visible_len: output.cache_visible_len,
        cache_computed_len: output.cache_computed_len,
    }
}

/// A tensor transfer on `component` that copies the relayed token `input`
/// into the relayed token `output`.
fn token_transfer(
    request_key: RequestKey,
    call_id: CallId,
    component: &str,
    input: &TensorRef,
    output: &TensorRef,
) -> Call {
    let mut call = Call::new(
        request_key,
        call_id,
        CallKind::Transfer(TransferMode::Tensor),
    );
    call.component = component.into();
    call.token_input = Some(input.clone());
    call.token_output = Some(output.clone());
    call.bounds = Bounds {
        max_transfer_bytes: input.max_bytes(),
        ..Bounds::default()
    };
    call
}

/// A tensor transfer on `component` that copies the generic product
/// `input` into the generic product `output`.
fn product_transfer(
    request_key: RequestKey,
    call_id: CallId,
    component: &str,
    input: &TensorRef,
    output: &TensorRef,
) -> Call {
    let mut call = Call::new(
        request_key,
        call_id,
        CallKind::Transfer(TransferMode::Tensor),
    );
    call.component = component.into();
    call.inputs = vec![input.clone()];
    call.outputs = vec![output.clone()];
    call.bounds = Bounds {
        max_transfer_bytes: input.max_bytes(),
        ..Bounds::default()
    };
    call
}
