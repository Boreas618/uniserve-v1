//! Storage and transfers against a real driver. These tests need a CUDA
//! device (two for the cross-device case) and use a few megabytes of memory;
//! run them with `cargo test -p uniserve-worker --test gpu_storage -- --ignored`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::process::Command;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, mpsc};
use std::time::Duration;

use uniserve_core::UnitId;
use uniserve_cuda::{CopyBatch, DeviceId, SourceOrder, Stream, memset_d32, reclaim};
use uniserve_ipc::{BlockTable, CallKind, ForwardMode, Locator, TransferTransport, WorkerEndpoint};
use uniserve_worker::ffi::{Backing, DType, Device};
use uniserve_worker::lane::JobQueues;
use uniserve_worker::storage::memory::{Memory, StorageView};
use uniserve_worker::storage::{
    BlockTableStorage, BlockTables, GroupShape, LatentStorage, OutputBuffer, OutputStorage,
};
use uniserve_worker::transfer::{
    Capacity, Spans, TransportKind, TransportOptions, Transports, exports, notify,
};
use uniserve_worker::{Error, ErrorCode, EventPool, WorkerError};

/// Export on `stream` and take the locations, complete at once.
fn located(
    transports: &Transports,
    source: &Spans,
    offset: Option<&[u64]>,
    consumers: &[usize],
    host: bool,
    stream: Option<&Stream>,
    mut retain: impl FnMut(Arc<uniserve_worker::transfer::Retirement>) -> uniserve_worker::Result<()>,
) -> uniserve_worker::Result<Vec<uniserve_worker::transfer::exports::Location>> {
    exports::export(
        transports,
        source,
        offset,
        consumers,
        host,
        stream,
        &mut retain,
    )?
    .locate(&mut retain)
}

const DEVICE: i32 = 0;

/// The child half of the two-process VMM test reads this variable.
const CHILD_LOCATOR: &str = "UNISERVE_TEST_VMM_LOCATOR";

/// Serializes the tests of this binary.
///
/// Some tests hold a stream in a host function to observe the work queued
/// behind it (see [`block`]). While a stream is held, a driver call that waits
/// for all of the context's queued work (`cuMemFree`, `cuMemFreeHost`) cannot
/// return, and it keeps driver locks that every other call needs, including
/// the calls the holding test makes before it releases the stream. No other
/// test may use the driver meanwhile, so every test takes this lock first.
static DRIVER: Mutex<()> = Mutex::new(());

fn exclusive_driver() -> MutexGuard<'static, ()> {
    DRIVER.lock().unwrap_or_else(PoisonError::into_inner)
}

fn stream() -> Stream {
    Stream::new(DeviceId(DEVICE)).unwrap()
}

/// Blocks `stream` in a host function until the returned sender is used.
///
/// Until then the caller must not release device or pinned memory: the
/// release would wait on the held stream (see [`DRIVER`]). Releases queued
/// earlier are flushed first for the same reason.
fn block(stream: &Stream) -> mpsc::Sender<()> {
    reclaim::flush().unwrap();
    let (release, released) = mpsc::channel::<()>();
    stream
        .launch_host_fn(move || {
            let _ = released.recv();
        })
        .unwrap();
    release
}

/// Copy a device view's bytes to the host (drains `stream`).
fn download(view: &StorageView, stream: &Stream) -> Vec<u8> {
    let bytes = view.byte_len().unwrap();
    let host = Memory::pinned(DEVICE, bytes.max(1)).unwrap();
    let mut batch = CopyBatch::new(SourceOrder::Stream);
    batch.push(host.address(), view.address(), bytes);
    unsafe { batch.submit(stream) }.unwrap();
    stream.synchronize().unwrap();
    unsafe { host.host_bytes() }.unwrap()[..bytes].to_vec()
}

fn words(bytes: &[u8]) -> Vec<i32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| i32::from_ne_bytes(*word))
        .collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_ne_bytes(*word))
        .collect()
}

/// A device f32 tensor holding `0, 1, 2, ...`.
fn counting(shape: &[i64], memory: Arc<Memory>, stream: &Stream) -> StorageView {
    let elements: i64 = shape.iter().product();
    let values: Vec<f32> = (0..elements).map(|value| value as f32).collect();
    let mut batch = CopyBatch::new(SourceOrder::DuringCall);
    batch.push(memory.address(), values.as_ptr() as u64, values.len() * 4);
    unsafe { batch.submit(stream) }.unwrap();
    stream.synchronize().unwrap();
    StorageView::contiguous(memory as Arc<dyn Backing>, 0, DType::F32, shape).unwrap()
}

fn endpoint(address_space: &str) -> WorkerEndpoint {
    WorkerEndpoint {
        worker_id: "worker".into(),
        rank: 0,
        node: "node".into(),
        address_space: address_space.into(),
        incarnation: "incarnation".into(),
    }
}

/// What a rank's control thread does for its event pool, done by a test: a
/// lane of the test's own waits for the events of deferred releases, and
/// [`Completions::advance`] hands the pool the completions that arrived.
struct Completions {
    events: Arc<EventPool>,
    jobs: JobQueues,
}

impl Completions {
    fn serve(events: &Arc<EventPool>) -> Self {
        let jobs = JobQueues::single_lane().unwrap();
        let lane = jobs.clone();
        events.set_submitter(Some(Box::new(move |device, _, events| {
            lane.after(CallKind::Forward(ForwardMode::Decode), device, events)
        })));
        Self {
            events: Arc::clone(events),
            jobs,
        }
    }

    fn advance(&self) {
        self.events.submit_deferred().unwrap();
        while let Some((job, completion)) = self.jobs.take() {
            self.events.complete(job, completion).unwrap();
        }
    }
}

fn transports(
    kinds: &[TransportKind],
    address_space: &str,
    slot: usize,
) -> (Transports, Arc<Capacity>, Completions) {
    let capacity = Arc::new(Capacity::new(1 << 24, 8, notify).unwrap());
    let options = TransportOptions {
        source: endpoint(address_space),
        acknowledgment_slot: slot,
        host_slots: HashSet::from([0, 1]),
        cross_host_consumers: false,
        vmm_pool_bytes: 1 << 22,
    };
    let events = EventPool::new();
    let completions = Completions::serve(&events);
    let transports = Transports::new(kinds, Arc::clone(&capacity), events, &options).unwrap();
    (transports, capacity, completions)
}

/// Reap `transports` until every export credit has returned.
fn settle(transports: &Transports, capacity: &Capacity, completions: &Completions) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while capacity.used() != 0 && std::time::Instant::now() < deadline {
        completions.advance();
        transports.reap().unwrap();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(capacity.used(), 0);
}

#[test]
#[ignore = "requires a CUDA device"]
fn block_table_installs_and_clears_write_device_tables() {
    let _driver = exclusive_driver();
    let stream = stream();
    let groups = vec![
        GroupShape::new(16, 1, None).unwrap(),
        GroupShape::new(32, 2, Some(64)).unwrap(),
    ];
    let storage = BlockTableStorage::allocate(&groups, 3, 4, Some(DEVICE), Some(&stream)).unwrap();
    let mut tables = BlockTables::new(groups, 3, 4, 64).unwrap();
    assert_eq!(
        words(&download(storage.table_shapes(), &stream)),
        [0, 16, -1, 1, 32, 64, 1, 32, 64]
    );

    let install = |tables: &mut BlockTables, assignments: &[BlockTable]| {
        let update = tables.prepare(assignments).unwrap();
        storage.install(&update, Some(&stream)).unwrap();
        tables.commit(update);
    };
    install(
        &mut tables,
        &[
            BlockTable {
                request_pool_idx: 3,
                group_id: 0,
                start_page: 0,
                unit_ids: [5, 6, 7, 8].map(UnitId).to_vec(),
                allocated_tokens: 64,
            },
            BlockTable {
                request_pool_idx: 3,
                group_id: 1,
                start_page: 1,
                unit_ids: [9, 10, 11, 12].map(UnitId).to_vec(),
                allocated_tokens: 96,
            },
        ],
    );
    // A shorter row replaces the longer one completely.
    install(
        &mut tables,
        &[BlockTable {
            request_pool_idx: 3,
            group_id: 0,
            start_page: 0,
            unit_ids: [13].map(UnitId).to_vec(),
            allocated_tokens: 16,
        }],
    );
    // A rejected update writes nothing.
    assert!(
        tables
            .prepare(&[BlockTable {
                request_pool_idx: 4,
                group_id: 0,
                start_page: 0,
                unit_ids: vec![UnitId(1)],
                allocated_tokens: 1,
            }])
            .is_err()
    );

    let units = words(&download(storage.unit_tables(), &stream));
    let row =
        |table: usize, slot: usize| &units[(table * 4 + slot) * 4..(table * 4 + slot + 1) * 4];
    assert_eq!(row(0, 3), [13, 0, 0, 0]);
    assert_eq!(row(1, 3), [9, 11, 0, 0]);
    assert_eq!(row(2, 3), [10, 12, 0, 0]);
    assert_eq!(
        words(&download(storage.start_pages(), &stream)),
        [0, 0, 0, 0, 0, 0, 0, 1]
    );
    assert_eq!(
        words(&download(storage.alloced_lens(), &stream)),
        [0, 0, 0, 16]
    );

    storage.set_verified(3, 12, Some(&stream)).unwrap();
    assert_eq!(
        words(&download(storage.verified_lengths(), &stream)),
        [0, 0, 0, 12]
    );
    storage.clear(&[3], Some(&stream)).unwrap();
    tables.release(&[3]);
    assert!(
        words(&download(storage.unit_tables(), &stream))
            .iter()
            .all(|&unit| unit == 0)
    );
    assert_eq!(
        words(&download(storage.verified_lengths(), &stream)),
        [0; 4]
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn output_readback_waits_for_device_copies_and_widens_tokens() {
    let _driver = exclusive_driver();
    let stream = stream();
    let events = EventPool::default();
    let tokens = Memory::device(DEVICE, 12).unwrap();
    unsafe { memset_d32(tokens.address(), 0xFFFF_FFFE, 3, &stream) }.unwrap();
    let column = StorageView::contiguous(tokens as Arc<dyn Backing>, 0, DType::I32, &[3]).unwrap();

    let mut output = OutputBuffer::new(
        OutputStorage::allocate(Some(DEVICE), 4).unwrap(),
        1,
        &[DEVICE],
        true,
    );
    let (offset, count) = output.capture(&column, Some(&stream), &events).unwrap();
    assert!(output.readback().is_err(), "readback before sealing");
    output.seal(&[&stream], &events).unwrap();
    // The lease does not ask the driver: its fences have completed once
    // it is told so, here by the test that waited for the stream.
    assert!(output.readback().is_err(), "readback before its fences");
    stream.synchronize().unwrap();
    output.fences_completed();
    output.readback().unwrap();
    assert_eq!(output.read_tokens(offset, count).unwrap(), [-2, -2, -2]);
    let timing = output.observe(0).unwrap();
    assert!(timing.iter().all(|&micros| micros < 60_000_000));
}

#[test]
#[ignore = "requires a CUDA device"]
fn device_exports_reach_local_shared_and_channel_readers() {
    let _driver = exclusive_driver();
    let stream = stream();
    let (transports, capacity, completions) = transports(
        &[
            TransportKind::Local,
            TransportKind::Shm,
            TransportKind::Channel,
        ],
        "space",
        0,
    );
    let source = counting(&[8, 4], Memory::device(DEVICE, 128).unwrap(), &stream);
    let spans = Spans::single(source).unwrap();
    // Slot 0 is on this host, slot 7 is not: local, shm and channel all apply.
    let locations = located(
        &transports,
        &spans,
        None,
        &[0, 7],
        false,
        Some(&stream),
        |_| Ok(()),
    )
    .unwrap();
    let kinds: Vec<_> = locations
        .iter()
        .map(|(_, locator)| TransportKind::of(locator))
        .collect();
    assert_eq!(
        kinds,
        [
            TransportKind::Local,
            TransportKind::Shm,
            TransportKind::Channel
        ]
    );

    let reader = Stream::new(DeviceId(DEVICE)).unwrap();
    for (transport, locator) in &locations {
        let memory = Memory::device(DEVICE, 32).unwrap();
        let destination = Spans::single(
            StorageView::contiguous(memory as Arc<dyn Backing>, 0, DType::F32, &[2, 4]).unwrap(),
        )
        .unwrap();
        let read = transport
            .fetch(
                locator,
                Device::Cuda(DEVICE),
                Some(&destination),
                Some(&[3..5, 0..4]),
                None,
                Some(&reader),
            )
            .unwrap();
        read.wait_ready();
        let value = read.result(Some(&reader)).unwrap();
        assert_eq!(
            floats(&download(value.first(), &reader)),
            (12..20).map(|value| value as f32).collect::<Vec<_>>(),
            "{:?}",
            TransportKind::of(locator)
        );
    }

    exports::release(&locations).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while capacity.used() != 0 && std::time::Instant::now() < deadline {
        completions.advance();
        transports.reap().unwrap();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(capacity.used(), 0, "every export retires after its readers");
    transports.close().unwrap();
}

#[test]
#[ignore = "requires a CUDA device"]
fn cancelled_reads_keep_credits_until_their_copy_drains() {
    let _driver = exclusive_driver();
    let (transports, capacity, _completions) = transports(&[TransportKind::Local], "space", 0);
    let stream = stream();
    let source = counting(&[256], Memory::device(DEVICE, 1024).unwrap(), &stream);
    let locations = located(
        &transports,
        &Spans::single(source).unwrap(),
        None,
        &[],
        false,
        Some(&stream),
        |_| Ok(()),
    )
    .unwrap();
    let (transport, locator) = &locations[0];
    let exported = capacity.used();

    // The destination's submitting stream is blocked: the transport copy
    // waits on its handoff fence, so the read cannot drain yet.
    let submitter = Stream::new(DeviceId(DEVICE)).unwrap();
    let release = block(&submitter);
    let destination = Spans::single(
        StorageView::contiguous(
            Memory::device(DEVICE, 1024).unwrap() as Arc<dyn Backing>,
            0,
            DType::F32,
            &[256],
        )
        .unwrap(),
    )
    .unwrap();
    let read = transport
        .fetch(
            locator,
            Device::Cuda(DEVICE),
            Some(&destination),
            None,
            None,
            Some(&submitter),
        )
        .unwrap();
    std::thread::sleep(Duration::from_millis(50));
    read.cancel().unwrap();
    // The consumer observes the cancellation as a resource failure.
    let cancelled = read.result(Some(&submitter)).unwrap_err();
    assert_eq!(
        WorkerError::classify(cancelled, None).code,
        ErrorCode::ResourceError
    );
    assert!(!read.retired());
    assert!(
        capacity.used() > exported,
        "a running read keeps its bytes after cancellation"
    );

    release.send(()).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while !read.retired() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(read.retired());
    assert_eq!(capacity.used(), exported);
    exports::release(&locations).unwrap();
    transports.close().unwrap();
}

/// A device export that does not fit the VMM pool is carried to its consumer
/// by a host mechanism, and fails when the rank binds none: the local
/// location alone reaches no other rank.
#[test]
#[ignore = "requires a CUDA device"]
fn exports_larger_than_the_vmm_pool_need_a_host_mechanism() {
    let _driver = exclusive_driver();
    let stream = stream();
    // 5 MiB of f32, more than the 4 MiB pool `transports` configures.
    let elements: i64 = 5 << 18;
    let memory = Memory::device(DEVICE, elements as usize * 4).unwrap();
    let spans = Spans::single(counting(&[elements], memory, &stream)).unwrap();

    let (device_only, capacity, completions) = transports(
        &[TransportKind::Local, TransportKind::CudaVmm],
        "producer",
        0,
    );
    let exported = located(
        &device_only,
        &spans,
        None,
        &[1],
        false,
        Some(&stream),
        |_| Ok(()),
    );
    assert!(matches!(exported, Err(Error::Resource(_))));
    settle(&device_only, &capacity, &completions);
    device_only.close().unwrap();

    let (with_shared, capacity, completions) = transports(
        &[
            TransportKind::Local,
            TransportKind::CudaVmm,
            TransportKind::Shm,
        ],
        "producer",
        0,
    );
    let locations = located(
        &with_shared,
        &spans,
        None,
        &[1],
        false,
        Some(&stream),
        |_| Ok(()),
    )
    .unwrap();
    let kinds: Vec<_> = locations
        .iter()
        .map(|(_, locator)| TransportKind::of(locator))
        .collect();
    assert_eq!(kinds, [TransportKind::Local, TransportKind::Shm]);
    exports::release(&locations).unwrap();
    settle(&with_shared, &capacity, &completions);
    with_shared.close().unwrap();
}

/// Resident cache pages and latent pages reach a VMM consumer where they lie:
/// an export larger than the VMM pool needs no host mechanism.
#[test]
#[ignore = "requires a CUDA device"]
fn resident_pages_larger_than_the_vmm_pool_export_over_vmm() {
    let _driver = exclusive_driver();
    let stream = stream();
    let (transports, capacity, completions) = transports(
        &[TransportKind::Local, TransportKind::CudaVmm],
        "producer",
        0,
    );

    // Six tokens of both columns: 6 MiB of bfloat16, more than the 4 MiB
    // pool `transports` configures.
    let plan = Arc::new(cache_plan("bfloat16", 1 << 17));
    let planes =
        uniserve_worker::CachePlanes::allocate(plan, Some(DEVICE), 2, Some(&stream)).unwrap();
    let cache_page = Spans::single(planes.page(0, 0, 1, 0, 6).unwrap()).unwrap();

    // One page of 768 units of 2048 f32 elements, also 6 MiB.
    let latents = LatentStorage::allocate(
        1,
        2,
        768,
        2048,
        DType::F32,
        false,
        Some(DEVICE),
        Some(&stream),
    )
    .unwrap();
    let latent_page = latents.spans(0, &[1], 768).unwrap();

    for source in [cache_page, latent_page] {
        assert_eq!(source.nbytes(), 6 << 20);
        let locations = located(
            &transports,
            &source,
            None,
            &[1],
            false,
            Some(&stream),
            |_| Ok(()),
        )
        .unwrap();
        let kinds: Vec<_> = locations
            .iter()
            .map(|(_, locator)| TransportKind::of(locator))
            .collect();
        assert_eq!(kinds, [TransportKind::Local, TransportKind::CudaVmm]);

        exports::release(&locations).unwrap();
        settle(&transports, &capacity, &completions);
    }
    transports.close().unwrap();
}

/// Producer half: export a device tensor through a CUDA VMM pool chunk to
/// consumer slot 1, run this test binary as the consumer, and require the
/// export to retire only after the consumer acknowledged its read.
#[test]
#[ignore = "requires a CUDA device"]
fn vmm_exports_are_imported_by_another_process() {
    let _driver = exclusive_driver();
    let stream = stream();
    let (transports, capacity, completions) = transports(
        &[TransportKind::Local, TransportKind::CudaVmm],
        "producer",
        0,
    );
    let source = counting(&[16, 8], Memory::device(DEVICE, 512).unwrap(), &stream);
    let mut retirements = Vec::new();
    let locations = located(
        &transports,
        &Spans::single(source).unwrap(),
        None,
        &[1],
        false,
        Some(&stream),
        |retirement| {
            retirements.push(retirement);
            Ok(())
        },
    )
    .unwrap();
    let (_, locator) = locations
        .iter()
        .find(|(_, locator)| TransportKind::of(locator) == TransportKind::CudaVmm)
        .unwrap();
    assert!(matches!(
        &locator.transport,
        TransferTransport::CudaVmm { acknowledgment_offset, .. } if *acknowledgment_offset >= 0
    ));

    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "vmm_export_consumer", "--ignored", "--nocapture"])
        .env(CHILD_LOCATOR, serde_json::to_string(locator).unwrap())
        .status()
        .unwrap();
    assert!(status.success(), "the consumer process failed");

    // The consumer acknowledged and exited: revocation retires the chunk.
    exports::release(&locations).unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while capacity.used() != 0 && std::time::Instant::now() < deadline {
        completions.advance();
        transports.reap().unwrap();
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(capacity.used(), 0);
    assert!(retirements.iter().all(|retirement| retirement.succeeded()));
    transports.close().unwrap();
}

/// Consumer half, run by `vmm_exports_are_imported_by_another_process`.
#[test]
#[ignore = "runs as the consumer process of the VMM export test"]
fn vmm_export_consumer() {
    let _driver = exclusive_driver();
    let Ok(locator) = std::env::var(CHILD_LOCATOR) else {
        return;
    };
    let locator: Locator = serde_json::from_str(&locator).unwrap();
    let (transports, _, _completions) = transports(&[TransportKind::CudaVmm], "consumer", 1);
    let transport = transports.get(TransportKind::CudaVmm).unwrap();
    let stream = stream();
    let destination = Spans::single(
        StorageView::contiguous(
            Memory::device(DEVICE, 128).unwrap() as Arc<dyn Backing>,
            0,
            DType::F32,
            &[4, 8],
        )
        .unwrap(),
    )
    .unwrap();
    let read = transport
        .fetch(
            &locator,
            Device::Cuda(DEVICE),
            Some(&destination),
            Some(&[2..6, 0..8]),
            None,
            Some(&stream),
        )
        .unwrap();
    read.wait_ready();
    let value = read.result(Some(&stream)).unwrap();
    assert_eq!(
        floats(&download(value.first(), &stream)),
        (16..48).map(|value| value as f32).collect::<Vec<_>>()
    );
    while !read.retired() {
        std::thread::sleep(Duration::from_millis(1));
    }
    transports.close().unwrap();
}

/// A one-group FP8 plan: two layers in two columns, 8-token pages of two
/// 4-wide heads (8-byte rows, 64-byte planes).
fn fp8_plan() -> uniserve_worker::CachePlan {
    cache_plan("float8_e4m3fn", 4)
}

/// A one-group plan storing `dtype`: two layers in two columns, 8-token
/// pages of two heads of `head_dim` elements.
fn cache_plan(dtype: &str, head_dim: u32) -> uniserve_worker::CachePlan {
    use uniserve_worker::ffi::{DType, Handle};
    use uniserve_worker::storage::{CacheGroupPlan, GroupAxis};

    let quantized = dtype == "float8_e4m3fn";
    let element_bytes = if quantized { 1 } else { 2 };
    let plane_bytes = u64::from(8 * 2 * head_dim * element_bytes);
    // Each plane carries one initialized flag, and an FP8 plane one f32 scale.
    let metadata_bytes = if quantized { 5 } else { 1 };
    uniserve_worker::CachePlan {
        columns: 2,
        plane_bytes,
        quantized,
        unit_bytes: 2 * 2 * (plane_bytes + metadata_bytes),
        dtype: dtype.parse().unwrap(),
        groups: vec![CacheGroupPlan {
            window: None,
            page_tokens: 8,
            units_per_page: 1,
            layers: vec!["a".into(), "b".into()],
            layer_ids: vec![0, 1],
            axis: GroupAxis {
                offset: 0,
                total: 2,
            },
            num_kv_heads: 2,
            total_kv_heads: 2,
            kv_head_offset: 0,
            head_dim,
            dtype: DType::from_name(dtype).unwrap(),
            compute_dtype: DType::BF16,
        }],
        handle: Handle::none(),
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn cache_planes_reset_scratch_and_restore_snapshots() {
    let _driver = exclusive_driver();
    use uniserve_worker::storage::UnitSnapshot;

    let stream = stream();
    let plan = Arc::new(fp8_plan());
    assert_eq!(plan.info(6).unit_bytes, 276);
    let planes =
        uniserve_worker::CachePlanes::allocate(plan, Some(DEVICE), 6, Some(&stream)).unwrap();
    let views: std::collections::HashMap<_, _> = planes.views().unwrap().into_iter().collect();
    assert_eq!(views["key.values"].shape(), &[2, 6, 64]);
    assert_eq!(views["key.scale"].shape(), &[2, 6]);
    assert!(
        download(&views["value.values"], &stream)
            .iter()
            .all(|&byte| byte == 0)
    );
    assert!(
        floats(&download(&views["value.scale"], &stream))
            .iter()
            .all(|&scale| scale == 1.0)
    );

    // Dirty units 1..=3 of every field, snapshot units 2 and 3, reset the
    // scratch units natively, and restore the snapshot.
    for view in views.values() {
        let bytes = view.byte_len().unwrap();
        unsafe { uniserve_cuda::memset_d8(view.address(), 7, bytes, &stream) }.unwrap();
    }
    let snapshot = UnitSnapshot::save(&planes, &[3, 2], &stream).unwrap();
    planes.zero_leading(3, Some(&stream)).unwrap();
    let values = download(&views["key.values"], &stream);
    let unit = |column: usize, unit: usize| {
        &values[(column * 6 + unit) * 64..(column * 6 + unit + 1) * 64]
    };
    assert!(unit(1, 2).iter().all(|&byte| byte == 0));
    assert!(
        unit(0, 4).iter().all(|&byte| byte == 7),
        "units past the scratch keep their bytes"
    );
    let scales = floats(&download(&views["key.scale"], &stream));
    assert_eq!(scales[6 + 3], 1.0);

    snapshot.restore(&planes, &stream).unwrap();
    let values = download(&views["key.values"], &stream);
    let restored = &values[(6 + 2) * 64..(6 + 4) * 64];
    assert!(restored.iter().all(|&byte| byte == 7));
    assert!(
        values[(6 + 1) * 64..(6 + 2) * 64]
            .iter()
            .all(|&byte| byte == 0),
        "unit 1 was not saved"
    );
    let flags = download(&views["value.initialized"], &stream);
    assert_eq!(&flags[1..4], &[0, 7, 7]);
}
