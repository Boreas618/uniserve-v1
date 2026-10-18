//! Native transports: exports, reads and retirement.
//!
//! The tests without an `ignore` attribute run on host memory. The others
//! need a CUDA device and use a few megabytes of device memory; run them
//! with `cargo test -p uniserve-worker --test transports -- --ignored`.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::ops::Range;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, mpsc};
use std::time::{Duration, Instant};

use uniserve_core::PrefixCacheDtype;
use uniserve_cuda::{CopyBatch, DeviceId, SourceOrder, Stream, reclaim};
use uniserve_ipc::{Locator, TensorTransfer, TransferTransport, WorkerEndpoint};
use uniserve_worker::ffi::{Backing, DType, MemoryKind};
use uniserve_worker::storage::{
    CacheGroupPlan, CachePlan, CachePlanes, GroupAxis, LatentPlan, LatentStorage, Memory,
    StorageView,
};
use uniserve_worker::transfer::exports::{self, Location};
use uniserve_worker::transfer::{
    Notify, Read, Retirement, Spans, TransferCapacity, Transport, TransportKind, TransportOptions,
    Transports, fetch,
};
use uniserve_worker::{Error, ErrorCode, EventPool, Result, WorkerError};

/// Export `source` on `stream` and take its locations, completing at once.
fn located(
    transports: &Transports,
    source: &Spans,
    offset: Option<&[u64]>,
    consumers: &[usize],
    stream: Option<&Stream>,
    mut retain: impl FnMut(Arc<Retirement>) -> Result<()>,
) -> Result<Vec<Location>> {
    exports::export(
        transports,
        source,
        offset,
        consumers,
        false,
        stream,
        &mut retain,
    )?
    .locate(&mut retain)
}

/// Read `region` of `locator` into `destination` on `device` (borrow it in
/// place without a destination), issued on `stream` and started at once.
fn fetch(
    transport: &Transport,
    locator: &Locator,
    device: MemoryKind,
    destination: Option<&Spans>,
    region: Option<&[Range<u64>]>,
    stream: Option<&Stream>,
) -> Result<Arc<Read>> {
    let read = transport.issue_fetch(locator, device, destination, region, None, stream)?;
    read.start()?;
    Ok(read)
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

/// The transports of `kinds` for consumer `slot` in `address_space`, with
/// `bytes` of transfer capacity, a VMM export pool of `vmm_pool_bytes` and
/// slots 0 and 1 on this host.
fn transports(
    kinds: &[TransportKind],
    address_space: &str,
    slot: usize,
    bytes: u64,
    vmm_pool_bytes: u64,
) -> Result<(Transports, Arc<TransferCapacity>, Arc<EventPool>)> {
    let capacity = Arc::new(TransferCapacity::new(bytes, 8)?);
    let events = EventPool::new();
    let options = TransportOptions {
        source: endpoint(address_space),
        acknowledgment_slot: slot,
        host_slots: HashSet::from([0, 1]),
        cross_host_consumers: false,
        vmm_pool_bytes,
    };
    let transports = Transports::new(kinds, Arc::clone(&capacity), Arc::clone(&events), &options)?;
    Ok((transports, capacity, events))
}

/// Host transports of `kinds` with `bytes` of transfer capacity.
fn host_transports(
    kinds: &[TransportKind],
    bytes: u64,
) -> Result<(Transports, Arc<TransferCapacity>)> {
    let (transports, capacity, _) = transports(kinds, "space", 0, bytes, 1 << 20)?;
    Ok((transports, capacity))
}

/// Retire released exports and reads until every transfer byte returned:
/// without a control loop taking job completions, the deferred releases
/// wait for their events on the host.
fn settle(transports: &Transports, capacity: &TransferCapacity, events: &EventPool) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(10);
    while capacity.used() != 0 && Instant::now() < deadline {
        events.drain()?;
        transports.reap()?;
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(capacity.used(), 0, "every export retires after its readers");
    Ok(())
}

/// A host tensor of `shape` holding `0, 1, 2, ...` as f32.
fn counting(shape: &[i64]) -> Result<Spans> {
    let elements: i64 = shape.iter().product();
    let memory = Memory::host(elements as usize * 4);
    for index in 0..elements {
        // SAFETY: fresh host memory of `elements` floats owned by the test.
        unsafe {
            (memory.address() as *mut f32)
                .add(index as usize)
                .write(index as f32)
        };
    }
    Spans::single(StorageView::contiguous(
        memory as Arc<dyn Backing>,
        0,
        DType::F32,
        shape,
    )?)
}

fn zeros(shape: &[i64]) -> Result<Spans> {
    let elements: i64 = shape.iter().product();
    Spans::single(StorageView::contiguous(
        Memory::host(elements as usize * 4) as Arc<dyn Backing>,
        0,
        DType::F32,
        shape,
    )?)
}

fn values(spans: &Spans) -> Vec<f32> {
    let mut values = Vec::new();
    for view in spans.views() {
        assert!(view.is_contiguous());
        // SAFETY: completed host reads; no copy writes the view any more.
        values.extend_from_slice(unsafe {
            std::slice::from_raw_parts(view.address() as *const f32, view.numel() as usize)
        });
    }
    values
}

fn kinds(locations: &[Location]) -> Vec<TransportKind> {
    locations
        .iter()
        .map(|(_, locator)| TransportKind::of(locator))
        .collect()
}

#[test]
fn local_reads_copy_or_borrow_and_retire_after_release() -> Result<()> {
    let (transports, capacity) = host_transports(&[TransportKind::Local], 1 << 12)?;
    let source = counting(&[4, 3])?;
    let mut retirements = Vec::new();
    let locations = located(&transports, &source, None, &[], None, |retirement| {
        retirements.push(retirement);
        Ok(())
    })?;
    assert_eq!(locations.len(), 1);
    let (transport, locator) = &locations[0];
    assert_eq!(capacity.used(), 48);

    // A copy into a Rust-owned destination of one region.
    let destination = zeros(&[2, 2])?;
    let read = fetch(
        transport,
        locator,
        MemoryKind::Host,
        Some(&destination),
        Some(&[1..3, 1..3]),
        None,
    )?;
    read.wait_ready();
    assert_eq!(values(&read.result(None::<&Stream>)?), [4.0, 5.0, 7.0, 8.0]);

    // A borrowed read sees the producer's storage in place.
    let borrowed = fetch(transport, locator, MemoryKind::Host, None, None, None)?;
    let view = borrowed.result(None::<&Stream>)?;
    assert_eq!(view.first().address(), source.first().address());

    transport.release(locator)?;
    assert!(
        !retirements[0].done(),
        "an open borrowed read holds the export"
    );
    borrowed.close()?;
    transports.reap()?;
    // The copy read's credit and bytes return when its task retires.
    while !read.retirement().succeeded() {
        std::thread::yield_now();
    }
    assert!(retirements[0].succeeded());
    assert_eq!(capacity.used(), 0);
    transports.close()
}

#[test]
fn shared_memory_and_channel_reads_stage_whole_payloads() -> Result<()> {
    let (transports, capacity) =
        host_transports(&[TransportKind::Shm, TransportKind::Channel], 1 << 12)?;
    let source = counting(&[3, 2])?;
    let locations = located(&transports, &source, Some(&[2, 0]), &[0], None, |_| Ok(()))?;
    // Slot 0 is on this host: shared memory reaches it, the channel does not.
    assert_eq!(kinds(&locations), [TransportKind::Shm]);
    let (transport, locator) = &locations[0];
    assert_eq!(locator.offset, [2, 0]);

    let destination = zeros(&[3, 2])?;
    let read = fetch(
        transport,
        locator,
        MemoryKind::Host,
        Some(&destination),
        None,
        None,
    )?;
    read.wait_ready();
    assert_eq!(
        values(&read.result(None::<&Stream>)?),
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    );

    // Borrowing whole rows reads in place; the region is logical.
    let rows = transport.borrow(locator, Some(&[3..5, 0..2]))?;
    assert_eq!(rows.nbytes(), 16);
    drop(rows);

    // Slot 2 is on another host: only the rank channel reaches it, and its
    // bytes travel inside the locator.
    let remote = located(&transports, &source, None, &[2], None, |_| Ok(()))?;
    let (channel, exported) = &remote[0];
    assert_eq!(TransportKind::of(exported), TransportKind::Channel);
    let destination = zeros(&[2, 2])?;
    let read = fetch(
        channel,
        exported,
        MemoryKind::Host,
        Some(&destination),
        Some(&[1..3, 0..2]),
        None,
    )?;
    read.wait_ready();
    assert_eq!(values(&read.result(None::<&Stream>)?), [2.0, 3.0, 4.0, 5.0]);

    transport.release(locator)?;
    while capacity.used() != 0 {
        transports.reap()?;
        std::thread::yield_now();
    }
    transports.close()
}

#[test]
fn fetches_cover_regions_from_shards_and_reject_missing_coverage() -> Result<()> {
    let (transports, _) = host_transports(&[TransportKind::Local], 1 << 12)?;
    // Two shards of a [4, 2] tensor: rows 0..2 and rows 2..4.
    let first = counting(&[2, 2])?;
    let second = counting(&[2, 2])?;
    let mut locations = located(&transports, &first, Some(&[0, 0]), &[], None, |_| Ok(()))?;
    let shard = located(&transports, &second, Some(&[2, 0]), &[], None, |_| Ok(()))?;
    let tensor = TensorTransfer {
        shape: vec![4, 2],
        locations: vec![locations[0].1.clone(), shard[0].1.clone()],
    };

    let destination = zeros(&[2, 2])?;
    let reads = fetch::fetch_tensor(
        &tensor,
        &destination,
        &transports,
        &[1..3, 0..2],
        None::<&Stream>,
        |_| Ok(()),
    )?;
    fetch::start(&reads)?;
    assert_eq!(reads.len(), 2);
    for read in &reads {
        read.wait_ready();
        read.result(None::<&Stream>)?;
    }
    assert_eq!(values(&destination), [2.0, 3.0, 0.0, 1.0]);

    let partial = TensorTransfer {
        shape: vec![4, 2],
        locations: vec![locations[0].1.clone()],
    };
    let untouched = counting(&[2, 2])?;
    assert!(
        fetch::fetch_tensor(
            &partial,
            &untouched,
            &transports,
            &[1..3, 0..2],
            None::<&Stream>,
            |_| Ok(())
        )
        .is_err()
    );
    assert_eq!(
        values(&untouched),
        [0.0, 1.0, 2.0, 3.0],
        "no read starts without full coverage"
    );

    locations.extend(shard);
    exports::release(&locations)?;
    transports.close()
}

#[test]
fn a_failed_export_revokes_every_accepted_location() -> Result<()> {
    // Room for one export only: the second mechanism's reservation fails.
    let (transports, capacity) = host_transports(&[TransportKind::Local, TransportKind::Shm], 24)?;
    let source = counting(&[2, 3])?;
    let accepted = AtomicUsize::new(0);
    let result = located(&transports, &source, None, &[0], None, |_| {
        accepted.fetch_add(1, Ordering::Relaxed);
        Ok(())
    });
    assert!(result.is_err());
    assert_eq!(accepted.load(Ordering::Relaxed), 1);
    while capacity.used() != 0 {
        transports.reap()?;
        std::thread::yield_now();
    }
    transports.close()
}

#[test]
fn observers_of_one_read_run_once_each() -> Result<()> {
    let (transports, _) = host_transports(&[TransportKind::Local], 1 << 12)?;
    let source = counting(&[2])?;
    let locations = located(&transports, &source, None, &[], None, |_| Ok(()))?;
    let (transport, locator) = &locations[0];
    let destination = zeros(&[2])?;
    let read = fetch(
        transport,
        locator,
        MemoryKind::Host,
        Some(&destination),
        None,
        None,
    )?;
    let counted = || {
        let calls = Arc::new(AtomicUsize::new(0));
        let observer: Notify = {
            let calls = Arc::clone(&calls);
            Arc::new(move || {
                calls.fetch_add(1, Ordering::Relaxed);
            })
        };
        (observer, calls)
    };
    let (early, early_calls) = counted();
    let (late, late_calls) = counted();
    read.add_done_callback(early);
    read.wait_ready();
    // An observer of a consumable read runs at once.
    read.add_done_callback(late);
    assert_eq!(late_calls.load(Ordering::Relaxed), 1);
    // The earlier observer runs on the read's thread once the read became
    // consumable, which may be after `wait_ready` returned, and before the
    // read retires; no observer runs again.
    while !read.retirement().succeeded() {
        std::thread::yield_now();
    }
    assert_eq!(early_calls.load(Ordering::Relaxed), 1);
    assert_eq!(late_calls.load(Ordering::Relaxed), 1);
    exports::release(&locations)?;
    transports.close()
}

/// The child half of the two-process shared-memory test reads this
/// variable.
const SHARED_LOCATOR: &str = "UNISERVE_TEST_SHM_LOCATOR";
/// The line the consumer prints once it holds its claim on the export.
const CLAIMED: &str = "shared export claimed";

/// Producer half: export a host tensor through shared memory to consumer
/// slot 1 of this host, run this test binary as the consumer, and require
/// the export to keep its bytes while the consumer's claim is open and to
/// return them once the consumer acknowledged it.
#[test]
fn shared_exports_retire_once_another_process_acknowledged_its_reads() -> Result<()> {
    let (transports, capacity, events) = transports(
        &[TransportKind::Local, TransportKind::Shm],
        "producer",
        0,
        1 << 12,
        1 << 20,
    )?;
    let source = counting(&[4, 3])?;
    let mut retirements = Vec::new();
    let locations = located(&transports, &source, None, &[1], None, |retirement| {
        retirements.push(retirement);
        Ok(())
    })?;
    let (_, locator) = locations
        .iter()
        .find(|(_, locator)| TransportKind::of(locator) == TransportKind::Shm)
        .ok_or(Error::State("no shared-memory location was exported"))?;

    let locator = serde_json::to_string(locator)
        .map_err(|error| Error::Invalid(format!("unserializable locator: {error}")))?;
    let executable = std::env::current_exe()
        .map_err(|error| Error::Invalid(format!("no test executable: {error}")))?;
    let mut consumer = Command::new(executable)
        .args([
            "--exact",
            "shared_export_consumer",
            "--ignored",
            "--nocapture",
        ])
        .env(SHARED_LOCATOR, locator)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| Error::Invalid(format!("cannot run the consumer: {error}")))?;
    let lost = || Error::State("the consumer's pipes are gone");
    let mut output = BufReader::new(consumer.stdout.take().ok_or_else(lost)?);
    let mut line = String::new();
    while line.trim_end() != CLAIMED {
        line.clear();
        let read = output
            .read_line(&mut line)
            .map_err(|error| Error::transport_source("reading the consumer's output", error))?;
        assert!(read > 0, "the consumer exited before it claimed the export");
    }

    // The consumer has read the export once and holds a second claim: the
    // revoked export keeps its bytes until that claim is acknowledged.
    exports::release(&locations)?;
    let deadline = Instant::now() + Duration::from_millis(200);
    while Instant::now() < deadline {
        events.drain()?;
        transports.reap()?;
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(capacity.used() > 0, "a claimed export returned its bytes");
    assert!(!retirements.iter().all(|retirement| retirement.done()));

    let mut input = consumer.stdin.take().ok_or_else(lost)?;
    input
        .write_all(b"\n")
        .map_err(|error| Error::transport_source("writing to the consumer", error))?;
    drop(input);
    let status = consumer
        .wait()
        .map_err(|error| Error::transport_source("waiting for the consumer", error))?;
    assert!(status.success(), "the consumer process failed");

    // Acknowledged: the export retires and its bytes return.
    settle(&transports, &capacity, &events)?;
    assert!(retirements.iter().all(|retirement| retirement.succeeded()));
    transports.close()
}

/// Consumer half, run by
/// `shared_exports_retire_once_another_process_acknowledged_its_reads`: copy
/// the export, then borrow it in place until the producer answers.
#[test]
#[ignore = "runs as the consumer process of the shared-memory export test"]
fn shared_export_consumer() -> Result<()> {
    let Ok(locator) = std::env::var(SHARED_LOCATOR) else {
        return Ok(());
    };
    let locator: Locator = serde_json::from_str(&locator)
        .map_err(|error| Error::Invalid(format!("unreadable locator: {error}")))?;
    let (transports, _, _) = transports(&[TransportKind::Shm], "consumer", 1, 1 << 12, 1 << 20)?;
    let transport = transports
        .get(TransportKind::Shm)
        .ok_or(Error::State("no shared-memory transport is bound"))?;

    // A copy read acknowledges its claim when its task retires.
    let destination = zeros(&[2, 3])?;
    let read = fetch(
        transport,
        &locator,
        MemoryKind::Host,
        Some(&destination),
        Some(&[1..3, 0..3]),
        None,
    )?;
    read.wait_ready();
    assert_eq!(
        values(&read.result(None::<&Stream>)?),
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
    while !read.retirement().succeeded() {
        std::thread::sleep(Duration::from_millis(1));
    }

    // A borrow holds its claim until it is dropped.
    let borrowed = transport.borrow(&locator, None)?;
    assert_eq!(borrowed.nbytes(), 48);
    println!("{CLAIMED}");
    std::io::stdout()
        .flush()
        .map_err(|error| Error::transport_source("writing to the producer", error))?;
    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|error| Error::transport_source("reading the producer's answer", error))?;
    drop(borrowed);
    transports.close()
}

const DEVICE: DeviceId = DeviceId(0);

/// The child half of the two-process VMM test reads this variable.
const CHILD_LOCATOR: &str = "UNISERVE_TEST_VMM_LOCATOR";

/// Serializes the device tests of this binary.
///
/// One test holds a stream in a host function to observe the work queued
/// behind it (see [`block`]). While a stream is held, a driver call that
/// waits for all of the context's queued work (`cuMemFree`,
/// `cuMemFreeHost`) cannot return, and it keeps driver locks that every
/// other call needs, including the calls the holding test makes before it
/// releases the stream. No other test may use the driver meanwhile, so every
/// device test takes this lock first.
static DRIVER: Mutex<()> = Mutex::new(());

fn exclusive_driver() -> MutexGuard<'static, ()> {
    DRIVER.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Blocks `stream` in a host function until the returned sender is used.
///
/// Until then the caller must not release device or pinned memory: the
/// release would wait on the held stream (see [`DRIVER`]). Releases queued
/// earlier are flushed first for the same reason.
fn block(stream: &Stream) -> Result<mpsc::Sender<()>> {
    reclaim::flush()?;
    let (release, released) = mpsc::channel::<()>();
    stream.launch_host_fn(move || {
        let _ = released.recv();
    })?;
    Ok(release)
}

/// Copy a device view's bytes to the host (drains `stream`).
fn download(view: &StorageView, stream: &Stream) -> Result<Vec<u8>> {
    let bytes = view.byte_len()?;
    let host = Memory::pinned(DEVICE, bytes.max(1))?;
    let mut batch = CopyBatch::new(SourceOrder::Stream);
    batch.push(host.address(), view.address(), bytes);
    // SAFETY: both ranges hold `bytes`; the stream drains below.
    unsafe { batch.submit(stream) }?;
    stream.synchronize()?;
    // SAFETY: the copy into the pinned buffer completed.
    Ok(unsafe { host.host_bytes() }.unwrap_or_default()[..bytes].to_vec())
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_ne_bytes(*word))
        .collect()
}

/// A device f32 tensor of `shape` holding `0, 1, 2, ...`.
fn device_counting(shape: &[i64], stream: &Stream) -> Result<StorageView> {
    let elements: i64 = shape.iter().product();
    let memory = Memory::device(DEVICE, elements as usize * 4)?;
    let values: Vec<f32> = (0..elements).map(|value| value as f32).collect();
    let mut batch = CopyBatch::new(SourceOrder::DuringCall);
    batch.push(memory.address(), values.as_ptr() as u64, values.len() * 4);
    // SAFETY: the allocation holds the values; the source is read during
    // the call.
    unsafe { batch.submit(stream) }?;
    stream.synchronize()?;
    StorageView::contiguous(memory as Arc<dyn Backing>, 0, DType::F32, shape)
}

/// An uninitialized device f32 destination of `shape`.
fn device_destination(shape: &[i64]) -> Result<Spans> {
    let elements: i64 = shape.iter().product();
    Spans::single(StorageView::contiguous(
        Memory::device(DEVICE, elements as usize * 4)? as Arc<dyn Backing>,
        0,
        DType::F32,
        shape,
    )?)
}

/// Device transports of `kinds` for consumer `slot` with a 4 MiB VMM pool.
fn device_transports(
    kinds: &[TransportKind],
    address_space: &str,
    slot: usize,
) -> Result<(Transports, Arc<TransferCapacity>, Arc<EventPool>)> {
    transports(kinds, address_space, slot, 1 << 24, 1 << 22)
}

#[test]
#[ignore = "requires a CUDA device"]
fn device_exports_reach_local_shared_and_channel_readers() -> Result<()> {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE)?;
    let (transports, capacity, events) = device_transports(
        &[
            TransportKind::Local,
            TransportKind::Shm,
            TransportKind::Channel,
        ],
        "space",
        0,
    )?;
    let spans = Spans::single(device_counting(&[8, 4], &stream)?)?;
    // Slot 0 is on this host, slot 7 is not: local, shm and channel all apply.
    let locations = located(
        &transports,
        &spans,
        None,
        &[0, 7],
        Some(&stream),
        |_| Ok(()),
    )?;
    assert_eq!(
        kinds(&locations),
        [
            TransportKind::Local,
            TransportKind::Shm,
            TransportKind::Channel
        ]
    );

    let reader = Stream::new(DEVICE)?;
    for (transport, locator) in &locations {
        let destination = device_destination(&[2, 4])?;
        let read = fetch(
            transport,
            locator,
            MemoryKind::Cuda(DEVICE),
            Some(&destination),
            Some(&[3..5, 0..4]),
            Some(&reader),
        )?;
        read.wait_ready();
        let value = read.result(Some(&reader))?;
        assert_eq!(
            floats(&download(value.first(), &reader)?),
            (12..20).map(|value| value as f32).collect::<Vec<_>>(),
            "{:?}",
            TransportKind::of(locator)
        );
    }

    exports::release(&locations)?;
    settle(&transports, &capacity, &events)?;
    transports.close()
}

#[test]
#[ignore = "requires a CUDA device"]
fn cancelled_reads_keep_credits_until_their_copy_drains() -> Result<()> {
    let _driver = exclusive_driver();
    let (transports, capacity, _) = device_transports(&[TransportKind::Local], "space", 0)?;
    let stream = Stream::new(DEVICE)?;
    let source = Spans::single(device_counting(&[256], &stream)?)?;
    let locations = located(&transports, &source, None, &[], Some(&stream), |_| Ok(()))?;
    let (transport, locator) = &locations[0];
    let exported = capacity.used();

    // The destination's submitting stream is blocked: the transport copy
    // waits on its handoff fence, so the read cannot drain yet.
    let submitter = Stream::new(DEVICE)?;
    let destination = device_destination(&[256])?;
    let release = block(&submitter)?;
    let read = fetch(
        transport,
        locator,
        MemoryKind::Cuda(DEVICE),
        Some(&destination),
        None,
        Some(&submitter),
    )?;
    std::thread::sleep(Duration::from_millis(50));
    read.cancel()?;
    // The consumer observes the cancellation as a resource failure.
    let Err(cancelled) = read.result(Some(&submitter)) else {
        return Err(Error::State("a cancelled read stayed consumable"));
    };
    assert_eq!(
        WorkerError::classify(cancelled, None).code,
        ErrorCode::ResourceError
    );
    assert!(!read.retirement().succeeded());
    assert!(
        capacity.used() > exported,
        "a running read keeps its bytes after cancellation"
    );

    release
        .send(())
        .map_err(|_| Error::State("the blocked stream was released already"))?;
    let deadline = Instant::now() + Duration::from_secs(10);
    while !read.retirement().succeeded() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(read.retirement().succeeded());
    assert_eq!(capacity.used(), exported);
    exports::release(&locations)?;
    transports.close()
}

/// A device export that does not fit the VMM pool is carried to its consumer
/// by a host mechanism, and fails when the rank binds none: the local
/// location alone reaches no other rank.
#[test]
#[ignore = "requires a CUDA device"]
fn exports_larger_than_the_vmm_pool_need_a_host_mechanism() -> Result<()> {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE)?;
    // 5 MiB of f32, more than the 4 MiB pool the transports configure.
    let spans = Spans::single(device_counting(&[5 << 18], &stream)?)?;

    let (device_only, capacity, events) = device_transports(
        &[TransportKind::Local, TransportKind::CudaVmm],
        "producer",
        0,
    )?;
    let exported = located(&device_only, &spans, None, &[1], Some(&stream), |_| Ok(()));
    assert!(matches!(exported, Err(Error::Resource(_))));
    settle(&device_only, &capacity, &events)?;
    device_only.close()?;

    let (with_shared, capacity, events) = device_transports(
        &[
            TransportKind::Local,
            TransportKind::CudaVmm,
            TransportKind::Shm,
        ],
        "producer",
        0,
    )?;
    let locations = located(&with_shared, &spans, None, &[1], Some(&stream), |_| Ok(()))?;
    assert_eq!(
        kinds(&locations),
        [TransportKind::Local, TransportKind::Shm]
    );
    exports::release(&locations)?;
    settle(&with_shared, &capacity, &events)?;
    with_shared.close()
}

/// A one-group bfloat16 plan: two layers in two columns, 8-token pages of
/// two heads of `head_dim` elements, each plane with one initialized flag.
fn cache_plan(head_dim: u32) -> CachePlan {
    let plane_bytes = u64::from(8 * 2 * head_dim * 2);
    CachePlan {
        columns: 2,
        plane_bytes,
        quantized: false,
        unit_bytes: 2 * 2 * (plane_bytes + 1),
        dtype: PrefixCacheDtype::BFloat16,
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
            dtype: DType::BF16,
            compute_dtype: DType::BF16,
        }],
        handle: uniserve_worker::ffi::Handle::none(),
    }
}

/// Resident cache pages and latent pages reach a VMM consumer where they lie:
/// an export larger than the VMM pool needs no host mechanism.
#[test]
#[ignore = "requires a CUDA device"]
fn resident_pages_larger_than_the_vmm_pool_export_over_vmm() -> Result<()> {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE)?;
    let (transports, capacity, events) = device_transports(
        &[TransportKind::Local, TransportKind::CudaVmm],
        "producer",
        0,
    )?;

    // Six tokens of both columns: 6 MiB of bfloat16, more than the 4 MiB
    // pool the transports configure.
    let plan = Arc::new(cache_plan(1 << 17));
    let planes = CachePlanes::allocate(plan, Some(DEVICE), 2, Some(&stream))?;
    let cache_page = Spans::single(planes.page(0, 0, 1, 0, 6)?)?;

    // One page of 768 units of 2048 f32 elements, also 6 MiB.
    let latents = LatentStorage::allocate(
        &LatentPlan {
            request_pool_size: 1,
            num_pages: 2,
            page_units: 768,
            latent_width: 2048,
            dtype: DType::F32,
            with_workspace: false,
        },
        Some(DEVICE),
        Some(&stream),
    )?;
    let latent_page = latents.spans(0, &[1], 768)?;

    for source in [cache_page, latent_page] {
        assert_eq!(source.nbytes(), 6 << 20);
        let locations = located(&transports, &source, None, &[1], Some(&stream), |_| Ok(()))?;
        assert_eq!(
            kinds(&locations),
            [TransportKind::Local, TransportKind::CudaVmm]
        );
        exports::release(&locations)?;
        settle(&transports, &capacity, &events)?;
    }
    transports.close()
}

/// Producer half: export a device tensor through a CUDA VMM pool chunk to
/// consumer slot 1, run this test binary as the consumer, and require the
/// export to retire only after the consumer acknowledged its read.
#[test]
#[ignore = "requires a CUDA device"]
fn vmm_exports_are_imported_by_another_process() -> Result<()> {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE)?;
    let (transports, capacity, events) = device_transports(
        &[TransportKind::Local, TransportKind::CudaVmm],
        "producer",
        0,
    )?;
    let source = Spans::single(device_counting(&[16, 8], &stream)?)?;
    let mut retirements = Vec::new();
    let locations = located(
        &transports,
        &source,
        None,
        &[1],
        Some(&stream),
        |retirement| {
            retirements.push(retirement);
            Ok(())
        },
    )?;
    let (_, locator) = locations
        .iter()
        .find(|(_, locator)| TransportKind::of(locator) == TransportKind::CudaVmm)
        .ok_or(Error::State("no CUDA VMM location was exported"))?;
    assert!(matches!(
        &locator.transport,
        TransferTransport::CudaVmm { acknowledgment_offset, .. } if *acknowledgment_offset >= 0
    ));

    let locator = serde_json::to_string(locator)
        .map_err(|error| Error::Invalid(format!("unserializable locator: {error}")))?;
    let executable = std::env::current_exe()
        .map_err(|error| Error::Invalid(format!("no test executable: {error}")))?;
    let status = Command::new(executable)
        .args(["--exact", "vmm_export_consumer", "--ignored", "--nocapture"])
        .env(CHILD_LOCATOR, locator)
        .status()
        .map_err(|error| Error::Invalid(format!("cannot run the consumer: {error}")))?;
    assert!(status.success(), "the consumer process failed");

    // The consumer acknowledged and exited: revocation retires the chunk.
    exports::release(&locations)?;
    settle(&transports, &capacity, &events)?;
    assert!(retirements.iter().all(|retirement| retirement.succeeded()));
    transports.close()
}

/// Consumer half, run by `vmm_exports_are_imported_by_another_process`.
#[test]
#[ignore = "runs as the consumer process of the VMM export test"]
fn vmm_export_consumer() -> Result<()> {
    let _driver = exclusive_driver();
    let Ok(locator) = std::env::var(CHILD_LOCATOR) else {
        return Ok(());
    };
    let locator: Locator = serde_json::from_str(&locator)
        .map_err(|error| Error::Invalid(format!("unreadable locator: {error}")))?;
    let (transports, _, _) = device_transports(&[TransportKind::CudaVmm], "consumer", 1)?;
    let transport = transports
        .get(TransportKind::CudaVmm)
        .ok_or(Error::State("no CUDA VMM transport is bound"))?;
    let stream = Stream::new(DEVICE)?;
    let destination = device_destination(&[4, 8])?;
    let read = fetch(
        transport,
        &locator,
        MemoryKind::Cuda(DEVICE),
        Some(&destination),
        Some(&[2..6, 0..8]),
        Some(&stream),
    )?;
    read.wait_ready();
    let value = read.result(Some(&stream))?;
    assert_eq!(
        floats(&download(value.first(), &stream)?),
        (16..48).map(|value| value as f32).collect::<Vec<_>>()
    );
    while !read.retirement().succeeded() {
        std::thread::sleep(Duration::from_millis(1));
    }
    transports.close()
}
