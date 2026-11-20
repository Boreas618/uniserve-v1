//! Behavior against a real driver. These tests need a CUDA device and use a
//! few megabytes of storage on device 0; run them with
//! `cargo test -p platform --test it -- --ignored --test-threads=1`.

use std::ffi::{CString, c_char, c_int, c_void};
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::{Mutex, MutexGuard, PoisonError, mpsc};
use std::thread;
use std::time::{Duration, Instant};

use platform::nccl::{Communicator, ReduceOp, UniqueId};
use platform::{
    Capture, CaptureMode, CaptureStatus, Cluster, CopyBatch, Device, DeviceAllocation,
    DeviceAttribute, DeviceId, DevicePtr, DeviceRange, Error, Event, EventFlags, FunctionAttribute,
    HandleType, Interleave, L2Promotion, LaunchConfig, Library, Nvml, OobFill, PeerMapping,
    PinnedBuffer, Pitched, PitchedBatch, PrimaryContext, Protection, SegmentedMapping,
    ShareableHandle, SourceOrder, Stream, Swizzle, TensorMap, TensorMapSpec, WaitValue,
    memcpy_async, memset_2d_d8, memset_d8, memset_d32, reclaim,
};

pub(crate) const DEVICE: DeviceId = DeviceId(0);

/// Serializes the tests of this binary.
///
/// Some tests hold a stream in a host function to observe the work queued
/// behind it (see [`block`]). While a stream is held, a driver call that waits
/// for all of the context's queued work (`cuMemFree`, `cuMemFreeHost`) cannot
/// return, and it keeps driver locks that every other call needs, including
/// the calls the holding test makes before it releases the stream. No other
/// test may use the driver meanwhile, so every test takes this lock first.
static DRIVER: Mutex<()> = Mutex::new(());

pub(crate) fn exclusive_driver() -> MutexGuard<'static, ()> {
    DRIVER.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(crate) fn read(buffer: &PinnedBuffer) -> Vec<u8> {
    // SAFETY: callers synchronize the stream that wrote the buffer first.
    unsafe { buffer.as_slice() }.to_vec()
}

/// Blocks `stream` in a host function until the returned sender is used.
///
/// Until then the caller must not release device or pinned storage: the
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

#[test]
#[ignore = "requires a CUDA device"]
fn device_queries_report_the_device() {
    let _driver = exclusive_driver();
    assert!(DeviceId::count().unwrap() > DEVICE.0);
    assert!(
        DEVICE
            .attribute(DeviceAttribute::MultiprocessorCount)
            .unwrap()
            > 0
    );
    assert!(DEVICE.uuid().unwrap().to_string().starts_with("GPU-"));

    let storage = DEVICE.memory_info().unwrap();
    assert!(storage.free <= storage.total && storage.total > 0);
}

#[test]
#[ignore = "requires a CUDA device"]
fn fills_and_copies_round_trip_through_device_storage() {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE).unwrap();
    let first = DeviceAllocation::linear(DEVICE, 4096).unwrap();
    let second = DeviceAllocation::linear(DEVICE, 4096).unwrap();
    let mut host = PinnedBuffer::new(DEVICE, 4096).unwrap();

    // SAFETY: every range lies inside the allocations above, which outlive the
    // synchronization at the end of each block.
    unsafe {
        memset_d32(first.address(), 0xa5a5_a5a5, 1024, &stream).unwrap();
        memset_d8(first.address() + 16, 7, 8, &stream).unwrap();
        memcpy_async(second.address(), first.address(), 4096, &stream).unwrap();
        memcpy_async(host.address(), second.address(), 4096, &stream).unwrap();
    }
    stream.synchronize().unwrap();

    let bytes = read(&host);
    assert!(bytes[..16].iter().all(|byte| *byte == 0xa5));
    assert!(bytes[16..24].iter().all(|byte| *byte == 7));
    assert!(bytes[24..].iter().all(|byte| *byte == 0xa5));

    let pattern: Vec<u8> = (0..4096).map(|index| (index % 251) as u8).collect();
    // SAFETY: no copy is pending on the buffer.
    unsafe { host.as_mut_slice() }.copy_from_slice(&pattern);
    // SAFETY: every range lies inside `first` or `host`, which outlive the
    // synchronization below.
    unsafe {
        memcpy_async(first.address(), host.address(), 4096, &stream).unwrap();
        memset_d8(host.address(), 0, 4096, &stream).unwrap();
        memcpy_async(host.address(), first.address(), 4096, &stream).unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(read(&host), pattern);
}

#[test]
#[ignore = "requires a CUDA device"]
fn operations_need_no_current_context_on_the_calling_thread() {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE).unwrap();
    let device = DeviceAllocation::linear(DEVICE, 256).unwrap();
    let host = PinnedBuffer::new(DEVICE, 256).unwrap();
    let event = Event::new(EventFlags::default());

    // A fresh thread has no current CUDA context.
    thread::scope(|scope| {
        scope.spawn(|| {
            let mut batch = CopyBatch::new(SourceOrder::Stream);
            batch.push(host.address(), device.address(), 256);
            // SAFETY: both ranges outlive the synchronization below.
            unsafe {
                memset_d8(device.address(), 3, 256, &stream).unwrap();
                batch.submit(&stream).unwrap();
            }
            event.record(&stream).unwrap();
            event.synchronize().unwrap();
            stream.synchronize().unwrap();
        });
    });

    assert!(read(&host).iter().all(|byte| *byte == 3));
}

#[test]
#[ignore = "requires a CUDA device"]
fn during_call_batches_read_host_sources_before_returning() {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE).unwrap();
    let device = DeviceAllocation::linear(DEVICE, 64).unwrap();
    let readback = PinnedBuffer::new(DEVICE, 64).unwrap();

    // Hold the stream so that the copy cannot run before the source changes.
    let release = block(&stream);
    let mut source: Vec<u8> = (0..64).collect();
    let mut batch = CopyBatch::new(SourceOrder::DuringCall);
    batch.push(device.address(), source.as_ptr() as u64, 64);
    // SAFETY: the device range outlives the synchronization below; the
    // pageable source is read before submit returns.
    unsafe { batch.submit(&stream).unwrap() };
    source.fill(0xff);

    let mut back = CopyBatch::new(SourceOrder::Stream);
    back.push(readback.address(), device.address(), 64);
    // SAFETY: as above.
    unsafe { back.submit(&stream).unwrap() };
    release.send(()).unwrap();
    stream.synchronize().unwrap();

    assert_eq!(read(&readback), (0..64).collect::<Vec<u8>>());
}

/// One stream tells another that work is done by writing a 32-bit flag the
/// other waits on, and nothing on the host polls or synchronizes in between: the
/// reading stream's copy observes the fill only because the wait held it until
/// the writer's write landed, and the two conditions exercised release on the
/// value they name.
#[test]
#[ignore = "requires a CUDA device"]
fn a_wait_on_a_written_flag_orders_one_stream_behind_another() {
    let _driver = exclusive_driver();
    let writing = Stream::new(DEVICE).unwrap();
    let reading = Stream::new(DEVICE).unwrap();
    let data = DeviceAllocation::linear(DEVICE, 4096).unwrap();
    let flag = DeviceAllocation::linear(DEVICE, 4).unwrap();
    let host = PinnedBuffer::new(DEVICE, 4096).unwrap();
    let flag_address = DevicePtr::from_address(flag.address());

    // The flag starts at zero before either side of the handoff is queued, so
    // the wait below cannot be satisfied by whatever the allocation happened to
    // contain.
    // SAFETY: the flag lies inside `flag`, which outlives the synchronization.
    unsafe { memset_d32(flag.address(), 0, 1, &writing) }.unwrap();
    writing.synchronize().unwrap();

    // The reader queues its wait before the writer queues anything, so a pass
    // says the wait did the ordering rather than the order of submission.
    // SAFETY: every range lies inside an allocation that outlives the
    // synchronization below.
    unsafe {
        reading
            .wait_value32(flag_address, 1, WaitValue::Equal)
            .unwrap();
        memcpy_async(host.address(), data.address(), 4096, &reading).unwrap();
        memset_d32(data.address(), 0x5a5a_5a5a, 1024, &writing).unwrap();
        writing.write_value32(flag_address, 1).unwrap();
    }
    reading.synchronize().unwrap();

    assert!(
        read(&host).iter().all(|byte| *byte == 0x5a),
        "the reader observed the fill because the flag ordered it"
    );

    // The flag reads back as written, so the wait released on the value it
    // names rather than on something else.
    // SAFETY: the copy reads inside `flag`, and the host buffer holds four bytes.
    unsafe { memcpy_async(host.address(), flag.address(), 4, &reading) }.unwrap();
    reading.synchronize().unwrap();
    assert_eq!(u32::from_ne_bytes(read(&host)[..4].try_into().unwrap()), 1);

    // The bit-test condition holds while no bit of the value is set and
    // releases once one is: an event recorded behind the wait says whether the
    // stream got past it, so the wait is observed without a host block. The
    // flag holds values without the named bit until the host has looked: the
    // writer's second write, which has another bit, is complete before the
    // event is queried, and only a later write sets the named bit.
    let passed = Event::new(EventFlags::default());
    // SAFETY: the flag lies inside `flag`, which outlives this block.
    unsafe { memset_d32(flag.address(), 0, 1, &writing) }.unwrap();
    writing.synchronize().unwrap();
    // SAFETY: the wait names the same flag the writer just wrote.
    reading
        .wait_value32(flag_address, 2, WaitValue::AnyBitSet)
        .unwrap();
    passed.record(&reading).unwrap();
    writing.write_value32(flag_address, 1).unwrap();
    writing.synchronize().unwrap();
    assert!(
        !passed.query().unwrap(),
        "a flag without the named bit leaves the wait outstanding"
    );

    // SAFETY: as above, and the write is ordered behind nothing the reader
    // needs beyond the flag itself.
    unsafe { memset_d32(flag.address(), 2, 1, &writing) }.unwrap();
    writing.write_value32(flag_address, 2).unwrap();
    reading.synchronize().unwrap();
    assert!(
        passed.query().unwrap(),
        "the wait releases once the flag has the named bit"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn events_order_streams_and_measure_time() {
    let _driver = exclusive_driver();
    let producer = Stream::new(DEVICE).unwrap();
    let consumer = producer.fork().unwrap();
    let device = DeviceAllocation::linear(DEVICE, 1024).unwrap();
    let host = PinnedBuffer::new(DEVICE, 1024).unwrap();
    let start = Event::new(EventFlags {
        timing: true,
        ..EventFlags::default()
    });
    let ready = Event::new(EventFlags {
        timing: true,
        ..EventFlags::default()
    });
    assert!(ready.query().unwrap(), "an unrecorded event is complete");

    let release = block(&producer);
    start.record(&producer).unwrap();
    // SAFETY: the ranges outlive the synchronization below.
    unsafe { memset_d8(device.address(), 9, 1024, &producer).unwrap() };
    ready.record(&producer).unwrap();
    consumer.wait_event(&ready).unwrap();
    // SAFETY: as above.
    unsafe { memcpy_async(host.address(), device.address(), 1024, &consumer).unwrap() };

    // The producer is held behind the host function until the release below,
    // so the event recorded behind it cannot have completed.
    assert!(!ready.query().unwrap());

    release.send(()).unwrap();
    consumer.synchronize().unwrap();
    assert!(ready.query().unwrap());
    assert!(read(&host).iter().all(|byte| *byte == 9));
    assert!(start.elapsed_ms(&ready).unwrap() >= 0.0);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_shared_stream_handle_outlives_the_handle_it_was_cloned_from() {
    let _driver = exclusive_driver();
    let created = Stream::new(DEVICE).unwrap();
    let shared = created.clone();
    assert_eq!(shared.raw(), created.raw());
    drop(created);

    // The remaining handle still submits to the stream and orders its work.
    let device = DeviceAllocation::linear(DEVICE, 1024).unwrap();
    let host = PinnedBuffer::new(DEVICE, 1024).unwrap();
    // SAFETY: the ranges outlive the synchronization below.
    unsafe { memset_d8(device.address(), 5, 1024, &shared).unwrap() };
    // SAFETY: as above.
    unsafe { memcpy_async(host.address(), device.address(), 1024, &shared).unwrap() };
    shared.synchronize().unwrap();
    assert!(read(&host).iter().all(|byte| *byte == 5));
}

#[test]
#[ignore = "requires a CUDA device"]
fn interprocess_events_export_and_open() {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE).unwrap();
    let event = Event::new(EventFlags {
        interprocess: true,
        ..EventFlags::default()
    });
    assert!(matches!(event.ipc_handle(), Err(Error::NotRecorded)));

    event.record(&stream).unwrap();
    let handle = event.ipc_handle().unwrap();
    // Opening a handle in its own process is rejected by the driver; the
    // export itself is what a peer receives.
    assert_ne!(handle.0, [0; 64]);
    stream.synchronize().unwrap();
}

/// A process that created interprocess events, destroyed its primary
/// contexts and exited leaves none of the shared memory the driver mapped
/// for those events, although another library's reference kept the context
/// alive: the driver removes that storage with its context, and a process's
/// exit alone does not.
#[test]
#[ignore = "requires a CUDA device"]
fn resetting_primary_contexts_releases_interprocess_event_storage() {
    let _driver = exclusive_driver();
    // The process half is the package's helper executable
    // (`tests/helpers/interprocess_event.rs`).
    let output = Command::new(env!("CARGO_BIN_EXE_interprocess-event"))
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "the event process failed: {stdout}{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mapped: Vec<&str> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix("mapped "))
        .collect();
    assert!(
        !mapped.is_empty(),
        "an interprocess event maps no shared memory: {stdout}"
    );
    for path in mapped {
        assert!(!Path::new(path).exists(), "{path} outlived its process");
    }
}

/// A retainable allocation's descriptor keeps its physical storage alive
/// after the allocation is released, as a Storage Server's does after its
/// creator exits: later mappings of the descriptor read the contents and
/// change their access in place, at an unchanged address.
#[test]
#[ignore = "requires a CUDA device"]
fn retained_storage_outlives_its_creator_and_changes_access_in_place() {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE).unwrap();
    let granularity = DeviceAllocation::granularity(DEVICE).unwrap();
    let mut host = PinnedBuffer::new(DEVICE, 4096).unwrap();
    let pattern: Vec<u8> = (0..4096).map(|index| (index % 239) as u8).collect();
    // SAFETY: no copy is pending on the buffer.
    unsafe { host.as_mut_slice() }.copy_from_slice(&pattern);

    let created = DeviceAllocation::retainable(DEVICE, granularity, &[]).unwrap();
    let bytes = created.len();
    // SAFETY: both ranges lie inside their allocations, which outlive the
    // synchronization.
    unsafe { memcpy_async(created.address(), host.address(), 4096, &stream).unwrap() };
    stream.synchronize().unwrap();
    let descriptor = created.export_descriptor().unwrap().unwrap();
    assert!(
        DeviceAllocation::exportable(DEVICE, granularity, &[])
            .unwrap()
            .export_descriptor()
            .unwrap()
            .is_none()
    );
    drop(created);
    reclaim::flush().unwrap();

    let retained =
        PeerMapping::import_retained(DEVICE, &descriptor, bytes, &[], Protection::ReadOnly)
            .unwrap();
    // SAFETY: as above; the mapping is read only for the copy.
    unsafe {
        memset_d8(host.address(), 0, 4096, &stream).unwrap();
        memcpy_async(host.address(), retained.address(), 4096, &stream).unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(read(&host), pattern);

    // A second mapping of the descriptor sees writes through the first.
    let again = PeerMapping::import_retained(DEVICE, &descriptor, bytes, &[], Protection::ReadOnly)
        .unwrap();

    // Write access is granted in place, at the same address.
    let address = retained.address();
    retained.protect(Protection::ReadWrite).unwrap();
    assert_eq!(retained.address(), address);
    // SAFETY: the write lies inside the now read-write mapping.
    unsafe {
        memset_d8(retained.address(), 9, 16, &stream).unwrap();
        memcpy_async(host.address(), again.address(), 4096, &stream).unwrap();
    }
    stream.synchronize().unwrap();
    let bytes = read(&host);
    assert!(bytes[..16].iter().all(|byte| *byte == 9));
    assert_eq!(&bytes[16..], &pattern[16..]);
    retained.protect(Protection::ReadOnly).unwrap();
}

#[test]
#[ignore = "requires a CUDA device"]
fn exportable_storage_maps_through_imports_and_aliases() {
    let _driver = exclusive_driver();
    let granularity = DeviceAllocation::granularity(DEVICE).unwrap();
    let stream = Stream::new(DEVICE).unwrap();
    let first = DeviceAllocation::exportable(DEVICE, granularity / 2, &[]).unwrap();
    let second = DeviceAllocation::exportable(DEVICE, granularity, &[]).unwrap();
    assert_eq!(first.len(), granularity);
    assert_eq!(
        first.handle_type(),
        Some(HandleType::probe(DEVICE).unwrap())
    );
    let host = PinnedBuffer::new(DEVICE, 2 * granularity).unwrap();

    // SAFETY: the ranges outlive the synchronization below.
    unsafe {
        memset_d8(first.address(), 1, granularity, &stream).unwrap();
        memset_d8(second.address(), 2, granularity, &stream).unwrap();
    }

    let exported = first.export().unwrap().expect("exportable storage exports");
    let imported = PeerMapping::import(DEVICE, &exported, first.len()).unwrap();
    let alias = PeerMapping::alias(&[&first, &second]).unwrap();
    assert_eq!(alias.len(), 2 * granularity);

    // SAFETY: as above.
    unsafe {
        memcpy_async(host.address(), imported.address(), granularity, &stream).unwrap();
    }
    stream.synchronize().unwrap();
    assert!(read(&host)[..granularity].iter().all(|byte| *byte == 1));

    // Writes through the alias land in the aliased allocations.
    // SAFETY: as above.
    unsafe {
        memset_d8(alias.address() + granularity as u64, 5, 16, &stream).unwrap();
        memcpy_async(host.address(), alias.address(), 2 * granularity, &stream).unwrap();
    }
    stream.synchronize().unwrap();
    let bytes = read(&host);
    assert!(bytes[..granularity].iter().all(|byte| *byte == 1));
    assert!(
        bytes[granularity..granularity + 16]
            .iter()
            .all(|byte| *byte == 5)
    );
    assert!(bytes[granularity + 16..].iter().all(|byte| *byte == 2));

    // Mappings keep their own references: they outlive the allocations.
    drop((first, second));
    // SAFETY: as above.
    unsafe {
        memcpy_async(host.address(), alias.address(), 16, &stream).unwrap();
    }
    stream.synchronize().unwrap();
    assert!(read(&host)[..16].iter().all(|byte| *byte == 1));

    assert!(linear_cannot_export());
    drop((imported, alias, host));
    reclaim::flush().unwrap();
}

fn linear_cannot_export() -> bool {
    let linear = DeviceAllocation::linear(DEVICE, 256).unwrap();
    linear.export().unwrap().is_none()
        && matches!(
            PeerMapping::alias(&[&linear]),
            Err(Error::InvalidArgument(_))
        )
}

#[test]
#[ignore = "requires a CUDA device"]
fn capturing_streams_report_capture_and_reject_storage_operations() {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE).unwrap();
    let device = DeviceAllocation::linear(DEVICE, 256).unwrap();
    assert_eq!(stream.capture_status().unwrap(), CaptureStatus::None);

    let capture = Capture::begin(&stream, CaptureMode::Relaxed).unwrap();
    assert_eq!(stream.capture_status().unwrap(), CaptureStatus::Active);

    let mut batch = CopyBatch::new(SourceOrder::Stream);
    batch.push(device.address(), device.address() + 128, 64);
    assert!(matches!(
        // SAFETY: rejected before any copy is enqueued.
        unsafe { batch.submit(&stream) },
        Err(Error::Capturing { .. })
    ));
    assert!(matches!(
        stream.launch_host_fn(|| {}),
        Err(Error::Capturing { .. })
    ));
    // Fills are capturable.
    // SAFETY: the range lies inside the allocation.
    unsafe { memset_d8(device.address(), 0, 256, &stream).unwrap() };

    // Ending the capture leaves the stream usable; the work this test only
    // checks is accepted is discarded with the graph.
    drop(capture);
    assert_eq!(stream.capture_status().unwrap(), CaptureStatus::None);
}

/// A graph executable instantiated from a captured graph runs its captured
/// work on a stream that did not capture it, after the graph is destroyed.
#[test]
#[ignore = "requires a CUDA device"]
fn graph_executables_launch_on_another_stream() {
    let _driver = exclusive_driver();
    let source = Stream::new(DEVICE).unwrap();
    let device = DeviceAllocation::linear(DEVICE, 256).unwrap();
    let host = PinnedBuffer::new(DEVICE, 256).unwrap();

    // The captured work fills the allocation with 7.
    let capture = Capture::begin(&source, CaptureMode::Relaxed).unwrap();
    // SAFETY: the fill lies inside the allocation.
    unsafe { memset_d8(device.address(), 7, 256, &source) }.unwrap();
    let launched = capture.end().unwrap().instantiate().unwrap();

    let stream = Stream::new(DEVICE).unwrap();
    // SAFETY: the ranges lie inside the allocation and the pinned buffer; the
    // stream orders the clearing fill, the launch and the read back.
    unsafe {
        memset_d8(device.address(), 0, 256, &stream).unwrap();
        launched.launch(&stream).unwrap();
        memcpy_async(host.address(), device.address(), 256, &stream).unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(read(&host), vec![7; 256]);
}

#[test]
#[ignore = "requires a CUDA device and NVML"]
fn nvml_attributes_device_storage_to_this_process() {
    let _driver = exclusive_driver();
    let _allocation = DeviceAllocation::linear(DEVICE, 1 << 20).unwrap();
    let nvml = Nvml::init().unwrap();

    let storage = nvml.memory_info(DEVICE).unwrap();
    assert!(storage.used <= storage.total);
    match nvml.process_storage(DEVICE) {
        Ok(bytes) => assert!(bytes >= 1 << 20),
        // NVML lists host PIDs; a container without the host PID namespace
        // cannot find itself.
        Err(Error::ProcessNotFound { pid, .. }) => assert_eq!(pid, std::process::id()),
        Err(error) => panic!("{error}"),
    }
}

/// NVML sessions keep opening and closing on several threads while CUPTI
/// attaches to the CUDA driver: every session thread ends. The attachment makes
/// the driver's lock functions for NVML do nothing from then on, so a session
/// opening or closing at that moment would keep the driver's lock and the
/// threads waiting for it would wait forever (`platform::nvml`, which orders the
/// two). The process half is the package's helper executable
/// (`tests/helpers/nvml_sessions.rs`): CUPTI attaches once per process.
#[test]
#[ignore = "requires a CUDA device, NVML and CUPTI"]
fn nvml_sessions_open_and_close_across_cupti_s_attachment() {
    let _driver = exclusive_driver();
    let mut child = Command::new(env!("CARGO_BIN_EXE_nvml-sessions"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // The helper's sessions and attachment take about a second; a thread that waits for the
    // driver's lock waits forever, so the wait is bounded and the process ended past it.
    let deadline = Instant::now() + Duration::from_secs(60);
    let mut ended = None;
    while ended.is_none() && Instant::now() < deadline {
        ended = child.try_wait().unwrap();
        thread::sleep(Duration::from_millis(100));
    }
    if ended.is_none() {
        child.kill().unwrap();
    }
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        ended.is_some_and(|status| status.success()),
        "the session threads did not all end within the bound: {stdout}{stderr}"
    );
    let finished = stdout
        .lines()
        .filter(|line| line.starts_with("sessions "))
        .count();
    assert_eq!(finished, 4, "{stdout}");
}

/// Rows of a pitched fill may each lie in a mapped segment while the range
/// they span holds a segment given back: the driver refuses such a fill,
/// and fills of one row each succeed. Cache planes given back in part
/// reset their units column by column for this reason.
#[test]
#[ignore = "requires a CUDA device"]
fn a_pitched_fill_across_a_released_segment_is_refused() {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE).unwrap();
    let granularity = HandleType::Descriptor.granularity(DEVICE).unwrap();
    // Two columns of two segments each, column-major.
    let mut segmented = SegmentedMapping::reserve(DEVICE, &[granularity; 4], &[]).unwrap();
    let _descriptors: Vec<_> = (0..4)
        .map(|index| segmented.create(index).unwrap())
        .collect();
    segmented.release(1).unwrap();
    segmented.release(3).unwrap();
    let pitch = 2 * granularity;
    // SAFETY: each row lies inside a mapped segment.
    let spanning =
        unsafe { memset_2d_d8(segmented.address(), pitch, 7, granularity / 2, 2, &stream) };
    let per_row: Vec<_> = (0..2)
        .map(|column| {
            // SAFETY: the row lies inside column `column`'s mapped segment.
            unsafe {
                memset_2d_d8(
                    segmented.address() + (column * pitch) as u64,
                    pitch,
                    7,
                    granularity / 2,
                    1,
                    &stream,
                )
            }
        })
        .collect();
    stream.synchronize().unwrap();
    println!("a pitched fill across a released segment: {spanning:?}");
    assert!(per_row.iter().all(Result::is_ok), "{per_row:?}");
    assert!(
        spanning.is_err(),
        "the driver filled rows across a released segment"
    );
}

/// A pitched copy whose rows lie in mapped segments while the range they
/// span holds a segment given back, as a strided copy of one cache unit
/// across the columns of planes given back in part: the driver refuses
/// it, and one contiguous copy per row, each inside its segment, is
/// served.
#[test]
#[ignore = "requires a CUDA device"]
fn a_pitched_copy_across_a_released_segment_is_refused() {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE).unwrap();
    let granularity = HandleType::Descriptor.granularity(DEVICE).unwrap();
    // Two columns of two segments each, column-major.
    let mut segmented = SegmentedMapping::reserve(DEVICE, &[granularity; 4], &[]).unwrap();
    let _descriptors: Vec<_> = (0..4)
        .map(|index| segmented.create(index).unwrap())
        .collect();
    segmented.release(1).unwrap();
    segmented.release(3).unwrap();
    let pitch = 2 * granularity;
    let width = granularity / 2;
    let target = DeviceAllocation::linear(DEVICE, 2 * width).unwrap();
    let host = PinnedBuffer::new(DEVICE, 2 * width).unwrap();
    // Both rows of the source as one pitched copy, which spans the released
    // segment between them, and as one contiguous copy per row, each inside a
    // mapped segment.
    let mut spanning = PitchedBatch::new(SourceOrder::Stream);
    spanning
        .push(Pitched {
            destination: target.address(),
            destination_pitch: width,
            destination_layer: 2,
            source: segmented.address(),
            source_pitch: pitch,
            source_layer: 2,
            width,
            height: 2,
            depth: 1,
        })
        .unwrap();
    let mut bounded = CopyBatch::new(SourceOrder::Stream);
    for row in 0..2 {
        bounded.push(
            target.address() + (row * width) as u64,
            segmented.address() + (row * pitch) as u64,
            width,
        );
    }
    // SAFETY: each row lies inside a mapped segment, the target and the
    // host buffer outlive the synchronization below.
    let (refused, copied) = unsafe {
        for column in 0..2_u8 {
            memset_d8(
                segmented.address() + (usize::from(column) * pitch) as u64,
                column + 1,
                width,
                &stream,
            )
            .unwrap();
        }
        let refused = spanning.submit(&stream);
        let copied = bounded
            .submit(&stream)
            .and_then(|()| memcpy_async(host.address(), target.address(), 2 * width, &stream));
        (refused, copied)
    };
    stream.synchronize().unwrap();
    println!("a pitched copy across a released segment: {refused:?}");
    assert!(
        refused.is_err(),
        "the driver copied rows across a released segment"
    );
    copied.unwrap();
    let bytes = read(&host);
    assert!(bytes[..width].iter().all(|&byte| byte == 1));
    assert!(bytes[width..].iter().all(|&byte| byte == 2));
}

/// A segment a mapping creates is mapped from its descriptor, as another
/// process maps it, and its creating handle stays with the mapping until
/// the segment is released: NVML attributes the storage to this process
/// meanwhile, as `created_bytes` counts it for capacity planning, which
/// books it apart from the process's own, and to no process once released.
#[test]
#[ignore = "requires a CUDA device and NVML"]
fn a_created_segment_is_counted_as_created_until_it_is_released() {
    let _driver = exclusive_driver();
    let nvml = Nvml::init().unwrap();
    let granularity = HandleType::Descriptor.granularity(DEVICE).unwrap();
    let bytes = (1_usize << 30).div_ceil(granularity) * granularity;
    let mut segmented = SegmentedMapping::reserve(DEVICE, &[bytes], &[]).unwrap();
    // Both NVML readings are taken with every queued release made, so what
    // separates them is this test's own segment and not an earlier test's
    // storage still waiting for the reclaimer.
    reclaim::flush().unwrap();
    let before = match nvml.process_storage(DEVICE) {
        Ok(before) => before,
        // NVML lists host PIDs; a container without the host PID namespace
        // cannot find itself.
        Err(Error::ProcessNotFound { .. }) => return,
        Err(error) => panic!("{error}"),
    };
    let created_before = platform::created_bytes(DEVICE);
    let _descriptor = segmented.create(0).unwrap();
    assert_eq!(
        platform::created_bytes(DEVICE),
        created_before + bytes as u64
    );
    let held = nvml.process_storage(DEVICE).unwrap();
    assert!(
        held >= before + bytes as u64 / 2,
        "NVML attributes the held creation to no process: {before} -> {held} bytes"
    );
    // What the process holds of its own leaves the created segment out.
    assert!(held - platform::created_bytes(DEVICE) < before + bytes as u64 / 2);

    segmented.release(0).unwrap();
    assert_eq!(platform::created_bytes(DEVICE), created_before);
    // `release` unmaps the segment at once but hands the creating handle to
    // the reclaimer, so the storage is freed when the queue has been made.
    reclaim::flush().unwrap();
    let released = nvml.process_storage(DEVICE).unwrap();
    assert!(
        released < before + bytes as u64 / 2,
        "NVML attributes the released segment to this process: {before} -> {released} bytes"
    );
}

#[test]
#[ignore = "requires a CUDA device and NCCL"]
fn single_rank_communicators_run_collectives() {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE).unwrap();
    let data = DeviceAllocation::linear(DEVICE, 16).unwrap();
    let host = PinnedBuffer::new(DEVICE, 16).unwrap();
    let id = UniqueId::new().unwrap();
    let communicator = Communicator::init(DEVICE, 1, 0, &id).unwrap();
    assert_eq!((communicator.rank(), communicator.size()), (0, 1));

    // SAFETY: the buffers outlive the synchronization below.
    unsafe { memset_d32(data.address(), 41, 4, &stream).unwrap() };
    let words = DeviceRange::new(DevicePtr::from_address(data.address()), data.len());
    communicator
        .all_reduce(
            &stream,
            words,
            words,
            4,
            platform::DataType::Int32,
            ReduceOp::Min,
        )
        .unwrap();
    communicator
        .broadcast(&stream, words, words, 4, platform::DataType::Int32, 0)
        .unwrap();
    // A collective whose stated ranges cannot hold the elements it asks for
    // fails before any NCCL call is made.
    let short = DeviceRange::new(DevicePtr::from_address(data.address()), 8);
    let refused = communicator
        .all_reduce(
            &stream,
            short,
            words,
            4,
            platform::DataType::Int32,
            ReduceOp::Min,
        )
        .unwrap_err();
    assert!(
        matches!(refused, Error::InvalidArgument { .. }),
        "{refused:?}"
    );
    // SAFETY: the buffer outlives the synchronization below.
    unsafe { memcpy_async(host.address(), data.address(), 16, &stream).unwrap() };
    stream.synchronize().unwrap();
    let words: Vec<u32> = read(&host)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_ne_bytes(*word))
        .collect();
    assert_eq!(words, [41; 4]);

    communicator.destroy().unwrap();
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_released_segment_is_mapped_again_at_its_address() {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE).unwrap();
    let granularity = HandleType::Descriptor.granularity(DEVICE).unwrap();
    let mut segmented = SegmentedMapping::reserve(DEVICE, &[granularity; 3], &[]).unwrap();
    let address = segmented.address();
    let descriptors: Vec<_> = (0..3)
        .map(|index| segmented.create(index).unwrap())
        .collect();
    let host = PinnedBuffer::new(DEVICE, 3 * granularity).unwrap();
    // SAFETY: every range lies inside a mapped segment, which outlives the
    // synchronization.
    unsafe {
        for index in 0..3 {
            memset_d8(
                address + (index * granularity) as u64,
                index as u8 + 1,
                granularity,
                &stream,
            )
            .unwrap();
        }
    }
    stream.synchronize().unwrap();

    // Another mapping maps the same storage from the segments' descriptors.
    let mut other = SegmentedMapping::reserve(DEVICE, &[granularity; 3], &[]).unwrap();
    for (index, descriptor) in descriptors.iter().enumerate() {
        other.import(index, descriptor).unwrap();
    }
    // SAFETY: as above.
    unsafe { memcpy_async(host.address(), other.address(), 3 * granularity, &stream).unwrap() };
    stream.synchronize().unwrap();
    let bytes = read(&host);
    for index in 0..3 {
        assert!(
            bytes[index * granularity..(index + 1) * granularity]
                .iter()
                .all(|byte| *byte == index as u8 + 1)
        );
    }

    // The last segment leaves both mappings; a new one takes its place at
    // the same addresses, and the others keep their bytes.
    segmented.release(2).unwrap();
    other.release(2).unwrap();
    assert!(!segmented.is_mapped(2) && segmented.is_mapped(1));
    drop(descriptors);
    let _fresh = segmented.create(2).unwrap();
    assert_eq!(segmented.address(), address);
    // SAFETY: as above.
    unsafe {
        memset_d8(address + (2 * granularity) as u64, 9, granularity, &stream).unwrap();
        memcpy_async(host.address(), address, 3 * granularity, &stream).unwrap();
    }
    stream.synchronize().unwrap();
    let bytes = read(&host);
    assert!(bytes[..granularity].iter().all(|byte| *byte == 1));
    assert!(
        bytes[granularity..2 * granularity]
            .iter()
            .all(|byte| *byte == 2)
    );
    assert!(bytes[2 * granularity..].iter().all(|byte| *byte == 9));
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_captured_graph_reads_remapped_storage_without_changing_shared_bytes() {
    let _driver = exclusive_driver();
    let granularity = HandleType::Descriptor.granularity(DEVICE).unwrap();
    let bytes = 2 * granularity;
    let mut active = SegmentedMapping::reserve(DEVICE, &[bytes], &[]).unwrap();
    let original = active.create(0).unwrap();
    let creator = active.allocation(0).unwrap();
    let fabric = active.fabric_handle(0);
    let mut shared = SegmentedMapping::reserve(DEVICE, &[bytes], &[]).unwrap();
    shared.import(0, &original).unwrap();
    let mut source = SegmentedMapping::reserve(DEVICE, &[granularity; 2], &[]).unwrap();
    let descriptors: Vec<_> = (0..2).map(|index| source.create(index).unwrap()).collect();
    let stream = Stream::new(DEVICE).unwrap();
    let host = PinnedBuffer::new(DEVICE, bytes).unwrap();
    // SAFETY: every fill lies in a live allocation; synchronization precedes
    // capture and the change of access rights.
    unsafe {
        memset_d8(active.address(), 7, bytes, &stream).unwrap();
        memset_d8(source.address(), 13, granularity, &stream).unwrap();
        memset_d8(
            source.address() + granularity as u64,
            17,
            granularity,
            &stream,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    source.protect(Protection::ReadOnly).unwrap();
    let capture = Capture::begin(&stream, CaptureMode::Relaxed).unwrap();
    // SAFETY: the copy borrows allocations that outlive the executable.
    unsafe { memcpy_async(host.address(), active.address(), bytes, &stream) }.unwrap();
    let read_back = capture.end().unwrap().instantiate().unwrap();
    read_back.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(read(&host), vec![7; bytes]);

    active
        .remap(
            &descriptors
                .iter()
                .map(|descriptor| (granularity, descriptor))
                .collect::<Vec<_>>(),
            Protection::ReadOnly,
        )
        .unwrap();
    read_back.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    let observed = read(&host);
    assert_eq!(&observed[..granularity], vec![13; granularity]);
    assert_eq!(&observed[granularity..], vec![17; granularity]);

    // The predecessor's alias still reads the original immutable bytes.
    // SAFETY: both copy ranges remain mapped through the synchronization.
    unsafe { memcpy_async(host.address(), shared.address(), bytes, &stream).unwrap() };
    stream.synchronize().unwrap();
    assert_eq!(read(&host), vec![7; bytes]);

    // A retained creating handle keeps a remote fabric export usable even
    // after its original virtual reservation points at different weights.
    if let Some(handle) = fabric {
        let imported =
            PeerMapping::import(DEVICE, &ShareableHandle::Fabric(handle), bytes).unwrap();
        // SAFETY: both allocations stay alive until the copy completes.
        unsafe { memcpy_async(host.address(), imported.address(), bytes, &stream).unwrap() };
        stream.synchronize().unwrap();
        assert_eq!(read(&host), vec![7; bytes]);
    }
    drop(creator);
}

/// The kernels of the launch, capture and rebind tests (`fixtures/kernels.ptx`).
const KERNELS: &[u8] = include_bytes!("fixtures/kernels.ptx");

/// The cluster kernel of the launch test (`fixtures/cluster.ptx`), for a
/// device that launches clusters.
const CLUSTER_KERNELS: &[u8] = include_bytes!("fixtures/cluster.ptx");

/// The source of the binary-image tests' kernel (`fixtures/add_constant.cu`).
const ADD_CONSTANT: &str = include_str!("fixtures/add_constant.cu");

/// The words of `buffer`, read after the writing stream was synchronized.
fn words(buffer: &PinnedBuffer) -> Vec<u32> {
    let bytes = read(buffer);
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect()
}

/// A library of [`KERNELS`] on `device`.
fn kernels(device: &Device) -> Library {
    Library::load(device, KERNELS).unwrap()
}

/// NVRTC of the CUDA 13 packages, by the soname its package installs on the
/// loader's default search path.
const LIBNVRTC: &str = "libnvrtc.so.13";

/// `nvrtcResult`'s `NVRTC_SUCCESS`.
const NVRTC_SUCCESS: c_int = 0;

/// [`ADD_CONSTANT`] compiled by NVRTC into a cubin for `device`'s compute
/// capability: a binary image, NUL bytes included.
fn add_constant_cubin(device: &Device) -> Vec<u8> {
    type Program = *mut c_void;
    type Create = unsafe extern "C" fn(
        *mut Program,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> c_int;
    type Compile = unsafe extern "C" fn(Program, c_int, *const *const c_char) -> c_int;
    type Size = unsafe extern "C" fn(Program, *mut usize) -> c_int;
    type Read = unsafe extern "C" fn(Program, *mut c_char) -> c_int;
    type Destroy = unsafe extern "C" fn(*mut Program) -> c_int;

    // SAFETY: loading NVRTC runs only its own initializers.
    let nvrtc = unsafe { libloading::Library::new(LIBNVRTC) }
        .expect("NVRTC of the CUDA 13 packages is on the loader's search path");
    // SAFETY: each signature is the `nvrtc.h` prototype of the symbol.
    let (create, compile, log_size, log, cubin_size, cubin, destroy) = unsafe {
        (
            *nvrtc.get::<Create>(b"nvrtcCreateProgram").unwrap(),
            *nvrtc.get::<Compile>(b"nvrtcCompileProgram").unwrap(),
            *nvrtc.get::<Size>(b"nvrtcGetProgramLogSize").unwrap(),
            *nvrtc.get::<Read>(b"nvrtcGetProgramLog").unwrap(),
            *nvrtc.get::<Size>(b"nvrtcGetCUBINSize").unwrap(),
            *nvrtc.get::<Read>(b"nvrtcGetCUBIN").unwrap(),
            *nvrtc.get::<Destroy>(b"nvrtcDestroyProgram").unwrap(),
        )
    };

    let source = CString::new(ADD_CONSTANT).unwrap();
    let name = c"add_constant.cu";
    let (major, minor) = device.architecture();
    let architecture = CString::new(format!("--gpu-architecture=sm_{major}{minor}")).unwrap();
    let options = [architecture.as_ptr()];
    let mut program: Program = std::ptr::null_mut();
    // SAFETY: the source and name are NUL-terminated and outlive the call; no
    // headers are given; NVRTC writes the program handle.
    let created = unsafe {
        create(
            &mut program,
            source.as_ptr(),
            name.as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
    };
    assert_eq!(created, NVRTC_SUCCESS, "nvrtcCreateProgram");

    // SAFETY: the program is live and the option array holds one C string.
    let compiled = unsafe { compile(program, 1, options.as_ptr()) };
    let mut size = 0;
    // SAFETY: NVRTC writes the log's size, NUL included.
    assert_eq!(unsafe { log_size(program, &mut size) }, NVRTC_SUCCESS);
    let mut text = vec![0u8; size.max(1)];
    // SAFETY: the buffer holds the log's size.
    let logged = unsafe { log(program, text.as_mut_ptr().cast()) };
    assert_eq!(logged, NVRTC_SUCCESS);
    assert_eq!(
        compiled,
        NVRTC_SUCCESS,
        "NVRTC refused {architecture:?}: {}",
        String::from_utf8_lossy(&text)
    );

    // SAFETY: NVRTC writes the cubin's size, then the cubin into a buffer of it.
    assert_eq!(unsafe { cubin_size(program, &mut size) }, NVRTC_SUCCESS);
    let mut image = vec![0u8; size];
    // SAFETY: the buffer holds the cubin's size.
    let read = unsafe { cubin(program, image.as_mut_ptr().cast()) };
    assert_eq!(read, NVRTC_SUCCESS);
    // SAFETY: the program is NVRTC's and destroyed once.
    unsafe { destroy(&mut program) };
    image
}

/// A binary library image (a cubin, compiled for the device it loads on)
/// contains NUL bytes, which the driver reads by the image's own headers; it
/// loads and its kernel runs.
#[test]
#[ignore = "requires a CUDA device"]
fn libraries_load_a_binary_image_and_launch_its_kernel() {
    let _driver = exclusive_driver();
    let device = Device::new(DEVICE).unwrap();
    assert!(matches!(
        Library::load(&device, &[]),
        Err(Error::InvalidArgument(_))
    ));

    let library = Library::load(&device, &add_constant_cubin(&device)).unwrap();
    let add_constant = library.kernel("add_constant").unwrap();
    let stream = Stream::new(DEVICE).unwrap();
    let storage = DeviceAllocation::linear(DEVICE, 32).unwrap();
    let host = PinnedBuffer::new(DEVICE, 32).unwrap();

    // SAFETY: the fill and the copy lie inside the allocation; the launch's
    // parameters are the kernel's, and the value they point at outlives it.
    unsafe { memset_d32(storage.address(), 1, 8, &stream) }.unwrap();
    let mut out = storage.address();
    let mut value = 2u32;
    let mut parameters: [*mut c_void; 2] = [(&raw mut out).cast(), (&raw mut value).cast()];
    add_constant
        .launch_ex(
            &stream,
            &LaunchConfig::new([1, 1, 1], [8, 1, 1]),
            &mut parameters,
        )
        .unwrap();
    // SAFETY: the copy reads the allocation into the pinned buffer of the same
    // size.
    unsafe { memcpy_async(host.address(), storage.address(), 32, &stream) }.unwrap();
    stream.synchronize().unwrap();

    assert_eq!(words(&host), vec![3; 8]);
}

/// A function reports the parameters of its own signature as the driver
/// declares them: the byte offset each value sits at in the packed layout a
/// captured kernel node stores, and the bytes it occupies. A caller that
/// publishes an entry point to another process takes the declaration from the
/// driver instead of copying a header.
#[test]
#[ignore = "requires a CUDA device"]
fn a_function_reports_the_parameters_of_its_signature() {
    let _driver = exclusive_driver();
    let device = Device::new(DEVICE).unwrap();
    let library = kernels(&device);

    // `write(.param .u64 out, .param .u32 value)`: eight bytes, then four.
    let write = library.kernel("write").unwrap();
    let parameters = write.parameters().unwrap();
    let layout: Vec<(usize, usize)> = parameters
        .iter()
        .map(|parameter| (parameter.offset(), parameter.size()))
        .collect();
    assert_eq!(layout, vec![(0, 8), (8, 4)]);
    assert_eq!(write.parameter(1).unwrap(), Some(parameters[1]));
    assert_eq!(write.parameter(2).unwrap(), None);

    // `report(.param .u64 out)`: one eight-byte parameter.
    let report = library.kernel("report").unwrap();
    let declared = report.parameters().unwrap();
    let layout: Vec<(usize, usize)> = declared
        .iter()
        .map(|parameter| (parameter.offset(), parameter.size()))
        .collect();
    assert_eq!(layout, vec![(0, 8)]);

    // A kernel that takes nothing reports an empty signature rather than
    // failing at the end of its walk.
    let nothing = library.kernel("nothing").unwrap();
    assert!(nothing.parameters().unwrap().is_empty());
    assert_eq!(nothing.parameter(0).unwrap(), None);

    // The declaration comes from the loaded image, not from the PTX text: a
    // kernel of a cubin image reports the same layout.
    let cubin = Library::load(&device, &add_constant_cubin(&device)).unwrap();
    let add_constant = cubin.kernel("add_constant").unwrap();
    let layout: Vec<(usize, usize)> = add_constant
        .parameters()
        .unwrap()
        .iter()
        .map(|parameter| (parameter.offset(), parameter.size()))
        .collect();
    assert_eq!(layout, vec![(0, 8), (8, 4)]);
}

/// One launch carries a three-dimensional grid and block, the dynamic
/// shared-memory size its program asks for, a thread-block cluster with its
/// scheduling policy, programmatic dependent launch and cooperative launch;
/// the kernel sees each of them.
#[test]
#[ignore = "requires a CUDA device"]
fn launch_ex_carries_the_grid_shared_memory_and_launch_attributes() {
    let _driver = exclusive_driver();
    let device = Device::new(DEVICE).unwrap();
    let library = kernels(&device);
    let report = library.kernel("report").unwrap();
    // Above 48 KiB a launch needs the function's attribute raised first.
    report
        .set_attribute(FunctionAttribute::MaxDynamicSharedMemory, 64 << 10)
        .unwrap();

    let stream = Stream::new(DEVICE).unwrap();
    let blocks = 8;
    let storage = DeviceAllocation::linear(DEVICE, 7 * 4 * blocks as usize).unwrap();
    let host = PinnedBuffer::new(DEVICE, 7 * 4 * blocks as usize).unwrap();
    let mut out = storage.address();
    let mut parameters: [*mut c_void; 1] = [(&raw mut out).cast()];

    let config = LaunchConfig {
        grid: [2, 2, 2],
        block: [2, 1, 1],
        shared_memory: 64 << 10,
        ..LaunchConfig::default()
    };
    // SAFETY: the allocation holds one seven-word record per block; the
    // parameter is the kernel's and outlives the launch.
    unsafe { memset_d8(storage.address(), 0xff, 7 * 4 * blocks as usize, &stream) }.unwrap();
    report.launch_ex(&stream, &config, &mut parameters).unwrap();
    // SAFETY: the copy reads the allocation into the pinned buffer.
    unsafe {
        memcpy_async(
            host.address(),
            storage.address(),
            7 * 4 * blocks as usize,
            &stream,
        )
    }
    .unwrap();
    stream.synchronize().unwrap();

    let observed = words(&host);
    for block in 0..blocks as u32 {
        let record = &observed[block as usize * 7..][..7];
        let (x, y, z) = (block % 2, block / 2 % 2, block / 4);
        assert_eq!(
            record,
            [x, y, z, 0, 0, 0, 64 << 10],
            "block {block} reports its launch"
        );
    }

    // Programmatic dependent launch and cooperative launch are attributes of
    // the same launch call; a single block executes each.
    for config in [
        LaunchConfig {
            grid: [1, 1, 1],
            block: [1, 1, 1],
            pdl: true,
            ..LaunchConfig::default()
        },
        LaunchConfig {
            grid: [1, 1, 1],
            block: [1, 1, 1],
            cooperative: true,
            ..LaunchConfig::default()
        },
    ] {
        // SAFETY: as above; the record at the allocation's start is inside it.
        unsafe { memset_d8(storage.address(), 0xff, 28, &stream) }.unwrap();
        report.launch_ex(&stream, &config, &mut parameters).unwrap();
        // SAFETY: as above.
        unsafe { memcpy_async(host.address(), storage.address(), 28, &stream) }.unwrap();
        stream.synchronize().unwrap();
        assert_eq!(words(&host)[..7], [0, 0, 0, 0, 0, 0, 0], "{config:?}");
    }

    // A cluster is one launch attribute more, on a device that launches
    // clusters. The kernel reads the cluster's own registers, so a launch that
    // dropped the attribute cannot pass: without it every block is a cluster
    // of one.
    if DEVICE.attribute(DeviceAttribute::ClusterLaunch).unwrap() != 0 {
        let clustered = Library::load(&device, CLUSTER_KERNELS).unwrap();
        let cluster = clustered.kernel("report_cluster").unwrap();
        let config = LaunchConfig {
            grid: [2, 2, 1],
            block: [1, 1, 1],
            cluster: Some(Cluster::new([2, 1, 1])),
            ..LaunchConfig::default()
        };
        // The launch writes one record of eight words per block of its grid,
        // and only that many: the pinned buffer still holds the earlier
        // section's records beyond them.
        let records = 4 * 8 * 4;
        // SAFETY: four records of eight words lie in the allocation.
        unsafe { memset_d8(storage.address(), 0xff, records, &stream) }.unwrap();
        cluster
            .launch_ex(&stream, &config, &mut parameters)
            .unwrap();
        // SAFETY: as above.
        unsafe { memcpy_async(host.address(), storage.address(), records, &stream) }.unwrap();
        stream.synchronize().unwrap();
        let observed = words(&host);
        let written = &observed[..records / 4];
        for (block, record) in written.as_chunks::<8>().0.iter().enumerate() {
            let block = block as u32;
            let (x, y) = (block % 2, block / 2);
            // The grid's block, the block's place inside its cluster (two
            // blocks wide, so the cluster-local x is the grid x modulo two),
            // its rank inside the cluster, the cluster's own dimensions, and
            // the thread's x index.
            assert_eq!(
                *record,
                [x, y, x, 0, x, 2, 1, 0],
                "block {block} of a clustered launch"
            );
        }
    }
}

/// Recapture rebuilds a graph from a new capture, so a program whose node
/// count changed gets a new executable; the previous one keeps replaying what
/// it captured.
#[test]
#[ignore = "requires a CUDA device"]
fn recapture_builds_an_executable_of_the_new_capture() {
    let _driver = exclusive_driver();
    let device = Device::new(DEVICE).unwrap();
    let library = kernels(&device);
    let write = library.kernel("write").unwrap();
    let stream = Stream::new(DEVICE).unwrap();
    let storage = DeviceAllocation::linear(DEVICE, 32).unwrap();
    let host = PinnedBuffer::new(DEVICE, 32).unwrap();
    let mut out = storage.address();

    let capture = Capture::begin(&stream, CaptureMode::Relaxed).unwrap();
    let mut one = 1u32;
    let mut parameters: [*mut c_void; 2] = [(&raw mut out).cast(), (&raw mut one).cast()];
    write
        .launch_ex(
            &stream,
            &LaunchConfig::new([1, 1, 1], [8, 1, 1]),
            &mut parameters,
        )
        .unwrap();
    let before = capture.end().unwrap();
    let before_exec = before.instantiate().unwrap();
    assert_eq!(before.node_count().unwrap(), 1);

    let capture = Capture::begin(&stream, CaptureMode::Relaxed).unwrap();
    let mut two = 2u32;
    let mut three = 3u32;
    for value in [&mut two, &mut three] {
        let mut parameters: [*mut c_void; 2] = [(&raw mut out).cast(), (&raw mut *value).cast()];
        write
            .launch_ex(
                &stream,
                &LaunchConfig::new([1, 1, 1], [8, 1, 1]),
                &mut parameters,
            )
            .unwrap();
    }
    let after = capture.end().unwrap();
    assert_eq!(after.node_count().unwrap(), 2);
    let after_exec = after.instantiate().unwrap();

    before_exec.launch(&stream).unwrap();
    after_exec.launch(&stream).unwrap();
    // SAFETY: the copy reads the allocation into the pinned buffer.
    unsafe { memcpy_async(host.address(), storage.address(), 32, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        words(&host),
        vec![3; 8],
        "the recaptured executable ran, and the earlier one still replays"
    );
}

/// An executable takes a new capture of its own topology in place: its next
/// launch runs the new capture's arguments, with no instantiation.
#[test]
#[ignore = "requires a CUDA device"]
fn an_executable_takes_a_capture_of_its_topology_in_place() {
    let _driver = exclusive_driver();
    let device = Device::new(DEVICE).unwrap();
    let library = kernels(&device);
    let write = library.kernel("write").unwrap();
    let stream = Stream::new(DEVICE).unwrap();
    let storage = DeviceAllocation::linear(DEVICE, 32).unwrap();
    let host = PinnedBuffer::new(DEVICE, 32).unwrap();
    let mut out = storage.address();

    let mut captured = |value: u32| {
        let mut value = value;
        let capture = Capture::begin(&stream, CaptureMode::Relaxed).unwrap();
        let mut parameters: [*mut c_void; 2] = [(&raw mut out).cast(), (&raw mut value).cast()];
        write
            .launch_ex(
                &stream,
                &LaunchConfig::new([1, 1, 1], [8, 1, 1]),
                &mut parameters,
            )
            .unwrap();
        capture.end().unwrap()
    };
    let first = captured(5);
    let second = captured(9);
    let exec = first.instantiate().unwrap();

    exec.update(&second).unwrap();
    exec.launch(&stream).unwrap();
    // SAFETY: the copy reads the allocation into the pinned buffer.
    unsafe { memcpy_async(host.address(), storage.address(), 32, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        words(&host),
        vec![9; 8],
        "the updated executable ran the new capture"
    );
}

/// What an executable costs as the graph it runs grows: the host time of its
/// instantiation (`cuGraphInstantiate`), of an update of an executable of the
/// same topology in place (`cuGraphExecUpdate`), and the device memory the
/// instantiation takes, for graphs of 64 to 8192 kernel nodes. Printed as one
/// JSON line per size; the update holds because both graphs are one topology.
#[test]
#[ignore = "requires a CUDA device; a measurement"]
fn instantiation_and_update_costs_by_node_count() {
    let _driver = exclusive_driver();
    let device = Device::new(DEVICE).unwrap();
    let library = kernels(&device);
    let write = library.kernel("write").unwrap();
    let stream = Stream::new(DEVICE).unwrap();
    let storage = DeviceAllocation::linear(DEVICE, 32).unwrap();
    let mut out = storage.address();

    // A chain of `nodes` launches of the one kernel, each writing `value`.
    let mut captured = |nodes: usize, value: u32| {
        let mut value = value;
        let capture = Capture::begin(&stream, CaptureMode::Relaxed).unwrap();
        for _ in 0..nodes {
            let mut parameters: [*mut c_void; 2] = [(&raw mut out).cast(), (&raw mut value).cast()];
            write
                .launch_ex(
                    &stream,
                    &LaunchConfig::new([1, 1, 1], [8, 1, 1]),
                    &mut parameters,
                )
                .unwrap();
        }
        capture.end().unwrap()
    };

    for nodes in [64, 256, 1024, 4096, 8192] {
        let first = captured(nodes, 1);
        let second = captured(nodes, 2);
        assert_eq!(second.node_count().unwrap(), nodes);

        let free = DEVICE.memory_info().unwrap().free;
        let began = std::time::Instant::now();
        let exec = first.instantiate().unwrap();
        let instantiate = began.elapsed();
        let bytes = free.saturating_sub(DEVICE.memory_info().unwrap().free);

        let began = std::time::Instant::now();
        exec.update(&second).unwrap();
        let update = began.elapsed();

        exec.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        println!(
            "{{\"nodes\":{nodes},\"instantiate_us\":{},\"update_us\":{},\"bytes\":{bytes}}}",
            instantiate.as_micros(),
            update.as_micros()
        );
    }
}

/// A tensor map encodes the layout a TMA kernel moves: the driver encodes a
/// two-dimensional description of device storage, a one-dimensional tensor
/// reaches it as rank two with a trailing dimension of one, and a rank the
/// driver refuses is named before the driver sees it.
#[test]
#[ignore = "requires a CUDA device"]
fn tensor_maps_encode_tiled_descriptors() {
    let _driver = exclusive_driver();
    // The device's retained context is what the allocation belongs to.
    let device = Device::new(DEVICE).unwrap();
    let storage = DeviceAllocation::linear(DEVICE, 4096).unwrap();

    // A one-dimensional tensor of 64 elements, as the rank the driver takes.
    let one_dimensional_strides = [64 * 4];
    let one_dimensional = TensorMapSpec {
        // CU_TENSOR_MAP_DATA_TYPE_FLOAT32
        data_type: 7,
        global_address: storage.address(),
        global_dim: &[64, 1],
        global_strides: &one_dimensional_strides,
        box_dim: &[4, 1],
        element_strides: &[1, 1],
        interleave: Interleave::None,
        swizzle: Swizzle::None,
        l2_promotion: L2Promotion::None,
        oob_fill: OobFill::None,
    };
    let encoded = TensorMap::encode_tiled(&device, &one_dimensional).unwrap();
    assert!(
        encoded.as_bytes().iter().any(|byte| *byte != 0),
        "the driver encoded the descriptor"
    );

    let strides = [16 * 4];
    let rank_two = TensorMapSpec {
        // CU_TENSOR_MAP_DATA_TYPE_FLOAT32
        data_type: 7,
        global_address: storage.address(),
        global_dim: &[16, 16],
        global_strides: &strides,
        box_dim: &[4, 4],
        element_strides: &[1, 1],
        interleave: Interleave::None,
        swizzle: Swizzle::B128,
        l2_promotion: L2Promotion::B128,
        oob_fill: OobFill::NanRequestZeroFma,
    };
    TensorMap::encode_tiled(&device, &rank_two).unwrap();

    let missing_stride = TensorMapSpec {
        global_strides: &[16, 32],
        ..rank_two
    };
    assert!(matches!(
        TensorMap::encode_tiled(&device, &missing_stride),
        Err(Error::InvalidArgument(_))
    ));
    let rank_zero = TensorMapSpec {
        global_dim: &[],
        ..one_dimensional
    };
    assert!(matches!(
        TensorMap::encode_tiled(&device, &rank_zero),
        Err(Error::InvalidArgument(_))
    ));
    // The driver on this platform encodes no rank-one map, so the API states
    // the rank it takes instead of handing the description over.
    let rank_one = TensorMapSpec {
        global_dim: &[64],
        global_strides: &[],
        box_dim: &[4],
        element_strides: &[1],
        ..one_dimensional
    };
    assert!(matches!(
        TensorMap::encode_tiled(&device, &rank_one),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
#[ignore = "requires a CUDA device and NCCL"]
fn a_communicator_splits_into_a_subgroup_and_a_rank_left_out_gets_none() {
    let _driver = exclusive_driver();
    let id = UniqueId::new().unwrap();
    let world = Communicator::init(DEVICE, 1, 0, &id).unwrap();
    let tp = world
        .split(&[0], 0)
        .unwrap()
        .expect("the only rank is in the subgroup");
    assert_eq!((tp.rank(), tp.size()), (0, 1));

    let stream = Stream::new(DEVICE).unwrap();
    let data = DeviceAllocation::linear(DEVICE, 16).unwrap();
    let host = PinnedBuffer::new(DEVICE, 16).unwrap();
    // SAFETY: the buffers outlive the synchronization below.
    unsafe { memset_d32(data.address(), 41, 4, &stream).unwrap() };
    let words = DeviceRange::new(DevicePtr::from_address(data.address()), data.len());
    tp.all_reduce(
        &stream,
        words,
        words,
        4,
        platform::DataType::Int32,
        ReduceOp::Min,
    )
    .unwrap();
    // SAFETY: the buffer outlives the synchronization below.
    unsafe { memcpy_async(host.address(), data.address(), 16, &stream).unwrap() };
    stream.synchronize().unwrap();
    let reduced: Vec<u32> = read(&host)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_ne_bytes(*word))
        .collect();
    assert_eq!(reduced, [41; 4]);

    tp.destroy().unwrap();
    // A rank outside the subgroup still takes part in the split, and NCCL
    // hands it no communicator.
    assert!(world.split(&[], 0).unwrap().is_none());
    world.destroy().unwrap();
}

#[test]
#[ignore = "requires a CUDA device"]
fn the_sticky_error_probe_answers_from_another_thread() {
    let _driver = exclusive_driver();
    let context = PrimaryContext::retain(DEVICE).unwrap();
    // Warm the probe on this thread, where the context is current.
    assert!(context.sticky_error().unwrap().is_none());
    // The control thread probes a device whose context it has never entered.
    let elsewhere = context.clone();
    let poisoned =
        std::thread::spawn(move || elsewhere.sticky_error().map(|error| error.is_some()))
            .join()
            .unwrap()
            .unwrap();
    assert!(
        !poisoned,
        "a healthy context must not read as sticky from a thread that has not entered it"
    );
}

/// One rank of [`two_ranks_exchange_blocks_and_reduce_across_their_group`],
/// run on its own thread as design 2.4 gives every device one.
///
/// Rank `r` sends block `j` of two `u32` elements as `[100·j + 10·r, + 1]`, so
/// both the block's owner and the sender are visible in what each rank reads
/// back after the exchange.
fn two_rank_half(rank: i32, id: UniqueId) -> Result<(), String> {
    let device = DeviceId(rank);
    let world = Communicator::init(device, 2, rank, &id).map_err(|error| error.to_string())?;
    let group = world
        .split(&[0, 1], rank)
        .map_err(|error| error.to_string())?
        .ok_or("both ranks are in the group")?;
    assert_eq!((group.rank(), group.size()), (rank, 2));

    let stream = Stream::new(device).map_err(|error| error.to_string())?;
    let send = DeviceAllocation::linear(device, 16).map_err(|error| error.to_string())?;
    let recv = DeviceAllocation::linear(device, 16).map_err(|error| error.to_string())?;
    let host = PinnedBuffer::new(device, 16).map_err(|error| error.to_string())?;

    let mut pattern = [[0u8; 4]; 4];
    for (element, chunk) in pattern.iter_mut().enumerate() {
        let value = 100 * (element / 2) as u32 + 10 * rank as u32 + (element % 2) as u32;
        *chunk = value.to_ne_bytes();
    }
    // SAFETY: the host array is live and the stream is synchronized below.
    unsafe {
        memcpy_async(send.address(), pattern.as_ptr() as u64, 16, &stream)
            .map_err(|error| error.to_string())?;
        memset_d32(recv.address(), 0, 4, &stream).map_err(|error| error.to_string())?;
    }

    // Design 4.3's shape: the group is reached through a shared reference.
    let send_range = DeviceRange::new(DevicePtr::from_address(send.address()), 16);
    let recv_range = DeviceRange::new(DevicePtr::from_address(recv.address()), 16);
    group
        .all_to_all(
            &stream,
            send_range,
            recv_range,
            2,
            platform::DataType::Uint32,
        )
        .map_err(|error| error.to_string())?;
    // Sum the exchanged window across the group, in place.
    group
        .all_reduce(
            &stream,
            recv_range,
            recv_range,
            4,
            platform::DataType::Uint32,
            ReduceOp::Sum,
        )
        .map_err(|error| error.to_string())?;
    // SAFETY: the copy fills the pinned buffer and the stream is synchronized.
    unsafe { memcpy_async(host.address(), recv.address(), 16, &stream) }
        .map_err(|error| error.to_string())?;
    stream.synchronize().map_err(|error| error.to_string())?;

    // Rank r read `[100·r, 100·r + 1, 100·r + 10, 100·r + 11]`; summed over
    // two ranks that is the same on both, and wrong block order, a wrong
    // per-peer count or a single rank's data all give a different line.
    let words: Vec<u32> = read(&host)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_ne_bytes(*word))
        .collect();
    assert_eq!(words, [100, 102, 120, 122], "rank {rank}");

    group.destroy().map_err(|error| error.to_string())?;
    world.destroy().map_err(|error| error.to_string())
}

/// Two ranks in one process, one thread per device (design 2.4): the shape the
/// workers of one process run a world's groups in, and the only test here that would catch a
/// collective's per-peer count being confused with a buffer's element count.
#[test]
#[ignore = "requires two CUDA devices and NCCL"]
fn two_ranks_exchange_blocks_and_reduce_across_their_group() {
    let _driver = exclusive_driver();
    let devices = Device::count().unwrap();
    assert!(
        devices >= 2,
        "the exchange needs two CUDA devices, and {devices} are visible"
    );
    let id = UniqueId::new().unwrap();
    let ranks: Vec<_> = (0..2)
        .map(|rank| std::thread::spawn(move || two_rank_half(rank, id)))
        .collect();
    for (rank, handle) in ranks.into_iter().enumerate() {
        match handle.join() {
            Ok(Ok(())) => {}
            Ok(Err(error)) => panic!("rank {rank}: {error}"),
            Err(panic) => std::panic::resume_unwind(panic),
        }
    }
}
