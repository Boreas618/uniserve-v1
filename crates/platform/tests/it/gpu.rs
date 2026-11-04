//! Behavior against a real driver. These tests need a CUDA device and use a
//! few megabytes of storage on device 0; run them with
//! `cargo test -p platform --test it -- --ignored --test-threads=1`.

use std::collections::BTreeSet;
use std::ffi::c_void;
use std::path::Path;
use std::process::Command;
use std::sync::{Mutex, MutexGuard, PoisonError, mpsc};
use std::thread;
use std::time::Duration;

use platform::nccl::{Communicator, ReduceOp, UniqueId};
use platform::{
    Algorithm, Capture, CaptureMode, CaptureStatus, Cluster, ComputeType, Convolution,
    ConvolutionArgs, CopyBatch, Cudnn, DataType, Device, DeviceAllocation, DeviceAttribute,
    DeviceId, DevicePtr, DeviceRange, Engine, Error, Event, EventFlags, FunctionAttribute,
    GraphNodeKind, HandleType, Interleave, KernelNodeParams, L2Promotion, LaunchConfig, Library,
    Matmul, MatmulArgs, MatmulDescriptor, Nvml, OobFill, PeerMapping, PinnedBuffer, PrimaryContext,
    Protection, SegmentedMapping, ShareableHandle, SourceOrder, Stream, Strided, StridedCopy,
    Swizzle, Tensor, TensorDataType, TensorMap, TensorMapSpec, VirtualPool, memcpy_async,
    memset_2d_d8, memset_d8, memset_d32, reclaim,
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

#[test]
#[ignore = "requires a CUDA device"]
fn strided_copies_match_the_reference_in_every_direction() {
    let _driver = exclusive_driver();
    // [layer, row, column] of u16: dense on the host, padded rows and layers
    // on the device, so the plan is one 3D pitched copy.
    let shape = [3_u64, 4, 6];
    let host_strides = [24_i64, 6, 1];
    let device_strides = [60_i64, 12, 1];
    let stream = Stream::new(DEVICE).unwrap();
    let device = DeviceAllocation::linear(DEVICE, 3 * 60 * 2).unwrap();
    let staging = DeviceAllocation::linear(DEVICE, 3 * 60 * 2).unwrap();
    let mut host = PinnedBuffer::new(DEVICE, 3 * 24 * 2).unwrap();
    let pattern: Vec<u8> = (0..3 * 24 * 2).map(|index| (index * 5 + 1) as u8).collect();
    // SAFETY: no copy is pending on the buffer.
    unsafe { host.as_mut_slice() }.copy_from_slice(&pattern);

    let operand = |address, strides| Strided {
        address,
        element_bytes: 2,
        shape: &shape,
        strides,
    };
    let host_operand = operand(host.address(), &host_strides[..]);

    // SAFETY: every operand lies inside its allocation, all of which outlive
    // the synchronization below.
    unsafe {
        memset_d8(device.address(), 0, device.len(), &stream).unwrap();

        // Host to device, device to device, device back to host.
        let mut upload = StridedCopy::for_device(DEVICE, SourceOrder::Stream).unwrap();
        upload
            .plan(&operand(device.address(), &device_strides), &host_operand)
            .unwrap();
        assert_eq!(upload.pitched().len(), 1);
        assert_eq!(upload.pitched().copies()[0].depth, 3);
        upload.submit(&stream).unwrap();

        let mut copy = StridedCopy::for_device(DEVICE, SourceOrder::Stream).unwrap();
        copy.plan(
            &operand(staging.address(), &device_strides),
            &operand(device.address(), &device_strides),
        )
        .unwrap();
        copy.submit(&stream).unwrap();

        memset_d8(host.address(), 0, host.len(), &stream).unwrap();
        let mut download = StridedCopy::for_device(DEVICE, SourceOrder::Stream).unwrap();
        download
            .plan(&host_operand, &operand(staging.address(), &device_strides))
            .unwrap();
        download.submit(&stream).unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(read(&host), pattern);
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

    thread::sleep(Duration::from_millis(20));
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

/// Names the role of the process `interprocess_event_process` runs in.
const EVENT_PROCESS: &str = "UNISERVE_CUDA_TEST_EVENT_PROCESS";

/// The `/dev/shm` objects this process maps that still have a name.
fn mapped_shared_memory() -> BTreeSet<String> {
    std::fs::read_to_string("/proc/self/maps")
        .unwrap()
        .lines()
        .filter_map(|line| line.find("/dev/shm/").map(|start| &line[start..]))
        .filter(|path| !path.ends_with(" (deleted)"))
        .map(str::to_owned)
        .collect()
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
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "gpu::interprocess_event_process",
            "--ignored",
            "--nocapture",
        ])
        .env(EVENT_PROCESS, "1")
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

/// The process half of
/// `resetting_primary_contexts_releases_interprocess_event_storage`: exports
/// an interprocess event as a rank does, reports the shared memory that
/// mapped, destroys its primary contexts and exits.
#[test]
#[ignore = "runs as the process of the interprocess event storage test"]
fn interprocess_event_process() {
    if std::env::var_os(EVENT_PROCESS).is_none() {
        return;
    }
    // A reference this process never releases, as the CUDA runtime holds
    // one in a rank: without it, the last owner's release below would
    // destroy the context by itself.
    let _runtime = std::mem::ManuallyDrop::new(PrimaryContext::retain(DEVICE).unwrap());
    let stream = Stream::new(DEVICE).unwrap();
    let before = mapped_shared_memory();
    let event = Event::new(EventFlags {
        interprocess: true,
        ..EventFlags::default()
    });
    event.record(&stream).unwrap();
    event.ipc_handle().unwrap();
    stream.synchronize().unwrap();
    for path in mapped_shared_memory().difference(&before) {
        println!("mapped {path}");
    }

    drop(event);
    drop(stream);
    // SAFETY: this process uses no device after the reset.
    let devices = unsafe { PrimaryContext::reset_all() }.unwrap();
    assert!(devices.contains(&DEVICE));
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

/// A pool's storage keeps its addresses and contents across a suspension,
/// and a graph captured over it before runs correctly after the restore; a
/// freed chunk is given out again at its address.
#[test]
#[ignore = "requires a CUDA device"]
fn a_pool_keeps_addresses_and_contents_across_a_suspension() {
    let _driver = exclusive_driver();
    let mut pool = VirtualPool::reserve(DEVICE, 64 << 20).unwrap();
    let first = pool.allocate(1 << 20).unwrap();
    let second = pool.allocate(1 << 20).unwrap();
    assert_ne!(first, second);
    assert!(pool.contains(first) && pool.contains(second));
    let stream = Stream::new(DEVICE).unwrap();
    let host = PinnedBuffer::new(DEVICE, 256).unwrap();

    // The first chunk holds 5s written before the suspension; a graph
    // captured now fills the second with 9s at every launch.
    // SAFETY: the ranges lie inside the pool's chunks.
    unsafe { memset_d8(first, 5, 256, &stream) }.unwrap();
    stream.synchronize().unwrap();
    let capture = Capture::begin(&stream, CaptureMode::Relaxed).unwrap();
    // SAFETY: the range lies inside the pool's chunk.
    unsafe { memset_d8(second, 9, 256, &stream) }.unwrap();
    let fill = capture.end().unwrap().instantiate().unwrap();

    let mapped = pool.mapped_bytes();
    pool.suspend().unwrap();
    assert!(
        pool.allocate(1 << 20).is_err(),
        "a suspended pool gives out nothing"
    );
    pool.restore().unwrap();
    assert_eq!(pool.mapped_bytes(), mapped);

    // SAFETY: the ranges lie inside the restored chunks, at their addresses.
    unsafe {
        memcpy_async(host.address(), first, 256, &stream).unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(read(&host), vec![5; 256]);
    fill.launch(&stream).unwrap();
    // SAFETY: as above.
    unsafe {
        memcpy_async(host.address(), second, 256, &stream).unwrap();
    }
    stream.synchronize().unwrap();
    assert_eq!(read(&host), vec![9; 256]);

    drop(fill);
    pool.free(second);
    assert_eq!(pool.allocate(1 << 20).unwrap(), second);
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
/// it, and a plan bounded by the segment length (`plan_within`) copies
/// each column on its own, which the driver serves.
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
    let shape = [2_u64, width as u64];
    let destination = Strided {
        address: target.address(),
        element_bytes: 1,
        shape: &shape,
        strides: &[width as i64, 1],
    };
    let source = Strided {
        address: segmented.address(),
        element_bytes: 1,
        shape: &shape,
        strides: &[pitch as i64, 1],
    };
    let mut spanning = StridedCopy::for_device(DEVICE, SourceOrder::Stream).unwrap();
    spanning.plan(&destination, &source).unwrap();
    assert_eq!(spanning.pitched().len(), 1, "one pitched copy of two rows");
    let mut bounded = StridedCopy::for_device(DEVICE, SourceOrder::Stream).unwrap();
    bounded
        .plan_within(&destination, &source, [u64::MAX, granularity as u64])
        .unwrap();
    assert!(
        bounded.pitched().is_empty(),
        "one contiguous copy per column"
    );
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
    let mut communicator = Communicator::init(DEVICE, 1, 0, &id).unwrap();
    assert_eq!((communicator.rank(), communicator.size()), (0, 1));

    // SAFETY: the buffers outlive the synchronization below.
    unsafe { memset_d32(data.address(), 41, 4, &stream).unwrap() };
    let words = DeviceRange::new(DevicePtr::from_address(data.address()), data.len());
    communicator
        .all_reduce(&stream, words, words, 4, DataType::I32, ReduceOp::Min)
        .unwrap();
    communicator
        .broadcast(&stream, words, words, 4, DataType::I32, 0)
        .unwrap();
    // A collective whose stated ranges cannot hold the elements it asks for
    // fails before any NCCL call is made.
    let short = DeviceRange::new(DevicePtr::from_address(data.address()), 8);
    let refused = communicator
        .all_reduce(&stream, short, words, 4, DataType::I32, ReduceOp::Min)
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

/// PTX text of the kernels the launch, capture and rebind tests use.
///
/// `report` writes, for each block, its three grid coordinates, the first
/// thread's three coordinates and the dynamic shared-memory size the driver
/// allocated for the launch, seven 32-bit words per block; only thread 0 of a
/// block stores, so a block of more than one thread still writes once.
///
/// `write` stores one value per thread, so a launch covers a contiguous run of
/// words and a rebinding test can tell whose work ran.
const KERNELS: &str = "\
.version 8.0\n\
.target sm_90\n\
.address_size 64\n\
.visible .entry report(.param .u64 out)\n\
{\n\
    .reg .u64 %rd<4>;\n\
    .reg .u32 %r<13>;\n\
    .reg .pred %p;\n\
    ld.param.u64 %rd1, [out];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ctaid.y;\n\
    mov.u32 %r3, %ctaid.z;\n\
    mov.u32 %r4, %tid.x;\n\
    mov.u32 %r5, %tid.y;\n\
    mov.u32 %r6, %tid.z;\n\
    mov.u32 %r7, %dynamic_smem_size;\n\
    mov.u32 %r8, %nctaid.x;\n\
    mov.u32 %r9, %nctaid.y;\n\
    mad.lo.u32 %r10, %r9, %r3, %r2;\n\
    mad.lo.u32 %r10, %r8, %r10, %r1;\n\
    mul.wide.u32 %rd2, %r10, 28;\n\
    add.u64 %rd2, %rd1, %rd2;\n\
    setp.eq.u32 %p, %r4, 0;\n\
    @%p st.global.u32 [%rd2], %r1;\n\
    add.u64 %rd3, %rd2, 4;\n\
    @%p st.global.u32 [%rd3], %r2;\n\
    add.u64 %rd3, %rd2, 8;\n\
    @%p st.global.u32 [%rd3], %r3;\n\
    add.u64 %rd3, %rd2, 12;\n\
    @%p st.global.u32 [%rd3], %r4;\n\
    add.u64 %rd3, %rd2, 16;\n\
    @%p st.global.u32 [%rd3], %r5;\n\
    add.u64 %rd3, %rd2, 20;\n\
    @%p st.global.u32 [%rd3], %r6;\n\
    add.u64 %rd3, %rd2, 24;\n\
    @%p st.global.u32 [%rd3], %r7;\n\
    ret;\n\
}\n\
.visible .entry report_cluster(.param .u64 out)\n\
{\n\
    .reg .u64 %rd<3>;\n\
    .reg .u32 %r<11>;\n\
    .reg .pred %p;\n\
    ld.param.u64 %rd1, [out];\n\
    mov.u32 %r1, %ctaid.x;\n\
    mov.u32 %r2, %ctaid.y;\n\
    mov.u32 %r3, %cluster_ctaid.x;\n\
    mov.u32 %r4, %cluster_ctaid.y;\n\
    mov.u32 %r5, %cluster_ctarank;\n\
    mov.u32 %r6, %cluster_nctaid.x;\n\
    mov.u32 %r7, %cluster_nctaid.y;\n\
    mov.u32 %r8, %nctaid.x;\n\
    mov.u32 %r9, %tid.x;\n\
    mad.lo.u32 %r10, %r8, %r2, %r1;\n\
    mul.wide.u32 %rd2, %r10, 32;\n\
    add.u64 %rd2, %rd1, %rd2;\n\
    setp.eq.u32 %p, %r9, 0;\n\
    @%p st.global.u32 [%rd2], %r1;\n\
    add.u64 %rd2, %rd2, 4;\n\
    @%p st.global.u32 [%rd2], %r2;\n\
    add.u64 %rd2, %rd2, 4;\n\
    @%p st.global.u32 [%rd2], %r3;\n\
    add.u64 %rd2, %rd2, 4;\n\
    @%p st.global.u32 [%rd2], %r4;\n\
    add.u64 %rd2, %rd2, 4;\n\
    @%p st.global.u32 [%rd2], %r5;\n\
    add.u64 %rd2, %rd2, 4;\n\
    @%p st.global.u32 [%rd2], %r6;\n\
    add.u64 %rd2, %rd2, 4;\n\
    @%p st.global.u32 [%rd2], %r7;\n\
    add.u64 %rd2, %rd2, 4;\n\
    @%p st.global.u32 [%rd2], %r9;\n\
    ret;\n\
}\n\
.visible .entry write(.param .u64 out, .param .u32 value)\n\
{\n\
    .reg .u64 %rd<3>;\n\
    .reg .u32 %r<3>;\n\
    ld.param.u64 %rd1, [out];\n\
    ld.param.u32 %r1, [value];\n\
    mov.u32 %r2, %tid.x;\n\
    mul.wide.u32 %rd2, %r2, 4;\n\
    add.u64 %rd2, %rd1, %rd2;\n\
    st.global.u32 [%rd2], %r1;\n\
    ret;\n\
}\n";

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

/// A library of [`KERNELS`] on device 0.
fn kernels(device: &Device) -> Library {
    Library::load(device, KERNELS.as_bytes()).unwrap()
}

/// A binary library image (a fatbin, so it loads on any device the PTX was
/// compiled for) contains NUL bytes, which the driver reads by the image's own
/// headers; it loads and its kernel runs.
#[test]
#[ignore = "requires a CUDA device"]
fn libraries_load_a_binary_image_and_launch_its_kernel() {
    let _driver = exclusive_driver();
    let device = Device::new(DEVICE).unwrap();
    assert!(matches!(
        Library::load(&device, &[]),
        Err(Error::InvalidArgument(_))
    ));

    let library =
        Library::load(&device, include_bytes!("../fixtures/add_constant.fatbin")).unwrap();
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

    // A cluster is one launch attribute more; it needs a compute capability
    // that has clusters (sm_90 and later), which is every device this branch
    // targets. The kernel reads the cluster's own registers, so a launch that
    // dropped the attribute cannot pass: without it every block is a cluster
    // of one.
    if device.architecture().0 >= 9 {
        let cluster = library.kernel("report_cluster").unwrap();
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

/// Capture records the node each launch adds: nothing before the first launch,
/// one node per launch after it, and the graph replays them in order.
#[test]
#[ignore = "requires a CUDA device"]
fn capture_records_the_node_of_each_launch() {
    let _driver = exclusive_driver();
    let device = Device::new(DEVICE).unwrap();
    let library = kernels(&device);
    let write = library.kernel("write").unwrap();
    let stream = Stream::new(DEVICE).unwrap();
    let storage = DeviceAllocation::linear(DEVICE, 32).unwrap();
    let host = PinnedBuffer::new(DEVICE, 32).unwrap();
    let mut out = storage.address();

    let mut capture = Capture::begin(&stream, CaptureMode::Relaxed).unwrap();
    assert!(capture.record().unwrap().is_empty(), "no operation yet");

    let mut first = 7u32;
    let mut parameters: [*mut c_void; 2] = [(&raw mut out).cast(), (&raw mut first).cast()];
    write
        .launch_ex(
            &stream,
            &LaunchConfig::new([1, 1, 1], [8, 1, 1]),
            &mut parameters,
        )
        .unwrap();
    assert_eq!(capture.record().unwrap().len(), 1, "the first launch");

    let mut second = 9u32;
    let mut parameters: [*mut c_void; 2] = [(&raw mut out).cast(), (&raw mut second).cast()];
    write
        .launch_ex(
            &stream,
            &LaunchConfig::new([1, 1, 1], [8, 1, 1]),
            &mut parameters,
        )
        .unwrap();
    assert_eq!(capture.record().unwrap().len(), 1, "the second launch");

    let graph = capture.end().unwrap();
    assert_eq!(graph.node_count().unwrap(), 2);
    assert_eq!(graph.nodes().unwrap().len(), 2);

    let executable = graph.instantiate().unwrap();
    executable.launch(&stream).unwrap();
    // SAFETY: the copy reads the allocation into the pinned buffer.
    unsafe { memcpy_async(host.address(), storage.address(), 32, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        words(&host),
        vec![9; 8],
        "both captured launches ran in order"
    );
}

/// Rebinding replaces a captured kernel node's function, grid, block, dynamic
/// shared memory and arguments; the executable then runs the new work, while
/// the template node is unchanged.
#[test]
#[ignore = "requires a CUDA device"]
fn rebinding_changes_a_kernel_node_function_and_arguments() {
    let _driver = exclusive_driver();
    let device = Device::new(DEVICE).unwrap();
    let library = kernels(&device);
    let write = library.kernel("write").unwrap();
    let report = library.kernel("report").unwrap();
    let stream = Stream::new(DEVICE).unwrap();
    let storage = DeviceAllocation::linear(DEVICE, 32).unwrap();
    let host = PinnedBuffer::new(DEVICE, 32).unwrap();
    let mut out = storage.address();

    let mut capture = Capture::begin(&stream, CaptureMode::Relaxed).unwrap();
    let mut value = 7u32;
    let mut parameters: [*mut c_void; 2] = [(&raw mut out).cast(), (&raw mut value).cast()];
    write
        .launch_ex(
            &stream,
            &LaunchConfig::new([1, 1, 1], [8, 1, 1]),
            &mut parameters,
        )
        .unwrap();
    let _ = capture.record().unwrap();
    let graph = capture.end().unwrap();
    let node = graph.nodes().unwrap().remove(0);
    let executable = graph.instantiate().unwrap();

    // The report kernel takes one argument and one block of two threads, none
    // of which the captured node had.
    let parameters: [*mut c_void; 1] = [(&raw mut out).cast()];
    let params = KernelNodeParams {
        function: &report,
        grid: [1, 1, 1],
        block: [2, 1, 1],
        shared_memory: 0,
        parameters: &parameters,
    };
    executable
        .set_kernel_node_params(node.clone(), &params)
        .unwrap();

    executable.launch(&stream).unwrap();
    // SAFETY: the copy reads the allocation into the pinned buffer.
    unsafe { memcpy_async(host.address(), storage.address(), 32, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(
        words(&host)[..7],
        [0, 0, 0, 0, 0, 0, 0],
        "the rebound function ran with its new grid"
    );

    // The template still holds the captured node, so a second executable of it
    // runs the original work.
    let original = graph.instantiate().unwrap();
    original.launch(&stream).unwrap();
    // SAFETY: as above.
    unsafe { memcpy_async(host.address(), storage.address(), 32, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(words(&host), vec![7; 8], "the template node is unchanged");
}

/// A row-major matrix product runs on an algorithm cuBLASLt's heuristic
/// offers, and its result is the arithmetic the descriptor names.
#[test]
fn a_matrix_product_runs_on_an_algorithm_the_heuristic_offers() {
    let _driver = exclusive_driver();
    let device = Device::new(DEVICE).unwrap();
    let stream = Stream::new(DEVICE).unwrap();
    let (m, n, k) = (4u64, 6, 3);
    // Small integers, so an FP32 product is exact and a tolerance would only
    // hide a wrong layout.
    let a: Vec<f32> = (0..m * k).map(|index| (index % 7) as f32 - 3.0).collect();
    let b: Vec<f32> = (0..k * n).map(|index| (index % 5) as f32 + 1.0).collect();
    let mut expected = vec![0f32; (m * n) as usize];
    for row in 0..m as usize {
        for col in 0..n as usize {
            let mut sum = 0f64;
            for inner in 0..k as usize {
                sum += a[row * k as usize + inner] as f64 * b[inner * n as usize + col] as f64;
            }
            expected[row * n as usize + col] = sum as f32;
        }
    }

    let a_device = DeviceAllocation::linear(DEVICE, a.len() * 4).unwrap();
    let b_device = DeviceAllocation::linear(DEVICE, b.len() * 4).unwrap();
    let c_device = DeviceAllocation::linear(DEVICE, expected.len() * 4).unwrap();
    let d_device = DeviceAllocation::linear(DEVICE, expected.len() * 4).unwrap();
    // SAFETY: each copy fills a device allocation of exactly the bytes it
    // reads from a live slice, and the stream is synchronized afterwards.
    unsafe {
        memcpy_async(a_device.address(), a.as_ptr() as u64, a.len() * 4, &stream).unwrap();
        memcpy_async(b_device.address(), b.as_ptr() as u64, b.len() * 4, &stream).unwrap();
        memset_d32(c_device.address(), 0, expected.len(), &stream).unwrap();
        memset_d32(d_device.address(), 0, expected.len(), &stream).unwrap();
    }
    stream.synchronize().unwrap();

    let matmul = Matmul::new(&device).unwrap();
    let descriptor = MatmulDescriptor::row_major(m, n, k, DataType::F32, ComputeType::F32);
    let algorithms = matmul.candidates(&descriptor, 0, 4).unwrap();
    assert!(
        !algorithms.is_empty(),
        "cuBLASLt's heuristic proposes an algorithm for a 4x6x3 product"
    );
    let algorithm = algorithms[0];
    let workspace = if algorithm.workspace_bytes() > 0 {
        DeviceAllocation::linear(DEVICE, algorithm.workspace_bytes()).unwrap()
    } else {
        DeviceAllocation::linear(DEVICE, 4).unwrap()
    };
    let range = |allocation: &DeviceAllocation| {
        DeviceRange::new(
            DevicePtr::from_address(allocation.address()),
            allocation.len(),
        )
    };
    let args = MatmulArgs {
        a: range(&a_device),
        b: range(&b_device),
        c: range(&c_device),
        d: range(&d_device),
        alpha: 1.0,
        beta: 0.0,
        // A multiplication without the bias epilogue names no bias vector.
        bias: DeviceRange::new(DevicePtr::from_address(0), 0),
        workspace: range(&workspace),
    };
    matmul.run(&descriptor, &algorithm, &args, &stream).unwrap();
    // A range too small for the descriptor's shape is refused by name, before
    // any cuBLASLt call.
    let short = MatmulArgs {
        a: DeviceRange::new(DevicePtr::from_address(a_device.address()), 8),
        ..args
    };
    let refused = matmul
        .run(&descriptor, &algorithm, &short, &stream)
        .unwrap_err();
    assert!(matches!(refused, Error::InvalidArgument(_)), "{refused:?}");
    stream.synchronize().unwrap();

    let host = PinnedBuffer::new(DEVICE, expected.len() * 4).unwrap();
    // SAFETY: the copy reads the result allocation into a pinned buffer of its
    // size, and the stream is synchronized afterwards.
    unsafe {
        memcpy_async(
            host.address(),
            d_device.address(),
            expected.len() * 4,
            &stream,
        )
    }
    .unwrap();
    stream.synchronize().unwrap();
    let result: Vec<f32> = read(&host)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect();
    assert_eq!(
        result, expected,
        "D = A·B with A 4x3 row-major, B 3x6 row-major"
    );

    // The algorithm is what the program's tuning data carries: the record
    // reproduces the same algorithm's bytes.
    let record = algorithm.to_bytes();
    let restored = Algorithm::from_bytes(&record, algorithm.workspace_bytes());
    assert_eq!(restored.to_bytes(), record, "the tuning record round-trips");
}

/// The read half of a rebind walk: a template's node gives back the kernel it
/// captured, its grid, block and the values of its arguments, and setting those
/// on another executable's node runs the same work.
#[test]
fn kernel_node_params_read_back_what_a_capture_recorded() {
    let _driver = exclusive_driver();
    let device = Device::new(DEVICE).unwrap();
    let library = kernels(&device);
    let write = library.kernel("write").unwrap();
    let stream = Stream::new(DEVICE).unwrap();
    let storage = DeviceAllocation::linear(DEVICE, 32).unwrap();
    let host = PinnedBuffer::new(DEVICE, 32).unwrap();
    let mut out = storage.address();
    let mut value = 9u32;
    let mut parameters: [*mut c_void; 2] = [(&raw mut out).cast(), (&raw mut value).cast()];

    let mut capture = Capture::begin(&stream, CaptureMode::Relaxed).unwrap();
    write
        .launch_ex(
            &stream,
            &LaunchConfig::new([2, 1, 1], [4, 1, 1]),
            &mut parameters,
        )
        .unwrap();
    let _ = capture.record().unwrap();
    let graph = capture.end().unwrap();
    let node = graph.nodes().unwrap().remove(0);

    assert_eq!(node.kind().unwrap(), GraphNodeKind::Kernel);
    let read = graph.kernel_node_params(&node).unwrap();
    assert_eq!(read.grid, [2, 1, 1], "the grid the launch carried");
    assert_eq!(read.block, [4, 1, 1], "the block the launch carried");
    assert_eq!(read.shared_memory, 0);
    assert_eq!(read.parameters.len(), 2, "the kernel takes two parameters");
    assert_eq!(
        u64::from_ne_bytes(read.parameters[0][..8].try_into().unwrap()),
        out,
        "the address the launch carried"
    );
    assert_eq!(
        u32::from_ne_bytes(read.parameters[1][..4].try_into().unwrap()),
        value,
        "the value the launch carried"
    );

    // Setting what was read on an executable's node runs the captured work.
    let executable = graph.instantiate().unwrap();
    executable
        .set_kernel_node_params_owned(node.clone(), &read)
        .unwrap();
    executable.launch(&stream).unwrap();
    // SAFETY: the copy reads the allocation into the pinned buffer.
    unsafe { memcpy_async(host.address(), storage.address(), 32, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(words(&host), vec![9; 8], "the read-back launch ran");
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

    let mut capture = Capture::begin(&stream, CaptureMode::Relaxed).unwrap();
    let mut one = 1u32;
    let mut parameters: [*mut c_void; 2] = [(&raw mut out).cast(), (&raw mut one).cast()];
    write
        .launch_ex(
            &stream,
            &LaunchConfig::new([1, 1, 1], [8, 1, 1]),
            &mut parameters,
        )
        .unwrap();
    capture.record().unwrap();
    let before = capture.end().unwrap();
    let before_exec = before.instantiate().unwrap();
    assert_eq!(before.node_count().unwrap(), 1);

    let mut capture = Capture::begin(&stream, CaptureMode::Relaxed).unwrap();
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
        capture.record().unwrap();
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

/// A tensor map encodes the layout a TMA kernel moves: the driver accepts a
/// rank-1 and a rank-2 description of device storage, and the description is
/// checked against its rank before the driver sees it.
#[test]
#[ignore = "requires a CUDA device"]
fn tensor_maps_encode_tiled_descriptors() {
    let _driver = exclusive_driver();
    // The device's retained context is what the allocation belongs to.
    let device = Device::new(DEVICE).unwrap();
    let storage = DeviceAllocation::linear(DEVICE, 4096).unwrap();

    let rank_one = TensorMapSpec {
        data_type: TensorDataType::Float32,
        global_address: storage.address(),
        global_dim: &[64],
        global_strides: &[],
        box_dim: &[4],
        element_strides: &[1],
        interleave: Interleave::None,
        swizzle: Swizzle::None,
        l2_promotion: L2Promotion::None,
        oob_fill: OobFill::None,
    };
    let encoded = TensorMap::encode_tiled(&device, &rank_one).unwrap();
    assert!(
        encoded.as_bytes().iter().any(|byte| *byte != 0),
        "the driver encoded the descriptor"
    );

    let strides = [16 * 4];
    let rank_two = TensorMapSpec {
        data_type: TensorDataType::Float32,
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
        ..rank_one
    };
    assert!(matches!(
        TensorMap::encode_tiled(&device, &rank_zero),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_communicator_splits_into_a_subgroup_and_a_rank_left_out_gets_none() {
    let _driver = exclusive_driver();
    let id = UniqueId::new().unwrap();
    let world = Communicator::init(DEVICE, 1, 0, &id).unwrap();
    let mut tp = world
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
    tp.all_reduce(&stream, words, words, 4, DataType::I32, ReduceOp::Min)
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

/// One tensor of floats as bytes, for a copy in or out.
fn float_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: a float slice is readable as its own bytes.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

/// The bytes of a copy read back as floats.
fn floats(bytes: Vec<u8>) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

/// The definition of a one-dimensional cross-correlation, computed on the
/// host: `y[n][k][o] = sum over c, j of x[n][c][i] * w[k][c][j]` with
/// `o = i · stride - padding + j · dilation`.
///
/// This is the reference the convolution's own numbers are compared with; it
/// is written from the definition, not from the device's result.
#[allow(clippy::too_many_arguments)]
fn forward_reference(
    x: &[f32],
    x_dims: [usize; 3],
    w: &[f32],
    w_dims: [usize; 3],
    y_dims: [usize; 3],
    stride: i64,
    padding: i64,
    dilation: i64,
) -> Vec<f32> {
    let [batch, channels, length] = x_dims;
    let [filters, _, kernel] = w_dims;
    let [_, _, out_length] = y_dims;
    let mut y = vec![0.0f32; batch * filters * out_length];
    for n in 0..batch {
        for k in 0..filters {
            for o in 0..out_length {
                let mut total = 0.0f64;
                for c in 0..channels {
                    for j in 0..kernel {
                        let i = (o as i64 + padding - j as i64 * dilation) / stride;
                        let aligned = i >= 0
                            && (i < length as i64)
                            && o as i64 == i * stride - padding + j as i64 * dilation;
                        if aligned {
                            let value = x[(n * channels + c) * length + i as usize];
                            let weight = w[(k * channels + c) * kernel + j];
                            total += f64::from(value) * f64::from(weight);
                        }
                    }
                }
                y[(n * filters + k) * out_length + o] = total as f32;
            }
        }
    }
    y
}

/// The definition of a one-dimensional transposed convolution, computed on
/// the host: every input element is scattered through the filter,
/// `y[n][k][i · stride - padding + j · dilation] += x[n][c][i] · w[c][k][j]`.
#[allow(clippy::too_many_arguments)]
fn transposed_reference(
    x: &[f32],
    x_dims: [usize; 3],
    w: &[f32],
    w_dims: [usize; 3],
    y_dims: [usize; 3],
    stride: i64,
    padding: i64,
    dilation: i64,
) -> Vec<f32> {
    let [batch, channels, length] = x_dims;
    let [_, filters, kernel] = w_dims;
    let [_, _, out_length] = y_dims;
    let mut y = vec![0.0f32; batch * filters * out_length];
    for n in 0..batch {
        for c in 0..channels {
            for i in 0..length {
                for k in 0..filters {
                    for j in 0..kernel {
                        let o = i as i64 * stride - padding + j as i64 * dilation;
                        if o >= 0 && o < out_length as i64 {
                            let value = x[(n * channels + c) * length + i];
                            let weight = w[(c * filters + k) * kernel + j];
                            y[(n * filters + k) * out_length + o as usize] += value * weight;
                        }
                    }
                }
            }
        }
    }
    y
}

/// Runs `convolution` through cuDNN on device storage, with the given host
/// tensors, and answers what the device computed.
fn convolve(
    cudnn: &Cudnn,
    stream: &Stream,
    convolution: &Convolution,
    x: &[f32],
    w: &[f32],
    y_len: usize,
    engine: Option<platform::Engine>,
) -> Vec<f32> {
    let plan = cudnn.plan(convolution, engine).unwrap();
    let input = DeviceAllocation::linear(DEVICE, x.len() * 4).unwrap();
    let filter = DeviceAllocation::linear(DEVICE, w.len() * 4).unwrap();
    let output = DeviceAllocation::linear(DEVICE, y_len * 4).unwrap();
    let workspace = DeviceAllocation::linear(DEVICE, plan.workspace_bytes().max(4)).unwrap();
    let mut host_input = PinnedBuffer::new(DEVICE, x.len() * 4).unwrap();
    let mut host_filter = PinnedBuffer::new(DEVICE, w.len() * 4).unwrap();
    let host_output = PinnedBuffer::new(DEVICE, y_len * 4).unwrap();

    // SAFETY: each copy lies inside the buffers it names, and every one of
    // them outlives the synchronization below.
    unsafe {
        host_input.as_mut_slice().copy_from_slice(float_bytes(x));
        host_filter.as_mut_slice().copy_from_slice(float_bytes(w));
        memcpy_async(input.address(), host_input.address(), x.len() * 4, stream).unwrap();
        memcpy_async(filter.address(), host_filter.address(), w.len() * 4, stream).unwrap();
    }
    let args = ConvolutionArgs::new(
        DeviceRange::new(DevicePtr::from_address(input.address()), x.len() * 4),
        DeviceRange::new(DevicePtr::from_address(filter.address()), w.len() * 4),
        DeviceRange::new(DevicePtr::from_address(output.address()), y_len * 4),
        DeviceRange::new(
            DevicePtr::from_address(workspace.address()),
            workspace.len(),
        ),
    );
    cudnn.run(&plan, args, stream).unwrap();
    // SAFETY: the copy reads the output allocation into a pinned buffer of
    // its size.
    unsafe {
        memcpy_async(host_output.address(), output.address(), y_len * 4, stream).unwrap();
    }
    stream.synchronize().unwrap();
    floats(read(&host_output))
}

/// Compares a computed convolution with the definition's numbers.
fn assert_close(observed: &[f32], expected: &[f32]) {
    assert_eq!(observed.len(), expected.len());
    for (index, (observed, expected)) in observed.iter().zip(expected).enumerate() {
        let tolerance = 1e-4 * expected.abs().max(1.0);
        assert!(
            (observed - expected).abs() <= tolerance,
            "element {index}: {observed} where the definition gives {expected}"
        );
    }
}

/// cuDNN's forward convolution computes the definition of a `Conv1d`: the
/// engine the heuristic picks, the plan built from it, and the variant pack
/// the call passes all agree with a host reference of the same numbers.
#[test]
#[ignore = "requires a CUDA device"]
fn a_convolution_matches_the_definition_of_a_conv1d() {
    let _driver = exclusive_driver();
    let device = Device::new(DEVICE).unwrap();
    let cudnn = Cudnn::new(&device).unwrap();
    assert!(cudnn.version() >= 9_000, "cuDNN 9 is bound");
    let stream = Stream::new(DEVICE).unwrap();

    let (x_dims, w_dims) = ([1usize, 2, 5], [3usize, 2, 3]);
    let (stride, padding, dilation) = (2i64, 1i64, 1i64);
    let out_length = ((x_dims[2] as i64 + 2 * padding - dilation * (w_dims[2] as i64 - 1) - 1)
        / stride
        + 1) as usize;
    let y_dims = [x_dims[0], w_dims[0], out_length];

    let x: Vec<f32> = (0..x_dims.iter().product())
        .map(|i| i as f32 * 0.5 - 1.0)
        .collect();
    let w: Vec<f32> = (0..w_dims.iter().product())
        .map(|i| ((i * 3) % 5) as f32 * 0.25 - 0.5)
        .collect();
    let expected = forward_reference(&x, x_dims, &w, w_dims, y_dims, stride, padding, dilation);

    let convolution = Convolution::new(
        Tensor::contiguous(vec![1, 2, 5], DataType::F32),
        Tensor::contiguous(vec![3, 2, 3], DataType::F32),
        Tensor::contiguous(vec![1, 3, out_length as i64], DataType::F32),
        vec![padding],
        vec![stride],
        vec![dilation],
    );
    let observed = convolve(&cudnn, &stream, &convolution, &x, &w, expected.len(), None);

    assert_close(&observed, &expected);

    // The plan's engine is the record a program carries: eight bytes that
    // round-trip, and a plan built from the recorded engine runs the same
    // shape.
    let plan = cudnn.plan(&convolution, None).unwrap();
    let record = plan.engine().to_bytes();
    let recorded = Engine::from_bytes(&record).unwrap();
    assert_eq!(recorded, plan.engine(), "the engine record round-trips");
    assert_eq!(
        cudnn.plan(&convolution, Some(recorded)).unwrap().engine(),
        plan.engine(),
        "the recorded engine is the one the heuristic proposes"
    );

    // A range shorter than its tensor is named before the driver sees it.
    let bytes = plan.bytes();
    let input = DeviceAllocation::linear(DEVICE, bytes.x as usize).unwrap();
    let filter = DeviceAllocation::linear(DEVICE, bytes.w as usize).unwrap();
    let output = DeviceAllocation::linear(DEVICE, bytes.y as usize).unwrap();
    let workspace = DeviceAllocation::linear(DEVICE, plan.workspace_bytes().max(4)).unwrap();
    let short = ConvolutionArgs::new(
        DeviceRange::new(DevicePtr::from_address(input.address()), bytes.x as usize),
        DeviceRange::new(DevicePtr::from_address(filter.address()), bytes.w as usize),
        DeviceRange::new(
            DevicePtr::from_address(output.address()),
            bytes.y as usize - 4,
        ),
        DeviceRange::new(
            DevicePtr::from_address(workspace.address()),
            workspace.len(),
        ),
    );
    let error = cudnn
        .run(&plan, short, &stream)
        .expect_err("an output range one element short");
    assert!(error.to_string().contains("its tensor needs"), "{error}");
}

/// cuDNN's backward data pass computes the definition of a `ConvTranspose1d`,
/// with the layer's own weight and the padding and output padding the
/// description carries, and the engine the tuning data recorded is the one
/// the plan runs.
#[test]
#[ignore = "requires a CUDA device"]
fn a_transposed_convolution_matches_the_definition_of_a_conv_transpose1d() {
    let _driver = exclusive_driver();
    let device = Device::new(DEVICE).unwrap();
    let cudnn = Cudnn::new(&device).unwrap();
    let stream = Stream::new(DEVICE).unwrap();

    // The audio decoder's upsampling: a stride-two layer, a kernel of twice
    // the stride, half its padding, and the same length doubling.
    let (x_dims, w_dims) = ([1usize, 2, 4], [2usize, 3, 4]);
    let (stride, padding, dilation, output_padding) = (2i64, 1i64, 1i64, 0i64);
    let out_length = ((x_dims[2] as i64 - 1) * stride - 2 * padding
        + dilation * (w_dims[2] as i64 - 1)
        + output_padding
        + 1) as usize;
    let y_dims = [x_dims[0], w_dims[1], out_length];

    let x: Vec<f32> = (0..x_dims.iter().product())
        .map(|i| (i % 5) as f32 * 0.25 - 0.5)
        .collect();
    let w: Vec<f32> = (0..w_dims.iter().product())
        .map(|i| ((i * 7) % 5) as f32 * 0.25 - 0.5)
        .collect();
    let expected = transposed_reference(&x, x_dims, &w, w_dims, y_dims, stride, padding, dilation);

    let convolution = Convolution::transpose(
        Tensor::contiguous(vec![1, 2, 4], DataType::F32),
        Tensor::contiguous(vec![2, 3, 4], DataType::F32),
        Tensor::contiguous(vec![1, 3, out_length as i64], DataType::F32),
        vec![padding],
        vec![stride],
        vec![dilation],
        vec![output_padding],
    );
    let observed = convolve(&cudnn, &stream, &convolution, &x, &w, expected.len(), None);

    assert_close(&observed, &expected);
}
