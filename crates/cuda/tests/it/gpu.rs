//! Behavior against a real driver. These tests need a CUDA device and use a
//! few megabytes of memory on device 0; run them with
//! `cargo test -p cuda -- --ignored`.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;
use std::sync::{Mutex, MutexGuard, PoisonError, mpsc};
use std::thread;
use std::time::Duration;

use cuda::nccl::{Communicator, DataType, ReduceOp, UniqueId};
use cuda::{
    CaptureStatus, CopyBatch, DeviceAllocation, DeviceAttribute, DeviceGuard, DeviceId, Error,
    Event, EventFlags, GraphExec, HandleType, Nvml, PeerMapping, PinnedBuffer, PrimaryContext,
    Protection, SegmentedMapping, SourceOrder, Stream, Strided, StridedCopy, VirtualPool,
    memcpy_async, memset_2d_d8, memset_d8, memset_d32, reclaim,
};

const DEVICE: DeviceId = DeviceId(0);

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

fn read(buffer: &PinnedBuffer) -> Vec<u8> {
    // SAFETY: callers synchronize the stream that wrote the buffer first.
    unsafe { buffer.as_slice() }.to_vec()
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

    let memory = DEVICE.memory_info().unwrap();
    assert!(memory.free <= memory.total && memory.total > 0);
}

#[test]
#[ignore = "requires a CUDA device"]
fn fills_and_copies_round_trip_through_device_memory() {
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
/// alive: the driver removes that memory with its context, and a process's
/// exit alone does not.
#[test]
#[ignore = "requires a CUDA device"]
fn resetting_primary_contexts_releases_interprocess_event_memory() {
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
/// `resetting_primary_contexts_releases_interprocess_event_memory`: exports
/// an interprocess event as a rank does, reports the shared memory that
/// mapped, destroys its primary contexts and exits.
#[test]
#[ignore = "runs as the process of the interprocess event memory test"]
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

/// A retainable allocation's descriptor keeps its physical memory alive
/// after the allocation is released, as a Storage Server's does after its
/// creator exits: later mappings of the descriptor read the contents and
/// change their access in place, at an unchanged address.
#[test]
#[ignore = "requires a CUDA device"]
fn retained_memory_outlives_its_creator_and_changes_access_in_place() {
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
fn exportable_memory_maps_through_imports_and_aliases() {
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

    let exported = first.export().unwrap().expect("exportable memory exports");
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
    type BeginCapture = unsafe extern "C" fn(*mut std::ffi::c_void, i32) -> i32;
    type EndCapture =
        unsafe extern "C" fn(*mut std::ffi::c_void, *mut *mut std::ffi::c_void) -> i32;
    type DestroyGraph = unsafe extern "C" fn(*mut std::ffi::c_void) -> i32;

    // SAFETY: loading libcuda runs only the driver's own initializers.
    let library = unsafe { libloading::Library::new("libcuda.so.1") }.unwrap();
    // SAFETY: the signatures match cuda.h, and `library` stays loaded while the
    // copied function pointers are used; relaxed mode (2) keeps the capture
    // from interacting with other test threads.
    let (begin, end, destroy) = unsafe {
        (
            *library
                .get::<BeginCapture>(b"cuStreamBeginCapture_v2")
                .unwrap(),
            *library.get::<EndCapture>(b"cuStreamEndCapture").unwrap(),
            *library.get::<DestroyGraph>(b"cuGraphDestroy").unwrap(),
        )
    };

    let stream = Stream::new(DEVICE).unwrap();
    let device = DeviceAllocation::linear(DEVICE, 256).unwrap();
    assert_eq!(stream.capture_status().unwrap(), CaptureStatus::None);

    // SAFETY: the stream is live and owned by this test.
    assert_eq!(unsafe { begin(stream.raw() as *mut _, 2) }, 0);
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

    let mut graph = std::ptr::null_mut();
    // SAFETY: ends the capture begun above and destroys the resulting graph.
    unsafe {
        assert_eq!(end(stream.raw() as *mut _, &mut graph), 0);
        assert_eq!(destroy(graph), 0);
    }
    assert_eq!(stream.capture_status().unwrap(), CaptureStatus::None);
}

/// A graph executable instantiated from a captured graph runs its captured
/// work on a stream that did not capture it, after the graph is destroyed.
#[test]
#[ignore = "requires a CUDA device"]
fn graph_executables_launch_on_another_stream() {
    let _driver = exclusive_driver();
    let (begin, end, destroy_graph) = capture_functions();

    let capture = Stream::new(DEVICE).unwrap();
    let device = DeviceAllocation::linear(DEVICE, 256).unwrap();
    let host = PinnedBuffer::new(DEVICE, 256).unwrap();

    // The captured work fills the allocation with 7; relaxed mode (2) keeps
    // the capture from interacting with other threads.
    let mut graph = std::ptr::null_mut();
    // SAFETY: the stream is live and owned by this test; the fill lies inside
    // the allocation; the graph ended by this capture is instantiated once
    // and destroyed after.
    let launched = unsafe {
        assert_eq!(begin(capture.raw() as RawHandle, 2), 0);
        memset_d8(device.address(), 7, 256, &capture).unwrap();
        assert_eq!(end(capture.raw() as RawHandle, &mut graph), 0);
        let launched = GraphExec::instantiate(DEVICE, graph as usize).unwrap();
        let _context = DeviceGuard::new(DEVICE).unwrap();
        assert_eq!(destroy_graph(graph), 0);
        launched
    };

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

type RawHandle = *mut std::ffi::c_void;
type BeginCapture = unsafe extern "C" fn(RawHandle, i32) -> i32;
type EndCapture = unsafe extern "C" fn(RawHandle, *mut RawHandle) -> i32;
type DestroyGraph = unsafe extern "C" fn(RawHandle) -> i32;

/// The driver's stream capture entry points, which the crate leaves to the
/// libraries that capture.
fn capture_functions() -> (BeginCapture, EndCapture, DestroyGraph) {
    // SAFETY: loading libcuda runs only the driver's own initializers; the
    // library stays loaded for the process, as the driver is.
    let library = Box::leak(Box::new(
        unsafe { libloading::Library::new("libcuda.so.1") }.unwrap(),
    ));
    // SAFETY: the signatures match cuda.h.
    unsafe {
        (
            *library
                .get::<BeginCapture>(b"cuStreamBeginCapture_v2")
                .unwrap(),
            *library.get::<EndCapture>(b"cuStreamEndCapture").unwrap(),
            *library.get::<DestroyGraph>(b"cuGraphDestroy").unwrap(),
        )
    }
}

/// A pool's memory keeps its addresses and contents across a suspension,
/// and a graph captured over it before runs correctly after the restore; a
/// freed chunk is given out again at its address.
#[test]
#[ignore = "requires a CUDA device"]
fn a_pool_keeps_addresses_and_contents_across_a_suspension() {
    let _driver = exclusive_driver();
    let (begin, end, destroy_graph) = capture_functions();
    let mut pool = VirtualPool::reserve(DEVICE, 64 << 20).unwrap();
    let first = pool.allocate(1 << 20).unwrap();
    let second = pool.allocate(1 << 20).unwrap();
    assert_ne!(first, second);
    assert!(pool.contains(first) && pool.contains(second));
    let stream = Stream::new(DEVICE).unwrap();
    let host = PinnedBuffer::new(DEVICE, 256).unwrap();

    // The first chunk holds 5s written before the suspension; a graph
    // captured now fills the second with 9s at every launch.
    let mut graph = std::ptr::null_mut();
    // SAFETY: the ranges lie inside the pool's chunks; the stream is owned
    // by this test and the graph is instantiated once and destroyed after.
    let fill = unsafe {
        memset_d8(first, 5, 256, &stream).unwrap();
        stream.synchronize().unwrap();
        assert_eq!(begin(stream.raw() as RawHandle, 2), 0);
        memset_d8(second, 9, 256, &stream).unwrap();
        assert_eq!(end(stream.raw() as RawHandle, &mut graph), 0);
        let fill = GraphExec::instantiate(DEVICE, graph as usize).unwrap();
        let _context = DeviceGuard::new(DEVICE).unwrap();
        assert_eq!(destroy_graph(graph), 0);
        fill
    };

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
fn nvml_attributes_device_memory_to_this_process() {
    let _driver = exclusive_driver();
    let _allocation = DeviceAllocation::linear(DEVICE, 1 << 20).unwrap();
    let nvml = Nvml::init().unwrap();

    let memory = nvml.memory_info(DEVICE).unwrap();
    assert!(memory.used <= memory.total);
    match nvml.process_memory(DEVICE) {
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
/// the segment is released: NVML attributes the memory to this process
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
    let before = match nvml.process_memory(DEVICE) {
        Ok(before) => before,
        // NVML lists host PIDs; a container without the host PID namespace
        // cannot find itself.
        Err(Error::ProcessNotFound { .. }) => return,
        Err(error) => panic!("{error}"),
    };
    let created_before = cuda::created_bytes(DEVICE);
    let _descriptor = segmented.create(0).unwrap();
    assert_eq!(cuda::created_bytes(DEVICE), created_before + bytes as u64);
    let held = nvml.process_memory(DEVICE).unwrap();
    assert!(
        held >= before + bytes as u64 / 2,
        "NVML attributes the held creation to no process: {before} -> {held} bytes"
    );
    // What the process holds of its own leaves the created segment out.
    assert!(held - cuda::created_bytes(DEVICE) < before + bytes as u64 / 2);

    segmented.release(0).unwrap();
    assert_eq!(cuda::created_bytes(DEVICE), created_before);
    let released = nvml.process_memory(DEVICE).unwrap();
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
    unsafe {
        memset_d32(data.address(), 41, 4, &stream).unwrap();
        communicator
            .all_reduce(
                data.address(),
                data.address(),
                4,
                DataType::Int32,
                ReduceOp::Min,
                &stream,
            )
            .unwrap();
        communicator
            .broadcast(
                data.address(),
                data.address(),
                4,
                DataType::Int32,
                0,
                &stream,
            )
            .unwrap();
        memcpy_async(host.address(), data.address(), 16, &stream).unwrap();
    }
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

    // Another mapping maps the same memory from the segments' descriptors.
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
    let (begin, end, destroy_graph) = capture_functions();
    let granularity = HandleType::Descriptor.granularity(DEVICE).unwrap();
    let bytes = 2 * granularity;
    let mut active = SegmentedMapping::reserve(DEVICE, &[bytes], &[]).unwrap();
    let original = active.create(0).unwrap();
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
    let mut graph = std::ptr::null_mut();
    // SAFETY: captured copies borrow allocations that outlive the executable.
    let read_back = unsafe {
        assert_eq!(begin(stream.raw() as RawHandle, 2), 0);
        memcpy_async(host.address(), active.address(), bytes, &stream).unwrap();
        assert_eq!(end(stream.raw() as RawHandle, &mut graph), 0);
        let executable = GraphExec::instantiate(DEVICE, graph as usize).unwrap();
        let _context = DeviceGuard::new(DEVICE).unwrap();
        assert_eq!(destroy_graph(graph), 0);
        executable
    };
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
}
