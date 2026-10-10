//! Behavior against a real driver. These tests need a CUDA device and use a
//! few megabytes of memory on device 0; run them with
//! `cargo test -p uniserve-cuda -- --ignored`.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::{Mutex, MutexGuard, PoisonError, mpsc};
use std::thread;
use std::time::Duration;

use uniserve_cuda::nccl::{Communicator, CtaPolicy, DataType, ReduceOp, UniqueId};
use uniserve_cuda::{
    CaptureStatus, CopyBatch, DeviceAllocation, DeviceAttribute, DeviceId, Error, Event,
    EventFlags, HandleType, Nvml, PeerMapping, PinnedBuffer, Pitched, PitchedBatch, SourceOrder,
    Stream, Strided, StridedCopy, memcpy_async, memset_d8, memset_d32, reclaim,
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

    // SAFETY: no copy is pending on the buffer.
    let pattern: Vec<u8> = (0..4096).map(|index| (index % 251) as u8).collect();
    unsafe { host.as_mut_slice() }.copy_from_slice(&pattern);
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

    // SAFETY: the signatures match cuda.h; relaxed mode (2) keeps the capture
    // from interacting with other test threads.
    let library = unsafe { libloading::Library::new("libcuda.so.1") }.unwrap();
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
    // SAFETY: rejected before any copy is enqueued.
    assert!(matches!(
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

#[test]
#[ignore = "requires a CUDA device"]
fn sm_partitions_create_streams_of_the_requested_size() {
    let _driver = exclusive_driver();
    let total = DEVICE
        .attribute(DeviceAttribute::MultiprocessorCount)
        .unwrap() as u32;
    let streams = Stream::partition(DEVICE, &[8, 16]).unwrap();

    assert_eq!(streams.len(), 2);
    assert_eq!(streams[0].sm_count().unwrap(), 8);
    assert_eq!(streams[1].sm_count().unwrap(), 16);
    assert!(streams[0].is_partitioned().unwrap());
    let fork = streams[1].fork().unwrap();
    assert!(fork.is_partitioned().unwrap());
    assert_eq!(fork.sm_count().unwrap(), 16);

    let full = Stream::new(DEVICE).unwrap();
    assert!(!full.is_partitioned().unwrap());
    assert_eq!(full.sm_count().unwrap(), total);
    assert!(matches!(
        Stream::partition(DEVICE, &[total + 1]),
        Err(Error::InvalidArgument(_))
    ));
}

/// Batched copies on an SM-partition stream deliver their bytes whether an
/// entry addresses mapped (exportable) or linear memory, in both source
/// orders, and a source read during the call may change once `submit`
/// returns.
#[test]
#[ignore = "requires a CUDA device"]
fn batched_copies_reach_mapped_memory_on_partition_streams() {
    let _driver = exclusive_driver();
    let stream = Stream::partition(DEVICE, &[8, 16]).unwrap().remove(0);
    let pattern: Vec<u8> = (0..4096).map(|index| (index % 251) as u8).collect();

    for (order, page_locked) in [
        (SourceOrder::Stream, true),
        (SourceOrder::DuringCall, false),
        (SourceOrder::DuringCall, true),
    ] {
        let mapped = DeviceAllocation::exportable(DEVICE, 4096, &[]).unwrap();
        let linear = DeviceAllocation::linear(DEVICE, 4096).unwrap();
        let readback = PinnedBuffer::new(DEVICE, 4096).unwrap();
        let mut pinned = PinnedBuffer::new(DEVICE, 4096).unwrap();
        // SAFETY: no copy is pending on the buffer.
        unsafe { pinned.as_mut_slice() }.copy_from_slice(&pattern);
        let mut pageable = pattern.clone();
        // A source read in stream order is page-locked; one read during the
        // call may also be ordinary host memory.
        let source = if page_locked {
            pinned.address()
        } else {
            pageable.as_ptr() as u64
        };

        // The first half goes to mapped memory and the second to linear.
        let mut upload = CopyBatch::new(order);
        upload.push(mapped.address(), source, 2048);
        upload.push(linear.address(), source + 2048, 2048);
        // SAFETY: every range outlives the synchronization below; a source
        // read during the call is not needed after `submit` returns.
        unsafe { upload.submit(&stream).unwrap() };
        if order == SourceOrder::DuringCall {
            pageable.fill(0xff);
            // SAFETY: the batch has read its sources.
            unsafe { pinned.as_mut_slice() }.fill(0xff);
        }

        // The mapped half moves on through device memory.
        let transit = DeviceAllocation::linear(DEVICE, 2048).unwrap();
        let mut forward = CopyBatch::new(SourceOrder::Stream);
        forward.push(transit.address(), mapped.address(), 2048);
        let mut download = CopyBatch::new(SourceOrder::Stream);
        download.push(readback.address(), transit.address(), 2048);
        download.push(readback.address() + 2048, linear.address(), 2048);
        // SAFETY: as above.
        unsafe {
            forward.submit(&stream).unwrap();
            download.submit(&stream).unwrap();
        }
        stream.synchronize().unwrap();

        assert_eq!(
            read(&readback),
            pattern,
            "{order:?} with a page-locked source: {page_locked}"
        );
    }
}

/// Pitched batches on an SM-partition stream deliver their rows into and
/// out of mapped (exportable) memory.
#[test]
#[ignore = "requires a CUDA device"]
fn pitched_batches_reach_mapped_memory_on_partition_streams() {
    let _driver = exclusive_driver();
    let stream = Stream::partition(DEVICE, &[8, 16]).unwrap().remove(0);
    let mapped = DeviceAllocation::exportable(DEVICE, 4096, &[]).unwrap();
    let linear = DeviceAllocation::linear(DEVICE, 4096).unwrap();
    let readback = PinnedBuffer::new(DEVICE, 1024).unwrap();
    let mut pinned = PinnedBuffer::new(DEVICE, 1024).unwrap();
    let pattern: Vec<u8> = (0..1024).map(|index| (index % 251) as u8).collect();
    // SAFETY: no copy is pending on the buffer.
    unsafe { pinned.as_mut_slice() }.copy_from_slice(&pattern);

    // 16 rows of 64 bytes: packed on the host, 256 bytes apart on the device.
    let rows = |destination, destination_pitch, source, source_pitch| Pitched {
        destination,
        destination_pitch,
        destination_layer: 16,
        source,
        source_pitch,
        source_layer: 16,
        width: 64,
        height: 16,
        depth: 1,
    };
    for (destination, source) in [
        ((mapped.address(), 256), (pinned.address(), 64)),
        ((linear.address(), 256), (mapped.address(), 256)),
        ((readback.address(), 64), (linear.address(), 256)),
    ] {
        let mut batch = PitchedBatch::new(SourceOrder::Stream);
        batch
            .push(rows(destination.0, destination.1, source.0, source.1))
            .unwrap();
        // SAFETY: every row outlives the synchronization below.
        unsafe { batch.submit(&stream).unwrap() };
    }
    stream.synchronize().unwrap();

    assert_eq!(read(&readback), pattern);
}

#[test]
#[ignore = "requires a CUDA device"]
fn forks_and_fences_outlive_their_partition_stream() {
    let _driver = exclusive_driver();
    let parent = Stream::partition(DEVICE, &[16]).unwrap().remove(0);
    let fork = parent.fork().unwrap();
    let device = DeviceAllocation::linear(DEVICE, 1024).unwrap();
    let host = PinnedBuffer::new(DEVICE, 1024).unwrap();
    let written = Event::new(EventFlags::default());

    // SAFETY: the ranges outlive the synchronizations below.
    unsafe { memset_d8(device.address(), 7, 1024, &parent).unwrap() };
    written.record(&parent).unwrap();
    drop(parent);

    // The fence recorded in the partition still orders a consumer of the
    // primary context after the recording stream is gone.
    let consumer = Stream::new(DEVICE).unwrap();
    consumer.wait_event(&written).unwrap();
    // SAFETY: as above.
    unsafe { memcpy_async(host.address(), device.address(), 1024, &consumer).unwrap() };
    consumer.synchronize().unwrap();
    assert!(read(&host).iter().all(|byte| *byte == 7));

    // The fork keeps the partition and still executes work in it.
    assert!(fork.is_partitioned().unwrap());
    assert_eq!(fork.sm_count().unwrap(), 16);
    // SAFETY: as above.
    unsafe {
        memset_d8(device.address(), 9, 1024, &fork).unwrap();
        memcpy_async(host.address(), device.address(), 1024, &fork).unwrap();
    }
    fork.synchronize().unwrap();
    assert!(read(&host).iter().all(|byte| *byte == 9));
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

#[test]
#[ignore = "requires a CUDA device and NCCL"]
fn single_rank_communicators_run_collectives() {
    let _driver = exclusive_driver();
    let stream = Stream::new(DEVICE).unwrap();
    let data = DeviceAllocation::linear(DEVICE, 16).unwrap();
    let host = PinnedBuffer::new(DEVICE, 16).unwrap();
    let id = UniqueId::new().unwrap();
    let mut communicator = Communicator::init(DEVICE, 1, 0, &id, CtaPolicy::Default).unwrap();
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
