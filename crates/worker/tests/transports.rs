//! Native transports on host memory: exports, reads and retirement.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use uniserve_ipc::{TensorTransfer, WorkerEndpoint};
use uniserve_worker::ffi::{Backing, DType, Device};
use uniserve_worker::storage::memory::{Memory, StorageView};
use uniserve_worker::transfer::{
    Capacity, Notify, Spans, TransportKind, TransportOptions, Transports, exports, fetch, notify,
};
use uniserve_worker::{EventPool, Result};

/// Export on `stream` and take the locations, complete at once.
fn located(
    transports: &Transports,
    source: &Spans,
    offset: Option<&[u64]>,
    consumers: &[usize],
    host: bool,
    stream: Option<&uniserve_cuda::Stream>,
    mut retain: impl FnMut(Arc<uniserve_worker::transfer::Retirement>) -> Result<()>,
) -> Result<Vec<uniserve_worker::transfer::exports::Location>> {
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

fn endpoint() -> WorkerEndpoint {
    WorkerEndpoint {
        worker_id: "worker".into(),
        rank: 0,
        node: "node".into(),
        address_space: "space".into(),
        incarnation: "incarnation".into(),
    }
}

fn options() -> TransportOptions {
    TransportOptions {
        source: endpoint(),
        acknowledgment_slot: 0,
        host_slots: HashSet::from([0]),
        cross_host_consumers: false,
        vmm_pool_bytes: 1 << 20,
    }
}

fn transports(kinds: &[TransportKind], bytes: u64) -> Result<(Transports, Arc<Capacity>)> {
    let capacity = Arc::new(Capacity::new(bytes, 8, notify)?);
    let transports = Transports::new(kinds, Arc::clone(&capacity), EventPool::new(), &options())?;
    Ok((transports, capacity))
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

#[test]
fn local_reads_copy_or_borrow_and_retire_after_release() -> Result<()> {
    let (transports, capacity) = transports(&[TransportKind::Local], 1 << 12)?;
    let source = counting(&[4, 3])?;
    let mut retirements = Vec::new();
    let locations = located(&transports, &source, None, &[], false, None, |retirement| {
        retirements.push(retirement);
        Ok(())
    })?;
    assert_eq!(locations.len(), 1);
    let (transport, locator) = &locations[0];
    assert_eq!(capacity.used(), 48);

    // A copy into a Rust-owned destination of one region.
    let destination = zeros(&[2, 2])?;
    let read = transport.fetch(
        locator,
        Device::Cpu,
        Some(&destination),
        Some(&[1..3, 1..3]),
        None,
        None,
    )?;
    read.wait_ready();
    assert_eq!(values(&read.result(None)?), [4.0, 5.0, 7.0, 8.0]);

    // A borrowed read sees the producer's storage in place.
    let borrowed = transport.fetch(locator, Device::Cpu, None, None, None, None)?;
    let view = borrowed.result(None)?;
    assert_eq!(view.first().address(), source.first().address());

    transport.release(locator)?;
    assert!(
        !retirements[0].done(),
        "an open borrowed read holds the export"
    );
    borrowed.close()?;
    transports.reap()?;
    // The copy read's credit and bytes return when its task retires.
    while !read.retired() {
        std::thread::yield_now();
    }
    assert!(retirements[0].succeeded());
    assert_eq!(capacity.used(), 0);
    transports.close()
}

#[test]
fn shared_memory_and_channel_reads_stage_whole_payloads() -> Result<()> {
    let (transports, capacity) =
        transports(&[TransportKind::Shm, TransportKind::Channel], 1 << 12)?;
    let source = counting(&[3, 2])?;
    let locations = located(
        &transports,
        &source,
        Some(&[2, 0]),
        &[0],
        false,
        None,
        |_| Ok(()),
    )?;
    let kinds: Vec<_> = locations
        .iter()
        .map(|(_, locator)| TransportKind::of(locator))
        .collect();
    // Slot 0 is on this host: shared memory reaches it, the channel does not.
    assert_eq!(kinds, [TransportKind::Shm]);
    let (transport, locator) = &locations[0];
    assert_eq!(locator.offset, [2, 0]);

    let destination = zeros(&[3, 2])?;
    let read = transport.fetch(locator, Device::Cpu, Some(&destination), None, None, None)?;
    read.wait_ready();
    assert_eq!(values(&read.result(None)?), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);

    // Borrowing whole rows reads in place; the region is logical.
    let rows = transport.borrow(locator, Some(&[3..5, 0..2]))?;
    assert_eq!(rows.nbytes(), 16);
    drop(rows);

    // Slot 1 is on another host: only the rank channel reaches it, and its
    // bytes travel inside the locator.
    let remote = located(&transports, &source, None, &[1], false, None, |_| Ok(()))?;
    let (channel, exported) = &remote[0];
    assert_eq!(TransportKind::of(exported), TransportKind::Channel);
    let destination = zeros(&[2, 2])?;
    let read = channel.fetch(
        exported,
        Device::Cpu,
        Some(&destination),
        Some(&[1..3, 0..2]),
        None,
        None,
    )?;
    read.wait_ready();
    assert_eq!(values(&read.result(None)?), [2.0, 3.0, 4.0, 5.0]);

    transport.release(locator)?;
    while capacity.used() != 0 {
        transports.reap()?;
        std::thread::yield_now();
    }
    transports.close()
}

#[test]
fn fetches_cover_regions_from_shards_and_reject_missing_coverage() -> Result<()> {
    let (transports, _) = transports(&[TransportKind::Local], 1 << 12)?;
    // Two shards of a [4, 2] tensor: rows 0..2 and rows 2..4.
    let first = counting(&[2, 2])?;
    let second = counting(&[2, 2])?;
    let mut locations = located(&transports, &first, Some(&[0, 0]), &[], false, None, |_| {
        Ok(())
    })?;
    let shard = located(
        &transports,
        &second,
        Some(&[2, 0]),
        &[],
        false,
        None,
        |_| Ok(()),
    )?;
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
        None,
        |_| Ok(()),
    )?;
    assert_eq!(reads.len(), 2);
    for read in &reads {
        read.wait_ready();
        read.result(None)?;
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
            None,
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
    let (transports, capacity) = transports(&[TransportKind::Local, TransportKind::Shm], 24)?;
    let source = counting(&[2, 3])?;
    let accepted = AtomicUsize::new(0);
    let result = located(&transports, &source, None, &[0], false, None, |_| {
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
    let (transports, _) = transports(&[TransportKind::Local], 1 << 12)?;
    let source = counting(&[2])?;
    let locations = located(&transports, &source, None, &[], false, None, |_| Ok(()))?;
    let (transport, locator) = &locations[0];
    let destination = zeros(&[2])?;
    let read = transport.fetch(locator, Device::Cpu, Some(&destination), None, None, None)?;
    let calls = Arc::new(AtomicUsize::new(0));
    let observer: Notify = {
        let calls = Arc::clone(&calls);
        Arc::new(move || {
            calls.fetch_add(1, Ordering::Relaxed);
        })
    };
    read.add_done_callback(Arc::clone(&observer));
    read.wait_ready();
    read.add_done_callback(observer);
    assert_eq!(calls.load(Ordering::Relaxed), 2);
    exports::release(&locations)?;
    transports.close()
}
