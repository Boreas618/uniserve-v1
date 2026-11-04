//! The device-storage allocator of design 4.1 against a real driver: pools as
//! reserved ranges, blocks created and mapped on demand, freed blocks given
//! out again, mappings booked against the device's budget before they are
//! made, and blocks exported as POSIX descriptors.
//!
//! These tests need a CUDA device; they take the driver lock of `gpu.rs`, so
//! they neither block nor are blocked by the tests there.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use platform::{Allocator, Budget, DeviceId, Error, PoolKind};

use crate::gpu::{DEVICE, exclusive_driver, read};

/// The budget the executor implements over the storage client's booking
/// request (design 8): it records every total booked and refuses totals above
/// a limit the test can change.
struct RecordingBudget {
    limit: AtomicU64,
    totals: Mutex<Vec<(DeviceId, PoolKind, u64)>>,
}

impl RecordingBudget {
    fn new(limit: u64) -> Arc<Self> {
        Arc::new(Self {
            limit: AtomicU64::new(limit),
            totals: Mutex::new(Vec::new()),
        })
    }

    fn set_limit(&self, limit: u64) {
        self.limit.store(limit, Ordering::Relaxed);
    }

    /// The last total booked for `pool`, if the budget booked one.
    fn booked(&self, pool: PoolKind) -> Option<u64> {
        self.totals
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .rev()
            .find(|(_, booked_pool, _)| *booked_pool == pool)
            .map(|(_, _, bytes)| *bytes)
    }
}

impl Budget for RecordingBudget {
    fn book(&self, device: DeviceId, pool: PoolKind, bytes: u64) -> platform::Result<()> {
        if bytes > self.limit.load(Ordering::Relaxed) {
            return Err(Error::InvalidArgument(format!(
                "the budget of {device} refuses {bytes} bytes for its {pool:?} pool"
            )));
        }
        self.totals
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((device, pool, bytes));
        Ok(())
    }
}

/// An allocator that books against `budget`, which stands for the executor's
/// budget over its storage client.
fn with_budget(device: &platform::Device, budget: &Arc<RecordingBudget>) -> Allocator {
    let recording: Arc<RecordingBudget> = Arc::clone(budget);
    let ledger: Arc<dyn Budget> = recording;
    Allocator::new(device, ledger).unwrap()
}

#[test]
#[ignore = "requires a CUDA device"]
fn an_allocation_over_the_device_budget_fails_before_it_maps_anything() {
    let _driver = exclusive_driver();
    let device = platform::Device::new(DEVICE).unwrap();
    let budget = RecordingBudget::new(0);
    let mut allocator = with_budget(&device, &budget);
    assert_eq!(allocator.device(), DEVICE);
    allocator.reserve(PoolKind::Scratch, 8 << 20).unwrap();

    let refused = allocator.allocate(PoolKind::Scratch, 1 << 20).unwrap_err();
    assert!(
        matches!(refused, Error::InvalidArgument(_)),
        "the budget's refusal reaches the caller unchanged, got {refused:?}"
    );
    assert_eq!(allocator.mapped_bytes(PoolKind::Scratch), 0);
    assert_eq!(budget.booked(PoolKind::Scratch), None);

    // With room, the same allocation maps storage, and the pool's mapped bytes
    // are exactly the total the budget was asked to book.
    budget.set_limit(8 << 20);
    let buffer = allocator.allocate(PoolKind::Scratch, 1 << 20).unwrap();
    assert!(buffer.len() >= 1 << 20);
    assert_eq!(buffer.device(), DEVICE);
    assert_eq!(buffer.pool(), PoolKind::Scratch);
    assert_eq!(allocator.mapped_bytes(PoolKind::Scratch), buffer.len());
    assert_eq!(budget.booked(PoolKind::Scratch), Some(buffer.len() as u64));
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_freed_block_is_given_out_again_at_its_address_with_its_contents() {
    let _driver = exclusive_driver();
    let device = platform::Device::new(DEVICE).unwrap();
    let budget = RecordingBudget::new(16 << 20);
    let mut allocator = with_budget(&device, &budget);
    allocator.reserve(PoolKind::Graph, 8 << 20).unwrap();
    let stream = device.stream().unwrap();

    let first = allocator.allocate(PoolKind::Graph, 4096).unwrap();
    let address = first.address();
    let mapped = allocator.mapped_bytes(PoolKind::Graph);
    // SAFETY: the block holds the bytes and outlives the fill, which the
    // stream runs in order before the block is freed below.
    unsafe { platform::memset_d8(address, 0x5a, 16, &stream) }.unwrap();
    stream.synchronize().unwrap();

    allocator.free(first);
    let host = platform::PinnedBuffer::new(DEVICE, 16).unwrap();
    let second = allocator.allocate(PoolKind::Graph, 4096).unwrap();
    assert_eq!(
        second.address(),
        address,
        "a freed block comes back at its address, so a captured graph stays valid"
    );
    assert_eq!(
        allocator.mapped_bytes(PoolKind::Graph),
        mapped,
        "reusing a freed block maps no new storage"
    );
    // SAFETY: the block is mapped and idle; the host buffer is as long as the
    // copy.
    unsafe { platform::memcpy_async(host.address(), second.address(), 16, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(read(&host), vec![0x5a; 16]);
}

#[test]
#[ignore = "requires a CUDA device"]
fn each_pool_has_its_own_range_and_a_block_exports_a_working_descriptor() {
    let _driver = exclusive_driver();
    let device = platform::Device::new(DEVICE).unwrap();
    let budget = RecordingBudget::new(64 << 20);
    let mut allocator = with_budget(&device, &budget);
    for pool in [PoolKind::Graph, PoolKind::Scratch, PoolKind::Resident] {
        allocator.reserve(pool, 8 << 20).unwrap();
    }

    let graph = allocator.allocate(PoolKind::Graph, 4096).unwrap();
    let scratch = allocator.allocate(PoolKind::Scratch, 4096).unwrap();
    let resident = allocator.allocate(PoolKind::Resident, 4096).unwrap();
    assert_ne!(graph.address(), scratch.address());
    assert_ne!(scratch.address(), resident.address());
    for (buffer, pool) in [
        (&graph, PoolKind::Graph),
        (&scratch, PoolKind::Scratch),
        (&resident, PoolKind::Resident),
    ] {
        assert_eq!(buffer.pool(), pool);
        assert_eq!(allocator.mapped_bytes(pool), buffer.len());
    }

    // The descriptor a buffer exports is the block's own storage: importing it
    // into another mapping of this process reads the bytes written through the
    // buffer.
    let descriptor = resident.export().unwrap();
    let mut imported =
        platform::SegmentedMapping::reserve(DEVICE, &[resident.len()], &[DEVICE]).unwrap();
    imported.import(0, &descriptor).unwrap();
    assert!(imported.is_mapped(0));

    let stream = device.stream().unwrap();
    let host = platform::PinnedBuffer::new(DEVICE, 16).unwrap();
    // SAFETY: both mappings hold the bytes and outlive the copies, which the
    // stream runs in order.
    unsafe { platform::memset_d8(resident.address(), 0x3c, 16, &stream) }.unwrap();
    // SAFETY: as above; the imported mapping addresses 16 mapped bytes.
    unsafe { platform::memcpy_async(host.address(), imported.address(), 16, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(read(&host), vec![0x3c; 16]);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_suspended_allocator_gives_its_budget_back_and_restores_its_blocks() {
    let _driver = exclusive_driver();
    let device = platform::Device::new(DEVICE).unwrap();
    let budget = RecordingBudget::new(64 << 20);
    let mut allocator = with_budget(&device, &budget);
    for pool in [PoolKind::Graph, PoolKind::Scratch] {
        allocator.reserve(pool, 8 << 20).unwrap();
    }
    let stream = device.stream().unwrap();

    let first = allocator.allocate(PoolKind::Scratch, 4096).unwrap();
    let second = allocator.allocate(PoolKind::Scratch, 4096).unwrap();
    let graph = allocator.allocate(PoolKind::Graph, 4096).unwrap();
    let (first_address, second_address, graph_address) =
        (first.address(), second.address(), graph.address());
    let scratch_bytes = first.len() + second.len();
    // SAFETY: both blocks hold the bytes and outlive the fills, which the
    // stream runs in order before the suspension.
    unsafe {
        platform::memset_d8(first_address, 0x11, 16, &stream).unwrap();
        platform::memset_d8(second_address, 0x22, 16, &stream).unwrap();
    }
    stream.synchronize().unwrap();

    // A parked standby gives the device's budget back: it counts what the
    // process maps, and the process maps nothing while suspended.
    allocator.suspend().unwrap();
    assert_eq!(allocator.mapped_bytes(PoolKind::Scratch), 0);
    assert_eq!(allocator.mapped_bytes(PoolKind::Graph), 0);
    assert_eq!(budget.booked(PoolKind::Scratch), Some(0));
    assert_eq!(budget.booked(PoolKind::Graph), Some(0));

    // Restoring maps the storage again at the same addresses, books exactly
    // what it maps, and brings every block's contents back.
    allocator.restore().unwrap();
    assert_eq!(first.address(), first_address, "addresses are restored");
    assert_eq!(second.address(), second_address, "addresses are restored");
    assert_eq!(graph.address(), graph_address, "addresses are restored");
    assert_eq!(allocator.mapped_bytes(PoolKind::Scratch), scratch_bytes);
    assert_eq!(allocator.mapped_bytes(PoolKind::Graph), graph.len());
    assert_eq!(
        budget.booked(PoolKind::Scratch),
        Some(scratch_bytes as u64),
        "the last total booked is exactly what the pool maps again"
    );
    assert_eq!(budget.booked(PoolKind::Graph), Some(graph.len() as u64));

    let host = platform::PinnedBuffer::new(DEVICE, 16).unwrap();
    // SAFETY: the block is mapped again and holds its saved contents; the host
    // buffer is as long as the copy.
    unsafe { platform::memcpy_async(host.address(), first_address, 16, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(read(&host), vec![0x11; 16]);
}

#[test]
#[ignore = "requires a CUDA device"]
fn a_parked_block_changes_access_in_place_and_keeps_its_address() {
    let _driver = exclusive_driver();
    let device = platform::Device::new(DEVICE).unwrap();
    let budget = RecordingBudget::new(32 << 20);
    let mut allocator = with_budget(&device, &budget);
    allocator.reserve(PoolKind::Resident, 8 << 20).unwrap();
    let stream = device.stream().unwrap();

    let block = allocator.allocate(PoolKind::Resident, 4096).unwrap();
    let address = block.address();
    let pattern: Vec<u8> = (0..16).map(|index| index as u8 + 1).collect();
    let host = platform::PinnedBuffer::new(DEVICE, 16).unwrap();
    // SAFETY: the host buffer is written before the copy and read after the
    // synchronization.
    unsafe { host.as_mut_slice() }.copy_from_slice(&pattern);
    // SAFETY: both ranges hold the bytes and outlive the copy.
    unsafe { platform::memcpy_async(address, host.address(), 16, &stream) }.unwrap();
    stream.synchronize().unwrap();

    // Park: the block turns read-only at the same address and keeps what it
    // holds, which is what keeps a captured graph that reads it valid.
    allocator
        .set_protection(&block, platform::Protection::ReadOnly)
        .unwrap();
    assert_eq!(block.address(), address);
    // SAFETY: the block is mapped; the host buffer is as long as the copy.
    unsafe { platform::memcpy_async(host.address(), address, 16, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(read(&host), pattern);

    // Promote: writable again at the same address, and the write lands.
    allocator
        .set_protection(&block, platform::Protection::ReadWrite)
        .unwrap();
    assert_eq!(block.address(), address);
    // SAFETY: the block is read-write again and holds the bytes.
    unsafe { platform::memset_d8(address, 0x77, 16, &stream) }.unwrap();
    stream.synchronize().unwrap();
    // SAFETY: as above.
    unsafe { platform::memcpy_async(host.address(), address, 16, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(read(&host), vec![0x77; 16]);
}
