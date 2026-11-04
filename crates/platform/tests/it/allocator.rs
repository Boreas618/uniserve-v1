//! The device-storage allocator of design 4.1 against a real driver: pools as
//! reserved ranges, blocks created and mapped on demand, freed blocks given
//! out again, mappings booked against the device's budget before they are
//! made, and blocks exported as POSIX descriptors.
//!
//! These tests need a CUDA device; they take the driver lock of `gpu.rs`, so
//! they neither block nor are blocked by the tests there.

use std::os::fd::AsRawFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use platform::{
    Allocator, Budget, Capture, CaptureMode, DeviceId, Error, PinnedBuffer, PoolKind, memcpy_async,
    memset_d8,
};

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

    /// How many bookings the budget was asked for, whatever their pool.
    fn bookings(&self) -> usize {
        self.totals
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
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
    let mut host = platform::PinnedBuffer::new(DEVICE, 16).unwrap();
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

#[test]
#[ignore = "requires a CUDA device"]
fn a_restore_a_shrunk_budget_refuses_parks_every_pool_again() {
    let _driver = exclusive_driver();
    let device = platform::Device::new(DEVICE).unwrap();
    let budget = RecordingBudget::new(64 << 20);
    let mut allocator = with_budget(&device, &budget);
    for pool in [PoolKind::Graph, PoolKind::Scratch, PoolKind::Resident] {
        allocator.reserve(pool, 64 << 20).unwrap();
    }
    let stream = device.stream().unwrap();
    let graph = allocator.allocate(PoolKind::Graph, 1 << 20).unwrap();
    let scratch = allocator.allocate(PoolKind::Scratch, 1 << 20).unwrap();
    // The resident block is the large one: the budget books each pool's *own*
    // total, so an allowance that refuses one pool's restore has to sit between
    // that pool's total and the others'. Twice the graph block, after the
    // driver rounds both, is larger than the graph block on every device.
    let resident = allocator
        .allocate(PoolKind::Resident, graph.len() * 2)
        .unwrap();
    let addresses = (graph.address(), scratch.address(), resident.address());
    // SAFETY: the block holds these bytes and outlives the fill, which the
    // stream runs before the suspension.
    unsafe { platform::memset_d8(graph.address(), 0x33, 16, &stream) }.unwrap();
    stream.synchronize().unwrap();

    allocator.suspend().unwrap();
    assert_eq!(allocator.mapped_bytes(PoolKind::Resident), 0);

    // The allowance shrinks to what the graph and scratch pools book, which is
    // below what the resident pool books: a budget books one pool's total, not
    // the sum over the three, so the refusal can only land on a pool whose own
    // total is over the allowance. Both figures come from the blocks the driver
    // actually gave out, because a block is rounded up to the device's
    // allocation granularity and a hard-coded limit would be device-dependent.
    let limit = graph.len().max(scratch.len()) as u64;
    assert!(
        limit < resident.len() as u64,
        "the allowance {limit} must be below the resident block's {} bytes",
        resident.len()
    );
    budget.set_limit(limit);
    let refused = allocator.restore().unwrap_err();
    let message = refused.to_string();
    assert!(message.contains("resident"), "{message}");
    // A refused restore names the cause too, not only the pool.
    assert!(message.contains("budget"), "{message}");
    for pool in [PoolKind::Graph, PoolKind::Scratch, PoolKind::Resident] {
        assert_eq!(
            allocator.mapped_bytes(pool),
            0,
            "{pool:?} stayed promoted after the failure"
        );
        assert_eq!(
            budget.booked(pool),
            Some(0),
            "{pool:?} stayed booked after the failure"
        );
    }

    // With the allowance back, the same restore brings every pool back at its
    // address with its contents, so nothing was lost to the failure.
    budget.set_limit(64 << 20);
    allocator.restore().unwrap();
    assert_eq!(
        (graph.address(), scratch.address(), resident.address()),
        addresses
    );
    let host = platform::PinnedBuffer::new(DEVICE, 16).unwrap();
    // SAFETY: the block is mapped again with its contents and the host buffer
    // is as long as the copy.
    unsafe { platform::memcpy_async(host.address(), graph.address(), 16, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(read(&host), vec![0x33; 16]);
}

/// A pool's storage keeps its addresses and contents across a suspension, and
/// a graph captured over it before runs correctly after the restore; a freed
/// block is given out again at its address.
///
/// The bookings are part of the behaviour: mapping a block books the pool's
/// total before the storage exists, a suspension books the pool down to
/// nothing and the restore books it back, and giving out a freed block books
/// nothing at all because it maps nothing. (That a refused booking leaves
/// nothing mapped is
/// [`an_allocation_over_the_device_budget_fails_before_it_maps_anything`].)
#[test]
#[ignore = "requires a CUDA device"]
fn a_pool_keeps_addresses_and_contents_across_a_suspension() {
    let _driver = exclusive_driver();
    let device = platform::Device::new(DEVICE).unwrap();
    let budget = RecordingBudget::new(64 << 20);
    let mut allocator = with_budget(&device, &budget);
    allocator.reserve(PoolKind::Graph, 64 << 20).unwrap();

    let first = allocator.allocate(PoolKind::Graph, 1 << 20).unwrap();
    let second = allocator.allocate(PoolKind::Graph, 1 << 20).unwrap();
    assert_ne!(first.address(), second.address());
    let mapped = allocator.mapped_bytes(PoolKind::Graph);
    assert_eq!(mapped, first.len() + second.len());
    assert_eq!(
        budget.booked(PoolKind::Graph),
        Some(mapped as u64),
        "the pool books the storage it maps, and the booking carries the pool's total"
    );
    let stream = device.stream().unwrap();
    let host = PinnedBuffer::new(DEVICE, 256).unwrap();

    // The first block holds 5s written before the suspension; a graph captured
    // now fills the second with 9s at every launch.
    // SAFETY: the ranges lie inside the allocator's blocks.
    unsafe { memset_d8(first.address(), 5, 256, &stream) }.unwrap();
    stream.synchronize().unwrap();
    let capture = Capture::begin(&stream, CaptureMode::Relaxed).unwrap();
    // SAFETY: the range lies inside the block.
    unsafe { memset_d8(second.address(), 9, 256, &stream) }.unwrap();
    let fill = capture.end().unwrap().instantiate().unwrap();

    allocator.suspend().unwrap();
    assert_eq!(
        allocator.mapped_bytes(PoolKind::Graph),
        0,
        "a suspended pool maps nothing"
    );
    assert!(
        allocator.allocate(PoolKind::Graph, 1 << 20).is_err(),
        "a suspended allocator gives out nothing"
    );
    assert_eq!(
        budget.booked(PoolKind::Graph),
        Some(0),
        "a suspended pool maps nothing, so the budget must stop counting it: an over-count \
         would refuse an allocation that fits"
    );
    allocator.restore().unwrap();
    assert_eq!(allocator.mapped_bytes(PoolKind::Graph), mapped);
    assert_eq!(
        budget.booked(PoolKind::Graph),
        Some(mapped as u64),
        "the storage counts again once the restore has mapped it"
    );

    // SAFETY: the ranges lie inside the restored blocks, at their addresses.
    unsafe { memcpy_async(host.address(), first.address(), 256, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(read(&host), vec![5; 256]);
    fill.launch(&stream).unwrap();
    // SAFETY: as above.
    unsafe { memcpy_async(host.address(), second.address(), 256, &stream) }.unwrap();
    stream.synchronize().unwrap();
    assert_eq!(read(&host), vec![9; 256]);

    drop(fill);
    // The baseline for "books nothing new" is taken after the restore, because
    // the suspension and the restore do book: the first books the pool down to
    // nothing, the second books back what it maps.
    let bookings = budget.bookings();
    let address = second.address();
    allocator.free(second);
    let again = allocator.allocate(PoolKind::Graph, 1 << 20).unwrap();
    assert_eq!(
        again.address(),
        address,
        "a freed block comes back at its address, so a captured graph stays valid"
    );
    assert_eq!(allocator.mapped_bytes(PoolKind::Graph), mapped);
    assert_eq!(
        budget.bookings(),
        bookings,
        "giving out a freed block books nothing new"
    );
}

/// The device's ledger behind several allocators of the same pool: it counts
/// what they hold together, adding what each reports and releasing what each
/// gives up.
struct SharedLedger {
    total: AtomicU64,
    limit: u64,
}

impl SharedLedger {
    fn new(limit: u64) -> Arc<Self> {
        Arc::new(Self {
            total: AtomicU64::new(0),
            limit,
        })
    }

    fn total(&self) -> u64 {
        self.total.load(Ordering::Relaxed)
    }

    fn add(&self, bytes: u64) -> platform::Result<()> {
        let held = self.total() + bytes;
        if held > self.limit {
            return Err(Error::InvalidArgument(format!(
                "the device's ledger refuses {held} bytes of a {}-byte allowance",
                self.limit
            )));
        }
        self.total.fetch_add(bytes, Ordering::Relaxed);
        Ok(())
    }
}

/// One allocator's place in a [`SharedLedger`]: a booking carries that
/// allocator's own pool total, so the adapter keeps the total it last reported
/// and passes the difference on — the shape a device with several allocators of
/// one pool needs (a rank's graph pools are several).
struct Adapter {
    ledger: Arc<SharedLedger>,
    last: Mutex<u64>,
}

impl Budget for Adapter {
    fn book(&self, _device: DeviceId, _pool: PoolKind, bytes: u64) -> platform::Result<()> {
        let mut last = self.last.lock().unwrap_or_else(PoisonError::into_inner);
        if bytes >= *last {
            self.ledger.add(bytes - *last)?;
        } else {
            self.ledger
                .total
                .fetch_sub(*last - bytes, Ordering::Relaxed);
        }
        // The ledger moved before this adapter's total does, so a refusal leaves
        // both as they were.
        *last = bytes;
        Ok(())
    }
}

fn with_adapter(device: &platform::Device, ledger: &Arc<SharedLedger>) -> Allocator {
    let adapter = Arc::new(Adapter {
        ledger: Arc::clone(ledger),
        last: Mutex::new(0),
    });
    let booking: Arc<dyn Budget> = adapter;
    Allocator::new(device, booking).unwrap()
}

#[test]
#[ignore = "requires a CUDA device"]
fn several_allocators_of_one_pool_book_one_ledger_through_an_adapter() {
    let _driver = exclusive_driver();
    let device = platform::Device::new(DEVICE).unwrap();
    let ledger = SharedLedger::new(32 << 20);
    let mut first = with_adapter(&device, &ledger);
    let mut second = with_adapter(&device, &ledger);
    first.reserve(PoolKind::Graph, 8 << 20).unwrap();
    second.reserve(PoolKind::Graph, 8 << 20).unwrap();

    // A booking carries the allocator's own pool total, not a device-wide one,
    // which is what lets an adapter tell the two apart.
    let one = first.allocate(PoolKind::Graph, 4096).unwrap();
    assert_eq!(first.mapped_bytes(PoolKind::Graph), one.len());
    assert_eq!(ledger.total(), one.len() as u64);
    let two = second.allocate(PoolKind::Graph, 4096).unwrap();
    assert_eq!(second.mapped_bytes(PoolKind::Graph), two.len());
    assert_eq!(
        ledger.total(),
        one.len() as u64 + two.len() as u64,
        "the ledger counts what both allocators hold"
    );

    // A suspension reports the pool down to nothing, and the ledger releases
    // exactly that allocator's share; the restore puts it back.
    first.suspend().unwrap();
    assert_eq!(ledger.total(), two.len() as u64);
    first.restore().unwrap();
    assert_eq!(ledger.total(), one.len() as u64 + two.len() as u64);
}

/// A buffer handed out before a suspension still names its block afterwards: the
/// pool released the block's storage, so exporting it is refused rather than
/// naming a handle the driver has taken back, and once the restore has mapped
/// new storage the same buffer exports that storage.
#[test]
#[ignore = "requires a CUDA device"]
fn an_exported_block_follows_the_storage_a_restore_maps() {
    let _driver = exclusive_driver();
    let device = platform::Device::new(DEVICE).unwrap();
    let budget = RecordingBudget::new(8 << 20);
    let mut allocator = with_budget(&device, &budget);
    allocator.reserve(PoolKind::Graph, 8 << 20).unwrap();
    let block = allocator.allocate(PoolKind::Graph, 4096).unwrap();
    assert!(
        block.export().unwrap().as_raw_fd() >= 0,
        "a mapped block exports a descriptor"
    );

    allocator.suspend().unwrap();
    let refused = block.export().unwrap_err();
    assert!(
        refused.to_string().contains("suspended"),
        "an exported answer names the suspension, not a driver error: {refused}"
    );

    // The restore maps new storage; the block the buffer holds must name it.
    allocator.restore().unwrap();
    assert!(
        block.export().unwrap().as_raw_fd() >= 0,
        "after the restore the block exports the storage the restore mapped"
    );
}
