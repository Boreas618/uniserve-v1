//! READs between two agents with the UCX backend in this process: the source registers its
//! storage and exports its metadata, the destination registers device storage, loads the source's
//! metadata and READs spans of the source's storage into its own, and the destination's bytes are
//! then the source's bytes inside the spans and its own outside.
//!
//! Each case runs with the transports UCX selects by itself, or limited to RDMA
//! ([`Transports::Rdma`]: `rc` between agents, `cuda_copy` within one), so that the bytes cross
//! the network adapter as a READ from another host does. Every case needs the NIXL package
//! (`docs/nixl.md`), a CUDA device ([`device`]) and a usable RDMA port, which the agents' backend
//! names from the kernel's port states; run them with
//! `cargo test -p storage-server --lib transfer::nixl -- --ignored`.

use std::os::fd::AsFd;
use std::time::Instant;

use ::storage::settings::Transports;
use platform::{
    DeviceAllocation, DeviceId, PeerMapping, PinnedBuffer, Protection, SegmentedMapping, Stream,
    memcpy_async,
};
use rustix::mm::{MapFlags, ProtFlags};

use super::super::{Access, Agent, Backend, Region, Segment, Span};

/// The length of the source's and the destination's region.
const REGION_BYTES: usize = 4 << 20;

/// The spans each READ moves, as (source offset, destination offset,
/// length). Offsets and lengths are not aligned to anything, and the spans
/// leave most of the destination unwritten.
const SPANS: [(usize, usize, usize); 2] = [
    ((1 << 20) + 4099, 64, 1 << 20),
    (13, (2 << 20) + 17, (256 << 10) + 5),
];

/// The keys the source's and the destination's regions are registered
/// under; a span names the key of the registration it lies in.
const SOURCE_KEY: u64 = 1;
const DESTINATION_KEY: u64 = 2;

/// The CUDA device every case copies on: the first the driver shows, which a case requires.
fn device() -> DeviceId {
    let count = DeviceId::count().expect("the CUDA driver loads");
    assert!(count > 0, "the case requires a CUDA device");
    DeviceId(0)
}

/// The storage the source registers.
#[derive(Clone, Copy)]
enum Source {
    /// A `platform::vmm` allocation (`cuMemCreate`, mapped into a reserved range), as the
    /// storage a Storage Server records is.
    Device,
    /// Ordinary host storage.
    Host,
    /// Host storage mapped read-only and registered for reads alone, as a Transfer Server
    /// exposes a read-only tensor.
    HostReadOnly,
}

/// The destination's storage.
#[derive(Clone, Copy)]
enum Destination {
    /// Device storage mapped from its POSIX descriptor, as a Transfer Server maps it.
    Device,
    /// Host storage mapped again at the address of a registration it released.
    HostRemap,
    /// Device storage mapped again at the address of a registration it released.
    DeviceRemap,
}

/// The source region's byte `index`: a sequence without short periods, so
/// that bytes read from a wrong offset differ from the expected ones.
fn source_byte(index: usize) -> u8 {
    ((index as u64 * 2_654_435_761) >> 16) as u8
}

/// The destination region's byte `index` before the READ.
fn destination_byte(index: usize) -> u8 {
    (index * 7 + 3) as u8
}

/// The destination region after the READ.
fn expected_destination() -> Vec<u8> {
    let mut expected: Vec<u8> = (0..REGION_BYTES).map(destination_byte).collect();
    for (source, destination, len) in SPANS {
        for offset in 0..len {
            expected[destination + offset] = source_byte(source + offset);
        }
    }
    expected
}

#[test]
#[ignore = "requires a CUDA device, a usable RDMA port and NIXL"]
fn a_read_moves_device_storage_into_device_storage_of_another_agent() {
    exchange(Source::Device, Destination::Device, Transports::Selected);
}

#[test]
#[ignore = "requires a CUDA device, a usable RDMA port and NIXL"]
fn a_read_moves_host_storage_into_device_storage_of_another_agent() {
    exchange(Source::Host, Destination::Device, Transports::Selected);
}

/// As [`a_read_moves_device_storage_into_device_storage_of_another_agent`], through the network
/// adapter.
#[test]
#[ignore = "requires a CUDA device, a usable RDMA port and NIXL"]
fn an_rdma_read_moves_device_storage_into_device_storage_of_another_agent() {
    exchange(Source::Device, Destination::Device, Transports::Rdma);
}

/// As [`a_read_moves_host_storage_into_device_storage_of_another_agent`], through the network
/// adapter.
#[test]
#[ignore = "requires a CUDA device, a usable RDMA port and NIXL"]
fn an_rdma_read_moves_host_storage_into_device_storage_of_another_agent() {
    exchange(Source::Host, Destination::Device, Transports::Rdma);
}

/// After a READ into host storage, the destination deregisters and unmaps
/// it with a direct system call, as the Transfer Server does, maps other
/// storage at the same address, registers it, and READs again: the second
/// READ lands in the new storage. A registration cache that learned nothing
/// of the unmapping would hand the second registration the first one's
/// pinned pages, and the READ would land there, leaving the new storage
/// unwritten while it reports success.
#[test]
#[ignore = "requires a CUDA device, a usable RDMA port and NIXL"]
fn an_rdma_read_into_host_storage_mapped_again_at_a_released_address_lands_there() {
    exchange(Source::Host, Destination::HostRemap, Transports::Rdma);
}

/// As [`an_rdma_read_into_host_storage_mapped_again_at_a_released_address_lands_there`],
/// for device storage: one segment of a reserved range is unmapped and
/// released, and another allocation is mapped in its place.
#[test]
#[ignore = "requires a CUDA device, a usable RDMA port and NIXL"]
fn an_rdma_read_into_device_storage_mapped_again_at_a_released_address_lands_there() {
    exchange(Source::Host, Destination::DeviceRemap, Transports::Rdma);
}

/// Host storage the source maps read-only, registered for reads alone, is
/// READ through the network adapter, as another host's Transfer Server
/// READs a read-only tensor. UCX registers host storage lazily by default,
/// so a registration that asked for write access would leave this READ
/// without a response rather than fail.
#[test]
#[ignore = "requires a CUDA device, a usable RDMA port and NIXL with read-only registration"]
fn an_rdma_read_moves_read_only_host_storage_into_device_storage_of_another_agent() {
    exchange(Source::HostReadOnly, Destination::Device, Transports::Rdma);
}

/// The storage a source registers, kept until its registration dropped.
enum Kept {
    Device(DeviceAllocation),
    Host(Vec<u8>),
    ReadOnly(HostMapping),
}

/// Has a source agent register `source` and a destination agent READ its spans into
/// `destination`, both agents over the host's usable RDMA ports with `transports`; the source's
/// region and metadata pass to the destination as a Transfer Server's own channel passes them.
fn exchange(source: Source, destination: Destination, transports: Transports) {
    let device = device();
    let backend = Backend::of_host(transports).unwrap();
    let source_agent = Agent::new("source", &backend).unwrap();
    let destination_agent = Agent::new("destination", &backend).unwrap();

    let bytes: Vec<u8> = (0..REGION_BYTES).map(source_byte).collect();
    let (kept, segment, access) = match source {
        Source::Device => {
            let allocation = DeviceAllocation::exportable(device, REGION_BYTES, &[]).unwrap();
            upload(device, allocation.address(), &bytes);
            (Kept::Device(allocation), Segment::Device, Access::ReadWrite)
        }
        Source::Host => (Kept::Host(bytes), Segment::Host, Access::ReadWrite),
        Source::HostReadOnly => (
            Kept::ReadOnly(HostMapping::of(&bytes, None, ProtFlags::READ)),
            Segment::Host,
            Access::Read,
        ),
    };
    let address = match &kept {
        Kept::Device(allocation) => allocation.address(),
        Kept::Host(bytes) => bytes.as_ptr() as u64,
        Kept::ReadOnly(mapping) => mapping.address,
    };
    let region = Region {
        segment,
        address,
        len: REGION_BYTES,
        key: SOURCE_KEY,
    };
    // SAFETY: `kept` outlives the registration, and only the destination's READs touch the
    // region while it is registered.
    let registration = unsafe { source_agent.register(region, access) }.unwrap();

    let peer = destination_agent
        .load_peer(&source_agent.metadata().unwrap())
        .unwrap();
    assert_eq!(peer, "source");
    match destination {
        Destination::Device => read_into_device(device, &destination_agent, &peer, region),
        Destination::HostRemap => read_into_host_mapped_again(&destination_agent, &peer, region),
        Destination::DeviceRemap => {
            read_into_device_mapped_again(device, &destination_agent, &peer, region);
        }
    }
    drop(registration);
    drop(kept);
}

/// READs the spans of `source` into device storage a Storage Server would retain, mapped as a
/// Transfer Server maps it: from the allocation's POSIX descriptor into a range of its own. The
/// bytes are written through the creating allocation and read back through it, so they cross
/// the mapping.
fn read_into_device(device: DeviceId, agent: &Agent, peer: &str, source: Region) {
    let allocation = DeviceAllocation::retainable(device, REGION_BYTES, &[]).unwrap();
    let descriptor = allocation.export_descriptor().unwrap().unwrap();
    let mapping = PeerMapping::import_retained(
        device,
        &descriptor,
        allocation.len(),
        &[],
        Protection::ReadWrite,
    )
    .unwrap();
    let before: Vec<u8> = (0..REGION_BYTES).map(destination_byte).collect();
    upload(device, allocation.address(), &before);
    let destination = Region {
        segment: Segment::Device,
        address: mapping.address(),
        len: REGION_BYTES,
        key: DESTINATION_KEY,
    };
    // SAFETY: `mapping` outlives the registration, and the host reads the
    // storage only after the READ completed.
    let _registration = unsafe { agent.register(destination, Access::ReadWrite) }.unwrap();

    read_spans(agent, peer, source, destination);
    assert_expected(&download(device, allocation.address(), REGION_BYTES));
}

/// READs [`SPANS`] of `source` into `destination` and waits for them.
fn read_spans(agent: &Agent, peer: &str, source: Region, destination: Region) {
    let spans = SPANS.map(|(source_offset, destination_offset, len)| Span {
        source: Region {
            address: source.address + source_offset as u64,
            len,
            ..source
        },
        destination: Region {
            address: destination.address + destination_offset as u64,
            len,
            ..destination
        },
    });
    // Once the READ completes, the destination bytes are readable by the host and by device work
    // launched after it.
    let started = Instant::now();
    agent.read(peer, &spans).unwrap().wait(started).unwrap();
}

/// Asserts that a destination's bytes after the READ are the expected ones.
fn assert_expected(after: &[u8]) {
    let mismatch = after
        .iter()
        .zip(&expected_destination())
        .position(|(byte, expected)| byte != expected);
    assert_eq!(
        mismatch, None,
        "the destination differs from the expected bytes"
    );
}

/// The `HostRemap` destination: two memfds mapped one after the other at
/// one address, each registered under a key of its own and READ into.
fn read_into_host_mapped_again(agent: &Agent, peer: &str, source: Region) {
    let before: Vec<u8> = (0..REGION_BYTES).map(destination_byte).collect();
    let mut address = None;
    for key in [DESTINATION_KEY, DESTINATION_KEY + 1] {
        let mapping = HostMapping::of(&before, address, ProtFlags::READ | ProtFlags::WRITE);
        address = Some(mapping.address);
        let destination = Region {
            segment: Segment::Host,
            address: mapping.address,
            len: REGION_BYTES,
            key,
        };
        // SAFETY: the mapping outlives the registration, which drops
        // first, and the host reads the bytes only after the READ.
        let registration = unsafe { agent.register(destination, Access::ReadWrite) }.unwrap();
        read_spans(agent, peer, source, destination);
        assert_expected(&mapping.bytes());
        // Deregistered, then unmapped: the next mapping takes the address.
        drop(registration);
        drop(mapping);
    }
}

/// The `DeviceRemap` destination: one segment of a reserved range mapped
/// from one allocation, then, after its release, from another, each
/// registered under a key of its own and READ into.
fn read_into_device_mapped_again(device: DeviceId, agent: &Agent, peer: &str, source: Region) {
    let granularity = DeviceAllocation::granularity(device).unwrap();
    let length = REGION_BYTES.next_multiple_of(granularity);
    let mut range = SegmentedMapping::reserve(device, &[length], &[]).unwrap();
    let before: Vec<u8> = (0..REGION_BYTES).map(destination_byte).collect();
    for key in [DESTINATION_KEY, DESTINATION_KEY + 1] {
        let allocation = DeviceAllocation::retainable(device, length, &[]).unwrap();
        upload(device, allocation.address(), &before);
        let descriptor = allocation.export_descriptor().unwrap().unwrap();
        range.import(0, &descriptor).unwrap();
        let destination = Region {
            segment: Segment::Device,
            address: range.address(),
            len: REGION_BYTES,
            key,
        };
        // SAFETY: the segment stays mapped until after the registration
        // dropped, and the host reads the bytes only after the READ.
        let registration = unsafe { agent.register(destination, Access::ReadWrite) }.unwrap();
        read_spans(agent, peer, source, destination);
        assert_expected(&download(device, allocation.address(), REGION_BYTES));
        drop(registration);
        range.release(0).unwrap();
    }
}

/// Copies `bytes` to device storage at `address` on `device` and waits for the copy.
fn upload(device: DeviceId, address: u64, bytes: &[u8]) {
    let stream = Stream::new(device).unwrap();
    let mut staging = PinnedBuffer::new(device, bytes.len()).unwrap();
    // SAFETY: no copy uses the buffer yet.
    unsafe { staging.as_mut_slice() }.copy_from_slice(bytes);
    // SAFETY: both ranges hold `bytes.len()` bytes and live past the
    // synchronization below.
    unsafe { memcpy_async(address, staging.address(), bytes.len(), &stream) }.unwrap();
    stream.synchronize().unwrap();
}

/// Copies `len` bytes of device storage at `address` on `device` to the host.
fn download(device: DeviceId, address: u64, len: usize) -> Vec<u8> {
    let stream = Stream::new(device).unwrap();
    let staging = PinnedBuffer::new(device, len).unwrap();
    // SAFETY: both ranges hold `len` bytes and live past the
    // synchronization below.
    unsafe { memcpy_async(staging.address(), address, len, &stream) }.unwrap();
    stream.synchronize().unwrap();
    // SAFETY: the copy that wrote the buffer completed.
    unsafe { staging.as_slice() }.to_vec()
}

/// `bytes` in a memfd mapped with `protection`, at `address` when given (which nothing maps
/// then), as a Transfer Server maps a tensor; unmapped on drop with `rustix`'s direct system
/// call, as the Transfer Server unmaps, which no library's hook of `munmap` observes.
struct HostMapping {
    address: u64,
    len: usize,
}

impl HostMapping {
    fn of(bytes: &[u8], address: Option<u64>, protection: ProtFlags) -> Self {
        let descriptor =
            rustix::fs::memfd_create(c"nixl-read", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
        let file = std::fs::File::from(descriptor);
        std::io::Write::write_all(&mut &file, bytes).unwrap();
        let (hint, flags) = match address {
            Some(address) => (
                address as *mut std::ffi::c_void,
                MapFlags::SHARED | MapFlags::FIXED_NOREPLACE,
            ),
            None => (std::ptr::null_mut(), MapFlags::SHARED),
        };
        // SAFETY: a new shared mapping of the file's `bytes.len()` bytes, at `hint` only when
        // nothing maps it; the mapping outlives the descriptor, which closes below.
        let mapped =
            unsafe { rustix::mm::mmap(hint, bytes.len(), protection, flags, file.as_fd(), 0) }
                .unwrap();
        if let Some(address) = address {
            assert_eq!(mapped as u64, address, "the address was not mapped again");
        }
        Self {
            address: mapped as u64,
            len: bytes.len(),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        // SAFETY: the mapping holds `len` readable bytes, which no READ
        // writes any more.
        unsafe { std::slice::from_raw_parts(self.address as *const u8, self.len) }.to_vec()
    }
}

impl Drop for HostMapping {
    fn drop(&mut self) {
        // SAFETY: the mapping `of` made, which nothing references any more.
        let unmapped =
            unsafe { rustix::mm::munmap(self.address as *mut std::ffi::c_void, self.len) };
        unmapped.unwrap();
    }
}
