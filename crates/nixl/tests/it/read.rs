//! READs between two processes of this host, each with an agent with the
//! UCX backend: the source registers its storage and exports its metadata,
//! the destination registers device storage, loads the source's metadata and
//! READs spans of the source's storage into its own, and the destination's
//! bytes are then the source's bytes inside the spans and its own outside.
//!
//! Each case runs twice: with the transports UCX selects by itself, which
//! between two processes of one host include CUDA IPC, and with UCX limited
//! to RDMA (`rc`) and CUDA copies within a process (`cuda_copy`), so that
//! the bytes cross the network adapter as a READ from another host does.
//!
//! Both processes use CUDA device 0 of the process; run with
//! `CUDA_VISIBLE_DEVICES` naming one GPU. `NIXL_CAPI_LIBRARY` names NIXL's
//! `libnixl_capi.so`; without it the loader's search path must find it.
//! `UCX_NET_DEVICES` limits UCX to the host's usable RDMA ports where UCX
//! cannot open some device. Run with
//! `cargo test -p nixl --test it -- --ignored`.

use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use cuda::{
    DeviceAllocation, DeviceId, PeerMapping, PinnedBuffer, Protection, SegmentedMapping, Stream,
    memcpy_async,
};
use nixl::{Access, Agent, Region, Segment, Span};

const DEVICE: DeviceId = DeviceId(0);

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
/// under; a span names the key of the registration it lies in. The source's
/// travels to the destination with its region.
const SOURCE_KEY: u64 = 1;
const DESTINATION_KEY: u64 = 2;

/// Names the role a re-executed test process plays: `source` or
/// `destination`.
const ROLE: &str = "UNISERVE_NIXL_TEST_ROLE";

/// Names the storage the source registers: `device`, `host`, or
/// `host-read-only` (host storage mapped read-only, registered for reads
/// alone).
const SOURCE_STORAGE: &str = "UNISERVE_NIXL_TEST_SOURCE_STORAGE";

/// Names the destination's storage: `device` (the default), or storage the
/// destination maps again at the address of a registration it released,
/// `host-remap` or `device-remap`.
const DESTINATION: &str = "UNISERVE_NIXL_TEST_DESTINATION";

/// Names NIXL's `libnixl_capi.so`.
const LIBRARY: &str = "NIXL_CAPI_LIBRARY";

/// How long one process may take to start, register and READ. A transfer
/// that never completes fails the test at this bound instead of hanging it.
const DEADLINE: Duration = Duration::from_secs(120);

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

/// The transports UCX may use in both processes.
#[derive(Clone, Copy)]
enum Transports {
    /// UCX's own selection (`UCX_TLS` unset).
    Selected,
    /// RDMA between the processes, through the network adapter.
    Rdma,
}

/// A source whose device storage is a `cuda::vmm` allocation is READ into
/// device storage of another process.
#[test]
#[ignore = "requires a CUDA device and NIXL"]
fn a_read_moves_device_storage_into_device_storage_of_another_process() {
    exchange("device", Transports::Selected);
}

/// A source whose storage is ordinary host storage is READ into device
/// storage of another process.
#[test]
#[ignore = "requires a CUDA device and NIXL"]
fn a_read_moves_host_storage_into_device_storage_of_another_process() {
    exchange("host", Transports::Selected);
}

/// As [`a_read_moves_device_storage_into_device_storage_of_another_process`],
/// through the network adapter.
#[test]
#[ignore = "requires a CUDA device, an RDMA adapter and NIXL"]
fn an_rdma_read_moves_device_storage_into_device_storage_of_another_process() {
    exchange("device", Transports::Rdma);
}

/// As [`a_read_moves_host_storage_into_device_storage_of_another_process`],
/// through the network adapter.
#[test]
#[ignore = "requires a CUDA device, an RDMA adapter and NIXL"]
fn an_rdma_read_moves_host_storage_into_device_storage_of_another_process() {
    exchange("host", Transports::Rdma);
}

/// After a READ into host storage, the destination deregisters and unmaps
/// it with a direct system call, as the Transfer Server does, maps other
/// storage at the same address, registers it, and READs again: the second
/// READ lands in the new storage. A registration cache that learned nothing
/// of the unmapping would hand the second registration the first one's
/// pinned pages, and the READ would land there, leaving the new storage
/// unwritten while it reports success.
#[test]
#[ignore = "requires a CUDA device, an RDMA adapter and NIXL"]
fn an_rdma_read_into_host_storage_mapped_again_at_a_released_address_lands_there() {
    exchange_into("host", "host-remap", Transports::Rdma);
}

/// As [`an_rdma_read_into_host_storage_mapped_again_at_a_released_address_lands_there`],
/// for device storage: one segment of a reserved range is unmapped and
/// released, and another allocation is mapped in its place.
#[test]
#[ignore = "requires a CUDA device, an RDMA adapter and NIXL"]
fn an_rdma_read_into_device_storage_mapped_again_at_a_released_address_lands_there() {
    exchange_into("host", "device-remap", Transports::Rdma);
}

/// Host storage the source maps read-only, registered for reads alone, is
/// READ through the network adapter, as another host's Transfer Server
/// READs a read-only tensor. UCX registers host storage lazily by default,
/// so a registration that asked for write access would leave this READ
/// unanswered rather than fail.
#[test]
#[ignore = "requires a CUDA device, an RDMA adapter and NIXL with read-only registration"]
fn an_rdma_read_moves_read_only_host_storage_into_device_storage_of_another_process() {
    exchange("host-read-only", Transports::Rdma);
}

/// Runs a source and a destination process and relays the source's region
/// and metadata to the destination, as a Transfer Server's own channel
/// would.
fn exchange(source_storage: &str, transports: Transports) {
    exchange_into(source_storage, "device", transports);
}

/// [`exchange`] into the destination storage `destination` names
/// ([`DESTINATION`]).
fn exchange_into(source_storage: &str, destination: &str, transports: Transports) {
    let mut source = Process::spawn("source", source_storage, destination, transports);
    let lines = source.lines();
    let mut region = None;
    let mut metadata = None;
    while region.is_none() || metadata.is_none() {
        let line = lines.recv_timeout(DEADLINE).unwrap_or_else(|error| {
            panic!(
                "the source published no region and metadata ({error}): {}",
                source.finish()
            )
        });
        if let Some((_, value)) = line.split_once("@region ") {
            region = Some(value.to_owned());
        } else if let Some((_, value)) = line.split_once("@metadata ") {
            metadata = Some(value.to_owned());
        }
    }

    let mut destination = Process::spawn("destination", source_storage, destination, transports);
    let mut input = destination.stdin();
    writeln!(input, "@region {}", region.unwrap()).unwrap();
    writeln!(input, "@metadata {}", metadata.unwrap()).unwrap();
    drop(input);
    let verified = destination.lines();
    let status = destination.finish();
    assert!(status.success(), "the destination failed: {status}");
    // The process exited, so its output ends: the reader drains it.
    assert!(
        verified.iter().any(|line| line.contains("@verified")),
        "the destination did not verify its bytes"
    );

    // The source serves until its input closes.
    drop(source.stdin());
    let status = source.finish();
    assert!(status.success(), "the source failed: {status}");
}

/// A re-executed test process in one role, killed if the test ends first.
struct Process {
    role: &'static str,
    child: Child,
}

impl Process {
    fn spawn(
        role: &'static str,
        source_storage: &str,
        destination: &str,
        transports: Transports,
    ) -> Self {
        let test = match role {
            "source" => "read::source_process",
            _ => "read::destination_process",
        };
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", test, "--ignored", "--nocapture"])
            .env(ROLE, role)
            .env(SOURCE_STORAGE, source_storage)
            .env(DESTINATION, destination)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());
        match transports {
            Transports::Selected => command.env_remove("UCX_TLS"),
            Transports::Rdma => command.env("UCX_TLS", "rc,cuda_copy"),
        };
        let child = command.spawn().unwrap();
        Self { role, child }
    }

    fn stdin(&mut self) -> ChildStdin {
        self.child.stdin.take().unwrap()
    }

    /// The process's output lines, read on a thread of their own so that a
    /// silent process cannot block the test past [`DEADLINE`]. Each line is
    /// also printed, so that a failure shows the process's output.
    fn lines(&mut self) -> mpsc::Receiver<String> {
        let stdout = self.child.stdout.take().unwrap();
        let role = self.role;
        let (sender, receiver) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if !line.contains("@metadata") {
                    println!("{role}: {line}");
                }
                if sender.send(line).is_err() {
                    break;
                }
            }
        });
        receiver
    }

    /// Waits for the process to exit, killing it at [`DEADLINE`].
    fn finish(&mut self) -> ExitStatus {
        let deadline = Instant::now() + DEADLINE;
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                return status;
            }
            if Instant::now() > deadline {
                self.child.kill().unwrap();
                panic!("a test process ran past {DEADLINE:?}");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// The source half of [`exchange`]: registers its region, publishes the
/// region and its metadata, and serves READs until its input closes.
#[test]
#[ignore = "runs as the source process of the READ tests"]
fn source_process() {
    if std::env::var(ROLE).as_deref() != Ok("source") {
        return;
    }
    let agent = new_agent("source");
    let bytes: Vec<u8> = (0..REGION_BYTES).map(source_byte).collect();

    // Device storage is a `cuda::vmm` allocation (`cuMemCreate`, mapped
    // into a reserved range), as the storage a Storage Server records is.
    let allocation;
    let read_only;
    let mut access = Access::ReadWrite;
    let region = match std::env::var(SOURCE_STORAGE).unwrap().as_str() {
        "device" => {
            allocation = DeviceAllocation::exportable(DEVICE, REGION_BYTES, &[]).unwrap();
            upload(allocation.address(), &bytes);
            Region {
                segment: Segment::Device,
                address: allocation.address(),
                len: REGION_BYTES,
                key: SOURCE_KEY,
            }
        }
        "host-read-only" => {
            read_only = ReadOnlyMapping::of(&bytes);
            access = Access::Read;
            Region {
                segment: Segment::Host,
                address: read_only.address,
                len: REGION_BYTES,
                key: SOURCE_KEY,
            }
        }
        _ => Region {
            segment: Segment::Host,
            address: bytes.as_ptr() as u64,
            len: REGION_BYTES,
            key: SOURCE_KEY,
        },
    };
    // SAFETY: `allocation`, `read_only` or `bytes` outlives the
    // registration, and only peers read the region while it is registered.
    let _registration = unsafe { agent.register(region, access) }.unwrap();

    println!("@region {} {} {}", region.address, region.len, region.key);
    println!("@metadata {}", hex(&agent.metadata().unwrap()));
    std::io::stdout().flush().unwrap();

    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
}

/// The destination half of [`exchange`]: READs the source's spans into its
/// device storage, waits on the host, and verifies the whole region.
#[test]
#[ignore = "runs as the destination process of the READ tests"]
fn destination_process() {
    if std::env::var(ROLE).as_deref() != Ok("destination") {
        return;
    }
    let mut source_address = 0;
    let mut source_key = 0;
    let mut metadata = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        if let Some(value) = line.strip_prefix("@region ") {
            let fields: Vec<&str> = value.split(' ').collect();
            assert_eq!(fields[1].parse::<usize>().unwrap(), REGION_BYTES);
            source_address = fields[0].parse().unwrap();
            source_key = fields[2].parse().unwrap();
        } else if let Some(value) = line.strip_prefix("@metadata ") {
            metadata = unhex(value);
        }
    }
    let source_storage = match std::env::var(SOURCE_STORAGE).unwrap().as_str() {
        "device" => Segment::Device,
        _ => Segment::Host,
    };
    let agent = new_agent("destination");
    let peer = agent.load_peer(&metadata).unwrap();
    assert_eq!(peer, "source");
    let source = Region {
        segment: source_storage,
        address: source_address,
        len: REGION_BYTES,
        key: source_key,
    };
    match std::env::var(DESTINATION).as_deref() {
        Ok("host-remap") => {
            read_into_host_mapped_again(&agent, &peer, source);
            println!("@verified");
            return;
        }
        Ok("device-remap") => {
            read_into_device_mapped_again(&agent, &peer, source);
            println!("@verified");
            return;
        }
        _ => {}
    }

    // The destination is storage a Storage Server would retain, mapped as a
    // Transfer Server maps it: from the allocation's POSIX descriptor into
    // a range of its own. The bytes are written through the creating
    // allocation and read back through it, so they cross the mapping.
    let allocation = DeviceAllocation::retainable(DEVICE, REGION_BYTES, &[]).unwrap();
    let descriptor = allocation.export_descriptor().unwrap().unwrap();
    let mapping = PeerMapping::import_retained(
        DEVICE,
        &descriptor,
        allocation.len(),
        &[],
        Protection::ReadWrite,
    )
    .unwrap();
    let before: Vec<u8> = (0..REGION_BYTES).map(destination_byte).collect();
    upload(allocation.address(), &before);
    let destination = Region {
        segment: Segment::Device,
        address: mapping.address(),
        len: REGION_BYTES,
        key: DESTINATION_KEY,
    };
    // SAFETY: `mapping` outlives the registration, and the host reads the
    // storage only after the READ completed.
    let _registration = unsafe { agent.register(destination, Access::ReadWrite) }.unwrap();

    read_spans(&agent, &peer, source, destination);
    assert_expected(&download(allocation.address(), REGION_BYTES));
    println!("@verified");
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
    agent.read(peer, &spans).unwrap().wait().unwrap();
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

/// The `host-remap` destination: two memfds mapped one after the other at
/// one address, each registered under a key of its own and READ into.
fn read_into_host_mapped_again(agent: &Agent, peer: &str, source: Region) {
    let before: Vec<u8> = (0..REGION_BYTES).map(destination_byte).collect();
    let mut address = None;
    for key in [DESTINATION_KEY, DESTINATION_KEY + 1] {
        let mapping = HostMapping::of(&before, address);
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

/// The `device-remap` destination: one segment of a reserved range mapped
/// from one allocation, then, after its release, from another, each
/// registered under a key of its own and READ into.
fn read_into_device_mapped_again(agent: &Agent, peer: &str, source: Region) {
    let granularity = DeviceAllocation::granularity(DEVICE).unwrap();
    let length = REGION_BYTES.next_multiple_of(granularity);
    let mut range = SegmentedMapping::reserve(DEVICE, &[length], &[]).unwrap();
    let before: Vec<u8> = (0..REGION_BYTES).map(destination_byte).collect();
    for key in [DESTINATION_KEY, DESTINATION_KEY + 1] {
        let allocation = DeviceAllocation::retainable(DEVICE, length, &[]).unwrap();
        upload(allocation.address(), &before);
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
        assert_expected(&download(allocation.address(), REGION_BYTES));
        drop(registration);
        range.release(0).unwrap();
    }
}

fn new_agent(name: &str) -> Agent {
    if let Some(library) = std::env::var_os(LIBRARY) {
        nixl::load_library(Path::new(&library)).unwrap();
    }
    Agent::new(name).unwrap()
}

/// Copies `bytes` to device storage at `address` and waits for the copy.
fn upload(address: u64, bytes: &[u8]) {
    let stream = Stream::new(DEVICE).unwrap();
    let mut staging = PinnedBuffer::new(DEVICE, bytes.len()).unwrap();
    // SAFETY: no copy uses the buffer yet.
    unsafe { staging.as_mut_slice() }.copy_from_slice(bytes);
    // SAFETY: both ranges hold `bytes.len()` bytes and live past the
    // synchronization below.
    unsafe { memcpy_async(address, staging.address(), bytes.len(), &stream) }.unwrap();
    stream.synchronize().unwrap();
}

/// Copies `len` bytes of device storage at `address` to the host.
fn download(address: u64, len: usize) -> Vec<u8> {
    let stream = Stream::new(DEVICE).unwrap();
    let staging = PinnedBuffer::new(DEVICE, len).unwrap();
    // SAFETY: both ranges hold `len` bytes and live past the
    // synchronization below.
    unsafe { memcpy_async(staging.address(), address, len, &stream) }.unwrap();
    stream.synchronize().unwrap();
    // SAFETY: the copy that wrote the buffer completed.
    unsafe { staging.as_slice() }.to_vec()
}

/// `bytes` in a memfd mapped read-write, at `address` when given (which
/// nothing maps then), as a Transfer Server maps an exclusive tensor;
/// unmapped on drop with a direct system call, as the Transfer Server
/// unmaps (`rustix`), which no library's hook of `munmap` observes.
struct HostMapping {
    address: u64,
    len: usize,
}

unsafe extern "C" {
    fn syscall(number: std::ffi::c_long, ...) -> std::ffi::c_long;
}

impl HostMapping {
    fn of(bytes: &[u8], address: Option<u64>) -> Self {
        use std::os::fd::{AsRawFd, FromRawFd};

        const PROT_READ_WRITE: std::ffi::c_int = 1 | 2;
        const MAP_SHARED: std::ffi::c_int = 1;
        const MAP_FIXED_NOREPLACE: std::ffi::c_int = 0x10_0000;
        // SAFETY: a NUL-terminated name; the descriptor is owned by `file`.
        let file = unsafe {
            let descriptor = memfd_create(c"nixl-remap".as_ptr(), 0);
            assert!(descriptor >= 0, "memfd_create failed");
            std::fs::File::from_raw_fd(descriptor)
        };
        std::io::Write::write_all(&mut &file, bytes).unwrap();
        let (hint, flags) = match address {
            Some(address) => (
                address as *mut std::ffi::c_void,
                MAP_SHARED | MAP_FIXED_NOREPLACE,
            ),
            None => (std::ptr::null_mut(), MAP_SHARED),
        };
        // SAFETY: a new shared mapping of the file's `bytes.len()` bytes, at
        // `hint` only when nothing maps it; the mapping outlives the
        // descriptor, which closes below.
        let mapped = unsafe {
            mmap(
                hint,
                bytes.len(),
                PROT_READ_WRITE,
                flags,
                file.as_raw_fd(),
                0,
            )
        };
        assert_ne!(mapped as isize, -1, "mmap failed");
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
        // The `munmap` system call's number.
        #[cfg(target_arch = "aarch64")]
        const MUNMAP: std::ffi::c_long = 215;
        #[cfg(target_arch = "x86_64")]
        const MUNMAP: std::ffi::c_long = 11;
        // SAFETY: the mapping `of` made, which nothing references any more.
        unsafe { syscall(MUNMAP, self.address, self.len) };
    }
}

/// `bytes` in a memfd mapped read-only (`PROT_READ`), as a Transfer Server
/// maps a read-only tensor; unmapped on drop.
struct ReadOnlyMapping {
    address: u64,
    len: usize,
}

unsafe extern "C" {
    fn memfd_create(name: *const std::ffi::c_char, flags: std::ffi::c_uint) -> std::ffi::c_int;
    fn mmap(
        address: *mut std::ffi::c_void,
        len: usize,
        protection: std::ffi::c_int,
        flags: std::ffi::c_int,
        descriptor: std::ffi::c_int,
        offset: i64,
    ) -> *mut std::ffi::c_void;
    fn munmap(address: *mut std::ffi::c_void, len: usize) -> std::ffi::c_int;
}

impl ReadOnlyMapping {
    fn of(bytes: &[u8]) -> Self {
        use std::os::fd::{AsRawFd, FromRawFd};

        const PROT_READ: std::ffi::c_int = 1;
        const MAP_SHARED: std::ffi::c_int = 1;
        // SAFETY: a NUL-terminated name; the descriptor is owned by `file`.
        let file = unsafe {
            let descriptor = memfd_create(c"nixl-read-only".as_ptr(), 0);
            assert!(descriptor >= 0, "memfd_create failed");
            std::fs::File::from_raw_fd(descriptor)
        };
        std::io::Write::write_all(&mut &file, bytes).unwrap();
        // SAFETY: a new shared, read-only mapping of the file's `bytes.len()`
        // bytes; the mapping outlives the descriptor, which closes below.
        let address = unsafe {
            mmap(
                std::ptr::null_mut(),
                bytes.len(),
                PROT_READ,
                MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        assert_ne!(address as isize, -1, "mmap failed");
        Self {
            address: address as u64,
            len: bytes.len(),
        }
    }
}

impl Drop for ReadOnlyMapping {
    fn drop(&mut self) {
        // SAFETY: the mapping `of` made, which nothing references any more.
        unsafe { munmap(self.address as *mut std::ffi::c_void, self.len) };
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&text[index..index + 2], 16).unwrap())
        .collect()
}
