//! READs between two processes of this host, each with an agent with the
//! UCX backend: the source registers its memory and exports its metadata,
//! the destination registers device memory, loads the source's metadata and
//! READs spans of the source's memory into its own, and the destination's
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
    DeviceAllocation, DeviceId, PeerMapping, PinnedBuffer, Protection, Stream, memcpy_async,
};
use nixl::{Agent, Memory, Region, Span};

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

/// Names the role a re-executed test process plays: `source` or
/// `destination`.
const ROLE: &str = "UNISERVE_NIXL_TEST_ROLE";

/// Names the memory the source registers: `device` or `host`.
const SOURCE_MEMORY: &str = "UNISERVE_NIXL_TEST_SOURCE_MEMORY";

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

/// A source whose device memory is a `cuda::vmm` allocation is READ into
/// device memory of another process.
#[test]
#[ignore = "requires a CUDA device and NIXL"]
fn a_read_moves_device_memory_into_device_memory_of_another_process() {
    exchange("device", Transports::Selected);
}

/// A source whose memory is ordinary host memory is READ into device
/// memory of another process.
#[test]
#[ignore = "requires a CUDA device and NIXL"]
fn a_read_moves_host_memory_into_device_memory_of_another_process() {
    exchange("host", Transports::Selected);
}

/// As [`a_read_moves_device_memory_into_device_memory_of_another_process`],
/// through the network adapter.
#[test]
#[ignore = "requires a CUDA device, an RDMA adapter and NIXL"]
fn an_rdma_read_moves_device_memory_into_device_memory_of_another_process() {
    exchange("device", Transports::Rdma);
}

/// As [`a_read_moves_host_memory_into_device_memory_of_another_process`],
/// through the network adapter.
#[test]
#[ignore = "requires a CUDA device, an RDMA adapter and NIXL"]
fn an_rdma_read_moves_host_memory_into_device_memory_of_another_process() {
    exchange("host", Transports::Rdma);
}

/// Runs a source and a destination process and relays the source's region
/// and metadata to the destination, as a Transfer Server's own channel
/// would.
fn exchange(source_memory: &str, transports: Transports) {
    let mut source = Process::spawn("source", source_memory, transports);
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

    let mut destination = Process::spawn("destination", source_memory, transports);
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
    fn spawn(role: &'static str, source_memory: &str, transports: Transports) -> Self {
        let test = match role {
            "source" => "read::source_process",
            _ => "read::destination_process",
        };
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", test, "--ignored", "--nocapture"])
            .env(ROLE, role)
            .env(SOURCE_MEMORY, source_memory)
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

    // Device memory is a `cuda::vmm` allocation (`cuMemCreate`, mapped
    // into a reserved range), as the memory a Storage Server records is.
    let allocation;
    let region = match std::env::var(SOURCE_MEMORY).unwrap().as_str() {
        "device" => {
            allocation = DeviceAllocation::exportable(DEVICE, REGION_BYTES, &[]).unwrap();
            upload(allocation.address(), &bytes);
            Region {
                memory: Memory::Device(0),
                address: allocation.address(),
                len: REGION_BYTES,
            }
        }
        _ => Region {
            memory: Memory::Host,
            address: bytes.as_ptr() as u64,
            len: REGION_BYTES,
        },
    };
    // SAFETY: `allocation` or `bytes` outlives the registration, and only
    // peers read the region while it is registered.
    let _registration = unsafe { agent.register(region) }.unwrap();

    println!("@region {} {}", region.address, region.len);
    println!("@metadata {}", hex(&agent.metadata().unwrap()));
    std::io::stdout().flush().unwrap();

    let mut line = String::new();
    std::io::stdin().read_line(&mut line).unwrap();
}

/// The destination half of [`exchange`]: READs the source's spans into its
/// device memory, waits on the host, and verifies the whole region.
#[test]
#[ignore = "runs as the destination process of the READ tests"]
fn destination_process() {
    if std::env::var(ROLE).as_deref() != Ok("destination") {
        return;
    }
    let mut source_address = 0;
    let mut metadata = Vec::new();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        if let Some(value) = line.strip_prefix("@region ") {
            let (address, len) = value.split_once(' ').unwrap();
            assert_eq!(len.parse::<usize>().unwrap(), REGION_BYTES);
            source_address = address.parse().unwrap();
        } else if let Some(value) = line.strip_prefix("@metadata ") {
            metadata = unhex(value);
        }
    }
    let source_memory = match std::env::var(SOURCE_MEMORY).unwrap().as_str() {
        "device" => Memory::Device(0),
        _ => Memory::Host,
    };
    let agent = new_agent("destination");

    // The destination is memory a Storage Server would retain, mapped as a
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
        memory: Memory::Device(0),
        address: mapping.address(),
        len: REGION_BYTES,
    };
    // SAFETY: `mapping` outlives the registration, and the host reads the
    // memory only after the READ completed.
    let _registration = unsafe { agent.register(destination) }.unwrap();

    let peer = agent.load_peer(&metadata).unwrap();
    assert_eq!(peer, "source");
    let spans = SPANS.map(|(source, destination, len)| Span {
        source: Region {
            memory: source_memory,
            address: source_address + source as u64,
            len,
        },
        destination: Region {
            memory: Memory::Device(0),
            address: mapping.address() + destination as u64,
            len,
        },
    });
    agent.read(&peer, &spans).unwrap().wait().unwrap();

    let after = download(allocation.address(), REGION_BYTES);
    let expected = expected_destination();
    let mismatch = after
        .iter()
        .zip(&expected)
        .position(|(byte, expected)| byte != expected);
    assert_eq!(
        mismatch, None,
        "the destination differs from the expected bytes"
    );
    println!("@verified");
}

fn new_agent(name: &str) -> Agent {
    if let Some(library) = std::env::var_os(LIBRARY) {
        nixl::load_library(Path::new(&library)).unwrap();
    }
    Agent::new(name).unwrap()
}

/// Copies `bytes` to device memory at `address` and waits for the copy.
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

/// Copies `len` bytes of device memory at `address` to the host.
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

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&text[index..index + 2], 16).unwrap())
        .collect()
}
