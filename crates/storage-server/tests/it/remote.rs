//! Transfers from another host: the destination host's Transfer Server has
//! the source host's Storage Server expose the source regions, and READs
//! them through NIXL.
//!
//! Two Storage Servers of this host stand for two hosts: each runs on a
//! thread of the test process with a socket and a TCP endpoint of its own,
//! and its own Transfer Server (the real one, run by this test binary as
//! `transfer::stand_in_transfer_server`), with its own NIXL agent. The
//! servers reach each other only by their endpoints, as hosts do.
//!
//! The failure test needs no GPU and no NIXL: every failure happens before
//! a READ. The READ tests need a CUDA device, an RDMA adapter and NIXL:
//! their Transfer Servers see this process's GPUs, and UCX is limited to
//! RDMA between processes (`rc`) and copies within one (`cuda_copy`), so
//! the bytes cross the network adapter as they do between hosts, not CUDA
//! IPC or shared memory. `NIXL_CAPI_LIBRARY` names NIXL's
//! `libnixl_capi.so`, and `UCX_NET_DEVICES` limits UCX to the host's usable
//! RDMA ports where UCX cannot open some device. Run them with
//! `cargo test -p storage-server --test it remote -- --ignored`.

use std::os::fd::AsFd;
use std::path::Path;

use storage::protocol::{Priority, Reply, Request, TransferEnd};
use storage::{Access, Client, Role};

use crate::server::{FORKING, Running, head, host, key, memfd, scope, tensor, worker};
use crate::transfer::{
    Gpus, end_of, read, remote_region, serving, span, tensor_region, transfer_pid, transfer_server,
};

/// A server whose Transfer Server sees this process's GPUs and READs from
/// another server through the network adapters only.
fn rdma_host() -> Running {
    // The server forks the Transfer Server; no test's dropped connection
    // must stay open in the child meanwhile.
    let _forking = FORKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let server = Running::with_transfer(|socket| {
        let mut launch = transfer_server(socket, Gpus::Visible);
        launch.arguments.insert(0, "UCX_TLS=rc,cuda_copy".into());
        Some(launch)
    });
    transfer_pid(&server.socket, None);
    server
}

/// The endpoint other hosts reach the server on `socket` at.
fn endpoint(socket: &Path) -> String {
    let mut operator = Client::connect(socket, Role::Operator).unwrap();
    match operator.call(&Request::Endpoint, &[]).unwrap().reply {
        Reply::Endpoint { tcp: Some(tcp) } => tcp,
        other => panic!("the server names no endpoint: {other:?}"),
    }
}

/// `bytes` bytes without short periods, so that bytes read from a wrong
/// offset, even one a multiple of 256 away, differ from the expected ones.
fn sequence(bytes: usize, seed: u64) -> Vec<u8> {
    (0..bytes as u64)
        .map(|index| (((index + seed) * 2_654_435_761) >> 16) as u8)
        .collect()
}

/// The first byte at which `actual` differs from `expected`.
fn first_difference(actual: &[u8], expected: &[u8]) -> Option<usize> {
    assert_eq!(actual.len(), expected.len());
    actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| actual != expected)
}

#[test]
fn a_read_from_another_host_fails_when_that_host_cannot_expose_its_source() {
    let destination_host = serving(Gpus::Hidden);
    let mut head = head(&destination_host.socket);
    let claim = head.claim(0, vec![scope("text", 0)], Vec::new()).unwrap();
    let mut worker = worker(&destination_host.socket, scope("text", 0), 1, claim);
    let destination = memfd(&[0; 64]);
    worker
        .put(
            vec![tensor(key(2), "cache", 0, 0, Access::Exclusive)],
            &[(host(64), destination.as_fd())],
        )
        .unwrap();

    // A host nothing listens on.
    let unreachable = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .to_string();
    // A host that holds the source, whose server runs no Transfer Server.
    let bare = Running::with_transfer(|_| None);
    let mut bare_head = crate::server::head(&bare.socket);
    let bare_claim = bare_head
        .claim(0, vec![scope("text", 0)], Vec::new())
        .unwrap();
    let mut bare_worker = crate::server::worker(&bare.socket, scope("text", 0), 1, bare_claim);
    let source = memfd(&sequence(64, 1));
    bare_worker
        .put(
            vec![tensor(key(1), "weights", 0, 0, Access::ReadOnly)],
            &[(host(64), source.as_fd())],
        )
        .unwrap();
    // A host whose Transfer Server runs, holding no such tensor.
    let empty = serving(Gpus::Hidden);

    let cases = [
        (unreachable, "Connection refused"),
        (endpoint(&bare.socket), "no Transfer Server"),
        (endpoint(&empty.socket), "no retained tensor"),
    ];
    for (source_host, reason) in cases {
        let task = worker
            .transfer(
                vec![span(
                    remote_region(&source_host, tensor_region(1)),
                    0,
                    tensor_region(2),
                    0,
                    16,
                )],
                Priority::Background,
                1,
            )
            .unwrap();
        match end_of(&mut worker, task) {
            (0, TransferEnd::Failed { message }) => assert!(
                message.contains(&source_host) && message.contains(reason),
                "the failure names neither {source_host} nor {reason:?}: {message}"
            ),
            other => panic!("a read from {source_host} ended {other:?}"),
        }
    }
    assert_eq!(read(&destination, 0, 64), [0; 64]);
}

/// READs between two hosts' GPUs and from host memory: the GPU tests.
mod device {
    use std::os::fd::AsFd;
    use std::time::Instant;

    use cuda::{DeviceAllocation, DeviceId};
    use storage::Access;
    use storage::protocol::{Link, Memory, Priority, TransferEnd};

    use super::{endpoint, first_difference, rdma_host, sequence};
    use crate::server::{FORKING, head, host, key, memfd, scope, tensor, worker};
    use crate::transfer::device::{device_bytes, device_storage, record};
    use crate::transfer::{end_of, remote_region, span, tensor_region, transfer_pid};

    /// The bytes each span of the READ test moves: long enough for the
    /// link's first timed chunks (64 MiB each) to measure it.
    const BYTES: usize = 256 << 20;

    #[test]
    #[ignore = "requires a CUDA device, an RDMA adapter and NIXL"]
    fn a_read_from_another_host_moves_device_and_host_memory_into_device_memory() {
        let source_host = rdma_host();
        let destination_host = rdma_host();
        let first = DeviceId(0);
        let last = DeviceId(DeviceId::count().unwrap() - 1);
        let granularity = DeviceAllocation::granularity(first).unwrap();

        // The source host: read-only weights on the first GPU, 4 KiB into
        // their storage, and an exclusive tensor in host memory.
        let mut source_head = head(&source_host.socket);
        let claim = source_head
            .claim(0, vec![scope("text", 0)], Vec::new())
            .unwrap();
        let mut source_worker = worker(&source_host.socket, scope("text", 0), 1, claim);
        let device_len = BYTES + granularity;
        let device_source = sequence(device_len, 5);
        let (_device_source, device_fd) = device_storage(first, &device_source);
        let host_len = BYTES + 4096;
        let host_source = sequence(host_len, 9);
        let host_fd = memfd(&host_source);
        source_worker
            .put(
                vec![
                    tensor(key(1), "weights", 0, 4096, Access::ReadOnly),
                    tensor(key(2), "offloaded", 1, 0, Access::Exclusive),
                ],
                &[
                    (record(first, device_len), device_fd.as_fd()),
                    (host(host_len as u64), host_fd.as_fd()),
                ],
            )
            .unwrap();

        // The destination host: one exclusive tensor on the last GPU.
        let mut destination_head = head(&destination_host.socket);
        let claim = destination_head
            .claim(0, vec![scope("text", 0)], Vec::new())
            .unwrap();
        let mut destination_worker = worker(&destination_host.socket, scope("text", 0), 1, claim);
        let destination_len = 2 * BYTES + granularity;
        let (destination, destination_fd) = device_storage(last, &vec![0; destination_len]);
        destination_worker
            .put(
                vec![tensor(key(3), "cache", 0, 0, Access::Exclusive)],
                &[(record(last, destination_len), destination_fd.as_fd())],
            )
            .unwrap();

        // Offsets are unaligned, and count from each tensor's first byte.
        let source = endpoint(&source_host.socket);
        let device_bytes_read = &device_source[4096 + 13..4096 + 13 + BYTES];
        let host_bytes_read = &host_source[7..7 + BYTES];
        // Reads the device source to `device_at` and the host source to
        // `host_at` of the destination, and checks the whole destination.
        let read = |worker: &mut storage::Client, device_at: usize, host_at: usize| {
            let started = Instant::now();
            let task = worker
                .transfer(
                    vec![
                        span(
                            remote_region(&source, tensor_region(1)),
                            13,
                            tensor_region(3),
                            device_at as u64,
                            BYTES as u64,
                        ),
                        span(
                            remote_region(&source, tensor_region(2)),
                            7,
                            tensor_region(3),
                            host_at as u64,
                            BYTES as u64,
                        ),
                    ],
                    Priority::Promotion,
                    1,
                )
                .unwrap();
            assert_eq!(
                end_of(worker, task),
                (2 * BYTES as u64, TransferEnd::Complete)
            );
            let elapsed = started.elapsed();

            let mut expected = vec![0; destination_len];
            expected[device_at..device_at + BYTES].copy_from_slice(device_bytes_read);
            expected[host_at..host_at + BYTES].copy_from_slice(host_bytes_read);
            assert_eq!(
                first_difference(&device_bytes(&destination), &expected),
                None,
                "the destination differs from the expected bytes"
            );
            elapsed
        };

        // The first task starts both agents, has the source host register
        // its regions and measures both links; the second finds them
        // registered and swaps the two destinations, so that every byte it
        // writes differs from the first task's.
        let ranges = [64, BYTES + 8192];
        let first_task = read(&mut destination_worker, ranges[0], ranges[1]);
        let second_task = read(&mut destination_worker, ranges[1], ranges[0]);
        for (task, elapsed) in [("first", first_task), ("second", second_task)] {
            eprintln!(
                "the {task} task read {} MiB from another host in {elapsed:?}: {:.1} GB/s",
                (2 * BYTES) >> 20,
                (2 * BYTES) as f64 / elapsed.as_secs_f64() / 1e9
            );
        }

        // The destination host lists both links from the source host, which
        // the READs measured.
        let transfer = destination_head.list().unwrap().transfer.unwrap();
        let gpu = |device: DeviceId| device.uuid().unwrap().0;
        for source_memory in [Some(gpu(first)), None] {
            let link = Link {
                source: Memory::Remote {
                    endpoint: source.clone(),
                    gpu: source_memory,
                },
                destination: Memory::Device { gpu: gpu(last) },
            };
            let measured = transfer
                .links
                .iter()
                .find(|measured| measured.link == link)
                .unwrap_or_else(|| panic!("{link:?} was not measured"));
            assert!(measured.bytes_per_second > 0);
            eprintln!(
                "{:?} -> {:?}: {:.1} GB/s",
                link.source,
                link.destination,
                measured.bytes_per_second as f64 / 1e9
            );
        }

        // Once the tasks ended and their exposures with them, neither
        // Transfer Server holds a GPU's context: NVML lists neither on a GPU
        // either used.
        let nvml = cuda::Nvml::init().unwrap();
        let pids =
            [&source_host, &destination_host].map(|host| transfer_pid(&host.socket, None) as u32);
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        for device in [first, last] {
            let uuid = device.uuid().unwrap();
            for pid in pids {
                while nvml.lists_process(uuid, pid).unwrap() {
                    assert!(
                        Instant::now() < deadline,
                        "Transfer Server {pid} still holds a context on {device}"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
        }
    }

    #[test]
    #[ignore = "requires a CUDA device, an RDMA adapter and NIXL"]
    fn a_source_host_whose_transfer_server_restarted_is_read_again() {
        let source_host = rdma_host();
        let destination_host = rdma_host();
        let first = DeviceId(0);
        let last = DeviceId(DeviceId::count().unwrap() - 1);
        let granularity = DeviceAllocation::granularity(first).unwrap();

        let mut source_head = head(&source_host.socket);
        let claim = source_head
            .claim(0, vec![scope("text", 0)], Vec::new())
            .unwrap();
        let mut source_worker = worker(&source_host.socket, scope("text", 0), 1, claim);
        let source_bytes = sequence(granularity, 3);
        let (_source, source_fd) = device_storage(first, &source_bytes);
        source_worker
            .put(
                vec![tensor(key(1), "weights", 0, 0, Access::ReadOnly)],
                &[(record(first, granularity), source_fd.as_fd())],
            )
            .unwrap();

        let mut destination_head = head(&destination_host.socket);
        let claim = destination_head
            .claim(0, vec![scope("text", 0)], Vec::new())
            .unwrap();
        let mut destination_worker = worker(&destination_host.socket, scope("text", 0), 1, claim);
        let (destination, destination_fd) = device_storage(last, &vec![0; 2 * granularity]);
        destination_worker
            .put(
                vec![tensor(key(2), "cache", 0, 0, Access::Exclusive)],
                &[(record(last, 2 * granularity), destination_fd.as_fd())],
            )
            .unwrap();

        let source = endpoint(&source_host.socket);
        let read_into = |worker: &mut storage::Client, offset: usize| {
            let task = worker
                .transfer(
                    vec![span(
                        remote_region(&source, tensor_region(1)),
                        0,
                        tensor_region(2),
                        offset as u64,
                        granularity as u64,
                    )],
                    Priority::Background,
                    1,
                )
                .unwrap();
            end_of(worker, task)
        };
        let complete = (granularity as u64, TransferEnd::Complete);
        assert_eq!(read_into(&mut destination_worker, 0), complete);

        // The source host's Transfer Server ends; its server starts another,
        // whose agent registers the region again when asked.
        {
            // The forks of the restart hold no test's connections.
            let _forking = FORKING
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let ended = transfer_pid(&source_host.socket, None);
            // SAFETY: SIGKILL to the Transfer Server this test's server
            // started.
            assert_eq!(unsafe { libc::kill(ended, libc::SIGKILL) }, 0);
            transfer_pid(&source_host.socket, Some(ended));
        }
        assert_eq!(read_into(&mut destination_worker, granularity), complete);

        let copied = device_bytes(&destination);
        assert_eq!(
            first_difference(&copied[..granularity], &source_bytes),
            None
        );
        assert_eq!(
            first_difference(&copied[granularity..], &source_bytes),
            None
        );
    }
}
