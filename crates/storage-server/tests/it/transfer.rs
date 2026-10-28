//! Transfers through the server's Transfer Server: copying between the
//! memory the server holds, completion, cancellation, priorities, and a
//! Transfer Server that crashes.
//!
//! The server runs on a thread of the test process; its Transfer Server is
//! the real one (`storage_server::run_transfer_server`), run as the
//! server's child process by this test binary (`stand_in_transfer_server`).
//! The host tests hide every GPU from it (`CUDA_VISIBLE_DEVICES` empty), so
//! it copies host memory with the CPU and no GPU is touched. The device
//! tests need a CUDA device and see the GPUs this process sees; run them
//! with `cargo test -p storage-server --test it transfer -- --ignored`.

use std::os::fd::AsFd;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::time::{Duration, Instant};

use storage::protocol::{
    ErrorKind, Event, Link, Memory, PageShape, Priority, Region, Reply, Request, Span, Throttle,
    TransferEnd,
};
use storage::{Access, Client, Role};
use storage_server::Launch;

use crate::server::{
    FORKING, Running, head, host, key, memfd, scope, server_error, tensor, worker,
};

/// The Transfer Server the server starts, as this test binary runs it.
#[test]
#[ignore = "the Transfer Server this file's tests start as the server's child"]
fn stand_in_transfer_server() {
    let Ok(socket) = std::env::var("STORAGE_TEST_SOCKET") else {
        return;
    };
    storage_server::run_transfer_server(
        Path::new(&socket),
        &storage_server::transfer::Options {
            chunk: Duration::from_millis(1),
            // The NIXL build the tests between hosts name; the loader's
            // search path otherwise.
            nixl_library: std::env::var_os("NIXL_CAPI_LIBRARY").map(Into::into),
        },
    )
    .unwrap();
}

/// Which GPUs the Transfer Server sees.
#[derive(Clone, Copy)]
pub(crate) enum Gpus {
    /// None: it copies host memory only.
    Hidden,
    /// Those this process sees.
    Visible,
}

/// The launch of this binary's stand-in Transfer Server for `socket`.
pub(crate) fn transfer_server(socket: &Path, gpus: Gpus) -> Launch {
    let mut arguments: Vec<std::ffi::OsString> = Vec::new();
    if let Gpus::Hidden = gpus {
        arguments.push("CUDA_VISIBLE_DEVICES=".into());
    }
    arguments.push(format!("STORAGE_TEST_SOCKET={}", socket.display()).into());
    // Its log at the level the test's environment names, warnings when
    // unset.
    let level = std::env::var("UNISERVE_STORAGE_LOG").unwrap_or_else(|_| "warn".into());
    arguments.push(format!("UNISERVE_STORAGE_LOG={level}").into());
    arguments.push(std::env::current_exe().unwrap().into());
    arguments.extend(
        [
            "--exact",
            "transfer::stand_in_transfer_server",
            "--ignored",
            "--nocapture",
            "--quiet",
        ]
        .map(Into::into),
    );
    Launch {
        program: "/usr/bin/env".into(),
        arguments,
    }
}

/// A server whose Transfer Server is connected and measured.
pub(crate) fn serving(gpus: Gpus) -> Running {
    // The server forks the Transfer Server; no test's dropped connection
    // must stay open in the child meanwhile.
    let _forking = FORKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let server = Running::with_transfer(|socket| Some(transfer_server(socket, gpus)));
    transfer_pid(&server.socket, None);
    server
}

/// The pid of the server's measured Transfer Server, once one other than
/// `not` is.
pub(crate) fn transfer_pid(socket: &Path, not: Option<i32>) -> i32 {
    let mut operator = Client::connect(socket, Role::Operator).unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Some(transfer) = operator.list().unwrap().transfer
            && !transfer.links.is_empty()
            && Some(transfer.pid) != not
        {
            return transfer.pid;
        }
        assert!(
            Instant::now() < deadline,
            "no Transfer Server reported its links"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The next transfer event `client` receives, within a minute.
pub(crate) fn transfer_event(client: &mut Client) -> Event {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        client.pump().unwrap();
        if let Some((event, _)) = client.next_event() {
            return event;
        }
        assert!(Instant::now() < deadline, "no transfer event arrived");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The end of task `task`, skipping its progress.
pub(crate) fn end_of(client: &mut Client, task: u64) -> (u64, TransferEnd) {
    loop {
        match transfer_event(client) {
            Event::Transferred {
                task: ended,
                bytes,
                end,
            } if ended == task => return (bytes, end),
            Event::TransferProgress { .. } | Event::Transferred { .. } => {}
            other => panic!("unexpected event {other:?}"),
        }
    }
}

/// Bytes of a pattern that differs at every position of a span.
pub(crate) fn pattern(bytes: usize, seed: u8) -> Vec<u8> {
    (0..bytes)
        .map(|index| (index as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

pub(crate) fn span(
    source: Region,
    source_offset: u64,
    destination: Region,
    offset: u64,
    bytes: u64,
) -> Span {
    Span {
        source,
        source_offset,
        destination,
        destination_offset: offset,
        bytes,
        rows: 1,
        source_pitch: 0,
        destination_pitch: 0,
    }
}

/// `rows` rows of `bytes` bytes at the pitches, from `source_offset` of
/// `source` to `offset` of `destination`.
pub(crate) fn rows_span(
    (source, source_offset, source_pitch): (Region, u64, u64),
    (destination, offset, destination_pitch): (Region, u64, u64),
    bytes: u64,
    rows: u64,
) -> Span {
    Span {
        rows,
        source_pitch,
        destination_pitch,
        ..span(source, source_offset, destination, offset, bytes)
    }
}

pub(crate) fn tensor_region(byte: u8) -> Region {
    Region::Tensor {
        key: key(byte),
        at: None,
    }
}

/// `region` at the Storage Server of another host listening at `endpoint`.
pub(crate) fn remote_region(endpoint: &str, region: Region) -> Region {
    Region::Remote {
        endpoint: endpoint.into(),
        region: Box::new(region),
    }
}

/// The bytes of `file` from `offset`.
pub(crate) fn read(file: &std::fs::File, offset: u64, bytes: usize) -> Vec<u8> {
    let mut buffer = vec![0; bytes];
    file.read_exact_at(&mut buffer, offset).unwrap();
    buffer
}

#[test]
fn a_transfer_copies_host_regions_and_ends_complete() {
    let server = serving(Gpus::Hidden);
    let mut head = head(&server.socket);
    let claim = head.claim(0, vec![scope("text", 0)], Vec::new()).unwrap();
    let mut worker = worker(&server.socket, scope("text", 0), 1, claim);

    // A read-only source and an exclusive destination, packed in storages
    // of their own, and a page pool of four 16-byte slots.
    let source_bytes = pattern(64, 7);
    let source = memfd(&source_bytes);
    let destination = memfd(&[0; 64]);
    worker
        .put(
            vec![tensor(key(1), "weights", 0, 0, Access::ReadOnly)],
            &[(host(64), source.as_fd())],
        )
        .unwrap();
    worker
        .put(
            vec![tensor(key(2), "cache", 0, 16, Access::Exclusive)],
            &[(host(64), destination.as_fd())],
        )
        .unwrap();
    let shape = PageShape {
        layout: key(3),
        kv_head_offset: 0,
        kv_head_count: 1,
        page_bytes: 16,
    };
    let Reply::PagePool { pool, .. } = worker
        .call(
            &Request::PagePool {
                shape,
                numa_node: None,
                bytes: None,
            },
            &[],
        )
        .unwrap()
        .reply
    else {
        panic!("the page pool was refused");
    };

    // Offsets count from the tensor's first byte: the destination tensor
    // starts at byte 16 of its storage. The pool's slot 2 then goes back
    // into the destination's last bytes.
    let task = worker
        .transfer(
            vec![
                span(tensor_region(1), 8, tensor_region(2), 0, 24),
                span(tensor_region(1), 0, Region::Pool { pool }, 32, 16),
            ],
            Priority::Warming,
            1,
        )
        .unwrap();
    assert_eq!(end_of(&mut worker, task), (40, TransferEnd::Complete));
    let pool_task = worker
        .transfer(
            vec![span(Region::Pool { pool }, 32, tensor_region(2), 32, 16)],
            Priority::Background,
            1,
        )
        .unwrap();
    assert_eq!(end_of(&mut worker, pool_task), (16, TransferEnd::Complete));

    let copied = read(&destination, 0, 64);
    assert_eq!(&copied[..16], &[0; 16]);
    assert_eq!(&copied[16..40], &source_bytes[8..32]);
    assert_eq!(&copied[40..48], &[0; 8]);
    assert_eq!(&copied[48..64], &source_bytes[..16]);
    // The source is read-only content, which no copy changes.
    assert_eq!(read(&source, 0, 64), source_bytes);

    // The listing names the host link the Transfer Server measured, and no
    // GPU context, since it sees no GPU.
    let listing = head.list().unwrap();
    let transfer = listing.transfer.expect("the Transfer Server is listed");
    let host_link = Link {
        source: Memory::Host,
        destination: Memory::Host,
    };
    assert!(
        transfer
            .links
            .iter()
            .any(|measured| measured.link == host_link && measured.bytes_per_second > 0)
    );
    assert!(transfer.contexts.is_empty());
    assert!(transfer.tasks.is_empty());
}

#[test]
fn a_span_of_rows_copies_each_row_at_its_pitches() {
    let server = serving(Gpus::Hidden);
    let mut head = head(&server.socket);
    let claim = head.claim(0, vec![scope("text", 0)], Vec::new()).unwrap();
    let mut worker = worker(&server.socket, scope("text", 0), 1, claim);
    let source_bytes = pattern(64, 3);
    let source = memfd(&source_bytes);
    let destination = memfd(&[0; 64]);
    worker
        .put(
            vec![
                tensor(key(1), "weights", 0, 0, Access::ReadOnly),
                tensor(key(2), "weights receiving", 1, 0, Access::Exclusive),
            ],
            &[(host(64), source.as_fd()), (host(64), destination.as_fd())],
        )
        .unwrap();

    // Four rows of four bytes, 16 bytes apart in the source from byte 2,
    // 8 bytes apart in the destination from byte 0: a column of a matrix
    // of 16-byte rows gathered into a matrix of 8-byte rows.
    let task = worker
        .transfer(
            vec![rows_span(
                (tensor_region(1), 2, 16),
                (tensor_region(2), 0, 8),
                4,
                4,
            )],
            Priority::Background,
            1,
        )
        .unwrap();
    assert_eq!(end_of(&mut worker, task), (16, TransferEnd::Complete));
    let copied = read(&destination, 0, 64);
    for row in 0..4 {
        let target = row * 8;
        let source = 2 + row * 16;
        assert_eq!(
            &copied[target..target + 4],
            &source_bytes[source..source + 4]
        );
        assert_eq!(&copied[target + 4..target + 8], &[0; 4]);
    }
    assert_eq!(&copied[32..], &[0; 32]);
}

#[test]
fn a_transfer_is_refused_whole_unless_every_span_fits_memory_of_its_deployment() {
    let server = serving(Gpus::Hidden);
    let mut head = head(&server.socket);
    let claim = head.claim(0, vec![scope("text", 0)], Vec::new()).unwrap();
    let mut worker = worker(&server.socket, scope("text", 0), 1, claim);
    let source = memfd(&pattern(32, 1));
    let destination = memfd(&[0; 32]);
    worker
        .put(
            vec![
                tensor(key(1), "weights", 0, 0, Access::ReadOnly),
                tensor(key(2), "latents", 1, 0, Access::Exclusive),
            ],
            &[(host(32), source.as_fd()), (host(32), destination.as_fd())],
        )
        .unwrap();
    let mut other_head = Client::connect(
        &server.socket,
        Role::Head {
            deployment: "other".into(),
            world: 0,
        },
    )
    .unwrap();

    let refused = [
        // A read-only tensor is never written.
        vec![span(tensor_region(2), 0, tensor_region(1), 0, 8)],
        // The destination's storage ends at 32 bytes.
        vec![span(tensor_region(1), 0, tensor_region(2), 30, 8)],
        // An unknown tensor.
        vec![span(tensor_region(9), 0, tensor_region(2), 0, 8)],
        // Rows overlap when a pitch is shorter than a row.
        vec![rows_span(
            (tensor_region(1), 0, 4),
            (tensor_region(2), 0, 8),
            8,
            2,
        )],
        // The last of four rows 8 bytes apart reaches past byte 32.
        vec![rows_span(
            (tensor_region(1), 0, 8),
            (tensor_region(2), 4, 8),
            8,
            4,
        )],
        // A good span does not carry a bad one.
        vec![
            span(tensor_region(1), 0, tensor_region(2), 0, 8),
            span(tensor_region(1), 0, tensor_region(2), 0, 0),
        ],
        // Another host's memory is a source only: this host READs it.
        vec![span(
            tensor_region(1),
            0,
            remote_region("10.0.0.2:7000", tensor_region(2)),
            0,
            8,
        )],
        // Another host's region is one of its tensors or page pools.
        vec![span(
            remote_region(
                "10.0.0.2:7000",
                remote_region("10.0.0.3:7000", tensor_region(1)),
            ),
            0,
            tensor_region(2),
            0,
            8,
        )],
    ];
    for spans in refused {
        let result = worker.transfer(spans, Priority::Background, 1);
        assert_eq!(server_error(result), ErrorKind::Invalid);
    }
    // Another deployment's head names none of this deployment's memory.
    let result = other_head.transfer(
        vec![span(tensor_region(1), 0, tensor_region(2), 0, 8)],
        Priority::Background,
        1,
    );
    assert_eq!(server_error(result), ErrorKind::Invalid);
    // A worker does not throttle links: the head of the executing world
    // does.
    let result = worker.throttle(vec![], Throttle::Halve);
    assert_eq!(server_error(result), ErrorKind::Invalid);
    // A refused transfer copied nothing.
    assert_eq!(read(&destination, 0, 32), [0; 32]);

    // A server that runs no Transfer Server refuses transfers as
    // unavailable.
    let bare = Running::with_transfer(|_| None);
    let claim = crate::server::head(&bare.socket)
        .claim(0, vec![scope("text", 0)], Vec::new())
        .unwrap();
    let mut worker = crate::server::worker(&bare.socket, scope("text", 0), 1, claim);
    worker
        .put(
            vec![tensor(key(1), "weights", 0, 0, Access::ReadOnly)],
            &[(host(32), source.as_fd())],
        )
        .unwrap();
    let result = worker.transfer(
        vec![span(tensor_region(1), 0, tensor_region(1), 0, 8)],
        Priority::Background,
        1,
    );
    assert_eq!(server_error(result), ErrorKind::Unavailable);
}

/// A head, a worker, and a source and a destination tensor of `bytes`
/// each, for transfers slow enough to observe: the host link is halved to
/// its floor.
struct Slow {
    head: Client,
    worker: Client,
    source: std::fs::File,
    destination: std::fs::File,
    bytes: u64,
}

fn slow(server: &Running, bytes: u64) -> Slow {
    let mut head = head(&server.socket);
    let claim = head.claim(0, vec![scope("text", 0)], Vec::new()).unwrap();
    let mut worker = worker(&server.socket, scope("text", 0), 1, claim);
    let source = memfd(&pattern(bytes as usize, 3));
    let destination = memfd(&vec![0; bytes as usize]);
    worker
        .put(
            vec![
                tensor(key(1), "weights", 0, 0, Access::ReadOnly),
                tensor(key(2), "cache", 1, 0, Access::Exclusive),
            ],
            &[
                (host(bytes), source.as_fd()),
                (host(bytes), destination.as_fd()),
            ],
        )
        .unwrap();
    let link = Link {
        source: Memory::Host,
        destination: Memory::Host,
    };
    for _ in 0..8 {
        head.throttle(vec![link.clone()], Throttle::Halve).unwrap();
    }
    Slow {
        head,
        worker,
        source,
        destination,
        bytes,
    }
}

#[test]
fn a_promotion_transfer_overtakes_background_transfers_on_its_link() {
    let server = serving(Gpus::Hidden);
    let Slow {
        mut worker,
        destination,
        bytes,
        ..
    } = slow(&server, 256 << 20);

    let background = worker
        .transfer(
            vec![span(tensor_region(1), 0, tensor_region(2), 0, bytes - 4096)],
            Priority::Background,
            1,
        )
        .unwrap();
    let promotion = worker
        .transfer(
            vec![span(
                tensor_region(1),
                0,
                tensor_region(2),
                bytes - 4096,
                4096,
            )],
            Priority::Promotion,
            1,
        )
        .unwrap();

    let mut ended = Vec::new();
    while ended.len() < 2 {
        if let Event::Transferred { task, end, .. } = transfer_event(&mut worker) {
            assert_eq!(end, TransferEnd::Complete);
            ended.push(task);
        }
    }
    assert_eq!(ended, [promotion, background]);
    assert_eq!(read(&destination, bytes - 4096, 4096), pattern(4096, 3));
}

#[test]
fn a_cancelled_transfer_ends_cancelled_and_stops_copying() {
    let server = serving(Gpus::Hidden);
    let Slow {
        mut head,
        mut worker,
        destination,
        bytes,
        ..
    } = slow(&server, 256 << 20);

    let task = worker
        .transfer(
            vec![span(tensor_region(1), 0, tensor_region(2), 0, bytes)],
            Priority::Warming,
            7,
        )
        .unwrap();
    // The task reports its progress while it copies.
    match transfer_event(&mut worker) {
        Event::TransferProgress {
            task: running,
            bytes: copied,
        } => assert!(running == task && copied < bytes),
        other => panic!("unexpected event {other:?}"),
    }
    // The head cancels the update's tasks, the worker's among them.
    assert_eq!(head.cancel(vec![], Some(7)).unwrap(), 1);
    let (copied, end) = end_of(&mut worker, task);
    assert_eq!(end, TransferEnd::Cancelled);
    assert!(copied < bytes);

    // Once it ended, no byte past what it reported is written.
    let tail = read(&destination, bytes - 4096, 4096);
    std::thread::sleep(Duration::from_millis(200));
    assert_eq!(read(&destination, bytes - 4096, 4096), tail);
    assert_eq!(tail, [0; 4096]);
    assert!(
        head.list()
            .unwrap()
            .transfer
            .is_some_and(|transfer| transfer.tasks.is_empty())
    );
}

#[test]
fn a_crashed_transfer_server_fails_its_tasks_and_the_server_starts_another() {
    let server = serving(Gpus::Hidden);
    let Slow {
        mut worker,
        source,
        destination,
        bytes,
        ..
    } = slow(&server, 256 << 20);
    let task = worker
        .transfer(
            vec![span(tensor_region(1), 0, tensor_region(2), 0, bytes)],
            Priority::Background,
            1,
        )
        .unwrap();
    assert!(matches!(
        transfer_event(&mut worker),
        Event::TransferProgress { .. }
    ));

    // The forks of the Transfer Server's restart hold no test's
    // connections.
    let _forking = FORKING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let crashed = transfer_pid(&server.socket, None);
    // SAFETY: SIGKILL to the Transfer Server this test's server started.
    assert_eq!(unsafe { libc::kill(crashed, libc::SIGKILL) }, 0);
    let (_, end) = end_of(&mut worker, task);
    assert!(
        matches!(&end, TransferEnd::Failed { message } if message.contains("Transfer Server")),
        "{end:?}"
    );
    // The submitter learns it at once, before the server starts another
    // Transfer Server (it waits seconds before a restart).
    let mut operator = Client::connect(&server.socket, Role::Operator).unwrap();
    assert!(operator.list().unwrap().transfer.is_none());

    // The server starts another, which copies again at a full budget.
    let restarted = transfer_pid(&server.socket, Some(crashed));
    assert_ne!(restarted, crashed);
    let task = worker
        .transfer(
            vec![span(tensor_region(1), 0, tensor_region(2), 0, bytes)],
            Priority::Background,
            2,
        )
        .unwrap();
    assert_eq!(end_of(&mut worker, task), (bytes, TransferEnd::Complete));
    assert_eq!(
        read(&destination, 0, bytes as usize),
        read(&source, 0, bytes as usize)
    );
}

/// An offload of a world's weights that the host cannot hold is refused
/// whole under the host rule: nothing moves, and the GPU's ledger still
/// books the weights.
#[test]
fn an_offload_the_host_cannot_hold_is_refused_and_moves_nothing() {
    use crate::server::{booked, device, ledger, world_worker};

    let server = serving(Gpus::Hidden);
    let mut head = head(&server.socket);
    let claim = head.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
    let mut first = world_worker(&server.socket, scope("text", 0), 1, claim);
    // A storage's record states its size; no host holds an exbibyte.
    let bytes = 1 << 60;
    let storage = memfd(&[1; 4]);
    first
        .put(
            vec![tensor(key(1), "weights", 0, 0, Access::ReadOnly)],
            &[(device(bytes, None), storage.as_fd())],
        )
        .unwrap();

    match head.move_weights(1, true) {
        Err(storage::Error::Server {
            kind: ErrorKind::Resource,
            message,
        }) => assert!(message.contains("the host can hold"), "{message}"),
        other => panic!("expected a refusal under the host rule, got {other:?}"),
    }
    assert_eq!(
        booked(&ledger(&mut head), storage::Item::Weights, &[1]),
        bytes
    );
}

/// GPU tests: device memory the test process allocates and retains at the
/// server, as a worker does.
pub(crate) mod device {
    use std::os::fd::{AsFd, OwnedFd};

    use std::time::{Duration, Instant};

    use cuda::{
        CopyBatch, DeviceAllocation, DeviceId, HandleType, Nvml, PinnedBuffer, ShareableHandle,
        SourceOrder, Stream,
    };
    use storage::Access;
    use storage::protocol::{
        Event, Link, Memory, Placement, Priority, Region, StorageRecord, Throttle, TransferEnd,
    };

    use super::{Gpus, end_of, pattern, read, serving, span, tensor_region};
    use crate::server::{head, host, key, memfd, scope, tensor, worker, world_worker};

    /// A retained device allocation of `bytes` on `device` holding
    /// `contents`, with its descriptor.
    pub(crate) fn device_storage(device: DeviceId, contents: &[u8]) -> (DeviceAllocation, OwnedFd) {
        let devices: Vec<DeviceId> = (0..DeviceId::count().unwrap()).map(DeviceId).collect();
        let allocation = DeviceAllocation::retainable(device, contents.len(), &devices).unwrap();
        let mut staging = PinnedBuffer::new(device, contents.len()).unwrap();
        // SAFETY: a pinned buffer this test owns, written before any copy.
        let host = unsafe { staging.as_mut_slice() };
        host[..contents.len()].copy_from_slice(contents);
        let stream = Stream::new(device).unwrap();
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        batch.push(allocation.address(), staging.address(), contents.len());
        // SAFETY: both ranges hold `contents.len()` bytes; the stream drains.
        unsafe { batch.submit(&stream) }.unwrap();
        stream.synchronize().unwrap();
        let descriptor = allocation.export_descriptor().unwrap().unwrap();
        (allocation, descriptor)
    }

    /// The bytes of `allocation`.
    pub(crate) fn device_bytes(allocation: &DeviceAllocation) -> Vec<u8> {
        let staging = PinnedBuffer::new(allocation.device(), allocation.len()).unwrap();
        let stream = Stream::new(allocation.device()).unwrap();
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        batch.push(staging.address(), allocation.address(), allocation.len());
        // SAFETY: both ranges hold the allocation's bytes; the stream drains.
        unsafe { batch.submit(&stream) }.unwrap();
        stream.synchronize().unwrap();
        // SAFETY: the copy into the buffer completed.
        let host = unsafe { staging.as_slice() };
        host[..allocation.len()].to_vec()
    }

    pub(crate) fn record(device: DeviceId, bytes: usize) -> StorageRecord {
        StorageRecord {
            placement: Placement::Device {
                gpu: device.uuid().unwrap().0,
            },
            bytes: bytes as u64,
            segment: None,
        }
    }

    /// The context bytes the Transfer Server of the server on `socket`
    /// measured on `gpu` when it started.
    fn measured_context(socket: &std::path::Path, gpu: [u8; 16]) -> u64 {
        let mut operator = storage::Client::connect(socket, storage::Role::Operator).unwrap();
        operator
            .list()
            .unwrap()
            .transfer
            .and_then(|transfer| {
                transfer
                    .contexts
                    .iter()
                    .find(|context| context.gpu == gpu)
                    .map(|context| context.bytes)
            })
            .expect("the Transfer Server measured its context on the GPU")
    }

    /// The Transfer Server's context as the deployment's ledger of `gpu`
    /// books it, if it does.
    fn booked_context(client: &mut storage::Client, gpu: [u8; 16]) -> Option<u64> {
        let (ledgers, _) = client.ledger(vec![gpu]).unwrap();
        ledgers
            .iter()
            .flat_map(|ledger| &ledger.entries)
            .find(|entry| entry.item == storage::Item::TransferContext)
            .map(|entry| entry.bytes)
    }

    /// Whether NVML lists the Transfer Server `pid` on `device`: whether it
    /// holds a context there.
    fn holds_context(nvml: &Nvml, device: DeviceId, pid: i32) -> bool {
        nvml.lists_process(device.uuid().unwrap(), pid as u32)
            .unwrap()
    }

    /// Wait, at most ten seconds, until the Transfer Server `pid` holds no
    /// context on `device`, and the ledger `client` reads books none.
    fn await_release(nvml: &Nvml, client: &mut storage::Client, device: DeviceId, pid: i32) {
        let gpu = device.uuid().unwrap().0;
        let deadline = Instant::now() + Duration::from_secs(10);
        while holds_context(nvml, device, pid) || booked_context(client, gpu).is_some() {
            assert!(
                Instant::now() < deadline,
                "the Transfer Server still holds or books its context on {device}"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// The Transfer Server holds a GPU's CUDA context only while a task
    /// needs the GPU: none once it started; while a task copies into the
    /// GPU, the context, booked in the deployment's ledger of the GPU at the
    /// bytes it measured; once the task ended, neither the context (NVML
    /// lists the process on the GPU no more) nor its booking. A following
    /// task on the GPU retains it again, and releases it in turn.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_gpu_holds_the_transfer_servers_context_only_while_a_task_needs_it() {
        let server = serving(Gpus::Visible);
        let pid = super::transfer_pid(&server.socket, None);
        let nvml = Nvml::init().unwrap();
        let first = DeviceId(0);
        let gpu = first.uuid().unwrap().0;
        let context = measured_context(&server.socket, gpu);
        assert!(context > 0);
        assert!(!holds_context(&nvml, first, pid));

        let mut head = head(&server.socket);
        let claim = head.claim(0, vec![scope("text", 0)], Vec::new()).unwrap();
        let mut worker = worker(&server.socket, scope("text", 0), 1, claim);
        let bytes = 1 << 30;
        let source_bytes = pattern(bytes, 17);
        let source = memfd(&source_bytes);
        let (destination, destination_fd) = device_storage(first, &vec![0; bytes]);
        worker
            .put(
                vec![
                    tensor(key(1), "offloaded", 0, 0, Access::ReadOnly),
                    tensor(key(2), "cache", 1, 0, Access::Exclusive),
                ],
                &[
                    (host(bytes as u64), source.as_fd()),
                    (record(first, bytes), destination_fd.as_fd()),
                ],
            )
            .unwrap();
        assert_eq!(booked_context(&mut head, gpu), None);

        // The host link into the GPU at its floor, so that the task lasts
        // long enough to observe.
        let link = Link {
            source: Memory::Host,
            destination: Memory::Device { gpu },
        };
        for _ in 0..8 {
            head.throttle(vec![link.clone()], Throttle::Halve).unwrap();
        }
        let task = worker
            .transfer(
                vec![span(tensor_region(1), 0, tensor_region(2), 0, bytes as u64)],
                Priority::Background,
                1,
            )
            .unwrap();
        loop {
            match super::transfer_event(&mut worker) {
                Event::TransferProgress { task: running, .. } if running == task => break,
                Event::Transferred { end, .. } => panic!("the task ended at once: {end:?}"),
                _ => {}
            }
        }
        assert_eq!(booked_context(&mut head, gpu), Some(context));
        assert!(holds_context(&nvml, first, pid));
        assert_eq!(
            end_of(&mut worker, task),
            (bytes as u64, TransferEnd::Complete)
        );
        await_release(&nvml, &mut head, first, pid);
        assert_eq!(device_bytes(&destination), source_bytes);

        // The next task retains the context again.
        let tail: u64 = 2 << 20;
        let task = worker
            .transfer(
                vec![span(tensor_region(1), 0, tensor_region(2), 0, tail)],
                Priority::Background,
                2,
            )
            .unwrap();
        assert_eq!(end_of(&mut worker, task), (tail, TransferEnd::Complete));
        await_release(&nvml, &mut head, first, pid);
    }

    /// A GPU whose memory ledger has no room for the Transfer Server's
    /// context refuses the context: a task that needs the GPU fails, naming
    /// the ledger, and the context is neither booked nor retained, so the
    /// ledger stays within its quota.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_ledger_without_room_for_the_context_fails_the_task_that_needs_the_gpu() {
        let server = serving(Gpus::Visible);
        let pid = super::transfer_pid(&server.socket, None);
        let nvml = Nvml::init().unwrap();
        let first = DeviceId(0);
        let gpu = first.uuid().unwrap().0;
        let context = measured_context(&server.socket, gpu);

        let mut head = head(&server.socket);
        let claim = head.claim(0, vec![scope("text", 0)], Vec::new()).unwrap();
        let mut worker = worker(&server.socket, scope("text", 0), 1, claim);
        let granularity = DeviceAllocation::granularity(first).unwrap();
        let source = memfd(&pattern(granularity, 5));
        let (destination, destination_fd) = device_storage(first, &vec![0; granularity]);
        worker
            .put(
                vec![
                    tensor(key(1), "offloaded", 0, 0, Access::ReadOnly),
                    tensor(key(2), "cache", 1, 0, Access::Exclusive),
                ],
                &[
                    (host(granularity as u64), source.as_fd()),
                    (record(first, granularity), destination_fd.as_fd()),
                ],
            )
            .unwrap();
        // The deployment's quota leaves room for half the context.
        let quota = granularity as u64 + 4096 + context / 2;
        let total = 1 << 40;
        worker
            .book(
                gpu,
                total,
                Some(quota),
                vec![(storage::Item::Process, 4096)],
            )
            .unwrap();

        let task = worker
            .transfer(
                vec![span(tensor_region(1), 0, tensor_region(2), 0, 4096)],
                Priority::Background,
                1,
            )
            .unwrap();
        match end_of(&mut worker, task) {
            (0, TransferEnd::Failed { message }) => assert!(
                message.contains("ledger") && message.contains("quota"),
                "{message}"
            ),
            other => panic!("a task without room for the context ended {other:?}"),
        }
        let (ledgers, _) = head.ledger(vec![gpu]).unwrap();
        let ledger = &ledgers[0];
        assert!(
            ledger
                .entries
                .iter()
                .all(|entry| entry.item != storage::Item::TransferContext)
        );
        assert!(ledger.used() <= quota);
        assert!(!holds_context(&nvml, first, pid));
        assert_eq!(device_bytes(&destination), vec![0; granularity]);
    }

    /// The weights booked on `gpu` in `head`'s ledger of it: each entry's
    /// worlds and bytes, sorted.
    fn weights_on(head: &mut storage::Client, gpu: [u8; 16]) -> Vec<(Vec<u64>, u64)> {
        let (ledgers, _) = head.ledger(vec![gpu]).unwrap();
        let mut weights: Vec<(Vec<u64>, u64)> = ledgers
            .iter()
            .flat_map(|ledger| &ledger.entries)
            .filter(|entry| entry.item == storage::Item::Weights)
            .map(|entry| (entry.worlds.clone(), entry.bytes))
            .collect();
        weights.sort();
        weights
    }

    /// The weights a world alone retains move to host memory and back
    /// through the Transfer Server, bytes intact: the offload leaves them in
    /// host memory and the GPU's ledger without them, while the weights its
    /// successor maps stay on the GPU; the restore returns them to the GPU
    /// they left and to its ledger.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_worlds_own_weights_move_to_host_memory_and_back() {
        let server = serving(Gpus::Visible);
        let first = DeviceId(0);
        let gpu = first.uuid().unwrap().0;
        let granularity = DeviceAllocation::granularity(first).unwrap();
        let size = granularity as u64;
        let own_bytes = pattern(granularity, 7);
        let shared_bytes = pattern(granularity, 13);

        let mut head = head(&server.socket);
        let claim = head.claim(1, vec![scope("text", 0)], Vec::new()).unwrap();
        let mut previous = world_worker(&server.socket, scope("text", 0), 1, claim);
        let (own, own_fd) = device_storage(first, &own_bytes);
        let (shared, shared_fd) = device_storage(first, &shared_bytes);
        previous
            .put(
                vec![
                    tensor(key(1), "own", 0, 0, Access::ReadOnly),
                    tensor(key(2), "shared", 1, 0, Access::ReadOnly),
                ],
                &[
                    (record(first, granularity), own_fd.as_fd()),
                    (record(first, granularity), shared_fd.as_fd()),
                ],
            )
            .unwrap();
        head.claim(2, vec![scope("text", 0)], Vec::new()).unwrap();
        let mut successor = world_worker(&server.socket, scope("text", 0), 2, claim);
        successor
            .get(vec![key(2)], storage::Mode::Read, Vec::new())
            .unwrap();
        // The previous world's processes exited: the server alone holds its
        // memory.
        drop(previous);
        drop((own, own_fd, shared, shared_fd));
        assert_eq!(
            weights_on(&mut head, gpu),
            [(vec![1], size), (vec![1, 2], size)]
        );

        // The bytes of world 1's own weights wherever they lie, copied into
        // host memory of the successor's: a copy retains nothing.
        let copy = memfd(&vec![0; granularity]);
        successor
            .put(
                vec![tensor(key(3), "copy", 0, 0, Access::Exclusive)],
                &[(host(size), copy.as_fd())],
            )
            .unwrap();
        let mut update = 0;
        let mut own_weights = |successor: &mut storage::Client| {
            update += 1;
            let task = successor
                .transfer(
                    vec![span(tensor_region(1), 0, tensor_region(3), 0, size)],
                    Priority::Background,
                    update,
                )
                .unwrap();
            assert_eq!(end_of(successor, task), (size, TransferEnd::Complete));
            read(&copy, 0, granularity)
        };
        let placement = |head: &mut storage::Client| {
            let listing = head.list().unwrap();
            let listed = listing.tensors.iter().find(|listed| listed.key == key(1));
            listed.unwrap().placement
        };

        assert_eq!(head.move_weights(1, true).unwrap(), (1, size));
        assert_eq!(weights_on(&mut head, gpu), [(vec![1, 2], size)]);
        assert!(matches!(placement(&mut head), Placement::Host { .. }));
        assert_eq!(own_weights(&mut successor), own_bytes);

        assert_eq!(head.move_weights(1, false).unwrap(), (1, size));
        assert_eq!(
            weights_on(&mut head, gpu),
            [(vec![1], size), (vec![1, 2], size)]
        );
        assert_eq!(placement(&mut head), Placement::Device { gpu });
        assert_eq!(own_weights(&mut successor), own_bytes);
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_transfer_copies_within_a_gpu_between_gpus_and_with_the_host() {
        let server = serving(Gpus::Visible);
        let mut head = head(&server.socket);
        let claim = head.claim(0, vec![scope("text", 0)], Vec::new()).unwrap();
        let mut worker = worker(&server.socket, scope("text", 0), 1, claim);

        let first = DeviceId(0);
        let granularity = DeviceAllocation::granularity(first).unwrap();
        let half = granularity / 2;
        let source_bytes = pattern(granularity, 11);
        let (_source, source_fd) = device_storage(first, &source_bytes);
        let (local, local_fd) = device_storage(first, &vec![0; granularity]);
        let host_source_bytes = pattern(half, 29);
        let host_source = memfd(&host_source_bytes);
        let host_destination = memfd(&vec![0; half]);
        worker
            .put(
                vec![
                    tensor(key(1), "weights", 0, 0, Access::ReadOnly),
                    tensor(key(2), "cache", 1, 0, Access::Exclusive),
                    tensor(key(3), "staged", 2, 0, Access::ReadOnly),
                    tensor(key(4), "offloaded", 3, 0, Access::Exclusive),
                ],
                &[
                    (record(first, granularity), source_fd.as_fd()),
                    (record(first, granularity), local_fd.as_fd()),
                    (host(half as u64), host_source.as_fd()),
                    (host(half as u64), host_destination.as_fd()),
                ],
            )
            .unwrap();
        let mut spans = vec![
            // Within the GPU, host to GPU, and GPU to host.
            span(
                tensor_region(1),
                0,
                tensor_region(2),
                half as u64,
                half as u64,
            ),
            span(tensor_region(3), 0, tensor_region(2), 0, half as u64),
            span(
                tensor_region(1),
                half as u64,
                tensor_region(4),
                0,
                half as u64,
            ),
        ];
        // Between two GPUs, when the process sees two.
        let peer = (DeviceId::count().unwrap() > 1).then(|| {
            let second = DeviceId(1);
            let (peer, peer_fd) = device_storage(second, &vec![0; granularity]);
            worker
                .put(
                    vec![tensor(key(5), "peer", 0, 0, Access::Exclusive)],
                    &[(record(second, granularity), peer_fd.as_fd())],
                )
                .unwrap();
            spans.push(span(
                tensor_region(1),
                0,
                tensor_region(5),
                0,
                granularity as u64,
            ));
            peer
        });
        let bytes: u64 = spans.iter().map(|span| span.bytes).sum();
        let task = worker.transfer(spans, Priority::Promotion, 1).unwrap();
        assert_eq!(end_of(&mut worker, task), (bytes, TransferEnd::Complete));

        let copied = device_bytes(&local);
        assert_eq!(&copied[..half], &host_source_bytes[..]);
        assert_eq!(&copied[half..], &source_bytes[..half]);
        assert_eq!(read(&host_destination, 0, half), &source_bytes[half..]);
        if let Some(peer) = &peer {
            assert_eq!(device_bytes(peer), source_bytes);
        }

        // The listing names every measured link and the device memory of
        // the Transfer Server's context on each GPU.
        let transfer = head.list().unwrap().transfer.unwrap();
        let gpu = |device: DeviceId| Memory::Device {
            gpu: device.uuid().unwrap().0,
        };
        let mut expected = vec![
            (gpu(first), gpu(first)),
            (Memory::Host, gpu(first)),
            (gpu(first), Memory::Host),
        ];
        if peer.is_some() {
            expected.push((gpu(first), gpu(DeviceId(1))));
        }
        for (source, destination) in expected {
            let link = Link {
                source,
                destination,
            };
            assert!(
                transfer
                    .links
                    .iter()
                    .any(|measured| measured.link == link && measured.bytes_per_second > 0),
                "{link:?} was not measured"
            );
        }
        for measured in &transfer.links {
            eprintln!(
                "{:?} -> {:?}: {:.1} GB/s",
                measured.link.source,
                measured.link.destination,
                measured.bytes_per_second as f64 / 1e9
            );
        }
        assert_eq!(transfer.contexts.len(), DeviceId::count().unwrap() as usize);
        for context in &transfer.contexts {
            eprintln!(
                "context on {}: {} MiB",
                cuda::DeviceUuid(context.gpu),
                context.bytes >> 20
            );
            assert!(context.bytes > 0);
        }
    }

    #[test]
    #[ignore = "requires a CUDA device whose allocations export fabric handles (IMEX)"]
    fn a_fabric_region_is_copied_by_the_handle_its_creator_exported() {
        let first = DeviceId(0);
        assert_eq!(HandleType::probe(first).unwrap(), HandleType::Fabric);
        let server = serving(Gpus::Visible);
        let mut head = head(&server.socket);
        let claim = head.claim(0, vec![scope("text", 0)], Vec::new()).unwrap();
        let mut worker = worker(&server.socket, scope("text", 0), 1, claim);

        // The destination lies on the last GPU this process sees, which
        // imports the handle: a GPU other than the source's when there are
        // two, as a GPU of another host of the fabric would.
        let last = DeviceId(DeviceId::count().unwrap() - 1);
        let granularity = DeviceAllocation::granularity(first).unwrap();
        let (destination, destination_fd) = device_storage(last, &vec![0; granularity]);
        worker
            .put(
                vec![tensor(key(2), "cache", 0, 0, Access::Exclusive)],
                &[(record(last, granularity), destination_fd.as_fd())],
            )
            .unwrap();

        // The source is memory this process created and holds, as the
        // creating worker of another host does; it names it by the fabric
        // handle it exports, which the server never holds.
        let source_bytes = pattern(granularity, 41);
        let (source, _) = device_storage(first, &source_bytes);
        let Some(ShareableHandle::Fabric(handle)) = source.export().unwrap() else {
            panic!("the allocation exports no fabric handle");
        };
        let region = Region::Fabric {
            handle: handle.to_vec(),
            gpu: first.uuid().unwrap().0,
            bytes: granularity as u64,
            offset: 0,
        };
        let half = granularity as u64 / 2;
        let task = worker
            .transfer(
                vec![
                    span(region.clone(), half, tensor_region(2), 0, half),
                    span(region, 0, tensor_region(2), half, half),
                ],
                Priority::Background,
                1,
            )
            .unwrap();
        assert_eq!(
            end_of(&mut worker, task),
            (granularity as u64, TransferEnd::Complete)
        );
        let copied = device_bytes(&destination);
        assert_eq!(&copied[..granularity / 2], &source_bytes[granularity / 2..]);
        assert_eq!(&copied[granularity / 2..], &source_bytes[..granularity / 2]);
    }
}
