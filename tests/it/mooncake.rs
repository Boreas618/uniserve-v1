//! The Mooncake connector against a running Mooncake master: a page one
//! Storage Server commits is uploaded by its connector, and another Storage
//! Server, whose pool never held it, finds it in Mooncake and prefetches it
//! back byte for byte.
//!
//! Ignored by default: it needs `UNISERVE_MOONCAKE_MASTER` to name a running
//! master (`host:port`) and `MOONCAKE_STORE_LIBRARY` to name
//! `libmooncake_store.so`, and fails when requested without them. Run it with
//! `cargo test -p uniserve --test it mooncake -- --ignored`. The connectors use
//! Mooncake's P2P handshake for metadata and
//! TCP for transfers. The first server contributes storage to Mooncake's pool
//! and the second contributes none, so the second is a client of the
//! first's segment.

#![cfg(test)]

use std::os::fd::{AsFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use storage::protocol::{Event, PageLevel, PageShape, PageSlot, Request, Response};
use storage::{Client, Key, Role, Scope};

const PAGE_BYTES: u64 = 4096;

/// A `uniserve storage-server serve` process on a socket of its own, with
/// its Mooncake connector, stopped when dropped.
struct Serving {
    child: Child,
    socket: PathBuf,
    _directory: tempfile::TempDir,
}

impl Serving {
    fn start(master: &str, segment_bytes: u64) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let socket = directory.path().join("storage.sock");
        let child = Command::new(env!("CARGO_BIN_EXE_uniserve"))
            .args(["storage-server", "serve"])
            .arg("--socket")
            .arg(&socket)
            .args(["--page-pool-gib", "1", "--mooncake-master", master])
            .args(["--mooncake-hostname", "127.0.0.1"])
            .args(["--mooncake-segment-bytes", &segment_bytes.to_string()])
            .spawn()
            .unwrap();
        let serving = Self {
            child,
            socket,
            _directory: directory,
        };
        wait_for("the server to listen", || service::serving(&serving.socket));
        // Lookups count L3 only while the connector is connected.
        wait_for("the connector to connect", || {
            operator(&serving.socket).list().unwrap().connector
        });
        serving
    }
}

impl Drop for Serving {
    fn drop(&mut self) {
        // SAFETY: a signal to the child this test started.
        unsafe { libc::kill(self.child.id() as i32, libc::SIGTERM) };
        let _ = self.child.wait();
    }
}

fn wait_for(what: &str, mut ready: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(60);
    while !ready() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(50));
    }
}

fn operator(socket: &Path) -> Client {
    Client::connect(socket, Role::Operator).unwrap()
}

fn scope() -> Scope {
    Scope {
        deployment: "model".into(),
        group: "text".into(),
        rank: 0,
    }
}

/// A head that claims the deployment, and a worker of its scope with the
/// pool of `shape` mapped.
fn claim_pool(socket: &Path, shape: PageShape) -> (Client, Client, u64, Mapping) {
    let mut head = Client::connect(
        socket,
        Role::Head {
            deployment: "model".into(),
            epoch: 0,
        },
    )
    .unwrap();
    let claim = head.claim(0, vec![scope()], Vec::new()).unwrap();
    head.publish(vec![0]).unwrap();
    let mut worker = Client::connect(
        socket,
        Role::Worker {
            scope: scope(),
            generation: 1,
            claim,
            epoch: 0,
        },
    )
    .unwrap();
    let (response, descriptors) = worker
        .call(
            &Request::Pool {
                shape,
                numa_node: None,
                bytes: None,
            },
            &[],
        )
        .unwrap();
    let Response::Pool { pool, slots, .. } = response else {
        panic!("the page pool was refused: {response:?}");
    };
    let storage = descriptors.into_iter().next().unwrap();
    let mapping = Mapping::new(&storage, (slots * PAGE_BYTES) as usize);
    (head, worker, pool, mapping)
}

/// A shared mapping of a page pool, as a worker makes one.
struct Mapping {
    address: *mut u8,
    bytes: usize,
}

impl Mapping {
    fn new(storage: &OwnedFd, bytes: usize) -> Self {
        // SAFETY: a new shared mapping of the pool's whole memfd.
        let address = unsafe {
            rustix::mm::mmap(
                std::ptr::null_mut(),
                bytes,
                rustix::mm::ProtFlags::READ | rustix::mm::ProtFlags::WRITE,
                rustix::mm::MapFlags::SHARED,
                storage.as_fd(),
                0,
            )
        }
        .unwrap();
        Self {
            address: address.cast(),
            bytes,
        }
    }

    fn slot(&mut self, slot: u64) -> &mut [u8] {
        let start = (slot * PAGE_BYTES) as usize;
        assert!(start + PAGE_BYTES as usize <= self.bytes);
        // SAFETY: one slot inside the mapping, which this test alone writes.
        unsafe { std::slice::from_raw_parts_mut(self.address.add(start), PAGE_BYTES as usize) }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: the mapping `new` made.
        let _ = unsafe { rustix::mm::munmap(self.address.cast(), self.bytes) };
    }
}

fn reply(client: &mut Client, request: Request) -> Response {
    client.call(&request, &[]).unwrap().0
}

#[test]
#[ignore = "requires a Mooncake master (UNISERVE_MOONCAKE_MASTER) and libmooncake_store.so (MOONCAKE_STORE_LIBRARY)"]
fn a_page_one_server_commits_is_prefetched_by_another_through_mooncake() {
    let master = std::env::var("UNISERVE_MOONCAKE_MASTER")
        .expect("UNISERVE_MOONCAKE_MASTER must name a running Mooncake master as host:port");
    // The Storage Servers this test starts load the library this variable names.
    std::env::var_os("MOONCAKE_STORE_LIBRARY")
        .expect("MOONCAKE_STORE_LIBRARY must name libmooncake_store.so");
    // Every run writes pages of its own, which earlier runs' objects in
    // the master cannot answer for.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let mut content = [0_u8; 32];
    content[..16].copy_from_slice(&nonce.to_le_bytes());
    let shape = PageShape {
        layout: Key(content),
        kv_head_offset: 0,
        kv_head_count: 8,
        page_bytes: PAGE_BYTES,
    };
    let key = Key([content[0] ^ 0x5a; 32]);
    let page: Vec<u8> = (0..PAGE_BYTES).map(|byte| (byte * 7 % 251) as u8).collect();

    // The first server commits the page; its connector uploads it.
    let first = Serving::start(&master, 256 << 20);
    let (mut head, _worker, pool, mut mapping) = claim_pool(&first.socket, shape);
    let Response::Reserved { slots } = reply(
        &mut head,
        Request::Reserve {
            pool,
            writer: Some(scope()),
            keys: vec![key],
            window: None,
        },
    ) else {
        panic!("the reservation was refused");
    };
    let [PageSlot::Reserved { slot }] = slots[..] else {
        panic!("the reservation answered {slots:?}");
    };
    mapping.slot(slot).copy_from_slice(&page);
    reply(
        &mut head,
        Request::Commit {
            pool,
            keys: vec![key],
        },
    );
    wait_for("the upload", || {
        let listing = operator(&first.socket).list().unwrap();
        listing
            .page_pools
            .iter()
            .all(|listed| listed.uploading == 0)
    });

    // The second server's pool never held the page: Mooncake does, and a
    // prefetch reads it into the pool.
    let second = Serving::start(&master, 0);
    let (mut head, _worker, pool, mut mapping) = claim_pool(&second.socket, shape);
    let missing = Key([0xee; 32]);
    assert_eq!(
        reply(
            &mut head,
            Request::Lookup {
                pool,
                keys: vec![key, missing],
            },
        ),
        Response::Lookup {
            levels: vec![PageLevel::Storage, PageLevel::Absent]
        }
    );
    assert_eq!(
        reply(
            &mut head,
            Request::Prefetch {
                pool,
                keys: vec![key, missing],
                tag: 1,
            },
        ),
        Response::Prefetching
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    let results = loop {
        head.pump().unwrap();
        if let Some((Event::Prefetched { tag: 1, results }, _)) = head.next_event() {
            break results;
        }
        assert!(Instant::now() < deadline, "the prefetch did not end");
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(results, vec![true, false]);
    let Response::Pinned { slots } = reply(
        &mut head,
        Request::Pin {
            pool,
            reader: scope(),
            keys: vec![key],
        },
    ) else {
        panic!("the pin was refused");
    };
    let [Some(slot)] = slots[..] else {
        panic!("the prefetched page is not in the pool: {slots:?}");
    };
    assert_eq!(mapping.slot(slot), &page[..]);
}
