//! A copy load for measuring how the Transfer Server's copies slow a
//! serving deployment's scheduler steps.
//!
//! The program is a second deployment on the serving deployment's Storage
//! Server (the one its daemon started, on the same socket). It claims its
//! own deployment as a head, connects as that deployment's worker under the
//! claim, allocates retainable CUDA VMM memory on GPU A and GPU B and a host
//! memfd on GPU A's NUMA node, publishes them as tensors, and then keeps the
//! Transfer Server copying into GPU A over three links, one at a time:
//!
//! | Link | Source → destination | Copy engines |
//! | --- | --- | --- |
//! | `local` | GPU A → GPU A | GPU A's |
//! | `host` | host memory → GPU A | GPU A's |
//! | `peer` | GPU B → GPU A | GPU A's, reading over the GPUs' link |
//!
//! For every link (in `--links` order) and every budget (in `--halvings`
//! order) it runs one point: at its start it halves the link's byte budget
//! `k` times (`Request::Throttle` with `Throttle::Halve`, so the budget is
//! `2^-k` of the link's measured bandwidth, down to the Transfer Server's
//! floor of 1/64), submits `--in-flight` tasks of `--task-mib` MiB each at
//! background priority and replaces each one that ends at once, for
//! `--copy-s` seconds. The link runs its tasks one after another; the others
//! wait queued, so the link never idles as long as the queued tasks outlast
//! the round trip that replaces a completed one (a 1 GiB task lasts about
//! 0.4 ms at a GB200's 2.8 TB/s within one GPU, hence 16 by default). It
//! then cancels the point's tasks, sends `Throttle::Restore` (the budget
//! grows back by 1/8 of the bandwidth per second, so it is whole again at
//! most 8 s later) and copies nothing for `--gap-s` seconds, which must
//! cover that restoration. The
//! first point starts at the wall-clock time `--start-at` (Unix seconds),
//! so an outside harness can align the points with its own measurement
//! phases; setup and a verification copy on every link run before then.
//!
//! Before it allocates anything it checks, from the server's listing, that
//! every GPU A ledger with a quota leaves room for the Transfer Server's
//! CUDA context, which the Transfer Server books in every ledger of a GPU
//! before it retains the context: a ledger without that room fails every
//! copy into GPU A. The verification copy then proves the room exists.
//!
//! It writes one JSON object per line to standard output, each with `t`
//! (wall-clock Unix seconds) and `event`:
//!
//! - `setup`: the GPUs' UUIDs, the links' measured bandwidths and the
//!   Transfer Server's context bytes, and every GPU A ledger.
//! - `verified`: a link's verification copy completed.
//! - `point_start`, then one `second` per whole second of copying with the
//!   bytes the point's tasks reported copied in it (from their
//!   `TransferProgress` and `Transferred` events), then `point_end` with
//!   the bytes copied in the copy window, its length and the bandwidth.
//! - `error`, and the program exits with status 1, on any failure.
//!
//! Run it with the serving deployment's runtime directory and both GPUs
//! visible, after the serving deployment is up (GPU ordinals are as this
//! process sees them):
//!
//! ```text
//! CUDA_VISIBLE_DEVICES=0,1 XDG_RUNTIME_DIR=/path/to/runtime \
//!   cargo run --release -p storage-server --example transfer_interference -- \
//!   --socket /path/to/runtime/uniserve/storage.sock --gpu-a 0 --gpu-b 1 \
//!   --links local,host,peer --halvings 0,2,4,6 --copy-s 30 --gap-s 15 \
//!   --start-at "$(date +%s --date='+60 seconds')"
//! ```

use std::collections::HashMap;
use std::io::Write as _;
use std::os::fd::{AsFd, OwnedFd};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, bail, ensure};
use cuda::{DeviceAllocation, DeviceAttribute, DeviceId};
use rustix::event::{PollFd, PollFlags, poll};
use storage::protocol::{Event, Link, Memory, Priority, Region, Span, Throttle, TransferEnd};
use storage::{
    Access, Client, GpuLedger, Key, Placement, Role, Scope, StorageRecord, TensorRecord,
};

/// The budget a restored link regains per second, as a share of its
/// bandwidth (the Transfer Server's `RESTORE_PER_SECOND`).
const RESTORE_PER_SECOND: f64 = 1.0 / 8.0;

/// The least share a halved link keeps (the Transfer Server's
/// `FLOOR_SHARE`).
const FLOOR_SHARE: f64 = 1.0 / 64.0;

/// The bytes of each link's verification copy.
const VERIFY_BYTES: u64 = 64 << 20;

/// How long the program waits for the Transfer Server to report its links,
/// and for a cancelled point's tasks to end.
const PATIENCE: Duration = Duration::from_secs(60);

/// The link kinds a point copies over, all into GPU A.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LinkKind {
    Local,
    Host,
    Peer,
}

impl LinkKind {
    fn parse(name: &str) -> anyhow::Result<Self> {
        match name {
            "local" => Ok(Self::Local),
            "host" => Ok(Self::Host),
            "peer" => Ok(Self::Peer),
            other => bail!("unknown link {other:?}: expected local, host or peer"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Host => "host",
            Self::Peer => "peer",
        }
    }
}

/// The command line.
struct Options {
    socket: PathBuf,
    deployment: String,
    gpu_a: DeviceId,
    gpu_b: DeviceId,
    links: Vec<LinkKind>,
    halvings: Vec<u32>,
    task_bytes: u64,
    in_flight: usize,
    copy: Duration,
    gap: Duration,
    start_at: f64,
}

impl Options {
    fn parse() -> anyhow::Result<Self> {
        let mut values: HashMap<String, String> = HashMap::new();
        let mut arguments = std::env::args().skip(1);
        while let Some(flag) = arguments.next() {
            let name = flag
                .strip_prefix("--")
                .with_context(|| format!("unexpected argument {flag:?}"))?;
            let value = arguments
                .next()
                .with_context(|| format!("{flag} needs a value"))?;
            values.insert(name.to_owned(), value);
        }
        let mut take = |name: &str, default: Option<&str>| -> anyhow::Result<String> {
            values
                .remove(name)
                .or_else(|| default.map(str::to_owned))
                .with_context(|| format!("--{name} is required"))
        };

        let options = Self {
            socket: take("socket", None)?.into(),
            deployment: take("deployment", Some("transfer-interference"))?,
            gpu_a: DeviceId(take("gpu-a", Some("0"))?.parse()?),
            gpu_b: DeviceId(take("gpu-b", Some("1"))?.parse()?),
            links: take("links", Some("local,host,peer"))?
                .split(',')
                .map(LinkKind::parse)
                .collect::<anyhow::Result<_>>()?,
            halvings: take("halvings", Some("0,2,4,6"))?
                .split(',')
                .map(str::parse)
                .collect::<Result<_, _>>()?,
            task_bytes: take("task-mib", Some("1024"))?.parse::<u64>()? << 20,
            in_flight: take("in-flight", Some("16"))?.parse()?,
            copy: Duration::from_secs_f64(take("copy-s", Some("30"))?.parse()?),
            gap: Duration::from_secs_f64(take("gap-s", Some("15"))?.parse()?),
            start_at: take("start-at", None)?.parse()?,
        };
        if let Some(unknown) = values.keys().next() {
            bail!("unknown option --{unknown}");
        }
        ensure!(
            options.gpu_a != options.gpu_b,
            "GPU A and GPU B must differ"
        );
        ensure!(
            options.in_flight >= 2,
            "--in-flight must keep at least two tasks"
        );
        // The longest restoration, from the floor, must end within a gap so
        // that every point starts from the whole budget.
        let restoration = (1.0 - FLOOR_SHARE) / RESTORE_PER_SECOND;
        ensure!(
            options.gap.as_secs_f64() >= restoration,
            "--gap-s must cover the budget's restoration, {restoration} s"
        );
        Ok(options)
    }
}

/// The current wall-clock time in Unix seconds.
fn wall() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

/// `text` as a JSON string literal.
fn quoted(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            character if (character as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => out.push(character),
        }
    }
    out.push('"');
    out
}

/// Print one JSON line: `event` and the already-encoded `fields`, stamped
/// with the wall-clock time.
fn emit(event: &str, fields: &[(&str, String)]) {
    let mut line = format!("{{\"t\":{:.6},\"event\":{}", wall(), quoted(event));
    for (name, value) in fields {
        line.push_str(&format!(",{}:{value}", quoted(name)));
    }
    line.push('}');
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{line}");
    let _ = stdout.flush();
}

fn uuid_hex(gpu: &[u8; 16]) -> String {
    gpu.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// A key of 32 bytes spelling `name`, distinct from content keys.
fn key(name: &str) -> Key {
    let mut bytes = [0_u8; 32];
    for (slot, byte) in bytes.iter_mut().zip(name.bytes()) {
        *slot = byte;
    }
    Key(bytes)
}

/// A tensor of `bytes` bytes over the whole storage `storage`.
fn tensor(name: &str, storage: u32, bytes: u64, access: Access) -> TensorRecord {
    TensorRecord {
        key: key(name),
        name: name.to_owned(),
        storage,
        offset: 0,
        dtype: "uint8".into(),
        shape: vec![bytes],
        strides: vec![1],
        access,
    }
}

/// A host memfd of `bytes` whose pages prefer NUMA node `node`, touched so
/// that they are allocated there before any copy registers them.
fn host_memfd(bytes: u64, node: Option<u32>) -> anyhow::Result<OwnedFd> {
    use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};

    let descriptor =
        rustix::fs::memfd_create(c"transfer-interference", rustix::fs::MemfdFlags::CLOEXEC)?;
    rustix::fs::ftruncate(&descriptor, bytes)?;
    let length = usize::try_from(bytes)?;
    // SAFETY: a new shared mapping of the memfd created above, which this
    // function alone writes and then unmaps.
    let address = unsafe {
        mmap(
            std::ptr::null_mut(),
            length,
            ProtFlags::READ | ProtFlags::WRITE,
            MapFlags::SHARED,
            &descriptor,
            0,
        )
    }?;
    if let Some(node) = node {
        // A node mask of 1024 bits, as the Storage Server's page pools bind
        // theirs; MPOL_PREFERRED falls back to other nodes when the node is
        // full.
        const MASK_BITS: u32 = 1024;
        ensure!(node < MASK_BITS, "NUMA node {node} is beyond the mask");
        let mut mask = [0_u64; (MASK_BITS / 64) as usize];
        mask[(node / 64) as usize] |= 1 << (node % 64);
        // SAFETY: mbind of the mapping made above with a mask of
        // `MASK_BITS` bits.
        let bound = unsafe {
            libc::syscall(
                libc::SYS_mbind,
                address,
                length,
                libc::MPOL_PREFERRED,
                mask.as_ptr(),
                u64::from(MASK_BITS) + 1,
                0_u32,
            )
        };
        ensure!(
            bound == 0,
            "mbind to NUMA node {node}: {}",
            std::io::Error::last_os_error()
        );
    }
    // SAFETY: the mapping holds `length` writable bytes; touching each one
    // allocates the pages under the policy bound above.
    unsafe { std::ptr::write_bytes(address.cast::<u8>(), 0x5a, length) };
    // SAFETY: the mapping made above, which nothing references any more.
    unsafe { munmap(address, length) }?;
    Ok(descriptor)
}

/// The Storage Server's listing once its Transfer Server reported its
/// links.
fn measured_listing(socket: &std::path::Path) -> anyhow::Result<storage::Listing> {
    let mut operator = Client::connect(socket, Role::Operator)?;
    let deadline = Instant::now() + PATIENCE;
    loop {
        let listing = operator.list()?;
        if listing
            .transfer
            .as_ref()
            .is_some_and(|transfer| !transfer.links.is_empty())
        {
            return Ok(listing);
        }
        ensure!(
            Instant::now() < deadline,
            "no Transfer Server reported its links"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// A ledger as a JSON object: its deployment, quota, booked bytes and the
/// room it leaves.
fn ledger_json(ledger: &GpuLedger) -> String {
    let used = ledger.used();
    let quota = ledger
        .quota
        .map_or_else(|| "null".to_owned(), |quota| quota.to_string());
    let room = ledger.quota.map_or_else(
        || "null".to_owned(),
        |quota| (quota as i128 - used as i128).to_string(),
    );
    let entries: Vec<String> = ledger
        .entries
        .iter()
        .map(|entry| format!("[{},{}]", quoted(&format!("{:?}", entry.item)), entry.bytes))
        .collect();
    format!(
        "{{\"deployment\":{},\"quota\":{quota},\"used\":{used},\"room\":{room},\"transfer_context\":{},\"entries\":[{}]}}",
        quoted(&ledger.deployment),
        ledger.transfer_context,
        entries.join(",")
    )
}

/// One link a point copies over: the source tensor's key, and the link the
/// Transfer Server budgets.
struct Route {
    kind: LinkKind,
    source: Key,
    link: Link,
}

/// The copy deployment's connections and the memory it published.
struct Copier {
    head: Client,
    worker: Client,
    destination: Key,
    task_bytes: u64,
    /// The allocations stay mapped in this process while the server holds
    /// their descriptors.
    _allocations: Vec<DeviceAllocation>,
    _host: OwnedFd,
}

/// The bytes a point's tasks reported, and the tasks still running.
#[derive(Default)]
struct Progress {
    /// Bytes each running task reported so far.
    running: HashMap<u64, u64>,
    /// Bytes every task of the point reported, in total.
    bytes: u64,
    completed: u64,
}

impl Copier {
    /// Submit one task of `bytes` from `route`'s source into GPU A.
    fn submit(&mut self, route: &Route, bytes: u64, update: u64) -> anyhow::Result<u64> {
        let span = Span {
            // Each tensor the program retains has one copy.
            source: Region::Tensor {
                key: route.source,
                at: None,
            },
            source_offset: 0,
            destination: Region::Tensor {
                key: self.destination,
                at: None,
            },
            destination_offset: 0,
            // One row of the whole tensor: the pitches go unused.
            bytes,
            rows: 1,
            source_pitch: 0,
            destination_pitch: 0,
        };
        Ok(self
            .worker
            .transfer(vec![span], Priority::Background, update)?)
    }

    /// Take the transfer events received so far into `progress`. Returns
    /// the tasks that completed, and fails on a task that failed, or that
    /// was cancelled unless `cancelling`.
    fn drain(&mut self, progress: &mut Progress, cancelling: bool) -> anyhow::Result<u64> {
        self.worker.pump()?;
        let mut completed = 0;
        while let Some((event, _)) = self.worker.next_event() {
            match event {
                Event::TransferProgress { task, bytes } => {
                    if let Some(seen) = progress.running.get_mut(&task) {
                        progress.bytes += bytes.saturating_sub(*seen);
                        *seen = (*seen).max(bytes);
                    }
                }
                Event::Transferred { task, bytes, end } => {
                    let Some(seen) = progress.running.remove(&task) else {
                        continue;
                    };
                    progress.bytes += bytes.saturating_sub(seen);
                    match end {
                        TransferEnd::Complete => completed += 1,
                        TransferEnd::Cancelled if cancelling => {}
                        TransferEnd::Cancelled => bail!("task {task} was cancelled"),
                        TransferEnd::Failed { message } => {
                            bail!("task {task} failed: {message}")
                        }
                    }
                }
                _ => {}
            }
        }
        progress.completed += completed;
        Ok(completed)
    }

    /// Wait until the worker's connection is readable or `until` passes.
    fn wait(&self, until: Instant) -> anyhow::Result<()> {
        let timeout = until.saturating_duration_since(Instant::now());
        let timeout = rustix::time::Timespec::try_from(timeout)?;
        let connection = self.worker.as_fd();
        let mut fds = [PollFd::new(&connection, PollFlags::IN)];
        match poll(&mut fds, Some(&timeout)) {
            Ok(_) | Err(rustix::io::Errno::INTR) => Ok(()),
            Err(error) => Err(error.into()),
        }
    }

    /// Copy `VERIFY_BYTES` over `route` once and require it to complete.
    fn verify(&mut self, route: &Route, update: u64) -> anyhow::Result<()> {
        let started = Instant::now();
        let task = self.submit(route, VERIFY_BYTES, update)?;
        let mut progress = Progress::default();
        progress.running.insert(task, 0);
        let deadline = started + PATIENCE;
        while !progress.running.is_empty() {
            ensure!(
                Instant::now() < deadline,
                "the verification copy over {} did not end",
                route.kind.name()
            );
            self.wait(deadline)?;
            self.drain(&mut progress, false)?;
        }
        emit(
            "verified",
            &[
                ("link", quoted(route.kind.name())),
                ("bytes", progress.bytes.to_string()),
                ("seconds", format!("{:.6}", started.elapsed().as_secs_f64())),
            ],
        );
        Ok(())
    }

    /// Run one point from `start`: budget `2^-halvings` on `route`'s link,
    /// copies for `copy`, then cancel and restore.
    fn point(
        &mut self,
        route: &Route,
        halvings: u32,
        start: Instant,
        copy: Duration,
        in_flight: usize,
        update: u64,
    ) -> anyhow::Result<()> {
        std::thread::sleep(start.saturating_duration_since(Instant::now()));
        for _ in 0..halvings {
            self.head
                .throttle(vec![route.link.clone()], Throttle::Halve)?;
        }
        let share = 0.5_f64.powi(halvings as i32).max(FLOOR_SHARE);
        let started = Instant::now();
        emit(
            "point_start",
            &[
                ("link", quoted(route.kind.name())),
                ("halvings", halvings.to_string()),
                ("share", format!("{share}")),
                ("update", update.to_string()),
            ],
        );

        let mut progress = Progress::default();
        for _ in 0..in_flight {
            let task = self.submit(route, self.task_bytes, update)?;
            progress.running.insert(task, 0);
        }

        // Copy until the window closes, replacing every completed task at
        // once and reporting each whole second's bytes.
        let deadline = started + copy;
        let mut second = 0_u64;
        let mut mark = started + Duration::from_secs(1);
        let mut at_mark = 0_u64;
        loop {
            let now = Instant::now();
            if now >= mark && mark <= deadline {
                emit(
                    "second",
                    &[
                        ("link", quoted(route.kind.name())),
                        ("halvings", halvings.to_string()),
                        ("second", second.to_string()),
                        ("bytes", (progress.bytes - at_mark).to_string()),
                    ],
                );
                at_mark = progress.bytes;
                second += 1;
                mark += Duration::from_secs(1);
            }
            if now >= deadline {
                break;
            }
            self.wait(mark.min(deadline))?;
            let completed = self.drain(&mut progress, false)?;
            for _ in 0..completed {
                let task = self.submit(route, self.task_bytes, update)?;
                progress.running.insert(task, 0);
            }
        }
        let window = started.elapsed().as_secs_f64();
        let copied = progress.bytes;

        // The point's tasks end once their chunks in flight completed; the
        // bytes they copy after the window are reported apart.
        let cancelled_at = Instant::now();
        self.worker.cancel(Vec::new(), Some(update))?;
        let patience = cancelled_at + PATIENCE;
        while !progress.running.is_empty() {
            ensure!(
                Instant::now() < patience,
                "the cancelled tasks of {} did not end",
                route.kind.name()
            );
            self.wait(patience)?;
            self.drain(&mut progress, true)?;
        }
        let drained = cancelled_at.elapsed().as_secs_f64();
        if halvings > 0 {
            self.head
                .throttle(vec![route.link.clone()], Throttle::Restore)?;
        }
        emit(
            "point_end",
            &[
                ("link", quoted(route.kind.name())),
                ("halvings", halvings.to_string()),
                ("share", format!("{share}")),
                ("bytes", copied.to_string()),
                ("window_s", format!("{window:.6}")),
                ("gbps", format!("{:.3}", copied as f64 / window / 1e9)),
                ("tasks_completed", progress.completed.to_string()),
                ("tail_bytes", (progress.bytes - copied).to_string()),
                ("cancel_s", format!("{drained:.6}")),
            ],
        );
        Ok(())
    }
}

fn run(options: &Options) -> anyhow::Result<()> {
    let count = DeviceId::count()?;
    ensure!(
        options.gpu_a.0 < count && options.gpu_b.0 < count,
        "this process sees {count} GPUs"
    );
    let gpu_a = options.gpu_a.uuid()?.0;
    let gpu_b = options.gpu_b.uuid()?.0;
    let numa_node = options
        .gpu_a
        .attribute(DeviceAttribute::HostNumaId)
        .ok()
        .and_then(|node| u32::try_from(node).ok());

    let device = |gpu: [u8; 16]| Memory::Device { gpu };
    let routes = [
        (LinkKind::Local, "source-a", device(gpu_a)),
        (LinkKind::Host, "source-host", Memory::Host),
        (LinkKind::Peer, "source-b", device(gpu_b)),
    ]
    .map(|(kind, source, from)| Route {
        kind,
        source: key(source),
        link: Link {
            source: from,
            destination: device(gpu_a),
        },
    });
    // `routes` lists the kinds in their declaration order.
    let route = |kind: LinkKind| &routes[kind as usize];

    // What the Transfer Server measured, and whether every GPU A ledger
    // with a quota leaves room for its context there.
    let listing = measured_listing(&options.socket)?;
    let transfer = listing
        .transfer
        .as_ref()
        .context("the Transfer Server left before the listing")?;
    let bandwidths: Vec<String> = routes
        .iter()
        .map(|route| {
            let bandwidth = transfer
                .links
                .iter()
                .find(|measured| measured.link == route.link)
                .map_or(0, |measured| measured.bytes_per_second);
            format!("{}:{bandwidth}", quoted(route.kind.name()))
        })
        .collect();
    let context = |gpu: [u8; 16]| {
        transfer
            .contexts
            .iter()
            .find(|context| context.gpu == gpu)
            .map_or(0, |context| context.bytes)
    };
    let ledgers: Vec<&GpuLedger> = listing
        .ledgers
        .iter()
        .filter(|ledger| ledger.gpu == gpu_a)
        .collect();
    emit(
        "setup",
        &[
            ("gpu_a", quoted(&uuid_hex(&gpu_a))),
            ("gpu_b", quoted(&uuid_hex(&gpu_b))),
            (
                "numa_node",
                numa_node.map_or_else(|| "null".to_owned(), |node| node.to_string()),
            ),
            ("task_bytes", options.task_bytes.to_string()),
            ("in_flight", options.in_flight.to_string()),
            ("bandwidths", format!("{{{}}}", bandwidths.join(","))),
            ("context_a", context(gpu_a).to_string()),
            ("context_b", context(gpu_b).to_string()),
            (
                "ledgers_a",
                format!(
                    "[{}]",
                    ledgers
                        .iter()
                        .map(|ledger| ledger_json(ledger))
                        .collect::<Vec<_>>()
                        .join(",")
                ),
            ),
        ],
    );
    for ledger in &ledgers {
        if let Some(quota) = ledger.quota {
            ensure!(
                ledger.used() + context(gpu_a) <= quota,
                "the ledger of deployment {} on GPU A books {} of its quota {}: no room for \
                 the Transfer Server's context of {} bytes",
                ledger.deployment,
                ledger.used(),
                quota,
                context(gpu_a)
            );
        }
    }

    // The copy deployment: its head, and its one worker under the claim.
    let scope = Scope {
        deployment: options.deployment.clone(),
        group: "copies".into(),
        rank: 0,
    };
    let mut head = Client::connect(
        &options.socket,
        Role::Head {
            deployment: options.deployment.clone(),
            world: 0,
        },
    )?;
    let claim = head.claim(0, vec![scope.clone()], Vec::new())?;
    let mut worker = Client::connect(
        &options.socket,
        Role::Worker {
            scope,
            generation: 1,
            claim,
            world: 0,
        },
    )?;

    // Every source and the destination are one task's bytes: each task
    // copies a whole source over the whole destination.
    let bytes = usize::try_from(options.task_bytes)?;
    let access = [options.gpu_a, options.gpu_b];
    let source_a = DeviceAllocation::retainable(options.gpu_a, bytes, &access)?;
    let destination_a = DeviceAllocation::retainable(options.gpu_a, bytes, &access)?;
    let source_b = DeviceAllocation::retainable(options.gpu_b, bytes, &access)?;
    let host = host_memfd(options.task_bytes, numa_node)?;
    let descriptor = |allocation: &DeviceAllocation| -> anyhow::Result<OwnedFd> {
        allocation
            .export_descriptor()?
            .context("a retainable allocation exports a descriptor")
    };
    let on_device = |gpu: [u8; 16], allocation: &DeviceAllocation| StorageRecord {
        placement: Placement::Device { gpu },
        bytes: allocation.len() as u64,
        segment: None,
        fabric: None,
    };
    let source_a_fd = descriptor(&source_a)?;
    let destination_a_fd = descriptor(&destination_a)?;
    let source_b_fd = descriptor(&source_b)?;
    let task_bytes = options.task_bytes;
    worker.put(
        vec![
            tensor("source-a", 0, task_bytes, Access::ReadOnly),
            tensor("destination-a", 1, task_bytes, Access::Exclusive),
            tensor("source-b", 2, task_bytes, Access::ReadOnly),
            tensor("source-host", 3, task_bytes, Access::ReadOnly),
        ],
        &[
            (on_device(gpu_a, &source_a), source_a_fd.as_fd()),
            (on_device(gpu_a, &destination_a), destination_a_fd.as_fd()),
            (on_device(gpu_b, &source_b), source_b_fd.as_fd()),
            (
                StorageRecord {
                    placement: Placement::Host { numa_node },
                    bytes: task_bytes,
                    segment: None,
                    fabric: None,
                },
                host.as_fd(),
            ),
        ],
    )?;

    let mut copier = Copier {
        head,
        worker,
        destination: key("destination-a"),
        task_bytes,
        _allocations: vec![source_a, destination_a, source_b],
        _host: host,
    };

    // Each link copies once before the measured points.
    for (index, kind) in options.links.iter().enumerate() {
        copier.verify(route(*kind), 1000 + index as u64)?;
    }

    // The points, from `--start-at` on the wall clock.
    let lead = options.start_at - wall();
    ensure!(
        lead > 0.0,
        "--start-at passed {:.3} s before the copies were ready",
        -lead
    );
    let first = Instant::now() + Duration::from_secs_f64(lead);
    let period = options.copy + options.gap;
    let mut index = 0_u32;
    for kind in &options.links {
        for &halvings in &options.halvings {
            let start = first + period * index;
            copier.point(
                route(*kind),
                halvings,
                start,
                options.copy,
                options.in_flight,
                u64::from(index) + 1,
            )?;
            index += 1;
        }
    }
    // The last gap passes before the program ends, so the restoration of
    // the last point's link completes while it still runs.
    std::thread::sleep((first + period * index).saturating_duration_since(Instant::now()));
    emit("done", &[("points", index.to_string())]);
    Ok(())
}

fn main() {
    let result = Options::parse().and_then(|options| run(&options));
    if let Err(error) = result {
        emit("error", &[("message", quoted(&format!("{error:#}")))]);
        std::process::exit(1);
    }
}
