//! Reads from another host through the hosts' Transfer Servers, for
//! checking the path between hosts and measuring its links.
//!
//! Each host runs a Storage Server that listens for other hosts on an
//! address they reach (`uniserve storage serve --tcp ADDRESS:0
//! --nixl-library PATH`), and this program against it, in one of two roles
//! of one deployment (`--deployment`, the same on both hosts):
//!
//! - `hold`, on the source host: claims the deployment as a head, connects
//!   as its worker, allocates `--mib` MiB of retainable device storage on
//!   `--gpu` and as much host storage on that GPU's NUMA node, fills each
//!   with a sequence of its own seed, publishes them as the tensors
//!   `remote-device` (read-only) and `remote-host` (read-only, or writable
//!   with `--host-access exclusive`), prints `holding` with its
//!   server's endpoint, and keeps them until the file `--release` exists
//!   (a path both hosts see, so a harness on the other host ends it). The
//!   device source's record carries the fabric handle the holder exports
//!   (`--record-fabric yes`, the default), so a reader whose GPU imports it
//!   copies that source over the NVLink fabric rather than READ it;
//!   `--record-fabric no` leaves every read of it to a READ.
//! - `read`, on the destination host (giving up on a read after
//!   `--patience-s` seconds): claims the deployment there,
//!   allocates `--mib` MiB of device storage on `--gpu`, and for each
//!   source of `--links` (`device`, `host`) reads that tensor whole from
//!   the server at `--source` (the holder's endpoint) into it `--rounds`
//!   times, one task at a time. The `fabric` link reads the device source
//!   by the fabric handle and GPU `holding` printed (`--fabric HEX
//!   --fabric-gpu UUID`), a region the reader's Transfer Server imports
//!   without asking the holder's server. The destination is cleared before each
//!   read and every byte checked afterwards. It then prints the bandwidths
//!   the Transfer Server measured for the links from the source host.
//!
//! With `--gpu none` a role uses no GPU: the holder publishes its host
//! source alone, and the reader reads into host storage, so the path
//! between the hosts' host storage is checked and measured without one.
//! `--into host` has a reader with a GPU read into host storage too.
//!
//! A source's sequence is a function of each byte's index and the seed
//! alone, so the reader checks the bytes without the holder: the device
//! source has seed `--seed`, the host source `--seed + 1`. The first read
//! of a link also connects the Transfer Servers, registers the regions and
//! measures the link; the later rounds time the link alone.
//!
//! Each line of output is one JSON object with `t` (wall-clock Unix
//! seconds) and `event`:
//!
//! - `holding`: the server's endpoint, the GPU's UUID and the bytes of each
//!   source; `released` once the holder let them go.
//! - `read`: the link, the round, the bytes, the seconds from submission to
//!   the task's end and the bandwidth in GB/s.
//! - `links`: each link from the source host the destination's Transfer
//!   Server measured, in bytes per second.
//! - `error`, and the program exits with status 1, on any failure, a byte
//!   that differs from the source's included.
//!
//! ```text
//! # Source host (CUDA_VISIBLE_DEVICES naming the GPU it may use):
//! remote_reads hold --socket $RUNTIME/uniserve/storage.sock --gpu 0 --mib 2048 \
//!   --release /shared/release-hold
//! # Destination host, with the endpoint `holding` printed:
//! remote_reads read --socket $RUNTIME/uniserve/storage.sock --gpu 0 --mib 2048 \
//!   --source 10.0.0.2:41234 --links device,host --rounds 4
//! ```

use std::io::Write as _;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::fs::FileExt as _;
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ::storage::protocol::{
    Event, FabricHandle, LinkEnd, Priority, Region, Reply, Request, Span, TransferEnd,
};
use ::storage::{Access, Client, Key, Placement, Role, Scope, StorageRecord, TensorRecord};
use anyhow::{Context, bail, ensure};
use clap::Parser as _;
use cuda::{
    CopyBatch, DeviceAllocation, DeviceAttribute, DeviceId, PinnedBuffer, ShareableHandle,
    SourceOrder, Stream,
};
use rustix::event::{PollFd, PollFlags, poll};

/// The two roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Mode {
    Hold,
    Read,
}

/// A source of the holder, and how the reader reaches it: its device or
/// host tensor through the holder's Storage Server (a NIXL READ), or its
/// device storage by the fabric handle the holder exported (a copy over
/// the NVLink fabric, which the reader's Transfer Server imports).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Source {
    Device,
    Host,
    Fabric,
}

impl Source {
    fn name(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Host => "host",
            Self::Fabric => "fabric",
        }
    }

    /// The seed of the source's sequence, from the holder's `seed`: the
    /// fabric source is the device source's storage.
    fn seed(self, seed: u64) -> u64 {
        match self {
            Self::Device | Self::Fabric => seed,
            Self::Host => seed + 1,
        }
    }

    /// The region a span reads the source from: the holder's tensor at its
    /// server's `endpoint`, or its device storage by `fabric`.
    fn region(self, endpoint: &str, fabric: Option<&Fabric>) -> anyhow::Result<Region> {
        let remote = |name: &str| Region::Remote {
            endpoint: endpoint.to_owned(),
            region: Box::new(Region::Tensor {
                key: key(name),
                at: None,
            }),
        };
        Ok(match self {
            Self::Device => remote(DEVICE_TENSOR),
            Self::Host => remote(HOST_TENSOR),
            Self::Fabric => {
                let fabric = fabric.context("the fabric source needs --fabric and --fabric-gpu")?;
                Region::Fabric {
                    handle: fabric.handle.clone(),
                    gpu: fabric.gpu,
                    bytes: fabric.bytes,
                    offset: 0,
                }
            }
        })
    }
}

/// The tensors the holder publishes its device and host sources as.
const DEVICE_TENSOR: &str = "remote-device";
const HOST_TENSOR: &str = "remote-host";

/// The holder's device storage as its fabric handle names it.
struct Fabric {
    handle: Vec<u8>,
    gpu: [u8; 16],
    bytes: u64,
}

/// The bytes `text` spells in hexadecimal.
fn unhex(text: &str) -> anyhow::Result<Vec<u8>> {
    ensure!(
        text.len().is_multiple_of(2),
        "{text:?} is not hexadecimal bytes"
    );
    (0..text.len())
        .step_by(2)
        .map(|index| Ok(u8::from_str_radix(&text[index..index + 2], 16)?))
        .collect()
}

/// Publish or read retained tensors across hosts.
#[derive(clap::Parser)]
struct Options {
    #[arg(value_enum)]
    mode: Mode,
    #[arg(long)]
    socket: PathBuf,
    #[arg(long, default_value = "remote-reads")]
    deployment: String,
    /// CUDA ordinal, or "none" for host storage alone.
    #[arg(long, default_value = "0")]
    gpu: String,
    #[arg(long = "mib", default_value = "2048", value_parser = parse_mib)]
    bytes: u64,
    #[arg(long, default_value = "17")]
    seed: u64,
    /// A file on shared storage whose creation ends the holder.
    #[arg(long, required_if_eq("mode", "hold"))]
    release: Option<PathBuf>,
    #[arg(long, required_if_eq("mode", "read"))]
    source: Option<String>,
    #[arg(long, requires = "fabric_gpu")]
    fabric: Option<String>,
    #[arg(long, requires = "fabric")]
    fabric_gpu: Option<String>,
    #[arg(long, value_enum, value_delimiter = ',', default_value = "device,host")]
    links: Vec<Source>,
    #[arg(long, default_value = "4")]
    rounds: u32,
    #[arg(long = "into", default_value = "device", value_parser = parse_target, action = clap::ArgAction::Set)]
    into_host: bool,
    #[arg(long, default_value = "read-only", value_parser = parse_access)]
    host_access: Access,
    #[arg(long, default_value = "yes", value_parser = clap::builder::BoolishValueParser::new(), action = clap::ArgAction::Set)]
    record_fabric: bool,
    /// Timeout for one read, including its first registration.
    #[arg(long = "patience-s", default_value = "300", value_parser = parse_seconds)]
    patience: Duration,
}

impl Options {
    fn gpu(&self) -> anyhow::Result<Option<DeviceId>> {
        match self.gpu.as_str() {
            "none" => Ok(None),
            ordinal => Ok(Some(DeviceId(ordinal.parse()?))),
        }
    }

    fn fabric(&self) -> anyhow::Result<Option<Fabric>> {
        self.fabric
            .as_ref()
            .zip(self.fabric_gpu.as_ref())
            .map(|(handle, gpu)| {
                Ok(Fabric {
                    handle: unhex(handle)?,
                    gpu: unhex(gpu)?
                        .try_into()
                        .map_err(|_| anyhow::anyhow!("--fabric-gpu is a 16-byte UUID"))?,
                    bytes: 0,
                })
            })
            .transpose()
    }
}

fn parse_mib(text: &str) -> anyhow::Result<u64> {
    let mib: u64 = text.parse()?;
    ensure!(mib > 0, "--mib must be positive");
    Ok(mib << 20)
}

fn parse_target(text: &str) -> anyhow::Result<bool> {
    match text {
        "host" => Ok(true),
        "device" => Ok(false),
        _ => bail!("expected host or device"),
    }
}

fn parse_access(text: &str) -> anyhow::Result<Access> {
    match text {
        "read-only" => Ok(Access::ReadOnly),
        "exclusive" => Ok(Access::Exclusive),
        _ => bail!("expected read-only or exclusive"),
    }
}

fn parse_seconds(text: &str) -> anyhow::Result<Duration> {
    Ok(Duration::try_from_secs_f64(text.parse()?)?)
}

/// The current wall-clock time in Unix seconds.
fn wall() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |elapsed| elapsed.as_secs_f64())
}

/// Print one JSON line with the wall-clock time and structured fields.
fn emit(event: &str, fields: serde_json::Value) {
    let mut value = fields;
    value["t"] = serde_json::json!(wall());
    value["event"] = serde_json::json!(event);
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "{value}");
    let _ = stdout.flush();
}

/// `bytes` in hexadecimal.
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
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

/// Byte `index` of the sequence of `seed`: a multiplicative hash of the
/// index, without short periods, so a byte read from a wrong offset (even a
/// multiple of 256 away) differs from the expected one.
fn sequence_byte(index: u64, seed: u64) -> u8 {
    ((index.wrapping_add(seed)).wrapping_mul(2_654_435_761) >> 16) as u8
}

/// Write the sequence of `seed` from index 0 into `bytes`.
fn fill(bytes: &mut [u8], seed: u64) {
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = sequence_byte(index as u64, seed);
    }
}

/// The first index at which `bytes` differs from the sequence of `seed`.
fn first_difference(bytes: &[u8], seed: u64) -> Option<usize> {
    bytes
        .iter()
        .enumerate()
        .position(|(index, &byte)| byte != sequence_byte(index as u64, seed))
}

/// The deployment's head (whose claim covers the program's one scope) and
/// its worker under the claim.
fn connect(options: &Options) -> anyhow::Result<(Client, Client)> {
    let scope = Scope {
        deployment: options.deployment.clone(),
        group: "reads".into(),
        rank: 0,
    };
    let mut head = Client::connect(
        &options.socket,
        Role::Head {
            deployment: options.deployment.clone(),
            world: 0,
        },
    )
    .with_context(|| format!("connecting to {}", options.socket.display()))?;
    let claim = head.claim(0, vec![scope.clone()], Vec::new())?;
    let worker = Client::connect(
        &options.socket,
        Role::Worker {
            scope,
            generation: 1,
            claim,
            world: 0,
        },
    )?;
    Ok((head, worker))
}

/// Retainable device storage of at least `bytes` on `gpu`, with its
/// descriptor and its storage record.
fn device_storage(
    gpu: DeviceId,
    bytes: u64,
) -> anyhow::Result<(DeviceAllocation, OwnedFd, StorageRecord)> {
    let allocation = DeviceAllocation::retainable(gpu, usize::try_from(bytes)?, &[gpu])?;
    let descriptor = allocation
        .export_descriptor()?
        .context("a retainable allocation exports a descriptor")?;
    let record = StorageRecord {
        placement: Placement::Device { gpu: gpu.uuid()?.0 },
        bytes: allocation.len() as u64,
        segment: None,
        fabric: None,
    };
    Ok((allocation, descriptor, record))
}

/// Copy `staging`'s first `bytes` bytes to `allocation`, or `allocation`'s
/// to `staging` when `to_host`, and wait for the copy.
fn copy(
    allocation: &DeviceAllocation,
    staging: &PinnedBuffer,
    bytes: usize,
    to_host: bool,
) -> anyhow::Result<()> {
    let stream = Stream::new(allocation.device())?;
    let mut batch = CopyBatch::new(SourceOrder::Stream);
    if to_host {
        batch.push(staging.address(), allocation.address(), bytes);
    } else {
        batch.push(allocation.address(), staging.address(), bytes);
    }
    // SAFETY: both ranges hold `bytes` bytes and nothing else uses them
    // until the stream drains below.
    unsafe { batch.submit(&stream) }?;
    stream.synchronize()?;
    Ok(())
}

/// A host memfd of `bytes` holding the sequence of `seed`, or zeros
/// without one, whose pages prefer NUMA node `node`.
fn host_storage(bytes: u64, node: Option<u32>, seed: Option<u64>) -> anyhow::Result<OwnedFd> {
    use rustix::mm::{MapFlags, ProtFlags, mmap, munmap};

    let descriptor = rustix::fs::memfd_create(c"remote-reads", rustix::fs::MemfdFlags::CLOEXEC)?;
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
    // SAFETY: the mapping holds `length` writable bytes, which this function
    // alone references until it unmaps them.
    let contents = unsafe { std::slice::from_raw_parts_mut(address.cast::<u8>(), length) };
    match seed {
        Some(seed) => fill(contents, seed),
        // Touched, so that the pages are allocated under the policy above.
        None => contents.fill(0),
    }
    // SAFETY: the mapping made above, which nothing references any more.
    unsafe { munmap(address, length) }?;
    Ok(descriptor)
}

/// The holder: publish its sources and keep them until `--release`
/// exists.
fn hold(options: &Options) -> anyhow::Result<()> {
    let gpu = options.gpu()?;
    let numa_node = gpu
        .and_then(|gpu| gpu.attribute(DeviceAttribute::HostNumaId).ok())
        .and_then(|node| u32::try_from(node).ok());
    let (mut head, mut worker) = connect(options)?;

    // The device source, on a GPU, holding its sequence.
    let device = match gpu {
        Some(gpu) => {
            let (device, descriptor, mut record) = device_storage(gpu, options.bytes)?;
            // The allocation's creator exports its fabric handle, which a
            // reader's Transfer Server imports where the GPUs share a fabric.
            if options.record_fabric
                && let Some(ShareableHandle::Fabric(handle)) = device.export()?
            {
                record.fabric = Some(FabricHandle(handle));
            }
            let mut staging = PinnedBuffer::new(gpu, device.len())?;
            // SAFETY: a pinned buffer this program owns, written before any
            // copy.
            fill(
                unsafe { staging.as_mut_slice() },
                Source::Device.seed(options.seed),
            );
            copy(&device, &staging, device.len(), false)?;
            Some((device, descriptor, record))
        }
        None => None,
    };
    // The host source, as large as the device source's rounded storage.
    let host_bytes = device
        .as_ref()
        .map_or(options.bytes, |(device, ..)| device.len() as u64);
    let host = host_storage(host_bytes, numa_node, Some(Source::Host.seed(options.seed)))?;

    let mut tensors = Vec::new();
    let mut storages: Vec<(StorageRecord, BorrowedFd<'_>)> = Vec::new();
    if let Some((device, descriptor, record)) = &device {
        tensors.push(tensor(
            DEVICE_TENSOR,
            storages.len() as u32,
            device.len() as u64,
            Access::ReadOnly,
        ));
        storages.push((*record, descriptor.as_fd()));
    }
    // Read-only host storage (as weights kept on the host are), which the
    // source's Transfer Server maps read-only and registers for reads
    // alone, or writable host storage (as host page pools are).
    tensors.push(tensor(
        HOST_TENSOR,
        storages.len() as u32,
        host_bytes,
        options.host_access,
    ));
    storages.push((
        StorageRecord {
            placement: Placement::Host { numa_node },
            bytes: host_bytes,
            segment: None,
            fabric: None,
        },
        host.as_fd(),
    ));
    worker.put(tensors, &storages)?;
    drop(storages);
    // The device source's fabric handle, which another host of the NVLink
    // fabric imports while this process holds the allocation.
    let fabric_handle = match device.as_ref().map(|(device, ..)| device.export()) {
        Some(Ok(Some(ShareableHandle::Fabric(handle)))) => Some(handle),
        _ => None,
    };
    let endpoint = match head.call(&Request::Endpoint, &[])?.reply {
        Reply::Endpoint { tcp: Some(tcp) } => tcp,
        other => bail!("the server accepts no other host ({other:?}): start it with --tcp"),
    };
    emit(
        "holding",
        serde_json::json!({
            "endpoint": endpoint,
            "gpu": match gpu {
                Some(gpu) => hex(&gpu.uuid()?.0),
                None => "none".to_owned(),
            },
            "bytes": host_bytes,
            "fabric": fabric_handle.map_or_else(String::new, |handle| hex(&handle)),
        }),
    );

    // The sources stay published until the harness creates the release
    // file; the program's end then ends its claim and the server releases
    // them.
    let release = options.release.as_ref().context("--release is required")?;
    while !release.exists() {
        std::thread::sleep(Duration::from_millis(200));
    }
    // The server retains published tensors past their worker's
    // connection: the scope retains none of them any more, so the next run
    // starts afresh.
    worker.retain(Vec::new(), Vec::new())?;
    drop((worker, head, device, host));
    emit("released", serde_json::json!({}));
    Ok(())
}

/// Wait for task `task` of `worker` to end, failing unless it completed
/// within `patience`.
fn wait_for(worker: &mut Client, task: u64, patience: Duration) -> anyhow::Result<u64> {
    let deadline = Instant::now() + patience;
    loop {
        worker.pump()?;
        while let Some((event, _)) = worker.next_event() {
            if let Event::Transferred {
                task: ended,
                bytes,
                end,
            } = event
                && ended == task
            {
                return match end {
                    TransferEnd::Complete => Ok(bytes),
                    TransferEnd::Cancelled => bail!("task {task} was cancelled"),
                    TransferEnd::Failed { message } => bail!("task {task} failed: {message}"),
                };
            }
        }
        let left = deadline.saturating_duration_since(Instant::now());
        ensure!(
            !left.is_zero(),
            "task {task} did not end within {patience:?}"
        );
        let timeout = rustix::time::Timespec::try_from(left)?;
        let connection = worker.as_fd();
        let mut fds = [PollFd::new(&connection, PollFlags::IN)];
        match poll(&mut fds, Some(&timeout)) {
            Ok(_) | Err(rustix::io::Errno::INTR) => {}
            Err(error) => return Err(error.into()),
        }
    }
}

/// The reader's destination: device storage with a pinned buffer to clear
/// and read it through, or host storage.
enum Destination {
    Device {
        allocation: DeviceAllocation,
        staging: PinnedBuffer,
    },
    Host {
        file: std::fs::File,
    },
}

impl Destination {
    /// Zero the destination's first `bytes` bytes.
    fn clear(&mut self, bytes: usize) -> anyhow::Result<()> {
        match self {
            Self::Device {
                allocation,
                staging,
            } => {
                // SAFETY: a pinned buffer this program owns, idle between
                // copies.
                unsafe { staging.as_mut_slice() }.fill(0);
                copy(allocation, staging, bytes, false)
            }
            Self::Host { file } => {
                let zeros = vec![0_u8; 64 << 20];
                let mut offset = 0;
                while offset < bytes {
                    let length = zeros.len().min(bytes - offset);
                    file.write_all_at(&zeros[..length], offset as u64)?;
                    offset += length;
                }
                Ok(())
            }
        }
    }

    /// The first of the destination's `bytes` bytes that differs from the
    /// sequence of `seed`, with its value and how many of the bytes differ
    /// and how many are zero, which tell a READ that wrote nothing (all
    /// zero, as cleared) from one that wrote wrong bytes.
    fn first_difference(&mut self, bytes: usize, seed: u64) -> anyhow::Result<Option<Mismatch>> {
        let check = |read: &[u8]| {
            first_difference(&read[..bytes], seed).map(|index| Mismatch {
                index,
                value: read[index],
                differing: read[..bytes]
                    .iter()
                    .enumerate()
                    .filter(|&(at, &byte)| byte != sequence_byte(at as u64, seed))
                    .count(),
                zero: read[..bytes].iter().filter(|&&byte| byte == 0).count(),
            })
        };
        match self {
            Self::Device {
                allocation,
                staging,
            } => {
                copy(allocation, staging, bytes, true)?;
                // SAFETY: the copy into the buffer completed.
                Ok(check(unsafe { staging.as_slice() }))
            }
            Self::Host { file } => {
                let mut read = vec![0_u8; bytes];
                file.read_exact_at(&mut read, 0)?;
                Ok(check(&read))
            }
        }
    }
}

/// Where a destination differs from its source's sequence.
struct Mismatch {
    index: usize,
    value: u8,
    differing: usize,
    zero: usize,
}

/// The reader: read each source `--rounds` times and check every byte.
fn read(options: &Options) -> anyhow::Result<()> {
    let source = options.source.clone().context("--source is required")?;
    let (mut head, mut worker) = connect(options)?;

    let into = if options.into_host {
        None
    } else {
        options.gpu()?
    };
    let (mut destination, bytes) = match into {
        Some(gpu) => {
            let (allocation, descriptor, record) = device_storage(gpu, options.bytes)?;
            let bytes = allocation.len();
            worker.put(
                vec![tensor("destination", 0, bytes as u64, Access::Exclusive)],
                &[(record, descriptor.as_fd())],
            )?;
            let staging = PinnedBuffer::new(gpu, bytes)?;
            (
                Destination::Device {
                    allocation,
                    staging,
                },
                bytes,
            )
        }
        None => {
            let descriptor = host_storage(options.bytes, None, None)?;
            worker.put(
                vec![tensor("destination", 0, options.bytes, Access::Exclusive)],
                &[(
                    StorageRecord {
                        placement: Placement::Host { numa_node: None },
                        bytes: options.bytes,
                        segment: None,
                        fabric: None,
                    },
                    descriptor.as_fd(),
                )],
            )?;
            (
                Destination::Host {
                    file: std::fs::File::from(descriptor),
                },
                usize::try_from(options.bytes)?,
            )
        }
    };

    // The holder's fabric handle names its whole device storage, as long
    // as the destination.
    let fabric = options.fabric()?.map(|fabric| Fabric {
        handle: fabric.handle.clone(),
        gpu: fabric.gpu,
        bytes: bytes as u64,
    });
    let mut update = 1;
    for &link in &options.links {
        for round in 0..options.rounds {
            // A cleared destination, so that a read that wrote nothing
            // fails the check.
            destination.clear(bytes)?;

            let started = Instant::now();
            let task = worker.transfer(
                vec![Span {
                    source: link.region(&source, fabric.as_ref())?,
                    source_offset: 0,
                    destination: Region::Tensor {
                        key: key("destination"),
                        at: None,
                    },
                    destination_offset: 0,
                    // One row of the whole source: the pitches go unused.
                    bytes: bytes as u64,
                    rows: 1,
                    source_pitch: 0,
                    destination_pitch: 0,
                }],
                Priority::Background,
                update,
            )?;
            update += 1;
            let copied = wait_for(&mut worker, task, options.patience)?;
            let seconds = started.elapsed().as_secs_f64();
            ensure!(
                copied == bytes as u64,
                "the read of {} copied {copied} of {bytes} bytes",
                link.name()
            );

            if let Some(mismatch) = destination.first_difference(bytes, link.seed(options.seed))? {
                bail!(
                    "byte {} read from the {} source is {:#04x}, not {:#04x}; {} of {bytes} \
                     bytes differ, {} are zero",
                    mismatch.index,
                    link.name(),
                    mismatch.value,
                    sequence_byte(mismatch.index as u64, link.seed(options.seed)),
                    mismatch.differing,
                    mismatch.zero
                );
            }
            emit(
                "read",
                serde_json::json!({
                    "link": link.name(),
                    "round": round,
                    "bytes": bytes,
                    "seconds": seconds,
                    "gbps": bytes as f64 / seconds / 1e9,
                }),
            );
        }
    }

    // The links from the source host the Transfer Server measured at their
    // first use.
    let listing = head.list()?;
    let transfer = listing
        .transfer
        .context("the Transfer Server left before the listing")?;
    let measured: Vec<serde_json::Value> = transfer
        .links
        .iter()
        .filter_map(|measured| match &measured.link.source {
            LinkEnd::Remote {
                endpoint,
                gpu: remote,
            } if *endpoint == source => Some(serde_json::json!({
                "source": remote.map_or_else(|| "host".to_owned(), |gpu| hex(&gpu)),
                "destination": match &measured.link.destination {
                    LinkEnd::Device { gpu } => hex(gpu),
                    LinkEnd::Host => "host".to_owned(),
                    other => format!("{other:?}"),
                },
                "bytes_per_second": measured.bytes_per_second,
            })),
            _ => None,
        })
        .collect();
    emit("links", serde_json::json!({"measured": measured}));
    // The server retains the destination past this connection: the scope
    // retains it no more, so the next reader of the deployment publishes
    // its own.
    worker.retain(Vec::new(), Vec::new())?;
    Ok(())
}

fn main() {
    let options = Options::parse();
    let result = match options.mode {
        Mode::Hold => hold(&options),
        Mode::Read => read(&options),
    };
    if let Err(error) = result {
        emit(
            "error",
            serde_json::json!({"message": format!("{error:#}")}),
        );
        std::process::exit(1);
    }
}
