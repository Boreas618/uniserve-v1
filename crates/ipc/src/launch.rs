//! The launch descriptor a rank process is started with.
//!
//! A rank is started with the head's address, its own identity and the launch
//! it belongs to, and the head answers the rank's registration with one
//! [`LaunchDescriptor`], which
//! carries every other launch value. The rank's runtime resolves the raw
//! values once, before any model work, and reports a value it cannot run
//! with as a usage error.
//!
//! Every launch value is stated explicitly: the launching side is the single
//! source of defaults and validates the values it owns, and the rank never
//! re-derives or re-checks one.
//!
//! Both launching sides, the head and a host's `uniserve-host` launcher, start
//! a rank with [`rank_command`], hand it the sockets it serves with
//! [`inherit_listener`] and own its process as a [`RankChild`].

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::net::TcpListener;
use std::os::fd::{AsRawFd, RawFd};
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, ExitStatus};

use serde::{Deserialize, Serialize};
use uniserve_core::{CanvasSampling, ComponentConfig};

use crate::LaneConfig;

/// The command that starts one rank with the interpreter `python`.
///
/// argv carries the address of the head the rank connects to and what its
/// registration states: its worker group and its rank, and the `generation`
/// of the group's launch it belongs to. They are the rank's only statement of
/// its identity; the launch descriptor repeats none of them. The caller adds
/// the environment and any inherited sockets.
pub fn rank_command(
    python: impl AsRef<OsStr>,
    head: &str,
    worker_id: &str,
    rank: u32,
    generation: u64,
) -> Command {
    let mut command = Command::new(python);
    command
        .arg("-m")
        .arg("uniserve.worker")
        .arg("--head")
        .arg(head)
        .arg("--worker-id")
        .arg(worker_id)
        .arg("--rank")
        .arg(rank.to_string())
        .arg("--generation")
        .arg(generation.to_string());
    command
}

/// Makes the process `command` starts inherit `listener` at the descriptor
/// number returned, which the caller names in that process's launch
/// descriptor.
///
/// A worker group's first rank serves the group's collective store. The
/// process that spawns that rank binds the store's listening socket before
/// the rank exists and passes it across the rank's exec, so the port is never
/// released between the reservation and the store serving on it.
///
/// The command owns `listener` from here on and closes it when the command is
/// dropped, so the socket stays open until the spawn has duplicated it and the
/// spawning process holds no listening copy afterwards. Close-on-exec is
/// cleared only in the forked child: the descriptor keeps the flag in the
/// spawning process, so no other process it starts meanwhile inherits the
/// socket. A failure to clear the flag fails the spawn.
pub fn inherit_listener(command: &mut Command, listener: TcpListener) -> RawFd {
    let fd = listener.as_raw_fd();

    // SAFETY: the closure runs in the forked child between fork and exec, and
    // only calls `fcntl`, which is async-signal-safe. `listener` is moved into
    // the closure, so the descriptor is open in the parent at fork time.
    unsafe {
        command.pre_exec(move || {
            // FD_CLOEXEC is the only descriptor flag, so clearing every flag
            // leaves the socket open across exec at the same number.
            if libc::fcntl(listener.as_raw_fd(), libc::F_SETFD, 0) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
    fd
}

/// The process of a rank that this process spawned, and is the parent that
/// reaps it.
///
/// A rank that ends in order destroys its CUDA contexts, which removes the
/// shared memory the driver keeps for each context's interprocess events. A
/// rank that is killed or crashes leaves that memory in `/dev/shm`, named
/// for the rank's process id ([`remove_event_memory`]), and the names of
/// its NCCL communicators' proxy memory ([`NcclSegments`]), which the
/// parent notes while the rank lives. Every wait here therefore observes
/// the exit first without reaping it, removes the memory named for the rank
/// while its zombie still holds the process id, so that no live process can
/// own those names, and the noted NCCL names, and then reaps it.
pub struct RankChild {
    child: Child,
    /// The exit status once the process was reaped; its id may name another
    /// process from then on, so it is never waited on again.
    status: Option<ExitStatus>,
    /// The NCCL shared memory the rank was last seen to map under names
    /// still linked ([`Self::note_nccl_segments`]).
    segments: NcclSegments,
}

impl RankChild {
    /// Spawns the rank `command` describes.
    pub fn spawn(command: &mut Command) -> std::io::Result<Self> {
        Ok(Self {
            child: command.spawn()?,
            status: None,
            segments: NcclSegments::default(),
        })
    }

    /// The rank's process id.
    pub fn id(&self) -> u32 {
        self.child.id()
    }

    /// Notes the NCCL shared memory the running rank maps under names still
    /// linked, which the reaping of a rank that died without destroying its
    /// communicators removes. A rank's communicators exist once it reported
    /// that it is ready: warmup and graph capture have connected them all.
    pub fn note_nccl_segments(&mut self) -> std::io::Result<()> {
        if self.status.is_none() {
            self.segments
                .extend(NcclSegments::mapped_by(Some(self.child.id()))?);
        }
        Ok(())
    }

    /// Sends the rank SIGKILL, unless it was already reaped.
    pub fn kill(&mut self) -> std::io::Result<()> {
        if self.status.is_some() {
            return Ok(());
        }
        self.child.kill()
    }

    /// Returns the rank's exit status if it has exited, after releasing what
    /// it left and reaping it, and `None` while it runs.
    pub fn try_wait(&mut self) -> std::io::Result<Option<ExitStatus>> {
        if self.status.is_none() && self.exited(libc::WNOHANG)? {
            self.reap()?;
        }
        Ok(self.status)
    }

    /// Waits for the rank to exit, releases what it left, reaps it and
    /// returns its exit status.
    pub fn wait(&mut self) -> std::io::Result<ExitStatus> {
        if let Some(status) = self.status {
            return Ok(status);
        }
        self.exited(0)?;
        self.reap()
    }

    /// Whether the rank has exited, observed with `waitid(WNOWAIT)`, which
    /// leaves it unreaped. Blocks until it exits unless `flags` holds
    /// `WNOHANG`.
    fn exited(&self, flags: libc::c_int) -> std::io::Result<bool> {
        let pid = self.child.id();
        loop {
            // SAFETY: an all-zero `siginfo_t` is a valid value for the call
            // to fill.
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            // SAFETY: `info` is a valid out-pointer and `pid` is this
            // process's unreaped child.
            let waited = unsafe {
                libc::waitid(
                    libc::P_PID,
                    pid,
                    &mut info,
                    libc::WEXITED | libc::WNOWAIT | flags,
                )
            };
            if waited == -1 {
                let error = std::io::Error::last_os_error();
                if error.kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(error);
            }
            // With WNOHANG, a child that has not exited leaves `si_pid` zero.
            // SAFETY: the call succeeded, so `info` describes the child.
            return Ok(unsafe { info.si_pid() } != 0);
        }
    }

    /// Releases what the exited, unreaped rank left and reaps it.
    fn reap(&mut self) -> std::io::Result<ExitStatus> {
        // The rank inherited this process's user. The removal is best
        // effort: the rank is reaped whatever it finds.
        // SAFETY: getuid has no preconditions.
        let uid = unsafe { libc::getuid() };
        let _ = remove_event_memory(uid, self.child.id());
        let _ = self.segments.remove();
        let status = self.child.wait()?;
        self.status = Some(status);
        Ok(status)
    }
}

/// Where POSIX shared-memory objects live on Linux.
const SHARED_MEMORY: &str = "/dev/shm";

/// Removes the shared memory the CUDA driver created for the interprocess
/// events of process `pid`, run by user `uid`, and returns the paths it
/// removed. It needs no CUDA: it encodes the driver's naming of that memory,
/// observed with driver 580.82.07 by tracing `shm_open` and `shm_unlink`
/// across processes, users, threads and contexts.
///
/// The driver backs each context's interprocess events with one POSIX
/// shared-memory object, which it creates exclusively (`O_CREAT | O_EXCL`,
/// mode 0600) with the context's first such event, and names
/// `cuda.shm.<uid>.<pid>.<n>`: the creating process's user id and process id
/// in lowercase hexadecimal (user 65534 is `fffe`), and the context's
/// ordinal in that process, from 1, in decimal. The process id is the
/// process's (`getpid`), whichever of its threads created the event, and a
/// process that opens another's event handle opens the other's object by
/// that name and creates none of its own. The driver unlinks the object only
/// when the context is destroyed (`uniserve_cuda::PrimaryContext::reset_all`),
/// which a process that is killed or crashes never does, so the parent that
/// reaps such a process removes it instead ([`RankChild`]).
///
/// The caller calls this after `pid` exited and before reaping it: until the
/// process is reaped its zombie holds the process id, so no live process can
/// own a name with that id. A process may also remove its own names as the
/// last thing before it ends without destroying its contexts; removing a
/// name leaves every mapping of the object valid until its holder ends. Exactly the names of that form with this user and
/// process id are removed. Every matching object is attempted; the first
/// failure other than an object already gone is returned.
pub fn remove_event_memory(uid: u32, pid: u32) -> std::io::Result<Vec<std::path::PathBuf>> {
    let owned = |name: &str| {
        let mut fields = name.split('.');
        let (Some("cuda"), Some("shm"), Some(user), Some(process), Some(ordinal), None) = (
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
        ) else {
            return false;
        };
        user == format!("{uid:x}")
            && process == format!("{pid:x}")
            && !ordinal.is_empty()
            && ordinal.bytes().all(|byte| byte.is_ascii_digit())
    };

    let mut removed = Vec::new();
    let mut failure = None;
    for entry in std::fs::read_dir(SHARED_MEMORY)? {
        let entry = entry?;
        if !entry.file_name().to_str().is_some_and(owned) {
            continue;
        }
        let path = entry.path();
        match std::fs::remove_file(&path) {
            Ok(()) => removed.push(path),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                failure.get_or_insert(error);
            }
        }
    }
    failure.map_or(Ok(removed), Err)
}

/// Where NCCL creates its shared-memory objects: `/dev/shm/nccl-<random>`.
const NCCL_PREFIX: &str = "/dev/shm/nccl-";

/// NCCL's named shared-memory objects that a process maps, each by path and
/// inode.
///
/// This encodes NCCL's naming, observed with NCCL 2.30.4 by tracing
/// `mkstemp`, `open` and `unlink` across the ranks of a group. NCCL creates
/// its shared memory under random names (`/dev/shm/nccl-XXXXXX`). A
/// connection buffer is created by one rank and unlinked by its peer once
/// the peer has attached; a pool per communicator is created by the
/// communicator's proxy thread, mapped by its own thread too, and unlinked
/// only when the communicator is destroyed or aborted. A process that ends
/// without either therefore leaves its pools' names, which carry no process
/// id: the names still linked among a process's mappings
/// (`/proc/<pid>/maps`, where an unlinked one reads `(deleted)`) are its own.
/// Every one a group's ranks created was created before they reported
/// ready, none while they served. The inode tells an object from a later
/// one of the same name, so a removal never touches another's.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct NcclSegments(Vec<(std::path::PathBuf, u64)>);

impl NcclSegments {
    /// The NCCL objects process `pid` (this process for `None`) maps under
    /// names still linked.
    pub fn mapped_by(pid: Option<u32>) -> std::io::Result<Self> {
        let maps = match pid {
            Some(pid) => std::fs::read_to_string(format!("/proc/{pid}/maps"))?,
            None => std::fs::read_to_string("/proc/self/maps")?,
        };
        let mut segments = Self::default();
        for line in maps.lines() {
            // address, permissions, offset, device, inode, path and, for an
            // unlinked object, "(deleted)".
            let fields: Vec<&str> = line.split_whitespace().collect();
            let (Some(inode), Some(path), None) = (fields.get(4), fields.get(5), fields.get(6))
            else {
                continue;
            };
            if !path.starts_with(NCCL_PREFIX) {
                continue;
            }
            if let Ok(inode) = inode.parse() {
                segments.add(std::path::PathBuf::from(path), inode);
            }
        }
        Ok(segments)
    }

    /// Adds the objects of `other` that are not held yet.
    pub fn extend(&mut self, other: Self) {
        for (path, inode) in other.0 {
            self.add(path, inode);
        }
    }

    fn add(&mut self, path: std::path::PathBuf, inode: u64) {
        if !self.0.contains(&(path.clone(), inode)) {
            self.0.push((path, inode));
        }
    }

    /// Unlinks every object whose name still names it, by inode, and
    /// returns the paths removed. Unlinking removes the name only: every
    /// mapping of the object stays valid until its holder ends. Every object
    /// is attempted; the first failure other than one already gone is
    /// returned.
    pub fn remove(&self) -> std::io::Result<Vec<std::path::PathBuf>> {
        use std::os::unix::fs::MetadataExt;

        let mut removed = Vec::new();
        let mut failure = None;
        for (path, inode) in &self.0 {
            let current = match std::fs::metadata(path) {
                Ok(metadata) => metadata.ino(),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
                Err(error) => {
                    failure.get_or_insert(error);
                    continue;
                }
            };
            if current != *inode {
                continue;
            }
            match std::fs::remove_file(path) {
                Ok(()) => removed.push(path.clone()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        failure.map_or(Ok(removed), Err)
    }
}

/// Every value one rank process is launched with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LaunchDescriptor {
    /// This rank's word in every published product's acknowledgment array.
    pub acknowledgment_slot: u32,
    /// Acknowledgment slots of the ranks on this rank's host.
    pub host_slots: Vec<u32>,
    /// Whether some consumer of this rank's products runs on another host.
    pub products_cross_hosts: bool,
    /// Most requests outstanding on the rank's channel at once.
    pub queue_depth: usize,
    /// The head's watchdog timeout in milliseconds (`--watchdog-timeout`):
    /// how long a batch may go without progress of the group's ranks before
    /// the head replaces them. The rank reports a lane stalled
    /// ([`WorkerResponse::Stalled`](crate::WorkerResponse)) once one job has
    /// run on it for half of it, so the report precedes the head's expiry;
    /// at least two.
    pub watchdog_timeout_ms: u64,

    /// Checkpoint path or repository; empty for the stub model.
    pub model: String,
    /// Torch device name of the rank's primary device (`cpu`, `cuda`, `cuda:N`).
    pub device: String,
    /// Global rank within the worker.
    pub rank: u32,
    /// Rank index among the worker's ranks on the same host.
    pub local_rank: u32,
    /// Number of ranks in the worker.
    pub world_size: u32,
    /// Components this worker places, by name.
    pub components: BTreeMap<String, ComponentConfig>,
    /// Component names every group of the deployment places.
    pub deployment_components: Vec<String>,
    /// Comma-separated transfer mechanisms this rank binds.
    pub transfer_backends: String,
    /// Comma-separated transfer mechanisms this rank exports through.
    pub export_backends: String,

    /// `host:port` of the worker group's host-collective rendezvous.
    pub rendezvous_address: Option<String>,
    /// Inherited listening socket the group's first rank accepts its
    /// rendezvous peers on.
    pub rendezvous_listen_fd: Option<i32>,
    /// Membership in the deployment's expert-parallel union.
    pub expert_parallel: Option<ExpertParallelDescriptor>,
    /// `model` or `experts`.
    pub role: String,
    /// Concurrent microbatches of a disaggregated attention rank.
    pub expert_microbatches: u32,
    /// Torch process-group backend override.
    pub distributed_backend: Option<String>,
    /// Generation-tower placement, `tower=gen:<device>`: the tower's modules
    /// on a device other than the rank's.
    pub mesh: Option<String>,
    /// Execution lanes and their SM partitions.
    pub lane: Vec<LaneConfig>,
    /// Load the deterministic stub model instead of a checkpoint.
    pub no_model: bool,

    /// Checkpoint format or loading mode.
    pub load_format: String,
    pub download_dir: Option<String>,
    pub load_threads: Option<u32>,
    /// Compute dtype name.
    pub model_dtype: String,
    /// Precision selectors, a JSON object.
    pub quantization_config: serde_json::Value,
    pub prefix_cache_dtype: Option<String>,
    /// Fraction of device memory this rank may hold for static allocations,
    /// weights and prefix cache, in (0, 1].
    pub mem_fraction_static: f64,
    /// Prefix cache capacity in tokens; `None` sizes it from memory.
    pub max_total_tokens: Option<u64>,
    /// Attention provider name of the rank's model, as `--attention-backend`
    /// gave it. The rank's attention registry (`uniserve.backends.attention`)
    /// owns the names and refuses one it does not provide when the model
    /// loads; `auto` selects a native kernel per call.
    pub attention_backend: String,
    pub block_size: Option<u32>,
    pub max_batch_calls: u32,
    pub max_batch_tokens: u32,
    pub max_request_pool_size: u32,
    pub max_model_len: u32,
    pub max_video_seconds: f64,
    /// Condition rows one video request may occupy; `None` provisions the
    /// largest condition set the rank's denoiser admits.
    pub max_condition_rows: Option<u32>,
    /// Image features the rank's encoder cache keeps resident.
    pub encoder_cache_entries: u32,
    pub ffmpeg: String,
    pub min_video_seconds: Option<f64>,

    /// `off`, `auto` or `full`.
    pub graph_policy: String,
    /// Comma-separated strictly increasing bucket lists; `None` keeps the
    /// worker's defaults.
    pub decode_graph_batch_sizes: Option<String>,
    pub prefill_cuda_graph: bool,
    pub prefill_outputs: bool,
    pub prefill_graph_token_sizes: Option<String>,
    pub flow_cuda_graph: bool,
    pub flow_graph_batch_sizes: Option<String>,
    /// Comma-separated `HEIGHTxWIDTH` image capture sizes.
    pub flow_graph_shapes: Option<String>,
    pub video_text_capacities: Option<String>,
    /// Resolution classes (`768p`) a video deployment prepares, crossed
    /// with `video_aspect_ratios` resolution-major, in the order the
    /// deployment names them. The video denoiser names the canvas of each
    /// pair; empty takes every class it offers.
    pub video_resolutions: Vec<String>,
    /// Display aspect ratios (`16:9`) a video deployment prepares; empty
    /// takes every ratio the video denoiser offers.
    pub video_aspect_ratios: Vec<String>,
    pub canvas_sampling: Option<CanvasSampling>,

    /// FlashInfer attention options, in bytes and backend names.
    pub flashinfer_workspace_size: u64,
    /// `true` or `false`; `None` lets the backend decide.
    pub flashinfer_use_tensor_core: Option<String>,
    pub flashinfer_decode_backend: String,
    pub flashinfer_prefill_backend: String,
    pub flashinfer_decode_split_tile_size: Option<u32>,
    pub flashinfer_prefill_split_tile_size: Option<u32>,
    pub flashinfer_disable_split_kv: bool,
}

/// One rank's place in the deployment's expert-parallel union.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExpertParallelDescriptor {
    /// Rank within the union.
    pub rank: u32,
    /// Ranks in the union.
    pub size: u32,
    /// Leading union ranks that run attention (disaggregated experts); zero
    /// for colocated expert parallelism.
    pub attention_ranks: u32,
    /// `host:port` of the union's rendezvous.
    pub address: Option<String>,
    /// Inherited listening socket, held only by union rank 0.
    pub listen_fd: Option<i32>,
    /// `alltoall`, `megamoe`, `dwdp` or `deepep`.
    pub exchange: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shared-memory object created under NCCL's prefix and mapped by
    /// this process; unmapped when dropped.
    #[cfg(target_os = "linux")]
    struct Mapped {
        path: std::path::PathBuf,
        address: *mut libc::c_void,
    }

    #[cfg(target_os = "linux")]
    impl Mapped {
        fn new(name: &str) -> Self {
            use std::os::fd::AsRawFd;

            let path = std::path::PathBuf::from(format!("{NCCL_PREFIX}{name}"));
            let file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            file.set_len(4096).unwrap();
            // SAFETY: maps one page of the file just sized, shared.
            let address = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    4096,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_SHARED,
                    file.as_raw_fd(),
                    0,
                )
            };
            assert_ne!(address, libc::MAP_FAILED);
            Self { path, address }
        }
    }

    #[cfg(target_os = "linux")]
    impl Drop for Mapped {
        fn drop(&mut self) {
            // SAFETY: unmaps the page `new` mapped.
            unsafe { libc::munmap(self.address, 4096) };
            let _ = std::fs::remove_file(&self.path);
        }
    }

    /// Of the NCCL objects this process maps, those still linked are held
    /// and removed by name, an unlinked one is not held, and an object that
    /// replaced a held one under its name is left.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_process_loses_exactly_the_nccl_names_it_maps() {
        let tag = std::process::id();
        let linked = Mapped::new(&format!("test-{tag}-linked"));
        let unlinked = Mapped::new(&format!("test-{tag}-unlinked"));
        std::fs::remove_file(&unlinked.path).unwrap();
        let replaced = Mapped::new(&format!("test-{tag}-replaced"));

        let held = NcclSegments::mapped_by(None).unwrap();
        // The replacement keeps the name under a new inode.
        std::fs::remove_file(&replaced.path).unwrap();
        std::fs::write(&replaced.path, b"").unwrap();

        let paths: Vec<&std::path::PathBuf> = held.0.iter().map(|(path, _)| path).collect();
        assert!(paths.contains(&&linked.path));
        assert!(!paths.contains(&&unlinked.path));
        let removed = held.remove().unwrap();
        assert!(removed.contains(&linked.path));
        assert!(!linked.path.exists());
        assert!(replaced.path.exists());
    }

    /// Of the objects named like the driver's, only those of the exited
    /// process's user and id are removed: not a live process's, another
    /// user's, ones with the ids in decimal, or a name of another form.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_dead_process_loses_exactly_its_event_memory() {
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let pid = child.id();
        // Its exit is observed without reaping it, so its id stays taken.
        // SAFETY: an all-zero `siginfo_t` is valid; the call fills it.
        let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
        // SAFETY: waits for this test's own child without reaping it.
        let waited =
            unsafe { libc::waitid(libc::P_PID, pid, &mut info, libc::WEXITED | libc::WNOWAIT) };
        assert_eq!(waited, 0);

        // SAFETY: getuid has no preconditions.
        let uid = unsafe { libc::getuid() };
        let path = |name: String| std::path::Path::new(SHARED_MEMORY).join(name);
        let dead = [
            path(format!("cuda.shm.{uid:x}.{pid:x}.1")),
            path(format!("cuda.shm.{uid:x}.{pid:x}.2")),
        ];
        let kept = [
            path(format!("cuda.shm.{uid:x}.{:x}.1", std::process::id())),
            path(format!("cuda.shm.{:x}.{pid:x}.1", uid + 1)),
            path(format!("cuda.shm.{uid}.{pid}.1")),
            path(format!("cuda.shm.{uid:x}.{pid:x}")),
            path(format!("cuda.shm.{uid:x}.{pid:x}.1.part")),
        ];
        // Created exclusively, so that an object of a live process's is
        // never touched.
        for file in dead.iter().chain(&kept) {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(file)
                .unwrap();
        }

        let removed = remove_event_memory(uid, pid);
        child.wait().unwrap();
        let survived: Vec<bool> = kept.iter().map(|file| file.exists()).collect();
        for file in dead.iter().chain(&kept) {
            let _ = std::fs::remove_file(file);
        }

        let mut removed = removed.unwrap();
        removed.sort();
        assert_eq!(removed, dead);
        assert_eq!(survived, [true; 5]);
    }
}
