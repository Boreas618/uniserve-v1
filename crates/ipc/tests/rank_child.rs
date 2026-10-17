//! A rank process's parent removes the CUDA driver memory a rank that died
//! without destroying its contexts left behind, when it reaps the rank.
//!
//! The ranks here are Python processes that wait to be killed after they
//! create an interprocess event through the driver, as a rank does when it
//! exports a device product (run with
//! `cargo test -p uniserve-ipc --test rank_child -- --ignored`), or after
//! they map shared memory under NCCL's prefix, as a communicator's proxy
//! does.
#![allow(clippy::expect_used, clippy::unwrap_used)]

use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use uniserve_ipc::launch::RankChild;

/// A stand-in rank: creates an interprocess event on device 0, which makes
/// the driver create its shared memory named for this process, and waits to
/// be killed.
const EVENT_HOLDER: &str = "\
import ctypes, time
cuda = ctypes.CDLL('libcuda.so.1')
device, context, event = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_void_p()
assert cuda.cuInit(0) == 0
assert cuda.cuDeviceGet(ctypes.byref(device), 0) == 0
assert cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device) == 0
assert cuda.cuCtxPushCurrent_v2(context) == 0
# CU_EVENT_DISABLE_TIMING | CU_EVENT_INTERPROCESS
assert cuda.cuEventCreate(ctypes.byref(event), 2 | 4) == 0
time.sleep(600)
";

/// The interprocess-event memory the driver created for process `pid`, by
/// its name in `/dev/shm` (`cuda.shm.<uid in hex>.<pid in hex>.<n>`).
fn event_memory(pid: u32) -> Vec<PathBuf> {
    // SAFETY: getuid has no preconditions.
    let prefix = format!("cuda.shm.{:x}.{pid:x}.", unsafe { libc::getuid() });
    std::fs::read_dir("/dev/shm")
        .unwrap()
        .map(|entry| entry.unwrap())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with(&prefix))
        .map(|entry| entry.path())
        .collect()
}

/// Kills and reaps a holder when a test ends, including when it panics.
struct Holder(RankChild);

impl Holder {
    /// Starts a holder and waits until its event memory exists.
    fn start() -> (Self, Vec<PathBuf>) {
        let mut command = Command::new("python3");
        command.args(["-c", EVENT_HOLDER]);
        let holder = Self(RankChild::spawn(&mut command).unwrap());

        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let memory = event_memory(holder.0.id());
            if !memory.is_empty() {
                return (holder, memory);
            }
            assert!(
                Instant::now() < deadline,
                "the holder created no event memory"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

impl Drop for Holder {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A rank killed by someone else is reaped by polling, as the head and a
/// host's launcher poll their ranks, and one killed by its parent is reaped
/// by waiting: either way no memory named for the dead rank remains, while
/// a live rank's stays.
#[test]
#[ignore = "requires a CUDA device"]
fn a_killed_rank_leaves_no_event_memory_once_reaped() {
    let (mut killed, killed_memory) = Holder::start();
    let (mut live, live_memory) = Holder::start();

    // SAFETY: signal delivery to this test's own child, which is unreaped.
    assert_eq!(
        unsafe { libc::kill(killed.0.id() as libc::pid_t, libc::SIGKILL) },
        0
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    let status = loop {
        if let Some(status) = killed.0.try_wait().unwrap() {
            break status;
        }
        assert!(Instant::now() < deadline, "the killed rank was not reaped");
        std::thread::sleep(Duration::from_millis(10));
    };
    assert_eq!(status.signal(), Some(libc::SIGKILL));
    assert!(
        killed_memory.iter().all(|path| !path.exists()),
        "{killed_memory:?} outlived its rank"
    );
    assert!(
        live_memory.iter().all(|path| path.exists()),
        "a live rank lost {live_memory:?}"
    );

    live.0.kill().unwrap();
    assert_eq!(live.0.wait().unwrap().signal(), Some(libc::SIGKILL));
    assert!(
        live_memory.iter().all(|path| !path.exists()),
        "{live_memory:?} outlived its rank"
    );
}

/// A stand-in rank that maps shared memory under NCCL's prefix, as a
/// communicator's proxy maps its pool, writes the object's path to the file
/// its argument names and waits to be killed.
const NCCL_HOLDER: &str = "\
import mmap, os, sys, tempfile, time
fd, path = tempfile.mkstemp(prefix='nccl-', dir='/dev/shm')
os.ftruncate(fd, 4096)
pool = mmap.mmap(fd, 4096)
with open(sys.argv[1] + '.tmp', 'w') as file:
    file.write(path)
os.rename(sys.argv[1] + '.tmp', sys.argv[1])
time.sleep(600)
";

/// Starts an [`NCCL_HOLDER`] and returns it with its object's path, once
/// the object is mapped.
fn hold_nccl(directory: &std::path::Path, name: &str) -> (Holder, PathBuf) {
    let path_file = directory.join(name);
    let mut command = Command::new("python3");
    command.args(["-c", NCCL_HOLDER]).arg(&path_file);
    let holder = Holder(RankChild::spawn(&mut command).unwrap());
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Ok(path) = std::fs::read_to_string(&path_file) {
            return (holder, PathBuf::from(path));
        }
        assert!(Instant::now() < deadline, "the holder mapped nothing");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// A rank killed by someone else leaves its NCCL names, which carry no
/// process id; its parent noted the names it mapped while it ran and
/// removes them when it reaps it, while a live rank's stay.
#[test]
fn a_killed_rank_leaves_none_of_its_nccl_names_once_reaped() {
    let directory = tempfile::tempdir().unwrap();
    let (mut killed, killed_path) = hold_nccl(directory.path(), "killed");
    let (mut live, live_path) = hold_nccl(directory.path(), "live");
    killed.0.note_nccl_segments().unwrap();
    live.0.note_nccl_segments().unwrap();

    // SAFETY: signal delivery to this test's own child, which is unreaped.
    assert_eq!(
        unsafe { libc::kill(killed.0.id() as libc::pid_t, libc::SIGKILL) },
        0
    );
    let deadline = Instant::now() + Duration::from_secs(30);
    while killed.0.try_wait().unwrap().is_none() {
        assert!(Instant::now() < deadline, "the killed rank was not reaped");
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(!killed_path.exists(), "{killed_path:?} outlived its rank");
    assert!(live_path.exists(), "a live rank lost {live_path:?}");

    live.0.kill().unwrap();
    live.0.wait().unwrap();
    assert!(!live_path.exists(), "{live_path:?} outlived its rank");
}
