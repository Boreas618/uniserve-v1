//! The node's GPU registration: which deployment each GPU of the node belongs to.
//!
//! A GPU belongs to one deployment. Every node keeps one registration file for all deployments
//! of every user ([`NODE`]), and the daemon of a deployment claims each GPU it starts an engine
//! on there before the engine starts ([`Gpus::claim`]): a GPU another deployment holds is
//! refused, naming the holder, and the engine does not start. The only sharing left on a GPU is
//! between two epochs of one deployment, whose quota the deployment's Storage Server keeps.
//!
//! The file holds one slot of [`SLOT`] bytes per GPU, at the GPU's node ordinal: a claim is an
//! open file description lock (`F_OFD_SETLK`) on the GPU's slot, and the slot's bytes name the
//! holder. The kernel releases a lock when the last descriptor of its open file description
//! closes, so a claim ends with the process that holds it, however it ends, and a slot's bytes
//! name a holder only while its lock is held. A hot restart hands the file's descriptor to the
//! daemon's successor, which goes on holding the claims ([`Gpus::hand_over`], [`Gpus::adopt`]).
//!
//! A GPU is named by its ordinal among the node's GPUs: the CUDA ordinal a launch states, read
//! through `CUDA_VISIBLE_DEVICES` when the daemon's environment narrows the devices it sees, so
//! every deployment names one physical GPU alike.

use std::collections::BTreeSet;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::fd::{AsRawFd as _, OwnedFd};
use std::os::unix::fs::{FileExt as _, OpenOptionsExt as _, PermissionsExt as _};
use std::path::{Path, PathBuf};

/// The node's registration file: one for the deployments of every user of the node, in the
/// system's temporary directory, which every user writes.
pub(crate) const NODE: &str = "uniserve-gpus";

/// The bytes of one GPU's slot: the holder's description, padded with NULs.
const SLOT: u64 = 256;

/// The mode of the registration file: every user's daemons claim in it.
const SHARED: u32 = 0o666;

/// The node's registration file, at [`NODE`] in the system's temporary directory.
pub(crate) fn node() -> PathBuf {
    std::env::temp_dir().join(NODE)
}

/// A daemon's claims of its node's GPUs for its deployment.
#[derive(Debug)]
pub(crate) struct Gpus {
    /// The registration file, opened once: its open file description holds the claims.
    file: File,
    /// How a slot this daemon holds names it.
    holder: String,
    /// The GPUs this daemon holds, by node ordinal.
    held: BTreeSet<u32>,
}

/// A GPU another deployment holds.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("GPU {gpu} belongs to another deployment ({holder}): a GPU serves one deployment")]
pub(crate) struct Held {
    /// The GPU, by node ordinal.
    pub(crate) gpu: u32,
    /// The holder as its slot names it.
    pub(crate) holder: String,
}

/// What a hot restart hands the successor of the claims: the GPUs held. The file's descriptor
/// travels beside it.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct Handoff {
    holder: String,
    held: BTreeSet<u32>,
}

impl Gpus {
    /// The claims of `deployment` in the registration file `path`, created readable and
    /// writable by every user when absent; none is held yet.
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be opened or created.
    pub(crate) fn open(path: &Path, deployment: &str) -> io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .mode(SHARED)
            .open(path)?;
        // The creating process's umask narrows the mode it was created with; its owner widens
        // it again so every user's daemons claim in it. Another user's file is left as it is.
        let owner = rustix::process::getuid().as_raw();
        let metadata = file.metadata()?;
        if std::os::unix::fs::MetadataExt::uid(&metadata) == owner
            && metadata.permissions().mode() & 0o777 != SHARED
        {
            file.set_permissions(std::fs::Permissions::from_mode(SHARED))?;
        }
        Ok(Self {
            file,
            holder: format!(
                "deployment {deployment} of user {owner}, daemon process {}",
                std::process::id()
            ),
            held: BTreeSet::new(),
        })
    }

    /// Claims `gpus`, by node ordinal, for this deployment: the GPUs it holds already are kept,
    /// and each other is locked and named this daemon's.
    ///
    /// # Errors
    ///
    /// [`Held`] for the first GPU another deployment holds; the GPUs claimed before it stay
    /// claimed. Another failure of the file as [`io::Error`] in [`Refusal::Io`].
    pub(crate) fn claim(&mut self, gpus: &[u32]) -> Result<(), Refusal> {
        for &gpu in gpus {
            if self.held.contains(&gpu) {
                continue;
            }
            if !lock(&self.file, gpu)? {
                return Err(Refusal::Held(Held {
                    gpu,
                    holder: self.named(gpu)?,
                }));
            }
            let mut slot = self.holder.as_bytes().to_vec();
            slot.resize(SLOT as usize, 0);
            self.file.write_all_at(&slot, offset(gpu))?;
            self.held.insert(gpu);
            tracing::info!(gpu, "claimed a GPU for the deployment");
        }
        Ok(())
    }

    /// The GPUs this daemon holds, by node ordinal.
    #[cfg(test)]
    pub(crate) fn held(&self) -> &BTreeSet<u32> {
        &self.held
    }

    /// The holder `gpu`'s slot names.
    fn named(&self, gpu: u32) -> io::Result<String> {
        let mut slot = vec![0; SLOT as usize];
        let read = self.file.read_at(&mut slot, offset(gpu))?;
        slot.truncate(read);
        let named = slot.split(|&byte| byte == 0).next().unwrap_or_default();
        Ok(String::from_utf8_lossy(named).into_owned())
    }

    /// The claims for a successor to go on holding ([`Gpus::adopt`]), and the file's descriptor,
    /// whose open file description holds the locks.
    ///
    /// # Errors
    ///
    /// Fails when the descriptor cannot be duplicated.
    pub(crate) fn hand_over(&self) -> io::Result<(Handoff, OwnedFd)> {
        Ok((
            Handoff {
                holder: self.holder.clone(),
                held: self.held.clone(),
            },
            OwnedFd::from(self.file.try_clone()?),
        ))
    }

    /// Goes on with the claims a predecessor handed over, held by the open file description of
    /// `descriptor`.
    pub(crate) fn adopt(handoff: Handoff, descriptor: OwnedFd) -> Self {
        Self {
            file: File::from(descriptor),
            holder: handoff.holder,
            held: handoff.held,
        }
    }
}

/// Why a claim was refused.
#[derive(Debug, thiserror::Error)]
pub(crate) enum Refusal {
    #[error(transparent)]
    Held(#[from] Held),
    #[error("the GPU registration file failed: {0}")]
    Io(#[from] io::Error),
}

/// Where `gpu`'s slot begins.
fn offset(gpu: u32) -> u64 {
    u64::from(gpu) * SLOT
}

/// Takes the open file description lock of `gpu`'s slot in `file` without waiting; whether it
/// was taken, `false` when another open file description holds it.
fn lock(file: &File, gpu: u32) -> io::Result<bool> {
    // SAFETY: `flock` is a plain C struct of integers, for which all zero bytes are a value.
    let mut range: libc::flock = unsafe { std::mem::zeroed() };
    range.l_type = libc::F_WRLCK as libc::c_short;
    range.l_whence = libc::SEEK_SET as libc::c_short;
    range.l_start = libc::off_t::try_from(offset(gpu)).map_err(io::Error::other)?;
    range.l_len = SLOT as libc::off_t;
    loop {
        // SAFETY: fcntl reads the lock description it is given, on a descriptor `file` owns.
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_OFD_SETLK, &range) } == 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        match error.raw_os_error() {
            Some(libc::EAGAIN | libc::EACCES) => return Ok(false),
            Some(libc::EINTR) => {}
            _ => return Err(error),
        }
    }
}

/// The node ordinal of the GPU a launch names by `ordinal`: the ordinal itself, or, when the
/// daemon's environment narrows the devices it sees (`CUDA_VISIBLE_DEVICES`), the device that
/// entry names.
///
/// # Errors
///
/// Fails when the variable lists no such entry, or names it otherwise than by an ordinal.
pub(crate) fn physical(ordinal: u32) -> Result<u32, String> {
    let Ok(visible) = std::env::var("CUDA_VISIBLE_DEVICES") else {
        return Ok(ordinal);
    };
    let entry = visible
        .split(',')
        .map(str::trim)
        .nth(ordinal as usize)
        .ok_or_else(|| {
            format!("CUDA_VISIBLE_DEVICES ({visible}) lists no device of ordinal {ordinal}")
        })?;
    entry.parse().map_err(|_| {
        format!(
            "CUDA_VISIBLE_DEVICES names device {entry} otherwise than by its ordinal, which the \
             node's GPU registration names GPUs by"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GPU belongs to the one deployment that claimed it: another deployment's claim of it is
    /// refused, naming the holder, while the GPUs no one holds are claimed, and a deployment
    /// claims its own GPUs again.
    #[test]
    fn a_second_deployment_is_refused_a_held_gpu_naming_the_holder() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(NODE);
        let mut first = Gpus::open(&path, "first").unwrap();
        let mut second = Gpus::open(&path, "second").unwrap();

        first.claim(&[0, 1]).unwrap();
        first.claim(&[1, 0]).unwrap();
        let Err(Refusal::Held(held)) = second.claim(&[2, 1]) else {
            panic!("a held GPU was claimed again");
        };
        assert_eq!(held.gpu, 1);
        assert!(held.holder.contains("deployment first"), "{}", held.holder);
        assert!(held.to_string().contains("deployment first"), "{held}");
        // The GPU before the held one was claimed.
        assert_eq!(second.held(), &BTreeSet::from([2]));
        assert!(matches!(first.claim(&[2]), Err(Refusal::Held(_))));
    }

    /// A claim held by another process is refused, and ends when that process exits.
    #[test]
    fn a_claim_of_another_process_ends_with_its_exit() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(NODE);
        // A process of its own takes the lock of GPU 0's slot as a daemon's claim does, and
        // holds it until it reads its standard input's end.
        let mut holder = std::process::Command::new("python3")
            .arg("-c")
            .arg(
                "import fcntl, os, struct, sys\n\
                 fd = os.open(sys.argv[1], os.O_RDWR)\n\
                 os.pwrite(fd, b'deployment other', 0)\n\
                 fcntl.fcntl(fd, 37, struct.pack('hhqqi', fcntl.F_WRLCK, 0, 0, 256, 0))\n\
                 print('held', flush=True)\n\
                 sys.stdin.read()\n",
            )
            .arg(&path)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        // The file exists before the holder opens it.
        let mut claims = Gpus::open(&path, "this").unwrap();
        let mut line = String::new();
        std::io::BufRead::read_line(
            &mut std::io::BufReader::new(holder.stdout.take().unwrap()),
            &mut line,
        )
        .unwrap();
        assert_eq!(line.trim(), "held");

        let Err(Refusal::Held(held)) = claims.claim(&[0]) else {
            panic!("a GPU another process holds was claimed");
        };
        assert_eq!(held.holder, "deployment other");

        drop(holder.stdin.take());
        assert!(holder.wait().unwrap().success());
        claims.claim(&[0]).unwrap();
    }

    /// A hot restart's successor holds the claims of the descriptor it was handed: the claims
    /// outlive the predecessor's own descriptor. (They end with the successor's process, as
    /// every claim ends with its holder's exit; a descriptor closed inside a process that forks
    /// may still be held by a child between its fork and its exec, so a release is observed at a
    /// process's exit, not at a drop.)
    #[test]
    fn a_successor_goes_on_with_the_claims() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(NODE);
        let mut predecessor = Gpus::open(&path, "first").unwrap();
        predecessor.claim(&[0]).unwrap();
        let (handoff, descriptor) = predecessor.hand_over().unwrap();
        drop(predecessor);
        let successor = Gpus::adopt(handoff, descriptor);
        assert_eq!(successor.held(), &BTreeSet::from([0]));

        let mut other = Gpus::open(&path, "second").unwrap();
        assert!(other.claim(&[0]).is_err());
    }

    /// A launch's ordinal names the device the daemon's visible devices list at it.
    #[test]
    fn an_ordinal_names_the_physical_gpu() {
        // The test's environment states no narrowing of its devices, or one by ordinals.
        if let Ok(visible) = std::env::var("CUDA_VISIBLE_DEVICES") {
            let first = visible.split(',').next().unwrap().trim();
            if let Ok(first) = first.parse::<u32>() {
                assert_eq!(physical(0), Ok(first));
            }
        } else {
            assert_eq!(physical(5), Ok(5));
        }
    }
}
