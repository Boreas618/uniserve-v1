//! The launch an engine process is started with (`uniserve-engine --launch <fd>`): one
//! inherited descriptor of a sealed anonymous file holding the process's [`EngineArgs`].
//!
//! The engine takes the launch as its whole description: one option, no second account of what
//! the process is or which build it serves. The launch carries the epoch's engine configuration
//! inline, so the process reads nothing else the daemon wrote. It lives in no file system: the
//! daemon of the node that starts the engine writes it into a memfd, seals the memfd so its
//! contents can no longer change, and passes the descriptor to the process it starts
//! ([`inherit`]), which reads it whole in its first instruction; the memory goes with the last
//! descriptor. The launch's contents travel to that daemon in the spawn request
//! (`process::Spawn::launch`), so an engine on any node reads a launch of its own host.

use std::io::{self, Seek as _, Write as _};
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd};
use std::os::unix::process::CommandExt as _;
use std::process::Command;

use proto::config::EngineArgs;

/// The seals that make a launch's memfd immutable: no write, no change of size, and no further
/// change of the seals themselves.
const SEALS: libc::c_int =
    libc::F_SEAL_WRITE | libc::F_SEAL_SHRINK | libc::F_SEAL_GROW | libc::F_SEAL_SEAL;

/// A sealed memfd that holds `launch`, positioned at its start, for the process it starts to read
/// ([`inherit`]). The descriptor is close-on-exec in this process.
///
/// # Errors
///
/// Fails when the launch does not encode, or the memfd cannot be created, written or sealed.
pub(crate) fn memfd(launch: &EngineArgs) -> io::Result<OwnedFd> {
    let bytes = serde_json::to_vec(launch)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    // SAFETY: the name is a NUL-terminated literal; a successful call returns a descriptor this
    // function owns.
    let descriptor = unsafe {
        libc::memfd_create(
            c"uniserve-launch".as_ptr(),
            libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
        )
    };
    if descriptor < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the descriptor was just created and is owned by nothing else.
    let mut file = std::fs::File::from(unsafe { OwnedFd::from_raw_fd(descriptor) });
    file.write_all(&bytes)?;
    file.rewind()?;
    // SAFETY: fcntl on the descriptor `file` owns, with an integer argument.
    if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, SEALS) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(OwnedFd::from(file))
}

/// Has the process `command` starts inherit `launch` at its descriptor number, named by the
/// engine's launch option (`proto::LAUNCH_OPTION`) on its command line.
///
/// The command owns `launch` from here on and closes it when the command is dropped, so the
/// memfd stays open until the spawn has duplicated it. Close-on-exec is cleared only in the
/// forked child: no other process this daemon starts meanwhile inherits the launch.
pub(crate) fn inherit(command: &mut Command, launch: OwnedFd) {
    let descriptor = launch.as_raw_fd();
    command
        .arg(proto::LAUNCH_OPTION)
        .arg(descriptor.to_string());
    // SAFETY: the closure runs in the forked child between fork and exec and only calls
    // `fcntl`, which is async-signal-safe. `launch` is moved into the closure, so the descriptor
    // is open in the parent at fork time.
    unsafe {
        command.pre_exec(move || {
            if libc::fcntl(launch.as_raw_fd(), libc::F_SETFD, 0) == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::io::Read as _;

    use serde_json::{Value, json};

    use super::*;
    use crate::config::Config;

    /// The launch of a standby (engine 7) of epoch 12 beside the process of epoch 11 it succeeds,
    /// with a rebalanced expert order.
    fn launch() -> EngineArgs {
        use proto::request::wire::{ExpertOrder, Predecessor};
        EngineArgs {
            engine_config: Config::default()
                .engine_config()
                .expect("the configuration resolves"),
            deployment: "model".to_owned(),
            epoch: 12,
            revision: 3,
            predecessor: Some(Predecessor { epoch: 11 }),
            expert_order: [(
                "model.layers.0.mlp".to_owned(),
                ExpertOrder {
                    experts: vec![1, 0],
                },
            )]
            .into_iter()
            .collect(),
            host: "rank-0".to_owned(),
            // The interfaces are data the engine reads; no socket is opened here.
            requests: "unix:requests.sock".to_owned(),
            processes: "unix:processes.sock".to_owned(),
            engine: 7,
            build_server: "build-server.sock".to_owned(),
            modules: crate::epoch::modules_of(&crate::config::Config::default()),
            devices: vec![0, 1],
            group: "engine".to_owned(),
            rank: 0,
            scheduler_node: true,
            world: None,
            standby_storage: crate::config::StandbyStorage::Resident,
            queue_depth: 4,
            world_size: 1,
        }
    }

    /// The launch states exactly the engine's fields, the epoch's engine
    /// configuration and the handover state among them, and the engine reads
    /// it back as the one type both sides use (`proto::config::EngineArgs`).
    /// A key the engine does not know is refused.
    #[test]
    fn a_launch_states_exactly_the_engines_fields() {
        let mut text = String::new();
        std::fs::File::from(memfd(&launch()).expect("the launch"))
            .read_to_string(&mut text)
            .expect("the launch reads");
        let value: Value = serde_json::from_str(&text).expect("JSON");
        let keys = value
            .as_object()
            .expect("an object")
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        let expected = [
            "engine_config",
            "epoch",
            "revision",
            "predecessor",
            "expert_order",
            "host",
            "requests",
            "processes",
            "engine",
            "build_server",
            "modules",
            "devices",
            "group",
            "rank",
            "scheduler_node",
            "world",
            "standby_storage",
            "queue_depth",
            "world_size",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
        assert_eq!(keys, expected);
        assert_eq!(value["engine"], json!(7));
        assert_eq!(value["epoch"], json!(12));
        assert_eq!(value["devices"], json!([0, 1]));
        assert_eq!(value["queue_depth"], json!(4));
        assert_eq!(value["world_size"], json!(1));
        assert_eq!(value["predecessor"]["epoch"], json!(11));
        assert_eq!(
            value["expert_order"]["model.layers.0.mlp"]["experts"],
            json!([1, 0])
        );
        assert!(value["engine_config"].is_object());

        // The engine reads back what the writer wrote.
        let read: EngineArgs = serde_json::from_str(&text).expect("the launch reads back");
        assert_eq!(read, launch());

        // A key this build does not know is refused.
        let mut extra = value;
        extra["generation"] = json!(7);
        serde_json::from_value::<EngineArgs>(extra).expect_err("an unknown key is refused");
    }

    /// A launch cannot change once written: the memfd refuses a write and a change of its size.
    #[test]
    fn a_launch_is_sealed() {
        let mut file = std::fs::File::from(memfd(&launch()).expect("the launch"));
        file.write_all(b"{}")
            .expect_err("a sealed launch refuses a write");
        file.set_len(0)
            .expect_err("a sealed launch refuses a change of its size");
    }
}
