//! The launch file a main process is started with (`uniserve engine --launch
//! <path>`, `specs/worker.md` §2.3, §7.4).
//!
//! The engine takes the file's contents as its whole description: one flag,
//! no second account of what the process is or which build it serves. The
//! file carries the epoch's engine configuration inline, so the process reads
//! nothing else the daemon wrote, and the daemon writes it atomically: a
//! temporary file beside the target, synced, then renamed.
//! The child reads it in its first instruction, so nothing may be observable
//! half-written.
//!
//! The writer is only used where the child shares this daemon's runtime
//! directory. A child on another host cannot open a path that exists only
//! here; that spawn is refused, and the general form (the launch contents
//! travelling in `process::Spawn`) is a protocol change this module does not
//! make.

use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use proto::config::EngineArgs;

/// The path of one engine process's launch file.
pub(crate) fn launch_path(directory: &Path, engine: u64) -> PathBuf {
    directory.join(format!("launch-{engine}.json"))
}

/// Writes `launch` for the process `engine` and answers the path it was
/// written to.
///
/// # Errors
///
/// Fails when the directory or the file cannot be written.
pub(crate) fn write(directory: &Path, engine: u64, launch: &EngineArgs) -> io::Result<PathBuf> {
    let path = launch_path(directory, engine);
    let bytes = serde_json::to_vec_pretty(launch)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
    write_atomically(&path, &bytes)?;
    Ok(path)
}

/// Writes `bytes` to `path` so that a reader sees either nothing or the whole
/// file: a temporary file in the same directory, synced, renamed into place.
fn write_atomically(path: &Path, bytes: &[u8]) -> io::Result<()> {
    /// Distinguishes the temporary files of one process writing twice in the
    /// same microsecond.
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let directory = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(directory)?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("launch");
    let temporary = directory.join(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let written = (|| {
        let mut file = std::fs::File::create(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        // The rename is the first moment the file exists under its real name,
        // and it is atomic, so a child either reads the whole file or is
        // started after a failure this call reports.
        std::fs::rename(&temporary, path)
    })();
    if written.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    written
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use serde_json::{Value, json};

    use super::*;
    use crate::options::ServerArgs;

    /// The launch of a standby (engine 7) of epoch 12 beside the process of epoch 11 it succeeds,
    /// with a rebalanced expert order.
    fn launch() -> EngineArgs {
        use proto::request::wire::{CacheSegment, CacheSegments, ExpertOrder, Predecessor};
        EngineArgs {
            engine_config: ServerArgs::default()
                .engine_config()
                .expect("the configuration resolves"),
            epoch: 12,
            predecessor: Some(Predecessor {
                epoch: 11,
                segments: vec![CacheSegments {
                    worker_group: "engine".to_owned(),
                    segments: vec![CacheSegment {
                        epoch: 11,
                        ordinal: 0,
                        units: 64,
                    }],
                    held: 1,
                    units: 64,
                    scratch: 0,
                }],
            }),
            expert_order: [(
                "model.layers.0.mlp".to_owned(),
                ExpertOrder {
                    experts: vec![1, 0],
                },
            )]
            .into_iter()
            .collect(),
            host: "rank-0".to_owned(),
            requests: "unix:/run/uniserve/requests.sock".to_owned(),
            processes: "unix:/run/uniserve/processes.sock".to_owned(),
            engine: 7,
            program_server: "/run/uniserve/programs.sock".to_owned(),
            devices: vec![0, 1],
            group: "engine".to_owned(),
            rank: 0,
            scheduler_node: true,
            peers: Vec::new(),
            watchdog_timeout_s: None,
            rollback_s: None,
            queue_depth: 4,
            world_size: 1,
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "uniserve-launch-{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos())
                .unwrap_or_default()
        ));
        std::fs::create_dir_all(&directory).expect("a scratch directory");
        directory
    }

    /// The launch file states exactly the engine's fields, the epoch's engine
    /// configuration and the handover state among them, and the engine reads
    /// it back as the one type both sides use (`proto::config::EngineArgs`).
    /// A key the engine does not know is refused.
    #[test]
    fn a_launch_file_states_exactly_the_engines_fields() {
        let directory = scratch("keys");
        let path = write(&directory, 7, &launch()).expect("the launch file");
        let text = std::fs::read_to_string(&path).expect("the written file");
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
            "predecessor",
            "expert_order",
            "host",
            "requests",
            "processes",
            "engine",
            "program_server",
            "devices",
            "group",
            "rank",
            "scheduler_node",
            "peers",
            "watchdog_timeout_s",
            "rollback_s",
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
        let read: EngineArgs = serde_json::from_str(&text).expect("the launch file reads back");
        assert_eq!(read, launch());

        // A key this build does not know is refused.
        let mut extra = value;
        extra["generation"] = json!(7);
        serde_json::from_value::<EngineArgs>(extra).expect_err("an unknown key is refused");
        let _ = std::fs::remove_dir_all(&directory);
    }

    /// The launch file is per process: a standby and a serving engine of one
    /// epoch write different names in one directory.
    #[test]
    fn a_launch_file_is_named_for_its_process() {
        let directory = scratch("names");
        let mut standby = launch();
        standby.engine = 8;
        let serving = write(&directory, 7, &launch()).expect("the serving launch");
        let standing = write(&directory, 8, &standby).expect("the standby launch");
        assert_ne!(serving, standing);
        assert_eq!(
            serving.file_name().and_then(|name| name.to_str()),
            Some("launch-7.json")
        );
        assert_eq!(
            standing.file_name().and_then(|name| name.to_str()),
            Some("launch-8.json")
        );
        // No temporary file survives a completed write.
        let leftovers = std::fs::read_dir(&directory)
            .expect("the directory")
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .count();
        assert_eq!(leftovers, 0);
        let _ = std::fs::remove_dir_all(&directory);
    }
}
