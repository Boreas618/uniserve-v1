//! The launch file a main process is started with (`uniserve engine --launch
//! <path>`, `specs/worker.md` §2.3, §7.4).
//!
//! The engine takes the file's contents as its whole description: one flag,
//! no second account of what the process is or which build it serves. The
//! daemon therefore owns both files the process reads — the epoch's build
//! config, written once per epoch, and this process's launch — and writes each
//! one atomically: a temporary file beside the target, synced, then renamed.
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

/// The path of one epoch's build config, which every process of that epoch
/// reads.
pub(crate) fn engine_config_path(directory: &Path, world: u64) -> PathBuf {
    directory.join(format!("engine-config-{world}.json"))
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

/// Writes `config` as the epoch's build config and answers the path it was
/// written to.
///
/// Two engines of one epoch read the same bytes, so the file is named for the
/// epoch's world rather than for a process.
///
/// # Errors
///
/// Fails when the directory or the file cannot be written.
pub(crate) fn write_engine_config(
    directory: &Path,
    world: u64,
    config: &proto::config::EngineConfig,
) -> io::Result<PathBuf> {
    let path = engine_config_path(directory, world);
    let bytes = serde_json::to_vec_pretty(config)
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

    fn launch() -> EngineArgs {
        EngineArgs {
            engine_config: PathBuf::from("/run/uniserve/engine-config-12.json"),
            model: PathBuf::from("/models/qwen3"),
            requests: "unix:/run/uniserve/requests.sock".to_owned(),
            processes: "unix:/run/uniserve/processes.sock".to_owned(),
            engine: 7,
            program_server: "/run/uniserve/programs.sock".to_owned(),
            devices: vec![0, 1],
            group: "engine".to_owned(),
            rank: 0,
            deployment: "qwen3".to_owned(),
            world: 12,
            scheduler_node: true,
            peers: Vec::new(),
            watchdog_timeout_s: None,
            rollback_s: None,
            queue_depth: 4,
            world_size: 1,
            standby: false,
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

    /// The launch file states exactly the engine's fields, and the engine reads
    /// it back as the one type both sides use (`proto::config::EngineArgs`).
    /// A key a newer daemon adds is read past rather than refused, so an
    /// engine of an older build still starts from it.
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
            "model",
            "requests",
            "processes",
            "engine",
            "program_server",
            "devices",
            "group",
            "rank",
            "deployment",
            "world",
            "scheduler_node",
            "peers",
            "watchdog_timeout_s",
            "rollback_s",
            "queue_depth",
            "world_size",
            "standby",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
        assert_eq!(keys, expected);
        assert_eq!(value["engine"], json!(7));
        assert_eq!(value["world"], json!(12));
        assert_eq!(value["devices"], json!([0, 1]));
        assert_eq!(value["queue_depth"], json!(4));
        assert_eq!(value["world_size"], json!(1));
        assert_eq!(value["standby"], json!(false));

        // The engine reads back what the writer wrote.
        let read: EngineArgs = serde_json::from_str(&text).expect("the launch file reads back");
        assert_eq!(read, launch());

        // A key this build does not know is read past.
        let mut extra = value;
        extra["generation"] = json!(7);
        let read: EngineArgs = serde_json::from_value(extra).expect("an unknown key is read past");
        assert_eq!(read, launch());
        let _ = std::fs::remove_dir_all(&directory);
    }

    /// The build config is one file per epoch, its bytes the epoch's, and it
    /// is written whole.
    #[test]
    fn the_engine_config_is_written_once_per_epoch() {
        let directory = scratch("build");
        let config = ServerArgs::default()
            .engine_config()
            .expect("the configuration resolves");
        let path = write_engine_config(&directory, 12, &config).expect("the engine config");
        assert_eq!(
            path.file_name().and_then(|name| name.to_str()),
            Some("engine-config-12.json")
        );
        let read: proto::config::EngineConfig =
            serde_json::from_slice(&std::fs::read(&path).expect("the file")).expect("its JSON");
        assert_eq!(read, config);

        // A second engine of the same epoch finds the same file, unchanged.
        let again = write_engine_config(&directory, 12, &config).expect("the same file");
        assert_eq!(again, path);
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
