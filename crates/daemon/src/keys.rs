//! The kernel keys a deployment's main processes launch, recorded by a start that serves nothing
//! (`uniserve keys`, `uniserve-engine --keys`).
//!
//! The launches are the ones a serving leader starts its main processes with: the model's facts
//! resolved into the engine configuration ([`Leader::prepare`]), one configuration per independent
//! replica (`replica_configs`), and each process placed on this node
//! (`crate::engine::control::place`). Each launch is run by the engine package the deployment
//! selects, resolved at this host's Build Server as a serving start resolves it, up to the profile
//! run of every device's runners; the daemon's request and process sockets are left unstated,
//! because a recorded start connects to neither.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::{Command, Stdio};

use anyhow::Context as _;
use proto::config::EngineArgs;

use crate::engine::control::{Interfaces, ensure_build_server, place, select_engine};
use crate::leader::{Leader, replica_configs};
use crate::options::{ServerArgs, Stated};

/// The launch of every main process the deployment `config` starts on this node, in replica
/// order; `stated` holds the option paths the operator set, and `families` the families whose
/// requests this build processes.
///
/// # Errors
///
/// Fails where a serving leader's start fails before it launches a process: the model's assets do
/// not resolve, an option of another family was stated, the options do not resolve into an engine
/// configuration, the replicas are not independent, or a process cannot be placed on this node.
pub async fn launches(
    families: &processing::Catalogue,
    config: &ServerArgs,
    stated: &Stated,
) -> anyhow::Result<Vec<EngineArgs>> {
    let prepared = Leader::prepare(families, config, stated).await?;
    replica_configs(&prepared.engine)?
        .into_iter()
        .zip(1..)
        .map(|(replica, engine)| {
            // A serving daemon numbers its main processes from one as it starts them; the number
            // keys nothing a recorded start reads.
            place(
                EngineArgs::new(replica),
                engine,
                &Interfaces {
                    requests: "",
                    processes: "",
                    build_server: Path::new(""),
                },
            )
        })
        .collect()
}

/// The kernel keys the deployment `config` launches, by operator: what every main process of it
/// records in its profile runs, each key once.
///
/// The engine package the deployment selects is fixed and built at this host's Build Server,
/// which is started when none serves, and each launch is run by its executable with `--keys`.
/// No weight is read, and no daemon, Storage Server or communicator is started.
///
/// # Errors
///
/// Fails as [`launches`] does, when the engine package cannot be selected or resolved, and when
/// a recording process fails or prints a line that is no `operator<TAB>key`.
pub async fn record(
    families: &processing::Catalogue,
    config: &ServerArgs,
    stated: &Stated,
) -> anyhow::Result<BTreeMap<String, BTreeSet<String>>> {
    let launches = launches(families, config, stated).await?;
    let config = config.clone();
    tokio::task::spawn_blocking(move || {
        let socket = ensure_build_server(&config.build_config)?;
        let mut modules = crate::epoch::modules_of(&config);
        modules.engine = select_engine(&socket, &modules.engine)?;

        let directory = tempfile::tempdir().context("creating the launch files' directory")?;
        let mut recorded: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for mut launch in launches {
            launch.build_server = socket.display().to_string();
            launch.modules = Some(modules.clone());
            let executable =
                crate::supervision::engine_executable(&launch).map_err(anyhow::Error::msg)?;
            let file = crate::engine::launch::write(directory.path(), launch.engine, &launch)
                .context("writing a launch file")?;

            // The process's logs are this command's: they go to its standard error.
            let output = Command::new(&executable)
                .arg("--launch")
                .arg(&file)
                .arg("--keys")
                .stderr(Stdio::inherit())
                .output()
                .with_context(|| format!("running {}", executable.display()))?;
            anyhow::ensure!(
                output.status.success(),
                "{} --keys ended with {}",
                executable.display(),
                output.status
            );
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                let (operator, key) = line
                    .split_once('\t')
                    .with_context(|| format!("{line:?} is no `operator<TAB>key` line"))?;
                recorded
                    .entry(operator.to_owned())
                    .or_default()
                    .insert(key.to_owned());
            }
        }
        Ok(recorded)
    })
    .await
    .context("the recording did not complete")?
}
