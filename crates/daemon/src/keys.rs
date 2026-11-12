//! The kernel keys a deployment's engines launch, recorded by a start that serves nothing
//! (`uniserve keys`, `uniserve-engine --keys`).
//!
//! The launches are the ones a serving leader starts its engines with: the model's facts
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

use crate::engine::control::{BuildServer, Interfaces, ensure_build_server, place};
use crate::leader::{Leader, fix_modules, named_modules, replica_configs};
use crate::options::{ServerArgs, Stated};

/// The launch of every engine the deployment `config` starts on this node, in replica
/// order; `stated` holds the option paths the operator set, and `families` the families whose
/// requests this build processes.
///
/// # Errors
///
/// Fails where a serving leader's start fails before it launches a process: the model's assets do
/// not resolve, an option of another family was stated, the options do not resolve into an engine
/// configuration, the replicas are not independent, or a process cannot be placed on this node.
pub async fn launches(
    families: &processing::Registry,
    config: &ServerArgs,
    stated: &Stated,
) -> anyhow::Result<Vec<EngineArgs>> {
    let prepared = Leader::prepare(families, config, stated).await?;
    // The module set is fixed when the keys are recorded (`record`): a launch built here names
    // the packages the configuration names.
    let named = named_modules(config, prepared.model_config.video().is_some());
    replica_configs(&prepared.engine)?
        .into_iter()
        .zip(1..)
        .map(|(replica, engine)| {
            // A serving daemon numbers its engines from one as it starts them; the number
            // keys nothing a recorded start reads.
            place(
                EngineArgs::new(replica, named.clone()),
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

/// The kernel keys the deployment `config` launches, by operator: what every engine of it
/// records in its profile runs, each key once.
///
/// The deployment's module set is fixed and built at this host's Build Server, which is started
/// when none serves, and each launch is run by the engine's executable with `--keys`.
/// No weight is read, and no daemon, Storage Server or communicator is started.
///
/// # Errors
///
/// Fails as [`launches`] does, when the module set cannot be fixed or the engine resolved, and when
/// a recording process fails or prints a line that is no `operator<TAB>key`.
pub async fn record(
    families: &processing::Registry,
    config: &ServerArgs,
    stated: &Stated,
) -> anyhow::Result<BTreeMap<String, BTreeSet<String>>> {
    let launches = launches(families, config, stated).await?;
    let socket = {
        let settings = config.build_config.clone();
        tokio::task::spawn_blocking(move || ensure_build_server(&settings))
            .await
            .context("starting the Build Server did not complete")??
    };
    let Some(named) = launches.first().map(|launch| launch.modules.clone()) else {
        return Ok(BTreeMap::new());
    };
    let servers = vec![BuildServer {
        host: config.host_identity.clone(),
        socket: socket.clone(),
        targets: Vec::new(),
    }];
    let modules = fix_modules(servers, config.host_identity.clone(), named)
        .await
        .map_err(anyhow::Error::from)?;
    tokio::task::spawn_blocking(move || {
        let directory = tempfile::tempdir().context("creating the launch files' directory")?;
        let mut recorded: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for mut launch in launches {
            launch.build_server = socket.display().to_string();
            launch.modules = modules.clone();
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
