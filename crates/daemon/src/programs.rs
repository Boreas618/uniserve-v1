//! The launches of a deployment's main processes, for a start that records the programs they
//! launch and serves nothing (`uniserve programs`, `engine::record`).
//!
//! The launches are the ones a serving leader starts its main processes with: the model's facts
//! resolved into the engine configuration ([`Leader::prepare`]), one configuration per independent
//! replica (`replica_configs`), and each process placed on this node
//! (`crate::engine::control::place`). The interfaces a serving process reaches - the daemon's
//! request and process sockets and the Program Server's - are left unstated, because a recorded
//! start connects to none of them.

use std::path::Path;

use proto::config::EngineArgs;

use crate::engine::control::{Interfaces, place};
use crate::leader::{Leader, replica_configs};
use crate::options::{ServerArgs, Stated};

/// The launch of every main process the deployment `config` starts on this node, in replica
/// order; `stated` holds the option paths the operator set, and `catalogue` the families whose
/// requests this build processes.
///
/// # Errors
///
/// Fails where a serving leader's start fails before it launches a process: the model's assets do
/// not resolve, an option of another family was stated, the options do not resolve into an engine
/// configuration, the replicas are not independent, or a process cannot be placed on this node.
pub async fn launches(
    catalogue: &nn::family::Catalogue,
    config: &ServerArgs,
    stated: &Stated,
) -> anyhow::Result<Vec<EngineArgs>> {
    let prepared = Leader::prepare(catalogue, config, stated).await?;
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
                    host: &config.host_identity,
                    requests: "",
                    processes: "",
                    program_server: Path::new(""),
                },
            )
        })
        .collect()
}
