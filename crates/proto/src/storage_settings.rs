//! A Storage Server's settings as the process interface carries them: the claim a node's daemon
//! receives states the settings of the Storage Server it starts (`process::Claim::storage`).

use crate::process::{
    HandoffTransport, MooncakeConnector, MooncakeTransport, NixlTransports, StorageSettings,
};

impl From<&storage::Settings> for StorageSettings {
    fn from(settings: &storage::Settings) -> Self {
        let server = &settings.server;
        let path = |path: &Option<std::path::PathBuf>| {
            path.as_ref()
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_default()
        };
        Self {
            tcp: server
                .tcp
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default(),
            host_reserve_gib: server.host_reserve_gib,
            transfer_calibration: path(&server.transfer_calibration),
            transfer_chunk_us: server.transfer_chunk_us,
            transfer_progress_ms: server.transfer_progress_ms,
            nixl_transports: match server.nixl_transports {
                storage::settings::Transports::Selected => NixlTransports::Selected,
                storage::settings::Transports::Rdma => NixlTransports::Rdma,
            }
            .into(),
            transfer_deadline_s: server.transfer_deadline_s,
            put_deadline_s: server.put_deadline_s,
            buffer_pool_trace: path(&server.buffer_pool_trace),
            child_restart_s: server.child_restart_s,
            child_restart_max_s: server.child_restart_max_s,
            shoot_down_s: server.shoot_down_s,
            directory: server
                .directory
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default(),
            directory_period_ms: server.directory_period_ms,
            handoff_transports: server
                .handoff_transports
                .iter()
                .map(|transport| {
                    i32::from(match transport {
                        scheduling::kv::Transport::Nixl => HandoffTransport::Nixl,
                        scheduling::kv::Transport::Mooncake => HandoffTransport::Mooncake,
                    })
                })
                .collect(),
            connector: settings
                .connector
                .as_ref()
                .map(|connector| MooncakeConnector {
                    master: connector.master.to_string(),
                    metadata: connector.metadata.to_string(),
                    transport: match connector.transport {
                        storage::connector::Transport::Tcp => MooncakeTransport::Tcp,
                        storage::connector::Transport::Rdma => MooncakeTransport::Rdma,
                    }
                    .into(),
                    device: connector.device.clone(),
                    segment_gib: connector.segment_gib,
                    events: connector
                        .events
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_default(),
                }),
        }
    }
}

impl TryFrom<&StorageSettings> for storage::Settings {
    type Error = String;

    /// The settings a message states.
    ///
    /// # Errors
    ///
    /// Fails when an endpoint, the metadata service or the transport is no value of its kind.
    fn try_from(message: &StorageSettings) -> Result<Self, Self::Error> {
        let path = |text: &str| (!text.is_empty()).then(|| std::path::PathBuf::from(text));
        let endpoint = |text: &str| -> Result<Option<service::Endpoint>, String> {
            if text.is_empty() {
                return Ok(None);
            }
            text.parse().map(Some).map_err(|error| format!("{error}"))
        };
        let tcp = endpoint(&message.tcp)?;
        let connector = match &message.connector {
            None => None,
            Some(connector) => Some(storage::connector::Settings {
                master: connector
                    .master
                    .parse()
                    .map_err(|error| format!("{error}"))?,
                metadata: connector.metadata.parse()?,
                transport: match MooncakeTransport::try_from(connector.transport) {
                    Ok(MooncakeTransport::Tcp) => storage::connector::Transport::Tcp,
                    Ok(MooncakeTransport::Rdma) => storage::connector::Transport::Rdma,
                    Err(_) => {
                        return Err(format!(
                            "{} names no Mooncake transport",
                            connector.transport
                        ));
                    }
                },
                device: connector.device.clone(),
                segment_gib: connector.segment_gib,
                events: endpoint(&connector.events)?,
            }),
        };
        let handoff_transports = message
            .handoff_transports
            .iter()
            .map(|transport| match HandoffTransport::try_from(*transport) {
                Ok(HandoffTransport::Nixl) => Ok(scheduling::kv::Transport::Nixl),
                Ok(HandoffTransport::Mooncake) => Ok(scheduling::kv::Transport::Mooncake),
                Err(_) => Err(format!("{transport} names no handoff transport")),
            })
            .collect::<Result<_, String>>()?;
        let nixl_transports = match NixlTransports::try_from(message.nixl_transports) {
            Ok(NixlTransports::Selected) => storage::settings::Transports::Selected,
            Ok(NixlTransports::Rdma) => storage::settings::Transports::Rdma,
            Err(_) => {
                return Err(format!(
                    "{} names no NIXL transports",
                    message.nixl_transports
                ));
            }
        };
        Ok(Self {
            server: storage::ServerSettings {
                tcp,
                host_reserve_gib: message.host_reserve_gib,
                transfer_calibration: path(&message.transfer_calibration),
                transfer_chunk_us: message.transfer_chunk_us,
                transfer_progress_ms: message.transfer_progress_ms,
                nixl_transports,
                transfer_deadline_s: message.transfer_deadline_s,
                put_deadline_s: message.put_deadline_s,
                buffer_pool_trace: path(&message.buffer_pool_trace),
                child_restart_s: message.child_restart_s,
                child_restart_max_s: message.child_restart_max_s,
                shoot_down_s: message.shoot_down_s,
                directory: endpoint(&message.directory)?,
                directory_period_ms: message.directory_period_ms,
                handoff_transports,
            },
            connector,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Settings cross the process interface unchanged.
    #[test]
    fn settings_cross_the_process_interface_unchanged() {
        let settings = storage::Settings {
            server: storage::ServerSettings {
                tcp: Some("0.0.0.0:0".parse().unwrap()),
                transfer_calibration: Some("transfer.json".into()),
                nixl_transports: storage::settings::Transports::Rdma,
                directory: Some("10.0.0.1:7100".parse().unwrap()),
                directory_period_ms: 250,
                handoff_transports: vec![
                    scheduling::kv::Transport::Mooncake,
                    scheduling::kv::Transport::Nixl,
                ],
                ..storage::ServerSettings::default()
            },
            connector: Some(storage::connector::Settings {
                master: "10.0.0.1:50051".parse().unwrap(),
                metadata: storage::connector::Metadata::Server("etcd://10.0.0.1:2379".into()),
                transport: storage::connector::Transport::Rdma,
                device: "mlx5_0".into(),
                segment_gib: 4,
                events: Some("10.0.0.1:50053".parse().unwrap()),
            }),
        };
        let message = StorageSettings::from(&settings);
        assert_eq!(storage::Settings::try_from(&message).unwrap(), settings);
        let defaults = storage::Settings::default();
        assert_eq!(
            storage::Settings::try_from(&StorageSettings::from(&defaults)).unwrap(),
            defaults
        );
    }
}
