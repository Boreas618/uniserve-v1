//! The connectors between worker groups: each edge data crosses between model parts and the
//! mechanisms that carry its products (`device_config.transfer`).
//!
//! The deployment's options state them as text, `source[:rank]->destination[:rank]=mechanisms`
//! entries separated by commas, which the daemon parses here and resolves against the rank layout.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use super::WorkerGroupId;

/// Transport available for product movement between worker pools.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferBackend {
    /// Keeps products within one address space.
    #[default]
    Local,
    /// Publishes products through POSIX shared storage.
    PosixShm,
    /// Publishes device products through CUDA VMM handles.
    CudaVmm,
    /// Carries a host product's bytes on the rank channel's data path.
    ///
    /// Shared storage reaches only one host. A product on this edge travels in
    /// the producing rank's result and in the consuming rank's batch, so it
    /// crosses hosts wherever the rank channel does.
    Channel,
}

impl TransferBackend {
    /// Returns the stable configuration spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::PosixShm => "posix_shm",
            Self::CudaVmm => "cuda_vmm",
            Self::Channel => "channel",
        }
    }
}

impl std::fmt::Display for TransferBackend {
    /// Formats the value for diagnostic output.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for TransferBackend {
    type Err = ConnectorConfigError;

    /// Parses the value from its string representation.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "local" => Ok(Self::Local),
            "posix_shm" => Ok(Self::PosixShm),
            "cuda_vmm" => Ok(Self::CudaVmm),
            "channel" => Ok(Self::Channel),
            _ => Err(ConnectorConfigError::message(format!(
                "unsupported transfer backend {value:?}"
            ))),
        }
    }
}

/// One explicit directed transfer edge between configured pool identities.
///
/// Products move by one device mechanism and one host mechanism, and a
/// product's location decides which carries it: a device product travels on
/// the edge's device mechanism, a host product on its host mechanism. An edge
/// may name only one of the two, in which case products of the other
/// location have no way across it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorEdge {
    /// Pool that produces the transferred product.
    pub source_worker: WorkerGroupId,
    /// One source member, or all members when omitted.
    pub source_rank: Option<u32>,
    /// Pool that consumes the transferred product.
    pub destination_worker: WorkerGroupId,
    /// One consumer member, or all members when omitted.
    pub destination_rank: Option<u32>,
    /// Mechanism carrying device products across this edge.
    pub device: Option<TransferBackend>,
    /// Mechanism carrying host products across this edge.
    pub host: Option<TransferBackend>,
}

impl ConnectorEdge {
    /// The mechanisms this edge carries products on, device first.
    pub fn mechanisms(&self) -> impl Iterator<Item = TransferBackend> + '_ {
        self.device.into_iter().chain(self.host)
    }
}

/// The connectors between worker groups, as vLLM-Omni names the edges data crosses between
/// model parts: each edge's data-plane transfer (`device_config.transfer`) and the rank layout
/// it is resolved against.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorConfig {
    /// Explicit directed transfer edges.
    pub edges: Vec<ConnectorEdge>,
}

impl std::fmt::Display for ConnectorConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, edge) in self.edges.iter().enumerate() {
            if index > 0 {
                formatter.write_str(",")?;
            }
            write!(formatter, "{}", edge.source_worker)?;
            if let Some(rank) = edge.source_rank {
                write!(formatter, ":{rank}")?;
            }
            write!(formatter, "->{}", edge.destination_worker)?;
            if let Some(rank) = edge.destination_rank {
                write!(formatter, ":{rank}")?;
            }
            formatter.write_str("=")?;
            if edge.device == Some(TransferBackend::Local)
                && edge.host == Some(TransferBackend::Local)
            {
                formatter.write_str("local")?;
            } else {
                for (index, mechanism) in edge.mechanisms().enumerate() {
                    if index > 0 {
                        formatter.write_str("+")?;
                    }
                    write!(formatter, "{mechanism}")?;
                }
            }
        }
        Ok(())
    }
}

impl ConnectorConfig {
    /// Parses `source[:rank]->destination[:rank]=mechanisms` directed bindings.
    ///
    /// `mechanisms` names the edge's device mechanism, its host mechanism, or
    /// both joined by `+` with the device mechanism first, as in
    /// `cuda_vmm+posix_shm`. `local` serves both locations. Entries are separated by
    /// commas, and an omitted rank covers every rank of that worker.
    ///
    /// # Errors
    ///
    /// Fails on a malformed entry, an invalid worker id or rank, an unknown
    /// or conflicting mechanism, or an edge that overlaps an earlier one.
    pub fn parse(s: &str) -> Result<Self, ConnectorConfigError> {
        let mut edges = Vec::new();

        let endpoint = |text: &str| -> Result<(WorkerGroupId, Option<u32>), ConnectorConfigError> {
            let (worker, rank) = match text.trim().split_once(':') {
                Some((worker, rank)) => (
                    worker,
                    Some(rank.parse::<u32>().map_err(|_| {
                        ConnectorConfigError::message("transfer rank must be a nonnegative integer")
                    })?),
                ),
                None => (text.trim(), None),
            };
            let worker = WorkerGroupId::new(worker).map_err(|error| {
                ConnectorConfigError::message(format!("invalid transfer worker: {error}"))
            })?;
            Ok((worker, rank))
        };

        for entry in s.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            // Parse and validate both endpoint identities before accepting the
            // transport so errors remain attributable to one edge.
            let (edge, backend) = entry.split_once('=').ok_or_else(|| {
                ConnectorConfigError::message(format!(
                    "transfer entry {entry:?} must be edge=backend"
                ))
            })?;
            let (src, dst) = edge.split_once("->").ok_or_else(|| {
                ConnectorConfigError::message(format!("transfer edge {edge:?} must be src->dst"))
            })?;
            let (src, source_rank) = endpoint(src)?;
            let (dst, destination_rank) = endpoint(dst)?;
            let (device, host) = Self::parse_mechanisms(backend.trim())?;

            // Two edges between the same workers overlap when their ranks
            // can name one pair; an omitted rank overlaps every rank.
            if edges.iter().any(|existing: &ConnectorEdge| {
                existing.source_worker == src
                    && existing.destination_worker == dst
                    && (existing.source_rank.is_none()
                        || source_rank.is_none()
                        || existing.source_rank == source_rank)
                    && (existing.destination_rank.is_none()
                        || destination_rank.is_none()
                        || existing.destination_rank == destination_rank)
            }) {
                return Err(ConnectorConfigError::message(format!(
                    "duplicate transfer edge {edge:?}"
                )));
            }

            edges.push(ConnectorEdge {
                source_worker: src,
                source_rank,
                destination_worker: dst,
                destination_rank,
                device,
                host,
            });
        }

        Ok(Self { edges })
    }

    /// Resolves an edge's mechanism text into its device and host mechanisms.
    fn parse_mechanisms(
        text: &str,
    ) -> Result<(Option<TransferBackend>, Option<TransferBackend>), ConnectorConfigError> {
        let backends = text
            .split('+')
            .map(|name| TransferBackend::from_str(name.trim()))
            .collect::<Result<Vec<_>, _>>()?;
        if backends.contains(&TransferBackend::Local) {
            // Local storage serves both locations and combines with nothing.
            if backends.len() > 1 {
                return Err(ConnectorConfigError::message(format!(
                    "transfer mechanisms {text:?} combine local with another mechanism"
                )));
            }
            return Ok((Some(TransferBackend::Local), Some(TransferBackend::Local)));
        }
        let mut device = None;
        let mut host = None;
        for backend in backends {
            let slot = match backend {
                TransferBackend::CudaVmm => &mut device,
                TransferBackend::PosixShm | TransferBackend::Channel => &mut host,
                TransferBackend::Local => unreachable!("local mechanisms return above"),
            };
            if slot.replace(backend).is_some() {
                return Err(ConnectorConfigError::message(format!(
                    "transfer mechanisms {text:?} name two mechanisms for one location"
                )));
            }
        }
        Ok((device, host))
    }
}

impl FromStr for ConnectorConfig {
    type Err = ConnectorConfigError;

    /// Parses the value from its string representation.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::parse(value)
    }
}

/// Error returned for an invalid transfer-transport mapping.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct ConnectorConfigError(String);

impl ConnectorConfigError {
    /// Returns the human-readable error message.
    fn message(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_channel_edge_is_a_transfer_backend() {
        // Host products cross hosts on the rank channel, so the mechanism has
        // to be nameable in a rank layout's transfer bindings.
        let transfer = ConnectorConfig::parse("video_decoder->muxer=channel").unwrap();
        assert_eq!(transfer.edges[0].host, Some(TransferBackend::Channel));
        assert_eq!(transfer.edges[0].device, None);
        assert_eq!(TransferBackend::Channel.as_str(), "channel");
    }

    #[test]
    fn an_edge_names_a_device_mechanism_and_a_host_mechanism() {
        // A product's location decides which mechanism carries it, so one
        // edge binds both, and each location has at most one.
        let both = ConnectorConfig::parse("decoder->muxer=cuda_vmm+posix_shm").unwrap();
        assert_eq!(both.edges[0].device, Some(TransferBackend::CudaVmm));
        assert_eq!(both.edges[0].host, Some(TransferBackend::PosixShm));

        let local = ConnectorConfig::parse("decoder->decoder=local").unwrap();
        assert_eq!(local.edges[0].device, Some(TransferBackend::Local));
        assert_eq!(local.edges[0].host, Some(TransferBackend::Local));

        assert!(ConnectorConfig::parse("a->b=posix_shm+channel").is_err());
        assert!(ConnectorConfig::parse("a->b=local+posix_shm").is_err());
        assert!(ConnectorConfig::parse("a->b=cuda_vmm+cuda_vmm").is_err());
    }

    #[test]
    fn transport_map_parses_edges() {
        let t =
            ConnectorConfig::parse("encoder->prefill=cuda_vmm,prefill->decode=posix_shm").unwrap();
        assert_eq!(
            t.edges[0],
            ConnectorEdge {
                source_worker: WorkerGroupId("encoder".to_owned()),
                source_rank: None,
                destination_worker: WorkerGroupId("prefill".to_owned()),
                destination_rank: None,
                device: Some(TransferBackend::CudaVmm),
                host: None,
            }
        );
        assert_eq!(t.edges[1].host, Some(TransferBackend::PosixShm));
        assert!(ConnectorConfig::parse("bad-entry").is_err());
        assert!(ConnectorConfig::parse("prefill->decode=tcp").is_err());
        assert!(
            ConnectorConfig::parse("prefill->decode=posix_shm,prefill->decode=cuda_vmm").is_err()
        );
        let split = ConnectorConfig::parse(
            "encoder:0->denoiser:0=posix_shm,encoder:0->denoiser:1=cuda_vmm",
        )
        .unwrap();
        assert_eq!(split.edges[1].source_rank, Some(0));
        assert_eq!(split.edges[1].destination_rank, Some(1));
        assert!(ConnectorConfig::parse("encoder:x->denoiser=posix_shm").is_err());
        assert!(
            ConnectorConfig::parse("encoder->denoiser=posix_shm,encoder:0->denoiser:1=cuda_vmm")
                .is_err()
        );
    }
}
