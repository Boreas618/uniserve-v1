//! The NIC priority matrix the Transfer Engine's RDMA transport is installed with: for each place
//! storage lies (host storage of a NUMA node, `cpu:N`; a GPU, `cuda:N` by its ordinal in this
//! process), the RDMA devices its transfers prefer and those they may take besides.
//!
//! The engine's own discovery takes every device with an active port, an Ethernet port of a host
//! whose fabric is InfiniBand among them, which connects to no other host's InfiniBand port: a
//! transfer whose storage lies near that port then fails. The Transfer Server installs the
//! transport with the devices of the host's fabric alone ([`crate::transfer::ports::fabric`]),
//! each preferred by the storage on its own NUMA node; storage of a node without one prefers
//! every device.

use std::collections::BTreeSet;
use std::path::Path;

use platform::{DeviceAttribute, DeviceId};

use crate::transfer::ports;

use super::{Error, Result};

/// Where the kernel states the host's NUMA nodes.
const NODES: &str = "/sys/devices/system/node";

/// The places storage lies and the devices each prefers and may take besides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Topology {
    locations: Vec<(String, Vec<String>, Vec<String>)>,
}

impl Topology {
    /// The matrix of this host: the devices of its RDMA fabric under the kernel's InfiniBand
    /// class, by their NUMA nodes, for host storage of each NUMA node and for each GPU this
    /// process sees, by the host NUMA node closest to it.
    ///
    /// # Errors
    ///
    /// Fails when the host has no usable RDMA port, or the kernel's or the driver's attributes
    /// cannot be read.
    pub(crate) fn of_host() -> Result<Self> {
        let class = Path::new(ports::CLASS);
        let fabric = ports::fabric(class).map_err(|error| {
            Error::TransferEngineArgument(format!(
                "reading the host's RDMA ports in {}: {error}",
                ports::CLASS
            ))
        })?;
        if fabric.is_empty() {
            return Err(Error::TransferEngineArgument(
                "the host has no usable RDMA port (an active link that holds an address)".into(),
            ));
        }
        let devices: Vec<(String, Option<u32>)> = fabric
            .into_iter()
            .map(|port| (port.device, port.numa))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        let mut places: Vec<(String, Option<u32>)> = Vec::new();
        for node in std::fs::read_dir(NODES)? {
            let name = node?.file_name().to_string_lossy().into_owned();
            if let Some(number) = name
                .strip_prefix("node")
                .and_then(|n| n.parse::<u32>().ok())
            {
                places.push((format!("cpu:{number}"), Some(number)));
            }
        }
        // A host whose driver shows no GPU states host storage alone.
        let gpus = DeviceId::count().unwrap_or(0);
        for ordinal in 0..gpus {
            let node = DeviceId(ordinal).attribute(DeviceAttribute::HostNumaId)?;
            places.push((format!("cuda:{ordinal}"), u32::try_from(node).ok()));
        }
        places.sort();
        Ok(Self::of(&devices, &places))
    }

    /// The matrix of `devices`, each with its NUMA node, for `places`, each with the NUMA node
    /// its storage lies nearest.
    fn of(devices: &[(String, Option<u32>)], places: &[(String, Option<u32>)]) -> Self {
        let locations = places
            .iter()
            .map(|(place, node)| {
                let (near, far): (Vec<_>, Vec<_>) = devices
                    .iter()
                    .partition(|(_, numa)| numa.is_some() && numa == node);
                let names = |devices: Vec<&(String, Option<u32>)>| {
                    devices.into_iter().map(|(name, _)| name.clone()).collect()
                };
                if near.is_empty() {
                    (place.clone(), names(far), Vec::new())
                } else {
                    (place.clone(), names(near), names(far))
                }
            })
            .collect();
        Self { locations }
    }

    /// The matrix as the RDMA transport reads it: a JSON object of each place's two lists.
    pub(crate) fn json(&self) -> String {
        let quoted = |names: &[String]| {
            let names: Vec<String> = names.iter().map(|name| format!("\"{name}\"")).collect();
            format!("[{}]", names.join(","))
        };
        let entries: Vec<String> = self
            .locations
            .iter()
            .map(|(place, preferred, others)| {
                format!("\"{place}\":[{},{}]", quoted(preferred), quoted(others))
            })
            .collect();
        format!("{{{}}}", entries.join(","))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Storage prefers the devices of its own NUMA node and may take the others; storage of a
    /// node without a device prefers every device.
    #[test]
    fn storage_prefers_the_devices_of_its_numa_node() {
        let devices = [
            ("mlx5_0".to_owned(), Some(0)),
            ("mlx5_1".to_owned(), Some(0)),
            ("mlx5_3".to_owned(), Some(1)),
        ];
        let places = [
            ("cpu:0".to_owned(), Some(0)),
            ("cpu:7".to_owned(), Some(7)),
            ("cuda:2".to_owned(), Some(1)),
        ];
        assert_eq!(
            Topology::of(&devices, &places).json(),
            r#"{"cpu:0":[["mlx5_0","mlx5_1"],["mlx5_3"]],"cpu:7":[["mlx5_0","mlx5_1","mlx5_3"],[]],"cuda:2":[["mlx5_3"],["mlx5_0","mlx5_1"]]}"#
        );
    }
}
