//! The host's transfer calibration: what the calibration step (`storage-server calibrate`,
//! [`measure`]) measured of the host's copy paths, which the Transfer Server copies by.
//!
//! The record states, as JSON:
//!
//! - how the step timed each path ([`Method`]: the bytes of each timed copy and the timed copies);
//! - the GPUs it calibrated, which are the GPUs the Transfer Server copies on, each with the
//!   device storage the Transfer Server's CUDA context takes there, as NVML attributes it;
//! - each link within the host ([`LinkCost`]): each GPU's own storage, each pair of GPUs whose
//!   copy engines address each other, each GPU to and from host storage on its NUMA node, and host
//!   storage to itself;
//! - the rate of copies from another host into each end here ([`RemoteCost`]): over the NVLink
//!   fabric into a GPU, and NIXL READs through the host's RDMA ports, when the step was asked to
//!   time them.
//!
//! Each path's [`Cost`] is its bandwidth and the fixed cost of one copy over it as the host
//! observes it. The Transfer Server cuts a link's copies into chunks of the configured chunk time
//! at its bandwidth, and of at least its fixed cost, and keeps as many chunks in flight as that
//! fixed cost lasts chunks, plus the one copying ([`Sizing`]); its byte budget is a share of the
//! bandwidth. A copy over a path the record does not state fails, naming the path: the Transfer
//! Server never measures a path itself, and never assumes a rate.
//!
//! The Storage Server reads the record its settings name (`storage_config.transfer_calibration`)
//! when it starts, refuses one that does not hold together ([`Calibration::check`]), and gives it
//! to its Transfer Server; the Transfer Server refuses to start when the driver shows no GPU of
//! the record's.

pub mod measure;

use std::path::Path;
use std::time::Duration;

use ::storage::protocol::{ContextBytes, Link, LinkBandwidth, LinkEnd};
use serde::{Deserialize, Serialize};

/// What the calibration step measured of one host's copy paths.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    /// How the step timed each path.
    pub method: Method,
    /// The GPUs calibrated, which the Transfer Server copies on, each with the device storage its
    /// CUDA context takes there; none for a host calibrated without its GPUs.
    pub gpus: Vec<ContextBytes>,
    /// Each link within the host.
    pub links: Vec<LinkCost>,
    /// The rate of copies from another host into each end of this host the step timed.
    pub remote: Vec<RemoteCost>,
}

/// How the calibration step timed each path: one untimed copy of `sample_bytes`, then `samples`
/// timed ones; a path's fixed cost is the least time of `samples` copies of one page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Method {
    pub sample_bytes: u64,
    pub samples: u32,
}

/// A path's cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Cost {
    /// The rate of copies long enough that their fixed cost does not count, in bytes per second.
    pub bytes_per_second: u64,
    /// The fixed cost of one copy, from its submission to the host's observation of its
    /// completion, in nanoseconds.
    pub overhead_ns: u64,
}

/// A link within the host and its cost.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkCost {
    pub link: Link,
    pub cost: Cost,
}

/// How a copy from another host's storage reaches this host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RemotePath {
    /// Another host's device storage of the NVLink fabric, imported by its fabric handle and
    /// copied by the copy engines of the destination's GPU.
    Fabric,
    /// A NIXL READ this host's agent posts, through the host's RDMA ports.
    Read,
}

/// The cost of copies from another host into `destination`, this host's storage or one of its
/// GPUs, along `path`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteCost {
    pub path: RemotePath,
    pub destination: LinkEnd,
    pub cost: Cost,
}

impl Calibration {
    /// The calibration the JSON document at `path` states, checked ([`Calibration::check`]).
    ///
    /// # Errors
    ///
    /// Fails when the document cannot be read, is no calibration, or does not hold together.
    pub fn read(path: &Path) -> Result<Self, crate::Error> {
        let document = std::fs::read(path).map_err(|error| {
            crate::Error::Calibration(format!("{} cannot be read: {error}", path.display()))
        })?;
        let calibration: Self = serde_json::from_slice(&document).map_err(|error| {
            crate::Error::Calibration(format!("{} is no calibration: {error}", path.display()))
        })?;
        calibration.check().map_err(|reason| {
            crate::Error::Calibration(format!(
                "{} does not hold together: {reason}",
                path.display()
            ))
        })?;
        Ok(calibration)
    }

    /// Whether the record holds together: the method times something, every path has a
    /// nonzero bandwidth and is stated once, every GPU its links or remote paths name is a
    /// calibrated GPU, and every calibrated GPU has its own link and its links to and from host
    /// storage, beside the link of host storage to itself.
    ///
    /// # Errors
    ///
    /// The first reason it does not.
    pub fn check(&self) -> Result<(), String> {
        if self.method.sample_bytes == 0 || self.method.samples == 0 {
            return Err("the method times no copy".into());
        }
        let calibrated = |end: &LinkEnd| match end {
            LinkEnd::Host => true,
            LinkEnd::Device { gpu } => self.gpus.iter().any(|context| context.gpu == *gpu),
            LinkEnd::Remote { .. } => false,
        };
        for (index, measured) in self.links.iter().enumerate() {
            let link = &measured.link;
            if !calibrated(&link.source) || !calibrated(&link.destination) {
                return Err(format!(
                    "{link:?} names an end that is neither host storage nor a calibrated GPU"
                ));
            }
            if measured.cost.bytes_per_second == 0 {
                return Err(format!("{link:?} has no bandwidth"));
            }
            if self.links[..index]
                .iter()
                .any(|earlier| earlier.link == *link)
            {
                return Err(format!("{link:?} is stated twice"));
            }
        }
        for (index, remote) in self.remote.iter().enumerate() {
            let fabric_into_host =
                remote.path == RemotePath::Fabric && remote.destination == LinkEnd::Host;
            if !calibrated(&remote.destination) || fabric_into_host {
                return Err(format!(
                    "copies {:?} into {:?} name no calibrated GPU of the fabric or end of the host",
                    remote.path, remote.destination
                ));
            }
            if remote.cost.bytes_per_second == 0 {
                return Err(format!(
                    "copies {:?} into {:?} have no bandwidth",
                    remote.path, remote.destination
                ));
            }
            let stated = self.remote[..index].iter().any(|earlier| {
                earlier.path == remote.path && earlier.destination == remote.destination
            });
            if stated {
                return Err(format!(
                    "copies {:?} into {:?} are stated twice",
                    remote.path, remote.destination
                ));
            }
        }
        let host = LinkEnd::Host;
        let mut required = vec![(host.clone(), host.clone())];
        for context in &self.gpus {
            let gpu = LinkEnd::Device { gpu: context.gpu };
            required.push((gpu.clone(), gpu.clone()));
            required.push((host.clone(), gpu.clone()));
            required.push((gpu, host.clone()));
        }
        for (source, destination) in required {
            let link = Link {
                source,
                destination,
            };
            if self.link(&link).is_none() {
                return Err(format!("{link:?} is not calibrated"));
            }
        }
        Ok(())
    }

    /// The cost of `link`, a link within the host.
    pub fn link(&self, link: &Link) -> Option<Cost> {
        self.links
            .iter()
            .find(|measured| measured.link == *link)
            .map(|measured| measured.cost)
    }

    /// The cost of copies from another host into `destination` along `path`.
    pub fn remote(&self, path: RemotePath, destination: &LinkEnd) -> Option<Cost> {
        self.remote
            .iter()
            .find(|remote| remote.path == path && remote.destination == *destination)
            .map(|remote| remote.cost)
    }

    /// Each link within the host and its bandwidth, as the listing states them.
    pub fn bandwidths(&self) -> Vec<LinkBandwidth> {
        self.links
            .iter()
            .map(|measured| LinkBandwidth {
                link: measured.link.clone(),
                bytes_per_second: measured.cost.bytes_per_second,
            })
            .collect()
    }
}

/// How a copy thread cuts and pipelines the chunks of its link, from the link's [`Cost`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Sizing {
    /// The bytes of one chunk: whole pages of the host's page size, at least one.
    pub(crate) bytes: u64,
    /// The chunks kept in flight.
    pub(crate) depth: usize,
    /// The link's bandwidth, in bytes per second, which its byte budget is a share of.
    pub(crate) bandwidth: f64,
}

impl Sizing {
    /// The sizing of a link of `cost` whose chunks last `chunk`, cut at the host's `page` size.
    /// A chunk lasts `chunk`, and at least the link's fixed cost of one copy, so that submitting
    /// a chunk never takes longer than copying it. An `asynchronous` runner (copy engines, READs)
    /// keeps one chunk copying and as many more in flight as its fixed cost lasts chunks, so the
    /// copy engine or the adapter never waits for the host to observe a completion and submit the
    /// next; a synchronous one (the CPU) copies one at a time.
    pub(crate) fn of(cost: Cost, chunk: Duration, page: u64, asynchronous: bool) -> Self {
        let overhead = Duration::from_nanos(cost.overhead_ns);
        let lasts = chunk.max(overhead).max(Duration::from_nanos(1));
        let bandwidth = cost.bytes_per_second as f64;
        let bytes = ((bandwidth * lasts.as_secs_f64()) as u64 / page * page).max(page);
        let depth = if asynchronous {
            1 + usize::try_from(overhead.as_nanos().div_ceil(lasts.as_nanos())).unwrap_or(1)
        } else {
            1
        };
        Self {
            bytes,
            depth,
            bandwidth,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GPU: [u8; 16] = [7; 16];

    fn cost(bytes_per_second: u64) -> Cost {
        Cost {
            bytes_per_second,
            overhead_ns: 10_000,
        }
    }

    fn link(source: LinkEnd, destination: LinkEnd, bytes_per_second: u64) -> LinkCost {
        LinkCost {
            link: Link {
                source,
                destination,
            },
            cost: cost(bytes_per_second),
        }
    }

    /// A calibration of host storage and one GPU, with READs into the GPU.
    fn calibration() -> Calibration {
        let gpu = LinkEnd::Device { gpu: GPU };
        Calibration {
            method: Method {
                sample_bytes: 64 << 20,
                samples: 8,
            },
            gpus: vec![ContextBytes {
                gpu: GPU,
                bytes: 500 << 20,
            }],
            links: vec![
                link(LinkEnd::Host, LinkEnd::Host, 20e9 as u64),
                link(gpu.clone(), gpu.clone(), 2e12 as u64),
                link(LinkEnd::Host, gpu.clone(), 200e9 as u64),
                link(gpu.clone(), LinkEnd::Host, 200e9 as u64),
            ],
            remote: vec![RemoteCost {
                path: RemotePath::Read,
                destination: gpu,
                cost: cost(50e9 as u64),
            }],
        }
    }

    /// A calibration reads back from the document it writes, and states each path's cost.
    #[test]
    fn a_calibration_reads_back_and_states_each_paths_cost() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("transfer.json");
        let written = calibration();
        std::fs::write(&path, serde_json::to_vec_pretty(&written).unwrap()).unwrap();
        let read = Calibration::read(&path).unwrap();
        assert_eq!(read, written);

        let gpu = LinkEnd::Device { gpu: GPU };
        let into_gpu = Link {
            source: LinkEnd::Host,
            destination: gpu.clone(),
        };
        assert_eq!(read.link(&into_gpu), Some(cost(200e9 as u64)));
        assert_eq!(read.remote(RemotePath::Read, &gpu), Some(cost(50e9 as u64)));
        assert_eq!(read.remote(RemotePath::Fabric, &gpu), None);
        assert_eq!(read.bandwidths().len(), 4);
    }

    /// A record that leaves out a GPU's link to host storage, names a GPU it did not calibrate,
    /// states a path twice or without bandwidth, or times nothing is refused.
    #[test]
    fn a_calibration_that_does_not_hold_together_is_refused() {
        let gpu = LinkEnd::Device { gpu: GPU };
        let other = LinkEnd::Device { gpu: [9; 16] };
        let mut missing = calibration();
        missing.links.retain(|measured| {
            measured.link.destination != LinkEnd::Host || measured.link.source == LinkEnd::Host
        });
        let mut stranger = calibration();
        stranger.links.push(link(other.clone(), gpu.clone(), 1));
        let mut twice = calibration();
        twice.links.push(link(LinkEnd::Host, LinkEnd::Host, 1));
        let mut idle = calibration();
        idle.links[0].cost.bytes_per_second = 0;
        let mut untimed = calibration();
        untimed.method.samples = 0;
        let mut fabric_into_host = calibration();
        fabric_into_host.remote.push(RemoteCost {
            path: RemotePath::Fabric,
            destination: LinkEnd::Host,
            cost: cost(1),
        });
        for refused in [missing, stranger, twice, idle, untimed, fabric_into_host] {
            assert!(refused.check().is_err(), "{refused:?} was not refused");
        }
    }

    /// A chunk lasts the chunk time at the link's bandwidth, in whole pages; a link whose fixed
    /// cost exceeds the chunk time cuts chunks lasting the fixed cost; copy engines keep two
    /// chunks in flight then, the CPU one.
    #[test]
    fn a_links_chunks_last_the_chunk_time_and_at_least_its_fixed_cost() {
        let page = 65536;
        let fast = Sizing::of(cost(2e12 as u64), Duration::from_millis(1), page, true);
        assert_eq!(fast.bytes, 2e9 as u64 / page * page);
        assert_eq!(fast.depth, 2);

        let slow = Cost {
            bytes_per_second: 1e9 as u64,
            overhead_ns: 2_000_000,
        };
        let sized = Sizing::of(slow, Duration::from_millis(1), page, true);
        assert_eq!(sized.bytes, 2e6 as u64 / page * page);
        assert_eq!(sized.depth, 2);
        assert_eq!(
            Sizing::of(slow, Duration::from_millis(1), page, false).depth,
            1
        );

        let crawling = Sizing::of(cost(1000), Duration::from_millis(1), page, true);
        assert_eq!(crawling.bytes, page);
    }
}
