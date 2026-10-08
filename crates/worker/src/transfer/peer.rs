//! Device storage every rank of a group maps from every other rank.
//!
//! Each rank allocates an exportable buffer, publishes its shareable handle
//! to the group (a fabric handle travels as bytes; a process descriptor is
//! served from this rank's grant socket), and maps all members' buffers back
//! to back in group order. Weight prefetch reads peers' shards through it.

use std::os::fd::AsRawFd;
use std::sync::Arc;

use uniserve_cuda::{DeviceAllocation, DeviceId, PeerMapping, ShareableHandle};

use super::{DescriptorGrants, fetch_descriptor};
use crate::storage::memory::Memory;
use crate::{Error, Result};

/// The export name under which a rank serves its peer allocation.
const PEER_EXPORT: &str = "peer-storage";

/// The host collectives a peer allocation needs from its group.
pub trait PeerExchange {
    /// This rank's position in the group.
    fn rank(&self) -> usize;
    /// Every member's `payload`, in rank order.
    fn all_gather(&self, payload: &[u8]) -> Result<Vec<Vec<u8>>>;
    /// Wait until every member reached this point.
    fn barrier(&self) -> Result<()>;
}

/// A member's published handle: `[kind][bytes]` where kind 0 is a 64-byte
/// fabric handle and kind 1 a grant endpoint name on `node`.
fn encode(node: &str, handle: &ShareableHandle, endpoint: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    match handle {
        ShareableHandle::Fabric(bytes) => {
            payload.push(0);
            payload.extend_from_slice(bytes);
        }
        ShareableHandle::Descriptor(_) => {
            payload.push(1);
            payload.extend_from_slice(node.as_bytes());
            payload.push(0);
            payload.extend_from_slice(endpoint.as_bytes());
        }
    }
    payload
}

fn decode(node: &str, payload: &[u8]) -> Result<ShareableHandle> {
    match payload.split_first() {
        Some((0, bytes)) => Ok(ShareableHandle::Fabric(
            <[u8; 64]>::try_from(bytes)
                .map_err(|_| Error::Invalid("a peer fabric handle has 64 bytes".into()))?,
        )),
        Some((1, rest)) => {
            let separator = rest
                .iter()
                .position(|&byte| byte == 0)
                .ok_or_else(|| Error::Invalid("a peer descriptor names its host".into()))?;
            let (host, endpoint) = (&rest[..separator], &rest[separator + 1..]);
            if host != node.as_bytes() {
                return Err(Error::Unsupported(
                    "peer storage across hosts requires fabric handles".into(),
                ));
            }
            let endpoint = std::str::from_utf8(endpoint)
                .map_err(|_| Error::Invalid("a peer grant endpoint is UTF-8".into()))?;
            Ok(ShareableHandle::Descriptor(
                fetch_descriptor(endpoint, PEER_EXPORT)
                    .map_err(|error| Error::Transport(error.to_string()))?,
            ))
        }
        _ => Err(Error::Invalid("a peer handle payload is malformed".into())),
    }
}

/// This rank's exportable allocation and the mapping of every member's.
pub struct PeerStorage {
    local: Arc<Memory>,
    peers: Arc<Memory>,
    bytes: usize,
}

impl PeerStorage {
    /// Allocate `bytes` (rounded up to the allocation granularity) on
    /// `device` and map every member's allocation, collectively over
    /// `group`. `node` identifies this host.
    pub fn allocate(
        device: i32,
        bytes: usize,
        node: &str,
        group: &dyn PeerExchange,
    ) -> Result<Self> {
        let local = Memory::exportable(device, bytes, &[])?;
        let allocation: &DeviceAllocation = local
            .allocation()
            .ok_or_else(|| Error::Invariant("peer storage is device memory".into()))?;
        let rounded = allocation.len();
        let handle = allocation
            .export()?
            .ok_or_else(|| Error::Invariant("peer storage is exportable".into()))?;

        // Serve a descriptor until every member imported it.
        let endpoint = format!("uniserve-peer-{}", uuid::Uuid::new_v4().simple());
        let grants = match &handle {
            ShareableHandle::Descriptor(fd) => {
                let grants = DescriptorGrants::new(&endpoint)
                    .map_err(|error| Error::Transport(error.to_string()))?;
                grants
                    .register(PEER_EXPORT, fd.as_raw_fd())
                    .map_err(|error| Error::Transport(error.to_string()))?;
                Some(grants)
            }
            ShareableHandle::Fabric(_) => None,
        };

        let payloads = group.all_gather(&encode(node, &handle, &endpoint))?;
        let handles = payloads
            .iter()
            .enumerate()
            .map(|(rank, payload)| {
                if rank == group.rank() {
                    // Map this rank's own allocation through its own handle.
                    return match &handle {
                        ShareableHandle::Fabric(bytes) => Ok(ShareableHandle::Fabric(*bytes)),
                        ShareableHandle::Descriptor(fd) => fd
                            .try_clone()
                            .map(ShareableHandle::Descriptor)
                            .map_err(|error| Error::Transport(error.to_string())),
                    };
                }
                decode(node, payload)
            })
            .collect::<Result<Vec<_>>>();
        // Every member fetched its descriptors before any grant closes.
        let barrier = group.barrier();
        if let Some(grants) = grants {
            grants
                .close()
                .map_err(|error| Error::Transport(error.to_string()))?;
        }
        let handles = handles?;
        barrier?;

        let peers = Memory::mapped(PeerMapping::peers(DeviceId(device), &handles, rounded)?);
        Ok(Self {
            local,
            peers,
            bytes: rounded,
        })
    }

    /// This rank's allocation.
    pub fn local(&self) -> &Arc<Memory> {
        &self.local
    }

    /// Every member's allocation, `bytes_each()` apart in rank order.
    pub fn peers(&self) -> &Arc<Memory> {
        &self.peers
    }

    pub fn bytes_each(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fabric_payloads_round_trip_and_descriptors_stay_on_their_host() -> Result<()> {
        let fabric = ShareableHandle::Fabric([7; 64]);
        let payload = encode("node-a", &fabric, "unused");
        assert!(
            matches!(decode("node-b", &payload)?, ShareableHandle::Fabric(bytes) if bytes == [7; 64])
        );

        let mut descriptor = vec![1];
        descriptor.extend_from_slice(b"node-a\0uniserve-peer-x");
        assert!(matches!(
            decode("node-b", &descriptor),
            Err(Error::Unsupported(_))
        ));
        assert!(decode("node-a", &[2]).is_err());
        Ok(())
    }
}
