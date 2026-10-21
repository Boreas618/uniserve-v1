//! Device storage every rank of a group maps from every other rank.
//!
//! Each rank allocates an exportable buffer, publishes its shareable handle
//! to the group (a fabric handle travels as bytes; a process descriptor is
//! served from this rank's grant socket), and maps all members' buffers back
//! to back in group order. Weight prefetch reads peers' shards through it.

use std::os::fd::AsRawFd;
use std::sync::Arc;

use cuda::{DeviceAllocation, DeviceId, PeerMapping, ShareableHandle};

use super::grants::{DescriptorGrants, EXPORT_NAME_BYTES};
use super::handle::AllocationHandle;
use crate::memory::allocation::Storage;
use crate::parallel::host::HostGroup;
use crate::{Error, Result};

/// The export name under which a rank serves its peer allocation; a grant
/// name has exactly [`EXPORT_NAME_BYTES`] bytes.
const PEER_EXPORT: &str = "uniserve-peer-storage-allocation";
const _: () = assert!(PEER_EXPORT.len() == EXPORT_NAME_BYTES);

/// A member's published handle: `host \0 endpoint \0 handle`, the
/// member's host, the grant socket serving its descriptor, and its
/// [`AllocationHandle`] (last, since a fabric handle may contain zeros).
fn encode(host: &str, handle: &ShareableHandle, endpoint: &str) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(host.as_bytes());
    payload.push(0);
    payload.extend_from_slice(endpoint.as_bytes());
    payload.push(0);
    payload.extend_from_slice(&AllocationHandle::of(handle).encode());
    payload
}

/// Import a member's published handle on host `host`.
fn decode(host: &str, payload: &[u8]) -> Result<ShareableHandle> {
    let mut fields = payload.splitn(3, |&byte| byte == 0);
    let (Some(member_host), Some(endpoint), Some(handle)) =
        (fields.next(), fields.next(), fields.next())
    else {
        return Err(Error::Invalid("a peer handle payload is malformed".into()));
    };
    let handle = AllocationHandle::decode(handle)?;
    if member_host != host.as_bytes() && !handle.crosses_hosts() {
        return Err(Error::Unsupported(
            "peer storage across hosts requires fabric handles".into(),
        ));
    }
    let endpoint = std::str::from_utf8(endpoint)
        .map_err(|_| Error::Invalid("a peer grant endpoint is UTF-8".into()))?;
    handle.import(endpoint, PEER_EXPORT)
}

/// The grant socket name a member publishes with `handle`, and the grants
/// serving its descriptor to the other members of its host; none for a
/// fabric handle, which travels as bytes.
fn serve(handle: &ShareableHandle) -> Result<(String, Option<DescriptorGrants>)> {
    let endpoint = format!("uniserve-peer-{}", uuid::Uuid::new_v4().simple());
    let grants = match handle {
        ShareableHandle::Descriptor(fd) => {
            let grants = DescriptorGrants::new(&endpoint)
                .map_err(|error| Error::transport_source("opening the descriptor grants", error))?;
            grants
                .register(PEER_EXPORT, fd.as_raw_fd())
                .map_err(|error| Error::transport_source("granting a peer's descriptor", error))?;
            Some(grants)
        }
        ShareableHandle::Fabric(_) => None,
    };
    Ok((endpoint, grants))
}

/// This rank's exportable allocation and the mapping of every member's.
pub(crate) struct PeerStorage {
    local: Arc<Storage>,
    peers: Arc<Storage>,
    bytes: usize,
}

impl PeerStorage {
    /// Allocate `bytes` (rounded up to the allocation granularity) on
    /// `device` and map every member's allocation, collectively over the
    /// host `group`, whose connections carry the handles. `host`
    /// identifies this host.
    pub(crate) fn allocate(
        device: DeviceId,
        bytes: usize,
        host: &str,
        group: &HostGroup,
    ) -> Result<Self> {
        let local = Storage::exportable(device, bytes, &[])?;
        let allocation: &DeviceAllocation = local
            .allocation()
            .ok_or_else(|| Error::Invariant("peer storage is device memory".into()))?;
        let rounded = allocation.len();
        let handle = allocation
            .export()?
            .ok_or_else(|| Error::Invariant("peer storage is exportable".into()))?;

        // Serve a descriptor until every member imported it.
        let (endpoint, grants) = serve(&handle)?;

        let payloads = group.all_gather_bytes(&encode(host, &handle, &endpoint))?;
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
                            .map_err(|error| {
                                Error::transport_source("duplicating a peer's descriptor", error)
                            }),
                    };
                }
                decode(host, payload)
            })
            .collect::<Result<Vec<_>>>();
        // Every member fetched its descriptors before any grant closes.
        let barrier = group.barrier();
        if let Some(grants) = grants {
            grants
                .close()
                .map_err(|error| Error::transport_source("closing the descriptor grants", error))?;
        }
        let handles = handles?;
        barrier?;

        let peers = Storage::mapped(PeerMapping::peers(device, &handles, rounded)?);
        Ok(Self {
            local,
            peers,
            bytes: rounded,
        })
    }

    /// This rank's allocation.
    pub(crate) fn local(&self) -> &Arc<Storage> {
        &self.local
    }

    /// Every member's allocation, `bytes_each()` apart in rank order.
    pub(crate) fn peers(&self) -> &Arc<Storage> {
        &self.peers
    }

    pub(crate) fn bytes_each(&self) -> usize {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fabric_payloads_round_trip_and_descriptors_stay_on_their_host() -> Result<()> {
        // A fabric handle containing zeros imports on any host.
        let mut bytes = [7; 64];
        bytes[3] = 0;
        let fabric = ShareableHandle::Fabric(bytes);
        let payload = encode("host-a", &fabric, "uniserve-peer-x");
        assert!(
            matches!(decode("host-b", &payload)?, ShareableHandle::Fabric(imported) if imported == bytes)
        );

        let file = std::fs::File::open("/dev/null")
            .map_err(|error| Error::transport_source("opening a stand-in descriptor", error))?;
        let descriptor = ShareableHandle::Descriptor(file.into());
        let payload = encode("host-a", &descriptor, "uniserve-peer-x");
        assert!(matches!(
            decode("host-b", &payload),
            Err(Error::Unsupported(_))
        ));

        assert!(decode("host-a", b"host-a\0uniserve-peer-x").is_err());
        assert!(decode("host-a", b"host-a\0uniserve-peer-x\0abc").is_err());
        Ok(())
    }

    #[test]
    fn a_descriptor_reaches_members_of_its_host_through_the_grant() -> Result<()> {
        use std::os::unix::fs::MetadataExt;

        // Any open file stands in for an allocation's process descriptor:
        // the grant passes the descriptor, not the allocation.
        let file = std::fs::File::open("/dev/null")
            .map_err(|error| Error::transport_source("opening a stand-in descriptor", error))?;
        let published = file
            .metadata()
            .map_err(|error| Error::transport_source("reading a descriptor's metadata", error))?;
        let handle = ShareableHandle::Descriptor(file.into());
        let (endpoint, grants) = serve(&handle)?;
        assert!(grants.is_some(), "a descriptor is served by a grant");

        // A member of the same host imports the descriptor from the grant.
        let ShareableHandle::Descriptor(received) =
            decode("host-a", &encode("host-a", &handle, &endpoint))?
        else {
            return Err(Error::State(
                "a descriptor payload imported as a fabric handle",
            ));
        };
        let received = std::fs::File::from(received)
            .metadata()
            .map_err(|error| Error::transport_source("reading a descriptor's metadata", error))?;
        assert_eq!(
            (received.dev(), received.ino()),
            (published.dev(), published.ino())
        );
        Ok(())
    }
}
