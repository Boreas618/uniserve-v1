//! A rank's mapping of its part of one host region of its host's Storage Server.
//!
//! The host regions of the buffer pool are the Storage Server's memory: one memfd per region and
//! rank index, which holds the rank's part of every unit at the same index, unit `u` at byte
//! `u × bytes` (`storage::protocol::Request::HostPart`). Ranks of one index share the part, in
//! every epoch that names the region. The KV cache maps its part of each cache group's host
//! region, one page a unit, and the media memory its parts of the encoder and latent regions;
//! each maps it here, writable and registered for device copies, and addresses its units
//! through it. The buffer pool decides which units hold what; the rank copies.

use std::sync::Arc;

use platform::{DeviceAttribute, DeviceId};

use crate::memory::allocation::Storage;
use crate::memory::client::StorageClient;
use crate::worker::{Error, Result};

/// The host NUMA node closest to `device`, on which a part the device copies to and from lies,
/// so the copies cross no link between nodes; `None` when the device does not state one.
pub(crate) fn numa_node(device: DeviceId) -> Option<u32> {
    device
        .attribute(DeviceAttribute::HostNumaId)
        .ok()
        .and_then(|node| u32::try_from(node).ok())
}

/// The pages a pool of request state holds when the requests in flight hold `running` pages:
/// theirs and as many again for preempted requests waiting to resume.
///
/// A request that resumes is admitted before every request that starts, so the requests still
/// to resume hold at most what the latest preemption stored, and an epoch on trial stores its
/// own while the epoch it succeeds keeps its pages for a rollback.
pub(crate) fn resuming_pages(running: u64) -> u64 {
    2 * running
}

/// The rank's part of one host region, mapped into the rank and registered with its device.
pub(crate) struct HostPart {
    /// The region's name in the session's layout, which errors name.
    region: String,
    storage: Arc<Storage>,
    /// How many units the part holds.
    units: u32,
    /// The bytes of one unit of the part.
    bytes: u64,
}

impl HostPart {
    /// Takes the rank's part of the host region `region` at the rank's host's Storage Server,
    /// `bytes` of every unit, and maps it registered with `device`. The server creates the part
    /// on `numa_node` at the first request of the rank index, and returns the part it holds
    /// otherwise, at its size.
    ///
    /// # Errors
    ///
    /// When the server refuses the part (no open session states the region, or the parts'
    /// bytes of a unit exceed the region's unit size), the part holds no unit, or its storage
    /// cannot be mapped.
    pub(crate) fn map(
        client: &StorageClient,
        device: DeviceId,
        region: &str,
        bytes: u64,
        numa_node: Option<u32>,
    ) -> Result<Self> {
        let (units, descriptor) = client.host_part(region, bytes, numa_node)?;
        if units == 0 {
            return Err(Error::Resource(
                "the rank's part of a host region holds no unit",
            ));
        }

        let length = u64::from(units)
            .checked_mul(bytes)
            .and_then(|length| usize::try_from(length).ok())
            .ok_or(Error::Resource(
                "a rank's part of a host region exceeds the address space",
            ))?;
        let storage = Storage::map_host(descriptor, length, device, true)?;
        tracing::info!(
            region,
            units,
            bytes,
            numa_node,
            "mapped the rank's part of a host region"
        );

        Ok(Self {
            region: region.to_owned(),
            storage,
            units,
            bytes,
        })
    }

    /// How many units the part holds.
    #[expect(dead_code, reason = "no caller in the engine")]
    pub(crate) fn units(&self) -> u32 {
        self.units
    }

    /// The bytes of one unit of the part.
    #[expect(dead_code, reason = "no caller in the engine")]
    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The host address of unit `unit`, the first of [`Self::bytes`] bytes a device copy reads
    /// or writes.
    ///
    /// # Errors
    ///
    /// When the unit lies outside the part.
    pub(crate) fn address(&self, unit: u64) -> Result<u64> {
        if unit >= u64::from(self.units) {
            return Err(Error::Invalid(format!(
                "unit {unit} lies outside the rank's part of host region {} of {} units",
                self.region, self.units
            )));
        }
        Ok(self.storage.address() + unit * self.bytes)
    }
}
