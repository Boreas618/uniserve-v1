//! The GPUs the Transfer Server copies on, as this process numbers them: the GPUs its host's
//! calibration states ([`super::calibration`]), found among those the driver shows by their
//! UUIDs, or every GPU the driver shows when the calibration step measures them.
//!
//! Discovery creates no CUDA context: it reads each GPU's identity, its closest host NUMA node,
//! the GPUs whose copy engines can address its storage, and whether it registers host storage for
//! reads alone.

use platform::{DeviceAttribute, DeviceId, HostRegistration};

/// One GPU, as this process numbers it.
pub(super) struct Gpu {
    pub(super) device: DeviceId,
    pub(super) uuid: [u8; 16],
    /// The host NUMA node closest to it.
    pub(super) numa_node: Option<u32>,
    /// The GPUs whose copy engines can address its storage, itself first.
    pub(super) readers: Vec<DeviceId>,
    /// Whether it registers host storage for reads alone
    /// (`CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED`).
    pub(super) read_only_registration: bool,
}

/// The GPUs this process copies on.
pub(super) struct Gpus {
    gpus: Vec<Gpu>,
}

impl Gpus {
    /// The GPUs of `uuids`, among those the driver shows; none without loading the driver when
    /// `uuids` is empty, so a Transfer Server calibrated without GPUs copies host storage alone.
    ///
    /// # Errors
    ///
    /// Fails when the driver cannot be loaded, does not show a GPU of `uuids`, or a GPU's
    /// identity or peers cannot be read.
    pub(super) fn of(uuids: &[[u8; 16]]) -> Result<Self, crate::Error> {
        if uuids.is_empty() {
            tracing::info!("no GPU is calibrated: the Transfer Server copies host storage alone");
            return Ok(Self { gpus: Vec::new() });
        }
        let shown = Self::shown()?;
        let mut devices = Vec::with_capacity(uuids.len());
        for uuid in uuids {
            let found = shown.iter().find(|gpu| gpu.uuid == *uuid).ok_or_else(|| {
                crate::Error::Calibration(format!(
                    "names GPU {}, which the driver does not show",
                    platform::DeviceUuid(*uuid)
                ))
            })?;
            devices.push(found.device);
        }
        Ok(Self::describe(&devices)?)
    }

    /// Every GPU the driver shows.
    ///
    /// # Errors
    ///
    /// Fails when the driver cannot be loaded or a GPU's identity or peers cannot be read.
    pub(super) fn shown() -> Result<Self, platform::Error> {
        let devices: Vec<DeviceId> = (0..DeviceId::count()?).map(DeviceId).collect();
        Self::describe(&devices)
    }

    /// `devices`, each with its identity, NUMA node, readers among `devices`, and registration
    /// capability.
    fn describe(devices: &[DeviceId]) -> Result<Self, platform::Error> {
        let mut gpus = Vec::with_capacity(devices.len());
        for &device in devices {
            let mut readers = vec![device];
            for &peer in devices {
                if peer != device && peer.can_access_peer(device)? {
                    readers.push(peer);
                }
            }
            gpus.push(Gpu {
                device,
                uuid: device.uuid()?.0,
                numa_node: device
                    .attribute(DeviceAttribute::HostNumaId)
                    .ok()
                    .and_then(|node| u32::try_from(node).ok()),
                readers,
                read_only_registration: HostRegistration::read_only_supported(device)?,
            });
        }
        Ok(Self { gpus })
    }

    pub(super) fn by_uuid(&self, uuid: [u8; 16]) -> Option<&Gpu> {
        self.gpus.iter().find(|gpu| gpu.uuid == uuid)
    }

    pub(super) fn by_device(&self, device: DeviceId) -> Option<&Gpu> {
        self.gpus.iter().find(|gpu| gpu.device == device)
    }

    /// Every GPU.
    pub(super) fn iter(&self) -> impl Iterator<Item = &Gpu> + '_ {
        self.gpus.iter()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.gpus.is_empty()
    }
}
