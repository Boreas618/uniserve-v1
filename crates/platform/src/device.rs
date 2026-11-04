//! Device ordinals, attributes, identity and storage information.

use std::fmt;

use crate::driver::driver;
use crate::{DeviceGuard, Result};

/// A CUDA device ordinal as numbered by the driver in this process (after
/// `CUDA_VISIBLE_DEVICES`). The same ordinal names the device in DLPack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeviceId(pub i32);

impl fmt::Display for DeviceId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "cuda:{}", self.0)
    }
}

/// The `CUdevice_attribute` values the worker queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum DeviceAttribute {
    /// Largest row pitch, in bytes, accepted by pitched copies.
    MaxPitch = 11,
    MultiprocessorCount = 16,
    ComputeCapabilityMajor = 75,
    ComputeCapabilityMinor = 76,
    VirtualMemoryManagementSupported = 102,
    /// Whether host storage registers for reads alone
    /// (`CU_MEMHOSTREGISTER_READ_ONLY`, [`crate::HostRegistration::read_only`]).
    ReadOnlyHostRegisterSupported = 113,
    HandleTypePosixFileDescriptorSupported = 103,
    HandleTypeFabricSupported = 128,
    /// The host NUMA node closest to the device, or -1 when the host has no
    /// NUMA affinity for it.
    HostNumaId = 134,
}

/// Device storage reported by `cuMemGetInfo`, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryInfo {
    pub free: u64,
    pub total: u64,
}

/// A device's 16-byte UUID. It identifies the same physical device in NVML,
/// whose ordinals ignore `CUDA_VISIBLE_DEVICES`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DeviceUuid(pub [u8; 16]);

impl fmt::Display for DeviceUuid {
    /// Formats as NVML's identifier, `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("GPU-")?;
        for (index, byte) in self.0.iter().enumerate() {
            if matches!(index, 4 | 6 | 8 | 10) {
                formatter.write_str("-")?;
            }
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// Start the CUDA profiler's capture range for `device`'s primary context
/// (`cuProfilerStart`), for external profilers launched with capture ranges.
pub fn profiler_start(device: DeviceId) -> Result<()> {
    let _guard = DeviceGuard::new(device)?;
    let driver = driver()?;
    // SAFETY: takes no arguments; applies to the current context.
    driver.check(unsafe { (driver.profiler_start)() }, "cuProfilerStart")
}

/// End the CUDA profiler's capture range (`cuProfilerStop`).
pub fn profiler_stop(device: DeviceId) -> Result<()> {
    let _guard = DeviceGuard::new(device)?;
    let driver = driver()?;
    // SAFETY: takes no arguments; applies to the current context.
    driver.check(unsafe { (driver.profiler_stop)() }, "cuProfilerStop")
}

impl DeviceId {
    /// The number of devices visible to this process.
    pub fn count() -> Result<i32> {
        let driver = driver()?;
        let mut count = 0;
        driver.check(
            // SAFETY: the driver writes one int to the local `count`.
            unsafe { (driver.device_get_count)(&mut count) },
            "cuDeviceGetCount",
        )?;
        Ok(count)
    }

    /// The driver's `CUdevice` handle for this ordinal.
    pub(crate) fn handle(self) -> Result<i32> {
        let driver = driver()?;
        let mut device = 0;
        driver.check(
            // SAFETY: the driver writes one CUdevice to the local `device` and
            // validates the ordinal.
            unsafe { (driver.device_get)(&mut device, self.0) },
            "cuDeviceGet",
        )?;
        Ok(device)
    }

    pub fn attribute(self, attribute: DeviceAttribute) -> Result<i32> {
        let driver = driver()?;
        let device = self.handle()?;
        let mut value = 0;
        driver.check(
            // SAFETY: the driver writes one int to the local `value` for a valid
            // attribute and device.
            unsafe { (driver.device_get_attribute)(&mut value, attribute as i32, device) },
            "cuDeviceGetAttribute",
        )?;
        Ok(value)
    }

    pub fn uuid(self) -> Result<DeviceUuid> {
        let driver = driver()?;
        let device = self.handle()?;
        let mut bytes = [0; 16];
        driver.check(
            // SAFETY: the driver writes the 16-byte CUuuid to the local `bytes`.
            unsafe { (driver.device_get_uuid)(&mut bytes, device) },
            "cuDeviceGetUuid_v2",
        )?;
        Ok(DeviceUuid(bytes))
    }

    /// Whether this device can directly address storage resident on `peer`.
    pub fn can_access_peer(self, peer: DeviceId) -> Result<bool> {
        let driver = driver()?;
        let (device, peer) = (self.handle()?, peer.handle()?);
        let mut accessible = 0;
        driver.check(
            // SAFETY: the driver writes one int to the local `accessible` for two
            // valid devices.
            unsafe { (driver.device_can_access_peer)(&mut accessible, device, peer) },
            "cuDeviceCanAccessPeer",
        )?;
        Ok(accessible != 0)
    }

    /// Fails with the error that left the device's primary context
    /// unusable, such as an illegal memory access or a device-side assert,
    /// which the driver returns from every call after it; passes while the
    /// context is usable. Waits for nothing on the device.
    pub fn context_error(self) -> Result<()> {
        self.memory_info().map(drop)
    }

    /// Free and total device storage, queried in the device's primary context.
    pub fn memory_info(self) -> Result<MemoryInfo> {
        let _guard = DeviceGuard::new(self)?;
        let driver = driver()?;
        let (mut free, mut total) = (0, 0);
        driver.check(
            // SAFETY: the driver writes two size_t values to the locals `free` and
            // `total` for the context the guard made current.
            unsafe { (driver.mem_get_info)(&mut free, &mut total) },
            "cuMemGetInfo_v2",
        )?;
        Ok(MemoryInfo {
            free: free as u64,
            total: total as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuids_format_as_nvml_identifiers() {
        let uuid = DeviceUuid([
            0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
            0xcd, 0xef,
        ]);

        assert_eq!(uuid.to_string(), "GPU-12345678-9abc-def0-0123-456789abcdef");
    }
}
