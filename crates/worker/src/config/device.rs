//! Device names and tower placement, parsed without the numerical library.
//!
//! A rank names devices with torch's spelling (`cpu`, `cuda`, `cuda:N`); the
//! worker accepts exactly that grammar. A bare `cuda` resolves to `cuda:0`
//! because a rank scoped by `CUDA_VISIBLE_DEVICES` owns the visible device
//! zero.

use uniserve_cuda::DeviceId;

use super::DescriptorError;
use crate::ffi::MemoryKind;

/// A device a rank computes on, parsed once from the launch descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Device {
    Cpu,
    /// A CUDA device within the process's visible devices.
    Cuda(DeviceId),
}

impl Device {
    /// Parse a torch device name. `option` names the launch setting in errors.
    pub(crate) fn parse(value: &str, option: &str) -> Result<Self, DescriptorError> {
        let invalid =
            || DescriptorError::new(format!("{option} must name a torch device, got {value:?}"));
        match value.trim() {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda(DeviceId(0))),
            name => {
                let index = name.strip_prefix("cuda:").ok_or_else(invalid)?;
                let index = index.parse::<i32>().map_err(|_| invalid())?;
                if index < 0 {
                    return Err(invalid());
                }
                Ok(Self::Cuda(DeviceId(index)))
            }
        }
    }

    pub(crate) fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    /// The CUDA device, if this is one.
    pub(crate) fn cuda(self) -> Option<DeviceId> {
        match self {
            Self::Cpu => None,
            Self::Cuda(device) => Some(device),
        }
    }

    /// The memory a binding on this device computes in.
    pub(crate) fn memory(self) -> MemoryKind {
        match self {
            Self::Cpu => MemoryKind::Host,
            Self::Cuda(device) => MemoryKind::Cuda(device),
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Cuda(device) => write!(formatter, "{device}"),
        }
    }
}

/// Resolve the generation (flow) tower device from `worker_config.mesh`.
///
/// The declaration is `tower=gen:<device>`: the text tower runs on the
/// rank's device, and model loading places the flow route, denoiser and
/// image decoder modules on `<device>`, which must differ from it. An empty
/// declaration places no tower apart.
pub(crate) fn generation_device(
    mesh: &str,
    device: Device,
) -> Result<Option<Device>, DescriptorError> {
    let mesh = mesh.trim();
    if mesh.is_empty() {
        return Ok(None);
    }

    let target = mesh
        .split_once('=')
        .filter(|(key, _)| key.trim().eq_ignore_ascii_case("tower"))
        .and_then(|(_, selection)| selection.split_once(':'))
        .filter(|(name, target)| {
            name.trim().eq_ignore_ascii_case("gen") && !target.trim().is_empty()
        })
        .map(|(_, target)| target.trim())
        .ok_or_else(|| {
            DescriptorError::new(format!(
                "worker_config.mesh must be tower=gen:<device>, got {mesh:?}"
            ))
        })?;
    let generation = Device::parse(target, "worker_config.mesh tower gen")?;
    if generation == device {
        return Err(DescriptorError::new(
            "worker_config.mesh tower gen must name a device other than the rank's",
        ));
    }
    Ok(Some(generation))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_cuda_names_the_first_visible_device() {
        assert_eq!(
            Device::parse("cuda", "rank device").ok(),
            Some(Device::Cuda(DeviceId(0)))
        );
        assert_eq!(
            Device::parse("cuda:3", "rank device").ok(),
            Some(Device::Cuda(DeviceId(3)))
        );
        assert_eq!(Device::parse("cpu", "rank device").ok(), Some(Device::Cpu));
        assert!(Device::parse("tpu", "rank device").is_err());
        assert!(Device::parse("cuda:-1", "rank device").is_err());
    }

    /// `tower=gen:<device>` places the generation tower on another device
    /// than the rank's; the text tower always runs on the rank's own.
    #[test]
    fn tower_placement_requires_a_distinct_generation_device() {
        let rank = Device::Cuda(DeviceId(0));
        assert_eq!(
            generation_device("tower=gen:cuda:1", rank).ok(),
            Some(Some(Device::Cuda(DeviceId(1))))
        );
        assert!(generation_device("tower=gen:cuda:0", rank).is_err());
        assert!(generation_device("tower=text:cuda:0;gen:cuda:1", rank).is_err());
        assert!(generation_device("other=1", rank).is_err());
        assert_eq!(generation_device(" ", rank).ok(), Some(None));
    }
}
