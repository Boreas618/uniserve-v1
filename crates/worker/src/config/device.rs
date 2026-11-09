//! Device names and tower layout, parsed without the numerical library.
//!
//! A rank names devices with torch's spelling (`cpu`, `cuda`, `cuda:N`); the
//! worker accepts exactly that grammar. A bare `cuda` resolves to `cuda:0`
//! because a rank scoped by `CUDA_VISIBLE_DEVICES` owns the visible device
//! zero.

use platform::DeviceId;

use super::DescriptorError;

/// A device a rank computes on, parsed once from the launch descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Device {
    Cpu,
    /// A CUDA device within the process's visible devices.
    Cuda(DeviceId),
}

impl Device {
    /// Parse a torch device name. `option` names the launch setting in errors.
    pub fn parse(value: &str, option: &str) -> Result<Self, DescriptorError> {
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
}

impl std::fmt::Display for Device {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Cuda(device) => write!(formatter, "{device}"),
        }
    }
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
}
