//! Device names and tower placement, parsed without the numerical library.
//!
//! A rank names devices with torch's spelling (`cpu`, `cuda`, `cuda:N`); the
//! worker accepts exactly that grammar. A bare `cuda` resolves to `cuda:0`
//! because a rank scoped by `CUDA_VISIBLE_DEVICES` owns the visible device
//! zero.

use std::collections::BTreeMap;

use super::DescriptorError;

/// A device a rank computes on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Device {
    Cpu,
    /// A CUDA ordinal within the process's visible devices.
    Cuda(i32),
}

impl Device {
    /// Parse a torch device name. `option` names the launch setting in errors.
    pub fn parse(value: &str, option: &str) -> Result<Self, DescriptorError> {
        let invalid =
            || DescriptorError::new(format!("{option} must name a torch device, got {value:?}"));
        match value.trim() {
            "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda(0)),
            name => {
                let index = name.strip_prefix("cuda:").ok_or_else(invalid)?;
                let index = index.parse::<i32>().map_err(|_| invalid())?;
                if index < 0 {
                    return Err(invalid());
                }
                Ok(Self::Cuda(index))
            }
        }
    }

    pub fn is_cuda(self) -> bool {
        matches!(self, Self::Cuda(_))
    }

    /// The CUDA ordinal, if this is a CUDA device.
    pub fn cuda_index(self) -> Option<i32> {
        match self {
            Self::Cpu => None,
            Self::Cuda(index) => Some(index),
        }
    }
}

impl std::fmt::Display for Device {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Cuda(index) => write!(formatter, "cuda:{index}"),
        }
    }
}

/// Resolve the generation (flow) tower device from `--mesh`.
///
/// The only key is `tower=text:<device>;gen:<device>`. The text tower must
/// be the rank's device and the generation tower a different one.
pub fn generation_device(mesh: &str, device: Device) -> Result<Option<Device>, DescriptorError> {
    if mesh.trim().is_empty() {
        return Ok(None);
    }

    let mut generation = None;
    for entry in mesh.split(',') {
        let (key, value) = entry.split_once('=').ok_or_else(|| {
            DescriptorError::new(format!(
                "invalid --mesh entry {entry:?}; expected key=value"
            ))
        })?;
        let key = key.trim().to_lowercase().replace('_', "-");
        if key != "tower" {
            return Err(DescriptorError::new(format!("unknown --mesh key {key:?}")));
        }
        if generation.is_some() {
            return Err(DescriptorError::new("duplicate --mesh key tower"));
        }

        let shape = || DescriptorError::new("tower params must use text:<device>;gen:<device>");
        let mut parameters = BTreeMap::new();
        for part in value.split(';') {
            let (name, target) = part.split_once(':').ok_or_else(shape)?;
            let name = name.trim().to_lowercase();
            if !matches!(name.as_str(), "text" | "gen") || target.trim().is_empty() {
                return Err(shape());
            }
            if parameters.insert(name.clone(), target.trim()).is_some() {
                return Err(DescriptorError::new(format!(
                    "duplicate tower params {name:?}"
                )));
            }
        }

        let target = parameters
            .get("gen")
            .ok_or_else(|| DescriptorError::new("tower params requires gen:<device>"))?;
        let text = match parameters.get("text") {
            Some(text) => Device::parse(text, "--mesh tower text")?,
            None => device,
        };
        if text != device {
            return Err(DescriptorError::new(
                "text expert device must match the Worker rank device",
            ));
        }
        let target = Device::parse(target, "--mesh tower gen")?;
        if target == text {
            return Err(DescriptorError::new(
                "tower text and gen devices must be different",
            ));
        }
        generation = Some(target);
    }
    Ok(generation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bare_cuda_names_the_first_visible_device() {
        assert_eq!(
            Device::parse("cuda", "--device").ok(),
            Some(Device::Cuda(0))
        );
        assert_eq!(
            Device::parse("cuda:3", "--device").ok(),
            Some(Device::Cuda(3))
        );
        assert_eq!(Device::parse("cpu", "--device").ok(), Some(Device::Cpu));
        assert!(Device::parse("tpu", "--device").is_err());
        assert!(Device::parse("cuda:-1", "--device").is_err());
    }

    #[test]
    fn tower_placement_requires_a_distinct_generation_device() {
        let text = Device::Cuda(0);
        assert_eq!(
            generation_device("tower=text:cuda:0;gen:cuda:1", text).ok(),
            Some(Some(Device::Cuda(1)))
        );
        assert!(generation_device("tower=gen:cuda:0", text).is_err());
        assert!(generation_device("tower=text:cuda:1;gen:cuda:2", text).is_err());
        assert!(generation_device("other=1", text).is_err());
        assert_eq!(generation_device(" ", text).ok(), Some(None));
    }
}
