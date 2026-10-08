//! The rank's command line.
//!
//! argv carries only the process identity and the descriptor's path, so a
//! running rank stays identifiable from the process table; every launch value
//! is in the descriptor.

use std::path::PathBuf;

use crate::config::DescriptorError;

/// Usage line printed with an argument or descriptor error.
pub const USAGE: &str = "usage: python -m uniserve_models.worker --launch-descriptor PATH \
                         [--worker-id ID] [--rank R] [--world-size N]";

/// Parsed command-line arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arguments {
    pub descriptor: PathBuf,
    /// Identity flags; informational, the descriptor is authoritative.
    pub worker_id: Option<String>,
    pub rank: Option<u32>,
    pub world_size: Option<u32>,
}

impl Arguments {
    /// Parse `--flag value` and `--flag=value` forms.
    pub fn parse<S: AsRef<str>>(argv: &[S]) -> Result<Self, DescriptorError> {
        let mut descriptor = None;
        let mut worker_id = None;
        let mut rank = None;
        let mut world_size = None;
        let mut items = argv.iter().map(AsRef::as_ref);
        while let Some(item) = items.next() {
            let (flag, inline) = match item.split_once('=') {
                Some((flag, value)) => (flag, Some(value.to_owned())),
                None => (item, None),
            };
            let mut value = || {
                inline
                    .clone()
                    .or_else(|| items.next().map(str::to_owned))
                    .ok_or_else(|| {
                        DescriptorError::new(format!("argument {flag}: expected one argument"))
                    })
            };
            let number = |name: &str, value: String| {
                value.parse::<u32>().map_err(|_| {
                    DescriptorError::new(format!("argument {name}: invalid int value: {value:?}"))
                })
            };
            match flag {
                "--launch-descriptor" => descriptor = Some(PathBuf::from(value()?)),
                "--worker-id" => worker_id = Some(value()?),
                "--rank" => rank = Some(number(flag, value()?)?),
                "--world-size" => world_size = Some(number(flag, value()?)?),
                other => {
                    return Err(DescriptorError::new(format!(
                        "unrecognized arguments: {other}"
                    )));
                }
            }
        }
        Ok(Self {
            descriptor: descriptor.ok_or_else(|| {
                DescriptorError::new("the following arguments are required: --launch-descriptor")
            })?,
            worker_id,
            rank,
            world_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_flags_accompany_the_descriptor() {
        let arguments = Arguments::parse(&[
            "--worker-id",
            "text",
            "--rank=1",
            "--world-size",
            "2",
            "--launch-descriptor",
            "/tmp/launch.json",
        ])
        .unwrap();
        assert_eq!(arguments.descriptor, PathBuf::from("/tmp/launch.json"));
        assert_eq!(arguments.worker_id.as_deref(), Some("text"));
        assert_eq!((arguments.rank, arguments.world_size), (Some(1), Some(2)));
    }

    #[test]
    fn a_missing_descriptor_or_unknown_flag_is_a_usage_error() {
        assert!(Arguments::parse(&["--rank", "0"]).is_err());
        assert!(Arguments::parse(&["--launch-descriptor", "x", "--verbose"]).is_err());
        assert!(Arguments::parse(&["--launch-descriptor"]).is_err());
        assert!(Arguments::parse(&["--launch-descriptor", "x", "--rank", "one"]).is_err());
    }
}
