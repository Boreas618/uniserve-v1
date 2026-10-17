//! The rank's command line.
//!
//! argv carries the head's address, the rank's identity and the launch it
//! belongs to: the rank connects to the head, registers under that identity
//! and launch, and receives every other launch value in the launch
//! descriptor the head answers with.

use crate::config::DescriptorError;

/// Usage line printed with an argument or descriptor error.
pub(crate) const USAGE: &str =
    "usage: python -m uniserve.worker --head ADDRESS --worker-id ID --rank R --generation G";

/// Parsed command-line arguments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Arguments {
    /// The head's address: `unix:<path>` or `<host>:<port>`.
    pub head: String,
    /// Worker group the rank was started for.
    pub worker_id: String,
    /// Rank within the group's process world.
    pub rank: u32,
    /// The generation of the group's launch the rank was started for.
    pub generation: u64,
}

impl Arguments {
    /// Parse `--flag value` and `--flag=value` forms.
    pub fn parse<S: AsRef<str>>(argv: &[S]) -> Result<Self, DescriptorError> {
        let mut head = None;
        let mut worker_id = None;
        let mut rank = None;
        let mut generation = None;
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
            fn number<T: std::str::FromStr>(
                name: &str,
                value: String,
            ) -> Result<T, DescriptorError> {
                value.parse().map_err(|_| {
                    DescriptorError::new(format!("argument {name}: invalid int value: {value:?}"))
                })
            }
            match flag {
                "--head" => head = Some(value()?),
                "--worker-id" => worker_id = Some(value()?),
                "--rank" => rank = Some(number(flag, value()?)?),
                "--generation" => generation = Some(number(flag, value()?)?),
                other => {
                    return Err(DescriptorError::new(format!(
                        "unrecognized arguments: {other}"
                    )));
                }
            }
        }
        let required = |name: &str| {
            DescriptorError::new(format!("the following arguments are required: {name}"))
        };
        Ok(Self {
            head: head.ok_or_else(|| required("--head"))?,
            worker_id: worker_id.ok_or_else(|| required("--worker-id"))?,
            rank: rank.ok_or_else(|| required("--rank"))?,
            generation: generation.ok_or_else(|| required("--generation"))?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_head_and_the_identity_are_parsed_in_either_form() {
        let arguments = Arguments::parse(&[
            "--worker-id",
            "text",
            "--rank=1",
            "--head",
            "unix:/tmp/head.sock",
            "--generation=2",
        ])
        .unwrap();
        assert_eq!(arguments.head, "unix:/tmp/head.sock");
        assert_eq!(arguments.worker_id, "text");
        assert_eq!(arguments.rank, 1);
        assert_eq!(arguments.generation, 2);
    }

    #[test]
    fn a_missing_identity_or_unknown_flag_is_a_usage_error() {
        let complete = [
            "--head",
            "h:1",
            "--worker-id",
            "w",
            "--rank",
            "0",
            "--generation",
            "1",
        ];
        assert!(Arguments::parse(&complete).is_ok());
        for missing in 0..4 {
            let mut argv = complete.to_vec();
            argv.drain(2 * missing..2 * missing + 2);
            assert!(Arguments::parse(&argv).is_err());
        }
        assert!(Arguments::parse(&[&complete[..], &["--verbose"]].concat()).is_err());
        assert!(Arguments::parse(&complete[..7]).is_err());
        assert!(Arguments::parse(&[&complete[..7], &["one"]].concat()).is_err());
    }
}
