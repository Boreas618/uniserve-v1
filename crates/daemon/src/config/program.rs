//! The programs a policy package runs (`[programs.<name>]`): where each one's module comes
//! from, how often its host steps it, and the arguments it is created with.
//!
//! A program's kind is never stated here: the module states it (`program::kind`), so the same
//! entry runs a control program on the leader daemon, a routing program on the leader's request
//! path, a scheduling program in every scheduler, or a KV program in every Storage Server.

use std::fmt;
use std::str::FromStr;
use std::time::Duration;

use proto::config::values;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// The prefix of a source that names a program installed at the deployment's Build Server.
const INSTALLED: &str = "uniserve:";

/// The directory of a policy package that holds its programs' crates.
const PACKAGE: &str = "programs/";

/// One program of a policy package.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Program {
    /// Where its module comes from.
    pub source: Source,
    /// How often its host calls `step`: a control program on the leader daemon and a KV
    /// program's tiering hook on the leader node's Storage Server. A routing program's state is
    /// saved this often. A control or KV program states it; the others need not.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "values::optional_duration"
    )]
    pub period: Option<Duration>,
    /// The table the program's `new` receives as its `Args`.
    #[serde(default, skip_serializing_if = "toml::Table::is_empty")]
    pub args: toml::Table,
}

/// Where a program's module comes from: written `programs/<dir>` or `uniserve:<package>`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Source {
    /// The crate in the package's `programs/<dir>`, which the deployment's Build Server compiles
    /// to a wasm32 module.
    Package(String),
    /// A program package installed at the deployment's Build Server, at the revision the apply
    /// fixes.
    Installed(String),
}

impl fmt::Display for Source {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Package(directory) => write!(formatter, "{PACKAGE}{directory}"),
            Self::Installed(package) => write!(formatter, "{INSTALLED}{package}"),
        }
    }
}

/// Whether `name` is one plain name: letters, digits, `-`, `_` and `.`, not only dots. A
/// program's directory, its installed package and its own name are such names, so none of them
/// leaves the directory it names a file of.
pub fn plain(name: &str) -> bool {
    !name.is_empty()
        && name.chars().any(|character| character != '.')
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "-_.".contains(character))
}

impl FromStr for Source {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, String> {
        let (source, name) = if let Some(package) = text.strip_prefix(INSTALLED) {
            (Self::Installed(package.to_owned()), package)
        } else if let Some(directory) = text.strip_prefix(PACKAGE) {
            (Self::Package(directory.to_owned()), directory)
        } else {
            return Err(format!(
                "`{text}` is no program source: write `{PACKAGE}<dir>` for a crate of the \
                 package or `{INSTALLED}<package>` for a package installed at the Build Server"
            ));
        };
        if !plain(name) {
            return Err(format!(
                "`{text}` names `{name}`, which is not one plain name (letters, digits, `-`, `_`, \
                 `.`)"
            ));
        }
        Ok(source)
    }
}

impl Serialize for Source {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for Source {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source reads from either of its two forms and prints as it reads; anything else, and a
    /// name that could leave its directory, is refused.
    #[test]
    fn a_source_is_a_package_crate_or_an_installed_package() {
        let package: Source = "programs/chunk".parse().unwrap();
        assert_eq!(package, Source::Package("chunk".to_owned()));
        assert_eq!(package.to_string(), "programs/chunk");
        let installed: Source = "uniserve:arrival".parse().unwrap();
        assert_eq!(installed, Source::Installed("arrival".to_owned()));
        assert_eq!(installed.to_string(), "uniserve:arrival");

        for refused in [
            "chunk",
            "programs/",
            "programs/../escape",
            "programs/a/b",
            "uniserve:",
            "uniserve:..",
        ] {
            assert!(refused.parse::<Source>().is_err(), "{refused}");
        }
    }

    /// A program reads from its `config.toml` table, its period as a duration and its arguments
    /// as the table they are.
    #[test]
    fn a_program_reads_from_its_table() {
        let program: Program = toml::from_str(
            "source = \"programs/chunk\"\nperiod = \"10s\"\nargs = { low = 512, tpot = \"50ms\" }\n",
        )
        .unwrap();
        assert_eq!(program.source, Source::Package("chunk".to_owned()));
        assert_eq!(program.period, Some(Duration::from_secs(10)));
        assert_eq!(program.args["low"].as_integer(), Some(512));
        assert!(
            toml::from_str::<Program>("source = \"uniserve:arrival\"\nkind = \"x\"\n").is_err()
        );
    }
}
