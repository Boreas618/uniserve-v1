//! The programs' saved states: one file per program in the deployment's `program-state/`
//! directory, which belongs to no revision (`specs/configuration.md` §8).
//!
//! A state is what the program's `save` export replied, kept with the `args` the program was
//! created with, written whole through a file renamed over the last one, so a crash leaves the
//! previous state or the new one: `<name>.state`, the JSON `{"args": <table>, "state": <value>}`.
//! It is read back when the program's instance is created again, after a restart of the leader
//! or a change of the program's module, and only for the same `args`: a program keeps its
//! arguments in its own fields, so a state of other arguments would leave the new ones unread,
//! and a program of new arguments is created from them. A state that does not decode as the
//! program's current type is set aside by the host (`program::Error::State`) and the program is
//! created anew.

use std::path::PathBuf;

use anyhow::Context as _;

/// The extension of a saved state's file.
const EXTENSION: &str = "state";

/// A deployment's `program-state/` directory on this node.
#[derive(Debug, Clone)]
pub(crate) struct States {
    directory: PathBuf,
}

impl States {
    /// The states kept in `directory`.
    pub(crate) fn new(directory: PathBuf) -> Self {
        Self { directory }
    }

    /// The file of program `name`'s state. A program's name is one plain name
    /// (`crate::config::program::plain`), so the file lies in the directory.
    fn path(&self, name: &str) -> PathBuf {
        self.directory.join(format!("{name}.{EXTENSION}"))
    }

    /// Program `name`'s saved state, when it saved one while created with `args`; `None`
    /// otherwise.
    ///
    /// # Errors
    ///
    /// Fails when the file exists and cannot be read, or holds no saved state.
    pub(crate) fn load(&self, name: &str, args: &toml::Table) -> anyhow::Result<Option<Vec<u8>>> {
        let path = self.path(name);
        let bytes = match std::fs::read(&path) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => {
                return Err(error).with_context(|| format!("reading {}", path.display()));
            }
        };
        let saved: Saved = serde_json::from_slice(&bytes)
            .with_context(|| format!("reading {}", path.display()))?;
        if saved.args != *args {
            return Ok(None);
        }
        Ok(Some(serde_json::to_vec(&saved.state)?))
    }

    /// Replaces program `name`'s saved state with `state`, which its `save` replied while it ran
    /// with `args`, synced.
    ///
    /// # Errors
    ///
    /// Fails when the state is no JSON or the file cannot be written.
    pub(crate) fn save(&self, name: &str, args: &toml::Table, state: &[u8]) -> anyhow::Result<()> {
        let saved = Saved {
            args: args.clone(),
            state: serde_json::from_slice(state).context("the program's state is no JSON")?,
        };
        crate::policy::revision::write_synced(&self.path(name), &serde_json::to_vec(&saved)?)
    }
}

/// A saved state's file.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct Saved {
    /// The arguments the program ran with.
    args: toml::Table,
    /// What its `save` replied.
    state: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A state reads back as saved for the arguments it was saved with, a later save replaces
    /// it, and a program that saved none, or saved one under other arguments, has none.
    #[test]
    fn a_state_reads_back_for_its_arguments() {
        let directory = tempfile::tempdir().unwrap();
        let states = States::new(directory.path().join("program-state"));
        let args: toml::Table = toml::from_str("low = 512").unwrap();
        assert_eq!(states.load("chunk", &args).unwrap(), None);

        states.save("chunk", &args, b"{\"changes\":1}").unwrap();
        states.save("chunk.v2", &args, b"[2]").unwrap();
        states.save("chunk", &args, b"{\"changes\":2}").unwrap();
        assert_eq!(
            states.load("chunk", &args).unwrap().as_deref(),
            Some(&b"{\"changes\":2}"[..])
        );
        assert_eq!(
            states.load("chunk.v2", &args).unwrap().as_deref(),
            Some(&b"[2]"[..])
        );

        let other: toml::Table = toml::from_str("low = 1024").unwrap();
        assert_eq!(states.load("chunk", &other).unwrap(), None);
    }
}
