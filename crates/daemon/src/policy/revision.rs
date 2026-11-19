//! The deployment's revisions: every converged package, kept in the deployment's persistent state
//! directory, and the link to the one that serves (`specs/configuration.md` §8).
//!
//! ```text
//! <state>/
//!   revisions/
//!     41/  config.toml  programs/  lock.toml  revision.toml
//!     42/  config.toml  programs/  lock.toml  revision.toml
//!   current -> revisions/41
//!   program-state/
//! ```
//!
//! A change writes its revision's directory, synced, before it converges ([`Revisions::begin`]);
//! once it converged, `current` is replaced by a link to it in one rename
//! ([`Revisions::converged`]), and a change that did not converge records why and leaves
//! `current` where it was ([`Revisions::ended`]). So the disk and the serving deployment
//! correspond at every moment: what serves is the revision `current` names, or the one that
//! converges. A leader that crashed while a revision converged finds it without a result when it
//! opens the directory again and records it abandoned. Nix keeps its generations the same way.
//!
//! The directory is per deployment and node, under the user's state directory
//! (`$XDG_STATE_HOME`, else `~/.local/state`), never the runtime directory, which ends with the
//! user's session.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::{Context as _, ensure};
use proto::module::ModuleSet;
use proto::reload::{Change, Effect, Effects};
use serde::{Deserialize, Serialize};

use super::Package;

/// The file of a revision that records its change.
const RECORD: &str = "revision.toml";

/// The file of a revision that holds the module revisions its apply fixed.
const LOCK: &str = "lock.toml";

/// The link to the revision that serves.
const CURRENT: &str = "current";

/// The directory of the revisions.
const REVISIONS: &str = "revisions";

/// The directory of the programs' saved states, which belong to no revision.
pub const PROGRAM_STATE: &str = "program-state";

/// The revisions a state directory keeps when no option states another number, beside the
/// current one and the one that converges.
pub const KEPT: usize = 64;

/// How a revision's change ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// The change is converging; nothing decided it yet.
    Converging,
    /// The deployment converged to it, and `current` names it.
    Converged,
    /// The change was refused or failed before it served, or its trial rolled it back; the
    /// deployment serves what `current` names.
    RolledBack,
    /// The leader ended while it converged, and it never served.
    Abandoned,
}

/// One option a revision changed: its path, how its change took effect, and its value before and
/// after, as its `config.toml` writes it (absent when the option is at its default).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Changed {
    pub path: String,
    pub effect: Effect,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous: Option<toml::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next: Option<toml::Value>,
}

/// What a revision's `revision.toml` records: when and by whom it was written, what it changed
/// and how, and how its change ended.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    /// The revision's number.
    pub revision: u64,
    /// When the change began, in RFC 3339.
    pub time: String,
    /// Who wrote it: `start` for the package a node started with, `operator` for a
    /// `PUT /admin/policy`, or the name of the program whose output it is.
    pub writer: String,
    /// The heaviest effect of the change; absent for a change of nothing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effect: Option<Effect>,
    /// How the change ended.
    pub result: Outcome,
    /// Why it did not converge.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// How long it took to converge or to fail, in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
    /// The options it changed.
    pub changes: Vec<Changed>,
}

/// A stored revision: its package, the modules its apply fixed and its record.
#[derive(Debug, Clone)]
pub struct Stored {
    pub package: Package,
    pub lock: Option<ModuleSet>,
    pub record: Record,
}

/// A deployment's persistent state directory on this node.
#[derive(Debug)]
pub struct Revisions {
    root: PathBuf,
}

/// The persistent state directory of `deployment` on this node: `uniserve/deployments/<name>`
/// in `$XDG_STATE_HOME`, else in `~/.local/state`.
///
/// # Errors
///
/// Fails when neither variable names a directory.
pub fn directory(deployment: &str) -> anyhow::Result<PathBuf> {
    let base = match std::env::var_os("XDG_STATE_HOME").filter(|value| !value.is_empty()) {
        Some(state) => PathBuf::from(state),
        None => std::env::var_os("HOME")
            .filter(|value| !value.is_empty())
            .map(|home| PathBuf::from(home).join(".local/state"))
            .context("neither XDG_STATE_HOME nor HOME names the user's state directory")?,
    };
    Ok(base.join("uniserve/deployments").join(deployment))
}

impl Revisions {
    /// Opens the state directory `root`, creating it when missing.
    ///
    /// # Errors
    ///
    /// Fails when the directory cannot be created.
    pub fn open(root: PathBuf) -> anyhow::Result<Self> {
        std::fs::create_dir_all(root.join(REVISIONS))
            .with_context(|| format!("creating the state directory {}", root.display()))?;
        std::fs::create_dir_all(root.join(PROGRAM_STATE))
            .with_context(|| format!("creating the state directory {}", root.display()))?;
        Ok(Self { root })
    }

    /// Records every revision that never got a result as abandoned: it was converging when its
    /// leader ended, and a leader that starts again converges no revision yet. A hot restart's
    /// successor, which goes on with the change its predecessor began, abandons nothing.
    ///
    /// # Errors
    ///
    /// Fails when the directory cannot be read or a record cannot be rewritten.
    pub fn abandon(&self) -> anyhow::Result<()> {
        for number in self.numbers()? {
            let mut record = self.record(number)?;
            if record.result == Outcome::Converging {
                record.result = Outcome::Abandoned;
                record.reason = Some("the leader ended while the revision converged".to_owned());
                self.write_record(&record)?;
            }
        }
        Ok(())
    }

    /// The state directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The directory of the programs' saved states.
    pub fn program_state(&self) -> PathBuf {
        self.root.join(PROGRAM_STATE)
    }

    /// The directory of revision `number`.
    fn path(&self, number: u64) -> PathBuf {
        self.root.join(REVISIONS).join(number.to_string())
    }

    /// The numbers of every stored revision, ascending.
    fn numbers(&self) -> anyhow::Result<Vec<u64>> {
        let directory = self.root.join(REVISIONS);
        let mut numbers = Vec::new();
        for entry in std::fs::read_dir(&directory)
            .with_context(|| format!("reading {}", directory.display()))?
        {
            let entry = entry.with_context(|| format!("reading {}", directory.display()))?;
            // A directory a crash left before its rename has a suffix and is no revision.
            if let Some(number) = entry
                .file_name()
                .to_str()
                .and_then(|name| name.parse().ok())
            {
                numbers.push(number);
            }
        }
        numbers.sort_unstable();
        Ok(numbers)
    }

    /// The revision `current` names, if any.
    ///
    /// # Errors
    ///
    /// Fails when the link names no revision.
    pub fn current(&self) -> anyhow::Result<Option<u64>> {
        let link = self.root.join(CURRENT);
        match std::fs::read_link(&link) {
            Ok(target) => {
                let number = target
                    .file_name()
                    .and_then(|name| name.to_str())
                    .and_then(|name| name.parse().ok())
                    .with_context(|| format!("{} names no revision", link.display()))?;
                Ok(Some(number))
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error).with_context(|| format!("reading {}", link.display())),
        }
    }

    /// Revision `number`'s record.
    ///
    /// # Errors
    ///
    /// Fails when the record cannot be read.
    pub fn record(&self, number: u64) -> anyhow::Result<Record> {
        let path = self.path(number).join(RECORD);
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("reading {}", path.display()))
    }

    /// Revision `number`: its package, lock and record.
    ///
    /// # Errors
    ///
    /// Fails when the revision is missing or does not read.
    pub fn load(&self, number: u64) -> anyhow::Result<Stored> {
        let path = self.path(number);
        let package = Package::read(&path)?;
        let lock = match std::fs::read_to_string(path.join(LOCK)) {
            Ok(text) => Some(
                toml::from_str(&text)
                    .with_context(|| format!("reading {}", path.join(LOCK).display()))?,
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("reading {}", path.join(LOCK).display()));
            }
        };
        Ok(Stored {
            package,
            lock,
            record: self.record(number)?,
        })
    }

    /// Writes the next revision, `package` as `writer` wrote it, with the `changes` it makes of
    /// `previous` (the current revision's package, `None` for a deployment that starts from
    /// nothing), synced, and returns its number. Its change is converging until
    /// [`Self::converged`] or [`Self::ended`] decides it.
    ///
    /// # Errors
    ///
    /// Fails when the revision cannot be written.
    pub fn begin(
        &self,
        package: &Package,
        writer: &str,
        previous: Option<&Package>,
        changes: &[Change],
    ) -> anyhow::Result<u64> {
        let number = self.numbers()?.last().map_or(1, |last| last + 1);
        let previous_table = previous
            .map(|package| package.text())
            .transpose()?
            .map(|text| text.parse::<toml::Table>())
            .transpose()
            .context("reading the previous revision's configuration")?;
        let next_table = package
            .text()?
            .parse::<toml::Table>()
            .context("reading the revision's configuration")?;
        let record = Record {
            revision: number,
            time: humantime::format_rfc3339_millis(SystemTime::now()).to_string(),
            writer: writer.to_owned(),
            effect: Effects::of(changes).heaviest(),
            result: Outcome::Converging,
            reason: None,
            seconds: None,
            changes: changes
                .iter()
                .map(|change| Changed {
                    path: change.path.clone(),
                    effect: change.effect,
                    previous: previous_table
                        .as_ref()
                        .and_then(|table| value_at(table, &change.path)),
                    next: value_at(&next_table, &change.path),
                })
                .collect(),
        };

        // The revision is written under a name that is no number and renamed into place, so a
        // crash never leaves a partial revision.
        let staging = self.root.join(REVISIONS).join(format!("{number}.new"));
        if staging.exists() {
            std::fs::remove_dir_all(&staging)
                .with_context(|| format!("removing {}", staging.display()))?;
        }
        package.write(&staging)?;
        write_synced(
            &staging.join(RECORD),
            toml::to_string_pretty(&record)
                .context("printing the revision's record")?
                .as_bytes(),
        )?;
        let path = self.path(number);
        std::fs::rename(&staging, &path)
            .with_context(|| format!("renaming {} into place", staging.display()))?;
        sync_directory(&self.root.join(REVISIONS))?;
        Ok(number)
    }

    /// Records the module revisions revision `number`'s apply fixed, which every restart of it
    /// runs rather than resolving `latest` again.
    ///
    /// # Errors
    ///
    /// Fails when the lock cannot be written.
    pub fn lock(&self, number: u64, modules: &ModuleSet) -> anyhow::Result<()> {
        let text = toml::to_string_pretty(modules).context("printing the lock")?;
        write_synced(&self.path(number).join(LOCK), text.as_bytes())
    }

    /// Records that the deployment converged to revision `number` after `seconds`, and makes it
    /// current: a new link is renamed over `current`, so `current` names one revision or the
    /// other at every moment. Prunes the revisions beyond the newest `kept`.
    ///
    /// # Errors
    ///
    /// Fails when the record or the link cannot be written.
    pub fn converged(&self, number: u64, seconds: f64, kept: usize) -> anyhow::Result<()> {
        let mut record = self.record(number)?;
        record.result = Outcome::Converged;
        record.seconds = Some(seconds);
        self.write_record(&record)?;

        let staging = self.root.join(format!("{CURRENT}.new"));
        if std::fs::symlink_metadata(&staging).is_ok() {
            std::fs::remove_file(&staging)
                .with_context(|| format!("removing {}", staging.display()))?;
        }
        std::os::unix::fs::symlink(Path::new(REVISIONS).join(number.to_string()), &staging)
            .with_context(|| format!("linking {}", staging.display()))?;
        std::fs::rename(&staging, self.root.join(CURRENT))
            .with_context(|| format!("replacing {}", self.root.join(CURRENT).display()))?;
        sync_directory(&self.root)?;
        self.prune(kept)
    }

    /// Records that revision `number`'s change ended as `outcome` after `seconds`, for
    /// `reason`; `current` stays.
    ///
    /// # Errors
    ///
    /// Fails when the record cannot be written.
    pub fn ended(
        &self,
        number: u64,
        outcome: Outcome,
        reason: String,
        seconds: f64,
    ) -> anyhow::Result<()> {
        let mut record = self.record(number)?;
        record.result = outcome;
        record.reason = Some(reason);
        record.seconds = Some(seconds);
        self.write_record(&record)
    }

    /// Removes the revisions older than the newest `kept`, beside `current` and every revision
    /// that converges.
    fn prune(&self, kept: usize) -> anyhow::Result<()> {
        let current = self.current()?;
        let numbers = self.numbers()?;
        let old = numbers.len().saturating_sub(kept);
        for number in numbers.into_iter().take(old) {
            if Some(number) == current || self.record(number)?.result == Outcome::Converging {
                continue;
            }
            let path = self.path(number);
            std::fs::remove_dir_all(&path)
                .with_context(|| format!("removing {}", path.display()))?;
        }
        Ok(())
    }

    /// Rewrites a revision's record in place, synced.
    fn write_record(&self, record: &Record) -> anyhow::Result<()> {
        ensure!(
            self.path(record.revision).is_dir(),
            "revision {} is not stored",
            record.revision
        );
        let text = toml::to_string_pretty(record).context("printing the revision's record")?;
        write_synced(&self.path(record.revision).join(RECORD), text.as_bytes())
    }
}

/// The value at the option path `path` of the configuration table `table`.
fn value_at(table: &toml::Table, path: &str) -> Option<toml::Value> {
    let mut parts = path.split('.');
    let mut value = table.get(parts.next()?)?;
    for part in parts {
        value = value.as_table()?.get(part)?;
    }
    Some(value.clone())
}

/// Writes `contents` to `path` through a file beside it renamed over it, the file synced before
/// the rename, creating the directories on the way.
///
/// # Errors
///
/// Fails when a directory or the file cannot be created, written or synced.
pub(crate) fn write_synced(path: &Path, contents: &[u8]) -> anyhow::Result<()> {
    use std::io::Write as _;

    let parent = path
        .parent()
        .with_context(|| format!("{} has no directory", path.display()))?;
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let staging = path.with_extension("new");
    let mut file = std::fs::File::create(&staging)
        .with_context(|| format!("creating {}", staging.display()))?;
    file.write_all(contents)
        .and_then(|()| file.sync_all())
        .with_context(|| format!("writing {}", staging.display()))?;
    std::fs::rename(&staging, path).with_context(|| format!("writing {}", path.display()))?;
    sync_directory(parent)
}

/// Syncs the directory `path`, so the names a rename put there survive a crash.
fn sync_directory(path: &Path) -> anyhow::Result<()> {
    std::fs::File::open(path)
        .and_then(|directory| directory.sync_all())
        .with_context(|| format!("syncing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn package(text: &str) -> Package {
        Package::of(crate::config::parse(text).unwrap().0, BTreeMap::new()).unwrap()
    }

    /// A revision becomes current only once it converged: before, `current` names the previous
    /// one; a change that failed leaves it; and each record states what changed and how.
    #[test]
    fn current_names_the_last_converged_revision() {
        let state = tempfile::tempdir().unwrap();
        let revisions = Revisions::open(state.path().join("qwen3")).unwrap();
        assert_eq!(revisions.current().unwrap(), None);

        let first = package("model = \"m\"\n");
        let start = proto::reload::changes(None, &first.config);
        let one = revisions.begin(&first, "start", None, &start).unwrap();
        assert_eq!(one, 1);
        assert_eq!(revisions.current().unwrap(), None);
        revisions.converged(one, 0.5, KEPT).unwrap();
        assert_eq!(revisions.current().unwrap(), Some(1));

        let second = package("model = \"m\"\n[scheduler_config]\nchunked_prefill_size = 4096\n");
        let changes = proto::reload::changes(Some(&first.config), &second.config);
        let two = revisions
            .begin(&second, "chunk", Some(&first), &changes)
            .unwrap();
        let record = revisions.record(two).unwrap();
        assert_eq!(record.writer, "chunk");
        assert_eq!(record.effect, Some(Effect::Step));
        assert_eq!(
            record.changes,
            [Changed {
                path: "scheduler_config.chunked_prefill_size".to_owned(),
                effect: Effect::Step,
                previous: None,
                next: Some(toml::Value::Integer(4096)),
            }]
        );
        revisions
            .ended(two, Outcome::RolledBack, "refused".to_owned(), 0.1)
            .unwrap();
        assert_eq!(revisions.current().unwrap(), Some(1));
        let stored = revisions.load(1).unwrap();
        assert_eq!(stored.package, first);
        assert_eq!(stored.record.result, Outcome::Converged);
    }

    /// A revision that was converging when its leader ended is recorded abandoned when a leader
    /// starts again, and `current` still names the one before it.
    #[test]
    fn a_crash_while_converging_abandons_the_revision() {
        let state = tempfile::tempdir().unwrap();
        let root = state.path().join("qwen3");
        let first = package("model = \"m\"\n");
        {
            let revisions = Revisions::open(root.clone()).unwrap();
            let one = revisions.begin(&first, "start", None, &[]).unwrap();
            revisions.converged(one, 0.1, KEPT).unwrap();
            revisions
                .begin(&package("model = \"n\"\n"), "operator", Some(&first), &[])
                .unwrap();
        }
        let revisions = Revisions::open(root).unwrap();
        assert_eq!(revisions.record(2).unwrap().result, Outcome::Converging);
        revisions.abandon().unwrap();
        assert_eq!(revisions.current().unwrap(), Some(1));
        assert_eq!(revisions.record(2).unwrap().result, Outcome::Abandoned);
    }

    /// The lock a revision's apply fixed reads back with it.
    #[test]
    fn a_revision_keeps_its_lock() {
        let state = tempfile::tempdir().unwrap();
        let revisions = Revisions::open(state.path().to_owned()).unwrap();
        let one = revisions
            .begin(&package("model = \"m\"\n"), "start", None, &[])
            .unwrap();
        let modules = ModuleSet {
            engine: proto::module::PackageRevision {
                backend: "uniserve".to_owned(),
                revision: 3,
                arguments: toml::Table::new(),
            },
            policy: proto::module::PackageRevision {
                backend: "arrival".to_owned(),
                revision: 1,
                arguments: toml::Table::new(),
            },
            kernels: BTreeMap::new(),
            programs: Default::default(),
            processing: Some(proto::module::PackageRevision {
                backend: "processing".to_owned(),
                revision: 2,
                arguments: toml::Table::new(),
            }),
        };
        revisions.lock(one, &modules).unwrap();
        assert_eq!(revisions.load(one).unwrap().lock, Some(modules));
    }

    /// Pruning keeps the newest revisions and the current one, however old.
    #[test]
    fn pruning_keeps_the_current_revision() {
        let state = tempfile::tempdir().unwrap();
        let revisions = Revisions::open(state.path().to_owned()).unwrap();
        let first = package("model = \"m\"\n");
        let one = revisions.begin(&first, "start", None, &[]).unwrap();
        revisions.converged(one, 0.1, KEPT).unwrap();
        for _ in 0..4 {
            let number = revisions.begin(&first, "operator", None, &[]).unwrap();
            revisions
                .ended(number, Outcome::RolledBack, "no".to_owned(), 0.0)
                .unwrap();
        }
        let last = revisions.begin(&first, "operator", None, &[]).unwrap();
        revisions.converged(last, 0.1, 2).unwrap();
        assert_eq!(revisions.numbers().unwrap(), [5, 6]);
        assert_eq!(revisions.current().unwrap(), Some(6));
    }
}
