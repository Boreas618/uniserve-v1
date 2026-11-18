//! Test support of the program crates: what a program's behavioural tests share, stated once.
//!
//! - [`module!`] builds the calling crate's package (or any workspace package) for
//!   `wasm32-unknown-unknown` with the cargo that runs the test and returns its module, so a
//!   test runs the module a Build Server would build, in the program host the leader daemon runs.
//! - [`Deployment`] is a control program's embedder: it answers `uniserve.stats` and
//!   `uniserve.group` from the [`Stat`]s a test states, `uniserve.events` from its records, and
//!   `uniserve.validate` with the report the test states, and keeps what the program logged and
//!   asked to validate ([`Seen`]). A routing program's module links with it as well, since it is
//!   also the `uniserve.log` embedder.
//!
//! Every helper fails the calling test with a message naming what failed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, OnceLock};

use program::{Config, Control, Filter, Function, Group, Host, Level, Query, Record, Report};
use scheduling::time::Duration;

/// The wasm32 module of the workspace package `$package`, built once per test process into the
/// test's own target directory (`CARGO_TARGET_TMPDIR`), so the build does not wait for the lock
/// of the build that runs the tests.
#[macro_export]
macro_rules! module {
    ($package:expr) => {
        $crate::build(
            ::std::path::Path::new(env!("CARGO")),
            ::std::path::Path::new(env!("CARGO_TARGET_TMPDIR")),
            $package,
        )
    };
}

/// Builds the workspace package `package` with `cargo` for `wasm32-unknown-unknown` into
/// `target/programs` and returns its module; [`module!`] calls it with the test's own cargo and
/// target directory. Each package is built once per process.
///
/// # Panics
///
/// When the build fails or writes no module of the package.
pub fn build(cargo: &Path, target: &Path, package: &str) -> Vec<u8> {
    static MODULES: OnceLock<Mutex<BTreeMap<String, Vec<u8>>>> = OnceLock::new();

    // One build at a time: concurrent tests of one process wait for the module the first
    // builds rather than starting a second cargo on the same target directory.
    let mut modules = MODULES
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(module) = modules.get(package) {
        return module.clone();
    }

    let build = Command::new(cargo)
        .args(["build", "--release", "--locked", "--offline", "--lib"])
        .args(["--message-format=json-render-diagnostics"])
        .args(["--target", "wasm32-unknown-unknown", "--package", package])
        .arg("--target-dir")
        .arg(target.join("programs"))
        .output()
        .unwrap_or_else(|error| panic!("running {}: {error}", cargo.display()));
    assert!(
        build.status.success(),
        "building {package} for wasm32: {}",
        String::from_utf8_lossy(&build.stderr)
    );

    // The library's artifact is named by the crate's library name, the package's name with
    // underscores.
    let library = package.replace('-', "_");
    let file: PathBuf = build
        .stdout
        .split(|byte| *byte == b'\n')
        .filter_map(|line| serde_json::from_slice::<serde_json::Value>(line).ok())
        .filter(|message| {
            message["reason"] == "compiler-artifact" && message["target"]["name"] == library
        })
        .flat_map(|message| message["filenames"].as_array().cloned().unwrap_or_default())
        .filter_map(|file| file.as_str().map(PathBuf::from))
        .find(|file| {
            file.extension()
                .is_some_and(|extension| extension == "wasm")
        })
        .unwrap_or_else(|| panic!("the build of {package} wrote no wasm32 module"));

    let module =
        std::fs::read(&file).unwrap_or_else(|error| panic!("reading {}: {error}", file.display()));
    modules.insert(package.to_owned(), module.clone());
    module
}

/// One aggregate of the `stats` table as a test states it: the value `uniserve.stats` answers
/// for a query of this name and exactly these labels, and `uniserve.group` for this group of a
/// query split by one of the labels. A window or a function of `None` matches any.
#[derive(Clone, Debug, PartialEq)]
pub struct Stat {
    pub name: String,
    pub labels: BTreeMap<String, String>,
    pub window: Option<Duration>,
    pub function: Option<Function>,
    pub revision: Option<u64>,
    pub value: f64,
}

impl Stat {
    /// The value `value` of series `name` with `labels`, for any window and aggregate.
    pub fn new(name: &str, labels: &[(&str, &str)], value: f64) -> Stat {
        Stat {
            name: name.to_owned(),
            labels: labels
                .iter()
                .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
                .collect(),
            window: None,
            function: None,
            revision: None,
            value,
        }
    }

    /// Only for queries over the window `window`.
    #[must_use]
    pub fn window(mut self, window: Duration) -> Stat {
        self.window = Some(window);
        self
    }

    /// Only for the aggregate `function`.
    #[must_use]
    pub fn function(mut self, function: Function) -> Stat {
        self.function = Some(function);
        self
    }

    /// Only for queries of the values measured under revision `revision`.
    #[must_use]
    pub fn revision(mut self, revision: u64) -> Stat {
        self.revision = Some(revision);
        self
    }

    /// Whether the stat answers a query of `query`'s name, window, function and revision.
    fn aggregates(&self, query: &Query) -> bool {
        self.name == query.name
            && self.window.is_none_or(|window| window == query.window)
            && self
                .function
                .is_none_or(|function| function == query.function)
            && self.revision == query.revision
    }
}

/// What a program asked of its [`Deployment`].
#[derive(Debug, Default)]
pub struct Seen {
    /// The records it logged, in order.
    pub logs: Vec<(Level, String)>,
    /// The configurations it validated, in order.
    pub validated: Vec<Config>,
}

/// A control program's embedder: the deployment's `stats` and `events` tables as a test states
/// them, and the report validation gives.
///
/// Cloning shares [`Seen`], so a test keeps a clone to read what the program asked after the
/// embedder moved into the instance.
#[derive(Clone, Debug)]
pub struct Deployment {
    /// The aggregates the `stats` table answers, the first matching one winning.
    pub stats: Vec<Stat>,
    /// The `events` table's records, oldest first. A filter's window is not applied: a test
    /// states the records its step's window holds.
    pub records: Vec<Record>,
    /// What validating any configuration reports.
    pub report: Result<Report, String>,
    pub seen: Arc<Mutex<Seen>>,
}

impl Default for Deployment {
    /// A deployment with no observation that takes every configuration at the next step.
    fn default() -> Self {
        Deployment {
            stats: Vec::new(),
            records: Vec::new(),
            report: Ok(Report {
                effect: Some("step".into()),
                interrupts: false,
                rooms: Vec::new(),
            }),
            seen: Arc::default(),
        }
    }
}

impl Deployment {
    /// The deployment with `stat` among its aggregates.
    #[must_use]
    pub fn stat(mut self, stat: Stat) -> Deployment {
        self.stats.push(stat);
        self
    }

    /// The deployment with `record` as its latest event.
    #[must_use]
    pub fn record(mut self, record: Record) -> Deployment {
        self.records.push(record);
        self
    }

    /// The deployment whose validation reports `report`.
    #[must_use]
    pub fn report(mut self, report: Result<Report, String>) -> Deployment {
        self.report = report;
        self
    }

    /// What the program logged and validated so far.
    pub fn seen(&self) -> std::sync::MutexGuard<'_, Seen> {
        self.seen
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl Host for Deployment {
    fn log(&mut self, level: Level, text: &str) {
        self.seen().logs.push((level, text.to_owned()));
    }
}

impl Control for Deployment {
    fn stats(&mut self, query: &Query) -> f64 {
        self.stats
            .iter()
            .find(|stat| stat.aggregates(query) && stat.labels == query.labels)
            .map_or(f64::NAN, |stat| stat.value)
    }

    /// The value of each stat of the query whose labels are the query's and one value of the
    /// key: the stated series of each group.
    fn group(&mut self, group: &Group) -> BTreeMap<String, f64> {
        let mut values = BTreeMap::new();
        for stat in &self.stats {
            if !stat.aggregates(&group.query) {
                continue;
            }
            let mut labels = stat.labels.clone();
            let Some(value) = labels.remove(&group.key) else {
                continue;
            };
            if labels == group.query.labels {
                values.entry(value).or_insert(stat.value);
            }
        }
        values
    }

    fn events(&mut self, filter: &Filter) -> Vec<Record> {
        let mut selected: Vec<Record> = self
            .records
            .iter()
            .filter(|record| record.kind == filter.kind)
            .filter(|record| {
                filter
                    .labels
                    .iter()
                    .all(|(key, value)| record.labels.get(key) == Some(value))
                    && filter
                        .fields
                        .iter()
                        .all(|(key, value)| record.fields.get(key) == Some(value))
            })
            .cloned()
            .collect();
        if let Some(limit) = filter.limit {
            let skip = selected.len().saturating_sub(limit as usize);
            selected.drain(..skip);
        }
        selected
    }

    fn validate(&mut self, config: &Config) -> Result<Report, String> {
        self.seen().validated.push(config.clone());
        self.report.clone()
    }
}

/// The `revision` event of a revision `writer` wrote that ended with `result` (`converged`,
/// `rolled_back`) at `time`.
pub fn revision(writer: &str, result: &str, time: scheduling::time::Instant) -> Record {
    Record {
        time,
        kind: "revision".into(),
        labels: BTreeMap::from([("writer".to_owned(), writer.to_owned())]),
        fields: BTreeMap::from([
            ("writer".to_owned(), writer.into()),
            ("result".to_owned(), result.into()),
        ]),
        revision: 1,
    }
}
