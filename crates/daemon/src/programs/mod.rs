//! The policy package's programs on the leader daemon (`specs/configuration.md` §4): the
//! control programs it steps every `period`, the routing program it steps for every request, and
//! the KV program it hands every node's Storage Server.
//!
//! Every program runs in an instance of its own of the program host (`program`), with fuel per
//! call and no WASI. A program's module comes from the deployment's Build Server, fixed at apply
//! with the rest of the module set ([`fix`]); what the module states decides its kind and so where
//! it runs. The engines run the scheduling program themselves, as their scheduling policy.
//!
//! - A control program reads `State` (the instant, the current configuration and the views of the
//!   observation store) and outputs a new configuration or none. The configuration it reads is the
//!   current revision's, whole and lossless: the JSON of every option. An output goes through the
//!   node's one apply as a revision written by the program, merged option by option with what
//!   other writers changed since its input ([`merge`]); a refused output, like any failure, is a
//!   `log` event of the program and leaves the deployment as it is.
//! - The routing program reads each request and the replicas, and chooses the replica that
//!   computes its prompt and the one that decodes it ([`Router`]).
//! - A program that panics, runs past its fuel or breaks the call convention is ended: the host
//!   calls it no more, what it wrote stays, and its failure is a `log` event. A new instance
//!   starts only when a change gives it another entry or module.
//! - A program's own fields are its state, saved in `program-state/` after every periodic step
//!   ([`States`]) and restored, when it still decodes as the program's type, whenever its instance
//!   is created again with the same arguments: after a restart of the leader and after a change
//!   of the program's module. A program given other arguments is created from them.
//!
//! A change of the package keeps the instance of every program whose source, arguments and
//! module are unchanged, and replaces the others between two of their calls
//! ([`Programs::replace`]).

mod control;
pub(crate) mod fix;
mod host;
pub(crate) mod kv;
mod routing;
mod state;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use program::{Budget, Kind};
use tokio::sync::{mpsc, oneshot, watch};
use tracing::info;

pub(crate) use self::host::{Measure, validate};
pub(crate) use self::kv::KvSetup;
pub(crate) use self::routing::Router;
pub(crate) use self::state::States;
use crate::config::Config;
use crate::observation::Store;
use crate::policy::Package;

/// What a control program's call runs within: a second's worth of instructions, ample for a
/// decision over a few series, with the fuel of decoding the configuration on top of it; and the
/// memory of a program that keeps moving averages over its own history.
const CONTROL: Budget = Budget {
    fuel: 1_000_000_000,
    per_byte: 200,
    memory: 256 << 20,
};

/// What a routing program's call runs within: it is on every request's path, so its budget is a
/// small fraction of a request's admission, beside decoding the replicas' states.
const ROUTING: Budget = Budget {
    fuel: 20_000_000,
    per_byte: 200,
    memory: 64 << 20,
};

/// How often a routing program's state is saved when its entry states no `period`.
const SAVE: std::time::Duration = std::time::Duration::from_secs(60);

/// The deployment as the programs read it: the revision that serves and its package.
#[derive(Debug)]
pub(crate) struct Current {
    pub(crate) revision: u64,
    pub(crate) package: Package,
}

/// One program as its apply fixed it: its entry in the configuration, the kind its module
/// states, and the module.
#[derive(Clone, Debug)]
pub(crate) struct Loaded {
    pub(crate) name: String,
    pub(crate) entry: crate::config::program::Program,
    pub(crate) kind: Kind,
    /// The SHA-256 of the module, in hex.
    pub(crate) digest: String,
    pub(crate) bytes: Arc<[u8]>,
}

/// What decides whether a program keeps its instance across a change: its source, its
/// arguments and its module. Its period is read anew at every call and does not.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Identity {
    source: String,
    args: toml::Table,
    digest: String,
}

impl Loaded {
    fn identity(&self) -> Identity {
        Identity {
            source: self.entry.source.to_string(),
            args: self.entry.args.clone(),
            digest: self.digest.clone(),
        }
    }
}

/// A control program's output on its way to the node's apply: the configuration it read and the
/// one it gave, and where the node answers with the revision it converged to or why it did not.
pub(crate) struct Output {
    /// The program's name, which the revision records as its writer.
    pub(crate) writer: String,
    pub(crate) input: Config,
    pub(crate) output: Config,
    pub(crate) done: oneshot::Sender<Result<u64, String>>,
}

/// The programs the leader runs.
pub(crate) struct Programs {
    store: Store,
    states: States,
    current: watch::Sender<Arc<Current>>,
    outputs: mpsc::UnboundedSender<Output>,
    control: Mutex<BTreeMap<String, control::Task>>,
    router: Arc<Router>,
    /// What a validation measures of a change that starts a kernel generation.
    measure: Measure,
    /// Cancelled once every program ended ([`Self::stop`]).
    ended: tokio_util::sync::CancellationToken,
}

impl Programs {
    /// The programs of the deployment `current` serves, recording into `store`, saving their
    /// states in `states` and handing their outputs to `outputs`; none runs until
    /// [`Self::replace`] starts them. `router` is the slot the leader routes requests through,
    /// and `measure` what a control program's validation reports of a kernel generation.
    pub(crate) fn new(
        store: Store,
        states: States,
        current: Current,
        outputs: mpsc::UnboundedSender<Output>,
        router: Arc<Router>,
        measure: Measure,
    ) -> Self {
        Self {
            store,
            states,
            current: watch::Sender::new(Arc::new(current)),
            outputs,
            control: Mutex::default(),
            router,
            measure,
            ended: tokio_util::sync::CancellationToken::new(),
        }
    }

    /// Cancelled once the programs ended with their deployment.
    pub(crate) fn ended(&self) -> tokio_util::sync::CancellationToken {
        self.ended.clone()
    }

    /// The KV program among `loaded`, as the leader hands it to every node's Storage Server;
    /// `None` when the package runs none.
    pub(crate) fn kv(loaded: &[Loaded]) -> Option<KvSetup> {
        loaded
            .iter()
            .find(|program| program.kind == Kind::Kv)
            .map(|program| KvSetup {
                name: program.name.clone(),
                module: Arc::clone(&program.bytes),
                args: program.entry.args.clone(),
                // A KV program states its period ([`fix`]).
                period: program.entry.period.unwrap_or(SAVE),
            })
    }

    /// Records what the Storage Server of `host` recorded of the KV program as a `log` event of
    /// the program, at the revision that serves.
    pub(crate) fn record(&self, host: &str, record: storage::protocol::ProgramRecord) {
        use storage::protocol::ProgramEvent;

        let logs = host::Logs::new(
            record.program.clone(),
            self.store.clone(),
            self.current.subscribe(),
        );
        let mut fields = crate::observation::Fields::new();
        fields.insert("host".to_owned(), host.into());
        match record.event {
            ProgramEvent::Log { level, text } => logs.record(level, &text, fields),
            ProgramEvent::Ended { reason } => {
                fields.insert("outcome".to_owned(), "ended".into());
                logs.record(program::Level::Warn, &reason, fields);
            }
            ProgramEvent::Refused { hook, reason } => {
                fields.insert("outcome".to_owned(), "refused".into());
                fields.insert("hook".to_owned(), hook.into());
                logs.record(program::Level::Warn, &reason, fields);
            }
            ProgramEvent::Skipped { reason } => {
                fields.insert("outcome".to_owned(), "skipped".into());
                logs.record(program::Level::Info, &reason, fields);
            }
        }
    }

    /// Makes `current` the deployment the programs read from their next call on.
    pub(crate) fn set_current(&self, current: Current) {
        self.current.send_replace(Arc::new(current));
    }

    /// The revision the programs read.
    pub(crate) fn current(&self) -> Arc<Current> {
        Arc::clone(&self.current.borrow())
    }

    /// Runs `loaded`, the package's programs: each control program and the routing program keeps
    /// its instance when its source, arguments and module are unchanged, takes its new period,
    /// and is otherwise replaced by a new instance, which restores the state the old one saved
    /// when it decodes. A program no longer listed ends, its state saved. The other kinds run
    /// elsewhere: the scheduling program in the engines, the KV program in the Storage Servers.
    pub(crate) async fn replace(&self, loaded: &[Loaded]) {
        let wanted: BTreeMap<&str, &Loaded> = loaded
            .iter()
            .filter(|program| program.kind == Kind::Control)
            .map(|program| (program.name.as_str(), program))
            .collect();

        // The instances that end save their states first, so a replacement restores from them.
        let ended: Vec<control::Task> = {
            let mut control = service::lock(&self.control);
            let replaced: Vec<String> = control
                .iter()
                .filter(|(name, task)| {
                    wanted
                        .get(name.as_str())
                        .is_none_or(|program| task.identity != program.identity())
                })
                .map(|(name, _)| name.clone())
                .collect();
            replaced
                .into_iter()
                .filter_map(|name| control.remove(&name))
                .collect()
        };
        for task in ended {
            task.stop().await;
        }

        for (name, program) in wanted {
            let period = program.entry.period.unwrap_or(SAVE);
            if let Some(task) = service::lock(&self.control).get(name) {
                task.set_period(period);
                continue;
            }
            info!(program = name, source = %program.entry.source, "a control program starts");
            let task = control::Task::start(control::Start {
                program: program.clone(),
                period,
                store: self.store.clone(),
                states: self.states.clone(),
                current: self.current.subscribe(),
                outputs: self.outputs.clone(),
                measure: Arc::clone(&self.measure),
            })
            .await;
            service::lock(&self.control).insert(name.to_owned(), task);
        }

        let routing = loaded.iter().find(|program| program.kind == Kind::Routing);
        self.router.replace(
            routing.map(|program| routing::Start {
                program: program.clone(),
                period: program.entry.period.unwrap_or(SAVE),
                store: self.store.clone(),
                states: self.states.clone(),
                current: self.current.subscribe(),
            }),
            &self.states,
        );
    }

    /// Ends every program, each with its state saved: the deployment comes down.
    pub(crate) async fn stop(&self) {
        let tasks: Vec<control::Task> = std::mem::take(&mut *service::lock(&self.control))
            .into_values()
            .collect();
        for task in tasks {
            task.stop().await;
        }
        self.router.replace(None, &self.states);
        self.ended.cancel();
    }
}

/// The configuration `writer`'s output makes of `current`: every option the output changed of
/// the configuration the program read (`input`) takes the output's value, and every other option
/// keeps `current`'s, so the changes other writers made since the program read survive its
/// write. An option the output returns to its default is removed from `current`'s statement.
///
/// # Errors
///
/// Fails when a configuration has no TOML form, or the merged one does not read.
pub(crate) fn merge(input: &Config, output: &Config, current: &Config) -> anyhow::Result<Config> {
    let changed = proto::reload::changes(Some(input), output);
    let written: toml::Table = crate::config::printed(output)?.parse()?;
    let mut merged: toml::Table = crate::config::printed(current)?.parse()?;
    for change in changed {
        let value = value_at(&written, &change.path);
        set_at(&mut merged, &change.path, value);
    }
    let (config, _) = crate::config::parse(&toml::to_string(&merged)?)?;
    Ok(config)
}

/// The value at the option path `path` of `table`.
fn value_at(table: &toml::Table, path: &str) -> Option<toml::Value> {
    let mut parts = path.split('.');
    let mut value = table.get(parts.next()?)?;
    for part in parts {
        value = value.as_table()?.get(part)?;
    }
    Some(value.clone())
}

/// Sets the option at `path` of `table` to `value`, creating the tables on the way, or removes
/// it for `None` (the option at its default), with the tables it leaves empty.
fn set_at(table: &mut toml::Table, path: &str, value: Option<toml::Value>) {
    match path.split_once('.') {
        None => {
            match value {
                Some(value) => table.insert(path.to_owned(), value),
                None => table.remove(path),
            };
        }
        Some((head, rest)) => {
            if value.is_none() && !table.contains_key(head) {
                return;
            }
            let inner = table
                .entry(head.to_owned())
                .or_insert_with(|| toml::Value::Table(toml::Table::new()));
            if !inner.is_table() {
                *inner = toml::Value::Table(toml::Table::new());
            }
            if let toml::Value::Table(inner) = inner {
                set_at(inner, rest, value);
                if inner.is_empty() {
                    table.remove(head);
                }
            }
        }
    }
}
