//! How a change of the configuration takes effect: each option's declared effect, the
//! comparison of two configurations derived from those declarations, and how the effects of one
//! change merge.
//!
//! Every structure of the configuration derives [`Reload`] (`proto_derive::Reload`), and every
//! field states its effect with `#[reload(...)]`, so the comparison names every option and a
//! field added without a declaration does not compile. What the declaration cannot check, that
//! the stated effect is the right one, is the behavioral test's: a deployment started with one
//! configuration and moved to another behaves as one started with the other.
//!
//! A deployment that starts from nothing compares against no configuration at all
//! ([`changes`] with `None`), so every option counts as changed and the start is the same
//! comparison as every later change.

use std::collections::BTreeSet;

pub use proto_derive::Reload;
use serde::{Deserialize, Serialize};

/// How a change of an option takes effect, from the lightest to the heaviest
/// (`specs/configuration.md` §7).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    /// Each scheduler reads the new value at its next step, from the configuration snapshot it
    /// takes when the step begins.
    Step,
    /// The experts' order: the new order is prepared on the copy stream beside the batches and
    /// committed at one computation boundary; the epoch, the graphs' addresses and the requests
    /// stay.
    Order,
    /// A program is replaced between two of its calls; no graph is involved.
    Program,
    /// A kernel generation: the engines capture the new generation's graphs in the background
    /// and switch bucket by bucket, keeping the old graphs until the commit.
    Generation,
    /// The deployment's Storage Server on each node is replaced by a new process, and the
    /// engines follow as an epoch.
    Storage,
    /// A new engine starts beside the serving one, which drains.
    Epoch,
    /// The daemon is replaced by a successor process, which applies the rest of the change.
    Daemon,
}

impl Effect {
    /// The option's effect as a configuration file and a report name it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Step => "step",
            Self::Order => "order",
            Self::Program => "program",
            Self::Generation => "generation",
            Self::Storage => "storage",
            Self::Epoch => "epoch",
            Self::Daemon => "daemon",
        }
    }

    /// Whether this effect starts processes in place of running ones: what a change must
    /// prepare before it serves.
    pub fn starts_processes(self) -> bool {
        matches!(self, Self::Storage | Self::Epoch | Self::Daemon)
    }
}

impl std::fmt::Display for Effect {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

/// One changed option: its path in the configuration and how its change takes effect.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Change {
    /// The option's path, such as `scheduler_config.chunked_prefill_size`.
    pub path: String,
    /// How its change takes effect.
    pub effect: Effect,
}

impl Change {
    /// The change of the option at `path`, taking effect as `effect`.
    pub fn new(path: String, effect: Effect) -> Self {
        Self { path, effect }
    }
}

/// A configuration structure whose fields declare how a change of each takes effect.
///
/// Derived, never written by hand: `#[derive(Reload)]` with one `#[reload(...)]` attribute per
/// field ([`proto_derive::Reload`]).
pub trait Reload {
    /// Appends to `changes` every option of `next`, under `path`, that differs from `previous`,
    /// with its effect; with no previous configuration every option is a change.
    fn reload(previous: Option<&Self>, next: &Self, path: &str, changes: &mut Vec<Change>);

    /// Takes from `next` the value of every option whose change would take effect as `effect`:
    /// what a process that applies that effect alone, such as the daemon's successor, serves
    /// with while the rest of the change is still to come.
    fn take(&mut self, next: &Self, effect: Effect);
}

/// The option path `key` under `path`.
pub fn join(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_owned()
    } else {
        format!("{path}.{key}")
    }
}

/// Every option `next` changes of `previous`, with its effect, in declaration order; every
/// option of `next` when the deployment starts from nothing (`previous` is `None`).
pub fn changes<T: Reload>(previous: Option<&T>, next: &T) -> Vec<Change> {
    let mut changes = Vec::new();
    T::reload(previous, next, "", &mut changes);
    changes
}

/// The effects one change executes once merged: what each component of the deployment does.
///
/// The daemon and the Storage Servers each replace their process or keep it. The engines of a
/// replica execute every effect the change has, but one rule merges them: an epoch includes
/// every other effect an engine has, since its new processes are built from the new values.
/// A kernel generation switches bucket by bucket and so includes no step, and the experts'
/// order, a kernel generation and a program replacement do not include one another: each
/// prepares and switches by itself within one change.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Effects {
    /// Whether the daemon restarts as a successor, which applies the rest.
    pub daemon: bool,
    /// Whether every node's Storage Server of the deployment is replaced.
    pub storage: bool,
    /// What the engines execute, the epoch alone when there is one.
    pub engine: BTreeSet<Effect>,
}

impl Effects {
    /// The merged effects of `changes`.
    pub fn of(changes: &[Change]) -> Self {
        let mut effects = Self::default();
        for change in changes {
            match change.effect {
                Effect::Daemon => effects.daemon = true,
                // A replaced Storage Server is no store the serving engines' sessions hold: they
                // follow as an epoch.
                Effect::Storage => {
                    effects.storage = true;
                    effects.engine.insert(Effect::Epoch);
                }
                effect => {
                    effects.engine.insert(effect);
                }
            }
        }
        if effects.engine.contains(&Effect::Epoch) {
            effects.engine = BTreeSet::from([Effect::Epoch]);
        }
        effects
    }

    /// The heaviest effect of the change, which a report names it by; `None` for a change of
    /// nothing.
    pub fn heaviest(&self) -> Option<Effect> {
        if self.daemon {
            return Some(Effect::Daemon);
        }
        let engine = self.engine.iter().next_back().copied();
        if self.storage {
            return engine.max(Some(Effect::Storage));
        }
        engine
    }

    /// Whether the engines execute `effect`.
    pub fn has(&self, effect: Effect) -> bool {
        self.engine.contains(&effect)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scheduler's limits: two options of one table.
    #[derive(Debug, Clone, Default, PartialEq, Reload)]
    struct Limits {
        #[reload(step)]
        chunk: u32,
        #[reload(by = budget)]
        budget: u32,
    }

    /// A budget that shrinks takes effect at the next step; one that grows needs new processes.
    fn budget(previous: Option<&u32>, next: &u32) -> Effect {
        match previous {
            Some(previous) if next <= previous => Effect::Step,
            _ => Effect::Epoch,
        }
    }

    /// A configuration of a nested table and options of three other effects.
    #[derive(Debug, Clone, Default, PartialEq, Reload)]
    struct Config {
        #[reload(fields)]
        limits: Limits,
        #[reload(generation)]
        kernel: String,
        #[reload(daemon)]
        level: String,
        #[reload(epoch)]
        layout: u32,
    }

    fn paths(changes: &[Change]) -> Vec<(&str, Effect)> {
        changes
            .iter()
            .map(|change| (change.path.as_str(), change.effect))
            .collect()
    }

    /// A start changes every option, each with its declared effect, under its full path.
    #[test]
    fn a_start_changes_every_option() {
        let start = changes(None, &Config::default());
        assert_eq!(
            paths(&start),
            [
                ("limits.chunk", Effect::Step),
                ("limits.budget", Effect::Epoch),
                ("kernel", Effect::Generation),
                ("level", Effect::Daemon),
                ("layout", Effect::Epoch),
            ]
        );
        let effects = Effects::of(&start);
        assert!(effects.daemon);
        assert_eq!(effects.engine, BTreeSet::from([Effect::Epoch]));
    }

    /// Only the options that differ are changes, and a value-dependent effect is decided from
    /// both values.
    #[test]
    fn only_differing_options_change() {
        let base = Config {
            limits: Limits {
                chunk: 8192,
                budget: 16,
            },
            ..Config::default()
        };
        assert!(changes(Some(&base), &base).is_empty());

        let mut smaller = base.clone();
        smaller.limits.chunk = 4096;
        smaller.limits.budget = 8;
        assert_eq!(
            paths(&changes(Some(&base), &smaller)),
            [
                ("limits.chunk", Effect::Step),
                ("limits.budget", Effect::Step)
            ]
        );

        let mut larger = base.clone();
        larger.limits.budget = 32;
        assert_eq!(
            paths(&changes(Some(&base), &larger)),
            [("limits.budget", Effect::Epoch)]
        );
    }

    /// An epoch includes the steps, generations and programs of the same change; the daemon
    /// and the Storage Servers act on their own, and a Storage Server's replacement moves the
    /// engines to an epoch.
    #[test]
    fn an_epoch_includes_the_engines_lighter_effects() {
        let change = |effects: &[Effect]| {
            effects
                .iter()
                .map(|effect| Change::new(String::new(), *effect))
                .collect::<Vec<_>>()
        };
        let light = Effects::of(&change(&[Effect::Step, Effect::Generation, Effect::Order]));
        assert_eq!(
            light.engine,
            BTreeSet::from([Effect::Step, Effect::Order, Effect::Generation])
        );
        assert_eq!(light.heaviest(), Some(Effect::Generation));

        let epoch = Effects::of(&change(&[Effect::Step, Effect::Epoch, Effect::Program]));
        assert_eq!(epoch.engine, BTreeSet::from([Effect::Epoch]));

        let storage = Effects::of(&change(&[Effect::Step, Effect::Storage]));
        assert!(storage.storage);
        assert_eq!(storage.engine, BTreeSet::from([Effect::Epoch]));
        assert_eq!(storage.heaviest(), Some(Effect::Epoch));

        assert_eq!(Effects::of(&[]).heaviest(), None);
    }

    /// Taking one effect's options moves exactly those, so a daemon's successor serves the new
    /// daemon options beside the old engine options.
    #[test]
    fn taking_an_effect_moves_its_options_alone() {
        let base = Config {
            limits: Limits {
                chunk: 8192,
                budget: 16,
            },
            ..Config::default()
        };
        let target = Config {
            limits: Limits {
                chunk: 4096,
                budget: 8,
            },
            kernel: "trtllm".to_owned(),
            level: "debug".to_owned(),
            layout: 2,
        };
        let mut daemon = base.clone();
        daemon.take(&target, Effect::Daemon);
        assert_eq!(daemon.level, "debug");
        assert_eq!(changes(Some(&daemon), &target).len(), 4);

        let mut stepped = base.clone();
        stepped.take(&target, Effect::Step);
        assert_eq!(stepped.limits, target.limits);
        assert_eq!(stepped.kernel, base.kernel);
    }
}
