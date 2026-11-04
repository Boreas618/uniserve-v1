//! The standby main processes of the serving epoch (design §2.5).
//!
//! Beside every node's serving main process the daemon keeps a parked
//! standby child of the same group, rank and world at the next launch
//! generation. The child has its own CUDA contexts, so a sticky error of the
//! serving process cannot touch it; it maps the epoch's weights and cache
//! read-only by the existing rules, captures its graphs on warmup scratch
//! and parks, keeping its graph pools, scratch and communicators resident or
//! suspended per `standby_storage`. When the serving child fails with a
//! process-fatal fault, the node's whole failure domain is promoted: one
//! promotion per parked child of the domain, all issued before any answer is
//! awaited, because a suspended child restores its communication together
//! with its peers. The epoch record then names the promoted generation, the
//! old generation is stopped, and each node starts a new standby.
//!
//! The failure-domain rules are the engine's (`crates/engine/src/worker/
//! group/{spawn,standby,relaunch}.rs`), carried over to main processes:
//!
//! - One domain holds every main process that shares a cross-node world; a
//!   node in no cross-node group is a domain of its own.
//! - A domain promotes wholly or not at all: if any member has no parked
//!   standby, every member is replaced by a relaunch instead.
//! - A standby exists only while the epoch serves. An epoch in precopy or on
//!   probation keeps none: a fault there reverts the update.
//! - A standby that fails to start, or a parked one that dies, leaves its
//!   domain to a relaunch.
//! - A GPU that needs a reset takes its node's standby with it; the node
//!   cannot run again and only migration to another node or replica helps.
//!
//! This module holds the rules only: the caller applies the [`Action`]s it
//! returns to the real children and reports the lifecycle back through
//! [`Roster::parked`], [`Roster::failed`], [`Roster::promoted`] and the
//! rest.

use std::collections::BTreeMap;

use crate::epoch::{Failure, Status};
use crate::options::StandbyStorage;

/// One node's main process of an epoch, as the standby rules see it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MainProcess {
    /// The node, by host identity.
    pub host: String,
    /// The child's group and rank, which its standby child repeats.
    pub group: String,
    pub rank: u32,
    /// The generation that serves.
    pub generation: u64,
    /// The generation the node's next child takes: the standby, or the
    /// serving child a relaunch starts when no standby can be promoted. The
    /// epoch layer takes it from the group and rank's **single generation
    /// counter**, and it is above every live generation of that group and
    /// rank when taken; these rules never mint a generation number. `0` means
    /// none was taken: the rules then return [`Action::Allocate`] and change
    /// nothing, and the caller takes a number, records it with
    /// [`Roster::allocated`] and asks again.
    pub standby_generation: u64,
    /// The cross-node world the node's process joins; `None` when the node
    /// is in no cross-node group and is therefore a failure domain alone.
    pub world: Option<u64>,
}

/// What a node's devices can keep of a parked standby.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageFacts {
    /// Whether a device of the node is short of the share of its storage a
    /// serving main process leaves beside it for graphs and activations.
    pub crowded: bool,
    /// Whether the standby's graph pools fit on host storage and can be
    /// restored at their addresses at the promotion. A node where they do
    /// not keep no standby at all.
    pub fits_suspended: bool,
}

/// The deployment's standby options (`worker_config`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StandbyOptions {
    /// `worker_config.standby`.
    pub enabled: bool,
    /// `worker_config.standby_storage`.
    pub storage: StandbyStorage,
}

/// What a spawned standby keeps on its devices, one mode for the whole
/// domain: one crowded device suspends every member.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Storage {
    /// The graph pools stay mapped while the standby parks.
    Resident,
    /// The graph pools move to host storage while it parks and back at the
    /// promotion, at their addresses.
    Suspended,
}

/// What the daemon is to do to a child, returned by the rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// Start the host's standby child: the serving child's group and rank at
    /// `generation`, in standby mode, keeping `storage`.
    Spawn {
        host: String,
        group: String,
        rank: u32,
        generation: u64,
        storage: Storage,
    },
    /// Promote the parked child of `generation` in place (`Promote` down its
    /// connection): same process, graphs at their addresses.
    Promote { host: String, generation: u64 },
    /// Stop the child of `generation` (`Stop { group, generation }`): the
    /// old generation once its successor serves, or a serving child the
    /// domain replaces.
    Stop {
        host: String,
        group: String,
        generation: u64,
    },
    /// Start the host's serving child again, the domain having no standby to
    /// promote.
    Relaunch {
        host: String,
        group: String,
        rank: u32,
        generation: u64,
    },
    /// Terminate a standby child that can no longer be promoted.
    Terminate { host: String, generation: u64 },
    /// The node needs a generation for its next child, and only the epoch
    /// layer's counter can give one: take a number from the group and rank's
    /// counter and record it with [`Roster::allocated`].
    Allocate {
        host: String,
        group: String,
        rank: u32,
    },
    /// The node cannot run again after a device fault: its requests migrate
    /// to another node or replica.
    Migrate { host: String },
}

/// Whether a failure replaces the whole failure domain (§2.5): the kinds
/// [`Failure`] names for a process that cannot go on, and a stalled job
/// (§2.4). A request-level failure replaces nothing: a refused request, a
/// request's non-finite logits, or a scheduling policy reverted after a
/// commit.
pub fn replaces_domain(failure: &Failure) -> bool {
    matches!(
        failure.kind.as_str(),
        "sticky"
            | "illegal_address"
            | "launch_failed"
            | "out_of_memory"
            | "process_exit"
            | "stalled"
    )
}

/// Where a domain is in the standby's life.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// It has no standby: never started, dropped, or its loss already
    /// replaced.
    Absent,
    /// Its standby children start and prepare.
    Starting,
    /// Every member's standby parked.
    Parked,
    /// The domain's promotion is out and `answered` members have answered.
    Promoting { answered: usize },
    /// The domain's members are started again, `awaiting` of them not yet
    /// serving.
    Relaunching { awaiting: usize },
    /// A replacement round waits for the epoch layer's generation numbers:
    /// the caller takes one per node, records it with [`Roster::allocated`],
    /// and the round's actions follow the last one.
    Allocating,
}

/// A node's standby child.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Standby {
    generation: u64,
    parked: bool,
}

#[derive(Debug, Clone)]
struct Node {
    process: MainProcess,
    standby: Option<Standby>,
    /// It cannot run again after a device fault; no action names it.
    gone: bool,
    /// Its epoch stopped serving there (drained, purged or switched away):
    /// its standby is gone and none starts until `serve` is called again.
    retired: bool,
}

/// The epoch's main processes and the standby rules over them.
#[derive(Debug, Clone)]
pub struct Roster {
    nodes: Vec<Node>,
    /// The nodes replaced together, by node index.
    domains: Vec<Vec<usize>>,
    /// The domain of each node, by node index.
    domain_of: Vec<usize>,
    phases: Vec<Phase>,
    /// The host whose failure started a replacement round that waits for
    /// generation numbers, per domain.
    deferred: Vec<Option<String>>,
}

impl Roster {
    /// The roster of the epoch's main processes, one per node; the failure
    /// domains are derived from the cross-node worlds they join.
    pub fn new(processes: Vec<MainProcess>) -> Self {
        let mut domains: Vec<Vec<usize>> = Vec::new();
        let mut world_domain: BTreeMap<u64, usize> = BTreeMap::new();
        let mut domain_of = vec![0; processes.len()];
        for (index, process) in processes.iter().enumerate() {
            // The nodes of one cross-node world share a domain; a node in no
            // world opens one of its own.
            let domain = match process.world {
                Some(world) => *world_domain.entry(world).or_insert_with(|| {
                    domains.push(Vec::new());
                    domains.len() - 1
                }),
                None => {
                    domains.push(Vec::new());
                    domains.len() - 1
                }
            };
            domain_of[index] = domain;
            domains[domain].push(index);
        }
        let phases = vec![Phase::Absent; domains.len()];
        let deferred = vec![None; domains.len()];
        Self {
            nodes: processes
                .into_iter()
                .map(|process| Node {
                    process,
                    standby: None,
                    gone: false,
                    retired: false,
                })
                .collect(),
            domains,
            domain_of,
            phases,
            deferred,
        }
    }

    /// The failure domains: the main processes replaced, promoted or
    /// relaunched together.
    pub fn domains(&self) -> Vec<Vec<&MainProcess>> {
        self.domains
            .iter()
            .map(|domain| {
                domain
                    .iter()
                    .map(|&node| &self.nodes[node].process)
                    .collect()
            })
            .collect()
    }

    /// Where the standby of each domain stands, for the daemon's report.
    pub fn state(&self) -> Vec<DomainState<'_>> {
        self.domains
            .iter()
            .enumerate()
            .map(|(domain, nodes)| DomainState {
                hosts: nodes
                    .iter()
                    .map(|&node| self.nodes[node].process.host.as_str())
                    .collect(),
                phase: self.phases[domain],
            })
            .collect()
    }

    fn domain_of(&self, host: &str) -> Option<usize> {
        self.nodes
            .iter()
            .position(|node| node.process.host == host)
            .map(|node| self.domain_of[node])
    }

    fn live(&self, domain: usize) -> Vec<usize> {
        self.domains[domain]
            .iter()
            .copied()
            .filter(|&node| !self.nodes[node].gone && !self.nodes[node].retired)
            .collect()
    }

    fn is_parked(&self, node: usize) -> bool {
        self.nodes[node]
            .standby
            .is_some_and(|standby| standby.parked)
    }

    /// Starts the standby of every domain that may keep one: `standby` is
    /// on, the epoch serves (an epoch in precopy or on probation keeps none),
    /// the domain has no standby and some member can run.
    ///
    /// `facts` is what each node's devices can keep, as the daemon observed
    /// them when the standby is about to start; a live member without facts
    /// keeps the domain from starting one.
    pub fn serve(
        &mut self,
        status: Status,
        options: StandbyOptions,
        facts: &BTreeMap<String, StorageFacts>,
    ) -> Vec<Action> {
        if !options.enabled || status != Status::Serving {
            return Vec::new();
        }
        let mut actions = Vec::new();
        for domain in 0..self.domains.len() {
            if self.phases[domain] != Phase::Absent {
                continue;
            }
            let live = self.live(domain);
            if live.is_empty() {
                continue;
            }
            let observed: Option<Vec<StorageFacts>> = live
                .iter()
                .map(|&node| facts.get(&self.nodes[node].process.host).copied())
                .collect();
            let Some(observed) = observed else {
                continue;
            };
            let Some(storage) = storage(options.storage, &observed) else {
                // No member can keep the standby even suspended: none is
                // kept, and the domain relaunches when it fails.
                continue;
            };
            if live
                .iter()
                .any(|&node| self.nodes[node].process.standby_generation == 0)
            {
                actions.extend(self.allocate_actions(domain));
                continue;
            }
            for node in live {
                let process = self.nodes[node].process.clone();
                let generation = process.standby_generation;
                actions.push(Action::Spawn {
                    host: process.host.clone(),
                    group: process.group.clone(),
                    rank: process.rank,
                    generation,
                    storage,
                });
                self.nodes[node].standby = Some(Standby {
                    generation,
                    parked: false,
                });
                // The number is spent; the next child needs a new one.
                self.nodes[node].process.standby_generation = 0;
            }
            self.phases[domain] = Phase::Starting;
        }
        actions
    }

    /// The standby child of `host` reported that it prepared and parked:
    /// `Report::Parked { generation }`, the report that gates the start of a
    /// standby (the child mapped the epoch's weights and cache read-only,
    /// captured its graphs on warmup scratch and parked). A child whose
    /// preparation failed reports that failure instead, which the caller
    /// feeds to [`Roster::standby_lost`]: a standby world is kept whole or not
    /// at all.
    pub fn parked(&mut self, host: &str, generation: u64) {
        let Some(domain) = self.domain_of(host) else {
            return;
        };
        let Some(node) = self.nodes.iter().position(|node| node.process.host == host) else {
            return;
        };
        match self.nodes[node].standby.as_mut() {
            Some(standby) if standby.generation == generation => standby.parked = true,
            _ => return,
        }
        if self.phases[domain] == Phase::Starting
            && self.live(domain).iter().all(|&node| self.is_parked(node))
        {
            self.phases[domain] = Phase::Parked;
        }
    }

    /// A standby child of `host` is gone: it failed to start or died, or it
    /// reported that its preparation failed (`Report::Parked` never came). A
    /// domain's standby is one world: a member's loss drops every member's
    /// standby, and the domain relaunches when it next fails.
    pub fn standby_lost(&mut self, host: &str) -> Vec<Action> {
        let Some(domain) = self.domain_of(host) else {
            return Vec::new();
        };
        if self.phases[domain] != Phase::Starting && self.phases[domain] != Phase::Parked {
            return Vec::new();
        }
        let lost = self.nodes.iter().position(|node| node.process.host == host);
        let mut actions = Vec::new();
        for node in self.live(domain) {
            if Some(node) == lost {
                self.nodes[node].standby = None;
                continue;
            }
            if let Some(standby) = self.nodes[node].standby.take() {
                actions.push(Action::Terminate {
                    host: self.nodes[node].process.host.clone(),
                    generation: standby.generation,
                });
            }
        }
        self.phases[domain] = Phase::Absent;
        actions
    }

    /// The serving child of `host` failed. A process-fatal failure replaces
    /// the whole domain: its parked standbys are promoted (every one asked
    /// before any answer is awaited), or, when any member has none, every
    /// member is started again. A request-level failure replaces nothing.
    ///
    /// The caller applies whatever actions come back and reports their
    /// outcome through [`Roster::promoted`], [`Roster::promote_failed`] and
    /// [`Roster::relaunched`]; it does not assume that a fault yields a
    /// promotion, because whether one is possible is the domain's state, not
    /// the failure's.
    pub fn failed(&mut self, host: &str, failure: &Failure) -> Vec<Action> {
        if !replaces_domain(failure) {
            return Vec::new();
        }
        let Some(domain) = self.domain_of(host) else {
            return Vec::new();
        };
        if self
            .nodes
            .iter()
            .find(|node| node.process.host == host)
            .is_some_and(|node| node.gone || node.retired)
        {
            return Vec::new();
        }
        if matches!(
            self.phases[domain],
            Phase::Promoting { .. } | Phase::Relaunching { .. } | Phase::Allocating
        ) {
            // The first failure of the round wins.
            return Vec::new();
        }
        let live = self.live(domain);
        if live.is_empty() {
            return Vec::new();
        }
        let mut promotions = Vec::with_capacity(live.len());
        for &node in &live {
            match self.nodes[node].standby {
                Some(standby) if standby.parked => promotions.push((node, standby.generation)),
                _ => {
                    promotions.clear();
                    break;
                }
            }
        }
        if !promotions.is_empty() && promotions.len() == live.len() {
            self.phases[domain] = Phase::Promoting { answered: 0 };
            return promotions
                .into_iter()
                .map(|(node, generation)| Action::Promote {
                    host: self.nodes[node].process.host.clone(),
                    generation,
                })
                .collect();
        }
        // A member has no parked standby: replace the domain by relaunching
        // every member; no promotion, not even of the members that have one.
        match self.relaunch_all(host, domain) {
            Some(actions) => actions,
            None => {
                // The round waits for the epoch layer's generation numbers.
                self.phases[domain] = Phase::Allocating;
                self.deferred[domain] = Some(host.to_owned());
                self.allocate_actions(domain)
            }
        }
    }

    /// The `Allocate` action of every live node of `domain` without a number.
    fn allocate_actions(&self, domain: usize) -> Vec<Action> {
        self.live(domain)
            .into_iter()
            .filter(|&node| self.nodes[node].process.standby_generation == 0)
            .map(|node| {
                let process = &self.nodes[node].process;
                Action::Allocate {
                    host: process.host.clone(),
                    group: process.group.clone(),
                    rank: process.rank,
                }
            })
            .collect()
    }

    /// Records the generation the epoch layer took for `host`'s next child,
    /// from the group and rank's single counter. A replacement round that was
    /// waiting for numbers starts with the last one.
    pub fn allocated(&mut self, host: &str, generation: u64) -> Vec<Action> {
        let Some(node) = self.nodes.iter().position(|node| node.process.host == host) else {
            return Vec::new();
        };
        self.nodes[node].process.standby_generation = generation;
        let Some(domain) = self.domain_of(host) else {
            return Vec::new();
        };
        if self.phases[domain] != Phase::Allocating {
            return Vec::new();
        }
        if self
            .live(domain)
            .iter()
            .any(|&node| self.nodes[node].process.standby_generation == 0)
        {
            return Vec::new();
        }
        let failed = self.deferred[domain].take();
        match failed {
            Some(failed) => self.relaunch_all(&failed, domain).unwrap_or_default(),
            None => Vec::new(),
        }
    }

    /// Starts every live member of `domain` again and drops its standbys.
    /// Returns `None` when any live member has no generation for its next
    /// child: the caller then asks the epoch layer for the numbers
    /// ([`Action::Allocate`]) and the round follows the last one.
    fn relaunch_all(&mut self, failed: &str, domain: usize) -> Option<Vec<Action>> {
        let live = self.live(domain);
        if live
            .iter()
            .any(|&node| self.nodes[node].process.standby_generation == 0)
        {
            return None;
        }
        let mut actions = Vec::new();
        for &node in &live {
            let process = self.nodes[node].process.clone();
            let generation = process.generation;
            let next = process.standby_generation;
            if let Some(standby) = self.nodes[node].standby.take() {
                actions.push(Action::Terminate {
                    host: process.host.clone(),
                    generation: standby.generation,
                });
            }
            if process.host != failed {
                actions.push(Action::Stop {
                    host: process.host.clone(),
                    group: process.group.clone(),
                    generation,
                });
            }
            actions.push(Action::Relaunch {
                host: process.host.clone(),
                group: process.group.clone(),
                rank: process.rank,
                generation: next,
            });
            // The number is spent; the next child needs a new one.
            self.nodes[node].process.standby_generation = 0;
        }
        self.phases[domain] = Phase::Relaunching {
            awaiting: live.len(),
        };
        Some(actions)
    }

    /// The promotion of the host's standby took effect:
    /// `Report::Promoted { generation }`, the report that the child serves in
    /// place after its `Command::Promote`. Once every member answered, the old
    /// generation is stopped and the domain may start a new standby.
    pub fn promoted(&mut self, host: &str, generation: u64) -> Vec<Action> {
        let Some(domain) = self.domain_of(host) else {
            return Vec::new();
        };
        let Phase::Promoting { answered } = self.phases[domain] else {
            return Vec::new();
        };
        let matches = self
            .nodes
            .iter()
            .find(|node| node.process.host == host)
            .and_then(|node| node.standby)
            .is_some_and(|standby| standby.generation == generation);
        if !matches {
            return Vec::new();
        }
        let live = self.live(domain);
        let answered = answered + 1;
        if answered < live.len() {
            self.phases[domain] = Phase::Promoting { answered };
            return Vec::new();
        }
        let mut actions = Vec::new();
        for &node in &live {
            let process = self.nodes[node].process.clone();
            actions.push(Action::Stop {
                host: process.host.clone(),
                group: process.group.clone(),
                generation: process.generation,
            });
            if let Some(standby) = self.nodes[node].standby.take() {
                self.nodes[node].process.generation = standby.generation;
                // The next child needs a number from the counter.
                self.nodes[node].process.standby_generation = 0;
            }
        }
        self.phases[domain] = Phase::Absent;
        actions
    }

    /// A member's promotion failed: the domain is replaced by a relaunch.
    pub fn promote_failed(&mut self, host: &str) -> Vec<Action> {
        let Some(domain) = self.domain_of(host) else {
            return Vec::new();
        };
        if !matches!(self.phases[domain], Phase::Promoting { .. }) {
            return Vec::new();
        }
        match self.relaunch_all(host, domain) {
            Some(actions) => actions,
            None => {
                self.phases[domain] = Phase::Allocating;
                self.deferred[domain] = Some(host.to_owned());
                self.allocate_actions(domain)
            }
        }
    }

    /// The serving child of `host` started again at `generation`.
    pub fn relaunched(&mut self, host: &str, generation: u64) {
        let Some(domain) = self.domain_of(host) else {
            return;
        };
        let Phase::Relaunching { awaiting } = self.phases[domain] else {
            return;
        };
        if let Some(node) = self.nodes.iter_mut().find(|node| node.process.host == host) {
            node.process.generation = generation;
            node.process.standby_generation = 0;
        }
        self.phases[domain] = if awaiting <= 1 {
            Phase::Absent
        } else {
            Phase::Relaunching {
                awaiting: awaiting - 1,
            }
        };
    }

    /// The node cannot run again after a device fault (a GPU that needs a
    /// reset): its standby died with the serving child, and only migration to
    /// another node or replica helps. The rest of its domain is relaunched.
    ///
    /// The returned [`Action::Migrate`] names the node whose requests the
    /// epoch layer moves; where its placement goes — another node or another
    /// replica — is the epoch layer's decision, and it must not leave the
    /// epoch short a node: a deployment that lost a node re-plans it, rather
    /// than serving with a domain's member missing.
    pub fn unusable(&mut self, host: &str) -> Vec<Action> {
        let Some(domain) = self.domain_of(host) else {
            return Vec::new();
        };
        let Some(node) = self.nodes.iter().position(|node| node.process.host == host) else {
            return Vec::new();
        };
        self.nodes[node].gone = true;
        let mut actions = Vec::new();
        if let Some(standby) = self.nodes[node].standby.take() {
            actions.push(Action::Terminate {
                host: host.to_owned(),
                generation: standby.generation,
            });
        }
        let live = self.live(domain);
        if live.is_empty() {
            self.phases[domain] = Phase::Absent;
            actions.push(Action::Migrate {
                host: host.to_owned(),
            });
            return actions;
        }
        // The members that remain shared a world with the dead node: their
        // serving children are replaced, and the requests migrate off the
        // node that cannot run. The epoch's placement loses the node, which
        // the caller decides.
        match self.relaunch_all(host, domain) {
            Some(relaunched) => actions.extend(relaunched),
            None => {
                self.phases[domain] = Phase::Allocating;
                self.deferred[domain] = Some(host.to_owned());
                actions.extend(self.allocate_actions(domain));
            }
        }
        actions.push(Action::Migrate {
            host: host.to_owned(),
        });
        actions
    }
    /// The node's epoch stopped serving there: its main process was drained,
    /// purged or switched away, so its standby is terminated and none starts
    /// until [`Roster::serve`] is called again.
    ///
    /// The standby layer owns the standby child, so it stops it: the epoch
    /// layer does not need to know its generation. A replacement round in
    /// flight owns the domain's children until it ends, so a retirement is
    /// refused then; the round's own actions stop them.
    ///
    /// A revert retires the target epoch's standbys this way, and the restored
    /// epoch starts a new standby at a fresh number from the same counter: a
    /// retired child's generation is never reused.
    pub fn retired(&mut self, host: &str) -> Vec<Action> {
        let Some(domain) = self.domain_of(host) else {
            return Vec::new();
        };
        if matches!(
            self.phases[domain],
            Phase::Promoting { .. } | Phase::Relaunching { .. }
        ) {
            return Vec::new();
        }
        let Some(node) = self.nodes.iter().position(|node| node.process.host == host) else {
            return Vec::new();
        };
        self.nodes[node].retired = true;
        let mut actions = Vec::new();
        if let Some(standby) = self.nodes[node].standby.take() {
            actions.push(Action::Terminate {
                host: host.to_owned(),
                generation: standby.generation,
            });
        }
        if self.live(domain).is_empty() {
            self.phases[domain] = Phase::Absent;
        } else if self.phases[domain] == Phase::Starting
            && self.live(domain).iter().all(|&node| self.is_parked(node))
        {
            self.phases[domain] = Phase::Parked;
        }
        actions
    }
}

/// A domain's standby, as the daemon reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainState<'a> {
    /// The domain's nodes, in roster order.
    pub hosts: Vec<&'a str>,
    /// Where its standby stands: `absent`, `starting`, `parked`,
    /// `promoting` or `relaunching`.
    pub phase: Phase,
}

impl Phase {
    /// The name `GET /admin/config` reports.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Absent => "absent",
            Self::Starting => "starting",
            Self::Parked => "parked",
            Self::Promoting { .. } => "promoting",
            Self::Relaunching { .. } => "relaunching",
            Self::Allocating => "allocating",
        }
    }
}

/// The one storage mode a domain's standby children take, or `None` when no
/// member can keep one.
fn storage(mode: StandbyStorage, observed: &[StorageFacts]) -> Option<Storage> {
    let mode = match mode {
        StandbyStorage::Resident => Storage::Resident,
        StandbyStorage::Suspended => Storage::Suspended,
        StandbyStorage::Auto => {
            if observed.iter().any(|facts| facts.crowded) {
                Storage::Suspended
            } else {
                Storage::Resident
            }
        }
    };
    match mode {
        Storage::Suspended if observed.iter().any(|facts| !facts.fits_suspended) => None,
        mode => Some(mode),
    }
}

#[cfg(test)]
mod tests;
