//! The cluster directory (`specs/kv-data-plane.md`): which pages each node of a deployment holds
//! on its device and host tiers, and which pages Mooncake Store holds.
//!
//! Every Storage Server reports its own node every `storage_config.directory_period_ms`: it
//! reads what its buffer pools hold (`BufferPool::holdings`) and records it in its own directory,
//! and, on every node but the leader's, also sends it to the leader node's server
//! (`storage_config.directory`, [`Request::Holdings`]) from a helper thread ([`report`]). The
//! leader's server so holds the deployment's whole view: the routing program's prefix hits are
//! read from it ([`Request::Hits`]) and the KV program's tiering hook steps over it
//! ([`Directory::view`]).
//!
//! A node's report replaces what the directory held of that node, and the directory forgets a
//! node whose reporting connection closed; nothing is written to disk, and a server that
//! restarts or reconnects states its node again with its next report. Which pages Mooncake Store
//! holds is not the directory's to keep: the store's master owns its objects. When routing needs
//! the store's hits (`kv_cache_config.mooncake.events`), the server that holds the directory
//! subscribes to the master's object events, stored and removed, and keeps only which keys the
//! store holds ([`events`]), as Dynamo's router indexes a shared tier.
//!
//! Everything here is a belief that may lag the tiers: the authoritative state of a page is its
//! holder's, and every request by key is answered by the holder's own index, so a stale entry
//! costs only reuse, never a wrong page (`specs/kv-data-plane.md` §2, D1 and D8).
//!
//! [`Request::Holdings`]: ::storage::protocol::Request::Holdings
//! [`Request::Hits`]: ::storage::protocol::Request::Hits

mod events;
mod report;

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::os::fd::{AsFd as _, BorrowedFd};
use std::time::Duration;

use ::storage::protocol::{Hits, Holding, Key, NodeHits};
use ::storage::runtime::Tier;

use crate::ClientId;
use crate::timer::Timer;

use events::Store;

/// The bit of a key's tiers in a node's index: its device caches and its host tier.
const DEVICE: u8 = 1;
const HOST: u8 = 2;

/// The cluster directory of every deployment this server serves, its reporting and its
/// subscription to Mooncake's object events.
pub(crate) struct Directory {
    /// This host's name, which its own node is known by.
    node: String,
    /// Per deployment, what each node reported.
    views: HashMap<String, View>,
    /// Which keys Mooncake Store holds, as its master's events stated them.
    store: Store,
    /// Fires every period: this node's report.
    timer: Timer,
    period: Duration,
    /// The leader node's server this node reports to; `None` on the leader's node.
    leader: Option<String>,
    /// The helper that sends this node's reports to the leader's server, started at the first.
    reporter: Option<report::Reporter>,
    /// The subscription to the master's events, on the server that holds the directory.
    subscription: Option<events::Subscription>,
}

/// One deployment's view: each node's report, by host name.
#[derive(Default)]
struct View {
    nodes: BTreeMap<String, Node>,
}

/// What one node reported.
struct Node {
    /// The connection the report came on; `None` for this server's own node.
    from: Option<ClientId>,
    tiers: Vec<Holding>,
    /// Each key the node holds, with the tiers it holds it on.
    index: HashMap<Key, u8>,
}

impl Node {
    fn new(from: Option<ClientId>, tiers: Vec<Holding>) -> Self {
        let mut index: HashMap<Key, u8> = HashMap::new();
        for holding in &tiers {
            let bit = match holding.tier {
                Tier::Device => DEVICE,
                Tier::Host => HOST,
                Tier::Remote => continue,
            };
            for key in &holding.keys {
                *index.entry(*key).or_default() |= bit;
            }
        }
        Self { from, tiers, index }
    }

    /// The leading keys of `chain` the node holds on the tier of `bit`.
    fn leading(&self, chain: &[Key], bit: u8) -> u32 {
        let held = chain
            .iter()
            .take_while(|key| self.index.get(*key).is_some_and(|tiers| tiers & bit != 0))
            .count();
        u32::try_from(held).unwrap_or(u32::MAX)
    }
}

impl Directory {
    /// The directory of a server serving with `settings` on the host `node`: reporting every
    /// period from now on, and subscribed to the master's events when the server holds the
    /// directory and the connector names the master's publisher.
    ///
    /// # Errors
    ///
    /// Fails when the timer or the subscription's thread cannot be made.
    pub(crate) fn new(node: String, settings: &::storage::Settings) -> io::Result<Self> {
        let period = settings.server.directory_period();
        let timer = Timer::new()?;
        timer.arm(period)?;
        let leader = settings.server.directory.as_ref().map(ToString::to_string);
        let events = settings
            .connector
            .as_ref()
            .and_then(|connector| connector.events.as_ref());
        let subscription = match (&leader, events) {
            (None, Some(endpoint)) => Some(events::Subscription::start(endpoint.to_string())?),
            _ => None,
        };
        Ok(Self {
            node,
            views: HashMap::new(),
            store: Store::default(),
            timer,
            period,
            leader,
            reporter: None,
            subscription,
        })
    }

    /// Serve with `settings` from now on: the next report goes to the leader they name, at the
    /// period they state. The subscription stays as the server started it: a change of the
    /// connector's settings starts a new server.
    pub(crate) fn configure(&mut self, settings: &::storage::ServerSettings) {
        self.leader = settings.directory.as_ref().map(ToString::to_string);
        let period = settings.directory_period();
        if period != self.period {
            self.period = period;
            if let Err(error) = self.timer.arm(period) {
                tracing::warn!(%error, "the directory's period is not armed");
            }
        }
    }

    /// The descriptors the server's loop waits on: the period's timer and the subscription's
    /// wake.
    pub(crate) fn descriptors(&self) -> impl Iterator<Item = BorrowedFd<'_>> {
        let wake = self
            .subscription
            .as_ref()
            .map(|subscription| subscription.wake());
        std::iter::once(self.timer.as_fd()).chain(wake)
    }

    /// Whether the period came, armed for the next: the node reports now.
    pub(crate) fn due(&mut self) -> bool {
        if !self.timer.fired() {
            return false;
        }
        if let Err(error) = self.timer.arm(self.period) {
            tracing::warn!(%error, "the directory's period is not armed again");
        }
        true
    }

    /// This node's report of what it holds of each deployment (`holdings`): recorded in this
    /// server's own directory, and sent to the leader's server when this node is not the
    /// leader's.
    pub(crate) fn report(&mut self, holdings: Vec<(String, Vec<Holding>)>) {
        if let Some(leader) = &self.leader {
            let reports = holdings
                .iter()
                .map(|(deployment, tiers)| report::Report {
                    deployment: deployment.clone(),
                    node: self.node.clone(),
                    tiers: tiers.clone(),
                })
                .collect();
            let reporter = match &mut self.reporter {
                Some(reporter) => Some(reporter),
                None => match report::Reporter::start() {
                    Ok(reporter) => Some(self.reporter.insert(reporter)),
                    Err(error) => {
                        tracing::warn!(%error, "the directory's reporter does not start");
                        None
                    }
                },
            };
            if let Some(reporter) = reporter {
                reporter.send(leader.clone(), reports);
            }
        }
        for (deployment, tiers) in holdings {
            let node = self.node.clone();
            self.record(&deployment, node, None, tiers);
        }
    }

    /// Records `tiers` as what `node` holds of `deployment`'s pages, reported on the connection
    /// `from` (`None` for this server's own node), replacing what the node reported before.
    pub(crate) fn record(
        &mut self,
        deployment: &str,
        node: String,
        from: Option<ClientId>,
        tiers: Vec<Holding>,
    ) {
        let view = self.views.entry(deployment.to_owned()).or_default();
        view.nodes.insert(node, Node::new(from, tiers));
    }

    /// The connection `client` closed: the nodes it reported are forgotten until they report
    /// again.
    pub(crate) fn gone(&mut self, client: ClientId) {
        for view in self.views.values_mut() {
            view.nodes.retain(|_, node| node.from != Some(client));
        }
    }

    /// The master's events that arrived, applied to what the directory knows of Mooncake Store.
    pub(crate) fn take_events(&mut self) {
        let Some(subscription) = &self.subscription else {
            return;
        };
        for (sequence, events) in subscription.drain() {
            self.store.apply(sequence, &events);
        }
    }

    /// How much of each of `chains` every node of `deployment` holds on each tier, and Mooncake
    /// Store ([`Request::Hits`](::storage::protocol::Request::Hits)).
    pub(crate) fn hits(&self, deployment: &str, chains: &[Vec<Key>]) -> Vec<Hits> {
        let view = self.views.get(deployment);
        chains
            .iter()
            .map(|chain| Hits {
                nodes: view
                    .map(|view| {
                        view.nodes
                            .iter()
                            .map(|(name, node)| NodeHits {
                                node: name.clone(),
                                device: node.leading(chain, DEVICE),
                                host: node.leading(chain, HOST),
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                store: self.store.leading(chain),
            })
            .collect()
    }

    /// `deployment`'s directory as the KV program's tiering hook reads it: this server's own
    /// node first, then every other node that reported, by name, and the keys Mooncake Store
    /// holds.
    pub(crate) fn view(&self, deployment: &str) -> scheduling::kv::Directory {
        let tier = |tier: Tier| match tier {
            Tier::Device => Some(scheduling::kv::Tier::Device),
            Tier::Host => Some(scheduling::kv::Tier::Host),
            Tier::Remote => None,
        };
        let node = |(name, node): (&String, &Node)| scheduling::kv::Node {
            name: name.clone(),
            tiers: node
                .tiers
                .iter()
                .filter_map(|holding| {
                    Some(scheduling::kv::Holding {
                        tier: tier(holding.tier)?,
                        keys: holding
                            .keys
                            .iter()
                            .map(|key| scheduling::kv::Key(key.0))
                            .collect(),
                        capacity: holding.capacity,
                    })
                })
                .collect(),
        };

        let mut nodes = Vec::new();
        if let Some(view) = self.views.get(deployment) {
            let own = view.nodes.get_key_value(&self.node);
            nodes.extend(own.map(node));
            let others = view.nodes.iter().filter(|(name, _)| **name != self.node);
            nodes.extend(others.map(node));
        }
        scheduling::kv::Directory {
            nodes,
            store: self
                .store
                .keys()
                .map(|key| scheduling::kv::Key(key.0))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(byte: u8) -> Key {
        Key([byte; 32])
    }

    fn holding(tier: Tier, keys: &[u8]) -> Holding {
        Holding {
            tier,
            keys: keys.iter().copied().map(key).collect(),
            capacity: 16,
        }
    }

    fn directory() -> Directory {
        Directory::new("leader".into(), &::storage::Settings::default()).unwrap()
    }

    /// A chain's hit on a node is its leading keys the node holds on each tier: a key past a
    /// gap counts for nothing, as a prefix past a missing page is computed.
    #[test]
    fn a_hit_is_the_leading_pages_a_node_holds_on_each_tier() {
        let mut directory = directory();
        directory.record(
            "d",
            "a".into(),
            Some(7),
            vec![
                holding(Tier::Device, &[1, 2]),
                holding(Tier::Host, &[1, 2, 3, 5]),
            ],
        );
        let chain: Vec<Key> = [1, 2, 3, 4, 5].into_iter().map(key).collect();
        let hits = directory.hits("d", &[chain]);
        assert_eq!(
            hits,
            [Hits {
                nodes: vec![NodeHits {
                    node: "a".into(),
                    device: 2,
                    host: 3,
                }],
                store: 0,
            }]
        );
    }

    /// A node's next report replaces what it reported before, and a closed connection forgets
    /// the nodes it reported while the server's own node stays.
    #[test]
    fn a_report_replaces_its_node_and_a_closed_connection_forgets_it() {
        let mut directory = directory();
        directory.record("d", "a".into(), Some(7), vec![holding(Tier::Host, &[1, 2])]);
        directory.record("d", "a".into(), Some(7), vec![holding(Tier::Host, &[9])]);
        directory.report(vec![("d".into(), vec![holding(Tier::Device, &[1])])]);
        let chain = vec![key(1), key(2)];
        let hit = |directory: &Directory| {
            directory.hits("d", std::slice::from_ref(&chain))[0]
                .nodes
                .iter()
                .map(|node| (node.node.clone(), node.device, node.host))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            hit(&directory),
            [("a".to_owned(), 0, 0), ("leader".to_owned(), 1, 0)]
        );

        directory.gone(7);
        assert_eq!(hit(&directory), [("leader".to_owned(), 1, 0)]);
        // The tiering hook reads the server's own node first.
        directory.record("d", "0".into(), Some(8), Vec::new());
        let view = directory.view("d");
        let names: Vec<&str> = view.nodes.iter().map(|node| node.name.as_str()).collect();
        assert_eq!(names, ["leader", "0"]);
    }
}
