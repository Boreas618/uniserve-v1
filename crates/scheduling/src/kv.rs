//! A KV program's inputs and outputs.
//!
//! A KV program decides how KV moves. One instance of it runs on every Storage Server of the
//! deployment, stepped at two hooks ([`Hook`]):
//!
//! - **handoff**, on the Storage Server of the decode side, for every prefill-to-decode handoff:
//!   whether the full pages are pushed into pages the decode replica reserved, read from the
//!   prefill replica's device memory, or pulled by key from a tier ([`Path`]), and over which
//!   transport;
//! - **tiering**, every `period` on the leader node's Storage Server, which holds the cluster
//!   directory: which pages to move, copy, prefetch or demote ([`Move`]).
//!
//! A KV program crate implements one [`Program`] for each hook and exports the pair as one
//! program, [`Kv`]: `export!(Kv<Direct, Demote>)`. Its outputs describe only KV actions, which
//! the Storage Server carries out; they change no configuration. A KV move between tiers and
//! hosts is called a move here; migration names only the migration of a request.

use std::fmt;

use serde::de::{self, DeserializeOwned, Deserializer};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::program::Program;
use crate::time::Instant;

/// A page's content key: the digest of the chain of pages before it, its cache group, its
/// modality and its tokens, rooted at the group's seed and the request's isolation key. Two
/// pages with one key hold the same KV.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Key(pub [u8; 32]);

impl fmt::Debug for Key {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in &self.0[..6] {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

/// A level of KV storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// A replica's device memory.
    Device,
    /// The node's host memory, which its Storage Server holds.
    Host,
    /// Mooncake Store, across hosts.
    Store,
}

/// A transfer implementation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transport {
    /// NIXL.
    Nixl,
    /// Mooncake Transfer Engine.
    Mooncake,
}

/// A handoff, as the decode side's Storage Server sees it when the prefill has computed the
/// prompt: the input of the handoff hook.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Handoff {
    /// The prompt's full pages, which have keys.
    pub pages: u32,
    /// The prompt's tokens after its last full page, which have no key: they and any recurrent
    /// state are pushed or read, never pulled.
    pub tail: u32,
    /// The leading full pages each tier already holds for the decode side.
    pub found: Vec<Found>,
    /// Whether the prefill replica runs on the decode side's node.
    pub local: bool,
    /// The transports both sides have.
    pub transports: Vec<Transport>,
}

/// The leading full pages of a handoff one tier holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Found {
    pub tier: Tier,
    pub pages: u32,
}

/// How a handoff's full pages reach the decode replica: the output of the handoff hook.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Path {
    /// The prefill writes into pages the decode reserved, overlapping its compute.
    Push(Transport),
    /// The decode reads the pages from the prefill's device memory, under a lease.
    Read(Transport),
    /// The decode pulls the pages by key from the node's host memory or from Mooncake Store.
    Pull(Tier),
}

/// The input of the tiering hook: the step's instant and the cluster directory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tiering {
    pub time: Instant,
    pub directory: Directory,
}

/// The cluster directory: which pages each node of the deployment holds on each of its tiers, as
/// the nodes last reported them, and which pages Mooncake Store holds, as its master's object
/// events stated them. It may lag the pages' authoritative state on their nodes and in the
/// store, which costs only reuse: a page a move names that is gone is skipped.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Directory {
    /// Every node that reported, the node of the Storage Server that steps the hook first.
    pub nodes: Vec<Node>,
    /// The pages Mooncake Store holds; empty when the directory subscribes to no master's
    /// events (`kv_cache_config.mooncake.events`).
    #[serde(default)]
    pub store: Vec<Key>,
}

/// One node of the directory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Node {
    /// The node's host name.
    pub name: String,
    /// What it holds on its device and host tiers.
    pub tiers: Vec<Holding>,
}

/// The pages one tier of a node holds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Holding {
    pub tier: Tier,
    /// The keys it holds, least recently used first.
    pub keys: Vec<Key>,
    /// The pages it can hold.
    pub capacity: u64,
}

/// One KV action of the tiering hook: the page `key` goes from one place to another, and stays
/// at its source when `copy`. A move to a lower tier of the same node is a demotion; a copy
/// toward the replica a queued request will run on is a prefetch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Move {
    pub key: Key,
    pub from: Location,
    pub to: Location,
    pub copy: bool,
}

/// A place a page is held: a node, by its index in [`Directory::nodes`], and a tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Location {
    pub node: usize,
    pub tier: Tier,
}

/// The input of a KV program's step: the hook the Storage Server steps it at.
///
/// The host encodes it with borrowed inputs (`Hook<&Handoff, &Tiering>`); the program decodes
/// the owned form.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Hook<H = Handoff, T = Tiering> {
    Handoff(H),
    Tiering(T),
}

/// The output of a KV program's step: a handoff's path, or the tiering hook's moves. On the wire
/// it is the inner value, so the host decodes the hook's own output type.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Action {
    Path(Path),
    Moves(Vec<Move>),
}

/// A KV program: the program `H` of the handoff hook and the program `T` of the tiering hook,
/// as one instance with one state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Kv<H, T> {
    pub handoff: H,
    pub tiering: T,
}

/// A KV program's arguments: the package's one `args` table, which each hook's program reads as
/// its own `Args`.
#[derive(Clone, Debug, PartialEq)]
pub struct Args<H, T> {
    pub handoff: H,
    pub tiering: T,
}

impl<'de, H: DeserializeOwned, T: DeserializeOwned> Deserialize<'de> for Args<H, T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let table = Value::deserialize(deserializer)?;
        Ok(Args {
            handoff: H::deserialize(&table).map_err(de::Error::custom)?,
            tiering: T::deserialize(&table).map_err(de::Error::custom)?,
        })
    }
}

impl<H, T> Program for Kv<H, T>
where
    H: Program<Input = Handoff, Output = Path>,
    T: Program<Input = Tiering, Output = Vec<Move>>,
{
    type Args = Args<H::Args, T::Args>;
    type Input = Hook;
    type Output = Action;

    fn new(args: Self::Args) -> Self {
        Kv {
            handoff: H::new(args.handoff),
            tiering: T::new(args.tiering),
        }
    }

    fn step(&mut self, hook: &Hook) -> Action {
        match hook {
            Hook::Handoff(handoff) => Action::Path(self.handoff.step(handoff)),
            Hook::Tiering(tiering) => Action::Moves(self.tiering.step(tiering)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The host encodes a hook with borrowed inputs and the program decodes it, and the program's
    /// action decodes as the hook's own output.
    #[test]
    fn hooks_and_actions_cross_the_wire_as_their_hooks_types() {
        let handoff = Handoff {
            pages: 3,
            tail: 8,
            found: vec![Found {
                tier: Tier::Host,
                pages: 2,
            }],
            local: false,
            transports: vec![Transport::Nixl],
        };
        let wire = serde_json::to_vec(&Hook::<&Handoff, &Tiering>::Handoff(&handoff)).unwrap();
        assert_eq!(
            serde_json::from_slice::<Hook>(&wire).unwrap(),
            Hook::Handoff(handoff)
        );

        let action = serde_json::to_vec(&Action::Path(Path::Pull(Tier::Store))).unwrap();
        assert_eq!(
            serde_json::from_slice::<Path>(&action).unwrap(),
            Path::Pull(Tier::Store)
        );
        let moves = serde_json::to_vec(&Action::Moves(Vec::new())).unwrap();
        assert_eq!(serde_json::from_slice::<Vec<Move>>(&moves).unwrap(), []);
    }
}
