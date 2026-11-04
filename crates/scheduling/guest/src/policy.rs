//! The scheduling policy interface and its instantiation message.

use serde::{Deserialize, Serialize};

use crate::decision::{Plan, Room, Shortage, Victim};
use crate::event::Event;
use crate::ids::PageId;
use crate::limits::{Limits, Parameters};
use crate::view::{
    CallView, ExpertLoad, ExpertOrder, RankView, ReplicaView, RequestView, Snapshot,
};

/// What the mechanism hands a policy at instantiation: the policy program's
/// arguments (design §5.1) and the state of everything already running.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Init {
    /// The program's arguments, from the deployment configuration.
    #[serde(default)]
    pub parameters: Parameters,
    /// Every object that exists at instantiation.
    #[serde(default)]
    pub state: Snapshot,
}

/// A scheduling policy: the implementation half of design §4.4.
///
/// The mechanism calls these methods; a policy holds its own state and returns
/// decisions. It is implemented inside a policy module and executed by the
/// mechanism as a WASM program, but the trait is the interface either side is
/// written against. The methods are infallible as the design gives them: a
/// policy never reports a bad decision here — the mechanism validates every
/// decision and a trap (including an exhausted execution budget) is the
/// policy's only way to fail.
pub trait Policy {
    /// Called once at instantiation with the whole current state, so a policy
    /// put in mid-flight knows everything that already exists.
    fn init(&mut self, init: &Init);

    /// Once per step: every event since the previous step, in order.
    fn observe(&mut self, events: &[Event]);

    /// Once per step: what this step does.
    fn dispatch(&mut self, room: &Room) -> Plan;

    /// When device KV units, the host KV page pool or the encoder cache is
    /// short: which objects to free, in order.
    fn reclaim(&mut self, shortage: &Shortage) -> Vec<Victim>;

    /// When a request reaches the main process: which DP rank serves it.
    fn route(&mut self, request: &RequestView, ranks: &[RankView]) -> usize;

    /// When a request is first called on a block: which replica runs it.
    fn place(&mut self, call: &CallView, replicas: &[ReplicaView]) -> usize;

    /// During an epoch's precopy: which pages to warm, in order.
    fn warm(&mut self, room: &Room) -> Vec<PageId>;

    /// Periodically: the placement of experts on ranks.
    fn experts(&mut self, load: &ExpertLoad) -> ExpertOrder;

    /// Once at instantiation: the policy's limits, from its parameters, within
    /// the build config's ceilings.
    fn limits(&self) -> Limits;
}
