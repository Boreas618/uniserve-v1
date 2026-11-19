//! A routing program's input and output.
//!
//! A routing program runs on the leader daemon and is stepped for every request: it reads the
//! request and the state of every replica ([`Routing`]) and chooses the replica that computes
//! the prompt and the replica that generates the tokens ([`Route`]). The leader checks every
//! route: an index outside `replicas` or a replica of the wrong role is refused, and the request
//! goes to the replica with the fewest requests in flight, equally loaded replicas in turn.

use serde::{Deserialize, Serialize};

use crate::kv::{Key, Tier};
use crate::time::Duration;

/// The input of a routing program's step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Routing {
    /// The request to route.
    pub request: Request,
    /// Every replica of the deployment; a route names them by their index here.
    pub replicas: Vec<Replica>,
}

/// A request as routing sees it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    /// The prompt's tokens.
    pub prompt: u32,
    /// The keys of the prompt's full pages, in order: the prefix a replica may already hold.
    pub prefix: Vec<Key>,
    /// The request's session hint, which keeps the turns of a conversation together.
    pub session: Option<String>,
}

/// A replica's state when the request is routed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Replica {
    /// The replica's id, as `replicas.<id>` names it in the configuration.
    pub id: String,
    /// Which requests the replica takes.
    pub role: Role,
    /// Prefill tokens waiting in its queue.
    pub queued: u64,
    /// The predicted time until its running batch completes.
    pub remaining: Duration,
    /// Requests running on it.
    pub running: u32,
    /// Its KV pages in use, by tier.
    pub usage: Vec<Usage>,
    /// The pages it could reserve now, which a push to it needs before the prefill writes.
    pub reservable: u64,
    /// How much of the request's prefix it holds, by tier, from the cluster directory: perhaps
    /// stale, which costs only reuse, never a wrong page.
    pub hits: Vec<Hit>,
}

/// The requests a replica takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Only prompts.
    Prefill,
    /// Only token generation.
    Decode,
    /// Both.
    Mixed,
}

impl Role {
    /// Whether the replica computes prompts.
    pub fn prefills(self) -> bool {
        matches!(self, Role::Prefill | Role::Mixed)
    }

    /// Whether the replica generates tokens.
    pub fn decodes(self) -> bool {
        matches!(self, Role::Decode | Role::Mixed)
    }
}

/// A replica's KV pages on one tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub tier: Tier,
    /// Pages in use.
    pub used: u64,
    /// Pages the tier holds.
    pub capacity: u64,
}

/// The prefix of a request a replica's node holds on one tier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hit {
    pub tier: Tier,
    /// The prefix's tokens held, a multiple of the page size.
    pub tokens: u32,
}

/// A routing program's output: the indices in [`Routing::replicas`] of the replica that computes
/// the prompt and the replica that generates the tokens; the same index without
/// disaggregation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Route {
    pub prefill: usize,
    pub decode: usize,
}
