//! Logical pages borrowed from a rank's retained prefix-cache planes.

use serde::{Deserialize, Serialize};

/// A complete page held by the scheduler until its transfer finishes.
/// `request` identifies the source request across source and target worlds;
/// it does not identify the physical pool or the page size.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Page {
    pub request: (u64, u64),
    pub worker_group: String,
    pub group: u32,
    pub start: u32,
    pub tokens: u32,
    pub total_tokens: u32,
    /// A state page represents its endpoint, rather than a token interval.
    pub state: bool,
    /// Retained history of a sliding or state group, in tokens.
    pub window: Option<u32>,
    /// Physical units, in the cache group's page order.
    pub units: Vec<u32>,
}

/// Accepted token facts and complete pages held by one source request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prefix {
    pub world: u64,
    pub request: u64,
    /// Origin world and request of the private continuation. Unlike the
    /// current source above, this survives successive world changes.
    pub resume: (u64, u64),
    /// The admitted request's logical extent, fixed across copy rounds.
    pub capacity_tokens: u32,
    pub tokens: Vec<u32>,
    pub computed: u32,
    pub cached: u32,
    pub isolation_key: Option<u64>,
    /// Numerical identity under which the existing pages were computed.
    pub state_seed: u64,
    pub pages: Vec<Page>,
}

/// The scheduler's frozen page references and their worker storage views.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    pub page_tokens: u32,
    pub prefixes: Vec<Prefix>,
    pub ranks: Vec<Rank>,
    pub nodes: std::collections::BTreeMap<String, String>,
}

/// One cache owner's storage views and its deployment node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rank {
    pub worker_group: String,
    pub rank: u32,
    pub host: String,
    pub storage: super::Rank,
    /// Host allocations and the logical pages they retain.
    pub allocations: Vec<(crate::Key, Page)>,
}

/// Source views resolved relative to a destination node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sources {
    pub host: String,
    /// Destination memory kind for measured transfer links on this host.
    pub host_destination: bool,
    pub sources: super::weights::Sources,
}

/// Complete page views at their device addresses or in retained host backing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pages {
    pub pages: Vec<Page>,
    pub host: bool,
}

/// Copy spans addressed to one cache owner on the destination node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Copy {
    pub worker_group: String,
    pub rank: u32,
    pub spans: Vec<crate::protocol::Span>,
}

/// A cache warming round. Leases remain held until `Release`; `Publish`
/// is legal only after all copies into reserved target pages have completed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Operation {
    Snapshot {
        world: u64,
        after: Vec<(u64, u32)>,
    },
    Reserve {
        prefixes: Vec<Prefix>,
        keep: bool,
        host: bool,
    },
    Coverage {
        prefixes: Vec<Prefix>,
        keep: bool,
    },
    Prefill {
        tokens: Option<u32>,
    },
    Retain {
        requests: Vec<(u64, u64)>,
    },
    /// Allocate host copies of the source's leased pages for rollback.
    Stage {
        prefixes: Vec<Prefix>,
    },
    /// Device destinations for every currently retained target page.
    Materialize,
    Publish {
        host: bool,
    },
    Release,
    Discard,
}
