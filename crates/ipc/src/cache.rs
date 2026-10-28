//! The cache groups a worker reports.

use serde::{Deserialize, Serialize};

/// History retention policy of a cache group.
///
/// A model whose layers read different histories (for example full and
/// sliding-window attention) reports one group per policy and page shape.
/// Every group draws its pages from the same unit pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CacheGroupKind {
    /// Full attention: every page is retained for the request's lifetime.
    #[default]
    Full,
    /// Sliding-window attention. A reader starting at absolute position `q`
    /// needs the history `[q - window, q)`, so pages that lie entirely before
    /// the oldest position any remaining reader needs are released while the
    /// request continues. `window` is the longest history any reader of the
    /// group needs, not the checkpoint's nominal window.
    SlidingWindow {
        /// History tokens any reader of the group needs. A table keeps one
        /// contiguous run of pages, so no prefix is retained outside the
        /// window.
        window: u32,
    },
    /// Checkpoints of the layers a worker keeps per request slot, such as a
    /// recurrent state: page `k` holds every such layer's state after token
    /// `(k + 1) * block_size - 1`. A reader continuing from token `t` needs
    /// only the page that ends at `t`, the page a one-token window holds;
    /// a table names units only for the pages that hold or receive a
    /// checkpoint and the padding sentinel, unit 0, for the others.
    State,
    /// The draft cache: the K/V of a drafter's layers, which read the whole
    /// history. It is optional derived state: a request decodes correctly
    /// without it, verifying no drafts, so its pages never bound a reuse
    /// of the other groups, and a table may name the padding sentinel for
    /// pages it holds no units for. The entry at position `i` depends on the
    /// tokens before `i + 2`, so a page's contents depend on the token after
    /// the page as well, which its key covers.
    Draft,
}

impl CacheGroupKind {
    /// Returns the history bound in tokens, or `None` for full attention
    /// and for state checkpoints.
    pub const fn window(self) -> Option<u32> {
        match self {
            Self::Full | Self::State | Self::Draft => None,
            Self::SlidingWindow { window } => Some(window),
        }
    }

    /// Whether the group holds state checkpoints rather than K/V rows.
    pub const fn is_state(self) -> bool {
        matches!(self, Self::State)
    }

    /// Whether the group holds the draft cache, which a request may lack.
    pub const fn is_draft(self) -> bool {
        matches!(self, Self::Draft)
    }
}

/// One cache group of a worker's unit pool, reported at handshake.
///
/// A logical page of the group holds `block_size` tokens of every layer in
/// the group and occupies `units_per_page` physical units. Unit and column
/// geometry belong to the worker; the scheduler only needs the page shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheGroup {
    /// History retention policy of the group.
    pub kind: CacheGroupKind,
    /// Tokens stored per logical page; a power of two.
    pub block_size: u32,
    /// Physical units one logical page occupies.
    pub units_per_page: u32,
    /// Global cache-layer ids of this rank's layers in the group, in the
    /// column order the worker stores them.
    pub layer_ids: Vec<u32>,
    /// KV heads this rank stores per layer; for a state group, the heads of
    /// the layers' state partition this rank keeps.
    pub num_kv_heads: u32,
    /// Logical KV heads of each layer across all tensor-parallel members.
    pub total_kv_heads: u32,
    /// First logical KV head stored by this rank.
    pub kv_head_offset: u32,
    /// Elements stored per KV head; one for a state group, whose pages hold
    /// bytes rather than K/V rows.
    pub head_dim: u32,
}

impl CacheGroup {
    /// Whether two groups share retention policy, page shape and logical head
    /// geometry, whatever layers and heads each rank stores.
    pub fn same_page_shape(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.block_size == other.block_size
            && self.units_per_page == other.units_per_page
            && self.total_kv_heads == other.total_kv_heads
            && self.head_dim == other.head_dim
    }
}
