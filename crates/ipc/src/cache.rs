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
}

impl CacheGroupKind {
    /// Returns the history bound in tokens, or `None` for full attention.
    pub const fn window(self) -> Option<u32> {
        match self {
            Self::Full => None,
            Self::SlidingWindow { window } => Some(window),
        }
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
    /// KV heads this rank stores per layer.
    pub num_kv_heads: u32,
    /// Logical KV heads of each layer across all tensor-parallel members.
    pub total_kv_heads: u32,
    /// First logical KV head stored by this rank.
    pub kv_head_offset: u32,
    /// Elements stored per KV head.
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
