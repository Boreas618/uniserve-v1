//! The graph catalog a deployment captures, and the bounds it puts on a batch.
//!
//! A deployment that serves captured graphs states which shapes it captured
//! (`cuda_graph_config`). Those shapes bound the batch the engine admits — a
//! rank never hands the device a batch larger than its largest bucket — and the
//! same figures bound the scheduling policy's declared limits. Both ends read
//! this one function for them, so the deployment, the graphs it captured and
//! the policy it runs cannot drift apart.

/// The bounds a captured graph catalog puts on a batch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Tiers {
    /// The largest decode batch, in rows.
    pub rows: Option<u32>,
    /// The largest batch, in tokens: a decode bucket contributes its rows (one
    /// token each), a prefill bucket its token count.
    pub tokens: Option<u32>,
}

/// The bounds the captured graph catalog states.
///
/// `rows` counts only the decode batch sizes: a prefill graph's token count
/// says nothing about how many requests a decode step may hold. `tokens` is the
/// largest of either list, because a decode bucket's token cost is its rows.
/// Both are `None` when the deployment states no catalog, in which case the
/// worker's own catalog bounds the batch and there is no stated figure to
/// hold a policy's limits against.
#[must_use]
pub fn largest_tier(decode_batch_sizes: &[u32], prefill_token_sizes: &[u32]) -> Tiers {
    Tiers {
        rows: decode_batch_sizes.iter().copied().max(),
        tokens: decode_batch_sizes
            .iter()
            .chain(prefill_token_sizes)
            .copied()
            .max(),
    }
}

#[cfg(test)]
mod tests {
    use super::{Tiers, largest_tier};

    #[test]
    fn the_tiers_are_the_largest_of_each_kind_of_graph() {
        assert_eq!(
            largest_tier(&[1, 2, 4, 8], &[512, 2048]),
            Tiers {
                rows: Some(8),
                tokens: Some(2048),
            }
        );
        assert_eq!(
            largest_tier(&[1, 2, 4, 8], &[]),
            Tiers {
                rows: Some(8),
                tokens: Some(8),
            },
            "a decode-only catalog bounds the batch by its largest batch of rows"
        );
        assert_eq!(
            largest_tier(&[], &[512]),
            Tiers {
                rows: None,
                tokens: Some(512),
            },
            "a prefill graph states no decode batch"
        );
        assert_eq!(largest_tier(&[], &[]), Tiers::default());
    }
}
