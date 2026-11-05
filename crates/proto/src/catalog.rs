//! The graph catalog a deployment captures, and the bound it puts on a batch.
//!
//! A deployment that serves captured graphs states which shapes it captured
//! (`cuda_graph_config`). Those shapes are the batch bound the engine admits —
//! a rank never hands the device a batch larger than its largest bucket — and
//! the same figure bounds the scheduling policy's declared limits. Both ends
//! read this one function for it, so the deployment, the graphs it captured and
//! the policy it runs cannot drift apart.

/// The largest shape the captured graphs bound a batch to: the largest decode
/// batch size or prefill token count the deployment states.
///
/// `None` when the deployment states no catalog at all, in which case the
/// worker's own catalog bounds the batch and there is no stated figure to hold
/// a policy's limits against.
#[must_use]
pub fn largest_tier(decode_batch_sizes: &[u32], prefill_token_sizes: &[u32]) -> Option<u32> {
    decode_batch_sizes
        .iter()
        .chain(prefill_token_sizes)
        .copied()
        .max()
}

#[cfg(test)]
mod tests {
    use super::largest_tier;

    #[test]
    fn the_largest_tier_is_the_largest_of_either_catalog() {
        assert_eq!(largest_tier(&[1, 2, 4, 8], &[512, 2048]), Some(2048));
        assert_eq!(largest_tier(&[1, 2, 4, 8], &[]), Some(8));
        assert_eq!(largest_tier(&[], &[512]), Some(512));
        assert_eq!(largest_tier(&[], &[]), None);
    }
}
