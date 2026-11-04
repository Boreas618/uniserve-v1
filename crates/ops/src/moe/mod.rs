//! The mixture-of-experts operators of a sparse layer.
//!
//! A sparse layer routes each token to `top` experts and combines their results:
//!
//! ```text
//! routes(router scores)                    → topk_softmax      ids, weights
//! map logical experts to this rank's       → expert_gather
//! group routes by expert for the GEMMs     → moe_sort          tile and permutation maps
//! FC1: act(gate)·up of every route         → gather_gemm
//! FC2: every route's expert product        → route_gemm
//! combine the routes of each token         → moe_unpermute
//! exchange routes with other expert ranks  → expert_exchange   (expert parallel only)
//! ```
//!
//! The shapes are the ones a checkpoint states (Qwen3-30B-A3B: 128 experts, top-k 8, hidden 2048,
//! expert width 768); every operator takes its compiled shapes from its descriptor and every runtime
//! extent — the token count, the offsets — from its arguments. The metadata operators are exact;
//! the two grouped GEMMs and the combine carry the rounding bound of their own reduction.
//!
//! # Expert count, local experts and the sort's buffers
//!
//! `moe_sort` groups the `tokens · top` routes into tiles of [`sort::Descriptor::tile_tokens`] rows
//! per local expert. Its buffers are sized for the worst routing distribution, not for one batch:
//! [`max_tiles`] tiles and [`max_permuted_rows`] permuted rows (the same bounds as
//! `flashinfer.fused_moe.cute_dsl.moe_utils.get_max_num_tiles` and `..._permuted_tokens`), so a
//! captured graph replays them for any routing the batch can produce.

pub mod combine;
pub mod exchange;
pub mod grouped;
pub mod routing;
pub mod sort;

pub use combine::{MoeUnpermute, UnpermuteArgs};
pub use exchange::{ExchangeArgs, ExchangeTransport, ExpertExchange};
pub use grouped::{GatherGemm, GatherGemmArgs, RouteGemm, RouteGemmArgs};
pub use routing::{ExpertGather, GatherArgs, TopkArgs, TopkSoftmax};
pub use sort::{MoeSort, SortArgs};

/// The most tiles `skip` routes over `local_experts` can produce at `tile` rows per tile.
///
/// The worst case is every expert with a single route and one with the rest, which collapses to
/// `(expanded + (tile − 1) · local_experts) / tile`.
pub fn max_tiles(tokens: u64, top: u64, local_experts: u64, tile: u64) -> u64 {
    let expanded = tokens * top;
    if expanded <= local_experts || tile == 0 {
        expanded
    } else {
        (expanded + (tile - 1) * local_experts) / tile
    }
}

/// The rows a sort's permutation buffers hold for `max_tiles` tiles of `tile` rows.
pub fn max_permuted_rows(tokens: u64, top: u64, local_experts: u64, tile: u64) -> u64 {
    max_tiles(tokens, top, local_experts, tile) * tile
}

/// The elements between consecutive rows of a view.
pub(crate) fn row_stride(view: &crate::tensor::TensorView) -> u64 {
    if view.strides.rank() >= 2 {
        view.strides.get(view.strides.rank() - 2)
    } else {
        0
    }
}

/// The device address a call structure carries.
pub(crate) fn address(pointer: crate::tensor::DevicePtr) -> u64 {
    crate::platform::address(pointer)
}

/// An extent as the `i64` a package reads.
pub(crate) fn to_i64(operator: &'static str, value: u64) -> crate::Result<i64> {
    i64::try_from(value).map_err(|_| crate::Error::argument(operator, "an extent above i64::MAX"))
}

/// The ABI number of an element type, as a package reads it.
pub(crate) fn dtype_code(dtype: crate::tensor::DType) -> i64 {
    dtype.as_i64()
}
