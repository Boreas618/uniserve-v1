//! The mixture-of-experts operators of a sparse layer.
//!
//! A sparse layer routes each token to `top` experts and combines their results:
//!
//! ```text
//! routes(router scores)                    → topk_softmax      ids, weights
//! group routes by expert for the GEMMs     → moe_sort          tile and permutation maps
//! FC1: act(gate)·up of every route         → gather_gemm
//! FC2: every route's expert product        → route_gemm
//! combine the routes of each token         → moe_unpermute
//! exchange routes with other expert ranks  → expert_exchange   (expert parallel only)
//! ```
//!
//! The top-k also counts, into execution's expert load ([`load`]), the tokens each expert takes.
//!
//! The shapes are the ones a checkpoint states (e.g. 128 experts, top-k 8, hidden 2048,
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
//!
//! # Index operands
//!
//! Route ids, expert maps, the sort's tile and permutation maps and the live counts are `U32`
//! tensors: 32-bit indices, which the packages read as `int32_t`. An entry a package writes as
//! `-1` (a route this rank does not hold, a tile's padding row) therefore reads as `u32::MAX`.

pub mod combine;
pub mod exchange;
pub mod grouped;
pub mod load;
pub mod mxfp4;
pub mod routing;
pub mod sort;

pub use combine::{MoeUnpermute, UnpermuteArgs};
pub use exchange::{ExchangeArgs, ExchangeTransport, ExpertExchange};
pub use grouped::{GatherGemm, GatherGemmArgs, RouteGemm, RouteGemmArgs};
pub use load::{ExpertLoad, layer_load};
pub use routing::{TopkArgs, TopkSoftmax};
pub use sort::{MoeSort, SortArgs};

/// The MMA tile the grouped expert products are compiled for, permuted rows by output columns.
///
/// The packages compile any tile of their list and the tile is part of the key; every product of
/// a block reads the sort's tiles of `TILE.0` rows, so the sort, both products and their
/// workspaces state this one tile.
pub const TILE: (u32, u32) = (128, 128);

/// The thread-block cluster the grouped expert products are compiled for, rows by columns.
pub const CLUSTER: (u32, u32) = (1, 1);

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
pub(crate) fn row_stride(view: &crate::operand::Operand) -> u64 {
    if view.rank() >= 2 {
        view.stride(view.rank() - 2)
    } else {
        0
    }
}

/// Check that `view` is a `u32` metadata vector whose elements are contiguous.
///
/// The entries index this metadata from its first element, so what they need is a unit stride on
/// the last axis. They do **not** need a row width of one: a vector's last axis is its length, so a
/// width check here refuses every list longer than a single element, which is every list a real
/// batch produces.
pub(crate) fn require_u32_vector(
    operator: &'static str,
    view: &crate::operand::Operand,
) -> crate::Result<()> {
    if view.dtype != tensor::DType::U32 {
        return Err(crate::Error::argument(
            operator,
            format!(
                "expected u32 metadata, found {}",
                crate::abi::name(view.dtype)
            ),
        ));
    }
    if view.last_stride() != 1 {
        return Err(crate::Error::argument(
            operator,
            format!(
                "expected a unit stride on the metadata's last axis, found {}",
                view.last_stride()
            ),
        ));
    }
    Ok(())
}

/// The device address a call structure carries.
pub(crate) fn address(pointer: platform::DevicePtr) -> u64 {
    pointer.address()
}

/// An extent as the `i64` a package reads.
pub(crate) fn to_i64(operator: &'static str, value: u64) -> crate::Result<i64> {
    i64::try_from(value).map_err(|_| crate::Error::argument(operator, "an extent above i64::MAX"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operand::Operand;
    use platform::DevicePtr;
    use tensor::DType;

    /// A sort's tile and route lists are as long as the sort's bounds say.
    ///
    /// A check that demanded a row width of one on these lists holds only for a sort of a single
    /// tile: a decode batch of 15 tokens with 8 routes each has 120 tiles.
    #[test]
    fn a_metadata_vector_longer_than_one_element_is_accepted() {
        let list = Operand::new(DevicePtr::from_address(0x1000), DType::U32, &[120]);
        assert!(require_u32_vector("moe_sort", &list).is_ok());
    }

    /// The same check still refuses what the entries cannot read.
    #[test]
    fn a_metadata_vector_of_the_wrong_element_type_is_refused() {
        let list = Operand::new(DevicePtr::from_address(0x1000), DType::BF16, &[120]);
        let error = require_u32_vector("moe_sort", &list).unwrap_err();
        assert!(
            error.to_string().contains("expected u32 metadata"),
            "{error}"
        );
    }
}
