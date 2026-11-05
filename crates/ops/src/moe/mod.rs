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

/// Check that `view` is an `i32` metadata vector whose elements are contiguous.
///
/// The entries index this metadata from its first element, so what they need is a unit stride on
/// the last axis. They do **not** need a row width of one: a vector's last axis is its length, so a
/// width check here refuses every list longer than a single element, which is every list a real
/// batch produces.
pub(crate) fn require_i32_vector(
    operator: &'static str,
    view: &crate::tensor::TensorView,
) -> crate::Result<()> {
    if view.dtype != crate::tensor::DType::I32 {
        return Err(crate::Error::argument(
            operator,
            format!("expected i32 metadata, found {}", view.dtype),
        ));
    }
    if view.strides.last() != 1 {
        return Err(crate::Error::argument(
            operator,
            format!(
                "expected a unit stride on the metadata's last axis, found {}",
                view.strides.last()
            ),
        ));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DType, DevicePtr, Shape, TensorView};

    /// A C++ entry's dtype numbers are the operator crate's, or the build says so here.
    ///
    /// `DType::as_i64()` is the numbering every packed word uses, and a C++ header cannot call it, so
    /// each native entry copies the numbers it names. A copy drifts: `gather_gemm_nvfp4` and
    /// `route_gemm_nvfp4` carried `bf16 = 3` and `e2m1 = 10` while the crate numbered them 4 and 7,
    /// and the refusal that surfaced named the *output's* dtype — a run of the slice cost, and the
    /// constant was the defect. Every `entry.h` under `kernels/` is read, so a new entry is covered
    /// without being listed.
    #[test]
    fn a_cpp_entry_states_the_operator_crates_dtype_numbers() {
        let kernels = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        let mut read = 0_usize;
        let mut constants = 0_usize;
        for package in std::fs::read_dir(&kernels).expect("kernels/").flatten() {
            for backend in std::fs::read_dir(package.path())
                .into_iter()
                .flatten()
                .flatten()
            {
                let header = backend.path().join("src/entry.h");
                let Ok(text) = std::fs::read_to_string(&header) else {
                    continue;
                };
                read += 1;
                for (name, dtype) in [
                    ("uniserve_dtype_bf16", DType::BF16),
                    ("uniserve_dtype_e4m3", DType::F8E4M3),
                    ("uniserve_dtype_e2m1", DType::F4E2M1),
                ] {
                    let Some(after) = text.split(&format!("constexpr int64_t {name} = ")).nth(1)
                    else {
                        continue;
                    };
                    let stated: i64 = after
                        .chars()
                        .take_while(char::is_ascii_digit)
                        .collect::<String>()
                        .parse()
                        .unwrap_or_else(|_| {
                            panic!("{}: {name} states no number", header.display())
                        });
                    constants += 1;
                    assert_eq!(
                        stated,
                        dtype.as_i64(),
                        "{}: {name} is {stated}, and the operator crate numbers {dtype} as {}",
                        header.display(),
                        dtype.as_i64()
                    );
                }
            }
        }
        assert!(read >= 2, "only {read} entry header(s) were read");
        assert!(
            constants >= 4,
            "only {constants} dtype constant(s) were stated"
        );
    }

    /// A plan's tile and route lists are as long as the plan says.
    ///
    /// The metadata check demanded a row width of one on these lists, which is true only of a plan
    /// with a single tile: the first decode batch (15 tokens, 8 routes each, 120 tiles) was refused
    /// by our own check with `expected 1 elements on the last axis, found 120`.
    #[test]
    fn a_metadata_vector_longer_than_one_element_is_accepted() {
        let list = TensorView::new(DevicePtr::new(0x1000), DType::I32, Shape::new(&[120]));
        assert!(require_i32_vector("moe_sort", &list).is_ok());
    }

    /// The same check still refuses what the entries cannot read.
    #[test]
    fn a_metadata_vector_of_the_wrong_element_type_is_refused() {
        let list = TensorView::new(DevicePtr::new(0x1000), DType::BF16, Shape::new(&[120]));
        let error = require_i32_vector("moe_sort", &list).unwrap_err();
        assert!(
            error.to_string().contains("expected i32 metadata"),
            "{error}"
        );
    }
}
