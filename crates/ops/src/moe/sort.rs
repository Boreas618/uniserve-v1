//! Grouping routes by expert: the sort the grouped GEMMs read.
//!
//! `moe_sort` turns `[tokens, top]` route ids into the metadata the two grouped GEMMs consume:
//! which expert each tile of [`Descriptor::tile_tokens`] permuted rows belongs to, how many rows of
//! each tile are live, and the two permutation maps between expanded route rows (`t · top + k`) and
//! permuted rows (expert-grouped). It moves no token data — that is `gather_gemm`'s gather — and it
//! writes indices only, so two implementations agree bit for bit.
//!
//! The program is a native library: the sort is a multi-stage histogram/scan/scatter pipeline with
//! host-side stage selection, and its stages are the package's business (`flashinfer`'s
//! `moe_utils.moe_sort`, `uniserve/backends/moe/cutedsl.py:167-186`). [`Op::launch`] hands it
//! [`Call`]; the buffers are the caller's, sized by [`super::max_tiles`] and
//! [`super::max_permuted_rows`] so they are invariant across a captured graph's replays.
//!
//! A route whose expert lies outside this rank's `[local_offset, local_offset + local_experts)`
//! window is left out: its expanded entry receives `-1` in [`Args::expanded_to_permuted`], and the
//! expert-parallel exchange carries the route to the rank that owns it.

use std::ffi::c_void;
use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

use super::{address, dtype_code, to_i64};

/// One compilation of `moe_sort`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Experts in the router's vocabulary.
    pub experts: u32,
    /// Routes per token.
    pub top: u32,
    /// Experts this rank holds.
    pub local_experts: u32,
    /// Permuted rows per tile.
    pub tile_tokens: u32,
    /// Element type of the route weights.
    pub scales: DType,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("experts", self.experts)
            .field("top", self.top)
            .field("local", self.local_experts)
            .field("tile", self.tile_tokens)
            .field("scales", self.scales)
            .finish(formatter)
    }
}

/// The arguments of one sort.
pub struct SortArgs {
    /// `[tokens, top]` i32 route ids.
    pub ids: TensorView,
    /// `[tokens, top]` route weights, in the descriptor's scale dtype.
    pub scales: TensorView,
    /// `[max_tiles]` i32: the expert each tile belongs to.
    pub tile_to_expert: TensorView,
    /// `[max_tiles]` i32: live rows of each tile.
    pub tile_to_limit: TensorView,
    /// `[tokens, top]` i32: permuted row of each expanded route, `-1` when not on this rank.
    pub expanded_to_permuted: TensorView,
    /// `[max_permuted_rows]` i32: expanded route of each permuted row.
    pub permuted_to_expanded: TensorView,
    /// `[1]` i32: padded rows the GEMMs must cover.
    pub total_padded_tokens: TensorView,
    /// `[1]` i32: tiles that hold live rows.
    pub num_non_exiting_tiles: TensorView,
    /// First logical expert this rank holds.
    pub local_offset: u32,
}

/// The call structure the package's entry receives.
///
/// Mirrored by hand in `crates/ops/kernels/moe_sort/<backend>/include/uniserve_moe_sort.h`.
#[repr(C)]
pub struct Call {
    pub ids: *const i32,
    pub scales: *const c_void,
    pub tile_to_expert: *mut i32,
    pub tile_to_limit: *mut i32,
    pub expanded_to_permuted: *mut i32,
    pub permuted_to_expanded: *mut i32,
    pub total_padded_tokens: *mut i32,
    pub num_non_exiting_tiles: *mut i32,
    pub tokens: i64,
    pub top: i64,
    pub experts: i64,
    pub local_experts: i64,
    pub local_offset: i64,
    pub tile_tokens: i64,
    pub scales_dtype: i64,
    pub error: [u8; 512],
}

/// The size and field offsets the package's header must mirror, in bytes.
pub const SORT_CALL_LAYOUT: CallLayout = CallLayout {
    size: 632,
    ids: 0,
    scales: 8,
    tile_to_expert: 16,
    tile_to_limit: 24,
    expanded_to_permuted: 32,
    permuted_to_expanded: 40,
    total_padded_tokens: 48,
    num_non_exiting_tiles: 56,
    tokens: 64,
    top: 72,
    experts: 80,
    local_experts: 88,
    local_offset: 96,
    tile_tokens: 104,
    scales_dtype: 112,
    error: 120,
};

/// The layout of [`Call`], for the header's mirror and its test.
#[derive(Debug, PartialEq, Eq)]
pub struct CallLayout {
    pub size: usize,
    pub ids: usize,
    pub scales: usize,
    pub tile_to_expert: usize,
    pub tile_to_limit: usize,
    pub expanded_to_permuted: usize,
    pub permuted_to_expanded: usize,
    pub total_padded_tokens: usize,
    pub num_non_exiting_tiles: usize,
    pub tokens: usize,
    pub top: usize,
    pub experts: usize,
    pub local_experts: usize,
    pub local_offset: usize,
    pub tile_tokens: usize,
    pub scales_dtype: usize,
    pub error: usize,
}

const _: () = assert!(
    std::mem::size_of::<Call>() == SORT_CALL_LAYOUT.size,
    "the moe_sort call structure's size changed without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, tile_to_expert) == SORT_CALL_LAYOUT.tile_to_expert,
    "the moe_sort call structure's outputs moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, tokens) == SORT_CALL_LAYOUT.tokens,
    "the moe_sort call structure's extents moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, error) == SORT_CALL_LAYOUT.error,
    "the moe_sort call structure's `error` moved without its header"
);

impl NativeCall for Call {
    fn message(&self) -> Option<&str> {
        let end = self.error.iter().position(|&byte| byte == 0)?;
        std::str::from_utf8(&self.error[..end]).ok()
    }
}

/// Grouping routes by expert.
pub struct MoeSort;

impl Op for MoeSort {
    const NAME: &'static str = "moe_sort";

    type Descriptor = Descriptor;
    type Args<'a> = SortArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let top = u64::from(descriptor.top);
        let experts = u64::from(descriptor.experts);
        let local = u64::from(descriptor.local_experts);
        let tile = u64::from(descriptor.tile_tokens);
        let tokens = args.ids.rows();
        let tiles = super::max_tiles(tokens, top, local, tile);
        let permuted = super::max_permuted_rows(tokens, top, local, tile);

        args.ids
            .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
        args.scales
            .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
        if args.ids.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("expected i32 route ids, found {}", args.ids.dtype),
            ));
        }
        if args.scales.dtype != descriptor.scales {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} route weights, found {}",
                    descriptor.scales, args.scales.dtype
                ),
            ));
        }
        if !matches!(descriptor.scales, DType::F32 | DType::BF16 | DType::F16) {
            return Err(Error::Unsupported {
                operator: Self::NAME,
                problem: format!("{} route weights are not a sort weight", descriptor.scales),
            });
        }
        for view in [&args.tile_to_expert, &args.tile_to_limit] {
            view.require_shape(Self::NAME, Shape::new(&[tiles]))?;
            view.require_row(Self::NAME, DType::I32, 1)?;
        }
        args.expanded_to_permuted
            .require_shape(Self::NAME, Shape::new(&[tokens, top]))?;
        args.expanded_to_permuted
            .require_row(Self::NAME, DType::I32, 1)?;
        args.permuted_to_expanded
            .require_shape(Self::NAME, Shape::new(&[permuted]))?;
        args.total_padded_tokens
            .require_shape(Self::NAME, Shape::new(&[1]))?;
        args.num_non_exiting_tiles
            .require_shape(Self::NAME, Shape::new(&[1]))?;
        for view in [
            &args.expanded_to_permuted,
            &args.permuted_to_expanded,
            &args.total_padded_tokens,
            &args.num_non_exiting_tiles,
        ] {
            if view.dtype != DType::I32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected i32 metadata, found {}", view.dtype),
                ));
            }
        }
        if u64::from(args.local_offset) + local > experts {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "rank window {} + {local} exceeds {experts} experts",
                    args.local_offset
                ),
            ));
        }

        let mut call = Call {
            ids: address(args.ids.ptr) as usize as *const i32,
            scales: address(args.scales.ptr) as usize as *const c_void,
            tile_to_expert: address(args.tile_to_expert.ptr) as usize as *mut i32,
            tile_to_limit: address(args.tile_to_limit.ptr) as usize as *mut i32,
            expanded_to_permuted: address(args.expanded_to_permuted.ptr) as usize as *mut i32,
            permuted_to_expanded: address(args.permuted_to_expanded.ptr) as usize as *mut i32,
            total_padded_tokens: address(args.total_padded_tokens.ptr) as usize as *mut i32,
            num_non_exiting_tiles: address(args.num_non_exiting_tiles.ptr) as usize as *mut i32,
            tokens: to_i64(Self::NAME, tokens)?,
            top: to_i64(Self::NAME, top)?,
            experts: to_i64(Self::NAME, experts)?,
            local_experts: to_i64(Self::NAME, local)?,
            local_offset: to_i64(Self::NAME, u64::from(args.local_offset))?,
            tile_tokens: to_i64(Self::NAME, tile)?,
            scales_dtype: dtype_code(descriptor.scales),
            error: [0; 512],
        };
        // SAFETY: `Call` is the call structure this package's entry declares, and its layout is
        // pinned by the assertions above and by the test below.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        SortArgs {
            tile_to_expert: twins.shadow_of(args.tile_to_expert),
            tile_to_limit: twins.shadow_of(args.tile_to_limit),
            expanded_to_permuted: twins.shadow_of(args.expanded_to_permuted),
            permuted_to_expanded: twins.shadow_of(args.permuted_to_expanded),
            total_padded_tokens: twins.shadow_of(args.total_padded_tokens),
            num_non_exiting_tiles: twins.shadow_of(args.num_non_exiting_tiles),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_layout_is_the_one_the_header_declares() {
        let layout = SORT_CALL_LAYOUT;
        assert_eq!(std::mem::size_of::<Call>(), layout.size);
        assert_eq!(std::mem::offset_of!(Call, ids), layout.ids);
        assert_eq!(std::mem::offset_of!(Call, scales), layout.scales);
        assert_eq!(
            std::mem::offset_of!(Call, tile_to_expert),
            layout.tile_to_expert
        );
        assert_eq!(
            std::mem::offset_of!(Call, tile_to_limit),
            layout.tile_to_limit
        );
        assert_eq!(
            std::mem::offset_of!(Call, expanded_to_permuted),
            layout.expanded_to_permuted
        );
        assert_eq!(
            std::mem::offset_of!(Call, permuted_to_expanded),
            layout.permuted_to_expanded
        );
        assert_eq!(
            std::mem::offset_of!(Call, total_padded_tokens),
            layout.total_padded_tokens
        );
        assert_eq!(
            std::mem::offset_of!(Call, num_non_exiting_tiles),
            layout.num_non_exiting_tiles
        );
        assert_eq!(std::mem::offset_of!(Call, tokens), layout.tokens);
        assert_eq!(std::mem::offset_of!(Call, top), layout.top);
        assert_eq!(std::mem::offset_of!(Call, experts), layout.experts);
        assert_eq!(
            std::mem::offset_of!(Call, local_experts),
            layout.local_experts
        );
        assert_eq!(
            std::mem::offset_of!(Call, local_offset),
            layout.local_offset
        );
        assert_eq!(std::mem::offset_of!(Call, tile_tokens), layout.tile_tokens);
        assert_eq!(
            std::mem::offset_of!(Call, scales_dtype),
            layout.scales_dtype
        );
        assert_eq!(std::mem::offset_of!(Call, error), layout.error);
    }

    /// The buffer bounds follow the worst routing distribution, not one batch: every expert with a
    /// single route plus one expert with the rest.
    #[test]
    fn buffer_bounds_cover_every_routing() {
        // 2 tokens, top 2 = 4 routes; 4 local experts, so each expert can hold one route.
        assert_eq!(super::super::max_tiles(2, 2, 4, 128), 4);
        // 1024 tokens, top 8 = 8192 routes over 128 experts at 128 rows per tile: one full expert
        // set (128 tiles of 64 rows) is the floor, the bound is (8192 + 127·128)/128 = 191.
        assert_eq!(super::super::max_tiles(1024, 8, 128, 128), 191);
        assert_eq!(
            super::super::max_permuted_rows(1024, 8, 128, 128),
            191 * 128
        );
    }
}
