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
//! `moe_utils.moe_sort`). [`Op::launch`] hands it
//! [`Call`]; the buffers are the caller's, sized by [`super::max_tiles`] and
//! [`super::max_permuted_rows`] so they are invariant across a captured graph's replays.
//!
//! A route whose expert lies outside this rank's `[local_offset, local_offset + local_experts)`
//! window is left out: the sort writes `-1` into its entry of [`SortArgs::expanded_to_permuted`] as it
//! counts, so a caller never pre-fills that map and no launch can leave an entry stale, and the
//! expert-parallel exchange carries the route to the rank that owns it.
//!
//! Two paddings are one word and they are two subjects. A **token** row the batch does not fill is a
//! row of the batch's extent that has no route; a **permuted** row past an expert's last live one is
//! padding inside a tile group, which [`SortArgs::total_padded_tokens`] counts. Neither count answers the
//! other's question, and a page that reads one as the other is the shape of defect this crate keeps
//! removing.

use std::ffi::{c_char, c_void};
use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi::{self, MESSAGE};
use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::operand::Operand;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

use super::{address, to_i64};

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
    /// Whether this program reads a live token count.
    ///
    /// It is part of the key because it is part of the [`Call`] structure and of the program's
    /// behaviour: a program built without the field compares no route against a count and marks
    /// no route `-2`, so pairing it with one that does on a bucket wider than its batch produces a
    /// difference that is the pair's rather than either program's. The value is always `true`
    /// here - the count is an operand whose *value* is runtime data, while whether the program
    /// reads one at all is what selects a compiled module - so the key gains a constant and no
    /// cell doubles.
    pub counted: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("experts", self.experts)
            .field("top", self.top)
            .field("local", self.local_experts)
            .field("tile", self.tile_tokens)
            .field("scales", abi::name(self.scales))
            .field("counted", self.counted)
            .finish(formatter)
    }
}

/// The arguments of one sort.
#[derive(Clone)]
pub struct SortArgs {
    /// `[tokens, top]` u32 route ids.
    pub ids: Operand,
    /// `[tokens, top]` route weights, in the descriptor's scale dtype.
    pub scales: Operand,
    /// `[max_tiles]` u32: the expert each tile belongs to.
    pub tile_to_expert: Operand,
    /// `[max_tiles]` u32: the permuted row one past each tile's last live row.
    pub tile_to_limit: Operand,
    /// `[tokens, top]` u32: permuted row of each expanded route, and `-1` for a route this rank does
    /// not hold, which the sort writes. The buffer needs no initial value.
    ///
    /// `-1` means one thing here and only one: **this rank does not hold the route**. A *token* row a
    /// bucket holds but the batch does not fill is a different subject - a row that is not live has no
    /// route for any rank to own - and its routes are written `-2` ([`SortArgs::live_tokens`]).
    pub expanded_to_permuted: Operand,
    /// `[max_permuted_rows]` u32: expanded route of each permuted row.
    pub permuted_to_expanded: Operand,
    /// `[max_permuted_rows]` u32: the token row each permuted row gathers, and `-1` on a tile's
    /// padding. It is the expanded route over `top`, written beside it, because a product that
    /// gathers its own rows indexes the token-major activations and takes no `top` to divide by.
    pub permuted_to_token: Operand,
    /// `[1]` u32: padded **permuted** rows the GEMMs must cover.
    ///
    /// This is the tail of an expert's last tile, and it is not the same subject as a token row a batch
    /// does not fill: one is a permuted row inside a tile group, the other is a row of the batch's own
    /// extent. A reader who needs the second cannot read it from this count.
    pub total_padded_tokens: Operand,
    /// `[1]` u32: tiles that hold live rows.
    pub num_non_exiting_tiles: Operand,
    /// `[1]` u32: the token rows **of this batch** that are live, device-resident.
    ///
    /// The subject is the batch's own rows, not the rows a buffer holds: `tokens` may be a bucket
    /// wider than the batch, and on an expert-parallel path it is a count of delivered route rows
    /// from many batches at once. Only the single-rank path, where `tokens` is the batch's row
    /// count, has padding to mark.
    ///
    /// A captured bucket is wider than the batch that fills it, and a row past the batch's extent has
    /// no route: the program writes `-2` into `expanded_to_permuted` for every route of such a
    /// row, which the combine skips because it refuses any negative index. The value lives on the
    /// device rather than in [`Call`]'s host scalars because it changes per replay of one captured
    /// graph, and a host value is read once at launch. `None` states no count, which every row live
    /// means here: a deployment that has not been given one behaves exactly as it did before the
    /// operand existed, and the kernel reads a null pointer as "all rows are live".
    ///
    /// It is a launch argument: the count's value changes per replay and changes no dimension,
    /// dtype or tile. What the key states is the separate and constant fact that this source reads
    /// a count at all ([`Descriptor::counted`]), so no cell doubles - and a module built before the
    /// count existed cannot be paired with this one, because two programs that disagree about what
    /// a padded row writes would report the pair's difference as a defect.
    pub live_tokens: Option<Operand>,
    /// First logical expert this rank holds.
    pub local_offset: u32,
}

/// The call structure the package's entry receives, declared to the packages as
/// `uniserve_moe_sort_call` (`uniserve_moe_sort.h`, generated with its layout asserted: [`abi`]).
///
/// The tile and permutation semantics follow TensorRT-LLM's routing pipeline: the outputs are the
/// tile metadata the grouped GEMMs read and the three permutation maps.
#[repr(C)]
#[derive(abi::Layout)]
pub struct Call {
    /// `[tokens, top]` i32 route ids: the router's experts, in the whole vocabulary.
    pub ids: *const i32,
    /// `[tokens, top]` route weights in `scales_dtype`, accepted and not read: the sort writes
    /// indices alone.
    pub scales: *const c_void,
    /// `[max_tiles]`: the expert of a tile.
    pub tile_to_expert: *mut i32,
    /// `[max_tiles]`: the permuted row a tile's live rows end at, counted from the routing's first
    /// permuted row (TensorRT-LLM's `mPtrCtaIdxXyToMnLimit`).
    pub tile_to_limit: *mut i32,
    /// `[tokens, top]`: each route's permuted row, -1 for a route this rank does not hold.
    pub expanded_to_permuted: *mut i32,
    /// `[max_permuted_rows]`: each permuted row's route.
    pub permuted_to_expanded: *mut i32,
    /// `[1]`: the padded permuted rows.
    pub total_padded_tokens: *mut i32,
    /// `[1]`: the tiles that hold live rows.
    pub num_non_exiting_tiles: *mut i32,
    pub tokens: i64,
    pub top: i64,
    pub experts: i64,
    pub local_experts: i64,
    pub local_offset: i64,
    pub tile_tokens: i64,
    pub scales_dtype: i64,
    /// The entry's message on a nonzero return.
    pub error: [c_char; MESSAGE],
    /// `[1]` the device count of live tokens, or null for all `tokens`.
    pub live_tokens: *const i32,
    /// `[max_permuted_rows]`: the token row a permuted row gathers, -1 on a tile's padding
    /// (TensorRT-LLM's `mPtrPermutedIdxToTokenIdx`).
    pub permuted_to_token: *mut i32,
}

impl NativeCall for Call {
    fn message(&self) -> Option<&str> {
        abi::message(&self.error)
    }
}

/// Grouping routes by expert.
pub struct MoeSort;

impl Op for MoeSort {
    const NAME: &'static str = proto::module::operator::MOE_SORT;

    type Descriptor = Descriptor;
    type Args<'a> = SortArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let top = u64::from(descriptor.top);
        let experts = u64::from(descriptor.experts);
        let local = u64::from(descriptor.local_experts);
        let tile = u64::from(descriptor.tile_tokens);
        let tokens = args.ids.rows();
        let tiles = super::max_tiles(tokens, top, local, tile);
        let permuted = super::max_permuted_rows(tokens, top, local, tile);

        args.ids.require_shape(Self::NAME, &[tokens, top])?;
        args.scales.require_shape(Self::NAME, &[tokens, top])?;
        if args.ids.dtype != DType::U32 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected u32 route ids, found {}",
                    abi::name(args.ids.dtype)
                ),
            ));
        }
        if args.scales.dtype != descriptor.scales {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "expected {} route weights, found {}",
                    abi::name(descriptor.scales),
                    abi::name(args.scales.dtype)
                ),
            ));
        }
        if !matches!(descriptor.scales, DType::F32 | DType::BF16 | DType::F16) {
            return Err(Error::Unsupported {
                operation: Self::NAME,
                reason: format!(
                    "{} route weights are not a sort weight",
                    abi::name(descriptor.scales)
                ),
            });
        }
        for view in [&args.tile_to_expert, &args.tile_to_limit] {
            view.require_shape(Self::NAME, &[tiles])?;
            super::require_u32_vector(Self::NAME, view)?;
        }
        args.expanded_to_permuted
            .require_shape(Self::NAME, &[tokens, top])?;
        super::require_u32_vector(Self::NAME, &args.expanded_to_permuted)?;
        args.permuted_to_expanded
            .require_shape(Self::NAME, &[permuted])?;
        args.permuted_to_token
            .require_shape(Self::NAME, &[permuted])?;
        args.total_padded_tokens.require_shape(Self::NAME, &[1])?;
        args.num_non_exiting_tiles.require_shape(Self::NAME, &[1])?;
        if let Some(live_tokens) = &args.live_tokens {
            live_tokens.require_shape(Self::NAME, &[1])?;
            live_tokens.require_row(Self::NAME, DType::U32, 1)?;
        }
        for view in [
            &args.expanded_to_permuted,
            &args.permuted_to_expanded,
            &args.permuted_to_token,
            &args.total_padded_tokens,
            &args.num_non_exiting_tiles,
        ] {
            if view.dtype != DType::U32 {
                return Err(Error::argument(
                    Self::NAME,
                    format!("expected u32 metadata, found {}", abi::name(view.dtype)),
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
        Ok(())
    }

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
            scales_dtype: abi::code(descriptor.scales)?,
            error: [0; MESSAGE],
            live_tokens: args.live_tokens.map_or(std::ptr::null(), |view| {
                address(view.ptr) as usize as *const i32
            }),
            permuted_to_token: address(args.permuted_to_token.ptr) as usize as *mut i32,
        };
        // SAFETY: `Call` is the call structure this package's entry declares: the declaration is
        // generated from this definition, with its layout asserted.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        SortArgs {
            tile_to_expert: twins.shadow_of(args.tile_to_expert),
            tile_to_limit: twins.shadow_of(args.tile_to_limit),
            expanded_to_permuted: twins.shadow_of(args.expanded_to_permuted),
            permuted_to_expanded: twins.shadow_of(args.permuted_to_expanded),
            permuted_to_token: twins.shadow_of(args.permuted_to_token),
            total_padded_tokens: twins.shadow_of(args.total_padded_tokens),
            num_non_exiting_tiles: twins.shadow_of(args.num_non_exiting_tiles),
            ..args
        }
    }

    /// Exact: the sort moves integers and compares floats, and moves no value.
    ///
    /// It produces an index map and a tile count from the routing scores; nothing it writes is a
    /// rounded number, so two implementations agree bit for bit and any difference is a defect
    /// rather than a rounding.
    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::exact()
    }
}

#[cfg(test)]
mod tests {
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
