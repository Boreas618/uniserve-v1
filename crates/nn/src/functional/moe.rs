//! The operators of a mixture-of-experts block: the router's top-k, the sort that groups the
//! routes by expert, the two grouped expert products, the weighted combine, the exchange of
//! routes between the ranks of an expert group, and the gated add of a shared expert.
//!
//! ```text
//! topk_softmax        router scores                 -> each token's expert ids and route weights
//! moe_sort            ids                           -> the route maps: tiles and permutations
//! gather_gemm         hidden rows, [up; gate] stack -> act(gate · x) · (up · x) per permuted row
//! route_gemm          gated rows, down stack        -> down · gated per permuted row
//! moe_unpermute       products, route weights       -> Σ_routes weight · product per token
//! ```
//!
//! A token's `top` routes are one row each of both products. The sort does not move rows: it
//! groups the routes of one expert into **tiles** of [`TILE_ROWS`] permuted rows and states the
//! maps between a route (`token · top + route`, the *expanded* row) and its *permuted* row, so
//! one launch multiplies every expert's weight. The maps' extents follow from the token count,
//! the top-k and the expert count alone, never from one batch's routing, so a captured graph
//! replays over the same addresses however the router routes a later batch.
//!
//! The packed forms ([`gather_gemm_nvfp4`], [`route_gemm_nvfp4`]) read calibrated NVFP4 expert
//! stacks and encode their own activation, as [`linear_nvfp4`](super::linear_nvfp4) does.
//!
//! ## Expert parallelism
//!
//! The ranks of an expert group each hold a window of the experts. A rank routes its tokens
//! over every expert, sends each route to the rank that holds its expert ([`expert_dispatch`]),
//! multiplies the routes it received with its own experts, leaves their sums where its peers
//! read them ([`moe_unpermute`] into the exchange's rows) and takes its own tokens' sums back
//! ([`expert_combine`]); or one launch does all of it ([`expert_exchange_fused`]). The
//! transport that carries the rows is the execution context's, so these functions state
//! tensors only. They compute on tensors of the device backend alone: one process holds one
//! rank, so CPU tensors are refused as every collective refuses them.

use std::ops::Range;

use ops::fp4::{BLOCK_ELEMENTS, ScaleLayout};
use ops::moe::ExchangeArgs;
use ops::moe::combine::{self, MoeUnpermute, UnpermuteArgs};
use ops::moe::exchange::{self, FusedLayer, Mode};
use ops::moe::grouped::{
    GatherGemm, GatherGemmArgs, GatherGemmDescriptor, GatherGemmNvfp4, GatherGemmNvfp4Args,
    GatherGemmNvfp4Descriptor, RouteGemm, RouteGemmArgs, RouteGemmDescriptor, RouteGemmNvfp4,
    RouteGemmNvfp4Args, RouteGemmNvfp4Descriptor,
};
use ops::moe::routing::{TopkArgs, TopkDescriptor, TopkSoftmax};
use ops::moe::sort::{self, MoeSort, SortArgs};
use ops::residual::{SigmoidGatedAdd, SigmoidGatedAddArgs, SigmoidGatedAddDescriptor};
use tensor::{D, DType, Device, Tensor};

use super::activation::activate;
use super::nvfp4::{decoded, encoded, integer, launch_quantize, remainder};
use super::{ACCUMULATE, Activation, as_rows, columns, on_cpu, row_tile, same_rows};
use crate::capability::ExpertExchange;
use crate::error::{Error, Result};
use crate::quantization::{Nvfp4Stack, ValueLayout};

/// Permuted rows per tile: the row extent of the tile the grouped products are compiled for,
/// which each expert's rows are padded to.
pub const TILE_ROWS: u32 = MMA_TILE.0;

/// The tile the grouped products' packages are compiled for, rows by columns.
const MMA_TILE: (u32, u32) = (128, 128);

/// The cluster shape the grouped products' packages are compiled for.
const MMA_CLUSTER: (u32, u32) = (1, 1);

/// The rows of the route maps [`moe_sort`] returns, `[SORT_MAPS, permuted rows]` U32: each row is
/// one map, in its leading entries.
pub const SORT_MAPS: usize = 7;

/// `[tiles]`: the expert each tile's rows belong to, as its index in this rank's window.
pub const TILE_TO_EXPERT: usize = 0;

/// `[tiles]`: the permuted row one past each tile's last live row.
pub const TILE_TO_LIMIT: usize = 1;

/// `[tokens · top]`: the permuted row of each route; `u32::MAX` for a route whose expert this
/// rank does not hold and `u32::MAX - 1` for a route of a token row the batch does not fill.
pub const EXPANDED_TO_PERMUTED: usize = 2;

/// `[permuted rows]`: the route of each permuted row; `u32::MAX` on a tile's padding rows.
pub const PERMUTED_TO_EXPANDED: usize = 3;

/// `[permuted rows]`: the token row each permuted row gathers; `u32::MAX` on padding rows.
pub const PERMUTED_TO_TOKEN: usize = 4;

/// `[1]`: the permuted rows of the tiles that hold routes.
pub const TOTAL_PADDED_TOKENS: usize = 5;

/// `[1]`: the tiles that hold routes.
pub const NUM_NON_EXITING_TILES: usize = 6;

/// The map entry of a route this rank does not hold and of a padding row: the kernels' `-1`.
const ABSENT: u32 = u32::MAX;

/// The map entry of a route of a token row the batch does not fill: the kernels' `-2`.
const NOT_LIVE: u32 = u32::MAX - 1;

/// The arrangement the gated product reads its `[up; gate]` stack's packed rows in: the two
/// halves interleaved row by row, then shuffled inside the blocks of one tile.
pub(crate) const GATED_VALUE_LAYOUT: ValueLayout = ValueLayout::InterleavedShuffled {
    epilogue_tile_m: MMA_TILE.0,
};

/// The arrangement the second product reads its `down` stack's packed rows in.
pub(crate) const DOWN_VALUE_LAYOUT: ValueLayout = ValueLayout::Shuffled {
    epilogue_tile_m: MMA_TILE.0,
};

/// The arrangement both packed products read their weights' block scales in.
pub(crate) const WEIGHT_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Swizzled128X4;

/// The arrangement the gated product reads its activation's block scales in: its kernels read
/// the tokens they route in the linear form.
const GATED_TOKEN_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Linear;

/// The arrangement the second product reads its activation's block scales in.
const ACTIVATION_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Swizzled128X4;

/// The bytes a packed product's entry writes before the package's workspace: a header holding
/// the padded route count.
const ENTRY_PREAMBLE: u64 = 16;

/// The router's choice: `(ids, weights)`, the `top` most probable experts of each token and
/// their probabilities, `[tokens, top]` U32 and FP32.
///
/// ```text
/// p        = softmax(scores)                       over the experts, in FP32
/// ids      = the top largest p, descending; equal ones in ascending expert order
/// weights  = p[ids] / max(Σ p[ids], FLT_EPSILON)   when `renormalize`
///          · scale[ids]                            when a scale is given
/// ids      = map[ids]                              when a map is given
/// ```
///
/// `scores` is `[tokens, experts]`; `scale`, when given, one factor per expert in a floating
/// type; `map`, when given, the `[experts]` U32 weight row of each expert, for a block whose
/// experts are not stacked in their own order. The selection is over the experts' own ids, so
/// which experts a token picks, and the order of equal scores, do not depend on the map.
///
/// # Errors
///
/// When `scores` is not a matrix, `top` is not between one and the expert count, the scale or
/// the map is not one entry per expert, or the operator refuses its arguments.
pub fn topk_softmax(
    scores: &Tensor,
    top: u32,
    renormalize: bool,
    scale: Option<&Tensor>,
    map: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = "topk_softmax";

    let &[tokens, experts] = scores.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("scores of {:?} are not [tokens, experts]", scores.dims()),
        ));
    };
    if top == 0 || top as usize > experts {
        return Err(Error::shape(
            OWNER,
            format!("a top of {top} is outside 1..={experts}"),
        ));
    }
    for (name, table) in [("scale", scale), ("map", map)] {
        if let Some(table) = table
            && table.dims() != [experts]
        {
            return Err(Error::shape(
                OWNER,
                format!(
                    "the {name} is {:?}, and {experts} experts take one entry each",
                    table.dims()
                ),
            ));
        }
    }
    let kept = top as usize;

    if on_cpu(scores) {
        let logits = scores.to_dtype(ACCUMULATE)?;
        let shifted = logits
            .broadcast_sub(&logits.max_keepdim(D::Minus1)?)?
            .exp()?;
        let probabilities = shifted.broadcast_div(&shifted.sum_keepdim(D::Minus1)?)?;

        // Descending, with equal probabilities in ascending expert order: the positions of one
        // probability are numbered as a group, and each group is ordered by expert.
        let descending = probabilities.arg_sort_last_dim(false)?;
        let sorted = probabilities.gather(&descending, D::Minus1)?;
        let previous = Tensor::cat(
            &[
                sorted.narrow(D::Minus1, 0, 1)?,
                sorted.narrow(D::Minus1, 0, experts - 1)?,
            ],
            D::Minus1,
        )?;
        let group = sorted
            .ne(&previous)?
            .to_dtype(DType::F64)?
            .cumsum(D::Minus1)?;
        let within = (group.affine(experts as f64, 0.0)? + descending.to_dtype(DType::F64)?)?
            .arg_sort_last_dim(true)?;

        let ids = descending
            .gather(&within, D::Minus1)?
            .narrow(D::Minus1, 0, kept)?
            .contiguous()?;
        let mut weights = probabilities.gather(&ids, D::Minus1)?;
        if renormalize {
            let total = weights
                .sum_keepdim(D::Minus1)?
                .maximum(f64::from(f32::EPSILON))?;
            weights = weights.broadcast_div(&total)?;
        }

        let selected = ids.flatten_all()?;
        if let Some(scale) = scale {
            let factors = scale.to_dtype(ACCUMULATE)?.index_select(&selected, 0)?;
            weights = (weights * factors.reshape((tokens, kept))?)?;
        }
        let ids = match map {
            Some(map) => map.index_select(&selected, 0)?.reshape((tokens, kept))?,
            None => ids,
        };
        return Ok((ids, weights));
    }

    let ids = backend::empty((tokens, kept), DType::U32)?;
    let weights = backend::empty((tokens, kept), DType::F32)?;

    backend::launch::<TopkSoftmax>(
        &TopkDescriptor {
            scores: scores.dtype(),
            experts: experts as u32,
            top,
            renormalize,
            scale: scale.map(Tensor::dtype),
        },
        TopkArgs {
            scores: backend::operand(scores)?,
            scale: scale.map(backend::operand).transpose()?,
            ids: backend::operand(&ids)?,
            weights: backend::operand(&weights)?,
            map: map.map(backend::operand).transpose()?,
        },
    )?;
    Ok((ids, weights))
}

/// Groups the routes `ids` by expert: the route maps the grouped products and the combine read,
/// one `[SORT_MAPS, permuted rows]` U32 tensor whose rows are the maps ([`TILE_TO_EXPERT`] to
/// [`NUM_NON_EXITING_TILES`]).
///
/// `ids` is `[tokens, top]` U32 and `weights` the routes' weights, whose element type the
/// kernel is compiled for. `experts` is the router's expert count and `local` the experts this
/// rank holds, a window of them. `live_tokens`, when given, is the `[1]` U32 count of token
/// rows the batch fills; without it every row is live.
///
/// The routes this rank holds are laid out expert by expert in the window's order, each
/// expert's from a tile boundary and in route order (`token · top + route`); an expert with no
/// route takes no tile. There are `max_tiles · TILE_ROWS` permuted rows for the worst routing
/// of this many routes ([`ops::moe::max_tiles`]), of which a batch fills the leading
/// [`TOTAL_PADDED_TOKENS`].
///
/// On the device the kernel writes the tiles that hold routes and their rows; the entries past
/// them keep what the memory held. The CPU formula writes zeros for those tiles and
/// `u32::MAX` for those rows.
///
/// # Errors
///
/// When `ids` is not `[tokens, top]` U32 with tokens, `weights` is another shape, `local` is
/// not a window of the experts, or the operator refuses its arguments.
pub fn moe_sort(
    ids: &Tensor,
    weights: &Tensor,
    experts: u32,
    local: Range<u32>,
    live_tokens: Option<&Tensor>,
) -> Result<Tensor> {
    const OWNER: &str = "moe_sort";

    let (tokens, top) = routes(OWNER, ids)?;
    if weights.dims() != ids.dims() {
        return Err(Error::shape(
            OWNER,
            format!(
                "route weights of {:?} for route ids of {:?}",
                weights.dims(),
                ids.dims()
            ),
        ));
    }
    if local.is_empty() || local.end > experts {
        return Err(Error::shape(
            OWNER,
            format!(
                "experts {}..{} are not a window of {experts}",
                local.start, local.end
            ),
        ));
    }

    let local_experts = local.end - local.start;
    let tiles = ops::moe::max_tiles(
        tokens as u64,
        top as u64,
        u64::from(local_experts),
        u64::from(TILE_ROWS),
    ) as usize;
    let rows = tiles * TILE_ROWS as usize;

    if on_cpu(ids) {
        return sort_formula(ids, &local, live_tokens, (tiles, rows));
    }

    let maps = backend::empty((SORT_MAPS, rows), DType::U32)?;
    let operand =
        |row: usize, entries: usize| -> Result<ops::Operand> { map_operand(&maps, row, entries) };

    backend::launch::<MoeSort>(
        &sort::Descriptor {
            experts,
            top: top as u32,
            local_experts,
            tile_tokens: TILE_ROWS,
            scales: weights.dtype(),
            counted: true,
        },
        SortArgs {
            ids: backend::operand(ids)?,
            scales: backend::operand(weights)?,
            tile_to_expert: operand(TILE_TO_EXPERT, tiles)?,
            tile_to_limit: operand(TILE_TO_LIMIT, tiles)?,
            expanded_to_permuted: backend::operand(&route_rows(&maps, tokens, top)?)?,
            permuted_to_expanded: operand(PERMUTED_TO_EXPANDED, rows)?,
            permuted_to_token: operand(PERMUTED_TO_TOKEN, rows)?,
            total_padded_tokens: operand(TOTAL_PADDED_TOKENS, 1)?,
            num_non_exiting_tiles: operand(NUM_NON_EXITING_TILES, 1)?,
            live_tokens: live_tokens.map(backend::operand).transpose()?,
            local_offset: local.start,
        },
    )?;
    Ok(maps)
}

/// The first product of the routed experts over a dense `[up; gate]` stack: for each permuted
/// row `p` of `maps`, whose route is of token `t` and expert `e`,
///
/// ```text
/// out[p] = act(gate_e · hidden[t]) · (up_e · hidden[t])
/// ```
///
/// with both products accumulated in FP32, the activation evaluated on the accumulators and one
/// rounding into `hidden`'s element type. `hidden` is `[tokens, hidden]`; `up_gate` is
/// `[experts, 2 · intermediate, hidden]` in the same type, each expert's up rows before its
/// gate rows, and holds the experts of the sort's window; `ids` is the `[tokens, top]` routes
/// that were sorted. The result is `[permuted rows, intermediate]`.
///
/// The device's kernel evaluates SiLU with the hardware's approximate exponential and
/// reciprocal, and leaves a tile's padding rows unspecified; the CPU formula evaluates the
/// activation exactly and writes zeros there. A device check compares live rows under the
/// operator's tolerance.
///
/// # Errors
///
/// When the stack is not `[experts, 2 · intermediate, hidden]` over `hidden`'s rows, the
/// activation is not SiLU or the tanh GELU, the maps are not a sort's of these routes, or the
/// operator refuses its arguments.
pub fn gather_gemm(
    hidden: &Tensor,
    up_gate: &Tensor,
    ids: &Tensor,
    maps: &Tensor,
    activation: Activation,
) -> Result<Tensor> {
    const OWNER: &str = "gather_gemm";

    let width = columns(OWNER, hidden)?;
    let (tokens, top) = routes(OWNER, ids)?;
    let &[experts, gated, contraction] = up_gate.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "an [up; gate] stack is [experts, 2 · intermediate, hidden], not {:?}",
                up_gate.dims()
            ),
        ));
    };
    if contraction != width as usize || gated % 2 != 0 || hidden.dims() != [tokens, contraction] {
        return Err(Error::shape(
            OWNER,
            format!(
                "a stack of {:?} over rows of {:?} routed as {:?}",
                up_gate.dims(),
                hidden.dims(),
                ids.dims()
            ),
        ));
    }
    require_gate(OWNER, activation)?;
    let (tiles, rows) = sorted(OWNER, maps, tokens * top)?;
    let intermediate = gated / 2;

    if on_cpu(hidden) {
        let live = live_rows(maps, rows)?;
        let expert = row_experts(maps, &live, rows)?;
        let token = map(maps, PERMUTED_TO_TOKEN, rows)?;
        let token = live.where_cond(&token, &token.zeros_like()?)?;

        let x = hidden.to_dtype(ACCUMULATE)?.index_select(&token, 0)?;
        let weight = up_gate.to_dtype(ACCUMULATE)?.index_select(&expert, 0)?;
        let projected = weight.matmul(&x.unsqueeze(2)?)?.squeeze(2)?;

        let up = projected.narrow(1, 0, intermediate)?;
        let gate = projected.narrow(1, intermediate, intermediate)?;
        let product = (activate(&gate, activation)? * up)?;

        return Ok(zero_padding(&product, &live)?.to_dtype(hidden.dtype())?);
    }

    let out = backend::empty((rows, intermediate), hidden.dtype())?;
    backend::launch::<GatherGemm>(
        &GatherGemmDescriptor {
            hidden: width,
            intermediate: intermediate as u32,
            experts: experts as u32,
            top: top as u32,
            activation,
            dtype: hidden.dtype(),
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        },
        GatherGemmArgs {
            hidden: backend::operand(hidden)?,
            weights: backend::operand(up_gate)?,
            ids: backend::operand(ids)?,
            // The combine's lookup, not this product's: the product writes row `p` at the
            // sort's permuted row `p`.
            expanded_to_permuted: backend::operand(&route_rows(maps, tokens, top)?)?,
            tile_to_expert: map_operand(maps, TILE_TO_EXPERT, tiles)?,
            tile_to_limit: map_operand(maps, TILE_TO_LIMIT, tiles)?,
            permuted_to_expanded: map_operand(maps, PERMUTED_TO_EXPANDED, rows)?,
            num_non_exiting_tiles: map_operand(maps, NUM_NON_EXITING_TILES, 1)?,
            out: backend::operand(&out)?,
        },
    )?;
    Ok(out)
}

/// The second product of the routed experts over a dense `down` stack: for each permuted row
/// `p` of `maps`, whose route is of expert `e`,
///
/// ```text
/// out[p] = down_e · gated[p]
/// ```
///
/// accumulated in FP32 and rounded once into `gated`'s element type. `gated` is the
/// `[permuted rows, intermediate]` rows of [`gather_gemm`]; `down` is
/// `[experts, hidden, intermediate]` in the same type. The result is `[permuted rows, hidden]`,
/// each route's product unweighted, which [`moe_unpermute`] sums.
///
/// The device's kernel writes live rows only; the CPU formula writes zeros on padding rows.
///
/// # Errors
///
/// When the stack is not `[experts, hidden, intermediate]` over `gated`'s rows, the maps are
/// not a sort's of these routes, or the operator refuses its arguments.
pub fn route_gemm(gated: &Tensor, down: &Tensor, ids: &Tensor, maps: &Tensor) -> Result<Tensor> {
    const OWNER: &str = "route_gemm";

    let intermediate = columns(OWNER, gated)?;
    let (tokens, top) = routes(OWNER, ids)?;
    let (tiles, rows) = sorted(OWNER, maps, tokens * top)?;
    let &[experts, width, contraction] = down.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "a down stack is [experts, hidden, intermediate], not {:?}",
                down.dims()
            ),
        ));
    };
    if contraction != intermediate as usize || gated.dims() != [rows, contraction] {
        return Err(Error::shape(
            OWNER,
            format!(
                "a stack of {:?} over gated rows of {:?} for {rows} permuted rows",
                down.dims(),
                gated.dims()
            ),
        ));
    }

    if on_cpu(gated) {
        let live = live_rows(maps, rows)?;
        let expert = row_experts(maps, &live, rows)?;

        let weight = down.to_dtype(ACCUMULATE)?.index_select(&expert, 0)?;
        let product = weight
            .matmul(&gated.to_dtype(ACCUMULATE)?.unsqueeze(2)?)?
            .squeeze(2)?;

        return Ok(zero_padding(&product, &live)?.to_dtype(gated.dtype())?);
    }

    let out = backend::empty((rows, width), gated.dtype())?;
    backend::launch::<RouteGemm>(
        &RouteGemmDescriptor {
            hidden: width as u32,
            intermediate,
            experts: experts as u32,
            top: top as u32,
            dtype: gated.dtype(),
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        },
        RouteGemmArgs {
            rows: backend::operand(gated)?,
            weights: backend::operand(down)?,
            ids: backend::operand(ids)?,
            tile_to_expert: map_operand(maps, TILE_TO_EXPERT, tiles)?,
            tile_to_limit: map_operand(maps, TILE_TO_LIMIT, tiles)?,
            permuted_to_expanded: map_operand(maps, PERMUTED_TO_EXPANDED, rows)?,
            num_non_exiting_tiles: map_operand(maps, NUM_NON_EXITING_TILES, 1)?,
            out: backend::operand(&out)?,
        },
    )?;
    Ok(out)
}

/// The first product of the routed experts over a calibrated NVFP4 `[up; gate]` stack: the
/// hidden rows are encoded and the gated product is taken over the two encodings.
///
/// ```text
/// x̂[t]    = nvfp4_quantize(hidden[t])                     under the stack's encode scale
/// up      = (up_e ⊙ x̂[t]) · tensor_scale[0, e]            ⊙: Σ over codes · block scales,
/// gate    = (gate_e ⊙ x̂[t]) · tensor_scale[1, e]             each row's factor included
/// out[p]  = act(gate) · up                                 in FP32, rounded once
/// ```
///
/// `hidden` is `[tokens, hidden]` BF16 or FP16. `up_gate` holds the experts of the sort's
/// window with its packed rows interleaved and shuffled as the product reads them and its
/// block scales swizzled; `top` is the routes of a token. The result is
/// `[permuted rows, intermediate]` in `hidden`'s element type.
///
/// The CPU formula encodes as [`nvfp4_quantize`](super::nvfp4_quantize)'s formula does and
/// evaluates the activation exactly; the device's products gate with SiLU. A device check
/// compares live rows under the operators' tolerance, which covers a code one step apart.
///
/// # Errors
///
/// When the stack is not `[experts, 2 · intermediate, hidden / 2]` over `hidden`'s rows of
/// whole blocks, the maps are not a sort's, or an operator refuses its arguments.
pub fn gather_gemm_nvfp4(
    hidden: &Tensor,
    up_gate: &Nvfp4Stack,
    maps: &Tensor,
    activation: Activation,
    top: u32,
) -> Result<Tensor> {
    const OWNER: &str = "gather_gemm_nvfp4";

    let width = columns(OWNER, hidden)?;
    let &[experts, gated, packed] = up_gate.values.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "a packed [up; gate] stack is [experts, 2 · intermediate, hidden / 2], not {:?}",
                up_gate.values.dims()
            ),
        ));
    };
    if packed * 2 != width as usize
        || !width.is_multiple_of(BLOCK_ELEMENTS)
        || gated % 2 != 0
        || hidden.rank() != 2
        || top == 0
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "a packed stack of {:?} over rows of {:?} with {top} routes a token",
                up_gate.values.dims(),
                hidden.dims()
            ),
        ));
    }
    require_gate(OWNER, activation)?;
    let tokens = hidden.dims()[0];
    let (tiles, rows) = sorted(OWNER, maps, tokens * top as usize)?;
    let intermediate = gated / 2;

    if on_cpu(hidden) {
        let live = live_rows(maps, rows)?;
        let expert = row_experts(maps, &live, rows)?;
        let token = map(maps, PERMUTED_TO_TOKEN, rows)?;
        let token = live.where_cond(&token, &token.zeros_like()?)?;

        let x = decoded_rows(hidden, &up_gate.encode_scale)?.index_select(&token, 0)?;

        // The stack's rows in their own order: up row `i` is stored at the shuffle of `2i` and
        // gate row `i` at the shuffle of `2i + 1`.
        let stored = decoded(&up_gate.values, &up_gate.scales, up_gate.layout)?;
        let half = Tensor::arange(0_u32, intermediate as u32, &Device::Cpu)?;
        let doubled = half.broadcast_mul(&integer(2, &half)?)?;
        let order = Tensor::cat(
            &[
                shuffled(&doubled)?,
                shuffled(&doubled.broadcast_add(&integer(1, &half)?)?)?,
            ],
            0,
        )?;
        let weight = stored.index_select(&order, 1)?.index_select(&expert, 0)?;
        let projected = weight.matmul(&x.unsqueeze(2)?)?.squeeze(2)?;

        // Each half is dequantized by its own projection's tensor scale.
        let scales = up_gate.global_scale.index_select(&expert, 1)?;
        let up = projected
            .narrow(1, 0, intermediate)?
            .broadcast_mul(&scales.get(0)?.unsqueeze(1)?)?;
        let gate = projected
            .narrow(1, intermediate, intermediate)?
            .broadcast_mul(&scales.get(1)?.unsqueeze(1)?)?;
        let product = (activate(&gate, activation)? * up)?;

        return Ok(zero_padding(&product, &live)?.to_dtype(hidden.dtype())?);
    }

    // 1. The hidden rows' encoding, in the layout the gated product reads its tokens in.
    let values = backend::empty((tokens, packed), DType::U8)?;
    let blocks = GATED_TOKEN_SCALE_LAYOUT.shape(tokens as u64, u64::from(width));
    let scales = backend::empty((blocks.dim(0) as usize, blocks.dim(1) as usize), DType::U8)?;
    let token_scale = backend::empty(tokens, DType::F32)?;
    launch_quantize(
        hidden,
        &up_gate.encode_scale,
        GATED_TOKEN_SCALE_LAYOUT,
        (
            backend::operand(&values)?,
            backend::operand(&scales)?,
            backend::operand(&token_scale)?,
        ),
    )?;

    // 2. The product, token-major: it gathers each permuted row's token itself.
    let out = backend::empty((rows, intermediate), hidden.dtype())?;
    let workspace = backend::empty(
        workspace_bytes(rows as u64, gated as u64) as usize,
        DType::U8,
    )?;
    backend::launch::<GatherGemmNvfp4>(
        &GatherGemmNvfp4Descriptor {
            hidden: width,
            intermediate: intermediate as u32,
            experts: experts as u32,
            top,
            activation,
            dtype: hidden.dtype(),
            // The weights' scale type is the checkpoint's; the layout is the tokens', which
            // the encoding above wrote.
            scale: up_gate.scales.dtype(),
            layout: GATED_TOKEN_SCALE_LAYOUT,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        },
        GatherGemmNvfp4Args {
            hidden: backend::operand(&values)?,
            hidden_scale: backend::operand(&scales)?,
            hidden_token_scale: backend::operand(&token_scale)?,
            weights: backend::operand(&up_gate.values)?,
            scales: backend::operand(&up_gate.scales)?,
            global_scale: backend::operand(&up_gate.global_scale)?,
            tile_to_expert: map_operand(maps, TILE_TO_EXPERT, tiles)?,
            tile_to_limit: map_operand(maps, TILE_TO_LIMIT, tiles)?,
            permuted_to_token: map_operand(maps, PERMUTED_TO_TOKEN, rows)?,
            num_non_exiting_tiles: map_operand(maps, NUM_NON_EXITING_TILES, 1)?,
            out: backend::operand(&out)?,
            workspace: backend::operand(&workspace)?,
        },
    )?;
    Ok(out)
}

/// The second product of the routed experts over a calibrated NVFP4 `down` stack: the gated
/// rows are encoded and each permuted row is multiplied by its expert's weight.
///
/// ```text
/// ĝ[p]    = nvfp4_quantize(gated[p])                      under the stack's encode scale
/// out[p]  = (down_e ⊙ ĝ[p]) · tensor_scale[0, e]          in FP32, rounded once
/// ```
///
/// `gated` is the `[permuted rows, intermediate]` rows of [`gather_gemm_nvfp4`]. `down` holds
/// the experts of the sort's window with its packed rows shuffled as the product reads them and
/// its block scales swizzled; `top` is the routes of a token. The result is
/// `[permuted rows, hidden]` in `gated`'s element type.
///
/// The CPU formula encodes as [`nvfp4_quantize`](super::nvfp4_quantize)'s formula does and
/// writes zeros on padding rows; a device check compares live rows under the operators'
/// tolerance.
///
/// # Errors
///
/// When the stack is not `[experts, hidden, intermediate / 2]` over `gated`'s rows of whole
/// blocks, the maps are not a sort's of these rows, or an operator refuses its arguments.
pub fn route_gemm_nvfp4(
    gated: &Tensor,
    down: &Nvfp4Stack,
    maps: &Tensor,
    top: u32,
) -> Result<Tensor> {
    const OWNER: &str = "route_gemm_nvfp4";

    let intermediate = columns(OWNER, gated)?;
    let &[experts, width, packed] = down.values.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "a packed down stack is [experts, hidden, intermediate / 2], not {:?}",
                down.values.dims()
            ),
        ));
    };
    let &[_, rows] = maps.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("route maps of {:?}", maps.dims()),
        ));
    };
    let (tiles, _) = sorted(OWNER, maps, 0)?;
    if packed * 2 != intermediate as usize
        || !intermediate.is_multiple_of(BLOCK_ELEMENTS)
        || gated.dims() != [rows, packed * 2]
        || top == 0
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "a packed stack of {:?} over gated rows of {:?} for {rows} permuted rows",
                down.values.dims(),
                gated.dims()
            ),
        ));
    }

    if on_cpu(gated) {
        let live = live_rows(maps, rows)?;
        let expert = row_experts(maps, &live, rows)?;

        let x = decoded_rows(gated, &down.encode_scale)?;
        let stored = decoded(&down.values, &down.scales, down.layout)?;
        let order = shuffled(&Tensor::arange(0_u32, width as u32, &Device::Cpu)?)?;
        let weight = stored.index_select(&order, 1)?.index_select(&expert, 0)?;

        let scales = down.global_scale.index_select(&expert, 1)?.get(0)?;
        let product = weight
            .matmul(&x.unsqueeze(2)?)?
            .squeeze(2)?
            .broadcast_mul(&scales.unsqueeze(1)?)?;

        return Ok(zero_padding(&product, &live)?.to_dtype(gated.dtype())?);
    }

    // 1. The gated rows' encoding: the first product's output is a 16-bit row, so this is an
    //    encoding of its own and not a wider operand of the second product.
    let values = backend::empty((rows, packed), DType::U8)?;
    let blocks = ACTIVATION_SCALE_LAYOUT.shape(rows as u64, u64::from(intermediate));
    let scales = backend::empty((blocks.dim(0) as usize, blocks.dim(1) as usize), DType::U8)?;
    let token_scale = backend::empty(rows, DType::F32)?;
    launch_quantize(
        gated,
        &down.encode_scale,
        ACTIVATION_SCALE_LAYOUT,
        (
            backend::operand(&values)?,
            backend::operand(&scales)?,
            backend::operand(&token_scale)?,
        ),
    )?;

    // 2. The product: one row per permuted row, the `top` routes of a token still separate.
    let out = backend::empty((rows, width), gated.dtype())?;
    let workspace = backend::empty(
        workspace_bytes(rows as u64, width as u64) as usize,
        DType::U8,
    )?;
    backend::launch::<RouteGemmNvfp4>(
        &RouteGemmNvfp4Descriptor {
            hidden: width as u32,
            intermediate,
            experts: experts as u32,
            top,
            dtype: gated.dtype(),
            scale: down.scales.dtype(),
            layout: ACTIVATION_SCALE_LAYOUT,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        },
        RouteGemmNvfp4Args {
            hidden: backend::operand(&values)?,
            hidden_scale: backend::operand(&scales)?,
            hidden_token_scale: backend::operand(&token_scale)?,
            weights: backend::operand(&down.values)?,
            scales: backend::operand(&down.scales)?,
            global_scale: backend::operand(&down.global_scale)?,
            tile_to_expert: map_operand(maps, TILE_TO_EXPERT, tiles)?,
            tile_to_limit: map_operand(maps, TILE_TO_LIMIT, tiles)?,
            permuted_to_expanded: map_operand(maps, PERMUTED_TO_EXPANDED, rows)?,
            num_non_exiting_tiles: map_operand(maps, NUM_NON_EXITING_TILES, 1)?,
            out: backend::operand(&out)?,
            workspace: backend::operand(&workspace)?,
        },
    )?;
    Ok(out)
}

/// The weighted sum of each token's routes:
///
/// ```text
/// out[t] = Σ_k weights[t, k] · products[row(t, k)]          over the routes that have a row
/// ```
///
/// in FP32 and in route order, rounded once into `products`' element type. `products` is the
/// `[permuted rows, hidden]` rows of the second product, `weights` the `[tokens, top]` route
/// weights and `maps` the sort's maps of those routes; a route the sort gave no row (its expert
/// is another rank's, or its token row is not live) contributes nothing, so a token with none
/// is zeros. `experts` is the router's expert count, which names the kernel. The result is
/// `[tokens, hidden]`.
///
/// `combined`, when given, is where the sums are written and what is returned: the rows an
/// expert exchange lends for the sums this rank's experts return to their tokens' ranks
/// ([`crate::ExpertExchange::combined`]), `[tokens, hidden]` in `products`' element type.
/// Without it the result is a new tensor.
///
/// # Errors
///
/// When the weights are not `[tokens, top]`, the maps are not a sort's of these routes over
/// `products`' rows, `combined` is another shape or type, or the operator refuses its
/// arguments.
pub fn moe_unpermute(
    products: &Tensor,
    maps: &Tensor,
    weights: &Tensor,
    experts: u32,
    combined: Option<&Tensor>,
) -> Result<Tensor> {
    const OWNER: &str = "moe_unpermute";

    let width = columns(OWNER, products)?;
    let (tokens, top) = routes(OWNER, weights)?;
    let (_, rows) = sorted(OWNER, maps, tokens * top)?;
    if products.dims() != [rows, width as usize] {
        return Err(Error::shape(
            OWNER,
            format!("products of {:?} for {rows} permuted rows", products.dims()),
        ));
    }
    if let Some(combined) = combined
        && (combined.dims() != [tokens, width as usize] || combined.dtype() != products.dtype())
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "the lent rows are {} {:?}, and {tokens} tokens of {} products take \
                 [{tokens}, {width}]",
                ops::abi::name(combined.dtype()),
                combined.dims(),
                ops::abi::name(products.dtype())
            ),
        ));
    }

    if on_cpu(products) {
        let row = map(maps, EXPANDED_TO_PERMUTED, tokens * top)?;
        let placed = row.lt(rows as f64)?;
        let row = placed.where_cond(&row, &row.zeros_like()?)?;

        let weight =
            (weights.to_dtype(ACCUMULATE)?.flatten_all()? * placed.to_dtype(ACCUMULATE)?)?;
        let sums = products
            .to_dtype(ACCUMULATE)?
            .index_select(&row, 0)?
            .broadcast_mul(&weight.unsqueeze(1)?)?
            .reshape((tokens, top, width as usize))?
            .sum(1)?
            .to_dtype(products.dtype())?;

        return match combined {
            Some(combined) => {
                combined.slice_set(&sums, 0, 0)?;
                Ok(combined.clone())
            }
            None => Ok(sums),
        };
    }

    let out = match combined {
        Some(combined) => combined.clone(),
        None => backend::empty((tokens, width as usize), products.dtype())?,
    };
    backend::launch::<MoeUnpermute>(
        &combine::Descriptor {
            // The second product writes one row per permuted row.
            layout: combine::Layout::Permuted,
            hidden: width,
            top: top as u32,
            experts,
            input: products.dtype(),
            scales: weights.dtype(),
            output: out.dtype(),
        },
        UnpermuteArgs {
            routes: backend::operand(products)?,
            expanded_to_permuted: backend::operand(&route_rows(maps, tokens, top)?)?,
            scales: backend::operand(weights)?,
            out: backend::operand(&out)?,
        },
    )?;
    Ok(out)
}

/// Sends the routes of this rank's tokens to the ranks that hold their experts: `hidden`,
/// `[tokens, hidden]`, with the routes' `ids` (weight rows of the whole group's experts) and
/// `weights`, both `[tokens, top]`. What the rank received stands in `exchange`'s tensors
/// afterwards.
///
/// `live_tokens`, when given, is the `[1]` U32 count of token rows the batch fills. `experts`
/// is the router's expert count and `intermediate` one expert's gated width, which name the
/// kernel. A rank that joins a step with no tokens sends zero rows.
///
/// # Errors
///
/// When the tensors are CPU tensors, more rows are sent than a rank's block holds, or the
/// exchange refuses.
pub fn expert_dispatch(
    hidden: &Tensor,
    ids: &Tensor,
    weights: &Tensor,
    live_tokens: Option<&Tensor>,
    exchange: &ExpertExchange,
    experts: u32,
    intermediate: u32,
) -> Result<()> {
    const OWNER: &str = "expert_dispatch";

    let descriptor = exchange_descriptor(
        Mode::Dispatch,
        exchange,
        (experts, intermediate),
        exchange.states.dtype(),
    )?;
    backend::collective::expert_exchange(
        &descriptor,
        ExchangeArgs {
            transport: None,
            tokens: sent(OWNER, hidden.dims()[0], exchange)?,
            state: backend::operand(hidden)?,
            ids: backend::operand(ids)?,
            weights: backend::operand(weights)?,
            live: live_tokens.map(backend::operand).transpose()?,
            out: unread(),
            fused: None,
            ranks: exchange.ranks,
        },
    )?;
    Ok(())
}

/// Takes back the sums of this rank's `tokens` tokens from the ranks whose experts computed
/// them, rounded once: `[tokens, hidden]` in the exchanged rows' element type.
///
/// Each rank has left the sums of the rows it received in `exchange`'s `combined` rows
/// ([`moe_unpermute`]). `live_tokens`, `experts` and `intermediate` are as [`expert_dispatch`]
/// takes them.
///
/// # Errors
///
/// When the exchange's tensors are CPU tensors, more rows are asked than a rank's block holds,
/// or the exchange refuses.
pub fn expert_combine(
    tokens: usize,
    live_tokens: Option<&Tensor>,
    exchange: &ExpertExchange,
    experts: u32,
    intermediate: u32,
) -> Result<Tensor> {
    const OWNER: &str = "expert_combine";

    let dtype = exchange.states.dtype();
    let descriptor = exchange_descriptor(Mode::Combine, exchange, (experts, intermediate), dtype)?;
    let out = own_rows(tokens, exchange, dtype)?;

    backend::collective::expert_exchange(
        &descriptor,
        ExchangeArgs {
            transport: None,
            tokens: sent(OWNER, tokens, exchange)?,
            state: unread(),
            ids: unread(),
            weights: unread(),
            live: live_tokens.map(backend::operand).transpose()?,
            out: backend::operand(&out)?,
            fused: None,
            ranks: exchange.ranks,
        },
    )?;
    Ok(out)
}

/// The whole expert layer of an expert group in one launch: the routes of this rank's tokens
/// are exchanged, every rank's packed experts run over the rows they receive, and each token's
/// sums come back. The result is `[tokens, hidden]` in `hidden`'s element type.
///
/// `hidden` is encoded first, into the rows `exchange` lends for it (`packed`,
/// `packed_scales`), where the peers' kernels read it. `ids` and `weights` are the
/// `[tokens, top]` routes. `up_gate` and `down` are this rank's stacks in the arrangement the
/// launch reads (each gate block before its up block, the block scales swizzled), and `tables`
/// its three `[local]` FP32 tables in the order the kernel multiplies them in: what
/// dequantizes the first product, what dequantizes the second, and the scale the gated rows are
/// encoded with. `experts` is the router's expert count.
///
/// # Errors
///
/// When the tensors are CPU tensors, more rows are sent than a rank's block holds, or an
/// operator refuses its arguments.
#[expect(
    clippy::too_many_arguments,
    reason = "the launch reads the routes, the lent rows and every operand of the layer's experts"
)]
pub fn expert_exchange_fused(
    hidden: &Tensor,
    ids: &Tensor,
    weights: &Tensor,
    live_tokens: Option<&Tensor>,
    exchange: &ExpertExchange,
    up_gate: &Nvfp4Stack,
    down: &Nvfp4Stack,
    tables: [&Tensor; 3],
    experts: u32,
) -> Result<Tensor> {
    const OWNER: &str = "expert_exchange_fused";

    let tokens = hidden.dims()[0];
    let local = up_gate.values.dims().first().copied().unwrap_or(0).max(1);
    let intermediate = up_gate.values.dims().get(1).copied().unwrap_or(0) / 2;

    // The rows are encoded straight into the rows the transport's peers read, with the encoder
    // the stack's other products take: the encoding is the module's, and the kernel starts
    // from the bytes.
    if tokens > 0 {
        let token_scale = backend::empty(tokens, DType::F32)?;
        let scales = backend::operand(&exchange.packed_scales.narrow(0, 0, tokens)?)?;

        launch_quantize(
            hidden,
            &up_gate.encode_scale,
            ScaleLayout::Linear,
            (
                backend::operand(&exchange.packed.narrow(0, 0, tokens)?)?,
                // The encoder writes each scale's byte, whatever type the transport labels
                // its rows.
                ops::Operand {
                    dtype: DType::U8,
                    ..scales
                },
                backend::operand(&token_scale)?,
            ),
        )?;
    }

    let address = |tensor: &Tensor| -> Result<u64> { Ok(backend::operand(tensor)?.ptr.address()) };
    let expert_bytes = |tensor: &Tensor| -> Result<i64> {
        i64::try_from(tensor.elem_count() / local)
            .map_err(|_| Error::shape(OWNER, "a scale table beyond i64".to_owned()))
    };
    let [up_gate_alpha, down_alpha, gated_scale] = tables;
    let layer = FusedLayer {
        up_gate: address(&up_gate.values)?,
        up_gate_scales: address(&up_gate.scales)?,
        down: address(&down.values)?,
        down_scales: address(&down.scales)?,
        up_gate_alpha: address(up_gate_alpha)?,
        down_alpha: address(down_alpha)?,
        gated_scale: address(gated_scale)?,
        up_gate_scale_bytes: expert_bytes(&up_gate.scales)?,
        down_scale_bytes: expert_bytes(&down.scales)?,
    };

    let descriptor = exchange_descriptor(
        Mode::Fused,
        exchange,
        (experts, intermediate as u32),
        hidden.dtype(),
    )?;
    let out = own_rows(tokens, exchange, hidden.dtype())?;

    backend::collective::expert_exchange(
        &descriptor,
        ExchangeArgs {
            transport: None,
            tokens: sent(OWNER, tokens, exchange)?,
            state: unread(),
            ids: backend::operand(ids)?,
            weights: backend::operand(weights)?,
            live: live_tokens.map(backend::operand).transpose()?,
            out: backend::operand(&out)?,
            fused: Some(layer),
            ranks: exchange.ranks,
        },
    )?;
    Ok(out)
}

/// A shared expert's join, in place: `target + sigmoid(gate) · update`, each row's update scaled
/// by the sigmoid of its own logit, in FP32 and rounded once into `target`'s element type.
///
/// `target` and `update` are `[..., columns]` of one element type and `gate` is `[..., 1]` in
/// the same type, one logit per row. `target` is taken by value, updated in its own storage and
/// returned: every tensor that shares its storage sees the sum.
///
/// # Errors
///
/// When `update` is not `target`'s shape and element type, `gate` is not one logit per row,
/// `target` is a strided view, or the operator refuses its arguments.
pub fn sigmoid_gated_add_(target: Tensor, update: &Tensor, gate: &Tensor) -> Result<Tensor> {
    const OWNER: &str = "sigmoid_gated_add_";

    let width = columns(OWNER, &target)?;
    same_rows(OWNER, "update", update, &target)?;

    let leading = &target.dims()[..target.rank() - 1];
    let logits = &gate.dims()[..gate.rank().saturating_sub(1)];
    if gate.dims().last() != Some(&1) || logits != leading || gate.dtype() != target.dtype() {
        return Err(Error::shape(
            OWNER,
            format!(
                "the gate is {} {:?}, and rows of {} {:?} take one logit each",
                ops::abi::name(gate.dtype()),
                gate.dims(),
                ops::abi::name(target.dtype()),
                target.dims()
            ),
        ));
    }
    if !target.is_contiguous() {
        return Err(Error::shape(
            OWNER,
            "a strided view has no rows to update in place".to_owned(),
        ));
    }

    if on_cpu(&target) {
        let scaled = activate(&gate.to_dtype(ACCUMULATE)?, Activation::Sigmoid)?
            .broadcast_mul(&update.to_dtype(ACCUMULATE)?)?;
        let sum = (target.to_dtype(ACCUMULATE)? + scaled)?.to_dtype(target.dtype())?;

        target.slice_set(&sum, 0, 0)?;
        return Ok(target);
    }

    backend::launch::<SigmoidGatedAdd>(
        &SigmoidGatedAddDescriptor {
            dtype: target.dtype(),
            columns: width,
            block: row_tile(width).min(1024),
        },
        SigmoidGatedAddArgs {
            target: backend::operand(&as_rows(OWNER, &target)?)?,
            update: backend::operand(&as_rows(OWNER, update)?)?,
            gate: backend::operand(&as_rows(OWNER, gate)?)?,
        },
    )?;
    Ok(target)
}

/// The token and route counts of `routes`, a `[tokens, top]` tensor of route ids or weights
/// with at least one token.
fn routes(owner: &'static str, routes: &Tensor) -> Result<(usize, usize)> {
    match *routes.dims() {
        [tokens, top] if tokens > 0 && top > 0 => Ok((tokens, top)),
        _ => Err(Error::shape(
            owner,
            format!("routes of {:?} are not [tokens, top]", routes.dims()),
        )),
    }
}

/// The tiles and permuted rows of `maps`, checked to be a sort's maps that hold `routes`
/// routes.
fn sorted(owner: &'static str, maps: &Tensor, routes: usize) -> Result<(usize, usize)> {
    match *maps.dims() {
        [SORT_MAPS, rows]
            if maps.dtype() == DType::U32
                && rows >= routes.max(1)
                && rows.is_multiple_of(TILE_ROWS as usize) =>
        {
            Ok((rows / TILE_ROWS as usize, rows))
        }
        _ => Err(Error::shape(
            owner,
            format!(
                "route maps of {} {:?} are not a sort's [{SORT_MAPS}, permuted rows] u32 maps of \
                 {routes} routes",
                ops::abi::name(maps.dtype()),
                maps.dims()
            ),
        )),
    }
}

/// The first `entries` entries of the map `row` of `maps`.
fn map(maps: &Tensor, row: usize, entries: usize) -> Result<Tensor> {
    Ok(maps.get(row)?.narrow(0, 0, entries)?)
}

/// The kernel argument of the first `entries` entries of the map `row` of `maps`.
fn map_operand(maps: &Tensor, row: usize, entries: usize) -> Result<ops::Operand> {
    Ok(backend::operand(&map(maps, row, entries)?)?)
}

/// The permuted row of each route of `maps`, as the `[tokens, top]` matrix the kernels read.
fn route_rows(maps: &Tensor, tokens: usize, top: usize) -> Result<Tensor> {
    Ok(map(maps, EXPANDED_TO_PERMUTED, tokens * top)?.reshape((tokens, top))?)
}

/// Refuses an activation the grouped products have no kernel for.
fn require_gate(owner: &'static str, activation: Activation) -> Result<()> {
    match activation {
        Activation::Silu | Activation::GeluTanh => Ok(()),
        Activation::Gelu | Activation::Sigmoid => Err(Error::shape(
            owner,
            format!("a grouped expert product gates with silu or gelu_tanh, not {activation}"),
        )),
    }
}

/// Which of the `rows` permuted rows of the CPU `maps` hold a route, as a U8 mask.
fn live_rows(maps: &Tensor, rows: usize) -> Result<Tensor> {
    Ok(map(maps, PERMUTED_TO_EXPANDED, rows)?.ne(f64::from(ABSENT))?)
}

/// The expert of each of the `rows` permuted rows of the CPU `maps`, as its index in the
/// stack; zero on the rows that are not `live`.
fn row_experts(maps: &Tensor, live: &Tensor, rows: usize) -> Result<Tensor> {
    let row = Tensor::arange(0_u32, rows as u32, &Device::Cpu)?;
    let tile = row.broadcast_div(&integer(TILE_ROWS, &row)?)?;
    let expert = map(maps, TILE_TO_EXPERT, rows / TILE_ROWS as usize)?.index_select(&tile, 0)?;

    Ok(live.where_cond(&expert, &expert.zeros_like()?)?)
}

/// `product`, `[rows, width]` FP32, with the rows that are not `live` zero.
fn zero_padding(product: &Tensor, live: &Tensor) -> Result<Tensor> {
    Ok(product.broadcast_mul(&live.to_dtype(ACCUMULATE)?.unsqueeze(1)?)?)
}

/// `rows`, `[m, k]`, encoded under `encode_scale` and decoded: the FP32 rows a packed product
/// reads, `code · block scale · row factor`.
fn decoded_rows(rows: &Tensor, encode_scale: &Tensor) -> Result<Tensor> {
    let (codes, scale) = encoded(rows, encode_scale)?;
    let factor = encode_scale.reshape(())?.recip()?;

    Ok(codes
        .broadcast_mul(&scale.unsqueeze(D::Minus1)?)?
        .broadcast_mul(&factor)?
        .reshape(rows.shape())?)
}

/// Where the shuffle of a packed stack stores each row of `rows`, a U32 tensor of row indices:
/// inside its block of thirty-two, row `r` moves to `(r mod 4) · 8 + r / 4`.
fn shuffled(rows: &Tensor) -> Result<Tensor> {
    let within = remainder(rows, 32)?;
    let block = rows.broadcast_sub(&within)?;
    let group = remainder(&within, 4)?.broadcast_mul(&integer(8, rows)?)?;

    Ok(((block + group)? + within.broadcast_div(&integer(4, rows)?)?)?)
}

/// The workspace one packed product of `routes` permuted rows and `width` output columns asks
/// for, in bytes: the package's own sizing of its row-maximum planes and tile bars, each padded
/// to a kibibyte, its scheduler's counter, and the entry's preamble before them.
///
/// The size follows from the bucket's worst-case route count, so every replay of a captured
/// bucket takes the same bytes.
fn workspace_bytes(routes: u64, width: u64) -> u64 {
    let kibibytes = |bytes: u64| bytes.div_ceil(1024) * 1024;
    let token_tile = u64::from(MMA_TILE.0) * u64::from(MMA_CLUSTER.0);
    let padded = routes.max(1).div_ceil(token_tile) * token_tile;

    // One FP32 row maximum per 128 elements of the padded tile grid.
    let row_maxima = width * padded / 128 * 4;
    // One half-word of tile bars per (token tile, width tile) pair.
    let bars = (padded / token_tile) * width.div_ceil(u64::from(MMA_TILE.1)) / 2 * 4;

    ENTRY_PREAMBLE + kibibytes(row_maxima) + kibibytes(bars) + 4
}

/// The descriptor of one launch of the expert exchange in `mode` over `exchange`'s group, for a
/// block of `experts` experts of `intermediate` gated width whose rows are `dtype`.
fn exchange_descriptor(
    mode: Mode,
    exchange: &ExpertExchange,
    (experts, intermediate): (u32, u32),
    dtype: DType,
) -> Result<exchange::Descriptor> {
    let (&[_, hidden], &[_, top]) = (exchange.states.dims(), exchange.ids.dims()) else {
        return Err(Error::shape(
            "expert_exchange",
            format!(
                "the exchange's rows are {:?} with routes of {:?}",
                exchange.states.dims(),
                exchange.ids.dims()
            ),
        ));
    };

    Ok(exchange::Descriptor {
        mode,
        hidden: hidden as u32,
        intermediate,
        top: top as u32,
        experts,
        max_tokens: exchange.rank_rows,
        ranks: exchange.ranks,
        attention: exchange.attention,
        dtype,
    })
}

/// `tokens`, the rows one launch of the exchange sends or takes back, as the launch states
/// them: at most one rank's block.
fn sent(owner: &'static str, tokens: usize, exchange: &ExpertExchange) -> Result<u32> {
    u32::try_from(tokens)
        .ok()
        .filter(|tokens| *tokens <= exchange.rank_rows)
        .ok_or_else(|| {
            Error::shape(
                owner,
                format!(
                    "the batch has {tokens} tokens, the exchange carries {} per rank",
                    exchange.rank_rows
                ),
            )
        })
}

/// The `[tokens, hidden]` rows of `dtype` an exchange writes this rank's sums into: a new
/// tensor, or no rows of the exchange's own for a rank that joined with no tokens.
fn own_rows(tokens: usize, exchange: &ExpertExchange, dtype: DType) -> Result<Tensor> {
    if tokens == 0 {
        return Ok(exchange.combined.narrow(0, 0, 0)?);
    }

    let hidden = exchange.states.dims().last().copied().unwrap_or(0);
    Ok(backend::empty((tokens, hidden), dtype)?)
}

/// The argument of a tensor a launch of the exchange does not read: no elements at no address.
fn unread() -> ops::Operand {
    ops::Operand::new(ops::DevicePtr::new(0), DType::U8, ops::Shape::new(&[0]))
}

/// The route maps of [`moe_sort`] on CPU tensors, for `tiles` tiles of `rows` permuted rows.
fn sort_formula(
    ids: &Tensor,
    local: &Range<u32>,
    live_tokens: Option<&Tensor>,
    (tiles, rows): (usize, usize),
) -> Result<Tensor> {
    let cpu = &Device::Cpu;
    let (tokens, top) = (ids.dims()[0], ids.dims()[1]);
    let count = tokens * top;
    let experts = (local.end - local.start) as usize;
    let tile = f64::from(TILE_ROWS);

    // The arithmetic is F64, which holds every index exactly.
    let route = Tensor::arange(0_f64, count as f64, cpu)?;
    let expert = (ids.flatten_all()?.to_dtype(DType::F64)? - f64::from(local.start))?;

    // A route is live when its token row is one the batch fills, and held when it is live and
    // its expert is one of this rank's.
    let live = match live_tokens {
        Some(live_tokens) => {
            let filled = live_tokens.to_dtype(DType::F64)?.reshape(())?;
            route.broadcast_lt(&filled.affine(top as f64, 0.0)?)?
        }
        None => route.ge(0.0)?,
    };
    let held = ((expert.ge(0.0)? * expert.lt(experts as f64)?)? * &live)?;

    // Each held route's group is its expert; every other route is in one group after them.
    let group = held.where_cond(&expert, &Tensor::full(experts as f64, count, cpu)?)?;
    let group_index = group.to_dtype(DType::U32)?;

    // Per expert: its routes, its whole tiles, and where its tiles and its routes begin.
    let routed = Tensor::zeros(experts + 1, DType::F64, cpu)?
        .scatter_add(&group_index, &Tensor::ones(count, DType::F64, cpu)?, 0)?
        .narrow(0, 0, experts)?;
    let group_tiles = (&routed / tile)?.ceil()?;
    let last_tile = group_tiles.cumsum(0)?;
    let first_tile = (&last_tile - &group_tiles)?;
    let first_row = (&first_tile * tile)?;
    let before = (routed.cumsum(0)? - &routed)?;

    // The routes in permuted order: by group, and by route within one. The sorted position of a
    // held route, less the held routes of the experts before its own, is its place in its
    // expert's rows.
    let order = (group.affine(count as f64, 0.0)? + &route)?.arg_sort_last_dim(true)?;
    let sorted_group = group_index.index_select(&order, 0)?;
    let place = (&route
        - before
            .pad_with_zeros(0, 0, 1)?
            .index_select(&sorted_group, 0)?)?;
    let row = (first_row
        .pad_with_zeros(0, 0, 1)?
        .index_select(&sorted_group, 0)?
        + place)?;

    // A route that is not held is given the row one past the last, which no map keeps.
    let row = held
        .index_select(&order, 0)?
        .where_cond(&row, &Tensor::full(rows as f64, count, cpu)?)?
        .to_dtype(DType::U32)?;

    let permuted_to_expanded = Tensor::full(ABSENT, rows + 1, cpu)?
        .scatter(&row, &order, 0)?
        .narrow(0, 0, rows)?;
    let padding = permuted_to_expanded.eq(f64::from(ABSENT))?;
    let permuted_to_token = padding.where_cond(
        &permuted_to_expanded,
        &permuted_to_expanded.broadcast_div(&integer(top as u32, &permuted_to_expanded)?)?,
    )?;

    let without_row = live.where_cond(
        &Tensor::full(ABSENT, count, cpu)?,
        &Tensor::full(NOT_LIVE, count, cpu)?,
    )?;
    let expanded_to_permuted = held.where_cond(
        &Tensor::zeros(count, DType::U32, cpu)?.scatter(&order, &row, 0)?,
        &without_row,
    )?;

    // A tile belongs to the expert whose tiles span it, and its limit is the end of its own
    // rows or of its expert's routes, whichever comes first.
    let index = Tensor::arange(0_f64, tiles as f64, cpu)?.unsqueeze(1)?;
    let spans = (index.broadcast_ge(&first_tile.unsqueeze(0)?)?
        * index.broadcast_lt(&last_tile.unsqueeze(0)?)?)?
    .to_dtype(DType::F64)?;
    let window = Tensor::arange(0_f64, experts as f64, cpu)?;
    let tile_to_expert = spans.broadcast_mul(&window.unsqueeze(0)?)?.sum(1)?;
    let routes_end = spans
        .broadcast_mul(&(first_row + routed)?.unsqueeze(0)?)?
        .sum(1)?;
    let tile_to_limit = index.squeeze(1)?.affine(tile, tile)?.minimum(&routes_end)?;

    let live_tiles = group_tiles.sum_all()?.reshape(1)?;
    let padded_rows = (&live_tiles * tile)?;

    // The maps, each in the leading entries of its row.
    let entries = |map: Tensor| -> Result<Tensor> {
        let map = map.to_dtype(DType::U32)?;
        let unused = rows - map.elem_count();

        Ok(map.pad_with_zeros(0, 0, unused)?)
    };
    Ok(Tensor::stack(
        &[
            entries(tile_to_expert)?,
            entries(tile_to_limit)?,
            entries(expanded_to_permuted)?,
            entries(permuted_to_expanded)?,
            entries(permuted_to_token)?,
            entries(padded_rows)?,
            entries(live_tiles)?,
        ],
        0,
    )?)
}

#[cfg(test)]
mod tests {
    use ops::fp4::ScaleLayout;

    use super::{
        ACTIVATION_SCALE_LAYOUT, GATED_TOKEN_SCALE_LAYOUT, WEIGHT_SCALE_LAYOUT, workspace_bytes,
    };

    /// The block-scale layouts each packed product's kernels are built for, held against the
    /// packages.
    ///
    /// A product's kernels are compiled with one layout for each operand, and nothing compares
    /// the two at a launch: a package built for one form and given a table in the other reads
    /// every block scale at the wrong offset and computes plausible numbers. The packages name
    /// the weights' table `mSfLayoutA` and the tokens' `mSfLayoutB`, and the two products
    /// differ in the second alone.
    #[test]
    fn a_packed_product_s_package_states_the_scale_layouts_its_function_passes() {
        let name = |layout: ScaleLayout| match layout {
            ScaleLayout::Linear => "\"linear\"",
            ScaleLayout::Swizzled128X4 => "\"swizzled_128x4\"",
        };

        for (operator, tokens) in [
            ("gather_gemm_nvfp4", GATED_TOKEN_SCALE_LAYOUT),
            ("route_gemm_nvfp4", ACTIVATION_SCALE_LAYOUT),
        ] {
            let package = std::path::Path::new(ops::KERNELS)
                .join(operator)
                .join("trtllm/package.toml");
            let stated = std::fs::read_to_string(&package)
                .unwrap_or_else(|error| panic!("{}: {error}", package.display()));

            for (option, layout) in [("mSfLayoutA", WEIGHT_SCALE_LAYOUT), ("mSfLayoutB", tokens)] {
                let line = format!("{option} = {}", name(layout));
                assert!(
                    stated.contains(&line),
                    "{operator}'s package does not state `{line}`"
                );
            }
        }
    }

    /// The packed products' workspaces follow the package's arithmetic over the bucket's route
    /// count: 15 tokens over 128 experts are 120 tiles of 128 permuted rows, for which the
    /// gated product (1536 columns) asks 16 + 740 356 bytes and the second (2048 columns)
    /// 16 + 987 140; a bucket of a single tile still carries the entry's preamble.
    #[test]
    fn a_packed_product_s_workspace_is_sized_from_its_route_count() {
        assert_eq!(workspace_bytes(15_360, 1536), 740_372);
        assert_eq!(workspace_bytes(15_360, 2048), 987_156);
        assert_eq!(workspace_bytes(8, 1536), 7_188);
        assert_eq!(workspace_bytes(8, 2048), 9_236);

        // The size grows with the routes and stays far below a fixed floor of mebibytes.
        assert!(workspace_bytes(4096, 2048) > workspace_bytes(8, 2048));
        assert!(workspace_bytes(4096, 2048) < 16 << 20);
    }
}
