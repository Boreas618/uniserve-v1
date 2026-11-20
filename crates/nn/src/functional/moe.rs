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
//! groups the routes of one expert into **tiles** of [`ops::moe::TILE`]'s permuted rows and states the
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

use ops::fp4::{BLOCK_ELEMENTS, ValueLayout};
use ops::moe::ExchangeArgs;
use ops::moe::TILE;
use ops::moe::combine::{self, MoeUnpermute, UnpermuteArgs};
use ops::moe::exchange::{self, ExpertWeights, FusedExperts, FusedLayer, Mode};
use ops::moe::grouped::{
    GatherGemm, GatherGemmArgs, GatherGemmDescriptor, GatherGemmNvfp4, GatherGemmNvfp4Args,
    GatherGemmNvfp4Descriptor, RouteGemm, RouteGemmArgs, RouteGemmDescriptor, RouteGemmNvfp4,
    RouteGemmNvfp4Args, RouteGemmNvfp4Descriptor,
};
use ops::moe::mxfp4::{
    GATED_SCALE_LAYOUT, GatherGemmMxfp4, GatherGemmMxfp4Args, GatherGemmMxfp4Descriptor,
    RouteGemmMxfp4, RouteGemmMxfp4Args, RouteGemmMxfp4Descriptor, TOKEN_SCALE_LAYOUT,
    WEIGHT_SCALE_LAYOUT,
};
use ops::moe::routing::{Scoring, TopkArgs, TopkDescriptor, TopkSoftmax};
use ops::moe::sort::{self, MoeSort, SortArgs};
use ops::mx::{ScaleLayout, ScaleRule};
use ops::residual::{SigmoidGatedAdd, SigmoidGatedAddArgs, SigmoidGatedAddDescriptor};
use tensor::{D, DType, Device, Tensor};

use super::activation::activate;
use super::mx::{decoded_mxfp4, decoded_mxfp8, encoded_rows, mxfp8_quantize};
use super::nvfp4::{decoded, encoded, integer, launch_quantize};
use super::{ACCUMULATE, Activation, Fused, as_rows, columns, compute, same_rows};
use crate::capability::ExpertExchange;
use crate::error::{Error, Result};
use crate::quantization::{Mxfp4Stack, Mxfp8Weight, Nvfp4Stack};

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

/// How a router routes a token: its scores, which experts it selects and how their weights are
/// formed ([`topk_softmax`]).
#[derive(Clone, Copy, Debug)]
pub struct Routing<'t> {
    /// Experts each token routes to.
    pub top: u32,
    /// How the logits become scores.
    pub scoring: Scoring,
    /// Whether the selected scores are divided by their sum.
    pub renormalize: bool,
    /// `[experts]` F32: added to the scores the selection is made by, and to no weight.
    pub bias: Option<&'t Tensor>,
    /// `[experts]` in a floating type: each selected expert's factor of its weight (a routed
    /// scaling factor is this, the same for every expert).
    pub scale: Option<&'t Tensor>,
}

impl Routing<'_> {
    /// A softmax router of `top` routes, renormalized when `renormalize`, with no bias or scale.
    pub const fn softmax(top: u32, renormalize: bool) -> Self {
        Self {
            top,
            scoring: Scoring::Softmax,
            renormalize,
            bias: None,
            scale: None,
        }
    }
}

/// The router's choice: `(ids, weights)`, the `top` experts of each token and their weights,
/// `[tokens, top]` U32 and FP32.
///
/// ```text
/// s        = softmax(logits)                         over the experts, in FP32   (Softmax)
///          = sqrt(softplus(logits))                  per expert, in FP32         (SqrtSoftplus)
/// ids      = the top largest s + bias, descending; equal ones in ascending expert order
/// weights  = s[ids] / max(Σ s[ids], FLT_EPSILON)     when `renormalize`          (Softmax)
///          = s[ids] / (Σ s[ids] + 1e-20)             when `renormalize`          (SqrtSoftplus)
///          · scale[ids]                              when a scale is given
/// ids      = map[ids]                                when a map is given
/// ```
///
/// `softplus(x)` is `ln(1 + eˣ)`, and `x` itself above 20 (PyTorch's threshold). `scores` is the
/// router's `[tokens, experts]` logits; `routing` states the scoring, the bias (selection only)
/// and the scale; `map`, when given, the `[experts]` U32 weight row of each expert, for a block
/// whose experts are not stacked in their own order. The selection is over the experts' own ids,
/// so which experts a token picks, and the order of equal keys, do not depend on the map.
///
/// `load`, when given, is the `[experts]` I64 counters of an expert load
/// ([`ops::moe::ExpertLoad`]), device memory execution lends: each route of a live token adds one
/// to its expert's own entry, in place, whatever slot the map names. `live_tokens` is the `[1]`
/// U32 count of leading token rows the batch fills; without it every row is live.
///
/// ```text
/// load[e] += |{ (t, k) : t < live, ids_before_map[t, k] = e }|
/// ```
///
/// # Errors
///
/// When `scores` is not a matrix, `top` is not between one and the expert count, the bias, the
/// scale, the map or the load is not one entry per expert, the bias is not F32, a load is given
/// for CPU scores (the counters are device memory a launch adds into), or the operator refuses
/// its arguments.
pub fn topk_softmax(
    scores: &Tensor,
    routing: &Routing<'_>,
    map: Option<&Tensor>,
    live_tokens: Option<&Tensor>,
    load: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = TopkSoftmaxCall::OWNER;

    let &[tokens, experts] = scores.dims() else {
        return Err(Error::shape(
            OWNER,
            format!("scores of {:?} are not [tokens, experts]", scores.dims()),
        ));
    };
    let top = routing.top;
    if top == 0 || top as usize > experts {
        return Err(Error::shape(
            OWNER,
            format!("a top of {top} is outside 1..={experts}"),
        ));
    }
    for (name, table) in [
        ("bias", routing.bias),
        ("scale", routing.scale),
        ("map", map),
        ("load", load),
    ] {
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
    if let Some(bias) = routing.bias
        && bias.dtype() != DType::F32
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "a selection bias is F32, not {}",
                ops::abi::name(bias.dtype())
            ),
        ));
    }
    let kept = top as usize;

    compute(TopkSoftmaxCall {
        scores,
        routing: *routing,
        map,
        live_tokens,
        load,
        tokens,
        experts,
        kept,
    })
}

/// A checked call of [`topk_softmax`]: `kept` is its routes a token.
struct TopkSoftmaxCall<'t> {
    scores: &'t Tensor,
    routing: Routing<'t>,
    map: Option<&'t Tensor>,
    live_tokens: Option<&'t Tensor>,
    load: Option<&'t Tensor>,
    tokens: usize,
    experts: usize,
    kept: usize,
}

impl TopkSoftmaxCall<'_> {
    const OWNER: &'static str = "topk_softmax";
}

/// PyTorch's softplus threshold: above it `softplus(x)` is `x` itself.
const SOFTPLUS_THRESHOLD: f32 = 20.0;

/// What the reference sqrt-softplus router adds to the selected scores' sum before dividing.
const RENORMALIZE_EPSILON: f64 = 1e-20;

impl Fused for TopkSoftmaxCall<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.scores.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            scores,
            routing,
            map,
            load,
            tokens,
            experts,
            kept,
            ..
        } = self;
        if load.is_some() {
            return Err(Error::Unsupported(
                "topk_softmax: an expert load is device memory a launch adds into, and the CPU \
                 formula keeps none"
                    .to_owned(),
            ));
        }
        let logits = scores.to_dtype(ACCUMULATE)?;
        let probabilities = match routing.scoring {
            Scoring::Softmax => {
                let shifted = logits
                    .broadcast_sub(&logits.max_keepdim(D::Minus1)?)?
                    .exp()?;
                shifted.broadcast_div(&shifted.sum_keepdim(D::Minus1)?)?
            }
            Scoring::SqrtSoftplus => {
                let scored: Vec<f32> = logits
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(|logit| {
                        let softplus = if logit > SOFTPLUS_THRESHOLD {
                            logit
                        } else {
                            logit.exp().ln_1p()
                        };
                        softplus.sqrt()
                    })
                    .collect();
                Tensor::from_vec(scored, (tokens, experts), &Device::Cpu)?
            }
        };
        let keys = match routing.bias {
            Some(bias) => probabilities.broadcast_add(&bias.unsqueeze(0)?)?,
            None => probabilities.clone(),
        };

        // Descending, with equal keys in ascending expert order: the positions of one key are
        // numbered as a group, and each group is ordered by expert.
        let descending = keys.arg_sort_last_dim(false)?;
        let sorted = keys.gather(&descending, D::Minus1)?;
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
        if routing.renormalize {
            let sum = weights.sum_keepdim(D::Minus1)?;
            let total = match routing.scoring {
                Scoring::Softmax => sum.maximum(f64::from(f32::EPSILON))?,
                Scoring::SqrtSoftplus => sum.affine(1.0, RENORMALIZE_EPSILON)?,
            };
            weights = weights.broadcast_div(&total)?;
        }

        let selected = ids.flatten_all()?;
        if let Some(scale) = routing.scale {
            let factors = scale.to_dtype(ACCUMULATE)?.index_select(&selected, 0)?;
            weights = (weights * factors.reshape((tokens, kept))?)?;
        }
        let ids = match map {
            Some(map) => map.index_select(&selected, 0)?.reshape((tokens, kept))?,
            None => ids,
        };
        Ok((ids, weights))
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            scores,
            routing,
            map,
            live_tokens,
            load,
            tokens,
            experts,
            kept,
        } = self;
        let ids = ops::empty((tokens, kept), DType::U32)?;
        let weights = ops::empty((tokens, kept), DType::F32)?;

        ops::launch::<TopkSoftmax>(
            &TopkDescriptor {
                scores: scores.dtype(),
                experts: experts as u32,
                top: routing.top,
                scoring: routing.scoring,
                renormalize: routing.renormalize,
                bias: routing.bias.is_some(),
                scale: routing.scale.map(Tensor::dtype),
            },
            TopkArgs {
                scores: ops::operand(scores)?,
                scale: routing.scale.map(ops::operand).transpose()?,
                bias: routing.bias.map(ops::operand).transpose()?,
                ids: ops::operand(&ids)?,
                weights: ops::operand(&weights)?,
                map: map.map(ops::operand).transpose()?,
                live: live_tokens.map(ops::operand).transpose()?,
                load: load.map(ops::operand).transpose()?,
            },
        )?;
        Ok((ids, weights))
    }
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
/// route takes no tile. There are `max_tiles` tiles of [`ops::moe::TILE`]'s permuted rows for the worst routing
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
    const OWNER: &str = MoeSortCall::OWNER;

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
        u64::from(TILE.0),
    ) as usize;
    let rows = tiles * TILE.0 as usize;

    compute(MoeSortCall {
        ids,
        weights,
        experts,
        local,
        live_tokens,
        tokens,
        top,
        local_experts,
        tiles,
        rows,
    })
}

/// A checked call of [`moe_sort`]: the maps hold `tiles` tiles of `rows` permuted rows in all.
struct MoeSortCall<'t> {
    ids: &'t Tensor,
    weights: &'t Tensor,
    experts: u32,
    local: Range<u32>,
    live_tokens: Option<&'t Tensor>,
    tokens: usize,
    top: usize,
    local_experts: u32,
    tiles: usize,
    rows: usize,
}

impl MoeSortCall<'_> {
    const OWNER: &'static str = "moe_sort";
}

impl Fused for MoeSortCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.ids.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            ids,
            local,
            live_tokens,
            tiles,
            rows,
            ..
        } = self;
        sort_formula(ids, &local, live_tokens, (tiles, rows))
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            ids,
            weights,
            experts,
            local,
            live_tokens,
            tokens,
            top,
            local_experts,
            tiles,
            rows,
        } = self;
        let maps = ops::empty((SORT_MAPS, rows), DType::U32)?;
        let operand = |row: usize, entries: usize| -> Result<ops::Operand> {
            map_operand(&maps, row, entries)
        };

        ops::launch::<MoeSort>(
            &sort::Descriptor::new(experts, top as u32, local_experts, weights.dtype()),
            SortArgs {
                ids: ops::operand(ids)?,
                scales: ops::operand(weights)?,
                tile_to_expert: operand(TILE_TO_EXPERT, tiles)?,
                tile_to_limit: operand(TILE_TO_LIMIT, tiles)?,
                expanded_to_permuted: ops::operand(&route_rows(&maps, tokens, top)?)?,
                permuted_to_expanded: operand(PERMUTED_TO_EXPANDED, rows)?,
                permuted_to_token: operand(PERMUTED_TO_TOKEN, rows)?,
                total_padded_tokens: operand(TOTAL_PADDED_TOKENS, 1)?,
                num_non_exiting_tiles: operand(NUM_NON_EXITING_TILES, 1)?,
                live_tokens: live_tokens.map(ops::operand).transpose()?,
                local_offset: local.start,
            },
        )?;
        Ok(maps)
    }
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
    const OWNER: &str = GatherGemmCall::OWNER;

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

    compute(GatherGemmCall {
        hidden,
        up_gate,
        ids,
        maps,
        activation,
        width,
        tokens,
        top,
        experts,
        tiles,
        rows,
        intermediate,
    })
}

/// A checked call of [`gather_gemm`]: the extents are its stack's and its maps'.
struct GatherGemmCall<'t> {
    hidden: &'t Tensor,
    up_gate: &'t Tensor,
    ids: &'t Tensor,
    maps: &'t Tensor,
    activation: Activation,
    width: u32,
    tokens: usize,
    top: usize,
    experts: usize,
    tiles: usize,
    rows: usize,
    intermediate: usize,
}

impl GatherGemmCall<'_> {
    const OWNER: &'static str = "gather_gemm";
}

impl Fused for GatherGemmCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.hidden.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            hidden,
            up_gate,
            maps,
            activation,
            rows,
            intermediate,
            ..
        } = self;
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

        Ok(zero_padding(&product, &live)?.to_dtype(hidden.dtype())?)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            hidden,
            up_gate,
            ids,
            maps,
            activation,
            width,
            tokens,
            top,
            experts,
            tiles,
            rows,
            intermediate,
        } = self;
        let out = ops::empty((rows, intermediate), hidden.dtype())?;
        ops::launch::<GatherGemm>(
            &GatherGemmDescriptor::new(
                width,
                intermediate as u32,
                experts as u32,
                top as u32,
                activation,
                hidden.dtype(),
            ),
            GatherGemmArgs {
                hidden: ops::operand(hidden)?,
                weights: ops::operand(up_gate)?,
                ids: ops::operand(ids)?,
                // The combine's lookup, not this product's: the product writes row `p` at the
                // sort's permuted row `p`.
                expanded_to_permuted: ops::operand(&route_rows(maps, tokens, top)?)?,
                tile_to_expert: map_operand(maps, TILE_TO_EXPERT, tiles)?,
                tile_to_limit: map_operand(maps, TILE_TO_LIMIT, tiles)?,
                permuted_to_expanded: map_operand(maps, PERMUTED_TO_EXPANDED, rows)?,
                num_non_exiting_tiles: map_operand(maps, NUM_NON_EXITING_TILES, 1)?,
                out: ops::operand(&out)?,
            },
        )?;
        Ok(out)
    }
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
    const OWNER: &str = RouteGemmCall::OWNER;

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

    compute(RouteGemmCall {
        gated,
        down,
        ids,
        maps,
        intermediate,
        top,
        tiles,
        rows,
        experts,
        width,
    })
}

/// A checked call of [`route_gemm`]: the extents are its stack's and its maps'.
struct RouteGemmCall<'t> {
    gated: &'t Tensor,
    down: &'t Tensor,
    ids: &'t Tensor,
    maps: &'t Tensor,
    intermediate: u32,
    top: usize,
    tiles: usize,
    rows: usize,
    experts: usize,
    width: usize,
}

impl RouteGemmCall<'_> {
    const OWNER: &'static str = "route_gemm";
}

impl Fused for RouteGemmCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.gated.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            gated,
            down,
            maps,
            rows,
            ..
        } = self;
        let live = live_rows(maps, rows)?;
        let expert = row_experts(maps, &live, rows)?;

        let weight = down.to_dtype(ACCUMULATE)?.index_select(&expert, 0)?;
        let product = weight
            .matmul(&gated.to_dtype(ACCUMULATE)?.unsqueeze(2)?)?
            .squeeze(2)?;

        Ok(zero_padding(&product, &live)?.to_dtype(gated.dtype())?)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            gated,
            down,
            ids,
            maps,
            intermediate,
            top,
            tiles,
            rows,
            experts,
            width,
            ..
        } = self;
        let out = ops::empty((rows, width), gated.dtype())?;
        ops::launch::<RouteGemm>(
            &RouteGemmDescriptor::new(
                width as u32,
                intermediate,
                experts as u32,
                top as u32,
                gated.dtype(),
            ),
            RouteGemmArgs {
                rows: ops::operand(gated)?,
                weights: ops::operand(down)?,
                ids: ops::operand(ids)?,
                tile_to_expert: map_operand(maps, TILE_TO_EXPERT, tiles)?,
                tile_to_limit: map_operand(maps, TILE_TO_LIMIT, tiles)?,
                permuted_to_expanded: map_operand(maps, PERMUTED_TO_EXPANDED, rows)?,
                num_non_exiting_tiles: map_operand(maps, NUM_NON_EXITING_TILES, 1)?,
                out: ops::operand(&out)?,
            },
        )?;
        Ok(out)
    }
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
    const OWNER: &str = GatherGemmNvfp4Call::OWNER;

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

    compute(GatherGemmNvfp4Call {
        hidden,
        up_gate,
        maps,
        activation,
        top,
        width,
        experts,
        gated,
        packed,
        tokens,
        tiles,
        rows,
        intermediate,
    })
}

/// A checked call of [`gather_gemm_nvfp4`]: the extents are its stack's and its maps'.
struct GatherGemmNvfp4Call<'t> {
    hidden: &'t Tensor,
    up_gate: &'t Nvfp4Stack,
    maps: &'t Tensor,
    activation: Activation,
    top: u32,
    width: u32,
    experts: usize,
    gated: usize,
    packed: usize,
    tokens: usize,
    tiles: usize,
    rows: usize,
    intermediate: usize,
}

impl GatherGemmNvfp4Call<'_> {
    const OWNER: &'static str = "gather_gemm_nvfp4";
}

impl Fused for GatherGemmNvfp4Call<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.hidden.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            hidden,
            up_gate,
            maps,
            activation,
            gated,
            rows,
            intermediate,
            ..
        } = self;
        let live = live_rows(maps, rows)?;
        let expert = row_experts(maps, &live, rows)?;
        let token = map(maps, PERMUTED_TO_TOKEN, rows)?;
        let token = live.where_cond(&token, &token.zeros_like()?)?;

        let x = decoded_rows(hidden, &up_gate.encode_scale)?.index_select(&token, 0)?;

        // The stack's rows in their own order, read back out of the arrangement the product
        // declares.
        let stored = decoded(&up_gate.values, &up_gate.scales, up_gate.layout)?;
        let order = stored_order(GatherGemmNvfp4Descriptor::WEIGHT_VALUES, gated)?;
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

        Ok(zero_padding(&product, &live)?.to_dtype(hidden.dtype())?)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            hidden,
            up_gate,
            maps,
            activation,
            top,
            width,
            experts,
            packed,
            tokens,
            tiles,
            rows,
            intermediate,
            ..
        } = self;
        // 1. The hidden rows' encoding, in the layout the gated product reads its tokens in. The
        //    weights' scale type is the checkpoint's.
        let descriptor = GatherGemmNvfp4Descriptor::new(
            width,
            intermediate as u32,
            experts as u32,
            top,
            activation,
            hidden.dtype(),
            up_gate.scales.dtype(),
        );
        let values = ops::empty((tokens, packed), DType::U8)?;
        let blocks = descriptor.layout.shape(tokens as u64, u64::from(width));
        let scales = ops::empty((blocks[0] as usize, blocks[1] as usize), DType::U8)?;
        let token_scale = ops::empty(tokens, DType::F32)?;
        launch_quantize(
            hidden,
            &up_gate.encode_scale,
            descriptor.layout,
            (
                ops::operand(&values)?,
                ops::operand(&scales)?,
                ops::operand(&token_scale)?,
            ),
        )?;

        // 2. The product, token-major: it gathers each permuted row's token itself.
        let out = ops::empty((rows, intermediate), hidden.dtype())?;
        let workspace = ops::empty(descriptor.workspace_bytes(rows as u64) as usize, DType::U8)?;
        ops::launch::<GatherGemmNvfp4>(
            &descriptor,
            GatherGemmNvfp4Args {
                hidden: ops::operand(&values)?,
                hidden_scale: ops::operand(&scales)?,
                hidden_token_scale: ops::operand(&token_scale)?,
                weights: ops::operand(&up_gate.values)?,
                scales: ops::operand(&up_gate.scales)?,
                global_scale: ops::operand(&up_gate.global_scale)?,
                tile_to_expert: map_operand(maps, TILE_TO_EXPERT, tiles)?,
                tile_to_limit: map_operand(maps, TILE_TO_LIMIT, tiles)?,
                permuted_to_token: map_operand(maps, PERMUTED_TO_TOKEN, rows)?,
                num_non_exiting_tiles: map_operand(maps, NUM_NON_EXITING_TILES, 1)?,
                out: ops::operand(&out)?,
                workspace: ops::operand(&workspace)?,
            },
        )?;
        Ok(out)
    }
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
    const OWNER: &str = RouteGemmNvfp4Call::OWNER;

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

    compute(RouteGemmNvfp4Call {
        gated,
        down,
        maps,
        top,
        intermediate,
        experts,
        width,
        packed,
        rows,
        tiles,
    })
}

/// A checked call of [`route_gemm_nvfp4`]: the extents are its stack's and its maps'.
struct RouteGemmNvfp4Call<'t> {
    gated: &'t Tensor,
    down: &'t Nvfp4Stack,
    maps: &'t Tensor,
    top: u32,
    intermediate: u32,
    experts: usize,
    width: usize,
    packed: usize,
    rows: usize,
    tiles: usize,
}

impl RouteGemmNvfp4Call<'_> {
    const OWNER: &'static str = "route_gemm_nvfp4";
}

impl Fused for RouteGemmNvfp4Call<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.gated.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            gated,
            down,
            maps,
            width,
            rows,
            ..
        } = self;
        let live = live_rows(maps, rows)?;
        let expert = row_experts(maps, &live, rows)?;

        let x = decoded_rows(gated, &down.encode_scale)?;
        let stored = decoded(&down.values, &down.scales, down.layout)?;
        let order = stored_order(RouteGemmNvfp4Descriptor::WEIGHT_VALUES, width)?;
        let weight = stored.index_select(&order, 1)?.index_select(&expert, 0)?;

        let scales = down.global_scale.index_select(&expert, 1)?.get(0)?;
        let product = weight
            .matmul(&x.unsqueeze(2)?)?
            .squeeze(2)?
            .broadcast_mul(&scales.unsqueeze(1)?)?;

        Ok(zero_padding(&product, &live)?.to_dtype(gated.dtype())?)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            gated,
            down,
            maps,
            top,
            intermediate,
            experts,
            width,
            packed,
            rows,
            tiles,
        } = self;
        // 1. The gated rows' encoding, in the layout the product reads them in: the first product's
        //    output is a 16-bit row, so this is an encoding of its own and not a wider operand of
        //    the second product.
        let descriptor = RouteGemmNvfp4Descriptor::new(
            width as u32,
            intermediate,
            experts as u32,
            top,
            gated.dtype(),
            down.scales.dtype(),
        );
        let values = ops::empty((rows, packed), DType::U8)?;
        let blocks = descriptor
            .layout
            .shape(rows as u64, u64::from(intermediate));
        let scales = ops::empty((blocks[0] as usize, blocks[1] as usize), DType::U8)?;
        let token_scale = ops::empty(rows, DType::F32)?;
        launch_quantize(
            gated,
            &down.encode_scale,
            descriptor.layout,
            (
                ops::operand(&values)?,
                ops::operand(&scales)?,
                ops::operand(&token_scale)?,
            ),
        )?;

        // 2. The product: one row per permuted row, the `top` routes of a token still separate.
        let out = ops::empty((rows, width), gated.dtype())?;
        let workspace = ops::empty(descriptor.workspace_bytes(rows as u64) as usize, DType::U8)?;
        ops::launch::<RouteGemmNvfp4>(
            &descriptor,
            RouteGemmNvfp4Args {
                hidden: ops::operand(&values)?,
                hidden_scale: ops::operand(&scales)?,
                hidden_token_scale: ops::operand(&token_scale)?,
                weights: ops::operand(&down.values)?,
                scales: ops::operand(&down.scales)?,
                global_scale: ops::operand(&down.global_scale)?,
                tile_to_expert: map_operand(maps, TILE_TO_EXPERT, tiles)?,
                tile_to_limit: map_operand(maps, TILE_TO_LIMIT, tiles)?,
                permuted_to_expanded: map_operand(maps, PERMUTED_TO_EXPANDED, rows)?,
                num_non_exiting_tiles: map_operand(maps, NUM_NON_EXITING_TILES, 1)?,
                out: ops::operand(&out)?,
                workspace: ops::operand(&workspace)?,
            },
        )?;
        Ok(out)
    }
}

/// The first product of the routed experts over an MXFP4 `[up; gate]` stack: the hidden rows
/// are encoded into MXFP8, each permuted row's gated product is taken over the two encodings with
/// the clamped SwiGLU, and the gated rows are encoded into MXFP8 again.
///
/// ```text
/// x̂[t]    = mxfp8_quantize(hidden[t])                          per 32 elements, covering scales
/// up      = Ŵ_up,e · x̂[t]       gate = Ŵ_gate,e · x̂[t]        over the decoded values, in FP32
/// g[p]    = silu(min(gate, L_e)) · clamp(up, −L_e, L_e)
/// out[p]  = mxfp8(g[p], ScaleRule::Ocp)                        codes, scales in the 128 × 4 atom
/// ```
///
/// The gated rows are encoded where the device's product encodes them, in its epilogue, and by
/// its rule: the OCP Microscaling scale `2^(⌊log₂ amax⌋ − 8)`, under which a block whose largest
/// magnitude has a significand above 1.75 saturates its largest quotients at ±448
/// ([`ScaleRule::Ocp`]). The reference encodes the down projection's input by the covering rule,
/// so such a block's largest elements differ from it by up to an eighth.
///
/// `hidden` is `[tokens, hidden]` BF16; `up_gate` holds the experts of the sort's window, its rows
/// interleaved and shuffled and its scales swizzled as the product reads them; `limit` is the
/// `[experts]` F32 clamp `L_e` of each expert of the stack; `top` is the routes of a token. The
/// result is `(codes, scales)`: `[permuted rows, intermediate]` E4M3 and their UE8M0 scales in
/// [`GATED_SCALE_LAYOUT`], which [`route_gemm_mxfp4`] reads.
///
/// The CPU formula encodes the hidden rows as [`mxfp8_quantize`](super::mxfp8_quantize)'s formula
/// does and the gated rows by the OCP rule, evaluates SiLU exactly and writes the encoding of
/// zeros on padding rows; the device's product leaves them unspecified. A device check compares the decoded live rows under the operator's tolerance, which
/// covers a code one step apart.
///
/// # Errors
///
/// When the stack is not `[experts, 2 · intermediate, hidden / 2]` over `hidden`'s rows, the
/// limits are not one F32 per expert, the maps are not a sort's, or an operator refuses its
/// arguments.
pub fn gather_gemm_mxfp4(
    hidden: &Tensor,
    up_gate: &Mxfp4Stack,
    limit: &Tensor,
    maps: &Tensor,
    top: u32,
) -> Result<(Tensor, Tensor)> {
    const OWNER: &str = GatherGemmMxfp4Call::OWNER;

    let width = columns(OWNER, hidden)?;
    let &[experts, gated, packed] = up_gate.codes.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "an MXFP4 [up; gate] stack is [experts, 2 · intermediate, hidden / 2], not {:?}",
                up_gate.codes.dims()
            ),
        ));
    };
    if packed * 2 != width as usize
        || gated % 2 != 0
        || hidden.rank() != 2
        || top == 0
        || up_gate.values != GatherGemmMxfp4Descriptor::WEIGHT_VALUES
        || up_gate.layout != WEIGHT_SCALE_LAYOUT
        || limit.dims() != [experts]
        || limit.dtype() != DType::F32
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "an MXFP4 stack of {:?} ({}, {} scales) over rows of {:?} with {top} routes a \
                 token and limits of {:?}",
                up_gate.codes.dims(),
                up_gate.values,
                up_gate.layout,
                hidden.dims(),
                limit.dims()
            ),
        ));
    }
    let tokens = hidden.dims()[0];
    let (tiles, rows) = sorted(OWNER, maps, tokens * top as usize)?;

    compute(GatherGemmMxfp4Call {
        hidden,
        up_gate,
        limit,
        maps,
        top,
        width,
        experts,
        tiles,
        rows,
        intermediate: gated / 2,
    })
}

/// A checked call of [`gather_gemm_mxfp4`].
struct GatherGemmMxfp4Call<'t> {
    hidden: &'t Tensor,
    up_gate: &'t Mxfp4Stack,
    limit: &'t Tensor,
    maps: &'t Tensor,
    top: u32,
    width: u32,
    experts: usize,
    tiles: usize,
    rows: usize,
    intermediate: usize,
}

impl GatherGemmMxfp4Call<'_> {
    const OWNER: &'static str = "gather_gemm_mxfp4";
}

impl Fused for GatherGemmMxfp4Call<'_> {
    type Output = (Tensor, Tensor);

    fn device(&self) -> &Device {
        self.hidden.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            hidden,
            up_gate,
            limit,
            maps,
            rows,
            intermediate,
            ..
        } = self;
        let live = live_rows(maps, rows)?;
        let expert = row_experts(maps, &live, rows)?;
        let token = map(maps, PERMUTED_TO_TOKEN, rows)?;
        let token = live.where_cond(&token, &token.zeros_like()?)?;

        let (encoded, scales) = mxfp8_quantize(hidden, ScaleLayout::Linear)?;
        let x = decoded_mxfp8(&encoded, &scales, ScaleLayout::Linear)?.index_select(&token, 0)?;

        // The stack's rows in their own order, read back out of the arrangement the product
        // declares, one expert at a time.
        let order = stored_order(up_gate.values, 2 * intermediate)?;
        let projected = expert_products(up_gate, &order, &x, (&live, &expert))?;

        // DeepSeek's clamped SwiGLU, each row at its expert's limit.
        let bound = limit.index_select(&expert, 0)?.unsqueeze(1)?;
        let up = projected.narrow(1, 0, intermediate)?;
        let up = up
            .broadcast_maximum(&bound.neg()?)?
            .broadcast_minimum(&bound)?;
        let gate = projected
            .narrow(1, intermediate, intermediate)?
            .broadcast_minimum(&bound)?;
        let product = zero_padding(&(gate.silu()? * up)?, &live)?;

        // The gated rows' MXFP8 encoding by the OCP rule the device's epilogue applies, its scales
        // in the atom the second product reads.
        let (quotients, linear) = encoded_rows(&product, ScaleRule::Ocp)?;
        let codes = Tensor::from_vec(quotients, (rows, intermediate), &Device::Cpu)?
            .to_dtype(DType::F8E4M3)?;
        let table = ops::mx::arranged(
            &linear,
            rows as u64,
            intermediate as u64,
            GATED_SCALE_LAYOUT,
        );
        let shape = GATED_SCALE_LAYOUT.shape(rows as u64, intermediate as u64);
        let scales = Tensor::from_raw_buffer(
            &table,
            DType::U8,
            &[shape[0] as usize, shape[1] as usize],
            &Device::Cpu,
        )?;
        Ok((codes, scales))
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            hidden,
            up_gate,
            limit,
            maps,
            top,
            width,
            experts,
            tiles,
            rows,
            intermediate,
        } = self;
        // 1. The hidden rows' MXFP8 encoding, in the layout the gated product gathers it in.
        let (codes, scales) = mxfp8_quantize(hidden, TOKEN_SCALE_LAYOUT)?;

        // 2. The product, token-major: it gathers each permuted row's token itself, and encodes
        //    its gated rows in its epilogue.
        let descriptor =
            GatherGemmMxfp4Descriptor::new(width, intermediate as u32, experts as u32, top);
        let out = ops::empty((rows, intermediate), DType::F8E4M3)?;
        let shape = GATED_SCALE_LAYOUT.shape(rows as u64, intermediate as u64);
        let out_scale = ops::empty((shape[0] as usize, shape[1] as usize), DType::U8)?;
        let workspace = ops::empty(descriptor.workspace_bytes(rows as u64) as usize, DType::U8)?;
        ops::launch::<GatherGemmMxfp4>(
            &descriptor,
            GatherGemmMxfp4Args {
                hidden: ops::operand(&codes)?,
                hidden_scale: ops::operand(&scales)?,
                weights: ops::operand(&up_gate.codes)?,
                scales: ops::operand(&up_gate.scales)?,
                limit: ops::operand(limit)?,
                tile_to_expert: map_operand(maps, TILE_TO_EXPERT, tiles)?,
                tile_to_limit: map_operand(maps, TILE_TO_LIMIT, tiles)?,
                permuted_to_token: map_operand(maps, PERMUTED_TO_TOKEN, rows)?,
                num_non_exiting_tiles: map_operand(maps, NUM_NON_EXITING_TILES, 1)?,
                out: ops::operand(&out)?,
                out_scale: ops::operand(&out_scale)?,
                workspace: ops::operand(&workspace)?,
            },
        )?;
        Ok((out, out_scale))
    }
}

/// The second product of the routed experts over an MXFP4 `down` stack: each permuted row's
/// MXFP8 gated row times its expert's weight.
///
/// ```text
/// out[p] = Ŵ_down,e · ĝ[p]          over the decoded values, in FP32, rounded once into BF16
/// ```
///
/// `codes` and `scales` are the gated rows [`gather_gemm_mxfp4`] wrote; `down` holds the experts
/// of the sort's window, its rows shuffled and its scales swizzled as the product reads them. The
/// result is `[permuted rows, hidden]` BF16, each route's product unweighted, which
/// [`moe_unpermute`] sums; the CPU formula writes zeros on padding rows.
///
/// # Errors
///
/// When the stack is not `[experts, hidden, intermediate / 2]` over the gated rows, the maps are
/// not a sort's of these rows, or the operator refuses its arguments.
pub fn route_gemm_mxfp4(
    codes: &Tensor,
    scales: &Tensor,
    down: &Mxfp4Stack,
    maps: &Tensor,
    top: u32,
) -> Result<Tensor> {
    const OWNER: &str = RouteGemmMxfp4Call::OWNER;

    let intermediate = columns(OWNER, codes)?;
    let &[experts, width, packed] = down.codes.dims() else {
        return Err(Error::shape(
            OWNER,
            format!(
                "an MXFP4 down stack is [experts, hidden, intermediate / 2], not {:?}",
                down.codes.dims()
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
        || codes.dims() != [rows, intermediate as usize]
        || codes.dtype() != DType::F8E4M3
        || top == 0
        || down.values != RouteGemmMxfp4Descriptor::WEIGHT_VALUES
        || down.layout != WEIGHT_SCALE_LAYOUT
    {
        return Err(Error::shape(
            OWNER,
            format!(
                "an MXFP4 stack of {:?} ({}, {} scales) over gated {} rows of {:?} for {rows} \
                 permuted rows",
                down.codes.dims(),
                down.values,
                down.layout,
                ops::abi::name(codes.dtype()),
                codes.dims()
            ),
        ));
    }

    compute(RouteGemmMxfp4Call {
        codes,
        scales,
        down,
        maps,
        top,
        intermediate,
        experts,
        width,
        rows,
        tiles,
    })
}

/// A checked call of [`route_gemm_mxfp4`].
struct RouteGemmMxfp4Call<'t> {
    codes: &'t Tensor,
    scales: &'t Tensor,
    down: &'t Mxfp4Stack,
    maps: &'t Tensor,
    top: u32,
    intermediate: u32,
    experts: usize,
    width: usize,
    rows: usize,
    tiles: usize,
}

impl RouteGemmMxfp4Call<'_> {
    const OWNER: &'static str = "route_gemm_mxfp4";
}

impl Fused for RouteGemmMxfp4Call<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.codes.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            codes,
            scales,
            down,
            maps,
            width,
            rows,
            ..
        } = self;
        let live = live_rows(maps, rows)?;
        let expert = row_experts(maps, &live, rows)?;

        let x = decoded_mxfp8(codes, scales, GATED_SCALE_LAYOUT)?;
        let order = stored_order(down.values, width)?;
        let product = expert_products(down, &order, &x, (&live, &expert))?;

        Ok(zero_padding(&product, &live)?.to_dtype(DType::BF16)?)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            codes,
            scales,
            down,
            maps,
            top,
            intermediate,
            experts,
            width,
            rows,
            tiles,
        } = self;
        let descriptor =
            RouteGemmMxfp4Descriptor::new(width as u32, intermediate, experts as u32, top);
        let out = ops::empty((rows, width), DType::BF16)?;
        let workspace = ops::empty(descriptor.workspace_bytes(rows as u64) as usize, DType::U8)?;
        ops::launch::<RouteGemmMxfp4>(
            &descriptor,
            RouteGemmMxfp4Args {
                hidden: ops::operand(codes)?,
                hidden_scale: ops::operand(scales)?,
                weights: ops::operand(&down.codes)?,
                scales: ops::operand(&down.scales)?,
                tile_to_expert: map_operand(maps, TILE_TO_EXPERT, tiles)?,
                tile_to_limit: map_operand(maps, TILE_TO_LIMIT, tiles)?,
                permuted_to_expanded: map_operand(maps, PERMUTED_TO_EXPANDED, rows)?,
                num_non_exiting_tiles: map_operand(maps, NUM_NON_EXITING_TILES, 1)?,
                out: ops::operand(&out)?,
                workspace: ops::operand(&workspace)?,
            },
        )?;
        Ok(out)
    }
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
    const OWNER: &str = MoeUnpermuteCall::OWNER;

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

    compute(MoeUnpermuteCall {
        products,
        maps,
        weights,
        experts,
        combined,
        width,
        tokens,
        top,
        rows,
    })
}

/// A checked call of [`moe_unpermute`]: the extents are its routes' and its maps'.
struct MoeUnpermuteCall<'t> {
    products: &'t Tensor,
    maps: &'t Tensor,
    weights: &'t Tensor,
    experts: u32,
    combined: Option<&'t Tensor>,
    width: u32,
    tokens: usize,
    top: usize,
    rows: usize,
}

impl MoeUnpermuteCall<'_> {
    const OWNER: &'static str = "moe_unpermute";
}

impl Fused for MoeUnpermuteCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.products.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            products,
            maps,
            weights,
            combined,
            width,
            tokens,
            top,
            rows,
            ..
        } = self;
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

        match combined {
            Some(combined) => {
                combined.slice_set(&sums, 0, 0)?;
                Ok(combined.clone())
            }
            None => Ok(sums),
        }
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            products,
            maps,
            weights,
            experts,
            combined,
            width,
            tokens,
            top,
            ..
        } = self;
        let out = match combined {
            Some(combined) => combined.clone(),
            None => ops::empty((tokens, width as usize), products.dtype())?,
        };
        ops::launch::<MoeUnpermute>(
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
                routes: ops::operand(products)?,
                expanded_to_permuted: ops::operand(&route_rows(maps, tokens, top)?)?,
                scales: ops::operand(weights)?,
                out: ops::operand(&out)?,
            },
        )?;
        Ok(out)
    }
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
    ops::collective::expert_exchange(
        &descriptor,
        ExchangeArgs {
            transport: None,
            tokens: sent(OWNER, hidden.dims()[0], exchange)?,
            capacity: exchange.step_rows,
            state: ops::operand(hidden)?,
            ids: ops::operand(ids)?,
            weights: ops::operand(weights)?,
            live: live_tokens.map(ops::operand).transpose()?,
            out: ops::Operand::unread(),
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

    ops::collective::expert_exchange(
        &descriptor,
        ExchangeArgs {
            transport: None,
            tokens: sent(OWNER, tokens, exchange)?,
            capacity: exchange.step_rows,
            state: ops::Operand::unread(),
            ids: ops::Operand::unread(),
            weights: ops::Operand::unread(),
            live: live_tokens.map(ops::operand).transpose()?,
            out: ops::operand(&out)?,
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
        let token_scale = ops::empty(tokens, DType::F32)?;
        let scales = ops::operand(&exchange.packed_scales.narrow(0, 0, tokens)?)?;

        launch_quantize(
            hidden,
            &up_gate.encode_scale,
            exchange::Descriptor::FUSED_TOKEN_SCALES,
            (
                ops::operand(&exchange.packed.narrow(0, 0, tokens)?)?,
                // The encoder writes each scale's byte, whatever type the transport labels
                // its rows.
                {
                    let mut bytes = scales.clone();
                    bytes.dtype = DType::U8;
                    bytes
                },
                ops::operand(&token_scale)?,
            ),
        )?;
    }

    // The layer's device addresses, as the package's `uniserve_fused_layer` declares them.
    let address = |tensor: &Tensor| -> Result<*const std::ffi::c_void> {
        Ok(ops::operand(tensor)?.ptr.address() as usize as *const std::ffi::c_void)
    };
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
        up_gate_alpha: address(up_gate_alpha)?.cast(),
        down_alpha: address(down_alpha)?.cast(),
        gated_scale: address(gated_scale)?.cast(),
        up_gate_scale_bytes: expert_bytes(&up_gate.scales)?,
        down_scale_bytes: expert_bytes(&down.scales)?,
        shared_up_gate: std::ptr::null(),
        shared_up_gate_scales: std::ptr::null(),
        shared_down: std::ptr::null(),
        shared_down_scales: std::ptr::null(),
    };

    let descriptor = exchange_descriptor(
        Mode::Fused,
        exchange,
        (experts, intermediate as u32),
        hidden.dtype(),
    )?;
    let out = own_rows(tokens, exchange, hidden.dtype())?;

    ops::collective::expert_exchange(
        &descriptor,
        ExchangeArgs {
            transport: None,
            tokens: sent(OWNER, tokens, exchange)?,
            capacity: exchange.step_rows,
            state: ops::Operand::unread(),
            ids: ops::operand(ids)?,
            weights: ops::operand(weights)?,
            live: live_tokens.map(ops::operand).transpose()?,
            out: ops::operand(&out)?,
            fused: Some(layer),
            ranks: exchange.ranks,
        },
    )?;
    Ok(out)
}

/// The whole expert layer of an expert group in one launch over MXFP4 experts: the routes of
/// this rank's `hidden` rows (`[tokens, hidden]` BF16) are exchanged, every rank's experts run
/// over the rows they receive, the `shared` experts over every row of this rank, and each row's
/// sum comes back: `[tokens, hidden]` in `hidden`'s element type (`ops::moe::exchange`'s
/// module documentation states the formula).
///
/// The launch encodes the rows into MXFP8 itself, as `mxfp8_quantize` does, so `hidden` is read
/// as it stands. `ids` and `weights` are the `[tokens, top]` routes, as the router's top-k writes
/// them (slots and FP32 weights). `up_gate` and `down` are this rank's stacks in the arrangement
/// the launch reads (`Descriptor::MXFP4_UP_GATE_VALUES`, `MXFP4_DOWN_VALUES` and
/// `MXFP4_SCALES`); `shared` is the shared experts' gated stack and down projection in the same
/// arrangements, which the exchange's transport was laid out for (`exchange.fused_experts`).
/// `experts` is the router's expert count.
///
/// # Errors
///
/// When the tensors are CPU tensors, more rows are sent than a rank's block holds, the exchange
/// was laid out for other experts, or an operator refuses its arguments.
#[expect(
    clippy::too_many_arguments,
    reason = "the launch reads the rows, the routes, the lent rows and every operand of the layer"
)]
pub fn expert_exchange_fused_mxfp4(
    hidden: &Tensor,
    ids: &Tensor,
    weights: &Tensor,
    live_tokens: Option<&Tensor>,
    exchange: &ExpertExchange,
    (up_gate, down): (&Mxfp4Stack, &Mxfp4Stack),
    shared: Option<(&Mxfp8Weight, &Mxfp8Weight)>,
    experts: u32,
) -> Result<Tensor> {
    const OWNER: &str = "expert_exchange_fused_mxfp4";

    let tokens = hidden.dims()[0];
    let local = up_gate.codes.dims().first().copied().unwrap_or(0).max(1);
    let intermediate = up_gate.codes.dims().get(1).copied().unwrap_or(0) / 2;
    let stated = exchange.fused_experts;
    let shared_count = match shared {
        Some((shared_up_gate, _)) => {
            (shared_up_gate.codes.dims()[0] / 2 / intermediate.max(1)) as u32
        }
        None => 0,
    };
    if stated.weights != ExpertWeights::Mxfp4 || stated.shared != shared_count {
        return Err(Error::shape(
            OWNER,
            format!(
                "the exchange's fused launch runs {} experts with {} shared, and this layer \
                 holds MXFP4 experts with {shared_count} shared",
                stated.weights, stated.shared
            ),
        ));
    }

    // The layer's device addresses, as the package's `uniserve_fused_layer` declares them.
    let address = |tensor: &Tensor| -> Result<*const std::ffi::c_void> {
        Ok(ops::operand(tensor)?.ptr.address() as usize as *const std::ffi::c_void)
    };
    let expert_bytes = |tensor: &Tensor| -> Result<i64> {
        let bytes = tensor.elem_count() * tensor.dtype().size_in_bytes().unwrap_or(1) / local;
        i64::try_from(bytes).map_err(|_| Error::shape(OWNER, "a scale table beyond i64".to_owned()))
    };
    let (shared_up_gate, shared_down) = match shared {
        Some((up_gate, down)) => (
            [address(&up_gate.codes)?, address(&up_gate.scales)?],
            [address(&down.codes)?, address(&down.scales)?],
        ),
        None => ([std::ptr::null(); 2], [std::ptr::null(); 2]),
    };
    let layer = FusedLayer {
        up_gate: address(&up_gate.codes)?,
        up_gate_scales: address(&up_gate.scales)?,
        down: address(&down.codes)?,
        down_scales: address(&down.scales)?,
        up_gate_alpha: std::ptr::null(),
        down_alpha: std::ptr::null(),
        gated_scale: std::ptr::null(),
        up_gate_scale_bytes: expert_bytes(&up_gate.scales)?,
        down_scale_bytes: expert_bytes(&down.scales)?,
        shared_up_gate: shared_up_gate[0],
        shared_up_gate_scales: shared_up_gate[1],
        shared_down: shared_down[0],
        shared_down_scales: shared_down[1],
    };

    let descriptor = exchange_descriptor(
        Mode::Fused,
        exchange,
        (experts, intermediate as u32),
        hidden.dtype(),
    )?;
    let out = own_rows(tokens, exchange, hidden.dtype())?;

    ops::collective::expert_exchange(
        &descriptor,
        ExchangeArgs {
            transport: None,
            tokens: sent(OWNER, tokens, exchange)?,
            capacity: exchange.step_rows,
            state: ops::operand(hidden)?,
            ids: ops::operand(ids)?,
            weights: ops::operand(weights)?,
            live: live_tokens.map(ops::operand).transpose()?,
            out: ops::operand(&out)?,
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
    const OWNER: &str = SigmoidGatedAddCall::OWNER;

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

    compute(SigmoidGatedAddCall {
        target,
        update,
        gate,
        width,
    })
}

/// A checked call of [`sigmoid_gated_add_`].
struct SigmoidGatedAddCall<'t> {
    target: Tensor,
    update: &'t Tensor,
    gate: &'t Tensor,
    width: u32,
}

impl SigmoidGatedAddCall<'_> {
    const OWNER: &'static str = "sigmoid_gated_add_";
}

impl Fused for SigmoidGatedAddCall<'_> {
    type Output = Tensor;

    fn device(&self) -> &Device {
        self.target.device()
    }

    fn cpu(self) -> Result<Self::Output> {
        let Self {
            target,
            update,
            gate,
            ..
        } = self;
        let scaled = activate(&gate.to_dtype(ACCUMULATE)?, Activation::Sigmoid)?
            .broadcast_mul(&update.to_dtype(ACCUMULATE)?)?;
        let sum = (target.to_dtype(ACCUMULATE)? + scaled)?.to_dtype(target.dtype())?;

        target.slice_set(&sum, 0, 0)?;
        Ok(target)
    }

    fn launch(self) -> Result<Self::Output> {
        let Self {
            target,
            update,
            gate,
            width,
        } = self;
        ops::launch::<SigmoidGatedAdd>(
            &SigmoidGatedAddDescriptor::new(target.dtype(), width),
            SigmoidGatedAddArgs {
                target: ops::operand(&as_rows(Self::OWNER, &target)?)?,
                update: ops::operand(&as_rows(Self::OWNER, update)?)?,
                gate: ops::operand(&as_rows(Self::OWNER, gate)?)?,
            },
        )?;
        Ok(target)
    }
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
                && rows.is_multiple_of(TILE.0 as usize) =>
        {
            Ok((rows / TILE.0 as usize, rows))
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
    Ok(ops::operand(&map(maps, row, entries)?)?)
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
    let tile = row.broadcast_div(&integer(TILE.0, &row)?)?;
    let expert = map(maps, TILE_TO_EXPERT, rows / TILE.0 as usize)?.index_select(&tile, 0)?;

    Ok(live.where_cond(&expert, &expert.zeros_like()?)?)
}

/// Each permuted row of `x` (`[rows, k]` FP32, on the CPU) times the matrix of its expert in
/// `stack`, that matrix's rows taken in `order` (`[n]`, their own order out of the stack's
/// arrangement): `[rows, n]` FP32, zero on the rows that are not `live`.
///
/// The live rows are grouped by the expert `experts` names for each, and each expert a live row
/// names is decoded alone and multiplies its rows at once. So the formula holds one expert's FP32
/// matrix at a time, whatever the experts and the sort's padded rows: a buffer of mostly padding
/// (a tile of 128 rows for each of a few routes over many experts) holds no copy of a matrix per
/// row.
fn expert_products(
    stack: &Mxfp4Stack,
    order: &Tensor,
    x: &Tensor,
    (live, experts): (&Tensor, &Tensor),
) -> Result<Tensor> {
    let (rows, width) = (x.dim(0)?, order.dim(0)?);
    let live: Vec<u8> = live.to_dtype(DType::U8)?.to_vec1()?;
    let experts: Vec<u32> = experts.to_vec1()?;

    let mut members: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    for (row, (&held, &expert)) in live.iter().zip(&experts).enumerate() {
        if held != 0 {
            members.entry(expert).or_default().push(row as u32);
        }
    }

    let mut product = Tensor::zeros((rows, width), ACCUMULATE, &Device::Cpu)?;
    for (expert, rows) in members {
        let one = |tensor: &Tensor| tensor.narrow(0, expert as usize, 1)?.contiguous();
        let matrix = decoded_mxfp4(&one(&stack.codes)?, &one(&stack.scales)?, stack.layout)?
            .squeeze(0)?
            .index_select(order, 0)?;
        let rows = Tensor::new(rows.as_slice(), &Device::Cpu)?;
        let rows_product = x.index_select(&rows, 0)?.matmul(&matrix.t()?)?;
        product = product.index_add(&rows, &rows_product, 0)?;
    }
    Ok(product)
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

/// Where `layout` stores each of the `rows` rows of a packed matrix, `[rows]` U32: the rows a
/// CPU formula reads back in the matrix's own order.
fn stored_order(layout: ValueLayout, rows: usize) -> Result<Tensor> {
    let order: Vec<u32> = (0..rows as u64)
        .map(|row| layout.destination(row, rows as u64) as u32)
        .collect();

    Ok(Tensor::from_vec(order, rows, &Device::Cpu)?)
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

    // Only a fused launch runs experts; the two directions state none, whatever the layer's are.
    let fused = match mode {
        Mode::Fused => exchange.fused_experts,
        Mode::Dispatch | Mode::Combine => FusedExperts::NONE,
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
        multiprocessors: exchange.multiprocessors,
        dtype,
        weights: fused.weights,
        shared: fused.shared,
        limit: fused.limit,
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
    Ok(ops::empty((tokens, hidden), dtype)?)
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
    let tile = f64::from(TILE.0);

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
