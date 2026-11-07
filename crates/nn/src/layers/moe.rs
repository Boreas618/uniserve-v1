//! The mixture-of-experts block: a replicated router, the top-k routes, the sort that groups them
//! by expert, the two grouped products, and the weighted combine.
//!
//! The block is one layer because its parts only make sense together: routing decides which expert
//! rows a token's entries go to, the sort turns that decision into the tile layout the grouped
//! products read, and the combine sums the routes back with the FP32 weights the router produced
//! (specs/worker.md 9, Qwen3-MoE rows; `uniserve/models/qwen3/transformer.py:111-141`).
//!
//! ## Buffers
//!
//! The routing, sort and product buffers are sized from the batch's token count and the routing
//! geometry alone, never from one batch's routing distribution: the sort's tile and permutation
//! buffers hold the worst case ([`ops::moe::max_tiles`], [`ops::moe::max_permuted_rows`]), so a
//! captured graph replays over the same addresses however the router routes a later batch. The
//! layer takes them from [`Forward::arena`], in a fixed order, which is what makes a captured
//! bucket's plan static.
//!
//! ## Expert parallelism
//!
//! One rank holding every expert is the colocated case this layer serves; an expert-parallel group
//! routes over `expert_exchange` and is built with the H3 step, whose `moe_config.exchange`
//! program selects the transport. A `Moe` layer built for a layout that does not hold every
//! expert refuses the forward rather than computing a routing the weights do not have.

use ops::fp4::{BLOCK_ELEMENTS, Nvfp4Quantize, QuantizeArgs, QuantizeDescriptor, ScaleLayout};
use ops::moe::combine::{self, MoeUnpermute, UnpermuteArgs};
use ops::moe::exchange::{self, ExchangeArgs, ExpertExchange, Mode};
use ops::moe::grouped::{
    Activation, GatherGemm, GatherGemmArgs, GatherGemmDescriptor, GatherGemmNvfp4,
    GatherGemmNvfp4Args, GatherGemmNvfp4Descriptor, RouteGemm, RouteGemmArgs, RouteGemmDescriptor,
    RouteGemmNvfp4, RouteGemmNvfp4Args, RouteGemmNvfp4Descriptor,
};
use ops::moe::routing::{TopkArgs, TopkDescriptor, TopkSoftmax};
use ops::moe::sort::{self, MoeSort, SortArgs};
use ops::{DType, DevicePtr, Op, Shape, TensorView};

use crate::error::{Error, Result};
use crate::forward::{ALIGNMENT, Forward};
use crate::layers::Linear;
use crate::weights::{QuantizedStack, Weight};

/// The scratch one grouped product of `routes` permuted rows and `width` columns needs, in bytes.
///
/// Read from the artifact rather than guessed, and read to its guard:
/// `getWorkspaceSizesInBytes` (`trtllmGen_bmm_export/BatchedGemmInterface.h`) asks for the gated
/// activation's row-max planes and tile bars — each padded to a kibibyte — **only when**
/// `mUseDeepSeekFp8 && mFusedAct`, and `run` additionally requires both operands to be E4M3
/// (`BatchedGemmInterface.h:617`). The encoded pair this block launches is E2M1 against E2M1, so
/// `run` never reads those planes for it, and the only entry that can apply is the persistent
/// scheduler's counter.
///
/// The planes are still reserved, and this is the reason rather than a hedge: the artifact's sizing
/// function tests the two options and not the dtypes, so a package that sets both is *told* to pass
/// a buffer of that size, and a caller that sized from the dtypes alone would be second-guessing a
/// program it cannot see. The formula is therefore the artifact's, over the plan's padded route
/// count; `routes` is padded to whole MMA tiles because the program walks whole tiles.
///
/// The size is a property of the plan, not of the step: it is computed from the route count the plan
/// states, so every replay of a captured bucket reserves the same number of bytes and nothing in a
/// step grows a buffer.
fn grouped_workspace_bytes(routes: u64, width: u64) -> u64 {
    let token_tile = u64::from(MMA_TILE.0) * u64::from(MMA_CLUSTER.0);
    let padded = routes.max(1).div_ceil(token_tile) * token_tile;
    // One FP32 row maximum per 128 elements of the padded tile grid.
    let row_max = width * padded / 128 * 4;
    // One half-word of tile bars per (token tile, width tile) pair.
    let bars = (padded / token_tile) * width.div_ceil(u64::from(MMA_TILE.1)) / 2 * 4;
    pad_1k(row_max) + pad_1k(bars) + 4
}

/// The preamble a product's entry writes into the shared scratch before the artifact's workspace: a
/// sixteen-byte header holding the padded route count.
///
/// The bytes are the entry's rather than the artifact's, so they cannot be read from
/// `getWorkspaceSizesInBytes`, and a caller that sized from the artifact alone would be short by
/// exactly this preamble, which the entry refuses with both numbers named, before any launch.
const ENTRY_PREAMBLE: u64 = 16;

/// The scratch the encoded pair needs for `routes` permuted rows: one buffer serves both products in
/// turn, so the larger requirement wins, each carrying its own entry's preamble before the artifact's
/// planes. The first product writes `2 · intermediate` columns, the second `hidden`.
fn encoded_scratch_bytes(routes: u64, hidden: u64, intermediate: u64) -> u64 {
    let gated = intermediate.saturating_mul(2);
    let first = ENTRY_PREAMBLE.saturating_add(grouped_workspace_bytes(routes, gated));
    let second = ENTRY_PREAMBLE.saturating_add(grouped_workspace_bytes(routes, hidden));
    first.max(second)
}

/// `bytes` rounded up to a kibibyte, the artifact's padding of one workspace plane.
const fn pad_1k(bytes: u64) -> u64 {
    bytes.div_ceil(1024) * 1024
}

/// The tile geometry the grouped products' packages are compiled for.
///
/// Like the row tile of a normalization this is program content (specs/worker.md 5.2, 5.4); the
/// values here are the packages' default tactic for the Qwen3-MoE shapes, declared once in
/// [`crate::layers::launch`] rather than at every call.
const MMA_TILE: (u32, u32) = (128, 128);
/// The cluster shape the grouped products' packages are compiled for.
const MMA_CLUSTER: (u32, u32) = (1, 1);

/// Rows per permutation tile: the row extent of `MMA_TILE`, which is what a sort's permutation
/// buffer is padded to.
pub const TILE_ROWS: u32 = MMA_TILE.0;

/// Elements of a row one activation-encoding program holds at a time.
///
/// A whole number of sixteen-element scale blocks, as the layout requires, and small enough that a
/// 2048-element row is walked in a handful of passes rather than materialized in one block.
/// Elements one activation-encoding program covers.
///
/// It is the encoder's block and therefore a key field of `nvfp4_quantize`, so a model that builds an
/// [`crate::layers::Nvfp4Linear`] states it rather than choosing one: two deployments that disagreed
/// would compile two programs for one launch site.
pub const ENCODE_CHUNK: u32 = 1024;

/// The encoder's chunk is a whole number of scale blocks; a chunk that were not would have a partial
/// block with no scale to name it.
const _: () = assert!(ENCODE_CHUNK.is_multiple_of(BLOCK_ELEMENTS));

/// The geometry one expert step's buffers are sized from.
///
/// Both paths route, sort and combine identically, so both are sized from these five numbers; the
/// encoded path adds the activation planes on top.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MoeGeometry {
    /// The hidden width.
    pub hidden: u64,
    /// One half of the gated product's width.
    pub intermediate: u64,
    /// Routes per token.
    pub top: u64,
    /// Experts the block routes over.
    pub experts: u64,
    /// Rows per permutation tile.
    pub tile: u64,
}

impl MoeGeometry {
    /// The router's top-k program.
    ///
    /// `renormalize` and `scale` are arguments rather than fields because they are the checkpoint's
    /// routing convention — whether the kept weights are renormalized, and whether a learned scale
    /// then multiplies them — and not the block's shape.
    pub fn topk_descriptor(
        &self,
        scores: DType,
        renormalize: bool,
        scale: Option<DType>,
    ) -> TopkDescriptor {
        TopkDescriptor {
            scores,
            experts: self.experts as u32,
            top: self.top as u32,
            renormalize,
            scale,
        }
    }

    /// The sort's descriptor, for a block whose rank holds `local_experts` experts.
    ///
    /// `local_experts` is an argument because the same block sorts for the dense case (every expert)
    /// and for an expert-parallel rank (its own), and the two are different programs.
    pub fn sort_descriptor(&self, local_experts: u32) -> sort::Descriptor {
        sort::Descriptor {
            experts: self.experts as u32,
            top: self.top as u32,
            local_experts,
            tile_tokens: self.tile as u32,
            scales: DType::F32,
            counted: true,
        }
    }

    /// The first product's descriptor over the dense packed stack.
    pub fn gather_descriptor(&self, hidden: DType, activation: Activation) -> GatherGemmDescriptor {
        GatherGemmDescriptor {
            hidden: self.hidden as u32,
            intermediate: self.intermediate as u32,
            experts: self.experts as u32,
            top: self.top as u32,
            activation,
            dtype: hidden,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        }
    }

    /// The second product's descriptor over the dense packed stack.
    pub fn route_descriptor(&self, hidden: DType) -> RouteGemmDescriptor {
        RouteGemmDescriptor {
            hidden: self.hidden as u32,
            intermediate: self.intermediate as u32,
            experts: self.experts as u32,
            top: self.top as u32,
            dtype: hidden,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        }
    }

    /// The combine's descriptor: the routes summed in FP32 and rounded once to `output`.
    pub fn combine_descriptor(&self, product: DType, output: DType) -> combine::Descriptor {
        combine::Descriptor {
            layout: combine::Layout::Permuted,
            hidden: self.hidden as u32,
            top: self.top as u32,
            experts: self.experts as u32,
            input: product,
            scales: DType::F32,
            output,
        }
    }

    /// The first product's descriptor over an encoded stack: the block scale's element type and the
    /// scale layout are the weights', so they are arguments.
    ///
    /// The grid is over *routes* — tokens × `top`, computed at launch — so this key covers every
    /// route count and is not compiled per bucket.
    pub fn gather_nvfp4_descriptor(
        &self,
        hidden: DType,
        activation: Activation,
        scale: DType,
        layout: ScaleLayout,
    ) -> GatherGemmNvfp4Descriptor {
        GatherGemmNvfp4Descriptor {
            hidden: self.hidden as u32,
            intermediate: self.intermediate as u32,
            experts: self.experts as u32,
            top: self.top as u32,
            activation,
            dtype: hidden,
            scale,
            layout,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        }
    }

    /// The second product's descriptor over an encoded stack, for the same reason and with the same
    /// grid: routes, not rows.
    pub fn route_nvfp4_descriptor(
        &self,
        hidden: DType,
        scale: DType,
        layout: ScaleLayout,
    ) -> RouteGemmNvfp4Descriptor {
        RouteGemmNvfp4Descriptor {
            hidden: self.hidden as u32,
            intermediate: self.intermediate as u32,
            experts: self.experts as u32,
            top: self.top as u32,
            dtype: hidden,
            scale,
            layout,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        }
    }
}

/// The encoded step's launches, in the order it runs them. A buffer names the one that writes it.
///
/// The two encodings are launches of their own rather than wider operands of the products: the router's
/// input rows and FC1's gated rows are different widths, so one program cannot serve both.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Launch {
    /// `nvfp4_quantize` over the router's input rows.
    Encode,
    /// `gather_gemm_nvfp4`: the fused up/gate product, one gated row per permuted route.
    FirstProduct,
    /// `nvfp4_quantize` over FC1's gated rows.
    SecondEncode,
    /// `route_gemm_nvfp4`: the down product, one permuted row of hidden width per route.
    SecondProduct,
}

impl Launch {
    /// The operator this launch is, as its key names it.
    pub fn operator(self) -> &'static str {
        match self {
            Self::Encode | Self::SecondEncode => "nvfp4_quantize",
            Self::FirstProduct => "gather_gemm_nvfp4",
            Self::SecondProduct => "route_gemm_nvfp4",
        }
    }
}

/// One buffer of the encoded step: its extent, its element type, and the launch that writes it.
///
/// The element type is the buffer's own and not a default: packed codes and block scales are one byte
/// each for two different reasons, and the per-row factors are FP32 because the products read them as
/// one scalar per row.
#[derive(Clone, Copy, Debug)]
pub struct Buffer {
    pub shape: Shape,
    pub dtype: DType,
    /// The step's launch that fills this buffer; the combine reads some of them and writes none.
    pub written_by: Launch,
}

impl Buffer {
    /// The bytes this buffer's extent occupies, unaligned.
    pub fn bytes(&self) -> Result<u64> {
        let element = self
            .dtype
            .bytes()
            .ok_or_else(|| Error::shape("Moe", format!("{} has no byte size", self.dtype)))?;
        Ok(self.shape.numel().saturating_mul(element))
    }
}

/// The buffers one encoded expert step allocates, for `tokens` rows and a block's geometry.
///
/// One table answers both questions the step has about its working set — how much the plan must
/// reserve, and what the launches write — so a buffer cannot be reserved without being written and
/// cannot be written without being reserved. That equality is what the plan and the step kept losing:
/// a term missing from the sum is a write past the arena, and a buffer the plan never heard of is space
/// nobody can attribute.
///
/// Every row count is the permutation's worst case (see [`encoded_bytes`]), and the activation planes'
/// layout is the encoder's plain one — the products' *weight* planes are their own descriptor field and
/// are not part of this table.
#[derive(Clone, Copy, Debug)]
pub struct EncodedStep {
    pub values: Buffer,
    pub scales: Buffer,
    pub token_scale: Buffer,
    pub gated: Buffer,
    /// Both products' shared scratch. The first product's entry writes its preamble and planes; the
    /// second reuses the same region after the first launch, which is why one buffer serves both.
    pub scratch: Buffer,
    pub gated_values: Buffer,
    pub gated_scales: Buffer,
    pub gated_token_scale: Buffer,
    pub products: Buffer,
}

impl EncodedStep {
    /// The step's buffers for `tokens` rows, at the permutation's worst-case height.
    pub fn new(
        tokens: u64,
        geometry: MoeGeometry,
        layout: ScaleLayout,
        gated_layout: ScaleLayout,
        dtype: DType,
    ) -> Result<Self> {
        let MoeGeometry {
            hidden,
            intermediate,
            top,
            experts,
            tile,
        } = geometry;
        if dtype.bytes().is_none() {
            return Err(Error::shape("Moe", format!("{dtype} has no byte size")));
        }
        let routes = ops::moe::max_permuted_rows(tokens, top, experts, tile);
        let buffer = |shape: Shape, dtype: DType, written_by: Launch| Buffer {
            shape,
            dtype,
            written_by,
        };
        Ok(Self {
            // The activation's packed codes, a byte per two codes, and one block scale each.
            values: buffer(Shape::new(&[tokens, hidden / 2]), DType::U8, Launch::Encode),
            scales: buffer(layout.shape(tokens, hidden), DType::U8, Launch::Encode),
            token_scale: buffer(Shape::new(&[tokens]), DType::F32, Launch::Encode),
            gated: buffer(
                Shape::new(&[routes, intermediate]),
                dtype,
                Launch::FirstProduct,
            ),
            gated_values: buffer(
                Shape::new(&[routes, intermediate / 2]),
                DType::U8,
                Launch::SecondEncode,
            ),
            gated_scales: buffer(
                gated_layout.shape(routes, intermediate),
                DType::U8,
                Launch::SecondEncode,
            ),
            gated_token_scale: buffer(Shape::new(&[routes]), DType::F32, Launch::SecondEncode),
            products: buffer(Shape::new(&[routes, hidden]), dtype, Launch::SecondProduct),
            // The products' shared scratch, the entries' preamble included: sized from the route count
            // the plan states, so a replay reserves the same addresses.
            scratch: buffer(
                Shape::new(&[encoded_scratch_bytes(routes, hidden, intermediate).max(1)]),
                DType::U8,
                Launch::FirstProduct,
            ),
        })
    }

    /// The buffers with their names, in the order the launches write them.
    ///
    /// The names are the table's own, so a report of what the step holds and a test of what it reserves
    /// read the same words.
    pub fn iter(&self) -> [(&'static str, &Buffer); 9] {
        [
            ("values", &self.values),
            ("scales", &self.scales),
            ("token_scale", &self.token_scale),
            ("gated", &self.gated),
            ("scratch", &self.scratch),
            ("gated_values", &self.gated_values),
            ("gated_scales", &self.gated_scales),
            ("gated_token_scale", &self.gated_token_scale),
            ("products", &self.products),
        ]
    }

    /// The bytes these buffers reserve from the arena, each aligned as the arena hands it out.
    pub fn bytes(&self) -> Result<u64> {
        let mut total: u64 = 0;
        for (_, buffer) in self.iter() {
            total = total.saturating_add(buffer.bytes()?.next_multiple_of(ALIGNMENT));
        }
        Ok(total)
    }
}

/// The bytes one forward's encoded step reserves from the arena, for `tokens` rows.
///
/// The step's buffers are the same for every layer and for every replay of a plan, so they are
/// stated rather than discovered: a deployment sizes its arena from this, and a reservation the step
/// made beyond it would be a write past the plan.
///
/// Every row count here is the permutation's **worst case**, not the routes a particular batch
/// happens to produce: `max_permuted_rows(tokens, top, experts, tile)` is what a skip-routed sort can
/// reach when every expert owns a single route — for the slice's own 15-token batch that is 15360
/// rows against 120 expanded routes. The host cannot know the tile count (it is device data the sort
/// produces), so the buffers are sized for the worst case and a replay over the same plan finds the
/// same addresses, which is the price of a captured graph. **Both paths pay it**: the combine indexes
/// its input in the permuted row space, so the gated rows and the products are permuted-height
/// whether the experts are dense or encoded, and the encoded path's extra cost is its activation
/// planes rather than the row height.
///
/// `layout` is the activation encoding's block-scale layout: a field of the products' descriptors,
/// not of the weights, whose scales have their own layout.
pub fn encoded_bytes(
    tokens: u64,
    geometry: MoeGeometry,
    layout: ScaleLayout,
    gated_layout: ScaleLayout,
    dtype: DType,
) -> Result<u64> {
    EncodedStep::new(tokens, geometry, layout, gated_layout, dtype)?.bytes()
}

/// The bytes the routing prefix reserves for `tokens` rows: the scores, the kept ids and weights,
/// the expert-mapped ids, and the sort's four metadata arrays and their two counters.
///
/// Every path through this block starts here, so a plan that covers the expert step but not the
/// prefix leaves the first launch of every routed layer writing past its region.
pub fn routing_bytes(tokens: u64, geometry: MoeGeometry, dtype: DType) -> Result<u64> {
    Ok(router_bytes(tokens, geometry, dtype)?.saturating_add(sort_bytes(tokens, geometry)?))
}

/// The bytes the replicated router reserves for `tokens` rows: its scores over the whole expert
/// vocabulary, and the kept ids, weights and expert-mapped ids of each route.
///
/// This half is the same on every path — a rank routes whole tokens the same way whether the experts
/// live on one rank or over a group — so the exchange states it once for its own row count and the
/// sort's half separately at the height it receives.
pub fn router_bytes(tokens: u64, geometry: MoeGeometry, dtype: DType) -> Result<u64> {
    let MoeGeometry { top, experts, .. } = geometry;
    let width_bytes = dtype
        .bytes()
        .ok_or_else(|| Error::shape("Moe", format!("{dtype} has no byte size")))?;
    let mut total: u64 = 0;
    for bytes in [
        tokens * experts * width_bytes, // the router's scores
        tokens * top * 4,               // its kept ids, i32
        tokens * top * 4,               // the route weights, f32
        tokens * top * 4,               // the expert-mapped ids, i32
    ] {
        total = total.saturating_add(bytes.next_multiple_of(ALIGNMENT));
    }
    Ok(total)
}

/// The bytes the permutation's metadata reserves for `tokens` rows over `geometry.experts` experts:
/// each tile's expert and live-row count, the two route maps, and the two counters.
///
/// The exchange sizes this half at the rows it **receives** and over the experts this rank holds, so
/// the same seven terms appear twice on an expert-parallel path with different numbers in them.
pub fn sort_bytes(tokens: u64, geometry: MoeGeometry) -> Result<u64> {
    let MoeGeometry {
        top, experts, tile, ..
    } = geometry;
    let tiles = ops::moe::max_tiles(tokens, top, experts, tile);
    let routes = ops::moe::max_permuted_rows(tokens, top, experts, tile);
    let mut total: u64 = 0;
    for bytes in [
        tiles * 4,        // each tile's expert
        tiles * 4,        // each tile's live rows
        tokens * top * 4, // the permuted row of each expanded route
        routes * 4,       // the expanded row of each permuted position
        routes * 4,       // the token row of each permuted position
        4,                // the padded row count
        4,                // the non-empty tile count
    ] {
        total = total.saturating_add(bytes.next_multiple_of(ALIGNMENT));
    }
    Ok(total)
}

/// The bytes the expert products reserve for `tokens` rows: the gated rows and the permuted
/// products, both at the permutation's height, in the activation's element type.
pub fn dense_bytes(tokens: u64, geometry: MoeGeometry, dtype: DType) -> Result<u64> {
    let MoeGeometry {
        hidden,
        intermediate,
        top,
        experts,
        tile,
    } = geometry;
    let width_bytes = dtype
        .bytes()
        .ok_or_else(|| Error::shape("Moe", format!("{dtype} has no byte size")))?;
    let routes = ops::moe::max_permuted_rows(tokens, top, experts, tile);
    let mut total: u64 = 0;
    for bytes in [
        routes * intermediate * width_bytes, // FC1's gated rows
        routes * hidden * width_bytes,       // FC2's permuted products
    ] {
        total = total.saturating_add(bytes.next_multiple_of(ALIGNMENT));
    }
    Ok(total)
}

/// The bytes one routed layer's whole expert step reserves for `tokens` rows: the routing prefix,
/// then the products — dense or encoded — as `encoded` says.
pub fn step_bytes(
    tokens: u64,
    geometry: MoeGeometry,
    layout: ScaleLayout,
    gated_layout: ScaleLayout,
    dtype: DType,
    encoded: bool,
) -> Result<u64> {
    let routing = routing_bytes(tokens, geometry, dtype)?;
    let expert = if encoded {
        encoded_bytes(tokens, geometry, layout, gated_layout, dtype)?
    } else {
        dense_bytes(tokens, geometry, dtype)?
    };
    Ok(routing.saturating_add(expert))
}

/// An expert-parallel deployment's plan inputs: the group's rank count, and the tokens per rank the
/// exchange carries.
///
/// `capacity` is a **load-time** input, not a request property: the plan is built when the model loads
/// and bound to the deployment's capacity, which is what lets it reserve an exchange worst case at all.
/// A request is admitted within that bound, and a batch larger than it is a deployment error rather
/// than a buffer that grows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Group {
    /// Ranks among which the experts are divided.
    pub ranks: u64,
    /// Tokens one rank's exchange carries.
    pub capacity: u64,
    /// Leading ranks of the exchange that hold no experts; zero when every rank holds its share.
    pub attention: u64,
}

impl Group {
    /// The rows a rank is sent: every rank's capacity, one after another.
    ///
    /// The exchange hands a rank its own `capacity` tokens first and then its peers', so the rows this
    /// rank sorts over are the whole group's, not its own batch's.
    pub fn received(&self) -> u64 {
        (self.ranks + self.attention).saturating_mul(self.capacity)
    }
}

/// The permuted rows an expert-parallel rank sorts over, for a group and the block's geometry.
///
/// The rows are the **received** ones, over the experts the rank holds: the group's tokens are
/// permuted by the expert that takes them, and this rank sees only those its own `experts / ranks`
/// tiles own. That product, not the single-rank one, is what the step's buffers are shaped by.
pub fn exchanged_rows(tokens: u64, geometry: MoeGeometry, group: Group) -> Result<u64> {
    let local = local_experts(geometry, group)?;
    if group.capacity == 0 {
        return Err(Error::shape(
            "Moe",
            "an expert-parallel group carries no rows: a capacity of zero is a deployment that \
             never stated what its exchange holds, and the plan cannot reserve an unknown worst case"
                .to_owned(),
        ));
    }
    let received = group.received();
    if tokens > group.capacity {
        return Err(Error::shape(
            "Moe",
            format!(
                "{tokens} tokens against an exchange capacity of {}: the capacity is the plan's, and \
                 a batch larger than it is a deployment that cannot be served",
                group.capacity
            ),
        ));
    }
    Ok(ops::moe::max_permuted_rows(
        received,
        geometry.top,
        local,
        geometry.tile,
    ))
}

/// The experts one rank holds, refusing a group that does not divide the vocabulary.
fn local_experts(geometry: MoeGeometry, group: Group) -> Result<u64> {
    if group.ranks == 0 || geometry.experts == 0 {
        return Err(Error::shape(
            "Moe",
            "an expert-parallel group needs a rank count and a vocabulary".to_owned(),
        ));
    }
    if !geometry.experts.is_multiple_of(group.ranks) {
        return Err(Error::shape(
            "Moe",
            format!(
                "{} experts do not divide into {} ranks",
                geometry.experts, group.ranks
            ),
        ));
    }
    Ok(geometry.experts / group.ranks)
}

/// The bytes one expert-parallel routed layer's step reserves for `tokens` rows of a group.
///
/// The single-rank [`step_bytes`] is not this size, and the difference is not a constant: a rank sorts
/// over `ranks · capacity` rows of the experts it holds, so the permutation's worst case changes with
/// the group. Four parts, each named:
///
/// - the **router**, at the batch's own height over the whole vocabulary: every rank routes whole
///   tokens the same way, so this half is the single-rank path's ([`router_bytes`]);
/// - the **sort's metadata**, at the received height over the rank's own experts ([`sort_bytes`]);
/// - the **gated rows and the products**, at the received height ([`dense_bytes`]'s two terms);
/// - the **combined rows**, one per received token, which no single-rank path has.
pub fn exchanged_bytes(
    tokens: u64,
    geometry: MoeGeometry,
    group: Group,
    dtype: DType,
) -> Result<u64> {
    let width_bytes = dtype
        .bytes()
        .ok_or_else(|| Error::shape("Moe", format!("{dtype} has no byte size")))?;
    let local = local_experts(geometry, group)?;
    let received = group.received();
    let rows = exchanged_rows(tokens, geometry, group)?;
    let received_geometry = MoeGeometry {
        experts: local,
        ..geometry
    };
    let mut total = router_bytes(tokens, geometry, dtype)?;
    for bytes in [
        sort_bytes(received, received_geometry)?,
        rows * geometry.intermediate * width_bytes, // FC1's gated rows
        rows * geometry.hidden * width_bytes,       // FC2's permuted products
        received * geometry.hidden * width_bytes,   // the combined rows, in receive order
    ] {
        total = total.saturating_add(bytes.next_multiple_of(ALIGNMENT));
    }
    Ok(total)
}

/// The per-expert scalars a fused exchange's kernel multiplies in, one table per use.
#[derive(Clone)]
pub struct FusedTables {
    /// `[local]` f32: dequantizes the first product, a weight's tensor scale times the rows' scale.
    pub up_gate_alpha: Weight,
    /// `[local]` f32: dequantizes the second product.
    pub down_alpha: Weight,
    /// `[local]` f32: the global scale the gated rows are encoded with.
    pub gated_scale: Weight,
}

/// The expert stacks a block multiplies.
///
/// A checkpoint is either dense (BF16/F16 expert matrices) or calibrated NVFP4 (the encoded fields
/// the loader publishes as a [`QuantizedStack`]); the routing, the sort and the combine are the
/// same, and only the two multiplies differ.
///
/// The variants differ by a few hundred bytes and the encoded one is already boxed; the remaining
/// difference is not worth another indirection on a value built once per layer.
#[derive(Clone)]
#[allow(
    clippy::large_enum_variant,
    reason = "a block's stacks are built once at load, never collected, and the encoded variant \
              already boxes its weights"
)]
enum ExpertStacks {
    /// `[experts, 2·intermediate, hidden]` and `[experts, hidden, intermediate]`.
    Dense { up_gate: Weight, down: Weight },
    /// The same two stacks and their block scales, packed two codes a byte. Boxed because a
    /// [`QuantizedStack`] carries three weights and the dense variant carries two views: the
    /// difference would otherwise size every block by its rarest variant.
    Encoded {
        up_gate: Box<QuantizedStack>,
        down: Box<QuantizedStack>,
    },
}

/// What a block's routing prefix produced: the views every expert product and the combine read.
///
/// All of them are the forward's arena, sized from the routing geometry, so a captured graph
/// replays over the same addresses however the router routes a later batch.
#[derive(Clone, Copy)]

struct Routes {
    /// `[tokens, top]` i32: the weight row of each route.
    local_ids: TensorView,
    /// `[tokens, top]` f32: the route's normalized weight.
    weights: TensorView,
    /// The permuted rows the products write, padding included.
    rows: u64,
    /// `[max_tiles]` i32: the expert of each tile.
    tile_to_expert: TensorView,
    /// `[max_tiles]` i32: the live rows of each tile.
    tile_to_limit: TensorView,
    /// `[tokens, top]` i32: the permuted row of each expanded route.
    expanded_to_permuted: TensorView,
    /// `[max_permuted_rows]` i32: the expanded route of each permuted row.
    permuted_to_expanded: TensorView,
    /// `[max_permuted_rows]` i32: the token row each permuted row gathers.
    permuted_to_token: TensorView,
    /// `[1]` i32: the live tiles.
    num_non_exiting_tiles: TensorView,
}

/// A mixture-of-experts block over a colocated expert group.
pub struct Moe {
    router: Linear,
    stacks: ExpertStacks,
    map: Option<Weight>,
    experts: u32,
    top: u32,
    hidden: u32,
    intermediate: u32,
    activation: Activation,
    renormalize: bool,
    scale: Option<Weight>,
    /// The per-expert tables a fused exchange reads, on an encoded expert-parallel block.
    fused: Option<FusedTables>,
    tile: u32,
    /// Experts of the vocabulary this rank holds; every expert in the colocated case.
    local: u32,
    /// The first expert of this rank's window.
    local_offset: u32,
}

impl Moe {
    /// The block-scale layout of the **activation** plane, which is the products' `layout` field.
    ///
    /// Every E2M1 artifact this deployment builds serves the activation side in the 128x4 swizzled
    /// form (`gather_gemm_nvfp4`'s `build.py` states it as the artifact's own rule), and the encoder
    /// is launched per descriptor, so a plain layout here asks the store for a program no artifact
    /// covers. The **weights'** plane is a different field of a different descriptor — a property of
    /// each encoded stack, which the loader publishes in the form that product's kernels read — and
    /// the two must not be swapped.
    pub const ACTIVATION_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Swizzled128X4;

    /// The block-scale layout of the **gated product's operand**, which is the one plane the two
    /// families disagree about.
    ///
    /// The artifact's gated (swiGlu) kernels read their tokens operand in the **linear** form and its
    /// plain kernels read theirs swizzled: `gather_gemm_nvfp4`'s manifest states
    /// `mSfLayoutB = linear` and `route_gemm_nvfp4`'s states it swizzled, and the artifact's own
    /// `BatchedGemmOptions.h` says it as a rule — "Tokens need use SF linear layout when being
    /// routed". One encoding for both is the defect that made the first end-to-end NVFP4 run produce
    /// a row no argmax could read.
    pub const GATED_TOKEN_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Linear;

    /// A block whose router is `router` and whose experts are `up_gate` and `down`.
    ///
    /// `up_gate` is `[experts, 2 · intermediate, hidden]` with the up projection first, `down` is
    /// `[experts, hidden, intermediate]`, and `router` has one output per expert. `map` is the
    /// expert-parallel map when the layout reorders experts; with every expert colocated it is
    /// absent and a route's logical id is its weight row.
    #[allow(
        clippy::too_many_arguments,
        reason = "every argument is a geometry or weight the design states separately; a \
                  configuration struct would restate them"
    )]
    pub fn new(
        router: Linear,
        up_gate: Weight,
        down: Weight,
        map: Option<Weight>,
        experts: u32,
        top: u32,
        activation: Activation,
        renormalize: bool,
        scale: Option<Weight>,
    ) -> Result<Self> {
        // The colocated case: every expert of the vocabulary is resident here.
        Self::build(
            router,
            ExpertStacks::Dense { up_gate, down },
            map,
            experts,
            experts,
            0,
            top,
            activation,
            renormalize,
            scale,
        )
    }

    /// A block whose experts are a calibrated NVFP4 checkpoint's encoded fields.
    ///
    /// Each stack is `[experts, 2·intermediate, hidden/2]` (up first, gate second) or
    /// `[experts, hidden, intermediate/2]` packed two E2M1 codes a byte, with its block scales and
    /// its per-expert tensor reciprocals — what [`WeightLoader::nvfp4_stack`] publishes. The block
    /// routes, sorts and combines exactly as the dense one does; only the two multiplies read
    /// packed weights, and both stay in FP32 until the store.
    ///
    /// [`WeightLoader::nvfp4_stack`]: crate::WeightLoader::nvfp4_stack
    #[allow(
        clippy::too_many_arguments,
        reason = "the same arguments as the dense constructor, with the encoded stacks in place \
                  of the dense weights"
    )]
    pub fn quantized(
        router: Linear,
        up_gate: QuantizedStack,
        down: QuantizedStack,
        map: Option<Weight>,
        experts: u32,
        top: u32,
        activation: Activation,
        renormalize: bool,
        scale: Option<Weight>,
    ) -> Result<Self> {
        // A quantized block is colocated: the encoded stacks hold every expert the router names.
        Self::build(
            router,
            ExpertStacks::Encoded {
                up_gate: Box::new(up_gate),
                down: Box::new(down),
            },
            map,
            experts,
            experts,
            0,
            top,
            activation,
            renormalize,
            scale,
        )
    }

    /// The block over `local` experts of the vocabulary starting at `local_offset`.
    #[allow(
        clippy::too_many_arguments,
        reason = "the block is named by its parts: the router, the two expert stacks, the map, \
                  the vocabulary, the resident window and the routing"
    )]
    fn build(
        router: Linear,
        stacks: ExpertStacks,
        map: Option<Weight>,
        experts: u32,
        local: u32,
        local_offset: u32,
        top: u32,
        activation: Activation,
        renormalize: bool,
        scale: Option<Weight>,
    ) -> Result<Self> {
        if top == 0 || top > ops::moe::routing::MAX_TOP {
            return Err(Error::shape(
                "Moe",
                format!(
                    "a top of {top} is outside the routers' 1..={}",
                    ops::moe::routing::MAX_TOP
                ),
            ));
        }
        if experts == 0 || experts > ops::moe::routing::MAX_EXPERTS {
            return Err(Error::shape(
                "Moe",
                format!(
                    "{experts} experts are outside a router's 1..={}",
                    ops::moe::routing::MAX_EXPERTS
                ),
            ));
        }
        if router.n() != experts {
            return Err(Error::shape(
                "Moe",
                format!(
                    "the router has {} outputs for {experts} experts",
                    router.n()
                ),
            ));
        }
        if local == 0
            || local_offset
                .checked_add(local)
                .is_none_or(|end| end > experts)
        {
            return Err(Error::shape(
                "Moe",
                format!("a window of {local} experts at {local_offset} does not lie in {experts}"),
            ));
        }
        // The two kinds state their geometry differently — a dense stack's output width is its
        // second axis, a packed one's is half of it — so each is checked against its own layout and
        // the block keeps only the geometry both agree on.
        let (up_gate_shape, down_shape) = match &stacks {
            ExpertStacks::Dense { up_gate, down } => (up_gate.shape(), down.shape()),
            ExpertStacks::Encoded { up_gate, down } => {
                let packed = up_gate.values.shape();
                let contraction = down.values.shape();
                if packed.rank() != 3
                    || packed.dim(0) != u64::from(local)
                    || contraction.rank() != 3
                    || contraction.dim(0) != u64::from(local)
                {
                    return Err(Error::shape(
                        "Moe",
                        format!(
                            "the encoded stacks are {:?} and {:?}, [{local}, 2·intermediate, \
                             hidden/2] and [{local}, hidden, intermediate/2] expected",
                            packed.dims(),
                            contraction.dims()
                        ),
                    ));
                }
                // The block scales are the checkpoint's own one-byte type, which the product's
                // descriptor states: `F8_E4M3` for a calibrated export, `U8` where an export labels
                // the same bytes as raw. Nothing converts them here, so a stack that is neither is
                // refused where the descriptor would be composed, with both dtypes named.
                for (name, stack) in [("up/gate", &up_gate.scales), ("down", &down.scales)] {
                    let scales = stack.dtype();
                    if !matches!(scales, DType::U8 | DType::F8E4M3) {
                        return Err(Error::shape(
                            "Moe",
                            format!(
                                "the {name} block scales are {scales}, and a product's kernel \
                                 reads one-byte scales: F8_E4M3 or U8"
                            ),
                        ));
                    }
                }
                // The tensor scales are the checkpoint's per expert and per projection: the fused
                // up/gate stack carries two scalars per expert and the down stack one. They are
                // published one packed table per projection, which is what the product reads, and a
                // table of another shape would dequantize one projection with another's scale, which
                // is a wrong product with nothing to show for it, so it is refused here.
                for (name, stack, projections) in [
                    ("up/gate", &up_gate.global_scale, 2),
                    ("down", &down.global_scale, 1),
                ] {
                    if stack.shape().dims() != [projections, u64::from(local)] {
                        return Err(Error::shape(
                            "Moe",
                            format!(
                                "the {name} tensor scales are {:?}, [{projections}, {local}] \
                                 expected: one packed table per projection",
                                stack.shape().dims()
                            ),
                        ));
                    }
                }
                // `[experts, 2·intermediate, hidden/2]` and `[experts, hidden, intermediate/2]`, so
                // the logical widths are twice the packed ones.
                let up = Shape::new(&[packed.dim(0), packed.dim(1), packed.dim(2) * 2]);
                let down = Shape::new(&[
                    contraction.dim(0),
                    contraction.dim(1),
                    contraction.dim(2) * 2,
                ]);
                (up, down)
            }
        };
        if up_gate_shape.rank() != 3 || up_gate_shape.dim(0) != u64::from(local) {
            return Err(Error::shape(
                "Moe",
                format!(
                    "the up/gate weight is {:?}, [{local}, 2·intermediate, hidden] expected",
                    up_gate_shape.dims()
                ),
            ));
        }
        let intermediate = up_gate_shape.dim(1) / 2;
        if down_shape.rank() != 3
            || down_shape.dim(0) != u64::from(local)
            || down_shape.dim(1) != up_gate_shape.dim(2)
            || down_shape.dim(2) != intermediate
        {
            return Err(Error::shape(
                "Moe",
                format!(
                    "the down weight is {:?}, [{local}, {}, {intermediate}] expected",
                    down_shape.dims(),
                    up_gate_shape.dim(2)
                ),
            ));
        }
        let hidden = u32::try_from(up_gate_shape.dim(2))
            .map_err(|_| Error::shape("Moe", "a hidden width above u32::MAX".to_owned()))?;
        let intermediate = u32::try_from(intermediate)
            .map_err(|_| Error::shape("Moe", "an expert width above u32::MAX".to_owned()))?;
        // The router reads the same rows the experts do, so its contraction length is the hidden
        // width the stacks just stated. A block whose router disagrees is a loading defect that would
        // otherwise surface as the router's own launch refusing its operand, three operators into a
        // forward and with none of the block's numbers in the message.
        if router.k() != hidden {
            return Err(Error::shape(
                "Moe",
                format!(
                    "the router contracts {} elements and the experts' rows are {hidden} wide",
                    router.k()
                ),
            ));
        }
        if let Some(map) = map
            && map.shape().numel() != u64::from(experts)
        {
            return Err(Error::shape(
                "Moe",
                format!(
                    "a map of {} entries for {experts} experts",
                    map.shape().numel()
                ),
            ));
        }
        if let Some(scale) = scale
            && scale.shape().numel() != u64::from(experts)
        {
            return Err(Error::shape(
                "Moe",
                format!(
                    "a scale of {} entries for {experts} experts",
                    scale.shape().numel()
                ),
            ));
        }
        Ok(Self {
            router,
            stacks,
            map,
            experts,
            top,
            hidden,
            intermediate,
            activation,
            renormalize,
            scale,
            fused: None,
            tile: 128,
            local,
            local_offset,
        })
    }

    /// An encoded block placed on one rank of a group whose exchange runs the experts in one launch.
    ///
    /// `up_gate` and `down` hold this rank's `[local, ...]` experts, in the layouts the fused kernel
    /// reads; `tables` are its per-expert scalars.
    #[allow(
        clippy::too_many_arguments,
        reason = "the block is named by its parts, as `expert_parallel` names them"
    )]
    pub fn fused_expert_parallel(
        router: Linear,
        up_gate: QuantizedStack,
        down: QuantizedStack,
        tables: FusedTables,
        map: Option<Weight>,
        experts: u32,
        local: u32,
        local_offset: u32,
        top: u32,
        activation: Activation,
        renormalize: bool,
    ) -> Result<Self> {
        let mut block = Self::build(
            router,
            ExpertStacks::Encoded {
                up_gate: Box::new(up_gate),
                down: Box::new(down),
            },
            map,
            experts,
            local,
            local_offset,
            top,
            activation,
            renormalize,
            None,
        )?;
        block.fused = Some(tables);
        Ok(block)
    }

    /// The same block placed on one rank of an expert-parallel group.
    ///
    /// `up_gate` and `down` hold this rank's `[local, ...]` experts only and `experts` is the
    /// router's whole vocabulary (the router is replicated, as the model's routing is). `local` is
    /// the number of experts the rank holds and `local_offset` the first of them in the router's
    /// order; the pair must tile the vocabulary across the group.
    #[allow(
        clippy::too_many_arguments,
        reason = "the block is named by its parts: the router, the two expert stacks, the map, \
                  the vocabulary, the resident window and the routing"
    )]
    pub fn expert_parallel(
        router: Linear,
        up_gate: Weight,
        down: Weight,
        map: Option<Weight>,
        experts: u32,
        local: u32,
        local_offset: u32,
        top: u32,
        activation: Activation,
        renormalize: bool,
        scale: Option<Weight>,
    ) -> Result<Self> {
        Self::build(
            router,
            ExpertStacks::Dense { up_gate, down },
            map,
            experts,
            local,
            local_offset,
            top,
            activation,
            renormalize,
            scale,
        )
    }

    /// The experts this rank holds.
    pub const fn local_experts(&self) -> u32 {
        self.local
    }

    /// The experts each token routes to.
    pub const fn top(&self) -> u32 {
        self.top
    }

    /// The number of experts the block routes over.
    pub const fn experts(&self) -> u32 {
        self.experts
    }

    /// Whether this block's experts are the checkpoint's encoded stacks.
    ///
    /// What the block launches depends on it: a dense block runs one grouped product per projection,
    /// an encoded one encodes each operand first and multiplies packed codes. A caller that
    /// enumerates programs — a key walk, a store's shopping list — asks this rather than the loading
    /// options, so the two cannot disagree about what the block will launch.
    pub const fn is_encoded(&self) -> bool {
        matches!(self.stacks, ExpertStacks::Encoded { .. })
    }

    /// The element type of the encoded stacks' block scales, or `None` for a dense block.
    pub fn scale_dtype(&self) -> Option<DType> {
        match &self.stacks {
            ExpertStacks::Dense { .. } => None,
            ExpertStacks::Encoded { up_gate, .. } => Some(up_gate.scales.dtype()),
        }
    }

    /// The layout the encoded stacks' block scales are stored in, or `None` for a dense block.
    pub fn scale_layout(&self) -> Option<ScaleLayout> {
        match &self.stacks {
            ExpertStacks::Dense { .. } => None,
            ExpertStacks::Encoded { up_gate, .. } => Some(up_gate.layout),
        }
    }

    /// The router's raw scores for `hidden`: `[tokens, experts]`, the gate's product before the
    /// top-k softens it.
    ///
    /// This is the array that separates a routing difference of *scale* from one of *selection*.
    /// The top-k operator applies the softmax itself, so the scores are the last place a factor can
    /// be seen directly: a score row a fraction of a reference's yields nearly equal weights, an
    /// order that follows that row rather than the reference's, and ids that mostly agree anyway —
    /// which no comparison of outputs can tell from a defect in the experts.
    ///
    /// `scores` is the caller's `[tokens, experts]` buffer of the input's dtype, so a dump reads
    /// host memory and a capture writes to the arena's.
    pub fn gate(&self, f: &Forward, hidden: TensorView, scores: TensorView) -> Result<()> {
        hidden
            .require_row("Moe::gate", hidden.dtype, u64::from(self.hidden))
            .map_err(Error::from)?;
        let expected = Shape::new(&[hidden.rows(), u64::from(self.experts)]);
        if scores.dtype != hidden.dtype || scores.shape != expected {
            return Err(Error::shape(
                "Moe::gate",
                format!(
                    "the scores are {expected:?} of {}, not {:?} of {}",
                    hidden.dtype, scores.shape, scores.dtype
                ),
            ));
        }
        self.router.forward(f, hidden, scores)
    }

    /// The routing prefix: the router, its top-k, the expert map, and the sort that groups the
    /// routes into the tiles a grouped product reads.
    ///
    /// Every expert product a block can run starts here and differs only in the two multiplies, so
    /// the prefix is one path: a variant that multiplies something else still routes, sorts and
    /// combines identically, and a heavy quantized product cannot drift from the dense one.
    /// The router's choice for `hidden`, and nothing else: the top-k expert ids and their
    /// weights, `[tokens, top]` i32 and F32, written into the caller's own buffers.
    ///
    /// The ids are the checkpoint's expert indices, *before* the expert map, so they are exactly
    /// what a reference's router names and a comparison of routing is an integer comparison. That
    /// keeps three findings apart that a whole MoE's output would confound: a wrong id (routing),
    /// a right id with a wrong product (the experts), and a right id with a wrong weight (the
    /// top-k's renormalization).
    ///
    /// No expert runs, so this costs one router row per token and the scores it needs, which are
    /// scratch inside `f`'s arena: nothing outlives the call but the two buffers written here. A
    /// caller that wants ids on the host passes views of its own memory; one that wants them in a
    /// capture's graph passes arena tensors, as the prefix does.
    pub fn route(
        &self,
        f: &Forward,
        hidden: TensorView,
        ids: TensorView,
        weights: TensorView,
    ) -> Result<()> {
        hidden
            .require_row("Moe::route", hidden.dtype, u64::from(self.hidden))
            .map_err(Error::from)?;

        let tokens = hidden.rows();
        let top = u64::from(self.top);
        let expected = Shape::new(&[tokens, top]);
        for (name, tensor, dtype) in [("ids", ids, DType::I32), ("weights", weights, DType::F32)] {
            if tensor.dtype != dtype {
                return Err(Error::shape(
                    "Moe::route",
                    format!("the {name} are {dtype}, not {}", tensor.dtype),
                ));
            }
            if tensor.shape != expected {
                return Err(Error::shape(
                    "Moe::route",
                    format!(
                        "the {name} are {expected:?} for {tokens} rows of top {top}, not {:?}",
                        tensor.shape
                    ),
                ));
            }
        }

        let experts = u64::from(self.experts);
        let scores = f
            .arena()
            .tensor(hidden.dtype, Shape::new(&[tokens, experts]))?;

        self.router.forward(f, hidden, scores)?;

        let routing = self.topk_descriptor(scores.dtype);
        let args = TopkArgs {
            scores,
            scale: self.scale.as_ref().map(Weight::view),
            ids,
            weights,
            map: self.map.as_ref().map(Weight::view),
        };

        f.operate::<TopkSoftmax>(&routing, args)?;

        Ok(())
    }

    fn routed(&self, f: &Forward, hidden: TensorView) -> Result<Routes> {
        let tokens = hidden.rows();
        let top = u64::from(self.top);
        let experts = u64::from(self.experts);
        let arena = f.arena();

        let ids = arena.tensor(DType::I32, Shape::new(&[tokens, top]))?;
        let weights = arena.tensor(DType::F32, Shape::new(&[tokens, top]))?;
        // The top-k writes each route's weight row: logical ids stop there.
        self.route(f, hidden, ids, weights)?;
        let local_ids = ids;

        // Group the routes by expert into the tiles the products read.
        let tiles = ops::moe::max_tiles(tokens, top, experts, u64::from(self.tile));
        let rows = ops::moe::max_permuted_rows(tokens, top, experts, u64::from(self.tile));
        let tile_to_expert = arena.tensor(DType::I32, Shape::new(&[tiles]))?;
        let tile_to_limit = arena.tensor(DType::I32, Shape::new(&[tiles]))?;
        let expanded_to_permuted = arena.tensor(DType::I32, Shape::new(&[tokens, top]))?;
        let permuted_to_expanded = arena.tensor(DType::I32, Shape::new(&[rows]))?;
        let permuted_to_token = arena.tensor(DType::I32, Shape::new(&[rows]))?;
        let total_padded_tokens = arena.tensor(DType::I32, Shape::new(&[1]))?;
        let num_non_exiting_tiles = arena.tensor(DType::I32, Shape::new(&[1]))?;
        let sort = self.sort_descriptor(self.experts);
        let args = SortArgs {
            ids: local_ids,
            scales: weights,
            tile_to_expert,
            tile_to_limit,
            expanded_to_permuted,
            permuted_to_expanded,
            permuted_to_token,
            total_padded_tokens,
            num_non_exiting_tiles,
            live_tokens: f.live_of(),
            local_offset: 0,
        };

        f.operate::<MoeSort>(&sort, args)?;

        Ok(Routes {
            local_ids,
            weights,
            rows,
            tile_to_expert,
            tile_to_limit,
            expanded_to_permuted,
            permuted_to_expanded,
            permuted_to_token,
            num_non_exiting_tiles,
        })
    }

    /// `out = Σ_routes weight · expert(hidden)`.
    ///
    /// `hidden` and `out` are `[tokens, hidden]`.
    pub fn forward(&self, f: &Forward, hidden: TensorView, out: TensorView) -> Result<()> {
        if self.local != self.experts || f.exchange.is_some() {
            return self.forward_expert_parallel(f, hidden, out);
        }
        let hidden_width = u64::from(self.hidden);
        hidden
            .require_row("Moe", hidden.dtype, hidden_width)
            .map_err(Error::from)?;
        out.require_row("Moe", out.dtype, hidden_width)
            .map_err(Error::from)?;
        if hidden.rows() != out.rows() {
            return Err(Error::shape(
                "Moe",
                format!(
                    "{} input rows against {} output rows",
                    hidden.rows(),
                    out.rows()
                ),
            ));
        }
        let ExpertStacks::Dense { up_gate, down } = &self.stacks else {
            // An encoded stack runs the packed products instead; the routing above is the same.
            return self.forward_encoded(f, hidden, out);
        };
        let intermediate = u64::from(self.intermediate);
        let arena = f.arena();

        // 1-3. The router, its top-k, the map and the sort: the prefix every expert product shares.
        let routes = self.routed(f, hidden)?;

        // 4. FC1: the gated product of each permuted row, then FC2.
        let gated = arena.tensor(hidden.dtype, Shape::new(&[routes.rows, intermediate]))?;
        let first = self.gather_descriptor(hidden.dtype);
        let args = GatherGemmArgs {
            hidden,
            weights: up_gate.view(),
            ids: routes.local_ids,
            // The map the product's reader uses, not the product's own lookup: FC1 writes row `p` at
            // the sort's permuted row `p`, and this is how a route is found again afterwards.
            expanded_to_permuted: routes.expanded_to_permuted,
            tile_to_expert: routes.tile_to_expert,
            tile_to_limit: routes.tile_to_limit,
            permuted_to_expanded: routes.permuted_to_expanded,
            num_non_exiting_tiles: routes.num_non_exiting_tiles,
            out: gated,
        };

        f.operate::<GatherGemm>(&first, args)?;

        // The combine indexes this buffer in the **permuted** row space, so it is permuted-height on
        // every path however few rows are live: FC2 writes only the rows it has, but it writes them
        // where the map says they are, not packed from zero.
        let products = arena.tensor(hidden.dtype, Shape::new(&[routes.rows, hidden_width]))?;
        let second = self.route_descriptor(hidden.dtype);
        let args = RouteGemmArgs {
            rows: gated,
            weights: down.view(),
            ids: routes.local_ids,
            tile_to_expert: routes.tile_to_expert,
            tile_to_limit: routes.tile_to_limit,
            permuted_to_expanded: routes.permuted_to_expanded,
            num_non_exiting_tiles: routes.num_non_exiting_tiles,
            out: products,
        };

        f.operate::<RouteGemm>(&second, args)?;

        // 5. Sum the routes back in FP32 and round once to the hidden dtype.
        let combine = self.combine_descriptor(products.dtype, out.dtype);
        let args = UnpermuteArgs {
            routes: products,
            expanded_to_permuted: routes.expanded_to_permuted,
            scales: routes.weights,
            out,
        };

        f.operate::<MoeUnpermute>(&combine, args)
    }

    /// The routing descriptor: the experts' softmax, the top-k, and whether the kept weights are
    /// renormalized. The scale's type is the model's — a calibrated router carries one.
    pub fn topk_descriptor(&self, scores: DType) -> TopkDescriptor {
        self.geometry().topk_descriptor(
            scores,
            self.renormalize,
            self.scale.as_ref().map(Weight::dtype),
        )
    }

    /// The block's shape and its rank's expert window, as the value every descriptor is built from.
    ///
    /// The layer's descriptor methods delegate here, so an enumeration that has only a
    /// configuration builds the same programs the loaded model launches.
    pub fn geometry(&self) -> MoeGeometry {
        MoeGeometry {
            hidden: u64::from(self.hidden),
            intermediate: u64::from(self.intermediate),
            top: u64::from(self.top),
            experts: u64::from(self.experts),
            tile: u64::from(self.tile),
        }
    }

    /// The sort's descriptor, for a block whose rank holds `local_experts` experts.
    ///
    /// `local_experts` is an argument and not the layer's field because the same layer sorts for
    /// the dense block (every expert) and for an expert-parallel rank (its own), and the two are
    /// different programs.
    pub fn sort_descriptor(&self, local_experts: u32) -> sort::Descriptor {
        self.geometry().sort_descriptor(local_experts)
    }

    /// The first product's descriptor: the gated FC1 over the dense packed stack.
    pub fn gather_descriptor(&self, hidden: DType) -> GatherGemmDescriptor {
        self.geometry().gather_descriptor(hidden, self.activation)
    }

    /// The second product's descriptor: FC2 over the dense packed stack.
    pub fn route_descriptor(&self, hidden: DType) -> RouteGemmDescriptor {
        self.geometry().route_descriptor(hidden)
    }

    /// The scratch one launch of the encoded products needs, for `routes` permuted rows — the exact
    /// number each product's entry compares its request against.
    ///
    /// One buffer serves both products — they run one after the other on the stream and neither keeps
    /// a view past its launch — so the answer is the larger of the two, and each side is its entry's
    /// preamble plus the artifact's own requirement: the gated first product writes
    /// `2 · intermediate` columns, the second `hidden`. At the caller's geometry (15 360 rows, 768
    /// intermediate, 2 048 hidden) that is 6 160 + 740 356 for the first and 8 208 + 987 140 for the
    /// second, so 995 348 bytes.
    pub fn workspace_bytes(&self, routes: u64) -> u64 {
        encoded_scratch_bytes(routes, u64::from(self.hidden), u64::from(self.intermediate))
    }

    /// The buffers this block's encoded step allocates for `tokens` rows — the same table the plan
    /// sizes the arena from, so what the step reserves and what it launches cannot drift.
    pub fn encoded_step(
        &self,
        tokens: u64,
        layout: ScaleLayout,
        gated_layout: ScaleLayout,
        dtype: DType,
    ) -> Result<EncodedStep> {
        EncodedStep::new(
            tokens,
            MoeGeometry {
                hidden: u64::from(self.hidden),
                intermediate: u64::from(self.intermediate),
                top: u64::from(self.top),
                experts: u64::from(self.experts),
                tile: u64::from(self.tile),
            },
            layout,
            gated_layout,
            dtype,
        )
    }

    /// The bytes one routed layer's whole expert step reserves from the arena, for `tokens` rows —
    /// the routing prefix included.
    ///
    /// [`step_bytes`] states the extents; this answers them for this block's geometry, which is what
    /// a deployment asks when it sizes the arena from its plan.
    pub fn step_bytes(
        &self,
        tokens: u64,
        layout: ScaleLayout,
        gated_layout: ScaleLayout,
        dtype: DType,
    ) -> Result<u64> {
        step_bytes(
            tokens,
            MoeGeometry {
                hidden: u64::from(self.hidden),
                intermediate: u64::from(self.intermediate),
                top: u64::from(self.top),
                experts: u64::from(self.experts),
                tile: u64::from(self.tile),
            },
            layout,
            gated_layout,
            dtype,
            matches!(self.stacks, ExpertStacks::Encoded { .. }),
        )
    }

    /// The descriptor of one activation encoding this block runs.
    ///
    /// The chunk a program walks a row in is the block: the encoder reduces over the whole row and
    /// scales every block of it against that row's factor, so the chunk is how much of the row one
    /// pass holds, not how many values a key spans.
    pub fn quantize_descriptor(
        &self,
        input: DType,
        columns: u32,
        layout: ScaleLayout,
    ) -> QuantizeDescriptor {
        Self::quantize_descriptor_of(input, columns, layout)
    }

    /// The encoder's descriptor, from the contraction length alone.
    ///
    /// The encoder's grid covers the rows in one launch — it is keyed by `columns`, not by a row
    /// count — so it is not compiled per bucket, and every activation the deployment encodes shares
    /// one program per width.
    pub fn quantize_descriptor_of(
        input: DType,
        columns: u32,
        layout: ScaleLayout,
    ) -> QuantizeDescriptor {
        QuantizeDescriptor {
            input,
            columns,
            layout,
            block: crate::layers::linear::Nvfp4Linear::encode_block(columns, ENCODE_CHUNK),
            pdl: crate::layers::launch::PDL,
        }
    }

    /// The first product's descriptor over an encoded stack: the block scale's element type and the
    /// scale layout are the weights', so they are arguments.
    pub fn gather_nvfp4_descriptor(
        &self,
        hidden: DType,
        scale: DType,
        layout: ScaleLayout,
    ) -> GatherGemmNvfp4Descriptor {
        self.geometry()
            .gather_nvfp4_descriptor(hidden, self.activation, scale, layout)
    }

    /// The second product's descriptor over an encoded stack.
    pub fn route_nvfp4_descriptor(
        &self,
        hidden: DType,
        scale: DType,
        layout: ScaleLayout,
    ) -> RouteGemmNvfp4Descriptor {
        self.geometry()
            .route_nvfp4_descriptor(hidden, scale, layout)
    }

    /// The combine's descriptor: the routes summed in FP32 and rounded once to `output`.
    ///
    /// The routes arrive in **permuted** rows, which is the convention `route_gemm` writes and the
    /// one the combine's module is compiled for; it is a descriptor field so a chain that ever
    /// writes expanded rows names its own module instead of reading this one's buffer wrongly.
    pub fn combine_descriptor(&self, product: DType, output: DType) -> combine::Descriptor {
        self.geometry().combine_descriptor(product, output)
    }

    /// `out = Σ_routes weight · expert(hidden)` over packed experts.
    ///
    /// The routing, the sort and the combine are the dense block's; the two products read the
    /// encoded stacks with E2M1 activations, so each of their operands is produced here by
    /// `nvfp4_quantize`:
    ///
    /// ```text
    /// encode(hidden [tokens, hidden])            -> codes, block scales, one factor per token
    /// gather_gemm_nvfp4                          -> gated rows [routes, intermediate], BF16
    /// encode(gated [routes, intermediate])        -> codes, block scales, one factor per route
    /// route_gemm_nvfp4                           -> rows [routes, hidden], BF16
    /// moe_unpermute                              -> out [tokens, hidden], FP32 sum, one rounding
    /// ```
    ///
    /// The encoder's global scale is the stack's [`QuantizedStack::encode_scale`] — one — because a
    /// calibrated activation scale cancels out of both the code and the reconstruction; the expert's
    /// own scales reach the product through its `[projections, experts]` table. No row-expansion
    /// operator appears: the encoded gather is token-major and expands as it multiplies.
    fn forward_encoded(&self, f: &Forward, hidden: TensorView, out: TensorView) -> Result<()> {
        let ExpertStacks::Encoded { up_gate, down } = &self.stacks else {
            return Err(Error::Unsupported(
                "Moe::forward_encoded: the block's experts are dense".to_owned(),
            ));
        };
        let tokens = hidden.rows();
        let hidden_width = u64::from(self.hidden);
        let intermediate = u64::from(self.intermediate);
        hidden
            .require_row("Moe", hidden.dtype, hidden_width)
            .map_err(Error::from)?;
        out.require_row("Moe", out.dtype, hidden_width)
            .map_err(Error::from)?;
        if out.rows() != tokens {
            return Err(Error::shape(
                "Moe",
                format!("{} input rows against {} output rows", tokens, out.rows()),
            ));
        }

        // 1-3. The router, its top-k, the map and the sort: the prefix every expert product shares.
        let routes = self.routed(f, hidden)?;
        let arena = f.arena();

        // 4. The activation's encoding: the same pass computes the block scales and the row factor
        //    the product multiplies by, and its global scale is the stack's neutral.
        //
        // Two activation encodings, because the artifact's two families are two: the gated product
        // (the artifact's `swiGlu` kernels) reads its operand's block scales in the linear form and
        // the plain product's kernels read them swizzled (`flashinferMetaInfo.h`: the gated family's
        // tokens operand is `SfLayout(0)`, the plain family's `SfLayout(3)`). One encoding for both is
        // the defect that made the first end-to-end NVFP4 run produce a row no argmax could read.
        let input_layout = Self::GATED_TOKEN_SCALE_LAYOUT;
        let gated_layout = Self::ACTIVATION_SCALE_LAYOUT;
        // The step's whole working set from one table: the plan sums the same extents to size the
        // arena, so every buffer a launch writes is a buffer the plan reserved.
        let step = self.encoded_step(tokens, input_layout, gated_layout, hidden.dtype)?;
        let values = arena.tensor(step.values.dtype, step.values.shape)?;
        let scales = arena.tensor(step.scales.dtype, step.scales.shape)?;
        let token_scale = arena.tensor(step.token_scale.dtype, step.token_scale.shape)?;
        let first = self.quantize_descriptor(
            hidden.dtype,
            u32::try_from(hidden_width)
                .map_err(|_| Error::shape("Moe", format!("{hidden_width} hidden elements")))?,
            input_layout,
        );
        let args = QuantizeArgs {
            input: hidden,
            values,
            scales,
            global_scale: up_gate.encode_scale.view(),
            token_scale,
        };
        f.operate::<Nvfp4Quantize>(&first, args)?;

        // 5. FC1: the token-major gather product, one gated row per permuted route.
        let gated = arena.tensor(step.gated.dtype, step.gated.shape)?;
        let workspace = arena.tensor(step.scratch.dtype, step.scratch.shape)?;
        // The scale's element type is the **checkpoint's**, read off the stack it belongs to; the
        // layout the product states is the **tokens plane's**, because that is the plane it reads
        // and the one this encoding wrote. The weights' own form is not in the descriptor: it is a
        // property of `up_gate`'s stack, published in the form this product's kernels index.
        let descriptor = self.gather_nvfp4_descriptor(
            hidden.dtype,
            up_gate.scales.dtype(),
            Self::GATED_TOKEN_SCALE_LAYOUT,
        );
        let args = GatherGemmNvfp4Args {
            hidden: values,
            hidden_scale: scales,
            hidden_token_scale: token_scale,
            weights: up_gate.values.view(),
            scales: up_gate.scales.view(),
            global_scale: up_gate.global_scale.view(),
            tile_to_expert: routes.tile_to_expert,
            tile_to_limit: routes.tile_to_limit,
            permuted_to_token: routes.permuted_to_token,
            num_non_exiting_tiles: routes.num_non_exiting_tiles,
            out: gated,
            workspace,
        };
        f.operate::<GatherGemmNvfp4>(&descriptor, args)?;

        // 6. The gated rows' own encoding: FC1's epilogue is a BF16 row, so this is a program of its
        //    own rather than a wider operand of the second product.
        let gated_values = arena.tensor(step.gated_values.dtype, step.gated_values.shape)?;
        let gated_scales = arena.tensor(step.gated_scales.dtype, step.gated_scales.shape)?;
        let gated_token_scale =
            arena.tensor(step.gated_token_scale.dtype, step.gated_token_scale.shape)?;
        let second = self.quantize_descriptor(
            hidden.dtype,
            u32::try_from(intermediate).map_err(|_| {
                Error::shape("Moe", format!("{intermediate} intermediate elements"))
            })?,
            gated_layout,
        );
        let args = QuantizeArgs {
            input: gated,
            values: gated_values,
            scales: gated_scales,
            global_scale: down.encode_scale.view(),
            token_scale: gated_token_scale,
        };
        f.operate::<Nvfp4Quantize>(&second, args)?;

        // 7. FC2: one row per permuted route, the `top` repeats still separate.
        let products = arena.tensor(step.products.dtype, step.products.shape)?;
        let descriptor = self.route_nvfp4_descriptor(
            hidden.dtype,
            down.scales.dtype(),
            Self::ACTIVATION_SCALE_LAYOUT,
        );
        let args = RouteGemmNvfp4Args {
            hidden: gated_values,
            hidden_scale: gated_scales,
            hidden_token_scale: gated_token_scale,
            weights: down.values.view(),
            scales: down.scales.view(),
            global_scale: down.global_scale.view(),
            tile_to_expert: routes.tile_to_expert,
            tile_to_limit: routes.tile_to_limit,
            permuted_to_expanded: routes.permuted_to_expanded,
            num_non_exiting_tiles: routes.num_non_exiting_tiles,
            out: products,
            workspace,
        };
        f.operate::<RouteGemmNvfp4>(&descriptor, args)?;

        // 8. Sum the routes back in FP32 and round once to the hidden dtype.
        let combine = self.combine_descriptor(products.dtype, out.dtype);
        let args = UnpermuteArgs {
            routes: products,
            expanded_to_permuted: routes.expanded_to_permuted,
            scales: routes.weights,
            out,
        };
        f.operate::<MoeUnpermute>(&combine, args)
    }

    /// `out = Σ_routes weight · expert(hidden)` over an expert-parallel group.
    ///
    /// The rank routes every token over the whole vocabulary, sends each token's payload to the
    /// ranks that hold its experts (`dispatch`), multiplies the routes it was sent with its own
    /// experts, and sums the ranks' partial results back into its own rows (`combine`,
    /// specs/worker.md 9's expert rows, `uniserve/backends/experts/alltoall.py` today).
    ///
    /// `exchange` is the executor's bundle for this group: the package's transport and the views of
    /// its workspace — the rows this rank received at dispatch, with their ids and weights, in the
    /// package's capacity-sized layout. The layer never allocates or owns any of it.
    ///
    /// Every buffer is the worst case for the group's capacity, so a captured graph replays over
    /// the same addresses however the router routes a later batch.
    pub fn forward_expert_parallel(
        &self,
        f: &Forward,
        hidden: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let exchange = f.exchange.ok_or_else(|| {
            Error::shape(
                "Moe",
                "an expert-parallel block reads the group's exchange views, and the forward \
                 borrows none"
                    .to_owned(),
            )
        })?;
        exchange.require_extents("Moe")?;
        let hidden_width = u64::from(self.hidden);
        let top = u64::from(self.top);
        let ranks = u64::from(exchange.ranks);
        let capacity = u64::from(exchange.capacity);
        let received = u64::from(exchange.ranks) * capacity;
        let tokens = hidden.rows();

        hidden
            .require_row("Moe", hidden.dtype, hidden_width)
            .map_err(Error::from)?;
        out.require_row("Moe", out.dtype, hidden_width)
            .map_err(Error::from)?;
        if out.rows() != tokens {
            return Err(Error::shape(
                "Moe",
                format!("{tokens} input rows against {} output rows", out.rows()),
            ));
        }
        // The group's geometry has to agree with the layout the block was built for.
        let expected_ranks = u64::from(self.experts / self.local.max(1));
        if ranks - u64::from(exchange.attention) != expected_ranks {
            return Err(Error::shape(
                "Moe",
                format!(
                    "the exchange has {ranks} ranks, {} experts over this rank's {} tile into \
                     {expected_ranks}",
                    self.experts, self.local
                ),
            ));
        }
        if capacity == 0 || received == 0 {
            return Err(Error::shape(
                "Moe",
                "the exchange's capacity is empty: a rank sends at least one row".to_owned(),
            ));
        }
        if tokens > capacity {
            return Err(Error::shape(
                "Moe",
                format!("the batch has {tokens} tokens, the exchange carries {capacity} per rank"),
            ));
        }
        for (name, view, dtype) in [
            ("ids", exchange.ids, DType::I32),
            ("weights", exchange.weights, DType::F32),
            ("live", exchange.live, DType::I32),
        ] {
            if view.dtype != dtype {
                return Err(Error::shape(
                    "Moe",
                    format!("the exchange's {name} are {}, not {dtype}", view.dtype),
                ));
            }
        }
        if exchange.states.dtype != hidden.dtype {
            return Err(Error::shape(
                "Moe",
                format!(
                    "the exchange carries {} rows, the batch is {}",
                    exchange.states.dtype, hidden.dtype
                ),
            ));
        }

        let experts = u64::from(self.experts);
        let arena = f.arena();

        // 1. The replicated router and its top-k: every rank routes whole tokens the same way.
        let scores = arena.tensor(hidden.dtype, Shape::new(&[tokens, experts]))?;
        let ids = arena.tensor(DType::I32, Shape::new(&[tokens, top]))?;
        let weights = arena.tensor(DType::F32, Shape::new(&[tokens, top]))?;
        self.router.forward(f, hidden, scores)?;
        let routing = self.topk_descriptor(scores.dtype);
        // 2. The top-k writes weight rows, so logical ids stop before anything leaves the rank and
        //    a peer sorts by the same physical order the weights are stacked in.
        let args = TopkArgs {
            scores,
            scale: self.scale.as_ref().map(Weight::view),
            ids,
            weights,
            map: self.map.as_ref().map(Weight::view),
        };

        f.operate::<TopkSoftmax>(&routing, args)?;
        let sent = ids;

        self.exchange_routes(f, exchange, hidden, sent, weights, out)
    }

    /// Takes part in the group's step with no tokens of this rank's own.
    ///
    /// Every rank runs every expert layer of a step, because a dispatch waits on its device for the
    /// same dispatch on every rank. A rank with no batch of its own calls this: its experts serve the
    /// tokens routed to them and it sends none.
    pub fn join_expert_step(&self, f: &Forward) -> Result<()> {
        let exchange = f.exchange.ok_or_else(|| {
            Error::shape(
                "Moe",
                "a join reads the group's exchange views, and the forward borrows none".to_owned(),
            )
        })?;
        exchange.require_extents("Moe")?;
        let width = u64::from(self.hidden);
        let top = u64::from(self.top);
        let base = DevicePtr::new(0);
        let rows = TensorView::new(base, exchange.states.dtype, Shape::new(&[0, width]));
        let ids = TensorView::new(base, DType::I32, Shape::new(&[0, top]));
        let weights = TensorView::new(base, DType::F32, Shape::new(&[0, top]));
        self.exchange_routes(f, exchange, rows, ids, weights, rows)
    }

    /// Dispatches `hidden`'s routes, runs this rank's experts over what it received, and combines
    /// each token's partial sums into `out`.
    fn exchange_routes(
        &self,
        f: &Forward,
        exchange: &crate::exchange::ExchangeViews,
        hidden: TensorView,
        sent: TensorView,
        weights: TensorView,
        out: TensorView,
    ) -> Result<()> {
        if exchange.fused {
            return self.fused_routes(f, exchange, hidden, sent, weights, out);
        }
        let ExpertStacks::Dense { up_gate, down } = &self.stacks else {
            return Err(Error::Unsupported(
                "Moe::forward_expert_parallel: an encoded stack has no exchange path yet; the \
                 quantized block is colocated"
                    .to_owned(),
            ));
        };
        // The received rows this walk's graph is shaped for, which the agreed step fits.
        let narrowed = exchange.narrowed();
        let exchange = &narrowed;
        let hidden_width = u64::from(self.hidden);
        let top = u64::from(self.top);
        let tokens = hidden.rows();
        let local = u64::from(self.local);
        let intermediate = u64::from(self.intermediate);
        let received = u64::from(exchange.ranks) * u64::from(exchange.step_rows);
        let arena = f.arena();
        let launch_tokens = u32::try_from(tokens)
            .map_err(|_| Error::shape("Moe", format!("{tokens} tokens exceed u32")))?;
        let empty = TensorView::new(DevicePtr::new(0), DType::U8, Shape::new(&[0]));

        // 3. Dispatch: each token's payload goes to the ranks that hold its experts.
        let exchange_descriptor = |mode: Mode| exchange::Descriptor {
            mode,
            hidden: self.hidden,
            intermediate: self.intermediate,
            top: self.top,
            experts: self.experts,
            max_tokens: exchange.capacity,
            ranks: exchange.ranks,
            attention: exchange.attention,
            dtype: exchange.states.dtype,
        };
        let dispatch = exchange_descriptor(Mode::Dispatch);
        let args = ExchangeArgs {
            transport: exchange.transport,
            tokens: launch_tokens,
            state: hidden,
            ids: sent,
            weights,
            live: f.live_of(),
            out: empty,
            fused: None,
            ranks: exchange.ranks,
        };

        (exchange.meet)()?;
        f.operate::<ExpertExchange>(&dispatch, args)?;
        f.settled()?;

        // A rank that holds no experts receives nothing: its routes went to the expert ranks, and
        // only their sums come back.
        if exchange.source_only {
            return self.combine_routes(f, exchange, launch_tokens, out);
        }

        // 4. The rows this rank was sent, grouped by its own experts.
        let tiles = ops::moe::max_tiles(received, top, local, u64::from(self.tile));
        let rows = ops::moe::max_permuted_rows(received, top, local, u64::from(self.tile));
        let tile_to_expert = arena.tensor(DType::I32, Shape::new(&[tiles]))?;
        let tile_to_limit = arena.tensor(DType::I32, Shape::new(&[tiles]))?;
        let expanded_to_permuted = arena.tensor(DType::I32, Shape::new(&[received, top]))?;
        let permuted_to_expanded = arena.tensor(DType::I32, Shape::new(&[rows]))?;
        let permuted_to_token = arena.tensor(DType::I32, Shape::new(&[rows]))?;
        let total_padded_tokens = arena.tensor(DType::I32, Shape::new(&[1]))?;
        let num_non_exiting_tiles = arena.tensor(DType::I32, Shape::new(&[1]))?;
        let sort = self.sort_descriptor(self.local);
        let args = SortArgs {
            ids: exchange.ids,
            scales: exchange.weights,
            tile_to_expert,
            tile_to_limit,
            expanded_to_permuted,
            permuted_to_expanded,
            permuted_to_token,
            total_padded_tokens,
            num_non_exiting_tiles,
            // The rows the step holds, which the agreed capacity decides on the device: the views'
            // extents are the transport's largest step, and the rows past the count are left over
            // from an earlier one.
            live_tokens: Some(exchange.live),
            local_offset: self.local_offset,
        };

        f.operate::<MoeSort>(&sort, args)?;

        // 5. FC1 and FC2 over the received rows, with this rank's experts.
        //
        // The products name the resident stack's expert count, not the vocabulary's: `up_gate` and
        // `down` hold `local` experts, and the sort rebased every received route into that window
        // with `local_offset`, so the vocabulary count would name a stack this rank does not hold
        // (and the operator's shape check would refuse the launch).
        let gated = arena.tensor(exchange.states.dtype, Shape::new(&[rows, intermediate]))?;
        let first = GatherGemmDescriptor {
            hidden: self.hidden,
            intermediate: self.intermediate,
            experts: self.local,
            top: self.top,
            activation: self.activation,
            dtype: exchange.states.dtype,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        };
        let args = GatherGemmArgs {
            hidden: exchange.states,
            weights: up_gate.view(),
            ids: exchange.ids,
            expanded_to_permuted,
            tile_to_expert,
            tile_to_limit,
            permuted_to_expanded,
            num_non_exiting_tiles,
            out: gated,
        };

        f.operate::<GatherGemm>(&first, args)?;

        // Permuted-height for the reason the dense path's is: the combine below indexes it through
        // `expanded_to_permuted`.
        let products = arena.tensor(exchange.states.dtype, Shape::new(&[rows, hidden_width]))?;
        let second = RouteGemmDescriptor {
            hidden: self.hidden,
            intermediate: self.intermediate,
            experts: self.local,
            top: self.top,
            dtype: exchange.states.dtype,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        };
        let args = RouteGemmArgs {
            rows: gated,
            weights: down.view(),
            ids: exchange.ids,
            tile_to_expert,
            tile_to_limit,
            permuted_to_expanded,
            num_non_exiting_tiles,
            out: products,
        };

        f.operate::<RouteGemm>(&second, args)?;

        // 6. Each received row's routes summed with their FP32 weights, in receive order, straight
        //    into the transport's view the peers read back.
        let combine_descriptor = combine::Descriptor {
            layout: combine::Layout::Permuted,
            hidden: self.hidden,
            top: self.top,
            experts: self.experts,
            input: products.dtype,
            scales: DType::F32,
            output: exchange.states.dtype,
        };
        let args = UnpermuteArgs {
            routes: products,
            expanded_to_permuted,
            scales: exchange.weights,
            out: exchange.combined,
        };

        f.operate::<MoeUnpermute>(&combine_descriptor, args)?;

        self.combine_routes(f, exchange, launch_tokens, out)
    }

    /// One launch exchanges the routes, runs every rank's experts and combines.
    ///
    /// The rows are encoded straight into the transport's symmetric views, where the peers read
    /// them, with the same encoder the stack's other products take: the encode is the layer's, and
    /// the kernel starts from the bytes.
    fn fused_routes(
        &self,
        f: &Forward,
        exchange: &crate::exchange::ExchangeViews,
        hidden: TensorView,
        sent: TensorView,
        weights: TensorView,
        out: TensorView,
    ) -> Result<()> {
        let ExpertStacks::Encoded { up_gate, down } = &self.stacks else {
            return Err(Error::Unsupported(
                "Moe: a fused exchange runs encoded experts, and this block's are dense".to_owned(),
            ));
        };
        let tables = self.fused.as_ref().ok_or_else(|| {
            Error::Unsupported(
                "Moe: this block was loaded without the tables a fused exchange reads".to_owned(),
            )
        })?;
        let tokens = hidden.rows();
        let launch_tokens = u32::try_from(tokens)
            .map_err(|_| Error::shape("Moe", format!("{tokens} tokens exceed u32")))?;
        if tokens > 0 {
            let token_scale = f.arena().tensor(DType::F32, Shape::new(&[tokens]))?;
            let descriptor =
                self.quantize_descriptor(hidden.dtype, self.hidden, ScaleLayout::Linear);
            let args = QuantizeArgs {
                input: hidden,
                values: TensorView::new(
                    exchange.packed.ptr,
                    exchange.packed.dtype,
                    Shape::new(&[tokens, exchange.packed.shape.last()]),
                ),
                scales: TensorView::new(
                    exchange.packed_scales.ptr,
                    DType::U8,
                    Shape::new(&[tokens, exchange.packed_scales.shape.last()]),
                ),
                global_scale: up_gate.encode_scale.view(),
                token_scale,
            };
            f.operate::<Nvfp4Quantize>(&descriptor, args)?;
        }
        let empty = TensorView::new(DevicePtr::new(0), DType::U8, Shape::new(&[0]));
        let address = |weight: &Weight| weight.view().ptr.address();
        let bytes = |weight: &Weight| weight.view().numel() / u64::from(self.local);
        let layer = exchange::FusedLayer {
            up_gate: address(&up_gate.values),
            up_gate_scales: address(&up_gate.scales),
            down: address(&down.values),
            down_scales: address(&down.scales),
            up_gate_alpha: address(&tables.up_gate_alpha),
            down_alpha: address(&tables.down_alpha),
            gated_scale: address(&tables.gated_scale),
            up_gate_scale_bytes: i64::try_from(bytes(&up_gate.scales))
                .map_err(|_| Error::shape("Moe", "a scale table beyond i64".to_owned()))?,
            down_scale_bytes: i64::try_from(bytes(&down.scales))
                .map_err(|_| Error::shape("Moe", "a scale table beyond i64".to_owned()))?,
        };
        let descriptor = exchange::Descriptor {
            mode: Mode::Fused,
            hidden: self.hidden,
            intermediate: self.intermediate,
            top: self.top,
            experts: self.experts,
            max_tokens: exchange.capacity,
            ranks: exchange.ranks,
            attention: exchange.attention,
            dtype: out.dtype,
        };
        let args = ExchangeArgs {
            transport: exchange.transport,
            tokens: launch_tokens,
            state: empty,
            ids: sent,
            weights,
            live: f.live_of(),
            out,
            fused: Some(layer),
            ranks: exchange.ranks,
        };
        (exchange.meet)()?;
        f.operate::<ExpertExchange>(&descriptor, args)?;
        f.settled()
    }

    /// Combine: the ranks' partial sums for this rank's tokens, rounded once to `out`.
    fn combine_routes(
        &self,
        f: &Forward,
        exchange: &crate::exchange::ExchangeViews,
        tokens: u32,
        out: TensorView,
    ) -> Result<()> {
        let empty = TensorView::new(DevicePtr::new(0), DType::U8, Shape::new(&[0]));
        let descriptor = exchange::Descriptor {
            mode: Mode::Combine,
            hidden: self.hidden,
            intermediate: self.intermediate,
            top: self.top,
            experts: self.experts,
            max_tokens: exchange.capacity,
            ranks: exchange.ranks,
            attention: exchange.attention,
            dtype: exchange.states.dtype,
        };
        let args = ExchangeArgs {
            transport: exchange.transport,
            tokens,
            state: empty,
            ids: empty,
            weights: empty,
            live: f.live_of(),
            out,
            fused: None,
            ranks: exchange.ranks,
        };

        (exchange.meet)()?;
        f.operate::<ExpertExchange>(&descriptor, args)?;
        f.settled()
    }

    /// The bound the combined rows' implementations may differ by.
    ///
    /// A dense block combines its own element type; an encoded one combines BF16, the element type
    /// a calibrated NVFP4 checkpoint's consumers run in.
    pub fn tolerance(&self) -> ops::Tolerance {
        let dtype = match &self.stacks {
            ExpertStacks::Dense { up_gate, .. } => up_gate.dtype(),
            ExpertStacks::Encoded { .. } => DType::BF16,
        };
        MoeUnpermute::tolerance(&combine::Descriptor {
            layout: combine::Layout::Permuted,
            hidden: self.hidden,
            top: self.top,
            experts: self.experts,
            input: dtype,
            scales: DType::F32,
            output: dtype,
        })
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::tensor::Backing;
    use ops::gemm::WeightOrder;
    use ops::{DevicePtr, Shape};

    /// A weight over a fake region: the constructor only reads its shape and dtype.
    fn weight(dtype: DType, shape: &[u64]) -> Weight {
        Weight {
            backing: Backing::new(DevicePtr::new(0x1_0000), 1 << 20),
            view: TensorView::new(DevicePtr::new(0x1_0000), dtype, Shape::new(shape)),
        }
    }

    fn block(experts: u32, top: u32, hidden: u64, intermediate: u64) -> Result<Moe> {
        let router = Linear::new(
            weight(DType::BF16, &[u64::from(experts), hidden]),
            None,
            WeightOrder::NxK,
        )?;
        Moe::new(
            router,
            weight(DType::BF16, &[u64::from(experts), 2 * intermediate, hidden]),
            weight(DType::BF16, &[u64::from(experts), hidden, intermediate]),
            None,
            experts,
            top,
            Activation::Silu,
            true,
            None,
        )
    }

    /// An expert-parallel block over `local` experts of `experts` at `offset`.
    pub(super) fn expert_parallel_block(experts: u32, local: u32, offset: u32) -> Result<Moe> {
        let router = Linear::new(
            weight(DType::BF16, &[u64::from(experts), 2048]),
            None,
            WeightOrder::NxK,
        )?;
        Moe::expert_parallel(
            router,
            weight(DType::BF16, &[u64::from(local), 1536, 2048]),
            weight(DType::BF16, &[u64::from(local), 2048, 768]),
            None,
            experts,
            local,
            offset,
            8,
            Activation::Silu,
            true,
            None,
        )
    }

    /// A quantized block's stacks are packed: the same Qwen3-MoE geometry, two codes a byte along
    /// the contraction, one block scale per sixteen elements and one reciprocal per expert.
    fn quantized_block(hidden: u64, intermediate: u64) -> Result<Moe> {
        let experts = 128;
        let router = Linear::new(
            weight(DType::BF16, &[u64::from(experts), hidden]),
            None,
            WeightOrder::NxK,
        )?;
        let stack = |rows: u64, columns: u64, projections: u64| QuantizedStack {
            values: weight(DType::U8, &[u64::from(experts), rows, columns / 2]),
            scales: weight(DType::U8, &[u64::from(experts), rows, columns / 16]),
            global_scale: weight(DType::F32, &[projections, u64::from(experts)]),
            layout: ops::fp4::ScaleLayout::Linear,
            encode_scale: weight(DType::F32, &[1]),
            input_scale: None,
            activation_scale: 1.0,
        };
        Moe::quantized(
            router,
            stack(2 * intermediate, hidden, 2),
            stack(hidden, intermediate, 1),
            None,
            experts,
            8,
            Activation::Silu,
            true,
            None,
        )
    }

    /// The scratch is the plan's, computed from the artifact's arithmetic and each entry's preamble: a
    /// bucket of 8 routes over 128 experts costs the row-max planes and tile bars of the wider product,
    /// and a bucket twice as wide costs more. What it must not be is a floor: the size the largest
    /// Qwen3-MoE case needs is far below the sixteen mebibytes an earlier constant reserved for it.
    ///
    /// The number is the one an entry compares its own request against, so it is stated for the
    /// geometry a deployment runs: the second product's entry asks 16 + 987 140.
    #[test]
    fn the_scratch_is_sized_from_the_route_count_and_not_from_a_floor() {
        let quantized = quantized_block(2048, 768).unwrap();
        let small = quantized.workspace_bytes(8);
        let large = quantized.workspace_bytes(4096);
        assert!(small > 0);
        assert!(large > small, "{large} is not more than {small}");
        assert!(large < 16 << 20, "{large} exceeds the floor it replaced");
        // 15 tokens over 128 experts is 120 permuted tiles of 128 rows: the routed rows the caller
        // launches. The gated product needs 16 + 740 356 and the contraction 16 + 987 140, so the
        // shared buffer is the contraction's.
        assert_eq!(quantized.workspace_bytes(15_360), 987_156);
        // A bucket small enough that the artifact's planes are a single tile still carries the
        // entry's preamble.
        assert_eq!(quantized.workspace_bytes(8), 9_236);
    }

    /// The encoded step states one buffer per launch and the plan states the same extents: the two
    /// readers of the table cannot disagree, which is the defect class the step has had twice (a term
    /// missing from the sum, and a buffer reserved at another height than the one it is written at).
    #[test]
    fn the_encoded_step_states_a_buffer_for_every_launch_and_the_plan_sizes_them() {
        let moe = quantized_block(2048, 768).expect("the encoded block");
        let tokens = 15;
        let step = moe
            .encoded_step(
                tokens,
                ScaleLayout::Linear,
                ScaleLayout::Swizzled128X4,
                DType::BF16,
            )
            .expect("the step's buffers");
        // The permutation's worst case: 120 expanded routes over 128 experts at 128-row tiles is one
        // tile per route, so the step's rows are 15 360 and not 120.
        let routes = 15_360;
        assert_eq!(
            step.values.shape.dims(),
            [tokens, 1024],
            "the activation's codes"
        );
        assert_eq!(step.scales.shape.dims(), [tokens, 128], "its block scales");
        assert_eq!(step.token_scale.shape.dims(), [tokens], "its row factor");
        assert_eq!(step.gated.shape.dims(), [routes, 768], "FC1's gated rows");
        assert_eq!(step.gated_values.shape.dims(), [routes, 384], "their codes");
        assert_eq!(
            step.gated_scales.shape.dims(),
            [routes, 48],
            "their block scales"
        );
        assert_eq!(
            step.gated_token_scale.shape.dims(),
            [routes],
            "their row factor"
        );
        assert_eq!(step.products.shape.dims(), [routes, 2048], "FC2's rows");
        assert_eq!(
            step.scratch.shape.dims(),
            [987_156],
            "both products' scratch, each entry's preamble included"
        );
        // Every buffer names a launch, and the table is in the order those launches run: a buffer
        // attributed to a launch that runs before its writer would be read uninitialized.
        let writers: Vec<Launch> = step
            .iter()
            .into_iter()
            .map(|(_, buffer)| buffer.written_by)
            .collect();
        assert!(
            writers.windows(2).all(|pair| pair[0] <= pair[1]),
            "{writers:?} is not the step's launch order"
        );
        // Packed codes are E2M1 (two per byte), the block scales one byte each, the factors one FP32
        // per row, and the rows that are activations carry the model's own type.
        for (name, buffer) in step.iter() {
            let expected = match name {
                "values" | "gated_values" | "scales" | "gated_scales" | "scratch" => DType::U8,
                "token_scale" | "gated_token_scale" => DType::F32,
                _ => DType::BF16,
            };
            assert_eq!(buffer.dtype, expected, "{name}");
        }
        // The plan is the routing prefix plus exactly this table, and nothing else.
        let geometry = MoeGeometry {
            hidden: 2048,
            intermediate: 768,
            top: 8,
            experts: 128,
            tile: 128,
        };
        let planned = encoded_bytes(
            tokens,
            geometry,
            ScaleLayout::Linear,
            ScaleLayout::Swizzled128X4,
            DType::BF16,
        )
        .expect("the encoded term");
        assert_eq!(planned, step.bytes().expect("the step's bytes"));
        assert_eq!(
            moe.step_bytes(
                tokens,
                ScaleLayout::Linear,
                ScaleLayout::Swizzled128X4,
                DType::BF16,
            )
            .expect("the step"),
            routing_bytes(tokens, geometry, DType::BF16).expect("the prefix") + planned
        );
    }

    /// The two products are named by the checkpoint's scale plane and layout, and the encoders' plane
    /// is the plain one: the four keys a store has to contain, as the store's own directory names them.
    ///
    /// The plane fields sat three lines apart in the same two calls and were read from the activation
    /// instead of from the stack; a key that drifts from what a layer launches is the failure a build
    /// reports as `MissingKernel` after any amount of GPU time, and it is caught here on the host.
    #[test]
    fn the_encoded_products_name_the_checkpoint_s_scale_plane_and_the_encoders_are_plain() {
        let moe = quantized_block(2048, 768).expect("the encoded block");
        let gather =
            moe.gather_nvfp4_descriptor(DType::BF16, DType::F8E4M3, ScaleLayout::Swizzled128X4);
        assert_eq!(gather.scale, DType::F8E4M3);
        assert_eq!(gather.layout, ScaleLayout::Swizzled128X4);
        // A descriptor prints the store's leaf; the operator's name is the directory above it.
        assert_eq!(
            ops::descriptor::key(&gather),
            "hidden_2048_intermediate_768_experts_128_top_8_activation_silu_dtype_bf16_scale_f8e4m3_layout_swizzled_128x4_tile_128x128_cluster_1x1"
        );
        let route =
            moe.route_nvfp4_descriptor(DType::BF16, DType::F8E4M3, ScaleLayout::Swizzled128X4);
        assert_eq!(
            ops::descriptor::key(&route),
            "hidden_2048_intermediate_768_experts_128_top_8_dtype_bf16_scale_f8e4m3_layout_swizzled_128x4_tile_128x128_cluster_1x1"
        );
        // The encoder writes plain scales, one per sixteen codes, over the row it encodes.
        let hidden = moe.quantize_descriptor(DType::BF16, 2048, ScaleLayout::Linear);
        assert_eq!(hidden.layout, ScaleLayout::Linear);
        assert_eq!(hidden.columns, 2048);
        let gated = moe.quantize_descriptor(DType::BF16, 768, ScaleLayout::Linear);
        assert_eq!(gated.layout, ScaleLayout::Linear);
        assert_eq!(gated.columns, 768);
    }

    /// The numbers a plan reserves at the slice's own geometry, stated so a change to any extent or to
    /// the arena's alignment shows up as a number here rather than as a write past the region on a
    /// device. 15 tokens over 128 experts at 128-row tiles is the 15 360-row permutation worst case.
    #[test]
    fn the_steps_at_the_slice_s_geometry_are_stated_numbers() {
        let geometry = MoeGeometry {
            hidden: 2048,
            intermediate: 768,
            top: 8,
            experts: 128,
            tile: 128,
        };
        assert_eq!(
            routing_bytes(15, geometry, DType::BF16).expect("the prefix"),
            130_304
        );
        assert_eq!(
            dense_bytes(15, geometry, DType::BF16).expect("the permuted pair"),
            86_507_520
        );
        assert_eq!(
            step_bytes(
                15,
                geometry,
                ScaleLayout::Linear,
                ScaleLayout::Swizzled128X4,
                DType::BF16,
                false,
            )
            .expect("dense step"),
            86_637_824
        );
        assert_eq!(
            encoded_bytes(
                15,
                geometry,
                ScaleLayout::Linear,
                ScaleLayout::Swizzled128X4,
                DType::BF16,
            )
            .expect("the encoded step"),
            94_209_536
        );
        assert_eq!(
            step_bytes(
                15,
                geometry,
                ScaleLayout::Linear,
                ScaleLayout::Swizzled128X4,
                DType::BF16,
                true,
            )
            .expect("encoded step"),
            94_339_840
        );
        assert_eq!(
            encoded_scratch_bytes(15_360, 2048, 768),
            987_156,
            "the products' scratch, both entries' preambles included"
        );
    }

    /// An expert-parallel step is its own reservation, and the difference from the single-rank one is
    /// not a constant: a rank sorts the whole group's rows over the experts it holds, and it keeps one
    /// combined row per received token that no single-rank path has.
    #[test]
    fn the_exchanged_step_reserves_the_received_rows_and_the_combined_ones() {
        let geometry = MoeGeometry {
            hidden: 2048,
            intermediate: 768,
            top: 8,
            experts: 128,
            tile: 128,
        };
        // Decode: the single-rank plan reserves a tile per route, which is far more than a group of
        // 120 received rows over 16 experts each needs.
        let decode = Group {
            ranks: 8,
            capacity: 15,
            attention: 0,
        };
        assert_eq!(decode.received(), 120);
        assert_eq!(
            exchanged_rows(15, geometry, decode).expect("the decode's rows"),
            2_944
        );
        let small = exchanged_bytes(15, geometry, decode, DType::BF16).expect("the decode's step");
        assert!(
            small
                < step_bytes(
                    15,
                    geometry,
                    ScaleLayout::Linear,
                    ScaleLayout::Swizzled128X4,
                    DType::BF16,
                    true,
                )
                .expect("the encoded single-rank step"),
            "{small} is not below the single-rank reservation"
        );
        // Prefill: the same group over a full batch is the case the single-rank plan gets wrong.
        let prefill = Group {
            ranks: 8,
            capacity: 1_024,
            attention: 0,
        };
        let rows = exchanged_rows(1_024, geometry, prefill).expect("the prefill's rows");
        assert_eq!(rows, 67_456);
        let single = step_bytes(
            1_024,
            geometry,
            ScaleLayout::Linear,
            ScaleLayout::Swizzled128X4,
            DType::BF16,
            false,
        )
        .expect("the single-rank step");
        let exchanged =
            exchanged_bytes(1_024, geometry, prefill, DType::BF16).expect("the exchanged step");
        // The combined rows alone are 8 192 tokens of hidden width.
        let combined = 8_192 * 2048 * 2;
        assert!(
            exchanged > single + combined,
            "{exchanged} does not hold the single-rank step {single} plus {combined}"
        );
        // Eight ranks divide the vocabulary; five do not, and a group of zero ranks or zero capacity
        // is a deployment that stated nothing.
        for group in [
            Group {
                ranks: 5,
                capacity: 1_024,
                attention: 0,
            },
            Group {
                ranks: 0,
                capacity: 15,
                attention: 0,
            },
            Group {
                ranks: 8,
                capacity: 0,
                attention: 0,
            },
        ] {
            assert!(
                exchanged_bytes(15, geometry, group, DType::BF16).is_err(),
                "{group:?} was accepted"
            );
        }
        // A batch larger than the group's capacity is not served by a bigger buffer.
        assert!(
            exchanged_bytes(
                16,
                geometry,
                Group {
                    ranks: 8,
                    capacity: 15,
                    attention: 0,
                },
                DType::BF16
            )
            .is_err()
        );
    }

    #[test]
    fn a_quantized_block_builds_from_the_same_geometry_as_a_dense_one() {
        let quantized = quantized_block(2048, 768).unwrap();
        assert_eq!(quantized.experts(), 128);
        assert_eq!(quantized.local_experts(), 128);
        assert_eq!(quantized.top(), 8);
        // The combine bound is BF16's, because that is what a calibrated checkpoint multiplies.
        assert_eq!(
            quantized.tolerance().atol,
            block(128, 8, 2048, 768).unwrap().tolerance().atol,
            "the dense BF16 block and the encoded block share a combine bound"
        );
    }

    /// The packed stacks must describe the geometry the router and the block were built for: a
    /// contraction of another width is refused at the constructor rather than at a launch.
    #[test]
    fn a_quantized_stack_of_another_geometry_is_refused() {
        let experts = 128u32;
        let router = Linear::new(
            weight(DType::BF16, &[u64::from(experts), 2048]),
            None,
            WeightOrder::NxK,
        )
        .unwrap();
        let stack = |rows: u64, columns: u64, projections: u64| QuantizedStack {
            values: weight(DType::U8, &[u64::from(experts), rows, columns / 2]),
            scales: weight(DType::U8, &[u64::from(experts), rows, columns / 16]),
            global_scale: weight(DType::F32, &[projections, u64::from(experts)]),
            layout: ops::fp4::ScaleLayout::Linear,
            encode_scale: weight(DType::F32, &[1]),
            input_scale: None,
            activation_scale: 1.0,
        };
        // Down's contraction is the gated width, so 512 experts' contraction does not fit 768.
        assert!(
            Moe::quantized(
                router,
                stack(1536, 2048, 2),
                stack(2048, 512, 1),
                None,
                experts,
                8,
                Activation::Silu,
                true,
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn the_qwen3_shapes_build() {
        let block = block(128, 8, 2048, 768).unwrap();
        assert_eq!(block.experts(), 128);
        assert_eq!(block.top(), 8);
    }

    #[test]
    fn a_routing_the_weights_do_not_have_is_refused() {
        // The router projects into the expert vocabulary, so one output per expert.
        let router = Linear::new(weight(DType::BF16, &[64, 2048]), None, WeightOrder::NxK).unwrap();
        let up_gate = weight(DType::BF16, &[128, 1536, 2048]);
        let down = weight(DType::BF16, &[128, 2048, 768]);
        assert!(
            Moe::new(
                router,
                up_gate,
                down,
                None,
                128,
                8,
                Activation::Silu,
                true,
                None
            )
            .is_err(),
            "a router of 64 outputs cannot route over 128 experts"
        );
        // An expert stack whose second product does not take the first's width.
        let wrong_down = weight(DType::BF16, &[128, 2048, 512]);
        assert!(block(128, 8, 2048, 768).is_ok());
        let router =
            Linear::new(weight(DType::BF16, &[128, 2048]), None, WeightOrder::NxK).unwrap();
        assert!(
            Moe::new(
                router,
                weight(DType::BF16, &[128, 1536, 2048]),
                wrong_down,
                None,
                128,
                8,
                Activation::Silu,
                true,
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn a_top_outside_the_router_is_refused() {
        assert!(block(8, 0, 64, 32).is_err(), "no route is not a block");
        assert!(block(8, 9, 64, 32).is_err(), "more routes than experts");
        assert!(block(0, 1, 64, 32).is_err(), "no experts is not a block");
        assert!(block(8, 8, 64, 32).is_ok());
    }

    #[test]
    fn a_map_or_scale_of_the_wrong_length_is_refused() {
        let router = Linear::new(weight(DType::BF16, &[8, 64]), None, WeightOrder::NxK).unwrap();
        let up_gate = weight(DType::BF16, &[8, 64, 64]);
        let down = weight(DType::BF16, &[8, 64, 32]);
        let short_map = weight(DType::I32, &[4]);
        assert!(
            Moe::new(
                router,
                up_gate,
                down,
                Some(short_map),
                8,
                2,
                Activation::Silu,
                true,
                None
            )
            .is_err(),
            "a map of four entries cannot route eight experts"
        );
    }

    #[test]
    fn the_combine_bound_is_the_route_count() {
        let block = block(128, 8, 2048, 768).unwrap();
        assert_eq!(
            block.tolerance(),
            MoeUnpermute::tolerance(&combine::Descriptor {
                layout: combine::Layout::Permuted,
                hidden: 2048,
                top: 8,
                experts: 128,
                input: DType::BF16,
                scales: DType::F32,
                output: DType::BF16,
            })
        );
    }
}

#[cfg(test)]
mod parallel_tests {
    use super::tests::expert_parallel_block;

    #[test]
    fn an_expert_parallel_block_holds_a_window_of_the_vocabulary() {
        let block = expert_parallel_block(128, 16, 96).unwrap();
        assert_eq!(block.experts(), 128);
        assert_eq!(block.local_experts(), 16);
        assert_eq!(block.top(), 8);
    }

    #[test]
    fn a_window_outside_the_vocabulary_is_refused() {
        assert!(
            expert_parallel_block(128, 16, 113).is_err(),
            "window past the last expert"
        );
        assert!(
            expert_parallel_block(128, 0, 0).is_err(),
            "no resident expert"
        );
        // A window that ends exactly at the last expert is the last rank of a legal split.
        let last = expert_parallel_block(128, 32, 96).unwrap();
        assert_eq!(last.experts(), 128);
        assert_eq!(last.local_experts(), 32);
    }
}
