//! The mixture-of-experts block: a replicated router, the top-k routes, the sort that groups them
//! by expert, the two grouped products, and the weighted combine.
//!
//! The block is one layer because its parts only make sense together: routing decides which expert
//! rows a token's entries go to, the sort turns that decision into the tile layout the grouped
//! products read, and the combine sums the routes back with the FP32 weights the router produced.
//!
//! ## Buffers
//!
//! The routing, sort and product buffers are sized from the batch's token count and the routing
//! geometry alone, never from one batch's routing distribution: the sort's tile and permutation
//! buffers hold the worst case ([`ops::moe::max_tiles`], [`ops::moe::max_permuted_rows`]), so a
//! captured graph replays over the same addresses however the router routes a later batch. The
//! layer takes them from [`Forward::arena`], in a fixed order, which is what makes a captured
//! bucket's reservations static.
//!
//! ## Expert parallelism
//!
//! One rank holding every expert is the colocated case this layer serves; an expert-parallel group
//! routes over `expert_exchange` and is built with the expert-parallel step, whose
//! `moe_config.exchange` program selects the transport. A `Moe` layer built for a layout that does
//! not hold every expert refuses the forward rather than computing a routing the weights do not
//! have.

mod dispatch;

pub use dispatch::DispatchViews;

use ops::activation::Activation;
use ops::fp4::{Nvfp4Quantize, QuantizeArgs, QuantizeDescriptor, ScaleLayout};
use ops::moe::combine::{self, MoeUnpermute, UnpermuteArgs};
use ops::moe::exchange::{self, ExchangeArgs, ExpertExchange, Mode};
use ops::moe::grouped::{
    GatherGemm, GatherGemmArgs, GatherGemmDescriptor, GatherGemmNvfp4, GatherGemmNvfp4Args,
    GatherGemmNvfp4Descriptor, RouteGemm, RouteGemmArgs, RouteGemmDescriptor, RouteGemmNvfp4,
    RouteGemmNvfp4Args, RouteGemmNvfp4Descriptor,
};
use ops::moe::routing::{TopkArgs, TopkDescriptor, TopkSoftmax};
use ops::moe::sort::{self, MoeSort, SortArgs};
use ops::{DType, DevicePtr, Shape, TensorView};

use crate::error::{Error, Result};
use crate::family::Split;
use crate::forward::Forward;
use crate::layers::linear::{ENCODE_CHUNK, Linear, encode_block};
use crate::quantization::Scheme;
use crate::weights::{
    ExpertStack, Nvfp4Names, PartBlock, QuantizedStack, Scope, StackLayout, ValueLayout, Weight,
    WeightLayout, WeightLoader, WeightRequest,
};
use ops::gemm::WeightOrder;

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
/// program it cannot see. The formula is therefore the artifact's, over the padded worst-case route
/// count; `routes` is padded to whole MMA tiles because the program walks whole tiles.
///
/// The size is a property of the bucket, not of the step: it is computed from the worst-case route
/// count of the bucket's tokens, so every replay of a captured bucket reserves the same number of
/// bytes and nothing in a step grows a buffer.
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
/// Like the row tile of a normalization this is program content; the
/// values here are the packages' default tactic for the shapes a checkpoint states (e.g. 128
/// experts, top-k 8, hidden 2048, expert width 768), declared once in
/// [`crate::layers::launch`] rather than at every call.
const MMA_TILE: (u32, u32) = (128, 128);
/// The cluster shape the grouped products' packages are compiled for.
const MMA_CLUSTER: (u32, u32) = (1, 1);

/// Rows per permutation tile: the row extent of `MMA_TILE`, which is what a sort's permutation
/// buffer is padded to.
const TILE_ROWS: u32 = MMA_TILE.0;

/// The block-scale layout the expert products read the **weights** in: the 128x4 swizzled form.
///
/// Both families carry their weight operand this way — `mSfLayoutA` is `SfLayout(3)` in every one of
/// the artifact's E2M1 kernels, and `BatchedGemmOptions.h` states the rule it follows: "mSfLayoutA
/// has to be R128cX when not batch M or not routed". The plane the two families disagree about is
/// the **tokens** operand ([`Moe::GATED_TOKEN_SCALE_LAYOUT`], [`Moe::ACTIVATION_SCALE_LAYOUT`]).
///
/// The shard stores the plain form, so the loader builds this table once at load.
pub const WEIGHT_SCALE_LAYOUT: ScaleLayout = ScaleLayout::Swizzled128X4;

/// The mathematics of a routed block, as its model's configuration states it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MoeSpec {
    /// Routed experts of the block: the router's vocabulary.
    pub experts: u32,
    /// Experts each token is routed to.
    pub top: u32,
    /// The hidden width the experts read and write.
    pub hidden: u32,
    /// One expert's intermediate width, before any tensor-parallel split.
    pub intermediate: u32,
    /// The gate's activation.
    pub activation: Activation,
    /// Whether the selected experts' weights renormalize to one.
    pub renormalize: bool,
}

/// The module names a checkpoint states a routed block's parameters under.
///
/// A model's checkpoint names its own modules; the block composes them as the checkpoint does:
/// the router is `<block>.<router>` and expert `e`'s projections `<block>.<experts>.<e>.<up>`,
/// `.<gate>` and `.<down>`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ExpertNames<'n> {
    /// The router's module.
    pub router: &'n str,
    /// The module list holding the experts.
    pub experts: &'n str,
    /// An expert's up projection.
    pub up: &'n str,
    /// An expert's gate projection.
    pub gate: &'n str,
    /// An expert's down projection.
    pub down: &'n str,
    /// The stacked tensors a checkpoint may hold its experts in instead of one module per expert,
    /// when its family writes them so.
    pub stacked: Option<StackedExperts<'n>>,
}

/// A routed block's experts as stacked tensors (Hugging Face transformers' fused experts, as
/// Qwen3.5 checkpoints write them): `<experts>.<gate_up>` is `[experts, 2·intermediate, hidden]`
/// with each expert's gate rows before its up rows, and `<experts>.<down>` is
/// `[experts, hidden, intermediate]`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct StackedExperts<'n> {
    /// The stacked gate and up projections.
    pub gate_up: &'n str,
    /// The stacked down projections.
    pub down: &'n str,
}

/// Where a block's experts sit: the expert each slot holds, and the table a route reads its slot
/// from.
#[derive(Clone, Copy, Debug)]
pub struct ExpertSlots<'s> {
    /// The expert slot `s` holds, for every slot of the vocabulary: a permutation of the experts.
    pub order: &'s [u32],
    /// `[experts]` i32: the slot of each expert ([`Moe::slot_map`] of `order`), as a published
    /// table's row. The model publishes every routed layer's rows as one table, so the rows share
    /// one allocation.
    pub map: Weight,
}
/// The experts of a router's vocabulary a block holds: `local` of the `experts`, from `offset`.
#[derive(Clone, Copy, Debug)]
struct Window {
    experts: u32,
    local: u32,
    offset: u32,
}

/// The experts one rank of a block holds: their slots' experts in slot order, the slot table a
/// route reads, and the expert group the window is this rank's share of.
#[derive(Clone, Copy)]
struct Resident<'s> {
    slots: &'s [u32],
    map: Weight,
    exchange: Split,
}

impl Resident<'_> {
    /// The window of `experts` these slots are.
    fn window(&self, experts: u32) -> Window {
        let local = u32::try_from(self.slots.len()).unwrap_or(u32::MAX);
        Window {
            experts,
            local,
            offset: self.exchange.rank * local,
        }
    }
}

/// The per-expert scalars a fused exchange's kernel multiplies in, one table per use.
#[derive(Clone)]
pub(crate) struct FusedTables {
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
    /// Ranks of the tensor group this block's experts are split over by their intermediate width:
    /// above one, each rank's output is a partial sum its forward reduces over the group.
    reduce: u32,
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

    /// The block over the resident `window` of the router's vocabulary.
    ///
    /// `stacks` hold the window's experts: dense, up `[local, 2 · intermediate, hidden]` (up rows
    /// first) and down `[local, hidden, intermediate]`, or their encoded fields. `router` has one
    /// output per expert of the vocabulary. `map` is the slot table when the layout reorders
    /// experts; absent, a route's logical id is its weight row. `routing` states the top, the
    /// gate's activation and the renormalization; `scale` is a router's per-expert scale.
    fn build(
        router: Linear,
        stacks: ExpertStacks,
        map: Option<Weight>,
        window: Window,
        routing: &MoeSpec,
        scale: Option<Weight>,
    ) -> Result<Self> {
        let Window {
            experts,
            local,
            offset: local_offset,
        } = window;
        let (top, activation, renormalize) = (routing.top, routing.activation, routing.renormalize);
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
                    "the router reduces over {} elements and the experts' rows are {hidden} wide",
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
            reduce: 1,
            tile: TILE_ROWS,
            local,
            local_offset,
        })
    }

    /// The routed block the module at `scope` states.
    ///
    /// The model supplies the block's mathematics (`spec`) and its checkpoint's module names
    /// (`names`): the router is `<scope>.<router>`, and expert `e`'s projections are
    /// `<scope>.<experts>.<e>.<up|gate|down>`. The block reads the router replicated and the experts
    /// `slots` places, in slot order, and holds:
    ///
    /// - over an expert-parallel group (`exchange`), this rank's contiguous window of the slots,
    ///   with the router over the whole vocabulary so every rank routes a token the same way;
    /// - over a tensor group (`tensor`), this rank's block of every expert's intermediate width
    ///   (the gate and up projections' rows, the down projection's columns), whose partial sums
    ///   [`Self::forward`] reduces over the group.
    ///
    /// A dense checkpoint's experts are stacked expert-major, up before gate. A calibrated NVFP4
    /// checkpoint's are read as packed stacks in the arrangement their products read: over a group,
    /// the fused exchange's (each gate block before its up block) with the per-expert scalars its
    /// kernel multiplies in; on one rank, the shuffled rows the trtllm-gen products read. The
    /// declaration the loader was given decides which, from the first held expert's weight.
    ///
    /// # Errors
    ///
    /// When the slots are no permutation of the vocabulary, a split does not divide the experts
    /// or their width, or a tensor is missing or of another extent.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        spec: MoeSpec,
        names: ExpertNames<'_>,
        slots: ExpertSlots<'_>,
        tensor: Split,
        exchange: Split,
    ) -> Result<Self> {
        let path = scope.path().to_owned();
        let refuse = |problem: String| Error::weight(&path, problem);
        let experts = spec.experts;
        if slots.order.len() != experts as usize {
            return Err(refuse(format!(
                "{} slots for {experts} experts",
                slots.order.len()
            )));
        }
        if exchange.size == 0 || !experts.is_multiple_of(exchange.size) {
            return Err(refuse(format!(
                "an expert group of {} does not divide {experts} experts",
                exchange.size
            )));
        }
        if tensor.size == 0 || !spec.intermediate.is_multiple_of(tensor.size) {
            return Err(refuse(format!(
                "a tensor group of {} does not divide an expert's {} intermediate columns",
                tensor.size, spec.intermediate
            )));
        }
        let hidden = u64::from(spec.hidden);
        let intermediate = u64::from(spec.intermediate);
        // The rank's block of every expert: the tensor group splits the intermediate width.
        let local_intermediate = intermediate / u64::from(tensor.size);
        // The rank's window of the slots.
        let local = experts / exchange.size;
        let offset = exchange.rank * local;
        let window = &slots.order[offset as usize..(offset + local) as usize];

        let router = Linear::new(
            scope
                .scope(names.router)
                .tensor("weight", &[u64::from(experts), hidden])?,
            None,
            WeightOrder::NxK,
        )?;
        let mut experts_scope = scope.scope(names.experts);
        let stem = experts_scope.path().to_owned();
        let stem = stem.as_str();
        let encoded = experts_scope
            .loader()
            .scheme(&format!("{stem}.{}.{}.weight", window[0], names.up))
            == Scheme::Nvfp4;

        let resident = Resident {
            slots: window,
            map: slots.map,
            exchange,
        };
        let mut block = if encoded {
            Self::load_encoded(
                experts_scope.loader(),
                (&path, stem),
                &spec,
                names,
                resident,
                router,
            )?
        } else if let Some(stacked) = names
            .stacked
            .filter(|stacked| experts_scope.contains(stacked.gate_up))
        {
            let (up_gate, down) = Self::load_stacked(
                &mut experts_scope,
                stacked,
                window,
                hidden,
                intermediate,
                tensor,
            )?;
            Self::build(
                router,
                ExpertStacks::Dense { up_gate, down },
                Some(resident.map),
                resident.window(experts),
                &spec,
                None,
            )?
        } else {
            let mut up_gate_parts = Vec::with_capacity(window.len() * 2);
            let mut down_parts = Vec::with_capacity(window.len());
            for expert in window {
                let projection = |module: &str| format!("{stem}.{expert}.{module}.weight");
                up_gate_parts.push(projection(names.up));
                up_gate_parts.push(projection(names.gate));
                down_parts.push(projection(names.down));
            }
            let rank = u64::from(tensor.rank);
            let count = u64::from(local);
            // A gate or up projection is `[intermediate, hidden]`, so a rank's block is its rows; a
            // down projection is `[hidden, intermediate]`, so a rank's block is its columns of every
            // row.
            let request = |name: String, parts: Vec<String>, shape: Shape, layout, block| {
                if tensor.size == 1 {
                    WeightRequest::parts(name, parts, shape, layout)
                } else {
                    WeightRequest::part_blocks(name, parts, shape, layout, block)
                }
            };
            let up_gate = experts_scope.load(request(
                format!("{path}.experts.up_gate"),
                up_gate_parts,
                Shape::new(&[count, 2 * local_intermediate, hidden]),
                WeightLayout::UP_GATE,
                PartBlock {
                    start: rank * local_intermediate * hidden,
                    rows: local_intermediate,
                    columns: hidden,
                    row_stride: hidden,
                },
            ))?;
            let down = experts_scope.load(request(
                format!("{path}.experts.down"),
                down_parts,
                Shape::new(&[count, hidden, local_intermediate]),
                WeightLayout::DOWN,
                PartBlock {
                    start: rank * local_intermediate,
                    rows: hidden,
                    columns: local_intermediate,
                    row_stride: intermediate,
                },
            ))?;
            Self::build(
                router,
                ExpertStacks::Dense { up_gate, down },
                Some(resident.map),
                resident.window(experts),
                &spec,
                None,
            )?
        };
        block.reduce = tensor.size;
        Ok(block)
    }

    /// The dense stacks of `window` read from a checkpoint's stacked expert tensors, in the
    /// arrangement the products read: each expert's up rows before its gate rows, and the down
    /// projection's columns of this tensor rank.
    fn load_stacked(
        scope: &mut Scope<'_, '_>,
        stacked: StackedExperts<'_>,
        window: &[u32],
        hidden: u64,
        intermediate: u64,
        tensor: Split,
    ) -> Result<(Weight, Weight)> {
        let path = scope.path().to_owned();
        let local_intermediate = intermediate / u64::from(tensor.size);
        let rank = u64::from(tensor.rank);
        let count = window.len() as u64;
        let gate_up = scope.name(stacked.gate_up);
        let down = scope.name(stacked.down);
        // One expert's slab of the stacked first projection is its gate rows, then its up rows; the
        // product reads up before gate, each rank its block of both halves.
        let mut up_gate_parts = Vec::with_capacity(window.len() * 2);
        let mut down_parts = Vec::with_capacity(window.len());
        for &expert in window {
            let slab = u64::from(expert) * 2 * intermediate * hidden;
            let half = |first_row: u64| PartBlock {
                start: slab + (first_row + rank * local_intermediate) * hidden,
                rows: local_intermediate,
                columns: hidden,
                row_stride: hidden,
            };
            up_gate_parts.push((gate_up.clone(), half(intermediate)));
            up_gate_parts.push((gate_up.clone(), half(0)));
            down_parts.push((
                down.clone(),
                PartBlock {
                    start: u64::from(expert) * hidden * intermediate + rank * local_intermediate,
                    rows: hidden,
                    columns: local_intermediate,
                    row_stride: intermediate,
                },
            ));
        }
        let up_gate = scope.load(WeightRequest::blocks(
            format!("{path}.up_gate"),
            up_gate_parts,
            Shape::new(&[count, 2 * local_intermediate, hidden]),
            WeightLayout::UP_GATE,
        ))?;
        let down = scope.load(WeightRequest::blocks(
            format!("{path}.down"),
            down_parts,
            Shape::new(&[count, hidden, local_intermediate]),
            WeightLayout::DOWN,
        ))?;
        Ok((up_gate, down))
    }

    /// The packed experts `resident` holds, as [`Self::load`] reads a calibrated checkpoint's:
    /// `path` is the block's module path and `stem` the path of its experts' module list.
    fn load_encoded(
        loader: &mut WeightLoader<'_>,
        (path, stem): (&str, &str),
        spec: &MoeSpec,
        names: ExpertNames<'_>,
        resident: Resident<'_>,
        router: Linear,
    ) -> Result<Self> {
        let experts = spec.experts;
        let (window, exchange) = (resident.slots, resident.exchange);
        let local = u32::try_from(window.len()).unwrap_or(u32::MAX);
        let spelling = Nvfp4Names::vllm();
        if exchange.size > 1 {
            // An encoded block over a group runs in the exchange's one fused launch, which reads
            // its own arrangement of the same bytes: each gate block before its up block, and the
            // block scales in the swizzled table.
            let (up_gate, up_scales) = loader.nvfp4_stack(
                &format!("{path}.experts.up_gate"),
                ExpertStack {
                    prefix: stem,
                    projections: &[names.up, names.gate],
                    slots: window,
                    vocabulary: experts,
                },
                spelling,
                StackLayout {
                    values: ValueLayout::GateFirst,
                    scales: WEIGHT_SCALE_LAYOUT,
                },
                true,
            )?;
            let (down, down_scales) = loader.nvfp4_stack(
                &format!("{path}.experts.down"),
                ExpertStack {
                    prefix: stem,
                    projections: &[names.down],
                    slots: window,
                    vocabulary: experts,
                },
                spelling,
                StackLayout {
                    values: ValueLayout::Plain,
                    scales: WEIGHT_SCALE_LAYOUT,
                },
                true,
            )?;
            // The kernel's three scalars per expert: each product dequantizes with the weight's
            // tensor scale times its rows' activation scale, and the gated rows encode with the
            // second activation scale's reciprocal. A fused first projection carries one tensor
            // scale per expert, which up and gate share.
            let up_alpha: Vec<f32> = up_scales
                .chunks(2)
                .map(|scales| scales[0] * up_gate.activation_scale)
                .collect();
            let down_alpha: Vec<f32> = down_scales
                .iter()
                .map(|scale| scale * down.activation_scale)
                .collect();
            let gated = vec![down.activation_scale.recip(); window.len()];
            let mut table = |what: &str, values: Vec<f32>| {
                let bytes: Vec<u8> = values
                    .iter()
                    .flat_map(|value| value.to_ne_bytes())
                    .collect();
                loader.constant(
                    &format!("{path}.experts.{what}"),
                    DType::F32,
                    Shape::new(&[u64::from(local)]),
                    &bytes,
                    "nn::Moe",
                )
            };
            let tables = FusedTables {
                up_gate_alpha: table("up_gate_alpha", up_alpha)?,
                down_alpha: table("down_alpha", down_alpha)?,
                gated_scale: table("gated_scale", gated)?,
            };
            let mut block = Self::build(
                router,
                ExpertStacks::Encoded {
                    up_gate: Box::new(up_gate),
                    down: Box::new(down),
                },
                Some(resident.map),
                resident.window(experts),
                spec,
                None,
            )?;
            block.fused = Some(tables);
            return Ok(block);
        }
        // On one rank the products read their A operand shuffled: the gated stack interleaves its
        // halves first. The shard's block scales are linear, and the published table is built from
        // them once here.
        let (up_gate, _) = loader.nvfp4_stack(
            &format!("{path}.experts.up_gate"),
            ExpertStack {
                prefix: stem,
                projections: &[names.up, names.gate],
                slots: window,
                vocabulary: experts,
            },
            spelling,
            StackLayout {
                values: ValueLayout::InterleavedShuffled {
                    epilogue_tile_m: 128,
                },
                scales: WEIGHT_SCALE_LAYOUT,
            },
            true,
        )?;
        let (down, _) = loader.nvfp4_stack(
            &format!("{path}.experts.down"),
            ExpertStack {
                prefix: stem,
                projections: &[names.down],
                slots: window,
                vocabulary: experts,
            },
            spelling,
            StackLayout {
                values: ValueLayout::Shuffled {
                    epilogue_tile_m: 128,
                },
                scales: WEIGHT_SCALE_LAYOUT,
            },
            true,
        )?;
        // A block on one rank holds every expert the router names.
        Self::build(
            router,
            ExpertStacks::Encoded {
                up_gate: Box::new(up_gate),
                down: Box::new(down),
            },
            Some(resident.map),
            Window {
                experts,
                local: experts,
                offset: 0,
            },
            spec,
            None,
        )
    }

    /// The slot of each expert, inverting `order`, the expert each slot holds: the row of the table
    /// a route reads its slot from.
    ///
    /// # Errors
    ///
    /// When `order` is no permutation of its experts.
    pub fn slot_map(order: &[u32]) -> Result<Vec<u32>> {
        let mut row = vec![u32::MAX; order.len()];
        for (slot, expert) in order.iter().enumerate() {
            let entry = row.get_mut(*expert as usize).ok_or_else(|| {
                Error::shape(
                    "Moe",
                    format!("a slot order names expert {expert} of {}", order.len()),
                )
            })?;
            *entry = u32::try_from(slot).unwrap_or(u32::MAX);
        }
        if row.contains(&u32::MAX) {
            return Err(Error::shape(
                "Moe",
                "a slot order is not a permutation of its experts".to_owned(),
            ));
        }
        Ok(row)
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

    /// The router's choice for `hidden`, and nothing else: the top-k expert ids and their
    /// weights, fresh `[tokens, top]` i32 and F32 tensors.
    ///
    /// The ids are the checkpoint's expert indices, *before* the expert map, so they are exactly
    /// what a reference's router names and a comparison of routing is an integer comparison. That
    /// keeps three findings apart that a whole MoE's output would confound: a wrong id (routing),
    /// a right id with a wrong product (the experts), and a right id with a wrong weight (the
    /// top-k's renormalization).
    ///
    /// No expert runs, so this costs one router row per token and the scores it needs, reserved
    /// from the forward's arena before the ids and weights.
    ///
    /// # Errors
    ///
    /// Rows of another width, an arena that cannot hold the tensors, or an operator's refusal.
    pub fn route(&self, f: &Forward, hidden: TensorView) -> Result<(TensorView, TensorView)> {
        hidden
            .require_row("Moe::route", hidden.dtype, u64::from(self.hidden))
            .map_err(Error::from)?;

        let tokens = hidden.rows();
        let top = u64::from(self.top);
        let scores = self.router.forward(f, hidden)?;
        let ids = f.empty(DType::I32, Shape::new(&[tokens, top]))?;
        let weights = f.empty(DType::F32, Shape::new(&[tokens, top]))?;

        let routing = self.topk_descriptor(scores.dtype);
        let args = TopkArgs {
            scores,
            scale: self.scale.as_ref().map(Weight::view),
            ids,
            weights,
            map: self.map.as_ref().map(Weight::view),
        };

        f.operate::<TopkSoftmax>(&routing, args)?;

        Ok((ids, weights))
    }

    /// The routing prefix: the router, its top-k, the expert map, and the sort that groups the
    /// routes into the tiles a grouped product reads.
    ///
    /// Every expert product a block can run starts here and differs only in the two multiplies, so
    /// the prefix is one path: a variant that multiplies something else still routes, sorts and
    /// combines identically, and a heavy quantized product cannot drift from the dense one.
    fn routed(&self, f: &Forward, hidden: TensorView) -> Result<Routes> {
        let tokens = hidden.rows();
        let top = u64::from(self.top);
        let experts = u64::from(self.experts);

        // The top-k writes each route's weight row: logical ids stop there.
        let (local_ids, weights) = self.route(f, hidden)?;

        // Group the routes by expert into the tiles the products read.
        let tiles = ops::moe::max_tiles(tokens, top, experts, u64::from(self.tile));
        let rows = ops::moe::max_permuted_rows(tokens, top, experts, u64::from(self.tile));
        let tile_to_expert = f.empty(DType::I32, Shape::new(&[tiles]))?;
        let tile_to_limit = f.empty(DType::I32, Shape::new(&[tiles]))?;
        let expanded_to_permuted = f.empty(DType::I32, Shape::new(&[tokens, top]))?;
        let permuted_to_expanded = f.empty(DType::I32, Shape::new(&[rows]))?;
        let permuted_to_token = f.empty(DType::I32, Shape::new(&[rows]))?;
        let total_padded_tokens = f.empty(DType::I32, Shape::new(&[1]))?;
        let num_non_exiting_tiles = f.empty(DType::I32, Shape::new(&[1]))?;
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
            live_tokens: f.live_tokens(),
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

    /// `Σ_routes weight · expert(hidden)`, summed over the tensor group: a fresh `[tokens, hidden]`
    /// tensor of `hidden`'s element type, reserved first, then written by [`Self::forward_into`].
    ///
    /// # Errors
    ///
    /// As [`Self::forward_into`], or an arena that cannot hold the output.
    pub fn forward(&self, f: &Forward, hidden: TensorView) -> Result<TensorView> {
        let out = f.empty(
            hidden.dtype,
            Shape::new(&[hidden.rows(), u64::from(self.hidden)]),
        )?;
        self.forward_into(f, hidden, out)?;
        Ok(out)
    }

    /// `Σ_routes weight · expert(hidden)`, summed over the tensor group, written into `update`,
    /// `[tokens, hidden]` of `hidden`'s element type: the caller's buffer the output must land in,
    /// such as a stack's update carried across its layers' scopes.
    ///
    /// A block whose experts the tensor group splits by their intermediate width holds one rank's
    /// block of every expert, so its output is a partial sum: it is reduced over the group here,
    /// once after the experts, and a forward that carries no group is refused. A block whose experts
    /// are split over an expert group, or whose forward borrows the group's exchange views, routes
    /// through the exchange, which returns the whole sum.
    ///
    /// # Errors
    ///
    /// When a view does not fit the block, an operator refuses, or a split block runs on a forward
    /// without a group.
    pub fn forward_into(&self, f: &Forward, hidden: TensorView, update: TensorView) -> Result<()> {
        self.combine(f, hidden, update)?;
        if self.reduce > 1 {
            if f.split().is_none() {
                return Err(Error::shape(
                    "Moe",
                    format!(
                        "experts split over {} tensor ranks run on a forward that carries no group \
                         to reduce over",
                        self.reduce
                    ),
                ));
            }
            f.all_reduce(update)?;
        }
        Ok(())
    }

    /// `out = Σ_routes weight · expert(hidden)` over this rank's experts and block.
    fn combine(&self, f: &Forward, hidden: TensorView, out: TensorView) -> Result<()> {
        if self.local != self.experts || f.dispatch().is_some() {
            return self.exchanged(f, hidden, out);
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

        // 1-3. The router, its top-k, the map and the sort: the prefix every expert product shares.
        let routes = self.routed(f, hidden)?;

        // 4. FC1: the gated product of each permuted row, then FC2.
        let gated = f.empty(hidden.dtype, Shape::new(&[routes.rows, intermediate]))?;
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
        let products = f.empty(hidden.dtype, Shape::new(&[routes.rows, hidden_width]))?;
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
    fn topk_descriptor(&self, scores: DType) -> TopkDescriptor {
        TopkDescriptor {
            scores,
            experts: self.experts,
            top: self.top,
            renormalize: self.renormalize,
            scale: self.scale.as_ref().map(Weight::dtype),
        }
    }

    /// The sort's descriptor, for a block whose rank holds `local_experts` experts.
    ///
    /// `local_experts` is an argument and not the layer's field because the same layer sorts for
    /// the dense block (every expert) and for an expert-parallel rank (its own), and the two are
    /// different programs.
    fn sort_descriptor(&self, local_experts: u32) -> sort::Descriptor {
        sort::Descriptor {
            experts: self.experts,
            top: self.top,
            local_experts,
            tile_tokens: self.tile,
            scales: DType::F32,
            counted: true,
        }
    }

    /// The first product's descriptor: the gated FC1 over the dense packed stack.
    fn gather_descriptor(&self, hidden: DType) -> GatherGemmDescriptor {
        GatherGemmDescriptor {
            hidden: self.hidden,
            intermediate: self.intermediate,
            experts: self.experts,
            top: self.top,
            activation: self.activation,
            dtype: hidden,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        }
    }

    /// The second product's descriptor: FC2 over the dense packed stack.
    fn route_descriptor(&self, hidden: DType) -> RouteGemmDescriptor {
        RouteGemmDescriptor {
            hidden: self.hidden,
            intermediate: self.intermediate,
            experts: self.experts,
            top: self.top,
            dtype: hidden,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        }
    }

    /// The descriptor of one activation encoding this block runs.
    ///
    /// The chunk a program walks a row in is the block: the encoder reduces over the whole row and
    /// scales every block of it against that row's factor, so the chunk is how much of the row one
    /// pass holds, not how many values a key spans. The grid covers the rows in one launch — it is
    /// keyed by `columns`, not by a row count — so every activation of one width shares one program.
    fn quantize_descriptor(
        &self,
        input: DType,
        columns: u32,
        layout: ScaleLayout,
    ) -> QuantizeDescriptor {
        QuantizeDescriptor {
            input,
            columns,
            layout,
            block: encode_block(columns, ENCODE_CHUNK),
            pdl: crate::layers::launch::PDL,
        }
    }

    /// The first product's descriptor over an encoded stack: the block scale's element type and the
    /// scale layout are the weights', so they are arguments.
    ///
    /// The grid is over *routes* — tokens × `top`, computed at launch — so this key covers every
    /// route count and is not compiled per bucket.
    fn gather_nvfp4_descriptor(
        &self,
        hidden: DType,
        scale: DType,
        layout: ScaleLayout,
    ) -> GatherGemmNvfp4Descriptor {
        GatherGemmNvfp4Descriptor {
            hidden: self.hidden,
            intermediate: self.intermediate,
            experts: self.experts,
            top: self.top,
            activation: self.activation,
            dtype: hidden,
            scale,
            layout,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        }
    }

    /// The second product's descriptor over an encoded stack, for the same reason and with the same
    /// grid: routes, not rows.
    fn route_nvfp4_descriptor(
        &self,
        hidden: DType,
        scale: DType,
        layout: ScaleLayout,
    ) -> RouteGemmNvfp4Descriptor {
        RouteGemmNvfp4Descriptor {
            hidden: self.hidden,
            intermediate: self.intermediate,
            experts: self.experts,
            top: self.top,
            dtype: hidden,
            scale,
            layout,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        }
    }

    /// The combine's descriptor: the routes summed in FP32 and rounded once to `output`.
    ///
    /// The routes arrive in **permuted** rows, which is the convention `route_gemm` writes and the
    /// one the combine's module is compiled for; it is a descriptor field so a chain that ever
    /// writes expanded rows names its own module instead of reading this one's buffer wrongly.
    fn combine_descriptor(&self, product: DType, output: DType) -> combine::Descriptor {
        combine::Descriptor {
            layout: combine::Layout::Permuted,
            hidden: self.hidden,
            top: self.top,
            experts: self.experts,
            input: product,
            scales: DType::F32,
            output,
        }
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
        // The activation's packed codes, a byte per two codes, one block scale per sixteen and one
        // FP32 factor per token.
        let values = f.empty(DType::U8, Shape::new(&[tokens, hidden_width / 2]))?;
        let scales = f.empty(DType::U8, input_layout.shape(tokens, hidden_width))?;
        let token_scale = f.empty(DType::F32, Shape::new(&[tokens]))?;
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
        // The gated rows are permuted-height: the combine indexes the products in the permuted row
        // space. One scratch serves both products in turn, each entry's preamble included.
        let gated = f.empty(hidden.dtype, Shape::new(&[routes.rows, intermediate]))?;
        let scratch = encoded_scratch_bytes(routes.rows, hidden_width, intermediate);
        let workspace = f.empty(DType::U8, Shape::new(&[scratch.max(1)]))?;
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
        let gated_values = f.empty(DType::U8, Shape::new(&[routes.rows, intermediate / 2]))?;
        let gated_scales = f.empty(DType::U8, gated_layout.shape(routes.rows, intermediate))?;
        let gated_token_scale = f.empty(DType::F32, Shape::new(&[routes.rows]))?;
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
        let products = f.empty(hidden.dtype, Shape::new(&[routes.rows, hidden_width]))?;
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
    /// experts, and sums the ranks' partial results back into its own rows (`combine`).
    ///
    /// `exchange` is the worker's bundle for this group: the package's transport and the views of
    /// its workspace — the rows this rank received at dispatch, with their ids and weights, in the
    /// package's layout, sized by the worker. The layer never allocates or owns any of it.
    ///
    /// Every buffer holds the most rows a rank may exchange, so a captured graph replays over
    /// the same addresses however the router routes a later batch.
    fn exchanged(&self, f: &Forward, hidden: TensorView, out: TensorView) -> Result<()> {
        let exchange = f.dispatch().ok_or_else(|| {
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
        let rank_rows = u64::from(exchange.rank_rows);
        let received = u64::from(exchange.ranks) * rank_rows;
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
        if rank_rows == 0 || received == 0 {
            return Err(Error::shape(
                "Moe",
                "the exchange holds no row: a rank sends at least one".to_owned(),
            ));
        }
        if tokens > rank_rows {
            return Err(Error::shape(
                "Moe",
                format!("the batch has {tokens} tokens, the exchange carries {rank_rows} per rank"),
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

        // 1. The replicated router and its top-k: every rank routes whole tokens the same way.
        let scores = self.router.forward(f, hidden)?;
        let ids = f.empty(DType::I32, Shape::new(&[tokens, top]))?;
        let weights = f.empty(DType::F32, Shape::new(&[tokens, top]))?;
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
        let exchange = f.dispatch().ok_or_else(|| {
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
        exchange: &DispatchViews,
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
                "Moe: an encoded stack has no exchange path yet; the \
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
            max_tokens: exchange.rank_rows,
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
            live: f.live_tokens(),
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
        let tile_to_expert = f.empty(DType::I32, Shape::new(&[tiles]))?;
        let tile_to_limit = f.empty(DType::I32, Shape::new(&[tiles]))?;
        let expanded_to_permuted = f.empty(DType::I32, Shape::new(&[received, top]))?;
        let permuted_to_expanded = f.empty(DType::I32, Shape::new(&[rows]))?;
        let permuted_to_token = f.empty(DType::I32, Shape::new(&[rows]))?;
        let total_padded_tokens = f.empty(DType::I32, Shape::new(&[1]))?;
        let num_non_exiting_tiles = f.empty(DType::I32, Shape::new(&[1]))?;
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
            // The rows the step holds, which the ranks agree on the device: the views'
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
        let gated = f.empty(exchange.states.dtype, Shape::new(&[rows, intermediate]))?;
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
        let products = f.empty(exchange.states.dtype, Shape::new(&[rows, hidden_width]))?;
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
        exchange: &DispatchViews,
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
            let token_scale = f.empty(DType::F32, Shape::new(&[tokens]))?;
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
            max_tokens: exchange.rank_rows,
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
            live: f.live_tokens(),
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
        exchange: &DispatchViews,
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
            max_tokens: exchange.rank_rows,
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
            live: f.live_tokens(),
            out,
            fused: None,
            ranks: exchange.ranks,
        };

        (exchange.meet)()?;
        f.operate::<ExpertExchange>(&descriptor, args)?;
        f.settled()
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::tensor::Backing;
    use ops::gemm::WeightOrder;
    use ops::{DevicePtr, Shape};

    /// The routing every test block uses: `top` of `experts` under SiLU, renormalized.
    fn routing(experts: u32, top: u32) -> MoeSpec {
        MoeSpec {
            experts,
            top,
            hidden: 0,
            intermediate: 0,
            activation: Activation::Silu,
            renormalize: true,
        }
    }

    /// Every expert of a vocabulary of `experts` resident.
    fn colocated(experts: u32) -> Window {
        Window {
            experts,
            local: experts,
            offset: 0,
        }
    }

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
        Moe::build(
            router,
            ExpertStacks::Dense {
                up_gate: weight(DType::BF16, &[u64::from(experts), 2 * intermediate, hidden]),
                down: weight(DType::BF16, &[u64::from(experts), hidden, intermediate]),
            },
            None,
            colocated(experts),
            &routing(experts, top),
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
        Moe::build(
            router,
            ExpertStacks::Dense {
                up_gate: weight(DType::BF16, &[u64::from(local), 1536, 2048]),
                down: weight(DType::BF16, &[u64::from(local), 2048, 768]),
            },
            None,
            Window {
                experts,
                local,
                offset,
            },
            &routing(experts, 8),
            None,
        )
    }

    /// A quantized block's stacks are packed: the same 128-expert geometry, two codes a byte along
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
            scales: weight(DType::F8E4M3, &[u64::from(experts), rows, columns / 16]),
            global_scale: weight(DType::F32, &[projections, u64::from(experts)]),
            layout: ops::fp4::ScaleLayout::Linear,
            encode_scale: weight(DType::F32, &[1]),
            input_scale: None,
            activation_scale: 1.0,
        };
        Moe::build(
            router,
            ExpertStacks::Encoded {
                up_gate: Box::new(stack(2 * intermediate, hidden, 2)),
                down: Box::new(stack(hidden, intermediate, 1)),
            },
            None,
            colocated(experts),
            &routing(experts, 8),
            None,
        )
    }

    /// The encoded products' scratch is computed from the artifact's arithmetic and each entry's
    /// preamble: a bucket of 8 routes costs the row-max planes and tile bars of the wider product,
    /// and a bucket twice as wide costs more. What it must not be is a floor: the size the largest
    /// 128-expert case needs is far below sixteen mebibytes.
    ///
    /// The number is the one an entry compares its own request against, so it is stated for the
    /// geometry a deployment runs (2048 hidden, 768 intermediate): the second product's entry asks
    /// 16 + 987 140.
    #[test]
    fn the_scratch_is_sized_from_the_route_count_and_not_from_a_floor() {
        let scratch = |routes: u64| encoded_scratch_bytes(routes, 2048, 768);
        let small = scratch(8);
        let large = scratch(4096);
        assert!(small > 0);
        assert!(large > small, "{large} is not more than {small}");
        assert!(large < 16 << 20, "{large} exceeds the floor it replaced");
        // 15 tokens over 128 experts is 120 permuted tiles of 128 rows: the routed rows the caller
        // launches. The gated product needs 16 + 740 356 and the contraction 16 + 987 140, so the
        // shared buffer is the contraction's.
        assert_eq!(scratch(15_360), 987_156);
        // A bucket small enough that the artifact's planes are a single tile still carries the
        // entry's preamble.
        assert_eq!(scratch(8), 9_236);
    }

    /// The two products are named by the checkpoint's scale plane, and each product and the encoding
    /// that feeds it name one activation layout: the gated product reads its operand's block scales
    /// in the linear form and the plain product in the swizzled one. The keys are the ones the
    /// encoded forward launches, as the store's own directory names them.
    ///
    /// A key that drifts from what a layer launches is the failure a build reports as
    /// `MissingKernel` after any amount of GPU time, and it is caught here on the host.
    #[test]
    fn the_encoded_products_name_the_checkpoint_s_scale_plane_and_their_activation_layouts() {
        let moe = quantized_block(2048, 768).expect("the encoded block");
        let arena = crate::Arena::counting();
        let f = Forward::recording(&arena);
        let hidden = arena
            .tensor(DType::BF16, Shape::new(&[15, 2048]))
            .expect("a counting arena");

        let (walked, launches) = crate::forward::recorded(|| moe.forward(&f, hidden));
        walked.expect("the recorded forward");

        let keys = |operator: &str| -> Vec<&str> {
            launches
                .iter()
                .filter(|(name, _)| name == operator)
                .map(|(_, key)| key.as_str())
                .collect()
        };
        assert_eq!(
            keys("gather_gemm_nvfp4"),
            [
                "hidden_2048_intermediate_768_experts_128_top_8_activation_silu_dtype_bf16_scale_f8e4m3_layout_linear_tile_128x128_cluster_1x1"
            ]
        );
        assert_eq!(
            keys("route_gemm_nvfp4"),
            [
                "hidden_2048_intermediate_768_experts_128_top_8_dtype_bf16_scale_f8e4m3_layout_swizzled_128x4_tile_128x128_cluster_1x1"
            ]
        );
        // The hidden rows are encoded for the gated product and the gated rows for the plain one.
        let encodings = keys("nvfp4_quantize");
        assert_eq!(encodings.len(), 2, "{encodings:?}");
        assert!(
            encodings[0].starts_with("input_bf16_columns_2048_layout_linear_"),
            "{encodings:?}"
        );
        assert!(
            encodings[1].starts_with("input_bf16_columns_768_layout_swizzled_128x4_"),
            "{encodings:?}"
        );
    }

    #[test]
    fn a_quantized_block_builds_from_the_same_geometry_as_a_dense_one() {
        let quantized = quantized_block(2048, 768).unwrap();
        assert_eq!(quantized.experts(), 128);
        assert_eq!(quantized.local_experts(), 128);
        assert_eq!(quantized.top(), 8);
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
            scales: weight(DType::F8E4M3, &[u64::from(experts), rows, columns / 16]),
            global_scale: weight(DType::F32, &[projections, u64::from(experts)]),
            layout: ops::fp4::ScaleLayout::Linear,
            encode_scale: weight(DType::F32, &[1]),
            input_scale: None,
            activation_scale: 1.0,
        };
        // Down's contraction is the gated width, so 512 experts' contraction does not fit 768.
        assert!(
            Moe::build(
                router,
                ExpertStacks::Encoded {
                    up_gate: Box::new(stack(1536, 2048, 2)),
                    down: Box::new(stack(2048, 512, 1)),
                },
                None,
                colocated(experts),
                &routing(experts, 8),
                None,
            )
            .is_err()
        );
    }

    #[test]
    fn a_block_of_128_experts_top_8_builds() {
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
            Moe::build(
                router,
                ExpertStacks::Dense { up_gate, down },
                None,
                colocated(128),
                &routing(128, 8),
                None,
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
            Moe::build(
                router,
                ExpertStacks::Dense {
                    up_gate: weight(DType::BF16, &[128, 1536, 2048]),
                    down: wrong_down,
                },
                None,
                colocated(128),
                &routing(128, 8),
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
            Moe::build(
                router,
                ExpertStacks::Dense { up_gate, down },
                Some(short_map),
                colocated(8),
                &routing(8, 2),
                None,
            )
            .is_err(),
            "a map of four entries cannot route eight experts"
        );
    }
}

#[cfg(test)]
mod parallel_tests {
    use super::tests::expert_parallel_block;
    use super::{Moe, WEIGHT_SCALE_LAYOUT};
    use ops::fp4::ScaleLayout;

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

    /// The block-scale layouts each product's kernels were built for, held against the packages.
    ///
    /// A product's planes are read by kernels that were **compiled** with a layout for each operand,
    /// and nothing checks the two against each other at runtime: the layouts are not options the
    /// runner's lookup compares (`GemmOptions` carries no `SfLayout`), so a package built for one
    /// form handed a plane in another reads every block factor at the wrong offset and answers
    /// plausible numbers. This is the one place the two statements meet.
    ///
    /// `mSfLayoutA` is the **weights** and `mSfLayoutB` the **tokens** — `BatchedGemmOptions.h`
    /// states both ("mSfLayoutA has to be R128cX", "Tokens need use SF linear layout when being
    /// routed") — and the two families disagree about the second only.
    #[test]
    fn a_products_package_states_the_scale_layouts_nn_publishes() {
        let kernels = std::path::Path::new(ops::KERNELS).to_path_buf();
        let cases = [
            (
                "gather_gemm_nvfp4",
                WEIGHT_SCALE_LAYOUT,
                Moe::GATED_TOKEN_SCALE_LAYOUT,
            ),
            (
                "route_gemm_nvfp4",
                WEIGHT_SCALE_LAYOUT,
                Moe::ACTIVATION_SCALE_LAYOUT,
            ),
        ];
        let name = |layout: ScaleLayout| match layout {
            ScaleLayout::Linear => "\"linear\"",
            ScaleLayout::Swizzled128X4 => "\"swizzled_128x4\"",
        };
        for (operator, weights, tokens) in cases {
            let package = kernels.join(operator).join("trtllm/package.toml");
            let stated = std::fs::read_to_string(&package)
                .unwrap_or_else(|error| panic!("{}: {error}", package.display()));
            for (option, expected) in [("mSfLayoutA", name(weights)), ("mSfLayoutB", name(tokens))]
            {
                let line = format!("{option} = {expected}");
                assert!(
                    stated.contains(&line),
                    "{operator}'s package does not state `{line}`: the kernels it publishes would \
                     read a plane nn publishes in another form, and nothing at launch would say so"
                );
            }
        }
    }
}
