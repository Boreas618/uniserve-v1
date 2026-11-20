//! The mixture-of-experts block of a transformer layer, and the shared expert beside it.
//!
//! ```text
//! logits  = router(x)                                  one logit per expert, replicated
//! routes  = topk_softmax(logits)                       each token's `top` experts and weights
//! out[t]  = Σ_routes weight · down_e(act(gate_e(x[t])) · up_e(x[t]))
//! ```
//!
//! The router's scores are a softmax over the experts or each expert's `sqrt(softplus(logit))`
//! ([`MoEConfig::scoring`]); a selection bias, when the checkpoint states one, decides which
//! experts a token takes without entering their weights, and a routed scaling factor multiplies
//! every weight ([`functional::Routing`]). The experts' gated activation may be clamped
//! ([`MoEConfig::limit`]: `act(min(gate, L)) · clamp(up, ±L)`).
//!
//! [`MoE`] is one module because its parts only make sense together: the routing decides which
//! expert's rows a token's routes go to, the sort turns that into the tiles the grouped products
//! read, and the combine sums the routes back with the FP32 weights the router produced
//! ([`crate::functional`]). The routing, the sort and the combine are the same for every
//! representation of the experts; only the two products differ.
//!
//! ## Loading
//!
//! A checkpoint's architecture stores a block's dense experts in one layout ([`ExpertLayout`]):
//! one module per expert (`<list>.<e>.<up|gate|down>.weight`) or two stacked tensors. Each
//! layout has its one mapping, and either reads them as two stacks in the arrangement the
//! products read: `[experts, 2 · intermediate, hidden]` with each expert's up rows before its
//! gate rows, and `[experts, hidden, intermediate]`. A calibrated NVFP4 checkpoint's experts
//! are read as packed stacks, which the representation the model's loader resolved decides
//! from the first held expert's weight; a calibrated export stores every packed expert as its
//! own module (ModelOpt's unified export), which is the one layout packed experts are read in.
//! A block-scaled export's MXFP4 experts are read the same way, into the stacks the MXFP4
//! products read ([`functional::gather_gemm_mxfp4`]); a rank holds whole MXFP4 experts, every
//! slot or its window of a tensor group that holds whole experts or of an expert group.
//!
//! ## Where the experts sit
//!
//! The router selects over the checkpoint's expert ids, and each route is then written as the
//! **slot** its expert is stored in ([`MoEConfig::order`]), so which experts a token picks does
//! not depend on where they lie. A deployment that redistributes its experts states another
//! order, and the same checkpoint loads under it.
//!
//! ## Expert load
//!
//! A block of the model's own stack is named among its routed layers ([`MoEConfig::name`]),
//! and a serving forward counts, by that name, the tokens each of its experts takes: the top-k of
//! [`MoE::forward`] adds each route of a live token into the counters execution's context lends
//! the layer ([`ops::moe::ExpertLoad`]), at the expert's own id, so the counts do not depend on
//! the order either. A forward under a context that lends none, and [`MoE::route`], count
//! nothing.
//!
//! ## Parallelism
//!
//! Over a tensor-parallel group ([`MoEConfig::split`]) a rank holds its block of every expert's
//! intermediate width, so its output is a partial sum that [`MoE::forward`] reduces over the
//! group. Packed experts are split the same way, each rank's block padded with zero rows and
//! columns to whole row tiles of the grouped products ([`Nvfp4Share`]): the products then run
//! at the padded width, and the zeros contribute nothing.
//!
//! A tensor-parallel group may divide the slots instead ([`MoEConfig::tensor_experts`]): rank
//! `r` of `P` holds the window `r` of the slots, every expert at its full width, and runs it
//! over every token of the batch. The sort gives a route to another rank's expert no row, so it
//! contributes nothing to the rank's sum, and the group's reduction adds each route's product
//! exactly once.
//!
//! Over an expert group ([`MoEConfig::exchange`]) a rank holds a window of the slots and
//! the block runs through the group's exchange, whose rows execution lends
//! ([`ExpertExchange`]): the routes travel to the ranks that hold their experts, and each
//! token's sums come back. Calibrated NVFP4 experts run in the exchange's one fused launch;
//! MXFP4 experts run either between its dispatch and combine or, when the placement states it
//! ([`MoEConfig::fused_exchange`]), in its fused launch, which also runs the block's shared
//! expert and sums both in FP32 (`ops::moe::exchange`'s module documentation states the formula).
//!
//! ## Shared expert
//!
//! A block may own an ungated shared expert every token passes through ([`MoEConfig::shared`],
//! DeepSeek's): its output is added to the routed output after the block's reduction, or the
//! fused launch computes it beside the routes. A shared expert with a gate, or one beside a
//! block that does not own it, is a [`SharedExpert`] of the model's own.

use std::ops::Range;

use ops::moe::exchange::{self, Descriptor as ExchangeDescriptor, ExpertWeights};
use ops::moe::grouped::{
    GatherGemmNvfp4Descriptor, RouteGemmNvfp4Descriptor, WEIGHT_SCALE_LAYOUT, packed_width,
};
use ops::moe::mxfp4::{self, GatherGemmMxfp4Descriptor, RouteGemmMxfp4Descriptor};
use tensor::{DType, Device, Shape, Tensor};

use crate::capability::ExpertExchange;
use crate::collective::{self, Split};
use crate::error::{Error, Result};
use crate::functional::{self, Activation, Routing, Scoring, WeightOrder};
use crate::modules::linear::{Linear, LinearConfig, Parallel};
use crate::modules::mlp::{GateUp, Mlp, MlpConfig};
use crate::quantization::{Mxfp4Stack, Mxfp8Weight, Nvfp4Share, Nvfp4Stack, Scheme};
use crate::var_builder::VarBuilder;
use crate::weights::{PartBlock, WeightLayout, WeightSource, extents};

/// The class the per-expert tables of a fused exchange are filed under: the producer of their
/// retained keys. Its text is part of every stored key, so it does not follow the type's name.
const PRODUCER: &str = "nn::Moe";

/// What a checkpoint states of a [`MoE`]: its widths, its routing, the names of its modules,
/// and this rank's share of its experts.
#[derive(Clone, Debug, PartialEq)]
pub struct MoEConfig {
    /// The width the experts read and write.
    pub hidden: u32,
    /// One expert's gated width, before any split.
    pub intermediate: u32,
    /// Routed experts of the block: the router's outputs.
    pub experts: u32,
    /// Experts each token is routed to.
    pub top: u32,
    /// The gate's activation.
    pub activation: Activation,
    /// Whether the selected experts' weights are renormalized to sum to one.
    pub renormalize: bool,
    /// How the router's logits become the scores it selects experts by and weighs them with.
    pub scoring: Scoring,
    /// The name of the router module's `[experts]` FP32 selection bias, added to the scores the
    /// selection is made by and to no weight, when the checkpoint states one.
    pub selection_bias: Option<&'static str>,
    /// The factor every route's weight is multiplied by after the renormalization, when the
    /// model states one (a routed scaling factor).
    pub routed_scale: Option<f32>,
    /// The clamp of the experts' gated activation, `act(min(gate, L)) · clamp(up, ±L)`, when the
    /// model states one.
    pub limit: Option<f32>,
    /// The router's module name.
    pub router: &'static str,
    /// The element type the router states its logits in: the activations' when `None`, or a
    /// wider one (F32) for a model whose reference routes on FP32 logits. The product reads the
    /// rows and its weight in the activations' type and accumulates in FP32, so FP32 logits of
    /// BF16 rows are the reference's FP32 product of the widened values up to the order of its
    /// sum: the products of BF16 values are exact in FP32.
    pub router_logits: Option<DType>,
    /// The name of the module list that holds the experts: expert `e`'s projections are
    /// `<list>.<e>.<up|gate|down>`.
    pub list: &'static str,
    /// An expert's up projection's module name.
    pub up: &'static str,
    /// An expert's gate projection's module name.
    pub gate: &'static str,
    /// An expert's down projection's module name.
    pub down: &'static str,
    /// How the checkpoint's architecture stores the dense experts.
    pub layout: ExpertLayout,
    /// The expert each slot holds: a permutation of the experts, in which the stacks are laid
    /// out and by which an expert group's ranks divide them. The experts' own order when a
    /// deployment has not redistributed them.
    pub order: Vec<u32>,
    /// The tensor-parallel group's split of the block, whose ranks' outputs the forward sums:
    /// this rank's block of each expert's intermediate width, or its window of the slots when
    /// the group holds whole experts ([`Self::tensor_experts`]).
    pub split: Split,
    /// Whether the tensor-parallel group divides the slots instead of each expert's width:
    /// rank `r` of [`Self::split`] holds the window `r` of the slots, every expert at its full
    /// intermediate width, and runs it over every token. The expert group's split is then
    /// whole, since no exchange moves a token.
    pub tensor_experts: bool,
    /// The expert group's split of the slots: this rank holds one contiguous window of them.
    pub exchange: Split,
    /// The block's module path among the model's routed layers
    /// ([`crate::causal_lm::expert_module`]): the name a deployment's expert order and
    /// execution's expert load state the layer by. `None` for a block outside the model's own
    /// decoder stack, such as a drafter's, which neither names.
    pub name: Option<String>,
    /// The ungated shared expert the block owns and runs beside its routed experts, when it owns
    /// one: added after the routed output, or computed by the fused launch of an exchange that
    /// runs the block in one ([`Self::fused_exchange`]). Its MLP is held whole on every rank.
    pub shared: Option<SharedExpertConfig>,
    /// Whether MXFP4 experts over an expert group run in the exchange's one fused launch
    /// (`TextPlacement::fused_exchange`), so they and the shared expert load in the arrangement
    /// that launch reads; otherwise they run between the exchange's dispatch and combine. A
    /// block of other experts, or one no expert group divides, reads no such arrangement.
    pub fused_exchange: bool,
}

/// How a checkpoint's architecture stores a block's dense experts, as its schema declares it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertLayout {
    /// One module per expert under the module list: `<list>.<e>.<up|gate|down>`.
    Modules,
    /// Two stacked tensors under the module list (transformers' fused experts): `gate_up` is
    /// `[experts, 2 · intermediate, hidden]` with each expert's gate rows before its up rows,
    /// and `down` is `[experts, hidden, intermediate]`.
    Stacked {
        /// The stacked first projection's tensor name.
        gate_up: &'static str,
        /// The stacked down projection's tensor name.
        down: &'static str,
    },
}

/// The representation a block's expert stacks are held in.
enum Experts {
    /// `[local, 2 · intermediate, hidden]`, up rows first, and `[local, hidden, intermediate]`.
    Dense { up_gate: Tensor, down: Tensor },
    /// Calibrated NVFP4 stacks in the arrangement the routed products read: the gated stack's
    /// halves interleaved and shuffled, the down stack shuffled, the block scales swizzled.
    Packed {
        up_gate: Nvfp4Stack,
        down: Nvfp4Stack,
    },
    /// MXFP4 stacks in the arrangement the MXFP4 routed products read, with each held expert's
    /// `[local]` F32 clamp of the gated activation.
    Mxfp4 {
        up_gate: Mxfp4Stack,
        down: Mxfp4Stack,
        limit: Tensor,
    },
    /// Calibrated NVFP4 stacks in the arrangement an expert group's fused exchange reads (each
    /// gate block before its up block), with the three `[local]` FP32 tables its kernel
    /// multiplies in.
    FusedNvfp4 {
        up_gate: Nvfp4Stack,
        down: Nvfp4Stack,
        /// Dequantizes the first product: the weight's tensor scale times its rows' scale.
        up_gate_alpha: Tensor,
        /// Dequantizes the second product.
        down_alpha: Tensor,
        /// The global scale the gated rows are encoded with.
        gated_scale: Tensor,
    },
    /// MXFP4 stacks in the arrangement an expert group's fused MXFP4 exchange reads
    /// (`ExchangeDescriptor::MXFP4_UP_GATE_VALUES`, `MXFP4_DOWN_VALUES`, `MXFP4_SCALES`), with
    /// the gated activation's clamp the launch is compiled for.
    FusedMxfp4 {
        up_gate: Mxfp4Stack,
        down: Mxfp4Stack,
        limit: Option<f32>,
    },
}

/// The shared expert a block owns ([`MoEConfig::shared`]).
enum Shared {
    /// Its own MLP, whose output is added after the routed block's (boxed: an MLP's projections
    /// are larger than the fused launch's two matrices).
    Module(Box<SharedExpert>),
    /// Its gated stack and down projection as one MXFP8 matrix each, in the arrangement the
    /// fused MXFP4 exchange reads, which computes it.
    Fused {
        up_gate: Mxfp8Weight,
        down: Mxfp8Weight,
    },
}

/// A mixture-of-experts block.
pub struct MoE {
    router: Linear,
    experts: Experts,
    /// `[experts]` U32: the slot of each expert, by which the top-k writes each route's weight
    /// row.
    map: Tensor,
    /// Experts the router selects over.
    count: u32,
    top: u32,
    /// This rank's block of an expert's gated width.
    intermediate: u32,
    activation: Activation,
    renormalize: bool,
    scoring: Scoring,
    /// `[experts]` F32: the router's selection bias, when the checkpoint states one.
    selection_bias: Option<Tensor>,
    /// `[experts]` F32: each expert's weight factor, the routed scaling factor, when the model
    /// states one.
    routed_scale: Option<Tensor>,
    /// The slots this rank holds; every slot when no group divides them.
    local: Range<u32>,
    /// Whether an expert group divides the slots, so the forward runs through its exchange.
    exchanges: bool,
    /// Ranks of the tensor-parallel group the experts' gated width or slots are split over:
    /// above one, the output is a partial sum the forward reduces.
    reduce: u32,
    /// The layer's name among the model's routed layers, by which its forwards find the
    /// expert load execution lends ([`MoEConfig::name`]).
    name: Option<String>,
    /// The router's logits' element type, the activations' when `None`
    /// ([`MoEConfig::router_logits`]).
    logits: Option<DType>,
    /// The shared expert the block owns.
    shared: Option<Shared>,
}

impl MoE {
    /// The block the module at `vb` states, at this rank's share of it.
    ///
    /// The router is replicated: every rank routes a token the same way. Over an expert group
    /// the rank holds its window of the slots; over a tensor-parallel group, its rows of every
    /// gate and up projection and its columns of every down projection, or its window of the
    /// slots, every expert whole, when the group holds whole experts. A calibrated NVFP4
    /// checkpoint's experts are read packed: without an expert group in the arrangement the
    /// routed products read, and over an expert group in the fused exchange's, with the
    /// per-expert scalars its kernel multiplies in.
    ///
    /// # Errors
    ///
    /// When the routing is outside what the router's kernel holds, the order is no permutation
    /// of the experts, a split does not divide the experts or their width, a tensor group that
    /// holds whole experts sits beside an expert group, packed experts are split by a
    /// tensor-parallel group in blocks the products cannot read, or a tensor is missing or of
    /// another extent.
    pub fn load(config: &MoEConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let refuse = |problem: String| Error::weight(vb.path(), problem);
        let (experts, top) = (config.experts, config.top);
        let exchange = config.exchange;

        // One group divides the slots: a tensor group that holds whole experts runs them over
        // every token, so no exchange beside it moves one.
        if config.tensor_experts && exchange.size() > 1 {
            return Err(refuse(format!(
                "a tensor group of {} that holds whole experts and an expert group of {} both \
                 divide the slots: one group divides them",
                config.split.size(),
                exchange.size()
            )));
        }
        // The tensor group's split as the share of each expert's width (`split`) and as the
        // window of the slots (`windows`): the window is the expert group's unless the tensor
        // group holds whole experts.
        let (split, windows, holder) = if config.tensor_experts {
            (Split::WHOLE, config.split, "tensor")
        } else {
            (config.split, exchange, "expert")
        };

        if experts == 0 || top == 0 || top > experts {
            return Err(refuse(format!(
                "a top of {top} is outside 1..={experts} experts"
            )));
        }

        let mut named = config.order.clone();
        named.sort_unstable();
        if !named.iter().copied().eq(0..experts) {
            return Err(refuse(format!(
                "an order of {} slots is not a permutation of {experts} experts",
                config.order.len()
            )));
        }
        if !experts.is_multiple_of(windows.size()) {
            return Err(refuse(format!(
                "rank {} of the {holder} group of {} does not hold a share of {experts} experts",
                windows.rank(),
                windows.size()
            )));
        }
        if !config.intermediate.is_multiple_of(split.size()) {
            return Err(refuse(format!(
                "rank {} of a tensor group of {} does not hold a share of an expert's {} \
                 intermediate columns",
                split.rank(),
                split.size(),
                config.intermediate
            )));
        }

        let held = experts / windows.size();
        let local = windows.rank() * held..(windows.rank() + 1) * held;
        let window = &config.order[local.start as usize..local.end as usize];
        let intermediate = config.intermediate / split.size();

        let router = Linear::new(
            vb.pp(config.router)
                .get((experts as usize, config.hidden as usize), "weight")?,
            None,
            WeightOrder::NxK,
        )?;

        // The router's selection bias and its routed scaling factor, one F32 per expert.
        let selection_bias = config
            .selection_bias
            .map(|name| {
                vb.pp(config.router)
                    .to_dtype(DType::F32)
                    .get(experts as usize, name)
            })
            .transpose()?;
        let routed_scale = config
            .routed_scale
            .map(|factor| -> Result<Tensor> {
                let bytes: Vec<u8> = std::iter::repeat_n(factor.to_le_bytes(), experts as usize)
                    .flatten()
                    .collect();
                vb.loader().constant(
                    &vb.name("routed_scale"),
                    DType::F32,
                    &Shape::from(experts as usize),
                    &bytes,
                    PRODUCER,
                )
            })
            .transpose()?;

        // The slot of each expert: the inverse of the order, as the top-k reads it.
        let slots = Tensor::arange(0_u32, experts, &Device::Cpu)?;
        let order = Tensor::new(config.order.as_slice(), &Device::Cpu)?;
        let map = vb.constant_from(
            "expert_slots",
            &slots.zeros_like()?.scatter(&order, &slots, 0)?,
        )?;

        let list = vb.pp(config.list);
        let scheme = list.pp(window[0]).pp(config.up).scheme("weight");
        if config.limit.is_some() && scheme != Scheme::Mxfp4 {
            return Err(refuse(
                "a clamped gated activation is fused into the MXFP4 products alone, and these \
                 experts are not MXFP4"
                    .to_owned(),
            ));
        }
        let packed = scheme == Scheme::Nvfp4;
        let (stacks, intermediate) = if scheme == Scheme::Mxfp4 {
            // A rank holds whole MXFP4 experts: its window of the slots, over a tensor group that
            // holds whole experts or over an expert group, whose exchange carries BF16 rows.
            if split.size() > 1 || config.activation != Activation::Silu {
                return Err(refuse(format!(
                    "MXFP4 experts are held whole and gated by SiLU: a tensor group of {} \
                     splitting each expert's width, or {} gating, has no MXFP4 product",
                    split.size(),
                    config.activation
                )));
            }
            let fused = config.fused_exchange && exchange.size() > 1;
            (Self::load_mxfp4(config, &vb, window, fused)?, intermediate)
        } else if packed {
            // A rank's block of packed experts: whole bytes and block scales of every row,
            // padded to the width the products compute over.
            let padded = packed_width(intermediate);
            let share = (split.size() > 1).then_some((
                u64::from(split.rank() * intermediate),
                u64::from(intermediate),
                u64::from(padded),
            ));
            if share.is_some() && (exchange.size() > 1 || !intermediate.is_multiple_of(16)) {
                return Err(refuse(format!(
                    "packed experts split over a tensor group of {} hold blocks of {intermediate} \
                     columns: whole block scales of sixteen, and no expert group beside it",
                    split.size()
                )));
            }

            let width = if share.is_some() {
                padded
            } else {
                intermediate
            };
            (
                Self::load_packed(config, &vb, window, exchange.size() > 1, share)?,
                width,
            )
        } else {
            let (up_gate, down) = Self::load_dense(config, &vb, window, split)?;
            (Experts::Dense { up_gate, down }, intermediate)
        };

        let shared = config
            .shared
            .as_ref()
            .map(|shared| Self::load_shared(config, shared, &vb, &stacks))
            .transpose()?;

        Ok(Self {
            router,
            experts: stacks,
            map,
            count: experts,
            top,
            intermediate,
            activation: config.activation,
            renormalize: config.renormalize,
            scoring: config.scoring,
            selection_bias,
            routed_scale,
            local,
            exchanges: exchange.size() > 1,
            // The tensor group sums its ranks' outputs whether it divides each expert's width
            // or the slots.
            reduce: config.split.size(),
            name: config.name.clone(),
            logits: config.router_logits,
            shared,
        })
    }

    /// The shared expert `shared` states, in the representation the block's `stacks` run it in:
    /// in the fused MXFP4 launch's arrangement beside fused MXFP4 experts, its own module
    /// otherwise.
    fn load_shared(
        config: &MoEConfig,
        shared: &SharedExpertConfig,
        vb: &VarBuilder<'_, '_>,
        stacks: &Experts,
    ) -> Result<Shared> {
        let refuse = |problem: String| Error::weight(vb.path(), problem);
        if shared.gate.is_some() {
            return Err(refuse(
                "a block owns an ungated shared expert alone; a gated one is a model's own                  SharedExpert"
                    .to_owned(),
            ));
        }
        if !matches!(stacks, Experts::FusedMxfp4 { .. }) {
            return Ok(Shared::Module(Box::new(SharedExpert::load(
                shared,
                vb.clone(),
            )?)));
        }

        // The launch runs shared experts of the routed width under the routed experts' clamp
        // and gating, its weights whole.
        let mlp = &shared.mlp;
        let GateUp::Split { gate, up } = mlp.gate_up else {
            return Err(refuse(
                "the fused exchange reads a shared expert's gate and up projections as two                  weights"
                    .to_owned(),
            ));
        };
        if mlp.hidden != config.hidden
            || !mlp.intermediate.is_multiple_of(config.intermediate)
            || mlp.activation != config.activation
            || mlp.limit != config.limit
            || mlp.bias
            || mlp.split.size() > 1
        {
            return Err(refuse(format!(
                "the fused exchange runs shared experts of the routed width under the routed                  gating and clamp, whole and without bias; this one is {} wide over rows of {},                  {} gated, clamped at {:?}",
                mlp.intermediate, mlp.hidden, mlp.activation, mlp.limit
            )));
        }

        let module = vb.pp(shared.module);
        let weight = |name: &str| module.pp(name).name("weight");
        let loader = vb.loader();
        // The gated stack is `[up; gate]`, which the launch's arrangement leads with the gate.
        let up_gate = loader.mxfp8_stack(
            &vb.name("shared.up_gate"),
            &[&weight(up), &weight(gate)],
            (
                ExchangeDescriptor::MXFP4_UP_GATE_VALUES,
                ExchangeDescriptor::MXFP4_SCALES,
            ),
        )?;
        let down = loader.mxfp8_stack(
            &vb.name("shared.down"),
            &[&weight(mlp.down)],
            (
                ExchangeDescriptor::MXFP4_DOWN_VALUES,
                ExchangeDescriptor::MXFP4_SCALES,
            ),
        )?;
        Ok(Shared::Fused { up_gate, down })
    }

    /// The dense stacks of the experts `window` names, this rank's block `split` of each
    /// expert's width, by the mapping of the layout the architecture declares
    /// ([`MoEConfig::layout`]).
    fn load_dense(
        config: &MoEConfig,
        vb: &VarBuilder<'_, '_>,
        window: &[u32],
        split: Split,
    ) -> Result<(Tensor, Tensor)> {
        let (hidden, intermediate) = (u64::from(config.hidden), u64::from(config.intermediate));
        let local_intermediate = intermediate / u64::from(split.size());
        let first = u64::from(split.rank()) * local_intermediate;
        let count = window.len() as u64;
        let list = vb.pp(config.list);

        let up_gate_shape = extents(&[count, 2 * local_intermediate, hidden]);
        let down_shape = extents(&[count, hidden, local_intermediate]);

        // A gate or up projection is `[intermediate, hidden]`, so a rank's block is its rows; a
        // down projection is `[hidden, intermediate]`, so a rank's block is its columns of
        // every row.
        let rows_of = |start: u64| PartBlock {
            start: start + first * hidden,
            rows: local_intermediate,
            columns: hidden,
            row_stride: hidden,
        };
        let columns_of = |start: u64| PartBlock {
            start: start + first,
            rows: hidden,
            columns: local_intermediate,
            row_stride: intermediate,
        };

        let (up_gate_name, up_gate, down_name, down) = match config.layout {
            ExpertLayout::Stacked { gate_up, down } => {
                let (gate_up, down) = (list.name(gate_up), list.name(down));

                // One expert's slab of the stacked first projection is its gate rows, then its
                // up rows; the product reads up before gate.
                let mut up_gate_parts = Vec::with_capacity(window.len() * 2);
                let mut down_parts = Vec::with_capacity(window.len());
                for &expert in window {
                    let slab = u64::from(expert) * 2 * intermediate * hidden;

                    up_gate_parts.push((gate_up.clone(), rows_of(slab + intermediate * hidden)));
                    up_gate_parts.push((gate_up.clone(), rows_of(slab)));
                    down_parts.push((
                        down.clone(),
                        columns_of(u64::from(expert) * hidden * intermediate),
                    ));
                }

                (
                    list.name("up_gate"),
                    WeightSource::Blocks {
                        parts: up_gate_parts,
                        shape: up_gate_shape.clone(),
                    },
                    list.name("down"),
                    WeightSource::Blocks {
                        parts: down_parts,
                        shape: down_shape.clone(),
                    },
                )
            }

            ExpertLayout::Modules => {
                let weight = |expert: u32, module: &str| list.pp(expert).pp(module).name("weight");
                let up_gate_parts = window
                    .iter()
                    .flat_map(|&expert| [weight(expert, config.up), weight(expert, config.gate)])
                    .collect();
                let down_parts = window
                    .iter()
                    .map(|&expert| weight(expert, config.down))
                    .collect();
                let block = |block: PartBlock| (split.size() > 1).then_some(block);

                (
                    vb.name("experts.up_gate"),
                    WeightSource::Parts {
                        parts: up_gate_parts,
                        shape: up_gate_shape.clone(),
                        block: block(rows_of(0)),
                    },
                    vb.name("experts.down"),
                    WeightSource::Parts {
                        parts: down_parts,
                        shape: down_shape.clone(),
                        block: block(columns_of(0)),
                    },
                )
            }
        };

        Ok((
            vb.load(
                &up_gate_name,
                &up_gate,
                WeightLayout::UP_GATE,
                &up_gate_shape,
                false,
            )?,
            vb.load(&down_name, &down, WeightLayout::DOWN, &down_shape, false)?,
        ))
    }

    /// The packed stacks of the experts `window` names: in the fused exchange's arrangement
    /// with its per-expert tables when the experts are `exchanged` over a group, and in the
    /// routed products' otherwise. `share` is a tensor-parallel rank's block of every expert's
    /// intermediate width, `(first, count, padded)` in columns, or `None` for whole experts.
    fn load_packed(
        config: &MoEConfig,
        vb: &VarBuilder<'_, '_>,
        window: &[u32],
        exchanged: bool,
        share: Option<(u64, u64, u64)>,
    ) -> Result<Experts> {
        let loader = vb.loader();
        let names = loader.representation().names();
        let list = vb.pp(config.list);
        // Each stack in the arrangements its reading operator declares: the fused exchange's,
        // or the routed products'.
        let (gated_layout, down_layout) = if exchanged {
            (
                (
                    exchange::Descriptor::FUSED_UP_GATE_VALUES,
                    exchange::Descriptor::FUSED_SCALES,
                ),
                (
                    exchange::Descriptor::FUSED_DOWN_VALUES,
                    exchange::Descriptor::FUSED_SCALES,
                ),
            )
        } else {
            (
                (
                    GatherGemmNvfp4Descriptor::WEIGHT_VALUES,
                    WEIGHT_SCALE_LAYOUT,
                ),
                (RouteGemmNvfp4Descriptor::WEIGHT_VALUES, WEIGHT_SCALE_LAYOUT),
            )
        };

        // The checkpoint's block scales are linear, and the published tables are built from
        // them once here.
        // A gate or up projection is `[intermediate, hidden]`, so a rank's block is its rows;
        // a down projection is `[hidden, intermediate]`, so a rank's block is its columns.
        let rows = share.map(|(first, count, padded)| Nvfp4Share::Rows {
            first,
            count,
            padded,
        });
        let columns = share.map(|(first, count, padded)| Nvfp4Share::Columns {
            first,
            count,
            padded,
        });
        let (up_gate, up_scales) = loader.nvfp4_stack_share(
            &vb.name("experts.up_gate"),
            list.path(),
            &[config.up, config.gate],
            window,
            config.experts,
            names,
            gated_layout,
            true,
            rows,
        )?;
        let (down, down_scales) = loader.nvfp4_stack_share(
            &vb.name("experts.down"),
            list.path(),
            &[config.down],
            window,
            config.experts,
            names,
            down_layout,
            true,
            columns,
        )?;
        let width = share.map_or(config.intermediate as usize, |(_, _, padded)| {
            padded as usize
        });
        Self::check_packed(config, vb, window.len(), width, &up_gate, &down)?;

        if !exchanged {
            return Ok(Experts::Packed { up_gate, down });
        }

        // The fused kernel's three scalars per expert: each product dequantizes with the
        // weight's tensor scale times its rows' activation scale, and the gated rows encode
        // with the second activation scale's reciprocal. A fused first projection carries one
        // tensor scale per expert, which up and gate share.
        let up_alpha: Vec<f32> = up_scales
            .chunks(2)
            .map(|scales| scales[0] * up_gate.activation_scale)
            .collect();
        let down_alpha: Vec<f32> = down_scales
            .iter()
            .map(|scale| scale * down.activation_scale)
            .collect();
        let gated = vec![down.activation_scale.recip(); window.len()];

        let table = |what: &str, values: Vec<f32>| -> Result<Tensor> {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect();

            loader.constant(
                &vb.name(&format!("experts.{what}")),
                DType::F32,
                &Shape::from(window.len()),
                &bytes,
                PRODUCER,
            )
        };
        Ok(Experts::FusedNvfp4 {
            up_gate_alpha: table("up_gate_alpha", up_alpha)?,
            down_alpha: table("down_alpha", down_alpha)?,
            gated_scale: table("gated_scale", gated)?,
            up_gate,
            down,
        })
    }

    /// The MXFP4 stacks of the experts `window` names, whole: in the arrangement the fused MXFP4
    /// exchange reads when `fused`, with the clamp its launch is compiled for; otherwise in the
    /// arrangement the MXFP4 routed products read, with the `[held]` F32 table of the gated
    /// activation's clamp: the block's limit, or none (an infinite one) without it.
    fn load_mxfp4(
        config: &MoEConfig,
        vb: &VarBuilder<'_, '_>,
        window: &[u32],
        fused: bool,
    ) -> Result<Experts> {
        let loader = vb.loader();
        let list = vb.pp(config.list);
        let (up_gate_layout, down_layout) = if fused {
            (
                (
                    ExchangeDescriptor::MXFP4_UP_GATE_VALUES,
                    ExchangeDescriptor::MXFP4_SCALES,
                ),
                (
                    ExchangeDescriptor::MXFP4_DOWN_VALUES,
                    ExchangeDescriptor::MXFP4_SCALES,
                ),
            )
        } else {
            (
                (
                    GatherGemmMxfp4Descriptor::WEIGHT_VALUES,
                    mxfp4::WEIGHT_SCALE_LAYOUT,
                ),
                (
                    RouteGemmMxfp4Descriptor::WEIGHT_VALUES,
                    mxfp4::WEIGHT_SCALE_LAYOUT,
                ),
            )
        };
        let up_gate = loader.mxfp4_stack(
            &vb.name("experts.up_gate"),
            list.path(),
            &[config.up, config.gate],
            window,
            up_gate_layout,
        )?;
        let down = loader.mxfp4_stack(
            &vb.name("experts.down"),
            list.path(),
            &[config.down],
            window,
            down_layout,
        )?;

        let (hidden, intermediate) = (config.hidden as usize, config.intermediate as usize);
        let held = window.len();
        if up_gate.codes.dims() != [held, 2 * intermediate, hidden / 2]
            || down.codes.dims() != [held, hidden, intermediate / 2]
        {
            return Err(Error::weight(
                vb.path(),
                format!(
                    "the MXFP4 stacks are {:?} and {:?}, [{held}, {}, {}] and [{held}, {hidden}, \
                     {}] expected",
                    up_gate.codes.dims(),
                    down.codes.dims(),
                    2 * intermediate,
                    hidden / 2,
                    intermediate / 2
                ),
            ));
        }

        if fused {
            return Ok(Experts::FusedMxfp4 {
                up_gate,
                down,
                limit: config.limit,
            });
        }

        let limit = config.limit.unwrap_or(f32::INFINITY);
        let bytes: Vec<u8> = std::iter::repeat_n(limit.to_le_bytes(), held)
            .flatten()
            .collect();
        let limit = loader.constant(
            &vb.name("experts.limit"),
            DType::F32,
            &Shape::from(held),
            &bytes,
            PRODUCER,
        )?;
        Ok(Experts::Mxfp4 {
            up_gate,
            down,
            limit,
        })
    }

    /// Checks that packed stacks of `held` experts of `intermediate` columns are the extents
    /// and types the block's config states and its products read.
    fn check_packed(
        config: &MoEConfig,
        vb: &VarBuilder<'_, '_>,
        held: usize,
        intermediate: usize,
        up_gate: &Nvfp4Stack,
        down: &Nvfp4Stack,
    ) -> Result<()> {
        let refuse = |problem: String| Error::weight(vb.path(), problem);
        let hidden = config.hidden as usize;

        let up_gate_dims = [held, 2 * intermediate, hidden / 2];
        let down_dims = [held, hidden, intermediate / 2];
        if up_gate.values.dims() != up_gate_dims || down.values.dims() != down_dims {
            return Err(refuse(format!(
                "the packed stacks are {:?} and {:?}, {up_gate_dims:?} and {down_dims:?} expected",
                up_gate.values.dims(),
                down.values.dims()
            )));
        }

        // The block scales are the checkpoint's own one-byte type, which each product's
        // descriptor states; the tensor scales are one packed table per projection, and a
        // table of another shape would dequantize one projection with another's scale.
        for (name, stack, projections) in [("up/gate", up_gate, 2), ("down", down, 1)] {
            if !matches!(stack.scales.dtype(), DType::U8 | DType::F8E4M3) {
                return Err(refuse(format!(
                    "the {name} block scales are {}, and a product's kernel reads one-byte \
                     scales: F8_E4M3 or U8",
                    ops::abi::name(stack.scales.dtype())
                )));
            }
            if stack.global_scale.dims() != [projections, held] {
                return Err(refuse(format!(
                    "the {name} tensor scales are {:?}, [{projections}, {held}] expected: one \
                     packed table per projection",
                    stack.global_scale.dims()
                )));
            }
        }
        Ok(())
    }

    /// Experts the router selects over.
    pub const fn experts(&self) -> u32 {
        self.count
    }

    /// The experts this rank holds.
    pub const fn local_experts(&self) -> u32 {
        self.local.end - self.local.start
    }

    /// The experts each token routes to.
    pub const fn top(&self) -> u32 {
        self.top
    }

    /// The router's choice for `x [tokens, hidden]`, and nothing else: `(ids, weights)`, the
    /// slots of each token's `top` experts and their weights, `[tokens, top]` U32 and FP32.
    ///
    /// No expert runs, so a comparison of routing against a reference is one router product and
    /// its top-k: it keeps a wrong id, a right id with a wrong weight and a wrong expert
    /// product apart, which a whole block's output would confound.
    ///
    /// # Errors
    ///
    /// Rows of another width, or an operator's refusal.
    pub fn route(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        self.select(x, None, None)
    }

    /// The forward's routing of `x`: [`MoE::route`], with each route of the batch's live token
    /// rows (`live_tokens`, every row without it) counted into the layer's expert load when the
    /// installed context lends one under the block's name.
    fn routed_ids(&self, x: &Tensor, live_tokens: Option<&Tensor>) -> Result<(Tensor, Tensor)> {
        let load = self.name.as_deref().and_then(ops::moe::layer_load);

        self.select(x, live_tokens, load.as_ref())
    }

    /// The router's product and top-k over `x`, adding the live routes into `load` when given.
    fn select(
        &self,
        x: &Tensor,
        live_tokens: Option<&Tensor>,
        load: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let routing = Routing {
            top: self.top,
            scoring: self.scoring,
            renormalize: self.renormalize,
            bias: self.selection_bias.as_ref(),
            scale: self.routed_scale.as_ref(),
        };
        // The router's product over the rows as they stand, its logits in their stated type.
        let logits = functional::linear_to(
            x,
            self.router.weight(),
            None,
            WeightOrder::NxK,
            self.logits.unwrap_or(x.dtype()),
        )?;
        functional::topk_softmax(&logits, &routing, Some(&self.map), live_tokens, load)
    }

    /// `Σ_routes weight · expert(x)` over `x [tokens, hidden]`, summed over the tensor-parallel
    /// group: `[tokens, hidden]` in `x`'s element type.
    ///
    /// `live_tokens` is the batch's count of token rows it fills
    /// ([`TokenBatch::live_tokens`](crate::TokenBatch::live_tokens)), when execution states it:
    /// the rows of a captured bucket past the batch route nowhere and their output is zeros.
    /// `exchange` is the expert group's exchange, lent when the block's experts are split over
    /// one; the block then runs through it. Experts a tensor-parallel group holds whole run
    /// over every token of `x` without one, each rank's routes to the others' experts adding
    /// nothing before the group's sum.
    ///
    /// # Errors
    ///
    /// When `x` does not fit the block, an operator refuses, experts split over an expert group
    /// run without its exchange, experts a tensor-parallel group holds whole run with one, or
    /// experts split over a tensor-parallel group run in a forward that has no group.
    pub fn forward(
        &self,
        x: &Tensor,
        live_tokens: Option<&Tensor>,
        exchange: Option<&ExpertExchange>,
    ) -> Result<Tensor> {
        let &[_, width] = x.dims() else {
            return Err(Error::shape(
                "MoE",
                format!("{:?} is not [tokens, hidden]", x.dims()),
            ));
        };
        if width != self.router.k() as usize {
            return Err(Error::shape(
                "MoE",
                format!(
                    "rows of {width} elements against experts of {}",
                    self.router.k()
                ),
            ));
        }

        // A tensor group that holds whole experts runs every token over its own window: its
        // ranks hold the same tokens, so a lent exchange would add their routes again.
        let tensor_window = self.reduce > 1 && self.local_experts() != self.count;
        if tensor_window && !self.exchanges && exchange.is_some() {
            return Err(Error::Unsupported(
                "MoE: experts held whole by a tensor group run over every token, and the \
                 forward lends an expert exchange"
                    .to_owned(),
            ));
        }
        let out = if self.exchanges || exchange.is_some() {
            self.exchanged(x, live_tokens, exchange)?
        } else {
            self.routed(x, live_tokens)?
        };

        if self.reduce > 1 {
            // One rank's block of every expert's width, or its window of the slots: the group
            // sums the ranks'.
            if collective::split().is_none() {
                return Err(Error::shape(
                    "MoE",
                    format!(
                        "experts split over {} tensor ranks run in a forward that has no group \
                         to reduce over",
                        self.reduce
                    ),
                ));
            }
            collective::all_reduce(&out)?;
        }

        // The shared expert is whole on every rank: it joins the reduced output, unless the
        // fused launch computed it.
        match &self.shared {
            Some(Shared::Module(shared)) => shared.forward(x, out),
            Some(Shared::Fused { .. }) | None => Ok(out),
        }
    }

    /// What an expert group's fused launch runs of this block, as its experts were loaded:
    /// the routed experts' representation, the shared experts it computes and the clamp;
    /// [`exchange::FusedExperts::NONE`] for a block that runs between an exchange's dispatch and
    /// combine, or no exchange.
    pub fn fused_experts(&self) -> exchange::FusedExperts {
        match &self.experts {
            Experts::FusedNvfp4 { .. } => exchange::FusedExperts::NVFP4,
            Experts::FusedMxfp4 { up_gate, limit, .. } => {
                // The shared experts' gated stack is `2 · shared · intermediate` rows.
                let intermediate = up_gate.codes.dims()[1] / 2;
                let shared = match &self.shared {
                    Some(Shared::Fused { up_gate, .. }) => {
                        up_gate.codes.dims()[0] / 2 / intermediate.max(1)
                    }
                    Some(Shared::Module(_)) | None => 0,
                };
                exchange::FusedExperts {
                    weights: ExpertWeights::Mxfp4,
                    shared: shared as u32,
                    limit: limit.unwrap_or(0.0).into(),
                }
            }
            Experts::Dense { .. } | Experts::Packed { .. } | Experts::Mxfp4 { .. } => {
                exchange::FusedExperts::NONE
            }
        }
    }

    /// Runs the group's step with no tokens of this rank's own.
    ///
    /// Every rank runs every expert layer of a step, because an exchange waits on its device
    /// for the same exchange of every rank. A rank with no batch of its own calls this: its
    /// experts serve the tokens routed to them and it sends none.
    ///
    /// # Errors
    ///
    /// An exchange whose tensors do not fit the block, or an operator's refusal.
    pub fn join_expert_step(&self, exchange: &ExpertExchange) -> Result<()> {
        self.check_exchange(exchange)?;

        // No rows of the exchange's own tensors: the element types and widths a launch reads.
        let none = |rows: &Tensor| rows.narrow(0, 0, 0);
        self.exchange_routes(
            &none(&exchange.states)?,
            &none(&exchange.ids)?,
            &none(&exchange.weights)?,
            None,
            exchange,
        )?;
        Ok(())
    }

    /// The block over this rank's own experts: every route's products, and their weighted sum.
    fn routed(&self, x: &Tensor, live_tokens: Option<&Tensor>) -> Result<Tensor> {
        let (ids, weights) = self.routed_ids(x, live_tokens)?;
        let maps =
            functional::moe_sort(&ids, &weights, self.count, self.local.clone(), live_tokens)?;

        let products = match &self.experts {
            Experts::Dense { up_gate, down } => {
                let gated = functional::gather_gemm(x, up_gate, &ids, &maps, self.activation)?;
                functional::route_gemm(&gated, down, &ids, &maps)?
            }

            Experts::Packed { up_gate, down } => {
                let gated =
                    functional::gather_gemm_nvfp4(x, up_gate, &maps, self.activation, self.top)?;
                functional::route_gemm_nvfp4(&gated, down, &maps, self.top)?
            }

            Experts::Mxfp4 {
                up_gate,
                down,
                limit,
            } => {
                let (codes, scales) =
                    functional::gather_gemm_mxfp4(x, up_gate, limit, &maps, self.top)?;
                functional::route_gemm_mxfp4(&codes, &scales, down, &maps, self.top)?
            }

            Experts::FusedNvfp4 { .. } | Experts::FusedMxfp4 { .. } => {
                return Err(Error::Unsupported(
                    "MoE: experts loaded for an expert group's fused exchange run through it"
                        .to_owned(),
                ));
            }
        };
        functional::moe_unpermute(&products, &maps, &weights, self.count, None)
    }

    /// The block over an expert group: every token is routed over all the experts, and its
    /// routes are computed by the ranks that hold them.
    fn exchanged(
        &self,
        x: &Tensor,
        live_tokens: Option<&Tensor>,
        exchange: Option<&ExpertExchange>,
    ) -> Result<Tensor> {
        let exchange = exchange.ok_or_else(|| {
            Error::shape(
                "MoE",
                "experts split over an expert group read the group's exchange, and the forward \
                 was lent none"
                    .to_owned(),
            )
        })?;
        self.check_exchange(exchange)?;

        if x.dtype() != exchange.states.dtype() {
            return Err(Error::shape(
                "MoE",
                format!(
                    "the exchange carries {} rows, the batch is {}",
                    ops::abi::name(exchange.states.dtype()),
                    ops::abi::name(x.dtype())
                ),
            ));
        }

        // The top-k writes slots, so no expert's own id leaves the rank and a peer sorts by the
        // order the stacks are laid out in.
        let (ids, weights) = self.routed_ids(x, live_tokens)?;
        self.exchange_routes(x, &ids, &weights, live_tokens, exchange)
    }

    /// Checks `exchange` against the block: its own tensors, then the widths and the group the
    /// block was loaded for.
    fn check_exchange(&self, exchange: &ExpertExchange) -> Result<()> {
        exchange.check()?;

        let (hidden, top) = (exchange.states.dims()[1], exchange.ids.dims()[1]);
        if hidden != self.router.k() as usize || top != self.top as usize {
            return Err(Error::shape(
                "MoE",
                format!(
                    "the exchange carries rows of {hidden} elements with {top} routes, the block \
                     routes rows of {} to {} experts",
                    self.router.k(),
                    self.top
                ),
            ));
        }

        // The ranks that hold experts hold equal windows of them.
        let holders = self.count / self.local_experts();
        if exchange.ranks - exchange.attention != holders {
            return Err(Error::shape(
                "MoE",
                format!(
                    "the exchange has {} ranks of which {} hold no experts, and {} experts in \
                     windows of {} take {holders}",
                    exchange.ranks,
                    exchange.attention,
                    self.count,
                    self.local_experts()
                ),
            ));
        }
        Ok(())
    }

    /// Sends the routes of `x`, runs this rank's experts over the routes it received, and takes
    /// back each of its tokens' sums: `[tokens, hidden]`.
    fn exchange_routes(
        &self,
        x: &Tensor,
        ids: &Tensor,
        weights: &Tensor,
        live_tokens: Option<&Tensor>,
        exchange: &ExpertExchange,
    ) -> Result<Tensor> {
        let stacks = match &self.experts {
            Experts::FusedMxfp4 {
                up_gate,
                down,
                limit,
            } if exchange.fused => {
                let stated = exchange.fused_experts;
                if stated.limit != limit.unwrap_or(0.0).into() {
                    return Err(Error::Unsupported(format!(
                        "MoE: the exchange's fused launch clamps the gated activation at {}, \
                         and the block's experts at {limit:?}",
                        stated.clamp()
                    )));
                }
                let shared = match &self.shared {
                    Some(Shared::Fused { up_gate, down }) => Some((up_gate, down)),
                    Some(Shared::Module(_)) | None => None,
                };
                return functional::expert_exchange_fused_mxfp4(
                    x,
                    ids,
                    weights,
                    live_tokens,
                    exchange,
                    (up_gate, down),
                    shared,
                    self.count,
                );
            }

            Experts::FusedNvfp4 {
                up_gate,
                down,
                up_gate_alpha,
                down_alpha,
                gated_scale,
            } if exchange.fused && exchange.fused_experts.weights == ExpertWeights::Nvfp4 => {
                return functional::expert_exchange_fused(
                    x,
                    ids,
                    weights,
                    live_tokens,
                    exchange,
                    up_gate,
                    down,
                    [up_gate_alpha, down_alpha, gated_scale],
                    self.count,
                );
            }

            stacks @ (Experts::Dense { .. } | Experts::Mxfp4 { .. }) if !exchange.fused => stacks,

            _ => {
                return Err(Error::Unsupported(format!(
                    "MoE: a fused exchange runs the experts it was laid out for ({} ones) and an \
                     exchange of two launches dense or MXFP4 ones, and this block's are loaded \
                     for {}",
                    exchange.fused_experts.weights,
                    if exchange.fused {
                        "the other composition"
                    } else {
                        "a fused exchange"
                    }
                )));
            }
        };
        let tokens = x.dims()[0];

        // 1. Dispatch: each route goes to the rank that holds its expert.
        functional::expert_dispatch(
            x,
            ids,
            weights,
            live_tokens,
            exchange,
            self.count,
            self.intermediate,
        )?;

        // A rank that holds no experts receives nothing: only its tokens' sums come back.
        if !exchange.source_only {
            // 2. The rows this rank received, to the rows the step holds per rank, grouped by
            //    its own experts. The ranks agree on the device how many of them are live.
            let received = exchange.ranks as usize * exchange.step_rows as usize;
            let states = exchange.states.narrow(0, 0, received)?;
            let received_ids = exchange.ids.narrow(0, 0, received)?;
            let received_weights = exchange.weights.narrow(0, 0, received)?;
            let maps = functional::moe_sort(
                &received_ids,
                &received_weights,
                self.count,
                self.local.clone(),
                Some(&exchange.live),
            )?;

            // 3. Both products over this rank's stacks, and each received row's sum left where
            //    the combine returns it from. The rows arrive in the activation type; MXFP4
            //    products encode them as their first product reads them.
            let products = match stacks {
                Experts::Dense { up_gate, down } => {
                    let gated = functional::gather_gemm(
                        &states,
                        up_gate,
                        &received_ids,
                        &maps,
                        self.activation,
                    )?;
                    functional::route_gemm(&gated, down, &received_ids, &maps)?
                }
                Experts::Mxfp4 {
                    up_gate,
                    down,
                    limit,
                } => {
                    let (codes, scales) =
                        functional::gather_gemm_mxfp4(&states, up_gate, limit, &maps, self.top)?;
                    functional::route_gemm_mxfp4(&codes, &scales, down, &maps, self.top)?
                }
                // Matched out before the dispatch: a fused exchange returned above.
                Experts::Packed { .. }
                | Experts::FusedNvfp4 { .. }
                | Experts::FusedMxfp4 { .. } => {
                    return Err(Error::Unsupported(
                        "MoE: packed NVFP4 experts run through a fused exchange".to_owned(),
                    ));
                }
            };
            functional::moe_unpermute(
                &products,
                &maps,
                &received_weights,
                self.count,
                Some(&exchange.combined.narrow(0, 0, received)?),
            )?;
        }

        // 4. Combine: the ranks' sums for this rank's tokens.
        functional::expert_combine(tokens, live_tokens, exchange, self.count, self.intermediate)
    }
}

/// The module names of a [`SharedExpert`] and the gated MLP it holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SharedExpertConfig {
    /// The shared expert's gated MLP.
    pub mlp: MlpConfig,
    /// The name of the module that holds the MLP.
    pub module: &'static str,
    /// The name of the `[1, hidden]` gate projection beside it, when the expert is gated; an
    /// ungated expert's output is added whole.
    pub gate: Option<&'static str>,
}

/// A shared expert beside a routed block: a gated MLP every token passes through, its output
/// added into the routed output, scaled by the sigmoid of a per-token gate when the checkpoint
/// states one (Qwen2-MoE, Qwen3-Next, Qwen3.5) and whole when it does not (DeepSeek):
///
/// ```text
/// routed ← routed + sigmoid(gate · x) · mlp(x)        gated
/// routed ← routed + mlp(x)                            ungated
/// ```
pub struct SharedExpert {
    mlp: Mlp,
    gate: Option<Linear>,
}

impl SharedExpert {
    /// The shared expert the modules under `vb` state, at the MLP's split. The gate, when there
    /// is one, is replicated: one logit per token on every rank.
    ///
    /// # Errors
    ///
    /// A module the loader refuses or a split the MLP refuses.
    pub fn load(config: &SharedExpertConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let mlp = Mlp::load(&config.mlp, vb.pp(config.module))?;
        let gate = config
            .gate
            .map(|gate| {
                Linear::load(
                    &LinearConfig {
                        inputs: config.mlp.hidden,
                        outputs: 1,
                        bias: false,
                        parallel: Parallel::None,
                    },
                    vb.pp(gate),
                )
            })
            .transpose()?;

        Ok(Self { mlp, gate })
    }

    /// Adds the shared expert's output of `x [..., hidden]`, gated when it has a gate, into
    /// `routed`, the routed block's output of the same rows, and returns the sum: in place for a
    /// gated expert, a new tensor for an ungated one.
    ///
    /// # Errors
    ///
    /// Rows a projection refuses, or an operator's refusal.
    pub fn forward(&self, x: &Tensor, routed: Tensor) -> Result<Tensor> {
        let update = self.mlp.forward(x)?;
        match &self.gate {
            Some(gate) => functional::sigmoid_gated_add_(routed, &update, &gate.forward(x)?),
            None => Ok((routed + update)?),
        }
    }
}
