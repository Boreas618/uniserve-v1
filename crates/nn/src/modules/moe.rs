//! The mixture-of-experts block of a transformer layer.
//!
//! ```text
//! scores  = router(x)                                  one score per expert, replicated
//! routes  = topk_softmax(scores)                       each token's `top` experts and weights
//! out[t]  = Σ_routes weight · down_e(act(gate_e(x[t])) · up_e(x[t]))
//! ```
//!
//! [`MoE`] is one module because its parts only make sense together: the routing decides which
//! expert's rows a token's routes go to, the sort turns that into the tiles the grouped products
//! read, and the combine sums the routes back with the FP32 weights the router produced
//! ([`crate::functional`]). The routing, the sort and the combine are the same for every
//! representation of the experts; only the two products differ.
//!
//! ## Loading
//!
//! A checkpoint stores a block's experts as one module per expert
//! (`<list>.<e>.<up|gate|down>.weight`) or as two stacked tensors, and either way the block
//! reads them as two stacks in the arrangement its products read: `[experts, 2 · intermediate,
//! hidden]` with each expert's up rows before its gate rows, and `[experts, hidden,
//! intermediate]`. A calibrated NVFP4 checkpoint's experts are read as packed stacks, which the
//! declaration the model's loader was given decides from the first held expert's weight.
//!
//! ## Where the experts sit
//!
//! The router selects over the checkpoint's expert ids, and each route is then written as the
//! **slot** its expert is stored in ([`MoEConfig::order`]), so which experts a token picks does
//! not depend on where they lie. A deployment that redistributes its experts states another
//! order, and the same checkpoint loads under it.
//!
//! ## Parallelism
//!
//! Over a tensor-parallel group ([`MoEConfig::split`]) a rank holds its block of every expert's
//! intermediate width, so its output is a partial sum that [`MoE::forward`] reduces over the
//! group. Over an expert group ([`MoEConfig::exchange`]) a rank holds a window of the slots and
//! the block runs through the group's exchange, whose rows execution lends
//! ([`ExpertExchange`]): the routes travel to the ranks that hold their experts, and each
//! token's sums come back.

use std::ops::Range;

use ops::moe::routing::{MAX_EXPERTS, MAX_TOP};
use uniserve_tensor::{DType, Device, Shape, Tensor};

use crate::capability::ExpertExchange;
use crate::collective::{self, Split};
use crate::error::{Error, Result};
use crate::functional::{
    self, Activation, DOWN_VALUE_LAYOUT, GATED_VALUE_LAYOUT, WEIGHT_SCALE_LAYOUT, WeightOrder,
};
use crate::modules::linear::Linear;
use crate::quantization::{Nvfp4Names, Nvfp4Stack, Scheme, ValueLayout};
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
    /// The router's module name.
    pub router: &'static str,
    /// The name of the module list that holds the experts: expert `e`'s projections are
    /// `<list>.<e>.<up|gate|down>`.
    pub list: &'static str,
    /// An expert's up projection's module name.
    pub up: &'static str,
    /// An expert's gate projection's module name.
    pub gate: &'static str,
    /// An expert's down projection's module name.
    pub down: &'static str,
    /// The stacked tensors a checkpoint of this family may hold its experts in instead,
    /// `[gate_up, down]` under the module list (Hugging Face transformers' fused experts):
    /// `gate_up` is `[experts, 2 · intermediate, hidden]` with each expert's gate rows before
    /// its up rows, and `down` is `[experts, hidden, intermediate]`. They are read when the
    /// checkpoint holds them.
    pub stacked: Option<[&'static str; 2]>,
    /// The expert each slot holds: a permutation of the experts, in which the stacks are laid
    /// out and by which an expert group's ranks divide them. The experts' own order when a
    /// deployment has not redistributed them.
    pub order: Vec<u32>,
    /// The tensor-parallel split of each expert's intermediate width.
    pub split: Split,
    /// The expert group's split of the slots: this rank holds one contiguous window of them.
    pub exchange: Split,
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
    /// Calibrated NVFP4 stacks in the arrangement an expert group's fused exchange reads (each
    /// gate block before its up block), with the three `[local]` FP32 tables its kernel
    /// multiplies in.
    Fused {
        up_gate: Nvfp4Stack,
        down: Nvfp4Stack,
        /// Dequantizes the first product: the weight's tensor scale times its rows' scale.
        up_gate_alpha: Tensor,
        /// Dequantizes the second product.
        down_alpha: Tensor,
        /// The global scale the gated rows are encoded with.
        gated_scale: Tensor,
    },
}

/// A mixture-of-experts block.
pub struct MoE {
    router: Linear,
    experts: Experts,
    /// `[experts]` U32: the slot of each expert, by which the top-k writes each route's weight
    /// row.
    map: Tensor,
    /// A router's per-expert scale, multiplied into the route weights. No served checkpoint
    /// states one, so [`MoE::load`] reads none.
    scale: Option<Tensor>,
    /// Experts the router selects over.
    count: u32,
    top: u32,
    /// This rank's block of an expert's gated width.
    intermediate: u32,
    activation: Activation,
    renormalize: bool,
    /// The slots this rank holds; every slot when no expert group divides them.
    local: Range<u32>,
    /// Ranks of the tensor-parallel group the experts' gated width is split over: above one,
    /// the output is a partial sum the forward reduces.
    reduce: u32,
}

impl MoE {
    /// The block the module at `vb` states, at this rank's share of it.
    ///
    /// The router is replicated: every rank routes a token the same way. Over an expert group
    /// the rank holds its window of the slots; over a tensor-parallel group, its rows of every
    /// gate and up projection and its columns of every down projection. A calibrated NVFP4
    /// checkpoint's experts are read packed: on one rank in the arrangement the routed products
    /// read, and over an expert group in the fused exchange's, with the per-expert scalars its
    /// kernel multiplies in.
    ///
    /// # Errors
    ///
    /// When the routing is outside what the router's kernel holds, the order is no permutation
    /// of the experts, a split does not divide the experts or their width, packed experts are
    /// split by a tensor-parallel group, or a tensor is missing or of another extent.
    pub fn load(config: &MoEConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let refuse = |problem: String| Error::weight(vb.path(), problem);
        let (experts, top) = (config.experts, config.top);
        let (split, exchange) = (config.split, config.exchange);

        if top == 0 || top > MAX_TOP || top > experts {
            return Err(refuse(format!(
                "a top of {top} is outside the routers' 1..={MAX_TOP} of {experts} experts"
            )));
        }
        if experts == 0 || experts > MAX_EXPERTS {
            return Err(refuse(format!(
                "{experts} experts are outside a router's 1..={MAX_EXPERTS}"
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
        if exchange.size == 0
            || exchange.rank >= exchange.size
            || !experts.is_multiple_of(exchange.size)
        {
            return Err(refuse(format!(
                "rank {} of an expert group of {} does not hold a share of {experts} experts",
                exchange.rank, exchange.size
            )));
        }
        if split.size == 0
            || split.rank >= split.size
            || !config.intermediate.is_multiple_of(split.size)
        {
            return Err(refuse(format!(
                "rank {} of a tensor group of {} does not hold a share of an expert's {} \
                 intermediate columns",
                split.rank, split.size, config.intermediate
            )));
        }

        let held = experts / exchange.size;
        let local = exchange.rank * held..(exchange.rank + 1) * held;
        let window = &config.order[local.start as usize..local.end as usize];
        let intermediate = config.intermediate / split.size;

        let router = Linear::new(
            vb.pp(config.router)
                .get((experts as usize, config.hidden as usize), "weight")?,
            None,
            WeightOrder::NxK,
        )?;

        // The slot of each expert: the inverse of the order, as the top-k reads it.
        let slots = Tensor::arange(0_u32, experts, &Device::Cpu)?;
        let order = Tensor::new(config.order.as_slice(), &Device::Cpu)?;
        let map = vb.constant_from(
            "expert_slots",
            &slots.zeros_like()?.scatter(&order, &slots, 0)?,
        )?;

        let list = vb.pp(config.list);
        let packed = list.pp(window[0]).pp(config.up).scheme("weight") == Scheme::Nvfp4;
        let stacks = if packed {
            if split.size > 1 {
                return Err(refuse(format!(
                    "packed experts are whole on a rank, and a tensor group of {} splits their \
                     width",
                    split.size
                )));
            }
            Self::load_packed(config, &vb, window, exchange.size > 1)?
        } else {
            let (up_gate, down) = Self::load_dense(config, &vb, window)?;
            Experts::Dense { up_gate, down }
        };

        Ok(Self {
            router,
            experts: stacks,
            map,
            scale: None,
            count: experts,
            top,
            intermediate,
            activation: config.activation,
            renormalize: config.renormalize,
            local,
            reduce: split.size,
        })
    }

    /// The dense stacks of the experts `window` names, this rank's block of each: from the
    /// checkpoint's stacked tensors when it holds them, and from one module per expert
    /// otherwise.
    fn load_dense(
        config: &MoEConfig,
        vb: &VarBuilder<'_, '_>,
        window: &[u32],
    ) -> Result<(Tensor, Tensor)> {
        let split = config.split;
        let (hidden, intermediate) = (u64::from(config.hidden), u64::from(config.intermediate));
        let local_intermediate = intermediate / u64::from(split.size);
        let first = u64::from(split.rank) * local_intermediate;
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

        let stacked = config.stacked.filter(|[gate_up, _]| list.contains(gate_up));
        let (up_gate_name, up_gate, down_name, down) = match stacked {
            Some([gate_up, down]) => {
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

            None => {
                let weight = |expert: u32, module: &str| list.pp(expert).pp(module).name("weight");
                let up_gate_parts = window
                    .iter()
                    .flat_map(|&expert| [weight(expert, config.up), weight(expert, config.gate)])
                    .collect();
                let down_parts = window
                    .iter()
                    .map(|&expert| weight(expert, config.down))
                    .collect();
                let block = |block: PartBlock| (split.size > 1).then_some(block);

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
    /// routed products' otherwise.
    fn load_packed(
        config: &MoEConfig,
        vb: &VarBuilder<'_, '_>,
        window: &[u32],
        exchanged: bool,
    ) -> Result<Experts> {
        let loader = vb.loader();
        let list = vb.pp(config.list);
        let (gated_layout, down_layout) = if exchanged {
            (ValueLayout::GateFirst, ValueLayout::Plain)
        } else {
            (GATED_VALUE_LAYOUT, DOWN_VALUE_LAYOUT)
        };

        // The checkpoint's block scales are linear, and the published tables are built from
        // them once here.
        let (up_gate, up_scales) = loader.nvfp4_stack(
            &vb.name("experts.up_gate"),
            list.path(),
            &[config.up, config.gate],
            window,
            config.experts,
            Nvfp4Names::vllm(),
            (gated_layout, WEIGHT_SCALE_LAYOUT),
            true,
        )?;
        let (down, down_scales) = loader.nvfp4_stack(
            &vb.name("experts.down"),
            list.path(),
            &[config.down],
            window,
            config.experts,
            Nvfp4Names::vllm(),
            (down_layout, WEIGHT_SCALE_LAYOUT),
            true,
        )?;
        Self::check_packed(config, vb, window.len(), &up_gate, &down)?;

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
        Ok(Experts::Fused {
            up_gate_alpha: table("up_gate_alpha", up_alpha)?,
            down_alpha: table("down_alpha", down_alpha)?,
            gated_scale: table("gated_scale", gated)?,
            up_gate,
            down,
        })
    }

    /// Checks that packed stacks of `held` experts are the extents and types the block's
    /// config states and its products read.
    fn check_packed(
        config: &MoEConfig,
        vb: &VarBuilder<'_, '_>,
        held: usize,
        up_gate: &Nvfp4Stack,
        down: &Nvfp4Stack,
    ) -> Result<()> {
        let refuse = |problem: String| Error::weight(vb.path(), problem);
        let (hidden, intermediate) = (config.hidden as usize, config.intermediate as usize);

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
        functional::topk_softmax(
            &self.router.forward(x)?,
            self.top,
            self.renormalize,
            self.scale.as_ref(),
            Some(&self.map),
        )
    }

    /// `Σ_routes weight · expert(x)` over `x [tokens, hidden]`, summed over the tensor-parallel
    /// group: `[tokens, hidden]` in `x`'s element type.
    ///
    /// `live_tokens` is the batch's count of token rows it fills
    /// ([`TokenBatch::live_tokens`](crate::TokenBatch::live_tokens)), when execution states it:
    /// the rows of a captured bucket past the batch route nowhere and their output is zeros.
    /// `exchange` is the expert group's exchange, lent when the block's experts are split over
    /// one; the block then runs through it.
    ///
    /// # Errors
    ///
    /// When `x` does not fit the block, an operator refuses, experts split over an expert group
    /// run without its exchange, or experts split over a tensor-parallel group run in a forward
    /// that has no group.
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

        let out = if self.local_experts() != self.count || exchange.is_some() {
            self.exchanged(x, live_tokens, exchange)?
        } else {
            self.routed(x, live_tokens)?
        };

        if self.reduce > 1 {
            // One rank's block of every expert's width: the group sums the ranks'.
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
        Ok(out)
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
        let (ids, weights) = self.route(x)?;
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

            Experts::Fused { .. } => {
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
        let (ids, weights) = self.route(x)?;
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
        let (up_gate, down) = match &self.experts {
            Experts::Fused {
                up_gate,
                down,
                up_gate_alpha,
                down_alpha,
                gated_scale,
            } if exchange.fused => {
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

            Experts::Dense { up_gate, down } if !exchange.fused => (up_gate, down),

            _ => {
                return Err(Error::Unsupported(
                    "MoE: a fused exchange runs packed experts and an exchange of two launches \
                     dense ones, and this block holds the other"
                        .to_owned(),
                ));
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
            //    the combine returns it from.
            let gated =
                functional::gather_gemm(&states, up_gate, &received_ids, &maps, self.activation)?;
            let products = functional::route_gemm(&gated, down, &received_ids, &maps)?;
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
