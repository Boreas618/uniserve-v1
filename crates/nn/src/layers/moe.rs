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
//! program selects the transport. A `Moe` layer built for a placement that does not hold every
//! expert refuses the forward rather than computing a routing the weights do not have.

use ops::moe::combine::{self, MoeUnpermute, UnpermuteArgs};
use ops::moe::exchange::{self, ExchangeArgs, ExpertExchange, Mode};
use ops::moe::grouped::{
    Activation, GatherGemm, GatherGemmArgs, GatherGemmDescriptor, RouteGemm, RouteGemmArgs,
    RouteGemmDescriptor,
};
use ops::moe::routing::{
    ExpertGather, GatherArgs, GatherDescriptor, TopkArgs, TopkDescriptor, TopkSoftmax,
};
use ops::moe::sort::{self, MoeSort, SortArgs};
use ops::{DType, DevicePtr, Op, Shape, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::Linear;
use crate::weights::Weight;

/// The tile geometry the grouped products' packages are compiled for.
///
/// Like the row tile of a normalization this is program content (specs/worker.md 5.2, 5.4); the
/// values here are the packages' default tactic for the Qwen3-MoE shapes, declared once in
/// [`crate::layers::launch`] rather than at every call.
const MMA_TILE: (u32, u32) = (128, 128);
/// The cluster shape the grouped products' packages are compiled for.
const MMA_CLUSTER: (u32, u32) = (1, 1);

/// A mixture-of-experts block over a colocated expert group.
pub struct Moe {
    router: Linear,
    up_gate: Weight,
    down: Weight,
    map: Option<Weight>,
    experts: u32,
    top: u32,
    hidden: u32,
    intermediate: u32,
    activation: Activation,
    renormalize: bool,
    scale: Option<Weight>,
    tile: u32,
    /// Experts of the vocabulary this rank holds; every expert in the colocated case.
    local: u32,
    /// The first expert of this rank's window.
    local_offset: u32,
}

impl Moe {
    /// A block whose router is `router` and whose experts are `up_gate` and `down`.
    ///
    /// `up_gate` is `[experts, 2 · intermediate, hidden]` with the up projection first, `down` is
    /// `[experts, hidden, intermediate]`, and `router` has one output per expert. `map` is the
    /// expert-parallel map when the placement reorders experts; with every expert colocated it is
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
            up_gate,
            down,
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
        if up_gate.shape().rank() != 3 || up_gate.shape().dim(0) != u64::from(local) {
            return Err(Error::shape(
                "Moe",
                format!(
                    "the up/gate weight is {:?}, [{local}, 2·intermediate, hidden] expected",
                    up_gate.shape().dims()
                ),
            ));
        }
        let intermediate = up_gate.shape().dim(1) / 2;
        if down.shape().rank() != 3
            || down.shape().dim(0) != u64::from(local)
            || down.shape().dim(1) != up_gate.shape().dim(2)
            || down.shape().dim(2) != intermediate
        {
            return Err(Error::shape(
                "Moe",
                format!(
                    "the down weight is {:?}, [{local}, {}, {intermediate}] expected",
                    down.shape().dims(),
                    up_gate.shape().dim(2)
                ),
            ));
        }
        let hidden = u32::try_from(up_gate.shape().dim(2))
            .map_err(|_| Error::shape("Moe", "a hidden width above u32::MAX".to_owned()))?;
        let intermediate = u32::try_from(intermediate)
            .map_err(|_| Error::shape("Moe", "an expert width above u32::MAX".to_owned()))?;
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
            up_gate,
            down,
            map,
            experts,
            top,
            hidden,
            intermediate,
            activation,
            renormalize,
            scale,
            tile: 128,
            local,
            local_offset,
        })
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
            up_gate,
            down,
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

    /// `out = Σ_routes weight · expert(hidden)`.
    ///
    /// `hidden` and `out` are `[tokens, hidden]`.
    pub fn forward(&self, f: &Forward, hidden: TensorView, out: TensorView) -> Result<()> {
        if self.local != self.experts || f.exchange.is_some() {
            return Err(Error::Unsupported(
                "Moe: this block's experts are split across ranks; call \
                 forward_expert_parallel so the routes travel over the exchange"
                    .to_owned(),
            ));
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
        let tokens = hidden.rows();
        let top = u64::from(self.top);
        let experts = u64::from(self.experts);
        let intermediate = u64::from(self.intermediate);
        let arena = f.arena();

        // 1. The replicated router, then its top-k.
        let scores = arena.tensor(hidden.dtype, Shape::new(&[tokens, experts]))?;
        let ids = arena.tensor(DType::I32, Shape::new(&[tokens, top]))?;
        let weights = arena.tensor(DType::F32, Shape::new(&[tokens, top]))?;
        self.router.forward(f, hidden, scores)?;
        let routing = TopkDescriptor {
            scores: scores.dtype,
            experts: self.experts,
            top: self.top,
            renormalize: self.renormalize,
            scale: self.scale.as_ref().map(Weight::dtype),
        };
        let args = TopkArgs {
            scores,
            scale: self.scale.as_ref().map(Weight::view),
            ids,
            weights,
        };
        let kernel = f.kernels().op::<TopkSoftmax>(&routing)?;
        TopkSoftmax::launch(kernel, &routing, args, f.stream)?;

        // 2. Logical ids become weight rows.
        let local_ids = match &self.map {
            Some(map) => {
                let mapped = arena.tensor(DType::I32, Shape::new(&[tokens, top]))?;
                let descriptor = GatherDescriptor {
                    experts: self.experts,
                    top: self.top,
                };
                let args = GatherArgs {
                    ids,
                    map: map.view(),
                    out: mapped,
                };
                let kernel = f.kernels().op::<ExpertGather>(&descriptor)?;
                ExpertGather::launch(kernel, &descriptor, args, f.stream)?;
                mapped
            }
            None => ids,
        };

        // 3. Group the routes by expert into the tiles the products read.
        let tiles = ops::moe::max_tiles(tokens, top, experts, u64::from(self.tile));
        let rows = ops::moe::max_permuted_rows(tokens, top, experts, u64::from(self.tile));
        let tile_to_expert = arena.tensor(DType::I32, Shape::new(&[tiles]))?;
        let tile_to_limit = arena.tensor(DType::I32, Shape::new(&[tiles]))?;
        let expanded_to_permuted = arena.tensor(DType::I32, Shape::new(&[tokens, top]))?;
        let permuted_to_expanded = arena.tensor(DType::I32, Shape::new(&[rows]))?;
        let total_padded_tokens = arena.tensor(DType::I32, Shape::new(&[1]))?;
        let num_non_exiting_tiles = arena.tensor(DType::I32, Shape::new(&[1]))?;
        let sort = sort::Descriptor {
            experts: self.experts,
            top: self.top,
            local_experts: self.experts,
            tile_tokens: self.tile,
            scales: DType::F32,
        };
        let args = SortArgs {
            ids: local_ids,
            scales: weights,
            tile_to_expert,
            tile_to_limit,
            expanded_to_permuted,
            permuted_to_expanded,
            total_padded_tokens,
            num_non_exiting_tiles,
            local_offset: 0,
        };
        let kernel = f.kernels().op::<MoeSort>(&sort)?;
        MoeSort::launch(kernel, &sort, args, f.stream)?;

        // 4. FC1: the gated product of each permuted row, then FC2.
        let gated = arena.tensor(hidden.dtype, Shape::new(&[rows, intermediate]))?;
        let first = GatherGemmDescriptor {
            hidden: self.hidden,
            intermediate: self.intermediate,
            experts: self.experts,
            top: self.top,
            activation: self.activation,
            dtype: hidden.dtype,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        };
        let args = GatherGemmArgs {
            hidden,
            weights: self.up_gate.view(),
            ids: local_ids,
            tile_to_expert,
            tile_to_limit,
            permuted_to_expanded,
            num_non_exiting_tiles,
            out: gated,
        };
        let kernel = f.kernels().op::<GatherGemm>(&first)?;
        GatherGemm::launch(kernel, &first, args, f.stream)?;

        let products = arena.tensor(hidden.dtype, Shape::new(&[tokens * top, hidden_width]))?;
        let second = RouteGemmDescriptor {
            hidden: self.hidden,
            intermediate: self.intermediate,
            experts: self.experts,
            top: self.top,
            dtype: hidden.dtype,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        };
        let args = RouteGemmArgs {
            rows: gated,
            weights: self.down.view(),
            ids: local_ids,
            tile_to_expert,
            tile_to_limit,
            permuted_to_expanded,
            num_non_exiting_tiles,
            out: products,
        };
        let kernel = f.kernels().op::<RouteGemm>(&second)?;
        RouteGemm::launch(kernel, &second, args, f.stream)?;

        // 5. Sum the routes back in FP32 and round once to the hidden dtype.
        let combine = combine::Descriptor {
            hidden: self.hidden,
            top: self.top,
            experts: self.experts,
            input: products.dtype,
            scales: DType::F32,
            output: out.dtype,
        };
        let args = UnpermuteArgs {
            routes: products,
            expanded_to_permuted,
            scales: weights,
            out,
        };
        let kernel = f.kernels().op::<MoeUnpermute>(&combine)?;
        MoeUnpermute::launch(kernel, &combine, args, f.stream).map_err(Error::from)
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
        // The group's geometry has to agree with the placement the block was built for.
        let expected_ranks = u64::from(self.experts / self.local.max(1));
        if ranks != expected_ranks {
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
            ("counts", exchange.counts, DType::I32),
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
        let local = u64::from(self.local);
        let intermediate = u64::from(self.intermediate);
        let arena = f.arena();
        let empty = TensorView::new(DevicePtr::new(0), DType::U8, Shape::new(&[0]));

        // 1. The replicated router and its top-k: every rank routes whole tokens the same way.
        let scores = arena.tensor(hidden.dtype, Shape::new(&[tokens, experts]))?;
        let ids = arena.tensor(DType::I32, Shape::new(&[tokens, top]))?;
        let weights = arena.tensor(DType::F32, Shape::new(&[tokens, top]))?;
        self.router.forward(f, hidden, scores)?;
        let routing = TopkDescriptor {
            scores: scores.dtype,
            experts: self.experts,
            top: self.top,
            renormalize: self.renormalize,
            scale: self.scale.as_ref().map(Weight::dtype),
        };
        let args = TopkArgs {
            scores,
            scale: self.scale.as_ref().map(Weight::view),
            ids,
            weights,
        };
        let kernel = f.kernels().op::<TopkSoftmax>(&routing)?;
        TopkSoftmax::launch(kernel, &routing, args, f.stream)?;

        // 2. Logical ids become weight rows before anything leaves the rank, so a peer sorts by the
        //    same physical order the weights are stacked in.
        let sent = match &self.map {
            Some(map) => {
                let mapped = arena.tensor(DType::I32, Shape::new(&[tokens, top]))?;
                let descriptor = GatherDescriptor {
                    experts: self.experts,
                    top: self.top,
                };
                let args = GatherArgs {
                    ids,
                    map: map.view(),
                    out: mapped,
                };
                let kernel = f.kernels().op::<ExpertGather>(&descriptor)?;
                ExpertGather::launch(kernel, &descriptor, args, f.stream)?;
                mapped
            }
            None => ids,
        };

        // 3. Dispatch: each token's payload goes to the ranks that hold its experts.
        let exchange_descriptor = |mode: Mode| exchange::Descriptor {
            mode,
            hidden: self.hidden,
            top: self.top,
            experts: self.experts,
            max_tokens: exchange.capacity,
            ranks: exchange.ranks,
            dtype: hidden.dtype,
        };
        let dispatch = exchange_descriptor(Mode::Dispatch);
        let args = ExchangeArgs {
            transport: exchange.transport,
            tokens: u32::try_from(tokens)
                .map_err(|_| Error::shape("Moe", format!("{tokens} tokens exceed u32")))?,
            state: hidden,
            ids: sent,
            weights,
            out: empty,
            counts: exchange.counts,
            ranks: exchange.ranks,
        };
        let kernel = f.kernels().op::<ExpertExchange>(&dispatch)?;
        ExpertExchange::launch(kernel, &dispatch, args, f.stream)?;

        // 4. The rows this rank was sent, grouped by its own experts.
        let tiles = ops::moe::max_tiles(received, top, local, u64::from(self.tile));
        let rows = ops::moe::max_permuted_rows(received, top, local, u64::from(self.tile));
        let tile_to_expert = arena.tensor(DType::I32, Shape::new(&[tiles]))?;
        let tile_to_limit = arena.tensor(DType::I32, Shape::new(&[tiles]))?;
        let expanded_to_permuted = arena.tensor(DType::I32, Shape::new(&[received, top]))?;
        let permuted_to_expanded = arena.tensor(DType::I32, Shape::new(&[rows]))?;
        let total_padded_tokens = arena.tensor(DType::I32, Shape::new(&[1]))?;
        let num_non_exiting_tiles = arena.tensor(DType::I32, Shape::new(&[1]))?;
        let sort = sort::Descriptor {
            experts: self.experts,
            top: self.top,
            local_experts: self.local,
            tile_tokens: self.tile,
            scales: DType::F32,
        };
        let args = SortArgs {
            ids: exchange.ids,
            scales: exchange.weights,
            tile_to_expert,
            tile_to_limit,
            expanded_to_permuted,
            permuted_to_expanded,
            total_padded_tokens,
            num_non_exiting_tiles,
            local_offset: self.local_offset,
        };
        let kernel = f.kernels().op::<MoeSort>(&sort)?;
        MoeSort::launch(kernel, &sort, args, f.stream)?;

        // 5. FC1 and FC2 over the received rows, with this rank's experts.
        let gated = arena.tensor(hidden.dtype, Shape::new(&[rows, intermediate]))?;
        let first = GatherGemmDescriptor {
            hidden: self.hidden,
            intermediate: self.intermediate,
            experts: self.experts,
            top: self.top,
            activation: self.activation,
            dtype: hidden.dtype,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        };
        let args = GatherGemmArgs {
            hidden: exchange.states,
            weights: self.up_gate.view(),
            ids: exchange.ids,
            tile_to_expert,
            tile_to_limit,
            permuted_to_expanded,
            num_non_exiting_tiles,
            out: gated,
        };
        let kernel = f.kernels().op::<GatherGemm>(&first)?;
        GatherGemm::launch(kernel, &first, args, f.stream)?;

        let products = arena.tensor(hidden.dtype, Shape::new(&[received * top, hidden_width]))?;
        let second = RouteGemmDescriptor {
            hidden: self.hidden,
            intermediate: self.intermediate,
            experts: self.experts,
            top: self.top,
            dtype: hidden.dtype,
            tile: MMA_TILE,
            cluster: MMA_CLUSTER,
        };
        let args = RouteGemmArgs {
            rows: gated,
            weights: self.down.view(),
            ids: exchange.ids,
            tile_to_expert,
            tile_to_limit,
            permuted_to_expanded,
            num_non_exiting_tiles,
            out: products,
        };
        let kernel = f.kernels().op::<RouteGemm>(&second)?;
        RouteGemm::launch(kernel, &second, args, f.stream)?;

        // 6. Each received row's routes summed with their FP32 weights, in receive order.
        let combined = arena.tensor(hidden.dtype, Shape::new(&[received, hidden_width]))?;
        let combine_descriptor = combine::Descriptor {
            hidden: self.hidden,
            top: self.top,
            experts: self.experts,
            input: products.dtype,
            scales: DType::F32,
            output: hidden.dtype,
        };
        let args = UnpermuteArgs {
            routes: products,
            expanded_to_permuted,
            scales: exchange.weights,
            out: combined,
        };
        let kernel = f.kernels().op::<MoeUnpermute>(&combine_descriptor)?;
        MoeUnpermute::launch(kernel, &combine_descriptor, args, f.stream)?;

        // 7. Combine: the ranks' partial sums for this rank's tokens, rounded once to `out`.
        let combine_exchange = exchange_descriptor(Mode::Combine);
        let args = ExchangeArgs {
            transport: exchange.transport,
            tokens: u32::try_from(tokens)
                .map_err(|_| Error::shape("Moe", format!("{tokens} tokens exceed u32")))?,
            state: combined,
            ids: empty,
            weights: empty,
            out,
            counts: exchange.counts,
            ranks: exchange.ranks,
        };
        let kernel = f.kernels().op::<ExpertExchange>(&combine_exchange)?;
        ExpertExchange::launch(kernel, &combine_exchange, args, f.stream).map_err(Error::from)
    }

    /// The bound the combined rows' implementations may differ by.
    pub fn tolerance(&self) -> ops::Tolerance {
        MoeUnpermute::tolerance(&combine::Descriptor {
            hidden: self.hidden,
            top: self.top,
            experts: self.experts,
            input: self.up_gate.dtype(),
            scales: DType::F32,
            output: self.up_gate.dtype(),
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
        // The stacks hold the window, not the vocabulary.
        assert!(expert_parallel_block(128, 32, 96).is_err());
    }
}
