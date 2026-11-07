use nn::layers::{
    Linear, Rotary, Sections, SegmentAttention, SegmentTables, UlyssesExchange, VsaTiles,
};
use nn::weights::{Weight, WeightLoader, WeightRequest};
use nn::{Forward, Shape, TensorView};
use ops::gemm::WeightOrder;

use crate::attention::{Factors, HeadProjection, HeadsToRows, projected_rows};
use crate::config::TransformerConfig;
use crate::partition::{Partition, Projection};
use crate::{Error, Result};

/// One layer's sparse attention: four projections, the tile preparation, and both attentions.
pub struct Sparse {
    /// The query, key and value projections, `[inner, hidden]` each.
    pub(crate) query: HeadProjection,
    pub(crate) key: HeadProjection,
    pub(crate) value: HeadProjection,
    /// The gate the compression is weighted by, `[inner, hidden]`: a fourth projection of the same
    /// fused matrix as the three above (`attention.py:139`).
    pub(crate) gate: HeadProjection,
    /// The output projection back to the residual's width, `[hidden, inner]`.
    pub output: Linear,
    /// Per-head normalization gains of the query and the key.
    pub query_gain: Weight,
    pub key_gain: Weight,
    /// The partial rotation of the head.
    pub rotary: Rotary,
    /// The tile preparation: normalization, rotation, padding and pooling.
    pub tiles: VsaTiles,
    /// The fine attention over the selected tiles of the rows, in the checkpoint's dtype.
    pub fine: SegmentAttention,
    /// Heads of the whole model, and the heads this rank holds.
    pub heads: u32,
    pub local_heads: u32,
    /// Elements of one head.
    pub head_dim: u32,
    /// Rows of one tile.
    pub tile_rows: u32,
    /// The Ulysses exchange, absent when one rank holds the sequence.
    exchange: Option<UlyssesExchange>,
    /// How this rank reaches its heads.
    projection: Projection,
    /// This rank's place in the group: which block of the gathered sequence holds its rows.
    rank: u64,
}

impl Sparse {
    /// Every checkpoint tensor one layer's sparse attention reads, as complete names.
    ///
    /// The gate compression's projection is the fourth of the fused matrix the query, key and value
    /// come from (`attention.py:139`), so it is one more weight than the dense branch reads.
    pub fn keys(index: u32) -> Vec<String> {
        let prefix = format!("transformer_blocks.{index}.attn");
        [
            "norm_q.weight",
            "norm_k.weight",
            "to_q.weight",
            "to_k.weight",
            "to_v.weight",
            "to_gate_compress.weight",
            "to_out.0.weight",
        ]
        .iter()
        .map(|suffix| format!("{prefix}.{suffix}"))
        .collect()
    }

    /// Read layer `index`'s sparse attention of a DiT partition.
    ///
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        index: u32,
        config: &TransformerConfig,
        rows: u32,
        rotated_axes: &[u32],
        tile_rows: u32,
        tile_bound: u32,
        partition: &Partition,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        let prefix = format!("transformer_blocks.{index}.attn");
        let hidden = u64::from(config.hidden_size);
        let head = u64::from(config.head_dim);
        let inner = u64::from(config.num_attention_heads) * head;
        let rotated: u32 = rotated_axes.iter().sum();
        if rotated == 0 || rotated > config.head_dim || !rotated.is_multiple_of(2) {
            return Err(Error::config(format!(
                "{rotated} rotated channels of a {} element head",
                config.head_dim
            )));
        }
        let query_gain = gain(loader, &format!("{prefix}.norm_q.weight"), head)?;
        let key_gain = gain(loader, &format!("{prefix}.norm_k.weight"), head)?;
        let frequencies = frequencies(loader, config, rotated, rotated_axes.len() as u32)?;
        // The rotation reads *this rank's* heads: under Ulysses the sequence is split across ranks
        // and each holds `heads / degree` of them (`Partition::denoiser_heads`), so a descriptor that
        // named the global count would refuse its own query view at any degree above one.
        let local_heads = partition.denoiser_heads(config.num_attention_heads);
        let rotary = Rotary::new(
            frequencies,
            config.head_dim,
            local_heads,
            local_heads,
            config.qk_norm_eps,
            crate::SPLIT_HALF,
            1.0,
        )?
        .partial(rotated)?
        // Contiguous sections, not the interleaved cycle: the reference's factors are one block
        // per coordinate axis (axis 0 first), and the cycle belongs to the text encoder's M-RoPE.
        .sectioned(Sections::new(rotated_axes)?)?;
        let exchange = match partition.ulysses_degree {
            1 => None,
            degree => Some(UlyssesExchange::new(degree as u64)?),
        };
        // The compression reduces over every key tile of the layout, so the layer states the
        // deployment's own tile bound once (`Layouts::tiles_bound`) and the score matrix is sized by
        // it rather than guessed from a layout at launch.
        let tiles = VsaTiles::new(tile_rows, local_heads, config.head_dim, config.qk_norm_eps)?
            .with_norm()
            .with_rotation(rotated)?
            .with_tile_bound(tile_bound)?;
        // The fine attention writes the rows the projections produce, so its output type is the
        // projection's own: the stack's, which is the checkpoint's representation at `auto`. Writing
        // BF16 here was a constant that agreed with a BF16 deployment and disagreed with every other
        // one - an FP16 stack would have had its attended rows narrowed to BF16 by a layer that never
        // asked what the stack was.
        let head_projection = |loader: &mut WeightLoader<'_>, name: &str| {
            HeadProjection::load(
                loader,
                &format!("{prefix}.{name}"),
                inner,
                hidden,
                partition,
            )
        };
        let query = head_projection(loader, "to_q.weight")?;
        let stack = query.dtype();
        // One line, once per process: the type the walk read for the weight that decides the fine
        // attention's output, named with the weight it came from. A launch key states this type, and
        // a key that says `f32` is either a deployment converted to FP32 or a lookup that answered
        // with something other than the checkpoint; the second is invisible without this print,
        // because both produce a valid key.
        {
            static ONCE: std::sync::Once = std::sync::Once::new();
            let name = format!("{prefix}.to_q.weight");
            ONCE.call_once(|| {
                eprintln!(
                    "# attention_sparse: {name} is {stack:?}, so the fine attention writes {stack:?}"
                );
            });
        }
        let fine = SegmentAttention::new(
            u64::from(tile_rows),
            u64::from(local_heads),
            head,
            u64::from(rows),
            stack,
        )?;
        Ok(Self {
            query,
            key: head_projection(loader, "to_k.weight")?,
            value: head_projection(loader, "to_v.weight")?,
            gate: head_projection(loader, "to_gate_compress.weight")?,
            output: projection(loader, &format!("{prefix}.to_out.0.weight"), hidden, inner)?,
            query_gain,
            key_gain,
            rotary,
            tiles,
            fine,
            heads: config.num_attention_heads,
            local_heads,
            head_dim: config.head_dim,
            tile_rows,
            exchange,
            projection: partition.projection,
            rank: partition.sequence_rank as u64,
        })
    }

    /// The rotary factor tables of `positions` `[3, rows]`, written into the two views.
    pub fn factors(
        &self,
        f: &Forward,
        positions: TensorView,
        cosine: TensorView,
        sine: TensorView,
    ) -> Result<()> {
        self.rotary
            .factors(f, positions, cosine, sine)
            .map_err(Error::from)
    }

    /// Attend `normalized`, this rank's `[rows, hidden]`, and write this rank's `[rows, hidden]`.
    ///
    /// `sizes` is `[tiles]` int32, the valid rows of each tile; `segments` is `[tiles]` int32, the
    /// segment each tile belongs to, and `keep` is the tile-count-long table of each segment's kept
    /// key tiles, as the reference builds it. All three cover the whole sequence, as do the rotary
    /// factors: under Ulysses a rank attends every row for its own heads, so a tile's selection and
    /// its compression are per head and need no other rank.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        f: &Forward,
        normalized: TensorView,
        factors: Factors,
        sizes: TensorView,
        segments: TensorView,
        keep: TensorView,
        scratch: &Scratch,
        out: TensorView,
    ) -> Result<()> {
        self.attend(
            f,
            normalized,
            factors,
            sizes,
            segments,
            keep,
            scratch,
            Vec::new(),
        )?;
        // The compression is folded into the fine rows, in place, and one binding names both the
        // buffer the fold writes and the buffer the projection reads: `add_gated` modifies its first
        // argument only, so folding into any other buffer would drop this branch without a shape
        // ever saying so (`vsa/segments.py:461-478`).
        let rows = scratch.query.shape.dim(0);
        let shape = Shape::new(&[rows, u64::from(self.local_heads), u64::from(self.head_dim)]);
        self.tiles.add_gated(
            f,
            scratch.fine.reshape(shape),
            scratch.compressed,
            scratch.gate.reshape(shape),
        )?;
        Ok(self.output.forward(f, scratch.fine, out)?)
    }

    /// Whether this layer's rows return from the other members' heads: the layer then runs as
    /// [`Self::attend`], [`Self::deliver`] and [`Self::receive`] rather than [`Self::forward`].
    pub fn exchanges(&self) -> bool {
        self.exchange.is_some()
    }

    /// Whether this layer's input rows are gathered ahead of it: the gather mode reads the whole
    /// sequence, and what fills it is started as the rows are normalized, which the layer before
    /// does interval by interval.
    pub fn gathers_ahead(&self) -> bool {
        self.exchange.is_some() && self.projection == Projection::Gather
    }

    /// This rank's rows `first..first + count` of the gathered sequence, of its `own` rows.
    pub fn gathered_rows(&self, scratch: &Scratch, own: u64, first: u64, count: u64) -> TensorView {
        let rows = scratch.gathered_rows;
        let width = rows.shape.last();
        rows.at(rows.offset_elements((self.rank * own + first) * width))
            .reshape(Shape::new(&[count, width]))
    }

    /// Starts delivering rows `first..first + count` of this rank's `own`, which
    /// [`Self::gathered_rows`] holds, and every member's rows of that interval here.
    pub fn start_gather(
        &self,
        f: &Forward,
        scratch: &Scratch,
        own: u64,
        first: u64,
        count: u64,
    ) -> Result<nn::Transfer> {
        let exchange = self
            .exchange
            .as_ref()
            .ok_or_else(|| Error::config("a layer of one rank gathers no rows".to_owned()))?;
        Ok(exchange.start_gather(f, scratch.gathered_rows, own, first, count)?)
    }

    /// Attends this rank's heads of every row, up to the fine rows and the compression.
    ///
    /// In the exchange mode the four projections alternate with their transfers: each member's
    /// heads of a projection travel while the next one multiplies, on the group's lane, so the two
    /// staging buffers take turns, and a buffer is written again only after the transfer that read
    /// it. The gate is read last, so its transfer is answered still running, and it travels beside
    /// the attention.
    #[allow(clippy::too_many_arguments)]
    pub fn attend(
        &self,
        f: &Forward,
        normalized: TensorView,
        factors: Factors,
        sizes: TensorView,
        segments: TensorView,
        keep: TensorView,
        scratch: &Scratch,
        arrived: Vec<nn::Transfer>,
    ) -> Result<Option<nn::Transfer>> {
        // The sequence is whole here whatever this rank owns of it: the buffers below are the rows of
        // every member, in member order, for this rank's heads.
        let rows = scratch.query.shape.dim(0);
        if !rows.is_multiple_of(u64::from(self.tile_rows)) {
            return Err(Error::config(format!(
                "{rows} rows do not fill {} tiles",
                self.tile_rows
            )));
        }
        let exchange = self.exchange.as_ref();
        let gate = match (exchange, self.projection) {
            (Some(exchange), Projection::Exchange) => {
                let (even, odd) = (scratch.staging, scratch.returned);
                let query = self
                    .query
                    .start(f, exchange, normalized, even, scratch.query)?;
                let key = self.key.start(f, exchange, normalized, odd, scratch.key)?;
                exchange.finish(f, query)?;
                let value = self
                    .value
                    .start(f, exchange, normalized, even, scratch.value)?;
                exchange.finish(f, key)?;
                let gate = self
                    .gate
                    .start(f, exchange, normalized, odd, scratch.gate)?;
                exchange.finish(f, value)?;
                Some(gate)
            }
            _ => {
                let source = match exchange {
                    Some(exchange) => {
                        for transfer in arrived {
                            exchange.finish(f, transfer)?;
                        }
                        scratch.gathered_rows
                    }
                    None => normalized,
                };
                self.query
                    .forward(f, exchange, source, scratch.staging, scratch.query)?;
                self.key
                    .forward(f, exchange, source, scratch.staging, scratch.key)?;
                self.value
                    .forward(f, exchange, source, scratch.staging, scratch.value)?;
                self.gate
                    .forward(f, exchange, source, scratch.staging, scratch.gate)?;
                None
            }
        };

        // The rows are `[tiles * tile_rows, heads, width]` for the preparation; the pooled views are
        // `[tiles, heads, width]` FP32 and stay tile-major, which is the order the reference's
        // permutation reads them in.
        let shape = Shape::new(&[rows, u64::from(self.local_heads), u64::from(self.head_dim)]);
        let (query, key, value) = (
            scratch.query.reshape(shape),
            scratch.key.reshape(shape),
            scratch.value.reshape(shape),
        );
        self.tiles.prepare(
            f,
            query,
            key,
            value,
            Some(&self.query_gain),
            Some(&self.key_gain),
            Some(factors.cosine),
            Some(factors.sine),
            sizes,
            scratch.pooled_query,
            scratch.pooled_key,
            scratch.pooled_value,
        )?;
        // `prepare` already stored zeros past each tile's valid size, for all three of Q, K and V,
        // before it pooled them (`vsa_prepare_tiles.py:812-831`), so the fine attention reads the
        // same zeroed rows without a second pass: a `zero_padding` per tensor here would repeat
        // three launches of every layer of every evaluation to write what is already written.
        self.tiles
            .scores(f, scratch.pooled_query, scratch.pooled_key, scratch.scores)?;
        // `segment_keep` is over-allocated to the tile count and filled by segment index, as the
        // reference builds it; `select` reads its entries by segment and ignores the rest.
        self.tiles.select(
            f,
            scratch.scores,
            segments,
            sizes,
            keep,
            scratch.indices,
            scratch.counts,
            0,
        )?;

        // The fine branch reads the selected tiles of the rows in the checkpoint's dtype.
        let tables = SegmentTables {
            indices: scratch.indices,
            counts: scratch.counts,
            sizes,
        };
        // The fine rows are `[rows, heads, width]` for the attention and the fold, and the same
        // bytes are `[rows, inner]` for the projection that reads them.
        let fine = scratch.fine.reshape(shape);
        self.fine.forward(f, query, key, value, &tables, fine)?;

        // Branch 3: one compression per (head, query tile) from the pooled scores and values, a
        // dense softmax over every *live* key tile in FP32 with one rounding to the rows' dtype. A
        // tile whose rows are all invalid is masked out of the softmax rather than zeroed - a zeroed
        // pooled value under a live score would still take weight and scale the others down.
        self.tiles.compress(
            f,
            scratch.scores,
            scratch.pooled_value,
            sizes,
            scratch.compressed,
        )?;
        Ok(gate)
    }

    /// Composes this rank's attended heads and starts returning them to their rows' owners.
    ///
    /// Every interval of an owner's `own` rows is composed into its payload and started here, as
    /// the reference's `token_chunks` does: the lane then carries them one after another, and
    /// [`Self::receive`] takes up an interval while the later ones travel. `gate` is the transfer
    /// [`Self::attend`] left running. An interval is whole row blocks of the composition, and one
    /// member's rows are whole rows of a tile's share.
    pub fn deliver(
        &self,
        f: &Forward,
        gate: Option<nn::Transfer>,
        scratch: &Scratch,
        own: u64,
    ) -> Result<Vec<Returning>> {
        let exchange = self
            .exchange
            .as_ref()
            .ok_or_else(|| Error::config("a layer of one rank returns no rows".to_owned()))?;
        if let Some(gate) = gate {
            exchange.finish(f, gate)?;
        }
        let members = exchange.ranks();
        let heads = u64::from(self.local_heads);
        let width = u64::from(self.head_dim);
        let shape = Shape::new(&[members * own, heads, width]);
        let (fine, gate) = (scratch.fine.reshape(shape), scratch.gate.reshape(shape));
        let slots: Vec<u32> = (0..members as u32).collect();
        let unit = (u64::from(self.tile_rows) / members).max(1);
        let total: u64 = RETURN_SHARES.iter().sum();
        let mut returning = Vec::new();
        let mut first = 0;
        let mut reached = 0;
        for share in RETURN_SHARES {
            reached += share;
            let end = (own * reached / total).next_multiple_of(unit).min(own);
            if end == first {
                continue;
            }
            let count = end - first;
            let block = Shape::new(&[members, count, heads, width]);
            let at = first * members * heads * width;
            let payload = scratch
                .staging
                .at(scratch.staging.offset_elements(at))
                .reshape(block);
            let landed = scratch
                .returned
                .at(scratch.returned.offset_elements(at))
                .reshape(block);
            self.tiles.compose(
                f,
                payload,
                fine,
                scratch.compressed,
                gate,
                own,
                first,
                count,
                &slots,
            )?;
            returning.push(Returning {
                first,
                count,
                landed,
                transfer: exchange.start(f, payload, landed)?,
            });
            first = end;
        }
        Ok(returning)
    }

    /// Takes up one returning interval and projects it into `out`, its `[count, hidden]` rows.
    pub fn receive(
        &self,
        f: &Forward,
        interval: Returning,
        scratch: &Scratch,
        out: TensorView,
    ) -> Result<()> {
        let exchange = self
            .exchange
            .as_ref()
            .ok_or_else(|| Error::config("a layer of one rank returns no rows".to_owned()))?;
        exchange.finish(f, interval.transfer)?;
        let every = u64::from(self.heads) * u64::from(self.head_dim);
        let merged = scratch
            .merged
            .at(scratch.merged.offset_elements(interval.first * every))
            .reshape(Shape::new(&[interval.count, every]));
        HeadsToRows::merge(f, exchange.ranks(), interval.landed, merged)?;
        Ok(self.output.forward(f, merged, out)?)
    }
}

/// The intervals an owner's rows return in, as shares of the rows.
///
/// The first interval's transfer is the part of the return nothing hides, and the last interval's
/// gather is the part of the next layer's input nothing hides, so the two ends are the small ones:
/// an interval's output projection and feed-forward take longer per row than either transfer.
const RETURN_SHARES: [u64; 4] = [1, 2, 2, 1];

/// One interval of this rank's rows on its way back from the members that attended it.
pub struct Returning {
    /// The interval's first row, of this rank's rows.
    pub first: u64,
    /// The interval's rows.
    pub count: u64,
    /// Where the members' heads of those rows land, `[members, count, heads, width]`.
    landed: TensorView,
    /// The transfer that fills `landed`.
    transfer: nn::Transfer,
}

/// The per-request device views a sparse stack's layers read.
///
/// `sizes` is `[tiles]` int32, the valid rows of each tile; `segments` is `[tiles]` int32, the video
/// segment of each tile; `keep` is the `segment_keep` table, over-allocated to the tile count as the
/// reference builds it.
#[derive(Clone, Copy)]
pub struct Segments {
    pub sizes: TensorView,
    pub segments: TensorView,
    pub keep: TensorView,
}

/// Every temporary view the sparse branch reads, reserved once for the whole stack.
///
/// The shapes do not depend on the layer, and the executor's arena is a bump allocator reset once
/// per forward, so a per-layer reservation would multiply the arena by the layer count - two
/// `tiles * tiles * heads` matrices per layer is gigabytes at a served canvas.
pub struct Scratch {
    pub query: TensorView,
    pub key: TensorView,
    pub value: TensorView,
    pub gate: TensorView,
    pub pooled_query: TensorView,
    pub pooled_key: TensorView,
    pub pooled_value: TensorView,
    pub scores: TensorView,
    pub indices: TensorView,
    pub counts: TensorView,
    pub fine: TensorView,
    pub compressed: TensorView,
    /// The exchange's staging buffer, a member's block after another, for this rank's rows.
    pub staging: TensorView,
    /// The staging buffer's twin: a projection's block while the one before it travels, and the
    /// landing of the attended rows on their way back.
    pub returned: TensorView,
    /// The attended rows of every head, as the output projection reads them.
    pub merged: TensorView,
    /// The whole sequence's normalized rows, in the gather mode.
    pub gathered_rows: TensorView,
}

impl Scratch {
    /// Reserve the stack's sparse scratch for a sequence of `total_rows` rows, of which `normalized`
    /// holds this rank's.
    pub fn reserve(
        layer: &Sparse,
        f: &Forward,
        normalized: TensorView,
        total_rows: u32,
    ) -> Result<Self> {
        let dtype = normalized.dtype;
        let rows = u64::from(total_rows);
        let own = normalized.shape.dim(0);
        let tiles64 = rows / u64::from(layer.tile_rows);
        let inner = u64::from(layer.local_heads) * u64::from(layer.head_dim);
        let heads = u64::from(layer.local_heads);
        let every = u64::from(layer.heads) * u64::from(layer.head_dim);
        let arena = f.arena();
        // A layer that does not communicate reserves nothing for it: on one rank these views are
        // buffers the layer already holds, and the branches that read them do not run.
        let (staging, returned, merged) = match layer.exchange {
            Some(_) => (
                arena.tensor(dtype, Shape::new(&[own, every]))?,
                arena.tensor(dtype, Shape::new(&[own, every]))?,
                arena.tensor(dtype, Shape::new(&[own, every]))?,
            ),
            None => (normalized, normalized, normalized),
        };
        let gathered_rows = match (&layer.exchange, layer.projection) {
            (Some(_), Projection::Gather) => {
                arena.tensor(dtype, Shape::new(&[rows, normalized.shape.last()]))?
            }
            _ => normalized,
        };
        Ok(Self {
            staging,
            returned,
            merged,
            gathered_rows,
            query: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            key: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            value: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            gate: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            pooled_query: arena.tensor(
                nn::DType::F32,
                Shape::new(&[tiles64, heads, u64::from(layer.head_dim)]),
            )?,
            pooled_key: arena.tensor(
                nn::DType::F32,
                Shape::new(&[tiles64, heads, u64::from(layer.head_dim)]),
            )?,
            pooled_value: arena.tensor(
                nn::DType::F32,
                Shape::new(&[tiles64, heads, u64::from(layer.head_dim)]),
            )?,
            scores: arena.tensor(nn::DType::F32, Shape::new(&[heads, tiles64, tiles64]))?,
            // The selection is as wide as the scores it comes from: `select` requires the index
            // matrix to have the scores' shape, and the reference returns one entry per key tile
            // with the kept ones first (`vsa/segments.py:205-208`).
            indices: arena.tensor(nn::DType::I32, Shape::new(&[heads, tiles64, tiles64]))?,
            counts: arena.tensor(nn::DType::I32, Shape::new(&[heads, tiles64]))?,
            fine: arena.tensor(dtype, Shape::new(&[rows, inner]))?,
            // `[heads, tiles, width]`: what `compress` writes and what the gate step reads, one row
            // per (head, query tile) - every row of a query tile compresses to the same vector.
            compressed: arena.tensor(
                dtype,
                Shape::new(&[heads, tiles64, u64::from(layer.head_dim)]),
            )?,
        })
    }
}

/// The rotary frequency table of the sparse path: the dense path's, built the same way.
///
/// One `rope_frequency_dim` spectrum repeated once per coordinate axis, because the reference hands
/// its rotary one axis spanning the whole head and flattens the three axes' factors into three
/// contiguous blocks (`attention.rs::frequencies` states the evidence).
fn frequencies(
    loader: &mut WeightLoader<'_>,
    config: &TransformerConfig,
    rotated: u32,
    axes: u32,
) -> Result<Weight> {
    let pairs = config.rope_frequency_dim;
    if pairs == 0 || axes == 0 || rotated != 2 * pairs * axes {
        return Err(Error::config(format!(
            "{rotated} rotated channels are not {axes} axes of {pairs} frequency pairs"
        )));
    }
    let mut bytes = Vec::with_capacity(rotated as usize * 2);
    for value in crate::attention::table(pairs, axes, config.rope_theta) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let name = format!("transformer.rotary.{pairs}x{axes}.{}", config.rope_theta);
    let half = u64::from(rotated / 2);
    Ok(loader.constant(
        &name,
        nn::DType::F32,
        Shape::new(&[half]),
        &bytes,
        "minimax-h3",
    )?)
}

fn projection(
    loader: &mut WeightLoader<'_>,
    name: &str,
    out_features: u64,
    in_features: u64,
) -> Result<Linear> {
    let weight = loader.load(&WeightRequest::region(
        name,
        0,
        Shape::new(&[out_features, in_features]),
    ))?;
    Ok(Linear::new(weight, None, WeightOrder::NxK)?)
}

fn gain(loader: &mut WeightLoader<'_>, name: &str, width: u64) -> Result<Weight> {
    Ok(loader.load(&WeightRequest::region(name, 0, Shape::new(&[width])))?)
}
