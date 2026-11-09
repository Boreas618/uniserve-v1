//! FastH3's video sparse attention layer: the VSA attention over this rank's heads of the whole
//! sequence, inside the sequence-parallel projections.
//!
//! Under Ulysses the four projections (query, key, value and the compression's gate) travel to their
//! heads one after another while the next one multiplies, and the attended rows return to their
//! owners interval by interval, each interval's gated composition started as soon as it is composed,
//! so the transformer can project and feed forward an interval while the later ones travel.

use nn::family::Projection;
use nn::layers::{
    Interval, Rotary, UlyssesAttention, UlyssesSpec, VsaAttention, VsaScratch, VsaSpec, VsaTables,
};
use nn::weights::{Weight, WeightLoader};
use nn::{Forward, Shape, TensorView};

use crate::attention::{Factors, sequence_split};
use crate::config::TransformerConfig;
use crate::partition::Partition;
use crate::{Error, Result};

/// The input projections a sparse layer reads, in the order the attention reads them.
const INPUTS: [&str; 4] = ["to_q", "to_k", "to_v", "to_gate_compress"];

/// The index of the compression's gate among [`INPUTS`].
const GATE: usize = 3;

/// One layer's sparse attention: four projections, the tile attention, and the output projection.
pub struct Sparse {
    ulysses: UlyssesAttention<VsaAttention>,
    /// Per-head normalization gains of the query and the key.
    query_gain: Weight,
    key_gain: Weight,
    /// Elements of one head.
    head_dim: u32,
    /// Rows of one tile.
    tile_rows: u32,
}

impl Sparse {
    /// Read layer `index`'s sparse attention of a DiT partition.
    ///
    /// `rows` bounds the keys of one call, `rotary` is the stack's (the tile preparation turns the
    /// channels it turns, by the same tables), and `tile_bound` is the deployment's key-tile bound
    /// every compression reduces over (`Layouts::tiles_bound`).
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        index: u32,
        config: &TransformerConfig,
        rows: u32,
        rotary: &Rotary,
        tile_rows: u32,
        tile_bound: u32,
        partition: &Partition,
        loader: &mut WeightLoader<'_>,
    ) -> Result<Self> {
        let head = u64::from(config.head_dim);
        let rotated = rotary.spectrum().rotated();
        let mut scope = loader.scope(format!("transformer_blocks.{index}.attn"));
        let query_gain = scope.scope("norm_q").tensor("weight", &[head])?;
        let key_gain = scope.scope("norm_k").tensor("weight", &[head])?;
        let local_heads = partition.denoiser_heads(config.num_attention_heads);
        let ulysses = UlyssesAttention::load(
            &mut scope,
            UlyssesSpec {
                hidden: config.hidden_size,
                heads: config.num_attention_heads,
                head_dim: config.head_dim,
                inputs: &INPUTS,
                output: "to_out.0",
            },
            sequence_split(partition)?,
            partition.projection,
            // The fine attention writes the rows the projections produce, in the stack's own type.
            |stack| {
                VsaAttention::new(VsaSpec {
                    tile_rows,
                    heads: local_heads,
                    head_dim: config.head_dim,
                    qk_norm: true,
                    epsilon: config.qk_norm_eps,
                    rotated,
                    max_tiles: tile_bound,
                    max_keys: rows,
                    output: stack,
                })
            },
        )?;
        Ok(Self {
            ulysses,
            query_gain,
            key_gain,
            head_dim: config.head_dim,
            tile_rows,
        })
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
        factors: Factors<'_>,
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
        // The compressed rows are broadcast over their query tile and folded into the fine rows in
        // place, and the output projection reads the same buffer.
        let rows = scratch.query.shape.dim(0);
        let shape = self.headed(rows);
        self.ulysses.inner().add_gated(
            f,
            scratch.vsa.fine.reshape(shape),
            scratch.vsa.compressed,
            scratch.gate.reshape(shape),
        )?;
        Ok(self
            .ulysses
            .return_rows(f, scratch.vsa.fine, scratch.staging, scratch.merged, out)?)
    }

    /// `[rows, local heads, head_dim]`.
    fn headed(&self, rows: u64) -> Shape {
        Shape::new(&[
            rows,
            u64::from(self.ulysses.local_heads()),
            u64::from(self.head_dim),
        ])
    }

    /// Whether this layer's rows return from the other members' heads: the layer then runs as
    /// [`Self::attend`], [`Self::deliver`] and [`Self::receive`] rather than [`Self::forward`].
    pub fn exchanges(&self) -> bool {
        self.ulysses.exchanges()
    }

    /// Whether this layer's input rows are gathered ahead of it: the gather mode reads the whole
    /// sequence, and what fills it is started as the rows are normalized, which the layer before
    /// does interval by interval.
    pub fn gathers_ahead(&self) -> bool {
        self.ulysses.exchanges() && self.ulysses.projection() == Projection::Gather
    }

    /// This rank's rows `first..first + count` of the gathered sequence, of its `own` rows.
    pub fn gathered_rows(&self, scratch: &Scratch, own: u64, first: u64, count: u64) -> TensorView {
        let rows = scratch.gathered_rows;
        let width = rows.shape.last();
        let rank = u64::from(self.ulysses.split().rank);
        rows.at(rows.offset_elements((rank * own + first) * width))
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
        if !self.ulysses.exchanges() {
            return Err(Error::config(
                "a layer of one rank gathers no rows".to_owned(),
            ));
        }
        Ok(self
            .ulysses
            .start_gather(f, scratch.gathered_rows, own, first, count)?)
    }

    /// Attends this rank's heads of every row, up to the fine rows and the compression.
    ///
    /// In the exchange mode the four projections alternate with their transfers: each member's
    /// heads of a projection travel while the next one multiplies, so the two staging buffers take
    /// turns, and a buffer is written again only after the transfer that read it. The gate is read
    /// last, so its transfer is answered still running, and it travels beside the attention.
    #[allow(clippy::too_many_arguments)]
    pub fn attend(
        &self,
        f: &Forward,
        normalized: TensorView,
        factors: Factors<'_>,
        sizes: TensorView,
        segments: TensorView,
        keep: TensorView,
        scratch: &Scratch,
        arrived: Vec<nn::Transfer>,
    ) -> Result<Option<nn::Transfer>> {
        // The sequence is whole here whatever this rank owns of it: the buffers below are the rows
        // of every member, in member order, for this rank's heads.
        let rows = scratch.query.shape.dim(0);
        if !rows.is_multiple_of(u64::from(self.tile_rows)) {
            return Err(Error::config(format!(
                "{rows} rows do not fill {} tiles",
                self.tile_rows
            )));
        }
        let ulysses = &self.ulysses;
        let projected = [scratch.query, scratch.key, scratch.value, scratch.gate];
        let gate = if ulysses.exchanges() && ulysses.projection() == Projection::Exchange {
            let (even, odd) = (scratch.staging, scratch.returned);
            let query = ulysses.start_projection(f, 0, normalized, even, projected[0])?;
            let key = ulysses.start_projection(f, 1, normalized, odd, projected[1])?;
            ulysses.finish(f, query)?;
            let value = ulysses.start_projection(f, 2, normalized, even, projected[2])?;
            ulysses.finish(f, key)?;
            let gate = ulysses.start_projection(f, GATE, normalized, odd, projected[GATE])?;
            ulysses.finish(f, value)?;
            Some(gate)
        } else {
            let source = if ulysses.exchanges() {
                for transfer in arrived {
                    ulysses.finish(f, transfer)?;
                }
                scratch.gathered_rows
            } else {
                normalized
            };
            for (index, out) in projected.into_iter().enumerate() {
                ulysses.project(f, index, source, scratch.staging, out)?;
            }
            None
        };

        // The rows are `[tiles * tile_rows, heads, width]` for the preparation and the attention.
        let shape = self.headed(rows);
        let tables = VsaTables {
            sizes,
            segments,
            keep,
        };
        ulysses.inner().attend(
            f,
            scratch.query.reshape(shape),
            scratch.key.reshape(shape),
            scratch.value.reshape(shape),
            Some((&self.query_gain, &self.key_gain)),
            Some((factors.cosine, factors.sine)),
            &tables,
            &scratch.vsa,
        )?;
        Ok(gate)
    }

    /// Composes this rank's attended heads and starts returning them to their rows' owners.
    ///
    /// Every interval of an owner's `own` rows is composed into its payload and started here: the
    /// lane then carries them one after another, and [`Self::receive`] takes up an interval while
    /// the later ones travel. `gate` is the transfer [`Self::attend`] left running. An interval is
    /// whole row blocks of the composition, and one member's rows are whole rows of a tile's share.
    pub fn deliver(
        &self,
        f: &Forward,
        gate: Option<nn::Transfer>,
        scratch: &Scratch,
        own: u64,
    ) -> Result<Vec<Returning>> {
        let ulysses = &self.ulysses;
        if !ulysses.exchanges() {
            return Err(Error::config(
                "a layer of one rank returns no rows".to_owned(),
            ));
        }
        if let Some(gate) = gate {
            ulysses.finish(f, gate)?;
        }
        let members = u64::from(ulysses.split().size);
        let heads = u64::from(ulysses.local_heads());
        let width = u64::from(self.head_dim);
        let shape = Shape::new(&[members * own, heads, width]);
        let (fine, gate) = (scratch.vsa.fine.reshape(shape), scratch.gate.reshape(shape));
        let slots: Vec<u32> = (0..members as u32).collect();
        let unit = u64::from(self.tile_rows) / members;
        let mut returning = Vec::new();
        for interval in UlyssesAttention::<VsaAttention>::intervals(own, unit) {
            let Interval { first, count } = interval;
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
            ulysses.inner().compose(
                f,
                payload,
                fine,
                scratch.vsa.compressed,
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
                transfer: ulysses.start_return(f, payload, landed)?,
            });
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
        let every = u64::from(self.ulysses.heads()) * u64::from(self.head_dim);
        let merged = scratch
            .merged
            .at(scratch.merged.offset_elements(interval.first * every))
            .reshape(Shape::new(&[interval.count, every]));
        Ok(self
            .ulysses
            .receive(f, interval.transfer, interval.landed, merged, out)?)
    }
}

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
/// The shapes do not depend on the layer, and the worker's arena is a bump allocator reset once
/// per forward, so a per-layer reservation would multiply the arena by the layer count - two
/// `tiles * tiles * heads` matrices per layer is gigabytes at a served canvas.
pub struct Scratch {
    pub query: TensorView,
    pub key: TensorView,
    pub value: TensorView,
    pub gate: TensorView,
    /// The sparse attention's stage views.
    pub vsa: VsaScratch,
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
        let ulysses = &layer.ulysses;
        let inner = u64::from(ulysses.local_heads()) * u64::from(layer.head_dim);
        let every = u64::from(ulysses.heads()) * u64::from(layer.head_dim);
        let arena = f.arena();
        // A layer that does not communicate reserves nothing for it: on one rank these views are
        // buffers the layer already holds, and the branches that read them do not run.
        let (staging, returned, merged) = if ulysses.exchanges() {
            (
                arena.tensor(dtype, Shape::new(&[own, every]))?,
                arena.tensor(dtype, Shape::new(&[own, every]))?,
                arena.tensor(dtype, Shape::new(&[own, every]))?,
            )
        } else {
            (normalized, normalized, normalized)
        };
        let gathered_rows = if layer.gathers_ahead() {
            arena.tensor(dtype, Shape::new(&[rows, normalized.shape.last()]))?
        } else {
            normalized
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
            vsa: VsaScratch::reserve(arena, ulysses.inner(), rows, dtype)?,
        })
    }
}
