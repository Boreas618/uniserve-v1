//! FastH3's video sparse attention layer: the VSA attention over this rank's heads of the whole
//! sequence, inside the sequence-parallel projections.
//!
//! The attention has two branches (`nn::layers::VsaAttention`): a fine attention over each query
//! tile's kept key tiles and a compressed attention over every tile's means, folded together by the
//! layer's own gate projection.
//!
//! Under Ulysses exchange the four projections (query, key, value and the compression's gate)
//! travel to their heads one after another while the next one multiplies, and the attended rows
//! return to their owners interval by interval ([`Sparse::deliver`], [`Sparse::receive`]), so the
//! transformer can project and feed forward an interval while the later ones travel.

use nn::family::{Projection, SequenceSplit};
use nn::layers::{
    Branches, Factors, Interval, Rotary, UlyssesAttention, UlyssesSpec, VsaAttention, VsaSpec,
    VsaTables,
};
use nn::weights::{Scope, Weight};
use nn::{Forward, Shape, Operand, Transfer};

use crate::attention::heads;
use crate::config::TransformerConfig;
use crate::{Error, Result};

/// The input projections a sparse layer reads, in the order the attention reads them.
const INPUTS: [&str; 4] = ["to_q", "to_k", "to_v", "to_gate_compress"];

/// The index of the compression's gate among [`INPUTS`].
const GATE: usize = 3;

/// One layer's sparse attention: four projections, the tile attention, and the output projection.
pub struct Sparse {
    ulysses: UlyssesAttention<VsaAttention>,
    query_gain: Weight,
    key_gain: Weight,
    head_dim: u32,
    tile_rows: u32,
}

impl Sparse {
    /// Read the sparse attention at `scope`.
    ///
    /// `rotary` is the stack's (the tile preparation turns the channels it turns, by the same
    /// factors); `bounds` is the deployment's `(packed rows, key tiles)`.
    pub(crate) fn load(
        scope: &mut Scope<'_, '_>,
        config: &TransformerConfig,
        rotary: &Rotary,
        tile_rows: u32,
        (keys, tiles): (u32, u32),
        sequence: SequenceSplit,
    ) -> Result<Self> {
        if tiles == 0 {
            return Err(Error::config(
                "a sparse layer reduces over the deployment's key tiles, and none were stated"
                    .to_owned(),
            ));
        }
        let head = u64::from(config.head_dim);
        let rotated = rotary.spectrum().rotated();
        let query_gain = scope.scope("norm_q").tensor("weight", &[head])?;
        let key_gain = scope.scope("norm_k").tensor("weight", &[head])?;
        let local_heads = config.num_attention_heads / sequence.split.size.max(1);
        let ulysses = UlyssesAttention::load(
            scope,
            UlyssesSpec {
                hidden: config.hidden_size,
                heads: config.num_attention_heads,
                head_dim: config.head_dim,
                inputs: &INPUTS,
                output: "to_out.0",
            },
            sequence.split,
            sequence.projection,
            // The fine attention writes the rows the projections produce, in the stack's type.
            |stack| {
                VsaAttention::new(VsaSpec {
                    tile_rows,
                    heads: local_heads,
                    head_dim: config.head_dim,
                    qk_norm: true,
                    epsilon: config.qk_norm_eps,
                    rotated,
                    max_tiles: tiles,
                    max_keys: keys,
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

    /// Whether this layer's rows cross ranks, so they return interval by interval
    /// ([`Self::deliver`], [`Self::receive`]) rather than in one [`Self::forward`].
    pub fn exchanges(&self) -> bool {
        self.ulysses.exchanges()
    }

    /// Whether this layer reads the whole sequence's normalized rows gathered from every rank
    /// ([`Projection::Gather`]): the stack then gathers the next layer's rows interval by interval
    /// while this layer computes ([`Self::start_gather`]).
    pub fn gathers_ahead(&self) -> bool {
        self.ulysses.exchanges() && self.ulysses.projection() == Projection::Gather
    }

    /// Starts delivering rows `first..first + count` of this rank's `own` rows, already in its
    /// block of `gathered` (the whole sequence, `[ranks · own, hidden]`), to every rank, and
    /// theirs here.
    pub fn start_gather(
        &self,
        f: &Forward,
        gathered: Operand,
        own: u64,
        (first, count): (u64, u64),
    ) -> Result<Transfer> {
        Ok(self.ulysses.start_gather(f, gathered, own, first, count)?)
    }

    /// This rank's index in the layer's sequence split.
    pub fn rank(&self) -> u32 {
        self.ulysses.split().rank
    }

    /// The ranks of the layer's sequence split.
    pub fn members(&self) -> u32 {
        self.ulysses.split().size
    }

    /// Takes up a gather [`Self::start_gather`] started.
    pub fn finish(&self, f: &Forward, transfer: Transfer) -> Result<()> {
        Ok(self.ulysses.finish(f, transfer)?)
    }

    /// Attend `normalized`, this rank's `[rows, hidden]`, into a fresh `[rows, hidden]`.
    pub(crate) fn forward(
        &self,
        f: &Forward,
        normalized: Operand,
        factors: Factors,
        tables: &VsaTables,
    ) -> Result<Operand> {
        if self.exchanges() {
            let own = normalized.rows();
            let output = f.empty(normalized.dtype, normalized.shape)?;
            // Under the gather projection the projections read the whole sequence's rows.
            let rows = self.ulysses.rows(f, normalized)?;
            let attended = self.attend(f, rows, factors, tables)?;
            for returning in self.deliver(f, attended, own)? {
                let (interval, ..) = returning;
                let rows = output.narrow(0, interval.first, interval.count)?;
                self.receive(f, returning, rows)?;
            }
            return Ok(output);
        }
        let (branches, gate, _) = self.attend(f, normalized, factors, tables)?;
        let shape = self.headed(branches.fine);
        self.ulysses.inner().add_gated(
            f,
            branches.fine.reshape(shape),
            branches.compressed,
            gate.reshape(shape),
        )?;
        Ok(self.ulysses.return_rows(f, branches.fine)?)
    }

    /// `[rows, local heads, head_dim]` of a `[rows, local heads · head_dim]` view.
    fn headed(&self, view: Operand) -> Shape {
        heads(view, self.ulysses.local_heads(), self.head_dim)
    }

    /// Attend this rank's heads of every row: the two branches and the gate projection.
    ///
    /// In the exchange mode the four projections alternate with their transfers, each member's
    /// heads of a projection travelling while the next one multiplies; the gate is projected last,
    /// so its transfer is answered still running and travels beside the attention.
    ///
    /// `rows` is this rank's own normalized rows, or under [`Projection::Gather`] across ranks the
    /// whole sequence's, gathered. The tile preparation turns the channels the stack's rotary
    /// turns, by the same `factors`.
    pub fn attend(
        &self,
        f: &Forward,
        rows: Operand,
        factors: Factors,
        tables: &VsaTables,
    ) -> Result<(Branches, Operand, Option<Transfer>)> {
        let ulysses = &self.ulysses;
        let normalized = rows;
        let ([query, key, value, gate], pending) =
            if self.exchanges() && ulysses.projection() == Projection::Exchange {
                let (query, sending_query) = ulysses.start_projection(f, 0, normalized)?;
                let (key, sending_key) = ulysses.start_projection(f, 1, normalized)?;
                ulysses.finish(f, sending_query)?;
                let (value, sending_value) = ulysses.start_projection(f, 2, normalized)?;
                ulysses.finish(f, sending_key)?;
                let (gate, sending_gate) = ulysses.start_projection(f, GATE, normalized)?;
                ulysses.finish(f, sending_value)?;
                ([query, key, value, gate], Some(sending_gate))
            } else {
                let projected = [0, 1, 2, GATE].map(|index| ulysses.project(f, index, rows));
                let [query, key, value, gate] = projected;
                ([query?, key?, value?, gate?], None)
            };
        let sequence = query.rows();
        if !sequence.is_multiple_of(u64::from(self.tile_rows)) {
            return Err(Error::config(format!(
                "{sequence} rows do not fill tiles of {}",
                self.tile_rows
            )));
        }
        let shape = self.headed(query);
        let branches = ulysses.inner().attend(
            f,
            (
                query.reshape(shape),
                key.reshape(shape),
                value.reshape(shape),
            ),
            Some((&self.query_gain, &self.key_gain)),
            Some(factors),
            tables,
        )?;
        Ok((branches, gate, pending))
    }

    /// Composes this rank's attended heads and starts returning them to their rows' owners, one
    /// interval of each owner's `own` rows at a time: the lane carries them one after another, and
    /// [`Self::receive`] takes up an interval while the later ones travel.
    pub fn deliver(
        &self,
        f: &Forward,
        (branches, gate, pending): (Branches, Operand, Option<Transfer>),
        own: u64,
    ) -> Result<Vec<(Interval, Operand, Transfer)>> {
        let ulysses = &self.ulysses;
        if let Some(pending) = pending {
            ulysses.finish(f, pending)?;
        }
        let members = u64::from(ulysses.split().size);
        let heads = u64::from(ulysses.local_heads());
        let width = u64::from(self.head_dim);
        let shape = Shape::new(&[members * own, heads, width]);
        let branches = Branches {
            fine: branches.fine.reshape(shape),
            compressed: branches.compressed,
        };
        let gate = gate.reshape(shape);
        let slots: Vec<u32> = (0..ulysses.split().size).collect();
        let unit = u64::from(self.tile_rows) / members.max(1);
        let mut returning = Vec::new();
        for interval in UlyssesAttention::<VsaAttention>::intervals(own, unit) {
            let payload = f.empty(
                branches.fine.dtype,
                Shape::new(&[members, interval.count, heads, width]),
            )?;
            ulysses
                .inner()
                .compose(f, payload, branches, gate, interval, &slots)?;
            let (landed, transfer) = ulysses.start_return(f, payload)?;
            returning.push((interval, landed, transfer));
        }
        Ok(returning)
    }

    /// Takes up one returning interval and projects it into `destination`, its `[count, hidden]`
    /// rows of the layer's output.
    pub fn receive(
        &self,
        f: &Forward,
        (_, landed, transfer): (Interval, Operand, Transfer),
        destination: Operand,
    ) -> Result<()> {
        Ok(self.ulysses.receive(f, transfer, landed, destination)?)
    }
}
