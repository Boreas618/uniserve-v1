//! Ulysses sequence parallelism around any attention: trade a row split for a head split and back.
//!
//! A member of a sequence-parallel group holds a contiguous slice of the packed rows and every head.
//! Its attention needs the opposite: every row of the sequence for its own share of the heads. Two
//! ways reach it, and the deployment states which ([`Projection`]):
//!
//! - **Exchange**: each member projects its own rows for every member's heads, one product per
//!   member so each member's block is contiguous, and one all-to-all delivers every block;
//! - **Gather**: each member gathers every member's normalized rows once and projects its own heads
//!   with a head-partitioned weight.
//!
//! After either, the attention reads `[rows of the whole sequence, this member's heads, head
//! width]`. Coming back, one all-to-all sends each member its rows of this member's heads, and the
//! received blocks are set side by side into `[this member's rows, every head]` before the output
//! projection. The return may be whole or by row intervals ([`UlyssesAttention::intervals`]), so an
//! interval's output projection runs while the later intervals travel.
//!
//! The collectives go through [`Forward`] (`all_gather`, `all_to_all`, `start_all_to_all`,
//! `start_gather_rows`, `finish`); the layer holds the split, never a communicator. The inner
//! attention `A` is any layer the model composes over the head-split rows; this layer owns the
//! projections around it.

use ops::{Shape, Strides, Operand};
use uniserve_tensor::DType;

use crate::error::{Error, Result};
use crate::family::{Projection, Split};
use crate::forward::Forward;
use crate::group::Transfer;
use crate::layers::{Access, Bias, Copy, Linear, Parallel};
use crate::weights::Scope;

/// The intervals a member's rows return in, as shares of the rows.
///
/// The first interval's transfer is the part of the return nothing hides, and the last interval's
/// is the part of the next layer's input nothing hides, so the two ends are the small ones.
const RETURN_SHARES: [u64; 4] = [1, 2, 2, 1];

/// The geometry and names of a [`UlyssesAttention`].
#[derive(Clone, Copy, Debug)]
pub struct UlyssesSpec<'n> {
    /// The width of a row.
    pub hidden: u32,
    /// The heads of the whole module, before any split.
    pub heads: u32,
    /// Channels of one head.
    pub head_dim: u32,
    /// The `[heads · head_dim, hidden]` input projections' module names, in the order the inner
    /// attention reads them.
    pub inputs: &'n [&'n str],
    /// The `[hidden, heads · head_dim]` output projection's module name.
    pub output: &'n str,
}

/// One input projection, as this member reaches its heads of every row.
struct HeadProjection {
    /// One member: the whole projection. Gather: this member's head rows of it. Exchange: one
    /// projection per member, in member order, each holding that member's head rows.
    members: Vec<Linear>,
}

/// One interval of a member's rows: `count` rows from `first`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    pub first: u64,
    pub count: u64,
}

/// Sequence-parallel attention: the head-split projections around an inner attention `A`.
pub struct UlyssesAttention<A> {
    inner: A,
    inputs: Vec<HeadProjection>,
    output: Linear,
    split: Split,
    projection: Projection,
    heads: u32,
    head_dim: u32,
}

impl<A> UlyssesAttention<A> {
    /// The projections the module at `scope` states around the attention `inner` builds, for member
    /// `split.rank` of a group of `split.size` reaching its heads by `projection`.
    ///
    /// `inner` is handed the element type the projections were read in - the stack's, which the
    /// attended rows are written in.
    ///
    /// # Errors
    ///
    /// A group whose size does not divide the heads, a projection the loader refuses, or the inner
    /// attention's refusal.
    pub fn load(
        scope: &mut Scope<'_, '_>,
        spec: UlyssesSpec<'_>,
        split: Split,
        projection: Projection,
        inner: impl FnOnce(DType) -> Result<A>,
    ) -> Result<Self> {
        if split.size == 0 || !spec.heads.is_multiple_of(split.size) {
            return Err(Error::shape(
                "UlyssesAttention",
                format!("{} heads over a group of {}", spec.heads, split.size),
            ));
        }
        let inner_width = spec.heads * spec.head_dim;
        let mut member = |name: &str, rank: u32| -> Result<Linear> {
            let parallel = if split.size == 1 {
                Parallel::None
            } else {
                Parallel::Column(Split {
                    size: split.size,
                    rank,
                })
            };
            Linear::load(
                &mut scope.scope(name),
                spec.hidden,
                inner_width,
                Bias::Absent,
                parallel,
            )
        };
        let inputs = spec
            .inputs
            .iter()
            .map(|name| {
                let members = match (split.size, projection) {
                    (1, _) => vec![member(name, 0)?],
                    (_, Projection::Gather) => vec![member(name, split.rank)?],
                    (_, Projection::Exchange) => (0..split.size)
                        .map(|rank| member(name, rank))
                        .collect::<Result<Vec<_>>>()?,
                };
                Ok(HeadProjection { members })
            })
            .collect::<Result<Vec<_>>>()?;
        let output = Linear::load(
            &mut scope.scope(spec.output),
            inner_width,
            spec.hidden,
            Bias::Absent,
            Parallel::None,
        )?;
        let inner = inner(output.weight().dtype)?;
        Ok(Self {
            inner,
            inputs,
            output,
            split,
            projection,
            heads: spec.heads,
            head_dim: spec.head_dim,
        })
    }

    /// The attention the projections surround.
    pub const fn inner(&self) -> &A {
        &self.inner
    }

    /// Whether the rows cross members: a group of more than one.
    pub const fn exchanges(&self) -> bool {
        self.split.size > 1
    }

    /// How this member reaches its heads.
    pub const fn projection(&self) -> Projection {
        self.projection
    }

    /// This member's place in the group.
    pub const fn split(&self) -> Split {
        self.split
    }

    /// The heads this member attends.
    pub const fn local_heads(&self) -> u32 {
        self.heads / self.split.size
    }

    /// The element type of the projections' weights: the stack's.
    pub fn dtype(&self) -> DType {
        self.output.weight().dtype
    }

    /// The rows the projections read: this member's own `normalized` rows, or the whole sequence's
    /// gathered into a fresh `[members · rows, hidden]` tensor when the deployment gathers.
    ///
    /// # Errors
    ///
    /// When the gather refuses its views, the arena cannot hold them or the forward carries no
    /// split.
    pub fn rows(&self, f: &Forward, normalized: Operand) -> Result<Operand> {
        if self.exchanges() && self.projection == Projection::Gather {
            self.require_split(f)?;
            let gathered = f.empty(
                normalized.dtype,
                Shape::new(&[
                    normalized.rows() * u64::from(self.split.size),
                    normalized.last(),
                ]),
            )?;
            f.all_gather(normalized, gathered)?;
            return Ok(gathered);
        }
        Ok(normalized)
    }

    /// The width of this member's heads of one projection: `local heads · head_dim`.
    fn local_width(&self) -> u64 {
        u64::from(self.local_heads()) * u64::from(self.head_dim)
    }

    /// Projection `index` of `rows` for every member's heads: one member's `[rows, local]` block
    /// after another in a fresh staging tensor, the exchange's send buffer.
    fn project_members(&self, f: &Forward, index: usize, rows: Operand) -> Result<Operand> {
        let local = self.local_width();
        let members = self.input(index)?.members.as_slice();
        let staging = f.empty(
            rows.dtype,
            Shape::new(&[members.len() as u64 * rows.rows(), local]),
        )?;
        let shape = Shape::new(&[rows.rows(), local]);
        let block = rows.rows() * local;
        for (member, projection) in members.iter().enumerate() {
            let at = staging.offset_elements(member as u64 * block);
            projection.launch(f, rows, staging.at(at).reshape(shape))?;
        }
        Ok(staging)
    }

    /// Input projection `index`.
    fn input(&self, index: usize) -> Result<&HeadProjection> {
        self.inputs.get(index).ok_or_else(|| {
            Error::shape(
                "UlyssesAttention",
                format!("input {index} of {} projections", self.inputs.len()),
            )
        })
    }

    /// Projection `index` of `rows` ([`Self::rows`]) as this member's heads of the whole sequence:
    /// a fresh `[sequence rows, local heads · head_dim]` tensor.
    ///
    /// # Errors
    ///
    /// When a projection or the exchange refuses its views, or the arena cannot hold them.
    pub fn project(&self, f: &Forward, index: usize, rows: Operand) -> Result<Operand> {
        match self.input(index)?.members.as_slice() {
            [only] => only.forward(f, rows),
            _ => {
                let staging = self.project_members(f, index, rows)?;
                let out = f.empty(staging.dtype, staging.shape)?;
                self.exchange(f, staging, out)?;
                Ok(out)
            }
        }
    }

    /// [`Self::project`] in the exchange mode, with the exchange started beside the forward's stream
    /// rather than waited for: the answered tensor is readable after [`Self::finish`] took the
    /// transfer up, and the send buffer belongs to the transfer until then.
    ///
    /// # Errors
    ///
    /// As [`Self::project`].
    pub fn start_projection(
        &self,
        f: &Forward,
        index: usize,
        rows: Operand,
    ) -> Result<(Operand, Transfer)> {
        let staging = self.project_members(f, index, rows)?;
        let out = f.empty(staging.dtype, staging.shape)?;
        self.require_split(f)?;
        let transfer = f.start_all_to_all(staging, out)?;
        Ok((out, transfer))
    }

    /// Orders the forward's stream after `transfer`: what it received is readable from here on.
    ///
    /// # Errors
    ///
    /// When the forward carries no split.
    pub fn finish(&self, f: &Forward, transfer: Transfer) -> Result<()> {
        self.require_split(f)?;
        f.finish(transfer)
    }

    /// Starts delivering rows `first..first + count` of this member's `own` rows to every member,
    /// and theirs here, in place in the whole sequence `gathered`, whose own block already holds
    /// this member's rows.
    ///
    /// # Errors
    ///
    /// When the forward carries no split.
    pub fn start_gather(
        &self,
        f: &Forward,
        gathered: Operand,
        own: u64,
        first: u64,
        count: u64,
    ) -> Result<Transfer> {
        self.require_split(f)?;
        f.start_gather_rows(gathered, own, first, count)
    }

    /// `attended` `[sequence rows, local heads · head_dim]` returned to each member's rows and this
    /// member's rows of every head projected: a fresh `[own rows, hidden]` tensor.
    ///
    /// With one member the attended rows are this member's and only the projection runs.
    ///
    /// # Errors
    ///
    /// When the exchange, the merge or the projection refuses its views, or the arena cannot hold
    /// them.
    pub fn return_rows(&self, f: &Forward, attended: Operand) -> Result<Operand> {
        if !self.exchanges() {
            return self.output.forward(f, attended);
        }
        let staging = f.empty(attended.dtype, attended.shape)?;
        self.exchange(f, attended, staging)?;
        let merged = self.merge(f, staging)?;
        self.output.forward(f, merged)
    }

    /// Starts returning one interval's payload: block `m` of `payload` is member `m`'s rows of this
    /// member's heads, and block `m` of the answered tensor will be this member's rows of member
    /// `m`'s heads once [`Self::receive`] took the transfer up.
    ///
    /// # Errors
    ///
    /// When the payload is not equal blocks, the arena cannot hold the landing, or the forward
    /// carries no split.
    pub fn start_return(&self, f: &Forward, payload: Operand) -> Result<(Operand, Transfer)> {
        let landed = f.empty(payload.dtype, payload.shape)?;
        self.check_blocks(payload, landed)?;
        self.require_split(f)?;
        let transfer = f.start_all_to_all(payload, landed)?;
        Ok((landed, transfer))
    }

    /// Takes up a returning interval: waits for `transfer`, sets the `landed` blocks side by side
    /// into `[count, heads · head_dim]` and projects them into `destination`, the interval's
    /// `[count, hidden]` rows of the layer's output, in place.
    ///
    /// # Errors
    ///
    /// When the merge or the projection refuses its views, or the arena cannot hold the merge.
    pub fn receive(
        &self,
        f: &Forward,
        transfer: Transfer,
        landed: Operand,
        destination: Operand,
    ) -> Result<()> {
        self.finish(f, transfer)?;
        let merged = self.merge(f, landed)?;
        self.output.launch(f, merged, destination)
    }

    /// The intervals `own` rows return in: [`RETURN_SHARES`] of them, each ending on a multiple of
    /// `unit` rows (or at the end), empty intervals dropped.
    pub fn intervals(own: u64, unit: u64) -> Vec<Interval> {
        let total: u64 = RETURN_SHARES.iter().sum();
        let unit = unit.max(1);
        let mut intervals = Vec::new();
        let (mut first, mut reached) = (0, 0);
        for share in RETURN_SHARES {
            reached += share;
            let end = (own * reached / total).next_multiple_of(unit).min(own);
            if end == first {
                continue;
            }
            intervals.push(Interval {
                first,
                count: end - first,
            });
            first = end;
        }
        intervals
    }

    /// Sends block `m` of `send` to member `m` and leaves in block `m` of `recv` what member `m`
    /// sent here.
    fn exchange(&self, f: &Forward, send: Operand, recv: Operand) -> Result<()> {
        self.check_blocks(send, recv)?;
        self.require_split(f)?;
        f.all_to_all(send, recv)
    }

    /// The blocks of `received`, one per member, set side by side in a fresh
    /// `[rows, members · local]` tensor.
    fn merge(&self, f: &Forward, received: Operand) -> Result<Operand> {
        let members = u64::from(self.split.size);
        let local = self.local_width();
        let rows = received.numel() / (members * local);
        let out = f.empty(received.dtype, Shape::new(&[rows, members * local]))?;
        let block = [members, rows, 1, local];
        let source = received.reshape(Shape::new(&block));
        let destination = Operand::with_strides(
            out.ptr,
            out.dtype,
            Shape::new(&block),
            Strides::new(&[local, members * local, local, 1]),
        );
        let strides = [local as i64, (members * local) as i64, local as i64, 1];
        Copy::new(out.dtype, Access::Vector)?.forward(f, source, destination, &strides)?;
        Ok(out)
    }

    /// Refuses two buffers that are not the group's equal blocks of one element type.
    fn check_blocks(&self, send: Operand, recv: Operand) -> Result<()> {
        let members = u64::from(self.split.size);
        if send.numel() != recv.numel()
            || !send.numel().is_multiple_of(members)
            || send.dtype != recv.dtype
        {
            return Err(Error::shape(
                "UlyssesAttention",
                format!(
                    "{} {} elements against {} {}: {members} equal blocks of one type",
                    send.numel(),
                    ops::abi::name(send.dtype),
                    recv.numel(),
                    ops::abi::name(recv.dtype)
                ),
            ));
        }
        Ok(())
    }

    /// Refuses a forward that is not split over this layer's group.
    fn require_split(&self, f: &Forward) -> Result<()> {
        let split = f.split().ok_or_else(|| {
            Error::Unsupported(
                "UlyssesAttention: a sequence-parallel exchange needs the group it runs over"
                    .to_owned(),
            )
        })?;
        if split.ranks != u64::from(self.split.size) {
            return Err(Error::shape(
                "UlyssesAttention",
                format!(
                    "the layer is built for {} members and the group has {}",
                    self.split.size, split.ranks
                ),
            ));
        }
        Ok(())
    }

    /// Channels of one head.
    pub const fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// The heads of the whole module.
    pub const fn heads(&self) -> u32 {
        self.heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A member's rows return in four intervals of shares 1:2:2:1, each ending on a whole unit, and
    /// together they are every row once.
    #[test]
    fn the_return_intervals_cover_every_row_once_on_whole_units() {
        let intervals = UlyssesAttention::<()>::intervals(600, 32);
        assert_eq!(
            intervals,
            [
                Interval {
                    first: 0,
                    count: 128
                },
                Interval {
                    first: 128,
                    count: 192
                },
                Interval {
                    first: 320,
                    count: 192
                },
                Interval {
                    first: 512,
                    count: 88
                },
            ]
        );
        let covered: u64 = intervals.iter().map(|interval| interval.count).sum();
        assert_eq!(covered, 600);
        // Rows fewer than a unit return in one interval.
        assert_eq!(
            UlyssesAttention::<()>::intervals(16, 32),
            [Interval {
                first: 0,
                count: 16
            }]
        );
    }
}
