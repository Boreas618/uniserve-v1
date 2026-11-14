//! Ulysses sequence parallelism around any attention: trade a row split for a head split and
//! back.
//!
//! A member of a sequence-parallel group holds a contiguous slice of the packed rows and every
//! head. Its attention needs the opposite: every row of the sequence for its own share of the
//! heads. Two ways reach it, and the deployment states which ([`Projection`]):
//!
//! - **Exchange**: each member projects its own rows for every member's heads, one product per
//!   member, and one all-to-all delivers every member's block;
//! - **Gather**: each member gathers every member's normalized rows once and projects its own
//!   heads with a head-partitioned weight.
//!
//! After either, the attention reads `[rows of the whole sequence, this member's heads · head
//! width]`. Coming back, one all-to-all sends each member its rows of this member's heads, and
//! the received blocks are set side by side into `[this member's rows, every head]` before the
//! output projection. The return may be whole ([`UlyssesAttention::return_rows`]) or by row
//! intervals ([`UlyssesAttention::intervals`], [`UlyssesAttention::start_exchange`],
//! [`UlyssesAttention::receive`]), so an interval's output projection runs while the later
//! intervals travel.
//!
//! Inside a tensor split the module holds its tensor rank's heads alone: each member's share is
//! a share of those, and the output projection reduces its partial sums over the tensor group,
//! the axis the caller's forward addresses.
//!
//! The module holds the projections around the attention and the split, never a communicator:
//! its exchanges and gathers go along the axis its config names of the forward's partition
//! ([`crate::collective::along`]), and the attention between the projections is the model's own.

use std::ops::Range;

use tensor::{DType, Tensor};

use crate::collective::{self, Split};
use crate::error::{Error, Result};
use crate::family::Projection;
use crate::modules::linear::{Linear, LinearConfig, Parallel};
use crate::var_builder::VarBuilder;

/// The intervals a member's rows return in, as shares of the rows.
///
/// The first interval's transfer is the part of the return nothing hides, and the last
/// interval's is the part of the next layer's input nothing hides, so the two ends are the
/// small ones.
const RETURN_SHARES: [usize; 4] = [1, 2, 2, 1];

/// The widths and names of a [`UlyssesAttention`], and this member's place in its group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UlyssesAttentionConfig {
    /// The width of a row.
    pub hidden: u32,
    /// The heads of the whole module, before any split.
    pub heads: u32,
    /// Channels of one head.
    pub head_dim: u32,
    /// The `[heads · head_dim, hidden]` input projections' module names, in the order the
    /// attention reads them.
    pub inputs: &'static [&'static str],
    /// The `[hidden, heads · head_dim]` output projection's module name.
    pub output: &'static str,
    /// The sequence-parallel group and this member's rank in it.
    pub split: Split,
    /// How a member reaches its heads of every row.
    pub projection: Projection,
    /// The tensor split of the heads: the module holds its tensor rank's share of them, and its
    /// output projection sums its share of the contraction over the tensor group.
    pub tensor: Split,
    /// The axis of the forward's partition its exchanges and gathers go along
    /// ([`crate::collective::along`]).
    pub axis: usize,
}

/// Sequence-parallel attention: the head-split projections around a model's attention.
pub struct UlyssesAttention {
    /// Each input projection as this member reaches its heads. One member: the whole
    /// projection. Gather: this member's head rows of it. Exchange: one projection per member,
    /// in member order, each holding that member's head rows.
    inputs: Vec<Vec<Linear>>,
    output: Linear,
    config: UlyssesAttentionConfig,
}

impl UlyssesAttention {
    /// The projections the module at `vb` states, for the member and the group the config
    /// states.
    ///
    /// # Errors
    ///
    /// A group whose size does not divide the heads, or a projection the loader refuses.
    pub fn load(config: &UlyssesAttentionConfig, vb: VarBuilder<'_, '_>) -> Result<Self> {
        let (split, tensor) = (config.split, config.tensor);
        let shares = split.size.saturating_mul(tensor.size);
        if shares == 0 || !config.heads.is_multiple_of(shares) || tensor.rank >= tensor.size {
            return Err(Error::shape(
                "UlyssesAttention",
                format!(
                    "{} heads over a group of {} inside a tensor split of {}",
                    config.heads, split.size, tensor.size
                ),
            ));
        }
        let inner_width = config.heads * config.head_dim;

        // One member's head rows of the input projection `name`: its share of its tensor rank's
        // heads, which lie together.
        let member = |name: &str, rank: u32| -> Result<Linear> {
            let parallel = if shares == 1 {
                Parallel::None
            } else {
                Parallel::Column(Split {
                    size: shares,
                    rank: tensor.rank * split.size + rank,
                })
            };

            Linear::load(
                &LinearConfig {
                    inputs: config.hidden,
                    outputs: inner_width,
                    bias: false,
                    parallel,
                },
                vb.pp(name),
            )
        };

        let inputs = config
            .inputs
            .iter()
            .map(|name| match (split.size, config.projection) {
                (1, _) => Ok(vec![member(name, 0)?]),
                (_, Projection::Gather) => Ok(vec![member(name, split.rank)?]),
                (_, Projection::Exchange) => (0..split.size)
                    .map(|rank| member(name, rank))
                    .collect::<Result<Vec<_>>>(),
            })
            .collect::<Result<Vec<_>>>()?;
        let output = Linear::load(
            &LinearConfig {
                inputs: inner_width,
                outputs: config.hidden,
                bias: false,
                parallel: if tensor.size == 1 {
                    Parallel::None
                } else {
                    Parallel::Row(tensor)
                },
            },
            vb.pp(config.output),
        )?;

        Ok(Self {
            inputs,
            output,
            config: *config,
        })
    }

    /// Whether the rows cross members: a group of more than one.
    pub const fn exchanges(&self) -> bool {
        self.config.split.size > 1
    }

    /// How this member reaches its heads.
    pub const fn projection(&self) -> Projection {
        self.config.projection
    }

    /// This member's place in the group.
    pub const fn split(&self) -> Split {
        self.config.split
    }

    /// The heads of the whole module.
    pub const fn heads(&self) -> u32 {
        self.config.heads
    }

    /// The heads this member attends: its share of its tensor rank's.
    pub const fn local_heads(&self) -> u32 {
        self.config.heads / (self.config.split.size * self.config.tensor.size)
    }

    /// Channels of one head.
    pub const fn head_dim(&self) -> u32 {
        self.config.head_dim
    }

    /// The element type of the projections' weights: the stack's.
    pub fn dtype(&self) -> DType {
        self.output.weight().dtype()
    }

    /// The width of this member's heads of one projection: `local heads · head_dim`.
    fn local_width(&self) -> usize {
        self.local_heads() as usize * self.config.head_dim as usize
    }

    /// The members of the group.
    fn members(&self) -> usize {
        self.config.split.size as usize
    }

    /// The rows the projections read: this member's own `normalized` rows, or the whole
    /// sequence's, `[members · rows, hidden]`, gathered when the deployment gathers.
    ///
    /// # Errors
    ///
    /// When the gather refuses its tensors or the forward has no group.
    pub fn rows(&self, normalized: &Tensor) -> Result<Tensor> {
        if !(self.exchanges() && self.config.projection == Projection::Gather) {
            return Ok(normalized.clone());
        }
        self.require_split()?;

        let &[rows, hidden] = normalized.dims() else {
            return Err(Error::shape(
                "UlyssesAttention",
                format!(
                    "normalized rows are [rows, hidden], not {:?}",
                    normalized.dims()
                ),
            ));
        };
        let gathered = ops::empty((rows * self.members(), hidden), normalized.dtype())?;

        collective::along(self.config.axis, || {
            collective::all_gather(normalized, &gathered)
        })?;
        Ok(gathered)
    }

    /// Input projection `index`.
    fn input(&self, index: usize) -> Result<&[Linear]> {
        self.inputs.get(index).map(Vec::as_slice).ok_or_else(|| {
            Error::shape(
                "UlyssesAttention",
                format!("input {index} of {} projections", self.inputs.len()),
            )
        })
    }

    /// Projection `index` of `rows` for every member's heads: one member's `[rows, local]`
    /// block after another, the send buffer of an exchange
    /// ([`Self::start_exchange`]).
    ///
    /// # Errors
    ///
    /// When a projection refuses the rows.
    pub fn member_blocks(&self, index: usize, rows: &Tensor) -> Result<Tensor> {
        let blocks = self
            .input(index)?
            .iter()
            .map(|projection| projection.forward(rows))
            .collect::<Result<Vec<Tensor>>>()?;

        Ok(Tensor::cat(&blocks, 0)?)
    }

    /// Projection `index` of `rows` ([`Self::rows`]) as this member's heads of the whole
    /// sequence: `[sequence rows, local heads · head_dim]`.
    ///
    /// # Errors
    ///
    /// When a projection or the exchange refuses its tensors.
    pub fn project(&self, index: usize, rows: &Tensor) -> Result<Tensor> {
        match self.input(index)? {
            [only] => only.forward(rows),
            _ => self.exchange(&self.member_blocks(index, rows)?),
        }
    }

    /// Starts exchanging the equal blocks of `send` beside the forward, which goes on computing:
    /// block `m` goes to member `m`, and block `m` of the returned tensor is what member `m`
    /// sent here. Returns that tensor with the value [`Self::finish`] takes.
    ///
    /// `send` is a projection's blocks ([`Self::member_blocks`]) or a returning interval's
    /// payload. It belongs to the exchange until [`Self::finish`] took it up: the caller keeps
    /// it until then, and what the returned tensor holds is readable after that.
    ///
    /// # Errors
    ///
    /// When `send` is not the group's equal blocks, or the forward has no group.
    pub fn start_exchange(&self, send: &Tensor) -> Result<(Tensor, usize)> {
        self.check_blocks(send)?;
        self.require_split()?;

        let received = ops::empty(send.shape(), send.dtype())?;
        let started = collective::along(self.config.axis, || {
            collective::start_all_to_all(send, &received)
        })?;
        Ok((received, started))
    }

    /// Orders the forward after the started collective `started`: what it received is readable
    /// from here on.
    ///
    /// # Errors
    ///
    /// When the forward has no group.
    pub fn finish(&self, started: usize) -> Result<()> {
        self.require_split()?;
        collective::along(self.config.axis, || collective::finish(started))
    }

    /// Starts delivering rows `rows` of this member's `own` rows to every member, and theirs
    /// here, in place in the whole sequence `gathered`, whose own block already holds this
    /// member's rows. Returns the value [`Self::finish`] takes.
    ///
    /// # Errors
    ///
    /// When the forward has no group.
    pub fn start_gather(&self, gathered: &Tensor, own: usize, rows: Range<usize>) -> Result<usize> {
        self.require_split()?;

        collective::along(self.config.axis, || {
            collective::start_gather_rows(
                gathered,
                own as u64,
                rows.start as u64,
                rows.len() as u64,
            )
        })
    }

    /// `attended`, `[sequence rows, local heads · head_dim]`, returned to each member's rows,
    /// and this member's rows of every head projected: `[own rows, hidden]`.
    ///
    /// With one member the attended rows are this member's and only the projection runs.
    ///
    /// # Errors
    ///
    /// When the exchange or the projection refuses its tensors.
    pub fn return_rows(&self, attended: &Tensor) -> Result<Tensor> {
        if !self.exchanges() {
            return self.output.forward(attended);
        }

        let received = self.exchange(attended)?;
        self.output.forward(&self.merge(&received)?)
    }

    /// Takes up a returning interval: orders the forward after `started`, sets the blocks of
    /// `received` side by side into `[count, heads · head_dim]` and projects them: the
    /// interval's `[count, hidden]` rows of the layer's output.
    ///
    /// # Errors
    ///
    /// When the projection refuses the rows or the forward has no group.
    pub fn receive(&self, started: usize, received: &Tensor) -> Result<Tensor> {
        self.finish(started)?;
        self.output.forward(&self.merge(received)?)
    }

    /// The intervals `own` rows return in: [`RETURN_SHARES`] of them, each ending on a multiple
    /// of `unit` rows (or at the end), empty intervals dropped.
    pub fn intervals(own: usize, unit: usize) -> Vec<Range<usize>> {
        let total: usize = RETURN_SHARES.iter().sum();
        let unit = unit.max(1);

        let mut intervals = Vec::new();
        let (mut first, mut reached) = (0, 0);
        for share in RETURN_SHARES {
            reached += share;
            let end = (own * reached / total).next_multiple_of(unit).min(own);
            if end == first {
                continue;
            }

            intervals.push(first..end);
            first = end;
        }
        intervals
    }

    /// Sends block `m` of `send` to member `m` and returns the tensor whose block `m` is what
    /// member `m` sent here.
    fn exchange(&self, send: &Tensor) -> Result<Tensor> {
        self.check_blocks(send)?;
        self.require_split()?;

        let received = ops::empty(send.shape(), send.dtype())?;
        collective::along(self.config.axis, || collective::all_to_all(send, &received))?;
        Ok(received)
    }

    /// The blocks of `received`, one per member, set side by side: `[rows, members · local]`.
    fn merge(&self, received: &Tensor) -> Result<Tensor> {
        let (members, local) = (self.members(), self.local_width());
        let rows = received.elem_count() / (members * local);

        Ok(received
            .reshape((members, rows, local))?
            .transpose(0, 1)?
            .reshape((rows, members * local))?)
    }

    /// Refuses a tensor that is not the group's equal blocks.
    fn check_blocks(&self, send: &Tensor) -> Result<()> {
        if !send.elem_count().is_multiple_of(self.members()) {
            return Err(Error::shape(
                "UlyssesAttention",
                format!(
                    "{} elements are not {} equal blocks",
                    send.elem_count(),
                    self.members()
                ),
            ));
        }
        Ok(())
    }

    /// Refuses a forward that is not split over this module's group along its axis.
    fn require_split(&self) -> Result<()> {
        let Some((_, ranks)) = collective::along(self.config.axis, || Ok(collective::split()))?
        else {
            return Err(Error::Unsupported(
                "UlyssesAttention: a sequence-parallel exchange needs the group it runs over"
                    .to_owned(),
            ));
        };

        if ranks != u64::from(self.config.split.size) {
            return Err(Error::shape(
                "UlyssesAttention",
                format!(
                    "the module is loaded for {} members and the group has {ranks}",
                    self.config.split.size
                ),
            ));
        }
        Ok(())
    }
}

/// The rows of a member's share of a sequence of `tokens` rows split over `members`: the
/// sequence padded to a whole number of rows per member, `⌈tokens / members⌉`.
pub fn share_rows(tokens: usize, members: u32) -> usize {
    tokens.div_ceil(members.max(1) as usize)
}

/// Member `split.rank`'s share of the sequence `rows` (`[tokens, width]`): rows `rank·r ..
/// (rank + 1)·r` of the sequence padded with zero rows to `size·r`, where `r` is
/// [`share_rows`]. A share past the sequence's last row is zero rows alone.
///
/// The members' shares, in member order, are the padded sequence: what [`gather`] restores.
///
/// # Errors
///
/// When `rows` is not a matrix or a tensor operation refuses it.
pub fn share(rows: &Tensor, split: Split) -> Result<Tensor> {
    let &[tokens, width] = rows.dims() else {
        return Err(Error::shape(
            "ulysses::share",
            format!("a sequence is [rows, width], not {:?}", rows.dims()),
        ));
    };
    let held = share_rows(tokens, split.size);
    let first = split.rank as usize * held;

    // The sequence's own rows of the share, then the zero rows that pad it.
    let own = tokens.saturating_sub(first).min(held);
    let padding = held - own;
    if padding == 0 {
        return Ok(rows.narrow(0, first, held)?.contiguous()?);
    }

    let zeros = Tensor::zeros((padding, width), rows.dtype(), rows.device())?;
    if own == 0 {
        return Ok(zeros);
    }
    Ok(Tensor::cat(&[&rows.narrow(0, first, own)?, &zeros], 0)?)
}

/// The sequence of `tokens` rows whose share this member holds in `share` ([`share`]): every
/// member's share gathered in member order over the forward's group, the padding rows left out.
/// Every member holds the same result.
///
/// # Errors
///
/// When the gather refuses its tensors, the forward has no group of `split`'s members, or the
/// shares do not hold `tokens` rows.
pub fn gather(share: &Tensor, split: Split, tokens: usize) -> Result<Tensor> {
    if split.size <= 1 {
        return Ok(share.clone());
    }
    let members = collective::split().map(|(_, members)| members);
    if members != Some(u64::from(split.size)) {
        return Err(Error::shape(
            "ulysses::gather",
            format!(
                "a sequence split over {} members gathers in a forward whose group is {members:?}",
                split.size
            ),
        ));
    }

    let gathered = collective::gathered(&share.contiguous()?)?;
    if gathered.dims().first().is_none_or(|rows| *rows < tokens) {
        return Err(Error::shape(
            "ulysses::gather",
            format!(
                "shares of {:?} hold fewer than the sequence's {tokens} rows",
                share.dims()
            ),
        ));
    }
    Ok(gathered.narrow(0, 0, tokens)?)
}

#[cfg(test)]
mod tests {
    use tensor::{DType, Device, Tensor};

    use super::{Split, UlyssesAttention, share, share_rows};

    /// A sequence of five rows over two members: three rows a share, the second share holding
    /// the sequence's last two rows and one zero row; over eight members, the shares past the
    /// sequence hold zero rows alone.
    #[test]
    fn a_share_is_its_members_rows_of_the_sequence_padded_with_zero_rows() {
        let values: Vec<f32> = (1..=10).map(|value| value as f32).collect();
        let rows = Tensor::from_vec(values, (5, 2), &Device::Cpu).unwrap();
        let rows_of = |split: Split| -> Vec<Vec<f32>> {
            share(&rows, split)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec2()
                .unwrap()
        };

        assert_eq!(share_rows(5, 2), 3);
        assert_eq!(
            rows_of(Split { size: 2, rank: 0 }),
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        );
        assert_eq!(
            rows_of(Split { size: 2, rank: 1 }),
            [[7.0, 8.0], [9.0, 10.0], [0.0, 0.0]]
        );
        assert_eq!(rows_of(Split { size: 8, rank: 6 }), [[0.0, 0.0]]);
        assert_eq!(rows_of(Split::WHOLE).len(), 5);
    }

    /// A member's rows return in four intervals of shares 1:2:2:1, each ending on a whole unit,
    /// and together they are every row once.
    #[test]
    fn the_return_intervals_cover_every_row_once_on_whole_units() {
        let intervals = UlyssesAttention::intervals(600, 32);
        assert_eq!(intervals, [0..128, 128..320, 320..512, 512..600]);

        // Rows fewer than a unit return in one interval.
        let short = UlyssesAttention::intervals(16, 32);
        assert_eq!((short.len(), short.first()), (1, Some(&(0..16))));
    }
}
