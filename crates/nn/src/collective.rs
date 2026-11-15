//! Collectives: how a partitioned forward meets the other ranks of its group.
//!
//! A module of a partitioned model computes one rank's share and calls a collective on its
//! tensors: a sum over every rank's tensor ([`all_reduce`]), a gather of every rank's tensor in
//! rank order ([`all_gather`]), an exchange of equal blocks between every pair of ranks
//! ([`all_to_all`]), or one rank's tensor on every rank ([`broadcast`]). Which communicator and
//! stream carry it is the execution context's, which a model never names (`ops::collective`).
//!
//! A rank's place has two readers. At load, a module takes the [`Split`] its family was given and
//! reads its share of each weight. In a forward, shape arithmetic over the partition reads
//! [`split`], which the context states, so a profile run follows the shapes the launching forward
//! of that rank does.
//!
//! Every tensor a collective takes is a contiguous tensor of the device backend, meta under a
//! meta context; a CPU tensor is refused, because one process holds one rank.

use tensor::Tensor;

use crate::Result;

/// One rank's place in a partition of ranks, as a module's `load` reads it.
///
/// A partition spans at least one rank and the place is one of them: [`Split::new`] refuses any
/// other pair, so a module reads `size` and `rank` without checking them again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Split {
    /// Ranks the partition spans, at least one; one is no partition.
    size: u32,
    /// This rank's position in it, below `size`.
    rank: u32,
}

impl Split {
    /// No partition: one rank holds the whole.
    pub const WHOLE: Self = Self { size: 1, rank: 0 };

    /// Rank `rank`'s place in a partition of `size` ranks.
    ///
    /// # Errors
    ///
    /// [`crate::Error::Shape`] for a partition of no rank, or a rank outside the partition.
    pub fn new(size: u32, rank: u32) -> Result<Self> {
        if size == 0 || rank >= size {
            return Err(crate::Error::shape(
                "Split",
                format!("rank {rank} is no place of a partition of {size} ranks"),
            ));
        }
        Ok(Self { size, rank })
    }

    /// Ranks the partition spans, at least one.
    pub const fn size(self) -> u32 {
        self.size
    }

    /// This rank's position in the partition, below [`Self::size`].
    pub const fn rank(self) -> u32 {
        self.rank
    }

    /// The partition that divides each place of this one into the places of `inner`, `inner`'s
    /// place varying fastest: `size · inner.size` places, this rank at
    /// `rank · inner.size + inner.rank`.
    pub const fn nest(self, inner: Self) -> Self {
        Self {
            size: self.size * inner.size,
            rank: self.rank * inner.size + inner.rank,
        }
    }
}

/// This rank's index in the forward's group and the group's member count, or `None` when the
/// forward is not partitioned or runs on CPU tensors with no context installed.
pub fn split() -> Option<(u64, u64)> {
    ops::collective::split()
}

/// Runs `operation` with the forward's collectives and [`split`] along axis `axis` of its
/// partition, and the axis addressed before again once it returns.
///
/// A module partitioned along several axes (a sequence split inside a tensor split) addresses
/// each collective's axis this way; outside any, the collectives go along the first. Which axis
/// is which is the family's statement, which execution follows when it states the axes'
/// groups. A collective started along an axis is finished along it.
///
/// # Errors
///
/// When no context is installed or it states no such axis, or what `operation` returns.
pub fn along<T>(axis: usize, operation: impl FnOnce() -> Result<T>) -> Result<T> {
    ops::collective::along(axis, operation)?
}

/// Sums every rank's `x` into every rank's `x`, in place.
///
/// `x` has the same shape and element type on every rank.
///
/// # Errors
///
/// When no context is installed or it carries no group, `x` is a strided view or a CPU tensor,
/// or the collective cannot be issued.
pub fn all_reduce(x: &Tensor) -> Result<()> {
    Ok(ops::collective::all_reduce(x)?)
}

/// Gathers every rank's `send` into `recv`, in rank order.
///
/// # Errors
///
/// As [`all_reduce`], and when `recv` does not hold the group's member count times `send`'s
/// elements of the same element type.
pub fn all_gather(send: &Tensor, recv: &Tensor) -> Result<()> {
    Ok(ops::collective::all_gather(send, recv)?)
}

/// An unwritten tensor that holds one block of `block`'s shape and element type per rank of the
/// forward's group, along the first axis: what a gather of every rank's block fills
/// ([`all_gather`], [`start_gather_rows`]).
///
/// It is a block of the context's allocator, so a model that gathers into a tensor of its own
/// names no allocation: it sets its own block and starts the gather of the rest.
///
/// # Errors
///
/// When no context is installed or it carries no group, or `block` has no axis.
pub fn sequence_like(block: &Tensor) -> Result<Tensor> {
    let Some((_, ranks)) = split() else {
        return Err(crate::Error::Unsupported(
            "a gather of every rank's block needs the group it runs over".to_owned(),
        ));
    };
    let Some((rows, rest)) = block.dims().split_first() else {
        return Err(crate::Error::shape(
            "sequence_like",
            "a block of no axis has no rank axis to gather along".to_owned(),
        ));
    };
    let dims = [&[rows * ranks as usize], rest].concat();

    Ok(ops::empty(dims, block.dtype())?)
}

/// Every rank's `send` in rank order: a new tensor of [`sequence_like`]'s shape, gathered.
///
/// # Errors
///
/// As [`sequence_like`] and [`all_gather`].
pub fn gathered(send: &Tensor) -> Result<Tensor> {
    let recv = sequence_like(send)?;

    all_gather(send, &recv)?;
    Ok(recv)
}

/// Exchanges equal blocks with every rank: block `j` of `send` goes to rank `j`, and block `j` of
/// `recv` is what rank `j` sent here.
///
/// # Errors
///
/// As [`all_reduce`], and when the tensors are not the group's member count of equal blocks of
/// one element type.
pub fn all_to_all(send: &Tensor, recv: &Tensor) -> Result<()> {
    Ok(ops::collective::all_to_all(send, recv)?)
}

/// Rank `root`'s `x` on every rank of the forward's group: `x` itself on `root`, and on every
/// other rank a new tensor of `x`'s shape and element type holding `root`'s elements. Every rank
/// passes a tensor of the same shape and element type, which only `root` reads.
///
/// # Errors
///
/// As [`all_reduce`], and when `root` is no rank of the group.
pub fn broadcast(x: &Tensor, root: u32) -> Result<Tensor> {
    Ok(ops::collective::broadcast(x, u64::from(root))?)
}

/// Starts [`all_to_all`] beside the forward, which goes on computing at once.
///
/// The tensors belong to the collective until [`finish`] is given the returned value: the caller
/// keeps both until then, and what `recv` holds is readable after that.
///
/// # Errors
///
/// As [`all_to_all`].
pub fn start_all_to_all(send: &Tensor, recv: &Tensor) -> Result<usize> {
    Ok(ops::collective::start_all_to_all(send, recv)?)
}

/// Starts delivering rows `first..first + count` of this rank's `owner_rows` rows to every rank,
/// and theirs here, in place in `gathered`.
///
/// `gathered` is the whole sequence `[ranks * owner_rows, width]`, rank by rank, and this rank's
/// rows of the interval are already in its own block. The sequence is whole once the collective
/// of each interval is finished ([`finish`]).
///
/// # Errors
///
/// As [`all_reduce`], and when `gathered` is not every rank's `owner_rows` rows or the interval
/// is not inside them.
pub fn start_gather_rows(
    gathered: &Tensor,
    owner_rows: u64,
    first: u64,
    count: u64,
) -> Result<usize> {
    Ok(ops::collective::start_gather_rows(
        gathered, owner_rows, first, count,
    )?)
}

/// Orders the forward after the started collective `started`: what it received is readable from
/// here on.
///
/// # Errors
///
/// When no context is installed or it carries no group, or `started` is not a value a started
/// collective returned.
pub fn finish(started: usize) -> Result<()> {
    Ok(ops::collective::finish(started)?)
}

/// Waits for what the forward issued so far, when it is not being captured.
///
/// A module calls this after a launch that waits on its device for the other ranks of the group
/// (an expert exchange): an eager forward that went on past such a launch could hold a lock its
/// peers need to reach the same launch. A captured forward, a meta forward and a forward on CPU
/// tensors have nothing to wait for.
///
/// # Errors
///
/// When the stream cannot be waited for.
pub fn settle() -> Result<()> {
    Ok(ops::collective::settle()?)
}

#[cfg(test)]
mod tests {
    use super::Split;

    #[test]
    fn a_split_places_its_rank_among_at_least_one() {
        assert!(Split::new(0, 0).is_err(), "a partition of no rank");
        assert!(Split::new(4, 4).is_err(), "a rank past the partition");

        let split = Split::new(4, 3).expect("a place of four ranks");
        assert_eq!((split.size(), split.rank()), (4, 3));
    }

    #[test]
    fn a_nested_split_varies_its_inner_place_fastest() {
        let outer = Split::new(2, 1).unwrap();
        let inner = Split::new(4, 3).unwrap();

        assert_eq!(outer.nest(inner), Split::new(8, 7).unwrap());
        assert_eq!(Split::WHOLE.nest(inner), inner);
    }
}
