//! Collectives: how a partitioned forward meets the other ranks of its group.
//!
//! A module of a partitioned model computes one rank's share and calls a collective on its
//! tensors: a sum over every rank's tensor ([`all_reduce`]), a gather of every rank's tensor in
//! rank order ([`all_gather`]), or an exchange of equal blocks between every pair of ranks
//! ([`all_to_all`]). Which communicator and stream carry it is the execution context's, which a
//! model never names (`backend::collective`).
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

/// One rank's place in a partition of `size` ranks, as a module's `load` reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Split {
    /// Ranks the partition spans; one is no partition.
    pub size: u32,
    /// This rank's position in it, below `size`.
    pub rank: u32,
}

impl Split {
    /// No partition: one rank holds the whole.
    pub const WHOLE: Self = Self { size: 1, rank: 0 };
}

/// This rank's index in the forward's group and the group's member count, or `None` when the
/// forward is not partitioned or runs on CPU tensors with no context installed.
pub fn split() -> Option<(u64, u64)> {
    backend::collective::split()
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
    Ok(backend::collective::all_reduce(x)?)
}

/// Gathers every rank's `send` into `recv`, in rank order.
///
/// # Errors
///
/// As [`all_reduce`], and when `recv` does not hold the group's member count times `send`'s
/// elements of the same element type.
pub fn all_gather(send: &Tensor, recv: &Tensor) -> Result<()> {
    Ok(backend::collective::all_gather(send, recv)?)
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

    Ok(backend::empty(dims, block.dtype())?)
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
    Ok(backend::collective::all_to_all(send, recv)?)
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
    Ok(backend::collective::start_all_to_all(send, recv)?)
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
    Ok(backend::collective::start_gather_rows(
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
    Ok(backend::collective::finish(started)?)
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
    Ok(backend::collective::settle()?)
}
