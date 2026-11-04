//! Sequence-parallel exchange: the Ulysses all-to-all.
//!
//! Ulysses splits a sequence across ranks by token and splits each head across ranks by dimension,
//! and one all-to-all per attention layer turns the first layout into the second and another turns
//! it back (specs/worker.md 4.3, `UlyssesExchange`). Both directions are NCCL collectives, not
//! kernel programs: every rank sends and receives the same number of bytes, so there is nothing to
//! compile per geometry and no reason for a package.
//!
//! The exchange is a mathematical operation of the model, so its shape contract lives here next to
//! the layer that uses it:
//!
//! ```text
//! round trip   input  [tokens, heads, head_dim]
//!              to     [tokens · heads / ranks, head_dim / ranks]   per rank, heads split by rank
//!              back   [tokens, heads, head_dim]
//! ```
//!
//! `tokens` must divide by the rank count and `head_dim` by the rank count. The operation is exact:
//! both directions only move bytes, and two implementations agree bit for bit, so the layer's
//! shadow comparison is an equality.

use crate::platform::{self, Communicator, Stream};
use crate::tensor::TensorView;
use crate::{Error, Result};

/// The operator name the runtime records for this exchange.
///
/// It is not a kernel program's name: an exchange has one implementation, which the platform's
/// communicator provides, and the epoch record names no program for it.
pub const NAME: &str = "ulysses_exchange";

/// Split the heads of a token-major tensor across `ranks` ranks.
///
/// Each rank ends with the heads of its slice at every token of the whole sequence: the input is
/// `[tokens, heads, head_dim]` and the output `[tokens · ranks', head_dim / ranks]` where `ranks'`
/// is the rank count, i.e. one row per (token, head-block) pair.
pub fn scatter(
    communicator: &mut Communicator,
    input: TensorView,
    output: TensorView,
    ranks: u64,
    stream: &Stream,
) -> Result<()> {
    let count = checked(input, output, ranks)?;
    let dtype = count.1;
    // Both views address `count.0` elements of `dtype` for each of `ranks` ranks, and both stay
    // alive until the collective completes: the caller owns them.
    let send = platform::DeviceRange::new(
        platform::RangePointer::from_address(input.ptr.address()),
        input.byte_len()? as usize,
    );
    let receive = platform::DeviceRange::new(
        platform::RangePointer::from_address(output.ptr.address()),
        output.byte_len()? as usize,
    );
    platform::all_to_all(communicator, send, receive, count.0, dtype, stream)
}

/// Put the heads of every rank back together.
pub fn gather(
    communicator: &mut Communicator,
    input: TensorView,
    output: TensorView,
    ranks: u64,
    stream: &Stream,
) -> Result<()> {
    let count = checked(output, input, ranks)?;
    let dtype = count.1;
    // As in `scatter`, with the roles of the two views exchanged.
    let send = platform::DeviceRange::new(
        platform::RangePointer::from_address(input.ptr.address()),
        input.byte_len()? as usize,
    );
    let receive = platform::DeviceRange::new(
        platform::RangePointer::from_address(output.ptr.address()),
        output.byte_len()? as usize,
    );
    platform::all_to_all(communicator, send, receive, count.0, dtype, stream)
}

/// Check one direction's shapes and give the elements each rank sends and the collective's type.
fn checked(
    large: TensorView,
    small: TensorView,
    ranks: u64,
) -> Result<(usize, platform::CollectiveType)> {
    if ranks == 0 {
        return Err(Error::argument(NAME, "an exchange has no ranks"));
    }
    small.require_rank(NAME, 2)?;
    let head_dim = large.shape.last();
    if !head_dim.is_multiple_of(ranks) {
        return Err(Error::argument(
            NAME,
            format!("head_dim {head_dim} does not divide across {ranks} ranks"),
        ));
    }
    let tokens = large.rows() * large.shape.last() / (head_dim.max(1) * ranks);
    let expected = tokens * ranks;
    if small.shape.dim(0) != expected {
        return Err(Error::argument(
            NAME,
            format!(
                "the exchanged view has {} rows, {expected} expected for {tokens} tokens over {ranks} ranks",
                small.shape.dim(0)
            ),
        ));
    }
    if small.shape.dim(1) * ranks != head_dim {
        return Err(Error::argument(
            NAME,
            format!(
                "the exchanged view's width {} is not head_dim {head_dim} over {ranks} ranks",
                small.shape.dim(1)
            ),
        ));
    }
    if small.dtype != large.dtype {
        return Err(Error::argument(
            NAME,
            format!("the exchange moves {} against {}", large.dtype, small.dtype),
        ));
    }
    let dtype = platform::collective_type(small.dtype).ok_or_else(|| {
        Error::unsupported(
            NAME,
            format!("a {} exchange has no collective element type", small.dtype),
        )
    })?;
    let per_rank = usize::try_from(small.numel())
        .map_err(|_| Error::argument(NAME, "a per-rank count above usize::MAX"))?;
    Ok((per_rank, dtype))
}
