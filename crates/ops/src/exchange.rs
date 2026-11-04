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

use crate::platform::{self, Communicator, DevicePtr, Stream};
use crate::tensor::{DType, Shape, TensorView};
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
    communicator: &Communicator,
    input: TensorView,
    output: TensorView,
    ranks: u64,
    stream: &Stream,
) -> Result<()> {
    let counts = checked(input, output, ranks)?;
    // SAFETY: both views address the counts checked above of `input.dtype`, and both stay alive
    // until the collective completes (the caller owns them).
    unsafe {
        platform::all_to_all(
            communicator,
            input.ptr,
            output.ptr,
            &counts,
            input.dtype.as_i64() as u32,
            stream,
        )
    }
    .map_err(Error::Platform)
}

/// Put the heads of every rank back together.
pub fn gather(
    communicator: &Communicator,
    input: TensorView,
    output: TensorView,
    ranks: u64,
    stream: &Stream,
) -> Result<()> {
    let counts = checked(output, input, ranks)?;
    // SAFETY: as in `scatter`, with the roles of the two views exchanged.
    unsafe {
        platform::all_to_all(
            communicator,
            input.ptr,
            output.ptr,
            &counts,
            input.dtype.as_i64() as u32,
            stream,
        )
    }
    .map_err(Error::Platform)
}

/// Check one direction's shapes and give the per-rank element counts.
fn checked(large: TensorView, small: TensorView, ranks: u64) -> Result<Vec<u64>> {
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
    let per_rank = small.numel();
    Ok(vec![per_rank; ranks as usize])
}

/// A view of `shape` elements of `dtype` at `pointer`, for the layer's buffers.
pub fn view(pointer: DevicePtr, dtype: DType, shape: Shape) -> TensorView {
    TensorView::new(pointer, dtype, shape)
}
