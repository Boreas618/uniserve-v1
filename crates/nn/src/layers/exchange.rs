//! Ulysses sequence-parallel exchange: the all-to-all that turns a token split into a head split and
//! back (specs/worker.md 4.3, `UlyssesExchange`).
//!
//! The exchange is a collective, not a kernel program: every rank sends the same number of bytes, so
//! there is nothing to compile per geometry and the epoch record fills no program for it. The layer
//! holds the group's rank count, which is the model's mathematical partition (specs/worker.md 3,
//! `Partition`), and calls the operator on the forward's communicator: `[tokens, heads, head_dim]`
//! becomes `[tokens × ranks, head_dim / ranks]` per rank after [`UlyssesExchange::scatter`] and back
//! after [`UlyssesExchange::gather`].

use ops::TensorView;
use ops::exchange;

use crate::error::{Error, Result};
use crate::forward::Forward;

/// The Ulysses exchange of one sequence-parallel group.
pub struct UlyssesExchange {
    ranks: u64,
}

impl UlyssesExchange {
    /// The exchange of a group of `ranks` ranks.
    pub fn new(ranks: u64) -> Result<Self> {
        if ranks == 0 {
            return Err(Error::shape("UlyssesExchange", "a group has no rank"));
        }
        if !ranks.is_power_of_two() {
            // Not a requirement of the collective, but every partition here is a power of two and a
            // mistyped degree is caught before it reaches a launch.
            return Err(Error::shape(
                "UlyssesExchange",
                format!("{ranks} ranks are not a power of two"),
            ));
        }
        Ok(Self { ranks })
    }

    /// How many ranks the group has.
    pub const fn ranks(&self) -> u64 {
        self.ranks
    }

    /// Split the heads of `input` across the ranks into `out`.
    pub fn scatter(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        let group = self.group(f)?;
        group.with(|communicator| {
            exchange::scatter(communicator, input, out, self.ranks, f.stream).map_err(Error::from)
        })
    }

    /// Put every rank's heads back together into `out`.
    pub fn gather(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        let group = self.group(f)?;
        group.with(|communicator| {
            exchange::gather(communicator, input, out, self.ranks, f.stream).map_err(Error::from)
        })
    }

    fn group<'a>(&self, f: &Forward<'a>) -> Result<&'a crate::Group> {
        let group = f.group.ok_or_else(|| {
            Error::Unsupported(
                "UlyssesExchange: a sequence-parallel exchange needs the group it runs over"
                    .to_owned(),
            )
        })?;
        if group.ranks() != self.ranks {
            return Err(Error::shape(
                "UlyssesExchange",
                format!(
                    "the layer is built for {} ranks and the group has {}",
                    self.ranks,
                    group.ranks()
                ),
            ));
        }
        Ok(group)
    }
}
