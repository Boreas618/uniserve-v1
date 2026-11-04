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
        self.check_views(input, out)?;
        let group = self.group(f)?;
        group.with(|communicator| {
            exchange::scatter(communicator, input, out, self.ranks, f.stream).map_err(Error::from)
        })
    }

    /// Put every rank's heads back together into `out`.
    pub fn gather(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        self.check_views(out, input)?;
        let group = self.group(f)?;
        group.with(|communicator| {
            exchange::gather(communicator, input, out, self.ranks, f.stream).map_err(Error::from)
        })
    }

    /// The geometry one exchange direction needs, checked before the collective is issued.
    ///
    /// `input` and `out` are the two sides of the exchange: `[tokens, heads, head_dim]` on one side
    /// and `[tokens · ranks, head_dim / ranks]` on the other, in whichever order the direction
    /// reads them. Split from [`Self::scatter`] and [`Self::gather`] because it needs no device: a
    /// caller's geometry is assertable on the host, and a wrong one is refused before a collective
    /// is issued rather than after a mismatched send.
    pub fn check_views(&self, wide: TensorView, narrow: TensorView) -> Result<()> {
        if wide.shape.rank() != 3 {
            return Err(Error::shape(
                "UlyssesExchange",
                format!(
                    "the wide side is [tokens, heads, head_dim], not {:?}",
                    wide.shape.dims()
                ),
            ));
        }
        let head_dim = wide.shape.last();
        if !head_dim.is_multiple_of(self.ranks) {
            return Err(Error::shape(
                "UlyssesExchange",
                format!(
                    "a head of {head_dim} features does not divide across {} ranks",
                    self.ranks
                ),
            ));
        }
        if narrow.shape.rank() != 2 {
            return Err(Error::shape(
                "UlyssesExchange",
                format!(
                    "the exchanged side is [tokens · ranks, head_dim / ranks], not {:?}",
                    narrow.shape.dims()
                ),
            ));
        }
        let tokens = wide.shape.dim(0);
        if narrow.shape.dim(0) != tokens * self.ranks {
            return Err(Error::shape(
                "UlyssesExchange",
                format!(
                    "the exchanged side has {} rows, {} expected for {tokens} tokens over {} ranks",
                    narrow.shape.dim(0),
                    tokens * self.ranks,
                    self.ranks
                ),
            ));
        }
        if narrow.shape.dim(1) * self.ranks != head_dim {
            return Err(Error::shape(
                "UlyssesExchange",
                format!(
                    "the exchanged side is {} wide, a head of {head_dim} over {} ranks",
                    narrow.shape.dim(1),
                    self.ranks
                ),
            ));
        }
        if narrow.dtype != wide.dtype {
            return Err(Error::shape(
                "UlyssesExchange",
                format!("the exchange moves {} against {}", wide.dtype, narrow.dtype),
            ));
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use ops::{DType, DevicePtr, Shape};

    /// A view of `shape` at a host address: these tests never issue a collective.
    fn view(shape: &[u64]) -> TensorView {
        TensorView::new(DevicePtr::new(0x2000), DType::BF16, Shape::new(shape))
    }

    /// A group is a power of two ranks: a mistyped degree is caught before it reaches a collective.
    #[test]
    fn a_group_is_a_positive_power_of_two() {
        for ranks in [1, 2, 4, 8] {
            assert_eq!(UlyssesExchange::new(ranks).unwrap().ranks(), ranks);
        }
        assert!(UlyssesExchange::new(0).is_err());
        assert!(UlyssesExchange::new(3).is_err());
        assert!(UlyssesExchange::new(6).is_err());
        let text = match UlyssesExchange::new(12) {
            Ok(_) => panic!("twelve is not a power of two"),
            Err(error) => format!("{error}"),
        };
        assert!(text.contains("12"), "{text}");
    }

    /// `[tokens, heads, head_dim]` over `ranks` becomes `[tokens · ranks, head_dim / ranks]`: each
    /// rank keeps its slice of every head at every token of the whole sequence, so the row count
    /// grows by the rank count while a row narrows by it.
    #[test]
    fn an_exchange_trades_head_width_for_rows_and_back() {
        let two = UlyssesExchange::new(2).unwrap();
        let wide = view(&[16, 8, 128]);
        let narrow = view(&[32, 64]);
        assert!(two.check_views(wide, narrow).is_ok());
        // Four ranks from the same geometry: twice the rows, a quarter of the width.
        let four = UlyssesExchange::new(4).unwrap();
        assert!(four.check_views(wide, view(&[64, 32])).is_ok());
        // The same views are still one exchange's two sides, so `gather` is the same check with the
        // arguments read the other way round.
        assert!(two.check_views(wide, narrow).is_ok());
    }

    /// Each way the geometry can be wrong is refused by name, because a mismatched collective is a
    /// deadlock or a wrong answer rather than an error.
    #[test]
    fn a_geometry_that_cannot_exchange_is_refused() {
        let two = UlyssesExchange::new(2).unwrap();
        let wide = view(&[16, 8, 128]);

        let rows = two.check_views(wide, view(&[16, 64]));
        assert!(
            format!("{}", rows.unwrap_err()).contains("32 expected"),
            "rows are tokens · ranks"
        );

        let width = two.check_views(wide, view(&[32, 128]));
        assert!(
            format!("{}", width.unwrap_err()).contains("a head of 128"),
            "width is head_dim / ranks"
        );

        let rank = two.check_views(wide, view(&[32]));
        assert!(
            format!("{}", rank.unwrap_err()).contains("exchanged side"),
            "a rank-1 side"
        );

        let heads = two.check_views(view(&[16]), view(&[32, 64]));
        assert!(
            format!("{}", heads.unwrap_err()).contains("wide side"),
            "a rank-1 wide side"
        );

        // A head of 96 features over 3 ranks divides, but 3 ranks are not a group.
        assert!(UlyssesExchange::new(3).is_err());
        // Over four ranks an odd head width does not divide.
        let four = UlyssesExchange::new(4).unwrap();
        let odd = four.check_views(view(&[16, 8, 100]), view(&[64, 25]));
        assert!(format!("{}", odd.unwrap_err()).contains("does not divide"));

        let dtype = two.check_views(
            wide,
            TensorView::new(DevicePtr::new(0x2000), DType::F32, Shape::new(&[32, 64])),
        );
        assert!(
            format!("{}", dtype.unwrap_err()).contains("moves"),
            "both sides are one type"
        );
    }
}
