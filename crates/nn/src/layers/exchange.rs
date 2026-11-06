//! Ulysses sequence-parallel exchange: the all-to-all that trades a token split for a head split
//! and back (specs/worker.md 4.3, `UlyssesExchange`).
//!
//! The exchange is a collective, not a kernel program: every rank sends the same number of bytes, so
//! there is nothing to compile per geometry and the epoch record fills no program for it.
//!
//! A member's buffer is `ranks` equal blocks. Going to the heads, block `m` of the send buffer holds
//! this rank's rows of member `m`'s heads, and block `m` of the receive buffer is member `m`'s rows of
//! this rank's heads: because a member's rows are one contiguous interval of the packed sequence, the
//! received blocks read in member order are the whole sequence in order, for this rank's heads.
//! Coming back the same collective moves the opposite blocks: block `m` of the send buffer is member
//! `m`'s rows of this rank's heads, and block `m` of the receive buffer is this rank's rows of member
//! `m`'s heads. The reference's `HeadExchange` (`uniserve/distributed/tokens.py`) is the same trade.

use ops::TensorView;

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

    /// Sends block `m` of `send` to member `m`, and leaves in block `m` of `recv` what member `m`
    /// sent here.
    pub fn exchange(&self, f: &Forward, send: TensorView, recv: TensorView) -> Result<()> {
        self.check_views(send, recv)?;
        self.group(f)?.all_to_all(send, recv, f.stream()?)
    }

    /// Starts [`Self::exchange`] beside the forward's stream, which goes on at once.
    pub fn start(&self, f: &Forward, send: TensorView, recv: TensorView) -> Result<crate::Transfer> {
        self.check_views(send, recv)?;
        self.group(f)?.start_all_to_all(send, recv, f.stream()?)
    }

    /// Orders the forward's stream after `transfer`: what it received is readable from here on.
    pub fn finish(&self, f: &Forward, transfer: crate::Transfer) -> Result<()> {
        self.group(f)?.finish(transfer, f.stream()?)
    }

    /// Gathers every member's `rows` into `out`, in member order: the whole sequence's rows.
    pub fn gather(&self, f: &Forward, rows: TensorView, out: TensorView) -> Result<()> {
        if out.numel() != rows.numel() * self.ranks {
            return Err(Error::shape(
                "UlyssesExchange",
                format!(
                    "{} ranks gather {} elements each into {}",
                    self.ranks,
                    rows.numel(),
                    out.numel()
                ),
            ));
        }
        self.group(f)?.all_gather(rows, out, f.stream()?)
    }

    /// The geometry one exchange needs, checked before the collective is issued.
    ///
    /// Split from [`Self::exchange`] because it needs no device: a caller's geometry is assertable
    /// on the host, and a wrong one is refused before a collective is issued rather than after a
    /// mismatched send.
    pub fn check_views(&self, send: TensorView, recv: TensorView) -> Result<()> {
        if send.numel() != recv.numel() {
            return Err(Error::shape(
                "UlyssesExchange",
                format!(
                    "the exchange sends {} elements and receives {}",
                    send.numel(),
                    recv.numel()
                ),
            ));
        }
        if !send.numel().is_multiple_of(self.ranks) {
            return Err(Error::shape(
                "UlyssesExchange",
                format!(
                    "{} elements are not {} equal blocks",
                    send.numel(),
                    self.ranks
                ),
            ));
        }
        if send.dtype != recv.dtype {
            return Err(Error::shape(
                "UlyssesExchange",
                format!("the exchange moves {} against {}", send.dtype, recv.dtype),
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

    /// The two sides hold the same bytes, as `ranks` equal blocks: a rank's rows of every member's
    /// heads against every member's rows of this rank's heads.
    #[test]
    fn an_exchange_moves_equal_blocks_between_two_equal_buffers() {
        let four = UlyssesExchange::new(4).unwrap();
        four.check_views(view(&[4, 16, 14 * 128]), view(&[64, 14 * 128]))
            .expect("four members' blocks of a rank's rows, against the whole sequence");
    }

    /// Each way the geometry can be wrong is refused by name, because a mismatched collective is a
    /// deadlock or a wrong answer rather than an error.
    #[test]
    fn a_geometry_that_cannot_exchange_is_refused() {
        let two = UlyssesExchange::new(2).unwrap();
        let sizes = two.check_views(view(&[16, 8]), view(&[16, 4]));
        assert!(format!("{}", sizes.unwrap_err()).contains("receives"));
        let blocks = UlyssesExchange::new(4)
            .unwrap()
            .check_views(view(&[6]), view(&[6]));
        assert!(format!("{}", blocks.unwrap_err()).contains("equal blocks"));
        let dtype = two.check_views(
            view(&[16, 8]),
            TensorView::new(DevicePtr::new(0x2000), DType::F16, Shape::new(&[16, 8])),
        );
        assert!(format!("{}", dtype.unwrap_err()).contains("moves"));
    }
}
