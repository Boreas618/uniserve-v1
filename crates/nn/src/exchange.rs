//! What an expert-parallel forward borrows from the package's exchange transport.
//!
//! A tensor-parallel deployment replicates the router and keeps each rank's experts, so a token
//! routed to another rank has to travel there. The package owns the transport — its buffers, its
//! ordering, its state — because the workspace layout is the program's, and dispatches and combines
//! through it (specs/worker.md 9, "Expert Parallel"; `ops::moe::exchange`).
//!
//! A model layer cannot build that transport: it may not create runtime objects, and the executor
//! is the one that calls the package's creation entry. What travels through [`Forward`] is
//! therefore only the views the transport wrote and the layer reads, lent by whoever owns them.
//! The executor builds one [`ExchangeViews`] per device and group after loading the model, and
//! lends it on every forward of a block whose experts are split.
//!
//! [`Forward`]: crate::Forward

use ops::TensorView;
use ops::moe::exchange::ExchangeTransport;

/// The received rows of one expert-parallel exchange, and the transport that moves them.
///
/// The four views are the transport's own device buffers: the layer reads them to sort and compute
/// its experts and writes the dispatched tiles back, and never allocates or frees them. `hidden` and
/// `top` describe one row of the received layout; `ranks x capacity` rows exist, of which `counts[r]`
/// hold the rows rank `r` sent.
pub struct ExchangeViews {
    /// The package's transport for this device and group.
    ///
    /// Opaque: the layer only hands it back to the exchange operator. Its lifetime belongs to the
    /// executor, which keeps it alive for every forward that borrows it.
    pub transport: ExchangeTransport,
    /// Ranks in the expert-parallel group; the leading axis of the received layout.
    pub ranks: u32,
    /// Rows one rank may exchange: the MoE plan's capacity, so the rows fit without a host read.
    pub capacity: u32,
    /// `[ranks * capacity, hidden]` in the activation element type: the dispatched activations, and
    /// the buffer the local experts' outputs are written back to.
    pub states: TensorView,
    /// `[ranks * capacity, top]` i32: the expert each received row is routed to.
    pub ids: TensorView,
    /// `[ranks * capacity, top]` f32: the router weight of each received row.
    pub weights: TensorView,
    /// `[ranks]` i32: the rows rank `r` sent.
    pub counts: TensorView,
}

impl ExchangeViews {
    /// Elements per received row.
    pub fn hidden(&self) -> u64 {
        self.states.shape.last()
    }

    /// Experts one token may route to, as the received layout carries them.
    pub fn top(&self) -> u64 {
        self.ids.shape.last()
    }

    /// Check that the views have the extents the exchange laid them out with.
    ///
    /// The executor builds this value, so a wrong extent is a wiring error; checking it here keeps
    /// it out of a launch that would read unrelated memory.
    pub fn require_extents(&self, name: &'static str) -> crate::Result<()> {
        let rows = u64::from(self.ranks) * u64::from(self.capacity);
        let hidden = self.hidden();
        let top = self.top();
        if top == 0 || hidden == 0 {
            return Err(crate::Error::shape(
                name,
                "an exchange of zero hidden elements or zero experts computes nothing".to_owned(),
            ));
        }
        for (view, expected, label) in [
            (&self.states, &[rows, hidden][..], "states"),
            (&self.ids, &[rows, top][..], "ids"),
            (&self.weights, &[rows, top][..], "weights"),
            (&self.counts, &[u64::from(self.ranks)][..], "counts"),
        ] {
            if view.shape.dims() != expected {
                return Err(crate::Error::shape(
                    name,
                    format!(
                        "the exchange's {label} is {:?}, {expected:?} expected",
                        view.shape.dims()
                    ),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, DevicePtr, Shape};

    fn views(
        ranks: u32,
        capacity: u32,
        hidden: u64,
        top: u64,
    ) -> (TensorView, TensorView, TensorView, TensorView) {
        let rows = u64::from(ranks) * u64::from(capacity);
        let base = DevicePtr::new(0x1_0000);
        (
            TensorView::new(base, DType::BF16, Shape::new(&[rows, hidden])),
            TensorView::new(base, DType::I32, Shape::new(&[rows, top])),
            TensorView::new(base, DType::F32, Shape::new(&[rows, top])),
            TensorView::new(base, DType::I32, Shape::new(&[u64::from(ranks)])),
        )
    }

    fn exchange(ranks: u32, capacity: u32, hidden: u64, top: u64) -> ExchangeViews {
        let (states, ids, weights, counts) = views(ranks, capacity, hidden, top);
        ExchangeViews {
            transport: {
                // SAFETY: the transport is opaque and never dereferenced by these tests; the null
                // handle only stands in for a value the executor would build.
                unsafe { ExchangeTransport::from_handle(core::ptr::null_mut()) }
            },
            ranks,
            capacity,
            states,
            ids,
            weights,
            counts,
        }
    }

    #[test]
    fn an_exchange_of_the_planned_extents_passes() {
        let exchange = exchange(2, 64, 2048, 8);
        assert_eq!(exchange.hidden(), 2048);
        assert_eq!(exchange.top(), 8);
        assert!(exchange.require_extents("test").is_ok());
    }

    #[test]
    fn a_count_view_of_the_wrong_rank_count_is_refused() {
        let mut exchange = exchange(2, 64, 2048, 8);
        let (_, _, _, counts) = views(4, 64, 2048, 8);
        exchange.counts = counts;
        assert!(exchange.require_extents("test").is_err());
    }

    #[test]
    fn an_exchange_with_no_expert_is_refused() {
        let exchange = exchange(1, 8, 128, 0);
        assert!(exchange.require_extents("test").is_err());
    }
}
