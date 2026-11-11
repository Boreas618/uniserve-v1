//! What an expert-parallel forward borrows from the package's exchange transport.
//!
//! A tensor-parallel deployment replicates the router and keeps each rank's experts, so a token
//! routed to another rank has to travel there. The package owns the transport — its buffers, its
//! ordering, its state — because the workspace layout is the program's, and dispatches and combines
//! through it (`ops::moe::exchange`).
//!
//! A model layer cannot build that transport: it may not create runtime objects, and the worker
//! is the one that calls the package's creation entry. What travels through [`Forward`] is
//! therefore only the views the transport wrote and the layer reads, lent by whoever owns them.
//! The worker builds one [`DispatchViews`] per device and group after loading the model, and
//! lends it on every forward of a block whose experts are split.
//!
//! [`Forward`]: crate::Forward

use std::sync::Arc;

use ops::Operand;
use ops::moe::exchange::ExchangeTransport;

/// The received rows of one expert-parallel exchange, and the transport that moves them.
///
/// The views are the transport's own device buffers: the layer reads them to sort and compute its
/// experts and writes its sums back, and never allocates or frees them. `hidden` and `top` describe
/// one row of the received layout. `ranks x rank_rows` rows exist, and `live` states how many of them
/// the running step holds, which the ranks agree on the device.
pub struct DispatchViews {
    /// The package's transport for this device and group.
    ///
    /// Opaque: the layer only hands it back to the exchange operator. Its lifetime belongs to the
    /// worker, which keeps it alive for every forward that borrows it.
    pub transport: ExchangeTransport,
    /// Ranks in the expert-parallel group; the leading axis of the received layout.
    pub ranks: u32,
    /// Leading ranks of the group that hold no experts.
    pub attention: u32,
    /// Rows of one rank's block of the received views: the most one rank may exchange in a step,
    /// which the worker sizes the transport for, so the rows fit without a host read.
    pub rank_rows: u32,
    /// Rows per rank the expert work of this walk is shaped for: the step's graph holds at most
    /// this many, so the received views are read to `ranks * step_rows` rows. At most `rank_rows`.
    pub step_rows: u32,
    /// `[ranks * rank_rows, hidden]` in the activation element type: the dispatched activations.
    pub states: Operand,
    /// `[ranks * rank_rows, top]` i32: the experts each received row is routed to.
    pub ids: Operand,
    /// `[ranks * rank_rows, top]` f32: the router weights of each received row.
    pub weights: Operand,
    /// `[ranks * rank_rows, hidden]` in the activation element type: where this rank's experts leave
    /// each received row's sum, for the combine to return.
    pub combined: Operand,
    /// `[1]` i32: the leading rows of these views the running step holds.
    pub live: Operand,
    /// `[rank_rows, hidden / 2]` u8: where a fused launch reads this rank's encoded rows.
    pub packed: Operand,
    /// `[rank_rows, hidden / 16]` block scales of those rows.
    pub packed_scales: Operand,
    /// One launch runs the whole expert layer, so the layer sorts and multiplies nothing itself.
    pub fused: bool,
    /// This rank holds no experts: it sends routes and takes sums back, and computes nothing.
    pub source_only: bool,
    /// Meets the group's other ranks on the host, during a warm walk only.
    ///
    /// The first use of a kernel takes a lock of the whole process and can wait on a device, and a
    /// launch that waits for its peers holds its device until they arrive. The ranks of a host are
    /// threads of one process, so a warm walk keeps every rank out of a waiting launch until every
    /// rank has finished the first uses before it. A replay finds every kernel ready and passes.
    pub meet: Arc<dyn Fn() -> crate::Result<()> + Send + Sync>,
}

impl DispatchViews {
    /// These views for a walk whose expert work holds `rows` rows per rank; zero keeps every row
    /// of a rank's block.
    #[must_use]
    pub fn at_rows(&self, rows: u32) -> Self {
        let step_rows = if rows == 0 {
            self.rank_rows
        } else {
            rows.min(self.rank_rows)
        };
        Self {
            transport: self.transport,
            ranks: self.ranks,
            attention: self.attention,
            rank_rows: self.rank_rows,
            step_rows,
            states: self.states,
            ids: self.ids,
            weights: self.weights,
            combined: self.combined,
            live: self.live,
            packed: self.packed,
            packed_scales: self.packed_scales,
            fused: self.fused,
            source_only: self.source_only,
            meet: Arc::clone(&self.meet),
        }
    }

    /// The received views, read to the rows the walk's expert work holds.
    #[must_use]
    pub fn narrowed(&self) -> Self {
        let rows = u64::from(self.ranks) * u64::from(self.step_rows);
        let head = |view: Operand| {
            Operand::new(
                view.ptr,
                view.dtype,
                crate::Shape::new(&[rows, view.shape.last()]),
            )
        };
        let mut narrowed = self.at_rows(self.step_rows);
        narrowed.states = head(self.states);
        narrowed.ids = head(self.ids);
        narrowed.weights = head(self.weights);
        narrowed.combined = head(self.combined);
        narrowed
    }

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
    /// The worker builds this value, so a wrong extent is a wiring error; checking it here keeps
    /// it out of a launch that would read unrelated memory.
    pub fn require_extents(&self, name: &'static str) -> crate::Result<()> {
        let rows = u64::from(self.ranks) * u64::from(self.rank_rows);
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
            (&self.combined, &[rows, hidden][..], "combined"),
            (&self.live, &[1][..], "live"),
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

    fn exchange(ranks: u32, rank_rows: u32, hidden: u64, top: u64) -> DispatchViews {
        let rows = u64::from(ranks) * u64::from(rank_rows);
        let base = DevicePtr::new(0x1_0000);
        DispatchViews {
            transport: {
                // SAFETY: the transport is opaque and never dereferenced by these tests; the null
                // handle only stands in for a value the worker would build.
                unsafe { ExchangeTransport::from_handle(core::ptr::null_mut()) }
            },
            ranks,
            attention: 0,
            rank_rows,
            step_rows: rank_rows,
            states: Operand::new(base, DType::BF16, Shape::new(&[rows, hidden])),
            ids: Operand::new(base, DType::I32, Shape::new(&[rows, top])),
            weights: Operand::new(base, DType::F32, Shape::new(&[rows, top])),
            combined: Operand::new(base, DType::BF16, Shape::new(&[rows, hidden])),
            live: Operand::new(base, DType::I32, Shape::new(&[1])),
            packed: Operand::new(
                base,
                DType::U8,
                Shape::new(&[u64::from(rank_rows), hidden / 2]),
            ),
            packed_scales: Operand::new(
                base,
                DType::F8E4M3,
                Shape::new(&[u64::from(rank_rows), hidden / 16]),
            ),
            fused: false,
            source_only: false,
            meet: Arc::new(|| Ok(())),
        }
    }

    #[test]
    fn an_exchange_of_the_sized_extents_passes() {
        let exchange = exchange(2, 64, 2048, 8);
        assert_eq!(exchange.hidden(), 2048);
        assert_eq!(exchange.top(), 8);
        assert!(exchange.require_extents("test").is_ok());
    }

    #[test]
    fn a_combined_view_of_the_wrong_rank_count_is_refused() {
        let mut views = exchange(2, 64, 2048, 8);
        views.combined = exchange(4, 64, 2048, 8).combined;
        assert!(views.require_extents("test").is_err());
    }

    #[test]
    fn an_exchange_with_no_expert_is_refused() {
        let exchange = exchange(1, 8, 128, 0);
        assert!(exchange.require_extents("test").is_err());
    }
}
