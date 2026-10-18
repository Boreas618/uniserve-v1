//! Startup warmup: graph budgets, capability preparation and capture, then
//! sealing.
//!
//! Phase order (identical on every rank, because capture runs collectives):
//!
//! 1. Graph budgets: each CUDA device's graph pools may hold what they hold
//!    now plus the grant left after product storage not yet resident.
//! 2. Capability preparation ([`WarmupHooks::prepare`]): media layouts,
//!    encoders and decoders, each runtime's placeholder steps.
//! 3. Capture ([`WarmupHooks::capture`]): graphs in each runtime's kind and
//!    shape order, every capture followed by a graph-storage check.
//! 4. Completion: kernel table, graph storage seal.
//!
//! Capability runtimes contribute phases 2 and 3 through [`WarmupHooks`].
//! The serving path itself is warmed by the engine once the rank reported:
//! it runs real requests of the shapes the rank states in its
//! `WorkerInfo` (`image_warmups`, `diffusion_lm_step_rows`) through its
//! scheduler.

use crate::error::WorkerError;
use crate::executor::backend::WorkerBackend;

/// Warmup work of one capability runtime. Every method runs on every rank
/// of the group in the same order.
pub(crate) trait WarmupHooks {
    /// Prepare modules that serve several sizes and run their placeholder
    /// steps.
    fn prepare(&mut self, _backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        Ok(())
    }

    /// Capture the runtime's graphs.
    fn capture(&mut self, _backend: &mut WorkerBackend) -> std::result::Result<(), WorkerError> {
        Ok(())
    }
}
