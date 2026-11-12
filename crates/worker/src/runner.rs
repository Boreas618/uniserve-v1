//! The runner: what serves one stage a rank holds.
//!
//! A runner owns its model through a capability trait of `nn::capability`, its input tensors and
//! its [`Graphs`], which hold the execution context its forwards compute under. It is called
//! only on its device's executor thread, one job at a time.
//!
//! There is one implementation per capability. Everything the implementations share (the
//! profile run, capture, replay, the input buffer, generations and shadow) is [`Graphs`]'s, which
//! reaches a runner through this trait alone: [`Runner::buckets`] states what warm-up captures
//! and [`Runner::forward`] is the one function that is run on the meta device, captured, and run
//! again before each replay.

use ipc::execution::{Batch, BatchOutput};
use uniserve_tensor::Tensor;

use crate::Result;
use crate::buckets::Bucket;
use crate::graphs::Graphs;

/// The executor's name for one runner of its device.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct RunnerId(pub u32);

/// One stage a rank holds, as the executor and [`Graphs`] drive it.
pub trait Runner: Send {
    /// The phases this runner serves, as the scheduler addresses them in a batch.
    fn phases(&self) -> &[ipc::PhaseId];

    /// The buckets warm-up captures, largest first.
    ///
    /// The ranks of a group state the same buckets in the same order: a collective captured in
    /// one rank's graph meets the same collective of every other rank's.
    fn buckets(&self) -> Vec<Bucket>;

    /// The forward of one bucket over the runner's input tensors: the one function that is run
    /// on the meta device, captured, and run again by `prepare_inputs` before each replay.
    ///
    /// It computes under the context installed on the thread, reads the runner's input tensors
    /// at their fixed addresses and returns the forward's outputs in an order the runner fixes
    /// per bucket. Its launches, allocations and host-to-device copies follow from the bucket
    /// alone, never from a request's values.
    ///
    /// # Errors
    ///
    /// What the model refuses, or a bucket this runner does not state.
    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>>;

    /// Writes into `columns`, the bytes of the runner's input tensors in the layout of their
    /// device block, the inputs of an idle step of `bucket`: rows that address no slot, page,
    /// token or request, so a forward over them reads and writes no request's state. Warm-up
    /// runs each bucket's forward over these before it captures it, and a generation is
    /// prepared over them beside the serving graphs.
    ///
    /// # Errors
    ///
    /// A bucket this runner does not state.
    fn idle_inputs(&self, bucket: &Bucket, columns: &mut [u8]) -> Result<()>;

    /// Chooses the smallest bucket that holds `batch`, fills the input tensors from the batch's
    /// columns and writes the host values the bucket's forward computes into the input buffer
    /// ([`Graphs::prepare_inputs`]).
    ///
    /// # Errors
    ///
    /// A batch no bucket holds, or columns the runner refuses.
    fn prepare_inputs(&mut self, batch: &Batch) -> Result<Bucket>;

    /// As [`Runner::prepare_inputs`], for a shadow batch of the generation being prepared: the
    /// inputs of a paired forward ([`Graphs::shadow`]), which must write no state a serving
    /// request reads, so the runner redirects what the forward writes to scratch memory.
    ///
    /// # Errors
    ///
    /// The default refuses: a runner that states no scratch memory shadows nothing.
    fn prepare_shadow_inputs(&mut self, batch: &Batch) -> Result<Bucket> {
        let _ = batch;
        Err(crate::Error::Unsupported(
            "this runner prepares no inputs for a paired forward".to_owned(),
        ))
    }

    /// One step: prepare the inputs, run the bucket's graph, read the outputs.
    ///
    /// # Errors
    ///
    /// As [`Runner::prepare_inputs`], or the device's failure.
    fn execute(&mut self, batch: &Batch) -> Result<BatchOutput>;

    /// The runner's graphs: what the executor captures, switches and releases at a batch
    /// boundary, and what a profile run of the runner is recorded in.
    fn graphs(&self) -> &Graphs;

    /// The runner's weights changed at their addresses: recompute what the runner or its model
    /// derived from them. It runs on the executor thread with nothing in flight.
    ///
    /// # Errors
    ///
    /// What recomputing a derived constant fails with.
    fn weights_changed(&mut self) -> Result<()> {
        Ok(())
    }

    /// States whether the forwards that follow are a warm-up's. The executor's capture job sets
    /// it on every runner at each of its turns, so batches served between two captures run with
    /// it set too, and clears it once no bucket is left or a capture failed. A runner whose
    /// forwards meet other ranks on the host during a warm-up, such as a member of an expert
    /// union, reads it; the default ignores it.
    fn set_warming(&self, warming: bool) {
        let _ = warming;
    }

    /// Writes the trajectories of preempted requests from their slots into the host latent
    /// pools, the `ipc::LatentPageCopy` entries of this runner's rank. It runs on the executor
    /// thread with nothing in flight, as a control operation (`Batches::offload_latents`).
    ///
    /// # Errors
    ///
    /// The default refuses: a runner that holds no trajectory offloads none.
    fn offload_latents(&mut self, offloads: &[ipc::LatentOffload]) -> Result<()> {
        let _ = offloads;
        Err(crate::Error::Unsupported(
            "this runner holds no trajectory to offload".to_owned(),
        ))
    }
}
