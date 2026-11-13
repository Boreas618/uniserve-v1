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
//!
//! ## What the executor hands a runner
//!
//! The executor thread owns the rank's device block tables ([`BlockTables`]): the buffer pool's
//! completions the scheduler maps and the requests it has the executor stamp change them, as jobs
//! of the executor's queue, so the tables a batch sees are those of every job before it. The
//! runners read them and keep none of their own. For every batch, and for every turn of a shadow
//! batch, the executor calls in this order:
//!
//! 1. [`Runner::write_tables`] on every runner of the rank, with the positions whose entries
//!    changed since the last call ([`BlockTables::take_changes`]), each change once. A runner
//!    that keeps a device block table writes them into it before its next forward reads it, and
//!    resets the per-unit state of the units a change marks `reset`.
//! 2. [`Runner::copy`] on the runners whose memory holds the copies that start with the batch
//!    ([`BlockTables::start_copy`]), when there are any, each runner once with its copies in the
//!    order they were queued. Copies start only with a batch the engine numbered and that every
//!    rank of the group runs; never with a shadow batch, an expert join, or a batch one rank runs
//!    alone.
//! 3. [`Runner::execute`] with the batch and the tables, or for a shadow turn
//!    [`Runner::prepare_shadow_inputs`] with the batch and the tables.
//! 4. [`Runner::wait_copies`] on every runner that was handed copies in step 2, once the batch's
//!    runner returned.
//!
//! ## Which runner runs a copy
//!
//! A runner states the memory it holds by region, the number the buffer pool's answer to `Open`
//! gave it ([`Runner::maps`]): the regions of the memory its owners lent it, such as the KV
//! cache's units and host pages, or the rank's media memory. A copy is run by a runner that maps
//! the regions of both runs of each of its pairs, whichever phase the batch it starts with is:
//! the batch's runner when it does, so the copy shares the batch's stream, and otherwise the
//! first runner of the rank that does, in the executor's order. A KV copy that starts with a
//! vision encode is run by the causal runner, and a trajectory's save that starts with an encode
//! by the denoiser, the one media runner that maps the trajectory rows. A copy without runs lies
//! in other ranks' memory and asks nothing of this rank. A copy with runs that no runner maps is
//! a defect of the rank's wiring: no runner is handed it, and the executor reports it failed
//! (`Finish { ok: false }`) and logs the regions no runner maps.
//!
//! A batch whose work is the host's, a host phase of a media request, runs off the executor
//! thread: the executor first calls [`Runner::start`], and when the runner started host work it
//! calls [`Runner::execute`] with the same batch once that work ended, other jobs of the queue
//! running in between. The batch keeps its number open meanwhile, so nothing it reads or writes
//! is given up before it ends.
//!
//! What a runner must keep, for the buffer pool's guarantees to hold (the obligations of the
//! control plane in the buffer pool's design):
//!
//! - Every address a forward reads or writes in a request's memory comes from the tables: a
//!   token's cache position, a recurrent-state checkpoint's position, an encoder output's or a
//!   media product's position, each turned into units by [`BlockTables::units`] under the row's
//!   slot. A position without an entry is refused before anything is launched.
//! - A batch writes only entries [`BlockTables::writable`] states writable; a row that would write
//!   a read-only entry, or a position without an entry, is refused before anything is launched.
//! - [`Runner::copy`] issues each copy from its first run to its second on the rank's copy stream,
//!   and writes nothing but the second run. The end of the batch the copies start with covers
//!   them: [`Runner::wait_copies`] returns only once the device has completed the copies the
//!   runner issued, also when `copy` failed after issuing part of them.
//! - [`Runner::execute`] returns only once the device has completed the batch: the executor counts
//!   the batch as ended, and lets the buffer pool reuse what it addressed, on its return.

use ipc::batch::{Batch, BatchOutput, Phase};
use tensor::Tensor;

use crate::Result;
use crate::buckets::Bucket;
use crate::completion::Waker;
use crate::graphs::Graphs;
use crate::memory::tables::{BlockCopy, BlockTables, Change};

/// The executor's name for one runner of its device.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct RunnerId(pub u32);

/// One stage a rank holds, as the executor and [`Graphs`] drive it.
pub trait Runner: Send {
    /// The phases this runner serves, as the scheduler names them in a batch.
    fn phases(&self) -> &[Phase];

    /// The buckets warm-up captures, largest first.
    ///
    /// The ranks of a group state the same buckets in the same order: a collective captured in
    /// one rank's graph meets the same collective of every other rank's.
    fn buckets(&self) -> Vec<Bucket>;

    /// The buckets of the forwards the runner runs eagerly ([`Graphs::eager`]) and captures no
    /// graph of. A profile run states their launches and their peak with the captured buckets',
    /// so the runner's region holds them and every generation resolves their kernels; warm-up
    /// captures none of them. The default states none.
    fn eager_buckets(&self) -> Vec<Bucket> {
        Vec::new()
    }

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
    /// columns, with every address of a request's memory taken from `tables`, and writes the
    /// host values the bucket's forward computes into the input buffer
    /// ([`Graphs::prepare_inputs`]).
    ///
    /// # Errors
    ///
    /// A batch no bucket holds, columns the runner refuses, or a position the tables hold no
    /// entry for, or a writable one for, where the batch writes.
    fn prepare_inputs(&mut self, batch: &Batch, tables: &BlockTables) -> Result<Bucket>;

    /// As [`Runner::prepare_inputs`], for a shadow batch of the generation being prepared: the
    /// inputs of a paired forward ([`Graphs::shadow`]), which must write no state a serving
    /// request reads, so the runner redirects what the forward writes to scratch memory.
    ///
    /// # Errors
    ///
    /// The default refuses: a runner that states no scratch memory shadows nothing.
    fn prepare_shadow_inputs(&mut self, batch: &Batch, tables: &BlockTables) -> Result<Bucket> {
        let _ = (batch, tables);
        Err(crate::Error::Unsupported(
            "this runner prepares no inputs for a paired forward".to_owned(),
        ))
    }

    /// One step: prepare the inputs from `batch` and `tables`, run the bucket's graph, read the
    /// outputs. It returns once the device has completed the step.
    ///
    /// For a batch whose host work [`Runner::start`] started, it is called once that work
    /// ended: it writes what the work produced into the memory `start` found the batch's blocks
    /// at, and reads nothing of `tables`, whose entries may have been taken away meanwhile.
    ///
    /// # Errors
    ///
    /// As [`Runner::prepare_inputs`], or the device's failure.
    fn execute(&mut self, batch: &Batch, tables: &BlockTables) -> Result<BatchOutput>;

    /// Starts the host work of `batch`, a batch with rows, on a thread that is not the
    /// executor's, and returns whether it started any. What the work reads of the rank's memory
    /// is found through `tables` and copied to the host before this returns; `resume` is called
    /// once, from the thread the work ended on, and the executor then calls
    /// [`Runner::execute`] with the same batch.
    ///
    /// # Errors
    ///
    /// What refuses the batch; no work of it is left running then. The default starts none: a
    /// device runner's [`Runner::execute`] does all of a batch's work.
    fn start(&mut self, batch: &Batch, tables: &BlockTables, resume: Waker) -> Result<bool> {
        let _ = (batch, tables, resume);
        Ok(false)
    }

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

    /// The rank's device block tables changed at `changes` since the last call, in slot, class
    /// and position order: each states the run a position holds now, or none, and whether the
    /// run's units are new to their owner and must be reset before a forward reads them. Called
    /// on every runner of the rank before the forward of every batch and every shadow turn,
    /// with every change exactly once.
    ///
    /// # Errors
    ///
    /// What writing the device's table fails with; the batch then fails. The default keeps no
    /// device table and accepts every change.
    fn write_tables(&mut self, changes: &[Change]) -> Result<()> {
        let _ = changes;
        Ok(())
    }

    /// Whether the runner's memory holds `region`, by the number the buffer pool's answer to
    /// `Open` gave it (module documentation). It is asked when copies start, once the rank's
    /// memory owners were told the numbers; a runner reaches some owners only through what
    /// issues its work, so it is asked mutably.
    ///
    /// The default maps none: a runner that holds no request memory is handed no copy.
    fn maps(&mut self, region: u8) -> bool {
        let _ = region;
        false
    }

    /// Issues `copies`, whose runs all lie in regions the runner maps ([`Runner::maps`]), on the
    /// rank's copy stream, each from its first run to its second, ahead of the batch the
    /// executor runs next, whose end covers them ([`Runner::wait_copies`]).
    ///
    /// # Errors
    ///
    /// When a copy cannot be issued. The executor then reports every one of `copies` failed
    /// (`Finish { ok: false }`) and still runs the batch, whose end covers whatever was issued.
    /// The default refuses every copy: a runner that maps no region is handed none.
    fn copy(&mut self, copies: &[BlockCopy]) -> Result<()> {
        match copies.first() {
            Some(copy) => Err(crate::Error::Unsupported(format!(
                "this runner holds no request memory to run the copies of request {}",
                copy.request.0
            ))),
            None => Ok(()),
        }
    }

    /// Returns once the device has completed every copy [`Runner::copy`] issued, also when
    /// another runner serves the batch the copies started with. The executor calls it on each
    /// runner it handed copies to once the batch's runner returned, and on every runner before
    /// a job that needs nothing in flight on the device.
    ///
    /// # Errors
    ///
    /// The device's failure; whether the copies ended is then unknown. The default waits for
    /// nothing: a runner that maps no region issues no copy.
    fn wait_copies(&mut self) -> Result<()> {
        Ok(())
    }
}
