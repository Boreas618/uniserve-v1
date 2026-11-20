//! The runners: what serves one stage a rank holds, one implementation of [`Runner`] per
//! capability a model exposes, with what they share: [`graphs`], [`buckets`], the [`sampler`]
//! and the [`thread_pool`]s of host work.
//!
//! A runner owns its model through a capability trait of `nn::capability`, its input tensors and
//! its [`Graphs`], which hold the execution context its forwards compute under. It is called
//! only on its device's worker thread, one job at a time.
//!
//! There is one implementation per capability. Everything the implementations share (the
//! profile run, capture, replay, the input buffer, generations and shadow) is [`Graphs`]'s, which
//! reaches a runner through [`Forward`] alone: [`Forward::buckets`] states what warm-up captures
//! and [`Forward::forward`] is the one function that is run on the meta device, rehearsed,
//! captured, and run again before each replay. A runner's worker captures on a thread of its own,
//! over a second instance of the runner's forward ([`Runner::capturer`]).
//!
//! ## What the worker hands a runner
//!
//! The worker thread owns the rank's device block tables ([`BlockTables`]): the buffer pool's
//! completions the scheduler maps and the requests it has the worker stamp change them, as jobs
//! of the worker's queue, so the tables a batch sees are those of every job before it. The
//! runners read them and keep none of their own. For every batch, and for every turn of a shadow
//! batch, the worker calls in this order:
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
//! first runner of the rank that does, in the worker's order. A KV copy that starts with a
//! vision encode is run by the causal runner, and a trajectory's save that starts with an encode
//! by the denoiser, the one media runner that maps the trajectory rows. A copy without runs lies
//! in other ranks' memory and asks nothing of this rank. A copy with runs that no runner maps is
//! a defect of the rank's wiring: no runner is handed it, and the worker reports it failed
//! (`Finish { ok: false }`) and logs the regions no runner maps.
//!
//! A batch whose work is the host's, a host phase of a media request, runs off the worker
//! thread: the worker first calls [`Runner::start`], and when the runner started host work it
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
//! - [`Runner::execute`] returns only once the device has completed the batch: the worker counts
//!   the batch as ended, and lets the buffer pool reuse what it addressed, on its return.

//!
//! ## The runners
//!
//! A runner holds its model through a capability trait of `nn` and dispatches on nothing else.
//!
//! | Runner | Capability | Phases |
//! | --- | --- | --- |
//! | [`CausalLmRunner`] | `nn::family::TextModel` | prefill, decode, verification |
//! | [`EncoderRunner`](encoder::EncoderRunner) | `nn::capability::Encoder` | a video request's text encoding |
//! | [`VisionRunner`](vision::VisionRunner) | `nn::capability::Vision` of a `TextModel` | a token request's image encoding |
//! | [`MediaHost`](media::MediaHost) | none | a video request's host work: reading its condition media, encoding its video and audio, muxing its artifact |
//! | [`DenoiserRunner`](denoiser::DenoiserRunner) | `nn::Denoiser` | a video request's latent preparation and its denoising steps, over its state in its slot's row |
//! | [`DecoderRunner`](decoder::DecoderRunner) | `nn::Decoder`, with `nn::Splicer` for a video | a video request's decode of its clips and its audio track, from the latents in the rank's media products |
//!
//! A media runner reads and writes a request's products and encoder outputs in the blocks the
//! rank's device block tables hold (`crate::memory::buffers::BufferMemory`), and issues the copies
//! that start with its batches through it, on its own stream.

pub mod buckets;
// What a video deployment captures and warms, from its model's declarations.
pub(crate) mod captures;
pub mod causal_lm;
// The host conversions of condition media: an image's patch rows for the vision runner, a
// video request's condition pixels and tracks for the media host.
pub(crate) mod conditions;
pub(crate) mod decoder;
pub(crate) mod denoiser;
pub(crate) mod encoder;
pub mod graphs;
pub(crate) mod media;
pub(crate) mod sampler;
pub(crate) mod thread_pool;
pub(crate) mod vision;

use platform::Stream;
use tensor::Tensor;

pub use causal_lm::CausalLmRunner;

use self::buckets::Bucket;
use self::graphs::Graphs;
use crate::batch::{Batch, BatchOutput, CallOutput, CallStatus, ErrorCode, Phase};
use crate::memory::tables::{BlockCopy, BlockTables, Change};
use crate::worker::Result;
use crate::worker::completion::Waker;

/// The worker's name for one runner of its device.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct RunnerId(pub u32);

/// What a runner computes, apart from how it serves: its buckets and the one forward that is run
/// on the meta device, captured, rehearsed and, unless its host values are fixed, run again
/// before each replay, over its graphs.
///
/// A runner is one ([`Runner`]), and so is the second instance of its forward a runner makes for
/// its worker's capture thread ([`Runner::capturer`]): the capture thread profiles, rehearses and
/// captures through this trait alone, and calls nothing else of a runner.
pub trait Forward: Send {
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
    /// on the meta device, rehearsed, captured, and run again by `prepare_inputs` before each
    /// replay unless its host values are fixed ([`Forward::fixed_host_values`]).
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

    /// Whether the host values the forward computes, which its host-to-device copies read,
    /// follow from the bucket and the graph's kernels alone, as the copies themselves do: then a
    /// replay's copies read the values its graph's capture computed, and `prepare_inputs` runs no
    /// forward before the replay. A runner whose forward computes a host value from its own
    /// state of the step (a bound, a timestep) answers `false`, the default, and its forward runs
    /// again under the meta context before each replay.
    fn fixed_host_values(&self) -> bool {
        false
    }

    /// Writes into `columns`, the bytes of the runner's input tensors in the layout of their
    /// device block, the inputs of an idle step of `bucket`: rows that address no slot, page,
    /// token or request, so a forward over them reads and writes no request's state. A rehearsal
    /// runs a bucket's forward over these, in memory of its own, before the bucket is captured
    /// under kernels whose first use a capture cannot record ([`Graphs::rehearse`]).
    ///
    /// # Errors
    ///
    /// A bucket this runner does not state.
    fn idle_inputs(&self, bucket: &Bucket, columns: &mut [u8]) -> Result<()>;

    /// The graphs the forward computes under: the runner's, which the worker replays, switches
    /// and releases at a batch boundary, or the graphs of a capture instance.
    fn graphs(&self) -> &Graphs;
}

/// The options a kernel generation captures a causal runner's graphs for: the deployment's
/// bucket catalog (`cuda_graph_config`, with the bounds of the deployment's batches), the
/// candidates a scored row reports (`max_logprobs`) and the drafts a verification holds at most
/// (`speculative_config.num_draft_tokens`). A runner's shape follows from them
/// ([`Runner::reshape`]).
#[derive(Clone, Debug, PartialEq)]
pub struct GraphOptions {
    /// The catalog of a rank outside an expert union, before the drafts make its decode buckets
    /// verifications (`Buckets::for_batch`).
    pub buckets: buckets::Buckets,
    /// The most likely tokens a scored row reports beside its own.
    pub candidates: u32,
    /// The drafts a verification holds at most; zero for a runner that speculates nothing.
    pub drafts: u32,
}

/// A runner's shape under a generation's options, measured before anything of it is made
/// ([`Runner::measure`]).
pub struct Measure {
    /// The shape's forward over graphs that hold no device, whose profile run states the
    /// shape's kernels, its region's peak and its input values ([`Graphs::profile_run`]).
    pub instance: Box<dyn Forward>,
    /// The device bytes the shape allocates beside its region and its graphs: what it does not
    /// share of the serving shape's storage, the recurrent state it lays out included.
    pub storage: u64,
    /// The recurrent state the shape lays out otherwise than the serving one, which the
    /// generation lays out beside it (`StateMemory::prepare`); `None` for a shape over the
    /// serving state.
    pub state: Option<crate::memory::state::StateConfig>,
    /// The kernels the shape launches outside its forwards, as `(operator, key)`: the copies of
    /// the state it lays out.
    pub keys: std::collections::BTreeSet<(&'static str, String)>,
}

/// One stage a rank holds, as the worker and [`Graphs`] drive it.
pub trait Runner: Forward {
    /// The phases this runner serves, as the scheduler names them in a batch.
    fn phases(&self) -> &[Phase];

    /// A second instance of the runner's forward over `graphs`, for the worker's capture
    /// thread: it shares the model's weights and every tensor the runner's forward reads at a
    /// fixed address beside its input tensors (slot state, cache planes, kept outputs), and takes
    /// its input tensors and allocator region from `graphs`. The worker makes two when it starts:
    /// one over graphs beside the runner's ([`Graphs::beside`]), whose captures address what the
    /// runner's graphs address and replay on the runner's stream, and one over a rehearsal's
    /// graphs ([`Graphs::rehearsal`]), whose forwards run over idle inputs in memory of their own.
    ///
    /// Over idle inputs ([`Forward::idle_inputs`]) the instance's forward writes, outside its
    /// graphs' region, only what idle inputs address: padding rows and pages no request holds.
    /// What the runner's forward writes at a fixed address that a later forward of a served step
    /// reads (a speculating runner's kept outputs) is the instance's own unless `graphs` lie
    /// beside the runner's.
    ///
    /// # Errors
    ///
    /// What allocating the instance's own tensors fails with.
    fn capturer(&self, graphs: Graphs) -> Result<Box<dyn Forward>>;

    /// The forward of the runner's shape under `options` over graphs that hold no device, and
    /// the device bytes the shape allocates beside its region: what a kernel generation of those
    /// options measures by a profile run of the forward ([`Graphs::profile_run`]) before anything
    /// of it is made (`Runner::reshape`). `None` for a runner whose computation no option of a
    /// generation shapes: it takes a generation's kernels in its graphs as they are.
    ///
    /// `model` is the model a generation of other drafts loads for the shape, loaded over meta
    /// weights: the causal model with its drafter loaded or released (`None` for the serving
    /// one).
    ///
    /// # Errors
    ///
    /// Options the runner cannot serve in place. The default measures nothing.
    fn measure(
        &self,
        options: &GraphOptions,
        model: Option<&std::sync::Arc<dyn nn::family::TextModel>>,
    ) -> Result<Option<Measure>> {
        let _ = (options, model);
        Ok(None)
    }

    /// Prepares the runner's shape under `options` for kernel generation `number`, whose
    /// kernels are `kernels`, beside the serving shape: its graphs over the given region (the
    /// runner's own for `None`) with input slots of the given value bytes
    /// ([`Graphs::over`]), and what else of it differs, allocated. Every bucket of the shape is
    /// then captured for its graphs ([`Runner::reshaped`], [`Runner::reshaped_capturer`]).
    /// Returns whether the runner prepared a shape: `false` for one whose computation no option
    /// shapes, which takes the generation in its graphs ([`Graphs::prepare`]). The options come
    /// with the model a generation of other drafts loaded for the shape, as
    /// [`Runner::measure`] takes it.
    ///
    /// # Errors
    ///
    /// Options the runner cannot serve in place, another shape held, or an allocation refused.
    /// The default prepares nothing.
    fn reshape(
        &mut self,
        number: u64,
        options: (
            &GraphOptions,
            Option<&std::sync::Arc<dyn nn::family::TextModel>>,
        ),
        kernels: std::sync::Arc<ops::Generation>,
        region: (Option<platform::DeviceRange>, u64),
    ) -> Result<bool> {
        let _ = (number, options, kernels, region);
        Ok(false)
    }

    /// The number of the generation whose shape [`Runner::reshape`] prepared, and the shape's
    /// graphs, which its captures are installed in; `None` while none is prepared.
    fn reshaped(&self) -> Option<(u64, &Graphs)> {
        None
    }

    /// The phases the prepared shape serves, which batches name from its switch on; none while
    /// no shape is prepared.
    fn reshaped_phases(&self) -> &[Phase] {
        &[]
    }

    /// A forward instance of the prepared shape over `graphs`, as [`Runner::capturer`] makes the
    /// serving shape's: over graphs beside the shape's ([`Graphs::beside`]) or a rehearsal's.
    ///
    /// # Errors
    ///
    /// No shape prepared, or what allocating the instance's own tensors fails with.
    fn reshaped_capturer(&self, graphs: Graphs) -> Result<Box<dyn Forward>> {
        let _ = graphs;
        Err(crate::worker::Error::State("the runner prepares no shape"))
    }

    /// Switches to the shape generation `number` prepared, whole: the runner serves it from its
    /// next step on, its graphs counting what the serving graphs counted
    /// ([`Graphs::take_over`]), and keeps the shape that served until the generation commits or
    /// rolls back. Returns whether it switched: `false` for a runner that prepared no shape.
    ///
    /// # Errors
    ///
    /// A shape prepared for another generation, or a device that cannot be waited for.
    fn switch_shape(&mut self, number: u64) -> Result<bool> {
        let _ = number;
        Ok(false)
    }

    /// Releases the shape the switch to generation `number` kept, once the device passed its
    /// last replay. Returns whether the runner kept one.
    ///
    /// # Errors
    ///
    /// A device that cannot be waited for.
    fn commit_shape(&mut self, number: u64) -> Result<bool> {
        let _ = number;
        Ok(false)
    }

    /// Gives back the shape of generation `number`: one prepared and never switched to is
    /// released, and after a switch to it the kept shape serves again from the next step on and
    /// the generation's is released. Returns whether the kept shape serves again.
    ///
    /// # Errors
    ///
    /// A device that cannot be waited for.
    fn rollback_shape(&mut self, number: u64) -> Result<bool> {
        let _ = number;
        Ok(false)
    }

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
        Err(crate::worker::Error::Unsupported(
            "this runner prepares no inputs for a paired forward".to_owned(),
        ))
    }

    /// One step: prepare the inputs from `batch` and `tables`, run the bucket's graph, read the
    /// outputs. It returns once the device has completed the step.
    ///
    /// For a batch whose work [`Runner::start`] started, it is called once that work ended: it
    /// writes what host work produced into the memory `start` found the batch's blocks at, or
    /// reads the outputs of the step `start` launched, and reads nothing of `tables`, whose
    /// entries may have been taken away meanwhile.
    ///
    /// # Errors
    ///
    /// As [`Runner::prepare_inputs`], or the device's failure.
    fn execute(&mut self, batch: &Batch, tables: &BlockTables) -> Result<BatchOutput>;

    /// Whether [`Runner::execute`] computes anything on the device for `batch`, a batch with
    /// rows, before it runs: what makes the batch a step a profiling session's window counts
    /// and the range markers bracket. Only an expert join can compute nothing, when it asks for
    /// a step this member of the union already took part in (`Variant::ExpertJoin`); the
    /// default computes every batch.
    fn computes(&self, batch: &Batch) -> bool {
        let _ = batch;
        true
    }

    /// Starts the work of `batch`, a batch with rows, that runs on beyond this call, and returns
    /// whether it started any: host work on a thread that is not the worker's, whose reads of
    /// the rank's memory are found through `tables` and copied to the host before this returns,
    /// or a step issued on the device that this call does not wait for. `resume` is called once,
    /// from the thread that saw the work end, and the worker then calls [`Runner::execute`]
    /// with the same batch; it performs other jobs in between, the next batch's start
    /// included, so a device runner issues the next step behind this one on its stream.
    ///
    /// # Errors
    ///
    /// What refuses the batch; no work of it is left running then. The default starts none: the
    /// runner's [`Runner::execute`] does all of a batch's work.
    fn start(&mut self, batch: &Batch, tables: &BlockTables, resume: Waker) -> Result<bool> {
        let _ = (batch, tables, resume);
        Ok(false)
    }

    /// The runner's weights changed at their addresses: recompute what the runner or its model
    /// derived from them. It runs on the worker thread with nothing in flight.
    ///
    /// # Errors
    ///
    /// What recomputing a derived constant fails with.
    fn weights_changed(&mut self) -> Result<()> {
        Ok(())
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
    /// worker runs next, whose end covers them ([`Runner::wait_copies`]).
    ///
    /// # Errors
    ///
    /// When a copy cannot be issued. The worker then reports every one of `copies` failed
    /// (`Finish { ok: false }`) and still runs the batch, whose end covers whatever was issued.
    /// The default refuses every copy: a runner that maps no region is handed none.
    fn copy(&mut self, copies: &[BlockCopy]) -> Result<()> {
        match copies.first() {
            Some(copy) => Err(crate::worker::Error::Unsupported(format!(
                "this runner holds no request memory to run the copies of request {}",
                copy.request.0
            ))),
            None => Ok(()),
        }
    }

    /// Returns once the device has completed every copy [`Runner::copy`] issued, also when
    /// another runner serves the batch the copies started with. The worker calls it on each
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

/// The forward instance of a runner whose work is the host's (`Runner::capturer`): it states no
/// bucket, so its capture thread is handed none.
pub struct HostForward {
    graphs: Graphs,
}

impl HostForward {
    /// The instance over `graphs`.
    pub const fn new(graphs: Graphs) -> Self {
        Self { graphs }
    }
}

impl Forward for HostForward {
    fn buckets(&self) -> Vec<Bucket> {
        Vec::new()
    }

    fn forward(&self, bucket: &Bucket) -> Result<Vec<Tensor>> {
        Err(crate::worker::Error::Unsupported(format!(
            "bucket {bucket} is a host call's, which has no forward"
        )))
    }

    fn idle_inputs(&self, bucket: &Bucket, _columns: &mut [u8]) -> Result<()> {
        Err(crate::worker::Error::Unsupported(format!(
            "bucket {bucket} is a host call's, which has no input tensors"
        )))
    }

    fn graphs(&self) -> &Graphs {
        &self.graphs
    }
}

/// The device memory of `bytes` bytes a forward instance over `graphs` (`Runner::capturer`)
/// writes at fixed addresses beside its graphs' region, where its runner's forward writes
/// `shared`: `shared` itself for an instance beside the runner's graphs, whose captures replay in
/// the runner's place, and zeroed memory of its own for a rehearsal's, so a rehearsal writes
/// nothing a serving forward reads. `None` for a runner that holds none.
///
/// # Errors
///
/// When the rehearsal's memory cannot be allocated or zeroed.
pub(crate) fn instance_memory(
    graphs: &Graphs,
    shared: Option<&std::sync::Arc<crate::memory::allocation::Storage>>,
    bytes: u64,
) -> Result<Option<std::sync::Arc<crate::memory::allocation::Storage>>> {
    let (Some(shared), Some(stream)) = (shared, graphs.stream()) else {
        return Ok(shared.cloned());
    };
    if !graphs.rehearses() {
        return Ok(Some(std::sync::Arc::clone(shared)));
    }

    let length = usize::try_from(bytes).map_err(|_| {
        crate::worker::Error::Invalid("a rehearsal's buffers exceed memory".to_owned())
    })?;
    let storage = crate::memory::allocation::Storage::linear(stream.device(), length)?;
    // SAFETY: the range is the whole allocation, which the instance keeps for as long as it
    // computes over it.
    unsafe { platform::memset_d8(storage.address(), 0, length, stream) }?;
    Ok(Some(storage))
}

/// A copy of device bytes: from an address, to an address, and how many bytes.
pub(crate) type DeviceCopy = (u64, u64, u64);

/// A call's output before anything is known of it: performed, with no token and the default
/// state.
pub(crate) fn performed() -> CallOutput {
    CallOutput {
        status: CallStatus::Ok,
        error_code: None,
        error_message: None,
        tokens: Vec::new(),
        step_index: 0,
        cache_computed_len: 0,
        cache_visible_len: 0,
        token_logprobs: Vec::new(),
        prompt_logprobs: Vec::new(),
        media_output: None,
        draft_token_ids: Vec::new(),
    }
}

/// The output of a call whose request's inputs this rank refused, saying why.
pub(crate) fn refused(message: String) -> CallOutput {
    CallOutput {
        status: CallStatus::Error,
        error_code: Some(ErrorCode::InputError),
        error_message: Some(message),
        ..performed()
    }
}

/// The bytes of one element of `dtype`.
///
/// # Errors
///
/// [`crate::worker::Error::Invalid`] for an element type with no whole-byte layout.
pub(crate) fn element_bytes(dtype: tensor::DType) -> Result<u64> {
    ops::abi::bytes(dtype).ok_or_else(|| {
        crate::worker::Error::Invalid(format!(
            "{} has no whole-byte element",
            ops::abi::name(dtype)
        ))
    })
}

/// The address of `tensor`'s first element: a device address, or for a meta tensor the address
/// it was made at, which nothing reads.
pub(crate) fn address(tensor: &Tensor) -> Result<u64> {
    Ok(ops::operand(tensor)?.ptr.address())
}

/// Issues `copies` on `stream`, behind the work issued on it before; with no stream, for a
/// runner whose memory holds no device, nothing is issued.
///
/// # Safety
///
/// Every range lies in device memory its owner keeps mapped until the stream has passed the
/// copies, and nothing but these copies writes the destinations meanwhile.
pub(crate) unsafe fn issue(copies: &[DeviceCopy], stream: Option<&Stream>) -> Result<()> {
    let Some(stream) = stream else {
        return Ok(());
    };

    for &(from, to, bytes) in copies {
        let bytes = usize::try_from(bytes)
            .map_err(|_| crate::worker::Error::Resource("a copy exceeds the address space"))?;

        // SAFETY: forwarded from the caller.
        unsafe { platform::memcpy_async(to, from, bytes, stream) }?;
    }
    Ok(())
}

/// Waits for everything issued on `stream`, a media runner's: its forwards, the copies of their
/// outputs and the copies that started with its batch. With no stream nothing was issued.
pub(crate) fn settle(stream: Option<&Stream>) -> Result<()> {
    match stream {
        Some(stream) => Ok(stream.synchronize()?),
        None => Ok(()),
    }
}
