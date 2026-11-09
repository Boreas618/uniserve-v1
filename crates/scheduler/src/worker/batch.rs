//! Logical submissions.
//!
//! The scheduler builds an [`ExecutionBatch`] of calls, each paired with the
//! [`RequestDispatch`] it chose, plus lifecycle commands, and hands it to
//! `Submissions`, which describes it to the device worker or gives it to the
//! simulated one.

use super::WorkerGroupId;
#[cfg(feature = "testing")]
use ipc::NewRequest;
use ipc::{
    BatchCommand, BlockTable, BufferAllocation, CacheUnitAllocation, Call, DecodeRange,
    ForwardBatch, LatentParams, RequestKey, TensorExport,
};
use std::collections::BTreeMap;

/// A persistent span in the buffer pool one rank of a worker backs.
///
/// Each rank of a worker group has its own pool, so a buffer a worker holds
/// has one span on every rank that backs it, each at that rank's offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankBuffer {
    /// Rank within the worker group whose pool the span addresses.
    pub rank: u32,
    /// The span, in that rank's pool.
    pub allocation: BufferAllocation,
}

impl RankBuffer {
    /// Places each of `allocations` at the same offset on every one of a
    /// worker's `ranks`, for spans taken from a pool whose layout every rank
    /// shares.
    pub fn on_every_rank(allocations: &[BufferAllocation], ranks: u32) -> Vec<Self> {
        allocations
            .iter()
            .flat_map(|allocation| {
                (0..ranks).map(move |rank| Self {
                    rank,
                    allocation: *allocation,
                })
            })
            .collect()
    }
}

/// Physical dispatch selected by the scheduler for a call.
///
/// The call itself is the shared IPC `Call`. These fields describe
/// its worker and allocations, which are gathered into physical batch arrays.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestDispatch {
    /// Worker routing stays outside the call sent across IPC.
    pub worker: WorkerGroupId,
    /// Worker-local request row when a replicated route owns its own address
    /// space. Unset dispatches retain the admission's canonical row;
    /// The worker applies the override to the request's `Start`.
    pub request_pool_idx: Option<u32>,
    /// Cache tables and newly acquired units used by this call.
    pub block_tables: Vec<BlockTable>,
    pub new_cache_units: Vec<CacheUnitAllocation>,
    /// Rows use a local call index: `ExecutionBatch::validate` requires every
    /// row's index to be zero.
    pub forward: ForwardBatch,
    pub latent: Option<LatentParams>,
    pub decode: Option<DecodeRange>,
    /// Persistent spans for the call's buffer inputs and outputs, on each
    /// rank of the worker that backs them. Request retirement retains every
    /// physical reader and writer allocation.
    pub buffers: Vec<RankBuffer>,
    /// Pages of the host KV cache the call's new units are filled with
    /// before its forward, on every rank of its worker group.
    pub page_loads: Vec<ipc::PageCopy>,
    /// For a video request's call, the worker routed to each component that
    /// reads its products, keyed by component name, as the request's graph
    /// connects its calls; empty for a call whose products leave the graph.
    /// Admission fixes a video request's route, so a product is published
    /// only toward the replicas that read it. `None` for calls outside a
    /// video graph.
    pub readers: Option<BTreeMap<String, WorkerGroupId>>,
}

/// One logical worker submission. Its rank projections are derived only inside a worker.
#[derive(Debug, Clone)]
pub struct ExecutionBatch {
    /// Logical batch identity used to correlate partial completions.
    pub id: u64,
    /// Shared call kinds paired with physical dispatch in scheduler order.
    pub requests: Vec<(Call, RequestDispatch)>,
    /// Ordered lifecycle and resource commands.
    pub commands: Vec<BatchCommand>,
    /// Published transfer descriptors supplied by an external storage owner.
    pub input_transfers: Vec<TensorExport>,
    /// External cache exports consumed by explicit cache installation.
    ///
    /// `ExecutionBatch::new` leaves this empty.
    pub cache_inputs: Vec<ipc::CacheTransfer>,
    /// Published pages the batch writes to the host KV cache.
    pub page_offloads: Option<PageOffloads>,
    /// The batch's columns in the layout the kernels read, filled when the
    /// mechanism assembles the batch (design §4.3, `worker::BatchColumns`).
    ///
    /// The batch carries them so that a submission hands the worker one value
    /// and nothing re-derives the layout from the calls. `None` means the
    /// mechanism could not state them — a batch no worker may serve, as
    /// opposed to one served on invented values.
    pub columns: Option<worker::batch::BatchColumns>,
    /// A media computation's own columns, and `None` for every token batch (2.4).
    ///
    /// One batch is one computation: a media batch states the computation its kernels were
    /// captured for and, per request, the buffers the model's family prepared, and it carries
    /// no token column. The two fields are therefore never both `Some`.
    ///
    /// The mechanism is the one that fills this, and it fills it only with facts it owns: the
    /// computation comes from the plan the scheduler selected the call in, the extent from the
    /// bucket the worker states it would serve that computation in, and the buffers from the
    /// request's own stated parameters. A media batch whose inputs the mechanism cannot state is
    /// left without columns and is refused at submission by name, never served on invented bytes.
    pub media: Option<worker::batch::MediaColumns>,
}

/// Batches compare by what the mechanism decided: their identity, their
/// calls and their commands. The columns are that decision's projection into
/// the worker's layout (equal calls project equally, and a batch with none
/// is the unservable one), so equality states only whether both carry them.
impl PartialEq for ExecutionBatch {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.requests == other.requests
            && self.commands == other.commands
            && self.input_transfers == other.input_transfers
            && self.cache_inputs == other.cache_inputs
            && self.page_offloads == other.page_offloads
            && self.columns.is_some() == other.columns.is_some()
            && self.media.is_some() == other.media.is_some()
    }
}

/// Published pages one worker group writes to the host KV cache, for
/// every rank of the group (`PageCopy::rank`), and preempted video
/// trajectories it writes to its host latent pools (`LatentPageCopy::rank`).
/// The group runs every call of a batch that carries them; a batch without
/// calls carries only them.
#[derive(Debug, Clone, PartialEq)]
pub struct PageOffloads {
    pub worker: WorkerGroupId,
    pub pages: Vec<ipc::PageCopy>,
    pub latents: Vec<ipc::LatentOffload>,
}

impl ExecutionBatch {
    /// Constructs a logical worker submission.
    pub fn new(
        id: u64,
        requests: Vec<(Call, RequestDispatch)>,
        commands: Vec<BatchCommand>,
        input_transfers: Vec<TensorExport>,
    ) -> Self {
        Self {
            id,
            requests,
            commands,
            input_transfers,
            cache_inputs: Vec::new(),
            page_offloads: None,
            columns: None,
            media: None,
        }
    }

    /// Iterates over request admissions carried by batch commands.
    #[cfg(feature = "testing")]
    pub fn admissions(&self) -> impl Iterator<Item = &NewRequest> {
        self.commands.iter().filter_map(|command| match command {
            BatchCommand::Start { request } => Some(request.as_ref()),
            _ => None,
        })
    }

    /// Removes unstarted work for terminated request epochs while preserving
    /// independent calls.
    ///
    /// Drops the calls and `Start` commands of every request key in
    /// `requests`, then the input transfers and cache inputs no remaining call
    /// reads. The requests' `Finish` and `Free` commands stay, since they still
    /// retire physical state and reader obligations on the workers. Returns the
    /// removed calls with the dispatches describing their resources. The batch
    /// may be left with neither calls nor commands; callers decide whether to
    /// discard it.
    pub(crate) fn retire_requests(
        &mut self,
        requests: &std::collections::HashSet<RequestKey>,
    ) -> Vec<(Call, RequestDispatch)> {
        let (retired, active): (Vec<_>, Vec<_>) = std::mem::take(&mut self.requests)
            .into_iter()
            .partition(|(call, _)| requests.contains(&call.request_key));
        self.requests = active;
        self.commands.retain_mut(|command| {
            if !requests.contains(&command.request_key()) {
                return true;
            }
            !matches!(command, BatchCommand::Start { .. })
        });
        let inputs = self
            .requests
            .iter()
            .flat_map(|(call, _)| call.tensor_inputs().chain(call.predicate.iter()))
            .collect::<std::collections::HashSet<_>>();
        self.input_transfers
            .retain(|payload| inputs.contains(&payload.product));
        self.cache_inputs.retain(|export| {
            self.requests
                .iter()
                .any(|(call, _)| call.cache_input == Some(export.source))
        });
        retired
    }

    /// Validates the scheduler's own bookkeeping: call identities, execution
    /// ownership and dispatch.
    ///
    /// Requires at least one call, command or page write; checks that call
    /// identities are unique and belong to this batch, that each dispatch
    /// describes its own call, and that every `Start` is unique and names a
    /// request with a call in this batch.
    #[cfg(feature = "testing")]
    pub fn validate(&self) -> anyhow::Result<()> {
        // A batch of page offloads alone writes the engine's last published
        // pages to the host KV cache before it closes its workers.
        anyhow::ensure!(
            !self.requests.is_empty() || !self.commands.is_empty() || self.page_offloads.is_some(),
            "logical batch must carry at least one call, command or page write"
        );

        let mut requests = std::collections::HashSet::with_capacity(self.requests.len());
        let mut identities = std::collections::HashSet::with_capacity(self.requests.len());
        for (call, dispatch) in &self.requests {
            anyhow::ensure!(
                call.call_id.batch_id == self.id,
                "call identity belongs to another logical batch"
            );
            requests.insert(call.request_key);
            // `WorkerGroupId`'s public field admits any string; only its
            // constructor checks the identifier syntax.
            WorkerGroupId::new(dispatch.worker.0.clone())?;
            anyhow::ensure!(!call.component.is_empty(), "call requires a component");
            anyhow::ensure!(
                identities.insert(call.call_id),
                "logical batch repeats a call identity"
            );

            // A dispatch describes one call, so its forward rows may only
            // name local call index zero.
            dispatch.forward.validate(1)?;
            if let Some(latent) = &dispatch.latent {
                anyhow::ensure!(
                    (latent.request_key, latent.call_id) == (call.request_key, call.call_id),
                    "logical call carries another call's latent execution"
                );
            }
            if let Some(decode) = &dispatch.decode {
                anyhow::ensure!(
                    (decode.request_key, decode.call_id) == (call.request_key, call.call_id),
                    "logical call carries another call's decode execution"
                );
            }
            for buffer in &dispatch.buffers {
                anyhow::ensure!(
                    call.buffer_inputs()
                        .chain(call.buffer_outputs())
                        .any(|tensor| tensor.buffer_id() == buffer.allocation.buffer),
                    "logical call carries a buffer execution for an unrelated tensor"
                );
            }
        }

        let mut admitted = std::collections::HashSet::new();
        for admission in self.admissions() {
            anyhow::ensure!(
                admitted.insert(admission.request_key),
                "logical batch repeats a request start"
            );
            anyhow::ensure!(
                requests.contains(&admission.request_key),
                "logical batch starts a request without a call"
            );
        }
        Ok(())
    }
}
