//! Logical submissions.
//!
//! The scheduler builds an [`ExecutionBatch`] of calls, each paired with the
//! [`RequestDispatch`] it chose, plus lifecycle commands, and lays it out once
//! as the `ipc::execution::Batch` the device worker reads
//! ([`ExecutionBatch::device`]). `Submissions` hands that batch to the device
//! worker, stamped with the step's generation, or gives the whole record to
//! the simulated one.

use super::WorkerGroupId;
use ipc::{BatchCommand, BufferAllocation, Call, DecodeRange, NewRequest, RequestKey};
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
/// The call itself is the shared IPC `Call`. These fields describe its worker
/// and the allocations the mechanism retires with it. What the call states to
/// the device — its rows, cache tables, fresh units and page loads — is not
/// here: it is written once into the batch's device layout
/// ([`ExecutionBatch::device`]).
#[derive(Debug, Clone, PartialEq)]
pub struct RequestDispatch {
    /// Worker routing stays outside the call sent across IPC.
    pub worker: WorkerGroupId,
    /// Worker-local request row when a replicated route owns its own address
    /// space. Unset dispatches retain the admission's canonical row;
    /// The worker applies the override to the request's `Start`.
    pub request_pool_idx: Option<u32>,
    pub decode: Option<DecodeRange>,
    /// Persistent spans for the call's buffer inputs and outputs, on each
    /// rank of the worker that backs them. Request retirement retains every
    /// physical reader and writer allocation.
    pub buffers: Vec<RankBuffer>,
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
    /// Published pages the batch writes to the host KV cache.
    pub page_offloads: Option<PageOffloads>,
    /// The batch as the device worker reads it, laid out once when the
    /// mechanism assembles the batch: its phase, shape and host columns (a
    /// token batch's typed columns, or a media computation's
    /// `MediaColumns`), every call's cache tables, fresh units and page
    /// loads, and the batch's page offloads.
    ///
    /// Its `calls` number this record's `requests` by index, and its
    /// `generation` is stamped at submission, since the step reads it then.
    /// `None` means the mechanism could not state the layout — a batch no
    /// worker may serve, refused at submission by name rather than served on
    /// invented values.
    pub device: Option<ipc::execution::Batch>,
}

/// Batches compare by what the mechanism decided: their identity, their
/// calls and their commands. The device layout is that decision's projection
/// into the worker's layout (equal calls project equally, and a batch with
/// none is the unservable one), so equality states only whether both carry it.
impl PartialEq for ExecutionBatch {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.requests == other.requests
            && self.commands == other.commands
            && self.page_offloads == other.page_offloads
            && self.device.is_some() == other.device.is_some()
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
    ) -> Self {
        Self {
            id,
            requests,
            commands,
            page_offloads: None,
            device: None,
        }
    }

    /// The captured shape of the batch's token columns, which a shadow pass
    /// compares by: a token batch's own shape, and a text encoding's prompt
    /// columns'. `None` for a batch with no layout and for a media
    /// computation that reads no token column.
    pub(crate) fn token_shape(&self) -> Option<ipc::execution::Shape> {
        let device = self.device.as_ref()?;
        match &device.media {
            None => Some(device.shape),
            Some(media) if media.computation.call == ipc::MediaCall::TextEncoding => {
                Some(crate::batching::columns::batch_shape(&device.columns))
            }
            Some(_) => None,
        }
    }

    /// Iterates over request admissions carried by batch commands.
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
    /// `requests`. The requests' `Finish` and `Free` commands stay, since they still
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
        retired
    }

    /// Validates the scheduler's own bookkeeping: call identities, execution
    /// ownership and dispatch.
    ///
    /// Requires at least one call, command or page write; checks that call
    /// identities are unique and belong to this batch, that each dispatch
    /// describes its own call, and that every `Start` is unique and names a
    /// request with a call in this batch.
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
