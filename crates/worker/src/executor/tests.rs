//! The executor's admission rules (batch ids, queue capacity, collective
//! sequences) and the retirement of lifecycle commands, observed through
//! the results it delivers.
//!
//! The rank under test holds no component: it serves lifecycle batches
//! (Start, Finish and Free commands), whose forwards submit no job, so each
//! batch runs from its admission to its result on the calling thread. The
//! retirement cases give the rank its storage owners on host memory: the
//! product store holds the request products whose readers a retirement
//! waits for.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use uniserve_core::{CallId, RequestId, SamplingParams};
use uniserve_ipc::{
    Batch as Plan, BatchCommand, BufferAllocation, DType as WireDType, DimBound, GenerationParams,
    NewRequest, RequestKey, ShapeBound, TensorRef, WorkerInfo,
};

use super::backend::{WorkerBackend, WorkerBatch};
use super::{Batch, Executor, Submission};
use crate::config::placement::{Components, GroupPlacement};
use crate::error::WorkerError;
use crate::ffi::MemoryKind;
use crate::storage::{BufferPool, OutputPlacement, ProductRead, ProductStore};
use crate::transfer::{TransferCapacity, TransportKind, TransportOptions, Transports};
use crate::worker::RankStorage;
use crate::{Error, ErrorCode, EventPool};

type Result<T> = std::result::Result<T, WorkerError>;

/// Request slots of the rank under test.
const SLOTS: u32 = 4;

/// An executor over a rank without components, admitting at most
/// `capacity` unconsumed batches. A `distributed` rank orders its
/// collective sequences.
pub(crate) fn rank(capacity: usize, distributed: bool) -> Result<Executor> {
    let info = WorkerInfo {
        request_slots: SLOTS,
        ..WorkerInfo::default()
    };
    let placement = GroupPlacement {
        world_size: 1,
        ..GroupPlacement::default()
    };
    let backend = WorkerBackend::new(info, 0, placement, Components::default())?;
    Executor::new(backend, capacity, distributed, false)
}

/// A lifecycle batch of `commands`.
fn lifecycle(id: u64, commands: Vec<BatchCommand>) -> Batch {
    let mut plan = Plan::new(id, Vec::new(), Vec::new());
    plan.commands = commands;
    let plan = Arc::new(plan);
    Batch::from_plan(&plan, WorkerBatch::new(Arc::clone(&plan)))
}

/// A batch without commands that carries collective sequence `sequence`:
/// the executor orders it as it orders a batch of calls on a distributed
/// rank.
fn collective(id: u64, sequence: u64) -> Batch {
    let plan = Arc::new(Plan::new(id, Vec::new(), Vec::new()));
    Batch::new(
        id,
        Some(sequence),
        HashSet::new(),
        HashSet::new(),
        WorkerBatch::new(plan),
    )
}

/// Epoch `epoch` of request `id`.
fn key(id: u64, epoch: u64) -> RequestKey {
    RequestKey::new(1, RequestId(id), epoch)
}

/// The Start of a text request in `slot`.
fn start(key: RequestKey, slot: u32) -> Result<BatchCommand> {
    let generation = GenerationParams {
        sampling: SamplingParams::default(),
        negative_token_ids: Vec::new(),
        finish_token_ids: Vec::new(),
        initial_position: 0,
    };
    let request = NewRequest::new(key, slot, Some(generation), None, 0)
        .map_err(|error| Error::Invalid(error.to_string()))?;
    Ok(BatchCommand::Start {
        request: Box::new(request),
    })
}

fn finish(key: RequestKey, retained: &[&TensorRef]) -> BatchCommand {
    BatchCommand::Finish {
        request_key: key,
        retained_buffers: retained.iter().map(|product| product.buffer_id()).collect(),
    }
}

fn free(product: &TensorRef) -> BatchCommand {
    BatchCommand::Free {
        buffer: product.buffer_id(),
    }
}

/// Admit `batch` and take its result: whether the batch succeeded, or
/// `None` while it has none.
fn run(executor: &mut Executor, batch: Batch) -> Result<Option<bool>> {
    let submission = executor.admit(batch, false)?;
    delivered(executor, &submission)
}

/// The result of `submission` after the executor advanced: whether the
/// batch succeeded, or `None` while it has none.
fn delivered(executor: &mut Executor, submission: &Arc<Submission>) -> Result<Option<bool>> {
    executor.advance()?;
    Ok(match executor.poll(submission) {
        Ok(Some(_)) => Some(true),
        Ok(None) => None,
        Err(_) => Some(false),
    })
}

#[test]
fn batch_ids_increase_and_a_refused_admission_consumes_no_id() -> Result<()> {
    let mut executor = rank(1, false)?;
    assert_eq!(run(&mut executor, lifecycle(2, Vec::new()))?, Some(true));

    // An id at or below the last admitted one is refused.
    for id in [2, 1] {
        let refused = executor.admit(lifecycle(id, Vec::new()), false);
        assert_eq!(
            refused.err().map(|error| error.code),
            Some(ErrorCode::InvalidDescriptor)
        );
    }

    // A full queue refuses a batch without consuming its id: the batch is
    // admitted under that id once the queued result was taken.
    let queued = executor.admit(lifecycle(3, Vec::new()), false)?;
    let full = executor.admit(lifecycle(4, Vec::new()), false);
    assert_eq!(
        full.err().map(|error| error.code),
        Some(ErrorCode::ResourceError)
    );
    assert_eq!(delivered(&mut executor, &queued)?, Some(true));
    assert_eq!(run(&mut executor, lifecycle(4, Vec::new()))?, Some(true));

    // A failed batch consumes its id: its request's slot lies outside
    // the pool.
    let failed = lifecycle(5, vec![start(key(1, 1), SLOTS + 1)?]);
    assert_eq!(run(&mut executor, failed)?, Some(false));
    assert!(executor.admit(lifecycle(5, Vec::new()), false).is_err());
    assert_eq!(run(&mut executor, lifecycle(6, Vec::new()))?, Some(true));
    Ok(())
}

#[test]
fn a_distributed_rank_refuses_a_collective_sequence_that_does_not_advance() -> Result<()> {
    // Another rank may already be inside collective 5: a repeated or an
    // earlier sequence fails its batch with an error that stops the rank.
    for refused in [5, 4] {
        let mut executor = rank(4, true)?;
        assert_eq!(run(&mut executor, collective(1, 5))?, Some(true));
        let submission = executor.admit(collective(2, refused), false)?;
        executor.advance()?;
        let Err(error) = executor.poll(&submission) else {
            return Err(Error::State("a collective sequence that does not advance ran").into());
        };
        assert_eq!(error.code, ErrorCode::InvariantViolation);
        assert!(error.fatal);
    }

    // Advancing sequences run, and lifecycle batches, which carry none,
    // pass between them.
    let mut executor = rank(4, true)?;
    assert_eq!(run(&mut executor, collective(1, 5))?, Some(true));
    assert_eq!(run(&mut executor, lifecycle(2, Vec::new()))?, Some(true));
    assert_eq!(run(&mut executor, collective(3, 6))?, Some(true));

    // A single rank takes part in no collective: it does not order them.
    let mut single = rank(4, false)?;
    assert_eq!(run(&mut single, collective(1, 5))?, Some(true));
    assert_eq!(run(&mut single, collective(2, 5))?, Some(true));
    Ok(())
}

/// A host rank holding a product store and local transports.
fn host_rank(capacity: usize) -> Result<Executor> {
    let mut executor = rank(capacity, false)?;
    let events = EventPool::new();
    let tensors = ProductStore::new(
        1024,
        1024,
        vec![MemoryKind::Host],
        0,
        0,
        BufferPool::allocate(1024, &[MemoryKind::Host])?,
        Arc::clone(&events),
    )?;
    let options = TransportOptions {
        source: WorkerInfo::default().endpoint,
        acknowledgment_slot: 0,
        host_slots: HashSet::new(),
        cross_host_consumers: false,
        vmm_pool_bytes: 1 << 20,
    };
    let transports = Transports::new(
        &[TransportKind::Local],
        Arc::new(TransferCapacity::new(1 << 12, 8)?),
        Arc::clone(&events),
        &options,
    )?;
    executor.backend_mut().set_rank_storage(RankStorage::new(
        events,
        tensors,
        Arc::new(transports),
        None,
        None,
    ));
    Ok(executor)
}

fn tensors(executor: &Executor) -> Result<&ProductStore> {
    let storage = executor
        .backend()
        .rank_storage()
        .ok_or(Error::State("the rank has no storage owners"))?;
    Ok(&storage.tensors)
}

/// Output `index` of call 1 of `owner`: two f32 values, stored at
/// `64 * index` in the buffer pool.
fn product(owner: RequestKey, index: u16) -> TensorRef {
    TensorRef {
        request_key: owner,
        producer_call_id: CallId::new(1, 0),
        output_index: index,
        generation: 1,
        dtype: WireDType::F32,
        shape_bound: ShapeBound {
            dims: vec![DimBound::Static(2)],
        },
    }
}

/// Write and commit `product`, as its producing call would.
fn publish(executor: &Executor, product: &TensorRef) -> Result<()> {
    let store = tensors(executor)?;
    let allocations = HashMap::from([(
        product.buffer_id(),
        BufferAllocation {
            buffer: product.buffer_id(),
            offset: 64 * u64::from(product.output_index),
            bytes: 8,
        },
    )]);
    let buffers = store.bind_outputs(
        &[(product.clone(), MemoryKind::Host)],
        OutputPlacement {
            allocations: Some(&allocations),
            ..OutputPlacement::default()
        },
    )?;
    store.produced(&buffers[0], &[2], None, None)?;
    store.commit_writes(&buffers)?;
    Ok(())
}

/// A read of `product` by a later call, which holds its storage until the
/// read completes.
fn read(executor: &Executor, product: &TensorRef) -> Result<Arc<ProductRead>> {
    Ok(tensors(executor)?.consume(
        product,
        CallId::new(9, 0),
        Some(MemoryKind::Host),
        None::<&uniserve_cuda::Stream>,
    )?)
}

fn complete(executor: &Executor, read: &Arc<ProductRead>) -> Result<()> {
    Ok(tensors(executor)?.complete_reads(
        std::slice::from_ref(read),
        None::<&uniserve_cuda::Stream>,
        &[],
    )?)
}

/// Whether a later call can read `product`.
fn readable(executor: &Executor, product: &TensorRef) -> Result<bool> {
    match read(executor, product) {
        Ok(lease) => complete(executor, &lease).map(|()| true),
        Err(_) => Ok(false),
    }
}

#[test]
fn a_finished_request_retires_after_its_readers_and_retains_products_until_free() -> Result<()> {
    let mut executor = host_rank(4)?;
    let first = key(1, 1);
    assert_eq!(
        run(&mut executor, lifecycle(1, vec![start(first, 1)?]))?,
        Some(true)
    );
    let [released, retained] = [product(first, 0), product(first, 1)];
    publish(&executor, &released)?;
    publish(&executor, &retained)?;
    let reader = read(&executor, &released)?;

    // The finish revokes the products it does not retain at once; its
    // epoch retires once the read of them completed.
    let finished = executor.admit(lifecycle(2, vec![finish(first, &[&retained])]), false)?;
    assert_eq!(delivered(&mut executor, &finished)?, None);
    assert!(!readable(&executor, &released)?);
    assert!(readable(&executor, &retained)?);

    // Until then its slot and the revoked product's storage are occupied:
    // another request cannot start in the slot, nor a product take the
    // storage.
    let second = key(2, 1);
    assert_eq!(
        run(&mut executor, lifecycle(3, vec![start(second, 1)?]))?,
        Some(false)
    );
    let successor = product(second, 0);
    assert!(publish(&executor, &successor).is_err());

    complete(&executor, &reader)?;
    assert_eq!(delivered(&mut executor, &finished)?, Some(true));
    assert_eq!(
        run(&mut executor, lifecycle(4, vec![start(second, 1)?]))?,
        Some(true)
    );
    publish(&executor, &successor)?;

    // The retained product outlives its request until its own Free.
    assert!(readable(&executor, &retained)?);
    assert_eq!(
        run(&mut executor, lifecycle(5, vec![free(&retained)]))?,
        Some(true)
    );
    assert!(!readable(&executor, &retained)?);
    Ok(())
}

#[test]
fn a_remote_epoch_and_a_product_freed_with_its_finish_are_released() -> Result<()> {
    let mut executor = host_rank(4)?;
    // Epoch 2 of request 1 never started on this rank, yet owns products
    // here (another rank's call produced them).
    let remote = key(1, 2);
    let local = key(3, 1);
    assert_eq!(
        run(&mut executor, lifecycle(1, vec![start(local, 2)?]))?,
        Some(true)
    );
    let [owned, freed, kept] = [product(remote, 0), product(remote, 1), product(local, 2)];
    for product in [&owned, &freed, &kept] {
        publish(&executor, product)?;
    }

    // Finishing it releases its products; one freed in the same batch is
    // not retained by its listing. The local request is untouched.
    let commands = vec![finish(remote, &[&freed]), free(&freed)];
    assert_eq!(run(&mut executor, lifecycle(2, commands))?, Some(true));
    assert!(!readable(&executor, &owned)?);
    assert!(!readable(&executor, &freed)?);
    assert!(readable(&executor, &kept)?);
    Ok(())
}
