//! Product storage lifetimes, imports and exports on host memory.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use cuda::Stream;
use ipc::{
    BufferAllocation, BufferId, Call, CallKind, DType as WireDType, DimBound, MediaCall,
    RequestKey, ShapeBound, TensorRef, TransferHandle, WorkerEndpoint,
};
use ipc::{CallId, RequestId};
use worker::ffi::MemoryKind;
use worker::memory::LocalMemory;
use worker::memory::{BufferPool, OutputPlacement, ProductBuffer, ProductStore, StorageView};
use worker::transfer::exports;
use worker::transfer::{TransferCapacity, TransportKind, TransportOptions, Transports};
use worker::{EventPool, Result};

const CPU: MemoryKind = MemoryKind::Host;

fn request() -> RequestKey {
    RequestKey::new(1, RequestId(1), 1)
}

fn reference(call: u32, output_index: u16, dtype: WireDType, dims: &[u32]) -> TensorRef {
    TensorRef {
        request_key: request(),
        producer_call_id: CallId::new(1, call),
        output_index,
        generation: 1,
        dtype,
        shape_bound: ShapeBound {
            dims: dims
                .iter()
                .map(|&extent| DimBound::Static(extent))
                .collect(),
        },
    }
}

fn allocation(product: &TensorRef, offset: u64, bytes: u64) -> BufferAllocation {
    BufferAllocation {
        buffer: product.buffer_id(),
        offset,
        bytes,
    }
}

fn store(pool_bytes: u64, request_capacity: usize, relay_depth: usize) -> Result<ProductStore> {
    ProductStore::new(
        1024,
        1024,
        vec![CPU],
        request_capacity,
        relay_depth,
        BufferPool::allocate(pool_bytes, &[CPU], &mut LocalMemory)?,
        EventPool::new(),
    )
}

fn transports(bytes: u64) -> Result<Transports> {
    let options = TransportOptions {
        source: WorkerEndpoint {
            host: "host".into(),
            address_space: "process".into(),
            incarnation: "incarnation".into(),
        },
        acknowledgment_slot: 0,
        host_slots: HashSet::new(),
        cross_host_consumers: false,
        vmm_pool_bytes: 1 << 20,
    };
    Transports::new(
        &[TransportKind::Local],
        Arc::new(TransferCapacity::new(bytes, 8)?),
        EventPool::new(),
        &options,
    )
}

fn fill(view: &StorageView, values: &[f32]) {
    assert!(view.is_contiguous() && view.numel() as usize == values.len());
    // SAFETY: host test storage reserved for this product, without readers.
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), view.address() as *mut f32, values.len());
    }
}

fn read(view: &StorageView) -> Vec<f32> {
    assert!(view.is_contiguous());
    // SAFETY: host test storage; every write completed.
    unsafe {
        std::slice::from_raw_parts(view.address() as *const f32, view.numel() as usize).to_vec()
    }
}

fn persistent(
    store: &ProductStore,
    product: &TensorRef,
    allocation: BufferAllocation,
    region: Option<Vec<Range<u64>>>,
) -> Result<ProductBuffer> {
    let allocations = HashMap::from([(product.buffer_id(), allocation)]);
    let regions: HashMap<BufferId, Vec<Range<u64>>> = region
        .into_iter()
        .map(|region| (product.buffer_id(), region))
        .collect();
    let mut buffers = store.bind_outputs(
        &[(product.clone(), CPU)],
        OutputPlacement {
            allocations: Some(&allocations),
            regions: Some(&regions),
            ..OutputPlacement::default()
        },
    )?;
    Ok(buffers.remove(0))
}

/// Write `values` into a bound product and commit it.
fn publish(
    store: &ProductStore,
    buffer: &ProductBuffer,
    values: &[f32],
    shape: &[u64],
) -> Result<()> {
    fill(&buffer.view(), values);
    store.produced(buffer, shape, None, None)?;
    store.commit_writes(std::slice::from_ref(buffer))
}

#[test]
fn released_storage_returns_only_after_its_last_reader() -> Result<()> {
    let store = store(256, 0, 0)?;
    let product = reference(1, 0, WireDType::F32, &[2, 3]);
    let buffer = persistent(&store, &product, allocation(&product, 0, 24), None)?;
    publish(&store, &buffer, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    let lease = store.consume(&product, CallId::new(2, 0), Some(CPU), None)?;
    assert_eq!(read(lease.value()), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    store.release_calls(&[(request(), product.producer_call_id)]);
    assert!(
        store
            .consume(&product, CallId::new(3, 0), Some(CPU), None)
            .is_err()
    );

    // The released range stays occupied while the lease is open.
    let next = reference(4, 0, WireDType::F32, &[2, 3]);
    assert!(persistent(&store, &next, allocation(&next, 0, 24), None).is_err());
    let ready = |store: &ProductStore| {
        store.retirement_ready(
            &HashSet::from([product.buffer_id()]),
            &HashSet::new(),
            &HashSet::new(),
        )
    };
    assert!(!ready(&store)?);

    store.complete_reads(&[Arc::clone(&lease)], None, &[])?;
    assert!(ready(&store)?);
    let reused = persistent(&store, &next, allocation(&next, 0, 24), None)?;
    assert_eq!(reused.view().address(), buffer.view().address());
    // The completed lease's view still reads its value.
    assert!(lease.complete());
    Ok(())
}

#[test]
fn deferred_writes_stay_undiscoverable_until_produced_and_committed() -> Result<()> {
    let store = store(256, 0, 0)?;
    let product = reference(1, 0, WireDType::F32, &[2]);
    let buffer = persistent(&store, &product, allocation(&product, 0, 8), None)?;
    store.defer_write(&buffer)?;
    store.commit_writes(std::slice::from_ref(&buffer))?;
    assert!(
        store
            .consume(&product, CallId::new(2, 0), None, None)
            .is_err()
    );

    publish(&store, &buffer, &[7.0, 8.0], &[2])?;
    let lease = store.consume(&product, CallId::new(2, 0), None, None)?;
    assert_eq!(read(lease.value()), vec![7.0, 8.0]);
    Ok(())
}

#[test]
fn a_failed_group_binding_releases_every_group_of_the_call() -> Result<()> {
    let store = store(256, 2, 1)?;
    let product = reference(1, 0, WireDType::F32, &[2]);
    let scalar = reference(1, 1, WireDType::I32, &[]);
    let allocations = HashMap::from([(product.buffer_id(), allocation(&product, 0, 8))]);
    // The relay output names no request slot.
    let slots = HashMap::new();
    let groups = vec![vec![(product.clone(), CPU)], vec![(scalar, CPU)]];
    let placement = OutputPlacement {
        request_slots: Some(&slots),
        allocations: Some(&allocations),
        ..OutputPlacement::default()
    };
    assert!(store.bind_output_groups(&groups, placement).is_err());

    // The first group's range was returned.
    persistent(&store, &product, allocation(&product, 0, 8), None)?;
    Ok(())
}

#[test]
fn relay_entries_are_bounded_by_the_unresolved_window() -> Result<()> {
    let store = store(256, 2, 1)?;
    let slots = HashMap::from([(request(), 1)]);
    let placement = OutputPlacement {
        request_slots: Some(&slots),
        ..OutputPlacement::default()
    };
    let first = reference(1, 0, WireDType::I32, &[]);
    let buffer = store
        .bind_outputs(&[(first.clone(), CPU)], placement)?
        .remove(0);
    store.write_scalar(&buffer, -3, None::<&Stream>)?;
    store.commit_writes(std::slice::from_ref(&buffer))?;

    let lease = store.consume(&first, CallId::new(5, 0), None, None)?;
    // SAFETY: host relay element written above.
    assert_eq!(unsafe { *(lease.value().address() as *const i32) }, -3);

    let second = reference(2, 0, WireDType::I32, &[]);
    assert!(
        store
            .bind_outputs(&[(second.clone(), CPU)], placement)
            .is_err()
    );
    store.complete_reads(&[lease], None, &[])?;
    store.release_calls(&[(request(), first.producer_call_id)]);
    store.release_requests(&HashSet::new(), &HashSet::new(), None::<&cuda::Stream>)?;
    store.bind_outputs(&[(second, CPU)], placement)?;
    Ok(())
}

fn call(consumers: Vec<u32>) -> Call {
    Call {
        request_key: request(),
        call_id: CallId::new(1, 0),
        coordinates: Default::default(),
        component: "image".into(),
        kind: CallKind::Media(MediaCall::ImageDecoding),
        bounds: Default::default(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        consumer_slots: consumers,
        token_input: None,
        token_output: None,
        vision_inputs: Vec::new(),
        latent_feature_input: None,
        encoder_output: None,
        latent_input: None,
        latent_output: None,
        image_input: None,
        image_output: None,
        completion_output: None,
        transition_output: None,
        predicate: None,
        image_index: None,
        sampling_state: None,
        input_token_ids: Vec::new(),
        readout: None,
        canvas: None,
        input_image: None,
        cache_input: None,
        cache_output: None,
        latent_restore: None,
    }
}

#[test]
fn shard_exports_carry_their_logical_origin_and_failures_return_bytes() -> Result<()> {
    let store = store(1024, 0, 0)?;
    let transports = transports(64)?;
    let product = reference(1, 0, WireDType::F32, &[4, 3]);
    // A shard holding rows [2, 4) of the logical [4, 3] tensor.
    let buffer = persistent(
        &store,
        &product,
        allocation(&product, 0, 24),
        Some(vec![2..4, 0..3]),
    )?;
    publish(&store, &buffer, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;

    let export = store.export_buffer(
        &call(vec![]),
        &product,
        &buffer,
        &transports,
        false,
        None,
        None,
    )?;
    let (export, locations) = store.located(&buffer, export)?;
    let TransferHandle::DeviceProduct { tensor, .. } = &export.value else {
        panic!("a product exports as a device product");
    };
    assert_eq!(tensor.shape, vec![4, 3]);
    assert_eq!(tensor.locations[0].offset, vec![2, 0]);
    assert_eq!(tensor.locations[0].shape, vec![2, 3]);
    assert_eq!(locations.len(), 1);
    assert_eq!(transports.capacity().used(), 24);

    // The second region does not fit the remaining bytes: the first
    // accepted export is revoked and its bytes return.
    let regions = vec![vec![0..2, 0..3], vec![0..2, 0..3]];
    assert!(
        store
            .export_buffer(
                &call(vec![]),
                &product,
                &buffer,
                &transports,
                false,
                Some(&regions),
                None
            )
            .is_err()
    );
    assert_eq!(transports.capacity().used(), 24);
    exports::release(&locations)?;
    assert_eq!(transports.capacity().used(), 0);
    Ok(())
}

#[test]
fn imports_share_pending_reads_and_widen_resident_shards() -> Result<()> {
    let producer = store(1024, 0, 0)?;
    let consumer = store(1024, 0, 0)?;
    let transports = transports(1024)?;
    let product = reference(1, 0, WireDType::F32, &[4, 3]);
    let values: Vec<f32> = (0..12).map(|value| value as f32).collect();
    let source = persistent(&producer, &product, allocation(&product, 0, 48), None)?;
    publish(&producer, &source, &values, &[4, 3])?;
    let export = producer.export_buffer(
        &call(vec![]),
        &product,
        &source,
        &transports,
        false,
        None,
        None,
    )?;
    let (export, _locations) = producer.located(&source, export)?;
    let TransferHandle::DeviceProduct { tensor, .. } = &export.value else {
        panic!("a product exports as a device product");
    };

    // The consumer holds rows [0, 2) and its allocation covers the tensor.
    let shard = persistent(
        &consumer,
        &product,
        allocation(&product, 0, 48),
        Some(vec![0..2, 0..3]),
    )?;
    publish(&consumer, &shard, &values[..6], &[2, 3])?;

    let allocations = HashMap::from([(product.buffer_id(), allocation(&product, 0, 48))]);
    let placement = OutputPlacement {
        allocations: Some(&allocations),
        ..OutputPlacement::default()
    };
    let first =
        consumer.import_tensor(&product, tensor, CPU, &transports, placement, None, None)?;
    let second =
        consumer.import_tensor(&product, tensor, CPU, &transports, placement, None, None)?;
    assert_eq!(first.value().address(), second.value().address());
    // Only the missing rows are read.
    assert_eq!(first.reads().len(), 1);
    for read in first.reads() {
        read.start()?;
        read.wait_ready();
    }
    consumer.complete_import(&first, None)?;
    assert_eq!(read(first.value()), values);

    consumer.complete_reads(&[first, second], None, &[])?;
    let lease = consumer.consume(&product, CallId::new(2, 0), None, None)?;
    assert_eq!(lease.value().shape(), &[4, 3]);
    assert_eq!(read(lease.value()), values);
    Ok(())
}

#[test]
fn a_released_import_leaves_its_product_importable() -> Result<()> {
    let producer = store(1024, 0, 0)?;
    let consumer = store(1024, 0, 0)?;
    let transports = transports(1024)?;
    let product = reference(1, 0, WireDType::F32, &[2, 3]);
    let values: Vec<f32> = (0..6).map(|value| value as f32).collect();
    let source = persistent(&producer, &product, allocation(&product, 0, 24), None)?;
    publish(&producer, &source, &values, &[2, 3])?;
    let export = producer.export_buffer(
        &call(vec![]),
        &product,
        &source,
        &transports,
        false,
        None,
        None,
    )?;
    let (export, _locations) = producer.located(&source, export)?;
    let TransferHandle::DeviceProduct { tensor, .. } = &export.value else {
        panic!("a product exports as a device product");
    };
    let allocations = HashMap::from([(product.buffer_id(), allocation(&product, 0, 24))]);
    let placement = OutputPlacement {
        allocations: Some(&allocations),
        ..OutputPlacement::default()
    };
    let import = |placement| -> Result<Arc<worker::memory::ProductRead>> {
        let read =
            consumer.import_tensor(&product, tensor, CPU, &transports, placement, None, None)?;
        for transfer in read.reads() {
            transfer.start()?;
            transfer.wait_ready();
        }
        consumer.complete_import(&read, None)?;
        Ok(read)
    };

    // A consumer reads the imported copy, which is then released (a
    // consumed predicate is) after its reader finished. Nothing has
    // reclaimed it yet: its accesses retired, but no binding has run since.
    let first = import(placement)?;
    consumer.complete_reads(std::slice::from_ref(&first), None, &[])?;
    consumer.release_calls(&[(request(), product.producer_call_id)]);

    // A later call reading the same product imports it again.
    let second = import(placement)?;
    assert_eq!(read(second.value()), values);

    // Released while that reader still holds it, as a copy whose fences are
    // pending is, the copy is not retired; the next reader revives it and
    // reads its value where it lies.
    consumer.release_calls(&[(request(), product.producer_call_id)]);
    let third =
        consumer.import_tensor(&product, tensor, CPU, &transports, placement, None, None)?;
    assert_eq!(third.value().address(), second.value().address());
    assert_eq!(read(third.value()), values);
    consumer.complete_reads(&[second, third], None, &[])?;
    Ok(())
}
