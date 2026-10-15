//! A TVM-FFI test harness that drives the native cache owners with the real
//! numerical library: Python loads this library with
//! `tvm_ffi.load_module`, passes the rank's import function
//! (`uniserve.worker.import_path`) and its cache plans
//! and inspects the Rust-owned storage through the returned DLPack tensors
//! (`tests/python/integration/storage`).
//!
//! `cache_transfer` builds a producer and a consumer cache, lets Python write
//! the producer's pages, exports the producer's slot through one transport,
//! imports it into the consumer (converting when the layouts differ) and
//! adopts it, then retires every access and returns both caches' tensors.

#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "harness/config.rs"]
mod config;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use config::Config;
use tvm_ffi::{Any, AnyView, Array, Map, String as FfiString};
use uniserve_core::{CallId, RequestId, UnitId};
use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{BlockTable, BufferId, CallKind, ForwardMode, RequestKey, WorkerEndpoint};
use uniserve_worker::ffi::{Function, Handle, Interpreter};
use uniserve_worker::lane::JobQueues;
use uniserve_worker::storage::memory::StorageView;
use uniserve_worker::storage::{CachePlan, PrefixCache};
use uniserve_worker::transfer::{Capacity, TransportKind, TransportOptions, Transports, notify};
use uniserve_worker::{EventPool, Result};

/// Named views as the map of DLPack tensors the test reads.
fn tensor_map(views: &[(String, StorageView)]) -> Result<Map<FfiString, tvm_ffi::Tensor>> {
    views
        .iter()
        .map(|(name, view)| Ok((FfiString::from(name.as_str()), view.to_tensor()?)))
        .collect()
}

fn failure(error: impl std::fmt::Display) -> tvm_ffi::Error {
    tvm_ffi::Error::new(tvm_ffi::RUNTIME_ERROR, &error.to_string(), "")
}

fn units(value: Any) -> Result<Vec<Vec<UnitId>>> {
    let groups = Array::<Any>::try_from(value)
        .map_err(|error| uniserve_worker::Error::numerical("units", &error))?;
    groups
        .iter()
        .map(|group| {
            let ids = Array::<Any>::try_from(group)
                .map_err(|error| uniserve_worker::Error::numerical("units", &error))?;
            ids.iter()
                .map(|id| {
                    i64::try_from(id)
                        .map(|id| UnitId(id as u32))
                        .map_err(|error| uniserve_worker::Error::numerical("units", &error))
                })
                .collect()
        })
        .collect()
}

fn tables(slot: u32, groups: &[Vec<UnitId>], tokens: u32, plan: &CachePlan) -> Vec<BlockTable> {
    groups
        .iter()
        .enumerate()
        .map(|(group, unit_ids)| BlockTable {
            request_pool_idx: slot,
            group_id: group as u32,
            start_page: 0,
            unit_ids: unit_ids.clone(),
            allocated_tokens: tokens.min(
                unit_ids.len() as u32 / plan.groups[group].units_per_page
                    * plan.groups[group].page_tokens,
            ),
        })
        .collect()
}

fn endpoint() -> WorkerEndpoint {
    WorkerEndpoint {
        worker_id: "harness".into(),
        rank: 0,
        node: "node".into(),
        address_space: "harness".into(),
        incarnation: "harness".into(),
    }
}

fn buffer(output_index: u16) -> BufferId {
    BufferId {
        owner: RequestKey::new(1, RequestId(1), 1),
        producer_call_id: CallId::new(1, 0),
        output_index,
        generation: 1,
    }
}

fn wait(mut done: impl FnMut() -> Result<bool>, what: &str) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(60);
    while !done()? {
        if Instant::now() > deadline {
            return Err(uniserve_worker::Error::Invariant(format!(
                "timed out waiting for {what}"
            )));
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}

fn cache_transfer(
    config: Map<FfiString, Any>,
) -> std::result::Result<Map<FfiString, Any>, tvm_ffi::Error> {
    run(Any::from(config)).map_err(failure)
}

fn run(config: Any) -> Result<Map<FfiString, Any>> {
    let config = Config::new("cache_transfer", config)?;
    Interpreter::install_import(config.get("import")?);
    let plan = |key: &str| -> Result<Arc<CachePlan>> {
        CachePlan::read(Handle::new(config.value(key)?)?).map(Arc::new)
    };
    let source_plan = plan("source_plan")?;
    let destination_plan = plan("destination_plan")?;
    let num_units = config.count("num_units")? as u32;
    let tokens = config.count("tokens")? as u32;
    let devices = [
        config.count("source_device")? as i32,
        config.count("destination_device")? as i32,
    ];
    let kind = TransportKind::parse(&config.string("transport")?)?;
    let fill: Function = Function::new("fill", config.get("fill")?);
    let source_units = units(config.value("source_units")?)?;
    let destination_units = units(config.value("destination_units")?)?;

    let streams = [
        Stream::new(DeviceId(devices[0]))?,
        Stream::new(DeviceId(devices[1]))?,
    ];
    let width = 64;
    let mut source = PrefixCache::new(
        Arc::clone(&source_plan),
        num_units,
        2,
        width,
        1,
        Some(devices[0]),
        Some(&streams[0]),
    )?;
    let mut destination = PrefixCache::new(
        Arc::clone(&destination_plan),
        num_units,
        2,
        width,
        2,
        Some(devices[1]),
        Some(&streams[1]),
    )?;

    // Python writes the producer's tokens through the bound views.
    source.install(
        &tables(1, &source_units, tokens, &source_plan),
        Some(&streams[0]),
    )?;
    streams[0].synchronize()?;
    let source_views = tensor_map(&source.views()?)?;
    fill.invoke(&[AnyView::from(&source_views)])?;

    // The harness is its own control thread: a lane of its own waits for
    // the events that deferred releases follow, and the harness hands the
    // pool that lane's completions while it waits for retirement.
    let events = EventPool::new();
    let jobs = JobQueues::single_lane()?;
    let lane = jobs.clone();
    events.set_submitter(Some(Box::new(move |device, events| {
        lane.after(CallKind::Forward(ForwardMode::Decode), device, events)
    })));

    let capacity = Arc::new(Capacity::new(1 << 28, 16, notify)?);
    let transports = Arc::new(Transports::new(
        &[kind],
        Arc::clone(&capacity),
        Arc::clone(&events),
        &TransportOptions {
            source: endpoint(),
            acknowledgment_slot: 0,
            host_slots: HashSet::from([0]),
            cross_host_consumers: false,
            vmm_pool_bytes: 1 << 26,
        },
    )?);

    let exported = buffer(0);
    // Issued on the stream at once, so the export is located here.
    let export = source.export(
        1,
        tokens,
        "consumer",
        exported,
        &transports,
        &[0],
        Some(&streams[0]),
    )?;
    let (transfer, locations) = source.locate_export(export)?;
    source.validate_transfers(&[(exported, transfer.clone())], &[])?;
    source.apply_transfers(vec![(exported, transfer.clone())], vec![]);

    // The consumer assigns fresh units and imports the whole extent.
    let assignments = tables(1, &destination_units, tokens, &destination_plan);
    let batch_tables = destination.tables().for_batch(1, &assignments)?;
    let initialized: Vec<u32> = destination_units
        .iter()
        .flatten()
        .map(|unit| unit.0)
        .collect();
    let import = destination.reserve_import(
        Arc::new(transfer.clone()),
        1,
        batch_tables,
        initialized,
        Arc::clone(&transports),
    )?;
    wait(|| Ok(import.done()), "the import copy")?;
    destination.install(&assignments, Some(&streams[1]))?;
    let installed = BufferId {
        output_index: 1,
        ..exported
    };
    destination.adopt_import(&import, installed, Some(&streams[1]))?;
    streams[1].synchronize()?;

    // Retire every access: the producer's exports, then both caches.
    source.release_calls(&[(exported.owner, exported.producer_call_id)]);
    source.release_exports(&[exported]);
    uniserve_worker::transfer::exports::release(&locations)?;
    wait(
        || {
            events.submit_deferred()?;
            while let Some((job, completion)) = jobs.take() {
                events.complete(job, completion)?;
            }
            transports.reap()?;
            destination.reap_imports()?;
            Ok(capacity.used() == 0 && import.retirement().done())
        },
        "transfer retirement",
    )?;
    let verified_tokens = i64::from(transfer.exported_extent);
    destination.close()?;
    source.close()?;
    transports.close()?;

    let destination_views = tensor_map(&destination.views()?)?;
    let result: Map<FfiString, Any> = [
        (FfiString::from("source"), Any::from(source_views)),
        (FfiString::from("destination"), Any::from(destination_views)),
        (
            FfiString::from("exported_extent"),
            Any::from(verified_tokens),
        ),
        (
            FfiString::from("transfer"),
            Any::from(FfiString::from(
                serde_json::to_string(&transfer).unwrap().as_str(),
            )),
        ),
    ]
    .into_iter()
    .collect();
    Ok(result)
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(cache_transfer, cache_transfer);
