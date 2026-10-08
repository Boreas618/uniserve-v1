//! A TVM-FFI test harness that drives the native KV owners with the real
//! numerical interface: Python loads this library with
//! `tvm_ffi.load_module`, passes `uniserve_models.interface.cache` functions
//! and inspects the Rust-owned storage through the returned DLPack tensors
//! (`tests/python/integration/storage`).
//!
//! `kv_transfer` builds a producer and a consumer cache, lets Python write
//! the producer's pages, exports the producer's slot through one transport,
//! imports it into the consumer (converting when the layouts differ) and
//! adopts it, then retires every access and returns both caches' tensors.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tvm_ffi::{Any, AnyView, Array, Map, String as FfiString};
use uniserve_core::{CallId, RequestId, UnitId};
use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{BlockTable, BufferId, RequestKey, WorkerEndpoint};
use uniserve_worker::ffi::{Function, Record};
use uniserve_worker::storage::{KVCache, KVPlan, tensor_map};
use uniserve_worker::transfer::{Capacity, TransportKind, TransportOptions, Transports, notify};
use uniserve_worker::{EventPool, Result};

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

fn tables(slot: u32, groups: &[Vec<UnitId>], tokens: u32, plan: &KVPlan) -> Vec<BlockTable> {
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

fn kv_transfer(
    config: Map<FfiString, Any>,
) -> std::result::Result<Map<FfiString, Any>, tvm_ffi::Error> {
    run(Any::from(config)).map_err(failure)
}

fn run(config: Any) -> Result<Map<FfiString, Any>> {
    let config = Record::new("kv_transfer", config)?;
    let source_plan = Arc::new(KVPlan::decode(config.value("source_plan")?)?);
    let destination_plan = Arc::new(KVPlan::decode(config.value("destination_plan")?)?);
    let num_units = config.count("num_units")? as u32;
    let tokens = config.count("tokens")? as u32;
    let devices = [
        config.count("source_device")? as i32,
        config.count("destination_device")? as i32,
    ];
    let kind = TransportKind::parse(&config.string("transport")?)?;
    let bind_units: Function = Function::new("cache.bind_units", config.get("bind_units")?);
    let bind_conversion: Function =
        Function::new("cache.bind_conversion", config.get("bind_conversion")?);
    let fill: Function = Function::new("fill", config.get("fill")?);
    let source_units = units(config.value("source_units")?)?;
    let destination_units = units(config.value("destination_units")?)?;

    let streams = [
        Stream::new(DeviceId(devices[0]))?,
        Stream::new(DeviceId(devices[1]))?,
    ];
    let width = 64;
    let mut source = KVCache::new(
        Arc::clone(&source_plan),
        num_units,
        2,
        width,
        1,
        Some(devices[0]),
        Some(&streams[0]),
        &bind_units,
        &bind_conversion,
    )?;
    let mut destination = KVCache::new(
        Arc::clone(&destination_plan),
        num_units,
        2,
        width,
        2,
        Some(devices[1]),
        Some(&streams[1]),
        &bind_units,
        &bind_conversion,
    )?;

    // Python writes the producer's tokens through the bound views.
    source.install(
        &tables(1, &source_units, tokens, &source_plan),
        Some(&streams[0]),
    )?;
    streams[0].synchronize()?;
    let source_views = tensor_map(&source.views()?)?;
    let source_layout = source.layout()?;
    fill.invoke(&[AnyView::from(&source_views), AnyView::from(&source_layout)])?;

    let capacity = Arc::new(Capacity::new(1 << 28, 16, notify)?);
    let transports = Arc::new(Transports::new(
        &[kind],
        Arc::clone(&capacity),
        EventPool::new(),
        &TransportOptions {
            source: endpoint(),
            acknowledgment_slot: 0,
            host_slots: HashSet::from([0]),
            cross_host_consumers: false,
            vmm_pool_bytes: 1 << 26,
        },
    )?);

    let exported = buffer(0);
    let (transfer, locations) = source.export(
        1,
        tokens,
        "consumer",
        exported,
        &transports,
        &[0],
        Some(&streams[0]),
    )?;
    source
        .manager()
        .validate_exports(&[(exported, transfer.clone())], &[])?;
    source
        .manager()
        .apply_exports(vec![(exported, transfer.clone())], vec![]);

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
    source
        .manager()
        .release_calls(&[(exported.owner, exported.producer_call_id)]);
    source.manager().release_exports(&[exported]);
    uniserve_worker::transfer::exports::release(&locations)?;
    wait(
        || {
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
    let destination_layout = destination.layout()?;
    let result: Map<FfiString, Any> = [
        (FfiString::from("source"), Any::from(source_views)),
        (FfiString::from("source_layout"), Any::from(source_layout)),
        (FfiString::from("destination"), Any::from(destination_views)),
        (
            FfiString::from("destination_layout"),
            Any::from(destination_layout),
        ),
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

tvm_ffi::tvm_ffi_dll_export_typed_func!(kv_transfer, kv_transfer);
