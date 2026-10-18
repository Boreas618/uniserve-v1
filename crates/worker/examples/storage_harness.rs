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
//!
//! `cache_lifecycle` drives one host cache through the lifecycle of a
//! request's pages: installs and recycling, an export, an import that is
//! abandoned and one that is adopted, a guidance prefix slot, and the
//! release of the exports, the slots and the prefixes. After each step it
//! hands Python the cache's live tensors and which slots' units another
//! owner could reuse.

#[path = "harness/config.rs"]
pub mod config;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use config::Config;
use tvm_ffi::{Any, AnyView, Array, Map, String as FfiString};
use uniserve_core::{CallId, RequestId, UnitId};
use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{BlockTable, BufferId, RequestKey, WorkerEndpoint};
use uniserve_worker::ffi::{Function, Handle, Interpreter};
use uniserve_worker::storage::{CachePlan, PrefixCache, StorageView, UnitColumn};
use uniserve_worker::transfer::exports;
use uniserve_worker::transfer::{TransferCapacity, TransportKind, TransportOptions, Transports};
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
        DeviceId(config.count("source_device")? as i32),
        DeviceId(config.count("destination_device")? as i32),
    ];
    let kind = TransportKind::parse(&config.string("transport")?)?;
    let fill: Function = Function::new("fill", config.get("fill")?);
    let source_units = units(config.value("source_units")?)?;
    let destination_units = units(config.value("destination_units")?)?;

    let streams = [Stream::new(devices[0])?, Stream::new(devices[1])?];
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

    // No control loop takes job completions here: the deferred releases
    // wait for their events on the host (`EventPool::drain`).
    let events = EventPool::new();
    let capacity = Arc::new(TransferCapacity::new(1 << 28, 16)?);
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
    exports::release(&locations)?;
    wait(
        || {
            events.drain()?;
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
                serde_json::to_string(&transfer)
                    .map_err(|error| uniserve_worker::Error::Invalid(error.to_string()))?
                    .as_str(),
            )),
        ),
    ]
    .into_iter()
    .collect();
    Ok(result)
}

/// The flattened unit ids of one slot's groups.
fn flat(groups: &[Vec<UnitId>]) -> Vec<u32> {
    groups.iter().flatten().map(|unit| unit.0).collect()
}

fn cache_lifecycle(
    config: Map<FfiString, Any>,
) -> std::result::Result<Map<FfiString, Any>, tvm_ffi::Error> {
    lifecycle(Any::from(config)).map_err(failure)
}

/// Request slots of the lifecycle cache: the exporting slot 1, the
/// importing slot 2 and the guidance prefix slot 3.
const LIFECYCLE_SLOTS: u32 = 3;

fn lifecycle(config: Any) -> Result<Map<FfiString, Any>> {
    let config = Config::new("cache_lifecycle", config)?;
    Interpreter::install_import(config.get("import")?);
    let plan = Arc::new(CachePlan::read(Handle::new(config.value("plan")?)?)?);
    let num_units = config.count("num_units")? as u32;
    let tokens = config.count("tokens")? as u32;
    let observe = Function::new("observe", config.get("observe")?);
    let slots = config
        .list("units")?
        .into_iter()
        .map(units)
        .collect::<Result<Vec<_>>>()?;
    let [source, destination, prefix] = slots.as_slice() else {
        return Err(uniserve_worker::Error::Interface(
            "cache_lifecycle field `units` names three slots' units".into(),
        ));
    };

    // Host storage: every write is made at once, in call order.
    let host = None::<&Stream>;
    let mut cache = PrefixCache::new(
        Arc::clone(&plan),
        num_units,
        LIFECYCLE_SLOTS,
        64,
        1,
        None,
        None,
    )?;
    let column = cache.bind_units(None)?;
    let views = tensor_map(&cache.views()?)?;

    // Whether another owner could reuse `units` now: recycling them is
    // refused while an export, an import or an execution overlaps them.
    // An unquantized single-dtype pool is recycled without writing.
    let reusable = |cache: &mut PrefixCache, column: &UnitColumn, units: &[Vec<UnitId>]| {
        cache.recycle(&flat(units), column, host).is_ok()
    };
    let step = |name: &str, facts: &[(&str, bool)]| -> Result<()> {
        let facts: Map<FfiString, Any> = facts
            .iter()
            .map(|&(fact, value)| (FfiString::from(fact), Any::from(value)))
            .collect();
        observe.invoke(&[
            AnyView::from(&Any::from(FfiString::from(name))),
            AnyView::from(&views),
            AnyView::from(&facts),
        ])?;
        Ok(())
    };

    // The scheduler assigns the exporting slot fresh units.
    cache.install(&tables(1, source, tokens, &plan), host)?;
    let free = reusable(&mut cache, &column, source);
    step("installed", &[("source_reusable", free)])?;

    // An export holds the units it reads until its readers retired.
    let capacity = Arc::new(TransferCapacity::new(1 << 24, 16)?);
    let events = EventPool::new();
    let transports = Arc::new(Transports::new(
        &[TransportKind::Local],
        Arc::clone(&capacity),
        Arc::clone(&events),
        &TransportOptions {
            source: endpoint(),
            acknowledgment_slot: 0,
            host_slots: HashSet::from([0]),
            cross_host_consumers: false,
            vmm_pool_bytes: 1 << 20,
        },
    )?);
    let exported = buffer(0);
    let export = cache.export(1, tokens, "consumer", exported, &transports, &[0], host)?;
    let (transfer, locations) = cache.locate_export(export)?;
    cache.validate_transfers(&[(exported, transfer.clone())], &[])?;
    cache.apply_transfers(vec![(exported, transfer.clone())], vec![]);
    let free = reusable(&mut cache, &column, source);
    step("exported", &[("source_reusable", free)])?;

    // An import reserves the units it resets until it retires, abandoned or
    // adopted.
    let assignments = tables(2, destination, tokens, &plan);
    let reserve = |cache: &mut PrefixCache| {
        let batch_tables = cache.tables().for_batch(2, &assignments)?;
        cache.reserve_import(
            Arc::new(transfer.clone()),
            2,
            batch_tables,
            flat(destination),
            Arc::clone(&transports),
        )
    };
    let abandoned = reserve(&mut cache)?;
    let free = reusable(&mut cache, &column, destination);
    step("importing", &[("destination_reusable", free)])?;
    cache.abandon_import(&abandoned)?;
    wait(
        || {
            cache.reap_imports()?;
            Ok(abandoned.retirement().done())
        },
        "the abandoned import's retirement",
    )?;
    let free = reusable(&mut cache, &column, destination);
    step("abandoned", &[("destination_reusable", free)])?;

    // The adopted import makes the exported extent the slot's verified
    // prefix.
    let import = reserve(&mut cache)?;
    wait(|| Ok(import.done()), "the import copy")?;
    cache.install(&assignments, host)?;
    let installed = BufferId {
        output_index: 1,
        ..exported
    };
    let adopted = cache.adopt_import(&import, installed, host)?;
    cache.apply_transfers(vec![], vec![(exported, installed, (*adopted).clone())]);
    wait(
        || {
            cache.reap_imports()?;
            Ok(import.retirement().done())
        },
        "the adopted import's retirement",
    )?;
    step("adopted", &[])?;

    // A guidance prefix slot the request holds besides its own.
    cache.install(&tables(3, prefix, tokens, &plan), host)?;
    cache.retain_prefix(exported.owner, 3);
    // Another request holds no prefix: releasing its prefixes finds none.
    cache.release_request_prefixes(RequestKey::new(1, RequestId(2), 1), host)?;
    step("prefixed", &[])?;

    // The finished request's exports retire once revoked, which returns the
    // units they read; then its slots and its prefix slot are released.
    cache.release_calls(&[(exported.owner, exported.producer_call_id)]);
    cache.release_exports(&[exported]);
    exports::release(&locations)?;
    wait(
        || {
            events.drain()?;
            transports.reap()?;
            Ok(capacity.used() == 0)
        },
        "the export's retirement",
    )?;
    let free = reusable(&mut cache, &column, source);
    step("unexported", &[("source_reusable", free)])?;
    cache.release_slots(&[1, 2], host)?;
    cache.release_request_prefixes(exported.owner, host)?;
    step("released", &[])?;

    cache.close()?;
    transports.close()?;
    Ok([(
        FfiString::from("exported_extent"),
        Any::from(i64::from(transfer.exported_extent)),
    )]
    .into_iter()
    .collect())
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(cache_transfer, cache_transfer);
tvm_ffi::tvm_ffi_dll_export_typed_func!(cache_lifecycle, cache_lifecycle);
