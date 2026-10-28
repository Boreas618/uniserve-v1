//! Publication of a batch's products to the ranks and workers that read
//! them.
//!
//! A batch exports a product once the producer has written it: the tensor store
//! exports its value through every export mechanism that reaches the producing
//! call's consumers, retains each location's retirement on the product's
//! buffer, and the product's descriptor joins the call's completion (and so the
//! batch result the engine routes to the consumers). The locations become the
//! buffer's registered exports at the batch's commit point and are revoked when
//! the buffer is released; a batch that fails before committing revokes them at
//! once. The same rules hold for every batch, a computation's, a transfer's or
//! what host tasks wrote, so this module is the one place that applies them.
//!
//! A call of a computation that prepares reads products other calls published
//! (`Call::inputs`, a predicate) and publishes its own (`Call::outputs`, a
//! completion scalar). Products live in the rank's [`ProductStore`]: imports
//! from other ranks reach it before the batch executes
//! ([`crate::executor::inputs`]), and every input the executor's preparation
//! names is then leased from it ([`bind_prepared`]); host codecs read some
//! inputs in place from their shared-memory exports instead. [`bind_outputs`]
//! reserves a call's outputs at their scheduler placement and [`export`]
//! publishes a written one.

use crate::executor::PendingOutput;
use crate::executor::batch::{Batch, CallState};
use crate::executor::computations::CallResources;
use crate::executor::latents::LatentLedger;
use crate::executor::reports::Publication;
use crate::ffi::{Backing, DType, MemoryKind};
use crate::storage::allocation::{Storage, StorageView};
use crate::storage::encoders::{self, EncoderHostPool};
use crate::storage::{
    LatentUpdate, OutputPlacement, ProductBuffer, ProductExport, ProductRead, ProductStore,
};
use crate::stream::{CUDAStream, Issue};
use crate::transfer::exports::{self, Location};
use crate::transfer::{SharedMapping, SharedRead, TransportKind, Transports};
use crate::{Error, Result};
use cuda::Stream;
use ipc::{
    Batch as BatchPlan, BufferAllocation, BufferId, Call, CallId, CallKind, MediaCall, RequestKey,
    TensorExport, TensorRef, TransferHandle, TransferTransport, WorkerInfo,
};
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::{Arc, Mutex};

/// The product exports a batch accepted before its commit point.
#[derive(Default)]
pub(crate) struct Publications {
    exports: Vec<(BufferId, Vec<Location>)>,
}

impl Publications {
    /// Accept an export another owner made of `buffer` (a latent version,
    /// say): its `locations` register with the batch's products and its
    /// descriptor joins the call's completion products.
    pub(crate) fn accept(
        &mut self,
        buffer: BufferId,
        locations: Vec<Location>,
        export: TensorExport,
        output: &mut PendingOutput,
    ) -> Result<()> {
        // Record the locations before the descriptor is accepted, so a
        // failure from here on revokes them with the batch.
        self.exports.push((buffer, locations));
        let mut products = output.products().to_vec();
        products.push(export);
        output.set_products(products)
    }

    /// Issue the export of an encoder feature to other ranks and workers to
    /// `stream`, behind the work that produced it. Readers on this rank
    /// lease it from the tensor store directly, so a feature is exported
    /// only when the rank publishes over a mechanism other ranks read, and
    /// only by the rank that reports its component's results: `None`
    /// otherwise. [`Self::locate`] publishes the export once that work was
    /// performed.
    pub(crate) fn issue_feature(
        tensors: &ProductStore,
        transports: &Transports,
        call: &Call,
        write: &ProductBuffer,
        reports: bool,
        stream: impl Issue,
    ) -> Result<Option<ProductExport>> {
        if !reports || !transports.exports_remotely() {
            return Ok(None);
        }
        let product = call
            .encoder_output
            .as_ref()
            .filter(|product| product.buffer_id() == write.id())
            .ok_or_else(|| {
                Error::Invariant("an exported feature names no encoder output".into())
            })?;
        tensors
            .export_buffer(call, product, write, transports, false, None, stream)
            .map(Some)
    }

    /// Publish an issued `export` of `write` once its work was performed:
    /// its locations register with the batch's products and its descriptor
    /// joins the call's completion products.
    pub(crate) fn locate(
        &mut self,
        tensors: &ProductStore,
        write: &ProductBuffer,
        export: ProductExport,
        output: &mut PendingOutput,
    ) -> Result<()> {
        let (export, locations) = tensors.located(write, export)?;
        self.accept(write.id(), locations, export, output)
    }

    /// Check, before the commit point, that no buffer already registers
    /// different locations.
    pub(crate) fn validate(&self, tensors: &ProductStore) -> Result<()> {
        for (buffer, locations) in &self.exports {
            tensors.validate_exports(*buffer, locations)?;
        }
        Ok(())
    }

    /// Register the locations with their buffers at the commit point; they
    /// are revoked when the buffers are released.
    pub(crate) fn register(&mut self, tensors: &ProductStore) -> Result<()> {
        for (buffer, locations) in self.exports.drain(..) {
            tensors.register_exports(buffer, locations)?;
        }
        Ok(())
    }

    /// Revoke every location of a batch that failed before its commit
    /// point; physical accesses still retire through the transports.
    pub(crate) fn revoke(&mut self) -> Result<()> {
        let locations: Vec<Location> = self
            .exports
            .drain(..)
            .flat_map(|(_, locations)| locations)
            .collect();
        exports::release(&locations)
    }
}

/// The rank's product owners a batch publishes through.
#[derive(Clone, Copy)]
pub(crate) struct ProductOwners<'a> {
    pub(crate) store: &'a ProductStore,
    pub(crate) transports: &'a Transports,
}

/// The scheduler placement of a call's outputs on this rank: buffer-pool
/// allocations, request relay slots, and per output the logical shape and
/// this rank's region of it.
pub(crate) struct Placement<'a> {
    pub(crate) allocations: &'a HashMap<BufferId, BufferAllocation>,
    pub(crate) slots: &'a HashMap<RequestKey, usize>,
}

/// One output this rank publishes: its logical shape and, for a sharded
/// product, the region this rank holds.
pub(crate) struct OutputShape {
    pub(crate) product: TensorRef,
    pub(crate) shape: Vec<u64>,
    pub(crate) region: Option<Vec<Range<u64>>>,
}

/// Reserve a call's persistent outputs at their placement on `device`.
pub(crate) fn bind_outputs(
    store: &ProductStore,
    outputs: &[OutputShape],
    device: MemoryKind,
    placement: &Placement<'_>,
) -> Result<Vec<ProductBuffer>> {
    if outputs.is_empty() {
        return Ok(Vec::new());
    }
    let shapes: HashMap<BufferId, Vec<u64>> = outputs
        .iter()
        .map(|output| (output.product.buffer_id(), output.shape.clone()))
        .collect();
    let regions: HashMap<BufferId, Vec<Range<u64>>> = outputs
        .iter()
        .filter_map(|output| {
            output
                .region
                .clone()
                .map(|region| (output.product.buffer_id(), region))
        })
        .collect();
    let bindings: Vec<(TensorRef, MemoryKind)> = outputs
        .iter()
        .map(|output| (output.product.clone(), device))
        .collect();
    store.bind_outputs(
        &bindings,
        OutputPlacement {
            request_slots: Some(placement.slots),
            allocations: Some(placement.allocations),
            regions: Some(&regions),
            shapes: Some(&shapes),
        },
    )
}

/// Export a written product to the call's consumers. `regions` limits the
/// export to parts of the value (encoded rows publish their prefixes only).
pub(crate) fn export(
    owners: ProductOwners<'_>,
    call: &Call,
    product: &TensorRef,
    write: &ProductBuffer,
    host: bool,
    regions: Option<&[Vec<Range<u64>>]>,
    stream: Option<&Stream>,
) -> Result<(TensorExport, Vec<Location>)> {
    // Issued on `stream` at once, so the export is located here.
    let export = owners.store.export_buffer(
        call,
        product,
        write,
        owners.transports,
        host,
        regions,
        stream,
    )?;
    owners.store.located(write, export)
}

/// The tensor descriptor of a supplied input product.
pub(crate) fn supplied<'a>(plan: &'a BatchPlan, product: &TensorRef) -> Option<&'a TensorExport> {
    plan.input_products
        .iter()
        .find(|export| export.product == *product)
}

/// The published tensor of a device or encoder product export.
pub(crate) fn tensor(export: &TensorExport) -> Result<&ipc::TensorTransfer> {
    match &export.value {
        TransferHandle::DeviceProduct { tensor, .. } | TransferHandle::Encoder { tensor, .. } => {
            Ok(tensor)
        }
        TransferHandle::Latent { .. } => Err(Error::Invalid(
            "a media input is not a latent trajectory".into(),
        )),
    }
}

/// What a call contributes to its batch's products beside the state every
/// batch keeps of it: what the executor's preparation bound of it.
#[derive(Default)]
pub(crate) struct ProductCall {
    /// The trajectory commit the call's forward recorded, applied to the
    /// ledger at the commit point.
    pub(crate) latent: Option<LatentUpdate>,
    /// The exports the call's forward issued, located when the batch
    /// commits: their device work is performed by then.
    pub(crate) exports: Vec<(ProductBuffer, ProductExport)>,
    /// The reports the executor's host tasks of the call send.
    pub(crate) reports: usize,
    /// How the call publishes what its tasks wrote once they finished.
    pub(crate) publication: Option<Publication>,
    /// Inputs read in place from their shared-memory exports, released once
    /// the call's host tasks finished.
    pub(crate) borrowed: Vec<Arc<Mutex<SharedRead>>>,
}

/// One output a call writes, as the executor prepared it.
#[derive(Debug, Deserialize)]
struct PreparedWrite {
    output: usize,
    shape: Vec<u64>,
    region: Option<Vec<[u64; 2]>>,
    /// `host` or `device`.
    storage: String,
    /// A host task fills it after the batch commits.
    deferred: bool,
    /// Host readers read its export.
    host_export: bool,
}

/// What one call reads and writes, as the executor prepared it.
#[derive(Debug, Deserialize)]
struct Prepared {
    /// Indices into the call's inputs.
    reads: Vec<usize>,
    writes: Vec<PreparedWrite>,
    /// `products`, `encoded` or `media`.
    publication: Option<String>,
    /// For a call reading rows of a round's decoded frames: the round
    /// position each read takes, read in place from its shared-memory
    /// export or leased.
    #[serde(default)]
    read_rows: Option<Vec<u32>>,
    /// Each input is a round's framed rows, read from the exports that
    /// reach this rank.
    #[serde(default)]
    exported_rows: bool,
    /// The product rows an encoding call's tasks frame into.
    #[serde(default)]
    rows: Vec<u64>,
    /// The reports the call's host tasks send.
    #[serde(default)]
    reports: usize,
    /// An image model's call: what it reads and writes of its own product
    /// fields and trajectory (`crate::executor::image`).
    #[serde(default)]
    image: Option<serde_json::Value>,
    /// A video latent preparation on a denoising rank: how the rank's
    /// request slot pages hold the request's samples
    /// (`::storage::manifest::StorageLayout`), which its completion reports
    /// and a restore lays the stored values out in.
    #[serde(default)]
    latent_layout: Option<serde_json::Value>,
}

/// The name of a call's output product.
fn product_name<'a>(info: &'a WorkerInfo, call: &Call, output: &TensorRef) -> Result<&'a str> {
    info.components
        .iter()
        .find(|component| component.name == call.component)
        .and_then(|component| component.outputs.get(output.output_index as usize))
        .map(|output| output.name.as_str())
        .ok_or_else(|| {
            Error::Invalid(format!(
                "component {} declares no output {}",
                call.component, output.output_index
            ))
        })
}

/// The JSON step the executor prepares and executes on a computation that
/// prepares: every call with what the worker knows of it.
pub(crate) fn prepared_step(
    plan: &BatchPlan,
    info: &WorkerInfo,
    computation: usize,
    calls: &[CallState<CallResources>],
) -> Result<serde_json::Value> {
    let encode = |error: serde_json::Error| Error::Invalid(format!("encoding a batch: {error}"));
    let mut entries = Vec::with_capacity(plan.calls.len());
    for (index, (call, state)) in plan.calls.iter().zip(calls).enumerate() {
        let range = plan
            .decode_ranges
            .iter()
            .find(|range| range.request_key == call.request_key && range.call_id == call.call_id);
        let names = call
            .outputs
            .iter()
            .map(|output| product_name(info, call, output).map(str::to_owned))
            .collect::<Result<Vec<_>>>()?;
        // The executor reads an encoding's image as the host prepared it
        // (`image::describe`), not its payload: an inline image's encoded
        // bytes and a carried image's values stay out of the step.
        let mut described = call.clone();
        described.input_image = None;
        described.carried_image = None;
        entries.push(serde_json::json!({
            "index": index,
            "slot": state.slot,
            "active": state.active,
            "refused": state.refused(),
            "call": serde_json::to_value(&described).map_err(encode)?,
            "range": serde_json::to_value(range).map_err(encode)?,
            "output_names": names,
            "latent": serde_json::Value::Null,
        }));
    }
    Ok(serde_json::json!({
        "batch_id": plan.batch_id,
        "computation": computation,
        "calls": entries,
    }))
}

/// What binding a prepared batch's products borrows.
pub(crate) struct PreparedOwners<'a> {
    pub(crate) owners: ProductOwners<'a>,
    pub(crate) info: &'a WorkerInfo,
    /// The rank's index in its worker group, which picks its own pages of a
    /// restored trajectory.
    pub(crate) rank: u32,
    /// The trajectory ledger of a computation that names the `latents`
    /// resource.
    pub(crate) ledger: Option<&'a LatentLedger>,
    /// The host encoder pool of a rank that maps one.
    pub(crate) encoders: Option<&'a EncoderHostPool>,
    /// The stream device products host tasks wrote are fenced on.
    pub(crate) host_stream: Option<Arc<CUDAStream>>,
    /// The inputs host codecs read in place from their shared-memory
    /// exports instead of leasing them from the product store.
    pub(crate) borrowed: &'a HashSet<BufferId>,
}

/// What a prepared batch's execution publishes behind it: each write that
/// no host task defers, with its call and whether host readers read it.
pub(crate) struct Published {
    writes: Vec<(usize, TensorRef, ProductBuffer, bool)>,
}

/// Lease what every call of `batch` reads, bind what it writes and take
/// each trajectory's banks from the ledger, as the executor prepared them
/// (`step["prepared"]`, one entry per call), noting the banks in `step`.
/// Returns the tensors the executor's execution reads and writes, in call
/// order, and the writes [`publish`] publishes behind it. A video encoder
/// call whose output lies in the host encoder pool (`Call::host_output`)
/// reads or writes the pool pages its rows fall on besides: their views
/// follow the call's writes, one per copy `step["calls"][i]["host"]` lists.
/// A latent preparation that restores a stored trajectory reads the stored
/// pages besides: their views follow the call's writes, in the order of the
/// page indices `step["calls"][i]["latent"]["restore"]` lists.
pub(crate) fn bind_prepared(
    owners: &PreparedOwners<'_>,
    plan: &BatchPlan,
    batch: &mut Batch<CallResources>,
    step: &mut serde_json::Value,
    issue: impl Issue,
) -> Result<(Vec<StorageView>, Published)> {
    let prepared = step
        .get("prepared")
        .cloned()
        .ok_or_else(|| Error::Invariant("a prepared batch has no preparation".into()))?;
    let prepared: Vec<Prepared> = serde_json::from_value(prepared)
        .map_err(|error| Error::Interface(format!("the executor's preparation: {error}")))?;
    if prepared.len() != plan.calls.len() {
        return Err(Error::Interface(
            "the executor prepared another number of calls".into(),
        ));
    }
    let slots = batch.slots(plan);
    let allocations: HashMap<BufferId, BufferAllocation> = plan
        .buffer_allocations
        .iter()
        .map(|allocation| (allocation.buffer, *allocation))
        .collect();
    let placement = Placement {
        allocations: &allocations,
        slots: &slots,
    };
    let store = owners.owners.store;
    let target = batch.target();
    let mut tensors = Vec::new();
    let mut published = Vec::new();
    for (index, (call, prepared)) in plan.calls.iter().zip(&prepared).enumerate() {
        let state = &batch.calls[index];
        if state.refused() {
            continue;
        }
        let slot = state.slot;

        // Reads.
        for (position, &input_index) in prepared.reads.iter().enumerate() {
            let input = call.inputs.get(input_index).ok_or_else(|| {
                Error::Interface("the executor prepared a read of an undeclared input".into())
            })?;
            if prepared.exported_rows {
                tensors.push(exported_rows(owners, plan, input)?);
                continue;
            }
            if let Some(rows) = &prepared.read_rows {
                let row = *rows
                    .get(position)
                    .ok_or_else(|| Error::Interface("a prepared row read names no row".into()))?;
                if owners.borrowed.contains(&input.buffer_id()) {
                    let read = borrow_row(owners, plan, input, u64::from(row))?;
                    tensors.push(row_view(&read)?);
                    batch.calls[index]
                        .forward
                        .products
                        .borrowed
                        .push(Arc::new(Mutex::new(read)));
                } else {
                    let read = lease(store, batch, input, call.call_id, MemoryKind::Host, None)?;
                    tensors.push(read.value().select(0, i64::from(row))?);
                }
                continue;
            }
            let read = lease(store, batch, input, call.call_id, target, issue)?;
            tensors.push(read.value().clone());
        }

        // Writes.
        let mut deferred = Vec::new();
        for write in &prepared.writes {
            let product = call.outputs.get(write.output).ok_or_else(|| {
                Error::Interface("the executor prepared an undeclared output".into())
            })?;
            let storage = match write.storage.as_str() {
                "host" => MemoryKind::Host,
                "device" => target,
                other => {
                    return Err(Error::Interface(format!(
                        "the executor prepared an output in {other:?} storage"
                    )));
                }
            };
            let bound = bind_outputs(
                store,
                &[OutputShape {
                    product: product.clone(),
                    shape: write.shape.clone(),
                    region: write
                        .region
                        .as_ref()
                        .map(|region| region.iter().map(|&[a, b]| a..b).collect()),
                }],
                storage,
                &placement,
            )?
            .into_iter()
            .next()
            .ok_or_else(|| Error::Invariant("a prepared output was not bound".into()))?;
            tensors.push(bound.view());
            if write.deferred {
                store.defer_write(&bound)?;
                deferred.push((product.clone(), bound));
            } else {
                batch.writes.push(bound.clone());
                published.push((index, product.clone(), bound, write.host_export));
            }
        }
        // A video encoder call's output in the host encoder pool, on a rank
        // that writes some of it.
        if let Some(output) = &call.host_output
            && batch.calls[index].active
            && let Some((views, host)) = host_copies(owners.encoders, output, prepared)?
        {
            tensors.extend(views);
            step["calls"][index]["host"] = host;
        }

        let forward = &mut batch.calls[index].forward.products;
        forward.reports = prepared.reports;
        forward.publication = publication(prepared, deferred, owners.host_stream.as_ref())?;

        // The trajectory ledger. An image model's call binds its scheduled
        // trajectory itself (`crate::executor::image::bind`).
        let trajectory = prepared
            .image
            .as_ref()
            .is_none_or(serde_json::Value::is_null)
            && matches!(
                call.kind,
                CallKind::Media(MediaCall::LatentPreparation | MediaCall::Denoising)
            );
        if let (true, true, CallKind::Media(kind)) =
            (trajectory, batch.calls[index].active, call.kind)
        {
            let ledger = owners
                .ledger
                .ok_or_else(|| Error::Invalid("this rank keeps no latent trajectories".into()))?;
            // A video latent preparation reports how the rank's pages hold
            // the request's samples, which key and record the trajectory
            // when a preemption stores it.
            let layout = match (&prepared.latent_layout, kind) {
                (Some(layout), MediaCall::LatentPreparation) => Some(layout),
                (None, MediaCall::LatentPreparation) if call.latent_restore.is_some() => {
                    return Err(Error::Interface(
                        "a restoring latent preparation declares no latent layout".into(),
                    ));
                }
                _ => None,
            };
            if let Some(layout) = layout {
                batch.calls[index]
                    .output
                    .set_latent_layout(owners.rank, layout.to_string());
            }
            // A resumed request's preparation copies the stored values into
            // the bank it initializes, laid out in the rank's pages, and
            // commits the stored step.
            let (banks, update, restored) = match (&call.latent_restore, layout) {
                (Some(restore), Some(layout)) => {
                    let layout = serde_json::from_value(layout.clone()).map_err(|error| {
                        Error::Interface(format!("a request's latent layout: {error}"))
                    })?;
                    let bound = ledger.bind_restore(slot, call, restore, owners.rank, layout)?;
                    (bound.banks, bound.update, Some((bound.pages, bound.copies)))
                }
                _ => {
                    let (banks, update) = ledger.bind(slot, call, kind)?;
                    (banks, update, None)
                }
            };
            let mut latent = serde_json::json!({
                "read": banks.read,
                "write": banks.write,
                "step": banks.step,
            });
            if let Some((pages, copies)) = restored {
                // The stored pages follow the call's other tensors.
                latent["restore"] = serde_json::json!({
                    "pages": pages.len(),
                    "copies": copies,
                });
                tensors.extend(pages);
            }
            step["calls"][index]["latent"] = latent;
            batch.calls[index].forward.products.latent = Some(update);
        }
    }
    Ok((tensors, Published { writes: published }))
}

/// The pool pages a video encoder call's output reads or writes and the
/// copies between them (`Call::host_output`), as `bind_prepared` hands them
/// to the executor: the views, and `{"import": bool, "copies": [[output,
/// page offset, bytes], ...]}` with one copy per view, the byte offset in
/// the call's output, the byte offset in the page and the bytes copied.
///
/// An import reads every row of the call's output from the pages; an export
/// writes the rows the rank wrote (the region the executor prepared its
/// write in, or all of them) to the pages the pool does not hold already.
/// Ranks that write the same rows write the same bytes. `None` for a rank
/// that writes none of the output, as a latent encoding's audio call on a
/// rank other than its first.
fn host_copies(
    pool: Option<&EncoderHostPool>,
    output: &ipc::HostOutput,
    prepared: &Prepared,
) -> Result<Option<(Vec<StorageView>, serde_json::Value)>> {
    if prepared.writes.is_empty() {
        return Ok(None);
    }
    let pool = pool.ok_or_else(|| {
        Error::Invalid("this rank maps no host encoder pool for the call's output".into())
    })?;
    if output.pool != pool.pool() {
        return Err(Error::Invalid(format!(
            "the call's output lies in encoder pool {}, not in this rank's pool {}",
            output.pool,
            pool.pool()
        )));
    }
    output
        .validate(Some(pool.page_bytes()))
        .map_err(|error| Error::Invalid(format!("the call's host output: {error}")))?;
    let [write] = prepared.writes.as_slice() else {
        return Err(Error::Interface(
            "a video encoder call with a host output prepares one write".into(),
        ));
    };
    let rows = *write
        .shape
        .first()
        .ok_or_else(|| Error::Interface("an encoder output has no rows".into()))?;
    if u64::from(output.rows[1] - output.rows[0]) != rows {
        return Err(Error::Interface(format!(
            "the executor prepared {rows} rows of an encoder output the engine names {:?} of",
            output.rows
        )));
    }
    // The rows this rank's copies cover.
    let copied = match (&write.region, output.import) {
        (Some(region), false) => {
            let rows = region
                .first()
                .ok_or_else(|| Error::Interface("an encoder write's region has no rows".into()))?;
            rows[0]..rows[1]
        }
        _ => 0..rows,
    };
    let mut views = Vec::new();
    let mut copies = Vec::new();
    for copy in encoders::page_copies(output.rows[0], copied, output.row_bytes, pool.page_bytes()) {
        let slot = output.slots.get(copy.page).ok_or_else(|| {
            Error::Invalid("an encoder output's rows lie beyond its pages".into())
        })?;
        // A page the pool held already is left as it is.
        let Some(slot) = slot else {
            continue;
        };
        views.push(pool.view(*slot)?);
        copies.push([copy.output, copy.offset, copy.bytes]);
    }
    Ok(Some((
        views,
        serde_json::json!({ "import": output.import, "copies": copies }),
    )))
}

/// How a call publishes what its host tasks write into `deferred`, its
/// writes the executor prepared as deferred: `None` for a call without
/// host tasks' products.
fn publication(
    prepared: &Prepared,
    deferred: Vec<(TensorRef, ProductBuffer)>,
    host_stream: Option<&Arc<CUDAStream>>,
) -> Result<Option<Publication>> {
    Ok(match prepared.publication.as_deref() {
        None => None,
        Some("products") => Some(Publication::Products {
            written: deferred,
            stream: host_stream.cloned(),
        }),
        Some("encoded") => {
            let (product, write) = deferred.into_iter().next().ok_or_else(|| {
                Error::Interface("an encoding call prepared no encoded rows".into())
            })?;
            Some(Publication::Encoded {
                product,
                write,
                rows: prepared.rows.clone(),
            })
        }
        Some("media") => Some(Publication::Media),
        Some(other) => {
            return Err(Error::Interface(format!(
                "the executor prepared the unknown publication {other:?}"
            )));
        }
    })
}

/// Lease one input for `consumer`, whole, in `storage`; what `stream`
/// issues next waits on its writer. The lease ends with the batch's other
/// reads, when it commits or retires.
fn lease(
    store: &ProductStore,
    batch: &mut Batch<CallResources>,
    product: &TensorRef,
    consumer: CallId,
    storage: MemoryKind,
    stream: impl Issue,
) -> Result<Arc<ProductRead>> {
    let read = store.consume(product, consumer, Some(storage), stream)?;
    // Kept before the check below, so a refused read still ends.
    batch.reads.push(Arc::clone(&read));
    if read.region().is_some() {
        return Err(Error::Invalid(
            "a prepared computation requires complete input coverage".into(),
        ));
    }
    Ok(read)
}

/// Record into `issue` the producer fences and the exports of what the
/// executor's execution wrote, behind it.
pub(crate) fn publish(
    owners: ProductOwners<'_>,
    plan: &BatchPlan,
    batch: &mut Batch<CallResources>,
    published: Published,
    issue: impl Issue,
) -> Result<()> {
    for (index, product, write, host) in published.writes {
        owners.store.write(&write, &write.view(), issue, None)?;
        let export = owners.store.export_buffer(
            &plan.calls[index],
            &product,
            &write,
            owners.transports,
            host,
            None,
            issue,
        )?;
        batch.calls[index]
            .forward
            .products
            .exports
            .push((write, export));
    }
    Ok(())
}

/// A borrowed shared-memory row as host bytes, viewed as long as the read
/// lives.
struct MappedBytes {
    mapping: Arc<SharedMapping>,
    offset: usize,
    bytes: usize,
}

impl Backing for MappedBytes {
    fn device(&self) -> MemoryKind {
        MemoryKind::Host
    }

    fn address(&self) -> u64 {
        (self.mapping.address() + self.offset) as u64
    }

    fn bytes(&self) -> usize {
        self.bytes
    }
}

/// A host uint8 view of a borrowed row's bytes.
fn row_view(read: &SharedRead) -> Result<StorageView> {
    let bytes = read.nbytes();
    let backing = MappedBytes {
        mapping: read.mapping()?,
        offset: read.offset(),
        bytes,
    };
    StorageView::contiguous(
        Arc::new(backing) as Arc<dyn Backing>,
        0,
        DType::U8,
        &[bytes as i64],
    )
}

/// Row `row` of a round's decoded frames, read in place from its
/// shared-memory export on this host.
fn borrow_row(
    owners: &PreparedOwners<'_>,
    plan: &BatchPlan,
    product: &TensorRef,
    row: u64,
) -> Result<SharedRead> {
    let export = supplied(plan, product)
        .ok_or_else(|| Error::Invalid("a borrowed input has no export".into()))?;
    let tensor = tensor(export)?;
    let transport = owners
        .owners
        .transports
        .get(TransportKind::PosixShm)
        .ok_or_else(|| Error::Invalid("borrowed rows require local shared storage".into()))?;
    let location = tensor
        .locations
        .iter()
        .find(|location| {
            matches!(location.transport, TransferTransport::PosixShm { .. })
                && location.source.host == owners.info.endpoint.host
                && location.offset[0] <= row
                && row < location.offset[0] + location.shape[0]
        })
        .ok_or_else(|| Error::Invalid("a borrowed row has no local shared storage".into()))?;
    let region: Vec<Range<u64>> = std::iter::once(row..row + 1)
        .chain(tensor.shape[1..].iter().map(|&extent| 0..extent))
        .collect();
    transport.borrow(location, Some(&region))
}

/// A round's framed unit rows, `[rows, width]` host uint8, read from the
/// exports that reach this rank.
fn exported_rows(
    owners: &PreparedOwners<'_>,
    plan: &BatchPlan,
    product: &TensorRef,
) -> Result<StorageView> {
    let export = supplied(plan, product).ok_or_else(|| {
        Error::Invalid("artifact assembly input has no transfer locations".into())
    })?;
    let tensor = tensor(export)?;
    if tensor.shape.len() != 2
        || tensor
            .locations
            .iter()
            .any(|location| location.dtype != "uint8")
    {
        return Err(Error::Invalid("encoded units require byte rows".into()));
    }
    let (count, width) = (tensor.shape[0] as usize, tensor.shape[1] as usize);
    let storage = Storage::host((count * width).max(1));
    let mut filled = vec![false; count];
    for location in &tensor.locations {
        if location.offset.get(1).copied().unwrap_or(0) != 0 {
            return Err(Error::Invalid(
                "encoded unit location lacks its length prefix".into(),
            ));
        }
        let first = location.offset[0] as usize;
        let rows = location.shape[0] as usize;
        let row_bytes = location.shape.get(1).copied().unwrap_or(0) as usize;
        if first + rows > count || filled[first..first + rows].iter().all(|&done| done) {
            continue;
        }
        let bytes: Vec<u8> = match &location.transport {
            TransferTransport::Channel { payload, .. } => payload.clone(),
            TransferTransport::PosixShm { .. }
                if location.source.host == owners.info.endpoint.host =>
            {
                let Some(transport) = owners.owners.transports.get(TransportKind::PosixShm) else {
                    continue;
                };
                let mut read = transport.borrow(location, None)?;
                let mapping = read.mapping()?;
                // SAFETY: the claimed range is mapped and ready until the
                // read is released below.
                let payload = unsafe {
                    std::slice::from_raw_parts(
                        (mapping.address() + read.offset()) as *const u8,
                        read.nbytes(),
                    )
                }
                .to_vec();
                read.release();
                payload
            }
            _ => continue,
        };
        if row_bytes == 0 || row_bytes > width || bytes.len() != rows * row_bytes {
            return Err(Error::Invalid(
                "encoded unit rows disagree with their location".into(),
            ));
        }
        for (offset, row) in bytes.chunks_exact(row_bytes).enumerate() {
            let target = (first + offset) * width;
            // SAFETY: the host allocation holds `count * width` bytes, this
            // row's range lies inside it and nothing else references it.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    (storage.address() as *mut u8).add(target),
                    row.len(),
                );
            }
            filled[first + offset] = true;
        }
    }
    if filled.iter().any(|&done| !done) {
        return Err(Error::Invalid(
            "artifact assembly requires every encoded media unit".into(),
        ));
    }
    StorageView::contiguous(
        storage as Arc<dyn Backing>,
        0,
        DType::U8,
        &[count as i64, width as i64],
    )
}
