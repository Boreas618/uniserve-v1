//! What the worker keeps for an image model's prepared lanes: the image
//! products, inline pixels, latent trajectories and guidance prefixes of
//! their batches.
//!
//! An image model's encoders, denoiser and decoder serve prepared lanes
//! ([`crate::executor::lanes`]) that run through the one batch state machine:
//! the executor states in its preparation what each call reads and writes,
//! then executes the batch on what the worker reserved. Its calls name their
//! products in fields of their own (`image_input`, `encoder_output`,
//! `image_output`, `latent_input`, `latent_output`), so the lane's hooks here
//! handle them beside the generic products:
//!
//! - [`describe`] adds to the preparation's step each call's input image (its
//!   host-prepared facts, or a resident image product's shape and metadata),
//!   its request's input image count and its progress;
//! - [`bind`], in the batch's last job, leases what each call reads (a
//!   resident image), passes its prepared pixels, reserves its feature and
//!   image products at their prepared shapes, lends its decoded bytes from
//!   the batch's output lease and binds its trajectory in the latent ledger
//!   ([`LatentLedger::bind_image`]); a denoising batch over the prefix cache
//!   also reserves its rows' slots (the guidance prefixes) and passes its
//!   attention rows ([`reserve_rows`]);
//! - [`publish`], behind the execution, fences what the calls wrote with
//!   its metadata and issues the exports remote readers import (features,
//!   trajectory versions), configures the PNG encodings of decoded images,
//!   releases a finished trajectory's guidance prefixes and records the
//!   batch's place in the prefix cache's and the ledger's stream orders;
//! - [`accept`] publishes the trajectory exports once their work was
//!   performed, [`retire`] withdraws what a failed batch issued, and
//!   [`host_stage`] encodes the decoded images after the batch's fences.
//!
//! Tensors the executor receives for a call, after the generic ones of every
//! call (image calls have none), in call order: the resident image read, the
//! prepared pixels, each prepared write, the lent bytes; then, for a
//! denoising batch over the prefix cache, one `int64 [5 + 2G, R]` host
//! tensor of its attention rows ([`reserve_rows`]).

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use ipc::{
    Batch as BatchPlan, BufferAllocation, BufferId, Call, CallKind, CallStatus, ErrorCode,
    LatentParams, MediaCall,
};
use serde::Deserialize;

use crate::executor::PendingOutput;
use crate::executor::batch::{Batch, CallState, HostStage};
use crate::executor::columns::{FORWARD_FIELDS, ForwardRows, field_major, host_tensor};
use crate::executor::image_inputs::{ImageFacts, Images, InlineImages};
use crate::executor::lanes::{LaneCall, Resources};
use crate::executor::latents::{Banks, LatentLedger};
use crate::executor::pages::{Pages, attention_pages};
use crate::executor::products::{ProductOwners, Publications};
use crate::executor::transfer::latents::{self, PendingExport};
use crate::ffi::MemoryKind;
use crate::host::HostTask;
use crate::host::image::{ImageResult, ImageTask, ImageWork};
use crate::lane::{Job, Recorder};
use crate::memory::allocation::StorageView;
use crate::memory::{
    AttentionRow, Metadata, OutputPlacement, PrefixCache, ProductBuffer, ProductStore, ValueRange,
};
use crate::request::{RequestPool, RequestProgress};
use crate::stream::{EventPool, Issue as _};
use crate::transfer::Transports;
use crate::{Error, Outcome, Result, lock};

/// Whether a lane serves an image model's calls: it names the image host
/// resource, or it is a prepared lane over the prefix cache (an image
/// denoiser, whose guidance prefixes and conditioning live there).
pub(crate) fn serves(resources: Resources, prepares: bool) -> bool {
    resources.images || (prepares && resources.pages)
}

/// The name a value range takes in the executor's step.
fn value_range_name(range: Option<ValueRange>) -> Option<&'static str> {
    range.map(|range| match range {
        ValueRange::Unit => "unit",
        ValueRange::SignedUnit => "signed_unit",
    })
}

/// The value range a prepared write names.
fn value_range(name: Option<&str>) -> Result<Option<ValueRange>> {
    match name {
        None => Ok(None),
        Some("unit") => Ok(Some(ValueRange::Unit)),
        Some("signed_unit") => Ok(Some(ValueRange::SignedUnit)),
        Some(other) => Err(Error::Interface(format!(
            "the executor prepared the unknown value range {other:?}"
        ))),
    }
}

/// Add to the preparation's step, per call of an image lane, what the
/// executor reads of it before anything is leased (`calls[i].image`): its
/// input image, the host-prepared facts of an inline one (`inputs`) or the
/// shape and metadata of a resident product, its request's input image
/// count and its progress.
pub(crate) fn describe(
    step: &mut serde_json::Value,
    plan: &BatchPlan,
    calls: &[CallState<LaneCall>],
    requests: &RequestPool,
    tensors: &ProductStore,
    inputs: Option<&InlineImages>,
) -> Result<()> {
    for (index, (call, state)) in plan.calls.iter().zip(calls).enumerate() {
        let input = if state.refused() || !state.active {
            Input::None
        } else if call.input_image.is_some() {
            let image = inputs
                .map(|inputs| inputs.image(index))
                .transpose()?
                .flatten()
                .ok_or_else(|| Error::Invariant("an inline image was not prepared".into()))?;
            Input::Inline(image.facts)
        } else if let Some(reference) = &call.image_input {
            let (shape, metadata) = tensors.metadata(reference)?;
            Input::Resident(shape, metadata)
        } else {
            Input::None
        };
        let input_images = requests
            .get(call.request_key.request_id.0)
            .map(|request| request.admission().input_images)
            .unwrap_or(0);
        // The host prepared an inline image's pixels already: the executor
        // reads its facts, not its payload.
        if call.input_image.is_some() {
            step["calls"][index]["call"]["input_image"] = serde_json::Value::Null;
        }
        step["calls"][index]["image"] = described(input, input_images, &state.output.progress)?;
    }
    Ok(())
}

/// A call's input image as [`describe`] states it.
enum Input {
    None,
    /// An inline image's host-prepared facts.
    Inline(ImageFacts),
    /// A resident image product's value shape and metadata.
    Resident(Vec<u64>, Option<Metadata>),
}

/// One call's `image` entry of the preparation's step.
fn described(
    input: Input,
    input_images: u32,
    progress: &RequestProgress,
) -> Result<serde_json::Value> {
    let input = match input {
        Input::None => serde_json::Value::Null,
        Input::Inline(facts) => serde_json::json!({"inline": facts.to_json()}),
        Input::Resident(
            shape,
            Some(Metadata::Image {
                height,
                width,
                value_range,
            }),
        ) => serde_json::json!({"resident": {
            "shape": shape,
            "height": height,
            "width": width,
            "value_range": value_range_name(value_range),
        }}),
        Input::Resident(..) => {
            return Err(Error::Invalid(
                "a resident image input requires image product metadata".into(),
            ));
        }
    };
    Ok(serde_json::json!({
        "input": input,
        "input_images": input_images,
        "progress": {
            "logical": progress.logical_position,
            "visible": progress.cache_visible_len,
        },
    }))
}

/// The product field a prepared image write names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Field {
    EncoderOutput,
    ImageOutput,
}

/// The metadata a prepared image write publishes with.
#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PreparedMetadata {
    Feature {
        height: u32,
        width: u32,
    },
    Image {
        height: u32,
        width: u32,
        value_range: Option<String>,
    },
}

impl PreparedMetadata {
    fn metadata(&self) -> Result<Metadata> {
        Ok(match self {
            Self::Feature { height, width } => Metadata::Feature {
                height: *height,
                width: *width,
            },
            Self::Image {
                height,
                width,
                value_range: range,
            } => Metadata::Image {
                height: *height,
                width: *width,
                value_range: value_range(range.as_deref())?,
            },
        })
    }
}

/// One image product a call writes, as the executor prepared it.
#[derive(Clone, Debug, Deserialize)]
struct PreparedWrite {
    field: Field,
    shape: Vec<u64>,
    #[serde(default)]
    metadata: Option<PreparedMetadata>,
}

/// What an image call reads and writes, as the executor prepared it
/// (`prepared[i].image`).
#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct Prepared {
    /// Lease the call's resident image (`image_input`).
    #[serde(default)]
    read: bool,
    /// Pass the call's host-prepared inline pixels.
    #[serde(default)]
    pixels: bool,
    #[serde(default)]
    writes: Vec<PreparedWrite>,
    /// Decoded RGB bytes lent from the batch's output lease.
    #[serde(default)]
    bytes: u64,
    /// Bind the call's trajectory in the latent ledger.
    #[serde(default)]
    latent: bool,
}

/// Each call's prepared image entry, `None` for a call the executor
/// prepared nothing of the image's for.
fn prepared(step: &serde_json::Value) -> Result<Vec<Option<Prepared>>> {
    let entries = step
        .get("prepared")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::Invariant("a prepared batch has no preparation".into()))?;
    entries
        .iter()
        .map(|entry| {
            entry
                .get("image")
                .filter(|image| !image.is_null())
                .map(|image| {
                    serde_json::from_value::<Prepared>(image.clone()).map_err(|error| {
                        Error::Interface(format!("the executor's image preparation: {error}"))
                    })
                })
                .transpose()
        })
        .collect()
}

/// Output words a prepared batch lends for its calls' decoded bytes.
pub(crate) fn lease_words(step: &serde_json::Value) -> Result<usize> {
    let bytes: u64 = prepared(step)?
        .iter()
        .flatten()
        .map(|prepared| prepared.bytes)
        .sum();
    usize::try_from(bytes.div_ceil(size_of::<i64>() as u64))
        .map_err(|_| Error::Invalid("decoded images exceed the output words".into()))
}

/// Bytes of one decoded pixel: its RGB channels.
const PIXEL_BYTES: u64 = 3;

/// One image product a call writes, bound at its prepared shape.
struct ImageWrite {
    field: Field,
    write: ProductBuffer,
    shape: Vec<u64>,
    metadata: Option<Metadata>,
}

/// What one call of an image lane holds through its batch.
#[derive(Default)]
pub(crate) struct ImageCall {
    /// The image products it writes.
    writes: Vec<ImageWrite>,
    /// Its trajectory's banks and pages in the latent ledger.
    latent: Option<(Banks, LatentParams)>,
    /// The trajectory exports it issued, published once their work was
    /// performed.
    exports: Vec<PendingExport>,
    /// The bytes its decoded image is staged in, with its size.
    bytes: Option<(StorageView, u32, u32)>,
    /// The PNG encoding of its decoded image.
    encoding: Option<Arc<HostTask<ImageTask>>>,
    /// The slots of its guidance prefixes (rows on slots other than its
    /// own).
    alternates: Vec<u32>,
}

/// What an image lane's batch holds through its last job.
#[derive(Default)]
pub(crate) struct ImageBatch {
    /// The batch uses the ledger's banks on its stream
    /// ([`LatentLedger::enter`]).
    entered: bool,
    /// It reserved rows over the prefix cache, whose place in the cache's
    /// stream order it records.
    ordered: bool,
    /// The host pixels its execution copies from, kept until the stream
    /// passed that work.
    pixels: Vec<StorageView>,
    /// The trajectory versions it exported from the ledger's pool, released
    /// from it if the batch fails before committing.
    versions: Vec<BufferId>,
}

/// The rank's owners an image lane's batch binds and publishes through.
pub(crate) struct Owners<'a> {
    pub(crate) store: &'a ProductStore,
    pub(crate) transports: &'a Transports,
    pub(crate) events: &'a EventPool,
    pub(crate) requests: &'a RequestPool,
    pub(crate) prefix_cache: Option<&'a mut PrefixCache>,
    pub(crate) pages: Option<&'a mut Pages>,
    pub(crate) ledger: Option<&'a mut LatentLedger>,
    pub(crate) images: Option<&'a Images>,
}

impl Owners<'_> {
    fn ledger(&mut self) -> Result<&mut LatentLedger> {
        self.ledger
            .as_deref_mut()
            .ok_or_else(|| Error::Invalid("this rank keeps no latent trajectories".into()))
    }

    fn prefix_cache(&mut self) -> Result<&mut PrefixCache> {
        self.prefix_cache
            .as_deref_mut()
            .ok_or_else(|| Error::Unsupported("image trajectories require a prefix cache".into()))
    }
}

/// `view`'s leading elements as `shape`.
fn shaped(view: &StorageView, shape: &[u64]) -> Result<StorageView> {
    let elements = shape.iter().try_fold(1_u64, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or_else(|| Error::Invalid("a prepared shape overflows".into()))
    })?;
    if !view.is_contiguous() || elements > view.numel() {
        return Err(Error::Invalid(
            "a prepared image write exceeds its product's storage".into(),
        ));
    }
    let signed: Vec<i64> = shape.iter().map(|&extent| extent as i64).collect();
    StorageView::contiguous(
        Arc::clone(view.memory()),
        view.offset(),
        view.dtype(),
        &signed,
    )
}

/// The scheduler's latent parameters of `call`.
fn latent_params<'p>(plan: &'p BatchPlan, call: &Call) -> Result<&'p LatentParams> {
    plan.latent_params
        .iter()
        .find(|params| params.request_key == call.request_key && params.call_id == call.call_id)
        .ok_or_else(|| Error::Invalid("trajectory call has no latent allocation".into()))
}

/// Bind every image call of `batch` as the executor prepared it
/// (`step["prepared"]`), recording the stream work into `job`, and note
/// each bound trajectory in `step` (`calls[i].latent`) and a denoising
/// batch's slots (`calls[i].image.allocated`). Returns the tensors the
/// execution reads and writes, in the module's order.
pub(crate) fn bind(
    owners: &mut Owners<'_>,
    plan: &BatchPlan,
    batch: &mut Batch<LaneCall>,
    record: &mut ImageBatch,
    step: &mut serde_json::Value,
    job: &Job,
    inputs: Option<&InlineImages>,
) -> Result<Vec<StorageView>> {
    let prepared = prepared(step)?;
    if prepared.len() != plan.calls.len() {
        return Err(Error::Interface(
            "the executor prepared another number of calls".into(),
        ));
    }
    let stream = batch.stream.clone();
    let issue = job.on(stream.as_ref());
    let target = batch.target();
    let allocations: HashMap<BufferId, BufferAllocation> = plan
        .buffer_allocations
        .iter()
        .map(|allocation| (allocation.buffer, *allocation))
        .collect();
    let mut tensors = Vec::new();
    for (index, (call, prepared)) in plan.calls.iter().zip(&prepared).enumerate() {
        let Some(prepared) = prepared else {
            continue;
        };
        if batch.calls[index].refused() {
            continue;
        }
        let active = batch.calls[index].active;
        let slot = batch.calls[index].slot;

        // The resident image it reads, whose size a decoding stages.
        let mut size = None;
        if prepared.read {
            let reference = call.image_input.as_ref().ok_or_else(|| {
                Error::Interface("the executor prepared a read of no resident image".into())
            })?;
            let read = owners
                .store
                .consume(reference, call.call_id, Some(target), issue)?;
            batch.reads.push(Arc::clone(&read));
            if read.region().is_some() {
                return Err(Error::Invalid("a resident image is read whole".into()));
            }
            if let Some(Metadata::Image { height, width, .. }) = read.metadata() {
                size = Some((height, width));
            }
            tensors.push(read.value().clone());
        }

        // Its inline pixels, prepared on the host.
        if prepared.pixels {
            let image = inputs
                .map(|inputs| inputs.image(index))
                .transpose()?
                .flatten()
                .ok_or_else(|| Error::Invariant("an inline image was not prepared".into()))?;
            record.pixels.push(image.pixels.clone());
            tensors.push(image.pixels);
        }

        // Its products, at their prepared shapes.
        for write in &prepared.writes {
            let metadata = write
                .metadata
                .as_ref()
                .map(PreparedMetadata::metadata)
                .transpose()?;
            let bound = reserve_write(owners.store, call, write, target, &allocations)?;
            batch.writes.push(bound.clone());
            tensors.push(shaped(&bound.view(), &write.shape)?);
            batch.calls[index].forward.image.writes.push(ImageWrite {
                field: write.field,
                write: bound,
                shape: write.shape.clone(),
                metadata,
            });
        }

        // Its trajectory in the ledger. A trajectory opens on its request's
        // exported conditioning context.
        if prepared.latent
            && active
            && call.kind == CallKind::Media(MediaCall::LatentPreparation)
            && let Some(prefix_cache) = owners.prefix_cache.as_deref()
        {
            let buffer = call.cache_input.ok_or_else(|| {
                Error::Invalid("image trajectory requires its conditioning prefix cache".into())
            })?;
            let visible = batch.calls[index].output.progress.cache_visible_len;
            prefix_cache.tables().coordinates(slot as u32, visible)?;
            prefix_cache.validate_conditioning(call.request_key, buffer, slot as u32, visible)?;
        }
        if prepared.latent && active {
            let params = latent_params(plan, call)?;
            let request = owners.requests.get(call.request_key.request_id.0)?;
            let image = request.admission().image.as_ref().ok_or_else(|| {
                Error::Invalid("an image trajectory requires image parameters".into())
            })?;
            let ledger = owners.ledger()?;
            if !record.entered {
                ledger.enter(issue)?;
                record.entered = true;
            }
            let bound = ledger.bind_image(slot, call, params, image)?;
            step["calls"][index]["latent"] = bound.to_json();
            size = Some((bound.params.height, bound.params.width));
            let state = &mut batch.calls[index];
            state.forward.products.latent = Some(bound.update);
            state.forward.image.latent = Some((bound.banks, bound.params));
        }

        // Its decoded bytes, lent from the batch's output lease, and their
        // PNG encoding on a reporting rank.
        if prepared.bytes > 0 {
            let (height, width) =
                size.ok_or_else(|| Error::Invalid("a decoded image's size is unknown".into()))?;
            let bytes = prepared.bytes;
            if bytes != PIXEL_BYTES * u64::from(height) * u64::from(width) {
                return Err(Error::Interface(format!(
                    "a {height}x{width} image stages {bytes} bytes"
                )));
            }
            if bytes > call.bounds.max_completion_bytes {
                return Err(Error::Invalid(
                    "image pixels exceed the call's completion byte bound".into(),
                ));
            }
            let count = usize::try_from(bytes)
                .map_err(|_| Error::Invalid("decoded bytes exceed the host range".into()))?;
            let view = lock(&batch.buffer).lend_bytes(count, batch.device, issue, owners.events)?;
            if batch.reports {
                let images = owners.images.ok_or_else(|| {
                    Error::Unsupported("image encoding requires the image host lane".into())
                })?;
                batch.calls[index].forward.image.encoding = Some(images.host.reserve()?);
            }
            tensors.push(view.clone());
            batch.calls[index].forward.image.bytes = Some((view, height, width));
        }
    }

    // A denoising batch's attention rows.
    let denoising = plan
        .calls
        .first()
        .is_some_and(|call| call.kind == CallKind::Media(MediaCall::Denoising));
    if denoising
        && owners.pages.is_some()
        && let Some(columns) = reserve_rows(owners, plan, batch, step, issue)?
    {
        record.ordered = true;
        tensors.push(columns);
    }
    Ok(tensors)
}

/// Reserve the product `write` names of `call` in `target` memory: a
/// feature in the encoder cache, or an image at the scheduler's placement
/// (`allocations`) and the prepared shape.
fn reserve_write(
    store: &ProductStore,
    call: &Call,
    write: &PreparedWrite,
    target: MemoryKind,
    allocations: &HashMap<BufferId, BufferAllocation>,
) -> Result<ProductBuffer> {
    match write.field {
        Field::EncoderOutput => {
            let product = call.encoder_output.clone().ok_or_else(|| {
                Error::Interface("the executor prepared an undeclared feature".into())
            })?;
            store
                .reserve_features(
                    &[(product, target)],
                    allocations,
                    OutputPlacement::default(),
                )?
                .pop()
                .ok_or_else(|| Error::Invariant("a feature was not reserved".into()))
        }
        Field::ImageOutput => {
            let product = call.image_output.clone().ok_or_else(|| {
                Error::Interface("the executor prepared an undeclared image".into())
            })?;
            let shapes = HashMap::from([(product.buffer_id(), write.shape.clone())]);
            store
                .bind_outputs(
                    &[(product, target)],
                    OutputPlacement {
                        allocations: Some(allocations),
                        shapes: Some(&shapes),
                        ..OutputPlacement::default()
                    },
                )?
                .pop()
                .ok_or_else(|| Error::Invariant("an image was not bound".into()))
        }
    }
}

/// Reserve what a denoising batch's active calls use of the prefix cache,
/// recording the stream work into `issue`, and return the host
/// `int64 [5 + 2G, R]` columns of their attention rows; `None` without an
/// active call.
///
/// Every slot the calls' forward rows address and each call's own slot get
/// their block tables installed and their new units reset; a row on another
/// slot holds one of the call's guidance prefixes, which the slot keeps
/// until the trajectory finishes; every row's extent is retained until the
/// batch's device work retires (writers their prefix and queries, readers
/// their prefix). Each call's conditioning is its request's exported
/// context.
///
/// The rows, per active call in call order: its forward rows in the plan's
/// order, then one conditioning row. Fields: call index, slot, sequence
/// length, query length, cache write, then per cache group the first
/// selected page and the page count. A row reads `seq_len - query_len`
/// tokens of its slot, but a read-only row on a slot the call also
/// prefills (a write row whose sequence is its query) reads that prefill's
/// tokens; the conditioning row reads the request's visible context on its
/// own slot, with no query of its own.
fn reserve_rows(
    owners: &mut Owners<'_>,
    plan: &BatchPlan,
    batch: &mut Batch<LaneCall>,
    step: &mut serde_json::Value,
    issue: Recorder<'_>,
) -> Result<Option<StorageView>> {
    let active: Vec<usize> = (0..plan.calls.len())
        .filter(|&index| batch.calls[index].active)
        .collect();
    if active.is_empty() {
        return Ok(None);
    }
    let forward = &plan.forward;
    let rows = ForwardRows::new(&forward.call_indices, plan.calls.len())?;

    // Conditioning K/V and earlier prefixes were written by batches over
    // the prefix cache, possibly on another stream.
    if let Some(pages) = owners.pages.as_deref() {
        pages.order().wait(issue)?;
    }

    // Install every addressed slot's tables and reset their new units.
    let slots: BTreeSet<u32> = active
        .iter()
        .flat_map(|&index| {
            rows.of(index)
                .iter()
                .map(|&row| forward.request_pool_indices[row])
                .chain(std::iter::once(batch.calls[index].slot as u32))
        })
        .collect();
    let pages = owners
        .pages
        .as_deref()
        .ok_or_else(|| Error::Invariant("a denoising lane lost its pages".into()))?;
    let prefix_cache = owners
        .prefix_cache
        .as_deref_mut()
        .ok_or_else(|| Error::Unsupported("image trajectories require a prefix cache".into()))?;
    pages.install(prefix_cache, plan, |slot| slots.contains(&slot), issue)?;

    let mut attention = Vec::new();
    let mut described: Vec<[i64; FORWARD_FIELDS]> = Vec::new();
    for &index in &active {
        let call = &plan.calls[index];
        let key = call.request_key;
        let own = batch.calls[index].slot as u32;
        let visible = batch.calls[index].output.progress.cache_visible_len;

        // The request's own conditioning prefix: the context it exported.
        let buffer = call.cache_input.ok_or_else(|| {
            Error::Invalid("image trajectory requires its conditioning prefix cache".into())
        })?;
        prefix_cache.tables().coordinates(own, visible)?;
        prefix_cache.validate_conditioning(key, buffer, own, visible)?;

        let call_rows = rows.of(index);
        let mut alternates = BTreeSet::new();
        for &row in call_rows {
            let slot = forward.request_pool_indices[row];
            let (seq_len, query_len) = (forward.seq_lens[row], forward.query_lens[row]);
            if seq_len < query_len {
                return Err(Error::Invalid(
                    "a forward row's query exceeds its sequence".into(),
                ));
            }
            if slot != own {
                if seq_len - query_len > prefix_cache.tables().allocated_length(slot) {
                    return Err(Error::Invalid(
                        "forward row exceeds alternative-prefix capacity".into(),
                    ));
                }
                prefix_cache.retain_prefix(key, slot);
                alternates.insert(slot);
            }
            // Readers retain their prefix, writers also their query tokens.
            let length = u64::from(if forward.write_cache[row] {
                seq_len
            } else {
                seq_len - query_len
            });
            prefix_cache.retain(key, slot, length, &batch.retirement)?;
        }

        for &row in call_rows {
            let slot = forward.request_pool_indices[row];
            let (seq_len, query_len, write) = (
                forward.seq_lens[row],
                forward.query_lens[row],
                forward.write_cache[row],
            );
            let prefix = if write {
                seq_len - query_len
            } else {
                prefill_prefix(forward, call_rows, slot).unwrap_or(seq_len - query_len)
            };
            attention.push(AttentionRow {
                slot,
                prefix,
                query: query_len,
                write,
            });
            described.push([
                index as i64,
                i64::from(slot),
                i64::from(seq_len),
                i64::from(query_len),
                i64::from(write),
            ]);
        }
        let visible32 = u32::try_from(visible)
            .map_err(|_| Error::Invalid("a conditioning prefix exceeds the token range".into()))?;
        let query = call_rows
            .last()
            .map_or(1, |&row| forward.query_lens[row].max(1));
        attention.push(AttentionRow {
            slot: own,
            prefix: visible32,
            query,
            write: false,
        });
        described.push([index as i64, i64::from(own), i64::from(visible32), 0, 0]);

        let allocated: Vec<[u32; 2]> = alternates
            .iter()
            .copied()
            .chain(std::iter::once(own))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(|slot| [slot, prefix_cache.tables().allocated_length(slot)])
            .collect();
        step["calls"][index]["image"]["allocated"] = serde_json::json!(allocated);
        batch.calls[index].forward.image.alternates = alternates.into_iter().collect();
    }

    // Field-major, each cache group's first selected page and page count of
    // every row follow the rows' own fields.
    let selected = attention_pages(prefix_cache, &attention)?;
    let mut columns = field_major(&described);
    for group in &selected {
        columns.extend(group.starts.iter().chain(&group.counts));
    }
    let fields = FORWARD_FIELDS + 2 * selected.len();
    host_tensor(&columns, &[fields as i64, attention.len() as i64]).map(Some)
}

/// The tokens a call prefills into `slot` in this batch: the query of its
/// write row on `slot` whose sequence is its query (a guidance prefix
/// written from position zero), which a read-only row on `slot` reads.
fn prefill_prefix(forward: &ipc::ForwardBatch, rows: &[usize], slot: u32) -> Option<u32> {
    rows.iter().copied().find_map(|row| {
        (forward.write_cache[row]
            && forward.request_pool_indices[row] == slot
            && forward.seq_lens[row] == forward.query_lens[row])
            .then_some(forward.query_lens[row])
    })
}

/// Record, behind the batch's execution, the publication of what its image
/// calls wrote: each product fenced at its prepared shape with its metadata
/// and, for a feature, its export to other ranks and workers; each
/// trajectory version's export; the PNG encodings of decoded images; the
/// release of a finished trajectory's guidance prefixes; and the batch's
/// place in the prefix cache's and the ledger's stream orders. The host
/// pixels the execution copied from return to their pool once the stream
/// passed it.
pub(crate) fn publish(
    owners: &mut Owners<'_>,
    plan: &BatchPlan,
    batch: &mut Batch<LaneCall>,
    record: &mut ImageBatch,
    job: &Job,
) -> Result<()> {
    let stream = batch.stream.clone();
    let issue = job.on(stream.as_ref());
    release_pixels(record, owners.events, issue)?;
    let exports = owners.transports.exports_remotely() && batch.reports;
    for (index, call) in plan.calls.iter().enumerate() {
        let state = &mut batch.calls[index];
        // Products, then their exports.
        let writes = std::mem::take(&mut state.forward.image.writes);
        for image in &writes {
            owners
                .store
                .produced(&image.write, &image.shape, issue, image.metadata)?;
            if image.field == Field::EncoderOutput
                && let Some(export) = Publications::issue_feature(
                    owners.store,
                    owners.transports,
                    call,
                    &image.write,
                    batch.reports,
                    issue,
                )?
            {
                state
                    .forward
                    .products
                    .exports
                    .push((image.write.clone(), export));
            }
        }
        state.forward.image.writes = writes;

        // The trajectory version the call wrote, for readers elsewhere.
        if let (Some((banks, params)), Some(product), true) = (
            state.forward.image.latent.clone(),
            call.latent_output.as_ref(),
            exports,
        ) {
            let step_index = match call.kind {
                CallKind::Media(MediaCall::LatentPreparation) => 0,
                CallKind::Media(MediaCall::Denoising) => banks.step + 1,
                _ => continue,
            };
            let step_index = u32::try_from(step_index)
                .map_err(|_| Error::Invariant("a latent step exceeds u32".into()))?;
            let slot = state.slot;
            let ledger = owners
                .ledger
                .as_deref_mut()
                .ok_or_else(|| Error::Invariant("a trajectory lost its ledger".into()))?;
            let (pool, storage) = ledger.parts();
            let export = latents::issue_export(
                pool,
                storage,
                owners.transports,
                call,
                product,
                slot,
                &params,
                banks.write,
                step_index,
                issue,
            )?;
            record.versions.push(export.buffer());
            batch.calls[index].forward.image.exports.push(export);
        }
    }

    // PNG encodings of decoded images, read from the lease once its fences
    // completed.
    for (index, call) in plan.calls.iter().enumerate() {
        let image = &batch.calls[index].forward.image;
        if let (Some(task), Some((pixels, height, width))) = (&image.encoding, &image.bytes) {
            // The encoding reads the lease after its fences, as a CPU reader
            // the task releases when it ends.
            lock(&batch.buffer).retain_reader()?;
            task.configure(ImageTask::new(ImageWork::Encode {
                pixels: pixels.clone(),
                lease: Arc::clone(&batch.buffer),
                height: *height,
                width: *width,
                maximum: usize::try_from(call.bounds.max_completion_bytes).unwrap_or(usize::MAX),
            }))?;
        }
    }

    // A finished trajectory's guidance prefixes retire with it.
    for (index, call) in plan.calls.iter().enumerate() {
        let state = &batch.calls[index];
        let (Some((banks, _)), false) = (
            &state.forward.image.latent,
            state.forward.image.alternates.is_empty(),
        ) else {
            continue;
        };
        if call.kind != CallKind::Media(MediaCall::Denoising) {
            continue;
        }
        let steps = owners
            .requests
            .get(call.request_key.request_id.0)?
            .admission()
            .image
            .as_ref()
            .map_or(0, |image| i64::from(image.steps));
        if banks.step + 1 < steps {
            continue;
        }
        let alternates = state.forward.image.alternates.clone();
        owners
            .prefix_cache()?
            .release_prefixes(call.request_key, &alternates, issue)?;
    }
    if std::mem::take(&mut record.ordered)
        && let Some(pages) = owners.pages.as_deref_mut()
    {
        // Later batches over the prefix cache on other streams order after
        // the guidance prefixes this batch wrote and the slots it released.
        pages.order_mut().record(issue)?;
    }

    if std::mem::take(&mut record.entered) {
        owners.ledger()?.leave(issue)?;
    }
    Ok(())
}

/// Return the host pixels the batch's execution copies from to their pool
/// once the stream passed the work `issue` issued before (the execution).
fn release_pixels(record: &mut ImageBatch, events: &EventPool, issue: Recorder<'_>) -> Result<()> {
    let pixels = std::mem::take(&mut record.pixels);
    if pixels.is_empty() || issue.stream().is_none() {
        return Ok(());
    }
    let fence = events.fence(issue, false)?;
    issue.defer(
        events,
        vec![fence],
        Box::new(move || {
            drop(pixels);
            Ok(())
        }),
    )
}

/// Publish the trajectory exports the batch's calls issued, now that their
/// work was performed: their locations register with the batch's
/// publications and their descriptors join the calls' completions.
pub(crate) fn accept(
    ledger: Option<&LatentLedger>,
    calls: &mut [CallState<LaneCall>],
    publications: &mut Publications,
) -> Result<()> {
    for state in calls {
        let exports = std::mem::take(&mut state.forward.image.exports);
        if exports.is_empty() {
            continue;
        }
        let pool = ledger
            .ok_or_else(|| Error::Invariant("exported trajectories lost their ledger".into()))?
            .pool();
        for export in exports {
            latents::locate_export(pool, export, publications, &mut state.output)?;
        }
    }
    Ok(())
}

/// Withdraw what a batch that failed before its commit point issued for its
/// image calls, recording the stream work into `issue`: unpublished
/// trajectory exports are revoked and every version it exported is released
/// from the ledger's pool, its PNG encodings are cancelled, its host pixels
/// return to their pool behind what the stream performed, and its use of
/// the ledger's banks ends.
pub(crate) fn retire(
    ledger: Option<&mut LatentLedger>,
    events: &EventPool,
    calls: &mut [CallState<LaneCall>],
    record: &mut ImageBatch,
    issue: Recorder<'_>,
) -> Result<()> {
    for state in calls.iter_mut() {
        for export in std::mem::take(&mut state.forward.image.exports) {
            export.revoke();
        }
        if let Some(task) = state.forward.image.encoding.take() {
            // An unconfigured encoding returns its lane capacity.
            let _ = task.cancel(true);
        }
    }
    let versions = std::mem::take(&mut record.versions);
    let entered = std::mem::take(&mut record.entered);
    release_pixels(record, events, issue)?;
    match ledger {
        Some(ledger) => {
            if !versions.is_empty() {
                ledger.pool().release_exports(&versions);
            }
            if entered {
                ledger.leave(issue)?;
            }
            Ok(())
        }
        None if versions.is_empty() && !entered => Ok(()),
        None => Err(Error::Invariant(
            "a failed image batch lost its latent ledger".into(),
        )),
    }
}

/// The host work of a batch's decoded images: their PNG encodings, run
/// once its fences completed; `None` without one.
pub(crate) fn host_stage(calls: &mut [CallState<LaneCall>]) -> Option<Box<dyn HostStage>> {
    let encodings: Vec<(usize, Arc<HostTask<ImageTask>>)> = calls
        .iter_mut()
        .enumerate()
        .filter_map(|(index, state)| Some((index, state.forward.image.encoding.take()?)))
        .collect();
    (!encodings.is_empty()).then(|| {
        Box::new(Encodings {
            encodings,
            submitted: false,
        }) as Box<dyn HostStage>
    })
}

/// The PNG encodings of a decoding batch's images, run once their pixels
/// reached the host.
struct Encodings {
    /// Each call's index in the batch and its encoding.
    encodings: Vec<(usize, Arc<HostTask<ImageTask>>)>,
    submitted: bool,
}

impl HostStage for Encodings {
    fn poll(&mut self) -> Result<bool> {
        if !self.submitted {
            // The output fences completed: the pixels are on the host.
            for (_, task) in &self.encodings {
                task.submit()?;
            }
            self.submitted = true;
        }
        Ok(self
            .encodings
            .iter()
            .all(|(_, task)| task.completion.done()))
    }

    fn finish(
        &mut self,
        _plan: &BatchPlan,
        _products: ProductOwners<'_>,
        outputs: &mut [PendingOutput],
    ) -> Result<()> {
        // The encoded images are delivered as media; nothing is published
        // as a product.
        for (call, task) in std::mem::take(&mut self.encodings) {
            let output = outputs
                .get_mut(call)
                .ok_or_else(|| Error::Invariant("an image encoding names no call".into()))?;
            match task.completion.outcome() {
                Some(Outcome::Success(result)) => {
                    if let ImageResult::Encoded(bytes) = &*result
                        && output.reports_output
                    {
                        let source = ipc::MediaSource::publish(bytes).map_err(|error| {
                            Error::Invalid(format!("publishing an encoded image failed: {error}"))
                        })?;
                        output.set_media(source)?;
                    }
                }
                Some(Outcome::Failed(error)) => {
                    tracing::error!("image encoding failed: {error}");
                    output.output.status = CallStatus::Error;
                    output.output.error_code = Some(ErrorCode::ComputeError);
                }
                // The call's host task always completes before its batch
                // materializes, so a missing outcome breaks the batch's order.
                Some(Outcome::Cancelled) | None => {
                    output.output.status = CallStatus::Error;
                    output.output.error_code = Some(ErrorCode::InvariantViolation);
                }
            }
        }
        Ok(())
    }

    fn cancel(&mut self) {
        for (_, task) in &self.encodings {
            let _ = task.cancel(false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A preparation's image entries parse per call: absent and null
    /// entries are calls without image work; writes name their product
    /// field, shape and metadata; and the decoded bytes of every call add up
    /// to the words the batch leases.
    #[test]
    fn prepared_image_entries_parse_per_call() -> Result<()> {
        let step = serde_json::json!({"prepared": [
            {"reads": [], "writes": [], "publication": null},
            {"reads": [], "writes": [], "publication": null, "image": null},
            {"reads": [], "writes": [], "publication": null, "image": {
                "read": true,
                "writes": [
                    {"field": "encoder_output", "shape": [4, 8],
                     "metadata": {"feature": {"height": 32, "width": 32}}},
                    {"field": "image_output", "shape": [3, 2, 2],
                     "metadata": {"image": {"height": 2, "width": 2, "value_range": "signed_unit"}}}
                ],
                "bytes": 12,
                "latent": true
            }},
            {"image": {"pixels": true, "bytes": 5}}
        ]});
        let entries = prepared(&step)?;
        assert!(entries[0].is_none() && entries[1].is_none());
        let image = entries[2].as_ref().ok_or(Error::State("an image entry"))?;
        assert!(image.read && image.latent && !image.pixels);
        assert_eq!(image.writes[0].field, Field::EncoderOutput);
        assert_eq!(image.writes[1].shape, [3, 2, 2]);
        assert_eq!(
            image.writes[1]
                .metadata
                .as_ref()
                .map(PreparedMetadata::metadata)
                .transpose()?,
            Some(Metadata::Image {
                height: 2,
                width: 2,
                value_range: Some(ValueRange::SignedUnit),
            })
        );
        // 12 + 5 bytes take three words.
        assert_eq!(lease_words(&step)?, 3);

        let unknown = serde_json::json!({"prepared": [{"image": {"writes": [
            {"field": "latent_output", "shape": [1]}
        ]}}]});
        assert!(prepared(&unknown).is_err());
        Ok(())
    }

    /// A call's entry states its inline image's facts or a resident
    /// image's shape, size and value range, its request's input images and
    /// its progress; a resident image without image metadata is refused.
    #[test]
    fn a_call_describes_its_input_image_and_progress() -> Result<()> {
        let progress = RequestProgress {
            logical_position: 40,
            cache_visible_len: 38,
            ..RequestProgress::default()
        };
        let facts = ImageFacts {
            shape: vec![1024, 588],
            grid: Some([32, 32]),
            height: 448,
            width: 448,
        };
        let inline = described(Input::Inline(facts), 2, &progress)?;
        assert_eq!(
            inline,
            serde_json::json!({
                "input": {"inline": {
                    "shape": [1024, 588], "grid": [32, 32], "height": 448, "width": 448
                }},
                "input_images": 2,
                "progress": {"logical": 40, "visible": 38},
            })
        );

        let metadata = Metadata::Image {
            height: 64,
            width: 32,
            value_range: Some(ValueRange::SignedUnit),
        };
        let resident = described(
            Input::Resident(vec![3, 64, 32], Some(metadata)),
            0,
            &progress,
        )?;
        assert_eq!(
            resident["input"],
            serde_json::json!({"resident": {
                "shape": [3, 64, 32], "height": 64, "width": 32, "value_range": "signed_unit"
            }})
        );
        assert_eq!(
            described(Input::None, 0, &progress)?["input"],
            serde_json::Value::Null
        );
        let feature = Metadata::Feature {
            height: 64,
            width: 32,
        };
        assert!(described(Input::Resident(vec![4, 8], Some(feature)), 0, &progress).is_err());
        Ok(())
    }

    fn forward(rows: &[(u32, u32, u32, bool)]) -> ipc::ForwardBatch {
        ipc::ForwardBatch {
            call_indices: vec![0; rows.len()],
            request_pool_indices: rows.iter().map(|row| row.0).collect(),
            seq_lens: rows.iter().map(|row| row.1).collect(),
            query_lens: rows.iter().map(|row| row.2).collect(),
            write_cache: rows.iter().map(|row| row.3).collect(),
        }
    }

    /// A read-only row on a slot its call prefills in the same batch reads
    /// the prefilled tokens; a row on a slot without such a prefill reads
    /// its scheduler context.
    #[test]
    fn a_branch_reads_the_prefix_its_call_prefills() {
        // A six-token prefill of slot 3, then branch rows on slots 3 and 4
        // and the call's own slot 1.
        let rows = forward(&[
            (3, 6, 6, true),
            (3, 20, 20, false),
            (4, 29, 20, false),
            (1, 31, 20, false),
        ]);
        let all: Vec<usize> = (0..4).collect();
        assert_eq!(prefill_prefix(&rows, &all, 3), Some(6));
        assert_eq!(prefill_prefix(&rows, &all, 4), None);
        assert_eq!(prefill_prefix(&rows, &all, 1), None);
        // A write row that continues a resident prefix is no prefill.
        let continued = forward(&[(3, 8, 2, true)]);
        assert_eq!(prefill_prefix(&continued, &[0], 3), None);
    }
}
