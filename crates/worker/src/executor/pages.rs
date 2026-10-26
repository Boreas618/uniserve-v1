//! The prefix cache pages of the computations whose calls attend to them.
//!
//! A computation whose manifest names the `pages` resource computes over the
//! prefix cache: its calls read (and a text call writes) the K/V of their
//! request slots' block tables. Every such computation computes on the device's
//! primary stream. The worker keeps what the executor does not decide:
//!
//! - [`Pages`], what those computations share: the primary stream's unit
//!   resets, the order of their batches across streams ([`StreamOrder`]), which
//!   cache transfers and image batches reading the cache wait on, and the block
//!   diffusion a generating rank serves;
//! - [`PageReservation`], what one batch reserves before the executor runs
//!   it: the active calls' block tables installed, their newly assigned
//!   units reset (those the host KV cache filled excepted), the extents
//!   they read and write retained until the batch's device work retires,
//!   the scalar products they relay bound, their relayed predicates leased,
//!   and every active row's attention pages selected
//!   (`PrefixCache::prepare_attention`, whose write check keeps exports,
//!   imports and executions of the units apart);
//! - [`RelayStaging`], the device columns the executor writes the relayed
//!   values to, which [`relay`] copies into their products by call.
//!
//! A token-denoising call reads its request's unchanged prefix: it receives
//! no units, copies no host KV cache page, writes no cache and publishes
//! only its completion flag, which its step's continuation decides on the
//! device. Its canvas step advances the request's submitted canvas here,
//! before the executor runs it.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cuda::{DeviceId, Stream};
use ipc::{
    Batch as BatchPlan, BlockTable, CallKind, CanvasSampling, DType, ForwardMode, RequestKey,
    ShapeBound, TensorRef,
};

use crate::executor::batch::{Batch, CallState, StreamOrder};
use crate::executor::columns::{ForwardRows, GroupPages, host_tensor};
use crate::executor::computations::CallResources;
use crate::ffi::{Backing, DType as Element, MemoryKind};
use crate::job::{Job, Recorder};
use crate::memory::allocation::{Storage, StorageView};
use crate::memory::{AttentionRow, OutputPlacement, PrefixCache, ProductBuffer, UnitColumn};
use crate::request::RequestPool;
use crate::{Error, ProductStore, Result};

/// What the rank's cache-backed computations share.
pub(crate) struct Pages {
    device: Option<DeviceId>,
    /// The device's primary stream, which every cache-backed computation
    /// computes on; `None` for computations on the host.
    stream: Option<Stream>,
    /// The unit resets of the primary stream.
    column: UnitColumn,
    /// Orders what committing computations' batches write (the K/V of their
    /// prefills, slot state) before work of other streams that reads it:
    /// recorded behind each such batch's commit, waited on by the next
    /// cache-backed batch and by transfers and image batches.
    order: StreamOrder,
    /// The block diffusion a generating diffusion language model serves.
    canvas: Option<CanvasSampling>,
}

impl Pages {
    pub(crate) fn new(
        device: Option<DeviceId>,
        stream: Option<Stream>,
        column: UnitColumn,
        canvas: Option<CanvasSampling>,
    ) -> Self {
        Self {
            device,
            stream,
            column,
            order: StreamOrder::default(),
            canvas,
        }
    }

    pub(crate) fn device(&self) -> Option<DeviceId> {
        self.device
    }

    pub(crate) fn stream(&self) -> Option<&Stream> {
        self.stream.as_ref()
    }

    /// The order of cache-backed batches across streams; work on another
    /// stream that reads what they wrote waits on it.
    pub(crate) fn order(&self) -> &StreamOrder {
        &self.order
    }

    /// The order, for work on other streams that also writes what
    /// cache-backed batches read (cache imports, guidance prefixes, latent
    /// images' K/V).
    pub(crate) fn order_mut(&mut self) -> &mut StreamOrder {
        &mut self.order
    }

    /// The unit reset column of the primary stream, for work that assigns K/V
    /// units beside the cache-backed batches.
    pub(crate) fn unit_column(&self) -> &UnitColumn {
        &self.column
    }

    /// Install `plan`'s block tables of the slots `holds` selects and reset
    /// their newly assigned units, recording the stream work into `issue`:
    /// before any of the batch's writes. Units the host KV cache's reads fill
    /// on the copy stream are overwritten whole there, and are not recycled.
    pub(crate) fn install(
        &self,
        prefix_cache: &mut PrefixCache,
        plan: &BatchPlan,
        holds: impl Fn(u32) -> bool,
        issue: Recorder<'_>,
    ) -> Result<()> {
        let tables: Vec<BlockTable> = plan
            .block_tables
            .iter()
            .filter(|table| holds(table.request_pool_idx))
            .cloned()
            .collect();
        prefix_cache.install(&tables, issue)?;
        let loaded: HashSet<u32> = plan
            .page_loads
            .iter()
            .flat_map(|copy| copy.units.iter().map(|unit| unit.0))
            .collect();
        let units: Vec<u32> = plan
            .new_cache_units
            .iter()
            .filter(|allocation| holds(allocation.request_pool_idx))
            .flat_map(|allocation| allocation.unit_ids.iter().map(|unit| unit.0))
            .filter(|unit| !loaded.contains(unit))
            .collect();
        if units.is_empty() {
            return Ok(());
        }
        prefix_cache.recycle(&units, &self.column, issue)
    }
}

/// The pages each cache group selects for `rows`, in the rows' order
/// (`PrefixCache::prepare_attention`). A group selects one table per unit
/// of its pages, all with the group's starts and lengths, so its first
/// table stands for the group.
pub(crate) fn attention_pages(
    prefix_cache: &mut PrefixCache,
    rows: &[AttentionRow],
) -> Result<Vec<GroupPages>> {
    let shapes = prefix_cache.tables().groups().to_vec();
    let selected = prefix_cache.prepare_attention(rows)?;
    let mut table = 0;
    let mut pages = Vec::with_capacity(shapes.len());
    for shape in &shapes {
        let selection = selected.get(table).ok_or_else(|| {
            Error::Invariant("attention pages disagree with the cache groups".into())
        })?;
        pages.push(GroupPages {
            starts: selection
                .starts
                .iter()
                .map(|&start| i64::from(start))
                .collect(),
            counts: selection
                .lengths
                .iter()
                .map(|&count| count as i64)
                .collect(),
        });
        table += shape.units_per_page as usize;
    }
    Ok(pages)
}

/// The device columns the relayed scalars of a computation's batches pass
/// through: the executor writes each executed call's values at the call's
/// index, and the batch copies them into the call's products on the primary
/// stream behind the executor's work.
pub(crate) struct RelayStaging {
    /// `int64 [calls]`: each call's sampled token and continuation.
    tokens: StorageView,
    /// `uint8 [2, calls]`: row 0 each call's transition decision, row 1
    /// its completion flag.
    flags: StorageView,
}

/// The row of the staging flags holding transition decisions.
const TRANSITIONS: i64 = 0;

/// The row of the staging flags holding completion flags.
const COMPLETIONS: i64 = 1;

impl RelayStaging {
    /// Staging for batches of at most `calls` calls on `device` (host
    /// memory without one).
    pub(crate) fn allocate(device: Option<DeviceId>, calls: usize) -> Result<Self> {
        let calls = calls.max(1);
        let bytes = calls * (size_of::<i64>() + 2);
        let memory: Arc<dyn Backing> = match device {
            Some(device) => Storage::device(device, bytes)?,
            None => Storage::host(bytes),
        };
        Ok(Self {
            tokens: StorageView::contiguous(Arc::clone(&memory), 0, Element::I64, &[calls as i64])?,
            flags: StorageView::contiguous(
                memory,
                calls * size_of::<i64>(),
                Element::U8,
                &[2, calls as i64],
            )?,
        })
    }

    /// The columns the executor writes: tokens, then flags.
    pub(crate) fn views(&self) -> [StorageView; 2] {
        [self.tokens.clone(), self.flags.clone()]
    }

    /// Calls the staging holds.
    fn calls(&self) -> usize {
        self.tokens.shape()[0] as usize
    }

    /// The staged values of calls `start..end` of `row` (`None`: the
    /// tokens).
    fn run(&self, row: Option<i64>, start: usize, end: usize) -> Result<StorageView> {
        let column = match row {
            None => self.tokens.clone(),
            Some(row) => self.flags.select(0, row)?,
        };
        column.narrow(0, start as i64, (end - start) as i64)
    }
}

/// What a cache-backed call contributes to its batch beside the state
/// every batch keeps of it.
#[derive(Default)]
pub(crate) struct PageCall {
    /// The call's sampled token relay, bound for an executing call.
    pub(crate) token_write: Option<ProductBuffer>,
    /// The call's transition decision relay.
    pub(crate) transition_write: Option<ProductBuffer>,
    /// The tokens the call's slot's tables hold once the batch installed
    /// them.
    pub(crate) allocated: u32,
}

/// Whether `plan`'s calls are token-denoising calls (a batch carries calls
/// of one kind).
pub(crate) fn denoising(plan: &BatchPlan) -> bool {
    plan.calls
        .first()
        .is_some_and(|call| call.kind == CallKind::Forward(ForwardMode::TokenDenoising))
}

/// What a batch of cache-backed calls reserved of the prefix cache: its
/// rows grouped by call, and each active row's selected pages in every
/// cache group.
pub(crate) struct PageReservation {
    pub(crate) rows: ForwardRows,
    pub(crate) pages: Vec<GroupPages>,
}

/// The owners a batch's page reservation borrows.
pub(crate) struct PageOwners<'a> {
    pub(crate) pages: &'a Pages,
    pub(crate) prefix_cache: &'a mut PrefixCache,
    pub(crate) tensors: &'a ProductStore,
    pub(crate) requests: &'a mut RequestPool,
}

impl PageReservation {
    /// Reserve what `batch`'s active calls use of the prefix cache and
    /// bind their relays and predicates, recording the stream work into
    /// `job`, after checking each active call's rows: they read and write
    /// the call's own slot, a text call has one row (a context prefill one
    /// per prompt run and vision block), and a token-denoising call's rows
    /// read its unchanged prefix, a canvas step's one block.
    pub(crate) fn reserve(
        owners: PageOwners<'_>,
        plan: &BatchPlan,
        batch: &mut Batch<CallResources>,
        job: &Job,
    ) -> Result<Self> {
        let PageOwners {
            pages,
            prefix_cache,
            tensors,
            requests,
        } = owners;
        let stream = batch.stream.clone();
        let issue = job.on(stream.as_ref());
        let denoising = denoising(plan);
        let active: HashSet<u32> = batch
            .calls
            .iter()
            .filter(|state| state.active)
            .map(|state| state.slot as u32)
            .collect();
        if denoising {
            if !plan.page_loads.is_empty() || !plan.page_offloads.is_empty() {
                return Err(Error::Invalid(
                    "token-denoising calls copy no pages of the host KV cache".into(),
                ));
            }
            if plan
                .new_cache_units
                .iter()
                .any(|allocation| active.contains(&allocation.request_pool_idx))
            {
                return Err(Error::Invalid(
                    "token-denoising calls write no prefix cache and receive no new units".into(),
                ));
            }
        }

        // The tables of skipped calls are not installed.
        pages.install(prefix_cache, plan, |slot| active.contains(&slot), issue)?;

        let rows = ForwardRows::new(&plan.forward.call_indices, plan.calls.len())?;
        check_rows(pages, prefix_cache, plan, batch, &rows, denoising)?;
        bind_relays(tensors, plan, batch, issue)?;
        lease_predicates(tensors, plan, batch, issue)?;
        let pages = select_pages(prefix_cache, plan, batch, &rows, denoising)?;
        // Each step advances its request's submitted canvas, once the batch
        // reserved everything else; queued steps precede their device
        // completion, and a later failure does not roll the canvas back.
        for (call, state) in plan.calls.iter().zip(&batch.calls) {
            if let (true, Some(step)) = (state.active, call.canvas) {
                requests.advance_canvas(call.request_key, step)?;
            }
        }
        Ok(Self { rows, pages })
    }
}

/// Check each active call's rows and retain the extents they read and
/// write, noting each call's allocated length.
fn check_rows(
    pages: &Pages,
    prefix_cache: &mut PrefixCache,
    plan: &BatchPlan,
    batch: &mut Batch<CallResources>,
    rows: &ForwardRows,
    denoising: bool,
) -> Result<()> {
    let forward = &plan.forward;
    let mut steps = 0;
    let mut readouts = 0;
    for (index, (call, state)) in plan.calls.iter().zip(&mut batch.calls).enumerate() {
        if !state.active {
            continue;
        }
        let slot = state.slot as u32;
        let visible = state.output.progress.cache_visible_len;
        let call_rows = rows.of(index);
        if call_rows.is_empty() {
            return Err(Error::Invalid(
                "a cache-backed call has no forward row".into(),
            ));
        }
        if call_rows
            .iter()
            .any(|&row| forward.request_pool_indices[row] != slot)
        {
            return Err(Error::Invalid(
                "a cache-backed call's forward row reads another slot".into(),
            ));
        }
        // The visible prefix lies within the slot's installed tables.
        prefix_cache.tables().coordinates(slot, visible)?;
        state.forward.pages.allocated = prefix_cache.tables().allocated_length(slot);
        if !denoising {
            if call_rows.len() > 1 && !call.writes_context() {
                return Err(Error::Invalid(
                    "a text call has exactly one forward row".into(),
                ));
            }
            for &row in call_rows {
                // Readers retain their prefix, writers also their query
                // tokens.
                let length = u64::from(if forward.write_cache[row] {
                    forward.seq_lens[row]
                } else {
                    forward.seq_lens[row] - forward.query_lens[row]
                });
                prefix_cache.retain(call.request_key, slot, length, &batch.retirement)?;
            }
            continue;
        }

        // A token-denoising call's rows read the request's unchanged
        // prefix: they write no cache and each evaluates tokens.
        if call_rows
            .iter()
            .any(|&row| forward.write_cache[row] || forward.query_lens[row] == 0)
        {
            return Err(Error::Invalid(
                "block rows must read the request's unchanged prefix".into(),
            ));
        }
        let queries: usize = call_rows
            .iter()
            .map(|&row| forward.query_lens[row] as usize)
            .sum();
        match (&call.readout, call.canvas) {
            (Some(_), None) => {
                if queries != call.input_token_ids.len() {
                    return Err(Error::Invalid(
                        "block rows must partition the call's tokens over its prefix".into(),
                    ));
                }
                readouts += 1;
            }
            (None, Some(step)) => {
                let sampling = pages.canvas.ok_or_else(|| {
                    Error::Invalid("this worker keeps no generating blocks".into())
                })?;
                let length = sampling.canvas_length;
                if call_rows.len() != 1
                    || call.bounds.max_tokens != length
                    || forward.query_lens[call_rows[0]] != length
                {
                    return Err(Error::Invalid(
                        "a block step is one read-only block row over its prefix".into(),
                    ));
                }
                if step.step >= sampling.max_steps {
                    return Err(Error::Invalid(
                        "the block step exceeds its step limit".into(),
                    ));
                }
                steps += 1;
            }
            _ => {
                return Err(Error::Invalid(
                    "a token-denoising call carries a readout or a canvas step".into(),
                ));
            }
        }
        prefix_cache.retain(call.request_key, slot, visible, &batch.retirement)?;
    }
    if steps > 0 && readouts > 0 {
        return Err(Error::Invalid(
            "a batch carries readouts or denoising steps, not both".into(),
        ));
    }
    Ok(())
}

/// Bind the batch's scalar products: the token relays of active calls and
/// the completion and transition flags of every call; skipped calls publish
/// false flags now. A token-denoising call publishes only a `uint8`
/// completion flag; no cache-backed call publishes a tensor product.
fn bind_relays(
    tensors: &ProductStore,
    plan: &BatchPlan,
    batch: &mut Batch<CallResources>,
    issue: Recorder<'_>,
) -> Result<()> {
    let target = batch.target();
    let slots = batch.slots(plan);
    let mut groups: indexmap::IndexMap<(DType, &ShapeBound), Vec<(TensorRef, MemoryKind)>> =
        indexmap::IndexMap::new();
    for (call, state) in plan.calls.iter().zip(&batch.calls) {
        if !call.outputs.is_empty() || call.image_output.is_some() || call.encoder_output.is_some()
        {
            return Err(Error::Unsupported(
                "cache-backed calls publish no tensor products besides their relays".into(),
            ));
        }
        if call.kind == CallKind::Forward(ForwardMode::TokenDenoising) {
            if call.token_output.is_some()
                || call.transition_output.is_some()
                || call.cache_output.is_some()
            {
                return Err(Error::Unsupported(
                    "token-denoising calls publish only a completion flag".into(),
                ));
            }
            if call
                .completion_output
                .as_ref()
                .is_some_and(|flag| flag.dtype != DType::U8)
            {
                return Err(Error::Invalid(
                    "a denoising step's completion flag is a uint8 scalar".into(),
                ));
            }
        }
        for scalar in [
            call.token_output.as_ref().filter(|_| state.active),
            call.completion_output.as_ref(),
            call.transition_output.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            // One relay group per dtype and shape.
            groups
                .entry((scalar.dtype, &scalar.shape_bound))
                .or_default()
                .push((scalar.clone(), target));
        }
    }
    let groups: Vec<_> = groups.into_values().collect();
    let bound = tensors.bind_output_groups(
        &groups,
        OutputPlacement {
            request_slots: Some(&slots),
            ..OutputPlacement::default()
        },
    )?;
    let owners: HashMap<RequestKey, usize> = plan
        .calls
        .iter()
        .enumerate()
        .map(|(index, call)| (call.request_key, index))
        .collect();
    for write in bound.into_iter().flatten() {
        let id = write.id();
        batch.writes.push(write.clone());
        let index = *owners
            .get(&id.owner)
            .ok_or_else(|| Error::Invariant("a relay names no call".into()))?;
        let call = &plan.calls[index];
        let state = &mut batch.calls[index];
        let names =
            |value: &Option<TensorRef>| value.as_ref().is_some_and(|value| value.buffer_id() == id);
        if names(&call.token_output) {
            state.forward.pages.token_write = Some(write);
        } else if names(&call.transition_output) {
            state.forward.pages.transition_write = Some(write);
        } else if names(&call.completion_output) {
            state.completion_write = Some(write);
        }
    }
    for state in batch.calls.iter().filter(|state| !state.active) {
        for write in [
            &state.completion_write,
            &state.forward.pages.transition_write,
        ]
        .into_iter()
        .flatten()
        {
            tensors.write_scalar(write, 0, issue)?;
        }
    }
    Ok(())
}

/// Lease every active call's relayed (`int64`) predicate on the primary stream,
/// which waits on each producer's fence: the executor gates the call on its
/// slot state, which the producer's commit advanced before.
fn lease_predicates(
    tensors: &ProductStore,
    plan: &BatchPlan,
    batch: &mut Batch<CallResources>,
    issue: Recorder<'_>,
) -> Result<()> {
    let requests: Vec<_> = plan
        .calls
        .iter()
        .zip(&batch.calls)
        .filter(|(_, state)| state.active)
        .filter_map(|(call, _)| {
            call.predicate
                .clone()
                .filter(|value| value.dtype == DType::I64)
                .map(|predicate| (predicate, call.call_id, None))
        })
        .collect();
    if !requests.is_empty() {
        let leased = tensors.consume_batch(&requests, Some(batch.target()), issue)?;
        batch.reads.extend(leased);
    }
    Ok(())
}

/// Select each active row's attention pages, by cache group. A text row
/// is the scheduler's, writing over `seq_len - query_len` visible tokens;
/// a token-denoising row reads its call's visible prefix and writes
/// nothing (the tables of skipped readouts may be uninstalled).
fn select_pages(
    prefix_cache: &mut PrefixCache,
    plan: &BatchPlan,
    batch: &Batch<CallResources>,
    rows: &ForwardRows,
    denoising: bool,
) -> Result<Vec<GroupPages>> {
    let forward = &plan.forward;
    let groups = prefix_cache.tables().groups().len();
    let mut prepared = Vec::new();
    let mut attention = Vec::new();
    for (index, state) in batch.calls.iter().enumerate() {
        if !state.active {
            continue;
        }
        for &row in rows.of(index) {
            let query = forward.query_lens[row];
            let prefix = if denoising {
                u32::try_from(state.output.progress.cache_visible_len)
                    .map_err(|_| Error::Invalid("a block prefix exceeds the token range".into()))?
            } else {
                forward.seq_lens[row] - query
            };
            prepared.push(row);
            attention.push(AttentionRow {
                slot: forward.request_pool_indices[row],
                prefix,
                query,
                write: !denoising,
            });
        }
    }
    let mut pages = vec![GroupPages::empty(forward.call_indices.len()); groups];
    if attention.is_empty() {
        return Ok(pages);
    }
    // The selection is in `prepared`'s order; the columns are by plan row.
    let selected = attention_pages(prefix_cache, &attention)?;
    for (group, selection) in pages.iter_mut().zip(selected) {
        for (position, &row) in prepared.iter().enumerate() {
            group.starts[row] = selection.starts[position];
            group.counts[row] = selection.counts[position];
        }
    }
    Ok(pages)
}

/// Each warmup step's page columns: a host `int64 [2G, rows]` tensor
/// holding, per cache group `g`, each row's first selected page and page
/// count, over the scratch tables warmup installed. Rows `[slot, prefix,
/// query]` write when `writes`.
pub(crate) fn warmup_pages(
    prefix_cache: &mut PrefixCache,
    steps: &[Vec<[u32; 3]>],
    writes: bool,
) -> Result<Vec<StorageView>> {
    let mut columns = Vec::with_capacity(steps.len());
    for rows in steps {
        let attention: Vec<AttentionRow> = rows
            .iter()
            .map(|&[slot, prefix, query]| AttentionRow {
                slot,
                prefix,
                query,
                write: writes,
            })
            .collect();
        let pages = attention_pages(prefix_cache, &attention)?;
        let words: Vec<i64> = pages
            .iter()
            .flat_map(|group| group.starts.iter().chain(&group.counts).copied())
            .collect();
        columns.push(host_tensor(
            &words,
            &[2 * pages.len() as i64, rows.len() as i64],
        )?);
    }
    Ok(columns)
}

/// Copy the relayed values the executor staged into their products, in
/// runs of consecutive calls: a text call's sampled token and transition
/// decision, a token-denoising call's completion flag.
pub(crate) fn relay(
    tensors: &ProductStore,
    staging: &RelayStaging,
    plan: &BatchPlan,
    batch: &Batch<CallResources>,
    job: &Job,
) -> Result<()> {
    if plan.calls.len() > staging.calls() {
        return Err(Error::Invalid(format!(
            "a batch of {} calls exceeds its computation's relay staging",
            plan.calls.len()
        )));
    }
    let stream = batch.stream.clone();
    let issue = job.on(stream.as_ref());
    type Pick = fn(&CallState<CallResources>) -> Option<&ProductBuffer>;
    let relays: Vec<(Option<i64>, Pick)> = if denoising(plan) {
        vec![(Some(COMPLETIONS), |state| state.completion_write.as_ref())]
    } else {
        vec![
            (None, |state| state.forward.pages.token_write.as_ref()),
            (Some(TRANSITIONS), |state| {
                state.forward.pages.transition_write.as_ref()
            }),
        ]
    };
    for (row, pick) in relays {
        let writes: Vec<(usize, ProductBuffer)> = batch
            .calls
            .iter()
            .enumerate()
            .filter(|(_, state)| state.active)
            .filter_map(|(index, state)| Some((index, pick(state)?.clone())))
            .collect();
        let mut start = 0;
        while start < writes.len() {
            let mut end = start + 1;
            while end < writes.len() && writes[end].0 == writes[end - 1].0 + 1 {
                end += 1;
            }
            let values = staging.run(row, writes[start].0, writes[end - 1].0 + 1)?;
            let run: Vec<ProductBuffer> = writes[start..end]
                .iter()
                .map(|(_, write)| write.clone())
                .collect();
            tensors.write_scalars(&run, &values, issue)?;
            start = end;
        }
    }
    Ok(())
}
