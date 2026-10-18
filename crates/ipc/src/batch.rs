//! Execution batches: the planned calls of one numerical call on one
//! component.
//!
//! These are the semantic types of the scheduler-to-worker protocol,
//! independent of wire framing. The engine lowers each scheduled batch into a
//! [`Batch`] (`ExecutionBatch::into_protocol`, which validates it), projects
//! it onto the calls each rank owns (`rank_projection`, which narrows forward
//! rows with [`ForwardBatch::select`] and sets each call's `consumer_slots`),
//! and sends it as the message of a [`Frame`]. A rank answers with one
//! [`BatchOutput`] of per-call [`RequestOutput`] completions, which it
//! validates where it builds the response. The rank runtime
//! (`uniserve-worker`) consumes these types directly; a frame carries a
//! message its sender validated, and its receiver does not check it again.
//!
//! Besides its calls, a batch carries the cache page tables of its requests
//! ([`BlockTable`], with the units newly acquired for them in
//! [`CacheUnitAllocation`]), its forward rows ([`ForwardBatch`]), latent
//! trajectories ([`LatentParams`]), decode cursors ([`DecodeRange`]),
//! persistent buffer spans ([`BufferAllocation`]) and request commands
//! ([`BatchCommand`]). [`Batch::validate`] reaches each nested `validate` and
//! adds the checks that span calls.

use super::*;

/// One request slot's complete unit table in one cache group.
///
/// `unit_ids` lists the units of logical pages `start_page..`, page-major:
/// the group's `units_per_page` units of page `start_page` come first. Pages
/// before `start_page` have been released by a sliding-window group, and a
/// worker reads none of them. `allocated_tokens` is the absolute token extent
/// the table covers, up to the end of its last page; the worker checks it
/// against the group's page shape, which this record does not carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockTable {
    /// Scheduler-assigned request-state row that owns the table.
    pub request_pool_idx: u32,
    /// Cache group addressed by the table.
    pub group_id: u32,
    /// First logical page the units cover; earlier pages are retired.
    pub start_page: u32,
    /// Physical unit identifiers of pages `start_page..`, page-major.
    pub unit_ids: Vec<UnitId>,
    /// Absolute token extent the table covers.
    pub allocated_tokens: u32,
}

/// Physical units newly acquired for one installed block table.
///
/// The worker resets every listed unit before a call reads or writes it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheUnitAllocation {
    /// Scheduler-assigned request-state row receiving the units.
    pub request_pool_idx: u32,
    /// Cache group receiving the units.
    pub group_id: u32,
    /// Newly allocated physical unit identifiers.
    pub unit_ids: Vec<UnitId>,
}

/// Whether `units` are real units (never the zero sentinel) without repeats.
fn distinct_units(units: &[UnitId]) -> bool {
    units.iter().all(|unit| unit.0 > 0) && units.iter().collect::<HashSet<_>>().len() == units.len()
}

impl BlockTable {
    /// Validates table identity and unit uniqueness.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.request_pool_idx > 0,
            "block table request slot must be positive"
        );
        ensure_valid!(
            distinct_units(&self.unit_ids),
            "block table repeats a unit or carries unit zero"
        );
        Ok(())
    }
}

impl CacheUnitAllocation {
    /// Validates newly allocated unit identities and ownership.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.request_pool_idx > 0,
            "cache-unit allocation request slot must be positive"
        );
        ensure_valid!(
            !self.unit_ids.is_empty() && distinct_units(&self.unit_ids),
            "cache-unit allocation is empty, repeats a unit, or carries unit zero"
        );
        Ok(())
    }
}

/// Aligned columns defining the model's forward rows, independent of wire framing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForwardBatch {
    /// Physical call index for each row; logical computation IDs remain unchanged.
    pub call_indices: Vec<u32>,
    /// Scheduler-owned request slots, including alternative CFG prefixes.
    pub request_pool_indices: Vec<u32>,
    /// Total attention lengths (cached prefix plus query), not allocated capacities.
    pub seq_lens: Vec<u32>,
    /// Query tokens evaluated by each row.
    pub query_lens: Vec<u32>,
    /// Whether the query interval persists in the prefix cache after its visible prefix.
    pub write_cache: Vec<bool>,
}

impl ForwardBatch {
    /// Append one model row to every column in the same order.
    pub fn push(
        &mut self,
        call_index: u32,
        request_pool_index: u32,
        seq_len: u32,
        query_len: u32,
        write_cache: bool,
    ) {
        self.call_indices.push(call_index);
        self.request_pool_indices.push(request_pool_index);
        self.seq_lens.push(seq_len);
        self.query_lens.push(query_len);
        self.write_cache.push(write_cache);
    }

    /// Move forward inputs into a physical batch, offsetting only physical row ownership.
    ///
    /// Fails without modifying this batch when an offset call index exceeds `u32`.
    pub fn append(&mut self, other: Self, call_offset: u32) -> ValidationResult<()> {
        // Offset every index before extending any column so a failure leaves
        // the columns aligned.
        let call_indices = other
            .call_indices
            .into_iter()
            .map(|index| {
                index.checked_add(call_offset).ok_or_else(|| {
                    invalid_message!(
                        "physical call index {index} with offset {call_offset} exceeds u32"
                    )
                })
            })
            .collect::<ValidationResult<Vec<_>>>()?;

        self.call_indices.extend(call_indices);
        self.request_pool_indices.extend(other.request_pool_indices);
        self.seq_lens.extend(other.seq_lens);
        self.query_lens.extend(other.query_lens);
        self.write_cache.extend(other.write_cache);
        Ok(())
    }

    /// Select a rank's calls and map their forward rows to the local call array.
    ///
    /// `calls` lists the batch-level call indices the rank owns, in the order
    /// of its projected call array. Rows of other calls are dropped, and each
    /// kept row's call index becomes its position in `calls`.
    pub fn select(&self, calls: &[usize]) -> Self {
        let mut selected = Self::default();
        for (row, index) in self.call_indices.iter().enumerate() {
            if let Some(local) = calls.iter().position(|call| *call == *index as usize) {
                selected.push(
                    local as u32,
                    self.request_pool_indices[row],
                    self.seq_lens[row],
                    self.query_lens[row],
                    self.write_cache[row],
                );
            }
        }
        selected
    }

    /// Validate column alignment and numerical ranges before indexing model inputs.
    pub fn validate(&self, call_count: usize) -> ValidationResult<()> {
        let rows = self.call_indices.len();
        ensure_valid!(
            self.request_pool_indices.len() == rows
                && self.seq_lens.len() == rows
                && self.query_lens.len() == rows
                && self.write_cache.len() == rows,
            "forward input columns have different lengths"
        );
        ensure_valid!(
            self.call_indices
                .iter()
                .all(|index| (*index as usize) < call_count),
            "forward call index is outside its batch"
        );
        ensure_valid!(
            self.request_pool_indices.iter().all(|slot| *slot > 0),
            "forward input carries the reserved request slot"
        );
        ensure_valid!(
            self.query_lens.iter().all(|length| *length > 0),
            "forward query length must be positive"
        );
        ensure_valid!(
            self.seq_lens
                .iter()
                .zip(&self.query_lens)
                .all(|(total, query)| total >= query),
            "forward sequence length is shorter than its query"
        );
        Ok(())
    }
}

/// Solver-step range and optional paged storage for one request trajectory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentParams {
    /// Request that owns the trajectory.
    pub request_key: RequestKey,
    /// Call that addresses the trajectory.
    pub call_id: CallId,
    /// Physical latent pages in logical order; empty for request-owned tensors.
    pub page_table: Vec<u32>,
    /// Logical units stored in pages; zero when the trajectory uses request tensors.
    pub latent_units: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Output width in pixels.
    pub width: u32,
    /// First denoising step assigned to the call.
    pub start_step: u32,
    /// Number of denoising steps assigned to the call.
    pub step_count: u32,
}

impl LatentParams {
    /// Validates latent page identities, shape, and byte bounds.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.call_id.batch_id > 0,
            "latent params call id must be positive"
        );
        ensure_valid!(
            self.height > 0 && self.width > 0,
            "latent params geometry must be positive"
        );
        ensure_valid!(
            (!self.page_table.is_empty() == (self.latent_units > 0))
                && self.page_table.iter().all(|page| *page > 0)
                && self.page_table.iter().collect::<HashSet<_>>().len() == self.page_table.len(),
            "latent params page table disagrees with its units, repeats a page, or carries page zero"
        );
        Ok(())
    }
}

/// Cursor and unit bounds for one diffusion decode call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeRange {
    /// Request that owns the media output.
    pub request_key: RequestKey,
    /// Decode call receiving the params.
    pub call_id: CallId,
    /// First decoder unit assigned to the call.
    pub cursor: u32,
    /// Maximum decoder units the call may process.
    pub max_units: u32,
}

impl DecodeRange {
    /// Validates decode identity and unit capacity.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.call_id.batch_id > 0,
            "decode params call id must be positive"
        );
        ensure_valid!(
            self.max_units > 0,
            "decode params unit bound must be positive"
        );
        Ok(())
    }
}

/// Scheduler-selected address span for one cross-call persistent buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferAllocation {
    /// Persistent buffer identity receiving the address span.
    pub buffer: BufferId,
    /// Byte offset within the persistent-buffer arena.
    pub offset: u64,
    /// Length of the reserved span in bytes.
    pub bytes: u64,
}

impl BufferAllocation {
    /// Validates the buffer identity and a non-empty span whose end does not
    /// overflow `u64`.
    pub fn validate(self) -> ValidationResult<()> {
        self.buffer.validate()?;
        ensure_valid!(self.bytes > 0, "buffer params byte extent must be positive");
        ensure_valid!(
            self.offset.checked_add(self.bytes).is_some(),
            "buffer params span overflows"
        );
        Ok(())
    }
}

/// One numerical call on one component, with the calls of every request that
/// participates in it. Execution domains, attention selection, and captured
/// buckets are derived by the worker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Batch {
    /// Batch identity assigned by the executor, strictly increasing in each
    /// worker's Submit order.
    pub batch_id: u64,
    /// Monotonic sequence shared by collective participants.
    pub collective_seq: u64,
    /// Calls executed by this invocation.
    pub calls: Vec<Call>,
    /// Complete scheduler-owned cache page tables required by the calls.
    pub block_tables: Vec<BlockTable>,
    /// Physical cache units newly allocated for this invocation.
    pub new_cache_units: Vec<CacheUnitAllocation>,
    /// Columnar model inputs carried directly by the worker message.
    pub forward: ForwardBatch,
    /// Scheduler-owned latent trajectory allocations.
    pub latent_params: Vec<LatentParams>,
    /// Scheduler-owned media decode allocations.
    pub decode_ranges: Vec<DecodeRange>,
    /// Address spans for persistent cross-call buffers.
    pub buffer_allocations: Vec<BufferAllocation>,
    /// Ordered request-state and buffer-lifetime commands.
    pub commands: Vec<BatchCommand>,
    /// Host-supplied input product values matched by `TensorRef` identity.
    pub input_products: Vec<TensorExport>,
    /// Imported cache exports consumed by explicit cache installation call kinds.
    pub cache_inputs: Vec<CacheTransfer>,
}

impl Batch {
    /// Constructs a run with admissions and calls using default metadata.
    ///
    /// `collective_seq` is `batch_id`, or one when `batch_id` is zero;
    /// [`Self::validate`] requires it to be positive.
    pub fn new(batch_id: u64, admissions: Vec<NewRequest>, calls: Vec<Call>) -> Self {
        Self {
            batch_id,
            collective_seq: batch_id.max(1),
            calls,
            block_tables: Vec::new(),
            new_cache_units: Vec::new(),
            forward: ForwardBatch::default(),
            latent_params: Vec::new(),
            decode_ranges: Vec::new(),
            buffer_allocations: Vec::new(),
            commands: admissions
                .into_iter()
                .map(|request| BatchCommand::Start {
                    request: Box::new(request),
                })
                .collect(),
            input_products: Vec::new(),
            cache_inputs: Vec::new(),
        }
    }

    /// Iterates over calls in submission order.
    pub fn calls(&self) -> impl Iterator<Item = &Call> {
        self.calls.iter()
    }

    /// Returns the number of calls in this run.
    pub fn call_count(&self) -> usize {
        self.calls.len()
    }

    /// Attaches ordered control commands to the run.
    pub fn with_commands(mut self, commands: Vec<BatchCommand>) -> Self {
        self.commands.extend(commands);
        self
    }

    /// Iterates over static request admissions in submission order.
    pub fn admissions(&self) -> impl Iterator<Item = &NewRequest> {
        self.commands.iter().filter_map(|command| match command {
            BatchCommand::Start { request } => Some(request.as_ref()),
            _ => None,
        })
    }

    /// Validates identities, call kinds, allocations, products, and commands.
    ///
    /// Runs every nested `validate`, including [`Call::validate`] for each
    /// call, and checks what spans calls, including one call per request, one
    /// kind and component per batch, allocation tables that match the calls
    /// addressing them, non-overlapping latent pages and buffer spans, and
    /// input products and cache inputs that some call declares.
    ///
    /// This is the contract of the batch a rank receives, whose buffer spans
    /// address that rank's own buffer pool.
    pub fn validate(&self) -> ValidationResult<()> {
        self.validate_batch(true)
    }

    /// Validates a worker group's batch before any rank's buffer spans are
    /// chosen.
    ///
    /// Each rank of a group backs its own buffer pool, so a group's batch
    /// carries no single set of spans: this checks everything
    /// [`Self::validate`] does except the spans, and requires the batch to
    /// carry none. Each rank's projection then passes [`Self::validate`]
    /// with the spans of that rank's pool.
    pub fn validate_unplaced(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.buffer_allocations.is_empty(),
            "an unplaced batch carries no buffer allocations"
        );
        self.validate_batch(false)
    }

    /// Validates the batch, and its buffer spans when `placed`.
    fn validate_batch(&self, placed: bool) -> ValidationResult<()> {
        // Establish the run envelope before validating relationships within it.
        ensure_valid!(
            !self.calls.is_empty() || !self.commands.is_empty(),
            "a submission batch must carry at least one call or control"
        );
        ensure_valid!(
            self.collective_seq > 0,
            "run collective sequence must be positive"
        );

        let mut computation_ids = HashSet::with_capacity(self.calls.len());
        for call in &self.calls {
            call.validate()?;
            ensure_valid!(
                call.call_id.batch_id == self.batch_id,
                "computation identity belongs to another logical batch"
            );
            ensure_valid!(
                computation_ids.insert(call.call_id),
                "a submission batch repeats a computation identity"
            );
        }
        // A batch is one numerical call on one component: every call in it
        // performs the same computation through the same component, so a rank
        // executes it as a single homogeneous group and returns one result.
        ensure_valid!(
            self.calls.windows(2).all(|pair| {
                pair[0].code == pair[1].code && pair[0].component == pair[1].component
            }),
            "a submission batch mixes call kinds or components"
        );

        let calls = self
            .calls
            .iter()
            .map(|call| ((call.request_key, call.call_id), call))
            .collect::<HashMap<_, _>>();

        // Cache allocations are subsets of unique block tables for the same pool group.
        let mut table_ids = HashSet::with_capacity(self.block_tables.len());
        for table in &self.block_tables {
            table.validate()?;
            ensure_valid!(
                table_ids.insert((table.request_pool_idx, table.group_id)),
                "run repeats a block table"
            );
        }

        let tables = self
            .block_tables
            .iter()
            .map(|table| ((table.request_pool_idx, table.group_id), table))
            .collect::<HashMap<_, _>>();

        let mut allocation_ids = HashSet::with_capacity(self.new_cache_units.len());
        for allocation in &self.new_cache_units {
            allocation.validate()?;
            let identity = (allocation.request_pool_idx, allocation.group_id);
            ensure_valid!(
                allocation_ids.insert(identity),
                "run repeats a cache-unit allocation"
            );
            let table = tables.get(&identity).ok_or_else(|| {
                invalid_message!("cache-unit allocation has no matching block table")
            })?;
            let table_units = table.unit_ids.iter().collect::<HashSet<_>>();
            ensure_valid!(
                allocation
                    .unit_ids
                    .iter()
                    .all(|unit| table_units.contains(unit)),
                "cache-unit allocation is outside its block table"
            );
        }

        self.forward.validate(self.calls.len())?;

        // Latent pages are exclusive across every trajectory placed in a run.
        let mut latent_ids = HashSet::with_capacity(self.latent_params.len());
        let mut latent_pages = HashSet::new();
        for params in &self.latent_params {
            params.validate()?;
            let identity = (params.request_key, params.call_id);
            ensure_valid!(
                latent_ids.insert(identity),
                "run repeats a latent params identity"
            );
            let call = calls
                .get(&identity)
                .ok_or_else(|| invalid_message!("latent params does not name a run call"))?;
            let addresses_trajectory = matches!(
                call.code,
                CallKind::Media(MediaCall::LatentPreparation)
                    | CallKind::Media(MediaCall::Denoising)
            ) || call.latent_input.is_some();
            ensure_valid!(
                addresses_trajectory,
                "latent params names work without a trajectory"
            );
            ensure_valid!(
                params
                    .page_table
                    .iter()
                    .all(|page| latent_pages.insert(*page)),
                "run latent allocations overlap physical pages"
            );
        }

        for call in &self.calls {
            let needs_latent = matches!(
                call.code,
                CallKind::Media(MediaCall::LatentPreparation)
                    | CallKind::Media(MediaCall::Denoising)
            ) || call.latent_input.is_some();
            ensure_valid!(
                !needs_latent || latent_ids.contains(&(call.request_key, call.call_id)),
                "call that addresses a trajectory has no latent params"
            );
        }

        // Decode ranges belong exactly to video and audio decoding and encoding
        // calls: each such call has one, and no other call does.
        let mut decode_ids = HashSet::with_capacity(self.decode_ranges.len());
        for params in &self.decode_ranges {
            params.validate()?;
            let identity = (params.request_key, params.call_id);
            ensure_valid!(
                decode_ids.insert(identity),
                "run repeats a decode params identity"
            );
            let call = calls
                .get(&identity)
                .ok_or_else(|| invalid_message!("decode params does not name a run call"))?;
            // A video request's latent encoding also covers a range: the
            // visual condition units of one round, or its single audio call.
            ensure_valid!(
                matches!(
                    call.code,
                    CallKind::Media(
                        MediaCall::VideoDecoding
                            | MediaCall::AudioDecoding
                            | MediaCall::VideoEncoding
                            | MediaCall::AudioEncoding
                            | MediaCall::LatentEncoding
                    )
                ),
                "decode params does not name media decode work"
            );
            // Audio reconstruction divides the sample timeline into as many
            // media units as its component has ranks, and one round covers
            // them all, so an audio decode range starts at the first unit.
            ensure_valid!(
                !matches!(call.code, CallKind::Media(MediaCall::AudioDecoding))
                    || (params.cursor == 0 && params.max_units >= 1),
                "audio decode range must start at the first media unit"
            );
            // Audio encoding consumes the assembled track as one host call.
            ensure_valid!(
                !matches!(call.code, CallKind::Media(MediaCall::AudioEncoding))
                    || (params.cursor == 0 && params.max_units == 1),
                "audio encode range must address its single sample stream"
            );
        }

        for call in &self.calls {
            let needs_range = matches!(
                call.code,
                CallKind::Media(
                    MediaCall::VideoDecoding
                        | MediaCall::AudioDecoding
                        | MediaCall::VideoEncoding
                        | MediaCall::AudioEncoding
                )
            );
            ensure_valid!(
                !needs_range || decode_ids.contains(&(call.request_key, call.call_id)),
                "media reconstruction call has no decode range"
            );
        }

        if placed {
            // Persistent buffers use non-overlapping half-open spans and cover every
            // declared output. `BufferAllocation::validate` has already rejected
            // an overflowing span end, so the addition below cannot wrap.
            let mut buffer_ids = HashSet::with_capacity(self.buffer_allocations.len());
            let mut buffer_spans = self
                .buffer_allocations
                .iter()
                .map(|params| {
                    params.validate()?;
                    ensure_valid!(
                        buffer_ids.insert(params.buffer),
                        "run repeats a buffer params identity"
                    );
                    Ok((params.offset, params.offset + params.bytes))
                })
                .collect::<ValidationResult<Vec<_>>>()?;
            buffer_spans.sort_unstable();

            ensure_valid!(
                buffer_spans.windows(2).all(|pair| pair[0].1 <= pair[1].0),
                "run buffer allocations overlap"
            );

            for call in &self.calls {
                for output in call.buffer_outputs() {
                    let params = self
                        .buffer_allocations
                        .iter()
                        .find(|params| params.buffer == output.buffer_id())
                        .ok_or_else(|| {
                            invalid_message!("persistent call output has no buffer params")
                        })?;
                    ensure_valid!(
                        params.bytes >= output.max_bytes(),
                        "buffer params is smaller than its declared output"
                    );
                }
            }
        }

        // Submission depth is one runnable call per request.
        let mut request_keys = HashSet::with_capacity(self.call_count());
        for call in &self.calls {
            ensure_valid!(
                request_keys.insert(call.request_key),
                "a submission batch carries multiple calls for one request"
            );
        }

        // Admissions create request state independently of call execution;
        // each is validated once, as the `Start` command below.
        let mut admitted = HashSet::new();
        for admission in self.admissions() {
            ensure_valid!(
                admitted.insert(admission.request_key),
                "a submission batch carries a duplicate admission"
            );
        }

        // A repeated control identity must carry exactly the same command.
        let mut command_identities: std::collections::HashMap<
            (RequestKey, u8, Option<BufferId>),
            BatchCommand,
        > = std::collections::HashMap::new();
        for command in &self.commands {
            command.validate()?;
            let identity = (
                command.request_key(),
                command.variant_index(),
                match command {
                    BatchCommand::Free { buffer } => Some(*buffer),
                    _ => None,
                },
            );
            if let Some(existing) = command_identities.get(&identity) {
                ensure_valid!(
                    existing == command,
                    "a submission batch reuses a command identity with different content"
                );
            } else {
                command_identities.insert(identity, command.clone());
            }
        }

        // Each input product payload must be valid, declared as an input or
        // predicate by some call, and supplied at most once.
        let mut supplied_inputs = HashSet::with_capacity(self.input_products.len());
        for payload in &self.input_products {
            ensure_valid!(
                self.calls().any(|call| match &payload.value {
                    TransferHandle::Encoder {
                        payload_kind: FeatureKind::Vision,
                        ..
                    } => {
                        call.vision_inputs
                            .iter()
                            .any(|block| block.feature == payload.product)
                    }
                    TransferHandle::Encoder {
                        payload_kind: FeatureKind::Latent,
                        ..
                    } => {
                        call.latent_feature_input.as_ref() == Some(&payload.product)
                    }
                    TransferHandle::DeviceProduct { .. } => {
                        call.inputs.contains(&payload.product)
                            || [&call.token_input, &call.image_input, &call.predicate]
                                .into_iter()
                                .any(|input| input.as_ref() == Some(&payload.product))
                    }
                    TransferHandle::Latent { .. } => {
                        call.latent_input.as_ref() == Some(&payload.product)
                    }
                }),
                "an input transfer has no consumer for its payload kind"
            );
            ensure_valid!(
                supplied_inputs.insert(&payload.product),
                "a submission batch repeats an input product payload"
            );
            payload.validate()?;
        }

        // Each imported cache export feeds exactly one installation call and
        // fits that call's transfer-byte bound.
        let mut cache_sources = HashSet::new();
        for export in &self.cache_inputs {
            export.validate()?;
            ensure_valid!(
                cache_sources.insert(export.source),
                "run repeats a cache input"
            );
            let consumers = self
                .calls()
                .filter(|call| call.cache_input == Some(export.source))
                .collect::<Vec<_>>();
            ensure_valid!(
                consumers.len() == 1
                    && consumers[0].code == CallKind::Transfer(TransferMode::CacheInstall),
                "cache transfer requires one installation consumer"
            );
            let bytes = export.tensors().try_fold(0_u64, |sum, tensor| {
                Ok::<_, ValidationError>(sum.saturating_add(tensor.validate()?))
            })?;
            ensure_valid!(
                bytes <= consumers[0].bounds.max_transfer_bytes,
                "cache input exceeds its installation transfer-byte bound"
            );
        }

        Ok(())
    }
}
