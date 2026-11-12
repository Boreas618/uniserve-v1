//! The recurrent state a causal language model keeps per request slot: its buffers, and the
//! copies between them and the KV cache's state pages.
//!
//! A model whose layers keep a recurrent state (`nn::family::TextModel::state`) states the
//! fields one layer keeps. [`StateMemory`] owns, for every field of every such layer:
//!
//! - the **live** buffer, `[slots, …]`: each slot's state after its request's last computed
//!   token, which every row of a batch reads and advances in place;
//! - the **checkpoint** buffer, `[slots, …]`: the state a row keeps at the page boundary the
//!   batch's `checkpoints` column names;
//! - in a deployment that speculates, the **trail** buffer, `[tokens, …]`: the state after
//!   every token of a verification, whose accepted length is known only once its tokens are
//!   sampled. The runner then settles each row's slot on the trail's entry after its last
//!   accepted token ([`StateMemory::settle`]).
//!
//! Request slot `s` (slots count from one) owns row `s - 1` of the live and checkpoint buffers
//! ([`StateMemory::row`]); a padding row of a batch states `nn::NO_SLOT` in the batch's state
//! slot column and reads and writes no row.
//!
//! The buffers are taken once from the device's pool of memory that lives as long as the rank
//! serves, so every captured graph addresses them unchanged.
//!
//! ## Copies
//!
//! Between forwards the state moves between the buffers and the cache's state pages through
//! `cache_block_snapshot` and `cache_block_scatter` ([`StateMemory::copy`]): a keep gathers a
//! slot's checkpoint row into the snapshot and scatters it into a page's planes after the
//! forward, and a restore gathers a page's planes into the snapshot and scatters them into the
//! slot's live row before it. Which rows copy what is the scheduler's decision, stated in the
//! batch (`ipc::execution::StateCopy`).
//!
//! A checkpoint lays each layer's fields out in field order from plane boundaries
//! ([`CacheLayout::state_planes`]); plane `p` of a state page is column `q % columns` of the
//! page's unit `q / columns` in field run `p / (units_per_page · columns)`, `q` being the
//! plane's position in its run.
//!
//! The two kernels name the rows they move by I64 index vectors on the device, the element type
//! they read them in. A batch's vectors are written from the host as the copies are issued,
//! into one of [`RING`] sets, so a set is written again only once the copies that read it
//! completed. The kernels launch through `backend::launch` under a context of the state's own,
//! over the stream of the runner that holds the state and the generation the rank serves from;
//! it allocates nothing, the snapshot and the index vectors being the state's memory.
//!
//! Like the cache, the state has a form that holds no device ([`StateMemory::meta`]): its
//! tensors are meta tensors and its copies are checked and issue nothing.

use std::collections::BTreeSet;
use std::sync::Arc;

use ipc::execution::{StateCopy, StateDirection};
use nn::RecurrentState;
use nn::family::{StateField, StateShape};
use ops::block_copy::{
    BlockShape, COPY_BLOCK, CacheBlockScatter, CacheBlockSnapshot, ScatterArgs, SnapshotArgs,
};
use platform::{
    Buffer, CopyBatch, DeviceId, DevicePtr, DeviceRange, Event, EventFlags, PoolKind, SourceOrder,
    Stream,
};
use uniserve_tensor::{DType, Tensor};

use crate::memory::cache::{CacheLayout, KvCacheConfig, KvCacheMemory};
use crate::memory::lend;
use crate::{Error, Result};

/// The sets of index vectors the copies write in turn: a batch's copies overlap at most those
/// of the batch before it, so with four sets a set is written again only after the device
/// passed the copies that read it.
const RING: usize = 4;

/// The recurrent state of one rank as it is opened: the model's fields and the capacity the
/// rank resolved for them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateConfig {
    /// The recurrent layers and the fields each keeps.
    pub shape: StateShape,
    /// Request slots: rows of every live and checkpoint buffer.
    pub slots: u32,
    /// The most rows of a batch: a batch carries at most a restore and a keep per row.
    pub rows: u32,
    /// Rows of every trail buffer: the most tokens a verification batch holds. Zero for a
    /// deployment that does not speculate, which keeps no trail.
    pub trail: u32,
}

impl StateConfig {
    /// The device bytes the state takes from the pool that lives as long as the rank serves,
    /// for a cache of `layout`: the live and checkpoint buffers of every field of every layer,
    /// a row per slot, the trail's row per token, and the copies' index vectors. The snapshot
    /// is the scratch pool's ([`CacheLayout::state_snapshot_bytes`]).
    pub fn bytes(&self, layout: &CacheLayout) -> u64 {
        let buffers = (2 * u64::from(self.slots) + u64::from(self.trail))
            * u64::from(self.shape.layers)
            * self.shape.layer_bytes();

        buffers + RING as u64 * self.index_bytes(layout)
    }

    /// The bytes of one set of index vectors: a slot row and a state page's planes for each of
    /// the copies a batch may carry.
    fn index_bytes(&self, layout: &CacheLayout) -> u64 {
        2 * u64::from(self.rows) * (u64::from(layout.checkpoint_planes()) + 1) * 8
    }
}

/// The state's memory on a device.
struct Buffers {
    device: DeviceId,
    /// `[layer][field]` the live buffers.
    live: Vec<Vec<Buffer>>,
    /// `[layer][field]` the checkpoint buffers.
    checkpoint: Vec<Vec<Buffer>>,
    /// `[layer][field]` the trail buffers; empty in a deployment that does not speculate.
    trail: Vec<Vec<Buffer>>,
    /// What one copy moves a state page through, `[page planes, plane_bytes]` bytes.
    snapshot: Buffer,
    /// The copies' index vectors, [`RING`] sets.
    indices: Vec<Buffer>,
    /// Recorded behind the last batch whose copies read each set of index vectors.
    read: Vec<Option<Event>>,
    /// The set the next batch's copies write.
    turn: usize,
    /// The stream the copies are issued on: the stream of the runner that holds the state.
    stream: Stream,
    /// What the two kernels launch under: that stream and the serving generation.
    context: backend::Context,
}

/// The recurrent-state buffers of one rank and the copies that move them.
pub struct StateMemory {
    config: StateConfig,
    /// `[layer][field]` where each field lies in a state page: its first plane and its bytes.
    planes: Vec<Vec<(u32, u64)>>,
    /// The planes of every checkpoint of a state page.
    page_planes: u32,
    /// Bytes of one plane of the cache.
    plane_bytes: u64,
    /// Columns of one unit of the cache.
    columns: u32,
    /// Units one state page spans.
    units_per_page: u32,
    /// Units of the cache's pool, the padding unit included.
    units: u32,
    /// Field runs of a state page.
    runs: u32,
    /// The memory on a device; `None` for a state that holds none.
    device: Option<Buffers>,
}

impl StateMemory {
    /// The state of `config` over the state pages of a cache of `cache`, holding no device:
    /// its tensors are meta tensors, and its copies are checked and issue nothing.
    ///
    /// # Errors
    ///
    /// A config of no slot, layer or field, or a cache whose layout holds no state group for
    /// the checkpoints.
    pub fn meta(config: StateConfig, cache: &KvCacheConfig) -> Result<Self> {
        if config.slots == 0 || config.shape.layers == 0 || config.shape.fields.is_empty() {
            return Err(Error::Invalid(format!(
                "a recurrent state of {} slots and {} layers of {} fields",
                config.slots,
                config.shape.layers,
                config.shape.fields.len()
            )));
        }

        let layout: &CacheLayout = &cache.layout;
        let group = layout
            .state_group()
            .map(|group| &layout.groups[group])
            .ok_or_else(|| {
                Error::Invalid("a model that keeps a recurrent state needs a state group".into())
            })?;
        if layout.state_fields() == 0 {
            return Err(Error::Invalid(
                "the cache holds no state plane for the recurrent state's checkpoints".into(),
            ));
        }

        let planes: Vec<Vec<(u32, u64)>> = (0..config.shape.layers)
            .map(|layer| layout.state_planes(layer, &config.shape.fields))
            .collect();

        // Every field's planes lie inside the checkpoints the layout states.
        let page_planes = layout.checkpoint_planes();
        for (first, bytes) in planes.iter().flatten() {
            if u64::from(*first) + bytes.div_ceil(layout.plane_bytes) > u64::from(page_planes) {
                return Err(Error::Invalid(format!(
                    "a checkpoint field of {bytes} bytes from plane {first} lies outside a state \
                     page of {page_planes} checkpoint planes"
                )));
            }
        }

        Ok(Self {
            config,
            planes,
            page_planes,
            plane_bytes: layout.plane_bytes,
            columns: layout.columns,
            units_per_page: group.units_per_page,
            units: cache.units,
            runs: layout.state_fields(),
            device: None,
        })
    }

    /// The state of `config` over the state pages of a cache of `cache`, on the device of
    /// `allocator`: the buffers and index vectors are taken from its pool that lives as long
    /// as the rank serves, and the snapshot from its scratch pool. The copies are issued on
    /// `stream`, the stream of the runner that holds the state, and launch with `kernels`. The
    /// buffers' contents are unspecified: a request's first row starts its slot's state.
    ///
    /// # Errors
    ///
    /// As [`Self::meta`], and when a pool cannot hold the state's memory.
    pub fn open(
        config: StateConfig,
        cache: &KvCacheConfig,
        allocator: &mut platform::Allocator,
        stream: Stream,
        kernels: Arc<ops::Generation>,
    ) -> Result<Self> {
        let mut state = Self::meta(config, cache)?;
        let shape = &state.config.shape;

        let mut take = |rows: u32| -> Result<Vec<Vec<Buffer>>> {
            (0..shape.layers)
                .map(|_| {
                    shape
                        .fields
                        .iter()
                        .map(|field| {
                            let bytes = u64::from(rows) * field.bytes();
                            Ok(allocator.allocate(PoolKind::Resident, bytes as usize)?)
                        })
                        .collect()
                })
                .collect()
        };
        let live = take(state.config.slots)?;
        let checkpoint = take(state.config.slots)?;
        let trail = if state.config.trail > 0 {
            take(state.config.trail)?
        } else {
            Vec::new()
        };

        let index_bytes = state.config.index_bytes(&cache.layout) as usize;
        let indices = (0..RING)
            .map(|_| allocator.allocate(PoolKind::Resident, index_bytes))
            .collect::<platform::Result<Vec<_>>>()?;
        let snapshot = allocator.allocate(
            PoolKind::Scratch,
            cache.layout.state_snapshot_bytes() as usize,
        )?;

        // The context allocates nothing: every tensor of a copy is over the state's buffers,
        // its snapshot, its index vectors or the cache's state planes.
        let device = allocator.device();
        let ordinal = usize::try_from(device.0)
            .map_err(|_| Error::Invalid(format!("{device} names no GPU ordinal")))?;
        let region = DeviceRange::new(DevicePtr::from_address(snapshot.address()), 0);
        let context = backend::Context::new(ordinal, stream.clone(), kernels, region);

        state.device = Some(Buffers {
            device,
            live,
            checkpoint,
            trail,
            snapshot,
            indices,
            read: (0..RING).map(|_| None).collect(),
            turn: 0,
            stream,
            context,
        });
        Ok(state)
    }

    /// The config the state was opened with.
    pub fn config(&self) -> &StateConfig {
        &self.config
    }

    /// The row of the live and checkpoint buffers that request slot `slot` owns: what the
    /// batch's state slot column states for the slot's row.
    ///
    /// # Errors
    ///
    /// A slot outside `1..=slots`.
    pub fn row(&self, slot: u32) -> Result<u32> {
        if slot == 0 || slot > self.config.slots {
            return Err(Error::Invalid(format!(
                "request slot {slot} is outside the recurrent state's slots 1..={}",
                self.config.slots
            )));
        }
        Ok(slot - 1)
    }

    /// The tensor of `shape` elements of `dtype` `offset` bytes into `buffer` of the device
    /// side, or the meta tensor of that shape.
    fn tensor(
        &self,
        buffer: impl FnOnce(&Buffers) -> &Buffer,
        offset: u64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        match &self.device {
            // SAFETY: the callers bound the window to the buffer, which the state owns for as
            // long as it lives; only the forwards of the runner that holds the state and the
            // copies this state issues, all on one stream, address it.
            Some(device) => unsafe {
                lend(
                    buffer(device).address() + offset,
                    shape,
                    dtype,
                    Some(device.device),
                )
            },
            // SAFETY: a meta tensor has no data and its address is never read.
            None => unsafe { lend(0, shape, dtype, None) },
        }
    }

    /// The tensors of one kind of buffer, `[layer][field]`: each `[rows, …]` in its field's
    /// element type and extents, or `[rows, bytes]` U8 when `raw`, as a copy moves it.
    fn tensors(
        &self,
        rows: u32,
        raw: bool,
        buffers: impl Fn(&Buffers) -> &Vec<Vec<Buffer>>,
    ) -> Result<Vec<Vec<Tensor>>> {
        let shape = |field: &StateField| {
            let mut dims = vec![rows as usize];
            if raw {
                dims.push(field.bytes() as usize);
            } else {
                dims.extend(&field.dims);
            }
            dims
        };

        (0..self.config.shape.layers as usize)
            .map(|layer| {
                self.config
                    .shape
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(index, field)| {
                        let dtype = if raw { DType::U8 } else { field.dtype };
                        self.tensor(
                            |device| &buffers(device)[layer][index],
                            0,
                            &shape(field),
                            dtype,
                        )
                    })
                    .collect()
            })
            .collect()
    }

    /// The live buffers: one entry per recurrent layer, each the layer's fields in the order
    /// of the model's `StateShape`, `[slots, …]`.
    pub fn live(&self) -> Result<Vec<Vec<Tensor>>> {
        self.tensors(self.config.slots, false, |buffers| &buffers.live)
    }

    /// The checkpoint buffers, in the same order and extents as [`Self::live`].
    pub fn checkpoint(&self) -> Result<Vec<Vec<Tensor>>> {
        self.tensors(self.config.slots, false, |buffers| &buffers.checkpoint)
    }

    /// The trail buffers, in the same order, `[trail, …]`; `None` in a deployment that does
    /// not speculate.
    pub fn trail(&self) -> Result<Option<Vec<Vec<Tensor>>>> {
        if self.config.trail == 0 {
            return Ok(None);
        }
        self.tensors(self.config.trail, false, |buffers| &buffers.trail)
            .map(Some)
    }

    /// The state as a forward is lent it, for a batch whose state rows the four columns
    /// describe (`nn::RecurrentState`): `slots` holds [`Self::row`] of each row's request slot
    /// and `nn::NO_SLOT` for a padding row. A `trailing` batch, a verification, also receives
    /// the trail.
    ///
    /// # Errors
    ///
    /// A trailing batch in a deployment whose state keeps no trail.
    pub fn lend(
        &self,
        query_offsets: Tensor,
        seq_lens: Tensor,
        slots: Tensor,
        checkpoints: Tensor,
        trailing: bool,
    ) -> Result<RecurrentState> {
        let trail = if trailing {
            Some(self.trail()?.ok_or_else(|| {
                Error::Invariant(
                    "a verification keeps a trail and the recurrent state holds none: the \
                     deployment does not speculate"
                        .into(),
                )
            })?)
        } else {
            None
        };

        Ok(RecurrentState {
            query_offsets,
            seq_lens,
            slots,
            checkpoints,
            live: self.live()?,
            checkpoint: self.checkpoint()?,
            trail,
        })
    }

    /// The rows of the cache's state runs a state page's planes lie in, in plane order: each
    /// plane's field run and its row there, `column · units + unit`.
    fn cells(&self, units: &[ipc::UnitId]) -> Result<Vec<(usize, i64)>> {
        if units.len() != self.units_per_page as usize {
            return Err(Error::Invalid(format!(
                "a state page of {} units, and the cache's state pages span {}",
                units.len(),
                self.units_per_page
            )));
        }
        if let Some(unit) = units
            .iter()
            .find(|unit| unit.0 == 0 || unit.0 >= self.units)
        {
            return Err(Error::Invalid(format!(
                "a state page names unit {} of a pool of units 1..{}",
                unit.0, self.units
            )));
        }

        let run = u64::from(self.units_per_page) * u64::from(self.columns);
        Ok((0..u64::from(self.page_planes))
            .map(|plane| {
                let position = plane % run;
                let column = position % u64::from(self.columns);
                let unit = units[(position / u64::from(self.columns)) as usize].0;

                (
                    (plane / run) as usize,
                    (column * u64::from(self.units) + u64::from(unit)) as i64,
                )
            })
            .collect())
    }

    /// The descriptor of a block copy over `field`, a `[blocks, width]` byte tensor.
    fn block_shape(field: &ops::Operand) -> BlockShape {
        BlockShape {
            dtype: DType::U8,
            shape: field.shape,
            strides: field.strides,
            index_stride: 1,
            width: field.shape.last(),
            block: COPY_BLOCK,
        }
    }

    /// Launches one move between `field` and `part` of the snapshot by the context installed
    /// on this thread: the rows `blocks` names of `field` into `part` when `gather`, and
    /// `part` into those rows otherwise.
    fn launch_move(gather: bool, field: &Tensor, blocks: &Tensor, part: &Tensor) -> Result<()> {
        let field = backend::operand(field)?;
        let blocks = backend::operand(blocks)?;
        let snapshot = backend::operand(part)?;

        if gather {
            backend::launch::<CacheBlockSnapshot>(
                &Self::block_shape(&field),
                SnapshotArgs {
                    field,
                    blocks,
                    snapshot,
                },
            )?;
        } else {
            backend::launch::<CacheBlockScatter>(
                &Self::block_shape(&field),
                ScatterArgs {
                    field,
                    blocks,
                    snapshot,
                },
            )?;
        }
        Ok(())
    }

    /// Launches one copy of `direction` by the context installed on this thread: between the
    /// row `slot` names of `buffers` (the raw live buffers of a restore, the raw checkpoint
    /// buffers of a keep) and the planes `cells` names of the cache's state `runs`, through
    /// the snapshot. `blocks(first, count)` is the I64 vector of entries
    /// `first .. first + count` of the copy's cell rows, and `snapshot(offset, rows, width)`
    /// the `[rows, width]` bytes `offset` bytes into the snapshot.
    #[expect(
        clippy::too_many_arguments,
        reason = "one copy's direction, its two sides, its indices and where its tensors lie"
    )]
    fn launch_copy(
        &self,
        direction: StateDirection,
        buffers: &[Vec<Tensor>],
        runs: &[Tensor],
        slot: &Tensor,
        cells: &[(usize, i64)],
        blocks: &dyn Fn(usize, usize) -> Result<Tensor>,
        snapshot: &dyn Fn(u64, usize, u64) -> Result<Tensor>,
    ) -> Result<()> {
        // One launch per field of every layer, between the slot's row and the field's planes
        // of the snapshot.
        let rows = |gather: bool| -> Result<()> {
            for (layer, fields) in self.planes.iter().enumerate() {
                for (index, (first, bytes)) in fields.iter().enumerate() {
                    let part = snapshot(u64::from(*first) * self.plane_bytes, 1, *bytes)?;
                    Self::launch_move(gather, &buffers[layer][index], slot, &part)?;
                }
            }
            Ok(())
        };

        // One launch per field run of the page, between the page's planes of the run and
        // their planes of the snapshot.
        let page = |gather: bool| -> Result<()> {
            let mut start = 0;
            while start < cells.len() {
                let run = cells[start].0;
                let count = cells[start..]
                    .iter()
                    .take_while(|(other, _)| *other == run)
                    .count();

                let field = runs.get(run).ok_or_else(|| {
                    Error::Invariant(format!("a state page reaches field run {run}"))
                })?;
                let part = snapshot(start as u64 * self.plane_bytes, count, self.plane_bytes)?;
                Self::launch_move(gather, field, &blocks(start, count)?, &part)?;

                start += count;
            }
            Ok(())
        };

        match direction {
            StateDirection::Keep => {
                rows(true)?;
                page(false)
            }
            StateDirection::Restore => {
                page(true)?;
                rows(false)
            }
        }
    }

    /// The kernels the state's copies launch, as `(operator, key)`: one copy of each direction
    /// stated under a meta context, which launches nothing, so a state that holds no device
    /// states them too.
    ///
    /// # Errors
    ///
    /// When a copy's descriptors refuse the state's tensors.
    pub fn keys(&self) -> Result<BTreeSet<(&'static str, String)>> {
        // SAFETY: meta tensors have no data and their addresses are never read.
        let meta = |shape: &[usize], dtype| unsafe { lend(0, shape, dtype, None) };

        let cells = self.columns as usize * self.units as usize;
        let runs = (0..self.runs)
            .map(|_| meta(&[cells, self.plane_bytes as usize], DType::U8))
            .collect::<Result<Vec<_>>>()?;
        let fields = self
            .config
            .shape
            .fields
            .iter()
            .map(|field| {
                meta(
                    &[self.config.slots as usize, field.bytes() as usize],
                    DType::U8,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let buffers = vec![fields; self.config.shape.layers as usize];

        let units: Vec<ipc::UnitId> = (1..=self.units_per_page).map(ipc::UnitId).collect();
        let page = self.cells(&units)?;

        let context = backend::Context::meta(None);
        context.record_launches();
        {
            let _installed = context.enter();

            for direction in [StateDirection::Restore, StateDirection::Keep] {
                self.launch_copy(
                    direction,
                    &buffers,
                    &runs,
                    &meta(&[1], DType::I64)?,
                    &page,
                    &|_, count| meta(&[count], DType::I64),
                    &|_, rows, width| meta(&[rows, width as usize], DType::U8),
                )?;
            }
        }

        Ok(context
            .take_launches()
            .into_iter()
            .map(|(operator, key, _)| (operator, key))
            .collect())
    }

    /// The generation the copies launch from, from the next copy on: set at a generation
    /// switch.
    pub fn set_kernels(&self, kernels: Arc<ops::Generation>) {
        if let Some(device) = &self.device {
            device.context.set_kernels(kernels);
        }
    }

    /// Issues a batch's copies of `direction` between the slots' buffers and the state pages
    /// of `cache`, on the stream the state was opened on: every restore before the batch's
    /// forward, every keep after it. A restore moves a page's planes into its slot's live row;
    /// a keep moves its slot's checkpoint row into a page's planes. Every copy is checked
    /// before anything is issued, and a state that holds no device issues nothing.
    ///
    /// # Errors
    ///
    /// More copies than the batch's rows allow, a slot outside the state's slots, a page of
    /// another unit count than the cache's state pages span, a unit that is the padding unit
    /// or outside the pool, or a launch's refusal.
    pub fn copy(
        &mut self,
        cache: &KvCacheMemory,
        copies: &[StateCopy],
        direction: StateDirection,
    ) -> Result<()> {
        let mut pages = Vec::new();
        for copy in copies.iter().filter(|copy| copy.direction == direction) {
            pages.push((self.row(copy.slot)?, self.cells(&copy.units)?));
        }
        if pages.len() > 2 * self.config.rows as usize {
            return Err(Error::Invalid(format!(
                "a batch makes {} state copies, and its {} rows allow {}",
                pages.len(),
                self.config.rows,
                2 * self.config.rows
            )));
        }
        if pages.is_empty() {
            return Ok(());
        }

        // The set of index vectors this batch writes: the copies that read it last completed.
        let Some(device) = &mut self.device else {
            return Ok(());
        };
        let turn = device.turn;
        if let Some(read) = device.read[turn].take() {
            read.synchronize()?;
        }

        // Per copy: its slot's row, then the row of each of its page's planes.
        let stride = self.page_planes as usize + 1;
        let mut values: Vec<i64> = Vec::with_capacity(pages.len() * stride);
        for (row, cells) in &pages {
            values.push(i64::from(*row));
            values.extend(cells.iter().map(|(_, cell)| *cell));
        }
        let mut write = CopyBatch::new(SourceOrder::DuringCall);
        write.push(
            device.indices[turn].address(),
            values.as_ptr() as u64,
            std::mem::size_of_val(values.as_slice()),
        );

        // SAFETY: the destination is this state's set of index vectors, which holds the rows
        // of the copies a batch may carry (checked above) and which no copy still reads; the
        // host values are read during the call.
        unsafe { write.submit(&device.stream) }?;

        let Some(device) = &self.device else {
            return Ok(());
        };
        let buffers = match direction {
            StateDirection::Restore => {
                self.tensors(self.config.slots, true, |buffers| &buffers.live)?
            }
            StateDirection::Keep => {
                self.tensors(self.config.slots, true, |buffers| &buffers.checkpoint)?
            }
        };
        let runs = cache.state_runs()?;
        let snapshot = |offset: u64, rows: usize, width: u64| {
            self.tensor(
                |device| &device.snapshot,
                offset,
                &[rows, width as usize],
                DType::U8,
            )
        };

        {
            let _installed = device.context.enter();

            for (index, (_, cells)) in pages.iter().enumerate() {
                let at = index * stride;
                let entries = |first: usize, count: usize| {
                    self.tensor(
                        |device| &device.indices[turn],
                        ((at + first) * 8) as u64,
                        &[count],
                        DType::I64,
                    )
                };

                self.launch_copy(
                    direction,
                    &buffers,
                    &runs,
                    &entries(0, 1)?,
                    cells,
                    &|first, count| entries(1 + first, count),
                    &snapshot,
                )?;
            }
        }

        let read = Event::new(EventFlags::default());
        read.record(&device.stream)?;
        if let Some(device) = &mut self.device {
            device.read[turn] = Some(read);
            device.turn = (turn + 1) % RING;
        }
        Ok(())
    }

    /// Settles request slot `slot` on the trail's entry of token `token`: every layer's live
    /// state becomes the state a verification's forward left after that token, the last its
    /// slot's request accepted. Issued on the stream the state was opened on, behind the
    /// forward that wrote the trail; a state that holds no device checks and issues nothing.
    ///
    /// # Errors
    ///
    /// A state that keeps no trail, a token past the trail, or a slot outside the state's.
    pub fn settle(&self, slot: u32, token: u32) -> Result<()> {
        if token >= self.config.trail {
            return Err(Error::Invariant(format!(
                "a verification settles slot {slot} on token {token} of a trail of {} rows",
                self.config.trail
            )));
        }
        let row = self.row(slot)?;

        let Some(device) = &self.device else {
            return Ok(());
        };
        let mut batch = CopyBatch::new(SourceOrder::Stream);
        for (live, trail) in device.live.iter().zip(&device.trail) {
            for ((live, trail), field) in live.iter().zip(trail).zip(&self.config.shape.fields) {
                let bytes = field.bytes();
                batch.push(
                    live.address() + u64::from(row) * bytes,
                    trail.address() + u64::from(token) * bytes,
                    bytes as usize,
                );
            }
        }

        // SAFETY: both ranges are one field's row inside buffers this state owns, bounded
        // above, and the copies are ordered on the stream behind the forward that wrote the
        // trail and ahead of the next forward that reads the slot.
        unsafe { batch.submit(&device.stream) }?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use ipc::{PrefixCacheDtype, UnitId};
    use ops::Op;

    use super::*;

    /// A recurrent state of two layers: a 12-byte window and 40 bytes of state.
    fn shape() -> StateShape {
        StateShape {
            layers: 2,
            fields: vec![
                StateField {
                    name: "conv",
                    dtype: DType::BF16,
                    dims: vec![2, 3],
                },
                StateField {
                    name: "recurrent",
                    dtype: DType::F32,
                    dims: vec![10],
                },
            ],
        }
    }

    /// The cache of two attention layers of 4-token pages of one 8-wide BF16 head beside that
    /// state: planes of 64 bytes, so a layer's checkpoint is two planes and a state page of
    /// four planes is two units of two columns.
    fn cache() -> KvCacheConfig {
        let layout = CacheLayout::hybrid(4, 2, 1, 8, PrefixCacheDtype::BFloat16, &shape())
            .expect("a hybrid layout");
        assert_eq!(layout.plane_bytes, 64);
        assert_eq!(layout.groups[1].units_per_page, 2);

        KvCacheConfig {
            layout: Arc::new(layout),
            units: 16,
            slots: 3,
            width: 4,
            segments: None,
            workspace_bytes: 64,
        }
    }

    fn config(trail: u32) -> StateConfig {
        StateConfig {
            shape: shape(),
            slots: 3,
            rows: 2,
            trail,
        }
    }

    fn copy(slot: u32, units: &[u32], direction: StateDirection) -> StateCopy {
        StateCopy {
            slot,
            units: units.iter().copied().map(UnitId).collect(),
            direction,
        }
    }

    /// The buffers are one tensor per layer and field, a row per slot in the field's own
    /// extents and element type, and a row per token for the trail.
    #[test]
    fn the_buffers_have_a_row_per_slot_in_each_fields_shape() -> Result<()> {
        let state = StateMemory::meta(config(5), &cache())?;

        for buffers in [state.live()?, state.checkpoint()?] {
            assert_eq!(buffers.len(), 2);
            for layer in &buffers {
                assert_eq!(layer.len(), 2);
                assert_eq!(
                    (layer[0].dims(), layer[0].dtype()),
                    (&[3, 2, 3][..], DType::BF16)
                );
                assert_eq!(
                    (layer[1].dims(), layer[1].dtype()),
                    (&[3, 10][..], DType::F32)
                );
            }
        }

        let trail = state
            .trail()?
            .expect("a speculating deployment keeps a trail");
        assert_eq!(trail[1][0].dims(), [5, 2, 3]);
        assert_eq!(trail[1][1].dims(), [5, 10]);

        // The live and checkpoint buffers of three slots and a trail of five tokens, two
        // layers of 52 bytes each; four sets of index vectors of a slot row and four planes
        // for each of the four copies a batch of two rows may carry.
        let layout = cache().layout;
        assert_eq!(
            config(5).bytes(&layout),
            (2 * 3 + 5) * 2 * 52 + 4 * (4 * 5 * 8)
        );
        assert_eq!(layout.state_snapshot_bytes(), 4 * 64);
        Ok(())
    }

    /// A deployment that does not speculate keeps no trail, and a verification is refused one.
    #[test]
    fn a_state_without_a_trail_lends_none() -> Result<()> {
        let state = StateMemory::meta(config(0), &cache())?;
        assert!(state.trail()?.is_none());

        // SAFETY: meta tensors have no data and their addresses are never read.
        let column = |rows: usize| unsafe { lend(0, rows, DType::U32, None) };
        let lent = state.lend(column(3)?, column(2)?, column(2)?, column(2)?, false)?;
        assert!(lent.trail.is_none());
        assert_eq!((lent.live.len(), lent.checkpoint.len()), (2, 2));

        let refused = state
            .lend(column(3)?, column(2)?, column(2)?, column(2)?, true)
            .err()
            .expect("a verification without a trail");
        assert!(
            refused.to_string().contains("does not speculate"),
            "{refused}"
        );
        assert!(state.settle(1, 0).is_err());
        Ok(())
    }

    /// A request slot owns the row before its number, and slot zero and a slot past the last
    /// own none.
    #[test]
    fn a_slot_owns_one_row() -> Result<()> {
        let state = StateMemory::meta(config(4), &cache())?;

        assert_eq!((state.row(1)?, state.row(3)?), (0, 2));
        assert!(state.row(0).is_err());
        assert!(state.row(4).is_err());

        // A settle names a slot of the state and a token of the trail.
        state.settle(3, 3)?;
        assert!(state.settle(3, 4).is_err());
        assert!(state.settle(4, 0).is_err());
        Ok(())
    }

    /// A copy is checked against the state's slots and the cache's state pages before it is
    /// issued.
    #[test]
    fn a_state_copy_names_a_slot_and_a_whole_state_page() -> Result<()> {
        let config_of_cache = cache();
        let cache = KvCacheMemory::meta(config_of_cache.clone())?;
        let mut state = StateMemory::meta(config(0), &config_of_cache)?;
        let mut restore =
            |copies: &[StateCopy]| state.copy(&cache, copies, StateDirection::Restore);
        let one = |slot: u32, units: &[u32]| [copy(slot, units, StateDirection::Restore)];

        restore(&one(2, &[5, 6]))?;
        assert!(restore(&one(4, &[5, 6])).is_err());

        // A state page spans two units, none of them the padding unit or outside the pool.
        let short = restore(&one(1, &[5])).expect_err("a page of one unit");
        assert!(short.to_string().contains("span 2"), "{short}");
        assert!(restore(&one(1, &[0, 6])).is_err());
        assert!(restore(&one(1, &[5, 16])).is_err());

        // A copy of the other direction is not this call's.
        restore(&[copy(9, &[5], StateDirection::Keep)])?;

        // A batch of two rows carries at most a restore and a keep per row.
        let many: Vec<StateCopy> = (0..5)
            .map(|_| copy(1, &[5, 6], StateDirection::Restore))
            .collect();
        let refused = restore(&many).expect_err("five copies of a batch of two rows");
        assert!(refused.to_string().contains("allow 4"), "{refused}");
        Ok(())
    }

    /// The copies state their kernels with no device: a snapshot and a scatter over each
    /// buffer field and over the state planes.
    #[test]
    fn the_copies_state_their_kernels_under_a_meta_context() -> Result<()> {
        let state = StateMemory::meta(config(0), &cache())?;

        let keys = state.keys()?;
        let of = |operator: &str| keys.iter().filter(|(name, _)| *name == operator).count();

        // Each operator over three shapes: the two fields' buffers and the state planes.
        assert_eq!(of(CacheBlockSnapshot::NAME), 3, "{keys:?}");
        assert_eq!(of(CacheBlockScatter::NAME), 3, "{keys:?}");
        assert_eq!(keys.len(), 6);
        Ok(())
    }

    /// A state needs the cache's state group to checkpoint into.
    #[test]
    fn a_state_needs_a_state_group() {
        let mut dense = cache();
        dense.layout = Arc::new(CacheLayout::dense(4, 2, 1, 8, PrefixCacheDtype::BFloat16));
        assert!(StateMemory::meta(config(0), &dense).is_err());

        let mut empty = config(0);
        empty.slots = 0;
        assert!(StateMemory::meta(empty, &cache()).is_err());
    }
}
