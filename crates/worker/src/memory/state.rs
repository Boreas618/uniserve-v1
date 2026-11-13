//! The recurrent state a causal language model keeps per request slot: its buffers, and the
//! copies between them and the KV cache's state pages.
//!
//! A model whose layers keep a recurrent state states the fields each layer keeps, as runs of
//! layers that keep the same ones (`nn::family::TextModel::state_layers`). [`StateMemory`]
//! owns, for every field of every such layer, in the runs' order:
//!
//! - the **live** buffer: each slot's state after its request's last computed token, which
//!   every row of a batch reads and advances in place;
//! - the **checkpoint** buffer, `[slots, …]`: the state a row keeps at the page boundary the
//!   batch's `checkpoints` column names;
//! - in a deployment that speculates with a trail, the **trail** buffer, `[tokens, …]`: the
//!   state after every token of a verification, whose accepted length is known only once its
//!   tokens are sampled. The runner then settles each row's slot on the trail's entry after
//!   its last accepted token ([`StateMemory::settle`]).
//!
//! Request slot `s` (slots count from one) owns row `s - 1` of every buffer, its **home row**
//! ([`StateMemory::row`]); a padding row of a batch states `nn::NO_SLOT` in the batch's state
//! slot column and reads and writes no row.
//!
//! ## The drafter's state
//!
//! A deployment that speculates with a drafter whose layers keep a state (`nn::Drafter::state`)
//! keeps it here too ([`StateConfig::drafter`], [`StateMemory::drafter`]): one **live** buffer
//! per field of every drafter layer, a home row per request slot, which the drafter's forwards
//! read and advance in place. It has no checkpoint, no trail and no verify slots, and no copy
//! moves it: a drafter states only fields that a forward from a page boundary does not read
//! from before it, so a request that resumes from a cached prefix needs none of its rows.
//!
//! ## Verify slots
//!
//! A field that keeps the state after each token of a decode-phase row in a slot of its own
//! (`nn::family::StateField::stepwise`) lets a deployment that speculates verify without a
//! trail: a request owns `Q` slots of it ([`StateConfig::owned`], the tokens a verification
//! holds), and a verification's acceptance chooses the slot that stands (`nn::VerifySlots`).
//! The live buffer of such a field holds `slots · Q` rows: rows `0..slots` are the home rows,
//! and request slot `s` owns rows `slots + (s − 1)(Q − 1) .. slots + s(Q − 1)` beside its
//! home row. Every other field's live buffer, and every checkpoint buffer, holds `slots` rows,
//! and such a state keeps no trail.
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
//! slot's live home row before it. Which rows copy what is the scheduler's decision, stated in
//! the batch by the checkpoint's position in the request's state table
//! (`ipc::batch::StateCopy`); the runner finds the page's run of units through the rank's
//! device block table, and a keep writes only a block the table states writable.
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
use std::ops::Range;
use std::sync::Arc;

use ipc::batch::StateDirection;
use nn::family::{StateField, StateShape};
use nn::{RecurrentState, VerifySlots};
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
    /// The runs of recurrent layers that keep the same fields, in the model's order of them
    /// (`nn::family::TextModel::state_layers`): every layer of the first run, then of the
    /// second.
    pub shapes: Vec<StateShape>,
    /// Request slots: the home rows of every live buffer, and the rows of every checkpoint
    /// buffer.
    pub slots: u32,
    /// The most rows of a batch: a batch carries at most a restore and a keep per row.
    pub rows: u32,
    /// Rows of every trail buffer: the most tokens a verification batch holds. Zero for a
    /// deployment that keeps no trail: one that does not speculate, or whose stepwise fields
    /// keep verify slots.
    pub trail: u32,
    /// The slots a request owns of each stepwise field (module documentation, verify slots):
    /// the tokens a verification holds, `draft_tokens + 1`, for a deployment that speculates,
    /// and one for every other.
    pub owned: u32,
    /// The runs of the drafter's layers that keep a state (`nn::Drafter::state`), in its order
    /// of them: a live home row per request slot of each field, and nothing else (module
    /// documentation, the drafter's state). Empty for a deployment that does not speculate or
    /// whose drafter keeps none.
    pub drafter: Vec<StateShape>,
}

/// Each recurrent layer's fields, in the order the state's buffers follow: the layers of the
/// first of `runs`, then of the second (`nn::family::TextModel::state_layers`).
pub(crate) fn layer_fields(runs: &[StateShape]) -> impl Iterator<Item = &[StateField]> {
    runs.iter()
        .flat_map(|run| (0..run.layers).map(move |_| run.fields.as_slice()))
}

impl StateConfig {
    /// The device bytes the state takes from the pool that lives as long as the rank serves,
    /// for a cache of `layout`: the live and checkpoint buffers of every field of every layer,
    /// each live buffer's rows ([`Self::live_rows`]), a checkpoint row per slot and the
    /// trail's row per token, the drafter's live row per slot of each of its fields, and the
    /// copies' index vectors. The snapshot is the scratch pool's
    /// ([`CacheLayout::state_snapshot_bytes`]).
    pub fn bytes(&self, layout: &CacheLayout) -> u64 {
        let (slots, trail) = (u64::from(self.slots), u64::from(self.trail));
        let buffers: u64 = layer_fields(&self.shapes)
            .flatten()
            .map(|field| (u64::from(self.live_rows(field)) + slots + trail) * field.bytes())
            .sum();
        let drafter: u64 = layer_fields(&self.drafter)
            .flatten()
            .map(|field| slots * field.bytes())
            .sum();

        buffers + drafter + RING as u64 * self.index_bytes(layout)
    }

    /// The rows of `field`'s live buffer: a home row per request slot, and the further slots
    /// each request owns of a stepwise field (module documentation, verify slots).
    pub fn live_rows(&self, field: &StateField) -> u32 {
        if field.stepwise {
            self.slots * self.owned
        } else {
            self.slots
        }
    }

    /// Whether a request owns several slots of a field: the deployment's forwards then lend
    /// `nn::VerifySlots`.
    pub fn verifies(&self) -> bool {
        self.owned > 1
            && layer_fields(&self.shapes)
                .flatten()
                .any(|field| field.stepwise)
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
    /// `[drafter layer][field]` the drafter's live buffers; empty for a drafter that keeps no
    /// state.
    drafter: Vec<Vec<Buffer>>,
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
    /// A config of no slot, layer or field, a drafter's field that keeps a slot per verified
    /// token, or a cache whose layout holds no state group for the checkpoints.
    pub fn meta(config: StateConfig, cache: &KvCacheConfig) -> Result<Self> {
        let layers: Vec<&[StateField]> = layer_fields(&config.shapes).collect();
        let fieldless = layer_fields(&config.shapes)
            .chain(layer_fields(&config.drafter))
            .any(|fields| fields.is_empty());
        if config.slots == 0 || config.owned == 0 || fieldless {
            return Err(Error::Invalid(format!(
                "a recurrent state of {} slots, of which a request owns {} of a stepwise field, \
                 whose every layer keeps a field",
                config.slots, config.owned
            )));
        }

        // The drafter's forwards are lent its live rows alone (module documentation).
        if let Some(field) = layer_fields(&config.drafter)
            .flatten()
            .find(|field| field.stepwise)
        {
            return Err(Error::Invalid(format!(
                "the drafter's state field {} keeps a slot per verified token, and a drafter's \
                 state keeps one slot per request",
                field.name
            )));
        }
        if layers.is_empty() {
            return Err(Error::Invalid(
                "a recurrent state of no recurrent layer holds nothing".into(),
            ));
        }
        if config.verifies() && config.trail > 0 {
            return Err(Error::Invalid(
                "a state whose requests own a slot of a stepwise field per verified token keeps \
                 no trail"
                    .into(),
            ));
        }

        let layout: &CacheLayout = &cache.layout;
        let group = layout
            .state_group()
            .map(|group| &layout.groups[group])
            .ok_or_else(|| {
                Error::Invalid("a model that keeps a recurrent state needs a state group".into())
            })?;
        if layout.state_fields() == 0 || layout.checkpoints.len() != layers.len() {
            return Err(Error::Invalid(format!(
                "the cache's state pages hold the checkpoints of {} recurrent layers, and the \
                 state keeps {}",
                layout.checkpoints.len(),
                layers.len()
            )));
        }

        let planes: Vec<Vec<(u32, u64)>> = layers
            .iter()
            .enumerate()
            .map(|(layer, fields)| layout.state_planes(layer as u32, fields))
            .collect();

        // Every field's planes lie inside its layer's checkpoint, which the layout laid out
        // from the same runs.
        for (fields, checkpoint) in planes.iter().zip(&layout.checkpoints) {
            let end = u64::from(checkpoint.first_plane) + u64::from(checkpoint.planes);

            for (first, bytes) in fields {
                if u64::from(*first) + bytes.div_ceil(layout.plane_bytes) > end {
                    return Err(Error::Invalid(format!(
                        "a checkpoint field of {bytes} bytes from plane {first} lies outside its \
                         layer's checkpoint, which ends at plane {end}"
                    )));
                }
            }
        }

        Ok(Self {
            config,
            planes,
            page_planes: layout.checkpoint_planes(),
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
        let config = state.config.clone();

        // One buffer per field of every layer of `runs`, of the rows `rows` states for the
        // field.
        let mut take =
            |runs: &[StateShape], rows: &dyn Fn(&StateField) -> u32| -> Result<Vec<Vec<Buffer>>> {
                layer_fields(runs)
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|field| {
                                let bytes = u64::from(rows(field)) * field.bytes();
                                Ok(allocator.allocate(PoolKind::Resident, bytes as usize)?)
                            })
                            .collect()
                    })
                    .collect()
            };
        let live = take(&config.shapes, &|field| config.live_rows(field))?;
        let checkpoint = take(&config.shapes, &|_| config.slots)?;
        let trail = if config.trail > 0 {
            take(&config.shapes, &|_| config.trail)?
        } else {
            Vec::new()
        };
        let drafter = take(&config.drafter, &|_| config.slots)?;

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
            drafter,
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

    /// The home row request slot `slot` owns of the live and checkpoint buffers: what the
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

    /// The tensors of one kind of buffer of the layers of `runs`, `[layer][field]`: each
    /// `[rows, …]` in its field's element type and extents, or `[rows, bytes]` U8 when `raw`,
    /// as a copy moves it, `rows` being what `rows` states for the field.
    fn tensors(
        &self,
        runs: &[StateShape],
        rows: impl Fn(&StateField) -> u32,
        raw: bool,
        buffers: impl Fn(&Buffers) -> &Vec<Vec<Buffer>>,
    ) -> Result<Vec<Vec<Tensor>>> {
        let shape = |field: &StateField| {
            let mut dims = vec![rows(field) as usize];
            if raw {
                dims.push(field.bytes() as usize);
            } else {
                dims.extend(&field.dims);
            }
            dims
        };

        layer_fields(runs)
            .enumerate()
            .map(|(layer, fields)| {
                fields
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

    /// The live buffers: one entry per recurrent layer of every run, in the runs' order, each
    /// the layer's fields in the order of its run's `StateShape`, `[rows, …]`: a home row per
    /// slot, and the further slots of a stepwise field ([`StateConfig::live_rows`]).
    pub fn live(&self) -> Result<Vec<Vec<Tensor>>> {
        self.tensors(
            &self.config.shapes,
            |field| self.config.live_rows(field),
            false,
            |buffers| &buffers.live,
        )
    }

    /// The checkpoint buffers, in the same order and extents as [`Self::live`] but a row per
    /// slot of every field: a checkpoint keeps the state a home row holds.
    pub fn checkpoint(&self) -> Result<Vec<Vec<Tensor>>> {
        self.tensors(
            &self.config.shapes,
            |_| self.config.slots,
            false,
            |buffers| &buffers.checkpoint,
        )
    }

    /// The trail buffers, in the same order, `[trail, …]`; `None` in a deployment that keeps
    /// no trail.
    pub fn trail(&self) -> Result<Option<Vec<Vec<Tensor>>>> {
        if self.config.trail == 0 {
            return Ok(None);
        }
        self.tensors(
            &self.config.shapes,
            |_| self.config.trail,
            false,
            |buffers| &buffers.trail,
        )
        .map(Some)
    }

    /// The drafter's live buffers (module documentation, the drafter's state): one entry per
    /// layer of the drafter's runs ([`StateConfig::drafter`]), each the layer's fields,
    /// `[slots, …]`, as `nn::RecurrentState::live` holds them for the drafter's forwards;
    /// `None` for a drafter that keeps no state.
    pub fn drafter(&self) -> Result<Option<Vec<Vec<Tensor>>>> {
        if layer_fields(&self.config.drafter).next().is_none() {
            return Ok(None);
        }
        self.tensors(
            &self.config.drafter,
            |_| self.config.slots,
            false,
            |buffers| &buffers.drafter,
        )
        .map(Some)
    }

    /// The state as a forward is lent it, for a batch whose state rows the four columns
    /// describe (`nn::RecurrentState`): `slots` holds [`Self::row`] of each row's request slot
    /// and `nn::NO_SLOT` for a padding row. A `trailing` batch, a verification, also receives
    /// the trail; a state whose requests own several slots of a stepwise field is lent
    /// `verify`, the rows' slots and accepted counts, by every batch instead.
    ///
    /// # Errors
    ///
    /// A trailing batch in a deployment whose state keeps no trail, and verify slots given to
    /// a state whose requests own one slot of every field, or withheld from one whose requests
    /// own several.
    pub fn lend(
        &self,
        query_offsets: Tensor,
        seq_lens: Tensor,
        slots: Tensor,
        checkpoints: Tensor,
        trailing: bool,
        verify: Option<VerifySlots>,
    ) -> Result<RecurrentState> {
        if verify.is_some() != self.config.verifies() {
            return Err(Error::Invariant(format!(
                "a batch is lent {} verify slots, and the state's requests own {} slots of a \
                 stepwise field",
                if verify.is_some() { "its" } else { "no" },
                self.config.owned
            )));
        }
        let trail = if trailing {
            Some(self.trail()?.ok_or_else(|| {
                Error::Invariant(
                    "a verification keeps a trail and the recurrent state holds none: the \
                     deployment does not speculate, or verifies over verify slots"
                        .into(),
                )
            })?)
        } else {
            None
        };

        Ok(RecurrentState {
            verify,
            query_offsets,
            seq_lens,
            slots,
            checkpoints,
            live: self.live()?,
            checkpoint: self.checkpoint()?,
            trail,
        })
    }

    /// The rows of the cache's state runs the planes of the state page in `units` lie in, in
    /// plane order: each plane's field run and its row there, `column · units + unit`.
    fn cells(&self, units: Range<u32>) -> Result<Vec<(usize, i64)>> {
        if units.len() != self.units_per_page as usize {
            return Err(Error::Invalid(format!(
                "a state page of {} units, and the cache's state pages span {}",
                units.len(),
                self.units_per_page
            )));
        }
        if units.start == 0 || units.end > self.units {
            return Err(Error::Invalid(format!(
                "a state page names units {}..{} of a pool of units 1..{}",
                units.start, units.end, self.units
            )));
        }

        let run = u64::from(self.units_per_page) * u64::from(self.columns);
        Ok((0..u64::from(self.page_planes))
            .map(|plane| {
                let position = plane % run;
                let column = position % u64::from(self.columns);
                let unit = units.start + (position / u64::from(self.columns)) as u32;

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
        let buffers = layer_fields(&self.config.shapes)
            .map(|fields| {
                fields
                    .iter()
                    .map(|field| {
                        meta(
                            &[self.config.slots as usize, field.bytes() as usize],
                            DType::U8,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        let page = self.cells(1..1 + self.units_per_page)?;

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
    /// forward, every keep after it. Each copy is a request slot and the units of its state
    /// page, the run the slot's checkpoint entry names. A restore moves a page's planes into
    /// its slot's live row; a keep moves its slot's checkpoint row into a page's planes. Every
    /// copy is checked before anything is issued, and a state that holds no device issues
    /// nothing.
    ///
    /// # Errors
    ///
    /// More copies than the batch's rows allow, a slot outside the state's slots, a page of
    /// another unit count than the cache's state pages span, a unit that is the padding unit
    /// or outside the pool, or a launch's refusal.
    pub fn copy(
        &mut self,
        cache: &KvCacheMemory,
        copies: &[(u32, Range<u32>)],
        direction: StateDirection,
    ) -> Result<()> {
        let mut pages = Vec::new();
        for (slot, units) in copies {
            pages.push((self.row(*slot)?, self.cells(units.clone())?));
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
        // A copy moves a slot's home row, which the first `slots` rows of every buffer hold.
        let (model, slots) = (&self.config.shapes, |_: &StateField| self.config.slots);
        let buffers = match direction {
            StateDirection::Restore => self.tensors(model, slots, true, |buffers| &buffers.live)?,
            StateDirection::Keep => {
                self.tensors(model, slots, true, |buffers| &buffers.checkpoint)?
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
        let layers = device
            .live
            .iter()
            .zip(&device.trail)
            .zip(layer_fields(&self.config.shapes));
        for ((live, trail), fields) in layers {
            for ((live, trail), field) in live.iter().zip(trail).zip(fields) {
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
    use ipc::PrefixCacheDtype;
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
                    stepwise: false,
                },
                StateField {
                    name: "recurrent",
                    dtype: DType::F32,
                    dims: vec![10],
                    stepwise: false,
                },
            ],
        }
    }

    /// The cache of two attention layers of 4-token pages of one 8-wide BF16 head beside that
    /// state: planes of 64 bytes, so a layer's checkpoint is two planes and a state page of
    /// four planes is two units of two columns.
    fn cache() -> KvCacheConfig {
        let layout = CacheLayout::hybrid(4, 2, 1, 8, PrefixCacheDtype::BFloat16, &[shape()])
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
            shapes: vec![shape()],
            slots: 3,
            rows: 2,
            trail,
            owned: 1,
            drafter: Vec::new(),
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
        let lent = state.lend(column(3)?, column(2)?, column(2)?, column(2)?, false, None)?;
        assert!(lent.trail.is_none());
        assert_eq!((lent.live.len(), lent.checkpoint.len()), (2, 2));

        let refused = state
            .lend(column(3)?, column(2)?, column(2)?, column(2)?, true, None)
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
            |copies: &[(u32, Range<u32>)]| state.copy(&cache, copies, StateDirection::Restore);

        restore(&[(2, 5..7)])?;
        assert!(restore(&[(4, 5..7)]).is_err());

        // A state page spans two units, none of them the padding unit or outside the pool.
        let short = restore(&[(1, 5..6)]).expect_err("a page of one unit");
        assert!(short.to_string().contains("span 2"), "{short}");
        assert!(restore(&[(1, 0..2)]).is_err());
        assert!(restore(&[(1, 15..17)]).is_err());

        // A batch of two rows carries at most a restore and a keep per row.
        let many: Vec<(u32, Range<u32>)> = (0..5).map(|_| (1, 5..7)).collect();
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

    /// The runs of a state whose layers keep different fields: the two layers of [`shape`],
    /// then one layer whose one field keeps a slot per verified token, 16 bytes.
    fn runs() -> Vec<StateShape> {
        vec![
            shape(),
            StateShape {
                layers: 1,
                fields: vec![StateField {
                    name: "steps",
                    dtype: DType::F32,
                    dims: vec![4],
                    stepwise: true,
                }],
            },
        ]
    }

    /// The cache of three attention layers beside [`runs`]: three columns of 64-byte planes,
    /// the two layers of the first run two planes each and the last layer one.
    fn runs_cache() -> Result<KvCacheConfig> {
        let layout = CacheLayout::hybrid(4, 3, 1, 8, PrefixCacheDtype::BFloat16, &runs())?;

        Ok(KvCacheConfig {
            layout: Arc::new(layout),
            ..cache()
        })
    }

    /// A state's buffers follow its runs in order, each layer with its own run's fields. A
    /// request owns `Q` slots of a stepwise field, whose live buffer holds them past the home
    /// rows, and one checkpoint row; such a state is lent verify slots by every batch and keeps
    /// no trail.
    #[test]
    fn a_state_of_runs_keeps_every_slot_its_requests_own() -> Result<()> {
        let cache = runs_cache()?;
        let config = StateConfig {
            shapes: runs(),
            owned: 3,
            ..config(0)
        };
        let state = StateMemory::meta(config.clone(), &cache)?;

        let live = state.live()?;
        let checkpoint = state.checkpoint()?;
        assert_eq!((live.len(), checkpoint.len()), (3, 3));
        for layer in &live[..2] {
            assert_eq!(layer[0].dims(), [3, 2, 3]);
            assert_eq!(layer[1].dims(), [3, 10]);
        }
        assert_eq!(live[2].len(), 1);
        assert_eq!(
            (live[2][0].dims(), live[2][0].dtype()),
            (&[9, 4][..], DType::F32)
        );
        assert_eq!(checkpoint[2][0].dims(), [3, 4]);

        // Each layer's checkpoint lies in its own planes of the state page: two planes for a
        // layer of the first run and one for the last.
        let first: Vec<u32> = cache
            .layout
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.first_plane)
            .collect();
        assert_eq!(first, [0, 2, 4]);
        assert_eq!(cache.layout.groups[1].units_per_page, 2);

        // Three slots of 52 bytes in each of two layers, live and checkpoint, nine live slots
        // and three checkpoints of 16 bytes, and four sets of index vectors of a slot row and
        // five planes for each of the four copies a batch of two rows may carry.
        assert_eq!(
            config.bytes(&cache.layout),
            2 * 2 * 3 * 52 + (9 + 3) * 16 + 4 * (4 * 6 * 8)
        );

        // Every batch is lent the requests' slots, and none keeps a trail.
        // SAFETY: meta tensors have no data and their addresses are never read.
        let column = |dims: &[usize]| unsafe { lend(0, dims, DType::U32, None) };
        let verify = || -> Result<VerifySlots> {
            Ok(VerifySlots {
                state_indices: column(&[2, 3])?,
                num_accepted: column(&[2])?,
            })
        };
        let columns = || -> Result<[Tensor; 4]> {
            Ok([column(&[3])?, column(&[2])?, column(&[2])?, column(&[2])?])
        };
        let [offsets, lens, slots, checkpoints] = columns()?;
        let lent = state.lend(offsets, lens, slots, checkpoints, false, Some(verify()?))?;
        assert!(lent.verify.is_some() && lent.trail.is_none());

        let [offsets, lens, slots, checkpoints] = columns()?;
        assert!(
            state
                .lend(offsets, lens, slots, checkpoints, false, None)
                .is_err()
        );
        let [offsets, lens, slots, checkpoints] = columns()?;
        assert!(
            state
                .lend(offsets, lens, slots, checkpoints, true, Some(verify()?))
                .is_err()
        );
        assert!(StateMemory::meta(StateConfig { trail: 4, ..config }, &cache).is_err());

        // A state whose requests own one slot is lent none.
        let single = StateMemory::meta(self::config(0), &self::cache());
        let [offsets, lens, slots, checkpoints] = columns()?;
        assert!(
            single?
                .lend(offsets, lens, slots, checkpoints, false, Some(verify()?))
                .is_err()
        );

        // A state of other runs than the ones its cache's state pages hold.
        assert!(StateMemory::meta(self::config(0), &cache).is_err());
        Ok(())
    }

    /// A drafter's state of one layer: a ring of four 8-wide BF16 keys and their three I32
    /// coordinates, 64 and 48 bytes a slot.
    fn ring() -> StateShape {
        StateShape {
            layers: 1,
            fields: vec![
                StateField {
                    name: "index_ring",
                    dtype: DType::BF16,
                    dims: vec![4, 8],
                    stepwise: false,
                },
                StateField {
                    name: "index_ring_positions",
                    dtype: DType::I32,
                    dims: vec![4, 3],
                    stepwise: false,
                },
            ],
        }
    }

    /// A drafter's state is a live home row per slot of each of its fields, beside the model's
    /// state and outside its checkpoints: the model's buffers and the copies' kernels are what
    /// they are without it. A drafter's field that keeps a slot per verified token is refused.
    #[test]
    fn a_drafters_state_is_one_live_row_per_slot() -> Result<()> {
        let drafting = StateConfig {
            drafter: vec![ring()],
            ..config(0)
        };
        let state = StateMemory::meta(drafting.clone(), &cache())?;
        let plain = StateMemory::meta(config(0), &cache())?;

        let drafter = state.drafter()?.expect("the drafter's state");
        assert_eq!(drafter.len(), 1);
        assert_eq!(
            (drafter[0][0].dims(), drafter[0][0].dtype()),
            (&[3, 4, 8][..], DType::BF16)
        );
        assert_eq!(
            (drafter[0][1].dims(), drafter[0][1].dtype()),
            (&[3, 4, 3][..], DType::I32)
        );
        assert!(plain.drafter()?.is_none());

        // Three slots of 112 bytes beside the model's state.
        let layout = cache().layout;
        assert_eq!(drafting.bytes(&layout), config(0).bytes(&layout) + 3 * 112);
        assert_eq!(state.live()?.len(), plain.live()?.len());
        assert_eq!(state.checkpoint()?.len(), plain.checkpoint()?.len());
        assert_eq!(state.keys()?, plain.keys()?);

        let mut stepwise = ring();
        stepwise.fields[0].stepwise = true;
        let refused = StateMemory::meta(
            StateConfig {
                drafter: vec![stepwise],
                ..config(0)
            },
            &cache(),
        )
        .err()
        .expect("a drafter's stepwise field");
        assert!(refused.to_string().contains("index_ring"), "{refused}");
        Ok(())
    }
}
