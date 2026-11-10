//! The recurrent state a causal language model keeps per request slot: its banks, the views a
//! forward reads them through, and the copies between the banks and the cache's state pages.
//!
//! A model whose layers keep a recurrent state ([`nn::family::TextModel::state`]) states the fields
//! one layer keeps; the runner owns two banks of every field of every layer, `[slots, …]` each:
//!
//! - the **live** banks hold each slot's state after its request's last computed token, which every
//!   row of a batch reads and advances in place;
//! - the **checkpoint** banks receive the state a row keeps at the page boundary the batch's
//!   `checkpoints` column names;
//! - in a deployment that speculates, the **trail** banks, `[tokens, …]` each, receive the state
//!   after every token of a verification, whose accepted length is known only once its tokens are
//!   sampled: the runner then settles each row's slot on the trail's entry after its last accepted
//!   token ([`StateBanks::settle`]).
//!
//! The banks are carved once from the device's resident pool, so every captured graph addresses
//! them unchanged. Between batches the state moves between the banks and the prefix cache's state
//! pages through `cache_block_snapshot` and `cache_block_scatter` ([`StateBanks::copy`]): a
//! checkpoint is gathered from the checkpoint banks into the snapshot scratch and scattered into
//! the page's planes after the forward, and a restore gathers a page's planes into the snapshot and
//! scatters them into the live banks before the forward. Which rows copy what is the mechanism's
//! decision, stated in the batch ([`crate::batch::StateCopy`]).
//!
//! A checkpoint lays each layer's fields out in field order from plane boundaries
//! ([`crate::memory::CachePlan::state_planes`]); plane `p` of a state page is column `q % columns`
//! of the page's unit `q / columns`, `q` its position in its field run.

use std::sync::{Arc, Mutex};

use nn::family::{StateField, StateShape};
use nn::{Forward, StateLayer, StateView};
use ops::block_copy::{
    BlockShape, COPY_BLOCK, CacheBlockScatter, CacheBlockSnapshot, ScatterArgs, SnapshotArgs,
};
use ops::{DType, DevicePtr, Shape, Strides, TensorView};
use platform::{Buffer, DeviceId, PoolKind};

use crate::arena::Pools;
use crate::memory::{CachePlan, StatePlanes};
use crate::runners::columns::{Column, encode_i32};
use crate::{Error, Result};
use ipc::execution::{StateCopy, StateDirection};

/// How many batches' index vectors may be in flight at once: a replay overlaps at most the one
/// before it, so a ring of four never reuses a set the device may still read.
const RING: usize = 4;

/// The state page geometry a copy addresses, read from the pool when the banks are bound.
#[derive(Clone)]
struct Pages {
    /// Each field run's planes, `[columns · units, plane_bytes]` bytes.
    runs: Vec<TensorView>,
    /// The pool's columns.
    columns: u32,
    /// The pool's units, the sentinel included.
    units: u32,
    /// Units one state page spans.
    units_per_page: u32,
}

/// The device bytes the banks of `shape` take for `slots` request slots and a trail of `trail`
/// tokens: both banks of every field of every layer, a row per slot and one for the padding slot,
/// and the trail's row per token. The capacity plan charges them beside the cache pool.
pub fn bank_bytes(shape: &StateShape, slots: u32, trail: u32) -> u64 {
    (2 * (u64::from(slots) + 1) + u64::from(trail)) * u64::from(shape.layers) * shape.layer_bytes()
}

/// The rows a batch's recurrent layers advance (`nn::StateView`'s rows): device columns of the
/// runner's, one entry per state row.
#[derive(Clone, Copy)]
pub(crate) struct StateRows {
    /// `[rows + 1]` i32 packed token offsets.
    pub(crate) query_offsets: TensorView,
    /// `[rows]` i32 tokens each row has seen.
    pub(crate) seq_lens: TensorView,
    /// `[rows]` i32 each row's slot.
    pub(crate) slots: TensorView,
    /// The rows.
    pub(crate) count: u64,
}

/// A slot's recurrent-state banks and the copies that move them.
pub(crate) struct StateBanks {
    shape: StateShape,
    /// Rows of every bank: the deployment's request slots and the padding slot zero.
    rows: u64,
    /// `[layer][field]` the live banks.
    live: Vec<Vec<Buffer>>,
    /// `[layer][field]` the checkpoint banks.
    kept: Vec<Vec<Buffer>>,
    /// `[layer][field]` the trail banks, `[trail_rows, …]`; empty in a deployment that does not
    /// speculate.
    trail: Vec<Vec<Buffer>>,
    /// Rows of every trail bank: the most tokens a verification batch holds.
    trail_rows: u64,
    /// `[layer][field]` where each field lies in a state page.
    planes: Vec<Vec<StatePlanes>>,
    plane_bytes: u64,
    /// The planes of one state page's checkpoints.
    page_planes: u32,
    /// The scratch one copy moves a page through, `[page_planes, plane_bytes]` bytes.
    snapshot: Buffer,
    /// `[rows]` i32 each row's checkpoint, which the forward reads.
    pub(crate) checkpoints: Column,
    /// The copies' index vectors, one set per batch in flight.
    indices: Vec<Column>,
    /// The event behind the last batch whose copies read each index set.
    fences: Vec<Option<platform::Event>>,
    turn: usize,
    pages: Pages,
    /// The copies one batch may carry: a restore and a checkpoint per row.
    copies: usize,
    encoded: Vec<u8>,
}

impl StateBanks {
    /// The banks of `shape` for `slots` request slots, carved from `pools` on `device`, over the
    /// cache pool `cache` whose plan is `plan`, for batches of at most `rows` rows, with a trail of
    /// `trail` tokens (zero for a deployment that does not speculate).
    ///
    /// # Errors
    ///
    /// A plan that holds no state group, or a pool that cannot hold the banks.
    #[allow(
        clippy::too_many_arguments,
        reason = "the state's shape, its three extents, the cache it checkpoints into and the \
                  device's pools"
    )]
    pub(crate) fn new(
        shape: StateShape,
        slots: u32,
        rows: u32,
        trail: u32,
        plan: &CachePlan,
        cache: &Arc<Mutex<crate::memory::PrefixCache>>,
        device: DeviceId,
        pools: &mut Pools,
    ) -> Result<Self> {
        let group = plan
            .groups
            .iter()
            .find(|group| group.state())
            .ok_or_else(|| {
                Error::Invalid("a model that keeps a recurrent state needs a state group".into())
            })?;
        let banks = u64::from(slots) + 1;
        let mut carve = |field: &StateField| -> Result<Buffer> {
            pools.allocate(PoolKind::Resident, banks * field.bytes())
        };
        let mut live = Vec::with_capacity(shape.layers as usize);
        let mut kept = Vec::with_capacity(shape.layers as usize);
        for _ in 0..shape.layers {
            live.push(shape.fields.iter().map(&mut carve).collect::<Result<_>>()?);
            kept.push(shape.fields.iter().map(&mut carve).collect::<Result<_>>()?);
        }
        let trail_rows = u64::from(trail);
        let mut trails = Vec::new();
        if trail_rows > 0 {
            for _ in 0..shape.layers {
                trails.push(
                    shape
                        .fields
                        .iter()
                        .map(|field| pools.allocate(PoolKind::Resident, trail_rows * field.bytes()))
                        .collect::<Result<Vec<_>>>()?,
                );
            }
        }
        let planes = (0..shape.layers)
            .map(|layer| plan.state_planes(layer, &shape.fields))
            .collect();
        let page_planes = plan.checkpoint_planes();
        let snapshot =
            pools.allocate(PoolKind::Scratch, u64::from(page_planes) * plan.plane_bytes)?;
        let copies = 2 * rows as usize;
        let indices = (0..RING)
            .map(|_| {
                Column::new(
                    device,
                    DType::I64,
                    copies as u64 * (u64::from(page_planes) + 1),
                )
            })
            .collect::<Result<_>>()?;
        let pages = {
            let cache = cache
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            Pages {
                runs: cache.state_views(),
                columns: plan.columns,
                units: cache.num_units(),
                units_per_page: group.units_per_page,
            }
        };
        if pages.runs.is_empty() {
            return Err(Error::Invalid(
                "the cache pool holds no state plane for the recurrent state's checkpoints".into(),
            ));
        }
        Ok(Self {
            shape,
            rows: banks,
            live,
            kept,
            trail: trails,
            trail_rows,
            planes,
            plane_bytes: plan.plane_bytes,
            page_planes,
            snapshot,
            checkpoints: Column::new(device, DType::I32, u64::from(rows))?,
            indices,
            fences: (0..RING).map(|_| None).collect(),
            turn: 0,
            pages,
            copies,
            encoded: Vec::new(),
        })
    }

    /// A `[rows, dims…]` view of `buffer` in `field`'s type.
    fn rows_of(buffer: &Buffer, field: &StateField, rows: u64) -> TensorView {
        let mut dims = vec![rows];
        dims.extend(&field.dims);
        TensorView::new(
            DevicePtr::new(buffer.address()),
            field.dtype,
            Shape::new(&dims),
        )
    }

    /// A bank's bytes, one row per slot.
    fn bank_bytes(&self, buffer: &Buffer, field: &StateField) -> TensorView {
        TensorView::new(
            DevicePtr::new(buffer.address()),
            DType::U8,
            Shape::new(&[self.rows, field.bytes()]),
        )
    }

    /// Runs `operation` with the state view of a batch whose state rows are `rows`, every layer's
    /// banks bound, the staged checkpoints column read for its rows, and the trail when `trailing`.
    ///
    /// # Errors
    ///
    /// A trailing batch in a deployment whose banks keep no trail, or what `operation` answers.
    pub(crate) fn with_view<R>(
        &self,
        rows: StateRows,
        trailing: bool,
        operation: impl FnOnce(StateView<'_>) -> Result<R>,
    ) -> Result<R> {
        if trailing && self.trail.is_empty() {
            return Err(Error::Invariant(
                "a verification keeps a trail and the state banks hold none: the deployment does \
                 not speculate"
                    .into(),
            ));
        }
        let bound = |banks: &[Buffer], rows: u64| -> Vec<TensorView> {
            banks
                .iter()
                .zip(&self.shape.fields)
                .map(|(buffer, field)| Self::rows_of(buffer, field, rows))
                .collect()
        };
        let views: Vec<(Vec<TensorView>, Vec<TensorView>, Vec<TensorView>)> = (0..self.live.len())
            .map(|layer| {
                (
                    bound(&self.live[layer], self.rows),
                    bound(&self.kept[layer], self.rows),
                    self.trail
                        .get(layer)
                        .map_or_else(Vec::new, |trail| bound(trail, self.trail_rows)),
                )
            })
            .collect();
        let layers: Vec<StateLayer<'_>> = views
            .iter()
            .map(|(live, checkpoint, trail)| StateLayer {
                live,
                checkpoint,
                trail: trailing.then_some(trail.as_slice()),
            })
            .collect();
        operation(StateView {
            query_offsets: rows.query_offsets,
            seq_lens: rows.seq_lens,
            slots: rows.slots,
            checkpoints: self.checkpoints.extent(0..rows.count),
            layers: &layers,
        })
    }

    /// Settles slot `slot` on the trail's entry of token `token`: every layer's live state becomes
    /// the state the trailing batch's forward left after that token, the last its slot's request
    /// accepted. Issued on `stream` behind the forward that wrote the trail; a recording forward
    /// moves nothing.
    ///
    /// # Errors
    ///
    /// A deployment whose banks keep no trail, a token past the trail or a slot past the banks, or
    /// a refused copy.
    pub(crate) fn settle(
        &self,
        f: &Forward<'_>,
        slot: u32,
        token: u32,
        stream: &platform::Stream,
    ) -> Result<()> {
        if self.trail.is_empty() || u64::from(token) >= self.trail_rows {
            return Err(Error::Invariant(format!(
                "a verification settles slot {slot} on token {token} of a trail of {} rows",
                self.trail_rows
            )));
        }
        if u64::from(slot) >= self.rows {
            return Err(Error::Invariant(format!(
                "a verification settles slot {slot} of banks of {} rows",
                self.rows
            )));
        }
        if f.is_recording() {
            return Ok(());
        }
        for (live, trail) in self.live.iter().zip(&self.trail) {
            for ((bank, entry), field) in live.iter().zip(trail).zip(&self.shape.fields) {
                let bytes = field.bytes();
                // SAFETY: both ranges are one field's row inside their own banks, which the checks
                // above bound, and the copy is ordered on the stream behind the forward that wrote
                // the trail and ahead of the next forward that reads the slot.
                unsafe {
                    platform::memcpy_async(
                        bank.address() + u64::from(slot) * bytes,
                        entry.address() + u64::from(token) * bytes,
                        bytes as usize,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// Stages a batch's checkpoints column, `rows` rows of them.
    pub(crate) fn stage(&mut self, checkpoints: &[i32]) -> Result<()> {
        encode_i32(&mut self.encoded, checkpoints);
        self.checkpoints.write(&self.encoded)
    }

    /// The pool rows a state page's planes lie in, in plane order: the field run each plane
    /// belongs to and its row there.
    fn cells(&self, units: &[ipc::UnitId]) -> Result<Vec<(usize, i64)>> {
        let pages = &self.pages;
        if units.len() != pages.units_per_page as usize {
            return Err(Error::Invalid(format!(
                "a state page of {} units, the pool's state pages span {}",
                units.len(),
                pages.units_per_page
            )));
        }
        let run = u64::from(pages.units_per_page) * u64::from(pages.columns);
        (0..u64::from(self.page_planes))
            .map(|plane| {
                let (field, position) = (plane / run, plane % run);
                let (unit, column) = (
                    position / u64::from(pages.columns),
                    position % u64::from(pages.columns),
                );
                let unit = units[unit as usize].0;
                if unit == 0 || unit >= pages.units {
                    return Err(Error::Invalid(format!(
                        "a state page names unit {unit} of a pool of {} units",
                        pages.units
                    )));
                }
                Ok((
                    field as usize,
                    (column * u64::from(pages.units) + u64::from(unit)) as i64,
                ))
            })
            .collect()
    }

    /// The descriptor of a copy over `field`, a `[blocks, width]` byte view.
    fn shape_of(field: &TensorView) -> BlockShape {
        BlockShape {
            dtype: DType::U8,
            shape: field.shape,
            strides: Strides::row_major(field.shape),
            index_stride: 1,
            width: field.shape.last(),
            block: COPY_BLOCK,
        }
    }

    /// Launches the batch's copies on `f`, in the order `direction` says: every restore before the
    /// forward, every checkpoint after it. A recording forward checks and reports them and moves
    /// nothing.
    ///
    /// # Errors
    ///
    /// More copies than the batch's rows allow, a page of the wrong size, or a launch's refusal.
    pub(crate) fn copy(
        &mut self,
        f: &Forward<'_>,
        copies: &[StateCopy],
        direction: StateDirection,
        stream: &platform::Stream,
    ) -> Result<()> {
        let copies: Vec<&StateCopy> = copies
            .iter()
            .filter(|copy| copy.direction == direction)
            .collect();
        if copies.is_empty() {
            return Ok(());
        }
        if copies.len() > self.copies {
            return Err(Error::Invalid(format!(
                "a batch makes {} state copies, its rows allow {}",
                copies.len(),
                self.copies
            )));
        }
        // The set this batch writes may still be read by a batch four replays back.
        if let Some(fence) = self.fences[self.turn].take() {
            fence.synchronize()?;
        }
        let stride = u64::from(self.page_planes) + 1;
        let mut values = Vec::with_capacity(copies.len() * stride as usize);
        let mut plans = Vec::with_capacity(copies.len());
        for copy in &copies {
            let cells = self.cells(&copy.units)?;
            values.push(i64::from(copy.slot));
            values.extend(cells.iter().map(|(_, cell)| *cell));
            plans.push(cells);
        }
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let indices = &mut self.indices[self.turn];
        indices.write(&bytes)?;
        let base = indices.device;

        for (index, cells) in plans.iter().enumerate() {
            let at = index as u64 * stride;
            let slot = TensorView::new(base.offset_elements(at), DType::I64, Shape::new(&[1]));
            match direction {
                StateDirection::Keep => {
                    self.banks_to_snapshot(f, slot)?;
                    self.snapshot_to_pages(f, base, at + 1, cells, true)?;
                }
                StateDirection::Restore => {
                    self.snapshot_to_pages(f, base, at + 1, cells, false)?;
                    self.snapshot_to_banks(f, slot)?;
                }
            }
        }
        if !f.is_recording() {
            let fence = platform::Event::new(platform::EventFlags::default());
            fence.record(stream)?;
            self.fences[self.turn] = Some(fence);
            self.turn = (self.turn + 1) % RING;
        }
        Ok(())
    }

    /// The snapshot's view of one field's planes: `[1, bytes]` from its first plane.
    fn snapshot_field(&self, planes: StatePlanes) -> TensorView {
        TensorView::new(
            DevicePtr::new(
                self.snapshot.address() + u64::from(planes.first_plane) * self.plane_bytes,
            ),
            DType::U8,
            Shape::new(&[1, planes.bytes]),
        )
    }

    /// Gathers slot `slot`'s checkpoint banks into the snapshot, field by field.
    fn banks_to_snapshot(&self, f: &Forward<'_>, slot: TensorView) -> Result<()> {
        for (layer, kept) in self.kept.iter().enumerate() {
            for ((buffer, field), planes) in
                kept.iter().zip(&self.shape.fields).zip(&self.planes[layer])
            {
                let bank = self.bank_bytes(buffer, field);
                f.operate::<CacheBlockSnapshot>(
                    &Self::shape_of(&bank),
                    SnapshotArgs {
                        field: bank,
                        blocks: slot,
                        snapshot: self.snapshot_field(*planes),
                    },
                )?;
            }
        }
        Ok(())
    }

    /// Scatters the snapshot into slot `slot`'s live banks, field by field.
    fn snapshot_to_banks(&self, f: &Forward<'_>, slot: TensorView) -> Result<()> {
        for (layer, live) in self.live.iter().enumerate() {
            for ((buffer, field), planes) in
                live.iter().zip(&self.shape.fields).zip(&self.planes[layer])
            {
                let bank = self.bank_bytes(buffer, field);
                f.operate::<CacheBlockScatter>(
                    &Self::shape_of(&bank),
                    ScatterArgs {
                        field: bank,
                        blocks: slot,
                        snapshot: self.snapshot_field(*planes),
                    },
                )?;
            }
        }
        Ok(())
    }

    /// Moves the snapshot's planes into the page's (`into`) or the page's into the snapshot, one
    /// launch per field run, through the index vector at element `first` of `base`.
    fn snapshot_to_pages(
        &self,
        f: &Forward<'_>,
        base: TensorView,
        first: u64,
        cells: &[(usize, i64)],
        into: bool,
    ) -> Result<()> {
        let mut start = 0;
        while start < cells.len() {
            let run = cells[start].0;
            let end = cells[start..]
                .iter()
                .position(|(field, _)| *field != run)
                .map_or(cells.len(), |offset| start + offset);
            let count = (end - start) as u64;
            let field =
                *self.pages.runs.get(run).ok_or_else(|| {
                    Error::Invariant(format!("a state page reaches field run {run}"))
                })?;
            let blocks = TensorView::new(
                base.offset_elements(first + start as u64),
                DType::I64,
                Shape::new(&[count]),
            );
            let snapshot = TensorView::new(
                DevicePtr::new(self.snapshot.address() + start as u64 * self.plane_bytes),
                DType::U8,
                Shape::new(&[count, self.plane_bytes]),
            );
            if into {
                f.operate::<CacheBlockScatter>(
                    &Self::shape_of(&field),
                    ScatterArgs {
                        field,
                        blocks,
                        snapshot,
                    },
                )?;
            } else {
                f.operate::<CacheBlockSnapshot>(
                    &Self::shape_of(&field),
                    SnapshotArgs {
                        field,
                        blocks,
                        snapshot,
                    },
                )?;
            }
            start = end;
        }
        Ok(())
    }

    /// One copy of each direction over a page of the pool's first units, for a recording forward:
    /// the operators and descriptors a served batch's copies launch.
    pub(crate) fn recorded_copies(&self) -> Vec<StateCopy> {
        let units: Vec<ipc::UnitId> = (1..=self.pages.units_per_page).map(ipc::UnitId).collect();
        [StateDirection::Restore, StateDirection::Keep]
            .into_iter()
            .map(|direction| StateCopy {
                slot: 1,
                units: units.clone(),
                direction,
            })
            .collect()
    }
}
