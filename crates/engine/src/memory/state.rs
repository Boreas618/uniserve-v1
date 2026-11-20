//! The recurrent state a causal language model keeps per request slot: its buffers, and the
//! copies between them and the units of the state region.
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
//! A deployment that speculates with a drafter whose layers keep a state (`nn::Drafter::state_layers`)
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
//! Between forwards the state moves between the buffers and the units of the state region, which
//! the cache holds (`KvCacheMemory::state_tensor`), through `cache_block_snapshot` and
//! `cache_block_scatter` ([`StateMemory::copy`]): a keep gathers a slot's checkpoint row into the
//! snapshot and scatters it into a unit's planes after the forward, and a restore gathers a
//! unit's planes into the snapshot and scatters them into the slot's live home row before it.
//! Which rows copy what is the scheduler's decision, stated in the batch by the checkpoint's
//! position in the request's state table (`crate::batch::StateCopy`); the runner finds the
//! checkpoint's block, one run of one unit of the state region, through the rank's device block
//! table, and a keep writes only a block the table states writable.
//!
//! A checkpoint lays each layer's fields out in field order from plane boundaries
//! ([`CacheLayout::state_planes`]); plane `p` of state unit `u` is row
//! `u · checkpoint_planes + p` of the units (`KvCacheMemory::state_rows`). It keeps of each field
//! the rows the field holds at zero drafts (`nn::family::StateField::kept_bytes`), the leading
//! part of the slot's row that a keep gathers and a restore scatters: what a forward from a block
//! boundary reads. The state region's units are thus the same whatever the deployment drafts.
//!
//! The two kernels name the rows they move by I64 index vectors on the device, the element type
//! they read them in. A batch's vectors are written from the host as the copies are issued,
//! into one of [`RING`] sets, so a set is written again only once the copies that read it
//! completed. The kernels launch through `ops::launch` under a context of the state's own,
//! over the stream of the runner that holds the state and the generation the rank serves from;
//! it allocates nothing, the snapshot and the index vectors being the state's memory.
//!
//! Like the cache, the state has a form that holds no device ([`StateMemory::meta`]): its
//! tensors are meta tensors and its copies are checked and issue nothing.
//!
//! ## Generations of other drafts
//!
//! The drafts and the buckets a kernel generation of other graph options captures for lay the
//! buffers out otherwise ([`StateConfig::of`]): a stepwise field's slots, a window's draft rows, a
//! ring's groups, the trail and the copies' rows. Such a generation lays its state out beside the
//! serving one from the same pool ([`StateMemory::prepare`]), over the same state units, and its
//! shapes are captured over that state's buffers ([`StateMemory::prepared`]). Its switch moves
//! every slot's state into them ([`StateMemory::switch`]), as [`Standing`] states what each slot's
//! last step left, and they serve from then on; the serving buffers are kept until the commit
//! gives them back ([`StateMemory::commit`]), and a rollback moves the slots' state back into them
//! ([`StateMemory::switch_back`]) before the generation's are given back
//! ([`StateMemory::release`]).
//!
//! A slot's state moves field by field ([`nn::family::StateRows`]): a whole field from the row
//! that stands, which for a stepwise field is the slot `num_accepted − 1` its verify slots name;
//! a window's history from the row `num_accepted − 1` of the slot's window into its row zero; a
//! ring's positions before the slot's verified length, each into its row of the other ring. The
//! state then stands in the slot's home row in every field, as a count of one accepted token
//! states, which the runner sets. The trail, the checkpoint buffers and the further verify slots
//! hold nothing between two steps, and the drafter's state moves as the model's does.

use std::collections::BTreeSet;
use std::sync::Arc;

use crate::batch::{StateDirection, Variant};
use nn::family::{StateField, StateRows, StateShape, TextModel};
use ops::block_copy::{
    BlockShape, CacheBlockScatter, CacheBlockSnapshot, ScatterArgs, SnapshotArgs,
};
use platform::{
    Allocator, Buffer, CopyBatch, DeviceId, DevicePtr, DeviceRange, Event, EventFlags, PoolKind,
    SourceOrder, Stream,
};
use tensor::{DType, Tensor};

use crate::memory::cache::{CacheLayout, KvCacheConfig, KvCacheMemory};
use crate::memory::lend;
use crate::runner::buckets::Buckets;
use crate::worker::{Error, Result};

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
    /// The runs of the drafter's layers that keep a state (`nn::Drafter::state_layers`), in its order
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

/// Whether a speculating deployment of a model whose state is `runs`, whose verifications hold
/// `tokens` tokens a row at most, keeps a trail of the state after each token of a
/// verification: a model with a state none of whose fields keeps a slot per verified token, and
/// one of whose fields a verification cannot advance in place.
///
/// A ring (`StateRows::Ring`) of `R` rows in groups of `g` positions advances in place when
/// `R ≥ tokens + g − 2` (`R ≥ tokens` for `g = 1`). A verification from position `V` writes the
/// rows of `V .. V + tokens`, each its own; after `a` accepted drafts the next forward starts at
/// `V + a + 1` and reads only the rows of the group open there, positions from `V + a + 2 − g`
/// on (`StateRows::Ring`). Those rows are the accepted tokens' own, or rows before `V` that no
/// row the verification wrote shares, so the rows of its rejected tokens are never read and are
/// overwritten when their positions are computed again: the live rows stand after any
/// acceptance.
pub(crate) fn keeps_trail(runs: &[StateShape], tokens: u32) -> bool {
    let mut fields = layer_fields(runs).flatten().peekable();
    if fields.peek().is_none() {
        return false;
    }

    let fields: Vec<&StateField> = fields.collect();
    let stepwise = fields.iter().any(|field| field.stepwise);
    let in_place = fields.iter().all(|field| match field.rows {
        StateRows::Ring { group } => {
            let rows = field.dims.first().copied().unwrap_or(0);
            rows >= tokens as usize + group.max(2) - 2
        }
        _ => false,
    });
    !stepwise && !in_place
}

impl StateConfig {
    /// The recurrent state of `model` for a runner of `slots` request slots over `buckets`, the
    /// catalog with its verifications (`Buckets::with_drafts`) when the deployment drafts `drafts`
    /// tokens a step, with its drafter's when it drafts with one that keeps a state; `None` for a
    /// model that keeps none. `model` is the one the drafts load: its fields are laid out for
    /// them.
    ///
    /// A request owns `drafts + 1` slots of a stepwise field when the deployment speculates, and
    /// its verifications are then lent verify slots and keep no trail; a speculating deployment
    /// whose every field is a ring keeps none either, its verifications advancing the rings in
    /// place ([`keeps_trail`]); any other keeps a trail of the tokens of its largest
    /// verification.
    pub fn of(model: &dyn TextModel, slots: u32, buckets: &Buckets, drafts: u32) -> Option<Self> {
        let shapes = model.state_layers();
        if shapes.is_empty() {
            return None;
        }

        let stepwise = layer_fields(&shapes).flatten().any(|field| field.stepwise);
        let owned = if stepwise && drafts > 0 {
            drafts + 1
        } else {
            1
        };
        let trail = if drafts > 0 && keeps_trail(&shapes, drafts + 1) {
            buckets
                .iter()
                .filter(|bucket| {
                    matches!(
                        bucket.variant,
                        Variant::VerifyGreedy | Variant::VerifySampled
                    )
                })
                .map(|bucket| bucket.tokens)
                .max()
                .unwrap_or(0)
        } else {
            0
        };

        // A drafter whose layers keep a state of their own keeps one live row a slot of each of
        // its fields beside the model's.
        let drafter = model
            .drafter()
            .filter(|_| drafts > 0)
            .map(|drafter| drafter.state_layers())
            .unwrap_or_default();

        Some(Self {
            shapes,
            slots,
            rows: buckets.bounds().0,
            trail,
            owned,
            drafter,
        })
    }

    /// The device bytes the state takes from the pool that lives as long as the rank serves,
    /// for a cache of `layout`: the live and checkpoint buffers of every field of every layer,
    /// each live buffer's rows ([`Self::live_rows`]), a checkpoint row per slot and the
    /// trail's row per token, the drafter's live row per slot of each of its fields, and the
    /// copies' index vectors. The snapshot is the scratch pool's ([`CacheLayout::state_bytes`]).
    ///
    /// # Errors
    ///
    /// A field whose element type has no whole-byte width (`StateField::bytes`).
    pub(crate) fn bytes(&self, layout: &CacheLayout) -> Result<u64> {
        let (slots, trail) = (u64::from(self.slots), u64::from(self.trail));
        let buffers = layer_fields(&self.shapes)
            .flatten()
            .map(|field| Ok((u64::from(self.live_rows(field)) + slots + trail) * field.bytes()?))
            .sum::<Result<u64>>()?;
        let drafter = layer_fields(&self.drafter)
            .flatten()
            .map(|field| Ok(slots * field.bytes()?))
            .sum::<Result<u64>>()?;

        Ok(buffers + drafter + RING as u64 * self.index_bytes(layout))
    }

    /// The rows of `field`'s live buffer: a home row per request slot, and the further slots
    /// each request owns of a stepwise field (module documentation, verify slots).
    pub(crate) fn live_rows(&self, field: &StateField) -> u32 {
        if field.stepwise {
            self.slots * self.owned
        } else {
            self.slots
        }
    }

    /// Whether a request owns several slots of a field: the deployment's forwards then lend
    /// `nn::VerifySlots`.
    pub(crate) fn verifies(&self) -> bool {
        self.owned > 1
            && layer_fields(&self.shapes)
                .flatten()
                .any(|field| field.stepwise)
    }

    /// The bytes of one set of index vectors: a slot row and a state unit's planes for each of
    /// the copies a batch may carry.
    fn index_bytes(&self, layout: &CacheLayout) -> u64 {
        2 * u64::from(self.rows) * (u64::from(layout.checkpoint_planes()) + 1) * 8
    }
}

/// The state's memory on a device: the buffers of one config of it, each taken from the
/// device's resident pool.
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
    /// The address of what one copy moves a state unit through, `[checkpoint planes,
    /// plane_bytes]` bytes of the scratch pool, which every config of the state shares
    /// ([`StateMemory`]'s `snapshot`).
    snapshot: u64,
    /// The copies' index vectors, [`RING`] sets.
    indices: Vec<Buffer>,
    /// Recorded behind the last batch whose copies read each set of index vectors.
    read: Vec<Option<Event>>,
    /// The set the next batch's copies write.
    turn: usize,
    /// The stream the copies are issued on: the stream of the runner that holds the state.
    stream: Stream,
    /// What the two kernels launch under: that stream and the serving generation.
    context: ops::Context,
}

impl Buffers {
    /// Every buffer of the config, in no particular order.
    fn into_buffers(self) -> impl Iterator<Item = Buffer> {
        [self.live, self.checkpoint, self.trail, self.drafter]
            .into_iter()
            .flatten()
            .flatten()
            .chain(self.indices)
    }

    /// Gives every buffer back to the resident pool of `allocator`, once the stream passed
    /// every copy and forward that addresses them.
    fn release(self, allocator: &mut Allocator) -> Result<()> {
        self.stream.synchronize()?;

        let mut released = Ok(());
        for buffer in self.into_buffers() {
            let result = allocator.release(buffer);
            if released.is_ok() {
                released = result;
            }
        }
        Ok(released?)
    }
}

/// What a slot's last step left of its state, which a switch to a state of other drafts moves
/// it by ([`StateMemory::switch`]): how many tokens of that step stand
/// (`nn::VerifySlots::num_accepted`; one for a runner whose forwards lend no verify slots), and
/// the tokens the slot's cache holds, its verified length, which the next forward starts at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Standing {
    /// Tokens of the slot's last step that stand, at least one.
    pub accepted: u32,
    /// Tokens the slot's cache holds: the position of the next forward's first token.
    pub verified: u32,
}

/// The tensors a state lends a runner's forwards ([`StateMemory::lent`]): the live and the
/// checkpoint buffers, the trail of a state that keeps one, and the drafter's live buffers of a
/// drafter that keeps a state, each in the order and extents `nn::RecurrentState` holds them.
pub struct Tensors {
    pub live: Vec<Vec<Tensor>>,
    pub checkpoint: Vec<Vec<Tensor>>,
    pub trail: Option<Vec<Vec<Tensor>>>,
    pub drafter: Option<Vec<Vec<Tensor>>>,
}

/// A state of another config than the serving one, which one kernel generation laid out
/// (module documentation, generations of other drafts).
struct Beside {
    /// The generation's number.
    number: u64,
    config: StateConfig,
    device: Option<Buffers>,
    /// Whether this is the state the generation's switch replaced, kept until its commit or its
    /// rollback; otherwise the generation's own, prepared or rolled back from.
    replaced: bool,
}

/// The recurrent-state buffers of one rank and the copies that move them.
pub struct StateMemory {
    config: StateConfig,
    /// `[layer][field]` where each field lies in a state unit: its first plane and its bytes.
    planes: Vec<Vec<(u32, u64)>>,
    /// The planes of every checkpoint of a state unit.
    page_planes: u32,
    /// Bytes of one plane of the cache.
    plane_bytes: u64,
    /// Units of the state region.
    units: u32,
    /// The cache's layout, which a state of another config lays its checkpoints out by.
    layout: Arc<CacheLayout>,
    /// The memory on a device; `None` for a state that holds none.
    device: Option<Buffers>,
    /// The snapshot every config of the state copies through, from the scratch pool; `None` for
    /// a state that holds no device.
    snapshot: Option<Buffer>,
    /// The state a kernel generation of other graph options laid out beside this one.
    beside: Option<Beside>,
}

impl StateMemory {
    /// The state of `config` over the state units of a cache of `cache`, holding no device: its
    /// tensors are meta tensors, and its copies are checked and issue nothing.
    ///
    /// # Errors
    ///
    /// A config of no slot, layer or field, a drafter's field that keeps a slot per verified
    /// token, or a cache whose layout holds no state group for the checkpoints.
    pub fn meta(config: StateConfig, cache: &KvCacheConfig) -> Result<Self> {
        Self::meta_over(config, &cache.layout, cache.states.0)
    }

    /// [`Self::meta`] over the state region of `units` units of a cache of `layout`.
    fn meta_over(config: StateConfig, layout: &Arc<CacheLayout>, units: u32) -> Result<Self> {
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

        if layout.state_group().is_none() {
            return Err(Error::Invalid(
                "a model that keeps a recurrent state needs a state group".into(),
            ));
        }
        if layout.state_bytes() == 0 || layout.checkpoints.len() != layers.len() {
            return Err(Error::Invalid(format!(
                "the cache's state units hold the checkpoints of {} recurrent layers, and the \
                 state keeps {}",
                layout.checkpoints.len(),
                layers.len()
            )));
        }

        let planes: Vec<Vec<(u32, u64)>> = layers
            .iter()
            .enumerate()
            .map(|(layer, fields)| layout.state_planes(layer as u32, fields))
            .collect::<Result<_>>()?;

        // Every field's planes lie inside its layer's checkpoint, which the layout laid out
        // from the same runs.
        for (fields, checkpoint) in planes.iter().zip(&layout.checkpoints) {
            let end = u64::from(checkpoint.first_plane) + u64::from(checkpoint.planes);

            for (first, bytes) in fields {
                if u64::from(*first) + bytes.div_ceil(layout.plane_bytes()) > end {
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
            plane_bytes: layout.plane_bytes(),
            units,
            layout: Arc::clone(layout),
            device: None,
            snapshot: None,
            beside: None,
        })
    }

    /// The state of `config` over the state units of a cache of `cache`, on the device of
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
        let snapshot =
            allocator.allocate(PoolKind::Scratch, cache.layout.state_bytes() as usize)?;
        state.device = Some(Self::buffers(
            &state.config,
            &cache.layout,
            allocator,
            (stream, kernels),
            snapshot.address(),
        )?);
        state.snapshot = Some(snapshot);
        Ok(state)
    }

    /// The buffers of `config` over the state units of a cache of `layout`, taken from the
    /// resident pool of `allocator`, its copies issued on the stream and launched with the
    /// kernels of `launch` through the snapshot at `snapshot`. Their contents are unspecified.
    fn buffers(
        config: &StateConfig,
        layout: &CacheLayout,
        allocator: &mut Allocator,
        (stream, kernels): (Stream, Arc<ops::Generation>),
        snapshot: u64,
    ) -> Result<Buffers> {
        // One buffer per field of every layer of `runs`, of the rows `rows` states for the
        // field.
        let mut take =
            |runs: &[StateShape], rows: &dyn Fn(&StateField) -> u32| -> Result<Vec<Vec<Buffer>>> {
                layer_fields(runs)
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|field| {
                                let bytes = u64::from(rows(field)) * field.bytes()?;
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

        let index_bytes = config.index_bytes(layout) as usize;
        let indices = (0..RING)
            .map(|_| allocator.allocate(PoolKind::Resident, index_bytes))
            .collect::<platform::Result<Vec<_>>>()?;

        // The context allocates nothing: every tensor of a copy is over the state's buffers,
        // its snapshot, its index vectors or the cache's state units.
        let device = allocator.device();
        let ordinal = usize::try_from(device.0)
            .map_err(|_| Error::Invalid(format!("{device} names no GPU ordinal")))?;
        let region = DeviceRange::new(DevicePtr::from_address(snapshot), 0);
        let context = ops::Context::new(ordinal, stream.clone(), kernels, region);

        Ok(Buffers {
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
        })
    }

    /// The config the state serves.
    pub fn config(&self) -> &StateConfig {
        &self.config
    }

    /// The config of the state kernel generation `number` laid out beside the serving one
    /// ([`Self::prepare`]); `None` when it laid out none.
    pub fn prepared_config(&self, number: u64) -> Option<&StateConfig> {
        self.beside
            .as_ref()
            .filter(|beside| beside.number == number && !beside.replaced)
            .map(|beside| &beside.config)
    }

    /// The device bytes the state's buffers take from the resident pool
    /// ([`StateConfig::bytes`]).
    ///
    /// # Errors
    ///
    /// As [`StateConfig::bytes`].
    pub fn bytes(&self) -> Result<u64> {
        self.config.bytes(&self.layout)
    }

    /// The state of `config` over the same state units, holding no device: what a kernel
    /// generation of other graph options measures a runner's shape over before it lays the
    /// state out ([`Self::prepare`]).
    ///
    /// # Errors
    ///
    /// As [`Self::meta`].
    pub fn alike(&self, config: StateConfig) -> Result<Self> {
        Self::meta_over(config, &self.layout, self.units)
    }

    /// Lays the state of `config` out for kernel generation `number` beside the serving one,
    /// over the same state units, its buffers taken from the resident pool of `allocator`,
    /// which the serving ones lie in, and its copies issued on the serving state's stream
    /// (module documentation, generations of other drafts). Its copies launch with the kernels
    /// [`Self::prepared`] is given. A state that holds no device lays out nothing.
    ///
    /// # Errors
    ///
    /// Another generation's state held beside this one, a config of other recurrent layers or
    /// request slots, what [`Self::meta`] refuses, or a pool that cannot hold the buffers.
    pub fn prepare(
        &mut self,
        number: u64,
        config: StateConfig,
        allocator: &mut Allocator,
    ) -> Result<()> {
        if let Some(beside) = &self.beside {
            return Err(Error::State(if beside.number == number {
                "the generation laid its state out already"
            } else {
                "another generation's state lies beside the serving one"
            }));
        }
        let checked = self.alike(config)?;
        if checked.planes != self.planes || checked.config.slots != self.config.slots {
            return Err(Error::Invalid(format!(
                "a state of {} slots whose checkpoints lie in other planes than the {} slots' \
                 the state units hold",
                checked.config.slots, self.config.slots
            )));
        }
        let config = checked.config;

        let device = match &self.device {
            Some(serving) => Some(Self::buffers(
                &config,
                &self.layout,
                allocator,
                (
                    serving.stream.clone(),
                    Arc::new(ops::Generation::new(number, [])),
                ),
                serving.snapshot,
            )?),
            None => None,
        };
        self.beside = Some(Beside {
            number,
            config,
            device,
            replaced: false,
        });
        Ok(())
    }

    /// The tensors of the state generation `number` laid out ([`Self::prepare`]), whose copies
    /// launch with `kernels`, the generation's, once it serves.
    ///
    /// # Errors
    ///
    /// No state of the generation prepared, or a config whose buffers the state cannot lend.
    pub fn prepared(&self, number: u64, kernels: Arc<ops::Generation>) -> Result<Tensors> {
        let beside = self
            .beside
            .as_ref()
            .filter(|beside| beside.number == number && !beside.replaced)
            .ok_or(Error::State("the generation laid out no state"))?;
        if let Some(device) = &beside.device {
            device.context.set_kernels(kernels);
        }
        self.lent_of(&beside.config, beside.device.as_ref())
    }

    /// The tensors the serving state lends a runner's forwards.
    ///
    /// # Errors
    ///
    /// A config whose buffers the state cannot lend.
    pub fn lent(&self) -> Result<Tensors> {
        self.lent_of(&self.config, self.device.as_ref())
    }

    /// The tensors of the state of `config` whose memory is `device`.
    fn lent_of(&self, config: &StateConfig, device: Option<&Buffers>) -> Result<Tensors> {
        let slots = |_: &StateField| config.slots;
        let tensors = |runs: &[StateShape],
                       rows: &dyn Fn(&StateField) -> u32,
                       buffers: &dyn Fn(&Buffers) -> &Vec<Vec<Buffer>>| {
            self.field_tensors(device, runs, rows, false, buffers)
        };

        Ok(Tensors {
            live: tensors(
                &config.shapes,
                &|field| config.live_rows(field),
                &|buffers| &buffers.live,
            )?,
            checkpoint: tensors(&config.shapes, &slots, &|buffers| &buffers.checkpoint)?,
            trail: match config.trail {
                0 => None,
                trail => Some(tensors(&config.shapes, &|_| trail, &|buffers| {
                    &buffers.trail
                })?),
            },
            drafter: match layer_fields(&config.drafter).next() {
                None => None,
                Some(_) => Some(tensors(&config.drafter, &slots, &|buffers| {
                    &buffers.drafter
                })?),
            },
        })
    }

    /// Kernel generation `number` switches the state it laid out in: every slot's state moves
    /// from the serving buffers into the generation's, as `standing` states what each slot's
    /// last step left, slot `s` at index `s − 1` (module documentation), and the generation's
    /// serve from then on, the copies issued on the stream ahead of the next forward. The
    /// buffers that served are kept until the generation commits or rolls back. Returns whether
    /// the generation laid a state out.
    ///
    /// The caller issues the switch behind every forward the serving buffers took, on the same
    /// stream, and states the counts those forwards left.
    ///
    /// # Errors
    ///
    /// Counts of other request slots than the state's, or the driver's refusal of the copies.
    pub fn switch(&mut self, number: u64, standing: &[Standing]) -> Result<bool> {
        let Some(beside) = self
            .beside
            .as_mut()
            .filter(|beside| beside.number == number && !beside.replaced)
        else {
            return Ok(false);
        };
        Self::restate(
            (&self.config, self.device.as_ref()),
            (&beside.config, beside.device.as_ref()),
            standing,
        )?;

        std::mem::swap(&mut self.config, &mut beside.config);
        std::mem::swap(&mut self.device, &mut beside.device);
        beside.replaced = true;
        Ok(true)
    }

    /// Kernel generation `number` rolls back: every slot's state moves from the generation's
    /// buffers back into those its switch kept ([`Self::switch`]), which serve again, and the
    /// generation's are held until [`Self::release`] gives them back. Returns whether the
    /// generation's switch kept a state.
    ///
    /// # Errors
    ///
    /// As [`Self::switch`].
    pub fn switch_back(&mut self, number: u64, standing: &[Standing]) -> Result<bool> {
        let Some(beside) = self
            .beside
            .as_mut()
            .filter(|beside| beside.number == number && beside.replaced)
        else {
            return Ok(false);
        };
        Self::restate(
            (&self.config, self.device.as_ref()),
            (&beside.config, beside.device.as_ref()),
            standing,
        )?;

        std::mem::swap(&mut self.config, &mut beside.config);
        std::mem::swap(&mut self.device, &mut beside.device);
        beside.replaced = false;
        Ok(true)
    }

    /// Kernel generation `number` committed: the buffers its switch replaced, or the ones it
    /// laid out and never served from, are given back to the resident pool of `allocator` once
    /// the stream passed them. Nothing is held for a generation that laid out no state.
    ///
    /// # Errors
    ///
    /// The driver's refusal to unmap a buffer; the rest are given back all the same.
    pub fn commit(&mut self, number: u64, allocator: &mut Allocator) -> Result<()> {
        match self.beside.take() {
            Some(beside) if beside.number == number => match beside.device {
                Some(device) => device.release(allocator),
                None => Ok(()),
            },
            other => {
                self.beside = other;
                Ok(())
            }
        }
    }

    /// Gives back the buffers kernel generation `number` laid out, which never served or whose
    /// rollback moved the slots' state out of them ([`Self::switch_back`]).
    ///
    /// # Errors
    ///
    /// The generation's state serves, since its switch was not rolled back; the driver's refusal
    /// to unmap a buffer.
    pub fn release(&mut self, number: u64, allocator: &mut Allocator) -> Result<()> {
        self.release_where(allocator, |held| held == number)
    }

    /// Gives back the buffers a kernel generation before `number` laid out, whose update ended:
    /// one a peer's process kept past its rollback while its ranks still addressed it.
    ///
    /// # Errors
    ///
    /// As [`Self::release`].
    pub fn release_before(&mut self, number: u64, allocator: &mut Allocator) -> Result<()> {
        self.release_where(allocator, |held| held < number)
    }

    /// Gives back the generation's state beside the serving one when `given` names its number.
    fn release_where(
        &mut self,
        allocator: &mut Allocator,
        given: impl Fn(u64) -> bool,
    ) -> Result<()> {
        match self.beside.take() {
            Some(beside) if given(beside.number) && beside.replaced => {
                self.beside = Some(beside);
                Err(Error::State(
                    "the generation's state serves: its switch is not rolled back",
                ))
            }
            Some(beside) if given(beside.number) => match beside.device {
                Some(device) => device.release(allocator),
                None => Ok(()),
            },
            other => {
                self.beside = other;
                Ok(())
            }
        }
    }

    /// Moves every slot's state from the buffers of `from` into those of `to`, two configs of
    /// the same recurrent layers over the same request slots, as `standing` states each slot's
    /// last step (module documentation, generations of other drafts): one copy batch on the
    /// stream, which a state that holds no device checks and issues nothing of.
    ///
    /// # Errors
    ///
    /// Configs of other layers, fields or request slots, counts of other slots, a field whose
    /// rows the other config does not hold, or the driver's refusal.
    fn restate(
        (from, source): (&StateConfig, Option<&Buffers>),
        (to, target): (&StateConfig, Option<&Buffers>),
        standing: &[Standing],
    ) -> Result<()> {
        if from.slots != to.slots || standing.len() != from.slots as usize {
            return Err(Error::Invalid(format!(
                "a state of {} slots moves into one of {} by the counts of {} slots",
                from.slots,
                to.slots,
                standing.len()
            )));
        }

        // The addresses of the live buffers of the model's layers, then of the drafter's when
        // both configs keep its state; none for a state that holds no device.
        let addresses = |device: Option<&Buffers>, drafter: bool| -> Vec<Vec<u64>> {
            device
                .map(|device| {
                    if drafter {
                        &device.drafter
                    } else {
                        &device.live
                    }
                })
                .map(|buffers| {
                    buffers
                        .iter()
                        .map(|fields| fields.iter().map(Buffer::address).collect())
                        .collect()
                })
                .unwrap_or_default()
        };

        let mut batch = CopyBatch::new(SourceOrder::Stream);
        let drafters = !from.drafter.is_empty() && !to.drafter.is_empty();
        let parts = [(false, &from.shapes, &to.shapes, from.owned)]
            .into_iter()
            .chain(drafters.then_some((true, &from.drafter, &to.drafter, 1)));
        for (drafter, runs, into, owned) in parts {
            let layers: Vec<&[StateField]> = layer_fields(runs).collect();
            let targets: Vec<&[StateField]> = layer_fields(into).collect();
            if layers.len() != targets.len() {
                return Err(Error::Invalid(format!(
                    "a state of {} recurrent layers moves into one of {}",
                    layers.len(),
                    targets.len()
                )));
            }
            let (sources, destinations) = (addresses(source, drafter), addresses(target, drafter));
            // The drafter's forwards are lent its live rows alone: its state stands where a
            // count of one reads it.
            let standing: Vec<Standing> = standing
                .iter()
                .map(|slot| Standing {
                    accepted: if drafter { 1 } else { slot.accepted },
                    ..*slot
                })
                .collect();

            for (layer, (fields, other)) in layers.iter().zip(&targets).enumerate() {
                if fields.len() != other.len() {
                    return Err(Error::Invalid(format!(
                        "recurrent layer {layer} keeps {} fields and moves into {}",
                        fields.len(),
                        other.len()
                    )));
                }
                for (index, (field, into)) in fields.iter().zip(other.iter()).enumerate() {
                    let at = |addresses: &Vec<Vec<u64>>| {
                        addresses
                            .get(layer)
                            .and_then(|fields| fields.get(index))
                            .copied()
                    };
                    let moved = moves(field, into, from.slots, owned, &standing)?;
                    if let (Some(from), Some(to)) = (at(&sources), at(&destinations)) {
                        for (destination, source, bytes) in moved {
                            batch.push(to + destination, from + source, bytes as usize);
                        }
                    }
                }
            }
        }

        let Some(device) = source else {
            return Ok(());
        };
        // SAFETY: every range is one slot's rows inside a buffer of one of the two states,
        // bounded by `moves`; the sources and the destinations are buffers of two states, so no
        // destination is read or written twice, and the copies are ordered on the stream behind
        // the forwards that wrote the sources and ahead of every forward that reads the
        // destinations.
        unsafe { batch.submit(&device.stream) }?;
        Ok(())
    }

    /// The home row request slot `slot` owns of the live and checkpoint buffers: what the
    /// batch's state slot column states for the slot's row.
    ///
    /// # Errors
    ///
    /// A slot outside `1..=slots`.
    pub(crate) fn row(&self, slot: u32) -> Result<u32> {
        if slot == 0 || slot > self.config.slots {
            return Err(Error::Invalid(format!(
                "request slot {slot} is outside the recurrent state's slots 1..={}",
                self.config.slots
            )));
        }
        Ok(slot - 1)
    }

    /// The tensor of `shape` elements of `dtype` `offset` bytes past the address `address`
    /// states of the device side `device`, or the meta tensor of that shape.
    fn tensor(
        device: Option<&Buffers>,
        address: impl FnOnce(&Buffers) -> u64,
        offset: u64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor> {
        match device {
            // SAFETY: the callers bound the window to the buffer, which the state owns for as
            // long as it lives; only the forwards of the runner that holds the state and the
            // copies this state issues, all on one stream, address it.
            Some(device) => unsafe {
                lend(address(device) + offset, shape, dtype, Some(device.device))
            },
            // SAFETY: a meta tensor has no data and its address is never read.
            None => unsafe { lend(0, shape, dtype, None) },
        }
    }

    /// The tensors of one kind of buffer of the layers of `runs` in `device`, `[layer][field]`:
    /// each `[rows, …]` in its field's element type and extents, or `[rows, bytes]` U8 when
    /// `raw`, as a copy moves it, `rows` being what `rows` states for the field.
    fn field_tensors(
        &self,
        device: Option<&Buffers>,
        runs: &[StateShape],
        rows: &dyn Fn(&StateField) -> u32,
        raw: bool,
        buffers: &dyn Fn(&Buffers) -> &Vec<Vec<Buffer>>,
    ) -> Result<Vec<Vec<Tensor>>> {
        let shape = |field: &StateField| -> Result<Vec<usize>> {
            let mut dims = vec![rows(field) as usize];
            if raw {
                dims.push(field.bytes()? as usize);
            } else {
                dims.extend(&field.dims);
            }
            Ok(dims)
        };

        layer_fields(runs)
            .enumerate()
            .map(|(layer, fields)| {
                fields
                    .iter()
                    .enumerate()
                    .map(|(index, field)| {
                        let dtype = if raw { DType::U8 } else { field.dtype };
                        Self::tensor(
                            device,
                            |device| buffers(device)[layer][index].address(),
                            0,
                            &shape(field)?,
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
        Ok(self.lent()?.live)
    }

    /// The checkpoint buffers, in the same order and extents as [`Self::live`] but a row per
    /// slot of every field: a checkpoint keeps the state a home row holds.
    pub fn checkpoint(&self) -> Result<Vec<Vec<Tensor>>> {
        Ok(self.lent()?.checkpoint)
    }

    /// The trail buffers, in the same order, `[trail, …]`; `None` in a deployment that keeps
    /// no trail.
    pub fn trail(&self) -> Result<Option<Vec<Vec<Tensor>>>> {
        Ok(self.lent()?.trail)
    }

    /// The drafter's live buffers (module documentation, the drafter's state): one entry per
    /// layer of the drafter's runs ([`StateConfig::drafter`]), each the layer's fields,
    /// `[slots, …]`, as `nn::RecurrentState::live` holds them for the drafter's forwards;
    /// `None` for a drafter that keeps no state.
    pub(crate) fn drafter(&self) -> Result<Option<Vec<Vec<Tensor>>>> {
        Ok(self.lent()?.drafter)
    }

    /// The descriptor of a block copy over `field`, a `[blocks, width]` byte tensor.
    fn block_shape(field: &ops::Operand) -> BlockShape {
        BlockShape::new(
            DType::U8,
            field.dims().iter().map(|&dim| dim as u64).collect(),
            field
                .strides()
                .iter()
                .map(|&stride| stride as u64)
                .collect(),
            1,
            field.last(),
        )
    }

    /// Launches one move between `field` and `part` of the snapshot by the context installed
    /// on this thread: the rows `blocks` names of `field` into `part` when `gather`, and
    /// `part` into those rows otherwise.
    fn launch_move(gather: bool, field: &Tensor, blocks: &Tensor, part: &Tensor) -> Result<()> {
        let field = ops::operand(field)?;
        let blocks = ops::operand(blocks)?;
        let snapshot = ops::operand(part)?;

        if gather {
            ops::launch::<CacheBlockSnapshot>(
                &Self::block_shape(&field),
                SnapshotArgs {
                    field,
                    blocks,
                    snapshot,
                },
            )?;
        } else {
            ops::launch::<CacheBlockScatter>(
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
    /// buffers of a keep) and one unit's planes of the state units `units`, through the
    /// snapshot. `planes` is the I64 vector of the unit's rows of `units`, and
    /// `snapshot(offset, rows, width)` the `[rows, width]` bytes `offset` bytes into the
    /// snapshot.
    fn launch_copy(
        &self,
        direction: StateDirection,
        (buffers, slot): (&[Vec<Tensor>], &Tensor),
        (units, planes): (&Tensor, &Tensor),
        snapshot: &dyn Fn(u64, usize, u64) -> Result<Tensor>,
    ) -> Result<()> {
        // One launch per field of every layer, between the part of the slot's row a checkpoint
        // keeps, its leading bytes, and the field's planes of the snapshot.
        let rows = |gather: bool| -> Result<()> {
            for (layer, fields) in self.planes.iter().enumerate() {
                for (index, (first, bytes)) in fields.iter().enumerate() {
                    let part = snapshot(u64::from(*first) * self.plane_bytes, 1, *bytes)?;
                    let kept = buffers[layer][index].narrow(1, 0, *bytes as usize)?;
                    Self::launch_move(gather, &kept, slot, &part)?;
                }
            }
            Ok(())
        };

        // One launch between the unit's planes, which lie back to back in the units, and the
        // whole snapshot.
        let page = |gather: bool| -> Result<()> {
            let part = snapshot(0, self.page_planes as usize, self.plane_bytes)?;
            Self::launch_move(gather, units, planes, &part)
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

        let rows = self.units as usize * self.page_planes as usize;
        let units = meta(&[rows, self.plane_bytes as usize], DType::U8)?;
        let buffers = layer_fields(&self.config.shapes)
            .map(|fields| {
                fields
                    .iter()
                    .map(|field| {
                        meta(
                            &[self.config.slots as usize, field.bytes()? as usize],
                            DType::U8,
                        )
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;

        let planes = meta(&[self.page_planes as usize], DType::I64)?;

        let context = ops::Context::meta(None);
        context.record_launches();
        {
            let _installed = context.enter();

            for direction in [StateDirection::Restore, StateDirection::Keep] {
                self.launch_copy(
                    direction,
                    (&buffers, &meta(&[1], DType::I64)?),
                    (&units, &planes),
                    &|_, rows, width| meta(&[rows, width as usize], DType::U8),
                )?;
            }
        }

        Ok(context
            .take_launches()
            .into_iter()
            .map(|(operator, key, ..)| (operator, key))
            .collect())
    }

    /// The generation the copies launch from, from the next copy on: set at a generation
    /// switch.
    pub(crate) fn set_kernels(&self, kernels: Arc<ops::Generation>) {
        if let Some(device) = &self.device {
            device.context.set_kernels(kernels);
        }
    }

    /// Issues a batch's copies of `direction` between the slots' buffers and the state units of
    /// `cache`, on the stream the state was opened on: every restore before the batch's forward,
    /// every keep after it. Each copy is a request slot and the state unit its checkpoint entry
    /// names, the one unit of the entry's run. A restore moves a unit's planes into its slot's
    /// live row; a keep moves its slot's checkpoint row into a unit's planes. Every copy is
    /// checked before anything is issued, and a state that holds no device issues nothing.
    ///
    /// # Errors
    ///
    /// More copies than the batch's rows allow, a slot outside the state's slots, a unit outside
    /// the state region, or a launch's refusal.
    pub fn copy(
        &mut self,
        cache: &KvCacheMemory,
        copies: &[(u32, u32)],
        direction: StateDirection,
    ) -> Result<()> {
        let mut pages = Vec::new();
        for (slot, unit) in copies {
            pages.push((self.row(*slot)?, cache.state_rows(*unit)?));
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

        // Per copy: its slot's row, then the row of each of its unit's planes.
        let stride = self.page_planes as usize + 1;
        let mut values: Vec<i64> = Vec::with_capacity(pages.len() * stride);
        for (row, planes) in &pages {
            values.push(i64::from(*row));
            values.extend(planes.clone());
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
        let held = Some(device);
        let buffers = match direction {
            StateDirection::Restore => {
                self.field_tensors(held, model, &slots, true, &|buffers| &buffers.live)?
            }
            StateDirection::Keep => {
                self.field_tensors(held, model, &slots, true, &|buffers| &buffers.checkpoint)?
            }
        };
        let units = cache.state_tensor()?;
        let snapshot = |offset: u64, rows: usize, width: u64| {
            Self::tensor(
                held,
                |device| device.snapshot,
                offset,
                &[rows, width as usize],
                DType::U8,
            )
        };

        {
            let _installed = device.context.enter();

            for index in 0..pages.len() {
                let at = index * stride;
                let entries = |first: usize, count: usize| {
                    Self::tensor(
                        held,
                        |device| device.indices[turn].address(),
                        ((at + first) * 8) as u64,
                        &[count],
                        DType::I64,
                    )
                };

                self.launch_copy(
                    direction,
                    (&buffers, &entries(0, 1)?),
                    (&units, &entries(1, self.page_planes as usize)?),
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
                let bytes = field.bytes()?;
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

/// The copies that move every slot's state of `field` into `into`, the same field of another
/// config, as `standing` states each slot's last step, slot `s` at index `s − 1`:
/// `(destination, source, bytes)`, the two offsets into the fields' live buffers. `slots` are
/// the request slots and `owned` the slots a request owns of a stepwise field under the config
/// of `field` (module documentation, generations of other drafts):
///
/// - a field no draft count changes moves whole from the row that stands: for a stepwise field
///   that a count above one names, the request's further slot `accepted − 2`
///   ([`StateConfig::live_rows`]), and the home row otherwise;
/// - a window moves its history, from row `accepted − 1` of the slot's window to row zero;
/// - a ring moves the positions before the slot's verified length that both rings hold, each
///   from its row of one ring to its row of the other, in runs of rows that follow each other in
///   both.
///
/// # Errors
///
/// Fields of other names, element types, rows or trailing extents, a field no draft count
/// changes whose extents differ, or a count of accepted tokens past the slots or the rows the
/// field holds.
fn moves(
    field: &StateField,
    into: &StateField,
    slots: u32,
    owned: u32,
    standing: &[Standing],
) -> Result<Vec<(u64, u64, u64)>> {
    if field.name != into.name
        || field.dtype != into.dtype
        || field.rows != into.rows
        || field.dims.get(1..) != into.dims.get(1..)
    {
        return Err(Error::Invalid(format!(
            "the state's field {} of {:?} moves into the field {} of {:?}",
            field.name, field.dims, into.name, into.dims
        )));
    }
    let (bytes, target, row) = (field.bytes()?, into.bytes()?, field.row_bytes()?);
    let (held, holds) = (
        field.dims.first().copied().unwrap_or(1) as u64,
        into.dims.first().copied().unwrap_or(1) as u64,
    );

    let mut moves = Vec::with_capacity(standing.len());
    for (slot, standing) in (0_u64..).zip(standing) {
        let accepted = u64::from(standing.accepted.max(1));
        if field.stepwise && accepted > u64::from(owned) {
            return Err(Error::Invalid(format!(
                "slot {} accepted {accepted} tokens, and a request owns {owned} slots of the \
                 field {}",
                slot + 1,
                field.name
            )));
        }
        let stands = if field.stepwise && accepted > 1 {
            u64::from(slots) + slot * u64::from(owned - 1) + accepted - 2
        } else {
            slot
        };
        let (source, destination) = (stands * bytes, slot * target);

        match field.rows {
            StateRows::Whole => {
                if bytes != target {
                    return Err(Error::Invalid(format!(
                        "the state's field {} of {:?} moves into one of {:?}",
                        field.name, field.dims, into.dims
                    )));
                }
                moves.push((destination, source, bytes));
            }
            StateRows::Window { history } => {
                let (first, history) = (accepted - 1, history as u64);
                if first + history > held || history > holds {
                    return Err(Error::Invalid(format!(
                        "a window of {held} rows reads {history} inputs from row {first}, into \
                         one of {holds}",
                    )));
                }
                moves.push((destination, source + first * row, history * row));
            }
            StateRows::Ring { .. } => {
                let end = u64::from(standing.verified);
                let mut position = end.saturating_sub(held.min(holds));
                while position < end {
                    let (from, to) = (position % held, position % holds);
                    let run = (held - from).min(holds - to).min(end - position);
                    moves.push((destination + to * row, source + from * row, run * row));
                    position += run;
                }
            }
        }
    }
    Ok(moves)
}

#[cfg(test)]
mod tests {
    use ops::Op;
    use proto::request::PrefixCacheDtype;

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
                    rows: nn::family::StateRows::Whole,
                },
                StateField {
                    name: "recurrent",
                    dtype: DType::F32,
                    dims: vec![10],
                    stepwise: false,
                    rows: nn::family::StateRows::Whole,
                },
            ],
        }
    }

    /// The keys and values of `layers` attention layers of 4-token pages of one 8-wide BF16
    /// head.
    fn keys_and_values(layers: u32) -> CacheLayout {
        CacheLayout::of(
            4,
            &nn::family::CacheField::keys_and_values(layers, 1, 8),
            PrefixCacheDtype::BFloat16,
        )
        .expect("a dense layout")
    }

    /// The cache of two attention layers of 4-token pages of one 8-wide BF16 head beside that
    /// state: planes of 64 bytes, so a layer's checkpoint is two planes and a state unit four,
    /// in a state region of eight units.
    fn cache() -> KvCacheConfig {
        let layout = keys_and_values(2)
            .with_state(&[shape()])
            .expect("a hybrid layout");
        assert_eq!(layout.plane_bytes(), 64);
        assert_eq!(layout.groups[1].units_per_page, 1);

        KvCacheConfig {
            layout: Arc::new(layout),
            units: 16,
            slots: 3,
            width: 4,
            segments: None,
            workspace_bytes: 64,
            states: (8, None),
            windows: Vec::new(),
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
            config(5).bytes(&layout).unwrap(),
            (2 * 3 + 5) * 2 * 52 + 4 * (4 * 5 * 8)
        );
        assert_eq!(layout.state_bytes(), 4 * 64);
        Ok(())
    }

    /// A verification advances a state of rings in place when each ring holds the verification's
    /// tokens beside the rows of the group open before them, `rows ≥ tokens + group − 2`, and
    /// keeps a trail otherwise; a state with any field that is no ring keeps a trail, unless a
    /// field keeps a slot per verified token, and a model of no state keeps none.
    #[test]
    fn a_ring_that_holds_a_verification_advances_in_place() {
        let field = |rows: usize, kind: nn::family::StateRows, stepwise: bool| StateField {
            name: "remainder",
            dtype: DType::F32,
            dims: vec![rows, 16],
            stepwise,
            rows: kind,
        };
        let state = |fields: Vec<StateField>| vec![StateShape { layers: 3, fields }];
        let ring = |rows, group| field(rows, nn::family::StateRows::Ring { group }, false);

        // DeepSeek-V4.1's compressor remainder, eight rows in pairs: a verification of six
        // tokens (five drafts) advances it in place, as one of eight does; one of nine would
        // rewrite the row of its own first token.
        assert!(!keeps_trail(&state(vec![ring(8, 2)]), 6));
        assert!(!keeps_trail(&state(vec![ring(8, 2)]), 8));
        assert!(keeps_trail(&state(vec![ring(8, 2)]), 9));

        // In groups of four the next forward may read the two rows before the verification's
        // first token, which a verification of seven tokens rewrites.
        assert!(!keeps_trail(&state(vec![ring(8, 4)]), 6));
        assert!(keeps_trail(&state(vec![ring(8, 4)]), 7));

        // A field beside the ring that is none keeps a trail, a stepwise one keeps verify slots
        // instead, and no state keeps nothing.
        let whole = field(4, nn::family::StateRows::Whole, false);
        assert!(keeps_trail(&state(vec![ring(8, 2), whole]), 2));
        let stepwise = field(4, nn::family::StateRows::Whole, true);
        assert!(!keeps_trail(&state(vec![ring(8, 2), stepwise]), 2));
        assert!(!keeps_trail(&[], 6));
    }

    /// A deployment that does not speculate keeps no trail.
    #[test]
    fn a_state_without_a_trail_lends_none() -> Result<()> {
        let state = StateMemory::meta(config(0), &cache())?;
        assert!(state.trail()?.is_none());
        assert_eq!((state.live()?.len(), state.checkpoint()?.len()), (2, 2));
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

    /// A copy is checked against the state's slots and the state region's units before it is
    /// issued: every unit of the region, the first included, holds a checkpoint.
    #[test]
    fn a_state_copy_names_a_slot_and_a_unit_of_the_state_region() -> Result<()> {
        let config_of_cache = cache();
        let cache = KvCacheMemory::meta(config_of_cache.clone())?;
        let mut state = StateMemory::meta(config(0), &config_of_cache)?;
        let mut restore =
            |copies: &[(u32, u32)]| state.copy(&cache, copies, StateDirection::Restore);

        restore(&[(2, 5)])?;
        restore(&[(1, 0)])?;
        assert!(restore(&[(4, 5)]).is_err());

        // The region's units are 0..8.
        let outside = restore(&[(1, 8)]).expect_err("a unit past the region");
        assert!(outside.to_string().contains("units 0..8"), "{outside}");

        // A batch of two rows carries at most a restore and a keep per row.
        let many: Vec<(u32, u32)> = (0..5).map(|_| (1, 5)).collect();
        let refused = restore(&many).expect_err("five copies of a batch of two rows");
        assert!(refused.to_string().contains("allow 4"), "{refused}");
        Ok(())
    }

    /// The copies state their kernels with no device: a snapshot and a scatter over each
    /// buffer field and over the state units.
    #[test]
    fn the_copies_state_their_kernels_under_a_meta_context() -> Result<()> {
        let state = StateMemory::meta(config(0), &cache())?;

        let keys = state.keys()?;
        let of = |operator: &str| keys.iter().filter(|(name, _)| *name == operator).count();

        // Each operator over three shapes: the two fields' buffers and the state units.
        assert_eq!(of(CacheBlockSnapshot::NAME), 3, "{keys:?}");
        assert_eq!(of(CacheBlockScatter::NAME), 3, "{keys:?}");
        assert_eq!(keys.len(), 6);
        Ok(())
    }

    /// A state needs the cache's state group to checkpoint into.
    #[test]
    fn a_state_needs_a_state_group() {
        let mut dense = cache();
        dense.layout = Arc::new(keys_and_values(2));
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
                    rows: nn::family::StateRows::Whole,
                }],
            },
        ]
    }

    /// The cache of three attention layers beside [`runs`]: three columns of 64-byte planes,
    /// the two layers of the first run two planes each and the last layer one.
    fn runs_cache() -> Result<KvCacheConfig> {
        let layout = keys_and_values(3).with_state(&runs())?;

        Ok(KvCacheConfig {
            layout: Arc::new(layout),
            ..cache()
        })
    }

    /// A state's buffers follow its runs in order, each layer with its own run's fields. A
    /// request owns `Q` slots of a stepwise field, whose live buffer holds them past the home
    /// rows, and one checkpoint row; such a state keeps no trail.
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

        // Each layer's checkpoint lies in its own planes of the state unit: two planes for a
        // layer of the first run and one for the last.
        let first: Vec<u32> = cache
            .layout
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.first_plane)
            .collect();
        assert_eq!(first, [0, 2, 4]);
        assert_eq!(cache.layout.checkpoint_planes(), 5);

        // Three slots of 52 bytes in each of two layers, live and checkpoint, nine live slots
        // and three checkpoints of 16 bytes, and four sets of index vectors of a slot row and
        // five planes for each of the four copies a batch of two rows may carry.
        assert_eq!(
            config.bytes(&cache.layout).unwrap(),
            2 * 2 * 3 * 52 + (9 + 3) * 16 + 4 * (4 * 6 * 8)
        );

        // Its requests own several slots of a stepwise field, so it keeps no trail, and one that
        // states a trail is refused.
        assert!(config.verifies() && state.trail()?.is_none());
        assert!(StateMemory::meta(StateConfig { trail: 4, ..config }, &cache).is_err());
        assert!(!self::config(0).verifies());

        // A state of other runs than the ones its cache's state units hold.
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
                    rows: nn::family::StateRows::Whole,
                },
                StateField {
                    name: "index_ring_positions",
                    dtype: DType::I32,
                    dims: vec![4, 3],
                    stepwise: false,
                    rows: nn::family::StateRows::Whole,
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
        assert_eq!(
            drafting.bytes(&layout).unwrap(),
            config(0).bytes(&layout).unwrap() + 3 * 112
        );
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
