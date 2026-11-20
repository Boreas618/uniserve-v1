//! The layout of a rank's KV cache, as the model's cache fields describe it.
//!
//! The model states its cache as fields ([`nn::family::CacheField`]): each one plane per owner
//! layer, one entry per `ratio` tokens, over a history. The layout gives every history a group
//! and every group's fields a **pool** of units:
//!
//! - the full-history fields form the first group, whose pool is the KV region's: a unit (unit 0
//!   is the padding unit no page holds) is one page of every owner of every field; the draft
//!   cache's group lies in the same units ([`CacheLayout::with_draft`]);
//! - each sliding window's fields form a group of their own, whose pool is a region of rank
//!   memory sized by the request slots ([`KvCacheConfig::windows`](super::KvCacheConfig)): a
//!   window keeps only the pages of the window of a running request's next tokens and no page
//!   outlives its request, so its units are a working set beside the slots, not a cache;
//! - a model that keeps a recurrent state per request keeps its checkpoints apart from the pools,
//!   in the scheduler's *state region* (`specs/engine-worker-port.md` §1.6): a state unit is one
//!   checkpoint of every recurrent layer, [`CacheLayout::checkpoint_planes`] planes of the KV
//!   pool's plane bytes back to back ([`CacheLayout::state_bytes`]), and the state group's page is
//!   one such unit.
//!
//! A pool is one storage per field, a *column* per owner: the owner's page of a unit is its
//! column's cell of the unit, and a column's cells lie back to back, so an owner's entries are the
//! contiguous `[units, entries, heads, width]` of its column and a page id is a unit id. A cell's
//! entries are a *plane*. The KV pool's *plane bytes* are its first field's first plane, one
//! layer's keys of one page for a model of keys and values, the quantum the state region's
//! checkpoints are counted in; the greatest common divisor of its planes is the quantum its
//! segments are counted in ([`CacheLayout::segment_plane`]).
//!
//! The model owns its fields, so the layout is derived from them ([`CacheLayout::of`],
//! [`CacheLayout::with_state`], [`CacheLayout::with_draft`]) and checked once
//! ([`CacheLayout::validated`]). How many units and request slots a rank's cache holds is not part
//! of it: those are the capacity a rank resolves ([`KvCacheConfig`](super::KvCacheConfig)).

use proto::request::PrefixCacheDtype;

use super::info::PrefixCacheInfo;
use crate::batch::cache::{CacheGroup, CacheGroupKind};
use nn::family::{
    CacheField, FieldOwner, FieldReader, History, KEY, StateField, StateShape, VALUE,
};
use tensor::DType;

use crate::memory::state::layer_fields;
use crate::worker::{Error, Result};

/// One recurrent layer's checkpoint, packed into consecutive planes of a state page.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct Checkpoint {
    /// The checkpoint's first plane among the state page's planes.
    pub first_plane: u32,
    /// The planes the checkpoint fills.
    pub planes: u32,
    /// The checkpoint's fields as a cache manifest names them; empty until the Storage Server
    /// keys state pages by them.
    pub fields: Vec<::storage::manifest::Field>,
}

/// One field of a pool: a storage of a column per owner, each column one plane a unit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldLayout {
    /// The field's name, which names its storages (`prefix_cache.{name}.values`, …) and the
    /// planes a forward finds ([`nn::KvCache::plane`]).
    pub name: &'static str,
    /// The owners in column order: column `c` holds owner `owners[c]`'s entries.
    pub owners: Vec<FieldOwner>,
    /// The layers that read an owner's entries in place of their own.
    pub readers: Vec<FieldReader>,
    /// The element type the entries are stored in: the deployment's, the record's own, or U8 for
    /// a block-scaled record.
    pub dtype: DType,
    /// The extents of one entry, `[heads, width]` of `dtype` ([`nn::family::Record::dims`]).
    pub dims: [u32; 2],
    /// Whether each plane keeps one F32 scale per page: a field in the deployment's FP8 type.
    pub scaled: bool,
}

impl FieldLayout {
    /// The bytes of one entry.
    pub fn entry_bytes(&self) -> u64 {
        u64::from(self.dims[0]) * u64::from(self.dims[1]) * element_bytes(self.dtype)
    }

    /// The entries column `column`'s page of `block` tokens holds: one per `ratio` tokens.
    pub fn entries(&self, column: usize, block: u32) -> u32 {
        self.owners
            .get(column)
            .map_or(0, |owner| block.checked_div(owner.ratio).unwrap_or(0))
    }

    /// The bytes of column `column`'s plane, its page of `block` tokens.
    pub fn plane_bytes(&self, column: usize, block: u32) -> u64 {
        u64::from(self.entries(column, block)) * self.entry_bytes()
    }

    /// The columns of the field: one per owner.
    pub fn columns(&self) -> u32 {
        self.owners.len() as u32
    }

    /// The bytes of the planes of every column of one unit, for pages of `block` tokens.
    pub fn unit_bytes(&self, block: u32) -> u64 {
        (0..self.owners.len())
            .map(|column| self.plane_bytes(column, block))
            .sum()
    }

    /// The bytes before column `column`'s first cell in a storage of `units` cells a column,
    /// for pages of `block` tokens: the columns before it, whole.
    pub fn column_offset(&self, column: usize, units: u32, block: u32) -> u64 {
        (0..column)
            .map(|before| self.plane_bytes(before, block) * u64::from(units))
            .sum()
    }

    /// The field as a cache field states it, resolved for a deployment whose cache element type
    /// is `deployment`.
    ///
    /// # Errors
    ///
    /// A record of no whole entry ([`nn::family::Record::dims`]).
    fn of(field: &CacheField, deployment: DType) -> Result<Self> {
        let dims = field
            .record
            .dims()
            .map_err(|error| Error::Invalid(error.to_string()))?;

        Ok(Self {
            name: field.name,
            owners: field.owners.clone(),
            readers: field.readers.clone(),
            dtype: field.record.dtype(deployment),
            dims,
            scaled: field.record.scaled(deployment),
        })
    }
}

/// The units of one pool: the fields every unit holds a page of, for pages of `block_size`
/// tokens.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolLayout {
    /// Tokens of one page, which a unit holds of every column of every field.
    pub block_size: u32,
    /// The pool's fields, in the model's order.
    pub fields: Vec<FieldLayout>,
}

impl PoolLayout {
    /// The field `name`, when the pool holds it.
    pub fn field(&self, name: &str) -> Option<&FieldLayout> {
        self.fields.iter().find(|field| field.name == name)
    }

    /// The bytes one unit's planes take over every column of every field: what a segment of
    /// the pool gives back per unit ([`CacheLayout::split_bytes`]).
    pub fn split_bytes(&self) -> u64 {
        self.fields
            .iter()
            .map(|field| field.unit_bytes(self.block_size))
            .sum()
    }

    /// Device bytes one unit takes over every column: its planes, one U8 flag per plane and one
    /// F32 scale per plane of a scaled field.
    pub fn unit_bytes(&self) -> u64 {
        let cells: u64 = self
            .fields
            .iter()
            .map(|field| u64::from(field.columns()) * (1 + if field.scaled { 4 } else { 0 }))
            .sum();
        self.split_bytes() + cells
    }

    /// The cells of one unit: one per column of every field.
    pub fn cells(&self) -> u64 {
        self.fields
            .iter()
            .map(|field| u64::from(field.columns()))
            .sum()
    }

    /// The pool's first field's first plane: the plane of a state page of a cache whose KV pool
    /// this is. Zero for a pool of no plane.
    pub fn plane_bytes(&self) -> u64 {
        self.fields
            .first()
            .map_or(0, |field| field.plane_bytes(0, self.block_size))
    }

    /// The greatest common divisor of the pool's planes: the quantum its segments are counted in
    /// (`super::segments::segment_units`), so that every column's part of a segment is whole
    /// allocation granules. Zero for a pool of no plane.
    pub fn segment_plane(&self) -> u64 {
        self.fields
            .iter()
            .flat_map(|field| {
                (0..field.owners.len()).map(|column| field.plane_bytes(column, self.block_size))
            })
            .fold(0, gcd64)
    }

    /// Whether a field of the pool keeps page scales.
    pub fn quantized(&self) -> bool {
        self.fields.iter().any(|field| field.scaled)
    }

    /// Whether the pool's fields are a model's keys and values of one record, every owner one
    /// entry per token over the same layers: the fields a draft group's layers lie over in
    /// spans of columns ([`CacheLayout::with_draft`]).
    fn keys_and_values(&self) -> Option<(&FieldLayout, &FieldLayout)> {
        let (key, value) = (self.field(KEY)?, self.field(VALUE)?);
        let uniform = |field: &FieldLayout| field.owners.iter().all(|owner| owner.ratio == 1);

        (key.owners == value.owners
            && key.dims == value.dims
            && key.dtype == value.dtype
            && uniform(key))
        .then_some((key, value))
    }
}

/// One cache group: the fields of one history and their page tables, the draft cache, or the
/// state group, whose pages hold the checkpoints of the layers kept per request slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupLayout {
    /// The group's history policy. A state group's pages hold each of its layers' state after
    /// the page's last token; a draft group's hold a drafter's keys and values, each layer's
    /// page lying over several columns of its unit.
    pub kind: CacheGroupKind,
    /// Tokens of one page.
    pub block_size: u32,
    /// Units one page occupies: units of its pool for a field group or a draft group, units of
    /// the state region for the state group, whose page is one checkpoint and one unit there.
    pub units_per_page: u32,
    /// The pool the group's pages are units of ([`CacheLayout::pools`]); the KV pool, zero, for
    /// the state group, whose pages lie in the state region.
    pub pool: usize,
    /// This rank's layers, ascending: the owners of the group's fields.
    pub layers: Vec<String>,
    /// The global cache-layer ids of those layers.
    pub layer_ids: Vec<u32>,
    /// Where this rank's layers start in the group's layer axis of a transfer between ranks.
    pub axis_offset: u32,
    /// The layers of that axis, in global cache-layer order.
    pub axis_layers: u32,
    /// The rows of an entry of the group's first field this rank stores: key/value heads of a
    /// key/value cache.
    pub num_kv_heads: u32,
    /// Those rows across the tensor-parallel group.
    pub total_kv_heads: u32,
    /// The first of them this rank stores.
    pub kv_head_offset: u32,
    /// The width of a row of the group's first field.
    pub head_dim: u32,
    /// The element type of the group's first field.
    pub dtype: DType,
}

impl GroupLayout {
    /// Whether the group's pages hold state checkpoints.
    pub fn state(&self) -> bool {
        self.kind.is_state()
    }

    /// Whether the group holds the draft cache.
    pub fn draft(&self) -> bool {
        self.kind.is_draft()
    }

    /// Whether the group holds entries every request needs, which the model's attention layers
    /// read: a field group, neither the state group nor the draft cache.
    pub fn attention(&self) -> bool {
        self.kind.required_rows()
    }

    /// Whether the group is a sliding window's, whose pages lie in a pool of their own.
    pub fn window(&self) -> bool {
        self.kind.window().is_some()
    }
}

/// The bytes of one element of `dtype`; zero for an element narrower than a byte, which no
/// cache plane stores and [`CacheLayout::validated`] refuses.
fn element_bytes(dtype: DType) -> u64 {
    ops::abi::bytes(dtype).unwrap_or(0)
}

/// The element type the cache stores for the engine's `dtype`.
fn stored(dtype: PrefixCacheDtype) -> DType {
    ops::abi::from_wire(dtype.dtype())
}

/// The layout of a rank's cache: its pools, its groups and the checkpoints of its state region.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheLayout {
    /// The pools, the KV pool first, then one per sliding-window group in group order.
    pub pools: Vec<PoolLayout>,
    /// The deployment's cache element type as the engine names it (`PrefixCacheInfo::dtype`):
    /// what the fields in the deployment's format store.
    pub dtype: PrefixCacheDtype,
    /// The groups: the full-history group, the sliding-window groups, the draft group, the
    /// state group last.
    pub groups: Vec<GroupLayout>,
    /// One checkpoint per recurrent layer, in the plane order of a unit of the state region.
    pub checkpoints: Vec<Checkpoint>,
}

impl CacheLayout {
    /// The cache of `fields`, a model's cache fields over pages of `block_size` tokens, whose
    /// fields in the deployment's format store `dtype`: the full-history fields' group over the
    /// KV pool, then a group and a pool per sliding window, shortest first. Every owner of a
    /// group is one of its layers, numbered by the model's layer index.
    ///
    /// # Errors
    ///
    /// Fields [`CacheField::check`] refuses, a cache of no full-history field (the KV pool is the
    /// full history's, which the prefix cache keys), or a layout [`Self::validated`] refuses.
    pub fn of(block_size: u32, fields: &[CacheField], dtype: PrefixCacheDtype) -> Result<Self> {
        CacheField::check(fields, block_size).map_err(|error| Error::Invalid(error.to_string()))?;
        let deployment = stored(dtype);

        let mut histories: Vec<History> = Vec::new();
        for field in fields {
            if !histories.contains(&field.history) {
                histories.push(field.history);
            }
        }
        // The full history first, then the windows by their length.
        histories.sort_by_key(|history| match history {
            History::Full => 0,
            History::Window { tokens } => u64::from(*tokens) + 1,
        });
        if histories.first() != Some(&History::Full) {
            return Err(Error::Invalid(
                "a cache keeps at least one full-history field: its pool is the one the prefix \
                 cache keys, and a sliding window's pages are never cached"
                    .into(),
            ));
        }

        let mut layout = Self {
            pools: Vec::new(),
            dtype,
            groups: Vec::new(),
            checkpoints: Vec::new(),
        };
        for history in histories {
            let held: Vec<FieldLayout> = fields
                .iter()
                .filter(|field| field.history == history)
                .map(|field| FieldLayout::of(field, deployment))
                .collect::<Result<_>>()?;
            let kind = match history {
                History::Full => CacheGroupKind::Full,
                History::Window { tokens } => CacheGroupKind::SlidingWindow { window: tokens },
            };

            let pool = layout.pools.len();
            layout.groups.push(group_of(kind, block_size, pool, &held));
            layout.pools.push(PoolLayout {
                block_size,
                fields: held,
            });
        }

        layout.validated()
    }

    /// The KV pool: the full-history group's, which the draft group lies in too.
    pub fn kv(&self) -> &PoolLayout {
        &self.pools[0]
    }

    /// Device bytes one unit of the KV pool takes over every column: its planes, flags and
    /// scales ([`PoolLayout::unit_bytes`]).
    pub fn unit_bytes(&self) -> u64 {
        self.kv().unit_bytes()
    }

    /// The KV pool's plane bytes ([`PoolLayout::plane_bytes`]): the quantum of the state region's
    /// checkpoints.
    pub fn plane_bytes(&self) -> u64 {
        self.kv().plane_bytes()
    }

    /// The quantum the KV pool's segments are counted in ([`PoolLayout::segment_plane`]).
    pub fn segment_plane(&self) -> u64 {
        self.kv().segment_plane()
    }

    /// Columns of the KV pool's first field: the columns a draft group's layers lie over.
    pub fn columns(&self) -> u32 {
        self.kv().fields.first().map_or(0, FieldLayout::columns)
    }

    /// How this rank lays its part of a unit out, as it states it to the Storage Server with its
    /// runs (`storage::protocol::UnitLayout`): the full-history group, the class of its index in
    /// the session's layout, which a handoff between replicas reshards by its key/value heads.
    /// The draft and the state groups are not handed off: a decode replica refills the one and
    /// the other lies in the state region.
    ///
    /// # Errors
    ///
    /// A cache a handoff does not carry, named: one with a sliding-window group, whose window a
    /// handoff would have to replay, or whose full-history fields are other than keys and values
    /// in the deployment's format and index keys, which the handoff's page layout does not state.
    pub(crate) fn unit_layout(&self) -> Result<::storage::protocol::UnitLayout> {
        if let Some(window) = self.groups.iter().find(|group| group.window()) {
            return Err(Error::Unsupported(format!(
                "a handoff between replicas carries no sliding-window group, and the cache keeps \
                 one of {:?} tokens",
                window.kind.window()
            )));
        }
        let pool = self.kv();
        let (key, _) = pool.keys_and_values().ok_or_else(|| {
            Error::Unsupported(format!(
                "a handoff between replicas carries keys and values, and the cache's full-history \
                 fields are {:?}",
                pool.fields
                    .iter()
                    .map(|field| field.name)
                    .collect::<Vec<_>>()
            ))
        })?;
        let index = match pool.fields.as_slice() {
            [_, _] => 0,
            [_, _, index] if index.name == nn::family::INDEX_KEY => {
                index.plane_bytes(0, pool.block_size)
            }
            fields => {
                return Err(Error::Unsupported(format!(
                    "a handoff between replicas carries keys, values and index keys, and the \
                     cache's full-history fields are {:?}",
                    fields.iter().map(|field| field.name).collect::<Vec<_>>()
                )));
            }
        };

        let classes = self
            .groups
            .iter()
            .enumerate()
            .filter(|(_, group)| group.attention())
            .map(|(class, group)| ::storage::protocol::PageLayout {
                class: class as u8,
                block_size: group.block_size,
                layers: group.layer_ids.clone(),
                kv_heads: group.num_kv_heads,
                kv_head_offset: group.kv_head_offset,
                total_kv_heads: group.total_kv_heads,
                head_bytes: (u64::from(key.dims[1]) * element_bytes(key.dtype)) as u32,
                quantized: pool.quantized(),
                index_bytes: index,
            })
            .collect();
        Ok(::storage::protocol::UnitLayout {
            columns: self.columns(),
            classes,
        })
    }

    /// The layout, once its fields are checked against each other.
    ///
    /// # Errors
    ///
    /// When a group has no layers, a page of a group takes no unit or holds no token, a field
    /// stores an element narrower than a byte or holds no column, a group names no pool or a pool
    /// of other pages, the state group is not the last group or its page is not one unit of the
    /// state region, the draft group does not lie over one unit of the KV pool, or a pool holds
    /// no plane.
    pub fn validated(self) -> Result<Self> {
        if self.groups.is_empty() || self.pools.is_empty() {
            return Err(Error::Invalid(
                "a cache layout needs at least one group and one pool".into(),
            ));
        }

        for group in &self.groups {
            if group.layers.is_empty() || group.layers.len() != group.layer_ids.len() {
                return Err(Error::Invalid(format!(
                    "cache group {:?} needs one global id per layer",
                    group.layers
                )));
            }
            if group.units_per_page == 0 || group.block_size == 0 {
                return Err(Error::Invalid(format!(
                    "cache group {:?} needs a page of at least one token and one unit",
                    group.layers
                )));
            }
            if group.kv_head_offset + group.num_kv_heads > group.total_kv_heads {
                return Err(Error::Invalid(format!(
                    "cache group {:?} stores heads it does not have",
                    group.layers
                )));
            }
            if group.axis_offset + group.layers.len() as u32 > group.axis_layers {
                return Err(Error::Invalid(format!(
                    "cache group {:?} lies outside its transfer layer axis",
                    group.layers
                )));
            }
            if group.state() {
                continue;
            }
            let pool = self.pools.get(group.pool).ok_or_else(|| {
                Error::Invalid(format!(
                    "cache group {:?} lies in pool {}, of {} pools",
                    group.layers,
                    group.pool,
                    self.pools.len()
                ))
            })?;
            // A draft page holds `span` runs of an attention page's tokens in one unit.
            let tokens = if group.draft() {
                group.block_size / self.draft_span().unwrap_or(1)
            } else {
                group.block_size
            };
            if pool.block_size != tokens || group.units_per_page != 1 {
                return Err(Error::Invalid(format!(
                    "cache group {:?} lays pages of {tokens} tokens over {} units of pool {}, \
                     whose unit is one page of {} tokens",
                    group.layers, group.units_per_page, group.pool, pool.block_size
                )));
            }
        }

        for (index, pool) in self.pools.iter().enumerate() {
            for field in &pool.fields {
                if element_bytes(field.dtype) == 0 || field.owners.is_empty() {
                    return Err(Error::Invalid(format!(
                        "the cache field {} of pool {index} stores {} in {} columns: a whole-byte \
                         element in at least one",
                        field.name,
                        ops::abi::name(field.dtype),
                        field.owners.len()
                    )));
                }
                if (0..field.owners.len())
                    .any(|column| field.plane_bytes(column, pool.block_size) == 0)
                {
                    return Err(Error::Invalid(format!(
                        "the cache field {} of pool {index} keeps a plane of no byte",
                        field.name
                    )));
                }
            }
            if pool.segment_plane() == 0 {
                return Err(Error::Invalid(format!(
                    "pool {index} of the cache holds no plane"
                )));
            }
            let groups = self
                .groups
                .iter()
                .filter(|group| !group.state() && group.pool == index && !group.draft())
                .count();
            if groups != 1 {
                return Err(Error::Invalid(format!(
                    "pool {index} of the cache holds the pages of {groups} field groups, and a \
                     pool is one group's"
                )));
            }
        }

        if let Some((last, rest)) = self.groups.split_last()
            && rest.iter().any(GroupLayout::state)
        {
            return Err(Error::Invalid(format!(
                "the state group must be the last cache group, which is {:?}",
                last.layers
            )));
        }
        if let Some(state) = self.groups.iter().find(|group| group.state())
            && state.units_per_page != 1
        {
            return Err(Error::Invalid(format!(
                "the state group's page is one unit of the state region, not {}",
                state.units_per_page
            )));
        }

        // A draft group's layers lie over the columns of one unit of the KV pool's keys and
        // values ([`Self::with_draft`]).
        if let Some(draft) = self.groups.iter().find(|group| group.draft())
            && (draft.pool != 0 || draft.layers.len() as u32 > self.columns())
        {
            return Err(Error::Invalid(format!(
                "the draft group's {} layers lie over the {} columns of one unit of the KV pool",
                draft.layers.len(),
                self.columns(),
            )));
        }
        Ok(self)
    }

    /// The device bytes one unit of the KV pool holds over every column in the planes a device
    /// cache splits into segments: every field's planes, which a segment's release gives back
    /// and its creation takes (`crate::memory::cache::segments`). The flags and scales of a unit
    /// are no part of it.
    pub fn split_bytes(&self) -> u64 {
        self.kv().split_bytes()
    }

    /// The bytes of one unit of the state region on this rank: every recurrent layer's
    /// checkpoint, [`Self::checkpoint_planes`] planes of [`Self::plane_bytes`] each. It is also
    /// the snapshot a recurrent-state copy moves a unit through, which the device's scratch
    /// pool holds once ([`StateMemory`](crate::memory::state::StateMemory)). Zero for a cache
    /// that keeps no state group.
    pub fn state_bytes(&self) -> u64 {
        if self.state_group().is_none() {
            return 0;
        }
        u64::from(self.checkpoint_planes()) * self.plane_bytes()
    }

    /// The planes of every checkpoint of a unit of the state region.
    pub fn checkpoint_planes(&self) -> u32 {
        self.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.first_plane.saturating_add(checkpoint.planes))
            .max()
            .unwrap_or(0)
    }

    /// Where each field of recurrent layer `layer`'s checkpoint lies in a unit of the state
    /// region, in the order of `fields`: the field's first plane among the unit's planes and the
    /// bytes a checkpoint keeps of it (`StateField::kept_bytes`), which fill whole planes from
    /// that one ([`Self::with_state`]).
    ///
    /// # Errors
    ///
    /// A field whose element type has no whole-byte width (`StateField::bytes`).
    pub fn state_planes(&self, layer: u32, fields: &[StateField]) -> Result<Vec<(u32, u64)>> {
        let mut plane = self
            .checkpoints
            .get(layer as usize)
            .map_or(0, |checkpoint| checkpoint.first_plane);

        fields
            .iter()
            .map(|field| {
                let bytes = field.kept_bytes()?;
                let at = (plane, bytes);
                plane += bytes.div_ceil(self.plane_bytes().max(1)) as u32;
                Ok(at)
            })
            .collect()
    }

    /// The cache as the rank reports it to the engine, for a KV pool of `num_units` units. The
    /// rank fills in what it learns later: its segments, identities and endpoint.
    pub fn info(&self, num_units: u32) -> PrefixCacheInfo {
        PrefixCacheInfo {
            num_units,
            dtype: self.dtype,
            groups: self
                .groups
                .iter()
                .map(|group| CacheGroup {
                    kind: group.kind,
                    block_size: group.block_size,
                    units_per_page: group.units_per_page,
                    layer_ids: group.layer_ids.clone(),
                    num_kv_heads: group.num_kv_heads,
                    total_kv_heads: group.total_kv_heads,
                    kv_head_offset: group.kv_head_offset,
                    head_dim: group.head_dim,
                })
                .collect(),
            plane_runs: Vec::new(),
            scratch_units: 0,
            segments: Vec::new(),
            held_segments: 0,
            inherited_start: 0,
            inherited_units: 0,
        }
    }

    /// The units of the KV pool a capacity of `tokens` tokens takes: the whole pages of those
    /// tokens in every field group of the KV pool. The draft group's pages take units the
    /// capacity holds as far as they are free, a sliding window's lie in a pool of their own, and
    /// checkpoints lie in the state region.
    pub fn token_units(&self, tokens: u64) -> u64 {
        self.groups
            .iter()
            .filter(|group| group.attention() && group.pool == 0)
            .map(|group| tokens / u64::from(group.block_size) * u64::from(group.units_per_page))
            .sum()
    }

    /// The cache with the state group of the recurrent state `runs`, the runs of recurrent
    /// layers that keep the same fields in the model's order of them
    /// (`nn::family::TextModel::state_layers`): the last group, whose page is one unit of the
    /// state region holding every recurrent layer's checkpoint in that order.
    ///
    /// A checkpoint lays its layer's fields out in field order, each from a plane boundary over
    /// the whole planes its bytes fill ([`Self::state_planes`]), and the checkpoints follow each
    /// other, [`Self::checkpoint_planes`] planes in all ([`Self::state_bytes`]). A checkpoint
    /// keeps one slot of every field, a stepwise one's included, and of each slot the rows it
    /// holds at zero drafts (`StateField::kept_bytes`): what a forward from a block boundary
    /// reads, so the state region's units, and every checkpoint they hold on the device and on
    /// the host, are the same whatever the deployment drafts.
    ///
    /// # Errors
    ///
    /// A layout that keeps a state group already, no recurrent layer, a run whose layers keep no
    /// field, or a field of no bytes.
    pub fn with_state(mut self, runs: &[StateShape]) -> Result<Self> {
        let recurrent: u32 = runs.iter().map(|run| run.layers).sum();
        let fieldless = runs
            .iter()
            .any(|run| run.layers > 0 && run.fields.is_empty());
        if recurrent == 0 || fieldless || self.state_group().is_some() {
            return Err(Error::Invalid(format!(
                "a state group of {recurrent} recurrent layers, each of which keeps a field, \
                 beside no other"
            )));
        }

        // Every field starts at a plane boundary, so a copy moves whole planes, and every
        // layer's checkpoint starts where the one before it ends.
        let plane_bytes = self.plane_bytes();
        let mut planes = 0_u64;
        for fields in layer_fields(runs) {
            let mut layer_planes = 0_u64;
            for field in fields {
                let bytes = field.kept_bytes()?;
                if bytes == 0 {
                    return Err(Error::Invalid(format!(
                        "the recurrent state's field {} holds no byte",
                        field.name
                    )));
                }
                layer_planes += bytes.div_ceil(plane_bytes.max(1));
            }

            let (first_plane, planes_held) = u32::try_from(planes)
                .ok()
                .zip(u32::try_from(layer_planes).ok())
                .ok_or_else(|| {
                    Error::Invalid("a checkpoint fills more planes than a page counts".into())
                })?;
            self.checkpoints.push(Checkpoint {
                first_plane,
                planes: planes_held,
                fields: Vec::new(),
            });
            planes += layer_planes;
        }

        let first = self.groups[0].clone();
        let ids = self.next_layer_id();
        self.groups.push(GroupLayout {
            kind: CacheGroupKind::State,
            block_size: first.block_size,
            // One checkpoint is one unit of the state region.
            units_per_page: 1,
            pool: 0,
            layers: (0..recurrent)
                .map(|layer| format!("recurrent.{layer}"))
                .collect(),
            layer_ids: (ids..ids + recurrent).collect(),
            axis_offset: 0,
            axis_layers: recurrent,
            // A state page holds bytes: one head of one element per layer.
            num_kv_heads: 1,
            total_kv_heads: 1,
            kv_head_offset: 0,
            head_dim: 1,
            dtype: DType::BF16,
        });

        self.validated()
    }

    /// The next global cache-layer id past every group's layers.
    fn next_layer_id(&self) -> u32 {
        self.groups
            .iter()
            .flat_map(|group| group.layer_ids.iter().copied())
            .max()
            .map_or(0, |id| id + 1)
    }

    /// The layout with the draft cache added: the fields `fields` of a drafter's layers
    /// (`nn::Drafter::cache`), which are the KV pool's keys and values (and index keys, when it
    /// keeps them), over fewer layers.
    ///
    /// A drafter has fewer layers than a unit has columns, so a draft page does not take a cell
    /// per layer. It takes **one unit** of the KV pool, and each of its layers lies over
    /// [`Self::draft_span`] of the unit's columns: column `c` of the layer's run holds the
    /// page's `c`-th run of `block_size / span` tokens. A draft page therefore holds `span`
    /// times the tokens of an attention page in one unit. The group sits before the state
    /// group, which stays last.
    ///
    /// # Errors
    ///
    /// Drafter fields that are not the KV pool's keys and values over the full history, a
    /// drafter of more layers than the pool has columns, or a layout the draft group makes
    /// invalid.
    pub fn with_draft(mut self, fields: &[CacheField]) -> Result<Self> {
        let pool = self.kv();
        let first = self.groups[0].clone();
        CacheField::check(fields, pool.block_size)
            .map_err(|error| Error::Invalid(error.to_string()))?;
        if let Some(field) = fields.iter().find(|field| field.history != History::Full) {
            return Err(Error::Unsupported(format!(
                "a drafter's cache field {} keeps a sliding window, and the draft cache keeps \
                 the full history",
                field.name
            )));
        }
        let (key, _) = pool.keys_and_values().ok_or_else(|| {
            Error::Invalid(
                "a draft cache lies over the KV pool's keys and values, and the pool keeps none"
                    .into(),
            )
        })?;

        // The drafter's fields are the pool's, field for field, over its own layers.
        let deployment = stored(self.dtype);
        let layers = fields.first().map_or(0, |field| field.owners.len() as u32);
        let mut matches = fields.len() == pool.fields.len();
        for (field, held) in fields.iter().zip(&pool.fields) {
            let laid = FieldLayout::of(field, deployment)?;
            let ratios = |field: &FieldLayout| -> Vec<u32> {
                field.owners.iter().map(|owner| owner.ratio).collect()
            };
            matches &= laid.name == held.name
                && laid.dims == held.dims
                && laid.dtype == held.dtype
                && laid.owners.len() as u32 == layers
                && ratios(&laid)
                    .iter()
                    .all(|ratio| Some(ratio) == ratios(held).first());
        }
        if !matches || layers == 0 || layers > key.columns() {
            return Err(Error::Invalid(format!(
                "a drafter's fields {:?} over the KV pool's {:?} of {} columns: the drafter keeps \
                 the pool's fields, field for field, over fewer layers",
                fields.iter().map(|field| field.name).collect::<Vec<_>>(),
                pool.fields
                    .iter()
                    .map(|field| field.name)
                    .collect::<Vec<_>>(),
                key.columns()
            )));
        }

        let span = span_of(key.columns(), layers);
        let ids = self.next_layer_id();
        let draft = GroupLayout {
            kind: CacheGroupKind::Draft,
            block_size: first.block_size * span,
            units_per_page: 1,
            pool: 0,
            layers: (0..layers).map(|layer| format!("draft.{layer}")).collect(),
            layer_ids: (ids..ids + layers).collect(),
            axis_offset: 0,
            axis_layers: layers,
            ..first
        };

        let at = self
            .groups
            .iter()
            .position(GroupLayout::state)
            .unwrap_or(self.groups.len());
        self.groups.insert(at, draft);
        self.validated()
    }

    /// The columns of a unit each layer of the draft group lies over; `None` without one.
    pub fn draft_span(&self) -> Option<u32> {
        let draft = self.groups.iter().find(|group| group.draft())?;
        Some(span_of(self.columns(), draft.layers.len() as u32))
    }

    /// The draft group's index, when the layout holds one.
    pub fn draft_group(&self) -> Option<usize> {
        self.groups.iter().position(GroupLayout::draft)
    }

    /// The state group's index, when the layout holds one.
    pub fn state_group(&self) -> Option<usize> {
        self.groups.iter().position(GroupLayout::state)
    }

    /// How many field groups the layout holds, which the model's attention layers read: they
    /// come first, the full history's and then the sliding windows'. The draft cache and the
    /// state group are none of them.
    pub fn attention_groups(&self) -> usize {
        self.groups.iter().filter(|group| group.attention()).count()
    }

    /// The sliding-window groups' indices with their pools, in group order.
    pub fn window_groups(&self) -> Vec<(usize, usize)> {
        self.groups
            .iter()
            .enumerate()
            .filter(|(_, group)| group.window())
            .map(|(index, group)| (index, group.pool))
            .collect()
    }
}

/// The group of `kind` over `fields` of `pool`, its pages of `block_size` tokens one unit each:
/// its layers the owners of its fields, ascending, by the model's layer index; its heads and head
/// width the first field's.
fn group_of(
    kind: CacheGroupKind,
    block_size: u32,
    pool: usize,
    fields: &[FieldLayout],
) -> GroupLayout {
    let mut layers: Vec<u32> = fields
        .iter()
        .flat_map(|field| field.owners.iter().map(|owner| owner.layer))
        .collect();
    layers.sort_unstable();
    layers.dedup();
    let first = fields.first();
    let heads = first.map_or(1, |field| field.dims[0]);

    GroupLayout {
        kind,
        block_size,
        units_per_page: 1,
        pool,
        layers: layers
            .iter()
            .map(|layer| format!("layers.{layer}"))
            .collect(),
        layer_ids: layers.clone(),
        axis_offset: 0,
        axis_layers: layers.last().map_or(0, |last| last + 1),
        num_kv_heads: heads,
        total_kv_heads: heads,
        kv_head_offset: 0,
        head_dim: first.map_or(1, |field| field.dims[1]),
        dtype: first.map_or(DType::U8, |field| field.dtype),
    }
}

/// The columns of a unit each of `layers` draft layers lies over: the largest power of two the
/// columns hold `layers` runs of, so a draft page's tokens are a power of two.
fn span_of(columns: u32, layers: u32) -> u32 {
    let share = (columns / layers.max(1)).max(1);
    1 << (u32::BITS - 1 - share.leading_zeros())
}

/// The greatest common divisor of two byte counts; `gcd64(0, n) == n`.
fn gcd64(left: u64, right: u64) -> u64 {
    if right == 0 {
        left
    } else {
        gcd64(right, left % right)
    }
}

#[cfg(test)]
mod tests {
    use nn::family::{Format, INDEX_KEY, Record, StateRows};

    use super::*;

    /// A recurrent state of thirty layers: a 48 KiB convolution window and 2 MiB of state.
    fn state() -> StateShape {
        StateShape {
            layers: 30,
            fields: vec![
                StateField {
                    name: "conv",
                    dtype: DType::BF16,
                    dims: vec![3, 8192],
                    stepwise: false,
                    rows: StateRows::Whole,
                },
                StateField {
                    name: "recurrent",
                    dtype: DType::F32,
                    dims: vec![32, 128, 128],
                    stepwise: false,
                    rows: StateRows::Whole,
                },
            ],
        }
    }

    /// The keys and values of `layers` layers of `heads` heads of `head_dim`, over `block`-token
    /// pages in `dtype`.
    fn dense(
        block: u32,
        layers: u32,
        heads: u32,
        head_dim: u32,
        dtype: PrefixCacheDtype,
    ) -> CacheLayout {
        CacheLayout::of(
            block,
            &CacheField::keys_and_values(layers, heads, head_dim),
            dtype,
        )
        .expect("the keys and values of a dense cache")
    }

    /// DeepSeek-V4.1-Flash's fields over 64-token pages: the window latent of 43 layers, and the
    /// compressed latent and index keys of four owners in `compressed` and `index`.
    fn latent(compressed: Format, index: Format) -> Vec<CacheField> {
        let compressors = || {
            [(2, 2), (8, 2), (14, 2), (20, 1)]
                .into_iter()
                .map(|(layer, ratio)| FieldOwner { layer, ratio })
                .collect::<Vec<_>>()
        };
        let record = |format, width| Record::rows(format, 1, width);
        vec![
            CacheField {
                name: "latent",
                owners: (0..43)
                    .map(|layer| FieldOwner { layer, ratio: 1 })
                    .collect(),
                readers: Vec::new(),
                history: History::Window { tokens: 128 },
                record: record(Format::MXFP8, 512),
            },
            CacheField {
                name: "compressed_latent",
                owners: compressors(),
                readers: (3..=7)
                    .map(|layer| FieldReader { layer, owner: 2 })
                    .collect(),
                history: History::Full,
                record: record(compressed, 512),
            },
            CacheField {
                name: INDEX_KEY,
                owners: compressors(),
                readers: Vec::new(),
                history: History::Full,
                record: record(index, 128),
            },
        ]
    }

    /// A cache of keys and values is one full-history group over the KV pool, a column per
    /// layer of each field: a unit is every layer's key and value plane and the flag of each.
    #[test]
    fn a_dense_cache_spans_every_column_in_its_unit() {
        let (layers, block, kv_heads, head_dim) = (48, 16, 4, 128);
        let layout = dense(
            block,
            layers,
            kv_heads,
            head_dim,
            PrefixCacheDtype::BFloat16,
        );

        let plane = u64::from(kv_heads) * u64::from(head_dim) * 2 * u64::from(block);
        assert_eq!((layout.pools.len(), layout.groups.len()), (1, 1));
        assert_eq!(layout.columns(), layers);
        assert_eq!(layout.plane_bytes(), plane);
        assert_eq!(layout.segment_plane(), plane);
        assert_eq!(layout.unit_bytes(), u64::from(layers) * (2 * plane + 2));
        assert_eq!(layout.split_bytes(), 2 * u64::from(layers) * plane);
        assert_eq!(layout.groups[0].layer_ids, (0..layers).collect::<Vec<_>>());
        assert_eq!(
            (layout.groups[0].num_kv_heads, layout.groups[0].head_dim),
            (kv_heads, head_dim)
        );

        // An FP8 cache stores one byte per element and one F32 scale per plane.
        let fp8 = dense(
            block,
            layers,
            kv_heads,
            head_dim,
            PrefixCacheDtype::Float8E4m3Fn,
        );
        assert_eq!(fp8.plane_bytes(), plane / 2);
        assert_eq!(fp8.unit_bytes(), u64::from(layers) * (plane + 2 + 8));
        assert!(fp8.kv().quantized());
    }

    /// DeepSeek's fields lay the full history's out at its own page bytes, the window apart: the
    /// KV pool's unit is the compressed latents and index keys of the four owners, at each
    /// owner's ratio, and the window's pool's unit is 43 layers of 64 latents of 528 bytes.
    #[test]
    fn a_window_lies_in_a_pool_of_its_own_beside_the_full_history() -> Result<()> {
        let layout = CacheLayout::of(
            64,
            &latent(Format::NVFP4, Format::MXFP4),
            PrefixCacheDtype::BFloat16,
        )?;

        // The full history first, then the window, each its own pool.
        let kinds: Vec<(CacheGroupKind, usize)> = layout
            .groups
            .iter()
            .map(|group| (group.kind, group.pool))
            .collect();
        assert_eq!(
            kinds,
            [
                (CacheGroupKind::Full, 0),
                (CacheGroupKind::SlidingWindow { window: 128 }, 1)
            ]
        );
        assert_eq!(layout.groups[0].layer_ids, [2, 8, 14, 20]);
        assert_eq!(layout.groups[1].layer_ids, (0..43).collect::<Vec<_>>());
        assert_eq!(layout.window_groups(), [(1, 1)]);
        assert_eq!(layout.attention_groups(), 2);

        // NVFP4: 288 bytes an entry, 32 a page for a ratio of two and 64 for one; MXFP4 keys of
        // 68 bytes alike.
        let compressed = 3 * 32 * 288 + 64 * 288;
        let index = 3 * 32 * 68 + 64 * 68;
        assert_eq!(layout.split_bytes(), compressed + index);
        assert_eq!(layout.unit_bytes(), compressed + index + 8);
        assert_eq!(layout.plane_bytes(), 32 * 288);
        assert_eq!(layout.segment_plane(), 128);
        let window = &layout.pools[1];
        assert_eq!(window.fields[0].dims, [1, 528]);
        assert_eq!(window.unit_bytes(), 43 * (64 * 528 + 1));

        // MXFP8 latents and MXFP8 index keys, 132 B each.
        let wide = CacheLayout::of(
            64,
            &latent(Format::MXFP8, Format::MXFP8),
            PrefixCacheDtype::BFloat16,
        )?;
        assert_eq!(
            wide.split_bytes(),
            3 * 32 * 528 + 64 * 528 + 3 * 32 * 132 + 64 * 132
        );

        // A token capacity takes the full history's pages alone: a window's lie in its pool.
        assert_eq!(layout.token_units(640), 10);
        Ok(())
    }

    /// A model's recurrent state adds the state group last, its checkpoints in planes of the KV
    /// pool's first plane; a drafter beside it lies over the KV pool's keys and values.
    #[test]
    fn a_state_and_a_drafter_follow_the_field_groups() -> Result<()> {
        let layout = dense(64, 10, 2, 256, PrefixCacheDtype::BFloat16).with_state(&[state()])?;

        // 48 KiB of convolution window take one 64 KiB plane, 2 MiB of state thirty-two.
        assert_eq!(layout.plane_bytes(), 64 << 10);
        assert_eq!(layout.checkpoint_planes(), 30 * 33);
        assert_eq!(layout.state_bytes(), 30 * 33 * (64 << 10));
        assert_eq!(layout.state_group(), Some(1));
        assert_eq!(layout.groups[1].layer_ids, (10..40).collect::<Vec<_>>());
        assert!(layout.clone().with_state(&[state()]).is_err());

        // One drafter layer over eight of the ten columns: pages of 512 tokens in one unit,
        // before the state group.
        let drafting = layout.with_draft(&CacheField::keys_and_values(1, 2, 256))?;
        assert_eq!(drafting.draft_group(), Some(1));
        assert_eq!(drafting.state_group(), Some(2));
        assert_eq!(drafting.draft_span(), Some(8));
        assert_eq!(drafting.groups[1].block_size, 512);
        assert_eq!(drafting.groups[1].layer_ids, [40]);
        Ok(())
    }

    /// What a layout cannot hold is refused by name: no full-history field, a drafter that
    /// keeps a window or other fields than the pool's, more drafter layers than columns.
    #[test]
    fn fields_the_pools_cannot_hold_are_refused() {
        let window_only: Vec<CacheField> = latent(Format::NVFP4, Format::MXFP4)
            .into_iter()
            .take(1)
            .collect();
        let refused = CacheLayout::of(64, &window_only, PrefixCacheDtype::BFloat16)
            .unwrap_err()
            .to_string();
        assert!(refused.contains("full-history"), "{refused}");

        let layout = || dense(64, 4, 2, 128, PrefixCacheDtype::BFloat16);
        let windowed = vec![CacheField {
            history: History::Window { tokens: 64 },
            ..CacheField::attention(nn::family::KEY, 1, 2, 128)
        }];
        let refused = layout().with_draft(&windowed).unwrap_err().to_string();
        assert!(refused.contains("sliding window"), "{refused}");
        assert!(
            layout()
                .with_draft(&CacheField::keys_and_values(1, 4, 128))
                .is_err()
        );
        assert!(
            layout()
                .with_draft(&CacheField::keys_and_values(5, 2, 128))
                .is_err()
        );
        let latent = CacheLayout::of(
            64,
            &latent(Format::NVFP4, Format::MXFP4),
            PrefixCacheDtype::BFloat16,
        )
        .unwrap();
        assert!(
            latent
                .with_draft(&CacheField::keys_and_values(1, 1, 512))
                .is_err()
        );
    }

    /// A handoff carries keys, values and index keys of the full history, and refuses a cache
    /// with a sliding window or other fields, naming them.
    #[test]
    fn a_handoff_carries_keys_and_values_alone() -> Result<()> {
        let unit = dense(64, 4, 2, 128, PrefixCacheDtype::Float8E4m3Fn).unit_layout()?;
        assert_eq!(unit.columns, 4);
        let page = &unit.classes[0];
        assert_eq!(
            (page.head_bytes, page.quantized, page.index_bytes),
            (128, true, 0)
        );

        let windowed = CacheLayout::of(
            64,
            &latent(Format::NVFP4, Format::MXFP4),
            PrefixCacheDtype::BFloat16,
        )?;
        let refused = windowed.unit_layout().unwrap_err().to_string();
        assert!(refused.contains("sliding-window"), "{refused}");

        let full: Vec<CacheField> = latent(Format::NVFP4, Format::MXFP4)
            .into_iter()
            .skip(1)
            .collect();
        let refused = CacheLayout::of(64, &full, PrefixCacheDtype::BFloat16)?
            .unit_layout()
            .unwrap_err()
            .to_string();
        assert!(refused.contains("compressed_latent"), "{refused}");
        Ok(())
    }

    /// The layout's own fields are checked against its groups.
    #[test]
    fn a_layout_whose_fields_disagree_is_refused() {
        let layout = || dense(16, 4, 2, 8, PrefixCacheDtype::BFloat16);
        assert!(layout().validated().is_ok());

        let mut heads = layout();
        heads.groups[0].kv_head_offset = 1;
        assert!(heads.validated().is_err());

        let mut units = layout();
        units.groups[0].units_per_page = 0;
        assert!(units.validated().is_err());

        let mut ids = layout();
        ids.groups[0].layer_ids.pop();
        assert!(ids.validated().is_err());

        let mut pool = layout();
        pool.groups[0].pool = 1;
        assert!(pool.validated().is_err());

        let mut shared = layout();
        shared.groups.push(shared.groups[0].clone());
        assert!(shared.validated().is_err());

        let mut none = layout();
        none.groups.clear();
        assert!(none.validated().is_err());
    }
}
