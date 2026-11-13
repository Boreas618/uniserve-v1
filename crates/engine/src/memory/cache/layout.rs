//! The layout of a rank's KV cache, as the model's cache layers describe it.
//!
//! The cache is one pool of units. A *unit* is the pool's allocation quantum (unit 0 is the
//! padding unit no page holds); a *column* is a layer's position inside a unit; a *plane* is the
//! bytes of one field (keys or values) of one column of one unit; a group's page of
//! `block_size` tokens occupies the group's `units_per_page` units. A cache whose model's
//! attention reads the tokens an indexer selects keeps a third field beside them, the *index
//! keys*: one plane per column of every unit, one key per `ratio` tokens of the column's page.
//!
//! The model owns the shape of its cache layers, so the layout is derived from it
//! ([`CacheLayout::dense`], [`CacheLayout::hybrid`], [`CacheLayout::with_index_keys`],
//! [`CacheLayout::with_draft`]) and checked once ([`CacheLayout::validated`]). How many units
//! and request slots a rank's cache holds is not part of it: those are the capacity a rank
//! resolves ([`KvCacheConfig`](super::KvCacheConfig)).

use proto::request::PrefixCacheDtype;

use super::info::PrefixCacheInfo;
use crate::batch::cache::{CacheGroup, CacheGroupKind};
use nn::family::{IndexKeys, StateField, StateShape};
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

/// One cache group: layers that share a history window and a page shape, the state group, whose
/// pages hold the checkpoints of the layers kept per request slot, or the draft cache.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupLayout {
    /// The group's history policy. A state group's pages hold each of its layers' state after
    /// the page's last token; a draft group's hold a drafter's keys and values, each layer's
    /// page lying over several columns of its unit.
    pub kind: CacheGroupKind,
    /// Tokens of one page.
    pub block_size: u32,
    /// Units one page occupies.
    pub units_per_page: u32,
    /// This rank's layers in column order: layer `k` lies in column `k % columns` of a page's
    /// unit `k / columns`.
    pub layers: Vec<String>,
    /// The global cache-layer ids of those layers.
    pub layer_ids: Vec<u32>,
    /// Where this rank's layers start in the group's layer axis of a transfer between ranks.
    pub axis_offset: u32,
    /// The layers of that axis, in global cache-layer order.
    pub axis_layers: u32,
    /// Key/value heads this rank stores per layer.
    pub num_kv_heads: u32,
    /// Key/value heads of each layer across the tensor-parallel group.
    pub total_kv_heads: u32,
    /// The first of them this rank stores.
    pub kv_head_offset: u32,
    /// Elements of one head.
    pub head_dim: u32,
    /// The element type the group's planes store.
    pub dtype: DType,
    /// The element type the group's layers compute in, through which a converted value rounds.
    pub compute_dtype: DType,
    /// The index keys each layer keeps beside its keys and values, in `compute_dtype`, as the
    /// model states them ([`nn::family::CacheShape::index_keys`]); `None` for layers that
    /// attend every key.
    pub index_keys: Option<IndexKeys>,
}

impl GroupLayout {
    /// Bytes one token of one layer's keys (or values) occupies.
    pub(crate) fn row_bytes(&self) -> u64 {
        u64::from(self.num_kv_heads) * u64::from(self.head_dim) * element_bytes(self.dtype)
    }

    /// Elements of one field of one page: `layers · block_size · heads · head_dim`.
    pub(crate) fn page_elements(&self) -> u64 {
        self.layers.len() as u64
            * u64::from(self.block_size)
            * u64::from(self.num_kv_heads)
            * u64::from(self.head_dim)
    }

    /// Bytes the index keys of one layer's `tokens` tokens occupy: one key of the layers'
    /// compute type per `ratio` tokens. Zero for a group that keeps none.
    pub(crate) fn index_bytes(&self, tokens: u32) -> u64 {
        self.index_keys.map_or(0, |keys| {
            let groups = tokens.checked_div(keys.ratio).unwrap_or(0);

            u64::from(groups) * u64::from(keys.width) * element_bytes(self.compute_dtype)
        })
    }

    /// Whether the group's pages hold state checkpoints.
    pub fn state(&self) -> bool {
        self.kind.is_state()
    }

    /// Whether the group holds the draft cache.
    pub fn draft(&self) -> bool {
        self.kind.is_draft()
    }

    /// Whether the group holds the keys and values every request needs, which the model's
    /// attention layers index: neither the state group nor the draft cache.
    pub fn attention(&self) -> bool {
        self.kind.required_rows()
    }
}

/// The bytes of one element of `dtype`; zero for an element narrower than a byte, which no
/// cache plane stores and [`CacheLayout::validated`] refuses.
fn element_bytes(dtype: DType) -> u64 {
    ops::abi::bytes(dtype).unwrap_or(0)
}

/// Device bytes one cell, a column of one unit, takes of the fields a forward reads: a key
/// plane, a value plane and the flag of each, one F32 scale per plane when `quantized`, and
/// `index_bytes` of index keys.
fn cell_bytes(plane_bytes: u64, quantized: bool, index_bytes: u64) -> u64 {
    let scale_bytes = if quantized { 2 * 4 } else { 0 };

    2 * plane_bytes + 2 + scale_bytes + index_bytes
}

/// The element type the cache stores for the engine's `dtype`.
fn stored(dtype: PrefixCacheDtype) -> DType {
    match dtype {
        PrefixCacheDtype::Float16 => DType::F16,
        PrefixCacheDtype::BFloat16 => DType::BF16,
        PrefixCacheDtype::Float32 => DType::F32,
        PrefixCacheDtype::Float8E4m3Fn => DType::F8E4M3,
    }
}

/// The layout of the unit pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheLayout {
    /// Columns of one unit: the greatest common divisor of the groups' layer counts.
    pub columns: u32,
    /// Bytes of one plane: one layer's keys (or values) of one page of the first group.
    pub plane_bytes: u64,
    /// Whether the planes store FP8 with one scale per plane.
    pub quantized: bool,
    /// Device bytes one unit takes over every column: its planes, flags, scales, index keys
    /// and state planes.
    pub unit_bytes: u64,
    /// The stored element type as the engine names it (`PrefixCacheInfo::dtype`).
    pub dtype: PrefixCacheDtype,
    /// The groups, attention groups first, then the draft group, the state group last.
    pub groups: Vec<GroupLayout>,
    /// One checkpoint per recurrent layer, in the state page's plane order.
    pub checkpoints: Vec<Checkpoint>,
    /// The dense representation whose pages the cache loads by converting them
    /// ([`Self::with_conversion`]); `None` for a cache that loads only pages of its own.
    pub converts_from: Option<PrefixCacheDtype>,
}

impl CacheLayout {
    /// The cache of one group of `layers` attention layers: one column per layer, each holding
    /// `block_size` rows of `kv_heads × head_dim` of `dtype`, with a scale per plane when
    /// `dtype` is FP8.
    pub fn dense(
        block_size: u32,
        layers: u32,
        kv_heads: u32,
        head_dim: u32,
        dtype: PrefixCacheDtype,
    ) -> Self {
        let element = stored(dtype);
        let row_bytes = u64::from(kv_heads) * u64::from(head_dim) * element_bytes(element);
        let plane_bytes = row_bytes * u64::from(block_size);

        // Every layer writes its own column, so a unit spans every column: a key plane, a value
        // plane and the two flags of each, and for FP8 one F32 scale per plane.
        let columns = layers.max(1);
        let quantized = matches!(dtype, PrefixCacheDtype::Float8E4m3Fn);

        Self {
            columns,
            plane_bytes,
            quantized,
            unit_bytes: u64::from(columns) * cell_bytes(plane_bytes, quantized, 0),
            dtype,
            groups: vec![GroupLayout {
                kind: CacheGroupKind::Full,
                block_size,
                units_per_page: 1,
                layers: (0..layers).map(|layer| format!("layers.{layer}")).collect(),
                layer_ids: (0..layers).collect(),
                axis_offset: 0,
                axis_layers: layers,
                num_kv_heads: kv_heads,
                total_kv_heads: kv_heads,
                kv_head_offset: 0,
                head_dim,
                dtype: element,
                compute_dtype: DType::BF16,
                index_keys: None,
            }],
            checkpoints: Vec::new(),
            converts_from: None,
        }
    }

    /// The layout whose attention layers keep the index keys `keys` the model states
    /// ([`nn::family::CacheShape::index_keys`]): every cell gains one index-key plane beside
    /// its key and value planes, `block_size / ratio` keys of `width` elements of the layers'
    /// compute type, whatever the cache stores its keys and values in. The plane has no flag
    /// and no scale: a reader reads the keys of completed groups alone, which their writer
    /// stored.
    ///
    /// The draft group stores what the first group's planes hold, so a drafter's layers keep
    /// the same index keys, whether the draft cache is added before or after.
    ///
    /// # Errors
    ///
    /// A layout that keeps index keys already or admits converted loads, which convert keys
    /// and values alone, or index keys the pages do not hold in whole ([`Self::validated`]).
    pub fn with_index_keys(mut self, keys: IndexKeys) -> Result<Self> {
        if self.groups.iter().any(|group| group.index_keys.is_some()) {
            return Err(Error::Invalid(
                "a cache layout states its index keys once".into(),
            ));
        }
        if self.converts_from.is_some() {
            return Err(Error::Invalid(
                "a cache that loads converted pages keeps no index keys".into(),
            ));
        }

        for group in &mut self.groups {
            if !group.state() {
                group.index_keys = Some(keys);
            }
        }
        self.unit_bytes += u64::from(self.columns) * self.index_bytes();

        self.validated()
    }

    /// Bytes of one index-key plane: one layer's index keys of one page of the first group.
    /// Zero for a cache whose layers keep none.
    pub fn index_bytes(&self) -> u64 {
        self.groups
            .first()
            .map_or(0, |group| group.index_bytes(group.block_size))
    }

    /// The shape of the index keys that `pages` pages of `tokens` tokens of one of `group`'s
    /// layers lend, `[pages, tokens / ratio, width]` in the group's compute type
    /// ([`nn::KvPlanes::index_keys`]); `None` for a group that keeps none.
    ///
    /// # Errors
    ///
    /// When a page's keys do not fill an index-key plane, so the pages would not be the
    /// planes of consecutive cells.
    pub(crate) fn index_shape(
        &self,
        group: &GroupLayout,
        pages: usize,
        tokens: u32,
    ) -> Result<Option<[usize; 3]>> {
        let Some(keys) = group.index_keys else {
            return Ok(None);
        };

        let bytes = group.index_bytes(tokens);
        if bytes == 0 || bytes != self.index_bytes() {
            return Err(Error::Invariant(format!(
                "the index keys of a page of {tokens} tokens take {bytes} bytes and an \
                 index-key plane {}",
                self.index_bytes()
            )));
        }
        Ok(Some([
            pages,
            (tokens / keys.ratio) as usize,
            keys.width as usize,
        ]))
    }

    /// The layout with converted loads admitted: pages another host computed in the groups'
    /// dense compute type load into this FP8 cache through `page_conversion` and
    /// `cache_copy_region`, encoded as the cache encodes its own writes.
    ///
    /// Only the direction that does not lower the destination's precision is admitted: a dense
    /// source into an FP8 cache. A cache with a state or draft group admits none, because those
    /// pages have no conversion and a hit needs every group; neither does a cache that keeps
    /// index keys, which a conversion does not carry.
    ///
    /// # Errors
    ///
    /// A cache that is not FP8, holds a state or draft group, keeps index keys, or whose groups
    /// do not compute in one dense element type.
    pub fn with_conversion(mut self) -> Result<Self> {
        if !self.quantized {
            return Err(Error::Invalid(
                "a dense cache loads no converted page: only an FP8 cache converts".into(),
            ));
        }
        if self
            .groups
            .iter()
            .any(|group| group.state() || group.draft())
        {
            return Err(Error::Invalid(
                "a cache with a state or draft group loads no converted page".into(),
            ));
        }
        if self.groups.iter().any(|group| group.index_keys.is_some()) {
            return Err(Error::Invalid(
                "a cache that keeps index keys loads no converted page".into(),
            ));
        }

        let compute = self.groups.first().map(|group| group.compute_dtype);
        let source = match compute {
            Some(DType::BF16) => PrefixCacheDtype::BFloat16,
            Some(DType::F16) => PrefixCacheDtype::Float16,
            other => {
                return Err(Error::Invalid(format!(
                    "a cache that computes in {other:?} loads no converted page"
                )));
            }
        };
        if self
            .groups
            .iter()
            .any(|group| Some(group.compute_dtype) != compute)
        {
            return Err(Error::Invalid(
                "a cache whose groups compute in several element types loads no converted page"
                    .into(),
            ));
        }

        self.converts_from = Some(source);
        Ok(self)
    }

    /// The layout of the cache the converted pages come from: this layout's groups and columns
    /// in the dense source type, as a deployment that stores that type lays them out, so its
    /// pages' shape and bytes are the source cache's. `None` without converted loads.
    pub fn source(&self) -> Option<Self> {
        let source = self.converts_from?;
        if matches!(source, PrefixCacheDtype::Float8E4m3Fn) {
            return None;
        }
        let element = stored(source);
        let stored_bytes = element_bytes(self.groups.first()?.dtype);
        if stored_bytes == 0 {
            return None;
        }

        // A plane is one layer's page, so it grows with the element width; a dense unit holds
        // both fields' planes and their flags per column and no scale.
        let plane_bytes = self.plane_bytes / stored_bytes * element_bytes(element);

        Some(Self {
            columns: self.columns,
            plane_bytes,
            quantized: false,
            unit_bytes: u64::from(self.columns) * cell_bytes(plane_bytes, false, 0),
            dtype: source,
            groups: self
                .groups
                .iter()
                .map(|group| GroupLayout {
                    dtype: element,
                    ..group.clone()
                })
                .collect(),
            checkpoints: Vec::new(),
            converts_from: None,
        })
    }

    /// The layout, once its fields are checked against each other.
    ///
    /// # Errors
    ///
    /// When a group has no layers, a page of a group takes no unit or holds no token, a group
    /// stores an element narrower than a byte, the groups' layer counts do not make the stated
    /// columns, the state group is not the last group, the draft group does not lie over one
    /// unit, a group's index keys are not whole groups of a cell's tokens that fill the
    /// cache's index-key plane, or a unit is smaller than its planes.
    pub fn validated(self) -> Result<Self> {
        if self.groups.is_empty() {
            return Err(Error::Invalid(
                "a cache layout needs at least one group".into(),
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
            if element_bytes(group.dtype) == 0 {
                return Err(Error::Invalid(format!(
                    "cache group {:?} stores {}, which has no whole-byte element",
                    group.layers,
                    ops::abi::name(group.dtype)
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
            self.check_index_keys(group)?;
        }

        if let Some((last, rest)) = self.groups.split_last()
            && rest.iter().any(GroupLayout::state)
        {
            return Err(Error::Invalid(format!(
                "the state group must be the last cache group, which is {:?}",
                last.layers
            )));
        }

        // A draft group's layers lie over the columns of one unit instead of one cell each
        // ([`Self::with_draft`]), so they bound no column count.
        let columns = self
            .groups
            .iter()
            .filter(|group| !group.draft())
            .fold(0, |columns, group| gcd(columns, group.layers.len() as u32));
        if let Some(draft) = self.groups.iter().find(|group| group.draft())
            && (draft.layers.len() as u32 > self.columns || draft.units_per_page != 1)
        {
            return Err(Error::Invalid(format!(
                "the draft group's {} layers lie over the {} columns of one unit per page, not \
                 {}",
                draft.layers.len(),
                self.columns,
                draft.units_per_page
            )));
        }
        if self.columns != columns {
            return Err(Error::Invalid(format!(
                "the cache layout's {} columns disagree with its groups' {columns}",
                self.columns
            )));
        }

        let column_bytes = self.plane_bytes + self.index_bytes();
        if self.plane_bytes == 0 || self.unit_bytes < u64::from(self.columns) * column_bytes {
            return Err(Error::Invalid(
                "a cache unit holds at least one plane and the index keys of every column".into(),
            ));
        }
        Ok(self)
    }

    /// Checks the index keys `group` keeps, when it keeps any: one key per `ratio` tokens of
    /// the tokens a cell holds of one of its layers, in bytes the index-key plane of every cell
    /// of the cache.
    fn check_index_keys(&self, group: &GroupLayout) -> Result<()> {
        let Some(keys) = group.index_keys else {
            return Ok(());
        };
        if group.state() {
            return Err(Error::Invalid(
                "the state group's pages hold checkpoints and no index keys".into(),
            ));
        }

        // A draft layer's page lies over its span of columns, another layer's over one.
        let tokens = if group.draft() {
            group.block_size / span_of(self.columns, group.layers.len() as u32)
        } else {
            group.block_size
        };
        if keys.width == 0 || keys.ratio == 0 || !tokens.is_multiple_of(keys.ratio) {
            return Err(Error::Invalid(format!(
                "cache group {:?} keeps index keys of {} elements per {} tokens, and a cell \
                 holds {tokens} tokens",
                group.layers, keys.width, keys.ratio
            )));
        }

        let bytes = group.index_bytes(tokens);
        if bytes == 0 || bytes != self.index_bytes() {
            return Err(Error::Invalid(format!(
                "cache group {:?} keeps {bytes} bytes of index keys a cell, and the cache's \
                 index-key plane is {} bytes",
                group.layers,
                self.index_bytes()
            )));
        }
        Ok(())
    }

    /// The field runs a state page's checkpoints need; zero without a state group.
    ///
    /// One field run holds `units_per_page · columns` planes of the state group's page, and the
    /// checkpoints' planes run across field runs in order, so the last checkpoint's last plane
    /// decides how many runs the cache holds. A state plane never shares storage with a key or
    /// value plane: every group indexes the same units, so the two at one (column, unit) are
    /// different bytes.
    pub fn state_fields(&self) -> u32 {
        let Some(group) = self.groups.iter().find(|group| group.state()) else {
            return 0;
        };
        let capacity = group.units_per_page.saturating_mul(self.columns);
        if capacity == 0 {
            return 0;
        }
        self.checkpoint_planes().div_ceil(capacity)
    }

    /// The bytes of the snapshot a recurrent-state copy moves a state page through: every
    /// checkpoint plane of the page. It is taken from the device's scratch pool once
    /// ([`StateMemory`](crate::memory::state::StateMemory)); zero for a cache that keeps no
    /// state checkpoints.
    pub fn state_snapshot_bytes(&self) -> u64 {
        if self.state_group().is_none() {
            return 0;
        }
        u64::from(self.checkpoint_planes()) * self.plane_bytes
    }

    /// The planes of every checkpoint of a state page.
    pub fn checkpoint_planes(&self) -> u32 {
        self.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.first_plane.saturating_add(checkpoint.planes))
            .max()
            .unwrap_or(0)
    }

    /// Where each field of recurrent layer `layer`'s checkpoint lies in a state page, in the
    /// order of `fields`: the field's first plane among the page's planes and its bytes, which
    /// fill whole planes from that one ([`Self::hybrid`]).
    pub fn state_planes(&self, layer: u32, fields: &[StateField]) -> Vec<(u32, u64)> {
        let mut plane = self
            .checkpoints
            .get(layer as usize)
            .map_or(0, |checkpoint| checkpoint.first_plane);

        fields
            .iter()
            .map(|field| {
                let at = (plane, field.bytes());
                plane += field.bytes().div_ceil(self.plane_bytes.max(1)) as u32;
                at
            })
            .collect()
    }

    /// The cache as the rank reports it to the engine, for a pool of `num_units` units. The
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
            identities: Vec::new(),
            storage_endpoint: None,
        }
    }

    /// The units a capacity of `tokens` tokens takes: the whole pages of those tokens in every
    /// attention group, and one checkpoint of the state group per request slot of
    /// `request_slots`, the latest each running request keeps. The draft group's pages take
    /// units the capacity holds as far as they are free.
    pub fn token_units(&self, tokens: u64, request_slots: u64) -> u64 {
        self.groups
            .iter()
            .filter(|group| !group.draft())
            .map(|group| {
                let pages = if group.state() {
                    request_slots
                } else {
                    tokens / u64::from(group.block_size)
                };
                pages * u64::from(group.units_per_page)
            })
            .sum()
    }

    /// Planes of one field one page of `group` spans: every column of every unit of the page.
    pub(crate) fn page_planes(&self, group: &GroupLayout) -> u64 {
        u64::from(group.units_per_page) * u64::from(self.columns)
    }

    /// The cache of a model of `layers` attention layers and the recurrent state `runs`, the
    /// runs of recurrent layers that keep the same fields in the model's order of them
    /// (`nn::family::TextModel::state_layers`): one attention group as [`Self::dense`] lays it
    /// out, and the state group after it, whose page holds every recurrent layer's checkpoint
    /// in that order.
    ///
    /// The columns are the common divisor of the attention layers and the recurrent layers of
    /// every run. A checkpoint lays its layer's fields out in field order, each from a plane
    /// boundary over the whole planes its bytes fill ([`Self::state_planes`]), and the
    /// checkpoints follow each other; the state page spans as many units as those planes fill
    /// in one field run, so a unit carries one state plane per column beside its key and value
    /// planes. A checkpoint keeps one slot of every field, a stepwise one's included.
    ///
    /// # Errors
    ///
    /// No attention layer, no recurrent layer, a run whose layers keep no field, or a field of
    /// no bytes.
    pub fn hybrid(
        block_size: u32,
        layers: u32,
        kv_heads: u32,
        head_dim: u32,
        dtype: PrefixCacheDtype,
        runs: &[StateShape],
    ) -> Result<Self> {
        let recurrent: u32 = runs.iter().map(|run| run.layers).sum();
        let fieldless = runs
            .iter()
            .any(|run| run.layers > 0 && run.fields.is_empty());
        if layers == 0 || recurrent == 0 || fieldless {
            return Err(Error::Invalid(format!(
                "a hybrid cache of {layers} attention layers and {recurrent} recurrent layers, \
                 each of which keeps a field"
            )));
        }

        let mut layout = Self::dense(block_size, layers, kv_heads, head_dim, dtype);
        let columns = gcd(layers, recurrent);
        let per_column = layout.unit_bytes / u64::from(layout.columns);
        layout.columns = columns;
        layout.groups[0].units_per_page = layers / columns;

        // Every field starts at a plane boundary, so a copy moves whole planes, and every
        // layer's checkpoint starts where the one before it ends.
        let mut planes = 0_u64;
        for fields in layer_fields(runs) {
            let mut layer_planes = 0_u64;
            for field in fields {
                if field.bytes() == 0 {
                    return Err(Error::Invalid(format!(
                        "the recurrent state's field {} holds no byte",
                        field.name
                    )));
                }
                layer_planes += field.bytes().div_ceil(layout.plane_bytes);
            }

            let (first_plane, planes_held) = u32::try_from(planes)
                .ok()
                .zip(u32::try_from(layer_planes).ok())
                .ok_or_else(|| {
                    Error::Invalid("a checkpoint fills more planes than a page counts".into())
                })?;
            layout.checkpoints.push(Checkpoint {
                first_plane,
                planes: planes_held,
                fields: Vec::new(),
            });
            planes += layer_planes;
        }

        let units_per_page = u32::try_from(planes.div_ceil(u64::from(columns))).map_err(|_| {
            Error::Invalid("a checkpoint fills more units than a page counts".into())
        })?;
        layout.groups.push(GroupLayout {
            kind: CacheGroupKind::State,
            block_size,
            units_per_page,
            layers: (0..recurrent)
                .map(|layer| format!("recurrent.{layer}"))
                .collect(),
            layer_ids: (layers..layers + recurrent).collect(),
            axis_offset: 0,
            axis_layers: recurrent,
            // A state page holds bytes: one head of one element per layer.
            num_kv_heads: 1,
            total_kv_heads: 1,
            kv_head_offset: 0,
            head_dim: 1,
            dtype: DType::BF16,
            compute_dtype: DType::BF16,
            index_keys: None,
        });

        // A unit holds its key and value planes and flags per column, and one state plane per
        // column of each field run its page's checkpoints span.
        layout.unit_bytes = u64::from(columns) * per_column
            + u64::from(layout.state_fields()) * u64::from(columns) * layout.plane_bytes;
        layout.validated()
    }

    /// The layout with the draft cache added: the keys and values of a drafter's `layers`
    /// attention layers, which store `kv_heads` heads of `head_dim` as the first group does.
    ///
    /// The draft's layers keep the index keys the first group's keep, in the same cells.
    ///
    /// A drafter has fewer layers than a unit has columns, so a draft page does not take a cell
    /// per layer. It takes **one unit**, and each of its layers lies over
    /// [`Self::draft_span`] of the unit's columns: column `c` of the layer's run holds the
    /// page's `c`-th run of `block_size / span` tokens. A draft page therefore holds `span`
    /// times the tokens of an attention page in one unit. The group sits before the state
    /// group, which stays last.
    ///
    /// # Errors
    ///
    /// A drafter of more layers than the cache has columns, heads the cache's planes do not
    /// hold, or a layout the draft group makes invalid.
    pub fn with_draft(mut self, layers: u32, kv_heads: u32, head_dim: u32) -> Result<Self> {
        let first = self
            .groups
            .first()
            .cloned()
            .ok_or_else(|| Error::Invalid("a draft cache beside no attention group".into()))?;
        if layers == 0 || layers > self.columns {
            return Err(Error::Invalid(format!(
                "a drafter of {layers} layers over a cache of {} columns",
                self.columns
            )));
        }
        if kv_heads != first.num_kv_heads || head_dim != first.head_dim {
            return Err(Error::Invalid(format!(
                "a drafter of {kv_heads} heads of {head_dim} over planes of {} heads of {}",
                first.num_kv_heads, first.head_dim
            )));
        }

        let span = span_of(self.columns, layers);
        let ids = self
            .groups
            .iter()
            .flat_map(|group| group.layer_ids.iter().copied())
            .max()
            .map_or(0, |id| id + 1);
        let draft = GroupLayout {
            kind: CacheGroupKind::Draft,
            block_size: first.block_size * span,
            units_per_page: 1,
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
        Some(span_of(self.columns, draft.layers.len() as u32))
    }

    /// The draft group's index, when the layout holds one.
    pub fn draft_group(&self) -> Option<usize> {
        self.groups.iter().position(GroupLayout::draft)
    }

    /// The state group's index, when the layout holds one.
    pub fn state_group(&self) -> Option<usize> {
        self.groups.iter().position(GroupLayout::state)
    }

    /// How many attention groups the layout holds, which the model's attention layers index:
    /// they come first. The draft cache and the state group are none of them.
    pub fn attention_groups(&self) -> usize {
        self.groups.iter().filter(|group| group.attention()).count()
    }
}

/// The columns of a unit each of `layers` draft layers lies over: the largest power of two the
/// columns hold `layers` runs of, so a draft page's tokens are a power of two.
fn span_of(columns: u32, layers: u32) -> u32 {
    let share = (columns / layers.max(1)).max(1);
    1 << (u32::BITS - 1 - share.leading_zeros())
}

/// The greatest common divisor of two counts; `gcd(0, n) == n`.
fn gcd(left: u32, right: u32) -> u32 {
    if right == 0 {
        left
    } else {
        gcd(right, left % right)
    }
}

#[cfg(test)]
mod tests {
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
                },
                StateField {
                    name: "recurrent",
                    dtype: DType::F32,
                    dims: vec![32, 128, 128],
                    stepwise: false,
                },
            ],
        }
    }

    /// A state page's checkpoints need one field run per `units_per_page · columns` planes.
    #[test]
    fn a_state_page_asks_for_the_field_runs_its_checkpoints_span() {
        let layout = |state: bool, checkpoints: Vec<(u32, u32)>| {
            let mut layout = CacheLayout::dense(64, 2, 1, 8, PrefixCacheDtype::BFloat16);
            if state {
                layout.groups[0].kind = CacheGroupKind::State;
            }
            layout.checkpoints = checkpoints
                .into_iter()
                .map(|(first_plane, planes)| Checkpoint {
                    first_plane,
                    planes,
                    fields: Vec::new(),
                })
                .collect();
            layout
        };

        // Two columns and one unit per page: a field run holds two planes, so four need two.
        assert_eq!(layout(true, vec![(0, 4)]).state_fields(), 2);
        // A checkpoint that fills one field run exactly asks for one.
        assert_eq!(layout(true, vec![(0, 2)]).state_fields(), 1);
        // A cache with no state group asks for none.
        assert_eq!(layout(false, vec![(0, 4)]).state_fields(), 0);
        assert_eq!(layout(true, Vec::new()).state_fields(), 0);
    }

    /// A hybrid cache of ten attention layers and thirty recurrent ones lays every recurrent
    /// layer's checkpoint in whole planes of one field run, and a unit pays one state plane per
    /// column beside its key and value planes.
    #[test]
    fn a_hybrid_cache_lays_each_checkpoint_field_from_a_plane_boundary() -> Result<()> {
        let state = state();
        let layout = CacheLayout::hybrid(
            64,
            10,
            2,
            256,
            PrefixCacheDtype::BFloat16,
            std::slice::from_ref(&state),
        )?;

        // A key plane is 64 tokens of two 256-wide BF16 heads: 64 KiB.
        assert_eq!(layout.plane_bytes, 64 << 10);
        assert_eq!(layout.columns, 10);
        assert_eq!(layout.groups.len(), 2);
        assert_eq!(layout.state_group(), Some(1));
        assert_eq!(layout.attention_groups(), 1);

        // 48 KiB of convolution window take one plane, 2 MiB of state thirty-two:
        // thirty-three per layer, 990 for the page, which 99 units of ten columns hold in one
        // field run.
        assert_eq!(layout.checkpoint_planes(), 990);
        assert_eq!(layout.groups[1].units_per_page, 99);
        assert_eq!(layout.state_fields(), 1);
        assert_eq!(
            layout.state_planes(2, &state.fields),
            vec![(66, 48 << 10), (67, 2 << 20)]
        );

        let dense = CacheLayout::dense(64, 10, 2, 256, PrefixCacheDtype::BFloat16);
        assert_eq!(
            layout.unit_bytes,
            dense.unit_bytes + 10 * layout.plane_bytes
        );
        Ok(())
    }

    /// A drafter of one layer lies over eight of a unit's ten columns: its pages hold 512
    /// tokens in one unit, before the state group, and leave the cache's columns and units as
    /// they were.
    #[test]
    fn a_draft_group_lies_over_the_columns_of_one_unit() -> Result<()> {
        let hybrid = || CacheLayout::hybrid(64, 10, 2, 256, PrefixCacheDtype::BFloat16, &[state()]);
        let unit_bytes = hybrid()?.unit_bytes;

        let drafting = hybrid()?.with_draft(1, 2, 256)?;
        assert_eq!(drafting.columns, 10);
        assert_eq!(drafting.unit_bytes, unit_bytes);
        assert_eq!(drafting.draft_group(), Some(1));
        assert_eq!(drafting.state_group(), Some(2));
        assert_eq!(drafting.draft_span(), Some(8));

        let draft = &drafting.groups[1];
        assert_eq!((draft.block_size, draft.units_per_page), (512, 1));
        assert_eq!(draft.layer_ids, vec![40]);

        // Two layers lie over four columns each; more layers than columns lie over none.
        assert_eq!(hybrid()?.with_draft(2, 2, 256)?.draft_span(), Some(4));
        assert!(hybrid()?.with_draft(11, 2, 256).is_err());
        // A drafter stores the heads the cache's planes hold.
        assert!(hybrid()?.with_draft(1, 4, 256).is_err());
        Ok(())
    }

    /// A dense cache has one column per layer, and a unit spans every column: each column's key
    /// and value plane and its two flags.
    #[test]
    fn a_dense_cache_spans_every_column_in_its_unit() -> Result<()> {
        let (layers, block_size, kv_heads, head_dim) = (48, 16, 4, 128);
        let layout = CacheLayout::dense(
            block_size,
            layers,
            kv_heads,
            head_dim,
            PrefixCacheDtype::BFloat16,
        )
        .validated()?;

        let plane = u64::from(kv_heads) * u64::from(head_dim) * 2 * u64::from(block_size);
        assert_eq!(layout.columns, layers);
        assert_eq!(layout.plane_bytes, plane);
        assert_eq!(layout.unit_bytes, u64::from(layers) * (2 * plane + 2));

        // An FP8 cache stores one byte per element and one F32 scale per plane.
        let fp8 = CacheLayout::dense(
            block_size,
            layers,
            kv_heads,
            head_dim,
            PrefixCacheDtype::Float8E4m3Fn,
        )
        .validated()?;
        assert_eq!(fp8.plane_bytes, plane / 2);
        assert_eq!(fp8.unit_bytes, u64::from(layers) * (plane + 2 + 8));
        Ok(())
    }

    /// Index keys of 128 elements per four tokens, as a model's cache shape states them.
    const KEYS: IndexKeys = IndexKeys {
        width: 128,
        ratio: 4,
    };

    /// A cache whose layers keep index keys holds one more plane per cell: sixteen keys of 128
    /// BF16 elements for a page of 64 tokens, whatever the cache stores its keys and values in,
    /// and nothing else of the layout moves.
    #[test]
    fn index_keys_add_one_plane_to_every_cell() -> Result<()> {
        let dense = |dtype| CacheLayout::dense(64, 10, 2, 256, dtype);
        let index_plane = 16 * 128 * 2;

        for dtype in [PrefixCacheDtype::BFloat16, PrefixCacheDtype::Float8E4m3Fn] {
            let without = dense(dtype).validated()?;
            assert_eq!(without.index_bytes(), 0);
            assert_eq!(without.groups[0].index_keys, None);

            let keyed = dense(dtype).with_index_keys(KEYS)?;
            assert_eq!(keyed.index_bytes(), index_plane);
            assert_eq!(keyed.groups[0].index_keys, Some(KEYS));
            assert_eq!(keyed.unit_bytes, without.unit_bytes + 10 * index_plane);
            assert_eq!(
                (keyed.columns, keyed.plane_bytes, keyed.quantized),
                (without.columns, without.plane_bytes, without.quantized)
            );
        }

        // The state group's pages hold checkpoints alone, in the planes they had.
        let state = state();
        let hybrid = CacheLayout::hybrid(
            64,
            10,
            2,
            256,
            PrefixCacheDtype::BFloat16,
            std::slice::from_ref(&state),
        )?;
        let keyed = hybrid.clone().with_index_keys(KEYS)?;
        assert_eq!(keyed.unit_bytes, hybrid.unit_bytes + 10 * index_plane);
        assert_eq!(keyed.groups[1].index_keys, None);
        assert_eq!(
            keyed.state_planes(2, &state.fields),
            hybrid.state_planes(2, &state.fields)
        );
        assert_eq!(keyed.token_units(640, 4), hybrid.token_units(640, 4));
        Ok(())
    }

    /// A drafter's layers keep the index keys of the cache's cells, whichever of the draft
    /// cache and the index keys the layout gains first.
    #[test]
    fn a_draft_group_keeps_the_index_keys_of_its_cells() -> Result<()> {
        let hybrid = || CacheLayout::hybrid(64, 10, 2, 256, PrefixCacheDtype::BFloat16, &[state()]);

        let keyed_first = hybrid()?.with_index_keys(KEYS)?.with_draft(1, 2, 256)?;
        let draft_first = hybrid()?.with_draft(1, 2, 256)?.with_index_keys(KEYS)?;
        assert_eq!(keyed_first, draft_first);

        let draft = &keyed_first.groups[1];
        assert!(draft.draft());
        assert_eq!(draft.index_keys, Some(KEYS));
        assert_eq!(
            keyed_first.unit_bytes,
            hybrid()?.with_index_keys(KEYS)?.unit_bytes
        );
        Ok(())
    }

    /// Index keys are whole groups of a page's tokens, stated once, and a cache that keeps
    /// them loads no converted page.
    #[test]
    fn index_keys_a_page_does_not_hold_are_refused() -> Result<()> {
        let dense = |dtype| CacheLayout::dense(64, 4, 2, 128, dtype);
        let bf16 = || dense(PrefixCacheDtype::BFloat16);
        let keys = |width, ratio| IndexKeys { width, ratio };

        // Five tokens a key do not divide a page of 64, and a key has elements.
        assert!(bf16().with_index_keys(keys(128, 5)).is_err());
        assert!(bf16().with_index_keys(keys(128, 0)).is_err());
        assert!(bf16().with_index_keys(keys(0, 4)).is_err());
        assert!(bf16().with_index_keys(KEYS)?.with_index_keys(KEYS).is_err());

        let fp8 = || dense(PrefixCacheDtype::Float8E4m3Fn);
        assert!(fp8().with_index_keys(KEYS)?.with_conversion().is_err());
        assert!(fp8().with_conversion()?.with_index_keys(KEYS).is_err());
        Ok(())
    }

    /// The layout's own fields are checked against its groups.
    #[test]
    fn a_layout_whose_fields_disagree_is_refused() {
        let dense = || CacheLayout::dense(16, 4, 2, 8, PrefixCacheDtype::BFloat16);
        assert!(dense().validated().is_ok());

        let mut columns = dense();
        columns.columns = 3;
        assert!(columns.validated().is_err());

        let mut heads = dense();
        heads.groups[0].kv_head_offset = 1;
        assert!(heads.validated().is_err());

        let mut units = dense();
        units.groups[0].units_per_page = 0;
        assert!(units.validated().is_err());

        let mut ids = dense();
        ids.groups[0].layer_ids.pop();
        assert!(ids.validated().is_err());

        let mut small = dense();
        small.unit_bytes = small.plane_bytes;
        assert!(small.validated().is_err());

        let mut none = dense();
        none.groups.clear();
        assert!(none.validated().is_err());
    }

    /// Only an FP8 cache of attention groups converts, from its groups' dense compute type, and
    /// the source layout is the one a deployment storing that type lays out.
    #[test]
    fn an_fp8_cache_converts_from_its_compute_type() -> Result<()> {
        let fp8 = CacheLayout::dense(64, 4, 2, 128, PrefixCacheDtype::Float8E4m3Fn)
            .validated()?
            .with_conversion()?;
        let bf16 = CacheLayout::dense(64, 4, 2, 128, PrefixCacheDtype::BFloat16).validated()?;

        assert_eq!(fp8.converts_from, Some(PrefixCacheDtype::BFloat16));
        assert_eq!(fp8.source(), Some(bf16.clone()));

        // The other direction lowers the destination's precision.
        assert!(bf16.clone().with_conversion().is_err());
        assert!(bf16.source().is_none());
        Ok(())
    }

    /// A token capacity takes whole pages of every attention group and one state page per
    /// request slot.
    #[test]
    fn a_token_capacity_counts_pages_and_one_checkpoint_per_slot() -> Result<()> {
        let layout = CacheLayout::hybrid(64, 10, 2, 256, PrefixCacheDtype::BFloat16, &[state()])?;

        // 640 tokens are ten attention pages of one unit; four slots keep four state pages of
        // 99 units.
        assert_eq!(layout.token_units(640, 4), 10 + 4 * 99);
        Ok(())
    }
}
