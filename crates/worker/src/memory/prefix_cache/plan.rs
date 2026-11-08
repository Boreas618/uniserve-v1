//! The cache unit pool's layout, as the model's cache layers describe it.
//!
//! The pool is one allocation unit per logical page of the widest resident
//! layer: a *unit* is one allocation unit of the pool (unit 0 is the padding
//! sentinel); a *column* is a layer's position inside a unit
//! (`columns = gcd(layers per group)`); a *plane* is the bytes of one
//! (field, column) of one unit; a group's logical page of `block_size` tokens
//! occupies `units_per_page` units.
//!
//! The model owns the geometry of its cache layers, so it supplies the plan:
//! [`CachePlan::validated`] checks the description and derives nothing hidden
//! from it. The worker owns the pool the plan describes: its size, its
//! Storage Server keys and its resets.

use ipc::PrefixCacheInfo;
use ipc::{CacheGroup, CacheGroupKind, PrefixCacheDtype};

use crate::memory::GroupShape;
use crate::memory::allocation::ElementType;
use crate::memory::request_slots::SlotField;
/// One field of a cache layer kept per request slot: `name` is
/// `layer/field`, `shape` one slot's extent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateField {
    pub name: String,
    pub dtype: ElementType,
    pub shape: Vec<i64>,
}

/// One layer's checkpoint, packed into consecutive planes of a state page.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct Checkpoint {
    pub first_plane: u32,
    pub planes: u32,
    pub fields: Vec<::storage::manifest::Field>,
}

impl StateField {
    /// The field as a device bank of the request slots.
    pub(crate) fn slot(&self) -> SlotField {
        SlotField {
            name: self.name.clone(),
            dtype: self.dtype,
            shape: self.shape.clone(),
            host: false,
        }
    }
}
use crate::{Error, Result};

/// Where this rank's layers of a group lie in the group's transfer layer
/// axis: a transfer tensor spans `total` layers in global cache-layer order,
/// and this rank's layers are the consecutive run starting at `offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupAxis {
    pub offset: u32,
    pub total: u32,
}

/// One cache group: layers sharing a history window and a K/V page shape,
/// or the state group, whose pages hold the checkpoints of the layers kept
/// per request slot (`state`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheGroupPlan {
    pub window: Option<u32>,
    /// Whether the group's pages hold state checkpoints: each of its layers'
    /// state after the page's last token, in bytes the library lays out.
    pub state: bool,
    /// Whether the group holds the draft cache, which a request may lack: a
    /// drafter's K/V, each layer's page laid over the columns of its unit
    /// (`uniserve.prefix_cache.units`).
    pub draft: bool,
    pub block_size: u32,
    pub units_per_page: u32,
    /// Resident layer names in column order: layer `k` lies in column
    /// `k % columns` of a page's unit `k / columns`.
    pub layers: Vec<String>,
    /// Global cache-layer ids of the resident layers.
    pub layer_ids: Vec<u32>,
    pub axis: GroupAxis,
    /// KV heads stored per layer on this rank.
    pub num_kv_heads: u32,
    pub total_kv_heads: u32,
    pub kv_head_offset: u32,
    pub head_dim: u32,
    /// Stored element type of the group's planes.
    pub dtype: ElementType,
    pub compute_dtype: ElementType,
}

impl CacheGroupPlan {
    /// Bytes one token of one layer's key (or value) occupies.
    pub(crate) fn row_bytes(&self) -> u64 {
        u64::from(self.num_kv_heads) * u64::from(self.head_dim) * self.dtype.bytes() as u64
    }

    pub(crate) fn shape(&self) -> Result<GroupShape> {
        // A state checkpoint's reader continuing from token `t` reads the
        // page ending at `t`, the page holding token `t - 1`: the pages a
        // one-token window reads.
        let window = if self.state { Some(1) } else { self.window };
        GroupShape::new(self.block_size, self.units_per_page, window)
    }

    /// Whether the group holds K/V rows a request needs: neither the state
    /// group's checkpoints nor the optional draft cache.
    pub(crate) fn required_rows(&self) -> bool {
        !self.state && !self.draft
    }

    /// The group as the engine sees it.
    fn kind(&self) -> CacheGroupKind {
        if self.state {
            return CacheGroupKind::State;
        }
        if self.draft {
            return CacheGroupKind::Draft;
        }
        self.window.map_or(CacheGroupKind::Full, |window| {
            CacheGroupKind::SlidingWindow { window }
        })
    }
}

/// The unit pool's layout.
pub struct CachePlan {
    pub columns: u32,
    pub plane_bytes: u64,
    pub quantized: bool,
    pub unit_bytes: u64,
    /// The published stored dtype (`PrefixCacheInfo::dtype`).
    pub dtype: PrefixCacheDtype,
    pub groups: Vec<CacheGroupPlan>,
    /// The fields of the layers kept per request slot (`state_fields`).
    pub states: Vec<StateField>,
    pub checkpoints: Vec<Checkpoint>,
}

impl CachePlan {
    /// The pool one group of `layers` dense layers describes: one column holding
    /// `layers` rows of `kv_heads x head_dim` of `dtype`, a scale per cell when
    /// `dtype` is a quantized one, no state fields and no checkpoints.
    ///
    /// A caller that has no model geometry yet - the mount's capacity path - gets
    /// the smallest honest pool from it instead of inventing fields.
    pub fn dense(
        block_size: u32,
        layers: u32,
        kv_heads: u32,
        head_dim: u32,
        dtype: PrefixCacheDtype,
    ) -> Self {
        let element = match dtype {
            PrefixCacheDtype::Float16 | PrefixCacheDtype::BFloat16 => 2,
            PrefixCacheDtype::Float32 => 4,
            PrefixCacheDtype::Float8E4m3Fn => 1,
        };
        let row_bytes = u64::from(kv_heads) * u64::from(head_dim) * element;
        let plane_bytes = row_bytes * u64::from(block_size);
        // One column per layer: every layer writes its own column's planes, so
        // a unit spans EVERY column - its key and value plane and its two
        // initialized flags - and `CachePlanes` addresses a cell as
        // `column * num_units + unit` with `columns * num_units` planes per
        // field. A unit's bytes therefore grow with the layer count, and the
        // flag bytes are per column too (`2 * columns`), because each field's
        // flag plane has one byte per cell.
        let columns = layers.max(1);
        // A quantized page reads through one f32 scale per cell and field, and a
        // cell is one unit of one column, so the scale bytes grow as the flags do.
        let quantized = matches!(dtype, PrefixCacheDtype::Float8E4m3Fn);
        let scale_bytes = if quantized { 2 * 4 } else { 0 };
        let per_column = 2 * plane_bytes + 2 + scale_bytes;
        Self {
            columns,
            plane_bytes,
            quantized,
            // The initialized flags live here and nowhere else: `fit_cache`'s
            // `unit_overhead` is the unit index columns, a different cost.
            unit_bytes: u64::from(columns) * per_column,
            dtype,
            groups: vec![CacheGroupPlan {
                window: None,
                state: false,
                draft: false,
                block_size,
                units_per_page: 1,
                layers: (0..layers).map(|layer| format!("layers.{layer}")).collect(),
                layer_ids: (0..layers).collect(),
                axis: GroupAxis {
                    offset: 0,
                    total: layers,
                },
                num_kv_heads: kv_heads,
                total_kv_heads: kv_heads,
                kv_head_offset: 0,
                head_dim,
                dtype: match dtype {
                    PrefixCacheDtype::Float16 => ElementType::F16,
                    PrefixCacheDtype::Float32 => ElementType::F32,
                    PrefixCacheDtype::Float8E4m3Fn => ElementType::F8E4M3,
                    PrefixCacheDtype::BFloat16 => ElementType::BF16,
                },
                compute_dtype: ElementType::BF16,
            }],
            states: Vec::new(),
            checkpoints: Vec::new(),
        }
    }
}

impl CachePlan {
    /// Bytes one unit's state snapshot occupies: its checkpoint blocks plus the index vector
    /// the copies select them by (design 4.5).
    ///
    /// A snapshot is Scratch, reserved once at pool construction and never allocated during a
    /// step, so the figure is stated where the pools are planned: it is what the scratch pool has
    /// to hold beside the shadow region. Zero for a model that keeps no state per unit, which is
    /// what makes a deployment without one pay nothing.
    pub fn state_snapshot_bytes(&self) -> u64 {
        let rows: u32 = self
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.planes)
            .sum();
        if rows == 0 || !self.groups.iter().any(|group| group.state) {
            return 0;
        }
        u64::from(rows) * self.plane_bytes + u64::from(rows) * 8
    }

    /// The unit pool a model describes, once its fields are checked against each other.
    ///
    /// # Errors
    ///
    /// Fails when a group has no layers, a logical page of a group takes no
    /// unit or more units than its block size allows, the groups' layer
    /// counts share no column count, the state group is not the last group,
    /// or a field of the pool is inconsistent with its groups.
    pub fn validated(self) -> Result<Self> {
        let Self {
            columns,
            plane_bytes,
            quantized,
            unit_bytes,
            dtype,
            groups,
            states,
            checkpoints,
        } = self;
        if groups.is_empty() {
            return Err(Error::Invalid(
                "the cache plan needs at least one resident group".into(),
            ));
        }
        for group in &groups {
            if group.layers.is_empty() || group.layers.len() != group.layer_ids.len() {
                return Err(Error::Invalid(format!(
                    "cache group {:?} needs one global id per resident layer",
                    group.layers
                )));
            }
            if group.units_per_page == 0 || group.block_size == 0 {
                return Err(Error::Invalid(format!(
                    "cache group {:?} needs a page of one block size and one unit",
                    group.layers
                )));
            }
            if group.kv_head_offset + group.num_kv_heads > group.total_kv_heads {
                return Err(Error::Invalid(format!(
                    "cache group {:?} stores heads it does not have",
                    group.layers
                )));
            }
            if group.axis.offset + group.layers.len() as u32 > group.axis.total {
                return Err(Error::Invalid(format!(
                    "cache group {:?} lies outside its transfer layer axis",
                    group.layers
                )));
            }
        }
        if let Some((last, rest)) = groups.split_last()
            && rest.iter().any(|group| group.state)
        {
            return Err(Error::Invalid(format!(
                "the state group {:?} must be the last cache group",
                last.layers
            )));
        }
        let columns_of_groups = groups
            .iter()
            .fold(0, |columns, group| gcd(columns, group.layers.len() as u32));
        if columns != columns_of_groups {
            return Err(Error::Invalid(format!(
                "the cache plan's {columns} columns disagree with its groups' \
                 {columns_of_groups}"
            )));
        }
        if plane_bytes == 0 || unit_bytes < u64::from(columns) * plane_bytes {
            return Err(Error::Invalid(
                "a cache unit holds at least one plane of every column".into(),
            ));
        }
        Ok(Self {
            columns,
            plane_bytes,
            quantized,
            unit_bytes,
            dtype,
            groups,
            states,
            checkpoints,
        })
    }

    /// The field runs a state page's checkpoints need, or zero when the plan
    /// holds no state group.
    ///
    /// One field storage holds `units_per_page * columns` planes of the state
    /// group's page - the cells one run of that page covers - and the
    /// checkpoints' plane positions run across fields in order, so the last
    /// checkpoint's last plane decides how many runs the pool must hold. A
    /// state plane never shares a field with a K/V plane: every group of the
    /// pool indexes the same units, so a K/V plane and a state plane at one
    /// (column, unit) are different bytes of the page.
    pub fn state_fields(&self) -> u32 {
        let Some(group) = self.groups.iter().find(|group| group.state) else {
            return 0;
        };
        let capacity = group.units_per_page.saturating_mul(self.columns);
        if capacity == 0 {
            return 0;
        }
        let last = self
            .checkpoints
            .iter()
            .map(|checkpoint| checkpoint.first_plane.saturating_add(checkpoint.planes))
            .max()
            .unwrap_or(0);
        last.div_ceil(capacity)
    }

    /// Whether groups store different element types: stale bytes of one type
    /// need not decode as finite values of another, so recycled units are
    /// zeroed completely.
    pub(crate) fn mixed_types(&self) -> bool {
        self.groups
            .iter()
            .any(|group| group.dtype != self.groups[0].dtype)
    }

    /// The largest page of any group: a unit's whole token range.
    pub(crate) fn unit_tokens(&self) -> u32 {
        self.groups
            .iter()
            .map(|group| group.block_size)
            .max()
            .unwrap_or(0)
    }

    pub(crate) fn group_shapes(&self) -> Result<Vec<GroupShape>> {
        self.groups.iter().map(CacheGroupPlan::shape).collect()
    }

    /// The compute dtype cache exports declare.
    pub(crate) fn compute_dtype(&self) -> ElementType {
        self.groups[0].compute_dtype
    }

    /// The pool description reported to the engine.
    pub fn info(&self, num_units: u32) -> PrefixCacheInfo {
        PrefixCacheInfo {
            num_units,
            dtype: self.dtype,
            groups: self
                .groups
                .iter()
                .map(|group| CacheGroup {
                    kind: group.kind(),
                    block_size: group.block_size,
                    units_per_page: group.units_per_page,
                    layer_ids: group.layer_ids.clone(),
                    num_kv_heads: group.num_kv_heads,
                    total_kv_heads: group.total_kv_heads,
                    kv_head_offset: group.kv_head_offset,
                    head_dim: group.head_dim,
                })
                .collect(),
            // The rank reports its page pools once it mapped them.
            page_pools: Vec::new(),
            // The worker reports the units its warmup lent once it warmed up.
            scratch_units: 0,
            segments: Vec::new(),
            held_segments: 0,
            inherited_start: 0,
            inherited_units: 0,
            // The worker reports its identity once it loaded the model.
            identities: Vec::new(),
            // The rank reports what its host KV cache takes once the engine
            // copies its pages through one.
            host_token_bytes: 0,
            storage_endpoint: None,
        }
    }

    /// Whether the model keeps layers per request slot, whose checkpoints
    /// fill the state group.
    pub(crate) fn states(&self) -> bool {
        self.groups.iter().any(|group| group.state)
    }

    /// Writable units needed to advance one request: one page of every
    /// K/V group and two state checkpoints. The input checkpoint stays
    /// readable until the call writing its successor has completed.
    pub(crate) fn progress_units(&self) -> u32 {
        self.groups
            .iter()
            .map(|group| group.units_per_page * if group.state { 2 } else { 1 })
            .sum()
    }

    /// The units a token capacity of `tokens` takes: the whole pages of
    /// those tokens in every K/V group, and one checkpoint of the state
    /// group per request slot of `request_slots`, the latest each running
    /// request keeps. The draft group's pages, optional state of those same
    /// tokens, take units the capacity holds as far as they are free.
    pub(crate) fn token_units(&self, tokens: u64, request_slots: u64) -> u64 {
        self.groups
            .iter()
            .filter(|group| !group.draft)
            .map(|group| {
                let pages = if group.state {
                    request_slots
                } else {
                    tokens / u64::from(group.block_size)
                };
                pages * u64::from(group.units_per_page)
            })
            .sum()
    }

    /// Planes of one field one page of `group` spans: every column of every
    /// unit of the page. A K/V group's page holds one plane per layer and
    /// field (its layers fill whole units); the state group's checkpoints
    /// fill the planes of its units in page order.
    pub(crate) fn page_planes(&self, group: &CacheGroupPlan) -> u64 {
        u64::from(group.units_per_page) * u64::from(self.columns)
    }

    /// Elements `E` and scale entries `S` of the largest K/V group page,
    /// which size a cache import workspace: `E = block_size * layers *
    /// heads * dim` and `S = block_size * layers * heads`, maximized over
    /// groups. Imports carry the K/V pages a request needs alone.
    pub(crate) fn page_extents(&self) -> (u64, u64) {
        self.groups
            .iter()
            .filter(|group| group.required_rows())
            .fold((0, 0), |(elements, scales), group| {
                let rows = u64::from(group.block_size)
                    * group.layers.len() as u64
                    * u64::from(group.num_kv_heads);
                (
                    elements.max(rows * u64::from(group.head_dim)),
                    scales.max(rows),
                )
            })
    }
}

/// The request state fields of a plan's `state_fields`: a JSON list of
/// `{"name", "dtype", "shape"}`, `shape` one slot's extent.
fn state_fields(text: &str) -> Result<Vec<StateField>> {
    #[derive(serde::Deserialize)]
    struct Field {
        name: String,
        dtype: String,
        shape: Vec<i64>,
    }
    let fields: Vec<Field> = serde_json::from_str(text)
        .map_err(|error| Error::Interface(format!("the cache plan's state fields: {error}")))?;
    fields
        .into_iter()
        .map(|field| {
            Ok(StateField {
                dtype: ElementType::parse(&field.dtype)?,
                name: field.name,
                shape: field.shape,
            })
        })
        .collect()
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

    /// A state page's checkpoints need one field storage per `units_per_page * columns` planes.
    #[test]
    fn a_state_page_asks_for_the_fields_its_checkpoints_span() {
        let group = |state: bool| CacheGroupPlan {
            window: None,
            state,
            draft: false,
            block_size: 64,
            units_per_page: 1,
            layers: vec!["layers.0".into(), "layers.1".into()],
            layer_ids: vec![0, 1],
            axis: GroupAxis {
                offset: 0,
                total: 2,
            },
            num_kv_heads: 1,
            total_kv_heads: 1,
            kv_head_offset: 0,
            head_dim: 8,
            dtype: ElementType::BF16,
            compute_dtype: ElementType::BF16,
        };
        let plan = |state: bool, checkpoints: Vec<Checkpoint>| CachePlan {
            columns: 2,
            plane_bytes: 16,
            quantized: false,
            unit_bytes: 64,
            dtype: PrefixCacheDtype::BFloat16,
            groups: vec![group(state)],
            states: Vec::new(),
            checkpoints,
        };
        let checkpoint = |first_plane: u32, planes: u32| Checkpoint {
            first_plane,
            planes,
            fields: Vec::new(),
        };
        // Two columns and one unit per page: a field holds two planes, so four planes need two.
        assert_eq!(plan(true, vec![checkpoint(0, 4)]).state_fields(), 2);
        // A checkpoint that fills one field exactly asks for one.
        assert_eq!(plan(true, vec![checkpoint(0, 2)]).state_fields(), 1);
        // A pool with no state group asks for none.
        assert_eq!(plan(false, vec![checkpoint(0, 4)]).state_fields(), 0);
        assert_eq!(plan(true, Vec::new()).state_fields(), 0);
    }

    /// A dense pool of a model with many layers per group plans, and a unit spans every column.
    ///
    /// The geometry was written for a one-column pool and the pools have been per-layer since R8:
    /// a unit holds each column's key and value plane and its two flags, which `CachePlanes`
    /// addresses as `column * num_units + unit`. A one-column figure made `CachePlan::validated` refuse
    /// every model with more than two layers, which is every model.
    #[test]
    fn a_dense_pool_spans_every_column_in_its_unit() -> Result<()> {
        let layers = 48;
        let block_size = 16;
        let kv_heads = 4;
        let head_dim = 128;
        let plan = (CachePlan::dense(
            block_size,
            layers,
            kv_heads,
            head_dim,
            PrefixCacheDtype::BFloat16,
        ))
        .validated()?;
        let column = u64::from(kv_heads) * u64::from(head_dim) * 2 * u64::from(block_size);
        assert_eq!(plan.columns, layers);
        assert_eq!(plan.unit_bytes, u64::from(layers) * (2 * column + 2));
        Ok(())
    }

    /// A plan's request state fields read as the fields of device banks.
    #[test]
    fn state_fields_read_as_device_bank_fields() -> Result<()> {
        let fields = state_fields(
            r#"[{"name": "layers.0.mixer/conv.values", "dtype": "bfloat16", "shape": [4, 8]},
                {"name": "layers.0.mixer/conv.initialized", "dtype": "bool", "shape": []}]"#,
        )?;
        assert_eq!(
            fields.iter().map(StateField::slot).collect::<Vec<_>>(),
            vec![
                SlotField {
                    name: "layers.0.mixer/conv.values".into(),
                    dtype: ElementType::BF16,
                    shape: vec![4, 8],
                    host: false,
                },
                SlotField {
                    name: "layers.0.mixer/conv.initialized".into(),
                    dtype: ElementType::Bool,
                    shape: vec![],
                    host: false,
                },
            ]
        );
        assert!(state_fields("[]")?.is_empty());
        assert!(state_fields(r#"[{"name": "x/y", "dtype": "complex", "shape": []}]"#).is_err());
        Ok(())
    }
}
