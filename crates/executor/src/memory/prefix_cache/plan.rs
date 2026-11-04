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
//! [`CachePlan::new`] validates the description and derives nothing hidden
//! from it. The executor owns the pool the plan describes: its size, its
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

/// The pool a model's cache layers describe, before it is validated.
///
/// The model owns its cache geometry: it counts its own layers, states the
/// widest row and the block size of each group, and names the fields its
/// request states hold. The executor plans the pool from that description.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheDescription {
    /// Layers per unit: one column each.
    pub columns: u32,
    /// Bytes of one (column, field) of one unit.
    pub plane_bytes: u64,
    /// Whether the planes hold FP8 values with per-plane scales.
    pub quantized: bool,
    /// Bytes one unit occupies: planes, flags and FP8 scales.
    pub unit_bytes: u64,
    /// The published stored element type.
    pub dtype: PrefixCacheDtype,
    /// The resident groups, in table order; the state group is last.
    pub groups: Vec<CacheGroupPlan>,
    /// The fields of the layers kept per request slot.
    pub states: Vec<StateField>,
    /// Each resident layer's checkpoint layout, when the model packs one.
    pub checkpoints: Vec<Checkpoint>,
}

impl CachePlan {
    /// Plan the unit pool a model describes.
    ///
    /// # Errors
    ///
    /// Fails when a group has no layers, a logical page of a group takes no
    /// unit or more units than its block size allows, the groups' layer
    /// counts share no column count, the state group is not the last group,
    /// or a field of the pool is inconsistent with its groups.
    pub fn new(description: CacheDescription) -> Result<Self> {
        let CacheDescription {
            columns,
            plane_bytes,
            quantized,
            unit_bytes,
            dtype,
            groups,
            states,
            checkpoints,
        } = description;
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
