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
/// One layer's checkpoint, packed into consecutive planes of a state page.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize)]
pub struct Checkpoint {
    pub first_plane: u32,
    pub planes: u32,
    pub fields: Vec<::storage::manifest::Field>,
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
/// the state group, whose pages hold the checkpoints of the layers kept per
/// request slot, or the draft cache. The group's kind is the engine's
/// `CacheGroupKind`; the plan adds the group's column layout on this rank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CacheGroupPlan {
    /// The group's history policy: a state group's pages hold each of its
    /// layers' state after the page's last token, in bytes the library lays
    /// out; a draft group's hold a drafter's K/V, each layer's page laid over
    /// the columns of its unit.
    pub kind: CacheGroupKind,
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
        GroupShape::new(self.block_size, self.units_per_page, self.kind)
    }

    /// The sliding window of a sliding-window group.
    pub(crate) fn window(&self) -> Option<u32> {
        self.kind.window()
    }

    /// Whether the group's pages hold state checkpoints.
    pub(crate) fn state(&self) -> bool {
        self.kind.is_state()
    }

    /// Whether the group holds the draft cache.
    pub(crate) fn draft(&self) -> bool {
        self.kind.is_draft()
    }

    /// Whether the group holds K/V rows a request needs: neither the state
    /// group's checkpoints nor the optional draft cache.
    pub(crate) fn required_rows(&self) -> bool {
        self.kind.required_rows()
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
                kind: CacheGroupKind::Full,
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
        if rows == 0 || !self.groups.iter().any(|group| group.state()) {
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
            && rest.iter().any(|group| group.state())
        {
            return Err(Error::Invalid(format!(
                "the state group {:?} must be the last cache group",
                last.layers
            )));
        }
        // A draft group's layers are laid over the columns of its units rather than one cell each
        // ([`CachePlan::with_draft`]), so they bound no column count.
        let columns_of_groups = groups
            .iter()
            .filter(|group| !group.draft())
            .fold(0, |columns, group| gcd(columns, group.layers.len() as u32));
        if let Some(draft) = groups.iter().find(|group| group.draft())
            && (draft.layers.len() as u32 > columns || draft.units_per_page != 1)
        {
            return Err(Error::Invalid(format!(
                "the draft group's {} layers lie over the {columns} columns of one unit per page, \
                 not {}",
                draft.layers.len(),
                draft.units_per_page
            )));
        }
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
        let Some(group) = self.groups.iter().find(|group| group.state()) else {
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

    pub(crate) fn group_shapes(&self) -> Result<Vec<GroupShape>> {
        self.groups.iter().map(CacheGroupPlan::shape).collect()
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

    /// The units a token capacity of `tokens` takes: the whole pages of
    /// those tokens in every K/V group, and one checkpoint of the state
    /// group per request slot of `request_slots`, the latest each running
    /// request keeps. The draft group's pages, optional state of those same
    /// tokens, take units the capacity holds as far as they are free.
    pub(crate) fn token_units(&self, tokens: u64, request_slots: u64) -> u64 {
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

    /// Planes of one field one page of `group` spans: every column of every
    /// unit of the page. A K/V group's page holds one plane per layer and
    /// field (its layers fill whole units); the state group's checkpoints
    /// fill the planes of its units in page order.
    pub(crate) fn page_planes(&self, group: &CacheGroupPlan) -> u64 {
        u64::from(group.units_per_page) * u64::from(self.columns)
    }
}

/// Where one field of a recurrent layer's checkpoint lies in a state page: its first plane and its
/// bytes, which fill whole planes from that one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatePlanes {
    /// The field's first plane among the state page's planes.
    pub first_plane: u32,
    /// The field's bytes per slot.
    pub bytes: u64,
}

impl CachePlan {
    /// The pool a model of `layers` attention layers and the recurrent `state` describes: one K/V
    /// group of the attention layers, as [`Self::dense`] lays it out, and the state group after it,
    /// whose page holds every recurrent layer's checkpoint.
    ///
    /// The columns are the two groups' common divisor. A checkpoint lays each layer's fields out
    /// in field order, each field from a plane boundary over the whole planes its bytes fill
    /// ([`Self::state_planes`]); the state page spans as many units as those planes fill in one
    /// field run, so a unit carries one plane of state per column beside its K/V planes and a K/V
    /// page's units pay one state plane each rather than a checkpoint's worth.
    ///
    /// # Errors
    ///
    /// A geometry with no attention layer, no recurrent layer or a field of no bytes.
    pub fn hybrid(
        block_size: u32,
        layers: u32,
        kv_heads: u32,
        head_dim: u32,
        dtype: PrefixCacheDtype,
        state: &nn::family::StateShape,
    ) -> Result<Self> {
        if layers == 0 || state.layers == 0 || state.fields.is_empty() {
            return Err(Error::Invalid(format!(
                "a hybrid cache of {layers} attention layers and {} recurrent layers of {} fields",
                state.layers,
                state.fields.len()
            )));
        }
        let mut plan = Self::dense(block_size, layers, kv_heads, head_dim, dtype);
        let columns = gcd(layers, state.layers);
        let per_column = plan.unit_bytes / u64::from(plan.columns);
        plan.columns = columns;
        let kv = &mut plan.groups[0];
        kv.units_per_page = layers / columns;

        // Every field starts at a plane boundary, so a copy moves whole planes.
        let mut layer_planes = 0_u64;
        for field in &state.fields {
            if field.bytes() == 0 {
                return Err(Error::Invalid(format!(
                    "the recurrent state's field {} holds no byte",
                    field.name
                )));
            }
            layer_planes += field.bytes().div_ceil(plan.plane_bytes);
        }
        let planes = layer_planes * u64::from(state.layers);
        let units_per_page = u32::try_from(planes.div_ceil(u64::from(columns))).map_err(|_| {
            Error::Invalid("a checkpoint fills more units than a page counts".into())
        })?;
        let layer_planes = u32::try_from(layer_planes)
            .map_err(|_| Error::Invalid("a layer's checkpoint fills too many planes".into()))?;
        plan.checkpoints = (0..state.layers)
            .map(|layer| Checkpoint {
                first_plane: layer * layer_planes,
                planes: layer_planes,
                // The manifest names a checkpoint's fields once the Storage Server keys state
                // pages; the device copies read the planes alone.
                fields: Vec::new(),
            })
            .collect();
        plan.groups.push(CacheGroupPlan {
            kind: CacheGroupKind::State,
            block_size,
            units_per_page,
            layers: (0..state.layers)
                .map(|layer| format!("recurrent.{layer}"))
                .collect(),
            layer_ids: (layers..layers + state.layers).collect(),
            axis: GroupAxis {
                offset: 0,
                total: state.layers,
            },
            // A state page holds bytes: one head of one element per layer.
            num_kv_heads: 1,
            total_kv_heads: 1,
            kv_head_offset: 0,
            head_dim: 1,
            dtype: ElementType::BF16,
            compute_dtype: ElementType::BF16,
        });
        // A unit holds its K/V planes and flags per column, and one state plane per column of each
        // field run its page's checkpoints span.
        plan.unit_bytes = u64::from(columns) * per_column
            + u64::from(plan.state_fields()) * u64::from(columns) * plan.plane_bytes;
        plan.validated()
    }

    /// The plan with the draft cache added: the K/V of a drafter's `layers` attention layers,
    /// which store `kv_heads` heads of `head_dim` each as the pool's first group does.
    ///
    /// A drafter has fewer layers than a unit has columns (one or two against the pool's ten), so a
    /// draft page does not take a cell per layer as a K/V group's page does. It takes **one unit**,
    /// and each of its layers lies over [`Self::draft_span`] of the unit's columns: column `c` of the
    /// layer's run holds the page's `c`-th run of `block_size / span` tokens. A draft page therefore
    /// holds `span` times the tokens of a K/V page in one unit, and the group's block size is that,
    /// a power of two as every page's is. The group sits before the state group, which stays last.
    ///
    /// # Errors
    ///
    /// A drafter of more layers than the pool has columns, a head geometry the pool's planes do not
    /// hold, or a plan that is no longer valid.
    pub fn with_draft(mut self, layers: u32, kv_heads: u32, head_dim: u32) -> Result<Self> {
        let first = self
            .groups
            .first()
            .cloned()
            .ok_or_else(|| Error::Invalid("a draft cache beside no K/V group".into()))?;
        if layers == 0 || layers > self.columns {
            return Err(Error::Invalid(format!(
                "a drafter of {layers} layers over a pool of {} columns",
                self.columns
            )));
        }
        if kv_heads != first.num_kv_heads || head_dim != first.head_dim {
            return Err(Error::Invalid(format!(
                "a drafter of {kv_heads} heads of {head_dim} over planes of {} heads of {}",
                first.num_kv_heads, first.head_dim
            )));
        }
        let span = Self::span_of(self.columns, layers);
        let ids = self
            .groups
            .iter()
            .flat_map(|group| group.layer_ids.iter().copied())
            .max()
            .map_or(0, |id| id + 1);
        let draft = CacheGroupPlan {
            kind: CacheGroupKind::Draft,
            block_size: first.block_size * span,
            units_per_page: 1,
            layers: (0..layers).map(|layer| format!("draft.{layer}")).collect(),
            layer_ids: (ids..ids + layers).collect(),
            axis: GroupAxis {
                offset: 0,
                total: layers,
            },
            ..first
        };
        let at = self
            .groups
            .iter()
            .position(|group| group.state())
            .unwrap_or(self.groups.len());
        self.groups.insert(at, draft);
        self.validated()
    }

    /// The columns of a unit each draft layer lies over: the largest power of two the columns hold
    /// `layers` runs of, so a draft page's tokens are a power of two.
    fn span_of(columns: u32, layers: u32) -> u32 {
        let share = (columns / layers.max(1)).max(1);
        1 << (u32::BITS - 1 - share.leading_zeros())
    }

    /// The columns of a unit each layer of the draft group lies over, or `None` for a plan without
    /// one.
    pub fn draft_span(&self) -> Option<u32> {
        let draft = self.groups.iter().find(|group| group.draft())?;
        Some(Self::span_of(self.columns, draft.layers.len() as u32))
    }

    /// The draft group's index, when the plan holds one.
    pub fn draft_group(&self) -> Option<usize> {
        self.groups.iter().position(|group| group.draft())
    }

    /// Where each field of a recurrent layer's checkpoint lies, field by field, for layer
    /// `layer` of `fields`: the layout [`Self::hybrid`] planned.
    pub fn state_planes(&self, layer: u32, fields: &[nn::family::StateField]) -> Vec<StatePlanes> {
        let first = self
            .checkpoints
            .get(layer as usize)
            .map_or(0, |checkpoint| checkpoint.first_plane);
        let mut plane = first;
        fields
            .iter()
            .map(|field| {
                let at = StatePlanes {
                    first_plane: plane,
                    bytes: field.bytes(),
                };
                plane += field.bytes().div_ceil(self.plane_bytes) as u32;
                at
            })
            .collect()
    }

    /// The planes of every checkpoint of a state page.
    pub fn checkpoint_planes(&self) -> u32 {
        self.checkpoints
            .iter()
            .map(|checkpoint| checkpoint.first_plane + checkpoint.planes)
            .max()
            .unwrap_or(0)
    }
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
            kind: if state {
                CacheGroupKind::State
            } else {
                CacheGroupKind::Full
            },
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

    /// A hybrid pool of ten attention layers and thirty recurrent ones lays every recurrent
    /// layer's checkpoint in whole planes of one field run, and a unit pays one state plane per
    /// column beside its K/V planes.
    #[test]
    fn a_hybrid_pool_lays_each_checkpoint_field_from_a_plane_boundary() -> Result<()> {
        let state = nn::family::StateShape {
            layers: 30,
            fields: vec![
                nn::family::StateField {
                    name: "conv",
                    dtype: ops::DType::BF16,
                    dims: vec![3, 8192],
                },
                nn::family::StateField {
                    name: "recurrent",
                    dtype: ops::DType::F32,
                    dims: vec![32, 128, 128],
                },
            ],
        };
        let plan = CachePlan::hybrid(64, 10, 2, 256, PrefixCacheDtype::BFloat16, &state)?;
        // A K plane is 64 tokens of two 256-wide BF16 heads: 64 KiB.
        assert_eq!(plan.plane_bytes, 64 << 10);
        assert_eq!(plan.columns, 10);
        assert_eq!(plan.groups.len(), 2);
        assert!(plan.groups[1].state());
        // 48 KiB of convolution window take one plane, 2 MiB of state thirty-two: thirty-three per
        // layer, 990 for the page, which 99 units of ten columns hold in one field run.
        assert_eq!(plan.checkpoint_planes(), 990);
        assert_eq!(plan.groups[1].units_per_page, 99);
        assert_eq!(plan.state_fields(), 1);
        let planes = plan.state_planes(2, &state.fields);
        assert_eq!(planes[0].first_plane, 66);
        assert_eq!(planes[1].first_plane, 67);
        assert_eq!(planes[1].bytes, 2 << 20);
        let dense = CachePlan::dense(64, 10, 2, 256, PrefixCacheDtype::BFloat16);
        assert_eq!(plan.unit_bytes, dense.unit_bytes + 10 * plan.plane_bytes);

        // A drafter of one layer lies over eight of a unit's ten columns: its pages hold 512
        // tokens in one unit, before the state group, and leave the pool's columns and units as
        // they were.
        let unit_bytes = plan.unit_bytes;
        let drafting = plan.with_draft(1, 2, 256)?;
        assert_eq!(drafting.columns, 10);
        assert_eq!(drafting.unit_bytes, unit_bytes);
        assert_eq!(drafting.draft_group(), Some(1));
        assert_eq!(drafting.draft_span(), Some(8));
        let draft = &drafting.groups[1];
        assert!(draft.draft() && drafting.groups[2].state());
        assert_eq!((draft.block_size, draft.units_per_page), (512, 1));
        assert_eq!(draft.layer_ids, vec![40]);
        // Two layers lie over four columns each; more layers than columns lie over none.
        let two = CachePlan::hybrid(64, 10, 2, 256, PrefixCacheDtype::BFloat16, &state)?
            .with_draft(2, 2, 256)?;
        assert_eq!(two.draft_span(), Some(4));
        assert!(
            CachePlan::hybrid(64, 10, 2, 256, PrefixCacheDtype::BFloat16, &state)?
                .with_draft(11, 2, 256)
                .is_err()
        );
        Ok(())
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
}
