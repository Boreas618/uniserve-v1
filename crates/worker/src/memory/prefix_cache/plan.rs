//! The cache unit pool's layout, read from the model's cache plan.
//!
//! The library's `cache_plan` groups the resident cache layers and derives
//! the pool's column planes (`uniserve.prefix_cache.units.plan_units`). It
//! returns a `CachePlan`, which the rank reads column by column and passes
//! back unchanged to the library's cache functions.
//!
//! Vocabulary: a *unit* is one allocation unit of the pool (unit 0 is the
//! padding sentinel); a *column* is a layer's position inside a unit
//! (`columns = gcd(layers per group)`); a *plane* is the bytes of one
//! (field, column) of one unit; a group's logical page of `block_size`
//! tokens occupies `units_per_page` units.

use ipc::PrefixCacheInfo;
use ipc::{CacheGroup, CacheGroupKind, PrefixCacheDtype};

use tvm_ffi::Any;

use crate::ffi::{DType, Handle, Interpreter, integers, texts};
use crate::memory::GroupShape;
use crate::memory::request_slots::SlotField;

/// One field of a cache layer kept per request slot: `name` is
/// `layer/field`, `shape` one slot's extent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StateField {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<i64>,
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

/// The library module of the cache functions.
pub(super) const CACHE: &str = "uniserve.prefix_cache.plan";

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
    pub dtype: DType,
    pub compute_dtype: DType,
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

    /// The group as the engine sees it.
    fn kind(&self) -> CacheGroupKind {
        if self.state {
            return CacheGroupKind::State;
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
    /// The library's `CachePlan`, passed back to its cache functions.
    pub handle: Handle,
}

impl CachePlan {
    /// Plan `loaded`'s unit pool for `block_size`-token pages of the widest
    /// rows (`cache_plan`). `prefix_cache_dtype` is `None` for the compute dtype.
    pub(crate) fn plan(
        loaded: &Handle,
        block_size: usize,
        prefix_cache_dtype: Option<&str>,
    ) -> Result<Self> {
        let handle = Self::describe(loaded, block_size, prefix_cache_dtype)?;
        Self::read(handle)
    }

    /// Whether the attention `backend` reads every resident layer's pages of
    /// `loaded`'s pool with `block_size`-token pages (`reads_pages`).
    pub(crate) fn reads_pages(
        loaded: &Handle,
        backend: &str,
        block_size: usize,
        prefix_cache_dtype: Option<&str>,
    ) -> Result<bool> {
        let plan = Self::describe(loaded, block_size, prefix_cache_dtype)?;
        Interpreter::import(&format!("{CACHE}.reads_pages"))?.call_as(&[
            loaded.view(),
            (&Any::from(tvm_ffi::String::from(backend))).into(),
            plan.view(),
        ])
    }

    fn describe(
        loaded: &Handle,
        block_size: usize,
        prefix_cache_dtype: Option<&str>,
    ) -> Result<Handle> {
        Handle::new(
            Interpreter::import(&format!("{CACHE}.cache_plan"))?.call_values(&[
                loaded.to_any(),
                Any::from(block_size as i64),
                Any::from(tvm_ffi::String::from(prefix_cache_dtype.unwrap_or(""))),
            ])?,
        )
    }

    /// Read a `CachePlan` through its column methods.
    pub fn read(handle: Handle) -> Result<Self> {
        let call = |method: &str| -> Result<Any> {
            Interpreter::import(&format!("{CACHE}.CachePlan.{method}"))?
                .call_values(&[handle.to_any()])
        };
        let count = |method: &str| -> Result<u64> {
            let value =
                i64::try_from(call(method)?).map_err(|error| Error::numerical(method, &error))?;
            Ok(value as u64)
        };
        let counts = |method: &str| -> Result<Vec<u32>> {
            let values =
                integers(call(method)?).map_err(|error| Error::numerical(method, &error))?;
            Ok(values.into_iter().map(|value| value as u32).collect())
        };
        let names = |method: &str| -> Result<Vec<String>> {
            texts(call(method)?).map_err(|error| Error::numerical(method, &error))
        };
        let dtypes = |method: &str| -> Result<Vec<DType>> {
            names(method)?
                .iter()
                .map(|name| DType::parse(name))
                .collect()
        };

        let dtype: tvm_ffi::String = call("dtype")?
            .try_into()
            .map_err(|error| Error::numerical("dtype", &error))?;
        let dtype = dtype.as_str().parse::<PrefixCacheDtype>().map_err(|_| {
            Error::Interface(format!(
                "cache plan dtype {:?} is not a cache dtype",
                dtype.as_str()
            ))
        })?;
        let quantized = bool::try_from(call("quantized")?)
            .map_err(|error| Error::numerical("quantized", &error))?;

        // One value per group; a window of -1 keeps the whole history.
        let windows =
            integers(call("windows")?).map_err(|error| Error::numerical("windows", &error))?;
        let kinds = names("kinds")?;
        let block_sizes = counts("block_sizes")?;
        let units_per_page = counts("units_per_page")?;
        let (axis_offsets, axis_sizes) = (counts("axis_offsets")?, counts("axis_sizes")?);
        let (num_kv_heads, total_kv_heads) = (counts("num_kv_heads")?, counts("total_kv_heads")?);
        let (kv_head_offsets, head_dims) = (counts("kv_head_offsets")?, counts("head_dims")?);
        let (stored, compute) = (dtypes("dtypes")?, dtypes("compute_dtypes")?);

        // Every group's resident layers follow one another.
        let mut layers = names("layers")?.into_iter();
        let mut layer_ids = counts("layer_ids")?.into_iter();
        let groups = counts("layer_counts")?
            .into_iter()
            .enumerate()
            .map(|(group, layer_count)| CacheGroupPlan {
                window: u32::try_from(windows[group]).ok(),
                state: kinds[group] == "state",
                block_size: block_sizes[group],
                units_per_page: units_per_page[group],
                layers: layers.by_ref().take(layer_count as usize).collect(),
                layer_ids: layer_ids.by_ref().take(layer_count as usize).collect(),
                axis: GroupAxis {
                    offset: axis_offsets[group],
                    total: axis_sizes[group],
                },
                num_kv_heads: num_kv_heads[group],
                total_kv_heads: total_kv_heads[group],
                kv_head_offset: kv_head_offsets[group],
                head_dim: head_dims[group],
                dtype: stored[group],
                compute_dtype: compute[group],
            })
            .collect();
        let states: tvm_ffi::String = call("state_fields")?
            .try_into()
            .map_err(|error| Error::numerical("state_fields", &error))?;

        Ok(Self {
            columns: count("columns")? as u32,
            plane_bytes: count("plane_bytes")?,
            quantized,
            unit_bytes: count("unit_bytes")?,
            dtype,
            groups,
            states: state_fields(states.as_str())?,
            handle,
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
    pub(crate) fn compute_dtype(&self) -> DType {
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
            identity: String::new(),
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

    /// The units a token capacity of `tokens` takes: the whole pages of
    /// those tokens in every K/V group, and one checkpoint of the state
    /// group per request slot of `request_slots`, the latest each running
    /// request keeps.
    pub(crate) fn token_units(&self, tokens: u64, request_slots: u64) -> u64 {
        self.groups
            .iter()
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
    /// groups. Imports carry K/V pages alone.
    pub(crate) fn page_extents(&self) -> (u64, u64) {
        self.groups
            .iter()
            .filter(|group| !group.state)
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
                dtype: DType::parse(&field.dtype)?,
                name: field.name,
                shape: field.shape,
            })
        })
        .collect()
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
                    dtype: DType::BF16,
                    shape: vec![4, 8],
                    host: false,
                },
                SlotField {
                    name: "layers.0.mixer/conv.initialized".into(),
                    dtype: DType::Bool,
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
