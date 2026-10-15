//! The KV unit pool's layout, read from the model's cache plan.
//!
//! The library's `cache_plan` groups the resident cache layers and derives
//! the pool's column planes (`uniserve_models.cache.units.plan_units`). It
//! returns a `CachePlan`, which the rank reads column by column and passes
//! back unchanged to the library's cache functions.
//!
//! Vocabulary: a *unit* is one allocation unit of the pool (unit 0 is the
//! padding sentinel); a *column* is a layer's position inside a unit
//! (`columns = gcd(layers per group)`); a *plane* is the bytes of one
//! (field, column) of one unit; a group's logical page of `page_tokens`
//! tokens occupies `units_per_page` units.

use uniserve_core::{KvCacheDtype, KvCacheGroup, KvGroupKind};
use uniserve_ipc::KvCacheInfo;

use tvm_ffi::Any;

use crate::ffi::{DType, Handle, integers, texts};
use crate::model::interface::Interface;
use crate::storage::GroupShape;
use crate::storage::memory::parse_dtype;
use crate::{Error, Result};

/// The library module of the cache functions.
pub(super) const CACHE: &str = "uniserve_models.interface.cache";

/// Where this rank's layers of a group lie in the group's transfer layer
/// axis: a transfer tensor spans `total` layers in global cache-layer order,
/// and this rank's layers are the consecutive run starting at `offset`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupAxis {
    pub offset: u32,
    pub total: u32,
}

/// One cache group: layers sharing a history window and a K/V page shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KVGroupPlan {
    pub window: Option<u32>,
    pub page_tokens: u32,
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

impl KVGroupPlan {
    /// Bytes one token of one layer's key (or value) occupies.
    pub fn row_bytes(&self) -> u64 {
        u64::from(self.num_kv_heads) * u64::from(self.head_dim) * self.dtype.bytes() as u64
    }

    pub fn shape(&self) -> Result<GroupShape> {
        GroupShape::new(self.page_tokens, self.units_per_page, self.window)
    }
}

/// The unit pool's layout.
pub struct KVPlan {
    pub columns: u32,
    pub plane_bytes: u64,
    pub quantized: bool,
    pub unit_bytes: u64,
    /// The published stored dtype (`KvCacheInfo::dtype`).
    pub dtype: KvCacheDtype,
    pub groups: Vec<KVGroupPlan>,
    /// The library's `CachePlan`, passed back to its cache functions.
    pub handle: Handle,
}

impl KVPlan {
    /// Plan `loaded`'s unit pool for `block_size`-token pages of the widest
    /// rows (`cache_plan`). `kv_dtype` is `None` for the compute dtype.
    pub fn plan(
        interface: &Interface,
        loaded: &Handle,
        block_size: usize,
        kv_dtype: Option<&str>,
    ) -> Result<Self> {
        let handle = Self::describe(interface, loaded, block_size, kv_dtype)?;
        Self::read(interface, handle)
    }

    /// Whether the attention `backend` reads every resident layer's pages of
    /// `loaded`'s pool with `block_size`-token pages (`reads_pages`).
    pub fn reads_pages(
        interface: &Interface,
        loaded: &Handle,
        backend: &str,
        block_size: usize,
        kv_dtype: Option<&str>,
    ) -> Result<bool> {
        let plan = Self::describe(interface, loaded, block_size, kv_dtype)?;
        interface
            .import(&format!("{CACHE}.reads_pages"))?
            .call_as(&[
                loaded.view(),
                (&Any::from(tvm_ffi::String::from(backend))).into(),
                plan.view(),
            ])
    }

    fn describe(
        interface: &Interface,
        loaded: &Handle,
        block_size: usize,
        kv_dtype: Option<&str>,
    ) -> Result<Handle> {
        Handle::new(
            interface
                .import(&format!("{CACHE}.cache_plan"))?
                .call_values(&[
                    loaded.to_any(),
                    Any::from(block_size as i64),
                    Any::from(tvm_ffi::String::from(kv_dtype.unwrap_or(""))),
                ])?,
        )
    }

    /// Read a `CachePlan` through its column methods.
    pub fn read(interface: &Interface, handle: Handle) -> Result<Self> {
        let call = |method: &str| -> Result<Any> {
            interface
                .import(&format!("{CACHE}.CachePlan.{method}"))?
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
                .map(|name| parse_dtype(name))
                .collect()
        };

        let dtype: tvm_ffi::String = call("dtype")?
            .try_into()
            .map_err(|error| Error::numerical("dtype", &error))?;
        let dtype = dtype.as_str().parse::<KvCacheDtype>().map_err(|_| {
            Error::Interface(format!(
                "cache plan dtype {:?} is not a KV dtype",
                dtype.as_str()
            ))
        })?;
        let quantized = bool::try_from(call("quantized")?)
            .map_err(|error| Error::numerical("quantized", &error))?;

        // One value per group; a window of -1 keeps the whole history.
        let windows =
            integers(call("windows")?).map_err(|error| Error::numerical("windows", &error))?;
        let page_tokens = counts("page_tokens")?;
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
            .map(|(group, layer_count)| KVGroupPlan {
                window: u32::try_from(windows[group]).ok(),
                page_tokens: page_tokens[group],
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

        Ok(Self {
            columns: count("columns")? as u32,
            plane_bytes: count("plane_bytes")?,
            quantized,
            unit_bytes: count("unit_bytes")?,
            dtype,
            groups,
            handle,
        })
    }

    /// Whether groups store different element types: stale bytes of one type
    /// need not decode as finite values of another, so recycled units are
    /// zeroed completely.
    pub fn mixed_types(&self) -> bool {
        self.groups
            .iter()
            .any(|group| group.dtype != self.groups[0].dtype)
    }

    /// Block tables: one per (group, unit position in a page), group-major.
    pub fn tables(&self) -> u32 {
        self.groups.iter().map(|group| group.units_per_page).sum()
    }

    /// The largest page of any group: a unit's whole token range.
    pub fn unit_tokens(&self) -> u32 {
        self.groups
            .iter()
            .map(|group| group.page_tokens)
            .max()
            .unwrap_or(0)
    }

    pub fn group_shapes(&self) -> Result<Vec<GroupShape>> {
        self.groups.iter().map(KVGroupPlan::shape).collect()
    }

    /// The compute dtype KV exports declare.
    pub fn compute_dtype(&self) -> DType {
        self.groups[0].compute_dtype
    }

    /// The pool description reported to the engine.
    pub fn info(&self, num_units: u32) -> KvCacheInfo {
        KvCacheInfo {
            num_units,
            unit_bytes: self.unit_bytes,
            dtype: self.dtype,
            groups: self
                .groups
                .iter()
                .map(|group| KvCacheGroup {
                    kind: group.window.map_or(KvGroupKind::Full, |window| {
                        KvGroupKind::SlidingWindow { window, sink: 0 }
                    }),
                    page_tokens: group.page_tokens,
                    units_per_page: group.units_per_page,
                    layer_ids: group.layer_ids.clone(),
                    num_kv_heads: group.num_kv_heads,
                    total_kv_heads: group.total_kv_heads,
                    kv_head_offset: group.kv_head_offset,
                    head_dim: group.head_dim,
                })
                .collect(),
        }
    }

    /// Elements `E` and scale entries `S` of the largest group page, which
    /// size a KV import workspace: `E = page_tokens * layers * heads * dim`
    /// and `S = page_tokens * layers * heads`, maximized over groups.
    pub fn page_extents(&self) -> (u64, u64) {
        self.groups
            .iter()
            .fold((0, 0), |(elements, scales), group| {
                let rows = u64::from(group.page_tokens)
                    * group.layers.len() as u64
                    * u64::from(group.num_kv_heads);
                (
                    elements.max(rows * u64::from(group.head_dim)),
                    scales.max(rows),
                )
            })
    }
}
