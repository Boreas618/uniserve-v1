//! The KV unit pool's layout, decoded from the model's cache plan.
//!
//! Python's `model.cache_plan` groups the resident cache layers and derives
//! the pool's column planes (`uniserve_models.cache.units.plan_units`). Rust
//! re-validates every relation it relies on before allocating, because a
//! malformed plan would make addressing corrupt device memory.
//!
//! Vocabulary: a *unit* is one allocation unit of the pool (unit 0 is the
//! padding sentinel); a *column* is a layer's position inside a unit
//! (`columns = gcd(layers per group)`); a *plane* is the bytes of one
//! (field, column) of one unit; a group's logical page of `page_tokens`
//! tokens occupies `units_per_page` units.

use uniserve_core::{KvCacheDtype, KvCacheGroup, KvGroupKind};
use uniserve_ipc::KvCacheInfo;

use tvm_ffi::Any;

use crate::ffi::{DType, Handle, Record};
use crate::storage::GroupShape;
use crate::storage::memory::parse_dtype;
use crate::{Error, Result};

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
    /// The plan map as Python returned it, passed back to bindings.
    record: Handle,
}

impl KVPlan {
    /// Decode and validate the `model.cache_plan` result.
    pub fn decode(value: Any) -> Result<Self> {
        let record = Handle::new(value.clone());
        let plan = Record::new("model.cache_plan", value)?;
        let groups = plan
            .get::<tvm_ffi::Array<Any>>("groups")?
            .iter()
            .enumerate()
            .map(|(index, group)| {
                let group = Record::new(format!("model.cache_plan[groups][{index}]"), group)?;
                decode_group(&group)
            })
            .collect::<Result<Vec<_>>>()?;
        let dtype = plan.string("dtype")?;
        let dtype = dtype.parse::<KvCacheDtype>().map_err(|_| {
            Error::Interface(format!("cache plan dtype {dtype:?} is not a KV dtype"))
        })?;

        let decoded = Self {
            columns: u32::try_from(plan.count("columns")?).map_err(|_| overflow("columns"))?,
            plane_bytes: plan.count("plane_bytes")? as u64,
            quantized: plan.get::<bool>("quantized")?,
            unit_bytes: plan.count("unit_bytes")? as u64,
            dtype,
            groups,
            record,
        };
        decoded.validate()?;
        Ok(decoded)
    }

    fn validate(&self) -> Result<()> {
        let invalid = |message: &str| Err(Error::Interface(format!("cache plan {message}")));
        if self.groups.is_empty() || self.columns == 0 || self.plane_bytes == 0 {
            return invalid("has no groups, columns or plane bytes");
        }
        for group in &self.groups {
            let layers = group.layers.len() as u32;
            if !group.page_tokens.is_power_of_two() {
                return invalid("page sizes must be powers of two");
            }
            if u64::from(group.page_tokens) * group.row_bytes() != self.plane_bytes {
                return invalid("page rows must fill one plane");
            }
            if layers == 0
                || !layers.is_multiple_of(self.columns)
                || group.units_per_page != layers / self.columns
            {
                return invalid("columns must divide every group's layers into its units per page");
            }
            if group.layer_ids.len() != group.layers.len() {
                return invalid("needs one global id per resident layer");
            }
            if group.num_kv_heads == 0
                || group.head_dim == 0
                || group.kv_head_offset + group.num_kv_heads > group.total_kv_heads
            {
                return invalid("head interval exceeds its logical bounds");
            }
            if self.quantized != (group.dtype == DType::F8E4M3) {
                return invalid("quantized storage holds FP8 planes, and only it does");
            }
        }
        if self.quantized != (self.dtype == KvCacheDtype::Float8E4m3Fn) {
            return invalid("publishes a dtype that disagrees with its quantization");
        }
        let metadata = 1 + if self.quantized { 4 } else { 0 };
        if u64::from(self.columns) * 2 * (self.plane_bytes + metadata) != self.unit_bytes {
            return invalid("unit bytes disagree with its planes");
        }
        Ok(())
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

    /// The plan map with the pool's unit count, as bindings receive it.
    pub fn layout(&self, num_units: u32) -> Result<tvm_ffi::Map<tvm_ffi::String, Any>> {
        let plan: tvm_ffi::Map<tvm_ffi::String, Any> =
            tvm_ffi::Map::try_from(Any::from(self.record.view()))
                .map_err(|error| Error::numerical("model.cache_plan", &error))?;
        Ok(plan
            .iter()
            .chain([(
                tvm_ffi::String::from("num_units"),
                Any::from(i64::from(num_units)),
            )])
            .collect())
    }
}

fn overflow(field: &str) -> Error {
    Error::Interface(format!("cache plan field `{field}` overflows"))
}

fn small(record: &Record, key: &str) -> Result<u32> {
    u32::try_from(record.count(key)?).map_err(|_| overflow(key))
}

fn decode_group(group: &Record) -> Result<KVGroupPlan> {
    let window: i64 = group.get("window")?;
    let ids = |key: &str| -> Result<Vec<u32>> {
        group
            .get::<tvm_ffi::Array<Any>>(key)?
            .iter()
            .map(|value| {
                i64::try_from(value)
                    .ok()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| overflow(key))
            })
            .collect()
    };
    let layers = group
        .get::<tvm_ffi::Array<Any>>("layers")?
        .iter()
        .map(|name| {
            tvm_ffi::String::try_from(name)
                .map(|name| name.as_str().to_owned())
                .map_err(|_| Error::Interface("cache plan layer names are strings".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let layer_ids = ids("layer_ids")?;
    let axis_layers = ids("axis_layers")?;

    // This rank's layers are one consecutive run of the group's transfer axis.
    let offset = layer_ids
        .first()
        .and_then(|first| axis_layers.iter().position(|layer| layer == first))
        .ok_or_else(|| {
            Error::Interface("cache group layers are missing from their transfer axis".into())
        })?;
    if axis_layers.get(offset..offset + layer_ids.len()) != Some(layer_ids.as_slice()) {
        return Err(Error::Interface(
            "cache group layers must be one consecutive transfer run".into(),
        ));
    }

    Ok(KVGroupPlan {
        window: u32::try_from(window).ok(),
        page_tokens: small(group, "page_tokens")?,
        units_per_page: small(group, "units_per_page")?,
        layers,
        layer_ids,
        axis: GroupAxis {
            offset: offset as u32,
            total: axis_layers.len() as u32,
        },
        num_kv_heads: small(group, "num_kv_heads")?,
        total_kv_heads: small(group, "total_kv_heads")?,
        kv_head_offset: small(group, "kv_head_offset")?,
        head_dim: small(group, "head_dim")?,
        dtype: parse_dtype(&group.string("dtype")?)?,
        compute_dtype: parse_dtype(&group.string("compute_dtype")?)?,
    })
}
