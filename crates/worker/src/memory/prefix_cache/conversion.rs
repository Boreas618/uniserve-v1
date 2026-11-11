//! Converted page loads: pages another host holds in another representation, loaded into this
//! rank's units in its own.
//!
//! A pool that admits converted loads ([`CachePlan::with_conversion`]: an FP8 pool, from its
//! groups' dense compute dtype) keeps, per K/V group, a staging tensor of host storage: an
//! exclusive tensor its host's Storage Server retains, into which the scheduler has the host's
//! Transfer Server READ a peer's pages of the dense source layout ([`ipc::PoolConversion`]). The
//! staged page is never committed to a host pool. A batch's load of it (`PageCopy::staged`) runs on
//! the rank's copy stream, in stream order behind the batch's unit resets, page by page:
//!
//! 1. both fields of the staged page are copied into the device scratch, keys at `[0, 8E)` and
//!    values at `[8E, 16E)` (the scratch layout of `ops::prefix_cache`);
//! 2. per field, `page_conversion` decodes the field into the FP32 block at `[16E, 20E)`, and
//!    `cache_copy_region` encodes the block into the page's units with the pool's own encoding:
//!    one E4M3 scale per (column, unit) from the span's amax, the initialized flag raised last.
//!
//! The scratch holds one page, so the pages convert one after another; stream order keeps each
//! page's copy behind the previous page's kernels. The page's units are a `[units_per_page]` i32
//! row of a small device table, written once per batch before the first page's kernels.
//!
//! A rank's recorded walk states the two operators' descriptors ([`record`]), so the epoch's
//! program set holds their modules beside the forwards'.

use std::sync::Arc;

use ::storage::Key;
use ipc::{PageCopy, PoolConversion};
use ops::prefix_cache::{
    CacheCopyRegion, ConversionArgs, ConversionDescriptor, CopyRegionArgs, CopyRegionDescriptor,
    Field, PageConversion, scratch_bytes,
};
use ops::{DevicePtr, Shape, Operand};
use uniserve_tensor::DType;
use platform::{CopyBatch, DeviceId, SourceOrder};

use super::host;
use super::plan::{CacheGroupPlan, CachePlan};
use super::planes::CachePlanes;
use crate::memory::allocation::{Backing, Storage};
use crate::memory::retained::Retained;
use crate::stream::Issue;
use crate::{Error, Result};

/// The most host bytes one group's staging tensor holds.
const STAGING_BYTES: u64 = 1 << 30;

/// Numbers the staging tensors this process maps, so no two share a key.
static STAGINGS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The two fields a K/V page holds, in the page encoding's order.
const FIELDS: [Field; 2] = [Field::Key, Field::Value];

/// The bytes the device holds for the converted loads of `plan`, `width` pages a batch at most:
/// the conversion scratch of the largest group's page and the unit table. Zero for a plan without
/// converted loads.
pub(crate) fn device_bytes(plan: &CachePlan, width: u32) -> u64 {
    if plan.converts_from.is_none() {
        return 0;
    }
    let elements = plan.groups.iter().map(elements).max().unwrap_or(0);
    let units = plan
        .groups
        .iter()
        .map(|group| group.units_per_page)
        .max()
        .unwrap_or(1);
    scratch_bytes(elements, 0) + u64::from(width) * u64::from(units) * 4
}

/// Elements of one field of one page of `group`: `layers · block_size · heads · head_dim`.
fn elements(group: &CacheGroupPlan) -> u64 {
    group.layers.len() as u64
        * u64::from(group.block_size)
        * u64::from(group.num_kv_heads)
        * u64::from(group.head_dim)
}

/// The descriptors one page of `group` converts with, a pair per field.
struct Programs {
    convert: [ConversionDescriptor; 2],
    copy: [CopyRegionDescriptor; 2],
    /// `E` of the group's page.
    elements: u64,
    block_size: u32,
}

impl Programs {
    /// The programs of `group` of the destination `plan`, whose pages come from `source`.
    fn of(plan: &CachePlan, source: DType, group: &CacheGroupPlan) -> Result<Self> {
        let compute = group.compute_dtype.to_ops().ok_or_else(|| {
            Error::Invalid(format!("a cache computing in {:?}", group.compute_dtype))
        })?;
        let element = group
            .dtype
            .to_ops()
            .ok_or_else(|| Error::Invalid(format!("a cache storing {:?}", group.dtype)))?;
        let layers = group.layers.len() as u32;
        let convert = FIELDS.map(|field| ConversionDescriptor {
            field,
            source,
            compute,
            head_dim: group.head_dim,
            heads: group.num_kv_heads,
            layers,
            scaled: false,
        });
        let copy = FIELDS.map(|field| CopyRegionDescriptor {
            field,
            columns: plan.columns,
            units_per_page: group.units_per_page,
            plane_bytes: plan.plane_bytes,
            heads: group.num_kv_heads,
            head_dim: group.head_dim,
            element,
            scaled: plan.quantized,
        });
        Ok(Self {
            convert,
            copy,
            elements: elements(group),
            block_size: group.block_size,
        })
    }
}

/// The device views one page's conversion launches over.
struct PageViews {
    /// `[scratch_bytes]` u8.
    scratch: Operand,
    /// The FP32 block of the scratch, `[layers, block_size, heads, head_dim]`.
    block: Operand,
    /// Per field: the stored values `[columns, units, plane_bytes]` u8, the scales
    /// `[columns, units]` f32 of a quantized pool and the initialized flags `[columns, units]` u8.
    values: [Operand; 2],
    scales: [Option<Operand>; 2],
    initialized: [Operand; 2],
    /// The page's units, `[units_per_page]` i32.
    units: Operand,
}

/// Launch the conversion of one staged page, already copied into the scratch, on `f`.
fn convert_page(f: &nn::Forward<'_>, programs: &Programs, views: &PageViews) -> Result<()> {
    for index in 0..FIELDS.len() {
        f.operate::<PageConversion>(
            &programs.convert[index],
            ConversionArgs {
                scratch: views.scratch,
                elements: programs.elements,
                scales: 0,
                block_size: programs.block_size,
                scale_heads: 0,
            },
        )?;
        f.operate::<CacheCopyRegion>(
            &programs.copy[index],
            CopyRegionArgs {
                values: views.values[index],
                scale: views.scales[index],
                initialized: views.initialized[index],
                source: views.block,
                units: views.units,
                offset: 0,
                count: u64::from(programs.block_size),
            },
        )?;
    }
    Ok(())
}

/// The shape of the FP32 block a page of `group` decodes into.
fn block_shape(group: &CacheGroupPlan) -> Shape {
    Shape::new(&[
        group.layers.len() as u64,
        u64::from(group.block_size),
        u64::from(group.num_kv_heads),
        u64::from(group.head_dim),
    ])
}

/// States the programs of `plan`'s converted loads on a recording forward: one page of every
/// group, over views at `placeholder`'s synthetic addresses that nothing reads. Nothing for a plan
/// without converted loads.
///
/// # Errors
///
/// When a group's descriptors refuse the page's views.
pub(crate) fn record(
    plan: &CachePlan,
    num_units: u32,
    f: &nn::Forward<'_>,
    placeholder: &dyn Fn(DType, Shape) -> Result<Operand>,
) -> Result<()> {
    let Some(source) = plan.source_plan() else {
        return Ok(());
    };
    let source = source
        .groups
        .first()
        .and_then(|group| group.dtype.to_ops())
        .ok_or_else(|| Error::Invalid("a conversion source of no element type".into()))?;
    let cells = Shape::new(&[u64::from(plan.columns), u64::from(num_units)]);
    for group in &plan.groups {
        let programs = Programs::of(plan, source, group)?;
        let field = |_: Field| -> Result<(Operand, Option<Operand>, Operand)> {
            Ok((
                placeholder(
                    DType::U8,
                    Shape::new(&[
                        u64::from(plan.columns),
                        u64::from(num_units),
                        plan.plane_bytes,
                    ]),
                )?,
                if plan.quantized {
                    Some(placeholder(DType::F32, cells)?)
                } else {
                    None
                },
                placeholder(DType::U8, cells)?,
            ))
        };
        let (key, value) = (field(Field::Key)?, field(Field::Value)?);
        let views = PageViews {
            scratch: placeholder(
                DType::U8,
                Shape::new(&[scratch_bytes(programs.elements, 0)]),
            )?,
            block: placeholder(DType::F32, block_shape(group))?,
            values: [key.0, value.0],
            scales: [key.1, value.1],
            initialized: [key.2, value.2],
            units: placeholder(DType::I32, Shape::new(&[u64::from(group.units_per_page)]))?,
        };
        convert_page(f, &programs, &views)?;
    }
    Ok(())
}

/// One group's staging tensor, mapped into the rank and registered with its device.
struct Staging {
    storage: Arc<Storage>,
    slots: u32,
    page_bytes: u64,
}

/// The converted loads of one rank's pool: the device scratch and unit table, and each group's
/// programs and staging tensor.
pub(crate) struct Conversion {
    plan: Arc<CachePlan>,
    programs: Vec<Programs>,
    staging: Vec<Staging>,
    /// The conversion scratch of the largest group's page, on the device.
    scratch: Arc<Storage>,
    scratch_bytes: u64,
    /// The units of the pages of one batch, `[width, units_per_page]` i32 on the device.
    units: Arc<Storage>,
    width: u32,
    units_per_page: u32,
}

impl Conversion {
    /// The converted loads of `plan`'s pool on `device`, `width` pages a batch at most, with each
    /// group's staging tensor of up to `width` source pages (at most `STAGING_BYTES`) allocated at
    /// the rank's host's Storage Server
    /// (`Request::AllocateHost`); `None` for a plan without converted loads. Answers what the rank
    /// reports of each group's conversion, in group order.
    ///
    /// # Errors
    ///
    /// When the server refuses a staging tensor or a device allocation fails.
    pub(crate) fn start(
        retained: &Retained,
        plan: &Arc<CachePlan>,
        device: DeviceId,
        width: u32,
    ) -> Result<Option<(Self, Vec<PoolConversion>)>> {
        let Some(source_plan) = plan.source_plan() else {
            return Ok(None);
        };
        let source = source_plan
            .groups
            .first()
            .and_then(|group| group.dtype.to_ops())
            .ok_or_else(|| Error::Invalid("a conversion source of no element type".into()))?;
        let mut programs = Vec::with_capacity(plan.groups.len());
        let mut staging = Vec::with_capacity(plan.groups.len());
        let mut reports = Vec::with_capacity(plan.groups.len());
        for (index, group) in plan.groups.iter().enumerate() {
            programs.push(Programs::of(plan, source, group)?);
            let shape = host::page_shape(&source_plan, index);
            // The tensor is named by the rank's process, device, group and mapping: it is this
            // rank's own writable staging, which nothing else retains or reads.
            let mut digest = ::storage::key::Digest::new(b"uniserve-converted-load-staging");
            digest.number(u64::from(std::process::id()));
            digest.number(device.0 as u64);
            digest.number(index as u64);
            digest.number(STAGINGS.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
            let key = Key(digest.finish());
            // One request's pages of the widest block table, at most `STAGING_BYTES` of them: a
            // page the scheduler finds no free slot for is computed instead.
            let slots =
                (STAGING_BYTES / shape.page_bytes.max(1)).clamp(1, u64::from(width.max(1))) as u32;
            let bytes = u64::from(slots) * shape.page_bytes;
            let got = retained.call("a converted-load staging tensor", |client| {
                client.allocate_host(key, format!("converted-load staging {index}"), bytes)
            })?;
            let (_, descriptor) = got.storages.into_iter().next().ok_or_else(|| {
                Error::Interface("a staging tensor arrived without its storage".into())
            })?;
            let storage = Storage::map_host(descriptor, bytes as usize, device, true)?;
            tracing::info!(
                group = index,
                slots,
                page_bytes = shape.page_bytes,
                "mapped a converted-load staging tensor"
            );
            staging.push(Staging {
                storage,
                slots,
                page_bytes: shape.page_bytes,
            });
            reports.push(PoolConversion {
                source: shape,
                staging: key,
                slots,
            });
        }
        let scratch_bytes = programs
            .iter()
            .map(|programs| scratch_bytes(programs.elements, 0))
            .max()
            .unwrap_or(0);
        let units_per_page = plan
            .groups
            .iter()
            .map(|group| group.units_per_page)
            .max()
            .unwrap_or(1);
        let scratch = Storage::device(device, scratch_bytes as usize)?;
        let units = Storage::device(
            device,
            (u64::from(width) * u64::from(units_per_page) * 4) as usize,
        )?;
        Ok(Some((
            Self {
                plan: Arc::clone(plan),
                programs,
                staging,
                scratch,
                scratch_bytes,
                units,
                width,
                units_per_page,
            },
            reports,
        )))
    }

    /// Load the staged pages of `loads` into their units on `f`'s stream (`issue`): their units
    /// are new units of the batch, which the batch's resets cleared before. The staged slots are
    /// pinned by the scheduler until the batch's output is accepted.
    ///
    /// # Errors
    ///
    /// When a page names a group or slot outside the rank's staging, more pages than a batch's
    /// unit table holds, or a launch fails.
    pub(crate) fn load(
        &self,
        f: &nn::Forward<'_>,
        planes: &CachePlanes,
        loads: &[&PageCopy],
        issue: impl Issue,
    ) -> Result<()> {
        if loads.is_empty() {
            return Ok(());
        }
        if loads.len() > self.width as usize {
            return Err(Error::Invalid(format!(
                "a batch converts {} pages and its unit table holds {}",
                loads.len(),
                self.width
            )));
        }
        // Every page's units, one row of the table each, written before the first launch.
        let stride = self.units_per_page as usize;
        let mut table = vec![0_i32; loads.len() * stride];
        for (row, copy) in loads.iter().enumerate() {
            for (column, unit) in copy.units.iter().enumerate() {
                table[row * stride + column] = i32::try_from(unit.0)
                    .map_err(|_| Error::Invalid(format!("unit {} past i32", unit.0)))?;
            }
        }
        let bytes: Vec<u8> = table.iter().flat_map(|unit| unit.to_ne_bytes()).collect();
        let mut write = CopyBatch::new(SourceOrder::DuringCall);
        write.push(self.units.address(), bytes.as_ptr() as u64, bytes.len());
        // SAFETY: the table is this conversion's, read only by the kernels issued after this copy
        // on the same stream; the host bytes are read before the call returns.
        unsafe { issue.copy(&write) }?;

        let views = planes.views();
        let field = |name: &str| {
            views
                .iter()
                .find(|(field, _)| field == name)
                .map(|(_, view)| view.address())
                .ok_or_else(|| Error::Invariant(format!("the cache planes hold no {name}")))
        };
        let cells = Shape::new(&[u64::from(self.plan.columns), u64::from(planes.num_units())]);
        let plane = Shape::new(&[
            u64::from(self.plan.columns),
            u64::from(planes.num_units()),
            self.plan.plane_bytes,
        ]);
        let mut values = Vec::with_capacity(2);
        let mut scales = Vec::with_capacity(2);
        let mut initialized = Vec::with_capacity(2);
        for name in super::planes::FIELDS {
            values.push(Operand::new(
                DevicePtr::new(field(&format!("{name}.values"))?),
                DType::U8,
                plane,
            ));
            scales.push(if self.plan.quantized {
                Some(Operand::new(
                    DevicePtr::new(field(&format!("{name}.scale"))?),
                    DType::F32,
                    cells,
                ))
            } else {
                None
            });
            // The flags are one byte per cell, which the kernel reads and raises as bytes.
            initialized.push(Operand::new(
                DevicePtr::new(field(&format!("{name}.initialized"))?),
                DType::U8,
                cells,
            ));
        }

        for (row, copy) in loads.iter().enumerate() {
            let group = copy.group_id as usize;
            let (programs, staging, plan) = match (
                self.programs.get(group),
                self.staging.get(group),
                self.plan.groups.get(group),
            ) {
                (Some(programs), Some(staging), Some(plan)) => (programs, staging, plan),
                _ => {
                    return Err(Error::Invalid(format!(
                        "a converted load names cache group {group}, which converts nothing"
                    )));
                }
            };
            if copy.host_page >= u64::from(staging.slots) || copy.units.len() != stride {
                return Err(Error::Invalid(format!(
                    "a converted load of slot {} with {} units, of a staging of {} slots and \
                     pages of {stride} units",
                    copy.host_page,
                    copy.units.len(),
                    staging.slots
                )));
            }
            // The staged page is the host page encoding of the source: keys, then values, each
            // `E` elements of the source dtype.
            let field_bytes = staging.page_bytes / 2;
            let slot = staging.storage.address() + copy.host_page * staging.page_bytes;
            let scratch = self.scratch.address();
            let mut stage = CopyBatch::new(SourceOrder::Stream);
            for index in 0..FIELDS.len() as u64 {
                stage.push(
                    scratch + index * 8 * programs.elements,
                    slot + index * field_bytes,
                    field_bytes as usize,
                );
            }
            // SAFETY: the scratch is this conversion's and the previous page's kernels, which read
            // it, precede this copy on the stream; the staged slot is pinned until the batch's
            // output is accepted.
            unsafe { issue.copy(&stage) }?;
            let page = PageViews {
                scratch: Operand::new(
                    DevicePtr::new(scratch),
                    DType::U8,
                    Shape::new(&[self.scratch_bytes]),
                ),
                block: Operand::new(
                    DevicePtr::new(scratch + 16 * programs.elements),
                    DType::F32,
                    block_shape(plan),
                ),
                values: [values[0], values[1]],
                scales: [scales[0], scales[1]],
                initialized: [initialized[0], initialized[1]],
                units: Operand::new(
                    DevicePtr::new(self.units.address() + (row * stride * 4) as u64),
                    DType::I32,
                    Shape::new(&[u64::from(plan.units_per_page)]),
                ),
            };
            convert_page(f, programs, &page)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use ipc::PrefixCacheDtype;
    use ops::Op;

    use super::*;
    use crate::runners::recording::Footprint;

    /// A dense pool of `dtype` as the engine plans one: 64-token pages of four layers of two
    /// 128-wide heads.
    fn plan(dtype: PrefixCacheDtype) -> CachePlan {
        CachePlan::dense(64, 4, 2, 128, dtype)
            .validated()
            .expect("a dense plan")
    }

    /// A converting FP8 pool asks for exactly the pages a BF16 deployment of the same model
    /// stores: the source plan's page shape is the BF16 pool's, layout digest and bytes alike, and
    /// it differs from the FP8 pool's own.
    #[test]
    fn an_fp8_pool_converts_the_pages_a_bf16_pool_stores() {
        let fp8 = plan(PrefixCacheDtype::Float8E4m3Fn)
            .with_conversion()
            .expect("an FP8 pool converts from its compute dtype");
        let source = fp8.source_plan().expect("a converting plan has a source");
        let bf16 = plan(PrefixCacheDtype::BFloat16);
        assert_eq!(host::page_shape(&source, 0), host::page_shape(&bf16, 0));
        assert_ne!(host::page_shape(&fp8, 0), host::page_shape(&bf16, 0));

        // The other direction lowers the destination's precision, and is refused.
        assert!(bf16.clone().with_conversion().is_err());
        assert!(bf16.source_plan().is_none());
    }

    /// A rank's recorded walk states one `page_conversion` and one `cache_copy_region` per field
    /// of its converted loads, so the epoch's program set holds their modules; a pool that loads
    /// only its own pages states none.
    #[test]
    fn a_converting_pool_states_its_page_load_programs() {
        let fp8 = plan(PrefixCacheDtype::Float8E4m3Fn)
            .with_conversion()
            .expect("a converting plan");
        let walk = |plan: &CachePlan| {
            let mut footprint = Footprint::default();
            footprint
                .walk(None, |walk| {
                    let forward = walk.forward(None)?;
                    record(plan, 8, &forward, &|dtype, shape| {
                        walk.placeholder(dtype, shape)
                    })
                })
                .expect("a recorded conversion launches nothing");
            footprint.keys
        };

        let keys = walk(&fp8);
        let names: BTreeSet<&str> = keys.keys().map(String::as_str).collect();
        assert_eq!(
            names,
            BTreeSet::from([PageConversion::NAME, CacheCopyRegion::NAME])
        );
        assert!(keys.values().all(|keys| keys.len() == 2), "{keys:?}");
        assert_eq!(walk(&plan(PrefixCacheDtype::Float8E4m3Fn)), BTreeMap::new());
    }
}
