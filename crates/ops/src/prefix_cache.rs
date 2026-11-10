//! Converting a staged host page of K/V into the destination pool's representation.
//!
//! A rank that loads a page another pool stores in another representation — a different element
//! type, or FP8 with other scale groups — stages the page's bytes in the workspace's scratch and
//! converts them, so loaded values round and scale exactly as locally written ones.
//!
//! Two programs do it, launched per page in this order, once per field:
//!
//! ```text
//! page_conversion     raw keys or values (+ their scales) → the workspace's f32 block
//! cache_copy_region   that f32 block → the destination page's units, with the destination encoding
//! ```
//!
//! The first decodes: an FP8 source's stored value is multiplied by its scale in FP32 and rounded
//! through `compute_dtype`, which is the rounding the source's own writes had; a dense source is
//! taken as it is. The second writes each layer's rows into its destination unit and column, grows
//! the unit's FP8 scale when the new rows need a larger one, rescales the unit's resident values, and
//! raises the column's initialized flag. It runs **one program per (column, unit slot)**, and that
//! program owns the written span of its cell: the scale grows once from one amax over the span
//! (`max(scale, max(abs(span), 1e-12) / 448)`) and the resident values are rescaled by `old/new`
//! before the write. A program per token would compute a partial amax each and race several scales
//! into one cell.
//!
//! ## The source: the host page encoding
//!
//! The staged page is a page of the host page pools (`worker::memory::prefix_cache::host`), which is
//! layer-major: keys `[layers, block_size, heads, head_dim]`, values in the same shape, and for an FP8
//! page the scales `[2, layers]` as `f32`, one per field and layer. A page assembled from several
//! source pools (another tensor-parallel split) carries one scale per source pool's head range, so
//! the scales are `[2, layers, groups]` with `groups = ceil(heads / scale_heads)`; one source pool is
//! `groups = 1`, the host page's own tail.
//!
//! ## The scratch layout
//!
//! A launch addresses `20 E + 8 S` bytes for `E = layers · block_size · heads · head_dim` (one field
//! of the page it converts) and `S` its scale entries (`2 · layers · groups`); a workspace holds
//! that figure for the largest page of its groups:
//!
//! ```text
//! [0, 8E)          raw keys      [layers, block_size, heads, head_dim]
//! [8E, 16E)        raw values    the same shape
//! [16E, 20E)       the FP32 block the two programs pass each other, the same shape
//! [20E, 20E + 8S)  the page's FP32 scales [2, layers, groups]
//! ```
//!
//! A staged page lands in it with one copy per field and one for its scale tail. Both operators take
//! the scratch base and its extents, so the addresses follow from the layout once and no caller
//! restates it.
//!
//! ## The unit fields
//!
//! A pool's unit field is `[columns, units, plane_bytes]` of raw bytes for values and `[columns, units]`
//! for scales and initialized flags; `columns` is the greatest common divisor of the resident groups'
//! layer counts and a group's page is `units_per_page` units, its layer `k` in column `k % columns` of
//! unit `k / columns`. A dense pool has no scale fields, and a field the pool does not store is absent
//! from the launch rather than empty.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, Shape, Strides, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// Which half of a K/V page a launch converts.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Field {
    /// The keys.
    Key,
    /// The values.
    Value,
}

impl Field {
    /// The field's name in the pool's field set.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Key => "key",
            Self::Value => "value",
        }
    }
}

impl fmt::Display for Field {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The bytes the conversion scratch of one workspace holds, for `elements` elements of one field
/// of a page and `scales` scale entries of the page.
pub fn scratch_bytes(elements: u64, scales: u64) -> u64 {
    20 * elements + 8 * scales
}

/// One compilation of `page_conversion`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ConversionDescriptor {
    /// The field this launch converts.
    pub field: Field,
    /// Element type of the stored source values.
    pub source: DType,
    /// The source's logical element type, through which a decoded value rounds.
    pub compute: DType,
    /// Elements per head.
    pub head_dim: u32,
    /// Heads of this rank's K/V group.
    pub heads: u32,
    /// Layers of this rank's K/V group.
    pub layers: u32,
    /// Whether the page carries one FP32 scale per field, layer and head group.
    pub scaled: bool,
}

impl fmt::Display for ConversionDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("field", self.field)
            .field("source", self.source)
            .field("compute", self.compute)
            .field("head_dim", self.head_dim)
            .field("heads", self.heads)
            .field("layers", self.layers)
            .field("scaled", self.scaled)
            .finish(formatter)
    }
}

/// The arguments of `page_conversion`.
#[derive(Clone, Copy)]
pub struct ConversionArgs {
    /// `[scratch_bytes]` u8, contiguous: the workspace's scratch, holding the staged page.
    pub scratch: TensorView,
    /// `E`: elements of one field of the page, `layers · block_size · heads · head_dim`.
    pub elements: u64,
    /// `S`: scale entries of the page, `2 · layers · groups`; 0 for a dense page.
    pub scales: u64,
    /// Tokens per page.
    pub block_size: u32,
    /// Heads of one scale group; 0 for a dense page.
    pub scale_heads: u32,
}

/// Decode one field of a staged host page into the workspace's FP32 block.
pub struct PageConversion;

impl Op for PageConversion {
    const NAME: &'static str = proto::program::operator::PAGE_CONVERSION;

    type Descriptor = ConversionDescriptor;
    type Args<'a> = ConversionArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if descriptor.head_dim == 0 || descriptor.heads == 0 || descriptor.layers == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a page with no heads, layers or features converts nothing".to_owned(),
            ));
        }
        if descriptor.scaled {
            if !matches!(descriptor.source, DType::F8E4M3) {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "a scaled source stores {} rather than E4M3",
                        descriptor.source
                    ),
                ));
            }
            if args.scale_heads == 0 {
                return Err(Error::argument(
                    Self::NAME,
                    "a scaled source names its scale group's heads".to_owned(),
                ));
            }
        } else if args.scale_heads != 0 {
            return Err(Error::argument(
                Self::NAME,
                "a dense source has no scale groups".to_owned(),
            ));
        }
        args.scratch.require_rank(Self::NAME, 1)?;
        if args.scratch.dtype != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                format!("the scratch is {}, not raw bytes", args.scratch.dtype),
            ));
        }
        let needed = scratch_bytes(args.elements, args.scales);
        if args.scratch.shape.dim(0) < needed {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the scratch holds {} bytes, the page needs {needed}",
                    args.scratch.shape.dim(0)
                ),
            ));
        }
        if args.block_size == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a page of zero tokens converts nothing".to_owned(),
            ));
        }
        let page = u64::from(descriptor.layers)
            * u64::from(args.block_size)
            * u64::from(descriptor.heads)
            * u64::from(descriptor.head_dim);
        if args.elements != page {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the descriptor's page field is {page} elements, the launch says {}",
                    args.elements
                ),
            ));
        }
        if descriptor.scaled {
            let groups = u64::from(descriptor.heads).div_ceil(u64::from(args.scale_heads));
            let needed_scales = 2 * u64::from(descriptor.layers) * groups;
            if needed_scales != args.scales {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "the page carries {needed_scales} scale entries, the launch says {}",
                        args.scales
                    ),
                ));
            }
        } else if args.scales != 0 {
            return Err(Error::argument(
                Self::NAME,
                "a dense page carries no scales".to_owned(),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let mut arguments = Arguments::new();
        arguments
            .view(args.scratch)
            .u64(args.elements)
            .u64(args.scales)
            .u32(args.block_size)
            .u32(args.scale_heads);
        // One program per (layer, token) row of the page, every head of it.
        kernel.launch(
            LaunchGrid::new([
                u64::from(descriptor.layers) * u64::from(args.block_size),
                1,
                1,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        ConversionArgs {
            scratch: twins.in_place(args.scratch, stream),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // One rounded multiply per stored value: the source's own writes rounded the same way, and a
        // narrow compute type rounds the product once more into the FP32 block.
        let extra =
            if descriptor.compute.unit_roundoff() > f64::from(DType::F32.unit_roundoff() as f32) {
                1.0
            } else {
                0.0
            };
        Tolerance::new(
            0.0,
            (1.0 + extra) * descriptor.compute.unit_roundoff() as f32,
        )
    }
}

/// One compilation of `cache_copy_region`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CopyRegionDescriptor {
    /// The field this launch writes.
    pub field: Field,
    /// Layers one unit holds: `gcd` of the resident groups' layer counts.
    pub columns: u32,
    /// Units one page of the field's group holds.
    pub units_per_page: u32,
    /// Bytes of one unit: `block_size · heads · head_dim · element bytes`.
    pub plane_bytes: u64,
    /// Heads of this rank's K/V group.
    pub heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Element type of the destination's stored values.
    pub element: DType,
    /// Whether the destination stores FP8 with a per-unit scale.
    pub scaled: bool,
}

impl fmt::Display for CopyRegionDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("field", self.field)
            .field("columns", self.columns)
            .field("units_per_page", self.units_per_page)
            .field("plane_bytes", self.plane_bytes)
            .field("heads", self.heads)
            .field("head_dim", self.head_dim)
            .field("element", self.element)
            .field("scaled", self.scaled)
            .finish(formatter)
    }
}

/// The arguments of `cache_copy_region`.
#[derive(Clone, Copy)]
pub struct CopyRegionArgs {
    /// `[columns, units, plane_bytes]` u8, contiguous: the field's stored values.
    pub values: TensorView,
    /// `[columns, units]` f32: the field's per-unit scales, present exactly when the destination
    /// stores FP8.
    pub scale: Option<TensorView>,
    /// `[columns, units]` u8: whether each column's rows of that unit were written.
    pub initialized: TensorView,
    /// `[units_per_page · columns, count, heads, head_dim]` f32, contiguous: the block
    /// `page_conversion` decoded, layer-major like the host page encoding.
    pub source: TensorView,
    /// `[units_per_page]` i32: the destination page's units, non-negative.
    pub units: TensorView,
    /// First destination token within the page.
    pub offset: u64,
    /// Tokens of the span.
    pub count: u64,
}

/// Write a decoded block into a destination page's units, with the destination's encoding.
pub struct CacheCopyRegion;

impl Op for CacheCopyRegion {
    const NAME: &'static str = proto::program::operator::CACHE_COPY_REGION;

    type Descriptor = CopyRegionDescriptor;
    type Args<'a> = CopyRegionArgs;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        if descriptor.columns == 0
            || descriptor.units_per_page == 0
            || descriptor.heads == 0
            || descriptor.head_dim == 0
        {
            return Err(Error::argument(
                Self::NAME,
                "a unit needs layers, units per page, heads and features".to_owned(),
            ));
        }
        let element_bytes = descriptor.element.bytes().ok_or_else(|| {
            Error::argument(
                Self::NAME,
                format!("{} has no byte width", descriptor.element),
            )
        })?;
        let expected_plane = descriptor.plane_bytes;
        args.values.require_rank(Self::NAME, 3)?;
        if args.values.dtype != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                "the stored values are raw bytes".to_owned(),
            ));
        }
        if args.values.shape.dim(0) != u64::from(descriptor.columns)
            || args.values.shape.dim(2) != expected_plane
            || args.values.strides != Strides::row_major(args.values.shape)
        {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the stored values are not [{}, units, {expected_plane}] contiguous bytes",
                    descriptor.columns
                ),
            ));
        }
        let shape = Shape::new(&[u64::from(descriptor.columns), args.values.shape.dim(1)]);
        match (&args.scale, descriptor.scaled) {
            (Some(scale), true) => {
                scale.require_shape(Self::NAME, shape)?;
                scale.require_row(Self::NAME, DType::F32, shape.last())?;
            }
            (None, false) => {}
            (None, true) => {
                return Err(Error::argument(
                    Self::NAME,
                    "an FP8 destination needs its per-unit scales".to_owned(),
                ));
            }
            (Some(_), false) => {
                return Err(Error::argument(
                    Self::NAME,
                    "a dense destination has no per-unit scales".to_owned(),
                ));
            }
        }
        args.initialized.require_shape(Self::NAME, shape)?;
        if args.initialized.dtype != DType::U8 {
            return Err(Error::argument(
                Self::NAME,
                "the initialized flags are one byte per cell".to_owned(),
            ));
        }
        if args.count == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a span of zero tokens writes nothing".to_owned(),
            ));
        }
        args.source.require_shape(
            Self::NAME,
            Shape::new(&[
                u64::from(descriptor.units_per_page) * u64::from(descriptor.columns),
                args.count,
                u64::from(descriptor.heads),
                u64::from(descriptor.head_dim),
            ]),
        )?;
        if args.source.dtype != DType::F32
            || args.source.strides != Strides::row_major(args.source.shape)
        {
            return Err(Error::argument(
                Self::NAME,
                "the decoded block is contiguous FP32".to_owned(),
            ));
        }
        args.units.require_shape(
            Self::NAME,
            Shape::new(&[u64::from(descriptor.units_per_page)]),
        )?;
        if args.units.dtype != DType::I32 {
            return Err(Error::argument(
                Self::NAME,
                format!("the unit ids are {}, not u32", args.units.dtype),
            ));
        }
        let block_size = expected_plane
            / (u64::from(descriptor.heads) * u64::from(descriptor.head_dim) * element_bytes);
        if args.offset + args.count > block_size {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "tokens {}..{} leave a page of {block_size} tokens",
                    args.offset,
                    args.offset + args.count
                ),
            ));
        }
        if u32::try_from(args.values.shape.dim(1)).is_err() {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the field's {} units per column do not fit a kernel parameter",
                    args.values.shape.dim(1)
                ),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let scale = args.scale.unwrap_or(args.initialized);
        let mut arguments = Arguments::new();
        arguments
            .view(args.values)
            .view(scale)
            .view(args.initialized)
            .view(args.source)
            .view(args.units)
            .u64(args.offset)
            .u64(args.count)
            // The field's units *per column*. `values` and `initialized` are addressed as
            // `[column, unit, …]` and the unit ids arrive per page, so a program needs the pool's own
            // shape to turn (column, unit) into an offset; the views do not carry extents into the
            // kernel.
            .u32(args.values.shape.dim(1) as u32);
        // One program per (column, unit slot): it owns the whole written span of that cell, so the
        // FP8 path's amax is one reduction over the span rather than one per token - the write
        // takes `max(scale, max(abs(span), 1e-12) / 448)` once and rescales the unit's resident
        // values by `old/new` before writing. A program per
        // token would give every token its own partial amax and race several scales into one cell.
        kernel.launch(
            LaunchGrid::new([
                u64::from(descriptor.columns),
                u64::from(descriptor.units_per_page),
                1,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        CopyRegionArgs {
            values: twins.in_place(args.values, stream),
            scale: args.scale.map(|scale| twins.in_place(scale, stream)),
            initialized: twins.in_place(args.initialized, stream),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The destination encodes each written value once into its stored type; scaling the unit up
        // rescales the resident values, one rounding each.
        Tolerance::new(0.0, 2.0 * descriptor.element.unit_roundoff() as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DevicePtr;

    fn view(dtype: DType, dims: &[u64]) -> TensorView {
        TensorView::new(DevicePtr::new(0x1000), dtype, Shape::new(dims))
    }

    /// Two layers of a 16-token page of four 64-wide heads, FP8 with one scale per field and layer.
    fn fp8_page() -> ConversionDescriptor {
        ConversionDescriptor {
            field: Field::Key,
            source: DType::F8E4M3,
            compute: DType::BF16,
            head_dim: 64,
            heads: 4,
            layers: 2,
            scaled: true,
        }
    }

    fn page_args(elements: u64, scales: u64, scale_heads: u32) -> ConversionArgs {
        ConversionArgs {
            scratch: view(DType::U8, &[scratch_bytes(elements, scales)]),
            elements,
            scales,
            block_size: 16,
            scale_heads,
        }
    }

    #[test]
    fn a_conversion_takes_one_field_of_a_layer_major_page() {
        let descriptor = fp8_page();
        let field = 2 * 16 * 4 * 64;

        // One source pool: the host page's own `[2, layers]` scale tail.
        assert!(PageConversion::check(&descriptor, &page_args(field, 2 * 2, 4)).is_ok());
        // Two source pools of two heads each: `[2, layers, 2]`.
        assert!(PageConversion::check(&descriptor, &page_args(field, 2 * 2 * 2, 2)).is_ok());
        // One layer's page, or a scale tail of the wrong length, is another page.
        assert!(PageConversion::check(&descriptor, &page_args(field / 2, 2 * 2, 4)).is_err());
        assert!(PageConversion::check(&descriptor, &page_args(field, 2, 4)).is_err());

        let dense = ConversionDescriptor {
            source: DType::BF16,
            scaled: false,
            ..descriptor
        };
        assert!(PageConversion::check(&dense, &page_args(field, 0, 0)).is_ok());
        assert!(PageConversion::check(&dense, &page_args(field, 2 * 2, 0)).is_err());
    }

    #[test]
    fn a_region_copy_reads_every_layer_of_the_page() {
        // Two units of two columns: the page's four layers.
        let descriptor = CopyRegionDescriptor {
            field: Field::Value,
            columns: 2,
            units_per_page: 2,
            plane_bytes: 16 * 4 * 64 * 2,
            heads: 4,
            head_dim: 64,
            element: DType::BF16,
            scaled: false,
        };
        let args = |layers: u64| CopyRegionArgs {
            values: view(DType::U8, &[2, 8, descriptor.plane_bytes]),
            scale: None,
            initialized: view(DType::U8, &[2, 8]),
            source: view(DType::F32, &[layers, 16, 4, 64]),
            units: view(DType::I32, &[2]),
            offset: 0,
            count: 16,
        };

        assert!(CacheCopyRegion::check(&descriptor, &args(4)).is_ok());
        assert!(CacheCopyRegion::check(&descriptor, &args(2)).is_err());
    }
}
