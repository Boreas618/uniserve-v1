//! Converting an imported span of K/V pages into the destination pool's representation.
//!
//! An importing rank receives another rank's cache export as raw keys and values. When the source does
//! not store this rank's representation — a different element type, FP8 with other scale groups, or a
//! page shape this rank does not store — the span is read into the workspace's scratch and converted,
//! so imported values round and scale exactly as locally written ones (`uniserve/prefix_cache/
//! conversion.py`; specs/worker.md 8, the prefix cache's conversion rows).
//!
//! Two programs do it, launched per span in this order, once per field:
//!
//! ```text
//! page_conversion     raw keys or values (+ their source scales) → the workspace's f32 scratch
//! cache_copy_region   that f32 block → the destination page's units, with the destination encoding
//! ```
//!
//! The first decodes: an FP8 source's stored value is multiplied by its page's scale in FP32 and
//! rounded through `compute_dtype`, which is the rounding the source's own writes had; a dense source
//! is taken as it is. The second writes each layer's rows into its destination unit, grows the unit's
//! FP8 scale when the new rows need a larger one, rescales the unit's resident values, and raises the
//! column's initialized flag.
//!
//! ## The scratch layout
//!
//! One workspace holds `20 E + 8 S` bytes for `E = block_size · heads · head_dim` (the largest group
//! page's elements) and `S` source scale entries:
//!
//! ```text
//! [0, 8E)          raw keys
//! [8E, 16E)        raw values
//! [16E, 20E)       the FP32 block the two programs pass each other
//! [20E, 20E + 8S)  the source pages' FP32 scales
//! ```
//!
//! Both operators take the scratch base and its two extents, so the addresses follow from the layout
//! once and no caller restates it.
//!
//! ## The unit fields
//!
//! A pool's unit field is `[columns, units, plane_bytes]` of raw bytes for values and `[columns, units]`
//! for scales and initialized flags; `columns` is the greatest common divisor of the resident groups'
//! layer counts and a group's page is `units_per_page` units. A dense pool has no scale fields, and a
//! field the pool does not store is absent from the launch rather than empty.

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

    /// The index of the field inside the scratch's raw halves.
    pub const fn index(self) -> u64 {
        match self {
            Self::Key => 0,
            Self::Value => 1,
        }
    }
}

impl fmt::Display for Field {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// The bytes the conversion scratch of one workspace holds.
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
    /// Whether the source carries one FP32 scale per page and head group.
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
pub struct ConversionArgs {
    /// `[scratch_bytes]` u8, contiguous: the workspace's scratch.
    pub scratch: TensorView,
    /// `E`: elements of the largest group page.
    pub elements: u64,
    /// `S`: source scale entries of the largest group page.
    pub scales: u64,
    /// Tokens of the span.
    pub count: u64,
    /// Token offset of the span's first token within its first source page.
    pub source_offset: u64,
    /// Tokens per page of the source.
    pub source_block_size: u32,
    /// Heads of one source scale group; 0 for a dense source.
    pub scale_heads: u32,
    /// Source pages the span covers.
    pub pages: u64,
}

/// Decode one field of an imported span into the workspace's FP32 block.
pub struct PageConversion;

impl Op for PageConversion {
    const NAME: &'static str = "page_conversion";

    type Descriptor = ConversionDescriptor;
    type Args<'a> = ConversionArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
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
                    "the scratch holds {} bytes, the span needs {needed}",
                    args.scratch.shape.dim(0)
                ),
            ));
        }
        let page = u64::from(descriptor.heads)
            * u64::from(descriptor.head_dim)
            * u64::from(args.source_block_size);
        if args.elements != page {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the descriptor's page is {page} elements, the launch says {}",
                    args.elements
                ),
            ));
        }
        let rows = args.count
            * u64::from(descriptor.layers)
            * u64::from(descriptor.heads)
            * u64::from(descriptor.head_dim);
        if args.count == 0 || rows > args.elements {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a span of {} tokens needs {rows} elements, the scratch holds {}",
                    args.count, args.elements
                ),
            ));
        }
        if descriptor.scaled {
            let groups = u64::from(descriptor.heads).div_ceil(u64::from(args.scale_heads));
            let needed_scales = args.count.div_ceil(u64::from(args.source_block_size))
                * u64::from(descriptor.layers)
                * groups;
            if needed_scales > args.scales {
                return Err(Error::argument(
                    Self::NAME,
                    format!(
                        "{needed_scales} scale entries are needed, the scratch holds {}",
                        args.scales
                    ),
                ));
            }
        }
        if args.pages == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a span covers at least one source page".to_owned(),
            ));
        }

        let mut arguments = Arguments::new();
        arguments
            .view(args.scratch)
            .u64(args.elements)
            .u64(args.scales)
            .u64(args.count)
            .u64(args.source_offset)
            .u32(args.source_block_size)
            .u32(args.scale_heads)
            .u64(args.pages);
        kernel.launch(
            LaunchGrid::new([
                args.count * u64::from(descriptor.layers),
                u64::from(descriptor.heads).div_ceil(u64::from(descriptor.head_dim.max(1))),
                1,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
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
pub struct CopyRegionArgs {
    /// `[columns, units, plane_bytes]` u8, contiguous: the field's stored values.
    pub values: TensorView,
    /// `[columns, units]` f32: the field's per-unit scales, present exactly when the destination
    /// stores FP8.
    pub scale: Option<TensorView>,
    /// `[columns, units]` u8: whether each column's rows of that unit were written.
    pub initialized: TensorView,
    /// `[count, columns, heads, head_dim]` f32, contiguous: the block `page_conversion` decoded.
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
    const NAME: &'static str = "cache_copy_region";

    type Descriptor = CopyRegionDescriptor;
    type Args<'a> = CopyRegionArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
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
                args.count,
                u64::from(descriptor.columns),
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

        let scale = args.scale.unwrap_or(args.initialized);
        let mut arguments = Arguments::new();
        arguments
            .view(args.values)
            .view(scale)
            .view(args.initialized)
            .view(args.source)
            .view(args.units)
            .u64(args.offset)
            .u64(args.count);
        // One program per (column, unit slot, token): it writes that column's row of one unit and
        // raises that cell's initialized flag.
        kernel.launch(
            LaunchGrid::new([
                u64::from(descriptor.columns),
                u64::from(descriptor.units_per_page),
                args.count,
            ])?,
            &mut arguments,
            stream,
        )
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
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
