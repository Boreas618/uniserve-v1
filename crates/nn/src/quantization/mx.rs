//! Block-scaled FP8 exports: the `fp8` producer's declaration, and the loading of its MXFP8
//! matrices and MXFP4 expert stacks into the arrangements their products declare.
//!
//! The producer's `quantization_config` states the representation of every quantized
//! projection: `weight_block_size [32, 32]` (one scale per 32 × 32 block of a weight),
//! `scale_fmt: ue8m0` (each scale a power of two stored as its exponent byte, `F8_E8M0`),
//! `activation_scheme: dynamic` (activations are encoded per call) and, optionally,
//! `expert_dtype: fp4` (a weight stored as packed bytes is MXFP4). Each quantized module stores
//! its `weight` and, beside it, its `scale` ([`SCALE`]); the shard decides which weights are
//! quantized and how ([`BlockScaled::resolve`]):
//!
//! | `weight` | `scale` | Scheme |
//! | --- | --- | --- |
//! | `F8_E4M3 [rows, columns]` | `[rows / 32, columns / 32]`, per block | [`Scheme::Mxfp8`] |
//! | `F8_E4M3 [rows, columns]` | `[rows, columns / 32]`, per 32 elements of a row | [`Scheme::Mxfp8`] |
//! | packed bytes `[rows, columns / 2]` | `[rows, columns / 32]` | [`Scheme::Mxfp4`] |
//!
//! Every other tensor is read as stored. A 32 × 32 block's scale is the scale of each of its rows'
//! 32-element blocks, so a block-scaled weight is an MXFP8 matrix once its scales are repeated
//! over their rows, which is exact; the loader publishes that table in the layout the product
//! reads ([`ops::mx::ScaleLayout`]) and the E4M3 codes as stored.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ops::Range;

use ops::fp4::ValueLayout;
use ops::mx::{BLOCK, ScaleLayout, WORD_ELEMENTS, arranged};
use tensor::{DType, Tensor};

use super::nvfp4::arrange;
use super::{Scheme, refuse};
use crate::checkpoint::Checkpoint;
use crate::config::Fields;
use crate::error::{Error, Result};
use crate::weights::{WeightKey, WeightLayout, WeightLoader, WeightSource, dim, extents};

/// The export's name of a quantized module's scale tensor, beside its `weight`.
pub const SCALE: &str = "scale";

/// The block a block-scaled export's FP8 weights are scaled in, rows by columns, and the only
/// one the MX products read.
const WEIGHT_BLOCK: [u32; 2] = [32, 32];

/// The class the tables this loader computes are filed under: the producer of their keys.
const PRODUCER: &str = "nn::mx";

/// What a block-scaled FP8 export declares (the `fp8` producer's `quantization_config`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlockScaled {
    /// Whether a weight stored as packed bytes beside a scale is MXFP4 (`expert_dtype: fp4`).
    packed_fp4: bool,
}

impl BlockScaled {
    /// The declaration `fields` states.
    ///
    /// # Errors
    ///
    /// When the block is not 32 × 32, the scales are not UE8M0, the activations are not encoded
    /// per call, or the expert type is another than FP4: no operator of this build multiplies
    /// such a weight, and reading it as this one would misread its scales.
    pub(super) fn read(fields: &Fields<'_>) -> Result<Self> {
        let owner = fields.owner();

        let block = fields.counts("weight_block_size")?;
        if block != WEIGHT_BLOCK {
            return Err(refuse(
                owner,
                format!(
                    "states `weight_block_size` {block:?}; the block-scaled products read 32 x 32 \
                     blocks, each block's scale repeated over its rows into MXFP8"
                ),
            ));
        }
        match fields.optional_text("scale_fmt")? {
            Some("ue8m0") => {}
            stated => {
                return Err(refuse(
                    owner,
                    format!(
                        "states `scale_fmt` {stated:?}; the MX products read power-of-two UE8M0 \
                         scales (`ue8m0`), and FP32 block scales have no operator"
                    ),
                ));
            }
        }
        let activations = fields.text("activation_scheme")?;
        if activations != "dynamic" {
            return Err(refuse(
                owner,
                format!(
                    "states `activation_scheme` `{activations}`; the MX products encode each \
                     activation per call (`dynamic`) and read no calibrated scale"
                ),
            ));
        }
        let packed_fp4 = match fields.optional_text("expert_dtype")? {
            None => false,
            Some("fp4") => true,
            Some(other) => {
                return Err(refuse(
                    owner,
                    format!(
                        "states `expert_dtype` `{other}`; packed experts are read as MXFP4 \
                         (`fp4`)"
                    ),
                ));
            }
        };
        Ok(Self { packed_fp4 })
    }

    /// The scale tensor beside `weight`, a weight's full name: `<module>.scale` for
    /// `<module>.weight`.
    pub fn scale_of(&self, weight: &str) -> String {
        let module = weight.strip_suffix(".weight").unwrap_or(weight);
        format!("{module}.{SCALE}")
    }

    /// Each weight of `checkpoint` that its shard stores with a scale beside it, with the scheme
    /// it is read in (module documentation's table).
    ///
    /// # Errors
    ///
    /// When a scale is not UE8M0 bytes, or a weight and its scale are none of the table's forms:
    /// such a weight would be read under a scale that does not describe it.
    pub(super) fn resolve(&self, checkpoint: &Checkpoint) -> Result<BTreeMap<String, Scheme>> {
        let mut scaled = BTreeMap::new();

        for name in checkpoint.names() {
            let Some(module) = name.strip_suffix(&format!(".{SCALE}")) else {
                continue;
            };
            let weight = format!("{module}.weight");
            let Some(stored) = checkpoint.tensor(&weight) else {
                continue;
            };
            let scale = checkpoint.locate(name)?;

            // `F8_E8M0` tensors are read as their exponent bytes (`crate::checkpoint`).
            if scale.dtype != DType::U8 {
                return Err(Error::weight(
                    name,
                    format!(
                        "is {}, and the declaration's `ue8m0` scales are exponent bytes",
                        ops::abi::name(scale.dtype)
                    ),
                ));
            }
            let (rows, stored_columns) = (dim(&stored.shape, 0), dim(&stored.shape, 1));
            let scales = [dim(&scale.shape, 0), dim(&scale.shape, 1)];
            let ranks = (stored.shape.rank(), scale.shape.rank());
            let per_block = |columns: u64| [rows.div_ceil(32), columns.div_ceil(32)];
            let per_row = |columns: u64| [rows, columns.div_ceil(32)];

            let scheme = match stored.dtype {
                DType::F8E4M3
                    if ranks == (2, 2)
                        && (scales == per_block(stored_columns)
                            || scales == per_row(stored_columns)) =>
                {
                    Scheme::Mxfp8
                }
                DType::U8 if self.packed_fp4 && ranks == (2, 2) => {
                    if scales != per_row(stored_columns * 2) {
                        return Err(Error::weight(
                            name,
                            format!(
                                "scales packed MXFP4 of [{rows}, {}] as {scales:?}; MXFP4 has one \
                                 scale per 32 elements of each row, {:?}",
                                stored_columns * 2,
                                per_row(stored_columns * 2)
                            ),
                        ));
                    }
                    Scheme::Mxfp4
                }
                other => {
                    return Err(Error::weight(
                        &weight,
                        format!(
                            "is {} {:?} with a scale of {:?} beside it, which is neither an MXFP8 \
                             matrix's (E4M3 under one scale per 32 x 32 block or per 32 elements \
                             of a row) nor a declared MXFP4 one's",
                            ops::abi::name(other),
                            stored.shape.dims(),
                            scale.shape.dims()
                        ),
                    ));
                }
            };
            scaled.insert(weight, scheme);
        }
        Ok(scaled)
    }
}

/// An MXFP8 matrix, as `mxfp8_gemm` reads it.
#[derive(Clone, Debug)]
pub struct Mxfp8Weight {
    /// `[rows, columns]` E4M3: the codes, as the checkpoint stores them.
    pub codes: Tensor,
    /// The UE8M0 scales, one per 32 elements of each row, in [`Self::layout`]; a grouped
    /// product's weight ([`WeightLoader::mxfp8_groups`]) holds them as the table of its groups'
    /// rows side by side along the contraction.
    pub scales: Tensor,
    /// The layout the scales are published in.
    pub layout: ScaleLayout,
}

/// A stack of MXFP4 experts, as `gather_gemm_mxfp4` and `route_gemm_mxfp4` read it.
#[derive(Clone, Debug)]
pub struct Mxfp4Stack {
    /// `[experts, rows, columns / 2]` U8: two E2M1 codes a byte, the rows in [`Self::values`].
    pub codes: Tensor,
    /// `[experts, padded rows, padded columns / 32]` U8: each expert's UE8M0 scales, rows
    /// permuted as the codes' are, in [`Self::layout`].
    pub scales: Tensor,
    /// The arrangement of each matrix's rows.
    pub values: ValueLayout,
    /// The layout of each expert's scales.
    pub layout: ScaleLayout,
}

/// One checkpoint matrix's share of an [`Mxfp8Weight`]: the weight tensor, and the rows and
/// columns of it the share holds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mxfp8Part {
    /// The weight's full name.
    pub weight: String,
    /// The rows of the share.
    pub rows: Range<u64>,
    /// The columns of the share: whole packed scale words.
    pub columns: Range<u64>,
}

/// The layout name a table of MX scales in `layout` is filed under.
fn scales_key(layout: ScaleLayout) -> WeightLayout {
    match layout {
        ScaleLayout::Linear => WeightLayout::named("mx_scales_linear"),
        ScaleLayout::Swizzled128X4 => WeightLayout::named("mx_scales_swizzled_128x4"),
        ScaleLayout::PackedMnMajor => WeightLayout::named("mx_scales_packed_mn_major"),
    }
}

/// The layout name a stack of MXFP4 codes whose rows are in `layout` is filed under.
fn codes_key(layout: ValueLayout) -> WeightLayout {
    match layout {
        ValueLayout::Plain => WeightLayout::named("mxfp4_codes"),
        ValueLayout::Shuffled { .. } => WeightLayout::named("mxfp4_codes_shuffled"),
        ValueLayout::InterleavedShuffled { .. } => {
            WeightLayout::named("mxfp4_codes_interleaved_shuffled")
        }
        ValueLayout::GateFirst => WeightLayout::named("mxfp4_codes_gate_first"),
    }
}

impl WeightLoader<'_> {
    /// The UE8M0 scale tensor beside the scaled weight `weight`, refused when the loader's
    /// representation states none.
    fn block_scale_of(&self, weight: &str) -> Result<String> {
        self.representation().block_scale(weight).ok_or_else(|| {
            Error::weight(
                weight,
                "is read as an MX matrix, and the checkpoint's representation states no scale \
                 beside it",
            )
        })
    }

    /// The linear `[rows, columns / 32]` UE8M0 table of the share `part` names, one scale per 32
    /// elements of each of its rows: a per-row table's own bytes, or a per-block table's
    /// scales repeated over their rows.
    fn row_scales(&self, part: &Mxfp8Part, scale: &str) -> Result<Vec<u8>> {
        let stored = self.checkpoint().locate(&part.weight)?;
        let (rows, columns) = (dim(&stored.shape, 0), dim(&stored.shape, 1));
        let table = self.checkpoint().locate(scale)?;
        let stored_blocks = dim(&table.shape, 1);
        let per_block = dim(&table.shape, 0) != rows || rows == 0;
        let bytes = self.checkpoint().read(scale)?;

        let first_block = part.columns.start / u64::from(BLOCK);
        let blocks = (part.columns.end - part.columns.start) / u64::from(BLOCK);
        let mut linear = Vec::with_capacity(((part.rows.end - part.rows.start) * blocks) as usize);
        for row in part.rows.clone() {
            // A block-scaled table's row of scales is the block row's.
            let stored_row = if per_block { row / 32 } else { row };
            let start = (stored_row * stored_blocks + first_block) as usize;
            linear.extend_from_slice(&bytes[start..start + blocks as usize]);
        }
        debug_assert!(part.columns.end <= columns);
        Ok(linear)
    }

    /// The MXFP8 matrix the shares `parts` of block-scaled weights make, stacked along their
    /// rows in order: their E4M3 codes as stored, and their scales repeated over their rows and
    /// published in `layout`, the arrangement the reading product declares.
    ///
    /// Every part covers the same columns. A share of one weight is its rows (a column-parallel
    /// rank's block) or its columns (a row-parallel rank's); several parts are whole weights'
    /// rows (a merged projection). `name` is the matrix's logical name, which its keys state.
    ///
    /// # Errors
    ///
    /// When a part is not an MXFP8 weight of the checkpoint, the shares lie outside their
    /// weights or on partial scale words, the parts disagree in their columns, or the codes are
    /// all zero.
    pub fn mxfp8(
        &self,
        name: &str,
        parts: &[Mxfp8Part],
        layout: ScaleLayout,
    ) -> Result<Mxfp8Weight> {
        let Some(first) = parts.first() else {
            return Err(Error::weight(
                name,
                "an MXFP8 matrix of no parts".to_owned(),
            ));
        };
        let columns = first.columns.clone();
        let width = columns.end - columns.start;
        let mut scale_names = Vec::with_capacity(parts.len());
        let mut height = 0;

        for part in parts {
            if self.scheme(&part.weight) != Scheme::Mxfp8 {
                return Err(Error::weight(
                    &part.weight,
                    "is read as an MXFP8 matrix, and the checkpoint does not store it as one"
                        .to_owned(),
                ));
            }
            let stored = self.checkpoint().locate(&part.weight)?;
            let (rows, stored_columns) = (dim(&stored.shape, 0), dim(&stored.shape, 1));
            if part.columns != columns
                || part.rows.start >= part.rows.end
                || part.rows.end > rows
                || columns.end > stored_columns
                || width == 0
                || !columns.start.is_multiple_of(WORD_ELEMENTS)
                || !width.is_multiple_of(WORD_ELEMENTS)
            {
                return Err(Error::weight(
                    &part.weight,
                    format!(
                        "rows {:?} and columns {:?} of a [{rows}, {stored_columns}] weight: one \
                         share of every part's columns, in whole scale words of {WORD_ELEMENTS}",
                        part.rows, part.columns
                    ),
                ));
            }
            scale_names.push(self.block_scale_of(&part.weight)?);
            height += part.rows.end - part.rows.start;
        }

        // The codes: one region or the strided columns of one weight, or whole weights' rows.
        let codes_shape = extents(&[height, width]);
        let source = match parts {
            [part] if width == dim(&self.checkpoint().locate(&part.weight)?.shape, 1) => {
                WeightSource::Region {
                    tensor: part.weight.clone(),
                    start: part.rows.start * width,
                    shape: codes_shape.clone(),
                }
            }
            [part] => WeightSource::Strided {
                tensor: part.weight.clone(),
                start: part.rows.start * dim(&self.checkpoint().locate(&part.weight)?.shape, 1)
                    + columns.start,
                shape: codes_shape.clone(),
                row_stride: dim(&self.checkpoint().locate(&part.weight)?.shape, 1),
            },
            several => {
                for part in several {
                    let stored = self.checkpoint().locate(&part.weight)?;
                    if part.rows != (0..dim(&stored.shape, 0))
                        || columns != (0..dim(&stored.shape, 1))
                    {
                        return Err(Error::weight(
                            &part.weight,
                            "a merged MXFP8 matrix stacks whole weights".to_owned(),
                        ));
                    }
                }
                WeightSource::Parts {
                    parts: several.iter().map(|part| part.weight.clone()).collect(),
                    shape: codes_shape.clone(),
                    block: None,
                }
            }
        };
        let codes = self.load(
            name,
            &source,
            WeightLayout::ROW_MAJOR,
            &codes_shape,
            Some(DType::F8E4M3),
            true,
        )?;

        // The scales: each part's rows of one scale per 32 elements, stacked, then arranged.
        let published = layout.shape(height, width);
        let scales_shape = extents(&published);
        let share = parts
            .iter()
            .map(|part| format!("{}[{:?},{:?}]", part.weight, part.rows, part.columns))
            .collect::<Vec<_>>()
            .join("+");
        let key = WeightKey {
            files: self.identities(&scale_names)?,
            dtype: layout.dtype(),
            layout: scales_key(layout),
            tensors: std::iter::once(format!("{share}__{PRODUCER}_scales"))
                .chain(scale_names.iter().cloned())
                .collect(),
            start: 0,
            shape: scales_shape.clone(),
            stride: None,
            provenance: None,
        };
        let scales = self.publish_computed(&key, &scales_shape, |loader| {
            let mut linear = Vec::new();
            for (part, scale) in parts.iter().zip(&scale_names) {
                linear.extend(loader.row_scales(part, scale)?);
            }
            Ok(Cow::Owned(arranged(&linear, height, width, layout)))
        })?;

        Ok(Mxfp8Weight {
            codes,
            scales,
            layout,
        })
    }

    /// The block-scaled weight `weight`, `[groups · n, k]`, read as the weight of a grouped
    /// product (`mxfp8_gemm` of `groups` groups, group `g` its rows `g · n ..`): its E4M3 codes
    /// as stored, and its scales repeated over their rows and published in `layout` as the table
    /// of the `[n, groups · k]` matrix whose column block `g` is group `g`'s rows, which is where
    /// the grouped product reads each group's scales. `name` is the matrix's logical name.
    ///
    /// # Errors
    ///
    /// When `weight` is not an MXFP8 weight of the checkpoint, its rows are not `groups` whole
    /// groups, its columns are not whole scale words, or its codes are all zero.
    pub fn mxfp8_groups(
        &self,
        name: &str,
        weight: &str,
        groups: u64,
        layout: ScaleLayout,
    ) -> Result<Mxfp8Weight> {
        if self.scheme(weight) != Scheme::Mxfp8 {
            return Err(Error::weight(
                weight,
                "is read as an MXFP8 matrix, and the checkpoint does not store it as one"
                    .to_owned(),
            ));
        }
        let stored = self.checkpoint().locate(weight)?;
        let (rows, columns) = (dim(&stored.shape, 0), dim(&stored.shape, 1));
        if groups == 0
            || !rows.is_multiple_of(groups)
            || columns == 0
            || !columns.is_multiple_of(WORD_ELEMENTS)
        {
            return Err(Error::weight(
                weight,
                format!(
                    "a [{rows}, {columns}] weight read as {groups} groups of its rows: whole \
                     groups, and columns in whole scale words of {WORD_ELEMENTS}"
                ),
            ));
        }
        let scale = self.block_scale_of(weight)?;

        let codes_shape = extents(&[rows, columns]);
        let codes = self.load(
            name,
            &WeightSource::Region {
                tensor: weight.to_owned(),
                start: 0,
                shape: codes_shape.clone(),
            },
            WeightLayout::ROW_MAJOR,
            &codes_shape,
            Some(DType::F8E4M3),
            true,
        )?;

        // The scales: the weight's rows of one scale per 32 elements, each group's row `j` laid
        // beside the other groups' row `j`, then arranged.
        let n = rows / groups;
        let published = layout.shape(n, groups * columns);
        let scales_shape = extents(&published);
        let key = WeightKey {
            files: self.identities(std::slice::from_ref(&scale))?,
            dtype: layout.dtype(),
            layout: scales_key(layout),
            tensors: vec![
                format!("{weight}[groups {groups}]__{PRODUCER}_scales"),
                scale.clone(),
            ],
            start: 0,
            shape: scales_shape.clone(),
            stride: None,
            provenance: None,
        };
        let part = Mxfp8Part {
            weight: weight.to_owned(),
            rows: 0..rows,
            columns: 0..columns,
        };
        let scales = self.publish_computed(&key, &scales_shape, |loader| {
            let linear = loader.row_scales(&part, &scale)?;
            let blocks = (columns / u64::from(BLOCK)) as usize;
            let mut beside = Vec::with_capacity(linear.len());
            for row in 0..n {
                for group in 0..groups {
                    let stored_row = (group * n + row) as usize;
                    beside
                        .extend_from_slice(&linear[stored_row * blocks..(stored_row + 1) * blocks]);
                }
            }
            Ok(Cow::Owned(arranged(&beside, n, groups * columns, layout)))
        })?;

        Ok(Mxfp8Weight {
            codes,
            scales,
            layout,
        })
    }

    /// The MXFP4 stack of the experts `slots` names under `prefix`, each expert's `projections`
    /// stacked along their rows (`<prefix>.<expert>.<projection>.weight`), in the arrangement the
    /// reading product declares: each matrix's rows moved into `layout.0`, and its scales'
    /// rows with them, published in `layout.1`.
    ///
    /// Every expert's projections are whole: a tensor-parallel group holds whole experts (the
    /// block's slots) rather than a share of each one's rows.
    ///
    /// # Errors
    ///
    /// When a projection is not an MXFP4 weight of the checkpoint, the experts disagree in their
    /// extents, the rows are not whole blocks of the arrangement, or the codes are all zero.
    pub fn mxfp4_stack(
        &self,
        name: &str,
        prefix: &str,
        projections: &[&str],
        slots: &[u32],
        layout: (ValueLayout, ScaleLayout),
    ) -> Result<Mxfp4Stack> {
        let (values, scale_layout) = layout;
        if slots.is_empty() || projections.is_empty() {
            return Err(Error::weight(
                name,
                "a stack has experts and at least one projection".to_owned(),
            ));
        }

        let mut weights = Vec::with_capacity(slots.len() * projections.len());
        let mut scale_names = Vec::with_capacity(weights.capacity());
        let mut extent = None;
        for expert in slots {
            for projection in projections {
                let weight = format!("{prefix}.{expert}.{projection}.weight");
                if self.scheme(&weight) != Scheme::Mxfp4 {
                    return Err(Error::weight(
                        &weight,
                        "is read as an MXFP4 expert, and the checkpoint does not store it as one"
                            .to_owned(),
                    ));
                }
                let stored = self.checkpoint().locate(&weight)?;
                let here = (dim(&stored.shape, 0), dim(&stored.shape, 1) * 2);
                match extent {
                    None => extent = Some(here),
                    Some(agreed) if agreed != here => {
                        return Err(Error::weight(
                            &weight,
                            format!(
                                "a [{}, {}] projection in a stack of {agreed:?}",
                                here.0, here.1
                            ),
                        ));
                    }
                    Some(_) => {}
                }
                scale_names.push(self.block_scale_of(&weight)?);
                weights.push(weight);
            }
        }
        let (rows, columns) = extent.unwrap_or_default();
        let count = slots.len() as u64;
        let height = rows * projections.len() as u64;
        if let Some(block) = values.block()
            && !height.is_multiple_of(block)
        {
            return Err(Error::weight(
                name,
                format!("a {values} arrangement permutes {block}-row blocks of {height} rows"),
            ));
        }

        let codes_shape = extents(&[count, height, columns / 2]);
        let codes = self.publish_computed(
            &WeightKey {
                files: self.identities(&weights)?,
                dtype: DType::U8,
                layout: codes_key(values),
                tensors: std::iter::once(format!("{name}__mxfp4_codes"))
                    .chain(weights.iter().cloned())
                    .collect(),
                start: 0,
                shape: codes_shape.clone(),
                stride: None,
                provenance: None,
            },
            &codes_shape,
            |loader| {
                let mut bytes = Vec::with_capacity((count * height * columns / 2) as usize);
                for weight in &weights {
                    bytes.extend(loader.checkpoint().read(weight)?);
                }
                if bytes.iter().all(|byte| *byte == 0) {
                    return Err(Error::weight(
                        name,
                        "every packed code of the stack is zero: the experts would give zeros"
                            .to_owned(),
                    ));
                }
                Ok(Cow::Owned(arrange(
                    values,
                    &bytes,
                    count,
                    height,
                    columns / 2,
                )))
            },
        )?;

        let entry = scale_layout.shape(height, columns);
        let scales_shape = extents(&[count, entry[0], entry[1]]);
        let scales = self.publish_computed(
            &WeightKey {
                files: self.identities(&scale_names)?,
                dtype: DType::U8,
                layout: scales_key(scale_layout),
                tensors: std::iter::once(format!("{name}__mxfp4_scales_{values}"))
                    .chain(scale_names.iter().cloned())
                    .collect(),
                start: 0,
                shape: scales_shape.clone(),
                stride: None,
                provenance: None,
            },
            &scales_shape,
            |loader| {
                let blocks = columns / u64::from(BLOCK);
                let per = projections.len();
                let mut published = Vec::with_capacity((count * entry[0] * entry[1]) as usize);
                for expert in scale_names.chunks(per) {
                    let mut linear = Vec::with_capacity((height * blocks) as usize);
                    for scale in expert {
                        linear.extend(loader.checkpoint().read(scale)?);
                    }
                    // The scales are read beside their rows, so they move with them.
                    let moved = arrange(values, &linear, 1, height, blocks);
                    published.extend(arranged(&moved, height, columns, scale_layout));
                }
                Ok(Cow::Owned(published))
            },
        )?;

        Ok(Mxfp4Stack {
            codes,
            scales,
            values,
            layout: scale_layout,
        })
    }
}
