//! Dense weights converted at load into FP8 or NVFP4, for products that encode their activations
//! at a scale measured on every call ([`Scheme::DynamicFp8`], [`Scheme::DynamicNvfp4`]).
//!
//! A converted weight is encoded once, on the host, from the checkpoint's own element values
//! widened to FP32, under one tensor scale taken from the largest magnitude of the **whole**
//! checkpoint tensor (`amax_w`):
//!
//! ```text
//! FP8     value[i]          = e4m3(x[i] · (448 / amax_w))                     nearest even, saturating
//! NVFP4   g                 = 6 · 448 / amax_w
//!         scale[r, b]       = e4m3((amax(block b of row r) / 6) · g)          nearest even, saturating
//!         code[r, i]        = e2m1(x[r, i] · 1 / (scale[r, b] · (1 / g)))      nearest even, saturating
//! ```
//!
//! An E2M1 code is two to a byte, the even element in the low nibble, and a block whose scale
//! rounds to zero encodes to zeros. The NVFP4 arithmetic is the activation encoder's CPU formula
//! ([`crate::functional::nvfp4_quantize`]), so a weight and an activation are one encoding.
//!
//! A tensor-parallel rank's share (a column split's rows, or a row split's columns, whole
//! sixteen-element blocks for NVFP4) is encoded alone under the whole tensor's scale, which is
//! exactly its part of the whole tensor's encoding: every layout serves the same numbers. The
//! share is published under the key of a dense share with the encoded element type and layout,
//! and `amax_w` as a `[1]` F32 weight keyed by the tensor's file, so a later load finds both
//! retained and reads nothing; the product's factors are computed from `amax_w` on the device
//! ([`ops::dynamic_scale`]).

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::ops::Range;

use float8::F8E4M3;
use half::f16;
use ops::dynamic_scale::{FP8_RANGE, NVFP4_RANGE};
use ops::fp4::{BLOCK_ELEMENTS, ScaleLayout};
use tensor::{DType, Shape, Tensor};

use super::Scheme;
use super::nvfp4::{scales_layout, single, swizzled_scales};
use crate::error::{Error, Result};
use crate::weights::{
    WeightKey, WeightLayout, WeightLoader, WeightSource, dim, extents, source_region,
};

/// The layout a tensor's largest magnitude is filed under.
const ABSMAX: WeightLayout = WeightLayout::named("absmax");

/// The arrangement the packed product reads a converted weight's block scales in.
const PRODUCT_SCALE_LAYOUT: ScaleLayout = crate::functional::PRODUCT_SCALE_LAYOUT;

/// The most host threads one encoding runs over: a projection is encoded in row bands, each band
/// on a thread of its own.
const THREADS: usize = 16;

/// The FP32 value of element `index` of `bytes`, which hold `dtype` elements: BF16, FP16 or
/// FP32, each widened exactly.
fn widened(dtype: DType, bytes: &[u8], index: usize) -> f32 {
    match dtype {
        DType::BF16 => {
            let bits = u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]);
            f32::from_bits(u32::from(bits) << 16)
        }
        DType::F16 => f16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]]).to_f32(),
        _ => f32::from_le_bytes([
            bytes[4 * index],
            bytes[4 * index + 1],
            bytes[4 * index + 2],
            bytes[4 * index + 3],
        ]),
    }
}

/// `rows` split into at most [`THREADS`] contiguous bands of whole rows, the first ones a row
/// longer when the count does not divide.
fn bands(rows: usize) -> Vec<Range<usize>> {
    let count = rows.clamp(1, THREADS);
    let (each, extra) = (rows / count, rows % count);

    let mut start = 0;
    (0..count)
        .map(|band| {
            let length = each + usize::from(band < extra);
            let range = start..start + length;
            start += length;
            range
        })
        .collect()
}

/// The largest magnitude of the `dtype` elements `bytes` holds, folded in bands on host threads.
fn largest_magnitude(dtype: DType, bytes: &[u8]) -> f32 {
    let width = dtype.size_in_bytes();
    let elements = bytes.len() / width;

    std::thread::scope(|scope| {
        let workers: Vec<_> = bands(elements)
            .into_iter()
            .map(|band| {
                scope.spawn(move || {
                    band.map(|index| widened(dtype, bytes, index).abs())
                        .fold(0.0_f32, f32::max)
                })
            })
            .collect();

        // A band's fold cannot panic, so a worker that does not join is not a value to keep.
        workers
            .into_iter()
            .filter_map(|worker| worker.join().ok())
            .fold(0.0_f32, f32::max)
    })
}

/// The E2M1 code of `value`, rounded to nearest with ties to the even code and saturating at
/// ±6: the low three bits index 0, 0.5, 1, 1.5, 2, 3, 4, 6 and the fourth is the sign.
fn e2m1_code(value: f32) -> u8 {
    let magnitude = value.abs();

    // Each step by the midpoint below it: a midpoint belongs to the neighbour whose code is even,
    // the lower one at 0.25, 1.25, 2.5 and 5 and the upper one at 0.75, 1.75 and 3.5.
    let code = if magnitude <= 0.25 {
        0
    } else if magnitude < 0.75 {
        1
    } else if magnitude <= 1.25 {
        2
    } else if magnitude < 1.75 {
        3
    } else if magnitude <= 2.5 {
        4
    } else if magnitude < 3.5 {
        5
    } else if magnitude <= 5.0 {
        6
    } else {
        7
    };
    if value < 0.0 { code | 8 } else { code }
}

/// `rows` rows of `columns` `dtype` elements in `bytes`, encoded as per-tensor E4M3 under the
/// multiplier `encode`: one byte per element, row-major.
fn encode_fp8(dtype: DType, bytes: &[u8], (rows, columns): (usize, usize), encode: f32) -> Vec<u8> {
    let mut values = vec![0_u8; rows * columns];

    std::thread::scope(|scope| {
        let mut rest = values.as_mut_slice();
        for band in bands(rows) {
            let (mine, after) = rest.split_at_mut(band.len() * columns);
            rest = after;

            scope.spawn(move || {
                for (offset, value) in mine.iter_mut().enumerate() {
                    let index = band.start * columns + offset;
                    *value = F8E4M3::from_f32(widened(dtype, bytes, index) * encode).to_bits();
                }
            });
        }
    });
    values
}

/// `rows` rows of `columns` `dtype` elements in `bytes`, encoded as NVFP4 under the global scale
/// `global` (module formula): `(values, scales)`, the packed codes `[rows, columns / 2]` and the
/// E4M3 block scales `[rows, columns / 16]` in the linear layout.
fn encode_nvfp4(
    dtype: DType,
    bytes: &[u8],
    (rows, columns): (usize, usize),
    global: f32,
) -> (Vec<u8>, Vec<u8>) {
    let block = BLOCK_ELEMENTS as usize;
    let blocks = columns / block;
    let mut values = vec![0_u8; rows * columns / 2];
    let mut scales = vec![0_u8; rows * blocks];

    // The factor the encoder divides a block's scale by: the reciprocal of the global scale.
    let tensor_scale = 1.0 / global;

    std::thread::scope(|scope| {
        let (mut rest_values, mut rest_scales) = (values.as_mut_slice(), scales.as_mut_slice());
        for band in bands(rows) {
            let (my_values, after) = rest_values.split_at_mut(band.len() * columns / 2);
            rest_values = after;
            let (my_scales, after) = rest_scales.split_at_mut(band.len() * blocks);
            rest_scales = after;

            scope.spawn(move || {
                for (local, row) in band.enumerate() {
                    for b in 0..blocks {
                        let first = row * columns + b * block;
                        let amax = (first..first + block)
                            .map(|index| widened(dtype, bytes, index).abs())
                            .fold(0.0_f32, f32::max);

                        let byte = F8E4M3::from_f32((amax / 6.0) * global).to_bits();
                        my_scales[local * blocks + b] = byte;

                        // A scale of zero encodes its block to zeros.
                        let relative = F8E4M3::from_bits(byte).to_f32() * tensor_scale;
                        let inverse = if relative != 0.0 { 1.0 / relative } else { 0.0 };

                        for pair in 0..block / 2 {
                            let even = widened(dtype, bytes, first + 2 * pair) * inverse;
                            let odd = widened(dtype, bytes, first + 2 * pair + 1) * inverse;
                            my_values[(local * columns + b * block) / 2 + pair] =
                                e2m1_code(even) | (e2m1_code(odd) << 4);
                        }
                    }
                }
            });
        }
    });
    (values, scales)
}

/// The share a converted projection reads: its checkpoint tensor, which is a dense matrix of
/// BF16, FP16 or FP32, and the share's rows and columns.
struct Share<'s> {
    tensor: &'s str,
    dtype: DType,
    rows: u64,
    columns: u64,
}

impl WeightLoader<'_> {
    /// Loads the dense matrix `source` names (a region or a strided column block of one
    /// checkpoint tensor, `shape` `[rows, columns]`) as per-tensor FP8 ([`Scheme::DynamicFp8`]):
    /// `(weight, amax)`, the `[rows, columns]` E4M3 share and the `[1]` F32 largest magnitude
    /// of the whole tensor its values are divided by `amax / 448` against.
    ///
    /// # Errors
    ///
    /// When the source is not one dense floating matrix, or every element of the tensor is zero.
    pub(crate) fn dynamic_fp8(
        &self,
        source: &WeightSource,
        shape: &Shape,
    ) -> Result<(Tensor, Tensor)> {
        let share = self.share(source, shape, Scheme::DynamicFp8)?;
        let amax = Cell::new(None);
        let weight_amax = self.weight_amax(&share, &amax)?;

        let key = self.share_key(source, DType::F8E4M3, WeightLayout::ROW_MAJOR)?;
        let weight = self.publish_computed(&key, shape, |loader| {
            let bytes = loader.share_bytes(source, share.dtype)?;
            let encode = FP8_RANGE / loader.tensor_amax(&share, &amax)?;

            Ok(Cow::Owned(encode_fp8(
                share.dtype,
                &bytes,
                (share.rows as usize, share.columns as usize),
                encode,
            )))
        })?;
        Ok((weight, weight_amax))
    }

    /// Loads the dense matrix `source` names as NVFP4 ([`Scheme::DynamicNvfp4`]):
    /// `(values, scales, amax)`, the `[rows, columns / 2]` U8 packed codes, their E4M3 block
    /// scales in the swizzled 128x4 layout the product reads, and the `[1]` F32 largest
    /// magnitude of the whole tensor, whose tensor scale is `amax / (6 · 448)`.
    ///
    /// # Errors
    ///
    /// When the source is not one dense floating matrix, its columns are not whole
    /// sixteen-element blocks, or every element of the tensor is zero.
    pub(crate) fn dynamic_nvfp4(
        &self,
        source: &WeightSource,
        shape: &Shape,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let share = self.share(source, shape, Scheme::DynamicNvfp4)?;
        let amax = Cell::new(None);
        let weight_amax = self.weight_amax(&share, &amax)?;

        // The scales the encode writes beside the values, kept for their own publication when
        // both are encoded by this load.
        let encoded_scales: RefCell<Option<Vec<u8>>> = RefCell::new(None);
        let (rows, columns) = (share.rows as usize, share.columns as usize);
        let encode = |loader: &Self| -> Result<(Vec<u8>, Vec<u8>)> {
            let bytes = loader.share_bytes(source, share.dtype)?;
            let global = NVFP4_RANGE / loader.tensor_amax(&share, &amax)?;
            let (values, linear) = encode_nvfp4(share.dtype, &bytes, (rows, columns), global);

            let blocks = share.columns / u64::from(BLOCK_ELEMENTS);
            Ok((values, swizzled_scales(&linear, share.rows, blocks)))
        };

        let values_key = self.share_key(source, DType::U8, WeightLayout::NVFP4_VALUES)?;
        let values = self.publish_computed(
            &values_key,
            &extents(&[share.rows, share.columns / 2]),
            |loader| {
                let (values, scales) = encode(loader)?;
                *encoded_scales.borrow_mut() = Some(scales);
                Ok(Cow::Owned(values))
            },
        )?;

        let scales_key =
            self.share_key(source, DType::F8E4M3, scales_layout(PRODUCT_SCALE_LAYOUT))?;
        let scales = self.publish_computed(
            &scales_key,
            &extents(&PRODUCT_SCALE_LAYOUT.shape(share.rows, share.columns)),
            |loader| {
                let stashed = encoded_scales.borrow_mut().take();
                match stashed {
                    Some(scales) => Ok(Cow::Owned(scales)),
                    None => Ok(Cow::Owned(encode(loader)?.1)),
                }
            },
        )?;
        Ok((values, scales, weight_amax))
    }

    /// The share `source` names, checked to be `shape`'s rows and columns of one dense floating
    /// matrix, whole sixteen-element blocks when `scheme` packs them.
    fn share<'s>(
        &self,
        source: &'s WeightSource,
        shape: &Shape,
        scheme: Scheme,
    ) -> Result<Share<'s>> {
        let Some(tensor) = single(source) else {
            return Err(Error::weight(
                &crate::weights::source_names(source).join("+"),
                "a weight converted at load is one checkpoint tensor".to_owned(),
            ));
        };
        let location = self.checkpoint().locate(tensor)?;
        if location.shape.rank() != 2
            || !matches!(location.dtype, DType::BF16 | DType::F16 | DType::F32)
        {
            return Err(Error::weight(
                tensor,
                format!(
                    "a weight converted at load is a dense BF16, FP16 or FP32 matrix, not {} {:?}",
                    ops::abi::name(location.dtype),
                    location.shape.dims()
                ),
            ));
        }

        let (rows, columns) = (dim(shape, 0), dim(shape, 1));
        let (_, region, _) = source_region(source);
        if shape.rank() != 2 || region.dims() != shape.dims() || rows == 0 || columns == 0 {
            return Err(Error::weight(
                tensor,
                format!(
                    "the share is {:?}, and the weight is asked as the matrix {:?}",
                    region.dims(),
                    shape.dims()
                ),
            ));
        }
        if scheme == Scheme::DynamicNvfp4 && !columns.is_multiple_of(u64::from(BLOCK_ELEMENTS)) {
            return Err(Error::weight(
                tensor,
                format!("a share of {columns} columns is not whole sixteen-element NVFP4 blocks"),
            ));
        }

        Ok(Share {
            tensor,
            dtype: location.dtype,
            rows,
            columns,
        })
    }

    /// The key a share of `source` converted to `dtype` in `layout` is retained under: the dense
    /// share's, with the encoded element type and layout.
    fn share_key(
        &self,
        source: &WeightSource,
        dtype: DType,
        layout: WeightLayout,
    ) -> Result<WeightKey> {
        let names = crate::weights::source_names(source);
        let (start, shape, stride) = source_region(source);

        Ok(WeightKey {
            files: self.identities(&names)?,
            dtype,
            layout,
            tensors: names,
            start,
            shape,
            stride,
            provenance: None,
        })
    }

    /// The share's checkpoint bytes, in the checkpoint's element type `dtype`, row-major.
    fn share_bytes(&self, source: &WeightSource, dtype: DType) -> Result<Vec<u8>> {
        let key = self.share_key(source, dtype, WeightLayout::ROW_MAJOR)?;
        let spans = self.spans(source, &key)?;

        self.gather(&spans)
    }

    /// The `[1]` F32 weight holding the largest magnitude of the share's whole tensor, retained
    /// under the tensor's file; `amax` keeps the value once this load has read it.
    fn weight_amax(&self, share: &Share<'_>, amax: &Cell<Option<f32>>) -> Result<Tensor> {
        let location = self.checkpoint().locate(share.tensor)?;
        let key = WeightKey {
            files: self.identities(&[share.tensor.to_owned()])?,
            dtype: DType::F32,
            layout: ABSMAX,
            tensors: vec![share.tensor.to_owned()],
            start: 0,
            shape: location.shape.clone(),
            stride: None,
            provenance: None,
        };

        self.publish_computed(&key, &Shape::from(1), |loader| {
            let value = loader.tensor_amax(share, amax)?;
            Ok(Cow::Owned(value.to_le_bytes().to_vec()))
        })
    }

    /// The largest magnitude of every element of the share's tensor, read once per load.
    ///
    /// # Errors
    ///
    /// When the tensor's every element is zero, or its largest magnitude is not a number: a
    /// trained projection holds a value, and no scale encodes the other.
    fn tensor_amax(&self, share: &Share<'_>, amax: &Cell<Option<f32>>) -> Result<f32> {
        if let Some(value) = amax.get() {
            return Ok(value);
        }

        let bytes = self.checkpoint().read(share.tensor)?;
        let value = largest_magnitude(share.dtype, &bytes);
        if !(value.is_finite() && value > 0.0) {
            return Err(Error::weight(
                share.tensor,
                format!(
                    "the tensor's largest magnitude is {value}: a weight converted at load is \
                     encoded under it, so it must be a positive number"
                ),
            ));
        }

        amax.set(Some(value));
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_value_rounds_to_the_nearest_e2m1_code_with_ties_to_the_even_one() {
        for (value, code) in [
            (0.0, 0),
            (0.25, 0),
            (0.26, 1),
            (0.75, 2),
            (1.25, 2),
            (1.75, 4),
            (2.5, 4),
            (3.5, 6),
            (5.0, 6),
            (5.01, 7),
            (100.0, 7),
            (-0.5, 9),
            (-6.0, 15),
        ] {
            assert_eq!(e2m1_code(value), code, "{value}");
        }
    }

    #[test]
    fn bands_cover_every_row_once_in_order() {
        for rows in [1, 7, 16, 17, 1000] {
            let bands = bands(rows);
            assert!(bands.len() <= THREADS);
            assert_eq!(bands.first().map(|band| band.start), Some(0));
            assert_eq!(bands.last().map(|band| band.end), Some(rows));
            assert!(bands.windows(2).all(|pair| pair[0].end == pair[1].start));
        }
    }
}
