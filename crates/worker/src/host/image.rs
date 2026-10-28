//! Image host work: preparing a request's inline input images and the
//! images a migrated request carries, and encoding decoded images, packed
//! 8-bit RGB pixels, to base64 PNG ([`ImageTask`]).
//!
//! A decoded image reaches the host as `height * width * 3` bytes in row-major
//! HWC order (`uniserve.media.pixels.quantize_image_hwc`). Encoding runs on a
//! host pool after the device-to-host copy completes. PNG is lossless, so the
//! delivered pixels equal the quantized ones whatever the compression; the fast
//! compression level trades output size for encoding time on large generated
//! images. A decoding that writes its request's feedback image product also
//! stages that product's BF16 values after the bytes, which the encoding
//! publishes after the PNG (`ipc::append_feedback`): the image's exact
//! feedback input, which the daemon keeps for a migration to carry.

use std::io::Write;
use std::sync::{Arc, Mutex};

use crate::executor::image_inputs::{ImageFacts, ImagePreparation, PreparedImage, Tower};
use crate::ffi::DType;
use crate::host::HostAction;
use crate::storage::allocation::StorageView;
use crate::storage::{OutputLease, PinnedPool};
use crate::{Error, Outcome, Result, lock};

/// Encode `pixels` (`height * width * 3` bytes, HWC) as a base64 PNG of at
/// most `maximum` bytes.
///
/// # Errors
/// `Invalid` when the pixels do not fill the raster, and when the encoded
/// image is empty or exceeds `maximum` (the call's completion bound).
pub(crate) fn png_base64(
    pixels: &[u8],
    height: u32,
    width: u32,
    maximum: usize,
) -> Result<Vec<u8>> {
    let expected = u64::from(height) * u64::from(width) * 3;
    if height == 0 || width == 0 || pixels.len() as u64 != expected {
        return Err(Error::Invalid(format!(
            "image pixels hold {} bytes; a {height}x{width} RGB raster holds {expected}",
            pixels.len()
        )));
    }

    let mut png = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut png, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_compression(png::Compression::Fast);
        let mut writer = encoder.write_header().map_err(encoding)?;
        writer.write_image_data(pixels).map_err(encoding)?;
        writer.finish().map_err(encoding)?;
    }

    let mut encoded = Vec::with_capacity(png.len().div_ceil(3) * 4);
    {
        let mut writer = base64::write::EncoderWriter::new(
            &mut encoded,
            &base64::engine::general_purpose::STANDARD,
        );
        writer
            .write_all(&png)
            .and_then(|()| writer.finish().map(drop))
            .map_err(|error| Error::Invalid(format!("base64 encoding failed: {error}")))?;
    }
    if encoded.is_empty() || encoded.len() > maximum {
        return Err(Error::Invalid(format!(
            "encoded image of {} bytes exceeds its completion bound of {maximum} bytes",
            encoded.len()
        )));
    }
    Ok(encoded)
}

fn encoding(error: png::EncodingError) -> Error {
    Error::Invalid(format!("PNG encoding failed: {error}"))
}

/// The pixels of a carried image (`ipc::Call::carried_image`) in a block of
/// `pool`: its BF16 values `[3, height, width]`, as the decoding that
/// generated it wrote them.
fn carry(image: &ipc::ImagePixels, pool: &Arc<PinnedPool>) -> Result<PreparedImage> {
    if !image.is_complete() {
        return Err(Error::Invalid(
            "a carried image's values do not fill its raster".into(),
        ));
    }
    let shape = [3, i64::from(image.height), i64::from(image.width)];
    let block = pool.lease(image.values.len())?;
    let pixels = StorageView::contiguous(block, 0, DType::BF16, &shape)?;
    // SAFETY: the block was leased for this preparation alone and holds
    // `values.len()` bytes; no device work reads it until the batch that
    // carries the image runs.
    unsafe {
        std::ptr::copy_nonoverlapping(
            image.values.as_ptr(),
            pixels.address() as *mut u8,
            image.values.len(),
        );
    }
    Ok(PreparedImage {
        facts: ImageFacts {
            shape: shape.to_vec(),
            grid: None,
            height: image.height,
            width: image.width,
        },
        pixels,
    })
}

/// One unit of image host work.
pub(crate) enum ImageWork {
    /// Decode and transform one inline image of a request into a block of
    /// `pool`, in `dtype`.
    Prepare {
        preparation: ImagePreparation,
        tower: Tower,
        payload: Arc<str>,
        input_images: u32,
        pool: Arc<PinnedPool>,
        dtype: DType,
    },
    /// Copy the decoded image product a migrated request carries
    /// (`ipc::Call::carried_image`) into a block of `pool`, as BF16 values
    /// `[3, height, width]`, which the encoder prepares as it prepares that
    /// product.
    Carry {
        image: Arc<ipc::ImagePixels>,
        pool: Arc<PinnedPool>,
    },
    /// Encode one image's quantized pixels, read back into `lease`, as a
    /// base64 PNG of at most `maximum` bytes. With `feedback`, the pixels
    /// are followed by the image product's BF16 values in that range, which
    /// follow the PNG.
    Encode {
        pixels: StorageView,
        lease: OutputLease,
        height: u32,
        width: u32,
        maximum: usize,
        feedback: Option<ipc::PixelRange>,
    },
}

/// What image host work produced.
pub(crate) enum ImageResult {
    Prepared(PreparedImage),
    Encoded(Vec<u8>),
}

/// A host-pool action of an image batch.
pub(crate) struct ImageTask {
    work: Mutex<ImageWork>,
}

impl ImageTask {
    pub(crate) fn new(work: ImageWork) -> Self {
        Self {
            work: Mutex::new(work),
        }
    }
}

impl HostAction for ImageTask {
    type Output = ImageResult;

    fn run(&self) -> Result<ImageResult> {
        let work = lock(&self.work);
        match &*work {
            ImageWork::Prepare {
                preparation,
                tower,
                payload,
                input_images,
                pool,
                dtype,
            } => preparation
                .prepare(*tower, payload, *input_images, pool, *dtype)
                .map(ImageResult::Prepared),
            ImageWork::Carry { image, pool } => carry(image, pool).map(ImageResult::Prepared),
            ImageWork::Encode {
                pixels,
                height,
                width,
                maximum,
                feedback,
                ..
            } => {
                let bytes = pixels.nbytes() as usize;
                // SAFETY: the lease's device-to-host copy of these bytes
                // completed (`ready`), and the task holds the lease's CPU
                // reader until it finishes.
                let pixels =
                    unsafe { std::slice::from_raw_parts(pixels.address() as *const u8, bytes) };
                let raster = (u64::from(*height) * u64::from(*width) * 3) as usize;
                let (rgb, values) = pixels.split_at(raster.min(pixels.len()));
                let mut encoded = png_base64(rgb, *height, *width, *maximum)?;
                if let Some(range) = feedback {
                    if values.len() as u64 != ipc::ImagePixels::value_bytes(*height, *width) {
                        return Err(Error::Invalid(format!(
                            "a {height}x{width} image's feedback values hold {} bytes",
                            values.len()
                        )));
                    }
                    ipc::append_feedback(&mut encoded, values, *range);
                }
                if encoded.len() > *maximum {
                    return Err(Error::Invalid(format!(
                        "an image's media of {} bytes exceeds its completion bound of {maximum} bytes",
                        encoded.len()
                    )));
                }
                Ok(ImageResult::Encoded(encoded))
            }
        }
    }

    fn release(&self) -> Result<()> {
        let work = lock(&self.work);
        if let ImageWork::Encode { lease, .. } = &*work {
            lock(lease).release_reader()?;
        }
        Ok(())
    }

    fn input_outcome(&self) -> Outcome {
        // Inputs are host bytes the task owns or a completed copy.
        Outcome::Success(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use base64::Engine;

    use super::*;

    fn decode(encoded: &[u8]) -> (u32, u32, Vec<u8>) {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .unwrap();
        let decoder = png::Decoder::new(Cursor::new(bytes));
        let mut reader = decoder.read_info().unwrap();
        let mut pixels = vec![0; reader.output_buffer_size().unwrap()];
        let frame = reader.next_frame(&mut pixels).unwrap();
        assert_eq!(frame.color_type, png::ColorType::Rgb);
        pixels.truncate(frame.buffer_size());
        (frame.height, frame.width, pixels)
    }

    #[test]
    fn encoded_images_decode_to_their_pixels() -> Result<()> {
        let (height, width) = (5, 7);
        let pixels: Vec<u8> = (0..height * width * 3)
            .map(|value| (value * 37 % 256) as u8)
            .collect();
        let encoded = png_base64(&pixels, height, width, 1 << 20)?;
        assert_eq!(decode(&encoded), (height, width, pixels));
        Ok(())
    }

    /// A carried image's values are staged as BF16 pixels of its raster,
    /// byte for byte; values that do not fill it are refused.
    #[test]
    fn carried_images_stage_their_values() -> Result<()> {
        let pool = PinnedPool::new(None);
        let image = ipc::ImagePixels {
            height: 2,
            width: 4,
            range: ipc::PixelRange::Unit,
            values: (0..2 * 4 * 6).map(|value| value as u8).collect(),
        };
        let prepared = carry(&image, &pool)?;
        assert_eq!(prepared.facts.shape, [3, 2, 4]);
        assert_eq!((prepared.facts.height, prepared.facts.width), (2, 4));
        assert_eq!(prepared.pixels.nbytes(), 48);
        // SAFETY: the view addresses the 48 staged bytes of a host block.
        let staged =
            unsafe { std::slice::from_raw_parts(prepared.pixels.address() as *const u8, 48) };
        assert_eq!(staged, image.values.as_slice());

        let short = ipc::ImagePixels {
            values: vec![0; 47],
            ..image
        };
        assert!(carry(&short, &pool).is_err());
        Ok(())
    }

    #[test]
    fn rasters_and_bounds_are_enforced() {
        let pixels = vec![0; 2 * 2 * 3];
        assert!(png_base64(&pixels, 2, 3, 1 << 20).is_err());
        assert!(png_base64(&pixels, 0, 2, 1 << 20).is_err());
        assert!(png_base64(&pixels, 2, 2, 16).is_err());
    }
}
