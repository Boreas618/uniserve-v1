//! The decoders' last step: a decoded raster becomes RGB bytes and a decoded waveform interleaved
//! PCM, on the device, inside the runner's walk.
//!
//! A decoder answers the model's own numbers: a normalized `[3, frames, rows, columns]` raster and a
//! planar `[channels, samples]` FP32 waveform. What a caller receives is `[frames, rows, columns, 3]`
//! bytes and `[samples, 2]` little-endian i16, which is media output and so the runtime's: the two
//! conversions are one device program each (`ops::rgb_bytes`, `ops::pcm_interleave`), launched by
//! the decoder runners right after the model's call that wrote their input, so a captured decode
//! records them with it. Nothing is rounded on the host.

use nn::Forward;
use nn::capability::PixelNorm;
use nn::layers::launch::PDL;
use ops::pcm::{self, PcmInterleave as PcmInterleaveOp};
use ops::rgb::{self, CHANNELS, RgbBytes as RgbBytesOp};
use ops::{Operand, Shape};
use platform::DeviceId;
use uniserve_tensor::DType;

use crate::runners::columns::{Column, encode_f32};
use crate::{Error, Result};

/// A view or a construction argument a conversion cannot take, named by its owner.
fn refused(owner: &'static str, problem: impl Into<String>) -> Error {
    Error::Invalid(format!("{owner}: {}", problem.into()))
}

/// The de-normalization and byte packing of a decoder's raster: `(x · std + mean) · 255`, clamped
/// into `[0, 255]` and rounded to the nearest byte with ties to even.
///
/// The constants are the decoder's [`PixelNorm`] and never change, so they live in one column of
/// this runner's, written once when it is built: a captured graph reads them where they stand
/// rather than from a launch argument uploaded per step. The geometry is the views', so no
/// descriptor field states it.
pub(crate) struct RgbBytes {
    dtype: DType,
    /// `[2 · 3]` f32: the three means, then the three standard deviations.
    constants: Column,
}

impl RgbBytes {
    /// The byte packing of a raster of `dtype` on `device`, de-normalized by `norm`.
    ///
    /// The constants are f32 whatever the raster's type, because the reference's de-normalization
    /// is f32 arithmetic.
    pub(crate) fn new(device: DeviceId, dtype: DType, norm: PixelNorm) -> Result<Self> {
        let channels = CHANNELS as usize;
        let mut constants = Column::new(device, DType::F32, 2 * u64::from(CHANNELS))?;
        let values: Vec<f32> = norm.mean.iter().chain(norm.std.iter()).copied().collect();
        debug_assert_eq!(values.len(), 2 * channels);
        let mut bytes = Vec::with_capacity(4 * values.len());
        encode_f32(&mut bytes, &values);
        constants.write(&bytes)?;
        Ok(Self { dtype, constants })
    }

    /// The `[3]` view of the means or, at `offset` three, of the standard deviations.
    fn constant(&self, offset: u64) -> Operand {
        Operand::new(
            self.constants.device.offset_elements(offset),
            DType::F32,
            Shape::new(&[u64::from(CHANNELS)]),
        )
    }

    /// Map `input [3, frames, rows, columns]` into `out [frames, rows, columns, 3]` bytes.
    ///
    /// Every extent is the views': the two shapes must describe one raster.
    pub(crate) fn forward(&self, f: &Forward, input: Operand, out: Operand) -> Result<()> {
        let channels = u64::from(CHANNELS);
        let shape = input.shape;
        if shape.rank() != 4 || shape.dim(0) != channels {
            return Err(refused(
                "RgbBytes",
                format!(
                    "{:?} is not [{channels} channels, frames, rows, columns]",
                    shape.dims()
                ),
            ));
        }
        if input.dtype != self.dtype {
            return Err(refused(
                "RgbBytes",
                format!(
                    "the raster is {}, the conversion was built for {}",
                    ops::abi::name(input.dtype),
                    ops::abi::name(self.dtype)
                ),
            ));
        }
        if out.shape.dims() != [shape.dim(1), shape.dim(2), shape.dim(3), channels]
            || out.dtype != DType::U8
        {
            return Err(refused(
                "RgbBytes",
                format!(
                    "{} {:?} is not the packed u8 form of {:?}",
                    ops::abi::name(out.dtype),
                    out.shape.dims(),
                    shape.dims()
                ),
            ));
        }
        let descriptor = rgb::Descriptor {
            dtype: self.dtype,
            channels: CHANNELS,
            block: rgb::BLOCK,
            pdl: PDL,
        };
        f.operate::<RgbBytesOp>(
            &descriptor,
            rgb::Args {
                input,
                mean: self.constant(0),
                std: self.constant(channels),
                out,
            },
        )?;
        Ok(())
    }
}

/// The channels a stereo stream interleaves.
const PCM_CHANNELS: u64 = 2;

/// Converts `planes [2, samples]` FP32 into `out [samples, 2]` i16: clamp into the i16 range, then
/// round half to even, which is the reference's own quantization.
///
/// The two channels are addressed through `planes`' own stride, so a view into a larger buffer
/// works as it stands; the sample count is the planes' last extent. FP32 only, because that is the
/// type `pcm_interleave` is compiled for.
pub(crate) fn interleave_pcm(f: &Forward, planes: Operand, out: Operand) -> Result<()> {
    if planes.shape.rank() != 2 || planes.shape.dim(0) != PCM_CHANNELS {
        return Err(refused(
            "PcmInterleave",
            format!("{:?} is not [2 channels, samples]", planes.shape.dims()),
        ));
    }
    if planes.dtype != DType::F32 {
        return Err(refused(
            "PcmInterleave",
            format!(
                "the PCM interleave reads f32 channels, not {}",
                ops::abi::name(planes.dtype)
            ),
        ));
    }
    let samples = planes.shape.dim(1);
    if out.shape.dims() != [samples, PCM_CHANNELS] || out.dtype != DType::I16 {
        return Err(refused(
            "PcmInterleave",
            format!(
                "{} {:?} is not [{samples}, 2] interleaved i16 samples",
                ops::abi::name(out.dtype),
                out.shape.dims()
            ),
        ));
    }
    // The second channel's view starts one channel stride into the planes, which is what makes a
    // strided pair work without a copy.
    let element = 4;
    let second = Operand::new(
        planes.ptr.offset(planes.strides.get(0) * element),
        DType::F32,
        Shape::new(&[samples]),
    );
    let first = Operand::new(planes.ptr, DType::F32, Shape::new(&[samples]));
    let descriptor = pcm::Descriptor {
        dtype: DType::F32,
        block: pcm::BLOCK,
        pdl: PDL,
    };
    let samples = u32::try_from(samples).map_err(|_| {
        refused(
            "PcmInterleave",
            format!("{samples} samples are wider than a kernel parameter"),
        )
    })?;
    f.operate::<PcmInterleaveOp>(
        &descriptor,
        pcm::Args {
            left: first,
            right: second,
            out,
            samples,
        },
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The packing is the reference's arithmetic: `(x · std + mean) · 255`, clamped and rounded to
    /// the nearest byte, ties to even — which this asserts against the operator's own host oracle.
    #[test]
    fn the_packing_is_the_reference_s_rounding() {
        assert_eq!(rgb::quantize(0.0, 0.5, 0.5), 128);
        assert_eq!(rgb::quantize(1.0, 0.5, 0.5), 255);
        assert_eq!(rgb::quantize(-1.0, 0.5, 0.5), 0);
        assert_eq!(rgb::quantize(0.0, 0.0, 1.0), 0);
    }

    /// A waveform the program cannot read is refused before a launch: the planes are two f32
    /// channels and the output their interleaved i16 samples.
    #[test]
    fn a_pcm_conversion_reads_two_float_channels() {
        let arena = nn::Arena::counting();
        let f = Forward::recording(&arena);
        let view = |dtype: DType, dims: &[u64]| arena.tensor(dtype, Shape::new(dims)).unwrap();
        let out = view(DType::I16, &[8, 2]);
        assert!(interleave_pcm(&f, view(DType::F32, &[2, 8]), out).is_ok());
        assert!(interleave_pcm(&f, view(DType::BF16, &[2, 8]), out).is_err());
        assert!(interleave_pcm(&f, view(DType::F32, &[1, 8]), out).is_err());
        assert!(interleave_pcm(&f, view(DType::F32, &[2, 8]), view(DType::F32, &[8, 2])).is_err());
    }
}
