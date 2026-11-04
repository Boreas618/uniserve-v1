//! The audio decoder's last step: two float channels become interleaved 16-bit PCM.
//!
//! A decoder produces `[channels, samples]` floats; what a container or a stream carries is
//! `[samples, 2]` little-endian i16. [`ops::pcm_interleave`] converts and interleaves in one program
//! — clamp into the i16 range, then round half to even, which is the reference's own quantization —
//! so the layer holds the element type and nothing else: the sample count is the call's.
//!
//! The operator reads the two channels as two separate views, so the layer splits the `[2, samples]`
//! planes it is handed by their own stride rather than assuming the pair is contiguous. Nothing is
//! rounded on the host.

use ops::pcm::{self, PcmInterleave as PcmInterleaveOp};
use ops::{DType, Shape, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;

/// The channels a stereo stream interleaves.
const CHANNELS: u64 = 2;

/// The interleave of a decoder's two output channels.
#[derive(Clone, Copy)]
pub struct PcmInterleave {
    dtype: DType,
}

impl PcmInterleave {
    /// The interleave of `dtype` float channels, which is the decoder's own output type.
    pub fn new(dtype: DType) -> Result<Self> {
        if !dtype.is_float() {
            return Err(Error::shape(
                "PcmInterleave",
                format!("a PCM conversion reads a float signal, not {dtype}"),
            ));
        }
        Ok(Self { dtype })
    }

    /// Convert `planes [2, samples]` into `out [samples, 2]` i16.
    ///
    /// The two channels are addressed through `planes`' own stride, so a view into a larger buffer
    /// works as it stands; the sample count is the planes' last extent.
    pub fn forward(&self, f: &Forward, planes: TensorView, out: TensorView) -> Result<()> {
        if planes.shape.rank() != 2 || planes.shape.dim(0) != CHANNELS {
            return Err(Error::shape(
                "PcmInterleave",
                format!("{:?} is not [2 channels, samples]", planes.shape.dims()),
            ));
        }
        if planes.dtype != self.dtype {
            return Err(Error::shape(
                "PcmInterleave",
                format!(
                    "the signal is {}, the layer was built for {}",
                    planes.dtype, self.dtype
                ),
            ));
        }
        let samples = planes.shape.dim(1);
        if out.shape.dims() != [samples, CHANNELS] {
            return Err(Error::shape(
                "PcmInterleave",
                format!(
                    "{:?} is not [{samples}, 2] interleaved samples",
                    out.shape.dims()
                ),
            ));
        }
        if out.dtype != DType::I16 {
            return Err(Error::shape(
                "PcmInterleave",
                format!("interleaved PCM is i16, not {}", out.dtype),
            ));
        }
        let element = self.dtype.bytes().ok_or_else(|| {
            Error::shape(
                "PcmInterleave",
                format!("{} has no whole-byte element size", self.dtype),
            )
        })?;
        // The second channel's view starts one channel stride into the planes, which is what makes a
        // strided pair work without a copy.
        let second = TensorView::new(
            planes.ptr.offset(planes.strides.get(0) * element),
            self.dtype,
            Shape::new(&[samples]),
        );
        let first = TensorView::new(planes.ptr, self.dtype, Shape::new(&[samples]));
        let descriptor = pcm::Descriptor {
            dtype: self.dtype,
            block: pcm::BLOCK,
            pdl: PDL,
        };
        f.operate::<PcmInterleaveOp>(
            &descriptor,
            pcm::Args {
                left: first,
                right: second,
                out,
                samples: u32::try_from(samples).map_err(|_| {
                    Error::shape(
                        "PcmInterleave",
                        format!("{samples} samples are wider than a kernel parameter"),
                    )
                })?,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pcm_conversion_reads_a_float_signal() {
        assert!(PcmInterleave::new(DType::F32).is_ok());
        assert!(PcmInterleave::new(DType::BF16).is_ok());
        assert!(PcmInterleave::new(DType::I16).is_err());
        assert!(PcmInterleave::new(DType::U8).is_err());
    }
}
