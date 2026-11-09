//! The audio decoder's activation block: SnakeBeta, an upsample and a low-pass, as one call.
//!
//! Every BigVGAN-style decoder block applies the same three stages to its signal — a SnakeBeta
//! activation, a transposed depth-wise convolution that widens the signal by `ratio`, and a
//! depth-wise low-pass that brings it back. Compiling them separately would put three passes over
//! the signal and two intermediate buffers in the arena for a chain that is elementwise per sample;
//! [`ops::audio_activation`] fuses the whole span from the replicate pad to the crop, and this layer
//! is the call a model composes.
//!
//! The signal length is a **launch value**: a request's sample count varies and only the geometry —
//! channels, taps, ratio, block — is compiled. The filters are shared across channels and the
//! SnakeBeta parameters are per channel, so the weights are four published views the layer holds.

use ops::audio_activation::{self, AudioActivation as AudioActivationOp};
use ops::{DType, Shape, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::layers::launch::PDL;
use crate::weights::Weight;

/// The activation block of one audio decoder stage.
pub struct AudioActivation {
    dtype: DType,
    channels: u32,
    kernel: u32,
    ratio: u32,
    block: u32,
    weights: Option<ActivationWeights>,
}

/// The four published views one block reads.
struct ActivationWeights {
    /// `[channels]` SnakeBeta's `alpha`, in log space as the checkpoint stores it.
    alpha: Weight,
    /// `[channels]` SnakeBeta's `beta`, in log space.
    beta: Weight,
    /// `[1, 1, kernel]` the transposed convolution's filter, shared by every channel.
    filter_up: Weight,
    /// `[1, 1, kernel]` the low-pass's filter.
    filter_down: Weight,
}

impl AudioActivation {
    /// The block of a `channels`-channel decoder stage with `kernel` taps, an upsample ratio of
    /// `ratio` and `block` channels per program.
    ///
    /// The taps are even, at least two and no shorter than the ratio: the reference pads and crops
    /// by `kernel/2`, and an odd kernel would shift the phase of the low-pass.
    pub fn new(dtype: DType, channels: u32, kernel: u32, ratio: u32, block: u32) -> Result<Self> {
        if channels == 0 || ratio == 0 || block == 0 {
            return Err(Error::shape(
                "AudioActivation",
                "a block has channels, a ratio and a program width".to_owned(),
            ));
        }
        if kernel < 2 || !kernel.is_multiple_of(2) || kernel < ratio {
            return Err(Error::shape(
                "AudioActivation",
                format!(
                    "expected an even kernel of at least two taps and no shorter than the ratio \
                     {ratio}, found {kernel}"
                ),
            ));
        }
        Ok(Self {
            dtype,
            channels,
            kernel,
            ratio,
            block,
            weights: None,
        })
    }

    /// The block with its four checkpoint views: the per-channel SnakeBeta parameters and the two
    /// shared filters.
    pub fn with_weights(
        mut self,
        alpha: Weight,
        beta: Weight,
        filter_up: Weight,
        filter_down: Weight,
    ) -> Result<Self> {
        let channels = u64::from(self.channels);
        let taps = u64::from(self.kernel);
        for (name, weight) in [("alpha", &alpha), ("beta", &beta)] {
            if weight.shape().dims() != [channels] || weight.dtype() != DType::F32 {
                return Err(Error::shape(
                    "AudioActivation",
                    format!(
                        "the {name} is {channels} f32 values per channel, not {:?} {}",
                        weight.shape().dims(),
                        weight.dtype()
                    ),
                ));
            }
        }
        for (name, weight) in [("up", &filter_up), ("down", &filter_down)] {
            if weight.shape().dims() != [1, 1, taps] || weight.dtype() != self.dtype {
                return Err(Error::shape(
                    "AudioActivation",
                    format!(
                        "the {name} filter is one [1, 1, {taps}] view of {}, not {:?} {}",
                        self.dtype,
                        weight.shape().dims(),
                        weight.dtype()
                    ),
                ));
            }
        }
        self.weights = Some(ActivationWeights {
            alpha,
            beta,
            filter_up,
            filter_down,
        });
        Ok(self)
    }

    /// The descriptor a block of this geometry launches, stated without its weights.
    ///
    /// A deployment answers "which programs must be compiled for me?" before any weight is read, and
    /// the geometry is what the key names — the signal's length is a launch value and not part of it.
    pub fn descriptor_of(
        dtype: DType,
        channels: u32,
        kernel: u32,
        ratio: u32,
        block: u32,
    ) -> audio_activation::Descriptor {
        audio_activation::Descriptor {
            dtype,
            channels,
            kernel,
            ratio,
            block,
            pdl: PDL,
        }
    }

    /// The descriptor's view of the geometry, which states the expected output length.
    ///
    /// One construction with [`AudioActivation::descriptor_of`], so the walk and the launch cannot
    /// state different descriptors for the same block.
    fn descriptor(&self) -> audio_activation::Descriptor {
        Self::descriptor_of(
            self.dtype,
            self.channels,
            self.kernel,
            self.ratio,
            self.block,
        )
    }

    /// The samples one channel has after the block, for an input of `length` samples.
    #[cfg(test)]
    pub fn output_length(&self, length: u64) -> Result<u64> {
        let descriptor = self.descriptor();
        descriptor
            .output(length)
            .map_err(|error| Error::shape("AudioActivation", format!("{error}")))
    }

    /// Apply the block: `input [channels, length]` becomes `out [channels, output_length(length)]`.
    ///
    /// The filters are the geometry's and the sample count is the call's, so a clip of another
    /// length replays the captured program.
    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        let weights = self.weights.as_ref().ok_or_else(|| {
            Error::shape(
                "AudioActivation",
                "the block has no weights: build it with `with_weights` before it is called"
                    .to_owned(),
            )
        })?;
        let descriptor = self.descriptor();
        let channels = u64::from(self.channels);
        // **`[channels, samples]` or a leading batch of them.** The block is elementwise per channel, so
        // a batch is the same channel set read again - the audio decoder carries a stereo pair through
        // buffers of `[PCM_CHANNELS, channels, length]` and hands the whole view here. One launch per
        // batch row, over the same descriptor: the program is per channel and per sample count, so a
        // batch cannot change the key and must not change the arithmetic.
        let batch = match input.shape.rank() {
            2 => 1,
            3 => input.shape.dim(0),
            _ => {
                return Err(Error::shape(
                    "AudioActivation",
                    format!(
                        "{:?} is neither [{channels} channels, samples] nor \
                         [batch, {channels} channels, samples]",
                        input.shape.dims()
                    ),
                ));
            }
        };
        let channel_axis = input.shape.rank() - 2;
        if input.shape.dim(channel_axis) != channels {
            return Err(Error::shape(
                "AudioActivation",
                format!("{:?} does not have {channels} channels", input.shape.dims()),
            ));
        }
        let length = input.shape.dim(input.shape.rank() - 1);
        if length == 0 {
            return Err(Error::shape(
                "AudioActivation",
                "a signal of no samples has no activation".to_owned(),
            ));
        }
        let expected = descriptor
            .output(length)
            .map_err(|error| Error::shape("AudioActivation", format!("{error}")))?;
        // The output's tail is the same for both ranks, and its batch, when it has one, is the input's.
        let written_tail = [channels, expected];
        let tail = &out.shape.dims()[out.shape.rank().saturating_sub(2)..];
        if out.shape.rank() != input.shape.rank()
            || out.shape.rank() < 2
            || tail != written_tail
            || (batch > 1 && out.shape.dim(0) != batch)
        {
            return Err(Error::shape(
                "AudioActivation",
                format!(
                    "{:?} is not {batch} × [{channels} channels, {expected} samples], the length \
                     {length} after the ratio {}",
                    out.shape.dims(),
                    self.ratio
                ),
            ));
        }
        if input.dtype != self.dtype || out.dtype != self.dtype {
            return Err(Error::shape(
                "AudioActivation",
                format!(
                    "the block is {}, the signal is {} and the output {}",
                    self.dtype, input.dtype, out.dtype
                ),
            ));
        }
        let length = u32::try_from(length).map_err(|_| {
            Error::shape(
                "AudioActivation",
                format!("{length} samples are wider than a kernel parameter"),
            )
        })?;
        let element = self.dtype.bytes().ok_or_else(|| {
            Error::shape(
                "AudioActivation",
                format!("{} has no whole-byte element size", self.dtype),
            )
        })?;
        let input_row = channels * u64::from(length) * element;
        let out_row = channels * expected * element;
        for row in 0..batch {
            let values = TensorView::new(
                input.ptr.offset(row * input_row),
                self.dtype,
                Shape::new(&[channels, u64::from(length)]),
            );
            let written = TensorView::new(
                out.ptr.offset(row * out_row),
                self.dtype,
                Shape::new(&[channels, expected]),
            );
            f.operate::<AudioActivationOp>(
                &descriptor,
                audio_activation::Args {
                    values,
                    alpha: weights.alpha.view(),
                    beta: weights.beta.view(),
                    filter_up: weights.filter_up.view(),
                    filter_down: weights.filter_down.view(),
                    out: written,
                    length,
                },
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_taps_are_even_and_no_shorter_than_the_ratio() {
        assert!(AudioActivation::new(DType::F32, 8, 12, 2, 4).is_ok());
        assert!(AudioActivation::new(DType::F32, 8, 12, 4, 4).is_ok());
        assert!(AudioActivation::new(DType::F32, 8, 12, 16, 4).is_err());
        assert!(AudioActivation::new(DType::F32, 8, 11, 2, 4).is_err());
        assert!(AudioActivation::new(DType::F32, 8, 0, 2, 4).is_err());
        assert!(AudioActivation::new(DType::F32, 0, 12, 2, 4).is_err());
        assert!(AudioActivation::new(DType::F32, 8, 12, 0, 4).is_err());
    }

    /// The output length is the operator's own arithmetic, not a second copy of it in the layer: the
    /// chain preserves the length through the ratio, so asking the descriptor and asking the layer
    /// is the same question.
    #[test]
    fn the_output_length_is_the_descriptor_s() {
        let block = AudioActivation::new(DType::F32, 8, 12, 2, 4).unwrap();
        for length in [16_u64, 64, 480] {
            assert_eq!(
                block.output_length(length).unwrap(),
                block.descriptor().output(length).unwrap()
            );
        }
    }
}
