//! Convolutions through cuDNN: the video VAE's `conv3d` and the audio decoder's one-dimensional
//! pair.
//!
//! A convolution's geometry is part of its descriptor, because the program is compiled per shape
//! family and selects a cuDNN engine for it from its tuning data (specs/worker.md 5.4, 9). The
//! layer holds the geometry the model states and calls the matching operator; the descriptor's
//! `tf32` flag is the model's numerical declaration for an FP32 convolution (the video VAE keeps it
//! on, as the `quality` preset does).

use ops::conv::{
    Args, Conv1d as Conv1dOp, Conv3d as Conv3dOp, ConvTranspose1d as ConvTranspose1dOp, Descriptor,
    Geometry,
};
use ops::{DType, Shape, Strides, TensorView};

use crate::error::{Error, Result};
use crate::forward::Forward;
use crate::weights::Weight;

/// One convolution's geometry, as the model states it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ConvGeometry {
    /// The filter's extent along each spatial axis.
    pub kernel: [u32; 3],
    /// The step between output positions along each spatial axis.
    pub stride: [u32; 3],
    /// The padding before the first input position along each spatial axis.
    pub padding: [u32; 3],
    /// The spacing between filter taps along each spatial axis.
    pub dilation: [u32; 3],
    /// How many input channels each output channel's filter reads.
    pub groups: u32,
}

impl ConvGeometry {
    /// The geometry of a `rank`-dimensional convolution: the trailing axes of the three-axis form.
    pub fn new(
        rank: usize,
        kernel: [u32; 3],
        stride: [u32; 3],
        padding: [u32; 3],
        dilation: [u32; 3],
        groups: u32,
    ) -> Result<Self> {
        if rank != 1 && rank != 3 {
            return Err(Error::shape(
                "ConvGeometry",
                format!("rank {rank} is not a 1-d or 3-d convolution"),
            ));
        }
        if groups == 0 {
            return Err(Error::shape(
                "ConvGeometry",
                "a convolution has at least one group".to_owned(),
            ));
        }
        Ok(Self {
            kernel,
            stride,
            padding,
            dilation,
            groups,
        })
    }

    /// The geometry as the operator's descriptor states it.
    fn descriptor(self) -> Geometry {
        Geometry {
            kernel: self.kernel,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
        }
    }
}

/// A three-dimensional convolution.
pub struct Conv3d {
    weight: Weight,
    bias: Option<Weight>,
    geometry: ConvGeometry,
    tf32: bool,
    /// Whether this convolution reads and writes channels-last (NDHWC) tensors.
    ///
    /// **The filter is not part of the order.** A checkpoint stores a convolution's weight
    /// `[out, in, kd, kh, kw]` packed, which is the KCRS form cuDNN's filter descriptor wants, and
    /// PyTorch's `channels_last_3d` is a memory format of *activations* - module weights stay dense.
    /// So a channels-last convolution reorders its input and output only.
    /// `crates/ops/kernels/conv3d/cudnn/src/conv.cu:383-386` currently derives NDHWC strides for the
    /// filter as well, which reads a packed KCRS weight in the wrong order; no site passes `true`
    /// today, so the flag is refused by the guard below before that line can run.
    ///
    /// The reference's stride-1 padded convolutions read channels-last and its strided ones
    /// channels-first, because the two orders can select different algorithms and, for a strided
    /// convolution, different rounding. **Stated per layer, never inferred from the stride**: a 1x1
    /// stride-1 convolution is channels-first in the reference (`conv_shortcut`, `quant_conv`), and
    /// deriving the order from the stride gets exactly those wrong.
    channels_last: bool,
}

/// The strides of a channels-last (`NDHWC`) buffer read as its logical `[n, c, d, h, w]` shape.
///
/// The channel axis runs fastest, then the spatial axes from the innermost out, then the batch —
/// the same order `conv3d`'s kernel derives from the descriptor's flag.
fn channels_last_strides(shape: Shape) -> Strides {
    let (channels, depth) = (shape.dim(1), shape.dim(2));
    let (rows, columns) = (shape.dim(3), shape.dim(4));
    let plane = columns * channels;
    let body = rows * plane;
    Strides::new(&[depth * body, 1, body, plane, channels])
}

/// Refuse a channels-last convolution whose operand is not stored channels-last.
///
/// The kernel derives every operand's strides from the descriptor's `channels_last` flag alone, so
/// the flag **is** the layout: a convolution declared channels-last over a channels-first buffer
/// reads the same bytes in another order and produces wrong numbers with no error anywhere. A model
/// that wants the order but not the storage needs a permute in front of the convolution, and this is
/// where that gap is named rather than measured.
fn channels_last_storage(
    owner: &'static str,
    operand: &'static str,
    view: TensorView,
) -> Result<()> {
    if view.shape.rank() != 5 {
        return Err(Error::shape(
            owner,
            format!(
                "a channels-last convolution takes 5-d operands, and its {operand} is {:?}",
                view.shape.dims()
            ),
        ));
    }
    let expected = channels_last_strides(view.shape);
    if view.strides != expected {
        return Err(Error::shape(
            owner,
            format!(
                "the {operand} of a channels-last convolution must be stored NDHWC: its strides are \
                 {:?} over {:?}, and the packed channels-last layout is {:?}. The kernel derives the \
                 strides from the descriptor, so a channels-first buffer would be read in the wrong \
                 order",
                view.strides.as_slice(),
                view.shape.dims(),
                expected.as_slice()
            ),
        ));
    }
    Ok(())
}

impl Conv3d {
    /// The convolution of `weight` (`[out, in, kd, kh, kw]` or its group form).
    pub fn new(
        weight: Weight,
        bias: Option<Weight>,
        geometry: ConvGeometry,
        tf32: bool,
        channels_last: bool,
    ) -> Result<Self> {
        if weight.shape().rank() != 5 {
            return Err(Error::shape(
                "Conv3d",
                format!("a filter is 5-d, not {:?}", weight.shape().dims()),
            ));
        }
        Ok(Self {
            weight,
            bias,
            geometry,
            tf32,
            channels_last,
        })
    }

    /// `out = conv3d(input)`.
    ///
    /// `input` is `[n, c, d, h, w]` and `out` `[n, k, d', h', w']`.
    ///
    /// A convolution built channels-last takes both operands **stored** channels-last, which a
    /// caller states by viewing an `NDHWC` buffer as its logical shape; the check is here because
    /// the kernel reads the order from the descriptor and would otherwise misread the bytes.
    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        if self.channels_last {
            channels_last_storage("Conv3d", "input", input)?;
            channels_last_storage("Conv3d", "out", out)?;
        }
        launch(
            f,
            "conv3d",
            input,
            &self.weight,
            self.bias.as_ref(),
            out,
            self.geometry,
            3,
            false,
            self.tf32,
            self.channels_last,
        )
    }

    /// The descriptor this layer's launch takes, for a walk that asks a model which programs it
    /// launches with no device.
    pub fn descriptor(&self, input: DType, output: DType) -> Descriptor {
        descriptor_of(
            input,
            &self.weight,
            self.bias.as_ref(),
            output,
            self.geometry,
            3,
            false,
            self.tf32,
            self.channels_last,
        )
    }

    /// The descriptor a three-dimensional convolution's launch, stated from its geometry and element types alone.
    ///
    /// A deployment answers "which programs must be compiled for me?" before any weight exists, so
    /// the construction stays beside the launch it describes rather than being restated by the walk.
    #[allow(clippy::too_many_arguments)]
    pub fn descriptor_of(
        input: DType,
        weight: DType,
        bias: bool,
        output: DType,
        geometry: ConvGeometry,
        tf32: bool,
        channels_last: bool,
    ) -> Descriptor {
        Descriptor {
            input,
            weight,
            output,
            geometry: geometry.descriptor(),
            bias,
            tf32,
            channels_last,
            rank: 3,
            transposed: false,
        }
    }
}

/// A one-dimensional convolution (the audio decoder's convolutions).
pub struct Conv1d {
    weight: Weight,
    bias: Option<Weight>,
    geometry: ConvGeometry,
    tf32: bool,
}

impl Conv1d {
    /// The convolution of `weight` (`[out, in, k]` or its group form).
    pub fn new(
        weight: Weight,
        bias: Option<Weight>,
        geometry: ConvGeometry,
        tf32: bool,
    ) -> Result<Self> {
        if weight.shape().rank() != 3 {
            return Err(Error::shape(
                "Conv1d",
                format!("a filter is 3-d, not {:?}", weight.shape().dims()),
            ));
        }
        Ok(Self {
            weight,
            bias,
            geometry,
            tf32,
        })
    }

    /// `out = conv1d(input)` on `[n, c, l]` sequences.
    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        launch(
            f,
            "conv1d",
            input,
            &self.weight,
            self.bias.as_ref(),
            out,
            self.geometry,
            1,
            false,
            self.tf32,
            false,
        )
    }

    /// The descriptor this layer's launch takes; see [`Conv3d::descriptor`].
    pub fn descriptor(&self, input: DType, output: DType) -> Descriptor {
        descriptor_of(
            input,
            &self.weight,
            self.bias.as_ref(),
            output,
            self.geometry,
            1,
            false,
            self.tf32,
            false,
        )
    }

    /// The descriptor a one-dimensional convolution's launch, stated from its geometry and element types alone.
    ///
    /// A deployment answers "which programs must be compiled for me?" before any weight exists, so
    /// the construction stays beside the launch it describes rather than being restated by the walk.
    #[allow(clippy::too_many_arguments)]
    pub fn descriptor_of(
        input: DType,
        weight: DType,
        bias: bool,
        output: DType,
        geometry: ConvGeometry,
        tf32: bool,
    ) -> Descriptor {
        Descriptor {
            input,
            weight,
            output,
            geometry: geometry.descriptor(),
            bias,
            tf32,
            channels_last: false,
            rank: 1,
            transposed: false,
        }
    }
}

/// A one-dimensional transposed convolution (the audio decoder's upsampling).
pub struct ConvTranspose1d {
    weight: Weight,
    bias: Option<Weight>,
    geometry: ConvGeometry,
    tf32: bool,
}

impl ConvTranspose1d {
    /// The transposed convolution of `weight` (`[in, out, k]` or its group form).
    pub fn new(
        weight: Weight,
        bias: Option<Weight>,
        geometry: ConvGeometry,
        tf32: bool,
    ) -> Result<Self> {
        if weight.shape().rank() != 3 {
            return Err(Error::shape(
                "ConvTranspose1d",
                format!("a filter is 3-d, not {:?}", weight.shape().dims()),
            ));
        }
        Ok(Self {
            weight,
            bias,
            geometry,
            tf32,
        })
    }

    /// `out = conv_transpose1d(input)` on `[n, c, l]` sequences.
    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
        launch(
            f,
            "conv_transpose1d",
            input,
            &self.weight,
            self.bias.as_ref(),
            out,
            self.geometry,
            1,
            true,
            self.tf32,
            false,
        )
    }

    /// The descriptor this layer's launch takes; see [`Conv3d::descriptor`].
    pub fn descriptor(&self, input: DType, output: DType) -> Descriptor {
        descriptor_of(
            input,
            &self.weight,
            self.bias.as_ref(),
            output,
            self.geometry,
            1,
            true,
            self.tf32,
            false,
        )
    }

    /// The descriptor a transposed one-dimensional convolution's launch, stated from its geometry and element types alone.
    ///
    /// A deployment answers "which programs must be compiled for me?" before any weight exists, so
    /// the construction stays beside the launch it describes rather than being restated by the walk.
    #[allow(clippy::too_many_arguments)]
    pub fn descriptor_of(
        input: DType,
        weight: DType,
        bias: bool,
        output: DType,
        geometry: ConvGeometry,
        tf32: bool,
    ) -> Descriptor {
        Descriptor {
            input,
            weight,
            output,
            geometry: geometry.descriptor(),
            bias,
            tf32,
            channels_last: false,
            rank: 1,
            transposed: true,
        }
    }
}

/// The descriptor one convolution launch takes.
///
/// The three layers share one launch, so they share the construction of its descriptor: an accessor
/// per layer that built it separately would be three chances for one to drift from the operator it
/// names, and the walk that asks a model for its programs reads exactly this.
#[allow(
    clippy::too_many_arguments,
    reason = "every argument is a geometry or an element type the descriptor states separately"
)]
fn descriptor_of(
    input: DType,
    weight: &Weight,
    bias: Option<&Weight>,
    output: DType,
    geometry: ConvGeometry,
    rank: u32,
    transposed: bool,
    tf32: bool,
    channels_last: bool,
) -> Descriptor {
    Descriptor {
        input,
        weight: weight.dtype(),
        output,
        geometry: geometry.descriptor(),
        bias: bias.is_some(),
        tf32,
        channels_last,
        rank,
        transposed,
    }
}

/// The launch the three convolutions share.
#[allow(
    clippy::too_many_arguments,
    reason = "the descriptor's own arguments plus the launch's views; one helper keeps the three \
              layers' launches identical"
)]
fn launch(
    f: &Forward,
    owner: &'static str,
    input: TensorView,
    weight: &Weight,
    bias: Option<&Weight>,
    out: TensorView,
    geometry: ConvGeometry,
    rank: u32,
    transposed: bool,
    tf32: bool,
    channels_last: bool,
) -> Result<()> {
    if bias.is_some_and(|bias| bias.dtype() != out.dtype) {
        return Err(Error::shape(
            owner,
            "the bias and the output have different element types".to_owned(),
        ));
    }
    let descriptor = descriptor_of(
        input.dtype,
        weight,
        bias,
        out.dtype,
        geometry,
        rank,
        transposed,
        tf32,
        channels_last,
    );
    let args = Args {
        input,
        weight: weight.view(),
        bias: bias.map(Weight::view),
        out,
    };
    match (rank, transposed) {
        (3, false) => f.operate::<Conv3dOp>(&descriptor, args),
        (1, false) => f.operate::<Conv1dOp>(&descriptor, args),
        (1, true) => f.operate::<ConvTranspose1dOp>(&descriptor, args),
        (rank, transposed) => Err(Error::Unsupported(format!(
            "Conv: rank {rank} transposed={transposed} is not a convolution this layer builds"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::{channels_last_storage, channels_last_strides};
    use ops::tensor::{DType, DevicePtr, Shape, Strides, TensorView};

    /// The encoder's first convolution's operand: 3 channels, 19 frames, 258 x 258 pixels.
    fn operand(strides: Strides) -> TensorView {
        TensorView::with_strides(
            DevicePtr::new(0x1_0000),
            DType::F32,
            Shape::new(&[1, 3, 19, 258, 258]),
            strides,
        )
    }

    #[test]
    fn a_channels_last_operand_is_packed_ndhwc() {
        // The kernel's own order: channel fastest, then width, height, depth, batch.
        let shape = Shape::new(&[1, 3, 19, 258, 258]);
        assert_eq!(
            channels_last_strides(shape).as_slice(),
            &[3 * 19 * 258 * 258, 1, 258 * 258 * 3, 258 * 3, 3]
        );
    }

    #[test]
    fn a_channels_first_buffer_is_refused_by_a_channels_last_convolution() {
        let shape = Shape::new(&[1, 3, 19, 258, 258]);
        // Packed channels-first: the layout every other layer of the model builds.
        let error = channels_last_storage("Conv3d", "input", operand(Strides::row_major(shape)))
            .expect_err("a channels-first buffer read channels-last is wrong numbers, not a plan");
        let text = format!("{error}");
        assert!(text.contains("must be stored NDHWC"), "{text}");

        // The same bytes viewed channels-last are accepted, which is what the encoder must build.
        channels_last_storage("Conv3d", "input", operand(channels_last_strides(shape)))
            .expect("the packed channels-last layout is what the descriptor promises");
    }
}
