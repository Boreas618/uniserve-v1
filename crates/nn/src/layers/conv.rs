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
use ops::{Op, TensorView};

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
}

impl Conv3d {
    /// The convolution of `weight` (`[out, in, kd, kh, kw]` or its group form).
    pub fn new(
        weight: Weight,
        bias: Option<Weight>,
        geometry: ConvGeometry,
        tf32: bool,
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
        })
    }

    /// `out = conv3d(input)`.
    ///
    /// `input` is `[n, c, d, h, w]` and `out` `[n, k, d', h', w']`.
    pub fn forward(&self, f: &Forward, input: TensorView, out: TensorView) -> Result<()> {
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
        )
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
        )
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
        )
    }
}

/// The launch the three convolutions share.
#[allow(clippy::too_many_arguments)]
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
) -> Result<()> {
    if bias.is_some_and(|bias| bias.dtype() != out.dtype) {
        return Err(Error::shape(
            owner,
            "the bias and the output have different element types".to_owned(),
        ));
    }
    let descriptor = Descriptor {
        input: input.dtype,
        weight: weight.dtype(),
        output: out.dtype,
        geometry: geometry.descriptor(),
        bias: bias.is_some(),
        tf32,
        rank,
        transposed,
    };
    let args = Args {
        input,
        weight: weight.view(),
        bias: bias.map(Weight::view),
        out,
    };
    match (rank, transposed) {
        (3, false) => {
            let kernel = f.kernels().op::<Conv3dOp>(&descriptor)?;
            Conv3dOp::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
        }
        (1, false) => {
            let kernel = f.kernels().op::<Conv1dOp>(&descriptor)?;
            Conv1dOp::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
        }
        (1, true) => {
            let kernel = f.kernels().op::<ConvTranspose1dOp>(&descriptor)?;
            ConvTranspose1dOp::launch(kernel, &descriptor, args, f.stream).map_err(Error::from)
        }
        (rank, transposed) => Err(Error::Unsupported(format!(
            "Conv: rank {rank} transposed={transposed} is not a convolution this layer builds"
        ))),
    }
}
