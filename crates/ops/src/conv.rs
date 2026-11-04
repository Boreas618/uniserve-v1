//! The convolutions, through cuDNN.
//!
//! `conv3d`, `conv1d` and `conv_transpose1d` are native programs: cuDNN frontend builds an engine
//! for one geometry and the package's tuning data names that engine, so the same descriptor always
//! selects the same engine and the same workspace (specs/worker.md 5.4, "cuBLASLt 和 cuDNN 的算法
//! 选择也作为调优数据记在对应的 program 包里"). The grid is the library's.
//!
//! The video VAE's FP32 convolutions keep TF32 allowed; that is a descriptor value, because it
//! changes the numerical contract of the module and therefore its key.
//!
//! ```text
//! conv3d            input [n, c, d, h, w]  weight [k, c/groups, t, h, w]   out [n, k, od, oh, ow]
//! conv1d            input [n, c, l]        weight [k, c/groups, t]         out [n, k, ol]
//! conv_transpose1d  input [n, c, l]        weight [c, k/groups, t]         out [n, k, ol]
//! ```

use std::ffi::c_void;
use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Kernel, NativeCall};
use crate::platform::{self, Stream};
use crate::tensor::{DType, Shape, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The geometry a convolution module is compiled for.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Geometry {
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

impl fmt::Display for Geometry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field(
                "kernel",
                format_args!("{}x{}x{}", self.kernel[0], self.kernel[1], self.kernel[2]),
            )
            .field(
                "stride",
                format_args!("{}x{}x{}", self.stride[0], self.stride[1], self.stride[2]),
            )
            .field(
                "padding",
                format_args!(
                    "{}x{}x{}",
                    self.padding[0], self.padding[1], self.padding[2]
                ),
            )
            .field(
                "dilation",
                format_args!(
                    "{}x{}x{}",
                    self.dilation[0], self.dilation[1], self.dilation[2]
                ),
            )
            .field("groups", self.groups)
            .finish(formatter)
    }
}

/// One compilation of a convolution.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the input.
    pub input: DType,
    /// Element type of the filter.
    pub weight: DType,
    /// Element type of the output.
    pub output: DType,
    /// The convolution's geometry.
    pub geometry: Geometry,
    /// Whether the operation has a bias term.
    pub bias: bool,
    /// Whether the library may use TF32 for FP32 inputs.
    pub tf32: bool,
    /// The spatial rank: 3 for `conv3d`, 1 for the one-dimensional pair.
    pub rank: u32,
    /// Whether the operation is a transposed convolution.
    pub transposed: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("input", self.input)
            .field("weight", self.weight)
            .field("output", self.output)
            .field("geometry", self.geometry)
            .field("bias", self.bias)
            .field("tf32", self.tf32)
            .field("rank", self.rank)
            .field("transposed", self.transposed)
            .finish(formatter)
    }
}

/// The call structure a convolution package's entry receives.
///
/// Mirrored by hand in the package's header (`include/uniserve_conv.h`); the assertions below and
/// the test beside this module pin its layout. The spatial extents are passed as three axes, so one
/// structure serves the 1-d and the 3-d operator and the unused axes are 1.
#[repr(C)]
pub struct Call {
    pub out: *mut c_void,
    pub input: *const c_void,
    pub weight: *const c_void,
    /// Null when the operation has no bias.
    pub bias: *const c_void,
    /// `[n, c, d, h, w]` of the input; `c` is the input's channel count.
    pub input_shape: [i64; 5],
    /// `[k, c/groups, t, h, w]` of the filter.
    pub weight_shape: [i64; 5],
    /// `[n, k, d, h, w]` of the output.
    pub out_shape: [i64; 5],
    pub kernel: [i32; 3],
    pub stride: [i32; 3],
    pub padding: [i32; 3],
    pub dilation: [i32; 3],
    pub groups: i32,
    /// 0 for a plain convolution, 1 for a transposed one.
    pub transposed: i32,
    /// 1 when the library may use TF32.
    pub tf32: i32,
    pub error: [u8; 512],
}

impl NativeCall for Call {
    fn message(&self) -> Option<&str> {
        let end = self.error.iter().position(|&byte| byte == 0)?;
        std::str::from_utf8(&self.error[..end]).ok()
    }
}

const _: () = assert!(
    std::mem::offset_of!(Call, input_shape) == 32,
    "the convolution call structure's geometry moved without its header"
);
const _: () = assert!(
    std::mem::offset_of!(Call, kernel) == 32 + 3 * 5 * 8,
    "the convolution call structure's filter moved without its header"
);

/// The arguments of a convolution.
#[derive(Clone, Copy)]
pub struct Args {
    /// `[n, c, …]`.
    pub input: TensorView,
    /// The filter.
    pub weight: TensorView,
    /// `[k]`, when the operation has a bias.
    pub bias: Option<TensorView>,
    /// `[n, k, …]`.
    pub out: TensorView,
}

/// A convolution through cuDNN.
pub struct Conv;

impl Conv {
    /// The launch shared by the three operators, which differ only in their descriptor's geometry.
    fn launch(
        operator: &'static str,
        kernel: &Kernel,
        descriptor: &Descriptor,
        args: Args,
        stream: &Stream,
    ) -> Result<()> {
        let rank = descriptor.rank;
        if !(1..=3).contains(&rank) {
            return Err(Error::argument(
                operator,
                format!("a convolution has 1 to 3 spatial axes, not {rank}"),
            ));
        }
        let rank = rank as usize;
        args.input.require_rank(operator, rank + 2)?;
        args.out.require_rank(operator, rank + 2)?;
        if args.input.dtype != descriptor.input || args.out.dtype != descriptor.output {
            return Err(Error::argument(
                operator,
                format!(
                    "expected {} input and {} output, found {} and {}",
                    descriptor.input, descriptor.output, args.input.dtype, args.out.dtype
                ),
            ));
        }
        let channels = args.input.shape.dim(1);
        let channels_per_group = channels / u64::from(descriptor.geometry.groups);
        match descriptor.transposed {
            false => args.weight.require_shape(
                operator,
                Shape::new(&[
                    args.out.shape.dim(1),
                    channels_per_group,
                    u64::from(descriptor.geometry.kernel[0]),
                    u64::from(descriptor.geometry.kernel[1]),
                    u64::from(descriptor.geometry.kernel[2]),
                ]),
            )?,
            true => args.weight.require_shape(
                operator,
                Shape::new(&[
                    channels,
                    args.out.shape.dim(1) / u64::from(descriptor.geometry.groups),
                    u64::from(descriptor.geometry.kernel[0]),
                    u64::from(descriptor.geometry.kernel[1]),
                    u64::from(descriptor.geometry.kernel[2]),
                ]),
            )?,
        }

        let bias = match (descriptor.bias, args.bias) {
            (true, Some(bias)) => {
                bias.require_shape(operator, Shape::new(&[args.out.shape.dim(1)]))?;
                platform::address(bias.ptr) as usize as *const c_void
            }
            (true, None) => {
                return Err(Error::argument(
                    operator,
                    "the descriptor declares a bias and the call passed none",
                ));
            }
            (false, Some(_)) => {
                return Err(Error::argument(
                    operator,
                    "the call passed a bias the descriptor does not declare",
                ));
            }
            (false, None) => std::ptr::null(),
        };

        let mut input_shape = [1_i64; 5];
        let mut weight_shape = [1_i64; 5];
        let mut out_shape = [1_i64; 5];
        // The spatial axes are the last `rank` axes; a 5-d structure orders them as [d, h, w] and a
        // 1-d one as [l, 1, 1], so the leading axes of the shorter ranks stay 1.
        let mut spatial = [1_u64; 3];
        for (axis, extent) in spatial.iter_mut().enumerate().take(rank) {
            *extent = args.input.shape.dim(2 + axis);
        }
        let mut out_spatial = [1_u64; 3];
        for (axis, extent) in out_spatial.iter_mut().enumerate().take(rank) {
            *extent = args.out.shape.dim(2 + axis);
        }
        for (target, (extents, spatial)) in [
            (&mut input_shape, (&args.input.shape, spatial)),
            (&mut out_shape, (&args.out.shape, out_spatial)),
        ] {
            target[0] = to_i64(extents.dim(0))?;
            target[1] = to_i64(extents.dim(1))?;
            target[2] = to_i64(spatial[0])?;
            target[3] = to_i64(spatial[1])?;
            target[4] = to_i64(spatial[2])?;
        }
        weight_shape[0] = to_i64(args.weight.shape.dim(0))?;
        weight_shape[1] = to_i64(args.weight.shape.dim(1))?;
        weight_shape[2] = to_i64(args.weight.shape.dim(2))?;
        weight_shape[3] = to_i64(args.weight.shape.dim(3))?;
        weight_shape[4] = to_i64(args.weight.shape.dim(4))?;

        let geometry = descriptor.geometry;
        let mut call = Call {
            out: platform::address(args.out.ptr) as usize as *mut c_void,
            input: platform::address(args.input.ptr) as usize as *const c_void,
            weight: platform::address(args.weight.ptr) as usize as *const c_void,
            bias,
            input_shape,
            weight_shape,
            out_shape,
            kernel: [
                geometry.kernel[0] as i32,
                geometry.kernel[1] as i32,
                geometry.kernel[2] as i32,
            ],
            stride: [
                geometry.stride[0] as i32,
                geometry.stride[1] as i32,
                geometry.stride[2] as i32,
            ],
            padding: [
                geometry.padding[0] as i32,
                geometry.padding[1] as i32,
                geometry.padding[2] as i32,
            ],
            dilation: [
                geometry.dilation[0] as i32,
                geometry.dilation[1] as i32,
                geometry.dilation[2] as i32,
            ],
            groups: geometry.groups as i32,
            transposed: i32::from(descriptor.transposed),
            tf32: i32::from(descriptor.tf32),
            error: [0; 512],
        };
        // SAFETY: `Call` is the call structure this package's entry declares, and its layout is
        // pinned by the assertions above and the test beside this module.
        unsafe { kernel.native(&mut call, stream) }
    }
}

fn to_i64(value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::argument("conv", "an extent above i64::MAX"))
}

/// A 3-d convolution: the video VAE's spatial convolutions.
pub struct Conv3d;

impl Op for Conv3d {
    const NAME: &'static str = "conv3d";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if descriptor.rank != 3 || descriptor.transposed {
            return Err(Error::argument(
                Self::NAME,
                "the descriptor is not a 3-d convolution",
            ));
        }
        Conv::launch(Self::NAME, kernel, descriptor, args, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // The contraction is over the filter's taps times its input channels; the bound is the
        // statistical one for a sum of mixed-sign products.
        let taps: u64 = descriptor
            .geometry
            .kernel
            .iter()
            .take(descriptor.rank as usize)
            .map(|&extent| u64::from(extent))
            .product();
        Tolerance::dot(taps, DType::F32, descriptor.output)
    }
}

/// A 1-d convolution: the audio decoder's and the VAE's temporal convolutions.
pub struct Conv1d;

impl Op for Conv1d {
    const NAME: &'static str = "conv1d";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if descriptor.rank != 1 || descriptor.transposed {
            return Err(Error::argument(
                Self::NAME,
                "the descriptor is not a 1-d convolution",
            ));
        }
        Conv::launch(Self::NAME, kernel, descriptor, args, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        Tolerance::dot(
            u64::from(descriptor.geometry.kernel[0]),
            DType::F32,
            descriptor.output,
        )
    }
}

/// A transposed 1-d convolution: the audio decoder's upsampling.
pub struct ConvTranspose1d;

impl Op for ConvTranspose1d {
    const NAME: &'static str = "conv_transpose1d";

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        if descriptor.rank != 1 || !descriptor.transposed {
            return Err(Error::argument(
                Self::NAME,
                "the descriptor is not a transposed 1-d convolution",
            ));
        }
        Conv::launch(Self::NAME, kernel, descriptor, args, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
        let _ = stream;
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Each output sums the taps that overlap it: `kernel / stride` contributions.
        let stride = u64::from(descriptor.geometry.stride[0].max(1));
        let taps = u64::from(descriptor.geometry.kernel[0]);
        Tolerance::dot(taps.div_ceil(stride), DType::F32, descriptor.output)
    }
}
