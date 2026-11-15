//! The convolution, through cuDNN: one operator for every rank and direction.
//!
//! `conv` is a native program: the call names the spatial extents as three axes and says whether
//! the pass is transposed, so one module computes a 1-d or a 3-d convolution, and a transposed one
//! as cuDNN's backward-data pass of the convolution it transposes. The key names the element types
//! alone; the rank, the direction, the window, the groups, the bias, the TF32 permission and the
//! storage order travel in the call. The grid is the library's.
//!
//! An FP32 convolution may allow TF32; the call carries that permission, and the module refuses it
//! for inputs of another precision.
//!
//! ```text
//! 3-d             input [n, c, d, h, w]  weight [k, c/groups, t, h, w]   out [n, k, od, oh, ow]
//! 1-d             input [n, c, l]        weight [k, c/groups, t]         out [n, k, ol]
//! transposed 1-d  input [n, c, l]        weight [c, k/groups, t]         out [n, k, ol]
//! ```

use std::ffi::c_void;
use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::frame::is_channels_last;
use crate::kernel::{Kernel, NativeCall};
use crate::operand::Operand;
use crate::strided;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// One compilation of a convolution.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Descriptor {
    /// Element type of the input.
    pub input: DType,
    /// Element type of the filter.
    pub weight: DType,
    /// Element type of the output.
    pub output: DType,
    /// The filter's extent along each spatial axis; an axis the operator's rank leaves unused is 1.
    pub kernel: [u32; 3],
    /// The step between output positions along each spatial axis.
    pub stride: [u32; 3],
    /// The zero padding before (and after) the input along each spatial axis.
    pub padding: [u32; 3],
    /// The spacing between filter taps along each spatial axis.
    pub dilation: [u32; 3],
    /// How many channel groups the filter is split into.
    pub groups: u32,
    /// Whether the operation has a bias term.
    pub bias: bool,
    /// Whether the library may use TF32 for FP32 inputs.
    pub tf32: bool,
    /// Whether the tensors are channels-last (NDHWC) rather than channels-first (NCDHW).
    ///
    /// The shapes in the call stay logical (`[n, c, d, h, w]`); the order is the strides the entry
    /// derives and the cuDNN engine it selects, not a different compiled module - so, like the
    /// filter's window and `tf32`, it travels in the call and is not part of the key.
    pub channels_last: bool,
    /// The spatial rank: 3 for a frame's convolution, 1 for a sequence's.
    pub rank: u32,
    /// Whether the operation is a transposed convolution.
    pub transposed: bool,
}

impl fmt::Display for Descriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Only what selects a compiled module is in the key. The filter's window (kernel, stride,
        // padding, dilation, groups), the bias, the TF32 choice and the rank travel in the call
        // structure (`Call`), which the entry reads: a convolution of a different window, or with a
        // bias, or in the transposed direction, launches the same module.
        // A key that named them would compile one module per clip length and per filter shape.
        Key::new()
            .field("input", abi::name(self.input))
            .field("weight", abi::name(self.weight))
            .field("output", abi::name(self.output))
            .finish(formatter)
    }
}

impl Descriptor {
    /// The descriptor of the convolution `args` states: the operands' element types, the filter's
    /// taps along each spatial axis from the filter's extents, the spatial rank from the input's
    /// axes (`[n, c, l]` is one, `[n, c, d, h, w]` three), a bias exactly when the call has one,
    /// and channels-last storage when a three-dimensional input is stored that way. What the
    /// operands cannot say is the caller's: the direction, the window (`stride`, `padding`,
    /// `dilation` per spatial axis, 1, 0 and 1 on an axis the rank leaves unused), the channel
    /// groups and the permission to narrow FP32 inputs to TF32.
    ///
    /// This is the one place a convolution's descriptor is made: the tensor library's
    /// convolutions on this backend and `nn`'s convolutions take it from here, so one call has
    /// one descriptor whichever path launched it.
    ///
    /// # Errors
    ///
    /// [`Error::Argument`] for an input of other than one or three spatial axes, a filter of
    /// another rank than the input, or a tap count beyond 32 bits.
    pub fn of(
        args: &Args,
        transposed: bool,
        [stride, padding, dilation]: [[u32; 3]; 3],
        groups: u32,
        tf32: bool,
    ) -> Result<Self> {
        let rank = args.input.rank().saturating_sub(2);
        if !matches!(rank, 1 | 3) || args.weight.rank() != rank + 2 {
            return Err(Error::argument(
                "convolution",
                format!(
                    "an input of {:?} and a filter of {:?} are no convolution over one or three \
                     axes",
                    args.input.dims(),
                    args.weight.dims()
                ),
            ));
        }

        let mut kernel = [1_u32; 3];
        for (axis, taps) in kernel[..rank].iter_mut().enumerate() {
            let extent = args.weight.dim(axis + 2);
            *taps = u32::try_from(extent).map_err(|_| {
                Error::argument("convolution", format!("a filter of {extent} taps"))
            })?;
        }

        Ok(Self {
            input: args.input.dtype,
            weight: args.weight.dtype,
            output: args.out.dtype,
            kernel,
            stride,
            padding,
            dilation,
            groups,
            bias: args.bias.is_some(),
            tf32,
            channels_last: rank == 3 && is_channels_last(&args.input),
            rank: rank as u32,
            transposed,
        })
    }
}

/// The call structure a convolution package's entry receives.
///
/// Mirrored by hand in `crates/ops/include/uniserve_conv.h`, which the convolution package links
/// into its `include/`; the assertions below and the test beside this module pin its layout. The
/// spatial extents are passed as three axes, so one
/// structure serves a 1-d and a 3-d convolution and the unused axes are 1.
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
    /// 1 when the tensors are channels-last (NDHWC), 0 for channels-first (NCDHW).
    pub channels_last: i32,
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
// The message's offset and the structure's size are pinned on both sides - here and by the
// header's `static_assert`s - because a field added to one mirror and not the other is a kernel
// reading its geometry out of its error buffer.
const _: () = assert!(
    std::mem::offset_of!(Call, error) == 216,
    "the convolution call's message moved: the headers pin the same offset"
);
const _: () = assert!(
    std::mem::size_of::<Call>() == 728,
    "the convolution call's size must equal the headers', which pin it at 728"
);

/// The arguments of a convolution.
#[derive(Clone)]
pub struct Args {
    /// `[n, c, …]`.
    pub input: Operand,
    /// The filter.
    pub weight: Operand,
    /// `[k]`, when the operation has a bias.
    pub bias: Option<Operand>,
    /// `[n, k, …]`.
    pub out: Operand,
}

/// The layout obligations a convolution's arguments carry.
///
/// The entry receives an address and a shape for each operand and derives the strides itself,
/// so an operand laid out any other way is read at the wrong offsets while the numbers still
/// look like a convolution.
///
/// **The filter is not an operand a channels-last memory format reorders.** A checkpoint stores
/// every module's weight packed in `[out, in, kd, kh, kw]` — the KCRS form the filter descriptor
/// wants — and no framework's channels-last format touches module weights, so the entry derives
/// row-major strides for the filter whatever the activations' order is. A filter in any other
/// order is read at the wrong offsets while the numbers still look like a convolution.
///
/// **The activations and the output are.** With `channels_last` set the entry derives
/// channels-last strides for both, so a buffer allocated in the other order is misread for the same
/// reason. A unit channel axis has no order to disagree about and is exempt.
///
/// **Channels-first activations and output are contiguous.** Without `channels_last` the entry
/// derives row-major strides for both, so a permuted, stepped or broadcast view, and a block of
/// a larger tensor, are refused: the caller makes such an operand contiguous first. An axis of
/// one element carries any stride.
///
/// **The bias is contiguous**: the entry reads its elements one after the other.
fn check_layouts(operator: &'static str, descriptor: &Descriptor, args: &Args) -> Result<()> {
    if !args.weight.is_row_major() {
        return Err(Error::argument(
            operator,
            "the filter is not packed row-major: a convolution reads `[out, in, kd, kh, kw]` in \
             the order a checkpoint stores it, in either activation layout"
                .to_owned(),
        ));
    }
    if descriptor.channels_last {
        for (operand, view) in [("input", &args.input), ("out", &args.out)] {
            // A unit channel axis carries any stride's order and no other, so only an axis with
            // more than one element can be misread.
            if view.dim(1) > 1 && view.stride(1) != 1 {
                return Err(Error::argument(
                    operator,
                    format!(
                        "the descriptor declares channels-last activations, and the {operand}'s \
                         channel axis has stride {} rather than 1: the entry derives \
                         channels-last strides for these operands, so the buffer must be \
                         allocated in that order or the descriptor must say channels-first",
                        view.stride(1)
                    ),
                ));
            }
        }
    } else {
        strided::require_contiguous(operator, "input", &args.input)?;
        strided::require_contiguous(operator, "out", &args.out)?;
    }
    if let Some(bias) = &args.bias {
        strided::require_contiguous(operator, "bias", bias)?;
    }
    Ok(())
}

/// The convolution through cuDNN.
pub struct Conv;

impl Op for Conv {
    const NAME: &'static str = proto::module::operator::CONV;

    type Descriptor = Descriptor;
    type Args<'a> = Args;

    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
        let operator = Self::NAME;
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
        // The declared layout is the strides the entry derives for every operand, so a
        // channels-last module reads the channel axis innermost in all three. A caller that
        // declares channels-last over channel-first storage therefore gets numbers with nothing
        // refusing it: the library indexes by the layout it was told, not by the bytes it holds.
        // A filter is the operand this bites first, because a checkpoint stores it channel-first
        // and a channels-last span is a layout its *weights* must be permuted into once.
        check_layouts(operator, descriptor, args)?;
        if args.input.dtype != descriptor.input || args.out.dtype != descriptor.output {
            return Err(Error::argument(
                operator,
                format!(
                    "expected {} input and {} output, found {} and {}",
                    abi::name(descriptor.input),
                    abi::name(descriptor.output),
                    abi::name(args.input.dtype),
                    abi::name(args.out.dtype)
                ),
            ));
        }
        let channels = args.input.dim(1);
        let channels_per_group = channels / u64::from(descriptor.groups);
        // A filter holds one kernel axis per spatial axis: a 1-d one is `[out, in, k]`, as a checkpoint
        // stores it, and the call structure's five slots pad it with ones below.
        let (first, second) = match descriptor.transposed {
            false => (args.out.dim(1), channels_per_group),
            true => (channels, args.out.dim(1) / u64::from(descriptor.groups)),
        };
        let taps = descriptor.kernel;
        let filter = [
            first,
            second,
            u64::from(taps[0]),
            u64::from(taps[1]),
            u64::from(taps[2]),
        ];
        args.weight.require_shape(operator, &filter[..rank + 2])?;

        match (descriptor.bias, &args.bias) {
            (true, Some(bias)) => {
                bias.require_shape(operator, &[args.out.dim(1)])?;
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
            (false, None) => {}
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let rank = descriptor.rank as usize;
        // The descriptor declares a bias exactly when the call passes one.
        let bias = args.bias.as_ref().map_or(std::ptr::null(), |bias| {
            bias.ptr.address() as usize as *const c_void
        });

        let mut input_shape = [1_i64; 5];
        let mut weight_shape = [1_i64; 5];
        let mut out_shape = [1_i64; 5];
        // The spatial axes are the last `rank` axes; a 5-d structure orders them as [d, h, w] and a
        // 1-d one as [l, 1, 1], so the leading axes of the shorter ranks stay 1.
        let mut spatial = [1_u64; 3];
        for (axis, extent) in spatial.iter_mut().enumerate().take(rank) {
            *extent = args.input.dim(2 + axis);
        }
        let mut out_spatial = [1_u64; 3];
        for (axis, extent) in out_spatial.iter_mut().enumerate().take(rank) {
            *extent = args.out.dim(2 + axis);
        }
        for (target, (view, spatial)) in [
            (&mut input_shape, (&args.input, spatial)),
            (&mut out_shape, (&args.out, out_spatial)),
        ] {
            target[0] = to_i64(view.dim(0))?;
            target[1] = to_i64(view.dim(1))?;
            target[2] = to_i64(spatial[0])?;
            target[3] = to_i64(spatial[1])?;
            target[4] = to_i64(spatial[2])?;
        }
        for (axis, extent) in weight_shape.iter_mut().enumerate().take(rank + 2) {
            *extent = to_i64(args.weight.dim(axis))?;
        }

        let mut call = Call {
            out: args.out.ptr.address() as usize as *mut c_void,
            input: args.input.ptr.address() as usize as *const c_void,
            weight: args.weight.ptr.address() as usize as *const c_void,
            bias,
            input_shape,
            weight_shape,
            out_shape,
            kernel: [
                descriptor.kernel[0] as i32,
                descriptor.kernel[1] as i32,
                descriptor.kernel[2] as i32,
            ],
            stride: [
                descriptor.stride[0] as i32,
                descriptor.stride[1] as i32,
                descriptor.stride[2] as i32,
            ],
            padding: [
                descriptor.padding[0] as i32,
                descriptor.padding[1] as i32,
                descriptor.padding[2] as i32,
            ],
            dilation: [
                descriptor.dilation[0] as i32,
                descriptor.dilation[1] as i32,
                descriptor.dilation[2] as i32,
            ],
            groups: descriptor.groups as i32,
            transposed: i32::from(descriptor.transposed),
            tf32: i32::from(descriptor.tf32),
            channels_last: i32::from(descriptor.channels_last),
            error: [0; 512],
        };
        // SAFETY: `Call` is the call structure this package's entry declares, and its layout is
        // pinned by the assertions above and the test beside this module.
        unsafe { kernel.native(&mut call, stream) }
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        Args {
            out: twins.shadow_of(args.out),
            ..args
        }
    }

    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance {
        // Each output sums the taps of its input channel that reach it, in FP32, rounded once into
        // the output; the bound is the statistical one for a sum of mixed-sign products. A
        // convolution's output reads every tap of its window along each spatial axis, whatever the
        // stride, and a transposed one's reads the `kernel / stride` taps that overlap it.
        let taps: u64 = (0..descriptor.rank as usize)
            .map(|axis| {
                let kernel = u64::from(descriptor.kernel[axis]);
                match descriptor.transposed {
                    false => kernel,
                    true => kernel.div_ceil(u64::from(descriptor.stride[axis].max(1))),
                }
            })
            .product();
        Tolerance::dot(taps, DType::F32, descriptor.output)
    }
}

fn to_i64(value: u64) -> Result<i64> {
    i64::try_from(value).map_err(|_| Error::argument(Conv::NAME, "an extent above i64::MAX"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use platform::DevicePtr;

    fn view(shape: &[u64], strides: &[u64]) -> Operand {
        Operand::with_strides(DevicePtr::from_address(0x1000), DType::BF16, shape, strides)
    }

    fn descriptor(channels_last: bool) -> Descriptor {
        Descriptor {
            input: DType::BF16,
            weight: DType::BF16,
            output: DType::BF16,
            kernel: [3, 3, 3],
            stride: [1, 1, 1],
            padding: [1, 1, 1],
            dilation: [1, 1, 1],
            groups: 1,
            bias: false,
            tf32: false,
            channels_last,
            rank: 3,
            transposed: false,
        }
    }

    /// `[n, c, d, h, w]` with the channels innermost: the order a channels-last module derives.
    fn channels_last(shape: &[u64]) -> Operand {
        let (c, d, h, w) = (shape[1], shape[2], shape[3], shape[4]);
        view(shape, &[d * h * w * c, 1, h * w * c, w * c, c])
    }

    /// A channels-last convolution over channel-first activations is refused, not misread.
    #[test]
    fn channels_last_refuses_channel_first_activations() {
        let shape = [1, 4, 2, 2, 2];
        let args = Args {
            input: view(&shape, &[32, 8, 4, 2, 1]),
            weight: view(&[8, 4, 3, 3, 3], &[108, 27, 9, 3, 1]),
            bias: None,
            out: channels_last(&shape),
        };
        let error = check_layouts(Conv::NAME, &descriptor(true), &args)
            .expect_err("a channels-last module over a channel-first buffer must refuse");
        let message = error.to_string();
        assert!(
            message.contains("the input's channel axis has stride 8"),
            "{message}"
        );
    }

    /// The filter stays packed in the order a checkpoint stores it, in either layout.
    #[test]
    fn a_channels_last_convolution_keeps_the_filter_packed() {
        let shape = [1, 4, 2, 2, 2];
        let packed = view(&[8, 4, 3, 3, 3], &[108, 27, 9, 3, 1]);
        // The filter reordered into channels-last is what the kernel used to derive, and it is the
        // one argument that must not travel that way: refusing it is what keeps a convolution's
        // numbers a convolution's numbers.
        let reordered = view(&[8, 4, 3, 3, 3], &[27, 1, 9, 3, 3]);
        let args = Args {
            input: channels_last(&shape),
            weight: packed,
            bias: None,
            out: channels_last(&shape),
        };
        check_layouts(Conv::NAME, &descriptor(true), &args)
            .expect("a packed filter is the form both layouts read");
        let args = Args {
            weight: reordered,
            ..args
        };
        let error = check_layouts(Conv::NAME, &descriptor(true), &args)
            .expect_err("a filter outside the checkpoint's order must refuse");
        assert!(
            error.to_string().contains("not packed row-major"),
            "{error}"
        );
    }

    /// `[n, c, l]` operands of a one-dimensional convolution of four channels into eight.
    fn one_dimensional(input: Operand, out: Operand) -> (Descriptor, Args) {
        let descriptor = Descriptor {
            kernel: [3, 1, 1],
            padding: [1, 0, 0],
            rank: 1,
            ..descriptor(false)
        };
        let args = Args {
            input,
            weight: view(&[8, 4, 3], &[12, 3, 1]),
            bias: None,
            out,
        };
        (descriptor, args)
    }

    /// The entry derives row-major strides for channels-first operands, so any other view of
    /// the input or the output is refused, in either direction.
    #[test]
    fn a_channels_first_convolution_refuses_an_input_or_an_output_that_is_not_contiguous() {
        let input = view(&[2, 4, 16], &[64, 16, 1]);
        let out = view(&[2, 8, 16], &[128, 16, 1]);

        let (descriptor, args) = one_dimensional(input.clone(), out.clone());
        Conv::check(&descriptor, &args).unwrap();

        // `[n, l, c]` storage read as `[n, c, l]`: the channels are innermost in memory.
        let permuted = view(&[2, 4, 16], &[64, 1, 4]);
        // Every other sample of a longer signal.
        let stepped = view(&[2, 4, 16], &[128, 32, 2]);
        // One sample repeated over the batch.
        let broadcast = view(&[2, 4, 16], &[0, 16, 1]);
        // Sixteen samples of rows that are twenty long.
        let window = view(&[2, 4, 16], &[80, 20, 1]);

        for strided_input in [permuted.clone(), stepped, broadcast, window] {
            let (descriptor, args) = one_dimensional(strided_input, out.clone());
            let message = Conv::check(&descriptor, &args).unwrap_err().to_string();
            assert!(message.contains("input must be contiguous"), "{message}");
        }

        let (descriptor, args) = one_dimensional(input, view(&[2, 8, 16], &[160, 20, 1]));
        let message = Conv::check(&descriptor, &args).unwrap_err().to_string();
        assert!(message.contains("out must be contiguous"), "{message}");

        // The transposed pass reads its operands the same way: `[in, out, taps]`.
        let transposed = Descriptor {
            transposed: true,
            padding: [1, 0, 0],
            ..descriptor
        };
        let args = Args {
            input: permuted,
            weight: view(&[4, 8, 3], &[24, 3, 1]),
            bias: None,
            out,
        };
        let message = Conv::check(&transposed, &args).unwrap_err().to_string();
        assert!(message.contains("input must be contiguous"), "{message}");

        // An axis of one element is never stepped along, whatever stride it carries.
        let single = view(&[1, 4, 16], &[7, 16, 1]);
        let (descriptor, args) = one_dimensional(single, view(&[1, 8, 16], &[0, 16, 1]));
        Conv::check(&descriptor, &args).unwrap();
    }

    #[test]
    fn a_three_dimensional_convolution_refuses_a_strided_volume_and_a_strided_bias() {
        let shape = [1, 4, 2, 2, 2];
        let args = Args {
            input: view(&shape, &[32, 8, 4, 2, 1]),
            weight: view(&[8, 4, 3, 3, 3], &[108, 27, 9, 3, 1]),
            bias: None,
            out: view(&[1, 8, 2, 2, 2], &[64, 8, 4, 2, 1]),
        };
        Conv::check(&descriptor(false), &args).unwrap();

        // Depth and height exchanged.
        let permuted = Args {
            input: view(&shape, &[32, 8, 2, 4, 1]),
            ..args.clone()
        };
        let message = Conv::check(&descriptor(false), &permuted)
            .unwrap_err()
            .to_string();
        assert!(message.contains("input must be contiguous"), "{message}");

        // Every other element of a longer bias.
        let biased = Descriptor {
            bias: true,
            ..descriptor(false)
        };
        let stepped = Args {
            bias: Some(view(&[8], &[2])),
            ..args.clone()
        };
        let message = Conv::check(&biased, &stepped).unwrap_err().to_string();
        assert!(message.contains("bias must be contiguous"), "{message}");

        let packed = Args {
            bias: Some(view(&[8], &[1])),
            ..args
        };
        Conv::check(&biased, &packed).unwrap();
    }

    /// A convolution's descriptor follows from its operands and the window the caller states:
    /// the backend's `conv2d` over a depth of one, its `conv1d`, and a channels-last `conv3d`.
    #[test]
    fn a_convolutions_descriptor_follows_from_its_operands() {
        let packed =
            |shape: &[u64]| Operand::new(DevicePtr::from_address(0x1000), DType::F32, shape);
        // A unit stride and dilation and no padding on every axis.
        let unit = [[1, 1, 1], [0, 0, 0], [1, 1, 1]];

        // `conv2d` of a `[1, 4, 8, 8]` input by a `[6, 4, 3, 3]` filter, as `[n, c, 1, h, w]`.
        let planar = Args {
            input: packed(&[1, 4, 1, 8, 8]),
            weight: packed(&[6, 4, 1, 3, 3]),
            bias: None,
            out: packed(&[1, 6, 1, 4, 4]),
        };
        let stated = Descriptor::of(&planar, false, [[1, 2, 2], [0, 1, 1], [1, 1, 1]], 1, false);
        assert_eq!(
            stated.unwrap(),
            Descriptor {
                input: DType::F32,
                weight: DType::F32,
                output: DType::F32,
                kernel: [1, 3, 3],
                stride: [1, 2, 2],
                padding: [0, 1, 1],
                dilation: [1, 1, 1],
                groups: 1,
                bias: false,
                tf32: false,
                channels_last: false,
                rank: 3,
                transposed: false,
            }
        );

        // A transposed `conv1d` of a `[2, 3, 5]` input by a `[3, 4, 3]` filter, with a bias.
        let line = Args {
            input: packed(&[2, 3, 5]),
            weight: packed(&[3, 4, 3]),
            bias: Some(packed(&[4])),
            out: packed(&[2, 4, 11]),
        };
        let transposed = Descriptor::of(&line, true, [[2, 1, 1], [0, 0, 0], [1, 1, 1]], 1, true);
        let transposed = transposed.unwrap();
        assert_eq!(
            (transposed.kernel, transposed.rank, transposed.bias),
            ([3, 1, 1], 1, true)
        );
        assert!(transposed.transposed && transposed.tf32 && !transposed.channels_last);

        // A channels-last volume is stated as one, and its key is the element types alone.
        let volume = Args {
            input: channels_last(&[1, 4, 2, 6, 6]),
            weight: view(&[8, 4, 3, 3, 3], &[108, 27, 9, 3, 1]),
            bias: None,
            out: channels_last(&[1, 8, 2, 6, 6]),
        };
        let stored = Descriptor::of(&volume, false, unit, 1, false).unwrap();
        assert!(stored.channels_last);
        assert_eq!(stored.kernel, [3, 3, 3]);
        assert_eq!(stored.to_string(), "input_bf16_weight_bf16_output_bf16");

        // A two-dimensional input is no convolution the operator computes.
        let flat = Args {
            input: packed(&[1, 4, 8, 8]),
            weight: packed(&[6, 4, 3, 3]),
            bias: None,
            out: packed(&[1, 6, 6, 6]),
        };
        assert!(Descriptor::of(&flat, false, unit, 1, false).is_err());
    }
}
