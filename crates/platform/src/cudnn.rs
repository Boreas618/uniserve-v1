//! cuDNN bindings (`libcudnn.so.9`), loaded at run time: the video VAE's
//! `conv3d` and the audio decoder's `conv1d` (`specs/worker.md` §9), with the
//! engine choice recorded in the program's tuning data (§5.4).
//!
//! cuDNN is bound here and nowhere else, through its backend API: a
//! convolution is described by three tensor descriptors, a convolution
//! descriptor, the forward operation and its operation graph; the engine
//! that runs it is chosen once from cuDNN's heuristic while the program's
//! tuning data is made, and an execution plan is built from that engine and
//! kept for the shape's life. Service does no measurement.
//!
//! # The engine choice as tuning data
//!
//! cuDNN names an engine by its **global index**, an integer that the loaded
//! cuDNN picks a fixed meaning for: the set of engines and what each computes
//! is a property of the cuDNN version, which the program's revision pins
//! together with its source and toolchain (§5.2). The index is therefore the
//! stable identity of a choice, small enough to sit in the descriptor key of
//! the program's package, and [`Engine`] travels as its eight bytes
//! ([`Engine::to_bytes`], [`Engine::from_bytes`]).
//!
//! The heuristic is asked again only at program load, never in service:
//! [`Cudnn::plan`] with a recorded [`Engine`] looks that engine up among the
//! heuristic's proposals for the shape. If the loaded cuDNN proposes no such
//! engine (a version change that the revision did not pin), the plan fails
//! with the index it looked for instead of silently running another engine.
//!
//! The workspace a convolution needs is a property of the engine and the
//! shape, not a separate choice: the recorded engine is the whole record, and
//! [`ConvolutionPlan::workspace_bytes`] answers the fixed size the plan
//! allocates.
//!
//! # Descriptors
//!
//! cuDNN's backend API hands out no descriptor that the caller has not
//! created: `cudnnBackendCreateDescriptor` allocates one and
//! `cudnnBackendGetAttribute` **fills** descriptors the caller created and
//! passed in, which is how the engine configurations of a heuristic result
//! and the engine of an engine configuration are read (cuDNN's own frontend
//! does the same). This module therefore owns every descriptor it creates and
//! destroys each exactly once; the values a caller sees are plain data.
//!
//! # Precision
//!
//! The multiplication is expressed with an FP32 compute type, which keeps the
//! video VAE's precision while allowing cuDNN to use TF32 tensor cores for
//! it: the engine's numerical notes report whether it down-converts inputs,
//! and the current implementation keeps that precision (§9). The audio
//! decoder's convolutions use the same FP32 type.
//!
//! # Streams and graphs
//!
//! Every call enters the device's primary context on the calling thread, so
//! the caller needs no current context of its own, and [`Cudnn::run`] issues
//! the convolution on the caller's stream, so a captured graph holds it like
//! any other work.

use std::ffi::{CStr, c_char, c_void};
use std::sync::{Arc, OnceLock};

use libloading::Library;

use crate::driver::Handle;
use crate::{DType, Device, DeviceId, Error, Result, Stream};

const LIBCUDNN: &str = "libcudnn.so.9";

/// `cudnnStatus_t`.
type Status = i32;
const STATUS_SUCCESS: Status = 0;

/// `cudnnBackendDescriptorType_t`: the kinds of descriptor this module
/// creates.
const DESCRIPTOR_CONVOLUTION: i32 = 1;
const DESCRIPTOR_ENGINE: i32 = 2;
const DESCRIPTOR_ENGINE_CONFIG: i32 = 3;
const DESCRIPTOR_ENGINE_HEURISTIC: i32 = 4;
const DESCRIPTOR_EXECUTION_PLAN: i32 = 5;
const DESCRIPTOR_OPERATION_CONVOLUTION_FORWARD: i32 = 10;
const DESCRIPTOR_OPERATION_GRAPH: i32 = 15;
const DESCRIPTOR_VARIANT_PACK: i32 = 16;
const DESCRIPTOR_TENSOR: i32 = 17;

/// `cudnnBackendAttributeName_t`: the attributes this module sets and reads.
const ATTR_CONVOLUTION_COMP_TYPE: i32 = 100;
const ATTR_CONVOLUTION_CONV_MODE: i32 = 101;
const ATTR_CONVOLUTION_DILATIONS: i32 = 102;
const ATTR_CONVOLUTION_FILTER_STRIDES: i32 = 103;
const ATTR_CONVOLUTION_POST_PADDINGS: i32 = 104;
const ATTR_CONVOLUTION_PRE_PADDINGS: i32 = 105;
const ATTR_CONVOLUTION_SPATIAL_DIMS: i32 = 106;
const ATTR_ENGINEHEUR_MODE: i32 = 200;
const ATTR_ENGINEHEUR_OPERATION_GRAPH: i32 = 201;
const ATTR_ENGINEHEUR_RESULTS: i32 = 202;
const ATTR_ENGINECFG_ENGINE: i32 = 300;
const ATTR_EXECUTION_PLAN_ENGINE_CONFIG: i32 = 401;
const ATTR_EXECUTION_PLAN_WORKSPACE_SIZE: i32 = 402;
const ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA: i32 = 700;
const ATTR_OPERATION_CONVOLUTION_FORWARD_BETA: i32 = 701;
const ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC: i32 = 702;
const ATTR_OPERATION_CONVOLUTION_FORWARD_W: i32 = 703;
const ATTR_OPERATION_CONVOLUTION_FORWARD_X: i32 = 704;
const ATTR_OPERATION_CONVOLUTION_FORWARD_Y: i32 = 705;
const ATTR_OPERATIONGRAPH_OPS: i32 = 801;
const ATTR_TENSOR_DATA_TYPE: i32 = 901;
const ATTR_TENSOR_DIMENSIONS: i32 = 902;
const ATTR_TENSOR_STRIDES: i32 = 903;
const ATTR_TENSOR_UNIQUE_ID: i32 = 906;
const ATTR_VARIANT_PACK_UNIQUE_IDS: i32 = 1000;
const ATTR_VARIANT_PACK_DATA_POINTERS: i32 = 1001;
const ATTR_VARIANT_PACK_WORKSPACE: i32 = 1003;
const ATTR_ENGINE_GLOBAL_INDEX: i32 = 1301;

/// `cudnnBackendAttributeType_t`: the C type of each attribute's value.
const TYPE_DATA_TYPE: i32 = 1;
const TYPE_INT64: i32 = 3;
const TYPE_FLOAT: i32 = 4;
const TYPE_VOID_PTR: i32 = 6;
const TYPE_CONVOLUTION_MODE: i32 = 7;
const TYPE_HEUR_MODE: i32 = 8;
const TYPE_BACKEND_DESCRIPTOR: i32 = 15;

/// `CUDNN_HEUR_MODE_A`: cuDNN's own heuristic, which measures nothing and
/// returns the engines it knows for the shape, best first. This is how the
/// tuning data is made (§5.4); service never calls it.
const HEURISTIC_MODE_A: i32 = 3;
/// `CUDNN_CROSS_CORRELATION`: the convolution PyTorch's `Conv*d` computes.
const CROSS_CORRELATION: i32 = 1;

/// The most engine configurations one heuristic query is asked to return.
const MAX_ENGINE_CONFIGS: usize = 32;

/// The unique ids that tie the three tensors of the operation graph to the
/// pointers of a variant pack, in the order [`Cudnn::run`] passes them.
const UID_X: i64 = 0;
const UID_W: i64 = 1;
const UID_Y: i64 = 2;

/// A cuDNN backend descriptor (`cudnnBackendDescriptor_t`), owned by this
/// module: created here, destroyed here, exactly once.
struct Descriptor {
    api: &'static Api,
    raw: Handle,
}

impl Descriptor {
    fn create(api: &'static Api, kind: i32) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        // SAFETY: the library writes one descriptor handle to the local.
        let status = unsafe { (api.backend_create_descriptor)(kind, &mut raw) };
        api.check(status, "cudnnBackendCreateDescriptor")?;
        Ok(Self { api, raw })
    }
}

impl Drop for Descriptor {
    fn drop(&mut self) {
        // SAFETY: the descriptor was created here, is live, and is destroyed
        // once.
        unsafe { (self.api.backend_destroy_descriptor)(self.raw) };
    }
}

/// `cudnnBackendGetAttribute`'s signature: it writes the element count and
/// the values.
type GetAttribute = unsafe extern "C" fn(Handle, i32, i32, i64, *mut i64, *mut c_void) -> Status;

/// The cuDNN entry points this module uses, resolved once per process.
struct Api {
    get_version: unsafe extern "C" fn() -> usize,
    create: unsafe extern "C" fn(*mut Handle) -> Status,
    destroy: unsafe extern "C" fn(Handle) -> Status,
    set_stream: unsafe extern "C" fn(Handle, Handle) -> Status,
    backend_create_descriptor: unsafe extern "C" fn(i32, *mut Handle) -> Status,
    backend_destroy_descriptor: unsafe extern "C" fn(Handle) -> Status,
    backend_set_attribute: unsafe extern "C" fn(Handle, i32, i32, i64, *const c_void) -> Status,
    backend_get_attribute: GetAttribute,
    backend_finalize: unsafe extern "C" fn(Handle) -> Status,
    backend_execute: unsafe extern "C" fn(Handle, Handle, Handle) -> Status,
    get_last_error: unsafe extern "C" fn(*mut c_char, usize) -> Status,
    // Declared last: the library outlives every copied function pointer.
    _library: Library,
}

fn api() -> Result<&'static Api> {
    static API: OnceLock<Result<Api>> = OnceLock::new();

    API.get_or_init(|| {
        // SAFETY: loading cuDNN runs only its own initializers.
        let library = unsafe { Library::new(LIBCUDNN) }.map_err(|source| Error::Load {
            library: LIBCUDNN,
            source: Arc::new(source),
        })?;
        // SAFETY: each signature matches the `cudnn_graph.h` prototype of the
        // symbol.
        unsafe {
            Ok(Api {
                get_version: symbol(&library, "cudnnGetVersion")?,
                create: symbol(&library, "cudnnCreate")?,
                destroy: symbol(&library, "cudnnDestroy")?,
                set_stream: symbol(&library, "cudnnSetStream")?,
                backend_create_descriptor: symbol(&library, "cudnnBackendCreateDescriptor")?,
                backend_destroy_descriptor: symbol(&library, "cudnnBackendDestroyDescriptor")?,
                backend_set_attribute: symbol(&library, "cudnnBackendSetAttribute")?,
                backend_get_attribute: symbol(&library, "cudnnBackendGetAttribute")?,
                backend_finalize: symbol(&library, "cudnnBackendFinalize")?,
                backend_execute: symbol(&library, "cudnnBackendExecute")?,
                get_last_error: symbol(&library, "cudnnGetLastErrorString")?,
                _library: library,
            })
        }
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// # Safety
/// `T` must be the function-pointer type of the symbol's C prototype.
unsafe fn symbol<T: Copy>(library: &Library, name: &'static str) -> Result<T> {
    // SAFETY: forwarded from the caller.
    unsafe { library.get::<T>(name.as_bytes()) }
        .map(|symbol| *symbol)
        .map_err(|source| Error::Symbol {
            library: LIBCUDNN,
            symbol: name,
            source: Arc::new(source),
        })
}

impl Api {
    fn check(&self, status: Status, call: &'static str) -> Result<()> {
        if status == STATUS_SUCCESS {
            return Ok(());
        }
        let mut message = [0 as c_char; 256];
        // SAFETY: the library writes at most `message.len()` bytes; the
        // buffer outlives the call.
        unsafe { (self.get_last_error)(message.as_mut_ptr(), message.len()) };
        // SAFETY: the library fills the buffer with a NUL-terminated string.
        let text = unsafe { CStr::from_ptr(message.as_ptr()) }
            .to_str()
            .unwrap_or("non-UTF-8 cuDNN error");
        Err(Error::Cudnn {
            call,
            code: status,
            message: text.to_owned(),
        })
    }

    /// Sets one attribute from `values`, whose C type the caller names.
    fn set<T>(
        &self,
        descriptor: &Descriptor,
        attribute: i32,
        kind: i32,
        values: &[T],
        call: &'static str,
    ) -> Result<()> {
        // SAFETY: `values` holds `values.len()` elements of the attribute's
        // documented C type, and both outlive the call; the descriptor is
        // live.
        let status = unsafe {
            (self.backend_set_attribute)(
                descriptor.raw,
                attribute,
                kind,
                values.len() as i64,
                values.as_ptr().cast(),
            )
        };
        self.check(status, call)
    }

    /// Sets one attribute from a single value.
    fn set_one<T>(
        &self,
        descriptor: &Descriptor,
        attribute: i32,
        kind: i32,
        value: &T,
        call: &'static str,
    ) -> Result<()> {
        self.set(
            descriptor,
            attribute,
            kind,
            std::slice::from_ref(value),
            call,
        )
    }

    /// Reads an attribute into `values`, whose C type the caller names, and
    /// answers how many elements the library wrote.
    fn get<T>(
        &self,
        descriptor: &Descriptor,
        attribute: i32,
        kind: i32,
        values: &mut [T],
        call: &'static str,
    ) -> Result<usize> {
        let mut written = 0i64;
        // SAFETY: `values` has room for `values.len()` elements of the
        // attribute's documented C type, and both outlive the call; the
        // descriptor is live.
        let status = unsafe {
            (self.backend_get_attribute)(
                descriptor.raw,
                attribute,
                kind,
                values.len() as i64,
                &mut written,
                values.as_mut_ptr().cast(),
            )
        };
        self.check(status, call)?;
        Ok((written.max(0) as usize).min(values.len()))
    }

    /// Reads the element count of an attribute that holds a list.
    fn count(&self, descriptor: &Descriptor, attribute: i32, kind: i32) -> Result<usize> {
        let mut written = 0i64;
        // SAFETY: an element count of zero asks for the list's length and
        // writes no element; the local outlives the call and the descriptor
        // is live.
        let status = unsafe {
            (self.backend_get_attribute)(
                descriptor.raw,
                attribute,
                kind,
                0,
                &mut written,
                std::ptr::null_mut(),
            )
        };
        self.check(status, "cudnnBackendGetAttribute")?;
        Ok(written.max(0) as usize)
    }

    fn finalize(&self, descriptor: &Descriptor, call: &'static str) -> Result<()> {
        // SAFETY: the descriptor is live and fully described.
        self.check(unsafe { (self.backend_finalize)(descriptor.raw) }, call)
    }
}

/// One tensor of a convolution (`cudnnBackendTensorDescriptor_t`): its
/// dimensions and strides in cuDNN's NCHW order (`N`, `C`, then the spatial
/// dimensions), its element type, and the unique id that ties it to a data
/// pointer in the variant pack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tensor {
    pub dims: Vec<i64>,
    pub strides: Vec<i64>,
    pub dtype: DType,
}

impl Tensor {
    /// A tensor with exactly the dimensions it has, whose strides are the
    /// row-major (contiguous) strides of those dimensions.
    pub fn contiguous(dims: Vec<i64>, dtype: DType) -> Self {
        let mut strides = vec![1i64; dims.len()];
        for index in (0..dims.len().saturating_sub(1)).rev() {
            strides[index] = strides[index + 1].saturating_mul(dims[index + 1]);
        }
        Self {
            dims,
            strides,
            dtype,
        }
    }

    /// The number of dimensions: `N` and `C` plus the spatial ones.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    fn check(&self) -> Result<()> {
        if self.dims.len() < 3 {
            return Err(Error::invalid(format!(
                "a convolution tensor needs a batch, a channel and at least one spatial dimension, not {:?}",
                self.dims
            )));
        }
        if self.dims.len() != self.strides.len() {
            return Err(Error::invalid(format!(
                "a convolution tensor of rank {} has {} strides",
                self.dims.len(),
                self.strides.len()
            )));
        }
        if self.dims.iter().any(|&dim| dim <= 0) {
            return Err(Error::invalid(format!(
                "a convolution tensor has an empty dimension: {:?}",
                self.dims
            )));
        }
        if self.dtype.cudnn() < 0 {
            return Err(Error::invalid(format!(
                "{:?} has no cuDNN element type",
                self.dtype
            )));
        }
        Ok(())
    }
}

/// A convolution's description: the input (`NCDHW`), filter (`KCDHW`) and
/// output (`NKDHW`) tensors, the padding, filter strides and dilations of its
/// spatial dimensions, and the compute type.
///
/// The spatial rank is the tensors' own: a rank-three input describes a
/// `conv1d` (the audio decoder), a rank-five input a `conv3d` (the video
/// VAE), and everything in between.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Convolution {
    pub x: Tensor,
    pub w: Tensor,
    pub y: Tensor,
    /// Padding before each spatial dimension.
    pub padding: Vec<i64>,
    /// Filter stride in each spatial dimension.
    pub stride: Vec<i64>,
    /// Filter dilation in each spatial dimension.
    pub dilation: Vec<i64>,
    /// The accumulation type; FP32 keeps the video VAE's precision while
    /// allowing TF32.
    pub compute: DType,
}

impl Convolution {
    /// A forward cross-correlation of `x` with `w` into `y` — what PyTorch's
    /// `Conv1d` and `Conv3d` compute — with one padding, stride and dilation
    /// per spatial dimension.
    pub fn new(
        x: Tensor,
        w: Tensor,
        y: Tensor,
        padding: Vec<i64>,
        stride: Vec<i64>,
        dilation: Vec<i64>,
    ) -> Self {
        Self {
            x,
            w,
            y,
            padding,
            stride,
            dilation,
            compute: DType::F32,
        }
    }

    /// The number of spatial dimensions.
    pub fn rank(&self) -> usize {
        self.x.rank().saturating_sub(2)
    }

    /// The bytes each of the three tensors spans, for checking the caller's
    /// buffers.
    pub fn bytes(&self) -> ConvolutionBytes {
        let element = |tensor: &Tensor| {
            let elements = tensor
                .dims
                .iter()
                .try_fold(1i64, |total, &dim| total.checked_mul(dim))
                .unwrap_or(0);
            elements.max(0) as u64 * tensor.dtype.bytes() as u64
        };
        ConvolutionBytes {
            x: element(&self.x),
            w: element(&self.w),
            y: element(&self.y),
        }
    }

    fn check(&self) -> Result<()> {
        self.x.check()?;
        self.w.check()?;
        self.y.check()?;
        let rank = self.rank();
        if self.w.rank() != self.x.rank() || self.y.rank() != self.x.rank() {
            return Err(Error::invalid(format!(
                "a convolution's input, filter and output need the same rank, not {}, {} and {}",
                self.x.rank(),
                self.w.rank(),
                self.y.rank()
            )));
        }
        for (name, values) in [
            ("padding", &self.padding),
            ("stride", &self.stride),
            ("dilation", &self.dilation),
        ] {
            if values.len() != rank {
                return Err(Error::invalid(format!(
                    "a {rank}-dimensional convolution needs {rank} {name} values, not {}",
                    values.len()
                )));
            }
        }
        if self.stride.iter().any(|&value| value <= 0) {
            return Err(Error::invalid(format!(
                "a convolution's stride is positive, not {:?}",
                self.stride
            )));
        }
        if self.dilation.iter().any(|&value| value <= 0) {
            return Err(Error::invalid(format!(
                "a convolution's dilation is positive, not {:?}",
                self.dilation
            )));
        }
        if self.padding.iter().any(|&value| value < 0) {
            return Err(Error::invalid(format!(
                "a convolution's padding is not negative, not {:?}",
                self.padding
            )));
        }
        if self.w.dims[1] != self.x.dims[1] {
            return Err(Error::invalid(format!(
                "the filter spans {} input channels, the input has {}",
                self.w.dims[1], self.x.dims[1]
            )));
        }
        if self.y.dims[0] != self.x.dims[0] || self.y.dims[1] != self.w.dims[0] {
            return Err(Error::invalid(format!(
                "the output is {} by {}, the batch is {} and the filter produces {} channels",
                self.y.dims[0], self.y.dims[1], self.x.dims[0], self.w.dims[0]
            )));
        }
        for index in 0..rank {
            let dim = 2 + index;
            let expected = (self.x.dims[dim] + 2 * self.padding[index]
                - self.dilation[index] * (self.w.dims[dim] - 1)
                - 1)
                / self.stride[index]
                + 1;
            if self.y.dims[dim] != expected {
                return Err(Error::invalid(format!(
                    "spatial dimension {index} of the output is {} where the input, filter, padding, stride and dilation give {expected}",
                    self.y.dims[dim]
                )));
            }
        }
        if self.compute != DType::F32 {
            return Err(Error::invalid(format!(
                "a convolution accumulates in FP32, not {:?}",
                self.compute
            )));
        }
        Ok(())
    }
}

/// The bytes each tensor of a convolution spans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvolutionBytes {
    pub x: u64,
    pub w: u64,
    pub y: u64,
}

/// The engine that runs a convolution: cuDNN's global index for it, which is
/// the program's tuning data for this shape (§5.4).
///
/// The index is the choice's stable identity for a pinned cuDNN version; see
/// the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Engine {
    global_index: i64,
}

impl Engine {
    /// The engine cuDNN's global index names.
    pub fn new(global_index: i64) -> Self {
        Self { global_index }
    }

    /// The index cuDNN knows this engine by.
    pub fn global_index(self) -> i64 {
        self.global_index
    }

    /// The engine as the tuning data's eight bytes: the index, little-endian,
    /// in a fixed order so a package written on one host reads on another.
    pub fn to_bytes(self) -> [u8; 8] {
        self.global_index.to_le_bytes()
    }

    /// The engine recorded in the tuning data's eight bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let raw: [u8; 8] = bytes.try_into().map_err(|_| {
            Error::invalid(format!("an engine record is 8 bytes, not {}", bytes.len()))
        })?;
        let global_index = i64::from_le_bytes(raw);
        if global_index < 0 {
            return Err(Error::invalid(format!(
                "an engine record names no engine: {global_index}"
            )));
        }
        Ok(Self { global_index })
    }
}

/// The descriptors one convolution's operation graph is built from, kept
/// alive together: the three tensors, the convolution description and the
/// forward operation it belongs to, and the graph itself.
///
/// Declared in dependency order: the graph is destroyed first and the tensors
/// it refers to last.
struct Operation {
    graph: Descriptor,
    // Held only for their lifetime: the graph refers to the forward
    // operation, which refers to the convolution description and the three
    // tensors.
    _forward: Descriptor,
    _convolution: Descriptor,
    _x: Descriptor,
    _w: Descriptor,
    _y: Descriptor,
}

impl Operation {
    fn describe(api: &'static Api, convolution: &Convolution) -> Result<Self> {
        let x = Self::tensor(api, &convolution.x, UID_X)?;
        let w = Self::tensor(api, &convolution.w, UID_W)?;
        let y = Self::tensor(api, &convolution.y, UID_Y)?;

        let descriptor = Descriptor::create(api, DESCRIPTOR_CONVOLUTION)?;
        let compute = convolution.compute.cudnn();
        let mode = CROSS_CORRELATION;
        let spatial = convolution.rank() as i64;
        let call = "cudnnBackendSetAttribute";
        api.set_one(
            &descriptor,
            ATTR_CONVOLUTION_COMP_TYPE,
            TYPE_DATA_TYPE,
            &compute,
            call,
        )?;
        api.set_one(
            &descriptor,
            ATTR_CONVOLUTION_CONV_MODE,
            TYPE_CONVOLUTION_MODE,
            &mode,
            call,
        )?;
        api.set_one(
            &descriptor,
            ATTR_CONVOLUTION_SPATIAL_DIMS,
            TYPE_INT64,
            &spatial,
            call,
        )?;
        // A symmetric padding: the padding before and after each spatial
        // dimension are the convolution layer's own, so the output keeps the
        // input's shape under stride one.
        api.set(
            &descriptor,
            ATTR_CONVOLUTION_PRE_PADDINGS,
            TYPE_INT64,
            &convolution.padding,
            call,
        )?;
        api.set(
            &descriptor,
            ATTR_CONVOLUTION_POST_PADDINGS,
            TYPE_INT64,
            &convolution.padding,
            call,
        )?;
        api.set(
            &descriptor,
            ATTR_CONVOLUTION_FILTER_STRIDES,
            TYPE_INT64,
            &convolution.stride,
            call,
        )?;
        api.set(
            &descriptor,
            ATTR_CONVOLUTION_DILATIONS,
            TYPE_INT64,
            &convolution.dilation,
            call,
        )?;
        api.finalize(&descriptor, "cudnnBackendFinalize")?;

        let forward = Descriptor::create(api, DESCRIPTOR_OPERATION_CONVOLUTION_FORWARD)?;
        // `y = alpha · conv(x, w) + beta · y`: FP32 scalars, the scale type of
        // an FP32 convolution.
        let (alpha, beta) = (1.0f32, 0.0f32);
        api.set_one(
            &forward,
            ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
            TYPE_FLOAT,
            &alpha,
            call,
        )?;
        api.set_one(
            &forward,
            ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
            TYPE_FLOAT,
            &beta,
            call,
        )?;
        for (attribute, tensor) in [
            (ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, &descriptor),
            (ATTR_OPERATION_CONVOLUTION_FORWARD_X, &x),
            (ATTR_OPERATION_CONVOLUTION_FORWARD_W, &w),
            (ATTR_OPERATION_CONVOLUTION_FORWARD_Y, &y),
        ] {
            api.set_one(
                &forward,
                attribute,
                TYPE_BACKEND_DESCRIPTOR,
                &tensor.raw,
                call,
            )?;
        }
        api.finalize(&forward, "cudnnBackendFinalize")?;

        let graph = Descriptor::create(api, DESCRIPTOR_OPERATION_GRAPH)?;
        api.set_one(
            &graph,
            ATTR_OPERATIONGRAPH_OPS,
            TYPE_BACKEND_DESCRIPTOR,
            &forward.raw,
            call,
        )?;
        api.finalize(&graph, "cudnnBackendFinalize")?;

        Ok(Self {
            graph,
            _forward: forward,
            _convolution: descriptor,
            _x: x,
            _w: w,
            _y: y,
        })
    }

    fn tensor(api: &'static Api, tensor: &Tensor, uid: i64) -> Result<Descriptor> {
        let descriptor = Descriptor::create(api, DESCRIPTOR_TENSOR)?;
        let dtype = tensor.dtype.cudnn();
        let call = "cudnnBackendSetAttribute";
        api.set_one(
            &descriptor,
            ATTR_TENSOR_DATA_TYPE,
            TYPE_DATA_TYPE,
            &dtype,
            call,
        )?;
        api.set(
            &descriptor,
            ATTR_TENSOR_DIMENSIONS,
            TYPE_INT64,
            &tensor.dims,
            call,
        )?;
        api.set(
            &descriptor,
            ATTR_TENSOR_STRIDES,
            TYPE_INT64,
            &tensor.strides,
            call,
        )?;
        api.set_one(&descriptor, ATTR_TENSOR_UNIQUE_ID, TYPE_INT64, &uid, call)?;
        api.finalize(&descriptor, "cudnnBackendFinalize")?;
        Ok(descriptor)
    }
}

/// A convolution with its engine chosen and its execution plan built: what a
/// layer keeps per shape and reuses for every call.
///
/// A plan belongs to the [`Cudnn`] handle that built it: it is used by that
/// handle's thread and dropped before the handle, whose context the
/// descriptors live in.
///
/// The plan's workspace need follows from the engine and the shape, so the
/// program's tuning data records only the engine ([`Engine`]); the workspace
/// is read back here whenever the plan is built and is the process's fixed
/// size for the shape.
///
/// Declared in dependency order: the plan is destroyed first, then the engine
/// configuration it was built from, then the operation graph.
pub struct ConvolutionPlan {
    plan: Descriptor,
    engine: Engine,
    workspace_bytes: usize,
    device: DeviceId,
    // Held only for their lifetime: the plan refers to the engine
    // configuration it was built from, which the heuristic's operation graph
    // produced.
    _config: Descriptor,
    _operation: Operation,
}

impl ConvolutionPlan {
    /// The engine the plan runs.
    pub fn engine(&self) -> Engine {
        self.engine
    }

    /// The workspace the plan needs, in bytes: the program's fixed workspace
    /// for this shape (§5.4).
    pub fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    /// The device the plan was built for.
    pub fn device(&self) -> DeviceId {
        self.device
    }
}

/// The device pointers of one execution of a convolution.
///
/// Every pointer addresses device storage the caller keeps alive until the
/// stream work completes, and that stays at its address for a captured
/// graph's life.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvolutionArgs {
    pub x: u64,
    pub w: u64,
    pub y: u64,
    /// The workspace's address; at least [`ConvolutionPlan::workspace_bytes`].
    pub workspace: u64,
}

/// A cuDNN handle for one device's primary context.
///
/// A handle is used by one thread at a time: the executor creates one per
/// device on that device's thread and keeps it.
pub struct Cudnn {
    api: &'static Api,
    device: DeviceId,
    handle: Handle,
}

// SAFETY: a cuDNN handle is used by one thread at a time; moving it between
// threads is safe because every call enters the device's context.
unsafe impl Send for Cudnn {}

impl Cudnn {
    /// Creates a cuDNN handle for `device`.
    ///
    /// # Errors
    ///
    /// [`Error::Load`] when `libcudnn.so.9` is absent, [`Error::Symbol`] when
    /// it predates an entry point, [`Error::Cudnn`] when the handle cannot be
    /// created.
    pub fn new(device: &Device) -> Result<Self> {
        let api = api()?;
        let _context = device.enter()?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: the library writes one handle to the local.
        api.check(unsafe { (api.create)(&mut handle) }, "cudnnCreate")?;
        Ok(Self {
            api,
            device: device.id(),
            handle,
        })
    }

    /// The device the handle belongs to.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The cuDNN version this process loaded (`cudnnGetVersion`).
    pub fn version(&self) -> usize {
        // SAFETY: takes no arguments and only reads the library's version.
        unsafe { (self.api.get_version)() }
    }

    /// The engines cuDNN's heuristic proposes for `convolution`, best first,
    /// at most `limit` of them.
    ///
    /// This is how the tuning data is made: the program's package runs it
    /// once on an idle GPU (§5.4) and records one [`Engine`] per shape.
    pub fn engines(&self, convolution: &Convolution, limit: usize) -> Result<Vec<Engine>> {
        convolution.check()?;
        let device = Device::new(self.device)?;
        let _context = device.enter()?;
        let operation = Operation::describe(self.api, convolution)?;
        let configs = self.configs(&operation, limit)?;
        if configs.is_empty() {
            return Err(Error::invalid(
                "cuDNN's heuristic proposes no engine for this convolution",
            ));
        }
        configs
            .iter()
            .map(|config| self.global_index(config))
            .collect()
    }

    /// Builds the execution plan for `convolution`, running the recorded
    /// `engine` when one is given (the program's tuning data) or the
    /// heuristic's first choice otherwise.
    ///
    /// The heuristic is asked once here, at program load; service reuses the
    /// plan.
    pub fn plan(
        &self,
        convolution: &Convolution,
        engine: Option<Engine>,
    ) -> Result<ConvolutionPlan> {
        convolution.check()?;
        let device = Device::new(self.device)?;
        let _context = device.enter()?;
        let operation = Operation::describe(self.api, convolution)?;
        let mut configs = self.configs(&operation, MAX_ENGINE_CONFIGS)?;
        if configs.is_empty() {
            return Err(Error::invalid(
                "cuDNN's heuristic proposes no engine for this convolution",
            ));
        }
        let position = match engine {
            Some(recorded) => {
                let mut position = None;
                for (index, config) in configs.iter().enumerate() {
                    if self.global_index(config)? == recorded {
                        position = Some(index);
                        break;
                    }
                }
                position.ok_or_else(|| {
                    Error::invalid(format!(
                        "cuDNN's heuristic proposes no engine {} for this convolution; \
                         the shape has to be measured again against this cuDNN version",
                        recorded.global_index
                    ))
                })?
            }
            None => 0,
        };
        let chosen = configs.swap_remove(position);
        let recorded = self.global_index(&chosen)?;

        let plan = Descriptor::create(self.api, DESCRIPTOR_EXECUTION_PLAN)?;
        self.api.set_one(
            &plan,
            ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
            TYPE_BACKEND_DESCRIPTOR,
            &chosen.raw,
            "cudnnBackendSetAttribute",
        )?;
        self.api.finalize(&plan, "cudnnBackendFinalize")?;
        let mut workspace_bytes = 0i64;
        self.api.get(
            &plan,
            ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
            TYPE_INT64,
            std::slice::from_mut(&mut workspace_bytes),
            "cudnnBackendGetAttribute",
        )?;
        if workspace_bytes < 0 {
            return Err(Error::invalid(format!(
                "cuDNN reports a negative workspace of {workspace_bytes} bytes for engine {}",
                recorded.global_index
            )));
        }

        Ok(ConvolutionPlan {
            plan,
            engine: recorded,
            workspace_bytes: workspace_bytes as usize,
            device: self.device,
            _config: chosen,
            _operation: operation,
        })
    }

    /// Runs `plan` on `args`, issued on `stream`.
    ///
    /// The call issues device work and returns without waiting for it; on a
    /// capturing stream it is captured into the graph.
    ///
    /// # Safety
    ///
    /// `args` must address device storage that outlives every execution of
    /// the convolution: each tensor at least [`Convolution::bytes`] long and
    /// the workspace at least [`ConvolutionPlan::workspace_bytes`].
    pub unsafe fn run(
        &self,
        plan: &ConvolutionPlan,
        args: ConvolutionArgs,
        stream: &Stream,
    ) -> Result<()> {
        if plan.device != self.device {
            return Err(Error::invalid(format!(
                "the plan was built for device {:?}, the handle belongs to {:?}",
                plan.device, self.device
            )));
        }
        if args.x == 0 || args.w == 0 || args.y == 0 {
            return Err(Error::invalid(
                "a convolution needs the addresses of its input, filter and output",
            ));
        }
        if plan.workspace_bytes > 0 && args.workspace == 0 {
            return Err(Error::invalid(format!(
                "the plan needs {} bytes of workspace, no address was given",
                plan.workspace_bytes
            )));
        }
        let device = Device::new(self.device)?;
        let _context = device.enter()?;
        // SAFETY: the handle and the stream are live.
        let status = unsafe { (self.api.set_stream)(self.handle, stream.handle()) };
        self.api.check(status, "cudnnSetStream")?;

        // The variant pack is built per execution, as cuDNN's own frontend
        // does: it holds the argument values of this call only, and a graph
        // replay resends them through the captured node.
        let variant = Descriptor::create(self.api, DESCRIPTOR_VARIANT_PACK)?;
        let pointers: [*mut c_void; 3] = [
            args.x as *mut c_void,
            args.w as *mut c_void,
            args.y as *mut c_void,
        ];
        let uids: [i64; 3] = [UID_X, UID_W, UID_Y];
        let workspace = args.workspace as *mut c_void;
        let call = "cudnnBackendSetAttribute";
        self.api.set(
            &variant,
            ATTR_VARIANT_PACK_DATA_POINTERS,
            TYPE_VOID_PTR,
            &pointers,
            call,
        )?;
        self.api.set(
            &variant,
            ATTR_VARIANT_PACK_UNIQUE_IDS,
            TYPE_INT64,
            &uids,
            call,
        )?;
        self.api.set_one(
            &variant,
            ATTR_VARIANT_PACK_WORKSPACE,
            TYPE_VOID_PTR,
            &workspace,
            call,
        )?;
        self.api.finalize(&variant, "cudnnBackendFinalize")?;
        // SAFETY: the handle, the execution plan and the variant pack are
        // live for the call, and the pointers they name are the caller's
        // device storage.
        self.api.check(
            unsafe { (self.api.backend_execute)(self.handle, plan.plan.raw, variant.raw) },
            "cudnnBackendExecute",
        )
    }

    /// The engine configurations cuDNN's heuristic proposes for `operation`,
    /// best first: descriptors this module created for the library to fill,
    /// so it owns and destroys them.
    fn configs(&self, operation: &Operation, limit: usize) -> Result<Vec<Descriptor>> {
        let heuristic = Descriptor::create(self.api, DESCRIPTOR_ENGINE_HEURISTIC)?;
        let mode = HEURISTIC_MODE_A;
        let call = "cudnnBackendSetAttribute";
        self.api.set_one(
            &heuristic,
            ATTR_ENGINEHEUR_OPERATION_GRAPH,
            TYPE_BACKEND_DESCRIPTOR,
            &operation.graph.raw,
            call,
        )?;
        self.api.set_one(
            &heuristic,
            ATTR_ENGINEHEUR_MODE,
            TYPE_HEUR_MODE,
            &mode,
            call,
        )?;
        self.api.finalize(&heuristic, "cudnnBackendFinalize")?;

        let available =
            self.api
                .count(&heuristic, ATTR_ENGINEHEUR_RESULTS, TYPE_BACKEND_DESCRIPTOR)?;
        let wanted = available.min(limit.clamp(1, MAX_ENGINE_CONFIGS));
        let mut configs = Vec::with_capacity(wanted);
        for _ in 0..wanted {
            configs.push(Descriptor::create(self.api, DESCRIPTOR_ENGINE_CONFIG)?);
        }
        if configs.is_empty() {
            return Ok(configs);
        }
        let mut handles: Vec<Handle> = configs.iter().map(|config| config.raw).collect();
        let written = self.api.get(
            &heuristic,
            ATTR_ENGINEHEUR_RESULTS,
            TYPE_BACKEND_DESCRIPTOR,
            &mut handles,
            "cudnnBackendGetAttribute",
        )?;
        // Keep the configurations the library filled; the rest are destroyed
        // with their `Descriptor`.
        configs.truncate(written);
        Ok(configs)
    }

    /// The engine a finalized engine configuration names.
    fn global_index(&self, config: &Descriptor) -> Result<Engine> {
        let engine = Descriptor::create(self.api, DESCRIPTOR_ENGINE)?;
        let mut handles = [engine.raw];
        let written = self.api.get(
            config,
            ATTR_ENGINECFG_ENGINE,
            TYPE_BACKEND_DESCRIPTOR,
            &mut handles,
            "cudnnBackendGetAttribute",
        )?;
        if written == 0 || handles[0].is_null() {
            return Err(Error::invalid(
                "a cuDNN engine configuration names no engine",
            ));
        }
        let mut index = 0i64;
        self.api.get(
            &engine,
            ATTR_ENGINE_GLOBAL_INDEX,
            TYPE_INT64,
            std::slice::from_mut(&mut index),
            "cudnnBackendGetAttribute",
        )?;
        if index < 0 {
            return Err(Error::invalid(format!(
                "a cuDNN engine has no global index: {index}"
            )));
        }
        Ok(Engine::new(index))
    }
}

impl Drop for Cudnn {
    fn drop(&mut self) {
        // The owner drained every stream it ran on; destroying the handle
        // needs no current context.
        // SAFETY: the handle is this value's and is destroyed once.
        let _ = unsafe { (self.api.destroy)(self.handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conv1d(x: Vec<i64>, w: Vec<i64>, y: Vec<i64>) -> Convolution {
        Convolution::new(
            Tensor::contiguous(x, DType::F32),
            Tensor::contiguous(w, DType::F32),
            Tensor::contiguous(y, DType::F32),
            vec![1],
            vec![1],
            vec![1],
        )
    }

    #[test]
    fn a_contiguous_tensor_has_row_major_strides() {
        let tensor = Tensor::contiguous(vec![2, 3, 4, 5], DType::F32);

        assert_eq!(tensor.strides, vec![60, 20, 5, 1]);
        assert_eq!(tensor.rank(), 4);
    }

    #[test]
    fn a_convolution_needs_one_padding_per_spatial_dimension() {
        let mut convolution = Convolution::new(
            Tensor::contiguous(vec![1, 8, 4, 4, 4], DType::F32),
            Tensor::contiguous(vec![8, 8, 3, 3, 3], DType::F32),
            Tensor::contiguous(vec![1, 8, 4, 4, 4], DType::F32),
            vec![1, 1, 1],
            vec![1, 1, 1],
            vec![1, 1, 1],
        );
        assert!(convolution.check().is_ok(), "{convolution:?}");

        convolution.padding = vec![1, 1];
        let error = convolution
            .check()
            .expect_err("two paddings for three dimensions");
        assert!(error.to_string().contains("3 padding values"), "{error}");
    }

    #[test]
    fn the_output_dimensions_follow_padding_stride_and_dilation() {
        // A three-dimensional convolution with padding, stride and dilation.
        let mut convolution = Convolution::new(
            Tensor::contiguous(vec![1, 4, 8, 8, 8], DType::F32),
            Tensor::contiguous(vec![8, 4, 3, 3, 3], DType::F32),
            Tensor::contiguous(vec![1, 8, 4, 4, 4], DType::F32),
            vec![1, 1, 1],
            vec![2, 2, 2],
            vec![1, 1, 1],
        );
        assert!(convolution.check().is_ok(), "{convolution:?}");

        convolution.y.dims[3] = 5;
        let error = convolution
            .check()
            .expect_err("a wrong output length is caught");
        assert!(error.to_string().contains("give 4"), "{error}");

        // The audio decoder's one-dimensional convolution: rank three.
        let audio = conv1d(vec![1, 6, 512], vec![12, 6, 7], vec![1, 12, 513]);
        assert!(audio.check().is_ok(), "{audio:?}");
        assert_eq!(audio.rank(), 1);
    }

    #[test]
    fn a_convolution_checks_its_channels_and_compute_type() {
        let convolution = conv1d(vec![1, 6, 512], vec![12, 5, 7], vec![1, 12, 508]);
        let error = convolution
            .check()
            .expect_err("the filter and the input disagree on channels");
        assert!(error.to_string().contains("filter spans 5"), "{error}");

        let mut convolution = conv1d(vec![1, 6, 512], vec![12, 6, 7], vec![1, 12, 507]);
        let error = convolution.check().expect_err("the output length is wrong");
        assert!(error.to_string().contains("give 508"), "{error}");

        convolution.y.dims[2] = 508;
        convolution.compute = DType::Bf16;
        let error = convolution
            .check()
            .expect_err("a convolution accumulates in FP32");
        assert!(error.to_string().contains("accumulates in FP32"), "{error}");
    }

    #[test]
    fn a_tensor_needs_a_stride_per_dimension() {
        let mut tensor = Tensor::contiguous(vec![1, 6, 512], DType::F32);
        tensor.strides.truncate(2);

        let error = tensor.check().expect_err("two strides for three dims");
        assert!(error.to_string().contains("has 2 strides"), "{error}");
    }

    #[test]
    fn the_bytes_of_each_tensor_follow_its_dimensions() {
        let convolution = Convolution::new(
            Tensor::contiguous(vec![1, 2, 2, 2, 2], DType::F32),
            Tensor::contiguous(vec![2, 2, 1, 1, 1], DType::F32),
            Tensor::contiguous(vec![1, 2, 2, 2, 2], DType::F32),
            vec![0, 0, 0],
            vec![1, 1, 1],
            vec![1, 1, 1],
        );

        assert_eq!(
            convolution.bytes(),
            ConvolutionBytes {
                x: 16 * 4,
                w: 4 * 4,
                y: 16 * 4,
            }
        );
    }

    #[test]
    fn an_engine_round_trips_through_the_tuning_bytes() {
        let engine = Engine::new(42);

        assert_eq!(engine.to_bytes(), 42i64.to_le_bytes());
        assert_eq!(Engine::from_bytes(&engine.to_bytes()).unwrap(), engine);

        for bytes in [&[0u8; 7][..], &[0u8; 9][..], &[0xff; 8][..]] {
            assert!(Engine::from_bytes(bytes).is_err(), "{bytes:?}");
        }
    }

    #[test]
    fn a_handle_moves_between_threads() {
        fn movable<T: Send>() {}

        movable::<Cudnn>();
    }
}
