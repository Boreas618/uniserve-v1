//! What loading, launching and shadowing an operator, and computing on the device backend's
//! tensors, can fail with.
//!
//! An operator never falls back and never ignores a mismatch: a kernel whose module does not match
//! the descriptor, an argument whose shape contradicts it, a twin region that is too small and a
//! native entry that reports an error are all failures of the launch, and the caller (the runner)
//! turns that into a failed precopy or a failed batch.
//!
//! The tensor library reports a device backend's failure as its own `Error::Backend`, which keeps
//! this type: a primitive converts with `?`, and a caller that needs the cause downcasts the boxed
//! error to [`Error`].

use tensor::{DType, DeviceLocation};

/// A failure of an operator, a launch, or a tensor operation on the device.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A call into the CUDA boundary failed: a stream, an event, a library, the communicator.
    #[error("platform: {0}")]
    Platform(#[from] platform::Error),

    /// The generation holds no kernel for this operator and descriptor.
    #[error(
        "generation {generation} has no kernel for operator `{operator}` descriptor `{descriptor}`"
    )]
    MissingKernel {
        operator: &'static str,
        descriptor: String,
        generation: u64,
    },

    /// The kernel is of the other form than the operator launches it as.
    #[error("operator `{operator}` launches `{kernel}` as a {expected}, but it is a {actual}")]
    KernelForm {
        operator: &'static str,
        kernel: String,
        expected: &'static str,
        actual: &'static str,
    },

    /// A native entry returned a nonzero status.
    #[error("native entry `{entry}` failed with status {status}{}", match message { Some(message) => format!(": {message}"), None => String::new() })]
    Native {
        entry: String,
        status: i32,
        message: Option<String>,
    },

    /// The module's metadata cannot be read, or lacks a field a launch needs.
    #[error("kernel metadata: {0}")]
    Metadata(String),

    /// An operator's launch words disagree with the kernel's own signature: their number, the
    /// parameter a word names at a position, a value its parameter's type does not hold, or the
    /// global scratch.
    #[error("operator `{operator}` launching `{kernel}`: {problem}")]
    Signature {
        /// The operator whose launch packed the words ([`crate::Op::NAME`]).
        operator: &'static str,
        /// The kernel's symbol, as its metadata names it.
        kernel: String,
        /// The disagreement, by position and by both names.
        problem: String,
    },

    /// A launch grid does not fit the driver's limits.
    #[error("grid: {0}")]
    Grid(String),

    /// An argument's shape, dtype, stride or rank does not match the operator.
    #[error("operator `{operator}`: {problem}")]
    Argument {
        operator: &'static str,
        problem: String,
    },

    /// A call no operator expresses: an operator's call it cannot compute (a collective with a
    /// non-uniform count), or a tensor operation this backend has no kernel for, or whose
    /// arguments no kernel it has covers.
    #[error("{operation} is unsupported: {reason}")]
    Unsupported {
        /// The operator's name or the tensor operation: `"topk_softmax"`, `"max_pool2d"`.
        operation: &'static str,
        /// What the kernels lack for it.
        reason: String,
    },

    /// A shape has more axes than the ABI holds, or a dtype has no whole-byte size.
    #[error("shape: {0}")]
    Shape(String),

    /// The twin region cannot hold the shadow buffers this operator needs.
    #[error("twin region has {available} bytes left, {needed} needed for operator `{operator}`")]
    Twins {
        operator: &'static str,
        needed: u64,
        available: u64,
    },

    /// The context's region cannot hold an allocation.
    #[error(
        "an allocation of {needed} bytes does not fit the region: {used} of {capacity} bytes are in use"
    )]
    OutOfMemory {
        /// The rounded size asked for.
        needed: u64,
        /// The furthest byte in use when the allocation was refused.
        used: u64,
        /// The region's size.
        capacity: u64,
    },

    /// An operation on the device ran on a thread where execution installed no context.
    #[error("{operation} needs an execution context, and none is installed on this thread")]
    NoContext {
        /// What was asked for, as the sentence's subject: `"a launch"`, `"a collective"`.
        operation: &'static str,
    },

    /// A launch was refused: its arguments broke the operator's rules, the generation holds no
    /// kernel for its key, or the kernel could not be launched.
    #[error("operator `{operator}` with key `{key}`: {source}")]
    Launch {
        /// The operator's name ([`crate::Op::NAME`]).
        operator: &'static str,
        /// The descriptor's key: the text the kernel is stored under.
        key: String,
        /// Why the call was refused.
        source: Box<Error>,
    },

    /// A host-to-device copy does not fit what is left of the input buffer.
    #[error(
        "a host-to-device copy of {needed} bytes at offset {at} does not fit the input buffer of {capacity} bytes"
    )]
    InputBufferExhausted {
        /// The bytes the copy carries.
        needed: u64,
        /// The aligned offset the copy would start at.
        at: u64,
        /// The input buffer's size.
        capacity: u64,
    },

    /// A context that launches was asked for a host-to-device copy with no input buffer set: the
    /// copy reads page-locked memory, and there is none to read.
    #[error(
        "a host-to-device copy under a launching context needs an input buffer, and none is set"
    )]
    NoInputBuffer,

    /// A collective was called without a group, with operands the group cannot move, or with a
    /// place in the group that does not exist.
    #[error("{collective}: {problem}")]
    Collective {
        /// The collective's name: `"all_gather"`, `"set_group"`.
        collective: &'static str,
        /// What is wrong with the call.
        problem: String,
    },

    /// The profiler could not launch a range marker.
    #[error("profiler range `{path}` of `{class}`: {source}")]
    Range {
        /// The module class the range belongs to.
        class: String,
        /// The module instance's path in the model.
        path: String,
        /// The profiler's own failure.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// The profiler could not launch a kernel's probe ([`Profiler::launch`](crate::Profiler::launch)).
    #[error("kernel probe: {0}")]
    Probe(Box<dyn std::error::Error + Send + Sync>),

    /// A tensor or a device of another backend, or of the CPU, where one of this backend's is
    /// needed.
    #[error("{operation} needs a tensor of the CUDA backend, and was given one on {location:?}")]
    ForeignTensor {
        /// What was asked for, as the sentence's subject: `"a kernel operand"`.
        operation: &'static str,
        /// Where the tensor or the device is.
        location: DeviceLocation,
    },

    /// A tensor with more axes than a kernel argument carries.
    #[error("a tensor of {rank} axes is no kernel argument: an operand carries at most {limit}")]
    Rank {
        /// The tensor's number of axes.
        rank: usize,
        /// `tensor::nditer::MAX_DIMS`.
        limit: usize,
    },

    /// An element type whose elements this backend cannot place in memory: a packed element
    /// with no byte layout, or a view into a packed tensor that does not start at a whole byte.
    #[error("{operation} cannot address {dtype:?} elements: {reason}")]
    ElementType {
        /// What was asked for: `"an allocation"`, `"a kernel operand"`.
        operation: &'static str,
        /// The element type.
        dtype: DType,
        /// What its layout lacks.
        reason: &'static str,
    },

    /// A byte count beyond what the device addresses.
    #[error("{elements} elements of {dtype:?} exceed the bytes a device addresses")]
    TooLarge {
        /// The element count asked for.
        elements: u64,
        /// The element type.
        dtype: DType,
    },

    /// A tensor was asked for on a GPU other than the one the installed context computes on.
    #[error("a tensor was asked for on GPU {asked} under a context of GPU {context}")]
    WrongDevice {
        /// The ordinal of the device the caller named.
        asked: usize,
        /// The ordinal of the context installed on the thread.
        context: usize,
    },

    /// A meta tensor, which has no data, was given to a context that launches.
    #[error("a meta tensor has no data for a launching context to compute on")]
    MetaOperand,

    /// A device-to-host read of a meta tensor, or under a meta context, which has no stream to
    /// copy on.
    #[error("a meta tensor has no data to read, and a meta context has no stream to read on")]
    MetaRead,

    /// A device-to-host read while the context's stream is being captured: the read would wait
    /// for kernels that have not run.
    #[error("a device-to-host read is refused while the stream is being captured")]
    CapturedRead,

    /// A graph break's host work failed, or the break was stated where nothing takes it.
    #[error("a graph break: {0}")]
    Break(#[source] Box<dyn std::error::Error + Send + Sync>),
}

impl Error {
    /// An argument mismatch of `operator`.
    pub(crate) fn argument(operator: &'static str, problem: impl Into<String>) -> Self {
        Self::Argument {
            operator,
            problem: problem.into(),
        }
    }
}

impl From<Error> for tensor::Error {
    /// The failure as the tensor library carries it across its backend traits.
    fn from(error: Error) -> Self {
        Self::Backend(Box::new(error))
    }
}

/// The result of an operator call or a device operation.
pub type Result<T, E = Error> = std::result::Result<T, E>;
