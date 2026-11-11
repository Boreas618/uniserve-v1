//! What the backend refuses.

use ops::platform::PlatformError;

/// A failure of the device backend.
#[derive(Debug, thiserror::Error)]
pub enum Error {
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
        /// The operator's name (`ops::Op::NAME`).
        operator: &'static str,
        /// The descriptor's key: the text the kernel is stored under.
        key: String,
        /// Why the kernel library refused the call.
        source: ops::Error,
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

    /// The kernel library refused a call that is not an operator launch, such as a copy.
    #[error(transparent)]
    Ops(#[from] ops::Error),

    /// A call into the CUDA boundary failed: a stream, an event or the communicator.
    #[error("platform: {0}")]
    Platform(#[from] PlatformError),
}

/// The backend's result.
pub type Result<T> = std::result::Result<T, Error>;
