//! What an operator is: a name, a descriptor, an argument struct, and three functions — launch it,
//! shadow it, and say how far two implementations of it may differ.
//!
//! An operator is a Rust type, not a trait object and not a name in a table. Its Rust signature is
//! the interface a kernel program fills: the package implements what [`Op::launch`] launches, and a
//! change of signature is a change of the crate (specs/worker.md 4.2). The design's trait is
//! reproduced here verbatim so that the catalogue of operators, the program packages and the
//! loaders all describe the same thing.

use std::fmt::Display;
use std::hash::Hash;

use crate::Result;
use crate::kernel::Kernel;
use crate::platform::Stream;
use crate::tolerance::Tolerance;
use crate::twins::Twins;

/// One operator.
///
/// `Descriptor` is everything a compilation may bake in — element types, dimensions, page sizes,
/// mask variants — and its `Display` is the canonical key the module is stored under. `Args` holds
/// the borrowed views and scalars of one call; everything a launch reads that may change between
/// calls is there, and nothing that may not.
pub trait Op {
    /// The operator's name, e.g. `"rms_norm"`; also the directory name of its programs
    /// (`crates/ops/kernels/<operator>/`, specs/worker.md 5.2).
    const NAME: &'static str;

    /// The parameter values one compilation covers, whose canonical text is the descriptor key.
    type Descriptor: Display + Eq + Hash;

    /// The borrowed views and scalars of one call.
    type Args<'a>;

    /// Pack the arguments, compute the grid and launch `kernel`.
    ///
    /// The kernel must be the one the caller resolved for this descriptor; `launch` reads the
    /// descriptor for what the module was compiled with and the arguments for what this call is.
    /// It is capture-safe: no allocation, no synchronization, no host read of device memory.
    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()>;

    /// Redirect the call's outputs into shadow buffers, for shadow execution (specs/worker.md 7.3).
    ///
    /// The new implementation of a replaced operator runs with the same inputs as the old one and
    /// writes into `twins` instead of the real outputs, so the two results can be compared without
    /// the difference of one operator accumulating through the next. An operand the operator
    /// modifies in place is redirected too, after its current bytes are copied into the twin.
    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a>;

    /// How far two implementations of this operator may be from another on this descriptor.
    ///
    /// The bound follows from the operator's mathematics and the descriptor's dtypes; see
    /// [`Tolerance`] and each operator's documentation for the reduction length and the dtype it is
    /// written with.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance;
}
