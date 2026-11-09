//! What an operator is: a name, a descriptor, an argument struct, and four functions — check a
//! call, launch it, shadow it, and say how far two implementations of it may differ.
//!
//! An operator is a Rust type, not a trait object and not a name in a table. Its Rust signature is
//! the interface a kernel program fills: the package implements what [`Op::launch`] launches, and a
//! change of signature is a change of the crate. The trait is
//! defined once here so that the catalogue of operators, the program packages and the
//! loaders all describe the same thing.

use std::fmt::Display;
use std::hash::Hash;

use crate::generation::Generation;
use crate::kernel::Kernel;
use crate::platform::Stream;
use crate::tolerance::Tolerance;
use crate::twins::Twins;
use crate::{Result, capture, descriptor};

/// One operator.
///
/// `Descriptor` is everything a compilation may bake in — element types, dimensions, page sizes,
/// mask variants — and its `Display` is the canonical key the module is stored under. `Args` holds
/// the borrowed views and scalars of one call; everything a launch reads that may change between
/// calls is there, and nothing that may not.
pub trait Op {
    /// The operator's name, e.g. `"rms_norm"`; also the directory name of its programs
    /// (`crates/ops/kernels/<operator>/`).
    const NAME: &'static str;

    /// The parameter values one compilation covers, whose canonical text is the descriptor key.
    type Descriptor: Display + Eq + Hash;

    /// The borrowed views and scalars of one call.
    type Args<'a>;

    /// Refuse arguments a program compiled for `descriptor` cannot be launched with.
    ///
    /// Every rule a call's views and scalars must meet against the descriptor is stated here and
    /// only here: extents, element types, strides, optional operands the descriptor requires or
    /// excludes, and index widths. It reads no device memory and needs no kernel, so a recording
    /// forward checks a call exactly as a launching one does, and a recorded descriptor key always
    /// comes from a call the device would accept.
    ///
    /// # Errors
    ///
    /// The first rule the arguments break, naming the operator and the operand.
    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()>;

    /// Pack the arguments, compute the grid and launch `kernel`.
    ///
    /// The kernel must be the one the caller resolved for this descriptor, and the caller has
    /// passed these arguments through [`Op::check`]; `launch` does not check them again. It reads
    /// the descriptor for what the module was compiled with and the arguments for what this call
    /// is. It is capture-safe: no allocation, no synchronization, no host read of device memory.
    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()>;

    /// Redirect the call's outputs into shadow buffers, for shadow execution.
    ///
    /// The new implementation of a replaced operator runs with the same inputs as the old one and
    /// writes into `twins` instead of the real outputs, so the two results can be compared without
    /// the difference of one operator accumulating through the next. An operand the operator
    /// modifies in place is redirected too, after its current bytes are copied into the twin.
    ///
    /// The twins borrow is not tied to the arguments' lifetime: `Forward::operate` owns the twins
    /// behind a `RefCell` borrowed for the length of the call, while the argument views carry the
    /// caller's lifetime, and the returned arguments hold views into the twins' buffers rather than
    /// a reference to the twins themselves.
    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a>;

    /// How far two implementations of this operator may be from another on this descriptor.
    ///
    /// The bound follows from the operator's mathematics and the descriptor's dtypes; see
    /// [`Tolerance`] and each operator's documentation for the reduction length and the dtype it is
    /// written with.
    fn tolerance(descriptor: &Self::Descriptor) -> Tolerance;
}

/// One checked call of an operator, with the operator's type erased.
///
/// A launching forward hands its calls to a [`Route`] as this, so a route chosen at runtime can
/// launch a call, or launch it with its outputs redirected, without naming the operator. Every
/// launch through it reports itself to the installed capture sink ([`crate::capture`]), as a
/// forward's own launches do.
pub trait Call {
    /// The operator's [`Op::NAME`].
    fn operator(&self) -> &'static str;

    /// The descriptor's key: the text the program is stored under.
    fn key(&self) -> String;

    /// The bound two implementations of this call may differ by ([`Op::tolerance`]).
    fn tolerance(&self) -> Tolerance;

    /// Launches `kernel` with the call's own arguments.
    ///
    /// # Errors
    ///
    /// The launch's failure.
    fn launch(&self, kernel: &Kernel, stream: &Stream) -> Result<()>;

    /// Launches `kernel` with the call's outputs redirected into `twins` ([`Op::shadow`]); the
    /// redirected arguments are checked first, because the twins are other views than the call's.
    ///
    /// # Errors
    ///
    /// The redirected arguments' refusal, or the launch's failure.
    fn launch_redirected(&self, kernel: &Kernel, twins: &mut Twins, stream: &Stream) -> Result<()>;
}

/// Where a launching forward sends its calls, when the runtime routes them itself.
///
/// The runtime installs a route on a forward to launch some calls otherwise than once with the
/// current generation's program: a program update's paired execution launches a replaced
/// operator's new and previous programs over the same inputs. A forward without a route, and every
/// call a route declines, launches the current program once.
pub trait Route {
    /// Launches `call` on `stream`, resolving programs from `current` or elsewhere, and answers
    /// whether it did; `false` leaves the call to the forward's own launch.
    ///
    /// # Errors
    ///
    /// A program the route cannot resolve, or a launch's failure.
    fn launch(&self, call: &dyn Call, current: &Generation, stream: &Stream) -> Result<bool>;
}

/// `O`'s call: its descriptor and its arguments, already checked against it ([`Op::check`]).
pub struct Checked<'d, 't, O: Op> {
    descriptor: &'d O::Descriptor,
    args: O::Args<'t>,
}

impl<'d, 't, O: Op> Checked<'d, 't, O> {
    /// The call of `O` with `args` under `descriptor`; the caller has checked them.
    pub const fn new(descriptor: &'d O::Descriptor, args: O::Args<'t>) -> Self {
        Self { descriptor, args }
    }

    /// Tells the capture sink, when one is installed, that the call's program launched.
    fn launched(&self) {
        if capture::installed() {
            capture::launched(O::NAME, &descriptor::key(self.descriptor));
        }
    }
}

impl<'t, O: Op> Call for Checked<'_, 't, O>
where
    O::Args<'t>: Copy,
{
    fn operator(&self) -> &'static str {
        O::NAME
    }

    fn key(&self) -> String {
        descriptor::key(self.descriptor)
    }

    fn tolerance(&self) -> Tolerance {
        O::tolerance(self.descriptor)
    }

    fn launch(&self, kernel: &Kernel, stream: &Stream) -> Result<()> {
        O::launch(kernel, self.descriptor, self.args, stream)?;
        self.launched();
        Ok(())
    }

    fn launch_redirected(&self, kernel: &Kernel, twins: &mut Twins, stream: &Stream) -> Result<()> {
        let redirected = O::shadow(self.args, twins, stream);
        O::check(self.descriptor, &redirected)?;
        O::launch(kernel, self.descriptor, redirected, stream)?;
        self.launched();
        Ok(())
    }
}
