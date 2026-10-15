//! TVM-FFI values exchanged with the Python numerical library.
//!
//! The worker receives exactly one value from Python at startup: the
//! function that resolves an import path to a library callable
//! ([`Interpreter::import`]). Library functions return opaque handles and
//! TVM-FFI values; a torch tensor returns as an opaque handle. Everything
//! Rust sends is a TVM-FFI value: integers, floats, strings, shapes, arrays,
//! maps and handles. Rust's own storage crosses as handles too: each view
//! is wrapped once as a torch tensor, which every call that takes the view
//! receives ([`crate::storage::memory::StorageView::handle`]).
//!
//! One thread of a rank executes Python: the [`Interpreter`]. A [`Function`]
//! may be called from any thread; the call is handed to the interpreter and
//! the caller waits for its result, so no thread ever waits for the GIL.
//! The control thread, which waits for no call while it serves, submits its
//! calls instead ([`Interpreter::submit`]).
//! Callers must not hold a lock while calling into Python: a callback may
//! block on work that needs it.

mod dlpack;
mod function;
mod handle;
mod interpreter;

pub use dlpack::{Backing, DType, Device, contiguous_strides, export};
pub use function::{DeferredCall, Function, integer_list, integers, is_none, list, strings, texts};
pub use handle::Handle;
pub use interpreter::{Interpreter, cooperative};
pub use tvm_ffi::{Array, Map, Shape, Tensor};
