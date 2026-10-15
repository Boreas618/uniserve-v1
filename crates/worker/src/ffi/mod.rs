//! TVM-FFI values exchanged with the Python numerical library.
//!
//! The worker receives exactly one value from Python at startup: the
//! interface table, a map from `<area>.<verb>` names to functions. Loading and
//! binding functions return opaque handles and per-component maps of further
//! functions. Everything Rust sends back is a TVM-FFI value: integers, floats,
//! strings, shapes, arrays, maps, tensors and the handles it was given.
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
mod record;
mod values;

pub use dlpack::{Backing, DType, Device, contiguous_strides, export};
pub use function::{DeferredCall, Function};
pub use handle::Handle;
pub use interpreter::{Interpreter, cooperative};
pub use record::Record;
pub use tvm_ffi::{Array, Map, Shape, Tensor};
pub use values::{integer_list, integers, is_none, list, strings, texts};
