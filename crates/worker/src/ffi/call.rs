//! Calls into the library with owned arguments ([`MethodCall`]): a method
//! of the executor or a library function, which a job hands to the builder
//! thread.
//!
//! A method takes its object first, then its scalar arguments and the
//! tensors it reads and writes; the Python contract of each is its
//! docstring.

use tvm_ffi::Any;

use super::{Function, Handle, Interpreter, Shape};
use crate::memory::allocation::StorageView;
use crate::{Error, Result};

/// One argument of a method call.
#[derive(Clone)]
pub(crate) enum Argument {
    Bool(bool),
    Integer(i64),
    Integers(Vec<i64>),
    Text(String),
    Texts(Vec<String>),
    Tensor(StorageView),
    Handle(Handle),
    None,
}

impl From<usize> for Argument {
    fn from(value: usize) -> Self {
        Self::Integer(value as i64)
    }
}

/// A method or library function and its owned arguments, called later by the
/// job that records it, on the builder thread (a call made on another thread is
/// handed to it). A view crosses as its torch tensor ([`StorageView::handle`]),
/// wrapped by the first call that passes it.
#[derive(Clone)]
pub(crate) struct MethodCall {
    function: Function,
    arguments: Vec<Argument>,
    /// A serving call: its run on the builder thread must not make the host
    /// wait for the device ([`Self::serving`]).
    serving: bool,
}

impl MethodCall {
    /// A call of `function` with `arguments`: for a method, its Python
    /// object first.
    pub(crate) fn new(function: &Function, arguments: Vec<Argument>) -> Self {
        Self {
            function: function.clone(),
            arguments,
            serving: false,
        }
    }

    /// Mark a serving call: its run on the builder thread is watched for
    /// driver calls that make the host wait for the device
    /// ([`cuda::cupti::blocking`]); such a call is counted
    /// ([`crate::profiling::blocking`]) and fails the call, before or past
    /// its batch's commit point as any failure of the call would.
    pub(crate) fn serving(mut self) -> Self {
        self.serving = true;
        self
    }

    /// The import path of the method called.
    pub(crate) fn name(&self) -> &str {
        self.function.name()
    }

    /// Make the call. The arguments' values, the first wrapping of a view
    /// no earlier call passed and the call itself take one handoff to the
    /// builder thread, where the call's references to Python objects are
    /// dropped too.
    pub(crate) fn invoke(self) -> Result<Any> {
        Interpreter::execute(move || {
            // The values reference the arguments' handles, which outlive
            // them: `self` drops after this closure's locals.
            let values = self
                .arguments
                .iter()
                .map(|argument| {
                    Ok(match argument {
                        Argument::Bool(value) => Any::from(*value),
                        Argument::Integer(value) => Any::from(*value),
                        Argument::Integers(values) => Any::from(Shape::from(values.as_slice())),
                        Argument::Tensor(view) => view.handle()?.to_any(),
                        Argument::Text(text) => Any::from(tvm_ffi::String::from(text.as_str())),
                        Argument::Texts(texts) => Any::from(crate::ffi::strings(texts)),
                        Argument::Handle(handle) => handle.to_any(),
                        Argument::None => Any::default(),
                    })
                })
                .collect::<Result<Vec<Any>>>()?;
            let watched = self.serving.then(cuda::cupti::blocking::Watched::enter);
            let result = self.function.call_values(&values);
            let Some(watched) = watched else {
                return result;
            };
            // The job that performs the call counts these calls with the rest
            // of its own (`crate::job`).
            let blocking = watched.calls();
            if blocking.is_empty() {
                return result;
            }
            result?;
            let calls = blocking
                .iter()
                .map(|(call, count)| format!("{call} x{count}"))
                .collect::<Vec<_>>()
                .join(", ");
            Err(Error::Numerical {
                function: self.function.name().to_owned(),
                kind: "InvariantViolation".into(),
                message: format!("the serving call made the host wait for the device: {calls}"),
                traceback: String::new(),
            })
        })
    }
}
