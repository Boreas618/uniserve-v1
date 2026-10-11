//! `Methods`: the Python half of a runner, reached by import path.
//!
//! A runner's Python half is a class of the library, for example
//! `uniserve_models.runner.causal_lm.CausalLMRunner`. [`Methods`] constructs
//! the object, resolves the methods a runner calls and builds their calls.
//! A method takes the runner first, then its scalar arguments and the
//! tensors it reads and writes; the Python contract of each is its
//! docstring.

use tvm_ffi::Any;

use crate::Result;
use crate::ffi::{Function, Handle, Shape};
use crate::model::interface::Interface;
use crate::parallel::joins::GraphFunctions;
use crate::storage::memory::StorageView;

/// One argument of a method call.
#[derive(Clone)]
pub(crate) enum Argument {
    Integer(i64),
    Float(f64),
    Integers(Vec<i64>),
    Floats(Vec<f64>),
    Text(String),
    Tensor(StorageView),
    Handle(Handle),
    None,
}

impl From<usize> for Argument {
    fn from(value: usize) -> Self {
        Self::Integer(value as i64)
    }
}

impl From<bool> for Argument {
    fn from(value: bool) -> Self {
        Self::Integer(i64::from(value))
    }
}

impl From<&[i64]> for Argument {
    fn from(values: &[i64]) -> Self {
        Self::Integers(values.to_vec())
    }
}

impl From<&StorageView> for Argument {
    fn from(view: &StorageView) -> Self {
        Self::Tensor(view.clone())
    }
}

/// A method and its owned arguments, called later, possibly on another
/// thread (a microbatch's forward). Tensors are exported when the call is
/// made, on the calling thread.
pub(crate) struct MethodCall {
    function: Function,
    arguments: Vec<Argument>,
}

impl MethodCall {
    /// Make the call.
    pub(crate) fn invoke(self) -> Result<Any> {
        let values = self
            .arguments
            .into_iter()
            .map(|argument| {
                Ok(match argument {
                    Argument::Integer(value) => Any::from(value),
                    Argument::Float(value) => Any::from(value),
                    Argument::Integers(values) => Any::from(Shape::from(values.as_slice())),
                    Argument::Floats(values) => {
                        Any::from(crate::ffi::list(values.into_iter().map(Any::from)))
                    }
                    Argument::Tensor(view) => Any::from(view.to_tensor()?),
                    Argument::Text(text) => Any::from(tvm_ffi::String::from(text.as_str())),
                    Argument::Handle(handle) => handle.to_any(),
                    Argument::None => Any::default(),
                })
            })
            .collect::<Result<Vec<Any>>>()?;
        self.function.call_values(&values)
    }
}

/// The Python runner: the object its methods take first.
pub(crate) struct Methods {
    /// The import path of the runner's class.
    class: &'static str,
    runner: Handle,
    /// Captures the methods called between its begin and end, and releases
    /// their graphs.
    pub(crate) graphs: GraphFunctions,
}

impl Methods {
    /// Construct the Python runner `class(*arguments)`. A constructor may
    /// run collectives, so the ranks of a group construct their runners in
    /// the same order.
    pub(crate) fn construct(
        interface: &Interface,
        class: &'static str,
        arguments: Vec<Argument>,
    ) -> Result<Self> {
        let runner = MethodCall {
            function: interface.import(class)?,
            arguments,
        }
        .invoke()?;
        Ok(Self {
            class,
            runner: Handle::new(runner)?,
            graphs: GraphFunctions::from_interface(interface)?,
        })
    }

    /// The Python runner.
    pub(crate) fn runner(&self) -> &Handle {
        &self.runner
    }

    /// The method `name` of the runner's class. A runner resolves the
    /// methods it calls once, when it is built, so a method the library
    /// does not have fails startup with its import path and a forward
    /// looks nothing up.
    pub(crate) fn method(&self, interface: &Interface, name: &str) -> Result<Function> {
        interface.import(&format!("{}.{name}", self.class))
    }

    /// A call of `method` on the runner with `arguments`.
    pub(crate) fn call(&self, method: &Function, arguments: Vec<Argument>) -> MethodCall {
        let mut all = Vec::with_capacity(arguments.len() + 1);
        all.push(Argument::Handle(self.runner.clone()));
        all.extend(arguments);
        MethodCall {
            function: method.clone(),
            arguments: all,
        }
    }
}
