//! `Methods`: the Python half of a runner, reached by import path.
//!
//! A runner's Python half is a class of the library, for example
//! `uniserve_models.runner.causal_lm.CausalLMRunner`. [`Methods`] constructs
//! the object and calls its methods by name. A method takes the runner
//! first, then its scalar arguments and the tensors it reads and writes;
//! the Python contract of each is its docstring.

use tvm_ffi::Any;

use crate::ffi::{Function, Handle, Shape};
use crate::model::interface::Interface;
use crate::parallel::joins::GraphFunctions;
use crate::storage::memory::StorageView;
use crate::{Error, Result};

/// One argument of a method call.
#[derive(Clone)]
pub(crate) enum Argument {
    Integer(i64),
    Float(f64),
    Integers(Vec<i64>),
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
                    Argument::Tensor(view) => Any::from(view.to_tensor()?),
                    Argument::Handle(handle) => handle.to_any(),
                    Argument::None => Any::default(),
                })
            })
            .collect::<Result<Vec<Any>>>()?;
        self.function.call_values(&values)
    }
}

/// The Python runner and its methods.
pub(crate) struct Methods {
    /// The import path of the runner's class.
    class: &'static str,
    runner: Handle,
    /// The methods serving calls, resolved at construction.
    methods: Vec<(&'static str, Function)>,
    /// Captures the methods called between its begin and end, and releases
    /// their graphs.
    pub(crate) graphs: GraphFunctions,
}

impl Methods {
    /// Construct the Python runner `class(*arguments)` and resolve
    /// `methods` of it, failing with the import path of a missing one.
    /// A constructor may run collectives, so the ranks of a group construct
    /// their runners in the same order.
    pub(crate) fn construct(
        interface: &Interface,
        class: &'static str,
        methods: &[&'static str],
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
            methods: methods
                .iter()
                .map(|&name| Ok((name, interface.import(&format!("{class}.{name}"))?)))
                .collect::<Result<_>>()?,
            graphs: GraphFunctions::from_interface(interface)?,
        })
    }

    /// A call of the method `name` on the runner with `arguments`.
    pub(crate) fn call(&self, name: &str, arguments: Vec<Argument>) -> Result<MethodCall> {
        let function = self
            .methods
            .iter()
            .find(|(method, _)| *method == name)
            .map(|(_, function)| function.clone())
            .ok_or_else(|| Error::Invariant(format!("{}.{name} was not resolved", self.class)))?;
        let mut all = Vec::with_capacity(arguments.len() + 1);
        all.push(Argument::Handle(self.runner.clone()));
        all.extend(arguments);
        Ok(MethodCall {
            function,
            arguments: all,
        })
    }
}
