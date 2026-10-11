//! The Python half of [`super::CausalLMRunner`]:
//! `uniserve_models.runner.causal_lm.CausalLMRunner`.
//!
//! [`Methods`] holds the Python runner and its methods, reached by import
//! path. A method takes the runner first, then its scalar arguments and the
//! tensors it reads and writes; the Python contract of each is its
//! docstring.

use tvm_ffi::Any;

use crate::ffi::{Function, Handle, Shape};
use crate::model::interface::Interface;
use crate::parallel::joins::GraphFunctions;
use crate::storage::memory::StorageView;
use crate::{Error, Result};

const RUNNER: &str = "uniserve_models.runner.causal_lm.CausalLMRunner";

/// The vocabulary a text runner computes over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Vocab {
    /// Real tokens, excluding padding.
    pub(crate) size: usize,
    /// This rank's shard of the padded vocabulary.
    pub(crate) start: usize,
    pub(crate) stop: usize,
}

impl Vocab {
    /// Columns of this rank's logits.
    pub(crate) fn width(&self) -> usize {
        self.stop - self.start
    }
}

/// One argument of a method call.
pub(crate) enum Argument {
    Integer(i64),
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
    runner: Handle,
    pub(crate) decode: Function,
    pub(crate) plan_decode: Function,
    pub(crate) prefill: Function,
    pub(crate) plan_prefill: Function,
    pub(crate) sample: Function,
    pub(crate) verify: Function,
    pub(crate) prompt_logprobs: Function,
    pub(crate) commit: Function,
    /// Captures the methods called between its begin and end.
    pub(crate) graphs: GraphFunctions,
}

/// What a runner is constructed over: the prepared component, its stream,
/// its capacities and the resident request state.
pub(crate) struct Construction<'a> {
    pub(crate) prepared: &'a Handle,
    pub(crate) stream: usize,
    pub(crate) rows: usize,
    pub(crate) tokens: usize,
    pub(crate) width: usize,
    pub(crate) page_tokens: &'a [i64],
    pub(crate) windowed: &'a [i64],
    /// The constructor's tensors in parameter order: `unit_tables`,
    /// `start_pages`, `table_shapes`, `verified_lengths`,
    /// `future_input_tokens`, `logical_lengths`, `predicates`,
    /// `sampling_positions`, `penalty_counts`, `prompt_logits`.
    pub(crate) state: [&'a StorageView; 10],
}

impl Methods {
    /// Construct the Python runner. Pipeline stages share the last stage's
    /// vocabulary in a collective, so every rank constructs its runners in
    /// the same order.
    pub(crate) fn construct(
        interface: &Interface,
        construction: &Construction<'_>,
    ) -> Result<Self> {
        let method = |name: &str| interface.import(&format!("{RUNNER}.{name}"));
        let mut arguments = vec![
            Argument::Handle(construction.prepared.clone()),
            construction.stream.into(),
            construction.rows.into(),
            construction.tokens.into(),
            construction.width.into(),
            construction.page_tokens.into(),
            construction.windowed.into(),
        ];
        arguments.extend(construction.state.iter().map(|&view| view.into()));
        let runner = MethodCall {
            function: interface.import(RUNNER)?,
            arguments,
        }
        .invoke()?;

        Ok(Self {
            runner: Handle::new(runner)?,
            decode: method("decode")?,
            plan_decode: method("plan_decode")?,
            prefill: method("prefill")?,
            plan_prefill: method("plan_prefill")?,
            sample: method("sample")?,
            verify: method("verify")?,
            prompt_logprobs: method("prompt_logprobs")?,
            commit: method("commit")?,
            graphs: GraphFunctions::from_interface(interface)?,
        })
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

    /// A fact the runner reports once, by its method name.
    fn fact(&self, interface: &Interface, name: &str) -> Result<Any> {
        interface
            .import(&format!("{RUNNER}.{name}"))?
            .call(&[self.runner.view()])
    }

    /// The vocabulary shard of this rank's logits.
    pub(crate) fn vocab(&self, interface: &Interface) -> Result<Vocab> {
        let malformed = || Error::Interface(format!("{RUNNER}.vocab returns four integers"));
        let values =
            crate::ffi::integers(self.fact(interface, "vocab")?).map_err(|_| malformed())?;
        let [size, _padded, start, stop] = values[..] else {
            return Err(malformed());
        };
        Ok(Vocab {
            size: size as usize,
            start: start as usize,
            stop: stop as usize,
        })
    }

    /// The width of the model's hidden states.
    pub(crate) fn hidden_size(&self, interface: &Interface) -> Result<usize> {
        let size = i64::try_from(self.fact(interface, "hidden_size")?)
            .map_err(|_| Error::Interface(format!("{RUNNER}.hidden_size returns an integer")))?;
        Ok(size as usize)
    }

    /// Whether an attention operator plans each launch on the host, so a
    /// captured forward is launched after `plan_decode` or `plan_prefill`.
    pub(crate) fn plans_attention(&self, interface: &Interface) -> Result<bool> {
        bool::try_from(self.fact(interface, "plans_attention")?)
            .map_err(|_| Error::Interface(format!("{RUNNER}.plans_attention returns a flag")))
    }
}
