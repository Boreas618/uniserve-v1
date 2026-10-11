//! The Python half of [`super::CausalLMRunner`]:
//! `uniserve_models.runner.causal_lm.CausalLMRunner`, its construction and
//! the facts it reports.

use crate::ffi::Handle;
use crate::model::interface::Interface;
use crate::runner::methods::{Argument, Methods};
use crate::storage::memory::StorageView;
use crate::{Error, Result};

const RUNNER: &str = "uniserve_models.runner.causal_lm.CausalLMRunner";

/// The methods of the Python runner the rank calls.
const METHODS: [&str; 12] = [
    "decode",
    "plan_decode",
    "prefill",
    "plan_prefill",
    "sample",
    "sample_top_k",
    "sample_greedy",
    "verify",
    "prompt_logprobs",
    "commit",
    "vocab",
    "plans_attention",
];

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

/// Construct the Python runner. Pipeline stages share the last stage's
/// vocabulary in a collective, so every rank constructs its runners in
/// the same order.
pub(crate) fn construct(interface: &Interface, construction: &Construction<'_>) -> Result<Methods> {
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
    Methods::construct(interface, RUNNER, &METHODS, arguments)
}

/// A size of the loaded model's causal LM, read before a runner exists:
/// the library function `name` over the loaded model.
fn size(interface: &Interface, loaded: &Handle, name: &str) -> Result<usize> {
    let function = format!("uniserve_models.runner.causal_lm.{name}");
    let value = interface.import(&function)?.call(&[loaded.view()])?;
    let value = i64::try_from(value).map_err(|error| Error::numerical(&function, &error))?;
    Ok(value as usize)
}

/// Real vocabulary entries of the loaded model's causal LM; zero on a rank
/// without one.
pub(crate) fn vocab_size(interface: &Interface, loaded: &Handle) -> Result<usize> {
    size(interface, loaded, "vocab_size")
}

/// The hidden width of the loaded model's causal LM; zero on a rank without
/// one.
pub(crate) fn hidden_size(interface: &Interface, loaded: &Handle) -> Result<usize> {
    size(interface, loaded, "hidden_size")
}

/// The vocabulary shard of this rank's logits.
pub(crate) fn vocab(methods: &Methods) -> Result<Vocab> {
    let malformed = || Error::Interface(format!("{RUNNER}.vocab returns four integers"));
    let values = crate::ffi::integers(methods.call("vocab", Vec::new())?.invoke()?)
        .map_err(|_| malformed())?;
    let [size, _padded, start, stop] = values[..] else {
        return Err(malformed());
    };
    Ok(Vocab {
        size: size as usize,
        start: start as usize,
        stop: stop as usize,
    })
}

/// Whether an attention operator plans each launch on the host, so a
/// captured forward is launched after `plan_decode` or `plan_prefill`.
pub(crate) fn plans_attention(methods: &Methods) -> Result<bool> {
    bool::try_from(methods.call("plans_attention", Vec::new())?.invoke()?)
        .map_err(|_| Error::Interface(format!("{RUNNER}.plans_attention returns a flag")))
}
