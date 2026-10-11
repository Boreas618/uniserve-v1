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
const METHODS: [&str; 11] = [
    "decode",
    "plan_decode",
    "prefill",
    "plan_prefill",
    "sample",
    "verify",
    "prompt_logprobs",
    "commit",
    "vocab",
    "hidden_size",
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

/// The width of the model's hidden states.
pub(crate) fn hidden_size(methods: &Methods) -> Result<usize> {
    let size = i64::try_from(methods.call("hidden_size", Vec::new())?.invoke()?)
        .map_err(|_| Error::Interface(format!("{RUNNER}.hidden_size returns an integer")))?;
    Ok(size as usize)
}

/// Whether an attention operator plans each launch on the host, so a
/// captured forward is launched after `plan_decode` or `plan_prefill`.
pub(crate) fn plans_attention(methods: &Methods) -> Result<bool> {
    bool::try_from(methods.call("plans_attention", Vec::new())?.invoke()?)
        .map_err(|_| Error::Interface(format!("{RUNNER}.plans_attention returns a flag")))
}
