//! The Python half of [`super::CausalLMRunner`]:
//! `uniserve.runner.causal_lm.CausalLMRunner`, its construction and
//! the facts it reports.

use crate::ffi::{Function, Handle, Interpreter};
use crate::runner::methods::{Argument, Methods};
use crate::storage::memory::StorageView;
use crate::{Error, Result};
use uniserve_cuda::DeviceId;

const RUNNER: &str = "uniserve.runner.causal_lm.CausalLMRunner";

/// The methods of the Python runner a forward calls, and the two its owner
/// calls: `kernels` for the kernel table and `close` at retirement.
pub(crate) struct Functions {
    pub(crate) decode: Function,
    pub(crate) plan_decode: Function,
    pub(crate) prefill: Function,
    pub(crate) plan_prefill: Function,
    pub(crate) sample: Function,
    pub(crate) sample_top_k: Function,
    pub(crate) sample_greedy: Function,
    pub(crate) verify: Function,
    pub(crate) prompt_logprobs: Function,
    pub(crate) commit: Function,
    pub(crate) kernels: Function,
    pub(crate) close: Function,
}

impl Functions {
    /// Resolve the methods of `methods`' runner.
    pub(crate) fn resolve(methods: &Methods) -> Result<Self> {
        let method = |name: &str| methods.method(name);
        Ok(Self {
            decode: method("decode")?,
            plan_decode: method("plan_decode")?,
            prefill: method("prefill")?,
            plan_prefill: method("plan_prefill")?,
            sample: method("sample")?,
            sample_top_k: method("sample_top_k")?,
            sample_greedy: method("sample_greedy")?,
            verify: method("verify")?,
            prompt_logprobs: method("prompt_logprobs")?,
            commit: method("commit")?,
            kernels: method("kernels")?,
            close: method("close")?,
        })
    }
}

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

/// What a runner is constructed over: the causal LM of the loaded model,
/// its stream and what its execution context borrows, its capacities and
/// the resident request state.
pub(crate) struct Construction<'a> {
    pub(crate) loaded: &'a Handle,
    /// The causal LM's module path in the model.
    pub(crate) path: &'a str,
    /// The CUDA ordinal the forwards compute on; `None` on the host.
    pub(crate) device: Option<DeviceId>,
    pub(crate) stream: usize,
    /// The stream's shared scratch (`Scratches`).
    pub(crate) scratch: Handle,
    /// The attention layers' view of the K/V unit pool.
    pub(crate) cache: &'a Handle,
    /// The expert exchange of the runner's microbatch.
    pub(crate) experts: Option<Handle>,
    /// The distributed expert weights' prefetch object.
    pub(crate) weights: Option<Handle>,
    /// The model's collectives on the stream (`StreamCollectives::natives`).
    pub(crate) collectives: Option<Handle>,
    pub(crate) rows: usize,
    pub(crate) tokens: usize,
    pub(crate) width: usize,
    pub(crate) block_size: &'a [i64],
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
pub(crate) fn construct(construction: &Construction<'_>) -> Result<Methods> {
    let optional =
        |handle: &Option<Handle>| handle.clone().map_or(Argument::None, Argument::Handle);
    let mut arguments = vec![
        Argument::Handle(construction.loaded.clone()),
        Argument::Text(construction.path.to_owned()),
        // The Python half takes the CUDA ordinal, `-1` for the host.
        Argument::Integer(construction.device.map_or(-1, |device| i64::from(device.0))),
        construction.stream.into(),
        Argument::Handle(construction.scratch.clone()),
        Argument::Handle(construction.cache.clone()),
        optional(&construction.experts),
        optional(&construction.weights),
        optional(&construction.collectives),
        construction.rows.into(),
        construction.tokens.into(),
        construction.width.into(),
        construction.block_size.into(),
        construction.windowed.into(),
    ];
    arguments.extend(construction.state.iter().map(|&view| view.into()));
    Methods::construct(RUNNER, arguments)
}

/// A size of the loaded model's causal LM, read before a runner exists:
/// the library function `name` over the loaded model.
fn size(loaded: &Handle, name: &str) -> Result<usize> {
    let function = format!("uniserve.runner.causal_lm.{name}");
    let value = Interpreter::import(&function)?.call(&[loaded.view()])?;
    let value = i64::try_from(value).map_err(|error| Error::numerical(&function, &error))?;
    Ok(value as usize)
}

/// Real vocabulary entries of the loaded model's causal LM; zero on a rank
/// without one.
pub(crate) fn vocab_size(loaded: &Handle) -> Result<usize> {
    size(loaded, "vocab_size")
}

/// The hidden width of the loaded model's causal LM; zero on a rank without
/// one.
pub(crate) fn hidden_size(loaded: &Handle) -> Result<usize> {
    size(loaded, "hidden_size")
}

/// A fact the runner reports once, by its method name.
fn fact(methods: &Methods, name: &str) -> Result<tvm_ffi::Any> {
    methods.call(&methods.method(name)?, Vec::new()).invoke()
}

/// The vocabulary shard of this rank's logits.
pub(crate) fn vocab(methods: &Methods) -> Result<Vocab> {
    let malformed = || Error::Interface(format!("{RUNNER}.vocab returns four integers"));
    let values = crate::ffi::integers(fact(methods, "vocab")?).map_err(|_| malformed())?;
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
    bool::try_from(fact(methods, "plans_attention")?)
        .map_err(|_| Error::Interface(format!("{RUNNER}.plans_attention returns a flag")))
}
