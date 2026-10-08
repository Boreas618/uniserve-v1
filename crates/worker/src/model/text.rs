//! The typed text binding of a causal language model (`text.bind`).
//!
//! `text.bind(prepared, stream, buffers, layout)` returns the step functions
//! of one lane binding over Rust-owned buffers. [`TextModel`] resolves them
//! once, failing startup naming any missing function, and converts each
//! call's integer arguments. The Python contract is documented in
//! `uniserve_models/interface/text.py`.

use tvm_ffi::{Any, AnyView};

use crate::ffi::{Fields, Function, Functions, Handle, Record, Shape};
use crate::storage::memory::StorageView;
use crate::{Error, Result};

use super::interface::Interface;

/// The vocabulary a text binding computes over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vocab {
    /// Real tokens, excluding padding.
    pub size: usize,
    /// The padded vocabulary partitioned across tensor-parallel ranks.
    pub padded: usize,
    /// This rank's shard of the padded vocabulary.
    pub start: usize,
    pub stop: usize,
}

impl Vocab {
    /// Columns of this rank's logits.
    pub fn width(&self) -> usize {
        self.stop - self.start
    }
}

/// One step invocation. Row and token counts include graph or decode-bucket
/// padding; `live` rows produce results.
pub struct StepArgs<'a> {
    pub graph: Option<&'a Handle>,
    pub decode: bool,
    pub rows: usize,
    pub live: usize,
    pub tokens: usize,
    /// Decode rows gather token, position and cache length from slot state.
    pub indexed: bool,
    pub widths: &'a [i64],
    pub queries: &'a [i64],
    pub prefixes: &'a [i64],
    /// First selected pages, `[table][row]`, read for windowed tables.
    pub starts: &'a [i64],
    pub causal: &'a [i64],
    /// Some row writes K/V (paged rather than prefix-only attention).
    pub writes: bool,
    pub logit_count: usize,
    /// Logits are every live row's last query token.
    pub all_last: bool,
    pub cache_only: bool,
    /// The sampling descriptor (`SamplingPlan::words`).
    pub sampling: &'a [i64],
    /// A decode graph's greedy selection is final after terminal policy.
    pub greedy: bool,
    /// The prompt scoring descriptor.
    pub prompt: &'a [i64],
}

/// The shapes a graph captures.
pub struct CaptureArgs<'a> {
    pub decode: bool,
    pub rows: usize,
    pub tokens: usize,
    pub widths: &'a [i64],
    pub queries: &'a [i64],
    pub prefixes: &'a [i64],
    pub starts: &'a [i64],
    pub causal: &'a [i64],
    pub writes: bool,
    pub cache_only: bool,
}

/// A bound text binding.
pub struct TextModel {
    step: Function,
    capture: Function,
    release: Function,
    commit: Function,
    close: Function,
    pub vocab: Vocab,
    pub hidden_size: usize,
}

fn integer(value: usize) -> Any {
    Any::from(value as i64)
}

fn flag(value: bool) -> Any {
    Any::from(i64::from(value))
}

fn shape(values: &[i64]) -> Any {
    Any::from(Shape::from(values))
}

/// Describe the vocabulary shard and hidden width of `prepared` (a bound
/// `component.prepare` handle): `text.describe`. Pipeline stages share the
/// last stage's vocabulary in a collective, so every rank calls this in the
/// same order.
pub fn describe(interface: &Interface, prepared: &Handle) -> Result<(Vocab, usize)> {
    let record = Record::new(
        "text.describe",
        interface
            .function("text.describe")?
            .call(&[prepared.view()])?,
    )?;
    let vocab = Vocab {
        size: record.count("vocab_size")?,
        padded: record.count("padded_vocab")?,
        start: record.count("vocab_start")?,
        stop: record.count("vocab_stop")?,
    };
    if vocab.stop <= vocab.start || vocab.size > vocab.padded {
        return Err(Error::Interface(
            "text.describe reports an invalid vocabulary shard".into(),
        ));
    }
    Ok((vocab, record.count("hidden_size")?))
}

impl TextModel {
    /// Bind the text steps of `prepared` (a bound `component.prepare`
    /// handle) on `stream` over `buffers`. `page_tokens` and `windowed`
    /// describe each numerical block table; `vocab` and `hidden_size` are
    /// what [`describe`] reported.
    #[expect(clippy::too_many_arguments, reason = "one binding's complete inputs")]
    pub fn bind(
        interface: &Interface,
        prepared: &Handle,
        stream: usize,
        buffers: &[(String, StorageView)],
        page_tokens: &[i64],
        windowed: &[i64],
        vocab: Vocab,
        hidden_size: usize,
    ) -> Result<Self> {
        let mut tensors = Fields::new();
        for (name, view) in buffers {
            tensors = tensors.with(name, view.to_tensor()?);
        }
        let layout = Fields::new()
            .with("page_tokens", Shape::from(page_tokens))
            .with("windowed", Shape::from(windowed))
            .into_any();
        let functions = Functions::new(
            "text.bind",
            interface.function("text.bind")?.call(&[
                prepared.view(),
                (&integer(stream)).into(),
                (&tensors.into_any()).into(),
                (&layout).into(),
            ])?,
        )?;
        Ok(Self {
            step: functions.get("step")?,
            capture: functions.get("capture")?,
            release: functions.get("release")?,
            commit: functions.get("commit")?,
            close: functions.get("close")?,
            vocab,
            hidden_size,
        })
    }

    /// Run one invocation.
    pub fn step(&self, args: &StepArgs<'_>) -> Result<()> {
        let graph = args.graph.map_or_else(Any::default, Handle::to_any);
        let values = [
            graph,
            flag(args.decode),
            integer(args.rows),
            integer(args.live),
            integer(args.tokens),
            flag(args.indexed),
            shape(args.widths),
            shape(args.queries),
            shape(args.prefixes),
            shape(args.starts),
            shape(args.causal),
            flag(args.writes),
            integer(args.logit_count),
            flag(args.all_last),
            flag(args.cache_only),
            shape(args.sampling),
            flag(args.greedy),
            shape(args.prompt),
        ];
        let views: Vec<AnyView<'_>> = values.iter().map(AnyView::from).collect();
        self.step.invoke(&views)
    }

    /// Capture the shapes an eager step just prepared, in `pool`.
    pub fn capture(&self, args: &CaptureArgs<'_>, pool: &Handle) -> Result<Handle> {
        let values = [
            flag(args.decode),
            integer(args.rows),
            integer(args.tokens),
            shape(args.widths),
            shape(args.queries),
            shape(args.prefixes),
            shape(args.starts),
            shape(args.causal),
            flag(args.writes),
            flag(args.cache_only),
            pool.to_any(),
        ];
        let views: Vec<AnyView<'_>> = values.iter().map(AnyView::from).collect();
        Ok(Handle::new(self.capture.call(&views)?))
    }

    /// The function releasing a captured graph.
    pub fn release(&self) -> &Function {
        &self.release
    }

    /// Install `count` staged commit rows into the slot state.
    pub fn commit(&self, count: usize) -> Result<()> {
        self.commit.invoke(&[(&integer(count)).into()])
    }

    /// Drop the binding's views. An aborted close keeps them until exit.
    pub fn close(&self, aborted: bool) -> Result<()> {
        self.close.invoke(&[(&Any::from(aborted)).into()])
    }
}
