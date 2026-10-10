//! The typed canvas binding of a token denoiser (`canvas.bind`).
//!
//! [`CanvasFacts::describe`] reads what the runtime sizes its buffers and
//! graphs from (`canvas.describe`); [`CanvasModel::bind`] resolves the step
//! functions of one lane binding over Rust-owned buffers, failing startup
//! naming any missing function, and converts each call's integer
//! arguments. The Python contract is documented in
//! `uniserve_models/interface/canvas.py`.

use tvm_ffi::{Any, AnyView};

use crate::cuda_graph::CUDAGraph;
use crate::ffi::{DType, Fields, Function, Functions, Handle, Record, Shape};
use crate::storage::memory::StorageView;
use crate::{Error, Result};

use super::interface::Interface;

/// What a token denoiser's rank computes, from `canvas.describe`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanvasFacts {
    /// Tokens of one generated canvas (`C`).
    pub canvas_length: usize,
    /// Real vocabulary entries (`V`), on every pipeline stage.
    pub vocab_size: usize,
    /// Hidden width of the token embeddings (`H`).
    pub hidden_size: usize,
    /// The self-conditioning embedding dtype.
    pub dtype: DType,
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: u32,
    /// Per-token widths of the final stage's attend state; empty elsewhere.
    pub attend_widths: Vec<usize>,
    pub attend_dtype: DType,
    /// This rank steps resident canvases (one pipeline stage holding the
    /// unsharded embedding table); otherwise it serves readouts only.
    pub generates: bool,
    /// The readout tail exchanges no tokens across ranks, so it may replay
    /// captured slot buckets in chunks.
    pub local_tail: bool,
    pub pipeline_rank: usize,
    pub pipeline_size: usize,
}

fn dtype(record: &Record, key: &str) -> Result<DType> {
    let name = record.string(key)?;
    DType::from_name(&name).ok_or_else(|| {
        Error::Interface(format!(
            "canvas.describe reports the unknown {key} {name:?}"
        ))
    })
}

fn token(value: i64) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::Interface("canvas.describe reports an invalid token id".into()))
}

impl CanvasFacts {
    /// Describe the token denoiser at `path` of the loaded model. `exchange`
    /// says routed experts use an expert exchange, and `attention_ranks`
    /// counts the attention ranks of a disaggregated expert union (zero
    /// when every rank holds its experts).
    pub fn describe(
        interface: &Interface,
        loaded: &Handle,
        path: &str,
        exchange: bool,
        attention_ranks: usize,
    ) -> Result<Self> {
        let request = Fields::new()
            .text("path", path)
            .with("exchange", exchange)
            .with("attention_ranks", attention_ranks as i64)
            .into_any();
        let record = Record::new(
            "canvas.describe",
            interface
                .function("canvas.describe")?
                .call(&[loaded.view(), (&request).into()])?,
        )?;
        let facts = Self {
            canvas_length: record.count("canvas_length")?,
            vocab_size: record.count("vocab_size")?,
            hidden_size: record.count("hidden_size")?,
            dtype: dtype(&record, "dtype")?,
            eos_token_ids: record
                .shape("eos_token_ids")?
                .into_iter()
                .map(token)
                .collect::<Result<_>>()?,
            pad_token_id: token(record.get("pad_token_id")?)?,
            attend_widths: record
                .shape("attend_widths")?
                .into_iter()
                .map(|width| {
                    usize::try_from(width).map_err(|_| {
                        Error::Interface("canvas.describe reports a negative width".into())
                    })
                })
                .collect::<Result<_>>()?,
            attend_dtype: dtype(&record, "attend_dtype")?,
            generates: record.flag("generates")?,
            local_tail: record.flag("local_tail")?,
            pipeline_rank: record.count("pipeline_rank")?,
            pipeline_size: record.count("pipeline_size")?,
        };
        if facts.canvas_length == 0
            || facts.hidden_size == 0
            || facts.vocab_size == 0
            || facts.pipeline_rank >= facts.pipeline_size
            || (facts.last_stage() && facts.attend_widths.is_empty())
        {
            return Err(Error::Interface(
                "canvas.describe reports an invalid token denoiser".into(),
            ));
        }
        Ok(facts)
    }

    /// The stage that holds the head and evaluates readout tails.
    pub fn last_stage(&self) -> bool {
        self.pipeline_rank + 1 == self.pipeline_size
    }
}

/// Bytes of a step chunk's self-conditioning product scratch for
/// `positions` canvas positions on CUDA `device` (`None`: the host).
pub fn scratch_bytes(
    interface: &Interface,
    positions: usize,
    hidden_size: usize,
    device: Option<i32>,
) -> Result<usize> {
    let bytes: i64 = interface.function("canvas.scratch_bytes")?.call_as(&[
        (&Any::from(positions as i64)).into(),
        (&Any::from(hidden_size as i64)).into(),
        (&Any::from(i64::from(device.unwrap_or(-1)))).into(),
    ])?;
    usize::try_from(bytes)
        .map_err(|_| Error::Interface("canvas.scratch_bytes reports a negative size".into()))
}

/// One numerical block table, as the canvas attention reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableLayout {
    pub page_tokens: u32,
    pub windowed: bool,
}

/// The served block-diffusion sampling a generating binding steps with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepOptions {
    pub sampling: uniserve_core::CanvasSampling,
    /// Canvases the head projects and the sampler steps at a time.
    pub step_rows: usize,
}

/// The host attention metadata of one canvas invocation.
///
/// Every list mirrors a device column the runtime wrote before the call:
/// `queries` and `prefixes` per sequence, `widths` per table and `starts`
/// per table and sequence (`[table][row]` flattened).
#[derive(Debug, Clone, Copy)]
pub struct Attention<'a> {
    pub rows: usize,
    /// Fixed visibility extent: every sequence's query tokens fit in it.
    pub visible: usize,
    pub queries: &'a [i64],
    pub prefixes: &'a [i64],
    pub widths: &'a [i64],
    pub starts: &'a [i64],
}

impl Attention<'_> {
    fn arguments(&self) -> [Any; 4] {
        [
            Any::from(Shape::from(self.queries)),
            Any::from(Shape::from(self.prefixes)),
            Any::from(Shape::from(self.widths)),
            Any::from(Shape::from(self.starts)),
        ]
    }

    fn replay_arguments(&self) -> [Any; 3] {
        [
            Any::from(Shape::from(self.queries)),
            Any::from(Shape::from(self.prefixes)),
            Any::from(Shape::from(self.starts)),
        ]
    }
}

/// A captured canvas function's shape.
#[derive(Debug, Clone, Copy)]
pub enum Capture<'a> {
    /// The pass up to the final layer's attention over `tokens` tokens.
    Attend {
        tokens: usize,
        attention: Attention<'a>,
    },
    /// The readout tail of `slots` slot rows.
    Tail { slots: usize },
    /// One denoising step; `first` when every row is at step zero.
    Step {
        first: bool,
        attention: Attention<'a>,
    },
}

/// A bound canvas binding of one lane.
pub struct CanvasModel {
    readout_attend: Function,
    readout_candidates: Function,
    readout_finish: Function,
    readout_broadcast: Function,
    step: Option<Function>,
    capture: Function,
    replay: Function,
    release: Function,
    kernels: Function,
}

fn integer(value: usize) -> Any {
    Any::from(value as i64)
}

impl CanvasModel {
    /// Bind the canvas functions of `prepared` (a bound `component.prepare`
    /// handle) over `buffers`. `step` is present on generating ranks.
    pub fn bind(
        interface: &Interface,
        prepared: &Handle,
        buffers: &[(String, StorageView)],
        tables: &[TableLayout],
        device: Option<i32>,
        step: Option<StepOptions>,
    ) -> Result<Self> {
        let mut tensors = Fields::new();
        for (name, view) in buffers {
            tensors = tensors.with(name, view.to_tensor()?);
        }
        let tables = crate::ffi::list(tables.iter().map(|table| {
            Fields::new()
                .with("page_tokens", i64::from(table.page_tokens))
                .with("windowed", table.windowed)
                .into_any()
        }));
        let mut options = Fields::new()
            .with("tables", tables)
            .with("device", i64::from(device.unwrap_or(-1)))
            .with(
                "step_rows",
                step.map_or(0, |options| options.step_rows as i64),
            );
        if let Some(options_value) = step {
            let sampling = options_value.sampling;
            options = options.with(
                "sampling",
                Fields::new()
                    .with("max_steps", i64::from(sampling.max_steps))
                    .with("entropy_bound", f64::from(sampling.entropy_bound))
                    .with("t_min", f64::from(sampling.t_min))
                    .with("t_max", f64::from(sampling.t_max))
                    .with(
                        "confidence_threshold",
                        f64::from(sampling.confidence_threshold),
                    )
                    .with(
                        "stability_threshold",
                        i64::from(sampling.stability_threshold),
                    )
                    .map(),
            );
        }
        let functions = Functions::new(
            "canvas.bind",
            interface.function("canvas.bind")?.call(&[
                prepared.view(),
                (&tensors.into_any()).into(),
                (&options.into_any()).into(),
            ])?,
        )?;
        Ok(Self {
            readout_attend: functions.get("readout_attend")?,
            readout_candidates: functions.get("readout_candidates")?,
            readout_finish: functions.get("readout_finish")?,
            readout_broadcast: functions.get("readout_broadcast")?,
            step: match step {
                Some(_) => Some(functions.get("step")?),
                None => None,
            },
            capture: functions.get("capture")?,
            replay: functions.get("replay")?,
            release: functions.get("release")?,
            kernels: functions.get("kernels")?,
        })
    }

    /// The function graphs are released with, for the runner's graph table.
    pub fn release(&self) -> Function {
        self.release.clone()
    }

    /// Evaluate `tokens` canvas tokens up to the final layer's attention,
    /// as the expert step of `expert_capacity` (zero without experts).
    pub fn readout_attend(
        &self,
        tokens: usize,
        attention: &Attention<'_>,
        expert_capacity: usize,
    ) -> Result<()> {
        let [queries, prefixes, widths, starts] = attention.arguments();
        self.readout_attend.call_values(&[
            integer(attention.rows),
            integer(tokens),
            integer(attention.visible),
            queries,
            prefixes,
            widths,
            starts,
            integer(expert_capacity),
        ])?;
        Ok(())
    }

    /// Select candidates `start..start + count` from a tail's output (a
    /// replayed tail graph or an eager tail).
    pub fn readout_candidates(&self, tail: &Handle, start: usize, count: usize) -> Result<()> {
        self.readout_candidates.invoke(&[
            tail.view(),
            (&integer(start)).into(),
            (&integer(count)).into(),
        ])
    }

    /// Complete every slot of a readout eagerly on the final stage; the
    /// returned tail's candidates are read with [`Self::readout_candidates`]
    /// at flat indices `slot * vocab + token`.
    /// An exchanging tail completes the readout's expert step of
    /// `expert_capacity`.
    pub fn readout_finish(&self, slots: usize, expert_capacity: usize) -> Result<Handle> {
        Handle::new(
            self.readout_finish
                .call_values(&[integer(slots), integer(expert_capacity)])?,
        )
    }

    /// Share the final stage's `count` candidate values with every stage.
    pub fn readout_broadcast(&self, count: usize) -> Result<()> {
        self.readout_broadcast.invoke(&[(&integer(count)).into()])
    }

    /// Step `attention.rows` gathered canvases eagerly, as the expert step
    /// of `expert_capacity`.
    pub fn step(
        &self,
        first: bool,
        attention: &Attention<'_>,
        expert_capacity: usize,
    ) -> Result<()> {
        let step = self
            .step
            .as_ref()
            .ok_or_else(|| Error::Invalid("this worker keeps no generating canvases".into()))?;
        let [queries, prefixes, widths, starts] = attention.arguments();
        step.call_values(&[
            integer(attention.rows),
            Any::from(first),
            queries,
            prefixes,
            widths,
            starts,
            integer(expert_capacity),
        ])?;
        Ok(())
    }

    /// Capture one canvas function into `pool` as the expert step of
    /// `expert_capacity`; the function also runs once eagerly before the
    /// capture.
    pub fn capture(
        &self,
        capture: Capture<'_>,
        pool: &Handle,
        expert_capacity: usize,
    ) -> Result<CUDAGraph> {
        let (name, arguments): (&str, Vec<Any>) = match capture {
            Capture::Attend { tokens, attention } => {
                let [queries, prefixes, widths, starts] = attention.arguments();
                (
                    "readout_attend",
                    vec![
                        integer(attention.rows),
                        integer(tokens),
                        integer(attention.visible),
                        queries,
                        prefixes,
                        widths,
                        starts,
                    ],
                )
            }
            Capture::Tail { slots } => ("readout_tail", vec![integer(slots)]),
            Capture::Step { first, attention } => {
                let [queries, prefixes, widths, starts] = attention.arguments();
                (
                    "step",
                    vec![
                        integer(attention.rows),
                        Any::from(first),
                        queries,
                        prefixes,
                        widths,
                        starts,
                    ],
                )
            }
        };
        let arguments = crate::ffi::list(arguments);
        let graph = self.capture.call(&[
            (&Any::from(tvm_ffi::String::from(name))).into(),
            (&Any::from(arguments)).into(),
            pool.view(),
            (&integer(expert_capacity)).into(),
        ])?;
        CUDAGraph::captured(graph)
    }

    /// Replay a captured function, planning its attention for the live host
    /// metadata first (`None` for a tail).
    pub fn replay(&self, graph: &Handle, attention: Option<&Attention<'_>>) -> Result<()> {
        let none = Any::from(());
        let [queries, prefixes, starts] = attention.map_or_else(
            || [none.clone(), none.clone(), none.clone()],
            Attention::replay_arguments,
        );
        let arguments: [AnyView<'_>; 4] = [
            graph.view(),
            (&queries).into(),
            (&prefixes).into(),
            (&starts).into(),
        ];
        self.replay.invoke(&arguments)
    }

    /// The canvas kernel records of this binding.
    pub fn kernels(&self) -> Result<Vec<Record>> {
        let value: tvm_ffi::Array<Any> = self.kernels.call_as(&[])?;
        value
            .iter()
            .enumerate()
            .map(|(index, record)| Record::new(format!("canvas.kernels[{index}]"), record))
            .collect()
    }
}
