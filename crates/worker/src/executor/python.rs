//! The rank's numerical half: the Python executor
//! (`uniserve.executor.Executor`).
//!
//! The worker owns the control and data planes (requests, batches, jobs,
//! products, transfers and memory); the executor owns every numerical decision
//! and computation of the components it serves. One executor serves a rank,
//! shared by what calls it (the executor computations, the expert union member,
//! warmup), and the backend closes it once after them. It serves each
//! computation under its index (`crate::executor::computations`). The worker
//! calls it at a few points, each a [`MethodCall`] that a job records, to be
//! performed on the builder thread behind the stream work recorded before it,
//! or a call startup makes itself:
//!
//! | call | when | result |
//! | --- | --- | --- |
//! | `manifest` | after loading | JSON of the computations and what capacity planning reads |
//! | `expert_storage_bytes`, `connect_experts` | after loading, before planning | the expert union |
//! | `bind` | after capacity planning | the computations are constructed on the granted storage |
//! | `warmup_phases`, `warmup_phase`, `warmup_finish` | before the rank reports | every computation prepared, warmed and captured |
//! | `seal` | after warmup's graph residency check | captures are refused from then on |
//! | `start` | before a batch that starts requests opens | JSON of the requests refused |
//! | `prepare` | a preparing computation's batch past its predicates | JSON of what each call reads and writes |
//! | `execute` | the job that carries a batch's device work | the computation's results |
//! | `commit` | that job, past the batch's commit point | — |
//! | `join_expert_step` | a rank with no step of its own joins the union's | whether a step ran, whether the union released |
//! | `park_experts`, `promote_experts`, `rebuild_experts` | a standby parks, is promoted | — |
//! | `release` | a slot retires | — |
//! | `kernels`, `close` | startup's report, shutdown | — |

use tvm_ffi::Any;

use crate::ffi::{Argument, MethodCall};
use crate::ffi::{Function, Handle, Interpreter};
use crate::memory::allocation::StorageView;
use crate::profiling::kernels::{KernelRecord, parse_listing};
use crate::{Error, Result};

const EXECUTOR: &str = "uniserve.executor.Executor";

/// The methods the worker calls, each with the executor first.
struct Functions {
    manifest: Function,
    bind: Function,
    warmup_phases: Function,
    warmup_phase: Function,
    warmup_finish: Function,
    seal: Function,
    run_eager: Function,
    start: Function,
    release: Function,
    prepare: Function,
    execute: Function,
    commit: Function,
    expert_storage_bytes: Function,
    connect_experts: Function,
    join_expert_step: Function,
    park_experts: Function,
    promote_experts: Function,
    rebuild_experts: Function,
    kernels: Function,
    close: Function,
}

/// The rank's Python executor, shared by what calls it.
pub(crate) struct PythonExecutor {
    executor: Handle,
    functions: Functions,
}

fn text(value: &str) -> Any {
    Any::from(tvm_ffi::String::from(value))
}

impl PythonExecutor {
    /// Construct the executor of the loaded model on this rank's
    /// deployment `config` (JSON; see `uniserve.executor.Executor`).
    pub(crate) fn new(loaded: &Handle, config: &str) -> Result<Self> {
        let method = |name: &str| Interpreter::import(&format!("{EXECUTOR}.{name}"));
        let executor = Handle::new(
            Interpreter::import(EXECUTOR)?.call_values(&[loaded.to_any(), text(config)])?,
        )?;
        Ok(Self {
            executor,
            functions: Functions {
                manifest: method("manifest")?,
                bind: method("bind")?,
                warmup_phases: method("warmup_phases")?,
                warmup_phase: method("warmup_phase")?,
                warmup_finish: method("warmup_finish")?,
                seal: method("seal")?,
                run_eager: method("run_eager")?,
                start: method("start")?,
                release: method("release")?,
                prepare: method("prepare")?,
                execute: method("execute")?,
                commit: method("commit")?,
                expert_storage_bytes: method("expert_storage_bytes")?,
                connect_experts: method("connect_experts")?,
                join_expert_step: method("join_expert_step")?,
                park_experts: method("park_experts")?,
                promote_experts: method("promote_experts")?,
                rebuild_experts: method("rebuild_experts")?,
                kernels: method("kernels")?,
                close: method("close")?,
            },
        })
    }

    fn call(&self, function: &Function, arguments: Vec<Argument>) -> MethodCall {
        let mut all = Vec::with_capacity(arguments.len() + 1);
        all.push(Argument::Handle(self.executor.clone()));
        all.extend(arguments);
        MethodCall::new(function, all)
    }

    fn texts(&self, function: &Function, arguments: Vec<Argument>) -> Result<String> {
        let value = self.call(function, arguments).invoke()?;
        tvm_ffi::String::try_from(value)
            .map(|text| text.as_str().to_owned())
            .map_err(|error| Error::numerical(function.name(), &error))
    }

    /// The integers a call made now returned: one, or a list of them.
    fn integers(&self, function: &Function, arguments: Vec<Argument>) -> Result<Vec<i64>> {
        let value = self.call(function, arguments).invoke()?;
        crate::ffi::integers(value.clone())
            .or_else(|_| i64::try_from(value).map(|value| vec![value]))
            .map_err(|error| Error::numerical(function.name(), &error))
    }

    fn unit(&self, function: &Function, arguments: Vec<Argument>) -> Result<()> {
        self.call(function, arguments).invoke().map(drop)
    }

    /// The JSON of the executor's computations and of what capacity planning
    /// and the handshake read.
    pub(crate) fn manifest(&self) -> Result<String> {
        self.texts(&self.functions.manifest, Vec::new())
    }

    /// Construct every computation on the storage capacity planning granted:
    /// `memory` is the JSON `bind` reads and `values` the handles and tensors
    /// it names, in its order (`crate::worker::computations`). Returns the JSON
    /// the executor states of its bound computations.
    pub(crate) fn bind(&self, memory: String, values: Vec<Argument>) -> Result<String> {
        let mut arguments = vec![Argument::Text(memory)];
        arguments.extend(values);
        self.texts(&self.functions.bind, arguments)
    }

    /// The JSON list of the computations' warmup phases, in the order every
    /// rank runs them: each names its computation, the scratch lengths of its
    /// slots `1..` (or none) and the attention rows of each of its steps.
    pub(crate) fn warmup_phases(&self) -> Result<String> {
        self.texts(&self.functions.warmup_phases, Vec::new())
    }

    /// The call that runs warmup phase `index` of its computation, given each
    /// of its steps' page columns.
    pub(crate) fn warmup_phase(&self, index: usize, pages: Vec<StorageView>) -> MethodCall {
        let mut arguments = vec![Argument::from(index)];
        arguments.extend(pages.into_iter().map(Argument::Tensor));
        self.call(&self.functions.warmup_phase, arguments)
    }

    /// The call that ends warmup after its phases: the captures that follow
    /// every phase and the end of the expert union's announced steps.
    pub(crate) fn warmup_finish(&self) -> MethodCall {
        self.call(&self.functions.warmup_finish, Vec::new())
    }

    /// Refuse further captures: every graph serving replays is resident.
    pub(crate) fn seal(&self) -> Result<()> {
        self.unit(&self.functions.seal, Vec::new())
    }

    /// Serve every forward and expert join eagerly (`eager`), or by the
    /// graphs captured at startup again (`uniserve.executor.run_eager`).
    /// Fails, switching nothing, for a rank whose computations serve by
    /// their graphs alone.
    pub(crate) fn run_eager(&self, eager: bool) -> Result<()> {
        self.unit(&self.functions.run_eager, vec![Argument::Bool(eager)])
    }

    /// The call that admits started requests (`requests`: JSON of their
    /// `NewRequest`s), whose completion's text is the JSON of the refused.
    pub(crate) fn start(&self, requests: String) -> MethodCall {
        self.call(&self.functions.start, vec![Argument::Text(requests)])
            .serving()
    }

    /// The call that prepares a batch of computation `computation` (`step`:
    /// JSON), whose completion's text is the JSON of what each call reads and
    /// writes.
    pub(crate) fn prepare(&self, computation: usize, step: String) -> MethodCall {
        self.call(
            &self.functions.prepare,
            vec![Argument::from(computation), Argument::Text(step)],
        )
        .serving()
    }

    /// The call that executes a batch of computation `computation` (`step`:
    /// JSON) on `tensors`, whose completion's integers are the computation's
    /// results: the record words it wrote, or each call's graph mode for a
    /// computation that prepares.
    pub(crate) fn execute(
        &self,
        computation: usize,
        step: String,
        tensors: Vec<StorageView>,
    ) -> MethodCall {
        let mut arguments = vec![Argument::from(computation), Argument::Text(step)];
        arguments.extend(tensors.into_iter().map(Argument::Tensor));
        self.call(&self.functions.execute, arguments).serving()
    }

    /// The call that advances computation `computation`'s slot state on the
    /// device by its last executed batch, past that batch's commit point.
    pub(crate) fn commit(&self, computation: usize) -> MethodCall {
        self.call(&self.functions.commit, vec![Argument::from(computation)])
            .serving()
    }

    /// Configure the expert union's step agreement over the union's host
    /// functions (`crate::parallel::host::bootstrap`), collectively, and
    /// return the bytes of the peer-mapped storage each microbatch's
    /// exchange needs (zero for none).
    pub(crate) fn expert_storage_bytes(&self, host: Vec<Any>) -> Result<usize> {
        let mut arguments = vec![self.executor.to_any()];
        arguments.extend(host);
        let function = &self.functions.expert_storage_bytes;
        let bytes: i64 = Interpreter::execute(|| function.call_values(&arguments))?
            .try_into()
            .map_err(|error| Error::numerical(function.name(), &error))?;
        usize::try_from(bytes)
            .map_err(|_| Error::Interface(format!("{} returns a byte count", function.name())))
    }

    /// Create the expert union's exchanges, collectively: `memory` is the
    /// JSON `connect_experts` reads (each microbatch's stream and peer
    /// addresses) and `storages` each microbatch's peer-mapped storage.
    pub(crate) fn connect_experts(
        &self,
        memory: String,
        storages: Vec<Option<StorageView>>,
    ) -> Result<()> {
        let mut arguments = vec![Argument::Text(memory)];
        arguments.extend(storages.into_iter().map(|storage| match storage {
            Some(view) => Argument::Tensor(view),
            None => Argument::None,
        }));
        self.unit(&self.functions.connect_experts, arguments)
    }

    /// The call that joins the expert union's next step for a rank with no
    /// step of its own (`leaving` once it shuts down); its completion's
    /// integers are whether a step ran and whether every rank left.
    pub(crate) fn join_expert_step(&self, leaving: bool) -> MethodCall {
        self.call(
            &self.functions.join_expert_step,
            vec![Argument::Bool(leaving)],
        )
        .serving()
    }

    /// Release the expert transports' workspaces while a standby parks,
    /// collectively over the union: those that keep their addresses are
    /// suspended, the others freed. Returns whether any was suspended and
    /// whether any was freed, which the promotion rebuilds.
    pub(crate) fn park_experts(&self) -> Result<(bool, bool)> {
        match self.integers(&self.functions.park_experts, Vec::new())?[..] {
            [suspended, released] => Ok((suspended != 0, released != 0)),
            _ => Err(Error::Interface(
                "park_experts returns whether it suspended and whether it released".into(),
            )),
        }
    }

    /// Map the suspended expert workspaces back at their addresses,
    /// collectively, at a standby's promotion.
    pub(crate) fn promote_experts(&self) -> Result<()> {
        self.unit(&self.functions.promote_experts, Vec::new())
    }

    /// Create the freed expert workspaces again and release every graph
    /// that addressed them, collectively: the warmup that follows captures
    /// them anew.
    pub(crate) fn rebuild_experts(&self) -> Result<()> {
        self.unit(&self.functions.rebuild_experts, Vec::new())
    }

    /// Forget retired slots. The release is queued on the builder thread
    /// behind every call made before it; nothing waits for it.
    pub(crate) fn release(&self, slots: &[usize]) {
        let call = self
            .call(
                &self.functions.release,
                vec![Argument::Integers(
                    slots.iter().map(|&slot| slot as i64).collect(),
                )],
            )
            .serving();
        Interpreter::submit(move || {
            if let Err(error) = call.invoke() {
                tracing::error!("releasing retired slots failed: {error}");
            }
        });
    }

    /// The kernels each served component's operators selected, by
    /// component.
    pub(crate) fn kernels(&self) -> Result<Vec<(String, Vec<KernelRecord>)>> {
        parse_listing(&self.texts(&self.functions.kernels, Vec::new())?)
    }

    /// The call that lists the kernels while the rank serves, whose
    /// completion's text is the listing [`Self::kernels`] reads.
    pub(crate) fn kernels_call(&self) -> MethodCall {
        self.call(&self.functions.kernels, Vec::new())
    }

    /// Release every computation and the expert union after the jobs drained.
    /// An aborted close keeps what device work may still address.
    pub(crate) fn close(&self, aborted: bool) -> Result<()> {
        self.unit(&self.functions.close, vec![Argument::Bool(aborted)])
    }
}
