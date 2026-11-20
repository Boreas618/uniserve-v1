//! A control program's life on the leader: one task per program that steps it every `period`
//! and hands its output to the node's apply, one call at a time.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use program::{Error, Instance, Level, Module, State};
use scheduling::time::Instant;
use tokio::sync::{mpsc, oneshot, watch};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use super::host::{Logs, Reader};
use super::{CONTROL, Current, Identity, Loaded, Output, States};
use crate::observation::{Fields, Store};

/// What a control program's task starts from.
pub(super) struct Start {
    pub(super) program: Loaded,
    pub(super) period: Duration,
    pub(super) store: Store,
    pub(super) states: States,
    pub(super) current: watch::Receiver<Arc<Current>>,
    pub(super) outputs: mpsc::UnboundedSender<Output>,
    /// What its validation measures of a change that starts a kernel generation.
    pub(super) measure: super::host::Measure,
    /// The leader's epoch records, which its validation compares a configuration against.
    pub(super) epochs: Arc<crate::epoch::Epochs>,
}

/// A running control program: the task that steps it, its period, and what ends it.
pub(super) struct Task {
    pub(super) identity: Identity,
    period: watch::Sender<Duration>,
    cancel: CancellationToken,
    handle: tokio::task::JoinHandle<()>,
}

impl Task {
    /// Starts the program: creates its instance, restoring its saved state when that decodes,
    /// and then the task that steps it every period until stopped. A program that does not start
    /// is recorded, and its task steps nothing.
    pub(super) async fn start(start: Start) -> Self {
        let identity = start.program.identity();
        let (period, periods) = watch::channel(start.period);
        let cancel = CancellationToken::new();
        let instance = instantiate(&start).await;
        let handle = tokio::spawn(run(start, instance, periods, cancel.clone()));
        Self {
            identity,
            period,
            cancel,
            handle,
        }
    }

    /// Steps the program every `period` from its next call on.
    pub(super) fn set_period(&self, period: Duration) {
        self.period.send_replace(period);
    }

    /// Ends the program between two calls, its state saved, and waits until it has.
    pub(super) async fn stop(self) {
        self.cancel.cancel();
        if let Err(error) = self.handle.await {
            warn!("a control program's task did not end cleanly: {error}");
        }
    }
}

/// The instance of `start`'s program, created on a blocking thread, since compiling its module
/// takes a while; `None`, recorded, when it does not start.
async fn instantiate(start: &Start) -> Option<Instance<Reader>> {
    let name = start.program.name.clone();
    let logs = Logs::new(name.clone(), start.store.clone(), start.current.clone());
    let (program, states) = (start.program.clone(), start.states.clone());
    let (store, current) = (start.store.clone(), start.current.clone());
    let measure = Arc::clone(&start.measure);
    let epochs = Arc::clone(&start.epochs);
    let reader = move || Reader {
        logs: Logs::new(name.clone(), store.clone(), current.clone()),
        measure: Arc::clone(&measure),
        epochs: Arc::clone(&epochs),
    };
    let created =
        tokio::task::spawn_blocking(move || create(&program, &states, &logs, reader)).await;
    match created {
        Ok(Ok(instance)) => Some(instance),
        Ok(Err(reason)) => {
            Logs::new(
                start.program.name.clone(),
                start.store.clone(),
                start.current.clone(),
            )
            .failure("ended", &format!("the program did not start: {reason}"));
            None
        }
        Err(error) => {
            warn!(
                program = start.program.name,
                "a control program's start did not complete: {error}"
            );
            None
        }
    }
}

/// The task of one control program, which steps `instance` every period.
async fn run(
    start: Start,
    instance: Option<Instance<Reader>>,
    mut periods: watch::Receiver<Duration>,
    cancel: CancellationToken,
) {
    let Start {
        program,
        store,
        states,
        current,
        outputs,
        ..
    } = start;
    let name = program.name.clone();
    let logs = || Logs::new(name.clone(), store.clone(), current.clone());
    let Some(mut instance) = instance else {
        cancel.cancelled().await;
        return;
    };

    loop {
        let period = *periods.borrow_and_update();
        tokio::select! {
            () = cancel.cancelled() => break,
            () = tokio::time::sleep(period) => {}
            // A new period restarts the wait.
            changed = periods.changed() => {
                if changed.is_err() {
                    break;
                }
                continue;
            }
        }

        let input = Arc::clone(&current.borrow());
        let config = input.package.config.clone();
        let stepped = {
            let name = name.clone();
            tokio::task::spawn_blocking(move || (step(&mut instance, &name, &config), instance))
                .await
        };
        let (result, returned) = match stepped {
            Ok(stepped) => stepped,
            Err(error) => {
                warn!(
                    program = name,
                    "a control program's step did not complete: {error}"
                );
                return;
            }
        };
        instance = returned;

        match result {
            Ok(Some(output)) if output != input.package.config => {
                let (done, answer) = oneshot::channel();
                let sent = outputs.send(Output {
                    writer: name.clone(),
                    input: input.package.config.clone(),
                    output,
                    done,
                });
                if sent.is_err() {
                    break;
                }
                // The next call waits for this output's change to end, so it reads its result.
                let answered = tokio::select! {
                    () = cancel.cancelled() => break,
                    answered = answer => answered,
                };
                match answered {
                    Ok(Ok(revision)) => {
                        info!(program = name, revision, "the program's output converged");
                    }
                    Ok(Err(reason)) => logs().failure("refused", &reason),
                    Err(_) => break,
                }
            }
            Ok(_) => {}
            Err(failure) => {
                logs().failure("ended", &failure);
                // An ended instance is called no more; what it wrote stays.
                if instance.ended().is_some() {
                    cancel.cancelled().await;
                    return;
                }
            }
        }
        save(&mut instance, &states, &name, &program.entry.args, &logs());
    }
    save(&mut instance, &states, &name, &program.entry.args, &logs());
}

/// The program's instance: restored from its saved state when there is one that decodes as the
/// program's type, and created from its arguments otherwise. Runs on a blocking thread, since
/// compiling the module takes a while.
pub(super) fn create<H: program::Host>(
    program: &Loaded,
    states: &States,
    logs: &Logs,
    host: impl Fn() -> H,
) -> Result<Instance<H>, String>
where
    Module<H>: Linked,
{
    let module = <Module<H> as Linked>::link(&program.bytes).map_err(|error| error.to_string())?;
    let budget = <Module<H> as Linked>::budget();
    let saved = states
        .load(&program.name, &program.entry.args)
        .map_err(|error| format!("{error:#}"))?;
    if let Some(saved) = saved {
        match Instance::restore(&module, host(), &budget, &saved) {
            Ok(instance) => {
                logs.record(Level::Info, "restored from its saved state", Fields::new());
                return Ok(instance);
            }
            Err(Error::State(reason)) => logs.record(
                Level::Info,
                &format!("its saved state does not decode as its type ({reason}): created anew"),
                Fields::new(),
            ),
            Err(error) => return Err(error.to_string()),
        }
    }
    Instance::new(&module, host(), &budget, &program.entry.args).map_err(|error| error.to_string())
}

/// How a program module is linked for its embedder, and the budget of its calls.
pub(super) trait Linked: Sized {
    fn link(bytes: &[u8]) -> Result<Self, Error>;
    fn budget() -> program::Budget;
}

impl Linked for Module<Reader> {
    fn link(bytes: &[u8]) -> Result<Self, Error> {
        Module::control(bytes)
    }

    fn budget() -> program::Budget {
        CONTROL
    }
}

impl Linked for Module<Logs> {
    fn link(bytes: &[u8]) -> Result<Self, Error> {
        Module::new(bytes)
    }

    fn budget() -> program::Budget {
        super::ROUTING
    }
}

/// One step of program `name`: the input of the instant now with `config`, the deployment's
/// whole configuration as JSON; the output read back as a configuration, or why the step failed
/// or its output does not read.
fn step(
    instance: &mut Instance<Reader>,
    name: &str,
    config: &crate::config::Config,
) -> Result<Option<crate::config::Config>, String> {
    let time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |since| {
            u64::try_from(since.as_nanos()).unwrap_or(u64::MAX)
        });
    let json = serde_json::to_value(config)
        .map_err(|error| format!("the configuration has no JSON form: {error}"))?;
    let state = State::new(name, Instant(time), program::Config(json));
    let output: Option<program::Config> =
        instance.step(&state).map_err(|error| error.to_string())?;
    output
        .map(|output| {
            serde_json::from_value(output.0)
                .map_err(|error| format!("its output is no configuration: {error}"))
        })
        .transpose()
}

/// Saves the state of program `name`'s instance, which runs with `args`, in `states`, recording
/// why when it cannot.
pub(super) fn save<H: program::Host>(
    instance: &mut Instance<H>,
    states: &States,
    name: &str,
    args: &toml::Table,
    logs: &Logs,
) {
    if instance.ended().is_some() {
        return;
    }
    let saved = instance
        .save()
        .map_err(|error| error.to_string())
        .and_then(|state| {
            states
                .save(name, args, &state)
                .map_err(|error| format!("{error:#}"))
        });
    if let Err(reason) = saved {
        logs.failure("state", &format!("its state was not saved: {reason}"));
    }
}
