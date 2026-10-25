//! A profiler window over executed batches, configured by environment.
//!
//! `UNISERVE_TORCH_PROFILER_DIR` enables it. Each executed batch with calls,
//! warmup batches included, is one step; the window opens at step
//! `UNISERVE_PROFILE_START_STEP` (default 1) and closes after
//! `UNISERVE_PROFILE_STEPS` steps (default 1), or at shutdown with a partial
//! window. The torch profiler records `UNISERVE_PROFILE_ACTIVITIES` (`CPU`,
//! `GPU`; default both), optionally with stacks and shapes, and the window
//! exports `{prefix}-pid{pid}[-rank{RANK}]-steps{a}-{b}-{YYYYmmdd-HHMMSS}`
//! `.trace.json.gz` and `.summary.txt`. `UNISERVE_CUDA_PROFILER` also opens a
//! CUDA profiler capture range. A failure to start is logged once and the
//! window is never retried; profiling never fails a batch.
//!
//! The window opens before the forward of its first step and closes after
//! the forward of its last, as jobs that batch's forward waits for like its
//! own ([`Profiler::next`], [`Profiler::complete`]). The torch profiler's
//! start, and its stop with the export, are calls submitted to the Python
//! thread. The CUDA capture range starts and ends in a job of the batch's
//! lane, whose execution thread makes the driver call. The control thread
//! makes neither call itself while it serves.

use std::collections::VecDeque;
use std::path::PathBuf;

use crossbeam_channel::{Receiver, Sender};
use cuda::DeviceId;

use crate::ffi::{Function, Handle};
use crate::lane::{Completion, Job, JobId, JobQueues};
use crate::runner::methods::{Argument, MethodCall};
use crate::stream::Issue;
use crate::{Error, Result};

use super::flag;

/// Window settings read from the environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProfileConfig {
    pub output_dir: PathBuf,
    pub prefix: String,
    pub activities: Vec<String>,
    pub start_step: usize,
    pub num_steps: usize,
    pub with_stack: bool,
    pub record_shapes: bool,
    pub cuda_profiler: bool,
}

impl ProfileConfig {
    /// Read the settings through `value`; `None` when profiling is off.
    pub(crate) fn from_env(value: impl Fn(&str) -> Option<String>) -> Result<Option<Self>> {
        let Some(directory) =
            value("UNISERVE_TORCH_PROFILER_DIR").filter(|value| !value.is_empty())
        else {
            return Ok(None);
        };
        let raw = value("UNISERVE_PROFILE_ACTIVITIES").unwrap_or_else(|| "CPU,GPU".to_owned());
        let mut activities = Vec::new();
        for name in raw.replace(',', " ").split_whitespace() {
            let name = name.to_ascii_uppercase();
            if !matches!(name.as_str(), "CPU" | "GPU") {
                return Err(Error::Invalid(format!(
                    "unknown profiler activity {name:?}; expected CPU or GPU"
                )));
            }
            if !activities.contains(&name) {
                activities.push(name);
            }
        }
        let steps = |name| {
            value(name)
                .and_then(|value| value.trim().parse::<i64>().ok())
                .unwrap_or(1)
                .max(1) as usize
        };
        Ok(Some(Self {
            output_dir: directory.into(),
            prefix: value("UNISERVE_PROFILE_PREFIX")
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "uniserve-worker".to_owned()),
            activities,
            start_step: steps("UNISERVE_PROFILE_START_STEP"),
            num_steps: steps("UNISERVE_PROFILE_STEPS"),
            with_stack: flag(value("UNISERVE_PROFILE_WITH_STACK").as_deref()),
            record_shapes: flag(value("UNISERVE_PROFILE_RECORD_SHAPES").as_deref()),
            cuda_profiler: flag(value("UNISERVE_CUDA_PROFILER").as_deref()),
        }))
    }
}

/// One job of the window's opening or closing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Start the torch profiler: a call on the builder thread.
    Start,
    /// Start the CUDA capture range: a driver call on the lane's thread.
    StartRange,
    /// Stop a torch profiler whose capture range did not start.
    Discard,
    /// End the CUDA capture range.
    StopRange,
    /// Stop the torch profiler and export the window.
    Export,
}

/// The step counter and the window.
pub(crate) struct Profiler {
    config: ProfileConfig,
    start: Function,
    stop: Function,
    /// The CUDA device a capture range applies to.
    device: Option<DeviceId>,
    seen: usize,
    captured: usize,
    first: usize,
    /// The window is open, or opening: steps count toward its length.
    active: bool,
    finished: bool,
    torch: Option<Handle>,
    /// The CUDA capture range was started.
    ranging: bool,
    /// The jobs still to submit to open or close the window, in order.
    phases: VecDeque<Phase>,
    /// The phase whose job is in flight.
    submitted: Option<Phase>,
    /// Where the lane's thread reports a capture range's driver call that
    /// failed.
    failures: (Sender<Error>, Receiver<Error>),
}

impl Profiler {
    pub(crate) fn new(
        config: ProfileConfig,
        start: Function,
        stop: Function,
        device: Option<DeviceId>,
    ) -> Self {
        Self {
            config,
            start,
            stop,
            device,
            seen: 0,
            captured: 0,
            first: 0,
            active: false,
            finished: false,
            torch: None,
            ranging: false,
            phases: VecDeque::new(),
            submitted: None,
            failures: crossbeam_channel::unbounded(),
        }
    }

    /// Account the start of one executed batch: its forward begins. At the
    /// window's first step the window is to open before that forward
    /// ([`Self::next`]).
    pub(crate) fn begin(&mut self) {
        self.seen += 1;
        if self.active || self.finished || self.seen < self.config.start_step {
            return;
        }
        self.first = self.seen;
        self.active = true;
        self.phases.extend([Phase::Start, Phase::StartRange]);
    }

    /// Account the end of the batch [`Self::begin`] started. After the
    /// window's last step the window is to close ([`Self::next`]).
    pub(crate) fn end(&mut self) {
        if self.active {
            self.captured += 1;
            if self.captured >= self.config.num_steps {
                self.finish();
            }
        }
    }

    /// The window takes no further step: what closes it is submitted next.
    fn finish(&mut self) {
        self.active = false;
        self.finished = true;
        self.phases.extend([Phase::StopRange, Phase::Export]);
    }

    /// Submit the next job of the window's opening or closing: a job of
    /// the lane, or calls for the builder thread. `None` when
    /// nothing is left to submit, or while the last one is in flight. The
    /// batch whose forward this is waits for the job as for its own and
    /// hands its completion to [`Self::complete`].
    pub(crate) fn next(&mut self, jobs: &JobQueues) -> Option<JobId> {
        if self.submitted.is_some() {
            return None;
        }
        while let Some(phase) = self.phases.pop_front() {
            match self.submit(phase, jobs) {
                Ok(Some(job)) => {
                    self.submitted = Some(phase);
                    return Some(job);
                }
                // The phase has nothing to do in this window.
                Ok(None) => {}
                Err(error) => self.failed(phase, &error),
            }
        }
        None
    }

    /// Apply the completion of the job [`Self::next`] submitted last.
    pub(crate) fn complete(&mut self, completion: Completion) {
        let Some(phase) = self.submitted.take() else {
            return;
        };
        let completed = completion.check().map_err(|failure| failure.error);
        let applied = match phase {
            Phase::Start => completed
                .and_then(|mut completion| completion.handle())
                .map(|profile| {
                    // No requested activity is available: nothing records.
                    if !profile.is_none() {
                        self.torch = Some(profile);
                    }
                }),
            Phase::StartRange => self.ranged(completed).map(|()| self.ranging = true),
            Phase::StopRange => self.ranged(completed),
            Phase::Discard | Phase::Export => completed.map(drop),
        };
        if let Err(error) = applied {
            self.failed(phase, &error);
        }
    }

    /// How a capture range's job ended: its own failure, or the failure
    /// its driver call reported.
    fn ranged(&self, completed: Result<Completion>) -> Result<()> {
        completed?;
        self.failures.1.try_recv().map_or(Ok(()), Err)
    }

    /// Submit `phase`'s job; `None` when the phase has nothing to do.
    fn submit(&mut self, phase: Phase, jobs: &JobQueues) -> Result<Option<JobId>> {
        let range = self.device.filter(|_| self.config.cuda_profiler);
        match phase {
            Phase::Start => {
                std::fs::create_dir_all(&self.config.output_dir).map_err(|error| {
                    Error::transport_source(
                        format!("creating {}", self.config.output_dir.display()),
                        error,
                    )
                })?;
                let calls = Job::new();
                calls.call_handle(self.start_call());
                Ok(Some(jobs.call(calls, true)))
            }
            Phase::StartRange => match range {
                Some(device) => self.range(jobs, move || cuda::profiler_start(device)),
                None => Ok(None),
            },
            Phase::StopRange => match range.filter(|_| std::mem::take(&mut self.ranging)) {
                Some(device) => self.range(jobs, move || cuda::profiler_stop(device)),
                None => Ok(None),
            },
            Phase::Discard => Ok(self.torch.take().map(|profile| {
                let calls = Job::new();
                calls.call(self.discard_call(profile));
                jobs.call(calls, true)
            })),
            Phase::Export => Ok(self.torch.take().map(|profile| {
                let calls = Job::new();
                calls.call(self.export_call(profile));
                jobs.call(calls, true)
            })),
        }
    }

    /// A job of the lane that makes `call`, a driver call of
    /// the capture range, on the lane's execution thread, behind the jobs
    /// submitted before it. The driver call is not issued on a stream.
    fn range(
        &self,
        jobs: &JobQueues,
        call: impl FnOnce() -> cuda::Result<()> + Send + 'static,
    ) -> Result<Option<JobId>> {
        let failures = self.failures.0.clone();
        let job = Job::new();
        job.on(None).launch_host_fn(Box::new(move || {
            if let Err(error) = call() {
                // The receiver lives as long as the profiler.
                let _ = failures.send(error.into());
            }
        }))?;
        jobs.start(job, true).map(Some)
    }

    /// Note that `phase` failed. A window that did not open is given up,
    /// and a torch profiler started for it is stopped.
    fn failed(&mut self, phase: Phase, error: &Error) {
        match phase {
            Phase::Start | Phase::StartRange => {
                self.active = false;
                self.finished = true;
                self.phases.clear();
                if self.torch.is_some() {
                    self.phases.push_back(Phase::Discard);
                }
                tracing::error!(
                    "failed to start UniServe worker profiler; disabling this capture: {error}"
                );
            }
            // The capture was given up already.
            Phase::Discard => {}
            Phase::StopRange => {
                tracing::error!("failed to stop CUDA profiler: {error}");
            }
            Phase::Export => {
                tracing::error!("failed to stop/export UniServe worker profiler: {error}")
            }
        }
    }

    /// The library call that starts the torch profiler and returns it.
    fn start_call(&self) -> MethodCall {
        MethodCall::new(
            &self.start,
            vec![
                Argument::Texts(self.config.activities.clone()),
                Argument::Bool(self.config.with_stack),
                Argument::Bool(self.config.record_shapes),
            ],
        )
    }

    /// The library call that stops `profile` without a window to export.
    fn discard_call(&self, profile: Handle) -> MethodCall {
        MethodCall::new(
            &self.stop,
            vec![
                Argument::Handle(profile),
                Argument::None,
                Argument::Bool(false),
            ],
        )
    }

    /// The library call that stops `profile` and exports the window.
    fn export_call(&self, profile: Handle) -> MethodCall {
        let stamp = time::OffsetDateTime::now_local()
            .unwrap_or_else(|_| time::OffsetDateTime::now_utc())
            .format(time::macros::format_description!(
                "[year][month][day]-[hour][minute][second]"
            ))
            .unwrap_or_default();
        let rank = std::env::var("RANK")
            .ok()
            .filter(|rank| !rank.is_empty())
            .or_else(|| std::env::var("LOCAL_RANK").ok())
            .map(|rank| format!("-rank{rank}"))
            .unwrap_or_default();
        let last = self.first + self.captured.saturating_sub(1);
        let base = format!(
            "{}-pid{}{rank}-steps{}-{last}-{stamp}",
            self.config.prefix,
            std::process::id(),
            self.first
        );
        let prefix = self.config.output_dir.join(base);
        let sort_device = self.config.activities.iter().any(|name| name == "GPU");
        MethodCall::new(
            &self.stop,
            vec![
                Argument::Handle(profile),
                Argument::Text(prefix.to_string_lossy().into_owned()),
                Argument::Bool(sort_device),
            ],
        )
    }

    /// Export an open window, including a partial one, at shutdown. The
    /// executor has drained and the rank no longer serves, so the calls are
    /// made here and waited for. A job of the window still in flight is not
    /// waited for.
    pub(crate) fn close(&mut self) {
        if self.active {
            self.finish();
        }
        while let Some(phase) = self.phases.pop_front() {
            let closed = match phase {
                // A window that did not open yet is not opened to be closed.
                Phase::Start | Phase::StartRange => Ok(()),
                Phase::StopRange => match self.device.filter(|_| std::mem::take(&mut self.ranging))
                {
                    Some(device) => cuda::profiler_stop(device).map_err(Error::from),
                    None => Ok(()),
                },
                Phase::Discard => self.torch.take().map_or(Ok(()), |profile| {
                    self.discard_call(profile).invoke().map(drop)
                }),
                Phase::Export => self.torch.take().map_or(Ok(()), |profile| {
                    self.export_call(profile).invoke().map(drop)
                }),
            };
            if let Err(error) = closed {
                self.failed(phase, &error);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use std::sync::{Arc, Mutex};

    use tvm_ffi::{Any, AnyView};

    use super::*;

    /// Take the completion of `job`, which a rank without lanes and
    /// without a started interpreter performed where it was submitted.
    fn completed(jobs: &JobQueues, job: JobId) -> Completion {
        let (id, completion) = jobs.take().unwrap();
        assert_eq!(id, job);
        completion
    }

    #[test]
    fn the_window_opens_before_its_first_step_and_exports_after_its_last() {
        let directory = tempfile::tempdir().unwrap();
        let started = Arc::new(Mutex::new(Vec::new()));
        let stopped = Arc::new(Mutex::new(Vec::new()));
        let start = {
            let started = Arc::clone(&started);
            tvm_ffi::Function::from_packed(move |args: &[AnyView]| {
                let activities = crate::ffi::texts(Any::from(args[0]))?;
                started.lock().unwrap().push(activities);
                Ok(Any::from(7_i64))
            })
        };
        let stop = {
            let stopped = Arc::clone(&stopped);
            tvm_ffi::Function::from_packed(move |args: &[AnyView]| {
                let profile = i64::try_from(args[0])?;
                let prefix = tvm_ffi::String::try_from(args[1])?.as_str().to_owned();
                stopped.lock().unwrap().push((profile, prefix));
                Ok(Any::default())
            })
        };
        let config = ProfileConfig {
            output_dir: directory.path().join("profiles"),
            prefix: "window".to_owned(),
            activities: vec!["CPU".to_owned()],
            start_step: 2,
            num_steps: 2,
            with_stack: false,
            record_shapes: false,
            cuda_profiler: false,
        };
        let mut profiler = Profiler::new(
            config,
            Function::new("test.start", start),
            Function::new("test.stop", stop),
            None,
        );
        let jobs = JobQueues::default();

        // The first step lies before the window: its forward waits for
        // nothing.
        profiler.begin();
        assert!(profiler.next(&jobs).is_none());
        profiler.end();
        assert!(profiler.next(&jobs).is_none());

        // The second step opens it: the profiler's start is a job the
        // forward waits for, and nothing more once it completed.
        profiler.begin();
        let job = profiler.next(&jobs).unwrap();
        assert!(profiler.next(&jobs).is_none());
        profiler.complete(completed(&jobs, job));
        assert!(profiler.next(&jobs).is_none());
        assert_eq!(*started.lock().unwrap(), [vec!["CPU".to_owned()]]);
        assert!(directory.path().join("profiles").is_dir());
        profiler.end();
        assert!(profiler.next(&jobs).is_none());

        // The window's last step ends it: the export is a job behind that
        // step's forward, named after the steps the window covered.
        profiler.begin();
        assert!(profiler.next(&jobs).is_none());
        profiler.end();
        let job = profiler.next(&jobs).unwrap();
        profiler.complete(completed(&jobs, job));
        let stopped = stopped.lock().unwrap();
        let [(profile, prefix)] = &stopped[..] else {
            panic!("the window is exported once, not {stopped:?}");
        };
        assert_eq!(*profile, 7);
        assert!(prefix.contains("window-pid"), "{prefix}");
        assert!(prefix.contains("-steps2-3-"), "{prefix}");

        // The window is not opened again.
        profiler.begin();
        assert!(profiler.next(&jobs).is_none());
        assert_eq!(started.lock().unwrap().len(), 1);
    }

    #[test]
    fn the_window_is_configured_from_the_environment() {
        let env = HashMap::from([
            ("UNISERVE_TORCH_PROFILER_DIR", "/tmp/profiles"),
            ("UNISERVE_PROFILE_ACTIVITIES", "gpu, cpu,GPU"),
            ("UNISERVE_PROFILE_START_STEP", "3"),
            ("UNISERVE_PROFILE_STEPS", "zero"),
            ("UNISERVE_CUDA_PROFILER", "yes"),
        ]);
        let lookup = |name: &str| env.get(name).map(|value| (*value).to_owned());
        let config = ProfileConfig::from_env(lookup).unwrap().unwrap();
        assert_eq!(config.activities, vec!["GPU".to_owned(), "CPU".to_owned()]);
        assert_eq!((config.start_step, config.num_steps), (3, 1));
        assert!(config.cuda_profiler && !config.with_stack);
        assert_eq!(config.prefix, "uniserve-worker");

        assert!(ProfileConfig::from_env(|_| None).unwrap().is_none());
        let invalid = HashMap::from([
            ("UNISERVE_TORCH_PROFILER_DIR", "/tmp"),
            ("UNISERVE_PROFILE_ACTIVITIES", "TPU"),
        ]);
        assert!(
            ProfileConfig::from_env(|name| invalid.get(name).map(|value| (*value).to_owned()))
                .is_err()
        );
    }
}
