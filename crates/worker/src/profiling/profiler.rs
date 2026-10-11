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

use std::path::PathBuf;

use tvm_ffi::Any;
use uniserve_cuda::DeviceId;

use crate::ffi::{Function, Handle, strings};
use crate::{Error, Result};

use super::flag;

/// Window settings read from the environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileConfig {
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
    pub fn from_env(value: impl Fn(&str) -> Option<String>) -> Result<Option<Self>> {
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

/// The step counter and the open window.
pub struct Profiler {
    config: ProfileConfig,
    start: Function,
    stop: Function,
    /// The CUDA device a capture range applies to.
    device: Option<DeviceId>,
    seen: usize,
    captured: usize,
    first: usize,
    active: bool,
    finished: bool,
    torch: Option<Handle>,
}

impl Profiler {
    pub fn new(
        config: ProfileConfig,
        start: Function,
        stop: Function,
        device: Option<i32>,
    ) -> Self {
        Self {
            config,
            start,
            stop,
            device: device.map(DeviceId),
            seen: 0,
            captured: 0,
            first: 0,
            active: false,
            finished: false,
            torch: None,
        }
    }

    /// Account one executed batch around `execute`.
    pub fn step<T>(&mut self, execute: impl FnOnce() -> T) -> T {
        self.begin();
        let result = execute();
        self.end();
        result
    }

    fn begin(&mut self) {
        self.seen += 1;
        if self.active || self.finished || self.seen < self.config.start_step {
            return;
        }
        self.first = self.seen;
        let started = (|| -> Result<()> {
            std::fs::create_dir_all(&self.config.output_dir).map_err(|error| {
                Error::Transport(format!(
                    "creating {}: {error}",
                    self.config.output_dir.display()
                ))
            })?;
            let profile = Handle::new(self.start.call_values(&[
                Any::from(strings(&self.config.activities)),
                Any::from(self.config.with_stack),
                Any::from(self.config.record_shapes),
            ])?)?;
            if !profile.is_none() {
                self.torch = Some(profile);
            }
            if self.config.cuda_profiler
                && let Some(device) = self.device
            {
                uniserve_cuda::profiler_start(device)?;
            }
            Ok(())
        })();
        match started {
            Ok(()) => self.active = true,
            Err(error) => {
                // A CUDA-profiler failure after torch started still retires
                // the open torch session; later batches never retry.
                if let Some(profile) = self.torch.take() {
                    let _ = self.stop.call_values(&[
                        profile.to_any(),
                        Any::default(),
                        Any::from(false),
                    ]);
                }
                self.finished = true;
                tracing::error!(
                    target: "uniserve_worker.profiling",
                    "failed to start UniServe worker profiler; disabling this capture: {error}"
                );
            }
        }
    }

    fn end(&mut self) {
        if self.active {
            self.captured += 1;
            if self.captured >= self.config.num_steps {
                self.finish();
            }
        }
    }

    /// Export an open window, including a partial one at shutdown.
    pub fn close(&mut self) {
        if self.active {
            self.finish();
        }
    }

    fn finish(&mut self) {
        self.active = false;
        self.finished = true;
        if self.config.cuda_profiler
            && let Some(device) = self.device
            && let Err(error) = uniserve_cuda::profiler_stop(device)
        {
            tracing::error!(target: "uniserve_worker.profiling", "failed to stop CUDA profiler: {error}");
        }
        let Some(profile) = self.torch.take() else {
            return;
        };
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
        if let Err(error) = self.stop.call_values(&[
            profile.to_any(),
            Any::from(tvm_ffi::String::from(prefix.to_string_lossy().as_ref())),
            Any::from(sort_device),
        ]) {
            tracing::error!(
                target: "uniserve_worker.profiling",
                "failed to stop/export UniServe worker profiler: {error}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

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
