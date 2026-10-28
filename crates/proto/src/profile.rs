//! The vocabulary of profiling sessions (`POST /admin/profile`) between the
//! leader daemon and the workers of the world that executes.
//!
//! The daemon validates a session's request, gives it an identity and hands
//! it to every worker as the payload of the worker hook
//! `uniserve.profile_start`, JSON-encoded; workers of another build read the
//! fields they know, and every field has a default.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A collector a session runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Collector {
    /// CUPTI's activity records and runtime and driver API calls.
    Timeline,
    /// Every fence probe record of the selected components.
    FenceTimeline,
    /// CUPTI's PM sampling: hardware counters at a fixed interval.
    PmSampling,
    /// CUPTI's range profiler: one-pass hardware counters per kernel or per
    /// component range.
    RangeMetrics,
    /// CUPTI's continuous PC sampling: program counters and stall reasons.
    PcSampling,
}

impl Collector {
    pub const ALL: [Self; 5] = [
        Self::Timeline,
        Self::FenceTimeline,
        Self::PmSampling,
        Self::RangeMetrics,
        Self::PcSampling,
    ];

    /// The collectors that hold a device's performance monitors while they
    /// run. CUPTI gives the monitors to one of them at a time: the second
    /// to start fails (`CUPTI_ERROR_HARDWARE_BUSY`, or an invalid
    /// parameter between the range profiler and PC sampling), so a session
    /// runs at most one.
    pub const MONITORS: [Self; 3] = [Self::PmSampling, Self::RangeMetrics, Self::PcSampling];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Timeline => "timeline",
            Self::FenceTimeline => "fence_timeline",
            Self::PmSampling => "pm_sampling",
            Self::RangeMetrics => "range_metrics",
            Self::PcSampling => "pc_sampling",
        }
    }

    /// The collectors in the order the quality-of-service guard turns them
    /// off, costliest first.
    pub const COSTLIEST_FIRST: [Self; 5] = [
        Self::RangeMetrics,
        Self::PcSampling,
        Self::Timeline,
        Self::PmSampling,
        Self::FenceTimeline,
    ];
}

/// How a timeline records a CUDA graph's launch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphTrace {
    /// One record of the whole launch.
    #[default]
    Whole,
    /// A record of each kernel, copy and fill node.
    Nodes,
}

/// The `timeline` collector's options.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TimelineOptions {
    pub graphs: GraphTrace,
}

/// The `pm_sampling` collector's options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PmSamplingOptions {
    /// Metric names of NVIDIA's performance library that PM sampling
    /// collects in one pass.
    pub metrics: Vec<String>,
    /// The sampling interval, nanoseconds of device time.
    pub interval_ns: u64,
}

impl Default for PmSamplingOptions {
    fn default() -> Self {
        Self {
            metrics: vec![
                "gr__cycles_active.avg".to_owned(),
                "sm__cycles_active.avg".to_owned(),
                "dram__throughput.avg.pct_of_peak_sustained_elapsed".to_owned(),
            ],
            interval_ns: 100_000,
        }
    }
}

/// How the range profiler delimits its ranges.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricRanges {
    /// A range per kernel launch, a CUDA graph's launch being one.
    #[default]
    Kernel,
    /// A range per probed component instance's call, around its entry and
    /// exit, for calls that do not replay a CUDA graph.
    Component,
}

/// The `range_metrics` collector's options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RangeMetricsOptions {
    /// Metric names one pass collects.
    pub metrics: Vec<String>,
    pub ranges: MetricRanges,
}

impl Default for RangeMetricsOptions {
    fn default() -> Self {
        Self {
            metrics: vec!["gpu__time_duration.sum".to_owned()],
            ranges: MetricRanges::Kernel,
        }
    }
}

/// The `pc_sampling` collector's options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PcSamplingOptions {
    /// Each SM samples a warp every `2^period` cycles, `period` from 5 to
    /// 31 (CUPTI's `CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD`).
    pub period: u32,
}

impl Default for PcSamplingOptions {
    fn default() -> Self {
        Self { period: 17 }
    }
}

/// A session's request, as `POST /admin/profile` takes it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SessionRequest {
    pub collectors: Vec<Collector>,
    /// Steps of the selected stages each worker skips before its window
    /// opens.
    pub start_step: u64,
    /// The window's steps of the selected stages; unbounded when unset.
    pub steps: Option<u64>,
    /// The window's length in seconds from the session's start; unbounded
    /// when unset.
    pub seconds: Option<f64>,
    /// The stages recorded, as call kinds (`prefill`, `decode`, ...); every
    /// stage when empty.
    pub stages: Vec<String>,
    /// The workers recorded, by rank; every worker when empty.
    pub workers: Vec<u32>,
    /// The components whose fence records and component ranges are kept,
    /// as patterns of their names (`*` matches any run); every probed
    /// component when empty.
    pub components: Vec<String>,
    /// Where each worker writes its file; the directory is created.
    pub output_dir: PathBuf,
    pub timeline: TimelineOptions,
    pub pm_sampling: PmSamplingOptions,
    pub range_metrics: RangeMetricsOptions,
    pub pc_sampling: PcSamplingOptions,
    /// How much longer than before the session a step may take on
    /// average, as a fraction (`0.1` for 10%), before the guard turns off
    /// the costliest collector, and then ends the session; unguarded when
    /// unset.
    pub max_slowdown: Option<f64>,
}

impl SessionRequest {
    /// The request's own faults: no collector, a collector named twice, a
    /// window of no step or no time, a negative slowdown, no output
    /// directory, an option table of a collector not run.
    ///
    /// # Errors
    ///
    /// Names the fault.
    pub fn validate(&self) -> Result<(), String> {
        if self.collectors.is_empty() {
            return Err("a session runs at least one collector".to_owned());
        }
        let mut seen = self.collectors.clone();
        seen.sort_unstable();
        seen.dedup();
        if seen.len() != self.collectors.len() {
            return Err("a collector is named twice".to_owned());
        }
        let monitors: Vec<&str> = self
            .collectors
            .iter()
            .filter(|collector| Collector::MONITORS.contains(collector))
            .map(|collector| collector.as_str())
            .collect();
        if monitors.len() > 1 {
            return Err(format!(
                "{} each hold the device's performance monitors, which CUPTI \
                 gives to one profiler at a time: a session runs at most one of them",
                monitors.join(" and ")
            ));
        }
        if self.steps == Some(0) {
            return Err("a window of steps holds at least one".to_owned());
        }
        if let Some(seconds) = self.seconds
            && !(seconds.is_finite() && seconds > 0.0)
        {
            return Err("a window of seconds is a positive number".to_owned());
        }
        if let Some(slowdown) = self.max_slowdown
            && !(slowdown.is_finite() && slowdown >= 0.0)
        {
            return Err("max_slowdown is a nonnegative fraction".to_owned());
        }
        if self.output_dir.as_os_str().is_empty() {
            return Err("output_dir names the directory the workers write to".to_owned());
        }
        let runs = |collector| self.collectors.contains(&collector);
        let unused = [
            (
                Collector::Timeline,
                self.timeline != TimelineOptions::default(),
            ),
            (
                Collector::PmSampling,
                self.pm_sampling != PmSamplingOptions::default(),
            ),
            (
                Collector::RangeMetrics,
                self.range_metrics != RangeMetricsOptions::default(),
            ),
            (
                Collector::PcSampling,
                self.pc_sampling != PcSamplingOptions::default(),
            ),
        ];
        if let Some((collector, _)) = unused
            .iter()
            .find(|&&(collector, stated)| stated && !runs(collector))
        {
            return Err(format!(
                "options of {} are stated, but the session does not run it",
                collector.as_str()
            ));
        }
        if self.pm_sampling.metrics.is_empty() || self.range_metrics.metrics.is_empty() {
            return Err("a metric collector names at least one metric".to_owned());
        }
        if !(5..=31).contains(&self.pc_sampling.period) {
            return Err("pc_sampling.period is from 5 to 31".to_owned());
        }
        if self.pm_sampling.interval_ns == 0 {
            return Err("pm_sampling.interval_ns is positive".to_owned());
        }
        Ok(())
    }
}

/// What the daemon hands a worker to start a session.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SessionStart {
    /// The session's identity.
    pub session: u64,
    /// The world the session belongs to.
    pub world: u64,
    pub request: SessionRequest,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> SessionRequest {
        SessionRequest {
            collectors: vec![Collector::Timeline],
            output_dir: "/tmp/profiles".into(),
            ..SessionRequest::default()
        }
    }

    #[test]
    fn a_request_reads_from_its_json_with_defaults() {
        let parsed: SessionRequest = serde_json::from_str(
            r#"{"collectors": ["timeline", "fence_timeline", "pm_sampling"], "steps": 8,
                "stages": ["decode"], "output_dir": "/tmp/p",
                "pm_sampling": {"interval_ns": 50000}}"#,
        )
        .unwrap();
        assert_eq!(
            parsed.collectors,
            [
                Collector::Timeline,
                Collector::FenceTimeline,
                Collector::PmSampling
            ]
        );
        assert_eq!(parsed.pm_sampling.interval_ns, 50_000);
        assert_eq!(
            parsed.pm_sampling.metrics,
            PmSamplingOptions::default().metrics
        );
        assert_eq!(parsed.timeline.graphs, GraphTrace::Whole);
        assert!(parsed.validate().is_ok());
        assert!(serde_json::from_str::<SessionRequest>(r#"{"collector": []}"#).is_err());
    }

    #[test]
    fn a_request_names_its_own_faults() {
        assert!(request().validate().is_ok());
        let faults = [
            SessionRequest {
                collectors: Vec::new(),
                ..request()
            },
            SessionRequest {
                collectors: vec![Collector::Timeline, Collector::Timeline],
                ..request()
            },
            SessionRequest {
                steps: Some(0),
                ..request()
            },
            SessionRequest {
                seconds: Some(-1.0),
                ..request()
            },
            SessionRequest {
                output_dir: PathBuf::new(),
                ..request()
            },
            SessionRequest {
                pc_sampling: PcSamplingOptions { period: 20 },
                ..request()
            },
            SessionRequest {
                collectors: vec![
                    Collector::Timeline,
                    Collector::PmSampling,
                    Collector::PcSampling,
                ],
                ..request()
            },
            SessionRequest {
                collectors: vec![Collector::RangeMetrics, Collector::PcSampling],
                ..request()
            },
            SessionRequest {
                collectors: vec![Collector::PcSampling],
                pc_sampling: PcSamplingOptions { period: 4 },
                ..request()
            },
        ];
        for fault in faults {
            assert!(fault.validate().is_err(), "{fault:?}");
        }
    }
}
