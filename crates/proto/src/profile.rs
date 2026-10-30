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

/// The shortest `pm_sampling` interval, nanoseconds: on GB200 a worker's
/// read decodes and evaluates a sample of the default metrics in about 14
/// µs, so samples every 20 µs keep one thread about 70% busy and every 10
/// µs outrun it (`docs/profiling.md`).
pub const PM_MIN_INTERVAL_NS: u64 = 20_000;

/// The kernel and graph launches of one step a `range_metrics` pass makes
/// room for: a pass of `pass_ranges` ranges is read every
/// `pass_ranges / RANGE_STEP_LAUNCHES` recorded steps.
pub const RANGE_STEP_LAUNCHES: usize = 32;

/// The device storage a range profiler pass takes per range. CUPTI takes the
/// pass's ranges' storage at its first start, rounded up to whole
/// [`RANGE_STORAGE_GRANULE`]s, and keeps it until the profiler is disabled.
/// Measured on GB200 with `nvidia-cuda-cupti` 13.4.92 as the device's free
/// storage (`cuMemGetInfo`) a pass's first start took: 727,711,744,
/// 1,746,927,616, 2,908,749,824 and 5,817,499,648 bytes for passes of 512,
/// 1,229, 2,048 and 4,096 ranges, which this size per range gives for
/// each; the same for one, two and four metrics.
const RANGE_STORAGE: u64 = 1_420_288;

/// The unit CUPTI takes a pass's device storage in: 2 MiB.
const RANGE_STORAGE_GRANULE: u64 = 2 << 20;

/// The `range_metrics` collector's options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RangeMetricsOptions {
    /// Metric names one pass collects.
    pub metrics: Vec<String>,
    pub ranges: MetricRanges,
    /// The ranges one pass holds, at least [`RANGE_STEP_LAUNCHES`]: its
    /// device storage grows with them ([`Self::pass_storage`]), and the
    /// worker reads it every `pass_ranges / RANGE_STEP_LAUNCHES` recorded
    /// steps. The default, 512, takes 728 MB.
    pub pass_ranges: usize,
}

impl Default for RangeMetricsOptions {
    fn default() -> Self {
        Self {
            metrics: vec!["gpu__time_duration.sum".to_owned()],
            ranges: MetricRanges::Kernel,
            pass_ranges: 512,
        }
    }
}

impl RangeMetricsOptions {
    /// The device storage one pass takes: 1,420,288 bytes per range, the
    /// pass's total rounded up to whole 2 MiB.
    pub fn pass_storage(&self) -> u64 {
        (self.pass_ranges as u64 * RANGE_STORAGE).next_multiple_of(RANGE_STORAGE_GRANULE)
    }

    /// Every how many recorded steps the worker reads a pass.
    pub fn read_steps(&self) -> u64 {
        (self.pass_ranges / RANGE_STEP_LAUNCHES).max(1) as u64
    }
}

/// The `pc_sampling` collector's options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PcSamplingOptions {
    /// Each SM samples a warp every `2^period` cycles
    /// (`CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD`), from
    /// [`PcSamplingOptions::MIN_PERIOD`] to 31.
    pub period: u32,
}

impl PcSamplingOptions {
    /// The shortest period a session samples at: below it, no hardware
    /// buffer of up to CUPTI's 512 MiB holds the samples of a busy GB200
    /// between CUPTI's reads of it, and the samples are lost
    /// (`docs/profiling.md`).
    pub const MIN_PERIOD: u32 = 12;
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
    /// The device storage the session holds on the GPU of each worker it
    /// selects while it runs: a range profiler pass for `range_metrics`
    /// ([`RangeMetricsOptions::pass_storage`]); 0 otherwise. The worker
    /// books it in the GPU's storage ledger beside its process.
    pub fn device_storage(&self) -> u64 {
        if self.collectors.contains(&Collector::RangeMetrics) {
            self.range_metrics.pass_storage()
        } else {
            0
        }
    }

    /// Why a deployment whose ranks keep PC sampling enabled
    /// (`profiling_config.graph_pc_sampling`) refuses this request, if it
    /// does: CUPTI then refuses PM sampling (the device's performance
    /// monitors are held), the range profiler and the whole-graph activity
    /// record a timeline that traces graphs whole enables.
    pub fn refused_beside_graph_pc_sampling(&self) -> Option<String> {
        let mut refused: Vec<&str> = self
            .collectors
            .iter()
            .filter(|collector| {
                matches!(collector, Collector::PmSampling | Collector::RangeMetrics)
            })
            .map(|collector| collector.as_str())
            .collect();
        if self.collectors.contains(&Collector::Timeline)
            && self.timeline.graphs == GraphTrace::Whole
        {
            refused.push("a timeline that traces graphs whole (use \"graphs\": \"nodes\")");
        }
        (!refused.is_empty()).then(|| {
            format!(
                "{} cannot run while profiling_config.graph_pc_sampling keeps CUPTI's PC sampling \
                 enabled on every rank",
                refused.join(" and ")
            )
        })
    }

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
        if !(PcSamplingOptions::MIN_PERIOD..=31).contains(&self.pc_sampling.period) {
            return Err(format!(
                "pc_sampling.period is from {} to 31: below {}, no hardware buffer of up to \
                 512 MiB holds the samples of a busy device",
                PcSamplingOptions::MIN_PERIOD,
                PcSamplingOptions::MIN_PERIOD
            ));
        }
        if self.range_metrics.pass_ranges < RANGE_STEP_LAUNCHES
            || u32::try_from(self.range_metrics.pass_ranges).is_err()
        {
            return Err(format!(
                "range_metrics.pass_ranges is from {RANGE_STEP_LAUNCHES} to {}",
                u32::MAX
            ));
        }
        if self.pm_sampling.interval_ns < PM_MIN_INTERVAL_NS {
            return Err(format!(
                "pm_sampling.interval_ns is at least {PM_MIN_INTERVAL_NS}: a worker decodes \
                 about 70,000 samples a second on one thread, and shorter intervals lose \
                 samples"
            ));
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
    fn graph_pc_sampling_refuses_the_collectors_cupti_cannot_run_beside_it() {
        let session = |collectors: Vec<Collector>, graphs: GraphTrace| SessionRequest {
            collectors,
            timeline: TimelineOptions { graphs },
            ..request()
        };
        for refused in [
            session(vec![Collector::PmSampling], GraphTrace::Whole),
            session(vec![Collector::RangeMetrics], GraphTrace::Whole),
            session(vec![Collector::Timeline], GraphTrace::Whole),
            session(
                vec![Collector::Timeline, Collector::PcSampling],
                GraphTrace::Whole,
            ),
        ] {
            let reason = refused.refused_beside_graph_pc_sampling();
            assert!(
                reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains("profiling_config.graph_pc_sampling")),
                "{refused:?}: {reason:?}"
            );
        }
        for accepted in [
            session(
                vec![
                    Collector::Timeline,
                    Collector::FenceTimeline,
                    Collector::PcSampling,
                ],
                GraphTrace::Nodes,
            ),
            session(vec![Collector::PcSampling], GraphTrace::Whole),
        ] {
            assert_eq!(
                accepted.refused_beside_graph_pc_sampling(),
                None,
                "{accepted:?}"
            );
        }
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
                pc_sampling: PcSamplingOptions { period: 11 },
                ..request()
            },
            SessionRequest {
                collectors: vec![Collector::RangeMetrics],
                range_metrics: RangeMetricsOptions {
                    pass_ranges: 16,
                    ..RangeMetricsOptions::default()
                },
                ..request()
            },
            SessionRequest {
                collectors: vec![Collector::PmSampling],
                pm_sampling: PmSamplingOptions {
                    interval_ns: 10_000,
                    ..PmSamplingOptions::default()
                },
                ..request()
            },
        ];
        for fault in faults {
            assert!(fault.validate().is_err(), "{fault:?}");
        }
    }
}
