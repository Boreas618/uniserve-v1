//! The vocabulary of profiling sessions (`profiling_config.session`) between the
//! leader daemon and the workers of the epoch that executes.
//!
//! The daemon validates a session's request, gives it an identity and starts
//! it on every worker with the engine command `StartProfile`, which carries
//! the [`SessionStart`] as the request interface's `ProfileSession`; each rank
//! answers the start, the end and a collector's release with its report
//! ([`RankStart`], [`RankStop`], [`RankRelease`]). The conversions between
//! these values and the request interface's messages are in `wire`.

mod wire;

use std::collections::BTreeMap;
use std::ops::RangeInclusive;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A collector a session runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Collector {
    /// CUPTI's activity records and runtime and driver API calls.
    Timeline,
    /// Every fence probe record of the selected module classes.
    FenceTimeline,
    /// CUPTI's PM sampling: hardware counters at a fixed interval.
    PmSampling,
    /// CUPTI's range profiler: one-pass hardware counters per kernel or per
    /// probed module range.
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
    /// Metric names of the collector's performance library that PM sampling collects in one
    /// pass; the collector's own default metrics when empty.
    pub metrics: Vec<String>,
    /// The sampling interval, nanoseconds of device time. A worker refuses an interval whose
    /// samples its reads cannot decode as fast as the device takes them, which it measures when
    /// the session starts.
    pub interval_ns: u64,
}

impl Default for PmSamplingOptions {
    fn default() -> Self {
        Self {
            metrics: Vec::new(),
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
    /// A range per probed module instance's call, around its entry and
    /// exit, for calls that do not replay a CUDA graph.
    Module,
}

/// The kernel and graph launches of one step a `range_metrics` pass makes
/// room for: a pass of `pass_ranges` ranges is read every
/// `pass_ranges / RANGE_STEP_LAUNCHES` recorded steps.
pub const RANGE_STEP_LAUNCHES: usize = 32;

/// The `range_metrics` collector's options.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RangeMetricsOptions {
    /// Metric names one pass collects; the collector's own default metrics when empty.
    pub metrics: Vec<String>,
    pub ranges: MetricRanges,
    /// The ranges one pass holds, at least [`RANGE_STEP_LAUNCHES`]: its
    /// device storage grows with them, and the worker reads it every
    /// `pass_ranges / RANGE_STEP_LAUNCHES` recorded steps. CUPTI states no
    /// size for a pass's device storage, so each worker measures what the
    /// pass's first start took when the session starts, and reserves that.
    pub pass_ranges: usize,
}

impl Default for RangeMetricsOptions {
    fn default() -> Self {
        Self {
            metrics: Vec::new(),
            ranges: MetricRanges::Kernel,
            pass_ranges: 512,
        }
    }
}

impl RangeMetricsOptions {
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
    /// (`CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD`), one of
    /// [`PcSamplingOptions::PERIODS`].
    pub period: u32,
    /// The hardware buffer CUPTI keeps the samples in, in MiB of process
    /// host storage, which CUPTI's own thread empties
    /// (`CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_HARDWARE_BUFFER_SIZE`), one
    /// of [`PcSamplingOptions::HARDWARE_BUFFERS_MIB`]. A buffer that fills
    /// between CUPTI's reads loses samples, and the worker then releases the
    /// collector as failed, naming the period and the buffer: a device whose
    /// samples outrun the buffer at the period samples at a longer period or
    /// into a larger buffer.
    pub hardware_buffer_mib: u32,
}

impl PcSamplingOptions {
    /// The sampling periods CUPTI accepts: 5 to 31, as its PC sampling
    /// API states for `CUPTI_PC_SAMPLING_CONFIGURATION_ATTR_TYPE_SAMPLING_PERIOD`.
    pub const PERIODS: RangeInclusive<u32> = 5..=31;

    /// The hardware buffers a session asks for, in MiB: up to CUPTI's
    /// default, which is its largest.
    pub const HARDWARE_BUFFERS_MIB: RangeInclusive<u32> = 1..=DEFAULT_PC_HARDWARE_BUFFER_MIB;

    /// The hardware buffer in bytes.
    pub fn hardware_buffer(&self) -> usize {
        (self.hardware_buffer_mib as usize) << 20
    }
}

/// The PC sampling hardware buffer a session takes when it states none, in MiB: CUPTI's own
/// default and largest buffer, which holds the samples of the shortest periods longest.
pub const DEFAULT_PC_HARDWARE_BUFFER_MIB: u32 = 512;

impl Default for PcSamplingOptions {
    fn default() -> Self {
        Self {
            period: 17,
            hardware_buffer_mib: DEFAULT_PC_HARDWARE_BUFFER_MIB,
        }
    }
}

/// How often a worker reads its armed samplers when a session states none, in milliseconds: often
/// enough that a PM sampler's buffers, which hold two read periods of samples, stay a few MiB at
/// the default interval, and seldom enough that the reads cost the rank's host little.
pub const DEFAULT_SAMPLER_READ_MS: u64 = 100;

/// A session's request, as `profiling_config.session` states it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
    /// The module classes whose fence records and module ranges are kept,
    /// as patterns of their names (`*` matches any run); every probed
    /// class when empty.
    pub modules: Vec<String>,
    /// Where each worker writes its file; the directory is created.
    pub output_dir: PathBuf,
    pub timeline: TimelineOptions,
    pub pm_sampling: PmSamplingOptions,
    pub range_metrics: RangeMetricsOptions,
    pub pc_sampling: PcSamplingOptions,
    /// How often each worker reads its armed samplers (`pm_sampling` and
    /// `pc_sampling`), in milliseconds: a PM sampler's counter data image and
    /// hardware buffer hold the samples of two read periods, and a PC take
    /// reads what CUPTI gathered since the last one. Neither interface states
    /// an event for samples gathered, so a worker reads them at this period.
    pub sampler_read_ms: u64,
    /// How much longer than before the session a step may take on
    /// average, as a fraction (`0.1` for 10%), before the guard turns off
    /// the costliest collector, and then ends the session; unguarded when
    /// unset.
    pub max_slowdown: Option<f64>,
    /// The span before the session, in seconds, whose average step time the guard compares the
    /// session's with.
    pub baseline_s: f64,
    /// The span of the session, in seconds, the guard averages its step time over, and the
    /// least time between two of its actions, so that each is measured.
    pub guard_window_s: f64,
}

/// The default `baseline_s` of a session: a minute of the serving epoch's steps, which holds
/// every scheduler's steps of the batch sizes its load visits.
pub const BASELINE_S: f64 = 60.0;

/// The default `guard_window_s` of a session: ten seconds of steps, long enough that one
/// collector's change of the step time stands out of the steps' second-to-second variation.
pub const GUARD_WINDOW_S: f64 = 10.0;

impl Default for SessionRequest {
    fn default() -> Self {
        Self {
            collectors: Vec::new(),
            start_step: 0,
            steps: None,
            seconds: None,
            stages: Vec::new(),
            workers: Vec::new(),
            modules: Vec::new(),
            output_dir: PathBuf::new(),
            timeline: TimelineOptions::default(),
            pm_sampling: PmSamplingOptions::default(),
            range_metrics: RangeMetricsOptions::default(),
            pc_sampling: PcSamplingOptions::default(),
            sampler_read_ms: DEFAULT_SAMPLER_READ_MS,
            max_slowdown: None,
            baseline_s: BASELINE_S,
            guard_window_s: GUARD_WINDOW_S,
        }
    }
}

impl SessionRequest {
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
        for (name, seconds) in [
            ("baseline_s", self.baseline_s),
            ("guard_window_s", self.guard_window_s),
        ] {
            if !(seconds.is_finite() && seconds > 0.0) {
                return Err(format!("{name} is a positive number of seconds"));
            }
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
        if !PcSamplingOptions::PERIODS.contains(&self.pc_sampling.period) {
            return Err(format!(
                "pc_sampling.period is from {} to {}, the periods CUPTI accepts",
                PcSamplingOptions::PERIODS.start(),
                PcSamplingOptions::PERIODS.end()
            ));
        }
        if !PcSamplingOptions::HARDWARE_BUFFERS_MIB.contains(&self.pc_sampling.hardware_buffer_mib)
        {
            return Err(format!(
                "pc_sampling.hardware_buffer_mib is from {} to {}, CUPTI's largest buffer",
                PcSamplingOptions::HARDWARE_BUFFERS_MIB.start(),
                PcSamplingOptions::HARDWARE_BUFFERS_MIB.end()
            ));
        }
        if self.sampler_read_ms == 0 {
            return Err("sampler_read_ms is a positive number of milliseconds".to_owned());
        }
        if self.range_metrics.pass_ranges < RANGE_STEP_LAUNCHES
            || u32::try_from(self.range_metrics.pass_ranges).is_err()
        {
            return Err(format!(
                "range_metrics.pass_ranges is from {RANGE_STEP_LAUNCHES} to {}",
                u32::MAX
            ));
        }
        if self.pm_sampling.interval_ns == 0 {
            return Err("pm_sampling.interval_ns is a positive number of nanoseconds".to_owned());
        }
        Ok(())
    }
}

/// What the daemon hands a worker to start a session.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SessionStart {
    /// The session's identity.
    pub session: u64,
    /// The epoch the session belongs to.
    pub epoch: u64,
    pub request: SessionRequest,
}

/// One rank's answer to a session's start.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankStart {
    pub rank: u32,
    /// Whether the session records the rank.
    pub selected: bool,
    /// Why the rank refused the session: its GPU's quota cannot fit the device storage the
    /// session's collectors hold. `None` when it did not.
    pub refused: Option<String>,
    /// The device storage, in bytes, the rank's collectors measured they hold on its GPU while
    /// the session runs (a range_metrics pass); 0 on a rank the session does not record.
    pub device_storage: u64,
    /// The rank's file; `None` when the session does not record the rank.
    pub path: Option<PathBuf>,
    /// Each collector's state on the rank (`prepared`, or why it cannot run), by the collector's
    /// name.
    pub collectors: BTreeMap<String, String>,
}

impl RankStart {
    /// The answer of rank `rank`, which the session does not record.
    pub fn unselected(rank: u32) -> Self {
        Self {
            rank,
            selected: false,
            refused: None,
            device_storage: 0,
            path: None,
            collectors: BTreeMap::new(),
        }
    }
}

/// One rank's answer to a session's end.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankStop {
    pub rank: u32,
    pub session: u64,
    /// Whether the session recorded the rank; a rank of no session or of another answers `false`
    /// and nothing else.
    pub selected: bool,
    /// The steps the rank's window recorded.
    pub steps: u64,
    /// Why the session ended.
    pub reason: String,
    /// What the rank's file holds; `None` for a rank that recorded no file.
    pub file: Option<FileReport>,
}

impl RankStop {
    /// The answer of rank `rank`, which session `session` does not record.
    pub fn unselected(rank: u32, session: u64) -> Self {
        Self {
            rank,
            session,
            selected: false,
            steps: 0,
            reason: String::new(),
            file: None,
        }
    }
}

/// A rank's profile file as its session's end left it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileReport {
    /// The file is whole: where it lies, its bytes and its rows by table.
    Written {
        path: PathBuf,
        bytes: u64,
        rows: BTreeMap<String, u64>,
    },
    /// The file could not be finished, for this reason.
    Failed { error: String },
}

/// Whether one rank ran the collector a release turned off.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct RankRelease {
    pub rank: u32,
    pub released: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> SessionRequest {
        SessionRequest {
            collectors: vec![Collector::Timeline],
            output_dir: "profiles".into(),
            ..SessionRequest::default()
        }
    }

    #[test]
    fn a_request_reads_from_its_json_with_defaults() {
        let parsed: SessionRequest = serde_json::from_str(
            r#"{"collectors": ["timeline", "fence_timeline", "pm_sampling"], "steps": 8,
                "stages": ["decode"], "output_dir": "profiles",
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
                pc_sampling: PcSamplingOptions {
                    period: 20,
                    ..PcSamplingOptions::default()
                },
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
                pc_sampling: PcSamplingOptions {
                    period: 4,
                    ..PcSamplingOptions::default()
                },
                ..request()
            },
            SessionRequest {
                collectors: vec![Collector::PcSampling],
                pc_sampling: PcSamplingOptions {
                    period: 32,
                    ..PcSamplingOptions::default()
                },
                ..request()
            },
            SessionRequest {
                collectors: vec![Collector::PcSampling],
                pc_sampling: PcSamplingOptions {
                    hardware_buffer_mib: 0,
                    ..PcSamplingOptions::default()
                },
                ..request()
            },
            SessionRequest {
                collectors: vec![Collector::PcSampling],
                pc_sampling: PcSamplingOptions {
                    hardware_buffer_mib: 513,
                    ..PcSamplingOptions::default()
                },
                ..request()
            },
            SessionRequest {
                sampler_read_ms: 0,
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
                    interval_ns: 0,
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
