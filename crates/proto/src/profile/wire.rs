//! The conversions between the profiling vocabulary and the request interface's messages: a
//! session's start (`ProfileSession`), a collector named in a command (`ProfileCollector`), and
//! each rank's reports (`ProfileStart`, `ProfileStop`, `ProfileRelease`).

use std::path::PathBuf;

use super::{
    Collector, FileReport, GraphTrace, MetricRanges, PcSamplingOptions, PmSamplingOptions,
    RangeMetricsOptions, RankRelease, RankStart, RankStop, SessionRequest, SessionStart,
    TimelineOptions,
};
use crate::request::wire;

impl From<Collector> for wire::ProfileCollector {
    fn from(collector: Collector) -> Self {
        match collector {
            Collector::Timeline => Self::Timeline,
            Collector::FenceTimeline => Self::FenceTimeline,
            Collector::PmSampling => Self::PmSampling,
            Collector::RangeMetrics => Self::RangeMetrics,
            Collector::PcSampling => Self::PcSampling,
        }
    }
}

impl Collector {
    /// The collector a command's `ProfileCollector` value names.
    ///
    /// # Errors
    ///
    /// A value that names no collector.
    pub fn from_wire(value: i32) -> Result<Self, String> {
        match wire::ProfileCollector::try_from(value) {
            Ok(wire::ProfileCollector::Timeline) => Ok(Self::Timeline),
            Ok(wire::ProfileCollector::FenceTimeline) => Ok(Self::FenceTimeline),
            Ok(wire::ProfileCollector::PmSampling) => Ok(Self::PmSampling),
            Ok(wire::ProfileCollector::RangeMetrics) => Ok(Self::RangeMetrics),
            Ok(wire::ProfileCollector::PcSampling) => Ok(Self::PcSampling),
            Ok(wire::ProfileCollector::Unspecified) | Err(_) => {
                Err(format!("{value} names no profiling collector"))
            }
        }
    }
}

impl From<&SessionStart> for wire::ProfileSession {
    fn from(start: &SessionStart) -> Self {
        let request = &start.request;
        Self {
            session: start.session,
            epoch: start.epoch,
            request: Some(wire::ProfileRequest {
                collectors: request
                    .collectors
                    .iter()
                    .map(|collector| wire::ProfileCollector::from(*collector) as i32)
                    .collect(),
                start_step: request.start_step,
                steps: request.steps,
                seconds: request.seconds,
                stages: request.stages.clone(),
                workers: request.workers.clone(),
                modules: request.modules.clone(),
                output_dir: request.output_dir.to_string_lossy().into_owned(),
                timeline_graphs: match request.timeline.graphs {
                    GraphTrace::Whole => wire::ProfileGraphTrace::Whole,
                    GraphTrace::Nodes => wire::ProfileGraphTrace::Nodes,
                } as i32,
                pm_metrics: request.pm_sampling.metrics.clone(),
                pm_interval_ns: request.pm_sampling.interval_ns,
                range_metrics: request.range_metrics.metrics.clone(),
                metric_ranges: match request.range_metrics.ranges {
                    MetricRanges::Kernel => wire::ProfileMetricRanges::Kernel,
                    MetricRanges::Module => wire::ProfileMetricRanges::Module,
                } as i32,
                pass_ranges: request.range_metrics.pass_ranges as u64,
                pc_period: request.pc_sampling.period,
                max_slowdown: request.max_slowdown,
            }),
        }
    }
}

impl TryFrom<&wire::ProfileSession> for SessionStart {
    type Error = String;

    /// The session a command starts.
    ///
    /// # Errors
    ///
    /// A session that states no request, or a value that names no collector, graph trace or
    /// range kind.
    fn try_from(session: &wire::ProfileSession) -> Result<Self, Self::Error> {
        let request = session
            .request
            .as_ref()
            .ok_or("a profiling session states no request")?;
        let collectors = request
            .collectors
            .iter()
            .map(|collector| Collector::from_wire(*collector))
            .collect::<Result<_, _>>()?;
        let graphs = match wire::ProfileGraphTrace::try_from(request.timeline_graphs) {
            Ok(wire::ProfileGraphTrace::Whole) => GraphTrace::Whole,
            Ok(wire::ProfileGraphTrace::Nodes) => GraphTrace::Nodes,
            Err(_) => {
                return Err(format!("{} names no graph trace", request.timeline_graphs));
            }
        };
        let ranges = match wire::ProfileMetricRanges::try_from(request.metric_ranges) {
            Ok(wire::ProfileMetricRanges::Kernel) => MetricRanges::Kernel,
            Ok(wire::ProfileMetricRanges::Module) => MetricRanges::Module,
            Err(_) => return Err(format!("{} names no range kind", request.metric_ranges)),
        };
        Ok(Self {
            session: session.session,
            epoch: session.epoch,
            request: SessionRequest {
                collectors,
                start_step: request.start_step,
                steps: request.steps,
                seconds: request.seconds,
                stages: request.stages.clone(),
                workers: request.workers.clone(),
                modules: request.modules.clone(),
                output_dir: PathBuf::from(&request.output_dir),
                timeline: TimelineOptions { graphs },
                pm_sampling: PmSamplingOptions {
                    metrics: request.pm_metrics.clone(),
                    interval_ns: request.pm_interval_ns,
                },
                range_metrics: RangeMetricsOptions {
                    metrics: request.range_metrics.clone(),
                    ranges,
                    pass_ranges: usize::try_from(request.pass_ranges)
                        .map_err(|_| format!("{} ranges is no pass", request.pass_ranges))?,
                },
                pc_sampling: PcSamplingOptions {
                    period: request.pc_period,
                },
                max_slowdown: request.max_slowdown,
            },
        })
    }
}

impl From<&RankStart> for wire::ProfileStart {
    fn from(start: &RankStart) -> Self {
        Self {
            rank: start.rank,
            selected: start.selected,
            refused: start.refused.clone().unwrap_or_default(),
            device_storage: start.device_storage,
            path: start
                .path
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned())
                .unwrap_or_default(),
            collectors: start.collectors.clone().into_iter().collect(),
        }
    }
}

impl From<&wire::ProfileStart> for RankStart {
    fn from(start: &wire::ProfileStart) -> Self {
        let text = |text: &str| (!text.is_empty()).then(|| text.to_owned());
        Self {
            rank: start.rank,
            selected: start.selected,
            refused: text(&start.refused),
            device_storage: start.device_storage,
            path: text(&start.path).map(PathBuf::from),
            collectors: start.collectors.clone().into_iter().collect(),
        }
    }
}

impl From<&RankStop> for wire::ProfileStop {
    fn from(stop: &RankStop) -> Self {
        Self {
            rank: stop.rank,
            session: stop.session,
            selected: stop.selected,
            steps: stop.steps,
            reason: stop.reason.clone(),
            file: stop.file.as_ref().map(|file| match file {
                FileReport::Written { path, bytes, rows } => {
                    wire::profile_stop::File::Written(wire::ProfileFile {
                        path: path.to_string_lossy().into_owned(),
                        bytes: *bytes,
                        rows: rows.clone().into_iter().collect(),
                    })
                }
                FileReport::Failed { error } => wire::profile_stop::File::Failed(error.clone()),
            }),
        }
    }
}

impl From<&wire::ProfileStop> for RankStop {
    fn from(stop: &wire::ProfileStop) -> Self {
        Self {
            rank: stop.rank,
            session: stop.session,
            selected: stop.selected,
            steps: stop.steps,
            reason: stop.reason.clone(),
            file: stop.file.as_ref().map(|file| match file {
                wire::profile_stop::File::Written(file) => FileReport::Written {
                    path: PathBuf::from(&file.path),
                    bytes: file.bytes,
                    rows: file.rows.clone().into_iter().collect(),
                },
                wire::profile_stop::File::Failed(error) => FileReport::Failed {
                    error: error.clone(),
                },
            }),
        }
    }
}

impl From<&RankRelease> for wire::ProfileRelease {
    fn from(release: &RankRelease) -> Self {
        Self {
            rank: release.rank,
            released: release.released,
        }
    }
}

impl From<&wire::ProfileRelease> for RankRelease {
    fn from(release: &wire::ProfileRelease) -> Self {
        Self {
            rank: release.rank,
            released: release.released,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    /// A session's start crosses the request interface unchanged, every option included.
    #[test]
    fn a_session_start_crosses_the_interface_unchanged() {
        let start = SessionStart {
            session: 7,
            epoch: 2,
            request: SessionRequest {
                collectors: vec![Collector::Timeline, Collector::RangeMetrics],
                start_step: 3,
                steps: Some(8),
                seconds: Some(1.5),
                stages: vec!["decode".to_owned()],
                workers: vec![0, 2],
                modules: vec!["*Attention".to_owned()],
                output_dir: "profiles".into(),
                timeline: TimelineOptions {
                    graphs: GraphTrace::Nodes,
                },
                pm_sampling: PmSamplingOptions::default(),
                range_metrics: RangeMetricsOptions {
                    ranges: MetricRanges::Module,
                    ..RangeMetricsOptions::default()
                },
                pc_sampling: PcSamplingOptions { period: 20 },
                max_slowdown: Some(0.1),
            },
        };
        let message = wire::ProfileSession::from(&start);
        assert_eq!(SessionStart::try_from(&message).unwrap(), start);
    }

    /// Each rank's reports cross the request interface unchanged, a rank without a file or a
    /// refusal included.
    #[test]
    fn rank_reports_cross_the_interface_unchanged() {
        let started = RankStart {
            rank: 1,
            selected: true,
            refused: None,
            device_storage: 1_420_288,
            path: Some("profiles/profile-7-epoch2-rank1.sqlite".into()),
            collectors: BTreeMap::from([("timeline".to_owned(), "prepared".to_owned())]),
        };
        assert_eq!(
            RankStart::from(&wire::ProfileStart::from(&started)),
            started
        );
        let unselected = RankStart::unselected(3);
        assert_eq!(
            RankStart::from(&wire::ProfileStart::from(&unselected)),
            unselected
        );

        for file in [
            Some(FileReport::Written {
                path: "profiles/profile-7-epoch2-rank1.sqlite".into(),
                bytes: 4096,
                rows: BTreeMap::from([("kernels".to_owned(), 12)]),
            }),
            Some(FileReport::Failed {
                error: "disk full".to_owned(),
            }),
            None,
        ] {
            let stopped = RankStop {
                rank: 1,
                session: 7,
                selected: true,
                steps: 8,
                reason: "window".to_owned(),
                file,
            };
            assert_eq!(RankStop::from(&wire::ProfileStop::from(&stopped)), stopped);
        }

        let released = RankRelease {
            rank: 2,
            released: true,
        };
        assert_eq!(
            RankRelease::from(&wire::ProfileRelease::from(&released)),
            released
        );
    }

    /// A command that names no collector is refused.
    #[test]
    fn an_unnamed_collector_is_refused() {
        assert!(Collector::from_wire(wire::ProfileCollector::Unspecified as i32).is_err());
        assert_eq!(
            Collector::from_wire(wire::ProfileCollector::PcSampling as i32),
            Ok(Collector::PcSampling)
        );
    }
}
