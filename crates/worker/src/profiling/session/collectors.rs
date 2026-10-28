//! A session's collectors on one rank, and their arming at step boundaries.
//!
//! Every CUPTI-based collector reaches CUPTI through the process's one
//! subscription (`cuda::cupti::subscriber`): the timeline enables its records
//! and callbacks on it, and the profiling APIs (range profiler, PM sampling,
//! PC sampling) belong to the same CUPTI. A collector is prepared when the
//! session starts ([`Collectors::prepare`]), armed and disarmed at step
//! boundaries ([`Collectors::arm`], [`Collectors::disarm`]), drained while
//! armed by the session's sampler thread ([`Collectors::drain`]), and
//! released when the session ends or the quality-of-service guard turns it
//! off ([`Collectors::release`]).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use cuda::cupti::pc_sampling::PcSampler;
use cuda::cupti::pm_sampling::PmSampler;
use cuda::cupti::range_profiler::{RangeProfiler, Ranges};
use cuda::cupti::timeline::{self, Timeline};
use cuda::{DeviceId, PrimaryContext};
use proto::profile::{Collector, GraphTrace, MetricRanges, SessionRequest};

use super::artifact::{Message, Rows};
use crate::profiling::fences::{Fences, Tap};
use crate::profiling::pc_sampling;

/// The fewest samples a PM sampler's counter data image and hardware
/// buffer hold.
const PM_MIN_SAMPLES: u64 = 256;

/// The device storage a session of `request` holds on the device of rank
/// `rank` while it runs ([`SessionRequest::device_storage`]), for a session
/// that selects the rank; 0 otherwise.
pub(crate) fn device_storage(request: &SessionRequest, rank: u32) -> u64 {
    if request.workers.is_empty() || request.workers.contains(&rank) {
        request.device_storage()
    } else {
        0
    }
}

/// A prepared collector.
enum Prepared {
    Timeline {
        graphs: GraphTrace,
        running: Option<Timeline>,
        graph_trace: crate::ffi::Function,
    },
    FenceTimeline {
        fences: Arc<Fences>,
        components: Vec<String>,
    },
    PmSampling(PmSampler),
    RangeMetrics {
        profiler: RangeProfiler,
        ranges: MetricRanges,
    },
    /// The session's own PC sampler of the context `context`, every
    /// `2^period` cycles, enabled while armed: CUPTI refuses to enable
    /// activity records (`CUPTI_ERROR_NOT_COMPATIBLE`) while PC sampling
    /// samples, so it is enabled at each arm after the timeline started
    /// and disabled at each disarm before the timeline stops. It attributes
    /// only kernels launched outside CUDA graphs ([`pc_sampling`]).
    PcSampling {
        context: usize,
        period: u32,
        sampler: Option<PcSampler>,
        /// Whether the stall reasons' names reached the file.
        reasons: bool,
    },
    /// The rank's PC sampler ([`pc_sampling::rank`]), enabled for the
    /// rank's life: the first arm sets the session's period and a hardware
    /// buffer that holds it, each arm starts it and each disarm stops it,
    /// and it stays enabled after the session.
    RankPcSampling {
        period: u32,
        /// Whether the session's period and buffer are set.
        configured: bool,
        /// Whether the stall reasons' names reached the file.
        reasons: bool,
    },
}

/// The collectors of a session on one rank.
pub(crate) struct Collectors {
    context: Option<PrimaryContext>,
    prepared: BTreeMap<Collector, Prepared>,
    rows: Rows,
    armed: bool,
    /// The NVTX switch before a timeline armed it.
    nvtx_before: Option<bool>,
    /// The collectors prepared and not released, a bit each ([`bit`]),
    /// which the session reads at every step without taking the lock the
    /// sampler thread holds while it drains.
    running: Arc<AtomicU32>,
}

/// The bit of `collector` in [`Collectors::running`].
pub(crate) fn bit(collector: Collector) -> u32 {
    let index = Collector::ALL
        .iter()
        .position(|&known| known == collector)
        .unwrap_or_default();
    1 << index
}

/// The armed component-range profiler, which the probes push and pop
/// ranges on: a probe's entry pushes and its exit pops.
static COMPONENT_RANGES: std::sync::Mutex<Option<ComponentRanges>> = std::sync::Mutex::new(None);

/// Whether [`COMPONENT_RANGES`] holds a profiler, read by every probe.
static COMPONENT_RANGES_ARMED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Publish or withdraw the component-range profiler.
fn publish_component_ranges(ranges: Option<ComponentRanges>) {
    let armed = ranges.is_some();
    *crate::lock(&COMPONENT_RANGES) = ranges;
    COMPONENT_RANGES_ARMED.store(armed, std::sync::atomic::Ordering::Release);
}

/// Whether a probe pushes and pops component ranges now.
pub(crate) fn component_ranges_armed() -> bool {
    COMPONENT_RANGES_ARMED.load(std::sync::atomic::Ordering::Acquire)
}

/// The armed component-range profiler: its object, reached by the probes.
struct ComponentRanges {
    /// The profiler, shared with the collector that owns it.
    profiler: *const RangeProfiler,
    names: Vec<String>,
}

// SAFETY: the profiler pointer is valid while the owning collector is armed
// (the collector clears it before it disarms or drops); CUPTI accepts the
// range calls from any thread.
unsafe impl Send for ComponentRanges {}

/// The probes' part: push or pop the component range of `probe`'s instance,
/// while component ranges are armed and `capturing` is false (a capture
/// pushes no host range).
pub(crate) fn component_range(probe: u32, capturing: bool) {
    if capturing {
        return;
    }
    let ranges = crate::lock(&COMPONENT_RANGES);
    let Some(ranges) = ranges.as_ref() else {
        return;
    };
    // SAFETY: the pointer is valid while it is published (above).
    let profiler = unsafe { &*ranges.profiler };
    let result = if probe.is_multiple_of(2) {
        match ranges.names.get((probe / 2) as usize) {
            Some(name) => profiler.push(name),
            None => return,
        }
    } else {
        profiler.pop()
    };
    if let Err(error) = result {
        tracing::debug!("a component range was not recorded: {error}");
    }
}

impl Collectors {
    /// Prepare `request`'s collectors on `device`: what fails to prepare is
    /// reported as unavailable, with its reason, and the session runs the
    /// others.
    pub(crate) fn prepare(
        request: &SessionRequest,
        device: Option<DeviceId>,
        rows: Rows,
    ) -> (Self, BTreeMap<String, String>) {
        let context = device.and_then(|device| PrimaryContext::retain(device).ok());
        let mut prepared = BTreeMap::new();
        let mut states = BTreeMap::new();
        for &collector in &request.collectors {
            match prepare(collector, request, context.as_ref()) {
                Ok(ready) => {
                    prepared.insert(collector, ready);
                    states.insert(collector.as_str().to_owned(), "prepared".to_owned());
                    rows.send(Message::Collector {
                        name: collector.as_str().to_owned(),
                        state: "prepared".to_owned(),
                        detail: String::new(),
                    });
                }
                Err(reason) => {
                    tracing::warn!(
                        "profiling collector {} is unavailable: {reason}",
                        collector.as_str()
                    );
                    states.insert(
                        collector.as_str().to_owned(),
                        format!("unavailable: {reason}"),
                    );
                    rows.send(Message::Collector {
                        name: collector.as_str().to_owned(),
                        state: "unavailable".to_owned(),
                        detail: reason,
                    });
                }
            }
        }
        let running = prepared
            .keys()
            .fold(0, |bits, &collector| bits | bit(collector));
        (
            Self {
                context,
                prepared,
                rows,
                armed: false,
                nvtx_before: None,
                running: Arc::new(AtomicU32::new(running)),
            },
            states,
        )
    }

    /// The set of collectors prepared and not released, as [`bit`]s.
    pub(crate) fn running(&self) -> Arc<AtomicU32> {
        Arc::clone(&self.running)
    }

    /// The collectors prepared and not released.
    pub(crate) fn names(&self) -> Vec<Collector> {
        self.prepared.keys().copied().collect()
    }

    /// Start every prepared collector, in the order of [`Collector`], which
    /// starts the timeline before PC sampling as CUPTI requires; one that
    /// fails to start is released, with its reason recorded.
    pub(crate) fn arm(&mut self) {
        if self.armed {
            return;
        }
        let _context = self
            .context
            .as_ref()
            .and_then(|context| context.enter().ok());
        let mut failed = Vec::new();
        for (&collector, prepared) in &mut self.prepared {
            let started = match prepared {
                Prepared::Timeline {
                    graphs,
                    running,
                    graph_trace,
                } => {
                    let rows = self.rows.clone();
                    let sink: timeline::Sink = Arc::new(move |delivered: timeline::Delivered| {
                        rows.send(Message::Timeline {
                            records: delivered.records,
                            names: delivered.names,
                        });
                    });
                    let graphs = match graphs {
                        GraphTrace::Whole => timeline::GraphTrace::Whole,
                        GraphTrace::Nodes => timeline::GraphTrace::Nodes,
                    };
                    Timeline::start(graphs, sink).and_then(|started| {
                        graph_trace
                            .call_values(&[tvm_ffi::Any::from(
                                graphs == timeline::GraphTrace::Whole,
                            )])
                            .map_err(|error| cuda::Error::InvalidArgument(error.to_string()))?;
                        *running = Some(started);
                        self.nvtx_before = Some(nvtx(true));
                        Ok(())
                    })
                }
                Prepared::FenceTimeline { fences, components } => {
                    let rows = self.rows.clone();
                    fences.tap(Tap {
                        components: components.clone(),
                        sink: Box::new(move |device, records| {
                            rows.send(Message::Fences { device, records });
                        }),
                    });
                    Ok(())
                }
                Prepared::PmSampling(sampler) => sampler.start(),
                Prepared::RangeMetrics { profiler, ranges } => profiler.start().map(|()| {
                    if *ranges == MetricRanges::Component
                        && let Some(fences) = Fences::installed()
                    {
                        publish_component_ranges(Some(ComponentRanges {
                            profiler: &raw const *profiler,
                            names: fences
                                .instances()
                                .iter()
                                .map(|instance| instance.path.clone())
                                .collect(),
                        }));
                    }
                }),
                Prepared::PcSampling {
                    context,
                    period,
                    sampler,
                    reasons,
                } => PcSampler::enable(
                    *context,
                    *period,
                    pc_sampling::MAX_PCS,
                    pc_sampling::hardware_buffer(*period),
                )
                .and_then(|mut enabled| {
                    enabled.start()?;
                    if !*reasons {
                        self.rows
                            .send(Message::PcReasons(enabled.stall_reasons().to_vec()));
                        *reasons = true;
                    }
                    *sampler = Some(enabled);
                    Ok(())
                }),
                Prepared::RankPcSampling {
                    period,
                    configured,
                    reasons,
                } => with_rank_sampler(|sampler| {
                    if !*configured {
                        sampler.set_period(*period)?;
                        if let Some(bytes) = pc_sampling::hardware_buffer(*period) {
                            sampler.set_hardware_buffer(bytes)?;
                        }
                        *configured = true;
                    }
                    sampler.start()?;
                    if !*reasons {
                        self.rows
                            .send(Message::PcReasons(sampler.stall_reasons().to_vec()));
                        *reasons = true;
                    }
                    Ok(())
                }),
            };
            if let Err(error) = started {
                failed.push((collector, error.to_string()));
            }
        }
        for (collector, reason) in failed {
            tracing::warn!(
                "profiling collector {} did not start: {reason}",
                collector.as_str()
            );
            self.release(collector, "failed", &reason);
        }
        self.armed = true;
    }

    /// Stop every armed collector and hand what it collected to the file,
    /// in the reverse order of [`Collectors::arm`]: PC sampling is disabled
    /// before the timeline stops.
    pub(crate) fn disarm(&mut self) {
        if !self.armed {
            return;
        }
        let _context = self
            .context
            .as_ref()
            .and_then(|context| context.enter().ok());
        publish_component_ranges(None);
        let mut failed = Vec::new();
        for (&collector, prepared) in self.prepared.iter_mut().rev() {
            let stopped = stop(prepared, &self.rows);
            if let Err(error) = stopped {
                failed.push((collector, error));
            }
        }
        if let Some(before) = self.nvtx_before.take() {
            nvtx(before);
        }
        self.armed = false;
        for (collector, reason) in failed {
            tracing::warn!(
                "profiling collector {} failed to stop: {reason}",
                collector.as_str()
            );
            self.release(collector, "failed", &reason);
        }
    }

    /// Hand what the armed samplers collected so far to the file: PM samples
    /// and PC samples, which the hardware buffers would otherwise
    /// overflow.
    pub(crate) fn drain(&mut self) {
        if !self.armed {
            return;
        }
        let _context = self
            .context
            .as_ref()
            .and_then(|context| context.enter().ok());
        for prepared in self.prepared.values_mut() {
            let drained = match prepared {
                Prepared::PmSampling(sampler) => sampler.decode().map(|decoded| {
                    if decoded.overflow {
                        tracing::warn!("a PM sampling read filled its image: samples were lost");
                    }
                    self.rows.send(Message::Pm {
                        metrics: sampler.metrics(),
                        samples: decoded.samples,
                    });
                }),
                Prepared::PcSampling {
                    sampler: Some(sampler),
                    ..
                } => sampler.take().map(|taken| {
                    if taken.total_samples > 0 {
                        self.rows.send(Message::Pc(taken));
                    }
                }),
                Prepared::RankPcSampling { .. } => with_rank_sampler(|sampler| {
                    let taken = sampler.take()?;
                    if taken.total_samples > 0 {
                        self.rows.send(Message::Pc(taken));
                    }
                    Ok(())
                }),
                _ => Ok(()),
            };
            if let Err(error) = drained {
                tracing::warn!("a profiling collector could not be drained: {error}");
            }
        }
    }

    /// Stop the range profiler, hand its ranges to the file and start it
    /// again, so that a long window never fills its counter data image.
    pub(crate) fn cycle_ranges(&mut self) {
        if !self.armed {
            return;
        }
        let _context = self
            .context
            .as_ref()
            .and_then(|context| context.enter().ok());
        if let Some(Prepared::RangeMetrics { profiler, .. }) =
            self.prepared.get_mut(&Collector::RangeMetrics)
        {
            match profiler.stop() {
                Ok(collected) => {
                    send_ranges(&self.rows, profiler, collected);
                    if let Err(error) = profiler.start() {
                        tracing::warn!("the range profiler did not start again: {error}");
                    }
                }
                Err(error) => tracing::warn!("the range profiler could not be read: {error}"),
            }
        }
    }

    /// Release `collector`: stop it if armed, hand what it collected to the
    /// file and record why it was released (`state`: `failed`, `released`
    /// by the quality-of-service guard, or `ended`).
    pub(crate) fn release(&mut self, collector: Collector, state: &str, detail: &str) -> bool {
        let Some(mut prepared) = self.prepared.remove(&collector) else {
            return false;
        };
        self.running.fetch_and(!bit(collector), Ordering::AcqRel);
        let _context = self
            .context
            .as_ref()
            .and_then(|context| context.enter().ok());
        if self.armed {
            if collector == Collector::RangeMetrics {
                publish_component_ranges(None);
            }
            if let Err(error) = stop(&mut prepared, &self.rows) {
                tracing::warn!(
                    "profiling collector {} failed to stop: {error}",
                    collector.as_str()
                );
            }
            if collector == Collector::Timeline
                && let Some(before) = self.nvtx_before.take()
            {
                nvtx(before);
            }
        }
        drop(prepared);
        self.rows.send(Message::Collector {
            name: collector.as_str().to_owned(),
            state: state.to_owned(),
            detail: detail.to_owned(),
        });
        true
    }

    /// Release every collector, the session ending.
    pub(crate) fn release_all(&mut self, reason: &str) {
        self.disarm();
        for collector in self.names() {
            self.release(collector, "ended", reason);
        }
    }
}

/// Stop one collector and hand what it collected to the file.
fn stop(prepared: &mut Prepared, rows: &Rows) -> std::result::Result<(), String> {
    match prepared {
        Prepared::Timeline {
            running,
            graph_trace,
            ..
        } => {
            // Dropping the timeline delivers what remains to the file.
            if let Some(timeline) = running.take() {
                drop(timeline);
                graph_trace
                    .call_values(&[tvm_ffi::Any::from(false)])
                    .map_err(|error| error.to_string())?;
            }
            Ok(())
        }
        Prepared::FenceTimeline { fences, .. } => {
            fences.poll(true);
            fences.untap();
            Ok(())
        }
        Prepared::PmSampling(sampler) => {
            sampler.stop().map_err(|error| error.to_string())?;
            let decoded = sampler.decode().map_err(|error| error.to_string())?;
            if decoded.overflow {
                tracing::warn!("a PM sampling read filled its image: samples were lost");
            }
            rows.send(Message::Pm {
                metrics: sampler.metrics(),
                samples: decoded.samples,
            });
            Ok(())
        }
        Prepared::RangeMetrics { profiler, .. } => {
            let collected = profiler.stop().map_err(|error| error.to_string())?;
            send_ranges(rows, profiler, collected);
            Ok(())
        }
        Prepared::PcSampling { sampler, .. } => {
            // Dropping the sampler disables PC sampling.
            let Some(mut sampler) = sampler.take() else {
                return Ok(());
            };
            sampler.stop().map_err(|error| error.to_string())?;
            let taken = sampler.take().map_err(|error| error.to_string())?;
            rows.send(Message::Pc(taken));
            Ok(())
        }
        Prepared::RankPcSampling { .. } => with_rank_sampler(|sampler| {
            sampler.stop()?;
            let taken = sampler.take()?;
            rows.send(Message::Pc(taken));
            Ok(())
        })
        .map_err(|error| error.to_string()),
    }
}

fn send_ranges(
    rows: &Rows,
    profiler: &RangeProfiler,
    collected: cuda::cupti::range_profiler::Collected,
) {
    if collected.dropped > 0 {
        tracing::warn!(
            "the range profiler dropped {} ranges: its counter data image was full",
            collected.dropped
        );
    }
    rows.send(Message::Ranges {
        metrics: profiler.metrics(),
        ranges: collected.ranges,
    });
}

/// Prepare one collector, or say why it is unavailable.
fn prepare(
    collector: Collector,
    request: &SessionRequest,
    context: Option<&PrimaryContext>,
) -> std::result::Result<Prepared, String> {
    let cupti = || {
        cuda::cupti::subscriber::subscribe().map_err(|error| {
            format!("the process holds no CUPTI subscription (a profiler may hold it): {error}")
        })
    };
    let device = || {
        context
            .map(|context| (context.handle(), context.device().0 as usize))
            .ok_or_else(|| "the rank has no CUDA device".to_owned())
    };
    match collector {
        Collector::Timeline => {
            cupti()?;
            Ok(Prepared::Timeline {
                graphs: request.timeline.graphs,
                running: None,
                graph_trace: crate::ffi::Interpreter::import("uniserve.cuda_graph.set_graph_trace")
                    .map_err(|error| error.to_string())?,
            })
        }
        Collector::FenceTimeline => {
            let fences = Fences::installed().ok_or_else(|| {
                "the rank has no fence probes (profiling_config.fence_components)".to_owned()
            })?;
            Ok(Prepared::FenceTimeline {
                fences: Arc::clone(fences),
                components: request.components.clone(),
            })
        }
        Collector::PmSampling => {
            cupti()?;
            let (_, device) = device()?;
            let options = &request.pm_sampling;
            // The image and the hardware buffer hold twice the samples taken
            // between two reads: a read whose image fills loses the samples
            // past it. The buffer takes the image's bytes per sample, which
            // exceed a raw record's (on GB200, 13.6 KB of image against 8 KiB
            // of buffer for the default metrics).
            let per_read = (super::DRAIN_PERIOD.as_nanos() as u64).div_ceil(options.interval_ns);
            let samples = (2 * per_read).max(PM_MIN_SAMPLES);
            PmSampler::sample_bytes(device, &options.metrics)
                .and_then(|bytes| {
                    PmSampler::enable(
                        device,
                        &options.metrics,
                        options.interval_ns,
                        (samples as usize * bytes).next_multiple_of(1 << 20),
                        u32::try_from(samples).unwrap_or(u32::MAX),
                    )
                })
                .map_err(|error| error.to_string())
                .map(Prepared::PmSampling)
        }
        Collector::RangeMetrics => {
            cupti()?;
            let (context, device) = device()?;
            let options = &request.range_metrics;
            let ranges = match options.ranges {
                MetricRanges::Kernel => Ranges::Kernels,
                MetricRanges::Component => {
                    if Fences::installed().is_none() {
                        return Err("component ranges need the rank's fence probes \
                             (profiling_config.fence_components)"
                            .to_owned());
                    }
                    Ranges::User
                }
            };
            RangeProfiler::enable(
                context,
                device,
                &options.metrics,
                ranges,
                options.pass_ranges,
            )
            .map_err(|error| error.to_string())
            .map(|profiler| Prepared::RangeMetrics {
                profiler,
                ranges: options.ranges,
            })
        }
        Collector::PcSampling => {
            cupti()?;
            let (context, _) = device()?;
            if pc_sampling::rank().is_some() {
                return Ok(Prepared::RankPcSampling {
                    period: request.pc_sampling.period,
                    configured: false,
                    reasons: false,
                });
            }
            Ok(Prepared::PcSampling {
                context,
                period: request.pc_sampling.period,
                sampler: None,
                reasons: false,
            })
        }
    }
}

/// Run `work` on the rank's PC sampler.
fn with_rank_sampler(work: impl FnOnce(&mut PcSampler) -> cuda::Result<()>) -> cuda::Result<()> {
    let sampler = pc_sampling::rank()
        .ok_or_else(|| cuda::Error::InvalidArgument("the rank keeps no PC sampler".to_owned()))?;
    work(&mut crate::lock(sampler))
}

/// Set the NVTX switch of the rank's ranges and of the library's; returns
/// the rank's switch before.
fn nvtx(enabled: bool) -> bool {
    let before = crate::profiling::ranges::set_enabled(enabled);
    let set = crate::ffi::Interpreter::import("uniserve.profiling.set_nvtx").and_then(|function| {
        function
            .call_values(&[tvm_ffi::Any::from(enabled)])
            .map(drop)
    });
    if let Err(error) = set {
        tracing::warn!("the library's NVTX switch was not set: {error}");
    }
    before
}
