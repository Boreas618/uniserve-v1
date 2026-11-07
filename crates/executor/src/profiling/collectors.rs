//! A session's collectors on one rank, and their arming at step boundaries.
//!
//! Every CUPTI-based collector reaches CUPTI through the process's one
//! subscription (`platform::cupti::subscriber`): the timeline enables its records
//! and callbacks on it, and the profiling APIs (range profiler, PM sampling,
//! PC sampling) belong to the same CUPTI. A collector is prepared when the
//! session starts ([`Collectors::prepare`]), armed and disarmed at step
//! boundaries ([`Collectors::arm`], [`Collectors::disarm`]), drained while
//! armed by the session's sampler thread ([`Collectors::drain`]), and
//! released when the session ends or the quality-of-service guard turns it
//! off ([`Collectors::release`]).

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

use platform::cupti::pc_sampling::PcSampler;
use platform::cupti::pm_sampling::PmSampler;
use platform::cupti::range_profiler::{RangeProfiler, Ranges};
use platform::cupti::records::Record;
use platform::cupti::timeline::{self, Timeline};
use platform::{DeviceId, PrimaryContext};
use proto::profile::{Collector, GraphTrace, MetricRanges, SessionRequest};

use super::artifact::{Message, Rows};
use super::fences::{Fences, Tap};
use super::pc_sampling;

/// The fewest samples a PM sampler's counter data image and hardware
/// buffer hold.
const PM_MIN_SAMPLES: u64 = 256;

/// A prepared collector.
enum Prepared {
    Timeline {
        graphs: GraphTrace,
        /// Whether this rank records into the process's timeline.
        joined: bool,
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
    /// The rank's PC sampler ([`pc_sampling::of`]), enabled for the
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
    device: Option<DeviceId>,
    context: Option<PrimaryContext>,
    prepared: BTreeMap<Collector, Prepared>,
    rows: Rows,
    armed: bool,
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

/// The process's timeline: the recording, and the ranks it records for.
#[derive(Default)]
struct SharedTimeline {
    running: Option<Timeline>,
    /// How the running timeline records a graph's launch.
    graphs: Option<GraphTrace>,
    /// The NVTX switch before the timeline turned it on.
    nvtx_before: Option<bool>,
}

static TIMELINE: Mutex<Option<SharedTimeline>> = Mutex::new(None);

/// Where each recording rank's records go, by the device ordinal CUPTI names.
///
/// Apart from [`TIMELINE`] because a timeline that stops delivers what remains from under that lock.
static TIMELINE_RANKS: Mutex<BTreeMap<u32, Rows>> = Mutex::new(BTreeMap::new());

/// The device a timeline record ran on, when it names one.
fn device_of(record: &Record) -> Option<u32> {
    match record {
        Record::Kernel(record) => Some(record.device),
        Record::Copy(record) => Some(record.device),
        Record::Fill(record) => Some(record.device),
        Record::Memory(record) => Some(record.device),
        Record::Graph(record) => Some(record.device),
        Record::Marker(_)
        | Record::MarkerData(_)
        | Record::NvLink(_)
        | Record::Pcie(_)
        | Record::Api(_) => None,
    }
}

/// Hands one delivery to the recording ranks: a record to its device's rank, and one that names no
/// device to all of them. Every rank gets the names, which later records refer to by id.
fn deliver(delivered: timeline::Delivered) {
    let ranks = crate::lock(&TIMELINE_RANKS);
    if let [(_, rows)] = ranks.iter().collect::<Vec<_>>()[..] {
        rows.send(Message::Timeline {
            records: delivered.records,
            names: delivered.names,
        });
        return;
    }
    let mut records: BTreeMap<u32, Vec<Record>> =
        ranks.keys().map(|&key| (key, Vec::new())).collect();
    for record in delivered.records {
        match device_of(&record) {
            Some(device) => {
                if let Some(held) = records.get_mut(&device) {
                    held.push(record);
                }
            }
            None => {
                for held in records.values_mut() {
                    held.push(record.clone());
                }
            }
        }
    }
    for (device, held) in records {
        if let Some(rows) = ranks.get(&device) {
            rows.send(Message::Timeline {
                records: held,
                names: delivered.names.clone(),
            });
        }
    }
}

/// The rank of `device` records into the process's timeline, starting it when no rank does.
fn join_timeline(device: DeviceId, graphs: GraphTrace, rows: &Rows) -> platform::Result<()> {
    let mut shared = crate::lock(&TIMELINE);
    let shared = shared.get_or_insert_with(SharedTimeline::default);
    let ordinal = device.0 as u32;
    if shared.running.is_some() {
        if shared.graphs != Some(graphs) {
            return Err(platform::Error::InvalidArgument(
                "the process's timeline records graphs another way".to_owned(),
            ));
        }
        crate::lock(&TIMELINE_RANKS).insert(ordinal, rows.clone());
        return Ok(());
    }
    crate::lock(&TIMELINE_RANKS).insert(ordinal, rows.clone());
    let traced = match graphs {
        GraphTrace::Whole => timeline::GraphTrace::Whole,
        GraphTrace::Nodes => timeline::GraphTrace::Nodes,
    };
    // A timeline of a graph's nodes needs the counters' whole-graph record out of the way.
    let by_default = super::graphs_whole_by_default();
    if graphs == GraphTrace::Nodes && by_default {
        platform::cupti::activity::hold_graph_trace(false)?;
    }
    let sink: timeline::Sink = Arc::new(deliver);
    match Timeline::start(traced, sink) {
        Ok(started) => {
            shared.running = Some(started);
            shared.graphs = Some(graphs);
            // An executable replays in the tracing mode it was instantiated in.
            super::set_graph_trace(graphs == GraphTrace::Whole);
            shared.nvtx_before = Some(super::ranges::set_enabled(true));
            Ok(())
        }
        Err(error) => {
            crate::lock(&TIMELINE_RANKS).remove(&ordinal);
            if by_default {
                platform::cupti::activity::hold_graph_trace(true)?;
            }
            Err(error)
        }
    }
}

/// The rank of `device` stops recording; the last to go stops the timeline, which delivers what
/// remains.
fn leave_timeline(device: DeviceId) {
    let mut shared = crate::lock(&TIMELINE);
    let Some(shared) = shared.as_mut() else {
        return;
    };
    let ordinal = device.0 as u32;
    let others = crate::lock(&TIMELINE_RANKS)
        .keys()
        .any(|&held| held != ordinal);
    if others {
        // What this rank's device ran so far reaches its file before its rows go.
        timeline::flush();
    } else if let Some(running) = shared.running.take() {
        drop(running);
        let by_default = super::graphs_whole_by_default();
        if by_default && let Err(error) = platform::cupti::activity::hold_graph_trace(true) {
            tracing::warn!("the counters' whole-graph record did not resume: {error}");
        }
        super::set_graph_trace(by_default);
        shared.graphs = None;
        if let Some(before) = shared.nvtx_before.take() {
            super::ranges::set_enabled(before);
        }
    }
    crate::lock(&TIMELINE_RANKS).remove(&ordinal);
}

/// The armed component-range profiler of each device, which the probes push and pop ranges on: a
/// probe's entry pushes and its exit pops.
static COMPONENT_RANGES: Mutex<Option<HashMap<i32, ComponentRanges>>> = Mutex::new(None);

/// Whether any device publishes a profiler, read by every probe.
static COMPONENT_RANGES_ARMED: AtomicBool = AtomicBool::new(false);

/// Publish or withdraw `device`'s component-range profiler.
fn publish_component_ranges(device: DeviceId, ranges: Option<ComponentRanges>) {
    let mut published = crate::lock(&COMPONENT_RANGES);
    let map = published.get_or_insert_with(HashMap::new);
    match ranges {
        Some(ranges) => {
            map.insert(device.0, ranges);
        }
        None => {
            map.remove(&device.0);
        }
    }
    COMPONENT_RANGES_ARMED.store(!map.is_empty(), Ordering::Release);
}

/// Whether a probe pushes and pops component ranges now.
pub(crate) fn component_ranges_armed() -> bool {
    COMPONENT_RANGES_ARMED.load(Ordering::Acquire)
}

/// The armed component-range profiler: its object, reached by the probes.
struct ComponentRanges {
    /// The profiler, shared with the collector that owns it.
    profiler: *const RangeProfiler,
    names: Vec<String>,
}

// SAFETY: the profiler pointer is valid while the owning collector is armed (the collector clears it
// before it disarms or drops); CUPTI accepts the range calls from any thread.
unsafe impl Send for ComponentRanges {}

/// The probes' part: push or pop the component range of `probe`'s instance on `device`, while
/// component ranges are armed and `capturing` is false (a capture pushes no host range).
pub(crate) fn component_range(device: DeviceId, probe: u32, capturing: bool) {
    if capturing {
        return;
    }
    let published = crate::lock(&COMPONENT_RANGES);
    let Some(ranges) = published.as_ref().and_then(|map| map.get(&device.0)) else {
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
                device,
                context,
                prepared,
                rows,
                armed: false,
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
                Prepared::Timeline { graphs, joined } => match self.device {
                    Some(device) => join_timeline(device, *graphs, &self.rows).map(|()| {
                        *joined = true;
                    }),
                    None => Err(platform::Error::InvalidArgument(
                        "the rank has no CUDA device".to_owned(),
                    )),
                },
                Prepared::FenceTimeline { fences, components } => match self.device {
                    Some(device) => {
                        let rows = self.rows.clone();
                        fences
                            .tap(
                                device,
                                Tap {
                                    components: components.clone(),
                                    sink: Box::new(move |device, records| {
                                        rows.send(Message::Fences { device, records });
                                    }),
                                },
                            )
                            .map_err(|error| platform::Error::InvalidArgument(error.to_string()))
                    }
                    None => Err(platform::Error::InvalidArgument(
                        "the rank has no CUDA device".to_owned(),
                    )),
                },
                Prepared::PmSampling(sampler) => sampler.start(),
                Prepared::RangeMetrics { profiler, ranges } => profiler.start().map(|()| {
                    if *ranges == MetricRanges::Component
                        && let (Some(fences), Some(device)) = (Fences::installed(), self.device)
                    {
                        publish_component_ranges(
                            device,
                            Some(ComponentRanges {
                                profiler: &raw const *profiler,
                                names: fences
                                    .instances()
                                    .iter()
                                    .map(|instance| instance.path.clone())
                                    .collect(),
                            }),
                        );
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
                } => with_rank_sampler(self.device, |sampler| {
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
        if let Some(device) = self.device {
            publish_component_ranges(device, None);
        }
        let mut failed = Vec::new();
        for (&collector, prepared) in self.prepared.iter_mut().rev() {
            let stopped = stop(prepared, self.device, &self.rows);
            if let Err(error) = stopped {
                failed.push((collector, error));
            }
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
                Prepared::RankPcSampling { .. } => with_rank_sampler(self.device, |sampler| {
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
    ///
    /// A stop or start CUPTI refuses releases the collector as failed, with
    /// CUPTI's result as the detail of its state: the profiler collects
    /// nothing more, and the ranges of a failed stop are lost.
    pub(crate) fn cycle_ranges(&mut self) {
        if !self.armed {
            return;
        }
        let _context = self
            .context
            .as_ref()
            .and_then(|context| context.enter().ok());
        let Some(Prepared::RangeMetrics { profiler, .. }) =
            self.prepared.get_mut(&Collector::RangeMetrics)
        else {
            return;
        };
        let cycled = profiler.stop().and_then(|collected| {
            send_ranges(&self.rows, profiler, collected);
            profiler.start()
        });
        if let Err(error) = cycled {
            let reason = error.to_string();
            tracing::warn!(
                "profiling collector {} failed between two passes: {reason}",
                Collector::RangeMetrics.as_str()
            );
            self.release(Collector::RangeMetrics, "failed", &reason);
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
            if collector == Collector::RangeMetrics
                && let Some(device) = self.device
            {
                publish_component_ranges(device, None);
            }
            if let Err(error) = stop(&mut prepared, self.device, &self.rows) {
                tracing::warn!(
                    "profiling collector {} failed to stop: {error}",
                    collector.as_str()
                );
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

/// Stop one collector of the rank of `device` and hand what it collected to the file.
fn stop(
    prepared: &mut Prepared,
    device: Option<DeviceId>,
    rows: &Rows,
) -> std::result::Result<(), String> {
    match prepared {
        Prepared::Timeline { joined, .. } => {
            if std::mem::take(joined)
                && let Some(device) = device
            {
                leave_timeline(device);
            }
            Ok(())
        }
        Prepared::FenceTimeline { fences, .. } => {
            if let Some(device) = device {
                fences.poll_device(device, true);
                fences.untap(device);
            }
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
        Prepared::RankPcSampling { .. } => with_rank_sampler(device, |sampler| {
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
    collected: platform::cupti::range_profiler::Collected,
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
    let self_device = context.map(PrimaryContext::device);
    let cupti = || {
        if !super::counters::cupti_serves() {
            return Err(super::counters::REFUSED.to_owned());
        }
        platform::cupti::subscriber::subscribe().map_err(|error| {
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
                joined: false,
            })
        }
        Collector::FenceTimeline => {
            let fences = Fences::installed().ok_or_else(|| {
                "the process has no fence probes (profiling_config.fence_components)".to_owned()
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
            let per_read =
                (super::session::DRAIN_PERIOD.as_nanos() as u64).div_ceil(options.interval_ns);
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
                MetricRanges::Kernel => Ranges::Launches,
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
            let context_device = self_device;
            if context_device.is_some_and(|device| pc_sampling::of(device).is_some()) {
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

/// Run `work` on the PC sampler of the rank of `device`.
fn with_rank_sampler(
    device: Option<DeviceId>,
    work: impl FnOnce(&mut PcSampler) -> platform::Result<()>,
) -> platform::Result<()> {
    let sampler = device.and_then(pc_sampling::of).ok_or_else(|| {
        platform::Error::InvalidArgument("the rank keeps no PC sampler".to_owned())
    })?;
    work(&mut crate::lock(&sampler))
}
