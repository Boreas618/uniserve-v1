//! Execution streams of a rank.
//!
//! Configured lanes partition one device's SMs (green contexts), each with
//! its own stream and the call kinds it serves. Without lanes, each CUDA
//! device has one full-device stream. Every device also has a preparation
//! stream for startup work outside any lane. Capability runtimes fork
//! module and microbatch streams from these and register them here so they
//! close in dependency order: preparation, modules, microbatches, lanes,
//! each newest first.

use uniserve_cuda::DeviceId;
use uniserve_ipc::{CallKind, LaneConfig};

use crate::config::{Device, WorkerConfig};
use crate::stream::CUDAStream;
use crate::{Error, Result};

/// One execution lane.
pub struct Lane {
    pub config: LaneConfig,
    pub stream: CUDAStream,
}

/// The rank's streams.
#[derive(Default)]
pub struct Lanes {
    lanes: Vec<Lane>,
    /// One full-device stream per CUDA device, without configured lanes.
    devices: Vec<CUDAStream>,
    preparation: Vec<CUDAStream>,
    modules: Vec<CUDAStream>,
    microbatches: Vec<CUDAStream>,
}

impl Lanes {
    /// Create the streams of `devices`. `event_slots` bounds each stream's
    /// outstanding submission fences; a lane bounded by `max_inflight` keeps
    /// one more than its bound.
    pub fn create(config: &WorkerConfig, devices: &[Device], event_slots: usize) -> Result<Self> {
        let cuda: Vec<DeviceId> = devices
            .iter()
            .filter_map(|device| device.cuda_index().map(DeviceId))
            .collect();
        let mut lanes = Self::default();
        if cuda.is_empty() {
            return Ok(lanes);
        }
        if !config.lanes.is_empty() {
            let [device] = cuda[..] else {
                return Err(Error::Unsupported(
                    "execution lanes require one physical device".into(),
                ));
            };
            if config.expert_microbatches > 1 && config.lanes.len() > 1 {
                return Err(Error::Unsupported(
                    "expert parallelism runs on exactly one execution lane".into(),
                ));
            }
            let counts: Vec<u32> = config
                .lanes
                .iter()
                .map(|lane| u32::try_from(lane.sm_budget).unwrap_or(u32::MAX))
                .collect();
            let slots: Vec<usize> = config
                .lanes
                .iter()
                .map(|lane| lane.max_inflight.unwrap_or(event_slots.saturating_sub(1)) + 1)
                .collect();
            let streams = CUDAStream::partition(device, &counts, &slots)?;
            lanes.lanes = config
                .lanes
                .iter()
                .cloned()
                .zip(streams)
                .map(|(config, stream)| Lane { config, stream })
                .collect();
        } else {
            for &device in &cuda {
                lanes
                    .devices
                    .push(CUDAStream::create(device, event_slots.max(1))?);
            }
        }
        for &device in &cuda {
            lanes.preparation.push(CUDAStream::create(device, 1)?);
        }
        Ok(lanes)
    }

    pub fn lanes(&self) -> &[Lane] {
        &self.lanes
    }

    /// The lane serving `kind`, or the device stream without lanes.
    pub fn stream_for(&self, kind: CallKind, device: DeviceId) -> Option<&CUDAStream> {
        if self.lanes.is_empty() {
            return self.devices.iter().find(|stream| stream.device() == device);
        }
        self.lanes
            .iter()
            .find(|lane| lane.config.call_kinds.contains(&kind))
            .map(|lane| &lane.stream)
    }

    /// The startup preparation stream of `device`.
    pub fn preparation(&self, device: DeviceId) -> Option<&CUDAStream> {
        self.preparation
            .iter()
            .find(|stream| stream.device() == device)
    }

    /// Fork the stream serving `kind` for a module that runs beside it.
    /// With lanes, exactly one lane must serve the kind.
    pub fn fork_module(&mut self, kind: CallKind, device: DeviceId) -> Result<&CUDAStream> {
        let matching: Vec<&CUDAStream> = if self.lanes.is_empty() {
            self.devices
                .iter()
                .filter(|stream| stream.device() == device)
                .collect()
        } else {
            self.lanes
                .iter()
                .filter(|lane| lane.config.call_kinds.contains(&kind))
                .map(|lane| &lane.stream)
                .collect()
        };
        let [origin] = matching[..] else {
            return Err(Error::Invalid(format!(
                "{} is served by {} execution lanes; a module stream needs exactly one",
                kind.as_str(),
                matching.len()
            )));
        };
        let stream = origin.fork()?;
        self.modules.push(stream);
        self.modules
            .last()
            .ok_or(Error::State("module stream was not registered"))
    }

    /// Fork the first lane (or device stream) for one more microbatch.
    pub fn fork_microbatch(&mut self) -> Result<&CUDAStream> {
        let origin = self
            .lanes
            .first()
            .map(|lane| &lane.stream)
            .or_else(|| self.devices.first())
            .ok_or(Error::State("a microbatch stream requires a CUDA lane"))?;
        let stream = origin.fork()?;
        self.microbatches.push(stream);
        self.microbatches
            .last()
            .ok_or(Error::State("microbatch stream was not registered"))
    }

    /// The owned stream whose handle is `handle`.
    pub fn owner(&self, handle: usize) -> Option<&CUDAStream> {
        self.streams()
            .find(|stream| stream.handle().is_ok_and(|raw| raw == handle))
    }

    /// Every owned stream, for synchronization before teardown.
    pub fn streams(&self) -> impl Iterator<Item = &CUDAStream> {
        self.preparation
            .iter()
            .chain(&self.modules)
            .chain(&self.microbatches)
            .chain(self.lanes.iter().map(|lane| &lane.stream))
            .chain(&self.devices)
    }

    /// Close in dependency order: preparation, modules, microbatches, then
    /// lanes, each newest first. Attempts every stream before returning the
    /// first failure.
    pub fn close(&mut self, aborted: bool) -> Result<()> {
        let mut failure = None;
        let mut close = |stream: &mut CUDAStream| {
            if let Err(error) = stream.close(aborted) {
                failure.get_or_insert(error);
            }
        };
        self.preparation.iter_mut().rev().for_each(&mut close);
        self.modules.iter_mut().rev().for_each(&mut close);
        self.microbatches.iter_mut().rev().for_each(&mut close);
        self.lanes
            .iter_mut()
            .rev()
            .for_each(|lane| close(&mut lane.stream));
        self.devices.iter_mut().rev().for_each(&mut close);
        failure.map_or(Ok(()), Err)
    }
}
