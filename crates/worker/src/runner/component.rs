//! A video model's components placed on a rank, and what their runners
//! are built from at startup.

use std::ops::Range;
use std::sync::Arc;

use uniserve_core::ComponentConfig;
use uniserve_cuda::Stream;
use uniserve_ipc::{CallKind, MediaCall, WorkerInfo};

use crate::capacity::video::VideoCapacity;
use crate::config::WorkerConfig;
use crate::cuda_graph::{GraphPool, GraphStorage};
use crate::ffi::{Device, Handle};
use crate::lane::Lanes;
use crate::model::components::Capability;
use crate::model::interface::Interface;
use crate::model::registry::Placed;
use crate::runner::denoiser::VideoModel;
use crate::{Error, Result};

/// A placed video component on this rank.
pub(crate) struct Component<R> {
    pub(crate) name: String,
    pub(crate) config: ComponentConfig,
    pub(crate) device: Device,
    pub(crate) runner: R,
}

impl<R> Component<R> {
    /// This rank's media units of a round covering units `[cursor, cursor +
    /// count)`: all of them for an undistributed component, otherwise the
    /// rank's run of `units_per_rank` units from its position.
    pub(crate) fn units(&self, rank: usize, cursor: u32, count: u32) -> Result<Range<u32>> {
        units(&self.config, rank, cursor, count)
    }
}

/// See [`Component::units`].
pub(crate) fn units(
    config: &ComponentConfig,
    rank: usize,
    cursor: u32,
    count: u32,
) -> Result<Range<u32>> {
    if config.distribution.is_none() {
        return Ok(cursor..cursor + count);
    }
    let position = config
        .ranks
        .iter()
        .position(|&member| member == rank)
        .ok_or_else(|| Error::Invalid("media units assigned outside the component".into()))?
        as u32;
    let per_rank = config.units_per_rank as u32;
    let first = (position * per_rank).min(count);
    let last = (position * per_rank + per_rank).min(count);
    Ok(cursor + first..cursor + last)
}

/// Graph storage keys of the video runners.
pub(crate) const DENOISER_GRAPHS: usize = 1000;
pub(crate) const DECODER_GRAPHS: usize = 1001;
pub(crate) const TEXT_GRAPHS: usize = 1002;
pub(crate) const AUDIO_GRAPHS: usize = 1003;

/// What a video model's runners on a rank are built from at startup.
pub(crate) struct VideoSetup<'a> {
    pub(crate) interface: &'a Interface,
    pub(crate) loaded: &'a Handle,
    pub(crate) model: &'a VideoModel,
    pub(crate) capacity: &'a VideoCapacity,
    pub(crate) info: &'a WorkerInfo,
    pub(crate) config: &'a WorkerConfig,
    pub(crate) lanes: &'a mut Lanes,
    pub(crate) graphs: &'a mut GraphStorage<String, Handle>,
    /// The attention backend the model loaded with.
    pub(crate) attention: String,
}

impl VideoSetup<'_> {
    /// A stream forked for one component's calls and the shared work areas
    /// of the execution contexts serialized on it: a Python `Scratch`,
    /// which grows to the largest context prepared and keeps what captured
    /// graphs address.
    pub(crate) fn stream(&mut self, kind: MediaCall, device: Device) -> Result<(Stream, Handle)> {
        let lanes = &mut *self.lanes;
        let Device::Cuda(index) = device else {
            return Err(Error::Unsupported(
                "video components compute on CUDA devices".into(),
            ));
        };
        let forked = lanes.fork_module(CallKind::Media(kind), uniserve_cuda::DeviceId(index))?;
        let stream = forked.stream()?.clone();
        let scratch = Handle::new(
            self.interface
                .import("uniserve.bindings.context.Scratch")?
                .call_values(&[])?,
        )?;
        Ok((stream, scratch))
    }

    /// The graph pool of `owner` on `device` under graph storage `key`;
    /// `None` on the host and when graphs are off.
    pub(crate) fn graph_pool(
        &mut self,
        key: usize,
        owner: &str,
        device: Device,
    ) -> Result<Option<Arc<GraphPool<Handle>>>> {
        let capture = self.config.graph_policy != "off" && self.config.device.is_cuda();
        let Device::Cuda(index) = device else {
            return Ok(None);
        };
        if !capture {
            return Ok(None);
        }
        let create = self.interface.function("graphs.pool")?;
        let pools = crate::cuda_graph::reserve_pools(
            self.graphs,
            &create,
            key,
            owner.to_owned(),
            &[index],
            None,
        )?;
        Ok(pools.into_iter().next())
    }
}

/// The device a placed component computes on.
pub(crate) fn placed_device(placed: &Placed) -> Device {
    match placed.device {
        crate::config::Device::Cuda(index) => Device::Cuda(index),
        crate::config::Device::Cpu => Device::Cpu,
    }
}

/// The module path of a placed component's call of a capability.
pub(crate) fn call_path(placed: &Placed, capability: Capability) -> Result<&str> {
    placed
        .calls
        .iter()
        .find(|call| call.is(capability))
        .map(|call| call.path.as_str())
        .ok_or_else(|| {
            Error::Invariant(format!(
                "component {} calls no {}",
                placed.name,
                capability.tag()
            ))
        })
}

/// The CUDA ordinal of a component's device.
pub(crate) fn device_index(device: Device) -> Result<i32> {
    match device {
        Device::Cuda(index) => Ok(index),
        Device::Cpu | Device::CudaHost => Err(Error::Unsupported(
            "video components compute on CUDA devices".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use uniserve_core::{ComponentDistribution, ParallelConfig};

    use super::*;

    fn config(ranks: &[usize], distribution: bool, units_per_rank: usize) -> ComponentConfig {
        ComponentConfig {
            ranks: ranks.to_vec(),
            parallel_config: ParallelConfig::default(),
            distribution: distribution.then_some(ComponentDistribution::TemporalUnits),
            units_per_rank,
        }
    }

    #[test]
    fn a_unit_ring_deals_each_round_from_its_first_member() {
        let ring = config(&[4, 5, 6, 7], true, 1);
        // A full round of four units starting at unit 8.
        assert_eq!(units(&ring, 5, 8, 4).unwrap(), 9..10);
        // A short round leaves the last members without a unit.
        assert!(units(&ring, 6, 12, 2).unwrap().is_empty());
        assert_eq!(units(&ring, 4, 12, 2).unwrap(), 12..13);
        // Members take consecutive runs of `units_per_rank`.
        assert_eq!(units(&config(&[0, 1], true, 2), 1, 0, 3).unwrap(), 2..3);
        // An undistributed component computes the whole round.
        assert_eq!(units(&config(&[0, 1], false, 1), 1, 3, 4).unwrap(), 3..7);
        assert!(units(&ring, 0, 0, 4).is_err());
    }
}
