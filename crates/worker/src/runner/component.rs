//! A video model's components placed on a rank, and what their runners
//! are built from at startup.

use std::ops::Range;
use std::sync::Arc;

use uniserve_core::ComponentConfig;
use uniserve_cuda::{DeviceId, Stream};
use uniserve_ipc::{CallKind, MediaCall, WorkerInfo};

use crate::capacity::video::VideoCapacity;
use crate::config::placement::Capability;
use crate::config::placement::Placed;
use crate::config::{Device, WorkerConfig};
use crate::cuda_graph::{GraphPool, GraphStorage};
use crate::ffi::{Handle, Interpreter};
use crate::runner::denoiser::VideoModel;
use crate::stream::Streams;
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
    /// rank's run of `units_per_rank` units from its position among the
    /// ranks the round is `dealt` to (`DecodeRange::ranks`), or among the
    /// component's ranks in configured order when it names none.
    pub(crate) fn units(
        &self,
        dealt: &[u32],
        rank: usize,
        cursor: u32,
        count: u32,
    ) -> Result<Range<u32>> {
        units(&self.config, dealt, rank, cursor, count)
    }
}

/// See [`Component::units`]. A member of the component outside the deal
/// holds no unit.
pub(crate) fn units(
    config: &ComponentConfig,
    dealt: &[u32],
    rank: usize,
    cursor: u32,
    count: u32,
) -> Result<Range<u32>> {
    if config.distribution.is_none() {
        return Ok(cursor..cursor + count);
    }
    if !config.ranks.contains(&rank) {
        return Err(Error::Invalid(
            "media units assigned outside the component".into(),
        ));
    }
    let position = if dealt.is_empty() {
        config.ranks.iter().position(|&member| member == rank)
    } else {
        dealt.iter().position(|&member| member as usize == rank)
    };
    let Some(position) = position else {
        return Ok(cursor..cursor);
    };
    let position = position as u32;
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
    pub(crate) loaded: &'a Handle,
    pub(crate) model: &'a VideoModel,
    pub(crate) capacity: &'a VideoCapacity,
    pub(crate) info: &'a WorkerInfo,
    /// Unresolved batches the rank's launch admits at once.
    pub(crate) queue_depth: usize,
    pub(crate) config: &'a WorkerConfig,
    pub(crate) streams: &'a mut Streams,
    pub(crate) graphs: &'a mut GraphStorage,
    /// The attention backend the model loaded with.
    pub(crate) attention: String,
}

impl VideoSetup<'_> {
    /// A stream forked for one component's calls and the shared work areas
    /// of the execution contexts serialized on it: a Python `Scratch`,
    /// which grows to the largest context prepared and keeps what captured
    /// graphs address.
    pub(crate) fn stream(&mut self, kind: MediaCall, device: Device) -> Result<(Stream, Handle)> {
        let streams = &mut *self.streams;
        let forked = streams.fork_module(CallKind::Media(kind), cuda_device(device)?)?;
        let stream = forked.stream()?.clone();
        let scratch = Handle::new(
            Interpreter::import("uniserve.bindings.context.Scratch")?.call_values(&[])?,
        )?;
        Ok((stream, scratch))
    }

    /// The graph pool on `device` under graph storage `key`; `None` on the
    /// host and when graphs are off.
    pub(crate) fn graph_pool(
        &mut self,
        key: usize,
        device: Device,
    ) -> Result<Option<Arc<GraphPool>>> {
        let capture = self.config.graph_policy != "off" && self.config.device.is_cuda();
        let Device::Cuda(device) = device else {
            return Ok(None);
        };
        if !capture {
            return Ok(None);
        }
        let create = Interpreter::import("uniserve.cuda_graph.pool")?;
        let pools = crate::cuda_graph::reserve_pools(self.graphs, &create, key, &[device], None)?;
        Ok(pools.into_iter().next())
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

/// The CUDA device a video component computes on.
pub(crate) fn cuda_device(device: Device) -> Result<DeviceId> {
    device
        .cuda()
        .ok_or_else(|| Error::Unsupported("video components compute on CUDA devices".into()))
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
            weights: Default::default(),
        }
    }

    #[test]
    fn a_unit_ring_deals_each_round_from_its_first_member() {
        let ring = config(&[4, 5, 6, 7], true, 1);
        // A full round of four units starting at unit 8.
        assert_eq!(units(&ring, &[], 5, 8, 4).unwrap(), 9..10);
        // A short round leaves the last members without a unit.
        assert!(units(&ring, &[], 6, 12, 2).unwrap().is_empty());
        assert_eq!(units(&ring, &[], 4, 12, 2).unwrap(), 12..13);
        // Members take consecutive runs of `units_per_rank`.
        let pairs = config(&[0, 1], true, 2);
        assert_eq!(units(&pairs, &[], 1, 0, 3).unwrap(), 2..3);
        // An undistributed component computes the whole round.
        assert_eq!(
            units(&config(&[0, 1], false, 1), &[], 1, 3, 4).unwrap(),
            3..7
        );
        assert!(units(&ring, &[], 0, 0, 4).is_err());
    }

    #[test]
    fn a_round_that_names_its_ranks_deals_in_their_order() {
        let ring = config(&[4, 5, 6, 7], true, 1);
        // The round's units go to ranks 6 and 4, in that order.
        assert_eq!(units(&ring, &[6, 4], 6, 8, 2).unwrap(), 8..9);
        assert_eq!(units(&ring, &[6, 4], 4, 8, 2).unwrap(), 9..10);
        // A member the round leaves out holds none.
        assert!(units(&ring, &[6, 4], 5, 8, 2).unwrap().is_empty());
        let pairs = config(&[0, 1], true, 2);
        assert_eq!(units(&pairs, &[1, 0], 1, 0, 3).unwrap(), 0..2);
        assert_eq!(units(&pairs, &[1, 0], 0, 0, 3).unwrap(), 2..3);
        assert!(units(&ring, &[4], 0, 0, 1).is_err());
    }
}
