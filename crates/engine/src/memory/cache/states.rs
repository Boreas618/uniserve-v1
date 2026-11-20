//! The rank's memory of its scheduler's state region: the units recurrent-state checkpoints are
//! kept in (`specs/engine-worker-port.md` §1.6, `specs/buffer-pool.md` §14.1).
//!
//! A unit is one checkpoint of every recurrent layer the rank holds: the layout's
//! [`CacheLayout::checkpoint_planes`] planes of its plane bytes, back to back
//! ([`CacheLayout::state_bytes`]), each layer's fields from plane boundaries
//! ([`CacheLayout::state_planes`]). The units are one storage, `prefix_cache.state.values`,
//! `[units, checkpoint_planes, plane_bytes]` bytes, unit `u` at `u · state_bytes`. Its name keys
//! it at the Storage Server, where every epoch that names the region finds it, so it stays as it
//! is whatever this module is called.
//!
//! On a device whose memory splits ([`super::segments`]) the storage is made of the state
//! region's own segments, each the fewest units whose bytes are whole allocation granules, keyed
//! by the region's segment table as the KV planes' segments are by theirs: two epochs of one
//! layout address the same checkpoints, the last segments are given back
//! ([`StateUnits::release_segments`]) and created again at their addresses
//! ([`StateUnits::restore_segments`]) by the functions the planes use. No unit is reset: a keep
//! writes a whole checkpoint before any restore reads it, and the buffer pool gives a unit to a
//! block only while no batch may still read it.
//!
//! No captured graph addresses the units. The state copies read and write them eagerly around a
//! forward, through the snapshot of [`StateMemory`](crate::memory::state::StateMemory), and the
//! host tier copies a unit whole to and from its host page.

use std::sync::Arc;

use platform::DeviceId;
use tensor::{DType, Tensor};

use super::info::{CachePlaneRun, CacheSegment};
use super::layout::CacheLayout;
use super::planes::{self, Segments};
use super::segments::{self, PlaneSegment};
use crate::memory::allocation::Storage;
use crate::memory::exclusive::{ExclusiveConfig, ExclusiveStorage};
use crate::memory::lend;
use crate::worker::{Error, Result};

/// The storage's name.
const STATES: &str = "prefix_cache.state.values";

/// The units of a state region on one rank: on a device, or none for a cache that holds no
/// device, whose tensor is a meta tensor.
pub(crate) struct StateUnits {
    /// The bytes of one unit: one checkpoint of every recurrent layer the rank holds.
    unit_bytes: u64,
    /// The planes of one unit, of `unit_bytes / planes` bytes each.
    planes: u32,
    /// The units of the region's numbering.
    units: u32,
    /// The device the units are on, with the other local devices that address them; `None` for
    /// units that hold no device.
    device: Option<(DeviceId, Vec<DeviceId>)>,
    /// The units' storage; `None` for units that hold no device.
    storage: Option<Arc<Storage>>,
    /// The segments the storage is made of, when it splits into them.
    segments: Option<Segments>,
    /// Where the Storage Server retains every unit's bytes, run by run; empty for storage local
    /// to the process and for units that hold no device.
    runs: Vec<CachePlaneRun>,
}

impl StateUnits {
    /// The units of a state region of `units` units for a cache of `layout`, holding no device.
    ///
    /// # Errors
    ///
    /// A layout that keeps no state group, or a region of no unit.
    pub(crate) fn meta(layout: &CacheLayout, units: u32) -> Result<Self> {
        let unit_bytes = layout.state_bytes();
        if unit_bytes == 0 || units == 0 {
            return Err(Error::Invalid(format!(
                "a state region of {units} units of {unit_bytes} bytes holds no checkpoint"
            )));
        }

        Ok(Self {
            unit_bytes,
            planes: layout.checkpoint_planes(),
            units,
            device: None,
            storage: None,
            segments: None,
            runs: Vec::new(),
        })
    }

    /// Takes the units of a state region of `units` units for a cache of `layout` on `device`
    /// from `owner`, addressed by `device` and each of `peers`, in `parts` segments when the region splits, and one storage otherwise (a
    /// rank of no region's, or local storage). What exists keeps its contents; what is created
    /// holds unspecified bytes, which no restore reads before a keep wrote them. The storage is
    /// keyed by the region's memory: the epoch of the first of `parts`, or the rank's own when
    /// the units do not split.
    ///
    /// # Errors
    ///
    /// As [`Self::meta`]; segments that do not hold `units` units, or a storage `owner` refuses.
    pub(crate) fn open(
        layout: &CacheLayout,
        units: u32,
        parts: Option<&[PlaneSegment]>,
        (device, peers): (DeviceId, &[DeviceId]),
        owner: &dyn ExclusiveStorage,
    ) -> Result<Self> {
        let mut states = Self::meta(layout, units)?;
        if parts.is_some_and(|parts| segments::units(parts) != units) {
            return Err(Error::Invalid(
                "a state region's segments do not hold its units".into(),
            ));
        }
        let peers = peers.to_vec();
        let mut runs = Vec::new();

        let (storage, split) = match parts {
            Some(parts) => {
                // One storage of one column, a checkpoint a unit.
                let storages = [(STATES.to_owned(), vec![states.unit_bytes])];
                let (storages, segments, _) =
                    planes::take_split(&storages, parts, (device, &peers), owner, &mut runs)?;
                let storage = storages.into_iter().next().ok_or_else(|| {
                    Error::Invariant("the state region's storage is missing".into())
                })?;
                (storage, Some(segments))
            }
            None => {
                let config = ExclusiveConfig {
                    name: STATES,
                    device,
                    bytes: (u64::from(units) * states.unit_bytes) as usize,
                    access: &peers,
                    epoch: None,
                };
                if let Some((key, gpu)) = owner.key(&config)? {
                    runs.push(CachePlaneRun {
                        field: format!("{STATES}/0"),
                        key,
                        gpu,
                        first: 0,
                        count: units,
                        offset: 0,
                        bytes: states.unit_bytes,
                    });
                }
                let (storage, _) = owner
                    .exclusive(std::slice::from_ref(&config))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        Error::Invariant("the state region's storage is missing".into())
                    })?;
                (storage, None)
            }
        };

        states.device = Some((device, peers));
        states.storage = Some(storage);
        states.segments = split;
        states.runs = runs;
        Ok(states)
    }

    /// The units of the region's numbering.
    pub(crate) fn units(&self) -> u32 {
        self.units
    }

    /// The bytes one unit takes, which a segment gives back per unit.
    pub(crate) fn unit_bytes(&self) -> u64 {
        self.unit_bytes
    }

    /// Where the Storage Server retains every unit's bytes, run by run: what the rank places
    /// for the Transfer Server (`storage::protocol::Request::Place`).
    pub(crate) fn runs(&self) -> &[CachePlaneRun] {
        &self.runs
    }

    /// The segments the units are made of, in unit order; `None` for units that do not split.
    pub(crate) fn segments(&self) -> Option<&[CacheSegment]> {
        self.segments.as_ref().map(Segments::segments)
    }

    /// The segments held, `[0, held)`; zero for units that do not split.
    pub(crate) fn held_segments(&self) -> u32 {
        self.segments.as_ref().map_or(0, Segments::held)
    }

    /// The bytes the units map on the device; a sleeping rank's segments are not in it.
    pub(crate) fn mapped_bytes(&self) -> u64 {
        let Some(storage) = &self.storage else {
            return 0;
        };
        let released = self.segments.as_ref().map_or(0, Segments::released_bytes);
        (storage.bytes() as u64).saturating_sub(released)
    }

    /// Refuses a unit outside the region's numbering.
    fn require(&self, unit: u32) -> Result<()> {
        if unit >= self.units {
            return Err(Error::Invalid(format!(
                "state unit {unit} is outside the state region's units 0..{}",
                self.units
            )));
        }
        Ok(())
    }

    /// The rows of the units' tensor ([`Self::tensor`]) unit `unit`'s planes lie in, in plane
    /// order.
    ///
    /// # Errors
    ///
    /// A unit outside the region's numbering.
    pub(crate) fn rows(&self, unit: u32) -> Result<std::ops::Range<i64>> {
        self.require(unit)?;
        let first = i64::from(unit) * i64::from(self.planes);
        Ok(first..first + i64::from(self.planes))
    }

    /// The units as `[units · checkpoint_planes, plane_bytes]` bytes, plane `p` of unit `u` at
    /// row `u · checkpoint_planes + p`: what the state copies gather from and scatter into.
    ///
    /// # Errors
    ///
    /// A tensor the units' storage cannot lend.
    pub(crate) fn tensor(&self) -> Result<Tensor> {
        let plane_bytes = self.unit_bytes / u64::from(self.planes.max(1));
        let shape = [
            self.units as usize * self.planes as usize,
            plane_bytes as usize,
        ];
        match &self.storage {
            // SAFETY: the units own the storage for as long as they live; the state copies,
            // issued on the stream of the runner that holds the cache, and the host tier's copies,
            // which touch only units the buffer pool gave them, address it.
            Some(storage) => unsafe { storage.external(0, shape.as_slice(), DType::U8) },
            // SAFETY: a meta tensor has no data and its address is never read.
            None => unsafe { lend(0, shape.as_slice(), DType::U8, None) },
        }
    }

    /// The device address of unit `unit`, whose `unit_bytes` lie contiguously from it.
    ///
    /// # Errors
    ///
    /// Units that hold no device, or a unit outside the region's numbering.
    pub(crate) fn address(&self, unit: u32) -> Result<u64> {
        self.require(unit)?;
        let storage = self.storage.as_ref().ok_or(Error::State(
            "the state region holds no device: its units have no address",
        ))?;
        Ok(storage.address() + u64::from(unit) * self.unit_bytes)
    }

    /// Gives back the storage of the segments from `first` on through `owner`, which retained
    /// them: the head withdrew their units, and no device work addresses them.
    ///
    /// # Errors
    ///
    /// For units that are not split, or when `owner` fails.
    pub(crate) fn release_segments(
        &mut self,
        first: u32,
        owner: &dyn ExclusiveStorage,
    ) -> Result<()> {
        let (Some((device, peers)), Some(storage), Some(segments)) =
            (&self.device, &self.storage, &mut self.segments)
        else {
            return Err(Error::Invalid(format!(
                "the state region of {} units is not split into segments, which is what gives \
                 storage back",
                self.units
            )));
        };
        planes::release_split(
            (*device, peers),
            segments,
            std::slice::from_ref(storage),
            first,
            owner,
        )
    }

    /// Creates the segments given back again at their addresses, the first `limit` of them or
    /// all, through `owner`. Their bytes are unspecified, as a created unit's are.
    ///
    /// # Errors
    ///
    /// For units that are not split, or when `owner` fails; the units are then as asleep as they
    /// were.
    pub(crate) fn restore_segments(
        &mut self,
        owner: &dyn ExclusiveStorage,
        limit: Option<u32>,
    ) -> Result<()> {
        let (Some((device, peers)), Some(storage), Some(segments)) =
            (&self.device, &self.storage, &mut self.segments)
        else {
            return Err(Error::Invalid(format!(
                "the state region of {} units is not split into segments, which is what gives \
                 storage back",
                self.units
            )));
        };
        planes::restore_split(
            (*device, peers),
            segments,
            std::slice::from_ref(storage),
            (owner, limit),
            |_, _| Ok(()),
        )
    }
}

#[cfg(test)]
mod tests {
    use nn::family::{StateField, StateShape};
    use proto::request::PrefixCacheDtype;

    use super::*;

    /// Two attention layers of 4-token pages of one 4-wide BF16 head (32-byte planes) beside two
    /// recurrent layers of a 40-byte state: two planes a layer, four a checkpoint.
    fn layout() -> CacheLayout {
        let state = StateShape {
            layers: 2,
            fields: vec![StateField {
                name: "state",
                dtype: DType::U8,
                dims: vec![40],
                stepwise: false,
                rows: nn::family::StateRows::Whole,
            }],
        };
        CacheLayout::of(
            4,
            &nn::family::CacheField::keys_and_values(2, 1, 4),
            PrefixCacheDtype::BFloat16,
        )
        .and_then(|layout| layout.with_state(&[state]))
        .expect("a hybrid layout")
    }

    /// A unit is one checkpoint's planes, and the units lie one after another: unit `u`'s planes
    /// are the rows from `u · planes`, and a unit past the numbering is refused.
    #[test]
    fn a_unit_is_one_checkpoints_planes() -> Result<()> {
        let layout = layout();
        assert_eq!(layout.checkpoint_planes(), 4);

        let states = StateUnits::meta(&layout, 3)?;
        assert_eq!(states.unit_bytes(), 4 * layout.plane_bytes());
        assert_eq!(states.rows(0)?, 0..4);
        assert_eq!(states.rows(2)?, 8..12);
        assert!(states.rows(3).is_err());
        assert_eq!(
            states.tensor()?.dims(),
            &[12, layout.plane_bytes() as usize][..]
        );

        // Units that hold no device have no address and nothing mapped.
        assert!(states.address(0).is_err());
        assert_eq!(states.mapped_bytes(), 0);
        Ok(())
    }

    /// A layout without a state group, or a region of no unit, holds no checkpoint.
    #[test]
    fn a_state_region_needs_a_state_group_and_a_unit() {
        let dense = CacheLayout::of(
            4,
            &nn::family::CacheField::keys_and_values(2, 1, 4),
            PrefixCacheDtype::BFloat16,
        )
        .expect("a dense layout");
        assert!(StateUnits::meta(&dense, 3).is_err());
        assert!(StateUnits::meta(&layout(), 0).is_err());
    }
}
