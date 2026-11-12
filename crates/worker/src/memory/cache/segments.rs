//! The segments a device cache's value planes are made of.
//!
//! Each value plane is made of segments along the units, each column's part of a segment one
//! allocation, mapped back to back so the plane keeps its layout and addresses. An epoch's own
//! units come first, split into at most [`SEGMENTS`] segments; an epoch that succeeds another on
//! the same devices maps, after them, the segments its predecessor holds, read-only. The last
//! segments are given back to the device when a successor epoch needs their storage, or all of
//! them while the deployment sleeps, and created again at their addresses afterwards; the engine
//! allocates none of their units meanwhile.

/// The most segments an epoch splits its own units into: the granularity in which the cache
/// gives storage back, an eighth of it.
pub const SEGMENTS: u32 = 8;

/// One segment of a cache's value planes, as the rank that maps it takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneSegment {
    /// The segment: the epoch that created its storage, its ordinal there and its units.
    pub segment: ipc::CacheSegment,
    /// Whether the predecessor epoch writes the segment: it is mapped read-only.
    pub inherited: bool,
    /// Whether the rank creates the segment when its storage does not exist; one that is not
    /// created is left unmapped, as a segment given back.
    pub create: bool,
}

/// How a cache of `num_units` units splits its value planes into segments whose every column's
/// part is whole allocation granules of `granularity` bytes, with planes of `plane_bytes` per
/// unit: the segments' count and the units each holds. `None` when the cache is too small to
/// split, or not a multiple of the split ([`segmented_units`] rounds it).
pub fn segment_layout(num_units: u32, plane_bytes: u64, granularity: u64) -> Option<(u32, u32)> {
    let granule = granule_units(plane_bytes, granularity)?;
    let segments = SEGMENTS.min(num_units / granule);
    if segments == 0 {
        return None;
    }

    let units = num_units / segments / granule * granule;
    (units * segments == num_units).then_some((segments, units))
}

/// `num_units` rounded down to the largest cache whose planes split into segments
/// ([`segment_layout`]); `num_units` itself when it is too small to split.
pub fn segmented_units(num_units: u32, plane_bytes: u64, granularity: u64) -> u32 {
    let Some(granule) = granule_units(plane_bytes, granularity) else {
        return num_units;
    };
    let segments = SEGMENTS.min(num_units / granule);
    if segments == 0 {
        return num_units;
    }
    num_units / segments / granule * granule * segments
}

/// The fewest units of at least `num_units` that split into the segments [`segmented_units`]
/// makes.
pub fn segmented_units_above(num_units: u32, plane_bytes: u64, granularity: u64) -> u32 {
    let Some(granule) = granule_units(plane_bytes, granularity) else {
        return num_units;
    };
    let segments = SEGMENTS.min(num_units / granule);
    if segments == 0 {
        return num_units;
    }

    let step = granule * segments;
    num_units.div_ceil(step) * step
}

/// The fewest units whose planes of `plane_bytes` fill whole granules.
fn granule_units(plane_bytes: u64, granularity: u64) -> Option<u32> {
    if plane_bytes == 0 || granularity == 0 {
        return None;
    }

    let (mut a, mut b) = (plane_bytes, granularity);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    u32::try_from(granularity / a).ok()
}

/// The segments of a cache of `num_units` units that `epoch` creates whole: its units split
/// into at most [`SEGMENTS`] equal segments ([`segment_layout`]), numbered from zero. `None`
/// when the cache does not split.
pub fn own(
    epoch: u64,
    num_units: u32,
    plane_bytes: u64,
    granularity: u64,
) -> Option<Vec<PlaneSegment>> {
    let (count, units) = segment_layout(num_units, plane_bytes, granularity)?;

    Some(
        (0..count)
            .map(|ordinal| PlaneSegment {
                segment: ipc::CacheSegment {
                    epoch,
                    ordinal,
                    units,
                },
                inherited: false,
                create: true,
            })
            .collect(),
    )
}

/// The exact segments a rank's planes were made of: as the serving rank of a standby's scope
/// holds them, or as an epoch's rank held them before its relaunch. None is inherited, and the
/// segments past the ones held are left unmapped.
pub fn adopted(table: &ipc::CacheTable) -> Vec<PlaneSegment> {
    table
        .segments
        .iter()
        .enumerate()
        .map(|(index, segment)| PlaneSegment {
            segment: *segment,
            inherited: false,
            create: (index as u32) < table.held,
        })
        .collect()
}

/// The units `segments` hold.
pub fn units(segments: &[PlaneSegment]) -> u32 {
    segments.iter().map(|part| part.segment.units).sum()
}

/// The units of the segments the predecessor writes, `[start, end)`: from the first inherited
/// segment's first unit to the last one's end. Empty when none is inherited.
pub fn inherited(segments: &[PlaneSegment]) -> std::ops::Range<u32> {
    let mut range = 0..0;
    let mut first = 0;

    for part in segments {
        let end = first + part.segment.units;
        if part.inherited {
            range = if range.is_empty() {
                first..end
            } else {
                range.start..end
            };
        }
        first = end;
    }
    range
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plane of a thirty-second of a granule fills a granule every thirty-two units. A cache
    /// fitted to a capacity keeps what it was asked for, and one fitted to the storage rounds
    /// into it.
    #[test]
    fn a_capacity_rounds_up_to_whole_segments_and_a_fit_rounds_down() {
        let (plane, granule) = (1 << 16, 1 << 21);

        assert_eq!(segmented_units(640, plane, granule), 512);
        assert_eq!(segmented_units_above(640, plane, granule), 768);
        assert_eq!(segmented_units_above(768, plane, granule), 768);
        // Too few units for a segment of whole granules stay as they are.
        assert_eq!(segmented_units_above(7, plane, granule), 7);
        assert_eq!(segment_layout(7, plane, granule), None);
        // A cache that is not a multiple of the split does not split until it is rounded.
        assert_eq!(segment_layout(640, plane, granule), None);
        assert_eq!(segment_layout(512, plane, granule), Some((8, 64)));
    }

    /// An epoch's own segments are equal and numbered from zero; adopted ones are exactly the
    /// table's, created as far as they are held.
    #[test]
    fn own_segments_split_the_units_and_adopted_ones_follow_the_table() {
        let segments = own(7, 512, 1 << 16, 1 << 21).expect("512 units split into eight");
        assert_eq!(segments.len(), 8);
        assert_eq!(units(&segments), 512);
        assert!(
            segments
                .iter()
                .enumerate()
                .all(|(ordinal, part)| part.segment.epoch == 7
                    && part.segment.ordinal == ordinal as u32
                    && part.create
                    && !part.inherited)
        );
        assert!(inherited(&segments).is_empty());

        let table = ipc::CacheTable {
            segments: segments.iter().map(|part| part.segment).collect(),
            held: 3,
        };
        let taken = adopted(&table);
        assert_eq!(taken.iter().filter(|part| part.create).count(), 3);
    }

    /// The inherited units are the range the predecessor's segments span.
    #[test]
    fn the_inherited_units_span_the_predecessors_segments() {
        let part = |ordinal: u32, inherited: bool| PlaneSegment {
            segment: ipc::CacheSegment {
                epoch: u64::from(inherited),
                ordinal,
                units: 64,
            },
            inherited,
            create: !inherited,
        };

        let segments = [part(0, false), part(1, false), part(0, true), part(1, true)];
        assert_eq!(inherited(&segments), 128..256);
    }
}
