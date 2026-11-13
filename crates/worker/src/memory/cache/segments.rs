//! The segments a device cache's value planes are made of.
//!
//! Each value plane is made of segments along the units, each column's part of a segment one
//! allocation, mapped back to back so the plane keeps its layout and addresses. The segments are
//! the memory of the cache's buffer pool region: the epoch that creates the region splits its
//! units into at most [`SEGMENTS`] segments ([`own`]), and every other epoch that names the
//! region maps the same segments, as the region's segment table states them ([`of_table`]). The
//! last segments are given back to the device when a region of another layout needs their
//! storage, or all of them while the deployment sleeps, and created again at their addresses
//! afterwards; the buffer pool gives none of their units to a block meanwhile.

use crate::{Error, Result};

/// The most segments an epoch splits its own units into: the granularity in which the cache
/// gives storage back, an eighth of it.
pub const SEGMENTS: u32 = 8;

/// One segment of a cache's value planes, as the rank that maps it takes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlaneSegment {
    /// The segment: the epoch that created its storage, its ordinal there and its units.
    pub segment: ipc::CacheSegment,
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
                create: true,
            })
            .collect(),
    )
}

/// The segments of a region's memory as its segment table states them
/// (`storage::runtime::Opened::segments`): the segments an epoch's ranks created, which a rank
/// of every epoch that names the region maps at the same units. Every rank that maps the
/// region holds its parts of each segment of the table; a part the Storage Server does not hold
/// is created.
///
/// # Errors
///
/// A table that does not lay its segments back to back from unit 0, which the planes' layout
/// requires.
pub fn of_table(table: &[(storage::Segment, u32, u32)]) -> Result<Vec<PlaneSegment>> {
    let mut next = 0;

    table
        .iter()
        .map(|(segment, first, units)| {
            if *first != next || *units == 0 {
                return Err(Error::Invalid(format!(
                    "the region's segment {}/{} holds {units} units from unit {first}, and the \
                     planes' segments lie back to back from unit 0: the next is at unit {next}",
                    segment.epoch, segment.ordinal
                )));
            }
            next += units;

            Ok(PlaneSegment {
                segment: ipc::CacheSegment {
                    epoch: segment.epoch,
                    ordinal: segment.ordinal,
                    units: *units,
                },
                create: true,
            })
        })
        .collect()
}

/// The units `segments` hold.
pub fn units(segments: &[PlaneSegment]) -> u32 {
    segments.iter().map(|part| part.segment.units).sum()
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

    /// An epoch's own segments are equal and numbered from zero, and created where missing.
    #[test]
    fn own_segments_split_the_units() {
        let segments = own(7, 512, 1 << 16, 1 << 21).expect("512 units split into eight");
        assert_eq!(segments.len(), 8);
        assert_eq!(units(&segments), 512);
        assert!(
            segments
                .iter()
                .enumerate()
                .all(|(ordinal, part)| part.segment.epoch == 7
                    && part.segment.ordinal == ordinal as u32
                    && part.create)
        );
    }

    /// A rank of another epoch that names the region lays its planes out from the region's
    /// segment table: the segments the creating epoch's ranks made, each with the identity that
    /// keys its parts, so both epochs map the same storages.
    #[test]
    fn a_region_s_table_gives_the_segments_its_creating_epoch_made() -> Result<()> {
        let made = own(3, 512, 1 << 16, 1 << 21).expect("512 units split into eight");
        let mut first = 0;
        let table: storage::runtime::Segments = made
            .iter()
            .map(|part| {
                let entry = (
                    storage::Segment {
                        epoch: part.segment.epoch,
                        ordinal: part.segment.ordinal,
                    },
                    first,
                    part.segment.units,
                );
                first += part.segment.units;
                entry
            })
            .collect();

        assert_eq!(of_table(&table)?, made);
        assert!(of_table(&[])?.is_empty());

        // A table whose segments do not lie back to back from unit 0 lays out no planes.
        let refused = of_table(&[table[0], table[2]])
            .expect_err("a table with a gap")
            .to_string();
        assert!(refused.contains("segment 3/2"), "{refused}");
        assert!(of_table(&table[1..]).is_err());
        Ok(())
    }
}
