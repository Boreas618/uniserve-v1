//! Room in a GPU's quota that a starting rank asks the epoch that serves beside it to give.
//!
//! An epoch update's target starts beside the serving epoch, and both divide one quota per GPU
//! (`docs/admin.md`, "The KV cache's storage and the weights"). Where the target's start takes
//! quota it measured exactly what it lacks: a weight publication the quota refuses, the
//! division's fewest bytes ([`fewest`]), and the warm process its own segments do not cover. There
//! it asks through [`RoomAsk`]: the daemon has the serving engines give back the tail of their KV
//! caches until each named GPU's quota holds the stated bytes beside everything it reserves,
//! and the ask returns once they did, or with why they could not.
//!
//! The bytes of an ask are absolute, as `ResizeCache { MakeRoom }` states them: what the quota
//! must have room for, not what it lacks. A rank turns what it lacks into that figure with
//! [`wanted`].

/// One GPU's room an ask names: the GPU's identity (its UUID's bytes) and the bytes its quota
/// must hold beside everything it reserves.
pub(crate) type DeviceRoom = ([u8; 16], u64);

/// The asks of a start for room in its GPUs' quotas.
pub(crate) trait RoomAsk: Send + Sync {
    /// Asks for `devices`'s room and waits until it is made.
    ///
    /// # Errors
    ///
    /// Why the room was not made: the serving engines could not give it back, the daemon does not
    /// make room for this process (one of an interface version before `proto::ROOM`), or the
    /// process ends first. The start fails with it, as it fails for want of room otherwise.
    fn ask(&self, devices: &[DeviceRoom]) -> Result<(), String>;
}

/// No room is asked of anyone: a start of a test with no serving epoch beside it to give any,
/// whose want of room fails it.
#[cfg(test)]
pub(crate) struct NoRoom;

#[cfg(test)]
impl RoomAsk for NoRoom {
    fn ask(&self, _: &[DeviceRoom]) -> Result<(), String> {
        Err("no serving epoch is asked to make room for this start".to_owned())
    }
}

/// The room to ask of a GPU whose quota leaves `room` bytes now (the lesser of the quota's room
/// and the device's free bytes, as the serving engines measure it) for a start that lacks
/// `lacking` bytes: the quota must then hold both.
pub(crate) fn wanted(room: u64, lacking: u64) -> u64 {
    room.saturating_add(lacking)
}

/// The fewest bytes a starting text rank's division takes of what its GPU's quota leaves now
/// (`ModelRunner::divide_unreserved`), the rank's process aside: the fixed `regions`, the
/// `slots` request slots' storages (`slot_bytes`), and the larger of the cache's fewest units
/// (`least` units of `unit_bytes`) and what its first segment takes beside the planes made for
/// every one of the `units` it numbers (`unsplit_bytes` a unit, `segment_bytes` a segment), with
/// the state region's first segment (`state_segment_bytes`, zero for a model of no state).
///
/// The units are what the rank numbers alone, before its host group agrees on the fewest of its
/// ranks', so the figure bounds the agreed division from above.
pub(crate) fn fewest(
    regions: &[(&str, u64)],
    slot_bytes: u64,
    (least, unit_bytes): (u32, u64),
    (units, unsplit_bytes, segment_bytes): (u32, u64, u64),
    state_segment_bytes: u64,
) -> u64 {
    let fixed = regions
        .iter()
        .fold(0_u64, |sum, (_, bytes)| sum.saturating_add(*bytes))
        .saturating_add(slot_bytes);
    let first = u64::from(units)
        .saturating_mul(unsplit_bytes)
        .saturating_add(segment_bytes)
        .saturating_add(state_segment_bytes);
    let least = u64::from(least).saturating_mul(unit_bytes);
    fixed.saturating_add(first.max(least))
}

/// The bytes by which a rank's warm process passes what its GPU's quota holds once its own
/// segments past its fewest are given back: `excess` (`division::excess`) less `own`, those
/// segments' bytes. Zero when they cover it.
pub(crate) fn warm_lacking(excess: u64, own: u64) -> u64 {
    excess.saturating_sub(own)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A division needs its fixed regions and slots, then whichever of its fewest units and its
    /// first segment with the planes of every numbered unit is larger.
    #[test]
    fn the_fewest_bytes_hold_the_fixed_regions_and_the_larger_cache_floor() {
        let regions = [("graph pool", 3_000), ("scratch pool", 500)];

        // The planes of 100 units at 10 bytes and a segment of 2,000 pass 16 units of 100.
        assert_eq!(
            fewest(&regions, 700, (16, 100), (100, 10, 2_000), 0),
            3_000 + 500 + 700 + 100 * 10 + 2_000
        );
        // The fewest units pass a first segment over few units.
        assert_eq!(
            fewest(&regions, 700, (64, 100), (10, 1, 1_000), 0),
            3_000 + 500 + 700 + 64 * 100
        );
        // A model's state region adds its first segment beside the cache's.
        assert_eq!(fewest(&[], 0, (1, 1), (10, 1, 100), 50), 10 + 100 + 50);
    }

    /// The ask is the room the quota holds now and what the start lacks, together.
    #[test]
    fn a_start_asks_for_the_room_it_has_and_the_room_it_lacks() {
        assert_eq!(wanted(1_000, 4_000), 5_000);
        assert_eq!(warm_lacking(5_000, 3_000), 2_000);
        assert_eq!(warm_lacking(2_000, 3_000), 0);
    }
}
