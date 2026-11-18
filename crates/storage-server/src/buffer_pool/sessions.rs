//! Sessions, regions and epochs: what `Open` and `Close` on the Storage
//! Server's socket, the loss of an engine and the end of an update's trial
//! do to the buffer pool.
//!
//! A session states its layout by name ([`BufferPool::admit`]): a region
//! another session stated under the same name is that region, a memory space
//! and a memory are named the same way, and the pool numbers each name once.
//! A memory, a group's device memory, the memory of the main process a
//! product buffer lies in, or a host part (a host and a part index), is a
//! bit of a block's validity, so a deployment's sessions state at most 64 of
//! them over the pool's life.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use storage::protocol::Segment;
use storage::runtime::{GroupId, Open, Segments, Tier, VERSION};

use super::{
    BufferPool, Class, ClientId, Deferred, Group, Memories, Region, Validity, disowned,
    insert_range, remove_range,
};

/// The most regions a pool numbers at once: a run names its region by a byte (§11), and a
/// region's number is reused once the region is closed (every session that named it ended).
const REGIONS: usize = u8::MAX as usize;

/// The most worker groups a session states: a key of a `Listing` names the groups that hold or
/// can load it as the bits of a byte.
const LISTED_GROUPS: u8 = u8::BITS as u8;

/// How a region numbers the units of its memory space: the space, the
/// bytes of a unit, and the first unit it may allocate with the limit of
/// its numbering.
type Numbering = (u8, u64, (u32, u32));

impl BufferPool {
    /// Adds a range of a memory space to its free ranges: its memory exists
    /// on every rank that maps the space and is in no region.
    pub fn grant(&mut self, space: u8, offset: u64, bytes: u64) {
        insert_range(self.ledger(space), offset, bytes);
    }

    /// The free ranges of a memory space, empty until a range is granted.
    fn ledger(&mut self, space: u8) -> &mut BTreeMap<u64, u64> {
        while self.quota.len() <= usize::from(space) {
            self.quota.push(BTreeMap::new());
        }
        &mut self.quota[usize::from(space)]
    }

    /// Takes a range out of a memory space's free ranges, before its
    /// memory is released. Returns false, and changes nothing, when a part
    /// of the range is in a region.
    pub fn withdraw(&mut self, space: u8, offset: u64, bytes: u64) -> bool {
        let ranges = self.quota.get_mut(usize::from(space));
        ranges.is_some_and(|ranges| remove_range(ranges, offset, bytes))
    }

    /// The bytes of the units of a segment, `units` from `first`, that a
    /// region numbered as `numbering` allocates from: those past its
    /// reserved ones. `None` when it allocates none of them.
    fn segment_bytes(numbering: Numbering, first: u32, units: u32) -> Option<(u64, u64)> {
        let (_, unit_bytes, (reserved, limit)) = numbering;
        let (from, to) = (first.max(reserved), first.saturating_add(units).min(limit));
        (from < to).then(|| {
            (
                u64::from(from) * unit_bytes,
                u64::from(to - from) * unit_bytes,
            )
        })
    }

    /// Why a segment stated by a region numbered as `numbering` cannot
    /// join its memory space's table and free ranges, if it cannot: it has
    /// no unit or lies past the numbering, or a segment of the table is the
    /// same segment or shares a unit with it, or a unit of it is in the
    /// free ranges or in a region.
    fn ungrantable(&self, numbering: Numbering, stated: (Segment, u32, u32)) -> Option<String> {
        let (space, _, (_, limit)) = numbering;
        let (segment, first, units) = stated;
        let name = format!("segment {}/{}", segment.epoch, segment.ordinal);
        let end = first
            .checked_add(units)
            .filter(|end| units > 0 && *end <= limit);
        let Some(end) = end else {
            return Some(format!(
                "{name} of {units} units from {first} lies past a numbering of {limit}"
            ));
        };
        let table = self.segments.get(usize::from(space)).into_iter().flatten();
        let taken = table
            .clone()
            .any(|(other, from, count)| *other == segment || (*from < end && first < from + count));
        if taken {
            return Some(format!("{name} or one of its units is in the table"));
        }

        let (offset, bytes) = Self::segment_bytes(numbering, first, units)?;
        let quota = self.quota.get(usize::from(space)).into_iter().flatten();
        let free = quota
            .clone()
            .any(|(start, length)| *start < offset + bytes && offset < start + length);
        let held = self
            .regions
            .iter()
            .filter(|region| region.space == space)
            .any(|region| {
                let from = offset / region.unit_bytes;
                let to = (offset + bytes).div_ceil(region.unit_bytes);
                let (from, to) = (from as u32, u32::try_from(to).unwrap_or(u32::MAX));
                !region.outside_holds(from, to)
            });
        (free || held).then(|| format!("a unit of {name} is free or in a region"))
    }

    /// Enters a segment stated by a region numbered as `numbering` into its
    /// memory space's table, and the units it allocates from among the
    /// segment's into the space's free ranges ([`Self::ungrantable`] says
    /// when it may).
    fn enter_segment(&mut self, numbering: Numbering, stated: (Segment, u32, u32)) {
        let (space, _, _) = numbering;
        let (_, first, units) = stated;
        if let Some((offset, bytes)) = Self::segment_bytes(numbering, first, units) {
            self.grant(space, offset, bytes);
        }
        while self.segments.len() <= usize::from(space) {
            self.segments.push(Vec::new());
        }
        let table = &mut self.segments[usize::from(space)];
        let at = table.partition_point(|(_, from, _)| *from < first);
        table.insert(at, stated);
    }

    /// How a region numbers its units: its memory space, the bytes of a
    /// unit, and its span.
    fn numbering(&self, region: u8) -> Option<Numbering> {
        let found = self.regions.get(usize::from(region))?;
        Some((found.space, found.unit_bytes, found.span()))
    }

    /// The engine's ranks created `segment` of the device region `region`,
    /// `units` units from `first`: it joins its memory space's table, and
    /// its units the free ranges a resize grows the space's regions from.
    ///
    /// # Errors
    ///
    /// Refuses, changing nothing, a segment of a region that is not a
    /// device region, or that is in the table, or a unit of which lies past
    /// the region's numbering, in the free ranges or in a region.
    pub fn grant_segment(
        &mut self,
        region: u8,
        segment: Segment,
        first: u32,
        units: u32,
    ) -> Result<(), String> {
        let numbering = self
            .numbering(region)
            .filter(|_| self.regions[usize::from(region)].tier == Tier::Device)
            .ok_or_else(|| format!("region {region} is no device region"))?;
        let stated = (segment, first, units);
        if let Some(refusal) = self.ungrantable(numbering, stated) {
            return Err(refusal);
        }
        self.enter_segment(numbering, stated);
        Ok(())
    }

    /// The engine's ranks are to release `segment` of the memory space of
    /// the device region `region`: its units leave the space's free ranges
    /// and the segment its table, so no region grows into them.
    ///
    /// # Errors
    ///
    /// Refuses, changing nothing, while a unit of the segment is in a
    /// region, or when the table holds no such segment.
    pub fn withdraw_segment(&mut self, region: u8, segment: Segment) -> Result<(), String> {
        let numbering = self
            .numbering(region)
            .ok_or_else(|| format!("no region {region}"))?;
        let space = usize::from(numbering.0);
        let name = format!("segment {}/{}", segment.epoch, segment.ordinal);
        let table = self.segments.get(space).map_or(&[][..], Vec::as_slice);
        let at = table
            .iter()
            .position(|(other, _, _)| *other == segment)
            .ok_or_else(|| format!("{name} is not in the table"))?;
        let (_, first, units) = table[at];
        if let Some((offset, bytes)) = Self::segment_bytes(numbering, first, units)
            && !self.withdraw(numbering.0, offset, bytes)
        {
            return Err(format!("a unit of {name} is in a region"));
        }
        self.segments[space].remove(at);
        Ok(())
    }

    /// The segment table of the memory space of a region.
    pub fn segments_of(&self, region: u8) -> Segments {
        let space = self.numbering(region).map(|(space, _, _)| space);
        space
            .and_then(|space| self.segments.get(usize::from(space)))
            .cloned()
            .unwrap_or_default()
    }

    /// The limit of the numbering of a region: the units from 0 its
    /// memory space's units are numbered up to.
    pub fn limit_of(&self, region: u8) -> u32 {
        self.numbering(region).map_or(0, |(_, _, (_, limit))| limit)
    }

    /// The limit of the numbering a layout's region `stated` is opened with:
    /// that of the region the pool holds under its name, whose numbering
    /// stays what the session that first named it stated, else the one it
    /// states. Two epochs of one layout whose engines measured their devices
    /// differently thereby number one region alike.
    fn stated_limit(&self, stated: &storage::runtime::Region) -> u32 {
        self.regions
            .iter()
            .find(|region| region.name == stated.name)
            .map_or(stated.limit, |region| region.span().1)
    }

    /// The number of the region stated under `name`.
    pub fn region_number(&self, name: &str) -> Option<u8> {
        let found = self.regions.iter().position(|region| region.name == name)?;
        Some(found as u8)
    }

    /// Opens a region named `name` of `limit` units over a memory space, of
    /// which the first `reserved` are never allocated, each unit of
    /// `unit_bytes` over its ranks and `part_bytes` on one of them. R10: its
    /// units enter it from the quota, up to `units` of them; the region
    /// states how many did.
    #[allow(clippy::too_many_arguments)]
    pub fn open_region(
        &mut self,
        name: &str,
        tier: Tier,
        space: u8,
        (unit_bytes, part_bytes): (u64, u64),
        reserved: u32,
        units: u32,
        limit: u32,
    ) -> u8 {
        // A closed region leaves its number to a new one: the lowest closed number, else the next.
        let reused = (0..self.regions.len()).find(|index| self.regions[*index].closed);
        let id = reused.unwrap_or(self.regions.len()) as u8;
        let bytes = (unit_bytes, part_bytes);
        let mut region = Region::new(id, name, tier, space, bytes, reserved, 0, limit);
        if let Some(quota) = self.quota.get_mut(usize::from(space)) {
            region.grow(units, |offset, bytes| remove_range(quota, offset, bytes));
        }
        match reused {
            Some(index) => self.regions[index] = region,
            None => self.regions.push(region),
        }
        id
    }

    /// Closes a region that no session names any more: its copies leave
    /// their entries and its memory returns to the quota.
    pub fn close_region(&mut self, region: u8) {
        let cached: Vec<_> = self
            .cache
            .iter()
            .filter(|(_, entry)| entry.copy_in(region).is_some())
            .map(|(key, _)| *key)
            .collect();
        for key in cached {
            self.evict_in(&key, region);
        }
        let found = &mut self.regions[usize::from(region)];
        let space = usize::from(found.space);
        for (offset, bytes) in found.shrink(u32::MAX) {
            insert_range(&mut self.quota[space], offset, bytes);
        }
        found.closed = true;
        self.settle_counts();
    }

    /// Opens the session `client` of the layout an engine epoch states in
    /// `open`: every name is resolved to the pool's number, and the regions
    /// new to the pool, or closed since, are opened with the units the
    /// layout asks for. The segments a device region states that its
    /// memory space's table lacks join it first, with their units
    /// ([`Self::grant_segment`]); a memory space new to the pool whose first
    /// region states none is granted that region's units asked for on the
    /// device, memory of the ranks' own processes, and its whole numbering
    /// on the host, whose memory the server creates; the space's other
    /// regions grow from what that leaves. A region the pool holds keeps
    /// its numbering, whatever limit the layout states for it
    /// ([`Self::limit_of`] answers it). Returns, per region of the layout,
    /// its number and its units.
    ///
    /// # Errors
    ///
    /// Refuses, changing nothing, a layout that does not fit: another version
    /// of the interface than this build's [`VERSION`], a name stated
    /// twice, a region stated again with another tier, space, unit size or
    /// reserved units, a segment that cannot join its space's table, an index
    /// that names no region of the right tier, a group of no scheduler's or
    /// of two, a group numbered at or past [`LISTED_GROUPS`], more than [`Memories::CAPACITY`]
    /// memories and host parts in use at once, or more regions than a run can number.
    pub fn admit(&mut self, client: ClientId, open: &Open) -> Result<Vec<(u8, u32)>, String> {
        if self.sessions.contains_key(&client) {
            return Err(format!("session {client} is open"));
        }
        self.fits(open)?;

        // Every check passed: nothing below refuses.
        // A closed region the session states again by its name opens again under its number and
        // numbering, before any new region takes a closed number.
        for stated in &open.regions {
            if let Some(found) = self
                .regions
                .iter_mut()
                .find(|region| region.name == stated.name)
            {
                found.closed = false;
            }
        }
        let mut reserved = self.memories_in_use();
        let named = self.named_by_sessions();
        let mut numbers = Vec::with_capacity(open.regions.len());
        for stated in &open.regions {
            let known = self
                .regions
                .iter()
                .position(|region| region.name == stated.name);
            let space = match known {
                Some(found) => self.regions[found].space,
                None => self.space(stated),
            };
            // The segments the epoch's ranks created that the space's
            // table lacks join it, before its regions grow.
            let limit = self.stated_limit(stated);
            let numbering = (space, stated.unit_bytes, (stated.reserved, limit));
            for segment in &stated.segments {
                if !self.tabled(space, segment) {
                    self.enter_segment(numbering, *segment);
                }
            }

            if let Some(found) = known {
                // A region every session left opens again with the units
                // asked for, as far as the quota holds them.
                let region = &mut self.regions[found];
                if !named.contains(&(found as u8))
                    && let Some(quota) = self.quota.get_mut(usize::from(region.space))
                {
                    let missing = stated.units.saturating_sub(region.units());
                    region.grow(missing, |offset, bytes| remove_range(quota, offset, bytes));
                }
                numbers.push(found as u8);
                continue;
            }
            let number = self.open_region(
                &stated.name,
                stated.tier,
                space,
                (stated.unit_bytes, stated.part_bytes),
                stated.reserved,
                stated.units,
                stated.limit,
            );
            if !stated.memory.is_empty() {
                let memory = self.number_memory((stated.memory.clone(), None), &mut reserved);
                self.regions[usize::from(number)].memory = Some(memory);
            }
            numbers.push(number);
        }

        let number = |index: u8| numbers[usize::from(index)];
        let classes = open.classes.iter().map(|class| Class {
            host: class.host.map(|(region, units)| (number(region), units)),
            ..class.clone()
        });
        let numbered = |lists: &Vec<Vec<u8>>| -> Vec<Vec<u8>> {
            let numbered = lists
                .iter()
                .map(|list| list.iter().map(|region| number(*region)));
            numbered.map(Iterator::collect).collect()
        };
        let mut groups = Vec::with_capacity(open.groups.len());
        for group in &open.groups {
            let memory = self.number_memory((group.memory.clone(), None), &mut reserved);
            let mut parts = Vec::with_capacity(group.parts.len());
            for stated in &group.parts {
                let mut set = Memories::NONE;
                for (host, part) in stated {
                    set |= Memories::one(
                        self.number_memory((host.clone(), Some(*part)), &mut reserved),
                    );
                }
                parts.push(set);
            }
            let found = Group {
                receives: numbered(&group.receives),
                parts,
                ..Group::new(memory, number(group.slots), numbered(&group.regions))
            };
            groups.push((group.id, found));
        }
        self.open(
            client,
            open.epoch,
            open.trial,
            (classes.collect(), open.trajectory),
            groups,
        );
        Ok(numbers
            .iter()
            .map(|number| (*number, self.regions[usize::from(*number)].units()))
            .collect())
    }

    /// The number of the memory space a region of a layout states. A space
    /// new to the pool is granted the units of the segments the region
    /// states when it states any, which the caller enters; otherwise the
    /// region's units asked for on the device, memory of the ranks' own
    /// processes, and its whole numbering on the host, whose memory the
    /// server creates.
    fn space(&mut self, stated: &storage::runtime::Region) -> u8 {
        if let Some(space) = self.spaces.iter().position(|space| *space == stated.space) {
            return space as u8;
        }
        self.spaces.push(stated.space.clone());
        let space = (self.spaces.len() - 1) as u8;
        self.ledger(space);
        if !stated.segments.is_empty() {
            return space;
        }
        let end = match stated.tier {
            Tier::Host => stated.limit,
            Tier::Device | Tier::Remote => stated.reserved + stated.units,
        };
        let first = u64::from(stated.reserved) * stated.unit_bytes;
        let bytes = u64::from(end - stated.reserved) * stated.unit_bytes;
        if bytes > 0 {
            self.grant(space, first, bytes);
        }
        space
    }

    /// The number of a memory a session states: the number the memory has while it is in use, or
    /// a number no memory in use holds, or the next. `reserved` holds the numbers in use
    /// ([`Self::memories_in_use`]) and those this admission numbered so far, and gains the
    /// number given. [`Self::fits`] refused a layout whose memories the numbers cannot hold.
    pub(super) fn number_memory(
        &mut self,
        memory: (String, Option<u32>),
        reserved: &mut Memories,
    ) -> u8 {
        let named = self.memories.iter().position(|known| *known == memory);
        let number = match named {
            Some(number) => number,
            None => {
                let free = (0..self.memories.len())
                    .find(|number| !reserved.meets(Memories::one(*number as u8)));
                match free {
                    Some(number) => {
                        self.memories[number] = memory;
                        number
                    }
                    None => {
                        self.memories.push(memory);
                        self.memories.len() - 1
                    }
                }
            }
        };
        let number = number as u8;
        *reserved |= Memories::one(number);
        number
    }

    /// The memories in use: those an open session's group addresses or copies with, those a
    /// region lies in, and those a block is valid in or a copy writes. A memory's number is
    /// reused only once none of them names it, so a block is never valid in a memory that came
    /// to hold its number later.
    pub(super) fn memories_in_use(&self) -> Memories {
        let groups = self.groups.values().map(|group| {
            let parts: Memories = group.parts.iter().copied().collect();
            parts | Memories::one(group.memory)
        });
        let regions = self
            .regions
            .iter()
            .filter_map(|region| region.memory.map(Memories::one));
        let blocks = self
            .blocks()
            .into_iter()
            .filter_map(|(block, ..)| block.valid.get().memories());
        let copies = self.copies.values().flatten().map(|copy| copy.memory);
        groups.chain(regions).chain(blocks).chain(copies).collect()
    }

    /// Whether a memory space's table holds a segment as stated.
    fn tabled(&self, space: u8, stated: &(Segment, u32, u32)) -> bool {
        self.segments
            .get(usize::from(space))
            .is_some_and(|table| table.contains(stated))
    }

    /// Whether the segments a layout states fit: each of a device region
    /// and inside its numbering; two that state one segment or share a
    /// unit of one space state the same; and each segment of a space the
    /// pool has is in its table as stated or may join it
    /// ([`Self::ungrantable`]).
    fn segments_fit(&self, open: &Open) -> Result<(), String> {
        let mut stated: Vec<(&str, (Segment, u32, u32))> = Vec::new();
        for region in &open.regions {
            if !region.segments.is_empty() && region.tier != Tier::Device {
                return Err(format!(
                    "region {} states segments off the device",
                    region.name
                ));
            }
            let space = self.spaces.iter().position(|space| *space == region.space);
            let limit = self.stated_limit(region);
            for segment in &region.segments {
                let (identity, first, units) = *segment;
                let end = first.checked_add(units);
                if units == 0 || end.is_none_or(|end| end > limit) {
                    return Err(format!(
                        "region {} states segment {}/{} past its numbering",
                        region.name, identity.epoch, identity.ordinal
                    ));
                }
                let clash = stated.iter().find(|(of, (other, from, count))| {
                    *of == region.space
                        && (*other == identity || (*from < first + units && first < from + count))
                        && (*other, *from, *count) != *segment
                });
                if clash.is_some() {
                    return Err(format!(
                        "segment {}/{} of {} is stated twice, or shares a unit with another",
                        identity.epoch, identity.ordinal, region.space
                    ));
                }
                stated.push((&region.space, *segment));

                if let Some(space) = space.map(|space| space as u8)
                    && !self.tabled(space, segment)
                {
                    let numbering = (space, region.unit_bytes, (region.reserved, limit));
                    if let Some(refusal) = self.ungrantable(numbering, *segment) {
                        return Err(format!("region {}: {refusal}", region.name));
                    }
                }
            }
        }
        Ok(())
    }

    /// Whether a layout fits the pool; see [`Self::admit`].
    fn fits(&self, open: &Open) -> Result<(), String> {
        if open.version != VERSION {
            return Err(format!(
                "the buffer pool serves version {VERSION} of its interface, not {}",
                open.version
            ));
        }
        self.segments_fit(open)?;
        let mut names = HashSet::new();
        for region in &open.regions {
            if !names.insert(region.name.as_str()) {
                return Err(format!("region {} is stated twice", region.name));
            }
            // A unit is a whole number of parts, one per rank that holds a part of it, and a
            // completion numbers parts by a word.
            let parts = region.unit_bytes.checked_div(region.part_bytes);
            if region.part_bytes == 0
                || region.unit_bytes % region.part_bytes != 0
                || parts.is_none_or(|parts| u32::try_from(parts).is_err())
            {
                return Err(format!(
                    "region {} states parts of {} bytes, which do not divide its units of {} bytes",
                    region.name, region.part_bytes, region.unit_bytes
                ));
            }
            // The units a layout asks for lie in the numbering it states; of a region the pool
            // holds, it gets those its own numbering has.
            let end = region.reserved.checked_add(region.units);
            // A unit is a whole number of parts, one on each rank that holds one.
            let parts = region.part_bytes > 0 && region.unit_bytes % region.part_bytes == 0;
            if region.unit_bytes == 0
                || !parts
                || region.tier == Tier::Remote
                || end.is_none_or(|end| end > region.limit)
            {
                return Err(format!(
                    "region {} states {} reserved and {} units of {} bytes, parts of {}, in {}, \
                     on {:?}",
                    region.name,
                    region.reserved,
                    region.units,
                    region.unit_bytes,
                    region.part_bytes,
                    region.limit,
                    region.tier
                ));
            }
            // Units that exist once lie in a memory of the device.
            if !region.memory.is_empty() && region.tier != Tier::Device {
                return Err(format!(
                    "region {} states a memory off the device",
                    region.name
                ));
            }
            // A region the pool holds keeps its numbering, whatever limit the layout states;
            // everything else that lays its units out, and the memory it lies in, is the same.
            if let Some(found) = self.regions.iter().find(|found| found.name == region.name) {
                let space = self.spaces.get(usize::from(found.space));
                let memory = found
                    .memory
                    .and_then(|bit| self.memories.get(usize::from(bit)));
                let lies = memory.map_or("", |(name, _)| name.as_str());
                let same = found.tier == region.tier
                    && space == Some(&region.space)
                    && found.unit_bytes == region.unit_bytes
                    && found.part_bytes == region.part_bytes
                    && found.span().0 == region.reserved
                    && lies == region.memory;
                if !same {
                    return Err(format!(
                        "region {} is stated with another layout than before",
                        region.name
                    ));
                }
            }
        }
        // A stated name the pool holds opens that region (again, if it was closed); every other
        // name takes a closed number the admission does not open again, else the next.
        let new = names
            .iter()
            .filter(|name| !self.regions.iter().any(|region| region.name == **name))
            .count();
        let reusable = self
            .regions
            .iter()
            .filter(|region| region.closed && !names.contains(&region.name.as_str()))
            .count();
        if self.regions.len() + new.saturating_sub(reusable) > REGIONS {
            return Err(format!(
                "a deployment's buffer pool numbers at most {REGIONS} regions"
            ));
        }

        let tier = |index: u8| {
            open.regions
                .get(usize::from(index))
                .map(|region| region.tier)
        };
        // A request names a class by a byte.
        if open.classes.len() > usize::from(u8::MAX) + 1 {
            return Err(format!("{} classes are too many", open.classes.len()));
        }
        if !open.classes.is_empty() && usize::from(open.trajectory) >= open.classes.len() {
            return Err(format!("class {} names no class", open.trajectory));
        }
        for (index, class) in open.classes.iter().enumerate() {
            if let Some((host, _)) = class.host
                && tier(host) != Some(Tier::Host)
            {
                return Err(format!(
                    "class {index}'s host region {host} is no host region"
                ));
            }
        }

        let mut ids = HashSet::new();
        for group in &open.groups {
            if !ids.insert(group.id) {
                return Err(format!("group {} is stated twice", group.id.0));
            }
            if group.id.0 >= LISTED_GROUPS {
                return Err(format!(
                    "group {} is numbered past the {LISTED_GROUPS} groups a session states",
                    group.id.0
                ));
            }
            let device = group.regions.iter().flatten();
            let device = device.chain(group.receives.iter().flatten());
            if group.regions.len() > open.classes.len()
                || group.receives.len() > open.classes.len()
                || group.parts.len() > open.classes.len()
                || device
                    .chain([&group.slots])
                    .any(|region| tier(*region) != Some(Tier::Device))
            {
                return Err(format!(
                    "group {} names a region that is not one of its device regions",
                    group.id.0
                ));
            }
        }
        // The memories in use keep their numbers; a memory the layout states that is not among
        // them takes one more.
        let in_use = self.memories_in_use();
        let mut needed = in_use.count() as usize;
        for memory in stated_memories(open) {
            let named = self.memories.iter().position(|known| *known == memory);
            if !named.is_some_and(|number| in_use.meets(Memories::one(number as u8))) {
                needed += 1;
            }
        }
        if needed > Memories::CAPACITY {
            return Err(format!(
                "the deployment's sessions would state {needed} memories and host parts in use, \
                 more than {}",
                Memories::CAPACITY
            ));
        }
        let mut scheduled = HashSet::new();
        if open.schedulers.len() > usize::from(u16::MAX) {
            return Err(format!("{} schedulers are too many", open.schedulers.len()));
        }
        for group in open.schedulers.iter().flatten() {
            if !ids.contains(group) || !scheduled.insert(*group) {
                return Err(format!(
                    "group {} is of no group of the layout, or of two schedulers",
                    group.0
                ));
            }
        }
        Ok(())
    }

    /// A session of an engine epoch, while one is open: the epoch's only
    /// one where a single engine of it opened one. An epoch whose engine has
    /// a standby has a session per engine, which their numbers tell apart.
    pub fn session(&self, epoch: u64) -> Option<ClientId> {
        self.sessions
            .iter()
            .find(|(_, of)| **of == epoch)
            .map(|(client, _)| *client)
    }

    /// The epoch of a session, while it is open.
    pub fn epoch(&self, client: ClientId) -> Option<u64> {
        self.sessions.get(&client).copied()
    }

    /// The name of a region, by a number the pool gave it: a region keeps its place among the
    /// pool's regions for the pool's life, so every number a run or a completion states names
    /// one.
    pub fn region_name(&self, region: u8) -> &str {
        &self.regions[usize::from(region)].name
    }

    /// The name of the device memory, or of the memory of a main process, numbered `memory`;
    /// `None` for a host part.
    fn memory_name(&self, memory: u8) -> Option<&str> {
        match self.memories.get(usize::from(memory))? {
            (name, None) => Some(name),
            (_, Some(_)) => None,
        }
    }

    /// The host and the part index of the host part numbered `memory`; `None` for a memory.
    pub(super) fn host_part(&self, memory: u8) -> Option<(&str, u32)> {
        match self.memories.get(usize::from(memory))? {
            (host, Some(part)) => Some((host, *part)),
            (_, None) => None,
        }
    }

    /// The memories and the host parts of `valid` by their names, as a listing and a task state
    /// them: none of either for a slot row, which is valid wherever its region is mapped.
    pub(crate) fn memory_names(&self, valid: Validity) -> (Vec<&str>, Vec<(&str, u32)>) {
        let set = valid.memories().unwrap_or(Memories::NONE);
        let names = set.iter().filter_map(|memory| self.memory_name(memory));
        let parts = set.iter().filter_map(|memory| self.host_part(memory));
        (names.collect(), parts.collect())
    }

    /// The regions a session names: its groups' slot rows and device
    /// regions, and its classes' host regions.
    fn named(&self, client: ClientId) -> BTreeSet<u8> {
        let groups = self.groups.iter().filter(|((of, _), _)| *of == client);
        let mut named: BTreeSet<u8> = groups
            .flat_map(|(_, group)| {
                let regions = group.regions.iter().flatten().copied();
                regions.chain([group.slots])
            })
            .collect();
        if let Some((classes, _)) = self.classes.get(&client) {
            named.extend(
                classes
                    .iter()
                    .flat_map(|class| class.host.map(|(region, _)| region)),
            );
        }
        named
    }

    /// The regions the open sessions name.
    fn named_by_sessions(&self) -> BTreeSet<u8> {
        self.sessions
            .keys()
            .flat_map(|client| self.named(*client))
            .collect()
    }

    /// Opens a session: an engine epoch, on trial or not, its classes with
    /// the one that holds stored trajectories, and its worker groups. Each
    /// group states the device memory its ranks address (a bit of a block's
    /// validity), the region of its slot rows, and per class the device
    /// regions its owners' blocks are taken from, the regions a `Replicate`
    /// into its memories takes runs in, and the host parts it copies with.
    pub(super) fn open(
        &mut self,
        client: ClientId,
        epoch: u64,
        trial: bool,
        classes: (Vec<Class>, u8),
        groups: Vec<(GroupId, Group)>,
    ) {
        self.classes.insert(client, classes);
        self.sessions.insert(client, epoch);
        self.trial.entry(epoch).or_insert(trial);
        for (group, found) in groups {
            self.groups.insert((client, group), found);
        }
    }

    /// Ends a session: it closes ([`Self::close`]), and every device region
    /// it named that no other open session names closes with it
    /// ([`Self::close_region`]). Host regions stay with their copies: they
    /// are the deployment's, and close with its pool.
    pub fn end(&mut self, client: ClientId) {
        let named = self.named(client);
        self.close(client);
        let still = self.named_by_sessions();
        for region in named.difference(&still) {
            if self.regions[usize::from(*region)].tier == Tier::Device {
                self.close_region(*region);
            }
        }
    }

    /// Closes a session, or takes note that its engine is gone (A7: its
    /// processes have exited and run no batch). Its copies end as failed;
    /// every stamp of its groups counts as reached; its commits and
    /// transfers that waited are dropped, since a block whose last batch
    /// may not have ended is never shared; its owners are detached. What it
    /// committed stays cached, and what its tasks hold stays held until
    /// they end (R9).
    pub fn close(&mut self, client: ClientId) {
        let mut mine: Vec<_> = self
            .copies
            .keys()
            .filter(|(queue, _)| queue.0 == client)
            .copied()
            .collect();
        // In the order of their requests, not of the hash table: what the
        // pool frees and stops follows its inputs alone (§18).
        mine.sort_unstable();
        for task in mine {
            let run_by_worker = self.copies[&task]
                .iter()
                .all(|copy| copy.deadline.is_none());
            if run_by_worker {
                self.finished(task.0, task.1, false);
            } else if self.answers.remove(&task).is_some() {
                self.evictions.remove(&task);
                self.stopped.push(task);
            }
        }

        let groups: Vec<_> = self
            .groups
            .keys()
            .filter(|(of, _)| *of == client)
            .copied()
            .collect();
        for group in groups {
            let Some(found) = self.groups.remove(&group) else {
                continue;
            };
            for (_, work) in found.deferred {
                match work {
                    Deferred::Release { blocks } => {
                        blocks.into_iter().for_each(|block| self.reclaim(block))
                    }
                    Deferred::Commit { block, .. } => {
                        block.disown();
                        self.reclaim(block);
                    }
                    Deferred::Replicate {
                        queue,
                        request,
                        block,
                        ..
                    } => {
                        self.answers.remove(&(queue, request));
                        self.reclaim(block);
                    }
                    Deferred::Lend { .. } | Deferred::Finish { .. } => {}
                }
            }
        }

        let owners: Vec<_> = self
            .owners
            .keys()
            .filter(|(of, _)| *of == client)
            .copied()
            .collect();
        for owner in owners {
            // R9: a task that writes for the owner holds what it writes.
            self.orphan(owner.0, owner.1);
            let Some(found) = self.owners.remove(&owner) else {
                continue;
            };
            for block in disowned(found) {
                self.reclaim(block);
            }
        }

        // The eviction orders of its queue pairs are dropped with it.
        for region in &mut self.regions {
            region.uninstall(client);
        }
        self.numbers.retain(|queue, _| queue.0 != client);
        // Its lends end with its owners; what a borrow holds stays held until it is returned.
        self.lends.retain(|(of, _), _| *of != client);
        self.completions.retain(|(queue, _, _)| queue.0 != client);
        self.sessions.remove(&client);
        self.classes.remove(&client);
        self.sweep();
        self.settle();
        self.settle_counts();
    }

    /// The update of an epoch on trial is committed: what the epoch made is
    /// visible to every epoch.
    pub fn commit_update(&mut self, epoch: u64) {
        if let Some(trial) = self.trial.get_mut(&epoch) {
            *trial = false;
        }
        for entry in self.cache.values_mut() {
            let makers = entry
                .device
                .iter_mut()
                .map(|(_, maker)| maker)
                .chain(entry.host.iter_mut().map(|(_, maker, _)| maker));
            for maker in makers {
                if *maker == Some(epoch) {
                    *maker = None;
                }
            }
        }
    }

    /// The update of an epoch on trial is rolled back, once its session is
    /// closed: what the epoch made stays visible to it alone and is evicted.
    pub fn roll_back(&mut self, epoch: u64) {
        if self.trial.get(&epoch) == Some(&true) && !self.sessions.values().any(|of| *of == epoch) {
            self.trial.remove(&epoch);
            self.sweep();
            self.settle_counts();
        }
    }

    /// Evicts the copies an epoch made whose update was rolled back, as far
    /// as nothing else holds them: no epoch can use them.
    pub(super) fn sweep(&mut self) {
        let rolled =
            |maker: &Option<u64>| maker.is_some_and(|epoch| !self.trial.contains_key(&epoch));
        let mut copies = Vec::new();
        for (key, entry) in &self.cache {
            copies.extend(
                entry
                    .copies()
                    .filter(|(_, maker)| rolled(maker))
                    .map(|(block, _)| (*key, block.region())),
            );
        }
        for (key, region) in copies {
            self.evict_in(&key, region);
        }
    }
}

/// The memories a layout states, each a bit of a block's validity: its
/// groups' device memories, the memories its regions lie in, and its groups'
/// host parts.
fn stated_memories(open: &Open) -> Vec<(String, Option<u32>)> {
    let devices = open.groups.iter().map(|group| (group.memory.clone(), None));
    let regions = open
        .regions
        .iter()
        .filter(|region| !region.memory.is_empty())
        .map(|region| (region.memory.clone(), None));
    let parts = open
        .groups
        .iter()
        .flat_map(|group| group.parts.iter().flatten());
    let parts = parts.map(|(host, part)| (host.clone(), Some(*part)));
    let mut stated = Vec::new();
    for memory in devices.chain(regions).chain(parts) {
        if !stated.contains(&memory) {
            stated.push(memory);
        }
    }
    stated
}
