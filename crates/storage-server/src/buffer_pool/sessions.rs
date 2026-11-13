//! Sessions, regions and epochs: what `Open` and `Close` on the Storage
//! Server's socket, the loss of an engine and the end of an update's trial
//! do to the buffer pool.
//!
//! A session states its layout by name ([`BufferPool::admit`]): a region
//! another session stated under the same name is that region, a memory space
//! and a device memory are named the same way, and the pool numbers each
//! name once. A device memory is a bit of a block's validity, so a
//! deployment's sessions state at most eight of them over the pool's life.

use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};

use storage::protocol::Segment;
use storage::runtime::{GroupId, Open, Segments, Tier};

use super::{
    BufferPool, Class, ClientId, Deferred, Group, MEMORIES, Region, disowned, insert_range,
    remove_range,
};

/// The most regions a pool numbers at once: a run names its region by a byte.
const REGIONS: usize = u8::MAX as usize;

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
        // A region that holds no unit and that no session names leaves its
        // number to a new one once every number is taken.
        let reused = (self.regions.len() >= REGIONS)
            .then(|| {
                let named = self.named_by_sessions();
                (0..self.regions.len()).find(|index| {
                    self.regions[*index].units() == 0 && !named.contains(&(*index as u8))
                })
            })
            .flatten();
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
    /// regions grow from what that leaves. Returns, per region of the
    /// layout, its number and its units.
    ///
    /// # Errors
    ///
    /// Refuses, changing nothing, a layout that does not fit: a name stated
    /// twice, a region stated again with another tier, space, unit size or
    /// numbering, a segment that cannot join its space's table, an index
    /// that names no region of the right tier, a group of no scheduler's or
    /// of two, more than [`MEMORIES`] device memories over the pool's
    /// sessions, or more regions than a run can number.
    pub fn admit(&mut self, client: ClientId, open: &Open) -> Result<Vec<(u8, u32)>, String> {
        if self.sessions.contains_key(&client) {
            return Err(format!("session {client} is open"));
        }
        self.fits(open)?;

        // Every check passed: nothing below refuses.
        for group in &open.groups {
            if !self.memories.contains(&group.memory) {
                self.memories.push(group.memory.clone());
            }
        }
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
            let numbering = (space, stated.unit_bytes, (stated.reserved, stated.limit));
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
            numbers.push(self.open_region(
                &stated.name,
                stated.tier,
                space,
                (stated.unit_bytes, stated.part_bytes),
                stated.reserved,
                stated.units,
                stated.limit,
            ));
        }

        let number = |index: u8| numbers[usize::from(index)];
        let classes = open.classes.iter().map(|class| Class {
            host: class.host.map(|(region, units)| (number(region), units)),
            ..class.clone()
        });
        let groups = open.groups.iter().map(|group| {
            let memory = self
                .memories
                .iter()
                .position(|memory| *memory == group.memory)
                .unwrap_or_default() as u8;
            let regions = group
                .regions
                .iter()
                .map(|regions| regions.iter().map(|region| number(*region)).collect())
                .collect();
            (group.id, memory, number(group.slots), regions)
        });
        let groups: Vec<_> = groups.collect();
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
            for segment in &region.segments {
                let (identity, first, units) = *segment;
                let end = first.checked_add(units);
                if units == 0 || end.is_none_or(|end| end > region.limit) {
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
                    let numbering = (space, region.unit_bytes, (region.reserved, region.limit));
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
        self.segments_fit(open)?;
        let mut names = HashSet::new();
        for region in &open.regions {
            if !names.insert(region.name.as_str()) {
                return Err(format!("region {} is stated twice", region.name));
            }
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
            if let Some(found) = self.regions.iter().find(|found| found.name == region.name) {
                let space = self.spaces.get(usize::from(found.space));
                let same = found.tier == region.tier
                    && space == Some(&region.space)
                    && found.unit_bytes == region.unit_bytes
                    && found.part_bytes == region.part_bytes
                    && found.span() == (region.reserved, region.limit);
                if !same {
                    return Err(format!(
                        "region {} is stated with another layout than before",
                        region.name
                    ));
                }
            }
        }
        let new = names
            .iter()
            .filter(|name| !self.regions.iter().any(|region| region.name == **name))
            .count();
        let named = self.named_by_sessions();
        let reusable = (0..self.regions.len())
            .filter(|index| self.regions[*index].units() == 0 && !named.contains(&(*index as u8)))
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
        if open.classes.len() >= usize::from(super::SLOT) {
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
        let mut memories: Vec<&String> = self.memories.iter().collect();
        for group in &open.groups {
            if !ids.insert(group.id) {
                return Err(format!("group {} is stated twice", group.id.0));
            }
            let device = group.regions.iter().flatten().chain([&group.slots]);
            if group.regions.len() > open.classes.len()
                || device
                    .into_iter()
                    .any(|region| tier(*region) != Some(Tier::Device))
            {
                return Err(format!(
                    "group {} names a region that is not one of its device regions",
                    group.id.0
                ));
            }
            if !memories.contains(&&group.memory) {
                memories.push(&group.memory);
            }
        }
        if memories.len() > MEMORIES {
            return Err(format!(
                "the deployment's sessions state {} device memories, more than {MEMORIES}",
                memories.len()
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

    /// The session of an engine epoch, while it is open.
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

    /// The name of a region, by its number.
    pub fn region_name(&self, region: u8) -> Option<&str> {
        let found = self.regions.get(usize::from(region))?;
        Some(&found.name)
    }

    /// The name of the device memory of a bit of a block's validity, by
    /// the bit's position.
    pub fn memory_name(&self, bit: u32) -> Option<&str> {
        self.memories.get(bit as usize).map(String::as_str)
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
                    .filter_map(|class| class.host.map(|(region, _)| region)),
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
    /// regions its owners' blocks are taken from.
    pub fn open(
        &mut self,
        client: ClientId,
        epoch: u64,
        trial: bool,
        classes: (Vec<Class>, u8),
        groups: Vec<(GroupId, u8, u8, Vec<Vec<u8>>)>,
    ) {
        self.classes.insert(client, classes);
        self.sessions.insert(client, epoch);
        self.trial.entry(epoch).or_insert(trial);
        for (group, memory, slots, regions) in groups {
            let found = Group {
                memory,
                slots,
                regions,
                completed: 0,
                deferred: VecDeque::new(),
            };
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
        let mine: Vec<_> = self
            .copies
            .keys()
            .filter(|(queue, _)| queue.0 == client)
            .copied()
            .collect();
        for task in mine {
            let run_by_worker = self.copies[&task]
                .iter()
                .all(|copy| copy.deadline.is_none());
            if run_by_worker {
                self.finished(task.0, task.1, false);
            } else if self.answers.remove(&task).is_some() {
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
                    Deferred::Transfer { .. } | Deferred::Finish { .. } => {}
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
