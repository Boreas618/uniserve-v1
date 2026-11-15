//! The engine's `Resize` of a region (SD §2.4 `Resize`): the units it asks for bound what the
//! region holds from then on. The buffer pool's shrink evicts in this queue pair's installed
//! order and never takes a private or pinned block, so it stops short while requests hold units
//! past the bound, this scheduler's or those of an epoch of the same layout that shares the
//! region, and its `Resized` states the units the region reached. Until the region reaches its
//! bound, the scheduler installs the whole of its eviction order there (`install_victims`), takes
//! none of the units past the bound for its own allocations (`read_capacity`), and asks the
//! buffer pool again once something there may leave ([`Scheduler::shrink_to_bounds`]); each such
//! answer sets the region's capacity, and is no drain's `Output::Shrunk`.
//!
//! A grow that stops short is not asked again: the memory it waits for enters the region's space
//! by the engine's grants, after which the engine resizes the region itself.
//!
//! A drain ends the bounds (`drain.rs`): a drain to another layout gives the device memory back
//! by its own shrink, and a drain beside an epoch of the same layout leaves the regions the two
//! share to that epoch, which resizes them to what it states when it is promoted, a rollback's
//! promotion included.

use crossbeam_channel::Sender;
use storage::runtime;

use super::Scheduler;

impl Scheduler {
    /// `Message::Resize`: `Resize { region, units }` at the buffer pool, whose `Resized` answers
    /// `reply` with the units the region reached. From here on the region holds at most `units`,
    /// whatever this shrink reaches.
    pub(super) fn resize(&mut self, region: u8, units: u32, reply: Sender<u32>) {
        if let Some(bound) = self
            .region_index(region)
            .and_then(|index| self.bounds.get_mut(index))
        {
            *bound = Some(units);
        }
        self.submit(runtime::Request::Resize { region, units });
        self.resizing.push((region, Some(reply)));
    }

    /// The units region `region` (an index) holds past its bound, as its last `Resized` stated
    /// them: what its shrink is still to take. 0 for a region with no bound.
    pub(super) fn excess(&self, region: usize) -> u32 {
        match (
            self.bounds.get(region).copied().flatten(),
            self.capacity.get(region),
        ) {
            (Some(bound), Some(&units)) => units.saturating_sub(bound),
            _ => 0,
        }
    }

    /// Asks each region that holds more units than its bound to shrink to it again, one `Resize`
    /// of a region at a time, once something there may leave: a free unit, or a unit of a block
    /// of this queue pair's installed order that nothing holds (the buffer pool's `available`
    /// word, which counts the units held past the bound as well). None is asked while nothing
    /// may leave, so the region reaches its bound as the units held past it come free, at the
    /// passes after they do.
    pub(super) fn shrink_to_bounds(&mut self) {
        let regions: Vec<(u8, usize)> = self
            .registers
            .capacity
            .iter()
            .filter_map(|&(number, _)| Some((number, self.region_index(number)?)))
            .collect();

        for (number, index) in regions {
            let Some(bound) = self.bounds.get(index).copied().flatten() else {
                continue;
            };
            let leaves = self.capacity_word(index, 0) > 0;
            if self.excess(index) == 0 || self.resizing_region(number) || !leaves {
                continue;
            }

            self.submit(runtime::Request::Resize {
                region: number,
                units: bound,
            });
            self.resizing.push((number, None));
        }
    }
}
