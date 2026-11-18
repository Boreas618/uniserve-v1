//! The engine's `Resize` of a region (SD §2.4 `Resize`): the units it asks for bound what the
//! region holds from then on. The buffer pool's shrink gives back the region's tail past the
//! bound: it evicts the blocks of this queue pair's installed order that lie there and never
//! takes a private or pinned block, so it stops short while requests hold units in the tail,
//! this scheduler's or those of an epoch of the same layout that shares the region, and its
//! `Resized` states the units the region reached. Until the region reaches its
//! bound, the scheduler installs the whole of its eviction order there (`install_victims`), takes
//! none of the units past the bound for its own allocations (`read_capacity`), and asks the
//! buffer pool again once something there may have left ([`Scheduler::shrink_to_bounds`]); each such
//! answer sets the region's capacity, and is no drain's `Output::Shrunk`. The bound and the
//! units are the KV decisions' (`Kv::bound`, `Kv::declared`, `Kv::excess`); the requests and
//! their replies are the scheduler's.
//!
//! A grow that stops short is not asked again: the memory it waits for enters the region's space
//! by the engine's grants, after which the engine resizes the region itself.
//!
//! An engine that must have the units past the bound back before it goes on (a release of the
//! tail segments that makes room) asks with `Message::Bound` instead of `Message::Resize`: its
//! reply is owed until a `Resized` states that the region holds no more than the bound
//! ([`Scheduler::answer_bounds`]), so it waits on that one reply instead of asking again.
//!
//! A drain ends the bounds (`drain.rs`): a drain to another layout gives the device memory back
//! by its own shrink, and a drain beside an epoch of the same layout leaves the regions the two
//! share to that epoch, which resizes them to what it states when it is promoted, a rollback's
//! promotion included.

use crossbeam_channel::Sender;
use scheduling::ids::PageId;
use storage::runtime;

use super::Scheduler;

impl Scheduler {
    /// `Message::Demote`: the cached blocks of region `region` (by number) a shrink to `units`
    /// may evict, those that lie past its first `units` units, which the buffer pool names
    /// (`Request::Tail`, answered in [`Scheduler::demote_tail`]); a region the session does not
    /// name demotes nothing.
    pub(super) fn demote(&mut self, region: u8, units: u32, reply: Sender<u32>) {
        if self.kv.region_index(region).is_none() {
            let _ = reply.send(0);
            return;
        }
        self.submit(runtime::Request::Tail { region, units });
        self.tails.push((region, reply));
    }

    /// The answer to a demotion's `Tail`: of the blocks `keys` names, those with no host copy are
    /// offered to the host tier (`Scheduler::offload`, which skips a block its class or the
    /// step's values give no host tier), and the demotion's reply is answered with the blocks
    /// offered once every store they started ended ([`Scheduler::answer_demotions`]).
    pub(super) fn demote_tail(&mut self, region: u8, keys: &[storage::Key]) {
        let Some(at) = self.tails.iter().position(|(owed, _)| *owed == region) else {
            return;
        };
        let (_, reply) = self.tails.remove(at);
        let pages = self.kv.undemoted(keys);
        let ids: Vec<PageId> = pages.iter().map(|(page, _)| *page).collect();
        self.offload(&ids);

        // The demotion waits for the stores it started, of the keys now stored or storing.
        let storing = self.host_stores();
        let keys = pages
            .into_iter()
            .map(|(_, key)| key)
            .filter(|key| storing.contains(key))
            .collect::<std::collections::HashSet<_>>();
        let demoted = u32::try_from(keys.len()).unwrap_or(u32::MAX);
        self.demoting.push((keys, demoted, reply));
        self.answer_demotions();
    }

    /// Answers each `Message::Demote` whose stores all ended.
    pub(super) fn answer_demotions(&mut self) {
        if self.demoting.is_empty() {
            return;
        }
        let storing = self.host_stores();
        let (ended, owed): (Vec<_>, Vec<_>) = std::mem::take(&mut self.demoting)
            .into_iter()
            .partition(|(keys, _, _)| keys.is_disjoint(&storing));
        self.demoting = owed;
        // A requester that stopped waiting needs no reply.
        for (_, demoted, reply) in ended {
            let _ = reply.send(demoted);
        }
    }

    /// `Message::Resize`: `Resize { region, units }` at the buffer pool, whose `Resized` answers
    /// `reply` with the units the region reached. From here on the region holds at most `units`,
    /// whatever this shrink reaches.
    pub(super) fn resize(&mut self, region: u8, units: u32, reply: Sender<u32>) {
        self.set_bound(region, units);
        self.resizing.push((region, Some(reply)));
    }

    /// `Message::Bound`: `Resize { region, units }` at the buffer pool, bounding the region to
    /// `units` as [`Scheduler::resize`] does; `reply` is answered once the region holds no more
    /// than `units` ([`Scheduler::answer_bounds`]).
    pub(super) fn bound(&mut self, region: u8, units: u32, reply: Sender<u32>) {
        self.set_bound(region, units);
        self.resizing.push((region, None));
        self.bounded.push((region, reply));
    }

    /// Bounds region `region` (by number) to `units` from here on, and asks the buffer pool for
    /// that size.
    fn set_bound(&mut self, region: u8, units: u32) {
        if let Some(index) = self.kv.region_index(region) {
            self.kv.bound(index, units);
            if let Some(released) = self.released.get_mut(index) {
                *released = false;
            }
        }
        self.submit(runtime::Request::Resize { region, units });
    }

    /// Answers each `Message::Bound` of region `region` (by number) whose region holds no more
    /// than its bound now, with the units it holds, after a `Resized` set the region's capacity.
    pub(super) fn answer_bounds(&mut self, region: u8) {
        let Some(index) = self.kv.region_index(region) else {
            return;
        };
        let units = self.kv.capacity(index);
        let reached = self.kv.bound_of(index).is_some_and(|bound| units <= bound);
        if !reached {
            return;
        }

        let (answered, owed): (Vec<_>, Vec<_>) = std::mem::take(&mut self.bounded)
            .into_iter()
            .partition(|(owed, _)| *owed == region);
        self.bounded = owed;
        // A requester that stopped waiting needs no reply.
        for (_, reply) in answered {
            let _ = reply.send(units);
        }
    }

    /// Answers every `Message::Bound` still owed with the units its region holds now: a drain
    /// ended the bounds, so no shrink toward them follows.
    pub(super) fn end_bounds(&mut self) {
        self.kv.end_bounds();
        for (region, reply) in std::mem::take(&mut self.bounded) {
            let units = self
                .kv
                .region_index(region)
                .map_or(0, |index| self.kv.capacity(index));
            let _ = reply.send(units);
        }
    }

    /// Asks each region that holds more units than its bound to shrink to it again, one `Resize`
    /// of a region at a time, once something there may leave: a request of this scheduler that
    /// gives units back was answered since the region's last `Resize` (`Scheduler::released`).
    /// The units held past the bound lie in the region's tail, which a block leaves only once
    /// its holders let it go, so none is asked while nothing was let go, and the region reaches
    /// its bound at the passes after the last of them is.
    pub(super) fn shrink_to_bounds(&mut self) {
        for (number, index) in self.kv.used().to_vec() {
            let Some(bound) = self.kv.bound_of(index) else {
                continue;
            };
            let released = self.released.get(index).copied().unwrap_or(false);
            if self.kv.excess(index) == 0 || self.resizing_region(number) || !released {
                continue;
            }

            self.submit(runtime::Request::Resize {
                region: number,
                units: bound,
            });
            self.resizing.push((number, None));
            if let Some(released) = self.released.get_mut(index) {
                *released = false;
            }
        }
    }
}
