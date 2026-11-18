//! The deterministic caching allocator of a runner's region.
//!
//! Tensors a forward creates take their memory here and return it when their last reference
//! drops, as PyTorch's caching allocator serves `torch.Tensor`. What differs is the guarantee a
//! captured CUDA graph needs: a graph replays the addresses its capture saw, and the forward that
//! writes a replay's host values must see the same ones. Placement is therefore a function of the
//! sequence of allocations and frees since the last [`Allocator::reset`] and of nothing else: no
//! clock, no thread identity, no hash order.
//!
//! The region is one address range. Below `top` it is a set of blocks, in use or free; from
//! `top` on it has never been handed out since the reset. An allocation takes the smallest free
//! block that holds it, lowest address first, and otherwise extends `top`. A freed block merges
//! with its free neighbours, and a free block that ends at `top` lowers `top` instead of staying
//! in the set, so a region whose tensors are all dropped is empty again.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, MutexGuard, PoisonError};

use crate::{Error, Result};

/// The size every block is rounded up to, and the alignment of every address.
///
/// Kernels read their operands through vector loads; 256 bytes covers every element type and
/// vector width, and matches the alignment of the shadow outputs in `ops`.
pub const ALIGNMENT: u64 = 256;

/// The allocator of one region, or of none.
///
/// An allocator over a device range hands out addresses inside it. An allocator without a range
/// only counts: its addresses start at zero and mean nothing, and [`Allocator::peak`] is the size
/// of the region a forward of the same shapes needs.
#[derive(Debug)]
pub struct Allocator {
    /// The region's first address; zero when the allocator only counts.
    base: u64,
    /// The region's size in bytes; unbounded when the allocator only counts.
    capacity: u64,
    blocks: Mutex<Blocks>,
}

/// The free blocks below `top`, indexed twice: by size for placement and by offset for merging.
#[derive(Debug, Default)]
struct Blocks {
    /// How many resets the region has seen; a block names the reset it was allocated under.
    resets: u64,
    /// The offset below which blocks are in use or free; everything from here on is unused or
    /// was given back.
    top: u64,
    /// The furthest end of any block handed out since the reset: no address from here on has
    /// been written by this forward.
    reached: u64,
    /// The furthest `top` reached under any reset.
    peak: u64,
    /// `(bytes, offset)` of every free block, so the first entry at or above a size is the
    /// smallest block that holds it at the lowest address.
    by_size: BTreeSet<(u64, u64)>,
    /// `offset -> bytes` of the same blocks, so a freed block finds its neighbours.
    by_offset: BTreeMap<u64, u64>,
}

impl Allocator {
    /// The allocator of the `capacity` bytes at device address `base`.
    pub fn new(base: u64, capacity: u64) -> Self {
        Self {
            base,
            capacity,
            blocks: Mutex::new(Blocks::default()),
        }
    }

    /// An allocator with no memory behind it, which measures what a forward needs.
    pub fn counting() -> Self {
        Self::new(0, u64::MAX)
    }

    /// An empty allocator over this one's region: it hands out the addresses this one does for
    /// the same sequence of allocations, and shares no block with it.
    pub fn alike(&self) -> Self {
        Self::new(self.base, self.capacity)
    }

    /// A block of at least `bytes` bytes: its address, and the reset it belongs to.
    ///
    /// The caller hands both back to [`Allocator::free`] with the same `bytes`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] when neither a free block nor the unused end of the region holds
    /// the block.
    pub fn allocate(&self, bytes: u64) -> Result<(u64, u64)> {
        let needed = rounded(bytes);
        let mut blocks = self.lock();

        // The smallest free block that holds the request, lowest address first.
        if let Some(&(size, offset)) = blocks.by_size.range((needed, 0)..).next() {
            blocks.remove(offset, size);
            if size > needed {
                // The rest of the block has no free neighbour: the block was free as a whole,
                // so what surrounded it is in use.
                blocks.insert(offset + needed, size - needed);
            }
            return Ok((self.base + offset, blocks.resets));
        }

        let offset = blocks.top;
        let end = offset
            .checked_add(needed)
            .filter(|end| *end <= self.capacity)
            .ok_or(Error::OutOfMemory {
                needed,
                used: blocks.top,
                capacity: self.capacity,
            })?;
        blocks.top = end;
        blocks.reached = blocks.reached.max(end);
        blocks.peak = blocks.peak.max(end);
        Ok((self.base + offset, blocks.resets))
    }

    /// A block of at least `bytes` bytes at addresses no block has had since the reset.
    ///
    /// The destination of a host-to-device copy takes its memory here. A captured graph's input
    /// copies are issued before the graph replays, so their destinations must hold their values
    /// from before the first node until their consumers read them: an address an earlier tensor
    /// of the forward had would be overwritten by that tensor's kernels during the replay.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] when the unused end of the region does not hold the block.
    pub fn allocate_unused(&self, bytes: u64) -> Result<(u64, u64)> {
        let needed = rounded(bytes);
        let mut blocks = self.lock();

        let offset = blocks.reached;
        let end = offset
            .checked_add(needed)
            .filter(|end| *end <= self.capacity)
            .ok_or(Error::OutOfMemory {
                needed,
                used: blocks.reached,
                capacity: self.capacity,
            })?;

        // What was given back between `top` and the block stays available to later tensors.
        // No free block ends at `top`, so this one has no free neighbour to merge with.
        let top = blocks.top;
        if offset > top {
            blocks.insert(top, offset - top);
        }

        blocks.top = end;
        blocks.reached = end;
        blocks.peak = blocks.peak.max(end);
        Ok((self.base + offset, blocks.resets))
    }

    /// Returns the block [`Allocator::allocate`] gave for `bytes` at `address` under `reset`.
    ///
    /// A block of an earlier reset is no longer part of the region's state: the forward that
    /// follows a reset starts from an empty region whatever its predecessor's tensors do.
    pub fn free(&self, address: u64, bytes: u64, reset: u64) {
        let mut blocks = self.lock();
        if reset != blocks.resets {
            return;
        }

        let mut offset = address - self.base;
        let mut size = rounded(bytes);

        // Merge with the free block that ends where this one starts.
        if let Some((&left, &left_size)) = blocks.by_offset.range(..offset).next_back()
            && left + left_size == offset
        {
            blocks.remove(left, left_size);
            offset = left;
            size += left_size;
        }

        // Merge with the free block that starts where this one ends.
        if let Some(&right_size) = blocks.by_offset.get(&(offset + size)) {
            blocks.remove(offset + size, right_size);
            size += right_size;
        }

        if offset + size == blocks.top {
            blocks.top = offset;
        } else {
            blocks.insert(offset, size);
        }
    }

    /// Empties the region: the start of a forward.
    pub fn reset(&self) {
        let mut blocks = self.lock();
        blocks.resets += 1;
        blocks.top = 0;
        blocks.reached = 0;
        blocks.by_size.clear();
        blocks.by_offset.clear();
    }

    /// The furthest byte of the region any forward reached.
    pub fn peak(&self) -> u64 {
        self.lock().peak
    }

    /// The bytes of the region below which blocks are in use or free: zero when every tensor
    /// allocated since the reset has been dropped.
    pub fn used(&self) -> u64 {
        self.lock().top
    }

    /// The block state. One worker thread issues a device's work, so the lock is uncontended;
    /// it exists because a tensor may be dropped on another thread. A panic while it was held
    /// cannot leave the indexes inconsistent in a way a later forward observes, since every
    /// forward starts with a reset.
    fn lock(&self) -> MutexGuard<'_, Blocks> {
        self.blocks.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Blocks {
    fn insert(&mut self, offset: u64, bytes: u64) {
        self.by_size.insert((bytes, offset));
        self.by_offset.insert(offset, bytes);
    }

    fn remove(&mut self, offset: u64, bytes: u64) {
        self.by_size.remove(&(bytes, offset));
        self.by_offset.remove(&offset);
    }
}

/// `bytes` rounded up to whole blocks of [`ALIGNMENT`]; an empty tensor still takes one, so two
/// live tensors never share an address.
fn rounded(bytes: u64) -> u64 {
    bytes.max(1).next_multiple_of(ALIGNMENT)
}

#[cfg(test)]
mod tests {
    use super::{ALIGNMENT, Allocator};
    use crate::Error;

    const BASE: u64 = 0x1000_0000;

    /// Runs a fixed forward-like sequence and returns every address it received.
    fn forward(allocator: &Allocator) -> Vec<u64> {
        let (a, reset) = allocator.allocate(1000).unwrap();
        let (b, _) = allocator.allocate(300).unwrap();
        allocator.free(a, 1000, reset);
        let (c, _) = allocator.allocate(512).unwrap();
        let (d, _) = allocator.allocate(4096).unwrap();
        allocator.free(b, 300, reset);
        let (e, _) = allocator.allocate(100).unwrap();
        vec![a, b, c, d, e]
    }

    #[test]
    fn addresses_are_aligned_and_inside_the_region() {
        let allocator = Allocator::new(BASE, 1 << 20);
        for address in forward(&allocator) {
            assert_eq!(address % ALIGNMENT, 0);
            assert!((BASE..BASE + (1 << 20)).contains(&address));
        }
    }

    #[test]
    fn the_same_sequence_after_a_reset_receives_the_same_addresses() {
        let allocator = Allocator::new(BASE, 1 << 20);
        let first = forward(&allocator);
        allocator.reset();
        let second = forward(&allocator);
        assert_eq!(first, second);
    }

    #[test]
    fn a_counting_allocator_places_blocks_as_a_region_does() {
        let region = Allocator::new(BASE, 1 << 20);
        let counting = Allocator::counting();
        let placed: Vec<u64> = forward(&region).iter().map(|a| a - BASE).collect();
        assert_eq!(placed, forward(&counting));
        assert_eq!(region.peak(), counting.peak());
    }

    #[test]
    fn a_freed_block_is_reused_by_the_smallest_fit_at_the_lowest_address() {
        let allocator = Allocator::new(BASE, 1 << 20);
        let (small, reset) = allocator.allocate(256).unwrap();
        let (_keep_a, _) = allocator.allocate(256).unwrap();
        let (large, _) = allocator.allocate(1024).unwrap();
        let (_keep_b, _) = allocator.allocate(256).unwrap();
        allocator.free(large, 1024, reset);
        allocator.free(small, 256, reset);

        // 256 bytes fit both holes; the smaller one is taken.
        assert_eq!(allocator.allocate(200).unwrap().0, small);
        // 512 bytes fit only the larger one, whose rest stays free for the next request.
        assert_eq!(allocator.allocate(512).unwrap().0, large);
        assert_eq!(allocator.allocate(512).unwrap().0, large + 512);
    }

    #[test]
    fn neighbouring_free_blocks_merge_and_the_end_of_the_region_is_given_back() {
        let allocator = Allocator::new(BASE, 1 << 20);
        let (a, reset) = allocator.allocate(256).unwrap();
        let (b, _) = allocator.allocate(256).unwrap();
        let (c, _) = allocator.allocate(256).unwrap();
        allocator.free(a, 256, reset);
        allocator.free(c, 256, reset);
        assert_eq!(allocator.used(), 512);
        allocator.free(b, 256, reset);
        assert_eq!(allocator.used(), 0);

        // The merged region serves one block as large as the three together.
        assert_eq!(allocator.allocate(768).unwrap().0, a);
        assert_eq!(allocator.peak(), 768);
    }

    #[test]
    fn a_block_of_an_earlier_reset_does_not_return_to_the_region() {
        let allocator = Allocator::new(BASE, 1 << 20);
        let (stale, reset) = allocator.allocate(256).unwrap();
        allocator.reset();
        let (first, _) = allocator.allocate(256).unwrap();
        let (second, _) = allocator.allocate(256).unwrap();
        allocator.free(stale, 256, reset);

        // Had the stale block been freed into the region, this would receive `first`'s address.
        let (third, _) = allocator.allocate(256).unwrap();
        assert_eq!((first, second, third), (BASE, BASE + 256, BASE + 512));
    }

    #[test]
    fn an_unused_block_takes_no_address_an_earlier_block_had() {
        let allocator = Allocator::new(BASE, 1 << 20);
        let (a, reset) = allocator.allocate(512).unwrap();
        let (b, _) = allocator.allocate(256).unwrap();
        allocator.free(b, 256, reset);
        allocator.free(a, 512, reset);
        assert_eq!(allocator.used(), 0);

        // An ordinary block reuses the start of the region; an unused one lies past both.
        let (unused, _) = allocator.allocate_unused(256).unwrap();
        assert_eq!(unused, BASE + 768);

        // The addresses given back before it still serve later tensors.
        assert_eq!(allocator.allocate(768).unwrap().0, BASE);
        assert_eq!(allocator.peak(), 1024);
    }

    #[test]
    fn unused_blocks_follow_one_another_and_repeat_after_a_reset() {
        let allocator = Allocator::new(BASE, 1 << 20);
        let sequence = |allocator: &Allocator| {
            let (a, reset) = allocator.allocate(300).unwrap();
            let (first, _) = allocator.allocate_unused(100).unwrap();
            allocator.free(a, 300, reset);
            let (second, _) = allocator.allocate_unused(100).unwrap();
            let (b, _) = allocator.allocate(300).unwrap();
            [a, first, second, b]
        };

        let before = sequence(&allocator);
        assert_eq!(before, [BASE, BASE + 512, BASE + 768, BASE]);
        allocator.reset();
        assert_eq!(sequence(&allocator), before);
    }

    #[test]
    fn an_allocation_beyond_the_region_is_refused_with_what_is_in_use() {
        let allocator = Allocator::new(BASE, 1024);
        allocator.allocate(768).unwrap();
        let refused = allocator.allocate(512).unwrap_err();
        assert!(matches!(
            refused,
            Error::OutOfMemory {
                needed: 512,
                used: 768,
                capacity: 1024
            }
        ));
    }

    #[test]
    fn empty_tensors_receive_distinct_addresses() {
        let allocator = Allocator::counting();
        let (a, _) = allocator.allocate(0).unwrap();
        let (b, _) = allocator.allocate(0).unwrap();
        assert_ne!(a, b);
    }
}
