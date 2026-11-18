//! What the KV program's moves hold: a move of host copies into Mooncake Store
//! (`crate::programs`) holds the copies it reads while its connector tasks run. A hold is a
//! reference to the block like a pin: eviction skips the block, and no reclaim returns its runs
//! while the hold stands. The holds of one holder end together, with its release.

use std::rc::Rc;

use storage::protocol::Key;

use super::{BufferPool, Entry, Memories};

impl BufferPool {
    /// Holds for `holder` the host copies of `keys` that every epoch may use:
    /// a copy an epoch on trial made is its own until its update is
    /// committed (R8). Returns, for each key held, the run of its host
    /// block. A key without such a copy is left out.
    pub fn hold(&mut self, holder: u64, keys: &[Key]) -> Vec<(Key, storage::runtime::Run)> {
        let mut held = Vec::new();
        for key in keys {
            let Some(Entry {
                host: Some((block, None, _)),
                ..
            }) = self.cache.get(key)
            else {
                continue;
            };
            // A host block is one run: its units in every part of its region.
            let run = match block.runs.borrow().as_slice() {
                [run] => run.stated(),
                _ => continue,
            };
            held.push((*key, run));
            let block = Rc::clone(block);
            // A hold is a holder: the copy is no longer evictable.
            self.touch(&block);
            self.holds.entry(holder).or_default().push(block);
        }
        self.settle_counts();
        held
    }

    /// The host parts that hold the host copy of `key`, by host and part index: none for a key
    /// without one, and for a copy of groups that state no host parts.
    pub fn held_on(&self, key: &Key) -> Vec<(&str, u32)> {
        let host = self.cache.get(key).and_then(|entry| entry.host.as_ref());
        let valid = host.and_then(|(block, ..)| block.valid.get().memories());
        let parts = valid.unwrap_or(Memories::NONE).iter();
        parts.filter_map(|part| self.host_part(part)).collect()
    }

    /// Ends every hold of `holder`; a block whose last holder this was is
    /// reclaimed. Returns whether the holder held anything.
    pub fn release_holds(&mut self, holder: u64) -> bool {
        let Some(blocks) = self.holds.remove(&holder) else {
            return false;
        };
        for block in blocks {
            self.reclaim(block);
        }
        self.settle_counts();
        true
    }
}
