//! What peers hold: another host's Storage Server reads host copies of this
//! deployment's blocks (`specs/buffer-pool.md` §11), and holds them while
//! it reads. A hold is a reference to the block like a pin: eviction skips
//! the block, and no reclaim returns its runs while the hold stands. The
//! holds of a peer end together, with its release or its connection.

use std::rc::Rc;

use storage::protocol::Key;

use super::{BufferPool, Entry};

impl BufferPool {
    /// Holds for `peer` the host copies of `keys` that every epoch may use:
    /// a copy an epoch on trial made is its own until its update is
    /// committed (R8). Returns, for each key held, the run of its host
    /// block and the tokens it holds. A key without such a copy is left
    /// out.
    pub fn hold(&mut self, peer: u64, keys: &[Key]) -> Vec<(Key, storage::runtime::Run, u16)> {
        let mut held = Vec::new();
        for key in keys {
            let Some(Entry {
                host: Some((block, None)),
                tokens,
                ..
            }) = self.cache.get(key)
            else {
                continue;
            };
            // A host block is one run: its units in every part of its region.
            let [run] = block.runs.as_slice() else {
                continue;
            };
            held.push((*key, run.stated(), *tokens));
            let block = Rc::clone(block);
            // A hold is a holder: the copy is no longer evictable.
            self.touch(&block);
            self.peers.entry(peer).or_default().push(block);
        }
        self.settle_counts();
        held
    }

    /// Ends every hold of `peer`; a block whose last holder this was is
    /// reclaimed. Returns whether the peer held anything.
    pub fn release_peer(&mut self, peer: u64) -> bool {
        let Some(blocks) = self.peers.remove(&peer) else {
            return false;
        };
        for block in blocks {
            self.reclaim(block);
        }
        self.settle_counts();
        true
    }
}
