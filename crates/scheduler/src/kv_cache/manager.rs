//! The reuse rules of each attention type, which the prefix cache applies
//! at every level: the device cache (L1), the host KV cache (L2) and the KV
//! storage (L3).
//!
//! Every cache group has one manager of its attention type. A reuse ends at
//! a page boundary of the prompt, a multiple of the largest block size; a
//! boundary is reusable only when every group's manager accepts it, and a
//! manager accepts a boundary when the pages a later reader of that group
//! needs are present at some level:
//!
//! - full attention reads every page before the boundary;
//! - a sliding window reads only the pages of the window before it, those
//!   covering `[boundary - window, boundary)`.
//!
//! The walk over the boundaries goes from the deepest down, and a boundary a
//! manager rejects does not end it: a sliding window absent at one boundary
//! may be present at an earlier one. The reuse is the deepest boundary every
//! manager accepts.

use std::ops::Range;

/// The rules of one cache group's attention type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AttentionManager {
    Full(FullAttentionManager),
    SlidingWindow(SlidingWindowManager),
}

/// A full-attention group: every token reads every earlier token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FullAttentionManager {
    /// Tokens per page.
    pub(crate) block_size: usize,
}

/// A sliding-window group: every token reads the `window` tokens before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SlidingWindowManager {
    /// Tokens per page.
    pub(crate) block_size: usize,
    /// History tokens a reader needs.
    pub(crate) window: usize,
}

impl FullAttentionManager {
    /// Every page before `boundary`.
    fn needed_pages(self, boundary: usize) -> Range<usize> {
        0..boundary / self.block_size
    }
}

impl SlidingWindowManager {
    /// The pages covering the window before `boundary`.
    pub(crate) fn needed_pages(self, boundary: usize) -> Range<usize> {
        let end = boundary / self.block_size;
        (boundary.saturating_sub(self.window) / self.block_size).min(end)..end
    }
}

impl AttentionManager {
    /// Tokens per page of the group.
    pub(crate) fn block_size(self) -> usize {
        match self {
            Self::Full(manager) => manager.block_size,
            Self::SlidingWindow(manager) => manager.block_size,
        }
    }

    /// The pages a reader that continues from token `boundary` needs; the
    /// boundary is a multiple of the block size.
    pub(crate) fn needed_pages(self, boundary: usize) -> Range<usize> {
        match self {
            Self::Full(manager) => manager.needed_pages(boundary),
            Self::SlidingWindow(manager) => manager.needed_pages(boundary),
        }
    }

    /// Whether a reuse at `boundary` finds every page it needs, given the
    /// group's misses: `misses[page]` counts the absent pages before
    /// `page`, for the pages the levels were asked about. A page beyond them
    /// is absent.
    pub(crate) fn accepts(self, boundary: usize, misses: &[usize]) -> bool {
        let pages = self.needed_pages(boundary);
        pages.end < misses.len() && misses[pages.end] == misses[pages.start]
    }
}

/// The running count of absent pages before each page, from whether each
/// page is present: the input of [`AttentionManager::accepts`].
pub(crate) fn misses(present: impl Iterator<Item = bool>) -> Vec<usize> {
    let mut total = 0;
    let mut misses = vec![0];
    for present in present {
        total += usize::from(!present);
        misses.push(total);
    }
    misses
}

/// The deepest multiple of `step` tokens, at most `limit`, that every
/// manager accepts with its group's `misses`; zero when none is.
pub(crate) fn deepest_boundary(
    managers: &[AttentionManager],
    step: usize,
    limit: usize,
    misses: &[Vec<usize>],
) -> usize {
    let step = step.max(1);
    let mut boundary = limit / step * step;
    while boundary > 0 {
        let accepted = managers
            .iter()
            .zip(misses)
            .all(|(manager, misses)| manager.accepts(boundary, misses));
        if accepted {
            return boundary;
        }
        boundary -= step;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn present(pages: &str) -> Vec<usize> {
        misses(pages.chars().map(|page| page == '#'))
    }

    /// A full group accepts a boundary whose every earlier page is present;
    /// a sliding window only the pages of its window.
    #[test]
    fn each_attention_type_needs_its_own_pages() {
        let full = AttentionManager::Full(FullAttentionManager { block_size: 4 });
        let window = AttentionManager::SlidingWindow(SlidingWindowManager {
            block_size: 4,
            window: 6,
        });
        let pages = present("#.##");
        assert!(full.accepts(4, &pages));
        assert!(!full.accepts(16, &pages));
        // The window before token 16 covers tokens 10..16: pages 2 and 3.
        assert!(window.accepts(16, &pages));
        assert!(!window.accepts(12, &pages));
    }

    /// A boundary one manager rejects does not end the walk: a window
    /// absent at the deepest boundary is present at an earlier one.
    #[test]
    fn the_walk_continues_past_a_rejected_boundary() {
        let managers = [
            AttentionManager::Full(FullAttentionManager { block_size: 4 }),
            AttentionManager::SlidingWindow(SlidingWindowManager {
                block_size: 4,
                window: 4,
            }),
        ];
        // The window before token 24 is page 5, absent; before 20, page 4.
        let misses = [present("######"), present("#.#.#.")];
        assert_eq!(deepest_boundary(&managers, 4, 24, &misses), 20);
        assert_eq!(
            deepest_boundary(&managers, 4, 24, &[present("##...."), present("######")]),
            8
        );
    }
}
