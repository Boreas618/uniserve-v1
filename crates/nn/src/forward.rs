//! What one forward borrows from the executor: the stream, the loaded kernels, the activation
//! arena and the communicator.
//!
//! A layer receives [`Forward`] and nothing else about the runtime: it cannot create a stream,
//! allocate device memory, capture a graph, or look up a program (specs/worker.md 4.3, `AGENTS.md`,
//! Model Library Boundary). Every device address a layer writes is either an argument of the
//! capability call or a view [`Arena`] handed out.
//!
//! ## The arena
//!
//! The design puts the concrete `Arena` of specs/worker.md 4.5 with the executor — a region of the
//! graph pool plus one static activation plan per captured bucket — but the executor also owns
//! `Forward`'s borrow of it, and the crate dependency runs `executor → nn`, so the type itself
//! lives here and the executor fills it. [`Arena`] is the bump region of one forward: the executor
//! resets it at the start of a forward, every layer takes its activations from it in a fixed order,
//! and because a captured bucket always executes the same sequence of layers on the same extents,
//! the sequence of reservations is the same on capture and on every replay. Replaying a graph
//! therefore touches exactly the addresses the plan recorded, and no step allocates.
//!
//! A single-threaded executor is what makes the interior cursor sound: one executor thread issues
//! every job of its device, so two forwards never share an arena at one time.

use std::cell::Cell;

use ops::{Generation, Stream};

use crate::error::{Error, Result};
use crate::group::Group;
use crate::tensor::{DType, DevicePtr, Shape, TensorView};

/// The alignment every arena reservation starts at.
///
/// Kernels read through vector loads, and an operator's output may become another one's input, so
/// activations start at the widest alignment the element types here need; 256 bytes covers every
/// one of them and the vector loads Triton emits.
pub const ALIGNMENT: u64 = 256;

/// The activation region of one forward.
pub struct Arena {
    base: DevicePtr,
    bytes: u64,
    cursor: Cell<u64>,
}

impl Arena {
    /// The arena of `bytes` bytes starting at `base`.
    ///
    /// `base` is a region the executor reserved in the graph pool for this captured bucket; the
    /// executor keeps it alive for as long as any graph that addresses it may replay.
    pub const fn new(base: DevicePtr, bytes: u64) -> Self {
        Self {
            base,
            bytes,
            cursor: Cell::new(0),
        }
    }

    /// The region's first address.
    pub const fn base(&self) -> DevicePtr {
        self.base
    }

    /// The region's size in bytes.
    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Bytes handed out since the last [`Arena::reset`].
    pub fn used(&self) -> u64 {
        self.cursor.get()
    }

    /// Start the next forward at the region's first address.
    ///
    /// The executor calls this once at the start of every forward, before the first layer runs. The
    /// reservations that follow are the ones the static plan recorded for this bucket.
    pub fn reset(&self) {
        self.cursor.set(0);
    }

    /// Reserve `bytes` bytes at [`ALIGNMENT`] and answer their first address.
    pub fn reserve(&self, bytes: u64) -> Result<DevicePtr> {
        let at = self.cursor.get().next_multiple_of(ALIGNMENT);
        let end = at.checked_add(bytes).ok_or(Error::Arena {
            needed: bytes,
            at,
            available: self.bytes,
        })?;
        if end > self.bytes {
            return Err(Error::Arena {
                needed: bytes,
                at,
                available: self.bytes,
            });
        }
        self.cursor.set(end);
        Ok(DevicePtr::new(self.base.address() + at))
    }

    /// Reserve a contiguous row-major tensor of `shape` elements of `dtype`.
    pub fn tensor(&self, dtype: DType, shape: Shape) -> Result<TensorView> {
        let bytes = dtype
            .bytes()
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "{dtype} elements are sub-byte; their storage layout names their bytes"
                ))
            })?
            .checked_mul(shape.numel())
            .ok_or_else(|| {
                Error::shape(
                    "Arena::tensor",
                    format!("{shape:?} elements overflow a byte count"),
                )
            })?;
        Ok(TensorView::new(self.reserve(bytes)?, dtype, shape))
    }
}

/// The borrowed execution resources of one forward (specs/worker.md 4.3).
pub struct Forward<'a> {
    /// The stream this forward's work is issued on.
    pub stream: &'a Stream,
    /// The generation of loaded kernels the operators resolve their programs from.
    pub kernels: &'a Generation,
    /// The activation region of this captured bucket.
    pub arena: &'a Arena,
    /// The communicator of this rank's parallel group, when the block is partitioned.
    pub group: Option<&'a Group>,
}

impl<'a> Forward<'a> {
    /// The kernels generation this forward launches against.
    pub fn kernels(&self) -> &Generation {
        self.kernels
    }

    /// The arena of this forward.
    pub fn arena(&self) -> &Arena {
        self.arena
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reservations_are_aligned_and_do_not_overlap() {
        let arena = Arena::new(DevicePtr::new(0x1_0000), 4096);
        let first = arena.tensor(DType::BF16, Shape::new(&[3])).unwrap();
        let second = arena.tensor(DType::F32, Shape::new(&[2])).unwrap();
        assert_eq!(first.ptr.address(), 0x1_0000);
        assert_eq!(second.ptr.address(), 0x1_0000 + ALIGNMENT);
        assert_eq!(arena.used(), ALIGNMENT + 8);
    }

    #[test]
    fn a_reset_replays_the_same_addresses() {
        let arena = Arena::new(DevicePtr::new(0x1_0000), 4096);
        let first = arena.tensor(DType::BF16, Shape::new(&[16])).unwrap();
        arena.reset();
        let again = arena.tensor(DType::BF16, Shape::new(&[16])).unwrap();
        assert_eq!(first.ptr, again.ptr);
    }

    #[test]
    fn a_reservation_past_the_region_fails() {
        let arena = Arena::new(DevicePtr::new(0), 512);
        assert!(arena.tensor(DType::F32, Shape::new(&[256])).is_err());
    }
}
