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

use std::cell::{Cell, RefCell};

use ops::twins::Twins;
use ops::{Generation, Op, Stream};

use crate::error::{Error, Result};
use crate::exchange::ExchangeViews;
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
    /// The arena of `bytes` bytes starting at `base`, aligned up to [`ALIGNMENT`].
    ///
    /// `base` is a region the executor reserved in the graph pool for this captured bucket; the
    /// executor keeps it alive for as long as any graph that addresses it may replay.
    ///
    /// The base is aligned **up** to [`ALIGNMENT`] and the same number of bytes is taken off the
    /// region's size, so that every reservation is 256-byte aligned whatever the allocator answered.
    /// An arena hands its reservations out at that granularity, so a base carrying the pool's own
    /// skew would put every buffer of a forward at the same skew — and the tensor-core programs these
    /// buffers feed are compiled for sixteen-byte tensor pointers, which turns that skew into a warp
    /// misaligned address inside a kernel that names no operand. An eight-byte-skewed region costs
    /// at most 255 bytes here and nothing anywhere else.
    pub fn new(base: DevicePtr, bytes: u64) -> Self {
        let address = base.address();
        let aligned = address.next_multiple_of(ALIGNMENT);
        let skew = aligned - address;
        Self {
            base: DevicePtr::new(aligned),
            bytes: bytes.saturating_sub(skew),
            cursor: Cell::new(0),
        }
    }

    /// The region's first address, aligned to [`ALIGNMENT`].
    pub const fn base(&self) -> DevicePtr {
        self.base
    }

    /// The region's usable size in bytes, after the base was aligned.
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

    /// The current offset, for a later [`Arena::release`].
    ///
    /// One forward reserves its layer-independent buffers once and then runs every layer; the
    /// scratch a single layer reserves is dead as soon as that layer's launches are enqueued. A
    /// captured bucket replays the same addresses, so a layer that took a fresh reservation each
    /// pass would ask the region for `layers` copies of it — which is neither what the plan states
    /// nor what a replayed graph needs.
    pub fn mark(&self) -> u64 {
        self.cursor.get()
    }

    /// Hand back everything reserved since `mark`.
    ///
    /// The memory stays in the region and the bytes are handed out again to the next layer, which
    /// is what makes a plan one layer deep. The caller owes that nothing it still uses was reserved
    /// after `mark`: the stream orders the layer's launches before the next layer's, so reusing the
    /// same addresses is what a captured graph already does.
    pub fn release(&self, mark: u64) {
        self.cursor.set(mark);
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

/// The one operator a shadow forward runs twice (specs/worker.md 7.3).
pub struct Shadow<'a> {
    /// The operator whose programs are being compared, by `Op::NAME`.
    pub operator: &'static str,
    /// The descriptor key of the program the previous generation served this operator with.
    ///
    /// The comparison pairs by operator, not by descriptor: an update may replace an operator by
    /// another compilation of the same mathematics, whose descriptor text differs (7.3 step 2).
    /// The old program is resolved by this key, because that is the one it was stored under; a
    /// launch whose own key the previous generation still holds is paired with that one instead,
    /// so an operator that appears under several descriptors is never compared against a program
    /// compiled for another.
    pub previous_key: String,
    /// The generation the previous program is resolved from.
    pub previous: &'a Generation,
    /// The twin buffers the new program writes into.
    pub twins: RefCell<&'a mut Twins>,
}

impl<'a> Shadow<'a> {
    /// The comparison of `operator`'s calls, against the program the previous generation served it
    /// with under `previous_key`.
    pub fn new(
        operator: &'static str,
        previous_key: impl Into<String>,
        previous: &'a Generation,
        twins: &'a mut Twins,
    ) -> Self {
        Self {
            operator,
            previous_key: previous_key.into(),
            previous,
            twins: RefCell::new(twins),
        }
    }
}

/// The borrowed execution resources of one forward (specs/worker.md 4.3).
pub struct Forward<'a> {
    /// The stream this forward's work is issued on and the generation of loaded kernels the
    /// operators resolve their programs from, or `None` for a **recording** forward: one that states
    /// the programs a graph launches instead of launching them.
    ///
    /// A recording forward exists so that the enumeration of a model's programs can be the model's own
    /// `forward` rather than a second traversal written beside it. It needs no device: the arena is a
    /// null-backed region, `operate` formats the descriptor and hands it to the capture sink, and no
    /// kernel is resolved. That is the difference between "the walk runs the model" (which would need
    /// a GPU and the checkpoint's shards) and this: the walk runs the model's *shape arithmetic*.
    /// The stream this forward's work is issued on, or `None` for a recording forward.
    stream: Option<&'a Stream>,
    /// The generation of loaded kernels the operators resolve their programs from, or `None` for a
    /// recording forward.
    pub kernels: Option<&'a Generation>,
    /// The activation region of this captured bucket.
    pub arena: &'a Arena,
    /// The communicator of this rank's parallel group, when the block is partitioned.
    pub group: Option<&'a Group>,
    /// The expert exchange's received views, when the block's experts are split across ranks.
    pub exchange: Option<&'a ExchangeViews>,
    /// The operator this forward shadows, when it is a 7.3 comparison pass.
    pub shadow: Option<Shadow<'a>>,
    /// `[1]` i32: the token rows of this batch that are live, when the caller states them.
    ///
    /// A captured bucket is wider than the batch that fills it. A layer that partitions work by
    /// token row needs to know which rows are real, and it cannot derive that from any shape: the
    /// arena is the bucket's. The value is device-resident because it changes per replay of one
    /// captured graph, and the buffer must outlive the bucket - a layer reads it on every replay.
    pub live_tokens: Option<TensorView>,
}

impl<'a> Forward<'a> {
    /// A forward with the resources every block needs and no partitioning.
    pub fn new(stream: &'a Stream, kernels: &'a Generation, arena: &'a Arena) -> Self {
        Self {
            stream: Some(stream),
            kernels: Some(kernels),
            arena,
            group: None,
            exchange: None,
            shadow: None,
            live_tokens: None,
        }
    }

    /// A forward that records the programs a graph launches instead of launching them.
    ///
    /// The arena is the caller's: a null-backed region is enough, because no kernel reads it. Every
    /// `operate` then formats the descriptor and reports it to the installed capture sink
    /// ([`ops::capture::Sink`]), so running a model's `forward` against this collects the exact set of
    /// programs it launches — the same set a device would launch, from the same code.
    pub fn recording(arena: &'a Arena) -> Self {
        Self {
            stream: None,
            kernels: None,
            arena,
            group: None,
            exchange: None,
            shadow: None,
            live_tokens: None,
        }
    }

    /// Whether this forward launches, rather than only recording what it would launch.
    pub const fn is_recording(&self) -> bool {
        self.stream.is_none()
    }

    /// The stream this forward's launches and collectives are issued on.
    ///
    /// A recording forward has none: every `operate` returns before a kernel is resolved, and a
    /// collective — which has to reach other ranks — refuses here by name rather than pretending.
    pub fn stream(&self) -> Result<&'a Stream> {
        self.stream.ok_or_else(|| {
            Error::Unsupported(
                "this forward records the programs a graph launches and has no stream, so it cannot \
                 collect across ranks"
                    .to_owned(),
            )
        })
    }

    /// The same forward over a parallel group.
    pub fn with_group(mut self, group: &'a Group) -> Self {
        self.group = Some(group);
        self
    }

    /// The same forward over an expert exchange.
    pub fn with_exchange(mut self, exchange: &'a ExchangeViews) -> Self {
        self.exchange = Some(exchange);
        self
    }

    /// The same forward as a shadow comparison of one operator's call.
    /// State which token rows of this batch are live, for the layers that partition by row.
    ///
    /// The buffer is `[1]` i32 on the device and must outlive the bucket: a captured graph reads it
    /// on every replay, so a buffer reallocated per request is a dangling pointer in a graph that
    /// outlives the request.
    pub fn with_live_tokens(mut self, tokens: TensorView) -> Self {
        self.live_tokens = Some(tokens);
        self
    }

    /// The token rows of this batch that are live, when the caller stated them.
    pub fn live_of(&self) -> Option<TensorView> {
        self.live_tokens
    }

    pub fn with_shadow(mut self, shadow: Shadow<'a>) -> Self {
        self.shadow = Some(shadow);
        self
    }

    /// Resolve `O`'s kernel for this descriptor and launch it.
    ///
    /// When this forward shadows `O` at this descriptor key (specs/worker.md 7.3), the launch is
    /// paired: the new kernel writes the twins and the previous generation's kernel writes the real
    /// outputs last, so the forward continues on the outputs it was going to compute anyway.
    /// Otherwise it is one launch of the current generation's kernel.
    pub fn operate<'t, O: Op>(&self, descriptor: &O::Descriptor, args: O::Args<'t>) -> Result<()>
    where
        O::Args<'t>: Copy,
    {
        // A recording forward states the program and stops: no kernel to resolve, no launch to make,
        // and the descriptor key formatted only when a sink will read it.
        let (Some(stream), Some(kernels)) = (self.stream, self.kernels) else {
            if ops::capture::installed() {
                ops::capture::launched(O::NAME, &ops::descriptor::key(descriptor));
            }
            return Ok(());
        };
        // Paired by operator: a program replaced by another compilation of the same mathematics
        // may carry a different descriptor text, and that is the update 7.3 step 2 exists for.
        let shadow = self
            .shadow
            .as_ref()
            .filter(|shadow| shadow.operator == O::NAME);
        let Some(shadow) = shadow else {
            let kernel = kernels.op::<O>(descriptor)?;
            O::launch(kernel, descriptor, args, stream)
                .map_err(|error| Error::launch(O::NAME, ops::descriptor::key(descriptor), error))?;
            Self::record::<O>(descriptor);
            return Ok(());
        };
        // Record the bound the pair is compared with before resolving either kernel, so a harness
        // reads it even when one of the two generations lacks this descriptor.
        let mut twins = shadow.twins.borrow_mut();
        twins.set_tolerance(O::tolerance(descriptor));
        let current = kernels.op::<O>(descriptor)?;
        // The previous program is resolved by the key its generation stored it under, and by the
        // launch's own key when that generation still has one: an operator served under several
        // descriptors is compared program for program, never against a neighbour's compilation.
        let key = ops::descriptor::key(descriptor);
        let previous = if shadow.previous.contains(O::NAME, &key) {
            shadow.previous.kernel(O::NAME, &key)?
        } else {
            shadow.previous.kernel(O::NAME, &shadow.previous_key)?
        };
        // Redirect the call's outputs into the twins and let the new program write them; the
        // previous program then writes the real outputs, which the forward continues on.
        let redirected = O::shadow(args, &mut twins, stream);
        O::launch(current, descriptor, redirected, stream)
            .map_err(|error| Error::launch(O::NAME, ops::descriptor::key(descriptor), error))?;
        Self::record::<O>(descriptor);
        drop(twins);
        O::launch(previous, descriptor, args, stream)
            .map_err(|error| Error::launch(O::NAME, ops::descriptor::key(descriptor), error))?;
        Self::record::<O>(descriptor);
        Ok(())
    }

    /// Tell the capture sink, when one is installed, that `O`'s program just launched.
    ///
    /// A capture records the graph nodes each operator launch added, and a program update rebinds
    /// one operator's parameters at a time through that table (`specs/worker.md` 4.5). The key is
    /// formatted only when a sink will read it: every launch of a serving step passes through here,
    /// and a capture is the exception, not the rule.
    fn record<O: Op>(descriptor: &O::Descriptor) {
        if ops::capture::installed() {
            ops::capture::launched(O::NAME, &ops::descriptor::key(descriptor));
        }
    }

    /// The kernels generation this forward launches against, or `None` for a recording forward.
    pub fn kernels(&self) -> Option<&'a Generation> {
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

    /// A recording forward states the program its graph would launch and launches nothing: no stream,
    /// no kernel generation, no device. That is what lets a model's own `forward` be the enumeration of
    /// its programs — a traversal that cannot miss a launch, on a machine that has no GPU and no shards.
    #[test]
    fn a_recording_forward_reports_the_program_it_would_launch() {
        struct Rows(Vec<(String, String)>);

        impl ops::capture::Sink for Rows {
            fn launched(&mut self, operator: &'static str, key: &str) {
                self.0.push((operator.to_owned(), key.to_owned()));
            }
        }

        let arena = Arena::new(DevicePtr::new(0), 1 << 20);
        let forward = Forward::recording(&arena);
        assert!(forward.is_recording());

        let descriptor = crate::layers::Silu::descriptor_of(DType::BF16, 1024);
        let extent = Shape::new(&[4, 1024]);
        let args = ops::activation::SiluArgs {
            x: arena.tensor(DType::BF16, extent).unwrap(),
            out: arena.tensor(DType::BF16, extent).unwrap(),
        };

        let mut rows = Rows(Vec::new());
        ops::capture::set_sink(Some(&mut rows));
        forward
            .operate::<ops::activation::Silu>(&descriptor, args)
            .expect("a recording forward launches nothing");
        ops::capture::set_sink(None);

        assert_eq!(
            rows.0,
            vec![("silu".to_owned(), ops::descriptor::key(&descriptor))]
        );
    }

    #[test]
    fn reservations_are_aligned_and_do_not_overlap() {
        let arena = Arena::new(DevicePtr::new(0x1_0000), 4096);
        let first = arena.tensor(DType::BF16, Shape::new(&[3])).unwrap();
        let second = arena.tensor(DType::F32, Shape::new(&[2])).unwrap();
        assert_eq!(first.ptr.address(), 0x1_0000);
        assert_eq!(second.ptr.address(), 0x1_0000 + ALIGNMENT);
        assert_eq!(arena.used(), ALIGNMENT + 8);
    }

    /// A pool may answer a region at eight-byte granularity, and the programs these buffers feed are
    /// compiled for sixteen-byte tensor pointers: the arena's own base is where that has to be
    /// fixed, because a reservation cannot correct a skew it inherits.
    #[test]
    fn a_skewed_region_is_aligned_up_before_anything_is_reserved() {
        let arena = Arena::new(DevicePtr::new(0x1_0008), 4096);
        assert_eq!(arena.base().address(), 0x1_0100);
        assert_eq!(arena.bytes(), 4096 - 0xF8);
        let tensor = arena.tensor(DType::BF16, Shape::new(&[8])).unwrap();
        assert_eq!(tensor.ptr.address() % ALIGNMENT, 0);
        assert_eq!(tensor.ptr.address(), 0x1_0100);
        // An already-aligned region keeps its address and its whole size.
        let held = Arena::new(DevicePtr::new(0x1_0000), 4096);
        assert_eq!(held.base().address(), 0x1_0000);
        assert_eq!(held.bytes(), 4096);
    }

    #[test]
    fn a_reset_replays_the_same_addresses() {
        let arena = Arena::new(DevicePtr::new(0x1_0000), 4096);
        let first = arena.tensor(DType::BF16, Shape::new(&[16])).unwrap();
        arena.reset();
        let again = arena.tensor(DType::BF16, Shape::new(&[16])).unwrap();
        assert_eq!(first.ptr, again.ptr);
    }

    /// One forward reserves its layer-independent buffers once and runs every layer: a layer's
    /// scratch is handed back at its end and the next layer receives the same addresses, which is
    /// what a replayed graph expects and what keeps the plan one layer deep.
    #[test]
    fn releasing_to_a_mark_hands_the_same_scratch_to_the_next_layer() {
        let arena = Arena::new(DevicePtr::new(0x1_0000), 4096);
        let held = arena.tensor(DType::BF16, Shape::new(&[16])).unwrap();
        let mark = arena.mark();
        let first = arena.tensor(DType::BF16, Shape::new(&[1024])).unwrap();
        arena.release(mark);
        let second = arena.tensor(DType::BF16, Shape::new(&[1024])).unwrap();
        assert_eq!(first.ptr, second.ptr);
        assert_eq!(second.ptr.address(), held.ptr.address() + ALIGNMENT);
        // The buffer reserved before the mark is still the arena's, and still where it was.
        assert_eq!(arena.used(), ALIGNMENT + 2048);
    }

    #[test]
    fn a_reservation_past_the_region_fails() {
        let arena = Arena::new(DevicePtr::new(0), 512);
        assert!(arena.tensor(DType::F32, Shape::new(&[256])).is_err());
    }
}
