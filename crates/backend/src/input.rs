//! The input buffer: where a forward's host values wait for their copy to the device.
//!
//! A host value a forward computes (positions, indices, coefficients) reaches the device by a
//! **host-to-device copy**: its bytes are written into page-locked host memory execution lends
//! the context ([`Context::set_input_buffer`](crate::Context::set_input_buffer)), at the next
//! aligned offset, and copied from there to the tensor's block. What the copy does besides
//! writing follows from the context:
//!
//! - a context that launches, on a stream that is not being captured, issues the copy at once;
//! - while the stream is being captured, the copy is recorded as an **input copy** of the graph,
//!   `(buffer offset, device address, bytes)`, and nothing is issued: execution issues a graph's
//!   input copies before each replay;
//! - a meta context issues nothing. With memory lent it writes the bytes, which is how the
//!   forward of `prepare_inputs` puts a replay's values where the graph's input copies read
//!   them; with none it only counts, which is how a profile run measures the buffer.
//!
//! Offsets depend on the sizes of the copies and their order alone, never on their values, so
//! the three forwards of one bucket agree on every offset from the same reset.

use ops::DevicePtr;
use ops::platform::{CaptureStatus, copy_from_host};

use crate::allocator::ALIGNMENT;
use crate::context::{self, State};
use crate::{Error, Result};

/// Copies `bytes` to the device address `address` by the context installed on this thread.
///
/// The caller allocated the destination; this reserves the bytes' place in the input buffer,
/// writes them there, and issues or records the copy as the module documentation states. An
/// empty slice reserves nothing.
///
/// # Errors
///
/// [`Error::NoContext`]; [`Error::NoInputBuffer`] under a context that launches and has no
/// input buffer; [`Error::InputBufferExhausted`] when the lent memory cannot hold the bytes;
/// [`Error::Platform`] and [`Error::Ops`] when the stream cannot be queried or the copy issued.
///
/// # Safety
///
/// `address` is the first of `bytes.len()` bytes on the context's device that stay valid until
/// the copy has completed, including the copies execution issues from a recorded input copy.
/// Under a meta context the address is never written.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "called by the storage layer, which allocates the tensor a host value is copied to"
    )
)]
pub(crate) unsafe fn copy_to_device(address: u64, bytes: &[u8]) -> Result<()> {
    let context = context::current("a host-to-device copy")?;
    if bytes.is_empty() {
        return Ok(());
    }
    let length = bytes.len() as u64;

    // Read before the cursor moves, so a copy that cannot be made leaves the buffer as it was.
    let captured = match context.stream() {
        Some(stream) => Some(stream.capture_status()? != CaptureStatus::None),
        None => None,
    };

    let host = {
        let mut state = context.state();
        if captured.is_some() && state.input_memory.is_none() {
            return Err(Error::NoInputBuffer);
        }

        let offset = reserve(&mut state, length)?;
        if captured == Some(true) {
            state.input_copies.push((offset, address, length));
        }
        state.input_memory.map(|(base, _)| base + offset)
    };

    let Some(host) = host else {
        return Ok(());
    };

    // SAFETY: `host` is `length` bytes inside the memory lent by `set_input_buffer`, whose caller
    // keeps it valid and writes it through this context alone; `reserve` gave these bytes to no
    // other copy since the reset, and `bytes` holds exactly `length` bytes of its own.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), host as *mut u8, bytes.len());
    }

    if let (Some(stream), Some(false)) = (context.stream(), captured) {
        // SAFETY: the source is the page-locked bytes just written, which nothing overwrites
        // before the next reset, and the caller keeps the destination valid until the copy
        // completed.
        unsafe { copy_from_host(DevicePtr::new(address), host, length, stream) }?;
    }
    Ok(())
}

/// Takes `bytes` bytes of the input buffer at its next aligned offset and returns the offset.
///
/// The offsets share the allocator's alignment, so a copy's source is aligned as its
/// destination block is. Lent memory bounds the cursor; with none the cursor only counts.
fn reserve(state: &mut State, bytes: u64) -> Result<u64> {
    let at = state.input_cursor.next_multiple_of(ALIGNMENT);
    let capacity = state.input_memory.map_or(u64::MAX, |(_, bytes)| bytes);

    let end = at.checked_add(bytes).filter(|end| *end <= capacity).ok_or(
        Error::InputBufferExhausted {
            needed: bytes,
            at,
            capacity,
        },
    )?;

    state.input_cursor = end;
    state.input_peak = state.input_peak.max(end);
    Ok(at)
}

#[cfg(test)]
mod tests {
    use super::copy_to_device;
    use crate::{Context, Error};

    /// A device address no meta context writes.
    const UNREAD: u64 = 0x8000;

    /// Makes the three host-to-device copies of one forward: 3, 300 and 1 bytes.
    fn forward(context: &Context, value: u8) {
        context.reset();
        let _guard = context.enter();

        // SAFETY: the context is meta, so the destination is never written.
        unsafe {
            copy_to_device(UNREAD, &[value; 3]).unwrap();
            copy_to_device(UNREAD, &[value + 1; 300]).unwrap();
            copy_to_device(UNREAD, &[value + 2; 1]).unwrap();
        }
    }

    #[test]
    fn copies_write_their_bytes_at_aligned_offsets_of_the_lent_memory() {
        // An ordinary heap buffer stands in for page-locked memory: a meta context only writes it.
        let mut memory = vec![0_u8; 1024];
        let context = Context::meta(None);
        // SAFETY: `memory` outlives the context's forwards and nothing else writes it meanwhile.
        unsafe { context.set_input_buffer(memory.as_mut_ptr(), memory.len() as u64) };

        forward(&context, 10);

        // 3 bytes at 0; 300 bytes at 256, ending at 556; 1 byte at the next multiple of 256.
        assert_eq!(memory[..3], [10; 3]);
        assert_eq!(memory[3], 0);
        assert_eq!(memory[256..556], [11; 300]);
        assert_eq!(memory[556], 0);
        assert_eq!(memory[768], 12);
        assert_eq!(context.input_peak(), 769);

        // A meta context issues nothing, so it has no input copy to hand to a graph.
        assert!(context.take_input_copies().is_empty());
    }

    #[test]
    fn the_same_copies_after_a_reset_take_the_same_offsets() {
        let mut memory = vec![0_u8; 1024];
        let context = Context::meta(None);
        // SAFETY: as above.
        unsafe { context.set_input_buffer(memory.as_mut_ptr(), memory.len() as u64) };

        forward(&context, 10);
        forward(&context, 20);

        // The second forward's values lie where the first forward's did.
        assert_eq!(memory[..3], [20; 3]);
        assert_eq!(memory[256..556], [21; 300]);
        assert_eq!(memory[768], 22);
        assert_eq!(context.input_peak(), 769);
    }

    #[test]
    fn a_meta_context_without_memory_counts_what_its_copies_need() {
        let context = Context::meta(None);

        forward(&context, 0);

        assert_eq!(context.input_peak(), 769);
    }

    #[test]
    fn a_copy_past_the_lent_memory_is_refused_and_takes_nothing() {
        let mut memory = vec![0_u8; 512];
        let context = Context::meta(None);
        // SAFETY: as above.
        unsafe { context.set_input_buffer(memory.as_mut_ptr(), memory.len() as u64) };
        let _guard = context.enter();

        // SAFETY: the context is meta, so the destination is never written.
        let refused = unsafe {
            copy_to_device(UNREAD, &[1; 300]).unwrap();
            copy_to_device(UNREAD, &[2; 300]).unwrap_err()
        };

        assert!(matches!(
            refused,
            Error::InputBufferExhausted {
                needed: 300,
                at: 512,
                capacity: 512
            }
        ));
        assert_eq!(context.input_peak(), 300);
    }

    #[test]
    fn a_copy_with_no_context_installed_names_what_is_missing() {
        // SAFETY: no context is installed, so nothing is written.
        let refused = unsafe { copy_to_device(UNREAD, &[1]) }.unwrap_err();
        assert!(matches!(
            refused,
            Error::NoContext {
                operation: "a host-to-device copy"
            }
        ));
    }
}
