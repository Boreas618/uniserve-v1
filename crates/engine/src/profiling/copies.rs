//! The device time of the host KV cache's copy streams, reported as the `copy` stream's `busy`
//! time of `ForwardStats::stream_us` ([`super::forward_stats`]).
//!
//! A rank's host tier issues its loads and stores as batches of copies (`cuMemcpyBatchAsync`),
//! which CUPTI's activity records do not report, so the copy streams are timed by CUDA events
//! instead: each issue of copies is bracketed by two timing events on its copy stream
//! ([`record`]), and the time between them is counted once both completed ([`take_us`]). A copy
//! stream runs nothing but these copies and waits for no other stream, so an issue's start event
//! completes when the stream reaches its first copy and its end event when the last copy ends:
//! the time between is the stream's busy time for that issue.

use std::sync::Mutex;

use platform::Event;
use service::lock;

/// The issues whose events have not both completed yet, as `(start, end)`, oldest first.
static PENDING: Mutex<Vec<(Event, Event)>> = Mutex::new(Vec::new());

/// Counts the copies between `start` and `end`, two timing events recorded on one copy stream
/// before and after them, once both completed.
pub(crate) fn record(start: Event, end: Event) {
    lock(&PENDING).push((start, end));
}

/// The microseconds of the issues that completed since the last take, which are then forgotten;
/// an issue still running stays for a later take. Never waits for the device. An issue whose
/// time cannot be read (a device fault) is logged and dropped.
pub(crate) fn take_us() -> u64 {
    let mut pending = lock(&PENDING);
    let mut total_ms = 0.0_f64;
    pending.retain(|(start, end)| {
        match end.query() {
            Ok(false) => return true,
            Ok(true) => {}
            Err(error) => {
                tracing::warn!("a host copy's end could not be read: {error}");
                return false;
            }
        }
        match start.elapsed_ms(end) {
            Ok(milliseconds) => total_ms += f64::from(milliseconds),
            Err(error) => tracing::warn!("a host copy's time could not be read: {error}"),
        }
        false
    });
    // Rounded once per take, so that issues shorter than a microsecond still add up.
    (total_ms * 1000.0).round() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use platform::{DeviceAllocation, DeviceId, EventFlags, PinnedBuffer, Stream};

    /// A batch of copies to pinned host memory, as the host tier stores pages, counts its
    /// device time once its end event completed, and only once.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn a_batch_of_host_copies_counts_its_device_time_once() -> platform::Result<()> {
        const BYTES: usize = 256 << 20;
        let device = DeviceId(0);
        let stream = Stream::new(device)?;
        let storage = DeviceAllocation::linear(device, BYTES)?;
        let host = PinnedBuffer::new(device, BYTES)?;
        let _ = take_us();

        let timing = EventFlags {
            timing: true,
            ..EventFlags::default()
        };
        let (start, end) = (Event::new(timing), Event::new(timing));
        start.record(&stream)?;
        let mut batch = platform::CopyBatch::new(platform::SourceOrder::Stream);
        batch.push(host.address(), storage.address(), BYTES / 2);
        batch.push(
            host.address() + (BYTES / 2) as u64,
            storage.address() + (BYTES / 2) as u64,
            BYTES / 2,
        );
        // SAFETY: both allocations hold `BYTES` and outlive the copies, which the test waits
        // for; nothing else reads or writes them.
        unsafe { batch.submit(&stream)? };
        end.record(&stream)?;
        record(start, end);
        stream.synchronize()?;

        assert!(take_us() > 0, "the batch's copies were not counted");
        assert_eq!(take_us(), 0, "a batch's copies are counted once");
        Ok(())
    }
}
