//! Holds an interprocess event on device 0 until it is killed, which is what a
//! supervised rank does when a test checks that the supervisor accounts for the
//! driver's shared memory per rank.
//!
//! The test that makes that check needs a *separate process* with its own
//! primary context, since the driver names the shared memory for the process
//! that created the event, and the thing under test is the supervisor reading
//! those names after a rank dies. `crates/daemon`'s supervisor test re-executes
//! its own test binary as that stand-in
//! (`supervision::supervisor::tests::event_holder_rank`), because a lib test
//! cannot count on another package's example being built; this binary is the
//! same role run by hand, so a probe needs no test harness:
//!
//!     cargo run -p platform --example interprocess_event_holder
//!
//! It prints the event's 64-byte interprocess handle as hex on one line and
//! then sleeps for an hour, so a second process can pass the handle to
//! `Event::open_ipc` or simply read the shared memory the driver created for
//! this one.

use std::time::Duration;

use platform::{Device, DeviceId, Event, EventFlags};

fn main() -> platform::Result<()> {
    let device = Device::new(DeviceId(0))?;
    let stream = device.stream()?;
    // No timing: a timing event cannot be interprocess. `interprocess` is what
    // makes the driver name shared memory for this process.
    let event = Event::new(EventFlags {
        timing: false,
        interprocess: true,
        external: false,
    });
    event.record(&stream)?;
    stream.synchronize()?;

    let handle = event.ipc_handle()?;
    let hex: String = handle.0.iter().map(|byte| format!("{byte:02x}")).collect();
    println!("{hex}");
    std::thread::sleep(Duration::from_secs(3600));
    Ok(())
}
