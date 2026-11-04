//! Holds an interprocess event on device 0 until it is killed, which is what a
//! supervised rank does when a test checks that the supervisor accounts for the
//! driver's shared memory per rank.
//!
//! `crates/daemon`'s supervisor test needs a *separate process* with its own
//! primary context that makes the driver create the shared memory it names for
//! that process — the thing under test is the supervisor reading those names
//! after a rank dies. A Python stand-in did that with `ctypes` against
//! `libcuda.so.1`; this binary does the same through `platform`'s types, so no
//! crate outside `platform` has to call the driver.
//!
//!     cargo run -p platform --example interprocess_event_holder
//!
//! It prints the event's 64-byte interprocess handle as hex on one line and
//! then sleeps for an hour, so a test can pass the handle to a second process
//! (`Event::open_ipc`) or simply read the shared memory the driver created for
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
