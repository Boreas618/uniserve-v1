//! A stand-in rank for the supervisor's tests: it holds an interprocess event on
//! device 0, which makes the driver create the shared memory it names for this
//! process, and sleeps until it is killed.
//!
//! The test needs a *separate process* with its own primary context — that is
//! what it checks the supervisor can account for — so the rank cannot be the
//! test process itself. It goes through `platform`'s types rather than binding
//! the driver, which keeps `platform` the only crate that calls CUDA.

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
    // The test polls for that shared memory; nothing has to be printed.
    std::thread::sleep(Duration::from_secs(600));
    Ok(())
}
