//! Prints what `PrimaryContext::sticky_error` answers once a context has gone
//! sticky, which is the evidence for design 2.4's watchdog.
//!
//! Run it by hand on the commit under test:
//!
//! ```text
//! cargo run -p platform --example sticky_probe
//! ```
//!
//! It cannot live in the test binary: a sticky error is fatal to the process
//! that holds it, so this program runs as its own process and exits 0 whether
//! or not the context was poisoned — what it prints is the observation.

use std::ffi::c_void;

use platform::{Device, DeviceId, Error, LaunchConfig, Library, PrimaryContext, Stream};

/// Stores through the pointer it is handed, so a launch with an address no
/// mapping backs faults on the device and poisons the context.
const STORE_BAD_ADDRESS: &str = r#"
.version 8.6
.target sm_90
.address_size 64

.visible .entry store_bad_address(.param .u64 target)
{
    .reg .b64 %target;
    ld.param.u64 %target, [target];
    st.global.u32 [%target], 1;
    ret;
}
"#;

fn main() {
    if let Err(error) = run() {
        eprintln!("the probe could not reach the fault: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    let device = Device::new(DeviceId(0))?;
    let context = PrimaryContext::retain(device.id())?;
    let stream = Stream::new(device.id())?;
    let library = Library::load(&device, STORE_BAD_ADDRESS.as_bytes())?;
    let kernel = library.kernel("store_bad_address")?;

    // Address 0x1000: on a device with a 4 KiB page and no small-address
    // mapping, nothing is mapped there, so the store is an invalid access.
    let mut target: u64 = 0x1000;
    let mut parameters = [std::ptr::addr_of_mut!(target).cast::<c_void>()];
    // `parameters` holds the one address the kernel takes; the storage it
    // points at is meant to be invalid, since the error is what this probe
    // observes.
    if let Err(error) = kernel.launch_ex(
        &stream,
        &LaunchConfig::new([1, 1, 1], [1, 1, 1]),
        &mut parameters,
    ) {
        // The launch is expected to be accepted — the fault is asynchronous —
        // but a refusal is still an observation worth printing.
        println!("the launch itself was refused: {error}");
        return Ok(());
    }

    // Synchronizing surfaces the fault, and from then on the context is
    // poisoned. A clean answer here means the store did not fault.
    match stream.synchronize() {
        Ok(()) => println!("synchronize: no fault, the store reached mapped memory"),
        Err(error) => println!("synchronize: {error}"),
    }

    match context.sticky_error() {
        Ok(None) => println!("sticky_error: none, the context is healthy"),
        Ok(Some(error)) => println!("sticky_error: {error}"),
        Err(error) => println!("sticky_error could not answer: {error}"),
    }
    Ok(())
}
