//! The process half of the device test
//! `gpu::resetting_primary_contexts_releases_interprocess_event_storage`
//! (`tests/it/gpu.rs`), which starts this executable as its child
//! (`CARGO_BIN_EXE_interprocess-event`).
//!
//! It exports an interprocess event on device 0 as a rank does, prints one
//! `mapped <path>` line for each POSIX shared-memory object the driver mapped
//! for it, destroys its primary contexts and exits. The test then checks that
//! none of those objects outlived the process.

use std::collections::BTreeSet;
use std::process::ExitCode;

use platform::{DeviceId, Event, EventFlags, PrimaryContext, Stream};

/// The device the event is created on.
const DEVICE: DeviceId = DeviceId(0);

/// The POSIX shared-memory objects this process maps that still have a name.
///
/// The driver backs a context's interprocess events with shared memory it
/// creates with `shm_open`, which glibc places under its shared-memory mount;
/// the kernel's list of the process's mappings (`/proc/self/maps`) names the
/// objects mapped there, and `(deleted)` marks one already unlinked.
fn mapped_shared_memory() -> std::io::Result<BTreeSet<String>> {
    Ok(std::fs::read_to_string("/proc/self/maps")?
        .lines()
        .filter_map(|line| line.find("/dev/shm/").map(|start| &line[start..]))
        .filter(|path| !path.ends_with(" (deleted)"))
        .map(str::to_owned)
        .collect())
}

fn run() -> Result<(), String> {
    // A reference this process never releases, as the CUDA runtime holds one
    // in a rank: without it, the last owner's release below would destroy the
    // context by itself.
    let runtime = PrimaryContext::retain(DEVICE).map_err(|error| error.to_string())?;
    let _runtime = std::mem::ManuallyDrop::new(runtime);
    let stream = Stream::new(DEVICE).map_err(|error| error.to_string())?;
    let before = mapped_shared_memory().map_err(|error| error.to_string())?;
    let event = Event::new(EventFlags {
        interprocess: true,
        ..EventFlags::default()
    });
    event.record(&stream).map_err(|error| error.to_string())?;
    event.ipc_handle().map_err(|error| error.to_string())?;
    stream.synchronize().map_err(|error| error.to_string())?;
    let after = mapped_shared_memory().map_err(|error| error.to_string())?;
    for path in after.difference(&before) {
        println!("mapped {path}");
    }

    drop(event);
    drop(stream);
    // SAFETY: this process uses no device after the reset.
    let devices = unsafe { PrimaryContext::reset_all() }.map_err(|error| error.to_string())?;
    if !devices.contains(&DEVICE) {
        return Err(format!("the reset destroyed no context of {DEVICE}"));
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("interprocess-event: {message}");
            ExitCode::FAILURE
        }
    }
}
