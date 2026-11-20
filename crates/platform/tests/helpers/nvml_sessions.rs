//! The process half of the device test
//! `gpu::nvml_sessions_open_and_close_across_cupti_s_attachment`
//! (`tests/it/gpu.rs`), which starts this executable as its child
//! (`CARGO_BIN_EXE_nvml-sessions`).
//!
//! Several threads open and close NVML sessions while CUPTI attaches to the
//! CUDA driver, as an engine's starts measure their processes while a profiling
//! session's metadata initializes CUPTI's profiling. CUPTI attaches once per
//! process, so the order between the attachment and the sessions is exercised
//! only in a process of its own. Each session thread prints `sessions <count>`
//! once it ended; a thread that waits for the driver's lock forever prints
//! nothing, and the test's bound on this process ends it.

use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use platform::{DeviceId, Nvml};

/// The threads that open and close sessions.
const THREADS: usize = 4;

/// How long the sessions run before CUPTI attaches, and after it.
const BESIDE: Duration = Duration::from_millis(300);

fn main() -> ExitCode {
    let stop = Arc::new(AtomicBool::new(false));
    let threads: Vec<_> = (0..THREADS)
        .map(|_| {
            let stop = Arc::clone(&stop);
            thread::spawn(move || -> platform::Result<u64> {
                let mut sessions = 0;
                while !stop.load(Ordering::Acquire) {
                    drop(Nvml::init()?);
                    sessions += 1;
                    // A pause between sessions lets the attaching thread take its turn.
                    thread::sleep(Duration::from_millis(1));
                }
                Ok(sessions)
            })
        })
        .collect();

    // CUPTI attaches through its profiling initialization, which the chip name needs, while
    // the sessions open and close: what a profiling session's metadata reads of its device.
    thread::sleep(BESIDE);
    if let Err(error) = DeviceId(0)
        .uuid()
        .and_then(|_| platform::cupti::metrics::chip_name(0))
    {
        eprintln!("CUPTI did not attach: {error}");
        return ExitCode::FAILURE;
    }
    println!("attached");
    thread::sleep(BESIDE);
    stop.store(true, Ordering::Release);

    let mut failed = false;
    for thread in threads {
        match thread.join() {
            Ok(Ok(sessions)) => println!("sessions {sessions}"),
            Ok(Err(error)) => {
                eprintln!("a session failed: {error}");
                failed = true;
            }
            Err(_) => failed = true,
        }
    }
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
