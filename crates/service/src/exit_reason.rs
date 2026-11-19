//! A supervised process's exit reason: the error it states as it ends, for the daemon that
//! supervises it, as a container states its termination message for its kubelet.
//!
//! A process can fail before anything else reports its failure: an engine that cannot read its
//! launch ends before it dials its daemon or starts its log. The daemon therefore gives every
//! process it starts an anonymous file of its own ([`create`]), inherited at the descriptor the
//! environment variable [`VARIABLE`] names ([`inherit`]). The process takes the file as it starts
//! ([`take`]) and states its fatal error there as it ends ([`state`], and every panic once
//! [`report_panics`] is installed), each statement replacing the one before, so the file holds
//! the process's last account of its end. The daemon reads it when it reaps the process
//! ([`read`]) and carries it in the process's exit report.
//!
//! The file is bounded on both sides: a process states at most [`LIMIT`] bytes, and the daemon
//! reads no more whatever the process wrote. A process its supervisor passed no file (one started
//! by a daemon that predates the variable) states nothing, and a process that ignores the
//! variable leaves the file empty, so either side may be of another build.

use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd as _, FromRawFd as _, OwnedFd, RawFd};
use std::os::unix::fs::FileExt as _;
use std::os::unix::process::CommandExt as _;
use std::process::Command;
use std::sync::OnceLock;

/// The environment variable that names the descriptor of a process's exit reason file.
pub const VARIABLE: &str = "UNISERVE_EXIT_REASON_FD";

/// The most bytes of an exit reason: what a process states, and what its supervisor reads.
pub const LIMIT: usize = 4096;

/// What ends a reason cut to [`LIMIT`].
const CUT: &str = "…";

/// The exit reason file this process took ([`take`]), which [`state`] writes.
static FILE: OnceLock<File> = OnceLock::new();

/// A new exit reason file for one process to inherit ([`inherit`]): an empty anonymous file,
/// close-on-exec in this process.
///
/// # Errors
///
/// Fails when the file cannot be created.
pub fn create() -> io::Result<OwnedFd> {
    // SAFETY: the name is a NUL-terminated literal; a successful call returns a descriptor this
    // function owns.
    let descriptor =
        unsafe { libc::memfd_create(c"uniserve-exit-reason".as_ptr(), libc::MFD_CLOEXEC) };
    if descriptor < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the descriptor was just created and is owned by nothing else.
    Ok(unsafe { OwnedFd::from_raw_fd(descriptor) })
}

/// Has the process `command` starts inherit `file` at its descriptor number, which [`VARIABLE`]
/// names in its environment.
///
/// The caller keeps `file` open until the spawn returned, and goes on holding it to [`read`] the
/// process's reason. Close-on-exec is cleared only in the forked child: no other process this
/// one starts meanwhile inherits the file.
pub fn inherit(command: &mut Command, file: &OwnedFd) {
    let descriptor = file.as_raw_fd();
    command.env(VARIABLE, descriptor.to_string());
    // SAFETY: the closure runs in the forked child between fork and exec and only calls `fcntl`,
    // which is async-signal-safe; the caller holds the descriptor open across the spawn.
    unsafe {
        command.pre_exec(move || {
            if libc::fcntl(descriptor, libc::F_SETFD, 0) == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

/// What the process that inherited `file` stated of its end: at most [`LIMIT`] bytes of its last
/// statement, read as UTF-8 with any invalid sequence replaced and trimmed; empty when it stated
/// nothing or the file cannot be read.
pub fn read(file: &OwnedFd) -> String {
    let Ok(reading) = file.try_clone().map(File::from) else {
        return String::new();
    };
    let mut bytes = vec![0; LIMIT];
    let mut filled = 0;
    while filled < bytes.len() {
        match reading.read_at(&mut bytes[filled..], filled as u64) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(_) => break,
        }
    }
    bytes.truncate(filled);
    String::from_utf8_lossy(&bytes).trim().to_owned()
}

/// Takes the exit reason file this process's supervisor passed it, at the descriptor
/// [`VARIABLE`] names, so that [`state`] writes it; does nothing when the variable names none.
///
/// Call it first in `main`, before any thread starts: it removes the variable from the
/// environment, and the descriptor turns close-on-exec, so no process this one starts takes the
/// file for its own.
pub fn take() {
    let Some(descriptor) = std::env::var(VARIABLE)
        .ok()
        .and_then(|value| value.parse::<RawFd>().ok())
    else {
        return;
    };
    // SAFETY: no other thread exists yet to read the environment (the caller's obligation).
    unsafe { std::env::remove_var(VARIABLE) };
    // SAFETY: fcntl takes integers only; a descriptor this process does not hold fails the call.
    if unsafe { libc::fcntl(descriptor, libc::F_SETFD, libc::FD_CLOEXEC) } == -1 {
        return;
    }
    // SAFETY: the supervisor passed this descriptor for the exit reason, and nothing else in the
    // process owns it.
    let _ = FILE.set(unsafe { File::from_raw_fd(descriptor) });
}

/// States `reason` as this process's exit reason: the file holds it alone from here on, cut to
/// [`LIMIT`] bytes at a character boundary. Does nothing in a process that took no file
/// ([`take`]); a write that fails is ignored, since the process is ending and its log still
/// states the error.
pub fn state(reason: &str) {
    if let Some(file) = FILE.get() {
        write(file, reason);
    }
}

/// Writes `reason`, cut to [`LIMIT`] bytes, as `file`'s whole content.
fn write(file: &File, reason: &str) {
    let reason = cut(reason.trim());
    if file.write_all_at(reason.as_bytes(), 0).is_ok() {
        let _ = file.set_len(reason.len() as u64);
    }
}

/// `reason` when it fits in [`LIMIT`] bytes, and otherwise its longest prefix that fits with
/// [`CUT`] after it, ending at a character boundary.
fn cut(reason: &str) -> std::borrow::Cow<'_, str> {
    if reason.len() <= LIMIT {
        return reason.into();
    }
    let mut end = LIMIT - CUT.len();
    while !reason.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{CUT}", &reason[..end]).into()
}

/// States every panic of this process as its exit reason ([`state`]) before the panic hook
/// installed until now reports it: install it after the hook that reports panics to the log
/// (`logging::init`), which it keeps.
pub fn report_panics() {
    let report = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic| {
        let thread = std::thread::current();
        state(&format!(
            "thread {} {panic}",
            thread.name().unwrap_or("<unnamed>")
        ));
        report(panic);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A process's statements replace each other, and its supervisor reads the last one.
    #[test]
    fn the_supervisor_reads_the_last_statement() {
        let file = create().expect("an exit reason file");
        assert_eq!(read(&file), "");

        let writer = File::from(file.try_clone().expect("a second descriptor"));
        write(
            &writer,
            "reading the launch: an unknown field, expected one of many fields",
        );
        write(
            &writer,
            "  reading the launch: unknown field `boundary_interval`\n",
        );
        assert_eq!(
            read(&file),
            "reading the launch: unknown field `boundary_interval`"
        );
    }

    /// A reason longer than the limit is cut to it at a character boundary, marked as cut, and the
    /// supervisor reads no more than the limit of what a process wrote past it.
    #[test]
    fn a_reason_is_bounded_on_both_sides() {
        let file = create().expect("an exit reason file");
        let writer = File::from(file.try_clone().expect("a second descriptor"));

        // Two-byte characters across the limit: the cut falls between two of them.
        let long = "é".repeat(LIMIT);
        write(&writer, &long);
        let stated = read(&file);
        assert!(stated.len() <= LIMIT, "{} bytes", stated.len());
        assert!(stated.ends_with(CUT));
        assert!(stated.trim_end_matches(CUT).chars().all(|c| c == 'é'));

        // A process of another build may write past the limit: the supervisor reads the limit.
        writer
            .write_all_at(&vec![b'x'; 3 * LIMIT], 0)
            .expect("a write");
        assert_eq!(read(&file), "x".repeat(LIMIT));
    }
}
