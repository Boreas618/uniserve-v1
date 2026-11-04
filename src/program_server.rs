//! The `uniserve program-server` command (design sections 6.1 and 6.3).
//!
//! One Program Server runs per host, beside the Storage Server. It compiles
//! program packages into the modules of their descriptors, keeps them in the
//! module store, and answers `install`, `prepare`, `resolve` and `list` on its
//! Unix socket. It launches no kernel and allocates no device memory, and it is
//! not on the serving path: a main process asks it for modules when it starts
//! and when a program changes, so a restart of the Program Server delays only
//! those two moments.
//!
//! The command's implementation is the `program_server` crate's
//! [`command`](program_server::command) module; this module mounts it into the
//! `uniserve` executable:
//!
//! ```ignore
//! let matches = uniserve::program_server::command()
//!     .get_matches_from(std::env::args_os().skip(1));
//! uniserve::program_server::run(uniserve::program_server::options(&matches)?)?;
//! ```
//!
//! The executable also gives the server the devices this host builds for
//! (design section 5.3): only the executable queries the platform crate
//! (design section 4.1), so `serve --target` names them, and a caller that
//! already holds the device list fills
//! [`ServeOptions::devices`](program_server::command::ServeOptions::devices)
//! instead.

pub use program_server::command::{
    Action, Options, ServeOptions, command, default_socket, options, run,
};
