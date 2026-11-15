//! The Build Server's behavior: the module store of design section 5.3, the
//! compile queue of section 6.2, the package manifest of section 5.2, the
//! protocol and client of section 6.3, the server's operations, and the
//! command that configures and runs it.

#![cfg(test)]

mod build;
mod builder;
mod command;
mod cover;
mod manifest;
mod protocol;
mod serve;
mod server;
mod store;
mod support;
