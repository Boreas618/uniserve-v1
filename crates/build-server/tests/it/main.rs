//! The Build Server's behavior: the module store of design section 5.3, the
//! compile queue of section 6.2, the package manifest of section 5.2, the
//! protocol and client of section 6.3, and the server's operations.

#![cfg(test)]

mod build;
mod builder;
mod command;
mod cover;
mod manifest;
mod module;
mod protocol;
mod serve;
mod server;
mod store;
