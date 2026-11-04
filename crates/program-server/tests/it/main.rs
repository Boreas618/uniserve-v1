//! The Program Server's behavior: the module store of design section 5.3, the
//! compile queue of section 6.2, the package manifest of section 5.2, the
//! protocol and client of section 6.3, and the server's four operations.

#![cfg(test)]

mod builder;
mod manifest;
mod program;
mod protocol;
mod server;
mod store;
