//! The request semantics of UniServe's model families.
//!
//! Each module states the rules one family applies to its requests, as pure
//! functions over plain values: checkpoint facts, input rules and planning.
//! The server plans and presents requests with them before admission, and
//! the worker derives what it admits from the same rules, so each rule is
//! written once. Nothing here reads files, opens connections or knows the
//! serving loop.

pub mod video;
