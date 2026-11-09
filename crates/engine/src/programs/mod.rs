//! The kernel programs a main process installs: which programs a generation takes from the
//! deployment and this host's Program Server store ([`select`]), and how their modules are
//! resolved and loaded onto a device ([`modules`]).

pub(crate) mod modules;
pub(crate) mod select;
