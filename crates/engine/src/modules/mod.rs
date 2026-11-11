//! The kernel packages a main process installs: which packages a generation takes from the
//! deployment and this host's Build Server store ([`select`]), and how their modules are
//! resolved and loaded onto a device ([`load`]).

pub(crate) mod load;
pub(crate) mod select;
