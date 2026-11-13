//! The packages a main process installs: which kernel packages a generation takes from the
//! deployment and this host's Build Server store ([`select`]), and how their modules and the
//! scheduling policy's are resolved and loaded ([`load`]).

pub(crate) mod load;
pub(crate) mod select;
