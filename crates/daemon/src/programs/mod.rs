//! The policy package's programs on the leader daemon (`specs/configuration.md` §4), and the KV
//! program it hands every node's Storage Server ([`kv`]).

pub(crate) mod kv;

pub(crate) use self::kv::KvSetup;
