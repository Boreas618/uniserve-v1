//! The records of the engine interface's oldest version, as a build of that version wrote them
//! (`tests/fixtures/engine-interface-14/`), which this build must read and must write back in no
//! field that build does not read (`docs/architecture.md`, Interface versions).
//!
//! The files are never written by a later build: the change that raises the window's oldest
//! version replaces them with the records a build of the new oldest version writes. A build of
//! version 14 wrote them from a deployment's default options with
//! `kv_cache_config.boundary_interval` stated (128):
//!
//! - `launch.json`: the launch of engine 7, a standby of epoch 12 on the scheduler node of a
//!   world of two hosts, as JSON: the bytes of the spawn request's `launch` and of the memfd the
//!   engine reads.
//! - `prepare.msgpack`, `takeover.msgpack`: the options and the takeover of a hot restart of one
//!   replica serving that launch, with a started daemon service's handoff, as the payloads of
//!   the handoff socket's frames (MessagePack with named fields).

use std::collections::BTreeSet;

use serde_json::Value;

/// The launch of version 14, as JSON.
pub(crate) const LAUNCH: &[u8] =
    include_bytes!("../../tests/fixtures/engine-interface-14/launch.json");

/// A hot restart's options of version 14, as the handoff socket's MessagePack.
pub(crate) const PREPARE: &[u8] =
    include_bytes!("../../tests/fixtures/engine-interface-14/prepare.msgpack");

/// A hot restart's takeover of version 14, as the handoff socket's MessagePack.
pub(crate) const TAKEOVER: &[u8] =
    include_bytes!("../../tests/fixtures/engine-interface-14/takeover.msgpack");

/// The fields `written` states that `oldest` does not, by their dotted paths: what this build
/// writes beyond what a build of the oldest version wrote for the same record, which that build
/// does not read and, in a configuration section, refuses. A field is written only when it holds
/// another value than its default, so a record decoded from `oldest` and written back states
/// none.
pub(crate) fn unread(written: &Value, oldest: &Value) -> Vec<String> {
    paths(written).difference(&paths(oldest)).cloned().collect()
}

/// The dotted path of every field of `value`'s objects; an array's elements share its path.
fn paths(value: &Value) -> BTreeSet<String> {
    fn walk(value: &Value, path: &str, paths: &mut BTreeSet<String>) {
        match value {
            Value::Object(fields) => {
                for (name, field) in fields {
                    let path = if path.is_empty() {
                        name.clone()
                    } else {
                        format!("{path}.{name}")
                    };
                    walk(field, &path, paths);
                    paths.insert(path);
                }
            }
            Value::Array(items) => {
                for item in items {
                    walk(item, path, paths);
                }
            }
            _ => {}
        }
    }

    let mut paths = BTreeSet::new();
    walk(value, "", &mut paths);
    paths
}
