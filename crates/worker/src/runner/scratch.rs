//! `Scratches`: the shared work areas of the rank's streams.
//!
//! Execution contexts serialized on one stream borrow one Python `Scratch`
//! (`uniserve.bindings.context.Scratch`) per device: a transient
//! work area grows to the largest context prepared and keeps what captured
//! graphs address. Rust holds the scratch of each stream and device and
//! passes it to every runner constructed on that stream, so their contexts
//! share it.

use std::collections::BTreeMap;

use crate::Result;
use crate::ffi::{Handle, Interpreter};

/// One Python `Scratch` per stream handle and device ordinal (`-1` for the
/// host).
#[derive(Default)]
pub(crate) struct Scratches {
    scratches: BTreeMap<(usize, i32), Handle>,
}

impl Scratches {
    /// The scratch of `stream` on `device`, created at its first use.
    pub(crate) fn get(&mut self, stream: usize, device: Option<i32>) -> Result<Handle> {
        let key = (stream, device.unwrap_or(-1));
        if let Some(scratch) = self.scratches.get(&key) {
            return Ok(scratch.clone());
        }
        let scratch = Handle::new(
            Interpreter::import("uniserve.bindings.context.Scratch")?.call_values(&[])?,
        )?;
        self.scratches.insert(key, scratch.clone());
        Ok(scratch)
    }
}
