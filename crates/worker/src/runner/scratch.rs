//! `Scratches`: the shared work areas of the rank's streams.
//!
//! Execution contexts serialized on one stream borrow one Python `Scratch`
//! (`uniserve.bindings.context.Scratch`) per device: a transient
//! work area grows to the largest context prepared and keeps what captured
//! graphs address. Rust holds the scratch of each stream and device and
//! passes it to every runner constructed on that stream, so their contexts
//! share it. A device scratch allocates from a pool of device memory Rust
//! owns ([`GraphPool`]), of its own, so that no graph's activations share
//! its blocks and its addresses stay fixed while graphs address it.

use cuda::DeviceId;
use std::collections::BTreeMap;
use std::sync::Arc;

use crate::Result;
use crate::cuda_graph::GraphPool;
use crate::ffi::{Handle, Interpreter};

/// A Python `Scratch`, with the pool it allocates from when it is a device
/// scratch.
type Scratch = (Handle, Option<Arc<GraphPool>>);

/// One Python `Scratch` per stream handle and CUDA device (`None` for the
/// host).
#[derive(Default)]
pub(crate) struct Scratches {
    scratches: BTreeMap<(usize, Option<DeviceId>), Scratch>,
}

impl Scratches {
    /// The scratch of `stream` on `device`, created at its first use.
    pub(crate) fn get(&mut self, stream: usize, device: Option<DeviceId>) -> Result<Handle> {
        let key = (stream, device);
        if let Some((scratch, _)) = self.scratches.get(&key) {
            return Ok(scratch.clone());
        }
        let pool = device.map(GraphPool::new).transpose()?.map(Arc::new);
        let scratch = Handle::new(
            Interpreter::import("uniserve.bindings.context.Scratch")?.call_values(&[
                pool.as_ref()
                    .map_or_else(tvm_ffi::Any::default, |pool| pool.value.to_any()),
                device.map_or_else(tvm_ffi::Any::default, |device| {
                    tvm_ffi::Any::from(i64::from(device.0))
                }),
            ])?,
        )?;
        self.scratches.insert(key, (scratch.clone(), pool));
        Ok(scratch)
    }

    /// The pools of the device scratches.
    pub(crate) fn pools(&self) -> Vec<Arc<GraphPool>> {
        self.scratches
            .values()
            .filter_map(|(_, pool)| pool.clone())
            .collect()
    }
}
