//! Call-level helpers every capability runtime shares: which rank reports
//! a component's host results, and host-gated call predicates.

use uniserve_cuda::Stream;
use uniserve_ipc::{Call, TensorRef};

use crate::ffi::Device;
use crate::worker::RankStorage;
use crate::{Error, Result};

/// The rank that reports a component's host results: its first member.
pub(crate) fn reports_output(
    info: &uniserve_ipc::WorkerInfo,
    rank: usize,
    component: &str,
) -> Result<bool> {
    match info.components.iter().find(|value| value.name == component) {
        Some(value) => Ok(value.config.ranks.first() == Some(&rank)),
        None if info.world_size == 1 => Ok(true),
        None => Err(Error::Unsupported(format!(
            "component {component:?} has no output rank"
        ))),
    }
}

/// Read a host-gated (U8) predicate after its producer's fence.
pub(crate) fn read_flag(
    storage: &RankStorage,
    call: &Call,
    predicate: &TensorRef,
    device: Option<i32>,
    stream: Option<&Stream>,
) -> Result<bool> {
    let read =
        storage
            .tensors
            .consume(predicate, call.call_id, device.map(Device::Cuda), stream)?;
    let mut value = [0_u8; 1];
    let copied = match stream {
        Some(stream) => {
            // SAFETY: one byte of a committed product, read after the stream
            // waited on its producer; the host byte outlives the
            // synchronized copy.
            unsafe {
                uniserve_cuda::memcpy_async(
                    value.as_mut_ptr() as u64,
                    read.value().address(),
                    1,
                    stream,
                )
            }
            .map_err(Error::from)
            .and_then(|()| stream.synchronize().map_err(Error::from))
        }
        None => {
            // SAFETY: a committed host product byte.
            value[0] = unsafe { *(read.value().address() as *const u8) };
            Ok(())
        }
    };
    storage.tensors.complete_reads(&[read], stream, &[])?;
    copied?;
    match value[0] {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(Error::Invalid(
            "call predicate is not a canonical boolean".into(),
        )),
    }
}
