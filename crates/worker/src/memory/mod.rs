//! The storage a rank holds outside its own process, and the weights it serves from.
//!
//! What an engine would lose by exiting, and that costs time or memory to rebuild, is storage
//! the host's Storage Server retains by descriptor: an engine maps it, a successor of the same
//! scope takes it again, and the server's quota keeps the epochs that map one GPU within one
//! limit. This module is the rank's side of that server.
//!
//! | Module | Holds |
//! | --- | --- |
//! | [`allocation`] | [`Storage`](allocation::Storage): one allocation the rank owns or maps, and the tensor over a window of it that execution lends to a forward |
//! | [`cache`] | [`KvCacheMemory`](cache::KvCacheMemory): the KV cache's planes, pooled page tables, host tier and scratch pages, lent to a causal forward as `nn::KvPlanes` and page-table tensors |
//! | [`client`] | [`StorageClient`](client::StorageClient): the rank's connection to the Storage Server, the exclusive storages it takes there and its share of each GPU's quota |
//! | [`exclusive`] | [`ExclusiveStorage`](exclusive::ExclusiveStorage): how an owner takes exclusive storage, at the server or local to the process |
//! | [`host_pages`] | [`HostPages`](host_pages::HostPages): a rank's mapping of one host page pool |
//! | [`division`] | How a GPU's quota is divided when a rank starts: the weights, the runners' regions, the scratch pool, the storages sized by request slots, and the cache's units from what remains |
//! | [`graph_pool`] | [`GraphPool`](graph_pool::GraphPool): the runners' regions and the scratch pool of one device, mapped at fixed addresses and reserved in the GPU's quota |
//! | [`weights`] | [`WeightStore`](weights::WeightStore): the retained weights of one device, which `nn::WeightLoader` publishes to and maps from |
//! | [`reload`] | A second load of a serving model: its order compared with the one that serves, and the load that publishes what changed for an update in place |
//! | [`state`] | [`StateMemory`](state::StateMemory): the live, checkpoint and trail buffers of a model's recurrent layers, and the copies between them and the cache's state pages |
//! | [`tables`] | [`BlockTables`](tables::BlockTables): the device block table of one rank of a worker group as its executor keeps it on the host, fed by the buffer pool's completions |
//!
//! ## Two kinds of device storage
//!
//! **Retained weights** are read-only and shared: one allocation per weight key and GPU,
//! mapped by every rank and epoch that loads the weight, with small weights packed into shared
//! allocations. A weight tensor is `backend::retained` over the mapping's address, so the
//! modules and graphs of a rank hold addresses that an update in place keeps while it changes
//! the bytes behind them.
//!
//! **Exclusive storages** have one writer: the cache planes, the encoder cache and the latent
//! buffers. Their owner takes them by name and length, resets what was created for it and
//! keeps what a predecessor left, and lends windows of them to forwards as
//! `backend::external` tensors.
//!
//! Host page pools are the third kind of storage the server keeps for a rank: host memory of
//! equal pages, which the rank maps, registers for device copies and copies pages to and from.
//!
//! ## What the process maps itself
//!
//! The memory a forward computes in is the process's own and is not retained: one region per
//! runner, which its captured graphs address, and the scratch pool. The rank reserves it in
//! the same quota, so the division of a GPU counts it beside the server's storages.

pub mod allocation;
pub mod cache;
pub mod client;
pub mod division;
pub mod exclusive;
pub mod graph_pool;
pub mod host_pages;
pub mod reload;
pub mod state;
pub mod tables;
pub mod weights;

use platform::DeviceId;
use uniserve_tensor::{DType, Shape, Tensor};

use crate::Result;

/// The tensor of `shape` elements of `dtype` at `address` of memory execution owns and lends to
/// a forward: on `device`, or a meta tensor, which has a shape and an element type and no data,
/// for an owner that holds no device.
///
/// # Safety
///
/// On a device, `address` is the first of the tensor's elements in memory the caller keeps
/// mapped, and unwritten by anything but the kernels launched on the tensor, until every tensor
/// over it is dropped and the stream work issued on them has completed. A meta tensor's address
/// is never read.
pub(crate) unsafe fn lend(
    address: u64,
    shape: impl Into<Shape>,
    dtype: DType,
    device: Option<DeviceId>,
) -> Result<Tensor> {
    let device = match device {
        Some(device) => backend::cuda(device.0 as usize),
        None => backend::meta(),
    };

    // SAFETY: forwarded from the caller.
    Ok(unsafe { backend::external(address, shape, dtype, &device) }?)
}

/// What the tests of this module's files share.
#[cfg(test)]
mod fixtures {
    use std::path::Path;

    use nn::FileIdentity;

    /// The identity of a checkpoint file named after `name` under `directory`, as a Hugging
    /// Face cache stores one: a file under `blobs/` is identified by its name alone, so the
    /// identity does not depend on when the file was written, and two calls of one name give
    /// one identity.
    pub(crate) fn identity(directory: &Path, name: &str) -> FileIdentity {
        let blobs = directory.join("blobs");
        std::fs::create_dir_all(&blobs).unwrap();

        // A blob's name is a 64-digit hexadecimal content hash; the fixture derives one from
        // the name it is given.
        let digest: String = name.bytes().map(|byte| format!("{byte:02x}")).collect();
        let path = blobs.join(format!("{digest:0<64.64}"));
        std::fs::write(&path, name).unwrap();

        FileIdentity::of(&path).unwrap()
    }
}
