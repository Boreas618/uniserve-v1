//! What the device cases share: the kernels a generation holds, taken from a running Build
//! Server through the engine's own module load (`engine::modules::load`), and device tensors with
//! stated values.
//!
//! The device is GPU 0 of the process (`CUDA_VISIBLE_DEVICES`' first) and the server the Build
//! Server of the deployment [`DEPLOYMENT`] on this host, which `uniserve build-server serve
//! --deployment device-cases` starts on its socket in the deployment's runtime directory
//! (`build_server::socket`).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use build_server::{Client, Target};
use engine::modules::load;
use ops::Generation;
use platform::{DeviceAllocation, DeviceId, Stream};
use proto::module::Packages;
use tensor::{DType, Tensor};

pub(crate) const DEVICE: DeviceId = DeviceId(0);

/// The deployment whose Build Server the device cases load their kernels from.
pub(crate) const DEPLOYMENT: &str = "device-cases";

/// A failure of the bench itself: a missing server, a module that does not load.
pub(crate) type Outcome<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// The Build Server's client and the architecture of the device modules are loaded into.
pub(crate) struct Kernels {
    client: Client,
    target: Target,
}

impl Kernels {
    /// Connects to the Build Server of [`DEPLOYMENT`] on this host.
    pub(crate) fn open() -> Outcome<Self> {
        let socket = build_server::socket(&service::RuntimeDirectory::of_user()?, DEPLOYMENT)?;
        let client = Client::connect(&socket).map_err(|error| {
            format!(
                "no Build Server serves on {}: start one with `uniserve build-server serve \
                 --deployment {DEPLOYMENT}` ({error})",
                socket.display()
            )
        })?;
        Ok(Self {
            client,
            target: Target::arch(DEVICE.compute_capability()?)?,
        })
    }

    /// Generation `number` holding one newly loaded kernel for each of `keys`, each from the
    /// package the Build Server selects for its key among those its store holds, as a
    /// deployment that names no backend has selected: every call loads its modules again, so two
    /// generations of the same keys hold distinct kernels of one implementation.
    pub(crate) fn generation(
        &mut self,
        number: u64,
        keys: &BTreeSet<(&'static str, String)>,
    ) -> Outcome<Arc<Generation>> {
        let mut wanted: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for (operator, key) in keys {
            wanted
                .entry((*operator).to_owned())
                .or_default()
                .insert(key.clone());
        }

        let targets = [self.target.clone()];
        let (packages, routes) =
            load::select(&mut self.client, &targets, &Packages::new(), &wanted)?;
        let entries = load::install(&mut self.client, DEVICE, &packages, &routes)?;
        Ok(Arc::new(Generation::new(number, entries)))
    }
}

/// Device memory holding `values` as FP32, and the retained tensor of `dims` over it. The
/// allocation is the tensor's storage: the caller keeps it for as long as the tensor is used.
pub(crate) fn weight(
    stream: &Stream,
    values: &[f32],
    dims: &[usize],
) -> Outcome<(DeviceAllocation, Tensor)> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    tensor(stream, &bytes, dims, DType::F32)
}

/// Device memory holding `bytes`, and the retained tensor of `dims` and `dtype` over it. The
/// allocation is the tensor's storage: the caller keeps it for as long as the tensor is used.
pub(crate) fn tensor(
    stream: &Stream,
    bytes: &[u8],
    dims: &[usize],
    dtype: DType,
) -> Outcome<(DeviceAllocation, Tensor)> {
    let allocation = DeviceAllocation::linear(DEVICE, bytes.len())?;

    // SAFETY: the destination is the allocation just made for exactly these bytes, the source
    // is the caller's slice, and the stream is waited for before the slice can go.
    unsafe {
        platform::memcpy_async(
            allocation.address(),
            bytes.as_ptr() as u64,
            bytes.len(),
            stream,
        )?;
    }
    stream.synchronize()?;

    let tensor = ops::retained(
        allocation.address(),
        dims,
        dtype,
        &ops::cuda(DEVICE.0 as usize),
    )?;
    Ok((allocation, tensor))
}
