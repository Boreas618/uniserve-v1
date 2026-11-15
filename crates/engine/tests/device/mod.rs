//! What the device cases share: the kernels a generation holds, taken from a running Build
//! Server through the engine's own module load (`engine::modules::load`), and device tensors with
//! stated values.
//!
//! The device is GPU 0 of the process (`CUDA_VISIBLE_DEVICES`' first) and the server the one
//! whose socket `UNISERVE_BUILD_SERVER_SOCKET` names (`uniserve build-server serve`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use build_server::{Client, Target};
use engine::modules::load;
use ops::Generation;
use platform::{DeviceAllocation, DeviceId, Stream};
use proto::module::Packages;
use tensor::{DType, Tensor};

pub(crate) const DEVICE: DeviceId = DeviceId(0);

/// A failure of the bench itself: a missing server, a module that does not load.
pub(crate) type Outcome<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// The Build Server's client and the architecture of the device modules are loaded into.
pub(crate) struct Kernels {
    client: Client,
    target: Target,
}

impl Kernels {
    /// Connects to the Build Server the environment names.
    pub(crate) fn open() -> Outcome<Self> {
        let socket = std::env::var_os("UNISERVE_BUILD_SERVER_SOCKET").ok_or(
            "UNISERVE_BUILD_SERVER_SOCKET names no Build Server: start one with `uniserve \
             build-server serve`",
        )?;
        Ok(Self {
            client: Client::connect(Path::new(&socket))?,
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
    let allocation = DeviceAllocation::linear(DEVICE, bytes.len())?;

    // SAFETY: the destination is the allocation just made for exactly these bytes, the source
    // is this vector, and the stream is waited for before the vector goes.
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
        DType::F32,
        &ops::cuda(DEVICE.0 as usize),
    )?;
    Ok((allocation, tensor))
}
