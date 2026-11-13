//! What the device cases share: the kernels a generation holds, resolved at a running Build
//! Server, and device tensors with stated values.
//!
//! The device is GPU 0 of the process (`CUDA_VISIBLE_DEVICES`' first) and the server the one
//! whose socket `UNISERVE_BUILD_SERVER_SOCKET` names (`uniserve build-server serve`).

use std::collections::{BTreeMap, BTreeSet};
use std::io::Read as _;
use std::os::fd::AsRawFd as _;
use std::path::Path;
use std::sync::Arc;

use build_server::{
    Client, Descriptors, ModuleForm, PackageRequest, PrepareRequest, Selector, Target,
};
use ops::{Generation, KernelEntry};
use platform::{DeviceAllocation, DeviceId, Stream};
use proto::module::ModuleSlot;
use tensor::{DType, Tensor};

pub(crate) const DEVICE: DeviceId = DeviceId(0);

/// A failure of the bench itself: a missing server, a module that does not load.
pub(crate) type Outcome<T = ()> = Result<T, Box<dyn std::error::Error>>;

/// The Build Server's client and the device modules are loaded into.
pub(crate) struct Kernels {
    device: platform::Device,
    client: Client,
    target: Target,
    /// The backend of each operator's installed package, from the store's listing.
    backends: BTreeMap<String, Vec<String>>,
}

impl Kernels {
    /// Connects to the Build Server the environment names.
    pub(crate) fn open() -> Outcome<Self> {
        let socket = std::env::var_os("UNISERVE_BUILD_SERVER_SOCKET").ok_or(
            "UNISERVE_BUILD_SERVER_SOCKET names no Build Server: start one with `uniserve \
             build-server serve`",
        )?;
        let mut client = Client::connect(Path::new(&socket))?;

        let mut backends: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (package, _) in client.list()? {
            let known = backends.entry(package.op).or_default();
            if !known.contains(&package.backend) {
                known.push(package.backend);
            }
        }

        Ok(Self {
            device: platform::Device::new(DEVICE)?,
            client,
            target: Target::arch(DEVICE.compute_capability()?),
            backends,
        })
    }

    /// Generation `number` holding one newly loaded kernel for each of `keys`: every call
    /// loads its modules again, so two generations of the same keys hold distinct kernels of
    /// one implementation.
    pub(crate) fn generation(
        &mut self,
        number: u64,
        keys: &BTreeSet<(&'static str, String)>,
    ) -> Outcome<Arc<Generation>> {
        let mut wanted: BTreeMap<&'static str, BTreeSet<String>> = BTreeMap::new();
        for (operator, key) in keys {
            wanted.entry(operator).or_default().insert(key.clone());
        }

        let mut packages = Vec::new();
        for operator in wanted.keys() {
            let backend = match self.backends.get(*operator).map(Vec::as_slice) {
                Some([only]) => only.clone(),
                Some(several) => {
                    return Err(format!(
                        "{operator} has the packages {several:?}: the case needs exactly one"
                    )
                    .into());
                }
                None => return Err(format!("the store has no package of {operator}").into()),
            };
            packages.push(PackageRequest {
                slot: ModuleSlot::all(*operator),
                selector: Selector::Latest {
                    backend,
                    arguments: Default::default(),
                },
            });
        }

        let request = PrepareRequest {
            targets: vec![self.target.clone()],
            packages,
            descriptors: wanted
                .iter()
                .map(|(operator, keys)| Descriptors {
                    slot: ModuleSlot::all(*operator),
                    keys: keys.iter().cloned().collect(),
                })
                .collect(),
        };

        let mut entries = Vec::new();
        for (slot, package) in self.client.prepare(&request, &mut |_| {})? {
            let operator = proto::module::operator::named(&slot.op)
                .ok_or_else(|| format!("{} is no operator of this build", slot.op))?;

            for key in wanted.get(operator).into_iter().flatten() {
                let module =
                    self.client
                        .resolve(operator, &package, &self.target, key, &mut |_| {})?;
                entries.push(load(&self.device, operator, key, module)?);
            }
        }
        Ok(Arc::new(Generation::new(number, entries)))
    }
}

/// Loads one resolved module into `device` as the kernel of `operator` for `key`.
fn load(
    device: &platform::Device,
    operator: &str,
    key: &str,
    module: build_server::Module,
) -> Outcome<KernelEntry> {
    let read = |file: &build_server::ModuleFile| -> Outcome<Vec<u8>> {
        let mut bytes = Vec::with_capacity(file.size as usize);
        (&file.file).read_to_end(&mut bytes)?;
        Ok(bytes)
    };

    let entry = match module.form {
        ModuleForm::Cubin => {
            let mut image = None;
            let mut metadata = None;
            for file in &module.files {
                if module.metadata.as_deref() == Some(file.name.as_str()) {
                    metadata = Some(read(file)?);
                } else {
                    image = Some(read(file)?);
                }
            }
            let image = image.ok_or("a cubin module carries no image")?;

            // The kernel's symbol is its metadata's name.
            let symbol = match &metadata {
                Some(metadata) => ops::CubinMetadata::parse(metadata)?.name,
                None => String::new(),
            };
            ops::install(
                device,
                operator,
                key,
                ops::Module::Cubin {
                    image: &image,
                    metadata: metadata.as_deref(),
                    entry: &symbol,
                },
            )?
        }
        ModuleForm::Native => {
            let symbol = module
                .entries
                .get(build_server::manifest::LAUNCH_ENTRY)
                .ok_or("a native module states no launch entry")?;

            // The library is the file that defines the entry, loaded where the store keeps it.
            let mut library = None;
            for file in &module.files {
                let defines = platform::NativeLibrary::defined_functions(&read(file)?)
                    .is_some_and(|names| names.iter().any(|name| name == symbol));
                if defines || module.files.len() == 1 {
                    library = Some(std::fs::read_link(format!(
                        "/proc/self/fd/{}",
                        file.file.as_raw_fd()
                    ))?);
                }
            }
            let path = library.ok_or("no file of the native module defines its entry")?;
            ops::install(
                device,
                operator,
                key,
                ops::Module::Native {
                    path: &path,
                    entry: symbol,
                },
            )?
        }
        other => return Err(format!("a {other:?} module is not a kernel").into()),
    };
    entry.ok_or_else(|| format!("{operator} is no operator of this build").into())
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

    let tensor = backend::retained(
        allocation.address(),
        dims,
        DType::F32,
        &backend::cuda(DEVICE.0 as usize),
    )?;
    Ok((allocation, tensor))
}
