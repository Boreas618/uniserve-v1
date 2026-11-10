//! A device bench: one GPU, its stream, and the Program Server that builds and resolves the
//! kernels a forward launches.
//!
//! A case runs its forward through [`Bench::run`]: a recording forward names the programs it
//! launches, the Program Server prepares and resolves every one not yet installed, and a launching
//! forward runs it on the device. The device is `CUDA_VISIBLE_DEVICES`' first and the server the
//! one whose socket `UNISERVE_PROGRAM_SERVER_SOCKET` names; a case that needs neither does not
//! open a bench.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Read as _;
use std::os::fd::AsRawFd as _;
use std::path::Path;

use nn::{Arena, Backing, DType, DevicePtr, Forward, Shape, TensorView, Weight};
use ops::{Generation, KernelEntry};
use platform::{DeviceAllocation, DeviceId, Stream};
use program_server::{
    Client, Descriptors, ModuleForm, PrepareRequest, ProgramRequest, Selector, Target,
};
use proto::program::ProgramSlot;

use super::{Outcome, encode};

const DEVICE: DeviceId = DeviceId(0);

/// Bytes of the arena a forward may reserve scratch from.
const ARENA_BYTES: usize = 64 << 20;

/// The programs a recording walk named, as `(operator, key)` pairs.
#[derive(Default)]
struct Launches(Vec<(&'static str, String)>);

impl ops::capture::Sink for Launches {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.push((operator, key.to_owned()));
    }
}

/// One device, its stream, the Program Server that supplies its kernels, and every buffer a
/// case allocated, freed when the bench drops.
pub(crate) struct Bench {
    device: platform::Device,
    stream: Stream,
    client: Client,
    target: Target,
    /// The backend each installed operator's package is, from the store's listing.
    backends: BTreeMap<String, Vec<String>>,
    /// Every kernel installed so far, by operator and key: each case asks only for the ones
    /// it has not met.
    kernels: HashMap<(&'static str, String), KernelEntry>,
    arena: Arena,
    buffers: Vec<DeviceAllocation>,
}

impl Bench {
    pub(crate) fn open() -> Outcome<Self> {
        let socket = std::env::var_os("UNISERVE_PROGRAM_SERVER_SOCKET").ok_or(
            "UNISERVE_PROGRAM_SERVER_SOCKET names no Program Server: start one with \
             `uniserve program-server serve` (see this file's header)",
        )?;
        let device = platform::Device::new(DEVICE)?;
        let stream = Stream::new(DEVICE)?;
        let target = Target::arch(DEVICE.compute_capability()?);
        let mut client = Client::connect(Path::new(&socket))?;
        let mut backends: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for package in client.list()?.packages {
            let known = backends.entry(package.op).or_default();
            if !known.contains(&package.backend) {
                known.push(package.backend);
            }
        }
        let scratch = DeviceAllocation::linear(DEVICE, ARENA_BYTES)?;
        let arena = Arena::new(DevicePtr::new(scratch.address()), ARENA_BYTES as u64);
        Ok(Self {
            device,
            stream,
            client,
            target,
            backends,
            kernels: HashMap::new(),
            arena,
            buffers: vec![scratch],
        })
    }

    /// A new device tensor of `dtype` and `dims`, row-major, its contents undefined.
    pub(crate) fn tensor(&mut self, dtype: DType, dims: &[u64]) -> Outcome<TensorView> {
        let shape = Shape::new(dims);
        let bytes = shape.numel() * dtype.bytes().ok_or("an element type without a width")?;
        let buffer = DeviceAllocation::linear(DEVICE, bytes.max(1) as usize)?;
        let view = TensorView::new(DevicePtr::new(buffer.address()), dtype, shape);
        self.buffers.push(buffer);
        Ok(view)
    }

    /// Copies `bytes` into the device at `address`.
    pub(crate) fn write(&self, address: DevicePtr, bytes: &[u8]) -> Outcome {
        // SAFETY: the destination is a live allocation of at least `bytes.len()` bytes, the
        // source is this slice, and the stream is synchronized before the slice can go.
        unsafe {
            platform::memcpy_async(
                address.address(),
                bytes.as_ptr() as u64,
                bytes.len(),
                &self.stream,
            )?;
        }
        self.stream.synchronize()?;
        Ok(())
    }

    /// A device tensor holding `bytes`.
    pub(crate) fn upload(
        &mut self,
        dtype: DType,
        dims: &[u64],
        bytes: &[u8],
    ) -> Outcome<TensorView> {
        let view = self.tensor(dtype, dims)?;
        if view.byte_len()? != bytes.len() as u64 {
            return Err(format!("{} bytes for a {dtype} tensor of {dims:?}", bytes.len()).into());
        }
        self.write(view.ptr, bytes)?;
        Ok(view)
    }

    /// `values` rounded to `dtype` in a new device tensor.
    pub(crate) fn values(
        &mut self,
        dtype: DType,
        dims: &[u64],
        values: &[f32],
    ) -> Outcome<TensorView> {
        let bytes = encode(dtype, values)?;
        self.upload(dtype, dims, &bytes)
    }

    /// A device weight holding `values` rounded to `dtype`.
    pub(crate) fn weight(&mut self, dtype: DType, dims: &[u64], values: &[f32]) -> Outcome<Weight> {
        let view = self.values(dtype, dims, values)?;
        Ok(Weight {
            backing: Backing::new(view.ptr, view.byte_len()?),
            view,
        })
    }

    /// The bytes of a row-major device tensor, after everything queued before.
    pub(crate) fn download(&self, view: TensorView) -> Outcome<Vec<u8>> {
        if !view.is_row_major() {
            return Err("a strided view is read through the tensor it lies in".into());
        }
        let mut bytes = vec![0_u8; view.byte_len()? as usize];
        // SAFETY: the source is a live device range of `bytes.len()` bytes and the destination
        // this vector; the stream is synchronized before the vector is read.
        unsafe {
            platform::memcpy_async(
                bytes.as_mut_ptr() as u64,
                view.ptr.address(),
                bytes.len(),
                &self.stream,
            )?;
        }
        self.stream.synchronize()?;
        Ok(bytes)
    }

    /// Runs `walk` on the device: a recording forward names its programs, the Program Server
    /// prepares and resolves the ones not yet installed, and a launching forward runs it.
    pub(crate) fn run(&mut self, walk: &dyn Fn(&Forward<'_>) -> nn::Result<()>) -> Outcome {
        let counting = Arena::counting();
        let mut launches = Launches::default();
        ops::capture::set_sink(Some(&mut launches));
        let recorded = walk(&Forward::recording(&counting));
        ops::capture::set_sink(None);
        recorded?;

        let mut missing: BTreeMap<&'static str, BTreeSet<String>> = BTreeMap::new();
        for (operator, key) in launches.0 {
            if !self.kernels.contains_key(&(operator, key.clone())) {
                missing.entry(operator).or_default().insert(key);
            }
        }
        if !missing.is_empty() {
            self.install(&missing)?;
        }

        let generation = Generation::new(1, self.kernels.values().cloned());
        walk(&Forward::new(&self.stream, &generation, &self.arena))?;
        self.stream.synchronize()?;
        Ok(())
    }

    /// Prepares every missing program and installs each key's module into this device.
    fn install(&mut self, missing: &BTreeMap<&'static str, BTreeSet<String>>) -> Outcome {
        let mut programs = Vec::new();
        for operator in missing.keys() {
            let backend = match self.backends.get(*operator).map(Vec::as_slice) {
                Some([only]) => only.clone(),
                Some(several) => {
                    return Err(format!(
                        "{operator} has the packages {several:?}: name the one to measure"
                    )
                    .into());
                }
                None => return Err(format!("the store has no package of {operator}").into()),
            };
            programs.push(ProgramRequest {
                slot: ProgramSlot::all(*operator),
                selector: Selector::Latest {
                    backend,
                    arguments: Default::default(),
                },
            });
        }
        let request = PrepareRequest {
            targets: vec![self.target.clone()],
            programs,
            descriptors: missing
                .iter()
                .map(|(operator, keys)| Descriptors {
                    slot: ProgramSlot::all(*operator),
                    keys: keys.iter().cloned().collect(),
                })
                .collect(),
        };
        let prepared = self.client.prepare(&request, &mut |_| {})?;
        for chosen in prepared.programs {
            let operator = proto::program::operator::named(&chosen.slot.op)
                .ok_or_else(|| format!("{} is no operator of this build", chosen.slot.op))?;
            for key in missing.get(operator).into_iter().flatten() {
                let module = self.client.resolve(
                    operator,
                    &chosen.program,
                    &self.target,
                    key,
                    &mut |_| {},
                )?;
                let entry = load(&self.device, operator, key, module)?;
                self.kernels.insert((operator, key.clone()), entry);
            }
        }
        Ok(())
    }
}

/// Loads one resolved module into `device` as the kernel of `operator` for `key`.
fn load(
    device: &platform::Device,
    operator: &str,
    key: &str,
    module: program_server::Module,
) -> Outcome<KernelEntry> {
    let read = |file: &program_server::ModuleFile| -> Outcome<Vec<u8>> {
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
            // The kernel's symbol is its metadata's name; without metadata `ops` takes the
            // module's single entry.
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
                .get(program_server::manifest::LAUNCH_ENTRY)
                .ok_or("a native module states no launch entry")?;
            // The library is the file that defines the entry; it is loaded where the store
            // keeps it, which the handle the server sent still names.
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
