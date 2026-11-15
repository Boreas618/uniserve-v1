//! Loaded kernels, their metadata, and the launch of a resolved kernel or the call of a native
//! entry, which [`launch()`](crate::launch()) dispatches and `platform` issues to the driver.
//!
//! A [`Kernel`] is a module that has been loaded: a cubin loaded into a device's context, or a
//! native library opened with the C entry it declares. A
//! [`Generation`](crate::Generation) holds one per (operator name, descriptor key) and is immutable
//! once built, so a kernel stays loaded exactly as long as a generation or a captured graph holds
//! it.
//!
//! ## What a cubin launch takes from here and what from the package
//!
//! The grid is the operator's: it follows from the arguments. The thread-block shape, dynamic shared
//! memory, cluster size, programmatic dependent launch, cooperative launch and global scratch come
//! from the module's own metadata, which is Triton's JSON field for field, so
//! a package cannot declare one launch shape and produce a module for another.
//!
//! ## The native entry ABI
//!
//! A native package declares one C entry per descriptor set:
//!
//! ```c
//! int32_t <entry>(const <CallStruct>* call, void* stream);
//! ```
//!
//! It returns 0 on success and nonzero on a parameter or driver error, with a NUL-terminated
//! message in the call structure's `error` field when it carries one. The structure is
//! `#[repr(C)]` in the operator's module and mirrored by hand in the package's header; each
//! operator pins its layout with a `const` assertion and a test. An entry uses only the stream it
//! is given, never allocates, synchronizes or reads device memory on the host, and is safe to call
//! while the stream is capturing.

use std::ffi::c_void;

use platform::{
    Cluster, Device, DeviceAttribute, DevicePtr, Function, FunctionAttribute, LaunchConfig,
    Library, NativeLibrary, Stream,
};

use crate::operand::Operand;
use crate::{Error, Result};

/// The raw stream handle a native entry receives (`void* stream`).
fn stream_handle(stream: &Stream) -> *mut c_void {
    stream.raw() as *mut c_void
}

/// The signature every native entry has.
pub type Entry = unsafe extern "C" fn(call: *const c_void, stream: *mut c_void) -> i32;

/// A `#[repr(C)]` call structure of a native package that reports its errors as text.
///
/// The default implementation answers `None`, for an entry that reports nothing beyond its status;
/// a structure with a `char error[N]` field (the convention of the trtllm-gen and cuDNN packages)
/// returns the message it finds.
pub trait NativeCall {
    /// The NUL-terminated message the entry wrote, or `None` when it wrote none.
    fn message(&self) -> Option<&str> {
        None
    }
}

/// The call of a native module's load entry ([`crate::LOAD_ENTRY`]): the directory of the module's
/// files, for a module that reads files published beside its library (the cubins a trtllm-gen
/// module launches). The package's header mirrors it as `uniserve_trtllm_load_call`.
#[repr(C)]
pub struct LoadCall {
    /// The NUL-terminated path of the directory the module's files were published in.
    pub directory: *const std::ffi::c_char,
    /// The entry's NUL-terminated error message.
    pub error: [u8; 512],
}

const _: () = assert!(std::mem::offset_of!(LoadCall, error) == 8);
const _: () = assert!(std::mem::size_of::<LoadCall>() == 520);

impl NativeCall for LoadCall {
    fn message(&self) -> Option<&str> {
        let end = self.error.iter().position(|&byte| byte == 0)?;
        std::str::from_utf8(&self.error[..end]).ok()
    }
}

/// A module's launch metadata, in Triton's field names.
///
/// Triton writes this JSON beside every cubin it compiles, and the Build Server writes the same
/// record for a cubin it compiles with nvcc; `ops` reads those fields directly rather than
/// defining a manifest of its own, so a module's launch shape travels with the module. Every field
/// below is required: a record that lacks one is refused at load instead of launched with a
/// guessed shape. The record's other keys (Triton's compile options, its version, the target) are
/// the compiler's and are not read.
#[derive(Clone, PartialEq, Eq, Debug, serde::Deserialize)]
pub struct CubinMetadata {
    /// The kernel's symbol in the cubin.
    pub name: String,
    /// Warps per thread block; the block is `num_warps × 32` threads.
    pub num_warps: u32,
    /// Dynamic shared memory, in bytes.
    pub shared: u32,
    /// Thread blocks per cluster (Triton's `num_ctas`).
    pub num_ctas: u32,
    /// Launch with programmatic dependent launch.
    pub launch_pdl: bool,
    /// Launch cooperatively.
    pub launch_cooperative_grid: bool,
    /// The parameters the program takes at launch, in order, as the module records them.
    ///
    /// Triton drops constexprs from this list, which is what makes it the authority a launch is
    /// checked against: the operator packs one word per entry here, plus the global scratch when the
    /// module declares one. The record states it as a JSON object whose key order is the
    /// parameters' order.
    #[serde(deserialize_with = "ordered_parameters")]
    pub signature: Vec<Parameter>,
    /// Global scratch bytes the kernel needs, handed to it as one of its last parameters.
    pub global_scratch_size: u64,
    /// The alignment global scratch must have.
    pub global_scratch_align: u64,
    /// Profile scratch bytes the kernel needs, handed to it after the global scratch.
    ///
    /// Triton's launcher passes the address whatever the size; a launch that runs no profiler passes
    /// null and the kernel never reads it.
    pub profile_scratch_size: u64,
    /// The alignment profile scratch must have.
    pub profile_scratch_align: u64,
}

/// One parameter of a kernel's runtime signature.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Parameter {
    /// The parameter's name in the compiled program, for error text.
    pub name: String,
    /// Its type as the module records it: `*bf16`, `*i32`, `i32`, `fp32`, ...
    pub ty: String,
}

impl Parameter {
    /// Whether the driver reads a device address out of this parameter.
    pub fn is_pointer(&self) -> bool {
        self.ty.starts_with('*')
    }
}

/// Reads the record's `signature` object into its parameters in the object's own key order, which
/// is the order the kernel takes them in.
fn ordered_parameters<'de, D>(deserializer: D) -> std::result::Result<Vec<Parameter>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct Ordered;

    impl<'de> serde::de::Visitor<'de> for Ordered {
        type Value = Vec<Parameter>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str("an object of the kernel's parameter names and their types")
        }

        fn visit_map<A>(self, mut map: A) -> std::result::Result<Self::Value, A::Error>
        where
            A: serde::de::MapAccess<'de>,
        {
            let mut parameters = Vec::with_capacity(map.size_hint().unwrap_or(0));
            while let Some((name, ty)) = map.next_entry::<String, String>()? {
                parameters.push(Parameter { name, ty });
            }
            Ok(parameters)
        }
    }

    deserializer.deserialize_map(Ordered)
}

impl CubinMetadata {
    /// Read the metadata record written beside a cubin.
    ///
    /// # Errors
    ///
    /// [`Error::Metadata`] when the record is not JSON, lacks one of the fields a launch reads, or
    /// states one with another type; a warp or cluster count of zero is refused the same way.
    pub fn parse(json: &[u8]) -> Result<Self> {
        let metadata: Self = serde_json::from_slice(json)
            .map_err(|error| Error::Metadata(format!("the module's metadata: {error}")))?;
        if metadata.num_warps == 0 || metadata.num_ctas == 0 {
            return Err(Error::Metadata(format!(
                "`{}` states {} warps per block and {} blocks per cluster: a launch needs at \
                 least one of each",
                metadata.name, metadata.num_warps, metadata.num_ctas
            )));
        }
        Ok(metadata)
    }

    /// The thread block's shape: `num_warps` warps of 32 threads.
    pub const fn threads(&self) -> u32 {
        self.num_warps * 32
    }

    /// Whether the kernel needs global scratch.
    pub const fn needs_scratch(&self) -> bool {
        self.global_scratch_size > 0
    }
}

/// The grid of one launch: how many thread blocks along each axis.
///
/// The thread block comes from the module's metadata, so a grid is only block counts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LaunchGrid {
    pub blocks: [u32; 3],
}

impl LaunchGrid {
    /// The grid of `blocks` thread blocks along each axis.
    pub fn new(blocks: [u64; 3]) -> Result<Self> {
        if blocks.contains(&0) {
            return Err(Error::Grid(format!("grid {blocks:?} has no block")));
        }
        if let Some(&too_many) = blocks.iter().find(|&&blocks| blocks > u64::from(u32::MAX)) {
            return Err(Error::Grid(format!(
                "grid {blocks:?} asks for {too_many} blocks, above the driver's 2^32-1"
            )));
        }
        Ok(Self {
            blocks: [blocks[0] as u32, blocks[1] as u32, blocks[2] as u32],
        })
    }

    /// A grid of `blocks` thread blocks on the first axis.
    pub fn linear(blocks: u64) -> Result<Self> {
        Self::new([blocks, 1, 1])
    }

    /// The grid of `ceiling / per_block` thread blocks on the first axis.
    pub fn covering(extent: u64, per_block: u64) -> Result<Self> {
        if per_block == 0 {
            return Err(Error::Grid("a launch covers no work per block".to_owned()));
        }
        Self::linear(extent.div_ceil(per_block))
    }
}

/// The arguments of one launch, packed into the array a driver launch takes.
///
/// Every parameter is one machine word: a device address for a tensor or a pointer, or a scalar.
/// The array handed to the driver points at those words, so the packing owns them and the operator
/// can build a call in one expression without naming a stack variable per argument.
#[derive(Default)]
pub struct Arguments {
    // One allocation per parameter on purpose: the driver's parameter array points into these
    // words, so their addresses must not change when a later parameter is packed.
    #[expect(
        clippy::vec_box,
        reason = "the driver reads the words through pointers, so each word keeps its address"
    )]
    words: Vec<Box<u64>>,
    /// For each word, whether the driver reads a device address out of it.
    kinds: Vec<Word>,
    /// The global scratch address, for a kernel whose metadata asks for one.
    scratch: Option<DevicePtr>,
    /// The word the scratch address is read from at launch.
    scratch_slot: Box<u64>,
    /// The word the profile scratch address is read from at launch.
    profile_slot: Box<u64>,
    parameters: Vec<*mut c_void>,
}

/// What the driver reads out of one packed parameter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Word {
    /// A device address.
    Pointer,
    /// A scalar value, in the low bytes of the word.
    Scalar,
}

impl Arguments {
    /// No arguments yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// A device address.
    pub fn pointer(&mut self, pointer: DevicePtr) -> &mut Self {
        self.packed(Word::Pointer, pointer.address())
    }

    /// The address a tensor view starts at.
    pub fn view(&mut self, view: &Operand) -> &mut Self {
        self.pointer(view.ptr)
    }

    /// An `u64` scalar.
    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.word(value)
    }

    /// An `i64` scalar.
    pub fn i64(&mut self, value: i64) -> &mut Self {
        self.word(value as u64)
    }

    /// An `u32` scalar.
    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.word(u64::from(value))
    }

    /// An `i32` scalar.
    pub fn i32(&mut self, value: i32) -> &mut Self {
        self.word(u64::from(value as u32))
    }

    /// An `f32` scalar, in the low word of a 64-bit slot as the ABI requires.
    pub fn f32(&mut self, value: f32) -> &mut Self {
        self.word(u64::from(value.to_bits()))
    }

    /// An `f64` scalar.
    pub fn f64(&mut self, value: f64) -> &mut Self {
        self.word(value.to_bits())
    }

    /// A boolean scalar, as the `i32` a C `_Bool` parameter takes.
    pub fn flag(&mut self, value: bool) -> &mut Self {
        self.i32(i32::from(value))
    }

    /// The global scratch the kernel's metadata asks for.
    ///
    /// A kernel whose metadata declares global scratch takes its address as the last parameter; a
    /// caller that omits it, or supplies one to a kernel that declares none, is launching with the
    /// wrong parameter list and fails here rather than in the driver.
    pub fn scratch(&mut self, buffer: Option<DevicePtr>) -> &mut Self {
        self.scratch = buffer;
        self
    }

    /// The parameter array of a launch of the kernel `metadata` describes.
    ///
    /// # Errors
    ///
    /// [`Error::Metadata`] when the packed words disagree with the kernel's signature, or the
    /// global scratch with what the metadata declares.
    pub fn parameters(&mut self, metadata: &CubinMetadata) -> Result<&mut [*mut c_void]> {
        match (metadata.needs_scratch(), self.scratch) {
            (true, None) => {
                return Err(Error::Metadata(
                    "the kernel declares global scratch and the operator passed none".to_owned(),
                ));
            }
            (false, Some(_)) => {
                return Err(Error::Metadata(
                    "the operator passed global scratch to a kernel that declares none".to_owned(),
                ));
            }
            _ => {}
        }

        self.check(metadata)?;
        self.parameters.clear();
        self.parameters.extend(
            self.words
                .iter_mut()
                .map(|word| word.as_mut() as *mut u64 as *mut c_void),
        );
        // A cubin module takes two parameters after its own signature, which the compiler's
        // launcher appends and its PTX declares whatever their size is: the global scratch address
        // and the profile scratch address. They are values the driver reads at launch, exactly like
        // the packed words above, so they are passed through slots of their own that outlive this
        // call.
        *self.scratch_slot = self.scratch.map_or(0, DevicePtr::address);
        *self.profile_slot = 0;
        self.parameters
            .push(self.scratch_slot.as_mut() as *mut u64 as *mut c_void);
        self.parameters
            .push(self.profile_slot.as_mut() as *mut u64 as *mut c_void);
        Ok(&mut self.parameters)
    }

    /// Whether a signature entry is one of the two scratch addresses the compiler appends.
    fn is_scratch(name: &str) -> bool {
        matches!(name, "global_scratch" | "profile_scratch")
    }

    /// Refuse a parameter list that disagrees with the program's own signature.
    ///
    /// The driver reads one value per parameter at the address it is handed, so a list that is short,
    /// long, or has a scalar where the program reads a pointer is not a near miss: the launch dies in
    /// the driver, or reads a device address it was never given, and neither names the operator.
    fn check(&self, metadata: &CubinMetadata) -> Result<()> {
        // `parameters` appends the global and the profile scratch address after the packed words, so
        // the launch passes two more words than the operator packed, whatever the scratch sizes are.
        //
        // The signature is held to the packed words alone, with any entry naming one of those two
        // addresses left out. A module the compiler builds today names neither: `_embedding_kernel`'s
        // signature is its seven parameters and `global_scratch_size`/`profile_scratch_size` are
        // separate fields, so its nine-parameter PTX is those seven plus the two appended here. A
        // module whose metadata does name them states two entries this operator never packs, and
        // filtering keeps that from being counted twice. Either way the count is "what the operator
        // states", which is the only thing it can be held to.
        let declared: Vec<&Parameter> = metadata
            .signature
            .iter()
            .filter(|parameter| !Self::is_scratch(&parameter.name))
            .collect();
        let expected = declared.len();
        let supplied = self.kinds.len();
        if expected != supplied {
            return Err(Error::Metadata(format!(
                "{} declares {expected} parameters ({}) and the operator packed {supplied} ({})",
                metadata.name,
                Self::describe(&metadata.signature),
                Self::describe_packed(&self.kinds),
            )));
        }
        for (position, (parameter, word)) in declared.iter().zip(&self.kinds).enumerate() {
            if parameter.is_pointer() != (*word == Word::Pointer) {
                return Err(Error::Metadata(format!(
                    "{} parameter {position} `{}` is `{}` and the operator packed a {} there",
                    metadata.name,
                    parameter.name,
                    parameter.ty,
                    match word {
                        Word::Pointer => "device address",
                        Word::Scalar => "scalar",
                    }
                )));
            }
        }
        Ok(())
    }

    /// The parameters of a signature, for error text.
    fn describe(signature: &[Parameter]) -> String {
        signature
            .iter()
            .map(|parameter| format!("`{}`: {}", parameter.name, parameter.ty))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The packed words, for error text.
    fn describe_packed(kinds: &[Word]) -> String {
        kinds
            .iter()
            .map(|word| match word {
                Word::Pointer => "address".to_owned(),
                Word::Scalar => "scalar".to_owned(),
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn word(&mut self, value: u64) -> &mut Self {
        self.packed(Word::Scalar, value)
    }

    fn packed(&mut self, kind: Word, value: u64) -> &mut Self {
        self.words.push(Box::new(value));
        self.kinds.push(kind);
        self
    }
}

/// A C entry of a loaded native library, with the name the package declares for it.
pub struct NativeEntry {
    name: String,
    function: Entry,
}

impl NativeEntry {
    /// Resolve `name` in `library` as an [`Entry`].
    ///
    /// The entry's signature is the package's interface with `ops`: `int32_t entry(const void* call,
    /// void* stream)`, the call structure being the operator's own `#[repr(C)]` type.
    pub fn resolve(library: &NativeLibrary, name: &str) -> Result<Self> {
        // SAFETY: the convention every package follows is exactly `Entry`; a package that declares
        // another signature for its entry name breaks this interface, which its header states.
        let function = unsafe { library.entry::<Entry>(name)? };
        Ok(Self {
            name: name.to_owned(),
            function,
        })
    }

    /// The name the package declares.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// A module loaded and ready to launch.
pub enum Kernel {
    /// A cubin loaded into a device's primary context, launched with its own metadata.
    Cubin {
        /// The loaded library; the kernel stays loadable while this is held.
        library: Library,
        /// The kernel inside it.
        function: Function,
        /// Its launch metadata.
        metadata: CubinMetadata,
    },
    /// A native library with the C entry the package declares.
    Native {
        /// The opened library; the kernel stays callable while this is held.
        library: NativeLibrary,
        /// The entry the package declares.
        entry: NativeEntry,
    },
}

impl Kernel {
    /// Load a cubin, fatbin or PTX image on `device` and resolve the kernel its metadata names.
    ///
    /// A module whose metadata declares more dynamic shared memory than a block may use without
    /// opting in (the device's `MaxSharedMemoryPerBlock`) has its function's limit raised here,
    /// once, before any launch: the launch path never sets an attribute, so nothing an operator
    /// does can surprise a capture.
    pub fn load_cubin(device: &Device, image: &[u8], metadata: CubinMetadata) -> Result<Self> {
        let library = Library::load(device, image)?;
        let function = library.kernel(&metadata.name)?;
        let default_limit = device
            .id()
            .attribute(DeviceAttribute::MaxSharedMemoryPerBlock)?;
        if i64::from(metadata.shared) > i64::from(default_limit) {
            function.set_attribute(
                FunctionAttribute::MaxDynamicSharedMemory,
                metadata.shared as i32,
            )?;
        }
        Ok(Self::Cubin {
            library,
            function,
            metadata,
        })
    }

    /// Open nothing and wrap a native library with the entry the package declares.
    ///
    /// The library is opened by whoever resolved the module ([`NativeLibrary::open`]); this
    /// resolves the package's C entry and keeps the two together, so the library stays loaded while
    /// any generation or captured graph can call it.
    pub fn load_native(library: NativeLibrary, entry: &str) -> Result<Self> {
        let entry = NativeEntry::resolve(&library, entry)?;
        Ok(Self::Native { library, entry })
    }

    /// Hand this native kernel's module the directory of its files through its load entry.
    ///
    /// `entry` is the symbol the package declares for [`crate::LOAD_ENTRY`]; the entry records
    /// `directory` and issues no device work, so it is called without a stream.
    ///
    /// # Errors
    ///
    /// [`Error::KernelForm`] for a cubin kernel, [`Error::Platform`] when the library does not
    /// export `entry`, [`Error::Metadata`] for a directory that is not a C string, and
    /// [`Error::Native`] with the entry's message when it refuses the directory.
    pub fn load_directory(&self, entry: &str, directory: &std::path::Path) -> Result<()> {
        use std::os::unix::ffi::OsStrExt as _;

        let path = std::ffi::CString::new(directory.as_os_str().as_bytes())
            .map_err(|_| Error::Metadata(format!("{} is not a C string", directory.display())))?;
        // SAFETY: the package's header declares the load entry with exactly `Entry`'s signature,
        // taking a `LoadCall`.
        let load = unsafe { self.native_symbol::<Entry>(entry)? };
        let mut call = LoadCall {
            directory: path.as_ptr(),
            error: [0; 512],
        };
        // SAFETY: `call` has the layout the header declares, and `path` outlives the call; the
        // entry issues no device work and takes no stream.
        let status = unsafe {
            load(
                &mut call as *mut LoadCall as *const c_void,
                std::ptr::null_mut(),
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(Error::Native {
                entry: entry.to_owned(),
                status,
                message: call.message().map(str::to_owned),
            })
        }
    }

    /// A further C entry of this native kernel's library, by the name its header gives it.
    ///
    /// A package's launch entry is the one [`Self::native`] calls; a lifecycle entry around it is
    /// resolved here, and the library stays loaded while the kernel is held.
    ///
    /// # Safety
    ///
    /// `F` is the signature the package's header declares for `symbol`.
    pub unsafe fn native_symbol<F: Copy>(&self, symbol: &str) -> Result<F> {
        let Self::Native { library, .. } = self else {
            return Err(Error::KernelForm {
                operator: "native_symbol",
                kernel: self.name().to_owned(),
                expected: "native",
                actual: "cubin",
            });
        };
        // SAFETY: forwarded from the caller.
        unsafe { library.entry::<F>(symbol) }.map_err(Error::Platform)
    }

    /// What this kernel is called, for diagnostics and for a capture's node table.
    pub fn name(&self) -> &str {
        match self {
            Self::Cubin { metadata, .. } => &metadata.name,
            Self::Native { entry, .. } => entry.name(),
        }
    }

    /// The launch metadata of a cubin kernel; a native kernel has none.
    pub fn metadata(&self) -> Option<&CubinMetadata> {
        match self {
            Self::Cubin { metadata, .. } => Some(metadata),
            Self::Native { .. } => None,
        }
    }

    /// Whether this kernel is a cubin.
    pub const fn is_cubin(&self) -> bool {
        matches!(self, Self::Cubin { .. })
    }

    /// Launch this cubin kernel with `grid` and `arguments`.
    ///
    /// A launch on a capturing stream is captured; the operator is written so that it is, and the
    /// nodes it adds are what a rebind later rewrites.
    pub fn launch(
        &self,
        grid: LaunchGrid,
        arguments: &mut Arguments,
        stream: &Stream,
    ) -> Result<()> {
        let Self::Cubin {
            function, metadata, ..
        } = self
        else {
            return Err(Error::KernelForm {
                operator: "launch",
                kernel: self.name().to_owned(),
                expected: "cubin",
                actual: "native",
            });
        };
        let parameters = arguments.parameters(metadata)?;

        // The grid is the operator's; the block, the dynamic shared memory, the cluster,
        // programmatic dependent launch and cooperative launch are the module's own metadata.
        let config = LaunchConfig {
            grid: grid.blocks,
            block: [metadata.threads(), 1, 1],
            shared_memory: metadata.shared,
            // A cluster of one block is no cluster: the attribute is only set when the module
            // asks for a real one.
            cluster: (metadata.num_ctas > 1).then(|| Cluster::new([metadata.num_ctas, 1, 1])),
            pdl: metadata.launch_pdl,
            cooperative: metadata.launch_cooperative_grid,
        };
        function
            .launch_ex(stream, &config, parameters)
            .map_err(Error::Platform)
    }

    /// Call this native kernel's entry with `call` on `stream`.
    ///
    /// # Safety
    ///
    /// `call` is the structure the package's entry declares, with the layout the package's header
    /// gives it; the entry's signature is [`Entry`].
    pub unsafe fn native<C: NativeCall>(&self, call: &mut C, stream: &Stream) -> Result<()> {
        let Self::Native { entry, .. } = self else {
            return Err(Error::KernelForm {
                operator: "native",
                kernel: self.name().to_owned(),
                expected: "native",
                actual: "cubin",
            });
        };
        // An entry computes on the device the calling thread holds current, as the libraries it
        // calls read it (`cudaGetDevice`), so the stream's context is current for the call as it
        // is for a cubin's launch: a thread that issues the work of several devices would
        // otherwise run the entry for another device than its stream's.
        let status = stream
            .with_context(|| {
                // SAFETY: forwarded from the caller: the entry is this package's `Entry`, and the
                // call structure has the layout the package's header declares for it.
                unsafe { (entry.function)(call as *mut C as *const c_void, stream_handle(stream)) }
            })
            .map_err(Error::Platform)?;
        if status == 0 {
            Ok(())
        } else {
            Err(Error::Native {
                entry: entry.name().to_owned(),
                status,
                message: call.message().map(str::to_owned),
            })
        }
    }
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cubin { metadata, .. } => formatter
                .debug_struct("Kernel::Cubin")
                .field("metadata", metadata)
                .finish_non_exhaustive(),
            Self::Native { entry, .. } => formatter
                .debug_struct("Kernel::Native")
                .field("entry", &entry.name())
                .finish_non_exhaustive(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Arguments, CubinMetadata, DevicePtr, Kernel, NativeLibrary};
    use crate::Error;

    /// A metadata record of a kernel taking `signature` (a JSON object, in parameter order) and
    /// `global_scratch_size` bytes of global scratch, the other fields as Triton writes them for a
    /// plain kernel.
    fn metadata(signature: &str, global_scratch_size: u64) -> CubinMetadata {
        let signature: serde_json::Value =
            serde_json::from_str(signature).expect("the signature is a JSON object");
        let record = serde_json::json!({
            "name": "_kernel",
            "num_warps": 4,
            "shared": 0,
            "num_ctas": 1,
            "launch_pdl": false,
            "launch_cooperative_grid": false,
            "signature": signature,
            "global_scratch_size": global_scratch_size,
            "global_scratch_align": 1,
            "profile_scratch_size": 0,
            "profile_scratch_align": 1,
        });
        CubinMetadata::parse(record.to_string().as_bytes()).expect("the metadata parses")
    }

    #[test]
    fn a_module_takes_its_own_parameters_and_then_the_two_scratch_addresses() {
        let module = metadata(r#"{"x":"*bf16","rows":"i32"}"#, 0);
        let mut arguments = Arguments::new();
        arguments.pointer(DevicePtr::from_address(0x1000)).u32(7);
        // The program's two parameters, then the global and the profile scratch address Triton's
        // launcher appends and its PTX declares whatever their size is.
        assert_eq!(arguments.parameters(&module).unwrap().len(), 4);
    }

    #[test]
    fn a_signature_that_names_its_scratch_is_not_counted_twice() {
        // Triton 3.7's metadata records the two scratch addresses inside the signature, and the launch
        // appends those two slots itself. A module of two parameters therefore takes two packed
        // words, not four: counting the named entries again refused every module built that way.
        let module = metadata(
            r#"{"x":"*bf16","rows":"i32","global_scratch":"*i8","profile_scratch":"*i8"}"#,
            0,
        );
        let mut arguments = Arguments::new();
        arguments.pointer(DevicePtr::from_address(0x1000)).u32(7);
        assert_eq!(arguments.parameters(&module).unwrap().len(), 4);
        let mut long = Arguments::new();
        long.pointer(DevicePtr::from_address(0x1000)).u32(7).u32(1);
        let error = long.parameters(&module).unwrap_err().to_string();
        assert!(error.contains("declares 2 parameters"), "{error}");
        assert!(error.contains("packed 3"), "{error}");
    }

    #[test]
    fn the_parameter_list_must_hold_one_word_per_parameter() {
        let module = metadata(r#"{"x":"*bf16","rows":"i32","eps":"fp32"}"#, 0);
        let mut short = Arguments::new();
        short.pointer(DevicePtr::from_address(0x1000)).u32(7);
        let error = short.parameters(&module).unwrap_err().to_string();
        assert!(error.contains("_kernel"), "{error}");
        assert!(error.contains("declares 3 parameters"), "{error}");
        assert!(error.contains("packed 2"), "{error}");
        assert!(error.contains("`eps`: fp32"), "{error}");
    }

    #[test]
    fn a_scalar_where_the_program_reads_an_address_is_refused_by_position() {
        let module = metadata(r#"{"x":"*bf16","rows":"i32"}"#, 0);
        let mut arguments = Arguments::new();
        arguments.u64(0).u32(7);
        let error = arguments.parameters(&module).unwrap_err().to_string();
        assert!(error.contains("parameter 0 `x` is `*bf16`"), "{error}");
        assert!(error.contains("scalar"), "{error}");
    }

    #[test]
    fn global_scratch_is_a_value_the_caller_supplies_and_its_absence_is_refused() {
        let module = metadata(r#"{"x":"*bf16"}"#, 4096);
        let mut arguments = Arguments::new();
        arguments.pointer(DevicePtr::from_address(0x1000));
        arguments.scratch(Some(DevicePtr::from_address(0x2000)));
        // One declared parameter and the two scratch addresses.
        assert_eq!(arguments.parameters(&module).unwrap().len(), 3);

        let mut without = Arguments::new();
        without.pointer(DevicePtr::from_address(0x1000));
        let error = without.parameters(&module).unwrap_err().to_string();
        assert!(error.contains("global scratch"), "{error}");
    }

    #[test]
    fn a_matching_list_packs_one_address_per_word() {
        let module = metadata(r#"{"x":"*fp32","out":"*fp32","rows":"i64"}"#, 0);
        let mut arguments = Arguments::new();
        arguments
            .pointer(DevicePtr::from_address(0x1000))
            .pointer(DevicePtr::from_address(0x2000))
            .u64(4);
        assert_eq!(arguments.parameters(&module).unwrap().len(), 5);
    }

    /// A load entry that refuses the module's directory refuses the module, with the entry's
    /// status: a module installed without its directory would fail at its first launch instead.
    /// `getpid` stands in for an entry that answers a nonzero status.
    #[test]
    fn a_load_entry_s_refusal_refuses_the_module() {
        let library = NativeLibrary::open("libc.so.6").expect("the C library loads");
        let kernel = Kernel::load_native(library, "getpid").expect("the entry resolves");
        let refused = kernel
            .load_directory("getpid", std::path::Path::new("/module"))
            .unwrap_err();
        assert!(
            matches!(&refused, Error::Native { entry, status, .. } if entry == "getpid" && *status != 0),
            "{refused}"
        );
        let missing = kernel
            .load_directory(
                "uniserve_no_such_load_entry",
                std::path::Path::new("/module"),
            )
            .unwrap_err();
        assert!(
            missing.to_string().contains("uniserve_no_such_load_entry"),
            "{missing}"
        );
    }
}
