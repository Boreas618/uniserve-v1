//! Loaded kernels, their metadata, and the one place a kernel is launched or a native entry called.
//!
//! A [`Kernel`] is a module that has been loaded: a cubin loaded into a device's context, or a
//! native library opened with the C entry it declares (specs/worker.md 4.2, 5.3). A
//! [`Generation`](crate::Generation) holds one per (operator name, descriptor key) and is immutable
//! once built, so a kernel stays loaded exactly as long as a generation or a captured graph holds
//! it (specs/worker.md 7.3, purge).
//!
//! ## What a cubin launch takes from here and what from the package
//!
//! The grid is the operator's: it follows from the arguments. The thread-block shape, dynamic shared
//! memory, cluster size, programmatic dependent launch, cooperative launch and global scratch come
//! from the module's own metadata, which is Triton's JSON field for field (specs/worker.md 5.3), so
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

use crate::platform::{self, Function, Library, NativeLibrary, Stream};
use crate::tensor::DevicePtr;
use crate::{Error, Result};

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

/// A module's launch metadata, in Triton's field names.
///
/// Triton writes this JSON beside every cubin it compiles; `ops` reads those fields directly rather
/// than defining a manifest of its own, so a module's launch shape travels with the module
/// (specs/worker.md 5.3).
#[derive(Clone, PartialEq, Eq, Debug)]
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
    /// Global scratch bytes the kernel needs, handed to it as its last parameter.
    pub global_scratch_size: u64,
    /// The alignment global scratch must have.
    pub global_scratch_align: u64,
}

impl CubinMetadata {
    /// Read the metadata Triton wrote beside a cubin.
    ///
    /// Only the fields above are read; other keys are skipped, including the arrays and objects a
    /// newer Triton may add, so a module from a newer compiler still loads when these fields mean
    /// what they meant.
    pub fn parse(json: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(json)
            .map_err(|error| Error::Metadata(format!("metadata is not UTF-8: {error}")))?;
        let mut reader = JsonReader::new(text);
        let fields = reader.object().map_err(Error::Metadata)?;
        let name = fields
            .string("name")
            .ok_or_else(|| Error::Metadata("metadata has no `name`".to_owned()))?;
        Ok(Self {
            name: name.to_owned(),
            num_warps: fields.integer("num_warps").unwrap_or(1).max(1) as u32,
            shared: fields.integer("shared").unwrap_or(0) as u32,
            num_ctas: fields.integer("num_ctas").unwrap_or(1).max(1) as u32,
            launch_pdl: fields.boolean("launch_pdl").unwrap_or(false),
            launch_cooperative_grid: fields.boolean("launch_cooperative_grid").unwrap_or(false),
            global_scratch_size: fields.integer("global_scratch_size").unwrap_or(0),
            global_scratch_align: fields.integer("global_scratch_align").unwrap_or(0),
        })
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
    /// The global scratch address, for a kernel whose metadata asks for one.
    scratch: Option<DevicePtr>,
    /// The word the scratch address is read from at launch.
    scratch_slot: Box<u64>,
    parameters: Vec<*mut c_void>,
}

impl Arguments {
    /// No arguments yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// A device address.
    pub fn pointer(&mut self, pointer: DevicePtr) -> &mut Self {
        self.word(platform::address(pointer))
    }

    /// The address a tensor view starts at.
    pub fn view(&mut self, view: crate::tensor::TensorView) -> &mut Self {
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

    /// The parameter array, for a launch.
    pub fn parameters(&mut self, metadata: Option<&CubinMetadata>) -> Result<&mut [*mut c_void]> {
        let needs_scratch = metadata.is_some_and(CubinMetadata::needs_scratch);
        match (needs_scratch, self.scratch) {
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

        self.parameters.clear();
        self.parameters.extend(
            self.words
                .iter_mut()
                .map(|word| word.as_mut() as *mut u64 as *mut c_void),
        );
        if let Some(scratch) = self.scratch {
            // The scratch address is a value the driver reads at launch, exactly like the packed
            // words above; keeping it in the same `words` vector would misplace it among the
            // kernel's own parameters, so it is passed through a slot of its own.
            *self.scratch_slot = platform::address(scratch);
            self.parameters
                .push(self.scratch_slot.as_mut() as *mut u64 as *mut c_void);
        }
        Ok(&mut self.parameters)
    }

    fn word(&mut self, value: u64) -> &mut Self {
        self.words.push(Box::new(value));
        self
    }
}

/// A C entry of a loaded native library, with the name the package declares for it.
pub struct NativeEntry {
    name: &'static str,
    function: Entry,
}

impl NativeEntry {
    /// Resolve `name` in `library` as an [`Entry`].
    ///
    /// The entry's signature is the package's contract with `ops`: `int32_t entry(const void* call,
    /// void* stream)`, the call structure being the operator's own `#[repr(C)]` type.
    pub fn resolve(library: &NativeLibrary, name: &'static str) -> Result<Self> {
        // SAFETY: the convention every package follows is exactly `Entry`; a package that declares
        // another signature for its entry name breaks this contract, which its header states.
        let function = unsafe { platform::native_entry::<Entry>(library, name)? };
        Ok(Self { name, function })
    }

    /// The name the package declares.
    pub const fn name(&self) -> &'static str {
        self.name
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
    /// A module whose metadata declares a dynamic shared-memory size beyond the driver's default
    /// has its function's limit raised here, once, before any launch: the launch path never sets an
    /// attribute, so nothing an operator does can surprise a capture.
    pub fn load_cubin(
        device: &platform::Device,
        image: &[u8],
        metadata: CubinMetadata,
    ) -> Result<Self> {
        let library = platform::load_library(device, image)?;
        let function = platform::library_kernel(&library, &metadata.name)?;
        if metadata.shared > platform::DEFAULT_SHARED_LIMIT {
            platform::raise_shared_limit(&function, metadata.shared)?;
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
    pub fn load_native(library: NativeLibrary, entry: &'static str) -> Result<Self> {
        let entry = NativeEntry::resolve(&library, entry)?;
        Ok(Self::Native { library, entry })
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
    /// nodes it adds are what a rebind later rewrites (specs/worker.md 4.5).
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
        let parameters = arguments.parameters(Some(metadata))?;
        platform::launch(
            function,
            stream,
            platform::Launch {
                blocks: grid.blocks,
                threads: metadata.threads(),
                shared: metadata.shared,
                cluster: metadata.num_ctas,
                pdl: metadata.launch_pdl,
                cooperative: metadata.launch_cooperative_grid,
            },
            parameters,
        )
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
        // SAFETY: the caller states that this symbol is the package's entry with the `Entry`
        // signature.
        // SAFETY: forwarded from the caller: the entry is this package's `Entry`, and the call
        // structure has the layout the package's header declares for it.
        let status = unsafe {
            (entry.function)(
                call as *mut C as *const c_void,
                platform::stream_handle(stream),
            )
        };
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

/// The scalar fields of Triton's metadata JSON, as read by [`CubinMetadata::parse`].
struct MetadataFields<'a> {
    fields: Vec<(&'a str, Scalar<'a>)>,
}

/// One scalar value of the metadata JSON.
enum Scalar<'a> {
    String(&'a str),
    Number(f64),
    Bool(bool),
    Other,
}

impl<'a> MetadataFields<'a> {
    fn field(&self, name: &str) -> Option<&Scalar<'a>> {
        self.fields
            .iter()
            .find_map(|(key, value)| (*key == name).then_some(value))
    }

    fn string(&self, name: &str) -> Option<&'a str> {
        match self.field(name) {
            Some(Scalar::String(value)) => Some(value),
            _ => None,
        }
    }

    fn integer(&self, name: &str) -> Option<u64> {
        match self.field(name) {
            Some(Scalar::Number(value)) if *value >= 0.0 => Some(*value as u64),
            _ => None,
        }
    }

    fn boolean(&self, name: &str) -> Option<bool> {
        match self.field(name) {
            Some(Scalar::Bool(value)) => Some(*value),
            _ => None,
        }
    }
}

/// A reader for the flat object Triton's metadata JSON is.
///
/// It reads one object of scalar values and skips any array or object it meets, which is all `ops`
/// needs and less than a general JSON parser would cost in a crate that depends on `platform` only.
struct JsonReader<'a> {
    text: &'a str,
    bytes: usize,
}

impl<'a> JsonReader<'a> {
    fn new(text: &'a str) -> Self {
        Self { text, bytes: 0 }
    }

    fn object(&mut self) -> std::result::Result<MetadataFields<'a>, String> {
        self.skip_whitespace();
        self.expect(b'{')?;
        let mut fields = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek() == Some(b'}') {
                self.bytes += 1;
                return Ok(MetadataFields { fields });
            }
            let key = self.string()?;
            self.skip_whitespace();
            self.expect(b':')?;
            self.skip_whitespace();
            let value = self.value()?;
            fields.push((key, value));
            self.skip_whitespace();
            match self.peek() {
                Some(b',') => self.bytes += 1,
                Some(b'}') => continue,
                _ => return Err("expected `,` or `}` in the metadata object".to_owned()),
            }
        }
    }

    fn value(&mut self) -> std::result::Result<Scalar<'a>, String> {
        match self.peek() {
            Some(b'"') => Ok(Scalar::String(self.string()?)),
            Some(b't') => {
                self.literal("true")?;
                Ok(Scalar::Bool(true))
            }
            Some(b'f') => {
                self.literal("false")?;
                Ok(Scalar::Bool(false))
            }
            Some(b'n') => {
                self.literal("null")?;
                Ok(Scalar::Other)
            }
            Some(b'[') => {
                self.skip_nested(b'[', b']')?;
                Ok(Scalar::Other)
            }
            Some(b'{') => {
                self.skip_nested(b'{', b'}')?;
                Ok(Scalar::Other)
            }
            Some(byte) if byte == b'-' || byte.is_ascii_digit() => {
                let start = self.bytes;
                while let Some(byte) = self.peek() {
                    if byte.is_ascii_digit() || matches!(byte, b'-' | b'+' | b'.' | b'e' | b'E') {
                        self.bytes += 1;
                    } else {
                        break;
                    }
                }
                let token = &self.text[start..self.bytes];
                token
                    .parse::<f64>()
                    .map(Scalar::Number)
                    .map_err(|error| format!("`{token}` is not a number in the metadata: {error}"))
            }
            _ => Err("unexpected value in the metadata object".to_owned()),
        }
    }

    fn string(&mut self) -> std::result::Result<&'a str, String> {
        self.expect(b'"')?;
        let start = self.bytes;
        while let Some(byte) = self.peek() {
            match byte {
                b'"' => {
                    let value = &self.text[start..self.bytes];
                    self.bytes += 1;
                    return Ok(value);
                }
                b'\\' => {
                    // Triton's metadata has no escaped key or value `ops` reads; a name with an
                    // escape is refused rather than unescaped wrongly.
                    return Err("escaped strings are not supported in kernel metadata".to_owned());
                }
                _ => self.bytes += 1,
            }
        }
        Err("unterminated string in the metadata".to_owned())
    }

    fn skip_nested(&mut self, open: u8, close: u8) -> std::result::Result<(), String> {
        let mut depth = 0_u32;
        while let Some(byte) = self.peek() {
            self.bytes += 1;
            if byte == open {
                depth += 1;
            } else if byte == close {
                depth -= 1;
                if depth == 0 {
                    return Ok(());
                }
            } else if byte == b'"' {
                // Step over a string, so a brace inside it does not count.
                self.bytes -= 1;
                self.string()?;
            }
        }
        Err("unterminated array or object in the metadata".to_owned())
    }

    fn literal(&mut self, text: &str) -> std::result::Result<(), String> {
        if self.text[self.bytes..].starts_with(text) {
            self.bytes += text.len();
            Ok(())
        } else {
            Err(format!("expected `{text}` in the metadata"))
        }
    }

    fn expect(&mut self, byte: u8) -> std::result::Result<(), String> {
        if self.peek() == Some(byte) {
            self.bytes += 1;
            Ok(())
        } else {
            Err(format!(
                "expected `{}` at byte {} of the metadata",
                char::from(byte),
                self.bytes
            ))
        }
    }

    fn peek(&self) -> Option<u8> {
        self.text.as_bytes().get(self.bytes).copied()
    }

    fn skip_whitespace(&mut self) {
        while self.peek().is_some_and(|byte| byte.is_ascii_whitespace()) {
            self.bytes += 1;
        }
    }
}
