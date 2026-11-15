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
//! `#[repr(C)]` in the operator's module, and the package's C declaration of it is generated from
//! that definition with its layout asserted ([`crate::abi`]). An entry uses only the stream it is
//! given, never allocates, synchronizes or reads device memory on the host, and is safe to call
//! while the stream is capturing.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_void};

use platform::{
    Cluster, Device, DeviceAttribute, DevicePtr, Function, FunctionAttribute, LaunchConfig,
    Library, NativeLibrary, Stream,
};

use crate::abi::MESSAGE;
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
/// a structure with an `error` field ([`crate::abi::MESSAGE`] bytes) returns the message it finds
/// there ([`crate::abi::message`]).
pub trait NativeCall {
    /// The NUL-terminated message the entry wrote, or `None` when it wrote none.
    fn message(&self) -> Option<&str> {
        None
    }
}

/// The call of a native module's load entry ([`crate::LOAD_ENTRY`]): the directory of the module's
/// files, for a module that reads files published beside its library (the cubins a trtllm-gen
/// module launches). Declared to the packages as `uniserve_load_call` (`uniserve_load.h`,
/// generated with its layout asserted: [`crate::abi`]).
#[repr(C)]
#[derive(crate::abi::Layout)]
pub struct LoadCall {
    /// The NUL-terminated path of the directory the module's files were published in.
    pub directory: *const c_char,
    /// The NUL-terminated message the entry writes before it returns a nonzero status.
    pub error: [c_char; MESSAGE],
}

impl NativeCall for LoadCall {
    fn message(&self) -> Option<&str> {
        crate::abi::message(&self.error)
    }
}

/// A module's launch metadata, in Triton's field names.
///
/// Triton writes this JSON beside every cubin it compiles, the Build Server's Triton driver adds
/// the launch signature and the compile-time definitions, and its CUDA driver writes the same
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
    /// The values the module was compiled with, by name: a Triton kernel's constexprs as its
    /// launch table bound them, a CUDA unit's preprocessor definitions.
    ///
    /// A launch that depends on one of them (the tile a grid is counted in) reads it here, from
    /// the module it launches, instead of restating it beside the package. A module compiled with
    /// none states an empty object.
    pub definitions: BTreeMap<String, serde_json::Value>,
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

    /// The compile-time definition `name`, which the module states as a non-negative integer.
    ///
    /// # Errors
    ///
    /// [`Error::Metadata`] naming the kernel and the definition when the module states none of
    /// that name, or states it as another value than a non-negative integer.
    pub fn integer(&self, name: &str) -> Result<u64> {
        let value = self.definitions.get(name).ok_or_else(|| {
            Error::Metadata(format!(
                "`{}` was compiled with no definition `{name}`",
                self.name
            ))
        })?;
        value.as_u64().ok_or_else(|| {
            Error::Metadata(format!(
                "`{}` defines `{name}` as {value}, which is no non-negative integer",
                self.name
            ))
        })
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

/// The arguments of one launch: each word named for the kernel parameter the operator means, and
/// packed into the array a driver launch takes once the kernel's own signature has accepted them.
///
/// The operator states every word with the name of the parameter it fills and its value in the
/// operator's own type. [`Arguments::parameters`] holds that list to the module's signature (the
/// same names in the same order, each value one its parameter's type holds) and only then writes
/// each word as the parameter's type lays it out, so the order and the names a launch packs are
/// checked by the module itself, and a disagreement is refused naming the operator, the position
/// and both names instead of reaching the driver as a misread word.
pub struct Arguments {
    /// The operator whose launch this is ([`crate::Op::NAME`]), for the refusal.
    operator: &'static str,
    /// The words in the order the operator packed them.
    stated: Vec<Word>,
    // One allocation per parameter on purpose: the driver's parameter array points into these
    // words, so their addresses must not change when a later parameter is written.
    #[expect(
        clippy::vec_box,
        reason = "the driver reads the words through pointers, so each word keeps its address"
    )]
    words: Vec<Box<u64>>,
    /// The global scratch address, for a kernel whose metadata asks for one.
    scratch: Option<DevicePtr>,
    /// The word the scratch address is read from at launch.
    scratch_slot: Box<u64>,
    /// The word the profile scratch address is read from at launch.
    profile_slot: Box<u64>,
    parameters: Vec<*mut c_void>,
}

/// One word an operator packs: the parameter it means and the value it gives it.
struct Word {
    /// The kernel parameter's name, as the kernel's `def` spells it.
    name: Cow<'static, str>,
    value: Value,
}

/// A packed value, in the type the operator holds it in.
///
/// It is laid out in the parameter's own type only once the signature has accepted it, so an
/// integer packed from a `u32` fills an `i32` parameter exactly when the `i32` holds it.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Value {
    /// A device address.
    Address(u64),
    /// An integer: every `u64` and every `i64` is one.
    Integer(i128),
    /// A single-precision float.
    F32(f32),
    /// A double-precision float.
    F64(f64),
}

impl Value {
    /// The value, for error text.
    fn describe(self) -> String {
        match self {
            Self::Address(address) => format!("the device address {address:#x}"),
            Self::Integer(value) => format!("the integer {value}"),
            Self::F32(value) => format!("the f32 {value}"),
            Self::F64(value) => format!("the f64 {value}"),
        }
    }
}

/// What a parameter of a module's signature reads, by its type's spelling.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reads {
    /// A device address: `*bf16`, `*i32`, ...
    Address,
    /// An integer of the range a type of that many bits and signedness holds.
    Integer { min: i128, max: i128 },
    /// `fp32`.
    F32,
    /// `fp64`.
    F64,
}

impl Reads {
    /// What a parameter of type `ty` reads, or `None` for a type no launch word fills.
    fn of(ty: &str) -> Option<Self> {
        if ty.starts_with('*') {
            return Some(Self::Address);
        }
        let integer = |bits: u32, signed: bool| {
            let (min, max) = if signed {
                (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1)
            } else {
                (0, (1_i128 << bits) - 1)
            };
            Some(Self::Integer { min, max })
        };
        match ty {
            "i8" => integer(8, true),
            "i16" => integer(16, true),
            "i32" => integer(32, true),
            "i64" => integer(64, true),
            "u8" => integer(8, false),
            "u16" => integer(16, false),
            "u32" => integer(32, false),
            "u64" => integer(64, false),
            "fp32" => Some(Self::F32),
            "fp64" => Some(Self::F64),
            _ => None,
        }
    }

    /// The word a parameter reading this holds for `value`, laid out as the driver reads it (the
    /// parameter's bytes are the word's low bytes), or `None` when the parameter cannot hold it.
    fn encode(self, value: Value) -> Option<u64> {
        match (self, value) {
            (Self::Address, Value::Address(address)) => Some(address),
            // Two's complement truncated to the word: the low bytes of a value the range holds are
            // the parameter's bytes, whatever its width.
            (Self::Integer { min, max }, Value::Integer(value)) => {
                (min..=max).contains(&value).then_some(value as u64)
            }
            (Self::F32, Value::F32(value)) => Some(u64::from(value.to_bits())),
            (Self::F64, Value::F64(value)) => Some(value.to_bits()),
            _ => None,
        }
    }
}

impl Arguments {
    /// No arguments yet, of a launch of `operator`.
    pub fn new(operator: &'static str) -> Self {
        Self {
            operator,
            stated: Vec::new(),
            words: Vec::new(),
            scratch: None,
            scratch_slot: Box::new(0),
            profile_slot: Box::new(0),
            parameters: Vec::new(),
        }
    }

    /// The device address of the parameter `name`.
    pub fn pointer(&mut self, name: impl Into<Cow<'static, str>>, pointer: DevicePtr) -> &mut Self {
        self.packed(name, Value::Address(pointer.address()))
    }

    /// The address a tensor view starts at, for the parameter `name`.
    pub fn view(&mut self, name: impl Into<Cow<'static, str>>, view: &Operand) -> &mut Self {
        self.pointer(name, view.ptr)
    }

    /// An `u64` value of the parameter `name`.
    pub fn u64(&mut self, name: impl Into<Cow<'static, str>>, value: u64) -> &mut Self {
        self.packed(name, Value::Integer(i128::from(value)))
    }

    /// An `i64` value of the parameter `name`.
    pub fn i64(&mut self, name: impl Into<Cow<'static, str>>, value: i64) -> &mut Self {
        self.packed(name, Value::Integer(i128::from(value)))
    }

    /// An `u32` value of the parameter `name`.
    pub fn u32(&mut self, name: impl Into<Cow<'static, str>>, value: u32) -> &mut Self {
        self.packed(name, Value::Integer(i128::from(value)))
    }

    /// An `i32` value of the parameter `name`.
    pub fn i32(&mut self, name: impl Into<Cow<'static, str>>, value: i32) -> &mut Self {
        self.packed(name, Value::Integer(i128::from(value)))
    }

    /// An `f32` value of the parameter `name`.
    pub fn f32(&mut self, name: impl Into<Cow<'static, str>>, value: f32) -> &mut Self {
        self.packed(name, Value::F32(value))
    }

    /// An `f64` value of the parameter `name`.
    pub fn f64(&mut self, name: impl Into<Cow<'static, str>>, value: f64) -> &mut Self {
        self.packed(name, Value::F64(value))
    }

    /// A boolean of the parameter `name`, as the integer 0 or 1 a C `_Bool` parameter takes.
    pub fn flag(&mut self, name: impl Into<Cow<'static, str>>, value: bool) -> &mut Self {
        self.packed(name, Value::Integer(i128::from(value)))
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
    /// [`Error::Signature`] when the packed words disagree with the kernel's signature (their
    /// number, a name at a position, or a value its parameter's type does not hold), or the
    /// global scratch with what the metadata declares.
    pub fn parameters(&mut self, metadata: &CubinMetadata) -> Result<&mut [*mut c_void]> {
        match (metadata.needs_scratch(), self.scratch) {
            (true, None) => {
                return Err(self.refused(
                    metadata,
                    "the kernel declares global scratch and the operator passed none".to_owned(),
                ));
            }
            (false, Some(_)) => {
                return Err(self.refused(
                    metadata,
                    "the operator passed global scratch to a kernel that declares none".to_owned(),
                ));
            }
            _ => {}
        }

        let words = self.check(metadata)?;
        self.words.clear();
        self.words.extend(words.into_iter().map(Box::new));
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

    /// The words of a launch of the kernel `metadata` describes, each laid out in its parameter's
    /// type, or the refusal of a list that disagrees with the program's own signature.
    ///
    /// The driver reads one value per parameter at the address it is handed, so a list that is
    /// short, long, names another parameter at a position, or holds a value its parameter's type
    /// does not (a scalar where the program reads an address, a float where it reads an integer,
    /// an integer beyond the parameter's range) is not a near miss: the launch dies in the driver,
    /// or the program reads another value than the operator meant, and neither names the operator.
    fn check(&self, metadata: &CubinMetadata) -> Result<Vec<u64>> {
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
        if declared.len() != self.stated.len() {
            return Err(self.refused(
                metadata,
                format!(
                    "the operator packs {} words ({}) and the kernel takes {} parameters ({})",
                    self.stated.len(),
                    self.describe_packed(),
                    declared.len(),
                    Self::describe(&declared),
                ),
            ));
        }

        let mut words = Vec::with_capacity(declared.len());
        for (position, (parameter, word)) in declared.iter().zip(&self.stated).enumerate() {
            if parameter.name != word.name {
                return Err(self.refused(
                    metadata,
                    format!(
                        "the operator packs `{}` at position {position}, where the kernel takes \
                         `{}`",
                        word.name, parameter.name
                    ),
                ));
            }
            let Some(reads) = Reads::of(&parameter.ty) else {
                return Err(self.refused(
                    metadata,
                    format!(
                        "parameter {position} `{}` is `{}`, which no launch word fills",
                        parameter.name, parameter.ty
                    ),
                ));
            };
            let Some(encoded) = reads.encode(word.value) else {
                return Err(self.refused(
                    metadata,
                    format!(
                        "the operator packs {} for `{}` at position {position}, which the \
                         kernel's `{}` parameter does not hold",
                        word.value.describe(),
                        word.name,
                        parameter.ty
                    ),
                ));
            };
            words.push(encoded);
        }
        Ok(words)
    }

    /// The refusal of this launch of the kernel `metadata` describes.
    fn refused(&self, metadata: &CubinMetadata, problem: String) -> Error {
        Error::Signature {
            operator: self.operator,
            kernel: metadata.name.clone(),
            problem,
        }
    }

    /// The parameters of a signature, for error text.
    fn describe(signature: &[&Parameter]) -> String {
        signature
            .iter()
            .map(|parameter| format!("`{}`: {}", parameter.name, parameter.ty))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// The packed words, for error text.
    fn describe_packed(&self) -> String {
        self.stated
            .iter()
            .map(|word| format!("`{}`", word.name))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn packed(&mut self, name: impl Into<Cow<'static, str>>, value: Value) -> &mut Self {
        self.stated.push(Word {
            name: name.into(),
            value,
        });
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
        // SAFETY: the package's load entry has `Entry`'s signature and takes a `LoadCall`, which
        // its generated header declares.
        let load = unsafe { self.native_symbol::<Entry>(entry)? };
        let mut call = LoadCall {
            directory: path.as_ptr(),
            error: [0; MESSAGE],
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

    /// The launch metadata of this kernel, which `operator` launches as a cubin.
    ///
    /// # Errors
    ///
    /// [`Error::KernelForm`] when the kernel is native.
    pub fn cubin(&self, operator: &'static str) -> Result<&CubinMetadata> {
        self.metadata().ok_or_else(|| Error::KernelForm {
            operator,
            kernel: self.name().to_owned(),
            expected: "cubin",
            actual: "native",
        })
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

    /// A metadata record of `_kernel` taking `signature` (a JSON object, in parameter order) and
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
            "definitions": {"BLOCK": 1024, "PDL": false},
        });
        CubinMetadata::parse(record.to_string().as_bytes()).expect("the metadata parses")
    }

    /// The words the driver reads at each of `parameters`' addresses.
    fn read(parameters: &[*mut std::ffi::c_void]) -> Vec<u64> {
        parameters
            .iter()
            // SAFETY: every entry points at a word the arguments own and keep while borrowed.
            .map(|&parameter| unsafe { *(parameter as *const u64) })
            .collect()
    }

    /// The refusal of `arguments` launching `module`: the operator, the kernel and the problem.
    fn refusal(
        arguments: &mut Arguments,
        module: &CubinMetadata,
    ) -> (&'static str, String, String) {
        match arguments.parameters(module) {
            Err(Error::Signature {
                operator,
                kernel,
                problem,
            }) => (operator, kernel, problem),
            Err(other) => panic!("a disagreement is a signature refusal, not {other}"),
            Ok(_) => panic!("the words disagree with the signature and were accepted"),
        }
    }

    #[test]
    fn a_launch_passes_each_named_word_in_its_parameters_type_then_the_two_scratch_addresses() {
        let module = metadata(
            r#"{"x":"*bf16","rows":"i32","offset":"i64","eps":"fp32","wide":"fp64","bits":"u8"}"#,
            0,
        );
        let mut arguments = Arguments::new("operator");
        arguments
            .pointer("x", DevicePtr::from_address(0x1000))
            .u32("rows", 7)
            .i32("offset", -3)
            .f32("eps", 0.5)
            .f64("wide", 0.25)
            .flag("bits", true);

        // The program's six parameters, then the global and the profile scratch address Triton's
        // launcher appends and its PTX declares whatever their size is. The driver reads each
        // parameter's bytes from the low bytes of its word: `-3` is the `i64` two's complement.
        let words = read(arguments.parameters(&module).unwrap());
        assert_eq!(
            words,
            [
                0x1000,
                7,
                (-3_i64) as u64,
                u64::from(0.5_f32.to_bits()),
                0.25_f64.to_bits(),
                1,
                0,
                0
            ]
        );
    }

    #[test]
    fn a_signature_that_names_its_scratch_is_not_counted_twice() {
        // Triton 3.7's metadata may record the two scratch addresses inside the signature, and the
        // launch appends those two slots itself. A module of two parameters therefore takes two
        // packed words, not four.
        let module = metadata(
            r#"{"x":"*bf16","rows":"i32","global_scratch":"*i8","profile_scratch":"*i8"}"#,
            0,
        );
        let mut arguments = Arguments::new("operator");
        arguments
            .pointer("x", DevicePtr::from_address(0x1000))
            .u32("rows", 7);
        assert_eq!(arguments.parameters(&module).unwrap().len(), 4);
    }

    #[test]
    fn a_list_of_another_length_is_refused_with_both_lists() {
        let module = metadata(r#"{"x":"*bf16","rows":"i32","eps":"fp32"}"#, 0);
        let mut short = Arguments::new("rms_norm");
        short
            .pointer("x", DevicePtr::from_address(0x1000))
            .u32("rows", 7);

        let (operator, kernel, problem) = refusal(&mut short, &module);
        assert_eq!((operator, kernel.as_str()), ("rms_norm", "_kernel"));
        assert!(problem.contains("packs 2 words (`x`, `rows`)"), "{problem}");
        assert!(problem.contains("takes 3 parameters"), "{problem}");
        assert!(problem.contains("`eps`: fp32"), "{problem}");
    }

    #[test]
    fn a_word_named_for_another_parameter_is_refused_by_position_and_both_names() {
        // Two words of one type packed in each other's place: the count and every kind agree, and
        // only the names tell the program would read the row count as the column count.
        let module = metadata(r#"{"x":"*bf16","rows":"i32","columns":"i32"}"#, 0);
        let mut swapped = Arguments::new("operator");
        swapped
            .pointer("x", DevicePtr::from_address(0x1000))
            .u32("columns", 64)
            .u32("rows", 7);

        let (operator, kernel, problem) = refusal(&mut swapped, &module);
        assert_eq!((operator, kernel.as_str()), ("operator", "_kernel"));
        assert!(
            problem.contains("packs `columns` at position 1, where the kernel takes `rows`"),
            "{problem}"
        );
    }

    #[test]
    fn a_scalar_where_the_program_reads_an_address_is_refused_and_so_is_the_reverse() {
        let module = metadata(r#"{"x":"*bf16","rows":"i32"}"#, 0);

        let mut scalar = Arguments::new("operator");
        scalar.u64("x", 0x1000).u32("rows", 7);
        let (_, _, problem) = refusal(&mut scalar, &module);
        assert!(
            problem.contains("the integer 4096 for `x` at position 0"),
            "{problem}"
        );
        assert!(problem.contains("`*bf16`"), "{problem}");

        let mut address = Arguments::new("operator");
        address
            .pointer("x", DevicePtr::from_address(0x1000))
            .pointer("rows", DevicePtr::from_address(0x2000));
        let (_, _, problem) = refusal(&mut address, &module);
        assert!(
            problem.contains("the device address 0x2000 for `rows` at position 1"),
            "{problem}"
        );
    }

    #[test]
    fn an_integer_the_parameters_type_does_not_hold_is_refused() {
        // A `u32` fills an `i32` parameter while the `i32` holds it; beyond `i32::MAX` the program
        // would read a negative number.
        let module = metadata(r#"{"rows":"i32","bits":"u16"}"#, 0);

        let mut wide = Arguments::new("operator");
        wide.u32("rows", 3_000_000_000).u32("bits", 1);
        let (_, _, problem) = refusal(&mut wide, &module);
        assert!(
            problem.contains("the integer 3000000000 for `rows` at position 0"),
            "{problem}"
        );
        assert!(
            problem.contains("`i32` parameter does not hold"),
            "{problem}"
        );

        let mut negative = Arguments::new("operator");
        negative.i32("rows", 1).i64("bits", -1);
        let (_, _, problem) = refusal(&mut negative, &module);
        assert!(problem.contains("the integer -1 for `bits`"), "{problem}");
        assert!(problem.contains("`u16`"), "{problem}");
    }

    #[test]
    fn a_float_and_an_integer_never_fill_each_others_parameters() {
        let module = metadata(r#"{"eps":"fp32","rows":"i32"}"#, 0);

        let mut integer = Arguments::new("operator");
        integer.u32("eps", 1).u32("rows", 7);
        let (_, _, problem) = refusal(&mut integer, &module);
        assert!(problem.contains("`eps` at position 0"), "{problem}");

        let mut float = Arguments::new("operator");
        float.f32("eps", 1e-6).f32("rows", 7.0);
        let (_, _, problem) = refusal(&mut float, &module);
        assert!(problem.contains("the f32 7 for `rows`"), "{problem}");

        // An `f64` is no `fp32`: the program would read half of it.
        let mut double = Arguments::new("operator");
        double.f64("eps", 1e-6).u32("rows", 7);
        let (_, _, problem) = refusal(&mut double, &module);
        assert!(problem.contains("the f64"), "{problem}");
    }

    #[test]
    fn a_parameter_of_a_type_no_word_fills_is_refused_by_name() {
        let module = metadata(r#"{"descriptor":"nvTmaDesc"}"#, 0);
        let mut arguments = Arguments::new("operator");
        arguments.u64("descriptor", 0);
        let (_, _, problem) = refusal(&mut arguments, &module);
        assert!(
            problem.contains("parameter 0 `descriptor` is `nvTmaDesc`"),
            "{problem}"
        );
    }

    #[test]
    fn global_scratch_is_a_value_the_caller_supplies_and_its_absence_is_refused() {
        let module = metadata(r#"{"x":"*bf16"}"#, 4096);
        let mut arguments = Arguments::new("operator");
        arguments.pointer("x", DevicePtr::from_address(0x1000));
        arguments.scratch(Some(DevicePtr::from_address(0x2000)));
        // One declared parameter and the two scratch addresses, the global one supplied.
        assert_eq!(
            read(arguments.parameters(&module).unwrap()),
            [0x1000, 0x2000, 0]
        );

        let mut without = Arguments::new("operator");
        without.pointer("x", DevicePtr::from_address(0x1000));
        let (_, _, problem) = refusal(&mut without, &module);
        assert!(problem.contains("global scratch"), "{problem}");
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
