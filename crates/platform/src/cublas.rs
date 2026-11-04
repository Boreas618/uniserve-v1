//! cuBLASLt bindings (`libcublasLt.so.13`), loaded at run time: the BF16 and
//! FP32 matrix multiplications of `nn::Linear` and the language model's
//! projections (`specs/worker.md` §9).
//!
//! cuBLASLt is bound here and nowhere else. The algorithm of a shape is
//! chosen once, offline, when the GEMM program's tuning data is made (§5.4):
//! this module turns a description of the multiplication into cuBLASLt
//! descriptors, offers the algorithms cuBLASLt proposes for it, and runs the
//! recorded one with the program's fixed workspace. Service never measures.
//!
//! A recorded algorithm is cuBLASLt's own opaque 64 bytes plus the workspace
//! size it needs; both are tuning data, so they travel as bytes
//! ([`Algorithm::to_bytes`], [`Algorithm::from_bytes`]).
//!
//! Every call enters the device's primary context on the calling thread, so
//! the caller needs no current context of its own, and every launch is issued
//! on the caller's stream, so a captured graph holds it like any other work.

use std::ffi::{CStr, c_char, c_void};
use std::sync::{Arc, OnceLock};

use libloading::Library;

use crate::driver::Handle;
use crate::{DataType, Device, DeviceId, DeviceRange, Error, Result, Stream};

const LIBCUBLASLT: &str = "libcublasLt.so.13";

/// `cublasStatus_t`.
type Status = i32;
const STATUS_SUCCESS: Status = 0;

/// The accumulation type (`cublasComputeType_t`): what products and sums are
/// formed in, which is what `ops::Op::tolerance` derives its bound from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComputeType {
    /// `CUBLAS_COMPUTE_32F`: FP32 products and sums.
    F32,
    /// `CUBLAS_COMPUTE_32F_FAST_TF32`: FP32 accumulation with inputs
    /// down-converted to TF32. The video VAE's convolutions allow TF32; a
    /// GEMM uses it only where the program's tolerance permits.
    Tf32,
    /// `CUBLAS_COMPUTE_32F_FAST_16BF`: FP32 accumulation with inputs
    /// down-converted to BF16.
    Bf16,
}

impl ComputeType {
    /// `cublasComputeType_t`.
    fn raw(self) -> i32 {
        match self {
            Self::F32 => 68,  // CUBLAS_COMPUTE_32F
            Self::Bf16 => 75, // CUBLAS_COMPUTE_32F_FAST_16BF
            Self::Tf32 => 77, // CUBLAS_COMPUTE_32F_FAST_TF32
        }
    }

    /// The scale type of `alpha` and `beta` with this compute type: FP32 for
    /// all of them, so the host passes `f32` values.
    fn scale(self) -> i32 {
        0 // CUDA_R_32F
    }
}

/// One matrix's layout (`cublasLtMatrixLayout_t`) in cuBLASLt's column-major
/// convention: `rows` rows, `cols` columns, `stride` the distance in
/// elements between two consecutive columns (the leading dimension), and the
/// batch fields for a batched multiplication.
///
/// A row-major matrix is described as its transpose; [`MatmulDescriptor`]
/// carries the flags that say so.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixLayout {
    pub rows: u64,
    pub cols: u64,
    pub stride: i64,
    pub batch: u64,
    /// Distance in elements between two batches; ignored when `batch` is 1.
    pub batch_stride: i64,
    pub dtype: DataType,
}

impl MatrixLayout {
    /// A contiguous column-major `rows × cols` matrix, one batch.
    pub fn column_major(rows: u64, cols: u64, dtype: DataType) -> Self {
        Self {
            rows,
            cols,
            stride: rows as i64,
            batch: 1,
            batch_stride: 0,
            dtype,
        }
    }

    /// A contiguous row-major `rows × cols` matrix, one batch, stored as the
    /// column-major `cols × rows` matrix.
    pub fn row_major(rows: u64, cols: u64, dtype: DataType) -> Self {
        Self {
            rows: cols,
            cols: rows,
            stride: cols as i64,
            batch: 1,
            batch_stride: 0,
            dtype,
        }
    }

    /// The elements the matrix spans, batch stride included.
    fn elements(&self) -> u64 {
        let one = self.rows * self.cols;
        if self.batch <= 1 {
            one
        } else {
            (self.batch - 1) * self.batch_stride.unsigned_abs() + one
        }
    }
}

/// What the multiplication does with `C` and `D` (`cublasLtEpilogue_t`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Epilogue {
    /// `D = alpha · op(A) · op(B) + beta · C`.
    Default,
    /// `D = alpha · op(A) · op(B) + bias + beta · C`, the bias broadcast over
    /// `D`'s rows: a linear layer's bias add fused into the GEMM.
    Bias,
}

impl Epilogue {
    fn raw(self) -> i32 {
        match self {
            Self::Default => 1, // CUBLASLT_EPILOGUE_DEFAULT
            Self::Bias => 4,    // CUBLASLT_EPILOGUE_BIAS
        }
    }
}

/// The order the caller stores the operands in.
///
/// cuBLASLt only multiplies column-major matrices: a row-major `D = A · B` is
/// the column-major `Dᵀ = Bᵀ · Aᵀ`, so the library is handed the operands
/// swapped, and `m` and `n` with them. The descriptor keeps the caller's
/// names — `a` is the caller's left operand — and this order says which way
/// round the library is given them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    ColumnMajor,
    RowMajor,
}

/// One matrix multiplication: element and compute types, the four matrix
/// layouts, the transpose flags, and the epilogue. It is the key under which
/// a GEMM program records its tuning data (§5.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulDescriptor {
    pub compute: ComputeType,
    /// The order `a`, `b`, `c` and `d` are stored in.
    pub order: Order,
    pub a: MatrixLayout,
    pub b: MatrixLayout,
    pub c: MatrixLayout,
    pub d: MatrixLayout,
    pub transpose_a: bool,
    pub transpose_b: bool,
    pub epilogue: Epilogue,
}

impl MatmulDescriptor {
    /// A row-major `m × k` times `k × n` multiplication of `dtype` into
    /// `dtype`: `D = A · B`, with `C` aliasing `D` under `beta = 0`.
    pub fn row_major(m: u64, n: u64, k: u64, dtype: DataType, compute: ComputeType) -> Self {
        Self {
            compute,
            order: Order::RowMajor,
            a: MatrixLayout::row_major(m, k, dtype),
            b: MatrixLayout::row_major(k, n, dtype),
            c: MatrixLayout::row_major(m, n, dtype),
            d: MatrixLayout::row_major(m, n, dtype),
            // The row-major form swaps the operands rather than transposing
            // them: `op(A)` is `Bᵀ` there, which is `b` as stored.
            transpose_a: false,
            transpose_b: false,
            epilogue: Epilogue::Default,
        }
    }

    /// The bytes each operand spans, for checking the caller's buffers.
    pub fn bytes(&self) -> MatmulBytes {
        let element = self.d.dtype.bytes() as u64;
        MatmulBytes {
            a: self.a.elements() * self.a.dtype.bytes() as u64,
            b: self.b.elements() * self.b.dtype.bytes() as u64,
            c: self.c.elements() * self.c.dtype.bytes() as u64,
            d: self.d.elements() * element,
            bias: self.d.rows * self.d.dtype.bytes() as u64,
        }
    }

    fn check(&self) -> Result<()> {
        if self.order == Order::RowMajor && (self.transpose_a || self.transpose_b) {
            return Err(Error::invalid(
                "a row-major multiplication reaches the library by swapping its operands, \
                 so it cannot also transpose them: leave transpose_a and transpose_b false",
            ));
        }
        for (name, layout) in [("A", self.a), ("B", self.b), ("C", self.c), ("D", self.d)] {
            if layout.rows == 0 || layout.cols == 0 || layout.batch == 0 {
                return Err(Error::invalid(format!(
                    "matrix {name} is empty: {} by {}, batch {}",
                    layout.rows, layout.cols, layout.batch
                )));
            }
            if layout.stride < layout.rows as i64 {
                return Err(Error::invalid(format!(
                    "matrix {name} has leading dimension {} for {} rows",
                    layout.stride, layout.rows
                )));
            }
        }
        Ok(())
    }
}

/// The bytes each operand of a multiplication spans.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatmulBytes {
    pub a: u64,
    pub b: u64,
    pub c: u64,
    pub d: u64,
    /// The bias vector of [`Epilogue::Bias`]: one element per row of `D` as
    /// the library sees it, of `D`'s type, and one vector for every batch.
    /// For a [`MatmulDescriptor::row_major`] descriptor the library sees the
    /// transposed matrices, so that is the matrix's columns.
    pub bias: u64,
}

/// An algorithm cuBLASLt proposes for one [`MatmulDescriptor`], with the
/// workspace it needs: the tuning data of a GEMM program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Algorithm {
    bytes: [u64; 8],
    workspace_bytes: usize,
}

impl Algorithm {
    /// The algorithm as the 64 bytes cuBLASLt accepts back
    /// (`cublasLtMatmulAlgo_t`), to be recorded in the program's tuning data.
    pub fn to_bytes(self) -> [u8; 64] {
        let mut bytes = [0u8; 64];
        for (index, word) in self.bytes.iter().enumerate() {
            bytes[index * 8..(index + 1) * 8].copy_from_slice(&word.to_ne_bytes());
        }
        bytes
    }

    /// The algorithm recorded in the tuning data, with the workspace it
    /// needs.
    pub fn from_bytes(bytes: &[u8; 64], workspace_bytes: usize) -> Self {
        let mut words = [0u64; 8];
        for (index, word) in words.iter_mut().enumerate() {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(&bytes[index * 8..(index + 1) * 8]);
            *word = u64::from_ne_bytes(raw);
        }
        Self {
            bytes: words,
            workspace_bytes,
        }
    }

    /// The workspace the algorithm needs, in bytes.
    pub fn workspace_bytes(self) -> usize {
        self.workspace_bytes
    }
}

/// The buffers and scalars of one multiplication:
/// `D = alpha · op(A) · op(B) + beta · C`, with the bias of
/// [`Epilogue::Bias`] and the algorithm's workspace.
///
/// Each range states where its buffer starts and how many bytes are valid from
/// there; [`Matmul::run`] checks them against the descriptor before any
/// cuBLASLt call. The storage stays at its address for the work's life,
/// including every replay of a graph that captured it.
#[derive(Clone, Copy, Debug)]
pub struct MatmulArgs {
    pub a: DeviceRange,
    pub b: DeviceRange,
    pub c: DeviceRange,
    pub d: DeviceRange,
    pub alpha: f32,
    pub beta: f32,
    /// The bias vector for [`Epilogue::Bias`]; an empty range otherwise.
    pub bias: DeviceRange,
    /// The workspace the algorithm runs in, at least its need.
    pub workspace: DeviceRange,
}

/// `cublasLtMatmulHeuristicResult_t`.
#[repr(C)]
#[derive(Clone, Copy)]
struct HeuristicResult {
    algo: [u64; 8],
    workspace_bytes: usize,
    status: Status,
    waves: f32,
    reserved: [i32; 4],
}

/// The most results one search returns.
const HEURISTIC_RESULTS: usize = 8;

// `cublasLt.h`'s numbers, each named after its enumerator; the unit test
// `the_attribute_numbers_are_the_headers` compares every one of them with the
// header itself, because a wrong number here is a wrong computation or a
// refused call, never a compile error.
const PREFERENCE_WORKSPACE_SIZE: i32 = 1; // CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES
const DESC_TRANSA: i32 = 3; // CUBLASLT_MATMUL_DESC_TRANSA
const DESC_TRANSB: i32 = 4; // CUBLASLT_MATMUL_DESC_TRANSB
const DESC_EPILOGUE: i32 = 7; // CUBLASLT_MATMUL_DESC_EPILOGUE
const DESC_BIAS_POINTER: i32 = 8; // CUBLASLT_MATMUL_DESC_BIAS_POINTER
const DESC_BIAS_DATA_TYPE: i32 = 26; // CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE
const LAYOUT_BATCH_COUNT: i32 = 5; // CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT
const LAYOUT_BATCH_STRIDE: i32 = 6; // CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET

/// The cuBLASLt entry points this module uses, resolved once per process.
struct Api {
    create: unsafe extern "C" fn(*mut Handle) -> Status,
    destroy: unsafe extern "C" fn(Handle) -> Status,
    matmul_desc_create: unsafe extern "C" fn(*mut Handle, i32, i32) -> Status,
    matmul_desc_destroy: unsafe extern "C" fn(Handle) -> Status,
    matmul_desc_set_attribute: unsafe extern "C" fn(Handle, i32, *const c_void, usize) -> Status,
    layout_create: unsafe extern "C" fn(*mut Handle, i32, u64, u64, i64) -> Status,
    layout_destroy: unsafe extern "C" fn(Handle) -> Status,
    layout_set_attribute: unsafe extern "C" fn(Handle, i32, *const c_void, usize) -> Status,
    preference_create: unsafe extern "C" fn(*mut Handle) -> Status,
    preference_destroy: unsafe extern "C" fn(Handle) -> Status,
    preference_set_attribute: unsafe extern "C" fn(Handle, i32, *const c_void, usize) -> Status,
    algo_get_heuristic: unsafe extern "C" fn(
        Handle,
        Handle,
        Handle,
        Handle,
        Handle,
        Handle,
        Handle,
        i32,
        *mut HeuristicResult,
        *mut i32,
    ) -> Status,
    matmul: unsafe extern "C" fn(
        Handle,
        Handle,
        *const c_void,
        *const c_void,
        Handle,
        *const c_void,
        Handle,
        *const c_void,
        *const c_void,
        Handle,
        *mut c_void,
        Handle,
        *const [u64; 8],
        *mut c_void,
        usize,
        Handle,
    ) -> Status,
    get_status_name: unsafe extern "C" fn(Status) -> *const c_char,
    get_status_string: unsafe extern "C" fn(Status) -> *const c_char,
    get_version: unsafe extern "C" fn() -> usize,
    // Declared last: the library outlives every copied function pointer.
    _library: Library,
}

/// Fails unless `range` states at least `needed` bytes from its address.
fn require_room(range: DeviceRange, needed: u64, what: &'static str) -> Result<()> {
    if (range.bytes() as u64) < needed {
        return Err(Error::invalid(format!(
            "the {what} range states {} bytes, but the multiplication needs {needed}",
            range.bytes()
        )));
    }
    Ok(())
}

fn api() -> Result<&'static Api> {
    static API: OnceLock<Result<Api>> = OnceLock::new();

    API.get_or_init(|| {
        // SAFETY: loading cuBLASLt runs only its own initializers.
        let library = unsafe { Library::new(LIBCUBLASLT) }.map_err(|source| Error::Load {
            library: LIBCUBLASLT,
            source: Arc::new(source),
        })?;
        // SAFETY: each signature matches the `cublasLt.h` prototype of the
        // symbol.
        unsafe {
            Ok(Api {
                create: symbol(&library, "cublasLtCreate")?,
                destroy: symbol(&library, "cublasLtDestroy")?,
                matmul_desc_create: symbol(&library, "cublasLtMatmulDescCreate")?,
                matmul_desc_destroy: symbol(&library, "cublasLtMatmulDescDestroy")?,
                matmul_desc_set_attribute: symbol(&library, "cublasLtMatmulDescSetAttribute")?,
                layout_create: symbol(&library, "cublasLtMatrixLayoutCreate")?,
                layout_destroy: symbol(&library, "cublasLtMatrixLayoutDestroy")?,
                layout_set_attribute: symbol(&library, "cublasLtMatrixLayoutSetAttribute")?,
                preference_create: symbol(&library, "cublasLtMatmulPreferenceCreate")?,
                preference_destroy: symbol(&library, "cublasLtMatmulPreferenceDestroy")?,
                preference_set_attribute: symbol(&library, "cublasLtMatmulPreferenceSetAttribute")?,
                algo_get_heuristic: symbol(&library, "cublasLtMatmulAlgoGetHeuristic")?,
                matmul: symbol(&library, "cublasLtMatmul")?,
                get_status_name: symbol(&library, "cublasLtGetStatusName")?,
                get_status_string: symbol(&library, "cublasLtGetStatusString")?,
                get_version: symbol(&library, "cublasLtGetVersion")?,
                _library: library,
            })
        }
    })
    .as_ref()
    .map_err(Clone::clone)
}

/// # Safety
/// `T` must be the function-pointer type of the symbol's C prototype.
unsafe fn symbol<T: Copy>(library: &Library, name: &'static str) -> Result<T> {
    // SAFETY: forwarded from the caller.
    unsafe { library.get::<T>(name.as_bytes()) }
        .map(|symbol| *symbol)
        .map_err(|source| Error::Symbol {
            library: LIBCUBLASLT,
            symbol: name,
            source: Arc::new(source),
        })
}

impl Api {
    fn check(&self, status: Status, call: &'static str) -> Result<()> {
        if status == STATUS_SUCCESS {
            return Ok(());
        }
        // SAFETY: cuBLASLt returns static, NUL-terminated strings owned by
        // the never-unloaded library.
        let (name, message) = unsafe {
            (
                text((self.get_status_name)(status)),
                text((self.get_status_string)(status)),
            )
        };
        Err(Error::Blas {
            call,
            code: status,
            name,
            message,
        })
    }
}

/// # Safety
/// `raw` must be null or a NUL-terminated string owned by the library.
unsafe fn text(raw: *const c_char) -> &'static str {
    if raw.is_null() {
        return "unrecognized cuBLASLt status";
    }
    // SAFETY: forwarded from the caller; the string is static.
    unsafe { CStr::from_ptr(raw) }
        .to_str()
        .unwrap_or("non-UTF-8 cuBLASLt status")
}

/// A cuBLASLt handle for one device's primary context.
///
/// `nn` keeps one per device: creating a handle costs a driver round trip and
/// the handle is reusable across multiplications.
pub struct Matmul {
    api: &'static Api,
    device: DeviceId,
    handle: Handle,
}

// SAFETY: a cuBLASLt handle is used by one thread at a time; moving it
// between threads is safe because every call enters the device's context.
unsafe impl Send for Matmul {}

impl Matmul {
    /// Creates a cuBLASLt handle for `device`.
    ///
    /// # Errors
    ///
    /// [`Error::Load`] when `libcublasLt.so.13` is absent, [`Error::Symbol`]
    /// when it predates an entry point, [`Error::Blas`] when the handle
    /// cannot be created.
    pub fn new(device: &Device) -> Result<Self> {
        let api = api()?;
        let _context = device.enter()?;
        let mut handle = std::ptr::null_mut();
        // SAFETY: the library writes one handle to the local.
        api.check(unsafe { (api.create)(&mut handle) }, "cublasLtCreate")?;
        Ok(Self {
            api,
            device: device.id(),
            handle,
        })
    }

    /// The device the handle belongs to.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// The algorithms cuBLASLt proposes for `descriptor`, best first, at most
    /// `limit`, none needing more than `workspace_bytes` of workspace.
    ///
    /// This is how tuning data is made: the program's package runs it once
    /// on an idle GPU (§5.4), records one [`Algorithm`] per descriptor, and
    /// service never asks the heuristic again.
    pub fn candidates(
        &self,
        descriptor: &MatmulDescriptor,
        workspace_bytes: usize,
        limit: usize,
    ) -> Result<Vec<Algorithm>> {
        descriptor.check()?;
        let limit = limit.clamp(1, HEURISTIC_RESULTS);
        let device = Device::new(self.device)?;
        let _context = device.enter()?;
        let descriptions = Descriptions::new(self.api, descriptor)?;
        let preference = Preference::new(self.api, workspace_bytes)?;
        let mut results = [HeuristicResult {
            algo: [0; 8],
            workspace_bytes: 0,
            status: 0,
            waves: 0.0,
            reserved: [0; 4],
        }; HEURISTIC_RESULTS];
        let mut found = 0;
        self.api.check(
            // SAFETY: every descriptor is live; the library writes at most
            // `HEURISTIC_RESULTS` results and their count.
            unsafe {
                (self.api.algo_get_heuristic)(
                    self.handle,
                    descriptions.matmul,
                    descriptions.a,
                    descriptions.b,
                    descriptions.c,
                    descriptions.d,
                    preference.raw,
                    limit as i32,
                    results.as_mut_ptr(),
                    &mut found,
                )
            },
            "cublasLtMatmulAlgoGetHeuristic",
        )?;
        Ok(results[..found as usize]
            .iter()
            .map(|result| Algorithm {
                bytes: result.algo,
                workspace_bytes: result.workspace_bytes,
            })
            .collect())
    }

    /// Runs `descriptor` with the recorded `algorithm`:
    /// `D = alpha · op(A) · op(B) + beta · C` (plus the bias for
    /// [`Epilogue::Bias`]), issued on `stream`.
    ///
    /// The call issues device work and returns without waiting for it; on a
    /// capturing stream it is captured into the graph.
    ///
    /// Before any cuBLASLt call, every range is checked against what the
    /// descriptor's shapes need ([`MatmulDescriptor::bytes`]), the bias range
    /// against [`Epilogue::Bias`]'s vector, and the workspace against
    /// `algorithm`'s need; a range too small is refused by name. The caller
    /// keeps the storage alive until the stream work completes.
    pub fn run(
        &self,
        descriptor: &MatmulDescriptor,
        algorithm: &Algorithm,
        args: &MatmulArgs,
        stream: &Stream,
    ) -> Result<()> {
        descriptor.check()?;
        let bytes = descriptor.bytes();
        require_room(args.a, bytes.a, "A")?;
        require_room(args.b, bytes.b, "B")?;
        require_room(args.c, bytes.c, "C")?;
        require_room(args.d, bytes.d, "D")?;
        if descriptor.epilogue == Epilogue::Bias {
            require_room(args.bias, bytes.bias, "bias")?;
        }
        if args.workspace.bytes() < algorithm.workspace_bytes {
            return Err(Error::invalid(format!(
                "the multiplication needs {} bytes of workspace, {} given",
                algorithm.workspace_bytes,
                args.workspace.bytes()
            )));
        }
        if stream.device() != self.device {
            return Err(Error::invalid(format!(
                "the multiplication runs on {}, not on {}'s stream",
                self.device,
                stream.device()
            )));
        }
        let device = Device::new(self.device)?;
        let _context = device.enter()?;
        let descriptions = Descriptions::new(self.api, descriptor)?;
        if descriptor.epilogue == Epilogue::Bias {
            descriptions.set_bias(self.api, args.bias.ptr().address(), descriptor.bias_type())?;
        }
        let (alpha, beta) = (args.alpha, args.beta);
        // The row-major form reaches the library with its operands swapped,
        // so the pointers go where their layouts went.
        let (left, right) = match descriptor.order {
            Order::ColumnMajor => (args.a, args.b),
            Order::RowMajor => (args.b, args.a),
        };
        // SAFETY: forwarded from the caller; every pointer is device storage
        // live for the launch, every descriptor is live, the stream is live.
        self.api.check(
            unsafe {
                (self.api.matmul)(
                    self.handle,
                    descriptions.matmul,
                    std::ptr::from_ref(&alpha).cast(),
                    left.ptr().address() as *const c_void,
                    descriptions.a,
                    right.ptr().address() as *const c_void,
                    descriptions.b,
                    std::ptr::from_ref(&beta).cast(),
                    args.c.ptr().address() as *const c_void,
                    descriptions.c,
                    args.d.ptr().address() as *mut c_void,
                    descriptions.d,
                    &algorithm.bytes,
                    args.workspace.ptr().address() as *mut c_void,
                    args.workspace.bytes(),
                    stream.handle(),
                )
            },
            "cublasLtMatmul",
        )
    }

    /// The cuBLASLt version this process loaded (`cublasLtGetVersion`).
    pub fn version(&self) -> usize {
        // SAFETY: takes no arguments and only reads the library's version.
        unsafe { (self.api.get_version)() }
    }
}

impl Drop for Matmul {
    fn drop(&mut self) {
        // The owner drained every stream it ran on; destroying the handle
        // needs no current context.
        // SAFETY: the handle is this value's and is destroyed once.
        let _ = unsafe { (self.api.destroy)(self.handle) };
    }
}

impl MatmulDescriptor {
    /// The element type cuBLASLt expects the bias in: `D`'s.
    fn bias_type(&self) -> i32 {
        self.d.dtype.cuda()
    }
}

/// The four matrix layouts and the operation descriptor of one
/// multiplication, destroyed together.
struct Descriptions {
    matmul: Handle,
    a: Handle,
    b: Handle,
    c: Handle,
    d: Handle,
}

impl Descriptions {
    fn new(api: &'static Api, descriptor: &MatmulDescriptor) -> Result<Self> {
        let mut descriptions = Self {
            matmul: std::ptr::null_mut(),
            a: std::ptr::null_mut(),
            b: std::ptr::null_mut(),
            c: std::ptr::null_mut(),
            d: std::ptr::null_mut(),
        };
        // The library's "A" is the caller's left operand in column-major
        // order, and its right one in row-major order: `Dᵀ = Bᵀ · Aᵀ`.
        let (left, right) = match descriptor.order {
            Order::ColumnMajor => (&descriptor.a, &descriptor.b),
            Order::RowMajor => (&descriptor.b, &descriptor.a),
        };
        for (slot, layout) in [
            (&mut descriptions.a, left),
            (&mut descriptions.b, right),
            (&mut descriptions.c, &descriptor.c),
            (&mut descriptions.d, &descriptor.d),
        ] {
            *slot = Self::layout(api, layout)?;
        }
        let mut matmul = std::ptr::null_mut();
        api.check(
            // SAFETY: the compute and scale types are cuBLASLt's constants.
            unsafe {
                (api.matmul_desc_create)(
                    &mut matmul,
                    descriptor.compute.raw(),
                    descriptor.compute.scale(),
                )
            },
            "cublasLtMatmulDescCreate",
        )?;
        descriptions.matmul = matmul;
        let transa = i32::from(descriptor.transpose_a);
        let transb = i32::from(descriptor.transpose_b);
        let epilogue = descriptor.epilogue.raw();
        for (attribute, value, size) in [
            (DESC_TRANSA, &transa as *const i32, size_of::<i32>()),
            (DESC_TRANSB, &transb as *const i32, size_of::<i32>()),
            (DESC_EPILOGUE, &epilogue as *const i32, size_of::<i32>()),
        ] {
            api.check(
                // SAFETY: each attribute's documented type is int32_t and the
                // local outlives the call.
                unsafe { (api.matmul_desc_set_attribute)(matmul, attribute, value.cast(), size) },
                "cublasLtMatmulDescSetAttribute",
            )?;
        }
        Ok(descriptions)
    }

    /// Sets the bias vector of the multiplication: the host pointer value of
    /// an argument is the attribute's content, not the storage it addresses.
    fn set_bias(&self, api: &'static Api, bias: u64, data_type: i32) -> Result<()> {
        let pointer = bias as *const c_void;
        for (attribute, value, size) in [
            (
                DESC_BIAS_POINTER,
                std::ptr::from_ref(&pointer).cast::<c_void>(),
                size_of::<*const c_void>(),
            ),
            (
                DESC_BIAS_DATA_TYPE,
                std::ptr::from_ref(&data_type).cast::<c_void>(),
                size_of::<i32>(),
            ),
        ] {
            api.check(
                // SAFETY: the bias pointer attribute's type is the pointer value
                // and the data type attribute's is int32_t; the locals outlive
                // the call.
                unsafe { (api.matmul_desc_set_attribute)(self.matmul, attribute, value, size) },
                "cublasLtMatmulDescSetAttribute",
            )?;
        }
        Ok(())
    }

    fn layout(api: &'static Api, layout: &MatrixLayout) -> Result<Handle> {
        let mut raw = std::ptr::null_mut();
        api.check(
            // SAFETY: the library writes one handle to the local.
            unsafe {
                (api.layout_create)(
                    &mut raw,
                    layout.dtype.cuda(),
                    layout.rows,
                    layout.cols,
                    layout.stride,
                )
            },
            "cublasLtMatrixLayoutCreate",
        )?;
        if layout.batch > 1 {
            let batch = layout.batch as i32;
            let stride = layout.batch_stride;
            for (attribute, value, size) in [
                (
                    LAYOUT_BATCH_COUNT,
                    std::ptr::from_ref(&batch).cast::<c_void>(),
                    size_of::<i32>(),
                ),
                (
                    LAYOUT_BATCH_STRIDE,
                    std::ptr::from_ref(&stride).cast::<c_void>(),
                    size_of::<i64>(),
                ),
            ] {
                if let Err(error) = api.check(
                    // SAFETY: the batch count's documented type is int32_t and
                    // the strided offset's is int64_t; the locals outlive the
                    // call.
                    unsafe { (api.layout_set_attribute)(raw, attribute, value, size) },
                    "cublasLtMatrixLayoutSetAttribute",
                ) {
                    // SAFETY: the handle was created above and is owned here.
                    unsafe { (api.layout_destroy)(raw) };
                    return Err(error);
                }
            }
        }
        Ok(raw)
    }
}

impl Drop for Descriptions {
    fn drop(&mut self) {
        let Some(api) = api().ok() else {
            return;
        };
        // SAFETY: each handle is live, owned here, and destroyed once.
        unsafe {
            for handle in [self.a, self.b, self.c, self.d] {
                if !handle.is_null() {
                    (api.layout_destroy)(handle);
                }
            }
            if !self.matmul.is_null() {
                (api.matmul_desc_destroy)(self.matmul);
            }
        }
    }
}

/// The preference of an algorithm search: the workspace ceiling the
/// multiplication may use, which is the program's fixed workspace for its
/// shape (§5.4).
struct Preference {
    api: &'static Api,
    raw: Handle,
}

impl Preference {
    fn new(api: &'static Api, workspace_bytes: usize) -> Result<Self> {
        let mut raw = std::ptr::null_mut();
        api.check(
            // SAFETY: the library writes one handle to the local.
            unsafe { (api.preference_create)(&mut raw) },
            "cublasLtMatmulPreferenceCreate",
        )?;
        let preference = Self { api, raw };
        api.check(
            // SAFETY: the attribute's documented type is size_t and the local
            // outlives the call.
            unsafe {
                (api.preference_set_attribute)(
                    raw,
                    PREFERENCE_WORKSPACE_SIZE,
                    std::ptr::from_ref(&workspace_bytes).cast(),
                    size_of::<usize>(),
                )
            },
            "cublasLtMatmulPreferenceSetAttribute",
        )?;
        Ok(preference)
    }
}

impl Drop for Preference {
    fn drop(&mut self) {
        // SAFETY: the handle is live and owned here; destroyed once.
        unsafe { (self.api.preference_destroy)(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every attribute number this binding passes to cuBLASLt is compared
    /// with `cublasLt.h`: the workspace preference was 0, which is the search
    /// mode attribute, and every heuristic call was refused for it. A wrong
    /// number here is never a compile error.
    #[test]
    fn the_attribute_numbers_are_the_headers() {
        crate::header::assert_constants(
            "cublasLt.h",
            &[
                (
                    "CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES",
                    PREFERENCE_WORKSPACE_SIZE as i64,
                ),
                ("CUBLASLT_MATMUL_DESC_TRANSA", DESC_TRANSA as i64),
                ("CUBLASLT_MATMUL_DESC_TRANSB", DESC_TRANSB as i64),
                ("CUBLASLT_MATMUL_DESC_EPILOGUE", DESC_EPILOGUE as i64),
                (
                    "CUBLASLT_MATMUL_DESC_BIAS_POINTER",
                    DESC_BIAS_POINTER as i64,
                ),
                (
                    "CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE",
                    DESC_BIAS_DATA_TYPE as i64,
                ),
                (
                    "CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT",
                    LAYOUT_BATCH_COUNT as i64,
                ),
                (
                    "CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET",
                    LAYOUT_BATCH_STRIDE as i64,
                ),
            ],
        );
    }

    #[test]
    fn a_recorded_algorithm_round_trips_through_its_tuning_bytes() {
        let algorithm = Algorithm::from_bytes(&[7u8; 64], 4096);

        assert_eq!(
            Algorithm::from_bytes(&algorithm.to_bytes(), 4096),
            algorithm
        );
        assert_eq!(algorithm.workspace_bytes(), 4096);
        assert_eq!(algorithm.to_bytes(), [7u8; 64]);
    }

    #[test]
    fn row_major_layouts_describe_the_transposed_matrices() {
        let descriptor = MatmulDescriptor::row_major(4, 6, 3, DataType::Bf16, ComputeType::F32);

        assert_eq!(
            descriptor.a,
            MatrixLayout {
                rows: 3,
                cols: 4,
                stride: 3,
                batch: 1,
                batch_stride: 0,
                dtype: DataType::Bf16,
            }
        );
        assert_eq!((descriptor.b.rows, descriptor.b.cols), (6, 3));
        // The row-major form reaches the library swapped, not transposed.
        assert_eq!(descriptor.order, Order::RowMajor);
        assert!(!descriptor.transpose_a && !descriptor.transpose_b);
        assert_eq!(
            descriptor.bytes(),
            MatmulBytes {
                a: 4 * 3 * 2,
                b: 3 * 6 * 2,
                c: 4 * 6 * 2,
                d: 4 * 6 * 2,
                // One bias vector for every batch: the rows of D as
                // cuBLASLt sees it, which for a row-major descriptor are the
                // matrix's columns (6), of D's type (2 bytes).
                bias: 6 * 2,
            }
        );
    }

    #[test]
    fn an_empty_matrix_is_rejected_before_any_driver_call() {
        let mut descriptor = MatmulDescriptor::row_major(4, 6, 3, DataType::F32, ComputeType::F32);
        descriptor.b.rows = 0;

        let error = descriptor.check().expect_err("an empty matrix is rejected");
        assert!(error.to_string().contains("matrix B is empty"), "{error}");
    }

    #[test]
    fn the_data_types_are_the_cuda_codes() {
        assert_eq!(DataType::F32.cuda(), 0);
        assert_eq!(DataType::Bf16.cuda(), 14);
        assert_eq!(DataType::F16.cuda(), 2);
        assert_eq!(ComputeType::F32.raw(), 68);
        assert_eq!(ComputeType::Tf32.raw(), 77);
        assert_eq!(ComputeType::Bf16.raw(), 75);
    }
}
