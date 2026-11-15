//! Generates the committed bindings of `src/sys/` from the C headers of the libraries the crate
//! loads.
//!
//! The crate opens `libcuda.so.1`, `libnvidia-ml.so.1`, `libnccl.so.2` and `libcupti.so.13` at
//! run time and needs no CUDA toolkit to build, so the numbers and structures it passes them are
//! not generated at build time: this program generates them once from the headers of the versions
//! the crate is written for, and the output is committed. Each generated file carries the version
//! macros of the headers it came from (`CUDA_VERSION`, `NCCL_VERSION_CODE`, `CUPTI_API_VERSION`),
//! which the crate also uses as the oldest library it accepts at load where that is the binding
//! contract.
//!
//! The headers are found through the standard mechanisms of the packages that ship them:
//!
//! - `cuda.h`, `nvml.h` and the CUPTI headers through the `pkg-config` module `cuda-13.0` of
//!   NVIDIA's CUDA packages (`cudart-13.0` for the runtime headers `nccl.h` includes);
//! - `nccl.h` through the compiler's default include path, where NVIDIA's `libnccl-dev` package
//!   installs it.
//!
//! Usage, one library per run, its bindings written to standard output:
//!
//! ```text
//! cargo run -p platform --example bindings -- cuda > crates/platform/src/sys/cuda.rs
//! cargo run -p platform --example bindings -- nvml > crates/platform/src/sys/nvml.rs
//! cargo run -p platform --example bindings -- nccl > crates/platform/src/sys/nccl.rs
//! cargo run -p platform --example bindings -- cupti > crates/platform/src/sys/cupti.rs
//! ```
//!
//! Every item is allowlisted by name: a binding the crate does not use is not generated, and a
//! name the headers no longer declare is missing from the output, which the crate's build then
//! reports at the use.

use std::process::ExitCode;

/// The `pkg-config` module of the CUDA release whose driver API, NVML and CUPTI headers the
/// bindings are generated from.
const CUDA_MODULE: &str = "cuda-13.0";

/// The `pkg-config` module of the same release's runtime headers, which `nccl.h` includes.
const RUNTIME_MODULE: &str = "cudart-13.0";

/// One library's headers and the items generated from them.
struct Library {
    /// The headers bindgen parses, each found on the include path as `#include <header>` finds
    /// it.
    headers: &'static [&'static str],
    /// The `pkg-config` modules whose include directories hold the headers and what they include.
    modules: &'static [&'static str],
    /// Types (structures, unions, enums and typedefs), by exact name.
    types: &'static [&'static str],
    /// Constants (`#define` values and enumerators of anonymous enums), by exact name.
    constants: &'static [&'static str],
}

/// `cuda.h`: the driver API the crate's driver table calls.
const CUDA: Library = Library {
    headers: &["cuda.h"],
    modules: &[CUDA_MODULE],
    types: &[
        "CUresult",
        "CUdevice_attribute",
        "CUmemLocation",
        "CUmemLocationType",
        "CUmemAllocationProp",
        "CUmemAllocationType",
        "CUmemAllocationHandleType",
        "CUmemAllocationGranularity_flags",
        "CUmemAccessDesc",
        "CUmemAccess_flags",
        "CUmemcpyAttributes",
        "CUmemcpySrcAccessOrder",
        "CUmemcpy3DOperand",
        "CUmemcpy3DOperandType",
        "CUDA_MEMCPY3D_BATCH_OP",
        "CUlaunchAttribute",
        "CUlaunchAttributeID",
        "CUlaunchAttributeValue",
        "CUlaunchConfig",
        "CUclusterSchedulingPolicy",
        "CUDA_KERNEL_NODE_PARAMS",
        "CUgraphEdgeData",
        "CUgraphDependencyType",
        "CUgraphNodeType",
        "CUstreamCaptureStatus",
        "CUstreamCaptureMode",
        "CUstreamWaitValue_flags",
        "CUstream_flags",
        "CUevent_flags",
        "CUevent_record_flags",
        "CUevent_wait_flags",
        "CUfunction_attribute",
        "CUipcEventHandle",
        "CUuuid",
        "CUtensorMap",
        "CUtensorMapInterleave",
        "CUtensorMapSwizzle",
        "CUtensorMapL2promotion",
        "CUtensorMapFloatOOBfill",
    ],
    constants: &[
        "CUDA_VERSION",
        "CU_MEMHOSTALLOC_PORTABLE",
        "CU_MEMHOSTREGISTER_PORTABLE",
        "CU_MEMHOSTREGISTER_READ_ONLY",
        "CU_GRAPH_KERNEL_NODE_PORT_DEFAULT",
        "CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC",
        "CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER",
    ],
};

/// `nvml.h`: the device and process queries of the crate's NVML session.
const NVML: Library = Library {
    headers: &["nvml.h"],
    modules: &[CUDA_MODULE],
    types: &["nvmlReturn_t", "nvmlProcessInfo_t", "nvmlMemory_v2_t"],
    constants: &["NVML_VALUE_NOT_AVAILABLE"],
};

/// `nccl.h`: the communicator configuration, rendezvous token and type codes.
const NCCL: Library = Library {
    headers: &["nccl.h"],
    modules: &[RUNTIME_MODULE],
    types: &[
        "ncclResult_t",
        "ncclUniqueId",
        "ncclConfig_t",
        "ncclDataType_t",
        "ncclRedOp_t",
    ],
    // `NCCL_CONFIG_UNDEF_INT` is `INT_MIN` of `<limits.h>`, which bindgen does not evaluate; the
    // crate states it as `i32::MIN`, the same definition.
    constants: &[
        "NCCL_VERSION_CODE",
        "NCCL_UNIQUE_ID_BYTES",
        "NCCL_API_MAGIC",
        "NCCL_SPLIT_NOCOLOR",
    ],
};

/// The CUPTI headers: callback domains and ids, activity kinds and flags, and the configuration
/// codes of the profiling APIs.
const CUPTI: Library = Library {
    headers: &[
        "cupti.h",
        "cupti_pcsampling.h",
        "cupti_pmsampling.h",
        "cupti_profiler_target.h",
    ],
    modules: &[CUDA_MODULE],
    types: &[
        "CUptiResult",
        "CUpti_CallbackDomain",
        "CUpti_ApiCallbackSite",
        "CUpti_CallbackData",
        "CUpti_driver_api_trace_cbid",
        "CUpti_ActivityKind",
        "CUpti_ActivityFlag",
        "CUpti_ActivityAttribute",
        "CUpti_PCSamplingConfigurationAttributeType",
        "CUpti_PCSamplingCollectionMode",
        "CUpti_PCSamplingOutputDataFormat",
        "CUpti_PmSampling_TriggerMode",
        "CUpti_PmSampling_HardwareBuffer_AppendMode",
        "CUpti_PmSampling_DecodeStopReason",
        "CUpti_ProfilerRange",
        "CUpti_ProfilerReplayMode",
    ],
    constants: &["CUPTI_API_VERSION", "CUPTI_STALL_REASON_STRING_SIZE"],
};

fn main() -> ExitCode {
    let name = std::env::args().nth(1).unwrap_or_default();
    let library = match name.as_str() {
        "cuda" => CUDA,
        "nvml" => NVML,
        "nccl" => NCCL,
        "cupti" => CUPTI,
        _ => {
            eprintln!("usage: bindings <cuda|nvml|nccl|cupti>");
            return ExitCode::FAILURE;
        }
    };

    match generate(&library) {
        Ok(bindings) => {
            print!("{bindings}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("bindings: {message}");
            ExitCode::FAILURE
        }
    }
}

/// The bindings of `library`, formatted, with a header naming their provenance.
fn generate(library: &Library) -> Result<String, String> {
    // The headers are included as a translation unit would include them, so the compiler searches
    // the packages' include directories and then its default include path.
    let includes: String = library
        .headers
        .iter()
        .map(|header| format!("#include <{header}>\n"))
        .collect();
    let mut builder = bindgen::Builder::default()
        .header_contents("bindings.h", &includes)
        // Enumerators keep their C names (`CUDA_SUCCESS`, not `cudaError_enum_CUDA_SUCCESS`),
        // as constants of the enum's C type.
        .prepend_enum_name(false)
        .derive_default(true)
        .derive_debug(true)
        .generate_comments(false)
        .formatter(bindgen::Formatter::Rustfmt);

    for module in library.modules {
        let found = pkg_config::Config::new()
            .cargo_metadata(false)
            .env_metadata(false)
            .probe(module)
            .map_err(|error| format!("pkg-config cannot find {module}: {error}"))?;
        for directory in found.include_paths {
            builder = builder.clang_arg(format!("-I{}", directory.display()));
        }
    }
    for name in library.types {
        builder = builder.allowlist_type(format!("^{name}$"));
    }
    for name in library.constants {
        builder = builder.allowlist_var(format!("^{name}$"));
    }

    let headers = library.headers.join("`, `");
    let bindings = builder
        .generate()
        .map_err(|error| format!("generating the bindings of `{headers}`: {error}"))?;

    // bindgen's own first line names its version; the headers' version is their version
    // constant, which every library's allowlist includes.
    Ok(format!(
        "// Generated from `{headers}` by `cargo run -p platform --example bindings`; do not \
         edit.\n{bindings}"
    ))
}
