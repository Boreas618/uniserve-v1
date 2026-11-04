//! Prints what the driver answers to `cuTensorMapEncodeTiled` for the tensor
//! map descriptions the kernels use, and which of them the driver refuses.
//!
//! `TensorMap::encode_tiled` reports the driver's status but not which
//! requirement of the description failed, and for a rank-one tensor map the
//! documented requirements are all met by a description the driver may still
//! refuse. So this program calls the entry point itself, with the pointer
//! forms the wrapper chooses between (no strides at all, and a valid stride
//! array for a rank-one tensor), and then the wrapper's own call, printing
//! every status. One run then separates our description from the driver's
//! rules.
//!
//! Run it on the machine under test:
//!
//! ```text
//! cargo run -p platform --example tensor_map_probe
//! ```
//!
//! It exits 1 when any description was refused.

use std::ffi::c_void;

use platform::{
    Device, DeviceAllocation, DeviceId, Error, Interleave, L2Promotion, NativeLibrary, OobFill,
    PrimaryContext, Swizzle, TensorDataType, TensorMap, TensorMapSpec,
};

/// `CUtensorMap` is 128 opaque bytes that the driver requires be 64-byte
/// aligned.
#[repr(C, align(64))]
struct RawTensorMap {
    opaque: [u64; 16],
}

/// `cuTensorMapEncodeTiled` as `cuda.h` declares it.
type EncodeTiled = unsafe extern "C" fn(
    *mut RawTensorMap,
    i32,
    u32,
    *mut c_void,
    *const u64,
    *const u64,
    *const u32,
    *const u32,
    i32,
    i32,
    i32,
    i32,
) -> i32;

/// The driver status names this probe prints; anything else is its number.
fn status_name(status: i32) -> &'static str {
    match status {
        0 => "CUDA_SUCCESS",
        1 => "CUDA_ERROR_INVALID_VALUE",
        2 => "CUDA_ERROR_OUT_OF_MEMORY",
        3 => "CUDA_ERROR_NOT_INITIALIZED",
        4 => "CUDA_ERROR_DEINITIALIZED",
        100 => "CUDA_ERROR_NO_DEVICE",
        101 => "CUDA_ERROR_INVALID_DEVICE",
        201 => "CUDA_ERROR_INVALID_CONTEXT",
        202 => "CUDA_ERROR_INVALID_HANDLE",
        400 => "CUDA_ERROR_INVALID_HANDLE",
        801 => "CUDA_ERROR_NOT_SUPPORTED",
        1001 => "CUDA_ERROR_UNKNOWN",
        _ => "see cuda.h",
    }
}

/// One description the probe asks the driver to encode.
struct Variant<'a> {
    name: &'a str,
    data_type: i32,
    dims: &'a [u64],
    strides: Option<&'a [u64]>,
    box_dim: &'a [u32],
    element_strides: &'a [u32],
    interleave: i32,
    swizzle: i32,
    l2_promotion: i32,
    oob_fill: i32,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("the probe could not start: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Error> {
    let device = Device::new(DeviceId(0))?;
    let context = PrimaryContext::retain(device.id())?;
    let _guard = context.enter()?;
    let storage = DeviceAllocation::linear(device.id(), 4096)?;
    let address = storage.address();

    let driver = NativeLibrary::open("libcuda.so.1")?;
    // SAFETY: both symbols are the driver's, resolved with the signatures
    // `cuda.h` declares: `cuTensorMapEncodeTiled` as above and
    // `cuDriverGetVersion(int *)`.
    let encode: EncodeTiled = unsafe { driver.entry("cuTensorMapEncodeTiled") }?;
    // SAFETY: as above.
    let driver_version: unsafe extern "C" fn(*mut i32) -> i32 =
        unsafe { driver.entry("cuDriverGetVersion") }?;
    let mut version = 0i32;
    // SAFETY: the local holds the version the driver writes.
    let status = unsafe { driver_version(&mut version) };
    println!(
        "driver {}.{} (cuDriverGetVersion: {} {})",
        version / 1000,
        version % 1000 / 10,
        status,
        status_name(status)
    );

    // A rank-one tensor map with no stride array at all, and one with a valid
    // stride array the driver is documented not to read. The zeroed stride is
    // what makes the second pointer safe to pass even to a driver that reads
    // it.
    let no_strides: [u64; 0] = [];
    let one_stride = [0u64];
    let element = [1u32];

    let variants = [
        Variant {
            name: "rank 1, no strides (null pointer), box 4",
            data_type: 7,
            dims: &[64],
            strides: Some(&no_strides),
            box_dim: &[4],
            element_strides: &element,
            interleave: 0,
            swizzle: 0,
            l2_promotion: 0,
            oob_fill: 0,
        },
        Variant {
            name: "rank 1, one zeroed stride (non-null pointer), box 4",
            data_type: 7,
            dims: &[64],
            strides: Some(&one_stride),
            box_dim: &[4],
            element_strides: &element,
            interleave: 0,
            swizzle: 0,
            l2_promotion: 0,
            oob_fill: 0,
        },
        Variant {
            name: "rank 1, box 8",
            data_type: 7,
            dims: &[64],
            strides: Some(&no_strides),
            box_dim: &[8],
            element_strides: &element,
            interleave: 0,
            swizzle: 0,
            l2_promotion: 0,
            oob_fill: 0,
        },
        Variant {
            name: "rank 1, global dim 128, box 4",
            data_type: 7,
            dims: &[128],
            strides: Some(&no_strides),
            box_dim: &[4],
            element_strides: &element,
            interleave: 0,
            swizzle: 0,
            l2_promotion: 0,
            oob_fill: 0,
        },
        Variant {
            name: "rank 1, box 4 with a single element box",
            data_type: 7,
            dims: &[64],
            strides: Some(&no_strides),
            box_dim: &[1],
            element_strides: &element,
            interleave: 0,
            swizzle: 0,
            l2_promotion: 0,
            oob_fill: 0,
        },
        Variant {
            name: "rank 2, stride 64, box 4x4, no swizzle",
            data_type: 7,
            dims: &[16, 16],
            strides: Some(&[64]),
            box_dim: &[4, 4],
            element_strides: &[1, 1],
            interleave: 0,
            swizzle: 0,
            l2_promotion: 0,
            oob_fill: 0,
        },
        Variant {
            name: "rank 2, stride 64, box 4x4, swizzle 128B",
            data_type: 7,
            dims: &[16, 16],
            strides: Some(&[64]),
            box_dim: &[4, 4],
            element_strides: &[1, 1],
            interleave: 0,
            swizzle: 3,
            l2_promotion: 2,
            oob_fill: 1,
        },
        Variant {
            name: "rank 3, strides 64 then 1024, box 4x4x4",
            data_type: 7,
            dims: &[16, 16, 16],
            strides: Some(&[64, 1024]),
            box_dim: &[4, 4, 4],
            element_strides: &[1, 1, 1],
            interleave: 0,
            swizzle: 0,
            l2_promotion: 0,
            oob_fill: 0,
        },
    ];

    let mut refused = 0usize;
    for variant in &variants {
        let mut raw = RawTensorMap { opaque: [0; 16] };
        let strides = match variant.strides {
            Some(strides) if !strides.is_empty() => strides.as_ptr(),
            _ => std::ptr::null(),
        };
        // SAFETY: every array holds one entry per dimension of its rank as
        // `cuda.h` requires, the destination is 128 bytes and 64-byte
        // aligned, and the storage the address names outlives the call.
        let status = unsafe {
            encode(
                &mut raw,
                variant.data_type,
                variant.dims.len() as u32,
                address as *mut c_void,
                variant.dims.as_ptr(),
                strides,
                variant.box_dim.as_ptr(),
                variant.element_strides.as_ptr(),
                variant.interleave,
                variant.swizzle,
                variant.l2_promotion,
                variant.oob_fill,
            )
        };
        let encoded = raw.opaque.iter().any(|word| *word != 0);
        // A rank-one description is expected to be refused on this platform:
        // that is what makes the crate take rank two to five, so it is not a
        // failure of the run.
        if status != 0 && variant.dims.len() > 1 {
            refused += 1;
        }
        println!(
            "{:>4} {:<16} {:<58} encoded {}",
            status,
            status_name(status),
            variant.name,
            encoded
        );
    }

    // The wrapper's own call, so the log shows whether the wrapper and the
    // driver agree on what it is given: rank one is refused by the crate (the
    // driver refuses it, see above), a one-dimensional tensor encodes as rank
    // two with a trailing dimension of one, and a two-dimensional tensor
    // encodes.
    println!();
    let spec = TensorMapSpec {
        data_type: TensorDataType::Float32,
        global_address: address,
        global_dim: &[16, 16],
        global_strides: &[64],
        box_dim: &[4, 4],
        element_strides: &[1, 1],
        interleave: Interleave::None,
        swizzle: Swizzle::None,
        l2_promotion: L2Promotion::None,
        oob_fill: OobFill::None,
    };
    report(
        "TensorMap::encode_tiled, rank 2",
        TensorMap::encode_tiled(&device, &spec),
    );
    let one_dimensional = TensorMapSpec {
        global_dim: &[64, 1],
        global_strides: &[256],
        box_dim: &[4, 1],
        element_strides: &[1, 1],
        ..spec
    };
    report(
        "TensorMap::encode_tiled, 1-D tensor as rank 2",
        TensorMap::encode_tiled(&device, &one_dimensional),
    );
    let rank_one = TensorMapSpec {
        global_dim: &[64],
        global_strides: &[],
        box_dim: &[4],
        element_strides: &[1],
        ..spec
    };
    match TensorMap::encode_tiled(&device, &rank_one) {
        Ok(_) => {
            println!("   ok   TensorMap::encode_tiled, rank 1 encoded (the crate would take it)")
        }
        Err(error) => println!(
            "   ok   TensorMap::encode_tiled, rank 1 refused, as the crate states: {error}"
        ),
    }

    // The wrapper and the driver disagreeing is the interesting case.
    if refused > 0 {
        eprintln!("\n{refused} description(s) refused by the driver");
        std::process::exit(1);
    }
    println!("\nevery description encoded");
    Ok(())
}

fn report(name: &str, outcome: Result<TensorMap, Error>) {
    match outcome {
        Ok(map) => println!(
            "   ok   {name} ({} bytes, first word {:#x})",
            map.as_bytes().len(),
            u64::from_ne_bytes(map.as_bytes()[..8].try_into().unwrap_or([0; 8]))
        ),
        Err(error) => println!("   FAIL {name}: {error}"),
    }
}
