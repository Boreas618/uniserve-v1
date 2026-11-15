//! The package ABI: what a kernel package and this crate agree on outside a call's values.
//!
//! The element type itself is the tensor library's [`DType`]. What a kernel package knows of it is
//! fixed outside this crate: a native package reads the number of each element type a call names
//! ([`ElementType`]), a module is stored under the key its descriptor prints ([`name`]), and a
//! build module receives the key's fields. The functions here are the one statement of those
//! numbers and spellings, so the tensor library may name or order its element types as it likes.
//! The numbers never move: a stored module was compiled against them, and an element type added
//! to the ABI takes the next number.
//!
//! ## The C headers
//!
//! A native package's entry receives one `#[repr(C)]` call structure, defined once in its
//! operator's module and deriving [`Layout`]. The C declarations a package compiles against are
//! generated from those definitions by cbindgen: one header per operator interface,
//! `uniserve_<operator>.h`, and `uniserve_abi.h` for what they share, the element-type numbering
//! ([`ElementType`], rendered as `uniserve_dtype`) and the error buffer ([`MESSAGE`], rendered as
//! `UNISERVE_MESSAGE`). Each header ends with the size and every field offset of each of its
//! structures as Rust lays it out, asserted, so a declaration the C compiler lays out otherwise
//! fails the package's build. The headers are files of the kernel-source library
//! `uniserve_kernels` (`crates/ops/python/src/uniserve_kernels/include/`), which the Build
//! Server's environment installs, digests into every module's identity and passes as the include
//! directory of every native compilation. `cargo test -p ops` regenerates them and compares them
//! with the committed files (`abi::headers`); `UPDATE_EXPECT=1` writes the new output.
//!
//! An enumeration a package reads, in a call field or in a definition its build compiles in, is
//! rendered with its enumerators named `<ENUMERATION>_<VARIANT>` in capitals: the element type a
//! key spells `bf16` is `UNISERVE_DTYPE_BF16`, the weight order `nxk` is
//! `UNISERVE_WEIGHT_ORDER_NXK`. A variant's name is its spelling in a kernel key, so a build module
//! spells the enumerator a descriptor field names by that one rule
//! (`uniserve_build_server.request.Request.enumerator`).

use std::ffi::{CStr, c_char};

use tensor::DType;

use crate::{Error, Result};

#[cfg(test)]
mod headers;

/// The derive of [`Layout`] for a `#[repr(C)]` call structure.
pub use ops_derive::Layout;

/// A call structure's layout as Rust gives it: what its generated C declaration asserts.
///
/// Derived (`#[derive(abi::Layout)]`) on every `#[repr(C)]` structure a native package receives,
/// from the structure's own definition.
pub trait Layout: Sized {
    /// The structure's name in this crate.
    const NAME: &'static str;
    /// Each field's name and byte offset, in declaration order.
    const FIELDS: &'static [(&'static str, usize)];
    /// The structure's size in bytes.
    const SIZE: usize = std::mem::size_of::<Self>();
}

/// The bytes of a call structure's `error` field: the NUL-terminated message an entry writes
/// when it returns nonzero, the NUL included.
pub const MESSAGE: usize = 512;

/// The message an entry wrote into a call's `error` field, or `None` when the field holds no
/// NUL-terminated UTF-8 text.
pub fn message(error: &[c_char; MESSAGE]) -> Option<&str> {
    // SAFETY: `c_char` is `i8` or `u8`, so the array is `MESSAGE` bytes of one alignment, and a
    // byte array of the same length reads exactly that memory.
    let bytes = unsafe { &*std::ptr::from_ref(error).cast::<[u8; MESSAGE]>() };
    CStr::from_bytes_until_nul(bytes).ok()?.to_str().ok()
}

/// An element type as a native package reads it: the number a call structure's element-type
/// fields carry, and the enumerator a build module's definitions name.
///
/// A variant's name is the element type's spelling in a kernel key ([`name`]), in capitals.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ElementType {
    U8 = 0,
    I32 = 1,
    I64 = 2,
    F16 = 3,
    BF16 = 4,
    F32 = 5,
    F8E4M3 = 6,
    /// NVFP4's element: two share a byte.
    F4E2M1 = 7,
    I16 = 8,
    U32 = 9,
    F64 = 10,
}

impl TryFrom<DType> for ElementType {
    type Error = Error;

    /// The number of `dtype`, or [`Error::Shape`] for an element type the ABI has no number for:
    /// no package reads one, so no call may claim to name it.
    fn try_from(dtype: DType) -> Result<Self> {
        Ok(match dtype {
            DType::U8 => Self::U8,
            DType::I32 => Self::I32,
            DType::I64 => Self::I64,
            DType::F16 => Self::F16,
            DType::BF16 => Self::BF16,
            DType::F32 => Self::F32,
            DType::F8E4M3 => Self::F8E4M3,
            DType::F4 => Self::F4E2M1,
            DType::I16 => Self::I16,
            DType::U32 => Self::U32,
            DType::F64 => Self::F64,
            other => {
                return Err(Error::Shape(format!(
                    "{} elements have no number in the package ABI",
                    name(other)
                )));
            }
        })
    }
}

/// The number a call structure's element-type field carries for `dtype` ([`ElementType`]).
///
/// # Errors
///
/// [`Error::Shape`] for an element type the ABI has no number for.
pub fn code(dtype: DType) -> Result<i64> {
    Ok(ElementType::try_from(dtype)? as i64)
}

/// The spelling of `dtype` inside a kernel key, and in a refusal that names an element type.
///
/// It is the tensor library's own spelling except for the NVFP4 element, which the stored
/// modules and their manifests spell with its exponent and mantissa widths.
pub fn name(dtype: DType) -> &'static str {
    match dtype {
        DType::F4 => "f4e2m1",
        other => other.as_str(),
    }
}

/// The number of bytes one element of `dtype` occupies, when it is a whole number of bytes.
///
/// A sub-byte element type is addressed by its encoded storage layout, which the operators that
/// read it (`gemm` on an NVFP4 weight, for one) name; everything that hands out bytes fails for
/// it rather than rounding.
pub fn bytes(dtype: DType) -> Option<u64> {
    dtype.size_in_bytes().map(|bytes| bytes as u64)
}

/// The element type an interface names (a configuration's `dtype`, `kv_cache_config.dtype`), as
/// the tensor library's.
pub const fn from_wire(dtype: proto::tensor::DType) -> DType {
    use proto::tensor::DType as Wire;

    match dtype {
        Wire::F16 => DType::F16,
        Wire::BF16 => DType::BF16,
        Wire::F32 => DType::F32,
        Wire::F8E4M3 => DType::F8E4M3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every element type the ABI states, with its number (which stored modules were compiled
    /// against), its key spelling and its whole-byte size.
    const TABLE: [(DType, i64, &str, Option<u64>); 11] = [
        (DType::U8, 0, "u8", Some(1)),
        (DType::I32, 1, "i32", Some(4)),
        (DType::I64, 2, "i64", Some(8)),
        (DType::F16, 3, "f16", Some(2)),
        (DType::BF16, 4, "bf16", Some(2)),
        (DType::F32, 5, "f32", Some(4)),
        (DType::F8E4M3, 6, "f8e4m3", Some(1)),
        (DType::F4, 7, "f4e2m1", None),
        (DType::I16, 8, "i16", Some(2)),
        (DType::U32, 9, "u32", Some(4)),
        (DType::F64, 10, "f64", Some(8)),
    ];

    #[test]
    fn numbers_spellings_and_sizes_are_the_stored_modules() {
        for (dtype, number, spelling, size) in TABLE {
            assert_eq!(code(dtype).unwrap(), number, "{spelling}");
            assert_eq!(name(dtype), spelling);
            assert_eq!(bytes(dtype), size, "{spelling}");
        }
    }

    /// A build module spells the enumerator of a key's element type from the key: the variant's
    /// name is the key's spelling, in capitals.
    #[test]
    fn a_key_spells_its_element_types_enumerator() {
        for (dtype, ..) in TABLE {
            let element = ElementType::try_from(dtype).unwrap();
            assert_eq!(format!("{element:?}"), name(dtype).to_uppercase());
        }
    }

    #[test]
    fn a_message_is_the_text_before_the_first_nul() {
        let mut error = [0 as c_char; MESSAGE];
        assert_eq!(message(&error), Some(""));
        for (slot, byte) in error.iter_mut().zip(b"no engine\0left over") {
            *slot = *byte as c_char;
        }
        assert_eq!(message(&error), Some("no engine"));
        // A buffer the entry filled to the end without a terminator holds no message.
        let full = [b'x' as c_char; MESSAGE];
        assert_eq!(message(&full), None);
    }

    #[test]
    fn an_interface_element_type_is_the_same_element_type() {
        use proto::tensor::DType as Wire;

        for (wire, dtype) in [
            (Wire::F16, DType::F16),
            (Wire::BF16, DType::BF16),
            (Wire::F32, DType::F32),
            (Wire::F8E4M3, DType::F8E4M3),
        ] {
            assert_eq!(from_wire(wire), dtype);
            assert_eq!(wire.element_bytes(), bytes(dtype).unwrap());
        }
    }
}
