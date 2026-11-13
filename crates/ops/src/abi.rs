//! Element types as the package ABI states them: a number in a launch word, a spelling in a
//! kernel key, and a whole number of bytes in memory.
//!
//! The element type itself is the tensor library's [`DType`]. What a kernel package knows of it
//! is fixed outside this crate: a package's C header carries the number of each element type it
//! reads, a module is stored under the key its descriptor prints, and a build script receives the
//! key's fields. The functions here are the one statement of those numbers and spellings on the
//! Rust side, so the tensor library may name or order its element types as it likes.
//!
//! | Element type | Number | Spelling | Bytes |
//! | --- | --- | --- | --- |
//! | `U8` | 0 | `u8` | 1 |
//! | `I32` | 1 | `i32` | 4 |
//! | `I64` | 2 | `i64` | 8 |
//! | `F16` | 3 | `f16` | 2 |
//! | `BF16` | 4 | `bf16` | 2 |
//! | `F32` | 5 | `f32` | 4 |
//! | `F8E4M3` | 6 | `f8e4m3` | 1 |
//! | `F4` | 7 | `f4e2m1` | two elements share one |
//! | `I16` | 8 | `i16` | 2 |
//! | `U32` | 9 | `u32` | 4 |
//! | `F64` | 10 | `f64` | 8 |
//!
//! The numbers never move: a stored module was compiled against them. An element type added to
//! the ABI takes the next number.

use tensor::DType;

use crate::{Error, Result};

/// The number a package header carries for `dtype`.
///
/// # Errors
///
/// [`Error::Shape`] for an element type the ABI has no number for: no package reads one, so no
/// launch word may claim to name it.
pub fn code(dtype: DType) -> Result<i64> {
    Ok(match dtype {
        DType::U8 => 0,
        DType::I32 => 1,
        DType::I64 => 2,
        DType::F16 => 3,
        DType::BF16 => 4,
        DType::F32 => 5,
        DType::F8E4M3 => 6,
        DType::F4 => 7,
        DType::I16 => 8,
        DType::U32 => 9,
        DType::F64 => 10,
        other => {
            return Err(Error::Shape(format!(
                "{} elements have no number in the package ABI",
                name(other)
            )));
        }
    })
}

/// The element type a package header numbers `code`, or `None` for a number the ABI does not
/// state.
pub const fn from_code(code: i64) -> Option<DType> {
    Some(match code {
        0 => DType::U8,
        1 => DType::I32,
        2 => DType::I64,
        3 => DType::F16,
        4 => DType::BF16,
        5 => DType::F32,
        6 => DType::F8E4M3,
        7 => DType::F4,
        8 => DType::I16,
        9 => DType::U32,
        10 => DType::F64,
        _ => return None,
    })
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
    match dtype.size_in_bytes() {
        0 => None,
        bytes => Some(bytes as u64),
    }
}

/// The element type of a tensor that crossed a process boundary.
pub const fn from_wire(dtype: proto::tensor::DType) -> DType {
    use proto::tensor::DType as Wire;

    match dtype {
        Wire::U8 => DType::U8,
        Wire::I16 => DType::I16,
        Wire::I32 => DType::I32,
        Wire::I64 => DType::I64,
        Wire::F16 => DType::F16,
        Wire::BF16 => DType::BF16,
        Wire::F32 => DType::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The documented table: number, spelling and whole-byte size of every element type the ABI
    /// states.
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
    fn numbers_spellings_and_sizes_are_the_documented_table() {
        for (dtype, number, spelling, size) in TABLE {
            assert_eq!(code(dtype).unwrap(), number, "{spelling}");
            assert_eq!(from_code(number), Some(dtype), "{spelling}");
            assert_eq!(name(dtype), spelling);
            assert_eq!(bytes(dtype), size, "{spelling}");
        }

        assert_eq!(from_code(11), None);
        assert_eq!(from_code(-1), None);
    }

    #[test]
    fn an_element_type_the_abi_does_not_state_has_a_spelling_and_no_number() {
        for (dtype, spelling) in [
            (DType::F6E2M3, "f6e2m3"),
            (DType::F6E3M2, "f6e3m2"),
            (DType::F8E8M0, "f8e8m0"),
        ] {
            assert_eq!(name(dtype), spelling);

            let refused = code(dtype).unwrap_err().to_string();
            assert!(refused.contains(spelling), "{refused}");
        }
    }

    #[test]
    fn a_wire_element_type_is_the_same_element_type() {
        use proto::tensor::DType as Wire;

        assert_eq!(from_wire(Wire::BF16), DType::BF16);
        assert_eq!(from_wire(Wire::I16), DType::I16);
        assert_eq!(from_wire(Wire::U8), DType::U8);
    }
}
