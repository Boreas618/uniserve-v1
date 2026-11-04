//! The element types device buffers hold: the one dtype type this crate and
//! the layers above it use, with the code each bound library gives it.
//!
//! Design 4.2 spells the type `DType` and makes it the element type of a
//! tensor view; the driver, cuBLASLt, cuDNN and NCCL each spell their own
//! code for the same thing, which is what this type answers with. There is no
//! second dtype type in the crate: a library that needs a code asks this one.

/// The element type of a device buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    F32,
    F16,
    Bf16,
    F8E4M3,
    F8E5M2,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U8,
}

impl DType {
    /// The element's size in bytes.
    pub fn bytes(self) -> usize {
        match self {
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::Bf16 | Self::I16 => 2,
            Self::F8E4M3 | Self::F8E5M2 | Self::I8 | Self::U8 => 1,
        }
    }

    /// `cudaDataType_t`, as the driver, cuBLASLt and cuDNN report it
    /// (`library_types.h`).
    pub fn cuda(self) -> i32 {
        match self {
            Self::F32 => 0,     // CUDA_R_32F
            Self::F64 => 1,     // CUDA_R_64F
            Self::F16 => 2,     // CUDA_R_16F
            Self::I8 => 3,      // CUDA_R_8I
            Self::U8 => 8,      // CUDA_R_8U
            Self::I32 => 10,    // CUDA_R_32I
            Self::U32 => 12,    // CUDA_R_32U
            Self::Bf16 => 14,   // CUDA_R_16BF
            Self::I16 => 20,    // CUDA_R_16I
            Self::I64 => 24,    // CUDA_R_64I
            Self::U64 => 26,    // CUDA_R_64U
            Self::F8E4M3 => 28, // CUDA_R_8F_E4M3
            Self::F8E5M2 => 29, // CUDA_R_8F_E5M2
        }
    }

    /// `cudnnDataType_t` (`cudnn_graph_v9.h`), whose codes differ from
    /// `cudaDataType_t`'s.
    pub fn cudnn(self) -> i32 {
        match self {
            Self::F32 => 0,                                      // CUDNN_DATA_FLOAT
            Self::F64 => 1,                                      // CUDNN_DATA_DOUBLE
            Self::F16 => 2,                                      // CUDNN_DATA_HALF
            Self::I8 => 3,                                       // CUDNN_DATA_INT8
            Self::I32 => 4,                                      // CUDNN_DATA_INT32
            Self::U8 => 6,                                       // CUDNN_DATA_UINT8
            Self::Bf16 => 9,                                     // CUDNN_DATA_BFLOAT16
            Self::F8E4M3 => 11,                                  // CUDNN_DATA_FP8_E4M3
            Self::F8E5M2 => 12,                                  // CUDNN_DATA_FP8_E5M2
            Self::I16 | Self::I64 | Self::U32 | Self::U64 => -1, // no counterpart
        }
    }

    /// `ncclDataType_t` (`nccl.h`), for a collective's buffers.
    pub fn nccl(self) -> i32 {
        match self {
            Self::I8 => 0,      // ncclInt8
            Self::U8 => 1,      // ncclUint8
            Self::I32 => 2,     // ncclInt32
            Self::U32 => 3,     // ncclUint32
            Self::I64 => 4,     // ncclInt64
            Self::U64 => 5,     // ncclUint64
            Self::F16 => 6,     // ncclFloat16
            Self::F32 => 7,     // ncclFloat32
            Self::F64 => 8,     // ncclFloat64
            Self::Bf16 => 9,    // ncclBfloat16
            Self::F8E4M3 => 10, // ncclFloat8e4m3
            Self::F8E5M2 => 11, // ncclFloat8e5m2
            Self::I16 => -1,    // no counterpart
        }
    }

    /// The unit round-off `u`, half the gap between 1 and the next
    /// representable number of this type: the `u` of design §7.3's error
    /// bound `γₙ = n·u/(1 − n·u)`, which an operator's tolerance is derived
    /// from.
    ///
    /// It is 0 for integer types, which have no rounding.
    pub fn unit_round_off(self) -> f64 {
        match self {
            // 53, 24, 11, 8 and 3 explicit bits of significand, plus the
            // implicit one.
            Self::F64 => libm_half_ulp(53),
            Self::F32 => libm_half_ulp(24),
            Self::F16 => libm_half_ulp(11),
            Self::Bf16 => libm_half_ulp(8),
            Self::F8E4M3 => libm_half_ulp(4),
            Self::F8E5M2 => libm_half_ulp(3),
            Self::I64 | Self::I32 | Self::I16 | Self::I8 | Self::U64 | Self::U32 | Self::U8 => 0.0,
        }
    }
}

/// `2^-bits`: half a unit in the last place for a significand of `bits` bits,
/// where the leading bit is implicit.
fn libm_half_ulp(bits: u32) -> f64 {
    2f64.powi(-(bits as i32))
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn a_type_reports_its_size_and_binary_identity() {
        assert_eq!(DType::F64.bytes(), 8);
        assert_eq!(DType::F32.bytes(), 4);
        assert_eq!(DType::Bf16.bytes(), 2);
        assert_eq!(DType::I8.bytes(), 1);
        assert_eq!(DType::U64.bytes(), 8);
    }

    /// The library codes are the binding's contract with the driver, cuBLASLt,
    /// cuDNN and NCCL; a wrong number here is a wrong computation, not a
    /// compile error.
    #[test]
    fn a_type_reports_each_librarys_code() {
        assert_eq!(DType::F32.cuda(), 0);
        assert_eq!(DType::F64.cuda(), 1);
        assert_eq!(DType::Bf16.cuda(), 14);
        assert_eq!(DType::I32.cuda(), 10);
        assert_eq!(DType::U8.cuda(), 8);
        assert_eq!(DType::F8E4M3.cuda(), 28);

        assert_eq!(DType::F32.cudnn(), 0);
        assert_eq!(DType::Bf16.cudnn(), 9);
        assert_eq!(DType::U8.cudnn(), 6);
        assert_eq!(DType::I64.cudnn(), -1);

        assert_eq!(DType::F32.nccl(), 7);
        assert_eq!(DType::Bf16.nccl(), 9);
        assert_eq!(DType::I8.nccl(), 0);
        assert_eq!(DType::I16.nccl(), -1);
    }

    /// The bound of design §7.3 is `γₙ = n·u/(1 − n·u)` with this `u`; its
    /// smallest value for FP32, `u = 2^-24`, is the one the standard bound
    /// uses.
    #[test]
    fn a_type_reports_its_unit_round_off() {
        assert_eq!(DType::F32.unit_round_off(), 2f64.powi(-24));
        assert_eq!(DType::F64.unit_round_off(), 2f64.powi(-53));
        assert_eq!(DType::Bf16.unit_round_off(), 2f64.powi(-8));
        assert_eq!(DType::F16.unit_round_off(), 2f64.powi(-11));
        assert_eq!(DType::I32.unit_round_off(), 0.0);

        // γₙ stays below one for a reduction of a hundred million terms in
        // FP32, which is why the bound is usable at the lengths a kernel sums.
        let n = 100_000_000.0;
        let u = DType::F32.unit_round_off();
        assert!(n * u / (1.0 - n * u) < 6.0);
    }
}
