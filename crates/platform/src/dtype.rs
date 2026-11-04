//! The numeric element types the bound libraries speak: the type of a
//! matrix element, a tensor element or a collective's buffer, with the code
//! each library gives it.
//!
//! `nn` interprets these in its own `DType` (design §4.3); this crate keeps
//! only what the driver, cuBLASLt, cuDNN and NCCL must be told about a
//! buffer's elements.

/// The element type of a device buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataType {
    F64,
    F32,
    Bf16,
    F16,
    I64,
    I32,
    I16,
    I8,
    U8,
}

impl DataType {
    /// The element's size in bytes.
    pub fn bytes(self) -> usize {
        match self {
            Self::F64 | Self::I64 => 8,
            Self::F32 | Self::I32 => 4,
            Self::Bf16 | Self::F16 | Self::I16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }

    /// `cudaDataType_t`, as the driver, cuBLASLt and cuDNN report it.
    pub fn cuda(self) -> i32 {
        match self {
            Self::F32 => 0,   // CUDA_R_32F
            Self::F64 => 1,   // CUDA_R_64F
            Self::F16 => 2,   // CUDA_R_16F
            Self::I8 => 3,    // CUDA_R_8I
            Self::I32 => 4,   // CUDA_R_32I
            Self::U8 => 5,    // CUDA_R_8U
            Self::I16 => 7,   // CUDA_R_16I
            Self::I64 => 9,   // CUDA_R_64I
            Self::Bf16 => 14, // CUDA_R_16BF
        }
    }

    /// `cudnnDataType_t`, whose codes differ from `cudaDataType_t`'s.
    pub fn cudnn(self) -> i32 {
        match self {
            Self::F32 => 0,              // CUDNN_DATA_FLOAT
            Self::F64 => 1,              // CUDNN_DATA_DOUBLE
            Self::F16 => 2,              // CUDNN_DATA_HALF
            Self::I8 => 3,               // CUDNN_DATA_INT8
            Self::I32 => 4,              // CUDNN_DATA_INT32
            Self::U8 => 6,               // CUDNN_DATA_UINT8
            Self::Bf16 => 9,             // CUDNN_DATA_BFLOAT16
            Self::I64 | Self::I16 => -1, // no cuDNN counterpart
        }
    }

    /// `ncclDataType_t`.
    pub fn nccl(self) -> i32 {
        match self {
            Self::I8 => 0,   // ncclInt8
            Self::U8 => 1,   // ncclUint8
            Self::I32 => 2,  // ncclInt32
            Self::I64 => 4,  // ncclInt64
            Self::F16 => 6,  // ncclFloat16
            Self::Bf16 => 9, // ncclBfloat16
            Self::F32 => 7,  // ncclFloat32
            Self::F64 => 8,  // ncclFloat64
            Self::I16 => 10, // ncclInt16
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_library_gets_its_own_code() {
        assert_eq!(DataType::F32.cuda(), 0);
        assert_eq!(DataType::F32.cudnn(), 0);
        assert_eq!(DataType::F32.nccl(), 7);

        assert_eq!(DataType::Bf16.cuda(), 14);
        assert_eq!(DataType::Bf16.cudnn(), 9);
        assert_eq!(DataType::Bf16.nccl(), 9);

        assert_eq!(DataType::F16.cuda(), 2);
        assert_eq!(DataType::F16.cudnn(), 2);
        assert_eq!(DataType::F16.nccl(), 6);
    }

    #[test]
    fn the_element_sizes_are_the_c_sizes() {
        assert_eq!(DataType::F64.bytes(), 8);
        assert_eq!(DataType::F32.bytes(), 4);
        assert_eq!(DataType::Bf16.bytes(), 2);
        assert_eq!(DataType::U8.bytes(), 1);
    }
}
