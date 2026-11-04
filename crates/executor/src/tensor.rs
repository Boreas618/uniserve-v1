//! Element types, storage kinds and borrowed tensor views.
//!
//! These are the vocabulary the executor's owners exchange: the resident
//! memory owners report what they hold as a [`Backing`], transfers describe
//! where a copy goes with a [`MemoryKind`], and kernels read the element type
//! of their operands from [`DType`]. The types carry no CUDA call; a `Backing`
//! implementor keeps the bytes mapped and unmoved while the value is alive.
//!
//! The shared computation library (`nn`) owns the tensor views models consume;
//! this module holds the same element vocabulary on the executor's side of
//! that boundary so that resident storage and transfers compile without the
//! model crates.

use cuda::DeviceId;

use crate::Result;

/// Where storage lives: host storage the CPU addresses, page-locked host
/// storage that CUDA copies read, or device storage of one GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    Host,
    /// Page-locked host storage, readable by CUDA copies.
    PinnedHost,
    Cuda(DeviceId),
}

impl MemoryKind {
    /// The CUDA device of device storage.
    pub fn cuda(self) -> Option<DeviceId> {
        match self {
            Self::Cuda(device) => Some(device),
            Self::Host | Self::PinnedHost => None,
        }
    }

    /// The locator spelling of transfer peers (`cuda:<ordinal>`, or `cpu` for
    /// host storage of either kind), which consumers read back with
    /// [`Self::from_locator`].
    pub fn locator_name(self) -> String {
        match self {
            Self::Cuda(device) => format!("cuda:{}", device.0),
            Self::Host | Self::PinnedHost => "cpu".to_owned(),
        }
    }

    /// The storage kind a transfer locator names; `cpu` is plain host storage.
    pub fn from_locator(locator: &str) -> Result<Self> {
        if locator == "cpu" {
            return Ok(Self::Host);
        }
        match locator.strip_prefix("cuda:") {
            Some(ordinal) => ordinal
                .parse::<i32>()
                .map(|ordinal| Self::Cuda(DeviceId(ordinal)))
                .map_err(|_| {
                    crate::Error::Invalid(format!("transfer locator {locator:?} names no device"))
                }),
            None => Err(crate::Error::Invalid(format!(
                "transfer locator {locator:?} names no storage kind"
            ))),
        }
    }
}

/// The element type of storage.
///
/// The set is the one checkpoints and kernels use: the floating-point types a
/// `dtype` or `quantization` can name, the integer types a kernel reads for
/// indices and masks, and the two 8-bit float types of FP8 KV pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Bool,
    U8,
    I8,
    I16,
    I32,
    I64,
    U32,
    F16,
    BF16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
}

impl DType {
    /// Storage width of one element in bytes.
    pub fn bytes(self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => 1,
            Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    /// The name a configuration, a checkpoint index or a descriptor key
    /// spells this type with.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            Self::U8 => "uint8",
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::U32 => "uint32",
            Self::F16 => "float16",
            Self::BF16 => "bfloat16",
            Self::F32 => "float32",
            Self::F64 => "float64",
            Self::F8E4M3 => "float8_e4m3fn",
            Self::F8E5M2 => "float8_e5m2",
        }
    }

    /// The type a name spells, or `None` for a name this crate does not serve.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name.trim_start_matches("torch.") {
            "bool" => Self::Bool,
            "uint8" => Self::U8,
            "int8" => Self::I8,
            "int16" => Self::I16,
            "int32" => Self::I32,
            "int64" => Self::I64,
            "uint32" => Self::U32,
            "float16" => Self::F16,
            "bfloat16" => Self::BF16,
            "float32" => Self::F32,
            "float64" => Self::F64,
            "float8_e4m3fn" => Self::F8E4M3,
            "float8_e5m2" => Self::F8E5M2,
            _ => return None,
        })
    }

    /// The type a name spells, failing for any other.
    pub fn parse(name: &str) -> Result<Self> {
        Self::from_name(name)
            .ok_or_else(|| crate::Error::Invalid(format!("unknown element type {name:?}")))
    }
}

/// The storage element type of a request message's dtype.
impl From<ipc::DType> for DType {
    fn from(dtype: ipc::DType) -> Self {
        match dtype {
            ipc::DType::U8 => Self::U8,
            ipc::DType::I16 => Self::I16,
            ipc::DType::I32 => Self::I32,
            ipc::DType::I64 => Self::I64,
            ipc::DType::F16 => Self::F16,
            ipc::DType::BF16 => Self::BF16,
            ipc::DType::F32 => Self::F32,
        }
    }
}

/// The message dtype of a storage element type, for the element types messages
/// carry.
impl TryFrom<DType> for ipc::DType {
    type Error = crate::Error;

    fn try_from(dtype: DType) -> Result<Self> {
        Ok(match dtype {
            DType::U8 => Self::U8,
            DType::I16 => Self::I16,
            DType::I32 => Self::I32,
            DType::I64 => Self::I64,
            DType::F16 => Self::F16,
            DType::BF16 => Self::BF16,
            DType::F32 => Self::F32,
            other => {
                return Err(crate::Error::Unsupported(format!(
                    "{} has no message dtype",
                    other.name()
                )));
            }
        })
    }
}

/// Storage a view can borrow: a device buffer, a host allocation or a segment
/// of either. Implementors keep the bytes valid and unmoved for as long as the
/// value is alive.
pub trait Backing: Send + Sync + 'static {
    fn device(&self) -> MemoryKind;
    /// Base address: a device pointer for CUDA storage, a host pointer
    /// otherwise.
    fn address(&self) -> u64;
    /// Length of the allocation in bytes.
    fn bytes(&self) -> usize;
    /// The most bytes one copy may address in this storage: the length of a
    /// segment of storage mapped in segments that are unmapped apart, where
    /// the driver refuses a copy whose range holds an unmapped segment.
    fn copy_span(&self) -> u64 {
        u64::MAX
    }
}

/// The strides, in elements, of a contiguous tensor of `shape`: the last axis
/// is the innermost.
pub fn contiguous_strides(shape: &[i64]) -> Vec<i64> {
    let mut strides = vec![1; shape.len()];
    for axis in (0..shape.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    strides
}

/// A borrowed view of storage: an address, an element type, a shape and the
/// strides between elements. Owning neither the storage nor the shape, it is
/// valid only while its [`Backing`] and its shape and stride arrays live.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    /// Device or host address of the first element.
    pub address: u64,
    pub dtype: DType,
    pub shape: &'a [u64],
    pub strides: &'a [u64],
}

impl TensorView<'_> {
    /// The bytes one element of `shape` with these strides spans, which a copy
    /// of the whole view addresses.
    pub fn bytes(&self) -> Result<u64> {
        if self.shape.len() != self.strides.len() {
            return Err(crate::Error::Invariant(
                "a view needs one stride per dimension".into(),
            ));
        }
        let mut elements: u64 = 1;
        for (&size, &stride) in self.shape.iter().zip(self.strides) {
            if size == 0 {
                return Ok(0);
            }
            elements = elements
                .checked_add((size - 1).saturating_mul(stride))
                .ok_or_else(|| crate::Error::Invalid("a view's extent overflows".into()))?;
        }
        Ok(elements.saturating_mul(self.dtype.bytes() as u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_types_carry_their_width_and_spelling() {
        for (name, dtype, bytes) in [
            ("bfloat16", DType::BF16, 2),
            ("float8_e4m3fn", DType::F8E4M3, 1),
            ("torch.int64", DType::I64, 8),
            ("float32", DType::F32, 4),
        ] {
            assert_eq!(
                DType::parse(name).expect("a known element type"),
                dtype,
                "{name}"
            );
            assert_eq!(dtype.bytes(), bytes, "{name}");
            assert_eq!(DType::from_name(dtype.name()), Some(dtype));
        }
        assert!(DType::from_name("float4").is_none());
    }

    #[test]
    fn a_locator_names_the_storage_kind_it_moves() {
        assert_eq!(
            MemoryKind::from_locator("cpu").expect("host storage"),
            MemoryKind::Host
        );
        assert_eq!(
            MemoryKind::from_locator("cuda:3").expect("device storage"),
            MemoryKind::Cuda(DeviceId(3))
        );
        assert_eq!(MemoryKind::Cuda(DeviceId(3)).locator_name(), "cuda:3");
        assert!(MemoryKind::from_locator("cuda").is_err());
        assert!(MemoryKind::from_locator("tpu:0").is_err());
    }

    #[test]
    fn contiguous_strides_run_from_the_inner_axis_out() {
        assert_eq!(contiguous_strides(&[4, 3, 2]), vec![6, 2, 1]);
        assert_eq!(contiguous_strides(&[7]), vec![1]);
        assert_eq!(contiguous_strides(&[]), Vec::<i64>::new());
    }

    #[test]
    fn a_view_spans_the_elements_its_strides_address() {
        let shape = [2_u64, 3];
        let strides = [3_u64, 1];
        let view = TensorView {
            address: 0,
            dtype: DType::BF16,
            shape: &shape,
            strides: &strides,
        };
        assert_eq!(view.bytes().expect("a byte span"), 12);
    }

    #[test]
    fn a_strided_view_spans_its_gaps() {
        // Two rows of three columns, every second column: [0] and [2] of each.
        let shape = [2_u64, 2];
        let strides = [6_u64, 2];
        let view = TensorView {
            address: 0,
            dtype: DType::F32,
            shape: &shape,
            strides: &strides,
        };
        assert_eq!(view.bytes().expect("a byte span"), 4 * (6 + 2));
    }
}
