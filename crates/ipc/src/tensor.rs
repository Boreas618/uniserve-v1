//! Tensor identities and bounded representations.
//!
//! The engine's scheduler declares a cross-call tensor product as a
//! [`TensorRef`]: a request-scoped identity plus a `proto::tensor::DType` and a
//! [`ShapeBound`] that fix the product's maximum size before its producing call
//! runs. A cross-call buffer, such as a published cache extent, is identified
//! by a [`BufferId`] alone.

use super::*;

/// One dimension of a bounded shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DimBound {
    /// A host-static extent.
    Static(u32),
    /// The single device-actual axis, bounded by this fixed maximum.
    Device {
        /// Maximum device-selected extent.
        max: u32,
    },
}

/// A shape that is host-static except for at most one device-actual axis, which
/// carries a fixed maximum. A single device-actual axis without static axes
/// denotes flat capacity; its actual tensor rank is published with the transfer.
/// An empty bound (the default) admits exactly one element.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct ShapeBound {
    /// Ordered tensor dimension bounds.
    pub dims: Vec<DimBound>,
}

impl ShapeBound {
    /// Whether a positive physical shape fits these logical dimensions.
    /// A single device dimension bounds flat capacity, irrespective of rank.
    pub fn contains_shape(&self, shape: &[u64]) -> bool {
        let Ok(elements) = tensor_elements(shape) else {
            return false;
        };
        match self.dims.as_slice() {
            [] => elements == 1,
            [DimBound::Device { max }] => elements <= u64::from(*max),
            bounds => {
                shape.len() == bounds.len()
                    && shape.iter().zip(bounds).all(|(&size, bound)| match bound {
                        DimBound::Static(expected) => size == u64::from(*expected),
                        DimBound::Device { max } => size <= u64::from(*max),
                    })
            }
        }
    }

    /// Rejects more than one device-actual dimension and any zero extent.
    pub fn validate(&self) -> ValidationResult<()> {
        let device_dims = self
            .dims
            .iter()
            .filter(|dim| matches!(dim, DimBound::Device { .. }))
            .count();
        ensure_valid!(
            device_dims <= 1,
            "a shape bound carries more than one device-actual dimension"
        );
        ensure_valid!(
            self.dims.iter().all(|dim| match dim {
                DimBound::Static(value) => *value > 0,
                DimBound::Device { max } => *max > 0,
            }),
            "a shape bound contains a zero extent"
        );
        Ok(())
    }

    /// Returns the maximum number of elements represented by this shape,
    /// saturating at `u64::MAX`; an empty bound yields one.
    pub fn max_elements(&self) -> u64 {
        self.dims.iter().fold(1_u64, |elements, dim| {
            elements.saturating_mul(u64::from(match dim {
                DimBound::Static(value) => *value,
                DimBound::Device { max } => *max,
            }))
        })
    }
}

/// Identity, element type, and bounded capacity of a tensor that survives its
/// producing call. Actual geometry and locations are published separately.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorRef {
    /// Request that owns the value.
    pub request_key: RequestKey,
    /// Call that declares the value.
    pub producer_call_id: CallId,
    /// Position in the producer's output list.
    pub output_index: u16,
    /// Nonzero allocation generation preventing identity reuse.
    pub generation: u32,
    /// Element data type.
    pub dtype: DType,
    /// Maximum logical tensor shape.
    pub shape_bound: ShapeBound,
}

/// Stable identity for one cross-call buffer, independent of its physical representation.
///
/// A [`TensorRef`] projects to one through [`TensorRef::buffer_id`]; a cache
/// export carries one directly. The engine routes buffers and frees them
/// (`BatchCommand::Free`) by this identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferId {
    /// Request that owns the buffer.
    pub owner: RequestKey,
    /// Call that first declares the buffer.
    pub producer_call_id: CallId,
    /// Position in the producer's output list.
    pub output_index: u16,
    /// Nonzero allocation generation preventing identity reuse.
    pub generation: u32,
}

impl BufferId {
    /// Validates a nonzero persistent-buffer identity.
    pub fn validate(self) -> ValidationResult<()> {
        ensure_valid!(self.generation > 0, "buffer id has no logical generation");
        Ok(())
    }
}

impl TensorRef {
    /// Returns the exact storage identity used for routing and retirement.
    pub const fn buffer_id(&self) -> BufferId {
        BufferId {
            owner: self.request_key,
            producer_call_id: self.producer_call_id,
            output_index: self.output_index,
            generation: self.generation,
        }
    }

    /// Validates allocation identity and tensor capacity.
    pub fn validate(&self) -> ValidationResult<()> {
        ensure_valid!(
            self.generation > 0,
            "product reference has no logical generation"
        );
        self.shape_bound.validate()
    }

    /// Returns the maximum storage size in bytes: the bound's maximum element
    /// count times the element width, saturating: what a buffer output's
    /// allocation must hold at least.
    pub fn max_bytes(&self) -> u64 {
        self.shape_bound
            .max_elements()
            .saturating_mul(self.dtype.element_bytes())
    }
}

/// Checked product of the extents; an empty shape yields one.
pub(crate) fn tensor_elements(shape: &[u64]) -> ValidationResult<u64> {
    shape.iter().try_fold(1u64, |count, &extent| {
        count
            .checked_mul(extent)
            .ok_or_else(|| invalid_message!("tensor element count overflows"))
    })
}
