//! `InputBuffers`: the input and output tensors of one runner's forwards.
//!
//! One allocation holds every column of a runner on its lane. A runner
//! method receives the columns it reads and writes as arguments, and a
//! captured graph addresses the columns its capture was given, so the
//! storage is fixed for the runner's life. Each runner states its columns
//! through [`Columns`], from the dimensions that size them.

use std::sync::Arc;

use uniserve_cuda::Stream;

use crate::ffi::{Backing, DType};
use crate::storage::memory::{Memory, StorageView};
use crate::{Error, Result};

/// Byte alignment of each carved column.
const ALIGNMENT: usize = 256;

/// The columns a runner's dimensions size.
pub(crate) trait Columns {
    /// A column's name.
    type Name: AsRef<str>;

    /// Every column: its name, element type and shape.
    fn columns(&self) -> Vec<(Self::Name, DType, Vec<i64>)>;
}

fn column_bytes(dtype: DType, shape: &[i64]) -> usize {
    (shape.iter().product::<i64>().max(1) as usize * dtype.bytes()).next_multiple_of(ALIGNMENT)
}

/// A runner's columns in one allocation.
pub(crate) struct InputBuffers<D: Columns> {
    pub(crate) dimensions: D,
    device: Option<i32>,
    columns: Vec<(D::Name, StorageView)>,
}

impl<D: Columns> InputBuffers<D> {
    /// Device bytes of the columns of `dimensions`.
    pub(crate) fn bytes(dimensions: &D) -> u64 {
        dimensions
            .columns()
            .iter()
            .map(|(_, dtype, shape)| column_bytes(*dtype, shape))
            .sum::<usize>() as u64
    }

    /// Allocate the columns on `device` (host memory when `None`) and zero
    /// them on `stream`, which drains before this returns.
    pub(crate) fn allocate(
        dimensions: D,
        device: Option<i32>,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        let layout = dimensions.columns();
        let bytes = Self::bytes(&dimensions) as usize;
        let memory = match device {
            Some(device) => Memory::device(device, bytes)?,
            None => Memory::host(bytes),
        };
        let mut columns = Vec::with_capacity(layout.len());
        let mut offset = 0;
        for (name, dtype, shape) in layout {
            let view = StorageView::contiguous(
                Arc::clone(&memory) as Arc<dyn Backing>,
                offset,
                dtype,
                &shape,
            )?;
            columns.push((name, view));
            offset += column_bytes(dtype, &shape);
        }
        match (device, stream) {
            (Some(_), Some(stream)) => {
                // SAFETY: the new allocation spans `bytes`, a multiple of
                // the column alignment. The stream drains below.
                unsafe { uniserve_cuda::memset_d32(memory.address(), 0, bytes / 4, stream)? };
                stream.synchronize()?;
            }
            // Host memory starts zeroed.
            (None, None) => {}
            _ => {
                return Err(Error::Invalid(
                    "input buffers are reset on a stream exactly for device storage".into(),
                ));
            }
        }
        Ok(Self {
            dimensions,
            device,
            columns,
        })
    }

    /// The CUDA device of the columns; `None` for host memory.
    pub(crate) fn device(&self) -> Option<i32> {
        self.device
    }

    pub(crate) fn column(&self, name: &str) -> Result<&StorageView> {
        self.columns
            .iter()
            .find(|(column, _)| column.as_ref() == name)
            .map(|(_, view)| view)
            .ok_or_else(|| Error::Invariant(format!("input buffers have no column {name}")))
    }
}
