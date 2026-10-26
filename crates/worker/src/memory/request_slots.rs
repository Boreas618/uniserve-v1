//! Per-request-slot state banks.

use cuda::DeviceId;
use std::sync::Arc;

use super::allocation::{Storage, StorageView};
use crate::ffi::{Backing, DType};
use crate::{Error, Result};

/// One field of per-slot state a model declares: `shape` is one slot's
/// extent; host fields live in page-locked host memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SlotField {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<i64>,
    pub host: bool,
}

impl SlotField {
    fn slot_bytes(&self) -> Result<usize> {
        self.shape
            .iter()
            .try_fold(self.dtype.bytes(), |bytes, &extent| {
                usize::try_from(extent)
                    .ok()
                    .and_then(|extent| bytes.checked_mul(extent))
            })
            .ok_or_else(|| Error::Invalid(format!("slot field {} has an invalid shape", self.name)))
    }
}

/// Banks of per-slot state: one allocation per field shaped
/// `[capacity + 1, ...shape]`, where row 0 is the padding slot and request
/// slots are 1-based. Banks are bound once. Contents are uninitialized until
/// their owner writes them.
pub(crate) struct RequestSlots {
    banks: Vec<Bank>,
}

/// One field's bank.
struct Bank {
    field: SlotField,
    view: StorageView,
}

impl RequestSlots {
    /// Device and host bytes of the banks: `(device, host)`.
    pub(crate) fn bytes(fields: &[SlotField], capacity: u32) -> Result<(u64, u64)> {
        let rows = capacity as u64 + 1;
        fields.iter().try_fold((0, 0), |(device, host), field| {
            let bytes = rows * field.slot_bytes()? as u64;
            Ok(if field.host {
                (device, host + bytes)
            } else {
                (device + bytes, host)
            })
        })
    }

    /// Allocate banks for `capacity` slots on `device` (host memory for
    /// every field when `None`).
    pub(crate) fn allocate(
        fields: &[SlotField],
        capacity: u32,
        device: Option<DeviceId>,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Invalid(
                "request slots need a positive capacity".into(),
            ));
        }
        let rows = i64::from(capacity) + 1;
        let banks = fields
            .iter()
            .map(|field| {
                let bytes = (rows as usize) * field.slot_bytes()?;
                let memory = match (device, field.host) {
                    (Some(device), false) => Storage::device(device, bytes.max(1))?,
                    (Some(device), true) => Storage::pinned(device, bytes.max(1))?,
                    (None, _) => Storage::host(bytes.max(1)),
                };
                let mut shape = vec![rows];
                shape.extend(&field.shape);
                let view =
                    StorageView::contiguous(memory as Arc<dyn Backing>, 0, field.dtype, &shape)?;
                Ok(Bank {
                    field: field.clone(),
                    view,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { banks })
    }

    /// Every bank keyed by field name, as bindings receive them.
    pub(crate) fn views(&self) -> Vec<(String, StorageView)> {
        self.banks
            .iter()
            .map(|bank| (bank.field.name.clone(), bank.view.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banks_add_a_padding_row() -> Result<()> {
        let fields = [
            SlotField {
                name: "tokens".into(),
                dtype: DType::I32,
                shape: vec![4],
                host: false,
            },
            SlotField {
                name: "seed".into(),
                dtype: DType::I64,
                shape: vec![],
                host: true,
            },
        ];
        assert_eq!(RequestSlots::bytes(&fields, 2)?, (3 * 16, 3 * 8));

        let slots = RequestSlots::allocate(&fields, 2, None)?;
        let shapes: Vec<(String, Vec<i64>)> = slots
            .views()
            .into_iter()
            .map(|(name, view)| (name, view.shape().to_vec()))
            .collect();
        assert_eq!(
            shapes,
            [("tokens".into(), vec![3, 4]), ("seed".into(), vec![3])]
        );
        Ok(())
    }
}
