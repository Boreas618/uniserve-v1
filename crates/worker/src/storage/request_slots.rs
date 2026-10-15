//! Per-request-slot state banks.

use std::sync::Arc;

use super::memory::{Memory, StorageView};
use crate::ffi::{Backing, DType};
use crate::{Error, Result};

/// One field of per-slot state a model declares: `shape` is one slot's
/// extent; host fields live in page-locked host memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotField {
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
/// slots are 1-based. Banks are bound once; a slot's state is a row view,
/// made once with its bank, so each row crosses to Python as one torch
/// tensor for the bank's life. Contents are uninitialized until their owner
/// writes them.
pub struct RequestSlots {
    capacity: u32,
    banks: Vec<Bank>,
}

/// One field's bank and its row views, the padding row first.
struct Bank {
    field: SlotField,
    view: StorageView,
    rows: Vec<StorageView>,
}

impl RequestSlots {
    /// Device and host bytes of the banks: `(device, host)`.
    pub fn bytes(fields: &[SlotField], capacity: u32) -> Result<(u64, u64)> {
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
    pub fn allocate(fields: &[SlotField], capacity: u32, device: Option<i32>) -> Result<Self> {
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
                    (Some(device), false) => Memory::device(device, bytes.max(1))?,
                    (Some(device), true) => Memory::pinned(device, bytes.max(1))?,
                    (None, _) => Memory::host(bytes.max(1)),
                };
                let mut shape = vec![rows];
                shape.extend(&field.shape);
                let view =
                    StorageView::contiguous(memory as Arc<dyn Backing>, 0, field.dtype, &shape)?;
                let slots = (0..rows)
                    .map(|row| view.select(0, row))
                    .collect::<Result<Vec<_>>>()?;
                Ok(Bank {
                    field: field.clone(),
                    view,
                    rows: slots,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { capacity, banks })
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Every bank keyed by field name, as bindings receive them.
    pub fn views(&self) -> Vec<(String, StorageView)> {
        self.banks
            .iter()
            .map(|bank| (bank.field.name.clone(), bank.view.clone()))
            .collect()
    }

    /// A field's whole bank.
    pub fn bank(&self, name: &str) -> Result<&StorageView> {
        self.find(name).map(|bank| &bank.view)
    }

    /// One request slot's row of a field; slots are 1-based.
    pub fn slot(&self, name: &str, slot: u32) -> Result<StorageView> {
        if slot == 0 || slot > self.capacity {
            return Err(Error::Invalid(format!(
                "request slot {slot} is outside capacity {}",
                self.capacity
            )));
        }
        Ok(self.find(name)?.rows[slot as usize].clone())
    }

    fn find(&self, name: &str) -> Result<&Bank> {
        self.banks
            .iter()
            .find(|bank| bank.field.name == name)
            .ok_or_else(|| Error::Invalid(format!("request slots have no field {name}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn banks_add_a_padding_row_and_index_slots_from_one() -> Result<()> {
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
        let bank = slots.bank("tokens")?;
        assert_eq!(bank.shape(), &[3, 4]);
        let second = slots.slot("tokens", 2)?;
        assert_eq!(second.shape(), &[4]);
        assert_eq!(second.address(), bank.address() + 2 * 16);
        assert!(slots.slot("tokens", 0).is_err());
        assert!(slots.slot("tokens", 3).is_err());
        assert!(slots.slot("missing", 1).is_err());
        Ok(())
    }
}
