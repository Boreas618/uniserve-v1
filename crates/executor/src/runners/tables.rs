//! The model-declared tables a media runner holds on the device (specs/worker.md 4.3).
//!
//! A capability states what it needs and fills it; the executor allocates it. Three kinds reach a
//! media runner and they differ in when they are made, not in how: a model's **constants** once per
//! load, a **layout**'s tables once per shape, and a request's own **state** once per layout too
//! (the model is deterministic for a layout, so two requests of one shape share one set). The
//! executor keys every table by its layout's name and the model's own name for it, and never
//! interprets the bytes.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use nn::capability::StateTable;
use ops::TensorView;
use platform::{DeviceId, PoolKind};

use crate::arena::Pools;
use crate::{Error, Result};

/// The pool a model's declared tables are booked against.
///
/// `Resident` is what the allocator's own definition describes for them - storage that lives as long
/// as the epoch, named and retained, filled by its owner - and it needs no fourth figure in the
/// device's budget. A deployment whose model states tables therefore has a resident pool that is not
/// zero, which is the honest figure rather than a special case.
const TABLE_POOL: PoolKind = PoolKind::Resident;

/// One table's device storage and the view a forward reads it through.
struct Held {
    /// The allocation, kept for as long as the table: freeing it would hand the block to the next
    /// allocation while a captured graph still addresses the address.
    _buffer: platform::Buffer,
    view: TensorView,
    /// The table's size: exactly the bytes this table books to the resident pool, so what a
    /// generation holds can be compared with what the plan reserved for it.
    bytes: u64,
}

/// Every table a runner holds, keyed by the layout it belongs to and the model's name for it.
#[derive(Default)]
pub(crate) struct Tables {
    allocator: Option<Arc<Mutex<Pools>>>,
    /// Keyed by the generation the table belongs to, its layout, and the model's name for it: a
    /// generation's tables retire with it, so the key is what `release` drops.
    entries: BTreeMap<(u64, String, &'static str), Held>,
}

impl Tables {
    /// A store with no table yet and no allocator: attached when the device has opened its pools.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Hands the store the device's pools, which every later allocation books against (4.1).
    pub(crate) fn attach(&mut self, pools: Arc<Mutex<Pools>>) {
        self.allocator = Some(pools);
    }

    /// The bytes `tables` need: what the plan books to `Resident` before admission (4.3).
    pub(crate) fn plan_bytes(tables: &[StateTable]) -> Result<u64> {
        tables
            .iter()
            .map(|table| {
                let elements = table.shape.dims().iter().product::<u64>();
                let width = table
                    .dtype
                    .bytes()
                    .ok_or_else(|| Error::Invalid(format!("{} has no whole-byte size", table.dtype)))?;
                width
                    .checked_mul(elements)
                    .ok_or_else(|| Error::Invalid("a state table's extent overflows".into()))
            })
            .sum()
    }

    /// The bytes `generation`'s tables of `layout` hold, which is what the resident pool books for
    /// them. `fill` compares it with the plan's own figure, so a deployment whose tables grew past
    /// its reservation is refused at load rather than after a request.
    pub(crate) fn held_bytes(&self, generation: u64, layout: &str) -> u64 {
        self.entries
            .iter()
            .filter(|((held, at, _), _)| *held == generation && at == layout)
            .map(|(_, table)| table.bytes)
            .sum()
    }

    /// Release every table `generation` holds: the rule that a generation's tables retire with it.
    pub(crate) fn release(&mut self, generation: u64) {
        self.entries.retain(|(held, _, _), _| *held != generation);
    }

    /// Allocates and fills `layout`'s tables for `generation`, once.
    ///
    /// `fill` writes the host bytes of each table in `tables`' order into buffers this method sized
    /// from the model's own extents; a table already held is left as it is, so a second load of one
    /// layout allocates nothing. The bytes travel to the device on `stream`, ordered before the
    /// capture that reads them. This runs at load, never on the request path (4.3).
    pub(crate) fn fill(
        &mut self,
        device: DeviceId,
        stream: &platform::Stream,
        generation: u64,
        layout: &str,
        tables: &[StateTable],
        fill: impl FnOnce(&mut [Vec<u8>]) -> Result<()>,
    ) -> Result<()> {
        let missing: Vec<&StateTable> = tables
            .iter()
            .filter(|table| {
                !self
                    .entries
                    .contains_key(&(generation, layout.to_owned(), table.name))
            })
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        let allocator = self
            .allocator
            .as_ref()
            .ok_or(Error::State(
                "the runner has no pools to allocate its model's tables from",
            ))?;
        let allocator = Arc::clone(allocator);
        let mut host: Vec<Vec<u8>> = Vec::with_capacity(missing.len());
        for table in &missing {
            let bytes = Self::plan_bytes(std::slice::from_ref(*table))?;
            host.push(vec![
                0u8;
                usize::try_from(bytes).map_err(|_| {
                    Error::Invalid("a state table is larger than memory".into())
                })?
            ]);
        }
        fill(&mut host)?;

        // Allocate under the lock, copy outside it: the copy is stream work, and a pinned lock
        // across a device call would block every other runner on this device.
        let mut allocated = Vec::with_capacity(missing.len());
        for (table, bytes) in missing.iter().zip(&host) {
            let buffer = allocator
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .allocator()
                .allocate(TABLE_POOL, bytes.len())?;
            allocated.push((*table, buffer));
        }
        for ((table, buffer), host) in allocated.into_iter().zip(host) {
            // SAFETY: the source is this table's host bytes, of exactly the length uploaded, and the
            // destination is an allocation of at least that length; the copy is stream-ordered before
            // the capture that reads it.
            unsafe {
                platform::memcpy_async(buffer.address(), host.as_ptr() as u64, host.len(), stream)?;
            }
            self.entries.insert(
                (generation, layout.to_owned(), table.name),
                Held {
                    view: TensorView::new(
                        ops::DevicePtr::new(buffer.address()),
                        table.dtype,
                        table.shape,
                    ),
                    _buffer: buffer,
                    bytes: host.len() as u64,
                },
            );
        }
        let planned = Self::plan_bytes(tables)?;
        let held = self.held_bytes(generation, layout);
        if held != planned {
            return Err(Error::Resource(
                "the tables this runner loaded for one layout are not the bytes the plan reserved \
                 for them: the deployment was admitted on the plan's figure",
            ));
        }
        let _ = device;
        Ok(())
    }

    /// The views of `layout`'s `tables`, in the model's order.
    pub(crate) fn views(
        &self,
        generation: u64,
        layout: &str,
        tables: &[StateTable],
    ) -> Result<Vec<TensorView>> {
        tables
            .iter()
            .map(|table| {
                self.entries
                    .get(&(generation, layout.to_owned(), table.name))
                    .map(|held| held.view)
                    .ok_or_else(|| {
                        Error::Invalid(format!(
                            "table {} of layout {layout} is not allocated",
                            table.name
                        ))
                    })
            })
            .collect()
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use ops::{DType, Shape};

    fn table(name: &'static str, dtype: DType, extents: &[u64]) -> StateTable {
        StateTable {
            name,
            dtype,
            shape: Shape::new(extents),
        }
    }

    /// The figure the plan reserves for a media class is the sum of what the model says its tables
    /// are, and nothing else: a deployment whose admission passes has already paid for them (4.3).
    #[test]
    fn the_plan_books_every_table_a_layout_declares() {
        let tables = [
            table("text_state", DType::F32, &[8, 2048]),
            table("latent_state", DType::F32, &[16, 64]),
            table("counters", DType::I32, &[32]),
        ];
        let bytes = Tables::plan_bytes(&tables).expect("a state table set has a size");
        assert_eq!(bytes, 8 * 2048 * 4 + 16 * 64 * 4 + 32 * 4);

        // The same set states the same figure however often it is asked: the resident figure is a
        // function of the model's declaration, not of when the plan is made.
        assert_eq!(Tables::plan_bytes(&tables).expect("the same set"), bytes);
        // A layout that declares no table costs nothing, so a causal-LM class's resident figure is
        // still zero.
        assert_eq!(Tables::plan_bytes(&[]).expect("no table"), 0);
    }

    /// An element type with no whole-byte size cannot be booked, and the plan must say so rather
    /// than round: a table whose size is unknown would pass admission with a figure no allocation
    /// honours.
    #[test]
    fn a_table_of_a_packed_element_type_is_refused() {
        let refused = Tables::plan_bytes(&[table("packed", DType::F4E2M1, &[16])]);
        assert!(refused.is_err(), "a packed table has no whole-byte size");
    }
}
