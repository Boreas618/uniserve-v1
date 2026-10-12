//! KV exports and installs of a transfer batch (see [`super::kv`]).

use std::collections::HashSet;

use super::super::inputs::Inputs;
use super::batch::TransferBatch;
use super::kv;
use crate::storage::UnitColumn;
use crate::{Error, Result};

impl TransferBatch<'_, '_> {
    /// Install the active calls' block tables on the batch's stream and
    /// reset the units newly assigned to them on `column` (the stream's
    /// unit column), except those an install's import resets itself.
    pub(super) fn reserve_cache(&mut self, column: &UnitColumn, inputs: &Inputs) -> Result<()> {
        let plan = self.plan;
        let issue = self.issue();
        let calls = &self.recorded.calls;
        let active: HashSet<u32> = self
            .active()
            .into_iter()
            .map(|index| calls[index].slot as u32)
            .collect();
        let imported: HashSet<u32> = self
            .active()
            .into_iter()
            .filter(|&index| {
                plan.calls[index]
                    .kv_input
                    .is_some_and(|source| inputs.cache_import(source).is_some())
            })
            .map(|index| calls[index].slot as u32)
            .collect();
        let assignments: Vec<_> = plan
            .block_tables
            .iter()
            .filter(|table| active.contains(&table.request_pool_idx))
            .cloned()
            .collect();
        let kv = self
            .storage
            .kv
            .as_mut()
            .ok_or_else(|| Error::Unsupported("KV transfers require a KV cache".into()))?;
        kv.install(&assignments, issue)?;
        let units: Vec<u32> = plan
            .new_cache_units
            .iter()
            .filter(|allocation| {
                active.contains(&allocation.request_pool_idx)
                    && !imported.contains(&allocation.request_pool_idx)
            })
            .flat_map(|allocation| allocation.unit_ids.iter().map(|unit| unit.0))
            .collect();
        if !units.is_empty() {
            kv.recycle(&units, column, issue)?;
        }
        Ok(())
    }

    /// Export every active call's visible KV extent. The exports are
    /// located, and join their calls' outputs, at the job's completion.
    pub(super) fn export_cache(&mut self) -> Result<()> {
        let plan = self.plan;
        for index in self.active() {
            let state = &self.recorded.calls[index];
            let (slot, visible) = (state.slot, state.output.progress.kv_visible_len);
            let export = kv::export(
                self.storage,
                index,
                &plan.calls[index],
                slot,
                visible,
                self.issue(),
            )?;
            self.recorded.exports.push_back(export);
        }
        Ok(())
    }

    /// Adopt every active call's prepared KV import into its slot; the
    /// call's KV extent becomes the installed one.
    pub(super) fn install_cache(&mut self, inputs: &Inputs) -> Result<()> {
        let plan = self.plan;
        let issue = self.issue();
        for index in self.active() {
            let call = &plan.calls[index];
            let source = call
                .kv_input
                .ok_or_else(|| Error::Invalid("KV installation requires an input buffer".into()))?;
            let import = inputs.cache_import(source).ok_or_else(|| {
                Error::Invalid("KV installation has no reserved physical input".into())
            })?;
            let (installation, extent) = kv::install(self.storage, call, import, issue)?;
            self.recorded.calls[index].output.set_cache_length(extent);
            self.recorded.installations.push(installation);
        }
        Ok(())
    }
}
