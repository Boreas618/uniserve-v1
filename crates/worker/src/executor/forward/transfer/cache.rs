//! Cache exports and installs of a transfer batch (see [`super::prefix_cache`]).

use std::collections::HashSet;

use super::super::inputs::Inputs;
use super::batch::TransferBatch;
use super::prefix_cache;
use crate::storage::UnitColumn;
use crate::{Error, Result};

impl TransferBatch<'_> {
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
                    .cache_input
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
        let prefix_cache =
            self.storage.prefix_cache.as_mut().ok_or_else(|| {
                Error::Unsupported("cache transfers require a prefix cache".into())
            })?;
        prefix_cache.install(&assignments, issue)?;
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
            prefix_cache.recycle(&units, column, issue)?;
        }
        Ok(())
    }

    /// Export every active call's visible cache extent. The exports are
    /// located, and join their calls' outputs, at the job's completion.
    pub(super) fn export_cache(&mut self) -> Result<()> {
        let plan = self.plan;
        for index in self.active() {
            let state = &self.recorded.calls[index];
            let (slot, visible) = (state.slot, state.output.progress.cache_visible_len);
            let export = prefix_cache::export(
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

    /// Adopt every active call's prepared cache import into its slot; the
    /// call's cache extent becomes the installed one.
    pub(super) fn install_cache(&mut self, inputs: &Inputs) -> Result<()> {
        let plan = self.plan;
        let issue = self.issue();
        for index in self.active() {
            let call = &plan.calls[index];
            let source = call.cache_input.ok_or_else(|| {
                Error::Invalid("cache installation requires an input buffer".into())
            })?;
            let import = inputs.cache_import(source).ok_or_else(|| {
                Error::Invalid("cache installation has no reserved physical input".into())
            })?;
            let (installation, extent) = prefix_cache::install(self.storage, call, import, issue)?;
            self.recorded.calls[index].output.set_cache_length(extent);
            self.recorded.installations.push(installation);
        }
        Ok(())
    }
}
