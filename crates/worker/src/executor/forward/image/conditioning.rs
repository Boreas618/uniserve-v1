//! Publication of an image's conditioning KV (`Transfer(KvExport)` to the
//! image generation destination).
//!
//! Before a request generates an image, it publishes its visible KV extent
//! as the conditioning export its latent preparation and denoising calls
//! name (`kv_input`), through the transfer runtime's KV export
//! ([`crate::executor::forward::transfer::kv`]). The export becomes
//! resident when the batch commits.

use uniserve_ipc::ForwardStats;

use super::batch::ImageBatch;
use crate::Result;
use crate::executor::forward::transfer::kv;

impl ImageBatch<'_> {
    /// Export every active call's visible KV extent.
    pub(super) fn export_conditioning(&mut self) -> Result<ForwardStats> {
        let plan = self.plan;
        let stream = self.borrow_stream()?;
        for index in self.active() {
            let export = kv::export(
                self.storage,
                index,
                &plan.calls[index],
                self.calls[index].slot,
                self.calls[index].output.progress.kv_visible_len,
                stream.as_ref(),
            )?;
            self.calls[index].output.output.kv_output = Some(export.transfer.clone());
            self.published.push(export);
        }
        Ok(ForwardStats::default())
    }

    /// Revoke the exports of a batch that failed before its commit.
    pub(super) fn revoke_conditioning(&mut self) -> Result<()> {
        let published = std::mem::take(&mut self.published);
        for export in &published {
            self.calls[export.call].output.output.kv_output = None;
        }
        kv::revoke(self.storage, published)
    }
}
