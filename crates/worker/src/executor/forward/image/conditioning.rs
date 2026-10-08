//! Publication of an image's conditioning KV (`Transfer(KvExport)` to the
//! image generation destination).
//!
//! Before a request generates an image, it publishes its visible KV extent
//! as the conditioning export its latent preparation and denoising calls
//! name (`kv_input`). The export aliases the request's live pages (an FP8
//! pool freezes them first), which the KV manager protects from writes
//! until the export retires; its transport locations are revoked when the
//! request's storage is released. The export becomes resident when the
//! batch commits.

use uniserve_ipc::{BufferId, ForwardStats, KvTransfer};

use super::batch::ImageBatch;
use crate::transfer::exports::{self, Location};
use crate::{Error, Result};

/// The KV transfer destination of image generation conditioning.
pub(super) const DESTINATION: &str = "gen";

/// One call's export, committed with its batch.
pub(super) struct Published {
    pub(super) call: usize,
    pub(super) buffer: BufferId,
    pub(super) transfer: KvTransfer,
    pub(super) locations: Vec<Location>,
}

impl ImageBatch<'_> {
    /// Export every active call's visible KV extent.
    pub(super) fn export_conditioning(&mut self) -> Result<ForwardStats> {
        let plan = self.plan;
        let stream = self.borrow_stream()?;
        for index in self.active() {
            let call = &plan.calls[index];
            let buffer = call
                .kv_output
                .ok_or_else(|| Error::Invalid("KV transfer requires an output buffer".into()))?;
            if self.storage.transports.is_empty() {
                return Err(Error::Unsupported(
                    "KV export requires a configured transport".into(),
                ));
            }
            let slot = self.calls[index].slot as u32;
            let visible = u32::try_from(self.calls[index].output.progress.kv_visible_len)
                .map_err(|_| Error::Invalid("visible KV exceeds the token range".into()))?;
            let consumers: Vec<usize> = call
                .consumer_slots
                .iter()
                .map(|&slot| slot as usize)
                .collect();
            let kv = self
                .storage
                .kv
                .as_mut()
                .ok_or_else(|| Error::Unsupported("KV export requires a KV cache".into()))?;
            let (transfer, locations) = kv.export(
                slot,
                visible,
                DESTINATION,
                buffer,
                &self.storage.transports,
                &consumers,
                stream.as_ref(),
            )?;
            self.calls[index].output.output.kv_output = Some(transfer.clone());
            self.published.push(Published {
                call: index,
                buffer,
                transfer,
                locations,
            });
        }
        Ok(ForwardStats::default())
    }

    /// Revoke the exports of a batch that failed before its commit.
    pub(super) fn revoke_conditioning(&mut self) -> Result<()> {
        let published = std::mem::take(&mut self.published);
        let mut result = Ok(());
        for export in &published {
            result = result.and(exports::release(&export.locations));
            self.calls[export.call].output.output.kv_output = None;
        }
        if let Some(kv) = self.storage.kv.as_mut() {
            let buffers: Vec<BufferId> = published.iter().map(|export| export.buffer).collect();
            kv.manager().release_exports(&buffers);
        }
        result
    }
}
