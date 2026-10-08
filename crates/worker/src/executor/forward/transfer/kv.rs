//! KV transfers: publication of a request's visible KV extent
//! (`KvExport`) and installation of an imported extent (`KvInstall`).
//!
//! An export publishes the extent past the destination's accepted base
//! through the rank's export mechanisms. It aliases the request's live
//! pages (an FP8 pool freezes them first), which the KV manager protects
//! from writes until the export retires; its locations are revoked when the
//! export buffer or its request is released. Exports become resident (the
//! base of the destination's next export) at the commit point of their
//! batch. Every export names the image generation destination: the
//! engine's KV consumers are an image's conditioning and installs of that
//! conditioning elsewhere.
//!
//! An install adopts a finished import of its source export into its
//! slot's pages: the slot's verified length becomes the exported extent on
//! the batch's stream, and the import's pages become the slot's prefix.
//! The installed buffer becomes resident beside its source at the commit
//! point, which also accepts the extent as the destination's base.
//!
//! These functions serve every runtime that executes KV transfers: the
//! image runtime publishes conditioning in its own batches, the transfer
//! runtime serves the rest.

use uniserve_cuda::Stream;
use uniserve_ipc::{BufferId, Call, KvTransfer};

use crate::storage::PendingImport;
use crate::transfer::exports::{self, Location};
use crate::worker::RankStorage;
use crate::{Error, Result};

/// The KV transfer destination of image generation conditioning.
pub(crate) const DESTINATION: &str = "gen";

/// One call's export, committed with its batch.
pub(crate) struct Published {
    /// The call's index in its batch.
    pub(crate) call: usize,
    pub(crate) buffer: BufferId,
    pub(crate) transfer: KvTransfer,
    pub(crate) locations: Vec<Location>,
}

/// An install accepted by its batch: the source export, the installed
/// buffer and the transfer, resident once the batch commits.
pub(crate) type Installation = (BufferId, BufferId, KvTransfer);

/// Export call `index`'s visible KV extent (`visible` tokens of `slot`) on
/// `stream`, which orders after the batches that wrote it.
pub(crate) fn export(
    storage: &mut RankStorage,
    index: usize,
    call: &Call,
    slot: usize,
    visible: u64,
    stream: Option<&Stream>,
) -> Result<Published> {
    let buffer = call
        .kv_output
        .ok_or_else(|| Error::Invalid("KV transfer requires an output buffer".into()))?;
    if storage.transports.exports_nothing() {
        return Err(Error::Unsupported(
            "KV export requires a configured transport".into(),
        ));
    }
    let slot = u32::try_from(slot).map_err(|_| Error::Invalid("request slot overflows".into()))?;
    let visible = u32::try_from(visible)
        .map_err(|_| Error::Invalid("visible KV exceeds the token range".into()))?;
    let consumers: Vec<usize> = call
        .consumer_slots
        .iter()
        .map(|&slot| slot as usize)
        .collect();
    let kv = storage
        .kv
        .as_mut()
        .ok_or_else(|| Error::Unsupported("KV export requires a KV cache".into()))?;
    let (transfer, locations) = kv.export(
        slot,
        visible,
        DESTINATION,
        buffer,
        &storage.transports,
        &consumers,
        stream,
    )?;
    Ok(Published {
        call: index,
        buffer,
        transfer,
        locations,
    })
}

/// Adopt `import` of `call`'s source export into its slot on `stream`;
/// returns the installation and the installed extent.
pub(crate) fn install(
    storage: &mut RankStorage,
    call: &Call,
    import: &PendingImport,
    stream: Option<&Stream>,
) -> Result<(Installation, u64)> {
    let source = call
        .kv_input
        .ok_or_else(|| Error::Invalid("KV installation requires an input buffer".into()))?;
    let installed = call
        .kv_output
        .ok_or_else(|| Error::Invalid("KV transfer requires an output buffer".into()))?;
    if import.buffer() != source {
        return Err(Error::Invariant(
            "a KV installation adopted another source's import".into(),
        ));
    }
    let kv = storage
        .kv
        .as_mut()
        .ok_or_else(|| Error::Invalid("KV installation requires cache storage".into()))?;
    let transfer = kv.adopt_import(import, installed, stream)?;
    let extent = u64::from(transfer.exported_extent);
    Ok(((source, installed, (*transfer).clone()), extent))
}

/// Check, before the commit point, that the exports and installs form
/// chains on their destinations' accepted bases.
pub(crate) fn validate(
    storage: &mut RankStorage,
    published: &[Published],
    installations: &[Installation],
) -> Result<()> {
    if published.is_empty() && installations.is_empty() {
        return Ok(());
    }
    let exported: Vec<(BufferId, KvTransfer)> = published
        .iter()
        .map(|export| (export.buffer, export.transfer.clone()))
        .collect();
    storage
        .kv
        .as_mut()
        .ok_or_else(|| Error::Unsupported("KV transfers require a KV cache".into()))?
        .manager()
        .validate_exports(&exported, installations)
}

/// Make the exports and installs resident at the commit point and register
/// the exports' locations, revoked with the export buffer's release.
pub(crate) fn apply(
    storage: &mut RankStorage,
    published: Vec<Published>,
    installations: Vec<Installation>,
) -> Result<()> {
    if published.is_empty() && installations.is_empty() {
        return Ok(());
    }
    let exported: Vec<(BufferId, KvTransfer)> = published
        .iter()
        .map(|export| (export.buffer, export.transfer.clone()))
        .collect();
    storage
        .kv
        .as_mut()
        .ok_or_else(|| Error::Unsupported("KV transfers require a KV cache".into()))?
        .manager()
        .apply_exports(exported, installations);
    for export in published {
        storage
            .tensors
            .register_exports(export.buffer, export.locations)?;
    }
    Ok(())
}

/// Revoke the exports of a batch that failed before its commit point and
/// release their physical reservations.
pub(crate) fn revoke(storage: &mut RankStorage, published: Vec<Published>) -> Result<()> {
    let mut result = Ok(());
    for export in &published {
        result = result.and(exports::release(&export.locations));
    }
    if let Some(kv) = storage.kv.as_mut() {
        let buffers: Vec<BufferId> = published.iter().map(|export| export.buffer).collect();
        kv.manager().release_exports(&buffers);
    }
    result
}
