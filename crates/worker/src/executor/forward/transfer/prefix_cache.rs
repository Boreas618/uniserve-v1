//! Cache transfers: publication of a request's visible cache extent
//! (`CacheExport`) and installation of an imported extent (`CacheInstall`).
//!
//! An export publishes the extent past the destination's accepted base
//! through the rank's export mechanisms. It aliases the request's live
//! pages (an FP8 pool freezes them first), which the cache manager protects
//! from writes until the export retires; its locations are revoked when the
//! export buffer or its request is released. Exports become resident (the
//! base of the destination's next export) at the commit point of their
//! batch. Every export names the image generation destination: the
//! engine's cache consumers are an image's conditioning and installs of that
//! conditioning elsewhere.
//!
//! An install adopts a finished import of its source export into its
//! slot's pages: the slot's verified length becomes the exported extent on
//! the batch's stream, and the import's pages become the slot's prefix.
//! The installed buffer becomes resident beside its source at the commit
//! point, which also accepts the extent as the destination's base.

use std::sync::Arc;

use uniserve_ipc::{BufferId, CacheTransfer, Call};

use crate::storage::{PendingImport, PrefixCache, PrefixCacheManager};
use crate::stream::Issue;
use crate::transfer::Retirement;
use crate::transfer::cache_export::CacheExport;
use crate::transfer::exports::{self, Location};
use crate::worker::RankStorage;
use crate::{Error, Result};

/// The cache transfer destination of image generation conditioning.
pub(crate) const DESTINATION: &str = "gen";

/// One call's export while the work `stream` was given for it is
/// outstanding ([`export`]).
pub(crate) struct Exporting {
    /// The call's index in its batch.
    call: usize,
    buffer: BufferId,
    export: CacheExport,
}

impl Exporting {
    /// The export with its transfer and locations, once the issuer's work
    /// was performed: at once after a stream, from the job's completion on
    /// after a recorder.
    pub(crate) fn locate(self, storage: &mut RankStorage) -> Result<Published> {
        let (transfer, locations) = self.export.locate(manager(storage)?)?;
        Ok(Published {
            call: self.call,
            buffer: self.buffer,
            transfer,
            locations,
        })
    }

    /// Revoke an export that is not published, under the same condition as
    /// [`Self::locate`].
    pub(crate) fn revoke(self, storage: &mut RankStorage) -> Result<()> {
        self.export.revoke(manager(storage)?);
        Ok(())
    }
}

/// The cache manager of a rank that transfers prefix cache.
fn manager(storage: &mut RankStorage) -> Result<&mut PrefixCacheManager<Arc<Retirement>>> {
    storage
        .prefix_cache
        .as_mut()
        .map(PrefixCache::manager)
        .ok_or_else(|| Error::Unsupported("cache transfers require a prefix cache".into()))
}

/// One call's export, committed with its batch.
pub(crate) struct Published {
    /// The call's index in its batch.
    pub(crate) call: usize,
    pub(crate) buffer: BufferId,
    pub(crate) transfer: CacheTransfer,
    pub(crate) locations: Vec<Location>,
}

/// An install accepted by its batch: the source export, the installed
/// buffer and the transfer, resident once the batch commits.
pub(crate) type Installation = (BufferId, BufferId, CacheTransfer);

/// Export call `index`'s visible cache extent (`visible` tokens of `slot`) in
/// the order of `stream`, which orders after the batches that wrote it.
pub(crate) fn export(
    storage: &mut RankStorage,
    index: usize,
    call: &Call,
    slot: usize,
    visible: u64,
    stream: impl Issue,
) -> Result<Exporting> {
    let buffer = call
        .cache_output
        .ok_or_else(|| Error::Invalid("cache transfer requires an output buffer".into()))?;
    if storage.transports.exports_nothing() {
        return Err(Error::Unsupported(
            "cache export requires a configured transport".into(),
        ));
    }
    let slot = u32::try_from(slot).map_err(|_| Error::Invalid("request slot overflows".into()))?;
    let visible = u32::try_from(visible)
        .map_err(|_| Error::Invalid("visible cache exceeds the token range".into()))?;
    let consumers: Vec<usize> = call
        .consumer_slots
        .iter()
        .map(|&slot| slot as usize)
        .collect();
    let prefix_cache = storage
        .prefix_cache
        .as_mut()
        .ok_or_else(|| Error::Unsupported("cache export requires a prefix cache".into()))?;
    let export = prefix_cache.export(
        slot,
        visible,
        DESTINATION,
        buffer,
        &storage.transports,
        &consumers,
        stream,
    )?;
    Ok(Exporting {
        call: index,
        buffer,
        export,
    })
}

/// Adopt `import` of `call`'s source export into its slot in `stream`'s
/// order; returns the installation and the installed extent.
pub(crate) fn install(
    storage: &mut RankStorage,
    call: &Call,
    import: &PendingImport,
    stream: impl Issue,
) -> Result<(Installation, u64)> {
    let source = call
        .cache_input
        .ok_or_else(|| Error::Invalid("cache installation requires an input buffer".into()))?;
    let installed = call
        .cache_output
        .ok_or_else(|| Error::Invalid("cache transfer requires an output buffer".into()))?;
    if import.buffer() != source {
        return Err(Error::Invariant(
            "a cache installation adopted another source's import".into(),
        ));
    }
    let prefix_cache = storage
        .prefix_cache
        .as_mut()
        .ok_or_else(|| Error::Invalid("cache installation requires cache storage".into()))?;
    let transfer = prefix_cache.adopt_import(import, installed, stream)?;
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
    let exported: Vec<(BufferId, CacheTransfer)> = published
        .iter()
        .map(|export| (export.buffer, export.transfer.clone()))
        .collect();
    storage
        .prefix_cache
        .as_mut()
        .ok_or_else(|| Error::Unsupported("cache transfers require a prefix cache".into()))?
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
    let exported: Vec<(BufferId, CacheTransfer)> = published
        .iter()
        .map(|export| (export.buffer, export.transfer.clone()))
        .collect();
    storage
        .prefix_cache
        .as_mut()
        .ok_or_else(|| Error::Unsupported("cache transfers require a prefix cache".into()))?
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
    if let Some(prefix_cache) = storage.prefix_cache.as_mut() {
        let buffers: Vec<BufferId> = published.iter().map(|export| export.buffer).collect();
        prefix_cache.manager().release_exports(&buffers);
    }
    result
}
