//! Latent trajectory transfers between latent pools.
//!
//! A trajectory's committed version lives in pages of one bank of its
//! slot's latent pool. Exporting a version publishes those pages, as the
//! logical tensor `[latent_units, width]`, through the rank's export
//! mechanisms (`TransferHandle::Latent`, carrying the version's image size
//! and step); the pool keeps the bank from being overwritten until every
//! location retired. Importing a version reads it into bank zero of the
//! pages the scheduler assigned the consuming request's slot; adopting the
//! import makes it the slot's committed trajectory, which the consuming
//! call then reads like one this rank computed.
//!
//! These functions serve every runtime that owns a latent pool: the image
//! runtime exports the versions its preparation and denoising produce, the
//! transfer runtime exports resident versions for tensor transfers, and
//! batch input preparation imports versions other ranks published.

use std::sync::Arc;

use uniserve_cuda::Stream;
use uniserve_ipc::{
    BufferId, Call, LatentParams, TensorExport, TensorRef, TensorTransfer, TransferHandle,
};

use crate::executor::PendingOutput;
use crate::executor::forward::products::Publications;
use crate::storage::memory::{dtype_name, parse_dtype};
use crate::storage::{LatentExport, LatentImport, LatentPool, LatentStorage};
use crate::stream::Issue;
use crate::transfer::exports;
use crate::transfer::{Read, ReadRef, Retirement, Transports, fetch};
use crate::{Error, Result};

/// A latent pool whose imports retain their transport reads and whose
/// exports retain their locations' retirements.
pub(crate) type Pool = LatentPool<Arc<LatentImport<ReadRef>>, Arc<LatentExport<Arc<Retirement>>>>;

/// The `u32` pages of a scheduler page table.
fn pages(params: &LatentParams) -> Vec<u32> {
    params.page_table.clone()
}

/// The logical shape of a trajectory version of `params` in `storage`.
fn shape(storage: &LatentStorage, params: &LatentParams) -> [u64; 2] {
    [u64::from(params.latent_units), u64::from(storage.width())]
}

/// Export the version of `slot`'s trajectory in `bank` (at solver `step`,
/// with `params`' pages and image size) as `call`'s output `product`, on
/// `stream`, which produced or last wrote it. The export is registered with
/// the pool at once, its locations join `publications` and its descriptor
/// the call's completion products. Returns the export's buffer, which the
/// batch releases from the pool if it fails before committing.
///
/// The export is issued on `stream` and located at once.
#[expect(
    clippy::too_many_arguments,
    reason = "one latent export's complete inputs"
)]
pub(crate) fn export(
    pool: &mut Pool,
    storage: &LatentStorage,
    transports: &Transports,
    call: &Call,
    product: &TensorRef,
    slot: usize,
    params: &LatentParams,
    bank: u8,
    step: u32,
    stream: Option<&Stream>,
    publications: &mut Publications,
    output: &mut PendingOutput,
) -> Result<BufferId> {
    let export = issue_export(
        pool, storage, transports, call, product, slot, params, bank, step, stream,
    )?;
    locate_export(pool, export, publications, output)
}

/// A latent version's export while the work it issued is outstanding.
pub(crate) struct PendingExport {
    source: Arc<LatentExport<Arc<Retirement>>>,
    exports: exports::Exports,
    export: TensorExport,
}

impl PendingExport {
    /// The buffer of the exported version, which the batch releases from
    /// the pool if it fails before committing.
    pub(crate) fn buffer(&self) -> BufferId {
        self.source.buffer
    }

    /// Revoke the locations of an export that is not published.
    pub(crate) fn revoke(&self) {
        self.exports.revoke();
    }
}

/// Issue the export of the version of `slot`'s trajectory in `bank` to
/// `stream`, behind the work that wrote it. [`locate_export`] publishes it
/// once that work was performed.
#[expect(
    clippy::too_many_arguments,
    reason = "one latent export's complete inputs"
)]
pub(crate) fn issue_export(
    pool: &mut Pool,
    storage: &LatentStorage,
    transports: &Transports,
    call: &Call,
    product: &TensorRef,
    slot: usize,
    params: &LatentParams,
    bank: u8,
    step: u32,
    stream: impl Issue,
) -> Result<PendingExport> {
    let shape = shape(storage, params);
    if !product.shape_bound.contains_shape(&shape)
        || parse_dtype(product.dtype.storage_name())? != storage.dtype()
    {
        return Err(Error::Invalid(
            "latent export disagrees with its declared representation".into(),
        ));
    }
    let buffer = product.buffer_id();
    let source = Arc::new(
        pool.prepare_export(
            buffer,
            slot,
            bank,
            params
                .page_table
                .iter()
                .map(|&page| page as usize)
                .collect(),
        )?,
    );
    let spans = storage.spans(bank, &pages(params), params.latent_units)?;
    pool.register_export(Arc::clone(&source));

    let consumers: Vec<usize> = call
        .consumer_slots
        .iter()
        .map(|&slot| slot as usize)
        .collect();
    let exported = exports::export(
        transports,
        &spans,
        None,
        &consumers,
        false,
        stream,
        |retirement| pool.retain_export(&source, retirement),
    );
    let exports = match exported {
        Ok(exports) => exports,
        Err(error) => {
            pool.release_exports(&[buffer]);
            return Err(error);
        }
    };

    // The transfer record is complete but for its locations.
    let export = TensorExport {
        product: product.clone(),
        value: TransferHandle::Latent {
            height: params.height,
            width: params.width,
            latent_units: params.latent_units,
            step,
            tensor: TensorTransfer {
                shape: shape.to_vec(),
                locations: Vec::new(),
            },
        },
    };
    Ok(PendingExport {
        source,
        exports,
        export,
    })
}

/// Publish an issued export once its work was performed: its locations
/// register with the batch's `publications` and its descriptor joins the
/// call's completion products. Returns the exported version's buffer. On
/// failure the version is released from the pool.
pub(crate) fn locate_export(
    pool: &Pool,
    export: PendingExport,
    publications: &mut Publications,
    output: &mut PendingOutput,
) -> Result<BufferId> {
    let PendingExport {
        source,
        exports,
        mut export,
    } = export;
    let buffer = source.buffer;
    let locations = match exports.locate(|retirement| pool.retain_export(&source, retirement)) {
        Ok(locations) => locations,
        Err(error) => {
            pool.release_exports(&[buffer]);
            return Err(error);
        }
    };
    if let TransferHandle::Latent { tensor, .. } = &mut export.value {
        tensor.locations = locations
            .iter()
            .map(|(_, locator)| locator.clone())
            .collect();
    }
    publications.accept(buffer, locations, export, output)?;
    Ok(buffer)
}

/// A latent version being read into a consuming request's slot.
pub(crate) struct LatentInput {
    write: Arc<LatentImport<ReadRef>>,
    reads: Vec<Arc<Read>>,
    step: u32,
    height: u32,
    width: u32,
}

impl LatentInput {
    /// The transport reads filling the destination pages.
    pub(crate) fn reads(&self) -> &[Arc<Read>] {
        &self.reads
    }

    /// Whether the consuming call adopted the import.
    pub(crate) fn adopted(&self) -> bool {
        self.write.adopted()
    }
}

/// Read the latent version `export` describes into bank zero of `params`'
/// pages of `slot` (the consuming call's), ordered after `stream`. The
/// version's size and step must be the call's. A refusal for read credits
/// starts no read and leaves the slot free, so the same import can be
/// admitted again once credits return.
pub(crate) fn import(
    pool: &mut Pool,
    storage: &LatentStorage,
    transports: &Transports,
    export: &TensorExport,
    params: &LatentParams,
    slot: usize,
    stream: Option<&Stream>,
) -> Result<LatentInput> {
    let TransferHandle::Latent {
        height,
        width,
        latent_units,
        step,
        tensor,
    } = &export.value
    else {
        return Err(Error::Invariant(
            "a latent import names another product kind".into(),
        ));
    };
    if (*latent_units, *height, *width, *step)
        != (
            params.latent_units,
            params.height,
            params.width,
            params.start_step,
        )
    {
        return Err(Error::Invalid(
            "latent transfer disagrees with its scheduler parameters".into(),
        ));
    }
    let dtype = tensor
        .locations
        .first()
        .map(|location| location.dtype.as_str());
    if tensor.shape != shape(storage, params) || dtype != Some(dtype_name(storage.dtype())) {
        return Err(Error::Invalid(
            "latent transfer does not match the destination pool layout".into(),
        ));
    }

    let slot = pool.slot(slot as i64)?;
    let write = Arc::new(
        pool.prepare_import(
            export.product.buffer_id(),
            slot,
            params
                .page_table
                .iter()
                .map(|&page| page as usize)
                .collect(),
            params.latent_units as usize,
        )?,
    );
    let destination = storage.spans(0, &pages(params), params.latent_units)?;
    pool.register_import(Arc::clone(&write));

    let region: Vec<_> = tensor.shape.iter().map(|&extent| 0..extent).collect();
    let fetched = fetch::fetch_tensor(tensor, &destination, transports, &region, stream, |read| {
        pool.retain_transfer(&write, ReadRef::new(read))
    });
    match fetched {
        Ok(reads) => Ok(LatentInput {
            write,
            reads,
            step: *step,
            height: *height,
            width: *width,
        }),
        Err(error) => {
            // Submitted reads were cancelled by the fetch; the pages return
            // once those retire.
            pool.abandon_import(&write)?;
            Err(error)
        }
    }
}

/// Make an imported version the consuming slot's committed trajectory:
/// order `stream` after every read's copy, then adopt it.
pub(crate) fn adopt(pool: &mut Pool, input: &LatentInput, stream: Option<&Stream>) -> Result<()> {
    let generation = i64::from(input.write.buffer.generation);
    let (step, height, width) = (
        i64::from(input.step),
        i64::from(input.height),
        i64::from(input.width),
    );
    pool.validate_adoption(&input.write, generation, step, height, width)?;
    for read in &input.reads {
        read.result(stream)?;
    }
    pool.adopt_import(&input.write, generation, step, height, width);
    Ok(())
}

/// Give up an import its batch did not adopt: cancel its reads; its pages
/// return once they retire.
pub(crate) fn abandon(pool: &Pool, input: &LatentInput) -> Result<()> {
    let mut result = Ok(());
    for read in pool.abandon_import(&input.write)? {
        result = result.and(read.read().cancel());
    }
    for read in &input.reads {
        result = result.and(read.close());
    }
    result
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use uniserve_core::{CallId, RequestId};
    use uniserve_ipc::{
        CallKind, DType as WireDType, DimBound, RequestKey, ShapeBound, TransferMode,
        WorkerEndpoint,
    };

    use super::*;
    use crate::ffi::DType;
    use crate::request::RequestProgress;
    use crate::storage::LatentUpdate;
    use crate::transfer::{Capacity, TransportKind, TransportOptions, notify};
    use crate::{EventPool, PendingOutput};

    const SLOT: usize = 1;
    /// Values per latent unit.
    const WIDTH: u32 = 3;

    fn key() -> RequestKey {
        RequestKey::new(1, RequestId(7), 1)
    }

    /// A pool of two slots and three usable two-unit pages on host memory.
    fn pool() -> Result<(Pool, LatentStorage)> {
        Ok((
            LatentPool::new(2, 4, 2)?,
            LatentStorage::allocate(2, 4, 2, WIDTH, DType::F32, false, None, None)?,
        ))
    }

    fn transports() -> Result<Transports> {
        let options = TransportOptions {
            source: WorkerEndpoint {
                worker_id: "worker".into(),
                rank: 0,
                node: "node".into(),
                address_space: "process".into(),
                incarnation: "incarnation".into(),
            },
            acknowledgment_slot: 0,
            host_slots: HashSet::new(),
            cross_host_consumers: false,
            vmm_pool_bytes: 1 << 20,
        };
        Transports::new(
            &[TransportKind::Local],
            Arc::new(Capacity::new(1 << 20, 8, notify)?),
            EventPool::new(),
            &options,
        )
    }

    /// Three units of a 16x16 image on `pages`, at solver step `step`.
    fn params(pages: &[u32], step: u32) -> LatentParams {
        LatentParams {
            request_key: key(),
            call_id: CallId::new(5, 0),
            page_table: pages.to_vec(),
            latent_units: 3,
            height: 16,
            width: 16,
            start_step: step,
            step_count: 0,
        }
    }

    fn latent(generation: u32) -> TensorRef {
        TensorRef {
            request_key: key(),
            producer_call_id: CallId::new(generation.into(), 0),
            output_index: 0,
            generation,
            dtype: WireDType::F32,
            shape_bound: ShapeBound {
                dims: vec![DimBound::Static(3), DimBound::Static(WIDTH)],
            },
        }
    }

    fn values(storage: &LatentStorage, bank: u8, pages: &[u32]) -> Result<Vec<f32>> {
        let mut values = Vec::new();
        for view in storage.spans(bank, pages, 3)?.views() {
            // SAFETY: host latent storage without concurrent writers.
            values.extend_from_slice(unsafe {
                std::slice::from_raw_parts(view.address() as *const f32, view.numel() as usize)
            });
        }
        Ok(values)
    }

    /// Commit a first version (generation 1, step 0) of `SLOT`'s trajectory
    /// on `pages` holding `0..9`; returns its bank.
    fn commit(pool: &mut Pool, storage: &LatentStorage, pages: &[u32]) -> Result<u8> {
        let update = LatentUpdate {
            request_pool_idx: SLOT as i64,
            params: params(pages, 0),
            expected_generation: 0,
            expected_step: 0,
            generation: 1,
            step: 0,
            release: false,
        };
        pool.validate_updates(std::slice::from_ref(&update))?;
        pool.apply_updates(&[update]);
        let bank = pool.current_bank(
            SLOT,
            0,
            1,
            3,
            16,
            16,
            &pages.iter().map(|&page| page as usize).collect::<Vec<_>>(),
        )?;
        let mut next = 0.0_f32;
        for view in storage.spans(bank, pages, 3)?.views() {
            // SAFETY: host latent storage without concurrent readers.
            let units = unsafe {
                std::slice::from_raw_parts_mut(view.address() as *mut f32, view.numel() as usize)
            };
            for value in units {
                *value = next;
                next += 1.0;
            }
        }
        Ok(bank)
    }

    /// The export of `SLOT`'s committed version as tensor transfer output
    /// `latent(2)`.
    fn publish(
        pool: &mut Pool,
        storage: &LatentStorage,
        transports: &Transports,
        pages: &[u32],
        bank: u8,
        publications: &mut Publications,
    ) -> Result<TensorExport> {
        let mut call = Call::new(
            key(),
            CallId::new(5, 0),
            CallKind::Transfer(TransferMode::Tensor),
        );
        call.latent_input = Some(latent(1));
        call.latent_output = Some(latent(2));
        let mut output = PendingOutput::new(&call, RequestProgress::default(), 0)?;
        export(
            pool,
            storage,
            transports,
            &call,
            &latent(2),
            SLOT,
            &params(pages, 0),
            bank,
            0,
            None,
            publications,
            &mut output,
        )?;
        output
            .products()
            .first()
            .cloned()
            .ok_or_else(|| Error::Invariant("the export published no product".into()))
    }

    #[test]
    fn a_trajectory_version_moves_into_another_pool() -> Result<()> {
        let transports = transports()?;
        let (mut source, source_storage) = pool()?;
        let bank = commit(&mut source, &source_storage, &[2, 3])?;
        let mut publications = Publications::default();
        let export = publish(
            &mut source,
            &source_storage,
            &transports,
            &[2, 3],
            bank,
            &mut publications,
        )?;
        // A published bank is not overwritten while it is exported.
        assert!(source.require_writable(bank, &[2, 3]).is_err());

        // The consumer's own pages receive the version, which becomes its
        // slot's committed trajectory once adopted.
        let (mut target, target_storage) = pool()?;
        let input = import(
            &mut target,
            &target_storage,
            &transports,
            &export,
            &params(&[3, 1], 0),
            SLOT,
            None,
        )?;
        for read in input.reads() {
            read.wait_ready();
        }
        adopt(&mut target, &input, None)?;
        assert!(input.adopted());
        assert_eq!(target.current_bank(SLOT, 0, 2, 3, 16, 16, &[3, 1])?, 0);
        assert_eq!(
            values(&target_storage, 0, &[3, 1])?,
            (0..9).map(|value| value as f32).collect::<Vec<_>>()
        );

        // Revoked and released, the version retires and frees its bank.
        for read in input.reads() {
            read.close()?;
        }
        publications.revoke()?;
        source.release_exports(&[latent(2).buffer_id()]);
        transports.reap()?;
        source.reap();
        source.require_writable(bank, &[2, 3])?;
        transports.close()
    }

    #[test]
    fn an_import_reads_only_the_version_its_call_was_scheduled() -> Result<()> {
        let transports = transports()?;
        let (mut source, source_storage) = pool()?;
        let bank = commit(&mut source, &source_storage, &[2, 3])?;
        let mut publications = Publications::default();
        let export = publish(
            &mut source,
            &source_storage,
            &transports,
            &[2, 3],
            bank,
            &mut publications,
        )?;

        // Another step of the trajectory is refused before any page is
        // claimed; the scheduled one is then admitted into the same slot.
        let (mut target, target_storage) = pool()?;
        assert!(
            import(
                &mut target,
                &target_storage,
                &transports,
                &export,
                &params(&[1, 2], 4),
                SLOT,
                None,
            )
            .is_err()
        );
        let input = import(
            &mut target,
            &target_storage,
            &transports,
            &export,
            &params(&[1, 2], 0),
            SLOT,
            None,
        )?;

        // An abandoned import never becomes the slot's trajectory, and its
        // pages return once its reads retired.
        abandon(&target, &input)?;
        assert!(!input.adopted());
        for read in input.reads() {
            read.retirement()
                .wait(Some(std::time::Duration::from_secs(10)));
        }
        transports.reap()?;
        target.reap();
        target.release_slots(&[SLOT as i64])?;
        target.reap();
        assert!(target.idle());
        publications.revoke()?;
        transports.close()
    }
}
