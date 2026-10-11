//! The latent pool of image trajectories and its native page copies.
//!
//! [`ImageLatents`] pairs the pool's page ownership and trajectory
//! visibility ([`LatentPool`]) with its device storage ([`LatentStorage`]).
//! A trajectory's committed latents live in pages of one of two banks; a
//! call that computes on them gathers the committed bank's pages into a
//! contiguous range of the step workspace, and writes its successor back
//! into the other bank's pages, so a failed batch never disturbs the
//! committed version. Gathers and scatters are whole-page driver copies on
//! the batch's stream.
//!
//! Every batch touching the pool (latent preparation, denoising, decoding)
//! runs between [`ImageLatents::enter`] and [`ImageLatents::leave`] on its
//! stream, which orders it after the previous such batch on any other
//! stream: the banks and the shared step workspace are then used by one
//! batch at a time on the device.

use std::collections::HashSet;
use uniserve_cuda::{CopyBatch, SourceOrder, Stream};

use uniserve_ipc::{BufferId, LatentParams, RequestKey};

use crate::executor::forward::launch::StreamOrder;
use crate::executor::forward::transfer::latents::Pool;
use crate::executor::lifecycle::RequestStorage;
use crate::storage::LatentStorage;
use crate::storage::memory::StorageView;
use crate::{Error, Result};

/// A trajectory's place in the step workspace for one call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Scratch {
    /// The trajectory's pages in logical order.
    pub(crate) pages: Vec<usize>,
    /// Its first unit in the step workspace.
    pub(crate) sample: u32,
    /// Its units (the latent rows of its image).
    pub(crate) units: u32,
}

impl Scratch {
    /// Units its pages span, padding included.
    pub(crate) fn padded(&self, page_units: usize) -> u32 {
        (self.pages.len() * page_units) as u32
    }
}

/// The image latent pool of a rank.
pub(crate) struct ImageLatents {
    pub(crate) pool: Pool,
    pub(crate) storage: LatentStorage,
    order: StreamOrder,
}

impl ImageLatents {
    pub(crate) fn new(pool: Pool, storage: LatentStorage) -> Self {
        Self {
            pool,
            storage,
            order: StreamOrder::default(),
        }
    }

    /// Begin a batch's use of the pool on `stream`.
    pub(crate) fn enter(&self, stream: Option<&Stream>) -> Result<()> {
        self.order.wait(stream)
    }

    /// End a batch's use of the pool on `stream`.
    pub(crate) fn leave(&mut self, stream: Option<&Stream>) -> Result<()> {
        self.order.record(stream)
    }

    /// The step workspace `[units, width]`.
    pub(crate) fn workspace(&self) -> Result<&StorageView> {
        self.storage
            .step()
            .ok_or_else(|| Error::Invariant("the image latent pool has no step workspace".into()))
    }

    /// Place a batch's trajectories in disjoint ranges of the step
    /// workspace, after validating their page tables against the pool.
    pub(crate) fn place(&self, params: &[&LatentParams]) -> Result<Vec<Scratch>> {
        let page_units = self.pool.page_units();
        let mut placed: Vec<Scratch> = Vec::with_capacity(params.len());
        for value in params {
            let occupied: Vec<(Vec<usize>, usize)> = placed
                .iter()
                .map(|scratch| (scratch.pages.clone(), scratch.sample as usize / page_units))
                .collect();
            let tables = [value
                .page_table
                .iter()
                .map(|&page| i64::from(page))
                .collect::<Vec<_>>()];
            let (mut pages, offset) =
                self.pool
                    .bind(&tables, &[i64::from(value.latent_units)], &occupied)?;
            placed.push(Scratch {
                pages: pages.remove(0),
                sample: (offset * page_units) as u32,
                units: value.latent_units,
            });
        }
        Ok(placed)
    }

    /// Copy `bank`'s pages of `scratch` into its step workspace range.
    pub(crate) fn gather(
        &self,
        bank: u8,
        scratch: &Scratch,
        stream: Option<&Stream>,
    ) -> Result<()> {
        self.copy(bank, scratch, false, stream)
    }

    /// Copy `scratch`'s step workspace range into `bank`'s pages.
    pub(crate) fn scatter(
        &self,
        bank: u8,
        scratch: &Scratch,
        stream: Option<&Stream>,
    ) -> Result<()> {
        self.copy(bank, scratch, true, stream)
    }

    fn copy(
        &self,
        bank: u8,
        scratch: &Scratch,
        scatter: bool,
        stream: Option<&Stream>,
    ) -> Result<()> {
        let page_units = self.pool.page_units() as i64;
        let workspace = self.workspace()?;
        let mut copies = CopyBatch::new(SourceOrder::Stream);
        for (index, &page) in scratch.pages.iter().enumerate() {
            let page = self.storage.page(bank, page as u32)?;
            let rows = workspace.narrow(
                0,
                i64::from(scratch.sample) + index as i64 * page_units,
                page_units,
            )?;
            let bytes = page.nbytes() as usize;
            if scatter {
                copies.push(page.address(), rows.address(), bytes);
            } else {
                copies.push(rows.address(), page.address(), bytes);
            }
        }
        match stream {
            // SAFETY: whole pages of this pool and disjoint workspace rows of
            // the batch, ordered on the batch's stream after their producers
            // (`enter`).
            Some(stream) => unsafe { copies.submit(stream) }.map_err(Error::from),
            None => {
                for (destination, source, bytes) in copies.iter() {
                    // SAFETY: host pages and workspace of this pool.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            source as *const u8,
                            destination as *mut u8,
                            bytes,
                        );
                    }
                }
                Ok(())
            }
        }
    }

    /// Set slot `slot`'s timestep (the solver update's reference time).
    pub(crate) fn fill_timestep(
        &self,
        slot: u32,
        value: f32,
        stream: Option<&Stream>,
    ) -> Result<()> {
        self.storage.fill_timestep(slot, value, stream).map(drop)
    }
}

impl RequestStorage for ImageLatents {
    fn admit_slots(&mut self, _slots: &[usize]) -> Result<()> {
        Ok(())
    }

    fn release_buffers(&mut self, buffers: &HashSet<BufferId>) -> Result<()> {
        self.pool
            .release_exports(&buffers.iter().copied().collect::<Vec<_>>());
        Ok(())
    }

    fn release_requests(
        &mut self,
        requests: &HashSet<RequestKey>,
        retained: &HashSet<BufferId>,
    ) -> Result<()> {
        let mut result = Ok(());
        // Finished requests' published versions retire once their
        // readers did, except retained ones.
        let exports: Vec<BufferId> = self
            .pool
            .exports()
            .map(|source| source.buffer)
            .filter(|buffer| requests.contains(&buffer.owner) && !retained.contains(buffer))
            .collect();
        self.pool.release_exports(&exports);
        for transfer in self.pool.cancel_imports(requests) {
            result = result.and(transfer.read().cancel());
        }
        result
    }

    fn retirement_ready(
        &mut self,
        _buffers: &HashSet<BufferId>,
        requests: &HashSet<RequestKey>,
        _retained: &HashSet<BufferId>,
    ) -> Result<bool> {
        Ok(self.pool.retirement_ready(requests))
    }

    fn retire_slots(&mut self, slots: &[usize]) -> Result<()> {
        let slots: Vec<i64> = slots.iter().map(|&slot| slot as i64).collect();
        for transfer in self.pool.release_slots(&slots)? {
            transfer.read().cancel()?;
        }
        Ok(())
    }

    fn reap(&mut self) -> Result<()> {
        self.pool.reap();
        Ok(())
    }
}
