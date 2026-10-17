//! Device storage of the latent pool: page banks, step workspace and
//! per-slot timesteps.
//!
//! [`LatentPool`](super::LatentPool) decides page ownership and trajectory
//! visibility; this owner holds the memory the numerical denoising steps
//! bind once and the native writes the pool's decisions need (timestep
//! fills). Page 0 is padding and request slots are 1-based.

use std::sync::Arc;

use uniserve_cuda::{DeviceId, Stream, memset_d8};

use crate::ffi::{Backing, DType, MemoryKind};
use crate::storage::memory::{Memory, StorageView};
use crate::stream::{Fill, Issue};
use crate::transfer::Spans;
use crate::{Error, Result};

/// The latent pool's storage on one device (or host memory for CPU lanes).
pub struct LatentStorage {
    num_pages: u32,
    page_units: u32,
    width: u32,
    request_pool_size: u32,
    /// `[2, num_pages, page_units, width]`: two banks that alternate across
    /// denoising steps.
    pages: StorageView,
    /// `[capacity_units, width]` contiguous step values, when declared.
    step: Option<StorageView>,
    /// `f32 [request_pool_size + 1, 1]`.
    timesteps: StorageView,
}

fn allocate(device: Option<DeviceId>, bytes: usize) -> Result<Arc<dyn Backing>> {
    Ok(match device {
        Some(device) => Memory::device(device, bytes.max(1))?,
        None => Memory::host(bytes.max(1)),
    })
}

/// Storage of the page banks. A latent export aliases the pages its
/// trajectory occupies, so device banks are one exportable allocation: a
/// CUDA VMM export names the pages where they lie, whatever the pool's size.
/// Every local peer device may address them, because readers on those
/// devices in this process copy from the pages directly.
fn allocate_banks(device: Option<DeviceId>, bytes: usize) -> Result<Arc<dyn Backing>> {
    Ok(match device {
        Some(device) => Memory::exportable(device, bytes.max(1), &Memory::peers(device)?)?,
        None => Memory::host(bytes.max(1)),
    })
}

impl LatentStorage {
    /// Allocate a pool's storage with zeroed banks (reset on `stream`, which
    /// is synchronized before returning).
    #[expect(clippy::too_many_arguments, reason = "the pool's complete geometry")]
    pub fn allocate(
        request_pool_size: u32,
        num_pages: u32,
        page_units: u32,
        width: u32,
        dtype: DType,
        with_step: bool,
        device: Option<DeviceId>,
        stream: Option<&Stream>,
    ) -> Result<Self> {
        if num_pages < 2 || page_units == 0 || width == 0 || request_pool_size == 0 {
            return Err(Error::Invalid(
                "latent storage needs slots, a usable page, units and elements".into(),
            ));
        }
        if !matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64) {
            return Err(Error::Invalid(
                "latent storage must use a floating dtype".into(),
            ));
        }
        if device.is_some() != stream.is_some() {
            return Err(Error::Invalid(
                "device latent storage is reset on a stream of its device".into(),
            ));
        }
        let (pages_count, units, width_i) = (
            i64::from(num_pages),
            i64::from(page_units),
            i64::from(width),
        );
        let page_bytes = (2 * pages_count * units * width_i) as usize * dtype.bytes();
        let banks = allocate_banks(device, page_bytes)?;
        let pages = StorageView::contiguous(
            Arc::clone(&banks),
            0,
            dtype,
            &[2, pages_count, units, width_i],
        )?;
        let capacity_units = (pages_count - 1) * units;
        let step = with_step
            .then(|| {
                let memory = allocate(device, (capacity_units * width_i) as usize * dtype.bytes())?;
                StorageView::contiguous(memory, 0, dtype, &[capacity_units, width_i])
            })
            .transpose()?;
        let timesteps = StorageView::contiguous(
            allocate(device, (request_pool_size as usize + 1) * 4)?,
            0,
            DType::F32,
            &[i64::from(request_pool_size) + 1, 1],
        )?;

        match stream {
            Some(stream) => {
                // SAFETY: the banks span `page_bytes`; the stream drains below.
                unsafe { memset_d8(banks.address(), 0, page_bytes, stream) }?;
                stream.synchronize()?;
            }
            None => {
                // Fresh host memory is already zero.
            }
        }
        Ok(Self {
            num_pages,
            page_units,
            width,
            request_pool_size,
            pages,
            step,
            timesteps,
        })
    }

    /// The element type of latent values.
    pub(crate) fn dtype(&self) -> DType {
        self.pages.dtype()
    }

    /// One page of `bank` as `[page_units, width]` units.
    pub(crate) fn page(&self, bank: u8, page: u32) -> Result<StorageView> {
        if bank > 1 || page == 0 || page >= self.num_pages {
            return Err(Error::Invalid("latent page is outside the pool".into()));
        }
        self.pages
            .select(0, i64::from(bank))?
            .select(0, i64::from(page))
    }

    /// Values per latent unit.
    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    /// The per-slot timesteps `[request_pool_size + 1, 1]` float32.
    pub(crate) fn timesteps(&self) -> &StorageView {
        &self.timesteps
    }

    /// The contiguous step workspace `[capacity_units, width]`, when
    /// declared.
    pub(crate) fn step(&self) -> Option<&StorageView> {
        self.step.as_ref()
    }

    /// A bank's consecutive pages as `[pages * page_units, width]` units.
    pub(crate) fn bank_view(&self, bank: u8, first: u32, count: u32) -> Result<StorageView> {
        if bank > 1 || first == 0 || count == 0 || first + count > self.num_pages {
            return Err(Error::Invalid(
                "latent bank view requires consecutive pool pages".into(),
            ));
        }
        self.pages
            .select(0, i64::from(bank))?
            .narrow(0, i64::from(first), i64::from(count))?
            .reshape(
                self.pages.dtype(),
                &[
                    i64::from(count) * i64::from(self.page_units),
                    i64::from(self.width),
                ],
            )
    }

    /// The first `units` units stored in `pages` of `bank`, page by page.
    pub fn spans(&self, bank: u8, pages: &[u32], units: u32) -> Result<Spans> {
        if bank > 1 || u64::from(units) > pages.len() as u64 * u64::from(self.page_units) {
            return Err(Error::Invalid("latent spans exceed their pages".into()));
        }
        let bank = self.pages.select(0, i64::from(bank))?;
        let views = pages
            .iter()
            .enumerate()
            .filter_map(|(index, &page)| {
                let taken = index as u32 * self.page_units;
                (taken < units).then(|| {
                    if page == 0 || page >= self.num_pages {
                        return Err(Error::Invalid("latent page is outside the pool".into()));
                    }
                    bank.select(0, i64::from(page))?.narrow(
                        0,
                        0,
                        i64::from((units - taken).min(self.page_units)),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Spans::new(views)
    }

    /// Set slot `slot`'s timestep to `value`, in the order of what `issue`
    /// issues.
    pub(crate) fn fill_timestep(
        &self,
        slot: u32,
        value: f32,
        issue: impl Issue,
    ) -> Result<StorageView> {
        if slot == 0 || slot > self.request_pool_size {
            return Err(Error::Invalid("latent slot is outside the pool".into()));
        }
        let row = self.timesteps.select(0, i64::from(slot))?;
        require_issuer(row.device(), issue)?;

        // SAFETY: one aligned word of the timestep column, which this value
        // owns.
        unsafe {
            issue.fill(Fill {
                address: row.address(),
                value: value.to_bits(),
                width: 4,
                count: 1,
            })
        }?;
        Ok(row)
    }
}

/// Latent storage on a device is written on a stream of that device, host
/// storage without one.
fn require_issuer(device: MemoryKind, issue: impl Issue) -> Result<()> {
    if matches!(device, MemoryKind::Cuda(_)) == issue.stream().is_some() {
        Ok(())
    } else {
        Err(Error::Invalid(
            "latent writes use a stream exactly for device storage".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_pages_spans_and_timesteps_follow_the_pool_layout() -> Result<()> {
        let storage = LatentStorage::allocate(2, 4, 2, 3, DType::F32, true, None, None)?;

        let spans = storage.spans(1, &[3, 1], 3)?;
        assert_eq!(spans.shape(), vec![3, 3]);
        assert_eq!(spans.views()[1].shape(), &[1, 3]);
        assert!(storage.spans(0, &[0], 1).is_err());
        assert_eq!(storage.bank_view(0, 1, 2)?.shape(), &[4, 3]);
        assert!(storage.bank_view(0, 3, 2).is_err());

        let row = storage.fill_timestep(2, 0.5, None)?;
        // SAFETY: host column of the test.
        assert_eq!(unsafe { *(row.address() as *const f32) }, 0.5);
        Ok(())
    }
}
