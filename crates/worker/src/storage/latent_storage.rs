//! Device storage of the latent pool: page banks, step workspace, page
//! table and per-slot timesteps.
//!
//! [`LatentPool`](super::LatentPool) decides page ownership and trajectory
//! visibility; this owner holds the memory the numerical denoising steps
//! bind once and the native writes the pool's decisions need (page-table
//! uploads, timestep fills). Page 0 is padding and request slots are 1-based.

use std::sync::Arc;

use uniserve_cuda::{CopyBatch, SourceOrder, Stream, memset_d8, memset_d32};

use super::memory::{Memory, StorageView};
use crate::ffi::{Backing, DType, Device};
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
    /// `i64 [num_pages - 1]`: the pages of the batch's bound step values.
    page_table: StorageView,
    /// `f32 [request_pool_size + 1, 1]`.
    timesteps: StorageView,
}

fn allocate(device: Option<i32>, bytes: usize) -> Result<Arc<dyn Backing>> {
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
fn allocate_banks(device: Option<i32>, bytes: usize) -> Result<Arc<dyn Backing>> {
    Ok(match device {
        Some(device) => Memory::exportable(device, bytes.max(1), &Memory::peers(device)?)?,
        None => Memory::host(bytes.max(1)),
    })
}

impl LatentStorage {
    /// Device bytes of a pool's storage.
    pub fn bytes(
        request_pool_size: u32,
        num_pages: u32,
        page_units: u32,
        width: u32,
        dtype: DType,
        with_step: bool,
    ) -> u64 {
        let element = dtype.bytes() as u64;
        let units = u64::from(num_pages) * u64::from(page_units) * u64::from(width);
        let step = if with_step {
            u64::from(num_pages.saturating_sub(1)) * u64::from(page_units) * u64::from(width)
        } else {
            0
        };
        (2 * units + step) * element
            + u64::from(num_pages.saturating_sub(1)) * 8
            + (u64::from(request_pool_size) + 1) * 4
    }

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
        device: Option<i32>,
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
        let page_table = StorageView::contiguous(
            allocate(device, (pages_count - 1) as usize * 8)?,
            0,
            DType::I64,
            &[pages_count - 1],
        )?;
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
            page_table,
            timesteps,
        })
    }

    pub fn num_pages(&self) -> u32 {
        self.num_pages
    }

    /// The element type of latent values.
    pub fn dtype(&self) -> DType {
        self.pages.dtype()
    }

    pub fn page_units(&self) -> u32 {
        self.page_units
    }

    /// The bound storage keyed as numerical bindings receive it.
    pub fn views(&self) -> Vec<(String, StorageView)> {
        let mut views = vec![
            ("pages".to_owned(), self.pages.clone()),
            ("page_table".to_owned(), self.page_table.clone()),
            ("timesteps".to_owned(), self.timesteps.clone()),
        ];
        if let Some(step) = &self.step {
            views.push(("step".to_owned(), step.clone()));
        }
        views
    }

    /// One page of `bank` as `[page_units, width]` units.
    pub fn page(&self, bank: u8, page: u32) -> Result<StorageView> {
        if bank > 1 || page == 0 || page >= self.num_pages {
            return Err(Error::Invalid("latent page is outside the pool".into()));
        }
        self.pages
            .select(0, i64::from(bank))?
            .select(0, i64::from(page))
    }

    /// Values per latent unit.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The per-slot timesteps `[request_pool_size + 1, 1]` float32.
    pub fn timesteps(&self) -> &StorageView {
        &self.timesteps
    }

    /// The contiguous step workspace `[capacity_units, width]`, when
    /// declared.
    pub fn step(&self) -> Option<&StorageView> {
        self.step.as_ref()
    }

    /// Both banks' pages flattened to rows: `[2 * num_pages, page_units * width]`.
    pub fn page_rows(&self) -> Result<StorageView> {
        self.pages.reshape(
            self.pages.dtype(),
            &[
                2 * i64::from(self.num_pages),
                i64::from(self.page_units) * i64::from(self.width),
            ],
        )
    }

    /// A bank's consecutive pages as `[pages * page_units, width]` units.
    pub fn bank_view(&self, bank: u8, first: u32, count: u32) -> Result<StorageView> {
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

    /// The step-workspace range of `count` pages starting at page-table
    /// offset `offset`, as `[count * page_units, width]`.
    pub fn step_view(&self, offset: u32, count: u32) -> Result<StorageView> {
        let step = self
            .step
            .as_ref()
            .ok_or_else(|| Error::Invalid("latent storage has no step workspace".into()))?;
        let units = i64::from(self.page_units);
        step.narrow(0, i64::from(offset) * units, i64::from(count) * units)
    }

    /// Write `pages` into the page table at `offset` in `stream` order. The
    /// ids are read during the call, so the host never waits on the device.
    pub fn upload_pages(&self, offset: u32, pages: &[u32], stream: Option<&Stream>) -> Result<()> {
        if u64::from(offset) + pages.len() as u64 > u64::from(self.num_pages - 1)
            || pages
                .iter()
                .any(|&page| page == 0 || page >= self.num_pages)
        {
            return Err(Error::Invalid(
                "latent page table update exceeds its pool".into(),
            ));
        }
        let ids: Vec<i64> = pages.iter().map(|&page| i64::from(page)).collect();
        let destination = self.page_table.address() + u64::from(offset) * 8;
        match (self.page_table.device(), stream) {
            (Device::Cuda(_), Some(stream)) => {
                let mut batch = CopyBatch::new(SourceOrder::DuringCall);
                batch.push(destination, ids.as_ptr() as u64, ids.len() * 8);
                // SAFETY: the destination lies in the page table; the
                // pageable source is consumed during the call.
                unsafe { batch.submit(stream) }.map_err(Error::from)
            }
            (Device::Cpu, None) => {
                // SAFETY: host table owned by this value; the caller orders
                // its readers.
                unsafe {
                    std::ptr::copy_nonoverlapping(ids.as_ptr(), destination as *mut i64, ids.len());
                }
                Ok(())
            }
            _ => Err(Error::Invalid(
                "latent writes use a stream exactly for device storage".into(),
            )),
        }
    }

    /// Set slot `slot`'s timestep to `value` on `stream`.
    pub fn fill_timestep(
        &self,
        slot: u32,
        value: f32,
        stream: Option<&Stream>,
    ) -> Result<StorageView> {
        if slot == 0 || slot > self.request_pool_size {
            return Err(Error::Invalid("latent slot is outside the pool".into()));
        }
        let row = self.timesteps.select(0, i64::from(slot))?;
        match (row.device(), stream) {
            // SAFETY: one aligned word of the timestep column.
            (Device::Cuda(_), Some(stream)) => {
                unsafe { memset_d32(row.address(), value.to_bits(), 1, stream) }?
            }
            (Device::Cpu, None) => {
                // SAFETY: host column owned by this value.
                unsafe { (row.address() as *mut f32).write(value) };
            }
            _ => {
                return Err(Error::Invalid(
                    "latent writes use a stream exactly for device storage".into(),
                ));
            }
        }
        Ok(row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_pages_spans_and_tables_follow_the_pool_layout() -> Result<()> {
        let storage = LatentStorage::allocate(2, 4, 2, 3, DType::F32, true, None, None)?;
        assert_eq!(
            LatentStorage::bytes(2, 4, 2, 3, DType::F32, true),
            (2 * 4 * 2 * 3 + 3 * 2 * 3) * 4 + 3 * 8 + 3 * 4
        );

        let spans = storage.spans(1, &[3, 1], 3)?;
        assert_eq!(spans.shape(), vec![3, 3]);
        assert_eq!(spans.views()[1].shape(), &[1, 3]);
        assert!(storage.spans(0, &[0], 1).is_err());
        assert_eq!(storage.bank_view(0, 1, 2)?.shape(), &[4, 3]);
        assert!(storage.bank_view(0, 3, 2).is_err());

        storage.upload_pages(1, &[2, 3], None)?;
        // SAFETY: host table of the test.
        let table =
            unsafe { std::slice::from_raw_parts(storage.page_table.address() as *const i64, 3) };
        assert_eq!(&table[1..], &[2, 3]);
        assert!(storage.upload_pages(2, &[1, 2], None).is_err());

        let row = storage.fill_timestep(2, 0.5, None)?;
        // SAFETY: host column of the test.
        assert_eq!(unsafe { *(row.address() as *const f32) }, 0.5);
        Ok(())
    }
}
