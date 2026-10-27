//! Device storage of the latent pool: its two page banks.
//!
//! [`LatentPool`](super::LatentPool) decides page ownership and trajectory
//! visibility; this owner holds the banks the numerical denoising steps
//! bind once, which trajectory transfers export from and import into.
//! Page 0 is padding and request slots are 1-based.

use std::sync::Arc;

use cuda::{DeviceId, Stream, memset_d8};

use crate::ffi::{Backing, DType};
use crate::memory::allocation::{Storage, StorageView};
use crate::memory::retained::{ExclusiveMemory, ExclusiveSpec};
use crate::transfer::Spans;
use crate::{Error, Result};

/// The dimensions of one latent pool, from which capacity planning sums its
/// storage ([`LatentPlan::bytes`]) and [`LatentStorage::allocate`] allocates
/// it: the one statement of the pool's tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatentPlan {
    /// One-based request slots.
    pub request_pool_size: usize,
    /// Pages including the reserved sentinel page.
    pub num_pages: usize,
    pub page_units: usize,
    /// Values per latent unit.
    pub latent_width: usize,
    /// The element type of latent values.
    pub dtype: DType,
}

impl LatentPlan {
    fn validate(&self) -> Result<()> {
        if self.num_pages < 2
            || self.page_units == 0
            || self.latent_width == 0
            || self.request_pool_size == 0
        {
            return Err(Error::Invalid(
                "latent storage needs slots, a usable page, units and elements".into(),
            ));
        }
        if !matches!(
            self.dtype,
            DType::F16 | DType::BF16 | DType::F32 | DType::F64
        ) {
            return Err(Error::Invalid(
                "latent storage must use a floating dtype".into(),
            ));
        }
        Ok(())
    }

    /// The two page banks, which alternate across denoising steps:
    /// `[2, num_pages, page_units, width]`.
    fn bank_shape(&self) -> [i64; 4] {
        [
            2,
            self.num_pages as i64,
            self.page_units as i64,
            self.latent_width as i64,
        ]
    }

    /// The bytes [`LatentStorage::allocate`] allocates for this pool.
    pub fn bytes(&self) -> Result<u64> {
        self.validate()?;
        let elements = |shape: &[i64]| shape.iter().product::<i64>() as u64;
        let value = self.dtype.bytes() as u64;
        Ok(elements(&self.bank_shape()) * value)
    }
}

/// The latent pool's storage on one device (or host memory for computations on
/// the host).
pub struct LatentStorage {
    num_pages: u32,
    page_units: u32,
    width: u32,
    /// `[2, num_pages, page_units, width]`: two banks that alternate across
    /// denoising steps.
    pages: StorageView,
}

/// Storage of the page banks, named `name` among the worker's exclusive
/// storages. A latent export aliases the pages its trajectory occupies, so
/// device banks are one exclusive storage `memory` provides, exported in
/// place: a CUDA VMM export names the pages where they lie, whatever the
/// pool's size. Every local peer device may address them, because readers
/// on those devices in this process copy from the pages directly. Returns
/// the banks and whether their memory is new.
fn allocate_banks(
    device: Option<DeviceId>,
    bytes: usize,
    name: &str,
    memory: &mut dyn ExclusiveMemory,
) -> Result<(Arc<dyn Backing>, bool)> {
    Ok(match device {
        Some(device) => {
            let peers = Storage::peers(device)?;
            let taken = memory
                .exclusive(&[ExclusiveSpec {
                    name,
                    device,
                    bytes: bytes.max(1),
                    access: &peers,
                }])?
                .pop()
                .ok_or_else(|| Error::Invariant("the latent banks' storage is missing".into()))?;
            (taken.storage, taken.created)
        }
        None => (Storage::host(bytes.max(1)), true),
    })
}

impl LatentStorage {
    /// Allocate the storage `plan` states. New banks are zeroed (reset on
    /// `stream`, which is synchronized before returning); banks a
    /// predecessor left keep their contents. Device banks are the exclusive
    /// storage `name` of `memory`.
    pub fn allocate(
        plan: &LatentPlan,
        device: Option<DeviceId>,
        stream: Option<&Stream>,
        name: &str,
        memory: &mut dyn ExclusiveMemory,
    ) -> Result<Self> {
        plan.validate()?;
        if device.is_some() != stream.is_some() {
            return Err(Error::Invalid(
                "device latent storage is reset on a stream of its device".into(),
            ));
        }
        let narrow = |value: usize, name: &str| {
            u32::try_from(value).map_err(|_| Error::Invalid(format!("latent {name} overflow")))
        };
        let dtype = plan.dtype;
        let bytes =
            |shape: &[i64], dtype: DType| shape.iter().product::<i64>() as usize * dtype.bytes();
        let bank_shape = plan.bank_shape();
        let page_bytes = bytes(&bank_shape, dtype);
        let (banks, created) = allocate_banks(device, page_bytes, name, memory)?;
        let pages = StorageView::contiguous(Arc::clone(&banks), 0, dtype, &bank_shape)?;
        // New banks are zeroed. Banks a predecessor left keep their finite
        // contents: a standby worker's predecessor still serves from them.
        match stream {
            Some(stream) if created => {
                // SAFETY: the banks span `page_bytes`; the stream drains below.
                unsafe { memset_d8(banks.address(), 0, page_bytes, stream) }?;
                stream.synchronize()?;
            }
            _ => {
                // Fresh host memory is already zero.
            }
        }
        Ok(Self {
            num_pages: narrow(plan.num_pages, "pages")?,
            page_units: narrow(plan.page_units, "page units")?,
            width: narrow(plan.latent_width, "width")?,
            pages,
        })
    }

    /// The element type of latent values.
    pub(crate) fn dtype(&self) -> DType {
        self.pages.dtype()
    }

    /// Values per latent unit.
    pub(crate) fn width(&self) -> u32 {
        self.width
    }

    /// Every page of `bank`, `[pages, page_units, width]`: the executor's
    /// view of a denoiser's bank, which it reads a request's pages from and
    /// writes them to.
    pub(crate) fn bank(&self, bank: u8) -> Result<StorageView> {
        if bank > 1 {
            return Err(Error::Invalid("a latent pool has two banks".into()));
        }
        self.pages.select(0, i64::from(bank))
    }

    /// The address of page `page` of `bank`, whose
    /// `page_units * width` values lie contiguously from it.
    pub(crate) fn page_address(&self, bank: u8, page: u64) -> Result<u64> {
        if bank > 1 || page == 0 || page >= u64::from(self.num_pages) {
            return Err(Error::Invalid("latent page is outside the pool".into()));
        }
        let page_bytes =
            u64::from(self.page_units) * u64::from(self.width) * self.dtype().bytes() as u64;
        let index = u64::from(bank) * u64::from(self.num_pages) + page;
        Ok(self.pages.address() + index * page_bytes)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_pages_and_spans_follow_the_pool_layout() -> Result<()> {
        let plan = LatentPlan {
            request_pool_size: 2,
            num_pages: 4,
            page_units: 2,
            latent_width: 3,
            dtype: DType::F32,
        };
        let storage = LatentStorage::allocate(
            &plan,
            None,
            None,
            "latent_banks",
            &mut crate::memory::LocalMemory,
        )?;
        // Capacity planning charges exactly what the pool allocates.
        let bytes = |view: &StorageView| view.numel() * view.dtype().bytes() as u64;
        assert_eq!(plan.bytes()?, bytes(&storage.pages));

        let spans = storage.spans(1, &[3, 1], 3)?;
        assert_eq!(spans.shape(), vec![3, 3]);
        assert_eq!(spans.views()[1].shape(), &[1, 3]);
        assert!(storage.spans(0, &[0], 1).is_err());
        assert_eq!(storage.bank(1)?.shape(), &[4, 2, 3]);
        assert!(storage.bank(2).is_err());
        // A page's address is where its bank's view holds it, which a
        // trajectory offload copies from.
        assert_eq!(
            storage.page_address(1, 3)?,
            storage.bank(1)?.select(0, 3)?.address()
        );
        assert!(storage.page_address(0, 0).is_err());
        assert!(storage.page_address(1, 4).is_err());
        Ok(())
    }
}
