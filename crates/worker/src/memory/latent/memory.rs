//! Device storage of the latent pool: page banks, step workspace and
//! per-slot timesteps.
//!
//! [`LatentPool`](super::LatentPool) decides page ownership and trajectory
//! visibility; this owner holds the memory the numerical denoising steps
//! bind once and the native writes the pool's decisions need (timestep
//! fills). Page 0 is padding and request slots are 1-based.

use std::sync::Arc;

use cuda::{DeviceId, Stream, memset_d8};

use crate::ffi::{Backing, DType, MemoryKind};
use crate::memory::allocation::{Storage, StorageView};
use crate::memory::retained::{ExclusiveMemory, ExclusiveSpec};
use crate::stream::{Fill, Issue};
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
    /// A step workspace beside the two banks (image denoising).
    pub with_workspace: bool,
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

    /// The contiguous step workspace of every usable page's units,
    /// `[(num_pages - 1) * page_units, width]`, when declared.
    fn step_shape(&self) -> Option<[i64; 2]> {
        self.with_workspace.then(|| {
            [
                (self.num_pages as i64 - 1) * self.page_units as i64,
                self.latent_width as i64,
            ]
        })
    }

    /// One float32 timestep per one-based slot: `[request_pool_size + 1, 1]`.
    fn timestep_shape(&self) -> [i64; 2] {
        [self.request_pool_size as i64 + 1, 1]
    }

    /// The bytes [`LatentStorage::allocate`] allocates for this pool.
    pub fn bytes(&self) -> Result<u64> {
        self.validate()?;
        let elements = |shape: &[i64]| shape.iter().product::<i64>() as u64;
        let value = self.dtype.bytes() as u64;
        Ok(elements(&self.bank_shape()) * value
            + self
                .step_shape()
                .map_or(0, |shape| elements(&shape) * value)
            + elements(&self.timestep_shape()) * DType::F32.bytes() as u64)
    }
}

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
        Some(device) => Storage::device(device, bytes.max(1))?,
        None => Storage::host(bytes.max(1)),
    })
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
        let step = plan
            .step_shape()
            .map(|shape| {
                let memory = allocate(device, bytes(&shape, dtype))?;
                StorageView::contiguous(memory, 0, dtype, &shape)
            })
            .transpose()?;
        let timestep_shape = plan.timestep_shape();
        let timesteps = StorageView::contiguous(
            allocate(device, bytes(&timestep_shape, DType::F32))?,
            0,
            DType::F32,
            &timestep_shape,
        )?;

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
            request_pool_size: narrow(plan.request_pool_size, "slots")?,
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

    /// Every page of `bank`, `[pages, page_units, width]`: the executor's
    /// view of a video denoiser's bank, which it gathers a request's pages
    /// from and scatters them to.
    pub(crate) fn bank(&self, bank: u8) -> Result<StorageView> {
        if bank > 1 {
            return Err(Error::Invalid("a latent pool has two banks".into()));
        }
        self.pages.select(0, i64::from(bank))
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
        let plan = LatentPlan {
            request_pool_size: 2,
            num_pages: 4,
            page_units: 2,
            latent_width: 3,
            dtype: DType::F32,
            with_workspace: true,
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
        let step = storage
            .step()
            .ok_or(Error::State("the pool declares a step workspace"))?;
        assert_eq!(
            plan.bytes()?,
            bytes(&storage.pages) + bytes(step) + bytes(storage.timesteps())
        );

        let spans = storage.spans(1, &[3, 1], 3)?;
        assert_eq!(spans.shape(), vec![3, 3]);
        assert_eq!(spans.views()[1].shape(), &[1, 3]);
        assert!(storage.spans(0, &[0], 1).is_err());
        assert_eq!(storage.bank(1)?.shape(), &[4, 2, 3]);
        assert!(storage.bank(2).is_err());

        let row = storage.fill_timestep(2, 0.5, None)?;
        // SAFETY: host column of the test.
        assert_eq!(unsafe { *(row.address() as *const f32) }, 0.5);
        Ok(())
    }
}
