//! The host latent region of a denoising rank (a video model's, or an image model's): where a
//! preempted request's trajectory is saved and where it is restored from when the request
//! resumes.
//!
//! A trajectory is its slot's row of the rank's trajectory rows ([`BufferMemory::trajectory`]):
//! a video model's float32 values, or an image model's latent rows, each of the row's width and
//! element type, in the order the row holds them. A `Save` copies the row into a block of the
//! trajectory class, cached under the trajectory's key (`storage::key::latent_content`), and a
//! `Restore` copies the block back into a slot's row ([`BufferMemory::copy`], or
//! [`BufferMemory::restore`] for a block saved in another encoding).
//!
//! The region is one unit space with one part per rank index that maps one, which the Storage
//! Server creates on the NUMA node closest to the rank's device
//! (`storage::protocol::Request::HostPart`). A unit of a part is one page of a trajectory row
//! ([`BankPages`]), so a block of `n` units holds the first `n` pages of the row. Every rank of a denoising group holds the request's
//! whole samples in its own row, so a saved trajectory has one host copy: the group's first rank
//! copies its row to the part it copies with ([`BufferMemory::part_holder`]), every rank of the
//! restoring group copies it back, and the scheduler sizes the block by the trajectory's pages. A
//! media deployment maps one part of the region, on the first rank of its process, which every
//! group and every epoch of the deployment that names the region copies with, whatever its
//! parallel configuration, so an epoch that succeeds another finds the trajectories its
//! predecessor saved; one that lays them out otherwise restores them through the copy manifest of
//! the two layouts ([`super::restore`]). The region is sized for the trajectories the groups can
//! have saved at once ([`pool_pages`]). The scheduler decides what is saved and restored and keys
//! it; the rank only copies, on a stream of its device.

use tensor::DType;

use crate::memory::buffers::BufferMemory;
use crate::memory::client::StorageClient;
use crate::memory::host_pages;
use crate::{Error, Result};

/// The version of the page encoding above: pages of another version are another layout.
const PAGE_ENCODING: u32 = 1;

/// The units of the rank's part: every trajectory the rank can have saved at once, twice the
/// trajectory pages of its rows.
///
/// - One preemption (a park, a sleep, a weight update, a shrink of the KV cache) saves every
///   trajectory the rank's rows hold: at most `pages.pages` pages, a video model's request slots'
///   pages or the pages an image model's scheduler assigns its trajectories.
/// - A saved trajectory is needed until the request's latent preparation restores it. A video
///   request that resumes one is admitted before every request that starts one, so the slots take
///   the resumed requests first, and the trajectories an epoch still has to restore are at most
///   those of its latest preemption.
/// - An epoch on trial saves its own trajectories while the epoch it succeeds keeps the ones it
///   saved, which a rollback restores: as many again.
///
/// A trajectory the region evicted beyond that, as when requests reach the engine in another
/// order, is a miss: its request starts from noise.
///
/// These are the units one denoising group's trajectories take, `pages` a rank's of it: the
/// group's first rank fills them in the part it copies with. Two groups save into units apart,
/// so the region's numbering is the sum of this over the groups that save there.
pub fn pool_pages(pages: BankPages) -> u64 {
    host_pages::resuming_pages(pages.pages)
}

/// The pages a rank holds trajectories in.
#[derive(Debug, Clone, Copy)]
pub struct BankPages {
    /// The element type of the values.
    pub dtype: DType,
    /// The values one page holds.
    pub page_values: u32,
    /// The most pages one trajectory occupies.
    pub trajectory_pages: u32,
    /// The pages that hold trajectories at once: the trajectory pages of every trajectory the rank
    /// holds.
    pub pages: u64,
}

impl BankPages {
    /// The bytes of one page: the bytes of every unit of the rank's part of the host latent
    /// region.
    ///
    /// # Errors
    ///
    /// An element type narrower than a byte, or a page whose bytes overflow.
    pub fn page_bytes(&self) -> Result<u64> {
        ops::abi::bytes(self.dtype)
            .and_then(|bytes| bytes.checked_mul(u64::from(self.page_values)))
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "a latent page of {} values of {} has no whole bytes",
                    self.page_values,
                    ops::abi::name(self.dtype)
                ))
            })
    }
}

/// The digest of the byte layout of a part's pages: the page encoding, the values' element type
/// and the values of a page.
fn layout(pages: &BankPages) -> [u8; 32] {
    let mut digest = ::storage::key::Digest::new(b"uniserve-latent-page-layout");
    digest.number(u64::from(PAGE_ENCODING));
    digest.text(ops::abi::name(pages.dtype));
    digest.number(u64::from(pages.page_values));
    digest.finish()
}

/// The digest of what determines the rank's share of every request's samples: the byte layout of
/// its pages, the most pages a trajectory occupies and `region`, the statement of the rank's share
/// (a video worker's statement of its layout and the rank's share of the denoiser). Two ranks of
/// one share lay every trajectory out alike, so a block one saved is a block the other restores
/// byte for byte.
pub fn share(pages: &BankPages, region: &str) -> [u8; 32] {
    let mut digest = ::storage::key::Digest::new(b"uniserve-latent-region");
    digest.bytes(&layout(pages));
    digest.number(u64::from(pages.trajectory_pages));
    digest.text(region);
    digest.finish()
}

/// Maps the rank's part of the host latent region `region`, its number and its name, into
/// `memory`: one page of `pages` of every unit, on the host NUMA node closest to the memory's
/// device, so the copies between them cross no inter-node link. Returns the part's units.
///
/// # Errors
///
/// A page without whole bytes; as [`BufferMemory::map_part`].
pub fn map_part(
    memory: &mut BufferMemory,
    client: &StorageClient,
    region: (u8, &str),
    pages: &BankPages,
) -> Result<u32> {
    let numa_node = memory.device().and_then(host_pages::numa_node);

    memory.map_part(client, region, pages.page_bytes()?, numa_node)
}
