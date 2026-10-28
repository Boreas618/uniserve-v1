//! The host encoder pool of a rank that executes encoder calls: the Storage
//! Server page pool holding encoder outputs (a video prompt's text
//! conditioning, its vision features, its condition latents; an image's
//! vision and latent features), each keyed by what the encoder read and what
//! computes it (`ipc::HostOutput`).
//!
//! The pool is the deployment's on the rank's host: every rank there that
//! executes an encoder maps the same pool, in every world of the
//! deployment, so an output one rank or world stored is found by another
//! that computes the same, and a resumed request's encoders read their
//! outputs instead of computing them again. A stored output takes the pages
//! its bytes fill, rows in order; the engine decides what is stored and
//! keys it, and a rank only copies the rows of its calls' outputs, on the
//! stream of the encoder that writes them.

use std::sync::Arc;

use cuda::DeviceId;
use ipc::EncoderPoolInfo;
use storage::Key;
use storage::protocol::{PageShape, Reply, Request};

use crate::ffi::{Backing, DType};
use crate::memory::allocation::{Storage, StorageView};
use crate::memory::retained::Retained;
use crate::{Error, Result};

/// The version of the page encoding: an output's bytes, rows in order,
/// in consecutive pages.
const PAGE_ENCODING: u32 = 1;

/// The bytes of one page. A stored output's last page holds the remainder of
/// its bytes, so the pool's pages exceed the outputs' bytes by less than a
/// page each.
pub(crate) const PAGE_BYTES: u64 = 256 << 10;

/// The names of a video request's encoder outputs, whose largest bytes
/// size the pool of a video encoding rank.
pub(crate) const OUTPUTS: [&str; 4] = [
    "vision_features",
    "conditioning",
    "condition_video_latents",
    "condition_audio_latents",
];

/// The pages the pool holds: the encoder outputs of every request whose
/// outputs can be needed at once, `largest` holding each output of a request
/// at its largest bytes (a video request's outputs; the vision and latent
/// features of one image of an image request).
///
/// A request's outputs are needed while it is in flight, to be read again
/// when it resumes. A world runs at most `request_slots` requests at once,
/// and as many preempted ones wait to resume (a video request that resumes
/// is admitted before every request that starts, `Scheduler::queue_media`):
/// twice the request slots' outputs. An image request reads and generates
/// images mostly smaller than the largest, whose features take fewer
/// pages. An output evicted beyond that is a miss, and its encoder computes
/// it again.
pub(crate) fn pool_pages(largest: &[u64], request_slots: usize) -> u64 {
    let request: u64 = largest.iter().map(|bytes| bytes.div_ceil(PAGE_BYTES)).sum();
    2 * request_slots as u64 * request
}

/// The pool layout identifier: its domain, encoding version and page size.
fn layout() -> [u8; 32] {
    let mut key = [0; 32];
    key[..16].copy_from_slice(b"encoder-pages\0\0\0");
    key[16..20].copy_from_slice(&PAGE_ENCODING.to_le_bytes());
    key[20..28].copy_from_slice(&PAGE_BYTES.to_le_bytes());
    key
}

/// The rank's view of its host's encoder pool, mapped and registered for
/// device copies.
pub(crate) struct EncoderHostPool {
    pool: u64,
    storage: Arc<Storage>,
    slots: u64,
    page_bytes: u64,
}

impl EncoderHostPool {
    /// Map the host's encoder pool at its Storage Server, registered with
    /// `device`, holding `pages` pages when the rank creates it.
    /// `identities` names what computes each encoder output the rank writes
    /// (the weights' values and the modules' identities), and `endpoint` is
    /// where an engine on another host reaches the server. Returns the pool
    /// and what the rank reports of it.
    pub(crate) fn start(
        retained: &Retained,
        device: DeviceId,
        pages: u64,
        identities: std::collections::BTreeMap<String, String>,
        endpoint: Option<&str>,
    ) -> Result<(Self, EncoderPoolInfo)> {
        let shape = PageShape {
            layout: Key(layout()),
            kv_head_offset: 0,
            kv_head_count: 0,
            page_bytes: PAGE_BYTES,
        };
        // The host's ranks share the pool across their NUMA nodes, so it
        // names none.
        let answer = retained.call("mapping the host encoder pool", |client| {
            client.call(
                &Request::PagePool {
                    shape,
                    numa_node: None,
                    bytes: Some(pages * PAGE_BYTES),
                },
                &[],
            )
        })?;
        let Reply::PagePool {
            pool,
            slots,
            page_bytes,
        } = answer.reply
        else {
            return Err(Error::Interface(format!(
                "the storage server answered the encoder pool with {:?}",
                answer.reply
            )));
        };
        let descriptor = answer.descriptors.into_iter().next().ok_or_else(|| {
            Error::Interface("the encoder pool arrived without its storage".into())
        })?;
        let storage = Storage::map_host(descriptor, (slots * page_bytes) as usize, device, true)?;
        tracing::info!(pool, slots, page_bytes, "mapped the host encoder pool");
        let info = EncoderPoolInfo {
            pool,
            page_bytes,
            identities,
            endpoint: endpoint.map(str::to_owned),
        };
        Ok((
            Self {
                pool,
                storage,
                slots,
                page_bytes,
            },
            info,
        ))
    }

    /// The pool's identity at its Storage Server.
    pub(crate) fn pool(&self) -> u64 {
        self.pool
    }

    /// The bytes of one page.
    pub(crate) fn page_bytes(&self) -> u64 {
        self.page_bytes
    }

    /// The `uint8 [page_bytes]` view of pool slot `slot`.
    pub(crate) fn view(&self, slot: u64) -> Result<StorageView> {
        if slot >= self.slots {
            return Err(Error::Invalid(format!(
                "encoder pool slot {slot} is not a slot of pool {} of {} slots",
                self.pool, self.slots
            )));
        }
        StorageView::contiguous(
            Arc::clone(&self.storage) as Arc<dyn Backing>,
            (slot * self.page_bytes) as usize,
            DType::U8,
            &[self.page_bytes as i64],
        )
    }
}

/// One copy between a call's output and a page of its stored output: the
/// byte offset in the output, the page's index in the stored output, the
/// byte offset in the page and the bytes copied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PageCopy {
    pub(crate) output: u64,
    pub(crate) page: usize,
    pub(crate) offset: u64,
    pub(crate) bytes: u64,
}

/// The copies of a call's output rows `rows` (call-local, end exclusive)
/// whose output holds the stored output's rows from `first` on, rows of
/// `row_bytes` bytes, in pages of `page_bytes`: the rows' bytes split at
/// the page boundaries they cross.
pub(crate) fn page_copies(
    first: u32,
    rows: std::ops::Range<u64>,
    row_bytes: u64,
    page_bytes: u64,
) -> Vec<PageCopy> {
    let base = u64::from(first) * row_bytes;
    let mut start = base + rows.start * row_bytes;
    let end = base + rows.end * row_bytes;
    let mut copies = Vec::new();
    while start < end {
        let page = start / page_bytes;
        let offset = start % page_bytes;
        let bytes = (page_bytes - offset).min(end - start);
        copies.push(PageCopy {
            output: start - base,
            page: page as usize,
            offset,
            bytes,
        });
        start += bytes;
    }
    copies
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A round's rows of a stored output split at every page boundary they
    /// cross, each piece at its place in the call's output and in its page.
    #[test]
    fn a_rounds_rows_split_at_the_page_boundaries() {
        // Rows of 3 bytes from stored row 2 on, pages of 8 bytes: stored
        // bytes 6..15 cross the boundary at 8.
        let copies = page_copies(2, 0..3, 3, 8);
        assert_eq!(
            copies,
            [
                PageCopy {
                    output: 0,
                    page: 0,
                    offset: 6,
                    bytes: 2,
                },
                PageCopy {
                    output: 2,
                    page: 1,
                    offset: 0,
                    bytes: 7,
                },
            ]
        );
        // Every byte of the rows is copied once.
        let copied: u64 = page_copies(5, 1..9, 7, 16)
            .iter()
            .map(|copy| copy.bytes)
            .sum();
        assert_eq!(copied, 8 * 7);
    }
}
