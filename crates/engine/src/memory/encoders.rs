//! The host encoder region: where the encoder outputs a rank computes (a video prompt's text
//! conditioning, its vision features, its condition latents; an image's vision and latent
//! features) are kept on the host, each a block of the encoder-output class cached under the key
//! of what the encoder read and what computes it (`storage::key::encoder_output`).
//!
//! The region is the deployment's on the rank's host: the Storage Server creates one part per
//! rank index (`storage::protocol::Request::HostPart`), which every epoch of the deployment that
//! names the region maps, so an output one epoch stored is found by another that computes the
//! same, and a resumed request's encoders read their outputs instead of computing them again. The
//! scheduler decides what is stored and loaded and keys it; the rank only copies, between its
//! product buffer and the part it copies with ([`BufferMemory::copy`],
//! [`BufferMemory::part_holder`]).
//!
//! An output lies whole in one rank's product buffer, and its host copy in one part: the copy
//! takes the consecutive units whose part holds the output's bytes, [`PAGE_BYTES`] each, rows in
//! order, since the buffer pool sizes it by the bytes of one part's unit
//! (`storage::runtime::Region::part_bytes`), and those units of any other part hold nothing of
//! it. A media deployment maps one part of the region, on the first rank of its process, which
//! every rank of it copies with, so an output one group stored is where another group loads it.
//! Those bytes are the output's rows in order under every parallel configuration, so the class
//! of encoder outputs states one encoding everywhere (`storage::runtime::Class::encoding`), and
//! a `Load` copies the bytes as they are: unlike a trajectory's, an output's restore needs no
//! copy manifest.

use crate::memory::buffers::BufferMemory;
use crate::memory::client::StorageClient;
use crate::worker::Result;

/// The version of the unit encoding: an output's bytes, rows in order, in consecutive units.
const PAGE_ENCODING: u32 = 1;

/// The bytes of one unit of a rank's part. A stored output's last unit holds the remainder of its
/// bytes, so the region's units exceed the outputs' bytes by less than a unit each.
pub(crate) const PAGE_BYTES: u64 = 256 << 10;

/// The units the region holds: the encoder outputs of every request whose outputs can be needed
/// at once, `largest` holding each output of a request at its largest bytes (a video request's
/// outputs; the vision and latent features of one image of an image request).
///
/// A request's outputs are needed while it is in flight, to be read again when it resumes. An
/// epoch runs at most `request_slots` requests at once, and as many preempted ones wait to resume
/// (a video request that resumes is admitted before every request that starts): twice the request
/// slots' outputs. An image request reads and generates images mostly smaller than the largest,
/// whose features take fewer units. An output evicted beyond that is a miss, and its encoder
/// computes it again.
///
/// These are the units one worker group's outputs take, `request_slots` its slots. An output
/// takes its units in every part and fills those of one part alone, so the outputs of two
/// groups take units apart: the region's numbering is the sum of this over the groups that
/// store outputs there.
pub(crate) fn pool_pages(largest: &[u64], request_slots: usize) -> u64 {
    let request: u64 = largest.iter().map(|bytes| bytes.div_ceil(PAGE_BYTES)).sum();
    crate::memory::host_pages::resuming_pages(request_slots as u64 * request)
}

/// The digest of the byte layout of the region's units: its domain, the unit encoding's version
/// and the unit's bytes. Two epochs whose regions have one layout hold outputs alike, so the
/// region's name carries it and an encoding of another version names another region.
pub(crate) fn layout() -> [u8; 32] {
    let mut key = [0; 32];
    key[..16].copy_from_slice(b"encoder-pages\0\0\0");
    key[16..20].copy_from_slice(&PAGE_ENCODING.to_le_bytes());
    key[20..28].copy_from_slice(&PAGE_BYTES.to_le_bytes());
    key
}

/// Maps the rank's part of the host encoder region `region`, its number and its name, into
/// `memory`, [`PAGE_BYTES`] of every unit. Returns the part's units.
///
/// # Errors
///
/// As [`BufferMemory::map_part`].
pub(crate) fn map_part(
    memory: &mut BufferMemory,
    client: &StorageClient,
    region: (u8, &str),
) -> Result<u32> {
    // Ranks of one index share the part across epochs whose devices may lie on other NUMA
    // nodes, so it names none.
    memory.map_part(client, region, PAGE_BYTES, None)
}

/// One copy between a call's output and a unit of its stored output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(test), expect(dead_code, reason = "only tests call it"))]
pub(crate) struct PageCopy {
    /// The byte offset in the call's output.
    pub output: u64,
    /// The unit's index in the stored output's run.
    pub page: usize,
    /// The byte offset in the unit.
    pub offset: u64,
    /// The bytes copied.
    pub bytes: u64,
}

/// The copies of a call's output rows `rows` (call-local, end exclusive) whose output holds the
/// stored output's rows from `first` on, rows of `row_bytes` bytes, in units of `page_bytes`: the
/// rows' bytes split at the unit boundaries they cross.
#[cfg_attr(not(test), expect(dead_code, reason = "only tests call it"))]
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

    /// A round's rows of a stored output split at every unit boundary they cross, each piece at
    /// its place in the call's output and in its unit.
    #[test]
    fn a_rounds_rows_split_at_the_page_boundaries() {
        // Rows of 3 bytes from stored row 2 on, units of 8 bytes: stored bytes 6..15 cross the
        // boundary at 8.
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
