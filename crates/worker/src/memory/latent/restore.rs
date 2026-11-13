//! The copies that lay a saved trajectory's pages out in a restoring
//! rank's trajectory row.
//!
//! A denoising rank's trajectory row holds the rank's part of a video
//! request's latent samples, in pages of the units of its part of the host
//! latent region: each modality's sample is a logical tensor of rows by
//! channels, the rank holds some of its rows, and its values lie at an offset
//! of the row's pages (`uniserve.worker.video.plan.SampleLayout.storage`).
//! The worker declares this as one storage of the copy manifest library
//! (`::storage::manifest`): the row's pages up to the last one holding a
//! value, each page a chunk keyed by the part of the samples it holds and
//! where, with the trajectory's content
//! ([`::storage::manifest::Storage::paged`]).
//!
//! A rank that restores a trajectory saved by ranks of another parallel
//! configuration (another Ulysses degree, other ranks) holds other rows at
//! other offsets than any saving rank did. The buffer pool tells it so: the
//! saved block records the encoding of the class that saved it, and the
//! `Restore`'s copy names that encoding beside the restoring class's
//! (`storage::runtime::Completion::Copy`); the runner that restores the
//! request computes both layouts of the request's samples from them. Its
//! copies are the manifest from the saving ranks' pages it reads (the source
//! epoch, each rank's part of the saved block) to its own row's pages (a
//! target world of one rank): a page the rank lays out as a saving rank did
//! has that page's key and is copied whole (a mapping, which host pages
//! cannot be for device pages); every other region of the rank's samples is
//! copied from the saved page that holds it, as byte runs split at the saved
//! pages' bounds ([`crate::memory::buffers::BufferMemory::restored`]). Both
//! epochs lie on the rank's GPU, which reads the host pages directly, so the
//! manifest's one link is the GPU's to itself. The copies of a rank depend on
//! the sources and its own pages alone, so ranks compute their own without
//! agreeing on anything.
//!
//! The class of stored trajectories states its tensors and its encoding by
//! `storage::runtime::encoding` over the [`Stored`] layouts of the ranks of
//! its layout, for the largest trajectory its rows hold.

use ::storage::manifest::{self, Gpu, Manifest, StorageLayout};
use ::storage::protocol::{Link, LinkBandwidth, LinkEnd};

use crate::{Error, Result};

/// The GPU that both epochs of a restore's manifest lie on: the restoring
/// rank's, which names the manifest's one link and nothing else.
const GPU: Gpu = [0; 16];

/// The content every page of a restore's manifest holds: the one trajectory
/// both epochs lay out, so the chunk keys tell layouts apart and nothing
/// more is needed of them.
const CONTENT: [u8; 32] = [0; 32];

/// A rank's pages of one request's trajectory: how they hold its samples
/// and the bytes of one page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stored {
    /// How the pages, back to back, hold the rank's part of the samples.
    pub layout: StorageLayout,
    /// The bytes of one page.
    pub page_bytes: u64,
}

/// `bytes` bytes from byte `offset` of page `page` of saving rank `source`
/// to byte `target` of the restoring rank's row, whose pages lie back to
/// back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageCopy {
    /// The saving rank, by its index in the sources.
    pub source: usize,
    /// The saving rank's page.
    pub page: u64,
    /// The first byte read in that page.
    pub offset: u64,
    /// The first byte written in the restoring rank's row.
    pub target: u64,
    /// The bytes copied.
    pub bytes: u64,
}

/// The copies that fill `target`, the restoring rank's row, from `sources`,
/// the saving ranks' pages it reads, all of one trajectory.
///
/// Fails when a description is malformed, when the sources do not hold
/// every element the target holds, or when the target holds padding or an
/// encoding that a copy of bytes does not reproduce.
pub fn copies(sources: &[Stored], target: &Stored) -> Result<Vec<PageCopy>> {
    let invalid = |error: ::storage::Error| Error::Invalid(format!("a latent restore: {error}"));
    let gpu = GPU;
    let fingerprint = |_: &str| Some(CONTENT);
    let paged = |stored: &Stored| {
        manifest::Storage::paged(stored.layout.clone(), stored.page_bytes, &fingerprint)
            .map_err(invalid)
    };
    let rank = |stored: &Stored| -> Result<manifest::Rank> {
        Ok(manifest::Rank {
            gpu: Some(gpu),
            storages: vec![paged(stored)?],
        })
    };
    let request = manifest::Request {
        source: manifest::Epoch {
            ranks: sources.iter().map(rank).collect::<Result<_>>()?,
        },
        target: manifest::Epoch {
            ranks: vec![rank(target)?],
        },
        gpus: vec![gpu],
        // Only the rank's GPU copies; the bandwidth orders nothing.
        links: vec![LinkBandwidth {
            link: Link {
                source: LinkEnd::Device { gpu },
                destination: LinkEnd::Device { gpu },
            },
            bytes_per_second: 1,
        }],
    };
    let computed = Manifest::compute(&request).map_err(invalid)?;
    if !computed.missing.is_empty() {
        return Err(Error::Invalid(format!(
            "the stored pages do not hold every latent value of the rank: {:?}",
            computed.missing
        )));
    }
    if !(computed.stagings.is_empty() && computed.fills.is_empty()) {
        return Err(Error::Invalid(
            "a latent layout declares padding or an encoding that a byte copy does not \
             reproduce"
                .into(),
        ));
    }

    let mut copies = Vec::new();
    // A target page that a stored page lays out alike is copied whole.
    for mapping in &computed.mappings {
        let source = mapping.source_rank as usize;
        copies.push(PageCopy {
            source,
            page: u64::from(mapping.source_chunk),
            offset: 0,
            target: u64::from(mapping.chunk) * target.page_bytes,
            bytes: sources[source].page_bytes,
        });
    }
    // Every other region, as runs of bytes within one stored page each. The
    // target's pages lie back to back in the slot, so only the stored
    // pages' bounds split a run.
    for copy in &computed.copies {
        let source = copy.source.rank as usize;
        let page_bytes = sources[source].page_bytes;
        for row in 0..copy.rows {
            let mut from = copy.source.offset + row * copy.source_pitch;
            let mut to = copy.target.offset + row * copy.target_pitch;
            let mut left = copy.length;
            while left > 0 {
                let within = from % page_bytes;
                let bytes = left.min(page_bytes - within);
                copies.push(PageCopy {
                    source,
                    page: from / page_bytes,
                    offset: within,
                    target: to,
                    bytes,
                });
                from += bytes;
                to += bytes;
                left -= bytes;
            }
        }
    }
    Ok(copies)
}

/// The layouts and bytes of sequence-parallel shares of a trajectory, which the tests of the
/// rank's media memory restore too.
#[cfg(test)]
pub(crate) mod tests {
    use ::storage::manifest::{Encoding, Field, Logical, Segment, Span};

    use super::*;

    impl Stored {
        /// The pages that hold the samples: the pages up to the last one a value lies in.
        pub(crate) fn pages(&self) -> u64 {
            self.layout.bytes.div_ceil(self.page_bytes.max(1))
        }
    }

    /// Values of one page: 64 float32.
    const PAGE_BYTES: u64 = 256;

    /// Rows and channels of the video and audio samples.
    const VIDEO: (u64, u64) = (40, 4);
    const AUDIO: (u64, u64) = (12, 2);

    /// Values of an element offset's alignment in the slot's pages.
    const ALIGNMENT: u64 = 16;

    /// A rank's slot pages holding rows `video` of the video sample and
    /// rows `audio` of the audio sample, each modality's values row-major
    /// at the next aligned element offset, as the worker lays a
    /// sequence-parallel rank's share out.
    pub(crate) fn share(video: (u64, u64), audio: (u64, u64)) -> Stored {
        let mut fields = Vec::new();
        let mut cursor = 0;
        for (name, (rows, width), (start, count)) in
            [("video", VIDEO, video), ("audio", AUDIO, audio)]
        {
            fields.push(Field {
                tensor: format!("{name}_latents"),
                source: String::new(),
                dtype: "float32".into(),
                element: 4,
                offset: cursor * 4,
                shape: vec![count, width],
                strides: vec![width, 1],
                logical: Logical {
                    shape: vec![rows, width],
                    dims: vec!["row".into(), "channel".into()],
                    units: vec![1, 0],
                    segments: vec![
                        vec![Segment {
                            span: Span::range(start, count),
                            local: 0,
                        }],
                        vec![Segment {
                            span: Span::range(0, width),
                            local: 0,
                        }],
                    ],
                },
                encoding: Encoding::Affine,
            });
            cursor += (count * width).next_multiple_of(ALIGNMENT);
        }
        // The storage ends with the last value it holds.
        let bytes = fields
            .iter()
            .map(|field| field.bytes().end)
            .max()
            .unwrap_or(0);
        Stored {
            layout: StorageLayout { bytes, fields },
            page_bytes: PAGE_BYTES,
        }
    }

    /// The value every rank holds at `row`, `channel` of `tensor`.
    fn value(tensor: usize, row: u64, channel: u64) -> [u8; 4] {
        ((tensor as u32 + 1) * 10_000 + row as u32 * 10 + channel as u32).to_le_bytes()
    }

    /// The bytes of a rank's slot pages, garbage where no field lies.
    pub(crate) fn pages(stored: &Stored) -> Vec<u8> {
        let mut bytes = vec![0xee; (stored.page_bytes * stored.pages()) as usize];
        for (tensor, field) in stored.layout.fields.iter().enumerate() {
            let rows = &field.logical.segments[0][0].span;
            for local in 0..rows.count {
                for channel in 0..field.shape[1] {
                    let at = field.address(&[local, channel]) as usize;
                    bytes[at..at + 4].copy_from_slice(&value(tensor, rows.index(local), channel));
                }
            }
        }
        bytes
    }

    /// The restoring rank's slot pages after `copies` from the stored
    /// ranks' pages.
    fn restored(sources: &[Stored], target: &Stored, copies: &[PageCopy]) -> Vec<u8> {
        let images: Vec<Vec<u8>> = sources.iter().map(pages).collect();
        let mut bytes = vec![0; (target.page_bytes * target.pages()) as usize];
        for copy in copies {
            let from = (copy.page * sources[copy.source].page_bytes + copy.offset) as usize;
            let to = copy.target as usize;
            let length = copy.bytes as usize;
            bytes[to..to + length].copy_from_slice(&images[copy.source][from..from + length]);
        }
        bytes
    }

    /// Whether the restored pages hold every value the target lays out
    /// where it lays it out.
    pub(crate) fn holds_its_samples(target: &Stored, bytes: &[u8]) -> bool {
        let expected = pages(target);
        target.layout.fields.iter().all(|field| {
            (0..field.shape[0]).all(|local| {
                (0..field.shape[1]).all(|channel| {
                    let at = field.address(&[local, channel]) as usize;
                    bytes[at..at + 4] == expected[at..at + 4]
                })
            })
        })
    }

    /// Ulysses 2 to one rank: the rank takes both stored ranks' rows; the
    /// pages it lays out as the first stored rank did are copied whole.
    #[test]
    fn two_sequence_parallel_ranks_become_one() {
        let sources = [share((0, 22), (0, 5)), share((22, 18), (5, 7))];
        let target = share((0, 40), (0, 12));
        let copies = copies(&sources, &target).unwrap();
        assert!(holds_its_samples(
            &target,
            &restored(&sources, &target, &copies)
        ));
        // The first video rows fill the first page alike under both.
        assert!(copies.contains(&PageCopy {
            source: 0,
            page: 0,
            offset: 0,
            target: 0,
            bytes: PAGE_BYTES,
        }));
        assert!(copies.iter().any(|copy| copy.source == 1));
    }

    /// One rank to Ulysses 2: each rank takes its own rows of the one
    /// stored rank's pages.
    #[test]
    fn one_rank_becomes_two_sequence_parallel_ranks() {
        let sources = [share((0, 40), (0, 12))];
        for target in [share((0, 22), (0, 5)), share((22, 18), (5, 7))] {
            let copies = copies(&sources, &target).unwrap();
            assert!(holds_its_samples(
                &target,
                &restored(&sources, &target, &copies)
            ));
        }
    }

    /// The layout the trajectory was stored in: every page is copied whole
    /// from the page of the same index.
    #[test]
    fn the_stored_layout_copies_every_page_whole() {
        let sources = [share((0, 22), (0, 5)), share((22, 18), (5, 7))];
        let target = share((22, 18), (5, 7));
        let mut copies = copies(&sources, &target).unwrap();
        copies.sort_by_key(|copy| copy.target);
        let pages: Vec<(usize, u64, u64)> = copies
            .iter()
            .map(|copy| (copy.source, copy.page, copy.bytes))
            .collect();
        assert_eq!(
            pages,
            (0..target.pages())
                .map(|page| (1, page, PAGE_BYTES))
                .collect::<Vec<_>>()
        );
    }

    /// Sources that do not hold the rank's rows refuse the restore.
    #[test]
    fn rows_no_source_holds_are_refused() {
        let sources = [share((0, 22), (0, 5))];
        assert!(copies(&sources, &share((0, 40), (0, 12))).is_err());
    }
}
