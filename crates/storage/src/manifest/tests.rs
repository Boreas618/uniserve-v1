//! Manifests applied to byte images of both worlds: the target's bytes
//! after the mappings, conversions, copies and fills equal what the
//! target holds when it loads its regions directly.

use std::collections::{BTreeMap, BTreeSet};

use crate::protocol::{Link, Memory, Placement};

use super::*;

/// The allocation granularity of these worlds: small, so that shards of a
/// few kilobytes lie on chunk bounds as gigabyte shards do on devices.
const GRANULE: u64 = 256;

/// The element of logical tensor `tensor` at `index`, as the bytes every
/// world holding it stores.
fn element(tensor: &str, index: &[u64]) -> [u8; 4] {
    let mut value: u32 = tensor
        .bytes()
        .fold(17, |hash, byte| hash.wrapping_mul(31) ^ u32::from(byte));
    for coordinate in index {
        value = value
            .wrapping_mul(1_000_003)
            .wrapping_add(*coordinate as u32 + 1);
    }
    value.to_le_bytes()
}

/// A row-major `u32` field holding the segments `segments` of a logical
/// tensor of `shape`, at `offset`.
fn field(
    tensor: &str,
    offset: u64,
    shape: &[u64],
    segments: Vec<Vec<Segment>>,
    units: Vec<u64>,
) -> Field {
    let local: Vec<u64> = segments
        .iter()
        .map(|segments| {
            segments
                .iter()
                .map(|segment| segment.local + segment.span.count)
                .max()
                .unwrap_or(0)
        })
        .collect();
    let mut strides = vec![1; local.len()];
    for dim in (0..local.len().saturating_sub(1)).rev() {
        strides[dim] = strides[dim + 1] * local[dim + 1];
    }
    Field {
        tensor: tensor.into(),
        source: "model".into(),
        dtype: "uint32".into(),
        element: 4,
        offset,
        shape: local,
        strides,
        logical: Logical {
            shape: shape.to_vec(),
            dims: (0..shape.len()).map(|dim| format!("d{dim}")).collect(),
            units,
            segments,
        },
        encoding: Encoding::Affine,
    }
}

fn whole(extent: u64) -> Vec<Segment> {
    vec![Segment {
        span: Span::range(0, extent),
        local: 0,
    }]
}

fn part(start: u64, count: u64) -> Vec<Segment> {
    vec![Segment {
        span: Span::range(start, count),
        local: 0,
    }]
}

/// Lay fields out back to back in one storage each `per_storage` fields,
/// aligning each storage's fields to 64 bytes.
fn storage(mut fields: Vec<Field>) -> Storage {
    let mut offset = 0;
    for field in &mut fields {
        field.offset = offset;
        offset = (field.bytes().end).next_multiple_of(64);
    }
    let layout = StorageLayout {
        bytes: offset,
        fields,
    };
    Storage::chunked(layout, GRANULE, &[1, 2, 4], &|_| Some([3; 32])).unwrap()
}

/// What a rank of a small mixture-of-experts model holds under tensor
/// parallelism `tp` (rank `t`) and expert parallelism `ep` (rank `e`):
/// queries split by heads, an output projection split by columns, eight
/// experts whose up and gate rows split by channels without expert
/// parallelism, a padded vocabulary
/// split by rows and a replicated norm.
fn moe_rank(tp: u64, t: u64, ep: u64, e: u64) -> Vec<Storage> {
    let (heads, head_dim, hidden) = (8, 4, 16);
    let q_rows = heads * head_dim / tp;
    let q = field(
        "q",
        0,
        &[heads * head_dim, hidden],
        vec![part(t * q_rows, q_rows), whole(hidden)],
        vec![head_dim, 0],
    );
    let o_columns = heads * head_dim / tp;
    let o = field(
        "o",
        0,
        &[hidden, heads * head_dim],
        vec![whole(hidden), part(t * o_columns, o_columns)],
        vec![0, head_dim],
    );
    let (experts, intermediate) = (8, 8);
    let local_experts = experts / ep;
    // An expert layer is partitioned over one group at a time: its
    // channels split over the tensor-parallel ranks without expert
    // parallelism only.
    let (channels, channel) = if ep > 1 { (1, 0) } else { (tp, t) };
    let width = intermediate / channels;
    let up_gate = field(
        "up_gate",
        0,
        &[experts, 2 * intermediate, hidden],
        vec![
            part(e * local_experts, local_experts),
            vec![
                Segment {
                    span: Span::range(channel * width, width),
                    local: 0,
                },
                Segment {
                    span: Span::range(intermediate + channel * width, width),
                    local: width,
                },
            ],
            whole(hidden),
        ],
        vec![1, 1, 0],
    );
    let down = field(
        "down",
        0,
        &[experts, hidden, intermediate],
        vec![
            part(e * local_experts, local_experts),
            whole(hidden),
            part(channel * width, width),
        ],
        vec![1, 0, 1],
    );
    // 50 tokens padded to 52 rows: the last shard ends in padding.
    let vocabulary: u64 = 50;
    let rows = 52 / tp;
    let held = rows.min(vocabulary.saturating_sub(t * rows));
    let mut embedding = field(
        "embedding",
        0,
        &[vocabulary, hidden],
        vec![part(t * rows, held), whole(hidden)],
        vec![1, 0],
    );
    embedding.shape[0] = rows;
    embedding.strides = vec![hidden, 1];
    let norm = field("norm", 0, &[hidden], vec![whole(hidden)], vec![0]);
    vec![
        storage(vec![q]),
        storage(vec![o]),
        storage(vec![up_gate]),
        storage(vec![down]),
        storage(vec![embedding]),
        storage(vec![norm]),
    ]
}

fn gpu(index: usize) -> Gpu {
    [index as u8 + 1; 16]
}

/// Links between `gpus` GPUs as a Transfer Server measures them, with the
/// host's.
fn links(gpus: usize) -> Vec<LinkBandwidth> {
    let mut links = vec![LinkBandwidth {
        link: Link {
            source: Memory::Host,
            destination: Memory::Host,
        },
        bytes_per_second: 50_000_000_000,
    }];
    for source in 0..gpus {
        for target in 0..gpus {
            links.push(LinkBandwidth {
                link: Link {
                    source: Memory::Device { gpu: gpu(source) },
                    destination: Memory::Device { gpu: gpu(target) },
                },
                // A copy within a GPU is faster than one between GPUs.
                bytes_per_second: if source == target {
                    4_000_000_000_000
                } else {
                    400_000_000_000
                },
            });
        }
    }
    links
}

/// The canonical bytes of an encoded field stored as this test's encoding:
/// elements in reverse order, each byte inverted.
fn encode(canonical: &[u8]) -> Vec<u8> {
    canonical
        .chunks(4)
        .rev()
        .flat_map(|element| element.iter().map(|byte| !byte))
        .collect()
}

fn decode(stored: &[u8]) -> Vec<u8> {
    encode(stored)
}

/// The bytes of a field as loading writes them: each held element its
/// logical value, padding zero.
fn field_image(field: &Field, canonical: bool) -> Vec<u8> {
    let (shape, strides, logical) = match (&field.encoding, canonical) {
        (
            Encoding::Encoded {
                canonical: Some(canonical),
                ..
            },
            true,
        ) => (&canonical.shape, &canonical.strides, &canonical.logical),
        _ => (&field.shape, &field.strides, &field.logical),
    };
    let numel: u64 = shape.iter().product();
    // The bytes from the first element to the last, gaps between rows of
    // a view included (their contents are no field's, left zero).
    let extent = shape
        .iter()
        .zip(strides)
        .map(|(size, stride)| size.saturating_sub(1) * stride)
        .sum::<u64>()
        + 1;
    let mut bytes = vec![0; (extent * field.element) as usize];
    let mut index = vec![0_u64; shape.len()];
    for _ in 0..numel {
        let held: Option<Vec<u64>> = index
            .iter()
            .enumerate()
            .map(|(dim, local)| {
                logical.segments[dim].iter().find_map(|segment| {
                    segment
                        .locals()
                        .contains(local)
                        .then(|| segment.span.index(local - segment.local))
                })
            })
            .collect();
        if let Some(logical_index) = held {
            let at = index.iter().zip(strides).map(|(i, s)| i * s).sum::<u64>() as usize * 4;
            bytes[at..at + 4].copy_from_slice(&element(&field.tensor, &logical_index));
        }
        for dim in (0..index.len()).rev() {
            index[dim] += 1;
            if index[dim] < shape[dim] {
                break;
            }
            index[dim] = 0;
        }
    }
    if matches!(field.encoding, Encoding::Encoded { .. }) && !canonical {
        return encode(&field_image(field, true));
    }
    bytes
}

/// A world's storages as loading writes them.
fn image(world: &World) -> Vec<Vec<Vec<u8>>> {
    world
        .ranks
        .iter()
        .map(|rank| {
            rank.storages
                .iter()
                .map(|storage| {
                    let mut bytes = vec![0xAB; storage.layout.padded(GRANULE) as usize];
                    for field in &storage.layout.fields {
                        let range = field.bytes();
                        bytes[range.start as usize..range.end as usize]
                            .copy_from_slice(&field_image(field, false));
                    }
                    bytes
                })
                .collect()
        })
        .collect()
}

/// The target world's storages after executing `manifest` on the source
/// world's bytes, starting from garbage.
fn apply(request: &Request, manifest: &Manifest) -> Vec<Vec<Vec<u8>>> {
    let source = image(&request.source);
    let mut target: Vec<Vec<Vec<u8>>> = request
        .target
        .ranks
        .iter()
        .map(|rank| {
            rank.storages
                .iter()
                .map(|storage| vec![0xCD; storage.layout.padded(GRANULE) as usize])
                .collect()
        })
        .collect();
    let mut stagings: Vec<Vec<u8>> = manifest
        .stagings
        .iter()
        .map(|staging| vec![0xEF; staging.bytes as usize])
        .collect();
    for mapping in &manifest.mappings {
        let from = &request.source.ranks[mapping.source_rank as usize].storages
            [mapping.source_storage as usize];
        let chunk = &from.chunks[mapping.source_chunk as usize];
        let into = &request.target.ranks[mapping.rank as usize].storages[mapping.storage as usize]
            .chunks[mapping.chunk as usize];
        assert_eq!(chunk.key, into.key);
        let bytes = source[mapping.source_rank as usize][mapping.source_storage as usize]
            [chunk.offset as usize..(chunk.offset + chunk.length) as usize]
            .to_vec();
        target[mapping.rank as usize][mapping.storage as usize]
            [into.offset as usize..(into.offset + into.length) as usize]
            .copy_from_slice(&bytes);
    }
    let fields_of = |world: &World, rank: u32, conversion: &Conversion| -> Vec<(u32, Field)> {
        conversion
            .fields
            .iter()
            .map(|(storage, tensor)| {
                let field = world.ranks[rank as usize].storages[*storage as usize]
                    .layout
                    .fields
                    .iter()
                    .find(|field| field.tensor == *tensor)
                    .unwrap()
                    .clone();
                (*storage, field)
            })
            .collect()
    };
    for conversion in &manifest.encodes {
        let staging = &manifest.stagings[conversion.staging as usize];
        for (storage, field) in fields_of(&request.source, staging.rank, conversion) {
            let range = field.bytes();
            let stored = &source[staging.rank as usize][storage as usize]
                [range.start as usize..range.end as usize];
            let Encoding::Encoded {
                canonical: Some(canonical),
                ..
            } = &field.encoding
            else {
                panic!("an encode converts encoded fields");
            };
            let canonical_bytes = decode(stored);
            let at = canonical.offset as usize;
            stagings[conversion.staging as usize][at..at + canonical_bytes.len()]
                .copy_from_slice(&canonical_bytes);
        }
    }
    for copy in &manifest.copies {
        for row in 0..copy.rows {
            let from = copy.source.offset + row * copy.source_pitch;
            let bytes = match copy.source.space {
                Space::Storage { storage } => source[copy.source.rank as usize][storage as usize]
                    [from as usize..(from + copy.length) as usize]
                    .to_vec(),
                Space::Staging { staging } => stagings[staging as usize]
                    [from as usize..(from + copy.length) as usize]
                    .to_vec(),
            };
            let to = (copy.target.offset + row * copy.target_pitch) as usize;
            match copy.target.space {
                Space::Storage { storage } => target[copy.target.rank as usize][storage as usize]
                    [to..to + bytes.len()]
                    .copy_from_slice(&bytes),
                Space::Staging { staging } => {
                    stagings[staging as usize][to..to + bytes.len()].copy_from_slice(&bytes);
                }
            }
        }
    }
    for conversion in &manifest.decodes {
        let staging = &manifest.stagings[conversion.staging as usize];
        for (storage, field) in fields_of(&request.target, staging.rank, conversion) {
            let Encoding::Encoded {
                canonical: Some(canonical),
                ..
            } = &field.encoding
            else {
                panic!("a decode converts encoded fields");
            };
            let length = (canonical.shape.iter().product::<u64>() * field.element) as usize;
            let at = canonical.offset as usize;
            let stored = encode(&stagings[conversion.staging as usize][at..at + length]);
            let range = field.bytes();
            target[staging.rank as usize][storage as usize]
                [range.start as usize..range.end as usize]
                .copy_from_slice(&stored);
        }
    }
    for fill in &manifest.fills {
        for row in 0..fill.rows {
            let at = (fill.offset + row * fill.pitch) as usize;
            target[fill.rank as usize][fill.storage as usize][at..at + fill.length as usize]
                .fill(0);
        }
    }
    target
}

/// Assert that every target field holds the bytes loading would write.
fn assert_fields_equal(request: &Request, applied: &[Vec<Vec<u8>>]) {
    let expected = image(&request.target);
    for (rank, storages) in request.target.ranks.iter().enumerate() {
        for (storage, held) in storages.storages.iter().enumerate() {
            for field in &held.layout.fields {
                // Every element of the field; the gaps of a view are no
                // field's bytes.
                let mut index = vec![0_u64; field.shape.len()];
                for _ in 0..field.numel() {
                    let at = field.address(&index) as usize;
                    let range = at..at + field.element as usize;
                    assert!(
                        applied[rank][storage][range.clone()] == expected[rank][storage][range],
                        "rank {rank} field {} differs at {index:?}",
                        field.tensor
                    );
                    for dim in (0..index.len()).rev() {
                        index[dim] += 1;
                        if index[dim] < field.shape[dim] {
                            break;
                        }
                        index[dim] = 0;
                    }
                }
            }
        }
    }
}

fn world(ranks: Vec<(Option<Gpu>, Vec<Storage>)>) -> World {
    World {
        ranks: ranks
            .into_iter()
            .map(|(gpu, storages)| Rank { gpu, storages })
            .collect(),
    }
}

#[test]
fn tensor_parallel_replicas_become_expert_parallel_ranks() {
    // Two replicas of tensor parallelism 2 on GPUs 0..4, then each GPU a
    // data-parallel rank of the four-way expert partition.
    let source = world(
        (0..4)
            .map(|index| (Some(gpu(index)), moe_rank(2, index as u64 % 2, 1, 0)))
            .collect(),
    );
    let target = world((0..4).map(|rank| (None, moe_rank(1, 0, 4, rank))).collect());
    let request = Request {
        source,
        target,
        gpus: (0..4).map(gpu).collect(),
        links: links(4),
    };
    let manifest = Manifest::plan(&request).unwrap();
    assert!(manifest.missing.is_empty());
    assert_fields_equal(&request, &apply(&request, &manifest));
    // The query rows and the norm each target finds on its GPU are mapped.
    let mapped: BTreeMap<u32, u64> =
        manifest
            .mappings
            .iter()
            .fold(BTreeMap::new(), |mut sum, mapping| {
                let chunk = &request.target.ranks[mapping.rank as usize].storages
                    [mapping.storage as usize]
                    .chunks[mapping.chunk as usize];
                *sum.entry(mapping.rank).or_default() += chunk.length;
                sum
            });
    assert_eq!(mapped.len(), 4);
    // Every span stays inside one chunk on each side, named at its GPU, and
    // the spans write exactly the target bytes the copies write, a strided
    // copy as spans of rows rather than a span per row.
    let spans: Vec<crate::protocol::Span> =
        manifest.spans(&request).collect::<Result<_>>().unwrap();
    assert_eq!(
        spans.iter().map(crate::protocol::Span::copied).sum::<u64>(),
        manifest.copied_bytes()
    );
    let rows: u64 = manifest.copies.iter().map(|copy| copy.rows).sum();
    assert!(
        (spans.len() as u64) < rows,
        "{} spans for {rows} rows",
        spans.len()
    );
    // A chunk's GPU, key and offset in its storage, by its key at a GPU.
    let chunk = |region: &crate::protocol::Region, world: &World, gpus: &[Gpu]| {
        let crate::protocol::Region::Tensor {
            key,
            at: Some(Placement::Device { gpu }),
        } = region
        else {
            panic!("a span names a chunk at its GPU");
        };
        world
            .ranks
            .iter()
            .zip(gpus)
            .enumerate()
            .filter(|(_, (_, held))| *held == gpu)
            .flat_map(|(rank, (held, _))| {
                held.storages
                    .iter()
                    .enumerate()
                    .flat_map(move |(storage, value)| {
                        value.chunks.iter().map(move |chunk| (rank, storage, chunk))
                    })
            })
            .find(|(_, _, chunk)| chunk.key == *key)
            .map(|(rank, storage, chunk)| (rank, storage, chunk.offset, chunk.length))
            .expect("a span names a chunk its rank holds")
    };
    let source_gpus: Vec<Gpu> = request
        .source
        .ranks
        .iter()
        .map(|rank| rank.gpu.unwrap())
        .collect();
    let mut written = BTreeSet::new();
    for span in &spans {
        let extent = |pitch: u64| (span.rows - 1) * pitch + span.bytes;
        let (_, _, _, source_length) = chunk(&span.source, &request.source, &source_gpus);
        assert!(span.source_offset + extent(span.source_pitch) <= source_length);
        let (rank, storage, start, length) =
            chunk(&span.destination, &request.target, &manifest.gpus);
        assert!(span.destination_offset + extent(span.destination_pitch) <= length);
        for row in 0..span.rows {
            let first = start + span.destination_offset + row * span.destination_pitch;
            for byte in first..first + span.bytes {
                assert!(
                    written.insert((rank, storage, byte)),
                    "a byte is written twice"
                );
            }
        }
    }
    let mut expected = BTreeSet::new();
    for copy in &manifest.copies {
        let Space::Storage { storage } = copy.target.space else {
            continue;
        };
        for row in 0..copy.rows {
            let first = copy.target.offset + row * copy.target_pitch;
            for byte in first..first + copy.length {
                expected.insert((copy.target.rank as usize, storage as usize, byte));
            }
        }
    }
    assert_eq!(written, expected);
}

#[test]
fn ranks_take_the_gpus_that_hold_their_experts() {
    // Two expert-parallel ranks on GPUs 0 and 1, then four ranks with
    // tensor-parallel attention and four-way experts on GPUs 0..4: the
    // ranks whose experts GPUs 0 and 1 hold take them.
    let source = world(vec![
        (Some(gpu(0)), moe_rank(1, 0, 2, 0)),
        (Some(gpu(1)), moe_rank(1, 0, 2, 1)),
    ]);
    let target = world(
        (0..4)
            .map(|rank| (None, moe_rank(4, rank, 4, rank)))
            .collect(),
    );
    let request = Request {
        source,
        target,
        gpus: (0..4).map(gpu).collect(),
        links: links(4),
    };
    let manifest = Manifest::plan(&request).unwrap();
    assert!(manifest.missing.is_empty());
    assert_fields_equal(&request, &apply(&request, &manifest));
    let holder = |rank: usize| manifest.gpus[rank];
    assert!(
        [holder(0), holder(1)].contains(&gpu(0)) && [holder(2), holder(3)].contains(&gpu(1)),
        "{:?}",
        manifest.gpus
    );
    // Each kept GPU maps its rank's experts instead of copying them.
    for rank in 0..4 {
        let on_kept_gpu = [gpu(0), gpu(1)].contains(&holder(rank));
        let experts_mapped = manifest
            .mappings
            .iter()
            .any(|mapping| mapping.rank == rank as u32 && mapping.storage == 2);
        assert_eq!(experts_mapped, on_kept_gpu, "rank {rank}");
    }
}

#[test]
fn running_target_ranks_keep_their_gpus() {
    // The same change with target ranks already running on GPUs 0..4 in
    // order: rank 1, whose experts GPU 0 holds, copies them to GPU 1.
    let source = world(vec![
        (Some(gpu(0)), moe_rank(1, 0, 2, 0)),
        (Some(gpu(1)), moe_rank(1, 0, 2, 1)),
    ]);
    let target = world(
        (0..4)
            .map(|rank| (Some(gpu(rank as usize)), moe_rank(4, rank, 4, rank)))
            .collect(),
    );
    let request = Request {
        source,
        target,
        gpus: (0..4).map(gpu).collect(),
        links: links(4),
    };
    let manifest = Manifest::plan(&request).unwrap();
    assert_eq!(manifest.gpus, (0..4).map(gpu).collect::<Vec<_>>());
    assert!(manifest.missing.is_empty());
    assert_fields_equal(&request, &apply(&request, &manifest));
    let experts_mapped = |rank: u32| {
        manifest
            .mappings
            .iter()
            .any(|mapping| mapping.rank == rank && mapping.storage == 2)
    };
    assert!(experts_mapped(0) && !experts_mapped(1));

    let mut partial = request.clone();
    partial.target.ranks[1].gpu = None;
    assert!(Manifest::plan(&partial).is_err());
}

#[test]
fn ranks_on_another_host_receive_over_the_links_from_its_gpus() {
    // Two expert-parallel ranks on GPUs 0 and 1 of one host grow to four
    // ranks on GPUs 0..4, GPUs 2 and 3 on another host: that host's
    // Transfer Server measured GPUs 0 and 1 by READ, and GPU 1 over the
    // NVLink fabric too.
    let source = world(vec![
        (Some(gpu(0)), moe_rank(1, 0, 2, 0)),
        (Some(gpu(1)), moe_rank(1, 0, 2, 1)),
    ]);
    let target = world(
        (0..4)
            .map(|rank| (None, moe_rank(4, rank, 4, rank)))
            .collect(),
    );
    let read = |source: usize, destination: usize, bytes_per_second: u64| LinkBandwidth {
        link: Link {
            source: Memory::Remote {
                endpoint: "10.0.0.1:4100".into(),
                gpu: Some(gpu(source)),
            },
            destination: Memory::Device {
                gpu: gpu(destination),
            },
        },
        bytes_per_second,
    };
    let mut measured = links(4);
    measured.retain(|measured| {
        let host = |memory: &Memory| match memory {
            Memory::Device { gpu: on } => (0..4)
                .position(|index| gpu(index) == *on)
                .map(|index| index / 2),
            _ => None,
        };
        host(&measured.link.source) == host(&measured.link.destination)
    });
    for destination in [2, 3] {
        measured.push(read(0, destination, 80_000_000_000));
        measured.push(read(1, destination, 80_000_000_000));
    }
    let request = Request {
        source,
        target,
        gpus: (0..4).map(gpu).collect(),
        links: measured.clone(),
    };
    let manifest = Manifest::plan(&request).unwrap();
    assert!(manifest.missing.is_empty());
    assert_fields_equal(&request, &apply(&request, &manifest));
    // The ranks on the other host receive from both source GPUs.
    let remote = [gpu(2), gpu(3)];
    let received: BTreeSet<Gpu> = manifest
        .links
        .iter()
        .filter(|load| remote.contains(&load.target) && load.bytes > 0)
        .map(|load| load.source)
        .collect();
    assert_eq!(received, BTreeSet::from([gpu(0), gpu(1)]));

    // The fabric's bandwidth stands for a pair also measured by READ.
    let mut fabric = request.clone();
    for destination in [2, 3] {
        fabric.links.push(LinkBandwidth {
            link: Link {
                source: Memory::Device { gpu: gpu(1) },
                destination: Memory::Device {
                    gpu: gpu(destination),
                },
            },
            bytes_per_second: 400_000_000_000,
        });
    }
    let manifest = Manifest::plan(&fabric).unwrap();
    let seconds = |source: usize| {
        manifest
            .links
            .iter()
            .filter(|load| load.source == gpu(source) && remote.contains(&load.target))
            .map(|load| load.seconds * 400e9 / load.bytes as f64)
            .collect::<Vec<_>>()
    };
    assert!(
        seconds(1).iter().all(|scale| (scale - 1.0).abs() < 1e-9),
        "{:?}",
        seconds(1)
    );

    // Without a link from the other host's GPUs, no plan reaches them.
    let mut unmeasured = request;
    unmeasured
        .links
        .retain(|measured| !matches!(measured.link.source, Memory::Remote { .. }));
    assert!(Manifest::plan(&unmeasured).is_err());
}

/// A `u32` field of `tokens` positions of 8 channels holding `segments`.
fn tokens(tensor: &str, total: u64, segments: Vec<Segment>) -> Field {
    field(tensor, 0, &[total, 8], vec![segments, whole(8)], vec![1, 0])
}

#[test]
fn position_mod_n_and_strided_indices_redistribute() {
    // 32 positions split contiguously over two ranks become position-mod-2
    // shares, and a compressed index of one entry per 4 positions, split
    // contiguously, lies with the positions its entries stand for.
    let contiguous = |rank: u64| {
        vec![
            storage(vec![tokens("kv", 32, part(16 * rank, 16))]),
            storage(vec![tokens("index", 8, part(4 * rank, 4))]),
        ]
    };
    let cyclic = |rank: u64| {
        vec![
            storage(vec![tokens(
                "kv",
                32,
                vec![Segment {
                    span: Span {
                        start: rank,
                        count: 16,
                        step: 2,
                    },
                    local: 0,
                }],
            )]),
            // Entry j stands for position 4j + 1, which lies on rank
            // (4j + 1) mod 2 = 1: rank 0 holds no entry.
            storage(vec![tokens(
                "index",
                8,
                if rank == 1 { part(0, 8) } else { Vec::new() },
            )]),
        ]
    };
    let mut target_ranks = Vec::new();
    for rank in 0..2 {
        let mut storages = cyclic(rank);
        if storages[1].layout.fields[0].logical.segments[0].is_empty() {
            storages.pop();
        }
        target_ranks.push((None, storages));
    }
    let request = Request {
        source: world(vec![
            (Some(gpu(0)), contiguous(0)),
            (Some(gpu(1)), contiguous(1)),
        ]),
        target: world(target_ranks),
        gpus: (0..2).map(gpu).collect(),
        links: links(2),
    };
    let manifest = Manifest::plan(&request).unwrap();
    assert!(manifest.missing.is_empty());
    assert_fields_equal(&request, &apply(&request, &manifest));
}

/// The columns `first..first + count` of a `[rows, columns]` matrix as a
/// view of the whole matrix's storage: rows `columns` elements apart, the
/// other columns between them.
fn column_view(rows: u64, columns: u64, first: u64, count: u64) -> Storage {
    let mut view = field(
        "o",
        0,
        &[rows, columns],
        vec![whole(rows), part(first, count)],
        vec![0, 1],
    );
    view.offset = first * view.element;
    view.strides = vec![columns, 1];
    let layout = StorageLayout {
        bytes: rows * columns * view.element,
        fields: vec![view],
    };
    Storage::chunked(layout, GRANULE, &[1, 2, 4], &|_| Some([3; 32])).unwrap()
}

#[test]
fn column_views_with_gaps_between_rows_redistribute() {
    // Quarters of a matrix's columns, each a view into a storage of the
    // whole matrix, become halves; the gaps are no field's bytes.
    let request = Request {
        source: world(
            (0..4)
                .map(|rank| {
                    (
                        Some(gpu(rank as usize)),
                        vec![column_view(64, 32, 8 * rank, 8)],
                    )
                })
                .collect(),
        ),
        target: world(
            (0..2)
                .map(|rank| (None, vec![column_view(64, 32, 16 * rank, 16)]))
                .collect(),
        ),
        gpus: (0..4).map(gpu).collect(),
        links: links(4),
    };
    let manifest = Manifest::plan(&request).unwrap();
    assert!(manifest.missing.is_empty());
    assert_fields_equal(&request, &apply(&request, &manifest));
}

/// A field stored in this test's encoding, whose canonical form is the
/// affine field itself at offset zero of its group's staging space.
fn encoded(mut field: Field, group: &str) -> Field {
    field.encoding = Encoding::Encoded {
        name: "reversed".into(),
        group: group.into(),
        canonical: Some(Canonical {
            offset: 0,
            shape: field.shape.clone(),
            strides: field.strides.clone(),
            logical: field.logical.clone(),
        }),
    };
    field
}

#[test]
fn encoded_fields_move_in_canonical_form() {
    // An encoded matrix split by rows over two ranks becomes the whole
    // matrix: each source converts to canonical, the target converts back.
    let rows = |start: u64, count: u64| {
        encoded(
            field(
                "scales",
                0,
                &[64, 16],
                vec![part(start, count), whole(16)],
                vec![1, 0],
            ),
            "scales",
        )
    };
    let request = Request {
        source: world(vec![
            (Some(gpu(0)), vec![storage(vec![rows(0, 32)])]),
            (Some(gpu(1)), vec![storage(vec![rows(32, 32)])]),
        ]),
        target: world(vec![(None, vec![storage(vec![rows(0, 64)])])]),
        gpus: vec![gpu(0)],
        links: links(2),
    };
    let manifest = Manifest::plan(&request).unwrap();
    assert_eq!(manifest.encodes.len(), 2);
    assert_eq!(manifest.decodes.len(), 1);
    assert!(manifest.mappings.is_empty());
    assert_fields_equal(&request, &apply(&request, &manifest));

    // The same encoded region on another GPU moves as it is stored.
    let request = Request {
        source: world(vec![(Some(gpu(1)), vec![storage(vec![rows(0, 64)])])]),
        target: world(vec![(None, vec![storage(vec![rows(0, 64)])])]),
        gpus: vec![gpu(0)],
        links: links(2),
    };
    let manifest = Manifest::plan(&request).unwrap();
    assert!(manifest.encodes.is_empty() && manifest.decodes.is_empty());
    assert_fields_equal(&request, &apply(&request, &manifest));
}

/// A tensor a listing lists: `key` at `location`, retained by `world` of
/// deployment `model`.
fn listed(key: Key, name: &str, location: Placement, world: u64) -> crate::protocol::ListedTensor {
    let scope = crate::Scope {
        deployment: "model".into(),
        group: "model".into(),
        rank: 0,
    };
    crate::protocol::ListedTensor {
        key,
        name: name.into(),
        scopes: vec![scope.clone()],
        retainers: vec![crate::protocol::Retainer { scope, world }],
        placement: location,
        storage_bytes: 0,
        dtype: "uint8".into(),
        shape: vec![0],
        access: crate::protocol::Access::ReadOnly,
        writer: None,
    }
}

/// A weight manifest's text naming `storages` on the rank's first device,
/// of sources with the digests `fingerprints`.
fn weight_manifest(fingerprints: &[(&str, &str)], storages: &[&Storage]) -> String {
    let fingerprints: BTreeMap<String, String> = fingerprints
        .iter()
        .map(|(source, digest)| (source.to_string(), digest.to_string()))
        .collect();
    let storages: Vec<weights::RetainedStorage> = storages
        .iter()
        .map(|storage| weights::RetainedStorage {
            device: Some(0),
            storage: (*storage).clone(),
        })
        .collect();
    serde_json::json!({
        "version": weights::VERSION,
        "fingerprints": fingerprints,
        "tensors": [],
        "storages": storages,
    })
    .to_string()
}

#[test]
fn a_worlds_weight_manifests_are_sources_on_the_gpus_holding_all_their_chunks() {
    let held = storage(vec![field("w", 0, &[256], vec![whole(256)], vec![1])]);
    let partial = storage(vec![field("v", 0, &[256], vec![whole(256)], vec![1])]);
    let other = storage(vec![field("u", 0, &[256], vec![whole(256)], vec![1])]);
    assert!(held.chunks.len() > 1);
    let ours = [("", "structure"), ("model", "checkpoint")];
    let manifests: BTreeMap<Key, String> = [
        // Two of world 1's ranks name the same storage.
        (Key([1; 32]), weight_manifest(&ours, &[&held, &partial])),
        (Key([2; 32]), weight_manifest(&ours, &[&held])),
        // A storage of another checkpoint holds no bytes this rank needs.
        (
            Key([3; 32]),
            weight_manifest(&[("", "structure"), ("model", "another")], &[&other]),
        ),
        // World 2's manifest is not world 1's.
        (Key([4; 32]), weight_manifest(&ours, &[&other])),
    ]
    .into_iter()
    .collect();
    let host = Placement::Host { numa_node: None };
    let device = |index| Placement::Device { gpu: gpu(index) };
    let mut tensors = vec![
        listed(Key([1; 32]), weights::NAME, host, 1),
        listed(Key([2; 32]), weights::NAME, host, 1),
        listed(Key([3; 32]), weights::NAME, host, 1),
        listed(Key([4; 32]), weights::NAME, host, 2),
    ];
    // GPU 0 holds every chunk of both storages, GPU 1 all of `held` but
    // not the last chunk of `partial`; the storage of the other checkpoint
    // and world 2's lie on GPU 0 too.
    for chunk in &held.chunks {
        tensors.push(listed(chunk.key, "weights", device(0), 1));
        tensors.push(listed(chunk.key, "weights", device(1), 1));
    }
    for (index, chunk) in partial.chunks.iter().enumerate() {
        tensors.push(listed(chunk.key, "weights", device(0), 1));
        if index + 1 < partial.chunks.len() {
            tensors.push(listed(chunk.key, "weights", device(1), 1));
        }
    }
    for chunk in &other.chunks {
        tensors.push(listed(chunk.key, "weights", device(0), 1));
    }
    let mut listing = crate::protocol::Listing {
        tensors,
        transfer: Some(crate::protocol::ListedTransfer {
            pid: 1,
            links: links(2),
            contexts: Vec::new(),
            tasks: Vec::new(),
        }),
        ..Default::default()
    };
    let fingerprints: BTreeMap<String, String> = ours
        .iter()
        .map(|(source, digest)| (source.to_string(), digest.to_string()))
        .collect();
    let read = |key: Key| Ok::<_, ()>(manifests.get(&key).cloned());

    let sources = weights::sources(&listing, "model", Some(&[1]), &fingerprints, read)
        .unwrap()
        .unwrap();
    let mut on_first = vec![held.clone(), partial.clone()];
    on_first.sort_by_key(|storage| serde_json::to_vec(storage).unwrap());
    assert_eq!(
        sources.world,
        World {
            ranks: vec![
                Rank {
                    gpu: Some(gpu(0)),
                    storages: on_first,
                },
                Rank {
                    gpu: Some(gpu(1)),
                    storages: vec![held.clone()],
                },
            ]
        }
    );
    assert_eq!(sources.links, links(2));

    // A coupled reshape can read multiple predecessor worlds. Storage
    // described by the second world then becomes an eligible source too.
    let combined = weights::sources(&listing, "model", Some(&[1, 2]), &fingerprints, read)
        .unwrap()
        .unwrap();
    let mut expected = vec![held.clone(), partial, other];
    expected.sort_by_key(|storage| serde_json::to_vec(storage).unwrap());
    assert_eq!(combined.world.ranks[0].storages, expected);
    assert_eq!(combined.world.ranks[1].storages, vec![held]);

    // Another deployment retains none of them, and without a Transfer
    // Server nothing is copied.
    assert!(
        weights::sources(&listing, "other", Some(&[1]), &fingerprints, read)
            .unwrap()
            .is_none()
    );
    listing.transfer = None;
    assert!(
        weights::sources(&listing, "model", Some(&[1]), &fingerprints, read)
            .unwrap()
            .is_none()
    );
}

/// A storage holding rows `start..start + count` of two `u32` tensors of
/// 64 rows by 16 columns, the first at byte 0 and the second at byte
/// `second`, in pages of 1 KiB: 64 bytes a row, 16 rows a page.
fn paged_rows(start: u64, count: u64, second: u64, content: [u8; 32]) -> Storage {
    let rows = |tensor: &str, offset: u64| {
        field(
            tensor,
            offset,
            &[64, 16],
            vec![part(start, count), whole(16)],
            vec![1, 0],
        )
    };
    let fields = vec![rows("first", 0), rows("second", second)];
    let layout = StorageLayout {
        bytes: fields.iter().map(|field| field.bytes().end).max().unwrap(),
        fields,
    };
    Storage::paged(layout, 1024, &|_| Some(content)).unwrap()
}

#[test]
fn a_page_is_keyed_by_what_it_holds_and_where() {
    // Rows 0..32 of both tensors on one rank of two, or all 64 rows on one
    // rank: the first pages hold the same rows of the first tensor at the
    // same positions under both, and the pages past them hold other rows.
    let content = [5; 32];
    let half = paged_rows(0, 32, 2048, content);
    let whole_rows = paged_rows(0, 64, 4096, content);
    let keys =
        |storage: &Storage| -> Vec<Key> { storage.chunks.iter().map(|chunk| chunk.key).collect() };
    let (half, whole_rows) = (keys(&half), keys(&whole_rows));
    assert_eq!((half.len(), whole_rows.len()), (4, 8));
    assert_eq!(half[..2], whole_rows[..2]);
    assert_ne!(half[2], whole_rows[2]);
    // Pages of distinct contents never share a key, nor do the pages of
    // one storage.
    let other = keys(&paged_rows(0, 32, 2048, [6; 32]));
    assert!(half.iter().zip(&other).all(|(one, two)| one != two));
    let distinct: BTreeSet<Key> = half.iter().copied().collect();
    assert_eq!(distinct.len(), 4);
    // Pages of no bytes are refused.
    let empty = StorageLayout {
        bytes: 0,
        fields: Vec::new(),
    };
    assert!(Storage::paged(empty, 0, &|_| Some(content)).is_err());
}
