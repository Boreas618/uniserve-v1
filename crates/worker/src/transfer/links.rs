//! Measured links between retained storage regions on deployment nodes.

use crate::Result;
use crate::memory::allocation::Storage;
use crate::memory::retained::Retained;
use std::os::fd::AsFd;
use storage::manifest::Gpu;
use storage::manifest::weights::Sources;
use storage::protocol::{Access, LinkBandwidth, Memory, Priority, Region, Span};
use storage::{Key, Placement, StorageRecord, TensorRecord};

/// Bytes per sampling window used by the Transfer Server estimator.
const LINK_SAMPLE: u64 = 64 << 20;

/// The links of `sources` into `gpu`, this rank's GPU on `device`, as the
/// host's Transfer Server measured them, after it measured the link from
/// every source GPU of another node it had not: for each, two copies of
/// [`LINK_SAMPLE`] byte windows from the longest retained chunk into a
/// scratch storage the rank creates and releases. Smaller chunks repeat
/// until the window is complete; one task per GPU owns the scratch.
pub(crate) fn measure(
    retained: &Retained,
    sources: &Sources,
    device: cuda::DeviceId,
    gpu: Gpu,
) -> Result<Vec<LinkBandwidth>> {
    let into = Memory::Device { gpu };
    let reaches = |links: &[LinkBandwidth], source: &Gpu| {
        links.iter().any(|measured| {
            measured.link.destination == into
                && match &measured.link.source {
                    Memory::Device { gpu: from }
                    | Memory::Remote {
                        gpu: Some(from), ..
                    } => from == source,
                    _ => false,
                }
        })
    };
    let mut unmeasured: Vec<(Gpu, String, Key, u64, u64)> = Vec::new();
    for (source, endpoint) in &sources.remote {
        if reaches(&sources.links, source) {
            continue;
        }
        let longest = sources
            .world
            .ranks
            .iter()
            .filter(|rank| rank.gpu == Some(*source))
            .flat_map(|rank| rank.storages.iter().flat_map(|storage| &storage.chunks))
            .filter(|chunk| chunk.length > 0)
            .max_by_key(|chunk| (chunk.length, chunk.key));
        match longest {
            Some(chunk) => unmeasured.push((
                *source,
                endpoint.clone(),
                chunk.key,
                chunk.length.min(LINK_SAMPLE),
                chunk.storage_offset,
            )),
            None => tracing::warn!(
                gpu = %cuda::DeviceUuid(*source),
                endpoint,
                "another node's GPU holds no bytes to measure its link"
            ),
        }
    }
    if unmeasured.is_empty() {
        return Ok(sources.links.clone());
    }

    let started = std::time::Instant::now();
    let bytes = 2 * LINK_SAMPLE;
    let scratch = Storage::segmented(device, &[bytes as usize], &[])?;
    let descriptor = scratch.create_segment(0)?;
    let key = measuring_key(retained.world(), &gpu);
    retained.call("retaining a link measurement's scratch", |client| {
        client.put(
            vec![TensorRecord {
                key,
                name: "link measurement scratch".into(),
                storage: 0,
                offset: 0,
                dtype: "uint8".into(),
                shape: vec![bytes],
                strides: vec![1],
                access: Access::Exclusive,
            }],
            &[(
                StorageRecord {
                    placement: Placement::Device { gpu },
                    bytes,
                    segment: None,
                    fabric: None,
                },
                descriptor.as_fd(),
            )],
        )
    })?;
    retained.use_keys([key]);
    let copied = unmeasured
        .iter()
        .try_for_each(|(source, endpoint, chunk, bytes, offset)| {
            // Repeat a retained chunk when it is smaller than the estimator's
            // sample window. Measurement never requires a model-sized allocation.
            let spans = (0..2 * LINK_SAMPLE.div_ceil(*bytes))
                .map(|_| Span {
                    source: Region::Remote {
                        endpoint: endpoint.clone(),
                        region: Box::new(Region::Tensor {
                            key: *chunk,
                            at: Some(Placement::Device { gpu: *source }),
                        }),
                    },
                    source_offset: *offset,
                    destination: Region::Tensor { key, at: None },
                    destination_offset: 0,
                    bytes: *bytes,
                    rows: 1,
                    source_pitch: 0,
                    destination_pitch: 0,
                })
                .collect();
            let task = retained.transfer(spans, Priority::Background, retained.world())?;
            retained.transferred(task)
        });
    retained.forget(&[key])?;
    drop(scratch);
    copied?;

    // The Transfer Server reports a link it measured before the end of the
    // task that measured it.
    let listing = retained.call("listing the storage server", |client| client.list())?;
    let links = listing
        .transfer
        .map(|transfer| transfer.links)
        .unwrap_or_default();
    let missing: Vec<String> = unmeasured
        .iter()
        .filter(|(source, ..)| !reaches(&links, source))
        .map(|(source, ..)| cuda::DeviceUuid(*source).to_string())
        .collect();
    if !missing.is_empty() {
        tracing::warn!(
            ?missing,
            "the links from these GPUs of another node stay unmeasured"
        );
    }
    tracing::info!(
        gpus = unmeasured.len(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "measured the links from another node's GPUs"
    );
    Ok(links)
}

/// The key a rank's link measurement scratch on `gpu` is retained under in
/// world `world`.
fn measuring_key(world: u64, gpu: &Gpu) -> Key {
    // Scratch is named by its owner, not by the bytes the transfer writes.
    let mut key = [0; 32];
    key[..8].copy_from_slice(b"linkbw\0\0");
    key[8..16].copy_from_slice(&world.to_le_bytes());
    key[16..].copy_from_slice(gpu);
    Key(key)
}
