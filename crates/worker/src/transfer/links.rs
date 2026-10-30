//! Measured links between retained storage regions on deployment nodes.

use crate::storage::allocation::Storage;
use crate::storage::retained::{ExclusiveSpec, Retained};
use crate::{Error, Result};
use ::storage::manifest::Binding;
use ::storage::manifest::weights::Sources;
use ::storage::protocol::{Access, LinkBandwidth, LinkEnd, Priority, Region, Span};
use ::storage::{Placement, StorageRecord, TensorRecord};
use std::collections::BTreeMap;
use std::os::fd::AsFd;

/// Bytes per sampling window used by the Transfer Server estimator.
const LINK_SAMPLE: u64 = 64 << 20;

/// The name of the scratch storage a measurement copies into.
const SCRATCH: &str = "link measurement scratch";

/// The links of `sources` into local storage `at`, after the host's Transfer
/// Server measures each unmeasured remote host or device source: two copies of
/// [`LINK_SAMPLE`] byte windows from the longest retained chunk into a
/// scratch storage the rank creates and releases. Smaller chunks repeat
/// until the window is complete. `device` owns the scratch allocation and
/// CUDA registration even when the destination is host storage. The
/// scratch is an exclusive storage of the rank's scope
/// ([`Retained::exclusive_key`]), so ranks of other worker groups on the
/// same GPU measure their links at once.
pub(crate) fn measure(
    retained: &Retained,
    sources: &Sources,
    device: cuda::DeviceId,
    at: Placement,
) -> Result<Vec<LinkBandwidth>> {
    let into = match at {
        Placement::Device { gpu } => LinkEnd::Device { gpu },
        Placement::Host { .. } => LinkEnd::Host,
    };
    let reaches = |links: &[LinkBandwidth], source: &LinkEnd| {
        links.iter().any(|measured| {
            measured.link.destination == into && (measured.link.source == *source ||
                matches!((source, &measured.link.source), (LinkEnd::Remote { gpu: Some(gpu), .. }, LinkEnd::Device { gpu: from }) if gpu == from))
        })
    };
    let mut unmeasured = BTreeMap::new();
    for rank in &sources.world.ranks {
        for chunk in rank.storages.iter().flat_map(|storage| &storage.chunks) {
            let Binding::Remote { endpoint, at } = &chunk.binding else {
                continue;
            };
            let source = LinkEnd::Remote {
                endpoint: endpoint.clone(),
                gpu: match at {
                    Placement::Device { gpu } => Some(*gpu),
                    Placement::Host { .. } => None,
                },
            };
            if chunk.length == 0 || reaches(&sources.links, &source) {
                continue;
            }
            let held = unmeasured.entry(source).or_insert((chunk, rank.gpu));
            if (chunk.length, chunk.key) > (held.0.length, held.0.key) {
                *held = (chunk, rank.gpu);
            }
        }
    }
    if unmeasured.is_empty() {
        return Ok(sources.links.clone());
    }

    let started = std::time::Instant::now();
    let bytes = 2 * LINK_SAMPLE;
    let (scratch, descriptor) = match at {
        Placement::Device { .. } => {
            let scratch = Storage::segmented(device, &[bytes as usize], &[])?;
            let descriptor = scratch.create_segment(0)?;
            (scratch, descriptor)
        }
        Placement::Host { .. } => Storage::host_retainable(device, bytes as usize)?,
    };
    let key = retained.exclusive_key(
        &ExclusiveSpec {
            name: SCRATCH,
            device,
            bytes: bytes as usize,
            access: &[],
        },
        &device.uuid()?.0,
    );
    retained.call("retaining a link measurement's scratch", |client| {
        client.put(
            vec![TensorRecord {
                key,
                name: SCRATCH.into(),
                storage: 0,
                offset: 0,
                dtype: "uint8".into(),
                shape: vec![bytes],
                strides: vec![1],
                access: Access::Exclusive,
            }],
            &[(
                StorageRecord {
                    placement: at,
                    bytes,
                    segment: None,
                    fabric: None,
                },
                descriptor.as_fd(),
            )],
        )
    })?;
    retained.use_keys([key]);
    let copied = unmeasured.iter().try_for_each(|(_, (chunk, rank_gpu))| {
        let region = chunk
            .binding
            .region(chunk.key, *rank_gpu)
            .map_err(|error| Error::Interface(error.to_string()))?;
        let bytes = chunk.length.min(LINK_SAMPLE);
        // Repeat a retained chunk when it is smaller than the estimator's
        // sample window. Measurement never requires a model-sized allocation.
        let spans = (0..2 * LINK_SAMPLE.div_ceil(bytes))
            .map(|_| Span {
                source: region.clone(),
                source_offset: chunk.storage_offset,
                destination: Region::Tensor { key, at: None },
                destination_offset: 0,
                bytes,
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
    let missing: Vec<_> = unmeasured
        .keys()
        .filter(|source| !reaches(&links, source))
        .collect();
    if !missing.is_empty() {
        tracing::warn!(?missing, "remote storage links remain unmeasured");
    }
    tracing::info!(
        links = unmeasured.len(),
        elapsed_ms = started.elapsed().as_millis() as u64,
        "measured remote storage links"
    );
    Ok(links)
}
