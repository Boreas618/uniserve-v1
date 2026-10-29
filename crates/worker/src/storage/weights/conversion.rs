//! Execute canonical weight copies on the receiving rank's preparation stream.
//!
//! Only one target group and one source group are staged at a time. Source
//! fields are gathered by the Transfer Server, converted without changing
//! numerical precision, and copied according to the agreed manifest. The
//! source world needs no callback and may keep serving throughout.

use std::collections::{BTreeMap, HashMap};
use std::os::fd::AsFd;
use std::sync::Arc;

use ::storage::manifest::{self, Conversion, Copy, Encoding, Field, Manifest, Space};
use ::storage::protocol::{Access, Priority, Region, Span};
use ::storage::{Key, Placement, StorageRecord, TensorRecord};
use cuda::{DeviceId, Stream};
use sha2::{Digest, Sha256};
use tvm_ffi::Any;

use crate::ffi::{Backing, DType, Interpreter, list};
use crate::storage::allocation::{Storage, StorageView};
use crate::storage::retained::Retained;
use crate::{Error, Result};

/// The agreed copy plan viewed from one receiver. Source transport bindings
/// may name other hosts; numerical layouts remain those of the agreed plan.
#[derive(Clone, Copy)]
pub(super) struct Conversions<'a> {
    pub retained: &'a Retained,
    pub request: &'a manifest::Request,
    pub manifest: &'a Manifest,
    pub sources: &'a manifest::World,
    pub rank: u32,
    pub device: DeviceId,
}

impl<'a> Conversions<'a> {
    /// Complete canonical copies before target fields can be read. Ordinary
    /// affine copies remain on the asynchronous receiving path.
    pub(super) fn execute(
        &self,
        targets: &[Arc<Storage>],
        receiving: &HashMap<Key, Key>,
    ) -> Result<u64> {
        let Self {
            retained,
            request,
            manifest,
            sources,
            rank,
            device,
        } = *self;
        let mut groups = BTreeMap::<Option<u32>, Vec<Copy>>::new();
        for copy in &manifest.copies {
            if copy.target.rank != rank {
                continue;
            }
            let target = match copy.target.space {
                Space::Staging { staging } => {
                    let group = &manifest.stagings[staging as usize].group;
                    // A checkpoint load supplies a whole numerical tensor.
                    // Do not decode an incomplete group before that load.
                    if manifest.missing.iter().any(|missing| {
                        missing.rank == rank
                            && missing
                                .tensor
                                .rsplit_once('/')
                                .is_some_and(|(tensor, _)| tensor == group)
                    }) {
                        continue;
                    }
                    Some(staging)
                }
                Space::Storage { .. } if matches!(copy.source.space, Space::Staging { .. }) => None,
                Space::Storage { .. } => continue,
            };
            groups.entry(target).or_default().push(*copy);
        }
        if groups.is_empty() {
            return Ok(0);
        }
        let stream = Stream::with_priority(device, i32::MAX)?;
        let mut moved = 0;
        for (target, copies) in groups {
            let destination = target
                .map(|index| {
                    Scratch::new(
                        retained,
                        manifest,
                        index,
                        rank,
                        false,
                        device,
                        manifest.stagings[index as usize].bytes,
                    )
                })
                .transpose()?;
            if let Some(destination) = &destination {
                // Canonical local padding is not covered by logical spans.
                // Initialize it before transfers and the numerical decode.
                // SAFETY: this operation owns the complete allocation and
                // synchronizes its only writer before another user sees it.
                unsafe {
                    cuda::memset_d8(
                        destination.storage.address(),
                        0,
                        destination.storage.bytes(),
                        &stream,
                    )
                }?;
                stream.synchronize()?;
            }
            let mut batches = BTreeMap::<Option<u32>, Vec<Copy>>::new();
            for copy in copies {
                let source = match copy.source.space {
                    Space::Staging { staging } => Some(staging),
                    Space::Storage { .. } => None,
                };
                batches.entry(source).or_default().push(copy);
            }
            for (source, copies) in batches {
                let canonical = if let Some(index) = source {
                    let conversion = manifest
                        .encodes
                        .iter()
                        .find(|entry| entry.staging == index)
                        .ok_or(Error::State("a source staging has its conversion"))?;
                    let (scratch, bytes) = self.encode(conversion, &stream)?;
                    moved += bytes;
                    Some(scratch)
                } else {
                    None
                };
                let mut selected = manifest.clone();
                selected.copies = copies;
                let spans = selected
                    .bind(request, sources)
                    .spans()
                    .map(|span| {
                        let mut span = span.map_err(invalid)?;
                        if let Some(index) = source {
                            let scratch =
                                canonical.as_ref().ok_or(Error::State("source is staged"))?;
                            remap(&mut span.source, manifest.staging_key(index), scratch)?;
                        }
                        if let Some(index) = target {
                            let scratch = destination
                                .as_ref()
                                .ok_or(Error::State("target is staged"))?;
                            remap(&mut span.destination, manifest.staging_key(index), scratch)?;
                        } else if let Region::Tensor { key, .. } = &mut span.destination {
                            *key = *receiving
                                .get(key)
                                .ok_or(Error::State("converted weights are writable"))?;
                        }
                        Ok(span)
                    })
                    .collect::<Result<Vec<_>>>()?;
                moved += transfer(retained, spans)?;
            }
            if let Some(index) = target {
                let conversion = manifest
                    .decodes
                    .iter()
                    .find(|entry| entry.staging == index)
                    .ok_or(Error::State("a target staging has its conversion"))?;
                let fields = fields(&request.target, manifest, conversion)?;
                let views = fields
                    .iter()
                    .map(|(storage, field)| {
                        field_view(Arc::clone(&targets[*storage as usize]), field.offset, field)
                    })
                    .collect::<Result<Vec<_>>>()?;
                convert(
                    &fields,
                    &views,
                    destination
                        .as_ref()
                        .ok_or(Error::State("target is staged"))?,
                    false,
                    &stream,
                )?;
            }
        }
        Ok(moved)
    }

    fn encode(&self, conversion: &Conversion, stream: &Stream) -> Result<(Scratch<'a>, u64)> {
        let Self {
            retained,
            request,
            manifest,
            sources,
            rank,
            device,
        } = *self;
        let index = conversion.staging;
        let staging = &manifest.stagings[index as usize];
        let fields = fields(&request.source, manifest, conversion)?;
        let mut bytes: u64 = 0;
        let mut offsets = Vec::new();
        for (_, field) in &fields {
            bytes = bytes.div_ceil(256) * 256;
            offsets.push(bytes);
            bytes += field_bytes(field);
        }
        let raw = Scratch::new(retained, manifest, index, rank, true, device, bytes)?;
        let mut spans = Vec::new();
        let mut views = Vec::new();
        for ((storage, field), offset) in fields.iter().zip(offsets) {
            let held = &sources.ranks[staging.rank as usize].storages[*storage as usize];
            let end = field.offset + field_bytes(field);
            for chunk in &held.chunks {
                let start = field.offset.max(chunk.offset);
                let stop = end.min(chunk.offset + chunk.length);
                if stop <= start {
                    continue;
                }
                spans.push(Span {
                    source: chunk
                        .binding
                        .region(chunk.key, sources.ranks[staging.rank as usize].gpu)
                        .map_err(invalid)?,
                    source_offset: chunk.storage_offset + start - chunk.offset,
                    destination: raw.region(),
                    destination_offset: offset + start - field.offset,
                    bytes: stop - start,
                    rows: 1,
                    source_pitch: 0,
                    destination_pitch: 0,
                });
            }
            views.push(field_view(Arc::clone(&raw.storage), offset, field)?);
        }
        let moved = transfer(retained, spans)?;
        let canonical = Scratch::new(
            retained,
            manifest,
            index,
            rank,
            false,
            device,
            staging.bytes,
        )?;
        convert(&fields, &views, &canonical, true, stream)?;
        Ok((canonical, moved))
    }
}

fn fields<'a>(
    world: &'a manifest::World,
    manifest: &Manifest,
    conversion: &Conversion,
) -> Result<Vec<(u32, &'a Field)>> {
    let rank = manifest.stagings[conversion.staging as usize].rank;
    conversion
        .fields
        .iter()
        .map(|(storage, name)| {
            let field = world.ranks[rank as usize].storages[*storage as usize]
                .layout
                .fields
                .iter()
                .find(|field| field.tensor == *name)
                .ok_or(Error::State("a converted field is declared"))?;
            Ok((*storage, field))
        })
        .collect()
}

fn field_bytes(field: &Field) -> u64 {
    field.element
        * (1 + field
            .shape
            .iter()
            .zip(&field.strides)
            .map(|(size, stride)| size.saturating_sub(1) * stride)
            .sum::<u64>())
}

fn field_view(storage: Arc<Storage>, offset: u64, field: &Field) -> Result<StorageView> {
    StorageView::strided(
        storage,
        offset as usize,
        DType::parse(&field.dtype)?,
        &field
            .shape
            .iter()
            .map(|&value| value as i64)
            .collect::<Vec<_>>(),
        &field
            .strides
            .iter()
            .map(|&value| value as i64)
            .collect::<Vec<_>>(),
    )
}

fn convert(
    fields: &[(u32, &Field)],
    views: &[StorageView],
    staging: &Scratch<'_>,
    to_linear: bool,
    stream: &Stream,
) -> Result<()> {
    for (_, field) in fields {
        if !matches!(&field.encoding, Encoding::Encoded { canonical: Some(form), .. } if !form.parameters.is_null())
        {
            return Err(Error::Invalid(format!(
                "{} declares no numerical conversion parameters",
                field.tensor
            )));
        }
    }
    let description =
        serde_json::to_string(&fields.iter().map(|(_, field)| field).collect::<Vec<_>>())
            .map_err(|error| Error::Interface(error.to_string()))?;
    let values = views
        .iter()
        .map(|view| Ok(view.handle()?.to_any()))
        .collect::<Result<Vec<_>>>()?;
    let bytes = StorageView::whole(Arc::clone(&staging.storage) as Arc<dyn Backing>, DType::U8)?;
    let result = Interpreter::import("uniserve.loading.conversion.convert")?.call_values(&[
        Any::from(tvm_ffi::String::from(description)),
        Any::from(list(values)),
        bytes.handle()?.to_any(),
        Any::from(to_linear),
        Any::from(stream.raw() as i64),
    ]);
    // Even a failed numerical call may have enqueued writes. No transfer or
    // release may race those writes, and the serving stream is not involved.
    let completed = stream.synchronize();
    result?;
    completed?;
    Ok(())
}

fn remap(region: &mut Region, expected: Key, scratch: &Scratch<'_>) -> Result<()> {
    match region {
        Region::Tensor { key, .. } if *key == expected => {
            *region = scratch.region();
            Ok(())
        }
        _ => Err(Error::State("a canonical copy names its staging key")),
    }
}

fn transfer(retained: &Retained, spans: Vec<Span>) -> Result<u64> {
    let bytes = spans.iter().map(|span| span.bytes * span.rows).sum();
    if !spans.is_empty() {
        let task = retained.transfer(spans, Priority::Background, retained.world())?;
        retained.transferred(task)?;
    }
    Ok(bytes)
}

fn invalid(error: manifest::Error) -> Error {
    Error::Interface(error.to_string())
}

/// A staging allocation is retained only for the synchronous operation that
/// borrows it. Transfers complete before this owner releases its key.
struct Scratch<'a> {
    retained: &'a Retained,
    key: Key,
    gpu: [u8; 16],
    storage: Arc<Storage>,
}

impl<'a> Scratch<'a> {
    fn new(
        retained: &'a Retained,
        manifest: &Manifest,
        index: u32,
        rank: u32,
        raw: bool,
        device: DeviceId,
        bytes: u64,
    ) -> Result<Self> {
        let mut digest = Sha256::new();
        digest.update(b"uniserve-weight-conversion\0");
        digest.update(manifest.staging_key(index).0);
        digest.update(retained.world().to_le_bytes());
        digest.update(rank.to_le_bytes());
        digest.update(retained.scope().group.as_bytes());
        digest.update([u8::from(raw)]);
        let key = Key(digest.finalize().into());
        let gpu = retained.gpu(device)?;
        let (storage, descriptor) = Storage::retainable(device, bytes as usize, &[])?;
        retained.call("retaining canonical conversion scratch", |client| {
            client.put(
                vec![TensorRecord {
                    key,
                    name: "weight conversion staging".into(),
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
                        bytes: storage.bytes() as u64,
                        segment: None,
                        fabric: None,
                    },
                    descriptor.as_fd(),
                )],
            )
        })?;
        retained.use_keys([key]);
        Ok(Self {
            retained,
            key,
            gpu,
            storage,
        })
    }

    fn region(&self) -> Region {
        Region::Tensor {
            key: self.key,
            at: Some(Placement::Device { gpu: self.gpu }),
        }
    }
}

impl Drop for Scratch<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.retained.forget(&[self.key]) {
            tracing::error!(%error, "releasing canonical conversion staging");
        }
    }
}
