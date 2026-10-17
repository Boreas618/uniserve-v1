//! Exports of products to other ranks and workers.

use super::*;
use crate::storage::memory::StorageView;
use crate::stream::Issue;
use crate::transfer::exports::{self, Exports, Location};
use crate::transfer::{Retirement, Spans, Transports};
use crate::{Error, Result};
use std::ops::Range;
use std::sync::Arc;
use uniserve_ipc::{
    BufferId, Call, CallKind, FeatureKind, MediaCall, TensorExport, TensorRef, TensorTransfer,
    TransferHandle,
};

/// A product's export while its issuer's work is outstanding: what the
/// transfer record says about the value, and each exported view's
/// locations ([`ProductStore::located`]).
pub struct ProductExport {
    pub(super) product: TensorRef,
    pub(super) shape: Vec<u64>,
    pub(super) height: u32,
    pub(super) width: u32,
    pub(super) value_range: String,
    pub(super) source_kind: Option<FeatureKind>,
    pub(super) exports: Vec<Exports>,
}

impl ProductExport {
    /// Revoke the locations of an export that is not published, once the
    /// issuer it was made with performed its work.
    pub(crate) fn revoke(self) {
        self.exports.into_iter().for_each(Exports::revoke);
    }
}

pub(super) fn same_locations(left: &[Location], right: &[Location]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.1 == right.1)
}

impl ProductStore {
    /// Retain an export's physical retirement on its source buffer.
    pub(crate) fn retain_export(
        &self,
        write: &ProductBuffer,
        retirement: Arc<Retirement>,
    ) -> Result<()> {
        let state = lock(&self.state);
        state.require_buffer(write)?;
        lock(write).retain_export(retirement);
        Ok(())
    }

    /// Check that `buffer` registers no export locations other than
    /// `locations`.
    pub(crate) fn validate_exports(&self, buffer: BufferId, locations: &[Location]) -> Result<()> {
        match lock(&self.exports).get(&buffer) {
            Some(existing) if !same_locations(existing, locations) => Err(Error::Invalid(
                "buffer already has different export locations".into(),
            )),
            _ => Ok(()),
        }
    }

    /// Record a committed product's export locations, revoked when the
    /// buffer is released. A buffer keeps one set of locations.
    pub(crate) fn register_exports(
        &self,
        buffer: BufferId,
        locations: Vec<Location>,
    ) -> Result<()> {
        let mut exports = lock(&self.exports);
        if let Some(existing) = exports.get(&buffer) {
            if !same_locations(existing, &locations) {
                return Err(Error::Invalid(
                    "buffer already has different export locations".into(),
                ));
            }
            return Ok(());
        }
        exports.insert(buffer, locations);
        Ok(())
    }

    /// Export a produced write through the transports reaching the call's
    /// consumers: its whole value, or each of `regions` (in value
    /// coordinates). Each export's retirement is retained on the buffer. On
    /// failure every location accepted by this call is revoked; on success
    /// the caller owns revocation of the returned locations until commit.
    #[expect(
        clippy::too_many_arguments,
        reason = "one product export's complete inputs"
    )]
    pub fn export_buffer(
        &self,
        call: &Call,
        product: &TensorRef,
        write: &ProductBuffer,
        transports: &Transports,
        host: bool,
        regions: Option<&[Vec<Range<u64>>]>,
        stream: impl Issue,
    ) -> Result<ProductExport> {
        let (feature, logical_shape, value_shape, extent) = {
            let value = lock(write);
            (
                value.feature,
                value
                    .logical_shape
                    .iter()
                    .map(|&extent| extent as u64)
                    .collect::<Vec<_>>(),
                value
                    .value_shape
                    .iter()
                    .map(|&extent| extent as u64)
                    .collect::<Vec<_>>(),
                value.extent as u64,
            )
        };
        let (view, region, metadata) = {
            let views = write.views();
            (views.view.clone(), views.region.clone(), views.metadata)
        };
        let value = value_view(&view, &value_shape, extent)?;

        let (height, width) = metadata.map_or((0, 0), Metadata::dimensions);
        let value_range = match metadata {
            Some(Metadata::Image {
                value_range: Some(range),
                ..
            }) => range.name().to_owned(),
            _ => String::new(),
        };
        let source_kind = if feature {
            if call.code == CallKind::Media(MediaCall::VisionEncoding)
                || !call.vision_inputs.is_empty()
            {
                Some(FeatureKind::Vision)
            } else if call.code == CallKind::Media(MediaCall::LatentEncoding)
                || call.latent_feature_input.is_some()
            {
                Some(FeatureKind::Latent)
            } else {
                return Err(Error::Invalid(
                    "encoder transfer has no feature source".into(),
                ));
            }
        } else {
            if height == 0 && !value_range.is_empty() {
                return Err(Error::Invalid(
                    "non-image tensor carries an image range".into(),
                ));
            }
            None
        };

        // A shard exports its logical shape and the bound region's offset.
        let (shape, origin) = match &region {
            Some(region) => (
                logical_shape,
                region.iter().map(|range| range.start).collect(),
            ),
            None => (value_shape.clone(), vec![0; value_shape.len()]),
        };
        if !product.shape_bound.contains_shape(&shape) {
            return Err(Error::Invalid(
                "product transfer changes its declared representation".into(),
            ));
        }

        let consumers: Vec<usize> = call
            .consumer_slots
            .iter()
            .map(|&slot| slot as usize)
            .collect();
        let mut exports: Vec<Exports> = Vec::new();
        let result = (|| -> Result<()> {
            let views: Vec<(StorageView, Vec<u64>)> = match regions {
                Some(regions) => regions
                    .iter()
                    .map(|requested| {
                        if !within(requested, &value_shape) {
                            return Err(Error::Invalid(
                                "consumer region exceeds the logical tensor".into(),
                            ));
                        }
                        let offset = origin
                            .iter()
                            .zip(requested)
                            .map(|(&base, range)| base + range.start)
                            .collect();
                        Ok((value.region(requested)?, offset))
                    })
                    .collect::<Result<_>>()?,
                None => vec![(value.clone(), origin.clone())],
            };
            for (view, offset) in views {
                exports.push(exports::export(
                    transports,
                    &Spans::single(view)?,
                    Some(&offset),
                    &consumers,
                    host,
                    stream,
                    |retirement| self.retain_export(write, retirement),
                )?);
            }
            Ok(())
        })();
        if let Err(error) = result {
            exports.into_iter().for_each(Exports::revoke);
            return Err(error);
        }

        Ok(ProductExport {
            product: product.clone(),
            shape,
            height,
            width,
            value_range,
            source_kind,
            exports,
        })
    }

    /// The transfer record and the locations of `export`, the export of
    /// `write`'s product.
    ///
    /// The caller keeps `export` until the issuer it was made with
    /// performed its work and calls this only then: at once after a
    /// stream, from the job's completion on after a recorder. Nothing of
    /// the export is known to another process before. On failure every
    /// location of the export is revoked; on success the caller owns
    /// revocation of the returned locations until commit.
    pub fn located(
        &self,
        write: &ProductBuffer,
        export: ProductExport,
    ) -> Result<(TensorExport, Vec<Location>)> {
        let ProductExport {
            product,
            shape,
            height,
            width,
            value_range,
            source_kind,
            exports,
        } = export;

        let mut locations: Vec<Location> = Vec::new();
        let mut exports = exports.into_iter();
        let result = exports.by_ref().try_for_each(|export| {
            locations.extend(export.locate(|retirement| self.retain_export(write, retirement))?);
            Ok(())
        });
        if let Err(error) = result {
            if let Err(cleanup) = exports::release(&locations) {
                tracing::warn!(target: "uniserve_worker.storage", "failed to revoke a failed product export: {cleanup}");
            }
            exports.for_each(|export| export.revoke());
            return Err(error);
        }

        let tensor = TensorTransfer {
            shape,
            locations: locations
                .iter()
                .map(|(_, locator)| locator.clone())
                .collect(),
        };
        let value = match source_kind {
            Some(payload_kind) => TransferHandle::Encoder {
                height,
                width,
                payload_kind,
                tensor,
            },
            None => TransferHandle::DeviceProduct {
                height,
                width,
                value_range,
                tensor,
            },
        };
        Ok((TensorExport { product, value }, locations))
    }

    /// Revoke every registered export location (shutdown); the transports
    /// retire them once their readers finished.
    pub(crate) fn revoke_exports(&self) -> Result<()> {
        let revoked: Vec<_> = lock(&self.exports)
            .drain()
            .flat_map(|(_, locations)| locations)
            .collect();
        exports::release(&revoked)
    }
}
