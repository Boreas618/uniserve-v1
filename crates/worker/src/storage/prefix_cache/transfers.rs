//! The prefix cache's transfer directory: the cache transfers resident on
//! this rank and the bases incremental transfers build on.
//!
//! A transfer's descriptor is resident from its batch's commit until its
//! producer call is released; the bases (the latest extent sent to, or
//! installed from, each destination of a request) outlive descriptors and
//! retire with their request. Validation of a batch's transfers precedes
//! any change, so a rejected batch leaves the directory unchanged.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ipc::CallId;
use ipc::{BufferId, CacheTransfer, RequestKey};

use crate::{BlockTables, Error, Result};

type Destinations = HashMap<(RequestKey, String), (BufferId, u32)>;

/// Resident cache transfers and the bases of incremental transfers.
#[derive(Default)]
pub(crate) struct TransferBases {
    resident: HashMap<BufferId, Arc<CacheTransfer>>,
    destination_bases: Destinations,
    installed_bases: Destinations,
}

impl TransferBases {
    pub(crate) fn resident(&self, buffer: BufferId) -> Option<&Arc<CacheTransfer>> {
        self.resident.get(&buffer)
    }

    /// Borrow a conditioning export covered by the request's installed keys and values.
    pub(crate) fn validate_conditioning(
        &self,
        request: RequestKey,
        buffer: BufferId,
        slot: u32,
        visible: u64,
        tables: &BlockTables,
    ) -> Result<&Arc<CacheTransfer>> {
        let export = self
            .resident(buffer)
            .ok_or_else(|| Error::Invalid("cache export buffer is not resident".into()))?;
        if buffer.owner != request {
            return Err(Error::Invalid(
                "cache conditioning buffer belongs to another request".into(),
            ));
        }
        if visible < u64::from(export.exported_extent)
            || tables.allocated_length(slot) < export.exported_extent
        {
            return Err(Error::Invalid(
                "cache conditioning allocation disagrees with its export".into(),
            ));
        }
        for group in 0..tables.groups().len() as u32 {
            tables.table(slot, group)?;
        }
        Ok(export)
    }

    /// The latest extent sent to this destination, retained even after its
    /// descriptor is released. The receiver already owns that prefix.
    pub(crate) fn destination_base(
        &self,
        request: RequestKey,
        destination: &str,
    ) -> Option<(BufferId, u32)> {
        self.destination_bases
            .get(&(request, destination.to_owned()))
            .copied()
    }

    /// Check the destination's accepted prefix before copying an incremental
    /// transfer into its assigned physical pages.
    pub(crate) fn validate_install(&self, transfer: &CacheTransfer) -> Result<()> {
        let installed = self
            .installed_bases
            .get(&(transfer.source.owner, transfer.destination.clone()))
            .copied();
        validate_base(transfer, installed)
    }

    /// Check all touched versions before any resource owner commits. Updates
    /// to one destination must form a chain in batch order; rejection leaves
    /// both resident descriptors and accepted bases unchanged.
    pub(crate) fn validate_exports(
        &self,
        exports: &[(BufferId, CacheTransfer)],
        installations: &[(BufferId, BufferId, CacheTransfer)],
    ) -> Result<()> {
        let mut resident = HashMap::new();
        let mut destination_bases = Destinations::new();
        let mut installed_bases = Destinations::new();

        for (buffer, transfer) in exports {
            if *buffer != transfer.source {
                return Err(Error::Invalid(
                    "cache export buffer differs from its source".into(),
                ));
            }
            let existing = resident
                .get(buffer)
                .copied()
                .or_else(|| self.resident(*buffer).map(AsRef::as_ref));
            if existing.is_some_and(|existing| existing != transfer) {
                return Err(Error::Invalid(
                    "cache export conflicts with its resident buffer".into(),
                ));
            }

            validate_next_base(transfer, &self.destination_bases, &mut destination_bases)?;
            resident.insert(*buffer, transfer);
        }

        for (source, installed, transfer) in installations {
            if *source != transfer.source || installed.owner != source.owner {
                return Err(Error::Invalid(
                    "installed cache buffer does not belong to its source".into(),
                ));
            }

            validate_next_base(transfer, &self.installed_bases, &mut installed_bases)?;
        }
        Ok(())
    }

    /// Commit an already checked batch. The executor must leave this
    /// directory unchanged between validation and the shared resource commit.
    pub(crate) fn apply_exports(
        &mut self,
        exports: Vec<(BufferId, CacheTransfer)>,
        installations: Vec<(BufferId, BufferId, CacheTransfer)>,
    ) {
        for (buffer, transfer) in exports {
            self.destination_bases.insert(
                (buffer.owner, transfer.destination.clone()),
                (transfer.source, transfer.exported_extent),
            );
            self.resident.insert(buffer, Arc::new(transfer));
        }

        for (source, installed, transfer) in installations {
            self.installed_bases.insert(
                (installed.owner, transfer.destination.clone()),
                (transfer.source, transfer.exported_extent),
            );
            let transfer = Arc::new(transfer);
            self.resident.insert(source, Arc::clone(&transfer));
            self.resident.insert(installed, transfer);
        }
    }

    /// Remove consumed descriptors and return their buffers for revocation.
    /// Physical exports and transfer bases have independent lifetimes.
    pub(crate) fn release_calls(&mut self, calls: &[(RequestKey, CallId)]) -> Vec<BufferId> {
        let calls: HashSet<_> = calls.iter().copied().collect();
        let mut released = Vec::new();
        self.resident.retain(|buffer, _| {
            if calls.contains(&(buffer.owner, buffer.producer_call_id)) {
                released.push(*buffer);
                false
            } else {
                true
            }
        });
        released
    }

    /// Forget a retired request epoch after its physical accesses have
    /// drained; other epochs of its request id keep their entries.
    pub(crate) fn drop_request(&mut self, request: RequestKey) {
        self.resident.retain(|buffer, _| buffer.owner != request);
        self.destination_bases
            .retain(|(owner, _), _| *owner != request);
        self.installed_bases
            .retain(|(owner, _), _| *owner != request);
    }

    /// Forget transfer descriptions and bases when the cache owner closes.
    /// Physical access retirement is independent and must already be drained.
    pub(crate) fn clear(&mut self) {
        self.resident.clear();
        self.destination_bases.clear();
        self.installed_bases.clear();
    }
}

fn validate_base(transfer: &CacheTransfer, current: Option<(BufferId, u32)>) -> Result<()> {
    let current = current.map_or((None, 0), |(buffer, extent)| (Some(buffer), extent));
    if current != (transfer.base, transfer.base_extent) {
        return Err(Error::Invalid(
            "cache transfer base does not match destination".into(),
        ));
    }
    Ok(())
}

fn validate_next_base(
    transfer: &CacheTransfer,
    resident: &Destinations,
    pending: &mut Destinations,
) -> Result<()> {
    let key = (transfer.source.owner, transfer.destination.clone());
    validate_base(
        transfer,
        pending.get(&key).or_else(|| resident.get(&key)).copied(),
    )?;
    pending.insert(key, (transfer.source, transfer.exported_extent));
    Ok(())
}

#[cfg(test)]
mod tests {
    use ipc::{CallId, RequestId};

    use super::*;

    fn buffer(request: u64) -> BufferId {
        BufferId {
            owner: RequestKey::new(1, RequestId(request), 1),
            producer_call_id: CallId::new(1, 0),
            output_index: 0,
            generation: 1,
        }
    }

    fn empty_transfer(source: BufferId, base: Option<BufferId>) -> CacheTransfer {
        CacheTransfer {
            groups: Vec::new(),
            source,
            destination: "decoder".into(),
            base,
            base_extent: 0,
            exported_extent: 0,
            compute_dtype: "bfloat16".into(),
        }
    }

    #[test]
    fn rejected_transfer_chains_leave_the_accepted_base_and_resident_values_unchanged() -> Result<()>
    {
        let mut cache = TransferBases::default();
        let first = empty_transfer(buffer(1), None);
        let second = empty_transfer(
            BufferId {
                producer_call_id: CallId::new(2, 0),
                ..first.source
            },
            Some(first.source),
        );
        let mut third = empty_transfer(
            BufferId {
                producer_call_id: CallId::new(3, 0),
                ..first.source
            },
            Some(first.source),
        );
        cache.validate_exports(&[(first.source, first.clone())], &[])?;
        cache.apply_exports(vec![(first.source, first.clone())], vec![]);

        // Even an empty suffix advances the accepted transfer base. The
        // third update cannot skip the second update in the same batch.
        assert!(
            cache
                .validate_exports(
                    &[
                        (second.source, second.clone()),
                        (third.source, third.clone())
                    ],
                    &[],
                )
                .is_err()
        );
        assert_eq!(
            cache.destination_base(first.source.owner, "decoder"),
            Some((first.source, 0))
        );
        assert_eq!(
            cache.resident(first.source).map(AsRef::as_ref),
            Some(&first)
        );
        assert_eq!(cache.resident(second.source), None);

        third.base = Some(second.source);
        let batch = vec![
            (second.source, second.clone()),
            (third.source, third.clone()),
        ];
        cache.validate_exports(&batch, &[])?;
        cache.apply_exports(batch, vec![]);
        cache.release_calls(&[
            (first.source.owner, first.source.producer_call_id),
            (second.source.owner, second.source.producer_call_id),
        ]);
        assert_eq!(cache.resident(first.source), None);
        assert_eq!(cache.resident(second.source), None);
        assert_eq!(
            cache.resident(third.source).map(AsRef::as_ref),
            Some(&third)
        );
        assert_eq!(
            cache.destination_base(first.source.owner, "decoder"),
            Some((third.source, 0))
        );
        Ok(())
    }

    #[test]
    fn installed_aliases_and_transfer_bases_retire_with_their_request() -> Result<()> {
        let mut cache = TransferBases::default();
        let first = empty_transfer(buffer(1), None);
        let local = BufferId {
            producer_call_id: CallId::new(5, 0),
            ..first.source
        };
        let other = empty_transfer(buffer(2), None);
        let installs = vec![
            (first.source, local, first.clone()),
            (other.source, other.source, other.clone()),
        ];
        cache.validate_install(&first)?;
        cache.validate_exports(&[], &installs)?;
        cache.apply_exports(vec![], installs);
        cache.release_calls(&[(first.source.owner, first.source.producer_call_id)]);
        assert_eq!(cache.resident(first.source), None);
        assert_eq!(cache.resident(local).map(AsRef::as_ref), Some(&first));

        let next = empty_transfer(
            BufferId {
                producer_call_id: CallId::new(2, 0),
                ..first.source
            },
            Some(first.source),
        );
        cache.validate_install(&next)?;
        assert!(cache.validate_install(&first).is_err());
        cache.drop_request(first.source.owner);
        cache.validate_install(&first)?;
        assert!(cache.validate_install(&next).is_err());
        assert_eq!(cache.resident(local), None);
        assert_eq!(
            cache.resident(other.source).map(AsRef::as_ref),
            Some(&other)
        );
        assert!(cache.validate_install(&other).is_err());
        Ok(())
    }
}
