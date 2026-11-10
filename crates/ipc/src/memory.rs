//! The memory operations a scheduler runs on every rank of its workers.
//!
//! Each operation runs at a boundary where no batch of the scheduler runs:
//! the scheduler drains its batches, hands the operation to its workers
//! ([`crate::execution::Batches::operate`]), and reads one reply per rank.
//! An operation is a typed message: each message here is the payload of one
//! [`EpochOperation`] variant, and its [`Operation`] impl states what each
//! rank answers. A rank answers one JSON text per rank ([`answers`]), which
//! the scheduler reads back as the operation's reply type ([`replies`]), so
//! a rank and its scheduler agree on the message types, not on the text.
//!
//! | Operation | What every rank does | Reply |
//! | --- | --- | --- |
//! | [`CacheStorage`] | reports its KV cache segments and its GPU's ledger | [`CacheStorageReport`] |
//! | [`ReleaseCache`] | gives back its cache segments from one on | [`Held`] |
//! | [`RestoreCache`] | creates given-back segments again | [`Held`] |
//! | [`Sleep`] | gives back every segment and its graph storage | nothing |
//! | [`Wake`] | restores what a sleep gave back | [`Held`] |
//! | [`UpdateWeights`] | puts the weights loaded for an update in place | [`WeightsUpdated`] |
//! | [`AdoptCache`] | maps the segments a predecessor lent writable | [`CacheAdopted`] |
//! | [`PublishCache`] | publishes the copied pages of a cache warming | nothing |
//! | [`CacheManifest`] | describes where the pages of a cache warming lie | `Option<storage::manifest::cache::Rank>` |

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::execution::EpochOperation;

/// A memory operation: a message that is one [`EpochOperation`], and what
/// each rank answers.
pub trait Operation: Into<EpochOperation> {
    /// What one rank answers.
    type Reply: Serialize + DeserializeOwned;
}

/// The replies of the ranks to an operation `O`, one JSON text each.
///
/// # Errors
///
/// When a reply is no reply of the operation.
pub fn replies<O: Operation>(answers: &[String]) -> serde_json::Result<Vec<O::Reply>> {
    answers
        .iter()
        .map(|answer| serde_json::from_str(answer))
        .collect()
}

/// The JSON texts of `replies`, one per rank, as a worker answers them.
///
/// # Errors
///
/// When a reply does not serialize.
pub fn answers<R: Serialize>(replies: &[R]) -> serde_json::Result<Vec<String>> {
    replies.iter().map(serde_json::to_string).collect()
}

/// States that each message is the payload of the [`EpochOperation`]
/// variant of its name, and the reply each rank answers it with.
macro_rules! operations {
    ($($message:ident => $reply:ty;)*) => {
        $(
            impl Operation for $message {
                type Reply = $reply;
            }

            impl From<$message> for EpochOperation {
                fn from(message: $message) -> Self {
                    Self::$message(message)
                }
            }
        )*
    };
}

operations! {
    CacheStorage => CacheStorageReport;
    ReleaseCache => Option<Held>;
    RestoreCache => Option<Held>;
    Sleep => ();
    Wake => Option<Held>;
    UpdateWeights => WeightsUpdated;
    AdoptCache => CacheAdopted;
    PublishCache => ();
    CacheManifest => Option<storage::manifest::cache::Rank>;
}

/// Report the rank's KV cache storage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheStorage;

/// What a rank reports of its KV cache storage and its GPU's ledger.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CacheStorageReport {
    /// The segments of each cache plane, in unit order; empty when the
    /// planes are not split.
    #[serde(default)]
    pub segments: Vec<crate::CacheSegment>,
    /// Segments the rank holds: the first given back.
    #[serde(default)]
    pub held: u32,
    /// Device bytes of one unit across the rank's planes.
    #[serde(default)]
    pub unit_bytes: u64,
    /// Free bytes of the rank's device, which bound what the cache creates
    /// beside processes outside the deployment.
    #[serde(default)]
    pub free_bytes: u64,
    /// The rank's host.
    #[serde(default)]
    pub host: String,
    /// The rank's scope and epoch, which name its entries in the ledger.
    #[serde(default)]
    pub scope: Option<storage::Scope>,
    #[serde(default)]
    pub epoch: Option<u64>,
    /// The deployment's ledger of the cache's GPU; `None` for a rank that
    /// keeps its storage itself, which no other epoch shares.
    #[serde(default)]
    pub ledger: Option<storage::GpuLedger>,
}

impl CacheStorageReport {
    /// The first unit of segment `index`.
    pub fn first_unit(&self, index: u32) -> u32 {
        self.segments
            .iter()
            .take(index as usize)
            .map(|segment| segment.units)
            .sum()
    }

    /// The device bytes of segment `index`: what the ledger books for it,
    /// or its units' bytes for a segment not created.
    pub fn segment_bytes(&self, index: usize) -> u64 {
        let segment = &self.segments[index];
        let booked = self.ledger.as_ref().map_or(0, |ledger| {
            ledger.bytes(|entry| {
                entry.item == storage::Item::CacheSegment
                    && entry.scope == self.scope
                    && entry.segment
                        == Some(storage::Segment {
                            epoch: segment.epoch,
                            ordinal: segment.ordinal,
                        })
            })
        });
        if booked > 0 {
            booked
        } else {
            u64::from(segment.units) * self.unit_bytes
        }
    }
}

/// The segments a rank's planes hold after an operation; a rank whose
/// planes are not split answers none (`Option<Held>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Held {
    pub held: u32,
}

/// Give back the cache segments from `from` on. No request holds a unit of
/// them: the scheduler withdrew them and waited.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseCache {
    pub from: u32,
}

/// Create the first `segments` given-back segments again, or all of them
/// when `None`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RestoreCache {
    #[serde(default)]
    pub segments: Option<u32>,
}

/// Give back every segment and the graph storage.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sleep;

/// Restore what a [`Sleep`] gave back.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Wake;

/// Put the weights the engine loaded beside the serving batches in place.
/// The update that keeps the requests in flight with their KV cache differs
/// only in what the scheduler does around it, so the ranks need no field.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpdateWeights;

/// What a rank answers of a weight update.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightsUpdated {
    /// The rank's worker group and rank; `None` from a worker of no
    /// storage scope.
    #[serde(default)]
    pub group: Option<String>,
    #[serde(default)]
    pub rank: Option<u32>,
    /// The identity of the rank's new weights, which keys the trajectories
    /// they compute.
    #[serde(default)]
    pub identity: Option<String>,
    /// The page identity of each of the rank's cache groups under the new
    /// weights (`PrefixCacheInfo::identities`); empty for a rank that keeps
    /// no paged cache.
    #[serde(default)]
    pub identities: Vec<String>,
    /// The weights the update rewrote.
    #[serde(default)]
    pub moved: usize,
}

/// Take over the predecessor epoch's pages in the planes the rank maps of
/// it: map writable the units `borrowed` it lent, and copy its cache flags
/// and scales of the units `units` into the rank's own, where the rank's
/// flag and scale planes are not the predecessor's (half-open unit ranges,
/// `[start, end)` each, in the rank's planes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdoptCache {
    pub borrowed: Vec<[u32; 2]>,
    #[serde(default)]
    pub units: Vec<[u32; 2]>,
}

/// What a rank answers of an [`AdoptCache`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheAdopted {
    /// The units of the predecessor's planes the rank maps, or whose flags
    /// and scales it copied.
    #[serde(default)]
    pub inherited: u32,
}

/// Publish the pages of a cache warming whose copies into the rank's units
/// completed; the ranks then serve them as cached prefixes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublishCache {
    pub pages: Vec<storage::manifest::cache::Page>,
}

/// Describe where the pages of a cache warming lie on the rank: at their
/// device addresses, or in retained host backing when `pages.host`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheManifest {
    pub pages: storage::manifest::cache::Pages,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An operation crosses a process boundary as its variant and message,
    /// a message without fields as the variant's tag alone.
    #[test]
    fn an_operation_crosses_as_its_message() {
        for operation in [
            EpochOperation::from(ReleaseCache { from: 3 }),
            EpochOperation::from(Sleep),
            EpochOperation::from(RestoreCache { segments: None }),
        ] {
            let text = serde_json::to_string(&operation).unwrap();
            assert_eq!(
                serde_json::from_str::<EpochOperation>(&text).unwrap(),
                operation
            );
        }
        assert_eq!(
            serde_json::to_value(EpochOperation::from(Sleep)).unwrap(),
            serde_json::json!({ "operation": "sleep" })
        );
    }

    /// The replies cross as the types they are, whatever field a reply
    /// leaves out.
    #[test]
    fn replies_cross_typed() {
        let held = answers(&[Some(Held { held: 2 }), None]).unwrap();
        assert_eq!(
            replies::<ReleaseCache>(&held).unwrap(),
            [Some(Held { held: 2 }), None]
        );
        // A worker without a storage scope answers no group.
        let updated = replies::<UpdateWeights>(&[r#"{"group":null}"#.to_owned()]).unwrap();
        assert_eq!(updated, [WeightsUpdated::default()]);
        assert!(replies::<CacheStorage>(&["7".to_owned()]).is_err());
    }
}
