//! A second load of a serving model (`specs/worker.md` §7.6).
//!
//! An in-place weight update and an expert redistribution are the same operation seen from a
//! rank: the model is loaded again against another target, and each weight whose key moved takes
//! the address of the weight it replaces. The load is the model's own, through `nn::WeightLoader`,
//! so nothing here knows a model's tensors, and a new weight reaches the Storage Server by the path
//! every weight takes.
//!
//! A load runs twice. [`Walk`] is the plan: it answers every key from nothing, so the model's load
//! states what it would read without moving a byte, and a checkpoint that loads other tensors is
//! refused before any storage is touched. [`Reloading`] is the load itself, beside the serving
//! weights: a key the rank holds already costs nothing, so an update publishes exactly what
//! changed.

use std::collections::BTreeMap;
use std::sync::Arc;

use nn::weights::WeightLayout;
use nn::{Backing, DType, DevicePtr, RetainedWeights, Shape, WeightKey};

use crate::memory::allocation::Backing as _;
use crate::memory::retained::Retained;
use crate::memory::weights::{LoadedWeight, WeightStore, retained_key};
use crate::{Error, Result};

/// The region a [`Walk`] answers: wide enough for any view, and read by nothing.
///
/// A model's load builds its layers from views and launches nothing, so a walk's model is a
/// structure of addresses nobody dereferences, as a shapes-only loader's is.
fn placeholder() -> Backing {
    Backing::new(DevicePtr::new(0), u64::MAX / 2)
}

/// A model's load with no storage behind it: the weights it would reach, in order.
#[derive(Default)]
pub struct Walk {
    log: Vec<LoadedWeight>,
}

impl Walk {
    /// What the load reached.
    pub fn into_log(self) -> Vec<LoadedWeight> {
        self.log
    }
}

impl RetainedWeights for Walk {
    fn retained(&mut self, key: &WeightKey) -> nn::Result<Option<Backing>> {
        self.log.push(LoadedWeight {
            key: key.clone(),
            writable: false,
        });
        Ok(Some(placeholder()))
    }

    fn publish(
        &mut self,
        key: &WeightKey,
        _dtype: DType,
        _shape: Shape,
        _bytes: &[u8],
    ) -> nn::Result<Backing> {
        self.log.push(LoadedWeight {
            key: key.clone(),
            writable: false,
        });
        Ok(placeholder())
    }

    fn writable(
        &mut self,
        key: &WeightKey,
        _dtype: DType,
        _shape: Shape,
        _bytes: &[u8],
    ) -> nn::Result<Backing> {
        self.log.push(LoadedWeight {
            key: key.clone(),
            writable: true,
        });
        Ok(placeholder())
    }
}

/// How two loads of one model may differ in what they read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matching {
    /// Another checkpoint of the model: the same tensors from other files.
    SameTensors,
    /// Another arrangement of the same checkpoint: a stack may hold its parts in another order.
    Rearranged,
}

/// One weight an update replaces: the key that serves, and the key that takes its place.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightPair {
    pub old: WeightKey,
    pub new: WeightKey,
}

/// The weights of `target` that replace the ones of `serving`, position by position.
///
/// A load's order is the identity of its weights (`LoadedWeight`), so the two logs are compared in
/// step, and whatever differs beside the files is a load of other tensors. `Err` is a sentence
/// naming the first difference.
pub fn paired(
    serving: &[LoadedWeight],
    target: &[LoadedWeight],
    matching: Matching,
) -> std::result::Result<Vec<WeightPair>, String> {
    if serving.len() != target.len() {
        return Err(format!(
            "it loads {} weights and the served model loads {}",
            target.len(),
            serving.len()
        ));
    }
    let mut pairs = Vec::new();
    let mut seen: BTreeMap<[u8; 32], [u8; 32]> = BTreeMap::new();
    for (old, new) in serving.iter().zip(target) {
        let (from, to) = (&old.key, &new.key);
        let tensor = from
            .region
            .tensors
            .first()
            .map_or("a computed weight", String::as_str);
        if old.writable != new.writable {
            return Err(format!(
                "{tensor} is a table on one side and a weight on the other"
            ));
        }
        if from.dtype != to.dtype {
            return Err(format!(
                "{tensor} is {} and the served {}",
                to.dtype, from.dtype
            ));
        }
        if from.layout != to.layout {
            return Err(format!(
                "{tensor} is laid out {} and the served {}",
                to.layout, from.layout
            ));
        }
        if from.region.shape != to.region.shape
            || from.region.start != to.region.start
            || from.region.stride != to.region.stride
        {
            return Err(format!(
                "{tensor} is {:?} and the served {:?}",
                to.region.shape.dims(),
                from.region.shape.dims()
            ));
        }
        if from.provenance.is_some() != to.provenance.is_some() {
            return Err(format!(
                "{tensor} is computed on one side and read on the other"
            ));
        }
        let same_tensors = match matching {
            Matching::SameTensors => from.region.tensors == to.region.tensors,
            Matching::Rearranged => {
                let mut left = from.region.tensors.clone();
                let mut right = to.region.tensors.clone();
                left.sort_unstable();
                right.sort_unstable();
                left == right
            }
        };
        if !same_tensors {
            return Err(format!(
                "the weight the served model reads from {tensor} is read from {:?}",
                to.region.tensors.first()
            ));
        }
        if old.writable || from == to {
            continue;
        }
        // A weight two layers share is reached twice and replaced once; two replacements of one
        // weight are two weights the served model reads as one, and no single address holds both.
        let (from_key, to_key) = (retained_key(from), retained_key(to));
        match seen.get(&from_key) {
            Some(known) if *known == to_key => continue,
            Some(_) => {
                return Err(format!(
                    "{tensor} is one weight of the served model and two of this one"
                ));
            }
            None => {
                seen.insert(from_key, to_key);
            }
        }
        pairs.push(WeightPair {
            old: from.clone(),
            new: to.clone(),
        });
    }
    Ok(pairs)
}

/// The slot of `old` each slot of `new` holds, when `new` is `old`'s stack in another slot order.
///
/// A dense expert stack is one contiguous allocation of whole experts, and its key names the parts
/// and their files in slot order, so the two keys say exactly which slot moved where. A part whose
/// file differs is another checkpoint's tensor of the same name, whose bytes the device does not
/// hold. Any other weight answers `None`: its bytes are the checkpoint's to give.
fn slots_of(old: &WeightKey, new: &WeightKey) -> Option<Vec<usize>> {
    if old.layout != new.layout
        || !(old.layout == WeightLayout::UP_GATE || old.layout == WeightLayout::DOWN)
        || old.region.shape.rank() == 0
    {
        return None;
    }
    let slots = usize::try_from(old.region.shape.dim(0)).ok()?;
    if slots == 0 || old.region.tensors.len() % slots != 0 {
        return None;
    }
    let width = old.region.tensors.len() / slots;
    let parts_of = |key: &WeightKey| -> Option<Vec<Vec<(String, nn::FileIdentity)>>> {
        if key.files.len() != key.region.tensors.len() {
            return None;
        }
        let parts: Vec<(String, nn::FileIdentity)> = key
            .region
            .tensors
            .iter()
            .cloned()
            .zip(key.files.iter().cloned())
            .collect();
        Some(parts.chunks(width).map(<[_]>::to_vec).collect())
    };
    let (old_slots, new_slots) = (parts_of(old)?, parts_of(new)?);
    let where_old: BTreeMap<&[(String, nn::FileIdentity)], usize> = old_slots
        .iter()
        .enumerate()
        .map(|(slot, parts)| (parts.as_slice(), slot))
        .collect();
    new_slots
        .iter()
        .map(|parts| where_old.get(parts.as_slice()).copied())
        .collect()
}

/// A second load beside the serving weights, on one rank (the module's statement).
pub(crate) struct Reloading<'a> {
    store: WeightStore<Arc<Retained>>,
    /// The rank's client, which a rearranged stack is copied and published through.
    storage: Arc<Retained>,
    device: platform::DeviceId,
    /// The weights that serve, which position the load's requests.
    serving: &'a [LoadedWeight],
    /// How many stacks the device rearranged without reading the checkpoint.
    gathered: usize,
}

impl<'a> Reloading<'a> {
    pub(crate) fn new(
        storage: Arc<Retained>,
        device: platform::DeviceId,
        serving: &'a [LoadedWeight],
    ) -> Self {
        Self {
            store: WeightStore::new(Arc::clone(&storage), device),
            storage,
            device,
            serving,
            gathered: 0,
        }
    }

    /// What the load reached, how many weights it published, and how many stacks the device
    /// rearranged.
    pub(crate) fn finish(mut self) -> (Vec<LoadedWeight>, usize, usize) {
        let published = self.store.published();
        (self.store.take_log(), published, self.gathered)
    }

    /// Publishes `key`'s stack from the serving one, when it is that stack in another slot order.
    ///
    /// The bytes are on the device already, at the addresses the rank serves from, so a
    /// redistribution moves them over the device's own memory rather than reading the checkpoint
    /// again: on a stream of its own, beside the batches, which read the same bytes.
    fn gather(&mut self, key: &WeightKey) -> Result<Option<Backing>> {
        let Some(old) = self.serving.get(self.store.logged()) else {
            return Ok(None);
        };
        let Some(slots) = slots_of(&old.key, key) else {
            return Ok(None);
        };
        let Some(source) = self
            .storage
            .mapped_weight(self.device, &retained_key(&old.key))?
        else {
            return Ok(None);
        };
        let element = key.dtype.bytes().ok_or_else(|| {
            Error::Invalid(format!("a {} stack has no whole-byte element", key.dtype))
        })?;
        let length = usize::try_from(key.region.shape.numel().saturating_mul(element))
            .map_err(|_| Error::Invalid("a stack larger than this host addresses".into()))?;
        let slot = length / slots.len();
        let base = source.address();
        let allocation = source.bytes();
        let backing = self.storage.store_on_device(
            self.device,
            &retained_key(key),
            length,
            allocation,
            |target, stream| {
                for (to, from) in slots.iter().enumerate() {
                    // SAFETY: both ranges lie inside live device allocations `length` long or
                    // longer, the source mapped read-only and the destination this call's own
                    // staging, and the caller synchronizes before either can go.
                    unsafe {
                        platform::memcpy_async(
                            target + (to * slot) as u64,
                            base + (*from * slot) as u64,
                            slot,
                            stream,
                        )?;
                    }
                }
                Ok(())
            },
        )?;
        self.store.adopt(key, backing);
        self.gathered += 1;
        Ok(Some(backing))
    }
}

impl RetainedWeights for Reloading<'_> {
    fn retained(&mut self, key: &WeightKey) -> nn::Result<Option<Backing>> {
        if let Some(backing) = self.store.retained(key)? {
            return Ok(Some(backing));
        }
        self.gather(key)
            .map_err(|error| nn::Error::Unsupported(error.to_string()))
    }

    fn publish(
        &mut self,
        key: &WeightKey,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
    ) -> nn::Result<Backing> {
        self.store.publish(key, dtype, shape, bytes)
    }

    /// The serving table stays: its address is what the captured graphs read, and the model
    /// writes it again once the new weights are in place, so this load allocates none.
    fn writable(
        &mut self,
        key: &WeightKey,
        _dtype: DType,
        _shape: Shape,
        _bytes: &[u8],
    ) -> nn::Result<Backing> {
        self.store.note_table(key);
        Ok(placeholder())
    }
}

#[cfg(test)]
mod tests {
    use nn::FileIdentity;
    use nn::weights::WeightRegion;

    use super::*;

    fn identity(name: &str) -> FileIdentity {
        let directory = std::env::temp_dir()
            .join("uniserve-reload-tests")
            .join("blobs");
        std::fs::create_dir_all(&directory).expect("a temporary directory");
        let digest: String = name
            .bytes()
            .flat_map(|byte| format!("{byte:02x}").into_bytes())
            .map(char::from)
            .take(64)
            .collect();
        let path = directory.join(format!("{digest:0<64}"));
        if !path.exists() {
            std::fs::write(&path, name).expect("a temporary checkpoint file");
        }
        FileIdentity::of(&path).expect("the identity of a checkpoint file")
    }

    fn weight(file: &str, tensors: &[&str], layout: WeightLayout, slots: u64) -> LoadedWeight {
        LoadedWeight {
            key: WeightKey {
                files: tensors.iter().map(|_| identity(file)).collect(),
                dtype: DType::BF16,
                layout,
                region: WeightRegion {
                    tensors: tensors.iter().map(|tensor| (*tensor).to_owned()).collect(),
                    start: 0,
                    shape: Shape::new(&[slots, 8]),
                    stride: None,
                },
                provenance: None,
            },
            writable: false,
        }
    }

    #[test]
    fn another_checkpoint_pairs_every_weight_by_its_position() {
        let serving = vec![
            weight("a", &["norm"], WeightLayout::ROW_MAJOR, 1),
            weight("a", &["head"], WeightLayout::ROW_MAJOR, 1),
        ];
        let target = vec![
            weight("b", &["norm"], WeightLayout::ROW_MAJOR, 1),
            weight("a", &["head"], WeightLayout::ROW_MAJOR, 1),
        ];
        let pairs = paired(&serving, &target, Matching::SameTensors).expect("the same tensors");
        // The head's file did not change, so its key did not, and nothing replaces it.
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].old, serving[0].key);
        assert_eq!(pairs[0].new, target[0].key);
    }

    #[test]
    fn a_checkpoint_of_other_tensors_is_refused_by_its_first_difference() {
        let serving = vec![weight("a", &["norm"], WeightLayout::ROW_MAJOR, 1)];
        let other = vec![weight("b", &["other"], WeightLayout::ROW_MAJOR, 1)];
        let refusal = paired(&serving, &other, Matching::SameTensors).expect_err("another tensor");
        assert!(refusal.contains("norm"), "{refusal}");
        let shorter: Vec<LoadedWeight> = Vec::new();
        assert!(paired(&serving, &shorter, Matching::SameTensors).is_err());
        let mut wider = weight("b", &["norm"], WeightLayout::ROW_MAJOR, 1);
        wider.key.region.shape = Shape::new(&[2, 8]);
        assert!(paired(&serving, &[wider], Matching::SameTensors).is_err());
    }

    #[test]
    fn a_rearranged_stack_names_the_slot_each_slot_came_from() {
        let old = weight(
            "a",
            &["e0.up", "e0.gate", "e1.up", "e1.gate"],
            WeightLayout::UP_GATE,
            2,
        );
        let new = weight(
            "a",
            &["e1.up", "e1.gate", "e0.up", "e0.gate"],
            WeightLayout::UP_GATE,
            2,
        );
        assert_eq!(slots_of(&old.key, &new.key), Some(vec![1, 0]));
        assert_eq!(
            paired(
                std::slice::from_ref(&old),
                std::slice::from_ref(&new),
                Matching::Rearranged
            )
            .expect("the same parts")
            .len(),
            1
        );
        // Another checkpoint's order of the same names is not a rearrangement of the same parts
        // in an update, which reads tensors in place.
        assert!(
            paired(
                std::slice::from_ref(&old),
                std::slice::from_ref(&new),
                Matching::SameTensors
            )
            .is_err()
        );
    }

    /// A stack of another checkpoint names the same tensors from other files: its bytes are that
    /// checkpoint's, and copying the serving stack under its key would publish the old experts as
    /// the new ones.
    #[test]
    fn a_stack_of_another_checkpoint_has_no_slots_on_the_device() {
        let old = weight(
            "a",
            &["e0.up", "e0.gate", "e1.up", "e1.gate"],
            WeightLayout::UP_GATE,
            2,
        );
        let new = weight(
            "b",
            &["e0.up", "e0.gate", "e1.up", "e1.gate"],
            WeightLayout::UP_GATE,
            2,
        );
        assert_eq!(slots_of(&old.key, &new.key), None);
    }

    #[test]
    fn a_weight_that_is_no_stack_has_no_slots() {
        let old = weight("a", &["w"], WeightLayout::ROW_MAJOR, 1);
        assert_eq!(slots_of(&old.key, &old.key), None);
    }

    #[test]
    fn a_walk_states_a_load_and_its_tables_without_storage() {
        let mut walk = Walk::default();
        let weight = weight("a", &["w"], WeightLayout::ROW_MAJOR, 1).key;
        assert!(walk.retained(&weight).expect("a walk answers").is_some());
        walk.writable(&weight, DType::BF16, Shape::new(&[1]), &[0, 0])
            .expect("a walk allocates nothing");
        let log = walk.into_log();
        assert_eq!(log.len(), 2);
        assert!(!log[0].writable && log[1].writable);
    }
}
