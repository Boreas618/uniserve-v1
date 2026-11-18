//! A second load of a serving model.
//!
//! An expert redistribution is, seen from a rank, the model loaded again against another expert
//! order of the served checkpoint, each weight whose key changed taking the address of the weight
//! it replaces. The load is the model's own, through `nn::WeightLoader`, so nothing here knows a
//! model's tensors, and a changed weight reaches the Storage Server by the path every weight
//! takes.
//!
//! The second load runs twice. First over [`MetaWeights`], which holds no storage: the model's
//! load states the weights it would reach, in order, without reading a payload byte, and
//! [`paired`] compares that order with the one that serves, so a target that loads other tensors
//! is refused before any storage is touched. Then over [`Reload`], beside the serving weights: a
//! key the rank holds already costs nothing, so a redistribution publishes exactly what changed,
//! and [`WeightStore::rebind`] puts it in place at a boundary where nothing runs.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap};

use nn::{FileIdentity, RetainedWeights, Shape, Tensor, WeightKey, WeightLayout};

use crate::memory::weights::{WeightStore, refusal};
use crate::worker::{Error, Result};

/// The first address [`MetaWeights`] states a tensor at, and the address of every table a
/// [`Reload`] states: an address nothing reads, far from zero so that no argument check takes
/// a weight for a null pointer.
const META_BASE: u64 = 1 << 47;

/// Retained storage on the meta device: every weight a load asks for is a tensor with a shape
/// and an element type and no data, and the store records the order the load asked in.
///
/// A loader over this store (`nn::WeightLoader::new`) runs a model's own load without reading
/// a weight's payload, because every key is held already; what comes out is the order of a
/// load of that checkpoint ([`MetaWeights::into_order`]) and a model nothing computes with.
pub struct MetaWeights {
    /// The weights the load reached, in its order, each with whether the model writes it after
    /// the load.
    order: RefCell<Vec<(WeightKey, bool)>>,
    /// The bytes of each weight of `order`, in its order.
    lengths: RefCell<Vec<u64>>,
    /// The address the next tensor is stated at.
    next: Cell<u64>,
}

impl Default for MetaWeights {
    fn default() -> Self {
        Self::new()
    }
}

impl MetaWeights {
    /// A store no load has asked yet.
    pub fn new() -> Self {
        Self {
            order: RefCell::new(Vec::new()),
            lengths: RefCell::new(Vec::new()),
            next: Cell::new(META_BASE),
        }
    }

    /// The weights the load reached, in its order, each with whether the model writes it after
    /// the load.
    pub fn into_order(self) -> Vec<(WeightKey, bool)> {
        self.order.into_inner()
    }

    /// The weights the load reached, in its order, each with whether the model writes it after
    /// the load and its bytes: what a load of them on a device would take.
    pub fn into_loads(self) -> Vec<(WeightKey, bool, u64)> {
        self.order
            .into_inner()
            .into_iter()
            .zip(self.lengths.into_inner())
            .map(|((key, writable), length)| (key, writable, length))
            .collect()
    }

    /// Records `key` and states its tensor of `shape` elements at an address of its own.
    fn tensor(&self, key: &WeightKey, shape: &Shape, writable: bool) -> nn::Result<Tensor> {
        self.order.borrow_mut().push((key.clone(), writable));

        // Every tensor takes at least one aligned block, so no two share an address.
        let length = key
            .dtype
            .storage_bytes(shape.elem_count())
            .map_or(u64::MAX, |bytes| bytes as u64);
        self.lengths.borrow_mut().push(length);
        let bytes = length.max(1);
        let address = self.next.get();
        self.next.set(
            address.saturating_add(
                // Two tensors lie a multiple of the allocator's alignment apart, as two
                // device allocations do.
                bytes
                    .div_ceil(ops::allocator::ALIGNMENT)
                    .saturating_mul(ops::allocator::ALIGNMENT),
            ),
        );

        Ok(ops::retained(
            address,
            shape.clone(),
            key.dtype,
            &ops::meta(),
        )?)
    }
}

impl RetainedWeights for MetaWeights {
    fn retained(&self, key: &WeightKey, shape: &Shape) -> nn::Result<Option<Tensor>> {
        self.tensor(key, shape, false).map(Some)
    }

    fn publish(&self, key: &WeightKey, shape: &Shape, _bytes: &[u8]) -> nn::Result<Tensor> {
        self.tensor(key, shape, false)
    }

    fn writable(&self, key: &WeightKey, shape: &Shape, _bytes: &[u8]) -> nn::Result<Tensor> {
        self.tensor(key, shape, true)
    }
}

/// The weights of `target` that replace the ones of `serving`, position by position: each pair
/// is the key that serves and the key that takes its place.
///
/// Both are the orders of two loads of one model ([`WeightStore::finish`],
/// [`MetaWeights::into_order`]). A load's order is the identity of its weights, so the two are
/// compared in step, and whatever differs beside the files is a load of other tensors, except
/// that a stack may hold the same parts in another order (another arrangement of the same
/// checkpoint, as an expert redistribution loads).
///
/// A weight whose key did not change, and a table the model writes after its load, has no
/// pair: the first serves on as it is, and the model writes the second again.
///
/// # Errors
///
/// [`Error::Invalid`] with a sentence naming the first difference, when `target` is not a load
/// of the same tensors.
pub fn paired(
    serving: &[(WeightKey, bool)],
    target: &[(WeightKey, bool)],
) -> Result<Vec<(WeightKey, WeightKey)>> {
    if serving.len() != target.len() {
        return Err(Error::Invalid(format!(
            "it loads {} weights and the served model loads {}",
            target.len(),
            serving.len()
        )));
    }

    let mut pairs = Vec::new();
    let mut replaced: HashMap<&WeightKey, &WeightKey> = HashMap::new();
    for ((from, table), (to, to_table)) in serving.iter().zip(target) {
        let tensor = from
            .tensors
            .first()
            .map_or("a computed weight", String::as_str);

        if table != to_table {
            return Err(Error::Invalid(format!(
                "{tensor} is a table on one side and a weight on the other"
            )));
        }
        if from.dtype != to.dtype {
            return Err(Error::Invalid(format!(
                "{tensor} is {} and the served {}",
                ops::abi::name(to.dtype),
                ops::abi::name(from.dtype)
            )));
        }
        if from.layout != to.layout {
            return Err(Error::Invalid(format!(
                "{tensor} is laid out {} and the served {}",
                to.layout, from.layout
            )));
        }
        if from.shape != to.shape || from.start != to.start || from.stride != to.stride {
            return Err(Error::Invalid(format!(
                "{tensor} is {:?} and the served {:?}",
                to.shape.dims(),
                from.shape.dims()
            )));
        }
        if from.provenance.is_some() != to.provenance.is_some() {
            return Err(Error::Invalid(format!(
                "{tensor} is computed on one side and read on the other"
            )));
        }

        // The same parts, in whatever order the target's stack holds them.
        let mut left: Vec<&String> = from.tensors.iter().collect();
        let mut right: Vec<&String> = to.tensors.iter().collect();
        left.sort_unstable();
        right.sort_unstable();
        if left != right {
            return Err(Error::Invalid(format!(
                "the weight the served model reads from {tensor} is read from {:?}",
                to.tensors.first()
            )));
        }

        if *table || from == to {
            continue;
        }

        // A weight two modules share is reached twice and replaced once; two replacements of
        // one weight are two weights the served model reads as one, and no single address
        // holds both.
        match replaced.get(from) {
            Some(known) if *known == to => continue,
            Some(_) => {
                return Err(Error::Invalid(format!(
                    "{tensor} is one weight of the served model and two of this one"
                )));
            }
            None => {
                replaced.insert(from, to);
            }
        }
        pairs.push((from.clone(), to.clone()));
    }

    Ok(pairs)
}

/// The slot of `serving` each slot of `target` holds, when `target` is `serving`'s stack in
/// another slot order.
///
/// A dense expert stack is one contiguous weight of whole experts, and its key names the parts
/// and their files in slot order, so the two keys say exactly which slot went where. A part
/// whose file differs is another checkpoint's tensor of the same name, whose bytes the device
/// does not hold. Any other weight gives `None`: its bytes are the checkpoint's to give.
fn slots_of(serving: &WeightKey, target: &WeightKey) -> Option<Vec<usize>> {
    let stack = serving.layout == WeightLayout::UP_GATE || serving.layout == WeightLayout::DOWN;
    if serving.layout != target.layout || !stack {
        return None;
    }

    let slots = *serving.shape.dims().first()?;
    let parts = serving.tensors.len();
    if slots == 0 || parts == 0 || !parts.is_multiple_of(slots) {
        return None;
    }
    let width = parts / slots;

    // The parts of each slot of a stack: its tensors' names with the files they are read from.
    let slots_in = |key: &WeightKey| -> Option<Vec<Vec<(String, FileIdentity)>>> {
        if key.files.len() != key.tensors.len() {
            return None;
        }

        let parts: Vec<(String, FileIdentity)> = key
            .tensors
            .iter()
            .cloned()
            .zip(key.files.iter().cloned())
            .collect();
        Some(parts.chunks(width).map(<[_]>::to_vec).collect())
    };
    let (serving_slots, target_slots) = (slots_in(serving)?, slots_in(target)?);

    let serving_slot: BTreeMap<&[(String, FileIdentity)], usize> = serving_slots
        .iter()
        .enumerate()
        .map(|(slot, parts)| (parts.as_slice(), slot))
        .collect();
    target_slots
        .iter()
        .map(|parts| serving_slot.get(parts.as_slice()).copied())
        .collect()
}

/// A second load beside the serving weights, on one rank.
///
/// The load publishes through the rank's [`WeightStore`], which keeps the weights that serve
/// mapped at their addresses. The weight the load reaches at a position replaces the serving
/// weight of that position, so it is published where that weight lies and
/// [`WeightStore::rebind`] maps it at the serving address afterwards. The model this load
/// builds computes nothing: its weight tensors are over the addresses the changed weights were
/// published at, which the rebind unmaps, and its tables have no storage.
pub struct Reload<'a> {
    store: &'a WeightStore,
    /// The weights that serve, in their load's order, which position the load's requests.
    serving: &'a [(WeightKey, bool)],
    /// How many weights the store had published when the load began.
    published: usize,
    /// How many stacks the device rearranged without reading the checkpoint.
    rearranged: Cell<usize>,
}

impl<'a> Reload<'a> {
    /// A load through `store` beside the weights of `serving`, the order of the load that
    /// serves.
    ///
    /// # Errors
    ///
    /// [`Error::State`] when the store has a load in progress: a position of this load would
    /// count that load's weights.
    pub fn new(store: &'a WeightStore, serving: &'a [(WeightKey, bool)]) -> Result<Self> {
        if store.reached() != 0 {
            return Err(Error::State(
                "a weight store serves one load at a time, and one is in progress",
            ));
        }

        Ok(Self {
            store,
            serving,
            published: store.published(),
            rearranged: Cell::new(0),
        })
    }

    /// Ends the load: publishes the shared allocations of its packed weights and returns the
    /// weights it reached, in its order.
    ///
    /// # Errors
    ///
    /// When the Storage Server refuses an allocation.
    pub fn finish(self) -> Result<Vec<(WeightKey, bool)>> {
        let order = self.store.finish()?;

        tracing::info!(
            device = %self.store.device(),
            published = self.store.published() - self.published,
            rearranged = self.rearranged.get(),
            "a rank published what the load changes beside the weights that serve"
        );
        Ok(order)
    }

    /// The serving weight the weight this load reaches next replaces: the one at its position.
    fn replaced(&self) -> Option<&'a WeightKey> {
        self.serving.get(self.store.reached()).map(|(key, _)| key)
    }

    /// Publishes the stack of `key` from the serving one, when it is that stack in another
    /// slot order; `None` for any other weight, which the loader then reads from the
    /// checkpoint.
    fn rearrange(&self, key: &WeightKey, shape: &Shape) -> Result<Option<Tensor>> {
        let Some(replaced) = self.replaced() else {
            return Ok(None);
        };
        let Some(slots) = slots_of(replaced, key) else {
            return Ok(None);
        };

        let tensor = self
            .store
            .publish_rearranged(key, shape, replaced, &slots)?;
        if tensor.is_some() {
            self.rearranged.set(self.rearranged.get() + 1);
        }
        Ok(tensor)
    }
}

impl RetainedWeights for Reload<'_> {
    fn retained(&self, key: &WeightKey, shape: &Shape) -> nn::Result<Option<Tensor>> {
        if let Some(tensor) = self.store.retained(key, shape)? {
            return Ok(Some(tensor));
        }

        self.rearrange(key, shape)
            .map_err(|error| refusal(key, error))
    }

    fn publish(&self, key: &WeightKey, shape: &Shape, bytes: &[u8]) -> nn::Result<Tensor> {
        match self.replaced() {
            Some(replaced) => self
                .store
                .publish_replacing(key, shape, bytes, replaced)
                .map_err(|error| refusal(key, error)),
            None => self.store.publish(key, shape, bytes),
        }
    }

    /// The serving table stays: its address is what the captured graphs read, and the model
    /// writes it again once the changed weights are in place, so this load allocates none and
    /// states the table as a tensor with no data.
    fn writable(&self, key: &WeightKey, shape: &Shape, _bytes: &[u8]) -> nn::Result<Tensor> {
        self.store.note_table(key);

        Ok(ops::retained(
            META_BASE,
            shape.clone(),
            key.dtype,
            &ops::meta(),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use nn::DType;

    use super::*;
    use crate::memory::fixtures::identity;

    /// The key of a weight read from `tensors` of the checkpoint file `file`, a stack of
    /// `slots` slots when it has several parts.
    fn weight(
        directory: &std::path::Path,
        file: &str,
        tensors: &[&str],
        layout: WeightLayout,
        slots: usize,
    ) -> (WeightKey, bool) {
        let key = WeightKey {
            files: tensors.iter().map(|_| identity(directory, file)).collect(),
            dtype: DType::BF16,
            layout,
            tensors: tensors.iter().map(|tensor| (*tensor).to_owned()).collect(),
            start: 0,
            shape: Shape::from((slots, 8)),
            stride: None,
            provenance: None,
        };
        (key, false)
    }

    #[test]
    fn another_expert_order_pairs_every_changed_weight_by_its_position() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let serving = [
            weight(root, "a", &["norm"], WeightLayout::ROW_MAJOR, 1),
            weight(root, "a", &["e0.up", "e1.up"], WeightLayout::UP_GATE, 2),
        ];
        let target = [
            weight(root, "a", &["norm"], WeightLayout::ROW_MAJOR, 1),
            weight(root, "a", &["e1.up", "e0.up"], WeightLayout::UP_GATE, 2),
        ];

        let pairs = paired(&serving, &target).unwrap();

        // The norm's key did not change, and nothing replaces it.
        assert_eq!(pairs, [(serving[1].0.clone(), target[1].0.clone())]);
    }

    #[test]
    fn a_checkpoint_of_other_tensors_is_refused_by_its_first_difference() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let serving = [weight(root, "a", &["norm"], WeightLayout::ROW_MAJOR, 1)];

        let other = [weight(root, "b", &["other"], WeightLayout::ROW_MAJOR, 1)];
        let difference = paired(&serving, &other).unwrap_err().to_string();
        assert!(difference.contains("norm"), "{difference}");

        assert!(paired(&serving, &[]).is_err());

        let wider = [weight(root, "b", &["norm"], WeightLayout::ROW_MAJOR, 2)];
        assert!(paired(&serving, &wider).is_err());

        let (key, _) = weight(root, "b", &["norm"], WeightLayout::ROW_MAJOR, 1);
        assert!(paired(&serving, &[(key, true)]).is_err());
    }

    /// A table the model writes after its load keeps its place in the order and has no pair:
    /// the serving table stays, and the model writes it again.
    #[test]
    fn a_table_is_compared_and_never_paired() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let (serving, _) = weight(root, "a", &["table"], WeightLayout::ROW_MAJOR, 1);
        let (target, _) = weight(root, "b", &["table"], WeightLayout::ROW_MAJOR, 1);

        let pairs = paired(&[(serving, true)], &[(target, true)]).unwrap();

        assert!(pairs.is_empty());
    }

    #[test]
    fn a_rearranged_stack_names_the_slot_each_slot_came_from() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let serving = [weight(
            root,
            "a",
            &["e0.up", "e0.gate", "e1.up", "e1.gate"],
            WeightLayout::UP_GATE,
            2,
        )];
        let target = [weight(
            root,
            "a",
            &["e1.up", "e1.gate", "e0.up", "e0.gate"],
            WeightLayout::UP_GATE,
            2,
        )];

        assert_eq!(slots_of(&serving[0].0, &target[0].0), Some(vec![1, 0]));

        assert_eq!(paired(&serving, &target).unwrap().len(), 1);
    }

    /// A stack of another checkpoint names the same tensors from other files: its bytes are
    /// that checkpoint's, and copying the serving stack under its key would publish the
    /// serving experts as the changed ones.
    #[test]
    fn a_stack_of_another_checkpoint_has_no_slots_on_the_device() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let tensors = ["e0.up", "e0.gate", "e1.up", "e1.gate"];
        let (serving, _) = weight(root, "a", &tensors, WeightLayout::UP_GATE, 2);
        let (target, _) = weight(root, "b", &tensors, WeightLayout::UP_GATE, 2);

        assert_eq!(slots_of(&serving, &target), None);
    }

    #[test]
    fn a_weight_that_is_no_stack_has_no_slots() {
        let directory = tempfile::tempdir().unwrap();
        let (plain, _) = weight(directory.path(), "a", &["w"], WeightLayout::ROW_MAJOR, 1);

        assert_eq!(slots_of(&plain, &plain), None);
    }
}
