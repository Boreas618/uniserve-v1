//! The rank's retained weights (`specs/worker.md` §8).
//!
//! A weight is converted once and kept: the loader reads a checkpoint tensor, converts it to the
//! deployment's representation, publishes the bytes to the Storage Server under a key, and every
//! later load of the same weight is a read-only mapping of what was published.
//!
//! The key is [`nn::WeightKey`]'s own four inputs, canonicalized by its `Display`: the identity of
//! every checkpoint file the weight was read from, the element type the loader resolved, the
//! layout the reading operator declares, and the region inside the logical tensor. Two rules come
//! with it (§8):
//!
//! - a **program layer's** backend or revision is not part of the key: a generation update changes
//!   which module reads a weight, not what the weight is, so the update keeps the storage warm;
//! - anything that changes the mathematics, the precision or the format is a build-config change,
//!   and then `dtype`, `layout` or `quantization` move the key with it.
//!
//! The bytes themselves live at the Storage Server, under the rank's claim, inside the GPU's
//! `budget` (§8's first operation table). This module owns the key and the store; how a key becomes
//! a mapped region is [`WeightStorage`], which the rank's `Retained` client implements.

use std::collections::BTreeMap;

use nn::{Backing, DType, RetainedWeights, Shape, WeightKey};
use sha2::{Digest, Sha256};

use crate::{Error, Result};

/// The domain tag of a retained weight's key, so a weight's key can never collide with the key of
/// another kind of storage this rank publishes (the cache's planes, the products, the latents).
const WEIGHT_KEY_DOMAIN: &[u8] = b"uniserve-retained-weight\0";

/// Where a rank's retained weights live: the Storage Server, behind the rank's client (§8).
///
/// Both calls take the storage key [`retained_key`] derives, and both answer an
/// [`nn::Backing`]: a read-only device region a layer's view is carved from. The bytes outlive the
/// epoch's worlds - a new epoch on the same build config maps what an earlier one published - so a
/// weight's storage record names no world and no segment.
pub trait WeightStorage: Send + Sync {
    /// The bytes already retained under `key`, mapped read-only, or `None` when the Storage Server
    /// holds none.
    fn fetch(&self, key: &[u8; 32]) -> Result<Option<Backing>>;

    /// Store `bytes` under `key` and answer the read-only region they are mapped at.
    ///
    /// Storing a key twice is the caller's error; a store may publish it again or answer the region
    /// it already holds.
    fn store(&self, key: &[u8; 32], bytes: &[u8]) -> Result<Backing>;
}

/// A shared storage, as two owners of one rank's client use it: the store itself and whatever else
/// the rank maps weights through.
impl<S: WeightStorage + ?Sized> WeightStorage for std::sync::Arc<S> {
    fn fetch(&self, key: &[u8; 32]) -> Result<Option<Backing>> {
        (**self).fetch(key)
    }

    fn store(&self, key: &[u8; 32], bytes: &[u8]) -> Result<Backing> {
        (**self).store(key, bytes)
    }
}

/// The storage key of `weight`, as §8 derives it from the model and its representation.
///
/// The digest covers [`nn::WeightKey`]'s canonical text, which is every file's identity, the
/// element type, the layout name and the region - and nothing about which program reads it.
pub fn retained_key(weight: &WeightKey) -> [u8; 32] {
    let mut digest = Sha256::new();
    digest.update(WEIGHT_KEY_DOMAIN);
    // Length-prefixed, so no two canonical texts can be concatenated into one key.
    let text = weight.to_string();
    digest.update((text.len() as u64).to_le_bytes());
    digest.update(text.as_bytes());
    digest.finalize().into()
}

/// The epoch's retained weights: the keys this rank has mapped, beside the storage that holds them.
///
/// The store is the main process's, built once per rank and shared by every model load of the
/// epoch. A hit is answered from the map; a miss asks the storage, and a miss there is not
/// remembered, because another rank of the deployment may publish the same weight while this one
/// serves.
pub struct WeightStore<S: WeightStorage> {
    storage: S,
    /// Every weight this rank has mapped, by storage key.
    mapped: BTreeMap<[u8; 32], Backing>,
}

impl<S: WeightStorage> WeightStore<S> {
    /// A store over `storage`, with nothing mapped yet.
    pub fn new(storage: S) -> Self {
        Self {
            storage,
            mapped: BTreeMap::new(),
        }
    }

    /// How many weights this rank holds, for the metrics report and for tests.
    pub fn mapped(&self) -> usize {
        self.mapped.len()
    }

    /// The storage behind the store.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// The backing of `key`, mapped if this rank has not mapped it yet.
    fn backing(&mut self, key: &WeightKey) -> Result<Option<Backing>> {
        let storage_key = retained_key(key);
        if let Some(backing) = self.mapped.get(&storage_key) {
            return Ok(Some(*backing));
        }
        let Some(backing) = self
            .storage
            .fetch(&storage_key)
            .map_err(|error| Error::State(format!("mapping retained weight {key}: {error}").leak()))?
        else {
            return Ok(None);
        };
        self.mapped.insert(storage_key, backing);
        Ok(Some(backing))
    }
}

impl<S: WeightStorage> RetainedWeights for WeightStore<S> {
    fn retained(&mut self, key: &WeightKey) -> nn::Result<Option<Backing>> {
        self.backing(key)
            .map_err(|error| nn::Error::Unsupported(error.to_string()))
    }

    fn publish(
        &mut self,
        key: &WeightKey,
        dtype: DType,
        shape: Shape,
        bytes: &[u8],
    ) -> nn::Result<Backing> {
        // The key already states the element type the bytes are in; a caller that publishes under
        // a key of another type would make the next load read the bytes as the wrong numbers.
        if dtype != key.dtype {
            return Err(nn::Error::Unsupported(format!(
                "a weight keyed as {} cannot be published as {dtype:?}",
                key.dtype
            )));
        }
        let expected = shape
            .numel()
            .checked_mul(u64::from(dtype.bytes().unwrap_or_default()))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| {
                nn::Error::Unsupported(format!(
                    "a weight of shape {} in {dtype:?} has no byte length",
                    shape
                ))
            })?;
        if bytes.len() != expected {
            return Err(nn::Error::Unsupported(format!(
                "a weight of shape {} in {dtype:?} is {expected} bytes, not {}",
                shape,
                bytes.len()
            )));
        }

        let storage_key = retained_key(key);
        // Publishing a key this rank already mapped answers the region it holds: the bytes are the
        // same weight, and the second publication would be the caller's error (§8, `publish`).
        if let Some(backing) = self.mapped.get(&storage_key) {
            return Ok(*backing);
        }
        let backing = self
            .storage
            .store(&storage_key, bytes)
            .map_err(|error| nn::Error::Unsupported(format!("publishing weight {key}: {error}")))?;
        self.mapped.insert(storage_key, backing);
        Ok(backing)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use nn::{FileIdentity, WeightLayout};
    use nn::weights::WeightRegion;

    use super::*;

    /// A storage that keeps bytes in host memory and hands out fake device addresses, so the key
    /// and the store's bookkeeping are tested without a device.
    #[derive(Default)]
    struct FakeStorage {
        published: Mutex<BTreeMap<[u8; 32], Vec<u8>>>,
        fetches: AtomicUsize,
        stores: AtomicUsize,
        next_address: AtomicUsize,
    }

    impl FakeStorage {
        fn address(&self) -> u64 {
            // An address that is never dereferenced: the test reads the keys and the lengths only.
            self.next_address.fetch_add(1 << 20, Ordering::Relaxed) as u64 + 0x1000
        }
    }

    impl WeightStorage for FakeStorage {
        fn fetch(&self, key: &[u8; 32]) -> Result<Option<Backing>> {
            self.fetches.fetch_add(1, Ordering::Relaxed);
            Ok(self.published.lock().unwrap().get(key).map(|bytes| {
                let _ = bytes;
                Backing::new(ops::DevicePtr::new(self.address()), bytes.len() as u64)
            }))
        }

        fn store(&self, key: &[u8; 32], bytes: &[u8]) -> Result<Backing> {
            self.stores.fetch_add(1, Ordering::Relaxed);
            self.published
                .lock()
                .unwrap()
                .insert(*key, bytes.to_vec());
            Ok(Backing::new(
                ops::DevicePtr::new(self.address()),
                bytes.len() as u64,
            ))
        }
    }

    /// The identity of a checkpoint file the test writes once per name, so two names are two
    /// checkpoint files and the same name is one file read twice (`FileIdentity::of` is the only
    /// constructor, and it derives the identity from the file itself).
    fn identity(name: &str) -> FileIdentity {
        let directory = std::env::temp_dir().join("uniserve-weight-key-tests");
        std::fs::create_dir_all(&directory).expect("a temporary directory");
        let path = directory.join(name);
        if !path.exists() {
            std::fs::write(&path, name).expect("a temporary checkpoint file");
        }
        FileIdentity::of(&path).expect("the identity of a checkpoint file")
    }

    fn region(tensor: &str) -> WeightRegion {
        WeightRegion {
            tensors: vec![tensor.to_owned()],
            start: 0,
            shape: ops::Shape::new(&[4, 8]),
            stride: None,
        }
    }

    fn key(file: &str, dtype: DType, tensor: &str) -> WeightKey {
        WeightKey {
            files: vec![identity(file)],
            dtype,
            layout: WeightLayout::named("row_major"),
            region: region(tensor),
        }
    }

    #[test]
    fn the_key_is_the_same_weight_reached_twice() {
        let first = retained_key(&key("blob:aaa", DType::BF16, "model.layers.0.mlp.up"));
        let second = retained_key(&key("blob:aaa", DType::BF16, "model.layers.0.mlp.up"));
        assert_eq!(first, second);
    }

    #[test]
    fn a_changed_file_dtype_layout_or_region_moves_the_key() {
        let base = key("blob:aaa", DType::BF16, "model.layers.0.mlp.up");
        // The same tensor read from a different checkpoint file.
        assert_ne!(
            retained_key(&base),
            retained_key(&key("blob:bbb", DType::BF16, "model.layers.0.mlp.up"))
        );
        // The same file and tensor in another representation (§8: precision moves the key).
        assert_ne!(
            retained_key(&base),
            retained_key(&key("blob:aaa", DType::F32, "model.layers.0.mlp.up"))
        );
        // The same tensor, another region of it.
        assert_ne!(
            retained_key(&base),
            retained_key(&key("blob:aaa", DType::BF16, "model.layers.0.mlp.down"))
        );
        // Another layout of the same bytes.
        let mut other = base.clone();
        other.layout = WeightLayout::named("column_major");
        assert_ne!(retained_key(&base), retained_key(&other));
    }

    #[test]
    fn a_program_change_does_not_move_the_key() {
        // §8: the key carries no program backend and no revision, so a generation that changes
        // which module reads a weight still finds it retained.
        let weight = key("blob:aaa", DType::BF16, "model.layers.0.mlp.up");
        let first = retained_key(&weight);
        let second = retained_key(&weight);
        assert_eq!(first, second);
    }

    #[test]
    fn a_published_weight_is_a_hit_and_asks_the_storage_once() {
        let mut store = WeightStore::new(FakeStorage::default());
        let weight = key("blob:aaa", DType::BF16, "model.layers.0.mlp.up");
        let bytes = vec![0_u8; 4 * 8 * 2];
        let published = store
            .publish(&weight, DType::BF16, ops::Shape::new(&[4, 8]), &bytes)
            .expect("a well-formed weight");

        // The same rank reading it back maps nothing again.
        let mapped = store.retained(&weight).expect("a hit").expect("a backing");
        assert_eq!(published.address(), mapped.address());
        assert_eq!(store.storage().stores.load(Ordering::Relaxed), 1);
        assert_eq!(store.storage().fetches.load(Ordering::Relaxed), 0);
        assert_eq!(store.mapped(), 1);
    }

    #[test]
    fn a_weight_another_rank_published_is_fetched_once_and_then_answered_from_the_map() {
        let storage = std::sync::Arc::new(FakeStorage::default());
        let weight = key("blob:aaa", DType::BF16, "model.layers.0.mlp.up");
        let bytes = vec![0_u8; 4 * 8 * 2];

        // Another rank of the deployment publishes it first.
        let mut other = WeightStore::new(Shared(Arc::clone(&storage)));
        other
            .publish(&weight, DType::BF16, ops::Shape::new(&[4, 8]), &bytes)
            .expect("a well-formed weight");

        let mut mine = WeightStore::new(Shared(Arc::clone(&storage)));
        assert!(mine.retained(&weight).expect("a hit").is_some());
        assert!(mine.retained(&weight).expect("a hit").is_some());
        assert_eq!(storage.fetches.load(Ordering::Relaxed), 1);
        assert_eq!(mine.mapped(), 1);
    }

    #[test]
    fn a_weight_nothing_published_is_a_miss_that_is_not_remembered() {
        let mut store = WeightStore::new(FakeStorage::default());
        let weight = key("blob:aaa", DType::BF16, "model.layers.0.mlp.up");
        assert!(store.retained(&weight).expect("a miss").is_none());
        assert!(store.retained(&weight).expect("a miss").is_none());
        // Another rank may publish it while this one serves, so a miss is asked again.
        assert_eq!(store.storage().fetches.load(Ordering::Relaxed), 2);
        assert_eq!(store.mapped(), 0);
    }

    #[test]
    fn publishing_under_a_key_of_another_element_type_is_refused() {
        let mut store = WeightStore::new(FakeStorage::default());
        let weight = key("blob:aaa", DType::BF16, "model.layers.0.mlp.up");
        let error = store
            .publish(&weight, DType::F32, ops::Shape::new(&[4, 8]), &[0_u8; 128])
            .expect_err("the key states the type the bytes are in");
        assert!(error.to_string().contains("cannot be published as"));
        assert_eq!(store.storage().stores.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn publishing_a_byte_count_the_shape_does_not_state_is_refused() {
        let mut store = WeightStore::new(FakeStorage::default());
        let weight = key("blob:aaa", DType::BF16, "model.layers.0.mlp.up");
        let error = store
            .publish(&weight, DType::BF16, ops::Shape::new(&[4, 8]), &[0_u8; 7])
            .expect_err("4 x 8 BF16 is 64 bytes");
        assert!(error.to_string().contains("is 64 bytes, not 7"));
        assert_eq!(store.storage().stores.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn publishing_the_same_weight_twice_answers_the_region_it_holds() {
        let mut store = WeightStore::new(FakeStorage::default());
        let weight = key("blob:aaa", DType::BF16, "model.layers.0.mlp.up");
        let bytes = vec![0_u8; 4 * 8 * 2];
        let first = store
            .publish(&weight, DType::BF16, ops::Shape::new(&[4, 8]), &bytes)
            .expect("a well-formed weight");
        let second = store
            .publish(&weight, DType::BF16, ops::Shape::new(&[4, 8]), &bytes)
            .expect("the region it holds");
        assert_eq!(first.address(), second.address());
        assert_eq!(store.storage().stores.load(Ordering::Relaxed), 1);
    }

    /// A storage shared by two stores, as two ranks of one deployment share the Storage Server.
    struct Shared(std::sync::Arc<FakeStorage>);

    impl WeightStorage for Shared {
        fn fetch(&self, key: &[u8; 32]) -> Result<Option<Backing>> {
            self.0.fetch(key)
        }

        fn store(&self, key: &[u8; 32], bytes: &[u8]) -> Result<Backing> {
            self.0.store(key, bytes)
        }
    }
}
