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
/// epoch - a new epoch on the same build config maps what an earlier one published - so a weight's
/// storage record names no epoch and no segment.
pub trait WeightStorage: Send + Sync {
    /// The bytes already retained under `key`, mapped read-only, or `None` when the Storage Server
    /// holds none.
    fn fetch(&self, device: platform::DeviceId, key: &[u8; 32]) -> Result<Option<Backing>>;

    /// The allocation granularity the Storage Server imports a retained allocation in, on
    /// `device`: the length it retains must be a positive multiple of it (§8).
    fn granularity(&self, device: platform::DeviceId) -> Result<usize>;

    /// Retain `bytes` under `key` in an allocation of `allocation` bytes and answer the read-only
    /// region they are mapped at.
    ///
    /// `bytes.len()` is the weight's true length, which is its shape's and is rarely a granule;
    /// `allocation` is that length rounded up, and is what the server imports. The bytes past
    /// `bytes.len()` are a tail nothing reads. A store that publishes the true length instead is
    /// refused by the server's import, which is why the rounding happens in [`WeightStore::publish`]
    /// rather than in the store.
    ///
    /// Storing a key twice is the caller's error; a store may publish it again or answer the region
    /// it already holds.
    fn store(
        &self,
        device: platform::DeviceId,
        key: &[u8; 32],
        bytes: &[u8],
        allocation: usize,
    ) -> Result<Backing>;

    /// `bytes` in a region of `allocation` bytes that kernels may write, held by this process.
    ///
    /// A store with no read-only mapping answers what [`Self::store`] does. The key is not
    /// retained: the launch that fills the region runs in every load.
    fn writable(
        &self,
        device: platform::DeviceId,
        key: &[u8; 32],
        bytes: &[u8],
        allocation: usize,
    ) -> Result<Backing> {
        self.store(device, key, bytes, allocation)
    }
}

/// A shared storage, as two owners of one rank's client use it: the store itself and whatever else
/// the rank maps weights through.
impl<S: WeightStorage + ?Sized> WeightStorage for std::sync::Arc<S> {
    fn fetch(&self, device: platform::DeviceId, key: &[u8; 32]) -> Result<Option<Backing>> {
        (**self).fetch(device, key)
    }

    fn writable(
        &self,
        device: platform::DeviceId,
        key: &[u8; 32],
        bytes: &[u8],
        allocation: usize,
    ) -> Result<Backing> {
        (**self).writable(device, key, bytes, allocation)
    }

    fn granularity(&self, device: platform::DeviceId) -> Result<usize> {
        (**self).granularity(device)
    }

    fn store(
        &self,
        device: platform::DeviceId,
        key: &[u8; 32],
        bytes: &[u8],
        allocation: usize,
    ) -> Result<Backing> {
        (**self).store(device, key, bytes, allocation)
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

/// One weight a model's load reached, in the load's order.
///
/// A load is deterministic in its order: the same code over a checkpoint of the same model asks for
/// the same weights one after another. A position in this list is therefore the identity of a
/// weight across two loads, which is what lets an update pair each serving weight with its
/// replacement although every key differs (§7.6).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedWeight {
    /// The weight's key.
    pub key: WeightKey,
    /// Whether the model writes this region after the load: it lives in this process alone, and
    /// no update replaces it, because the model writes it again.
    pub writable: bool,
}

/// The epoch's retained weights: the keys this rank has mapped, beside the storage that holds them.
///
/// The store is the main process's, built once per rank and shared by every model load of the
/// epoch. A hit is answered from the map; a miss asks the storage, and a miss there is not
/// remembered, because another rank of the deployment may publish the same weight while this one
/// serves.
pub struct WeightStore<S: WeightStorage> {
    storage: S,
    /// The device this store's weights are mapped on. A `Backing` is one device's address, so a
    /// rank that serves two devices holds one store per device: the storage is the same client.
    device: platform::DeviceId,
    /// Every weight this rank has mapped, by storage key.
    mapped: BTreeMap<[u8; 32], Backing>,
    /// How many weights this rank read back from the storage rather than converting itself.
    from_store: usize,
    /// How many weights this rank published to the storage, which is how many keys it added to a
    /// store that never releases one (§8: a content-addressed cache that outlives its epochs).
    published: usize,
    /// Every weight the load reached, in its order, for the updates of §7.6.
    log: Vec<LoadedWeight>,
}

impl<S: WeightStorage> WeightStore<S> {
    /// A store over `storage` that maps every weight on `device`, with nothing mapped yet.
    pub fn new(storage: S, device: platform::DeviceId) -> Self {
        Self {
            storage,
            device,
            mapped: BTreeMap::new(),
            from_store: 0,
            published: 0,
            log: Vec::new(),
        }
    }

    /// The weights the load reached, in its order, leaving the log empty.
    pub fn take_log(&mut self) -> Vec<LoadedWeight> {
        std::mem::take(&mut self.log)
    }

    /// How many weights the load reached so far: the position of the next one.
    pub(crate) fn logged(&self) -> usize {
        self.log.len()
    }

    /// Records a weight this load published by another path than [`RetainedWeights::publish`].
    pub(crate) fn adopt(&mut self, key: &WeightKey, backing: Backing) {
        self.log.push(LoadedWeight {
            key: key.clone(),
            writable: false,
        });
        self.mapped.insert(retained_key(key), backing);
        self.published += 1;
    }

    /// Records a table the load reached and holds no storage for.
    pub(crate) fn note_table(&mut self, key: &WeightKey) {
        self.log.push(LoadedWeight {
            key: key.clone(),
            writable: true,
        });
    }

    /// How many weights this rank holds, for the metrics report and for tests.
    pub fn mapped(&self) -> usize {
        self.mapped.len()
    }

    /// How many weights this rank read back from the storage instead of converting them.
    pub fn from_store(&self) -> usize {
        self.from_store
    }

    /// How many weights this rank published to the storage.
    ///
    /// A retained store never releases a key, so this is exactly what the host gained: a run that
    /// published nothing grew the store by nothing, and one that published after a key change added
    /// that many records beside the ones the previous key left.
    pub fn published(&self) -> usize {
        self.published
    }

    /// The storage behind the store.
    pub fn storage(&self) -> &S {
        &self.storage
    }

    /// The allocation a weight of `bytes` bytes is retained in: its true length rounded up to the
    /// granularity the Storage Server imports a retained allocation in (§8).
    ///
    /// The weight's length is its shape's, so it is a whole number of granules only by accident: an
    /// embedding of 622329856 bytes against a 2097152-byte granule is refused by the import. The
    /// rounded length is what the store allocates and publishes; the tensor keeps `bytes`, and a
    /// reader maps the allocation and reads only the weight.
    fn allocation_length(&self, bytes: usize) -> Result<usize> {
        let granularity = self.storage.granularity(self.device).map_err(|error| {
            Error::State(format!("the granularity of {}: {error}", self.device).leak())
        })?;
        if granularity == 0 {
            return Err(Error::Invariant(
                "the storage states a zero allocation granularity".into(),
            ));
        }
        bytes
            .div_ceil(granularity)
            .checked_mul(granularity)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "a weight of {bytes} bytes has no whole-granule allocation"
                ))
            })
    }

    /// The backing of `key`, mapped if this rank has not mapped it yet.
    fn backing(&mut self, key: &WeightKey) -> Result<Option<Backing>> {
        let storage_key = retained_key(key);
        if let Some(backing) = self.mapped.get(&storage_key) {
            return Ok(Some(*backing));
        }
        let Some(backing) = self
            .storage
            .fetch(self.device, &storage_key)
            .map_err(|error| {
                Error::State(format!("mapping retained weight {key}: {error}").leak())
            })?
        else {
            return Ok(None);
        };
        self.mapped.insert(storage_key, backing);
        self.from_store += 1;
        Ok(Some(backing))
    }
}

impl<S: WeightStorage> Drop for WeightStore<S> {
    /// One line, when the epoch's store is released: what this rank's loads read back and what they
    /// published.
    ///
    /// A retained store never releases a key, so a reader looking at its size after a key change has
    /// no way to tell a run that reused it from one that grew it. These are those two figures, and
    /// they are printed once per store rather than per weight because the question is about the
    /// store, not about a weight.
    fn drop(&mut self) {
        if self.from_store == 0 && self.published == 0 {
            return;
        }
        tracing::info!(
            device = %self.device,
            mapped = self.mapped.len(),
            from_store = self.from_store,
            published = self.published,
            "the epoch's retained weights: what this rank mapped, read back and published"
        );
    }
}

impl<S: WeightStorage> RetainedWeights for WeightStore<S> {
    fn retained(&mut self, key: &WeightKey) -> nn::Result<Option<Backing>> {
        let backing = self
            .backing(key)
            .map_err(|error| nn::Error::Unsupported(error.to_string()))?;
        if backing.is_some() {
            self.log.push(LoadedWeight {
                key: key.clone(),
                writable: false,
            });
        }
        Ok(backing)
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
            .checked_mul(dtype.bytes().unwrap_or_default())
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
        self.log.push(LoadedWeight {
            key: key.clone(),
            writable: false,
        });
        // Publishing a key this rank already mapped answers the region it holds: the bytes are the
        // same weight, and the second publication would be the caller's error (§8, `publish`).
        if let Some(backing) = self.mapped.get(&storage_key) {
            return Ok(*backing);
        }
        let allocation = self
            .allocation_length(bytes.len())
            .map_err(|error| nn::Error::Unsupported(format!("sizing weight {key}: {error}")))?;
        let backing = self
            .storage
            .store(self.device, &storage_key, bytes, allocation)
            .map_err(|error| {
                nn::Error::Unsupported(format!("publishing weight {key}: {}", with_causes(&error)))
            })?;
        self.mapped.insert(storage_key, backing);
        self.published += 1;
        Ok(backing)
    }

    fn writable(
        &mut self,
        key: &WeightKey,
        _dtype: DType,
        _shape: Shape,
        bytes: &[u8],
    ) -> nn::Result<Backing> {
        self.log.push(LoadedWeight {
            key: key.clone(),
            writable: true,
        });
        let allocation = self
            .allocation_length(bytes.len())
            .map_err(|error| nn::Error::Unsupported(format!("sizing table {key}: {error}")))?;
        self.storage
            .writable(self.device, &retained_key(key), bytes, allocation)
            .map_err(|error| nn::Error::Unsupported(format!("allocating table {key}: {error}")))
    }
}

/// `error` and every error that caused it, in the order they caused one another: a refusal of the
/// Storage Server says why under the call that met it.
fn with_causes(error: &dyn std::error::Error) -> String {
    let mut text = error.to_string();
    let mut cause = error.source();
    while let Some(next) = cause {
        text.push_str(": ");
        text.push_str(&next.to_string());
        cause = next.source();
    }
    text
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use nn::weights::WeightRegion;
    use nn::{FileIdentity, WeightLayout};

    use super::*;

    /// A storage that keeps bytes in host memory and hands out fake device addresses, so the key
    /// and the store's bookkeeping are tested without a device.
    struct FakeStorage {
        published: Mutex<BTreeMap<[u8; 32], Vec<u8>>>,
        fetches: AtomicUsize,
        stores: AtomicUsize,
        next_address: AtomicUsize,
        /// The granule the driver imports a retained allocation in. One byte unless a test states
        /// otherwise, so a test that is not about rounding is not about rounding.
        granularity: usize,
        /// What the last store was handed: the weight's true length and its allocation.
        handed: Mutex<Option<(usize, usize)>>,
    }

    impl Default for FakeStorage {
        fn default() -> Self {
            Self {
                published: Mutex::new(BTreeMap::new()),
                fetches: AtomicUsize::new(0),
                stores: AtomicUsize::new(0),
                next_address: AtomicUsize::new(0),
                granularity: 1,
                handed: Mutex::new(None),
            }
        }
    }

    impl FakeStorage {
        /// A storage whose import, like the driver's, takes whole granules and refuses anything
        /// else - the refusal that ended an epoch's model load.
        fn with_granularity(granularity: usize) -> Self {
            Self {
                granularity,
                ..Self::default()
            }
        }

        fn address(&self) -> u64 {
            // An address that is never dereferenced: the test reads the keys and the lengths only.
            self.next_address.fetch_add(1 << 20, Ordering::Relaxed) as u64 + 0x1000
        }
    }

    impl WeightStorage for FakeStorage {
        fn fetch(&self, _device: platform::DeviceId, key: &[u8; 32]) -> Result<Option<Backing>> {
            self.fetches.fetch_add(1, Ordering::Relaxed);
            Ok(self.published.lock().unwrap().get(key).map(|bytes| {
                let _ = bytes;
                Backing::new(ops::DevicePtr::new(self.address()), bytes.len() as u64)
            }))
        }

        fn granularity(&self, _device: platform::DeviceId) -> Result<usize> {
            Ok(self.granularity)
        }

        fn store(
            &self,
            _device: platform::DeviceId,
            key: &[u8; 32],
            bytes: &[u8],
            allocation: usize,
        ) -> Result<Backing> {
            self.stores.fetch_add(1, Ordering::Relaxed);
            *self.handed.lock().unwrap() = Some((bytes.len(), allocation));
            if allocation < bytes.len() || !allocation.is_multiple_of(self.granularity) {
                return Err(Error::Invalid(format!(
                    "a retained allocation of {allocation} bytes is not a positive multiple of \
                     the {}-byte granularity",
                    self.granularity
                )));
            }
            self.published.lock().unwrap().insert(*key, bytes.to_vec());
            Ok(Backing::new(
                ops::DevicePtr::new(self.address()),
                allocation as u64,
            ))
        }
    }

    /// The identity of a checkpoint file, as a Hugging Face cache stores it.
    ///
    /// A file under `blobs/` is identified by its name alone, so the fixture states identity
    /// without waiting for a write to settle: `FileIdentity::of` gives a file younger than a second
    /// an identity that matches no later read, and a checkpoint under test must not depend on that.
    fn identity(name: &str) -> FileIdentity {
        let directory = std::env::temp_dir()
            .join("uniserve-weight-key-tests")
            .join("blobs");
        std::fs::create_dir_all(&directory).expect("a temporary directory");
        // A blob's name is a 64-digit hexadecimal content hash; the content is the fixture's own.
        let digest: String = name
            .bytes()
            .flat_map(|byte| format!("{byte:02x}").into_bytes())
            .map(char::from)
            .take(64)
            .collect();
        let blob = format!("{digest:0<64}");
        let path = directory.join(blob);
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
            provenance: None,
        }
    }

    /// A computed weight: no file names it, so its provenance does.
    fn computed(producer: &'static str, bytes: &[u8]) -> WeightKey {
        WeightKey {
            files: Vec::new(),
            dtype: DType::F32,
            layout: WeightLayout::named("row_major"),
            region: region("denoiser.video.student_heads.weight"),
            provenance: Some(nn::weights::Provenance::computed(producer, bytes)),
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
        let mut store = WeightStore::new(FakeStorage::default(), platform::DeviceId(0));
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
        let mut other = WeightStore::new(Shared(Arc::clone(&storage)), platform::DeviceId(0));
        other
            .publish(&weight, DType::BF16, ops::Shape::new(&[4, 8]), &bytes)
            .expect("a well-formed weight");

        let mut mine = WeightStore::new(Shared(Arc::clone(&storage)), platform::DeviceId(0));
        assert!(mine.retained(&weight).expect("a hit").is_some());
        assert!(mine.retained(&weight).expect("a hit").is_some());
        assert_eq!(storage.fetches.load(Ordering::Relaxed), 1);
        assert_eq!(mine.mapped(), 1);
    }

    #[test]
    fn a_computed_weight_of_one_deployment_is_not_answered_to_another() {
        let storage = Arc::new(FakeStorage::default());
        let shape = ops::Shape::new(&[4, 8]);
        let base = computed("transformer", &[0_u8; 128]);
        let reference = computed("transformer_ref", &[1_u8; 128]);

        let mut first = WeightStore::new(Shared(Arc::clone(&storage)), platform::DeviceId(0));
        first
            .publish(&base, DType::F32, shape, &[0_u8; 128])
            .expect("the base deployment publishes its fused heads");

        // The reference deployment asks for its own table under the same label and extent. It
        // must not be answered the base's bytes: a label is not an identity, and the mapping is
        // indistinguishable from a correct one once it is handed over.
        let mut second = WeightStore::new(Shared(Arc::clone(&storage)), platform::DeviceId(0));
        assert!(second.retained(&reference).expect("a miss").is_none());
        second
            .publish(&reference, DType::F32, shape, &[1_u8; 128])
            .expect("the reference deployment publishes its own");
        assert_eq!(storage.stores.load(Ordering::Relaxed), 2);

        // And each deployment's own restart maps its own table rather than recomputing it.
        let mut restarted = WeightStore::new(Shared(Arc::clone(&storage)), platform::DeviceId(0));
        assert!(restarted.retained(&reference).expect("a hit").is_some());

        // The two figures a reader needs to tell a run that grew the store from one that reused it:
        // the first deployment published and read nothing back, the restart read and published
        // nothing.
        assert_eq!((first.from_store(), first.published()), (0, 1));
        assert_eq!((second.from_store(), second.published()), (0, 1));
        assert_eq!((restarted.from_store(), restarted.published()), (1, 0));
    }

    #[test]
    fn a_weight_nothing_published_is_a_miss_that_is_not_remembered() {
        let mut store = WeightStore::new(FakeStorage::default(), platform::DeviceId(0));
        let weight = key("blob:aaa", DType::BF16, "model.layers.0.mlp.up");
        assert!(store.retained(&weight).expect("a miss").is_none());
        assert!(store.retained(&weight).expect("a miss").is_none());
        // Another rank may publish it while this one serves, so a miss is asked again.
        assert_eq!(store.storage().fetches.load(Ordering::Relaxed), 2);
        assert_eq!(store.mapped(), 0);
    }

    #[test]
    fn publishing_under_a_key_of_another_element_type_is_refused() {
        let mut store = WeightStore::new(FakeStorage::default(), platform::DeviceId(0));
        let weight = key("blob:aaa", DType::BF16, "model.layers.0.mlp.up");
        let error = store
            .publish(&weight, DType::F32, ops::Shape::new(&[4, 8]), &[0_u8; 128])
            .expect_err("the key states the type the bytes are in");
        assert!(error.to_string().contains("cannot be published as"));
        assert_eq!(store.storage().stores.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn publishing_a_byte_count_the_shape_does_not_state_is_refused() {
        let mut store = WeightStore::new(FakeStorage::default(), platform::DeviceId(0));
        let weight = key("blob:aaa", DType::BF16, "model.layers.0.mlp.up");
        let error = store
            .publish(&weight, DType::BF16, ops::Shape::new(&[4, 8]), &[0_u8; 7])
            .expect_err("4 x 8 BF16 is 64 bytes");
        assert!(error.to_string().contains("is 64 bytes, not 7"));
        assert_eq!(store.storage().stores.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn publishing_the_same_weight_twice_answers_the_region_it_holds() {
        let mut store = WeightStore::new(FakeStorage::default(), platform::DeviceId(0));
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

    /// A weight's byte length is its shape's, so it is a whole number of granules only by accident;
    /// the Storage Server imports a retained allocation in whole granules and refuses the rest.
    #[test]
    fn a_weight_whose_length_is_not_a_granule_is_retained_in_a_rounded_allocation() {
        const GRANULE: usize = 1 << 21;
        let storage = std::sync::Arc::new(FakeStorage::with_granularity(GRANULE));
        let mut store = WeightStore::new(
            Shared(std::sync::Arc::clone(&storage)),
            platform::DeviceId(0),
        );
        // The embedding that ended the epoch's model load is 622329856 bytes against a 2097152-byte
        // granule, which is 296 granule and 190210304 bytes over; this is the same shape, small.
        let bytes = vec![3_u8; 30];
        let published = store
            .publish(
                &key("blob:aaa", DType::BF16, "model.embed_tokens"),
                DType::BF16,
                ops::Shape::new(&[3, 5]),
                &bytes,
            )
            .expect("a weight retained in a rounded allocation");
        // What the store was handed: the true length to state, and the granule to import.
        let (length, allocation) = storage.handed.lock().unwrap().expect("a store");
        assert_eq!(length, 30);
        assert_eq!(allocation, GRANULE);
        assert_eq!(published.bytes(), GRANULE as u64);
        // A reader maps the allocation, and a second publication answers the same region.
        let again = store
            .publish(
                &key("blob:aaa", DType::BF16, "model.embed_tokens"),
                DType::BF16,
                ops::Shape::new(&[3, 5]),
                &bytes,
            )
            .expect("the region it holds");
        assert_eq!(published.address(), again.address());
    }

    /// A storage shared by two stores, as two ranks of one deployment share the Storage Server.
    struct Shared(std::sync::Arc<FakeStorage>);

    impl WeightStorage for Shared {
        fn fetch(&self, _device: platform::DeviceId, key: &[u8; 32]) -> Result<Option<Backing>> {
            self.0.fetch(platform::DeviceId(0), key)
        }

        fn granularity(&self, _device: platform::DeviceId) -> Result<usize> {
            self.0.granularity(platform::DeviceId(0))
        }

        fn store(
            &self,
            _device: platform::DeviceId,
            key: &[u8; 32],
            bytes: &[u8],
            allocation: usize,
        ) -> Result<Backing> {
            self.0.store(platform::DeviceId(0), key, bytes, allocation)
        }
    }
}
