//! The rank's retained weights.
//!
//! A weight is converted once and kept: `nn::WeightLoader` reads a checkpoint tensor, converts
//! it to the deployment's representation and publishes the bytes through
//! [`nn::RetainedWeights`], which [`WeightStore`] implements over the host's Storage Server.
//! Every later load of the same weight, by this rank, another rank of the GPU or a later
//! epoch, is a read-only mapping of what was published.
//!
//! ## The key
//!
//! A weight is filed under the digest of [`nn::WeightKey`]'s text ([`storage_key`]): the
//! identity of every checkpoint file the weight was read from, the element type the loader
//! resolved, the layout the reading operator declares, and the region inside the logical
//! tensor. No kernel's name or revision enters it, so a generation that changes which kernel
//! reads a weight finds the weight retained; a change of mathematics, precision or format
//! changes the element type, the layout or the checkpoint, and the key with it.
//!
//! A record names the allocation on the GPU that published it, so each GPU keeps a copy of its
//! own under the key of that GPU (`storage::key::device_weight`): two ranks of one host never
//! read one allocation across the link between their devices.
//!
//! ## The allocations
//!
//! A weight is published as a device allocation the Storage Server retains by its descriptor,
//! outside the rank's own regions, and every reader maps it read-only: a kernel reads the
//! weight in device memory with no copy at load, and none can write what the epochs share. A
//! record that states host storage (an epoch's weights the daemon moved to the host,
//! `storage::protocol::Request::Offload`) is refused by name rather than mapped, because every
//! forward would then read the model across the host link; the daemon moves such weights back
//! before the epoch's engines start (`Request::Restore`).
//!
//! The server imports an allocation in whole multiples of the device's allocation granularity
//! (2 MiB), so a weight of its own allocation takes its length rounded up to one. A weight
//! smaller than the granularity (a norm, a bias, a quantization scale) is **packed**: it lies
//! in an allocation shared with other small weights, which is published as one storage whose
//! tensors lie at their offsets when the load ends ([`WeightStore::finish`]). Every weight has
//! a mapping of its own, a packed one included.
//!
//! A table a model writes after its load ([`nn::RetainedWeights::writable`]) is device storage
//! of this process alone, since published storage is read-only.
//!
//! ## An update in place
//!
//! A weight tensor is `ops::retained` over the address the weight is mapped at, and the
//! modules and the captured graphs of a rank hold that address. An in-place weight update
//! (`/admin/update_weights`) keeps it: the checkpoint is loaded again beside the weights that
//! serve ([`Reload`](crate::memory::reload::Reload)), each weight whose key changed is
//! published where the one it replaces lies (in an allocation of the same length, or at the
//! same offset of a shared allocation of its own), and [`WeightStore::rebind`] then maps each
//! replacement's allocation at the address of the weight it replaces. No module is rebuilt and
//! no published byte is written.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::os::fd::{AsFd, OwnedFd};
use std::sync::{Arc, Mutex};

use nn::{RetainedWeights, Shape, Tensor, WeightKey};
use platform::{DeviceId, HandleType, Protection, Stream};
use storage::protocol::Access;
use storage::{Key, Location, Mode, StorageRecord, TensorRecord};

use crate::memory::allocation::Storage;
use crate::memory::client::{StorageClient, byte_tensor};
use crate::worker::error::report;
use crate::worker::{Error, Result};
use service::lock;

/// The key `weight` is filed under at the Storage Server, before the GPU that holds a copy is
/// named: the digest of its text, which is every file's identity, the element type, the layout
/// name and the region, and nothing about which kernel reads it.
pub(crate) fn storage_key(weight: &WeightKey) -> Key {
    storage::key::weight(&weight.to_string())
}

/// The allocation a weight of `length` bytes takes when it has one of its own: its length
/// rounded up to `granularity`, in which the Storage Server imports a retained allocation.
/// `None` when the rounded length overflows.
///
/// A weight's length is its shape's, so it is a whole multiple of the granularity only by
/// accident: an embedding of 622329856 bytes against a granularity of 2097152 is refused by
/// the import unless its allocation is rounded.
fn allocation_bytes(length: usize, granularity: usize) -> Option<usize> {
    length.div_ceil(granularity).checked_mul(granularity)
}

/// Where a weight of `length` bytes goes in a shared allocation of `allocation` bytes that
/// holds weights at `taken`: at `at` when that range is free, else at the aligned end of what
/// the allocation holds; `None` when it does not fit there.
fn packed_offset(
    taken: &[Range<usize>],
    length: usize,
    at: Option<usize>,
    allocation: usize,
) -> Option<usize> {
    let offset = match at {
        Some(at) => at,
        None => taken
            .iter()
            .map(|range| range.end)
            .max()
            .unwrap_or(0)
            // A packed weight's first byte is aligned as a device allocation of its own is, so a
            // kernel reads a packed weight as it reads any other.
            .next_multiple_of(ops::allocator::ALIGNMENT as usize),
    };
    let range = offset..offset.checked_add(length)?;

    let free = taken
        .iter()
        .all(|taken| taken.end <= range.start || range.end <= taken.start);
    (range.end <= allocation && free).then_some(offset)
}

/// The bytes `shape` elements of `key`'s element type take.
fn byte_length(key: &WeightKey, shape: &Shape) -> Result<u64> {
    let element = ops::abi::bytes(key.dtype).ok_or_else(|| {
        Error::Unsupported(format!(
            "a weight of {} has no whole-byte element",
            ops::abi::name(key.dtype)
        ))
    })?;

    (shape.elem_count() as u64)
        .checked_mul(element)
        .ok_or_else(|| Error::Invalid(format!("{:?} elements overflow a length", shape.dims())))
}

/// A store's failure as a loader reports it for the weight `key`. A driver failure keeps its
/// type, so it is classified by its result code wherever the load's failure arrives; any other
/// failure is stated with the weight it met.
pub(crate) fn refusal(key: &WeightKey, error: Error) -> nn::Error {
    match error {
        Error::Nn(error) => error,
        Error::Cuda(error) => nn::Error::Ops(ops::Error::Platform(error)),
        other => nn::Error::Weight {
            tensor: key.to_string(),
            problem: report(&other),
        },
    }
}

/// A rank's read-only mapping of one retained weight.
///
/// The mapping is the weight's own, also when its allocation is shared with other weights, so
/// an update maps another allocation at this weight's address without moving its neighbours.
/// The mappings of a shared allocation share its physical memory.
#[derive(Clone)]
struct Mapping {
    storage: Arc<Storage>,
    /// The weight's first byte in the allocation.
    offset: u64,
    /// The weight's length in bytes, as its record states it.
    length: u64,
}

impl Mapping {
    /// The device address of the weight's first byte.
    fn address(&self) -> u64 {
        self.storage.address() + self.offset
    }
}

/// An allocation small weights are packed into, until its one storage record and their tensor
/// records are published.
struct Pack {
    /// The allocation's writable mapping, which the weights are copied in through.
    written: Arc<Storage>,
    /// The descriptor the allocation is published with, and each weight's mapping imports.
    descriptor: OwnedFd,
    /// The byte ranges its weights take.
    taken: Vec<Range<usize>>,
    /// The records of its weights, published with the allocation.
    tensors: Vec<TensorRecord>,
}

/// What a [`WeightStore`] changes while it serves loads.
#[derive(Default)]
struct State {
    /// Every weight this rank has mapped on the device, by the key of the device's copy.
    mapped: BTreeMap<Key, Mapping>,
    /// The shared allocations of the weights packed since the last [`WeightStore::finish`].
    packs: Vec<Pack>,
    /// The tables a model writes after its load, which live as long as the store.
    tables: Vec<Arc<Storage>>,
    /// The weights the load in progress reached, in its order, each with whether the model
    /// writes it after the load.
    order: Vec<(WeightKey, bool)>,
    /// How many weights this rank read back from the Storage Server instead of converting.
    from_store: usize,
    /// How many weights this rank published.
    published: usize,
    /// Of the weights published, how many were packed, and the allocations they share.
    packed: usize,
    packed_allocations: usize,
}

/// A rank's retained weights on one device: the mappings of the weights it serves from, and
/// the publication of the ones nobody published yet.
///
/// The store lives as long as the rank serves from its weights: a tensor it returned is over an
/// address the store keeps mapped, and [`WeightStore::rebind`] and [`WeightStore::forget`] are
/// the only calls that change what an address holds. A rank that serves two devices holds one
/// store per device, over the same client.
///
/// A store serves one load at a time. A load asks for its weights in a deterministic order (the
/// same code over a checkpoint of the same model asks for the same weights one after another),
/// which the store records: [`WeightStore::finish`] or [`WeightStore::abandon`] ends the load
/// and returns its order. A position in that order identifies a weight across two loads, which
/// is what pairs each serving weight with its replacement in an update although their keys
/// differ ([`paired`](crate::memory::reload::paired)).
pub struct WeightStore {
    client: Arc<StorageClient>,
    /// The device this store's weights are mapped on.
    device: DeviceId,
    /// The UUID of the device's GPU, which the key of every copy on it names.
    gpu: [u8; 16],
    /// The granularity the Storage Server imports a retained allocation in, on the device.
    granularity: usize,
    /// The stream the store's copies run on, beside the streams the rank serves on.
    stream: Stream,
    state: Mutex<State>,
}

impl WeightStore {
    /// A store over `client` that maps every weight on `device`, with nothing mapped yet.
    ///
    /// # Errors
    ///
    /// When the device does not state its identity or its allocation granularity, or gives no
    /// stream.
    pub fn new(client: Arc<StorageClient>, device: DeviceId) -> Result<Self> {
        let gpu = client.gpu(device)?;
        let granularity = HandleType::Descriptor.granularity(device)?;
        if granularity == 0 {
            return Err(Error::Invariant(format!(
                "{device} states a zero allocation granularity"
            )));
        }

        Ok(Self {
            client,
            device,
            gpu,
            granularity,
            stream: Stream::new(device)?,
            state: Mutex::new(State::default()),
        })
    }

    /// The device this store's weights are mapped on.
    pub(crate) fn device(&self) -> DeviceId {
        self.device
    }

    /// Ends the load: publishes the shared allocations of the weights packed since the last
    /// call and returns the weights the load reached, in its order, each with whether the
    /// model writes it after the load.
    ///
    /// A packed weight is retained only from here on: until its allocation is published, a
    /// reader of another process misses it.
    ///
    /// # Errors
    ///
    /// When the Storage Server refuses an allocation: its weights are not retained, and the
    /// load fails.
    pub fn finish(&self) -> Result<Vec<(WeightKey, bool)>> {
        let mut state = lock(&self.state);

        for pack in std::mem::take(&mut state.packs) {
            if pack.tensors.is_empty() {
                continue;
            }

            let record = StorageRecord {
                location: Location::Device { gpu: self.gpu },
                segment: None,
                bytes: pack.written.bytes() as u64,
                fabric: None,
            };
            self.client
                .call("publishing an allocation of packed weights", |client| {
                    client.put(pack.tensors, &[(record, pack.descriptor.as_fd())])
                })?;
        }

        Ok(std::mem::take(&mut state.order))
    }

    /// Ends a load that will not serve: returns the weights it reached, in its order, for the
    /// caller to [`forget`](Self::forget). The shared allocations it packed stay unpublished.
    pub(crate) fn abandon(&self) -> Vec<(WeightKey, bool)> {
        std::mem::take(&mut lock(&self.state).order)
    }

    /// How many weights the load in progress reached: the position of the next one.
    pub(crate) fn reached(&self) -> usize {
        lock(&self.state).order.len()
    }

    /// Records a table the load in progress reached and that this store holds no storage for.
    pub(crate) fn note_table(&self, key: &WeightKey) {
        lock(&self.state).order.push((key.clone(), true));
    }

    /// How many weights this rank read back from the Storage Server instead of converting them.
    pub fn from_store(&self) -> usize {
        lock(&self.state).from_store
    }

    /// How many weights this rank published to the Storage Server.
    ///
    /// The server keeps a weight for as long as an epoch retains it, so this is what the host
    /// gained: a load that published nothing grew the store by nothing, and one that published
    /// after a key change added that many records beside the ones the previous key left.
    pub fn published(&self) -> usize {
        lock(&self.state).published
    }

    /// The device bytes a replacement of the weight under `key` takes beside it: the weight's
    /// own length when it shares an allocation, else its whole allocation.
    ///
    /// # Errors
    ///
    /// When this rank maps no weight under `key` and the Storage Server holds none.
    pub(crate) fn replacement_bytes(&self, key: &WeightKey) -> Result<u64> {
        let mapping = self
            .mapping(&mut lock(&self.state), self.key(key))?
            .ok_or_else(|| Error::Invariant(format!("the rank maps no weight {key}")))?;

        Ok(if self.packs(mapping.length) {
            mapping.length
        } else {
            mapping.storage.bytes() as u64
        })
    }

    /// Publishes `bytes` under `key` as [`RetainedWeights::publish`] does, placed as the weight
    /// under `replaced` lies, so that [`Self::rebind`] maps it at that weight's address: a
    /// packed weight at the same offset of a shared allocation. A `replaced` this rank does not
    /// map places the weight as a first load does.
    pub(crate) fn publish_replacing(
        &self,
        key: &WeightKey,
        shape: &Shape,
        bytes: &[u8],
        replaced: &WeightKey,
    ) -> Result<Tensor> {
        self.publish_bytes(key, shape, bytes, Some(replaced))
    }

    /// Publishes under `key` the stack that is the weight under `replaced` with its slots in
    /// another order: slot `to` of the published weight holds slot `slots[to]` of the one that
    /// serves. Returns `None`, publishing nothing, when this rank maps no weight under
    /// `replaced`.
    ///
    /// The bytes are on the device already, at the address the rank serves from, so they are
    /// copied over the device's own memory rather than read from the checkpoint again: on the
    /// store's stream, beside the batches, which read the same bytes.
    ///
    /// # Errors
    ///
    /// When the two weights differ in length or `slots` is no order of the stack's slots, and
    /// as [`RetainedWeights::publish`].
    pub(crate) fn publish_rearranged(
        &self,
        key: &WeightKey,
        shape: &Shape,
        replaced: &WeightKey,
        slots: &[usize],
    ) -> Result<Option<Tensor>> {
        let length = byte_length(key, shape)?;
        let Some(source) = self.mapping(&mut lock(&self.state), self.key(replaced))? else {
            return Ok(None);
        };

        let count = slots.len() as u64;
        if count == 0
            || source.length != length
            || !length.is_multiple_of(count)
            || slots.iter().any(|from| *from as u64 >= count)
        {
            return Err(Error::Invalid(format!(
                "{key} of {length} bytes is no order of the {} slots of {replaced}, of {} bytes",
                slots.len(),
                source.length
            )));
        }
        let slot = (length / count) as usize;
        let base = source.address();

        let tensor = self.publish_with(
            key,
            shape,
            length as usize,
            Some(replaced),
            |target, stream| {
                for (to, from) in slots.iter().enumerate() {
                    // SAFETY: both ranges lie inside live device allocations of `length` bytes
                    // from the weights' first bytes: the source is the serving weight's
                    // read-only mapping, which `source` holds until the copies completed, and
                    // the destination is the writable mapping this publication created.
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
        drop(source);

        Ok(Some(tensor))
    }

    /// Puts each pair's replacement at the address of the weight that serves, and ends the
    /// rank's retention of the weights replaced. Each pair is the key that serves and the key
    /// that takes its place ([`paired`](crate::memory::reload::paired)).
    ///
    /// The address is what the rank's captured graphs and its modules' tensors hold, so it
    /// stays; the allocation behind it becomes the replacement's, which the Storage Server
    /// retains under the key a load of that checkpoint asks for. A relaunch therefore finds it
    /// without this rank saying anything, and the mapping stays read-only throughout: no
    /// process writes a published weight.
    ///
    /// The caller has retired every device read of the weights replaced. Returns how many
    /// weights moved.
    ///
    /// # Errors
    ///
    /// When the rank maps no weight under a key of a pair, or a replacement does not lie as
    /// the weight it replaces does: nothing moved, and the rank serves on. A driver failure
    /// while the weights move leaves the rank with weights of two checkpoints, and the caller
    /// retires the rank.
    pub fn rebind(&self, pairs: &[(WeightKey, WeightKey)]) -> Result<usize> {
        let mut state = lock(&self.state);

        // Every pair is checked before any mapping changes, so a refusal changes nothing.
        let mut named = BTreeSet::new();
        for (serving, replacement) in pairs {
            let (serving_key, replacement_key) = (self.key(serving), self.key(replacement));
            if !named.insert(serving_key) || !named.insert(replacement_key) {
                return Err(Error::Invariant(format!(
                    "the weight {serving} or its replacement {replacement} is named by two pairs"
                )));
            }

            let target = state.mapped.get(&serving_key).ok_or_else(|| {
                Error::Invariant(format!("the rank maps no weight {serving} to rebind"))
            })?;
            let source = state.mapped.get(&replacement_key).ok_or_else(|| {
                Error::Invariant(format!(
                    "the rank mapped no weight {replacement} to rebind onto"
                ))
            })?;

            // The replacement's allocation is mapped over the whole of the serving one: a
            // packed weight's replacement lies at its offset of a shared allocation, so the
            // address the rank's tensors and graphs hold is the replacement's first byte.
            if (target.offset, target.storage.bytes()) != (source.offset, source.storage.bytes()) {
                return Err(Error::Invariant(format!(
                    "the weight {serving} lies at byte {} of {} and its replacement at byte {} \
                     of {}",
                    target.offset,
                    target.storage.bytes(),
                    source.offset,
                    source.storage.bytes()
                )));
            }
        }

        let mut released = BTreeSet::new();
        for (serving, replacement) in pairs {
            let (serving_key, replacement_key) = (self.key(serving), self.key(replacement));
            let (Some(target), Some(source)) = (
                state.mapped.remove(&serving_key),
                state.mapped.remove(&replacement_key),
            ) else {
                return Err(Error::Invariant(format!(
                    "the weight {serving} or its replacement {replacement} left the store while \
                     it moved"
                )));
            };
            target.storage.rebind(&source.storage)?;

            // The mapping the replacement was published through ends with `source`: its
            // address is one nothing reads, and the allocation lives on behind the serving
            // address.
            state.mapped.insert(
                replacement_key,
                Mapping {
                    length: source.length,
                    ..target
                },
            );
            released.insert(serving_key);
        }

        self.client
            .end_retention(&released, "ending the retention of replaced weights")?;
        Ok(released.len())
    }

    /// Unmaps the weights of `keys` and ends the rank's retention of them: a load that will
    /// not serve leaves no storage behind. The caller holds no tensor over them that a kernel
    /// may still read.
    pub(crate) fn forget(&self, keys: &[WeightKey]) -> Result<()> {
        let mut state = lock(&self.state);

        let mut released = BTreeSet::new();
        for key in keys {
            let key = self.key(key);
            state.mapped.remove(&key);
            released.insert(key);
        }

        // A weight packed in an allocation not published yet is not published with it.
        for pack in &mut state.packs {
            pack.tensors
                .retain(|tensor| !released.contains(&tensor.key));
        }

        self.client
            .end_retention(&released, "releasing an abandoned weight load")
    }

    /// The key this device's copy of `weight` is retained under.
    fn key(&self, weight: &WeightKey) -> Key {
        storage::key::device_weight(&storage_key(weight), &self.gpu)
    }

    /// Whether a weight of `length` bytes shares an allocation with other weights rather than
    /// taking one of its own.
    fn packs(&self, length: u64) -> bool {
        length < self.granularity as u64
    }

    /// The tensor of `shape` elements of `key`'s element type over `mapping`.
    fn tensor(&self, mapping: &Mapping, key: &WeightKey, shape: &Shape) -> Result<Tensor> {
        let expected = byte_length(key, shape)?;
        if expected != mapping.length {
            return Err(Error::Invalid(format!(
                "the store holds {} bytes under {key}, and {:?} of {} take {expected}",
                mapping.length,
                shape.dims(),
                ops::abi::name(key.dtype)
            )));
        }

        Ok(ops::retained(
            mapping.address(),
            shape.clone(),
            key.dtype,
            &ops::cuda(self.device.0 as usize),
        )?)
    }

    /// The rank's mapping of the weight retained under `key`, mapped now when the rank has not
    /// mapped it yet; `None` when the Storage Server holds nothing under the key. A miss is not
    /// remembered: another rank of the deployment may publish the weight while this one serves.
    fn mapping(&self, state: &mut State, key: Key) -> Result<Option<Mapping>> {
        if let Some(mapping) = state.mapped.get(&key) {
            return Ok(Some(mapping.clone()));
        }

        let Some(mapping) = self.fetch(key)? else {
            return Ok(None);
        };
        state.mapped.insert(key, mapping.clone());
        state.from_store += 1;
        Ok(Some(mapping))
    }

    /// Maps the weight the Storage Server retains under `key` read-only on the store's device;
    /// `None` when it holds nothing under the key.
    fn fetch(&self, key: Key) -> Result<Option<Mapping>> {
        let (got, descriptors) = self.client.call("taking a retained weight", |client| {
            client.get(vec![key], Mode::Read, Vec::new())
        })?;
        let Some(tensor) = got.tensors.iter().find(|tensor| tensor.key == key) else {
            return Ok(None);
        };
        let (record, descriptor) = got
            .storages
            .iter()
            .zip(descriptors)
            .nth(tensor.storage as usize)
            .ok_or_else(|| Error::Interface("a retained weight names no storage".into()))?;

        // Host storage registered with the device is addressable from a kernel, but every
        // forward would read the model's bytes across the host link: the record is refused by
        // name, where a mapping of it would hide the cost behind a slow run.
        if let Location::Host { .. } = record.location {
            return Err(Error::Interface(format!(
                "the weight {} lies in host storage; a weight is mapped from a device allocation, \
                 so the epoch's weights are moved back to their GPUs before its engines start",
                key.hex()
            )));
        }
        let storage = Storage::map_device(
            &descriptor,
            record.bytes as usize,
            self.device,
            &[],
            Protection::ReadOnly,
        )?;

        let length = tensor.shape.iter().product::<u64>();
        let inside = tensor
            .offset
            .checked_add(length)
            .is_some_and(|end| end <= storage.bytes() as u64);
        if !inside {
            return Err(Error::Interface(format!(
                "the weight {} is recorded as {length} bytes at byte {} of an allocation of {}",
                key.hex(),
                tensor.offset,
                storage.bytes()
            )));
        }

        Ok(Some(Mapping {
            storage,
            offset: tensor.offset,
            length,
        }))
    }

    /// Publishes the host bytes `bytes` under `key`, placed as `replaced` lies when it names a
    /// weight this rank maps.
    fn publish_bytes(
        &self,
        key: &WeightKey,
        shape: &Shape,
        bytes: &[u8],
        replaced: Option<&WeightKey>,
    ) -> Result<Tensor> {
        // The key states the element type the bytes are in, so a length the shape does not
        // state would make the next load read the bytes as other numbers.
        let expected = byte_length(key, shape)?;
        if bytes.len() as u64 != expected {
            return Err(Error::Invalid(format!(
                "{:?} of {} is {expected} bytes, not {}",
                shape.dims(),
                ops::abi::name(key.dtype),
                bytes.len()
            )));
        }

        self.publish_with(key, shape, bytes.len(), replaced, |target, stream| {
            // SAFETY: `target` is the first of `bytes.len()` writable device bytes of the
            // mapping this publication created, and the pageable source is read before the
            // call returns.
            unsafe {
                platform::memcpy_async(target, bytes.as_ptr() as u64, bytes.len(), stream)?;
            }
            Ok(())
        })
    }

    /// Publishes a weight of `length` bytes under `key`, whose bytes `write` copies on the
    /// device, and returns the tensor over its read-only mapping.
    ///
    /// `write` is given the device address of the weight's first byte in a writable mapping
    /// and the store's stream to issue its copies on; the stream is waited for before the
    /// weight's record is published, so no reader maps a weight that is partly written.
    fn publish_with(
        &self,
        key: &WeightKey,
        shape: &Shape,
        length: usize,
        replaced: Option<&WeightKey>,
        write: impl FnOnce(u64, &Stream) -> Result<()>,
    ) -> Result<Tensor> {
        if length == 0 {
            return Err(Error::Invalid(format!("the weight {key} holds no bytes")));
        }
        let storage_key = self.key(key);

        let mut state = lock(&self.state);
        state.order.push((key.clone(), false));

        // Publishing a key this rank already mapped returns the tensor over the mapping it
        // holds: the bytes are the same weight.
        if let Some(mapping) = state.mapped.get(&storage_key) {
            return self.tensor(mapping, key, shape);
        }

        let mapping = if self.packs(length as u64) {
            // The replacement of a packed weight lies at that weight's offset of a shared
            // allocation of its own, which the update then maps at the serving address.
            let at = replaced
                .and_then(|replaced| state.mapped.get(&self.key(replaced)))
                .filter(|replaced| self.packs(replaced.length))
                .map(|replaced| replaced.offset as usize);

            let mapping = self.pack(&mut state, storage_key, length, at, write)?;
            state.packed += 1;
            mapping
        } else {
            // The copy of a whole allocation runs with the store free: a reader of this rank
            // waits for no device work of a load.
            drop(state);
            let mapping = self.publish_own(storage_key, length, write)?;
            state = lock(&self.state);
            mapping
        };

        let tensor = self.tensor(&mapping, key, shape)?;
        state.mapped.insert(storage_key, mapping);
        state.published += 1;
        Ok(tensor)
    }

    /// Publishes a weight of `length` bytes in an allocation of its own and maps it read-only.
    ///
    /// The allocation is the weight's length rounded up to the granularity, which is what the
    /// record states and the server imports. The tensor states the weight's own length, and
    /// the bytes past it are read by nothing.
    fn publish_own(
        &self,
        key: Key,
        length: usize,
        write: impl FnOnce(u64, &Stream) -> Result<()>,
    ) -> Result<Mapping> {
        let allocation = allocation_bytes(length, self.granularity).ok_or_else(|| {
            Error::Invalid(format!(
                "a weight of {length} bytes has no allocation of whole multiples of {}",
                self.granularity
            ))
        })?;

        let (written, descriptor) = Storage::retained(self.device, allocation, &[])?;
        write(written.address(), &self.stream)?;
        self.stream.synchronize()?;

        let record = StorageRecord {
            location: Location::Device { gpu: self.gpu },
            segment: None,
            bytes: written.bytes() as u64,
            fabric: None,
        };
        let tensor = byte_tensor(key, weight_name(&key), 0, length as u64, Access::ReadOnly);
        self.client.call("publishing a retained weight", |client| {
            client.put(vec![tensor], &[(record, descriptor.as_fd())])
        })?;
        drop(written);

        // The weight is mapped as any reader maps it. When another rank published the key
        // first, the server kept that allocation, and this is the one both read.
        self.fetch(key)?.ok_or_else(|| {
            Error::Interface("a published weight was not found under its own key".into())
        })
    }

    /// Places a weight of `length` bytes in an allocation shared with other weights smaller
    /// than the granularity, at `at` or the first free aligned offset, has `write` copy its
    /// bytes in and maps it read-only. The allocation is published by [`Self::finish`].
    fn pack(
        &self,
        state: &mut State,
        key: Key,
        length: usize,
        at: Option<usize>,
        write: impl FnOnce(u64, &Stream) -> Result<()>,
    ) -> Result<Mapping> {
        let allocation = self.granularity;

        let found = state.packs.iter().enumerate().find_map(|(index, pack)| {
            packed_offset(&pack.taken, length, at, allocation).map(|offset| (index, offset))
        });
        let (index, offset) = match found {
            Some(found) => found,
            None => {
                let offset = packed_offset(&[], length, at, allocation).ok_or_else(|| {
                    Error::Invariant(format!(
                        "a weight of {length} bytes at {at:?} does not fit an empty allocation \
                         of {allocation} bytes"
                    ))
                })?;
                let (written, descriptor) = Storage::retained(self.device, allocation, &[])?;

                state.packs.push(Pack {
                    written,
                    descriptor,
                    taken: Vec::new(),
                    tensors: Vec::new(),
                });
                state.packed_allocations += 1;
                (state.packs.len() - 1, offset)
            }
        };
        let pack = &mut state.packs[index];

        write(pack.written.address() + offset as u64, &self.stream)?;
        self.stream.synchronize()?;

        // The weight's own read-only mapping of the allocation: the writable one is the pack's
        // until it is published, and an update rebinds this one alone.
        let storage = Storage::map_device(
            &pack.descriptor,
            allocation,
            self.device,
            &[],
            Protection::ReadOnly,
        )?;
        pack.taken.push(offset..offset + length);
        pack.tensors.push(byte_tensor(
            key,
            weight_name(&key),
            offset as u64,
            length as u64,
            Access::ReadOnly,
        ));

        Ok(Mapping {
            storage,
            offset: offset as u64,
            length: length as u64,
        })
    }

    /// The weight retained under `key` as a tensor, when the store or the server holds it.
    fn find(&self, key: &WeightKey, shape: &Shape) -> Result<Option<Tensor>> {
        let mut state = lock(&self.state);

        let Some(mapping) = self.mapping(&mut state, self.key(key))? else {
            return Ok(None);
        };
        let tensor = self.tensor(&mapping, key, shape)?;
        state.order.push((key.clone(), false));
        Ok(Some(tensor))
    }

    /// A table of `shape` elements holding `bytes`, in device storage of this process that
    /// kernels may write.
    fn table(&self, key: &WeightKey, shape: &Shape, bytes: &[u8]) -> Result<Tensor> {
        let expected = byte_length(key, shape)?;
        if bytes.len() as u64 != expected {
            return Err(Error::Invalid(format!(
                "{:?} of {} is {expected} bytes, not {}",
                shape.dims(),
                ops::abi::name(key.dtype),
                bytes.len()
            )));
        }

        let storage = Storage::retainable(self.device, bytes.len())?;
        // SAFETY: the storage holds at least `bytes.len()` writable device bytes, and the
        // pageable source is read before the call returns.
        unsafe {
            platform::memcpy_async(
                storage.address(),
                bytes.as_ptr() as u64,
                bytes.len(),
                &self.stream,
            )?;
        }
        self.stream.synchronize()?;

        let tensor = ops::retained(
            storage.address(),
            shape.clone(),
            key.dtype,
            &ops::cuda(self.device.0 as usize),
        )?;

        let mut state = lock(&self.state);
        state.order.push((key.clone(), true));
        state.tables.push(storage);
        Ok(tensor)
    }
}

impl Drop for WeightStore {
    /// One line when the store goes: what this rank's loads read back and what they published.
    ///
    /// A reader looking at the Storage Server's size after a key change cannot tell a rank that
    /// reused its weights from one that grew the store; these two figures do, once per store
    /// because the question is about the store and not about a weight.
    fn drop(&mut self) {
        let state = lock(&self.state);
        if state.from_store == 0 && state.published == 0 {
            return;
        }

        tracing::info!(
            device = %self.device,
            mapped = state.mapped.len(),
            from_store = state.from_store,
            published = state.published,
            packed = state.packed,
            packed_allocations = state.packed_allocations,
            "the rank's retained weights: what it mapped, read back and published"
        );
    }
}

impl RetainedWeights for WeightStore {
    fn retained(&self, key: &WeightKey, shape: &Shape) -> nn::Result<Option<Tensor>> {
        self.find(key, shape).map_err(|error| refusal(key, error))
    }

    fn publish(&self, key: &WeightKey, shape: &Shape, bytes: &[u8]) -> nn::Result<Tensor> {
        self.publish_bytes(key, shape, bytes, None)
            .map_err(|error| refusal(key, error))
    }

    /// The table lives in this process alone and is not retained: the launch that fills it
    /// runs in every load.
    fn writable(&self, key: &WeightKey, shape: &Shape, bytes: &[u8]) -> nn::Result<Tensor> {
        self.table(key, shape, bytes)
            .map_err(|error| refusal(key, error))
    }
}

/// The name a weight's record is listed under at the Storage Server.
fn weight_name(key: &Key) -> String {
    format!("weight-{}", key.hex())
}

#[cfg(test)]
mod tests {
    use nn::{DType, WeightLayout};

    use super::*;
    use crate::memory::fixtures::identity;

    fn key(directory: &std::path::Path, file: &str, dtype: DType, tensor: &str) -> WeightKey {
        WeightKey {
            files: vec![identity(directory, file)],
            dtype,
            layout: WeightLayout::ROW_MAJOR,
            tensors: vec![tensor.to_owned()],
            start: 0,
            shape: Shape::from((4, 8)),
            stride: None,
            provenance: None,
        }
    }

    #[test]
    fn a_changed_file_element_type_layout_or_region_changes_the_key() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let base = key(root, "aaa", DType::BF16, "model.layers.0.mlp.up");

        // The same weight reached again is the same key.
        assert_eq!(
            storage_key(&base),
            storage_key(&key(root, "aaa", DType::BF16, "model.layers.0.mlp.up"))
        );

        // The same tensor read from another checkpoint file.
        assert_ne!(
            storage_key(&base),
            storage_key(&key(root, "bbb", DType::BF16, "model.layers.0.mlp.up"))
        );
        // The same file and tensor in another representation.
        assert_ne!(
            storage_key(&base),
            storage_key(&key(root, "aaa", DType::F32, "model.layers.0.mlp.up"))
        );
        // Another tensor of the same file.
        assert_ne!(
            storage_key(&base),
            storage_key(&key(root, "aaa", DType::BF16, "model.layers.0.mlp.down"))
        );
        // Another layout of the same bytes.
        let other = WeightKey {
            layout: WeightLayout::UP_GATE,
            ..base.clone()
        };
        assert_ne!(storage_key(&base), storage_key(&other));
    }

    /// A rank's load publishes through its store on one thread while the engine reads the store
    /// on another and the worker rebinds it at a boundary, all over the rank's one client.
    #[test]
    fn a_store_and_its_client_are_shared_by_the_threads_of_a_rank() {
        fn shared<T: Send + Sync>() {}
        shared::<WeightStore>();
        shared::<StorageClient>();
    }

    /// A weight's length is its shape's, and the Storage Server imports whole multiples of the
    /// granularity: the allocation is the length rounded up, and a multiple stays as it is.
    #[test]
    fn a_weight_of_its_own_takes_its_length_rounded_up_to_the_granularity() {
        assert_eq!(allocation_bytes(130, 64), Some(192));
        assert_eq!(allocation_bytes(128, 64), Some(128));
        assert_eq!(
            allocation_bytes(622_329_856, 2_097_152),
            Some(297 * 2_097_152)
        );
        assert_eq!(allocation_bytes(usize::MAX, 2_097_152), None);
    }

    /// Packed weights lie back to back at offsets a kernel can read from, and a weight that
    /// replaces another lies at that weight's offset of an allocation where the range is free.
    #[test]
    fn a_packed_weight_lies_at_an_aligned_free_offset_of_its_allocation() {
        const ALLOCATION: usize = 1 << 21;

        // The offset of a weight of `length` bytes in an allocation whose weights take the
        // byte ranges `taken`, each a first byte and the byte past the last.
        let offset = |taken: &[[usize; 2]], length: usize, at: Option<usize>| {
            let taken: Vec<Range<usize>> = taken.iter().map(|[start, end]| *start..*end).collect();
            packed_offset(&taken, length, at, ALLOCATION)
        };

        // The first weight starts the allocation, and the next one follows at the alignment.
        assert_eq!(offset(&[], 64, None), Some(0));
        assert_eq!(offset(&[[0, 64]], 100, None), Some(256));
        assert_eq!(offset(&[[0, 64], [256, 356]], 8, None), Some(512));

        // A weight that does not fit behind what the allocation holds goes elsewhere, and one
        // that ends at the allocation's last byte fits.
        let nearly_full = [[0, ALLOCATION - 256]];
        assert_eq!(offset(&nearly_full, 257, None), None);
        assert_eq!(offset(&nearly_full, 256, None), Some(ALLOCATION - 256));

        // A replacement is placed where the weight it replaces lies, when nothing holds the
        // range there and the allocation holds the weight from there.
        assert_eq!(offset(&[], 64, Some(512)), Some(512));
        assert_eq!(offset(&[[0, 64]], 64, Some(512)), Some(512));
        assert_eq!(offset(&[[500, 600]], 64, Some(512)), None);
        assert_eq!(offset(&[], 64, Some(ALLOCATION - 32)), None);
    }
}
