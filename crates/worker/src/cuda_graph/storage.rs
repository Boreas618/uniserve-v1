//! The device storage of the rank's graph pools and scratch, which Rust
//! owns.
//!
//! Each pool is a [`VirtualPool`]: device storage at addresses that stay
//! fixed for the pool's life, which the captured graphs and the operator
//! scratch address. torch allocates from it through a pluggable allocator
//! (`torch.cuda.MemPool(allocator=...)`, `uniserve.cuda_graph.pool`): its
//! caching allocator asks the pool for segments and sub-allocates its
//! blocks in them, so captures sharing a pool reuse each other's freed
//! blocks as they do in a pool of torch's own. A segment torch frees stays
//! mapped and is given out again by the same pool: a graph may still
//! address it, so its physical storage returns only when Rust drops the pool,
//! after the graphs that address it.
//!
//! torch's allocator calls carry no pool, so the library exports one pair
//! of entry points per pool slot (`uniserve_graph_malloc_<slot>`,
//! `uniserve_graph_free_<slot>`); a pool's torch allocator names its slot's
//! pair, and its calls reach its pool from whichever thread allocates.
//!
//! A standby worker suspends its pools while it parks ([`PoolSlot::suspend`]):
//! their contents move to host storage and their physical storage is released,
//! keeping every address, and its promotion restores them before the graphs
//! run again.

use std::ffi::{CStr, c_int, c_void};
use std::sync::{Arc, Mutex};

use cuda::{DeviceId, VirtualPool};

use crate::{Error, Result, lock};

/// The pools a process can hold at once.
pub(crate) const SLOTS: usize = 32;

/// The pool of each slot, while its [`PoolSlot`] lives.
static POOLS: [Mutex<Option<Arc<Mutex<VirtualPool>>>>; SLOTS] = [const { Mutex::new(None) }; SLOTS];

/// One pool of device storage and the slot its torch allocator calls.
pub(crate) struct PoolSlot {
    slot: usize,
    pool: Arc<Mutex<VirtualPool>>,
}

impl PoolSlot {
    /// Reserves `bytes` of addresses on `device` for a new pool in a free
    /// slot.
    ///
    /// # Errors
    ///
    /// Fails when every slot holds a pool or the reservation fails.
    pub(crate) fn reserve(device: DeviceId, bytes: usize) -> Result<Self> {
        let pool = Arc::new(Mutex::new(VirtualPool::reserve(device, bytes)?));
        for (slot, entry) in POOLS.iter().enumerate() {
            let mut entry = lock(entry);
            if entry.is_none() {
                *entry = Some(Arc::clone(&pool));
                return Ok(Self { slot, pool });
            }
        }
        Err(Error::State("every graph storage slot holds a pool"))
    }

    /// The slot whose entry points the pool's torch allocator calls.
    pub(crate) fn slot(&self) -> usize {
        self.slot
    }

    /// The bytes of device storage the pool maps.
    pub(crate) fn mapped_bytes(&self) -> usize {
        lock(&self.pool).mapped_bytes()
    }

    /// Copies the pool's contents to host storage and releases its device
    /// storage, keeping its addresses. The caller drained every stream that
    /// reads or writes the pool and launches nothing that does until
    /// [`Self::restore`].
    pub(crate) fn suspend(&self) -> Result<()> {
        Ok(lock(&self.pool).suspend()?)
    }

    /// Maps the pool's device storage back at its addresses with its
    /// contents.
    pub(crate) fn restore(&self) -> Result<()> {
        Ok(lock(&self.pool).restore()?)
    }
}

impl Drop for PoolSlot {
    fn drop(&mut self) {
        // The pool's storage is released once torch's last call into the
        // slot returned; later calls find no pool.
        *lock(&POOLS[self.slot]) = None;
    }
}

/// The path of the library holding the allocator entry points, which a
/// torch pluggable allocator loads by name.
pub(crate) fn library_path() -> Result<String> {
    // SAFETY: `Dl_info` is plain data, all of whose fields may be zero.
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    let address = uniserve_graph_malloc_0 as *const c_void;
    // SAFETY: dladdr reads the address of a function of this library and
    // writes the local `info`.
    let found = unsafe { libc::dladdr(address, &mut info) };
    if found == 0 || info.dli_fname.is_null() {
        return Err(Error::State("the worker library's path is unknown"));
    }
    // SAFETY: dladdr returned a NUL-terminated path owned by the loader.
    let path = unsafe { CStr::from_ptr(info.dli_fname) };
    Ok(path.to_string_lossy().into_owned())
}

/// A segment of `size` bytes from the pool of `slot`; null when the slot
/// holds no pool or the pool cannot give it out, which torch reports as an
/// out-of-storage error.
fn allocate(slot: usize, size: isize) -> *mut c_void {
    let Some(pool) = lock(&POOLS[slot]).clone() else {
        tracing::warn!(
            slot,
            "torch allocated from a graph storage slot that holds no pool"
        );
        return std::ptr::null_mut();
    };
    let bytes = usize::try_from(size).unwrap_or(0);
    match lock(&pool).allocate(bytes) {
        Ok(address) => address as *mut c_void,
        Err(error) => {
            tracing::warn!(slot, bytes, %error, "a graph storage pool cannot give out a segment");
            std::ptr::null_mut()
        }
    }
}

/// torch freed the segment at `address` of the pool of `slot`.
fn free(slot: usize, address: *mut c_void) {
    if let Some(pool) = lock(&POOLS[slot]).clone() {
        lock(&pool).free(address as u64);
    }
}

/// The allocator entry points of each slot, with the signatures torch's
/// `CUDAPluggableAllocator` calls: `malloc(size, device, stream)` and
/// `free(pointer, size, device, stream)`.
macro_rules! entry_points {
    ($($slot:literal => $malloc:ident, $free:ident;)*) => {
        $(
            #[unsafe(no_mangle)]
            pub(crate) extern "C" fn $malloc(size: isize, _device: c_int, _stream: *mut c_void) -> *mut c_void {
                allocate($slot, size)
            }

            #[unsafe(no_mangle)]
            pub(crate) extern "C" fn $free(
                address: *mut c_void,
                _size: isize,
                _device: c_int,
                _stream: *mut c_void,
            ) {
                free($slot, address)
            }
        )*
    };
}

entry_points! {
    0 => uniserve_graph_malloc_0, uniserve_graph_free_0;
    1 => uniserve_graph_malloc_1, uniserve_graph_free_1;
    2 => uniserve_graph_malloc_2, uniserve_graph_free_2;
    3 => uniserve_graph_malloc_3, uniserve_graph_free_3;
    4 => uniserve_graph_malloc_4, uniserve_graph_free_4;
    5 => uniserve_graph_malloc_5, uniserve_graph_free_5;
    6 => uniserve_graph_malloc_6, uniserve_graph_free_6;
    7 => uniserve_graph_malloc_7, uniserve_graph_free_7;
    8 => uniserve_graph_malloc_8, uniserve_graph_free_8;
    9 => uniserve_graph_malloc_9, uniserve_graph_free_9;
    10 => uniserve_graph_malloc_10, uniserve_graph_free_10;
    11 => uniserve_graph_malloc_11, uniserve_graph_free_11;
    12 => uniserve_graph_malloc_12, uniserve_graph_free_12;
    13 => uniserve_graph_malloc_13, uniserve_graph_free_13;
    14 => uniserve_graph_malloc_14, uniserve_graph_free_14;
    15 => uniserve_graph_malloc_15, uniserve_graph_free_15;
    16 => uniserve_graph_malloc_16, uniserve_graph_free_16;
    17 => uniserve_graph_malloc_17, uniserve_graph_free_17;
    18 => uniserve_graph_malloc_18, uniserve_graph_free_18;
    19 => uniserve_graph_malloc_19, uniserve_graph_free_19;
    20 => uniserve_graph_malloc_20, uniserve_graph_free_20;
    21 => uniserve_graph_malloc_21, uniserve_graph_free_21;
    22 => uniserve_graph_malloc_22, uniserve_graph_free_22;
    23 => uniserve_graph_malloc_23, uniserve_graph_free_23;
    24 => uniserve_graph_malloc_24, uniserve_graph_free_24;
    25 => uniserve_graph_malloc_25, uniserve_graph_free_25;
    26 => uniserve_graph_malloc_26, uniserve_graph_free_26;
    27 => uniserve_graph_malloc_27, uniserve_graph_free_27;
    28 => uniserve_graph_malloc_28, uniserve_graph_free_28;
    29 => uniserve_graph_malloc_29, uniserve_graph_free_29;
    30 => uniserve_graph_malloc_30, uniserve_graph_free_30;
    31 => uniserve_graph_malloc_31, uniserve_graph_free_31;
}
