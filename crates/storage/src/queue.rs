//! The queue pair between a scheduler and the buffer pool: a submission queue
//! and a completion queue in shared memory, with the registers the two sides
//! publish levels in.
//!
//! One memfd holds a header, the registers and two rings. The engine's end
//! produces the submission queue and consumes the completion queue; the
//! buffer pool's end does the opposite. Each ring has one producer index and
//! one consumer index, each written by one side only, so neither side takes a
//! lock, makes a system call or waits for the other: a write into a full ring
//! is refused and the caller keeps the record.
//!
//! A record is a length word followed by its bytes, padded to eight bytes; what
//! the bytes mean is the protocol's. An index counts the bytes ever produced
//! or consumed, so the bytes in a ring are the difference of its two indices.
//!
//! The layout is what two builds share, in bytes from the start of the memfd,
//! every word a native 64-bit integer: the magic at 0, a ring's bytes at 8 and
//! the number of registers at 16; the submission queue's producer index at 128
//! and its consumer index at 256; the completion queue's producer index at 384
//! and its consumer index at 512; the registers from 640; then, from the next
//! multiple of 128, the submission queue's bytes followed by the completion
//! queue's.
//!
//! Each end keeps its own index privately and only publishes it, and checks
//! the other side's index and every length word before it uses them, so a
//! side that writes nonsense into the shared memory gets an error from the
//! other and cannot make it read or write outside the rings.

use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{Error, Result};

/// Identifies the layout below; a mapping that does not start with it is not
/// a queue pair of this layout.
const MAGIC: u64 = 0x756e_6973_7170_0001;

/// The separation of words written by different sides, so that one side's
/// index does not share a cache line with the other's.
const LINE: usize = 128;

/// The header: the magic, a ring's bytes, the number of registers.
const HEADER_RING_BYTES: usize = 8;
const HEADER_REGISTERS: usize = 16;

/// The four indices, one cache line each, in the order: submission produced,
/// submission consumed, completion produced, completion consumed.
const INDICES: usize = LINE;

/// The registers follow the indices.
const REGISTERS: usize = INDICES + 4 * LINE;

/// The bytes of a record's length word.
const LENGTH: usize = 8;

/// One end of a queue pair.
pub struct QueuePair {
    /// The mapping of the whole memfd.
    base: NonNull<u8>,
    bytes: usize,
    memory: OwnedFd,

    /// A ring's bytes, a power of two.
    ring_bytes: usize,
    registers: usize,
    /// Where the submission ring's bytes begin; the completion ring follows.
    rings: usize,

    /// Whether this is the engine's end.
    engine: bool,
    /// This end's own indices: the bytes it produced into the ring it writes
    /// and the bytes it consumed from the ring it reads.
    produced: u64,
    consumed: u64,
}

// SAFETY: the mapping is shared memory addressed only through atomics and
// byte copies; moving an end to another thread moves its private indices with
// it, and `&mut self` keeps each end's ring operations on one thread at a time.
unsafe impl Send for QueuePair {}

impl QueuePair {
    /// Creates a queue pair whose rings each hold `ring_bytes` (a power of two
    /// of at least 64) with `registers` registers, all zero, and returns the
    /// buffer pool's end.
    pub fn create(ring_bytes: usize, registers: usize) -> Result<Self> {
        if !ring_bytes.is_power_of_two() || ring_bytes < 64 {
            return Err(Error::Protocol(format!(
                "a ring of {ring_bytes} bytes is not a power of two of at least 64"
            )));
        }
        let rings = (REGISTERS + registers * 8).next_multiple_of(LINE);
        let bytes = rings + 2 * ring_bytes;

        // SAFETY: the name is a NUL-terminated literal; the call creates a new
        // descriptor this process owns.
        let descriptor =
            unsafe { libc::memfd_create(c"uniserve-queue-pair".as_ptr(), libc::MFD_CLOEXEC) };
        if descriptor < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        // SAFETY: the descriptor was just created and is owned here alone.
        let memory = unsafe { OwnedFd::from_raw_fd(descriptor) };
        // SAFETY: a plain system call on an owned descriptor.
        if unsafe { libc::ftruncate(memory.as_raw_fd(), bytes as libc::off_t) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }

        let base = map(&memory, bytes)?;
        // SAFETY: the mapping holds `bytes` zeroed bytes, the header lies inside
        // it at eight-byte aligned offsets, and nothing else maps it yet.
        unsafe {
            base.as_ptr()
                .add(HEADER_RING_BYTES)
                .cast::<u64>()
                .write(ring_bytes as u64);
            base.as_ptr()
                .add(HEADER_REGISTERS)
                .cast::<u64>()
                .write(registers as u64);
            base.as_ptr().cast::<u64>().write(MAGIC);
        }
        Ok(Self {
            base,
            bytes,
            memory,
            ring_bytes,
            registers,
            rings,
            engine: false,
            produced: 0,
            consumed: 0,
        })
    }

    /// Maps an existing queue pair as the engine's end (`engine`) or as the
    /// buffer pool's, which a Storage Server restarted in place does with the
    /// queue pairs it receives. The end continues from the indices the
    /// previous holder of that end published.
    pub fn open(memory: OwnedFd, engine: bool) -> Result<Self> {
        // SAFETY: `stat` is plain data the call fills for an owned descriptor.
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: as above.
        if unsafe { libc::fstat(memory.as_raw_fd(), &mut stat) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let bytes = stat.st_size as usize;
        if bytes < REGISTERS {
            return Err(Error::Protocol(format!(
                "a queue pair of {bytes} bytes has no header"
            )));
        }

        let base = map(&memory, bytes)?;
        // SAFETY: the mapping holds at least the header, at aligned offsets.
        let (magic, ring_bytes, registers) = unsafe {
            (
                base.as_ptr().cast::<u64>().read(),
                base.as_ptr().add(HEADER_RING_BYTES).cast::<u64>().read() as usize,
                base.as_ptr().add(HEADER_REGISTERS).cast::<u64>().read() as usize,
            )
        };
        let rings = registers
            .checked_mul(8)
            .and_then(|words| words.checked_add(REGISTERS))
            .map(|end| end.next_multiple_of(LINE));
        let fits = rings
            .and_then(|rings| ring_bytes.checked_mul(2)?.checked_add(rings))
            .is_some_and(|total| total == bytes);
        if magic != MAGIC || !ring_bytes.is_power_of_two() || ring_bytes < 64 || !fits {
            // SAFETY: the mapping was made above with this length.
            unsafe { libc::munmap(base.as_ptr().cast(), bytes) };
            return Err(Error::Protocol(
                "the memory is not a queue pair of this layout".into(),
            ));
        }

        let mut pair = Self {
            base,
            bytes,
            memory,
            ring_bytes,
            registers,
            rings: rings.unwrap_or_default(),
            engine,
            produced: 0,
            consumed: 0,
        };
        pair.produced = pair.index(pair.writes(), false).load(Ordering::Acquire);
        pair.consumed = pair.index(pair.reads(), true).load(Ordering::Acquire);
        Ok(pair)
    }

    /// The memfd, which travels to the other end and to a successor.
    pub fn descriptor(&self) -> BorrowedFd<'_> {
        self.memory.as_fd()
    }

    /// The largest record a ring takes.
    pub fn max_record(&self) -> usize {
        self.ring_bytes - LENGTH
    }

    /// Register `index`: one word both sides read, written by the side the
    /// protocol names for it.
    pub fn register(&self, index: usize) -> &AtomicU64 {
        assert!(
            index < self.registers,
            "register {index} of {}",
            self.registers
        );
        // SAFETY: the register lies inside the mapping at an aligned offset, and
        // the mapping lives as long as `self`.
        unsafe {
            &*self
                .base
                .as_ptr()
                .add(REGISTERS + index * 8)
                .cast::<AtomicU64>()
        }
    }

    /// Writes a request into the submission queue (the engine's end). False:
    /// the queue has no room, and nothing was written.
    pub fn submit(&mut self, request: &[u8]) -> Result<bool> {
        debug_assert!(self.engine, "only the engine's end submits");
        self.push(request)
    }

    /// Takes the next completion (the engine's end) into `completion`. False:
    /// none has arrived.
    pub fn next_completion(&mut self, completion: &mut Vec<u8>) -> Result<bool> {
        debug_assert!(self.engine, "only the engine's end reads completions");
        self.pop(completion)
    }

    /// Takes the next request (the buffer pool's end) into `request`. False:
    /// the submission queue is empty.
    pub fn next_request(&mut self, request: &mut Vec<u8>) -> Result<bool> {
        debug_assert!(!self.engine, "only the buffer pool's end reads requests");
        self.pop(request)
    }

    /// Writes a completion into the completion queue (the buffer pool's end).
    /// False: the queue has no room, and nothing was written.
    pub fn complete(&mut self, completion: &[u8]) -> Result<bool> {
        debug_assert!(!self.engine, "only the buffer pool's end completes");
        self.push(completion)
    }

    /// The bytes the ring this end writes can still take, records' length words
    /// included.
    pub fn room(&self) -> Result<usize> {
        let consumed = self.index(self.writes(), true).load(Ordering::Acquire);
        let used = self.produced.wrapping_sub(consumed);
        if used > self.ring_bytes as u64 {
            return Err(Error::Protocol(format!(
                "a queue's consumer index {consumed} is not behind its producer index {}",
                self.produced
            )));
        }
        Ok(self.ring_bytes - used as usize)
    }

    /// The ring this end writes: 0 is the submission queue, 1 the completion queue.
    fn writes(&self) -> usize {
        if self.engine { 0 } else { 1 }
    }

    /// The ring this end reads.
    fn reads(&self) -> usize {
        1 - self.writes()
    }

    /// The producer index of `ring`, or its consumer index.
    fn index(&self, ring: usize, consumer: bool) -> &AtomicU64 {
        let offset = INDICES + (2 * ring + usize::from(consumer)) * LINE;
        // SAFETY: the index lies inside the mapping at an aligned offset, and the
        // mapping lives as long as `self`.
        unsafe { &*self.base.as_ptr().add(offset).cast::<AtomicU64>() }
    }

    /// The first byte of `ring`'s bytes.
    fn data(&self, ring: usize) -> *mut u8 {
        // SAFETY: both rings lie inside the mapping.
        unsafe { self.base.as_ptr().add(self.rings + ring * self.ring_bytes) }
    }

    fn push(&mut self, record: &[u8]) -> Result<bool> {
        let need = LENGTH + record.len().next_multiple_of(8);
        if record.len() > self.max_record() {
            return Err(Error::Protocol(format!(
                "a record of {} bytes exceeds a ring of {} bytes",
                record.len(),
                self.ring_bytes
            )));
        }
        if need > self.room()? {
            return Ok(false);
        }

        let ring = self.writes();
        let mask = self.ring_bytes - 1;
        let at = self.produced as usize & mask;
        // SAFETY: `at` is a multiple of eight below the ring's bytes, so the
        // length word lies inside the ring; the consumer does not read these
        // bytes before the index below is published.
        unsafe {
            self.data(ring)
                .add(at)
                .cast::<u64>()
                .write(record.len() as u64)
        };
        self.copy_in(ring, (at + LENGTH) & mask, record);

        self.produced += need as u64;
        self.index(ring, false)
            .store(self.produced, Ordering::Release);
        Ok(true)
    }

    fn pop(&mut self, record: &mut Vec<u8>) -> Result<bool> {
        let ring = self.reads();
        let produced = self.index(ring, false).load(Ordering::Acquire);
        let available = produced.wrapping_sub(self.consumed);
        if available == 0 {
            return Ok(false);
        }
        if available > self.ring_bytes as u64 || !available.is_multiple_of(8) {
            return Err(Error::Protocol(format!(
                "a queue's producer index {produced} does not follow its consumer index {}",
                self.consumed
            )));
        }

        let mask = self.ring_bytes - 1;
        let at = self.consumed as usize & mask;
        // SAFETY: `at` is a multiple of eight below the ring's bytes.
        let length = unsafe { self.data(ring).add(at).cast::<u64>().read() };
        let need = (length as usize)
            .checked_next_multiple_of(8)
            .map(|padded| padded + LENGTH);
        let need = match need {
            Some(need) if length <= self.max_record() as u64 && need as u64 <= available => need,
            _ => {
                return Err(Error::Protocol(format!(
                    "a record states {length} bytes where {available} were produced"
                )));
            }
        };

        record.clear();
        record.resize(length as usize, 0);
        self.copy_out(ring, (at + LENGTH) & mask, record);

        self.consumed += need as u64;
        self.index(ring, true)
            .store(self.consumed, Ordering::Release);
        Ok(true)
    }

    /// Copies `bytes` into `ring` from offset `at`, wrapping at its end.
    fn copy_in(&self, ring: usize, at: usize, bytes: &[u8]) {
        let first = bytes.len().min(self.ring_bytes - at);
        // SAFETY: `at + first` lies inside the ring and the rest starts at its
        // first byte; `bytes` is at most the ring's bytes long, and `push`
        // checked that the ring has this much room.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data(ring).add(at), first);
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr().add(first),
                self.data(ring),
                bytes.len() - first,
            );
        }
    }

    /// Copies `bytes.len()` bytes out of `ring` from offset `at`, wrapping at its end.
    fn copy_out(&self, ring: usize, at: usize, bytes: &mut [u8]) {
        let first = bytes.len().min(self.ring_bytes - at);
        // SAFETY: as `copy_in`; `pop` checked the length against the ring.
        unsafe {
            std::ptr::copy_nonoverlapping(self.data(ring).add(at), bytes.as_mut_ptr(), first);
            std::ptr::copy_nonoverlapping(
                self.data(ring),
                bytes.as_mut_ptr().add(first),
                bytes.len() - first,
            );
        }
    }
}

impl Drop for QueuePair {
    fn drop(&mut self) {
        // SAFETY: the mapping was made with this length and nothing borrows it
        // beyond `self`.
        unsafe { libc::munmap(self.base.as_ptr().cast(), self.bytes) };
    }
}

/// Maps `bytes` of `memory` shared and writable.
fn map(memory: &OwnedFd, bytes: usize) -> Result<NonNull<u8>> {
    // SAFETY: a new shared mapping of an owned descriptor at an address the
    // kernel chooses.
    let address = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            bytes,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_SHARED,
            memory.as_raw_fd(),
            0,
        )
    };
    if address == libc::MAP_FAILED {
        return Err(std::io::Error::last_os_error().into());
    }
    NonNull::new(address.cast())
        .ok_or_else(|| Error::Protocol("mmap returned a null address".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two ends of a new queue pair.
    fn ends(ring_bytes: usize, registers: usize) -> (QueuePair, QueuePair) {
        let pool = QueuePair::create(ring_bytes, registers).unwrap();
        let engine =
            QueuePair::open(pool.descriptor().try_clone_to_owned().unwrap(), true).unwrap();
        (engine, pool)
    }

    #[test]
    fn records_arrive_whole_and_in_order_across_the_end_of_a_ring() {
        let (mut engine, mut pool) = ends(64, 0);
        let mut taken = Vec::new();

        // Records of every length up to the largest, many times the ring's bytes.
        for round in 0..200usize {
            let record: Vec<u8> = (0..round % 57).map(|byte| (round + byte) as u8).collect();
            assert!(engine.submit(&record).unwrap());
            assert!(pool.next_request(&mut taken).unwrap());
            assert_eq!(taken, record);

            assert!(pool.complete(&record).unwrap());
            assert!(engine.next_completion(&mut taken).unwrap());
            assert_eq!(taken, record);
        }
        assert!(!pool.next_request(&mut taken).unwrap());
        assert!(!engine.next_completion(&mut taken).unwrap());
    }

    #[test]
    fn a_full_queue_refuses_a_record_and_takes_it_once_one_was_consumed() {
        let (mut engine, mut pool) = ends(64, 0);
        let mut taken = Vec::new();

        // Each record takes 16 bytes of the 64.
        for value in 0..4u8 {
            assert!(engine.submit(&[value; 8]).unwrap());
        }
        assert!(!engine.submit(&[4; 8]).unwrap());
        assert_eq!(engine.room().unwrap(), 0);

        assert!(pool.next_request(&mut taken).unwrap());
        assert_eq!(taken, [0; 8]);
        assert!(engine.submit(&[4; 8]).unwrap());

        for value in 1..5u8 {
            assert!(pool.next_request(&mut taken).unwrap());
            assert_eq!(taken, [value; 8]);
        }
        assert!(!pool.next_request(&mut taken).unwrap());
    }

    #[test]
    fn a_record_longer_than_a_ring_is_an_error() {
        let (mut engine, _pool) = ends(64, 0);
        assert!(engine.submit(&[0; 56]).unwrap());
        assert!(engine.submit(&[0; 57]).is_err());
    }

    #[test]
    fn a_successor_continues_from_the_published_indices() {
        let (mut engine, mut pool) = ends(64, 0);
        let mut taken = Vec::new();
        assert!(engine.submit(b"first").unwrap());
        assert!(engine.submit(b"second").unwrap());
        assert!(pool.next_request(&mut taken).unwrap());
        assert!(pool.complete(b"done").unwrap());

        // The buffer pool's end is handed to another holder.
        let mut successor =
            QueuePair::open(pool.descriptor().try_clone_to_owned().unwrap(), false).unwrap();
        drop(pool);
        assert!(successor.next_request(&mut taken).unwrap());
        assert_eq!(taken, b"second");
        assert!(successor.complete(b"again").unwrap());

        assert!(engine.next_completion(&mut taken).unwrap());
        assert_eq!(taken, b"done");
        assert!(engine.next_completion(&mut taken).unwrap());
        assert_eq!(taken, b"again");
    }

    /// Writes one word into the shared memory at `offset` of the layout.
    fn write_word(pair: &QueuePair, offset: i64, value: u64) {
        let bytes = value.to_ne_bytes();
        // SAFETY: a write of eight bytes from a live buffer to an open descriptor.
        let written = unsafe {
            libc::pwrite(
                pair.descriptor().as_raw_fd(),
                bytes.as_ptr().cast(),
                8,
                offset,
            )
        };
        assert_eq!(written, 8);
    }

    #[test]
    fn an_index_or_a_length_out_of_range_is_an_error_to_the_other_end() {
        let (mut engine, mut pool) = ends(64, 0);
        let mut taken = Vec::new();

        // The submission queue's producer index, further ahead than a ring holds.
        write_word(&engine, 128, 72);
        assert!(pool.next_request(&mut taken).is_err());

        // A length word, the first of the submission queue's bytes, that states
        // more than was produced.
        write_word(&engine, 128, 16);
        write_word(&engine, 640, 40);
        assert!(pool.next_request(&mut taken).is_err());

        // The submission queue's consumer index, ahead of what the engine produced.
        write_word(&engine, 256, 1 << 40);
        assert!(engine.submit(b"record").is_err());
    }

    #[test]
    fn registers_are_shared_and_start_at_zero() {
        let (engine, pool) = ends(64, 3);
        assert_eq!(pool.register(2).load(Ordering::Acquire), 0);
        engine.register(2).store(7, Ordering::Release);
        assert_eq!(pool.register(2).load(Ordering::Acquire), 7);
        assert_eq!(pool.register(0).load(Ordering::Acquire), 0);
    }

    #[test]
    fn memory_that_is_not_a_queue_pair_is_refused() {
        let pool = QueuePair::create(64, 0).unwrap();
        write_word(&pool, 0, 0);
        assert!(QueuePair::open(pool.descriptor().try_clone_to_owned().unwrap(), true).is_err());
    }

    #[test]
    fn two_threads_exchange_every_record_once_while_the_rings_fill_and_empty() {
        let (mut engine, mut pool) = ends(256, 0);
        const RECORDS: u64 = 200_000;

        // The buffer pool's side answers each request with its bytes.
        let server = std::thread::spawn(move || {
            let mut request = Vec::new();
            let mut served = 0;
            while served < RECORDS {
                if pool.next_request(&mut request).unwrap() {
                    while !pool.complete(&request).unwrap() {
                        std::hint::spin_loop();
                    }
                    served += 1;
                } else {
                    std::hint::spin_loop();
                }
            }
        });

        let mut completion = Vec::new();
        let (mut sent, mut received) = (0u64, 0u64);
        while received < RECORDS {
            if sent < RECORDS {
                // A record of varying length that carries its own number.
                let mut record = sent.to_le_bytes().to_vec();
                record.resize(8 + (sent % 23) as usize, sent as u8);
                if engine.submit(&record).unwrap() {
                    sent += 1;
                }
            }
            while engine.next_completion(&mut completion).unwrap() {
                let number = u64::from_le_bytes(completion[..8].try_into().unwrap());
                assert_eq!(number, received);
                assert_eq!(completion.len(), 8 + (number % 23) as usize);
                assert!(completion[8..].iter().all(|byte| *byte == number as u8));
                received += 1;
            }
        }
        server.join().unwrap();
    }
}
