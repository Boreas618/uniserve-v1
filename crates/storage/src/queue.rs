//! The queue pair between a scheduler and the buffer pool: a submission queue
//! and a completion queue in shared memory, with the registers the two sides
//! publish levels in, and a doorbell for each end.
//!
//! One memfd holds a header, the doorbell words, the registers and two rings.
//! The engine's end produces the submission queue and consumes the completion
//! queue; the buffer pool's end does the opposite. Each ring has one producer
//! index and one consumer index, each written by one side only, so neither
//! side takes a lock to use a ring: a write into a full ring is refused and
//! the caller keeps the record.
//!
//! A record is a length word followed by its bytes, padded to eight bytes; what
//! the bytes mean is the protocol's. An index counts the bytes ever produced
//! or consumed, so the bytes in a ring are the difference of its two indices.
//!
//! **Doorbells.** Every reader and writer of a queue pair is a host thread:
//! the scheduler's thread and the engine's workers on one side (a worker
//! writes its group's `completed` register after it awaited the batch's
//! stream event), the buffer pool's thread on the other. No device code reads
//! or writes a queue pair, so every wait on one is a host wait, and each end
//! blocks on a doorbell instead of polling. A doorbell is an eventfd per end,
//! made with the queue pair by the buffer pool's end and travelling with its
//! memfd ([`QueueDescriptors`]).
//!
//! An end that is about to sleep declares what it waits for ([`Interest`]:
//! records the other end writes, room the other end makes by consuming,
//! registers the other end writes) in a word of its own and checks that
//! nothing of that kind arrived since its last look ([`QueuePair::arm`]);
//! then it blocks on its doorbell ([`QueuePair::doorbell`]) among whatever
//! else it waits on. The other end counts every publication by kind
//! ([`QueuePair::submit`] and [`QueuePair::complete`] write records, the reads
//! that consume them make room, [`QueuePair::notify_registers`] follows
//! register writes) and rings the doorbell only when the sleeping end declared
//! that kind, clearing the declaration as it rings. A busy end therefore costs
//! the other end no system call, at most one ring reaches an end per sleep,
//! and no publication is lost between an end's last look and its sleep: the
//! declaration and the count are each followed by a sequentially consistent
//! fence before the other side's word is read, so either the sleeping end
//! sees the count or the publishing end sees the declaration.
//!
//! The layout is what two builds share, in bytes from the start of the memfd,
//! every word a native 64-bit integer: the magic at 0, a ring's bytes at 8, the
//! number of registers at 16 and the separation `S` of words written by
//! different sides at 24; the submission queue's producer index at `S` and its
//! consumer index at `2S`; the completion queue's producer index at `3S` and its
//! consumer index at `4S`; the buffer pool's end's declared interest at `5S` and
//! the counts of the engine's publications to it at `6S` (records, room,
//! registers, a word each); the engine's end's declared interest at `7S` and the
//! counts of the buffer pool's publications to it at `8S`; the registers from
//! `9S`; then, from the next multiple of `S`, the submission queue's bytes
//! followed by the completion queue's. The creator states `S` in the header
//! ([`SEPARATION`] of its build's target), and an end that opens the memory lays
//! the pair out by the header's value.
//!
//! Each end keeps its own index privately and only publishes it, and checks
//! the other side's index and every length word before it uses them, so a
//! side that writes nonsense into the shared memory gets an error from the
//! other and cannot make it read or write outside the rings. The doorbell
//! words are hints: a side that writes nonsense into them wakes the other end
//! without cause or delays its own wake-ups, and reaches no other queue pair.

use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering, fence};
use std::time::{Duration, Instant};

use crate::{Error, Result};

/// Identifies the layout below; a mapping that does not start with it is not
/// a queue pair of this layout.
const MAGIC: u64 = 0x756e_6973_7170_0003;

/// The separation of words written by different sides, so that one side's
/// index shares no cache line, nor the pair of lines an adjacent-line prefetcher
/// fetches together, with the other's: the destructive interference size of the
/// build's target, as crossbeam's `CachePadded` declares it per architecture
/// (128 bytes on x86-64 and AArch64). A queue pair's creator states it in the
/// header.
pub const SEPARATION: usize = std::mem::align_of::<crossbeam_utils::CachePadded<u8>>();

/// The header: the magic, a ring's bytes, the number of registers and the
/// separation, a word each.
const HEADER_RING_BYTES: usize = 8;
const HEADER_REGISTERS: usize = 16;
const HEADER_SEPARATION: usize = 24;
const HEADER_BYTES: usize = 32;

/// The words written by different sides, in separations from the start of the memfd: the four
/// indices after the header's separation, in the order submission produced, submission consumed,
/// completion produced, completion consumed; then the doorbell words, per end, the buffer pool's
/// first, a separation holding the kinds it waits for, written by that end, and a separation
/// holding the count of each kind the other end published, written by the other end.
const INDICES: usize = 1;
const DOORBELLS: usize = INDICES + 4;

/// The registers follow the doorbell words, in separations from the start of the memfd.
const REGISTERS: usize = DOORBELLS + 4;

/// Where the registers begin in a layout of separation `separation`.
fn registers_at(separation: usize) -> Option<usize> {
    separation.checked_mul(REGISTERS)
}

/// Where the rings begin in a layout of separation `separation` with `registers` registers: the
/// next multiple of the separation after them.
fn rings_at(separation: usize, registers: usize) -> Option<usize> {
    registers
        .checked_mul(8)?
        .checked_add(registers_at(separation)?)?
        .checked_next_multiple_of(separation)
}

/// The bytes of a record's length word.
const LENGTH: usize = 8;

/// The buffer pool's end and the engine's, as the doorbell words and the
/// doorbells are indexed.
const POOL: usize = 0;
const ENGINE: usize = 1;

/// The kinds of publication an end counts for the other, one count word each
/// in the order of [`Interest`]'s bits.
const KINDS: usize = 3;

bitflags::bitflags! {
    /// What an end of a queue pair waits for: kinds of publication of the
    /// other end's, each of which rings this end's doorbell while this end
    /// declares it ([`QueuePair::arm`]).
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct Interest: u64 {
        /// A record the other end wrote into the ring this end reads.
        const RECORDS = 1 << 0;
        /// Room the other end made in the ring this end writes, by consuming
        /// records of it.
        const ROOM = 1 << 1;
        /// Registers the other end wrote ([`QueuePair::notify_registers`]).
        const REGISTERS = 1 << 2;
    }
}

/// The count of each kind of publication the other end made to this end, as
/// one look at the pair saw them ([`QueuePair::signals`]): what an end compares
/// with when it declares that it sleeps ([`QueuePair::arm`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Signals([u64; KINDS]);

/// The descriptors one queue pair travels as, between the buffer pool and an
/// engine and from a Storage Server to its successor: its shared memory, the
/// doorbell the buffer pool's end waits on, and the doorbell the engine's end
/// waits on. Each end holds all three: it waits on its own doorbell and rings
/// the other's.
#[derive(Debug)]
pub struct QueueDescriptors {
    pub memory: OwnedFd,
    pub pool: OwnedFd,
    pub engine: OwnedFd,
}

impl QueueDescriptors {
    /// The descriptors of one queue pair, as a message carries them.
    pub const COUNT: usize = 3;

    /// The descriptors in the order a message carries them: the memory, the
    /// buffer pool's doorbell, the engine's doorbell.
    pub fn into_array(self) -> [OwnedFd; Self::COUNT] {
        [self.memory, self.pool, self.engine]
    }

    /// The next queue pair's descriptors of `received`, in the order
    /// [`QueueDescriptors::into_array`] gives them; `None` when fewer remain.
    pub fn take(received: &mut impl Iterator<Item = OwnedFd>) -> Option<Self> {
        Some(Self {
            memory: received.next()?,
            pool: received.next()?,
            engine: received.next()?,
        })
    }
}

/// One end of a queue pair.
pub struct QueuePair {
    /// The mapping of the whole memfd.
    base: NonNull<u8>,
    bytes: usize,
    memory: OwnedFd,
    /// The doorbells, indexed by end ([`POOL`], [`ENGINE`]): this end waits on
    /// its own and rings the other's.
    doorbells: [OwnedFd; 2],

    /// A ring's bytes, a power of two.
    ring_bytes: usize,
    registers: usize,
    /// The separation of words written by different sides, as the header states it.
    separation: usize,
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
    /// of at least 64) with `registers` registers, all zero, and its two
    /// doorbells, and returns the buffer pool's end.
    pub fn create(ring_bytes: usize, registers: usize) -> Result<Self> {
        if !ring_bytes.is_power_of_two() || ring_bytes < 64 {
            return Err(Error::Protocol(format!(
                "a ring of {ring_bytes} bytes is not a power of two of at least 64"
            )));
        }
        let rings = rings_at(SEPARATION, registers)
            .ok_or_else(|| Error::Protocol(format!("{registers} registers overflow a layout")))?;
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
        let doorbells = [doorbell()?, doorbell()?];

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
            base.as_ptr()
                .add(HEADER_SEPARATION)
                .cast::<u64>()
                .write(SEPARATION as u64);
            base.as_ptr().cast::<u64>().write(MAGIC);
        }
        Ok(Self {
            base,
            bytes,
            memory,
            doorbells,
            ring_bytes,
            registers,
            separation: SEPARATION,
            rings,
            engine: false,
            produced: 0,
            consumed: 0,
        })
    }

    /// Maps an existing queue pair as the engine's end (`engine`) or as the
    /// buffer pool's, which a Storage Server restarted in place does with the
    /// queue pairs it receives. The end continues from the indices the
    /// previous holder of that end published, and waits on no doorbell until
    /// it declares so.
    pub fn open(descriptors: QueueDescriptors, engine: bool) -> Result<Self> {
        let QueueDescriptors {
            memory,
            pool,
            engine: engine_doorbell,
        } = descriptors;
        // SAFETY: `stat` is plain data the call fills for an owned descriptor.
        let mut stat: libc::stat = unsafe { std::mem::zeroed() };
        // SAFETY: as above.
        if unsafe { libc::fstat(memory.as_raw_fd(), &mut stat) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let bytes = stat.st_size as usize;
        if bytes < HEADER_BYTES {
            return Err(Error::Protocol(format!(
                "a queue pair of {bytes} bytes has no header"
            )));
        }

        let base = map(&memory, bytes)?;
        // SAFETY: the mapping holds at least the header, at aligned offsets.
        let (magic, ring_bytes, registers, separation) = unsafe {
            (
                base.as_ptr().cast::<u64>().read(),
                base.as_ptr().add(HEADER_RING_BYTES).cast::<u64>().read() as usize,
                base.as_ptr().add(HEADER_REGISTERS).cast::<u64>().read() as usize,
                base.as_ptr().add(HEADER_SEPARATION).cast::<u64>().read() as usize,
            )
        };
        // The indices start a separation in, after the header's line, and the doorbell words
        // hold a count per kind of publication within their separation.
        let separated = separation.is_power_of_two()
            && separation >= HEADER_BYTES
            && separation >= Interest::all().bits().count_ones() as usize * 8;
        let rings = separated.then(|| rings_at(separation, registers)).flatten();
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
            doorbells: [pool, engine_doorbell],
            ring_bytes,
            registers,
            separation,
            rings: rings.unwrap_or_default(),
            engine,
            produced: 0,
            consumed: 0,
        };
        pair.produced = pair.index(pair.writes(), false).load(Ordering::Acquire);
        pair.consumed = pair.index(pair.reads(), true).load(Ordering::Acquire);
        Ok(pair)
    }

    /// The queue pair's descriptors, in the order a message carries them
    /// ([`QueueDescriptors::into_array`]).
    pub fn descriptors(&self) -> [BorrowedFd<'_>; QueueDescriptors::COUNT] {
        [
            self.memory.as_fd(),
            self.doorbells[POOL].as_fd(),
            self.doorbells[ENGINE].as_fd(),
        ]
    }

    /// Copies of the queue pair's descriptors, which travel to the other end,
    /// to a successor, or to another mapping of the same end in this process.
    pub fn duplicate(&self) -> Result<QueueDescriptors> {
        Ok(QueueDescriptors {
            memory: self.memory.try_clone()?,
            pool: self.doorbells[POOL].try_clone()?,
            engine: self.doorbells[ENGINE].try_clone()?,
        })
    }

    /// The largest record a ring takes.
    pub fn max_record(&self) -> usize {
        self.ring_bytes - LENGTH
    }

    /// Register `index`: one word both sides read, written by the side the
    /// protocol names for it. A writer that changed registers the other end
    /// may wait for says so ([`QueuePair::notify_registers`]).
    pub fn register(&self, index: usize) -> &AtomicU64 {
        assert!(
            index < self.registers,
            "register {index} of {}",
            self.registers
        );
        // The layout was checked to hold the registers when the end was made.
        let at = REGISTERS * self.separation + index * 8;
        // SAFETY: the register lies inside the mapping at an aligned offset, and
        // the mapping lives as long as `self`.
        unsafe { &*self.base.as_ptr().add(at).cast::<AtomicU64>() }
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

    /// States that this end wrote registers the other end reads: rings the
    /// other end's doorbell when it waits for registers. One call follows a
    /// batch of register writes.
    pub fn notify_registers(&self) {
        self.publish(Interest::REGISTERS);
    }

    /// The count of each kind of publication the other end made to this end
    /// so far. An end takes it before it looks at the pair, and states it when
    /// it declares that it sleeps ([`QueuePair::arm`]).
    pub fn signals(&self) -> Signals {
        let own = self.own();
        Signals(std::array::from_fn(|kind| {
            self.signal(own, kind).load(Ordering::Acquire)
        }))
    }

    /// Declares that this end is about to sleep until the other end publishes
    /// what `interest` names, and checks that the other end published none of
    /// it since `seen` ([`QueuePair::signals`], taken before this end's last
    /// look at the pair). True: the declaration stands, and the other end's
    /// next such publication rings [`QueuePair::doorbell`], so the caller may
    /// block on it. False: something arrived, the declaration is withdrawn,
    /// and the caller looks at the pair again instead of sleeping.
    ///
    /// The caller withdraws a standing declaration once it wakes, whatever
    /// woke it ([`QueuePair::disarm`]), so that the other end makes no system
    /// call while this end is busy. An empty `interest` declares nothing and
    /// is true: nothing this end waits for comes through the pair.
    pub fn arm(&self, interest: Interest, seen: Signals) -> bool {
        if interest.is_empty() {
            return true;
        }
        self.interest(self.own())
            .store(interest.bits(), Ordering::Relaxed);
        // Pairs with the fence of `publish`: either the counts read next see
        // the other end's publication, or the other end sees the declaration.
        fence(Ordering::SeqCst);
        let arrived = self.arrived(interest, seen);
        if arrived {
            self.disarm();
        }
        !arrived
    }

    /// Withdraws this end's declaration ([`QueuePair::arm`]): the other end
    /// rings its doorbell no more until it declares again. A ring already
    /// sent stays in the doorbell until [`QueuePair::drain`] takes it.
    pub fn disarm(&self) {
        self.interest(self.own()).store(0, Ordering::Relaxed);
    }

    /// The descriptor of this end's doorbell: readable once the other end
    /// rang it, until [`QueuePair::drain`].
    pub fn doorbell(&self) -> BorrowedFd<'_> {
        self.doorbells[self.own()].as_fd()
    }

    /// Takes every ring of this end's doorbell, so that it is no longer
    /// readable.
    pub fn drain(&self) {
        let mut count = [0_u8; 8];
        // SAFETY: an eventfd read of its eight-byte counter into a live
        // buffer; the descriptor is non-blocking, so an unrung doorbell
        // answers EAGAIN, which leaves nothing to take.
        unsafe {
            libc::read(
                self.doorbell().as_raw_fd(),
                count.as_mut_ptr().cast(),
                count.len(),
            )
        };
    }

    /// Blocks until the other end publishes what `interest` names after
    /// `seen`, or `timeout` passes (`None` waits without bound): the wait of
    /// an end that waits on nothing but the pair. True when the other end
    /// published such a thing, before or during the wait; false when the
    /// time passed first. The declaration is withdrawn and the doorbell
    /// drained before it returns.
    ///
    /// # Errors
    ///
    /// What `poll` fails with.
    pub fn wait(
        &self,
        interest: Interest,
        seen: Signals,
        timeout: Option<Duration>,
    ) -> Result<bool> {
        let deadline = timeout.and_then(|timeout| Instant::now().checked_add(timeout));
        loop {
            if !self.arm(interest, seen) {
                return Ok(true);
            }
            let remaining =
                deadline.map(|deadline| deadline.saturating_duration_since(Instant::now()));
            if remaining.is_some_and(|remaining| remaining.is_zero()) {
                self.disarm();
                return Ok(false);
            }

            let mut doorbell = libc::pollfd {
                fd: self.doorbell().as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            let milliseconds = remaining.map_or(-1, |remaining| {
                // Rounded up, so the poll does not return before the deadline.
                let rounded = remaining.as_nanos().div_ceil(1_000_000);
                libc::c_int::try_from(rounded).unwrap_or(libc::c_int::MAX)
            });
            // SAFETY: poll borrows one live `pollfd` for the call.
            let polled = unsafe { libc::poll(&mut doorbell, 1, milliseconds) };
            let failed = (polled < 0).then(std::io::Error::last_os_error);
            self.disarm();
            self.drain();
            // A ring, a timeout or a signal: the next declaration looks at the
            // counts again, which tells a publication from a stale ring.
            if let Some(error) = failed
                && error.kind() != std::io::ErrorKind::Interrupted
            {
                return Err(error.into());
            }
        }
    }

    /// Whether the other end published something `interest` names since
    /// `seen`.
    fn arrived(&self, interest: Interest, seen: Signals) -> bool {
        let own = self.own();
        (0..KINDS)
            .filter(|kind| interest.bits() & (1 << kind) != 0)
            .any(|kind| self.signal(own, kind).load(Ordering::Acquire) != seen.0[kind])
    }

    /// The ring this end writes: 0 is the submission queue, 1 the completion queue.
    fn writes(&self) -> usize {
        if self.engine { 0 } else { 1 }
    }

    /// The ring this end reads.
    fn reads(&self) -> usize {
        1 - self.writes()
    }

    /// This end, as the doorbells are indexed.
    fn own(&self) -> usize {
        if self.engine { ENGINE } else { POOL }
    }

    /// The other end.
    fn other(&self) -> usize {
        1 - self.own()
    }

    /// The producer index of `ring`, or its consumer index.
    fn index(&self, ring: usize, consumer: bool) -> &AtomicU64 {
        let offset = (INDICES + 2 * ring + usize::from(consumer)) * self.separation;
        // SAFETY: the index lies inside the mapping at an aligned offset, and the
        // mapping lives as long as `self`.
        unsafe { &*self.base.as_ptr().add(offset).cast::<AtomicU64>() }
    }

    /// The kinds `end` declared it waits for.
    fn interest(&self, end: usize) -> &AtomicU64 {
        let offset = (DOORBELLS + 2 * end) * self.separation;
        // SAFETY: the word lies inside the mapping at an aligned offset, and the
        // mapping lives as long as `self`.
        unsafe { &*self.base.as_ptr().add(offset).cast::<AtomicU64>() }
    }

    /// The count of publications of `kind` (a bit index of [`Interest`]) made
    /// to `end`.
    fn signal(&self, end: usize, kind: usize) -> &AtomicU64 {
        let offset = (DOORBELLS + 2 * end + 1) * self.separation + kind * 8;
        // SAFETY: as `interest`.
        unsafe { &*self.base.as_ptr().add(offset).cast::<AtomicU64>() }
    }

    /// Counts a publication of `kind` to the other end, and rings the other
    /// end's doorbell when it declared that it waits for it. The declaration
    /// is cleared by the ring, so one ring reaches an end per sleep however
    /// many publications follow.
    fn publish(&self, kind: Interest) {
        let other = self.other();
        let index = kind.bits().trailing_zeros() as usize;
        self.signal(other, index).fetch_add(1, Ordering::Release);
        // Pairs with the fence of `arm`.
        fence(Ordering::SeqCst);
        let interest = self.interest(other);
        if interest.load(Ordering::Relaxed) & kind.bits() != 0
            && interest.swap(0, Ordering::AcqRel) != 0
        {
            ring(&self.doorbells[other]);
        }
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
        self.publish(Interest::RECORDS);
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
        self.publish(Interest::ROOM);
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

/// A new doorbell: a non-blocking eventfd, unrung.
fn doorbell() -> Result<OwnedFd> {
    // SAFETY: the call creates a new descriptor this process owns.
    let descriptor = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    if descriptor < 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    // SAFETY: the descriptor was just created and is owned here alone.
    Ok(unsafe { OwnedFd::from_raw_fd(descriptor) })
}

/// Rings `doorbell`. A write that fails changes nothing the waiting end
/// relies on: the eventfd refuses an addition only when its counter is
/// already near its limit, which leaves it readable.
fn ring(doorbell: &OwnedFd) {
    let one = 1_u64.to_ne_bytes();
    // SAFETY: an eventfd write of an eight-byte addend from a live buffer.
    unsafe { libc::write(doorbell.as_raw_fd(), one.as_ptr().cast(), one.len()) };
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
        let engine = QueuePair::open(pool.duplicate().unwrap(), true).unwrap();
        (engine, pool)
    }

    /// A bound no correct wait comes near: a wait that reaches it lost a
    /// wake-up.
    const LOST: Duration = Duration::from_secs(30);

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
        let mut successor = QueuePair::open(pool.duplicate().unwrap(), false).unwrap();
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
        let [memory, ..] = pair.descriptors();
        // SAFETY: a write of eight bytes from a live buffer to an open descriptor.
        let written = unsafe { libc::pwrite(memory.as_raw_fd(), bytes.as_ptr().cast(), 8, offset) };
        assert_eq!(written, 8);
    }

    #[test]
    fn an_index_or_a_length_out_of_range_is_an_error_to_the_other_end() {
        let (mut engine, mut pool) = ends(64, 0);
        let mut taken = Vec::new();

        let separation = SEPARATION as i64;
        // The submission queue's producer index, further ahead than a ring holds.
        write_word(&engine, INDICES as i64 * separation, 72);
        assert!(pool.next_request(&mut taken).is_err());

        // A length word, the first of the submission queue's bytes, that states
        // more than was produced.
        write_word(&engine, INDICES as i64 * separation, 16);
        write_word(&engine, REGISTERS as i64 * separation, 40);
        assert!(pool.next_request(&mut taken).is_err());

        // The submission queue's consumer index, ahead of what the engine produced.
        write_word(&engine, (INDICES as i64 + 1) * separation, 1 << 40);
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
        assert!(QueuePair::open(pool.duplicate().unwrap(), true).is_err());

        // A separation that would lay the indices over the header, or is no power of two.
        for separation in [16, 96] {
            let pool = QueuePair::create(64, 0).unwrap();
            write_word(&pool, HEADER_SEPARATION as i64, separation);
            assert!(
                QueuePair::open(pool.duplicate().unwrap(), true).is_err(),
                "{separation}"
            );
        }
    }

    /// An end asleep on an empty ring wakes when the other end writes a
    /// record, and an end asleep for registers when the other end states it
    /// wrote some.
    #[test]
    fn a_sleeping_end_wakes_when_the_other_end_publishes() {
        let (mut engine, mut pool) = ends(64, 1);

        let sleeper = std::thread::spawn(move || {
            let mut request = Vec::new();
            // One look before anything was published, for both waits.
            let seen = pool.signals();
            if !pool.next_request(&mut request).unwrap() {
                assert!(pool.wait(Interest::RECORDS, seen, Some(LOST)).unwrap());
                assert!(pool.next_request(&mut request).unwrap());
            }
            assert_eq!(request, b"wake");

            assert!(pool.wait(Interest::REGISTERS, seen, Some(LOST)).unwrap());
            pool.register(0).load(Ordering::Acquire)
        });

        // The request is written whether the sleeper sleeps yet or not.
        assert!(engine.submit(b"wake").unwrap());
        // Registers are stated only once the request was taken, so that the
        // sleeper waits for them after it took it.
        while engine.room().unwrap() != 64 {
            std::thread::yield_now();
        }
        engine.register(0).store(9, Ordering::Release);
        engine.notify_registers();
        assert_eq!(sleeper.join().unwrap(), 9);
    }

    /// A publication of a kind the sleeping end does not wait for leaves it
    /// asleep, and one that came before its declaration keeps it awake.
    #[test]
    fn an_end_wakes_for_what_it_declared_and_for_nothing_else() {
        let (mut engine, pool) = ends(64, 1);

        let seen = pool.signals();
        engine.notify_registers();
        assert!(
            !pool
                .wait(Interest::RECORDS, seen, Some(Duration::from_millis(50)))
                .unwrap()
        );

        // The registers were stated before the declaration: the wait ends at
        // once, without the doorbell.
        assert!(pool.wait(Interest::REGISTERS, seen, Some(LOST)).unwrap());

        // A record written while nobody declares anything rings nothing.
        let seen = pool.signals();
        assert!(engine.submit(b"quiet").unwrap());
        let mut doorbell = libc::pollfd {
            fd: pool.doorbell().as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        };
        // SAFETY: poll borrows one live `pollfd` for the call.
        assert_eq!(unsafe { libc::poll(&mut doorbell, 1, 0) }, 0);
        assert!(pool.wait(Interest::RECORDS, seen, Some(LOST)).unwrap());
    }

    /// Two threads that each sleep on their doorbell whenever their ring is
    /// empty or full exchange every record once: no publication racing a
    /// sleep is lost, in either direction, for records and for room.
    #[test]
    fn ends_that_sleep_on_their_doorbells_lose_no_wake_up() {
        let (mut engine, mut pool) = ends(256, 0);
        const RECORDS: u64 = 100_000;

        // The buffer pool's side answers each request with its bytes,
        // sleeping when no request waits and when the completion queue is full.
        let server = std::thread::spawn(move || {
            let mut request = Vec::new();
            let mut served = 0;
            while served < RECORDS {
                let seen = pool.signals();
                if !pool.next_request(&mut request).unwrap() {
                    assert!(pool.wait(Interest::RECORDS, seen, Some(LOST)).unwrap());
                    continue;
                }
                loop {
                    let seen = pool.signals();
                    if pool.complete(&request).unwrap() {
                        break;
                    }
                    assert!(pool.wait(Interest::ROOM, seen, Some(LOST)).unwrap());
                }
                served += 1;
            }
        });

        let mut completion = Vec::new();
        let (mut sent, mut received) = (0u64, 0u64);
        while received < RECORDS {
            let seen = engine.signals();
            let mut progressed = false;
            if sent < RECORDS {
                // A record of varying length that carries its own number.
                let mut record = sent.to_le_bytes().to_vec();
                record.resize(8 + (sent % 23) as usize, sent as u8);
                if engine.submit(&record).unwrap() {
                    sent += 1;
                    progressed = true;
                }
            }
            while engine.next_completion(&mut completion).unwrap() {
                let number = u64::from_le_bytes(completion[..8].try_into().unwrap());
                assert_eq!(number, received);
                assert_eq!(completion.len(), 8 + (number % 23) as usize);
                assert!(completion[8..].iter().all(|byte| *byte == number as u8));
                received += 1;
                progressed = true;
            }
            if !progressed {
                // Nothing to read, and the submission queue full or done.
                let interest = if sent < RECORDS {
                    Interest::RECORDS | Interest::ROOM
                } else {
                    Interest::RECORDS
                };
                assert!(engine.wait(interest, seen, Some(LOST)).unwrap());
            }
        }
        server.join().unwrap();
    }
}
