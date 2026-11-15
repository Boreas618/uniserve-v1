//! The CPU executor: the threads the CPU backend computes its parallel operations on.
//!
//! A process has one executor. Its owner starts it with [`CpuExecutor::install`] and an
//! [`ExecutorConfig`]; a process that installs none gets one started from
//! [`ExecutorConfig::for_process`] by the first CPU operation that runs in parallel. The executor
//! keeps two groups of threads on one CPU set, each serving the kind of work it schedules well:
//!
//! - a rayon pool, on which matrix products (`gemm`), convolutions and sorting split their work
//!   by work stealing; the calling thread waits for the pool;
//! - a barrier pool, on which element-wise maps split their elements into chunks that the calling
//!   thread and the pool's workers take from a shared cursor. Its idle workers spin before they
//!   park, so a map that follows another one starts without a wake-up through the kernel.
//!
//! An operation uses one group, so each caller keeps at most [`ExecutorConfig::threads`] threads
//! computing.

use std::any::Any;
use std::cell::{Cell, UnsafeCell};
use std::hint::spin_loop;
use std::ops::Range;
use std::os::unix::thread::JoinHandleExt;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::thread::{JoinHandle, Thread};

use crate::{Error, Result};

/// How the CPU executor runs: on which CPUs, with how many threads, and how it splits
/// element-wise work.
///
/// [`ExecutorConfig::for_process`] states the defaults; an owner changes the fields it decides:
///
/// ```no_run
/// use tensor::cpu_backend::{CpuExecutor, ExecutorConfig};
///
/// # fn main() -> tensor::Result<()> {
/// let config = ExecutorConfig {
///     threads: 8,
///     ..ExecutorConfig::for_process()?
/// };
/// CpuExecutor::install(config)?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutorConfig {
    /// The host CPUs, by number, every thread of the executor is pinned to; their order and any
    /// repetition do not matter. By default the CPUs the starting thread may run on
    /// (`sched_getaffinity`), which are the process's unless that thread narrowed its own.
    pub cpus: Vec<usize>,
    /// The threads that compute one operation, the calling thread included for an element-wise
    /// map. By default one per CPU of `cpus`.
    pub threads: usize,
    /// The busy-wait iterations (`std::hint::spin_loop`) an idle element-wise worker watches for
    /// its next map before it parks; 0 parks at once. Spinning starts a following map without a
    /// wake-up through the kernel, for the CPU time it occupies. By default 25,000.
    pub idle_spins: u32,
    /// The fewest elements an element-wise map splits across threads; a smaller one runs on the
    /// calling thread, where the split would cost more than it saves. By default 65,536.
    pub parallel_elements: usize,
    /// The elements of one chunk of a split element-wise map, the unit a thread takes from the
    /// shared cursor: large enough to amortize taking it, small enough that the threads finish
    /// together. By default 16,384.
    pub chunk_elements: usize,
}

impl ExecutorConfig {
    /// The default configuration: every CPU the calling thread may run on, one thread per CPU,
    /// and the default element-wise split.
    ///
    /// # Errors
    ///
    /// The operating system's refusal to state the calling thread's CPUs.
    pub fn for_process() -> Result<Self> {
        let cpus = CpuMask::of_calling_thread()?.cpus();

        Ok(Self {
            threads: cpus.len(),
            cpus,
            idle_spins: 25_000,
            parallel_elements: 64 * 1024,
            chunk_elements: 16 * 1024,
        })
    }

    /// The configuration with its CPUs in ascending order without repetition.
    ///
    /// # Errors
    ///
    /// [`Error::CpuExecutor`] for a configuration that names no CPU, no thread or empty chunks.
    fn checked(mut self) -> Result<Self> {
        self.cpus.sort_unstable();
        self.cpus.dedup();

        if self.cpus.is_empty() {
            return Err(Error::CpuExecutor("the configuration names no CPU".to_owned()).bt());
        }
        if self.threads == 0 {
            return Err(Error::CpuExecutor("the configuration has no thread".to_owned()).bt());
        }
        if self.chunk_elements == 0 {
            return Err(Error::CpuExecutor(
                "the configuration's chunks hold no element".to_owned(),
            )
            .bt());
        }
        Ok(self)
    }
}

/// The process's executor, set once: by [`CpuExecutor::install`] or by the first operation that
/// needs it.
static EXECUTOR: OnceLock<CpuExecutor> = OnceLock::new();

/// Serializes the starts of [`EXECUTOR`], so a racing install and first use start one executor.
static STARTING: Mutex<()> = Mutex::new(());

/// The threads the CPU backend computes its parallel operations on; see the module
/// documentation.
pub struct CpuExecutor {
    config: ExecutorConfig,
    rayon: rayon::ThreadPool,
    barrier: BarrierPool,
}

impl CpuExecutor {
    /// Starts the process's executor from `config`.
    ///
    /// An owner installs the executor before the process's first parallel CPU operation, which
    /// would otherwise start one from [`ExecutorConfig::for_process`].
    ///
    /// # Errors
    ///
    /// [`Error::CpuExecutor`] when the process's executor already runs, or for a configuration
    /// that names no CPU, no thread or empty chunks; the operating system's refusal to start a
    /// thread or to pin it to the CPUs.
    pub fn install(config: ExecutorConfig) -> Result<()> {
        // A configuration is refused for itself, whether or not an executor runs.
        let config = config.checked()?;

        // A poisoned lock guards no data: a panic while starting left `EXECUTOR` unset.
        let _starting = STARTING.lock().unwrap_or_else(PoisonError::into_inner);
        if EXECUTOR.get().is_some() {
            return Err(Error::CpuExecutor(
                "the process's executor already runs: it is installed before the first parallel \
                 CPU operation"
                    .to_owned(),
            )
            .bt());
        }

        let executor = Self::start(config)?;
        EXECUTOR.get_or_init(|| executor);
        Ok(())
    }

    /// The process's executor: the one installed, or one started from
    /// [`ExecutorConfig::for_process`] on first use.
    ///
    /// # Errors
    ///
    /// When no executor runs, the errors of starting the default one, as for
    /// [`CpuExecutor::install`].
    pub fn current() -> Result<&'static Self> {
        if let Some(executor) = EXECUTOR.get() {
            return Ok(executor);
        }

        let _starting = STARTING.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(executor) = EXECUTOR.get() {
            return Ok(executor);
        }
        let executor = Self::start(ExecutorConfig::for_process()?)?;
        Ok(EXECUTOR.get_or_init(|| executor))
    }

    /// The configuration the executor runs with, its CPUs in ascending order.
    pub fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    /// An executor of its own for `config`. Dropping it stops its threads.
    ///
    /// # Errors
    ///
    /// As [`CpuExecutor::install`], but for an executor that already runs.
    pub(crate) fn start(config: ExecutorConfig) -> Result<Self> {
        let config = config.checked()?;
        let mask = CpuMask::of(&config.cpus);

        // The pool's threads are spawned here rather than by rayon, so each one is pinned through
        // its handle and a refusal fails the start.
        let rayon = rayon::ThreadPoolBuilder::new()
            .num_threads(config.threads)
            .thread_name(|index| format!("tensor-cpu-{index}"))
            .spawn_handler(|thread| {
                let mut builder = std::thread::Builder::new();
                if let Some(name) = thread.name() {
                    builder = builder.name(name.to_owned());
                }
                if let Some(size) = thread.stack_size() {
                    builder = builder.stack_size(size);
                }
                let handle = builder.spawn(|| thread.run())?;
                mask.pin(&handle)
            })
            .build()
            .map_err(|error| {
                Error::CpuExecutor(format!("starting the rayon threads: {error}")).bt()
            })?;

        // The calling thread computes a share of each element-wise map beside the workers.
        let barrier = BarrierPool::start(config.threads - 1, config.idle_spins, &mask)?;

        Ok(Self {
            config,
            rayon,
            barrier,
        })
    }

    /// Runs `f` on the rayon pool, where the rayon parallel iterators and `gemm`'s rayon
    /// parallelism inside it split their work, and returns its result. The calling thread waits;
    /// a call from a thread of the pool runs `f` on that thread.
    pub(crate) fn run<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        self.rayon.install(f)
    }

    /// Whether an element-wise map of `elements` elements is split across threads.
    pub(crate) fn splits(&self, elements: usize) -> bool {
        elements >= self.config.parallel_elements
    }

    /// Calls `f` once with every chunk of `0..elements`, each [`ExecutorConfig::chunk_elements`]
    /// long but the last, on the calling thread and the barrier pool's workers, and returns when
    /// every chunk is done. A panic of `f` is raised again on the calling thread.
    ///
    /// Called by `f` itself, the inner call runs its chunks on its own thread: the workers are
    /// busy with the outer one.
    pub(crate) fn for_each_chunk(&self, elements: usize, f: impl Fn(Range<usize>) + Sync) {
        let chunk = self.config.chunk_elements;
        let chunks = elements.div_ceil(chunk);
        let range = |index: usize| index * chunk..elements.min((index + 1) * chunk);

        if IN_BARRIER_POOL.with(Cell::get) {
            (0..chunks).for_each(|index| f(range(index)));
            return;
        }

        // Each thread takes the next chunk when it finishes one, so a fast thread absorbs the
        // work of a slow one instead of waiting for it at the barrier.
        let cursor = AtomicUsize::new(0);
        self.barrier.execute(|_| {
            loop {
                let index = cursor.fetch_add(1, Ordering::Relaxed);
                if index >= chunks {
                    break;
                }
                f(range(index));
            }
        });
    }
}

/// A CPU affinity mask in the layout of the kernel's `cpu_set_t`: CPU `n` is bit `n % W` of word
/// `n / W`, for words of `W` bits.
struct CpuMask(Vec<libc::c_ulong>);

impl CpuMask {
    const WORD_BITS: usize = libc::c_ulong::BITS as usize;

    /// The words of a `cpu_set_t`, which covers every CPU of a host with at most 1024.
    const SET_WORDS: usize = size_of::<libc::cpu_set_t>() / size_of::<libc::c_ulong>();

    /// The CPUs the calling thread may run on.
    ///
    /// # Errors
    ///
    /// The operating system's refusal, other than to a mask narrower than its own.
    fn of_calling_thread() -> Result<Self> {
        let mut words = Self::SET_WORDS;
        loop {
            let mut mask = Self(vec![0; words]);

            // SAFETY: the mask's buffer holds `mask.bytes()` bytes, at most which the call writes.
            let status =
                unsafe { libc::sched_getaffinity(0, mask.bytes(), mask.0.as_mut_ptr().cast()) };
            if status == 0 {
                return Ok(mask);
            }

            // The kernel refuses a mask narrower than its count of possible CPUs; a host past
            // `cpu_set_t`'s 1024 is asked again with a wider one.
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::EINVAL) {
                return Err(error.into());
            }
            words = words.checked_mul(2).ok_or_else(|| {
                Error::CpuExecutor("the kernel's CPU mask exceeds the address space".to_owned())
            })?;
        }
    }

    /// The mask of `cpus`, at least as wide as a `cpu_set_t`.
    fn of(cpus: &[usize]) -> Self {
        let words = cpus
            .iter()
            .map(|cpu| cpu / Self::WORD_BITS + 1)
            .max()
            .unwrap_or(0)
            .max(Self::SET_WORDS);

        let mut mask = vec![0; words];
        for cpu in cpus {
            mask[cpu / Self::WORD_BITS] |= 1 << (cpu % Self::WORD_BITS);
        }
        Self(mask)
    }

    /// The CPUs of the mask, in ascending order.
    fn cpus(&self) -> Vec<usize> {
        let mut cpus = Vec::new();
        for (index, &word) in self.0.iter().enumerate() {
            for bit in 0..Self::WORD_BITS {
                if word & (1 << bit) != 0 {
                    cpus.push(index * Self::WORD_BITS + bit);
                }
            }
        }
        cpus
    }

    /// The length of the mask in bytes.
    fn bytes(&self) -> usize {
        self.0.len() * size_of::<libc::c_ulong>()
    }

    /// Restricts the thread of `handle` to the mask's CPUs.
    ///
    /// # Errors
    ///
    /// The operating system's refusal, as for a mask of no CPU the thread may use.
    fn pin<T>(&self, handle: &JoinHandle<T>) -> std::io::Result<()> {
        // SAFETY: the thread of `handle` is not joined while the handle lives, and the call reads
        // `self.bytes()` bytes of the mask.
        let status = unsafe {
            libc::pthread_setaffinity_np(
                handle.as_pthread_t(),
                self.bytes(),
                self.0.as_ptr().cast(),
            )
        };
        match status {
            0 => Ok(()),
            error => Err(std::io::Error::from_raw_os_error(error)),
        }
    }
}

/// Per-worker control on its own 64-byte cache line to prevent false sharing.
#[repr(C, align(64))]
struct Slot {
    /// `Release`-stored by parent to signal work. `Acquire`-loaded by worker.
    go: AtomicUsize,
    /// `Release`-stored by worker (and its subtree) when done. `Acquire`-loaded by parent.
    done: AtomicUsize,
}

/// Closure pointer written by main before the `Release` on each worker's `go`.
struct WorkDesc {
    trampoline: unsafe fn(*const (), usize),
    data: *const (),
}

struct BarrierPoolInner {
    slots: Box<[Slot]>,
    /// Thread handles for unpark.
    handles: OnceLock<Box<[Thread]>>,
    work: UnsafeCell<WorkDesc>,
    stop: AtomicBool,
    panic_payload: Mutex<Option<Box<dyn Any + Send>>>,
}

// SAFETY: `work`, the one field that is not `Sync`, is written by `execute` under `call_lock`
// before it signals the workers' `go` (`Release`); a worker reads it only after its `Acquire` load
// of `go`, which that write happens before; and `execute` does not write it again before every
// worker has signalled `done`. Every other field is `Sync`.
unsafe impl Sync for BarrierPoolInner {}
// SAFETY: the raw pointer in `work` is dereferenced only under the protocol stated for `Sync`;
// every other field is `Send`.
unsafe impl Send for BarrierPoolInner {}

/// The element-wise workers: threads that run one closure together with the calling thread and
/// wait at a barrier until all of them returned.
///
/// The calling thread signals every worker and runs its own share; on completion it waits for
/// every worker's `done`.
struct BarrierPool {
    inner: Box<BarrierPoolInner>,
    threads: Vec<JoinHandle<()>>,
    /// Serializes concurrent callers. Stores the current generation.
    call_lock: Mutex<usize>,
}

thread_local! {
    static IN_BARRIER_POOL: Cell<bool> = const { Cell::new(false) };
}

unsafe fn call_trampoline<F: Fn(usize)>(data: *const (), tid: usize) {
    // SAFETY: `data` points to the `F` that `execute` stored with this trampoline, which it keeps
    // alive until every worker has signalled `done`.
    unsafe {
        (*(data as *const F))(tid);
    }
}

impl BarrierPool {
    /// A pool of `n_workers` threads pinned to `mask`, each spinning `idle_spins` iterations
    /// before it parks.
    ///
    /// # Errors
    ///
    /// The operating system's refusal to start a thread or to pin it; the threads already started
    /// are stopped.
    fn start(n_workers: usize, idle_spins: u32, mask: &CpuMask) -> Result<Self> {
        let inner = Box::new(BarrierPoolInner {
            slots: (0..n_workers)
                .map(|_| Slot {
                    go: AtomicUsize::new(0),
                    done: AtomicUsize::new(0),
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            handles: OnceLock::new(),
            work: UnsafeCell::new(WorkDesc {
                trampoline: |_, _| {},
                data: std::ptr::null(),
            }),
            stop: AtomicBool::new(false),
            panic_payload: Mutex::new(None),
        });
        let inner_ptr = &*inner as *const BarrierPoolInner as usize;

        // A pool that fails to start drops as it is: `Drop` stops the threads started so far.
        let mut pool = BarrierPool {
            inner,
            threads: Vec::with_capacity(n_workers),
            call_lock: Mutex::new(0),
        };
        for tid in 0..n_workers {
            let thread = std::thread::Builder::new()
                .name(format!("tensor-bp-{tid}"))
                .spawn(move || {
                    // SAFETY: `inner_ptr` is the address of the pool's boxed
                    // `BarrierPoolInner`, which stays in place until `Drop for BarrierPool` has
                    // joined this thread.
                    let inner = unsafe { &*(inner_ptr as *const BarrierPoolInner) };
                    worker(inner, tid, idle_spins);
                })?;

            // The thread joins the pool before a refusal returns, so `Drop` stops it as well.
            let pinned = mask.pin(&thread);
            pool.threads.push(thread);
            pinned?;
        }

        pool.inner
            .handles
            .set(
                pool.threads
                    .iter()
                    .map(|jh| jh.thread().clone())
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            )
            .ok();
        Ok(pool)
    }

    /// Runs `f` on every worker, with the worker's index, and on the calling thread, with the
    /// number of workers, and returns when all of them returned. A panic of `f` on a worker is
    /// raised again on the calling thread.
    fn execute<F: Fn(usize) + Sync>(&self, f: F) {
        let n = self.threads.len();
        if n == 0 {
            f(0);
            return;
        }

        // A poisoned lock still holds a valid generation: `BarrierGuard` waited for every worker
        // before the panic that poisoned it left `execute`.
        let mut guard = self
            .call_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let new_generation = guard.wrapping_add(1);
        *guard = new_generation;

        // SAFETY: `call_lock` serializes the callers of `execute`, and the previous call returned
        // only after every worker signalled `done` (`BarrierGuard`), so no worker reads `work`
        // while it is written.
        unsafe {
            let work = &mut *self.inner.work.get();
            work.trampoline = call_trampoline::<F>;
            work.data = &f as *const F as *const ();
        }

        #[expect(
            clippy::unwrap_used,
            reason = "`start` sets the handles before it returns"
        )]
        let handles = self.inner.handles.get().unwrap();
        for (slot, handle) in self.inner.slots.iter().zip(handles.iter()) {
            slot.go.store(new_generation, Ordering::Release);
            handle.unpark();
        }

        // Guard waits for workers and propagates panics on drop.
        let _guard = BarrierGuard {
            inner: &self.inner,
            generation: new_generation,
        };

        // Main thread participates as worker n (workers are 0..n-1).
        f(n);
        // _guard drops here, waiting for the workers.
    }
}

/// The loop of worker `tid`: wait for a generation, run the work, signal `done`.
fn worker(inner: &BarrierPoolInner, tid: usize, idle_spins: u32) {
    IN_BARRIER_POOL.with(|flag| flag.set(true));
    let slot = &inner.slots[tid];
    let mut generation = 0usize;

    loop {
        // Spin briefly then park, waiting for next work item.
        let mut spins = 0u32;
        loop {
            let g = slot.go.load(Ordering::Acquire);
            if g != generation {
                generation = g;
                break;
            }
            if inner.stop.load(Ordering::Relaxed) {
                return;
            }
            spins += 1;
            if spins < idle_spins {
                spin_loop();
            } else {
                // Park deposits a token; unpark() before park() returns immediately — no race.
                std::thread::park();
                spins = 0;
            }
        }
        if inner.stop.load(Ordering::Relaxed) {
            break;
        }

        // Execute own work, catching panics so done is always signalled.
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: `execute` wrote `work` before signalling `go`, which this worker's
            // `Acquire` load observed, and does not write it again before every worker has
            // signalled `done`.
            let work = unsafe { &*inner.work.get() };
            // SAFETY: `trampoline` is `call_trampoline::<F>` for the closure `data` points to,
            // which `execute` keeps alive until every worker has signalled `done`.
            unsafe { (work.trampoline)(work.data, tid) };
        })) {
            let mut stored = inner
                .panic_payload
                .lock()
                .unwrap_or_else(PoisonError::into_inner);
            if stored.is_none() {
                *stored = Some(payload);
            }
        }

        // Signal done unconditionally so the calling thread never deadlocks.
        slot.done.store(generation, Ordering::Release);
    }
}

/// RAII guard that blocks until all workers have finished.
struct BarrierGuard<'a> {
    inner: &'a BarrierPoolInner,
    generation: usize,
}

impl Drop for BarrierGuard<'_> {
    fn drop(&mut self) {
        for slot in self.inner.slots.iter() {
            while slot.done.load(Ordering::Acquire) != self.generation {
                spin_loop();
            }
        }
        // Always clear any stored panic. Propagate unless main is not already unwinding.
        let payload = self
            .inner
            .panic_payload
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        if !std::thread::panicking()
            && let Some(p) = payload
        {
            std::panic::resume_unwind(p);
        }
    }
}

impl Drop for BarrierPool {
    fn drop(&mut self) {
        if self.threads.is_empty() {
            return;
        }
        self.inner.stop.store(true, Ordering::Relaxed);
        let new_generation = self
            .call_lock
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .wrapping_add(1);

        // Signal every slot and unpark every thread started, so parked workers wake immediately;
        // a pool that failed to start has fewer threads than slots.
        for slot in self.inner.slots.iter() {
            slot.go.store(new_generation, Ordering::Release);
        }
        for thread in &self.threads {
            thread.thread().unpark();
        }
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}
