//! A node's main process for one epoch.
//!
//! The daemon starts one main process per node and epoch and hands it one
//! launch ([`proto::config::EngineArgs`], read from the sealed file the
//! process inherits, `proto::LAUNCH_OPTION`): the epoch's engine
//! configuration, what the process takes over from the one it succeeds, the
//! request address, the Storage Server namespacing, the Build Server's
//! socket, the devices, the rank and the durations. [`run`] takes it, starts
//! the control thread, one scheduler per DP rank and one worker per device,
//! and serves until its daemon commands it down or the epoch fails. [`serve`]
//! is the same control thread over two given connections and a [`Start`], the
//! function that builds the started epoch ([`Engine`]): the binary's loads
//! ranks, a test's builds fakes.
//!
//! ## The modules, by the thread that runs them
//!
//! | Module | Holds | Thread |
//! | --- | --- | --- |
//! | `process` | The control thread: the daemon's request and process interfaces, the lifecycle, routing to schedulers, event streams, reports and metrics; the start (ranks, the session with the Storage Server, the [`Engine`]); jobs of generation updates, epochs and operations, and the weight binding (the changes of what the ranks hold on their devices) they act through | control, operations, start |
//! | [`scheduler`] | One scheduler per data-parallel rank: requests, admission, the memory decisions over the buffer pool, batches, updates, drains; the scheduling policy host | one per DP rank |
//! | [`batch`] | What a scheduler hands its executor and gets back: [`Batch`](batch::Batch), [`BatchOutput`](batch::BatchOutput), [`Capacity`](batch::Capacity), the products' order | — |
//! | [`executor`] | A scheduler's executor: the [`Executor`](executor::Executor) trait the scheduler calls and [`WorkerExecutor`](executor::WorkerExecutor), which numbers a group's jobs and hands them to the workers of the group on this node and to peers on other nodes; the peer link | one waiter per worker group |
//! | [`worker`] | One [`Worker`](worker::Worker) per device, the only thread that issues on its streams: its job queue, sequences, watchdog, the completion of its jobs | one per device |
//! | [`runner`] | The runners (causal language model, encoder, vision, denoiser, decoder, media host), graphs, buckets, the sampler | worker |
//! | [`memory`] | A rank's memory: block tables, product buffers, the KV cache's planes, recurrent state, weights, the division of the quota, the client of the Storage Server (head and rank) | worker, start |
//! | `profiling`, `parallel` | Profiling sessions and markers; the host groups ranks agree over | worker, start |
//! | [`modules`] | Resolution of packages at the Build Server and the load of their modules | start, jobs |

pub mod batch;
pub mod executor;
pub mod memory;
pub mod modules;
mod parallel;
mod process;
mod profiling;
pub mod runner;
pub mod scheduler;
pub mod worker;

pub use process::ranks::text_layout;
pub use process::{Engine, Failed, Start, connect, run, serve};
pub use scheduler::Error;
