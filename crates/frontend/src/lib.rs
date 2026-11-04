//! The frontend: the deployment's application process (`uniserve frontend`).
//!
//! It serves the HTTP and OpenAI API, processes every request's input and
//! output, keeps the facts of the requests in flight and migrates them between
//! epochs, and exports the serving address's metrics. It links no numerical
//! code: models, the scheduler and the executors are the main process's
//! (`specs/worker.md` §2.1, §3).
//!
//! The daemon starts it, tells it which epoch serves, and hands it a successor
//! on a hot restart together with its connections and the streams in flight
//! (§7.5.1). Its modules are layered, each using only those below it:
//!
//! - `tokenizer`: the served checkpoint's tokenizer and incremental decoding;
//! - `model`: the served checkpoint's files, configuration and family
//!   profile;
//! - `options`: the deployment-file values the application reads.

pub mod model;
pub mod options;
pub mod tokenizer;

#[cfg(any(test, feature = "testing"))]
pub mod test_support;
