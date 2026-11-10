//! The application's view of the served checkpoint, which the daemon's HTTP
//! and OpenAI API reads. It runs no numerical code: models, the scheduler
//! and the workers are the main process's, and it reads only a family's
//! request-processing side (`processing::Family::serving`). Its modules are layered, each
//! using only those below it:
//!
//! - `tokenizer`: the served checkpoint's tokenizer and incremental decoding;
//! - `model`: the served checkpoint's files, configuration and family
//!   profile.

pub mod model;
pub mod tokenizer;

#[cfg(any(test, feature = "testing"))]
pub mod test_support;
