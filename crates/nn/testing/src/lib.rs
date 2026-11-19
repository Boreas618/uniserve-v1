//! Test support of the neural-network library (`nn`) and the model families (`models`): what
//! their integration tests share, stated once.
//!
//! - [`checkpoint`]: checkpoints a test writes, in the names the transformers and diffusers
//!   formats declare (`model.safetensors`, or shards listed by `model.safetensors.index.json`),
//!   committed fixtures embedded in a test binary ([`fixture!`]) and written out for the
//!   readers that open paths, and the adapter container the tests declare ([`Stated`]).
//! - [`release`]: released checkpoints by Hugging Face repository id and pinned commit,
//!   resolved through the standard Hugging Face cache.
//! - [`forward`]: meta tensors, forwards recorded under a meta context, and the checks every
//!   profile run makes of what it recorded.
//! - [`text`]: what a causal language model's forward is lent: its batch, its state, and the
//!   model itself, loaded as execution loads it.
//! - [`values`]: CPU tensors of stated values, their elements, and comparisons of them.
//! - [`reference`]: direct ports of the reference formulas of a hyper-connected residual and of a
//!   hashed n-gram memory, in FP64, and the comparison of an output with them under an
//!   operator's bound.
//!
//! Every helper fails the calling test with a message naming what failed: a fixture that cannot
//! be written or a release that cannot be resolved is the test's failure, not a value to handle.

pub mod checkpoint;
pub mod forward;
pub mod reference;
pub mod release;
pub mod text;
pub mod values;

pub use checkpoint::{Fixture, Stated, Stored, checkpoint, stored, write_checkpoint, write_file};
pub use forward::{
    Launch, Recording, assert_generic, assert_operators, meta, operators, record, report,
};
pub use release::Release;
pub use text::PAGE_TOKENS;
pub use values::{assert_close, decode, encode, sequence, tensor, values};
