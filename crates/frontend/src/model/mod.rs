//! The served checkpoint's files and JSON metadata.
//!
//! [`assets`] locates a checkpoint in a local directory, the local Hugging
//! Face Hub cache, or the Hub itself; reads the root `config.json` or diffusers
//! pipeline index that selects the model family; reads the tokenizer's and
//! generation's metadata; and reads tensor shapes from a safetensors header
//! without loading weights. The resolved model, its family profile and its
//! output parsers are the frontend's as well; they move here from the daemon
//! with the rest of the application layer.

pub mod assets;
