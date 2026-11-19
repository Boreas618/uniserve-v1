//! The processing program of the release: the request side of the tree's model families
//! (`processing::FAMILIES`) served through the processing interface (`scheduling::processing`).
//!
//! Built for `wasm32-unknown-unknown` (`package.toml`), the crate's module is the program an
//! epoch's module set fixes (`worker_config.processing`) and the leader daemon loads, so an
//! epoch's request side is built from the same tree as its engine and nothing in the daemon names
//! a model. What the module runs (the binding of a family to a checkpoint, the chat templates and
//! the output parsers) is the `processing` library's (`processing::program`); this crate is the
//! module's exports of it over the release's families (`processing::program::Release`).

// The module's exports: the release's families through the processing interface, on wasm32.
scheduling::export_processing!(processing::program::Program<processing::program::Release>);
