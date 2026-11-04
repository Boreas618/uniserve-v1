//! The served model, which the application reads (`frontend::model`).
//!
//! The daemon keeps the module so the management side names the same paths it
//! always did; the model's facts, its parameters and its asset resolution are
//! the frontend's (delivery item 2).

pub use frontend::model::*;
