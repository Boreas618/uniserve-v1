//! The daemon's integration tests: daemons in this process, and in one
//! process of their own, driven over the process interface as engines drive
//! them, with real child processes; and the leader's input processing and
//! output assembly over real tokenizers and templates.

#![cfg(test)]

mod chat_template;
mod model_preprocessing;
mod nodes;
mod qwen3_output;
mod supervision;
mod support;
