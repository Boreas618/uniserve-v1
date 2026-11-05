//! The programs a deployment compiles: one `operator<TAB>key` row per launch the model can make.
//!
//! A kernel store is built for a target before any weight is read, so the deployment's key set must
//! be stated from the model's own geometry: each component says which launches its `forward` makes,
//! through the same construction that launches them (`nn`'s `descriptor_of` accessors and the model's
//! own config arithmetic). The alternative — a table in another crate that restates the configuration
//! — is a second implementation of the model, and it drifts silently: the store then lacks a program
//! the model asks for, and the refusal arrives at a launch instead of at a build.
//!
//! Rows are deduplicated and ordered, because a key set is a set: two layers of one shape share one
//! program.

use ops::Op;

/// The launches a component makes, as the operator and the key its descriptor states.
#[derive(Default)]
pub struct Programs {
    rows: Vec<(String, String)>,
}

impl Programs {
    /// Add one launch.
    ///
    /// The operator is the one the launch names ([`Op::NAME`]) and the key is the descriptor's own
    /// text, so a row is exactly what a store is asked for at that launch.
    pub fn launch<O: Op>(&mut self, descriptor: &O::Descriptor) {
        self.rows
            .push((O::NAME.to_owned(), ops::descriptor::key(descriptor)));
    }

    /// Add rows another component stated, keeping its operators and keys as they are.
    ///
    /// A denoiser composes two stacks and their orchestration; absorbing each stack's rows rather than
    /// restating them is what keeps one construction per program.
    pub fn extend(&mut self, rows: impl IntoIterator<Item = (String, String)>) {
        self.rows.extend(rows);
    }

    /// The rows, deduplicated and ordered by operator and then key.
    pub fn finish(mut self) -> Vec<(String, String)> {
        self.rows.sort();
        self.rows.dedup();
        self.rows
    }

    /// Whether no launch has been added.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}
