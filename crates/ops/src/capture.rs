//! Which graph nodes a capture belongs to.
//!
//! A captured graph is replayed as a whole, but a kernel program may be replaced one operator at a
//! time (specs/worker.md 4.5): the executor hands a graph node new parameters for the operator whose
//! program changed. The capture records node handles in launch order, so it needs the same order
//! annotated with the operator each launch belongs to. That annotation travels through this module:
//! the layer that launches installs a [`Sink`] for the duration of a capture, and every launch calls
//! [`launched`] with the operator's name and its descriptor key.
//!
//! The sink is one thread's, because one capture runs on one thread. It is not a runtime object of
//! any model: the executor owns it, installs it around its own capture, and removes it afterwards.

use std::cell::Cell;
use std::ptr::NonNull;

/// What a capture records about each launch.
pub trait Sink {
    /// The launch of `operator`'s program with this descriptor key is the next node.
    fn launched(&mut self, operator: &'static str, key: &str);
}

thread_local! {
    /// The sink installed for this thread, or none.
    static SINK: Cell<Option<NonNull<dyn Sink + 'static>>> = const { Cell::new(None) };
}

/// Install `sink` for the launches on this thread, or remove it with `None`.
///
/// The caller keeps `sink` alive and exclusively borrowed from this call until it removes it with
/// `None` on the same thread; a sink may not launch operators itself, directly or through a layer.
/// A sink owns its data (it borrows nothing), which is what lets one thread hold it: the executor's
/// capture recorder collects handles it owns.
pub fn set_sink(sink: Option<&mut (dyn Sink + 'static)>) {
    let pointer = sink.map(NonNull::from);
    SINK.with(|cell| cell.set(pointer));
}

/// Tell the installed sink about a launch, if one is installed.
///
/// Called by the layer that launches, right after the launch: `operator` is [`crate::Op::NAME`] and
/// `key` the descriptor's [`Display`](std::fmt::Display) text, which is the program's selection key.
pub fn launched(operator: &'static str, key: &str) {
    SINK.with(|cell| {
        let Some(mut pointer) = cell.get() else {
            return;
        };
        // SAFETY: `set_sink(Some)` is called with a pointer to a sink the caller keeps alive and
        // exclusively borrowed until `set_sink(None)` on this thread, and a sink may not launch, so
        // no other reference to it exists here.
        let sink = unsafe { pointer.as_mut() };
        sink.launched(operator, key);
    });
}

#[cfg(test)]
mod tests {
    use super::{Sink, launched, set_sink};

    struct Recording(Vec<(String, String)>);

    impl Sink for Recording {
        fn launched(&mut self, operator: &'static str, key: &str) {
            self.0.push((operator.to_owned(), key.to_owned()));
        }
    }

    #[test]
    fn launches_reach_the_installed_sink_and_stop_when_it_is_removed() {
        let mut recording = Recording(Vec::new());
        set_sink(Some(&mut recording));
        launched("rms_norm", "dtype_bf16_columns_2048");
        launched("gemm", "input_bf16_k_2048");
        set_sink(None);
        launched("top_k", "dtype_f32_width_2048");

        assert_eq!(
            recording.0,
            [
                ("rms_norm".to_owned(), "dtype_bf16_columns_2048".to_owned()),
                ("gemm".to_owned(), "input_bf16_k_2048".to_owned()),
            ]
        );
    }
}
