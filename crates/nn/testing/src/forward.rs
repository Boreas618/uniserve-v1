//! Forwards on the meta device: their inputs, their recordings, and the checks a profile run
//! makes of what it recorded.
//!
//! A forward over meta tensors under a meta context launches nothing: it checks every operand,
//! records the operator and the kernel key each launch would run, and counts what its allocator
//! hands out. A recording is therefore what a deployment sizes a stage by, and a test asserts
//! on it without a device.

use std::collections::{BTreeMap, BTreeSet};

use nn::{DType, Shape, Split, Tensor};

/// One launch of a recorded forward: its operator and its kernel key.
pub type Launch = (&'static str, String);

/// What a forward recorded under a meta context stated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recording {
    /// The launches, in launch order.
    pub launches: Vec<Launch>,
    /// The peak of the context's allocator, in bytes: the activations the forward needs.
    pub peak: u64,
    /// The peak of the host values the forward moved to the device, in bytes.
    pub host_bytes: u64,
}

/// A meta tensor of `shape` elements of `dtype` at an address of its own (`ops::meta_tensor`):
/// an input of a profile run, which takes no block of the context's allocator.
pub fn meta(shape: impl Into<Shape>, dtype: DType) -> Tensor {
    let shape = shape.into();

    ops::meta_tensor(shape.clone(), dtype)
        .unwrap_or_else(|error| panic!("a meta input of {shape:?} {dtype:?}: {error}"))
}

/// Runs `forward` under a new meta context that records its launches, and returns what it
/// returned with the context's [`Recording`].
///
/// `axes` states this rank's place on each collective axis of the forward, in the order the
/// forward addresses them (a group's single axis, or a denoiser's axes,
/// `nn::family::DenoiserSplit::axes`); a whole split stands for an axis the forward does not
/// partition, and a forward of no partitioned axis is recorded with none.
pub fn record<T>(
    axes: &[Split],
    forward: impl FnOnce() -> nn::Result<T>,
) -> (nn::Result<T>, Recording) {
    let context = ops::Context::meta(None);

    if axes.iter().any(|axis| *axis != Split::WHOLE) {
        // A meta context issues no collective, so each axis states its place alone: `ranks`
        // zero for an axis the forward does not partition.
        let places = axes
            .iter()
            .map(|axis| {
                if *axis == Split::WHOLE {
                    (None, 0, 0)
                } else {
                    (None, u64::from(axis.rank), u64::from(axis.size))
                }
            })
            .collect();
        context
            .set_axes(places)
            .unwrap_or_else(|error| panic!("the axes {axes:?} of a meta context: {error}"));
    }
    context.record_launches();

    let output = {
        let _guard = context.enter();
        forward()
    };
    let launches = context
        .take_launches()
        .into_iter()
        .map(|(operator, key, ..)| (operator, key))
        .collect();

    let recording = Recording {
        launches,
        peak: context.peak(),
        host_bytes: context.input_peak(),
    };
    (output, recording)
}

/// The operators of `launches`, in launch order.
pub fn operators(launches: &[Launch]) -> Vec<&'static str> {
    launches.iter().map(|(operator, _)| *operator).collect()
}

/// Asserts that `operators` holds every operator of `expected` and none of `absent`, naming
/// `what` launched them.
pub fn assert_operators(
    what: &str,
    operators: &BTreeSet<&'static str>,
    expected: &[&str],
    absent: &[&str],
) {
    for operator in expected {
        assert!(
            operators.contains(operator),
            "{what} launches no {operator}: {operators:?}"
        );
    }
    for operator in absent {
        assert!(
            !operators.contains(operator),
            "{what} launches {operator}: {operators:?}"
        );
    }
}

/// Asserts that `what` launched something and that every launch of `launches` names its
/// operator and its kernel key, neither naming a model family: no family id or model class of
/// the families this tree serves (`processing::FAMILIES`), in any case. A kernel is chosen by what a layer computes, never by which model it serves.
pub fn assert_generic(what: &str, launches: &[Launch]) {
    assert!(!launches.is_empty(), "{what} launched nothing");

    let families: BTreeSet<String> = processing::FAMILIES
        .iter()
        .flat_map(|family| {
            [family.id]
                .into_iter()
                .chain(family.classes.iter().copied())
        })
        .map(str::to_lowercase)
        .collect();

    for (index, (operator, key)) in launches.iter().enumerate() {
        assert!(
            !operator.is_empty() && !key.is_empty(),
            "{what}: launch {index} states operator {operator:?} and key {key:?}"
        );

        let named = format!("{operator} {key}").to_lowercase();
        if let Some(family) = families
            .iter()
            .find(|family| named.contains(family.as_str()))
        {
            panic!("{what}: {operator} {key} names the model family {family}");
        }
    }
}

/// Checks `recording` of the run `run` as [`assert_generic`] does and prints what a deployment
/// sizes the run by: its launches, its distinct kernel keys, its allocator and host peaks, its
/// operators by kind (the generic primitives, `ops::GENERIC_OPERATORS`, and the fused operators
/// of `nn`), and every key.
pub fn report(run: &str, recording: &Recording) {
    let Recording {
        launches,
        peak,
        host_bytes,
    } = recording;
    assert_generic(run, launches);

    // Each operator's launches and its distinct keys.
    let mut operators: BTreeMap<&str, (usize, BTreeSet<&str>)> = BTreeMap::new();
    for (operator, key) in launches {
        let entry = operators.entry(operator).or_default();
        entry.0 += 1;
        entry.1.insert(key);
    }
    let keys: usize = operators.values().map(|(_, keys)| keys.len()).sum();

    println!(
        "{run}: {} launches, {keys} kernel keys, allocator peak {peak} bytes ({:.1} MiB), \
         {host_bytes} bytes of host values",
        launches.len(),
        *peak as f64 / (1 << 20) as f64
    );
    for (kind, generic) in [("generic primitives", true), ("fused operators", false)] {
        let listed: Vec<String> = operators
            .iter()
            .filter(|(operator, _)| ops::GENERIC_OPERATORS.contains(operator) == generic)
            .map(|(operator, (count, keys))| format!("{operator} {count}/{}", keys.len()))
            .collect();

        println!("  {kind} (launches/keys): {}", listed.join(", "));
    }
    for (operator, (_, keys)) in &operators {
        for key in keys {
            println!("    {operator}: {key}");
        }
    }
}
