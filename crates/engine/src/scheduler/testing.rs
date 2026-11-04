//! The policy source a test world installs, and the WebAssembly-text encoder
//! the switch's own tests state their modules with.
//!
//! A deployed epoch takes its policy from the Program Server (§5.1, §7.3); a
//! test world has none, and a scheduler with no policy decides nothing — no
//! plan, no admission, no batch — so a world that submits a request must be
//! given a module. [`packaged_policy`] is that source: the real `fcfs` program
//! this tree builds, read from `SCHED_POLICY_WASM`, with the arguments that make
//! its declared limits fit the capacity the caller's world was built for. It
//! installs through the same precopy and switchover a deployment uses, so a
//! caller's fixtures exercise the mechanism rather than a stand-in for it, and
//! there is no second policy in the tree to keep honest.
//!
//! [`toy_module`] is separate and older: a module written as WebAssembly text
//! for the switch's own unit tests, which need answers a real policy never
//! gives (a trap, a spent fuel budget, a decision outside the limits). It is
//! not a policy anyone serves with.
//!
//! Both live here so the crate states the call protocol in one place, and both
//! are compiled only under the `testing` feature.

#[cfg(test)]
use scheduling::guest::decision::{Plan, Victim};
#[cfg(test)]
use scheduling::guest::ids::PageId;
use scheduling::guest::limits::{Limits, Parameters};
#[cfg(test)]
use scheduling::guest::view::ExpertOrder;
#[cfg(test)]
use scheduling::guest::wire::Call;
#[cfg(test)]
use serde::Serialize;

use super::policy::{BatchCapacity, PolicyProgram, ceilings_of};

/// Where a module keeps its message buffer and its replies.
#[cfg(test)]
const MESSAGE: u32 = 4096;
#[cfg(test)]
const PLAN: u32 = 8192;
#[cfg(test)]
const LIMITS: u32 = 12288;
#[cfg(test)]
const VICTIMS: u32 = 16384;
#[cfg(test)]
const PAGES: u32 = 20480;
#[cfg(test)]
const EXPERTS: u32 = 24576;

/// The answer a toy policy module gives, and the calls it fails in.
#[cfg(test)]
pub(crate) struct Toy {
    pub(crate) plan: Plan,
    pub(crate) limits: Limits,
    pub(crate) victims: Vec<Victim>,
    pub(crate) pages: Vec<PageId>,
    pub(crate) experts: ExpertOrder,
    /// The call whose export traps (`unreachable`).
    pub(crate) trap: Option<Call>,
    /// The call whose export never returns.
    pub(crate) spin: Option<Call>,
}

#[cfg(test)]
impl Default for Toy {
    fn default() -> Self {
        Self {
            plan: Plan::default(),
            limits: fitting_limits(),
            victims: Vec::new(),
            pages: Vec::new(),
            experts: ExpertOrder::default(),
            trap: None,
            spin: None,
        }
    }
}

/// The module bytes of `toy`.
#[cfg(test)]
pub(crate) fn toy_module(toy: &Toy) -> Vec<u8> {
    let bodies = [
        (Call::Dispatch, PLAN, json(&toy.plan)),
        (Call::Reclaim, VICTIMS, json(&toy.victims)),
        (Call::Warm, PAGES, json(&toy.pages)),
        (Call::Experts, EXPERTS, json(&toy.experts)),
    ];
    let mut functions = vec![
        format!("(func (export \"sched_alloc\") (param i32) (result i32) i32.const {MESSAGE})"),
        "(func (export \"sched_free\") (param i32 i32))".to_owned(),
        "(func (export \"sched_init\") (param i32 i32) (result i64) i64.const 0)".to_owned(),
        "(func (export \"sched_observe\") (param i32 i32) (result i64) i64.const 0)".to_owned(),
    ];
    let mut data = vec![format!(
        "(data (i32.const {PLAN}) \"{}\")",
        escape(&json(&toy.plan))
    )];
    for (call, offset, answer) in bodies {
        data.push(format!(
            "(data (i32.const {offset}) \"{}\")",
            escape(&answer)
        ));
        functions.push(buffered(call, offset, answer.len(), toy));
    }
    functions.push(buffered(Call::Limits, LIMITS, json(&toy.limits).len(), toy));
    data.push(format!(
        "(data (i32.const {LIMITS}) \"{}\")",
        escape(&json(&toy.limits))
    ));
    for (call, index) in [(Call::Route, 0), (Call::Place, 0)] {
        functions.push(indexed(call, index, toy));
    }
    let source = format!(
        "(module (memory (export \"memory\") 1) {} {})",
        data.join(" "),
        functions.join(" ")
    );
    wat::parse_str(&source).expect("the toy policy module is valid WebAssembly text")
}

/// The export of a call that answers with a reply buffer.
#[cfg(test)]
fn buffered(call: Call, offset: u32, len: usize, toy: &Toy) -> String {
    let params = if call == Call::Limits {
        String::new()
    } else {
        "(param i32 i32)".to_owned()
    };
    let packed = (u64::from(offset) << 32) | u64::try_from(len).expect("a reply fits u64");
    if toy.trap == Some(call) {
        format!(
            "(func (export \"{}\") {params} (result i64) unreachable)",
            call.export()
        )
    } else if toy.spin == Some(call) {
        format!(
            "(func (export \"{}\") {params} (result i64) (loop br 0) unreachable)",
            call.export()
        )
    } else {
        format!(
            "(func (export \"{}\") {params} (result i64) i64.const {packed})",
            call.export()
        )
    }
}

/// The export of a call that answers with the chosen index.
#[cfg(test)]
fn indexed(call: Call, index: u32, toy: &Toy) -> String {
    if toy.trap == Some(call) {
        format!(
            "(func (export \"{}\") (param i32 i32) (result i32) unreachable)",
            call.export()
        )
    } else {
        format!(
            "(func (export \"{}\") (param i32 i32) (result i32) i32.const {index})",
            call.export()
        )
    }
}

/// `value` as the protocol's JSON.
#[cfg(test)]
fn json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("an interface value is JSON")
}

/// `text` escaped for a WebAssembly text data segment.
#[cfg(test)]
fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Limits a test's build config allows: below every ceiling the mechanism
/// states for a pool the tests build.
#[cfg(test)]
pub(crate) fn fitting_limits() -> Limits {
    Limits {
        chunked_prefill_size: 256,
        max_batch: 4,
        max_num_batched_tokens: 512,
        ..Limits::default()
    }
}

/// The arguments that state `limits` as a program's parameters (§4.4, §7.7):
/// every limit a deployment can state.
fn parameters_of(limits: Limits) -> Parameters {
    Parameters {
        chunked_prefill_size: Some(limits.chunked_prefill_size),
        max_queued_requests: limits.max_queued_requests,
        max_batch: Some(limits.max_batch),
        max_num_batched_tokens: Some(limits.max_num_batched_tokens),
        prefetch_min_tokens: Some(limits.prefetch_min_tokens),
        timeout: Some(limits.timeout),
        ..Parameters::default()
    }
}

/// The `fcfs` policy module this tree builds, and the program that names it:
/// the source a caller installs through the real precopy and switchover.
///
/// The program is `fcfs` revision one with the arguments that state
/// [`ceilings_of`]`(capacity)` — the same derivation the mechanism's own
/// integration worlds use to install their policy — so the limits it declares
/// fit any fixture's build config and the mechanism's check of a policy's
/// limits against the ceilings cannot refuse it.
///
/// The module is read from `SCHED_POLICY_WASM`, which this function **requires**:
/// a test that calls it needs the export, and an unset variable fails here with
/// the command that builds it rather than serving a policy nobody named. There
/// is no prebuilt module in the tree, so the bytes a test runs are always the
/// ones this tree's `fcfs` compiles to.
///
/// A second policy builds too (`scheduling_policy_priority.wasm`); a caller
/// testing a switchover between two policies needs a second module path, which
/// this function does not name.
pub fn packaged_policy(capacity: BatchCapacity) -> (PolicyProgram, Vec<u8>) {
    (packaged_program(capacity), packaged_module())
}

/// The program `fcfs` revision one runs under for a world of `capacity`.
fn packaged_program(capacity: BatchCapacity) -> PolicyProgram {
    PolicyProgram {
        backend: "fcfs".to_owned(),
        revision: 1,
        arguments: parameters_of(ceilings_of(capacity)),
        stated: std::collections::BTreeMap::new(),
    }
}

/// The module `SCHED_POLICY_WASM` names: the `fcfs` program this tree builds
/// for `wasm32-unknown-unknown`.
fn packaged_module() -> Vec<u8> {
    let path = std::env::var("SCHED_POLICY_WASM").unwrap_or_else(|_| {
        panic!(
            "SCHED_POLICY_WASM names the scheduling policy module this source serves with, and it              is unset. Build it with `cargo build -p scheduling-policy-fcfs --target \
             wasm32-unknown-unknown --release` and set SCHED_POLICY_WASM to the file it wrote, \
             e.g. target/wasm32-unknown-unknown/release/scheduling_policy_fcfs.wasm"
        )
    });
    std::fs::read(&path).unwrap_or_else(|error| {
        panic!("SCHED_POLICY_WASM names `{path}`, which cannot be read: {error}")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The capacity of a pool the tests build their scheduler for.
    fn capacity() -> BatchCapacity {
        BatchCapacity {
            max_batch: 8,
            max_num_batched_tokens: 1024,
        }
    }

    #[test]
    fn the_packaged_program_states_limits_the_build_config_allows() {
        let program = packaged_program(capacity());
        assert_eq!(program.backend, "fcfs", "the module is this tree's fcfs");
        assert_eq!(program.revision, 1);
        assert!(
            program.stated.is_empty(),
            "no argument is stated, so a Program Server resolves none for it"
        );

        // The arguments state the ceilings themselves, so the limits the module
        // declares are the ones the mechanism checks a policy against and
        // cannot be refused for a world of this capacity.
        let limits = Limits::from_parameters(&program.arguments);
        let ceilings = ceilings_of(capacity());
        assert_eq!(limits.max_batch, ceilings.max_batch);
        assert_eq!(
            limits.max_num_batched_tokens,
            ceilings.max_num_batched_tokens
        );
        assert_eq!(limits.chunked_prefill_size, ceilings.chunked_prefill_size);
        assert_eq!(limits.prefetch_min_tokens, ceilings.prefetch_min_tokens);
        assert_eq!(limits.timeout, ceilings.timeout);
    }

    #[test]
    fn a_world_built_for_fewer_calls_states_fewer() {
        // `fcfs`'s own default declares 128 calls, which a test world built for
        // two would be refused for: the program's arguments are what make the
        // declaration fit.
        let small = packaged_program(BatchCapacity {
            max_batch: 2,
            max_num_batched_tokens: 64,
        });
        assert_eq!(small.arguments.max_batch, Some(2));
        assert_eq!(small.arguments.max_num_batched_tokens, Some(64));
        assert_eq!(
            Limits::from_parameters(&small.arguments).max_batch,
            2,
            "the module declares what this program states, not the package default"
        );
    }
}
