//! The WebAssembly-text policy modules the policy switch's cases state.
//!
//! A [`Toy`] answers what a real policy never gives — a trap, a spent fuel budget, a decision
//! outside the limits — so the switch's refusals can be stated. It is not a policy anyone serves
//! with; a test epoch installs the packaged `fcfs` module (`testkit::packaged_policy`).

use scheduling::guest::decision::{DispatchOutput, Victim};
use scheduling::guest::limits::Limits;
use scheduling::guest::wire::Call;
use serde::Serialize;

/// Where a module keeps its message buffer and its replies.
const MESSAGE: u32 = 4096;
const OUTPUT: u32 = 8192;
const LIMITS: u32 = 12288;
const VICTIMS: u32 = 16384;

/// The answer a toy policy module gives, and the calls it fails in.
pub(crate) struct Toy {
    pub(crate) output: DispatchOutput,
    pub(crate) limits: Limits,
    pub(crate) victims: Vec<Victim>,
    /// The call whose export traps (`unreachable`).
    pub(crate) trap: Option<Call>,
    /// The call whose export never returns.
    pub(crate) spin: Option<Call>,
    /// The rank index `route` answers.
    pub(crate) route: u32,
}

impl Default for Toy {
    fn default() -> Self {
        Self {
            output: DispatchOutput::default(),
            limits: fitting_limits(),
            victims: Vec::new(),
            trap: None,
            spin: None,
            route: 0,
        }
    }
}

/// The module bytes of `toy`.
pub(crate) fn toy_module(toy: &Toy) -> Vec<u8> {
    let bodies = [
        (Call::Dispatch, OUTPUT, json(&toy.output)),
        (Call::Preempt, VICTIMS, json(&toy.victims)),
    ];
    let mut functions = vec![
        format!("(func (export \"sched_alloc\") (param i32) (result i32) i32.const {MESSAGE})"),
        "(func (export \"sched_free\") (param i32 i32))".to_owned(),
        "(func (export \"sched_init\") (param i32 i32) (result i64) i64.const 0)".to_owned(),
        "(func (export \"sched_observe\") (param i32 i32) (result i64) i64.const 0)".to_owned(),
    ];
    let mut data = vec![format!(
        "(data (i32.const {OUTPUT}) \"{}\")",
        escape(&json(&toy.output))
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
    for (call, index) in [(Call::Route, toy.route), (Call::Place, 0)] {
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
fn json<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).expect("an interface value is JSON")
}

/// `text` escaped for a WebAssembly text data segment.
fn escape(text: &str) -> String {
    text.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Limits a test's build config allows: below every ceiling the mechanism
/// states for a pool the tests build.
pub(crate) fn fitting_limits() -> Limits {
    Limits {
        chunked_prefill_size: 256,
        max_batch: 4,
        max_num_batched_tokens: 512,
        ..Limits::default()
    }
}
