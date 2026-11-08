//! Committing a sampling step's accepted tokens into the request slots' state.
//!
//! Once a step's tokens are chosen, one launch folds them into the slots: the next input token, the
//! continuation and stop predicate, the three lengths a slot tracks, and the penalty counts that bias
//! later sampling. The state is the worker's — `ops` names it and writes it, allocates nothing and
//! keeps nothing (`uniserve/sampling/state.py`; specs/worker.md 9, the slot-state row).
//!
//! ```text
//! decode    verified += 1                            logical += 1   sampling += 1
//! verify    verified  = visible + points              logical  = host_logical + points
//!                                                     sampling = host_sampling + points
//! explicit  verified  = host cache length             logical, sampling from the host columns
//! all       future[slot, 0] = token;  predicate[slot] = continuation;
//!           penalties[slot, token] += valid · active   (for the rows that count penalties)
//! ```
//!
//! The rows come from the step's `[8, capacity]` int64 column table — slot, mode, task, the host cache
//! length, host logical and sampling positions, the verification base visible length and whether the
//! row counts penalties. Slots are unique within a step, so each row's reads and writes of its slot
//! need no ordering; the penalty accumulation is an atomic add, as an indexed accumulation is.
//!
//! **Two sentinels make a row inert, and they are what lets a warm batch be recorded.** A row whose
//! *task* is `-1` samples nothing: its token-state stores and its penalty add are skipped, and the
//! per-task views are not read (the loads are masked off). A row whose *slot* is `-1` has no
//! destination at all: **every store into the slot state is skipped**, including the three lengths
//! this operator maintains. So `count` may be the bucket's whole row count with padding rows staged
//! as `slot = task = -1`, and the launch changes nothing — which is how a warmup capture can record
//! the serving graph without advancing a slot's counts or its positions.
//!
//! The masked-by-slot rule is deliberate where it differs from the reference
//! (`uniserve/sampling/state.py::_commit_kernel` stores the cache length under the row mask, not
//! under `slot >= 0`, so a padded row there writes `state[-1]`: a host-side bug that a CUDA launch
//! would turn into a write outside the tensor). The sentinel is honoured here on every store, and the
//! catalogue states it.

use std::fmt;

use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::platform::Stream;
use crate::tensor::{DType, TensorView};
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

/// The rows of the commit column table, in order.
pub const COMMIT_FIELDS: usize = 8;

/// The column table's row indices.
pub mod column {
    /// The row's destination slot.
    pub const SLOT: usize = 0;
    /// The row's mode: decode, explicit, verify or cache.
    pub const MODE: usize = 1;
    /// The task the row sampled, or -1 when it sampled none.
    pub const TASK: usize = 2;
    /// The host cache length of an explicit row.
    pub const CACHE_LENGTH: usize = 3;
    /// The host logical position of an explicit row.
    pub const LOGICAL: usize = 4;
    /// The host sampling position of an explicit row.
    pub const SAMPLING: usize = 5;
    /// The verification base's visible length.
    pub const VISIBLE: usize = 6;
    /// Whether the row counts penalties.
    pub const PENALTIES: usize = 7;
}

/// The mode of a commit row.
pub mod mode {
    /// One accepted token: the three lengths advance by one.
    pub const DECODE: i64 = 0;
    /// The host columns state the lengths.
    pub const EXPLICIT: i64 = 1;
    /// A verified span: the lengths move to `base + points`.
    pub const VERIFY: i64 = 2;
    /// The lengths are left as they are.
    pub const CACHE: i64 = 3;
}

/// One compilation of `slot_commit`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct CommitDescriptor {
    /// Element type of the column table, the sampled tokens, the accepted spans, and the slot's
    /// logical and sampling positions.
    ///
    /// The predicates are not in it: `continuation` and `predicates` are `U8` by construction — the
    /// launch refuses any other type for them — because they are booleans a kernel test and stores,
    /// not lengths it compares.
    pub index: DType,
    /// Element type of the slot's verified length.
    pub verified: DType,
    /// Element type of the slot's penalty counts.
    pub penalties: DType,
    /// Elements between two columns of the commit table.
    pub stride: u64,
    /// Elements between two slots of the next-input-token matrix.
    pub future_stride: u64,
    /// The vocabulary width of the penalty counts.
    pub vocab: u32,
    /// The lanes one program covers: the step's row count rounded up to a power of two.
    pub block: u32,
}

impl fmt::Display for CommitDescriptor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Key::new()
            .field("index", self.index)
            .field("verified", self.verified)
            .field("penalties", self.penalties)
            .field("stride", self.stride)
            .field("future_stride", self.future_stride)
            .field("vocab", self.vocab)
            .field("block", self.block)
            .finish(formatter)
    }
}

/// The arguments of `slot_commit`.
///
/// The per-task columns are indexed by each row's task; the state tensors are indexed by its slot.
///
/// A sampled token is a row of `penalties`, so the operator installs only a token that is one: a
/// value at or above the descriptor's `vocab` writes no next input column, no predicate and no
/// penalty weight, and leaves the slot's token state as it was. A sampler can answer such a value
/// (an `argmax` over a row with no numeric value answers the row's width), and installing it would
/// hand the next step an id the embedding cannot address and land a count in the next slot's row.
/// Whether the batch is served is the caller's decision, taken where the tokens are read back.
#[derive(Clone, Copy)]
pub struct CommitArgs {
    /// `[8, capacity]` i64: the step's commit rows, `stride` elements between columns.
    pub columns: TensorView,
    /// `[tasks]` i64: the sampled token of each task.
    pub tokens: TensorView,
    /// `[tasks]`: whether the task's sample is valid.
    pub valid: TensorView,
    /// `[tasks]`: whether the task is active in this step.
    pub active: TensorView,
    /// `[tasks]` u8: whether the task continues.
    pub continuation: TensorView,
    /// `[tasks]`: the accepted span's length per task.
    pub points: TensorView,
    /// `[slots, width]` i64: the next input token goes to column 0.
    pub future: TensorView,
    /// `[slots]` u8: receives the continuation predicate.
    pub predicates: TensorView,
    /// `[slots]`: the slot's logical position.
    pub logical: TensorView,
    /// `[slots]`: the slot's sampling position.
    pub sampling: TensorView,
    /// `[slots]` i32: the slot's verified length.
    pub verified: TensorView,
    /// `[slots, vocab]` i32: accumulates the accepted tokens' penalty weights.
    pub penalties: TensorView,
    /// The step's commit rows.
    pub count: u64,
}

/// Apply a sampling step's commit rows to the slots' state.
pub struct SlotCommit;

/// Check one launch's views and descriptor against the operator's contract.
///
/// Extracted from [`SlotCommit::launch`] so that the refusals — which are the whole reason a wrong
/// call fails loudly instead of corrupting a slot's state — are testable without a device.
fn validate(descriptor: &CommitDescriptor, args: &CommitArgs) -> Result<()> {
    if args.count == 0 {
        return Err(Error::argument(
            SlotCommit::NAME,
            "a step of zero rows commits nothing".to_owned(),
        ));
    }
    if descriptor.block == 0 || !descriptor.block.is_power_of_two() {
        return Err(Error::argument(
            SlotCommit::NAME,
            format!("the {} lane block is not a power of two", descriptor.block),
        ));
    }
    if u64::from(descriptor.block) < args.count {
        return Err(Error::argument(
            SlotCommit::NAME,
            format!(
                "a block of {} lanes cannot cover {} rows",
                descriptor.block, args.count
            ),
        ));
    }
    args.columns.require_rank(SlotCommit::NAME, 2)?;
    if args.columns.shape.dim(0) < COMMIT_FIELDS as u64 {
        return Err(Error::argument(
            SlotCommit::NAME,
            format!(
                "the commit table has {} columns, a row is {COMMIT_FIELDS}",
                args.columns.shape.dim(0)
            ),
        ));
    }
    check_type(
        SlotCommit::NAME,
        "the commit table",
        &args.columns,
        descriptor.index,
    )?;
    if args.columns.strides.get(1) != 1 || args.columns.strides.get(0) != descriptor.stride {
        return Err(Error::argument(
            SlotCommit::NAME,
            format!(
                "the commit columns are {} elements apart and {} apart within a column, the \
                 descriptor says {}",
                args.columns.strides.get(0),
                args.columns.strides.get(1),
                descriptor.stride
            ),
        ));
    }
    if args.columns.shape.dim(1) < args.count {
        return Err(Error::argument(
            SlotCommit::NAME,
            format!(
                "the commit table holds {} rows, the step commits {}",
                args.columns.shape.dim(1),
                args.count
            ),
        ));
    }
    for view in [
        &args.tokens,
        &args.valid,
        &args.active,
        &args.continuation,
        &args.points,
    ] {
        view.require_rank(SlotCommit::NAME, 1)?;
    }
    // `valid` and `active` are flags whose product is the penalty weight; the kernel casts that
    // product to `i32`, so the views must be integer — a float flag would silently truncate. The
    // runner's own convention is `i32`, and `u8` costs nothing to accept.
    for (name, view) in [
        ("the valid flags", &args.valid),
        ("the active flags", &args.active),
    ] {
        if !matches!(view.dtype, DType::I32 | DType::I64 | DType::U8) {
            return Err(Error::argument(
                SlotCommit::NAME,
                format!(
                    "{name} are {}, an integer flag is i32, i64 or u8",
                    view.dtype
                ),
            ));
        }
    }
    for (name, view) in [
        ("the tokens", &args.tokens),
        ("the points", &args.points),
        ("the logical positions", &args.logical),
        ("the sampling positions", &args.sampling),
    ] {
        check_type(SlotCommit::NAME, name, view, descriptor.index)?;
    }
    for (name, view) in [
        ("the continuation", &args.continuation),
        ("the predicates", &args.predicates),
    ] {
        check_type(SlotCommit::NAME, name, view, DType::U8)?;
    }
    check_type(
        SlotCommit::NAME,
        "the verified lengths",
        &args.verified,
        descriptor.verified,
    )?;
    check_type(
        SlotCommit::NAME,
        "the penalty counts",
        &args.penalties,
        descriptor.penalties,
    )?;
    args.predicates.require_rank(SlotCommit::NAME, 1)?;
    args.logical.require_rank(SlotCommit::NAME, 1)?;
    args.sampling.require_rank(SlotCommit::NAME, 1)?;
    args.verified.require_rank(SlotCommit::NAME, 1)?;
    args.future.require_rank(SlotCommit::NAME, 2)?;
    check_type(
        SlotCommit::NAME,
        "the next-input tokens",
        &args.future,
        descriptor.index,
    )?;
    if args.future.strides.get(0) != descriptor.future_stride {
        return Err(Error::argument(
            SlotCommit::NAME,
            format!(
                "the next-input rows are {} elements apart, the descriptor says {}",
                args.future.strides.get(0),
                descriptor.future_stride
            ),
        ));
    }
    let slots = args.future.shape.dim(0);
    for (name, view) in [
        ("the predicates", &args.predicates),
        ("the logical positions", &args.logical),
        ("the sampling positions", &args.sampling),
        ("the verified lengths", &args.verified),
    ] {
        if view.shape.dim(0) < slots {
            return Err(Error::argument(
                SlotCommit::NAME,
                format!(
                    "the {name} hold {} slots, the next-input matrix {slots}",
                    view.shape.dim(0)
                ),
            ));
        }
    }
    args.penalties.require_rank(SlotCommit::NAME, 2)?;
    if args.penalties.shape.dim(0) < slots
        || args.penalties.shape.dim(1) != u64::from(descriptor.vocab)
    {
        return Err(Error::argument(
            SlotCommit::NAME,
            format!("the penalty counts are not [{slots}, {}]", descriptor.vocab),
        ));
    }
    Ok(())
}

impl Op for SlotCommit {
    const NAME: &'static str = "slot_commit";

    type Descriptor = CommitDescriptor;
    type Args<'a> = CommitArgs;

    fn launch(
        kernel: &Kernel,
        descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        validate(descriptor, &args)?;

        let mut arguments = Arguments::new();
        arguments
            .view(args.columns)
            .view(args.tokens)
            .view(args.valid)
            .view(args.active)
            .view(args.continuation)
            .view(args.points)
            .view(args.future)
            .view(args.predicates)
            .view(args.logical)
            .view(args.sampling)
            .view(args.verified)
            .view(args.penalties)
            .i64(args.count as i64);
        kernel.launch(LaunchGrid::linear(1)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins, stream: &Stream) -> Self::Args<'a> {
        CommitArgs {
            future: twins.in_place(args.future, stream),
            predicates: twins.in_place(args.predicates, stream),
            logical: twins.in_place(args.logical, stream),
            sampling: twins.in_place(args.sampling, stream),
            verified: twins.in_place(args.verified, stream),
            penalties: twins.in_place(args.penalties, stream),
            ..args
        }
    }

    fn tolerance(_descriptor: &Self::Descriptor) -> Tolerance {
        // The state is integral: the lengths are set or advanced by one, and the penalty counts add
        // integer weights whose sums do not depend on the order of the additions.
        Tolerance::exact()
    }
}

/// Check a view's element type against the descriptor.
fn check_type(
    operator: &'static str,
    name: &str,
    view: &TensorView,
    expected: DType,
) -> Result<()> {
    if view.dtype != expected {
        return Err(Error::argument(
            operator,
            format!("{name} are {}, the descriptor says {expected}", view.dtype),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{DevicePtr, Shape, Strides};

    const SLOTS: u64 = 4;
    const VOCAB: u64 = 8;
    const FUTURE: u64 = 16;

    fn view(dtype: DType, shape: &[u64], strides: &[u64]) -> TensorView {
        TensorView::with_strides(
            DevicePtr::new(0x1000),
            dtype,
            Shape::new(shape),
            Strides::new(strides),
        )
    }

    fn descriptor() -> CommitDescriptor {
        CommitDescriptor {
            index: DType::I64,
            verified: DType::I32,
            penalties: DType::I32,
            stride: SLOTS,
            future_stride: FUTURE,
            vocab: VOCAB as u32,
            block: 4,
        }
    }

    /// A decoded step of three rows over four slots, every view the shape the descriptor asks for.
    fn args() -> CommitArgs {
        CommitArgs {
            count: 3,
            columns: view(DType::I64, &[COMMIT_FIELDS as u64, SLOTS], &[SLOTS, 1]),
            tokens: view(DType::I64, &[4], &[1]),
            valid: view(DType::I32, &[4], &[1]),
            active: view(DType::I32, &[4], &[1]),
            continuation: view(DType::U8, &[4], &[1]),
            points: view(DType::I64, &[4], &[1]),
            future: view(DType::I64, &[SLOTS, FUTURE], &[FUTURE, 1]),
            predicates: view(DType::U8, &[SLOTS], &[1]),
            logical: view(DType::I64, &[SLOTS], &[1]),
            sampling: view(DType::I64, &[SLOTS], &[1]),
            verified: view(DType::I32, &[SLOTS], &[1]),
            penalties: view(DType::I32, &[SLOTS, VOCAB], &[VOCAB, 1]),
        }
    }

    fn refusal(descriptor: &CommitDescriptor, args: &CommitArgs) -> String {
        format!(
            "{}",
            validate(descriptor, args).expect_err("the launch should have been refused")
        )
    }

    #[test]
    fn a_step_of_decoded_rows_passes_every_check() {
        validate(&descriptor(), &args()).expect("a plain step is valid");
    }

    #[test]
    fn a_step_of_no_rows_is_refused_and_the_block_covers_the_rows() {
        let mut empty = args();
        empty.count = 0;
        assert!(refusal(&descriptor(), &empty).contains("zero rows"));

        let mut narrow = descriptor();
        narrow.block = 2;
        assert!(refusal(&narrow, &args()).contains("cannot cover 3 rows"));

        let mut uneven = descriptor();
        uneven.block = 6;
        assert!(refusal(&uneven, &args()).contains("power of two"));
    }

    #[test]
    fn the_column_table_is_eight_columns_apart_by_the_descriptor_stride() {
        let mut table = args();
        table.columns = view(DType::I64, &[COMMIT_FIELDS as u64, SLOTS], &[SLOTS + 1, 1]);
        assert!(refusal(&descriptor(), &table).contains("the descriptor says"));

        let mut short = args();
        short.columns = view(
            DType::I64,
            &[(COMMIT_FIELDS - 1) as u64, SLOTS],
            &[SLOTS, 1],
        );
        assert!(refusal(&descriptor(), &short).contains("a row is"));

        // Fewer rows than the step commits: the mask cannot reach the missing ones.
        let mut few = args();
        few.columns = view(DType::I64, &[COMMIT_FIELDS as u64, 2], &[SLOTS, 1]);
        assert!(refusal(&descriptor(), &few).contains("the step commits 3"));
    }

    #[test]
    fn a_float_flag_is_refused_because_the_penalty_weight_is_an_integer_product() {
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let mut flags = args();
            flags.valid = view(dtype, &[4], &[1]);
            assert!(
                refusal(&descriptor(), &flags).contains("integer flag"),
                "{dtype}"
            );
        }
        let mut wide = args();
        wide.active = view(DType::U8, &[4], &[1]);
        validate(&descriptor(), &wide).expect("a u8 flag is an integer");
    }

    #[test]
    fn the_predicates_are_bytes_and_the_state_views_hold_a_slot_each() {
        let mut continuation = args();
        continuation.continuation = view(DType::I32, &[4], &[1]);
        assert!(refusal(&descriptor(), &continuation).contains("continuation"));

        let mut short = args();
        short.logical = view(DType::I64, &[2], &[1]);
        assert!(refusal(&descriptor(), &short).contains("hold 2 slots"));

        let mut penalties = args();
        penalties.penalties = view(DType::I32, &[SLOTS, VOCAB - 1], &[VOCAB - 1, 1]);
        assert!(refusal(&descriptor(), &penalties).contains("penalty counts are not"));

        let mut future = args();
        future.future = view(DType::I64, &[SLOTS, FUTURE], &[FUTURE + 1, 1]);
        assert!(refusal(&descriptor(), &future).contains("next-input rows are"));
    }

    #[test]
    fn the_tokens_carry_the_index_type_the_descriptor_names() {
        let mut tokens = args();
        tokens.tokens = view(DType::I32, &[4], &[1]);
        assert!(refusal(&descriptor(), &tokens).contains("the tokens"));
    }

    #[test]
    fn the_state_is_integral_so_the_bound_is_exact() {
        assert_eq!(SlotCommit::tolerance(&descriptor()), Tolerance::exact());
    }
}
