//! Committing a sampling step's accepted tokens into the request slots' state.
//!
//! Once a step's tokens are chosen, one launch folds them into the slots: the next input token, the
//! continuation and stop predicate, the three lengths a slot tracks, and the penalty counts that bias
//! later sampling. The state is the worker's — `ops` names it and writes it, allocates nothing and
//! keeps nothing.
//!
//! ```text
//! decode    verified += 1                            logical += 1   sampling += 1
//! verify    verified  = visible + points              logical  = host_logical + points
//!                                                     sampling = host_sampling + points
//! explicit  verified  = host cache length             logical, sampling from the host columns
//! count     nothing but the penalty add below          (a verification's accepted drafts)
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
//! The masked-by-slot rule is deliberate: a length stored under the row mask rather than under
//! `slot >= 0` would make a padded row write `state[-1]`, which a CUDA launch turns into a write
//! outside the tensor. The sentinel is honoured here on every store, and the
//! catalogue states it.

use std::fmt;

use platform::Stream;
use tensor::DType;

use crate::abi;
use crate::descriptor::Key;
use crate::kernel::{Arguments, Kernel, LaunchGrid};
use crate::operand::Operand;
use crate::tolerance::Tolerance;
use crate::{Error, Op, Result, Twins};

use proto::module::commit::COMMIT_FIELDS;

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
            .field("index", abi::name(self.index))
            .field("verified", abi::name(self.verified))
            .field("penalties", abi::name(self.penalties))
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
#[derive(Clone)]
pub struct CommitArgs {
    /// `[8, capacity]` i64: the step's commit rows, `stride` elements between columns.
    pub columns: Operand,
    /// `[tasks]` i64: the sampled token of each task.
    pub tokens: Operand,
    /// `[tasks]`: whether the task's sample is valid.
    pub valid: Operand,
    /// `[tasks]`: whether the task is active in this step.
    pub active: Operand,
    /// `[tasks]` u8: whether the task continues.
    pub continuation: Operand,
    /// `[tasks]`: the accepted span's length per task.
    pub points: Operand,
    /// `[slots, width]` i64: the next input token goes to column 0.
    pub future: Operand,
    /// `[slots]` u8: receives the continuation predicate.
    pub predicates: Operand,
    /// `[slots]`: the slot's logical position.
    pub logical: Operand,
    /// `[slots]`: the slot's sampling position.
    pub sampling: Operand,
    /// `[slots]` i32: the slot's verified length.
    pub verified: Operand,
    /// `[slots, vocab]` i32: accumulates the accepted tokens' penalty weights.
    pub penalties: Operand,
    /// The step's commit rows.
    pub count: u64,
}

/// Apply a sampling step's commit rows to the slots' state.
pub struct SlotCommit;

impl Op for SlotCommit {
    const NAME: &'static str = proto::module::operator::SLOT_COMMIT;

    type Descriptor = CommitDescriptor;
    type Args<'a> = CommitArgs;

    /// The refusals here are the whole reason a wrong call fails loudly instead of corrupting a
    /// slot's state, and they need no device.
    fn check(descriptor: &Self::Descriptor, args: &Self::Args<'_>) -> Result<()> {
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
        if args.columns.dim(0) < COMMIT_FIELDS as u64 {
            return Err(Error::argument(
                SlotCommit::NAME,
                format!(
                    "the commit table has {} columns, a row is {COMMIT_FIELDS}",
                    args.columns.dim(0)
                ),
            ));
        }
        check_type(
            SlotCommit::NAME,
            "the commit table",
            &args.columns,
            descriptor.index,
        )?;
        if args.columns.stride(1) != 1 || args.columns.stride(0) != descriptor.stride {
            return Err(Error::argument(
                SlotCommit::NAME,
                format!(
                    "the commit columns are {} elements apart and {} apart within a column, the \
                     descriptor says {}",
                    args.columns.stride(0),
                    args.columns.stride(1),
                    descriptor.stride
                ),
            ));
        }
        if args.columns.dim(1) < args.count {
            return Err(Error::argument(
                SlotCommit::NAME,
                format!(
                    "the commit table holds {} rows, the step commits {}",
                    args.columns.dim(1),
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
                        abi::name(view.dtype)
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
        if args.future.stride(0) != descriptor.future_stride {
            return Err(Error::argument(
                SlotCommit::NAME,
                format!(
                    "the next-input rows are {} elements apart, the descriptor says {}",
                    args.future.stride(0),
                    descriptor.future_stride
                ),
            ));
        }
        let slots = args.future.dim(0);
        for (name, view) in [
            ("the predicates", &args.predicates),
            ("the logical positions", &args.logical),
            ("the sampling positions", &args.sampling),
            ("the verified lengths", &args.verified),
        ] {
            if view.dim(0) < slots {
                return Err(Error::argument(
                    SlotCommit::NAME,
                    format!(
                        "the {name} hold {} slots, the next-input matrix {slots}",
                        view.dim(0)
                    ),
                ));
            }
        }
        args.penalties.require_rank(SlotCommit::NAME, 2)?;
        if args.penalties.dim(0) < slots || args.penalties.dim(1) != u64::from(descriptor.vocab) {
            return Err(Error::argument(
                SlotCommit::NAME,
                format!("the penalty counts are not [{slots}, {}]", descriptor.vocab),
            ));
        }
        Ok(())
    }

    fn launch(
        kernel: &Kernel,
        _descriptor: &Self::Descriptor,
        args: Self::Args<'_>,
        stream: &Stream,
    ) -> Result<()> {
        let mut arguments = Arguments::new();
        arguments
            .view(&args.columns)
            .view(&args.tokens)
            .view(&args.valid)
            .view(&args.active)
            .view(&args.continuation)
            .view(&args.points)
            .view(&args.future)
            .view(&args.predicates)
            .view(&args.logical)
            .view(&args.sampling)
            .view(&args.verified)
            .view(&args.penalties)
            .i64(args.count as i64);
        kernel.launch(LaunchGrid::linear(1)?, &mut arguments, stream)
    }

    fn shadow<'a>(args: Self::Args<'a>, twins: &mut Twins) -> Self::Args<'a> {
        CommitArgs {
            future: twins.in_place(args.future),
            predicates: twins.in_place(args.predicates),
            logical: twins.in_place(args.logical),
            sampling: twins.in_place(args.sampling),
            verified: twins.in_place(args.verified),
            penalties: twins.in_place(args.penalties),
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
fn check_type(operator: &'static str, name: &str, view: &Operand, expected: DType) -> Result<()> {
    if view.dtype != expected {
        return Err(Error::argument(
            operator,
            format!(
                "{name} are {}, the descriptor says {}",
                abi::name(view.dtype),
                abi::name(expected)
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use platform::DevicePtr;

    const SLOTS: u64 = 4;
    const VOCAB: u64 = 8;
    const FUTURE: u64 = 16;

    fn view(dtype: DType, shape: &[u64], strides: &[u64]) -> Operand {
        Operand::with_strides(DevicePtr::from_address(0x1000), dtype, shape, strides)
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
            SlotCommit::check(descriptor, args).expect_err("the launch should have been refused")
        )
    }

    #[test]
    fn a_step_of_decoded_rows_passes_every_check() {
        SlotCommit::check(&descriptor(), &args()).expect("a plain step is valid");
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
                "{dtype:?}"
            );
        }
        let mut wide = args();
        wide.active = view(DType::U8, &[4], &[1]);
        SlotCommit::check(&descriptor(), &wide).expect("a u8 flag is an integer");
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
