//! Committing a sampling step's accepted tokens into the request slots' state.
//!
//! Once a step's tokens are chosen, one launch folds them into the slots: the next input token, the
//! continuation and stop predicate, the three lengths a slot tracks, and the penalty counts that bias
//! later sampling. The state is the executor's — `ops` names it and writes it, allocates nothing and
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
//! row counts penalties — and a row with no sampled task (`task < 0`) keeps its token state. Slots are
//! unique within a step, so each row's reads and writes of its slot need no ordering; the penalty
//! accumulation is an atomic add, as an indexed accumulation is.

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
    /// Element type of the column table, the tokens, the continuation and the state's lengths.
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
        if args.count == 0 {
            return Err(Error::argument(
                Self::NAME,
                "a step of zero rows commits nothing".to_owned(),
            ));
        }
        if descriptor.block == 0 || !descriptor.block.is_power_of_two() {
            return Err(Error::argument(
                Self::NAME,
                format!("the {} lane block is not a power of two", descriptor.block),
            ));
        }
        if u64::from(descriptor.block) < args.count {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "a block of {} lanes cannot cover {} rows",
                    descriptor.block, args.count
                ),
            ));
        }
        args.columns.require_rank(Self::NAME, 2)?;
        if args.columns.shape.dim(0) < COMMIT_FIELDS as u64 {
            return Err(Error::argument(
                Self::NAME,
                format!(
                    "the commit table has {} columns, a row is {COMMIT_FIELDS}",
                    args.columns.shape.dim(0)
                ),
            ));
        }
        check_type(
            Self::NAME,
            "the commit table",
            &args.columns,
            descriptor.index,
        )?;
        if args.columns.strides.get(1) != 1 || args.columns.strides.get(0) != descriptor.stride {
            return Err(Error::argument(
                Self::NAME,
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
                Self::NAME,
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
            view.require_rank(Self::NAME, 1)?;
        }
        for (name, view) in [
            ("the tokens", &args.tokens),
            ("the points", &args.points),
            ("the logical positions", &args.logical),
            ("the sampling positions", &args.sampling),
        ] {
            check_type(Self::NAME, name, view, descriptor.index)?;
        }
        for (name, view) in [
            ("the continuation", &args.continuation),
            ("the predicates", &args.predicates),
        ] {
            check_type(Self::NAME, name, view, DType::U8)?;
        }
        check_type(
            Self::NAME,
            "the verified lengths",
            &args.verified,
            descriptor.verified,
        )?;
        check_type(
            Self::NAME,
            "the penalty counts",
            &args.penalties,
            descriptor.penalties,
        )?;
        args.predicates.require_rank(Self::NAME, 1)?;
        args.logical.require_rank(Self::NAME, 1)?;
        args.sampling.require_rank(Self::NAME, 1)?;
        args.verified.require_rank(Self::NAME, 1)?;
        args.future.require_rank(Self::NAME, 2)?;
        check_type(
            Self::NAME,
            "the next-input tokens",
            &args.future,
            descriptor.index,
        )?;
        if args.future.strides.get(0) != descriptor.future_stride {
            return Err(Error::argument(
                Self::NAME,
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
                    Self::NAME,
                    format!(
                        "the {name} hold {} slots, the next-input matrix {slots}",
                        view.shape.dim(0)
                    ),
                ));
            }
        }
        args.penalties.require_rank(Self::NAME, 2)?;
        if args.penalties.shape.dim(0) < slots
            || args.penalties.shape.dim(1) != u64::from(descriptor.vocab)
        {
            return Err(Error::argument(
                Self::NAME,
                format!("the penalty counts are not [{slots}, {}]", descriptor.vocab),
            ));
        }

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

    fn shadow<'a>(args: Self::Args<'a>, twins: &'a mut Twins, stream: &Stream) -> Self::Args<'a> {
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
