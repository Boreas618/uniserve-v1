//! Committing a step's chosen tokens into the request slots' state.
//!
//! A slot's state is what the next step of its request reads on the device, with no host round
//! trip: its next input token, its continuation predicate, its logical and sampling positions,
//! its verified cache length and the counts its penalties read. [`slot_commit_`] folds one
//! step's rows into that state, in place.
//!
//! ```text
//! decode    verified += 1                      logical += 1                 sampling += 1
//! verify    verified  = visible + points       logical  = host + points     sampling = host + points
//! explicit  verified  = host cache length      logical  = host              sampling = host
//! cache     verified  = host cache length      (a row that sampled nothing moves nothing else)
//! count     nothing but the count below
//! sampled   future[slot, 0] = token;  predicates[slot] = continuation
//!           counts[slot, token] += 1           when the row counts and its task is valid and active
//! ```
//!
//! The table's fields and modes are `proto::module::commit`'s. Two entries make a row inert: a
//! row whose task is negative sampled nothing and installs no token, position or count, and a
//! row whose slot is negative has no destination and stores nothing at all. A table padded
//! with rows of both changes nothing, which is what lets a forward that commits nothing make
//! the same launch as one that does.

use ops::commit::{CommitArgs, CommitDescriptor, SlotCommit};
use proto::module::commit::{COMMIT_FIELDS, column, mode};
use tensor::{DType, Device, Tensor};

use super::nvfp4::integer;
use super::sampling::{require, signed};
use super::{Fused, compute};
use crate::capability::NO_SLOT;
use crate::error::{Error, Result};

/// The bits of a task's token that are the token: bit 31 carries nothing a slot installs.
const TOKEN_MASK: u32 = (1 << 31) - 1;

/// `state`, a contiguous CPU tensor of slot state, as one vector over the same storage: what
/// an in-place write through the view changes is the state's.
fn flat(owner: &'static str, name: &str, state: &Tensor) -> Result<Tensor> {
    if !state.is_contiguous() {
        return Err(Error::shape(
            owner,
            format!(
                "the {name} is a strided view of {:?}: the commit writes a slot's state where it \
                 lies",
                state.dims()
            ),
        ));
    }
    Ok(state.reshape(state.elem_count())?)
}

/// Folds the rows of `table` into the slots' state, in place.
///
/// - `table`: `[COMMIT_FIELDS, rows]` I32, field-major: each row's slot, mode, task, host cache
///   length, host logical and sampling positions, verification base and whether it counts
///   (`proto::module::commit::column`). It may be the leading rows of a wider table: the
///   launch then covers the wider table's rows, of which it commits these;
/// - `tokens`: `[tasks]` U32, the token each task chose. A value at or above the vocabulary is
///   no token (an [`argmax`](super::argmax) over a row with no numeric entry returns one) and
///   installs no input token, no predicate and no count;
/// - `valid`, `active`: `[tasks]` I32 or U8 flags; a task counts when both are set;
/// - `continuation`: `[tasks]` U8, whether the task's request continues;
/// - `points`: `[tasks]` I32, the length of the span a verification accepted;
/// - `future`: `[slots, width]` U32, whose column 0 receives a slot's next input token;
/// - `predicates`: `[slots]` U8, which receives the continuation;
/// - `logical`, `sampling`, `verified`: `[slots]` I32, a slot's two positions and its verified
///   cache length;
/// - `counts`: `[slots, vocabulary]` I32, how often each slot has seen each token.
///
/// A row reads its task's entries of the task columns and writes its slot's entries of the
/// state (module documentation). The rows that write one slot's state are one per step; the
/// count is an accumulation, so several rows may count into one slot.
///
/// # Errors
///
/// When a tensor is not the shape or element type stated above, a state tensor is a strided
/// view, or the operator refuses its arguments.
#[expect(
    clippy::too_many_arguments,
    reason = "the operands of the commit kernel, in its order"
)]
pub fn slot_commit_(
    table: &Tensor,
    tokens: &Tensor,
    valid: &Tensor,
    active: &Tensor,
    continuation: &Tensor,
    points: &Tensor,
    future: &Tensor,
    predicates: &Tensor,
    logical: &Tensor,
    sampling: &Tensor,
    verified: &Tensor,
    counts: &Tensor,
) -> Result<()> {
    const OWNER: &str = SlotCommitCall::OWNER;

    require(
        OWNER,
        "commit table",
        table,
        DType::I32,
        &[Some(COMMIT_FIELDS), None],
    )?;
    let rows = table.dims()[1];
    if rows == 0 {
        return Err(Error::shape(
            OWNER,
            "a step of no rows commits nothing".to_owned(),
        ));
    }

    let tasks = tokens.elem_count();
    require(OWNER, "tokens", tokens, DType::U32, &[Some(tasks)])?;
    for (name, flags) in [("valid flags", valid), ("active flags", active)] {
        if flags.dims() != [tasks] || !matches!(flags.dtype(), DType::I32 | DType::U8) {
            return Err(Error::shape(
                OWNER,
                format!(
                    "the {name} are {} {:?}, i32 or u8 [{tasks}] expected",
                    ops::abi::name(flags.dtype()),
                    flags.dims()
                ),
            ));
        }
    }
    require(
        OWNER,
        "continuation",
        continuation,
        DType::U8,
        &[Some(tasks)],
    )?;
    require(OWNER, "points", points, DType::I32, &[Some(tasks)])?;

    require(
        OWNER,
        "next input tokens",
        future,
        DType::U32,
        &[None, None],
    )?;
    let (slots, width) = (future.dims()[0], future.dims()[1]);
    require(OWNER, "predicates", predicates, DType::U8, &[Some(slots)])?;
    for (name, state) in [
        ("logical positions", logical),
        ("sampling positions", sampling),
        ("verified lengths", verified),
    ] {
        require(OWNER, name, state, DType::I32, &[Some(slots)])?;
    }
    require(OWNER, "counts", counts, DType::I32, &[Some(slots), None])?;
    let vocabulary = u32::try_from(counts.dims()[1])
        .map_err(|_| Error::shape(OWNER, "a vocabulary wider than a kernel takes"))?;

    compute(SlotCommitCall {
        table,
        tokens,
        valid,
        active,
        continuation,
        points,
        future,
        predicates,
        logical,
        sampling,
        verified,
        counts,
        rows,
        width,
        vocabulary,
    })
}

/// A checked call of [`slot_commit_`]: a step of `rows` rows, next input tokens `width` wide
/// and counts over `vocabulary` tokens.
struct SlotCommitCall<'t> {
    table: &'t Tensor,
    tokens: &'t Tensor,
    valid: &'t Tensor,
    active: &'t Tensor,
    continuation: &'t Tensor,
    points: &'t Tensor,
    future: &'t Tensor,
    predicates: &'t Tensor,
    logical: &'t Tensor,
    sampling: &'t Tensor,
    verified: &'t Tensor,
    counts: &'t Tensor,
    rows: usize,
    width: usize,
    vocabulary: u32,
}

impl SlotCommitCall<'_> {
    const OWNER: &'static str = "slot_commit_";
}

impl Fused for SlotCommitCall<'_> {
    type Output = ();

    fn device(&self) -> &Device {
        self.table.device()
    }

    fn cpu(self) -> Result<()> {
        let Self {
            table,
            tokens,
            valid,
            active,
            continuation,
            points,
            future,
            predicates,
            logical,
            sampling,
            verified,
            counts,
            rows,
            width,
            vocabulary,
        } = self;

        let field = |index: usize| -> Result<Tensor> {
            Ok(table.narrow(0, index, 1)?.squeeze(0)?.contiguous()?)
        };
        let (slot, row_mode, task) = (
            field(column::SLOT)?,
            field(column::MODE)?,
            field(column::TASK)?,
        );
        let zero = slot.zeros_like()?;
        let no_index = Tensor::full(NO_SLOT, rows, &Device::Cpu)?;

        // 1. What each row sampled: its task's entries, and nothing for a row without a task.
        let sampled = task.ge(0_i32)?;
        let selected = sampled.where_cond(&task.view(DType::U32)?, &no_index.zeros_like()?)?;
        let of_task = |entries: &Tensor| entries.index_select(&selected, 0);

        let token = of_task(tokens)?.broadcast_bitwise_and(&integer(TOKEN_MASK, tokens)?)?;
        let token_valid = token.broadcast_lt(&integer(vocabulary, &token)?)?;
        let accepted = sampled.where_cond(&of_task(points)?, &zero)?;
        let weight = sampled
            .mul(&of_task(valid)?.ne(0_u8)?)?
            .mul(&of_task(active)?.ne(0_u8)?)?;

        // 2. The lengths each row leaves its slot with.
        let dedicated = slot.ge(0_i32)?;
        let slot_index = dedicated.where_cond(&slot.view(DType::U32)?, &no_index.zeros_like()?)?;
        let of_slot = |state: &Tensor| -> Result<Tensor> {
            Ok(dedicated.where_cond(&state.index_select(&slot_index, 0)?, &zero)?)
        };

        // A decode advances what the slot holds by one, a verification moves to its base plus
        // the span it accepted, and every other row states the host's value.
        let decode = row_mode.eq(mode::DECODE as i32)?;
        let verify = row_mode.eq(mode::VERIFY as i32)?;
        let one = Tensor::new(1_i32, &Device::Cpu)?;
        let settled = |state: &Tensor, base: usize, host: usize| -> Result<Tensor> {
            let spanned = verify.where_cond(&(field(base)? + &accepted)?, &field(host)?)?;

            Ok(decode.where_cond(&of_slot(state)?.broadcast_add(&one)?, &spanned)?)
        };
        let cache = settled(verified, column::VISIBLE, column::CACHE_LENGTH)?;
        let position = settled(logical, column::LOGICAL, column::LOGICAL)?;
        let sampling_position = settled(sampling, column::SAMPLING, column::SAMPLING)?;

        // 3. The stores, each under the rows that make it. A row the mask leaves out names no
        //    slot, which the scatter skips.
        let states = dedicated.mul(&row_mode.ne(mode::COUNT as i32)?)?;
        let moves = sampled.mul(&states)?;
        let installs = moves.mul(&token_valid)?;
        let at = |mask: &Tensor| mask.where_cond(&slot_index, &no_index);

        flat(Self::OWNER, "verified lengths", verified)?.scatter_set(&at(&states)?, &cache, 0)?;
        flat(Self::OWNER, "logical positions", logical)?.scatter_set(&at(&moves)?, &position, 0)?;
        flat(Self::OWNER, "sampling positions", sampling)?.scatter_set(
            &at(&moves)?,
            &sampling_position,
            0,
        )?;
        flat(Self::OWNER, "predicates", predicates)?.scatter_set(
            &at(&installs)?,
            &of_task(continuation)?,
            0,
        )?;

        // Column 0 of the slot's row of next input tokens.
        let first_column = slot_index.broadcast_mul(&integer(width as u32, &slot_index)?)?;
        flat(Self::OWNER, "next input tokens", future)?.scatter_set(
            &installs.where_cond(&first_column, &no_index)?,
            &token,
            0,
        )?;

        // 4. The counts: an accumulation, so two rows of one slot and token both count.
        let counted = sampled
            .mul(&dedicated)?
            .mul(&token_valid)?
            .mul(&field(column::PENALTIES)?.ne(0_i32)?)?;
        let cell = slot_index
            .to_dtype(DType::I64)?
            .broadcast_mul(&Tensor::new(i64::from(vocabulary), &Device::Cpu)?)?
            .broadcast_add(&token.to_dtype(DType::I64)?)?;
        let no_cell = Tensor::full(i64::MAX, rows, &Device::Cpu)?;
        flat(Self::OWNER, "counts", counts)?.scatter_add_set(
            &counted.where_cond(&cell, &no_cell)?,
            &weight.to_dtype(DType::I32)?,
            0,
        )?;
        Ok(())
    }

    fn launch(self) -> Result<()> {
        let Self {
            table,
            tokens,
            valid,
            active,
            continuation,
            points,
            future,
            predicates,
            logical,
            sampling,
            verified,
            counts,
            rows,
            vocabulary,
            ..
        } = self;

        // The fields of the table lie `stride` elements apart: the stride follows the table the
        // rows were taken from, so every step of a runner states one key.
        ops::launch::<SlotCommit>(
            &CommitDescriptor::new(
                table.stride()[0] as u64,
                future.stride()[0] as u64,
                vocabulary,
            )?,
            CommitArgs {
                columns: ops::operand(table)?,
                tokens: signed(tokens)?,
                valid: ops::operand(valid)?,
                active: ops::operand(active)?,
                continuation: ops::operand(continuation)?,
                points: ops::operand(points)?,
                future: signed(future)?,
                predicates: ops::operand(predicates)?,
                logical: ops::operand(logical)?,
                sampling: ops::operand(sampling)?,
                verified: ops::operand(verified)?,
                penalties: ops::operand(counts)?,
                count: rows as u64,
            },
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tensor::WithDType;
    use tensor::shape::ShapeWithOneHole;

    use super::*;

    const SLOTS: usize = 3;
    const VOCABULARY: usize = 8;

    /// The CPU tensor of `values` in `shape`.
    fn tensor<T: WithDType>(values: &[T], shape: impl ShapeWithOneHole) -> Tensor {
        Tensor::from_slice(values, shape, &Device::Cpu).unwrap()
    }

    /// The elements of `tensor`, in row-major order.
    fn read<T: WithDType>(tensor: &Tensor) -> Vec<T> {
        tensor.flatten_all().unwrap().to_vec1().unwrap()
    }

    /// One row of a commit table, by field: the fields a test does not state are zero.
    fn row(slot: i32, mode: i64, task: i32) -> [i32; COMMIT_FIELDS] {
        let mut fields = [0; COMMIT_FIELDS];
        fields[column::SLOT] = slot;
        fields[column::MODE] = mode as i32;
        fields[column::TASK] = task;
        fields
    }

    /// The field-major `[COMMIT_FIELDS, rows]` table of `rows`.
    fn table(rows: &[[i32; COMMIT_FIELDS]]) -> Tensor {
        let fields: Vec<i32> = (0..COMMIT_FIELDS)
            .flat_map(|field| rows.iter().map(move |row| row[field]))
            .collect();

        tensor(&fields, (COMMIT_FIELDS, rows.len()))
    }

    /// The state of three slots before a step, as `(future, predicates, logical, sampling,
    /// verified, counts)`: every slot's next input token 9, and positions that tell the slots
    /// apart.
    fn state() -> [Tensor; 6] {
        [
            tensor(&[9_u32; SLOTS], (SLOTS, 1)),
            tensor(&[0_u8; SLOTS], SLOTS),
            tensor(&[10_i32, 20, 30], SLOTS),
            tensor(&[1_i32, 2, 3], SLOTS),
            tensor(&[10_i32, 20, 30], SLOTS),
            tensor(&[0_i32; SLOTS * VOCABULARY], (SLOTS, VOCABULARY)),
        ]
    }

    /// Commits `rows` over `tokens`, every task valid and active, into `state`.
    fn commit(
        rows: &[[i32; COMMIT_FIELDS]],
        tokens: &[u32],
        continuation: &[u8],
        points: &[i32],
        state: &[Tensor; 6],
    ) {
        let [future, predicates, logical, sampling, verified, counts] = state;
        let set = vec![1_i32; tokens.len()];

        slot_commit_(
            &table(rows),
            &tensor(tokens, tokens.len()),
            &tensor(&set, tokens.len()),
            &tensor(&set, tokens.len()),
            &tensor(continuation, tokens.len()),
            &tensor(points, tokens.len()),
            future,
            predicates,
            logical,
            sampling,
            verified,
            counts,
        )
        .unwrap();
    }

    /// The counts of `slot` after a step.
    fn counts_of(state: &[Tensor; 6], slot: usize) -> Vec<i32> {
        read::<i32>(&state[5])[slot * VOCABULARY..(slot + 1) * VOCABULARY].to_vec()
    }

    #[test]
    fn a_decode_row_installs_its_token_and_advances_its_slot_by_one() {
        let state = state();
        let mut counting = row(2, mode::DECODE, 0);
        counting[column::PENALTIES] = 1;

        commit(
            // Slot 2 counts its token, slot 0 does not, and the last row is padding.
            &[counting, row(0, mode::DECODE, 1), row(-1, mode::DECODE, -1)],
            &[5, 3, 7],
            &[1, 0, 1],
            &[0, 0, 0],
            &state,
        );

        let [future, predicates, logical, sampling, verified, _] = &state;
        assert_eq!(read::<u32>(future), [3, 9, 5]);
        assert_eq!(read::<u8>(predicates), [0, 0, 1]);
        assert_eq!(read::<i32>(logical), [11, 20, 31]);
        assert_eq!(read::<i32>(sampling), [2, 2, 4]);
        assert_eq!(read::<i32>(verified), [11, 20, 31]);

        let mut counted = vec![0; VOCABULARY];
        counted[5] = 1;
        assert_eq!(counts_of(&state, 2), counted);
        assert_eq!(counts_of(&state, 0), vec![0; VOCABULARY]);
        assert_eq!(counts_of(&state, 1), vec![0; VOCABULARY]);
    }

    #[test]
    fn each_mode_leaves_its_slot_with_the_lengths_it_states() {
        let state = state();

        // Slot 0 installs a prefill's token at the host's coordinates.
        let mut explicit = row(0, mode::EXPLICIT, 0);
        explicit[column::CACHE_LENGTH] = 7;
        explicit[column::LOGICAL] = 8;
        explicit[column::SAMPLING] = 9;

        // Slot 1 moves to its verification's base plus the three tokens it accepted, and
        // counts one of the drafts before the token it installs.
        let mut verify = row(1, mode::VERIFY, 1);
        verify[column::LOGICAL] = 100;
        verify[column::SAMPLING] = 50;
        verify[column::VISIBLE] = 40;
        let mut count = row(1, mode::COUNT, 2);
        count[column::PENALTIES] = 1;

        // Slot 2 states its cache length and samples nothing.
        let mut cache = row(2, mode::CACHE, -1);
        cache[column::CACHE_LENGTH] = 12;

        commit(
            &[explicit, verify, count, cache],
            &[4, 6, 2],
            &[1, 1, 0],
            &[0, 3, 0],
            &state,
        );

        let [future, predicates, logical, sampling, verified, _] = &state;
        assert_eq!(read::<u32>(future), [4, 6, 9]);
        assert_eq!(read::<u8>(predicates), [1, 1, 0]);
        assert_eq!(read::<i32>(logical), [8, 103, 30]);
        assert_eq!(read::<i32>(sampling), [9, 53, 3]);
        assert_eq!(read::<i32>(verified), [7, 43, 12]);

        // Only the count row asked for a count.
        let mut counted = vec![0; VOCABULARY];
        counted[2] = 1;
        assert_eq!(counts_of(&state, 1), counted);
        assert_eq!(counts_of(&state, 0), vec![0; VOCABULARY]);
    }

    #[test]
    fn a_value_that_is_no_token_installs_nothing_and_counts_accumulate() {
        let state = state();
        let counting = |slot: i32, mode: i64, task: i32| {
            let mut fields = row(slot, mode, task);
            fields[column::PENALTIES] = 1;
            fields
        };

        commit(
            &[
                // An argmax over a row with no numeric entry returns the vocabulary itself.
                counting(0, mode::DECODE, 0),
                // Two rows count one token into one slot.
                counting(1, mode::COUNT, 1),
                counting(1, mode::COUNT, 1),
            ],
            &[VOCABULARY as u32, 6],
            &[1, 1],
            &[0, 0],
            &state,
        );

        // Slot 0's step advanced, and it holds the input token and the counts it had.
        let [future, predicates, logical, ..] = &state;
        assert_eq!(read::<u32>(future), [9, 9, 9]);
        assert_eq!(read::<u8>(predicates), [0, 0, 0]);
        assert_eq!(read::<i32>(logical), [11, 20, 30]);
        assert_eq!(counts_of(&state, 0), vec![0; VOCABULARY]);

        let mut counted = vec![0; VOCABULARY];
        counted[6] = 2;
        assert_eq!(counts_of(&state, 1), counted);
    }

    #[test]
    fn a_task_that_is_not_valid_and_active_installs_its_token_without_counting_it() {
        let state = state();
        let [future, predicates, logical, sampling, verified, counts] = &state;
        let mut counting = row(1, mode::DECODE, 0);
        counting[column::PENALTIES] = 1;

        slot_commit_(
            &table(&[counting]),
            &tensor(&[5_u32], 1),
            &tensor(&[1_u8], 1),
            &tensor(&[0_u8], 1),
            &tensor(&[1_u8], 1),
            &tensor(&[0_i32], 1),
            future,
            predicates,
            logical,
            sampling,
            verified,
            counts,
        )
        .unwrap();

        assert_eq!(read::<u32>(future), [9, 5, 9]);
        assert_eq!(counts_of(&state, 1), vec![0; VOCABULARY]);
    }

    /// The rows a step commits are the leading rows of the table its runner holds: the key
    /// names that table's rows, so every step of the runner states one key.
    #[test]
    fn the_commit_states_its_kernel_by_the_table_its_rows_lead() {
        let meta = |shape: tensor::Shape, dtype: DType| {
            ops::retained(1 << 40, shape, dtype, &ops::meta()).unwrap()
        };
        let context = ops::Context::meta(None);
        context.record_launches();

        {
            let _guard = context.enter();
            let table = meta((COMMIT_FIELDS, 8).into(), DType::I32);

            slot_commit_(
                &table.narrow(1, 0, 3).unwrap(),
                &meta(3.into(), DType::U32),
                &meta(3.into(), DType::I32),
                &meta(3.into(), DType::I32),
                &meta(3.into(), DType::U8),
                &meta(3.into(), DType::I32),
                &meta((4, 1).into(), DType::U32),
                &meta(4.into(), DType::U8),
                &meta(4.into(), DType::I32),
                &meta(4.into(), DType::I32),
                &meta(4.into(), DType::I32),
                &meta((4, 100).into(), DType::I32),
            )
            .unwrap();
        }

        let launches = context.take_launches();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].0, "slot_commit");
        assert_eq!(
            launches[0].1,
            "index_i32_verified_i32_penalties_i32_stride_8_future_stride_1_vocab_100_block_8"
        );
    }
}
