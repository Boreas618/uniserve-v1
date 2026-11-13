//! Sampling: the token selection that ends a causal forward.
//!
//! [`Sampler`] composes the sampling functions of `nn::functional` into the selection of one
//! token per row, inside a bucket's captured graph: a request's logit adjustments and penalties,
//! its candidate filtering, its draw, and the log-probabilities reported for the chosen token.
//! It takes and returns tensors and creates no context, stream or memory: a runner lends it the
//! two per-slot tables, gives each forward its input tensors, and reads the tensors a forward
//! returns after the replay.
//!
//! ## The two selections
//!
//! A drawn token is a fixed sequence of launches over one row per sequence being sampled
//! ([`Sampler::forward`]):
//!
//! ```text
//! logits [rows, vocabulary]  → logits_shape      → shaped [rows, vocabulary] f32
//! shaped                     → top_k             → values [rows, width] f32, ids [rows, width]
//! values + ids               → filter_candidates → values (dropped = -inf), valid [rows]
//! values + ids               → candidate_order   → the same rows in ascending token id
//! ordered values             → softmax           → probabilities [rows, width] f32
//! probabilities + draws      → inverse_cdf       → tokens [rows]
//! ```
//!
//! A greedy token is another sequence ([`Sampler::greedy`]), which is why a bucket is one or
//! the other:
//!
//! ```text
//! logits [rows, vocabulary]  → logits_shape      → shaped [rows, vocabulary] f32
//! shaped                     → argmax            → tokens [rows]
//! ```
//!
//! ## Reported log-probabilities
//!
//! [`Sampler::score`] reports one token per row on the model's raw distribution: `log_softmax`
//! of the logits before the adjustments, the penalties, the temperature and the filtering,
//! which is vLLM's default (`logprobs_mode = raw_logprobs`). A logit bias, an allowed set or a
//! banned token therefore changes which token is chosen and not the score reported for it.
//!
//! ```text
//! logits                     → log_softmax       → logprobs [rows, vocabulary] f32
//! logprobs + tokens          → logprob_gather    → logprob [rows] f32, rank [rows] i32
//! logprobs                   → top_logprobs ×2   → top [rows, n] f32, ids [rows, n]   (n > 0)
//! ```
//!
//! The same forward scores a chosen token and a prompt token: the tokens are the ones a
//! selection returned, or the targets a batch names for its prompt's positions. `n` is the
//! deployment's `max_logprobs`; a request takes the first of them it asks for, and a captured
//! graph computes them for every row because whether a row's request asked is not a value it
//! can branch on.
//!
//! ## Logit adjustments
//!
//! `logits_shape` applies a row's adjustments before its penalties, in vLLM's order: its slot's
//! row of the bias table (the request's logit bias, `-inf` outside its allowed set) when the
//! row's flag names it, then `-inf` at every token its ban mask sets (the last token of a bad
//! word whose prefix the history ends with, the end-of-sequence tokens under a `min_tokens`
//! floor). Both selections shape through it, so a greedy row's argmax is over the adjusted
//! logits too. A row with no adjustment is shaped bit for bit as without them, by the same
//! launch.
//!
//! ## The draws are inputs
//!
//! The sampler never generates randomness. A row's draw is one FP32 value in `[0, 1)` that the
//! runner computes from the row's [`DrawAddress`] ([`draws`]): the request's Philox stream and
//! the position of the token being sampled, so a restored or replayed step draws the same
//! number, and so does every rank of a group.
//!
//! ## Greedy rows
//!
//! A row whose temperature is zero asks for its largest shaped logit. The shaper divides such
//! a row by one, so its shaped logits keep the logits' order. A bucket whose rows are all
//! greedy takes [`Sampler::greedy`]. A greedy row beside a drawn row is served by the draw:
//! [`parameters`] encodes it with a top-k width of one, so its candidate set is its largest
//! shaped logit and its draw, whatever the uniform, selects it. The two selections agree on
//! ties, because `top_k` orders equal scores by ascending token id and `argmax` returns the
//! lowest column. The one tie they order differently is between the two zeros: `top_k` orders
//! bit patterns and puts `+0.0` above `-0.0`, where `argmax` compares them equal.
//!
//! ## Rows that cannot be sampled
//!
//! `filter_candidates` reports, per row, whether the row's kept candidates form a usable
//! distribution: a row keeps nothing when its min-p is above one or when every logit is `-inf`
//! (an empty allowed set, every token banned). The sampler does not branch on it, since the
//! value is device data inside a captured graph: [`Sampler::forward`] returns the flags beside
//! the tokens, and such a row's token is `0`. A greedy row with no numeric logit returns the
//! vocabulary's width, which is no token. Whether such a request continues or fails is its
//! reader's decision.
//!
//! ## Slot state
//!
//! The penalties read how often a request has seen each token, and the bias table holds a
//! request's logit bias and allowed set. Both are one row per request slot in memory the
//! runner owns. A request that takes a slot starts both rows: [`Sampler::count_row`] and
//! [`Sampler::bias_row`] give the rows' values, which the runner copies over the slot's row
//! before the step that reads them. [`Sampler::commit_`] then counts each chosen token into
//! its slot as it installs it, so the next step's penalties read it without a host round trip.
//!
//! ## Host columns
//!
//! What a forward reads of a batch reaches it as input tensors the runner fills. The values of
//! those tensors are plain functions of the batch's columns, stated here beside the forwards
//! that read them: [`parameters`], [`draws`], [`adjustments`], [`commit_table`] and
//! [`inert_commit`]. [`scored`] is the other direction: a row of a forward's scores as a
//! completion reports it.

use ipc::philox::DrawAddress;
use nn::{DType, NO_SLOT, Tensor, functional};
use ops::sampler::ban_words;
use proto::module::commit::{COMMIT_FIELDS, column};
use proto::module::sampler::{PARAMETER_COLUMNS, TEMPERATURE, TOP_K};

use crate::error::{Error, Result};

/// What a deployment fixes of a sampler.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SamplerConfig {
    /// The width of a logit row.
    pub vocabulary: u32,
    /// The request slots the two per-slot tables hold a row for.
    pub slots: u32,
    /// The candidates a drawn row keeps: at most the vocabulary. A request's top-k is served
    /// only up to this width and a request that states none draws from this many, so a
    /// selection over a wider candidate list is the forward of a sampler of that width over
    /// the same two tables.
    pub width: u32,
    /// The most likely tokens a scored row reports beside its own: the deployment's
    /// `max_logprobs`, and zero to report none. The selection that finds them bounds the count
    /// by the vocabulary (215 over 151936 tokens), and a count past the bound is refused by
    /// the first forward that scores, which a profile run makes at load.
    pub candidates: u32,
    /// The token columns a row counts besides its slot's counts, as a verification counts the
    /// drafts before the position it samples; zero for rows that count none.
    pub extras: u32,
}

/// The token selection of a causal language model.
///
/// It holds the two per-slot tables every selection reads and every commit advances, as
/// tensors over memory its runner owns: external device tensors when serving, meta tensors for
/// a profile run, CPU tensors on the host. Both are `[slots, vocabulary]`, so a row's slot
/// names its request's row of each.
pub struct Sampler {
    config: SamplerConfig,
    /// `[slots, vocabulary]` I32: how often each slot's request has seen each token.
    counts: Tensor,
    /// `[slots, vocabulary]` FP32: each slot's additive row, its request's logit bias with
    /// `-inf` outside an allowed set.
    bias: Tensor,
}

impl Sampler {
    /// A sampler of `config` over the count table `counts` and the bias table `bias`.
    ///
    /// The caller keeps the tables' memory valid, and writes it only between forwards, for as
    /// long as the sampler or a graph captured over it exists. A slot's rows hold what its
    /// previous request left until the caller starts them ([`Self::count_row`],
    /// [`Self::bias_row`]).
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for an empty vocabulary or no slots, a candidate width that is not
    /// between one and the vocabulary, more reported candidates than the vocabulary has
    /// tokens, or a table that is not `[slots, vocabulary]` in its element type.
    pub fn new(config: SamplerConfig, counts: Tensor, bias: Tensor) -> Result<Self> {
        let SamplerConfig {
            vocabulary,
            slots,
            width,
            candidates,
            ..
        } = config;

        if vocabulary == 0 || slots == 0 {
            return Err(Error::Invalid(format!(
                "a sampler of {vocabulary} tokens over {slots} slots has nothing to select"
            )));
        }
        if width == 0 || width > vocabulary {
            return Err(Error::Invalid(format!(
                "a candidate width of {width} is outside 1..={vocabulary}"
            )));
        }
        if candidates > vocabulary {
            return Err(Error::Invalid(format!(
                "{candidates} reported candidates of a vocabulary of {vocabulary}"
            )));
        }

        let shape = [slots as usize, vocabulary as usize];
        for (name, table, dtype) in [("count", &counts, DType::I32), ("bias", &bias, DType::F32)] {
            if table.dims() != shape || table.dtype() != dtype {
                return Err(Error::Invalid(format!(
                    "the {name} table is {:?} {:?}, {dtype:?} {shape:?} expected",
                    table.dtype(),
                    table.dims()
                )));
            }
        }

        Ok(Self {
            config,
            counts,
            bias,
        })
    }

    /// The sampler's config.
    pub const fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Draws one token per row: `(tokens, valid)`, `[rows]` U32 and `[rows]` I32.
    ///
    /// - `logits`: `[rows, vocabulary]` FP32, as the head returns them;
    /// - `slots`: `[rows]` U32, each row's request slot, [`NO_SLOT`] for a padding row;
    /// - `parameters`: `[rows, PARAMETER_COLUMNS]` FP32, as [`parameters`] encodes them;
    /// - `draws`: `[rows]` FP32 uniform values in `[0, 1)`, as [`draws`] computes them;
    /// - `adjustments`, `bans`: `[rows, 2]` U32 and `[masks, ceil(vocabulary / 32)]` U32, as
    ///   [`adjustments`] encodes them;
    /// - `extras`: `[rows, extras]` U32, the tokens each row counts besides its slot's, for a
    ///   sampler whose config states some.
    ///
    /// A row draws from the candidates its own top-k, nucleus and min-p keep of its `width`
    /// largest shaped logits, in ascending token id, so a draw selects the same token whatever
    /// order equal scores were kept in. `valid` is `1` for a row whose candidates form a
    /// distribution; a row whose flag is `0` holds token `0` and its reader does not use it.
    ///
    /// # Errors
    ///
    /// A column that is not the shape or element type stated above, extra tokens the config
    /// does not state (or the reverse), or an operator's refusal.
    #[expect(
        clippy::too_many_arguments,
        reason = "the input columns of one selection, as the shaper and the draw read them"
    )]
    pub fn forward(
        &self,
        logits: &Tensor,
        slots: &Tensor,
        parameters: &Tensor,
        draws: &Tensor,
        adjustments: &Tensor,
        bans: &Tensor,
        extras: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        // The shaped rows are read by the selection of the candidates and by nothing after it.
        let (values, ids) = {
            let shaped = self.shape(logits, slots, parameters, adjustments, bans, extras)?;
            functional::top_k(&shaped, self.config.width)?
        };
        let (values, valid) = functional::filter_candidates_(values, &ids, parameters)?;

        // Token order, so the draw does not depend on the order the selection kept equal
        // scores in.
        let (values, ids) = functional::candidate_order(&values, &ids, self.config.vocabulary)?;
        let probabilities = functional::softmax(&values)?;
        let tokens = functional::inverse_cdf(&probabilities, &ids, draws)?;

        Ok((tokens, valid))
    }

    /// Takes each row's largest shaped logit as its token, the lowest id among equal ones:
    /// `[rows]` U32.
    ///
    /// The columns are [`Self::forward`]'s without the draws. The shaped logits are the
    /// adjusted and penalized ones, so a greedy request honours its logit bias, its allowed
    /// set and its banned tokens, as vLLM's greedy sampler does. A row with no numeric shaped
    /// logit returns the vocabulary's width, which is no token; its reader refuses it.
    ///
    /// A bucket takes this selection only when every row it serves is greedy: an argmax is not
    /// what a row that states a temperature asked for.
    ///
    /// # Errors
    ///
    /// As [`Self::forward`].
    pub fn greedy(
        &self,
        logits: &Tensor,
        slots: &Tensor,
        parameters: &Tensor,
        adjustments: &Tensor,
        bans: &Tensor,
        extras: Option<&Tensor>,
    ) -> Result<Tensor> {
        let shaped = self.shape(logits, slots, parameters, adjustments, bans, extras)?;

        Ok(functional::argmax(&shaped)?)
    }

    /// Scores one token per row on the raw distribution of `logits`: `[logprobs, ranks]`,
    /// `[rows]` FP32 and `[rows]` I32, then `[top_logprobs, top_ids]`, `[rows, candidates]`
    /// FP32 and U32, when the config reports candidates.
    ///
    /// `logits` is `[rows, vocabulary]` FP32 and `tokens` `[rows]` U32: the tokens a selection
    /// chose, or the prompt tokens a batch scores, one per position, with [`NO_SLOT`] for a
    /// position that scores none. A row's rank is one plus the tokens strictly more likely
    /// than its own, and its candidates are its most likely tokens, the largest first and
    /// equal ones in ascending id. A row of [`NO_SLOT`] reports `0.0` and rank `0`; its
    /// candidates are computed all the same and its reader skips them ([`scored`]).
    ///
    /// The scores are those of the raw distribution (module documentation), so they do not
    /// depend on the row's slot, parameters or adjustments.
    ///
    /// # Errors
    ///
    /// Columns of another shape or element type, more reported candidates than one selection
    /// holds for the vocabulary, or an operator's refusal.
    pub fn score(&self, logits: &Tensor, tokens: &Tensor) -> Result<Vec<Tensor>> {
        let logprobs = functional::log_softmax(logits)?;
        let (logprob, rank) = functional::logprob_gather(&logprobs, tokens)?;

        let mut scores = vec![logprob, rank];
        if self.config.candidates > 0 {
            let (values, ids) = functional::top_logprobs(&logprobs, self.config.candidates)?;
            scores.extend([values, ids]);
        }
        Ok(scores)
    }

    /// Commits a step's rows into the slots' state, in place: each row's chosen token becomes
    /// its slot's next input token and joins the slot's counts, and the slot's positions
    /// advance as the row's mode states (`nn::functional::slot_commit_`).
    ///
    /// - `table`: `[COMMIT_FIELDS, rows]` I32, as [`commit_table`] encodes it: the leading
    ///   rows of the table the runner holds;
    /// - `tokens`: `[tasks]` U32, the tokens a selection returned;
    /// - `valid`, `active`: `[tasks]` I32 or U8, whether each task's token counts;
    /// - `continuation`, `points`: `[tasks]` U8 and I32, whether each task's request
    ///   continues and the length of the span a verification accepted;
    /// - `future`, `predicates`, `logical`, `sampling`, `verified`: the runner's per-slot
    ///   state, `[slots, width]` U32, `[slots]` U8 and three `[slots]` I32.
    ///
    /// The counts the commit advances are this sampler's, so the next selection of a slot
    /// reads the tokens committed into it. A row whose slot and task are negative commits
    /// nothing ([`inert_commit`]), and a value that is no token of the vocabulary installs
    /// nothing.
    ///
    /// # Errors
    ///
    /// Columns or state of another shape or element type, or the operator's refusal.
    #[expect(
        clippy::too_many_arguments,
        reason = "the columns and the slot state of the commit kernel, in its order"
    )]
    pub fn commit_(
        &self,
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
    ) -> Result<()> {
        Ok(functional::slot_commit_(
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
            &self.counts,
        )?)
    }

    /// The count row a request starts its slot with: `[vocabulary]` I32, one count per
    /// occurrence of each token of its history.
    ///
    /// These are the counts the commits of those tokens leave on a slot that held the request
    /// from its first token, so a request that generated tokens elsewhere, before a transfer,
    /// a migration or a preemption, is penalized as if it had not moved. A token outside the
    /// vocabulary counts nothing. The runner copies the row over row `slot` of the count table
    /// before the step that reads it; a request with no history starts at zeros.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a slot the tables hold no row for.
    pub fn count_row(&self, slot: u32, history: &[u32]) -> Result<Vec<i32>> {
        self.check_slot("a new slot", slot)?;

        let mut row = vec![0_i32; self.config.vocabulary as usize];
        for &token in history {
            if let Some(count) = row.get_mut(token as usize) {
                *count += 1;
            }
        }
        Ok(row)
    }

    /// The bias row a request installs for its slot: `[vocabulary]` FP32, the bias of each
    /// biased token, `-inf` at every token outside its allowed set, and zero elsewhere.
    ///
    /// An allowed set outranks a bias: a token the request may not sample stays `-inf`
    /// whatever bias it states. A token outside the vocabulary is ignored. `bias` and
    /// `allowed` are the request's (`ipc::batch::NewSlot`); the runner copies the row over row
    /// `slot` of the bias table before the step whose rows are flagged to read it.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a slot the tables hold no row for.
    pub fn bias_row(
        &self,
        slot: u32,
        bias: &[(u32, f32)],
        allowed: Option<&[u32]>,
    ) -> Result<Vec<f32>> {
        self.check_slot("a bias row", slot)?;

        let vocabulary = self.config.vocabulary as usize;
        let mut row = match allowed {
            Some(allowed) => {
                let mut row = vec![f32::NEG_INFINITY; vocabulary];
                for &token in allowed {
                    if let Some(entry) = row.get_mut(token as usize) {
                        *entry = 0.0;
                    }
                }
                row
            }
            None => vec![0.0; vocabulary],
        };

        for &(token, bias) in bias {
            if let Some(entry) = row.get_mut(token as usize)
                && *entry != f32::NEG_INFINITY
            {
                *entry = bias;
            }
        }
        Ok(row)
    }

    /// Refuses `slot`, named by `what`, when the tables hold no row for it.
    fn check_slot(&self, what: &str, slot: u32) -> Result<()> {
        if slot >= self.config.slots {
            return Err(Error::Invalid(format!(
                "{what} names slot {slot} of tables of {} slots",
                self.config.slots
            )));
        }
        Ok(())
    }

    /// The shaped logits of both selections: the rows' adjustments, penalties and temperature
    /// over this sampler's two tables.
    fn shape(
        &self,
        logits: &Tensor,
        slots: &Tensor,
        parameters: &Tensor,
        adjustments: &Tensor,
        bans: &Tensor,
        extras: Option<&Tensor>,
    ) -> Result<Tensor> {
        let stated = extras.map_or(0, |extras| extras.dims().last().copied().unwrap_or(0));
        if stated != self.config.extras as usize {
            return Err(Error::Invalid(format!(
                "the sampler counts {} extra tokens a row and the batch states {stated}",
                self.config.extras
            )));
        }

        Ok(functional::logits_shape(
            logits,
            &self.counts,
            slots,
            parameters,
            extras,
            &self.bias,
            adjustments,
            bans,
        )?)
    }
}

/// The parameter column of a bucket of `bucket_rows` rows: `[bucket_rows,
/// PARAMETER_COLUMNS]` FP32, row-major, from a batch's `columns` (`ipc::batch::Columns::parameters`),
/// one row of [`PARAMETER_COLUMNS`] per row of the batch.
///
/// A greedy row, one whose temperature is zero, is given a top-k width of one: a greedy
/// selection reads no width, and a drawing selection then draws the row from its one largest
/// candidate, which is the token the greedy selection chooses (module documentation). The rows
/// the batch does not fill are zeros: a greedy row over a padding slot, whose token nothing
/// reads.
///
/// # Errors
///
/// [`Error::Invalid`] when the batch's column is not whole rows, or holds more rows than the
/// bucket.
pub fn parameters(columns: &[f32], bucket_rows: usize) -> Result<Vec<f32>> {
    let width = PARAMETER_COLUMNS as usize;
    if !columns.len().is_multiple_of(width) || columns.len() > bucket_rows * width {
        return Err(Error::Invalid(format!(
            "{} sampling parameters for a bucket of {bucket_rows} rows of {width}",
            columns.len()
        )));
    }

    let mut encoded = vec![0.0; bucket_rows * width];
    encoded[..columns.len()].copy_from_slice(columns);
    for row in encoded[..columns.len()].chunks_exact_mut(width) {
        if row[TEMPERATURE as usize] == 0.0 {
            row[TOP_K as usize] = 1.0;
        }
    }
    Ok(encoded)
}

/// The draw column of a bucket of `bucket_rows` rows: `[bucket_rows]` FP32, each row's uniform
/// draw in `[0, 1)` at its address (`ipc::philox`), and zero for the rows the batch does not
/// fill.
///
/// A draw is a function of its request's stream and the position of the token it samples, and
/// of nothing else: the same address gives the same draw in any batch, on any rank and after a
/// restart.
///
/// # Errors
///
/// [`Error::Invalid`] when the batch states more addresses than the bucket has rows.
pub fn draws(addresses: &[DrawAddress], bucket_rows: usize) -> Result<Vec<f32>> {
    if addresses.len() > bucket_rows {
        return Err(Error::Invalid(format!(
            "{} draw addresses for a bucket of {bucket_rows} rows",
            addresses.len()
        )));
    }

    let mut encoded = vec![0.0; bucket_rows];
    for (draw, address) in encoded.iter_mut().zip(addresses) {
        *draw = address.uniform();
    }
    Ok(encoded)
}

/// The adjustment columns of a bucket of `bucket_rows` rows over a `vocabulary`-wide row:
/// `(rows, masks)`, `[bucket_rows, 2]` U32 and `[masks, ceil(vocabulary / 32)]` U32, both
/// row-major, from a batch's `biased` and `banned` columns (`ipc::batch::Columns`), each empty
/// or one entry per row.
///
/// A row's two entries are its bias flag, `1` when the row adds its slot's bias row, and the
/// index of its ban mask, [`NO_SLOT`] for a row that bans nothing. The masks are one per row
/// that bans a token, in row order, so a batch whose rows ban nothing encodes none; the runner
/// writes them at the head of its mask tensor and leaves the rest as it is, since nothing
/// reads a mask no row names. Bit `t mod 32` of word `t / 32` of a mask is set for a banned
/// token `t`; a banned token outside the vocabulary bans nothing. The rows the batch does not
/// fill add nothing and ban nothing.
///
/// # Errors
///
/// [`Error::Invalid`] when the batch states flags or banned tokens for more rows than the
/// bucket has.
pub fn adjustments(
    biased: &[bool],
    banned: &[Vec<u32>],
    bucket_rows: usize,
    vocabulary: u32,
) -> Result<(Vec<u32>, Vec<u32>)> {
    if biased.len() > bucket_rows || banned.len() > bucket_rows {
        return Err(Error::Invalid(format!(
            "adjustments of {} flagged and {} banning rows for a bucket of {bucket_rows} rows",
            biased.len(),
            banned.len()
        )));
    }
    let words = ban_words(vocabulary) as usize;

    let mut rows = Vec::with_capacity(2 * bucket_rows);
    let mut masks = Vec::new();
    for row in 0..bucket_rows {
        let flagged = biased.get(row).copied().unwrap_or(false);
        rows.push(u32::from(flagged));

        let tokens = banned.get(row).map_or(&[][..], Vec::as_slice);
        if tokens.is_empty() {
            rows.push(NO_SLOT);
            continue;
        }

        let index = masks.len() / words;
        rows.push(index as u32);
        masks.resize(masks.len() + words, 0_u32);
        let mask = &mut masks[index * words..];
        for &token in tokens.iter().filter(|token| **token < vocabulary) {
            mask[(token / 32) as usize] |= 1 << (token % 32);
        }
    }
    Ok((rows, masks))
}

/// The commit table of a runner whose table holds `capacity` rows: `[COMMIT_FIELDS, capacity]`
/// I32, field-major, from a batch's `[COMMIT_FIELDS, rows]` I64 table
/// (`ipc::batch::Columns::commit`).
///
/// The batch's rows fill the table's leading rows and every row past them is inert, slot and
/// task `-1`, rather than zero: a zero slot is slot 0, a real destination, so a commit over
/// the whole table would otherwise install a token there.
///
/// # Errors
///
/// [`Error::Invariant`] when the batch's table is not [`COMMIT_FIELDS`] fields of `rows` rows
/// or holds more rows than the runner's; [`Error::Invalid`] for a value outside I32.
pub fn commit_table(commit: &[i64], rows: usize, capacity: usize) -> Result<Vec<i32>> {
    if commit.len() != COMMIT_FIELDS * rows || rows > capacity {
        return Err(Error::Invariant(format!(
            "a commit table of {} values for {COMMIT_FIELDS} fields of {rows} rows in a table \
             of {capacity}",
            commit.len()
        )));
    }

    let mut table = vec![0_i32; COMMIT_FIELDS * capacity];
    for row in rows..capacity {
        table[column::SLOT * capacity + row] = -1;
        table[column::TASK * capacity + row] = -1;
    }
    for field in 0..COMMIT_FIELDS {
        for row in 0..rows {
            let value = commit[field * rows + row];
            table[field * capacity + row] = i32::try_from(value).map_err(|_| {
                Error::Invalid(format!("a commit table value {value} does not fit i32"))
            })?;
        }
    }
    Ok(table)
}

/// The commit table of a batch that commits nothing: `[COMMIT_FIELDS, rows]` I64, field-major,
/// every row's slot and task `-1` and every other field zero.
///
/// A row with a negative slot has no destination and a row with a negative task sampled
/// nothing, so a commit over this table stores nothing while its launch is still made: the
/// graph a warm-up captures with it is the graph a serving batch replays.
pub fn inert_commit(rows: usize) -> Vec<i64> {
    let mut table = vec![0_i64; COMMIT_FIELDS * rows];
    for row in 0..rows {
        table[column::SLOT * rows + row] = -1;
        table[column::TASK * rows + row] = -1;
    }
    table
}

/// One scored token as a completion reports it: the token with its log-probability and rank,
/// then its row's candidates, the largest first.
///
/// `values` and `ids` are the row's candidates as [`Sampler::score`] returned them. A
/// candidate's rank is its competition rank among them, one plus the candidates strictly above
/// it, which is its rank in the row, since every token above a kept candidate is kept before
/// it. A candidate whose id is no token of the vocabulary, as in a row with fewer tokens than
/// candidates, is not reported. The token itself may be among its candidates; the scheduler
/// keeps the candidates each request asks for and drops that repetition.
pub fn scored(
    token: u32,
    logprob: f32,
    rank: u32,
    values: &[f32],
    ids: &[u32],
    vocabulary: u32,
) -> Vec<ipc::TokenLogprob> {
    let mut entries = Vec::with_capacity(1 + values.len());
    entries.push(ipc::TokenLogprob {
        token_id: token,
        logprob,
        rank,
    });

    for (index, (&value, &id)) in values.iter().zip(ids).enumerate() {
        if id >= vocabulary {
            continue;
        }
        let above = values[..index]
            .iter()
            .filter(|other| **other > value)
            .count();

        entries.push(ipc::TokenLogprob {
            token_id: id,
            logprob: value,
            rank: above as u32 + 1,
        });
    }
    entries
}

#[cfg(test)]
mod tests {
    use ipc::philox::{request_sampling_key, sampling_uniform};
    use nn::Device;
    use proto::module::commit::mode;
    use proto::module::sampler::TOP_P;
    use tensor::WithDType;
    use tensor::shape::ShapeWithOneHole;

    use super::*;

    /// The offsets of a row's min-p threshold and of its repetition, frequency and presence
    /// penalties, which `proto::module::sampler` states by position.
    const MIN_P: usize = 3;
    const PENALTIES: [usize; 3] = [4, 5, 6];

    /// Six logits whose order and gaps every feature below changes: token 2 is the largest and
    /// token 0 the next.
    const LOGITS: [f32; 6] = [1.0, 0.5, 2.0, -0.5, 0.0, -1.0];

    /// The probabilities the candidate filters are stated over, by token.
    const MASS: [f32; 6] = [0.05, 0.4, 0.1, 0.25, 0.15, 0.05];

    /// The CPU tensor of `values` in `shape`.
    fn tensor<T: WithDType>(values: &[T], shape: impl ShapeWithOneHole) -> Tensor {
        Tensor::from_slice(values, shape, &Device::Cpu).unwrap()
    }

    /// The elements of `tensor`, in row-major order.
    fn read<T: WithDType>(tensor: &Tensor) -> Vec<T> {
        tensor.flatten_all().unwrap().to_vec1().unwrap()
    }

    /// A count table and a bias table of zeros, as a runner holds them before any request.
    fn tables(vocabulary: usize, slots: usize) -> (Tensor, Tensor) {
        (
            Tensor::zeros((slots, vocabulary), DType::I32, &Device::Cpu).unwrap(),
            Tensor::zeros((slots, vocabulary), DType::F32, &Device::Cpu).unwrap(),
        )
    }

    /// A sampler over `tables` that keeps every token as a candidate.
    fn sampler(tables: &(Tensor, Tensor), candidates: u32) -> Sampler {
        let [slots, vocabulary] = *tables.0.dims() else {
            panic!("a table is [slots, vocabulary]");
        };
        let config = SamplerConfig {
            vocabulary: vocabulary as u32,
            slots: slots as u32,
            width: vocabulary as u32,
            candidates,
            extras: 0,
        };

        Sampler::new(config, tables.0.clone(), tables.1.clone()).unwrap()
    }

    /// Copies `row` over row `slot` of `table`, as a runner starts a slot's row.
    fn install<T: WithDType>(table: &Tensor, slot: u32, row: &[T]) {
        table
            .slice_set(&tensor(row, (1, row.len())), 0, slot as usize)
            .unwrap();
    }

    /// One request's row of parameters: no top-k, the whole nucleus, no min-p, no penalty.
    fn request(temperature: f32) -> [f32; PARAMETER_COLUMNS as usize] {
        let mut row = [0.0; PARAMETER_COLUMNS as usize];
        row[TEMPERATURE as usize] = temperature;
        row[TOP_P as usize] = 1.0;
        row[PENALTIES[0]] = 1.0;
        row
    }

    /// `request` with the parameter at `offset` set to `value`.
    fn with(
        mut request: [f32; PARAMETER_COLUMNS as usize],
        offset: usize,
        value: f32,
    ) -> [f32; PARAMETER_COLUMNS as usize] {
        request[offset] = value;
        request
    }

    /// The input columns of a batch of `requests` whose rows adjust nothing, as a runner
    /// encodes them: `(parameters, adjustments, bans)`.
    fn columns(
        requests: &[[f32; PARAMETER_COLUMNS as usize]],
        vocabulary: usize,
    ) -> (Tensor, Tensor, Tensor) {
        columns_adjusted(requests, vocabulary, &[], &[])
    }

    /// [`columns`] for a batch whose rows state the bias flags `biased` and the banned tokens
    /// `banned`.
    fn columns_adjusted(
        requests: &[[f32; PARAMETER_COLUMNS as usize]],
        vocabulary: usize,
        biased: &[bool],
        banned: &[Vec<u32>],
    ) -> (Tensor, Tensor, Tensor) {
        let rows = requests.len();
        let encoded = parameters(&requests.concat(), rows).unwrap();
        let (flags, masks) = adjustments(biased, banned, rows, vocabulary as u32).unwrap();

        // The runner's mask tensor holds a mask per row; the batch's masks are its head.
        let words = ban_words(vocabulary as u32) as usize;
        let mut bans = vec![0_u32; rows * words];
        bans[..masks.len()].copy_from_slice(&masks);

        (
            tensor(&encoded, (rows, PARAMETER_COLUMNS as usize)),
            tensor(&flags, (rows, 2)),
            tensor(&bans, (rows, words)),
        )
    }

    /// Draws one token per row of `logits`, every row on its slot of `slots`, for rows that
    /// adjust nothing: the tokens and their validity.
    fn draw(
        sampler: &Sampler,
        logits: &[&[f32]],
        slots: &[u32],
        requests: &[[f32; PARAMETER_COLUMNS as usize]],
        uniforms: &[f32],
    ) -> (Vec<u32>, Vec<i32>) {
        let rows = logits.len();
        let vocabulary = logits[0].len();
        let (parameters, adjustments, bans) = columns(requests, vocabulary);

        let (tokens, valid) = sampler
            .forward(
                &tensor(&logits.concat(), (rows, vocabulary)),
                &tensor(slots, rows),
                &parameters,
                &tensor(uniforms, rows),
                &adjustments,
                &bans,
                None,
            )
            .unwrap();
        (read(&tokens), read(&valid))
    }

    /// The greedy token of each row of `logits`, for rows that adjust nothing.
    fn greedy(
        sampler: &Sampler,
        logits: &[&[f32]],
        slots: &[u32],
        requests: &[[f32; PARAMETER_COLUMNS as usize]],
    ) -> Vec<u32> {
        let rows = logits.len();
        let vocabulary = logits[0].len();
        let (parameters, adjustments, bans) = columns(requests, vocabulary);

        let tokens = sampler
            .greedy(
                &tensor(&logits.concat(), (rows, vocabulary)),
                &tensor(slots, rows),
                &parameters,
                &adjustments,
                &bans,
                None,
            )
            .unwrap();
        read(&tokens)
    }

    // ---- The reference: each feature's rule over one row, in plain arithmetic -----------------

    /// The probability of each token of `logits`: their softmax, in FP64. A token of `-inf`
    /// has none.
    fn distribution(logits: &[f32]) -> Vec<f64> {
        let largest = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let terms: Vec<f64> = logits
            .iter()
            .map(|logit| f64::from(*logit - largest).exp())
            .collect();
        let total: f64 = terms.iter().sum();

        terms.iter().map(|term| term / total).collect()
    }

    /// The token a uniform `draw` selects from `mass`: the first, in token order, whose
    /// cumulative probability exceeds the draw.
    fn drawn(mass: &[f64], draw: f32) -> u32 {
        let mut cumulative = 0.0;
        let token = mass.iter().position(|mass| {
            cumulative += mass;
            cumulative > f64::from(draw)
        });

        token.expect("a draw below one is reached by a whole distribution") as u32
    }

    /// The token of the largest of `logits`, the lowest among equal ones.
    fn largest(logits: &[f32]) -> u32 {
        let mut best = 0;
        for (token, logit) in logits.iter().enumerate() {
            if *logit > logits[best] {
                best = token;
            }
        }
        best as u32
    }

    /// The log-probability and the competition rank of `token` in the raw distribution of
    /// `logits`.
    fn raw_score(logits: &[f32], token: u32) -> (f64, i32) {
        let mass: f64 = logits.iter().map(|logit| f64::from(*logit).exp()).sum();
        let own = logits[token as usize];
        let above = logits.iter().filter(|logit| **logit > own).count();

        (f64::from(own) - mass.ln(), above as i32 + 1)
    }

    /// vLLM's penalties over `logits` for a request that has seen each token `counts` times:
    /// the repetition penalty divides a positive logit and multiplies another, the frequency
    /// penalty is subtracted once per occurrence and the presence penalty once.
    fn penalized(logits: &[f32], counts: &[i32], penalties: [f32; 3]) -> Vec<f32> {
        let [repetition, frequency, presence] = penalties;

        logits
            .iter()
            .zip(counts)
            .map(|(logit, count)| {
                if *count == 0 {
                    return *logit;
                }
                let repeated = if *logit > 0.0 {
                    logit / repetition
                } else {
                    logit * repetition
                };
                repeated - frequency * *count as f32 - presence
            })
            .collect()
    }

    /// The tokens a row keeps of its distribution `mass` under a top-k width, a nucleus and a
    /// min-p threshold, as the probabilities it then draws from.
    fn kept(mass: &[f32], top_k: usize, top_p: f64, min_p: f64) -> Vec<f64> {
        let mut order: Vec<usize> = (0..mass.len()).collect();
        order.sort_by(|a, b| mass[*b].total_cmp(&mass[*a]).then(a.cmp(b)));
        if top_k > 0 {
            order.truncate(top_k);
        }

        // The nucleus: the most likely tokens until the mass before one is above `top_p`,
        // over the top-k's renormalized mass.
        let total: f64 = order.iter().map(|token| f64::from(mass[*token])).sum();
        let mut before = 0.0;
        order.retain(|token| {
            let keep = before <= top_p;
            before += f64::from(mass[*token]) / total;
            keep
        });

        // Min-p: at least that share of the most likely token's probability.
        let most = f64::from(mass[order[0]]);
        order.retain(|token| f64::from(mass[*token]) >= most * min_p);

        let total: f64 = order.iter().map(|token| f64::from(mass[*token])).sum();
        (0..mass.len())
            .map(|token| {
                if order.contains(&token) {
                    f64::from(mass[token]) / total
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// The logits of [`MASS`]: a row whose raw distribution is those probabilities.
    fn mass_logits() -> Vec<f32> {
        MASS.iter().map(|mass| mass.ln()).collect()
    }

    /// Draws [`MASS`] under `request` at each of `uniforms`, one row a draw.
    fn draw_mass(request: [f32; PARAMETER_COLUMNS as usize], uniforms: &[f32]) -> Vec<u32> {
        let tables = tables(MASS.len(), 1);
        let logits = mass_logits();
        let rows = uniforms.len();

        let (tokens, valid) = draw(
            &sampler(&tables, 0),
            &vec![&logits[..]; rows],
            &vec![0; rows],
            &vec![request; rows],
            uniforms,
        );
        assert_eq!(valid, vec![1; rows]);
        tokens
    }

    // ---- Greedy and temperature ---------------------------------------------------------------

    #[test]
    fn greedy_takes_the_largest_logit_and_the_lowest_token_among_equal_ones() {
        let tables = tables(6, 1);
        let rows: [&[f32]; 3] = [
            &LOGITS,
            // Three equal largest logits: the lowest token of them.
            &[0.5, 2.0, 2.0, -1.0, 2.0, 0.0],
            &[-3.0, -2.0, -1.0, -1.0, -5.0, -9.0],
        ];

        let tokens = greedy(&sampler(&tables, 0), &rows, &[0; 3], &[request(0.0); 3]);

        assert_eq!(tokens, rows.map(largest));
        assert_eq!(tokens, [2, 1, 2]);
    }

    #[test]
    fn temperature_divides_the_logits_a_row_draws_from() {
        let uniforms = [0.02_f32, 0.1, 0.3, 0.9];

        for temperature in [0.5_f32, 1.0, 2.0] {
            let tokens = draw_mass(request(temperature), &uniforms);

            let scaled: Vec<f32> = mass_logits()
                .iter()
                .map(|logit| logit / temperature)
                .collect();
            let expected = uniforms.map(|uniform| drawn(&distribution(&scaled), uniform));
            assert_eq!(tokens, expected, "temperature {temperature}");
        }

        // At one the row is drawn from its own probabilities, whose cumulative sums are 0.05,
        // 0.45, 0.55, 0.8, 0.95 and 1; at a half the likely tokens are likelier still.
        assert_eq!(draw_mass(request(1.0), &uniforms), [0, 1, 1, 4]);
        assert_eq!(draw_mass(request(0.5), &uniforms), [1, 1, 1, 3]);
    }

    // ---- Candidate filtering ------------------------------------------------------------------

    #[test]
    fn top_k_keeps_a_rows_largest_candidates() {
        let uniforms = [0.0_f32, 0.6, 0.7, 0.99];
        let tokens = draw_mass(with(request(1.0), TOP_K as usize, 2.0), &uniforms);

        let expected = uniforms.map(|uniform| drawn(&kept(&MASS, 2, 1.0, 0.0), uniform));
        assert_eq!(tokens, expected);

        // Tokens 1 and 3 hold 0.4 and 0.25: 8/13 and 5/13 of what is kept.
        assert_eq!(tokens, [1, 1, 3, 3]);
    }

    #[test]
    fn top_p_keeps_the_most_likely_candidates_that_hold_the_nucleus() {
        let uniforms = [0.0_f32, 0.4, 0.6, 0.9, 0.99];
        let tokens = draw_mass(with(request(1.0), TOP_P as usize, 0.7), &uniforms);

        let expected = uniforms.map(|uniform| drawn(&kept(&MASS, 0, 0.7, 0.0), uniform));
        assert_eq!(tokens, expected);

        // 0.4, 0.25 and 0.15 are kept: the mass before the next candidate is 0.8. They are a
        // half, 5/16 and 3/16 of what is kept, at tokens 1, 3 and 4.
        assert_eq!(tokens, [1, 1, 3, 4, 4]);
    }

    #[test]
    fn min_p_drops_the_candidates_below_a_share_of_the_most_likely() {
        let uniforms = [0.0_f32, 0.6, 0.7, 0.99];
        let tokens = draw_mass(with(request(1.0), MIN_P, 0.5), &uniforms);

        let expected = uniforms.map(|uniform| drawn(&kept(&MASS, 0, 1.0, 0.5), uniform));
        assert_eq!(tokens, expected);

        // Half of 0.4 is 0.2: tokens 1 and 3 stay.
        assert_eq!(tokens, [1, 1, 3, 3]);
    }

    #[test]
    fn the_three_filters_apply_in_the_order_top_k_nucleus_min_p() {
        let filtered = with(
            with(with(request(1.0), TOP_K as usize, 4.0), TOP_P as usize, 0.8),
            MIN_P,
            0.3,
        );
        let uniforms = [0.0_f32, 0.3, 0.6, 0.8, 0.99];
        let tokens = draw_mass(filtered, &uniforms);

        // Four candidates of 0.9 together; the nucleus keeps 0.4, 0.25 and 0.15, whose mass
        // before the fourth is 8/9; and 0.3 of 0.4 keeps all three.
        let expected = uniforms.map(|uniform| drawn(&kept(&MASS, 4, 0.8, 0.3), uniform));
        assert_eq!(tokens, expected);
        assert_eq!(tokens, [1, 1, 3, 3, 4]);
    }

    /// A request's top-k is served up to the sampler's candidate width, so a selection over a
    /// wider candidate list is the forward of a wider sampler over the same tables.
    #[test]
    fn a_wider_sampler_over_the_same_tables_keeps_more_candidates() {
        let tables = tables(MASS.len(), 1);
        let of_width = |width: u32| {
            let config = SamplerConfig {
                vocabulary: MASS.len() as u32,
                slots: 1,
                width,
                candidates: 0,
                extras: 0,
            };
            Sampler::new(config, tables.0.clone(), tables.1.clone()).unwrap()
        };
        let logits = mass_logits();
        let uniforms = [0.5_f32, 0.99];
        let select = |width: u32, top_k: f32| {
            let (tokens, _) = draw(
                &of_width(width),
                &[&logits[..]; 2],
                &[0; 2],
                &[with(request(1.0), TOP_K as usize, top_k); 2],
                &uniforms,
            );
            tokens
        };

        // A top-k of four over two candidates is the two, and over six it is the four; a
        // request that states no top-k draws from the width.
        let two = uniforms.map(|uniform| drawn(&kept(&MASS, 2, 1.0, 0.0), uniform));
        let four = uniforms.map(|uniform| drawn(&kept(&MASS, 4, 1.0, 0.0), uniform));
        assert_eq!(select(2, 4.0), two);
        assert_eq!(select(6, 4.0), four);
        assert_eq!(select(2, 0.0), two);
        assert_eq!((two, four), ([1, 3], [2, 4]));
    }

    // ---- Penalties and the count table --------------------------------------------------------

    #[test]
    fn penalties_read_the_counts_of_each_rows_slot() {
        let tables = tables(6, 2);
        let sampler = sampler(&tables, 0);

        // Slot 1's request arrives with a history; slot 0's has generated nothing.
        let counts = sampler.count_row(1, &[2, 2, 5]).unwrap();
        assert_eq!(counts, [0, 0, 2, 0, 0, 1]);
        install(&tables.0, 1, &counts);

        let cases = [
            // The same penalty on a slot that has seen nothing changes nothing.
            (0, [4.0, 0.0, 0.0]),
            // Repetition: token 2's 2.0 becomes 0.5 and token 5's -1.0 becomes -4.0.
            (1, [4.0, 0.0, 0.0]),
            // Frequency, per occurrence: 2.0 less 2 · 0.6 is below token 0's 1.0 ...
            (1, [1.0, 0.6, 0.0]),
            // ... and 2.0 less 2 · 0.4 is not.
            (1, [1.0, 0.4, 0.0]),
            // Presence, once: 2.0 less 1.5 is below 1.0, and 2.0 less 0.5 is not.
            (1, [1.0, 0.0, 1.5]),
            (1, [1.0, 0.0, 0.5]),
        ];
        let penalties = |penalties: [f32; 3], temperature: f32| {
            let mut row = request(temperature);
            for (offset, penalty) in PENALTIES.iter().zip(penalties) {
                row[*offset] = penalty;
            }
            row
        };
        let seen = |slot: u32| {
            if slot == 1 {
                counts.clone()
            } else {
                vec![0; 6]
            }
        };

        let tokens = greedy(
            &sampler,
            &[&LOGITS[..]; 6],
            &cases.map(|(slot, _)| slot),
            &cases.map(|(_, stated)| penalties(stated, 0.0)),
        );

        let expected =
            cases.map(|(slot, stated)| largest(&penalized(&LOGITS, &seen(slot), stated)));
        assert_eq!(tokens, expected);
        assert_eq!(tokens, [2, 0, 0, 2, 0, 2]);

        // All three at once, drawn: the row's distribution is that of its penalized logits.
        let stated = [1.5, 0.3, 0.2];
        let uniforms = [0.1_f32, 0.5, 0.9];
        let (tokens, _) = draw(
            &sampler,
            &[&LOGITS[..]; 3],
            &[1; 3],
            &[penalties(stated, 1.0); 3],
            &uniforms,
        );

        let mass = distribution(&penalized(&LOGITS, &counts, stated));
        assert_eq!(tokens, uniforms.map(|uniform| drawn(&mass, uniform)));
    }

    /// A verification's row counts the drafts before the position it samples besides its
    /// slot's own history.
    #[test]
    fn a_row_counts_its_extra_tokens_besides_its_slots() {
        let tables = tables(6, 1);
        let config = SamplerConfig {
            vocabulary: 6,
            slots: 1,
            width: 6,
            candidates: 0,
            extras: 2,
        };
        let sampler = Sampler::new(config, tables.0.clone(), tables.1.clone()).unwrap();
        let presence = with(request(0.0), PENALTIES[2], 1.5);
        let (parameters, adjustments, bans) = columns(&[presence; 2], 6);

        let select = |extras: Option<&Tensor>| {
            sampler.greedy(
                &tensor(&[LOGITS, LOGITS].concat(), (2, 6)),
                &tensor(&[0_u32, 0], 2),
                &parameters,
                &adjustments,
                &bans,
                extras,
            )
        };

        // The first row has drafted token 2, which its presence penalty then lowers below
        // token 0; the second has drafted nothing.
        let drafted = tensor(&[2, NO_SLOT, NO_SLOT, NO_SLOT], (2, 2));
        assert_eq!(read::<u32>(&select(Some(&drafted)).unwrap()), [0, 2]);

        // A sampler states how many extra tokens a row carries, and a batch carries as many.
        assert!(select(None).is_err());
        assert!(select(Some(&tensor(&[2_u32, 2], (2, 1)))).is_err());
    }

    // ---- Logit adjustments --------------------------------------------------------------------

    #[test]
    fn a_logit_bias_is_added_to_the_rows_flagged_to_read_their_slots_row() {
        let tables = tables(6, 2);
        let sampler = sampler(&tables, 0);

        let row = sampler.bias_row(1, &[(2, -4.0), (3, 3.0)], None).unwrap();
        assert_eq!(row, [0.0, 0.0, -4.0, 3.0, 0.0, 0.0]);
        install(&tables.1, 1, &row);

        // Slot 1 flagged, slot 1 not flagged, and slot 0 flagged, whose row is zeros.
        let (parameters, adjustments, bans) =
            columns_adjusted(&[request(0.0); 3], 6, &[true, false, true], &[]);
        let logits = tensor(&[LOGITS; 3].concat(), (3, 6));

        let tokens = sampler
            .greedy(
                &logits,
                &tensor(&[1_u32, 1, 0], 3),
                &parameters,
                &adjustments,
                &bans,
                None,
            )
            .unwrap();

        let biased: Vec<f32> = LOGITS
            .iter()
            .zip(&row)
            .map(|(logit, bias)| logit + bias)
            .collect();
        assert_eq!(
            read::<u32>(&tokens),
            [largest(&biased), largest(&LOGITS), largest(&LOGITS)]
        );
        assert_eq!(read::<u32>(&tokens), [3, 2, 2]);

        // The score reported for the biased row's token is the raw distribution's: token 3 is
        // the fifth most likely of the model's own logits.
        let scores = sampler.score(&logits, &tokens).unwrap();
        let (logprob, rank) = raw_score(&LOGITS, 3);
        assert!((f64::from(read::<f32>(&scores[0])[0]) - logprob).abs() < 1e-5);
        assert_eq!(read::<i32>(&scores[1]), [rank, 1, 1]);
        assert_eq!(rank, 5);
    }

    /// vLLM's order: the bias is added before the penalties, so the repetition penalty divides
    /// the biased logit.
    #[test]
    fn the_repetition_penalty_scales_a_biased_logit() {
        let tables = tables(6, 1);
        let sampler = sampler(&tables, 0);

        install(&tables.0, 0, &sampler.count_row(0, &[3]).unwrap());
        install(
            &tables.1,
            0,
            &sampler.bias_row(0, &[(3, 3.5)], None).unwrap(),
        );

        let repeated = with(request(0.0), PENALTIES[0], 2.0);
        let (parameters, adjustments, bans) = columns_adjusted(&[repeated], 6, &[true], &[]);

        let tokens = sampler
            .greedy(
                &tensor(&LOGITS, (1, 6)),
                &tensor(&[0_u32], 1),
                &parameters,
                &adjustments,
                &bans,
                None,
            )
            .unwrap();

        // Token 3: (-0.5 + 3.5) / 2 is 1.5, below token 2's 2.0. Penalized first it would be
        // -0.5 · 2 + 3.5, which is above it.
        assert_eq!(read::<u32>(&tokens), [2]);
    }

    #[test]
    fn an_allowed_set_restricts_what_a_row_selects() {
        let tables = tables(6, 1);
        let sampler = sampler(&tables, 0);

        // Tokens 1 and 2 are allowed; token 4's bias is of a token the request may not sample.
        let row = sampler
            .bias_row(0, &[(1, 2.0), (4, 9.0)], Some(&[1, 2]))
            .unwrap();
        let outside = f32::NEG_INFINITY;
        assert_eq!(row, [outside, 2.0, 0.0, outside, outside, outside]);
        install(&tables.1, 0, &row);

        let uniforms = [0.0_f32, 0.1, 0.7, 0.999];
        let (parameters, adjustments, bans) =
            columns_adjusted(&[request(1.0); 4], 6, &[true; 4], &[]);
        let logits = tensor(&[LOGITS; 4].concat(), (4, 6));
        let slots = tensor(&[0_u32; 4], 4);

        let (tokens, valid) = sampler
            .forward(
                &logits,
                &slots,
                &parameters,
                &tensor(&uniforms, 4),
                &adjustments,
                &bans,
                None,
            )
            .unwrap();

        // 0.5 + 2.0 at token 1 against 2.0 at token 2: e^2.5 and e^2 of their sum.
        let shaped: Vec<f32> = LOGITS
            .iter()
            .zip(&row)
            .map(|(logit, bias)| logit + bias)
            .collect();
        let mass = distribution(&shaped);
        assert_eq!(
            read::<u32>(&tokens),
            uniforms.map(|uniform| drawn(&mass, uniform))
        );
        assert_eq!(read::<u32>(&tokens), [1, 1, 2, 2]);
        assert_eq!(read::<i32>(&valid), [1; 4]);

        // A greedy row is restricted too.
        let greedy = sampler
            .greedy(&logits, &slots, &parameters, &adjustments, &bans, None)
            .unwrap();
        assert_eq!(read::<u32>(&greedy), [1; 4]);
    }

    #[test]
    fn banned_tokens_are_never_selected() {
        const VOCABULARY: usize = 40;

        let tables = tables(VOCABULARY, 1);
        let sampler = sampler(&tables, 0);

        // Token 35 is the most likely, then 3, then 20: one in each word of a ban mask.
        let mut logits: Vec<f32> = (0..VOCABULARY)
            .map(|token| (token % 7) as f32 * 0.25)
            .collect();
        logits[35] = 5.0;
        logits[3] = 4.0;
        logits[20] = 3.0;

        let banned = [vec![3, 35], Vec::new(), vec![35]];
        let (parameters, adjustments, bans) =
            columns_adjusted(&[request(0.0); 3], VOCABULARY, &[], &banned);
        let rows = tensor(&logits.repeat(3), (3, VOCABULARY));
        let slots = tensor(&[0_u32; 3], 3);

        let tokens = sampler
            .greedy(&rows, &slots, &parameters, &adjustments, &bans, None)
            .unwrap();
        assert_eq!(read::<u32>(&tokens), [20, 35, 3]);

        // Drawn, a row under the first mask draws from its logits without the two banned
        // tokens: every row here adds no bias row and names mask 0.
        let uniforms = [0.0_f32, 0.5, 0.9999];
        let (parameters, _, _) = columns_adjusted(&[request(1.0); 3], VOCABULARY, &[], &banned);
        let banning = tensor(&[0_u32; 6], (3, 2));
        let (tokens, _) = sampler
            .forward(
                &rows,
                &slots,
                &parameters,
                &tensor(&uniforms, 3),
                &banning,
                &bans,
                None,
            )
            .unwrap();

        let mut allowed = logits.clone();
        allowed[3] = f32::NEG_INFINITY;
        allowed[35] = f32::NEG_INFINITY;
        let mass = distribution(&allowed);
        assert_eq!(
            read::<u32>(&tokens),
            uniforms.map(|uniform| drawn(&mass, uniform))
        );
        assert!(
            read::<u32>(&tokens)
                .iter()
                .all(|token| ![3, 35].contains(token))
        );
    }

    // ---- Reported log-probabilities -----------------------------------------------------------

    #[test]
    fn a_token_is_scored_on_the_raw_distribution_with_its_rank_and_the_rows_candidates() {
        let tables = tables(6, 1);
        let sampler = sampler(&tables, 3);

        // Tokens 1 and 3 are equally likely.
        let logits = [1.0_f32, 0.5, 2.0, 0.5, 0.0, -1.0];
        let tokens = [1_u32, 3, 2, 5];
        let scores = sampler
            .score(&tensor(&logits.repeat(4), (4, 6)), &tensor(&tokens, 4))
            .unwrap();
        let [logprobs, ranks, top_values, top_ids] = &scores[..] else {
            panic!("a sampler that reports candidates returns four tensors");
        };

        let expected = tokens.map(|token| raw_score(&logits, token));
        for (actual, (logprob, _)) in read::<f32>(logprobs).iter().zip(expected) {
            assert!((f64::from(*actual) - logprob).abs() < 1e-5);
        }

        // A competition rank: the two equal tokens share the third.
        assert_eq!(read::<i32>(ranks), expected.map(|(_, rank)| rank));
        assert_eq!(read::<i32>(ranks), [3, 3, 1, 6]);

        // Every row's three most likely tokens, the largest first and equal ones by id.
        assert_eq!(read::<u32>(top_ids), [2, 0, 1].repeat(4));
        for (actual, token) in read::<f32>(top_values).iter().zip([2, 0, 1].repeat(4)) {
            assert!((f64::from(*actual) - raw_score(&logits, token).0).abs() < 1e-5);
        }

        // What a completion reports of the first row: its token, then its candidates.
        let values = read::<f32>(top_values);
        let reported = scored(
            1,
            read::<f32>(logprobs)[0],
            3,
            &values[..3],
            &read::<u32>(top_ids)[..3],
            6,
        );
        let listed: Vec<(u32, u32)> = reported
            .iter()
            .map(|entry| (entry.token_id, entry.rank))
            .collect();
        assert_eq!(listed, [(1, 3), (2, 1), (0, 2), (1, 3)]);

        // A sampler that reports no candidates returns the token's own score alone.
        let plain = self::sampler(&tables, 0)
            .score(&tensor(&logits, (1, 6)), &tensor(&[2_u32], 1))
            .unwrap();
        assert_eq!(plain.len(), 2);
    }

    #[test]
    fn prompt_tokens_are_scored_at_the_positions_a_batch_names() {
        let tables = tables(6, 1);
        let sampler = sampler(&tables, 0);

        // One row of logits per prompt position, and the prompt token each position scores:
        // the second scores none.
        let positions: [[f32; 6]; 4] = [
            LOGITS,
            [0.0, 3.0, 0.0, 0.0, 1.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [2.0, 0.5, 2.0, -0.5, 0.0, 4.0],
        ];
        let targets = [3, NO_SLOT, 0, 5];

        let scores = sampler
            .score(&tensor(&positions.concat(), (4, 6)), &tensor(&targets, 4))
            .unwrap();
        let (logprobs, ranks) = (read::<f32>(&scores[0]), read::<i32>(&scores[1]));

        for (position, target) in targets.iter().enumerate() {
            if *target == NO_SLOT {
                assert_eq!((logprobs[position], ranks[position]), (0.0, 0));
                continue;
            }
            let (logprob, rank) = raw_score(&positions[position], *target);
            assert!((f64::from(logprobs[position]) - logprob).abs() < 1e-5);
            assert_eq!(ranks[position], rank);
        }

        // Fifth of six, unscored, first among six equal tokens, and the most likely.
        assert_eq!(ranks, [5, 0, 1, 1]);
    }

    // ---- Draws --------------------------------------------------------------------------------

    #[test]
    fn a_draw_is_a_function_of_its_requests_seed_and_the_position_it_samples() {
        let stream = request_sampling_key(Some(7), 3, 11, 0);
        let address = |position: u64| DrawAddress {
            key: stream,
            position,
        };

        // Positions 12 and 13 of one request, then position 12 again, in a bucket of four.
        let uniforms = draws(&[address(12), address(13), address(12)], 4).unwrap();
        assert_eq!(
            uniforms,
            [
                sampling_uniform(stream, 12, 0, 0),
                sampling_uniform(stream, 13, 0, 0),
                sampling_uniform(stream, 12, 0, 0),
                0.0,
            ]
        );
        assert_ne!(uniforms[0], uniforms[1]);
        assert!(uniforms.iter().all(|uniform| (0.0..1.0).contains(uniform)));

        // The same seed draws the same value when the request is sent again as another one.
        let again = DrawAddress {
            key: request_sampling_key(Some(7), 5, 90, 2),
            position: 12,
        };
        assert_eq!(draws(&[again], 1).unwrap(), [uniforms[0]]);
        assert!(draws(&[again; 2], 1).is_err());

        // The token a row draws is the one its uniform selects, so the same address selects
        // the same token twice.
        let tokens = draw_mass(request(1.0), &uniforms[..3]);
        let mass = distribution(&mass_logits());
        assert_eq!(tokens, [0, 1, 2].map(|row| drawn(&mass, uniforms[row])));
        assert_eq!(tokens[0], tokens[2]);
    }

    #[test]
    fn a_greedy_row_beside_a_drawn_row_is_drawn_from_its_one_largest_candidate() {
        let tables = tables(6, 1);
        let sampler = sampler(&tables, 0);

        // Two greedy rows whose largest logit three tokens share, at both ends of the draw's
        // range, and a drawn row between them.
        let tied = [0.5_f32, 2.0, 2.0, -1.0, 2.0, 0.0];
        let logits = mass_logits();
        let rows: [&[f32]; 3] = [&tied, &logits, &tied];
        let requests = [request(0.0), request(1.0), request(0.0)];
        let uniforms = [0.0_f32, 0.5, 0.9999];

        let (tokens, valid) = draw(&sampler, &rows, &[0; 3], &requests, &uniforms);

        // The greedy rows' token is the one the greedy selection takes, whatever their draw.
        let greedy = greedy(&sampler, &[&tied], &[0], &[request(0.0)]);
        assert_eq!(greedy, [1]);
        assert_eq!(tokens, [1, drawn(&distribution(&logits), uniforms[1]), 1]);
        assert_eq!(valid, [1; 3]);
    }

    #[test]
    fn a_row_whose_candidates_are_all_dropped_is_reported_and_holds_token_zero() {
        let tables = tables(6, 2);
        let sampler = sampler(&tables, 0);

        // Slot 1's request may sample nothing at all.
        install(&tables.1, 1, &sampler.bias_row(1, &[], Some(&[])).unwrap());

        // A min-p above one keeps no candidate; the second row is an ordinary one.
        let requests = [with(request(1.0), MIN_P, 2.0), request(1.0), request(1.0)];
        let (parameters, adjustments, bans) =
            columns_adjusted(&requests, 6, &[false, false, true], &[]);
        let logits = tensor(&[LOGITS; 3].concat(), (3, 6));
        let slots = tensor(&[0_u32, 0, 1], 3);

        let (tokens, valid) = sampler
            .forward(
                &logits,
                &slots,
                &parameters,
                &tensor(&[0.5_f32; 3], 3),
                &adjustments,
                &bans,
                None,
            )
            .unwrap();

        assert_eq!(read::<i32>(&valid), [0, 1, 0]);
        let tokens = read::<u32>(&tokens);
        assert_eq!(tokens[1], drawn(&distribution(&LOGITS), 0.5));
        assert_eq!((tokens[0], tokens[2]), (0, 0));

        // A greedy row with no numeric logit returns the vocabulary's width.
        let greedy = sampler
            .greedy(&logits, &slots, &parameters, &adjustments, &bans, None)
            .unwrap();
        assert_eq!(read::<u32>(&greedy), [2, 2, 6]);
    }

    // ---- The commit ---------------------------------------------------------------------------

    #[test]
    fn a_commit_counts_each_chosen_token_into_its_slot() {
        let tables = tables(6, 2);
        let sampler = sampler(&tables, 0);

        // The runner's slot state: next input tokens, predicates, positions, verified lengths.
        let future = tensor(&[9_u32, 9], (2, 1));
        let predicates = tensor(&[0_u8, 0], 2);
        let logical = tensor(&[40_i32, 70], 2);
        let sampling = tensor(&[4_i32, 7], 2);
        let verified = tensor(&[40_i32, 70], 2);
        let commit = |table: &[i64], tokens: &Tensor| {
            let table = commit_table(table, 2, 4).unwrap();

            sampler
                .commit_(
                    &tensor(&table, (COMMIT_FIELDS, 4)),
                    tokens,
                    &tensor(&[1_i32, 1], 2),
                    &tensor(&[1_i32, 1], 2),
                    &tensor(&[1_u8, 1], 2),
                    &tensor(&[0_i32, 0], 2),
                    &future,
                    &predicates,
                    &logical,
                    &sampling,
                    &verified,
                )
                .unwrap();
        };

        // Both rows penalize a token they have seen by more than token 2 leads token 0.
        let presence = with(request(0.0), PENALTIES[2], 3.0);
        let (parameters, adjustments, bans) = columns(&[presence; 2], 6);
        let logits = tensor(&[LOGITS, LOGITS].concat(), (2, 6));
        let slots = tensor(&[1_u32, 0], 2);
        let select = || {
            sampler
                .greedy(&logits, &slots, &parameters, &adjustments, &bans, None)
                .unwrap()
        };

        let first = select();
        assert_eq!(read::<u32>(&first), [2, 2]);

        // The batch's table: row 0 decodes into slot 1 and counts its token, row 1 decodes
        // into slot 0 and does not.
        let mut table = vec![0_i64; COMMIT_FIELDS * 2];
        table[column::SLOT * 2..][..2].copy_from_slice(&[1, 0]);
        table[column::MODE * 2..][..2].copy_from_slice(&[mode::DECODE; 2]);
        table[column::TASK * 2..][..2].copy_from_slice(&[0, 1]);
        table[column::PENALTIES * 2..][..2].copy_from_slice(&[1, 0]);
        commit(&table, &first);

        assert_eq!(
            read::<i32>(&tables.0),
            [[0; 6], [0, 0, 1, 0, 0, 0]].concat()
        );
        assert_eq!(read::<u32>(&future), [2, 2]);
        assert_eq!(read::<u8>(&predicates), [1, 1]);
        assert_eq!(read::<i32>(&logical), [41, 71]);
        assert_eq!(read::<i32>(&sampling), [5, 8]);
        assert_eq!(read::<i32>(&verified), [41, 71]);

        // The next step of slot 1 is penalized for the token it committed; slot 0's is not.
        let second = select();
        assert_eq!(read::<u32>(&second), [0, 2]);

        // A table that commits nothing stores nothing.
        commit(&inert_commit(2), &second);
        assert_eq!(
            read::<i32>(&tables.0),
            [[0; 6], [0, 0, 1, 0, 0, 0]].concat()
        );
        assert_eq!(read::<u32>(&future), [2, 2]);
        assert_eq!(read::<i32>(&logical), [41, 71]);
    }

    // ---- Launches -----------------------------------------------------------------------------

    /// A meta tensor of `shape` elements of `dtype`: an input, or a table, of a profile run.
    fn meta(shape: impl Into<nn::Shape>, dtype: DType) -> Tensor {
        use std::sync::atomic::{AtomicU64, Ordering};

        static NEXT: AtomicU64 = AtomicU64::new(1 << 40);
        let address = NEXT.fetch_add(1 << 28, Ordering::Relaxed);

        backend::retained(address, shape, dtype, &backend::meta()).unwrap()
    }

    /// The same forwards over meta tensors are a profile run: each selection states its
    /// operators in launch order, with the keys of the config's widths.
    #[test]
    fn each_forward_states_its_launch_sequence_under_a_meta_context() {
        let config = SamplerConfig {
            vocabulary: 1000,
            slots: 8,
            width: 64,
            candidates: 5,
            extras: 0,
        };
        let sampler = Sampler::new(
            config,
            meta((8, 1000), DType::I32),
            meta((8, 1000), DType::F32),
        )
        .unwrap();

        let context = backend::Context::meta(None);
        let _guard = context.enter();
        let operators = |context: &backend::Context| -> Vec<&'static str> {
            context
                .take_launches()
                .into_iter()
                .map(|(operator, _, _)| operator)
                .collect()
        };

        let logits = meta((4, 1000), DType::F32);
        let slots = meta(4, DType::U32);
        let parameters = meta((4, PARAMETER_COLUMNS as usize), DType::F32);
        let adjustments = meta((4, 2), DType::U32);
        let bans = meta((4, ban_words(1000) as usize), DType::U32);

        context.record_launches();
        let (tokens, valid) = sampler
            .forward(
                &logits,
                &slots,
                &parameters,
                &meta(4, DType::F32),
                &adjustments,
                &bans,
                None,
            )
            .unwrap();
        assert_eq!((tokens.dims(), tokens.dtype()), (&[4][..], DType::U32));
        assert_eq!((valid.dims(), valid.dtype()), (&[4][..], DType::I32));

        let launches = context.take_launches();
        let stated: Vec<(&str, &str)> = launches
            .iter()
            .map(|(operator, key, _)| (*operator, key.as_str()))
            .collect();
        assert_eq!(
            stated,
            [
                (
                    "logits_shape",
                    "dtype_f32_vocabulary_1000_block_1024_extras_0_divide_true_pdl_false"
                ),
                ("top_k", "dtype_f32_tile_2048_ids_i32_pdl_false"),
                ("filter_candidates", "width_64_tile_8192_ids_i32_pdl_false"),
                ("candidate_order", "values_f32_ids_i32_pdl_false"),
                ("softmax", "dtype_f32_columns_64_tile_8192_pdl_false"),
                (
                    "inverse_cdf",
                    "width_64_tile_8192_ids_i32_tokens_i32_pdl_false"
                ),
            ]
        );

        context.record_launches();
        sampler
            .greedy(&logits, &slots, &parameters, &adjustments, &bans, None)
            .unwrap();
        assert_eq!(operators(&context), ["logits_shape", "argmax"]);

        context.record_launches();
        let scores = sampler.score(&logits, &tokens).unwrap();
        assert_eq!(
            operators(&context),
            [
                "log_softmax",
                "logprob_gather",
                "top_logprobs",
                "top_logprobs"
            ]
        );
        let shapes: Vec<&[usize]> = scores.iter().map(Tensor::dims).collect();
        assert_eq!(shapes, [&[4][..], &[4], &[4, 5], &[4, 5]]);

        context.record_launches();
        sampler
            .commit_(
                &meta((COMMIT_FIELDS, 16), DType::I32)
                    .narrow(1, 0, 4)
                    .unwrap(),
                &tokens,
                &meta(4, DType::I32),
                &meta(4, DType::I32),
                &meta(4, DType::U8),
                &meta(4, DType::I32),
                &meta((8, 1), DType::U32),
                &meta(8, DType::U8),
                &meta(8, DType::I32),
                &meta(8, DType::I32),
                &meta(8, DType::I32),
            )
            .unwrap();
        let launches = context.take_launches();
        assert_eq!(launches.len(), 1);
        assert_eq!(
            (launches[0].0, launches[0].1.as_str()),
            (
                "slot_commit",
                "index_i32_verified_i32_penalties_i32_stride_16_future_stride_1_vocab_1000_\
                 block_16"
            )
        );
    }

    // ---- Construction and the host encodings --------------------------------------------------

    #[test]
    fn a_sampler_is_built_for_tables_of_its_slots_and_vocabulary() {
        let config = SamplerConfig {
            vocabulary: 6,
            slots: 2,
            width: 6,
            candidates: 6,
            extras: 0,
        };
        let build = |config: SamplerConfig, tables: (Tensor, Tensor)| {
            Sampler::new(config, tables.0, tables.1)
        };
        let built = build(config, tables(6, 2)).unwrap();
        assert_eq!(*built.config(), config);

        // The width is how many candidates a row keeps: not zero, not more than the vocabulary;
        // and a row reports no more candidates than it has tokens.
        for refused in [
            SamplerConfig { width: 0, ..config },
            SamplerConfig { width: 7, ..config },
            SamplerConfig {
                candidates: 7,
                ..config
            },
            SamplerConfig { slots: 0, ..config },
            SamplerConfig {
                vocabulary: 0,
                ..config
            },
        ] {
            assert!(build(refused, tables(6, 2)).is_err(), "{refused:?}");
        }

        // The tables are one row per slot, of the vocabulary, counts and biases in that order.
        assert!(build(config, tables(6, 3)).is_err());
        assert!(build(config, tables(5, 2)).is_err());
        let (counts, bias) = tables(6, 2);
        assert!(Sampler::new(config, bias, counts).is_err());

        // A slot outside the tables starts no row.
        assert!(built.count_row(2, &[]).is_err());
        assert!(built.bias_row(2, &[], None).is_err());
    }

    #[test]
    fn a_slots_rows_start_from_its_requests_history_and_bias() {
        let tables = tables(8, 1);
        let sampler = sampler(&tables, 0);

        // One count per occurrence, and nothing for a token outside the vocabulary.
        assert_eq!(
            sampler.count_row(0, &[3, 1, 3, 9, 3]).unwrap(),
            [0, 1, 0, 3, 0, 0, 0, 0]
        );
        assert_eq!(sampler.count_row(0, &[]).unwrap(), [0; 8]);

        // A bias without an allowed set leaves every other token as it is.
        assert_eq!(
            sampler
                .bias_row(0, &[(2, -1.5), (7, 4.0), (8, 1.0)], None)
                .unwrap(),
            [0.0, 0.0, -1.5, 0.0, 0.0, 0.0, 0.0, 4.0]
        );
        assert_eq!(sampler.bias_row(0, &[], None).unwrap(), [0.0; 8]);
    }

    #[test]
    fn a_greedy_row_is_encoded_with_a_top_k_width_of_one() {
        let (temperature, top_k) = (TEMPERATURE as usize, TOP_K as usize);
        let width = PARAMETER_COLUMNS as usize;
        let greedy = with(request(0.0), top_k, 40.0);
        let sampled = with(request(0.8), top_k, 40.0);

        let encoded = parameters(&[greedy, sampled].concat(), 3).unwrap();

        // The greedy row keeps its one largest candidate, the drawn row the width it asked
        // for, and the row the batch does not fill is zeros.
        assert_eq!(encoded[top_k], 1.0);
        assert_eq!(encoded[width..2 * width], sampled);
        assert_eq!(encoded[width + temperature], 0.8);
        assert_eq!(encoded[2 * width..], [0.0; PARAMETER_COLUMNS as usize]);
        assert_eq!(encoded[..top_k], greedy[..top_k]);
        assert_eq!(encoded[top_k + 1..width], greedy[top_k + 1..]);

        // A column that is not whole rows, or holds more rows than the bucket.
        assert!(parameters(&greedy[..5], 3).is_err());
        assert!(parameters(&[greedy, sampled].concat(), 1).is_err());
    }

    /// One flag pair per row and one ban mask per row that bans a token: bit `t mod 32` of
    /// word `t / 32`; a row the batch does not fill, a token past the vocabulary and a row
    /// that bans nothing set nothing.
    #[test]
    fn a_batch_encodes_one_ban_mask_per_banning_row() {
        let biased = [true, false, true];
        let banned = [vec![0, 31, 99, 100], Vec::new(), vec![32]];

        let (rows, masks) = adjustments(&biased, &banned, 4, 100).unwrap();

        assert_eq!(rows, [1, 0, 0, NO_SLOT, 1, 1, 0, NO_SLOT]);
        // Tokens 0 and 31 share word 0, token 99 is bit 3 of word 3 and 100 is outside; token
        // 32 is bit 0 of word 1.
        assert_eq!(masks, [1 | 1 << 31, 0, 0, 1 << 3, 0, 1, 0, 0]);

        // A batch that adjusts nothing encodes no mask and flags no row.
        let (rows, masks) = adjustments(&[], &[], 4, 100).unwrap();
        assert!(masks.is_empty());
        assert_eq!(rows, [0, NO_SLOT].repeat(4));

        // A batch of more rows than the bucket is refused.
        assert!(adjustments(&biased, &banned, 2, 100).is_err());
    }

    #[test]
    fn a_commit_table_fills_the_runners_table_and_leaves_the_rest_inert() {
        // Two rows of a batch: slots 5 and 6, tasks 0 and 1, and a field per row to tell
        // them apart.
        let mut batch = vec![0_i64; COMMIT_FIELDS * 2];
        batch[column::SLOT * 2..][..2].copy_from_slice(&[5, 6]);
        batch[column::TASK * 2..][..2].copy_from_slice(&[0, 1]);
        batch[column::LOGICAL * 2..][..2].copy_from_slice(&[100, 200]);

        let table = commit_table(&batch, 2, 4).unwrap();

        let field = |index: usize| &table[index * 4..(index + 1) * 4];
        assert_eq!(field(column::SLOT), [5, 6, -1, -1]);
        assert_eq!(field(column::TASK), [0, 1, -1, -1]);
        assert_eq!(field(column::LOGICAL), [100, 200, 0, 0]);
        assert_eq!(field(column::MODE), [0; 4]);

        // A table that is not whole rows, more rows than the runner's table, a value past I32.
        assert!(commit_table(&batch[..15], 2, 4).is_err());
        assert!(commit_table(&batch, 2, 1).is_err());
        batch[column::LOGICAL * 2] = i64::from(i32::MAX) + 1;
        assert!(commit_table(&batch, 2, 4).is_err());

        // The table of a batch that commits nothing names no slot and no task.
        let inert = inert_commit(3);
        assert_eq!(inert.len(), COMMIT_FIELDS * 3);
        for (index, value) in inert.iter().enumerate() {
            let named = [column::SLOT, column::TASK].contains(&(index / 3));
            assert_eq!(*value, if named { -1 } else { 0 });
        }
    }

    /// A scored token is reported first, then its row's candidates in their order with their
    /// competition ranks, and a candidate the row had no token for is not reported.
    #[test]
    fn a_scored_token_reports_itself_then_its_ranked_candidates() {
        let entries = scored(
            7,
            -1.5,
            3,
            &[-0.25, -1.0, -1.0, -1.5, f32::NEG_INFINITY],
            &[2, 9, 11, 7, NO_SLOT],
            1_024,
        );

        let listed: Vec<(u32, f32, u32)> = entries
            .iter()
            .map(|entry| (entry.token_id, entry.logprob, entry.rank))
            .collect();
        assert_eq!(
            listed,
            [
                (7, -1.5, 3),
                (2, -0.25, 1),
                (9, -1.0, 2),
                (11, -1.0, 2),
                (7, -1.5, 4),
            ]
        );

        // Without candidates only the token is reported.
        assert_eq!(scored(5, -0.5, 1, &[], &[], 8).len(), 1);
    }
}
