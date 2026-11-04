//! The check of a policy's declared limits against the build config.
//!
//! §4.4: `limits()` is read once at instantiation and its values "在 build
//! config 给出的上限以内" — within the ceilings the build config gives. A limit
//! above its ceiling, or zero where the mechanism needs a positive count, is a
//! policy failure (`BadLimits`), which reverts the update rather than serving
//! under limits the epoch was not built for.
//!
//! The mechanism passes the ceilings as a [`Limits`] of its own, from the build
//! config: the token, call and queue bounds of the deployment, the floor of a
//! prefix-hit deepening it guarantees, and the starvation deadline it allows.

use scheduling_guest::limits::Limits;
use scheduling_guest::time::Nanos;

use crate::error::PolicyError;

/// The longest starvation deadline §4.4 allows, the way sched_ext caps
/// `timeout_ms` at 30 seconds: a policy may not let a resident request wait
/// longer than this without failing.
pub const MAX_STARVATION_TIMEOUT: Nanos = Nanos(30_000_000_000);

/// Checks `limits` against the build config's `ceilings`.
///
/// Each limit is a value the mechanism uses as a hard constraint of the step or
/// of admission, so a value the epoch was not built for is refused rather than
/// clamped: clamping would serve under limits the policy did not choose, and
/// the policy's own decisions would then not fit the room the mechanism gives
/// it.
pub(crate) fn check(limits: &Limits, ceilings: &Limits) -> Result<(), PolicyError> {
    positive_at_most(
        "chunked_prefill_size",
        u64::from(limits.chunked_prefill_size),
        u64::from(ceilings.chunked_prefill_size),
    )?;
    positive_at_most(
        "max_batch",
        limits.max_batch as u64,
        ceilings.max_batch as u64,
    )?;
    positive_at_most(
        "max_num_batched_tokens",
        u64::from(limits.max_num_batched_tokens),
        u64::from(ceilings.max_num_batched_tokens),
    )?;

    // The waiting bound is unbounded when it is `None`; unbounded exceeds any
    // stated ceiling. The policy's own bound may be tighter than the
    // deployment's, which is a decision the policy is free to make.
    match (limits.max_queued_requests, ceilings.max_queued_requests) {
        (Some(queued), Some(ceiling)) if queued > ceiling => {
            return Err(outside(format!(
                "max_queued_requests is {queued}, above the build config's {ceiling}"
            )));
        }
        (None, Some(ceiling)) => {
            return Err(outside(format!(
                "max_queued_requests is unbounded, but the build config bounds it at {ceiling}"
            )));
        }
        _ => {}
    }

    // The floor of a prefix-hit deepening: the mechanism deepens only by at
    // least it (§4.4), and the build config's value is the floor the epoch
    // guarantees. A policy may prefetch more eagerly than that, but not more
    // sparingly, so its floor may not be below the build config's.
    if limits.prefetch_min_tokens < ceilings.prefetch_min_tokens {
        return Err(outside(format!(
            "prefetch_min_tokens is {}, below the build config's floor of {}",
            limits.prefetch_min_tokens, ceilings.prefetch_min_tokens
        )));
    }

    if limits.timeout.0 == 0 {
        return Err(outside(
            "timeout is zero; a resident request would be starved from the first step",
        ));
    }
    if limits.timeout > MAX_STARVATION_TIMEOUT {
        return Err(outside(format!(
            "timeout is {} ns, above the {} ns §4.4 allows",
            limits.timeout.0, MAX_STARVATION_TIMEOUT.0
        )));
    }
    if limits.timeout > ceilings.timeout {
        return Err(outside(format!(
            "timeout is {} ns, above the build config's {} ns",
            limits.timeout.0, ceilings.timeout.0
        )));
    }
    Ok(())
}

/// Refuses a limit that is zero or above its ceiling.
fn positive_at_most(name: &str, limit: u64, ceiling: u64) -> Result<(), PolicyError> {
    if limit == 0 {
        return Err(outside(format!("{name} is zero")));
    }
    if limit > ceiling {
        return Err(outside(format!(
            "{name} is {limit}, above the build config's {ceiling}"
        )));
    }
    Ok(())
}

/// The failure a limit outside the build config is.
fn outside(message: impl Into<String>) -> PolicyError {
    PolicyError::BadLimits {
        message: message.into(),
    }
}
