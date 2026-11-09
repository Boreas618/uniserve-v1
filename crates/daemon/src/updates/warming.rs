//! The warming of an epoch switch (`specs/worker.md` §7.4 step 1): the
//! requests in flight reach the target's cache before the switchover, so that
//! each computes at most a bounded tail again at the promotion.
//!
//! The pages reach a target through the host KV cache. A serving scheduler
//! publishes every page of a request to the host as it completes, under a key
//! of its tokens alone, and a request that migrates carries those tokens, so the
//! target's admission finds them there and computes only the tokens past the
//! last page the host holds. What warming adds is the wait: the switchover
//! comes once every request's tail is within `update_config.cache_tail_tokens`.
//!
//! - **Rounds.** A round reads, from every scheduler, the tokens each request
//!   computed and the tokens the host holds of it (`Operation::Warmed`). The
//!   requests run on throughout, so a tail shrinks only as far as pages complete.
//! - **The limit.** `cache_tail_tokens`, else the largest source or target
//!   cache page: a page is the most a request computes past its last full one.
//! - **The prefill budget.** A request in its prefill computes pages faster than
//!   copies can follow, so after the first round that did not reach the limit,
//!   the source prefills at most `cache_prefill_tokens` per step, else a page.
//!   Decode keeps its budget, requests stay admitted, and the caller restores
//!   the budget at the promotion or when the preparation fails.
//! - **The bound.** A limit smaller than a request's pending page, or a host
//!   KV cache that does not keep up, would hold the switch forever, so the wait
//!   ends after [`LIMIT`]: the requests past the limit compute their tail on the
//!   target, which costs time and no correctness, and the report says so.
//!
//! A deployment without a host KV cache answers no request and warms nothing.

use std::future::Future;
use std::time::{Duration, Instant};

use serde::Serialize;
use storage::manifest::cache::{Operation, Snapshot};

/// How often a round reads the schedulers.
const ROUND: Duration = Duration::from_millis(50);

/// The longest a switch waits for its requests' tails.
pub(crate) const LIMIT: Duration = Duration::from_secs(60);

/// What the warming read when it ended.
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct Warming {
    pub(crate) rounds: u32,
    /// The longest tail the last round read, in tokens.
    pub(crate) tail_tokens: u32,
    /// The tail it waited for.
    pub(crate) limit_tokens: u32,
    /// The requests in flight, and the tokens the host holds of them.
    pub(crate) warmed_requests: u32,
    pub(crate) warmed_tokens: u64,
    /// Whether the source's prefill budget was limited.
    pub(crate) prefill_limited: bool,
    /// Whether the wait ended at [`LIMIT`] with a tail past the limit.
    pub(crate) timed_out: bool,
}

/// Runs the rounds against `ask`, the serving epoch's schedulers, until every
/// tail is within `tail_tokens`, else the largest of a source page and
/// `target_page`. Leaves the prefill budget it limited for the caller to
/// restore (`Operation::Prefill { tokens: None }`).
///
/// # Errors
///
/// Fails with the first answer a scheduler refused.
pub(crate) async fn warm<Ask, Asked>(
    mut ask: Ask,
    tail_tokens: Option<u32>,
    prefill_tokens: Option<u32>,
    target_page: u32,
    limit: Duration,
) -> Result<Warming, String>
where
    Ask: FnMut(Operation) -> Asked,
    Asked: Future<Output = Result<Vec<Snapshot>, String>>,
{
    let started = Instant::now();
    let mut warming = Warming::default();
    loop {
        let snapshots = ask(Operation::Warmed).await?;
        let page = snapshots
            .iter()
            .map(|snapshot| snapshot.page_tokens)
            .max()
            .unwrap_or(1)
            .max(target_page);
        warming.rounds += 1;
        warming.limit_tokens = tail_tokens.unwrap_or(page);
        warming.tail_tokens = 0;
        warming.warmed_requests = 0;
        warming.warmed_tokens = 0;
        for prefix in snapshots.iter().flat_map(|snapshot| &snapshot.prefixes) {
            warming.warmed_requests += 1;
            warming.warmed_tokens += u64::from(prefix.cached);
            warming.tail_tokens = warming
                .tail_tokens
                .max(prefix.computed.saturating_sub(prefix.cached));
        }
        if warming.tail_tokens <= warming.limit_tokens {
            return Ok(warming);
        }
        if started.elapsed() >= limit {
            warming.timed_out = true;
            return Ok(warming);
        }
        if !warming.prefill_limited {
            ask(Operation::Prefill {
                tokens: Some(prefill_tokens.unwrap_or(page)),
            })
            .await?;
            warming.prefill_limited = true;
        }
        tokio::time::sleep(ROUND).await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use storage::manifest::cache::Prefix;

    use super::*;

    fn prefix(computed: u32, cached: u32) -> Prefix {
        Prefix {
            epoch: 0,
            request: 1,
            resume: (0, 1),
            capacity_tokens: 0,
            tokens: Vec::new(),
            computed,
            cached,
            isolation_key: None,
            state_seeds: Vec::new(),
            pages: Vec::new(),
        }
    }

    fn snapshot(prefixes: Vec<Prefix>) -> Snapshot {
        Snapshot {
            prefixes,
            ranks: Vec::new(),
            page_tokens: 64,
            nodes: Default::default(),
        }
    }

    /// A source whose rounds read `reads` in turn, and that records every
    /// prefill budget it was given.
    #[expect(
        clippy::type_complexity,
        reason = "a test reads the operation and the record of the slots it touched, once"
    )]
    fn source(
        reads: Vec<Vec<Prefix>>,
    ) -> (
        impl FnMut(Operation) -> std::future::Ready<Result<Vec<Snapshot>, String>>,
        Arc<Mutex<Vec<Option<u32>>>>,
    ) {
        let budgets = Arc::new(Mutex::new(Vec::new()));
        let told = Arc::clone(&budgets);
        let mut reads = reads.into_iter();
        let mut last = Vec::new();
        let ask = move |operation| {
            std::future::ready(Ok(match operation {
                Operation::Warmed => {
                    if let Some(next) = reads.next() {
                        last = next;
                    }
                    vec![snapshot(last.clone())]
                }
                Operation::Prefill { tokens } => {
                    told.lock().expect("the budgets").push(tokens);
                    Vec::new()
                }
                other => panic!("a warming asks no {other:?}"),
            }))
        };
        (ask, budgets)
    }

    /// A tail within the limit needs no second round and limits no prefill.
    #[tokio::test]
    async fn a_request_within_the_limit_warms_at_once() {
        let (ask, budgets) = source(vec![vec![prefix(1000, 960)]]);
        let warming = warm(ask, None, None, 0, LIMIT).await.expect("it warms");
        assert_eq!(warming.rounds, 1);
        assert_eq!(warming.tail_tokens, 40);
        assert_eq!(warming.limit_tokens, 64, "a page, unless stated");
        assert!(!warming.prefill_limited);
        assert!(budgets.lock().expect("the budgets").is_empty());
    }

    /// `cache_tail_tokens` says how long the switch waits: the tail of the same
    /// reads that a page admits at once is waited for under a smaller limit.
    #[tokio::test]
    async fn the_stated_tail_is_what_the_rounds_wait_for() {
        let reads = vec![vec![prefix(1000, 960)], vec![prefix(1024, 1024)]];
        let (ask, _) = source(reads);
        let warming = warm(ask, Some(8), None, 0, LIMIT).await.expect("it warms");
        assert_eq!(warming.rounds, 2);
        assert_eq!(warming.tail_tokens, 0);
        assert_eq!(warming.limit_tokens, 8);
        assert_eq!(warming.warmed_tokens, 1024);
    }

    /// The first round past the limit limits the source's prefill, to
    /// `cache_prefill_tokens`, once, and to a page when none is stated.
    #[tokio::test]
    async fn a_round_past_the_limit_limits_the_prefill_once() {
        let reads = vec![
            vec![prefix(4096, 0)],
            vec![prefix(4160, 2048)],
            vec![prefix(4200, 4160)],
        ];
        let (ask, budgets) = source(reads.clone());
        let warming = warm(ask, None, Some(128), 0, LIMIT)
            .await
            .expect("it warms");
        assert_eq!(warming.rounds, 3);
        assert!(warming.prefill_limited);
        assert_eq!(*budgets.lock().expect("the budgets"), vec![Some(128)]);

        let (ask, budgets) = source(reads);
        warm(ask, None, None, 0, LIMIT).await.expect("it warms");
        assert_eq!(*budgets.lock().expect("the budgets"), vec![Some(64)]);
    }

    /// The default tail is the largest page of either epoch.
    #[tokio::test]
    async fn the_target_page_can_be_the_larger() {
        let (ask, _) = source(vec![vec![prefix(1000, 900)]]);
        let warming = warm(ask, None, None, 128, LIMIT).await.expect("it warms");
        assert_eq!(warming.limit_tokens, 128);
        assert_eq!(warming.rounds, 1);
    }

    /// A tail that does not shrink ends the wait at its bound, and says so.
    #[tokio::test]
    async fn a_tail_that_stays_ends_at_the_bound() {
        let (ask, _) = source(vec![vec![prefix(1000, 0)]]);
        let limit = Duration::from_millis(120);
        let warming = warm(ask, None, None, 0, limit).await.expect("it ends");
        assert!(warming.timed_out);
        assert_eq!(warming.tail_tokens, 1000);
    }

    /// A deployment without a host KV cache answers no request: nothing waits.
    #[tokio::test]
    async fn no_request_warms_nothing() {
        let (ask, _) = source(vec![Vec::new()]);
        let warming = warm(ask, None, None, 0, LIMIT).await.expect("it warms");
        assert_eq!(warming.rounds, 1);
        assert_eq!(warming.warmed_requests, 0);
    }
}
