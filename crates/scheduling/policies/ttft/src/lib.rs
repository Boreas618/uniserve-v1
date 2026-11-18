//! The `ttft` routing program: each request to the prefill replica predicted to give its first
//! token soonest.
//!
//! For every request the program predicts, for each replica that computes prompts, when the
//! request's first token would be ready there: the replica's running batch completes
//! (`remaining`), then the prefill tokens queued before the request and the prompt's tokens the
//! replica does not already hold are computed at `rate` tokens per second. The prefix a replica
//! holds is its longest hit on any tier of the cluster directory, which is perhaps stale: a stale
//! hit costs only reuse. The replica of the earliest prediction computes the prompt, ties going to
//! the one with fewer running requests, then to the first.
//!
//! A replica that also decodes (`mixed`) decodes what it computed, with no handoff. Otherwise the
//! decode replica is the one with the fewest running requests among the decode replicas that can
//! reserve the prompt's pages (`page` tokens each), since the push of the prompt's KV needs them
//! reserved before the prefill writes; with none that can, the one with the fewest running
//! requests. A deployment without a replica of a role gets a route the leader refuses and replaces
//! with its own rule, and the program logs why.
//!
//! The program keeps no state beyond its arguments: what it decides from is in each input.

use scheduling::prelude::*;

/// The program's arguments, its package entry's `args`.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Args {
    /// A replica's prefill throughput, in tokens per second.
    pub rate: f64,
    /// The tokens of a KV page.
    pub page: u32,
}

/// The program; its arguments are all its state.
#[derive(Deserialize, Serialize)]
pub struct Ttft {
    args: Args,
}

impl Ttft {
    /// The seconds until `replica` would give the first token of `request`.
    fn predict(&self, request: &Request, replica: &Replica) -> f64 {
        let held = replica.hits.iter().map(|hit| hit.tokens).max().unwrap_or(0);
        let computed = request.prompt.saturating_sub(held);
        let tokens = replica.queued as f64 + f64::from(computed);
        replica.remaining.as_secs_f64() + tokens / self.args.rate
    }
}

impl Program for Ttft {
    type Args = Args;
    type Input = Routing;
    type Output = Route;

    fn new(args: Args) -> Self {
        Ttft { args }
    }

    fn step(&mut self, routing: &Routing) -> Route {
        let request = &routing.request;
        let replicas = || routing.replicas.iter().enumerate();

        let prefill = replicas()
            .filter(|(_, replica)| replica.role.prefills())
            .map(|(index, replica)| (index, replica, self.predict(request, replica)))
            .min_by(|(a, first, at), (b, second, bt)| {
                at.total_cmp(bt)
                    .then(first.running.cmp(&second.running))
                    .then(a.cmp(b))
            })
            .map(|(index, _, _)| index);
        let Some(prefill) = prefill else {
            warn!("no replica computes prompts");
            return Route {
                prefill: 0,
                decode: 0,
            };
        };
        if routing.replicas[prefill].role.decodes() {
            return Route {
                prefill,
                decode: prefill,
            };
        }

        // The fewest running requests among the decode replicas that can reserve the prompt's
        // pages, else among every decode replica.
        let pages = u64::from(request.prompt.div_ceil(self.args.page.max(1)));
        let fewest = |reserving: bool| {
            replicas()
                .filter(|(_, replica)| replica.role.decodes())
                .filter(|(_, replica)| !reserving || replica.reservable >= pages)
                .min_by_key(|(index, replica)| (replica.running, *index))
                .map(|(index, _)| index)
        };
        let decode = fewest(true).or_else(|| fewest(false)).unwrap_or_else(|| {
            warn!("no replica decodes");
            prefill
        });
        Route { prefill, decode }
    }
}

export!(Ttft);
