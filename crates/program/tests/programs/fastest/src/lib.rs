//! The routing program that sends each request to the prefill replica that would finish its
//! prompt soonest: the prompt's uncached tokens at the replica's prefill rate after its queue
//! and its running batch. A mixed replica decodes what it prefilled, with no handoff; otherwise
//! the decode replica with the fewest running requests that can reserve the prompt's pages
//! decodes.

use scheduling::prelude::*;

#[derive(Deserialize, Serialize)]
pub struct Args {
    /// A replica's prefill throughput, in tokens per second.
    rate: f64,
    /// Tokens of a KV page.
    page: u32,
}

#[derive(Deserialize, Serialize)]
pub struct Fastest {
    args: Args,
    /// The requests routed, which survive restarts with the program's state.
    routed: u64,
}

impl Fastest {
    /// The seconds until `replica` would have computed the prompt of `request`.
    fn finish(&self, request: &Request, replica: &Replica) -> f64 {
        let hit = replica.hits.iter().map(|hit| hit.tokens).max().unwrap_or(0);
        let tokens = replica.queued + u64::from(request.prompt.saturating_sub(hit));
        replica.remaining.as_secs_f64() + tokens as f64 / self.args.rate
    }
}

impl Program for Fastest {
    type Args = Args;
    type Input = Routing;
    type Output = Route;

    fn new(args: Args) -> Self {
        Fastest { args, routed: 0 }
    }

    fn step(&mut self, routing: &Routing) -> Route {
        let request = &routing.request;
        let replicas = routing.replicas.iter().enumerate();

        let prefill = replicas
            .clone()
            .filter(|(_, replica)| replica.role.prefills())
            .min_by(|(_, a), (_, b)| self.finish(request, a).total_cmp(&self.finish(request, b)))
            .map(|(index, _)| index)
            .unwrap_or_else(|| panic!("no replica computes prompts"));

        if routing.replicas[prefill].role.decodes() {
            self.routed += 1;
            return Route {
                prefill,
                decode: prefill,
            };
        }

        // A push needs the decode side to reserve the prompt's pages first.
        let pages = u64::from(request.prompt.div_ceil(self.args.page));
        let decode = replicas
            .filter(|(_, replica)| replica.role.decodes() && replica.reservable >= pages)
            .min_by_key(|(_, replica)| replica.running)
            .map_or(prefill, |(index, _)| index);

        self.routed += 1;
        Route { prefill, decode }
    }
}

export!(Fastest);
