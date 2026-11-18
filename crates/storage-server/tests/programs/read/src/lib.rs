//! A KV program that reads every handoff's pages from the prefill replica over Mooncake Transfer
//! Engine, whatever transports the two sides have, and every period moves each page of every
//! node one tier down: a device page to the node's host memory, a host page into Mooncake Store,
//! keeping the host copy.
//!
//! Its handoff hook's state reads the `pulled` count another KV program's handoff hook saves
//! (`program-kv`), so it is restored from that program's state.

use scheduling::prelude::*;

/// The program takes no argument; any table is accepted.
#[derive(Deserialize, Serialize)]
pub struct Args {}

/// The handoff hook.
#[derive(Deserialize, Serialize)]
pub struct Read {
    /// Handoffs another program served by a pull, carried over in its state.
    #[serde(default)]
    pulled: u64,
    /// Handoffs this program answered.
    #[serde(default)]
    reads: u64,
}

impl Program for Read {
    type Args = Args;
    type Input = Handoff;
    type Output = Path;

    fn new(_: Args) -> Self {
        Read {
            pulled: 0,
            reads: 0,
        }
    }

    fn step(&mut self, _: &Handoff) -> Path {
        self.reads += 1;
        info!("read {} after {} pulls", self.reads, self.pulled);
        Path::Read(Transport::Mooncake)
    }
}

/// The tiering hook.
#[derive(Deserialize, Serialize)]
pub struct Lower {}

impl Program for Lower {
    type Args = Args;
    type Input = Tiering;
    type Output = Vec<Move>;

    fn new(_: Args) -> Self {
        Lower {}
    }

    fn step(&mut self, tiering: &Tiering) -> Vec<Move> {
        let mut moves = Vec::new();
        for (node, held) in tiering.directory.nodes.iter().enumerate() {
            for holding in &held.tiers {
                let to = match holding.tier {
                    Tier::Device => Tier::Host,
                    Tier::Host => Tier::Store,
                    Tier::Store => continue,
                };
                moves.extend(holding.keys.iter().map(|&key| Move {
                    key,
                    from: Location {
                        node,
                        tier: holding.tier,
                    },
                    to: Location { node, tier: to },
                    copy: true,
                }));
            }
        }
        info!(
            "{} move(s) over {} node(s)",
            moves.len(),
            tiering.directory.nodes.len()
        );
        moves
    }
}

export!(Kv<Read, Lower>);
