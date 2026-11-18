//! A KV program: at a handoff it pulls the full pages from the nearest tier that holds them all
//! and otherwise pushes them; every period it demotes each node's least recently used device
//! pages above its watermark to the node's host memory.

use scheduling::prelude::*;

#[derive(Deserialize, Serialize)]
pub struct Args {
    /// The share of a node's device pages above which the tiering hook demotes.
    watermark: f64,
}

/// The handoff hook.
#[derive(Deserialize, Serialize)]
pub struct Nearest {
    /// Handoffs served by a pull.
    pulled: u64,
}

impl Program for Nearest {
    type Args = Args;
    type Input = Handoff;
    type Output = Path;

    fn new(_: Args) -> Self {
        Nearest { pulled: 0 }
    }

    fn step(&mut self, handoff: &Handoff) -> Path {
        let mut found: Vec<&Found> = handoff
            .found
            .iter()
            .filter(|found| found.pages >= handoff.pages && found.tier != Tier::Device)
            .collect();
        found.sort_by_key(|found| found.tier);

        // The tail and any recurrent state are pushed by the Storage Server either way; the
        // full pages are pulled where a tier holds them all.
        if let Some(found) = found.first() {
            self.pulled += 1;
            return Path::Pull(found.tier);
        }
        Path::Push(
            handoff
                .transports
                .first()
                .copied()
                .unwrap_or(Transport::Nixl),
        )
    }
}

/// The tiering hook.
#[derive(Deserialize, Serialize)]
pub struct Demote {
    watermark: f64,
}

impl Program for Demote {
    type Args = Args;
    type Input = Tiering;
    type Output = Vec<Move>;

    fn new(args: Args) -> Self {
        Demote {
            watermark: args.watermark,
        }
    }

    fn step(&mut self, tiering: &Tiering) -> Vec<Move> {
        let mut moves = Vec::new();
        for (node, held) in tiering.directory.nodes.iter().enumerate() {
            for holding in held
                .tiers
                .iter()
                .filter(|holding| holding.tier == Tier::Device)
            {
                let keep = (holding.capacity as f64 * self.watermark) as usize;
                let excess = holding.keys.len().saturating_sub(keep);

                // Least recently used first.
                moves.extend(holding.keys[..excess].iter().map(|&key| Move {
                    key,
                    from: Location {
                        node,
                        tier: Tier::Device,
                    },
                    to: Location {
                        node,
                        tier: Tier::Host,
                    },
                    copy: false,
                }));
            }
        }
        moves
    }
}

export!(Kv<Nearest, Demote>);
