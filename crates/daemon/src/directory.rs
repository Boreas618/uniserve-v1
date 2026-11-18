//! The leader's reading of the cluster directory: how much of a request's prefix each replica's
//! node holds, which the routing program weighs (`scheduling::routing::Replica::hits`).
//!
//! The directory lies on the leader node's Storage Server, which every node reports its device
//! and host pages to, and which knows Mooncake Store's pages from its master's object events
//! (`storage_server::directory`). A request's prefix is the chain of its prompt's page keys,
//! rooted at the seed of the epoch a replica serves and the request's isolation key
//! (`storage::key`): the leader computes one chain per distinct seed among the replicas, of cache
//! group 0 at the deployment's page size, and asks the server how many of each chain's leading
//! pages every node holds ([`Directory::hits`]). A replica's hit on a tier is its node's, in
//! tokens; the store's is every replica's. What the directory states may lag the tiers, which
//! costs only reuse: a replica the program prefers for a page gone since computes it.

use std::sync::Mutex;

use scheduling::kv::Tier;
use scheduling::routing::Hit;
use storage::Key;

/// The leader's connection to the directory on its node's Storage Server, made at the first
/// question and made again after it failed.
pub(crate) struct Directory {
    deployment: String,
    client: Mutex<Option<storage::Client>>,
}

/// One replica as its prefix is looked up: the seed of its cache group 0, its page size in
/// tokens, and the host of its scheduler's node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Placed {
    pub(crate) seed: [u8; 32],
    pub(crate) block: u32,
    pub(crate) host: String,
}

impl Directory {
    /// The directory of `deployment`, on this node's server of it.
    pub(crate) fn new(deployment: &str) -> Self {
        Self {
            deployment: deployment.to_owned(),
            client: Mutex::new(None),
        }
    }

    /// Per replica of `replicas`, the prefix of the prompt `tokens` of isolation key
    /// `isolation` its node holds on each tier and Mooncake Store holds, in tokens, with the
    /// keys of the first replica's chain (the routing input's `prefix`). Blocks on the server's
    /// answer. A directory that cannot be read states no hit, and the program routes without.
    pub(crate) fn hits(
        &self,
        tokens: &[u32],
        isolation: Option<u64>,
        replicas: &[Placed],
    ) -> (Vec<Key>, Vec<Vec<Hit>>) {
        // One chain per distinct seed and page size: the replicas of one epoch share it.
        type Shape<'a> = (&'a [u8; 32], u32);
        let mut chains: Vec<(Shape<'_>, Vec<Key>)> = Vec::new();
        let of: Vec<usize> = replicas
            .iter()
            .map(|replica| {
                let shape = (&replica.seed, replica.block);
                match chains.iter().position(|(known, _)| *known == shape) {
                    Some(index) => index,
                    None => {
                        chains.push((
                            shape,
                            chain(&replica.seed, isolation, tokens, replica.block),
                        ));
                        chains.len() - 1
                    }
                }
            })
            .collect();
        let prefix = chains
            .first()
            .map(|(_, keys)| keys.clone())
            .unwrap_or_default();
        if chains.iter().all(|(_, keys)| keys.is_empty()) {
            return (prefix, vec![Vec::new(); replicas.len()]);
        }

        let asked: Vec<Vec<Key>> = chains.iter().map(|(_, keys)| keys.clone()).collect();
        let Some(answered) = self.ask(asked) else {
            return (prefix, vec![Vec::new(); replicas.len()]);
        };
        let hits = replicas
            .iter()
            .zip(of)
            .map(|(replica, chain)| {
                let Some(found) = answered.get(chain) else {
                    return Vec::new();
                };
                let tokens = |pages: u32| pages.saturating_mul(replica.block);
                let node = found.nodes.iter().find(|node| node.node == replica.host);
                vec![
                    Hit {
                        tier: Tier::Device,
                        tokens: tokens(node.map_or(0, |node| node.device)),
                    },
                    Hit {
                        tier: Tier::Host,
                        tokens: tokens(node.map_or(0, |node| node.host)),
                    },
                    Hit {
                        tier: Tier::Store,
                        tokens: tokens(found.store),
                    },
                ]
            })
            .collect();
        (prefix, hits)
    }

    /// The server's answer to `chains`; `None`, logged, when it cannot be reached or refuses.
    fn ask(&self, chains: Vec<Vec<Key>>) -> Option<Vec<storage::protocol::Hits>> {
        let mut client = service::lock(&self.client);
        if client.is_none() {
            let connected = service::RuntimeDirectory::of_user()
                .and_then(|runtime| storage::socket(&runtime, &self.deployment))
                .map_err(storage::Error::Io)
                .and_then(|socket| {
                    let role = storage::Role::Head {
                        deployment: self.deployment.clone(),
                        epoch: 0,
                        generation: 0,
                    };
                    storage::Client::connect(&socket, role)
                });
            match connected {
                Ok(connected) => *client = Some(connected),
                Err(error) => {
                    tracing::debug!(%error, "the cluster directory is not reached");
                    return None;
                }
            }
        }
        let answered = client.as_mut()?.hits(chains);
        match answered {
            Ok(hits) => Some(hits),
            Err(error) => {
                tracing::debug!(%error, "the cluster directory did not answer");
                *client = None;
                None
            }
        }
    }
}

/// The keys of the complete pages of `block` tokens of `tokens` in cache group 0, rooted at the
/// group's seed `seed` under the isolation key `isolation`: the chain a scheduler of that seed
/// keys the prompt's pages by (`storage::key::page_key`).
pub(crate) fn chain(
    seed: &[u8; 32],
    isolation: Option<u64>,
    tokens: &[u32],
    block: u32,
) -> Vec<Key> {
    let root = storage::key::page_root(&Key(*seed), isolation);
    let mut parent = root;
    tokens
        .chunks_exact(block.max(1) as usize)
        .map(|page| {
            parent = storage::key::page_key(&parent, 0, 0, page);
            parent
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A prompt's chain keys its complete pages alone, as a scheduler keys them: a seed or an
    /// isolation key of its own roots another chain.
    #[test]
    fn a_chain_keys_the_complete_pages_from_the_seed_and_isolation_key() {
        let tokens: Vec<u32> = (0..200).collect();
        let keys = chain(&[1; 32], None, &tokens, 64);
        assert_eq!(keys.len(), 3);
        let root = storage::key::page_root(&Key([1; 32]), None);
        let first = storage::key::page_key(&root, 0, 0, &tokens[..64]);
        assert_eq!(keys[0], first);
        assert_eq!(
            keys[1],
            storage::key::page_key(&first, 0, 0, &tokens[64..128])
        );

        assert_ne!(chain(&[2; 32], None, &tokens, 64)[0], keys[0]);
        assert_ne!(chain(&[1; 32], Some(7), &tokens, 64)[0], keys[0]);
        // A shared prefix shares its keys.
        let mut other = tokens.clone();
        other[150] = 9_999;
        assert_eq!(chain(&[1; 32], None, &other, 64)[..2], keys[..2]);
    }
}
