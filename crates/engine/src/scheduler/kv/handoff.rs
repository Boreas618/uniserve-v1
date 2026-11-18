//! The KV of a handoff (`specs/kv-cache.md` §4.1, §4.2, §5.1): what the decode side's tiers hold
//! of a lend's pages, which the KV program's handoff hook decides over ([`Handoff`]), and the
//! import the path it chose calls for. The scheduler's `handoff.rs` owns the roles, the lend and
//! the exchange with the engine; this file owns the pages.
//!
//! A push or a read imports every page of the lend into blocks of the importing request's own
//! (`Import`), whose full pages are committed under their keys once filled, and it runs from the
//! lend's tokens; a pull reads the full pages from the tier as any admission reads a prefix, ends
//! the lend with an import of no page and computes the tokens past its hit, the prompt's partial
//! last page among them. An import that fails leaves the request to an admission of its own,
//! which computes its prompt. A model that keeps a recurrent state ends the lend at once and
//! computes its prompt: its live state lies in the prefill replica's slot, which no import
//! reaches.

use proto::request::Lend;
use scheduling::kv::{Found, Handoff, Path, Tier, Transport};
use storage::runtime::{self, Completion, Lender, OwnerId};

use super::super::{Blocks, Request, WorkerGroup};
use super::cache::{DEVICE, HOST, REMOTE};
use super::tables::{commit, kv_group, kv_owner, shapes, state_shape};
use super::{Kv, Step};

/// The first cache class of `request`'s model that every request reads: the class whose pages a
/// handoff counts.
fn attention_class(request: &Request, groups: &[WorkerGroup]) -> Option<usize> {
    shapes(request, groups)
        .iter()
        .position(|(kind, _)| kind.required_rows())
}

impl Kv {
    /// The leading full pages of `request`'s prompt that `cache` holds in the remote store: what
    /// a decode replica may pull from there.
    pub(crate) fn stored_pages(&self, request: &Request, groups: &[WorkerGroup]) -> u32 {
        let Some(class) = attention_class(request, groups) else {
            return 0;
        };
        let keys = request.keys.get(class).map_or(&[][..], Vec::as_slice);
        let group = kv_group(request, groups).unwrap_or(runtime::GroupId(0));
        let leading = self
            .expected_hit(keys, group)
            .iter()
            .take_while(|tiers| **tiers & REMOTE != 0)
            .count();
        u32::try_from(leading).unwrap_or(u32::MAX)
    }

    /// The handoff of `request`'s `lend` as this replica's tiers hold its pages by `cache`, the
    /// input of the KV program's handoff hook; `local` when the lend lies in this host's buffer
    /// pool. `None` when the request imports nothing and its path is a pull of what the tiers
    /// hold: its model keeps a recurrent state, or it has no class every request reads.
    pub(crate) fn handoff(
        &self,
        request: &Request,
        lend: &Lend,
        local: bool,
        groups: &[WorkerGroup],
    ) -> Option<Handoff> {
        if state_shape(request, groups).is_some() {
            return None;
        }
        let class = attention_class(request, groups)?;
        let group = kv_group(request, groups)?;
        let block = shapes(request, groups)
            .get(class)
            .map_or(1, |&(_, block)| block.max(1));
        let pages = lend.tokens / block;
        let keys = request.keys.get(class).map_or(&[][..], Vec::as_slice);
        let keys = keys.get(..pages as usize).unwrap_or(keys);
        let tiers = self.expected_hit(keys, group);
        let leading = |bit: u8| {
            let held = tiers.iter().take_while(|tiers| **tiers & bit != 0).count();
            u32::try_from(held).unwrap_or(u32::MAX)
        };
        Some(Handoff {
            pages,
            tail: lend.tokens % block,
            found: vec![
                Found {
                    tier: Tier::Device,
                    pages: leading(DEVICE),
                },
                Found {
                    tier: Tier::Host,
                    pages: leading(HOST),
                },
                Found {
                    tier: Tier::Store,
                    pages: leading(REMOTE).max(lend.stored.min(pages)),
                },
            ],
            local,
            transports: vec![Transport::Nixl],
        })
    }

    /// The first round of the admission of `request`, which imports a lend from `lender` once
    /// its path is known: for a push or a read, the import of every lent page of each class every
    /// request reads; for a pull, the admission's own round and an import of no page, which ends
    /// the lend. A request with no lend, no path or no owner yet asks its admission's own round.
    pub(crate) fn handoff_requests(
        &self,
        request: &mut Request,
        lender: Option<Lender>,
        step: &Step,
    ) -> Vec<runtime::Request> {
        let (Some(lend), Some(path), Some(owner), Some(lender)) = (
            request.generation().and_then(|input| input.lend.clone()),
            request.path,
            kv_owner(request),
            lender,
        ) else {
            return self.admission_requests(request, step);
        };
        // The Storage Server's span of the handoff names the request.
        let trace = request.input.trace();
        let push = match path {
            Path::Push(_) => true,
            Path::Read(_) => false,
            Path::Pull(_) => {
                let mut asks = self.admission_requests(request, step);
                asks.push(runtime::Request::Import {
                    owner,
                    tables: Vec::new(),
                    lender,
                    push: false,
                    trace,
                });
                return asks;
            }
        };

        // Every page of the lend, of every class every request reads.
        let shapes = shapes(request, step.groups).to_vec();
        let tables: Vec<(u8, u32, u32)> = shapes
            .iter()
            .enumerate()
            .filter(|(_, (kind, _))| kind.required_rows())
            .filter_map(|(class, &(_, block))| {
                let pages = lend.tokens.div_ceil(block.max(1));
                Some((u8::try_from(class).ok()?, 0, pages))
            })
            .filter(|(.., pages)| *pages > 0)
            .collect();
        request.blocks = vec![Blocks::default(); shapes.len()];
        request.num_cached_tokens = 0;
        request.num_host_tokens = 0;
        vec![runtime::Request::Import {
            owner,
            tables,
            lender,
            push,
            trace,
        }]
    }

    /// Applies the answer to `request`'s `Import` of `tables` for `owner` and returns the
    /// requests it calls for. Blocks imported for a lend of `tokens` tokens hold its prompt up to
    /// them: each class's full pages are committed under their keys, read-only from now on, its
    /// partial last page is the request's to write on, and the blocks of its first chunk and the
    /// low watermark follow, so that it runs from the lend's tokens. A failed import leaves the
    /// request to the admission's own round, which computes what no tier holds. An import of no
    /// page holds nothing.
    pub(super) fn imported(
        &self,
        request: &mut Request,
        (owner, tables): (OwnerId, &[(u8, u32, u32)]),
        completion: &Completion,
        step: &Step,
    ) -> Vec<runtime::Request> {
        if tables.is_empty() {
            return Vec::new();
        }
        let Some(tokens) = request
            .generation()
            .and_then(|input| input.lend.as_ref())
            .map(|lend| lend.tokens)
        else {
            return Vec::new();
        };
        let Completion::Imported { .. } = completion else {
            return self.admission_requests(request, step);
        };

        let shapes = shapes(request, step.groups).to_vec();
        let mut follow = Vec::new();
        for &(class, _, pages) in tables {
            let Some(&(_, block)) = shapes.get(usize::from(class)) else {
                continue;
            };
            let full = tokens / block.max(1);
            let keys = request.keys.get(usize::from(class));
            for at in 0..full {
                let Some(key) = keys.and_then(|keys| keys.get(at as usize)) else {
                    continue;
                };
                let tokens = u16::try_from(block).unwrap_or(u16::MAX);
                follow.push(commit(owner, class, at, *key, tokens));
            }
            if let Some(table) = request.blocks.get_mut(usize::from(class)) {
                *table = Blocks {
                    first: 0,
                    shared: full,
                    end: pages,
                    asked: pages,
                };
            }
        }
        request.num_cached_tokens = tokens;

        // The first chunk's blocks, from the lend's tokens on.
        let mut blocks = request.blocks.clone();
        self.first_call_asks(request, (owner, tokens), (&mut blocks, &mut follow), step);
        request.blocks = blocks;
        follow
    }
}
