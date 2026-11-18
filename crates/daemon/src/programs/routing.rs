//! The routing program on the leader's request path: for every request it chooses the replica
//! that computes the prompt and the one that generates the tokens (`specs/configuration.md` §4).
//!
//! The leader checks every route it is given: an index outside the replicas, a prefill replica
//! that computes no prompts or a decode replica that generates no tokens is refused, recorded as
//! a `log` event of the program, and the request goes where the leader's own rule sends it, the
//! replica with the fewest requests in flight. So does every request while no routing program
//! runs, or after the one that ran ended.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use program::Instance;
use scheduling::routing::{Route, Routing};
use tokio::sync::watch;
use tracing::{info, warn};

use super::control::{create, save};
use super::host::Logs;
use super::{Current, Identity, Loaded, States};
use crate::observation::Store;

/// What a routing program's instance starts from.
pub(super) struct Start {
    pub(super) program: Loaded,
    /// How often its state is saved.
    pub(super) period: Duration,
    pub(super) store: Store,
    pub(super) states: States,
    pub(super) current: watch::Receiver<Arc<Current>>,
}

/// The routing program's instance, which every request's route goes through, one at a time.
#[derive(Default)]
pub(crate) struct Router {
    slot: Mutex<Option<Routed>>,
}

/// A running routing program.
struct Routed {
    name: String,
    identity: Identity,
    instance: Instance<Logs>,
    /// The program's log, beside the one its instance's host writes.
    logs: Logs,
    /// The arguments it was created with, which its saved state is kept under.
    args: toml::Table,
    states: States,
    period: Duration,
    saved: std::time::Instant,
}

impl Router {
    /// Whether a routing program runs: the leader builds a route's input only then.
    pub(crate) fn routes(&self) -> bool {
        service::lock(&self.slot).is_some()
    }

    /// The route of `routing` the routing program chooses; `None` when no program routes, it
    /// ended, or its route is refused, each recorded: the leader then routes by its own rule.
    pub(crate) fn route(&self, routing: &Routing) -> Option<Route> {
        let mut slot = service::lock(&self.slot);
        let routed = slot.as_mut()?;
        let route = match routed.instance.step::<_, Route>(routing) {
            Ok(route) => route,
            Err(error) => {
                routed.logs.failure("ended", &error.to_string());
                warn!(program = routed.name, "the routing program ended: {error}");
                *slot = None;
                return None;
            }
        };
        if let Err(reason) = check(routing, route) {
            routed.logs.failure("refused", &reason);
            return None;
        }
        if routed.saved.elapsed() >= routed.period {
            save(
                &mut routed.instance,
                &routed.states,
                &routed.name,
                &routed.args,
                &routed.logs,
            );
            routed.saved = std::time::Instant::now();
        }
        Some(route)
    }

    /// Runs `start`'s program in place of the one that runs, between two routes: the running
    /// instance stays when its source, arguments and module are the same, and is otherwise saved
    /// and replaced by a new one, which restores the saved state when it decodes. `None` ends
    /// the running one.
    pub(super) fn replace(&self, start: Option<Start>, states: &States) {
        let mut slot = service::lock(&self.slot);
        if let (Some(start), Some(routed)) = (&start, slot.as_mut())
            && routed.identity == start.program.identity()
        {
            routed.period = start.period;
            return;
        }
        if let Some(mut routed) = slot.take() {
            save(
                &mut routed.instance,
                states,
                &routed.name,
                &routed.args,
                &routed.logs,
            );
        }
        let Some(start) = start else {
            return;
        };

        let Start {
            program,
            period,
            store,
            states,
            current,
        } = start;
        let name = program.name.clone();
        let logs = Logs::new(name.clone(), store.clone(), current.clone());
        let host = || Logs::new(name.clone(), store.clone(), current.clone());
        match create(&program, &states, &logs, host) {
            Ok(instance) => {
                info!(program = name, source = %program.entry.source, "the routing program starts");
                *slot = Some(Routed {
                    identity: program.identity(),
                    name,
                    instance,
                    logs,
                    args: program.entry.args.clone(),
                    states,
                    period,
                    saved: std::time::Instant::now(),
                });
            }
            Err(reason) => {
                logs.failure("ended", &format!("the program did not start: {reason}"));
            }
        }
    }
}

/// Why `route` is no route of `routing`'s request: an index outside its replicas, or a replica
/// of the wrong role.
fn check(routing: &Routing, route: Route) -> Result<(), String> {
    let replicas = &routing.replicas;
    let replica = |index: usize, what: &str| {
        replicas.get(index).ok_or_else(|| {
            format!(
                "its {what} replica {index} is none of the {} replicas",
                replicas.len()
            )
        })
    };
    let prefill = replica(route.prefill, "prefill")?;
    let decode = replica(route.decode, "decode")?;
    if !prefill.role.prefills() {
        return Err(format!(
            "its prefill replica {} computes no prompts",
            prefill.id
        ));
    }
    if !decode.role.decodes() {
        return Err(format!(
            "its decode replica {} generates no tokens",
            decode.id
        ));
    }
    // A request computed and generated in one place is a mixed replica's: a prefill replica
    // hands every request off, and a decode replica continues only what another handed off.
    if route.prefill == route.decode && prefill.role != scheduling::routing::Role::Mixed {
        return Err(format!(
            "it computes and generates the request on replica {}, which is not mixed",
            prefill.id
        ));
    }
    Ok(())
}
