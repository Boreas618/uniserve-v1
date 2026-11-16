//! The replay of a buffer pool's trace and the pool's states in the terms of the design model
//! (`specs/buffer-pool.md` §18): verification, not serving. The Storage Server records traces
//! ([`super::trace`]); this module, built for the buffer pool's tests and with the crate's
//! `verification` feature (the `buffer-pool-verify` command), reads them back.
//!
//! [`replay`] drives a new buffer pool with a trace's inputs, checks every completion it
//! produces against the run's, and writes after every input the pool's state in the terms of
//! the design model ([`stated`]), which `support/verification/buffer-pool/steps.py` checks against the
//! model's steps. The enumeration of small configurations ([`super::enumeration`]) drives its
//! pools with the same inputs ([`apply`]) and writes its states in the same terms.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Write as _;
use std::io::Write;
use std::path::Path;
use std::rc::Rc;

use storage::protocol::{Key, Segment};
use storage::runtime::{Completion, GroupId, OwnerId, Request, RequestId, Stamps, Tier};

use super::trace::{
    CLOSE, COMPLETED, COMPLETION, DONE, ENDED, GRANT, HOLD, IMAGE, Input, OPEN, PEER_COPIES,
    PUBLISH, RELEASE_PEER, REQUEST, ROLL_BACK, START, TICK, Task, WITHDRAW,
};
use super::{Block, BufferPool, ClientId, Copy, Deferred, Held, Memories, QueueId, Target};

/// Acts on an input as the buffer pool's thread does: the replay's pool and
/// the enumeration's are driven by the same inputs the thread records. An
/// unreadable request is answered by the thread and changes nothing.
///
/// # Errors
///
/// Fails on an image that is not one.
pub(crate) fn apply(pool: &mut BufferPool, input: &Input) -> Result<(), String> {
    match input {
        Input::Image(image) => {
            *pool = rmp_serde::from_slice(image).map_err(|error| error.to_string())?;
        }
        Input::Open(client, open) => {
            // A layout the pool refused changed nothing, in the run too.
            let _ = pool.admit(*client, open);
        }
        Input::Close(client) => pool.end(*client),
        Input::Request(queue, record) => {
            if let Ok((id, request)) = Request::decode(record) {
                pool.take(*queue, id, request);
            }
        }
        Input::Completed(client, group, value) => pool.advance(*client, *group, *value),
        Input::Tick(now) => {
            pool.tick(*now);
        }
        Input::Start(_) => {}
        Input::Done(task, tokens) => pool.done(*task, *tokens),
        Input::Ended(task, ok) => pool.ended(*task, *ok),
        Input::Publish(epoch) => pool.commit_update(*epoch),
        Input::RollBack(epoch) => pool.roll_back(*epoch),
        Input::Hold(peer, keys) => {
            pool.hold(*peer, keys);
        }
        Input::ReleasePeer(peer) => {
            pool.release_peer(*peer);
        }
        Input::Grant(region, segment, first, units) => {
            if let Some(number) = pool.region_number(region) {
                let _ = pool.grant_segment(number, *segment, *first, *units);
            }
        }
        Input::Withdraw(region, segment) => {
            if let Some(number) = pool.region_number(region) {
                let _ = pool.withdraw_segment(number, *segment);
            }
        }
        Input::PeerCopies(task, held) => {
            pool.peer_copies(*task, held);
        }
    }
    Ok(())
}

/// The copies a completion asks a worker to run: the model's `ncopy`
/// counts them.
pub(crate) fn copies_asked(completion: &Completion) -> u32 {
    match completion {
        Completion::Copy { to, .. } => to.len() as u32,
        _ => 0,
    }
}

/// A completion's record as a replay states it.
fn shown(record: &[u8]) -> String {
    match Completion::decode(record) {
        Ok((request, completion)) => format!("{completion:?} for request {}", request.0),
        Err(error) => format!("an unreadable completion ({error})"),
    }
}

/// What a replay replayed: every input of its trace, of which the requests taken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Replayed {
    pub inputs: u64,
    pub requests: u64,
}

/// Replays the trace whose inputs are in the file `inputs` (the completions
/// are beside it, [`files`]) on a new buffer pool. Each completion the pool
/// produces is checked against the run's, in order; the pool's own check
/// holds after every input; and each task the run started is the next the
/// pool started. With `states`, the pool's state in the model's terms is
/// written there after every input ([`stated`], with [`classes`]), each
/// after the step that led to it (`step N N+1 <input>`, [`stated_input`])
/// and after the layout of the session ([`layout`]) whenever an input may
/// change it. The owners stated are those an `Attach` names. A run's last
/// completions may be missing from its file when the run stopped before it
/// wrote them; any other difference is an error.
///
/// Returns what it replayed.
///
/// # Errors
///
/// The first difference: the input after which it arose, with the run's
/// record and the replay's.
pub fn replay(inputs: &Path, mut states: Option<&mut dyn Write>) -> Result<Replayed, String> {
    let read =
        |path: &Path| std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()));
    let bytes = read(inputs)?;
    let mut records = Vec::new();
    for (at, kind, body) in frames(&bytes)? {
        let input = Input::decode(kind, body)
            .map_err(|error| format!("the input at byte {at}: {error}"))?;
        records.push((at, input));
    }
    let completions = read(&inputs.with_extension("completions"))?;
    let mut recorded = VecDeque::new();
    for (at, kind, mut body) in frames(&completions)? {
        if kind != COMPLETION {
            return Err(format!("the completion at byte {at} is of the kind {kind}"));
        }
        recorded.push_back((at, queue(&mut body)?, body.to_vec()));
    }

    // An attach is written alike in every version, so this build's reads
    // each one, whatever its session's version.
    let mut owners = Vec::new();
    for (_, input) in &records {
        if let Input::Request(queue, record) = input
            && let Ok((_, Request::Attach { owner, .. })) = Request::decode(record)
            && !owners.contains(&(*queue, owner))
        {
            owners.push((*queue, owner));
        }
    }

    let mut pool = BufferPool::new((0, 0));
    let mut names = SessionNames::default();
    let mut started = VecDeque::new();
    let (mut copies, mut replayed, mut requests) = (0, 0, 0);
    let mut write = |text: String| match states.as_mut() {
        Some(states) => states
            .write_all(text.as_bytes())
            .map_err(|error| error.to_string()),
        None => Ok(()),
    };
    write(format!(
        "{}state 0\n{}\n\n",
        layout(&pool, &names),
        stated(&pool, &names, &owners, 0)
    ))?;
    for (at, input) in &records {
        let named = |pool: &BufferPool, names: &SessionNames| {
            format!(
                "the input at byte {at} ({})",
                stated_input(pool, names, input)
            )
        };
        if let Input::Start(task) = input {
            match started.pop_front() {
                Some(found) if found == *task => continue,
                found => {
                    return Err(format!(
                        "{}: the run started the task of request {}, the replay {:?}",
                        named(&pool, &names),
                        task.1.0,
                        found.map(|(_, request): Task| request.0)
                    ));
                }
            }
        }
        // The input as the pool it acts on names it: a closing session's
        // epoch is named before it closes.
        let text = stated_input(&pool, &names, input);
        apply(&mut pool, input).map_err(|error| format!("{}: {error}", named(&pool, &names)))?;
        names.update(&pool);
        if let Input::Image(_) = input {
            // The tasks a pool started and the run did not record are gone with it.
            started.clear();
        }
        replayed += 1;
        if let Input::Request(..) = input {
            requests += 1;
        }

        for (queue, request, completion) in pool.completions() {
            copies += copies_asked(&completion);
            let mut record = Vec::new();
            completion.encode(request, &mut record);
            match recorded.pop_front() {
                Some((_, ran, ran_record)) if ran == queue && ran_record == record => {}
                Some((_, ran, ran_record)) => {
                    return Err(format!(
                        "after {}: the run completed {} on queue pair {ran:?}, the replay {completion:?} for request {} on {queue:?}",
                        named(&pool, &names),
                        shown(&ran_record),
                        request.0
                    ));
                }
                None => {}
            }
        }
        started.extend(pool.started());
        pool.stopped();
        pool.check()
            .map_err(|inconsistency| format!("after {}: {inconsistency}", named(&pool, &names)))?;

        let changes_layout = matches!(input, Input::Image(_) | Input::Open(..) | Input::Close(_));
        write(format!(
            "step {} {replayed} {}\n{}state {replayed}\n{}\n{}\n\n",
            replayed - 1,
            text,
            if changes_layout {
                layout(&pool, &names)
            } else {
                String::new()
            },
            stated(&pool, &names, &owners, copies),
            classes(&pool, &names)
        ))?;
    }
    if let Some((at, queue, record)) = recorded.pop_front() {
        return Err(format!(
            "the run completed {} on queue pair {queue:?} (byte {at} of its completions), and the replay did not",
            shown(&record)
        ));
    }
    Ok(Replayed {
        inputs: replayed,
        requests,
    })
}

/// The names of the sessions in the terms of [`stated`], which qualify
/// their owners and groups (`<session>.<owner>`, `<session>.g<group>`). A
/// session is one of the model's epochs (`KvCore`'s `Epochs`: the clients,
/// one session each), named after its engine epoch: `e<epoch>`, or
/// `e<epoch>-<n>` for the n-th session of an epoch that had others, as an
/// engine with a parked standby has (a session per engine). A session is
/// named when it opens and keeps its name until it closes, so a standby's
/// session keeps its name when the serving engine's session closes and the
/// standby takes over. A session never named, as the enumeration's, is
/// `e<epoch>`.
#[derive(Default)]
pub(crate) struct SessionNames {
    names: BTreeMap<ClientId, String>,
    /// Per engine epoch, the sessions of it named so far.
    opened: BTreeMap<u64, u32>,
}

impl SessionNames {
    /// Names the sessions `pool` has open that have no name yet, in the
    /// order of their clients, and forgets those of closed sessions.
    pub(crate) fn update(&mut self, pool: &BufferPool) {
        self.names
            .retain(|client, _| pool.sessions.contains_key(client));
        let mut open: Vec<(ClientId, u64)> = pool
            .sessions
            .iter()
            .map(|(client, epoch)| (*client, *epoch))
            .collect();
        open.sort_unstable();
        for (client, epoch) in open {
            if self.names.contains_key(&client) {
                continue;
            }
            let count = self.opened.entry(epoch).or_insert(0);
            *count += 1;
            let name = match *count {
                1 => format!("e{epoch}"),
                n => format!("e{epoch}-{n}"),
            };
            self.names.insert(client, name);
        }
    }

    /// The name of a client's session; `e?` for a client with no session open.
    fn of(&self, pool: &BufferPool, client: ClientId) -> String {
        match pool.sessions.get(&client) {
            None => "e?".to_owned(),
            Some(epoch) => self
                .names
                .get(&client)
                .cloned()
                .unwrap_or_else(|| format!("e{epoch}")),
        }
    }
}

/// An input to `pool` as `support/verification/buffer-pool/steps.py` reads it: its
/// kind, and what it names in the terms of [`stated`] (owners and groups
/// qualified by their session's name, positions by number, keys by their
/// 32 bytes in hex, stamps as `<group>:<sequence>`, a request's session as
/// `e=`; an update's commit and rollback name the engine epoch, `e<epoch>`).
pub(crate) fn stated_input(pool: &BufferPool, names: &SessionNames, input: &Input) -> String {
    let session = |client: ClientId| names.of(pool, client);
    match input {
        Input::Image(_) => "image".into(),
        Input::Open(..) => "open".into(),
        Input::Close(client) => format!("close e={}", session(*client)),
        Input::Request(queue, record) => match Request::decode(record) {
            Ok((_, request)) => stated_request(&request, &session(queue.0)),
            Err(_) => "unreadable".into(),
        },
        Input::Completed(client, group, value) => format!(
            "completed g={}.g{} value={value}",
            session(*client),
            group.0
        ),
        Input::Tick(_) => "tick".into(),
        Input::Start(_) => "start".into(),
        Input::Done(..) => "done".into(),
        Input::Ended(_, ok) => format!("ended ok={}", u8::from(*ok)),
        Input::Publish(epoch) => format!("publish e=e{epoch}"),
        Input::RollBack(epoch) => format!("roll-back e=e{epoch}"),
        Input::Hold(..) => "hold".into(),
        Input::ReleasePeer(_) => "release-peer".into(),
        Input::Grant(..) => "grant".into(),
        Input::Withdraw(..) => "withdraw".into(),
        Input::PeerCopies(_, held) => format!(
            "peer-copies held={}",
            held.iter()
                .take_while(|encoding| encoding.is_some())
                .count()
        ),
    }
}

fn stated_request(request: &Request, session: &str) -> String {
    let owned = |owner: &OwnerId| format!("{session}.{}", owner.0);
    let keys = |keys: &[Key]| {
        let named: Vec<String> = keys.iter().map(Key::hex).collect();
        named.join(",")
    };
    let after = |stamps: &Stamps| {
        let named: Vec<String> = stamps
            .iter()
            .map(|(group, at)| format!("{session}.g{}:{at}", group.0))
            .collect();
        named.join(",")
    };
    let span = |at: u32, count: usize| {
        let named: Vec<String> = (0..count as u32)
            .map(|offset| (at + offset).to_string())
            .collect();
        named.join(",")
    };
    let text = match request {
        Request::Attach { owner, group } => {
            format!("attach o={} g={session}.g{}", owned(owner), group.0)
        }
        Request::Detach {
            owner,
            after: stamps,
        } => format!("detach o={} after={}", owned(owner), after(stamps)),
        Request::Allocate {
            owner,
            class,
            at,
            count,
            ..
        } => {
            format!(
                "allocate o={} class={class} ps={}",
                owned(owner),
                span(*at, *count as usize)
            )
        }
        Request::Pin {
            owner,
            class,
            at,
            keys: named,
        } => format!(
            "pin o={} class={class} ps={} ks={}",
            owned(owner),
            span(*at, named.len()),
            keys(named)
        ),
        Request::Release {
            owner,
            class,
            at,
            count,
            after: stamps,
        } => format!(
            "release o={} class={class} ps={} after={}",
            owned(owner),
            span(*at, *count as usize),
            after(stamps)
        ),
        Request::Commit {
            owner,
            class,
            at,
            key,
            after: stamps,
            ..
        } => format!(
            "commit o={} class={class} ps={at} ks={} after={}",
            owned(owner),
            keys(&[*key]),
            after(stamps)
        ),
        Request::Clone {
            owner,
            class,
            at,
            key,
        } => {
            format!(
                "clone o={} class={class} ps={at} ks={}",
                owned(owner),
                keys(&[*key])
            )
        }
        Request::Transfer {
            owner,
            group,
            after: stamps,
        } => {
            format!(
                "transfer o={} to={session}.g{} after={}",
                owned(owner),
                group.0,
                after(stamps)
            )
        }
        Request::Evict { keys: named } => format!("evict ks={}", keys(named)),
        Request::Victims { .. } => "victims".into(),
        Request::List => "list".into(),
        Request::Resize { .. } => "resize".into(),
        Request::Load {
            owner,
            class,
            at,
            keys: named,
        } => format!(
            "load o={} class={class} ps={} ks={}",
            owned(owner),
            span(*at, named.len()),
            keys(named)
        ),
        Request::Store {
            group,
            keys: named,
            tier,
        } => {
            let tier = match tier {
                Tier::Device => "device",
                Tier::Host => "host",
                Tier::Remote => "remote",
            };
            format!(
                "store g={session}.g{} ks={} tier={tier}",
                group.0,
                keys(named)
            )
        }
        Request::Save { owner, key, .. } => format!("save o={} ks={}", owned(owner), keys(&[*key])),
        Request::Restore { owner, key } => {
            format!("restore o={} ks={}", owned(owner), keys(&[*key]))
        }
        Request::Finish {
            ok, after: stamps, ..
        } => format!("finish ok={} after={}", u8::from(*ok), after(stamps)),
        Request::Prefetch {
            group,
            class,
            keys: named,
            source,
            ..
        } => {
            let source = match source {
                storage::runtime::Source::Remote => "remote",
                storage::runtime::Source::Peers => "peers",
            };
            format!(
                "prefetch g={session}.g{} class={class} ks={} source={source}",
                group.0,
                keys(named)
            )
        }
        Request::Replicate {
            owner,
            class,
            at,
            group,
            after: stamps,
            ..
        } => format!(
            "replicate o={} class={class} ps={at} to={session}.g{} after={}",
            owned(owner),
            group.0,
            after(stamps)
        ),
    };
    format!("{text} e={session}")
}

/// What a state in the model's terms leaves out and the model has no terms
/// for, in lines that start with `#`, which `support/verification/buffer-pool/steps.py`
/// reads and `agreement.py` never sees: the class of each key cached in a
/// class other than 0 (the model has one class, so a key cached in another is
/// outside its terms, A6; a key with no such line is of the class 0), each
/// key cached on the device in more than one region, and the units each
/// owner of an open session holds in each class other than 0, as runs
/// (`# held e<epoch>.<owner> class=<class> <region:first-last>...`):
/// [`stated`] states an owner's table of the class 0 alone, so a unit of the
/// model's region that another class's block holds is neither free nor held
/// in its terms, and such a state is outside them.
pub(crate) fn classes(pool: &BufferPool, names: &SessionNames) -> String {
    let classes: BTreeMap<String, u8> = pool
        .cache
        .iter()
        .filter(|(_, entry)| entry.class != 0)
        .map(|(key, entry)| (key.hex(), entry.class))
        .collect();
    let mut lines: Vec<String> = classes
        .into_iter()
        .map(|(key, class)| format!("# class {key} {class}"))
        .collect();
    // A key cached on the device in several regions (one per layout): the
    // model caches a key in one device block.
    let mut copies: Vec<String> = pool
        .cache
        .iter()
        .filter(|(_, entry)| entry.device.len() > 1)
        .map(|(key, entry)| format!("# device {} {}", key.hex(), entry.device.len()))
        .collect();
    copies.sort_unstable();
    lines.extend(copies);
    // The units of an owner's tables of the other classes, in any region:
    // a draft model's cache pages, say, taken from the region of the class 0.
    let mut held = Vec::new();
    for ((client, number), owner) in &pool.owners {
        if !pool.sessions.contains_key(client) {
            continue;
        }
        for (class, table) in owner.tables.iter().enumerate().skip(1) {
            // Each run of a block as `region:first-last`, or `region:unit`.
            let units: Vec<String> = table
                .values()
                .flat_map(|entry| entry.block().stated())
                .map(|(region, first, count)| match count {
                    1 => format!("{region}:{first}"),
                    _ => format!("{region}:{first}-{}", first + count - 1),
                })
                .collect();
            if !units.is_empty() {
                held.push(format!(
                    "# held {}.{} class={class} {}",
                    names.of(pool, *client),
                    number.0,
                    units.join(" ")
                ));
            }
        }
    }
    held.sort_unstable();
    lines.extend(held);
    lines.join("\n")
}

/// The layout of the pool's sessions in the terms of [`stated`], or `layout
/// none` while no session is open: per group, qualified by its session's
/// name ([`SessionNames`]), the bit of its memory, the region of its slot
/// rows, its regions of the class 0 and the bits of the host parts it copies
/// that class with; each open session by its name, with groups on this
/// server or none (a deployment over several hosts opens a session on each
/// host's server); per class, its host region; per region a group names,
/// its memory space and the bytes of a unit (the unit `i` of a region
/// occupies the bytes from `i` units of its space, so regions of two layouts
/// over one space share memory).
pub(crate) fn layout(pool: &BufferPool, names: &SessionNames) -> String {
    if pool.sessions.is_empty() {
        return "layout none\n".into();
    }
    let mut text = String::new();
    // By epoch, session and group.
    let groups: BTreeMap<(u64, ClientId, u8), _> = pool
        .groups
        .iter()
        .filter_map(|((client, group), found)| {
            let epoch = pool.sessions.get(client)?;
            Some(((*epoch, *client, group.0), found))
        })
        .collect();
    let mut named = BTreeSet::new();
    for ((_, client, group), found) in groups {
        let kv: Vec<u8> = found
            .regions
            .first()
            .into_iter()
            .flatten()
            .copied()
            .collect();
        named.extend(kv.iter().copied());
        named.insert(found.slots);
        let kv: Vec<String> = kv.iter().map(u8::to_string).collect();
        let parts = found.parts.first().copied().unwrap_or(Memories::NONE);
        let parts: Vec<String> = parts.iter().map(|memory| format!("m{memory}")).collect();
        let _ = writeln!(
            text,
            "layout group {}.g{group} memory m{} rows {} kv {} parts {}",
            names.of(pool, client),
            found.memory,
            found.slots,
            kv.join(","),
            parts.join(",")
        );
    }
    let mut sessions: Vec<(u64, ClientId)> = pool
        .sessions
        .iter()
        .map(|(client, epoch)| (*epoch, *client))
        .collect();
    sessions.sort_unstable();
    for (_, client) in sessions {
        let _ = writeln!(text, "layout session {}", names.of(pool, client));
    }
    let hosts: BTreeSet<u8> = pool
        .classes
        .values()
        .flat_map(|(classes, _)| {
            classes
                .iter()
                .filter_map(|class| class.host.map(|(region, _)| region))
        })
        .collect();
    for region in &hosts {
        let _ = writeln!(text, "layout host {region}");
    }
    for region in named.iter().chain(&hosts) {
        if let Some(found) = pool.regions.get(usize::from(*region)) {
            let _ = writeln!(
                text,
                "layout region {region} space {} bytes {}",
                found.space, found.unit_bytes
            );
        }
    }
    text
}

/// The pool's state in the terms of the design model, one line per fact, in
/// order: the copies the workers were asked for (`copies`), each group's
/// `completed` word, the free units as runs (`region:first-last`), each owner
/// of `owners` by the queue pair that attaches it (live with its group, slot
/// and its table of the class 0 by position, or gone, or new), each cached
/// key with its copies, the
/// memories that hold each block an owner or the cache holds on the device
/// and the host parts that hold each host copy, the copies in flight with
/// their keys and the host parts a store or a save writes, what waits for a
/// stamp with the stamps it still waits for (`after=g<n>:<sequence>,...`,
/// one `release` line per group and stamp a release waits for), and the
/// tasks with their keys and the host parts a prefetch writes, a task of
/// several keys as one per key. A block a waiting commit holds states its
/// memories too. A unit is `region:unit`, a key its 32 bytes in hex. `support/verification/buffer-pool/agreement.py` writes the
/// model's states in the same lines.
pub(crate) fn stated(
    pool: &BufferPool,
    names: &SessionNames,
    owners: &[(QueueId, OwnerId)],
    copies: u32,
) -> String {
    let unit = |block: &Rc<Block>| {
        let runs: Vec<String> = block
            .stated()
            .into_iter()
            .map(|run| format!("{}:{}", run.0, run.1))
            .collect();
        runs.join("+")
    };
    let units = |blocks: BTreeSet<String>| blocks.into_iter().collect::<Vec<_>>().join(" ");
    // Memories by their numbers: `m2,m3`.
    let bits = |memories: Memories| {
        let named: Vec<String> = memories.iter().map(|memory| format!("m{memory}")).collect();
        named.join(",")
    };
    let mut lines = vec![format!("ncopy {copies}")];

    // Each group of an open session, qualified by its epoch.
    let group_name =
        |client: ClientId, group: GroupId| format!("{}.g{}", names.of(pool, client), group.0);
    let owner_name =
        |client: ClientId, owner: OwnerId| format!("{}.{}", names.of(pool, client), owner.0);
    let completed: BTreeMap<String, u64> = pool
        .groups
        .iter()
        .filter(|((client, _), _)| pool.sessions.contains_key(client))
        .map(|((client, group), found)| (group_name(*client, *group), found.completed))
        .collect();
    lines.push(
        completed
            .iter()
            .fold("completed".to_owned(), |line, (group, at)| {
                format!("{line} {group}={at}")
            }),
    );

    // Each of the model's epochs the state names: an open session, with its
    // engine epoch's trial; and an engine epoch with no session open that is
    // still on trial, or whose update was rolled back and that made a copy
    // still cached (R8), as `e<epoch>`.
    let trial = |epoch: u64| match pool.trial.get(&epoch) {
        Some(true) => "yes",
        Some(false) => "no",
        None => "rolled",
    };
    for (client, epoch) in &pool.sessions {
        let name = names.of(pool, *client);
        lines.push(format!("epoch {name} open trial={}", trial(*epoch)));
    }
    let mut closed: BTreeSet<u64> = pool
        .trial
        .iter()
        .filter(|(_, trial)| **trial)
        .map(|(epoch, _)| *epoch)
        .collect();
    closed.extend(
        pool.cache
            .values()
            .flat_map(|entry| entry.copies().filter_map(|(_, maker)| *maker)),
    );
    for epoch in closed {
        if !pool.sessions.values().any(|of| *of == epoch) {
            lines.push(format!("epoch e{epoch} closed trial={}", trial(epoch)));
        }
    }

    // The free units as runs, `region:first-last` or `region:unit` for one,
    // each as long as it can be, by region and first unit: a state grows with
    // what is held, not with the regions' size.
    let mut free = Vec::new();
    for (id, region) in pool.regions.iter().enumerate() {
        let mut runs: Vec<(u32, u32)> = region
            .free_runs()
            .filter(|(_, count)| *count > 0)
            .map(|(first, count)| (first, first + count - 1))
            .collect();
        runs.sort_unstable();
        let mut joined: Vec<(u32, u32)> = Vec::with_capacity(runs.len());
        for (first, last) in runs {
            match joined.last_mut() {
                Some(previous) if previous.1 + 1 == first => previous.1 = last,
                _ => joined.push((first, last)),
            }
        }
        free.extend(joined.into_iter().map(|(first, last)| {
            if first == last {
                format!("{id}:{first}")
            } else {
                format!("{id}:{first}-{last}")
            }
        }));
    }
    lines.push(format!("free {}", free.join(" ")));

    // An owner's number is taken once on its queue pair: a number at or
    // below the highest taken there that names no owner is of one that is
    // gone. The owners stated are those of `owners` and every live one (a
    // pool's image hands over owners no input of the trace attaches), of
    // the open sessions.
    let mut stated_owners = BTreeMap::new();
    for (queue, number) in owners {
        if !pool.sessions.contains_key(&queue.0) || pool.owners.contains_key(&(queue.0, *number)) {
            continue;
        }
        let taken = pool.numbers.get(queue).map_or(0, |numbers| numbers.0);
        let is = if number.0 <= taken { "gone" } else { "new" };
        let name = owner_name(queue.0, *number);
        stated_owners.insert(name.clone(), format!("owner {name} {is}"));
    }
    for ((client, number), owner) in &pool.owners {
        if !pool.sessions.contains_key(client) {
            continue;
        }
        let name = owner_name(*client, *number);
        let mut line = format!(
            "owner {name} live {} slot={}",
            group_name(*client, owner.group),
            unit(&owner.slot)
        );
        for (position, held) in owner.tables.first().into_iter().flatten() {
            let kind = if matches!(held, Held::Private(_)) {
                "private"
            } else {
                "shared"
            };
            let _ = write!(line, " p{position}={kind}:{}", unit(held.block()));
        }
        stated_owners.insert(name, line);
    }
    lines.extend(stated_owners.into_values());

    let mut valid = BTreeMap::new();
    let mut note = |block: &Rc<Block>| {
        // A noted block is a KV block, never a slot row, so its validity names its memories.
        let held_in = block.valid.get().memories().unwrap_or(Memories::NONE);
        let memories: Vec<String> = held_in.iter().map(|memory| format!("m{memory}")).collect();
        valid.insert(unit(block), memories.join(" "));
    };
    for owner in pool.owners.values() {
        owner
            .tables
            .first()
            .into_iter()
            .flatten()
            .for_each(|(_, held)| note(held.block()));
    }
    // A commit that waits holds its block, whose memories the commit's
    // effect keeps when the owner is gone.
    for (_, work) in pool.groups.values().flat_map(|group| &group.deferred) {
        if let Deferred::Commit { block, .. } = work {
            note(block);
        }
    }
    for (key, entry) in &pool.cache {
        entry.device.iter().for_each(|(block, _)| note(block));
        entry.host.iter().for_each(|(block, ..)| note(block));
        let device = entry
            .device
            .first()
            .map_or("-".to_owned(), |(block, _)| unit(block));
        let host = entry
            .host
            .as_ref()
            .map_or("-".to_owned(), |(block, ..)| unit(block));
        // The epoch on trial each copy is visible to alone (R8), or `-`.
        let maker = |maker: Option<u64>| maker.map_or("-".to_owned(), |epoch| format!("e{epoch}"));
        let made = entry
            .device
            .first()
            .map_or("-".to_owned(), |(_, of)| maker(*of));
        let host_made = entry
            .host
            .as_ref()
            .map_or("-".to_owned(), |(_, of, _)| maker(*of));
        lines.push(format!(
            "key {} d={device} h={host} remote={} vd={made} vh={host_made}",
            key.hex(),
            u8::from(entry.remote)
        ));
    }
    lines.extend(
        valid
            .into_iter()
            .map(|(unit, memories)| format!("valid {unit} {memories}").trim_end().to_owned()),
    );

    let copy_line = |client: ClientId, copy: &Copy| {
        // The key a copy caches its block under: a load's, a store's and a
        // save's; a clone and a restore cache nothing.
        let key = match &copy.to {
            Target::Owner { key: Some(key), .. } | Target::Entry { key, .. } => key.hex(),
            _ => "-".to_owned(),
        };
        let (kind, owner, position, memory) = match copy.to {
            Target::Owner {
                owner,
                at,
                key: None,
                ..
            } => (
                "clone",
                owner_name(client, owner),
                at.to_string(),
                copy.memory,
            ),
            Target::Owner {
                owner,
                at,
                key: Some(_),
                ..
            } => (
                "load",
                owner_name(client, owner),
                at.to_string(),
                copy.memory,
            ),
            Target::Slot { owner } => (
                "restore",
                owner_name(client, owner),
                "-".to_owned(),
                Memories::NONE,
            ),
            Target::Entry { .. }
                if copy
                    .reads
                    .first()
                    .is_some_and(|block| block.class.is_none()) =>
            {
                // A save reads its owner's slot, while that owner is there.
                let owner = pool
                    .owners
                    .iter()
                    .find(|(_, owner)| Rc::ptr_eq(&owner.slot, &copy.reads[0]));
                let owner =
                    owner.map_or("-".to_owned(), |((of, owner), _)| owner_name(*of, *owner));
                ("save", owner, "-".to_owned(), Memories::NONE)
            }
            _ => ("store", "-".to_owned(), "-".to_owned(), Memories::NONE),
        };
        // A copy into device memory writes one memory.
        let memory = memory
            .iter()
            .next()
            .map_or("-".to_owned(), |memory| format!("m{memory}"));
        // The host parts a store or a save writes: a store's fill those its
        // host block lacks, otherwise those of its group.
        let parts = if matches!(kind, "store" | "save") {
            bits(copy.memory)
        } else {
            String::new()
        };
        // The session whose request asked for the copy.
        format!(
            "kind={kind} o={owner} p={position} read={} write={} fill={} mem={memory} k={key} parts={parts} s={}",
            unit(&copy.reads[0]),
            unit(&copy.write),
            u8::from(copy.fill),
            names.of(pool, client)
        )
    };

    // The stamps work still waits for: the one it waits at and those of its
    // `rest` that its group's `completed` word has not reached, by group.
    let pending = |client: ClientId, at: (GroupId, u64), rest: &Stamps| {
        let completed = |group: GroupId| {
            pool.groups
                .get(&(client, group))
                .map_or(0, |found| found.completed)
        };
        let waits: BTreeMap<String, u64> = std::iter::once(at)
            .chain(rest.iter().copied())
            .filter(|(group, stamp)| *stamp > completed(*group))
            .map(|(group, stamp)| (group_name(client, group), stamp))
            .collect();
        let named: Vec<String> = waits
            .into_iter()
            .map(|(group, stamp)| format!("{group}:{stamp}"))
            .collect();
        named.join(",")
    };

    for ((client, group), found) in &pool.groups {
        for (stamp, work) in &found.deferred {
            let at = (*group, *stamp);
            match work {
                Deferred::Release { blocks } => {
                    let held: BTreeSet<String> = blocks.iter().map(&unit).collect();
                    lines.push(format!(
                        "release after={} {}",
                        pending(*client, at, &Vec::new()),
                        units(held)
                    ));
                }
                Deferred::Commit {
                    owner,
                    at: position,
                    block,
                    key,
                    rest,
                    ..
                } => lines.push(format!(
                    "commit o={} p={position} u={} k={} after={}",
                    owner_name(*client, *owner),
                    unit(block),
                    key.hex(),
                    pending(*client, at, rest)
                )),
                Deferred::Finish {
                    queue,
                    copy,
                    ok,
                    rest,
                    ..
                } => {
                    let copies = pool
                        .copies
                        .get(&(*queue, *copy))
                        .map_or(&[][..], Vec::as_slice);
                    let after = pending(*client, at, rest);
                    lines.extend(copies.iter().map(|copy| {
                        format!(
                            "finish ok={} after={after} {}",
                            u8::from(*ok),
                            copy_line(queue.0, copy)
                        )
                    }));
                }
                Deferred::Transfer {
                    owner, to, rest, ..
                } => lines.push(format!(
                    "transfer o={} to={} after={}",
                    owner_name(*client, *owner),
                    group_name(*client, *to),
                    pending(*client, at, rest)
                )),
                Deferred::Replicate {
                    owner,
                    at: position,
                    block,
                    to,
                    rest,
                    ..
                } => lines.push(format!(
                    "replicate o={} p={position} to={} u={} after={}",
                    owner_name(*client, *owner),
                    group_name(*client, *to),
                    unit(block),
                    pending(*client, at, rest)
                )),
            }
        }
    }

    for (task, copies) in &pool.copies {
        let expired = u8::from(!pool.answers.contains_key(task));
        match copies.first().map(|copy| &copy.to) {
            Some(Target::Group { owner, to }) => {
                let held = copies
                    .iter()
                    .flat_map(|copy| &copy.reads)
                    .map(&unit)
                    .collect();
                let writes = copies
                    .iter()
                    .filter(|copy| copy.fill)
                    .map(|copy| unit(&copy.write))
                    .collect();
                let slot = copies
                    .iter()
                    .find(|copy| !copy.fill)
                    .map_or("-".to_owned(), |copy| unit(&copy.write));
                let line = format!(
                    "task transfer o={} to={} slot={slot} units={} writes={}",
                    owner_name(task.0.0, *owner),
                    group_name(task.0.0, *to),
                    units(held),
                    units(writes)
                );
                lines.push(line.trim_end().to_owned());
            }
            Some(Target::Memory {
                owner, at, group, ..
            }) => lines.push(format!(
                "task replicate o={} p={at} to={} u={}",
                owner_name(task.0.0, *owner),
                group_name(task.0.0, *group),
                unit(&copies[0].write)
            )),
            Some(Target::Entry { remote, .. }) if copies[0].deadline.is_some() => {
                // A task of several keys is a task per key to the model,
                // which makes one per key. A prefetch writes the host parts
                // of the group that asked for it; a put writes none.
                let kind = if *remote { "put" } else { "prefetch" };
                for copy in copies {
                    if let Target::Entry { key, .. } = &copy.to {
                        // A prefetch's session, whose group's host parts it
                        // writes; a put's is none of the model's.
                        let asked = if *remote {
                            "-".to_owned()
                        } else {
                            names.of(pool, task.0.0)
                        };
                        lines.push(format!(
                            "task {kind} h={} expired={expired} k={} parts={} s={asked}",
                            unit(&copy.write),
                            key.hex(),
                            bits(copy.memory)
                        ));
                    }
                }
            }
            _ => lines.extend(
                copies
                    .iter()
                    .map(|copy| format!("copy {}", copy_line(task.0.0, copy))),
            ),
        }
    }
    // A fact is stated once: two releases of the same blocks at the same
    // stamp are one line, as they are one release of the model.
    lines.sort_unstable();
    lines.dedup();
    lines.join("\n")
}

/// A trace's records read back: the inverse of [`Input::encode`].
impl Input {
    /// The input of a record of the kind `kind` with the body `body`.
    pub(super) fn decode(kind: u8, mut body: &[u8]) -> Result<Self, String> {
        let body = &mut body;
        let input = match kind {
            IMAGE => Self::Image(rest(body)),
            OPEN => {
                let client = u32::from_le_bytes(field(body)?);
                let open = rmp_serde::from_slice(&rest(body)).map_err(|error| error.to_string())?;
                Self::Open(client, open)
            }
            CLOSE => Self::Close(u32::from_le_bytes(field(body)?)),
            REQUEST => Self::Request(queue(body)?, rest(body)),
            COMPLETED => Self::Completed(
                u32::from_le_bytes(field(body)?),
                GroupId(u8::from_le_bytes(field(body)?)),
                u64::from_le_bytes(field(body)?),
            ),
            TICK => Self::Tick(u64::from_le_bytes(field(body)?)),
            START => Self::Start(task(body)?),
            DONE => Self::Done(task(body)?, u16::from_le_bytes(field(body)?)),
            ENDED => Self::Ended(task(body)?, field::<1>(body)? != [0]),
            PUBLISH => Self::Publish(u64::from_le_bytes(field(body)?)),
            ROLL_BACK => Self::RollBack(u64::from_le_bytes(field(body)?)),
            HOLD => {
                let peer = u64::from_le_bytes(field(body)?);
                let mut keys = Vec::new();
                while !body.is_empty() {
                    keys.push(Key(field(body)?));
                }
                Self::Hold(peer, keys)
            }
            RELEASE_PEER => Self::ReleasePeer(u64::from_le_bytes(field(body)?)),
            GRANT => {
                let segment = segment(body)?;
                let (first, units) = (
                    u32::from_le_bytes(field(body)?),
                    u32::from_le_bytes(field(body)?),
                );
                Self::Grant(named(body)?, segment, first, units)
            }
            WITHDRAW => {
                let segment = segment(body)?;
                Self::Withdraw(named(body)?, segment)
            }
            PEER_COPIES => {
                let task = task(body)?;
                let mut held = Vec::new();
                while !body.is_empty() {
                    held.push(match field::<1>(body)? {
                        [0] => None,
                        _ => Some((field(body)?, field(body)?)),
                    });
                }
                Self::PeerCopies(task, held)
            }
            other => return Err(format!("no input is of the kind {other}")),
        };
        if body.is_empty() {
            Ok(input)
        } else {
            Err("a record has bytes after its last field".into())
        }
    }
}

/// The next `N` bytes of a body.
fn field<const N: usize>(body: &mut &[u8]) -> Result<[u8; N], String> {
    let (head, tail) = body
        .split_first_chunk::<N>()
        .ok_or("a record ends inside a field")?;
    *body = tail;
    Ok(*head)
}

/// The bytes left of a body.
fn rest(body: &mut &[u8]) -> Vec<u8> {
    std::mem::take(body).to_vec()
}

fn queue(body: &mut &[u8]) -> Result<QueueId, String> {
    Ok((
        u32::from_le_bytes(field(body)?),
        u16::from_le_bytes(field(body)?),
    ))
}

fn task(body: &mut &[u8]) -> Result<Task, String> {
    Ok((queue(body)?, RequestId(u64::from_le_bytes(field(body)?))))
}

fn segment(body: &mut &[u8]) -> Result<Segment, String> {
    Ok(Segment {
        epoch: u64::from_le_bytes(field(body)?),
        ordinal: u32::from_le_bytes(field(body)?),
    })
}

fn named(body: &mut &[u8]) -> Result<String, String> {
    String::from_utf8(rest(body)).map_err(|error| error.to_string())
}

/// A record of a file: where it begins, in bytes, its kind and its body.
type Record<'a> = (u64, u8, &'a [u8]);

/// The records of a file.
fn frames(bytes: &[u8]) -> Result<Vec<Record<'_>>, String> {
    let mut found = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let length = bytes.get(at + 1..at + 5).map(|length| {
            u32::from_le_bytes([length[0], length[1], length[2], length[3]]) as usize
        });
        let body = length.and_then(|length| bytes.get(at + 5..at + 5 + length));
        let Some(body) = body else {
            return Err(format!("the record at byte {at} ends early"));
        };
        found.push((at as u64, bytes[at], body));
        at += 5 + body.len();
    }
    Ok(found)
}
