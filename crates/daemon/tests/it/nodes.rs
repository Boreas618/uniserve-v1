//! A deployment of two nodes: a leader and a member daemon in this process,
//! joined over loopback. An engine connects to the leader only, and the
//! leader routes what it asks for the member's node to the member. A test of
//! what a node claims at its host's Storage Server runs each daemon in a
//! process of its own, whose runtime directory is its host's ([`Hosts`]).

use daemon::engine::control::{BuildServer, EngineControl, prepare_epoch};
use daemon::{Daemon, DaemonOptions};
use proto::process::{
    Claim, Exit, NodesQuery, PrepareModules, Scope, ThrottleTransfers, reply, request,
};

use crate::support::{
    BuildServerStandIn, Engine, StorageServerStandIn, launch, module_set, package, sleeper,
};

/// A leader on host `leader` and a member on host `member`, each host with its stand-in Build
/// Server.
struct Deployment {
    // The member is dropped first: dropping the leader first ends the
    // member's service, which is what `the_members_processes_end_with_the_leader`
    // observes on its own.
    member: Option<Daemon>,
    leader: Option<Daemon>,
    address: String,
    _build_servers: Vec<BuildServerStandIn>,
    _directories: Vec<tempfile::TempDir>,
}

impl Deployment {
    fn start() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let member = BuildServerStandIn::serve(&directory.path().join("member"), false, 0);
        let mut deployment = Self::with_member_build_server(&member);
        deployment._build_servers.push(member);
        deployment._directories.push(directory);
        deployment
    }

    /// A deployment whose member's host is served by the Build Server `member`.
    fn with_member_build_server(member: &BuildServerStandIn) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let leader = BuildServerStandIn::serve(&directory.path().join("leader"), false, 0);
        // The leader's join listener, bound on a port the kernel picks; the member joins at the
        // endpoint it bound.
        let (join, endpoint) = Daemon::bind_join(&"127.0.0.1:0".parse().unwrap()).unwrap();
        let options = |node_rank: u32, host: &str| DaemonOptions {
            deployment: crate::support::DEPLOYMENT.to_owned(),
            runtime: if host == "member" {
                member.runtime()
            } else {
                leader.runtime()
            },
            nnodes: 2,
            node_rank,
            host: host.to_owned(),
            dist_init_addr: Some(endpoint.clone()),
            socket: directory.path().join(format!("{host}.sock")),
            join_timeout: daemon::JOIN_TIMEOUT,
            build_server: if host == "member" {
                member.settings()
            } else {
                leader.settings()
            },
            storage: storage::Settings::default(),
        };

        // The leader's start returns once every member joined, and a
        // member's once it joined, so they start together.
        let leader_daemon = std::thread::spawn({
            let options = options(0, "leader");
            move || Daemon::start(options, Some(join)).unwrap()
        });
        let member_daemon = Daemon::start(options(1, "member"), None).unwrap();
        let leader_daemon = leader_daemon.join().unwrap();
        assert_eq!(
            member_daemon.address(),
            None,
            "only the leader serves engines"
        );
        Self {
            address: leader_daemon.address().unwrap().to_owned(),
            member: Some(member_daemon),
            leader: Some(leader_daemon),
            _build_servers: vec![leader],
            _directories: vec![directory],
        }
    }
}

#[test]
fn the_leader_routes_each_node_s_requests_to_its_daemon() {
    let deployment = Deployment::start();
    let (mut engine, welcome) = Engine::hello(&deployment.address, proto::ENGINE_INTERFACE_VERSION);
    assert_eq!(welcome.refusal, None);
    let nodes: Vec<(u32, &str)> = welcome
        .nodes
        .iter()
        .map(|node| (node.node_rank, node.host.as_str()))
        .collect();
    assert_eq!(nodes, [(0, "leader"), (1, "member")]);

    // The member's exit comes back through the leader, naming the node.
    engine.spawn(launch("member", "group", 1, 2, package::FAILS));
    assert_eq!(
        engine.exit(),
        Exit {
            host: "member".into(),
            group: "group".into(),
            rank: 1,
            generation: 2,
            status: "exit status: 1".into(),
        }
    );

    // A socket other nodes connect to is named by the address the leader
    // reaches the member at.
    let reserved = engine.reserve("member", "group", true);
    assert!(
        reserved.address.starts_with("127.0.0.1:"),
        "{reserved:?} names the member's address"
    );

    let refused = engine.request(request::Kind::Spawn(sleeper("elsewhere", "group", 0, 0)));
    assert!(
        refused.error.contains("elsewhere"),
        "the refusal names the host no daemon runs on: {refused:?}"
    );
}

#[test]
fn a_stop_and_an_engine_s_close_reach_every_node() {
    let deployment = Deployment::start();
    let mut engine = Engine::connect(&deployment.address);
    let on_leader = engine.start(sleeper("leader", "stopped", 0, 0));
    let on_member = engine.start(sleeper("member", "stopped", 1, 0));

    // The stop is answered once both nodes stopped the group (each killed and reaped its
    // processes), and the engine's next request is answered in turn.
    engine.stop("stopped");
    assert!(!on_leader.runs() && !on_member.runs());
    let kept = engine.start(sleeper("member", "kept", 1, 0));

    drop(engine);
    kept.wait();
}

#[test]
fn the_members_processes_end_with_the_leader() {
    let mut deployment = Deployment::start();
    let mut engine = Engine::connect(&deployment.address);
    let process = engine.start(sleeper("member", "group", 1, 0));

    drop(deployment.leader.take());
    // The member's service ends when the leader's connection closes.
    deployment.member.take().unwrap().wait().unwrap();
    assert!(!process.runs(), "a member's process outlived the leader");
}

#[test]
fn a_node_joins_a_running_deployment_and_the_next_engine_runs_processes_there() {
    let directory = tempfile::tempdir().unwrap();
    let build_server = BuildServerStandIn::serve(directory.path(), false, 0);
    let (join, endpoint) = Daemon::bind_join(&"127.0.0.1:0".parse().unwrap()).unwrap();
    let options = |nnodes: u32, node_rank: u32, host: &str| DaemonOptions {
        deployment: crate::support::DEPLOYMENT.to_owned(),
        runtime: build_server.runtime(),
        nnodes,
        node_rank,
        host: host.to_owned(),
        dist_init_addr: Some(endpoint.clone()),
        socket: directory.path().join(format!("{host}.sock")),
        join_timeout: daemon::JOIN_TIMEOUT,
        build_server: build_server.settings(),
        storage: storage::Settings::default(),
    };

    // A leader of one node that names its address serves at once.
    let leader = Daemon::start(options(1, 0, "leader"), Some(join)).unwrap();
    let address = leader.address().unwrap().to_owned();
    let (mut serving, welcome) = Engine::hello(&address, proto::ENGINE_INTERFACE_VERSION);
    assert_eq!(welcome.nodes.len(), 1);

    // A node joins while an engine is served; a second node of the same
    // rank is refused.
    let member = Daemon::start(options(2, 1, "member"), None).unwrap();
    let error = Daemon::start(options(2, 1, "again"), None).err().unwrap();
    assert!(
        format!("{error:#}").contains("node rank 1 is taken"),
        "{error:#}"
    );

    // The engine served before learns the node by asking the leader, and
    // throttles its transfers there, which a node that holds no claim of
    // the deployment takes as nothing to do.
    let known = serving.request(request::Kind::Nodes(NodesQuery {}));
    let Some(reply::Kind::Nodes(known)) = known.kind else {
        panic!("{known:?}");
    };
    let hosts: Vec<&str> = known.nodes.iter().map(|node| node.host.as_str()).collect();
    assert_eq!(hosts, ["leader", "member"]);
    let throttled = serving.request(request::Kind::ThrottleTransfers(ThrottleTransfers {
        host: "member".into(),
        deployment: "model".into(),
        halve: true,
    }));
    assert_eq!(throttled.error, "");

    // The next engine learns the node and runs a process there, whose exit
    // comes back through the leader.
    let (mut engine, welcome) = Engine::hello(&address, proto::ENGINE_INTERFACE_VERSION);
    let nodes: Vec<(u32, &str)> = welcome
        .nodes
        .iter()
        .map(|node| (node.node_rank, node.host.as_str()))
        .collect();
    assert_eq!(nodes, [(0, "leader"), (1, "member")]);
    engine.spawn(launch("member", "group", 1, 0, package::FAILS));
    assert_eq!(engine.exit().host, "member");

    drop(engine);
    drop(serving);
    drop(member);
    drop(leader);
}

/// The leader has a node prepare an epoch's module set at the node's own Build Server, and the
/// node answers with the set as that Build Server prepared it. While the Build Server builds, which
/// takes minutes for a store that misses the engine's executable, the node serves its other
/// requests: a spawn there is answered before the preparation is.
#[test]
fn a_node_prepares_the_module_set_at_its_build_server_and_serves_meanwhile() {
    let directory = tempfile::tempdir().unwrap();
    let modules = BuildServerStandIn::serve(&directory.path().join("member-modules"), true, 0);
    let deployment = Deployment::with_member_build_server(&modules);
    let mut engine = Engine::connect(&deployment.address);

    let set = module_set();
    let preparing = engine.send(request::Kind::PrepareModules(PrepareModules {
        host: "member".into(),
        modules: Some((&set).into()),
    }));
    modules.asked();
    let process = engine.start(sleeper("member", "group", 1, 0));
    assert!(process.runs(), "the spawn answered while the node prepares");

    modules.release();
    let prepared = engine.reply(preparing);
    assert_eq!(prepared.error, "");
    let Some(reply::Kind::ModulesPrepared(prepared)) = prepared.kind else {
        panic!("the preparation answered no module set: {prepared:?}");
    };
    let prepared = proto::module::ModuleSet::try_from(prepared.modules.unwrap()).unwrap();
    assert_eq!(prepared, set);
    engine.stop("group");
}

/// A node whose daemon is lost while its Build Server prepares never answers: the leader fails
/// the preparation at once, naming the node.
#[test]
fn a_preparation_fails_naming_its_node_when_the_node_is_lost() {
    let directory = tempfile::tempdir().unwrap();
    let modules = BuildServerStandIn::serve(&directory.path().join("member-modules"), true, 0);
    let mut deployment = Deployment::with_member_build_server(&modules);
    let mut engine = Engine::connect(&deployment.address);

    let preparing = engine.send(request::Kind::PrepareModules(PrepareModules {
        host: "member".into(),
        modules: Some((&module_set()).into()),
    }));
    modules.asked();
    drop(deployment.member.take());

    let failed = engine.reply(preparing);
    assert!(
        failed.error.contains("node 1 (member)") && failed.kind.is_none(),
        "{failed:?}"
    );
}

/// An epoch's module set is prepared at the Build Server of every host the epoch runs on, the
/// leader's own at its socket and the member's through the member's daemon, at once: each Build
/// Server is asked before either answers. The set is the one the leader's Build Server prepared.
#[test]
fn an_epoch_is_prepared_at_every_host_it_runs_on_at_once() {
    let directory = tempfile::tempdir().unwrap();
    let leader_modules =
        BuildServerStandIn::serve(&directory.path().join("leader-modules"), true, 0);
    let member_modules =
        BuildServerStandIn::serve(&directory.path().join("member-modules"), true, 0);
    let deployment = Deployment::with_member_build_server(&member_modules);
    let control = EngineControl::new(
        &directory.path().join("requests.sock"),
        deployment.address.clone(),
        "leader".into(),
        "deployment".into(),
        leader_modules.socket(),
    )
    .unwrap();
    let local = BuildServer {
        host: "leader".into(),
        socket: leader_modules.socket(),
        targets: Vec::new(),
    };
    let set = module_set();
    let hosts = ["leader".to_owned(), "member".to_owned()];

    let prepared = std::thread::scope(|scope| {
        let preparing = scope.spawn(|| prepare_epoch(&local, Some(&control), &hosts, &set));
        leader_modules.asked();
        member_modules.asked();
        leader_modules.release();
        member_modules.release();
        preparing.join().unwrap()
    });
    assert_eq!(prepared.unwrap(), set);
}

/// A host whose Build Server prepares other modules than the leader's fails the epoch's
/// preparation, naming the host: every host runs the revisions the epoch records.
#[test]
fn a_host_that_prepares_other_modules_fails_the_preparation() {
    let directory = tempfile::tempdir().unwrap();
    let leader_modules =
        BuildServerStandIn::serve(&directory.path().join("leader-modules"), false, 0);
    let member_modules =
        BuildServerStandIn::serve(&directory.path().join("member-modules"), false, 1);
    let deployment = Deployment::with_member_build_server(&member_modules);
    let control = EngineControl::new(
        &directory.path().join("requests.sock"),
        deployment.address.clone(),
        "leader".into(),
        "deployment".into(),
        leader_modules.socket(),
    )
    .unwrap();
    let local = BuildServer {
        host: "leader".into(),
        socket: leader_modules.socket(),
        targets: Vec::new(),
    };

    let refused = prepare_epoch(
        &local,
        Some(&control),
        &["leader".to_owned(), "member".to_owned()],
        &module_set(),
    )
    .unwrap_err();
    assert!(format!("{refused:#}").contains("member"), "{refused:#}");
}

/// A deployment of two hosts, `leader` and `member`, each node's daemon in a process of its own
/// (the `daemon-host` binary) whose runtime directory is its host's, where a Storage Server
/// stand-in and a Build Server stand-in serve: a claim and a publication at a node reach that
/// host's Storage Server alone, as on two machines.
struct Hosts {
    leader: StorageServerStandIn,
    member: StorageServerStandIn,
    /// The address engines reach the leader at.
    address: String,
    /// The node processes, the member's last.
    processes: Vec<std::process::Child>,
    /// Each host's Build Server, the leader's first.
    build_servers: Vec<BuildServerStandIn>,
    directory: tempfile::TempDir,
}

impl Hosts {
    fn start() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let runtime = |host: &str| {
            let runtime = directory.path().join(host);
            std::fs::create_dir_all(&runtime).unwrap();
            runtime
        };
        let store = |host: &str| {
            let runtime = service::RuntimeDirectory::within(&runtime(host)).unwrap();
            StorageServerStandIn::serve(
                storage::socket(&runtime, crate::support::DEPLOYMENT).unwrap(),
            )
        };
        let build = |host: &str| BuildServerStandIn::serve(&runtime(host), false, 0);
        let (leader, member) = (store("leader"), store("member"));
        let build_servers = vec![build("leader"), build("member")];

        // Each line a node process writes, as it writes it.
        let node = |node_rank: u32, host: &str, join: &str| {
            let mut process = std::process::Command::new(env!("CARGO_BIN_EXE_daemon-host"))
                .arg("--socket")
                .arg(directory.path().join(format!("{host}.sock")))
                .args(["--host", host, "--nnodes", "2"])
                .args(["--node-rank", &node_rank.to_string(), "--join", join])
                .env("XDG_RUNTIME_DIR", runtime(host))
                .stdout(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            let lines =
                std::io::BufRead::lines(std::io::BufReader::new(process.stdout.take().unwrap()));
            (process, lines)
        };
        let (leader_process, mut leader_lines) = node(0, "leader", "127.0.0.1:0");
        let mut next = move |expected: &str| {
            let line = leader_lines.next().unwrap().unwrap();
            line.strip_prefix(expected)
                .unwrap_or_else(|| panic!("the leader wrote {line:?}, not {expected:?}"))
                .to_owned()
        };
        let join = next("joining ");
        let (member_process, mut member_lines) = node(1, "member", &join);
        assert_eq!(next("serving"), "", "the leader did not start");
        let address = next("address ");
        assert_eq!(
            member_lines.next().unwrap().unwrap(),
            "serving",
            "the member did not start"
        );

        Self {
            leader,
            member,
            address,
            processes: vec![leader_process, member_process],
            build_servers,
            directory,
        }
    }

    /// The engine control of the leader's daemon, for the deployment the engines claim.
    fn control(&self) -> EngineControl {
        EngineControl::new(
            &self.directory.path().join("requests.sock"),
            self.address.clone(),
            "leader".into(),
            "deployment".into(),
            self.build_servers[0].socket(),
        )
        .unwrap()
    }
}

impl Drop for Hosts {
    fn drop(&mut self) {
        // The member first: its service ends when the leader's connection closes, which
        // `the_members_processes_end_with_the_leader` observes on its own.
        for process in self.processes.iter_mut().rev() {
            let _ = process.kill();
            let _ = process.wait();
        }
    }
}

/// Claims epoch `epoch` of the deployment at `host`'s Storage Server for the deployment's rank
/// `rank`, as the epoch's engine there does before it maps storage.
fn claim(engine: &mut Engine, host: &str, epoch: u64, rank: u32) {
    let claimed = engine.request(request::Kind::Claim(Claim {
        host: host.into(),
        deployment: "deployment".into(),
        scopes: vec![Scope {
            group: "model".into(),
            rank,
        }],
        storage: None,
        epoch,
        retire: Vec::new(),
    }));
    assert!(
        matches!(claimed.kind, Some(reply::Kind::Claimed(_))),
        "{host} did not claim epoch {epoch}: {claimed:?}"
    );
}

/// An epoch is published at the nodes that claimed it, and only there. The deployment's first
/// epoch places every rank on the leader, so the member, which holds no rank, never claimed and
/// is not asked; a later epoch that places ranks on the member is claimed there too and published
/// at both. A node asked to publish an epoch it holds no claim of refuses, naming itself.
#[test]
fn an_epoch_is_published_at_the_nodes_that_claimed_it() {
    let hosts = Hosts::start();
    let mut engine = Engine::connect(&hosts.address);
    let control = hosts.control();

    claim(&mut engine, "leader", 1, 0);
    control.publish_epochs(&[1], &["leader".into()]).unwrap();
    assert_eq!(hosts.leader.published(), [vec![1]]);
    assert!(hosts.member.published().is_empty());

    let refused = control
        .publish_epochs(&[1], &["member".into()])
        .unwrap_err();
    assert!(
        format!("{refused:#}").contains("publishing epochs at member: this node holds no claim"),
        "{refused:#}"
    );

    claim(&mut engine, "leader", 2, 0);
    claim(&mut engine, "member", 2, 1);
    control
        .publish_epochs(&[2], &["leader".into(), "member".into()])
        .unwrap();
    assert_eq!(hosts.leader.published(), [vec![1], vec![2]]);
    assert_eq!(hosts.member.published(), [vec![2]]);
}
