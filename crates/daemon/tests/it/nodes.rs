//! A deployment of two nodes: a leader and a member daemon in this process,
//! joined over loopback. An engine connects to the leader only, and the
//! leader routes what it asks for the member's node to the member.

use daemon::engine::control::{BuildServer, EngineControl, PrepareSteps, prepare_epoch};
use daemon::{Daemon, DaemonOptions};
use proto::process::{Exit, NodesQuery, PrepareModules, ThrottleTransfers, reply, request};

use crate::support::{
    BuildServerStandIn, Engine, TIMEOUT, alive, eventually, launch, module_set, sleeper,
};

/// A leader on host `leader` and a member on host `member`.
struct Deployment {
    // The member is dropped first: dropping the leader first ends the
    // member's service, which is what `the_members_processes_end_with_the_leader`
    // observes on its own.
    member: Option<Daemon>,
    leader: Option<Daemon>,
    address: String,
    _directory: tempfile::TempDir,
}

impl Deployment {
    fn start() -> Self {
        Self::with_member_build_server(build_server::command::Settings::default())
    }

    /// A deployment whose member's host is served by the Build Server `settings` names.
    fn with_member_build_server(settings: build_server::command::Settings) -> Self {
        let directory = tempfile::tempdir().unwrap();
        // The leader's join listener, bound on a port the kernel picks; the member joins at the
        // endpoint it bound.
        let (join, endpoint) = Daemon::bind_join(&"127.0.0.1:0".parse().unwrap()).unwrap();
        let options = |node_rank: u32, host: &str| DaemonOptions {
            nnodes: 2,
            node_rank,
            host: host.to_owned(),
            dist_init_addr: Some(endpoint.clone()),
            socket: directory.path().join(format!("{host}.sock")),
            executable: "uniserve".into(),
            join_timeout: TIMEOUT,
            devices: || Ok(Vec::new()),
            build_server: if host == "member" {
                settings.clone()
            } else {
                build_server::command::Settings::default()
            },
            storage: storage::Settings::default(),
        };

        // The leader's start returns once every member joined, and a
        // member's once it joined, so they start together.
        let leader = std::thread::spawn({
            let options = options(0, "leader");
            move || Daemon::start(options, Some(join)).unwrap()
        });
        let member = Daemon::start(options(1, "member"), None).unwrap();
        let leader = leader.join().unwrap();
        assert_eq!(member.address(), None, "only the leader serves engines");
        Self {
            address: leader.address().unwrap().to_owned(),
            member: Some(member),
            leader: Some(leader),
            _directory: directory,
        }
    }
}

#[test]
fn the_leader_routes_each_node_s_requests_to_its_daemon() {
    let deployment = Deployment::start();
    let (mut engine, welcome) = Engine::hello(&deployment.address, proto::ENGINE_INTERFACE_VERSION);
    assert_eq!(welcome.refusal, "");
    let nodes: Vec<(u32, &str)> = welcome
        .nodes
        .iter()
        .map(|node| (node.node_rank, node.host.as_str()))
        .collect();
    assert_eq!(nodes, [(0, "leader"), (1, "member")]);

    // The member's exit comes back through the leader, naming the node.
    engine.spawn(launch("member", "group", 1, 2, "/bin/false", &[]));
    assert_eq!(
        engine.exit(),
        Exit {
            host: "member".into(),
            group: "group".into(),
            rank: 1,
            generation: 2,
            status: "exit status: 1".into(),
            last_line: String::new(),
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
    let on_leader = engine.spawn(sleeper("leader", "stopped", 0, 0));
    let on_member = engine.spawn(sleeper("member", "stopped", 1, 0));

    // The stop is answered once both nodes stopped the group, and the
    // engine's next request is answered in turn.
    engine.stop("stopped");
    assert!(eventually(|| !alive(on_leader) && !alive(on_member)));
    let kept = engine.spawn(sleeper("member", "kept", 1, 0));

    drop(engine);
    assert!(eventually(|| !alive(kept)), "a process outlived its engine");
}

#[test]
fn the_members_processes_end_with_the_leader() {
    let mut deployment = Deployment::start();
    let mut engine = Engine::connect(&deployment.address);
    let pid = engine.spawn(sleeper("member", "group", 1, 0));

    drop(deployment.leader.take());
    // The member's service ends when the leader's connection closes.
    deployment.member.take().unwrap().wait().unwrap();
    assert!(!alive(pid), "a member's process outlived the leader");
}

#[test]
fn a_node_joins_a_running_deployment_and_the_next_engine_runs_processes_there() {
    let directory = tempfile::tempdir().unwrap();
    let (join, endpoint) = Daemon::bind_join(&"127.0.0.1:0".parse().unwrap()).unwrap();
    let options = |nnodes: u32, node_rank: u32, host: &str| DaemonOptions {
        nnodes,
        node_rank,
        host: host.to_owned(),
        dist_init_addr: Some(endpoint.clone()),
        socket: directory.path().join(format!("{host}.sock")),
        executable: "uniserve".into(),
        join_timeout: TIMEOUT,
        devices: || Ok(Vec::new()),
        build_server: build_server::command::Settings::default(),
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
    engine.spawn(launch("member", "group", 1, 0, "/bin/false", &[]));
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
    let modules =
        BuildServerStandIn::serve(directory.path().join("member/build-server.sock"), true, 0);
    let deployment = Deployment::with_member_build_server(modules.settings());
    let mut engine = Engine::connect(&deployment.address);

    let set = module_set();
    let preparing = engine.send(request::Kind::PrepareModules(PrepareModules {
        host: "member".into(),
        modules: serde_json::to_vec(&set).unwrap(),
    }));
    modules.asked();
    let pid = engine.spawn(sleeper("member", "group", 1, 0));
    assert!(alive(pid), "the spawn answered while the node prepares");

    modules.release();
    let prepared = engine.reply(preparing);
    assert_eq!(prepared.error, "");
    let Some(reply::Kind::ModulesPrepared(prepared)) = prepared.kind else {
        panic!("the preparation answered no module set: {prepared:?}");
    };
    let prepared: proto::module::ModuleSet = serde_json::from_slice(&prepared.modules).unwrap();
    assert_eq!(prepared, set);
    engine.stop("group");
}

/// A node whose daemon is lost while its Build Server prepares never answers: the leader fails
/// the preparation at once, naming the node.
#[test]
fn a_preparation_fails_naming_its_node_when_the_node_is_lost() {
    let directory = tempfile::tempdir().unwrap();
    let modules =
        BuildServerStandIn::serve(directory.path().join("member/build-server.sock"), true, 0);
    let mut deployment = Deployment::with_member_build_server(modules.settings());
    let mut engine = Engine::connect(&deployment.address);

    let preparing = engine.send(request::Kind::PrepareModules(PrepareModules {
        host: "member".into(),
        modules: serde_json::to_vec(&module_set()).unwrap(),
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
        BuildServerStandIn::serve(directory.path().join("leader/build-server.sock"), true, 0);
    let member_modules =
        BuildServerStandIn::serve(directory.path().join("member/build-server.sock"), true, 0);
    let deployment = Deployment::with_member_build_server(member_modules.settings());
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
        let preparing = scope
            .spawn(|| prepare_epoch(&local, Some(&control), &hosts, &set, &PrepareSteps::new()));
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
        BuildServerStandIn::serve(directory.path().join("leader/build-server.sock"), false, 0);
    let member_modules =
        BuildServerStandIn::serve(directory.path().join("member/build-server.sock"), false, 1);
    let deployment = Deployment::with_member_build_server(member_modules.settings());
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
        &PrepareSteps::new(),
    )
    .unwrap_err();
    assert!(format!("{refused:#}").contains("member"), "{refused:#}");
}
