//! A deployment of two nodes: a leader and a member daemon in this process,
//! joined over loopback. An engine connects to the leader only, and the
//! leader routes what it asks for the member's node to the member.

use daemon::{Daemon, DaemonOptions};
use proto::process::{Exit, NodesQuery, ThrottleTransfers, reply, request};

use crate::support::{Engine, TIMEOUT, alive, eventually, launch, sleeper};

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
        let directory = tempfile::tempdir().unwrap();
        // A free port for the leader's address: the leader binds it on every
        // interface once this probe released it.
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let options = |node_rank: u32, host: &str| DaemonOptions {
            nnodes: 2,
            node_rank,
            host: host.to_owned(),
            dist_init_addr: Some(format!("127.0.0.1:{port}")),
            socket: directory.path().join(format!("{host}.sock")),
            executable: "uniserve".into(),
            join_timeout: TIMEOUT,
        };

        // The leader's start returns once every member joined, and a
        // member's once it joined, so they start together.
        let leader = std::thread::spawn({
            let options = options(0, "leader");
            move || Daemon::start(options).unwrap()
        });
        let member = Daemon::start(options(1, "member")).unwrap();
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
    let (mut engine, welcome) = Engine::hello(&deployment.address, proto::PROTOCOL_MAJOR);
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
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let options = |nnodes: u32, node_rank: u32, host: &str| DaemonOptions {
        nnodes,
        node_rank,
        host: host.to_owned(),
        dist_init_addr: Some(format!("127.0.0.1:{port}")),
        socket: directory.path().join(format!("{host}.sock")),
        executable: "uniserve".into(),
        join_timeout: TIMEOUT,
    };

    // A leader of one node that names its address serves at once.
    let leader = Daemon::start(options(1, 0, "leader")).unwrap();
    let address = leader.address().unwrap().to_owned();
    let (mut serving, welcome) = Engine::hello(&address, proto::PROTOCOL_MAJOR);
    assert_eq!(welcome.nodes.len(), 1);

    // A node joins while an engine is served; a second node of the same
    // rank is refused.
    let member = Daemon::start(options(2, 1, "member")).unwrap();
    let error = Daemon::start(options(2, 1, "again")).err().unwrap();
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
    let (mut engine, welcome) = Engine::hello(&address, proto::PROTOCOL_MAJOR);
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
