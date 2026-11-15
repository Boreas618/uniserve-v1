//! Round trips of the peer link's messages through their encoding, as the peer link carries them.

use std::net::TcpListener;
use std::time::Duration;

use proto::request::RequestId;
use service::{Connection, Message};

use crate::batch::cache::CacheGroupKind;
use crate::batch::{self, Capacity, ErrorCode, TimingCounters};
use crate::executor::peers::messages::{
    DataOperation, FromPeer, MainProcessRequest, MainProcessResponse, MainProcessResponseError,
    RankOperation, ResponseKind, ToPeer, decode_requests,
};

/// Returns `message` as a receiver decodes it from the payload a sender encoded.
fn round_trip<M: Message>(message: &M) -> M {
    M::decode(&message.encode().unwrap()).unwrap()
}

/// A token group's capacity: prefill and decode on eight slots, over a full-attention and a
/// sliding-window cache group.
fn group_capacity() -> Capacity {
    use crate::batch::{Phase, Shape, Variant};

    let mut rows = [0; Phase::COUNT];
    rows[Phase::Prefill.index()] = 8;
    rows[Phase::Decode.index()] = 8;
    Capacity {
        phases: (1 << Phase::Prefill.index()) | (1 << Phase::Decode.index()),
        slots: 8,
        kv_cache: Some(proto::config::KvCacheConfig::default()),
        cache_groups: vec![
            (CacheGroupKind::Full, 64),
            (CacheGroupKind::SlidingWindow { window: 128 }, 64),
        ],
        seeds: vec![storage::Key([1; 32]), storage::Key([2; 32])],
        buckets: vec![
            Shape::decode(8, Variant::DecodeGreedy),
            Shape::prefill(8, 512, Variant::PrefillOutput),
        ],
        rows,
        max_model_len: 4096,
        denoise_steps: 0,
        bos_token_id: 1,
        eos_token_ids: vec![2, 3],
        feature_bytes: 0,
        encoder_identity: String::new(),
        latent_seed: 0,
        media_bytes: Vec::new(),
        pipeline: None,
        call_window: 2,
    }
}

/// A decode batch of two requests on worker group 2, as the scheduler hands it to its worker.
fn group_batch() -> batch::Batch {
    batch::Batch {
        id: batch::BatchId(7),
        group: storage::runtime::GroupId(2),
        phase: batch::Phase::Decode,
        generation: 3,
        shape: batch::Shape::decode(2, batch::Variant::DecodeSampled),
        requests: vec![RequestId(1), RequestId(2)],
        columns: batch::Columns {
            input_ids: vec![17, 23],
            slots: vec![1, 2],
            ..batch::Columns::default()
        },
        media: Vec::new(),
        new_slots: Vec::new(),
    }
}

/// A release and the give-up of a copy, each stamped by worker group 2.
fn stamped() -> Vec<storage::runtime::Request> {
    use storage::runtime::{GroupId, OwnerId, Request};

    vec![
        Request::Release {
            owner: OwnerId(5),
            class: 1,
            at: 4,
            count: 2,
            after: vec![(GroupId(2), 9)],
        },
        Request::Finish {
            copy: storage::runtime::RequestId(31),
            ok: false,
            after: vec![(GroupId(2), 9)],
        },
    ]
}

/// A session of one device region with one segment, as the scheduler node opened it.
fn session() -> (storage::runtime::Open, storage::runtime::Opened) {
    use storage::runtime::{Open, Opened, Region, Tier, VERSION};

    let segment = storage::Segment {
        epoch: 1,
        ordinal: 0,
    };
    let open = Open {
        version: VERSION,
        epoch: 1,
        trial: false,
        regions: vec![Region {
            name: "kv 0 0011".to_owned(),
            tier: Tier::Device,
            space: "kv 0 0011".to_owned(),
            unit_bytes: 256,
            part_bytes: 128,
            reserved: 1,
            units: 0,
            limit: 129,
            segments: vec![(segment, 0, 129)],
            memory: String::new(),
        }],
        classes: Vec::new(),
        trajectory: 0,
        groups: Vec::new(),
        schedulers: vec![Vec::new()],
        peers: vec!["10.0.0.3:7000".to_owned()],
    };
    let opened = Opened {
        regions: vec![(0, 128)],
        segments: vec![vec![(segment, 0, 129)]],
        limits: vec![129],
        registers: vec![storage::runtime::Registers {
            completed: vec![0],
            capacity: vec![(0, 1)],
        }],
    };
    (open, opened)
}

/// Every operation of the data plane a peer runs.
fn operations() -> Vec<DataOperation> {
    use storage::runtime::GroupId;

    let groups = vec![GroupId(0), GroupId(2)];
    let order = std::collections::BTreeMap::from([("model.layers.0.mlp".to_owned(), vec![1, 0])]);
    vec![
        DataOperation::ReleaseSegments {
            groups: groups.clone(),
            region: 4,
            from: 3,
        },
        DataOperation::CreateSegments {
            groups: groups.clone(),
            region: 4,
            count: Some(2),
        },
        DataOperation::CreateSegments {
            groups: groups.clone(),
            region: 5,
            count: None,
        },
        DataOperation::SegmentTables {
            groups: groups.clone(),
            region: 4,
        },
        DataOperation::FittingSegments {
            groups: groups.clone(),
            region: 5,
        },
        DataOperation::WarmSegments {
            groups: groups.clone(),
            region: 4,
        },
        DataOperation::DeviceRooms { groups, region: 5 },
        DataOperation::Room,
        DataOperation::GraphBytes,
        DataOperation::Sleep,
        DataOperation::SuspendGraphs,
        DataOperation::RestoreGraphs,
        DataOperation::SetKernels,
        DataOperation::ReserveServing,
        DataOperation::Pair {
            checkpoint: "/models/next".to_owned(),
            experts: order.clone(),
            rearranged: true,
        },
        DataOperation::Load,
        DataOperation::Stage,
        DataOperation::Discard,
        DataOperation::UpdateWeights,
        DataOperation::CommitExperts { order },
        DataOperation::ServingOrder,
        DataOperation::Prepared { generation: 4 },
        DataOperation::Uncompared,
        DataOperation::Rank(RankOperation::LoadAdapter {
            slot: 1,
            path: "/adapters/a".into(),
        }),
    ]
}

/// One fixture per kind of request, and a data frame of every operation.
fn request_fixtures() -> Vec<MainProcessRequest> {
    use storage::runtime::{Completion, GroupId, OwnerId, Request};

    let allocate = Request::Allocate {
        owner: OwnerId(5),
        class: 1,
        at: 4,
        count: 1,
        units: 0,
    };
    let allocated = Completion::Allocated {
        runs: vec![(0, 70, 1)],
        evicted: Vec::new(),
    };
    let (open, opened) = session();
    let mut fixtures = vec![
        MainProcessRequest::session(&open, &opened),
        MainProcessRequest::submit(12, false, group_batch()),
        MainProcessRequest::submit(13, true, group_batch()),
        MainProcessRequest::shadow(14, group_batch()),
        MainProcessRequest::map(
            GroupId(2),
            3,
            storage::runtime::RequestId(30),
            &allocate,
            &allocated,
        ),
        MainProcessRequest::stamp(GroupId(2), &stamped()),
        MainProcessRequest::Switch { generation: 2 },
        MainProcessRequest::Commit { generation: 2 },
        MainProcessRequest::Rollback { generation: 2 },
        MainProcessRequest::close(),
    ];
    fixtures.extend(operations().into_iter().map(MainProcessRequest::data));
    fixtures
}

fn response_fixtures() -> Vec<MainProcessResponse> {
    let (open, opened) = session();
    let runs = vec![storage::protocol::UnitRun {
        field: "prefix_cache.key.values/0".to_owned(),
        key: storage::Key([3; 32]),
        location: storage::Location::Device { gpu: [5; 16] },
        first: 0,
        count: 129,
        offset: 4096,
        bytes: 64,
    }];
    vec![
        MainProcessResponse::Placed {
            endpoint: "10.0.0.3:7000".to_owned(),
            created: vec![(open.regions[0].name.clone(), opened.segments[0].clone())],
            places: vec![(open.regions[0].name.clone(), "a/0,b/0".to_owned(), 1, runs)],
            capacities: vec![(storage::runtime::GroupId(2), group_capacity())],
        },
        MainProcessResponse::Result {
            result: batch::BatchOutput {
                timing: TimingCounters {
                    queued_us: 3,
                    device_us: 40,
                    copy_us: 0,
                    host_us: 2,
                },
                ..batch::BatchOutput::default()
            },
            requests: Vec::new(),
            ended: 12,
        },
        MainProcessResponse::result(batch::BatchOutput::default(), &stamped(), 14),
        MainProcessResponse::Operated {
            reply: rmp_serde::to_vec(&[3_u32]).unwrap(),
        },
        MainProcessResponse::Error {
            error: MainProcessResponseError {
                message: "device fault on decode".into(),
                code: Some(ErrorCode::ComputeError),
                fatal: false,
                phase: Some("execute".into()),
                route: Some("und.decode".into()),
            },
            requests: Vec::new(),
            ended: 0,
        },
        MainProcessResponse::failed("a device fault".into(), &stamped()[1..], 9),
    ]
}

/// `request` as the peer reads it from the frame the scheduler node sent it in as `id`.
fn request_round_trip(id: u64, request: MainProcessRequest) -> MainProcessRequest {
    let frame = round_trip(&ToPeer { id, request });
    assert_eq!(frame.id, id);
    frame.request
}

/// `response` as the scheduler node reads it from the frame a peer answered request `id` with.
fn response_round_trip(id: u64, response: MainProcessResponse) -> MainProcessResponse {
    let frame = round_trip(&FromPeer { id, response });
    assert_eq!(frame.id, id);
    frame.response
}

#[test]
fn every_request_kind_round_trips_through_ipc() {
    for (id, request) in (1..).zip(request_fixtures()) {
        assert_eq!(request_round_trip(id, request.clone()), request);
    }
}

#[test]
fn every_response_kind_round_trips_through_ipc() {
    let fixtures = response_fixtures();
    for kind in [
        ResponseKind::Placed,
        ResponseKind::Result,
        ResponseKind::Operated,
        ResponseKind::Error,
    ] {
        assert!(
            fixtures.iter().any(|response| response.kind() == kind),
            "no fixture covers response kind {kind:?}"
        );
    }
    for (id, response) in (1..).zip(fixtures) {
        assert_eq!(
            response_round_trip(id, response.clone()),
            response,
            "round trip for {:?}",
            response.kind()
        );
    }
}

#[test]
fn the_buffer_pool_s_requests_read_back_from_the_frames_that_carry_them() {
    let MainProcessRequest::Stamp { requests, .. } = request_round_trip(
        1,
        MainProcessRequest::stamp(storage::runtime::GroupId(2), &stamped()),
    ) else {
        panic!("a stamp frame reads back as a stamp frame");
    };
    assert_eq!(decode_requests(&requests).unwrap(), stamped());

    let response = response_round_trip(
        2,
        MainProcessResponse::result(batch::BatchOutput::default(), &stamped(), 14),
    );
    let MainProcessResponse::Result {
        requests, ended, ..
    } = response
    else {
        panic!("a result frame reads back as a result frame");
    };
    assert_eq!(decode_requests(&requests).unwrap(), stamped());
    assert_eq!(ended, 14);

    // A failure returns the `Finish` of the copies its batch started, readable as requests.
    let response = response_round_trip(
        3,
        MainProcessResponse::failed("a device fault".into(), &stamped()[1..], 9),
    );
    let MainProcessResponse::Error {
        requests, ended, ..
    } = response
    else {
        panic!("an error frame reads back as an error frame");
    };
    assert_eq!(decode_requests(&requests).unwrap(), &stamped()[1..]);
    assert_eq!(ended, 9);

    // A record that is not a request of the buffer pool's interface is an error to its reader.
    assert!(decode_requests(&[vec![0; 4]]).is_err());
}

#[test]
fn a_message_decodes_only_as_the_message_it_is() {
    let request = ToPeer {
        id: 1,
        request: MainProcessRequest::close(),
    }
    .encode()
    .unwrap();
    let refused = FromPeer::decode(&request).unwrap_err();
    assert_eq!(refused.kind(), std::io::ErrorKind::InvalidData);

    // A message cut short, as a frame of another build may be, is refused: the frame's length
    // field counts the shortened message, so the connection receives it whole and its decoding
    // fails.
    let mut cut = FromPeer {
        id: 1,
        response: MainProcessResponse::result(batch::BatchOutput::default(), &[], 0),
    }
    .encode()
    .unwrap();
    cut.pop();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut connection =
        Connection::connect(&service::Endpoint::tcp(listener.local_addr().unwrap())).unwrap();
    let (peer, _) = listener.accept().unwrap();
    for mut segment in service::frame::segments(&cut, Vec::new()).unwrap() {
        while !segment.sent() {
            segment.send(std::os::fd::AsFd::as_fd(&peer)).unwrap();
        }
    }

    let received = connection.recv::<FromPeer>(Duration::from_secs(5));
    assert!(
        received
            .as_ref()
            .is_err_and(|error| error.kind() == std::io::ErrorKind::InvalidData),
        "{received:?}"
    );
}
