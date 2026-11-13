//! Round trips of the peer link's messages through a [`Frame`], as a channel carries them.

use std::io::Write as _;
use std::net::TcpListener;
use std::time::{Duration, Instant};

use ipc::*;
use ipc::{CacheGroupKind, RequestId};
use serde::Serialize;

/// Returns `message` as a receiver decodes it from the frame a sender made.
fn round_trip<M: Serialize + serde::de::DeserializeOwned>(message: &M) -> M {
    Frame::encode(message).unwrap().decode().unwrap()
}

/// A token group's capacity: prefill and decode on eight slots, over a full-attention and a
/// sliding-window cache group.
fn group_capacity() -> Capacity {
    use ipc::batch::{Phase, Shape, Variant};

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
        media_bytes: Vec::new(),
        call_window: 2,
    }
}

/// A group's capacity reaches the scheduler node as the peer's first rank stated it, its
/// phases, buckets, rows and cache groups included.
#[test]
fn a_group_s_capacity_round_trips_through_the_info_response() {
    let capacity = group_capacity();

    let decoded = round_trip(&MainProcessResponse::info(capacity.clone()));
    let MainProcessResponse::Info {
        capacity: decoded, ..
    } = decoded
    else {
        panic!("the decoded response is no info response: {decoded:?}");
    };
    assert_eq!(decoded, capacity);
}

/// A decode batch of two requests on worker group 2, as the scheduler hands it to its executor.
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

/// One fixture per `RequestKind`, the submit frame with and without a correlation identity.
fn request_fixtures() -> Vec<MainProcessRequest> {
    use storage::runtime::{Completion, GroupId, OwnerId, Request};

    let mut submit = MainProcessRequest::submit(12, false, group_batch());
    submit.set_message_id(Some(91));
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
    vec![
        MainProcessRequest::info(),
        submit,
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
        MainProcessRequest::close(),
    ]
}

fn response_fixtures() -> Vec<MainProcessResponse> {
    vec![
        MainProcessResponse::Info {
            message_id: Some(17),
            capacity: group_capacity(),
        },
        MainProcessResponse::Result {
            message_id: Some(17),
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
        MainProcessResponse::Ok {
            message_id: Some(17),
        },
        MainProcessResponse::Error {
            message_id: Some(17),
            error: MainProcessResponseError {
                message: "device fault on decode".into(),
                code: Some(ErrorCode::ComputeError),
                fatal: false,
                phase: Some("execute".into()),
                route: Some("und.decode".into()),
            },
        },
        MainProcessResponse::Stalled {
            stall: Stall {
                batch_id: Some(12),
                route: Some("decode".into()),
                method: Some("uniserve.runner.forward".into()),
                running_ms: 150_000,
            },
        },
    ]
}

#[test]
fn every_request_kind_round_trips_through_ipc() {
    let fixtures = request_fixtures();
    for kind in [
        RequestKind::Info,
        RequestKind::Submit,
        RequestKind::Shadow,
        RequestKind::Map,
        RequestKind::Stamp,
        RequestKind::Close,
    ] {
        assert!(
            fixtures.iter().any(|request| request.kind() == kind),
            "no fixture covers request kind {kind:?}"
        );
    }
    for request in fixtures {
        assert_eq!(
            round_trip(&request),
            request,
            "round trip for {:?}",
            request.kind()
        );
    }
}

#[test]
fn every_response_kind_round_trips_through_ipc() {
    let fixtures = response_fixtures();
    for kind in [
        ResponseKind::Info,
        ResponseKind::Result,
        ResponseKind::Ok,
        ResponseKind::Error,
        ResponseKind::Stalled,
    ] {
        assert!(
            fixtures.iter().any(|response| response.kind() == kind),
            "no fixture covers response kind {kind:?}"
        );
    }
    for response in fixtures {
        assert_eq!(
            round_trip(&response),
            response,
            "round trip for {:?}",
            response.kind()
        );
    }
}

#[test]
fn the_buffer_pool_s_requests_read_back_from_the_frames_that_carry_them() {
    let MainProcessRequest::Stamp { requests, .. } = round_trip(&MainProcessRequest::stamp(
        storage::runtime::GroupId(2),
        &stamped(),
    )) else {
        panic!("a stamp frame reads back as a stamp frame");
    };
    assert_eq!(decode_requests(&requests).unwrap(), stamped());

    let response = round_trip(&MainProcessResponse::result(
        batch::BatchOutput::default(),
        &stamped(),
        14,
    ));
    let MainProcessResponse::Result {
        requests, ended, ..
    } = response
    else {
        panic!("a result frame reads back as a result frame");
    };
    assert_eq!(decode_requests(&requests).unwrap(), stamped());
    assert_eq!(ended, 14);

    // A record that is not a request of the buffer pool's interface is an error to its reader.
    assert!(decode_requests(&[vec![0; 4]]).is_err());
}

#[test]
fn a_frame_states_the_length_of_the_message_it_carries() {
    let frame = Frame::encode(&MainProcessRequest::info()).unwrap();
    let (field, message) = frame.as_bytes().split_at(Frame::LENGTH_BYTES);

    assert_eq!(Frame::message_len(field.try_into().unwrap()), message.len());
}

#[test]
fn a_frame_decodes_only_as_the_message_it_carries() {
    let request = Frame::encode(&MainProcessRequest::info()).unwrap();
    assert!(matches!(
        request.decode::<MainProcessResponse>(),
        Err(IpcError::Encoding(_))
    ));

    // A message cut short, as a frame of another build may be, is refused:
    // the frame's length field counts the shortened message, so a channel
    // receives it whole and its decoding fails.
    let mut cut = Frame::encode(&MainProcessResponse::ok())
        .unwrap()
        .into_bytes();
    cut.pop();
    let length = u32::try_from(cut.len() - Frame::LENGTH_BYTES).unwrap();
    cut[..Frame::LENGTH_BYTES].copy_from_slice(&length.to_le_bytes());

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let mut channel = Channel::connect(&listener.local_addr().unwrap().to_string()).unwrap();
    let (mut peer, _) = listener.accept().unwrap();
    peer.write_all(&cut).unwrap();

    let deadline = Instant::now() + Duration::from_secs(5);
    let received = loop {
        match channel.try_recv::<MainProcessResponse>() {
            Ok(None) => {
                assert!(Instant::now() < deadline, "the cut frame did not arrive");
                channel.wait(Duration::from_millis(50)).unwrap();
            }
            received => break received,
        }
    };
    assert!(
        matches!(received, Err(IpcError::Encoding(_))),
        "{received:?}"
    );
}
