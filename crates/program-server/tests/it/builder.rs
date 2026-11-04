//! The compile queue (design section 6.2): one build per module path, the two
//! urgencies, the queue order, and the limits of jobs and host memory.

use program_server::builder::{BuildState, Builder, Limits, Urgency};
use program_server::store::ModulePath;

fn limits(max_jobs: usize, host_reserve: u64) -> Limits {
    Limits {
        max_jobs,
        host_reserve,
    }
}

fn path(key: &str) -> ModulePath {
    ModulePath::new("rms_norm", "triton", 1, "sm_100a", key).unwrap()
}

#[test]
fn one_path_is_one_build_however_many_ask_for_it() {
    let mut builder = Builder::new(limits(4, 0));
    let first = builder.submit(&path("dtype_bf16"), Urgency::Waiting, 1);
    let second = builder.submit(&path("dtype_bf16"), Urgency::Waiting, 2);
    assert_eq!(first.path, second.path);
    assert_eq!(builder.pending(), 1);

    let started = builder.start(u64::MAX, |_| 1);
    assert_eq!(started, vec![path("dtype_bf16")]);
    let ended = builder.finish(&path("dtype_bf16"), Ok(())).unwrap();
    assert_eq!(ended.state, BuildState::Committed);
    assert_eq!(ended.waiters, vec![1, 2]);
    assert_eq!(builder.pending(), 0);
}

#[test]
fn a_waiting_build_moves_ahead_of_the_background_builds_queued_first() {
    let mut builder = Builder::new(limits(1, 0));
    builder.submit(&path("first"), Urgency::Background, 1);
    builder.submit(&path("second"), Urgency::Background, 2);
    // A resolve needs `second` now: it becomes Waiting and goes ahead.
    builder.submit(&path("second"), Urgency::Waiting, 3);

    let started = builder.start(u64::MAX, |_| 1);
    assert_eq!(started, vec![path("second")]);
    assert_eq!(
        builder.state(&path("first")),
        Some(BuildState::Queued),
        "the background build keeps its place"
    );
}

#[test]
fn builds_of_one_urgency_start_in_arrival_order() {
    let mut builder = Builder::new(limits(2, 0));
    builder.submit(&path("first"), Urgency::Waiting, 1);
    builder.submit(&path("second"), Urgency::Waiting, 2);

    assert_eq!(
        builder.start(u64::MAX, |_| 1),
        vec![path("first"), path("second")]
    );
}

#[test]
fn a_build_starts_only_when_the_host_memory_it_needs_stays_free() {
    let mut builder = Builder::new(limits(4, 100));
    builder.submit(&path("large"), Urgency::Waiting, 1);

    let peak = |_: &program_server::builder::Build| 60_u64;
    // 150 available - 100 reserved = 50, less than the build's 60.
    assert!(builder.start(150, peak).is_empty());
    assert_eq!(builder.state(&path("large")), Some(BuildState::Queued));

    // 170 - 100 = 70, enough.
    assert_eq!(builder.start(170, peak), vec![path("large")]);
}

#[test]
fn a_build_that_does_not_fit_leaves_its_place_to_one_that_does() {
    let mut builder = Builder::new(limits(4, 0));
    builder.submit(&path("large"), Urgency::Waiting, 1);
    builder.submit(&path("small"), Urgency::Waiting, 2);

    let started = builder.start(50, |build| {
        if build.path.descriptor == "large" {
            100
        } else {
            10
        }
    });
    assert_eq!(started, vec![path("small")]);
    assert_eq!(builder.state(&path("large")), Some(BuildState::Queued));
}

#[test]
fn no_more_builds_start_than_the_job_limit_allows() {
    let mut builder = Builder::new(limits(1, 0));
    builder.submit(&path("first"), Urgency::Waiting, 1);
    builder.submit(&path("second"), Urgency::Waiting, 2);

    assert_eq!(builder.start(u64::MAX, |_| 1), vec![path("first")]);
    // One build runs; the queue does not start another until it ends.
    assert!(builder.start(u64::MAX, |_| 1).is_empty());
    builder.finish(&path("first"), Ok(())).unwrap();
    assert_eq!(builder.start(u64::MAX, |_| 1), vec![path("second")]);
}

#[test]
fn a_failed_build_keeps_its_log_for_the_waiters() {
    let mut builder = Builder::new(limits(1, 0));
    builder.submit(&path("broken"), Urgency::Waiting, 7);
    builder.start(u64::MAX, |_| 1);
    let ended = builder
        .finish(&path("broken"), Err("nvcc: not found".to_owned()))
        .unwrap();
    assert_eq!(
        ended.state,
        BuildState::Failed("nvcc: not found".to_owned())
    );
    assert_eq!(
        builder.state(&path("broken")),
        Some(BuildState::Failed("nvcc: not found".to_owned()))
    );
}

#[test]
fn a_path_built_again_forgets_the_build_that_ended_before() {
    let mut builder = Builder::new(limits(1, 0));
    builder.submit(&path("retried"), Urgency::Waiting, 1);
    builder.start(u64::MAX, |_| 1);
    builder
        .finish(&path("retried"), Err("host memory".to_owned()))
        .unwrap();

    builder.submit(&path("retried"), Urgency::Waiting, 2);
    assert_eq!(builder.state(&path("retried")), Some(BuildState::Queued));
    assert_eq!(builder.pending(), 1);
}
