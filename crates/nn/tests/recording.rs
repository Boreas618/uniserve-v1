//! What a recording forward answers: the shapes and programs the launching forward of the same
//! rank would, with collectives that check their views and issue nothing.
//!
//! A recording forward has no device, so every view here addresses a counting arena that no memory
//! backs; nothing reads it.

use nn::layers::{ParallelLMHead, UlyssesExchange};
use nn::{Arena, DType, DevicePtr, Forward, Shape, Split, TensorView, Weight};
use ops::Op;
use ops::logits::VocabUnpad;

/// The programs a walk reported, as `(operator, key)` pairs.
#[derive(Default)]
struct Launches(Vec<(String, String)>);

impl ops::capture::Sink for Launches {
    fn launched(&mut self, operator: &'static str, key: &str) {
        self.0.push((operator.to_owned(), key.to_owned()));
    }
}

/// Runs `walk` with a capture sink installed and answers what it reported.
fn recorded<T>(walk: impl FnOnce() -> T) -> (T, Vec<(String, String)>) {
    let mut launches = Launches::default();
    ops::capture::set_sink(Some(&mut launches));
    let answer = walk();
    ops::capture::set_sink(None);
    (answer, launches.0)
}

/// A BF16 weight of `dims` elements over a region that is never read.
fn weight(dims: &[u64]) -> Weight {
    let shape = Shape::new(dims);
    let backing = nn::Backing::new(DevicePtr::new(0x1_0000), shape.numel() * 2);
    Weight {
        backing,
        view: TensorView::new(DevicePtr::new(0x1_0000), DType::BF16, shape),
    }
}

/// Rank 1 of 2 of a vocabulary of 30, sharded into rows 15..30 and padded to 16.
fn head() -> nn::Result<ParallelLMHead> {
    ParallelLMHead::new(weight(&[16, 8]), 15, 15, 30, 16)
}

/// A vocabulary-parallel head over two ranks gathers both shards and unpads the gathered width:
/// a recording forward split as rank 1 of 2 asks for the same gather buffer and states the same
/// unpad program as that rank's launching forward.
#[test]
fn a_split_recording_answers_the_gathered_width_of_its_rank() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena).with_split(1, 2).unwrap();
    assert_eq!(f.split(), Some(Split { rank: 1, ranks: 2 }));

    let head = head().unwrap();
    let hidden = arena.tensor(DType::BF16, Shape::new(&[3, 8])).unwrap();
    let projected = arena.tensor(DType::BF16, Shape::new(&[3, 16])).unwrap();
    let gathered = arena.tensor(DType::BF16, Shape::new(&[3, 32])).unwrap();
    let out = arena.tensor(DType::F32, Shape::new(&[3, 30])).unwrap();

    let (answer, launches) = recorded(|| head.forward(&f, hidden, projected, Some(gathered), out));
    answer.expect("two shards of 16 gather into 32 columns");
    let unpad = ops::descriptor::key(&head.unpad_descriptor(DType::BF16, DType::F32, 32));
    assert!(
        launches
            .iter()
            .any(|(operator, key)| operator == <VocabUnpad as Op>::NAME && *key == unpad),
        "{launches:?}"
    );

    // A gather buffer of one shard is what an unsplit head reads, and a split of two refuses it.
    let one_shard = arena.tensor(DType::BF16, Shape::new(&[3, 16])).unwrap();
    assert!(
        head.forward(&f, hidden, projected, Some(one_shard), out)
            .is_err()
    );
    let unsplit = Forward::recording(&arena);
    assert!(head.forward(&unsplit, hidden, projected, None, out).is_ok());
}

/// A sequence-parallel exchange over two ranks moves two equal blocks: a recording forward split
/// as rank 1 of 2 accepts the views that rank exchanges and refuses a layer built for another
/// group size, as the launching forward of that group does.
#[test]
fn a_split_recording_checks_an_exchange_against_its_member_count() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena).with_split(1, 2).unwrap();
    let send = arena.tensor(DType::BF16, Shape::new(&[2, 4, 64])).unwrap();
    let recv = arena.tensor(DType::BF16, Shape::new(&[8, 64])).unwrap();

    let two = UlyssesExchange::new(2).unwrap();
    two.exchange(&f, send, recv).unwrap();
    let transfer = two.start(&f, send, recv).unwrap();
    two.finish(&f, transfer).unwrap();
    let rows = arena.tensor(DType::BF16, Shape::new(&[4, 64])).unwrap();
    two.gather(&f, rows, recv).unwrap();

    let four = UlyssesExchange::new(4).unwrap();
    let refused = four.exchange(&f, send, recv).unwrap_err();
    assert!(
        format!("{refused}").contains("built for 4 ranks"),
        "{refused}"
    );
}

/// Collectives are in place or shape-preserving, so a recording forward checks their views and
/// issues nothing: no program is reported and every view keeps its shape.
#[test]
fn collectives_on_a_recording_forward_report_nothing() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena).with_split(1, 2).unwrap();
    let rows = arena.tensor(DType::BF16, Shape::new(&[4, 64])).unwrap();
    let gathered = arena.tensor(DType::BF16, Shape::new(&[8, 64])).unwrap();
    let blocks = arena.tensor(DType::BF16, Shape::new(&[8, 64])).unwrap();

    let (answers, launches) = recorded(|| {
        [
            f.all_reduce(rows),
            f.all_gather(rows, gathered),
            f.all_to_all(gathered, blocks),
            f.start_all_to_all(gathered, blocks)
                .and_then(|transfer| f.finish(transfer)),
            f.start_gather_rows(gathered, 4, 0, 4)
                .and_then(|transfer| f.finish(transfer)),
        ]
    });
    for answer in answers {
        answer.unwrap();
    }
    assert!(launches.is_empty(), "{launches:?}");

    // The views a launching group would refuse are refused here too.
    assert!(
        f.all_gather(rows, rows).is_err(),
        "one rank's rows are not two ranks'"
    );
    let odd = arena.tensor(DType::BF16, Shape::new(&[3])).unwrap();
    assert!(
        f.all_to_all(odd, odd).is_err(),
        "three elements are not two equal blocks"
    );
}

/// A forward without a group or a split has no one to collect with, and says so.
#[test]
fn a_collective_needs_a_split() {
    let arena = Arena::counting();
    let f = Forward::recording(&arena);
    assert_eq!(f.split(), None);
    let rows = arena.tensor(DType::BF16, Shape::new(&[4, 64])).unwrap();
    assert!(f.all_reduce(rows).is_err());
    assert!(Forward::recording(&arena).with_split(2, 2).is_err());
    assert!(Forward::recording(&arena).with_split(0, 0).is_err());
}
