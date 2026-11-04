//! A program (design section 5.1): the backend, revision and arguments that
//! fill one operator, and the slot it fills.

use program_server::{PhaseKind, Program, ProgramSlot};

#[test]
fn a_program_keeps_its_arguments_between_a_record_and_its_next_read() {
    let mut arguments = toml::Table::new();
    arguments.insert("chunked_prefill_size".into(), toml::Value::Integer(8192));
    arguments.insert(
        "workspace_size".into(),
        toml::Value::String("512MiB".into()),
    );
    let program = Program {
        backend: "trtllm".into(),
        revision: 3,
        arguments,
    };

    let text = toml::to_string(&program).unwrap();
    let read: Program = toml::from_str(&text).unwrap();
    assert_eq!(read, program);
}

#[test]
fn a_program_of_no_arguments_round_trips_too() {
    let program = Program {
        backend: "fcfs".into(),
        revision: 1,
        arguments: toml::Table::new(),
    };
    assert_eq!(
        toml::from_str::<Program>(&toml::to_string(&program).unwrap()).unwrap(),
        program
    );
}

#[test]
fn a_slot_names_an_operator_and_the_phase_it_fills() {
    assert_eq!(ProgramSlot::all("rms_norm").phase, None);
    assert_eq!(
        ProgramSlot::phase("paged_attention", PhaseKind::Decode).phase,
        Some(PhaseKind::Decode)
    );
    assert_eq!(PhaseKind::Prefill.to_string(), "prefill");
    assert_eq!(PhaseKind::Decode.to_string(), "decode");
}

#[test]
fn the_two_slots_of_one_operator_are_different_slots() {
    let all = ProgramSlot::all("paged_attention");
    let prefill = ProgramSlot::phase("paged_attention", PhaseKind::Prefill);
    let decode = ProgramSlot::phase("paged_attention", PhaseKind::Decode);
    assert_ne!(all, prefill);
    assert_ne!(prefill, decode);
}
