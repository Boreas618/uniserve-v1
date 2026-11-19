//! The processing program's module, built for wasm32 as the Build Server builds it and run in
//! the program host as the leader daemon runs it: its exports over the release's families, its
//! use of the daemon's vocabulary, and how a fault in it ends the one instance it happened in.
#![cfg(test)]

use std::collections::BTreeMap;

use program::{Budgets, Error, Host, Level, Module, Processor, Reason, Vocabulary};
use scheduling::processing::{
    Bind, Bound, ChatOptions, Content, Event, Message, Open, Render, RootFile, Segment, Serving,
};
use serde_json::json;

/// The tests' embedder: a vocabulary of single-character tokens beside ChatML's and the
/// thinking delimiters, which records what the program logged.
#[derive(Default)]
struct Characters {
    logged: Vec<String>,
}

impl Host for Characters {
    fn log(&mut self, _: Level, text: &str) {
        self.logged.push(text.to_owned());
    }
}

impl Vocabulary for Characters {
    fn token(&mut self, text: &str) -> Option<u32> {
        match text {
            "<|im_start|>" => Some(1),
            "<|im_end|>" => Some(2),
            "<think>" => Some(5),
            "</think>" => Some(6),
            _ => {
                let mut chars = text.chars();
                let first = chars.next()?;
                chars.next().is_none().then_some(u32::from(first))
            }
        }
    }

    fn encode(&mut self, text: &str) -> Result<Vec<u32>, String> {
        Ok(text.chars().map(u32::from).collect())
    }
}

/// The processing program's module, compiled once per test process.
fn module() -> Module<Characters> {
    static MODULE: std::sync::OnceLock<Module<Characters>> = std::sync::OnceLock::new();
    MODULE
        .get_or_init(|| {
            Module::processing(&program_testing::module!("processing-program")).unwrap()
        })
        .clone()
}

/// A fresh instance of the module.
fn instance() -> Processor<Characters> {
    Processor::new(&module(), Characters::default(), &Budgets::LEADER).unwrap()
}

/// The binding of a Qwen3 checkpoint whose template is ChatML.
fn qwen3() -> Bind {
    let template = "{% for message in messages %}<|im_start|>{{ message.role }}\n\
        {{ message.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n";
    Bind {
        family: "qwen3".to_owned(),
        files: BTreeMap::from([
            (
                "config.json".to_owned(),
                r#"{"architectures": ["Qwen3MoeForCausalLM"], "max_position_embeddings": 32768}"#
                    .to_owned(),
            ),
            (
                "tokenizer_config.json".to_owned(),
                json!({"eos_token": "<|im_end|>", "chat_template": template}).to_string(),
            ),
        ]),
        chat: ChatOptions {
            reasoning: true,
            ..ChatOptions::default()
        },
        ..Bind::default()
    }
}

/// The module declares the release's families, and a checkpoint resolves by its declarations.
#[test]
fn the_module_declares_the_release_families() {
    let families = instance().families().unwrap();
    let ids: Vec<&str> = families.iter().map(|family| family.id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "qwen3",
            "qwen3_5_moe",
            "qwen4_exp",
            "minimax_h3",
            "deepseek_v41"
        ]
    );
    assert!(matches!(families[0].serving, Serving::Chat { .. }));
    assert!(matches!(families[3].serving, Serving::Video(_)));

    let config = r#"{"architectures": ["Qwen3MoeForCausalLM"]}"#;
    let resolved =
        scheduling::processing::resolve(&families, [(RootFile::Config, config)]).unwrap();
    assert_eq!(resolved.id, "qwen3");
}

/// A bound instance states its checkpoint's facts through the daemon's vocabulary, renders a
/// request into its prompt and parses its response, one call per update.
#[test]
fn a_bound_instance_renders_and_parses_through_its_exports() {
    let mut processor = instance();
    let Bound::Chat(facts) = processor.bind(&qwen3()).unwrap().unwrap() else {
        panic!("a chat family binds to chat facts");
    };
    assert_eq!(facts.max_model_len, 32768);
    assert_eq!(facts.primary_eos, Some(2));

    let render = Render {
        messages: vec![Message::User {
            content: Content::Text("hi".to_owned()),
        }],
        ..Render::default()
    };
    let prompt = processor.render(&render).unwrap().unwrap();
    assert_eq!(
        prompt.segments,
        // The environment drops a template's one trailing newline, as Hugging Face's does.
        [Segment::Text(
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant".to_owned()
        )]
    );

    processor
        .open(&Open {
            tools: Some(Vec::new()),
            prompt: vec![1, 5],
        })
        .unwrap()
        .unwrap();
    let mut events = processor.push("a</th").unwrap().unwrap();
    events.extend(processor.push("ink>b").unwrap().unwrap());
    events.extend(processor.finish().unwrap().unwrap());
    assert_eq!(
        events,
        [
            Event::Reasoning("a".to_owned()),
            Event::Content("b".to_owned()),
        ]
    );
}

/// The binding of a DeepSeek-V4.1 checkpoint, which publishes no chat template.
fn deepseek_v41() -> Bind {
    Bind {
        family: "deepseek_v41".to_owned(),
        files: BTreeMap::from([
            (
                "config.json".to_owned(),
                json!({"architectures": ["DeepseekV41ForCausalLM"],
                    "text_config": {"max_position_embeddings": 1048576}})
                .to_string(),
            ),
            (
                "tokenizer_config.json".to_owned(),
                json!({"eos_token": {"__type": "AddedToken", "content": "<｜end▁of▁sentence｜>"}})
                    .to_string(),
            ),
        ]),
        chat: ChatOptions {
            reasoning: true,
            ..ChatOptions::default()
        },
        ..Bind::default()
    }
}

/// A DeepSeek-V4.1 instance encodes a request with the family's encoder, in thinking mode, and
/// parses a DSML response cut inside a call's text argument into the call closed where the text
/// ended; its parser's state crosses to another instance mid-call.
#[test]
fn a_deepseek_v41_instance_encodes_and_parses_dsml_calls() {
    let mut processor = instance();
    let Bound::Chat(facts) = processor.bind(&deepseek_v41()).unwrap().unwrap() else {
        panic!("a chat family binds to chat facts");
    };
    assert_eq!(facts.max_model_len, 1_048_576);

    let render = Render {
        messages: vec![Message::User {
            content: Content::Text("hi".to_owned()),
        }],
        ..Render::default()
    };
    let prompt = processor.render(&render).unwrap().unwrap();
    let [Segment::Text(text)] = prompt.segments.as_slice() else {
        panic!("one text segment: {prompt:?}");
    };
    assert!(text.starts_with("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75"));
    assert!(
        text.ends_with("<｜User｜>hi<｜Assistant｜><think>"),
        "{text}"
    );

    processor
        .open(&Open {
            tools: Some(Vec::new()),
            prompt: vec![5],
        })
        .unwrap()
        .unwrap();
    let mut events = processor
        .push("r</think>\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"write\">\n")
        .unwrap()
        .unwrap();
    events.extend(
        processor
            .push("<｜DSML｜ parameter name=\"text\" string=\"true\">ab")
            .unwrap()
            .unwrap(),
    );
    let state = processor.save().unwrap();

    let mut successor = instance();
    successor.bind(&deepseek_v41()).unwrap().unwrap();
    successor.restore(&state).unwrap().unwrap();
    events.extend(successor.push("c").unwrap().unwrap());
    events.extend(successor.finish().unwrap().unwrap());
    let arguments: String = events
        .iter()
        .filter_map(|event| match event {
            Event::Arguments(text) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(events[0], Event::Reasoning("r".to_owned()));
    assert_eq!(
        events[1],
        Event::ToolCall {
            name: "write".to_owned()
        }
    );
    assert_eq!(arguments, r#"{"text": "abc"}"#);
}

/// A parser's state saved by one instance is restored by another, which goes on where the
/// first stood.
#[test]
fn a_saved_parser_is_restored_in_another_instance() {
    let mut first = instance();
    first.bind(&qwen3()).unwrap().unwrap();
    first
        .open(&Open {
            tools: None,
            prompt: vec![5],
        })
        .unwrap()
        .unwrap();
    assert_eq!(
        first.push("why</th").unwrap().unwrap(),
        [Event::Reasoning("why".to_owned())]
    );
    let state = first.save().unwrap();

    let mut second = instance();
    second.bind(&qwen3()).unwrap().unwrap();
    second.restore(&state).unwrap().unwrap();
    assert_eq!(
        second.push("ink>so").unwrap().unwrap(),
        [Event::Content("so".to_owned())]
    );
}

/// A family's refusal is a reply, and the instance serves on; a fault of the program ends its
/// instance alone, with the panic's text, and every later call of it reports the same.
#[test]
fn a_fault_ends_its_instance_alone() {
    let mut processor = instance();
    let refused = processor
        .bind(&Bind {
            family: "unknown".to_owned(),
            ..qwen3()
        })
        .unwrap();
    assert!(refused.is_err());

    // A render before any binding is a defect of its host, which the program traps on.
    let ended = processor.render(&Render::default()).unwrap_err();
    let Error::Ended {
        reason: Reason::Panic(text),
        ..
    } = &ended
    else {
        panic!("the instance ends on its panic: {ended}");
    };
    assert!(text.contains("binds"), "{text}");
    assert!(processor.families().is_err(), "the instance stays ended");

    let mut other = instance();
    assert!(other.bind(&qwen3()).unwrap().is_ok());
}

/// A module of another kind of program does not load as a processing program.
#[test]
fn another_kind_of_module_is_refused() {
    let control = program_testing::module!("policy-chunk");
    let refused = Module::<Characters>::processing(&control).err();
    assert!(matches!(refused, Some(Error::Load(_))), "{refused:?}");
}
