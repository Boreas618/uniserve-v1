//! The cost of the processing program's calls in the program host against the same calls made
//! natively: rendering long agentic conversations, and parsing a long streamed completion.
//!
//! `cargo bench -p processing-program --bench program -- [--template FILE] [--iterations N]`
//!
//! The module is the crate's own, built for wasm32 by the cargo that runs the bench, as the Build
//! Server builds it, and run in the program host under the leader's budgets as the daemon runs it;
//! the native calls are the same code linked into the bench. Both bind the Qwen3 family to a
//! checkpoint whose template is `--template` (a checkpoint's `chat_template`, the `processing`
//! crate's ChatML template when unset). A conversation is an agent's: a system prompt, tool
//! schemas, and turns of a user's request, the assistant's reasoning and a tool call, and the
//! tool's long result, each turn about 1,800 tokens at four characters a token; the conversations
//! are 16, 64, 192, 384 and 576 turns, about 30,000, 120,000, 350,000, 700,000 and 1,000,000
//! tokens: the input lengths of agentic coding sessions, whose requests' median is about 250,000
//! tokens and longest about 1,000,000. The completion is 10,000 tokens of reasoning, content and a
//! tool call, pushed as one engine event of one token and of eight tokens at a time; an update that
//! decodes to no text measures the cost of a call alone.
//!
//! Every time is the mean of `--iterations` calls after one warm-up call, but the module's
//! compilation, timed once; the report states each side's time and the program's overhead, and
//! what loading the module and binding an instance cost.

// A bench stops at the first call that fails, naming it.
#![expect(
    clippy::expect_used,
    reason = "a bench stops at a failed call, naming it"
)]

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use processing::Vocabulary as _;
use processing::program::{Program, Release};
use program::{Budgets, Host, Level, Module, Processor, Vocabulary};
use scheduling::processing::{
    Bind, ChatOptions, Content, Function, FunctionCall, Message, Open, Processing, Render, Tool,
    ToolCall,
};

/// The control tokens of the bench's vocabulary, by text; every other text is one token per
/// character.
const CONTROLS: [(&str, u32); 4] = [
    ("<|im_start|>", 151_644),
    ("<|im_end|>", 151_645),
    ("<think>", 151_667),
    ("</think>", 151_668),
];

/// The id of `text` in the bench's vocabulary.
fn token(text: &str) -> Option<u32> {
    CONTROLS
        .iter()
        .find(|(control, _)| *control == text)
        .map(|(_, id)| *id)
        .or_else(|| {
            let mut chars = text.chars();
            let first = chars.next()?;
            chars.next().is_none().then_some(u32::from(first))
        })
}

/// The bench's vocabulary, as the native calls read it.
struct Native;

impl processing::Vocabulary for Native {
    fn token_id(&self, text: &str) -> Option<u32> {
        token(text)
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>, processing::VocabularyError> {
        Ok(text.chars().map(u32::from).collect())
    }
}

/// The bench's vocabulary, as the program host answers the module with it.
struct Hosted;

impl Host for Hosted {
    fn log(&mut self, _: Level, _: &str) {}
}

impl Vocabulary for Hosted {
    fn token(&mut self, text: &str) -> Option<u32> {
        token(text)
    }

    fn encode(&mut self, text: &str) -> Result<Vec<u32>, String> {
        Ok(text.chars().map(u32::from).collect())
    }
}

/// The bench's arguments.
struct Arguments {
    template: Option<String>,
    iterations: u32,
}

impl Arguments {
    /// Reads `--template FILE` and `--iterations N`; `cargo bench` passes `--bench` too.
    fn parse() -> Self {
        let mut arguments = Self {
            template: None,
            iterations: 20,
        };
        let mut words = std::env::args().skip(1);
        while let Some(word) = words.next() {
            match word.as_str() {
                "--template" => {
                    let path = words.next().expect("--template names a file");
                    let text = std::fs::read_to_string(&path).expect("the template reads");
                    arguments.template = Some(text);
                }
                "--iterations" => {
                    arguments.iterations = words
                        .next()
                        .and_then(|count| count.parse().ok())
                        .expect("--iterations is a count");
                }
                _ => {}
            }
        }
        arguments
    }
}

/// The binding of a Qwen3 checkpoint whose template is `template`, answering reasoning apart.
fn bind(template: &str) -> Bind {
    Bind {
        family: "qwen3".to_owned(),
        files: BTreeMap::from([
            (
                "config.json".to_owned(),
                r#"{"architectures": ["Qwen3MoeForCausalLM"], "max_position_embeddings": 1048576}"#
                    .to_owned(),
            ),
            (
                "tokenizer_config.json".to_owned(),
                serde_json::json!({"eos_token": "<|im_end|>", "chat_template": template})
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

/// `words` words of text a model or a tool writes, `seed` varying it.
fn prose(words: usize, seed: usize) -> String {
    const WORDS: [&str; 12] = [
        "the",
        "request",
        "reads",
        "file",
        "src/lib.rs",
        "returns",
        "value",
        "error",
        "while",
        "parser",
        "token",
        "function",
    ];
    (0..words)
        .map(|index| WORDS[(index * 7 + seed) % WORDS.len()])
        .collect::<Vec<_>>()
        .join(" ")
}

/// The tools of an agent's conversation: shell, read, write and search.
fn tools() -> Vec<Tool> {
    ["run_shell", "read_file", "write_file", "search_code"]
        .into_iter()
        .map(|name| Tool::Function {
            function: Function {
                name: name.to_owned(),
                description: Some(prose(40, name.len())),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": prose(12, 1)},
                        "content": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["path"],
                }),
                strict: None,
            },
        })
        .collect()
}

/// An agent's conversation of `turns` turns, about 1,600 tokens a turn.
fn conversation(turns: usize) -> Render {
    let mut messages = vec![Message::System {
        content: Content::Text(prose(400, 0)),
    }];
    for turn in 0..turns {
        messages.push(Message::User {
            content: Content::Text(prose(80, turn)),
        });
        messages.push(Message::Assistant {
            content: Some(Content::Text(prose(60, turn + 1))),
            reasoning: Some(prose(200, turn + 2)),
            tool_calls: vec![ToolCall {
                id: format!("call_{turn}"),
                function: FunctionCall {
                    name: "read_file".to_owned(),
                    arguments: Some(
                        serde_json::json!({"path": format!("src/{turn}.rs"), "limit": 400})
                            .to_string(),
                    ),
                },
            }],
        });
        messages.push(Message::Tool {
            content: Content::Text(prose(900, turn + 3)),
            tool_call_id: format!("call_{turn}"),
        });
    }
    Render {
        messages,
        tools: Some(tools()),
        ..Render::default()
    }
}

/// The decoded text of a completion of `tokens` tokens of reasoning, content and one tool call,
/// cut into the deltas of engine events of `per_event` tokens of four characters each.
fn completion(tokens: usize, per_event: usize) -> Vec<String> {
    let reasoning = prose(tokens * 3 / 5, 5);
    let content = prose(tokens / 5, 6);
    let call = format!(
        "<tool_call>\n{}\n</tool_call>",
        serde_json::json!({"name": "write_file", "arguments": {"path": "a.rs", "content": prose(tokens / 6, 7)}})
    );
    let text = format!("{reasoning}</think>\n\n{content}{call}");
    let chars: Vec<char> = text.chars().collect();
    chars
        .chunks(4 * per_event)
        .map(|chunk| chunk.iter().collect())
        .collect()
}

/// The mean time of `iterations` calls of `call`, after one warm-up call.
fn mean(iterations: u32, mut call: impl FnMut()) -> Duration {
    call();
    let start = Instant::now();
    for _ in 0..iterations {
        call();
    }
    start.elapsed() / iterations
}

fn main() {
    let arguments = Arguments::parse();
    let template = arguments
        .template
        .clone()
        .unwrap_or_else(|| include_str!("../../src/chat/chatml.jinja").to_owned());
    let bind = bind(&template);

    // What an epoch's load and a request's fresh instance cost: compiling the module, once per
    // epoch, and instantiating and binding an instance, which a request that finds no idle
    // instance in its binding's pool waits for.
    let bytes = program_testing::module!("processing-program");
    let compiling = Instant::now();
    let module = Module::<Hosted>::processing(&bytes).expect("the module loads");
    let compiled = compiling.elapsed();
    let instance = || {
        let mut hosted = Processor::new(&module, Hosted, &Budgets::LEADER).expect("an instance");
        hosted
            .bind(&bind)
            .expect("the instance answers")
            .expect("the family binds");
        hosted
    };
    let bound = mean(arguments.iterations, || drop(instance()));
    println!("load: compiling the module {compiled:?}, a bound instance {bound:?}");
    let mut hosted = instance();
    let (mut native, _) = Program::<Release>::bind_with(bind, &Native).expect("the family binds");
    assert!(Native.token_id("<think>").is_some());

    println!("render: turns, prompt characters, native, program, overhead");
    for turns in [16, 64, 192, 384, 576] {
        let render = conversation(turns);
        let prompt = native.render(render.clone()).expect("the request renders");
        let characters: usize = prompt
            .segments
            .iter()
            .map(|segment| match segment {
                scheduling::processing::Segment::Text(text) => text.len(),
                scheduling::processing::Segment::Image(_) => 0,
            })
            .sum();
        let native_time = mean(arguments.iterations, || {
            native.render(render.clone()).expect("the request renders");
        });
        let hosted_time = mean(arguments.iterations, || {
            hosted
                .render(&render)
                .expect("the instance answers")
                .expect("the request renders");
        });
        println!(
            "render: {turns}, {characters}, {native_time:?}, {hosted_time:?}, {:.2}x",
            hosted_time.as_secs_f64() / native_time.as_secs_f64()
        );
    }

    println!(
        "parse: tokens per event, events, native per 10k tokens, program per 10k tokens, \
         program core share at 10k tokens/s"
    );
    let open = Open {
        tools: Some(tools()),
        prompt: vec![CONTROLS[0].1, CONTROLS[2].1],
    };
    // The cost of a call alone: an update that decodes to no text, which parses into nothing.
    hosted.open(&open).expect("answers").expect("opens");
    native.open(open.clone()).expect("the parser opens");
    let calls = 10_000;
    let native_call = mean(arguments.iterations, || {
        for _ in 0..calls {
            native.push("").expect("the text parses");
        }
    }) / calls;
    let hosted_call = mean(arguments.iterations, || {
        for _ in 0..calls {
            hosted.push("").expect("answers").expect("parses");
        }
    }) / calls;
    println!("call: an empty update, native {native_call:?}, program {hosted_call:?}");
    for per_event in [1, 8] {
        let deltas = completion(10_000, per_event);
        let native_time = mean(arguments.iterations, || {
            native.open(open.clone()).expect("the parser opens");
            for delta in &deltas {
                native.push(delta).expect("the text parses");
            }
            native.finish().expect("the text ends");
        });
        let hosted_time = mean(arguments.iterations, || {
            hosted.open(&open).expect("answers").expect("opens");
            for delta in &deltas {
                hosted.push(delta).expect("answers").expect("parses");
            }
            hosted.finish().expect("answers").expect("ends");
        });
        // The completion is 10,000 tokens, so its parse time is the core time of one second of
        // parsing at 10,000 tokens a second.
        println!(
            "parse: {per_event}, {}, {native_time:?}, {hosted_time:?}, {:.3}%",
            deltas.len(),
            100.0 * hosted_time.as_secs_f64()
        );
    }
}
