//! The processing program over the release's families, driven through its interface as the
//! module's exports drive it, with a vocabulary of the tests' own in place of the daemon's.

use std::collections::BTreeMap;

use scheduling::processing::video::Binding as VideoBinding;
use scheduling::processing::{ChatOptions, Event, ImageBlock, Processing};
use serde_json::{Map, json};

use super::*;
use crate::chat::CHATML_TEMPLATE;
use crate::minimax_h3::{ConditionPacking, DenoiserInfo, ReferenceLimits, ServedCanvas, VideoTask};
use crate::qwen_vl::ProcessorConfig;
use crate::testing::{Characters, chat};

/// The program of the release's families.
type Release = Program<super::Release>;

/// The id of `<think>`, of `</think>` and of `<|im_end|>` in [`Characters::chat`].
const THINK: u32 = 5;
const END_THINK: u32 = 6;
const IM_END: u32 = 2;

/// The text of a Qwen-VL processor file, as a vision checkpoint publishes it.
const VISION_PROCESSOR: &str = r#"{"patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
    "size": {"shortest_edge": 4096, "longest_edge": 16777216}}"#;

/// The files of a Qwen3 checkpoint as Qwen releases it: its template in its tokenizer
/// configuration, its context bound at the root of its configuration, and generation defaults
/// that end at two ids.
fn qwen3_files() -> BTreeMap<String, String> {
    let tokenizer = json!({"eos_token": "<|im_end|>", "chat_template": CHATML_TEMPLATE});
    BTreeMap::from([
        (
            "config.json".to_owned(),
            r#"{"architectures": ["Qwen3MoeForCausalLM"], "max_position_embeddings": 40960}"#
                .to_owned(),
        ),
        ("tokenizer_config.json".to_owned(), tokenizer.to_string()),
        (
            "generation_config.json".to_owned(),
            r#"{"eos_token_id": [151645, 151643], "temperature": 0.6, "top_k": 20}"#.to_owned(),
        ),
    ])
}

/// The files of a Qwen3 checkpoint exported since `transformers` writes templates as files of
/// their own, as NVIDIA's ModelOpt NVFP4 export of Qwen3-30B-A3B publishes them: [`qwen3_files`]
/// with the template in `chat_template.jinja` alone and a tokenizer configuration that states
/// none.
fn qwen3_export_files() -> BTreeMap<String, String> {
    let mut files = qwen3_files();
    files.insert(
        "tokenizer_config.json".to_owned(),
        r#"{"eos_token": "<|im_end|>"}"#.to_owned(),
    );
    files.insert("chat_template.jinja".to_owned(), CHATML_TEMPLATE.to_owned());
    files
}

/// The files of a Qwen checkpoint with a vision tower: its language model a section of its
/// configuration, its template a file of its own, which writes an image's placeholder between
/// its markers, and the Qwen-VL processor's two files.
fn vision_files(class: &str) -> BTreeMap<String, String> {
    let template = "{% for message in messages %}{% if message.content is string %}\
        {{ message.content }}{% else %}{% for part in message.content %}\
        {% if part.type == 'image' %}<|vision_start|><|image_pad|><|vision_end|>\
        {% else %}{{ part.text }}{% endif %}{% endfor %}{% endif %}{% endfor %}";
    BTreeMap::from([
        (
            "config.json".to_owned(),
            json!({"architectures": [class], "text_config": {"max_position_embeddings": 262144}})
                .to_string(),
        ),
        (
            "tokenizer_config.json".to_owned(),
            r#"{"eos_token": "<|im_end|>"}"#.to_owned(),
        ),
        ("chat_template.jinja".to_owned(), template.to_owned()),
        (
            "preprocessor_config.json".to_owned(),
            VISION_PROCESSOR.to_owned(),
        ),
        (
            "video_preprocessor_config.json".to_owned(),
            VISION_PROCESSOR.to_owned(),
        ),
    ])
}

/// The binding of `family` to a checkpoint of `files`, answering reasoning apart.
fn binding(family: &str, files: BTreeMap<String, String>) -> Bind {
    Bind {
        family: family.to_owned(),
        files,
        chat: ChatOptions {
            reasoning: true,
            ..ChatOptions::default()
        },
        ..Bind::default()
    }
}

/// The instance `bind` binds, and its facts.
fn bound(bind: Bind) -> (Release, Bound) {
    Release::bind_with(bind, &Characters::chat()).unwrap()
}

/// The facts of a chat binding.
fn chat_facts(bound: Bound) -> ChatFacts {
    let Bound::Chat(facts) = bound else {
        panic!("a chat family binds to chat facts: {bound:?}");
    };
    facts
}

/// The prompt of one user message `hello` that `bind` renders.
fn hello(bind: Bind) -> Vec<Segment> {
    let (mut program, _) = bound(bind);
    let prompt = program
        .render(chat::request(vec![chat::user("hello")]))
        .unwrap();
    prompt.segments
}

/// Every family of the release is declared with the classes it claims and what its binding
/// reads of a checkpoint, and a checkpoint resolves by the declarations as the daemon resolves
/// it.
#[test]
fn every_family_is_declared_with_its_claims_and_files() {
    let declared = Release::families();
    let ids: Vec<&str> = declared.iter().map(|family| family.id.as_str()).collect();
    assert_eq!(ids, ["qwen3", "qwen3_5_moe", "qwen4_exp", "minimax_h3"]);

    let names = |family: &interface::Family| match &family.serving {
        interface::Serving::Chat { files } => files
            .iter()
            .map(|file| (file.name.clone(), file.required))
            .collect::<Vec<_>>(),
        interface::Serving::Video(_) => panic!("{} serves chat", family.id),
    };
    // A Qwen3 checkpoint may keep its template in a file of its own or in its tokenizer
    // configuration, so the file is read when published and required of none.
    assert_eq!(
        names(&declared[0]),
        [
            ("config.json".to_owned(), true),
            ("tokenizer_config.json".to_owned(), true),
            ("generation_config.json".to_owned(), false),
            ("chat_template.jinja".to_owned(), false),
        ]
    );
    let vision = names(&declared[1]);
    for file in [
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ] {
        assert!(vision.contains(&(file.to_owned(), true)), "{vision:?}");
    }
    assert_eq!(names(&declared[2]), vision);

    let interface::Serving::Video(pipeline) = &declared[3].serving else {
        panic!("MiniMax-H3 serves video");
    };
    assert_eq!(declared[3].root, RootFile::ModularIndex);
    assert!(pipeline.base_pin.is_some());
    assert!(pipeline.processing_parts.contains(&"tokenizer".to_owned()));

    let config = r#"{"architectures": ["Qwen3_5MoeForConditionalGeneration"]}"#;
    let resolved = interface::resolve(&declared, [(RootFile::Config, config)]).unwrap();
    assert_eq!(resolved.id, "qwen3_5_moe");
}

/// A Qwen3 checkpoint binds to the facts its files state: its context bound, the EOS ids of its
/// generation defaults and its tokenizer's `eos_token`, its sampling defaults, and the output
/// formats its family parses; a deployment's `max_model_len` takes the context bound's place.
#[test]
fn a_qwen3_checkpoint_binds_to_the_facts_its_files_state() {
    let facts = chat_facts(bound(binding("qwen3", qwen3_files())).1);
    assert_eq!(facts.max_model_len, 40960);
    assert_eq!(facts.primary_eos, Some(IM_END));
    assert_eq!(facts.eos, [IM_END, 151643, 151645]);
    assert_eq!(facts.sampling.temperature, Some(0.6));
    assert_eq!(facts.sampling.top_k, Some(20));
    assert_eq!(facts.sampling.top_p, None);
    assert!(facts.tools && facts.reasoning && !facts.images);

    let configured = Bind {
        max_model_len: Some(4096),
        ..binding("qwen3", qwen3_files())
    };
    assert_eq!(chat_facts(bound(configured).1).max_model_len, 4096);
}

/// A checkpoint that does not publish or state what its family reads is refused, naming it;
/// so are options the family does not take.
#[test]
fn a_checkpoint_without_what_its_family_reads_is_refused() {
    let refused = |bind: Bind| match Release::bind_with(bind, &Characters::chat()) {
        Ok(_) => panic!("the binding is refused"),
        Err(error) => error,
    };

    let mut files = qwen3_files();
    files.remove("tokenizer_config.json");
    let error = refused(binding("qwen3", files));
    assert!(error.contains("tokenizer_config.json"), "{error}");

    let mut files = qwen3_files();
    files.insert(
        "config.json".to_owned(),
        r#"{"text_config": {}}"#.to_owned(),
    );
    let error = refused(binding("qwen3", files));
    assert!(error.contains("max_position_embeddings"), "{error}");

    let mut files = vision_files("Qwen3_5MoeForConditionalGeneration");
    files.remove("chat_template.jinja");
    let error = refused(binding("qwen3_5_moe", files));
    assert!(error.contains("chat_template.jinja"), "{error}");

    let options = Bind {
        processor_options: Map::from_iter([("max_pixels".to_owned(), 1.into())]),
        ..binding("qwen3", qwen3_files())
    };
    let error = refused(options);
    assert!(error.contains("mm_processor_kwargs"), "{error}");

    let error = refused(binding("unknown", qwen3_files()));
    assert!(error.contains("unknown"), "{error}");
}

/// A chat request renders into the prompt its template writes, one text segment the daemon
/// tokenizes, and a request without messages is refused.
#[test]
fn a_request_renders_into_the_prompt_its_template_writes() {
    let (mut program, _) = bound(binding("qwen3", qwen3_files()));
    let prompt = program
        .render(chat::request(vec![chat::user("hello")]))
        .unwrap();
    assert_eq!(
        prompt.segments,
        [Segment::Text(
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n".to_owned()
        )]
    );

    let refusal = program.render(chat::request(Vec::new())).unwrap_err();
    assert_eq!(refusal.param, None);
    assert!(
        refusal.message.contains("at least one message"),
        "{refusal}"
    );
}

/// A checkpoint's template renders from where the checkpoint keeps it, as `transformers` loads
/// it: a Qwen3 release's from its tokenizer configuration, a later Qwen3 export's from its
/// `chat_template.jinja` alone, and the file before the field in a Qwen3 or Qwen3.6 checkpoint
/// that publishes both. A Qwen3 checkpoint with neither is refused, naming both places, and a
/// deployment's template replaces the checkpoint's in every layout.
#[test]
fn a_template_renders_from_where_its_checkpoint_keeps_it() {
    let chatml = [Segment::Text(
        "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n".to_owned(),
    )];
    assert_eq!(hello(binding("qwen3", qwen3_files())), chatml);
    assert_eq!(hello(binding("qwen3", qwen3_export_files())), chatml);

    // Released checkpoints that publish both hold one template in the two places; a file that
    // differs from the field shows which one renders.
    let file = "{{ messages[0].content }} from the file";
    let vision = vision_files("Qwen3_5MoeForConditionalGeneration");
    let field = json!({"eos_token": "<|im_end|>", "chat_template": CHATML_TEMPLATE});
    for (family, mut files) in [("qwen3", qwen3_files()), ("qwen3_5_moe", vision)] {
        files.insert("tokenizer_config.json".to_owned(), field.to_string());
        files.insert("chat_template.jinja".to_owned(), file.to_owned());
        assert_eq!(
            hello(binding(family, files)),
            [Segment::Text("hello from the file".to_owned())],
            "{family}"
        );
    }

    let mut neither = qwen3_export_files();
    neither.remove("chat_template.jinja");
    let Err(error) = Release::bind_with(binding("qwen3", neither), &Characters::chat()) else {
        panic!("a checkpoint without a template is refused");
    };
    assert!(
        error.contains("chat_template.jinja") && error.contains("tokenizer_config.json"),
        "{error}"
    );

    for files in [qwen3_files(), qwen3_export_files()] {
        let configured = Bind {
            chat: ChatOptions {
                template: Some("{{ messages[0].content }}!".to_owned()),
                ..ChatOptions::default()
            },
            ..binding("qwen3", files)
        };
        assert_eq!(hello(configured), [Segment::Text("hello!".to_owned())]);
    }
}

/// A family that takes images places each image's block where its template writes the
/// image's placeholder, sized by its processor's rule; a family that takes none refuses image
/// parts, and a prompt whose placeholders are not one per image is refused.
#[test]
fn image_parts_become_blocks_where_the_template_writes_their_placeholder() {
    for (family, class) in [
        ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
        ("qwen4_exp", "Qwen4ExpForConditionalGeneration"),
    ] {
        let (mut program, facts) = bound(binding(family, vision_files(class)));
        assert!(chat_facts(facts).images);

        let size = ImageSize {
            width: 640,
            height: 480,
        };
        let mut request = chat::request(vec![chat::user_parts(vec![
            chat::text_part("look: "),
            chat::image_part("data:image/png;base64,AA"),
            chat::text_part(" what is it?"),
        ])]);
        request.images = vec![size];
        let prompt = program.render(request.clone()).unwrap();

        let rule = ProcessorConfig::parse(VISION_PROCESSOR, VISION_PROCESSOR).unwrap();
        let block = rule.block(size.width, size.height).unwrap();
        assert_eq!(
            prompt.segments,
            [
                Segment::Text("look: <|vision_start|>".to_owned()),
                Segment::Image(ImageBlock {
                    tokens: block.tokens,
                    positions: block.positions,
                }),
                Segment::Text("<|vision_end|> what is it?".to_owned()),
            ],
            "{family}"
        );

        request.images.push(size);
        let refusal = program.render(request).unwrap_err();
        assert!(
            refusal.message.contains("1 image placeholder(s) for 2"),
            "{refusal}"
        );
    }

    let (mut program, _) = bound(binding("qwen3", qwen3_files()));
    let refusal = program
        .render(chat::request(vec![chat::user_parts(vec![
            chat::image_part("data:image/png;base64,AA"),
        ])]))
        .unwrap_err();
    assert!(refusal.message.contains("image_url"), "{refusal}");
}

/// Merges adjacent events of one kind, so a comparison does not depend on how text was cut.
fn merged(events: Vec<Event>) -> Vec<Event> {
    let mut merged: Vec<Event> = Vec::new();
    for event in events {
        match (merged.last_mut(), event) {
            (Some(Event::Reasoning(text)), Event::Reasoning(delta))
            | (Some(Event::Content(text)), Event::Content(delta))
            | (Some(Event::Arguments(text)), Event::Arguments(delta)) => text.push_str(&delta),
            (_, event) => merged.push(event),
        }
    }
    merged
}

/// Opens `program`'s parser for a prompt that ends inside reasoning, with `tools`, and parses
/// `deltas`.
fn parse(
    program: &mut Release,
    tools: Option<Vec<interface::Tool>>,
    deltas: &[&str],
) -> Vec<Event> {
    program
        .open(Open {
            tools,
            prompt: vec![IM_END, u32::from(b'\n'), THINK],
        })
        .unwrap();
    let mut events = Vec::new();
    for delta in deltas {
        events.extend(program.push(delta).unwrap());
    }
    events.extend(program.finish().unwrap());
    merged(events)
}

/// A response parses into the reasoning, the content and the calls its family writes: Qwen3's
/// JSON calls as written, Qwen3.6's tags typed by the request's tool schemas.
#[test]
fn a_response_parses_into_reasoning_content_and_calls() {
    let tool = chat::function(
        "lookup",
        json!({"type": "object", "properties": {"limit": {"type": "integer"}}}),
    );

    let (mut program, _) = bound(binding("qwen3", qwen3_files()));
    let events = parse(
        &mut program,
        Some(vec![tool.clone()]),
        &[
            "why</thi",
            "nk>so<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"limit\": 3}}\n</tool_call>",
        ],
    );
    assert_eq!(
        events,
        [
            Event::Reasoning("why".to_owned()),
            Event::Content("so".to_owned()),
            Event::ToolCall {
                name: "lookup".to_owned()
            },
            Event::Arguments(r#"{"limit": 3}"#.to_owned()),
        ]
    );

    let (mut program, _) = bound(binding(
        "qwen3_5_moe",
        vision_files("Qwen3_5MoeForConditionalGeneration"),
    ));
    let events = parse(
        &mut program,
        Some(vec![tool]),
        &[
            "</think><tool_call>\n<function=lookup>\n<parameter=limit>\n3\n</parameter>\n</function>\n</tool_call>",
        ],
    );
    assert_eq!(
        events,
        [
            Event::ToolCall {
                name: "lookup".to_owned()
            },
            Event::Arguments(r#"{"limit":3}"#.to_owned()),
        ]
    );
}

/// Without callable tools a call's markup is content, and a deployment that does not answer
/// reasoning apart streams the delimiters as content.
#[test]
fn unparsed_formats_are_content() {
    let (mut program, _) = bound(binding("qwen3", qwen3_files()));
    let events = parse(
        &mut program,
        None,
        &["a</think><tool_call>\n{}\n</tool_call>"],
    );
    assert_eq!(
        events,
        [
            Event::Reasoning("a".to_owned()),
            Event::Content("<tool_call>\n{}\n</tool_call>".to_owned()),
        ]
    );

    let unparsed = Bind {
        chat: ChatOptions::default(),
        ..binding("qwen3", qwen3_files())
    };
    let (mut program, _) = bound(unparsed);
    let events = parse(&mut program, None, &["a</think>b"]);
    assert_eq!(events, [Event::Content("a</think>b".to_owned())]);
}

/// A malformed call fails the response, naming the format.
#[test]
fn a_malformed_call_fails_the_response() {
    let (mut program, _) = bound(binding("qwen3", qwen3_files()));
    program
        .open(Open {
            tools: Some(Vec::new()),
            prompt: vec![IM_END],
        })
        .unwrap();
    let error = program
        .push("<tool_call>\n{\"arguments\": {}}\n</tool_call>")
        .unwrap_err();
    assert!(error.contains("malformed json tool call"), "{error}");
}

/// A parser restored in another instance from the bytes one saved goes on where it stood, so
/// the two together produce what one would have.
#[test]
fn a_parser_restored_from_its_saved_state_goes_on() {
    let deltas = [
        "why</th",
        "ink>so<tool_call>\n{\"name\": \"f\", \"argu",
        "ments\": {}}\n</tool_call>!",
    ];
    let whole = {
        let (mut program, _) = bound(binding("qwen3", qwen3_files()));
        parse(&mut program, Some(Vec::new()), &deltas)
    };

    for cut in 1..deltas.len() {
        let (mut first, _) = bound(binding("qwen3", qwen3_files()));
        first
            .open(Open {
                tools: Some(Vec::new()),
                prompt: vec![THINK],
            })
            .unwrap();
        let mut events = Vec::new();
        for delta in &deltas[..cut] {
            events.extend(first.push(delta).unwrap());
        }
        let state = first.save();

        let (mut second, _) = bound(binding("qwen3", qwen3_files()));
        second.restore(&state).unwrap();
        for delta in &deltas[cut..] {
            events.extend(second.push(delta).unwrap());
        }
        events.extend(second.finish().unwrap());
        assert_eq!(merged(events), whole, "cut after delta {cut}");
    }

    let (mut program, _) = bound(binding("qwen3", qwen3_files()));
    assert!(program.restore(b"not a state").is_err());
    assert!(program.push("x").is_err(), "a parser opens first");
}

/// A prompt whose tail closes reasoning starts generation in content, and one whose tail holds
/// no delimiter starts there too.
#[test]
fn the_prompt_tail_decides_where_generation_starts() {
    for tail in [vec![IM_END, END_THINK], vec![IM_END]] {
        let (mut program, _) = bound(binding("qwen3", qwen3_files()));
        program
            .open(Open {
                tools: None,
                prompt: tail,
            })
            .unwrap();
        assert_eq!(
            program.push("answer").unwrap(),
            [Event::Content("answer".to_owned())]
        );
    }
}

/// A MiniMax-H3 pipeline binds once its denoiser described itself, states what it serves,
/// holds a request to its rules and processes one through the tokenizer the daemon holds.
#[test]
fn a_minimax_h3_pipeline_binds_to_its_denoiser_and_processes_requests() {
    let canvas = |width, height| scheduling::processing::video::Canvas { width, height };
    let description = DenoiserInfo {
        tasks: vec![VideoTask::TextToVideoAudio],
        schedule_points: 9,
        video_shift: 12.0,
        audio_shift: 3.0,
        canvases: vec![ServedCanvas {
            short_edge: 768,
            aspect_ratio: "16:9".to_owned(),
            canvas: canvas(1344, 768),
        }],
        references: ReferenceLimits::default(),
        condition_packing: ConditionPacking::dense(0),
    };
    let bind = Bind {
        family: "minimax_h3".to_owned(),
        files: BTreeMap::from([
            (
                "processor/preprocessor_config.json".to_owned(),
                VISION_PROCESSOR.to_owned(),
            ),
            (
                "processor/video_preprocessor_config.json".to_owned(),
                VISION_PROCESSOR.to_owned(),
            ),
        ]),
        video: Some(VideoBinding {
            description: description.encode(),
            max_seconds: 10.0,
        }),
        ..Bind::default()
    };
    let (program, bound) = Release::bind_with(bind.clone(), &Characters::chat()).unwrap();
    let Bound::Video(facts) = bound else {
        panic!("a video family binds to video facts");
    };
    assert_eq!(facts.denoise_steps, 8);

    let spec = |size| VideoSpec {
        prompt: "a river".to_owned(),
        negative_prompt: None,
        size,
        length: Some(video::Length::Seconds(5.0)),
        fps: None,
        seed: 7,
        num_inference_steps: None,
        guidance_scale: None,
        references: Vec::new(),
        extra_params: Map::new(),
    };
    assert!(matches!(
        program.check_video(&spec(Some(canvas(640, 640)))),
        Err(InputError::Invalid { .. })
    ));
    let input = program
        .process_with(
            &Process {
                spec: spec(None),
                media: Vec::new(),
            },
            &Characters::chat(),
        )
        .unwrap();
    assert_eq!(input.size, canvas(1344, 768));
    assert!(input.conditions.is_empty());
    assert!(!input.prompt_token_ids.is_empty());

    let unbound = Bind {
        video: None,
        ..bind
    };
    assert!(Release::bind_with(unbound, &Characters::chat()).is_err());
}
