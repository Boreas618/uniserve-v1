//! The renderer's behavior over the messages of the processing interface.

use expect_test::expect;
use scheduling::processing::{Function, Message, Refusal, Render, Tool};
use serde_json::{Map, Value};

use super::metadata::{NamedSpecialToken, SpecialTokens};
use super::{
    CHATML_TEMPLATE, ChatTemplateContentFormat, GenerationPromptMode, HfChatRenderer,
    MultimodalRenderInfo,
};
use crate::testing::chat;

type Result<T> = std::result::Result<T, Refusal>;

fn sample_request(messages: Vec<Message>) -> Render {
    chat::request(messages)
}

/// The template arguments `pairs` state.
fn kwargs<const N: usize, K: Into<String>>(pairs: [(K, Value); N]) -> Map<String, Value> {
    pairs
        .into_iter()
        .map(|(name, value)| (name.into(), value))
        .collect()
}

/// Renders `request` for a new assistant turn.
fn render(template: Option<&str>, request: &Render) -> Result<String> {
    render_as(template, request, GenerationPromptMode::StartNewAssistant)
}

/// Renders `request`, ending the prompt as `mode` states.
fn render_as(
    template: Option<&str>,
    request: &Render,
    mode: GenerationPromptMode,
) -> Result<String> {
    HfChatRenderer::new(template.map(str::to_owned), Map::new(), None)
        .map_err(|error| Refusal::new(error.to_string()))?
        .render(request, mode)
}

fn render_mm(
    template: &str,
    request: &Render,
    content_format: Option<ChatTemplateContentFormat>,
) -> Result<String> {
    HfChatRenderer::new(Some(template.to_string()), Map::new(), content_format)
        .map_err(|error| Refusal::new(error.to_string()))?
        .with_multimodal(Some(MultimodalRenderInfo {
            placeholder_token: "<image>".to_string(),
        }))
        .render(request, GenerationPromptMode::StartNewAssistant)
}

fn image_request() -> Render {
    sample_request(vec![chat::user_parts(vec![
        chat::text_part("a"),
        chat::image_part("data:image/png;base64,test"),
        chat::text_part("b"),
    ])])
}

#[test]
fn string_content_format_replaces_image_with_placeholder_text() {
    let rendered = render_mm(
        "{{ messages[0].content }}",
        &image_request(),
        Some(ChatTemplateContentFormat::String),
    )
    .unwrap();

    assert_eq!(rendered, "a<image>b");
}

#[test]
fn openai_content_format_normalizes_image_url_for_template() {
    let rendered = render_mm(
        "{% for item in messages[0].content %}{% if item.type == 'image' %}<|image_pad|>{% else %}{{ item.text }}{% endif %}{% endfor %}",
        &image_request(),
        Some(ChatTemplateContentFormat::OpenAi),
    )
    .unwrap();

    assert_eq!(rendered, "a<|image_pad|>b");
}

#[test]
fn auto_content_format_preserves_mixed_system_and_user_shapes() {
    // The template both loops over content items and tests `content is
    // string`, so `Auto` selects `Preserve`: the system string still
    // supports `+` concatenation while the user parts stay a list.
    let request = sample_request(vec![
        chat::system("policy"),
        chat::user_parts(vec![
            chat::text_part("a"),
            chat::image_part("data:image/png;base64,test"),
            chat::text_part("b"),
        ]),
    ]);
    let rendered = render_mm(
        "{% if messages[0].role == 'system' %}{{ 'S:' + messages[0].content + '|' }}{% endif %}{% for message in messages[1:] %}{% if message.content is string %}{{ message.content }}{% else %}{% for item in message.content %}{% if item.type == 'image' %}<image>{% else %}{{ item.text }}{% endif %}{% endfor %}{% endif %}{% endfor %}",
        &request,
        None,
    )
    .unwrap();

    assert_eq!(rendered, "S:policy|a<image>b");
}

#[test]
fn chat_template_supports_pycompat_templates() {
    let request = sample_request(vec![chat::user("<think>hello")]);

    let rendered = render(
        Some(
            "{% for message in messages %}{% if message.content.startswith('<think>') %}think{% else %}plain{% endif %}{% endfor %}",
        ),
        &request,
    )
    .unwrap();

    assert_eq!(rendered, "think");
}

#[test]
fn chat_template_passes_continue_final_message_to_template() {
    let request = sample_request(vec![chat::assistant("The capital of")]);

    assert_eq!(
        render(
            Some("{% if continue_final_message %}continue{% else %}new{% endif %}"),
            &request,
        )
        .unwrap(),
        "new"
    );

    assert_eq!(
        render_as(
            Some("{% if continue_final_message %}continue{% else %}new{% endif %}"),
            &request,
            GenerationPromptMode::ContinueFinalAssistant,
        )
        .unwrap(),
        "continue"
    );
}

#[test]
fn chat_template_flattens_text_parts_for_string_templates() {
    let request = sample_request(vec![chat::user_parts(vec![
        chat::text_part("hello"),
        chat::text_part(" world"),
    ])]);

    let rendered = render(Some("{{ messages[0].content }}"), &request).unwrap();

    assert_eq!(rendered, "hello world");
}

#[test]
fn chat_template_exposes_developer_tools() {
    let request = sample_request(vec![chat::developer(
        "policy",
        Some(vec![Tool::Function {
            function: Function {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                }),
                strict: Some(true),
            },
        }]),
    )]);

    let rendered = render(
        Some("{{ messages[0].role }}|{{ messages[0].content }}|{{ messages[0].tools[0].function.name }}|{{ messages[0].tools[0].function.parameters.required[0] }}"),
        &request,
    )
    .unwrap();

    assert_eq!(rendered, "developer|policy|get_weather|city");
}

#[test]
fn chat_template_keeps_string_text_for_openai_detected_templates() {
    // The template tests `content is string` without looping over content
    // items, so `Auto` selects `String` and plain text arrives as a string.
    let request = sample_request(vec![chat::user("hello")]);

    let rendered = render(
        Some(
            "{%- for message in messages %}{%- if message.content is string %}{%- set content = message.content %}{{ content }}{%- endif %}{%- endfor %}",
        ),
        &request,
    )
    .unwrap();

    assert_eq!(rendered, "hello");
}

#[test]
fn chat_template_emits_openai_text_blocks_for_structured_templates() {
    let request = sample_request(vec![chat::user_parts(vec![
        chat::text_part("hello"),
        chat::text_part("world"),
    ])]);

    let rendered = render(
        Some(
            "{%- for message in messages %}{%- for item in message.content %}{{ item.text }}|{%- endfor %}{%- endfor %}",
        ),
        &request,
    )
    .unwrap();

    assert_eq!(rendered, "hello|world|");
}

#[test]
fn chat_template_requires_a_template() {
    let request = sample_request(vec![chat::user("hello")]);
    let error = render(None, &request).unwrap_err();

    assert_eq!(
        error.message,
        "chat template is required but none was configured"
    );
}

#[test]
fn chat_template_injects_special_tokens_into_context() {
    let request = sample_request(vec![chat::user("hello")]);
    let special_tokens = SpecialTokens {
        bos_token: Some(NamedSpecialToken::Text("<bos>".to_string())),
        ..Default::default()
    };

    let rendered = HfChatRenderer::new(
        Some("{{ bos_token }}|{{ bos_token is defined }}".to_string()),
        Map::new(),
        None,
    )
    .unwrap()
    .with_special_tokens(Some(special_tokens))
    .render(&request, GenerationPromptMode::StartNewAssistant)
    .unwrap();

    // Chat templates follow Python Jinja2, which prints booleans as
    // Python literals.
    assert_eq!(rendered, "<bos>|True");
}

#[test]
fn chat_template_exposes_assistant_reasoning_separately() {
    let request = sample_request(vec![chat::assistant_turn("inner", "outer", Vec::new())]);

    let rendered = render(
        Some("{{ messages[0].reasoning_content }}|{{ messages[0].content }}"),
        &request,
    )
    .unwrap();

    assert_eq!(rendered, "inner|outer");
}

#[test]
fn chat_template_forces_string_content_format_when_configured() {
    let request = sample_request(vec![chat::user_parts(vec![
        chat::text_part("hello"),
        chat::text_part(" world"),
    ])]);

    let rendered = HfChatRenderer::new(
        Some(
            "{%- if messages[0].content is string -%}{{ messages[0].content }}{%- else -%}{%- for item in messages[0].content %}{{ item.text }}|{%- endfor -%}{%- endif -%}".to_string(),
        ),
        Map::new(),
        Some(ChatTemplateContentFormat::String),
    )
    .unwrap()
    .render(&request, GenerationPromptMode::StartNewAssistant)
    .unwrap();

    assert_eq!(rendered, "hello world");
}

#[test]
fn chat_template_forces_openai_content_format_when_configured() {
    let request = sample_request(vec![chat::user_parts(vec![
        chat::text_part("hello"),
        chat::text_part(" world"),
    ])]);

    let rendered = HfChatRenderer::new(
        Some("{{ messages[0].content[0].text }}{{ messages[0].content[1].text }}".to_string()),
        Map::new(),
        Some(ChatTemplateContentFormat::OpenAi),
    )
    .unwrap()
    .render(&request, GenerationPromptMode::StartNewAssistant)
    .unwrap();

    assert_eq!(rendered, "hello world");
}

#[test]
fn chat_template_exposes_default_template_kwargs() {
    let request = sample_request(vec![chat::user("hello")]);
    let renderer = HfChatRenderer::new(
        Some("{{ enable_thinking }}|{{ default_only }}".to_string()),
        kwargs([
            ("enable_thinking".to_string(), Value::Bool(true)),
            ("default_only".to_string(), Value::String("x".to_string())),
        ]),
        None,
    )
    .unwrap();

    let rendered = renderer
        .render(&request, GenerationPromptMode::StartNewAssistant)
        .unwrap();

    assert_eq!(rendered, "True|x");
}

/// A request's template kwargs replace defaults of the same name and add
/// their own; the others keep their default values.
#[test]
fn request_template_kwargs_override_default_template_kwargs() {
    let mut request = sample_request(vec![chat::user("hello")]);
    request.template_kwargs = Some(kwargs([
        ("enable_thinking".to_string(), Value::Bool(true)),
        ("request_only".to_string(), Value::String("y".to_string())),
    ]));
    let renderer = HfChatRenderer::new(
        Some("{{ enable_thinking }}|{{ default_only }}|{{ request_only }}".to_string()),
        kwargs([
            ("enable_thinking".to_string(), Value::Bool(false)),
            ("default_only".to_string(), Value::String("x".to_string())),
        ]),
        None,
    )
    .unwrap();

    let rendered = renderer
        .render(&request, GenerationPromptMode::StartNewAssistant)
        .unwrap();

    assert_eq!(rendered, "True|x|y");
}

#[test]
fn chat_template_reasoning_effort_overrides_default_template_kwargs() {
    let mut request = sample_request(vec![chat::user("hello")]);
    request.reasoning_effort = Some("max".to_owned());

    let renderer = HfChatRenderer::new(
        Some("{{ reasoning_effort }}".to_string()),
        kwargs([(
            "reasoning_effort".to_string(),
            Value::String("medium".to_string()),
        )]),
        None,
    )
    .unwrap();

    let rendered = renderer
        .render(&request, GenerationPromptMode::StartNewAssistant)
        .unwrap();

    assert_eq!(rendered, "max");
}

#[test]
fn chatml_template_omits_reasoning_for_prior_assistant_messages() {
    let request = sample_request(vec![
        chat::user("Hi. Tell me about the capital of France in short"),
        chat::assistant_turn(
            "\nOkay, the user is asking... I think that's all.\n",
            "Paris is the capital of France.",
            Vec::new(),
        ),
        chat::user("Tell me about Paris more."),
    ]);

    let rendered = render(Some(CHATML_TEMPLATE), &request).unwrap();

    expect![[r#"
        <|im_start|>user
        Hi. Tell me about the capital of France in short<|im_end|>
        <|im_start|>assistant
        Paris is the capital of France.<|im_end|>
        <|im_start|>user
        Tell me about Paris more.<|im_end|>
        <|im_start|>assistant
    "#]]
    .assert_eq(&rendered);
}

#[test]
fn chatml_template_renders_single_user_prompt() {
    let request = sample_request(vec![chat::user("hello")]);

    let rendered = render(Some(CHATML_TEMPLATE), &request).unwrap();

    expect![[r#"
        <|im_start|>user
        hello<|im_end|>
        <|im_start|>assistant
    "#]]
    .assert_eq(&rendered);
}

#[test]
fn chatml_template_renders_system_user_prompt() {
    let request = sample_request(vec![chat::system("be brief"), chat::user("hello")]);

    let rendered = render(Some(CHATML_TEMPLATE), &request).unwrap();

    expect![[r#"
        <|im_start|>system
        be brief<|im_end|>
        <|im_start|>user
        hello<|im_end|>
        <|im_start|>assistant
    "#]]
    .assert_eq(&rendered);
}

#[test]
fn chatml_template_respects_forced_openai_content_format() {
    // The ChatML template emits content only when it is a string. A forced
    // `OpenAi` format still wins over detection, so the part list renders
    // as empty user content.
    let request = sample_request(vec![chat::user("hello")]);

    let rendered = HfChatRenderer::new(
        Some(CHATML_TEMPLATE.to_string()),
        Map::new(),
        Some(ChatTemplateContentFormat::OpenAi),
    )
    .unwrap()
    .render(&request, GenerationPromptMode::StartNewAssistant)
    .unwrap();

    expect![[r#"
        <|im_start|>user
        <|im_end|>
        <|im_start|>assistant
    "#]]
    .assert_eq(&rendered);
}

#[test]
fn chatml_template_keeps_reasoning_after_the_last_user_query() {
    let request = sample_request(vec![
        chat::user("What is 1 + 1?"),
        chat::assistant_turn("need simple arithmetic", "2", Vec::new()),
    ]);

    let rendered = render_as(
        Some(CHATML_TEMPLATE),
        &request,
        GenerationPromptMode::NoGenerationPrompt,
    )
    .unwrap();

    expect![[r#"
        <|im_start|>user
        What is 1 + 1?<|im_end|>
        <|im_start|>assistant
        <think>
        need simple arithmetic
        </think>

        2<|im_end|>
    "#]]
    .assert_eq(&rendered);
}

#[test]
fn chat_template_exposes_tools_to_templates_when_auto_enabled() {
    let mut request = sample_request(vec![chat::user("hello")]);
    request.tools = Some(vec![Tool::Function {
        function: Function {
            name: "get_weather".to_string(),
            description: Some("Get weather".to_string()),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }),
            strict: None,
        },
    }]);

    let rendered = render(
        Some("{{ tools[0].function.name }}|{{ tools[0].function.parameters.required[0] }}"),
        &request,
    )
    .unwrap();

    assert_eq!(rendered, "get_weather|city");
}

/// Tools reach the template in the shape of the client's tool objects:
/// optional fields the tool leaves unset are absent rather than null.
#[test]
fn chat_template_omits_unset_optional_tool_fields() {
    let parameters = serde_json::json!({
        "type": "object",
        "properties": {"city": {"type": "string"}},
    });
    let mut request = sample_request(vec![chat::user("hello")]);
    request.tools = Some(vec![
        Tool::Function {
            function: Function {
                name: "get_weather".to_string(),
                description: None,
                parameters: parameters.clone(),
                strict: None,
            },
        },
        Tool::Function {
            function: Function {
                name: "get_time".to_string(),
                description: Some("Get time".to_string()),
                parameters: parameters.clone(),
                strict: Some(true),
            },
        },
    ]);

    let rendered = render(Some("{{ tools | tojson }}"), &request).unwrap();

    assert_eq!(
        serde_json::from_str::<Value>(&rendered).unwrap(),
        serde_json::json!([
            {
                "type": "function",
                "function": {"name": "get_weather", "parameters": parameters},
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get time",
                    "parameters": parameters,
                    "strict": true,
                },
            },
        ])
    );
}

#[test]
fn chat_template_exposes_assistant_tool_calls_and_tool_messages() {
    let request = sample_request(vec![
        chat::assistant_turn(
            "",
            "",
            vec![chat::tool_call(
                "call_1",
                "get_weather",
                r#"{"city":"Paris"}"#,
            )],
        ),
        chat::tool_response("Sunny", "call_1"),
    ]);

    let rendered = render(
        Some(
            "{{ messages[0].tool_calls[0].function.name }}|{{ messages[0].tool_calls[0].function.arguments.city }}|{{ messages[1].tool_call_id }}|{{ messages[1].content }}",
        ),
        &request,
    )
    .unwrap();

    assert_eq!(rendered, "get_weather|Paris|call_1|Sunny");
}

#[test]
fn chat_template_tool_call_argument_items_method_is_not_shadowed_by_field() {
    // An `items` key in tool arguments must not shadow the dict `items()`
    // method (see `TemplateMap`), and iteration follows the key order of
    // the arguments JSON.
    let request = sample_request(vec![chat::assistant_turn(
        "",
        "",
        vec![chat::tool_call(
            "call_1",
            "add",
            r#"{"items":"operands","x":2,"y":1.0}"#,
        )],
    )]);

    let rendered = render(
        Some(
            "{%- set arguments = messages[0].tool_calls[0].function.arguments -%}
{%- for key, value in arguments.items() -%}{{ key }}={{ value }};{%- endfor -%}
|{{ arguments['items'] }}",
        ),
        &request,
    )
    .unwrap();

    assert_eq!(rendered, "items=operands;x=2;y=1.0;|operands");
}

// The content-shape detection, prior assistant turns through the ChatML template, and the
// key order and number spelling of tool arguments.

/// Renders `request` for a new assistant turn.
fn hf_render(
    template: &str,
    format: Option<ChatTemplateContentFormat>,
    request: &Render,
) -> String {
    HfChatRenderer::new(Some(template.to_owned()), Map::new(), format)
        .expect("template should compile")
        .render(request, GenerationPromptMode::StartNewAssistant)
        .expect("render should succeed")
}

fn base_request(messages: Vec<Message>) -> Render {
    chat::request(messages)
}

/// One user message with two text parts, rendered without a generation prompt.
///
/// Flattened to a string, the content reads `"ab"`; in the OpenAI format it is
/// a list of two text-part objects.
fn multipart_user() -> Render {
    base_request(vec![chat::user_parts(vec![
        chat::text_part("a"),
        chat::text_part("b"),
    ])])
}

#[test]
fn auto_detection_treats_alias_loop_template_as_string() {
    // The template only aliases content as `parts` and tests the alias with
    // `is string`. Without a loop over a message's `content`, `Auto` selects
    // the string format, so the parts arrive flattened and the string branch
    // renders.
    let template = "{%- for message in messages -%}{%- set parts = message.content -%}\
{%- if parts is string -%}STR:{{ parts }}{%- else -%}LIST{%- endif -%}{%- endfor -%}";
    let rendered = hf_render(template, None, &multipart_user());
    assert_eq!(rendered, "STR:ab");
}

#[test]
fn auto_detection_treats_length_and_index_access_template_as_string() {
    // Length / index access on content must not be mistaken for a content loop.
    let template = "{%- for message in messages -%}len={{ message.content|length }};\
first={{ message.content[0] }}{%- endfor -%}";
    let rendered = hf_render(template, None, &multipart_user());
    // String content "ab": length 2, index 0 is the character 'a'.
    assert_eq!(rendered, "len=2;first=a");
}

#[test]
fn auto_detection_treats_direct_content_loop_template_as_openai() {
    let template = "{%- for message in messages -%}\
{%- for part in message.content -%}{{ part.text }}|{%- endfor -%}{%- endfor -%}";
    let rendered = hf_render(template, None, &multipart_user());
    // Each part surfaces as a structured `{text: ...}` object.
    assert_eq!(rendered, "a|b|");
}

fn chat_history() -> Render {
    base_request(vec![
        chat::user("What is the capital of France?"),
        chat::assistant("The capital of France is Paris."),
        chat::user("And of Italy?"),
    ])
}

#[test]
fn chatml_preserves_prior_assistant_completion_text_byte_identically() {
    let request = chat_history();
    let rendered = hf_render(CHATML_TEMPLATE, None, &request);

    // The prior assistant turn's visible completion text must survive
    // intact inside the rendered prompt, framed by the assistant turn markers.
    assert!(
        rendered.contains("<|im_start|>assistant\nThe capital of France is Paris.<|im_end|>"),
        "assistant completion text should round-trip byte-for-byte, got:\n{rendered}"
    );
}

/// Deliberately non-alphabetical key order plus mixed numeric spellings.
///
/// Key order survives because the workspace builds `serde_json` with
/// `preserve_order` and the renderer exposes the parsed tool-call arguments to
/// templates as insertion-ordered maps.
const MIXED_ARGS: &str = r#"{"zulu":2,"alpha":1.00,"mike":"hi","delta":[3,4]}"#;

fn assistant_tool_call_history(arguments: &str) -> Vec<Message> {
    vec![
        chat::user("Run the tool."),
        chat::assistant_turn(
            "",
            "",
            vec![chat::tool_call("call-1", "do_thing", arguments)],
        ),
        chat::tool_response("{\"ok\":true}", "call-1"),
    ]
}

#[test]
fn configured_template_preserves_tool_argument_order_via_items_iteration() {
    // The HF path exposes tool-call arguments as a structured map. Iterating
    // `.items()` must yield the original (non-alphabetical) key order, and the
    // numeric spellings must match serde_json's normalized output.
    let request = base_request(assistant_tool_call_history(MIXED_ARGS));
    let template = "{%- set args = messages[1].tool_calls[0].function.arguments -%}\
{%- for key, value in args.items() -%}{{ key }}={{ value }};{%- endfor -%}";
    let rendered = hf_render(template, None, &request);

    // Key order preserved; integer stays 2; `1.00` -> 1.0; array stringified.
    assert_eq!(rendered, "zulu=2;alpha=1.0;mike=hi;delta=[3, 4];");
}

#[test]
fn configured_template_tojson_preserves_key_order_and_number_precision() {
    // Rendering the arguments back through HF `tojson` must keep insertion order
    // (no implicit sort) and the normalized number spelling.
    let request = base_request(assistant_tool_call_history(MIXED_ARGS));
    let template =
        "{{ messages[1].tool_calls[0].function.arguments|tojson(separators=[',', ':']) }}";
    let rendered = hf_render(template, None, &request);

    assert_eq!(
        rendered,
        r#"{"zulu":2,"alpha":1.0,"mike":"hi","delta":[3,4]}"#
    );
}
