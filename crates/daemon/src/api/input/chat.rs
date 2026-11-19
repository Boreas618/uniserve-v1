//! The request side of chat: a validated chat request as the epoch's processing program reads
//! it (`scheduling::processing::Render`).
//!
//! The program renders the prompt with the family's template; the daemon states what the prompt
//! is made of, in the OpenAI shape the client sent it in: the messages, the tools the model may
//! call (`ChatCompletionRequest::callable_tools`), the request's template arguments and reasoning
//! effort, and the displayed size of each image it fetched, in the order the conversation's
//! image parts appear.

use scheduling::processing::{self as interface, ImageSize, Render};

use crate::api::input::ImageInput;
use crate::api::protocol::{
    ChatCompletionRequest, ChatMessage, ContentPart, MessageContent, Tool, ToolCall,
};

/// The render message of `request`, whose fetched images are `images` in
/// `ChatCompletionRequest::image_urls` order.
pub fn render(request: &ChatCompletionRequest, images: &[ImageInput]) -> Render {
    Render {
        messages: request.messages.iter().map(message).collect(),
        tools: callable_tools(request),
        template_kwargs: request.chat_template_kwargs.as_ref().map(|kwargs| {
            kwargs
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect()
        }),
        reasoning_effort: request.reasoning_effort.and_then(|effort| {
            // The protocol's own name for the effort, which the template reads.
            serde_json::to_value(effort)
                .ok()
                .and_then(|value| value.as_str().map(str::to_owned))
        }),
        images: images
            .iter()
            .map(|image| ImageSize {
                width: image.width(),
                height: image.height(),
            })
            .collect(),
    }
}

/// The tools the model may call, which its template is given and its tool calls are parsed
/// against; `None` when it may call none.
pub fn callable_tools(request: &ChatCompletionRequest) -> Option<Vec<interface::Tool>> {
    request.callable_tools().map(tools)
}

/// One message in the interface's shape.
fn message(message: &ChatMessage) -> interface::Message {
    match message {
        ChatMessage::System { content, .. } => interface::Message::System {
            content: content_of(content),
        },
        ChatMessage::Developer {
            content,
            tools: declared,
            ..
        } => interface::Message::Developer {
            content: content_of(content),
            tools: declared.as_deref().map(tools),
        },
        ChatMessage::User { content, .. } => interface::Message::User {
            content: content_of(content),
        },
        ChatMessage::Assistant {
            content,
            tool_calls,
            reasoning,
            ..
        } => interface::Message::Assistant {
            content: content.as_ref().map(content_of),
            reasoning: reasoning.clone(),
            tool_calls: tool_calls
                .as_deref()
                .unwrap_or_default()
                .iter()
                .map(tool_call)
                .collect(),
        },
        ChatMessage::Tool {
            content,
            tool_call_id,
        } => interface::Message::Tool {
            content: content_of(content),
            tool_call_id: tool_call_id.clone(),
        },
    }
}

/// A message's content in the interface's shape: an image part stands for the next fetched
/// image.
fn content_of(content: &MessageContent) -> interface::Content {
    match content {
        MessageContent::Text(text) => interface::Content::Text(text.clone()),
        MessageContent::Parts(parts) => interface::Content::Parts(
            parts
                .iter()
                .map(|part| match part {
                    ContentPart::Text { text } => {
                        interface::ContentPart::Text { text: text.clone() }
                    }
                    ContentPart::ImageUrl { .. } => interface::ContentPart::Image,
                })
                .collect(),
        ),
    }
}

/// Function tools in the interface's shape.
fn tools(tools: &[Tool]) -> Vec<interface::Tool> {
    tools
        .iter()
        .map(|Tool::Function { function }| interface::Tool::Function {
            function: interface::Function {
                name: function.name.clone(),
                description: function.description.clone(),
                parameters: function.parameters.clone(),
                strict: function.strict,
            },
        })
        .collect()
}

/// An earlier assistant turn's call in the interface's shape.
fn tool_call(call: &ToolCall) -> interface::ToolCall {
    interface::ToolCall {
        id: call.id.clone(),
        function: interface::FunctionCall {
            name: call.function.name.clone(),
            arguments: call.function.arguments.clone(),
        },
    }
}
