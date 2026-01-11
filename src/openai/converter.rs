//! OpenAI → Kiro 协议转换器
//!
//! 负责将 OpenAI Chat Completions API 请求格式转换为 Kiro API 请求格式

use uuid::Uuid;

use crate::kiro::model::requests::conversation::{
    AssistantMessage, ConversationState, CurrentMessage, HistoryAssistantMessage,
    HistoryUserMessage, KiroImage, Message, UserInputMessage, UserInputMessageContext, UserMessage,
};
use crate::kiro::model::requests::tool::{
    InputSchema, Tool, ToolResult, ToolSpecification, ToolUseEntry,
};

use super::types::{ChatCompletionRequest, ChatMessage, ContentPart, MessageContent};

/// 模型映射：将模型名映射到 Kiro 模型 ID
///
/// 支持的映射：
/// - *sonnet* → claude-sonnet-4.5
/// - *opus* → claude-opus-4.5
/// - *haiku* 或其他 → claude-haiku-4.5（默认）
pub fn map_model(model: &str) -> Option<String> {
    let model_lower = model.to_lowercase();

    if model_lower.contains("sonnet") {
        Some("claude-sonnet-4.5".to_string())
    } else if model_lower.contains("opus") {
        Some("claude-opus-4.5".to_string())
    } else {
        // haiku 或其他未知模型默认使用 haiku
        Some("claude-haiku-4.5".to_string())
    }
}

/// 转换结果
#[derive(Debug)]
pub struct ConversionResult {
    /// 转换后的 Kiro 请求
    pub conversation_state: ConversationState,
    /// 原始模型名（用于响应）
    pub original_model: String,
}

/// 转换错误
#[derive(Debug)]
pub enum ConversionError {
    UnsupportedModel(String),
    EmptyMessages,
    InvalidImageUrl(String),
}

impl std::fmt::Display for ConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConversionError::UnsupportedModel(model) => write!(f, "模型不支持: {}", model),
            ConversionError::EmptyMessages => write!(f, "消息列表为空"),
            ConversionError::InvalidImageUrl(url) => write!(f, "无效的图片 URL: {}", url),
        }
    }
}

impl std::error::Error for ConversionError {}

/// 将 OpenAI 请求转换为 Kiro 请求
pub fn convert_request(req: &ChatCompletionRequest) -> Result<ConversionResult, ConversionError> {
    // 1. 映射模型
    let model_id = map_model(&req.model)
        .ok_or_else(|| ConversionError::UnsupportedModel(req.model.clone()))?;

    // 2. 检查消息列表
    if req.messages.is_empty() {
        return Err(ConversionError::EmptyMessages);
    }

    // 3. 生成会话 ID 和代理 ID
    let conversation_id = Uuid::new_v4().to_string();
    let agent_continuation_id = Uuid::new_v4().to_string();

    // 4. 提取系统消息和构建历史
    let (system_content, history, last_user_content, last_images, tool_results) =
        process_messages(&req.messages, &model_id)?;

    // 5. 转换工具定义
    let mut tools = convert_tools(&req.tools);

    // 6. 收集历史中使用的工具名称，为缺失的工具生成占位符定义
    let history_tool_names = collect_history_tool_names(&history);
    let existing_tool_names: std::collections::HashSet<_> = tools
        .iter()
        .map(|t| t.tool_specification.name.to_lowercase())
        .collect();

    for tool_name in history_tool_names {
        if !existing_tool_names.contains(&tool_name.to_lowercase()) {
            tools.push(create_placeholder_tool(&tool_name));
        }
    }

    // 7. 验证并过滤 tool_use/tool_result 配对
    let validated_tool_results = validate_tool_pairing(&history, &tool_results);

    // 8. 构建 UserInputMessageContext
    let mut context = UserInputMessageContext::new();
    if !tools.is_empty() {
        context = context.with_tools(tools);
    }
    if !validated_tool_results.is_empty() {
        context = context.with_tool_results(validated_tool_results);
    }

    // 9. 构建当前消息
    let mut user_input = UserInputMessage::new(last_user_content, &model_id)
        .with_context(context)
        .with_origin("AI_EDITOR");

    if !last_images.is_empty() {
        user_input = user_input.with_images(last_images);
    }

    let current_message = CurrentMessage::new(user_input);

    // 10. 构建完整历史（包含系统消息）
    let mut full_history = Vec::new();

    // 添加系统消息作为 user + assistant 配对
    if !system_content.is_empty() {
        let user_msg = HistoryUserMessage::new(&system_content, &model_id);
        full_history.push(Message::User(user_msg));

        let assistant_msg = HistoryAssistantMessage::new("I will follow these instructions.");
        full_history.push(Message::Assistant(assistant_msg));
    }

    // 添加对话历史
    full_history.extend(history);

    // 11. 构建 ConversationState
    let conversation_state = ConversationState::new(conversation_id)
        .with_agent_continuation_id(agent_continuation_id)
        .with_agent_task_type("vibe")
        .with_chat_trigger_type("MANUAL")
        .with_current_message(current_message)
        .with_history(full_history);

    Ok(ConversionResult {
        conversation_state,
        original_model: req.model.clone(),
    })
}

/// 处理消息列表，提取系统消息、历史和最后的用户消息
fn process_messages(
    messages: &[ChatMessage],
    model_id: &str,
) -> Result<(String, Vec<Message>, String, Vec<KiroImage>, Vec<ToolResult>), ConversionError> {
    let mut system_content = String::new();
    let mut history: Vec<Message> = Vec::new();
    let mut last_user_content = String::new();
    let mut last_images: Vec<KiroImage> = Vec::new();
    let mut tool_results: Vec<ToolResult> = Vec::new();

    // 收集用户消息缓冲区
    let mut user_buffer: Vec<(String, Vec<KiroImage>)> = Vec::new();

    for (i, msg) in messages.iter().enumerate() {
        let is_last = i == messages.len() - 1;

        match msg.role.as_str() {
            "system" => {
                // 系统消息
                let text = extract_text_content(&msg.content);
                if !system_content.is_empty() {
                    system_content.push('\n');
                }
                system_content.push_str(&text);
            }
            "user" => {
                let (text, images) = extract_content_with_images(&msg.content)?;

                if is_last {
                    // 最后一条用户消息作为 currentMessage
                    last_user_content = text;
                    last_images = images;
                } else {
                    // 加入缓冲区
                    user_buffer.push((text, images));
                }
            }
            "assistant" => {
                // 处理累积的用户消息
                if !user_buffer.is_empty() {
                    let merged = merge_user_buffer(&user_buffer, model_id);
                    history.push(Message::User(merged));
                    user_buffer.clear();
                }

                // 添加 assistant 消息
                let assistant = convert_assistant_message(msg)?;
                history.push(Message::Assistant(assistant));
            }
            "tool" => {
                // 工具结果消息
                if let Some(tool_call_id) = &msg.tool_call_id {
                    let content = extract_text_content(&msg.content);
                    tool_results.push(ToolResult::success(tool_call_id, content));
                }
            }
            _ => {}
        }
    }

    // 处理结尾的孤立用户消息（非最后一条）
    if !user_buffer.is_empty() {
        let merged = merge_user_buffer(&user_buffer, model_id);
        history.push(Message::User(merged));

        // 自动配对一个 "OK" 的 assistant 响应
        let auto_assistant = HistoryAssistantMessage::new("OK");
        history.push(Message::Assistant(auto_assistant));
    }

    Ok((
        system_content,
        history,
        last_user_content,
        last_images,
        tool_results,
    ))
}

/// 提取文本内容
fn extract_text_content(content: &Option<MessageContent>) -> String {
    match content {
        Some(MessageContent::Text(s)) => s.clone(),
        Some(MessageContent::Parts(parts)) => {
            let mut texts = Vec::new();
            for part in parts {
                if let ContentPart::Text { text } = part {
                    texts.push(text.clone());
                }
            }
            texts.join("\n")
        }
        None => String::new(),
    }
}

/// 提取内容和图片
fn extract_content_with_images(
    content: &Option<MessageContent>,
) -> Result<(String, Vec<KiroImage>), ConversionError> {
    let mut texts = Vec::new();
    let mut images = Vec::new();

    match content {
        Some(MessageContent::Text(s)) => {
            texts.push(s.clone());
        }
        Some(MessageContent::Parts(parts)) => {
            for part in parts {
                match part {
                    ContentPart::Text { text } => {
                        texts.push(text.clone());
                    }
                    ContentPart::ImageUrl { image_url } => {
                        if let Some(image) = parse_image_url(&image_url.url)? {
                            images.push(image);
                        }
                    }
                }
            }
        }
        None => {}
    }

    Ok((texts.join("\n"), images))
}

/// 解析图片 URL（支持 base64 data URL 和 HTTP URL）
fn parse_image_url(url: &str) -> Result<Option<KiroImage>, ConversionError> {
    if url.starts_with("data:") {
        // data:image/png;base64,xxxxx
        let parts: Vec<&str> = url.splitn(2, ',').collect();
        if parts.len() != 2 {
            return Err(ConversionError::InvalidImageUrl(url.to_string()));
        }

        let header = parts[0];
        let data = parts[1];

        // 解析 media type
        let format = if header.contains("image/png") {
            "png"
        } else if header.contains("image/jpeg") || header.contains("image/jpg") {
            "jpeg"
        } else if header.contains("image/gif") {
            "gif"
        } else if header.contains("image/webp") {
            "webp"
        } else {
            return Err(ConversionError::InvalidImageUrl(url.to_string()));
        };

        Ok(Some(KiroImage::from_base64(format, data.to_string())))
    } else if url.starts_with("http://") || url.starts_with("https://") {
        // HTTP URL - 暂不支持，需要下载图片
        // TODO: 实现 HTTP URL 图片下载
        tracing::warn!("HTTP 图片 URL 暂不支持: {}", url);
        Ok(None)
    } else {
        Err(ConversionError::InvalidImageUrl(url.to_string()))
    }
}

/// 合并用户消息缓冲区
fn merge_user_buffer(buffer: &[(String, Vec<KiroImage>)], model_id: &str) -> HistoryUserMessage {
    let mut content_parts = Vec::new();
    let mut all_images = Vec::new();

    for (text, images) in buffer {
        if !text.is_empty() {
            content_parts.push(text.clone());
        }
        all_images.extend(images.clone());
    }

    let content = content_parts.join("\n");
    let mut user_msg = UserMessage::new(&content, model_id);

    if !all_images.is_empty() {
        user_msg = user_msg.with_images(all_images);
    }

    HistoryUserMessage {
        user_input_message: user_msg,
    }
}

/// 转换 assistant 消息
fn convert_assistant_message(msg: &ChatMessage) -> Result<HistoryAssistantMessage, ConversionError> {
    let text_content = extract_text_content(&msg.content);
    let mut tool_uses = Vec::new();

    // 处理工具调用
    if let Some(tool_calls) = &msg.tool_calls {
        for call in tool_calls {
            let input: serde_json::Value =
                serde_json::from_str(&call.function.arguments).unwrap_or(serde_json::json!({}));
            tool_uses.push(
                ToolUseEntry::new(&call.id, &call.function.name).with_input(input),
            );
        }
    }

    let mut assistant = AssistantMessage::new(text_content);
    if !tool_uses.is_empty() {
        assistant = assistant.with_tool_uses(tool_uses);
    }

    Ok(HistoryAssistantMessage {
        assistant_response_message: assistant,
    })
}

/// 转换工具定义
fn convert_tools(tools: &Option<Vec<super::types::Tool>>) -> Vec<Tool> {
    let Some(tools) = tools else {
        return Vec::new();
    };

    tools
        .iter()
        .filter(|t| t.tool_type == "function")
        .map(|t| {
            let description = t.function.description.clone().unwrap_or_default();
            // 限制描述长度为 10000 字符
            let description = match description.char_indices().nth(10000) {
                Some((idx, _)) => description[..idx].to_string(),
                None => description,
            };

            let input_schema = t
                .function
                .parameters
                .clone()
                .map(|p| InputSchema::from_json(serde_json::json!(p)))
                .unwrap_or_else(|| {
                    InputSchema::from_json(serde_json::json!({
                        "type": "object",
                        "properties": {},
                        "required": []
                    }))
                });

            Tool {
                tool_specification: ToolSpecification {
                    name: t.function.name.clone(),
                    description,
                    input_schema,
                },
            }
        })
        .collect()
}

/// 收集历史消息中使用的所有工具名称
fn collect_history_tool_names(history: &[Message]) -> Vec<String> {
    let mut tool_names = Vec::new();

    for msg in history {
        if let Message::Assistant(assistant_msg) = msg {
            if let Some(ref tool_uses) = assistant_msg.assistant_response_message.tool_uses {
                for tool_use in tool_uses {
                    if !tool_names.contains(&tool_use.name) {
                        tool_names.push(tool_use.name.clone());
                    }
                }
            }
        }
    }

    tool_names
}

/// 为历史中使用但不在 tools 列表中的工具创建占位符定义
fn create_placeholder_tool(name: &str) -> Tool {
    Tool {
        tool_specification: ToolSpecification {
            name: name.to_string(),
            description: "Tool used in conversation history".to_string(),
            input_schema: InputSchema::from_json(serde_json::json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": true
            })),
        },
    }
}

/// 验证并过滤 tool_use/tool_result 配对
fn validate_tool_pairing(history: &[Message], tool_results: &[ToolResult]) -> Vec<ToolResult> {
    use std::collections::HashSet;

    // 收集所有历史中的 tool_use_id
    let mut valid_tool_use_ids: HashSet<String> = HashSet::new();

    for msg in history {
        if let Message::Assistant(assistant_msg) = msg {
            if let Some(ref tool_uses) = assistant_msg.assistant_response_message.tool_uses {
                for tool_use in tool_uses {
                    valid_tool_use_ids.insert(tool_use.tool_use_id.clone());
                }
            }
        }
    }

    // 过滤并验证 tool_results
    let mut filtered_results = Vec::new();

    for result in tool_results {
        if valid_tool_use_ids.contains(&result.tool_use_id) {
            filtered_results.push(result.clone());
            valid_tool_use_ids.remove(&result.tool_use_id);
        } else {
            tracing::warn!(
                "跳过孤立的 tool_result：找不到对应的 tool_use，tool_use_id={}",
                result.tool_use_id
            );
        }
    }

    // 检测孤立的 tool_use
    for orphaned_id in &valid_tool_use_ids {
        tracing::warn!(
            "检测到孤立的 tool_use：找不到对应的 tool_result，tool_use_id={}",
            orphaned_id
        );
    }

    filtered_results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_model_claude() {
        assert_eq!(map_model("claude-sonnet-4").unwrap(), "claude-sonnet-4.5");
        assert_eq!(map_model("claude-opus-4").unwrap(), "claude-opus-4.5");
        assert_eq!(map_model("claude-haiku-4").unwrap(), "claude-haiku-4.5");
    }

    #[test]
    fn test_map_model_default_to_haiku() {
        // 未知模型默认使用 haiku
        assert_eq!(map_model("gpt-4").unwrap(), "claude-haiku-4.5");
        assert_eq!(map_model("unknown-model").unwrap(), "claude-haiku-4.5");
    }

    #[test]
    fn test_parse_image_url_base64() {
        let url = "data:image/png;base64,iVBORw0KGgo=";
        let result = parse_image_url(url).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_parse_image_url_invalid() {
        let url = "invalid://url";
        let result = parse_image_url(url);
        assert!(result.is_err());
    }
}
