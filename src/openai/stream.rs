//! OpenAI 流式响应处理模块
//!
//! 实现 Kiro → OpenAI 流式响应转换

use std::collections::HashMap;

use uuid::Uuid;

use crate::kiro::model::events::Event;

use super::types::{
    ChatCompletionChunk, ChunkChoice, Delta, DeltaFunction, DeltaToolCall, Usage,
};

/// 上下文窗口大小（200k tokens）
const CONTEXT_WINDOW_SIZE: i32 = 200_000;

/// 流处理上下文
pub struct StreamContext {
    /// 请求的模型名称
    pub model: String,
    /// 响应 ID
    pub response_id: String,
    /// 创建时间戳
    pub created: i64,
    /// 输入 tokens（估算值）
    pub input_tokens: i32,
    /// 从 contextUsageEvent 计算的实际输入 tokens
    pub context_input_tokens: Option<i32>,
    /// 输出 tokens 累计
    pub output_tokens: i32,
    /// 是否已发送初始 chunk
    pub initial_sent: bool,
    /// 是否有工具调用
    pub has_tool_use: bool,
    /// 工具调用索引映射 (tool_id -> index)
    pub tool_indices: HashMap<String, i32>,
    /// 下一个工具索引
    pub next_tool_index: i32,
    /// 是否在流式响应中包含 usage
    pub include_usage: bool,
    /// 停止原因
    pub finish_reason: Option<String>,
}

impl StreamContext {
    /// 创建新的流处理上下文
    pub fn new(model: impl Into<String>, input_tokens: i32, include_usage: bool) -> Self {
        Self {
            model: model.into(),
            response_id: format!("chatcmpl-{}", Uuid::new_v4().to_string().replace('-', "")),
            created: chrono::Utc::now().timestamp(),
            input_tokens,
            context_input_tokens: None,
            output_tokens: 0,
            initial_sent: false,
            has_tool_use: false,
            tool_indices: HashMap::new(),
            next_tool_index: 0,
            include_usage,
            finish_reason: None,
        }
    }

    /// 生成初始 chunk（包含 role）
    pub fn generate_initial_chunk(&mut self) -> ChatCompletionChunk {
        self.initial_sent = true;
        ChatCompletionChunk {
            id: self.response_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: None,
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
            system_fingerprint: None,
        }
    }

    /// 处理 Kiro 事件并转换为 OpenAI chunk
    pub fn process_kiro_event(&mut self, event: &Event) -> Vec<ChatCompletionChunk> {
        match event {
            Event::AssistantResponse(resp) => self.process_assistant_response(&resp.content),
            Event::ToolUse(tool_use) => self.process_tool_use(tool_use),
            Event::ContextUsage(context_usage) => {
                // 从上下文使用百分比计算实际的 input_tokens
                let actual_input_tokens = (context_usage.context_usage_percentage
                    * (CONTEXT_WINDOW_SIZE as f64)
                    / 100.0) as i32;
                self.context_input_tokens = Some(actual_input_tokens);
                tracing::debug!(
                    "收到 contextUsageEvent: {}%, 计算 input_tokens: {}",
                    context_usage.context_usage_percentage,
                    actual_input_tokens
                );
                Vec::new()
            }
            Event::Error {
                error_code,
                error_message,
            } => {
                tracing::error!("收到错误事件: {} - {}", error_code, error_message);
                Vec::new()
            }
            Event::Exception {
                exception_type,
                message,
            } => {
                if exception_type == "ContentLengthExceededException" {
                    self.finish_reason = Some("length".to_string());
                }
                tracing::warn!("收到异常事件: {} - {}", exception_type, message);
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    /// 处理助手响应事件
    fn process_assistant_response(&mut self, content: &str) -> Vec<ChatCompletionChunk> {
        if content.is_empty() {
            return Vec::new();
        }

        // 估算 tokens
        self.output_tokens += estimate_tokens(content);

        // 过滤 thinking 标签（OpenAI 格式不支持 thinking）
        let filtered_content = filter_thinking_tags(content);
        if filtered_content.is_empty() {
            return Vec::new();
        }

        vec![ChatCompletionChunk {
            id: self.response_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: Some(filtered_content),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
            system_fingerprint: None,
        }]
    }

    /// 处理工具使用事件
    fn process_tool_use(
        &mut self,
        tool_use: &crate::kiro::model::events::ToolUseEvent,
    ) -> Vec<ChatCompletionChunk> {
        self.has_tool_use = true;

        // 获取或分配工具索引
        let tool_index = if let Some(&idx) = self.tool_indices.get(&tool_use.tool_use_id) {
            idx
        } else {
            let idx = self.next_tool_index;
            self.next_tool_index += 1;
            self.tool_indices
                .insert(tool_use.tool_use_id.clone(), idx);
            idx
        };

        let is_first_chunk = !self.tool_indices.contains_key(&tool_use.tool_use_id)
            || self.tool_indices.get(&tool_use.tool_use_id) == Some(&tool_index)
                && tool_use.input.is_empty();

        // 估算 tokens
        if !tool_use.input.is_empty() {
            self.output_tokens += (tool_use.input.len() as i32 + 3) / 4;
        }

        // 构建 tool_call delta
        let tool_call = if is_first_chunk || tool_use.input.is_empty() {
            // 第一个 chunk 包含完整信息
            DeltaToolCall {
                index: tool_index,
                id: Some(tool_use.tool_use_id.clone()),
                call_type: Some("function".to_string()),
                function: Some(DeltaFunction {
                    name: Some(tool_use.name.clone()),
                    arguments: if tool_use.input.is_empty() {
                        None
                    } else {
                        Some(tool_use.input.clone())
                    },
                }),
            }
        } else {
            // 后续 chunk 只包含增量参数
            DeltaToolCall {
                index: tool_index,
                id: None,
                call_type: None,
                function: Some(DeltaFunction {
                    name: None,
                    arguments: Some(tool_use.input.clone()),
                }),
            }
        };

        vec![ChatCompletionChunk {
            id: self.response_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: None,
                    content: None,
                    tool_calls: Some(vec![tool_call]),
                },
                finish_reason: None,
            }],
            usage: None,
            system_fingerprint: None,
        }]
    }

    /// 生成最终 chunk
    pub fn generate_final_chunk(&mut self) -> Vec<ChatCompletionChunk> {
        let mut chunks = Vec::new();

        // 确定 finish_reason
        let finish_reason = if let Some(ref reason) = self.finish_reason {
            reason.clone()
        } else if self.has_tool_use {
            "tool_calls".to_string()
        } else {
            "stop".to_string()
        };

        // 发送带有 finish_reason 的 chunk
        chunks.push(ChatCompletionChunk {
            id: self.response_id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some(finish_reason),
            }],
            usage: None,
            system_fingerprint: None,
        });

        // 如果需要包含 usage，发送最后一个带 usage 的 chunk
        if self.include_usage {
            let final_input_tokens = self.context_input_tokens.unwrap_or(self.input_tokens);
            chunks.push(ChatCompletionChunk {
                id: self.response_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created: self.created,
                model: self.model.clone(),
                choices: vec![],
                usage: Some(Usage {
                    prompt_tokens: final_input_tokens,
                    completion_tokens: self.output_tokens,
                    total_tokens: final_input_tokens + self.output_tokens,
                }),
                system_fingerprint: None,
            });
        }

        chunks
    }

    /// 获取最终的 usage
    pub fn get_usage(&self) -> Usage {
        let final_input_tokens = self.context_input_tokens.unwrap_or(self.input_tokens);
        Usage {
            prompt_tokens: final_input_tokens,
            completion_tokens: self.output_tokens,
            total_tokens: final_input_tokens + self.output_tokens,
        }
    }
}

/// 过滤 thinking 标签
fn filter_thinking_tags(content: &str) -> String {
    // 简单过滤：移除 <thinking>...</thinking> 标签及其内容
    let mut result = content.to_string();

    // 移除完整的 thinking 块
    while let Some(start) = result.find("<thinking>") {
        if let Some(end) = result[start..].find("</thinking>") {
            let end_pos = start + end + "</thinking>".len();
            // 移除 thinking 块后面的换行符
            let after = &result[end_pos..];
            let trim_len = if after.starts_with("\n\n") {
                2
            } else if after.starts_with('\n') {
                1
            } else {
                0
            };
            result = format!("{}{}", &result[..start], &result[end_pos + trim_len..]);
        } else {
            // 没有找到结束标签，移除开始标签后的所有内容
            result = result[..start].to_string();
            break;
        }
    }

    result
}

/// 简单的 token 估算
fn estimate_tokens(text: &str) -> i32 {
    let chars: Vec<char> = text.chars().collect();
    let mut chinese_count = 0;
    let mut other_count = 0;

    for c in &chars {
        if *c >= '\u{4E00}' && *c <= '\u{9FFF}' {
            chinese_count += 1;
        } else {
            other_count += 1;
        }
    }

    // 中文约 1.5 字符/token，英文约 4 字符/token
    let chinese_tokens = (chinese_count * 2 + 2) / 3;
    let other_tokens = (other_count + 3) / 4;

    (chinese_tokens + other_tokens).max(1)
}

/// 将 chunk 转换为 SSE 字符串
pub fn chunk_to_sse(chunk: &ChatCompletionChunk) -> String {
    format!(
        "data: {}\n\n",
        serde_json::to_string(chunk).unwrap_or_default()
    )
}

/// 生成 [DONE] SSE 字符串
pub fn done_sse() -> String {
    "data: [DONE]\n\n".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_thinking_tags() {
        assert_eq!(filter_thinking_tags("hello"), "hello");
        assert_eq!(
            filter_thinking_tags("<thinking>test</thinking>\n\nhello"),
            "hello"
        );
        assert_eq!(
            filter_thinking_tags("before<thinking>test</thinking>\n\nafter"),
            "beforeafter"
        );
    }

    #[test]
    fn test_estimate_tokens() {
        assert!(estimate_tokens("Hello") > 0);
        assert!(estimate_tokens("你好") > 0);
    }

    #[test]
    fn test_chunk_to_sse() {
        let chunk = ChatCompletionChunk {
            id: "test".to_string(),
            object: "chat.completion.chunk".to_string(),
            created: 0,
            model: "test".to_string(),
            choices: vec![],
            usage: None,
            system_fingerprint: None,
        };
        let sse = chunk_to_sse(&chunk);
        assert!(sse.starts_with("data: "));
        assert!(sse.ends_with("\n\n"));
    }
}
