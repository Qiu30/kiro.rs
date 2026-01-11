//! OpenAI API Handler 函数

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    Json as JsonExtractor,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Json, Response},
};
use bytes::Bytes;
use futures::{Stream, StreamExt, stream};
use tokio::time::interval;
use uuid::Uuid;

use crate::kiro::model::events::Event;
use crate::kiro::model::requests::kiro::KiroRequest;
use crate::kiro::parser::decoder::EventStreamDecoder;
use crate::kiro::provider::KiroProvider;
use crate::request_log::RequestLogger;

use super::converter::{ConversionError, convert_request};
use super::stream::{StreamContext, chunk_to_sse, done_sse};
use super::types::{
    ChatCompletionRequest, ChatCompletionResponse, Choice, ErrorResponse, ResponseMessage,
    ToolCall, FunctionCall, Usage,
};

/// 应用状态
#[derive(Clone)]
pub struct AppState {
    pub api_key: String,
    pub kiro_provider: Option<Arc<KiroProvider>>,
    pub profile_arn: Option<String>,
    pub request_logger: Option<Arc<RequestLogger>>,
}

impl AppState {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            kiro_provider: None,
            profile_arn: None,
            request_logger: None,
        }
    }

    pub fn with_kiro_provider(mut self, provider: KiroProvider) -> Self {
        self.kiro_provider = Some(Arc::new(provider));
        self
    }

    pub fn with_profile_arn(mut self, arn: impl Into<String>) -> Self {
        self.profile_arn = Some(arn.into());
        self
    }

    pub fn with_request_logger(mut self, logger: Arc<RequestLogger>) -> Self {
        self.request_logger = Some(logger);
        self
    }
}

/// POST /v1/chat/completions
///
/// OpenAI 兼容的聊天完成端点
pub async fn chat_completions(
    State(state): State<AppState>,
    JsonExtractor(payload): JsonExtractor<ChatCompletionRequest>,
) -> Response {
    tracing::info!(
        model = %payload.model,
        max_tokens = ?payload.effective_max_tokens(),
        stream = %payload.is_stream(),
        message_count = %payload.messages.len(),
        "Received POST /v1/chat/completions request"
    );

    // 检查 KiroProvider 是否可用
    let provider = match &state.kiro_provider {
        Some(p) => p.clone(),
        None => {
            tracing::error!("KiroProvider 未配置");
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse::new(
                    "server_error",
                    "Kiro API provider not configured",
                )),
            )
                .into_response();
        }
    };

    // 获取凭据上下文以记录使用的凭据 ID
    let credential_id = match provider.token_manager().acquire_context().await {
        Ok(ctx) => ctx.id,
        Err(e) => {
            tracing::error!("获取凭据上下文失败: {}", e);
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse::new(
                    "server_error",
                    "No available credentials",
                )),
            )
                .into_response();
        }
    };

    // 记录请求日志
    if let Some(logger) = &state.request_logger {
        logger.log_request(crate::request_log::RequestLogEntry {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            model: payload.model.clone(),
            max_tokens: payload.effective_max_tokens(),
            stream: payload.is_stream(),
            message_count: payload.messages.len(),
            credential_id,
            success: true,
        });
    }

    // 转换请求
    let conversion_result = match convert_request(&payload) {
        Ok(result) => result,
        Err(e) => {
            let message = match &e {
                ConversionError::UnsupportedModel(model) => {
                    format!("模型不支持: {}", model)
                }
                ConversionError::EmptyMessages => "消息列表为空".to_string(),
                ConversionError::InvalidImageUrl(url) => {
                    format!("无效的图片 URL: {}", url)
                }
            };
            tracing::warn!("请求转换失败: {}", e);
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorResponse::new("invalid_request_error", message)),
            )
                .into_response();
        }
    };

    // 构建 Kiro 请求
    let kiro_request = KiroRequest {
        conversation_state: conversion_result.conversation_state,
        profile_arn: state.profile_arn.clone(),
    };

    let request_body = match serde_json::to_string(&kiro_request) {
        Ok(body) => body,
        Err(e) => {
            tracing::error!("序列化请求失败: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(
                    "server_error",
                    format!("序列化请求失败: {}", e),
                )),
            )
                .into_response();
        }
    };

    tracing::debug!("Kiro request body: {}", request_body);

    // 估算输入 tokens（简化版本）
    let input_tokens = estimate_input_tokens(&payload);

    if payload.is_stream() {
        // 流式响应
        handle_stream_request(
            provider,
            &request_body,
            &conversion_result.original_model,
            input_tokens,
            payload.include_usage_in_stream(),
        )
        .await
    } else {
        // 非流式响应
        handle_non_stream_request(
            provider,
            &request_body,
            &conversion_result.original_model,
            input_tokens,
        )
        .await
    }
}

/// 估算输入 tokens
fn estimate_input_tokens(payload: &ChatCompletionRequest) -> i32 {
    let mut total = 0;

    for msg in &payload.messages {
        if let Some(content) = &msg.content {
            let text = match content {
                super::types::MessageContent::Text(s) => s.clone(),
                super::types::MessageContent::Parts(parts) => {
                    parts
                        .iter()
                        .filter_map(|p| {
                            if let super::types::ContentPart::Text { text } = p {
                                Some(text.clone())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ")
                }
            };
            // 简单估算：中文约 1.5 字符/token，英文约 4 字符/token
            let chars: Vec<char> = text.chars().collect();
            let chinese = chars
                .iter()
                .filter(|c| **c >= '\u{4E00}' && **c <= '\u{9FFF}')
                .count();
            let other = chars.len() - chinese;
            total += ((chinese * 2 + 2) / 3 + (other + 3) / 4) as i32;
        }
    }

    total.max(1)
}

/// 处理流式请求
async fn handle_stream_request(
    provider: Arc<KiroProvider>,
    request_body: &str,
    model: &str,
    input_tokens: i32,
    include_usage: bool,
) -> Response {
    // 调用 Kiro API
    let response = match provider.call_api_stream(request_body).await {
        Ok(resp) => resp,
        Err(e) => {
            tracing::error!("Kiro API 调用失败: {}", e);
            return (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse::new(
                    "server_error",
                    format!("上游 API 调用失败: {}", e),
                )),
            )
                .into_response();
        }
    };

    // 创建流处理上下文
    let mut ctx = StreamContext::new(model, input_tokens, include_usage);

    // 生成初始 chunk
    let initial_chunk = ctx.generate_initial_chunk();

    // 创建 SSE 流
    let stream = create_sse_stream(response, ctx, initial_chunk);

    // 返回 SSE 响应
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(Body::from_stream(stream))
        .unwrap()
}

/// Ping 事件间隔（25秒）
const PING_INTERVAL_SECS: u64 = 25;

/// 创建 ping 事件的 SSE 字符串
fn create_ping_sse() -> Bytes {
    Bytes::from(": ping\n\n")
}

/// 创建 SSE 事件流
fn create_sse_stream(
    response: reqwest::Response,
    ctx: StreamContext,
    initial_chunk: super::types::ChatCompletionChunk,
) -> impl Stream<Item = Result<Bytes, Infallible>> {
    // 先发送初始 chunk
    let initial_stream = stream::iter(vec![Ok(Bytes::from(chunk_to_sse(&initial_chunk)))]);

    // 然后处理 Kiro 响应流
    let body_stream = response.bytes_stream();

    let processing_stream = stream::unfold(
        (
            body_stream,
            ctx,
            EventStreamDecoder::new(),
            false,
            interval(Duration::from_secs(PING_INTERVAL_SECS)),
        ),
        |(mut body_stream, mut ctx, mut decoder, finished, mut ping_interval)| async move {
            if finished {
                return None;
            }

            tokio::select! {
                chunk_result = body_stream.next() => {
                    match chunk_result {
                        Some(Ok(chunk)) => {
                            if let Err(e) = decoder.feed(&chunk) {
                                tracing::warn!("缓冲区溢出: {}", e);
                            }

                            let mut sse_data = Vec::new();
                            for result in decoder.decode_iter() {
                                match result {
                                    Ok(frame) => {
                                        if let Ok(event) = Event::from_frame(frame) {
                                            let chunks = ctx.process_kiro_event(&event);
                                            for chunk in chunks {
                                                sse_data.push(Ok(Bytes::from(chunk_to_sse(&chunk))));
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!("解码事件失败: {}", e);
                                    }
                                }
                            }

                            Some((stream::iter(sse_data), (body_stream, ctx, decoder, false, ping_interval)))
                        }
                        Some(Err(e)) => {
                            tracing::error!("读取响应流失败: {}", e);
                            // 发送最终事件并结束
                            let final_chunks = ctx.generate_final_chunk();
                            let mut sse_data: Vec<Result<Bytes, Infallible>> = final_chunks
                                .into_iter()
                                .map(|c| Ok(Bytes::from(chunk_to_sse(&c))))
                                .collect();
                            sse_data.push(Ok(Bytes::from(done_sse())));
                            Some((stream::iter(sse_data), (body_stream, ctx, decoder, true, ping_interval)))
                        }
                        None => {
                            // 流结束，发送最终事件
                            let final_chunks = ctx.generate_final_chunk();
                            let mut sse_data: Vec<Result<Bytes, Infallible>> = final_chunks
                                .into_iter()
                                .map(|c| Ok(Bytes::from(chunk_to_sse(&c))))
                                .collect();
                            sse_data.push(Ok(Bytes::from(done_sse())));
                            Some((stream::iter(sse_data), (body_stream, ctx, decoder, true, ping_interval)))
                        }
                    }
                }
                _ = ping_interval.tick() => {
                    tracing::trace!("发送 ping 保活事件");
                    let sse_data: Vec<Result<Bytes, Infallible>> = vec![Ok(create_ping_sse())];
                    Some((stream::iter(sse_data), (body_stream, ctx, decoder, false, ping_interval)))
                }
            }
        },
    )
    .flatten();

    initial_stream.chain(processing_stream)
}

/// 处理非流式请求
async fn handle_non_stream_request(
    provider: Arc<KiroProvider>,
    request_body: &str,
    model: &str,
    input_tokens: i32,
) -> Response {
    // 调用 Kiro API
    let response = match provider.call_api(request_body).await {
        Ok(resp) => resp,
        Err(e) => {
            tracing::error!("Kiro API 调用失败: {}", e);
            return (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse::new(
                    "server_error",
                    format!("上游 API 调用失败: {}", e),
                )),
            )
                .into_response();
        }
    };

    // 读取响应体
    let body_bytes = match response.bytes().await {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::error!("读取响应体失败: {}", e);
            return (
                StatusCode::BAD_GATEWAY,
                Json(ErrorResponse::new(
                    "server_error",
                    format!("读取响应失败: {}", e),
                )),
            )
                .into_response();
        }
    };

    // 解析事件流
    let mut decoder = EventStreamDecoder::new();
    if let Err(e) = decoder.feed(&body_bytes) {
        tracing::warn!("缓冲区溢出: {}", e);
    }

    let mut text_content = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut finish_reason = "stop".to_string();
    let mut context_input_tokens: Option<i32> = None;
    let mut output_tokens = 0;

    // 收集工具调用的增量 JSON
    let mut tool_json_buffers: std::collections::HashMap<String, (String, String)> =
        std::collections::HashMap::new();

    for result in decoder.decode_iter() {
        match result {
            Ok(frame) => {
                if let Ok(event) = Event::from_frame(frame) {
                    match event {
                        Event::AssistantResponse(resp) => {
                            // 过滤 thinking 标签
                            let filtered = filter_thinking_tags(&resp.content);
                            text_content.push_str(&filtered);
                            output_tokens += estimate_output_tokens(&resp.content);
                        }
                        Event::ToolUse(tool_use) => {
                            finish_reason = "tool_calls".to_string();

                            // 累积工具的 JSON 输入
                            let entry = tool_json_buffers
                                .entry(tool_use.tool_use_id.clone())
                                .or_insert_with(|| (tool_use.name.clone(), String::new()));
                            entry.1.push_str(&tool_use.input);

                            // 如果是完整的工具调用，添加到列表
                            if tool_use.stop {
                                tool_calls.push(ToolCall {
                                    id: tool_use.tool_use_id.clone(),
                                    call_type: "function".to_string(),
                                    function: FunctionCall {
                                        name: entry.0.clone(),
                                        arguments: entry.1.clone(),
                                    },
                                });
                            }

                            output_tokens += (tool_use.input.len() as i32 + 3) / 4;
                        }
                        Event::ContextUsage(context_usage) => {
                            let actual_input_tokens = (context_usage.context_usage_percentage
                                * 200_000.0
                                / 100.0) as i32;
                            context_input_tokens = Some(actual_input_tokens);
                        }
                        Event::Exception { exception_type, .. } => {
                            if exception_type == "ContentLengthExceededException" {
                                finish_reason = "length".to_string();
                            }
                        }
                        _ => {}
                    }
                }
            }
            Err(e) => {
                tracing::warn!("解码事件失败: {}", e);
            }
        }
    }

    // 使用从 contextUsageEvent 计算的 input_tokens
    let final_input_tokens = context_input_tokens.unwrap_or(input_tokens);

    // 构建响应
    let response_body = ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4().to_string().replace('-', "")),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_string(),
                content: if text_content.is_empty() {
                    None
                } else {
                    Some(text_content)
                },
                tool_calls: if tool_calls.is_empty() {
                    None
                } else {
                    Some(tool_calls)
                },
            },
            finish_reason: Some(finish_reason),
        }],
        usage: Some(Usage {
            prompt_tokens: final_input_tokens,
            completion_tokens: output_tokens,
            total_tokens: final_input_tokens + output_tokens,
        }),
        system_fingerprint: None,
    };

    (StatusCode::OK, Json(response_body)).into_response()
}

/// 过滤 thinking 标签
fn filter_thinking_tags(content: &str) -> String {
    let mut result = content.to_string();

    while let Some(start) = result.find("<thinking>") {
        if let Some(end) = result[start..].find("</thinking>") {
            let end_pos = start + end + "</thinking>".len();
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
            result = result[..start].to_string();
            break;
        }
    }

    result
}

/// 估算输出 tokens
fn estimate_output_tokens(text: &str) -> i32 {
    let chars: Vec<char> = text.chars().collect();
    let chinese = chars
        .iter()
        .filter(|c| **c >= '\u{4E00}' && **c <= '\u{9FFF}')
        .count();
    let other = chars.len() - chinese;
    ((chinese * 2 + 2) / 3 + (other + 3) / 4).max(1) as i32
}
