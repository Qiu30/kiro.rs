//! 请求日志模块
//!
//! 提供内存中的请求日志记录功能，用于 Admin UI 实时显示

use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::Mutex;
use serde::Serialize;

/// 最大日志条目数
const MAX_LOG_ENTRIES: usize = 50;

/// 单个请求日志条目
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestLogEntry {
    /// 唯一请求 ID
    pub id: String,
    /// 时间戳 (RFC3339)
    pub timestamp: String,
    /// 模型名称
    pub model: String,
    /// 请求的最大 token 数
    pub max_tokens: i32,
    /// 是否启用流式传输
    pub stream: bool,
    /// 消息数量
    pub message_count: usize,
    /// 使用的凭据 ID
    pub credential_id: u64,
    /// 请求是否成功
    pub success: bool,
}

/// 线程安全的请求日志记录器
pub struct RequestLogger {
    logs: Arc<Mutex<VecDeque<RequestLogEntry>>>,
}

impl RequestLogger {
    /// 创建新的请求日志记录器
    pub fn new() -> Self {
        Self {
            logs: Arc::new(Mutex::new(VecDeque::with_capacity(MAX_LOG_ENTRIES))),
        }
    }

    /// 记录一个新的请求
    pub fn log_request(&self, entry: RequestLogEntry) {
        let mut logs = self.logs.lock();

        // 如果达到最大容量，移除最旧的条目
        if logs.len() >= MAX_LOG_ENTRIES {
            logs.pop_front();
        }

        logs.push_back(entry);
    }

    /// 获取所有日志条目（最新的在前）
    pub fn get_logs(&self) -> Vec<RequestLogEntry> {
        let logs = self.logs.lock();
        logs.iter().rev().cloned().collect()
    }
}

impl Default for RequestLogger {
    fn default() -> Self {
        Self::new()
    }
}
