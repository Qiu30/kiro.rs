//! OpenAI 兼容 API 模块
//!
//! 提供 OpenAI Chat Completions API 兼容接口，
//! 将 OpenAI 格式请求转换为 Kiro API 格式。

mod converter;
mod handlers;
mod router;
mod stream;
mod types;

pub use router::create_router_with_provider;
