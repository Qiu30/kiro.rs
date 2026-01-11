//! OpenAI API 路由配置

use axum::{
    Router, middleware,
    routing::post,
};

use std::sync::Arc;

use crate::common::auth;
use crate::kiro::provider::KiroProvider;
use crate::request_log::RequestLogger;

use super::handlers::{AppState, chat_completions};
use super::types::ErrorResponse;

use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Json, Response},
};

/// API Key 认证中间件
async fn auth_middleware(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    match auth::extract_api_key(&request) {
        Some(key) if auth::constant_time_eq(&key, &state.api_key) => next.run(request).await,
        _ => {
            let error = ErrorResponse::authentication_error();
            (StatusCode::UNAUTHORIZED, Json(error)).into_response()
        }
    }
}

/// CORS 中间件层
fn cors_layer() -> tower_http::cors::CorsLayer {
    use tower_http::cors::{Any, CorsLayer};

    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
}

/// 创建带有 KiroProvider 的 OpenAI API 路由
///
/// # 端点
/// - `POST /v1/chat/completions` - OpenAI 兼容的聊天完成端点
///
/// # 认证
/// 所有 `/v1` 路径需要 API Key 认证，支持：
/// - `Authorization: Bearer <token>` header
///
/// # 参数
/// - `api_key`: API 密钥，用于验证客户端请求
/// - `kiro_provider`: 可选的 KiroProvider，用于调用上游 API
pub fn create_router_with_provider(
    api_key: impl Into<String>,
    kiro_provider: Option<KiroProvider>,
    profile_arn: Option<String>,
    request_logger: Option<Arc<RequestLogger>>,
) -> Router {
    let mut state = AppState::new(api_key);
    if let Some(provider) = kiro_provider {
        state = state.with_kiro_provider(provider);
    }
    if let Some(arn) = profile_arn {
        state = state.with_profile_arn(arn);
    }
    if let Some(logger) = request_logger {
        state = state.with_request_logger(logger);
    }

    // 需要认证的 /v1 路由
    let v1_routes = Router::new()
        .route("/chat/completions", post(chat_completions))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ));

    Router::new()
        .nest("/v1", v1_routes)
        .layer(cors_layer())
        .with_state(state)
}
