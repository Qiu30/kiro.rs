// 凭据状态响应
export interface CredentialsStatusResponse {
  total: number
  available: number
  currentId: number
  credentials: CredentialStatusItem[]
}

// 单个凭据状态
export interface CredentialStatusItem {
  id: number
  priority: number
  disabled: boolean
  failureCount: number
  isCurrent: boolean
  expiresAt: string | null
  authMethod: string | null
  hasProfileArn: boolean
  allowedModels?: string[]
}

// 余额响应
export interface BalanceResponse {
  id: number
  subscriptionTitle: string | null
  currentUsage: number
  usageLimit: number
  remaining: number
  usagePercentage: number
  nextResetAt: number | null
}

// 成功响应
export interface SuccessResponse {
  success: boolean
  message: string
}

// 错误响应
export interface AdminErrorResponse {
  error: {
    type: string
    message: string
  }
}

// 请求类型
export interface SetDisabledRequest {
  disabled: boolean
}

export interface SetPriorityRequest {
  priority: number
}

// 添加凭据请求
export interface AddCredentialRequest {
  refreshToken: string
  authMethod?: 'social' | 'idc'
  clientId?: string
  clientSecret?: string
  priority?: number
  region?: string
  allowedModels?: string[]
}

// 添加凭据响应
export interface AddCredentialResponse {
  success: boolean
  message: string
  credentialId: number
}

// 更新凭据请求
export interface UpdateCredentialRequest {
  refreshToken?: string
  authMethod?: 'social' | 'idc'
  clientId?: string
  clientSecret?: string
  region?: string
  priority?: number
  allowedModels?: string[]
}

// 凭据详情响应
export interface CredentialDetailResponse {
  id: number
  priority: number
  authMethod: string | null
  region: string | null
  hasRefreshToken: boolean
  hasClientId: boolean
  hasClientSecret: boolean
  allowedModels: string[]
}

// 支持的模型列表
export const SUPPORTED_MODELS = [
  { value: 'sonnet', label: 'Claude Sonnet 4.5' },
  { value: 'opus', label: 'Claude Opus 4.5' },
  { value: 'haiku', label: 'Claude Haiku 4.5' },
] as const

// 请求日志条目
export interface RequestLogEntry {
  id: string
  timestamp: string
  model: string
  maxTokens: number
  stream: boolean
  messageCount: number
  credentialId: number
  success: boolean
}

// 请求日志响应
export interface RequestLogsResponse {
  total: number
  logs: RequestLogEntry[]
}
