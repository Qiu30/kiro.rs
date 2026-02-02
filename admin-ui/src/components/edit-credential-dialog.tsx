import { useState, useEffect } from 'react'
import { toast } from 'sonner'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Checkbox } from '@/components/ui/checkbox'
import { useCredentialDetail, useUpdateCredential } from '@/hooks/use-credentials'
import { extractErrorMessage } from '@/lib/utils'
import { SUPPORTED_MODELS } from '@/types/api'

interface EditCredentialDialogProps {
  credentialId: number | null
  open: boolean
  onOpenChange: (open: boolean) => void
}

type AuthMethod = 'social' | 'idc'

export function EditCredentialDialog({ credentialId, open, onOpenChange }: EditCredentialDialogProps) {
  const [refreshToken, setRefreshToken] = useState('')
  const [authMethod, setAuthMethod] = useState<AuthMethod>('social')
  const [region, setRegion] = useState('')
  const [clientId, setClientId] = useState('')
  const [clientSecret, setClientSecret] = useState('')
  const [priority, setPriority] = useState('0')
  const [allowedModels, setAllowedModels] = useState<string[]>([])

  const { data: detail, isLoading } = useCredentialDetail(open ? credentialId : null)
  const { mutate, isPending } = useUpdateCredential()

  // 当详情加载完成时，填充表单
  useEffect(() => {
    if (detail) {
      setAuthMethod((detail.authMethod?.toLowerCase() as AuthMethod) || 'social')
      setRegion(detail.region || '')
      setPriority(String(detail.priority))
      setAllowedModels(detail.allowedModels || [])
      // 敏感字段不填充，保持为空
      setRefreshToken('')
      setClientId('')
      setClientSecret('')
    }
  }, [detail])

  const resetForm = () => {
    setRefreshToken('')
    setAuthMethod('social')
    setRegion('')
    setClientId('')
    setClientSecret('')
    setPriority('0')
    setAllowedModels([])
  }

  const handleModelToggle = (model: string) => {
    setAllowedModels((prev) =>
      prev.includes(model)
        ? prev.filter((m) => m !== model)
        : [...prev, model]
    )
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()

    if (credentialId === null) return

    // IdC/Builder-ID/IAM 需要额外字段（如果要更新的话）
    if (authMethod === 'idc') {
      // 如果原来没有 clientId/clientSecret，且用户没有输入，则报错
      if (!detail?.hasClientId && !clientId.trim()) {
        toast.error('IdC/Builder-ID/IAM 认证需要填写 Client ID')
        return
      }
      if (!detail?.hasClientSecret && !clientSecret.trim()) {
        toast.error('IdC/Builder-ID/IAM 认证需要填写 Client Secret')
        return
      }
    }

    // 构建更新请求，只包含有变化的字段
    const updateData: Record<string, unknown> = {}

    // 敏感字段：只有用户输入了才更新
    if (refreshToken.trim()) {
      updateData.refreshToken = refreshToken.trim()
    }
    if (clientId.trim()) {
      updateData.clientId = clientId.trim()
    }
    if (clientSecret.trim()) {
      updateData.clientSecret = clientSecret.trim()
    }

    // 非敏感字段：总是更新
    updateData.authMethod = authMethod
    updateData.region = region.trim() || undefined
    updateData.priority = parseInt(priority) || 0
    updateData.allowedModels = allowedModels.length > 0 ? allowedModels : []

    mutate(
      {
        id: credentialId,
        data: updateData,
      },
      {
        onSuccess: (data) => {
          toast.success(data.message)
          onOpenChange(false)
          resetForm()
        },
        onError: (error: unknown) => {
          toast.error(`更新失败: ${extractErrorMessage(error)}`)
        },
      }
    )
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>编辑凭据 #{credentialId}</DialogTitle>
        </DialogHeader>

        {isLoading ? (
          <div className="py-8 text-center text-muted-foreground">加载中...</div>
        ) : (
          <form onSubmit={handleSubmit}>
            <div className="space-y-4 py-4">
              {/* Refresh Token */}
              <div className="space-y-2">
                <label htmlFor="refreshToken" className="text-sm font-medium">
                  Refresh Token
                  {detail?.hasRefreshToken && (
                    <span className="text-muted-foreground ml-2">(已配置，留空保持不变)</span>
                  )}
                </label>
                <Input
                  id="refreshToken"
                  type="password"
                  placeholder={detail?.hasRefreshToken ? '留空保持原值' : '请输入 Refresh Token'}
                  value={refreshToken}
                  onChange={(e) => setRefreshToken(e.target.value)}
                  disabled={isPending}
                />
              </div>

              {/* 认证方式 */}
              <div className="space-y-2">
                <label htmlFor="authMethod" className="text-sm font-medium">
                  认证方式
                </label>
                <select
                  id="authMethod"
                  value={authMethod}
                  onChange={(e) => setAuthMethod(e.target.value as AuthMethod)}
                  disabled={isPending}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <option value="social">Social</option>
                  <option value="idc">IdC/Builder-ID/IAM</option>
                </select>
              </div>

              <div className="space-y-2">
                <label htmlFor="region" className="text-sm font-medium">
                  刷新 Token 地域
                </label>
                <Input
                  id="region"
                  placeholder="例如 us-east-1（留空则使用全局 region）"
                  value={region}
                  onChange={(e) => setRegion(e.target.value)}
                  disabled={isPending}
                />
              </div>

              {/* IdC/Builder-ID/IAM 额外字段 */}
              {authMethod === 'idc' && (
                <>
                  <div className="space-y-2">
                    <label htmlFor="clientId" className="text-sm font-medium">
                      Client ID
                      {detail?.hasClientId && (
                        <span className="text-muted-foreground ml-2">(已配置，留空保持不变)</span>
                      )}
                    </label>
                    <Input
                      id="clientId"
                      placeholder={detail?.hasClientId ? '留空保持原值' : '请输入 Client ID'}
                      value={clientId}
                      onChange={(e) => setClientId(e.target.value)}
                      disabled={isPending}
                    />
                  </div>
                  <div className="space-y-2">
                    <label htmlFor="clientSecret" className="text-sm font-medium">
                      Client Secret
                      {detail?.hasClientSecret && (
                        <span className="text-muted-foreground ml-2">(已配置，留空保持不变)</span>
                      )}
                    </label>
                    <Input
                      id="clientSecret"
                      type="password"
                      placeholder={detail?.hasClientSecret ? '留空保持原值' : '请输入 Client Secret'}
                      value={clientSecret}
                      onChange={(e) => setClientSecret(e.target.value)}
                      disabled={isPending}
                    />
                  </div>
                </>
              )}

              {/* 优先级 */}
              <div className="space-y-2">
                <label htmlFor="priority" className="text-sm font-medium">
                  优先级
                </label>
                <Input
                  id="priority"
                  type="number"
                  min="0"
                  placeholder="数字越小优先级越高"
                  value={priority}
                  onChange={(e) => setPriority(e.target.value)}
                  disabled={isPending}
                />
                <p className="text-xs text-muted-foreground">
                  数字越小优先级越高，默认为 0
                </p>
              </div>

              {/* 允许的模型 */}
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  允许的模型
                </label>
                <div className="space-y-2">
                  {SUPPORTED_MODELS.map((model) => (
                    <div key={model.value} className="flex items-center space-x-2">
                      <Checkbox
                        id={`edit-model-${model.value}`}
                        checked={allowedModels.includes(model.value)}
                        onCheckedChange={() => handleModelToggle(model.value)}
                        disabled={isPending}
                      />
                      <label
                        htmlFor={`edit-model-${model.value}`}
                        className="text-sm cursor-pointer"
                      >
                        {model.label}
                      </label>
                    </div>
                  ))}
                </div>
                <p className="text-xs text-muted-foreground">
                  不选择任何模型表示支持所有模型
                </p>
              </div>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => onOpenChange(false)}
                disabled={isPending}
              >
                取消
              </Button>
              <Button type="submit" disabled={isPending}>
                {isPending ? '保存中...' : '保存'}
              </Button>
            </DialogFooter>
          </form>
        )}
      </DialogContent>
    </Dialog>
  )
}
