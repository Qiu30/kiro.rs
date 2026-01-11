import { Activity } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { useRequestLogs } from '@/hooks/use-credentials'
import { cn } from '@/lib/utils'

export function RequestLog() {
  const { data, isLoading } = useRequestLogs()

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('zh-CN', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: false,
    })
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="h-5 w-5" />
          实时请求日志
          {data && (
            <Badge variant="secondary" className="ml-auto">
              {data.total} 条记录
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div
          className="h-[400px] overflow-y-auto space-y-2 font-mono text-sm bg-muted/30 rounded-md p-4"
        >
          {isLoading && (
            <div className="text-center text-muted-foreground py-8">
              加载中...
            </div>
          )}

          {data && data.logs.length === 0 && (
            <div className="text-center text-muted-foreground py-8">
              暂无请求记录
            </div>
          )}

          {data?.logs.map((log) => (
            <div
              key={log.id}
              className={cn(
                "flex items-center gap-3 p-2 rounded border",
                "hover:bg-muted/50 transition-colors"
              )}
            >
              <span className="text-muted-foreground shrink-0">
                {formatTimestamp(log.timestamp)}
              </span>

              <Badge
                className={cn(
                  "shrink-0",
                  log.success
                    ? "bg-green-500 hover:bg-green-600 text-white"
                    : "bg-red-500 hover:bg-red-600 text-white"
                )}
              >
                {log.success ? "200" : "ERR"}
              </Badge>

              <Badge variant="outline" className="shrink-0">
                #{log.credentialId}
              </Badge>

              <span className="font-semibold shrink-0">
                {log.model}
              </span>

              <span className="text-muted-foreground">
                {log.maxTokens} tokens
              </span>

              {log.stream && (
                <Badge variant="secondary" className="shrink-0">
                  Stream
                </Badge>
              )}

              <span className="text-muted-foreground ml-auto shrink-0">
                {log.messageCount} msg
              </span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
