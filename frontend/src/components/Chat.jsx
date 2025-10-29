import React, { useRef, useState } from 'react'
import {
  Card,
  Input,
  Button,
  Space,
  List,
  Tag,
  message as antdMessage,
  Spin,
  Typography,
  Tooltip,
} from 'antd'
import {
  SendOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons'
import { API_BASE } from '../api'

const { Text, Paragraph } = Typography

function uid() {
  return Math.random().toString(36).slice(2)
}

const ROLE_STYLES = {
  user: {
    tag: { color: 'processing', label: 'USER' },
    bubble: {
      background: '#1d4ed8',
      color: '#e0f2fe',
      alignSelf: 'flex-end',
    },
  },
  ai: {
    tag: { color: 'success', label: 'ASSISTANT' },
    bubble: {
      background: '#065f46',
      color: '#d1fae5',
      alignSelf: 'flex-start',
    },
  },
  tool: {
    tag: { color: 'geekblue', label: 'TOOL' },
    bubble: {
      background: '#111827',
      color: '#facc15',
      alignSelf: 'stretch',
    },
  },
}

const STATUS_COLORS = {
  ok: 'green',
  degraded: 'orange',
  error: 'red',
  unknown: 'default',
}

function formatLatency(latency) {
  if (latency === undefined || latency === null || Number.isNaN(latency)) {
    return null
  }
  const value = Number(latency)
  if (Number.isNaN(value)) return null
  return `${value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')}s`
}

function stringifyFallback(fallback) {
  if (!fallback || typeof fallback !== 'object') {
    return ''
  }
  const tool = fallback.tool || 'fallback'
  const status = (fallback.status || 'unknown').toUpperCase()
  const obs = fallback.observation ? ` · ${fallback.observation}` : ''
  return `${tool} → ${status}${obs}`
}

export default function Chat() {
  const [sessionId] = useState(() => `web-${uid()}`)
  const [text, setText] = useState('现在几点？请用工具获取')
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)
  const esRef = useRef(null)

  const appendItem = (item) => {
    setItems((prev) => [...prev, { ...item, ts: Date.now() }])
  }

  const startStream = () => {
    if (!text.trim()) {
      antdMessage.warning('请输入问题')
      return
    }
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }

    setLoading(true)
    const url = `${API_BASE}/chat/stream?session_id=${encodeURIComponent(sessionId)}&message=${encodeURIComponent(text)}`
    const es = new EventSource(url)
    esRef.current = es

    const userMessage = { id: uid(), role: 'user', content: text }
    const aiPlaceholder = { id: 'ai-live', role: 'ai', content: '' }

    setItems((prev) => {
      const withoutLive = prev.filter((item) => item.id !== 'ai-live')
      return [...withoutLive, userMessage, aiPlaceholder]
    })
    setText('')

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)

        if (data.event === 'end') {
          setLoading(false)
          es.close()
          esRef.current = null
          setItems((prev) =>
            prev.flatMap((item) => {
              if (item.id !== 'ai-live') return [item]
              const textContent = (item.content || '').trim()
              if (!textContent) {
                return []
              }
              return [{ ...item, id: uid(), content: textContent }]
            })
          )
          return
        }

        if (data.event === 'error') {
          setLoading(false)
          antdMessage.error(data.message || 'SSE 通道错误')
          es.close()
          esRef.current = null
          return
        }

        if (data.role === 'ai' && typeof data.delta === 'string') {
          setItems((prev) =>
            prev.map((item) =>
              item.id === 'ai-live'
                ? { ...item, content: `${item.content || ''}${data.delta}` }
                : item
            )
          )
          return
        }

        if (data.role === 'tool') {
          appendItem({
            id: uid(),
            role: 'tool',
            node: data.event,
            tool: data.tool || data.event || 'tool',
            status: data.status || 'unknown',
            latency: typeof data.latency === 'number' ? data.latency : Number(data.latency),
            tries: data.tries,
            observation: data.content ?? data.observation ?? '',
            error: data.error,
            fallback: data.fallback,
          })
          return
        }

        if (data.role && data.role !== 'ai') {
          appendItem({
            id: uid(),
            role: data.role,
            node: data.event,
            content: data.content || '',
          })
        }
      } catch (err) {
        console.error('SSE parse error', err, ev.data)
      }
    }

    es.onerror = (err) => {
      console.error('EventSource error', err)
      setLoading(false)
      es.close()
      esRef.current = null
      setItems((prev) => prev.filter((item) => item.id !== 'ai-live'))
    }
  }

  const stopStream = () => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
      setLoading(false)
      setItems((prev) => prev.filter((item) => item.id !== 'ai-live'))
    }
  }

  const renderToolItem = (item) => {
    const status = (item.status || 'unknown').toLowerCase()
    const statusColor = STATUS_COLORS[status] || 'default'
    const latency = formatLatency(item.latency)
    const fallbackText = stringifyFallback(item.fallback)

    return (
      <div style={{ width: '100%' }}>
        <Space size={6} wrap align="center" style={{ marginBottom: 6 }}>
          <Tag color={ROLE_STYLES.tool.tag.color}>{ROLE_STYLES.tool.tag.label}</Tag>
          {item.node && <Tag color="purple">{item.node}</Tag>}
          <Tag color="blue">{item.tool}</Tag>
          <Tag color={statusColor}>{status.toUpperCase()}</Tag>
          {typeof item.tries === 'number' && item.tries > 1 && (
            <Tag color="magenta">尝试 {item.tries}</Tag>
          )}
          {latency && (
            <Tag icon={<ClockCircleOutlined />} color="geekblue">
              {latency}
            </Tag>
          )}
        </Space>
        <div
          style={{
            background: ROLE_STYLES.tool.bubble.background,
            color: ROLE_STYLES.tool.bubble.color,
            borderRadius: 10,
            border: '1px solid #1f2937',
            padding: 12,
            whiteSpace: 'pre-wrap',
          }}
        >
          <Paragraph style={{ marginBottom: 0, color: ROLE_STYLES.tool.bubble.color }}>
            {item.observation || '（工具未返回内容）'}
          </Paragraph>
          {item.error && (
            <Text type="danger" style={{ display: 'block', marginTop: 6 }}>
              错误：{item.error}
            </Text>
          )}
          {fallbackText && (
            <Text type="secondary" style={{ display: 'block', marginTop: 6 }}>
              兜底：{fallbackText}
            </Text>
          )}
        </div>
      </div>
    )
  }

  const renderDefaultItem = (item) => {
    const roleStyle = ROLE_STYLES[item.role] || ROLE_STYLES.ai
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: roleStyle.bubble.alignSelf === 'flex-end' ? 'flex-end' : 'flex-start', width: '100%' }}>
        <Tag color={roleStyle.tag.color}>{roleStyle.tag.label}</Tag>
        <div
          style={{
            background: roleStyle.bubble.background,
            color: roleStyle.bubble.color,
            borderRadius: 10,
            border: '1px solid #1f2937',
            padding: 12,
            minWidth: 120,
            maxWidth: '100%',
            whiteSpace: 'pre-wrap',
          }}
        >
          {item.content ? <Text style={{ color: roleStyle.bubble.color }}>{item.content}</Text> : <Spin size="small" />}
        </div>
      </div>
    )
  }

  const renderItem = (item) => {
    if (item.role === 'tool') {
      return renderToolItem(item)
    }
    return renderDefaultItem(item)
  }

  return (
    <Card
      style={{ background: '#0b1220', borderRadius: 16, border: '1px solid #1f2937' }}
      bodyStyle={{ padding: 16 }}
    >
      <Space.Compact style={{ width: '100%', marginBottom: 12 }}>
        <Input.TextArea
          autoSize={{ minRows: 1, maxRows: 4 }}
          placeholder="问点什么，比如：现在几点？请用工具获取"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onPressEnter={(e) => {
            if (!e.shiftKey) {
              e.preventDefault()
              startStream()
            }
          }}
          style={{ background: '#0f172a', color: '#e5e7eb' }}
        />
        <Tooltip title="发送">
          <Button type="primary" icon={<SendOutlined />} onClick={startStream}>
            发送
          </Button>
        </Tooltip>
        <Tooltip title="终止当前请求">
          <Button icon={<ThunderboltOutlined />} onClick={stopStream} disabled={!loading}>
            停止
          </Button>
        </Tooltip>
      </Space.Compact>

      <List
        dataSource={items}
        rowKey={(item) => item.id}
        split={false}
        renderItem={(item) => (
          <List.Item style={{ padding: '8px 0', display: 'flex', justifyContent: item.role === 'user' ? 'flex-end' : 'flex-start' }}>
            {renderItem(item)}
          </List.Item>
        )}
      />
    </Card>
  )
}
