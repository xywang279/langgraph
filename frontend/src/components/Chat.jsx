import React, { useRef, useState } from 'react'
import { Card, Input, Button, Space, List, Tag, message as antdMessage, Spin } from 'antd'
import { SendOutlined, ThunderboltOutlined } from '@ant-design/icons'
import { API_BASE } from '../api'

function uid() { return Math.random().toString(36).slice(2) }

export default function Chat() {
  const [sessionId] = useState(() => `web-${uid()}`)
  const [text, setText] = useState('现在几点？请用工具获取')
  const [items, setItems] = useState([]) // {id, role, content}
  const [loading, setLoading] = useState(false)
  const esRef = useRef(null)

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

    // 先写入用户消息与一个空的 AI 占位
    setItems(prev => [
      ...prev,
      { id: uid(), role: 'user', content: text },
      { id: 'ai-live', role: 'ai', content: '' },
    ])
    setText('')

    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        if (data.event === 'end') {
          setLoading(false)
          es.close()
          esRef.current = null
          return
        }
        if (data.event === 'error') {
          setLoading(false)
          antdMessage.error(data.message || 'SSE 错误')
          es.close()
          esRef.current = null
          return
        }
        // 工具与节点事件
        if (data.role && data.role !== 'ai') {
          setItems(prev => [...prev, { id: uid(), role: data.role, content: `[${data.event}] ${String(data.content || '')}` }])
          return
        }
        // AI 增量
        if (data.role === 'ai' && typeof data.delta === 'string') {
          setItems(prev => prev.map(x => x.id === 'ai-live' ? { ...x, content: (x.content || '') + data.delta } : x))
        }
      } catch (e) {
        console.error('SSE parse error', e, ev.data)
      }
    }
    es.onerror = (e) => {
      console.error('EventSource error', e)
      setLoading(false)
      es.close()
      esRef.current = null
    }
  }

  const stopStream = () => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
      setLoading(false)
    }
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
          onChange={e => setText(e.target.value)}
          onPressEnter={(e) => {
            if (!e.shiftKey) { e.preventDefault(); startStream() }
          }}
          style={{ background: '#0f172a', color: '#e5e7eb' }}
        />
        <Button type="primary" icon={<SendOutlined />} onClick={startStream}>
          发送
        </Button>
        <Button icon={<ThunderboltOutlined />} onClick={stopStream} disabled={!loading}>
          停止
        </Button>
      </Space.Compact>

      <List
        dataSource={items}
        split={false}
        renderItem={(it) => (
          <List.Item style={{ padding: '8px 0' }}>
            <div style={{ width: '100%' }}>
              <Tag color={it.role === 'user' ? 'processing' : (it.role === 'ai' ? 'success' : 'warning')}>
                {it.role.toUpperCase()}
              </Tag>
              <div style={{
                whiteSpace: 'pre-wrap',
                color: it.role === 'user' ? '#e5e7eb' : (it.role === 'ai' ? '#d1fae5' : '#fde68a'),
                background: it.role === 'user' ? '#0f172a' : (it.role === 'ai' ? '#065f46' : '#3f3f46'),
                padding: 10,
                borderRadius: 10,
                border: '1px solid #1f2937',
              }}>
                {it.content || <Spin size="small" />}
              </div>
            </div>
          </List.Item>
        )}
      />
    </Card>
  )
}
