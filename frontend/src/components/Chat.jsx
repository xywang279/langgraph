import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Card,
  Input,
  Button,
  Flex,
  List,
  Tag,
  message as antdMessage,
  Spin,
  Typography,
  Tooltip,
  Steps,
  Modal,
  Checkbox,
  Alert,
  Table,
  theme as antdTheme,
  Divider,
  Space,
} from 'antd'
import {
  SendOutlined,
  StopOutlined,
  ReloadOutlined,
  CopyOutlined,
  RedoOutlined,
  DeleteOutlined,
  FileTextOutlined,
  ClockCircleOutlined,
  MessageOutlined,
  GlobalOutlined,
  SearchOutlined,
  FileTextTwoTone,
  PlusOutlined,
  EditOutlined,
} from '@ant-design/icons'
import { API_BASE, API_KEY, createThread, fetchThreadMessages, fetchThreads, deleteThread, renameThread } from '../api'

const { Text, Paragraph } = Typography
const { TextArea } = Input

const uid = () => Math.random().toString(36).slice(2)
const INITIAL_PLAN = { steps: [], currentStep: 0, activeStepId: null }

const ROLE_CONFIG = {
  user: {
    tag: { color: 'processing', label: 'USER' },
    bubble: { background: '#e0f2ff', color: '#0f172a', alignSelf: 'flex-end', border: '1px solid #bfdbfe', textAlign: 'right' },
  },
  ai: {
    tag: { color: 'success', label: 'ASSISTANT' },
    bubble: { background: '#dcfce7', color: '#065f46', alignSelf: 'flex-start', border: '1px solid #bbf7d0', textAlign: 'left' },
  },
  tool: {
    tag: { color: 'geekblue', label: 'TOOL' },
    bubble: { background: '#fff9db', color: '#854d0e', alignSelf: 'flex-start', border: '1px solid #fde68a', textAlign: 'left' },
  },
  system: {
    tag: { color: 'purple', label: 'SYSTEM' },
    bubble: { background: '#ede9fe', color: '#3730a3', alignSelf: 'flex-start', border: '1px solid #ddd6fe', textAlign: 'left' },
  },
}

const STEP_STATUS_TEXT = {
  pending: 'Pending',
  waiting: 'Waiting',
  in_progress: 'In progress',
  completed: 'Completed',
  cancelled: 'Cancelled',
  failed: 'Failed',
}

const STEP_STATUS_MAP = {
  pending: 'wait',
  waiting: 'process',
  in_progress: 'process',
  completed: 'finish',
  cancelled: 'error',
  failed: 'error',
}

const heroPrompts = [
  { title: 'Plan a trip', desc: 'Complex planning with steps', prompt: 'Plan a 3-day trip to Tokyo with budget and itinerary', icon: <GlobalOutlined /> },
  { title: 'Web Research', desc: 'Live tool usage & citations', prompt: 'Find the latest LangGraph community updates', icon: <SearchOutlined /> },
  { title: 'Human-in-the-loop', desc: 'Interrupt & approval workflow', prompt: 'Prepare a rollout plan and ask me to approve before execution', icon: <MessageOutlined /> },
  { title: 'Document Q&A', desc: 'Context aware chat', prompt: 'Summarize my latest uploaded document', icon: <FileTextTwoTone /> },
]

const TABLE_PREVIEW_ROWS = 10

const countPipes = (line) => (line.match(/\|/g) || []).length

const looksLikeTableRow = (line) => {
  const trimmed = (line || '').trim()
  if (countPipes(trimmed) < 2) return false
  const cells = trimmed.split('|').filter((cell) => cell.trim())
  return cells.length >= 2
}

const isTableDivider = (line) => {
  const trimmed = (line || '').trim()
  if (countPipes(trimmed) < 2) return false
  const cleaned = trimmed.replace(/\|/g, '').replace(/:/g, '').replace(/-/g, '').trim()
  return cleaned === ''
}

const splitTableRow = (line) =>
  (line || '')
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((cell) => cell.trim())

const parseMarkdownTableAt = (lines, startIndex) => {
  const header = lines[startIndex]
  const divider = lines[startIndex + 1]
  if (!looksLikeTableRow(header) || !isTableDivider(divider)) return null
  const rows = [splitTableRow(header)]
  let idx = startIndex + 2
  while (idx < lines.length && looksLikeTableRow(lines[idx])) {
    rows.push(splitTableRow(lines[idx]))
    idx += 1
  }
  return { rows, endIndex: idx }
}

const splitMessageBlocks = (content) => {
  if (!content) return []
  const lines = String(content).split(/\r?\n/)
  const blocks = []
  let buffer = []
  let inCode = false

  const flushBuffer = () => {
    if (buffer.length === 0) return
    blocks.push({ type: 'text', content: buffer.join('\n') })
    buffer = []
  }

  for (let idx = 0; idx < lines.length; ) {
    const line = lines[idx]
    const trimmed = line.trim()
    if (trimmed.startsWith('```')) {
      if (!inCode) {
        flushBuffer()
        inCode = true
        buffer.push(line)
        idx += 1
        continue
      }
      buffer.push(line)
      blocks.push({ type: 'code', content: buffer.join('\n') })
      buffer = []
      inCode = false
      idx += 1
      continue
    }

    if (inCode) {
      buffer.push(line)
      idx += 1
      continue
    }

    const table = parseMarkdownTableAt(lines, idx)
    if (table) {
      flushBuffer()
      blocks.push({ type: 'table', rows: table.rows })
      idx = table.endIndex
      continue
    }

    buffer.push(line)
    idx += 1
  }

  flushBuffer()
  return blocks
}

const extractFirstTableRows = (content) => {
  const blocks = splitMessageBlocks(content)
  const tableBlock = blocks.find((block) => block.type === 'table')
  return tableBlock?.rows || null
}

const buildTableModel = (rows) => {
  if (!Array.isArray(rows) || rows.length === 0) return null
  const normalized = rows.map((row) =>
    Array.isArray(row) ? row.map((cell) => (cell ?? '').toString()) : []
  )
  if (normalized.length === 1) {
    const row = normalized[0]
    const colCount = Math.max(row.length, 1)
    const columns = Array.from({ length: colCount }, (_, idx) => ({
      title: `col_${idx + 1}`,
      dataIndex: `col_${idx}`,
      key: `col_${idx}`,
      ellipsis: true,
    }))
    const dataSource = [
      row.reduce(
        (acc, cell, idx) => ({ ...acc, [`col_${idx}`]: cell ?? '' }),
        { key: 'row_0' }
      ),
    ]
    return { columns, dataSource, rowCount: dataSource.length, colCount }
  }

  let header = normalized[0] || []
  let body = normalized.slice(1)
  const headerEmpty = header.every((cell) => !String(cell).trim())
  if (headerEmpty && body.length) {
    header = body[0]
    body = body.slice(1)
  }

  const colCount = Math.max(header.length, ...body.map((row) => row.length), 0)
  if (colCount === 0) return null

  const columns = Array.from({ length: colCount }, (_, idx) => ({
    title: header[idx] || `col_${idx + 1}`,
    dataIndex: `col_${idx}`,
    key: `col_${idx}`,
    ellipsis: true,
  }))

  const dataSource = body.map((row, rowIdx) => {
    const record = { key: `row_${rowIdx}` }
    for (let colIdx = 0; colIdx < colCount; colIdx += 1) {
      record[`col_${colIdx}`] = row[colIdx] ?? ''
    }
    return record
  })

  return { columns, dataSource, rowCount: dataSource.length, colCount }
}

const TableBlock = ({ rows, maxPreviewRows = TABLE_PREVIEW_ROWS, defaultExpanded = false }) => {
  const { token: antdToken } = antdTheme.useToken()
  const [expanded, setExpanded] = useState(defaultExpanded)
  const model = useMemo(() => buildTableModel(rows), [rows])

  if (!model) {
    return (
      <Text type="secondary" style={{ fontSize: 12 }}>
        No table data.
      </Text>
    )
  }

  const { columns, dataSource, rowCount, colCount } = model
  const canExpand = dataSource.length > maxPreviewRows
  const visibleRows = expanded ? dataSource : dataSource.slice(0, maxPreviewRows)

  return (
    <div
      style={{
        border: `1px solid ${antdToken.colorBorderSecondary}`,
        borderRadius: 10,
        background: antdToken.colorFillTertiary,
        padding: 10,
      }}
    >
      <Flex align="center" justify="space-between" wrap style={{ marginBottom: 8 }}>
        <Flex align="center" gap={8}>
          <Tag color="blue" style={{ marginInlineEnd: 0 }}>
            TABLE
          </Tag>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {rowCount} rows · {colCount} cols
          </Text>
        </Flex>
        {canExpand && (
          <Button size="small" type="link" onClick={() => setExpanded((prev) => !prev)}>
            {expanded ? '收起' : `展开 (${dataSource.length})`}
          </Button>
        )}
      </Flex>
      <Table
        size="small"
        columns={columns}
        dataSource={visibleRows}
        pagination={false}
        scroll={{ x: 'max-content' }}
      />
    </div>
  )
}

const MessageBubble = ({
  item,
  isSelected,
  antdToken,
  toggleSelect,
  deleteMessages,
  startStream,
  lastUserPrompt,
  renderSources,
  renderContent,
  roleStyle,
  isError,
}) => {
  const relativeTime = item.ts
    ? (() => {
        const diff = Date.now() - item.ts
        if (diff < 60_000) return `${Math.floor(diff / 1000)}s ago`
        if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`
        return new Date(item.ts).toLocaleTimeString()
      })()
    : ''

  const handleCopy = () => {
    if (item.content) {
      navigator.clipboard?.writeText(item.content)
      antdMessage.success('Copied')
    }
  }

  const handleRetry = () => {
    if (item.role === 'user' && item.content) {
      startStream(item.content)
    } else if (lastUserPrompt) {
      startStream(lastUserPrompt)
    }
  }

  return (
    <Flex vertical align={roleStyle.bubble.alignSelf === 'flex-end' ? 'flex-end' : 'flex-start'} style={{ width: '100%' }}>
      <Flex gap={6} align="center">
        <Checkbox checked={isSelected} onChange={() => toggleSelect(item.id)} />
        <Tag color={roleStyle.tag.color}>{roleStyle.tag.label}</Tag>
        {item.stepId && <Tag color="cyan">{item.stepId}</Tag>}
        {item.tool && <Tag color="blue">{item.tool}</Tag>}
        {item.status && <Tag color={isError ? 'red' : 'gold'}>{item.status}</Tag>}
        <Tag color="default" style={{ color: antdToken.colorTextSecondary }}>
          <ClockCircleOutlined /> {relativeTime || (item.ts ? new Date(item.ts).toLocaleString() : '')}
        </Tag>
      </Flex>
      <div
        style={{
          background: isError ? antdToken.colorErrorBg : roleStyle.bubble.background,
          color: isError ? antdToken.colorError : roleStyle.bubble.color,
          borderRadius: 12,
          border: isError ? `1px solid ${antdToken.colorErrorBorder}` : roleStyle.bubble.border || `1px solid ${antdToken.colorBorder}`,
          padding: 12,
          minWidth: 160,
          maxWidth: '70%',
          whiteSpace: 'pre-wrap',
          textAlign: roleStyle.bubble.textAlign,
          marginTop: 4,
          position: 'relative',
          boxShadow: isSelected ? `0 0 0 2px ${antdToken.colorPrimary}` : 'none',
        }}
      >
        <Flex gap={4} justify="flex-end" style={{ marginBottom: 6 }}>
          <Tooltip title="Copy">
            <Button size="small" type="text" onClick={handleCopy} icon={<CopyOutlined />} />
          </Tooltip>
          {isError && (
            <Tooltip title="Retry">
              <Button size="small" type="text" onClick={handleRetry} icon={<RedoOutlined />} />
            </Tooltip>
          )}
          <Tooltip title="Delete">
            <Button size="small" type="text" icon={<DeleteOutlined />} onClick={() => deleteMessages(new Set([item.id]))} />
          </Tooltip>
        </Flex>
        {item.content
          ? renderContent(item, { textColor: isError ? antdToken.colorError : roleStyle.bubble.color })
          : <Spin size="small" />}
        {renderSources(item.sources)}
      </div>
    </Flex>
  )
}

export default function Chat({ token, userId, setUserId }) {
  const [sessionId, setSessionId] = useState(null)
  const [text, setText] = useState('')
  const [messages, setMessages] = useState([])
  const [planState, setPlanState] = useState(INITIAL_PLAN)
  const [status, setStatus] = useState('idle')
  const [loading, setLoading] = useState(false)
  const [resumeLoading, setResumeLoading] = useState(false)
  const [interruptInfo, setInterruptInfo] = useState(null)
  const [interruptVisible, setInterruptVisible] = useState(false)
  const [rememberNext, setRememberNext] = useState(false)
  const [sseNotice, setSseNotice] = useState('idle')
  const [lastUserPrompt, setLastUserPrompt] = useState('')
  const [selectedIds, setSelectedIds] = useState(new Set())
  const [threads, setThreads] = useState([])
  const [threadsLoading, setThreadsLoading] = useState(false)
  const [threadActionId, setThreadActionId] = useState(null)

  const esRef = useRef(null)
  const liveMessageRef = useRef(null)
  const lastResultRef = useRef({})
  const sourcesRef = useRef([])
  const chatAreaRef = useRef(null)

  const { token: antdToken } = antdTheme.useToken()
  const planSteps = planState.steps
  const resolvedToken = useMemo(() => token || API_KEY || '', [token])
  const activeThread = useMemo(() => threads.find((item) => item.thread_id === sessionId), [threads, sessionId])

  const statusBanner = useMemo(() => {
    const statusTextMap = {
      planning: 'Planning...',
      planning_completed: 'Plan ready',
      executing: 'Executing',
      waiting: 'Waiting for input',
      completed: 'Completed',
      idle: 'Idle',
    }
    const textLabel = statusTextMap[status] || status
    const type =
      status === 'completed'
        ? 'success'
        : status === 'waiting'
        ? 'warning'
        : status === 'executing' || status === 'planning' || sseNotice === 'streaming'
        ? 'info'
        : status === 'error'
        ? 'error'
        : 'info'
    return (
      <Alert
        type={type}
        showIcon
        message={textLabel}
        description={
          sseNotice === 'error'
            ? 'Stream interrupted or failed.'
            : sseNotice === 'streaming'
            ? 'Receiving live response...'
            : undefined
        }
        style={{ borderRadius: 8, background: antdToken.colorFillTertiary }}
      />
    )
  }, [status, sseNotice, antdToken])

  const clearLiveMessage = useCallback(() => {
    liveMessageRef.current = null
    sourcesRef.current = []
    setMessages((prev) => prev.filter((item) => item.id !== 'ai-live'))
  }, [])

  const stopStream = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
      setLoading(false)
      setResumeLoading(false)
      clearLiveMessage()
      setSseNotice('stopped')
    }
  }, [clearLiveMessage])

  const loadThreadMessages = useCallback(
    async (threadId) => {
      if (!threadId || !userId) return
      setMessages([])
      setPlanState(INITIAL_PLAN)
      setStatus('idle')
      setSseNotice('idle')
      setSelectedIds(new Set())
      lastResultRef.current = {}
      liveMessageRef.current = null
      sourcesRef.current = []
      try {
        const res = await fetchThreadMessages(threadId, userId, resolvedToken)
        const list = Array.isArray(res.items) ? res.items : []
        setMessages(
          list.map((item) => ({
            id: item.id || uid(),
            role: item.role || 'system',
            content: item.content || '',
            ts: item.created_at ? item.created_at * 1000 : Date.now(),
            metadata: item.metadata,
          }))
        )
      } catch (err) {
        antdMessage.error(err?.message || 'Failed to load messages')
      }
    },
    [userId, resolvedToken]
  )

  const refreshThreads = useCallback(async () => {
    if (!userId) {
      setThreads([])
      return
    }
    setThreadsLoading(true)
    try {
      const res = await fetchThreads(userId, resolvedToken)
      const items = Array.isArray(res.items) ? res.items : []
      setThreads(items)
      if (!sessionId && items.length > 0) {
        const nextId = items[0].thread_id || items[0].id
        setSessionId(nextId)
        await loadThreadMessages(nextId)
      }
    } catch (err) {
      antdMessage.error(err?.message || 'Failed to load threads')
    } finally {
      setThreadsLoading(false)
    }
  }, [userId, resolvedToken, sessionId, loadThreadMessages])

  const selectThread = useCallback(
    async (threadId) => {
      if (!threadId) return
      stopStream()
      setSessionId(threadId)
      await loadThreadMessages(threadId)
    },
    [stopStream, loadThreadMessages]
  )

  const startNewThread = useCallback(
    async (titleHint = '') => {
      if (!userId) {
        antdMessage.warning('User ID is required')
        return null
      }
      setThreadActionId('new')
      try {
        const res = await createThread(userId, resolvedToken, titleHint)
        const newId = res.thread_id || res.id
        if (newId) {
          setSessionId(newId)
          await loadThreadMessages(newId)
          refreshThreads()
        }
        return newId || null
      } catch (err) {
        antdMessage.error(err?.message || 'Failed to create thread')
        return null
      } finally {
        setThreadActionId(null)
      }
    },
    [userId, resolvedToken, loadThreadMessages, refreshThreads]
  )

  const handleDeleteThread = useCallback(
    async (threadId) => {
      if (!threadId || !userId) return
      setThreadActionId(threadId)
      try {
        await deleteThread(threadId, userId, resolvedToken)
        if (sessionId === threadId) {
          setSessionId(null)
          setMessages([])
          setPlanState(INITIAL_PLAN)
        }
        refreshThreads()
      } catch (err) {
        antdMessage.error(err?.message || 'Failed to delete thread')
      } finally {
        setThreadActionId(null)
      }
    },
    [userId, resolvedToken, sessionId, refreshThreads]
  )

  const handleRenameThread = useCallback(
    async (thread) => {
      if (!thread) return
      const next = window.prompt('New thread title', thread.title || 'Untitled chat')
      if (!next || next.trim() === (thread.title || '').trim()) return
      setThreadActionId(thread.thread_id)
      try {
        await renameThread(thread.thread_id, userId, next.trim(), resolvedToken)
        refreshThreads()
      } catch (err) {
        antdMessage.error(err?.message || 'Rename failed')
      } finally {
        setThreadActionId(null)
      }
    },
    [userId, resolvedToken, refreshThreads]
  )

  useEffect(() => {
    if (!userId) {
      setThreads([])
      setSessionId(null)
      setMessages([])
      setPlanState(INITIAL_PLAN)
      setStatus('idle')
      setSseNotice('idle')
      return
    }
    refreshThreads()
  }, [userId, refreshThreads])

  const appendMessage = useCallback((message) => {
    setMessages((prev) => [
      ...prev,
      {
        id: message.id || uid(),
        ts: Date.now(),
        ...message,
      },
    ])
  }, [])

  const updatePlanWithPayload = useCallback((payload) => {
    setPlanState((prev) => {
      const steps = Array.isArray(payload.plan) ? payload.plan : prev.steps
      return {
        steps,
        currentStep: typeof payload.current_step === 'number' ? payload.current_step : prev.currentStep,
        activeStepId: payload.active_step_id || prev.activeStepId,
      }
    })
  }, [])

  const finalizeLiveMessage = useCallback(
    ({ force = false } = {}) => {
      const live = liveMessageRef.current
      if (!live) return
      const content = (live.content || '').trim()
      if (content || force) {
        appendMessage({
          role: 'ai',
          content: content || '(empty response)',
          sources: sourcesRef.current,
        })
      }
      clearLiveMessage()
    },
    [appendMessage, clearLiveMessage]
  )

  const handlePlanEvent = useCallback(
    (data) => {
      updatePlanWithPayload(data)
      lastResultRef.current = {}
    },
    [updatePlanWithPayload]
  )

  const handleStepUpdate = useCallback(
    (payload) => {
      if (!payload || !payload.step_id) return

      setPlanState((prev) => {
        const steps = prev.steps.map((step) => (step.id === payload.step_id ? { ...step, ...payload } : step))
        return { ...prev, steps }
      })

      if (typeof payload.result === 'string' && payload.result.trim()) {
        const prevResult = lastResultRef.current[payload.step_id]
        if (prevResult !== payload.result) {
          lastResultRef.current[payload.step_id] = payload.result
          appendMessage({
            role: 'ai',
            content: payload.result.trim(),
            stepId: payload.step_id,
            sources: sourcesRef.current,
          })
          sourcesRef.current = []
        }
      }
    },
    [appendMessage]
  )

  const mergeLiveChunk = useCallback((delta) => {
    setMessages((prev) => {
      const withoutLive = prev.filter((item) => item.id !== 'ai-live')
      const live =
        liveMessageRef.current ||
        {
          id: 'ai-live',
          role: 'ai',
          content: '',
          sources: [],
        }
      live.content = `${live.content || ''}${delta || ''}`
      liveMessageRef.current = live
      return [...withoutLive, { ...live }]
    })
  }, [])

  const handleMessageChunk = useCallback(
    (data) => {
      if (data.role === 'ai') {
        mergeLiveChunk(data.delta)
        return
      }

      if (data.role === 'tool') {
        appendMessage({
          role: 'tool',
          observation: data.observation,
          tool: data.tool,
          status: data.status,
          latency: data.latency,
          content: data.content || data.observation || '',
          node: data.node,
          stepId: data.step_id,
        })
        return
      }

      appendMessage({
        role: data.role || 'system',
        content: data.content || '',
      })
    },
    [appendMessage, mergeLiveChunk]
  )

  const handleStreamEvent = useCallback(
    (event) => {
      if (!event || typeof event !== 'object') return

      switch (event.event) {
        case 'plan':
          handlePlanEvent(event)
          setStatus('planning_completed')
          setSseNotice('planning_completed')
          break
        case 'status':
          setStatus(event.status || 'idle')
          break
        case 'step_update':
          handleStepUpdate(event.payload)
          break
        case 'interrupt':
          setInterruptInfo(event.payload || null)
          setInterruptVisible(true)
          setLoading(false)
          setResumeLoading(false)
          clearLiveMessage()
          break
        case 'cancelled':
          if (Array.isArray(event.plan)) {
            setPlanState((prev) => ({ ...prev, steps: event.plan }))
          }
          setStatus('cancelled')
          setResumeLoading(false)
          setInterruptVisible(false)
          clearLiveMessage()
          break
        case 'retrieval':
          sourcesRef.current = (event.payload && event.payload.retrieval_results) || []
          break
        case 'end':
          finalizeLiveMessage()
          setLoading(false)
          setResumeLoading(false)
          setSseNotice('completed')
          refreshThreads()
          break
        case 'error':
          setLoading(false)
          setResumeLoading(false)
          setSseNotice('error')
          antdMessage.error(event.message || 'Execution error')
          clearLiveMessage()
          break
        default:
          if ('role' in event) {
            handleMessageChunk(event)
          } else if (event.payload?.plan && Array.isArray(event.payload.plan)) {
            setPlanState((prev) => ({ ...prev, steps: event.payload.plan }))
          }
      }
    },
    [handlePlanEvent, handleStepUpdate, handleMessageChunk, finalizeLiveMessage, clearLiveMessage, refreshThreads]
  )

  const attachEventSource = useCallback(
    (es) => {
      es.onmessage = (ev) => {
        if (!ev?.data) return
        try {
          const parsed = JSON.parse(ev.data)
          handleStreamEvent(parsed)
        } catch (err) {
          console.warn('failed to parse stream event', err)
        }
      }

      es.onerror = () => {
        setLoading(false)
        setResumeLoading(false)
        clearLiveMessage()
        setSseNotice('error')
        es.close()
        esRef.current = null
      }
    },
    [handleStreamEvent, clearLiveMessage]
  )

  const buildStreamUrl = useCallback(
    (basePath, params) => {
      const search = new URLSearchParams(params)
      if (resolvedToken) {
        search.set('api_key', resolvedToken)
      }
      return `${API_BASE}${basePath}?${search.toString()}`
    },
    [resolvedToken]
  )

  const initializeStream = useCallback(
    ({ url, userMessage, resetPlan = false, expectStream = true }) => {
      if (!url) return

      if (esRef.current) {
        esRef.current.close()
        esRef.current = null
      }

      if (resetPlan) {
        setPlanState(INITIAL_PLAN)
        setStatus('planning')
        setSseNotice('planning')
        lastResultRef.current = {}
      }

      sourcesRef.current = []
      liveMessageRef.current = null

      if (userMessage) {
        const userItem = { id: uid(), role: 'user', content: userMessage }
        const aiPlaceholder = {
          id: 'ai-live',
          role: 'ai',
          content: '',
          sources: [],
        }
        setMessages((prev) => {
          const withoutLive = prev.filter((item) => item.id !== 'ai-live')
          return [...withoutLive, userItem, aiPlaceholder]
        })
      }

      const es = new EventSource(url)
      esRef.current = es
      attachEventSource(es)

      if (!expectStream) {
        setLoading(false)
      }
    },
    [attachEventSource]
  )

  const startStream = useCallback(
    async (overridePrompt) => {
      const promptValue = typeof overridePrompt === 'string' ? overridePrompt : text
      const prompt = (promptValue || '').trim()
      if (!prompt) {
        antdMessage.warning('Please enter a prompt')
        return
      }

      if (!userId?.trim()) {
        antdMessage.warning('User ID is required')
        return
      }

      setLoading(true)
      setSseNotice('streaming')
      setResumeLoading(false)
      setInterruptVisible(false)
      setInterruptInfo(null)

      let threadId = sessionId
      if (!threadId) {
        threadId = await startNewThread(prompt.slice(0, 60))
      }
      if (!threadId) {
        setLoading(false)
        setSseNotice('idle')
        return
      }

      const url = buildStreamUrl('/chat/stream', {
        session_id: threadId,
        user_id: userId,
        message: prompt,
        remember: rememberNext ? '1' : '0',
      })

      initializeStream({
        url,
        userMessage: prompt,
        resetPlan: true,
        expectStream: true,
      })
      setSessionId(threadId)
      setLastUserPrompt(prompt)
      if (typeof overridePrompt === 'undefined') {
        setText('')
      }
      if (rememberNext) {
        setRememberNext(false)
      }
    },
    [text, userId, sessionId, rememberNext, initializeStream, buildStreamUrl, startNewThread]
  )

  const resendLast = useCallback(() => {
    if (!lastUserPrompt) {
      antdMessage.info('No previous prompt to resend')
      return
    }
    startStream(lastUserPrompt)
  }, [lastUserPrompt, startStream])

  const toggleSelect = useCallback((id) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
  }, [])

  const deleteMessages = useCallback((ids) => {
    if (!ids || ids.size === 0) return
    setMessages((prev) => prev.filter((m) => !ids.has(m.id)))
    setSelectedIds(new Set())
  }, [])

  const copyMessages = useCallback((msgs) => {
    if (!msgs || msgs.length === 0) {
      antdMessage.info('No messages to copy')
      return
    }
    const txt = msgs.map((m) => `[${m.role}] ${m.content || ''}`).join('\n\n')
    navigator.clipboard?.writeText(txt)
    antdMessage.success('Copied')
  }, [])

  const exportJSON = useCallback(
    (msgs) => {
      const payload = msgs.map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        ts: m.ts,
        status: m.status,
        stepId: m.stepId,
        tool: m.tool,
      }))
      copyMessages([{ role: 'export', content: JSON.stringify(payload, null, 2) }])
    },
    [copyMessages]
  )

  const exportMarkdown = useCallback(
    (msgs) => {
      const md = msgs
        .map((m) => `### ${m.role}${m.stepId ? ` (${m.stepId})` : ''}\n${m.content || ''}`)
        .join('\n\n')
      copyMessages([{ role: 'export', content: md }])
    },
    [copyMessages]
  )

  const copySelectedOrLast = useCallback(
    (count = 5) => {
      const pool = messages
      const candidates = selectedIds.size > 0 ? pool.filter((m) => selectedIds.has(m.id)) : pool.slice(-count)
      copyMessages(candidates)
    },
    [messages, selectedIds, copyMessages]
  )

  const resume = useCallback(
    (action) => {
      if (!sessionId) return
      if (!['continue', 'cancel'].includes(action)) return

      const url = buildStreamUrl('/chat/continue', {
        thread_id: sessionId,
        action,
        user_id: userId,
        nonce: Date.now(),
      })

      setInterruptVisible(false)
      setInterruptInfo(null)

      if (action === 'continue') {
        setLoading(true)
        setResumeLoading(true)
        initializeStream({
          url,
          resetPlan: false,
          expectStream: true,
        })
      } else {
        setResumeLoading(true)
        initializeStream({
          url,
          resetPlan: false,
          expectStream: false,
        })
      }
    },
    [sessionId, userId, initializeStream, buildStreamUrl]
  )

  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight
    }
  }, [messages])

  const renderSources = (sources) => {
    if (!Array.isArray(sources) || sources.length === 0) return null
    return (
      <div
        style={{
          marginTop: 12,
          padding: 10,
          borderRadius: 8,
          background: antdToken.colorFillSecondary,
          border: '1px solid ' + antdToken.colorBorderSecondary,
        }}
      >
        <Flex vertical gap={8} style={{ width: '100%' }}>
          <Text type="secondary" strong>
            Sources
          </Text>
          {sources.map((source, index) => {
            const name =
              (source.metadata && source.metadata.filename) ||
              source.document_id ||
              `chunk-${index + 1}`
            const score =
              typeof source.score === 'number'
                ? ` (score ${source.score.toFixed(2)})`
                : ''
            const metadata = source.metadata || {}
            const tableRows =
              metadata.structure_type === 'table'
                ? (Array.isArray(metadata.rows) ? metadata.rows : extractFirstTableRows(source.content))
                : null
            return (
              <div
                key={`${source.chunk_id || source.document_id || index}`}
                style={{
                  background: antdToken.colorBgContainer,
                  borderRadius: 6,
                  padding: 8,
                  border: '1px solid ' + antdToken.colorBorderSecondary,
                }}
              >
                <Flex gap={8} align="center" style={{ marginBottom: 4 }}>
                  <Tag color="cyan">{name}</Tag>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {score || 'retrieved'}
                  </Text>
                </Flex>
                {tableRows ? (
                  <TableBlock rows={tableRows} />
                ) : (
                  <Paragraph
                    ellipsis={{ rows: 2, expandable: true, symbol: 'more' }}
                    style={{ marginBottom: 0, color: antdToken.colorTextSecondary, fontSize: 13 }}
                  >
                    {source.content || '(empty snippet)'}
                  </Paragraph>
                )}
              </div>
            )
          })}
        </Flex>
      </div>
    )
  }

  const renderMessageContent = (item, { textColor }) => {
    const blocks = splitMessageBlocks(item.content || '')
    if (!blocks.length) {
      return (
        <Text style={{ color: textColor, whiteSpace: 'pre-wrap', display: 'block' }}>
          {item.content}
        </Text>
      )
    }

    return (
      <Flex vertical gap={8}>
        {blocks.map((block, index) => {
          const key = `${block.type}-${index}`
          if (block.type === 'table') {
            return <TableBlock key={key} rows={block.rows} />
          }
          const baseStyle = {
            color: textColor,
            whiteSpace: 'pre-wrap',
            display: 'block',
            margin: 0,
          }
          if (block.type === 'code') {
            return (
              <pre
                key={key}
                style={{
                  ...baseStyle,
                  fontFamily:
                    'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
                }}
              >
                {block.content}
              </pre>
            )
          }
          return (
            <Text key={key} style={baseStyle}>
              {block.content}
            </Text>
          )
        })}
      </Flex>
    )
  }

  const renderMessage = (item) => {
    const roleStyle = ROLE_CONFIG[item.role] || ROLE_CONFIG.system
    const isError =
      item.status === 'error' || (item.role === 'system' && /error|failed/i.test(item.content || ''))

    return (
      <MessageBubble
        item={item}
        isSelected={selectedIds.has(item.id)}
        antdToken={antdToken}
        toggleSelect={toggleSelect}
        deleteMessages={deleteMessages}
        startStream={startStream}
        lastUserPrompt={lastUserPrompt}
        renderSources={renderSources}
        renderContent={renderMessageContent}
        roleStyle={roleStyle}
        isError={isError}
      />
    )
  }

  const planSidebar = useMemo(() => {
    if (!planSteps.length) {
      return (
        <Flex vertical justify="center" align="center" style={{ height: '100%', gap: 8 }}>
          <Text type="secondary" strong>
            No active workflow
          </Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            Start a chat to see the plan here.
          </Text>
        </Flex>
      )
    }
    const finishedSteps = planSteps.filter((step) => step.status === 'completed').length
    const percent = planSteps.length === 0 ? 0 : Math.round((finishedSteps / planSteps.length) * 100)
    const activeStep = planSteps.find((step) => step.id === planState.activeStepId)

    return (
      <Flex vertical gap={16} style={{ padding: 16, height: '100%', overflow: 'hidden' }}>
        <Flex justify="space-between" align="center">
          <Text strong style={{ fontSize: 14, letterSpacing: 0.5 }}>
            WORKFLOW
          </Text>
          <Tag color={status === 'idle' ? 'default' : 'blue'}>{status.toUpperCase()}</Tag>
        </Flex>
        {activeStep && (
          <Alert
            message="Active Step"
            description={`${activeStep.title} - ${STEP_STATUS_TEXT[activeStep.status] || activeStep.status}`}
            type="info"
            showIcon
            style={{ borderRadius: 8 }}
          />
        )}
        <Divider style={{ margin: '4px 0' }} />
        <div style={{ flex: 1, overflowY: 'auto', paddingRight: 8 }}>
          <Steps
            direction="vertical"
            size="small"
            items={planSteps.map((step) => ({
              title: step.title,
              description: step.description,
              status: STEP_STATUS_MAP[step.status] || 'wait',
            }))}
          />
          <Divider />
          <Text type="secondary" style={{ fontSize: 12 }}>
            Progress: {percent}% ({finishedSteps}/{planSteps.length})
          </Text>
        </div>
      </Flex>
    )
  }, [planSteps, planState.activeStepId, status])

  const hero = (
    <Flex vertical gap={16} align="center" justify="center" style={{ padding: '32px 0' }}>
      <div
        style={{
          width: 64,
          height: 64,
          borderRadius: 18,
          background: antdToken.colorPrimaryBg,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <MessageOutlined style={{ fontSize: 28, color: antdToken.colorPrimary }} />
      </div>
      <Flex vertical align="center" gap={8} style={{ textAlign: 'center' }}>
        <Text style={{ fontSize: 28, fontWeight: 700, color: antdToken.colorText }}>
          How can I help you today?
        </Text>
        <Text type="secondary">
          I can execute complex plans, search the web, and help you solve problems with full transparency.
        </Text>
      </Flex>
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: 12,
          width: '100%',
          maxWidth: 720,
        }}
      >
        {heroPrompts.map((item) => (
          <Card
            key={item.title}
            hoverable
            onClick={() => {
              setText(item.prompt)
              startStream(item.prompt)
            }}
            style={{ borderRadius: 14, boxShadow: antdToken.boxShadowTertiary }}
          >
            <Flex align="center" justify="space-between">
              <div>
                <Text strong>{item.title}</Text>
                <div>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {item.desc}
                  </Text>
                </div>
              </div>
              <div style={{ fontSize: 18, color: antdToken.colorPrimary }}>{item.icon}</div>
            </Flex>
          </Card>
        ))}
      </div>
    </Flex>
  )

  return (
    <Flex
      style={{
        background: '#f7f9fc',
        height: 'calc(100vh - 120px)',
        borderRadius: 18,
        border: '1px solid ' + antdToken.colorBorder,
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          width: 300,
          background: '#f7f9fc',
          color: '#0f172a',
          borderRight: '1px solid #e5e7eb',
          padding: 16,
          display: 'flex',
          flexDirection: 'column',
          gap: 14,
          boxShadow: 'inset -1px 0 0 #e5e7eb',
        }}
      >
        <Flex justify="space-between" align="center">
          <Flex align="center" gap={10}>
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: 10,
                background: '#e7f0ff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <MessageOutlined style={{ color: '#2563eb' }} />
            </div>
            <div>
              <Text strong style={{ fontSize: 16, color: '#0f172a' }}>
                聊天记录
              </Text>
              <div>
                <Text type="secondary" style={{ color: '#475569', fontSize: 12 }}>
                  ChatGPT 风格侧边栏
                </Text>
              </div>
            </div>
          </Flex>
          <Tooltip title="刷新会话列表">
            <Button
              icon={<ReloadOutlined />}
              size="small"
              type="text"
              onClick={refreshThreads}
              loading={threadsLoading}
              style={{ color: '#475569' }}
            />
          </Tooltip>
        </Flex>
        <Button
          icon={<PlusOutlined />}
          block
          onClick={() => startNewThread()}
          loading={threadActionId === 'new'}
          disabled={!userId}
          style={{
            background: '#2563eb',
            border: '1px solid #1d4ed8',
            color: '#ffffff',
            fontWeight: 700,
            height: 46,
            borderRadius: 12,
            boxShadow: '0 10px 30px rgba(37, 99, 235, 0.18)',
          }}
        >
          新建对话
        </Button>
        <div
          style={{
            padding: 12,
            borderRadius: 12,
            background: '#ffffff',
            border: '1px solid #e5e7eb',
          }}
        >
          <Text style={{ color: '#0f172a', fontWeight: 600 }}>当前用户</Text>
          <div style={{ marginTop: 6, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Tag color="geekblue" style={{ border: 'none', background: '#e0e7ff', color: '#3730a3' }}>
              {userId || '未设置'}
            </Tag>
            <Tag color="green" style={{ border: 'none', background: '#dcfce7', color: '#166534' }}>
              {sessionId ? '已连接' : '待开始'}
            </Tag>
          </div>
        </div>
        <Text type="secondary" style={{ color: '#475569', fontSize: 12, letterSpacing: 1 }}>
          RECENT
        </Text>
        <div style={{ flex: 1, overflowY: 'auto', marginRight: -6, paddingRight: 6 }}>
          {threads.length === 0 ? (
            <div
              style={{
                border: '1px dashed #e2e8f0',
                borderRadius: 12,
                padding: 14,
                color: '#475569',
                textAlign: 'center',
              }}
            >
              <MessageOutlined style={{ fontSize: 18, color: '#94a3b8' }} /> 开始聊天以创建新的会话
            </div>
          ) : (
            <List
              dataSource={threads}
              rowKey={(item) => item.thread_id}
              renderItem={(item) => {
                const isActive = item.thread_id === sessionId
                const updatedText = item.updated_at ? new Date(item.updated_at * 1000).toLocaleString() : ''
                return (
                  <List.Item
                    style={{
                      cursor: 'pointer',
                      background: 'transparent',
                      borderRadius: 12,
                      marginBottom: 10,
                      border: isActive ? '1px solid #d0d7ff' : '1px solid #e5e7eb',
                      padding: 12,
                      transition: 'all 0.2s ease',
                      color: '#0f172a',
                    }}
                    onClick={() => selectThread(item.thread_id)}
                  >
                    <Flex justify="space-between" align="flex-start" gap={8} style={{ width: '100%' }}>
                      <Flex vertical gap={4} style={{ flex: 1 }}>
                        <Text strong style={{ color: '#0f172a' }}>
                          {item.title || 'Untitled chat'}
                        </Text>
                        <Text type="secondary" style={{ fontSize: 11, color: '#94a3b8' }}>
                          {updatedText}
                        </Text>
                      </Flex>
                      <Space size="small">
                        <Tooltip title="重命名会话">
                          <Button
                            size="small"
                            type="text"
                            icon={<EditOutlined />}
                            onClick={(e) => {
                              e.stopPropagation()
                              handleRenameThread(item)
                            }}
                            loading={threadActionId === item.thread_id}
                            style={{ color: '#475569' }}
                          />
                        </Tooltip>
                        <Tooltip title="删除会话">
                          <Button
                            size="small"
                            type="text"
                            danger
                            icon={<DeleteOutlined />}
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDeleteThread(item.thread_id)
                            }}
                            loading={threadActionId === item.thread_id}
                          />
                        </Tooltip>
                      </Space>
                    </Flex>
                  </List.Item>
                )
              }}
            />
          )}
        </div>
      </div>

      <Flex vertical style={{ flex: 1, padding: 24, overflow: 'hidden', gap: 16 }}>
        <Flex justify="space-between" align="center" wrap style={{ gap: 12 }}>
          <div>
            <Text strong style={{ fontSize: 18 }}>
              {activeThread?.title || 'New chat'}
            </Text>
            <div>
              <Text type="secondary" style={{ fontSize: 12 }}>
                Thread: {sessionId || 'pending'} - User: {userId || 'not set'}
              </Text>
            </div>
          </div>
          {statusBanner}
        </Flex>

        <div
          ref={chatAreaRef}
          style={{
            flex: 1,
            overflowY: 'auto',
            padding: '0 8px',
          }}
        >
          {messages.length === 0 ? (
            hero
          ) : (
            <List
              dataSource={messages}
              rowKey={(item) => item.id}
              split={false}
              renderItem={(item) => (
                <List.Item style={{ padding: '0', margin: 0 }}>
                  {renderMessage(item)}
                </List.Item>
              )}
            />
          )}
        </div>

        <Card
          style={{
            borderRadius: 14,
            boxShadow: antdToken.boxShadowTertiary,
            border: '1px solid ' + antdToken.colorBorder,
          }}
          bodyStyle={{ padding: 12 }}
        >
          <Flex vertical gap="small">
            <Flex gap="small" wrap align="center">
              <Button size="small" type="text" onClick={() => exportJSON(messages)}>
                JSON
              </Button>
              <Button size="small" type="text" onClick={() => exportMarkdown(messages)}>
                MD
              </Button>
              <Button size="small" type="text" onClick={() => copySelectedOrLast(5)}>
                Copy last 5
              </Button>
              <Checkbox checked={rememberNext} onChange={(e) => setRememberNext(e.target.checked)}>
                Remember this turn
              </Checkbox>
            </Flex>
            <Flex gap="small" align="flex-end">
              <TextArea
                autoSize={{ minRows: 2, maxRows: 4 }}
                placeholder="Type your message..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                onPressEnter={(e) => {
                  if (!e.shiftKey) {
                    e.preventDefault()
                    startStream()
                  }
                }}
                style={{ flex: 1, borderRadius: 10 }}
                disabled={loading}
              />
              <Tooltip title="Send">
                <Button
                  type="primary"
                  icon={<SendOutlined />}
                  onClick={startStream}
                  loading={loading && !resumeLoading}
                  disabled={!text.trim() || !userId?.trim() || loading}
                  style={{ height: '100%' }}
                />
              </Tooltip>
              <Tooltip title="Stop current request">
                <Button
                  icon={<StopOutlined />}
                  onClick={stopStream}
                  disabled={!loading && !resumeLoading}
                  danger
                  style={{ height: '100%' }}
                />
              </Tooltip>
            </Flex>
          </Flex>
        </Card>
      </Flex>

      <div
        style={{
          width: 320,
          background: '#f7f9fc',
          borderLeft: '1px solid ' + antdToken.colorBorder,
          padding: 12,
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        {planSidebar}
      </div>
      <Modal
        open={interruptVisible}
        title="Execution paused"
        onCancel={() => {
          setInterruptVisible(false)
          setInterruptInfo(null)
        }}
        footer={[
          <Button key="cancel" icon={<StopOutlined />} onClick={() => resume('cancel')} loading={resumeLoading}>
            Cancel
          </Button>,
          <Button
            key="continue"
            type="primary"
            onClick={() => resume('continue')}
            icon={<SendOutlined />}
            loading={resumeLoading}
          >
            Continue
          </Button>,
        ]}
        centered
      >
        <Flex vertical gap={12} style={{ width: '100%' }}>
          <Text>{interruptInfo?.message || 'The workflow requires confirmation. Continue?'}</Text>
          {interruptInfo?.options && (
            <Flex wrap gap="small">
              {interruptInfo.options.map((option) => (
                <Tag key={option} color="blue">
                  {option}
                </Tag>
              ))}
            </Flex>
          )}
        </Flex>
      </Modal>
    </Flex>
  )
}
