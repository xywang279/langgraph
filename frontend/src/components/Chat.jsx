import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
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
  Steps,
  Modal,
  Upload,
  Checkbox,
  Divider,
  Progress,
  Empty,
  Tabs,
} from 'antd'
import {
  SendOutlined,
  StopOutlined,
  ReloadOutlined,
  CloudUploadOutlined,
  SaveOutlined,
  UserOutlined,
  DatabaseOutlined,
  InboxOutlined,
} from '@ant-design/icons'

import { API_BASE } from '../api'

const { Text, Paragraph } = Typography
const { Dragger } = Upload

const uid = () => Math.random().toString(36).slice(2)

const INITIAL_PLAN = { steps: [], currentStep: 0, activeStepId: null }

const ROLE_CONFIG = {
  user: {
    tag: { color: 'processing', label: 'USER' },
    bubble: { background: '#1d4ed8', color: '#e0f2fe', alignSelf: 'flex-end' },
  },
  ai: {
    tag: { color: 'success', label: 'ASSISTANT' },
    bubble: { background: '#065f46', color: '#d1fae5', alignSelf: 'flex-start' },
  },
  tool: {
    tag: { color: 'geekblue', label: 'TOOL' },
    bubble: { background: '#111827', color: '#facc15', alignSelf: 'stretch' },
  },
  system: {
    tag: { color: 'purple', label: 'SYSTEM' },
    bubble: { background: '#312e81', color: '#ede9fe', alignSelf: 'flex-start' },
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

const usePersistentUserId = () => {
  const readId = () => {
    try {
      return window.localStorage.getItem('lg-user-id') || 'user-default'
    } catch (err) {
      console.warn('unable to read user-id from storage', err)
      return 'user-default'
    }
  }

  const [userId, setUserId] = useState(readId)

  useEffect(() => {
    try {
      window.localStorage.setItem('lg-user-id', userId)
    } catch (err) {
      console.warn('unable to persist user-id', err)
    }
  }, [userId])

  return [userId, setUserId]
}

export default function Chat() {
  const [userId, setUserId] = usePersistentUserId()
  const [sessionId, setSessionId] = useState(() => `web-${uid()}`)
  const [text, setText] = useState('What time is it? Please use the tools.')
  const [messages, setMessages] = useState([])
  const [planState, setPlanState] = useState(INITIAL_PLAN)
  const [status, setStatus] = useState('idle')
  const [lastUpdate, setLastUpdate] = useState(null)
  const [loading, setLoading] = useState(false)
  const [resumeLoading, setResumeLoading] = useState(false)
  const [interruptInfo, setInterruptInfo] = useState(null)
  const [interruptVisible, setInterruptVisible] = useState(false)
  const [rememberNext, setRememberNext] = useState(false)
  const [documents, setDocuments] = useState([])
  const [documentsLoading, setDocumentsLoading] = useState(false)

  const esRef = useRef(null)
  const liveMessageRef = useRef(null)
  const lastResultRef = useRef({})
  const sourcesRef = useRef([])

  const planSteps = planState.steps

  const resetSession = useCallback(() => {
    setSessionId(`web-${uid()}`)
    setMessages([])
    setPlanState(INITIAL_PLAN)
    setStatus('idle')
    setLastUpdate(null)
    lastResultRef.current = {}
    liveMessageRef.current = null
    sourcesRef.current = []
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
    }
  }, [])

  const appendMessage = useCallback(
    (message) => {
      setMessages((prev) => [
        ...prev,
        {
          id: message.id || uid(),
          ts: Date.now(),
          ...message,
        },
      ])
    },
    [setMessages]
  )

  const updatePlanWithPayload = useCallback(
    (payload) => {
      setPlanState((prev) => {
        const steps = Array.isArray(payload.plan) ? payload.plan : prev.steps
        return {
          steps,
          currentStep:
            typeof payload.current_step === 'number'
              ? payload.current_step
              : prev.currentStep,
          activeStepId: payload.active_step_id || prev.activeStepId,
        }
      })
    },
    []
  )

  const clearLiveMessage = useCallback(() => {
    liveMessageRef.current = null
    sourcesRef.current = []
    setMessages((prev) => prev.filter((item) => item.id !== 'ai-live'))
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
        const steps = prev.steps.map((step) =>
          step.id === payload.step_id ? { ...step, ...payload } : step
        )
        return { ...prev, steps }
      })

      setLastUpdate({
        stepId: payload.step_id,
        status: payload.status,
        result: payload.result,
        timestamp: Date.now(),
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
          sourcesRef.current =
            (event.payload && event.payload.retrieval_results) || []
          break
        case 'end':
          finalizeLiveMessage()
          setLoading(false)
          setResumeLoading(false)
          break
        case 'error':
          setLoading(false)
          setResumeLoading(false)
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
    [
      handlePlanEvent,
      handleStepUpdate,
      handleMessageChunk,
      finalizeLiveMessage,
      clearLiveMessage,
    ]
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
        es.close()
        esRef.current = null
      }
    },
    [handleStreamEvent, clearLiveMessage]
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

  const buildStreamUrl = useCallback(
    (basePath, params) => {
      const search = new URLSearchParams(params)
      return `${API_BASE}${basePath}?${search.toString()}`
    },
    []
  )

  const startStream = useCallback(() => {
    if (!text.trim()) {
      antdMessage.warning('Please enter a prompt')
      return
    }

    if (!userId.trim()) {
      antdMessage.warning('User ID is required')
      return
    }

    setLoading(true)
    setResumeLoading(false)
    setInterruptVisible(false)
    setInterruptInfo(null)

    const url = buildStreamUrl('/chat/stream', {
      session_id: sessionId,
      user_id: userId,
      message: text,
      remember: rememberNext ? '1' : '0',
    })

    initializeStream({
      url,
      userMessage: text,
      resetPlan: true,
      expectStream: true,
    })
    setText('')
    if (rememberNext) {
      setRememberNext(false)
    }
  }, [
    text,
    userId,
    sessionId,
    rememberNext,
    initializeStream,
    buildStreamUrl,
  ])

  const stopStream = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
      setLoading(false)
      setResumeLoading(false)
      clearLiveMessage()
    }
  }, [clearLiveMessage])

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

  const refreshDocuments = useCallback(async () => {
    if (!userId) return
    setDocumentsLoading(true)
    try {
      const res = await fetch(
        `${API_BASE}/documents?${new URLSearchParams({ user_id: userId })}`
      )
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const body = await res.json()
      setDocuments(Array.isArray(body.items) ? body.items : [])
    } catch (err) {
      console.warn('document list fetch failed', err)
      antdMessage.error('Failed to load documents')
    } finally {
      setDocumentsLoading(false)
    }
  }, [userId])

  useEffect(() => {
    refreshDocuments()
    const timer = setInterval(refreshDocuments, 5000)
    return () => clearInterval(timer)
  }, [refreshDocuments])

  const uploadProps = useMemo(
    () => ({
      multiple: false,
      customRequest: async ({ file, onSuccess, onError }) => {
        if (!userId) {
          onError(new Error('User ID is required'))
          return
        }
        const form = new FormData()
        form.append('user_id', userId)
        form.append('file', file)
        try {
          const res = await fetch(`${API_BASE}/documents`, {
            method: 'POST',
            body: form,
          })
          if (!res.ok) throw new Error(`HTTP ${res.status}`)
          antdMessage.success('Document upload started')
          onSuccess()
          refreshDocuments()
        } catch (err) {
          console.error('upload failed', err)
          onError(err)
          antdMessage.error('Upload failed')
        }
      },
      showUploadList: false,
    }),
    [userId, refreshDocuments]
  )

  const renderSources = (sources) => {
    if (!Array.isArray(sources) || sources.length === 0) return null
    return (
      <div
        style={{
          marginTop: 12,
          padding: 12,
          borderRadius: 10,
          background: '#0f172a',
          border: '1px solid #1f2937',
        }}
      >
        <Space direction="vertical" size={8} style={{ width: '100%' }}>
          <Text type="secondary">Sources</Text>
          {sources.map((source, index) => {
            const name =
              (source.metadata && source.metadata.filename) ||
              source.document_id ||
              `chunk-${index + 1}`
            const score =
              typeof source.score === 'number'
                ? ` (score ${source.score.toFixed(2)})`
                : ''
            return (
              <div
                key={`${source.chunk_id || source.document_id || index}`}
                style={{
                  background: '#0b1220',
                  borderRadius: 8,
                  padding: 10,
                  border: '1px solid #1f2937',
                }}
              >
                <Space size={8} align="center">
                  <Tag color="cyan">{name}</Tag>
                  <Tag color="geekblue">{score || 'retrieved'}</Tag>
                </Space>
                <Paragraph style={{ marginBottom: 0, color: '#cbd5f5' }}>
                  {source.content || '(empty snippet)'}
                </Paragraph>
              </div>
            )
          })}
        </Space>
      </div>
    )
  }

  const renderMessage = (item) => {
    const roleStyle = ROLE_CONFIG[item.role] || ROLE_CONFIG.system
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems:
            roleStyle.bubble.alignSelf === 'flex-end'
              ? 'flex-end'
              : 'flex-start',
          width: '100%',
        }}
      >
        <Space size={6} align="center">
          <Tag color={roleStyle.tag.color}>{roleStyle.tag.label}</Tag>
          {item.stepId && (
            <Tag color="cyan" style={{ fontWeight: 500 }}>
              {item.stepId}
            </Tag>
          )}
          {item.tool && <Tag color="blue">{item.tool}</Tag>}
          {item.status && <Tag color="gold">{item.status}</Tag>}
        </Space>
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
          {item.content ? (
            <Text style={{ color: roleStyle.bubble.color }}>
              {item.content}
            </Text>
          ) : (
            <Spin size="small" />
          )}
          {renderSources(item.sources)}
        </div>
      </div>
    )
  }

  const planBanner = useMemo(() => {
    if (!planSteps.length) return null
    const finishedSteps = planSteps.filter(
      (step) => step.status === 'completed'
    ).length
    const percent =
      planSteps.length === 0
        ? 0
        : Math.round((finishedSteps / planSteps.length) * 100)
    const activeStep = planSteps.find(
      (step) => step.id === planState.activeStepId
    )

    return (
      <div style={{ background: '#111827', padding: 12, borderRadius: 12 }}>
        <Space direction="vertical" style={{ width: '100%' }} size={8}>
          <Space
            align="center"
            style={{ width: '100%', justifyContent: 'space-between' }}
          >
            <Text style={{ color: '#f9fafb' }}>
              Workflow progress ({percent}%)
            </Text>
            <Tag color="blue">{status.toUpperCase()}</Tag>
          </Space>
          <Steps
            size="small"
            items={planSteps.map((step) => ({
              title: step.title,
              description: step.description,
              status: STEP_STATUS_MAP[step.status] || 'wait',
            }))}
          />
          {activeStep && (
            <Text type="secondary">
              Active step: {activeStep.title} ·{' '}
              {STEP_STATUS_TEXT[activeStep.status] || activeStep.status}
            </Text>
          )}
        </Space>
      </div>
    )
  }, [planSteps, planState.activeStepId, status])

  return (
    <Card
      style={{ background: '#0b1220', borderRadius: 16, border: '1px solid #1f2937' }}
      styles={{ body: { padding: 16 } }}
    >
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Space
          align="center"
          size="middle"
          style={{ width: '100%', flexWrap: 'wrap' }}
        >
          <Input
            prefix={<UserOutlined />}
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="User ID"
            style={{ width: 240, background: '#0f172a', color: '#e5e7eb' }}
          />
          <Tooltip title="Refresh documents">
            <Button
              icon={<DatabaseOutlined />}
              onClick={refreshDocuments}
              loading={documentsLoading}
            >
              Refresh Docs
            </Button>
          </Tooltip>
          <Tooltip title="Start a new session">
            <Button icon={<ReloadOutlined />} onClick={resetSession}>
              New Session
            </Button>
          </Tooltip>
          <Checkbox
            checked={rememberNext}
            onChange={(e) => setRememberNext(e.target.checked)}
          >
            Remember next answer
          </Checkbox>
        </Space>

        <Tabs
          defaultActiveKey="chat"
          items={[
            {
              key: 'chat',
              label: 'Chat',
              children: (
                <Space direction="vertical" size="large" style={{ width: '100%' }}>
                  {planBanner}
                  <List
                    dataSource={messages}
                    rowKey={(item) => item.id}
                    split={false}
                    renderItem={(item) => (
                      <List.Item style={{ padding: '12px 0' }}>
                        {renderMessage(item)}
                      </List.Item>
                    )}
                    style={{ maxHeight: 440, overflowY: 'auto' }}
                  />
                  <Space.Compact style={{ width: '100%' }}>
                    <Input.TextArea
                      autoSize={{ minRows: 1, maxRows: 4 }}
                      placeholder="Ask something, e.g. 'Summarize my latest document'"
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
                    <Tooltip title="Send">
                      <Button
                        type="primary"
                        icon={<SendOutlined />}
                        onClick={startStream}
                        loading={loading && !resumeLoading}
                      >
                        Send
                      </Button>
                    </Tooltip>
                    <Tooltip title="Stop current request">
                      <Button
                        icon={<StopOutlined />}
                        onClick={stopStream}
                        disabled={!loading && !resumeLoading}
                      >
                        Stop
                      </Button>
                    </Tooltip>
                  </Space.Compact>
                </Space>
              ),
            },
            {
              key: 'documents',
              label: 'Knowledge Base',
              children: (
                <Card
                  size="small"
                  title={
                    <Space align="center">
                      <CloudUploadOutlined />
                      <span>Knowledge Upload</span>
                    </Space>
                  }
                  style={{ background: '#0f172a', border: '1px solid #1f2937' }}
                >
                  <Space direction="vertical" size="large" style={{ width: '100%' }}>
                    <Dragger {...uploadProps} disabled={!userId}>
                      <p className="ant-upload-drag-icon">
                        <InboxOutlined />
                      </p>
                      <p className="ant-upload-text">
                        Click or drag a document to upload for retrieval
                      </p>
                      <p className="ant-upload-hint">
                        Documents are chunked, embedded, and scoped to your user ID.
                      </p>
                    </Dragger>

                    <Divider style={{ margin: '12px 0' }} />

                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text style={{ color: '#e5e7eb' }}>Uploaded documents</Text>
                      {documents.length === 0 ? (
                        <Empty
                          image={Empty.PRESENTED_IMAGE_SIMPLE}
                          description="No documents yet"
                        />
                      ) : (
                        <List
                          dataSource={documents}
                          rowKey={(item) => item.id}
                          renderItem={(item) => (
                            <List.Item
                              style={{
                                background: '#0b1220',
                                borderRadius: 8,
                                marginBottom: 8,
                                border: '1px solid #1f2937',
                              }}
                            >
                              <Space
                                direction="vertical"
                                size={4}
                                style={{ width: '100%' }}
                              >
                                <Space align="center">
                                  <Tag color="blue">{item.filename}</Tag>
                                  <Tag color="purple">
                                    {item.status?.toUpperCase()}
                                  </Tag>
                                  {item.error && (
                                    <Tag color="red">error: {item.error}</Tag>
                                  )}
                                </Space>
                                <Space
                                  align="center"
                                  style={{ justifyContent: 'space-between' }}
                                >
                                  <Text type="secondary">
                                    {new Date(item.updated_at * 1000 || Date.now()).toLocaleString()}
                                  </Text>
                                  {item.status === 'processing' && (
                                    <Progress
                                      percent={40}
                                      size="small"
                                      showInfo={false}
                                    />
                                  )}
                                </Space>
                              </Space>
                            </List.Item>
                          )}
                        />
                      )}
                    </Space>
                  </Space>
                </Card>
              ),
            },
          ]}
        />
      </Space>

      <Modal
        open={interruptVisible}
        title="Execution paused"
        onCancel={() => {
          setInterruptVisible(false)
          setInterruptInfo(null)
        }}
        footer={[
          <Button
            key="cancel"
            icon={<StopOutlined />}
            onClick={() => resume('cancel')}
            loading={resumeLoading}
          >
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
        <Space direction="vertical" size={12} style={{ width: '100%' }}>
          <Text>
            {interruptInfo?.message ||
              'The workflow requires confirmation. Continue?'}
          </Text>
          {interruptInfo?.options && (
            <Space>
              {interruptInfo.options.map((option) => (
                <Tag key={option}>{option}</Tag>
              ))}
            </Space>
          )}
        </Space>
      </Modal>
    </Card>
  )
}
