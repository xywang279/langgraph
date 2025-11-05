import React, { 
  useRef, 
  useState, 
  useCallback, 
  useEffect,
  useMemo,
  memo 
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
  Progress,
} from 'antd'
import {
  SendOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined,
  PlayCircleOutlined,
  StopOutlined,
} from '@ant-design/icons'
import { API_BASE } from '../api'

const { Text, Paragraph } = Typography

// 工具函数
const uid = () => Math.random().toString(36).slice(2)

// 常量配置
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
  system: {
    tag: { color: 'purple', label: 'SYSTEM' },
    bubble: {
      background: '#312e81',
      color: '#ede9fe',
      alignSelf: 'flex-start',
    },
  },
}

const STATUS_COLORS = {
  ok: 'green',
  degraded: 'orange',
  error: 'red',
  unknown: 'default',
}

const STEP_STATUS_MAP = {
  pending: 'wait',
  waiting: 'process',
  in_progress: 'process',
  completed: 'finish',
  cancelled: 'error',
  failed: 'error',
}

const STEP_STATUS_TEXT = {
  pending: 'Pending',
  waiting: 'Waiting for confirmation',
  in_progress: 'In progress',
  completed: 'Completed',
  cancelled: 'Cancelled',
  failed: 'Failed',
}

const STATUS_HINTS = {
  planning: 'Planning the workflow…',
  planning_completed: 'Plan ready. Executing steps.',
  planning_degraded: 'Plan generated (fallback template).',
  waiting: 'Awaiting your confirmation for the current step.',
  executing: 'Executing current step…',
  completed: 'All steps complete. You can start a new request.',
  cancelled: 'Workflow cancelled.',
  idle: 'Send a message to start.',
}

const INITIAL_PLAN = { steps: [], currentStep: 0, activeStepId: null }

// 格式化函数
const formatLatency = (latency) => {
  if (latency == null || Number.isNaN(Number(latency))) return null
  const value = Number(latency)
  return `${value.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')}s`
}

const stringifyFallback = (fallback) => {
  if (!fallback || typeof fallback !== 'object') return ''
  const tool = fallback.tool || 'fallback'
  const status = (fallback.status || 'unknown').toUpperCase()
  const obs = fallback.observation ? ` - ${fallback.observation}` : ''
  return `${tool} - ${status}${obs}`
}

const mapStepStatus = (status) => STEP_STATUS_MAP[status] || 'wait'

// 自定义Hook - 状态管理
const useChatState = () => {
  const [sessionId] = useState(() => `web-${uid()}`)
  const [text, setText] = useState('What time is it? Please use the tools.')
  const [items, setItems] = useState([])
  const [planState, setPlanState] = useState(INITIAL_PLAN)
  const [status, setStatus] = useState('idle')
  const [lastUpdate, setLastUpdate] = useState(null)
  const [loading, setLoading] = useState(false)
  const [resumeLoading, setResumeLoading] = useState(false)
  const [interruptInfo, setInterruptInfo] = useState(null)
  const [interruptVisible, setInterruptVisible] = useState(false)

  return {
    sessionId,
    text,
    setText,
    items,
    setItems,
    planState,
    setPlanState,
    status,
    setStatus,
    lastUpdate,
    setLastUpdate,
    loading,
    setLoading,
    resumeLoading,
    setResumeLoading,
    interruptInfo,
    setInterruptInfo,
    interruptVisible,
    setInterruptVisible,
  }
}

// 自定义Hook - 流处理
const useStreamHandler = ({ 
  setItems, 
  setPlanState, 
  setStatus, 
  setLoading, 
  setResumeLoading,
  setInterruptInfo,
  setInterruptVisible,
  setLastUpdate
}) => {
  const esRef = useRef(null)
  const lastResultRef = useRef({})

  const removeLivePlaceholder = useCallback(() => {
    setItems(prev => prev.filter(item => item.id !== 'ai-live'))
  }, [setItems])

  const appendItem = useCallback((item) => {
    setItems(prev => [...prev, { ...item, ts: Date.now() }])
  }, [setItems])

  const applyPlanEvent = useCallback((data) => {
    setPlanState({
      steps: Array.isArray(data.plan) ? data.plan : [],
      currentStep: typeof data.current_step === 'number' ? data.current_step : 0,
      activeStepId: data.active_step_id || null,
    })
    lastResultRef.current = {}
  }, [setPlanState])

  const applyStepUpdate = useCallback((payload) => {
    if (!payload || !payload.step_id) return
    setPlanState(prev => {
      const steps = prev.steps.map(step =>
        step.id === payload.step_id ? { ...step, ...payload } : step
      )
      return {
        ...prev,
        steps,
      }
    })
    
    setLastUpdate({
      stepId: payload.step_id,
      status: payload.status,
      result: payload.result,
      timestamp: Date.now(),
    })
  }, [setPlanState, setLastUpdate])

  const handleMessageChunk = useCallback((data) => {
    if (data.role === 'ai') {
      setItems(prev => {
        const withoutLive = prev.filter(item => item.id !== 'ai-live')
        const live = prev.find(item => item.id === 'ai-live') || {
          id: 'ai-live',
          role: 'ai',
          content: '',
        }
        return [...withoutLive, { ...live, content: `${live.content || ''}${data.delta || ''}` }]
      })
      return
    }

    if (data.role === 'tool') {
      appendItem({ id: uid(), role: 'tool', ...data })
      return
    }

    appendItem({ 
      id: uid(), 
      role: data.role || 'system', 
      content: data.content || '' 
    })
  }, [setItems, appendItem])

  const handleStreamEvent = useCallback((data) => {
    if (!data || typeof data !== 'object') return
    
    switch (data.event) {
      case 'end':
        setLoading(false)
        setResumeLoading(false)
        removeLivePlaceholder()
        break
      case 'error':
        setLoading(false)
        setResumeLoading(false)
        antdMessage.error(data.message || 'Execution error')
        removeLivePlaceholder()
        break
      case 'plan':
        applyPlanEvent(data)
        setStatus('planning_completed')
        break
      case 'step_update': {
        applyStepUpdate(data.payload)
        if (data.payload) {
          const stepId = data.payload.step_id || '__unknown__'
          const result =
            typeof data.payload.result === 'string'
              ? data.payload.result.trim()
              : ''
          if (result && lastResultRef.current[stepId] !== result) {
            lastResultRef.current[stepId] = result
            appendItem({
              id: uid(),
              role: 'ai',
              content: result,
              stepId,
            })
          }
        }
        break
      }
      case 'status':
        setStatus(data.status || 'idle')
        break
      case 'interrupt':
        setInterruptInfo(data.payload || null)
        setInterruptVisible(true)
        setLoading(false)
        setResumeLoading(false)
        removeLivePlaceholder()
        break
      case 'cancelled':
        if (Array.isArray(data.plan)) {
          setPlanState(prev => ({ ...prev, steps: data.plan }))
        }
        setStatus('cancelled')
        setResumeLoading(false)
        setInterruptVisible(false)
        removeLivePlaceholder()
        break
      default:
        if ('role' in data) {
          handleMessageChunk(data)
        } else if (data.payload?.plan && Array.isArray(data.payload.plan)) {
          setPlanState(prev => ({ ...prev, steps: data.payload.plan }))
        }
    }
  }, [
    setLoading,
    setResumeLoading,
    removeLivePlaceholder,
    applyPlanEvent,
    setStatus,
    applyStepUpdate,
    setInterruptInfo,
    setInterruptVisible,
    setPlanState,
    handleMessageChunk
  ])

  const attachEventSource = useCallback((es) => {
    es.onmessage = (ev) => {
      if (!ev?.data) return
      try {
        const parsed = JSON.parse(ev.data)
        handleStreamEvent(parsed)
      } catch (error) {
        console.warn('Failed to parse event', error)
      }
    }
    
    es.onerror = () => {
      setLoading(false)
      setResumeLoading(false)
      removeLivePlaceholder()
      es.close()
      esRef.current = null
    }
  }, [
    handleStreamEvent,
    setLoading,
    setResumeLoading,
    removeLivePlaceholder
  ])

  const initializeStream = useCallback(({ url, userMessage, resetPlan = false, expectStream = true }) => {
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

    if (userMessage) {
      const userItem = { id: uid(), role: 'user', content: userMessage }
      const aiPlaceholder = { id: 'ai-live', role: 'ai', content: '' }
      setItems(prev => {
        const withoutLive = prev.filter(item => item.id !== 'ai-live')
        return [...withoutLive, userItem, aiPlaceholder]
      })
    }

    const es = new EventSource(url)
    esRef.current = es
    attachEventSource(es)

    if (!expectStream) {
      setLoading(false)
    }
  }, [
    setPlanState,
    setStatus,
    setItems,
    attachEventSource,
    setLoading
  ])

  return {
    esRef,
    removeLivePlaceholder,
    appendItem,
    applyPlanEvent,
    applyStepUpdate,
    handleMessageChunk,
    handleStreamEvent,
    attachEventSource,
    initializeStream,
  }
}

// Memoized ToolItem component
const ToolItem = memo(({ item, resolveStepTitle }) => {
  const statusTag = (item.status || 'unknown').toLowerCase()
  const statusColor = STATUS_COLORS[statusTag] || 'default'
  const latency = formatLatency(item.latency)
  const fallbackText = stringifyFallback(item.fallback)

  return (
    <div style={{ width: '100%' }}>
      <Space size={6} wrap align="center" style={{ marginBottom: 6 }}>
        <Tag color={ROLE_STYLES.tool.tag.color}>{ROLE_STYLES.tool.tag.label}</Tag>
        {item.node && <Tag color="purple">{item.node}</Tag>}
        {item.step_id && <Tag color="cyan">{resolveStepTitle(item.step_id)}</Tag>}
        <Tag color="blue">{item.tool}</Tag>
        <Tag color={statusColor}>{statusTag.toUpperCase()}</Tag>
        {typeof item.tries === 'number' && item.tries > 1 && (
          <Tag color="magenta">tries {item.tries}</Tag>
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
          {item.observation || '(no tool output)'}
        </Paragraph>
        {item.error && (
          <Text type="danger" style={{ display: 'block', marginTop: 6 }}>
            Error: {item.error}
          </Text>
        )}
        {fallbackText && (
          <Text type="secondary" style={{ display: 'block', marginTop: 6 }}>
            Fallback: {fallbackText}
          </Text>
        )}
      </div>
    </div>
  )
})

// Memoized DefaultItem component
const DefaultItem = memo(({ item }) => {
  const roleStyle = ROLE_STYLES[item.role] || ROLE_STYLES.system
  const isStepResult = Boolean(item.fromStep)

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems:
          roleStyle.bubble.alignSelf === 'flex-end' ? 'flex-end' : 'flex-start',
        width: '100%',
      }}
    >
      <Space size={6} align="center">
        <Tag color={roleStyle.tag.color}>{roleStyle.tag.label}</Tag>
        {isStepResult && (
          <Tag color="cyan" style={{ fontWeight: 500 }}>
            STEP RESULT
          </Tag>
        )}
      </Space>
      <div
        style={{
          background: roleStyle.bubble.background,
          color: roleStyle.bubble.color,
          borderRadius: 10,
          border: '1px solid #ced9e8ff',
          padding: 12,
          minWidth: 120,
          maxWidth: '100%',
          whiteSpace: 'pre-wrap',
        }}
      >
        {item.stepId && isStepResult && (
          <Text type="secondary" style={{ display: 'block', marginBottom: 6 }}>
            {`Step: ${item.stepId}`}
          </Text>
        )}
        {item.content ? (
          <Text style={{ color: roleStyle.bubble.color }}>{item.content}</Text>
        ) : (
          <Spin size="small" />
        )}
      </div>
    </div>
  )
})

// Memoized PlanStatus component
const PlanStatus = memo(({ planState, status, lastUpdate, resolveStepTitle }) => {
  const steps = planState.steps
  const totalSteps = steps.length
  const finishedSteps = steps.filter(step => step.status === 'completed').length
  const percent = totalSteps === 0 ? 0 : Math.round((finishedSteps / totalSteps) * 100)
  const activeStep = steps.find(step => step.id === planState.activeStepId)

  if (steps.length === 0) return null

  return (
    <div style={{ background: '#111827', padding: 12, borderRadius: 12 }}>
      <Space direction="vertical" style={{ width: '100%' }} size={8}>
        <Space align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
          <Space size={6}>
            <Tag color="blue" icon={<PlayCircleOutlined />}>
              {status.toUpperCase()}
            </Tag>
            <Text style={{ color: '#93c5fd' }}>{STATUS_HINTS[status] || 'Workflow running'}</Text>
          </Space>
          <Progress
            strokeWidth={6}
            percent={percent}
            size="small"
            status={status === 'cancelled' ? 'exception' : percent === 100 ? 'success' : 'active'}
            style={{ minWidth: 140 ,color: '#e0f2fe' }}
          />
        </Space>

        <Steps
          size="small"
          current={Math.min(planState.currentStep, Math.max(totalSteps - 1, 0))}
          items={steps.map((step) => ({
            key: step.id,
            title: step.title,
            status: mapStepStatus(step.status),
            description: (
              <Tooltip placement="right" title={step.description || 'No additional details'}>
                <Text style={{ color: '#e8dab1ff' }}>
                  {STEP_STATUS_TEXT[step.status] || step.status}
                  {/* {STEP_STATUS_TEXT[step.result] ? ` · ${step.result}` : ''} */}
                </Text>
              </Tooltip>
            ),
          }))}
        />

        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <Text style={{ color: '#bfdbfe', fontWeight: 500 }}>
            {activeStep ? `Current step: ${activeStep.title}` : 'All steps completed'}
          </Text>
          {activeStep && Array.isArray(activeStep.tool_names) && activeStep.tool_names.length > 0 && (
            <Tooltip title={`Suggested tools: ${activeStep.tool_names.join(', ')}`}>
              <Text style={{ color: '#60a5fa', cursor: 'pointer' }}>Suggested tools</Text>
            </Tooltip>
          )}
        </Space>

        {activeStep && activeStep.description && (
          <Text type="secondary" ellipsis={{ rows: 2 }}>
            {activeStep.description}
          </Text>
        )}

        {lastUpdate && (
          <Text type="secondary" style={{ fontSize: 12,color: '#f1f2f4ff' }}>
            {`Last update: ${resolveStepTitle(lastUpdate.stepId)} -> ${
              STEP_STATUS_TEXT[lastUpdate.status] || lastUpdate.status
            }${lastUpdate.result ? `, result: ${lastUpdate.result}` : ''}`}
          </Text>
        )}
      </Space>
    </div>
  )
})

export default function Chat() {
  const {
    sessionId,
    text,
    setText,
    items,
    setItems,
    planState,
    setPlanState,
    status,
    setStatus,
    lastUpdate,
    setLastUpdate,
    loading,
    setLoading,
    resumeLoading,
    setResumeLoading,
    interruptInfo,
    setInterruptInfo,
    interruptVisible,
    setInterruptVisible,
  } = useChatState()

  const {
    esRef,
    removeLivePlaceholder,
    initializeStream,
  } = useStreamHandler({
    setItems,
    setPlanState,
    setStatus,
    setLoading,
    setResumeLoading,
    setInterruptInfo,
    setInterruptVisible,
    setLastUpdate,
  })

  const startStream = useCallback(() => {
    if (!text.trim()) {
      antdMessage.warning('Please enter a prompt')
      return
    }

    setLoading(true)
    setResumeLoading(false)
    setInterruptVisible(false)
    setInterruptInfo(null)

    const url = `${API_BASE}/chat/stream?session_id=${encodeURIComponent(sessionId)}&message=${encodeURIComponent(text)}`
    initializeStream({ url, userMessage: text, resetPlan: true })
    setText('')
  }, [
    text,
    sessionId,
    setLoading,
    setResumeLoading,
    setInterruptVisible,
    setInterruptInfo,
    initializeStream,
    setText
  ])

  const resume = useCallback((action) => {
    if (!sessionId) return
    const url = `${API_BASE}/chat/continue?thread_id=${encodeURIComponent(sessionId)}&action=${action}&nonce=${Date.now()}`
    console.debug('[resume] sending continue request', url)
    setInterruptVisible(false)
    setInterruptInfo(null)

    if (action === 'continue') {
      setLoading(true)
      setResumeLoading(true)
      const live = { id: 'ai-live', role: 'ai', content: '' }
      setItems(prev => {
        const withoutLive = prev.filter(item => item.id !== 'ai-live')
        return [...withoutLive, live]
      })
      initializeStream({ url, resetPlan: false, expectStream: true })
    } else {
      setResumeLoading(true)
      initializeStream({ url, resetPlan: false, expectStream: false })
    }
  }, [
    sessionId,
    setLoading,
    setResumeLoading,
    setItems,
    setInterruptVisible,
    setInterruptInfo,
    initializeStream
  ])

  const stopStream = useCallback(() => {
    if (esRef.current) {
      esRef.current.close()
      esRef.current = null
      setLoading(false)
      setResumeLoading(false)
      removeLivePlaceholder()
    }
  }, [esRef, setLoading, setResumeLoading, removeLivePlaceholder])

  const resolveStepTitle = useCallback((stepId) => {
    const target = planState.steps.find(step => step.id === stepId)
    return target ? target.title : stepId
  }, [planState.steps])

  // Memoize renderItem function
  const renderItem = useMemo(() => (item) => {
    if (item.role === 'tool') {
      return <ToolItem item={item} resolveStepTitle={resolveStepTitle} />
    }
    return <DefaultItem item={item} />
  }, [resolveStepTitle])

  // Component cleanup
  useEffect(() => {
    return () => {
      if (esRef.current) {
        esRef.current.close()
      }
    }
  }, [esRef])

  return (
    <Card
      style={{ background: '#0b1220', borderRadius: 16, border: '1px solid #1f2937' }}
      styles={{ body: { padding: 16 } }}
    >
      <Space size="middle" direction="vertical" style={{ width: '100%' }}>
        <Space.Compact style={{ width: '100%' }}>
          <Input.TextArea
            autoSize={{ minRows: 1, maxRows: 4 }}
            placeholder="Ask something, e.g. 'What time is it? Please use the tools.'"
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
          <Tooltip title="Send" placement="top">
            <Button type="primary" icon={<SendOutlined />} onClick={startStream}>
              Send
            </Button>
          </Tooltip>
          <Tooltip title="Stop current request">
            <Button 
              icon={<ThunderboltOutlined />} 
              onClick={stopStream} 
              disabled={!loading && !resumeLoading}
            >
              Stop
            </Button>
          </Tooltip>
        </Space.Compact>

        <PlanStatus 
          planState={planState} 
          status={status} 
          lastUpdate={lastUpdate}
          resolveStepTitle={resolveStepTitle}
        />

        <List
          dataSource={items}
          rowKey={(item) => item.id}
          split={false}
          renderItem={renderItem}
        />
      </Space>

      <Modal
        open={interruptVisible}
        title="Needs confirmation"
        onCancel={() => {
          setInterruptVisible(false)
          setInterruptInfo(null)
        }}
        footer={null}
        centered
      >
        <Space direction="vertical" style={{ width: '100%' }} size={16}>
          <Text>{interruptInfo?.message || 'Execution paused. Continue?'}</Text>
          <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
            <Button 
              icon={<StopOutlined />} 
              onClick={() => resume('cancel')} 
              loading={resumeLoading}
            >
              Cancel
            </Button>
            <Button 
              type="primary" 
              onClick={() => resume('continue')} 
              loading={resumeLoading}
            >
              Continue
            </Button>
          </Space>
        </Space>
      </Modal>
    </Card>
  )
}
