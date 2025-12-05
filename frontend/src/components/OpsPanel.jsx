import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { Card, Flex, Alert, Form, InputNumber, Button, Typography, message as antdMessage, theme as antdTheme, Table, Tag } from 'antd'
import { SaveOutlined, ReloadOutlined, LineChartOutlined } from '@ant-design/icons'
import { API_BASE, API_KEY, fetchToolBudgetConfig, updateToolBudgetConfig, fetchIngestionJobs, fetchVectorHealth } from '../api'

const { Paragraph } = Typography

export default function OpsPanel({ apiToken, userId, tenantId }) {
  const { token: antdToken } = antdTheme.useToken()
  const resolvedToken = useMemo(() => apiToken || API_KEY || '', [apiToken])

  const [toolBudget, setToolBudget] = useState(null)
  const [toolBudgetDraft, setToolBudgetDraft] = useState({
    max_tasks: 6,
    max_parallel: 3,
    total_latency: 12,
  })
  const [toolBudgetLoading, setToolBudgetLoading] = useState(false)
  const [toolBudgetSaving, setToolBudgetSaving] = useState(false)
  const [jobs, setJobs] = useState([])
  const [jobsLoading, setJobsLoading] = useState(false)
  const [health, setHealth] = useState([])
  const [healthLoading, setHealthLoading] = useState(false)

  const refreshToolBudget = useCallback(async () => {
    setToolBudgetLoading(true)
    try {
      const data = await fetchToolBudgetConfig(resolvedToken)
      setToolBudget(data)
      setToolBudgetDraft(data)
      antdMessage.success('Tool budget loaded')
    } catch (err) {
      antdMessage.error('Failed to load tool budget settings')
    } finally {
      setToolBudgetLoading(false)
    }
  }, [resolvedToken])

  useEffect(() => {
    refreshToolBudget()
  }, [refreshToolBudget])

  const refreshJobs = useCallback(async () => {
    if (!userId || !tenantId) return
    setJobsLoading(true)
    try {
      const data = await fetchIngestionJobs(userId, tenantId, resolvedToken, 50)
      setJobs(Array.isArray(data.items) ? data.items : [])
    } catch (err) {
      antdMessage.error(err?.message || 'Failed to load ingestion jobs')
    } finally {
      setJobsLoading(false)
    }
  }, [userId, tenantId, resolvedToken])

  const refreshHealth = useCallback(async () => {
    if (!userId || !tenantId) return
    setHealthLoading(true)
    try {
      const data = await fetchVectorHealth(userId, tenantId, resolvedToken)
      setHealth(Array.isArray(data.items) ? data.items : [])
    } catch (err) {
      antdMessage.error(err?.message || 'Failed to load vector health')
    } finally {
      setHealthLoading(false)
    }
  }, [userId, tenantId, resolvedToken])

  useEffect(() => {
    refreshJobs()
    refreshHealth()
  }, [refreshJobs, refreshHealth])

  const handleBudgetFieldChange = useCallback((field, value) => {
    setToolBudgetDraft((prev) => ({
      ...prev,
      [field]: typeof value === 'number' ? value : prev[field],
    }))
  }, [])

  const budgetChanged = useMemo(() => {
    if (!toolBudget) return false
    return (
      toolBudget.max_tasks !== toolBudgetDraft.max_tasks ||
      toolBudget.max_parallel !== toolBudgetDraft.max_parallel ||
      toolBudget.total_latency !== toolBudgetDraft.total_latency
    )
  }, [toolBudget, toolBudgetDraft])

  const handleBudgetSave = useCallback(async () => {
    if (!toolBudget) return
    const payload = {}
    if (toolBudgetDraft.max_tasks !== toolBudget.max_tasks) {
      payload.max_tasks = toolBudgetDraft.max_tasks
    }
    if (toolBudgetDraft.max_parallel !== toolBudget.max_parallel) {
      payload.max_parallel = toolBudgetDraft.max_parallel
    }
    if (toolBudgetDraft.total_latency !== toolBudget.total_latency) {
      payload.total_latency = toolBudgetDraft.total_latency
    }
    if (Object.keys(payload).length === 0) {
      antdMessage.info('No budget changes to apply')
      return
    }
    setToolBudgetSaving(true)
    try {
      const next = await updateToolBudgetConfig(payload, resolvedToken)
      setToolBudget(next)
      setToolBudgetDraft(next)
      antdMessage.success('Tool budget updated')
    } catch (err) {
      antdMessage.error(err?.message || 'Failed to update tool budget')
    } finally {
      setToolBudgetSaving(false)
    }
  }, [toolBudget, toolBudgetDraft, resolvedToken])

  const handleBudgetReset = useCallback(() => {
    if (toolBudget) {
      setToolBudgetDraft(toolBudget)
    }
  }, [toolBudget])

  const openMetricsFeed = useCallback(() => {
    const params = resolvedToken ? `?api_key=${encodeURIComponent(resolvedToken)}` : ''
    const url = `${API_BASE}/metrics${params}`
    window.open(url, '_blank', 'noopener,noreferrer')
  }, [resolvedToken])

  return (
    <Card
      size="small"
      style={{
        background: antdToken.colorBgContainer,
        border: '1px solid ' + antdToken.colorBorder,
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Flex vertical gap="large">
        <Alert
          type="info"
          showIcon
          message="Guardrail budget"
          description="Tune max tool calls, concurrency, and latency budget without redeploying. Metrics are exported for Grafana/Prometheus via /metrics."
        />
        <Form
          layout="vertical"
          requiredMark={false}
          style={{
            background: antdToken.colorFillTertiary,
            padding: 16,
            borderRadius: 12,
            border: '1px solid ' + antdToken.colorBorder,
          }}
        >
          <Form.Item label="Max tool calls per request">
            <InputNumber
              min={1}
              max={32}
              value={toolBudgetDraft.max_tasks}
              onChange={(value) => handleBudgetFieldChange('max_tasks', value)}
              disabled={toolBudgetLoading}
            />
          </Form.Item>
          <Form.Item label="Max parallel (non-exclusive) tools">
            <InputNumber
              min={1}
              max={10}
              value={toolBudgetDraft.max_parallel}
              onChange={(value) => handleBudgetFieldChange('max_parallel', value)}
              disabled={toolBudgetLoading}
            />
          </Form.Item>
          <Form.Item label="Total latency budget (seconds)">
            <InputNumber
              min={1}
              max={120}
              step={0.5}
              value={toolBudgetDraft.total_latency}
              onChange={(value) => handleBudgetFieldChange('total_latency', value)}
              disabled={toolBudgetLoading}
            />
          </Form.Item>
        </Form>
        <Flex wrap gap="small">
          <Button
            type="primary"
            icon={<SaveOutlined />}
            onClick={handleBudgetSave}
            disabled={!budgetChanged || toolBudgetLoading}
            loading={toolBudgetSaving}
          >
            Apply changes
          </Button>
          <Button onClick={handleBudgetReset} disabled={!toolBudgetDraft || toolBudgetLoading}>
            Reset form
          </Button>
          <Button icon={<ReloadOutlined />} onClick={refreshToolBudget} loading={toolBudgetLoading}>
            Refresh from server
          </Button>
          <Button icon={<LineChartOutlined />} onClick={openMetricsFeed}>
            Open metrics feed
          </Button>
        </Flex>
        <Paragraph type="secondary" style={{ marginBottom: 0 }}>
          Metrics endpoint: <code style={{ color: '#f472b6' }}>{`${API_BASE}/metrics`}</code>.
        </Paragraph>

        <Card
          title="Ingestion Jobs"
          size="small"
          extra={
            <Button size="small" icon={<ReloadOutlined />} onClick={refreshJobs} loading={jobsLoading}>
              刷新
            </Button>
          }
        >
          <Table
            size="small"
            loading={jobsLoading}
            dataSource={jobs}
            rowKey={(row) => row.id}
            pagination={false}
            columns={[
              { title: 'Job ID', dataIndex: 'id', ellipsis: true },
              { title: 'Source', dataIndex: 'source' },
              {
                title: 'Progress',
                render: (_, row) => `${row.processed ?? 0}/${row.total ?? 0}`,
              },
              {
                title: 'Status',
                dataIndex: 'status',
                render: (v) => <Tag color={v === 'failed' ? 'red' : v === 'completed' ? 'green' : 'blue'}>{(v || '').toUpperCase()}</Tag>,
              },
              { title: 'Updated', dataIndex: 'updated_at', render: (t) => (t ? new Date(t * 1000).toLocaleString() : '--') },
            ]}
            locale={{ emptyText: 'No jobs' }}
          />
        </Card>

        <Card
          title="Vector Health"
          size="small"
          extra={
            <Button size="small" icon={<ReloadOutlined />} onClick={refreshHealth} loading={healthLoading}>
              刷新
            </Button>
          }
        >
          <Table
            size="small"
            loading={healthLoading}
            dataSource={health}
            rowKey={(row) => row.document_id + (row.version_id || '')}
            pagination={false}
            columns={[
              { title: 'Doc ID', dataIndex: 'document_id', ellipsis: true },
              { title: 'Version', dataIndex: 'version_id', ellipsis: true },
              { title: 'Expected', dataIndex: 'expected_vectors' },
              { title: 'Stored', dataIndex: 'stored_vectors' },
              {
                title: 'Health',
                dataIndex: 'health_status',
                render: (v) => <Tag color={v === 'healthy' ? 'green' : v === 'degraded' ? 'orange' : 'gray'}>{v || 'unknown'}</Tag>,
              },
              { title: 'Notes', dataIndex: 'notes', ellipsis: true },
            ]}
            locale={{ emptyText: 'No health data' }}
          />
        </Card>
      </Flex>
    </Card>
  )
}
