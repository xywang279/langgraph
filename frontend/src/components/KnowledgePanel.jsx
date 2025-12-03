import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Card,
  Flex,
  Divider,
  Upload,
  Empty,
  List,
  Tag,
  Progress,
  Alert,
  Typography,
  Spin,
  Button,
  message as antdMessage,
  Switch,
  Input,
  Select,
  Checkbox,
  Dropdown,
  theme as antdTheme,
} from 'antd'
import {
  CloudUploadOutlined,
  InboxOutlined,
  ReloadOutlined,
  CopyOutlined,
  CloseOutlined,
  FilePdfOutlined,
  FileWordOutlined,
  FileImageOutlined,
  FileTextOutlined,
  SearchOutlined,
  MoreOutlined,
} from '@ant-design/icons'
import { API_BASE, API_KEY, buildAuthHeaders } from '../api'

const { Dragger } = Upload
const { Text } = Typography

export default function KnowledgePanel({ userId, apiToken }) {
  const { token: antdToken } = antdTheme.useToken()
  const resolvedToken = useMemo(() => apiToken || API_KEY || '', [apiToken])
  const authHeaders = useMemo(() => buildAuthHeaders(resolvedToken), [resolvedToken])

  const [documents, setDocuments] = useState([])
  const [documentsLoading, setDocumentsLoading] = useState(false)
  const [documentsError, setDocumentsError] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [actionLoadingId, setActionLoadingId] = useState(null)
  const [downloadingId, setDownloadingId] = useState(null)
  const [selectedIds, setSelectedIds] = useState(new Set())
  const [filterStatus, setFilterStatus] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')
  const [lastRefreshed, setLastRefreshed] = useState(null)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [selectedDocId, setSelectedDocId] = useState(null)

  const formatBytes = (value) => {
    if (value === null || value === undefined) return null
    const numeric = typeof value === 'number' ? value : Number(value)
    if (!Number.isFinite(numeric) || numeric < 0) return null
    if (numeric < 1024) return `${numeric} B`
    const units = ['KB', 'MB', 'GB', 'TB']
    const exponent = Math.min(units.length - 1, Math.floor(Math.log(numeric) / Math.log(1024)))
    const num = numeric / 1024 ** (exponent + 1)
    return `${num.toFixed(num >= 10 ? 0 : 1)} ${units[exponent]}`
  }

  const toNumber = (value) => {
    if (typeof value === 'number' && Number.isFinite(value)) return value
    if (typeof value === 'string' && value.trim() !== '') {
      const parsed = Number(value)
      if (Number.isFinite(parsed)) return parsed
    }
    return null
  }

  const renderFileIcon = (mime = '') => {
    const lower = (mime || '').toLowerCase()
    if (lower.includes('pdf')) return <FilePdfOutlined style={{ color: '#e11d48' }} />
    if (lower.includes('word') || lower.includes('doc')) return <FileWordOutlined style={{ color: '#2563eb' }} />
    if (lower.includes('image') || lower.includes('png') || lower.includes('jpg')) return <FileImageOutlined style={{ color: '#22c55e' }} />
    return <FileTextOutlined style={{ color: '#6366f1' }} />
  }

  const buildUrl = (path) => {
    const search = new URLSearchParams({ user_id: userId })
    if (resolvedToken) {
      search.set('api_key', resolvedToken)
    }
    return `${API_BASE}${path}?${search.toString()}`
  }

  const refreshDocuments = useCallback(async () => {
    if (!userId) return
    setDocumentsLoading(true)
    setDocumentsError(null)
    try {
      const res = await fetch(`${API_BASE}/documents?${new URLSearchParams({ user_id: userId })}`, {
        headers: { ...authHeaders },
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const body = await res.json()
      setDocuments(Array.isArray(body.items) ? body.items : [])
      setLastRefreshed(Date.now())
    } catch (err) {
      const message = err?.message || 'Failed to load documents'
      setDocumentsError(message)
      antdMessage.error(message)
    } finally {
      setDocumentsLoading(false)
    }
  }, [userId, authHeaders])

  const hasProcessingDocs = useMemo(
    () => documents.some((item) => item.status === 'processing'),
    [documents]
  )

  useEffect(() => {
    if (!userId) return
    refreshDocuments()
  }, [userId, refreshDocuments])

  useEffect(() => {
    if (!userId) return
    const shouldPoll = autoRefresh && (hasProcessingDocs || uploading)
    if (!shouldPoll) return
    const timer = setInterval(refreshDocuments, 5000)
    return () => clearInterval(timer)
  }, [userId, hasProcessingDocs, uploading, refreshDocuments, autoRefresh])

  const uploadProps = useMemo(
    () => ({
      multiple: false,
      customRequest: async ({ file, onSuccess, onError }) => {
        if (!userId) {
          onError(new Error('User ID is required'))
          return
        }
        setUploading(true)
        const form = new FormData()
        form.append('user_id', userId)
        form.append('file', file)
        try {
          const res = await fetch(`${API_BASE}/documents`, {
            method: 'POST',
            headers: { ...authHeaders },
            body: form,
          })
          if (!res.ok) throw new Error(`HTTP ${res.status}`)
          antdMessage.success('Document upload started')
          onSuccess()
          refreshDocuments()
        } catch (err) {
          onError(err)
          antdMessage.error(err?.message || 'Upload failed')
        } finally {
          setUploading(false)
        }
      },
      showUploadList: false,
    }),
    [userId, authHeaders, refreshDocuments]
  )

  const toggleSelect = (id) => {
    setSelectedIds((prev) => {
      const next = new Set(prev)
      if (next.has(id)) {
        next.delete(id)
      } else {
        next.add(id)
      }
      return next
    })
    setSelectedDocId(id)
  }

  const filteredDocuments = useMemo(() => {
    const term = searchTerm.trim().toLowerCase()
    return documents.filter((doc) => {
      const matchStatus = filterStatus === 'all' ? true : doc.status === filterStatus
      const matchTerm = term
        ? (doc.filename || '').toLowerCase().includes(term) || (doc.id || '').toLowerCase().includes(term)
        : true
      return matchStatus && matchTerm
    })
  }, [documents, filterStatus, searchTerm])

  const selectedDoc = useMemo(
    () => documents.find((d) => (d.id || d.document_id || d.pk) === selectedDocId) || null,
    [documents, selectedDocId]
  )

  const handleRetry = async (docId) => {
    if (!userId) {
      antdMessage.warning('User ID is required')
      return
    }
    setActionLoadingId(docId)
    try {
      const res = await fetch(buildUrl(`/documents/${docId}/retry`), {
        method: 'POST',
        headers: { ...authHeaders },
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      antdMessage.success('Retry queued')
      refreshDocuments()
    } catch (err) {
      antdMessage.error(err?.message || 'Retry failed')
    } finally {
      setActionLoadingId(null)
    }
  }

  const handleDelete = async (docId) => {
    if (!userId) {
      antdMessage.warning('User ID is required')
      return
    }
    setActionLoadingId(docId)
    try {
      const res = await fetch(buildUrl(`/documents/${docId}`), {
        method: 'DELETE',
        headers: { ...authHeaders },
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      antdMessage.success('Document deleted')
      refreshDocuments()
      if (selectedDocId === docId) setSelectedDocId(null)
    } catch (err) {
      antdMessage.error(err?.message || 'Delete failed')
    } finally {
      setActionLoadingId(null)
    }
  }

  const handleDownload = async (doc) => {
    if (!userId) {
      antdMessage.warning('User ID is required')
      return
    }
    const docId = doc.id || doc.document_id || doc.pk
    if (!docId) return
    setDownloadingId(docId)
    try {
      const res = await fetch(buildUrl(`/documents/${docId}/download`), {
        headers: { ...authHeaders },
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const blob = await res.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = doc.filename || 'document'
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      antdMessage.error(err?.message || 'Download failed')
    } finally {
      setDownloadingId(null)
    }
  }

  const handleBulkAction = async (action) => {
    if (selectedIds.size === 0) return
    const ids = Array.from(selectedIds)
    for (const id of ids) {
      if (action === 'delete') {
        await handleDelete(id)
      } else if (action === 'retry') {
        await handleRetry(id)
      }
    }
    setSelectedIds(new Set())
  }

  const totalCount = documents.length
  const readyCount = documents.filter((d) => d.status === 'ready').length
  const processingCount = documents.filter((d) => d.status === 'processing').length

  const detailCard = selectedDoc ? (
    <Card
      style={{ height: '100%', borderRadius: 16, background: '#ffffff', boxShadow: '0 18px 40px rgba(15, 23, 42, 0.08)' }}
      bodyStyle={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 16 }}
    >
      <Flex align="center" justify="space-between">
        <Flex align="center" gap={12}>
          <div
            style={{
              width: 48,
              height: 48,
              borderRadius: 12,
              background: antdToken.colorPrimaryBg,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {renderFileIcon(selectedDoc.mime_type || selectedDoc.content_type || selectedDoc.type)}
          </div>
          <Flex vertical>
            <Text strong style={{ fontSize: 18 }}>{selectedDoc.filename}</Text>
            <Flex gap={8} wrap>
              <Tag color="default">{selectedDoc.id}</Tag>
              <Tag color={selectedDoc.status === 'ready' ? 'green' : selectedDoc.status === 'failed' ? 'red' : 'blue'}>
                {selectedDoc.status?.toUpperCase()}
              </Tag>
            </Flex>
          </Flex>
        </Flex>
        <Button icon={<CloseOutlined />} type="text" onClick={() => setSelectedDocId(null)} />
      </Flex>

      <Card style={{ borderRadius: 12, background: '#f8fafc', border: '1px solid #e5e7eb' }} bodyStyle={{ padding: 16 }}>
        <Text strong style={{ fontSize: 16 }}>Document Metadata</Text>
        <Divider style={{ margin: '8px 0' }} />
        <Flex vertical gap={8}>
          <Flex justify="space-between">
            <Text type="secondary">File Size:</Text>
            <Text>{formatBytes(selectedDoc.size_bytes ?? selectedDoc.size) || '–'}</Text>
          </Flex>
          <Flex justify="space-between">
            <Text type="secondary">Content Type:</Text>
            <Text>{selectedDoc.mime_type || selectedDoc.content_type || selectedDoc.type || '–'}</Text>
          </Flex>
          <Flex justify="space-between">
            <Text type="secondary">Uploaded At:</Text>
            <Text>{new Date(selectedDoc.updated_at * 1000 || Date.now()).toLocaleString()}</Text>
          </Flex>
          <Flex justify="space-between">
            <Text type="secondary">Chunks Generated:</Text>
            <Text>{selectedDoc.chunk_count ?? selectedDoc.num_chunks ?? '–'}</Text>
          </Flex>
          <Flex justify="space-between">
            <Text type="secondary">Embeddings:</Text>
            <Text>{selectedDoc.embedding_count ?? selectedDoc.num_embeddings ?? '–'}</Text>
          </Flex>
        </Flex>
      </Card>

      <Card style={{ borderRadius: 12, background: '#f8fafc', flex: 1 }} bodyStyle={{ padding: 16, height: '100%' }}>
        <Text strong style={{ fontSize: 16 }}>Preview</Text>
        <Divider style={{ margin: '8px 0' }} />
        <div
          style={{
            flex: 1,
            border: '1px dashed #e2e8f0',
            borderRadius: 12,
            background: '#f8fafc',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: antdToken.colorTextSecondary,
            minHeight: 200,
          }}
        >
          <div style={{ textAlign: 'center' }}>
            <InboxOutlined style={{ fontSize: 32 }} />
            <div style={{ marginTop: 8 }}>Preview not available. Download to view.</div>
          </div>
        </div>
      </Card>
    </Card>
  ) : (
    <Card
      style={{
        height: '100%',
        borderRadius: 16,
        background: '#fff',
        boxShadow: antdToken.boxShadowTertiary,
      }}
      bodyStyle={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 16 }}
    >
      <Flex vertical align="center" gap={8} style={{ textAlign: 'center' }}>
        <div
          style={{
            width: 72,
            height: 72,
            borderRadius: 20,
            background: antdToken.colorPrimaryBg,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <CloudUploadOutlined style={{ fontSize: 32, color: antdToken.colorPrimary }} />
        </div>
        <Text style={{ fontSize: 22, fontWeight: 700 }}>Upload Knowledge</Text>
        <Text type="secondary">
          Drag and drop files here. We support PDF, Markdown, TXT, and Images.
        </Text>
      </Flex>

      <Dragger {...uploadProps} disabled={!userId} style={{ background: '#f9fafb', border: '1px dashed #e2e8f0' }}>
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">Click or drag file to this area to upload</p>
        <p className="ant-upload-hint">Support for single or bulk upload.</p>
      </Dragger>

      {uploading && (
        <Alert type="info" showIcon message="Uploading..." description="Your file is being uploaded and processed." />
      )}

      {documentsError && (
        <Alert
          type="error"
          showIcon
          message="Failed to load documents"
          description={documentsError}
          action={
            <Button type="primary" icon={<ReloadOutlined />} size="small" onClick={refreshDocuments}>
              Retry
            </Button>
          }
        />
      )}

      <Flex gap={12} style={{ marginTop: 'auto' }}>
        <Card
          style={{ flex: 1, textAlign: 'center', borderRadius: 12, border: '1px solid #e5e7eb' }}
          bodyStyle={{ padding: 12 }}
        >
          <Text type="secondary">Total Documents</Text>
          <div style={{ fontSize: 22, fontWeight: 700 }}>{totalCount}</div>
        </Card>
        <Card
          style={{ flex: 1, textAlign: 'center', borderRadius: 12, border: '1px solid #e5e7eb' }}
          bodyStyle={{ padding: 12 }}
        >
          <Text type="secondary">Ready</Text>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#16a34a' }}>{readyCount}</div>
        </Card>
        <Card
          style={{ flex: 1, textAlign: 'center', borderRadius: 12, border: '1px solid #e5e7eb' }}
          bodyStyle={{ padding: 12 }}
        >
          <Text type="secondary">Processing</Text>
          <div style={{ fontSize: 22, fontWeight: 700, color: '#2563eb' }}>{processingCount}</div>
        </Card>
      </Flex>
    </Card>
  )

  return (
    <Flex gap={16} style={{ height: '100%', width: '100%' }}>
      {/* Left column: list & filters */}
      <div
        style={{
          width: 400,
          background: '#fdfefe',
          border: '1px solid #e5e7eb',
          borderRadius: 12,
          padding: 16,
          display: 'flex',
          flexDirection: 'column',
          gap: 12,
        }}
      >
        <Flex align="center" justify="space-between">
          <Text strong style={{ fontSize: 18 }}>Documents</Text>
          <Button size="small" icon={<ReloadOutlined />} onClick={refreshDocuments} loading={documentsLoading} />
        </Flex>
        <Input
          placeholder="Search documents..."
          allowClear
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          prefix={<SearchOutlined />}
        />
        <Select
          value={filterStatus}
          onChange={setFilterStatus}
          options={[
            { value: 'all', label: 'All Status' },
            { value: 'processing', label: 'Processing' },
            { value: 'ready', label: 'Ready' },
            { value: 'failed', label: 'Failed' },
          ]}
        />
        <Flex align="center" justify="space-between">
          <Text type="secondary" style={{ fontSize: 12 }}>
            {filteredDocuments.length} / {documents.length} items
            {lastRefreshed ? ` • Updated ${new Date(lastRefreshed).toLocaleTimeString()}` : ''}
          </Text>
          <Switch size="small" checked={autoRefresh} onChange={setAutoRefresh} checkedChildren="Auto" unCheckedChildren="Manual" />
        </Flex>
        <div style={{ flex: 1, overflowY: 'auto' }}>
          <List
            dataSource={filteredDocuments}
            rowKey={(item) => item.id}
            renderItem={(item) => {
              const sizeText = formatBytes(item.size_bytes ?? item.size)
              const mimeText = item.mime_type || item.content_type || item.type
              const docId = item.id || item.document_id || item.pk
              const isSelected = selectedIds.has(docId) || selectedDocId === docId
              const actionItems = [
                { key: 'download', label: 'Download' },
                { key: 'retry', label: 'Retry' },
                { key: 'delete', label: 'Delete' },
              ]
              return (
                <List.Item
                  style={{
                    padding: 12,
                    borderRadius: 12,
                    marginBottom: 10,
                    background: isSelected ? '#eef4ff' : '#ffffff',
                    border: isSelected ? '1px solid #d0d7ff' : '1px solid #e5e7eb',
                    cursor: 'pointer',
                    boxShadow: isSelected ? '0 8px 20px rgba(15, 23, 42, 0.08)' : 'none',
                  }}
                  onClick={() => toggleSelect(docId)}
                >
                  <Flex align="center" gap={12} style={{ width: '100%' }}>
                    <div
                      style={{
                        width: 40,
                        height: 40,
                        borderRadius: 12,
                        background: antdToken.colorPrimaryBg,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                      }}
                    >
                      {renderFileIcon(mimeText)}
                    </div>
                    <Flex vertical style={{ flex: 1, minWidth: 0 }}>
                      <Text strong ellipsis={{ rows: 2, tooltip: item.filename }}>
                        {item.filename}
                      </Text>
                      <Flex gap={8} wrap align="center">
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {sizeText || '--'}
                        </Text>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {new Date(item.updated_at * 1000 || Date.now()).toLocaleDateString()}
                        </Text>
                        <Tag color={item.status === 'failed' ? 'red' : item.status === 'processing' ? 'blue' : 'green'}>
                          {item.status?.toUpperCase()}
                        </Tag>
                      </Flex>
                    </Flex>
                    <Dropdown
                      menu={{
                        items: actionItems,
                        onClick: ({ key }) => {
                          if (key === 'download') {
                            handleDownload(item)
                          } else if (key === 'retry') {
                            handleRetry(docId)
                          } else if (key === 'delete') {
                            handleDelete(docId)
                          }
                        },
                      }}
                      trigger={['click']}
                    >
                      <Button
                        type="text"
                        shape="circle"
                        icon={<MoreOutlined />}
                        onClick={(e) => e.stopPropagation()}
                        style={{ width: 32, height: 32 }}
                      />
                    </Dropdown>
                  </Flex>
                </List.Item>
              )
            }}
          />
        </div>
        {selectedIds.size > 0 && (
          <Flex gap={8}>
            <Button size="small" onClick={() => handleBulkAction('retry')} block>
              Retry
            </Button>
            <Button size="small" danger onClick={() => handleBulkAction('delete')} block>
              Delete
            </Button>
          </Flex>
        )}
      </div>

      {/* Right column */}
      <div style={{ flex: 1, background: '#ffffff', borderRadius: 14, padding: 16, border: '1px solid #e5e7eb' }}>
        {detailCard}
      </div>
    </Flex>
  )
}
