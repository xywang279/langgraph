import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { Layout, Typography, Card, Form, Input, Button, Space, Alert, message, Tabs, Tag } from 'antd'
import Chat from './components/Chat.jsx'
import KnowledgePanel from './components/KnowledgePanel.jsx'
import OpsPanel from './components/OpsPanel.jsx'
import { login, register as registerUser, API_KEY } from './api.js'

const { Header, Content, Footer } = Layout
const { Title, Paragraph, Text } = Typography

const usePersistentUserId = () => {
  const readId = () => {
    try {
      return window.localStorage.getItem('lg-user-id') || ''
    } catch (err) {
      console.warn('unable to read user-id from storage', err)
      return ''
    }
  }

  const [userId, setUserId] = useState(readId)

  useEffect(() => {
    try {
      if (userId) {
        window.localStorage.setItem('lg-user-id', userId)
      } else {
        window.localStorage.removeItem('lg-user-id')
      }
    } catch (err) {
      console.warn('unable to persist user-id', err)
    }
  }, [userId])

  return [userId, setUserId]
}

export default function App() {
  const readStoredToken = useCallback(() => {
    try {
      return window.localStorage.getItem('lg-api-key') || API_KEY || ''
    } catch (err) {
      console.warn('unable to read stored token', err)
      return API_KEY || ''
    }
  }, [])

  const [token, setToken] = useState(readStoredToken)
  const [loginLoading, setLoginLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('chat')
  const [authMode, setAuthMode] = useState('login')
  const [form] = Form.useForm()
  const [userId, setUserId] = usePersistentUserId()
  const [tenantId, setTenantId] = useState(() => {
    try {
      return window.localStorage.getItem('lg-tenant-id') || 'tenant-demo'
    } catch (err) {
      return 'tenant-demo'
    }
  })

  useEffect(() => {
    try {
      if (tenantId) {
        window.localStorage.setItem('lg-tenant-id', tenantId)
      } else {
        window.localStorage.removeItem('lg-tenant-id')
      }
    } catch (err) {
      console.warn('persist tenant failed', err)
    }
  }, [tenantId])

  const persistToken = useCallback(
    (value) => {
      setToken(value)
      try {
        if (value) {
          window.localStorage.setItem('lg-api-key', value)
        } else {
          window.localStorage.removeItem('lg-api-key')
        }
      } catch (err) {
        console.warn('failed to persist token', err)
      }
    },
    [setToken]
  )

  const handleAuth = useCallback(
    async (values) => {
      setLoginLoading(true)
      try {
        const fn = authMode === 'login' ? login : registerUser
        const res = await fn(values.username.trim(), values.password)
        persistToken(res.token)
        setUserId(res.user)
        message.success(authMode === 'login' ? `已登录：${res.user}` : `注册成功并已登录：${res.user}`)
      } catch (err) {
        console.error('auth failed', err)
        message.error(err?.message || '认证失败')
      } finally {
        setLoginLoading(false)
      }
    },
    [authMode, persistToken, setUserId]
  )

  const handleLogout = useCallback(() => {
    persistToken('')
    setUserId('')
    message.success('已退出登录')
  }, [persistToken, setUserId])

  const loginCard = useMemo(
    () => (
      <Card
        title="登录以继续"
        bordered={false}
        style={{
          marginTop: 24,
          boxShadow: '0 8px 26px rgba(37, 99, 235, 0.10)',
          border: '1px solid #e2e8f0',
          maxWidth: 520,
        }}
      >
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <Alert
            type="info"
            showIcon
            message="需要先获取会话 Token"
            description="使用后台配置的账号密码登录，前端会把返回的 Bearer Token 保存在浏览器（lg-api-key）。"
          />
          <Form layout="vertical" form={form} initialValues={{ username: 'admin', password: 'admin123' }} onFinish={handleAuth}>
            <Form.Item label="用户名" name="username" rules={[{ required: true, message: '请输入用户名' }]}>
              <Input placeholder="admin" autoComplete="username" />
            </Form.Item>
            <Form.Item label="密码" name="password" rules={[{ required: true, message: '请输入密码' }]}>
              <Input.Password placeholder="••••••••" autoComplete="current-password" />
            </Form.Item>
            <Space size="middle" wrap>
              <Button type="primary" htmlType="submit" loading={loginLoading}>
                {authMode === 'login' ? '登录' : '注册并登录'}
              </Button>
              <Button htmlType="button" onClick={() => form.resetFields()} disabled={loginLoading}>
                重置
              </Button>
              <Button type="link" htmlType="button" onClick={() => setAuthMode(authMode === 'login' ? 'register' : 'login')}>
                {authMode === 'login' ? '没有账号？去注册' : '已有账号？去登录'}
              </Button>
            </Space>
          </Form>
          <Paragraph type="secondary" style={{ marginBottom: 0 }}>
            也可以在部署时设置 <code>VITE_API_KEY</code> 作为默认 Token。
          </Paragraph>
        </Space>
      </Card>
    ),
    [authMode, form, handleAuth, loginLoading]
  )

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#2563eb', boxShadow: '0 2px 6px rgba(37, 99, 235, 0.35)' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Title level={2} style={{ color: '#ffffff', margin: 0 }}>
            Minimal Agent · React + AntD + SSE
          </Title>
          {token ? (
            <Space>
              <Tag color="gold">{userId ? `当前用户：${userId}` : '已登录'}</Tag>
              <Input
                size="small"
                value={tenantId}
                onChange={(e) => setTenantId(e.target.value)}
                placeholder="tenant id"
                style={{ width: 180, background: '#fff' }}
              />
              <Button danger size="small" onClick={handleLogout}>
                退出登录
              </Button>
            </Space>
          ) : null}
        </div>
      </Header>
      <Content style={{ padding: 5, background: '#f5f7ff', display: 'flex', justifyContent: 'center' }}>
        <div
          style={{
            width: '100%',
            maxWidth: 1500,
          }}
        >
          {token ? (
            <Tabs
              activeKey={activeTab}
              onChange={setActiveTab}
              tabBarStyle={{ display: 'flex', justifyContent: 'center' }}
              centered
              style={{ width: '100%', textAlign: 'center' }}
              items={[
                {
                  key: 'chat',
                  label: '聊天',
                  children: <Chat token={token} userId={userId} setUserId={setUserId} tenantId={tenantId} />,
                },
                {
                  key: 'documents',
                  label: '知识库',
                  children: <KnowledgePanel userId={userId} apiToken={token} tenantId={tenantId} />,
                },
                {
                  key: 'ops',
                  label: '运维',
                  children: <OpsPanel apiToken={token} userId={userId} tenantId={tenantId} />,
                },
              ]}
            />
          ) : (
            <div
              style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                width: '100%',
                minHeight: 'calc(100vh - 200px)',
                padding: '24px 0',
              }}
            >
              {loginCard}
            </div>
          )}
        </div>
      </Content>
      <Footer style={{ textAlign: 'center', background: '#e2e8f0', color: '#1e293b' }}>
        Built with FastAPI · LangGraph · AntD
      </Footer>
    </Layout>
  )
}
