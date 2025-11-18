import React from 'react'
import { Layout, Typography } from 'antd'
import Chat from './components/Chat.jsx'

const { Header, Content, Footer } = Layout
const { Title } = Typography

export default function App() {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#2563eb', boxShadow: '0 2px 6px rgba(37, 99, 235, 0.35)' }}>
        <Title level={3} style={{ color: '#ffffff', margin: 0 }}>Minimal Agent · React + AntD + SSE</Title>
      </Header>
      <Content style={{ padding: 24, background: '#f5f7ff' }}>
        <div style={{ maxWidth: 960, margin: '0 auto' }}>
          <Chat />
        </div>
      </Content>
      <Footer style={{ textAlign: 'center', background: '#e2e8f0', color: '#1e293b' }}>
        Built with FastAPI × LangGraph × AntD
      </Footer>
    </Layout>
  )
}
