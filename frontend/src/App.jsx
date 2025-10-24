import React from 'react'
import { Layout, Typography } from 'antd'
import Chat from './components/Chat.jsx'

const { Header, Content, Footer } = Layout
const { Title } = Typography

export default function App() {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ background: '#0b1220' }}>
        <Title level={3} style={{ color: '#a6ffec', margin: 0 }}>Minimal Agent • React + AntD + SSE</Title>
      </Header>
      <Content style={{ padding: 16, background: '#0f172a' }}>
        <div style={{ maxWidth: 920, margin: '0 auto' }}>
          <Chat />
        </div>
      </Content>
      <Footer style={{ textAlign: 'center', background: '#0b1220', color: '#7aa2f7' }}>
        Built with FastAPI × LangGraph × AntD
      </Footer>
    </Layout>
  )
}
