import React from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { ReadableStream } from 'stream/web'
import Chat from '../Chat.jsx'

vi.mock('../../api', () => ({
  API_BASE: 'http://test.local',
  API_KEY: '',
  buildAuthHeaders: () => ({}),
  createThread: vi.fn().mockResolvedValue({ thread_id: 'thread-1' }),
  fetchThreads: vi.fn().mockResolvedValue({ items: [] }),
  fetchThreadMessages: vi.fn().mockResolvedValue({ items: [] }),
  deleteThread: vi.fn().mockResolvedValue({}),
  renameThread: vi.fn().mockResolvedValue({}),
}))

const buildStream = (chunks) => {
  const encoder = new TextEncoder()
  return new ReadableStream({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)))
      controller.close()
    },
  })
}

describe('Chat streaming', () => {
  const originalFetch = global.fetch

  beforeEach(() => {
    global.fetch = vi.fn()
  })

  afterEach(() => {
    global.fetch = originalFetch
    vi.clearAllMocks()
  })

  it('renders interrupt modal on interrupt event', async () => {
    const payloads = [
      'data: {"event":"interrupt","payload":{"message":"Please confirm","options":["continue","cancel"]}}\n\n',
      'data: {"event":"end"}\n\n',
    ]
    const response = {
      ok: true,
      status: 200,
      body: buildStream(payloads),
      json: async () => ({}),
      text: async () => '',
    }
    global.fetch.mockResolvedValue(response)

    render(<Chat token="token" userId="user-1" setUserId={vi.fn()} />)

    const input = screen.getByPlaceholderText('Type your message...')
    fireEvent.change(input, {
      target: { value: 'Hello' },
    })
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter', shiftKey: false })

    expect(await screen.findByText('Please confirm')).toBeInTheDocument()
    expect(await screen.findByRole('button', { name: /continue/i })).toBeInTheDocument()
  })
})
