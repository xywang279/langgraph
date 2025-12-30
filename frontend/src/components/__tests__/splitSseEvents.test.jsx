import { describe, it, expect } from 'vitest'
import { splitSseEvents } from '../Chat.jsx'

describe('splitSseEvents', () => {
  it('parses a single event payload', () => {
    const input = 'data: {"event":"end"}\n\n'
    const { events, rest } = splitSseEvents(input)
    expect(events).toHaveLength(1)
    expect(events[0]).toBe('{"event":"end"}')
    expect(rest).toBe('')
  })

  it('parses multiple events and keeps remainder', () => {
    const input = 'data: {"event":"plan"}\n\n' + 'data: {"event":"status"}\n\npartial'
    const { events, rest } = splitSseEvents(input)
    expect(events).toHaveLength(2)
    expect(events[0]).toBe('{"event":"plan"}')
    expect(events[1]).toBe('{"event":"status"}')
    expect(rest).toBe('partial')
  })

  it('joins multi-line data blocks', () => {
    const input = 'data: {"event":"message",\n' + 'data: "chunk"}\n\n'
    const { events } = splitSseEvents(input)
    expect(events).toHaveLength(1)
    expect(events[0]).toBe('{"event":"message",\n"chunk"}')
  })
})
