// 简单的 API 基址配置
export const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';
export const API_KEY = import.meta.env.VITE_API_KEY || '';

export const buildAuthHeaders = (token) => {
  const key = token || API_KEY;
  return key ? { Authorization: `Bearer ${key}` } : {};
};

export async function login(username, password) {
  const resp = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!resp.ok) {
    throw new Error(`Login failed: ${resp.status}`);
  }
  return resp.json();
}

export async function register(username, password) {
  const resp = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!resp.ok) {
    throw new Error(`Register failed: ${resp.status}`);
  }
  return resp.json();
}

export async function fetchToolBudgetConfig(token) {
  const resp = await fetch(`${API_BASE}/config/tool-budget`, {
    headers: {
      ...buildAuthHeaders(token),
    },
  });
  if (!resp.ok) {
    throw new Error(`Failed to load tool budget: ${resp.status}`);
  }
  return resp.json();
}

export async function createThread(userId, token, title) {
  const resp = await fetch(`${API_BASE}/chat/threads`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...buildAuthHeaders(token) },
    body: JSON.stringify({ user_id: userId, title }),
  });
  if (!resp.ok) {
    throw new Error(`Failed to create thread: ${resp.status}`);
  }
  return resp.json();
}

export async function fetchThreads(userId, token) {
  const search = new URLSearchParams({ user_id: userId });
  const resp = await fetch(`${API_BASE}/chat/threads?${search.toString()}`, {
    headers: { ...buildAuthHeaders(token) },
  });
  if (!resp.ok) {
    throw new Error(`Failed to load threads: ${resp.status}`);
  }
  return resp.json();
}

export async function fetchThreadMessages(threadId, userId, token, limit = 200) {
  const search = new URLSearchParams({ user_id: userId, limit: String(limit) });
  const resp = await fetch(`${API_BASE}/chat/threads/${threadId}/messages?${search.toString()}`, {
    headers: { ...buildAuthHeaders(token) },
  });
  if (!resp.ok) {
    throw new Error(`Failed to load messages: ${resp.status}`);
  }
  return resp.json();
}

export async function deleteThread(threadId, userId, token) {
  const search = new URLSearchParams({ user_id: userId });
  const resp = await fetch(`${API_BASE}/chat/threads/${threadId}?${search.toString()}`, {
    method: 'DELETE',
    headers: { ...buildAuthHeaders(token) },
  });
  if (!resp.ok) {
    throw new Error(`Failed to delete thread: ${resp.status}`);
  }
  return resp.json();
}

export async function renameThread(threadId, userId, title, token) {
  const search = new URLSearchParams({ user_id: userId });
  const resp = await fetch(`${API_BASE}/chat/threads/${threadId}?${search.toString()}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...buildAuthHeaders(token) },
    body: JSON.stringify({ title }),
  });
  if (!resp.ok) {
    throw new Error(`Failed to rename thread: ${resp.status}`);
  }
  return resp.json();
}

export async function updateToolBudgetConfig(payload, token) {
  const resp = await fetch(`${API_BASE}/config/tool-budget`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...buildAuthHeaders(token) },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    throw new Error(`Failed to update tool budget: ${resp.status}`);
  }
  return resp.json();
}
