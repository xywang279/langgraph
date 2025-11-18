// 简单的 API 基址配置
export const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

export async function fetchToolBudgetConfig() {
  const resp = await fetch(`${API_BASE}/config/tool-budget`);
  if (!resp.ok) {
    throw new Error(`Failed to load tool budget: ${resp.status}`);
  }
  return resp.json();
}

export async function updateToolBudgetConfig(payload) {
  const resp = await fetch(`${API_BASE}/config/tool-budget`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    throw new Error(`Failed to update tool budget: ${resp.status}`);
  }
  return resp.json();
}
