export interface ApiEnvelope<T> {
  code: number;
  message: string;
  data: T;
}

export interface TokenData {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface RegisteredUser {
  user_id: number;
  username: string;
}

export interface SessionSummary {
  session_id: string;
  title: string;
  updated_at?: string | null;
}

export interface StoredMessage {
  role: string;
  content?: string | null;
}

export interface Citation {
  source: string;
  content: string;
  score?: number;
  dense_score?: number;
  keyword_score?: number;
}

export interface ChatEvent {
  type: "delta" | "done" | "error";
  content?: string;
  session_id?: string;
  citations?: Citation[];
  message?: string;
}

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

export function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

export async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as Partial<ApiEnvelope<unknown>>;
    return body.message || `请求失败（HTTP ${response.status}）`;
  } catch {
    return `请求失败（HTTP ${response.status}）`;
  }
}

export async function login(username: string, password: string): Promise<TokenData> {
  const response = await fetch(apiUrl("/auth/token"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }

  const body = (await response.json()) as ApiEnvelope<TokenData>;
  if (body.code !== 0 || !body.data?.access_token) {
    throw new Error(body.message || "登录响应格式不正确");
  }
  return body.data;
}

export async function registerUser(
  username: string,
  password: string,
): Promise<RegisteredUser> {
  const response = await fetch(apiUrl("/auth/register"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }

  const body = (await response.json()) as ApiEnvelope<RegisteredUser>;
  if (body.code !== 0 || !body.data?.user_id) {
    throw new Error(body.message || "注册响应格式不正确");
  }
  return body.data;
}

export async function logout(token: string): Promise<void> {
  const response = await fetch(apiUrl("/auth/logout"), {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new Error(await readError(response));
  }
}

export async function listSessions(token: string): Promise<SessionSummary[]> {
  const response = await fetch(apiUrl("/sessions"), {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) throw new Error(await readError(response));
  const body = (await response.json()) as ApiEnvelope<SessionSummary[]>;
  if (body.code !== 0 || !Array.isArray(body.data)) {
    throw new Error(body.message || "历史会话响应格式不正确");
  }
  return body.data;
}

export async function getSessionMessages(
  token: string,
  sessionId: string,
): Promise<StoredMessage[]> {
  const response = await fetch(
    apiUrl(`/sessions/${encodeURIComponent(sessionId)}/messages`),
    { headers: { Authorization: `Bearer ${token}` } },
  );
  if (!response.ok) throw new Error(await readError(response));
  const body = (await response.json()) as ApiEnvelope<{
    messages: StoredMessage[];
  }>;
  if (body.code !== 0 || !Array.isArray(body.data?.messages)) {
    throw new Error(body.message || "会话内容响应格式不正确");
  }
  return body.data.messages;
}

export async function renameSession(
  token: string,
  sessionId: string,
  title: string,
): Promise<void> {
  const response = await fetch(
    apiUrl(`/sessions/${encodeURIComponent(sessionId)}/title`),
    {
      method: "PATCH",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ title }),
    },
  );
  if (!response.ok) throw new Error(await readError(response));
}

export async function streamChat(
  token: string,
  message: string,
  sessionId: string | null,
  onEvent: (event: ChatEvent) => void,
): Promise<void> {
  const response = await fetch(apiUrl("/chat"), {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ message, session_id: sessionId, stream: true }),
  });

  if (!response.ok) {
    throw new Error(await readError(response));
  }
  if (!response.body) {
    throw new Error("浏览器没有收到可读取的流式响应");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const dispatchBlock = (block: string) => {
    const data = block
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (data) {
      onEvent(JSON.parse(data) as ChatEvent);
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    blocks.forEach(dispatchBlock);
    if (done) {
      if (buffer.trim()) dispatchBlock(buffer);
      break;
    }
  }
}
