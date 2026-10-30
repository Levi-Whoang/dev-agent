import { useEffect, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import {
  getSessionMessages,
  listSessions,
  listKnowledgeDocuments,
  uploadKnowledgeDocument,
  replaceKnowledgeDocument,
  renameKnowledgeDocument,
  deleteKnowledgeDocument,
  deleteSession as deleteSessionRequest,
  login,
  logout as revokeToken,
  renameSession,
  registerUser,
  streamChat,
} from "./api";
import type { ChatEvent, Citation, KnowledgeDocument, SessionSummary, StoredMessage } from "./api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
}

const TOKEN_KEY = "dev-agent.access-token";
const REFRESH_TOKEN_KEY = "dev-agent.refresh-token";
const USERNAME_KEY = "dev-agent.username";
const starters = ["知识库里包含哪些内容？", "请总结一下主要信息。", "这个项目能做什么？"];

function createMessage(role: Message["role"], content: string): Message {
  return { id: crypto.randomUUID(), role, content, citations: [] };
}

export default function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem(TOKEN_KEY) || "");
  const [username, setUsername] = useState(() => sessionStorage.getItem(USERNAME_KEY) || "");
  const [loginName, setLoginName] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [registrationCode, setRegistrationCode] = useState("");
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [loginError, setLoginError] = useState("");
  const [loginNotice, setLoginNotice] = useState("");
  const [loginBusy, setLoginBusy] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [chatError, setChatError] = useState("");
  const [history, setHistory] = useState<SessionSummary[]>([]);
  const [historyError, setHistoryError] = useState("");
  const [historyBusy, setHistoryBusy] = useState(false);
  const [renamingSessionId, setRenamingSessionId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [deletingSessionId, setDeletingSessionId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [documentError, setDocumentError] = useState("");
  const [documentBusy, setDocumentBusy] = useState(false);
  const [replacingDocumentId, setReplacingDocumentId] = useState<string | null>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const replaceInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  useEffect(() => {
    if (!token) {
      setHistory([]);
      return;
    }
    let cancelled = false;
    setHistoryBusy(true);
    listSessions(token)
      .then((items) => {
        if (!cancelled) {
          setHistory(items);
          setHistoryError("");
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setHistoryError(error instanceof Error ? error.message : "历史会话加载失败");
        }
      })
      .finally(() => {
        if (!cancelled) setHistoryBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  useEffect(() => {
    if (!token) {
      setDocuments([]);
      return;
    }
    let cancelled = false;
    listKnowledgeDocuments(token)
      .then((items) => { if (!cancelled) setDocuments(items); })
      .catch((error) => {
        if (!cancelled) setDocumentError(error instanceof Error ? error.message : "文档列表加载失败");
      });
    return () => { cancelled = true; };
  }, [token]);

  async function refreshDocuments() {
    const items = await listKnowledgeDocuments(token);
    setDocuments(items);
  }

  async function handleDocumentFile(file: File | undefined) {
    if (!file) return;
    setDocumentBusy(true);
    setDocumentError("");
    try {
      if (replacingDocumentId) {
        await replaceKnowledgeDocument(token, replacingDocumentId, file);
      } else {
        await uploadKnowledgeDocument(token, file);
      }
      await refreshDocuments();
    } catch (error) {
      setDocumentError(error instanceof Error ? error.message : "文档处理失败");
    } finally {
      setDocumentBusy(false);
      setReplacingDocumentId(null);
    }
  }

  async function handleRenameDocument(document: KnowledgeDocument) {
    const name = window.prompt("输入新的文档名称", document.name)?.trim();
    if (!name || name === document.name) return;
    setDocumentBusy(true);
    setDocumentError("");
    try {
      await renameKnowledgeDocument(token, document.document_id, name);
      await refreshDocuments();
    } catch (error) {
      setDocumentError(error instanceof Error ? error.message : "文档改名失败");
    } finally {
      setDocumentBusy(false);
    }
  }

  async function handleDeleteDocument(document: KnowledgeDocument) {
    if (!window.confirm(`确定从知识库删除“${document.name}”吗？`)) return;
    setDocumentBusy(true);
    setDocumentError("");
    try {
      await deleteKnowledgeDocument(token, document.document_id);
      setDocuments((items) => items.filter((item) => item.document_id !== document.document_id));
    } catch (error) {
      setDocumentError(error instanceof Error ? error.message : "文档删除失败");
    } finally {
      setDocumentBusy(false);
    }
  }

  function displayMessages(stored: StoredMessage[]): Message[] {
    const visible: Message[] = stored
      .filter((item) => item.role === "user" || item.role === "assistant")
      .filter((item) => typeof item.content === "string" && item.content.length > 0)
      .map((item) => ({
        id: crypto.randomUUID(),
        role: item.role as Message["role"],
        content: item.content as string,
        citations: [],
      }));
    const citations = stored.flatMap((item) => {
      if (item.role !== "tool" || !item.content) return [];
      try {
        const parsed = JSON.parse(item.content) as { citations?: Citation[] };
        return Array.isArray(parsed.citations) ? parsed.citations : [];
      } catch {
        return [];
      }
    });
    const lastAssistant = visible
      .map((item) => item.role === "assistant")
      .lastIndexOf(true);
    if (lastAssistant >= 0) visible[lastAssistant].citations = citations;
    return visible;
  }

  async function openSession(item: SessionSummary) {
    if (streaming || historyBusy) return;
    setHistoryBusy(true);
    setHistoryError("");
    try {
      const stored = await getSessionMessages(token, item.session_id);
      setMessages(displayMessages(stored));
      setSessionId(item.session_id);
      setInput("");
      setChatError("");
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "会话加载失败");
    } finally {
      setHistoryBusy(false);
    }
  }

  async function saveSessionTitle(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!renamingSessionId) return;
    const title = renameValue.trim().replace(/\s+/g, " ");
    if (!title) {
      setHistoryError("会话名称不能为空");
      return;
    }
    setHistoryBusy(true);
    setHistoryError("");
    try {
      await renameSession(token, renamingSessionId, title);
      setHistory((items) =>
        items.map((item) =>
          item.session_id === renamingSessionId ? { ...item, title } : item,
        ),
      );
      setRenamingSessionId(null);
      setRenameValue("");
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "修改会话名称失败");
    } finally {
      setHistoryBusy(false);
    }
  }

  async function removeSession(item: SessionSummary) {
    if (streaming || historyBusy) return;
    const title = item.title || "新对话";
    if (!window.confirm(`确定删除会话“${title}”吗？此操作无法撤销。`)) return;

    setDeletingSessionId(item.session_id);
    setHistoryBusy(true);
    setHistoryError("");
    try {
      await deleteSessionRequest(token, item.session_id);
      setHistory((items) => items.filter((entry) => entry.session_id !== item.session_id));
      if (sessionId === item.session_id) {
        setSessionId(null);
        setMessages([]);
        setInput("");
        setChatError("");
      }
      if (renamingSessionId === item.session_id) {
        setRenamingSessionId(null);
        setRenameValue("");
      }
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "删除会话失败");
    } finally {
      setDeletingSessionId(null);
      setHistoryBusy(false);
    }
  }

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoginError("");
    setLoginNotice("");
    setLoginBusy(true);
    try {
      const normalizedUsername = loginName.trim();
      if (authMode === "register") {
        if (password.length < 8) {
          throw new Error("密码至少需要 8 个字符");
        }
        if (password !== confirmPassword) {
          throw new Error("两次输入的密码不一致");
        }
        const user = await registerUser(normalizedUsername, password, registrationCode);
        setLoginName(user.username);
        setPassword("");
        setConfirmPassword("");
        setRegistrationCode("");
        setAuthMode("login");
        setLoginNotice("注册成功，请使用新账号登录");
      } else {
        const result = await login(normalizedUsername, password);
        sessionStorage.setItem(TOKEN_KEY, result.access_token);
        sessionStorage.setItem(REFRESH_TOKEN_KEY, result.refresh_token);
        sessionStorage.setItem(USERNAME_KEY, normalizedUsername);
        setToken(result.access_token);
        setUsername(normalizedUsername);
        setPassword("");
      }
    } catch (error) {
      setLoginError(error instanceof Error ? error.message : "登录失败，请稍后重试");
    } finally {
      setLoginBusy(false);
    }
  }

  async function sendMessage(rawMessage: string) {
    const message = rawMessage.trim();
    if (!message || !token || streaming) return;

    setInput("");
    setChatError("");
    setStreaming(true);
    const userMessage = createMessage("user", message);
    const assistantMessage = createMessage("assistant", "");
    setMessages((current) => [...current, userMessage, assistantMessage]);

    const updateAssistant = (update: (current: Message) => Message) => {
      setMessages((current) =>
        current.map((item) => (item.id === assistantMessage.id ? update(item) : item)),
      );
    };

    try {
      await streamChat(token, message, sessionId, (event: ChatEvent) => {
        if (event.type === "delta" && event.content) {
          updateAssistant((item) => ({ ...item, content: item.content + event.content }));
        } else if (event.type === "done") {
          if (event.session_id) {
            setSessionId(event.session_id);
            void listSessions(token).then(setHistory).catch(() => undefined);
          }
          updateAssistant((item) => ({ ...item, citations: event.citations || [] }));
        } else if (event.type === "error") {
          throw new Error(event.message || "生成回复时发生错误");
        }
      });
    } catch (error) {
      const messageText = error instanceof Error ? error.message : "请求失败，请稍后重试";
      setChatError(messageText);
      if (/401|未认证|认证失败|JWT/i.test(messageText)) {
        sessionStorage.removeItem(TOKEN_KEY);
        sessionStorage.removeItem(REFRESH_TOKEN_KEY);
        sessionStorage.removeItem(USERNAME_KEY);
        setToken("");
        setUsername("");
        setMessages([]);
        setSessionId(null);
      }
    } finally {
      setStreaming(false);
    }
  }

  function handleComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void sendMessage(input);
    }
  }

  function startNewChat() {
    if (streaming) return;
    setMessages([]);
    setSessionId(null);
    setChatError("");
    setInput("");
  }

  async function handleLogout() {
    try {
      await revokeToken(token);
    } catch {
      // 令牌已过期或服务不可达时仍清除浏览器中的登录状态。
    } finally {
      sessionStorage.removeItem(TOKEN_KEY);
      sessionStorage.removeItem(REFRESH_TOKEN_KEY);
      sessionStorage.removeItem(USERNAME_KEY);
      setToken("");
      setUsername("");
      setMessages([]);
      setSessionId(null);
      setChatError("");
      setHistory([]);
    }
  }

  if (!token) {
    return (
      <main className="login-page">
        <section className="login-card">
          <div className="brand-mark">D</div>
          <p className="eyebrow">DEV AGENT · KNOWLEDGE ASSISTANT</p>
          <h1>{authMode === "login" ? "欢迎回来" : "创建账号"}</h1>
          <p className="login-copy">{authMode === "login" ? "登录后即可向你的知识库提问。" : "注册成功后即可登录并开始使用。"}</p>
          <form className="login-form" onSubmit={handleLogin}>
            <label htmlFor="username">用户名</label>
            <input
              id="username"
              autoComplete="username"
              minLength={authMode === "register" ? 3 : 1}
              maxLength={64}
              pattern={authMode === "register" ? "[A-Za-z0-9_.-]{3,64}" : undefined}
              value={loginName}
              onChange={(event) => setLoginName(event.target.value)}
              placeholder="3–64 位字母、数字、点、横线或下划线"
              required
            />
            <label htmlFor="password">密码</label>
            <input
              id="password"
              type="password"
              autoComplete={authMode === "login" ? "current-password" : "new-password"}
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              placeholder={authMode === "register" ? "至少 8 个字符" : "输入登录密码"}
              minLength={authMode === "register" ? 8 : 1}
              required
            />
            {authMode === "register" && (
              <>
                <label htmlFor="registration-code">注册验证码</label>
                <input
                  id="registration-code"
                  type="password"
                  autoComplete="one-time-code"
                  value={registrationCode}
                  onChange={(event) => setRegistrationCode(event.target.value)}
                  placeholder="输入管理员提供的验证码"
                  minLength={8}
                  maxLength={128}
                  required
                />
                <label htmlFor="confirm-password">确认密码</label>
                <input
                  id="confirm-password"
                  type="password"
                  autoComplete="new-password"
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                  placeholder="再次输入密码"
                  minLength={8}
                  required
                />
              </>
            )}
            {loginNotice && <div className="notice">{loginNotice}</div>}
            {loginError && <div className="alert">{loginError}</div>}
            <button className="primary-button login-button" disabled={loginBusy}>
              {loginBusy ? "请稍候…" : authMode === "login" ? "登录并开始对话" : "创建账号"}
              <span aria-hidden="true">→</span>
            </button>
          </form>
          <p className="auth-switch">
            {authMode === "login" ? "还没有账号？" : "已经注册过？"}
            <button
              type="button"
              onClick={() => {
                setAuthMode(authMode === "login" ? "register" : "login");
                setLoginError("");
                setLoginNotice("");
                setPassword("");
                setConfirmPassword("");
                setRegistrationCode("");
              }}
            >
              {authMode === "login" ? "立即注册" : "返回登录"}
            </button>
          </p>
        </section>
      </main>
    );
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-mark small">D</div>
          <div>
            <strong>Dev Agent</strong>
            <span>知识助手</span>
          </div>
        </div>
        <button className="new-chat-button" onClick={startNewChat} disabled={streaming}>
          <span className="plus">＋</span> 新建对话
        </button>
        <div className="sidebar-label">当前会话</div>
        <div className="session-card">
          <span className="session-dot" />
          <span>
            {sessionId
              ? history.find((item) => item.session_id === sessionId)?.title || `会话 ${sessionId.slice(0, 8)}`
              : "新对话"}
          </span>
        </div>
        <div className="history-heading">
          <span className="sidebar-label">历史会话</span>
          {historyBusy && <span className="history-loading">加载中</span>}
        </div>
        <div className="history-list">
          {history.map((item) => (
            <div
              className={`history-row ${sessionId === item.session_id ? "active" : ""}`}
              key={item.session_id}
            >
              {renamingSessionId === item.session_id ? (
                <form className="rename-form" onSubmit={(event) => void saveSessionTitle(event)}>
                  <input
                    aria-label="会话名称"
                    autoFocus
                    maxLength={120}
                    value={renameValue}
                    onChange={(event) => setRenameValue(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Escape") setRenamingSessionId(null);
                    }}
                  />
                  <button type="submit" disabled={historyBusy} title="保存名称">✓</button>
                  <button type="button" disabled={historyBusy} title="取消" onClick={() => setRenamingSessionId(null)}>×</button>
                </form>
              ) : (
                <>
                  <button
                    className="history-item"
                    onClick={() => void openSession(item)}
                    disabled={streaming || historyBusy}
                    title={item.title || item.session_id}
                  >
                    <span className="history-item-title">{item.title || "新对话"}</span>
                    <span className="history-item-date">
                      {item.updated_at
                        ? new Date(item.updated_at).toLocaleString("zh-CN", {
                            month: "2-digit",
                            day: "2-digit",
                            hour: "2-digit",
                            minute: "2-digit",
                          })
                        : ""}
                    </span>
                  </button>
                  <button
                    className="rename-button"
                    disabled={streaming || historyBusy}
                    title="修改会话名称"
                    aria-label={`修改会话名称：${item.title || item.session_id}`}
                    onClick={() => {
                      setRenamingSessionId(item.session_id);
                      setRenameValue(item.title || "");
                      setHistoryError("");
                    }}
                  >
                    ✎
                  </button>
                  <button
                    className="delete-button"
                    disabled={streaming || historyBusy}
                    title="删除会话"
                    aria-label={`删除会话：${item.title || item.session_id}`}
                    onClick={() => void removeSession(item)}
                  >
                    {deletingSessionId === item.session_id ? "…" : "×"}
                  </button>
                </>
              )}
            </div>
          ))}
          {!historyBusy && history.length === 0 && (
            <div className="history-empty">还没有历史会话</div>
          )}
          {historyError && <div className="history-error">{historyError}</div>}
        </div>
        <section className="documents-panel" aria-label="知识库文档管理">
          <div className="documents-heading">
            <span className="sidebar-label">知识库文档</span>
            <button
              className="document-upload-button"
              disabled={documentBusy}
              onClick={() => uploadInputRef.current?.click()}
            >
              {documentBusy ? "处理中…" : "＋ 上传"}
            </button>
            <input
              ref={uploadInputRef}
              className="visually-hidden"
              type="file"
              accept=".txt,.md,.csv,.json,.html,.htm,.pdf,.docx,.xlsx,.pptx"
              onChange={(event) => {
                void handleDocumentFile(event.target.files?.[0]);
                event.currentTarget.value = "";
              }}
            />
            <input
              ref={replaceInputRef}
              className="visually-hidden"
              type="file"
              accept=".txt,.md,.csv,.json,.html,.htm,.pdf,.docx,.xlsx,.pptx"
              onChange={(event) => {
                void handleDocumentFile(event.target.files?.[0]);
                event.currentTarget.value = "";
              }}
            />
          </div>
          <div className="documents-list">
            {documents.map((document) => (
              <div className="document-row" key={document.document_id} title={`${document.name} · ${document.format.toUpperCase()}`}>
                <span className="document-label">
                  <span className="document-name">{document.name}</span>
                  <span className="document-meta">{document.format.toUpperCase()} · {document.uploaded_at ? new Date(document.uploaded_at).toLocaleDateString("zh-CN") : ""}</span>
                </span>
                <button
                  title="替换文档内容"
                  aria-label={`替换文档：${document.name}`}
                  disabled={documentBusy}
                  onClick={() => {
                    setReplacingDocumentId(document.document_id);
                    replaceInputRef.current?.click();
                  }}
                >⇧</button>
                <button title="修改文档名称" aria-label={`修改文档名称：${document.name}`} disabled={documentBusy} onClick={() => void handleRenameDocument(document)}>✎</button>
                <button title="删除文档" aria-label={`删除文档：${document.name}`} disabled={documentBusy} onClick={() => void handleDeleteDocument(document)}>×</button>
              </div>
            ))}
            {documents.length === 0 && <div className="documents-empty">还没有上传文档</div>}
            {documentError && <div className="documents-error">{documentError}</div>}
          </div>
          <div className="documents-hint">支持 PDF、Word、Excel、PPT、TXT、Markdown、CSV、JSON、HTML，单个文件最大 20 MB。</div>
        </section>
        <div className="sidebar-bottom">
          <div className="connection-state"><span /> API 已连接</div>
          <div className="user-row">
            <div className="avatar">{username.slice(0, 1).toUpperCase()}</div>
            <div className="user-name"><strong>{username}</strong><span>已登录</span></div>
            <button className="logout-button" onClick={() => void handleLogout()} title="退出登录" aria-label="退出登录">
              ↗
            </button>
          </div>
        </div>
      </aside>

      <section className="workspace">
        <header className="topbar">
          <div className="mobile-brand"><div className="brand-mark small">D</div><strong>Dev Agent</strong></div>
          <div className="topbar-actions">
            <div className="topbar-status"><span className="status-pulse" /> AI 助手在线</div>
            <button className="mobile-logout" onClick={() => void handleLogout()}>退出</button>
          </div>
        </header>

        <div className={`conversation ${messages.length ? "has-messages" : "empty"}`}>
          {messages.length === 0 ? (
            <div className="welcome-panel">
              <div className="welcome-icon"><span>✳</span></div>
              <p className="eyebrow">你的专属知识助手</p>
              <h1>今天想了解什么？</h1>
              <p className="welcome-copy">基于知识库检索资料，回答会附上对应的证据来源。</p>
              <div className="starter-list">
                {starters.map((starter) => (
                  <button key={starter} onClick={() => void sendMessage(starter)} disabled={streaming}>
                    <span>{starter}</span><span aria-hidden="true">↗</span>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article className={`message-row ${message.role}`} key={message.id}>
                  {message.role === "assistant" && <div className="message-avatar">✳</div>}
                  <div className="message-body">
                    <div className="message-label">{message.role === "user" ? "你" : "Dev Agent"}</div>
                    <div className={`message-content ${message.role}`}>
                      {message.content || (streaming && message.role === "assistant" ? <span className="typing"><i /><i /><i /></span> : null)}
                    </div>
                    {message.citations.length > 0 && (
                      <div className="sources">
                        <div className="sources-title"><span>⌕</span> 引用来源 <span className="source-count">{message.citations.length}</span></div>
                        {message.citations.map((citation, index) => (
                          <details className="source-card" key={`${citation.source}-${index}`}>
                            <summary><span className="file-icon">▤</span><span className="source-name">{citation.source}</span><span className="source-score">{typeof citation.score === "number" ? `${Math.round(citation.score * 100)}%` : "来源"}</span></summary>
                            <p>{citation.content}</p>
                          </details>
                        ))}
                      </div>
                    )}
                  </div>
                  {message.role === "user" && <div className="user-message-avatar">{username.slice(0, 1).toUpperCase()}</div>}
                </article>
              ))}
              {chatError && <div className="chat-error">{chatError}</div>}
              <div ref={bottomRef} />
            </div>
          )}
        </div>

        <div className="composer-wrap">
          {messages.length === 0 && <div className="composer-hint">按 Enter 发送，Shift + Enter 换行</div>}
          <form className="composer" onSubmit={(event) => { event.preventDefault(); void sendMessage(input); }}>
            <textarea
              aria-label="输入问题"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder="向知识库提问…"
              rows={1}
              maxLength={4000}
              disabled={streaming}
            />
            <button className="send-button" type="submit" disabled={!input.trim() || streaming} aria-label={streaming ? "正在生成" : "发送消息"}>
              {streaming ? <span className="send-spinner" /> : <span>↑</span>}
            </button>
          </form>
          <div className="composer-disclaimer">AI 生成的内容可能存在错误，请结合引用来源核对。</div>
        </div>
      </section>
    </main>
  );
}
