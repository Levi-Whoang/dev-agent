/* 前端逻辑：登录态 + 会话列表 + SSE 流式渲染
 *
 * 为什么不用 EventSource：
 * EventSource 只能发 GET，带不了 Authorization 头，也发不了 POST body。
 * 所以这里用 fetch + ReadableStream 自己解析 SSE，等价但可控。
 */

const API_BASE = location.protocol === 'file:' ? 'http://127.0.0.1:8000' : '';

const state = {
  token: localStorage.getItem('dev-agent-token') || '',
  user: null,
  sessions: [],
  sessionId: null,
  streaming: false,
  authMode: 'login',
};

/* ---------- DOM ---------- */
const $ = (id) => document.getElementById(id);
const authView = $('auth-view');
const appView = $('app-view');
const authForm = $('auth-form');
const authHint = $('auth-hint');
const authSubmit = $('auth-submit');
const sessionList = $('session-list');
const messages = $('messages');
const input = $('input');
const sendBtn = $('send');
const connBadge = $('conn-badge');
const chatTitle = $('chat-title');

/* ---------- 通用请求 ---------- */
async function api(path, { method = 'GET', body, auth = true } = {}) {
  const headers = { 'Content-Type': 'application/json' };
  if (auth && state.token) headers.Authorization = `Bearer ${state.token}`;

  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  const payload = await res.json().catch(() => ({}));

  if (!res.ok) {
    if (res.status === 401 && auth) {
      logout('登录已过期，请重新登录');
    }
    throw new Error(payload.message || `请求失败（HTTP ${res.status}）`);
  }
  return payload.data;
}

/* ---------- 登录 / 注册 ---------- */
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    state.authMode = tab.dataset.mode;
    document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
    $('email-row').hidden = state.authMode !== 'register';
    authSubmit.textContent = state.authMode === 'register' ? '注册并登录' : '登录';
    authHint.textContent = '';
  });
});

authForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  authHint.textContent = '';
  authHint.classList.remove('ok');
  authSubmit.disabled = true;
  authSubmit.textContent = '请稍候…';

  try {
    const username = $('username').value.trim();
    const password = $('password').value;
    const path = state.authMode === 'register' ? '/auth/register' : '/auth/login';
    const body =
      state.authMode === 'register'
        ? { username, password, email: $('email').value.trim() || null }
        : { username, password };

    const data = await api(path, { method: 'POST', body, auth: false });
    state.token = data.token.access_token;
    state.user = data.user;
    localStorage.setItem('dev-agent-token', state.token);

    await enterApp();
  } catch (error) {
    authHint.textContent = error.message;
  } finally {
    authSubmit.disabled = false;
    authSubmit.textContent = state.authMode === 'register' ? '注册并登录' : '登录';
  }
});

function logout(message = '') {
  state.token = '';
  state.user = null;
  state.sessions = [];
  state.sessionId = null;
  localStorage.removeItem('dev-agent-token');
  appView.hidden = true;
  authView.hidden = false;
  if (message) {
    authHint.textContent = message;
  }
}

$('logout').addEventListener('click', () => logout());

/* ---------- 进入主界面 ---------- */
async function enterApp() {
  authView.hidden = true;
  appView.hidden = false;
  $('current-user').textContent = state.user?.username || '';

  await refreshSessions();

  if (state.sessions.length) {
    await selectSession(state.sessions[0].session_id);
  } else {
    newSession();
  }
}

async function refreshSessions() {
  try {
    state.sessions = await api('/sessions');
  } catch (error) {
    state.sessions = [];
    setBadge(error.message, 'err');
  }
  renderSessions();
}

function renderSessions() {
  sessionList.innerHTML = '';

  state.sessions.forEach((item) => {
    const li = document.createElement('li');
    li.className = item.session_id === state.sessionId ? 'active' : '';

    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = item.title || item.session_id.slice(0, 8);

    const rename = document.createElement('button');
    rename.className = 'rename';
    rename.textContent = '✎';
    rename.title = '重命名';
    rename.addEventListener('click', (event) => {
      // 不加这句，点铅笔会连带触发 li 上的“选中会话”
      event.stopPropagation();
      beginRename(item, li, name);
    });

    const del = document.createElement('button');
    del.className = 'del';
    del.textContent = '×';
    del.title = '删除会话';
    del.addEventListener('click', async (event) => {
      event.stopPropagation();
      await api(`/sessions/${item.session_id}`, { method: 'DELETE' });
      state.sessions = state.sessions.filter((s) => s.session_id !== item.session_id);
      if (state.sessionId === item.session_id) {
        state.sessionId = state.sessions[0]?.session_id || null;
        if (state.sessionId) await selectSession(state.sessionId);
        else newSession();
      }
      renderSessions();
    });

    li.append(name, rename, del);
    li.addEventListener('click', () => selectSession(item.session_id));
    sessionList.append(li);
  });
}

/* 就地编辑标题。
 *
 * 为什么不用 prompt()：
 * prompt 是阻塞式模态框，会盖住整个页面、不能改样式、移动端体验更差，
 * 而且拿不到“用户点了取消”和“用户清空了内容”的区别。
 * 换成把 <span> 直接替换成 <input>，宽度不跳、上下文不丢。
 */
function beginRename(item, li, nameEl) {
  // 已经在编辑这一行了就别再套一层输入框
  if (li.querySelector('.rename-input')) return;

  const fallback = item.title || item.session_id.slice(0, 8);

  const editor = document.createElement('input');
  editor.className = 'rename-input';
  editor.value = item.title || '';
  editor.maxLength = 60;
  editor.spellcheck = false;

  li.classList.add('editing');
  nameEl.replaceWith(editor);
  editor.focus();
  editor.select();

  // Enter 按下后我们把输入框移出 DOM，浏览器还会补发一个 blur，
  // 没有这个闩就会被提交两次（第二次拿的还是旧标题）。
  let settled = false;

  async function finish(save) {
    if (settled) return;
    settled = true;

    // 取消时不能回填 editor.value，否则“打了字又按 Esc”会把半截标题留下
    const next = save ? editor.value.trim() : '';
    li.classList.remove('editing');
    nameEl.textContent = next || fallback;
    editor.replaceWith(nameEl);

    // 取消 / 清空 / 没改动 —— 都不用发请求
    if (!save || !next || next === item.title) return;

    try {
      await api(`/sessions/${item.session_id}`, {
        method: 'PATCH',
        body: { title: next },
      });
      item.title = next;
      if (state.sessionId === item.session_id) chatTitle.textContent = next;
      setBadge('已重命名', 'ok');
    } catch (error) {
      // 失败就把服务端的真实标题拉回来，别在界面上留一个假名字
      setBadge(error.message, 'err');
      await refreshSessions();
    }
  }

  editor.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      finish(true);
    } else if (event.key === 'Escape') {
      event.preventDefault();
      finish(false);
    }
  });
  editor.addEventListener('blur', () => finish(true));
  // 在输入框里点来点去，不该被当成“选中这个会话”
  editor.addEventListener('click', (event) => event.stopPropagation());
}

async function newSession() {
  try {
    const data = await api('/sessions', { method: 'POST', body: {} });
    state.sessionId = data.session_id;
    await refreshSessions();
    clearMessages();
    chatTitle.textContent = '新对话';
    input.focus();
  } catch (error) {
    setBadge(error.message, 'err');
  }
}

async function selectSession(sessionId) {
  state.sessionId = sessionId;
  chatTitle.textContent =
    state.sessions.find((s) => s.session_id === sessionId)?.title || '会话';
  renderSessions();

  clearMessages();
  try {
    const data = await api(`/sessions/${sessionId}/messages`);
    data.messages.forEach((message) => {
      // 历史里的 thought 是 ReAct 的思考步骤，渲染成弱化的旁注
      if (message.role === 'thought') addThought(message.content);
      else addBubble(message.role === 'user' ? 'user' : 'assistant', message.content);
    });
  } catch (error) {
    setBadge(error.message, 'err');
  }
}

$('new-chat').addEventListener('click', newSession);

$('clear-chat').addEventListener('click', async () => {
  if (!state.sessionId) return;
  await api(`/sessions/${state.sessionId}/clear`, { method: 'POST' });
  clearMessages();
  setBadge('已清空', '');
});

/* ---------- 消息渲染 ---------- */
function clearMessages() {
  messages.innerHTML = '';
}

function addBubble(role, text = '') {
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  avatar.textContent = role === 'user' ? '你' : role === 'thought' ? '✻' : 'W';

  const bubble = document.createElement('div');
  bubble.className = role === 'thought' ? 'bubble thought' : 'bubble';
  bubble.textContent = text;

  wrap.append(avatar, bubble);
  messages.append(wrap);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

/* 历史回放里的思考步骤：不占一个完整的问答位，缩成旁注 */
function addThought(text) {
  return addBubble('thought', text);
}

/* 把正在流式的气泡"降级"成思考。
 *
 * ReAct 里模型会先输出一段自然语言再调工具，那段文字是推理过程，不是答案。
 * 如果直接留在气泡里继续追加，最后得到的会是"我要查一下 + 正式答案"
 * 黏在一起的一段话，读起来像模型在自说自话。
 * 所以收到 thought 事件时把当前气泡就地冻结，另起一个接答案。
 */
function demoteToThought(bubble, text) {
  const wrap = bubble.parentElement;
  wrap.classList.add('thought-mode');
  wrap.querySelector('.avatar').textContent = '✻';
  bubble.classList.add('thought');
  bubble.textContent = text;
}

function setBadge(text, kind = '') {
  connBadge.textContent = text;
  connBadge.className = `badge ${kind}`;
}

/* ---------- 发送 + 流式 ---------- */
input.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('composer').requestSubmit();
  }
});

input.addEventListener('input', () => {
  input.style.height = 'auto';
  input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
});

$('composer').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (state.streaming) return;

  const text = input.value.trim();
  if (!text) return;

  if (!state.sessionId) await newSession();

  input.value = '';
  input.style.height = 'auto';
  addBubble('user', text);

  const bubble = addBubble('assistant', '');
  const cursor = document.createElement('span');
  cursor.className = 'cursor';
  bubble.append(cursor);

  const note = document.createElement('div');
  note.className = 'tool-note';
  note.hidden = true;
  messages.append(note);

  state.streaming = true;
  sendBtn.disabled = true;
  setBadge('思考中…', 'busy');

  let answer = ''; // 当前这一轮正在流式累积的正文
  let currentBubble = bubble; // thought 出现后会换成一个新气泡
  let aborted = ''; // error 事件留下的说明，done 时不能把它冲掉

  const putCursor = () => {
    currentBubble.textContent = answer;
    currentBubble.append(cursor);
    messages.scrollTop = messages.scrollHeight;
  };

  try {
    await streamChat(
      text,
      state.sessionId,
      (eventName, data) => {
        if (eventName === 'meta') {
          // 后端可能新建了会话，同步一下当前 id
          state.sessionId = data.session_id || state.sessionId;
          return;
        }
        if (eventName === 'delta') {
          answer += data.text;
          putCursor();
          return;
        }
        if (eventName === 'thought') {
          // 这一轮的正文其实是推理：冻结它，另起一个气泡接答案
          demoteToThought(currentBubble, data.text);
          note.hidden = true;
          answer = '';
          currentBubble = addBubble('assistant', '');
          putCursor();
          setBadge(`第 ${data.step} 步推理`, 'busy');
          return;
        }
        if (eventName === 'tool_start') {
          note.hidden = false;
          note.textContent = `🔍 正在调用 ${data.name} …`;
          setBadge('调用工具中…', 'busy');
          return;
        }
        if (eventName === 'tool_result') {
          const failed = data.result && data.result.error;
          note.textContent = failed
            ? `⚠️ ${data.name} 失败：${data.result.error}`
            : `✅ ${data.name} 已返回结果`;
          return;
        }
        if (eventName === 'error') {
          aborted = data.message || '出错了';
          setBadge(terminationLabel(data.termination), 'err');
          return;
        }
        if (eventName === 'done') {
          if (aborted) {
            currentBubble.textContent =
              answer || `出错了：${aborted}`;
          } else if (data.answer) {
            answer = data.answer;
            currentBubble.textContent = answer;
          }
          note.hidden = true;
          setBadge(traceLabel(data), data.termination === 'answer' ? 'ok' : 'err');
          return;
        }
      }
    );
  } catch (error) {
    currentBubble.textContent = answer || `请求失败：${error.message}`;
    setBadge('连接失败', 'err');
  } finally {
    cursor.remove();
    state.streaming = false;
    sendBtn.disabled = false;
    await refreshSessions();
  }
});

/* 终止原因说人话，别把 max_steps 这种内部枚举直接甩给用户 */
function terminationLabel(reason) {
  return (
    {
      max_steps: '推理步数用尽',
      loop: '检测到重复操作',
      error: '出错',
    }[reason] || '出错'
  );
}

/* 收尾角标带上 ReAct 的用量，方便判断这次回答花了多少步 */
function traceLabel(data) {
  if (data.termination && data.termination !== 'answer') {
    return terminationLabel(data.termination);
  }
  const steps = data.steps || 0;
  const calls = data.tool_calls || 0;
  return calls ? `就绪 · ${steps} 步 / ${calls} 次工具` : '就绪';
}

async function streamChat(message, sessionId, onEvent) {
  const res = await fetch(`${API_BASE}/chat/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${state.token}`,
    },
    body: JSON.stringify({ message, session_id: sessionId }),
  });

  if (!res.ok) {
    const payload = await res.json().catch(() => ({}));
    if (res.status === 401) logout('登录已过期，请重新登录');
    throw new Error(payload.message || `HTTP ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE 以空行分隔一条消息；按 \n\n 切，剩下的留到下个 chunk
    let index;
    while ((index = buffer.indexOf('\n\n')) >= 0) {
      const raw = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);

      const parsed = parseSseBlock(raw);
      if (parsed) onEvent(parsed.event, parsed.data);
    }
  }
}

function parseSseBlock(raw) {
  let event = 'message';
  const dataLines = [];

  raw.split('\n').forEach((line) => {
    if (line.startsWith('event:')) event = line.slice(6).trim();
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
  });

  if (!dataLines.length) return null;

  try {
    return { event, data: JSON.parse(dataLines.join('\n')) };
  } catch {
    return { event, data: { text: dataLines.join('\n') } };
  }
}

/* ---------- 启动：有 token 就先验证 ---------- */
(async function boot() {
  if (!state.token) return;
  try {
    state.user = await api('/auth/me');
    await enterApp();
  } catch {
    logout();
  }
})();
