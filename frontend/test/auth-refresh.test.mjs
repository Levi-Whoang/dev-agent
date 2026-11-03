import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { stripTypeScriptTypes } from "node:module";

// 加载实际 API 模块，替换 Vite 环境表达式以便在 Node 中运行。
async function loadApi() {
  const source = await readFile(new URL("../src/api.ts", import.meta.url), "utf8");
  const code = stripTypeScriptTypes(source.replace("import.meta.env.VITE_API_BASE_URL", '"/api"'));
  return import(`data:text/javascript;base64,${Buffer.from(code).toString("base64")}#${crypto.randomUUID()}`);
}

// 创建独立的标签页存储替身，避免使用真实浏览器令牌。
function createStorage() {
  const values = new Map([
    ["dev-agent.access-token", "old-access"], ["dev-agent.refresh-token", "old-refresh"],
    ["dev-agent.username", "alice"],
  ]);
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
}

// 构造统一 JSON 响应。
function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify({ code: 0, data }), { status });
}

test("多个 401 请求只刷新一次，较慢的旧响应直接使用新令牌", async (context) => {
  const storage = createStorage();
  context.mock.method(globalThis, "fetch", async (url, init) => {
    if (url.endsWith("/auth/refresh")) {
      refreshCount++;
      await new Promise((resolve) => setTimeout(resolve, 20));
      return jsonResponse({ access_token: "new-access", refresh_token: "new-refresh" });
    }
    if (init.headers.get("Authorization") === "Bearer new-access") return jsonResponse([]);
    if (url.endsWith("/knowledge/documents")) await new Promise((resolve) => setTimeout(resolve, 50));
    return jsonResponse(null, 401);
  });
  const previous = globalThis.sessionStorage;
  globalThis.sessionStorage = storage;
  context.after(() => { globalThis.sessionStorage = previous; });
  let refreshCount = 0;
  const api = await loadApi();
  await Promise.all([api.listSessions("old-access"), api.listSessions("old-access"), api.listKnowledgeDocuments("old-access")]);
  assert.equal(refreshCount, 1);
  assert.equal(storage.getItem("dev-agent.access-token"), "new-access");
});

for (const status of [200, 401]) {
  test(`旧刷新返回 ${status} 时不能覆盖或清除后登录账号`, async (context) => {
    const storage = createStorage();
    const previous = globalThis.sessionStorage;
    globalThis.sessionStorage = storage;
    context.after(() => { globalThis.sessionStorage = previous; });
    context.mock.method(globalThis, "fetch", async (url) => {
      if (!url.endsWith("/auth/refresh")) return jsonResponse(null, 401);
      storage.setItem("dev-agent.username", "bob");
      storage.setItem("dev-agent.access-token", "bob-access");
      storage.setItem("dev-agent.refresh-token", "bob-refresh");
      return jsonResponse({ access_token: "alice-new", refresh_token: "alice-refresh" }, status);
    });
    const api = await loadApi();
    await assert.rejects(api.listSessions("old-access"));
    assert.equal(storage.getItem("dev-agent.access-token"), "bob-access");
    assert.equal(storage.getItem("dev-agent.refresh-token"), "bob-refresh");
  });
}
