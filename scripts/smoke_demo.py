"""Live HTTP smoke check for a running demo instance.

Checks the configured LLM, MySQL session persistence, and RAG retrieval through
the actual streaming /chat endpoint. It creates and then deletes one temporary
session. It never prints environment values or the full generated answer.
"""

import argparse
from getpass import getpass
import json
import re
import sys
from uuid import uuid4

import httpx


def get_auth_headers(client: httpx.Client) -> dict[str, str]:
    username = input("演示账号用户名：").strip()
    password = getpass("演示账号密码：")
    if not username or not password:
        raise RuntimeError("用户名和密码不能为空")
    response = client.post(
        "/auth/token",
        json={"username": username, "password": password},
    )
    response.raise_for_status()
    token = response.json().get("data", {}).get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Token endpoint returned no access token")
    return {"Authorization": f"Bearer {token}"}


def run(base_url: str) -> None:
    session_id = f"demo-smoke-{uuid4().hex}"
    timeout = httpx.Timeout(120.0, connect=5.0)

    with httpx.Client(
        base_url=base_url.rstrip("/"),
        timeout=timeout,
    ) as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json().get("status") != "ok":
            raise RuntimeError("API health check returned an unexpected status")
        print("PASS API health")
        client.headers.update(get_auth_headers(client))

        answer_parts = []
        received_done = False
        citations = []
        try:
            with client.stream(
                "POST",
                "/chat",
                json={
                    "message": "星河示例科技的办公地点在哪里？",
                    "session_id": session_id,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                if not response.headers.get("content-type", "").startswith(
                    "text/event-stream"
                ):
                    raise RuntimeError("/chat did not return an SSE response")

                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event["type"] == "delta":
                        answer_parts.append(event["content"])
                    elif event["type"] == "done":
                        received_done = event.get("session_id") == session_id
                        citations = event.get("citations", [])
                    elif event["type"] == "error":
                        raise RuntimeError("The streamed Agent request returned an error")

            answer = "".join(answer_parts)
            normalized_answer = re.sub(r"\s+", "", answer)
            if not received_done:
                raise RuntimeError("The stream did not finish with a done event")
            if "星河市演示区样例路1号" not in normalized_answer:
                raise RuntimeError("RAG response did not contain the demo evidence")
            normalized_citations = re.sub(
                r"\s+",
                "",
                "".join(item.get("content", "") for item in citations),
            )
            if "星河市演示区样例路1号" not in normalized_citations:
                raise RuntimeError("RAG response did not return its source citation")
            print("PASS streaming LLM + RAG answer")

            history = client.get(f"/sessions/{session_id}/messages")
            history.raise_for_status()
            messages = history.json()["data"]["messages"]
            if not any(message.get("role") == "tool" for message in messages):
                raise RuntimeError("Session history has no persisted tool result")
            if not any(
                message.get("role") == "assistant"
                and message.get("content") == answer
                for message in messages
            ):
                raise RuntimeError("Session history has no persisted final answer")
            print("PASS MySQL session history")
        finally:
            cleanup = client.delete(f"/sessions/{session_id}")
            if cleanup.status_code not in (200, 404):
                cleanup.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="URL of the running demo API",
    )
    args = parser.parse_args()

    try:
        run(args.base_url)
    except Exception as error:
        print(f"FAIL live demo smoke: {error}", file=sys.stderr)
        return 1

    print("All live demo checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
