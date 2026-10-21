"""Live HTTP smoke check for a running demo instance.

Checks the configured LLM, MySQL session persistence, and RAG retrieval through
the actual streaming /chat endpoint. It creates and then deletes one temporary
session. It never prints environment values or the full generated answer.
"""

import argparse
import json
import re
import sys
from uuid import uuid4

import httpx


def run(base_url: str) -> None:
    session_id = f"demo-smoke-{uuid4().hex}"
    timeout = httpx.Timeout(120.0, connect=5.0)

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        health = client.get("/health")
        health.raise_for_status()
        if health.json().get("status") != "ok":
            raise RuntimeError("API health check returned an unexpected status")
        print("PASS API health")

        answer_parts = []
        received_done = False
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
                    elif event["type"] == "error":
                        raise RuntimeError("The streamed Agent request returned an error")

            answer = "".join(answer_parts)
            normalized_answer = re.sub(r"\s+", "", answer)
            if not received_done:
                raise RuntimeError("The stream did not finish with a done event")
            if "星河市演示区样例路1号" not in normalized_answer:
                raise RuntimeError("RAG response did not contain the demo evidence")
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
