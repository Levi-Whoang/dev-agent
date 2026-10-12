import json
from pathlib import Path
import os
from pathlib import Path

class SessionManager:
    def __init__(self, file_path="sessions.json"):
        self.file_path = Path(file_path)

        if self.file_path.exists():
            with self.file_path.open("r", encoding="utf-8") as f:
                try:
                    self.sessions = json.load(f)
                except json.JSONDecodeError:
                    self.sessions = {}
        else:
            self.sessions = {}

    def get_messages(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = []

        return self.sessions[session_id]

    def save(self):
        temp_path = self.file_path.with_suffix(".tmp")

        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(
                self.sessions,
                f,
                ensure_ascii=False,
                indent=2
            )
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, self.file_path)