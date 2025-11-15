# multi-agent-system/tools/memory_bank_tool.py
import os
import json
from pathlib import Path
from typing import Any, Dict

MEMORY_DIR = Path(os.getcwd()) / "multi-agent-system" / "state" / "memory_bank"
MEMORY_DIR.mkdir(parents=True, exist_ok=True)

class MemoryBank:
    """
    Very simple file-backed memory bank keyed by patient_id/session_id.
    In production replace with a DB.
    """
    def __init__(self, base_dir: Path = MEMORY_DIR):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, key: str, data: Dict[str, Any]):
        path = self.base_dir / f"{key}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, key: str):
        path = self.base_dir / f"{key}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
