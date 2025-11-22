# Basic A2A message schema helpers
import json
from typing import Dict

def make_message(frm: str, to: str, job_id: str, msg_type: str, payload: Dict):
    return {
        "from": frm,
        "to": to,
        "job_id": job_id,
        "type": msg_type,
        "payload": payload,
    }

def serialize(msg: Dict) -> str:
    return json.dumps(msg)
