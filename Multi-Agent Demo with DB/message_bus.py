"""
message_bus.py
--------------
File-based inter-agent message bus.
In production this would be Redis, Kafka, or a cloud broker.
"""

import json
import os
import time
import fcntl
from datetime import datetime

MESSAGES_DIR = "messages"
LOCK_FILE    = "messages/.lock"


def _ensure():
    os.makedirs(MESSAGES_DIR, exist_ok=True)
    if not os.path.exists(LOCK_FILE):
        open(LOCK_FILE, "w").close()


def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def send_message(from_agent: str, to_agent: str,
                 message_type: str, content: dict):
    _ensure()
    msg = {
        "id":        _ts(),
        "from":      from_agent,
        "to":        to_agent,
        "type":      message_type,
        "content":   content,
        "timestamp": datetime.now().isoformat(),
        "read":      False
    }
    filename = f"{MESSAGES_DIR}/{_ts()}_{from_agent}_to_{to_agent}.json"
    with open(LOCK_FILE, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(filename, "w") as f:
            json.dump(msg, f, indent=2)
        fcntl.flock(lock, fcntl.LOCK_UN)


def get_messages(for_agent: str, message_type: str = None,
                 unread_only: bool = True) -> list:
    _ensure()
    results = []
    for filename in sorted(os.listdir(MESSAGES_DIR)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(MESSAGES_DIR, filename)
        try:
            with open(filepath, "r") as f:
                msg = json.load(f)
        except Exception:
            continue
        if msg["to"] not in [for_agent, "ALL"]:
            continue
        if unread_only and msg.get("read"):
            continue
        if message_type and msg["type"] != message_type:
            continue
        results.append((filepath, msg))
    return results


def mark_read(filepath: str):
    try:
        with open(filepath, "r") as f:
            msg = json.load(f)
        msg["read"] = True
        with open(filepath, "w") as f:
            json.dump(msg, f, indent=2)
    except Exception:
        pass


def wait_for_message(for_agent: str, message_type: str,
                     timeout: int = 180) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        msgs = get_messages(for_agent, message_type=message_type)
        if msgs:
            filepath, msg = msgs[0]
            mark_read(filepath)
            return msg
        time.sleep(1)
    raise TimeoutError(
        f"[{for_agent}] Timed out waiting for '{message_type}' after {timeout}s"
    )


def clear_all():
    _ensure()
    cleared = 0
    for f in os.listdir(MESSAGES_DIR):
        if f.endswith(".json"):
            os.remove(os.path.join(MESSAGES_DIR, f))
            cleared += 1
    print(f"  Message bus cleared ({cleared} messages removed).")
