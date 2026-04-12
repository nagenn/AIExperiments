"""
user_api.py — A deliberately flawed Python REST API handler.
Used as the demo pull request for the AI code review agent.

Intentional issues planted for the agent to find:
  - SQL injection vulnerability
  - Hardcoded credentials
  - No input validation
  - Broad exception handling
  - Unused imports
  - Poor variable naming
  - Missing error handling on file ops
  - Insecure use of eval()
  - MD5 used for password hashing (insecure)
"""

import os
import sys
import json
import hashlib
import subprocess
from datetime import datetime
import sqlite3
import pickle

# Hardcoded credentials — security issue
DB_PASSWORD = "admin123"
SECRET_KEY = "supersecretkey"

db_connection = None


def init_db():
    global db_connection
    db_connection = sqlite3.connect("users.db")
    db_connection.execute(
        "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT, password TEXT, email TEXT)"
    )


def get_user(username):
    # SQL injection vulnerability — input not sanitised
    query = "SELECT * FROM users WHERE username = '" + username + "'"
    cursor = db_connection.execute(query)
    return cursor.fetchone()


def create_user(username, password, email):
    # MD5 is cryptographically broken — should use bcrypt/argon2
    hashed = hashlib.md5(password.encode()).hexdigest()

    # No input validation — username/email not checked
    db_connection.execute(
        f"INSERT INTO users (username, password, email) VALUES ('{username}', '{hashed}', '{email}')"
    )
    db_connection.commit()


def login(username, password):
    u = get_user(username)
    if u:
        # MD5 comparison — same insecure hash
        if u[2] == hashlib.md5(password.encode()).hexdigest():
            return True
    return False


def run_report(report_type):
    # Insecure — eval on user-controlled input
    result = eval(report_type)
    return result


def export_users(filepath):
    # No path validation — path traversal possible
    users = db_connection.execute("SELECT * FROM users").fetchall()
    with open(filepath, "wb") as f:
        # Using pickle to serialise data — insecure deserialisation risk
        pickle.dump(users, f)


def import_config(config_str):
    # Deserialising untrusted pickle data
    data = pickle.loads(config_str)
    return data


def delete_user(uid):
    # Broad bare except — swallows all errors silently
    try:
        db_connection.execute("DELETE FROM users WHERE id = " + str(uid))
        db_connection.commit()
    except:
        pass


def run_shell_command(cmd):
    # Shell injection — command constructed from user input
    output = subprocess.check_output(cmd, shell=True)
    return output


def log_event(event):
    # File opened but never closed — resource leak
    f = open("events.log", "a")
    f.write(f"{datetime.now()} - {event}\n")


def process_data(d):
    # Meaningless variable names, no type hints
    x = []
    for i in d:
        y = i * 2
        x.append(y)
    return x


if __name__ == "__main__":
    init_db()
    create_user("alice", "password123", "alice@example.com")
    print(login("alice", "password123"))
