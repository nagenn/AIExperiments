import requests
from config import GITHUB_TOKEN, GITHUB_REPO


HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


def fetch_pr_diff(pr_number: int) -> str:
    """Fetch the unified diff for a pull request."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/pulls/{pr_number}"
    headers = {**HEADERS, "Accept": "application/vnd.github.v3.diff"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.text


def fetch_pr_files(pr_number: int) -> list[dict]:
    """Fetch the list of files changed in a pull request."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/pulls/{pr_number}/files"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def fetch_file_content(repo: str, file_path: str, ref: str) -> str:
    """Fetch raw file content at a specific git ref."""
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}?ref={ref}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 404:
        return ""
    response.raise_for_status()
    import base64
    content = response.json().get("content", "")
    return base64.b64decode(content).decode("utf-8", errors="replace")


def post_pr_comment(pr_number: int, body: str) -> dict:
    """Post a general comment on the pull request."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues/{pr_number}/comments"
    response = requests.post(url, headers=HEADERS, json={"body": body})
    response.raise_for_status()
    return response.json()


def post_inline_comment(pr_number: int, commit_sha: str, path: str, line: int, body: str) -> dict:
    """Post an inline review comment. Falls back to PR comment if line is not in diff."""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/pulls/{pr_number}/comments"
    payload = {
        "body": body,
        "commit_id": commit_sha,
        "path": path,
        "line": line,
        "side": "RIGHT",
    }
    response = requests.post(url, headers=HEADERS, json=payload)

    if response.status_code == 422:
        # Line not in diff — fall back to regular PR comment
        fallback_body = f"**`{path}` line {line}:** {body}"
        return post_pr_comment(pr_number, fallback_body)

    response.raise_for_status()
    return response.json()