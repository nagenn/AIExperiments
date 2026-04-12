import subprocess
import tempfile
import os
import json


def run_bandit(code: str, filename: str = "review_target.py") -> dict:
    """
    Write code to a temp file and run bandit security scanner on it.
    Returns a dict with 'issues' (list of dicts) and 'summary'.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Strip any directory prefix — just use the bare filename
        bare_filename = os.path.basename(filename)
        filepath = os.path.join(tmpdir, bare_filename)
        with open(filepath, "w") as f:
            f.write(code)

        result = subprocess.run(
            ["bandit", "-r", filepath, "-f", "json", "-q"],
            capture_output=True,
            text=True,
        )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError:
            return {
                "issues": [],
                "summary": "Bandit could not parse output.",
                "raw_output": result.stdout + result.stderr,
            }

        issues = []
        for item in data.get("results", []):
            issues.append({
                "severity": item.get("issue_severity", "UNKNOWN"),
                "confidence": item.get("issue_confidence", "UNKNOWN"),
                "description": item.get("issue_text", ""),
                "line": item.get("line_number", 0),
                "test_id": item.get("test_id", ""),
            })

        metrics = data.get("metrics", {}).get("_totals", {})
        summary = (
            f"HIGH: {metrics.get('SEVERITY.HIGH', 0)}, "
            f"MEDIUM: {metrics.get('SEVERITY.MEDIUM', 0)}, "
            f"LOW: {metrics.get('SEVERITY.LOW', 0)}"
        )

        return {
            "issues": issues,
            "summary": summary,
            "raw_output": result.stdout,
        }
