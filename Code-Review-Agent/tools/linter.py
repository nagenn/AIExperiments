import subprocess
import tempfile
import os


def run_pylint(code: str, filename: str = "review_target.py") -> dict:
    """
    Write code to a temp file and run pylint on it.
    Returns a dict with 'score' and 'issues' (list of strings).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Strip any directory prefix — just use the bare filename
        bare_filename = os.path.basename(filename)
        filepath = os.path.join(tmpdir, bare_filename)
        with open(filepath, "w") as f:
            f.write(code)

        result = subprocess.run(
            [
                "pylint",
                filepath,
                "--output-format=text",
                "--score=yes",
                "--disable=C0114,C0115,C0116",   # suppress missing-docstring warnings for brevity
            ],
            capture_output=True,
            text=True,
        )

        output = result.stdout + result.stderr
        lines = output.splitlines()

        # Extract score line
        score_line = next((l for l in lines if "Your code has been rated" in l), "Score not available")

        # Extract issue lines (warnings/errors/conventions)
        issues = [
            l for l in lines
            if any(tag in l for tag in [": E", ": W", ": C", ": R"])
        ]

        return {
            "score": score_line.strip(),
            "issues": issues,
            "raw_output": output,
        }
