"""Push evals/latest.json to the evals branch. Used by GitHub Actions.

Does not print tokens. No-op if the report file is missing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from report import DEFAULT_PATH, load_latest, write_latest

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, check=False, text=True, capture_output=True, **kwargs)


def main() -> int:
    if not DEFAULT_PATH.exists():
        print("no evals/latest.json; skip publish")
        return 0

    run_url = os.environ.get("GITHUB_RUN_URL")
    if run_url:
        report = load_latest()
        report["run_url"] = run_url
        write_latest(report)

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY", "alexoviedo999/last-mile-delivery-agents")
    if not token:
        print("GITHUB_TOKEN unset; report written locally only")
        return 0

    stash = Path(tempfile.mkdtemp()) / "latest.json"
    shutil.copy2(DEFAULT_PATH, stash)
    hist_src = DEFAULT_PATH.parent / "history.jsonl"
    hist_stash = stash.with_name("history.jsonl")
    if hist_src.exists():
        shutil.copy2(hist_src, hist_stash)

    _run(["git", "config", "user.name", "github-actions[bot]"])
    _run(["git", "config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com"])
    # Shallow Actions checkouts do not create origin/evals from `git fetch origin evals`.
    fetch = _run(
        ["git", "fetch", "origin", "+refs/heads/evals:refs/remotes/origin/evals"]
    )
    has_remote = _run(["git", "rev-parse", "--verify", "origin/evals"]).returncode == 0
    if has_remote:
        co = _run(["git", "checkout", "-B", "evals", "origin/evals"])
        if co.returncode != 0:
            print("checkout origin/evals failed:", (co.stderr or "")[:400], file=sys.stderr)
            return 1
    else:
        print("no origin/evals yet; creating orphan evals branch")
        _run(["git", "checkout", "--orphan", "evals"])
        _run(["git", "reset"])

    DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stash, DEFAULT_PATH)
    if hist_stash.exists():
        dest_hist = DEFAULT_PATH.parent / "history.jsonl"
        if dest_hist.exists():
            existing = dest_hist.read_text(encoding="utf-8")
            extra = hist_stash.read_text(encoding="utf-8")
            if extra not in existing:
                dest_hist.write_text(existing + extra, encoding="utf-8")
        else:
            shutil.copy2(hist_stash, dest_hist)

    evals_dir = ROOT / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    _run(["git", "add", "-f", "evals/latest.json", "evals/history.jsonl"])
    commit = _run(["git", "commit", "-m", "Publish DeepEval trajectory report"])
    if commit.returncode != 0:
        print("nothing to commit (report unchanged)")
        return 0

    remote = f"https://x-access-token:{token}@github.com/{repo}.git"
    push = subprocess.run(
        ["git", "push", remote, "HEAD:evals"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    if push.returncode != 0 and "non-fast-forward" in (push.stderr or ""):
        # evals is a generated report branch; rebase onto remote then retry.
        _run(["git", "pull", "--rebase", remote, "evals"])
        push = subprocess.run(
            ["git", "push", remote, "HEAD:evals"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
    if push.returncode != 0:
        err = (push.stderr or "").replace(token, "***")
        print("push failed:", err[:500], file=sys.stderr)
        return push.returncode
    print("published evals/latest.json to origin/evals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
