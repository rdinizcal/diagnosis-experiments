"""Persistent solver worker for opt-in candidate evaluation.

The worker is intended to be a verdict oracle equivalent to the legacy
subprocess engine. It executes immutable trace setup exactly once, then serves
JSON-line requests by push/add/check/pop around a sandboxed z3 expression.
Protocol stdout is reserved for JSON; stray setup stdout is redirected to
stderr so parent/worker framing stays safe.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

V_SATISFIED = "SATISFIED"
V_VIOLATED = "VIOLATED"
V_UNDECIDED = "UNDECIDED"
V_ERROR = "ERROR"
PROPERTY_ASSERTION_MARKER = "z3solver.add(Not("
_DEF_RE = re.compile(r"^\s*def\s+\w+\s*\(")


def split_property_script(text: str) -> tuple[list[str], int]:
    """Return setup-prefix lines and the unique mutable property line index."""
    lines = text.splitlines(keepends=True)
    marker_indexes = [
        idx for idx, line in enumerate(lines)
        if PROPERTY_ASSERTION_MARKER in line
    ]
    if len(marker_indexes) != 1:
        raise ValueError(
            f"expected exactly one {PROPERTY_ASSERTION_MARKER!r} marker; "
            f"found {len(marker_indexes)}"
        )
    marker_idx = marker_indexes[0]
    return lines[:marker_idx], marker_idx


def extract_trace_setup(text: str) -> str:
    """Dedent a generated property function prefix into executable setup code."""
    prefix_lines, _ = split_property_script(text)
    def_idx = None
    for idx, line in enumerate(prefix_lines):
        if _DEF_RE.match(line):
            def_idx = idx
            break

    if def_idx is None:
        return "".join(prefix_lines)

    preamble = prefix_lines[:def_idx]
    body = prefix_lines[def_idx + 1:]
    return "".join(preamble) + textwrap.dedent("".join(body))


def build_solver_namespace(property_path: str | Path) -> dict[str, object]:
    """Execute trace setup once and return the z3 namespace and solver."""
    setup_code = extract_trace_setup(Path(property_path).read_text(encoding="utf-8"))
    namespace: dict[str, object] = {}
    with redirect_stdout(sys.stderr):
        exec(compile(setup_code, f"<trace-setup:{property_path}>", "exec"), namespace)
    if "z3solver" not in namespace:
        raise RuntimeError("trace setup did not define z3solver")
    return namespace


def make_eval_globals(namespace: dict[str, object]) -> dict[str, object]:
    """Return the sandboxed eval namespace: z3 and trace names, no builtins."""
    eval_globals = dict(namespace)
    eval_globals["__builtins__"] = {}
    return eval_globals


def check_expr(
    namespace: dict[str, object],
    eval_globals: dict[str, object],
    expression: str,
    timeout_ms: int,
) -> tuple[str, float, str | None]:
    """Evaluate one expression against the shared solver with push/pop isolation."""
    import z3

    solver = namespace["z3solver"]
    solver.push()
    try:
        solver.set("timeout", int(timeout_ms))
        constraint = eval(expression, eval_globals)  # noqa: S307 - sandboxed globals
        solver.add(constraint)
        start = time.perf_counter()
        result = solver.check()
        solve_seconds = time.perf_counter() - start
    except Exception as exc:
        solver.pop()
        return V_ERROR, 0.0, f"{type(exc).__name__}: {exc}"
    solver.pop()

    if result == z3.unsat:
        return V_SATISFIED, solve_seconds, None
    if result == z3.sat:
        return V_VIOLATED, solve_seconds, None
    return V_UNDECIDED, solve_seconds, None


class WorkerCrash(RuntimeError):
    """Raised when a worker process exits or violates the JSON protocol."""


class SolverWorker:
    """Parent-side handle for a long-lived ``diagnosis.worker`` subprocess."""

    def __init__(
        self,
        property_path: str | Path,
        python_executable: Optional[str] = None,
        log_path: str | Path | None = None,
    ) -> None:
        self.property_path = str(property_path)
        self.python_executable = python_executable or sys.executable
        self.log_path = None if log_path is None else str(log_path)
        self.proc: Optional[subprocess.Popen[str]] = None
        self._log_fh = None

    def is_alive(self) -> bool:
        """Return whether the worker process is currently running."""
        return self.proc is not None and self.proc.poll() is None

    def start(self) -> None:
        """Start the worker and wait for its readiness JSON line."""
        stderr = subprocess.PIPE
        if self.log_path is not None:
            self._log_fh = open(self.log_path, "a", encoding="utf-8")
            stderr = self._log_fh
        self.proc = subprocess.Popen(
            [self.python_executable, "-m", "diagnosis.worker", self.property_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
        )
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            self._reap()
            raise WorkerCrash("worker exited before readiness")
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            self._reap()
            raise WorkerCrash(f"worker sent non-JSON readiness: {line!r}") from exc
        if not message.get("ready"):
            self._reap()
            raise WorkerCrash(f"worker failed to initialize: {message.get('error')}")

    def check(self, expression: str, timeout_ms: int) -> tuple[str, float]:
        """Send one expression and return ``(verdict, solve_seconds)``."""
        if not self.is_alive():
            raise WorkerCrash("worker is not running")
        assert self.proc is not None
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None

        try:
            self.proc.stdin.write(
                json.dumps({"expr": expression, "timeout_ms": int(timeout_ms)}) + "\n"
            )
            self.proc.stdin.flush()
        except OSError as exc:
            raise WorkerCrash(f"failed to send worker request: {exc}") from exc

        line = self.proc.stdout.readline()
        if not line:
            self._reap()
            raise WorkerCrash("worker exited before reply")
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkerCrash(f"worker sent non-JSON reply: {line!r}") from exc
        return str(message.get("verdict", V_ERROR)), float(message.get("solve_seconds", 0.0))

    def stop(self) -> None:
        """Request graceful shutdown, killing the worker if needed."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                assert self.proc.stdin is not None
                self.proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()
        self._reap()

    def restart(self) -> None:
        """Stop and start a fresh worker process."""
        self.stop()
        self.start()

    def _reap(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        if self.proc is not None:
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
            self.proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def __enter__(self) -> "SolverWorker":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def _write_json(message: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    """Run the JSON-lines worker protocol."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _write_json({"ready": False, "error": "usage: python -m diagnosis.worker <property.py>"})
        return 2

    start = time.perf_counter()
    try:
        namespace = build_solver_namespace(args[0])
        eval_globals = make_eval_globals(namespace)
    except Exception as exc:
        _write_json({"ready": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1

    _write_json({"ready": True, "setup_seconds": time.perf_counter() - start})

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _write_json({"verdict": V_ERROR, "solve_seconds": 0.0, "error": str(exc)})
            continue
        if request.get("cmd") == "shutdown":
            break
        expression = request.get("expr")
        if not isinstance(expression, str):
            _write_json({"verdict": V_ERROR, "solve_seconds": 0.0, "error": "missing expr"})
            continue
        verdict, solve_seconds, error = check_expr(
            namespace,
            eval_globals,
            expression,
            int(request.get("timeout_ms", 3600 * 1000)),
        )
        _write_json({"verdict": verdict, "solve_seconds": solve_seconds, "error": error})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
