r"""
campaign_queue.py - the ONLY program of the HCOOH-H2S / HCOOH-N2O publication campaign that starts ORCA.

Agents never start orca.exe. They write an ORCA input (or a command line), drop a JSON job spec into
queue_orchestrator/jobs/ (README.md next to this file; or use --enqueue / enqueue()), and
one runner process executes the specs on the campaign's 8 cores:

    jobs\  (pending)  ->  running\  ->  done\ (ok | reused)   or   failed\ (failed | timeout | stalled)

  * one 8-rank job at a time, or up to 4 concurrent grid jobs of <= 2 ranks (concurrency_group "grid");
    the sum of ranks in flight never exceeds --max-ranks (default and hard cap 8)
  * the machine-wide ORCA lock of goat_isomer_search.py / hcooh-hsh-mw-predict.py
    (%TEMP%\orca_one_job_at_a_time.lock): held while kind=orca jobs run (across a whole grid batch), never
    while a kind=command job runs (the FTMW script takes it itself); a live foreign holder is waited for
  * resume: kind=orca whose <basename>.out says "ORCA TERMINATED NORMALLY" and whose <basename>.inp.sha1
    sidecar matches the current .inp is marked "reused" without starting ORCA; an existing .out is never
    overwritten (renamed to <basename>.out.<YYYYmmdd-HHMMSS>.bak first)
  * every job runs in a Windows Job Object (KILL_ON_JOB_CLOSE | BREAKAWAY_OK) at below-normal priority;
    a kill also terminates every descendant found in a Toolhelp32 snapshot (MPI processes that broke away)
  * timeout_s and stall_s (no file change for that long) are enforced every 2 s; Ctrl+C or any error
    kills every running job, puts its spec back into jobs\, releases the lock and exits
  * launch-rate guard: process starts are counted exactly (Job Object accounting, TotalProcesses) plus any
    orca*-named process outside our jobs; > 240 starts in 60 s while grid jobs run -> grid concurrency 4 -> 2
    (persisted in guard_state.json; never raised again automatically)
  * foreign ORCA: before a serial job or a new grid batch, orca*.exe processes of another driver (one that
    does not take the lock) are waited for (retry every 60 s), unless --allow-foreign-orca

Usage (any cwd; all paths absolute)
  python campaign_queue.py --status                 table of pending/running/done/failed + wall per project
  python campaign_queue.py --dry-run                validate every spec, print the execution order; exit 2 if invalid
  python campaign_queue.py --once                   drain the queue, then exit (Commander only)
  python campaign_queue.py                          daemon: polls jobs\ every 30 s forever (Commander only)
  python campaign_queue.py --enqueue spec.json      validate + write atomically (file or inline JSON); same spec = no-op
  python campaign_queue.py --retry ID               move a failed job back to jobs\
  options: --queue-dir PATH (default queue_orchestrator), --max-ranks N (1-8), --only PROJECT,
           --allow-foreign-orca (do not wait for orca*.exe processes started by other drivers)
  files in the queue folder: PAUSE (no new launches while it exists), STOP (no new launches; exit when idle)

Exit codes: 0 ok; 1 a job failed in this --once run; 2 invalid spec / refused enqueue; 3 another runner is running.
Test-only environment overrides (defaults are the production values): CAMPAIGN_QUEUE_ORCA (orca.exe path),
CAMPAIGN_QUEUE_LOCK_FILE (lock path).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import IO, Any

# ==============================================================================
# 0. CONSTANTS
# ==============================================================================
DEFAULT_QUEUE_DIR = Path(os.environ.get("COCHEM_QUEUE_DIR", str(Path(__file__).resolve().parent)))
DEFAULT_ORCA_BIN = Path(r"C:\ORCA_6.1.1\orca.exe")
ORCA_BIN = Path(os.environ.get("CAMPAIGN_QUEUE_ORCA") or DEFAULT_ORCA_BIN)        # override: tests only
DEFAULT_LOCK_FILE = Path(tempfile.gettempdir()) / "orca_one_job_at_a_time.lock"    # same as goat_isomer_search.py
LOCK_FILE: Path = Path(os.environ.get("CAMPAIGN_QUEUE_LOCK_FILE") or DEFAULT_LOCK_FILE)

PROJECTS = ("hcooh-hsh", "hcooh-nno", "shared")
KINDS = ("orca", "command")
MAX_RANKS_CAP = 8                     # campaign budget: never more MPI ranks in flight than this
GRID_GROUP = "grid"
GRID_MAX_RANKS = 2
GRID_CONCURRENCY = 4
GRID_CONCURRENCY_REDUCED = 2
LAUNCH_LIMIT = 240                    # process starts per LAUNCH_WINDOW_S (= 4 per second)
LAUNCH_WINDOW_S = 60.0
POLL_S = 2.0                          # running jobs: exit / timeout / launch-rate check
SCAN_EVERY_S = 30.0                   # jobs\ rescan (also right after any job finishes)
LOCK_RETRY_S = 60.0                   # also the retry interval while foreign ORCA processes run
FOREIGN_SETTLE_S = 3.0                # re-check after this before waiting for foreign ORCA processes
STALL_CHECK_EVERY_S = 30.0            # or stall_s / 3 if that is shorter (min POLL_S)
HEARTBEAT_EVERY_S = 30.0
COMMAND_SCAN_DEPTH = 3                # FTMW script writes into <work-dir>\isomer-NN\<stage>\
TERMINATED_NORMALLY = "ORCA TERMINATED NORMALLY"
LOCK_BUSY_TEXT = "Another ORCA driver is running"
MAX_LOCK_REQUEUES = 5
OK_OUTCOMES = ("ok", "reused")
STATES = ("pending", "running", "done", "failed")
ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
NAME_RE = re.compile(r"^(\d{3})_([A-Za-z0-9_.-]+)\.json$")

REQUIRED_FIELDS: dict[str, tuple[type, ...]] = {
    "id": (str,), "project": (str,), "kind": (str,), "priority": (int,), "ranks": (int,),
    "timeout_s": (int, float), "stall_s": (int, float), "cwd": (str,), "description": (str,)}
KIND_FIELDS: dict[str, dict[str, tuple[type, ...]]] = {"orca": {"basename": (str,)}, "command": {"command": (list,)}}
OPTIONAL_FIELDS: dict[str, tuple[type, ...]] = {
    "concurrency_group": (str, type(None)), "depends_on": (list,), "meta": (dict,), "est_wall_s": (int, float)}
RUNTIME_FIELDS = ("started", "started_epoch", "finished", "wall_s", "exit_code", "outcome", "pid", "inp_sha1",
                  "output", "note", "proc_starts", "requeued")
CAL_COLUMNS = ("id", "kind", "ranks", "wall_s", "outcome", "finished")

_IS_WIN = os.name == "nt"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def override_warning() -> str:
    """Non-empty when a test-only environment override replaces a production default."""
    notes = [f"{name}={val}" for name, val, default in (("CAMPAIGN_QUEUE_ORCA", ORCA_BIN, DEFAULT_ORCA_BIN),
                                                        ("CAMPAIGN_QUEUE_LOCK_FILE", LOCK_FILE, DEFAULT_LOCK_FILE))
             if val != default]
    return ("WARNING test-only override active: " + ", ".join(notes)) if notes else ""


def fmt_wall(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "-"
    s = float(seconds)
    if s < 120:
        return f"{s:.1f} s"
    if s < 7200:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.2f} h"


# ==============================================================================
# 1. QUEUE FOLDER, ATOMIC FILE HELPERS, LOG
# ==============================================================================
@dataclass
class Queue:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.jobs = self.root / "jobs"
        self.running = self.root / "running"
        self.done = self.root / "done"
        self.failed = self.root / "failed"
        self.logs = self.root / "logs"
        self.log_file = self.root / "queue_log.txt"
        self.calibration_csv = self.root / "calibration.csv"
        self.heartbeat = self.root / "HEARTBEAT"
        self.runner_pid = self.root / "runner.pid"
        self.guard_state = self.root / "guard_state.json"
        self.pause_file = self.root / "PAUSE"
        self.stop_file = self.root / "STOP"

    def ensure(self) -> None:
        for d in (self.jobs, self.running, self.done, self.failed, self.logs):
            d.mkdir(parents=True, exist_ok=True)

    def folder(self, state: str) -> Path:
        return {"pending": self.jobs, "running": self.running, "done": self.done, "failed": self.failed}[state]


def spec_files(folder: Path) -> list[Path]:
    """*.json in a folder, sorted by name; *.tmp (being written by an agent) and dot-files are ignored."""
    try:
        names = sorted(e.name for e in os.scandir(folder) if e.is_file() and e.name.endswith(".json")
                       and not e.name.startswith("."))
    except OSError:
        return []
    return [folder / n for n in names]


def _retry(fn, attempts: int = 20, delay: float = 0.25):
    """Windows: a file another process has open briefly cannot be renamed (WinError 5/32) - retry."""
    for i in range(attempts):
        try:
            return fn()
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def safe_replace(src: Path, dst: Path, attempts: int = 20) -> bool:
    try:
        _retry(lambda: os.replace(src, dst), attempts)
        return True
    except OSError:
        return False


def write_json(path: Path, data: dict) -> None:
    """Atomic: <path>.tmp then os.replace (readers never see half a file)."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _retry(lambda: os.replace(tmp, path))


def read_json(path: Path) -> Any:
    return _retry(lambda: json.loads(path.read_text(encoding="utf-8")), attempts=8)


def write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    _retry(lambda: os.replace(tmp, path))


def qlog(q: Queue, job_id: str, event: str, wall: Any = None, detail: str = "") -> None:
    """One line per event in queue_log.txt: ISO timestamp, id, event, wall [, detail]."""
    parts = [now_iso(), job_id or "-", event, fmt_wall(wall) if wall is not None else "-"]
    if detail:
        parts.append(detail)
    line = "\t".join(parts)
    print(line, flush=True)
    try:
        with open(q.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def strip_runtime(spec: dict, keep_requeued: bool = True) -> dict:
    out = {k: v for k, v in spec.items() if k not in RUNTIME_FIELDS}
    if keep_requeued and "requeued" in spec:
        out["requeued"] = spec["requeued"]
    return out


def job_fields(spec: dict) -> dict:
    """The part of a spec an agent wrote (runtime fields removed): used to compare re-enqueues."""
    return json.loads(json.dumps(strip_runtime(spec, keep_requeued=False), sort_keys=True))


# ==============================================================================
# 2. SPEC VALIDATION
# ==============================================================================
def _typed(value: Any, types: tuple[type, ...]) -> bool:
    if isinstance(value, bool) and bool not in types:
        return False                                  # JSON true/false is not an int here
    return isinstance(value, types)


def _tname(types: tuple[type, ...]) -> str:
    return "|".join("null" if t is type(None) else {"str": "string", "int": "integer", "float": "number",
                                                       "list": "list", "dict": "object"}.get(t.__name__, t.__name__)
                    for t in types)


def is_grid(spec: dict) -> bool:
    return spec.get("concurrency_group") == GRID_GROUP


def inp_nprocs(text: str) -> int:
    """MPI processes an ORCA input asks for: %pal nprocs N (any layout) or ! PALn; absent -> 1; several -> max."""
    body = re.sub(r"#[^\n]*", "", text)
    found: list[int] = []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("!"):
            found += [int(n) for n in re.findall(r"(?i)(?<![A-Za-z0-9_-])PAL(\d+)\b", s)]
    for block in re.finditer(r"(?is)%pal\b(.*?)\bend\b", body):
        found += [int(n) for n in re.findall(r"(?i)\bnprocs\s+(\d+)", block.group(1))]
    return max(found) if found else 1


def _abs_path(p: str) -> bool:
    return PureWindowsPath(p).is_absolute() if _IS_WIN else Path(p).is_absolute()


def validate_spec(spec: Any, file_name: str | None = None, check_files: bool = True) -> list[str]:
    """Problems with one job spec (empty list = valid). Duplicate ids are checked by the queue scan."""
    if not isinstance(spec, dict):
        return ["spec is not a JSON object"]
    problems: list[str] = []
    for k, types in REQUIRED_FIELDS.items():
        if k not in spec:
            problems.append(f"missing field '{k}'")
        elif not _typed(spec[k], types):
            problems.append(f"field '{k}' must be {_tname(types)}, got {type(spec[k]).__name__}")
    kind = spec.get("kind")
    if isinstance(kind, str) and kind not in KINDS:
        problems.append(f"kind must be 'orca' or 'command', got {kind!r}")
    kind_fields = KIND_FIELDS.get(kind, {}) if isinstance(kind, str) else {}
    for k, types in kind_fields.items():
        if k not in spec:
            problems.append(f"missing field '{k}' (required for kind={kind})")
        elif not _typed(spec[k], types):
            problems.append(f"field '{k}' must be {_tname(types)}, got {type(spec[k]).__name__}")
    for k, types in OPTIONAL_FIELDS.items():
        if k in spec and not _typed(spec[k], types):
            problems.append(f"field '{k}' must be {_tname(types)}, got {type(spec[k]).__name__}")
    allowed = set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS) | set(RUNTIME_FIELDS) | set(kind_fields)
    for k in spec:
        if k not in allowed:
            if any(k in f for f in KIND_FIELDS.values()):
                problems.append(f"field '{k}' is not allowed for kind={kind}")
            else:
                problems.append(f"unknown field '{k}' (free-form notes go under \"meta\")")
    if problems:
        return problems                               # the checks below need well-typed fields

    jid, ranks, prio = spec["id"], spec["ranks"], spec["priority"]
    if not ID_RE.match(jid):
        problems.append(f"id {jid!r} must match ^[A-Za-z0-9_.-]+$")
    if spec["project"] not in PROJECTS:
        problems.append(f"project must be one of {', '.join(PROJECTS)}, got {spec['project']!r}")
    if not 1 <= ranks <= MAX_RANKS_CAP:
        problems.append(f"ranks must be 1-{MAX_RANKS_CAP}, got {ranks}")
    if not 0 <= prio <= 999:
        problems.append(f"priority must be 0-999 (3-digit file prefix), got {prio}")
    for k in ("timeout_s", "stall_s"):
        if not spec[k] > 0:
            problems.append(f"{k} must be > 0, got {spec[k]}")
    if "est_wall_s" in spec and not spec["est_wall_s"] > 0:
        problems.append("est_wall_s must be > 0")
    if file_name is not None and 0 <= prio <= 999:
        expected = f"{prio:03d}_{jid}.json"
        if file_name != expected:
            problems.append(f"file name {file_name!r} must be {expected!r} (priority:03d + '_' + id + '.json')")
    group = spec.get("concurrency_group")
    if group not in (None, GRID_GROUP):
        problems.append(f"concurrency_group must be \"grid\" or absent, got {group!r}")
    if group == GRID_GROUP and ranks > GRID_MAX_RANKS:
        problems.append(f"grid jobs may use at most {GRID_MAX_RANKS} ranks, got {ranks}")
    for dep in spec.get("depends_on", []):
        if not isinstance(dep, str) or not ID_RE.match(dep):
            problems.append(f"depends_on entry {dep!r} is not a job id")
        elif dep == jid:
            problems.append("a job cannot depend on itself")

    cwd = spec["cwd"]
    cwd_ok = _abs_path(cwd)
    if not cwd_ok:
        problems.append(f"cwd must be an absolute path, got {cwd!r}")
    elif check_files and not Path(cwd).is_dir():
        problems.append(f"cwd does not exist: {cwd}")
        cwd_ok = False
    if kind == "orca":
        base = spec["basename"]
        if not ID_RE.match(base):
            problems.append(f"basename {base!r} must be a plain file stem (^[A-Za-z0-9_.-]+$)")
        elif cwd_ok and check_files:
            inp = Path(cwd) / f"{base}.inp"
            if not inp.is_file():
                problems.append(f"input not found: {inp} (write the .inp before enqueuing)")
            else:
                try:
                    n = inp_nprocs(inp.read_text(encoding="utf-8", errors="replace"))
                except OSError as exc:
                    problems.append(f"cannot read {inp}: {exc}")
                else:
                    if n != ranks:
                        problems.append(f"ranks={ranks} but {inp.name} asks for {n} MPI process(es) "
                                        f"(%pal nprocs / ! PALn; absent = 1); they must match")
    elif kind == "command":
        cmd = spec["command"]
        if not cmd or not all(isinstance(c, str) for c in cmd):
            problems.append("command must be a non-empty list of strings")
        elif not _abs_path(cmd[0]):
            problems.append(f"command[0] must be an absolute path to a program, got {cmd[0]!r}")
        elif check_files and not Path(cmd[0]).is_file():
            problems.append(f"command[0] does not exist: {cmd[0]}")
    return problems


def load_spec(path: Path, check_files: bool = True) -> tuple[dict | None, list[str]]:
    try:
        spec = read_json(path)
    except (OSError, ValueError) as exc:
        return None, [f"cannot read JSON: {exc}"]
    return (spec if isinstance(spec, dict) else None), validate_spec(spec, path.name, check_files)


# ==============================================================================
# 3. QUEUE SCAN, DEPENDENCIES, SCHEDULING (pure functions)
# ==============================================================================
@dataclass
class Entry:
    path: Path
    state: str
    spec: dict | None
    problems: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def id(self) -> str:
        if self.spec and isinstance(self.spec.get("id"), str):
            return self.spec["id"]
        m = NAME_RE.match(self.path.name)
        return m.group(2) if m else self.path.stem

    @property
    def valid(self) -> bool:
        return self.spec is not None and not self.problems

    def sort_key(self) -> tuple:
        p = self.spec.get("priority") if self.spec else None
        return (p if isinstance(p, int) and not isinstance(p, bool) else 10 ** 6, self.name)


_VALIDATION_CACHE: dict[str, tuple[tuple, dict | None, list[str]]] = {}


def _stamp(path: Path) -> tuple:
    try:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return (None, None)


def load_pending(path: Path) -> Entry:
    """Pending spec, validated; cached on (spec stamp, input stamp) so a 30-s rescan of ~1000 specs is cheap."""
    key = str(path)
    stamp = _stamp(path)
    cached = _VALIDATION_CACHE.get(key)
    if cached and cached[0][0] == stamp:
        spec = cached[1]
        inp_stamp = _stamp(_inp_path(spec)) if spec else None
        if cached[0][1] == inp_stamp:
            return Entry(path, "pending", spec, list(cached[2]))
    spec, problems = load_spec(path)
    if spec is not None and not problems:            # invalid specs are re-checked every scan (may get fixed)
        _VALIDATION_CACHE[key] = ((stamp, _stamp(_inp_path(spec))), spec, [])
    else:
        _VALIDATION_CACHE.pop(key, None)
    return Entry(path, "pending", spec, problems)


def _inp_path(spec: dict | None) -> Path:
    if spec and spec.get("kind") == "orca" and isinstance(spec.get("cwd"), str) and isinstance(spec.get("basename"), str):
        return Path(spec["cwd"]) / f"{spec['basename']}.inp"
    return Path("__no_input__")


@dataclass
class Scan:
    pending: list[Entry]
    running: list[Entry]
    done: list[Entry]
    failed: list[Entry]

    def ids(self, state: str) -> set[str]:
        return {e.id for e in getattr(self, state)}


def scan_queue(q: Queue, read_finished: bool = False) -> Scan:
    """
    All specs by state. Pending ones are validated (duplicate ids across all four folders are invalid).
    Running/done/failed are identified by file name unless read_finished (status needs their contents).
    """
    pending = [load_pending(p) for p in spec_files(q.jobs)]
    others: dict[str, list[Entry]] = {}
    for state in ("running", "done", "failed"):
        entries = []
        for p in spec_files(q.folder(state)):
            spec = None
            if read_finished or state == "running":
                try:
                    spec = read_json(p)
                except (OSError, ValueError):
                    spec = None
            entries.append(Entry(p, state, spec if isinstance(spec, dict) else None))
        others[state] = entries
    where: dict[str, list[str]] = {}
    for state, entries in [("pending", pending)] + list(others.items()):
        for e in entries:
            where.setdefault(e.id, []).append(f"{q.folder(state).name}\\{e.name}")
    for e in pending:
        places = where.get(e.id, [])
        if len(places) > 1:
            e.problems = e.problems + [f"duplicate id {e.id!r}: " + ", ".join(places)]
    return Scan(pending, others["running"], others["done"], others["failed"])


@dataclass
class Classified:
    ready: list[Entry]
    waiting: list[tuple[Entry, str]]
    blocked: list[tuple[Entry, str]]
    skipped: list[tuple[Entry, str]]


def classify(pending_valid: list[Entry], done_ids: set[str], failed_ids: set[str], running_ids: set[str],
             invalid_ids: set[str] = frozenset(), only: str | None = None,
             max_ranks: int = MAX_RANKS_CAP) -> Classified:
    r"""
    Ready = valid pending specs whose depends_on are all in done\ (ok/reused), sorted by priority then file
    name. A job that (transitively) depends on a failed job is blocked and never runs.
    """
    by_id = {e.id: e for e in pending_valid}
    memo: dict[str, str] = {}

    def state_of(jid: str, stack: tuple = ()) -> str:
        if jid in done_ids:
            return "done"
        if jid in failed_ids:
            return "failed"
        if jid in running_ids:
            return "running"
        if jid in memo:
            return memo[jid]
        if jid in stack:
            return "cycle"
        if jid in by_id:
            sub = [state_of(d, stack + (jid,)) for d in by_id[jid].spec.get("depends_on", [])]
            res = "blocked" if any(s in ("failed", "blocked", "cycle") for s in sub) else "pending"
            memo[jid] = res
            return res
        return "invalid" if jid in invalid_ids else "unknown"

    out = Classified([], [], [], [])
    for e in sorted(pending_valid, key=Entry.sort_key):
        deps = e.spec.get("depends_on", [])
        states = {d: state_of(d, (e.id,)) for d in deps}
        bad = [f"{d} {s}" for d, s in states.items() if s in ("failed", "blocked", "cycle")]
        if bad:
            out.blocked.append((e, "dependency " + ", ".join(bad)))
            continue
        open_deps = [f"{d} ({s})" for d, s in states.items() if s != "done"]
        if open_deps:
            out.waiting.append((e, "waits for " + ", ".join(open_deps)))
        elif only and e.spec["project"] != only:
            out.skipped.append((e, f"--only {only}"))
        elif e.spec["ranks"] > max_ranks:
            out.waiting.append((e, f"ranks {e.spec['ranks']} > --max-ranks {max_ranks}"))
        else:
            out.ready.append(e)
    return out


def plan_launches(ready: list[dict], running: list[dict], max_ranks: int = MAX_RANKS_CAP,
                  grid_concurrency: int = GRID_CONCURRENCY) -> list[dict]:
    """
    Which ready specs to start now (pure). Head serial job: only when nothing runs, alone. Head grid job:
    consecutive ready grid jobs from the head, up to the grid concurrency and the rank cap; never next to
    a running serial job.
    """
    if not ready:
        return []
    head = ready[0]
    if not is_grid(head):
        return [head] if not running and head["ranks"] <= max_ranks else []
    if any(not is_grid(r) for r in running):
        return []
    slots = grid_concurrency - len(running)
    used = sum(r["ranks"] for r in running)
    launch = []
    for spec in ready:
        if slots <= 0 or not is_grid(spec) or used + spec["ranks"] > max_ranks:
            break
        launch.append(spec)
        used += spec["ranks"]
        slots -= 1
    return launch


def guard_decision(events: list[tuple[float, int]], now: float, concurrency: int, limit: int = LAUNCH_LIMIT,
                   window_s: float = LAUNCH_WINDOW_S,
                   reduced: int = GRID_CONCURRENCY_REDUCED) -> tuple[int, int, str | None]:
    """
    Launch-rate guard (pure). events = (time, number of new process starts) samples. Returns
    (new grid concurrency, starts in the last window_s, WARNING text or None). Trips when the starts in
    the sliding window exceed the limit; only ever lowers the concurrency.
    """
    starts = sum(n for t, n in events if now - window_s < t <= now)
    if starts > limit and concurrency > reduced:
        return reduced, starts, (f"WARNING launch-rate guard: {starts} process starts in the last {window_s:.0f} s "
                                 f"(limit {limit}); grid concurrency {concurrency} -> {reduced} "
                                 f"(delete guard_state.json to reset)")
    return concurrency, starts, None


# ==============================================================================
# 4. RESUME (input fingerprint + "ORCA TERMINATED NORMALLY")
# ==============================================================================
def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_path(cwd: Path, base: str) -> Path:
    return cwd / f"{base}.inp.sha1"


def terminated_normally(out: Path) -> bool:
    """ORCA prints the line at the very end; read only the last 256 KB of a possibly large .out."""
    try:
        with open(out, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 262144))
            return TERMINATED_NORMALLY.encode() in f.read()
    except OSError:
        return False


def can_reuse(cwd: Path, base: str) -> tuple[bool, str]:
    """(True, why) when <base>.out finished normally for exactly the current <base>.inp."""
    inp, out, side = cwd / f"{base}.inp", cwd / f"{base}.out", sidecar_path(cwd, base)
    if not inp.is_file():
        return False, "no input"
    if not out.is_file():
        return False, "no output yet"
    if not terminated_normally(out):
        return False, "output did not terminate normally"
    try:
        recorded = side.read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError):
        return False, "no input fingerprint (.inp.sha1)"
    if recorded != sha1_file(inp):
        return False, "input changed since the finished run"
    return True, "finished output for the unchanged input"


def backup_out(out: Path) -> Path | None:
    """Never overwrite an .out: rename it to <name>.out.<YYYYmmdd-HHMMSS>.bak."""
    if not out.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dst = out.with_name(f"{out.name}.{stamp}.bak")
    k = 1
    while dst.exists():
        dst = out.with_name(f"{out.name}.{stamp}-{k}.bak")
        k += 1
    _retry(lambda: os.replace(out, dst))
    return dst


# ==============================================================================
# 5. MACHINE-WIDE ORCA LOCK (as goat_isomer_search.py; waits instead of SystemExit)
# ==============================================================================
class LockBusy(RuntimeError):
    def __init__(self, info: dict) -> None:
        self.info = info
        super().__init__(f"{LOCK_BUSY_TEXT} (PID {info.get('pid', '?')}: {info.get('what', '?')}, since "
                         f"{info.get('started', '?')})")


def pid_alive(pid: int) -> bool:
    """True if a process with this PID exists (never signals or kills it)."""
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5                     # access denied: it exists
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259                       # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def lock_holder() -> dict | None:
    """The lock's JSON if a live process other than this one holds it."""
    try:
        info = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        pid = int(info.get("pid", -1))
    except (OSError, ValueError, AttributeError):
        return None
    return info if pid > 0 and pid != os.getpid() and pid_alive(pid) else None


def acquire_orca_lock(what: str) -> None:
    """
    Machine-wide 'one ORCA driver at a time' lock (same file and JSON as goat_isomer_search.py and
    hcooh-hsh-mw-predict.py). A lock left by a crashed run (its PID no longer exists) is taken over.
    Raises LockBusy (never SystemExit) while another live driver holds it.
    """
    if LOCK_FILE.exists():
        try:
            info = json.loads(LOCK_FILE.read_text(encoding="utf-8"))
            pid = int(info.get("pid", -1))
        except (OSError, ValueError, AttributeError):
            info, pid = {}, -1
        if pid > 0 and pid != os.getpid() and pid_alive(pid):
            raise LockBusy(info)
        LOCK_FILE.unlink(missing_ok=True)
    try:
        with open(LOCK_FILE, "x", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "what": what, "started": now_iso()}, f)
    except FileExistsError:                                         # another driver won the race
        raise LockBusy(lock_holder() or {"what": "unknown (just created)"})


def release_orca_lock() -> None:
    try:
        if int(json.loads(LOCK_FILE.read_text(encoding="utf-8")).get("pid", -1)) == os.getpid():
            LOCK_FILE.unlink()
    except (OSError, ValueError):
        pass


def hold_lock() -> bool:
    try:
        return int(json.loads(LOCK_FILE.read_text(encoding="utf-8")).get("pid", -1)) == os.getpid()
    except (OSError, ValueError, AttributeError):
        return False


# ==============================================================================
# 6. WINDOWS PROCESS CONTROL (ctypes: Job Objects, Toolhelp32, keep-awake)
# ==============================================================================
_K32 = None
_STRUCTS: dict[str, Any] = {}


def _k32():
    """kernel32 with typed signatures (own WinDLL instance, so nothing else's argtypes are touched)."""
    global _K32
    if _K32 is not None:
        return _K32
    import ctypes
    from ctypes import wintypes as w

    class FILETIME(ctypes.Structure):
        _fields_ = [("dwLowDateTime", w.DWORD), ("dwHighDateTime", w.DWORD)]

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", w.DWORD), ("cntUsage", w.DWORD), ("th32ProcessID", w.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", w.DWORD), ("cntThreads", w.DWORD),
                    ("th32ParentProcessID", w.DWORD), ("pcPriClassBase", w.LONG), ("dwFlags", w.DWORD),
                    ("szExeFile", w.WCHAR * 260)]

    class BasicLimit(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", w.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", w.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [("Basic", BasicLimit), ("IoInfo", ctypes.c_ulonglong * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class Accounting(ctypes.Structure):           # JOBOBJECT_BASIC_ACCOUNTING_INFORMATION (class 1)
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", w.DWORD), ("TotalProcesses", w.DWORD), ("ActiveProcesses", w.DWORD),
                    ("TotalTerminatedProcesses", w.DWORD)]

    k = ctypes.WinDLL("kernel32", use_last_error=True)
    sig = {
        "CreateJobObjectW": (w.HANDLE, [ctypes.c_void_p, w.LPCWSTR]),
        "SetInformationJobObject": (w.BOOL, [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]),
        "QueryInformationJobObject": (w.BOOL, [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD,
                                               ctypes.POINTER(w.DWORD)]),
        "AssignProcessToJobObject": (w.BOOL, [w.HANDLE, w.HANDLE]),
        "TerminateJobObject": (w.BOOL, [w.HANDLE, w.UINT]),
        "IsProcessInJob": (w.BOOL, [w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)]),
        "CloseHandle": (w.BOOL, [w.HANDLE]),
        "OpenProcess": (w.HANDLE, [w.DWORD, w.BOOL, w.DWORD]),
        "TerminateProcess": (w.BOOL, [w.HANDLE, w.UINT]),
        "WaitForSingleObject": (w.DWORD, [w.HANDLE, w.DWORD]),
        "GetProcessTimes": (w.BOOL, [w.HANDLE] + [ctypes.POINTER(FILETIME)] * 4),
        "CreateToolhelp32Snapshot": (w.HANDLE, [w.DWORD, w.DWORD]),
        "Process32FirstW": (w.BOOL, [w.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]),
        "Process32NextW": (w.BOOL, [w.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]),
        "SetThreadExecutionState": (w.DWORD, [w.DWORD]),
        "QueryFullProcessImageNameW": (w.BOOL, [w.HANDLE, w.DWORD, w.LPWSTR, ctypes.POINTER(w.DWORD)]),
    }
    for name, (res, args) in sig.items():
        fn = getattr(k, name)
        fn.restype, fn.argtypes = res, args
    _STRUCTS.update(FILETIME=FILETIME, PROCESSENTRY32W=PROCESSENTRY32W, ExtendedLimit=ExtendedLimit,
                    Accounting=Accounting, ctypes=ctypes, w=w)
    _K32 = k
    return k


_INVALID_HANDLE = (1 << 64) - 1


def process_snapshot() -> list[tuple[int, int, str]]:
    """(pid, parent pid, exe name) of every process (Toolhelp32; ~1 ms; empty off Windows)."""
    if not _IS_WIN:
        return []
    k = _k32()
    ctypes = _STRUCTS["ctypes"]
    snap = k.CreateToolhelp32Snapshot(0x2, 0)                       # TH32CS_SNAPPROCESS
    if not snap or snap == _INVALID_HANDLE:
        return []
    out = []
    try:
        e = _STRUCTS["PROCESSENTRY32W"]()
        e.dwSize = ctypes.sizeof(e)
        ok = k.Process32FirstW(snap, ctypes.byref(e))
        while ok:
            out.append((int(e.th32ProcessID), int(e.th32ParentProcessID), str(e.szExeFile)))
            ok = k.Process32NextW(snap, ctypes.byref(e))
    finally:
        k.CloseHandle(snap)
    return out


def _open(pid: int, access: int):
    return _k32().OpenProcess(access, False, pid) if _IS_WIN else None


def creation_time(pid: int) -> float | None:
    """Process creation time (epoch seconds), None if it cannot be queried."""
    if not _IS_WIN:
        return None
    k = _k32()
    ctypes = _STRUCTS["ctypes"]
    h = _open(pid, 0x1000)                                           # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        ft = [_STRUCTS["FILETIME"]() for _ in range(4)]
        if not k.GetProcessTimes(h, *[ctypes.byref(x) for x in ft]):
            return None
        ticks = (ft[0].dwHighDateTime << 32) | ft[0].dwLowDateTime
        return ticks / 1e7 - 11644473600.0
    finally:
        k.CloseHandle(h)


def image_name(pid: int) -> str | None:
    if not _IS_WIN:
        return None
    k = _k32()
    ctypes, w = _STRUCTS["ctypes"], _STRUCTS["w"]
    h = _open(pid, 0x1000)
    if not h:
        return None
    try:
        buf = ctypes.create_unicode_buffer(1024)
        n = w.DWORD(1024)
        return buf.value if k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)) else None
    finally:
        k.CloseHandle(h)


def descendants(root_pid: int, snapshot: list[tuple[int, int, str]] | None = None) -> list[int]:
    """Every live descendant of root_pid by parent-pid chain (breadth first)."""
    snap = process_snapshot() if snapshot is None else snapshot
    children: dict[int, list[int]] = {}
    for pid, ppid, _ in snap:
        if pid != ppid:
            children.setdefault(ppid, []).append(pid)
    out, todo, seen = [], [root_pid], {root_pid, os.getpid()}
    while todo:
        for c in children.get(todo.pop(0), []):
            if c not in seen:
                seen.add(c)
                out.append(c)
                todo.append(c)
    return out


def snapshot_descendants(root_pid: int) -> list[tuple[int, float]]:
    """Live descendants of root_pid with their creation times, taken now (identity for a later kill)."""
    out = []
    for pid in descendants(root_pid):
        ct = creation_time(pid)
        if ct is not None:
            out.append((pid, ct))
    return out


def terminate_pid(pid: int, not_before: float, expected_ct: float | None = None) -> bool:
    """
    Terminates pid if it was created at/after not_before and, when expected_ct is given, at exactly that
    creation time (from snapshot_descendants): a recycled PID - other processes start all the time on this
    machine - is never killed.
    """
    if not _IS_WIN:
        try:
            os.kill(pid, 9)
            return True
        except OSError:
            return False
    ct = creation_time(pid)
    if ct is not None and ct < not_before - 2.0:
        return False
    if expected_ct is not None and (ct is None or abs(ct - expected_ct) > 0.5):
        return False
    k = _k32()
    h = _open(pid, 0x0001 | 0x1000 | 0x00100000)                    # TERMINATE | QUERY_LIMITED | SYNCHRONIZE
    if not h:
        return False
    try:
        ok = bool(k.TerminateProcess(h, 1))
        k.WaitForSingleObject(h, 5000)
        return ok
    finally:
        k.CloseHandle(h)


def create_kill_job(proc: subprocess.Popen) -> int | None:
    """
    Windows Job Object (KILL_ON_JOB_CLOSE | BREAKAWAY_OK, as hcooh-hsh-mw-predict.py) holding the job's
    root process; everything it starts later joins (MPI may break away, hence the snapshot kill too).
    If the runner dies, the handle closes and the whole job goes with it.
    """
    if not _IS_WIN:
        return None
    k = _k32()
    ctypes = _STRUCTS["ctypes"]
    job = k.CreateJobObjectW(None, None)
    info = _STRUCTS["ExtendedLimit"]()
    info.Basic.LimitFlags = 0x2000 | 0x800
    if job and k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) \
            and k.AssignProcessToJobObject(job, int(proc._handle)):
        return job
    if job:
        k.CloseHandle(job)
    return None


def job_total_processes(job: int | None) -> int | None:
    """Processes ever assigned to the job (TotalProcesses, JobObjectBasicAccountingInformation)."""
    if not job:
        return None
    k = _k32()
    ctypes = _STRUCTS["ctypes"]
    acc = _STRUCTS["Accounting"]()
    if k.QueryInformationJobObject(job, 1, ctypes.byref(acc), ctypes.sizeof(acc), None):
        return int(acc.TotalProcesses)
    return None


def pid_in_jobs(pid: int, jobs: list[int]) -> bool:
    if not _IS_WIN or not jobs:
        return False
    k = _k32()
    ctypes, w = _STRUCTS["ctypes"], _STRUCTS["w"]
    h = _open(pid, 0x1000)
    if not h:
        return False
    try:
        for job in jobs:
            res = w.BOOL(0)
            if k.IsProcessInJob(h, job, ctypes.byref(res)) and res.value:
                return True
        return False
    finally:
        k.CloseHandle(h)


def foreign_orca(snapshot: list[tuple[int, int, str]] | None = None) -> list[tuple[int, str]]:
    """
    orca*.exe processes on this machine as (pid, exe). Asked only while the runner itself runs nothing, so none
    of them is ours: another driver (e.g. a search script that does not take the ORCA lock) is using cores.
    """
    snap = process_snapshot() if snapshot is None else snapshot
    return [(pid, exe) for pid, _, exe in snap if exe.lower().startswith("orca") and pid != os.getpid()]


def close_handle(h: int | None) -> None:
    if h and _IS_WIN:
        _k32().CloseHandle(h)


def keep_awake() -> bool:
    """Idle-sleep request for the lifetime of this process (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""
    if not _IS_WIN:
        return False
    _k32().SetThreadExecutionState(0x80000000 | 0x00000001)
    return True


def newest_mtime(folder: Path, depth: int = 1) -> float:
    """
    Newest modification time in a folder, descending depth-1 levels of subfolders. os.scandir reads the
    times from the directory listing on Windows (no file is opened); vanished entries are skipped.
    """
    newest = 0.0
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    newest = max(newest, entry.stat().st_mtime)
                    if depth > 1 and entry.is_dir(follow_symlinks=False):
                        newest = max(newest, newest_mtime(Path(entry.path), depth - 1))
                except OSError:
                    pass
    except OSError:
        pass
    return newest


def child_env(kind: str) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["PATH"] = str(ORCA_BIN.parent) + os.pathsep + env.get("PATH", "")
    env["OPI_ORCA"] = str(ORCA_BIN.parent)       # OPI-based command jobs find the same ORCA as kind=orca jobs
    if kind == "command":
        env["PYTHONUNBUFFERED"] = "1"            # live log -> stall detection and lock-busy detection
    return env


# ==============================================================================
# 7. ENQUEUE / RETRY (for the agents)
# ==============================================================================
def _find_by_id(q: Queue, jid: str) -> list[tuple[str, Path]]:
    found = []
    for state in STATES:
        for p in spec_files(q.folder(state)):
            m = NAME_RE.match(p.name)
            if (m and m.group(2) == jid) or (not m and p.stem == jid):
                found.append((state, p))
    return found


def enqueue(spec: dict, queue_dir: Path | str = DEFAULT_QUEUE_DIR, replace_pending: bool = False) -> tuple[str, str]:
    r"""
    Validates a spec and writes jobs\<priority:03d>_<id>.json atomically (.tmp + os.replace).
    Returns (status, message); status: "enqueued", "exists" (identical job already queued/running/finished:
    no-op), "replaced" (replace_pending=True and a pending spec with this id differed), "conflict", "invalid".
    """
    q = Queue(Path(queue_dir))
    q.ensure()
    problems = validate_spec(spec)
    if problems:
        return "invalid", "invalid spec: " + "; ".join(problems)
    spec = strip_runtime(spec, keep_requeued=False)
    name = f"{spec['priority']:03d}_{spec['id']}.json"
    existing = _find_by_id(q, spec["id"])
    old_pending: list[Path] = []
    for state, path in existing:
        try:
            other = read_json(path)
        except (OSError, ValueError):
            other = None
        same = isinstance(other, dict) and job_fields(other) == job_fields(spec)
        if same:
            extra = f" (outcome {other.get('outcome')})" if state in ("done", "failed") else ""
            hint = "; to retry: campaign_queue.py --retry " + spec["id"] if state == "failed" else ""
            return "exists", f"{spec['id']}: identical job already {state}{extra}: {path} - nothing to do{hint}"
        if state == "pending" and replace_pending:
            old_pending.append(path)
            continue
        return "conflict", (f"{spec['id']}: id already used by a different job in {state}: {path} "
                            f"(choose a new id{'; or replace_pending=True' if state == 'pending' else ''})")
    target = q.jobs / name
    write_json(target, spec)
    for p in old_pending:
        if p != target:
            p.unlink(missing_ok=True)
    return ("replaced" if old_pending else "enqueued"), f"{spec['id']}: {'replaced' if old_pending else 'enqueued'} {target}"


def _move_back_to_jobs(src: Path, q: Queue, reset_requeued: bool) -> Path:
    r"""running\ or failed\ -> jobs\ without a moment where the id is visible twice (.tmp is ignored)."""
    tmp = q.jobs / (src.name + ".tmp")
    _retry(lambda: os.replace(src, tmp))
    try:
        spec = read_json(tmp)
        spec = strip_runtime(spec, keep_requeued=not reset_requeued) if isinstance(spec, dict) else spec
        tmp.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError):
        pass
    dst = q.jobs / src.name
    _retry(lambda: os.replace(tmp, dst))
    return dst


def retry_failed(q: Queue, jid: str) -> tuple[bool, str]:
    hits = [(s, p) for s, p in _find_by_id(q, jid)]
    failed = [p for s, p in hits if s == "failed"]
    if not failed:
        return False, f"{jid}: not in failed\\ ({', '.join(s for s, _ in hits) or 'not found'})"
    if len(hits) > 1:
        return False, f"{jid}: id also in " + ", ".join(f"{s}\\{p.name}" for s, p in hits if s != "failed")
    dst = _move_back_to_jobs(failed[0], q, reset_requeued=True)
    qlog(q, jid, "retry (failed -> jobs)")
    return True, f"{jid}: moved back to {dst}"


# ==============================================================================
# 8. THE RUNNER
# ==============================================================================
@dataclass
class RunningJob:
    spec: dict
    name: str
    path: Path                              # running\<name>
    proc: subprocess.Popen
    handle: int | None
    t0: float
    out_fh: IO[bytes]
    out_path: Path
    inp_sha1: str | None = None
    log_offset: int = 0
    total_procs: int = 0
    next_stall_check: float = 0.0
    rate_warned: bool = False

    @property
    def id(self) -> str:
        return self.spec["id"]


class Runner:
    def __init__(self, q: Queue, max_ranks: int = MAX_RANKS_CAP, only: str | None = None, once: bool = False,
                 allow_foreign: bool = False):
        self.q, self.max_ranks, self.only, self.once = q, max_ranks, only, once
        self.allow_foreign = allow_foreign
        self.foreign_logged: tuple | None = None
        self.running: dict[str, RunningJob] = {}
        self.grid_concurrency = GRID_CONCURRENCY
        self.events: list[tuple[float, int]] = []      # (time, new process starts)
        self.seen_orca: set[int] = set()
        self.failures = 0
        self.invalid_logged: dict[str, str] = {}
        self.lock_holder_logged: tuple | None = None
        self.last_heartbeat = 0.0
        self.starts_60s = 0

    # ---- single runner -------------------------------------------------------------------------
    def claim_runner(self) -> str | None:
        """None if this process is now the runner, else a message naming the live runner."""
        f = self.q.runner_pid
        if f.exists():
            try:
                info = json.loads(f.read_text(encoding="utf-8"))
                pid = int(info.get("pid", -1))
            except (OSError, ValueError, AttributeError):
                info, pid = {}, -1
            if pid > 0 and pid != os.getpid() and pid_alive(pid):
                exe = image_name(pid)
                if exe is None or "python" in Path(exe).name.lower():
                    return (f"another campaign_queue runner is running (PID {pid}, since {info.get('started', '?')}); "
                            f"if it is not, delete {f}")
            f.unlink(missing_ok=True)
        try:
            with open(f, "x", encoding="utf-8") as fh:
                json.dump({"pid": os.getpid(), "started": now_iso(), "exe": sys.executable,
                           "mode": "once" if self.once else "daemon"}, fh)
        except FileExistsError:
            return f"another campaign_queue runner just started ({f})"
        return None

    def release_runner(self) -> None:
        try:
            if int(json.loads(self.q.runner_pid.read_text(encoding="utf-8")).get("pid", -1)) == os.getpid():
                self.q.runner_pid.unlink()
        except (OSError, ValueError):
            pass

    # ---- start-up ------------------------------------------------------------------------------
    def requeue_leftovers(self) -> None:
        r"""Specs left in running\ by a crashed runner go back to jobs\ (their orphans are killed first)."""
        for p in spec_files(self.q.running):
            try:
                spec = read_json(p)
            except (OSError, ValueError):
                spec = {}
            if not isinstance(spec, dict):
                spec = {}
            pid, t0 = spec.get("pid"), spec.get("started_epoch")
            note = ""
            if isinstance(pid, int) and isinstance(t0, (int, float)) and pid_alive(pid):
                ct = creation_time(pid)
                if ct is not None and abs(ct - t0) < 10:
                    for d, dct in snapshot_descendants(pid):
                        terminate_pid(d, t0, dct)
                    terminate_pid(pid, t0, ct)
                    note = f"killed orphan PID {pid} of the crashed run"
            jid = spec.get("id", p.stem)
            dst = _move_back_to_jobs(p, self.q, reset_requeued=False)
            qlog(self.q, jid, "requeued", None, f"left in running\\ by a stopped runner -> {dst.name}"
                 + (f"; {note}" if note else ""))

    def load_guard_state(self) -> None:
        try:
            st = json.loads(self.q.guard_state.read_text(encoding="utf-8"))
            c = int(st.get("grid_concurrency", GRID_CONCURRENCY))
            if c < self.grid_concurrency:
                self.grid_concurrency = max(1, c)
                qlog(self.q, "-", "guard", None, f"grid concurrency {self.grid_concurrency} (guard tripped "
                     f"{st.get('tripped', '?')}; delete {self.q.guard_state.name} to reset)")
        except (OSError, ValueError, AttributeError):
            pass

    # ---- lock ----------------------------------------------------------------------------------
    def _wait(self, seconds: float) -> bool:
        """Sleeps in POLL_S steps; False if STOP appeared."""
        end = time.time() + seconds
        while time.time() < end:
            if self.q.stop_file.exists():
                return False
            self.heartbeat()
            time.sleep(min(POLL_S, max(0.0, end - time.time())))
        return True

    def wait_for_lock(self, what: str, keep: bool) -> str:
        """
        Acquires the machine-wide lock (keep=False: releases it again at once, i.e. 'wait until free').
        Nothing running: waits, retrying every 60 s, logging once per holder -> 'ok' or 'stopped' (STOP).
        Jobs running: one attempt only, so their timeouts stay enforced -> 'ok' or 'busy'.
        """
        if hold_lock():
            if not keep:
                release_orca_lock()
            return "ok"
        while True:
            try:
                acquire_orca_lock(what)
                if self.lock_holder_logged:
                    qlog(self.q, "-", "lock-free", None, "ORCA lock acquired after waiting")
                self.lock_holder_logged = None
                if not keep:
                    release_orca_lock()
                return "ok"
            except LockBusy as busy:
                key = (busy.info.get("pid"), busy.info.get("started"))
                if key != self.lock_holder_logged:
                    self.lock_holder_logged = key
                    qlog(self.q, "-", "lock-wait", None, f"{busy}; retrying every {LOCK_RETRY_S:.0f} s")
            if self.running:
                return "busy"
            if not self._wait(LOCK_RETRY_S):
                return "stopped"

    def wait_for_foreign_orca(self) -> str:
        """
        Before a serial job or the first job of a grid batch (nothing of ours runs): while orca*.exe processes of
        another driver are running (one that does not take the lock), wait and retry every 60 s, logging once per
        set of PIDs -> 'ok' or 'stopped' (STOP). Keeps the 8-rank budget and the calibration timings honest.
        --allow-foreign-orca switches this off.
        """
        if self.allow_foreign or self.running or not _IS_WIN:
            return "ok"
        procs = foreign_orca()
        if procs:                                   # a job that just ended may still be exiting
            time.sleep(FOREIGN_SETTLE_S)
            procs = foreign_orca()
        while procs:
            key = tuple(sorted(pid for pid, _ in procs))
            if key != self.foreign_logged:
                self.foreign_logged = key
                names = sorted({exe for _, exe in procs})
                qlog(self.q, "-", "foreign-orca-wait", None,
                     f"{len(procs)} ORCA process(es) not started by this queue are running (PIDs "
                     f"{', '.join(map(str, key[:12]))}: {', '.join(names[:4])}); waiting, retry every "
                     f"{LOCK_RETRY_S:.0f} s (--allow-foreign-orca runs anyway)")
            if not self._wait(LOCK_RETRY_S):
                return "stopped"
            procs = foreign_orca()
        if self.foreign_logged:
            self.foreign_logged = None
            qlog(self.q, "-", "foreign-orca-gone", None, "no foreign ORCA process left")
        return "ok"

    # ---- heartbeat -----------------------------------------------------------------------------
    def heartbeat(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self.last_heartbeat < HEARTBEAT_EVERY_S:
            return
        self.last_heartbeat = now
        lines = [now_iso(), f"pid {os.getpid()}", f"mode {'once' if self.once else 'daemon'}",
                 f"running {', '.join(self.running) or '-'}",
                 f"ranks in flight {sum(j.spec['ranks'] for j in self.running.values())}/{self.max_ranks}",
                 f"grid concurrency {self.grid_concurrency}", f"process starts in last 60 s {self.starts_60s}",
                 f"holds ORCA lock {hold_lock()}"]
        try:
            write_text_atomic(self.q.heartbeat, "\n".join(lines) + "\n")
        except OSError:
            pass

    # ---- launching -----------------------------------------------------------------------------
    def finish_reused(self, entry: Entry, why: str) -> None:
        spec = dict(entry.spec)
        t = now_iso()
        cwd = Path(spec["cwd"])
        spec.update(started=t, finished=t, wall_s=0.0, exit_code=None, outcome="reused", pid=None,
                    inp_sha1=sha1_file(cwd / f"{spec['basename']}.inp"),
                    output=str(cwd / f"{spec['basename']}.out"), note=why)
        dst = self.q.done / entry.name
        tmp = self.q.jobs / (entry.name + ".tmp")
        if not safe_replace(entry.path, tmp):
            return
        write_json(tmp, spec)
        _retry(lambda: os.replace(tmp, dst))
        qlog(self.q, spec["id"], "reused", 0.0, why)
        self.after_finish(spec)

    def launch(self, entry: Entry) -> str:
        """Starts one job: 'started', 'reused', 'skipped' or 'stopped'."""
        spec = dict(entry.spec)
        jid, kind, cwd = spec["id"], spec["kind"], Path(spec["cwd"])
        if kind == "orca":
            ok, why = can_reuse(cwd, spec["basename"])
            if ok:
                self.finish_reused(entry, why)
                return "reused"
            if self.wait_for_foreign_orca() == "stopped":
                return "stopped"
            what = (f"campaign_queue grid batch (first {jid})" if is_grid(spec) else f"campaign_queue {jid}")
            got = self.wait_for_lock(what, keep=True)
            if got != "ok":
                return "stopped" if got == "stopped" else "skipped"
        else:
            if self.wait_for_foreign_orca() == "stopped":
                return "stopped"
            got = self.wait_for_lock(f"campaign_queue pre-check for {jid}", keep=False)
            if got != "ok":
                return "stopped" if got == "stopped" else "skipped"
            release_orca_lock()                                  # the command takes the lock itself
        running_path = self.q.running / entry.name
        if not safe_replace(entry.path, running_path):
            qlog(self.q, jid, "skipped", None, f"cannot move {entry.path} to running\\ (file in use); next scan")
            return "skipped"
        log_offset = 0
        inp_sha1 = None
        try:
            if kind == "orca":
                base = spec["basename"]
                inp, out_path = cwd / f"{base}.inp", cwd / f"{base}.out"
                inp_sha1 = sha1_file(inp)
                bak = backup_out(out_path)
                if bak:
                    qlog(self.q, jid, "backup", None, f"{out_path.name} -> {bak.name}")
                sidecar_path(cwd, base).unlink(missing_ok=True)
                argv = [str(ORCA_BIN), f"{base}.inp"]
                out_fh = open(out_path, "wb")
            else:
                out_path = self.q.logs / f"{jid}.log"
                out_fh = open(out_path, "ab")
                log_offset = out_fh.tell()
                out_fh.write(f"===== {now_iso()} campaign_queue start: {subprocess.list2cmdline(spec['command'])}"
                             f"\n===== cwd {cwd}\n".encode("utf-8"))
                out_fh.flush()
                argv = list(spec["command"])
        except OSError as exc:
            self.fail_unstarted(spec, running_path, f"could not prepare the job: {exc}")
            return "skipped"
        flags = (subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.BELOW_NORMAL_PRIORITY_CLASS) if _IS_WIN else 0
        t0 = time.time()
        try:
            proc = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=out_fh, stderr=subprocess.STDOUT,
                                    env=child_env(kind), creationflags=flags)
        except OSError as exc:
            out_fh.close()
            self.fail_unstarted(spec, running_path, f"could not start: {exc}")
            return "skipped"
        handle = create_kill_job(proc)
        job = RunningJob(spec, entry.name, running_path, proc, handle, t0, out_fh, out_path, inp_sha1, log_offset)
        job.next_stall_check = t0 + self.stall_interval(spec)
        self.running[jid] = job
        spec.update(started=now_iso(), started_epoch=round(t0, 3), pid=proc.pid, output=str(out_path))
        if inp_sha1:
            spec["inp_sha1"] = inp_sha1
        write_json(running_path, spec)
        detail = (f"pid {proc.pid}, {spec['ranks']} rank(s){', grid' if is_grid(spec) else ''}, "
                  f"timeout {fmt_wall(spec['timeout_s'])}, stall {fmt_wall(spec['stall_s'])}"
                  + ("" if handle or not _IS_WIN else ", WARNING no Job Object (snapshot kill only)"))
        qlog(self.q, jid, "started", None, detail)
        return "started"

    def fail_unstarted(self, spec: dict, running_path: Path, note: str) -> None:
        t = now_iso()
        spec.update(started=t, finished=t, wall_s=0.0, exit_code=None, outcome="failed", pid=None, note=note)
        write_json(running_path, spec)
        safe_replace(running_path, self.q.failed / running_path.name, attempts=120)
        self.failures += 1
        qlog(self.q, spec["id"], "finished failed", 0.0, note)
        self.release_lock_if_idle()
        self.after_finish(spec)

    @staticmethod
    def stall_interval(spec: dict) -> float:
        return max(POLL_S, min(STALL_CHECK_EVERY_S, spec["stall_s"] / 3.0))

    # ---- monitoring ----------------------------------------------------------------------------
    def last_activity(self, job: RunningJob) -> float:
        cwd = Path(job.spec["cwd"])
        depth = 1 if job.spec["kind"] == "orca" else COMMAND_SCAN_DEPTH
        t = max(job.t0, newest_mtime(cwd, depth))
        try:
            t = max(t, os.fstat(job.out_fh.fileno()).st_mtime)     # our own handle: always current
        except (OSError, ValueError):
            pass
        return t

    def account(self, job: RunningJob, now: float) -> None:
        total = job_total_processes(job.handle)
        if total is not None and total > job.total_procs:
            self.events.append((now, total - job.total_procs))
            job.total_procs = total

    def rate_tick(self, now: float) -> None:
        """Counts new process starts (job accounting + orca* processes outside our jobs) and applies the guard."""
        for job in self.running.values():
            self.account(job, now)
        if _IS_WIN and self.running:
            snap = process_snapshot()
            live = {pid for pid, _, exe in snap if exe.lower().startswith("orca")}
            new = live - self.seen_orca
            handles = [j.handle for j in self.running.values() if j.handle]
            outside = [pid for pid in new if not pid_in_jobs(pid, handles)]
            if outside:
                self.events.append((now, len(outside)))
            self.seen_orca = live
        self.events = [(t, n) for t, n in self.events if now - t < LAUNCH_WINDOW_S]
        grid_running = any(is_grid(j.spec) for j in self.running.values())
        conc = self.grid_concurrency
        if grid_running:
            conc, self.starts_60s, msg = guard_decision(self.events, now, self.grid_concurrency)
            if msg:
                self.grid_concurrency = conc
                qlog(self.q, "-", "guard", None, msg)
                try:
                    write_json(self.q.guard_state, {"grid_concurrency": conc, "tripped": now_iso(),
                                                    "starts_in_60s": self.starts_60s})
                except OSError:
                    pass
        else:
            _, self.starts_60s, _ = guard_decision(self.events, now, conc)
            if self.starts_60s > LAUNCH_LIMIT:
                for job in self.running.values():
                    if not job.rate_warned:
                        job.rate_warned = True
                        qlog(self.q, job.id, "guard", None, f"WARNING {self.starts_60s} process starts in 60 s "
                             f"(limit {LAUNCH_LIMIT}) from a serial job")

    def poll(self) -> int:
        """Checks every running job; returns how many finished."""
        now = time.time()
        self.rate_tick(now)
        finished = 0
        for job in list(self.running.values()):
            rc = job.proc.poll()
            if rc is not None:
                self.finalize(job, rc)
                finished += 1
            elif now - job.t0 > job.spec["timeout_s"]:
                self.kill(job)
                self.finalize(job, job.proc.returncode, "timeout", f"killed after timeout_s={job.spec['timeout_s']}")
                finished += 1
            elif now >= job.next_stall_check:
                job.next_stall_check = now + self.stall_interval(job.spec)
                idle = now - self.last_activity(job)
                if idle > job.spec["stall_s"]:
                    self.kill(job)
                    self.finalize(job, job.proc.returncode, "stalled",
                                  f"killed: no file change for {idle:.0f} s (stall_s={job.spec['stall_s']})")
                    finished += 1
        return finished

    def kill(self, job: RunningJob) -> None:
        """Snapshot descendants first, then TerminateJobObject, then every survivor, then the root."""
        pids = snapshot_descendants(job.proc.pid) if _IS_WIN else []
        if job.handle:
            _k32().TerminateJobObject(job.handle, 1)
        for pid, ct in pids:
            if pid_alive(pid):
                terminate_pid(pid, job.t0, ct)
        if job.proc.poll() is None:
            try:
                job.proc.kill()
            except OSError:
                pass
        try:
            job.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            if _IS_WIN:
                subprocess.run(["taskkill", "/PID", str(job.proc.pid), "/T", "/F"], capture_output=True)
            try:
                job.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass

    def finalize(self, job: RunningJob, rc: int | None, outcome: str | None = None, note: str = "") -> None:
        self.account(job, time.time())
        if outcome is None:
            try:
                self.sweep_leftovers(job)
            except OSError:
                pass
        close_handle(job.handle)                    # anything the job left behind dies with it
        job.handle = None
        try:
            job.out_fh.close()
        except OSError:
            pass
        self.running.pop(job.id, None)
        wall = time.time() - job.t0
        spec = job.spec
        if outcome is None:
            if spec["kind"] == "orca":
                normal = terminated_normally(job.out_path)
                if rc == 0 and normal:
                    outcome = "ok"
                    write_text_atomic(sidecar_path(Path(spec["cwd"]), spec["basename"]), job.inp_sha1 + "\n")
                else:
                    outcome = "failed"
                    note = f"exit code {rc}" + ("" if normal else f", no '{TERMINATED_NORMALLY}' in {job.out_path.name}")
            elif rc == 0:
                outcome = "ok"
            else:
                outcome = "failed"
                note = f"exit code {rc}; log {job.out_path}"
                if self.lock_was_busy(job) and int(spec.get("requeued", 0)) < MAX_LOCK_REQUEUES:
                    spec["requeued"] = int(spec.get("requeued", 0)) + 1
                    write_json(job.path, spec)
                    _move_back_to_jobs(job.path, self.q, reset_requeued=False)
                    qlog(self.q, job.id, "requeued", wall, "the command found the ORCA lock taken by another driver")
                    return
        spec.update(finished=now_iso(), wall_s=round(wall, 1), exit_code=rc, outcome=outcome,
                    proc_starts=job.total_procs or None)
        if note:
            spec["note"] = note
        write_json(job.path, spec)
        dest = (self.q.done if outcome in OK_OUTCOMES else self.q.failed) / job.name
        if not safe_replace(job.path, dest, attempts=120):
            qlog(self.q, job.id, "ERROR", wall, f"cannot move {job.path} to {dest.parent.name}\\; left in running\\")
        if outcome not in OK_OUTCOMES:
            self.failures += 1
        qlog(self.q, job.id, f"finished {outcome}", wall, note)
        self.release_lock_if_idle()
        self.after_finish(spec)

    def release_lock_if_idle(self) -> None:
        """The ORCA lock is taken per kind=orca job (held across a grid batch while any of it runs)."""
        if not any(j.spec["kind"] == "orca" for j in self.running.values()) and hold_lock():
            release_orca_lock()

    def sweep_leftovers(self, job: RunningJob) -> None:
        """
        The root exited by itself: kill descendants that outlived it (MS-MPI ranks that broke away from the
        Job Object would otherwise keep cores busy). The root PID cannot be recycled yet (Popen still holds its
        handle); descendants are matched by creation time.
        """
        if not _IS_WIN:
            return
        left = [(pid, ct) for pid, ct in snapshot_descendants(job.proc.pid) if pid_alive(pid)]
        killed = [pid for pid, ct in left if terminate_pid(pid, job.t0, ct)]
        if killed:
            qlog(self.q, job.id, "cleanup", None, f"killed {len(killed)} leftover process(es) of the finished job: "
                 + ", ".join(map(str, killed[:20])))

    def lock_was_busy(self, job: RunningJob) -> bool:
        try:
            with open(job.out_path, "rb") as f:
                f.seek(job.log_offset)
                return LOCK_BUSY_TEXT.encode() in f.read()
        except OSError:
            return False

    def after_finish(self, spec: dict) -> None:
        if not spec["id"].startswith("cal_"):
            return
        new = not self.q.calibration_csv.exists()
        try:
            with open(self.q.calibration_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(CAL_COLUMNS)
                w.writerow([spec.get(c, "") if spec.get(c) is not None else "" for c in CAL_COLUMNS])
        except OSError as exc:
            qlog(self.q, spec["id"], "ERROR", None, f"calibration.csv not written: {exc}")

    # ---- scheduling ----------------------------------------------------------------------------
    def schedule(self) -> int:
        """Starts whatever may start now; returns the number of jobs started or reused."""
        n = 0
        while True:
            scan = scan_queue(self.q)
            for e in scan.pending:
                if e.problems and self.invalid_logged.get(e.name) != "; ".join(e.problems):
                    self.invalid_logged[e.name] = "; ".join(e.problems)
                    qlog(self.q, e.id, "invalid", None, f"{e.name}: " + "; ".join(e.problems))
            valid = [e for e in scan.pending if e.valid]
            invalid_ids = {e.id for e in scan.pending if not e.valid}
            cls = classify(valid, scan.ids("done"), scan.ids("failed"), set(self.running) | scan.ids("running"),
                           invalid_ids, self.only, self.max_ranks)
            ready_specs = [e.spec for e in cls.ready]
            batch = plan_launches(ready_specs, [j.spec for j in self.running.values()], self.max_ranks,
                                  self.grid_concurrency)
            if not batch:
                return n
            by_id = {e.id: e for e in cls.ready}
            progressed = False
            for spec in batch:
                if self.q.stop_file.exists():
                    return n
                res = self.launch(by_id[spec["id"]])
                if res in ("started", "reused"):
                    n += 1
                    progressed = True
                if res == "reused":
                    break                    # re-plan: the slot is free again
                if res == "stopped":
                    return n
            if not progressed:
                return n

    def loop(self) -> int:
        next_scan = 0.0
        while True:
            finished = self.poll() if self.running else 0
            now = time.time()
            stop = self.q.stop_file.exists()
            pause = self.q.pause_file.exists()
            launched = 0
            scanned = False
            if not stop and not pause and (finished or now >= next_scan):
                launched = self.schedule()
                scanned = True
                next_scan = time.time() + SCAN_EVERY_S if (self.running or not launched) else 0.0
                self.heartbeat(force=True)
            if not any(j.spec["kind"] == "orca" for j in self.running.values()) and hold_lock():
                release_orca_lock()
            if not self.running:
                if stop:
                    qlog(self.q, "-", "runner-stop", None, f"{self.q.stop_file.name} found: exiting")
                    try:
                        self.q.stop_file.unlink()
                    except OSError:
                        pass
                    return 0
                if self.once and scanned and not launched:
                    return self.drained()
            self.heartbeat()
            time.sleep(POLL_S)

    def drained(self) -> int:
        scan = scan_queue(self.q)
        valid = [e for e in scan.pending if e.valid]
        cls = classify(valid, scan.ids("done"), scan.ids("failed"), scan.ids("running"),
                       {e.id for e in scan.pending if not e.valid}, self.only, self.max_ranks)
        left = ([f"{e.name}: blocked ({why})" for e, why in cls.blocked]
                + [f"{e.name}: {why}" for e, why in cls.waiting]
                + [f"{e.name}: invalid" for e in scan.pending if not e.valid])
        qlog(self.q, "-", "runner-drained", None, f"{self.failures} failed in this run; {len(left)} left in jobs\\"
             + ("" if not left else ": " + " | ".join(left[:20])))
        return 1 if self.failures else 0

    def abort(self, reason: str) -> None:
        for job in list(self.running.values()):
            try:
                self.kill(job)
            except Exception:
                pass
            close_handle(job.handle)
            try:
                job.out_fh.close()
            except OSError:
                pass
            self.running.pop(job.id, None)
            try:
                _move_back_to_jobs(job.path, self.q, reset_requeued=False)
                qlog(self.q, job.id, "requeued", time.time() - job.t0, f"killed: {reason}")
            except OSError as exc:
                qlog(self.q, job.id, "killed", time.time() - job.t0, f"{reason}; spec left in running\\ ({exc})")

    def run(self) -> int:
        self.q.ensure()
        msg = self.claim_runner()
        if msg:
            print(msg, file=sys.stderr)
            return 3
        old_handlers = {}
        try:
            def _interrupt(signum, frame):
                raise KeyboardInterrupt
            if hasattr(signal, "SIGBREAK"):
                old_handlers[signal.SIGBREAK] = signal.signal(signal.SIGBREAK, _interrupt)
            awake = keep_awake()
            qlog(self.q, "-", "runner-start", None,
                 f"pid {os.getpid()}, {'once' if self.once else 'daemon'}, max ranks {self.max_ranks}, "
                 f"only {self.only or 'all'}, ORCA {ORCA_BIN}, lock {LOCK_FILE}, keep-awake {awake}"
                 + (f"; {override_warning()}" if override_warning() else ""))
            self.load_guard_state()
            self.requeue_leftovers()
            return self.loop()
        except KeyboardInterrupt:
            qlog(self.q, "-", "runner-interrupted", None, "Ctrl+C: killing running jobs")
            self.abort("runner interrupted")
            return 130
        except BaseException as exc:
            qlog(self.q, "-", "runner-crash", None, f"{type(exc).__name__}: {exc}; killing running jobs")
            self.abort(f"runner error {type(exc).__name__}")
            raise
        finally:
            release_orca_lock()
            self.release_runner()
            for sig, h in old_handlers.items():
                signal.signal(sig, h)


# ==============================================================================
# 9. --dry-run AND --status
# ==============================================================================
def simulate(valid: list[Entry], done_ids: set[str], failed_ids: set[str], running_ids: set[str],
             invalid_ids: set[str], only: str | None, max_ranks: int, grid_concurrency: int):
    """Execution order assuming every job succeeds: list of steps (lists of Entries started together)."""
    sim_done = set(done_ids) | set(running_ids)
    remaining = list(valid)
    steps: list[list[Entry]] = []
    while True:
        cls = classify(remaining, sim_done, failed_ids, set(), invalid_ids, only, max_ranks)
        batch = plan_launches([e.spec for e in cls.ready], [], max_ranks, grid_concurrency)
        if not batch:
            return steps, cls
        ids = {s["id"] for s in batch}
        step = [e for e in cls.ready if e.id in ids]
        steps.append(step)
        sim_done |= ids
        remaining = [e for e in remaining if e.id not in ids]


def _job_line(e: Entry) -> str:
    s = e.spec
    extra = ""
    if s["kind"] == "orca":
        ok, _ = can_reuse(Path(s["cwd"]), s["basename"])
        extra = "  (would reuse: finished, input unchanged)" if ok else ""
    return (f"{e.name:<40} {s['project']:<10} {s['kind']:<8} {s['ranks']} rank(s)  "
            f"timeout {fmt_wall(s['timeout_s'])}{extra}")


def dry_run(q: Queue, only: str | None, max_ranks: int) -> int:
    q.ensure()
    scan = scan_queue(q)
    invalid = [e for e in scan.pending if not e.valid]
    valid = [e for e in scan.pending if e.valid]
    grid_c = GRID_CONCURRENCY
    try:
        grid_c = min(grid_c, int(json.loads(q.guard_state.read_text(encoding="utf-8"))["grid_concurrency"]))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    print(f"campaign_queue DRY RUN - nothing is started.  Queue: {q.root}")
    print(f"max ranks {max_ranks}, grid concurrency {grid_c} (grid jobs <= {GRID_MAX_RANKS} ranks), "
          f"only {only or 'all projects'}, ORCA {ORCA_BIN}")
    if override_warning():
        print(override_warning())
    print(f"specs: {len(scan.pending)} pending ({len(valid)} valid, {len(invalid)} invalid), {len(scan.running)} running, "
          f"{len(scan.done)} done, {len(scan.failed)} failed")
    if invalid:
        print("\nINVALID specs (fix or delete them; exit code 2):")
        for e in invalid:
            print(f"  {e.name}:")
            for p in e.problems:
                print(f"      - {p}")
    procs = foreign_orca()
    if procs and not scan.running:
        print(f"\nNOTE: {len(procs)} orca*.exe process(es) not started by this queue are running now (PIDs "
              f"{', '.join(str(p) for p, _ in procs[:12])}); a runner waits until they are gone "
              f"(or start it with --allow-foreign-orca)")
    if scan.running:
        print("\nrunning now (assumed to finish ok): " + ", ".join(e.id for e in scan.running))
    steps, cls = simulate(valid, scan.ids("done"), scan.ids("failed"), scan.ids("running"),
                          {e.id for e in invalid}, only, max_ranks, grid_c)
    print("\nexecution order (a step starts when the previous one is finished; grid steps refill as jobs end):")
    if not steps:
        print("  (nothing to run)")
    collapse = len(steps) > 40
    i = 0
    while i < len(steps):
        step = steps[i]
        j = i
        if collapse and is_grid(step[0].spec):
            while j + 1 < len(steps) and is_grid(steps[j + 1][0].spec):
                j += 1
        if collapse and j - i >= 4:
            njobs = sum(len(s) for s in steps[i:j + 1])
            print(f"  steps {i + 1}-{j + 1}: grid batch, {njobs} jobs, up to {grid_c} concurrent "
                  f"(first {steps[i][0].name}, last {steps[j][-1].name})")
            i = j + 1
            continue
        if is_grid(step[0].spec):
            label = f"grid x{len(step)} concurrent ({sum(e.spec['ranks'] for e in step)} ranks)"
        else:
            label = "serial"
        for k, e in enumerate(step):
            head = f"  step {i + 1:>3}  {label:<30}" if k == 0 else " " * 42
            print(f"{head} {_job_line(e)}")
        i += 1
    if cls.blocked:
        print("\nblocked (a dependency failed; these never run):")
        for e, why in cls.blocked:
            print(f"  {e.name}: {why}")
    if cls.waiting:
        print("\nnot schedulable now:")
        for e, why in cls.waiting:
            print(f"  {e.name}: {why}")
    if cls.skipped:
        print(f"\nskipped by --only {only}: {len(cls.skipped)} job(s)")
    return 2 if invalid else 0


def status(q: Queue) -> int:
    q.ensure()
    scan = scan_queue(q, read_finished=True)
    now = time.time()
    rows = []
    valid = [e for e in scan.pending if e.valid]
    cls = classify(valid, scan.ids("done"), scan.ids("failed"), scan.ids("running"),
                   {e.id for e in scan.pending if not e.valid})
    notes = {e.name: "ready" for e in cls.ready}
    notes.update({e.name: why for e, why in cls.waiting})
    notes.update({e.name: f"BLOCKED: {why}" for e, why in cls.blocked})
    for e in scan.pending:
        if not e.valid:
            notes[e.name] = "INVALID: " + "; ".join(e.problems)
    for state in STATES:
        for e in sorted(getattr(scan, state), key=Entry.sort_key):
            s = e.spec or {}
            wall = s.get("wall_s")
            if state == "running" and isinstance(s.get("started_epoch"), (int, float)):
                wall = now - s["started_epoch"]
            note = notes.get(e.name, "") if state == "pending" else (s.get("note") or "")
            rows.append((state, s.get("priority", "?"), e.id, s.get("project", "?"), s.get("kind", "?"),
                         s.get("ranks", "?"), fmt_wall(wall), s.get("outcome", "") or "", note))
    counts = {st: len(getattr(scan, st)) for st in STATES}
    print(f"campaign_queue status  {now_iso()}  queue {q.root}  ORCA {ORCA_BIN}")
    if override_warning():
        print("  " + override_warning())
    print("  " + ", ".join(f"{counts[s]} {s}" for s in STATES))
    runner = "not running"
    try:
        info = json.loads(q.runner_pid.read_text(encoding="utf-8"))
        if pid_alive(int(info.get("pid", -1))):
            runner = f"PID {info['pid']} ({info.get('mode', '?')}, since {info.get('started', '?')})"
        else:
            runner = f"not running (stale runner.pid, PID {info.get('pid')})"
    except (OSError, ValueError, AttributeError):
        pass
    print(f"  runner: {runner}")
    try:
        hb = q.heartbeat.read_text(encoding="utf-8").splitlines()[0]
        age = now - datetime.fromisoformat(hb).timestamp()
        print(f"  heartbeat: {hb} ({age:.0f} s ago)")
    except (OSError, ValueError, IndexError):
        print("  heartbeat: none")
    holder = lock_holder()
    print(f"  ORCA lock ({LOCK_FILE}): " + (f"held by PID {holder.get('pid')} ({holder.get('what')}, since "
                                            f"{holder.get('started')})" if holder else "free"))
    procs = foreign_orca()
    print(f"  orca*.exe processes on this machine: {len(procs)}"
          + (f" (PIDs {', '.join(str(p) for p, _ in procs[:12])}; the runner waits for them unless they are its "
             f"own or --allow-foreign-orca)" if procs else ""))
    try:
        g = json.loads(q.guard_state.read_text(encoding="utf-8"))
        print(f"  launch-rate guard: TRIPPED {g.get('tripped')} -> grid concurrency {g.get('grid_concurrency')}")
    except (OSError, ValueError, AttributeError):
        print(f"  launch-rate guard: not tripped (grid concurrency {GRID_CONCURRENCY})")
    for f, txt in ((q.pause_file, "PAUSE present: no new launches"), (q.stop_file, "STOP present: runner exits when idle")):
        if f.exists():
            print(f"  {txt}")
    print()
    if not rows:
        print("(no jobs)")
    else:
        hdr = ("state", "prio", "id", "project", "kind", "ranks", "wall", "outcome", "note")
        widths = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(8)]
        widths = [min(w, 44) for w in widths]
        print("  ".join(str(h).ljust(w) for h, w in zip(hdr[:8], widths)) + "  note")
        for r in rows:
            print("  ".join(str(c)[:44].ljust(w) for c, w in zip(r[:8], widths)) + "  " + str(r[8]))
    print("\ntotal wall per project (finished jobs):")
    per: dict[str, dict[str, float]] = {}
    for e in scan.done + scan.failed:
        s = e.spec or {}
        d = per.setdefault(s.get("project", "?"), {"n": 0, "ok": 0.0, "failed": 0.0})
        d["n"] += 1
        w = s.get("wall_s") if isinstance(s.get("wall_s"), (int, float)) else 0.0
        d["ok" if e.state == "done" else "failed"] += w
    if not per:
        print("  (none)")
    for proj in sorted(per):
        d = per[proj]
        print(f"  {proj:<10} {int(d['n'])} job(s)  total {fmt_wall(d['ok'] + d['failed'])}  "
              f"(done {fmt_wall(d['ok'])}, failed {fmt_wall(d['failed'])})")
    return 0


# ==============================================================================
# 10. CLI
# ==============================================================================
def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Campaign ORCA queue (the only ORCA launcher of the campaign).")
    m = p.add_mutually_exclusive_group()
    m.add_argument("--status", action="store_true", help="table of all jobs; total wall per project")
    m.add_argument("--dry-run", action="store_true", help="validate all specs, print the execution order, run nothing")
    m.add_argument("--once", action="store_true", help="drain the queue, then exit")
    m.add_argument("--enqueue", metavar="SPEC_JSON", help="validate a spec (file or inline JSON) and add it to jobs\\ (idempotent)")
    m.add_argument("--retry", metavar="ID", help="move a failed job back to jobs\\")
    p.add_argument("--queue-dir", default=str(DEFAULT_QUEUE_DIR), help=f"queue folder (default {DEFAULT_QUEUE_DIR})")
    p.add_argument("--max-ranks", type=int, default=MAX_RANKS_CAP, help=f"MPI ranks in flight, 1-{MAX_RANKS_CAP}")
    p.add_argument("--only", choices=PROJECTS, help="run only this project's jobs")
    p.add_argument("--allow-foreign-orca", action="store_true",
                   help="start jobs even while orca*.exe processes of another driver run (default: wait for them)")
    args = p.parse_args(argv)
    if not 1 <= args.max_ranks <= MAX_RANKS_CAP:
        p.error(f"--max-ranks must be 1-{MAX_RANKS_CAP} (campaign budget)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    q = Queue(Path(args.queue_dir))
    if args.status:
        return status(q)
    if args.dry_run:
        return dry_run(q, args.only, args.max_ranks)
    if args.enqueue:
        try:
            text = args.enqueue if args.enqueue.lstrip().startswith("{") else Path(args.enqueue).read_text(encoding="utf-8")
            spec = json.loads(text)
        except (OSError, ValueError) as exc:
            print(f"cannot read {args.enqueue}: {exc}", file=sys.stderr)
            return 2
        st, msg = enqueue(spec, q.root) if isinstance(spec, dict) else ("invalid", "spec is not a JSON object")
        print(msg)
        return 0 if st in ("enqueued", "exists", "replaced") else 2
    if args.retry:
        q.ensure()
        ok, msg = retry_failed(q, args.retry)
        print(msg)
        return 0 if ok else 2
    return Runner(q, args.max_ranks, args.only, once=args.once, allow_foreign=args.allow_foreign_orca).run()


if __name__ == "__main__":
    sys.exit(main())
