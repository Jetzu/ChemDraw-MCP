"""Persistent state of the ChemDraw agent bridge (standard library only).

Everything lives under ``$CHEMDRAWMCP_HOME`` (default ``~/.chemdrawmcp``):

* ``agent_settings.json``  - model/effort per backend (menus: see the model catalog below) and the default backend (``CHEMDRAWMCP_AGENT_SETTINGS``
  overrides the path);
* ``agent_sessions.json``  - the current conversation id per backend plus the turn history, so a bridge restart
  keeps conversations resumable;
* ``bridge.json``          - port/token/pid of the running bridge (written by :mod:`chemdrawmcp.bridge.server`).
"""
from __future__ import annotations

import glob
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

BACKENDS = ("claude", "codex")
BACKEND_LABELS = {"claude": "Claude Code", "codex": "Codex"}
HISTORY_LIMIT = 200


# ---------------------------------------------------------------------------
# model catalog (follows new model releases without code changes)
# ---------------------------------------------------------------------------
# The model / effort menus are read from the caches the CLIs keep for their own /model pickers and refresh by
# themselves:
#   Claude Code: <CLAUDE_CONFIG_DIR or ~/.claude>/cache/model-catalog/*-cc.json
#   Codex:       <CODEX_HOME or ~/.codex>/models_cache.json   (+ the default model from config.toml)
# Claude models are offered by family alias (opus, sonnet, ...), which Claude Code resolves to the newest model of
# that family, so a saved choice keeps following new releases.  Without a cache the built-in lists below are used.
CLAUDE_ALIASES = ("fable", "opus", "sonnet", "haiku")  # fallback; the cache lists the aliases it knows
MODEL_CHOICES = {  # fallback when the CLI has not written its model cache yet
    "claude": [("", "default")] + [(a, a) for a in CLAUDE_ALIASES],
    "codex": [("", "default (config.toml)")],
}
EFFORT_CHOICES = {  # fallback, see effort_choices()
    "claude": ["", "low", "medium", "high", "xhigh", "max"],
    "codex": ["", "minimal", "low", "medium", "high", "xhigh"],
}
_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
_catalog_memo: dict = {}


def _claude_config_dir() -> str:
    return os.path.expanduser(os.environ.get("CLAUDE_CONFIG_DIR") or os.path.join("~", ".claude"))


def _codex_home() -> str:
    return os.path.expanduser(os.environ.get("CODEX_HOME") or os.path.join("~", ".codex"))


def _json_file(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _sorted_efforts(levels) -> List[str]:
    seen = list(dict.fromkeys(str(e) for e in levels if e))
    return [e for e in _EFFORT_ORDER if e in seen] + [e for e in seen if e not in _EFFORT_ORDER]


def _read_claude_catalog(files: List[str]) -> Tuple[List[dict], Dict[str, str]]:
    best, best_t = None, -1.0
    for p in files:
        d = _json_file(p)
        if not isinstance(d, dict) or not p.endswith("-cc.json"):
            continue
        try:
            t = float(d.get("fetchedAt") or 0)
        except (TypeError, ValueError):
            t = 0.0
        if t > best_t:
            best, best_t = d, t
    if best is None:
        return [], {}
    masked = _json_file(os.path.join(_claude_config_dir(), "cache", "model-catalog", "masked-ids.json"))
    masked = set(masked.get("ids") or []) if isinstance(masked, dict) else set()
    try:
        config = best["catalog"]["config"]
        models = config["models"]
    except (KeyError, TypeError):
        return [], {}
    aliases = {}
    targets = config.get("provider_alias_targets") if isinstance(config, dict) else None
    for alias, t in (targets.items() if isinstance(targets, dict) else []):
        if isinstance(t, dict) and t.get("default"):
            aliases[str(alias)] = str(t["default"])
    out = []
    for m in models if isinstance(models, list) else []:
        if not isinstance(m, dict) or not m.get("id") or m["id"] in masked:
            continue
        rt = m.get("runtime") if isinstance(m.get("runtime"), dict) else {}
        out.append({"id": str(m["id"]), "name": str(m.get("name") or m["id"]),
                    "listed": str(m.get("section") or "main") == "main",
                    "family": str(rt.get("family") or "").lower(),
                    "efforts": _sorted_efforts(rt.get("effort_levels") or []),
                    "default_effort": str(rt.get("default_effort") or "")})
    return out, aliases


def _read_codex_catalog(files: List[str]) -> List[dict]:
    d = _json_file(files[0]) if files else None
    models = d.get("models") if isinstance(d, dict) else None
    out = []
    for m in models if isinstance(models, list) else []:
        if not isinstance(m, dict) or not m.get("slug"):
            continue
        levels = [r.get("effort") for r in m.get("supported_reasoning_levels") or [] if isinstance(r, dict)]
        prio = m.get("priority")
        out.append({"id": str(m["slug"]), "name": str(m.get("display_name") or m["slug"]),
                    "listed": m.get("visibility", "list") == "list", "family": "",
                    "priority": prio if isinstance(prio, (int, float)) else 1e9,
                    "efforts": _sorted_efforts(levels), "default_effort": str(m.get("default_reasoning_level") or "")})
    out.sort(key=lambda m: m["priority"])
    return out, {}


def model_catalog(backend: str) -> List[dict]:
    """Models the installed CLI offers, newest/most relevant first ([] when its cache is missing).  Each entry:
    {"id", "name", "family", "listed", "efforts", "default_effort"}.  Re-read whenever the cache file changes."""
    if backend == "claude":
        files = sorted(glob.glob(os.path.join(_claude_config_dir(), "cache", "model-catalog", "*.json")))
    else:
        files = [os.path.join(_codex_home(), "models_cache.json")]
    stamp = []
    for p in files:
        try:
            stamp.append((p, os.path.getmtime(p)))
        except OSError:
            pass
    stamp = tuple(stamp)
    hit = _catalog_memo.get(backend)
    if not (hit and hit[0] == stamp):
        try:
            cat, aliases = (_read_claude_catalog(files) if backend == "claude"
                            else _read_codex_catalog([p for p, _ in stamp]))
        except Exception:
            cat, aliases = [], {}
        hit = _catalog_memo[backend] = (stamp, cat, aliases)
    return hit[1]


def model_aliases(backend: str) -> Dict[str, str]:
    """Claude Code's aliases and the model each currently points to ({} for Codex or without a cache)."""
    model_catalog(backend)
    hit = _catalog_memo.get(backend)
    return dict(hit[2]) if hit else {}


def _codex_config_value(key: str) -> str:
    """A top-level string value of Codex's config.toml ("" when unset)."""
    try:
        with open(os.path.join(_codex_home(), "config.toml"), encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s.startswith("["):
                    break
                m = re.match(r'%s\s*=\s*["\']([^"\']*)["\']' % re.escape(key), s)
                if m:
                    return m.group(1)
    except Exception:
        pass
    return ""


def model_choices(backend: str) -> List[Tuple[str, str]]:
    """(value, label) pairs for the model menu; value "" = the CLI's own default.  Any other name still works."""
    cat = model_catalog(backend)
    if backend == "claude":
        if not cat:
            return list(MODEL_CHOICES["claude"])
        by_target = {}
        for alias, target in model_aliases(backend).items():
            by_target.setdefault(target, alias)
        out = [("", "default")]
        for m in cat:
            if not m["listed"]:
                continue  # older models: still accepted by name (/model <id>)
            alias = by_target.get(m["id"])
            out.append((alias, "%s · %s" % (alias, m["name"])) if alias else (m["id"], m["name"]))
        return out
    cfg = _codex_config_value("model")
    out = [("", "default (%s)" % cfg if cfg else "default (config.toml)")]
    listed = [(m["id"], m["id"]) for m in cat if m["listed"]]
    if not listed and cfg:
        listed = [(cfg, cfg)]
    return out + listed


def _catalog_entry(backend: str, model: str) -> Optional[dict]:
    cat = model_catalog(backend)
    model = (model or "").strip()
    if not cat:
        return None
    if not model:
        if backend != "codex":
            return None
        model = _codex_config_value("model")
    base = model.split("[")[0]
    if backend == "claude":
        base = model_aliases(backend).get(base, base)
    return next((m for m in cat if m["id"] == base), None)


def effort_choices(backend: str, model: Optional[str] = None) -> List[str]:
    """Effort levels ("" = default first): those of ``model`` when the catalog knows it, else the union over every
    model of the backend (or the built-in list when there is no catalog)."""
    if model is not None:
        m = _catalog_entry(backend, model)
        if m and m["efforts"]:
            return [""] + m["efforts"]
    levels = _sorted_efforts(e for m in model_catalog(backend) for e in m["efforts"])
    return [""] + levels if levels else list(EFFORT_CHOICES[backend])


def resolve_model(backend: str, model: str) -> Tuple[str, str]:
    """(model to run, note).  A saved model the CLI no longer offers in its menu is replaced by one that works: a
    superseded Claude model by its family alias (= the newest model of that family), an unknown Codex model by
    Codex's default."""
    model = (model or "").strip()
    cat = model_catalog(backend)
    if not model or not cat:
        return model, ""
    base = model.split("[")[0]
    ids = {m["id"] for m in cat}
    if backend == "claude":
        aliases = model_aliases(backend) or {a: "" for a in CLAUDE_ALIASES}
        if base in aliases or base in {m["id"] for m in cat if m["listed"]}:
            return model, ""
        entry = next((m for m in cat if m["id"] == base), None)
        fam = entry["family"] if entry and entry["family"] else next(
            (a for a in aliases if base.startswith("claude-%s-" % a)), "")
        if fam in aliases:
            new = fam + model[len(base):]  # keeps a context suffix such as [1m]
            return new, "%s has been superseded; using %s (the newest %s model)" % (model, new, fam)
        return model, ""
    if base in ids or _codex_config_value("model_provider") not in ("", "openai"):
        return model, ""
    return "", "%s is not in Codex's model list; using Codex's default model" % model


def resolve_effort(backend: str, model: str, effort: str) -> Tuple[str, str]:
    """(effort to run, note): an effort the chosen model does not support falls back to the model's default."""
    effort = (effort or "").strip()
    m = _catalog_entry(backend, model)
    if not effort or not m or not m["efforts"] or effort in m["efforts"]:
        return effort, ""
    return "", "%s does not support effort %s; using its default" % (m["name"], effort)


def catalog_snapshot() -> dict:
    """Menus for a panel: model options (value, label), effort levels per backend and per offered model."""
    out: dict = {"model_options": {}, "model_choices": {}, "effort_choices": {}, "model_efforts": {}}
    for b in BACKENDS:
        opts = model_choices(b)
        out["model_options"][b] = [list(o) for o in opts]
        out["model_choices"][b] = [o[0] for o in opts]
        out["effort_choices"][b] = effort_choices(b)
        out["model_efforts"][b] = {o[0]: effort_choices(b, o[0]) for o in opts}
    return out


# Codex's --approve-for-me as config overrides: the flag exists on `codex exec` but not on `codex exec resume`, so
# follow-up turns failed with "unexpected argument"; -c works on both.
CODEX_APPROVAL_ARGS = ["-c", 'approvals_reviewer="auto_review"', "-c", 'approval_policy="on-request"',
                       "-c", 'sandbox_mode="workspace-write"']



# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

def home_dir() -> str:
    """``$CHEMDRAWMCP_HOME`` or ``~/.chemdrawmcp`` (created when possible)."""
    d = os.environ.get("CHEMDRAWMCP_HOME") or os.path.join(os.path.expanduser("~"), ".chemdrawmcp")
    d = os.path.abspath(os.path.expanduser(d))
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def settings_path() -> str:
    return os.environ.get("CHEMDRAWMCP_AGENT_SETTINGS") or os.path.join(home_dir(), "agent_settings.json")


def sessions_path() -> str:
    return os.path.join(home_dir(), "agent_sessions.json")


def bridge_info_path() -> str:
    return os.path.join(home_dir(), "bridge.json")


def workspace_dir() -> str:
    """Working directory the CLIs run in (``CHEMDRAWMCP_BRIDGE_CWD`` or ``<home>/workspace``).  Claude Code keys
    its resumable sessions by working directory, so every turn must use the same one."""
    d = os.environ.get("CHEMDRAWMCP_BRIDGE_CWD") or os.path.join(home_dir(), "workspace")
    d = os.path.abspath(os.path.expanduser(d))
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return os.path.expanduser("~")
    return d


def read_json(path: str) -> Any:
    """Parsed JSON or ``None`` (missing/corrupt file)."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: str, obj: Any, private: bool = False) -> bool:
    """Atomic JSON write (temp file + rename); ``private`` restricts the file to the user (tokens)."""
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = "%s.%d.tmp" % (path, os.getpid())
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(tmp, flags, 0o600 if private else 0o644)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# agent settings (model / effort / default backend)
# ---------------------------------------------------------------------------

def load_settings() -> dict:
    d = read_json(settings_path())
    return d if isinstance(d, dict) else {}


def save_settings(d: dict) -> None:
    write_json(settings_path(), d)


def backend_settings(backend: str) -> dict:
    """{"model": str, "effort": str} for a backend, persisted per user."""
    d = load_settings().get(backend, {})
    if not isinstance(d, dict):
        d = {}
    return {"model": str(d.get("model", "") or ""), "effort": str(d.get("effort", "") or "")}


def set_backend_settings(backend: str, model: Optional[str] = None, effort: Optional[str] = None) -> dict:
    d = load_settings()
    cur = d.get(backend, {})
    if not isinstance(cur, dict):
        cur = {}
    if model is not None:
        cur["model"] = model
    if effort is not None:
        cur["effort"] = effort
    d[backend] = cur
    d.setdefault("default_backend", backend)
    save_settings(d)
    return {"model": str(cur.get("model", "") or ""), "effort": str(cur.get("effort", "") or "")}


def stored_default_backend() -> Optional[str]:
    d = load_settings().get("default_backend")
    return d if d in BACKENDS else None


def set_default_backend(backend: str) -> None:
    d = load_settings()
    d["default_backend"] = backend
    save_settings(d)


def settings_snapshot(default: Optional[str] = None) -> dict:
    """Everything the panel needs to render its settings UI."""
    return {
        "claude": backend_settings("claude"),
        "codex": backend_settings("codex"),
        "default_backend": default or stored_default_backend() or "claude",
        **catalog_snapshot(),  # model_options, model_choices, effort_choices, model_efforts
    }


# ---------------------------------------------------------------------------
# sessions + history
# ---------------------------------------------------------------------------

class SessionStore:
    """Conversation ids per backend and the turn history, persisted to ``agent_sessions.json``.

    Thread-safe; every mutation is written through immediately so a crash of the bridge loses nothing."""

    def __init__(self, path: Optional[str] = None, history_limit: int = HISTORY_LIMIT):
        self.path = path or sessions_path()
        self.history_limit = history_limit
        self._lock = threading.RLock()
        self._sessions: Dict[str, Optional[str]] = {b: None for b in BACKENDS}
        self._history: List[dict] = []
        self.load()

    def load(self) -> None:
        d = read_json(self.path)
        with self._lock:
            if isinstance(d, dict):
                s = d.get("sessions") or {}
                for b in BACKENDS:
                    v = s.get(b) if isinstance(s, dict) else None
                    self._sessions[b] = str(v) if v else None
                h = d.get("history") or []
                self._history = [e for e in h if isinstance(e, dict)][-self.history_limit:]

    def save(self) -> None:
        with self._lock:
            write_json(self.path, {"sessions": dict(self._sessions), "history": list(self._history),
                                   "saved": time.time()})

    def sessions(self) -> Dict[str, Optional[str]]:
        with self._lock:
            return dict(self._sessions)

    def get_session(self, backend: str) -> Optional[str]:
        with self._lock:
            return self._sessions.get(backend)

    def set_session(self, backend: str, session_id: Optional[str]) -> None:
        with self._lock:
            self._sessions[backend] = session_id or None
            self.save()

    def clear(self, backend: Optional[str] = None, history: bool = False) -> Dict[str, Optional[str]]:
        """Forget the conversation of one backend (or all); the next run starts a new one."""
        with self._lock:
            for b in ([backend] if backend else list(BACKENDS)):
                self._sessions[b] = None
                if history:
                    self._history = [e for e in self._history if e.get("backend") != b]
            self.save()
            return dict(self._sessions)

    def add_history(self, entry: dict) -> None:
        with self._lock:
            self._history.append(dict(entry))
            del self._history[:-self.history_limit]
            self.save()

    def history(self, backend: Optional[str] = None, limit: Optional[int] = None) -> List[dict]:
        with self._lock:
            out = [dict(e) for e in self._history if not backend or e.get("backend") == backend]
        if limit:
            out = out[-limit:]
        return out
