"""
Ollama HTTP API 後端：模型列表、切換、chat 推論。

環境變數：
  OLLAMA_BASE_URL   預設 http://127.0.0.1:11434
  OLLAMA_MODEL      預設模型名（覆寫 ollama_models.json default_model）
  OLLAMA_TIMEOUT    秒，預設 300
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from code.paths import config_dir

CONFIG_PATH = config_dir() / "ollama_models.json"

_current_model: str = ""
_last_error: str = ""


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _load_config() -> dict:
    if CONFIG_PATH.is_file():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except Exception:
            pass
    return {}


def _base_url() -> str:
    return _env("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def _timeout() -> float:
    try:
        return float(_env("OLLAMA_TIMEOUT", "300") or "300")
    except ValueError:
        return 300.0


def _aliases() -> dict[str, str]:
    cfg = _load_config()
    raw = cfg.get("aliases") or {}
    return {str(k): str(v) for k, v in raw.items()}


def _friendly_names() -> dict[str, str]:
    cfg = _load_config()
    raw = cfg.get("friendly_names") or {}
    return {str(k): str(v) for k, v in raw.items()}


def default_model_name() -> str:
    env = _env("OLLAMA_MODEL")
    if env:
        return env
    cfg = _load_config()
    return str(cfg.get("default_model") or "qwen2.5:3b")


def _broken_tags() -> set[str]:
    cfg = _load_config()
    if "broken_ollama_tags" in cfg:
        raw = cfg.get("broken_ollama_tags") or []
    else:
        raw = ["gemma_2b_ot"]
    return {str(x).lower() for x in raw}


def fallback_model_name() -> str:
    cfg = _load_config()
    return resolve_model_name(str(cfg.get("fallback_model") or "qwen2.5:3b"))


def _is_broken_tag(name: str) -> bool:
    n = _slug_from_tag(name or "").lower()
    broken = _broken_tags()
    return n in broken or f"{n}:latest" in broken or name.lower() in broken


def resolve_model_name(target: str) -> str:
    """slug / 別名 / 實際 Ollama 名 → Ollama model tag。"""
    t = (target or "").strip()
    if not t:
        return current_model_name() or default_model_name()
    if t.startswith("base:"):
        t = t[5:].strip()
    stem = t.split(":")[0]
    aliases = _aliases()
    if t in aliases:
        resolved = aliases[t]
    elif stem in aliases:
        resolved = aliases[stem]
    else:
        base = Path(t).name if ("/" in t or "\\" in t) else t
        if base in aliases:
            resolved = aliases[base]
        elif base.split(":")[0] in aliases:
            resolved = aliases[base.split(":")[0]]
        else:
            resolved = t
    if _is_broken_tag(resolved):
        fb = fallback_model_name()
        print(f"⚠️ 模型「{t}」在 Ollama 匯入異常，改以 {fb} 代替")
        return fb
    return resolved


def current_model_name() -> str:
    return _current_model or default_model_name()


def init(*, model: str | None = None) -> dict:
    """啟動時設定預設模型並探測 Ollama。"""
    global _current_model, _last_error
    _current_model = resolve_model_name(model or default_model_name())
    health = ping()
    if health.get("ok"):
        tags = [m.get("name") for m in health.get("models") or []]
        if _current_model not in tags and not _model_name_matches(_current_model, tags):
            _last_error = (
                f"Ollama 中找不到模型「{_current_model}」。"
                f"已安裝：{', '.join(tags[:8]) or '(無)'}。"
                f"請 ollama pull 或 ollama create。"
            )
            print(f"⚠️ {_last_error}")
        else:
            print(f"🦙 Ollama 就緒 | 模型={_current_model} | {_base_url()}")
    else:
        _last_error = health.get("error") or "Ollama 無法連線"
        print(f"⚠️ Ollama：{_last_error}")
    return health


def ping() -> dict:
    """GET /api/tags + 版本。"""
    try:
        data = _request("GET", "/api/tags", None)
        models = data.get("models") or []
        return {
            "ok": True,
            "base_url": _base_url(),
            "models": models,
            "model_count": len(models),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "base_url": _base_url()}


def is_ready() -> bool:
    return bool(ping().get("ok"))


def _model_name_matches(name: str, installed: list[str]) -> bool:
    """允許 qwen3_4b_ot 對上 qwen3_4b_ot:latest。"""
    n = (name or "").lower()
    for tag in installed:
        t = (tag or "").lower()
        if t == n or t.split(":")[0] == n or t.startswith(n + ":"):
            return True
    return False


def _installed_tags() -> list[str]:
    health = ping()
    if not health.get("ok"):
        return []
    return [m.get("name") or "" for m in health.get("models") or [] if m.get("name")]


def _slug_from_tag(tag: str) -> str:
    """UI／切換用 slug（保留 hub tag 如 qwen3:4b）。"""
    t = (tag or "").strip()
    if t.endswith(":latest"):
        return t[: -len(":latest")]
    return t


# UI 不顯示的重複／相容別名（仍可用於 resolve_model_name）
_UI_HIDDEN_ALIASES = frozenset({
    "gemma2b_ot",
    "qwen_ot_merged_model",
    "phi4_merged_model",
})

# 同一 target 只顯示一個別名時的優先 slug
_CANONICAL_ALIAS_BY_TARGET: dict[str, str] = {}


def _is_hub_path_alias(slug: str) -> bool:
    return "/" in (slug or "") or (slug or "").startswith("base:")


def _is_base_mirror_alias(slug: str) -> bool:
    return (slug or "").startswith("base_")


def _alias_redirect_target(slug: str) -> str:
    return _slug_from_tag(_aliases().get(slug) or slug)


def _classify_stage(slug: str) -> tuple[str, str]:
    s = (slug or "").lower()
    aliases = _aliases()
    if slug in aliases:
        tgt = _slug_from_tag(aliases[slug])
        if tgt != _slug_from_tag(slug) and not tgt.endswith("_ot") and "merged_model" not in tgt:
            return "alias", "Semi-Shield 別名"
    if s.startswith("base_") or s.startswith("base:"):
        return "base", "微調前"
    if any(k in s for k in ("_ot", "merged_model")):
        return "finetuned", "微調後"
    if ":" in slug and not s.endswith("_ot"):
        return "base", "Ollama 基底"
    return "base", "Ollama"


def _is_qwen3_model(name: str) -> bool:
    return "qwen3" in (name or "").lower()


def _request(method: str, path: str, body: dict | None) -> dict:
    url = f"{_base_url()}{path}"
    data = None
    headers = {"Content-Type": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=_timeout()) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail[:400]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"無法連線 Ollama（{_base_url()}）。請確認 ollama serve 已啟動。"
        ) from e


def switch_model(target: str) -> dict:
    """切換目前模型（Ollama 懶載入，無需卸載顯存）。"""
    global _current_model, _last_error
    name = resolve_model_name(target)
    if _is_broken_tag(target) and name != target:
        print(f"⚠️ 「{target}」匯入異常，已導向 {name}")
    health = ping()
    if not health.get("ok"):
        return {"ok": False, "error": health.get("error") or "Ollama 未連線"}

    tags = [m.get("name") for m in health.get("models") or []]
    req_slug = _slug_from_tag(target)
    if not _model_name_matches(name, tags):
        hint = (
            f"請執行 import_ollama_models.bat --only {req_slug}"
            if req_slug.endswith("_ot")
            else f"請執行：ollama pull {name} 或依 docs/OLLAMA.md 建立 Modelfile。"
        )
        return {
            "ok": False,
            "error": f"Ollama 未安裝模型「{friendly_label(req_slug or name)}」（{name}）。{hint}",
            "slug": req_slug,
            "installed": tags,
        }

    for tag in tags:
        if tag and (
            tag.lower() == name.lower()
            or tag.lower().split(":")[0] == name.lower()
            or _slug_from_tag(tag).lower() == name.lower()
        ):
            name = tag
            break

    prev = _current_model
    _current_model = name
    _last_error = ""
    switched = prev != name
    slug = _slug_from_tag(name)
    if switched:
        if req_slug != slug and req_slug != name:
            message = f"已切換為 {friendly_label(req_slug)}"
        else:
            message = f"已切換為 {friendly_label(slug)}"
    else:
        message = "已是目前使用的模型"
    print(f"🦙 Ollama 切換：{prev} → {name}")
    return {
        "ok": True,
        "switched": switched,
        "from_cache": True,
        "message": message,
        "model": name,
        "slug": req_slug or slug,
        "ollama_tag": slug,
        "label": friendly_label(req_slug or slug),
    }


def friendly_label(name: str) -> str:
    n = _slug_from_tag(name or "")
    base = n.split(":")[0]
    names = _friendly_names()
    if n in names:
        return names[n]
    if base in names:
        return names[base]
    return n.replace("_", " ").replace("-", " ")


def _installed_model_meta(tag: str, m: dict) -> dict:
    slug = _slug_from_tag(tag)
    stage, stage_label = _classify_stage(slug)
    size_gb = round((m.get("size") or 0) / (1024**3), 2)
    param = (m.get("details") or {}).get("parameter_size") or ""
    kind = "本地微調權重" if stage == "finetuned" and "merged" not in slug else (
        "Semi-Shield 別名" if stage == "finetuned" else "官方基底"
    )
    return {
        "slug": slug,
        "label": friendly_label(slug),
        "path": tag,
        "load_path": tag,
        "stage": stage,
        "stage_label": stage_label,
        "base_model_id": None,
        "desc": f"{kind} · {size_gb} GB" + (f" · {param}" if param else ""),
        "source": "ollama",
        "available": True,
        "unavailable_reason": "",
        "ollama": True,
        "backend": "ollama",
    }


def _installed_slug_set(tags: list[str]) -> set[str]:
    return {_slug_from_tag(t) for t in tags}


def _should_show_installed_tag(slug: str, installed_slugs: set[str]) -> bool:
    if _is_broken_tag(slug):
        return False
    if slug in _UI_HIDDEN_ALIASES or "merged_model" in slug:
        target = _aliases().get(slug)
        if target and _slug_from_tag(target) in installed_slugs:
            return False
    if _is_base_mirror_alias(slug):
        target = _aliases().get(slug)
        if target and _slug_from_tag(target) in installed_slugs:
            return False
    return True


def _should_show_alias_in_ui(alias_slug: str, target: str, seen_slugs: set[str]) -> bool:
    if alias_slug in _UI_HIDDEN_ALIASES:
        return False
    if _is_hub_path_alias(alias_slug) or _is_base_mirror_alias(alias_slug):
        return False
    if alias_slug == target:
        return False
    target_slug = _slug_from_tag(target)
    # target 已是安裝中的 OT 模型時，別名條目多餘
    if target_slug in seen_slugs and (target_slug.endswith("_ot") or "merged_model" in target_slug):
        return False
    return True


def _alias_redirect_items(seen_tags: list[str], cur: str, cur_slug: str) -> list[dict]:
    """別名 slug（如 gemma_2b_ot）→ 實際可用 Ollama tag（如 gemma2:2b）。"""
    out: list[dict] = []
    seen_slugs = {_slug_from_tag(t) for t in seen_tags}
    chosen_by_target: dict[str, str] = {}

    for alias_slug, target in _aliases().items():
        if not _should_show_alias_in_ui(alias_slug, target, seen_slugs):
            continue
        if not _model_name_matches(target, seen_tags):
            continue
        target_slug = _slug_from_tag(target)
        canonical = _CANONICAL_ALIAS_BY_TARGET.get(target_slug)
        if canonical and alias_slug != canonical:
            continue
        if target_slug in chosen_by_target:
            continue
        chosen_by_target[target_slug] = alias_slug

        resolved = resolve_model_name(alias_slug)
        stage, stage_label = _classify_stage(alias_slug)
        load_tag = target
        for t in seen_tags:
            if _model_name_matches(target, [t]) or _slug_from_tag(t) == target_slug:
                load_tag = t
                break
        out.append({
            "slug": alias_slug,
            "label": friendly_label(alias_slug),
            "path": load_tag,
            "load_path": load_tag,
            "stage": stage,
            "stage_label": stage_label,
            "base_model_id": target_slug,
            "desc": f"→ {_slug_from_tag(target)}",
            "source": "ollama",
            "available": True,
            "unavailable_reason": "",
            "active": (
                _model_name_matches(resolved, [cur])
                or alias_slug == cur_slug
                or _model_name_matches(load_tag, [cur])
            ),
            "ollama": True,
            "backend": "ollama",
        })
    return out


def list_models_for_ui() -> list[dict]:
    """供 /api/llm/models 使用。"""
    health = ping()
    cur = current_model_name()
    cur_slug = _slug_from_tag(cur)
    items: list[dict] = []
    seen: set[str] = set()
    installed_tags: list[str] = []

    if health.get("ok"):
        raw_models = health.get("models") or []
        installed_tags = [
            (m.get("name") or "").strip()
            for m in raw_models
            if (m.get("name") or "").strip()
        ]
        installed_slugs = _installed_slug_set(installed_tags)

        for m in raw_models:
            tag = (m.get("name") or "").strip()
            if not tag or tag in seen:
                continue
            slug = _slug_from_tag(tag)
            if not _should_show_installed_tag(slug, installed_slugs):
                continue
            seen.add(tag)
            meta = _installed_model_meta(tag, m)
            meta["active"] = _model_name_matches(slug, [cur]) or slug == cur_slug
            items.append(meta)

        alias_slugs = {i["slug"] for i in items}
        for alias_item in _alias_redirect_items(installed_tags, cur, cur_slug):
            if alias_item["slug"] in alias_slugs:
                continue
            items.append(alias_item)

    for alias_slug, ollama_name in _aliases().items():
        if (
            alias_slug.startswith("base:")
            or _is_hub_path_alias(alias_slug)
            or _is_base_mirror_alias(alias_slug)
            or alias_slug in _UI_HIDDEN_ALIASES
            or alias_slug in seen
        ):
            continue
        if _model_name_matches(ollama_name, list(seen)):
            continue
        if alias_slug in {i["slug"] for i in items}:
            continue
        stage, stage_label = _classify_stage(alias_slug)
        items.append({
            "slug": alias_slug,
            "label": friendly_label(alias_slug),
            "path": ollama_name,
            "load_path": ollama_name,
            "stage": stage,
            "stage_label": stage_label + "（未安裝）",
            "base_model_id": None,
            "desc": f"請執行 import_ollama_models.bat 或 ollama pull {ollama_name}",
            "source": "ollama",
            "available": False,
            "unavailable_reason": f"尚未在 Ollama 安裝 {ollama_name}",
            "active": False,
            "ollama": True,
            "backend": "ollama",
        })

    items.sort(
        key=lambda x: (
            0 if x.get("active") else 1,
            {"finetuned": 0, "alias": 1, "base": 2}.get(x.get("stage") or "base", 2),
            x.get("label") or "",
        )
    )
    return items


def current_info() -> dict:
    name = current_model_name()
    slug = _slug_from_tag(name)
    health = ping()
    tags = _installed_tags()
    stage, stage_label = _classify_stage(slug)
    model_ok = _model_name_matches(name, tags) or _model_name_matches(slug, tags)
    return {
        "slug": slug,
        "label": friendly_label(slug),
        "path": name,
        "load_path": name,
        "stage": stage,
        "stage_label": stage_label,
        "loaded": bool(health.get("ok")) and model_ok,
        "edge_mode": False,
        "device": "ollama",
        "runtime_device": "ollama",
        "speed_mode": "ollama",
        "backend": "ollama",
        "ollama_base_url": _base_url(),
        "ollama_ready": health.get("ok", False),
        "ollama_error": _last_error,
        "model_count": health.get("model_count", 0),
    }


def _is_gemma_model(name: str) -> bool:
    return "gemma" in (name or "").lower()


def _normalize_messages(messages: list[dict], *, model: str) -> list[dict]:
    out: list[dict] = []
    direct_hint = (
        "請直接以繁體中文回答使用者，勿輸出推理過程或內心獨白。"
        if _is_qwen3_model(model)
        else ""
    )
    gemma = _is_gemma_model(model)
    system_parts: list[str] = []

    for m in messages or []:
        role = (m.get("role") or "user").lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            if gemma:
                system_parts.append(content)
                continue
            role = "system"
            if direct_hint and direct_hint not in content:
                content = content.rstrip() + "\n" + direct_hint
        elif role in ("assistant", "agent", "bot", "ai"):
            role = "assistant"
        else:
            role = "user"
        out.append({"role": role, "content": content})

    if gemma and system_parts:
        blob = "\n\n".join(system_parts).strip()
        merged = False
        for i, m in enumerate(out):
            if m["role"] == "user":
                body = m.get("content") or ""
                out[i] = {
                    "role": "user",
                    "content": f"{blob}\n\n{body}".strip() if body else blob,
                }
                merged = True
                break
        if not merged:
            out.insert(0, {"role": "user", "content": blob})

    if direct_hint and not gemma and not any(m.get("role") == "system" for m in out):
        out.insert(0, {"role": "system", "content": direct_hint})
    return out


def _extract_chat_content(data: dict) -> str:
    msg = data.get("message") or {}
    content = (msg.get("content") or "").strip()
    if content:
        return content
    return (msg.get("thinking") or "").strip()


def _polish_qwen3_reply(text: str) -> str:
    """Qwen3 常把 chain-of-thought 寫進 content，取最後像回答的句子。"""
    if not text:
        return text
    t = text.strip()
    meta = re.compile(
        r"分析使用者|使用者本則問題|從斷點|若有對話紀錄|"
        r"作為\s*AI|這是一個友好的|必須依上文理解|"
        r"^首先[，,]?\s*用[户戶]|但指示說|指示說|可能的回應",
        re.I,
    )
    t = re.sub(r"^可能的回應[：:]\s*", "", t)
    t = re.sub(r"^[「\"'](.+)[」\"']$", r"\1", t.strip())
    for ln in reversed([x.strip() for x in t.splitlines() if x.strip()]):
        if meta.search(ln):
            continue
        if len(ln) <= 160 and re.search(r"您好|你好|很高興|協助|請問", ln):
            return ln.strip("\"'""「」")
    parts = re.split(r"(?<=[。！？!?])", t)
    for p in reversed(parts):
        p = p.strip()
        if not p or meta.search(p):
            continue
        if 6 <= len(p) <= 220 and not re.match(r"^(首先|作為|根據|指令)", p):
            return p
    if meta.search(t):
        return ""
    if len(t) > 400:
        tail = t[-280:].strip()
        return "" if meta.search(tail) else tail
    return t


def _looks_like_broken_output(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if re.search(r"peg-native format|Ollama HTTP 500|Ollama 推論失敗", t, re.I):
        return True
    if re.fullmatch(r"[<\s/|_\\imstartimend]+", t):
        return True
    if re.search(r"\[UNK_BYTE_|You're\[UNK", t):
        return True
    if re.fullmatch(r"(?:</>\s*)+", t):
        return True
    if t.count("</>") >= 3:
        return True
    if len(t) <= 12 and not re.search(r"[\u4e00-\u9fff]", t):
        return True
    return False


def _chat_once(
    model: str,
    messages: list[dict],
    *,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, str | None]:
    """單次 chat；回傳 (content, error)。"""
    token_budget = max(64, int(max_new_tokens or 512))
    if _is_qwen3_model(model):
        token_budget = max(token_budget, 384)
    body = {
        "model": model,
        "messages": _normalize_messages(messages, model=model),
        "stream": False,
        "think": False,
        "options": {
            "num_predict": token_budget,
            "temperature": float(temperature),
        },
    }
    try:
        data = _request("POST", "/api/chat", body)
        content = _extract_chat_content(data)
        if _is_qwen3_model(model):
            content = _polish_qwen3_reply(content)
        if _looks_like_broken_output(content):
            reason = data.get("done_reason") or "broken_output"
            return "", f"模型輸出異常（{reason}）"
        return (content or "").strip(), None
    except Exception as e:
        return "", str(e)


def chat(
    messages: list[dict],
    *,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """POST /api/chat 非串流；失敗時自動改以 fallback 模型重試。"""
    global _current_model, _last_error
    if not is_ready():
        msg = _last_error or "Ollama 未連線，請先執行 ollama serve"
        return f"⚠️ {msg}"

    model = resolve_model_name(current_model_name())
    if model != current_model_name():
        switch_model(model)

    tried: list[str] = []
    candidates = [model]
    fb = fallback_model_name()
    if fb not in candidates and not _is_gemma_model(model):
        candidates.append(fb)
    if _is_gemma_model(model):
        base = resolve_model_name("gemma2:2b")
        if base not in candidates:
            candidates.append(base)

    last_err = ""
    for m in candidates:
        if m in tried:
            continue
        tried.append(m)
        content, err = _chat_once(
            m, messages, max_new_tokens=max_new_tokens, temperature=temperature
        )
        if content:
            if m != current_model_name():
                print(f"🦙 Ollama 已自動改以 {m} 回覆（原模型異常）")
                switch_model(m)
            _last_error = ""
            return content
        last_err = err or "empty"
        print(f"❌ Ollama chat 失敗（{m}）：{last_err[:200]}")

    _last_error = last_err
    return (
        f"Ollama 推論失敗：{last_err[:240]}。"
        f"請在模型選單改選 {friendly_label(fallback_model_name())}。"
    )


def prime_chat() -> bool:
    """暖機：短問答丟棄。"""
    try:
        chat(
            [
                {
                    "role": "system",
                    "content": "你是 Semi-Shield Cyber Agent。一句繁中打招呼即可，勿推理。",
                },
                {"role": "user", "content": "你好"},
            ],
            max_new_tokens=64,
        )
        return True
    except Exception as e:
        print(f"⚠️ Ollama 暖機略過：{e}")
        return False


def is_small_model(name: str | None = None) -> bool:
    n = (name or current_model_name()).lower()
    return bool(
        re.search(
            r"gemma|0\.5b|1\.5b|2b|phi|mini|small",
            n,
        )
    )
