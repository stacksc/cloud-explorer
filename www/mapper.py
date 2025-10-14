#!/usr/bin/env python3
import os
import json, re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import hashlib
import random

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np

np.seterr(all='ignore')

from openai import OpenAI, AsyncOpenAI

sync_client = OpenAI()
async_client = AsyncOpenAI()

import asyncio
from tqdm import tqdm

def normalize_and_clamp(v):
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.clip(v, -1e3, 1e3)

    if v.ndim == 1:
        n = np.linalg.norm(v)
        return np.zeros_like(v) if n == 0 or not np.isfinite(n) else v / n

    # handle batch (2-D)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    v = v / norms
    return np.clip(v, -1.0, 1.0)

KEYWORDS = {
    "encryption": 0.15,
    "encrypt": 0.15,
    "identity": 0.1,
    "auth": 0.1,
    "password": 0.1,
    "mfa": 0.15,
    "key": 0.1,
    "certificate": 0.1,
    "tls": 0.1,
    "logging": 0.1,
    "audit": 0.1,
    "network": 0.05,
    "firewall": 0.1,
    "policy": 0.05,
    "access": 0.05,
    "iam": 0.1,
}

# === CONFIG ===
CATALOG_JSON = "/Users/cstacks/git/Apps.SystemTools/TorchLite/nist_data/output_catalog.json"
PROVIDERS_DIR = "./providers"
OUTPUT_JSON = "./command_to_controls.json"

# --- Speed knobs ---
MATCH_CHUNK = 2048          # commands per similarity chunk (tune 1024–4096)
DTYPE = np.float32          # smaller = faster; keep 32-bit for stability

# Narratives
MAX_LLM_CALLS = 2500        # total LLM calls allowed in a run
NARRATE_TOP_N_PER_CMD = 3   # you already have this; used in post-step


# Cache files (distinct)
EMB_CACHE_FILE = ".emb_cache.json"      # embedding vectors
NARR_CACHE_FILE = ".narr_cache.json"    # narrative text
AI_CACHE_FILE = ".ai_context_cache.json"  # optional misc cache

TOP_K = 10 
MIN_SCORE = 0.2
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
BATCH_SIZE = 100

# === LLM CONFIG ===
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.8
GENERATE_NARRATIVES = True

# Load caches
def load_json_safe(path):
    if os.path.exists(path):
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            return {}
    return {}

def save_json_safe(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# embedding cache
CACHE: Dict[str, List[float]] = load_json_safe(EMB_CACHE_FILE)

# narrative cache
NARR_CACHE: Dict[str, str] = load_json_safe(NARR_CACHE_FILE)

# AI context cache
ai_cache: Dict[str, dict] = load_json_safe(AI_CACHE_FILE)


def get_ai_context(text):
    if text in ai_cache:
        return ai_cache[text]
    ai_context = generate_ai_context(text)  # your expensive call
    ai_cache[text] = ai_context
    return ai_context

# -------------------- Narrative cache --------------------
def load_json(path: str) -> dict:
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

NARR_CACHE: Dict[str, str] = load_json(NARR_CACHE_FILE)

def cache_key_for_narrative(provider: str, full_cmd: str, control_id: str, control_title: str, control_obj: str) -> str:
    m = hashlib.sha1()
    m.update(provider.encode("utf-8"))
    m.update(full_cmd.encode("utf-8"))
    m.update(control_id.encode("utf-8"))
    m.update((control_title or "").encode("utf-8"))
    m.update((control_obj or "").encode("utf-8"))
    return m.hexdigest()

_BOILERPLATE_SNIPPETS = [
    "can be used to address aspects of this control",
    "related configurations or actions in the cloud environment",
    "helps meet the requirements of the control at a high level",
]

def postprocess_narrative(text: str) -> str:
    if not text:
        return ""
    t = text.strip()

    # Trim overlong outputs to ~3 sentences
    sentences = re.split(r"(?<=[.!?])\s+", t)
    if len(sentences) > 3:
        t = " ".join(sentences[:3]).strip()

    # Remove or rewrite boilerplate-y phrasing
    lower = t.lower()
    if any(s in lower for s in _BOILERPLATE_SNIPPETS):
        t = re.sub(
            r"\bcan be used to address aspects of this control\b",
            random.choice([
                "provides concrete evidence relevant to this requirement",
                "directly validates configuration relevant to this requirement",
                "offers actionable verification against this requirement"
            ]),
            t,
            flags=re.IGNORECASE
        )
        t = re.sub(
            r"\brelated configurations or actions in the cloud environment\b",
            random.choice([
                "the specific settings under test",
                "the exact policy knobs and telemetry that matter",
                "the enforceable controls surfaced by the API"
            ]),
            t,
            flags=re.IGNORECASE
        )

    # Tighten whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_prose(text: str) -> str:
    if not text:
        return ""
    # Remove {{ ... }} blocks
    text = re.sub(r"\{\{.*?\}\}", "", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def keyword_boost(text_a: str, text_b: str) -> float:
    score = 0.0
    a_low, b_low = text_a.lower(), text_b.lower()
    for kw, bonus in KEYWORDS.items():
        if kw in a_low and kw in b_low:
            score += bonus
    return score

# -------------------- Embedding cache --------------------
def load_cache(path: str) -> Dict[str, List[float]]:
    if Path(path).exists():
        try:
            return json.loads(Path(path).read_text())
        except Exception:
            return {}
    return {}

def save_cache(path: str, cache: Dict[str, List[float]]):
    Path(path).write_text(json.dumps(cache))

def batch_embed_texts(texts: List[str]) -> List[np.ndarray]:
    if not texts:
        return []
    resp = sync_client.embeddings.create(input=texts, model=EMBED_MODEL)
    return [np.array(item.embedding, dtype=np.float32) for item in resp.data]

def get_embeddings(texts: List[str]) -> Dict[str, np.ndarray]:
    """
    Given a list of texts, return a dict {text: vector}.
    Uses cache and batches API calls.
    """
    result: Dict[str, np.ndarray] = {}
    to_embed: List[Tuple[str, str]] = []  # (text, cache_key)

    for t in texts:
        t = (t or "").strip()
        if not t:
            continue

        # 🔹 Use stable hash for cache key
        cache_key = hashlib.sha1(t.encode("utf-8")).hexdigest()

        if cache_key in CACHE:
            result[t] = np.array(CACHE[cache_key], dtype=np.float32)
        else:
            to_embed.append((t, cache_key))

    for i in tqdm(range(0, len(to_embed), BATCH_SIZE), desc="Embedding", unit="batch"):
        batch = to_embed[i:i+BATCH_SIZE]
        batch_texts = [txt for txt, _ in batch]
        vecs = batch_embed_texts(batch_texts)

        for (txt, cache_key), vec in zip(batch, vecs):
            CACHE[cache_key] = vec.tolist()
            result[txt] = vec

    return result

def safe_cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return -1.0
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return -1.0
    return float(np.dot(a, b) / (na * nb))

# -------------------- Data models --------------------
@dataclass
class Control:
    id: str
    norm_ids: List[str]
    title: str
    family: str
    description: str   # cleaned short prose
    full_text: str     # full concatenated text for embeddings
    vec: Optional[np.ndarray]

@dataclass
class CommandEntry:
    key: str
    provider: str
    service: str
    full_cmd: str
    bin_hint: str
    description: str
    emb_text: str
    vec: Optional[np.ndarray] = None

# -------------------- Builders / Loaders --------------------
def build_control_text(c: dict) -> str:
    """
    Extract the best human-readable description from a control.
    Prefers OSCAL 'statement.prose'. Cleans {{ insert: }} placeholders.
    """
    if "parts" in c:
        parts = c["parts"].values() if isinstance(c["parts"], dict) else c["parts"]
        for part in parts:
            if isinstance(part, dict):
                if part.get("name") == "statement" and "prose" in part:
                    return clean_prose(part["prose"])
                if "prose" in part:
                    return clean_prose(part["prose"])
                if "text" in part:
                    return clean_prose(part["text"])
    if c.get("text"):
        return clean_prose(c["text"])
    bits = []
    if c.get("title"): bits.append(c["title"])
    if c.get("family"): bits.append(c["family"])
    return " – ".join(bits)

def load_controls(path: str) -> Dict[str, Control]:
    raw = json.loads(Path(path).read_text())
    ctrls: Dict[str, Control] = {}
    for cid, c in raw.items():
        clean_desc = build_control_text(c)        # short readable text
        rich_bits = []
        if c.get("title"): rich_bits.append(c["title"])
        if c.get("family"): rich_bits.append(c["family"])
        if clean_desc: rich_bits.append(clean_desc)
        full_text = " ".join(rich_bits)

        ctrls[cid] = Control(
            id=cid,
            norm_ids=list(dict.fromkeys(c.get("norm_ids", []))),
            title=c.get("title", ""),
            family=c.get("family", ""),
            description=clean_desc,
            full_text=full_text,
            vec=None
        )
    return ctrls

def flatten_provider_file(path: Path) -> List[CommandEntry]:
    data = json.loads(path.read_text())
    provider = path.stem

    # Normalize root for inconsistent formats
    if provider in data and isinstance(data[provider], dict):
        root = data[provider]
    elif "subcommands" in data:
        root = data
    elif "commands" in data:
        root = data
    else:
        root = {"subcommands": data}

    results: List[CommandEntry] = []

    def extract_help(node: dict) -> str:
        # prefer 'help' or 'usage', fallback to description-like text
        for key in ("help", "usage", "description", "summary"):
            if key in node and isinstance(node[key], str) and node[key].strip():
                return node[key].strip()
        return ""

    def walk(node: dict, tokens: List[str], parent_help: str = ""):
        subs = None
        # support multiple possible nesting keys
        for key in ("subcommands", "commands", "resources"):
            if key in node and isinstance(node[key], dict):
                subs = node[key]
                break

        if subs:
            for k, child in subs.items():
                group_help = parent_help or extract_help(node)
                walk(child, tokens + [k], group_help)
        else:
            full_cmd = " ".join(tokens)
            service = tokens[0] if tokens else ""
            help_txt = extract_help(node)
            options = list((node.get("options") or node.get("params") or {}).keys())

            # Build embedding text with fallback
            emb_parts = [
                f"Provider: {provider}",
                f"Command: {full_cmd}",
            ]
            if parent_help:
                emb_parts.append(f"Group description: {parent_help}")
            if help_txt:
                emb_parts.append(f"Help: {help_txt}")
            if options:
                emb_parts.append(f"Options: {', '.join(options)}")

            emb_text = " ".join(emb_parts).strip()
            if not emb_text:
                # Safety fallback to avoid empty embedding
                emb_text = f"Provider: {provider} Command: {full_cmd or provider}"

            key = f"{provider}:{full_cmd}" if full_cmd else f"{provider}:<root>"

            results.append(
                CommandEntry(
                    key=key,
                    provider=provider,
                    service=service,
                    full_cmd=full_cmd,
                    bin_hint=provider,
                    description=help_txt or parent_help or "",
                    emb_text=emb_text,
                )
            )

    # Handle either top-level list or dict
    if not any(k in root for k in ("subcommands", "commands", "resources")):
        walk(root, [])
    else:
        for k, sub in (root.get("subcommands") or root.get("commands") or root.get("resources") or {}).items():
            walk(sub, [k])

    return results

def load_all_commands(providers_dir: str) -> List[CommandEntry]:
    entries: List[CommandEntry] = []
    for file in sorted(Path(providers_dir).glob("*.json")):
        print(f"📂 Loading provider file: {file.name}")
        entries.extend(flatten_provider_file(file))
    print(f"✅ Loaded {len(entries)} leaf commands from {providers_dir}")
    return entries

async def generate_control_narrative(cmd: CommandEntry, ctrl: Control, relation: str) -> str:
    if not GENERATE_NARRATIVES:
        return ""

    ck = cache_key_for_narrative(
        cmd.provider,
        cmd.full_cmd,
        ctrl.id,
        ctrl.title,
        ctrl.description or ctrl.full_text
    )
    
    if ck in NARR_CACHE:
        return NARR_CACHE[ck]

    system_msg = ("You are a senior cloud compliance analyst..."
                  "• Intent: ...\n• Evidence: ...")

    user_msg = (f"CONTROL: {ctrl.id} {ctrl.title}\n"
                f"Objective: {(ctrl.description or ctrl.title or '').strip()}\n"
                f"OPERATION: {cmd.bin_hint} {cmd.full_cmd}".strip() + "\n"
                f"Description: {(cmd.description or cmd.emb_text)}\n"
                f"Relation: {relation}\n"
                "Write 2–3 sentences. Use the Intent/Evidence structure.")

    # simple bounded retry
    for attempt in range(3):
        try:
            resp = await async_client.chat.completions.create(
                model=LLM_MODEL,
                temperature=0,
                messages=[{"role":"system","content":system_msg},
                          {"role":"user","content":user_msg}],
                timeout=30,
            )
            text = postprocess_narrative(resp.choices[0].message.content.strip())
            NARR_CACHE[ck] = text
            return text
        except Exception:
            await asyncio.sleep(0.8 * (attempt + 1))
    return f"Intent: {ctrl.title or ctrl.family}.\nEvidence: `{cmd.bin_hint} {cmd.full_cmd}` {relation} relevant settings."

# -------------------- Matcher --------------------

def sanitize_vectors(vectors: List[np.ndarray], ids: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Fully sanitize and normalize vectors to guarantee finite, unit-norm data.
    """
    clean_vecs, clean_ids = [], []
    for v, i in zip(vectors, ids):
        if v is None or not isinstance(v, np.ndarray):
            continue
        v = np.nan_to_num(v.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        if v.ndim != 1 or v.size == 0:
            continue
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < 1e-12:
            continue
        v = v / n
        clean_vecs.append(v)
        clean_ids.append(i)

    if not clean_vecs:
        raise RuntimeError("❌ No valid vectors to sanitize.")

    mat = np.stack(clean_vecs).astype(np.float32)
    # Secondary pass: eliminate any new NaNs/Infs, renormalize, clamp
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    mat = np.clip(mat, -1.0, 1.0)

    print(f"✅ Sanitized control vectors: {len(clean_vecs)}/{len(vectors)}")
    return mat, clean_ids


def sanitize_single(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Guarantee safe single vector for query."""
    if v is None or not isinstance(v, np.ndarray):
        return None
    v = np.nan_to_num(v.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if v.ndim != 1 or v.size == 0:
        return None
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < 1e-12:
        return None
    v = np.clip(v / n, -1.0, 1.0)
    return v.astype(np.float32)

def match_commands_to_controls(cmds: List[CommandEntry], ctrls: Dict[str, Control]) -> Dict[str, dict]:
    """
    Vectorized cosine matcher in chunks (no sklearn, no LLM in the hot path).
    """
    # --- Build control matrix (unit-norm, finite, clipped) ---
    ctrl_vecs, ctrl_ids = [], []
    for cid, c in ctrls.items():
        v = c.vec
        if v is None or not isinstance(v, np.ndarray): 
            continue
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(DTYPE, copy=False)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < 1e-8: 
            continue
        ctrl_vecs.append((v / n).clip(-1.0, 1.0))
        ctrl_ids.append(cid)

    if not ctrl_vecs:
        raise RuntimeError("❌ No valid control vectors after sanitization!")

    ctrl_matrix = np.vstack(ctrl_vecs).astype(DTYPE, copy=False)  # (C, D)
    ctrl_T = np.nan_to_num(ctrl_matrix.T, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Build command matrix (same sanitation) ---
    keys, meta, cmd_vecs = [], [], []
    for cmd in cmds:
        v = cmd.vec
        if v is None or not isinstance(v, np.ndarray): 
            continue
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(DTYPE, copy=False)
        n = np.linalg.norm(v)
        if not np.isfinite(n) or n < 1e-8: 
            continue
        cmd_vecs.append((v / n).clip(-1.0, 1.0))
        keys.append(cmd.key)
        meta.append(cmd)

    if not cmd_vecs:
        return {}

    cmd_matrix = np.vstack(cmd_vecs).astype(DTYPE, copy=False)    # (N, D)

    # --- Chunked similarity + Top-K ---
    K = TOP_K
    N = cmd_matrix.shape[0]
    output: Dict[str, dict] = {}

    for start in tqdm(range(0, N, MATCH_CHUNK), desc="Matching", unit="chunk"):
        end = min(start + MATCH_CHUNK, N)
        block = cmd_matrix[start:end]                              # (B, D)
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        sims = block @ ctrl_T                                      # (B, C)
        sims = np.nan_to_num(sims, nan=0.0, posinf=0.0, neginf=0.0)

        # Get unsorted top-K indices per row
        idx_unsorted = np.argpartition(-sims, K-1, axis=1)[:, :K]
        row_idx = np.arange(sims.shape[0])[:, None]
        # Sort those K by score descending
        idx_sorted = np.argsort(-sims[row_idx, idx_unsorted], axis=1)
        topk = idx_unsorted[row_idx, idx_sorted]

        # Build output (no LLM yet)
        for r, cmd in enumerate(meta[start:end]):
            matched = []
            for rank, j in enumerate(topk[r]):
                score = float(sims[r, j])
                if score < MIN_SCORE:
                    continue
                cobj = ctrls[ctrl_ids[j]]
                matched.append({
                    "id": cobj.id,
                    "norm_ids": cobj.norm_ids,
                    "title": cobj.title,
                    "family": cobj.family,
                    "description": cobj.description,
                    "full_text": cobj.full_text,
                    "match_score": round(score, 3),
                    "match_type": "semantic",
                    "rationale": "Vectorized cosine similarity.",
                    # placeholder; we'll fill ai_context in post-step
                    "ai_context": None,
                })

            output[cmd.key] = {
                "provider": cmd.provider,
                "service": cmd.service,
                "command": f"{cmd.bin_hint} {cmd.full_cmd}".strip(),
                "description": cmd.description,
                "matched_controls": matched,
            }

    return output

import itertools
counter = itertools.count(1)

async def enrich_narratives_async(mapping, ctrls, max_calls=MAX_LLM_CALLS, top_n_per_cmd=NARRATE_TOP_N_PER_CMD):
    if not GENERATE_NARRATIVES or max_calls <= 0:
        return

    sem = asyncio.Semaphore(12)
    lock = asyncio.Lock()
    completed = 0
    SAVE_EVERY = 500

    async def process_one(cmd_key, bundle):
        nonlocal completed
        cmd_desc = bundle.get("description", "")
        cmd_stub = CommandEntry(
            key=cmd_key,
            provider=bundle.get("provider", ""),
            service=bundle.get("service", ""),
            full_cmd=bundle.get("command", ""),
            bin_hint=bundle.get("command", "").split(" ", 1)[0] if bundle.get("command") else "",
            description=cmd_desc,
            emb_text=cmd_desc or "",
        )
        matches = sorted(bundle.get("matched_controls", []),
                         key=lambda m: m["match_score"], reverse=True)[:top_n_per_cmd]

        for m in matches:
            async with lock:
                i = next(counter)
                if i > max_calls:
                    return
            ctrl = ctrls[m["id"]]
            async with sem:
                narrative = await generate_control_narrative(cmd_stub, ctrl, "verifies")
                m["ai_context"] = {
                    "relation": "verifies",
                    "narrative_seed": narrative,
                    "confidence": round(m["match_score"], 3),
                }

            async with lock:
                completed += 1
                if completed % SAVE_EVERY == 0:
                    print(f"💾 Checkpoint: {completed} narratives")
                    # ✅ only save small stuff; do NOT dump mapping here
                    save_json_safe(NARR_CACHE_FILE, NARR_CACHE)
                    # optional: a tiny progress file
                    save_json_safe(".progress.json", {"completed": completed})
            
    # Schedule in manageable chunks
    items = list(mapping.items())
    BATCH = 250
    for b in range(0, len(items), BATCH):
        batch = items[b:b+BATCH]
        tasks = [asyncio.create_task(process_one(k, v)) for k, v in batch]
        await asyncio.gather(*tasks)

def build_ai_context(cmd: CommandEntry, control: Control, score: float) -> dict:
    """
    Build reasoning context and an actually useful narrative.
    """
    text = (cmd.description or cmd.emb_text or "").lower()

    if any(k in text for k in ["create", "add", "enable", "set", "configure", "update", "put"]):
        relation = "implements"
    elif any(k in text for k in ["show", "list", "get", "describe", "view", "check"]):
        relation = "verifies"
    elif any(k in text for k in ["delete", "remove", "disable", "revoke"]):
        relation = "remediates"
    else:
        relation = "supports"

    control_intent = control.title or (control.description.split(".")[0] if control.description else control.family)
    capability = cmd.description or (cmd.emb_text.split("Help:")[-1].strip()[:200] if "Help:" in cmd.emb_text else cmd.emb_text[:200])

    # Generate concise narrative (2–3 sentences) with structure
    narrative = generate_control_narrative(cmd, control, relation)

    return {
        "relation": relation,
        "control_intent": control_intent,
        "command_capability": capability,
        "potential_action": f"Run `{cmd.bin_hint} {cmd.full_cmd}` and record results as evidence for {control.id.upper()}.",
        "compliance_relevance": f"This command {relation} the control by exposing or enforcing the exact settings auditors will review.",
        "narrative_seed": narrative,   # your frontend reads this; we’re overwriting with improved text
        "confidence": round(score, 3)
    }

# -------------------- Main --------------------
def main():
    # 1) Load controls & command entries (no embeddings yet)
    controls = load_controls(CATALOG_JSON)
    commands = load_all_commands(PROVIDERS_DIR)

    # 2) Collect the exact texts we will embed (controls + commands)
    ctrl_texts = [c.full_text for c in controls.values()]
    cmd_texts = [ce.emb_text for ce in commands]
    all_texts = ctrl_texts + cmd_texts
    print(f"📊 Preparing to embed {len(all_texts)} total texts...")

    # 3) Embed and cache
    embeddings = get_embeddings(all_texts)

    # 4) Inject vectors back into control objects
    missing_ctrl = 0
    for c in controls.values():
        c.vec = embeddings.get(c.full_text)
        if c.vec is None:
            missing_ctrl += 1

    # 5) Inject vectors into command entries
    missing_cmd = 0
    for ce in commands:
        ce.vec = embeddings.get(ce.emb_text)
        if ce.vec is None:
            missing_cmd += 1

    print(f"🔎 Controls with vectors: {len(controls)-missing_ctrl}/{len(controls)}")
    print(f"🔎 Commands with vectors: {len(commands)-missing_cmd}/{len(commands)}")
    print("🧹 Normalizing and clamping embeddings...")

    for c in controls.values():
        if c.vec is not None:
            c.vec = normalize_and_clamp(c.vec)
    for ce in commands:
        if ce.vec is not None:
            ce.vec = normalize_and_clamp(ce.vec)
    
    # Final sanity sweep before matching
    for c in controls.values():
        if c.vec is not None:
            if not np.isfinite(c.vec).all() or np.linalg.norm(c.vec) == 0:
                c.vec = None
    for ce in commands:
        if ce.vec is not None:
            if not np.isfinite(ce.vec).all() or np.linalg.norm(ce.vec) == 0:
                ce.vec = None

    # 6) Match
    mapping = match_commands_to_controls(commands, controls)
    asyncio.run(enrich_narratives_async(mapping, controls, max_calls=MAX_LLM_CALLS, top_n_per_cmd=NARRATE_TOP_N_PER_CMD))

    # (Optional) quick sample print to verify matches exist
    sample_with_matches = next((k for k, v in mapping.items() if v["matched_controls"]), None)
    if sample_with_matches:
        print(f"🧪 Sample matched command: {sample_with_matches}")
        print(json.dumps(mapping[sample_with_matches]["matched_controls"][:3], indent=2, ensure_ascii=False))
    else:
        print("⚠️ No matches survived MIN_SCORE filter. Try lowering MIN_SCORE or check embeddings.")

    # 7) Save
    Path(OUTPUT_JSON).write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    print(f"✅ Wrote {len(mapping)} command mappings to {OUTPUT_JSON}")

    # 8) Persist embedding + narrative caches
    save_cache(EMB_CACHE_FILE, CACHE)
    save_json_safe(NARR_CACHE_FILE, NARR_CACHE)
    save_json_safe(AI_CACHE_FILE, ai_cache)


if __name__ == "__main__":
    main()

