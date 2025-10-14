# torlite/flask_server.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from deepdiff import DeepDiff
from typing import Any, Dict, List, Tuple
from openai import OpenAI
from markupsafe import Markup
import os
import logging
import re
import json
import markdown
import gzip
import io
import hashlib
import time
from flask import Flask, request, render_template, redirect, url_for, session, jsonify, send_from_directory, make_response, send_file, Response, stream_with_context, abort
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import tempfile, uuid
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from collections import Counter, defaultdict

from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.ext.flask.flask_middleware import FlaskMiddleware

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
WWW_DIR = os.path.join(BASE_DIR, "www")
PROVIDERS_DIR = os.path.join(WWW_DIR, "providers")
_dd_key_re = re.compile(r"\['([^']+)'\]|\[(\d+)\]")

app = Flask(
    __name__,
    static_folder=WWW_DIR,
    static_url_path="",
    template_folder=WWW_DIR  # 👈 this is critical
)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-please")

# Application Insights setup
key = os.getenv("APPINSIGHTS_INSTRUMENTATIONKEY")
if key:
    # Attach Flask middleware for request telemetry
    middleware = FlaskMiddleware(
        app,
        exporter=AzureExporter(connection_string=f"InstrumentationKey={key}"),
        sampler=ProbabilitySampler(1.0)
    )

    # Send logs to Azure
    handler = AzureLogHandler(connection_string=f"InstrumentationKey={key}")
    logging.getLogger(__name__).addHandler(handler)

# Allow large local uploads (e.g., two 60MB files)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512MB

# Approx baseline monthly costs per resource type (you can refine later)
COST_PROFILE = {
    "Microsoft.Compute/virtualMachines": 120,
    "Microsoft.Storage/storageAccounts": 40,
    "Microsoft.Network/publicIPAddresses": 8,
    "Microsoft.Sql/servers": 200,
    "Microsoft.KeyVault/vaults": 30,
    "unknown": 20
}

USERS = {
    "admin": generate_password_hash("L0stC0c0nut*"),
    "demo": generate_password_hash("StrongPassword123!")
}

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# ---------------------------- static routes ----------------------------

@app.route("/")
@login_required
def index():
    """Serve the main app page (index.html)."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/providers/<path:filename>")
def providers(filename):
    """Serve provider JSON files (az.json, aws.json, etc)."""
    return send_from_directory(PROVIDERS_DIR, filename)


# keep both names to be safe with your frontend
@app.route("/controls_mapping.json")
def controls_mapping():
    return send_from_directory(app.static_folder, "controls_mapping.json")

@app.route("/command_to_controls/<provider>.json")
def serve_provider_mapping(provider):
    provider = provider.lower()
    valid = {"az", "aws", "gcloud", "oci", "ovhai", "aliyun", "ibmcloud"}
    if provider not in valid:
        return jsonify({"error": "Invalid provider"}), 400

    # look inside the static folder (www/)
    gz_path = os.path.join(app.static_folder, f"{provider}_command_to_controls.json.gz")
    if os.path.exists(gz_path):
        with open(gz_path, "rb") as f:
            data = f.read()
        resp = make_response(data)
        resp.headers["Content-Encoding"] = "gzip"
        resp.headers["Content-Type"] = "application/json"
        resp.headers["Cache-Control"] = "no-cache"
        return resp

    return jsonify({"error": f"File not found for provider '{provider}'"}), 404

# ----------------------------- helpers --------------------------------

def collect_resources(node: Any, out: Dict[str, Any]) -> None:
    """
    Recursively walk JSON and collect any dict that looks like a resource
    (has an 'id' key). Works for Azure discovery snapshots.
    """
    if isinstance(node, dict):
        # If it looks like a resource, collect and stop descending this branch.
        rid = node.get("id")
        if isinstance(rid, str) and rid:
            out[rid] = node
            return
        # Otherwise, keep walking.
        for v in node.values():
            collect_resources(v, out)
    elif isinstance(node, list):
        for item in node:
            collect_resources(item, out)

def dd_path_tokens(dd_path: str) -> List[Any]:
    """
    Convert DeepDiff path like: root['properties']['tags']['env'] or root['arr'][0]
    into tokens: ['properties','tags','env'] or ['arr', 0]
    """
    tokens: List[Any] = []
    for key, idx in _dd_key_re.findall(dd_path):
        if key:
            tokens.append(key)
        else:
            tokens.append(int(idx))
    return tokens

def tokens_to_dot(tokens: List[Any]) -> str:
    """Turn tokens into a display path like properties.tags.env or arr[0].name."""
    parts: List[str] = []
    for t in tokens:
        if isinstance(t, int):
            # index
            if parts:
                parts[-1] = f"{parts[-1]}[{t}]"
            else:
                parts.append(f"[{t}]")
        else:
            parts.append(t)
    return ".".join(parts)

def get_by_tokens(obj: Any, tokens: List[Any]) -> Any:
    """Follow tokens into obj to retrieve a value."""
    cur = obj
    for t in tokens:
        if isinstance(t, int):
            cur = cur[t] if isinstance(cur, list) and 0 <= t < len(cur) else None
        else:
            cur = cur.get(t) if isinstance(cur, dict) else None
        if cur is None:
            break
    return cur

def fast_hash(obj: Any) -> str:
    """
    Compute a stable hash of a JSON-serializable object.
    Used to skip identical resources before DeepDiff.
    """
    try:
        # sort_keys ensures consistent hash for same content
        data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(data).hexdigest()
    except Exception:
        # fallback: repr-based hash if not serializable
        return hashlib.sha1(repr(obj).encode("utf-8")).hexdigest()

def diff_single_resource(rid: str, a: dict, b: dict) -> List[Dict[str, Any]]:
    """Compute per-resource DeepDiff and return flattened result list."""
    out = []
    dd = DeepDiff(a, b, ignore_order=True, verbose_level=1, cache_size=5000)

    # 1) direct value changes
    for dd_path, detail in dd.get("values_changed", {}).items():
        tokens = dd_path_tokens(dd_path)
        field_path = tokens_to_dot(tokens)
        out.append({
            "type": "changed",
            "path": f"{rid}.{field_path}",
            "old_value": detail.get("old_value"),
            "new_value": detail.get("new_value"),
        })

    # 2) keys added
    for dd_path in dd.get("dictionary_item_added", []):
        tokens = dd_path_tokens(dd_path)
        field_path = tokens_to_dot(tokens)
        new_val = get_by_tokens(b, tokens)
        out.append({
            "type": "changed",
            "path": f"{rid}.{field_path}",
            "old_value": None,
            "new_value": new_val,
        })

    # 3) keys removed
    for dd_path in dd.get("dictionary_item_removed", []):
        tokens = dd_path_tokens(dd_path)
        field_path = tokens_to_dot(tokens)
        old_val = get_by_tokens(a, tokens)
        out.append({
            "type": "changed",
            "path": f"{rid}.{field_path}",
            "old_value": old_val,
            "new_value": None,
        })

    # 4) type changes
    for dd_path, detail in dd.get("type_changes", {}).items():
        tokens = dd_path_tokens(dd_path)
        field_path = tokens_to_dot(tokens)
        out.append({
            "type": "changed",
            "path": f"{rid}.{field_path}",
            "old_value": detail.get("old_value"),
            "new_value": detail.get("new_value"),
        })

    # 5) iterable item add/remove
    for dd_path, new_val in dd.get("iterable_item_added", {}).items():
        tokens = dd_path_tokens(dd_path)
        field_path = tokens_to_dot(tokens)
        out.append({
            "type": "changed",
            "path": f"{rid}.{field_path}",
            "old_value": None,
            "new_value": new_val,
        })
    for dd_path, old_val in dd.get("iterable_item_removed", {}).items():
        tokens = dd_path_tokens(dd_path)
        field_path = tokens_to_dot(tokens)
        out.append({
            "type": "changed",
            "path": f"{rid}.{field_path}",
            "old_value": old_val,
            "new_value": None,
        })

    return out

def deep_diff(snapshot1: Any, snapshot2: Any) -> List[Dict[str, Any]]:
    """
    Optimized diff: uses hashing to skip identical resources and
    parallelizes per-resource DeepDiff for speed.
    """
    start = time.time()

    res1: Dict[str, Any] = {}
    res2: Dict[str, Any] = {}
    collect_resources(snapshot1, res1)
    collect_resources(snapshot2, res2)

    ids1 = set(res1.keys())
    ids2 = set(res2.keys())

    added_ids = sorted(ids2 - ids1)
    removed_ids = sorted(ids1 - ids2)
    common_ids = sorted(ids1 & ids2)

    out: List[Dict[str, Any]] = []

    # Added / Removed
    for rid in added_ids:
        out.append({"type": "added", "path": rid, "value": res2[rid]})
    for rid in removed_ids:
        out.append({"type": "removed", "path": rid, "value": res1[rid]})

    # --- Pre-hash both sides to skip identical resources ---
    hashes1 = {rid: fast_hash(res1[rid]) for rid in common_ids}
    hashes2 = {rid: fast_hash(res2[rid]) for rid in common_ids}
    changed_ids = [rid for rid in common_ids if hashes1[rid] != hashes2[rid]]

    print(f"⚡ Identical resources skipped: {len(common_ids) - len(changed_ids)} / {len(common_ids)}", flush=True)

    # --- Parallel diff only the changed ones ---
    results = []
    max_workers = min(4, os.cpu_count() or 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(diff_single_resource, rid, res1[rid], res2[rid]): rid for rid in changed_ids}
        for fut in as_completed(futures):
            try:
                results.extend(fut.result())
            except Exception as e:
                rid = futures[fut]
                print(f"❌ DeepDiff failed for {rid}: {e}", flush=True)

    out.extend(results)
    print(f"✅ deep_diff completed in {time.time() - start:.2f}s, total={len(out)} diffs", flush=True)
    return out

def load_maybe_gzip(file_storage):
    import sys

    raw = file_storage.read()
    size = len(raw)
    print(f"DEBUG: received {size} bytes from {file_storage.filename}", flush=True)

    # Quick sanity: must be valid JSON text or gzip start
    head = raw[:4]
    print(f"DEBUG: first bytes {head}", flush=True)

    # Detect gzip
    if head[:2] == b'\x1f\x8b':
        print("DEBUG: detected gzip header", flush=True)
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
                text = gz.read().decode("utf-8")
            print("DEBUG: gzip decompression OK", flush=True)
            return json.loads(text)
        except Exception as e:
            print("❌ gzip decompression failed:", e, flush=True)
            raise

    # Not gzip: attempt plain UTF-8 JSON parse
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except json.JSONDecodeError as e:
        print(f"❌ JSON error at char {e.pos}, line {e.lineno}, col {e.colno}", flush=True)
        tail = raw[max(0, e.pos-120):e.pos+120].decode("utf-8", errors="ignore")
        print("---- around error ----", flush=True)
        print(tail, flush=True)
        print("---- end snippet ----", flush=True)
        raise

# ------------------------------ API -----------------------------------

@app.route("/compare", methods=["POST"])
@login_required
def compare():
    try:
        f1 = request.files.get("file1")
        f2 = request.files.get("file2")
        if not f1 or not f2:
            return jsonify({"error": "Both files required"}), 400

        # just log a peek without consuming
        f1_bytes = f1.stream.read(8)
        f1.stream.seek(0)
        f2_bytes = f2.stream.read(8)
        f2.stream.seek(0)
        print(f"File1: {f1.filename}, first bytes: {f1_bytes}")
        print(f"File2: {f2.filename}, first bytes: {f2_bytes}")

        data1 = load_maybe_gzip(f1)
        data2 = load_maybe_gzip(f2)

        diffs = deep_diff(data1, data2)
        added   = [d for d in diffs if d["type"] == "added"]
        removed = [d for d in diffs if d["type"] == "removed"]
        changed = [d for d in diffs if d["type"] == "changed"]

        return jsonify({"added": added, "removed": removed, "changed": changed})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No data provided"}), 400

        added = data.get("added", [])
        removed = data.get("removed", [])
        changed = data.get("changed", [])

        # ---------------- Basic counts ----------------
        summary = {
            "total_added": len(added),
            "total_removed": len(removed),
            "total_changed": len(changed),
            "net_delta": len(added) - len(removed),
        }

        # ---------------- Group by region ----------------
        def get_region(path: str, value: dict) -> str:
            rid = path or value.get("id", "")
            parts = rid.split("/")
            for i, p in enumerate(parts):
                if p.lower() in ("locations", "location"):
                    return parts[i+1] if i+1 < len(parts) else "unknown"
            return (value.get("location") or "unknown").lower()

        region_stats = defaultdict(lambda: {"added": 0, "removed": 0, "changed": 0})
        for it in added:
            region_stats[get_region(it["path"], it.get("value", {}))]["added"] += 1
        for it in removed:
            region_stats[get_region(it["path"], it.get("value", {}))]["removed"] += 1
        for it in changed:
            region_stats[get_region(it["path"], it.get("value", {}))]["changed"] += 1

        # ---------------- Group by resource type ----------------
        def get_type(value: dict) -> str:
            if isinstance(value, dict):
                return value.get("type") or value.get("kind") or "unknown"
            return "unknown"

        type_stats = defaultdict(lambda: {"added": 0, "removed": 0, "changed": 0})
        for it in added:
            type_stats[get_type(it.get("value", {}))]["added"] += 1
        for it in removed:
            type_stats[get_type(it.get("value", {}))]["removed"] += 1
        for it in changed:
            t = get_type(it.get("value", {}) or it.get("new_value", {}))
            type_stats[t]["changed"] += 1

        # ---------------- Cost estimation ----------------
        def estimate_cost(value: dict) -> float:
            t = get_type(value)
            return COST_PROFILE.get(t, COST_PROFILE["unknown"])

        total_spent = sum(estimate_cost(it.get("value", {})) for it in added)
        total_saved = sum(estimate_cost(it.get("value", {})) for it in removed)

        summary["estimated_spent"] = total_spent
        summary["estimated_saved"] = total_saved
        summary["net_cost_delta"] = total_spent - total_saved

        # ---------------- Compliance & Risk ----------------
        risky_types = [
            "Microsoft.Network/networkSecurityGroups",
            "Microsoft.KeyVault/vaults",
            "Microsoft.Authorization/roleAssignments"
        ]
        non_compliant_regions = {"brazilsouth", "eastasia", "koreacentral"}

        compliance_flags = []
        for it in added + changed:
            val = it.get("value", {})
            t = get_type(val)
            if t in risky_types:
                compliance_flags.append(f"High-risk change in {t} ({it['path']})")
            region = get_region(it.get("path", ""), val)
            if region in non_compliant_regions:
                compliance_flags.append(f"Resource deployed in non-compliant region: {region}")

        # ---------------- Insights ----------------
        insights = []
        if summary["total_added"] > summary["total_removed"]:
            insights.append(
                f"Net increase of {summary['net_delta']} resources, "
                f"with {summary['total_added']} added vs {summary['total_removed']} removed."
            )
        if summary["total_removed"] > 1000:
            insights.append("Large number of removals detected — possible cleanup or decommissioning effort.")
        if "unknown" in region_stats:
            insights.append("Some resources had no region metadata; may need cleanup or metadata enrichment.")
        if any(v["changed"] > 50 for v in type_stats.values()):
            insights.append("High number of changes within a single resource type — possible policy update or scaling event.")

        # ---------------- Anomalies ----------------
        anomalies = []
        for r, stats in region_stats.items():
            if stats["added"] > (summary["total_added"] * 0.5):
                anomalies.append(f"Region {r} had over 50% of all additions.")
            if stats["removed"] > (summary["total_removed"] * 0.5):
                anomalies.append(f"Region {r} had over 50% of all removals.")
        for t, stats in type_stats.items():
            if stats["added"] > 1000:
                anomalies.append(f"Unusually high additions of {t} resources ({stats['added']}).")

        # ---------------- AI Narrative ----------------
        narrative = None
        try:
            raw_narrative = generate_narrative(summary, region_stats, type_stats, insights, anomalies, compliance_flags)
        
            # Normalize AI output before sending to UI
            cleaned_narrative = format_narrative(raw_narrative)
            narrative_html = markdown.markdown(str(cleaned_narrative), extensions=["extra", "sane_lists"])
        except Exception as e:
            narrative_html = f"<p>(Narrative generation failed: {e})</p>"

        # ---------------- Return payload ----------------
        result = {
            "summary": summary,
            "regions": region_stats,
            "resource_types": type_stats,
            "insights": insights,
            "anomalies": anomalies,
            "compliance_flags": compliance_flags,
            "narrative": narrative_html,
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set. Please configure it in environment variables.")

client = OpenAI(api_key=api_key)

def format_narrative(raw_text: str) -> str:
    """
    Normalize GPT output before Markdown rendering.
    Cleans headers, bold markers, and numbered lists for consistent exec-style HTML.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # Remove markdown headers like ###, ####
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

    # Convert bold markers **text** → <strong>text</strong>
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)

    # Handle numbered lists (1., 2., etc.)
    if re.search(r"^\d+\.", text, re.MULTILINE):
        items = re.split(r"\n(?=\d+\.)", text)
        items = [re.sub(r"^\d+\.\s*", "", i).strip() for i in items if i.strip()]
        lis = "".join(
            f"<li style='margin-bottom:10px; line-height:1.6;'>{i}</li>"
            for i in items
        )
        return Markup(f"<ol style='padding-left:20px;'>{lis}</ol>")

    # Fallback: treat as paragraphs
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    return Markup("".join(
        f"<p style='margin:0 0 12px 0; line-height:1.6;'>{p}</p>"
        for p in paras
    ))

def generate_narrative(summary, regions, types, insights, anomalies, compliance_flags):
    prompt = f"""
    Analyze the following cloud drift data and write an executive summary
    for executives, compliance officers, and engineers. Cover:
    - Overall drift trend (growth vs shrink)
    - Estimated financial impact ($ spent, $ saved, net delta)
    - High-impact regions or resource types
    - Security/compliance risks
    - Operational anomalies
    - Recommended next steps

    Summary: {json.dumps(summary, indent=2)}
    Regions: {json.dumps(regions, indent=2)}
    Resource Types: {json.dumps(types, indent=2)}
    Insights: {insights}
    Anomalies: {anomalies}
    Compliance Flags: {compliance_flags}
    """

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior cloud risk and compliance analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Narrative generation failed: {e})"

UPLOAD_DIR = "/home/site/wwwroot/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def parse_bytes(raw: bytes):
    if raw[:2] == b"\x1f\x8b":
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            return json.load(gz)
    return json.loads(raw.decode("utf-8-sig"))

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f: return jsonify({"error": "file missing"}), 400
    raw = f.read()
    # store raw bytes as-is (don’t parse yet)
    token = f"{uuid.uuid4()}_{secure_filename(f.filename)}"
    path = os.path.join(UPLOAD_DIR, token)
    with open(path, "wb") as out:
        out.write(raw)
    return jsonify({"id": token})

@app.route("/compare_ids", methods=["POST"])
def compare_ids():
    try:
        payload = request.get_json(force=True)
        id1, id2 = payload.get("id1"), payload.get("id2")
        p1, p2 = os.path.join(UPLOAD_DIR, id1), os.path.join(UPLOAD_DIR, id2)
        if not (os.path.exists(p1) and os.path.exists(p2)):
            return jsonify({"error": "Upload IDs not found"}), 400

        with open(p1, "rb") as f: data1 = parse_bytes(f.read())
        with open(p2, "rb") as f: data2 = parse_bytes(f.read())

        diffs = deep_diff(data1, data2)
        return jsonify({
            "added":   [d for d in diffs if d["type"] == "added"],
            "removed": [d for d in diffs if d["type"] == "removed"],
            "changed": [d for d in diffs if d["type"] == "changed"],
        })
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Malformed JSON (line {e.lineno}, col {e.colno})"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        if username in USERS and check_password_hash(USERS[username], password):
            session["user"] = username
            return redirect(url_for("index"))
        else:
            return render_template("login.html", error="Invalid username or password")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


VISITOR_COUNT = 0
LAST_VISITORS = {}
TTL = 300  # only count the same IP once every 5 minutes

@app.route("/api/visit", methods=["POST"])
def add_visitor():
    """Increment the visitor count for unique IPs within TTL."""
    global VISITOR_COUNT
    ip = request.remote_addr
    now = time.time()

    if ip not in LAST_VISITORS or now - LAST_VISITORS[ip] > TTL:
        VISITOR_COUNT += 1
        LAST_VISITORS[ip] = now

    return jsonify({"visitors": VISITOR_COUNT})

@app.route("/api/visitors", methods=["GET"])
def get_visitors():
    """Return current visitor count (read-only)."""
    return jsonify({"visitors": VISITOR_COUNT})

@app.route("/<provider>_command_to_controls.json.gz")
def serve_mapping(provider):
    provider = provider.lower()
    valid = {"az", "aws", "gcloud", "oci", "ovhai", "aliyun", "ibmcloud"}
    if provider not in valid:
        abort(404)
    path = os.path.join(app.static_folder, f"{provider}_command_to_controls.json.gz")
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="application/json", as_attachment=False)

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_RUN_PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)

