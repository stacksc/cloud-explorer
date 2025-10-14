import json, gzip, os, re

src = "command_to_controls.json.gz"
out_dir = "split_controls"
os.makedirs(out_dir, exist_ok=True)

def detect_provider(command: str) -> str:
    command = command.lower()
    if command.startswith("az:"):
        return "azure"
    elif command.startswith("aws:"):
        return "aws"
    elif command.startswith(("ibmcloud", "ibmcloud:")):
        return "ibmcloud"
    elif command.startswith(("aliyun", "aliyun:")):
        return "aliyun"
    elif command.startswith(("gcloud", "gcloud:")):
        return "gcloud"
    elif command.startswith(("oci", "oci:")):
        return "oci"
    elif command.startswith(("ovhai", "ovhai:")):
        return "ovhai"

with gzip.open(src, "rt", encoding="utf-8") as f:
    data = json.load(f)

grouped = {"azure": {}, "aws": {}, "gcloud": {}, "oci": {}, "ovhai": {}, "aliyun": {}, "ibmcloud": {}}

for cmd, val in data.items():
    provider = detect_provider(cmd)
    grouped[provider][cmd] = val

for provider, subset in grouped.items():
    if not subset:
        continue
    out_path = os.path.join(out_dir, f"{provider}_command_to_controls.json.gz")
    with gzip.open(out_path, "wt", encoding="utf-8") as gz:
        json.dump(subset, gz, separators=(",", ":"))
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"✅ {provider}: {len(subset)} commands → {size_mb:.2f} MB → {out_path}")

