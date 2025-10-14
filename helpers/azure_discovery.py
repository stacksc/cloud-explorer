import json
import subprocess
from collections import defaultdict
from datetime import datetime

def run_az(cmd):
    """Run an az cli command and return parsed JSON output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}\n{result.stderr}")
    return json.loads(result.stdout) if result.stdout.strip() else {}

def normalize_type(res_type):
    """Simplify Azure resource type to fit under categories"""
    parts = res_type.split("/")
    if len(parts) >= 2:
        provider, rtype = parts[0], parts[1]
        if "Compute" in provider: return "compute", rtype
        if "Storage" in provider: return "storage", rtype
        if "Network" in provider: return "network", rtype
        if "Sql" in provider or "DB" in provider: return "database", rtype
        return provider.lower(), rtype
    return "other", res_type

def build_discovery():
    # 1. Get current tenant
    account = run_az("az account show --output json")
    tenant_id = account.get("tenantId", "unknown")

    # 2. Get all subscriptions
    subs = run_az("az account list --all --output json")

    discovery = {
        "provider": "azure",
        "tenant": tenant_id,
        "subscriptions": {}
    }

    for sub in subs:
        sid = sub["id"]
        sname = sub["name"]

        print(f"🔍 Discovering subscription: {sname} ({sid})")

        # Switch context
        subprocess.run(f"az account set --subscription {sid}", shell=True)

        # List all resources
        resources = run_az("az resource list --output json")

        services = defaultdict(lambda: defaultdict(list))

        for res in resources:
            category, rtype = normalize_type(res["type"])
            services[category][rtype].append({
                "name": res.get("name"),
                "id": res.get("id"),
                "location": res.get("location"),
                "tags": res.get("tags", {}),
                "kind": res.get("kind"),
                "sku": res.get("sku"),
                "properties": res.get("properties", {})
            })

        discovery["subscriptions"][sid] = {
            "name": sname,
            "services": services
        }

    return discovery

if __name__ == "__main__":
    data = build_discovery()
    ts = datetime.utcnow().strftime("%Y-%m-%d")
    filename = f"azure_discovery_{ts}.json"

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Discovery snapshot saved to {filename}")

