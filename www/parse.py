#!/usr/bin/env python3
import json
import click
from pathlib import Path

# Default path to the catalog JSON file
DEFAULT_CATALOG_JSON = "/Users/cstacks/git/Apps.SystemTools/TorchLite/nist_data/catalog.json"

def extract_parts(control, cid):
    """Recursively pull out all prose-bearing parts."""
    for part in control.get("parts", []):
        if "prose" in part:
            yield {
                "control_id": cid,
                "part_id": part.get("id"),
                "type": part.get("name"),
                "prose": part.get("prose")
            }
        if part.get("parts"):
            nested = {"parts": part.get("parts")}
            yield from extract_parts(nested, cid)


def parse_catalog(catalog_path=DEFAULT_CATALOG_JSON, out_path=None):
    try:
        with open(catalog_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        click.secho(f"❌ Failed to load catalog at '{catalog_path}': {e}", fg='red')
        return

    groups = data.get('catalog', {}).get('groups', [])
    click.secho(f"📦 Found {len(groups)} control groups", fg='green')

    grouped = {}

    for group in groups:
        fam_title = group.get('title', '<no family>')
        click.secho(f"\n📘 Group: {fam_title}", fg='blue')

        for control in group.get('controls', []):
            cid = control.get('id')
            title = control.get('title', '<no title>')

            props = control.get('props', [])
            # Deduplicate norm_ids
            norm_ids = sorted(set(p["value"] for p in props if p.get("name") == "label"))

            parts = list(extract_parts(control, cid))

            # Separate into buckets
            statements = [p for p in parts if p["type"] in ("statement", "item")]
            guidance = [p for p in parts if p["type"] == "guidance"]
            objectives = [p for p in parts if p["type"] == "assessment-objective"]
            objects = [p for p in parts if p["type"] == "assessment-objects"]

            # Text blob = just statements (for embeddings)
            text_blob = f"{cid} {title} " + " ".join(p["prose"] for p in statements)

            grouped[cid] = {
                "id": cid,
                "title": title,
                "family": fam_title,
                "props": props,
                "norm_ids": norm_ids,
                "statements": statements,
                "guidance": guidance,
                "assessment_objectives": objectives,
                "assessment_objects": objects,
                "text": text_blob.strip()
            }

    if out_path:
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(grouped, f, indent=2, ensure_ascii=False)
            click.secho(f"✅ Wrote grouped controls to '{out_path}'", fg='green')
        except Exception as e:
            click.secho(f"❌ Failed to write output to '{out_path}': {e}", fg='red')
    else:
        click.secho(f"⚠️  No output path provided; skipping write", fg='yellow')


if __name__ == "__main__":
    @click.command()
    @click.option("--catalog", default=DEFAULT_CATALOG_JSON, help="Path to catalog.json")
    @click.option("--out", default=None, help="Optional path to write grouped JSON")
    def main(catalog, out):
        parse_catalog(catalog, out)

    main()

