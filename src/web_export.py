from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return text or "experiment"


def ensure_web_catalog(web_root: Path) -> Path:
    data_dir = web_root / "data"
    experiments_dir = data_dir / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = data_dir / "catalog.json"
    if not catalog_path.exists():
        catalog_path.write_text(
            json.dumps({"version": 1, "experiments": []}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return catalog_path


def load_catalog(web_root: Path) -> Dict:
    catalog_path = ensure_web_catalog(web_root)
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def save_catalog(web_root: Path, catalog: Dict) -> None:
    catalog_path = ensure_web_catalog(web_root)
    catalog["version"] = 1
    catalog["experiments"] = sorted(
        list(catalog.get("experiments", [])),
        key=lambda item: (str(item.get("category", "")), str(item.get("display_name", ""))),
    )
    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True), encoding="utf-8")


def rebuild_catalog_from_manifests(web_root: Path) -> Dict:
    web_root = web_root.resolve()
    experiments_root = web_root / "data" / "experiments"
    experiments: List[Dict] = []

    if experiments_root.exists():
        for manifest_path in sorted(experiments_root.glob("*/*/manifest.json")):
            manifest = _load_manifest(manifest_path)
            if not manifest:
                continue
            rel_parts = manifest_path.relative_to(experiments_root).parts
            if len(rel_parts) < 3:
                continue
            edits_path = manifest_path.with_name("edits.json")
            experiments.append(
                _catalog_entry(
                    exp_id=str(manifest.get("id", "")),
                    category=str(manifest.get("category", rel_parts[0])),
                    motion_model=str(manifest.get("motion_model", "")),
                    display_name=str(manifest.get("display_name", rel_parts[1])),
                    source_run=str(manifest.get("source_run", "")),
                    checkpoint=str(manifest.get("checkpoint", "")),
                    sim_count=int(manifest.get("sim_count", len(manifest.get("sims", [])))),
                    manifest_path=manifest_path,
                    edits_path=edits_path,
                    web_root=web_root,
                )
            )

    catalog = {"version": 1, "experiments": experiments}
    save_catalog(web_root, catalog)
    return catalog



def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))



def _load_manifest(manifest_path: Path) -> Dict:
    return _load_json(manifest_path)



def _load_edits(edits_path: Path) -> Dict:
    data = _load_json(edits_path)
    sims = data.get("sims", [])
    if not isinstance(sims, list):
        sims = []
    return {
        "nickname": str(data.get("nickname", "")).strip(),
        "sims": [sim for sim in sims if isinstance(sim, dict)],
    }



def _existing_catalog_entry(catalog: Dict, experiment_id: str) -> Dict:
    for entry in list(catalog.get("experiments", [])):
        if str(entry.get("id")) == experiment_id:
            return dict(entry)
    return {}



def _existing_sim_comment(existing_edits: Dict, *, sim_name: str, source_sim_index: int) -> str:
    for sim in list(existing_edits.get("sims", [])):
        if not isinstance(sim, dict):
            continue
        if str(sim.get("id", "")) == sim_name:
            return str(sim.get("comment", "")).strip()
        try:
            if int(sim.get("source_sim_index", -1)) == int(source_sim_index):
                return str(sim.get("comment", "")).strip()
        except (TypeError, ValueError):
            continue
    return ""



def _build_edits(existing_edits: Dict, sim_entries: List[Dict]) -> Dict:
    return {
        "nickname": str(existing_edits.get("nickname", "")).strip(),
        "sims": [
            {
                "id": str(sim.get("id", "")),
                "source_sim_index": int(sim.get("source_sim_index", 0)),
                "comment": _existing_sim_comment(
                    existing_edits,
                    sim_name=str(sim.get("id", "")),
                    source_sim_index=int(sim.get("source_sim_index", 0)),
                ),
            }
            for sim in sim_entries
        ],
    }



def _discover_source_sims(source_dir: Path) -> List[Dict]:
    sim_dirs = sorted(
        p
        for p in source_dir.iterdir()
        if p.is_dir() and p.name.startswith("sim_") and (p / "timeline.json").exists()
    )
    sims: List[Dict] = []
    for sim_dir in sim_dirs:
        summary_path = sim_dir / "summary.json"
        summary = {}
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sim_index = int(summary.get("sim_index", sim_dir.name.split("_", 1)[1]))
        sims.append(
            {
                "sim_index": sim_index,
                "sim_dir": sim_dir,
                "summary": summary,
            }
        )
    if sims:
        return sims

    if (source_dir / "timeline.json").exists():
        summary_path = source_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        return [{"sim_index": int(summary.get("sim_index", 0)), "sim_dir": source_dir, "summary": summary}]

    raise FileNotFoundError(f"No publishable simulations found in {source_dir}")



def _parse_selected_indices(raw: str) -> set[int] | None:
    text = raw.strip()
    if not text:
        return None
    return {int(part.strip()) for part in text.split(",") if part.strip()}



def _catalog_entry(*, exp_id: str, category: str, motion_model: str, display_name: str, source_run: str, checkpoint: str, sim_count: int, manifest_path: Path, edits_path: Path, web_root: Path) -> Dict:
    return {
        "id": exp_id,
        "category": category,
        "motion_model": motion_model,
        "display_name": display_name,
        "source_run": source_run,
        "checkpoint": checkpoint,
        "sim_count": sim_count,
        "manifest_path": str(manifest_path.relative_to(web_root)).replace("\\", "/"),
        "edits_path": str(edits_path.relative_to(web_root)).replace("\\", "/"),
    }



def publish_experiment_bundle(
    *,
    source_dir: Path,
    web_root: Path,
    category: str,
    motion_model: str,
    display_name: str,
    selected_indices: Iterable[int] | None = None,
    experiment_id: str | None = None,
    source_run: str = "",
    checkpoint: str = "",
) -> Path:
    source_dir = source_dir.resolve()
    web_root = web_root.resolve()
    selected = set(selected_indices) if selected_indices is not None else None
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = {}
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))

    manifest_run_dir = str(source_manifest.get("run_dir", "")).strip()
    inferred_source_run = Path(manifest_run_dir).name if manifest_run_dir else source_dir.name

    sims = _discover_source_sims(source_dir)
    chosen = [item for item in sims if selected is None or int(item["sim_index"]) in selected]
    if not chosen:
        raise ValueError("No matching simulations selected for publishing.")

    category_slug = slugify(category)
    display_slug = slugify(display_name)
    exp_id = experiment_id or f"{category_slug}--{display_slug}"
    dest_root = web_root / "data" / "experiments" / category_slug / display_slug
    existing_edits = _load_edits(dest_root / "edits.json")
    if dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    sim_entries: List[Dict] = []
    for index, item in enumerate(chosen):
        source_sim_dir = Path(item["sim_dir"])
        summary = dict(item.get("summary", {}))
        sim_name = f"sim_{index:03d}"
        dest_sim_dir = dest_root / sim_name
        shutil.copytree(source_sim_dir, dest_sim_dir, dirs_exist_ok=True)
        sim_entries.append(
            {
                "id": sim_name,
                "source_sim_index": int(item["sim_index"]),
                "status": (
                    "cleared"
                    if bool(summary.get("cleared", False))
                    else "timed_out"
                    if bool(summary.get("timed_out", False))
                    else "ended"
                ),
                "cleared": bool(summary.get("cleared", False)),
                "timed_out": bool(summary.get("timed_out", False)),
                "final_time": float(summary.get("final_time", 0.0)),
                "steps": int(summary.get("steps", 0)),
                "return": float(summary.get("return", 0.0)),
                "viewer_path": f"{sim_name}/index.html",
                "timeline_path": f"{sim_name}/timeline.json",
                "summary_path": f"{sim_name}/summary.json",
            }
        )

    manifest = {
        "id": exp_id,
        "category": category,
        "motion_model": motion_model,
        "display_name": display_name,
        "source_run": source_run or inferred_source_run,
        "source_run_path": manifest_run_dir or str(source_dir),
        "checkpoint": checkpoint or source_manifest.get("checkpoint_tag", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sim_count": len(sim_entries),
        "sims": sim_entries,
    }
    manifest_path = dest_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    edits = _build_edits(existing_edits, sim_entries)
    edits_path = dest_root / "edits.json"
    edits_path.write_text(json.dumps(edits, indent=2, sort_keys=True), encoding="utf-8")

    catalog = load_catalog(web_root)
    experiments = [entry for entry in catalog.get("experiments", []) if entry.get("id") != exp_id]
    experiments.append(
        _catalog_entry(
            exp_id=exp_id,
            category=category,
            motion_model=motion_model,
            display_name=display_name,
            source_run=source_run or inferred_source_run,
            checkpoint=checkpoint or source_manifest.get("checkpoint_tag", ""),
            sim_count=len(sim_entries),
            manifest_path=manifest_path,
            edits_path=edits_path,
            web_root=web_root,
        )
    )
    catalog["experiments"] = experiments
    save_catalog(web_root, catalog)
    return dest_root



def publish_single_sim_bundle(
    *,
    source_dir: Path,
    web_root: Path,
    category: str,
    motion_model: str,
    display_name: str,
    experiment_id: str | None = None,
    source_run: str = "",
    checkpoint: str = "",
) -> Path:
    source_dir = source_dir.resolve()
    web_root = web_root.resolve()
    if not (source_dir / "timeline.json").exists():
        raise FileNotFoundError(f"No timeline.json found in {source_dir}")

    summary_path = source_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    source_sim_index = int(summary.get("sim_index", 0))

    category_slug = slugify(category)
    display_slug = slugify(display_name)
    exp_id = experiment_id or f"{category_slug}--{display_slug}"
    dest_root = web_root / "data" / "experiments" / category_slug / display_slug
    dest_root.mkdir(parents=True, exist_ok=True)

    manifest_path = dest_root / "manifest.json"
    edits_path = dest_root / "edits.json"
    existing_manifest: Dict = _load_manifest(manifest_path)
    existing_edits: Dict = _load_edits(edits_path)

    sim_name = f"sim_{source_sim_index:03d}"
    dest_sim_dir = dest_root / sim_name
    if dest_sim_dir.exists():
        shutil.rmtree(dest_sim_dir)
    shutil.copytree(source_dir, dest_sim_dir)

    sim_entry = {
        "id": sim_name,
        "source_sim_index": source_sim_index,
        "status": (
            "cleared"
            if bool(summary.get("cleared", False))
            else "timed_out"
            if bool(summary.get("timed_out", False))
            else "ended"
        ),
        "cleared": bool(summary.get("cleared", False)),
        "timed_out": bool(summary.get("timed_out", False)),
        "final_time": float(summary.get("final_time", 0.0)),
        "steps": int(summary.get("steps", 0)),
        "return": float(summary.get("return", 0.0)),
        "viewer_path": f"{sim_name}/index.html",
        "timeline_path": f"{sim_name}/timeline.json",
        "summary_path": f"{sim_name}/summary.json",
    }

    existing_sims = {
        str(item.get("id")): item
        for item in list(existing_manifest.get("sims", []))
        if isinstance(item, dict) and item.get("id")
    }
    existing_sims[sim_name] = sim_entry
    merged_sims = sorted(existing_sims.values(), key=lambda item: int(item.get("source_sim_index", 0)))

    manifest = {
        "id": exp_id,
        "category": category,
        "motion_model": motion_model,
        "display_name": display_name,
        "source_run": source_run or existing_manifest.get("source_run", "") or source_dir.parent.name,
        "source_run_path": str(source_dir.parent),
        "checkpoint": checkpoint or existing_manifest.get("checkpoint", ""),
        "created_at": existing_manifest.get("created_at", datetime.now(timezone.utc).isoformat()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "sim_count": len(merged_sims),
        "sims": merged_sims,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    edits = _build_edits(existing_edits, merged_sims)
    edits_path.write_text(json.dumps(edits, indent=2, sort_keys=True), encoding="utf-8")

    catalog = load_catalog(web_root)
    experiments = [entry for entry in catalog.get("experiments", []) if entry.get("id") != exp_id]
    experiments.append(
        _catalog_entry(
            exp_id=exp_id,
            category=category,
            motion_model=motion_model,
            display_name=display_name,
            source_run=manifest["source_run"],
            checkpoint=manifest["checkpoint"],
            sim_count=len(merged_sims),
            manifest_path=manifest_path,
            edits_path=edits_path,
            web_root=web_root,
        )
    )
    catalog["experiments"] = experiments
    save_catalog(web_root, catalog)
    return dest_root


__all__ = [
    "ensure_web_catalog",
    "load_catalog",
    "publish_experiment_bundle",
    "publish_single_sim_bundle",
    "rebuild_catalog_from_manifests",
    "save_catalog",
    "slugify",
]
