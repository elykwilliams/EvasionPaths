#!/usr/bin/env python3
"""Build browser timeline pages for checkpoint rollout artifacts."""

from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path
from typing import Dict, List, Tuple

from UI import _write_timeline_html


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate HTML viewers for RL checkpoint Reeb artifacts.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Output directory from rl_reeb_checkpoints.py.",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=1,
        help="Episode index to visualize (1-based, 0 = all episodes).",
    )
    parser.add_argument("--open", action="store_true", help="Open gallery page in browser.")
    return parser


def _find_timeline_jsons(input_dir: Path, episode_index: int) -> List[Path]:
    checkpoints = sorted(path for path in input_dir.iterdir() if path.is_dir())
    timelines: List[Path] = []

    for checkpoint_dir in checkpoints:
        if episode_index > 0:
            timeline = checkpoint_dir / f"episode_{episode_index:03d}" / "timeline.json"
            if timeline.exists():
                timelines.append(timeline)
            continue

        for episode_dir in sorted(checkpoint_dir.glob("episode_*")):
            timeline = episode_dir / "timeline.json"
            if timeline.exists():
                timelines.append(timeline)

    return timelines


def _load_checkpoint_summary(checkpoint_dir: Path) -> Dict:
    summary_path = checkpoint_dir / "checkpoint_summary.json"
    if not summary_path.exists():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _build_gallery_html(rows: List[Tuple[str, str, str, Dict]]) -> str:
    lines = [
        "<!doctype html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='utf-8' />",
        "  <meta name='viewport' content='width=device-width, initial-scale=1' />",
        "  <title>RL Reeb Checkpoint Gallery</title>",
        "  <style>",
        "    body { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; margin: 20px; color: #102a43; }",
        "    table { width: 100%; border-collapse: collapse; }",
        "    th, td { border: 1px solid #d9e2ec; padding: 8px; text-align: left; }",
        "    th { background: #f0f4f8; }",
        "    a { color: #0b7285; text-decoration: none; }",
        "    a:hover { text-decoration: underline; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <h2>RL Reeb Checkpoint Gallery</h2>",
        "  <table>",
        "    <thead>",
        "      <tr>",
        "        <th>Checkpoint</th>",
        "        <th>Episode</th>",
        "        <th>Clear Rate</th>",
        "        <th>Mean Return</th>",
        "        <th>Mean Final Time</th>",
        "        <th>Viewer</th>",
        "      </tr>",
        "    </thead>",
        "    <tbody>",
    ]

    for checkpoint_tag, episode_name, rel_html, summary in rows:
        clear_rate = float(summary.get("clear_rate", 0.0))
        mean_return = float(summary.get("mean_return", 0.0))
        mean_final_time = float(summary.get("mean_final_time", 0.0))
        lines.extend(
            [
                "      <tr>",
                f"        <td>{checkpoint_tag}</td>",
                f"        <td>{episode_name}</td>",
                f"        <td>{clear_rate:.3f}</td>",
                f"        <td>{mean_return:.3f}</td>",
                f"        <td>{mean_final_time:.3f}</td>",
                f"        <td><a href='{rel_html}'>Open timeline</a></td>",
                "      </tr>",
            ]
        )

    lines.extend(
        [
            "    </tbody>",
            "  </table>",
            "</body>",
            "</html>",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = _make_parser().parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    timelines = _find_timeline_jsons(input_dir, int(args.episode_index))
    if not timelines:
        raise SystemExit("No timeline.json files found for the selected episode index.")

    gallery_rows: List[Tuple[str, str, str, Dict]] = []
    for timeline_path in timelines:
        timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
        html_path = timeline_path.parent / "index.html"
        _write_timeline_html(html_path, timeline)

        checkpoint_dir = timeline_path.parent.parent
        checkpoint_tag = checkpoint_dir.name
        episode_name = timeline_path.parent.name
        summary = _load_checkpoint_summary(checkpoint_dir)
        rel_html = str(html_path.relative_to(input_dir))
        gallery_rows.append((checkpoint_tag, episode_name, rel_html, summary))

    gallery_rows.sort(key=lambda item: (item[0], item[1]))
    gallery_html = _build_gallery_html(gallery_rows)
    gallery_path = input_dir / "index.html"
    gallery_path.write_text(gallery_html, encoding="utf-8")

    print(f"Wrote {len(timelines)} timeline viewer pages.")
    print(f"Gallery: {gallery_path}")
    if args.open:
        webbrowser.open(gallery_path.resolve().as_uri())


if __name__ == "__main__":
    main()
