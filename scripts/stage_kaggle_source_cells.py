"""Render committed Kaggle source cells into one immutable run directory."""

import argparse
from pathlib import Path

SOURCE_FILES = {
    "paper": ("00_setup.py", "10_paper.py", "90_collect.py"),
    "resume": ("00_setup.py", "10_resume.py", "90_collect.py"),
    "smoke": ("00_setup.py", "10_smoke.py", "90_collect.py"),
}


def render_sources(
    *,
    source_dir: Path,
    output_dir: Path,
    mode: str,
    expected_sha: str,
    run_id: str,
    method: str,
    seed: int,
    stop_after_task_index: int | None = None,
    parent_run_id: str = "",
) -> list[Path]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing non-empty source output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    replacements = {
        "__EXPECTED_SHA__": expected_sha,
        "__RUN_ID__": run_id,
        '"__METHOD__"': repr(method),
        '"__SEED__"': repr(str(seed)),
        '"__STOP_AFTER_TASK_INDEX__"': repr(
            str(stop_after_task_index) if stop_after_task_index is not None else ""
        ),
        "__PARENT_RUN_ID__": parent_run_id,
    }
    rendered: list[Path] = []
    for filename in SOURCE_FILES[mode]:
        source = source_dir / filename
        text = source.read_text(encoding="utf-8")
        for old, new in replacements.items():
            text = text.replace(old, new)
        unresolved = sorted(token for token in replacements if token in text)
        if unresolved:
            raise RuntimeError(f"unresolved placeholders in {source}: {unresolved}")
        destination = output_dir / filename
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(destination)
        rendered.append(destination)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("kaggle/sources"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(SOURCE_FILES), required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--method",
        choices=("slao", "slao_merged_b_init", "seqlora", "ftba_mb"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stop-after-task-index", type=int)
    parser.add_argument("--parent-run-id", default="")
    args = parser.parse_args()
    paths = render_sources(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        expected_sha=args.expected_sha,
        run_id=args.run_id,
        method=args.method,
        seed=args.seed,
        stop_after_task_index=args.stop_after_task_index,
        parent_run_id=args.parent_run_id,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
