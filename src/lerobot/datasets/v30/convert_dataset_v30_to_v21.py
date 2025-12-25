#!/usr/bin/env python

"""
Convert a local LeRobot dataset from v3.0 (file-based) to v2.1 (episode-based).

This script is the reverse of `convert_dataset_v21_to_v30.py`:
- v3.0: data/video are stored in chunked "file-xxx.*" shards and episode boundaries are
  reconstructed using `meta/episodes/**/*.parquet`.
- v2.1: each episode is stored as its own parquet/mp4 file plus legacy JSONL metadata.

The conversion is **non-destructive**: it never modifies the input dataset directory and
always writes a new dataset directory.

Usage:
  python -m lerobot.datasets.v30.convert_dataset_v30_to_v21 \
    --input-dir /path/to/dataset_v30 \
    --output-dir /path/to/dataset_v21
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import tqdm

from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_PATH,
    INFO_PATH,
    STATS_PATH,
    unflatten_dict,
    write_json,
)

V21 = "v2.1"
V30 = "v3.0"

V21_DATA_PATH_TEMPLATE = "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
V21_VIDEO_PATH_TEMPLATE = "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_jsonable(value: Any) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore[assignment]

    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None  # type: ignore[assignment]

    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]

    if np is not None:
        if isinstance(value, np.ndarray):
            return _to_jsonable(value.tolist())
        if isinstance(value, np.generic):
            return value.item()

    if torch is not None and isinstance(value, torch.Tensor):
        return _to_jsonable(value.detach().cpu().tolist())

    return value


def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(_to_jsonable(item), ensure_ascii=False) + "\n")


def _validate_v30_dataset(input_dir: Path) -> dict[str, Any]:
    info_path = input_dir / INFO_PATH
    if not info_path.exists():
        raise FileNotFoundError(f"Missing `{INFO_PATH}` under `{input_dir}`.")

    info = _load_json(info_path)
    version = info.get("codebase_version", "unknown")
    if version != V30:
        raise ValueError(f"Expected v3.0 dataset at `{input_dir}` but found `{version}`.")

    return info


def _get_video_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    return sorted([key for key, ft in features.items() if ft.get("dtype") == "video"])


def _load_v3_episodes_df(input_dir: Path) -> pd.DataFrame:
    episodes_dir = input_dir / "meta" / "episodes"
    paths = sorted(episodes_dir.glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode parquet files found under `{episodes_dir}`.")

    dfs = [pd.read_parquet(p) for p in paths]
    episodes_df = pd.concat(dfs, ignore_index=True)
    episodes_df = episodes_df.sort_values("episode_index").reset_index(drop=True)
    return episodes_df


def _convert_info(
    info_v30: dict[str, Any],
    output_dir: Path,
    chunks_size: int,
    total_videos: int,
) -> dict[str, Any]:
    info_v21 = deepcopy(info_v30)
    info_v21["codebase_version"] = V21
    info_v21["total_chunks"] = (int(info_v21["total_episodes"]) + chunks_size - 1) // chunks_size
    info_v21["total_videos"] = total_videos
    info_v21["data_path"] = V21_DATA_PATH_TEMPLATE
    info_v21["video_path"] = V21_VIDEO_PATH_TEMPLATE if total_videos > 0 else None
    write_json(info_v21, output_dir / INFO_PATH)
    return info_v21


def _maybe_update_video_feature_info(
    info_v21: dict[str, Any],
    output_dir: Path,
    *,
    video_keys: list[str],
    first_episode_index: int,
    chunks_size: int,
) -> None:
    """Update `features[*].info` for video keys based on the actual output files."""
    try:
        from lerobot.datasets.video_utils import get_video_info
    except Exception as exc:  # pragma: no cover
        logging.warning(f"Cannot probe output videos to update `meta/info.json` ({exc}).")
        return

    for video_key in video_keys:
        chunk_index = first_episode_index // chunks_size
        sample_path = output_dir / V21_VIDEO_PATH_TEMPLATE.format(
            chunk_index=chunk_index, video_key=video_key, episode_index=first_episode_index
        )
        if not sample_path.exists():
            logging.warning(f"Missing output video file for probing: `{sample_path}`.")
            continue

        info = info_v21.get("features", {}).get(video_key, {}).get("info", None)
        if not isinstance(info, dict):
            info_v21.setdefault("features", {}).setdefault(video_key, {})["info"] = {}

        info_v21["features"][video_key]["info"] = get_video_info(sample_path)


def _copy_global_stats(input_dir: Path, output_dir: Path) -> None:
    src = input_dir / STATS_PATH
    if not src.exists():
        logging.warning(f"Missing `{STATS_PATH}` under `{input_dir}`. Skipping.")
        return
    dst = output_dir / STATS_PATH
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _convert_tasks(input_dir: Path, output_dir: Path) -> None:
    tasks_parquet = input_dir / "meta" / "tasks.parquet"
    if not tasks_parquet.exists():
        logging.warning(f"Missing `meta/tasks.parquet` under `{input_dir}`. Writing empty `tasks.jsonl`.")
        _write_jsonl(output_dir / "meta" / "tasks.jsonl", [])
        return

    df_tasks = pd.read_parquet(tasks_parquet)
    if "task_index" not in df_tasks.columns:
        raise ValueError(f"Expected a `task_index` column in `{tasks_parquet}`.")

    df_tasks = df_tasks.sort_values("task_index")
    items: list[dict[str, Any]] = []
    for task_str, row in df_tasks.iterrows():
        items.append({"task_index": int(row["task_index"]), "task": str(task_str)})

    _write_jsonl(output_dir / "meta" / "tasks.jsonl", items)


def _convert_legacy_episodes_jsonl(episodes_df: pd.DataFrame, output_dir: Path) -> None:
    required = {"episode_index", "tasks", "length"}
    missing = required - set(episodes_df.columns)
    if missing:
        raise ValueError(f"Missing columns in v3 episodes metadata: {sorted(missing)}.")

    items: list[dict[str, Any]] = []
    for row in episodes_df[["episode_index", "tasks", "length"]].itertuples(index=False):
        episode_index = int(row.episode_index)
        length = int(row.length)
        tasks = row.tasks
        if tasks is None or (isinstance(tasks, float) and pd.isna(tasks)):
            tasks_list: list[str] = []
        elif isinstance(tasks, (list, tuple)):
            tasks_list = [str(t) for t in tasks]
        else:
            tasks_list = list(tasks) if hasattr(tasks, "__iter__") and not isinstance(tasks, str) else [str(tasks)]

        items.append({"episode_index": episode_index, "tasks": tasks_list, "length": length})

    _write_jsonl(output_dir / "meta" / "episodes.jsonl", items)


def _convert_legacy_episodes_stats_jsonl(episodes_df: pd.DataFrame, output_dir: Path) -> None:
    stats_cols = [c for c in episodes_df.columns if isinstance(c, str) and c.startswith("stats/")]
    if not stats_cols:
        logging.warning("No `stats/*` columns found in v3 `meta/episodes` parquet. Writing empty stats jsonl.")
        _write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", [])
        return

    items: list[dict[str, Any]] = []
    for _, row in episodes_df[["episode_index", *stats_cols]].iterrows():
        episode_index = int(row["episode_index"])
        flat_stats = {col: row[col] for col in stats_cols}
        nested = unflatten_dict(flat_stats)  # {"stats": {...}}
        ep_stats = nested.get("stats", {})

        items.append({"episode_index": episode_index, "stats": _to_jsonable(ep_stats)})

    _write_jsonl(output_dir / "meta" / "episodes_stats.jsonl", items)


def _convert_data(input_dir: Path, output_dir: Path, episodes_df: pd.DataFrame, chunks_size: int) -> None:
    required = {"data/chunk_index", "data/file_index"}
    missing = required - set(episodes_df.columns)
    if missing:
        raise ValueError(f"Missing columns in v3 episodes metadata: {sorted(missing)}.")

    groups = episodes_df.groupby(["data/chunk_index", "data/file_index"], sort=True)
    for (chunk_index, file_index), eps in tqdm.tqdm(groups, desc="convert data shards"):
        chunk_index_int = int(chunk_index)
        file_index_int = int(file_index)

        src_path = input_dir / DEFAULT_DATA_PATH.format(chunk_index=chunk_index_int, file_index=file_index_int)
        if not src_path.exists():
            raise FileNotFoundError(f"Missing data shard: `{src_path}`.")

        df = pd.read_parquet(src_path)
        if "episode_index" not in df.columns:
            raise ValueError(f"Missing `episode_index` column in `{src_path}`.")

        for episode_index, ep_df in df.groupby("episode_index", sort=True):
            episode_index_int = int(episode_index)
            out_chunk = episode_index_int // chunks_size
            out_path = output_dir / V21_DATA_PATH_TEMPLATE.format(
                chunk_index=out_chunk, episode_index=episode_index_int
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ep_df = ep_df.sort_values("frame_index") if "frame_index" in ep_df.columns else ep_df
            ep_df.to_parquet(out_path, index=False)


def _split_videos_by_episode(
    input_dir: Path,
    output_dir: Path,
    episodes_df: pd.DataFrame,
    video_key: str,
    fps: int,
    chunks_size: int,
    video_codec: str,
    crf: int,
    preset: str,
) -> None:
    try:
        import av  # pyav
    except Exception as exc:  # pragma: no cover
        raise ImportError("Video conversion requires `pyav` but it is not available.") from exc

    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    missing = {chunk_col, file_col, "episode_index", "length"} - set(episodes_df.columns)
    if missing:
        raise ValueError(f"Missing columns in v3 episodes metadata: {sorted(missing)}.")

    groups = episodes_df.groupby([chunk_col, file_col], sort=True)
    for (chunk_index, file_index), eps in tqdm.tqdm(groups, desc=f"convert videos ({video_key})"):
        chunk_index_int = int(chunk_index)
        file_index_int = int(file_index)

        src_video = input_dir / f"videos/{video_key}/chunk-{chunk_index_int:03d}/file-{file_index_int:03d}.mp4"
        if not src_video.exists():
            raise FileNotFoundError(f"Missing video shard: `{src_video}`.")

        eps = eps.sort_values("episode_index")
        input_container = av.open(str(src_video), mode="r")
        input_stream = input_container.streams.video[0]
        width = input_stream.codec_context.width
        height = input_stream.codec_context.height

        frame_iter = input_container.decode(input_stream)

        for row in eps.itertuples(index=False):
            episode_index = int(getattr(row, "episode_index"))
            length = int(getattr(row, "length"))

            out_chunk = episode_index // chunks_size
            out_path = output_dir / V21_VIDEO_PATH_TEMPLATE.format(
                chunk_index=out_chunk, video_key=video_key, episode_index=episode_index
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)

            output_container = av.open(str(out_path), mode="w", options={"movflags": "faststart"})
            output_stream = output_container.add_stream(
                video_codec,
                rate=fps,
                options={"crf": str(crf), "preset": preset},
            )
            output_stream.width = width
            output_stream.height = height
            output_stream.pix_fmt = "yuv420p"

            for i in range(length):
                try:
                    frame = next(frame_iter)
                except StopIteration as exc:
                    raise RuntimeError(
                        f"Video shard `{src_video}` ended early while writing episode={episode_index} "
                        f"({i+1}/{length} frames)."
                    ) from exc

                for packet in output_stream.encode(frame):
                    output_container.mux(packet)

            for packet in output_stream.encode(None):
                output_container.mux(packet)
            output_container.close()

        input_container.close()


def convert_dataset(
    input_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    skip_videos: bool = False,
    video_codec: str = "libx264",
    crf: int = 23,
    preset: str = "veryfast",
) -> None:
    info = _validate_v30_dataset(input_dir)
    video_keys = _get_video_keys(info)
    fps = int(info.get("fps", 30))
    chunks_size = int(info.get("chunks_size", DEFAULT_CHUNK_SIZE))

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: `{output_dir}`. Use `--overwrite`.")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)

    episodes_df = _load_v3_episodes_df(input_dir)

    logging.info(f"Writing v2.1 dataset to `{output_dir}` (episodes={len(episodes_df)})")

    info_v21 = _convert_info(info, output_dir, chunks_size, total_videos=len(video_keys))
    _copy_global_stats(input_dir, output_dir)
    _convert_tasks(input_dir, output_dir)
    _convert_legacy_episodes_jsonl(episodes_df, output_dir)
    _convert_legacy_episodes_stats_jsonl(episodes_df, output_dir)
    _convert_data(input_dir, output_dir, episodes_df, chunks_size)

    if skip_videos or len(video_keys) == 0:
        if len(video_keys) > 0 and skip_videos:
            logging.warning("Skipping video conversion; output will not contain per-episode mp4 files.")
        return

    for video_key in video_keys:
        _split_videos_by_episode(
            input_dir,
            output_dir,
            episodes_df,
            video_key,
            fps=fps,
            chunks_size=chunks_size,
            video_codec=video_codec,
            crf=crf,
            preset=preset,
        )

    first_episode_index = int(episodes_df["episode_index"].min()) if len(episodes_df) > 0 else 0
    _maybe_update_video_feature_info(
        info_v21,
        output_dir,
        video_keys=video_keys,
        first_episode_index=first_episode_index,
        chunks_size=chunks_size,
    )
    write_json(info_v21, output_dir / INFO_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to a v3.0 dataset directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write the v2.1 dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output directory if it exists.")
    parser.add_argument("--skip-videos", action="store_true", help="Do not write per-episode videos.")
    parser.add_argument("--video-codec", type=str, default="libx264", help="PyAV encoder (e.g. libx264).")
    parser.add_argument("--crf", type=int, default=23, help="Constant rate factor (quality) for video encoder.")
    parser.add_argument("--preset", type=str, default="veryfast", help="Encoder preset (speed/quality tradeoff).")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, ...).")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    convert_dataset(
        args.input_dir,
        args.output_dir,
        overwrite=args.overwrite,
        skip_videos=args.skip_videos,
        video_codec=args.video_codec,
        crf=args.crf,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()
