"""Wiki-Site 渲染守护进程：Vite 管理 + 文件监听 + manifest 重建 + 渲染队列。

LLM 渲染由 Claude Code 的 render-wiki-page skill 负责，
本模块只负责基础设施（Vite server、文件监听、manifest 同步）和渲染队列管理。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from watchdog.events import (
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from tools.wiki_site._wiki_helpers import ENTITY_TYPES
from tools.wiki_site.manifest import build_manifest, write_manifest_json
from tools.wiki_site.wiki_site_config import WikiSiteConfig, load_config

logger = logging.getLogger("wiki-site")

RENDER_QUEUE_FILENAME = "render-queue.json"


def _read_render_queue(queue_path: Path) -> list[str]:
    """读取渲染队列。"""
    if not queue_path.exists():
        return []
    try:
        data = json.loads(queue_path.read_text(encoding="utf-8"))
        return list(data) if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_render_queue(queue_path: Path, paths: list[str]) -> None:
    """写入渲染队列。"""
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    unique = list(dict.fromkeys(paths))
    queue_path.write_text(
        json.dumps(unique, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _rebuild_manifest_and_sync(wiki_dir: Path, app_data_dir: Path) -> None:
    """重建 manifest.json 并同步 edges.json 到 app data 目录。"""
    app_data_dir.mkdir(parents=True, exist_ok=True)
    entries = build_manifest(wiki_dir)
    write_manifest_json(entries, app_data_dir / "manifest.json")

    edges_path = wiki_dir / "graph" / "edges.json"
    if edges_path.exists():
        (app_data_dir / "edges.json").write_bytes(edges_path.read_bytes())

    import shutil
    db_src = wiki_dir.parent / "results" / "harness.db"
    db_dst = app_data_dir.parent.parent / "public" / "harness.db"
    if db_src.exists():
        db_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(db_src), str(db_dst))


class _WikiEventHandler(FileSystemEventHandler):
    """监听 Wiki 目录变化，将变更文件加入渲染队列并重建 manifest。"""

    def __init__(
        self,
        wiki_dir: Path,
        app_data_dir: Path,
        queue_path: Path,
        debounce_seconds: int,
    ) -> None:
        self.wiki_dir = wiki_dir
        self.app_data_dir = app_data_dir
        self.queue_path = queue_path
        self.debounce_seconds = debounce_seconds
        self._pending: dict[str, float] = {}
        self.last_activity = time.time()

    def _schedule(self, path: str) -> None:
        self._pending[path] = time.time()
        self.last_activity = time.time()

    def process_pending(self) -> None:
        """处理超过防抖时间的待渲染文件，加入渲染队列。"""
        now = time.time()
        ready = [
            p
            for p, ts in list(self._pending.items())
            if now - ts >= self.debounce_seconds
        ]

        if not ready:
            return

        for p in ready:
            self._pending.pop(p, None)

        current_queue = _read_render_queue(self.queue_path)
        current_queue.extend(ready)
        _write_render_queue(self.queue_path, current_queue)

        _rebuild_manifest_and_sync(self.wiki_dir, self.app_data_dir)
        self.last_activity = time.time()
        logger.info("已将 %d 个文件加入渲染队列", len(ready))

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.is_directory and event.src_path.endswith(".md"):
            self._schedule(event.src_path)

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory and event.src_path.endswith(".md"):
            self._schedule(event.src_path)

    def on_deleted(self, event: FileDeletedEvent) -> None:
        if event.is_directory or not event.src_path.endswith(".md"):
            return
        deleted = Path(event.src_path)
        for subdir in ENTITY_TYPES.values():
            tsx = self.app_data_dir.parent / "pages" / subdir / f"{deleted.stem}.tsx"
            if tsx.exists():
                tsx.unlink()
                logger.info("已删除: %s", tsx)
        self._pending.pop(event.src_path, None)
        _rebuild_manifest_and_sync(self.wiki_dir, self.app_data_dir)
        self.last_activity = time.time()


def serve(
    wiki_dir: Path,
    config_path: Path | None = None,
    app_dir: Path | None = None,
) -> None:
    """启动 Vite dev server + 文件监听 + manifest 重建。

    LLM 渲染不在此进程中执行，而是通过渲染队列文件与 Claude Code skill 协调。
    """
    project_root = Path.cwd()
    if app_dir is None:
        app_dir = project_root / "tools" / "wiki-site" / "app"

    config = load_config(config_path) if config_path else WikiSiteConfig()
    app_data_dir = app_dir / "src" / "data"
    queue_path = project_root / ".wiki-site" / RENDER_QUEUE_FILENAME

    app_data_dir.mkdir(parents=True, exist_ok=True)

    _rebuild_manifest_and_sync(wiki_dir, app_data_dir)

    all_md = [
        str(p)
        for p in sorted(wiki_dir.rglob("*.md"))
        if p.name not in ("index.md", "log.md", "query_pack.md", "gap_map.md")
    ]
    if all_md:
        _write_render_queue(queue_path, all_md)
        logger.info("初始渲染队列: %d 个实体", len(all_md))

    vite_proc = subprocess.Popen(
        ["npx", "vite", "--host", "--port", str(config.port)],
        cwd=str(app_dir),
    )
    logger.info("Vite dev server 启动，端口 %d", config.port)

    pid_path = project_root / ".wiki-site" / ".pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")

    handler = _WikiEventHandler(
        wiki_dir, app_data_dir, queue_path, config.debounce_seconds
    )
    observer = Observer()
    observer.schedule(handler, str(wiki_dir), recursive=True)
    results_dir = project_root / "results"
    if results_dir.exists():
        observer.schedule(handler, str(results_dir), recursive=False)
    observer.start()

    def shutdown(*_: Any) -> None:
        observer.stop()
        observer.join(timeout=5)
        if vite_proc.poll() is None:
            vite_proc.terminate()
            try:
                vite_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                vite_proc.kill()
        if pid_path.exists():
            pid_path.unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    print(f"\n  Wiki-Site 已启动: http://localhost:{config.port}")
    print(f"  渲染队列: {queue_path}\n")

    try:
        while True:
            handler.process_pending()
            if (
                config.auto_shutdown_minutes > 0
                and time.time() - handler.last_activity
                > config.auto_shutdown_minutes * 60
            ):
                logger.info("无活动超时，自动关闭")
                break
            time.sleep(0.5)
    finally:
        if observer.is_alive():
            observer.stop()
            observer.join(timeout=5)
        if vite_proc.poll() is None:
            vite_proc.terminate()
        if pid_path.exists():
            pid_path.unlink()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser(description="Wiki-Site 渲染守护进程")
    parser.add_argument("wiki_dir", help="research-wiki/ 目录路径")
    parser.add_argument("--config", help="配置文件路径")
    parser.add_argument("--app-dir", help="React 应用目录路径")
    args = parser.parse_args()
    serve(
        wiki_dir=Path(args.wiki_dir),
        config_path=Path(args.config) if args.config else None,
        app_dir=Path(args.app_dir) if args.app_dir else None,
    )
