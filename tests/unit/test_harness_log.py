from __future__ import annotations

import threading
from pathlib import Path

from core.harness.log import HarnessLog


def test_concurrent_inserts(tmp_path: Path) -> None:
    """多线程并发写入不丢数据且不抛异常。"""
    db_path = str(tmp_path / "test.db")
    log = HarnessLog(db_path, "run-concurrent", git_sha="abc123")
    log.create_table("items", {"value": "INTEGER"})

    n_threads = 8
    n_per_thread = 50
    errors: list[Exception] = []

    def worker(start: int) -> None:
        try:
            for i in range(n_per_thread):
                log.insert("items", {"value": start + i})
        except Exception as e:
            errors.append(e)

    threads = [
        threading.Thread(target=worker, args=(t * n_per_thread,))
        for t in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"并发写入异常: {errors}"

    rows = log.query(
        "SELECT COUNT(*) as cnt FROM items WHERE run_id = ?", ("run-concurrent",)
    )
    assert rows[0]["cnt"] == n_threads * n_per_thread
    log.close()


def test_context_manager(tmp_path: Path) -> None:
    """测试上下文管理器正常关闭。"""
    db_path = str(tmp_path / "ctx.db")
    with HarnessLog(db_path, "run-ctx", git_sha="abc") as log:
        log.create_table("t", {"x": "TEXT"})
        log.insert("t", {"x": "hello"})

    log2 = HarnessLog(db_path, "run-ctx-check", git_sha="abc")
    rows = log2.query("SELECT status FROM _runs WHERE run_id = ?", ("run-ctx",))
    assert rows[0]["status"] == "completed"
    log2.close()


def test_context_manager_on_exception(tmp_path: Path) -> None:
    """异常时上下文管理器设置 status=failed。"""
    db_path = str(tmp_path / "err.db")
    try:
        with HarnessLog(db_path, "run-err", git_sha="abc"):
            raise ValueError("boom")
    except ValueError:
        pass

    log2 = HarnessLog(db_path, "run-check", git_sha="abc")
    rows = log2.query("SELECT status FROM _runs WHERE run_id = ?", ("run-err",))
    assert rows[0]["status"] == "failed"
    log2.close()
