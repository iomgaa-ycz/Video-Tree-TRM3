"""HarnessLog：SQLite 薄包装，提供灵活的结构化日志能力。"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _get_git_sha() -> str | None:
    """获取当前 git commit SHA。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _now_iso() -> str:
    """返回当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.now(timezone.utc).isoformat()


class HarnessLog:
    """SQLite 薄包装，为科研项目提供统一的结构化日志接口。

    参数:
        db_path: SQLite 数据库文件路径。
        run_id: 本次运行的唯一标识。
        git_sha: 代码版本，默认自动获取。
        config_snapshot: 本次运行的配置快照。
    """

    def __init__(
        self,
        db_path: str,
        run_id: str,
        git_sha: str | None = None,
        config_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self._run_id = run_id
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_fixed_tables()
        resolved_sha = git_sha or _get_git_sha()
        config_json = json.dumps(config_snapshot, ensure_ascii=False) if config_snapshot else None
        self._conn.execute(
            "INSERT OR IGNORE INTO _runs (run_id, git_sha, started_at, config, status) VALUES (?, ?, ?, ?, ?)",
            (run_id, resolved_sha, _now_iso(), config_json, "running"),
        )
        self._conn.commit()

    def _init_fixed_tables(self) -> None:
        """创建 _runs 和 _events 固定表。"""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS _runs (
                run_id TEXT PRIMARY KEY,
                git_sha TEXT,
                started_at TEXT,
                finished_at TEXT,
                config JSON,
                status TEXT DEFAULT 'running',
                skills_version TEXT,
                prompts_version TEXT,
                questions_ref TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS _events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                timestamp TEXT,
                event_type TEXT,
                payload JSON
            )
        """)
        self._conn.commit()

    def create_table(
        self,
        name: str,
        columns: dict[str, str],
        primary_key: str | None = None,
    ) -> None:
        """创建自定义表，自动追加 run_id 和 timestamp 列。

        参数:
            name: 表名。
            columns: 列定义，如 {"epoch": "INTEGER", "loss": "REAL"}。
            primary_key: 主键列名。
        """
        all_columns = {"run_id": "TEXT", "timestamp": "TEXT"}
        all_columns.update(columns)
        col_defs = []
        for col_name, col_type in all_columns.items():
            pk_suffix = " PRIMARY KEY" if col_name == primary_key else ""
            col_defs.append(f"{col_name} {col_type}{pk_suffix}")
        sql = f"CREATE TABLE IF NOT EXISTS {name} ({', '.join(col_defs)})"
        self._conn.execute(sql)
        self._conn.commit()

    def insert(self, table: str, record: dict[str, Any], mode: str = "append") -> None:
        """插入一条记录，自动填充 run_id 和 timestamp。

        参数:
            table: 目标表名。
            record: 要插入的数据。
            mode: "append" 或 "upsert"。
        """
        enriched = {"run_id": self._run_id, "timestamp": _now_iso()}
        enriched.update(record)
        cols = list(enriched.keys())
        placeholders = ", ".join(["?"] * len(cols))
        col_names = ", ".join(cols)
        values = [enriched[c] for c in cols]
        if mode == "upsert":
            sql = f"INSERT OR REPLACE INTO {table} ({col_names}) VALUES ({placeholders})"
        else:
            sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"
        self._conn.execute(sql, values)
        self._conn.commit()

    def insert_many(self, table: str, records: list[dict[str, Any]], mode: str = "append") -> None:
        """批量插入多条记录。

        参数:
            table: 目标表名。
            records: 要插入的数据列表。
            mode: "append" 或 "upsert"。
        """
        for record in records:
            self.insert(table, record, mode=mode)

    def execute(self, sql: str, params: tuple = ()) -> None:
        """执行原生 SQL 写操作。

        参数:
            sql: SQL 语句。
            params: 参数元组。
        """
        self._conn.execute(sql, params)
        self._conn.commit()

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        """执行原生 SQL 查询，返回 list[dict]。

        参数:
            sql: SQL 查询语句。
            params: 参数元组。

        返回:
            查询结果列表，每行为一个字典。
        """
        cursor = self._conn.execute(sql, params)
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """向 _events 表写入一条事件。

        参数:
            event_type: 事件类型标识。
            payload: 事件数据。
        """
        self._conn.execute(
            "INSERT INTO _events (run_id, timestamp, event_type, payload) VALUES (?, ?, ?, ?)",
            (self._run_id, _now_iso(), event_type, json.dumps(payload, ensure_ascii=False)),
        )
        self._conn.commit()

    def close(self, status: str = "completed") -> None:
        """更新运行状态并关闭连接。

        参数:
            status: 最终状态，"completed" 或 "failed"。
        """
        self._conn.execute(
            "UPDATE _runs SET finished_at = ?, status = ? WHERE run_id = ?",
            (_now_iso(), status, self._run_id),
        )
        self._conn.commit()
        self._conn.close()

    def register_schema(self, wiki_dir: str) -> None:
        """将当前 db 中所有自定义表的 schema 导出注册到 Wiki。

        参数:
            wiki_dir: Wiki 根目录。
        """
        tables = self.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE '_%'"
        )
        for table_row in tables:
            table_name = table_row["name"]
            subprocess.run(
                [
                    "python3", ".claude/tools/research_wiki.py",
                    "add_entity", wiki_dir,
                    "--type", "schema",
                    "--id", table_name,
                    "--title", f"表结构: {table_name}",
                ],
                check=True,
            )

    def __enter__(self) -> HarnessLog:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        status = "failed" if exc_type is not None else "completed"
        self.close(status=status)
